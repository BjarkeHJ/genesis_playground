#include <filesystem>
#include <chrono>
#include <iostream>

#include "controller_node.hpp"
#include "stop_watch.hpp"
#include "frame_utils.hpp"

PolicyControllerNode::PolicyControllerNode() 
    : Node("onnx_node") 
    , policy_(OnnxPolicyConfig{
        .policy_path = declare_parameter<std::string>("model_path", "models/run_w_noise/policy.onnx"),
        .intra_op_threads = static_cast<int>(declare_parameter<int64_t>("num_threads_intra", 1)),
        .inter_op_threads = static_cast<int>(declare_parameter<int64_t>("num_threads_inter", 1)),
    })
{
    if (policy_.act_dim() != act::ACT_DIM) throw std::runtime_error("model act_dim != 4"); // hardcoded 

    auto px4_qos = rclcpp::SensorDataQoS();

    timer_ = rclcpp::create_timer(
        this,
        get_clock(),
        std::chrono::milliseconds(20),
        std::bind(&PolicyControllerNode::step, this)
    );
    traj_sub_ = this->create_subscription<PolyTrajectoryMsg>(
        "planner/trajectory",
        rclcpp::QoS(1).reliable().transient_local(),
        std::bind(&PolicyControllerNode::trajectory_callback, this, std::placeholders::_1)
    );
    p_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "px4_0/payload_odom",
        rclcpp::QoS(1),
        std::bind(&PolicyControllerNode::payload_callback, this, std::placeholders::_1)
    );
    d_odom_sub_ = this->create_subscription<VehicleOdometry>(
        "px4_0/fmu/out/vehicle_odometry",
        px4_qos,
        std::bind(&PolicyControllerNode::drone_callback, this, std::placeholders::_1)
    );
    status_sub_ = this->create_subscription<VehicleStatus>(
        "px4_0/fmu/out/vehicle_status_v1",
        px4_qos,
        std::bind(&PolicyControllerNode::status_callback, this, std::placeholders::_1)
    );

    offboard_ctrl_pub_ = this->create_publisher<OffboardControlMode>(
        "px4_0/fmu/in/offboard_control_mode",
        rclcpp::QoS(10)
    );
    att_sp_pub_ = this->create_publisher<VehicleAttitudeSetpoint>(
        "px4_0/fmu/in/vehicle_attitude_setpoint_v1",
        px4_qos
    );

    mapper_.set_hover_thrust(static_cast<float>(this->declare_parameter("hover_thrust", 0.6)));

    policy_.print_shape();    
    RCLCPP_INFO(this->get_logger(), "PolicyControllerNode Started...");
}

void PolicyControllerNode::step() {
    const rclcpp::Time now = this->get_clock()->now();

    // ready? + fault check
    const auto fresh = [&](const rclcpp::Time& s, double max_age) {
        return s.nanoseconds() > 0 && (now - s).seconds() < max_age;
    };
    if (!active_trajectory_.traj) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "no trajectory - not streaming");
        return;
    }
    if (!fresh(drone_state_.stamp, 0.1) || !fresh(payload_state_.stamp, 0.1)) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "stale drone/payload odometry - not streaming");
        return;
    }
    if (fault_) {
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "fault latched - not streaming (send new trajectory to clear)");
        return;
    }

    // START TIMING
    StopWatch sw_;
    sw_.tic();

    if (!in_offboard_) engaged_ = false;
    else if (!engaged_) on_engage(now);

    const double t = engaged_
        ? std::max(0.0, (now - std::max(active_trajectory_.t_zero, t_engage_)).seconds())
        : 0.0;
    
    const auto to_body = [](const DroneState_t& s) { return obs::BodyState(s.pos, s.vel, s.quat, s.angvel); }; // should just be the DroneState_t???
    
    const size_t n = obs::write_observation(
        policy_.obs(), obs_cfg_, 
        to_body(drone_state_), to_body(payload_state_), 
        *active_trajectory_.traj, t, 
        last_actions_
    );
    if (n != obs::ObsConfig::dim()) {
        RCLCPP_ERROR(this->get_logger(), "obs size %zu != %zu", n, obs::ObsConfig::dim());
        fault_ = true;
        return;
    }

    publish_offboard();
    if (!engaged_) return;

    act::Action a = act::clip(policy_.run(), act::ActionConfig{}.clip);
    // a[0] = 0.3;
    // a[1] = 0.0;
    // a[2] = 0.0;
    // a[3] = 0.5;
    last_actions_ = a;
    
    const act::AttitudeTarget_t target = mapper_.map(a, frame::yaw_of(drone_state_.quat));
    if (!publish_setpoint(target)) {
        fault_ = true;
        return;
    }

    RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "t=%.2f  a=[%.2f %.2f %.2f %.2f]  thrust=%.3f",
        t, a[0], a[1], a[2], a[3], target.thrust
    );

    double t_timer = sw_.toc(); // TIMER
    // std::cout << "Time: " << t_timer << " [ms]" << std::endl;
}

void PolicyControllerNode::on_engage(const rclcpp::Time& now) {
    mapper_.reset(frame::yaw_of(drone_state_.quat));
    last_actions_.fill(0.0f);
    t_engage_ = now;
    engaged_ = true;
    RCLCPP_INFO(this->get_logger(), "Engaged: policy in control");
}

void PolicyControllerNode::status_callback(const VehicleStatus::SharedPtr msg) {
    in_offboard_ = msg->nav_state == VehicleStatus::NAVIGATION_STATE_OFFBOARD;
}

void PolicyControllerNode::drone_callback(const VehicleOdometry::SharedPtr msg) {
    using namespace frame;
    if (msg->pose_frame != VehicleOdometry::POSE_FRAME_NED ||
        msg->velocity_frame != VehicleOdometry::VELOCITY_FRAME_NED) {
        RCLCPP_WARN(this->get_logger(), "VehicleOdometry is not in NED frame");
    }

    drone_state_.pos = ned_to_enu(vec3(msg->position));
    drone_state_.vel = ned_to_enu(vec3(msg->velocity));
    drone_state_.quat = ned_frd_to_enu_flu(quat_wxyz(msg->q));
    drone_state_.angvel = frd_to_flu(vec3(msg->angular_velocity)); // body rates
    drone_state_.stamp = this->get_clock()->now(); // relative time for stalesness

}

void PolicyControllerNode::payload_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    // CALLBACKS SHOULD HANDLE FRAME CONVENTIONS FRD/FLU
    geometry_msgs::msg::Point& p = msg->pose.pose.position;
    geometry_msgs::msg::Quaternion& q = msg->pose.pose.orientation;
    geometry_msgs::msg::Vector3& v = msg->twist.twist.linear;
    geometry_msgs::msg::Vector3& w = msg->twist.twist.angular;
    payload_state_.pos = Eigen::Vector3d(p.x, p.y, p.z).cast<float>();
    payload_state_.vel = Eigen::Vector3d(v.x, v.y, v.z).cast<float>();
    payload_state_.angvel = Eigen::Vector3d(w.x, w.y, w.z).cast<float>();
    payload_state_.quat = Eigen::Quaterniond(q.w, q.x, q.y, q.z).cast<float>();
    payload_state_.stamp = this->get_clock()->now();
}

void PolicyControllerNode::trajectory_callback(const PolyTrajectoryMsg::SharedPtr msg) {
    try {
        active_trajectory_.from_msg(*msg);
        active_trajectory_.t_zero = this->get_clock()->now(); //...
        fault_ = false;
        RCLCPP_INFO(this->get_logger(), "New trajectory id=%u", active_trajectory_.traj->id());
    }
    catch (const std::invalid_argument& e) {
        RCLCPP_ERROR(this->get_logger(), "Rejected trajectory: %s", e.what());
    }
}

void PolicyControllerNode::publish_offboard() {
    OffboardControlMode msg;
    msg.position = false;
    msg.velocity = false;
    msg.acceleration = false;
    msg.attitude = true;
    msg.body_rate = false;
    msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
    offboard_ctrl_pub_->publish(msg);
}

bool PolicyControllerNode::publish_setpoint(const act::AttitudeTarget_t& target) {
    using namespace frame;

    const bool finite = target.q_enu_flu.coeffs().allFinite() 
                        && std::isfinite(target.yaw_rate_enu) 
                        && std::isfinite(target.thrust);

    if (!finite) {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
            "Non-finite attitude target - not publishing"
        );
        return false;
    }
    
    VehicleAttitudeSetpoint sp{};
    sp.timestamp = this->get_clock()->now().nanoseconds() / 1000;
    sp.q_d = arr_wxyz(enu_flu_to_ned_frd(target.q_enu_flu.normalized()));
    sp.yaw_sp_move_rate = yaw_rate_enu_to_ned(target.yaw_rate_enu);
    sp.thrust_body = {0.0f, 0.0f, -std::clamp(target.thrust, 0.0f, 1.0f)}; // FRD: thrust along -z

    att_sp_pub_->publish(sp);
    return true;
}   

act::AttitudeTarget_t PolicyControllerNode::hold_target() const {
    const float yaw = frame::yaw_of(drone_state_.quat);
    return {
        .q_enu_flu = Eigen::Quaternionf(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ())),
        .yaw_rate_enu = 0.0f,
        .thrust = 0.5f // hover thrust estimate???
    };
}



// Main exec
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PolicyControllerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

