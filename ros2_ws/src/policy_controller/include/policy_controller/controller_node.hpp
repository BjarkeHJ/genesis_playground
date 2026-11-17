#ifndef CONTROLLER_NODE_HPP_
#define CONTROLLER_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <px4_msgs/msg/vehicle_status.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_msgs/msg/vehicle_attitude_setpoint.hpp>
#include <px4_msgs/msg/offboard_control_mode.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "policy.hpp"
#include "poly_trajectory.hpp"
#include "action_mapping.hpp"
#include "observation.hpp"

#include <policy_interfaces/msg/poly_trajectory_msg.hpp>
using PolyTrajectoryMsg = policy_interfaces::msg::PolyTrajectoryMsg;

struct DroneState_t {
    Eigen::Vector3f pos;
    Eigen::Vector3f vel;
    Eigen::Vector3f angvel;
    Eigen::Quaternionf quat;
    rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
};
struct PayloadState_t : DroneState_t {}; // naming for clarity (inherits all from t_DroneState_)

struct ActiveTrajectory_t {
    std::shared_ptr<const PolyTrajectory> traj;
    rclcpp::Time t_zero{0, 0, RCL_ROS_TIME};

    void from_msg(const PolyTrajectoryMsg& msg) {
        const size_t S = msg.durations.size();
        if (msg.n_coeff != N_COEFF || msg.coeffs.size() != S * N_COEFF * 3) {
            throw std::invalid_argument("coeffs size does not match durations * n_coeff * 3");
        }

        std::vector<SegCoeffs> coeffs(S);
        for (size_t s = 0; s < S; ++s) {
            for (int n = 0; n < N_COEFF; ++n) {
                for (int d = 0; d < 3; ++d) {
                    coeffs[s](n, d) = msg.coeffs[(s * N_COEFF + n) * 3 + d];
                }
            }
        }
        
        traj = std::make_shared<const PolyTrajectory>(msg.id, std::move(coeffs), msg.durations);
        t_zero = rclcpp::Time(msg.header.stamp);
    }
};


class PolicyControllerNode : public rclcpp::Node {
public:
    PolicyControllerNode();
    ~PolicyControllerNode() = default;

private: 
    using VehicleStatus = px4_msgs::msg::VehicleStatus;
    using VehicleOdometry = px4_msgs::msg::VehicleOdometry;
    using OffboardControlMode = px4_msgs::msg::OffboardControlMode;
    using VehicleAttitudeSetpoint = px4_msgs::msg::VehicleAttitudeSetpoint;

    void status_callback(const VehicleStatus::SharedPtr msg);
    void drone_callback(const VehicleOdometry::SharedPtr msg);
    void payload_callback(const nav_msgs::msg::Odometry::SharedPtr msg); // To be custom odometry publisher based on mocap (om future imu/rtk)
    void trajectory_callback(const PolyTrajectoryMsg::SharedPtr msg);
    void step();
    void publish_offboard();
    bool publish_setpoint(const act::AttitudeTarget_t& target);
    void on_engage(const rclcpp::Time& now);
    act::AttitudeTarget_t hold_target() const;
    
    bool in_offboard_{false};

    // ONNX Runtime
    OnnxPolicy policy_;

    // action mapping
    act::ActionMapper mapper_;
    bool engaged_{false};
    rclcpp::Time t_engage_{0, 0, RCL_ROS_TIME};
    bool fault_{false};

    // observation
    obs::ObsConfig obs_cfg_;
    

    // buffers
    act::Action last_actions_{0, 0, 0, 0};
    DroneState_t drone_state_; // recent drone state
    PayloadState_t payload_state_; // recent payload state
    ActiveTrajectory_t active_trajectory_; // active trajectory

    // ROS Stuff
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Subscription<PolyTrajectoryMsg>::SharedPtr traj_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr p_odom_sub_;
    rclcpp::Subscription<VehicleOdometry>::SharedPtr d_odom_sub_;
    rclcpp::Subscription<VehicleStatus>::SharedPtr status_sub_;
    rclcpp::Publisher<OffboardControlMode>::SharedPtr offboard_ctrl_pub_;
    rclcpp::Publisher<VehicleAttitudeSetpoint>::SharedPtr att_sp_pub_;
};

#endif