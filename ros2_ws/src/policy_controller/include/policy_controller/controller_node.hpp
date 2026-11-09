#ifndef CONTROLLER_NODE_HPP_
#define CONTROLLER_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "policy.hpp"
#include "poly_trajectory.hpp"

struct DroneState_t {
    Eigen::Vector3f pos;
    Eigen::Vector3f vel;
    Eigen::Vector3f angvel;
    Eigen::Quaternionf quat;
    rclcpp::Time stamp;
};
struct PayloadState_t : DroneState_t {}; // naming for clarity (inherits all from t_DroneState_)

struct ActiveTrajectory_t {
    std::shared_ptr<const PolyTrajectory> traj;
    rclcpp::Time t_zero;
};

class PolicyControllerNode : public rclcpp::Node {
public:
    PolicyControllerNode();
    ~PolicyControllerNode() = default;

private: 
    void drone_callback(); // TODO
    void payload_callback(); // TODO
    void trajectory_callback(); // TODO
    void step();
    
    OnnxPolicy policy_;

    // buffers
    std::vector<float> last_actions_;
    DroneState_t drone_state_; // recent drone state
    PayloadState_t payload_state_; // recent payload state
    ActiveTrajectory_t active_trajectory_; // active trajectory

    // ROS Stuff
    rclcpp::TimerBase::SharedPtr timer_;
};

#endif