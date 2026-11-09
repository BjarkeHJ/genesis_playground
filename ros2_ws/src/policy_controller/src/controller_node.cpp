#include <filesystem>
#include <chrono>
#include <iostream>

#include "controller_node.hpp"
#include "obs_writer.hpp"
#include "stop_watch.hpp"

PolicyControllerNode::PolicyControllerNode() 
    : Node("onnx_node") 
    , policy_(OnnxPolicyConfig{
        .policy_path = declare_parameter<std::string>("model_path", "models/run_w_noise/policy.onnx"),
        .intra_op_threads = static_cast<int>(declare_parameter<int64_t>("num_threads_intra", 1)),
        .inter_op_threads = static_cast<int>(declare_parameter<int64_t>("num_threads_inter", 1)),
    })
{
    last_actions_.assign(policy_.act_dim(), 0.0);
    if (policy_.obs_dim() != 50) throw std::runtime_error("model obs_dim != 50"); // hardcoded 

    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(10),
        std::bind(&PolicyControllerNode::step, this)
    );
    policy_.print_shape();
    
    RCLCPP_INFO(this->get_logger(), "PolicyControllerNode Started...");
}

void PolicyControllerNode::step() {
    StopWatch sw_;
    sw_.tic();

    // write observation tensor: MUST match order and size of trained model
    ObsWriter w(policy_.obs());
    for (size_t i=0; i<policy_.obs_dim(); ++i) {
        w.put(1.0f);
    }
    
    // Ensure entire observation vector is filled
    assert(w.written() == policy_.obs_dim());

    auto act = policy_.run();

    double t = sw_.toc();

    std::cout << "Time: " << t << " [ms]" << std::endl;

    std::cout << "Action: ";
    for (float a : act) {
        std::cout << a << " ";
    }
    std::cout << "\n";


    std::ranges::copy(act, last_actions_.begin());
}

void PolicyControllerNode::drone_callback() {
    drone_state_.pos = {0.0f, 0.0f, 5.0f};
    drone_state_.vel = {0.0f, 0.0f, 0.0f};
    drone_state_.angvel = {0.0f, 0.0f, 0.0f};
    drone_state_.quat = {1.0f, 0.0f, 0.0f, 0.0f};
    drone_state_.stamp = this->get_clock()->now();
}

void PolicyControllerNode::payload_callback() {
    payload_state_.pos = {0.0f, 0.0f, 2.0f};
    payload_state_.vel = {0.0f, 0.0f, 0.0f};
    payload_state_.angvel = {0.0f, 0.0f, 0.0f};
    payload_state_.quat = {1.0f, 0.0f, 0.0f, 0.0f};
    payload_state_.stamp = this->get_clock()->now();
}

void PolicyControllerNode::trajectory_callback() {

}



// Main exec
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PolicyControllerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

