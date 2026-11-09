#include "policy.hpp"

// Anonymous ns
namespace {
const OnnxPolicyConfig& validate(const OnnxPolicyConfig& c) {
    if (!std::filesystem::is_regular_file(c.policy_path)) {
        throw std::invalid_argument("policy model not found: " + c.policy_path.string());
    }
    if (c.intra_op_threads < 1) {
        throw std::invalid_argument("intra_op_threads must be >= 1");
    }
    return c;
}

Ort::SessionOptions make_options(const OnnxPolicyConfig& c) {
    Ort::SessionOptions o;
    o.SetIntraOpNumThreads(c.intra_op_threads);
    o.SetInterOpNumThreads(c.inter_op_threads);
    o.SetExecutionMode(c.exec_mode);
    o.SetGraphOptimizationLevel(c.graph_opt_level);
    return o;
}

size_t resolve_numel(std::vector<int64_t> shape) {
    for (auto& d : shape) if (d < 0) d = 1;
    return std::accumulate(
        shape.begin(), shape.end(), size_t{1},
        [](size_t acc, int64_t d) { return acc * static_cast<size_t>(d); }
    );
}
} // Anonymous ns 

// Constructor
OnnxPolicy::OnnxPolicy(OnnxPolicyConfig cfg) 
    : cfg_(validate(cfg))
    , env_(cfg_.log_level, "onnx_policy")
    , session_(env_, cfg_.policy_path.c_str(), make_options(cfg_)) 
{
    set_dims_(); // write shapes, buffers, and bind tensors
}

std::span<const float> OnnxPolicy::run() {
    session_.Run(run_opts_, binding_);
    return act_buf_;
}

void OnnxPolicy::set_dims_() {
    obs_shape_ = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    act_shape_ = session_.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    obs_buf_.assign(resolve_numel(obs_shape_), 0.0f);
    act_buf_.assign(resolve_numel(act_shape_), 0.0f);
    obs_tensor_ = obs_tensor_.CreateTensor<float>(
        mem_info_, obs_buf_.data(), obs_buf_.size(), obs_shape_.data(), obs_shape_.size());
    act_tensor_ = act_tensor_.CreateTensor<float>(
        mem_info_, act_buf_.data(), act_buf_.size(), act_shape_.data(), act_shape_.size());
    
    Ort::AllocatorWithDefaultOptions alloc;
    std::string input_name = session_.GetInputNameAllocated(0, alloc).get();
    std::string output_name = session_.GetOutputNameAllocated(0, alloc).get();
    binding_ = Ort::IoBinding(session_);
    binding_.BindInput(input_name.c_str(), obs_tensor_);
    binding_.BindOutput(output_name.c_str(), act_tensor_);
}

// MISC
void OnnxPolicy::print_shape() { 
    std::cout << "Input Shape [";
    for (size_t i=0; i<obs_shape_.size(); ++i) {
        std::cout << obs_shape_[i];
        if (i+1 < obs_shape_.size()) {
            std::cout << ", ";
        }
    }
    std::cout << "]";
    std::cout << " Total size: " << obs_buf_.size() << "\n";

    std::cout << "Output Shape [";
    for (size_t i=0; i<act_shape_.size(); ++i) {
        std::cout << act_shape_[i];
        if (i+1 < act_shape_.size()) {
            std::cout << ", ";
        }
    }
    std::cout << "]";
    std::cout << " Total size: " << act_buf_.size() << "\n";

}
