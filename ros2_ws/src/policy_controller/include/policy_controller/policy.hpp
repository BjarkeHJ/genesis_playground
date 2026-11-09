#ifndef POLICY_HPP_
#define POLICY_HPP_

#include <onnxruntime_cxx_api.h>
#include <filesystem>
#include <string>
#include <span>
#include <vector>
#include <numeric>
#include <functional>
#include <iostream>

struct OnnxPolicyConfig {
    std::filesystem::path policy_path;
    int intra_op_threads = 1;
    int inter_op_threads = 1;
    ExecutionMode exec_mode = ORT_SEQUENTIAL;
    OrtLoggingLevel log_level = ORT_LOGGING_LEVEL_WARNING;
    GraphOptimizationLevel graph_opt_level = GraphOptimizationLevel::ORT_ENABLE_EXTENDED;
};

class OnnxPolicy {
public:
    explicit OnnxPolicy(OnnxPolicyConfig cfg);
    ~OnnxPolicy() = default;

    std::span<float> obs() { return obs_buf_; } // exposes the buffer span and caller fill in place
    std::span<const float> run(); // infer -> act_buf_ (pre-req: obs() fill)
    
    std::span<const float> run(std::span<const float> o) { // convenience
        if (o.size() != obs_buf_.size()) throw std::invalid_argument("obs dim");
        std::ranges::copy(o, obs_buf_.begin());
        return run();
    }

    size_t obs_dim() const { return obs_buf_.size(); }
    size_t act_dim() const { return act_buf_.size(); }
    const OnnxPolicyConfig get_config() const { return cfg_; }

    // Temporary stuff
    void print_shape();

private:
    OnnxPolicyConfig cfg_;
    Ort::Env env_;
    Ort::Session session_;
    Ort::MemoryInfo mem_info_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};

    // Buffers
    std::vector<float> obs_buf_;
    std::vector<float> act_buf_;
    std::vector<int64_t> obs_shape_;
    std::vector<int64_t> act_shape_;
    Ort::Value obs_tensor_{nullptr};
    Ort::Value act_tensor_{nullptr};
    Ort::IoBinding binding_{nullptr};
    Ort::RunOptions run_opts_;

    // Helpers
    void set_dims_();
};

#endif // POLICY_HPP_