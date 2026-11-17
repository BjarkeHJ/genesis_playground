#ifndef ACTION_MAPPING_HPP_
#define ACTION_MAPPING_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <span>
#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "frame_utils.hpp"

namespace act {

inline constexpr size_t ACT_DIM = 4;
using Action = std::array<float, ACT_DIM>;

inline constexpr float deg2rad(float d) { return d * std::numbers::pi_v<float> / 180.0f; }

struct ActionConfig {
    // Mirrors FlightControllerConfig + AttitudeControlParams
    float clip = 1.0f;
    float max_throttle = 1.0f;
    float max_tilt = deg2rad(30.0f);
    float max_yaw_rate = deg2rad(45.0f);
    bool tilt_comp = true;
    float tilt_comp_max = deg2rad(60.0f);
    float yaw_err_max = deg2rad(45.0f);
    float hover_sim = 0.4f;
    float dt = 0.02f;
};

struct AttitudeTarget_t {
    Eigen::Quaternionf q_enu_flu{Eigen::Quaternionf::Identity()};
    float yaw_rate_enu{0.0f};
    float thrust{0.0f};
};

inline Action clip(std::span<const float> raw, float c) {
    if (raw.size() != ACT_DIM) throw std::runtime_error("act::clip expected 4 actions");
    Action a{};
    for (size_t i = 0; i < ACT_DIM; ++i) {
        a[i] = std::clamp(raw[i], -c, c);
    }
    return a;
}

inline Eigen::Quaternionf quat_from_euler(float roll, float pitch, float yaw) {
    return Eigen::Quaternionf(
        Eigen::AngleAxis(yaw, Eigen::Vector3f::UnitZ())
        * Eigen::AngleAxis(pitch, Eigen::Vector3f::UnitY())
        * Eigen::AngleAxis(roll, Eigen::Vector3f::UnitX())
    );
}

class ActionMapper {
public:
    explicit ActionMapper(ActionConfig cfg = {}, float hover_px4 = 0.5f) 
        : cfg_(cfg), hover_px4_(hover_px4) {}
    
    void reset(float yaw_now) { yaw_sp_ = yaw_now; }

    void set_hover_thrust(float hover_px4) { hover_px4_ = hover_px4; }

    AttitudeTarget_t map(const Action& a, float yaw_now) {
        const float h = cfg_.hover_sim;
        float thrust_sim = (a[0] >= 0.0)
            ? h + a[0] * (cfg_.max_throttle - h)
            : h * (1.0f + a[0]);
        float thrust = (thrust_sim / h) * hover_px4_;

        const float roll = a[1] * cfg_.max_tilt;
        const float pitch = a[2] * cfg_.max_tilt;
        const float yaw_rate = a[3] * cfg_.max_yaw_rate;

        if (cfg_.tilt_comp) {
            const float cos_tilt = std::max(std::cos(roll) * std::cos(pitch), std::cos(cfg_.tilt_comp_max));
            thrust /= cos_tilt;
        }

        const float yaw_err = std::clamp(frame::wrap_pi(yaw_sp_ + yaw_rate * cfg_.dt - yaw_now), 
                                        -cfg_.yaw_err_max, cfg_.yaw_err_max);
        yaw_sp_ = frame::wrap_pi(yaw_now + yaw_err);

        return {
            .q_enu_flu = quat_from_euler(roll, pitch, yaw_sp_),
            .yaw_rate_enu = yaw_rate,
            .thrust = thrust
        };
    }

    float yaw_sp() const { return yaw_sp_; }
    const ActionConfig& config() const { return cfg_; }

private:
    ActionConfig cfg_;
    float hover_px4_;
    float yaw_sp_{0.0f};

};

} // namespace act

#endif