#ifndef OBSERVATION_HPP_
#define OBSERVATION_HPP

// Observation vector, mirroring the training env:
//   slung_payload_control/spc_env.py  SlungPayloadControlEnv._compute_buffers + _observations
// All inputs in the training convention: world ENU, body FLU, quaternions body->world.
//
// Layout (K = 5 -> dim 50):
//   [ 0, 3K)   payload position error  traj(t+tau_k) - p_pl   heading frame, tau-scaled, clamped   [k0x,k0y,k0z,k1x,...]
//   [3K, 6K)   payload velocity error  traj'(t+tau_k) - v_pl  heading frame, tau-scaled, clamped
//   +3         payload projected gravity  (body)
//   +3         payload body rates          * scale_pl_br, clamped
//   +2         swing angles  (heading)     * scale_pl_sa, clamped
//   +2         swing rates   (heading)     * scale_pl_sr, clamped
//   +3         drone projected gravity     (body)
//   +3         drone body rates            * scale_d_br, clamped
//   +4         last (clipped) actions

#include <array>
#include <cmath>
#include <numbers>
#include <span>
#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "frame_utils.hpp"
#include "obs_writer.hpp"
#include "poly_trajectory.hpp"

namespace obs {

inline constexpr size_t K = 5; // numbe of preview points

struct ObsConfig {                       // mirrors EnvConfig.obs + preview + traj limits
    std::array<float, K> preview_t{0.0f, 0.3f, 0.8f, 1.5f, 2.0f};
    float scale_pl_p = 1.0f / 3.0f;
    float scale_pl_v = 1.0f / 6.0f;
    float scale_pl_sa = 1.0f / (60.0f * std::numbers::pi_v<float> / 180.0f);
    float scale_pl_sr = 1.0f / std::numbers::pi_v<float>;
    float scale_pl_br = 1.0f / std::numbers::pi_v<float>;
    float scale_d_br = 1.0f / std::numbers::pi_v<float>;
    float vmax = 3.0f;
    float amax = 6.f;
    static constexpr size_t dim() { return 20 + 6 * 5; }
};

struct BodyState {
    Eigen::Vector3f pos;
    Eigen::Vector3f vel;
    Eigen::Quaternionf quat;
    Eigen::Vector3f rates;
};

inline size_t write_observation(
    std::span<float> buf, 
    const ObsConfig& c, 
    const BodyState& d, 
    const BodyState& p, 
    const PolyTrajectory& traj, 
    double t, 
    std::span<const float> last_actions)
{
    if (buf.size() != ObsConfig::dim()) throw std::runtime_error("observation buffer size != ObsConfig::dim()");
    ObsWriter w(buf);
    const Eigen::Vector3f down(0.0f, 0.0f, -1.0f);
    
    // heading frame
    const Eigen::Matrix3f R_h = Eigen::AngleAxisf(-frame::yaw_of(d.quat), Eigen::Vector3f::UnitZ()).toRotationMatrix();

    // task
    std::array<TrajRef, K> ref;
    for (size_t k = 0; k < K; ++k) {
        ref[k] = traj.eval(t + c.preview_t[k]);
    }

    for (size_t k = 0; k < K; ++k) {
        const float tau = c.preview_t[k];
        const float pos_scale = 1.0f / (1.0f / c.scale_pl_p + c.vmax * tau + 0.5f * c.amax * tau * tau);
        w.put_clamped(Eigen::Vector3f(R_h * (ref[k].pos.cast<float>() - p.pos)), pos_scale);
    }

    for (size_t k = 0; k < K; ++k) {
        const float tau = c.preview_t[k];
        const float vel_scale = 1.0f / (1.0f / c.scale_pl_v + c.amax * tau);
        w.put_clamped(Eigen::Vector3f(R_h * (ref[k].vel.cast<float>() - p.vel)), vel_scale);
    }

    // payload
    w.put(Eigen::Vector3f(p.quat.conjugate() * down)); // proj gravity (body)
    w.put_clamped(p.rates, c.scale_pl_br);

    const Eigen::Vector3f r = R_h * (p.pos - d.pos);
    const Eigen::Vector3f rd = R_h * (p.vel - d.vel);
    const float den0 = std::max(r.x() * r.x() + r.z() * r.z(), 1e-9f);
    const float den1 = std::max(r.y() * r.y() + r.z() * r.z(), 1e-9f);
    const Eigen::Vector2f swing_ang(std::atan2(r.x(), -r.z()), std::atan2(r.y(), -r.z()));
    const Eigen::Vector2f swing_rate(
        (r.x() * rd.z() - r.z() * rd.x()) / den0,
        (r.y() * rd.z() - r.z() * rd.y()) / den1
    );
    w.put_clamped(swing_ang, c.scale_pl_sa);
    w.put_clamped(swing_rate, c.scale_pl_sr);

    // drone
    w.put(Eigen::Vector3f(d.quat.conjugate() * down)); // proj grav (body)
    w.put_clamped(d.rates, c.scale_d_br);

    // last act
    w.put(last_actions);

    return w.written();
}

} // namepsace obs
#endif