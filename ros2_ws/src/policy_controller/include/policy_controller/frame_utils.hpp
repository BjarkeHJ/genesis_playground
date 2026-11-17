#ifndef FRAME_UTILS_HPP_
#define FRAME_UTILS_HPP_

// Frame conventions
// Policy / training : world ENU (x East, y North, z Up),   body FLU (x Fwd, y Left,  z Up)
// PX4               : world NED (x North, y East, z Down), body FRD (x Fwd, y Right, z Down)
// Quaternions are Hamilton, stored (w, x, y, z), rotating body -> world.
//
// Both base changes are 180 deg rotations, so every mapping is its own inverse:
// ENU <-> NED : (x, y, z) -> (y, x, -z)   == rotation pi about (1, 1, 0)/sqrt(2)
// FLU <-> FRD : (x, y, z) -> (x, -y, -z)  == rotation pi about x
// The two named directions are kept anyway so call sites read correctly.

#include <array>
#include <cmath>
#include <numbers>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace frame {

inline float wrap_pi(float a) {
    return std::remainder(a, 2.0f * std::numbers::pi_v<float>); // [-pi, pi]
}

// yaw = heading of body x-axis in the world frame (ZYX Euler).
// Same formula as torch_math_helpers.quat_yaw, valid in either convention.
inline float yaw_of(const Eigen::Quaternionf& q) {
    return std::atan2(2.0f * (q.w() * q.z() + q.x() * q.y()),
                      1.0f - 2.0f * (q.y() * q.y() + q.z() * q.z()));
}

// ENU yaw: from East, CCW positive.  NED yaw: from North, CW positive.
inline float yaw_enu_to_ned(float yaw_enu) { return wrap_pi(std::numbers::pi_v<float> / 2.0f - yaw_enu); }
inline float yaw_ned_to_enu(float yaw_ned) { return wrap_pi(std::numbers::pi_v<float> / 2.0f - yaw_ned); }

// Yaw rate about world z (z flips between ENU and NED)
inline float yaw_rate_enu_to_ned(float r) { return -r; }
inline float yaw_rate_ned_to_enu(float r) { return -r; }

// ---------------------------------------------------------------- vectors
// World-frame vectors: position, linear velocity, acceleration
inline Eigen::Vector3f ned_to_enu(const Eigen::Vector3f& v) { return {v.y(), v.x(), -v.z()}; }
inline Eigen::Vector3f enu_to_ned(const Eigen::Vector3f& v) { return {v.y(), v.x(), -v.z()}; }

// Body-frame vectors: angular velocity, body forces/torques
inline Eigen::Vector3f frd_to_flu(const Eigen::Vector3f& v) { return {v.x(), -v.y(), -v.z()}; }
inline Eigen::Vector3f flu_to_frd(const Eigen::Vector3f& v) { return {v.x(), -v.y(), -v.z()}; }

// ---------------------------------------------------------------- attitude
namespace detail {
// q_NED<-ENU : rotation pi about (1,1,0)/sqrt(2)   (self-inverse up to sign)
inline const Eigen::Quaternionf& q_ned_enu() {
    static const Eigen::Quaternionf q(0.0f, std::numbers::sqrt2_v<float> / 2.0f,
                                            std::numbers::sqrt2_v<float> / 2.0f, 0.0f);
    return q;
}
// q_FLU<-FRD : rotation pi about x                  (self-inverse up to sign)
inline const Eigen::Quaternionf& q_flu_frd() {
    static const Eigen::Quaternionf q(0.0f, 1.0f, 0.0f, 0.0f);
    return q;
}
// Keep w >= 0 so q and -q (same rotation) compare and log consistently
inline Eigen::Quaternionf canonical(Eigen::Quaternionf q) {
    if (q.w() < 0.0f) q.coeffs() *= -1.0f;
    return q.normalized();
}
} // namespace detail

// q_world<-body:  q_enu_flu = q_ENU<-NED * q_ned_frd * q_FRD<-FLU
inline Eigen::Quaternionf ned_frd_to_enu_flu(const Eigen::Quaternionf& q_ned_frd) {
    return detail::canonical(detail::q_ned_enu() * q_ned_frd * detail::q_flu_frd());
}
// q_ned_frd = q_NED<-ENU * q_enu_flu * q_FLU<-FRD
inline Eigen::Quaternionf enu_flu_to_ned_frd(const Eigen::Quaternionf& q_enu_flu) {
    return detail::canonical(detail::q_ned_enu() * q_enu_flu * detail::q_flu_frd());
}

// ---------------------------------------------------------------- px4_msgs array adapters
// px4_msgs use std::array<float, N>; quaternions are stored [w, x, y, z].
inline Eigen::Vector3f vec3(const std::array<float, 3>& a) { return {a[0], a[1], a[2]}; }
inline std::array<float, 3> arr3(const Eigen::Vector3f& v) { return {v.x(), v.y(), v.z()}; }

inline Eigen::Quaternionf quat_wxyz(const std::array<float, 4>& a) { return {a[0], a[1], a[2], a[3]}; }
inline std::array<float, 4> arr_wxyz(const Eigen::Quaternionf& q) { return {q.w(), q.x(), q.y(), q.z()}; }

} // namespace frame

#endif