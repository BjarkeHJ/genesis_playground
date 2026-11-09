#ifndef POLY_TRAJECTORY_HPP_
#define POLY_TRAJECTORY_HPP_

#include <cmath>
#include <utility>
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <Eigen/Core>

constexpr int N_COEFF = 8;
using SegCoeffs = Eigen::Matrix<double, N_COEFF, 3>;  // coeffs[s] : rows n, cols xyz

struct TrajRef {
    Eigen::Vector3d pos;
    Eigen::Vector3d vel;
};

class PolyTrajectory {
public:
    PolyTrajectory(uint32_t id, std::vector<SegCoeffs> coeffs, std::vector<double> durations)
        : id_(id)
        , coeffs_(std::move(coeffs))
        , durations_(std::move(durations)) 
    {
        if (coeffs_.empty() || coeffs_.size() != durations_.size()) {
            throw std::invalid_argument("coeffs/durations size mismatch");
        }        
        t_nodes_.resize(durations_.size() + 1, 0.0);
        for (size_t s=0; s<durations_.size(); ++s) {
            if (!std::isfinite(durations_[s]) || durations_[s] <= 0.0) {
                throw std::invalid_argument("non-positive duration");
            }
            t_nodes_[s + 1] = t_nodes_[s] + durations_[s];
        }
    }

    TrajRef eval(double t) const {
        // t = seconds since start
        t = std::clamp(t, 0.0, t_end());       
        // Find which segment 't' corresponds to
        auto it = std::upper_bound(t_nodes_.begin(), t_nodes_.end(), t);
        int seg = std::clamp<int>(int(it - t_nodes_.begin()) - 1, 0, n_segments() - 1);

        const double T = durations_[seg]; // segment time
        const double tau = (t - t_nodes_[seg]) / T; // duration-normalized segment time
        const SegCoeffs& c = coeffs_[seg];

        // basis rows: d^r/dtau^r of tau^n = n!/(n-r)! * tau^(n-r)
        Eigen::Matrix<double, 1, N_COEFF> b0, b1;
        b0.setZero(); b1.setZero();
        double p = 1.0; // tau^n
        for (int n=0; n<N_COEFF; ++n) {
            b0(n) = p;
            if (n + 1 < N_COEFF) b1(n + 1) = (n + 1) * p;
            p *= tau;
        }

        return 
        {
            (b0 * c).transpose(),
            (b1 * c).transpose() / T,
        };
    }

    uint32_t id() const { return id_; }
    int n_segments() const { return int(durations_.size()); }
    double t_end() const { return t_nodes_.back(); }

private:
    uint32_t id_;
    std::vector<SegCoeffs> coeffs_;
    std::vector<double> durations_;
    std::vector<double> t_nodes_;
};

#endif