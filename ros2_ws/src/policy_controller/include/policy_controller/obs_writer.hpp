#ifndef OBS_WRITER_HPP_
#define OBS_WRITER_HPP_

#include <algorithm>
#include <cassert>
#include <span>
#include <cstddef>

#include <Eigen/Core>

class ObsWriter {
public:
    explicit ObsWriter(std::span<float> buf) : buf_(buf) {}

    // raw value or scaled 
    void put(float x) { buf_[i_++] = x; }
    void put_clamped(float x, float scale) { put(std::clamp(x * scale, -1.0f, 1.0f)); }

    // put array of known size
    template<int N>
    void put(const Eigen::Matrix<float, N, 1>& v) {
        for (int k = 0; k < N; ++k) put(v[k]);
    }

    template<int N>
    void put_clamped(const Eigen::Matrix<float, N, 1>& v, float scale) {
        for (int k = 0; k < N; ++k) put_clamped(v[k], scale);
    }

    void put(std::span<const float> v) { for (float x : v) put(x); }

    size_t written() const { return i_; }
private:
    std::span<float> buf_;
    size_t i_ = 0;
};


#endif // OBS_WRITER_HPP_