#ifndef OBS_WRITER_HPP_
#define OBS_WRITER_HPP_

#include <span>
#include <array>
#include <algorithm>
#include <cstddef>

class ObsWriter {
public:
    explicit ObsWriter(std::span<float> buf) : buf_(buf) {}

    // raw value or scaled 
    void put(float x) { buf_[i_++] = x; }
    void put(float x, float scale) { buf_[i_++] = x * scale; }

    // put array of known size
    template<size_t N>
    void put(const std::array<float, N>& v, float scale=0.f) {
        for (float x : v) scale ? put(x, scale) : put(x);
    }
    size_t written() const { return i_; }
private:
    std::span<float> buf_;
    size_t i_ = 0;
};


#endif // OBS_WRITER_HPP_