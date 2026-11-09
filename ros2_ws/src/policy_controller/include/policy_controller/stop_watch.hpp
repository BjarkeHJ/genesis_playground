#ifndef STOP_WATCH_HPP
#define STOP_WATCH_HPP

#include <chrono>

struct StopWatch {
    using clock = std::chrono::high_resolution_clock;
    clock::time_point start_time;

    inline void tic() {
        start_time = clock::now();
    }

    inline double toc() {
        clock::time_point end_time = clock::now();
        std::chrono::duration<double, std::milli> elaps = end_time - start_time;
        return elaps.count();
    }
};

#endif