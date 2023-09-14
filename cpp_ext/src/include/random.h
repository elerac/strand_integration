#pragma once
#include <cmath>
#include <random>
#include <chrono>
#include <vector>
#include <opencv2/core.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
std::mt19937 generator(seed);
inline std::uniform_real_distribution<float> uniform01(0.0f, 1.0f);

// Sample uniform distribution on sphere
inline cv::Vec3f randSphere()
{
    // http://corysimon.github.io/articles/uniformdistn-on-sphere/
    float theta = (2 * M_PI) * uniform01(generator);
    float phi = std::acos(1 - 2 * uniform01(generator));
    float sin_phi = std::sin(phi);
    float x = sin_phi * std::cos(theta);
    float y = sin_phi * std::sin(theta);
    float z = std::cos(phi);
    return cv::Vec3f(x, y, z);
}

// Sample uniform distribution
inline float randUniform(float low, float high)
{
    std::uniform_real_distribution<float> uniform(low, high);
    return uniform(generator);
}