#pragma once
#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "line.h"
#include "camera.h"
#include "random.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

inline PluckerLine lineFromDepthDirection(const cv::Point &point, const Camera &camera, float depth, const cv::Vec3f &direction)
{
    cv::Vec3f point_local = camera.getLocalPointFromDepth(point, depth);
    cv::Vec3f point_world = camera.transformPointLocal2World(point_local);
    cv::Vec3f direction_normalized = direction / cv::norm(direction);
    return PluckerLine(point_world, point_world + direction_normalized);
}

inline float deg2rad(const float deg)
{
    return deg * (M_PI / 180.0);
}

inline float rad2deg(const float rad)
{
    return rad / (M_PI / 180.0);
}

template <typename T>
std::vector<size_t> argsort(const std::vector<T> &vec)
{
    std::vector<size_t> index_vec(vec.size());
    std::iota(index_vec.begin(), index_vec.end(), 0);
    std::stable_sort(index_vec.begin(), index_vec.end(), [&vec](size_t x, size_t y)
                     { return vec[x] < vec[y]; });

    return index_vec;
}

template <typename T>
size_t argmin(const std::vector<T> &vec)
{
    return std::min_element(vec.begin(), vec.end()) - vec.begin();
}

template <typename T>
void print1dvector(const std::vector<T> &vector)
{
    for (const auto &v : vector)
    {
        std::cout << v << ", ";
    }
    std::cout << std::endl;
}

template <typename T>
void print2dvector(const std::vector<std::vector<T>> &vectors)
{
    for (const auto &vector : vectors)
    {
        print1dvector(vector);
    }
}

cv::Scalar hsv2bgr(float h, float s = 1, float v = 1)
{
    assert(0.0 <= h && h <= 2 * M_PI);
    assert(0.0 <= s && s <= 1.0);
    assert(0.0 <= v && v <= 1.0);

    h *= (180.0f / M_PI); // rad2deg

    auto f = [&](float n)
    {
        float k = std::fmod(n + h / 60.0f, 6.0f);
        return v - v * s * std::max(0.0f, std::min({k, 4.0f - k, 1.0f}));
    };

    float red = f(5);
    float green = f(3);
    float blue = f(1);
    return 255.0 * cv::Scalar(blue, green, red);
}

void clamp(cv::InputArray _src, cv::OutputArray _dst, float low, float high)
{
    cv::Mat1f src = _src.getMat();
    CV_Assert(src.type() == CV_32FC1);
    _dst.create(src.size(), src.type());
    cv::Mat1f dst = _dst.getMat();

    dst.forEach(
        [&](float &pixel, const int *position) -> void
        {
            cv::Point point(position[1], position[0]);
            float pixel_src = src(point);
            pixel = std::clamp(pixel_src, low, high);
        });
}

void applyColorMapToAngle(cv::InputArray _img_angle, cv::OutputArray _img_color)
{
    // Get image and convert radian to degree
    cv::Mat1f img_angle = _img_angle.getMat() * (179.0f / M_PI);

    // Create output image
    _img_color.create(img_angle.size(), CV_8UC3);
    cv::Mat3b img_color = _img_color.getMat();

    // Hue, Saturation, Value
    cv::Mat1b img_hue;
    img_angle.convertTo(img_hue, CV_8UC1);
    cv::Mat1b img_saturation(img_hue.size(), 255);
    cv::Mat1b img_value(img_hue.size(), 255);

    // Merge HSV
    cv::Mat3b img_hsv;
    cv::merge(std::vector<cv::Mat>({img_hue, img_saturation, img_value}), img_hsv);

    cv::cvtColor(img_hsv, img_color, cv::COLOR_HSV2BGR);
}

std::string zfill(const int value, const unsigned width)
{
    std::ostringstream oss;
    oss << std::setw(width) << std::setfill('0') << value;
    return oss.str();
}