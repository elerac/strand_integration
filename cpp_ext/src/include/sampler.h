#pragma once
#include <vector>
#include <cmath>
#include <opencv2/core.hpp>
#include <random.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

inline cv::Point2f sample_point_along_line(float angle, const cv::Point &center, float radius)
{
    // Sample along 1D line
    float s = randUniform(-radius, radius);
    // Rotate
    float x = s * std::cos(M_PI - angle) + center.x;
    float y = s * std::sin(M_PI - angle) + center.y;
    return cv::Point2f(x, y);
}

std::vector<cv::Point2f> sample_point_along_line(float angle, const cv::Point &center, float radius, size_t num)
{
    std::vector<cv::Point2f> points;
    points.reserve(num);
    for (size_t i = 0; i < num; i++)
    {
        auto point = sample_point_along_line(angle, center, radius);
        points.emplace_back(point);
    }
    return points;
}

std::vector<float> sample_values(const cv::Mat1f &img, const std::vector<cv::Point2f> &points)
{
    int height = img.rows;
    int width = img.cols;

    std::vector<float> values;
    values.reserve(points.size());

    for (const cv::Point &p : points)
    {
        float val;
        if (0 <= p.x && p.x < width && 0 <= p.y && p.y < height)
        {
            // Inside image
            val = img.at<float>(p);
        }
        else
        {
            // Outside image
            val = 0.0f;
        }
        values.emplace_back(val);
    }

    return values;
}