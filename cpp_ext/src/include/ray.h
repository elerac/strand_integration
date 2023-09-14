#pragma once
#include <opencv2/core.hpp>

class Ray
{
public:
    cv::Point3f origin;
    cv::Point3f direction;

    Ray(const cv::Point3f &_origin, const cv::Point3f &_direction) : origin(_origin), direction(_direction){};

    cv::Point3f operator()(float t) const
    {
        return origin + t * direction;
    };
};
