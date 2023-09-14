#pragma once
#include <cmath>
#include <vector>
#include <omp.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "random.h"
#include "costfunctions.h" // angdiff

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif

void calcOrientation2d(
    cv::InputArray _img_src,
    cv::OutputArray _img_orientation2d,
    cv::OutputArray _img_confidence,
    std::function<cv::Mat(double)> getKernel = [](double theta) -> cv::Mat
    { return cv::getGaborKernel(cv::Size(31, 31), 2, theta, 3, 0.5); },
    int num = 64)
{
    // Input and Output mat
    cv::Mat1f img_src = _img_src.getMat();
    _img_orientation2d.create(img_src.size(), CV_32FC1);
    _img_confidence.create(img_src.size(), CV_32FC1);
    cv::Mat1f img_orientation2d = _img_orientation2d.getMat();
    cv::Mat1f img_confidence = _img_confidence.getMat();

    // Add jitter to handle background area
    double _min, max;
    cv::minMaxLoc(img_src, &_min, &max);
    double jitter = max / 10000.0;
    cv::Mat1f img_jitter(img_src.size());
    cv::randu(img_jitter, cv::Scalar(-jitter * 0.5), cv::Scalar(jitter * 0.5));
    img_src += img_jitter;

    // Generate vector of angle
    std::vector<float> angle_vector;
    angle_vector.reserve(num);
    for (float i = 0; i < num; i++)
    {
        float theta = i * (M_PI / (float)num);
        angle_vector.emplace_back(theta);
    }

    // Filtering each image
    std::vector<cv::Mat1f> imvec_filtered;
    imvec_filtered.reserve(num);
    imvec_filtered.resize(num);
#pragma omp parallel for
    for (int i = 0; i < num; i++)
    {
        // Get kernel
        float theta = angle_vector.at(i);
        cv::Mat kernel = getKernel(theta);

        // 2D conv
        cv::Mat1f img_filtered;
        cv::filter2D(img_src, img_filtered, CV_32F, kernel);
        imvec_filtered.at(i) = img_filtered;
    }

    // Get orientation2d and confidence from filtered images
    int height = img_src.size().height;
    int width = img_src.size().width;
#pragma omp parallel for
    for (int iy = 0; iy < height; iy++)
    {
        for (int ix = 0; ix < width; ix++)
        {
            cv::Point point(ix, iy);

            // Extract sequential response at (ix, iy) pixel
            std::vector<float> response_values;
            response_values.reserve(angle_vector.size());
            for (const auto &img_filtered : imvec_filtered)
            {
                float response = std::abs(img_filtered(point));
                response_values.emplace_back(response);
            }

            // Orientation from the highest response
            size_t i_max = std::max_element(response_values.begin(), response_values.end()) - response_values.begin();
            float orientation2d = angle_vector.at(i_max);

            // Confidence from variance
            float variance = 0, sum_of_response = 0;
            for (size_t i = 0; i < num; i++)
            {
                float response = response_values.at(i);
                float d = angdiff(orientation2d, angle_vector.at(i));
                variance += d * d * response;
                sum_of_response += response;
            }
            variance /= sum_of_response;
            float confidence = 1.0 / (variance * variance);

            // Set values
            // orientation2d = (M_PI + M_PI_2) - orientation2d;
            // orientation2d = M_PI_2 + orientation2d;
            orientation2d = (M_PI - orientation2d) - M_PI_2;

            while (!(0 <= orientation2d && orientation2d < M_PI))
            {
                if (orientation2d < 0)
                {
                    orientation2d += M_PI;
                }
                if (orientation2d >= M_PI)
                {
                    orientation2d -= M_PI;
                }
            }

            img_orientation2d(point) = orientation2d;
            img_confidence(point) = confidence;
        }
    }
}