#pragma once
#include <opencv2/core.hpp>

cv::Mat1f composeExtrinsicMatrix(cv::InputArray _rotation_matrix, cv::InputArray _translation_vector)
{
    cv::Mat1f rotation_matrix = _rotation_matrix.getMat();
    cv::Mat1f translation_vector = _translation_vector.getMat();
    CV_Assert(rotation_matrix.type() == CV_32FC1 && rotation_matrix.size() == cv::Size(3, 3));
    CV_Assert(translation_vector.type() == CV_32FC1 && translation_vector.size() == cv::Size(1, 3));

    cv::Mat1f extrinsic_matrix = cv::Mat::eye(4, 4, CV_32FC1); // 4x4 matrix
    rotation_matrix.copyTo(extrinsic_matrix(cv::Range(0, 3), cv::Range(0, 3)));
    translation_vector.copyTo(extrinsic_matrix(cv::Range(0, 3), cv::Range(3, 4)));
    return extrinsic_matrix;
}

cv::Mat1f composeProjectionMatrix(cv::InputArray _intrinsic_matrix, cv::InputArray _extrinsic_matrix)
{
    cv::Mat1f intrinsic_matrix = _intrinsic_matrix.getMat();
    cv::Mat1f extrinsic_matrix = _extrinsic_matrix.getMat();
    CV_Assert(intrinsic_matrix.type() == CV_32FC1 && intrinsic_matrix.size() == cv::Size(3, 3));
    CV_Assert(extrinsic_matrix.type() == CV_32FC1 && extrinsic_matrix.size() == cv::Size(4, 4));

    cv::Mat1f projection_matrix; // 3x4 matrix
    projection_matrix = intrinsic_matrix * cv::Mat::eye(cv::Size(4, 3), CV_32FC1) * extrinsic_matrix;
    return projection_matrix;
}