#pragma once
#include <cmath>
#include <memory>
#include <limits>
#include <opencv2/core.hpp>
#include "dataframe.h"
#include "sampler.h"

#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif

#define ENABLE_INTENSITY_COST 0

inline float angdiff(float angle1, float angle2)
{
    assert(0 <= angle1 && angle1 <= M_PI + 0.1);
    assert(0 <= angle2 && angle2 <= M_PI + 0.1);
    return M_PI_2 - std::abs(std::abs(angle1 - angle2) - M_PI_2);
}

// Normarized cross correlation (NCC) [-1.0, 1.0]
float ncc(const std::vector<float> &vec1, const std::vector<float> &vec2)
{
    size_t size1 = vec1.size();
    size_t size2 = vec2.size();
    assert(size1 == size2);

    double dot = 0, norm1 = 0, norm2 = 0;
    for (size_t i = 0; i < size1; i++)
    {
        float v1 = vec1.at(i);
        float v2 = vec2.at(i);
        dot += v1 * v2;
        norm1 += v1 * v1;
        norm2 += v2 * v2;
    }

    return dot / (std::sqrt(norm1 * norm2) + 1e-8);
}

float geometricCostPerView(const std::shared_ptr<SingleViewData> &view, const std::vector<cv::Point2f> &points_to_sample, float projected_angle)
{
    // gi (Sp,i,lp,i) defines angular difference between the detected orientation on the 2D samples Sp,i and the direction of 2D line lp,i:
    std::vector<float> sampled_orientation2d = sample_values(view->img_orientation2d, points_to_sample);
    std::vector<float> sampled_confidence = sample_values(view->img_confidence, points_to_sample);

    double cost = 0.0, sum_of_confidence = 0.0;
    size_t num_sample = points_to_sample.size();
    for (size_t s = 0; s < num_sample; s++)
    {
        float orientation2d = sampled_orientation2d.at(s);
        float confidence = sampled_confidence.at(s);
        cost += angdiff(orientation2d, projected_angle) * confidence;
        sum_of_confidence += confidence;
    }

    cost /= (sum_of_confidence + 1e-8);

    return cost;
}

float intensityCostPerView(const std::shared_ptr<SingleViewData> &neighbor_view, const std::vector<cv::Point2f> &points_to_sample_neighbor, const std::vector<float> &sampled_intensity_reference)
{
    std::vector<float> sampled_intensity_neighbor = sample_values(neighbor_view->img_intensity, points_to_sample_neighbor);

    float ncc_value = ncc(sampled_intensity_reference, sampled_intensity_neighbor);

    // Flip ncc value range [-1, 1] to [1, 0]
    float cost = -0.5f * ncc_value + 0.5f;

    return cost;
}

float calculateCost(const std::shared_ptr<SingleViewData> &reference_view, const MultiViewData &neighbor_views, const cv::Point &point, const PluckerLine &line, float radius, float num_sampling, float alpha = 0.1)
{
    // --- Reference view ---
    // We first project Lp to the reference image and get the corresponding 2D line lp,0.
    float projected_angle_reference = reference_view->camera.projectLine(line);

    // We then sample κ number of points uniformly along lp,0, centered at the p with radius rκ, obtaining κ 2D samples for the reference view Sp,0.
    std::vector<cv::Point2f> points_to_sample_reference = sample_point_along_line(projected_angle_reference, point, radius, num_sampling);

    // Initialize cost values
    double cost_geometric = 0.0, sum_of_gamma = 1e-8;
    double cost_intensity = 0.0;
    double num_neighbor = neighbor_views.size() + 1e-8;

    // Geometric cost of reference view
    double gamma_0 = 1 + num_neighbor; // `1` means 0-th (reference) view
    cost_geometric += gamma_0 * geometricCostPerView(reference_view, points_to_sample_reference, projected_angle_reference);
    sum_of_gamma += gamma_0;

#if ENABLE_INTENSITY_COST
    // Sampling intensity of reference view (use later)
    std::vector<float> sampled_intensity_reference = sample_values(reference_view->img_intensity, points_to_sample_reference);
#endif

    // --- Neighbor views ---
    double gamma_i = 1;
    size_t num_sample_points = points_to_sample_reference.size();
    for (const auto &neighbor_view : neighbor_views)
    {
        // Sample points of neighbor views
        std::vector<cv::Point2f> points_to_sample_neighbor;
        points_to_sample_neighbor.reserve(num_sample_points);
        for (const auto &point_on_reference : points_to_sample_reference)
        {
            // Once we obtain Sp,0, we shoot rays from the reference view’s origin towards each sample in Sp,0.
            PluckerLine ray = reference_view->camera.generateRayLine(point_on_reference);

            // We find intersection points with the 3D line Lp
            cv::Vec3f point_intersect = closestPoint(line, ray);

            // and re-project the points into i-th view and get corresponding 2D samples Sp,i
            cv::Vec2f point_on_neighbor = neighbor_view->camera.projectPoint(point_intersect);

            points_to_sample_neighbor.emplace_back(point_on_neighbor);
        }

        // Geometric cost of neighbor view
        float projected_angle_neighbor = neighbor_view->camera.projectLine(line);
        float cost = geometricCostPerView(neighbor_view, points_to_sample_neighbor, projected_angle_neighbor);

        // If the all sampled points are out of image
        if (cost == 0)
        {
            cost = 10000;
        }

        cost_geometric += gamma_i * cost;
        sum_of_gamma += gamma_i;

#if ENABLE_INTENSITY_COST
        // Intensity cost of neighbor view
        cost_intensity += intensityCostPerView(reference_view, points_to_sample_neighbor, sampled_intensity_reference);
#endif
    }

    cost_geometric /= sum_of_gamma;
    cost_intensity /= num_neighbor;

    return (1 - alpha) * cost_geometric + alpha * cost_intensity;
}