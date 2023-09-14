#pragma once
#include <memory>
#include <omp.h>
#include <opencv2/core.hpp>
#include "random.h"
#include "dataframe.h"
#include "costfunctions.h"

void refinement(std::shared_ptr<SingleViewData> &reference_view, const MultiViewData &neighbor_views, float radius, int num_sampling, float depth_perturbation = 4.f, float direction_perturbation = 0.1f)
{
    size_t height = reference_view->height(), width = reference_view->width();
    float min_depth = reference_view->min_depth, max_depth = reference_view->max_depth;
    float depth_perturbation_half = depth_perturbation * 0.5f;

#pragma omp parallel for schedule(dynamic, 1)
    for (int iy = 0; iy < height; iy++)
    {
        for (size_t ix = 0; ix < width; ix++)
        {
            cv::Point index(ix, iy);
            if (!reference_view->img_mask(index))
            {
                continue;
            }

            // --- Old line ---
            PluckerLine line_old = reference_view->img_line(index);
            float cost_old = calculateCost(reference_view, neighbor_views, index, line_old, radius, num_sampling);

            // --- New line ---
            // Random perturbation
            float d_depth = randUniform(-depth_perturbation_half, depth_perturbation_half);
            float depth = reference_view->getDepth(index);
            float depth_new = std::clamp(depth + d_depth, min_depth, max_depth); // depth value must be positive
            cv::Vec3f d_direction = direction_perturbation * randSphere();
            cv::Vec3f direction = reference_view->getDirection(index);
            cv::Vec3f direction_new = cv::normalize(direction + d_direction);

            // Construct new line
            PluckerLine line_new = lineFromDepthDirection(index, reference_view->camera, depth_new, direction_new);
            float cost_new = calculateCost(reference_view, neighbor_views, index, line_new, radius, num_sampling);

            // Update
            if (cost_new < cost_old)
            {
                reference_view->img_line(index) = line_new;
            }
        }
    }
}