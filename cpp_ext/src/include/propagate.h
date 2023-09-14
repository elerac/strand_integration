#pragma once
#include <memory>
#include <vector>
#include <omp.h>
#include <opencv2/core.hpp>
#include "dataframe.h"
#include "line.h"
#include "costfunctions.h"

std::vector<PluckerLine> collectLineHypotheses(const std::shared_ptr<SingleViewData> &view, const cv::Point &point)
{
    // Table of offset from target pixel position (red-black pattern)
    std::vector<cv::Point> offsets{
        cv::Point(0, 0),
        //
        cv::Point(1, 0),
        cv::Point(-1, 0),
        cv::Point(0, 1),
        cv::Point(0, -1),
        //
        cv::Point(5, 0),
        cv::Point(-5, 0),
        cv::Point(0, 5),
        cv::Point(0, -5),
        // Above 9 pixels are used for faster setting ([Galliani et al, ICCV2015], Fig2(c))
        cv::Point(3, 0),
        cv::Point(-3, 0),
        cv::Point(0, 3),
        cv::Point(0, -3),
        //
        cv::Point(2, 1),
        cv::Point(2, -1),
        cv::Point(-2, 1),
        cv::Point(-2, -1),
        //
        cv::Point(1, 2),
        cv::Point(1, -2),
        cv::Point(-1, 2),
        cv::Point(-1, -2),
    };

    cv::Mat_<PluckerLine> img_line = view->img_line;
    Camera camera = view->camera;

    std::vector<PluckerLine> line_hypotheses;
    cv::Rect rect(0, 0, img_line.cols, img_line.rows);

    // We first shoot a ray from the camera center of the reference view through the reference pixel.
    PluckerLine ray = camera.generateRayLine(point);

    for (const auto &offset : offsets)
    {
        cv::Point point_neighbor = point + offset;
        if (point_neighbor.inside(rect))
        {
            // 3D line of the neighboring pixel
            PluckerLine line = img_line(point_neighbor);

            // We then find a 3D point on the ray that has the minimum distance to the 3D line of the neighboring pixel.
            cv::Vec3f new_position = closestPoint(ray, line);

            // A new line L is defined by that 3D point and the line direction from the neighboring pixel.
            PluckerLine new_line(new_position, new_position + line.direction());

            line_hypotheses.emplace_back(new_line);
        }
    }

    line_hypotheses.shrink_to_fit();
    return line_hypotheses;
}

void propagateSub(std::shared_ptr<SingleViewData> &reference_view, const MultiViewData &neighbor_views, float radius, int num_sampling, bool update_black)
{
    size_t width = reference_view->width(), height = reference_view->height();

#pragma omp parallel for schedule(dynamic, 1)
    for (int iy = 0; iy < height; iy++)
    {
        bool is_odd_row = (iy % 2);
        size_t offset = (is_odd_row == update_black);
        for (size_t ix = offset; ix < width; ix += 2)
        {
            cv::Point point(ix, iy);
            if (!reference_view->img_mask(point))
            {
                continue;
            }

            std::vector<PluckerLine> line_hypotheses = collectLineHypotheses(reference_view, point);

            std::vector<float> cost_vector;
            cost_vector.reserve(line_hypotheses.size());
            for (const auto &line : line_hypotheses)
            {
                float cost = calculateCost(reference_view, neighbor_views, point, line, radius, num_sampling);
                cost_vector.emplace_back(cost);
            }

            size_t index_minimal_cost = argmin(cost_vector);
            PluckerLine new_line = line_hypotheses.at(index_minimal_cost);

            reference_view->img_line(point) = new_line;
        }
    }
}

void propagate(std::shared_ptr<SingleViewData> &reference_view, const MultiViewData &neighbor_views, float radius, int num_sampling)
{
    // Black
    propagateSub(reference_view, neighbor_views, radius, num_sampling, true);

    // Red
    propagateSub(reference_view, neighbor_views, radius, num_sampling, false);
}