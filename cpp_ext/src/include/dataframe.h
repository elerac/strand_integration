#pragma once
#include <vector>
#include <memory>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "line.h"
#include "geometry.h"
#include "camera.h"
#include "utils.h"
#include "random.h"

struct DirectionalPoint
{
    cv::Point3f point;
    cv::Vec3f direction;
};

class SingleViewData
{
public:
    // Images
    cv::Mat1f img_intensity;
    cv::Mat1f img_orientation2d;
    cv::Mat1f img_confidence;

    // Camera
    Camera camera;

    // 3D line map: Lp (unknown parameters)
    cv::Mat_<PluckerLine> img_line;

    // Represent valid pixel
    cv::Mat1b img_mask;

    // Depth limits
    float min_depth = 0;
    float max_depth = std::numeric_limits<float>::infinity();

public:
    cv::Size size() const
    {
        // Get size in order of (img_line, img_intensity, img_orientation2d, img_confidence, img_mask)
        std::vector<cv::Size> sizes = {img_line.size(), img_intensity.size(), img_orientation2d.size(), img_confidence.size(), img_mask.size()};

        // Get the first non-zero size
        for (auto &size : sizes)
        {
            if (size.width > 0 && size.height > 0)
            {
                return size;
            }
        }

        // Return zero size if all sizes are zero
        return cv::Size(0, 0);
    }

    size_t width() const
    {
        return size().width;
    }

    size_t height() const
    {
        return size().height;
    }

    SingleViewData() {}

    SingleViewData(size_t width, size_t height, const Camera &_camera) : camera(_camera)
    {
        img_intensity.create(height, width);
        img_orientation2d.create(height, width);
        img_confidence.create(height, width);
        img_line.create(height, width);
        img_mask.create(height, width);
    }

    SingleViewData(const cv::Mat1f &_img_intensity, const cv::Mat1f &_img_orientation2d, const cv::Mat1f &_img_confidence, const Camera &_camera) : img_intensity(_img_intensity), img_orientation2d(_img_orientation2d), img_confidence(_img_confidence), camera(_camera)
    {
        img_line.create(_img_intensity.size());

        CV_Assert(size() == img_orientation2d.size());
        CV_Assert(size() == img_confidence.size());

        initializeLine(0.01, 256);

        img_mask = 255.0f * cv::Mat1f::ones(size());
    }

    // Set random values to cv::Mat_<PluckerLine>
    void initializeLine(float low, float high)
    {
        img_line.create(size());
        this->min_depth = low;
        this->max_depth = high;
        set_random_line();
    }

    // Set random values to cv::Mat_<PluckerLine>
    void set_random_line()
    {
        cv::Mat1f img_depth;
        cv::Mat3f img_direction;

        img_depth.create(size());
        img_direction.create(size());

        for (int y = 0; y < height(); y++)
        {
            for (int x = 0; x < width(); x++)
            {
                img_depth(y, x) = randUniform(min_depth, max_depth);
                img_direction(y, x) = randSphere();
            }
        }

        set_line(img_depth, img_direction);
    }

    void set_line(const cv::Mat1f &img_depth, const cv::Mat3f &img_direction)
    {
        CV_Assert(img_depth.size() == img_direction.size());

        cv::Size new_size = img_depth.size();
        img_line.create(new_size);

        for (int y = 0; y < height(); y++)
        {
            for (int x = 0; x < width(); x++)
            {
                float depth = img_depth(y, x);
                cv::Vec3f direction = img_direction(y, x);

                PluckerLine line = lineFromDepthDirection(cv::Point(x, y), camera, depth, direction);
                img_line(y, x) = line;
            }
        }
    }

    void rescale(float scale)
    {
        assert(scale > 0);

        cv::resize(img_intensity, img_intensity, cv::Size(), scale, scale);
        cv::resize(img_orientation2d, img_orientation2d, cv::Size(), scale, scale);
        cv::resize(img_confidence, img_confidence, cv::Size(), scale, scale);
        cv::resize(img_line, img_line, cv::Size(), scale, scale);
        cv::resize(img_mask, img_mask, cv::Size(), scale, scale, cv::INTER_NEAREST);

        camera.resize(scale);
    }

    // Return 3D point of line
    cv::Point3f getPoint(const cv::Point &index) const
    {
        PluckerLine line = img_line(index);
        PluckerLine ray = camera.generateRayLine(index);
        cv::Point3f point = closestPoint(ray, line);
        return point;
    }

    // Return 3D direction of line
    cv::Point3f getDirection(const cv::Point &index) const
    {
        PluckerLine line = img_line(index);
        cv::Vec3f direction = line.unitdirection();
        return direction;
    }

    // Return 3D point and direction of line
    DirectionalPoint getDirectionalPoint(const cv::Point &index) const
    {
        cv::Point3f point = getPoint(index);
        cv::Vec3f direction = getDirection(index);
        return DirectionalPoint({point, direction});
    }

    // Return point cloud with direction at non-masked pixel
    std::vector<DirectionalPoint> getDirectionalPointCloud() const
    {
        std::vector<DirectionalPoint> pointcloud;

        for (size_t iy = 0; iy < height(); iy++)
        {
            for (size_t ix = 0; ix < width(); ix++)
            {
                cv::Point index(ix, iy);
                if (img_mask(index))
                {
                    DirectionalPoint point = getDirectionalPoint(index);
                    pointcloud.emplace_back(point);
                }
            }
        }
        pointcloud.shrink_to_fit();

        return pointcloud;
    }

    void getAngle(cv::OutputArray _dst) const
    {
        _dst.create(size(), CV_32FC1);
        cv::Mat1f dst = _dst.getMat();

        dst.forEach(
            [&](float &pixel, const int *position) -> void
            {
                cv::Point index(position[1], position[0]);
                PluckerLine line = img_line(index);
                float angle = camera.projectLine(line);
                pixel = angle;
            });
    }

    float getDepth(const cv::Point &index) const
    {
        cv::Vec3f point_world = getPoint(index);
        cv::Vec3f point_local = camera.transformPointWorld2Local(point_world);
        float z = point_local[2];
        return z;
    }

    void getDepth(cv::OutputArray _dst) const
    {
        _dst.create(size(), CV_32FC1);
        cv::Mat1f dst = _dst.getMat();

        dst.forEach(
            [&](float &pixel, const int *position) -> void
            {
                cv::Point index(position[1], position[0]);
                pixel = getDepth(index);
            });
    }

    void getDirection(cv::OutputArray _dst) const
    {
        _dst.create(size(), CV_32FC3);
        cv::Mat3f dst = _dst.getMat();

        dst.forEach(
            [&](cv::Vec3f &pixel, const int *position) -> void
            {
                cv::Point index(position[1], position[0]);
                pixel = getDirection(index);
            });
    }
};

class MultiViewData : public std::vector<std::shared_ptr<SingleViewData>>
{
public:
    MultiViewData(){};

private:
    std::vector<std::vector<size_t>> index_map;

public:
    void rescale(float scale)
    {
        for (auto &view : *this)
        {
            view->rescale(scale);
        }
    }

    void construct_neighboar_view_index_map()
    {
        // Construct index map of neighboar view
        // based on the distance of the camera origins
        std::vector<std::vector<float>> distance_map;
        auto inf = std::numeric_limits<float>::infinity();
        for (const auto &v1 : *this)
        {
            std::vector<float> distance_vector;
            for (const auto &v2 : *this)
            {
                auto origin1 = v1->camera.origin;
                auto origin2 = v2->camera.origin;
                float distance = cv::norm(origin1 - origin2);
                if (origin1 == origin2)
                {
                    // Set infinity to ignore same view
                    distance = inf;
                }

                distance_vector.emplace_back(distance);
            }
            auto neighbor_index_vector = argsort(distance_vector);

            distance_map.emplace_back(distance_vector);
            index_map.emplace_back(neighbor_index_vector);
        }
        index_map.shrink_to_fit();
    }

    std::vector<size_t> get_neighbor_index_vector(size_t pos, size_t num)
    {
        assert(num <= size() - 1);

        if (index_map.size() != size())
        {
            construct_neighboar_view_index_map();
        }

        auto neighbor_index_vector = index_map.at(pos);
        neighbor_index_vector.resize(num);
        return neighbor_index_vector;
    }

    MultiViewData get_neighbor(size_t pos, size_t num)
    {
        MultiViewData neighbor_views;
        auto neighbor_index_vector = get_neighbor_index_vector(pos, num);
        for (const size_t &i : neighbor_index_vector)
        {
            neighbor_views.emplace_back(at(i));
        }
        neighbor_views.shrink_to_fit();
        return neighbor_views;
    }

    std::vector<DirectionalPoint> getMergedDirectionalPointCloud() const
    {
        std::vector<DirectionalPoint> merged_pointcloud;
        for (const auto &view : *this)
        {
            auto pointcloud = view->getDirectionalPointCloud();
            merged_pointcloud.insert(merged_pointcloud.end(), pointcloud.begin(), pointcloud.end());
        }

        return merged_pointcloud;
    }
};
