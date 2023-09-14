#pragma once
#include <opencv2/core.hpp>
#include "geometry.h"
#include "line.h"
#include "ray.h"

class Camera
{
public:
    float fx, fy, f, px, py;
    cv::Point3f origin; // origin world
    cv::Matx33f intrinsic_matrix;
    cv::Matx44f extrinsic_matrix;     // World-to-Local
    cv::Matx44f extrinsic_matrix_inv; // Local-to-World
    cv::Matx34f projection_matrix;
    cv::Matx<float, 3, 6> line_projection_matrix;

private:
    float div_fx, div_fy;

public:
    void initialize()
    {
        extrinsic_matrix_inv = extrinsic_matrix.inv();
        projection_matrix = composeProjectionMatrix(intrinsic_matrix, extrinsic_matrix);
        line_projection_matrix = composeLineProjectionMatrix(projection_matrix);

        fx = intrinsic_matrix(0, 0);
        fy = intrinsic_matrix(1, 1);
        f = (fx + fy) * 0.5f;
        px = intrinsic_matrix(0, 2);
        py = intrinsic_matrix(1, 2);
        div_fx = 1 / fx;
        div_fy = 1 / fy;

        origin = transformPointLocal2World(cv::Vec3f(0, 0, 0));
    }

    Camera() {}
    Camera(const cv::Matx33f &K, const cv::Matx33f &R, const cv::Vec3f &t)
    {
        cv::Matx44f Rt;

        // Copy R and t
        for (size_t j = 0; j < 3; j++)
        {
            // i = 0, 1, 2
            for (size_t i = 0; i < 3; i++)
            {
                Rt(j, i) = R(j, i);
            }
            // i = 3
            Rt(j, 3) = t(j);
        }

        // Set last row
        Rt(3, 0) = 0;
        Rt(3, 1) = 0;
        Rt(3, 2) = 0;
        Rt(3, 3) = 1;

        // Set intrinsic and extrinsic matrix
        intrinsic_matrix = K;
        extrinsic_matrix = Rt;
        initialize();
    }

    Camera(const cv::Matx33f &_intrinsic_matrix, const cv::Matx44f &_extrinsic_matrix) : intrinsic_matrix(_intrinsic_matrix), extrinsic_matrix(_extrinsic_matrix)
    {
        initialize();
    }

    void resize(float scale)
    {
        intrinsic_matrix(0, 0) *= scale;
        intrinsic_matrix(1, 1) *= scale;
        intrinsic_matrix(0, 2) *= scale;
        intrinsic_matrix(1, 2) *= scale;
        initialize();
    }

    cv::Vec3f transformPoint(const cv::Vec3f &point_src, const cv::Matx44f &transform_matrix) const
    {
        cv::Vec4f point_src_homo(point_src[0], point_src[1], point_src[2], 1.0f);
        cv::Vec4f point_dst_homo = transform_matrix * point_src_homo;
        cv::Vec3f point_dst(point_dst_homo[0], point_dst_homo[1], point_dst_homo[2]);
        point_dst /= point_dst_homo[3];
        return point_dst;
    }

    cv::Vec3f transformPointWorld2Local(const cv::Vec3f &point_world) const
    {
        cv::Vec3f point_local = transformPoint(point_world, extrinsic_matrix);
        return point_local;
    }

    cv::Vec3f transformPointLocal2World(const cv::Vec3f &point_local) const
    {
        cv::Vec3f point_world = transformPoint(point_local, extrinsic_matrix_inv);
        return point_world;
    }

    // Get 3D point from depth
    cv::Vec3f getLocalPointFromDepth(const cv::Point2f &point, float depth) const
    {
        cv::Point3f point_local(depth * (point.x - px) * div_fx, depth * (point.y - py) * div_fy, depth);
        return point_local;
    }

    // Ray from the camera center through the pixel in Plucker coordinates
    PluckerLine generateRayLine(const cv::Point2f &point) const
    {
        cv::Point3f point_on_sensor_local = getLocalPointFromDepth(point, f);
        cv::Point3f point_on_sensor_world = transformPointLocal2World(point_on_sensor_local);
        PluckerLine ray(origin, point_on_sensor_world);
        return ray;
    }

    Ray generateRay(const cv::Point2f &point) const
    {
        cv::Point3f point_on_sensor_local = getLocalPointFromDepth(point, f);
        cv::Point3f point_on_sensor_world = transformPointLocal2World(point_on_sensor_local);
        Ray ray(origin, point_on_sensor_world - origin);
        return ray;
    }

    cv::Vec2f projectPoint(const cv::Vec3f &point_world) const
    {
        cv::Vec4f point_world_homo(point_world[0], point_world[1], point_world[2], 1.0f);
        cv::Vec3f point_homo = projection_matrix * point_world_homo;
        cv::Vec2f point(point_homo[0], point_homo[1]);
        return point / point_homo[2];
    }

    float projectLine(const PluckerLine &line) const
    {
        return projectLines(line, line_projection_matrix);
    }

    float projectLine(const cv::Vec<float, 6> &line) const
    {
        return projectLines(PluckerLine(line), line_projection_matrix);
    }
};