/*
 * References
 * - Matthew T. Mason, Mechanics of Manipulation Lecture note 9, https://www.cs.cmu.edu/afs/cs/academic/class/16741-s07/www/index.html
 * - Yan-Bin Jia, "Plücker Coordinates for Lines in the Space", https://faculty.sites.iastate.edu/jia/files/inline-files/plucker-coordinates.pdf
 * - Peter Corke, https://github.com/petercorke/spatialmath-matlab/blob/master/Plucker.m
 */

#pragma once
#include <cmath> // std::atanf
#include <opencv2/core.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CV_32FC6 CV_MAKETYPE(CV_32F, 6)

inline bool isClose(const float a, const float b, const float eps = 1e-03)
{
    return std::abs(a - b) < eps;
}

template <typename _Tp, int cn>
inline cv::Vec<_Tp, cn> cross(const cv::Vec<_Tp, cn> &a, const cv::Vec<_Tp, cn> &b)
{
    return a.cross(b);
}

template <typename _Tp, int cn>
inline _Tp dot(const cv::Vec<_Tp, cn> &a, const cv::Vec<_Tp, cn> &b)
{
    return a.dot(b);
}

template <typename _Tp, int cn>
inline _Tp norm2(const cv::Vec<_Tp, cn> &a)
{
    return a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
}

// A line expressed as Plücker coordinates
class PluckerLine : public cv::Vec<float, 6>
{
public:
    // Directly initialize with six parameters
    PluckerLine(float dx, float dy, float dz, float mx, float my, float mz) : cv::Vec<float, 6>(dx, dy, dz, mx, my, mz) {}

    PluckerLine(const cv::Vec<float, 6> &vec) : cv::Vec<float, 6>(vec) {}

    // Initialize from two points (cv::Point3f)
    PluckerLine(const cv::Point3f &point1, const cv::Point3f &point2) : PluckerLine(cv::Vec3f(point1),
                                                                                    cv::Vec3f(point2)) {}

    // Initialize from two points (cv::Vec3f)
    PluckerLine(const cv::Vec3f &vec1, const cv::Vec3f &vec2) : PluckerLine(cv::Vec4f(vec1[0], vec1[1], vec1[2], 1),
                                                                            cv::Vec4f(vec2[0], vec2[1], vec2[2], 1)) {}

    // Initialize from two points in homogeneous
    PluckerLine(const cv::Vec4f &vec1, const cv::Vec4f &vec2) : PluckerLine(vec1[3] * vec2[0] - vec2[3] * vec1[0],
                                                                            vec1[3] * vec2[1] - vec2[3] * vec1[1],
                                                                            vec1[3] * vec2[2] - vec2[3] * vec1[2],
                                                                            vec1[1] * vec2[2] - vec2[1] * vec1[2],
                                                                            vec1[2] * vec2[0] - vec2[2] * vec1[0],
                                                                            vec1[0] * vec2[1] - vec2[0] * vec1[1]) {}

    // Default value (x-axis)
    PluckerLine() : PluckerLine(cv::Vec3f(0, 0, 0), cv::Vec3f(1, 0, 0)){};

    PluckerLine reverse() const
    {
        return PluckerLine((*this)[5], (*this)[4], (*this)[3], (*this)[2], (*this)[1], (*this)[0]);
    }

    // direction = point2 - point1
    cv::Vec3f direction() const
    {
        return cv::Vec3f((*this)[0], (*this)[1], (*this)[2]);
    }

    void direction(const cv::Vec3f &new_direction)
    {
        (*this)[0] = new_direction[0];
        (*this)[1] = new_direction[1];
        (*this)[2] = new_direction[2];
    }

    // Direction as a unit vector
    cv::Vec3f unitdirection() const
    {
        return cv::normalize(direction());
    }

    // moment = point1 x point2 (x: cross product)
    cv::Vec3f moment() const
    {
        return cv::Vec3f((*this)[3], (*this)[4], (*this)[5]);
    }

    void moment(const cv::Vec3f &new_moment)
    {
        (*this)[3] = new_moment[0];
        (*this)[4] = new_moment[1];
        (*this)[5] = new_moment[2];
    }

    // Move to new position and keep same direction
    void position(const cv::Vec3f &new_position)
    {
        cv::Vec3f new_moment = new_position.cross(new_position + direction());
        moment(new_moment);
    }

    // Sample a point on line
    cv::Vec3f point(const float lambda = 0.0f) const
    {
        // Find point on line closest to origin
        cv::Vec3f d = direction(), m = moment();
        cv::Vec3f point_o = d.cross(m) / d.dot(d);

        // Move point along line
        return point_o + lambda * unitdirection();
    }
};

namespace cv
{
    // Define `DataType<PluckerLine>` enables to construct `cv::Mat_<PluckerLine>`
    // See also https://github.com/opencv/opencv/blob/1339ebaa84b923d34e1f4ec4a8a2d2e3f45df37f/modules/core/include/opencv2/core/matx.hpp#L449
    template <>
    class DataType<PluckerLine>
    {
    public:
        typedef Vec<float, 6> value_type;
        typedef Vec<typename DataType<float>::work_type, 6> work_type;
        typedef float channel_type;
        typedef value_type vec_type;

        enum
        {
            generic_type = 0,
            channels = 6,
            fmt = DataType<channel_type>::fmt + ((channels - 1) << 8),
#ifdef OPENCV_TRAITS_ENABLE_DEPRECATED
            depth = DataType<channel_type>::depth,
            type = CV_MAKETYPE(depth, channels),
#endif
            _dummy_enum_finalizer = 0
        };
    };

    namespace traits
    {
        template <>
        struct Depth<PluckerLine>
        {
            enum
            {
                value = Depth<float>::value
            };
        };
        template <>
        struct Type<PluckerLine>
        {
            enum
            {
                value = CV_MAKETYPE(Depth<float>::value, 6)
            };
        };
    }
} // namespace

inline bool operator==(const PluckerLine &line1, const PluckerLine &line2)
{
    return isClose(dot(cv::normalize(line1), cv::normalize(line2)), 1);
}

// Reciprocal product; virtual product
float reciprocal(const PluckerLine &line1, const PluckerLine &line2)
{
    cv::Vec3d d1 = line1.direction(), d2 = line2.direction();
    cv::Vec3d m1 = line1.moment(), m2 = line2.moment();
    return dot(d1, m2) + dot(m1, d2);
}

inline bool isParallel(const PluckerLine &line1, const PluckerLine &line2)
{
    cv::Vec3f d1 = line1.direction(), d2 = line2.direction();
    return isClose(cv::norm(cross(d1, d2)), 0.0f);
}

// Distance between 3D point and line
float distance(const cv::Vec3f &point, PluckerLine &line)
{
    cv::Vec3f ud = line.unitdirection();
    cv::Vec3f mp = cross(line.point(), ud) - cross(point, ud);
    return cv::norm(mp);
}

// Minimal distance between two lines
float distance(const PluckerLine &line1, const PluckerLine &line2)
{
    if (isParallel(line1, line2))
    {
        cv::Vec3f d1 = line1.direction(), d2 = line2.direction();
        cv::Vec3f m1 = line1.moment(), m2 = line2.moment();
        float norm_d1 = cv::norm(d1);
        float norm_d2 = cv::norm(d2);
        float s = norm_d2 / norm_d1;
        return cv::norm(cross(d1, (m1 - m2 / s))) / (norm_d1 * norm_d1);
    }
    else
    {
        return std::abs(reciprocal(line1, line2)) / cv::norm(cross(line1.direction(), line2.direction()));
    }
}

inline bool isIntersect(const PluckerLine &line1, const PluckerLine &line2)
{
    // return isClose(reciprocal(line1, line2), 0.0f);
    return isClose(distance(line1, line2), 0.f);
}

float angle(const PluckerLine &line1, const PluckerLine &line2)
{
    cv::Vec3f ud1 = line1.unitdirection(), ud2 = line2.unitdirection();
    return std::asin(cv::norm(cross(ud1, ud2)));
}

// Return point on the `line1` where the distance to the `line2` is closest
cv::Point3f closestPoint(const PluckerLine &line1, const PluckerLine &line2)
{
    cv::Vec3f d1 = line1.direction(), d2 = line2.direction();
    cv::Vec3f m1 = line1.moment(), m2 = line2.moment();
    cv::Vec3f cross_d1d2 = cross(d1, d2);
    return (cross(-m1, cross(d2, cross_d1d2)) + dot(m2, cross_d1d2) * d1) / norm2(cross_d1d2);
}

cv::Mat1f composeLineProjectionMatrix(cv::InputArray _projection_matrix)
{
    cv::Mat1f projection_matrix = _projection_matrix.getMat();
    CV_Assert(projection_matrix.type() == CV_32FC1 && projection_matrix.size() == cv::Size(4, 3));

    // Line projection matrix is [P2 ∧ P3]
    //                           [P3 ∧ P1]
    //                           [P1 ∧ P2]  (3x6 matrix)
    // where PiT are the rows of the point proection matrix (aka camera matrix)
    // [Hartley and Zisserman, Eq.(8.3)]

    // Separate projection matrix into individual row
    cv::Vec4f projection_matrix0 = projection_matrix(cv::Range(0, 1), cv::Range::all());
    cv::Vec4f projection_matrix1 = projection_matrix(cv::Range(1, 2), cv::Range::all());
    cv::Vec4f projection_matrix2 = projection_matrix(cv::Range(2, 3), cv::Range::all());

    // (P1 ∧ P2) = the intersection of the planes P1, P2
    // which can be caluculated by using Plucker line scheme and reversing order
    // [Hartley and Zisserman, Eq.(3.9) and Eq.(3.10)]
    auto line1 = PluckerLine(projection_matrix1, projection_matrix2).reverse();
    auto line2 = PluckerLine(projection_matrix2, projection_matrix0).reverse();
    auto line3 = PluckerLine(projection_matrix0, projection_matrix1).reverse();

    // Copy values to line projection matrix
    cv::Mat1f line_projection_matrix(cv::Size(6, 3), CV_32FC1); // 3x6 matrix
    PluckerLine line_array[3] = {line1, line2, line3};
    for (size_t i = 0; i < 3; i++)
    {
        PluckerLine line = line_array[i];
        for (size_t j = 0; j < 6; j++)
        {
            line_projection_matrix.at<float>(i, j) = line[j];
        }
    }

    return line_projection_matrix;
}

inline float projectLines(const PluckerLine &line, const cv::Matx<float, 3, 6> &line_projection_matrix)
{
    // See also [Hartley and Zisserman, Eq.(8.4)]
    // Reversing order and dot product represents the product (L|Lˆ) which defined in [Hartley and Zisserman, Eq.(3.13)]
    cv::Vec<float, 6> line_reversed = line.reverse();
    cv::Vec3f projected_line = line_projection_matrix * line_reversed; // (3x6) x (6x1) = (3x1)

    // 2D line parameters of ax + by + c = 0
    float a = projected_line[0];
    float b = projected_line[1];
    // float c = projected_line[2];

    // The vector n=(a, b) is perpendicular to the line,
    // so the angle of line is calculated as follows
    float angle = std::atan2(-a, b); // [-pi, pi]

    if (angle < 0)
    {
        angle += M_PI;
    }

    angle = M_PI - angle;

    return angle;
}