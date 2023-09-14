#include <iostream>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/make_iterator.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "cv_typecaster.h"
#include "camera.h"
#include "dataframe.h"
#include "propagate.h"
#include "refinement.h"
#include "sampler.h"
#include "pbrt.h"
#include "line.h"
#include "orientation.h"

namespace nb = nanobind;
using namespace nb::literals;

typedef nb::ndarray<nb::numpy, const float, nb::shape<nb::any, 3>, nb::c_contig, nb::device::cpu> ndarray_f32_3d;

NB_MODULE(_strandtools_impl, m)
{
    nb::class_<Camera>(m, "Camera")
        .def(nb::init<const cv::Matx33f &, const cv::Matx44f &>())
        .def(nb::init<const cv::Matx33f &, const cv::Matx33f &, const cv::Vec3f &>(), "K"_a.noconvert(), "R"_a.noconvert(), "t"_a.noconvert())
        .def(nb::init<>())
        .def("projectPoint", &Camera::projectPoint)
        .def("resize", &Camera::resize)
        .def("projectLine", [](Camera &self, const cv::Vec<float, 6> &line)
             { return self.projectLine(line); })
        .def("projectLine", [](Camera &self, const cv::Mat_<cv::Vec<float, 6>> &img_line)
             {
                 size_t height = img_line.rows;
                 size_t width = img_line.cols;
                 cv::Mat1f img_angle(height, width);

                 for (size_t y = 0; y < height; y++)
                 {
                     for (size_t x = 0; x < width; x++)
                     {
                         cv::Vec<float, 6> line = img_line(y, x);
                         img_angle(y, x) = self.projectLine(line);
                     }
                 }
                 return img_angle;
                 //
             })
        .def("transformPointWorld2Local", &Camera::transformPointWorld2Local)
        .def("transformPointLocal2World", &Camera::transformPointLocal2World)
        .def_ro("intrinsic_matrix", &Camera::intrinsic_matrix)
        .def_ro("extrinsic_matrix", &Camera::extrinsic_matrix)
        .def_ro("K", &Camera::intrinsic_matrix)
        .def_prop_ro(
            "R", [](Camera &self)
            {
                cv::Matx33f R;
                for (size_t j = 0; j < 3; j++)
                {
                    for (size_t i = 0; i < 3; i++)
                    {
                        R(j, i) = self.extrinsic_matrix(j, i);
                    }
                }
                return R;
                //
            },
            nb::rv_policy::move)
        .def_prop_ro(
            "t", [](Camera &self)
            {
                cv::Vec3f t;
                for (size_t j = 0; j < 3; j++)
                {
                    t(j) = self.extrinsic_matrix(j, 3);
                }
                return t;
                //
            },
            nb::rv_policy::move);

    nb::class_<SingleViewData>(m, "SingleViewData")
        .def(nb::init<>())
        .def(nb::init<size_t, size_t, const Camera &>())
        .def(nb::init<const cv::Mat1f &, const cv::Mat1f &, const cv::Mat1f &, const Camera &>())
        .def("getDirectionalPoint", [](SingleViewData &self)
             {
                 std::vector<DirectionalPoint> directional_point = self.getDirectionalPointCloud();
                 size_t num = directional_point.size();

                 float *points_data = new float[num * 3]();
                 float *directions_data = new float[num * 3]();
                 size_t shape[2] = {num, 3};

                 for (size_t i = 0; i < num; i++)
                 {
                     cv::Vec3f point = cv::Vec3f(directional_point[i].point);
                     cv::Vec3f direction = directional_point[i].direction;

                     points_data[i * 3 + 0] = point[0];
                     points_data[i * 3 + 1] = point[1];
                     points_data[i * 3 + 2] = point[2];

                     directions_data[i * 3 + 0] = direction[0];
                     directions_data[i * 3 + 1] = direction[1];
                     directions_data[i * 3 + 2] = direction[2];
                 }

                 // Delete 'data' when the 'owner' capsule expires
                 nb::capsule points_owner(points_data, [](void *p) noexcept
                                          { delete[] (float *)p; });

                 nb::capsule directions_owner(directions_data, [](void *p) noexcept
                                              { delete[] (float *)p; });

                 nb::ndarray<nb::numpy, float, nb::shape<nb::any, 2>> points(points_data, 2, shape, points_owner);
                 nb::ndarray<nb::numpy, float, nb::shape<nb::any, 2>> directions(directions_data, 2, shape, directions_owner);

                 return std::make_tuple(points, directions);
                 //
             })
        .def("set_random_line", &SingleViewData::set_random_line)
        .def("set_line", &SingleViewData::set_line)
        .def("release_line", [](SingleViewData &self)
             { self.img_line.release(); })
        .def("rescale", &SingleViewData::rescale)
        .def("size", [](SingleViewData &self)
             {
                 cv::Size size = self.size();
                 return std::make_tuple(size.width, size.height);
                 //
             })
        .def_rw("img_intensity", &SingleViewData::img_intensity)
        .def_rw("img_orientation2d", &SingleViewData::img_orientation2d)
        .def_rw("img_confidence", &SingleViewData::img_confidence)
        .def_rw("img_line", &SingleViewData::img_line)
        .def_prop_rw(
            "img_direction", [](SingleViewData &self)
            {
                cv::Mat3f img_direction;
                self.getDirection(img_direction);
                return img_direction;
                //
            },
            [](SingleViewData &self, const cv::Mat3f &img_direction)
            {
                cv::Mat1f img_depth;
                self.getDepth(img_depth);

                self.set_line(img_depth, img_direction);
                //
            },
            nb::rv_policy::move)
        .def_prop_rw(
            "img_depth", [](SingleViewData &self)
            {
                cv::Mat1f img_depth;
                self.getDepth(img_depth);
                return img_depth;
                //
            },
            [](SingleViewData &self, const cv::Mat1f &img_depth)
            {
                cv::Mat3f img_direction;
                self.getDirection(img_direction);

                self.set_line(img_depth, img_direction);
                //
            },
            nb::rv_policy::move)
        .def_rw("img_mask", &SingleViewData::img_mask)
        .def_rw("min_depth", &SingleViewData::min_depth)
        .def_rw("max_depth", &SingleViewData::max_depth)
        .def_rw("camera", &SingleViewData::camera);

    nb::class_<std::shared_ptr<SingleViewData>>(m, "SingleViewDataPtr");

    nb::class_<std::vector<std::shared_ptr<SingleViewData>>>(m, "MultiViewDataBase");

    nb::class_<MultiViewData, std::vector<std::shared_ptr<SingleViewData>>>(m, "MultiViewData")
        .def(nb::init<>())
        .def("resize", [](MultiViewData &self, size_t n)
             { self.resize(n); })
        .def("reserve", &MultiViewData::reserve)
        .def("get_neighbor", &MultiViewData::get_neighbor)
        .def("get_neighbor_index_vector", &MultiViewData::get_neighbor_index_vector, nb::arg("pos"), nb::arg("num"))
        .def(
            "__len__", [](const MultiViewData &self)
            { return self.size(); },
            nb::is_operator())
        .def(
            "__getitem__", [](const MultiViewData &self, size_t i)
            { return self[i]; },
            nb::is_operator())
        .def("__setitem__", [](MultiViewData &self, size_t i, std::shared_ptr<SingleViewData> view)
             { self[i] = view; })
        .def("append", [](MultiViewData &self, std::shared_ptr<SingleViewData> view)
             { self.push_back(view); })
        .def(
            "__iter__", [](MultiViewData &self)
            { return nb::make_iterator(nb::type<MultiViewData>(), "iterator", self.begin(), self.end()); },
            nb::keep_alive<0, 1>());

    m.def("propagate", &propagate, "Update the 3D line map via spatial propagation");

    m.def("refinement", &refinement, "Refine the 3D line map via random perturbation");

    m.def("pbrtcamera", &pbrtcamera, "Read camera from pbrt file");

    m.def("sample_point_along_line", [](float angle, const cv::Vec2f &center, float radius, size_t num)
          {
              std::vector<cv::Point2f> points = sample_point_along_line(angle, cv::Point(center), radius, num);
              std::vector<cv::Vec2f> points_vec2f;
              points_vec2f.reserve(num);
              for (size_t i = 0; i < num; i++)
              {
                  points_vec2f.emplace_back(points[i]);
              }
              return points_vec2f;
              //
          });

    m.def(
        "generate_orientation_map", [](const cv::Mat1f &img_src, size_t num = 36, float sigma = 2.0, float lambd = 6.0, float gamma = 0.25)
        {
            // Kernel generation function
            int ksize = 6 * std::ceil(sigma * std::max(1.0, 1.2 / gamma)) + 1;
            auto getKernel = [&](double theta) -> cv::Mat
            { return cv::getGaborKernel(cv::Size(ksize, ksize), sigma, theta, lambd, gamma); };

            // Calculate orientation map
            cv::Mat1f img_orientation2d, img_confidence;
            calcOrientation2d(img_src, img_orientation2d, img_confidence, getKernel, num);

            return std::make_tuple(img_orientation2d, img_confidence);
            //
        },
        "img_src"_a, "num"_a = 36, "sigma"_a = 2.0, "lambd"_a = 6.0, "gamma"_a = 0.25);

    // 3D Line Filtering
    m.def(
        "line_filtering", [](MultiViewData &multiviewdata, size_t num_neighbor = 6, size_t num_least_consisten_neigbor = 2, float thresh_position = 0.1f, float thresh_angle = 10.0f * M_PI / 180.0f, bool verbose = false)
        {
            for (size_t i = 0; i < multiviewdata.size(); i++)
            {
                if (verbose)
                {
                    std::cout << "\r"
                              << "3D Line filtering " << std::to_string(i) << "-th view" << std::flush;
                }

                auto reference_view = multiviewdata.at(i);
                auto neighbor_views = multiviewdata.get_neighbor(i, num_neighbor);
                size_t height = reference_view->height(), width = reference_view->width();
                // #pragma omp parallel for
                for (int iy = 0; iy < height; iy++)
                {
                    for (size_t ix = 0; ix < width; ix++)
                    {
                        // Regerence line and 3D position
                        cv::Point index_reference(ix, iy);
                        PluckerLine line_reference = reference_view->img_line(index_reference);
                        cv::Vec3f position_reference = reference_view->getPoint(index_reference);

                        // Each neighbor views
                        int sum_of_valid_line = 0;
                        std::vector<float> position_diff_vector;
                        std::vector<float> angle_diff_vector;

                        for (const auto &neighbor_view : neighbor_views)
                        {
                            // We project corresponding 3D line position into neighboring views and get 3D lines from the neighboring views.
                            cv::Point index_neighbor = cv::Point(neighbor_view->camera.projectPoint(position_reference));

                            bool is_valid_line = false;
                            cv::Rect rect(0, 0, neighbor_view->width(), neighbor_view->height());

                            if (index_neighbor.inside(rect))
                            {
                                // Neighbor line and 3D position
                                PluckerLine line_neighbor = neighbor_view->img_line(index_neighbor);
                                cv::Vec3f position_neighbor = neighbor_view->getPoint(cv::Point(index_neighbor));

                                // To check the consistency of the estimated lines, we compare 3D position and 3D line direction.
                                float position_diff = cv::norm(position_reference - position_neighbor);
                                float angle_diff = angle(line_reference, line_neighbor);
                                bool criteria_position = position_diff < thresh_position;
                                bool criteria_angle = angle_diff < thresh_angle;
                                is_valid_line = criteria_position && criteria_angle;
                            }

                            sum_of_valid_line += is_valid_line;
                        }

                        // We keep the reconstructed point from the reference pixel if it is consistent with at least two neighboring views.
                        bool keep_line = sum_of_valid_line >= num_least_consisten_neigbor;

                        reference_view->img_mask(index_reference) *= keep_line;
                    }
                }
            }

            if (verbose)
            {
                std::cout << std::endl;
            }

            //
        },
        "multiviewdata"_a, "num_neighbor"_a = 6, "num_least_consisten_neigbor"_a = 2, "thresh_position"_a = 0.1f, "thresh_angle"_a = 10.0f * M_PI / 180.0f, "verbose"_a = false);

    m.def(
        "eval_consisntency", [](std::shared_ptr<SingleViewData> &reference_view, std::shared_ptr<SingleViewData> &neighbor_view)
        {
            // This function returns the difference between the 3D position and the 3D line angle of the corresponding pixels in the reference view and the neighboring view.
            // If the corresponding pixel is not valid (i.e., outside the image or mask), the value is filled with NaN.

            size_t height = reference_view->height(), width = reference_view->width();

            cv::Mat1f img_position_diff(height, width);
            cv::Mat1f img_angle_diff(height, width);

            float nan = std::numeric_limits<float>::quiet_NaN();

            cv::Rect rect(0, 0, neighbor_view->width(), neighbor_view->height());

#pragma omp parallel for schedule(dynamic, 1)
            for (int iy = 0; iy < height; iy++)
            {
                for (size_t ix = 0; ix < width; ix++)
                {
                    if (reference_view->img_mask(iy, ix) == 0)
                    {
                        img_position_diff(iy, ix) = nan;
                        img_angle_diff(iy, ix) = nan;
                        continue;
                    }

                    // Regerence line and 3D position
                    cv::Point index_reference(ix, iy);
                    PluckerLine line_reference = reference_view->img_line(index_reference);
                    cv::Vec3f position_reference = reference_view->getPoint(index_reference);

                    // We project corresponding 3D line position into neighboring views and get 3D lines from the neighboring views.
                    cv::Point index_neighbor = cv::Point(neighbor_view->camera.projectPoint(position_reference));

                    if (!index_neighbor.inside(rect))
                    {
                        img_position_diff(iy, ix) = nan;
                        img_angle_diff(iy, ix) = nan;
                        continue;
                    }

                    if (neighbor_view->img_mask(index_neighbor) == 0)
                    {
                        img_position_diff(iy, ix) = nan;
                        img_angle_diff(iy, ix) = nan;
                        continue;
                    }

                    // Neighbor line and 3D position
                    PluckerLine line_neighbor = neighbor_view->img_line(index_neighbor);
                    cv::Vec3f position_neighbor = neighbor_view->getPoint(cv::Point(index_neighbor));

                    // To check the consistency of the estimated lines, we compare 3D position and 3D line direction.
                    float position_diff = cv::norm(position_reference - position_neighbor);
                    float angle_diff = angle(line_reference, line_neighbor);

                    img_position_diff(iy, ix) = position_diff;
                    img_angle_diff(iy, ix) = angle_diff;
                }
            }

            // return img_position_diff;
            return std::make_tuple(img_position_diff, img_angle_diff);
            //
        },
        nb::rv_policy::move);

    m.def(
        "eval_line_segment_partial", [](ndarray_f32_3d points1, ndarray_f32_3d directions1, ndarray_f32_3d points2, ndarray_f32_3d directions2, float thres_radius, float thres_angle)
        {
            size_t num1 = points1.shape(0);
            size_t num2 = points2.shape(0);
            float thres_radius2 = thres_radius * thres_radius;

            // Check the vector size
            if (num1 != directions1.shape(0) || num2 != directions2.shape(0))
            {
                throw std::runtime_error("The vector size of points and directions must be the same.");
            }

            // Check range of angle threshold
            if (thres_angle < 0.0f || thres_angle > M_PI / 2.0f)
            {
                throw std::runtime_error("The angle threshold must be in the range of [0, pi/2].");
            }

            // Allocate memory for the return value
            bool *is_valid_raw = new bool[num1];
            nb::capsule owner(is_valid_raw, [](void *p) noexcept
                              { delete[] (bool *)p; });
            nb::ndarray<nb::numpy, bool, nb::shape<nb::any>> is_valid(is_valid_raw, {num1}, owner);

#pragma omp parallel for schedule(dynamic, 1)
            for (int i = 0; i < num1; i++)
            {
                float x1 = points1(i, 0), y1 = points1(i, 1), z1 = points1(i, 2);
                cv::Vec3f d1 = cv::Vec3f(directions1(i, 0), directions1(i, 1), directions1(i, 2));
                cv::Vec3f ud1 = cv::normalize(d1);

                is_valid(i) = false;

                for (size_t j = 0; j < num2; j++)
                {
                    float x2 = points2(j, 0), y2 = points2(j, 1), z2 = points2(j, 2);
                    float radius2 = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2);
                    if (radius2 < thres_radius2)
                    {
                        cv::Vec3f d2 = cv::Vec3f(directions2(j, 0), directions2(j, 1), directions2(j, 2));
                        cv::Vec3f ud2 = cv::normalize(d2);

                        float angle = std::asin(cv::norm(ud1.cross(ud2)));
                        if (angle < thres_angle)
                        {
                            is_valid(i) = true;
                            break;
                        }
                    }
                }
            }

            return is_valid;
            //
        },
        "points1"_a, "directions1"_a, "points2"_a, "directions2"_a, "thres_position"_a, "thres_angle"_a);
}
