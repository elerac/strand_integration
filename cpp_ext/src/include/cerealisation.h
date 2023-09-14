#pragma once
#include <memory>
#include <cereal/cereal.hpp>
#include <opencv2/core.hpp>
#include "camera.h"
#include "dataframe.h"

namespace cereal
{
	// Reference of serializing cv::Mat
	// https://www.patrikhuber.ch/blog/2015/05/serialising-opencv-matrices-using-boost-and-cereal/
	template <class Archive>
	void save(Archive &archive, const cv::Mat &mat)
	{
		int rows, cols, type;
		bool continuous;

		rows = mat.rows;
		cols = mat.cols;
		type = mat.type();
		continuous = mat.isContinuous();

		archive &rows &cols &type &continuous;

		if (continuous)
		{
			const int data_size = rows * cols * static_cast<int>(mat.elemSize());
			auto mat_data = cereal::binary_data(mat.ptr(), data_size);
			archive &mat_data;
		}
		else
		{
			const int row_size = cols * static_cast<int>(mat.elemSize());
			for (int i = 0; i < rows; i++)
			{
				auto row_data = cereal::binary_data(mat.ptr(i), row_size);
				archive &row_data;
			}
		}
	};

	template <class Archive>
	void load(Archive &archive, cv::Mat &mat)
	{
		int rows, cols, type;
		bool continuous;

		archive &rows &cols &type &continuous;

		if (continuous)
		{
			mat.create(rows, cols, type);
			const int data_size = rows * cols * static_cast<int>(mat.elemSize());
			auto mat_data = cereal::binary_data(mat.ptr(), data_size);
			archive &mat_data;
		}
		else
		{
			mat.create(rows, cols, type);
			const int row_size = cols * static_cast<int>(mat.elemSize());
			for (int i = 0; i < rows; i++)
			{
				auto row_data = cereal::binary_data(mat.ptr(i), row_size);
				archive &row_data;
			}
		}
	};

	template <class Archive, typename _Tp, int m, int n>
	void serialize(Archive &archive, cv::Matx<_Tp, m, n> &matx)
	{
		const int size = m * n;
		for (int i = 0; i < size; i++)
		{
			archive &matx.val[i];
		}
	}

	template <class Archive, typename _Tp>
	void serialize(Archive &archive, cv::Size_<_Tp> &size)
	{
		archive &size.width;
		archive &size.height;
	}

	template <class Archive, typename _Tp>
	void serialize(Archive &archive, cv::Point_<_Tp> &pt)
	{
		archive &pt.x;
		archive &pt.y;
	}

	template <class Archive, typename _Tp>
	void serialize(Archive &archive, cv::Point3_<_Tp> &pt)
	{
		archive &pt.x;
		archive &pt.y;
		archive &pt.z;
	}

	template <class Archive>
	void serialize(Archive &archive, Camera &camera)
	{
		archive &camera.intrinsic_matrix;
		archive &camera.extrinsic_matrix;
		camera.initialize();
	}

	template <class Archive>
	void serialize(Archive &archive, SingleViewData &view)
	{
		archive &view.img_intensity;
		archive &view.img_orientation2d;
		archive &view.img_confidence;

		archive &view.camera;

		archive &view.img_line;
		archive &view.img_mask;

		archive &view.min_depth;
		archive &view.max_depth;
	}
} /* namespace */

