#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <opencv2/core.hpp>

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

// Type caster for cv::Mat_<_Tp>
template <typename T>
struct type_caster<T, enable_if_t<is_base_of_template_v<T, cv::Mat_>>>
{
    // Info about cv::Mat_<_Tp>
    using _Tp = typename T::value_type;                            // cv::Vec or scalar type
    static constexpr int channels = cv::DataType<_Tp>::channels;   // number of channels
    using channel_type = typename cv::DataType<_Tp>::channel_type; // channel type
    static constexpr bool is_scalar = channels == 1;               // true if _Tp is scalar type
    static constexpr size_t ndim = is_scalar ? 2 : 3;              // number of dimensions

    // Define ndarray
    using ndarray_shape = std::conditional_t<is_scalar, shape<any, any>, shape<any, any, channels>>;
    using NDArray = ndarray<channel_type, numpy, ndarray_shape>;
    using NDArrayCaster = type_caster<NDArray>;

    NB_TYPE_CASTER(T, NDArrayCaster::Name);

    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept
    {
        // Check if src is ndarray
        NDArrayCaster caster;
        bool is_valid_ndarray = caster.from_python(src, flags, cleanup);
        if (!is_valid_ndarray)
        {
            return false;
        }

        // Convert ndarray to cv::Mat_<_Tp>
        const NDArray &array = caster.value;
        int rows = array.shape(0);
        int cols = array.shape(1);

        value.create(rows, cols);
        memcpy(value.data, array.data(), rows * cols * channels * sizeof(channel_type));

        return true;
    }

    static handle from_cpp(const cv::Mat_<_Tp> &mat, rv_policy policy, cleanup_list *cleanup) noexcept
    {
        size_t shape[ndim];

        if constexpr (is_scalar)
        {
            shape[0] = (size_t)mat.rows;
            shape[1] = (size_t)mat.cols;
        }
        else
        {
            shape[0] = (size_t)mat.rows;
            shape[1] = (size_t)mat.cols;
            shape[2] = (size_t)channels;
        }

        void *ptr = (void *)mat.data;

        switch (policy)
        {
        case rv_policy::automatic:
            policy = rv_policy::copy;
            break;

        case rv_policy::automatic_reference:
            policy = rv_policy::reference;
            break;

        default: // leave policy unchanged
            break;
        }

        object owner;
        if (policy == rv_policy::move)
        {
            T *temp = new T(std::move(mat));
            owner = capsule(temp, [](void *p) noexcept
                            { delete (T *)p; });
            ptr = temp->data;
        }

        rv_policy array_rv_policy = policy == rv_policy::move ? rv_policy::reference : policy;

        // Convert cv::Mat_<_Tp> to ndarray
        object o = steal(NDArrayCaster::from_cpp(NDArray(ptr, ndim, shape), policy, cleanup));
        return o.release();
    }
};

template <typename T>
using is_vec_or_matx = std::disjunction<std::is_same<T, cv::Vec<typename T::value_type, T::channels>>, std::is_same<T, cv::Matx<typename T::value_type, T::rows, T::cols>>>;

// Type caster for cv::Vec or cv::Matx
template <typename T>
struct type_caster<T, enable_if_t<is_vec_or_matx<T>::value>>
{

    // Info about cv::Vec or cv::Matx
    using _Tp = typename T::value_type;  // scalar type
    static constexpr int rows = T::rows; // number of rows
    static constexpr int cols = T::cols; // number of columns (cn)
    static constexpr bool is_matx = std::is_same_v<T, cv::Matx<typename T::value_type, T::rows, T::cols>>;
    static constexpr size_t ndim = is_matx ? 2 : 1; // number of dimensions

    // Define ndarray
    using ndarray_shape = std::conditional_t<is_matx, shape<rows, cols>, shape<rows>>;
    using NDArray = ndarray<_Tp, numpy, ndarray_shape>;
    using NDArrayCaster = type_caster<NDArray>;

    NB_TYPE_CASTER(T, NDArrayCaster::Name);

    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept
    {
        // Check if src is ndarray
        NDArrayCaster caster;
        bool is_valid_ndarray = caster.from_python(src, flags, cleanup);
        if (!is_valid_ndarray)
        {
            return false;
        }

        // Convert ndarray to cv::Vec or cv::Matx
        const NDArray &array = caster.value;
        memcpy(value.val, array.data(), rows * cols * sizeof(_Tp));

        return true;
    }

    static handle from_cpp(const T &matx, rv_policy policy, cleanup_list *cleanup) noexcept
    {
        size_t shape[ndim];

        if constexpr (is_matx)
        {
            shape[0] = (size_t)rows;
            shape[1] = (size_t)cols;
        }
        else
        {
            shape[0] = (size_t)rows;
        }

        void *ptr = (void *)matx.val;

        switch (policy)
        {
        case rv_policy::automatic:
            policy = rv_policy::copy;
            break;

        case rv_policy::automatic_reference:
            policy = rv_policy::reference;
            break;

        default: // leave policy unchanged
            break;
        }

        object owner;
        if (policy == rv_policy::move)
        {
            T *temp = new T(std::move(matx));
            owner = capsule(temp, [](void *p) noexcept
                            { delete (T *)p; });
            ptr = temp->val;
        }

        rv_policy array_rv_policy = policy == rv_policy::move ? rv_policy::reference : policy;

        // Convert cv::Vec or cv::Matx to ndarray
        object o = steal(NDArrayCaster::from_cpp(NDArray(ptr, ndim, shape), policy, cleanup));
        return o.release();
    }
};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
