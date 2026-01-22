#pragma once

#include "tensorflow/core/framework/op_kernel.h"
#include "unsupported/Eigen/CXX11/Tensor"

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace tensorflow::functor
{
template <typename Device, typename T> struct FlashAttnFunctor
{
    void operator()(const Device &d,  typename TTypes<T, 3>::ConstTensor Q,
                    typename TTypes<T, 3>::ConstTensor K, typename TTypes<T, 3>::ConstTensor V,
                    typename TTypes<T, 3>::Tensor Out, typename TTypes<float, 2>::Tensor stats, 
                    bool causal_mask, float dropout_rate, float scale, uint64 seed, uint64 offset)
                     const;
};

template <typename Device, typename T>
struct FlashAttnGradFunctor
{
    void operator()(const Device &d,
                    typename TTypes<T, 3>::ConstTensor Q,
                    typename TTypes<T, 3>::ConstTensor K,
                    typename TTypes<T, 3>::ConstTensor V,
                    typename TTypes<T, 3>::ConstTensor Out,
                    typename TTypes<float, 2>::ConstTensor Stats,
                    typename TTypes<T, 3>::ConstTensor dO,
                    typename TTypes<T, 3>::Tensor dQ,
                    typename TTypes<T, 3>::Tensor dK,
                    typename TTypes<T, 3>::Tensor dV,
                    bool causal_mask, float dropout_rate, float scale,
                    uint64 seed, uint64 offset) const;
};

}; // namespace tensorflow::functor