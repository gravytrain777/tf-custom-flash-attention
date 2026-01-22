#define EIGEN_USE_THREADS

#include "sdpa.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace tensorflow
{

template <typename Device, typename T> class SdpaFwdOp : public OpKernel
{
private:
    float dropout_rate_;
    float scale_;
    bool causal_mask_;

public:
    explicit SdpaFwdOp(OpKernelConstruction *context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("dropout", &dropout_rate_));
        OP_REQUIRES_OK(context, context->GetAttr("scale", &scale_));
        OP_REQUIRES_OK(context, context->GetAttr("causal_mask", &causal_mask_));
    }

    void Compute(OpKernelContext *context) override
    {
        const Tensor &q_tensor = context->input(0);
        const Tensor &k_tensor = context->input(1);
        const Tensor &v_tensor = context->input(2);
        const Tensor &seed_tensor = context->input(3);
        const Tensor &offset_tensor = context->input(4);

        OP_REQUIRES(context, TensorShapeUtils::IsScalar(seed_tensor.shape()),
                    errors::InvalidArgument("seed must be scalar"));
        OP_REQUIRES(context, TensorShapeUtils::IsScalar(offset_tensor.shape()),
                    errors::InvalidArgument("offset must be scalar"));

        uint64 seed_val = seed_tensor.scalar<uint64>()();
        uint64 offset_val = offset_tensor.scalar<uint64>()();

        const int B = q_tensor.dim_size(0);
        const int S_q = q_tensor.dim_size(1);
        const int D_qk = q_tensor.dim_size(2);
        const int S_kv = k_tensor.dim_size(1);
        const int D_v = v_tensor.dim_size(2);

        OP_REQUIRES(context, k_tensor.dim_size(0) == B,
                    errors::InvalidArgument("K batch size mismatch"));
        OP_REQUIRES(context, k_tensor.dim_size(2) == D_qk,
                    errors::InvalidArgument("K feature size mismatch"));
        OP_REQUIRES(context, v_tensor.dim_size(0) == B,
                    errors::InvalidArgument("V batch size mismatch"));
        OP_REQUIRES(context, v_tensor.dim_size(1) == S_kv,
                    errors::InvalidArgument("V seq size mismatch"));

        if (causal_mask_)
        {
            OP_REQUIRES(context, S_q == S_kv,
                        errors::InvalidArgument("seq size Q != K while causal_mask = True"));
        }

        OP_REQUIRES(
            context, std::max(D_qk, D_v) <= 128,
            errors::InvalidArgument("CUDA kernel does not support feature size over 128. Set "
                                    "smaller feature size or use more heads"));

        Tensor *output_tensor = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, TensorShape({B, S_q, D_v}), &output_tensor));

        Tensor *stats_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({B, S_q}), &stats_tensor));

        auto Q = q_tensor.template tensor<T, 3>();
        auto K = k_tensor.template tensor<T, 3>();
        auto V = v_tensor.template tensor<T, 3>();
        auto Out = output_tensor->template tensor<T, 3>();
        auto Stats = stats_tensor->template tensor<float, 2>();

        const Device &d = context->template eigen_device<Device>();
        functor::FlashAttnFunctor<Device, T>()(d, Q, K, V, Out, Stats, causal_mask_, dropout_rate_,
                                               scale_, seed_val, offset_val);
    };
};

template <typename Device, typename T> class SdpaBwdOp : public OpKernel
{
private:
    float dropout_rate_;
    float scale_;
    bool causal_mask_;

public:
    explicit SdpaBwdOp(OpKernelConstruction *context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("dropout", &dropout_rate_));
        OP_REQUIRES_OK(context, context->GetAttr("scale", &scale_));
        OP_REQUIRES_OK(context, context->GetAttr("causal_mask", &causal_mask_));
    }

    void Compute(OpKernelContext *context) override
    {
        const Tensor &q_tensor = context->input(0);
        const Tensor &k_tensor = context->input(1);
        const Tensor &v_tensor = context->input(2);
        const Tensor &out_tensor = context->input(3);
        const Tensor &stats_tensor = context->input(4);
        const Tensor &do_tensor = context->input(5);
        const Tensor &seed_tensor = context->input(6);
        const Tensor &offset_tensor = context->input(7);

        OP_REQUIRES(context, TensorShapeUtils::IsScalar(seed_tensor.shape()),
                    errors::InvalidArgument("seed must be scalar"));
        OP_REQUIRES(context, TensorShapeUtils::IsScalar(offset_tensor.shape()),
                    errors::InvalidArgument("offset must be scalar"));

        uint64 seed_val = seed_tensor.scalar<uint64>()();
        uint64 offset_val = offset_tensor.scalar<uint64>()();

        const int B = q_tensor.dim_size(0);
        const int S_q = q_tensor.dim_size(1);
        const int D_qk = q_tensor.dim_size(2);
        const int S_kv = k_tensor.dim_size(1);
        const int D_v = v_tensor.dim_size(2);

        OP_REQUIRES(context, k_tensor.dim_size(0) == B,
                    errors::InvalidArgument("K batch size mismatch"));
        OP_REQUIRES(context, k_tensor.dim_size(2) == D_qk,
                    errors::InvalidArgument("K feature size mismatch"));

        OP_REQUIRES(context, v_tensor.dim_size(0) == B,
                    errors::InvalidArgument("V batch size mismatch"));
        OP_REQUIRES(context, v_tensor.dim_size(1) == S_kv,
                    errors::InvalidArgument("V seq size mismatch"));

        OP_REQUIRES(context, out_tensor.dim_size(0) == B,
                    errors::InvalidArgument("O batch size mismatch"));
        OP_REQUIRES(context, out_tensor.dim_size(1) == S_q,
                    errors::InvalidArgument("O seq size mismatch"));
        OP_REQUIRES(context, out_tensor.dim_size(2) == D_v,
                    errors::InvalidArgument("O feature size mismatch"));

        OP_REQUIRES(context, do_tensor.dim_size(0) == B,
                    errors::InvalidArgument("dO batch size mismatch"));
        OP_REQUIRES(context, do_tensor.dim_size(1) == S_q,
                    errors::InvalidArgument("dO seq size mismatch"));
        OP_REQUIRES(context, do_tensor.dim_size(2) == D_v,
                    errors::InvalidArgument("dO feature size mismatch"));

        OP_REQUIRES(context, stats_tensor.dim_size(0) == B,
                    errors::InvalidArgument("S batch size mismatch"));
        OP_REQUIRES(context, stats_tensor.dim_size(1) == S_q,
                    errors::InvalidArgument("S seq size mismatch"));

        if (causal_mask_)
        {
            OP_REQUIRES(context, S_q == S_kv,
                        errors::InvalidArgument("seq size Q != K while causal_mask = True"));
        }

        OP_REQUIRES(
            context, std::max(D_qk, D_v) <= 128,
            errors::InvalidArgument("CUDA kernel does not support feature size over 128. Set "
                                    "smaller feature size or use more heads"));

        Tensor *dq_tensor = nullptr;
        Tensor *dk_tensor = nullptr;
        Tensor *dv_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, q_tensor.shape(), &dq_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(1, k_tensor.shape(), &dk_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(2, v_tensor.shape(), &dv_tensor));

        auto Q = q_tensor.template tensor<T, 3>();
        auto K = k_tensor.template tensor<T, 3>();
        auto V = v_tensor.template tensor<T, 3>();
        auto Out = out_tensor.template tensor<T, 3>();
        auto dO = do_tensor.template tensor<T, 3>();
        auto Stats = stats_tensor.template tensor<float, 2>();

        auto dQ = dq_tensor->template tensor<T, 3>();
        auto dK = dk_tensor->template tensor<T, 3>();
        auto dV = dv_tensor->template tensor<T, 3>();

        const Device &d = context->template eigen_device<Device>();
        functor::FlashAttnGradFunctor<Device, T>()(d, Q, K, V, Out, Stats, dO, dQ, dK, dV,
                                                   causal_mask_, dropout_rate_, scale_, seed_val,
                                                   offset_val);
    };
};

namespace functor
{
// cpu implemetation
float get_random_float_cpu(uint64 seed, uint64 offset, int64 b, int i, int j, int stride_m,
                           int stride_n)
{
    uint64 batch_stride = (uint64)stride_m * (uint64)stride_n;
    uint64 idx = (uint64)b * batch_stride + (uint64)i * (uint64)stride_n + (uint64)j;

    uint64 global_idx = offset + idx;

    uint64 state = seed + global_idx;
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;

    uint32 val = (state >> 32) & 0xFFFFFFFF;
    return static_cast<float>(val) / 4294967296.0f;
}

template <typename T> struct FlashAttnFunctor<CPUDevice, T>
{
    void operator()(const CPUDevice &d, typename TTypes<T, 3>::ConstTensor Q,
                    typename TTypes<T, 3>::ConstTensor K, typename TTypes<T, 3>::ConstTensor V,
                    typename TTypes<T, 3>::Tensor Out, typename TTypes<float, 2>::Tensor stats,
                    bool causal_mask, float dropout_rate, float scale, uint64 seed,
                    uint64 offset) const
    {
        const int B = Q.dimension(0);
        const int S_q = Q.dimension(1);
        const int S_kv = K.dimension(1);
        const int D_qk = Q.dimension(2);
        const int D_v = V.dimension(2);

        const float neg_inf = -1e9f;
        const float zero = 0.0f;
        const float one = 1.0f;
        const float eps = 1e-6f;
        float dp_inverse_scale = one;
        if (dropout_rate > 0.0f)
        {
            dp_inverse_scale = one / (one - dropout_rate);
        }

        auto shard = [&](int64 start, int64 end)
        {
            for (int64 b = start; b < end; ++b)
            {
                // --- Loop Query (i) ---
                for (int i = 0; i < S_q; ++i)
                {
                    float m_i = neg_inf;
                    float l_i = zero;

                    std::vector<float> O_accum(D_v, zero);

                    // --- Loop Keys (j) ---
                    for (int j = 0; j < S_kv; ++j)
                    {

                        if (causal_mask && (j > i))
                        {
                            continue;
                        }

                        float score = zero;
                        for (int k = 0; k < D_qk; ++k)
                        {
                            score += static_cast<float>(Q(b, i, k) * K(b, j, k));
                        }
                        score *= scale;

                        // Online Softmax
                        float m_prev = m_i;
                        m_i = (score > m_prev) ? score : m_prev;

                        float ratio_prev = exp(m_prev - m_i);
                        float ratio_curr_l = exp(score - m_i);
                        float ratio_curr_o = ratio_curr_l;

                        // Dropout
                        if (dropout_rate > 0.0f)
                        {
                            if (get_random_float_cpu(seed, offset, b, i, j, S_q, S_kv) <
                                dropout_rate)
                            {
                                ratio_curr_o = zero;
                            }
                            else
                            {
                                ratio_curr_o *= dp_inverse_scale;
                            }
                        }

                        l_i = l_i * ratio_prev + ratio_curr_l;

                        // update Accumulator (O_accum)

                        for (int k = 0; k < D_v; ++k)
                        {
                            O_accum[k] *= ratio_prev;

                            float v_val = static_cast<float>(V(b, j, k));
                            O_accum[k] += v_val * ratio_curr_o;
                        }

                    } // End Loop j

                    // Write Output
                    stats(b, i) = m_i + Eigen::numext::log(l_i + eps);

                    float inv_l = (l_i > eps) ? (one / l_i) : zero;

                    for (int k = 0; k < D_v; ++k)
                    {
                        Out(b, i, k) = static_cast<T>(O_accum[k] * inv_l);
                    }

                } // End Loop i
            } // End Loop b
        };

        // estimate cost
        double compute_cycles = static_cast<double>(S_q) * S_kv * (2.0 * D_qk + 2.0 * D_v + 10.0);
        Eigen::TensorOpCost cost(0, 0, compute_cycles);
        d.parallelFor(B, cost, shard);
    }
};

template <typename T> struct FlashAttnGradFunctor<CPUDevice, T>
{
    void operator()(const CPUDevice &d, typename TTypes<T, 3>::ConstTensor Q,
                    typename TTypes<T, 3>::ConstTensor K, typename TTypes<T, 3>::ConstTensor V,
                    typename TTypes<T, 3>::ConstTensor Out,
                    typename TTypes<float, 2>::ConstTensor Stats,
                    typename TTypes<T, 3>::ConstTensor dO, typename TTypes<T, 3>::Tensor dQ,
                    typename TTypes<T, 3>::Tensor dK, typename TTypes<T, 3>::Tensor dV,
                    bool causal_mask, float dropout_rate, float scale, uint64 seed,
                    uint64 offset) const
    {
        const int B = Q.dimension(0);
        const int S_q = Q.dimension(1);
        const int S_kv = K.dimension(1);
        const int D_k = Q.dimension(2);
        const int D_v = V.dimension(2);

        const float zero = 0.0f;
        const float one = 1.0f;
        const float eps = 1e-6f;

        float dp_inverse_scale = one;
        if (dropout_rate > 0.0f)
        {
            dp_inverse_scale = one / (one - dropout_rate);
        }

        dQ.setZero();
        dK.setZero();
        dV.setZero();

        auto shard = [&](int64 start, int64 end)
        {
            for (int64 b = start; b < end; ++b)
            {
                for (int i = 0; i < S_q; ++i)
                {
                    float stats_val = Stats(b, i);
                    float D_i = zero;
                    for (int k = 0; k < D_v; ++k)
                    {
                        D_i += static_cast<float>(dO(b, i, k) * Out(b, i, k));
                    }

                    std::vector<float> dQ_accum(D_k, zero);
                    for (int j = 0; j < S_kv; ++j)
                    {
                        if (causal_mask && (j > i))
                        {
                            continue;
                        }

                        // (a) Q * K^T
                        float score = zero;
                        for (int k = 0; k < D_k; ++k)
                        {
                            score += static_cast<float>(Q(b, i, k) * K(b, j, k));
                        }
                        score *= scale;

                        // (b) Softmax
                        float prob = std::exp(score - stats_val);
                        if (prob < eps)
                            continue;

                        float prob_raw = prob;
                        bool is_dropped(false);

                        // (c) Dropout
                        if (dropout_rate > 0.0f)
                        {
                            if (get_random_float_cpu(seed, offset, b, i, j, S_q, S_kv) <
                                dropout_rate)
                            {
                                is_dropped = true;
                                prob = zero;
                            }
                            else
                            {
                                prob *= dp_inverse_scale;
                            }
                        }

                        float dS_val = zero;
                        if (is_dropped)
                        {
                            dS_val = -prob_raw * D_i * scale;
                        }
                        else
                        {
                            // 1. dV (Accumulate)
                            for (int k = 0; k < D_v; ++k)
                            {
                                float val_dO = static_cast<float>(dO(b, i, k));
                                dV(b, j, k) =
                                    static_cast<T>(static_cast<float>(dV(b, j, k)) + prob * val_dO);
                            }

                            // 2. dP_val = dO_i . V_j
                            float dP_val = zero;
                            for (int k = 0; k < D_v; ++k)
                            {
                                dP_val += static_cast<float>(dO(b, i, k) * V(b, j, k));
                            }

                            // 3. dS = P_scaled * (dP - D / sigma) * scale
                            dS_val = prob * (dP_val - D_i / dp_inverse_scale) * scale;
                        }

                        //  Update dK & Accumulate dQ (Merged) ---
                        for (int k = 0; k < D_k; ++k)
                        {
                            float val_Q = static_cast<float>(Q(b, i, k));
                            float val_K = static_cast<float>(K(b, j, k));

                            dK(b, j, k) =
                                static_cast<T>(static_cast<float>(dK(b, j, k)) + dS_val * val_Q);

                            dQ_accum[k] += dS_val * val_K;
                        }

                    } // End Loop j

                    for (int k = 0; k < D_k; ++k)
                    {
                        dQ(b, i, k) = static_cast<T>(dQ_accum[k]);
                    }
                } // End Loop i
            } // End Loop b
        };

        double compute_cycles = static_cast<double>(S_q) * S_kv * (4.0 * D_k + 4.0 * D_v + 30.0);
        Eigen::TensorOpCost cost(0, 0, compute_cycles);
        d.parallelFor(B, cost, shard);
    }
};
} // namespace functor

#define REGISTER_CPU_KERNEL_F(type)                                                                \
    REGISTER_KERNEL_BUILDER(Name("Sdpa").Device(DEVICE_CPU).TypeConstraint<type>("T"),             \
                            SdpaFwdOp<CPUDevice, type>);                                           \
    REGISTER_KERNEL_BUILDER(Name("SdpaGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"),         \
                            SdpaBwdOp<CPUDevice, type>);

REGISTER_CPU_KERNEL_F(float);
REGISTER_CPU_KERNEL_F(Eigen::half);
#undef REGISTER_CPU_KERNEL_F

#if GOOGLE_CUDA
#define REGISTER_GPU_KERNEL_F(type)                                                                \
    REGISTER_KERNEL_BUILDER(Name("Sdpa")                                                           \
                                .Device(DEVICE_GPU)                                                \
                                .TypeConstraint<type>("T")                                         \
                                .HostMemory("seed")                                                \
                                .HostMemory("offset"),                                             \
                            SdpaFwdOp<GPUDevice, type>);                                           \
    REGISTER_KERNEL_BUILDER(Name("SdpaGrad")                                                       \
                                .Device(DEVICE_GPU)                                                \
                                .TypeConstraint<type>("T")                                         \
                                .HostMemory("seed")                                                \
                                .HostMemory("offset"),                                             \
                            SdpaBwdOp<GPUDevice, type>);

REGISTER_GPU_KERNEL_F(float);
REGISTER_GPU_KERNEL_F(Eigen::half);
#undef REGISTER_GPU_KERNEL_F
#endif

} // namespace tensorflow