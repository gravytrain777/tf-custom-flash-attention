#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "sdpa.h"
#include "sdpa_utils.cuh"
#include <algorithm>

// use both kv loop & q loop to avoid atomic add

namespace tensorflow::functor
{

template <typename T, int D_MAX>
__global__ void FlashAttnBwdKernel_KV(const T *__restrict__ Q, const T *__restrict__ K,
                                      const T *__restrict__ V, const T *__restrict__ Out,
                                      const float *__restrict__ Stats, const T *__restrict__ dO,
                                      T *__restrict__ dK, T *__restrict__ dV, int B, int S_q,
                                      int S_kv, int D_qk, int D_v, bool causal_mask,
                                      float dropout_rate, float scale, uint64 seed, uint64 offset)
{
    constexpr int Br = traits::FlashTraitsKV<D_MAX>::Br;
    constexpr int Bc = traits::FlashTraitsKV<D_MAX>::Bc;
    const float dp_inverse = (dropout_rate > EPSF) ? (1.0f / (1.0f - dropout_rate)) : 1.0f;

    extern __shared__ char smem_[];
    T *s_K = reinterpret_cast<T *>(smem_);
    T *s_V = s_K + Bc * D_MAX;
    T *s_Q = s_V + Bc * D_MAX;
    T *s_dO = s_Q + Br * D_MAX;
    float *s_Stats = reinterpret_cast<float *>(s_dO + Br * D_MAX); // Br * 1
    float *s_dOsum = s_Stats + Br;                                 // Br * 1

    const int tid = threadIdx.x; // kv id in column (0 ~ Bc)

    const int tiles_per_batch = (S_kv + Bc - 1) / Bc;
    const bool block_valid = blockIdx.x < tiles_per_batch * B;
    const int b = block_valid ? blockIdx.x / tiles_per_batch : 0;

    const int tile_idx_kv = blockIdx.x % tiles_per_batch;
    const int tile_offset_kv = tile_idx_kv * Bc;
    const int kv_offset = tile_offset_kv + tid;

    const bool active = block_valid && (tid < Bc) && (kv_offset < S_kv);
    const size_t batch_offset_q = static_cast<size_t>(b * S_q);
    const size_t batch_offset_kv = static_cast<size_t>(b * S_kv);
    const int64 global_rng_idx = seed + offset + kv_offset;

    // Local accumulators for the specific column(s) this thread handles
    float r_dK[D_MAX];
    float r_dV[D_MAX];
#pragma unroll
    for (int d = 0; d < D_MAX; ++d)
    {
        r_dK[d] = ZEROF;
        r_dV[d] = ZEROF;
    }

    // 1. Load K_j and V_j tile into Shared Memory
    if (block_valid)
    {
        load_tile<T, D_MAX>(s_K, K, tid, blockDim.x, Bc, D_qk, S_kv, tile_offset_kv, batch_offset_kv);
        load_tile<T, D_MAX>(s_V, V, tid, blockDim.x, Bc, D_v, S_kv, tile_offset_kv, batch_offset_kv);
    }

    T *k_ptr = s_K + tid * D_MAX;
    T *v_ptr = s_V + tid * D_MAX;

    // 2. Inner Loop: Iterate over Query blocks (FlashAttention-2 style)
    for (int tile_offset_q = 0; tile_offset_q < S_q; tile_offset_q += Br)
    {
        // Check if this entire query block is masked for this KV tile
        const bool tile_is_masked = (!block_valid) || (causal_mask && ((tile_offset_q + Br - 1) < tile_offset_kv));

        __syncthreads();

        if (!tile_is_masked)
        {
            // Load Q_i and dO_i into Shared Memory
            load_tile<T, D_MAX>(s_Q, Q, tid, blockDim.x, Br, D_qk, S_q, tile_offset_q, batch_offset_q);
            load_tile<T, D_MAX>(s_dO, dO, tid, blockDim.x, Br, D_v, S_q, tile_offset_q, batch_offset_q);
        }

        __syncthreads();

        if (!tile_is_masked)
        {
            // load stats & dOsum
            for (int r = tid; r < Br; r += blockDim.x)
            {
                const int q_offset = tile_offset_q + r;
                float dOsum_val = ZEROF;
                float stats_val = ZEROF;
                if (q_offset < S_q)
                {
                    const size_t idx = batch_offset_q + q_offset;
                    for (int d = 0; d < D_v; ++d)
                    {
                        dOsum_val += static_cast<float>(s_dO[r * D_MAX + d] * Out[idx * D_v + d]);
                    }
                    stats_val = Stats[idx];
                }
                s_dOsum[r] = dOsum_val;
                s_Stats[r] = stats_val;
            }
        }

        __syncthreads();

        if (active && !tile_is_masked)
        {
            // Process current Q-block row by row
            for (int i = 0; i < Br && (tile_offset_q + i) < S_q; ++i)
            {
                const int q_offset = tile_offset_q + i;
                const size_t q_idx = batch_offset_q + q_offset;

                // Current thread calculates contribution for column j = kv_offset
                const bool not_masked = !(causal_mask && (q_offset < kv_offset));

                if (not_masked)
                {
                    T *q_ptr = s_Q + i * D_MAX;
                    T *dO_ptr = s_dO + i * D_MAX;

                    float score = ZEROF;
                    for (int d = 0; d < D_qk; ++d)
                    {
                        score += static_cast<float>(q_ptr[d] * k_ptr[d]);
                    }
                    float p_ij = Eigen::numext::exp(score * scale - s_Stats[i]);
                    if (p_ij > EPSF)
                    {
                        float dS = scale;
                        bool is_dropped = false;
                        if (dropout_rate > EPSF)
                        {
                            if (get_random_float_v2(global_rng_idx, q_idx * S_kv) < dropout_rate)
                            {
                                is_dropped = true;
                                dS *= -p_ij * s_dOsum[i];
                                p_ij = ZEROF;
                            }
                            else
                            {
                                p_ij *= dp_inverse;
                            }
                        }

                        if (!is_dropped)
                        {
                            float dP = ZEROF;
                            for (int d = 0; d < D_v; ++d)
                            {
                                dP += static_cast<float>(dO_ptr[d] * v_ptr[d]);
                            }
                            dS *= p_ij * (dP - s_dOsum[i] / dp_inverse);
                        }

#pragma unroll
                        for (int d = 0; d < D_MAX; ++d)
                        {
                            if (d < D_v)
                            {
                                r_dV[d] += p_ij * static_cast<float>(dO_ptr[d]);
                            }
                            if (d < D_qk)
                            {
                                r_dK[d] += dS * static_cast<float>(q_ptr[d]);
                            }
                        }
                    }
                }
            }
        }
    }

    // 3. Final global write for dK and dV (One write per sequence element)
    if (active)
    {
        const size_t kv_idx = batch_offset_kv + kv_offset;
#pragma unroll
        for (int d = 0; d < D_MAX; ++d)
        {
            if (d < D_qk)
            {
                dK[kv_idx * D_qk + d] = static_cast<T>(r_dK[d]);
            }
            if (d < D_v)
            {
                dV[kv_idx * D_v + d] = static_cast<T>(r_dV[d]);
            }
        }
    }
}

template <typename T, int D_MAX>
__global__ void FlashAttnBwdKernel_Q(const T *__restrict__ Q, const T *__restrict__ K,
                                     const T *__restrict__ V, const T *__restrict__ Out,
                                     const float *__restrict__ Stats, const T *__restrict__ dO,
                                     T *__restrict__ dQ, int B, int S_q, int S_kv, int D_qk,
                                     int D_v, bool causal_mask, float dropout_rate, float scale,
                                     uint64 seed, uint64 offset)
{
    constexpr int Br = traits::FlashTraitsQ<D_MAX>::Br;
    constexpr int Bc = traits::FlashTraitsQ<D_MAX>::Bc;
    const float dp_inverse = (dropout_rate > EPSF) ? (1.0f / (1.0f - dropout_rate)) : 1.0f;

    // Shared Memory Layout:
    // Q_tile[Br][D_qk] | dO_tile[Br][D_v] | K_tile[Bc][D_qk] | V_tile[Bc][D_v]
    extern __shared__ char smem_[];
    T *s_Q = reinterpret_cast<T *>(smem_);
    T *s_dO = s_Q + Br * D_MAX;
    T *s_K = s_dO + Br * D_MAX;
    T *s_V = s_K + Bc * D_MAX;

    const int tid = threadIdx.x;
    const int tiles_per_batch = (S_q + Br - 1) / Br;
    const bool block_valid = blockIdx.x < tiles_per_batch * B;
    const int b = block_valid ? blockIdx.x / tiles_per_batch : 0;
    const int tile_idx_q = blockIdx.x % tiles_per_batch;
    const int tile_offset_q = tile_idx_q * Br;
    const int q_offset = tile_offset_q + tid;
    const bool active = block_valid && (tid < Br) && (q_offset < S_q);
    const size_t batch_offset_q = static_cast<size_t>(b * S_q);
    const size_t batch_offset_kv = static_cast<size_t>(b * S_kv);
    const size_t q_idx = batch_offset_q + q_offset;
    const uint64 global_rng_idx = seed + offset + static_cast<uint64>(q_idx * S_kv);

    // dQ accumulator
    float r_dQ[D_MAX];
#pragma unroll
    for (int d = 0; d < D_MAX; ++d)
    {
        r_dQ[d] = ZEROF;
    }

    // 1. Load Q_i and dO_i tile into Shared Memory
    if (block_valid)
    {
        load_tile<T, D_MAX>(s_Q, Q, tid, blockDim.x, Br, D_qk, S_q, tile_offset_q, batch_offset_q);
        load_tile<T, D_MAX>(s_dO, dO, tid, blockDim.x, Br, D_v, S_q, tile_offset_q, batch_offset_q);
    }

    T *q_ptr = s_Q + tid * D_MAX;
    T *dO_ptr = s_dO + tid * D_MAX;

    // 2. pre sum dOsum
    float dOsum = ZEROF;
    float stats_val = ZEROF;
    if (active)
    {
        stats_val = Stats[q_idx];
        const size_t out_offset = q_idx * D_v;
        for (int d = 0; d < D_v; ++d)
        {
            dOsum += static_cast<float>(dO_ptr[d] * Out[out_offset + d]);
        }
    }

    // 3. Inner Loop
    for (int tile_offset_kv = 0; tile_offset_kv < S_kv; tile_offset_kv += Bc)
    {
        const bool tile_is_masked = (!block_valid) || (causal_mask && (tile_offset_kv > tile_offset_q + Br - 1));

        __syncthreads();
        if (!tile_is_masked)
        {
            // Load K_j and V_j into Shared Memory
            load_tile<T, D_MAX>(s_K, K, tid, blockDim.x, Bc, D_qk, S_kv, tile_offset_kv, batch_offset_kv);
            load_tile<T, D_MAX>(s_V, V, tid, blockDim.x, Bc, D_v, S_kv, tile_offset_kv, batch_offset_kv);
        }

        __syncthreads();

        if (active && !tile_is_masked)
        {
            for (int j = 0; j < Bc && (tile_offset_kv + j) < S_kv; ++j)
            {
                const int kv_offset = tile_offset_kv + j;
                const bool not_masked = !(causal_mask && (q_offset < kv_offset));

                if (not_masked)
                {
                    T *k_ptr = s_K + j * D_MAX;

                    float score = ZEROF;
                    for (int d = 0; d < D_qk; ++d)
                    {
                        score += static_cast<float>(q_ptr[d] * k_ptr[d]);
                    }
                    float p_ij = Eigen::numext::exp(score * scale - stats_val);
                    if (p_ij > EPSF)
                    {
                        float dS = scale;
                        bool is_dropped = false;
                        if (dropout_rate > EPSF)
                        {
                            if (get_random_float_v2(global_rng_idx, kv_offset) < dropout_rate)
                            {
                                is_dropped = true;
                                dS *= -p_ij * dOsum;
                                p_ij = ZEROF;
                            }
                            else
                            {
                                p_ij *= dp_inverse;
                            }
                        }

                        if (!is_dropped)
                        {
                            T *v_ptr = s_V + j * D_MAX;
                            float dP = ZEROF;
                            for (int d = 0; d < D_v; ++d)
                            {
                                dP += static_cast<float>(dO_ptr[d] * v_ptr[d]);
                            }
                            dS *= p_ij * (dP - dOsum / dp_inverse);
                        }

#pragma unroll
                        for (int d = 0; d < D_MAX; ++d)
                        {
                            if (d < D_qk)
                                r_dQ[d] += dS * static_cast<float>(k_ptr[d]);
                        }
                    }
                }
            }
        }
    }

    if (active)
    {
        const size_t out_offset = q_idx * D_qk;
#pragma unroll
        for (int d = 0; d < D_MAX; ++d)
        {
            if (d < D_qk)
            {
                dQ[out_offset + d] = static_cast<T>(r_dQ[d]);
            }
        }
    }
}

template <typename T> struct FlashAttnGradFunctor<GPUDevice, T>
{
    void operator()(const GPUDevice &d, typename TTypes<T, 3>::ConstTensor Q,
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
        const int D_qk = Q.dimension(2);
        const int S_kv = K.dimension(1);
        const int D_v = V.dimension(2);

        int max_D = std::max(D_qk, D_v);
        if (max_D <= 8)
        {
            LaunchKernel<8>(d, Q, K, V, Out, Stats, dO, dQ, dK, dV, B, S_q, S_kv, D_qk, D_v,
                            causal_mask, dropout_rate, scale, seed, offset);
        }
        else if (max_D <= 16)
        {
            LaunchKernel<16>(d, Q, K, V, Out, Stats, dO, dQ, dK, dV, B, S_q, S_kv, D_qk, D_v,
                             causal_mask, dropout_rate, scale, seed, offset);
        }
        else if (max_D <= 32)
        {
            LaunchKernel<32>(d, Q, K, V, Out, Stats, dO, dQ, dK, dV, B, S_q, S_kv, D_qk, D_v,
                             causal_mask, dropout_rate, scale, seed, offset);
        }
        else if (max_D <= 64)
        {
            LaunchKernel<64>(d, Q, K, V, Out, Stats, dO, dQ, dK, dV, B, S_q, S_kv, D_qk, D_v,
                             causal_mask, dropout_rate, scale, seed, offset);
        }
        else if (max_D <= 128)
        {
            LaunchKernel<128>(d, Q, K, V, Out, Stats, dO, dQ, dK, dV, B, S_q, S_kv, D_qk, D_v,
                              causal_mask, dropout_rate, scale, seed, offset);
        }
    }

    template <int D_MAX>
    void LaunchKernel(const GPUDevice &d, typename TTypes<T, 3>::ConstTensor Q,
                      typename TTypes<T, 3>::ConstTensor K, typename TTypes<T, 3>::ConstTensor V,
                      typename TTypes<T, 3>::ConstTensor Out,
                      typename TTypes<float, 2>::ConstTensor Stats,
                      typename TTypes<T, 3>::ConstTensor dO, typename TTypes<T, 3>::Tensor dQ,
                      typename TTypes<T, 3>::Tensor dK, typename TTypes<T, 3>::Tensor dV, int B,
                      int S_q, int S_kv, int D_qk, int D_v, bool causal_mask, float dropout_rate,
                      float scale, uint64 seed, uint64 offset) const
    {
        LaunchKernelKV<D_MAX>(d, Q, K, V, Out, Stats, dO, dK, dV, B, S_q, S_kv, D_qk, D_v,
                              causal_mask, dropout_rate, scale, seed, offset);
        LaunchKernelQ<D_MAX>(d, Q, K, V, Out, Stats, dO, dQ, B, S_q, S_kv, D_qk, D_v, causal_mask,
                             dropout_rate, scale, seed, offset);
    }

    template <int D_MAX>
    void LaunchKernelKV(const GPUDevice &d, typename TTypes<T, 3>::ConstTensor Q,
                        typename TTypes<T, 3>::ConstTensor K, typename TTypes<T, 3>::ConstTensor V,
                        typename TTypes<T, 3>::ConstTensor Out,
                        typename TTypes<float, 2>::ConstTensor Stats,
                        typename TTypes<T, 3>::ConstTensor dO, typename TTypes<T, 3>::Tensor dK,
                        typename TTypes<T, 3>::Tensor dV, int B, int S_q, int S_kv, int D_qk,
                        int D_v, bool causal_mask, float dropout_rate, float scale, uint64 seed,
                        uint64 offset) const
    {
        constexpr int Br = traits::FlashTraitsKV<D_MAX>::Br;
        constexpr int Bc = traits::FlashTraitsKV<D_MAX>::Bc;
        constexpr int threads_per_block = traits::FlashTraitsKV<D_MAX>::Th;
        // Shared Memory size: Q, dO (Br rows) + K, V (Bc rows) + Stats, dOsum (Br floats)
        int smem_size = (2 * Br + 2 * Bc) * D_MAX * sizeof(T) + 2 * Br * sizeof(float);
        int blocks_per_batch = (S_kv + Bc - 1) / Bc;
        int total_blocks = B * blocks_per_batch;
        TF_CHECK_OK(GpuLaunchKernel(
            FlashAttnBwdKernel_KV<T, D_MAX>, total_blocks, threads_per_block, smem_size, d.stream(),
            Q.data(), K.data(), V.data(), Out.data(), Stats.data(), dO.data(), dK.data(), dV.data(),
            B, S_q, S_kv, D_qk, D_v, causal_mask, dropout_rate, scale, seed, offset));
    }

    template <int D_MAX>
    void LaunchKernelQ(const GPUDevice &d, typename TTypes<T, 3>::ConstTensor Q,
                       typename TTypes<T, 3>::ConstTensor K, typename TTypes<T, 3>::ConstTensor V,
                       typename TTypes<T, 3>::ConstTensor Out,
                       typename TTypes<float, 2>::ConstTensor Stats,
                       typename TTypes<T, 3>::ConstTensor dO, typename TTypes<T, 3>::Tensor dQ,
                       int B, int S_q, int S_kv, int D_qk, int D_v, bool causal_mask,
                       float dropout_rate, float scale, uint64 seed, uint64 offset) const
    {
        constexpr int Br = traits::FlashTraitsQ<D_MAX>::Br;
        constexpr int Bc = traits::FlashTraitsQ<D_MAX>::Bc;
        constexpr int threads_per_block = traits::FlashTraitsQ<D_MAX>::Th;

        // Shared Memory size: Q, dO (Br rows) + K, V (Bc rows) + Stats, dOsum (Br floats)
        int smem_size = (2 * Br + 2 * Bc) * D_MAX * sizeof(T);
        int blocks_per_batch = (S_q + Br - 1) / Br;
        int total_blocks = B * blocks_per_batch;
        TF_CHECK_OK(GpuLaunchKernel(FlashAttnBwdKernel_Q<T, D_MAX>, total_blocks, threads_per_block,
                                    smem_size, d.stream(), Q.data(), K.data(), V.data(), Out.data(),
                                    Stats.data(), dO.data(), dQ.data(), B, S_q, S_kv, D_qk, D_v,
                                    causal_mask, dropout_rate, scale, seed, offset));
    }
};

template struct FlashAttnGradFunctor<GPUDevice, float>;
template struct FlashAttnGradFunctor<GPUDevice, Eigen::half>;

} // namespace tensorflow::functor

#endif