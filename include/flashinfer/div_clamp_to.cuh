#ifndef FLASHINFER_ACTIVATION_CUH_
#define FLASHINFER_ACTIVATION_CUH_

#include "math.cuh"
#include "utils.cuh"
#include "vec_dtypes.cuh"

namespace flashinfer {

namespace customfn {


template <typename Fp8T, typename InputT>
struct Fp8Traits;


template <typename InputT>
struct Fp8Traits<__nv_fp8_e4m3, InputT> {
    static __device__ __forceinline__ constexpr InputT min_value() {
        return -448.0;
    }
    static __device__ __forceinline__ constexpr InputT max_value() {
        return 448.0;
    }
};


template <typename InputT>
struct Fp8Traits<__nv_fp8_e5m2, InputT> {
    static __device__ __forceinline__ constexpr InputT min_value() {
        return -57344.0;
    }
    static __device__ __forceinline__ constexpr InputT max_value() {
        return 57344.0;
    }
};


template <typename ClampT, typename InputT>
__device__ __forceinline__ InputT div_and_clamp(const InputT& x, const InputT& s) {
  const InputT out = x / s;
  return std::min(std::max(out, Fp8Traits<ClampT, InputT>::min_value()), Fp8Traits<ClampT, InputT>::max_value());
}


template <typename OutputT, typename InputT>
__global__ void div_clamp_to(OutputT* __restrict__ output, 
                             const InputT* __restrict__ input, 
                             const float* __restrict__ scale, 
                             const int hidden_size) {
  constexpr uint32_t vec_size = 16 / sizeof(InputT);
  const int64_t token_idx = blockIdx.x;
  const int64_t thread_idx = threadIdx.x;
  const int64_t stride = blockDim.x;
  const int64_t offset = token_idx * hidden_size;

#pragma unroll 1
  for (uint32_t idx = thread_idx; idx < hidden_size / vec_size; idx += stride) {
    vec_t<float, vec_size> input_vec, scale_vec, output_vec;
    input_vec.cast_load(input + offset + idx * vec_size);
    scale_vec.cast_load(scale + idx * vec_size);
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
        output_vec[i] = div_and_clamp<OutputT, float>(input_vec[i], scale_vec[i]);
    }
    output_vec.cast_store(output + offset + idx * vec_size);
  }

  const int64_t remaining_offset = hidden_size - hidden_size % (stride * vec_size);
  // process the remaining elements
#pragma unroll 1
  for (int64_t idx = thread_idx; idx < hidden_size % (stride * vec_size); idx += stride) {
    float x = float(input[offset + remaining_offset + idx]);
    float s = scale[remaining_offset + idx];
    output[offset + remaining_offset + idx] = OutputT(div_and_clamp<OutputT, float>(x, s));
  }
}


}  // namespace customfn
}  // namespace flashinfer

#endif  // FLASHINFER_ACTIVATION_CUH_
