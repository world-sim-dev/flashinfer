#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <flashinfer/div_clamp_to.cuh>

#include "flashinfer_ops.h"
#include "pytorch_extension_utils.h"

using namespace flashinfer;

void div_clamp_to(torch::Tensor& output, 
                  const torch::Tensor& input,
                  const torch::Tensor& scale) {
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(scale.is_cuda(), "scale must be a CUDA tensor");
    TORCH_CHECK(scale.dim() == 1, "Expected 1D tensor for scale");
    TORCH_CHECK(input.size(-1) == scale.size(0), "hidden_size must match");
    TORCH_CHECK(output.scalar_type() == torch::kFloat8_e4m3fn || output.scalar_type() == torch::kFloat8_e5m2,
                "output must be Float8_e4m3fn or Float8_e5m2");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 || input.scalar_type() == torch::kHalf,
                "input must be Float8_e4m3fn or Float8_e5m2");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat,
                "scale must be BFloat16 or Half");

    int hidden_size = input.size(-1);
    int64_t num_tokens = input.numel() / input.size(-1);
    dim3 grid(num_tokens);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), input_type, [&] {
        DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP8(output.scalar_type(), output_type, [&] {
            uint32_t vec_size = 16 / sizeof(input_type);
            dim3 block(std::min(hidden_size / vec_size, 1024U));

            flashinfer::customfn::div_clamp_to<output_type, input_type><<<grid, block, 0, stream>>>(
                static_cast<output_type*>(output.data_ptr()),
                static_cast<input_type*>(input.data_ptr()), 
                static_cast<float*>(scale.data_ptr()),
                hidden_size
            );
            return true;
        });
        return true;     
    });
}