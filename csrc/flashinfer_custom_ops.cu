#include "pytorch_extension_utils.h"

void div_clamp_to(at::Tensor& out, const at::Tensor& input, const at::Tensor& scale,
                  int64_t cuda_stream);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("div_clamp_to", &div_clamp_to, "Div clamp to"); }
