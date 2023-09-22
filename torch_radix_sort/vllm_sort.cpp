#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include "ATen/core/TensorBody.h"

#define CHECK_DEVICE(x) \
  TORCH_CHECK(x.device().type() == torch::kCUDA, #x " must be on CUDA")
#define CHECK_SHAPE(x, ...)                                   \
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), \
              #x " must have shape (" #__VA_ARGS__ ")")

void vllm_sort_cuda(float *src, const uint batch_size, const uint dim_len);
// (const torch::Tensor x1, const torch::Tensor x2, const torch::Tensor cos,
//  const torch::Tensor sin, torch::Tensor out1, torch::Tensor out2,
// const bool conj)
torch::Tensor vllm_sort(const torch::Tensor src, const unsigned int dim) {
  float *data = reinterpret_cast<float *>(src.data_ptr());
  auto shape = src.sizes();
  auto shape_len = shape.size();
  unsigned int batch_size = src.sizes()[0];
  uint len = shape[shape_len - 1];
  vllm_sort_cuda(data, batch_size, len);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vllm_sort", &vllm_sort, "Applty vllm sort");
}
