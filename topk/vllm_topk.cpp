
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cstdio>
#include <vector>
#include "ATen/core/TensorBody.h"
void top_k(const torch::Tensor src, const std::vector<int> &top_ks);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("top_k", &top_k, "Apply vllm top_k");
}
