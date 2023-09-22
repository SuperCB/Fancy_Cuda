
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>

#define MAX_BLOCK_SZ 128
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#define NUM_THREADS 128
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
typedef unsigned int uint;
template <typename T>
void check(T err, const char *const func, const char *const file,
           const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

size_t get_max_sharedmemory_size() {
  size_t mem_size = 0;
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  if (deviceCount == 0) {
    std::cerr << "No CUDA-capable devices found." << std::endl;
    return -1;
  }

  for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, i);

    std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
    std::cout << "Shared Memory Size: " << deviceProp.sharedMemPerBlock
              << " bytes" << std::endl;
    mem_size = deviceProp.sharedMemPerBlock;
    break;
  }
  return mem_size;
}
uint *allocate_memory(const unsigned int len) {
  unsigned int *elems;
  checkCudaErrors(cudaMalloc(&elems, sizeof(unsigned int) * len));
  checkCudaErrors(cudaMemset(elems, 0, sizeof(unsigned int) * len));
  return elems;
}
