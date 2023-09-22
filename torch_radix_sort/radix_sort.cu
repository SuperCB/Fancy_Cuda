#include <iostream>
#include "utils.h"
typedef unsigned int uint;

void sum_scan_blelloch(uint *const dst, const uint *const src,
                       const uint batch_size, const uint dim_len);
__global__ void gpu_add_block_sums(uint *const output, const uint *const input,
                                   uint *const g_block_sums,
                                   const uint batch_size,
                                   const uint global_block_sum_len) {
  uint d_block_sum_val = g_block_sums[blockIdx.x * blockDim.y + blockIdx.y];
  uint base_index = blockIdx.x * global_block_sum_len;

  uint index = 2 * blockIdx.y * blockDim.y + threadIdx.x;
  if (index < global_block_sum_len) {
    output[index + base_index] = input[index + base_index] + d_block_sum_val;
    if (index + blockDim.x < global_block_sum_len)
      output[index + base_index + blockDim.x] =
          input[index + base_index + blockDim.x] + d_block_sum_val;
  }
}

__global__ void gpu_radix_sort_local(uint *out_sorted, const uint *input,
                                     uint *const g_prefix_sums,
                                     uint *const g_block_sums,
                                     const uint input_shift_width,
                                     const uint input_len,
                                     const uint block_size) {
  extern __shared__ uint shmem[];
  uint *s_data = shmem;
  // s_mask_out[] will be scanned in place
  uint s_mask_out_len = block_size + 1;
  uint *s_mask_out = &s_data[block_size];
  uint *s_merged_scan_mask_out = &s_mask_out[s_mask_out_len];
  uint *s_mask_out_sums = &s_merged_scan_mask_out[block_size];
  uint *scan_mask_out_sums = &s_mask_out_sums[4];

  uint threadId = threadIdx.x;
  // which batch begin with
  uint base_index = blockIdx.x * input_len;

  uint index = block_size * blockIdx.x + threadId;
  if (index < input_len) {
    s_data[threadId] = input[index + base_index];
  } else {
    s_data[threadId] = 0;
  }

  __syncthreads();

  uint t_data = s_data[threadId];
  uint bit_extract = (t_data >> input_shift_width) & 3;

  for (uint i = 0; i < 4; ++i) {
    // Zero out s_mask_out
    s_mask_out[threadId] = 0;
    if (threadId == 0) s_mask_out[s_mask_out_len - 1] = 0;

    __syncthreads();

    // build bit mask output
    bool val_equals_i = false;
    if (index < input_len) {
      val_equals_i = (bit_extract == i);
      s_mask_out[threadId] = val_equals_i;
    }
    __syncthreads();

    // Scan mask outputs (Hillis-Steele)
    uint sum = 0;
    uint max_steps = (uint)log2f(block_size);
    for (uint d = 0; d < max_steps; d++) {
      int partner = threadId - (1 << d);
      if (partner >= 0) {
        sum = s_mask_out[threadId] + s_mask_out[partner];
      } else {
        sum = s_mask_out[threadId];
      }
      __syncthreads();
      s_mask_out[threadId] = sum;
      __syncthreads();
    }

    // Shift elements to produce the same effect as exclusive scan
    uint temp_val = 0;
    temp_val = s_mask_out[threadId];
    __syncthreads();
    s_mask_out[threadId + 1] = temp_val;
    __syncthreads();

    if (threadId == 0) {
      // Zero out first element to produce the same effect as exclusive scan
      s_mask_out[0] = 0;
      uint total_sum = s_mask_out[s_mask_out_len - 1];
      s_mask_out_sums[i] = total_sum;
      g_block_sums[blockIdx.x * gridDim.y + blockIdx.y] = total_sum;
    }
    __syncthreads();

    if (val_equals_i && (index < input_len)) {
      s_merged_scan_mask_out[threadId] = s_mask_out[threadId];
    }

    __syncthreads();
  }

  // Scan mask output sums
  // Just do a naive scan since the array is really small
  if (threadId == 0) {
    uint run_sum = 0;
    for (uint i = 0; i < 4; ++i) {
      scan_mask_out_sums[i] = run_sum;
      run_sum += s_mask_out_sums[i];
    }
  }

  __syncthreads();

  if (index < input_len) {
    uint prefix_sum = s_merged_scan_mask_out[threadId];
    uint new_pos = prefix_sum + scan_mask_out_sums[bit_extract];

    __syncthreads();

    s_data[new_pos] = t_data;
    s_merged_scan_mask_out[new_pos] = prefix_sum;

    __syncthreads();

    g_prefix_sums[index + base_index] = s_merged_scan_mask_out[threadId];
    out_sorted[index + base_index] = s_data[threadId];
  }
}

__global__ void shuffle(uint *const output, const uint *const input,
                        uint *g_scan_block_sums, uint *g_prefix_sums,
                        const uint shift_width, const uint block_size,
                        const uint input_len) {
  uint threadId = threadIdx.x;
  uint index = block_size * blockIdx.y + threadId;
  uint base_index = input_len * blockIdx.x;

  if (index < input_len) {
    index = base_index + index;
    uint t_data = input[index];
    uint t_2bit_extract = (t_data >> shift_width) & 3;
    uint t_prefix_sum = g_prefix_sums[index];
    uint data_glbl_pos =
        g_scan_block_sums[t_2bit_extract * gridDim.x + blockIdx.x] +
        t_prefix_sum + base_index;
    __syncthreads();
    output[data_glbl_pos] = t_data;
  }
}
__global__ void int2float(float *input, const int batch_size,
                          const int dim_len) {
  uint index = blockIdx.y * blockDim.y + threadIdx.x;
  uint base_index = blockIdx.x * dim_len;
  if (index < dim_len) {
    base_index += index;
    reinterpret_cast<int &>(input[index]) =
        (reinterpret_cast<int &>(input[index]) >> 31 & 0x1)
            ? reinterpret_cast<int &>(input[index]) & 0x7fffffff
            : ~reinterpret_cast<int &>(input[index]);
  }
}
__global__ void float2int(float *input, const int batch_size,
                          const int dim_len) {
  uint index = blockIdx.y * blockDim.y + threadIdx.x;
  uint base_index = blockIdx.x * dim_len;
  if (index < dim_len) {
    base_index += index;
    reinterpret_cast<int &>(input[index]) =
        (reinterpret_cast<int &>(input[index]) >> 31 & 0x1)
            ? ~reinterpret_cast<int &>(input[index])
            : reinterpret_cast<int &>(input[index]) | 0x80000000;
  }
}
void vllm_sort_cuda(float *src, const uint batch_size, const uint dim_len) {
  uint block_size = NUM_THREADS;
  uint global_block_num = dim_len / block_size;

  dim3 grid(batch_size, global_block_num);
  dim3 block(block_size);
  float2int<<<grid, block>>>(src, batch_size, dim_len);
  int *src_int = reinterpret_cast<int *>(src);
}

void radix_sort(uint *const dst, uint *src, const uint input_len,
                const uint batch_size) {
  uint block_size = NUM_THREADS;
  uint global_block_num = input_len / block_size;
  // Take advantage of the fact that integer division drops the decimals
  if (input_len % block_size != 0) global_block_num += 1;

  uint *g_prefix_sums = allocate_memory(batch_size * input_len);
  uint *g_block_sums = allocate_memory(4 * batch_size * global_block_num);
  uint *g_scan_block_sums = allocate_memory(4 * batch_size * global_block_num);

  // shared memory consists of 3 arrays the size of the block-wise input
  //  and 2 arrays the size of n in the current n-way split (4)
  uint s_data_len = block_size;
  uint s_mask_out_len = block_size + 1;
  uint s_merged_scan_mask_out_len = block_size;
  uint s_mask_out_sums_len = 4; // 4-way split
  uint s_scan_mask_out_sums_len = 4;
  uint shmem_size_per_block =
      (s_data_len + s_mask_out_len + s_merged_scan_mask_out_len +
       s_mask_out_sums_len + s_scan_mask_out_sums_len) *
      sizeof(uint);

  dim3 grid(batch_size, global_block_num);
  dim3 block(block_size);

  for (uint shift_width = 0; shift_width <= 30; shift_width += 2) {
    gpu_radix_sort_local<<<grid, block, shmem_size_per_block>>>(
        dst, src, g_prefix_sums, g_block_sums, shift_width, input_len,
        block_size);

    sum_scan_blelloch(g_scan_block_sums, g_block_sums, batch_size,
                      4 * global_block_num);

    shuffle<<<grid, block>>>(src, dst, g_scan_block_sums, g_prefix_sums,
                             shift_width, block_size, input_len);
  }
  checkCudaErrors(
      cudaMemcpy(dst, src, sizeof(uint) * input_len, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaFree(g_scan_block_sums));
  checkCudaErrors(cudaFree(g_block_sums));
  checkCudaErrors(cudaFree(g_prefix_sums));
}

int main() {
  std::cout << "fuck" << std::endl;
  return 0;
}
