#include "utils.h"
__global__ void scan_in_block(uint *const output, const uint *const input,
                              uint *const g_scan_block_sums,
                              const uint block_size,
                              const uint global_block_sum_len) {
  extern __shared__ uint shared_out[];

  int threadId = threadIdx.x;
  int ai = threadId;
  int bi = threadId + blockDim.x;

  shared_out[threadId] = 0;
  shared_out[threadId + blockDim.x] = 0;
  shared_out[threadId + blockDim.x + (blockDim.x >> LOG_NUM_BANKS)] = 0;

  __syncthreads();

  uint dim_len = global_block_sum_len;
  uint base_index = blockIdx.x * global_block_sum_len;
  uint index = gridDim.y * blockIdx.y + threadIdx.x;

  if (index < dim_len) {
    shared_out[ai + CONFLICT_FREE_OFFSET(ai)] = input[base_index + index];
    if (index + blockDim.y < dim_len)
      shared_out[bi + CONFLICT_FREE_OFFSET(bi)] =
          input[base_index + index + blockDim.x];
  }

  /*
    For both upsweep and downsweep:
    Sequential indices with conflict free padding
     Amount of padding = target index / num banks
    Sweeps are pivoted on the last element of shared memory
  */

  // Upsweep/Reduce step
  int offset = 1;
  for (int d = block_size >> 1; d > 0; d >>= 1) {
    __syncthreads();
    if (threadId < d) {
      int ai = offset * ((threadId << 1) + 1) - 1;
      int bi = offset * ((threadId << 1) + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);
      shared_out[bi] += shared_out[ai];
    }
    offset <<= 1;
  }

  // Save the total sum on the global block sums array
  // Then clear the last element on the shared memory
  if (threadId == 0) {
    g_scan_block_sums[blockIdx.x * blockDim.y + blockIdx.y] =
        shared_out[block_size - 1 + CONFLICT_FREE_OFFSET(block_size - 1)];
    shared_out[block_size - 1 + CONFLICT_FREE_OFFSET(block_size - 1)] = 0;
  }

  // Downsweep step
  for (int d = 1; d < block_size; d <<= 1) {
    offset >>= 1;
    __syncthreads();

    if (threadId < d) {
      int ai = offset * ((threadId << 1) + 1) - 1;
      int bi = offset * ((threadId << 1) + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      uint temp = shared_out[ai];
      shared_out[ai] = shared_out[bi];
      shared_out[bi] += temp;
    }
  }
  __syncthreads();

  if (index < dim_len) {
    output[index + base_index] = shared_out[ai + CONFLICT_FREE_OFFSET(ai)];
    if (index + blockDim.x < dim_len)
      output[index + blockDim.x + base_index] =
          shared_out[bi + CONFLICT_FREE_OFFSET(bi)];
  }
}
void sum_scan_blelloch(uint *const dst, const uint *const src,
                       const uint batch_size, const uint dim_len) {
  // Zero out d_out
  checkCudaErrors(cudaMemset(dst, 0, dim_len * sizeof(uint)));

  uint block_size = NUM_THREADS / 2;
  uint block_num = dim_len / NUM_THREADS;
  if (dim_len % block_size != 0) block_num += 1;

  /* Conflict free padding requires that shared memory be more than 2 * block_sz
  solve the problem of  bank conflicts */
  uint shmem_size = NUM_THREADS + ((NUM_THREADS) >> LOG_NUM_BANKS);
  uint *scan_block_sums = allocate_memory(batch_size * block_num);

  dim3 grid(batch_size, block_num);
  dim3 block(block_size);

  scan_in_block<<<grid, block, sizeof(uint) * shmem_size>>>(
      dst, src, scan_block_sums, block_size, dim_len);

  if (block_num <= block_size) {
    uint *t_dummy_blocks_sums = allocate_memory(batch_size);
    scan_in_block<<<1, block_size, sizeof(uint) * shmem_size>>>(
        scan_block_sums, scan_block_sums, t_dummy_blocks_sums, block_num,
        shmem_size, block_size);
    checkCudaErrors(cudaFree(t_dummy_blocks_sums));
  } else {
    uint *t_in_block_sums;
    checkCudaErrors(
        cudaMalloc(&t_in_block_sums, sizeof(uint) * batch_size * block_num));
    checkCudaErrors(cudaMemcpy(t_in_block_sums, scan_block_sums,
                               sizeof(uint) * batch_size * block_num,
                               cudaMemcpyDeviceToDevice));
    sum_scan_blelloch(scan_block_sums, t_in_block_sums, batch_size, block_num);
    checkCudaErrors(cudaFree(t_in_block_sums));
  }

  gpu_add_block_sums<<<grid, block>>>(dst, dst, scan_block_sums, batch_size,
                                      dim_len);

  checkCudaErrors(cudaFree(scan_block_sums));
}
