#include <iostream>
#include <random>
#include <fstream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "cuda_utils.cuh"

// The cuda kernel with shared memory optimization
__global__ void softmax_kernel(float* __restrict__ xd, float* __restrict__ resd, int M, int N) {
  // max and norm reduction will happen in shared memory (static)
  __shared__ float smem[1024];

  int row = blockIdx.x;
  int tid = threadIdx.x;
  // number of threads in a warp
  unsigned int warp_size = 32;
  if (row >= M) return;

  float* input_row = xd + row * N;
  float* output_row = resd + row * N;
  float local_max = -INFINITY;
  float local_norm = 0.0f;

  for (int i = tid; i < N; i += blockDim.x) {
      float x = input_row[i];
      if (x > local_max) {
          local_norm *= expf(local_max - x);
          local_max = x;
      }
      local_norm += expf(x - local_max);
  }
  __syncthreads();

  // each thread will have its own local max
  // we store it in shared memory for reduction
  // smem[tid] = local_max;
  // __syncthreads();

  // warp level reduction using XOR shuffle ('exchanges' the values in the threads)
  // note: if there are 256 threads in one block (8 warps of 32 threads each)
  // the following for loop reduces the value in all the 8 warps
  // the 8 warps contain the 8 maximum values of the 32 threads that reside in those warps
  // float val = smem[tid];
  float val = local_max;
  for (int offset = warp_size / 2; offset > 0; offset /= 2) {
      val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }

  // when blockDim is greater than 32, we need to do a block level reduction
  // AFTER warp level reductions since we have the 8 maximum values that needs to be reduced again
  // the global max will be stored in the first warp
  if (blockDim.x > warp_size) {
      if (tid % warp_size == 0) {
          // which warp are we at?
          // store the value in its first thread index
          smem[tid / warp_size] = val;
      }
      __syncthreads();

      // first warp will do global reduction only
      // this is possible because we stored the values in the shared memory
      // so the threads in the first warp will read from it and then reduce
      if (tid < warp_size) {
          val = (tid < CEIL_DIV(blockDim.x, warp_size)) ? smem[tid] : -INFINITY;
          for (int offset = warp_size / 2; offset > 0; offset /= 2) {
              val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
          }
          if (tid == 0) smem[0] = val;
      }
  } else {
      // this is for when the number of threads in a block are not
      // greater than the warp size, in that case we already reduced
      // so we can store the value
      if (tid == 0) smem[0] = val;
  }
  __syncthreads();

  // we got the global row max now
  float row_max = smem[0];
  __syncthreads();

  // each thread will have its own local_norm
  // we will store the corrected local_norm in the shared memory
  // smem[tid] = local_norm * expf(local_max - row_max);
  // __syncthreads();

  // same reduction algorithm as above, but instead of max reduction
  // we do a sum reduction i.e. we accumulate the values
  // val = smem[tid];
  val = local_norm * expf(local_max - row_max);
  for (int offset = warp_size / 2; offset > 0; offset /= 2) {
      val += __shfl_down_sync(0xffffffff, val, offset);
  }

  if (blockDim.x > warp_size) {
      if (tid % warp_size == 0) {
          smem[tid / warp_size] = val;
      }
      __syncthreads();

      // first warp will do global reduction
      if (tid < warp_size) {
          val = (tid < CEIL_DIV(blockDim.x, warp_size)) ? smem[tid] : 0.0f;
          for (int offset = warp_size / 2; offset > 0; offset /= 2) {
              val += __shfl_down_sync(0xffffffff, val, offset);
          }
          if (tid == 0) smem[0] = val;
      }
  } else {
      if (tid == 0) smem[0] = val;
  }
  __syncthreads();

  float row_norm = smem[0];
  __syncthreads();

  // finally, compute softmax
  for (int i = tid; i < N; i += blockDim.x) {
      output_row[i] = expf(input_row[i] - row_max) / row_norm;
  }
}


int main(int argc, char *argv[]) {
  // Read the inputs from command line
  if (argc != 2) throw std::invalid_argument("Incorrect # of args (expected 1)");

  const char *in_file = argv[1];
  std::ifstream input_file(in_file);
  if (!input_file) throw std::invalid_argument("Bad file");

  // Allocate/move data using cudaMalloc and cudaMemCpy
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // extracting # rows / # cols
  int num_rows, num_cols;
  input_file >> num_rows; input_file >> num_cols;
  int tot_input = num_rows*num_cols;

  float *input = (float*) malloc(tot_input*sizeof(float));
  float *output = (float*) malloc(tot_input*sizeof(float));

  for (int i=0; i<tot_input; ++i) {
    input_file >> input[i];
  }

  cudaError_t err;
  // Allocate/move data using cudaMalloc and cudaMemCpy
  float *in_arr, *out_arr;
  err=cudaMalloc(&in_arr, tot_input * sizeof(float));
  if (err != cudaSuccess) {std::cerr << "CUDA memcpy failed for kernel: " << cudaGetErrorString(err) << std::endl; return -1;}
  err=cudaMalloc(&out_arr, tot_input * sizeof(float));
  if (err != cudaSuccess) {std::cerr << "CUDA memcpy failed for kernel: " << cudaGetErrorString(err) << std::endl; return -1;}
  err=cudaMemcpy(in_arr, input, tot_input * sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {std::cerr << "CUDA memcpy failed for kernel: " << cudaGetErrorString(err) << std::endl; return -1;}

  // Launch the kernel
  // Grid and block dimensions
  dim3 threadsPerBlock(1024);  // block size of 16x16 threads
  dim3 blocksPerGrid(num_rows);

  cudaEventRecord(start);
  softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(in_arr, out_arr, num_rows, num_cols);
  cudaEventRecord(stop);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(error) << std::endl;
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {std::cerr << "CUDA synchronization failed: " << cudaGetErrorString(err) << std::endl; return -1;}

  // Print the output
  cudaMemcpy(output, out_arr, tot_input*sizeof(float), cudaMemcpyDeviceToHost);
  cudaEventSynchronize(stop);
  float milliseconds=0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("No of rows: %d\nNo of cols: %d", num_rows, num_cols);
  int precision = 7; float val;
  for (int i=0; i<tot_input; ++i) {
    if (i % num_cols == 0) {
      printf("\nRow %d: ", i / num_cols);
    }
    // val = roundf(output[i] * pow(10, precision)) / (float)pow(10, precision);   // used to stop undefined rounding behaviour from changing value
    printf("%.3f ", output[i]);
  } 
  printf("\n");
  // printf("\n%f\n", milliseconds);

  // Clean up the memory
  free(input);
  free(output);
  cudaFree(in_arr);
  cudaFree(out_arr);
  return 0;
}
