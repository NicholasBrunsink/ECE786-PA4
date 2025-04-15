#include <iostream>
#include <cuda.h>
#include <random>
#include <fstream>
#include <iomanip>


// The cuda kernel with shared memory optimization
__global__ void softmax_kernel(float* input, float* output, int rows, int cols) {
  __shared__ float smem[1024];
  int index = blockIdx.x * cols;
  int threadId = threadIdx.x;
  
  if (blockIdx.x < rows) {
    float local_max = -10e9f;
    float prev_max = 0.0f;
    float local_sum = 0.0f;

    for (int sub_i=threadId; sub_i<cols; sub_i+=blockDim.x) {
      if (input[index+sub_i] > local_max) {
          prev_max = local_max;
          local_max = input[index+sub_i];
          local_sum = local_sum * exp(prev_max - local_max) + exp(input[index+sub_i] - local_max);  // correct and add current term
        } else {  
          local_sum += exp(input[index+sub_i] - local_max);    // no correction needed
        }
    }
    if (local_max == -10e9) return; // stop computation if no local max found (rare edge case)
    // reduce local max down to 1 value across threads
    smem[threadId] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride>0; stride/=2) {
      if (threadId < stride)
        smem[threadId] = smem[threadId] > smem[threadId+stride] ? smem[threadId] : smem[threadId+stride];
      __syncthreads();  
    }
    float row_max = smem[0];
    __syncthreads();
    smem[threadId] = local_sum * exp(local_max-row_max);
    __syncthreads();

    // reduce local norms down to 1 value using shared mem
    for (int stride = blockDim.x / 2; stride>0; stride/=2) {
      if (threadId < stride)
        smem[threadId] += smem[threadId+stride];
      __syncthreads();  
    }

    float row_sum = smem[0];
    __syncthreads();
    for (int sub_i=threadId; sub_i<cols; sub_i+=blockDim.x) {
      output[index+sub_i] = exp(input[index+sub_i] - row_max) / row_sum;
    }
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
  int precision = 4; float val;
  for (int i=0; i<tot_input; ++i) {
    if (i % num_cols == 0) {
      printf("\nRow %d: ", i / num_cols);
    }
    val = output[i];
    // if (i / num_cols == 1761) {
    //   val = roundf(output[i] * pow(10, precision)) / (float)pow(10, precision);   // used to stop undefined rounding behaviour from changing value
    // }
    printf("%.3F ", val);
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
