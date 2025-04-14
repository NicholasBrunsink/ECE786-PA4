#include <iostream>
#include <cuda.h>
#include <random>
#include <fstream>
#include <iomanip>


// The cuda kernel without shared memory optimization
__global__ void softmax_kernel(float* input, float* output, int rows, int cols) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int index = row*cols; 

  if (row < rows) {
    float row_max = -10e9;
    for (int sub_i=0; sub_i<cols; sub_i++) {
      if (input[index+sub_i] > row_max) row_max = input[index+sub_i];
    }

    float row_sum=0;
    for (int sub_i=0; sub_i<cols; sub_i++) {
      row_sum += exp(input[index+sub_i] - row_max);
    }

    for (int sub_i=0; sub_i<cols; sub_i++) {
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
  dim3 threadsPerBlock(1024);
  dim3 blocksPerGrid(num_rows / threadsPerBlock.x + (num_rows % threadsPerBlock.x != 0));   // each kernel handles a row. Each block holds 1024 threads

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
  // int precision = 6;
  for (int i=0; i<tot_input; ++i) {
    if (i % num_cols == 0) {
      printf("\nRow %d: ", i / num_cols);
    }
    printf("%.3F ", output[i]);
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
