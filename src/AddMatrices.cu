/**
  * Copyright 2014, André R. Brodtkorb
  * Released under GPLv3
  */

#include <vector>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>

/**
  * Only this function runs on the GPU
  * Note that a, b, and c point to memory areas on the GPU itself
  */
__global__ void addMatricesKernel(float* c, float* a, float* b,
                          unsigned int cols, unsigned int rows) {
    unsigned int global_x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int global_y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int k = global_y*cols + global_x;
    c[k] = a[k] + b[k];
}

/**
  * Reference solution on the CPU
  */
void addFunctionCPU(float* c, float* a, float* b,
                 unsigned int cols, unsigned int rows) {
    for (unsigned int j=0; j<rows; ++j) {
        for (unsigned int i=0; i<cols; ++i) {
            unsigned int k = j*cols + i;
            c[k] = a[k] + b[k];
        }
    }
}

void addFunctionGPU(float* c, float* a, float* b,
                 unsigned int cols, unsigned int rows) {
    cudaError_t cudaStatus;

    float* gpu_a = 0;
    float* gpu_b = 0;
    float* gpu_c = 0;
    
    /**
      * Be ware: should really be using cudaMalloc2D to get proper alignment
      * of each row. However, this way makes it easier to understand, and looks
      * a lot like traditional CPU code.
      */
    cudaStatus = cudaMalloc((void**)&gpu_a, cols*rows*sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed!" << std::endl;
        cudaFree(gpu_a);
        exit(-1);
    }
    
    cudaStatus = cudaMalloc((void**)&gpu_b, cols*rows*sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed!" << std::endl;
        cudaFree(gpu_b);
        exit(-1);
    }
    
    cudaStatus = cudaMalloc((void**)&gpu_c, cols*rows*sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed!" << std::endl;
        cudaFree(gpu_c);
        exit(-1);
    }
    

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(gpu_a, a, cols*rows*sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy failed!" << std::endl;
        cudaFree(gpu_c);
        exit(-1);
    }
    
    cudaStatus = cudaMemcpy(gpu_b, b, cols*rows*sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy failed!" << std::endl;
        cudaFree(gpu_c);
        exit(-1);
    }

    // "Launch" the kernel on the GPU
    dim3 block(8, 8);
    dim3 grid(cols/8, rows/8);
    addMatricesKernel<<<grid, block>>>(gpu_c, gpu_a, gpu_b, cols, rows);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cout << "addMatricesKernel failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        exit(-1);
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching addKernel!" << std::endl;
        exit(-1);
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, gpu_c, cols*rows*sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaMemcpy failed!" << std::endl;
        exit(-1);
    }
    
    // Lazy, should really check for errors here.
    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_c);
    
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaDeviceReset failed!" << std::endl;
        exit(-1);
    }
}




int main(int argc, char** argv) {
    const unsigned int cols = 512;
    const unsigned int rows = 512;
    std::vector<float> a(rows*cols), b(rows*cols), c(rows*cols), c_ref(rows*cols);

    //Set seed
    srand(int(time(NULL)));

    //Initialize a, b, c
    for (unsigned int j=0; j<rows; ++j) {
        for (unsigned int i=0; i<cols; ++i) {
            a[j*cols+i] = static_cast<float>(rand()) / static_cast <float> (RAND_MAX);
            b[j*cols+i] = static_cast<float>(rand()) / static_cast <float> (RAND_MAX);
            c[j*cols+i] = 0;
            c_ref[j*cols+i] = a[j*cols+i] + b[j*cols+i];
        }
    }

    // Add vectors in parallel.
    addFunctionGPU(&c[0], &a[0], &b[0], cols, rows);

    //Check that c is actually computed correctly.
    float diff = 0.0f;
    float avg = 0.0f;
    for (unsigned int j=0; j<rows; ++j) {
        for (unsigned int i=0; i<cols; ++i) {
            avg += c[j*cols+i];
            diff += abs(c_ref[j*cols+i] - c[j*cols+i]);
        }
    }
    avg = avg / static_cast<float>(rows*cols);

    std::cout << "Difference between CPU and GPU result (should be 0): " << diff << std::endl;
    std::cout << "C average (should be within [0.99, 1.01]): " << avg << std::endl;

    return 0;
}
