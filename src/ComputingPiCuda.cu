/**
  * Copyright 2014, André R. Brodtkorb
  * Released under GPLv3
  */

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <math.h>
#include <time.h>


#ifdef _WIN32
#include <sys/timeb.h>
#include <windows.h>
#else
#include <sys/time.h>
#include <unistd.h>
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"



//Based on Stroustrup, adapted for CUDA
__device__ float generateRandomNumber(long& last_draw) {
    last_draw = last_draw*1103515245 + 12345;
    long abs = last_draw & 0x7fffffff;
    return abs / 2147483648.0; 
}


/**
  * @param output Where to place results
  * @param seed Seed used to seed the RNG (Linear congruential generator)
  * Uses only for 1 thread per block
  */
__global__ void computePiKernel1(unsigned int* output, unsigned int seed) {
    unsigned int tid = blockIdx.x;
    long spacing = 18446744073709551615ul / static_cast<unsigned long>(gridDim.x);
    long last_draw = seed + tid*spacing; //Initialize the LCG to seed, and keep track of last drawn long
    int n_inside = 0;
    
    //Generate coordinate
    float x = generateRandomNumber(last_draw);
    float y = generateRandomNumber(last_draw);

    //Compute radius
    float r = sqrt(x*x + y*y);

    //Check if within circle
    if (r <= 1.0f) {
        output[tid] = 1;
    }
    else {
        output[tid] = 0;
    }
}




/**
  * @param output Where to place results
  * @param seed Seed used to seed the RNG (Linear congruential generator)
  * Uses only for 32 threads per block
  */
__global__ void computePiKernel2(unsigned int* output, unsigned int seed) {
    unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
    long spacing = 18446744073709551615ul / static_cast<unsigned long>(gridDim.x*blockDim.x);
    long last_draw = seed + tid*spacing; //Initialize the LCG to seed, and keep track of last drawn long
    int n_inside = 0;
    
    //Generate coordinate
    float x = generateRandomNumber(last_draw);
    float y = generateRandomNumber(last_draw);

    //Compute radius
    float r = sqrt(x*x + y*y);

    //Check if within circle
    if (r <= 1.0f) {
        output[tid] = 1;
    }
    else {
        output[tid] = 0;
    }
}



/**
  * @param output Where to place results
  * @param seed Seed used to seed the RNG (Linear congruential generator)
  * Uses 32 threads per block and shared memory
  */
__global__ void computePiKernel3(unsigned int* output, unsigned int seed) {
    unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
    long spacing = 18446744073709551615ul / static_cast<unsigned long>(blockDim.x*gridDim.x);
    long last_draw = seed + tid*spacing; //Initialize the LCG to seed, and keep track of last drawn long
    int n_inside = 0;

    __shared__ int inside[32];
    
    //Generate coordinate
    float x = generateRandomNumber(last_draw);
    float y = generateRandomNumber(last_draw);

    //Compute radius
    float r = sqrt(x*x + y*y);

    //Check if within circle
    if (r <= 1.0f) {
        inside[threadIdx.x] = 1;
    }
    else {
        inside[threadIdx.x] = 0;
    }

    //Use shared memory reduction to find number of inside per block
    //We don't need __syncthreads, as all threads are within the same warp
    volatile int* p = &inside[0]; //To help the compiler not cache this variable...
    if (threadIdx.x < 16) {
        p[threadIdx.x] = p[threadIdx.x] + p[threadIdx.x+16];
        p[threadIdx.x] = p[threadIdx.x] + p[threadIdx.x+8];
        p[threadIdx.x] = p[threadIdx.x] + p[threadIdx.x+4];
        p[threadIdx.x] = p[threadIdx.x] + p[threadIdx.x+2];
        p[threadIdx.x] = p[threadIdx.x] + p[threadIdx.x+1];
    }
    
    if (threadIdx.x == 0) {
        output[blockIdx.x] = inside[threadIdx.x];
    }
}




/**
  * @param output Where to place results
  * @param seed Seed used to seed the RNG (Linear congruential generator)
  * Uses 32 threads per block, multiple evaluations per thread, and shared memory
  */
__global__ void computePiKernel4(unsigned int* output, unsigned int seed, unsigned int iters_per_thread) {
    unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
    long spacing = 18446744073709551615ul / static_cast<unsigned long>(blockDim.x*gridDim.x);
    long last_draw = seed + tid*spacing; //Initialize the LCG to seed, and keep track of last drawn long
    int n_inside = 0;

    __shared__ int inside[32];

    inside[threadIdx.x] = 0;

    for (unsigned int i=0; i<iters_per_thread; ++i) {
        //Generate coordinate
        float x = generateRandomNumber(last_draw);
        float y = generateRandomNumber(last_draw);

        //Compute radius
        float r = sqrt(x*x + y*y);

        //Check if within circle
        if (r <= 1.0f) {
            ++inside[threadIdx.x];
        }
    }

    //Use shared memory reduction to find number of inside per block
    //We don't need __syncthreads, as all threads are within the same warp
    volatile int* p = &inside[0]; //To help the compiler not cache this variable...
    if (threadIdx.x < 16) {
        p[threadIdx.x] = p[threadIdx.x] + p[threadIdx.x+16];
        p[threadIdx.x] = p[threadIdx.x] + p[threadIdx.x+8];
        p[threadIdx.x] = p[threadIdx.x] + p[threadIdx.x+4];
        p[threadIdx.x] = p[threadIdx.x] + p[threadIdx.x+2];
        p[threadIdx.x] = p[threadIdx.x] + p[threadIdx.x+1];
    }
    
    if (threadIdx.x == 0) {
        output[blockIdx.x] = inside[threadIdx.x];
    }
}


/**
  * @param output Where to place results
  * @param seed Seed used to seed the RNG (Linear congruential generator)
  * Uses 512 threads per block, multiple evaluations per thread, and shared memory
  */
__global__ void computePiKernel5(unsigned int* output, unsigned int seed, unsigned int iters_per_thread) {
    unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
    long spacing = 18446744073709551615ul / static_cast<unsigned long>(blockDim.x*gridDim.x);
    long last_draw = seed + tid*spacing; //Initialize the LCG to seed, and keep track of last drawn long
    int n_inside = 0;

    extern __shared__ int inside[];


    for (unsigned int i=0; i<iters_per_thread; ++i) {
        //Generate coordinate
        float x = generateRandomNumber(last_draw);
        float y = generateRandomNumber(last_draw);

        //Compute radius
        float r = sqrt(x*x + y*y);

        //Check if within circle
        if (r <= 1.0f) {
            ++n_inside;
        }
    }
    
    inside[threadIdx.x] = n_inside;

    __syncthreads(); // Ensure all threads have reached this point
    // Reduce from 512 to 256
    if(threadIdx.x < 256) { inside[threadIdx.x] = inside[threadIdx.x] + inside[threadIdx.x + 256]; }
    __syncthreads();
    // Reduce from 256 to 128
    if(threadIdx.x < 128) { inside[threadIdx.x] = inside[threadIdx.x] + inside[threadIdx.x + 128]; }
    __syncthreads();
    // Reduce from 128 to 64
    if(threadIdx.x < 64) { inside[threadIdx.x] = inside[threadIdx.x] + inside[threadIdx.x + 64]; }
    __syncthreads();

    //Use shared memory reduction to find number of inside per block
    //We don't need __syncthreads, as all threads are within the same warp
    volatile int* p = &inside[0]; //To help the compiler not cache this variable...
    if (threadIdx.x < 32) {
        p[threadIdx.x] = p[threadIdx.x] + p[threadIdx.x+32];
        p[threadIdx.x] = p[threadIdx.x] + p[threadIdx.x+16];
        p[threadIdx.x] = p[threadIdx.x] + p[threadIdx.x+8];
        p[threadIdx.x] = p[threadIdx.x] + p[threadIdx.x+4];
        p[threadIdx.x] = p[threadIdx.x] + p[threadIdx.x+2];
        p[threadIdx.x] = p[threadIdx.x] + p[threadIdx.x+1];
    }
    
    if (threadIdx.x == 0) {
        output[blockIdx.x] = inside[threadIdx.x];
    }
}








inline double getCurrentTime() {
#ifdef WIN32
    LARGE_INTEGER f;
    LARGE_INTEGER t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return t.QuadPart/(double) f.QuadPart;
#else
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    return tv.tv_sec+tv.tv_usec*1e-6;
#endif
};

template<typename T>
T getFromCin() {
    std::string line;
    T retval;
    while (std::getline(std::cin, line)) {
        std::stringstream ss(line);
        if (ss >> retval && ss.eof()) {
            return retval;
        }
        std::cout << "You entered '" << line << "', which I was unable to interpret. Please try again" << std::endl;
    }
}


float computePi(int kernel_no, int n_points) {
    cudaError_t cudaStatus;

    dim3 grid;
    dim3 block;
    int gpu_data_elements;
    const unsigned int iters_per_thread = 1000;
    const unsigned int threads_per_block = 512;

    switch(kernel_no) {
    case 1:
        grid = dim3(n_points, 1, 1); 
        block = dim3(1, 1, 1);
        gpu_data_elements = n_points;
        break;
    case 2:
        grid = dim3(n_points/32, 1, 1); 
        block = dim3(32, 1, 1);
        //since we don't actually draw the correct number of poinst, we must update variable
        n_points = grid.x*block.x;
        gpu_data_elements = n_points;
        break;
    case 3:
        grid = dim3(n_points/32, 1, 1); 
        block = dim3(32, 1, 1);
        gpu_data_elements = grid.x;
        //since we don't actually draw the correct number of poinst, we must update variable
        n_points = grid.x*block.x;
        break;
    case 4:
        grid = dim3(max(1, n_points/(32*iters_per_thread)), 1, 1); 
        block = dim3(32, 1, 1);
        gpu_data_elements = grid.x;
        //since we don't actually draw the correct number of poinst, we must update variable
        n_points = grid.x*block.x*iters_per_thread;
        break;
    case 5:
        grid = dim3(max(1, n_points/(threads_per_block*iters_per_thread)), 1, 1); 
        block = dim3(threads_per_block, 1, 1);
        gpu_data_elements = grid.x;
        //since we don't actually draw the correct number of poinst, we must update variable
        n_points = grid.x*block.x*iters_per_thread;
        break;
    default:
        std::cout << "Unknown kernel number: " << kernel_no << std::endl;
        return 0.0f;
    }
    std::cout << "Using " << n_points << " points." << std::endl;

    // Allocate GPU buffer for output data: one element per block
    unsigned int* gpu_data;
    cudaStatus = cudaMalloc((void**)&gpu_data, gpu_data_elements*sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed!" << std::endl;
        cudaFree(gpu_data);
        return 0.0f;
    }

    //Set new seed for each iteration
    unsigned int seed = getCurrentTime()*1000; //Set random seed
    
    // Launch a kernel on the GPU with a given set of blocks and threads.
    switch(kernel_no) {
    case 1:
        computePiKernel1<<<grid, block>>>(gpu_data, seed);
        break;
    case 2:
        computePiKernel2<<<grid, block>>>(gpu_data, seed);
        break;
    case 3:
        computePiKernel3<<<grid, block>>>(gpu_data, seed);
        break;
    case 4:
        computePiKernel4<<<grid, block>>>(gpu_data, seed, iters_per_thread);
        break;
    case 5:
        computePiKernel5<<<grid, block, threads_per_block*sizeof(int)>>>(gpu_data, seed, iters_per_thread);
        break;
    default:
        std::cout << "Unknown kernel number: " << kernel_no << std::endl;
        return 0.0f;
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cout << "computePiKernel failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(gpu_data);
        return 0.0f;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching addKernel!" << std::endl;
        cudaFree(gpu_data);
        return 0.0f;
    }

    // Copy output vector from GPU buffer to host memory.
    std::vector<unsigned int> cpu_data(gpu_data_elements);
    cudaStatus = cudaMemcpy(&cpu_data[0], gpu_data, cpu_data.size()*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaMemcpy failed!" << std::endl;
        cudaFree(gpu_data);
        return 0.0f;
    }

    // Free GPU data
    cudaStatus = cudaFree(gpu_data);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaFree failed!" << std::endl;
    }

    int n_inside = 0;
    for (int i=0; i<cpu_data.size(); ++i) {
        n_inside += cpu_data[i];
    }

    //Estimate Pi
    float pi = 4.0f * n_inside / static_cast<float>(n_points);

    return pi;
}

int main(int argc, char** argv) {
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaSetDevice failed!" << std::endl;
        return -1;
    }

    std::cout << "Estimating value of Pi (Press CTRL+C to exit)" << std::endl;
    std::cout << "True value of Pi:   3.14159265359..." << std::endl;
    for(;;) {
        std::cout << "Please enter number of iterations: " << std::endl;
        int n_points = getFromCin<int>();
        std::cout << "Please enter kernel to use: " << std::endl;
        int kernel = getFromCin<int>();
        double tic = getCurrentTime();
        float pi = computePi(kernel, n_points);
        double toc = getCurrentTime();
        std::cout << "Estimated Pi to be: " << std::fixed << pi << " in " << (toc-tic) << " seconds." << std::endl;
    }
        
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaDeviceReset failed!" << std::endl;
        return -1;
    }

    return 0;
}























