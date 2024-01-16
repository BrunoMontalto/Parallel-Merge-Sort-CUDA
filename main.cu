#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <device_functions.h>


#include <stdio.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <cstdlib>

#include <omp.h>

#include <cstdio>
#include <chrono>

#include <algorithm>

#include "sequentialMergeSort.h"







class Entity {
public:
    float x, y, angle;
    short spriteIndex;
    float distanceFromPlayer;

    __host__ __device__ Entity() {}

    __host__ __device__ Entity(float x, float y, float angle, short spriteIndex, float distanceFromPlayer)
        : x(x), y(y), angle(angle), spriteIndex(spriteIndex), distanceFromPlayer(distanceFromPlayer) {}

    __host__ __device__ bool operator==(const Entity& other) const {
        return distanceFromPlayer == other.distanceFromPlayer;
    }

    __host__ __device__ bool operator!=(const Entity& other) const {
        return distanceFromPlayer != other.distanceFromPlayer;
    }

    __host__ __device__ bool operator<(const Entity& other) const {
        return distanceFromPlayer < other.distanceFromPlayer;
    }

    __host__ __device__ bool operator<=(const Entity& other) const {
        return distanceFromPlayer <= other.distanceFromPlayer;
    }

    __host__ __device__ bool operator>(const Entity& other) const {
        return distanceFromPlayer > other.distanceFromPlayer;
    }

    __host__ __device__ bool operator>=(const Entity& other) const {
        return distanceFromPlayer >= other.distanceFromPlayer;
    }
};


// Specialization of std::numeric_limits for Entity
namespace std {
    template <>
    class numeric_limits<Entity> {
    public:
        static Entity min() noexcept {
            return Entity(0, 0, 0, 0, -1);
        }
    };

}


template<class T>
__device__ int binarySearchFirstDesc(T* arr, int low, int high, T elem) {
    while (low <= high) {
        int mid = low + (high - low) / 2;

        if (arr[mid] <= elem) {
            high = mid - 1;
        }
        else if (arr[mid] > elem) {
            low = mid + 1;
        }
    }
    return low;
}


template<class T>
__device__ int binarySearchLastDesc(T* arr, int low, int high, T elem) {
    while (low <= high) {
        int mid = low + (high - low) / 2;

        if (arr[mid] < elem) {
            high = mid - 1;
        }
        else {
            low = mid + 1;
        }
    }
    return low;
}


template<class T>
__global__ void mergeSortDesc(T* arr, T* out, int size, int chunkSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size) {
        return;
    }

    T myElem = arr[i];
    int chunkIndex = i / chunkSize;


    int odd = chunkIndex & 1;
    int low = (chunkIndex + 1 - 2 * odd) * chunkSize;
    int high = low + chunkSize - 1;

    int otherGreaterElems = (odd ? binarySearchLastDesc(arr, low, high, myElem) : binarySearchFirstDesc(arr, low, high, myElem)) - low;
    int ourGreaterElems = i - chunkIndex * chunkSize;
    int finalIndex = (chunkIndex - odd) * chunkSize + otherGreaterElems + ourGreaterElems;

    out[finalIndex] = myElem;
}


template<class T>
__global__ void mergeSortDescSM_Till128(T* arr, int size, int logn) {
    __shared__ T sharedmem[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size) {
        return;
    }

    sharedmem[tid] = arr[i];
    __syncthreads();


    int chunkSize = 1;

    int chunkIndex = tid;
    for (int iter = 0; iter < 8 && iter < logn; iter++) {
        T myElem = sharedmem[tid];

        int odd = chunkIndex & 1;

        int low = (chunkIndex + 1 - 2 * odd) * chunkSize;
        int high = low + chunkSize - 1;


        int otherGreaterElems = (odd ? binarySearchLastDesc(sharedmem, low, high, myElem) : binarySearchFirstDesc(sharedmem, low, high, myElem)) - low;
        int ourGreaterElems = tid - chunkIndex * chunkSize;
        int finalIndex = (chunkIndex - odd) * chunkSize + otherGreaterElems + ourGreaterElems;

        chunkSize *= 2;
        chunkIndex >>= 1;

        __syncthreads();

        sharedmem[finalIndex] = myElem;

        __syncthreads();
    }

    arr[i] = sharedmem[tid];
}


template<class T>
__global__ void mergeSortDescSM256(T* arr, T* out, int size) {
    __shared__ T otherChunk[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size) {
        return;
    }

    T myElem = arr[i];
    int chunkIndex = i >> 8;


    int odd = chunkIndex & 1;

    otherChunk[tid] = arr[i + (1 - 2 * odd) * 256];
    __syncthreads();


    int otherGreaterElems = odd ? binarySearchLastDesc(otherChunk, 0, 255, myElem) : binarySearchFirstDesc(otherChunk, 0, 255, myElem);
    int ourGreaterElems = tid;
    int finalIndex = (chunkIndex - odd) * 256 + otherGreaterElems + ourGreaterElems;

    out[finalIndex] = myElem;
}



template<class T>
__global__ void mergeSortDescSM_From256(T* arr, T* out, int size, int chunkSize) {
    extern __shared__ T otherChunk[];
    int i = blockIdx.x * chunkSize + threadIdx.x;

    if (i >= size) {
        return;
    }


    int chunkIndex = i / chunkSize;
    int odd = chunkIndex & 1;


    for (int iter = 0; iter < chunkSize / blockDim.x; iter++) {
        int new_i = i + iter * blockDim.x;
        otherChunk[new_i - chunkIndex * chunkSize] = arr[new_i + (1 - 2 * odd) * chunkSize];
    }
    __syncthreads();

    for (int iter = 0; iter < chunkSize / blockDim.x; iter++) {
        int new_i = i + iter * blockDim.x;
        T myElem = arr[new_i];

        int otherGreaterElems = odd ? binarySearchLastDesc(otherChunk, 0, chunkSize - 1, myElem) : binarySearchFirstDesc(otherChunk, 0, chunkSize - 1, myElem);
        int ourGreaterElems = new_i - chunkIndex * chunkSize;
        int finalIndex = (chunkIndex - odd) * chunkSize + otherGreaterElems + ourGreaterElems;

        out[finalIndex] = myElem;
    }
}




template<class T>
bool isSortedDesc(T* arr, int size) {
    bool res = true;
    for (int i = 0; i < size - 1; i++) {
        if (arr[i] < arr[i + 1]) {
            printf("element %i < element %i\n", i, i + 1);
            res = false;
        }
    }
    return res;
}


template <typename T>
bool checkArraysHaveSameElements(const T* arr1, const T* arr2, int size) {
    T* sortedArr1 = static_cast<T*>(malloc(size * sizeof(T)));
    T* sortedArr2 = static_cast<T*>(malloc(size * sizeof(T)));

    std::memcpy(sortedArr1, arr1, size * sizeof(T));
    std::memcpy(sortedArr2, arr2, size * sizeof(T));

    std::sort(sortedArr1, sortedArr1 + size);
    std::sort(sortedArr2, sortedArr2 + size);

    bool result = std::equal(sortedArr1, sortedArr1 + size, sortedArr2);

    free(sortedArr1);
    free(sortedArr2);

    return result;
}


template<class T>
bool checkResult(T* res, T* arr, int size) {
    return isSortedDesc(res, size) && checkArraysHaveSameElements(arr, res, size);
}


void cuda_check(cudaError_t err, const char* str)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "%d - %s (%s)\n", err, cudaGetErrorString(err), str);
        exit(1);
    }
}


int smallestGreaterPowerOf2(int n) {
    int power = 256;
    while (power < n) {
        power <<= 1;
    }
    return power;
}


template<class T>
T* parallelMergeSortDesc(T* arr, int numels, float* elapsedRecords) {
    //allocation begin
    int size = smallestGreaterPowerOf2(numels);
    T* arrHost, * arrDevice, * tempDevice;



    arrHost = (T*)malloc(size * sizeof(T));
    cudaError_t err = cudaMalloc((void**)&arrDevice, size * sizeof(T));
    cuda_check(err, "cudaMalloc");
    err = cudaMalloc((void**)&tempDevice, size * sizeof(T));
    cuda_check(err, "cudaMalloc");


#pragma omp parallel for
    for (int i = 0; i < numels; ++i) {
        arrHost[i] = arr[i];
    }
    //add pad
    T min = std::numeric_limits<T>::min();
#pragma omp parallel for
    for (int i = numels; i < size; ++i) {
        arrHost[i] = min;
    }
    //allocation end






    // Launch the CUDA kernel
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    printf("Launching with blockSize: %d, numBlocks: %d, numels: %d, size: %d\n", blockSize, numBlocks, numels, size);

    // Copy the array from the host to the device
    err = cudaMemcpy(arrDevice, arrHost, size * sizeof(T), cudaMemcpyHostToDevice);
    cuda_check(err, "cudaMemcopy");

    int logn = int(log2(size));

    //create start and stop events
    cudaEvent_t start_i, stop_i;
    err = cudaEventCreate(&start_i);
    cuda_check(err, "cudaEventCreate");
    err = cudaEventCreate(&stop_i);
    cuda_check(err, "cudaEventCreate");

    int chunkSize = 1;
    for (int i = 0; i < logn; i++) {
        //i start event
        err = cudaEventRecord(start_i);
        cuda_check(err, "cudaEventRecord");

        //launch kernel and wait
        mergeSortDesc << <numBlocks, blockSize >> > (arrDevice, tempDevice, size, chunkSize);
        chunkSize *= 2;
        err = cudaDeviceSynchronize();
        cuda_check(err, "cudaDeviceSynchronize");

        //swap pointers
        T* temp = arrDevice;
        arrDevice = tempDevice;
        tempDevice = temp;

        //i stop event
        err = cudaEventRecord(stop_i);
        cuda_check(err, "cudaEventRecord");
        err = cudaEventSynchronize(stop_i);
        cuda_check(err, "cudaEventSynchronize");

        //get elapsed
        err = cudaEventElapsedTime(&elapsedRecords[i], start_i, stop_i);
        cuda_check(err, "cudaElapsedTime");
    }


    // Copy the array from the device to the host
    err = cudaMemcpy(arrHost, arrDevice, size * sizeof(T), cudaMemcpyDeviceToHost);

    cuda_check(err, "cudaMemcopy");

    float totalTime = 0;
    for (int i = 0; i < logn; i++) {
        totalTime += elapsedRecords[i];
        printf(" [ITERATION %i] %.8f ms elapsed\n", i, elapsedRecords[i]);
    }
    printf(" [TOTAL] %.8f ms elapsed\n", totalTime);

    // Free memory
    err = cudaFree(arrDevice);
    cuda_check(err, "cudaFree");
    err = cudaFree(tempDevice);
    cuda_check(err, "cudaFree");

    err = cudaEventDestroy(start_i);
    cuda_check(err, "cudaEventDestroy");
    err = cudaEventDestroy(stop_i);
    cuda_check(err, "cudaEventDestroy");

    return arrHost;
}



template<class T>
T* parallelMergeSortDescSM8(T* arr, int numels, float* elapsedRecords) {
    //allocation begin
    int size = smallestGreaterPowerOf2(numels);
    T* arrHost, * arrDevice, * tempDevice;

    arrHost = (T*)malloc(size * sizeof(T));
    cudaError_t err = cudaMalloc((void**)&arrDevice, size * sizeof(T));
    cuda_check(err, "cudaMalloc");
    err = cudaMalloc((void**)&tempDevice, size * sizeof(T));
    cuda_check(err, "cudaMalloc");

#pragma omp parallel for
    for (int i = 0; i < numels; ++i) {
        arrHost[i] = arr[i];
    }
    //add pad
    T min = std::numeric_limits<T>::min();
#pragma omp parallel for
    for (int i = numels; i < size; ++i) {
        arrHost[i] = min;
    }
    //allocation end





    // Launch the CUDA kernel
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    printf("Launching with blockSize: %d, numBlocks: %d, numels: %d, size: %d\n", blockSize, numBlocks, numels, size);

    // Copy the array from the host to the device
    err = cudaMemcpy(arrDevice, arrHost, size * sizeof(T), cudaMemcpyHostToDevice);
    cuda_check(err, "cudaMemcopy");

    int logn = int(log2(size));

    //create start and stop events
    cudaEvent_t start_i, stop_i;
    err = cudaEventCreate(&start_i);
    cuda_check(err, "cudaEventCreate");
    err = cudaEventCreate(&stop_i);
    cuda_check(err, "cudaEventCreate");

    // 0 to 7 iteration //
    //i start event
    err = cudaEventRecord(start_i);
    cuda_check(err, "cudaEventRecord");

    //launch kernel and wait
    mergeSortDescSM_Till128 << <numBlocks, blockSize >> > (arrDevice, size, logn);
    err = cudaDeviceSynchronize();
    cuda_check(err, "cudaDeviceSynchronize");

    //i stop event
    err = cudaEventRecord(stop_i);
    cuda_check(err, "cudaEventRecord");
    err = cudaEventSynchronize(stop_i);
    cuda_check(err, "cudaEventSynchronize");

    //get elapsed
    err = cudaEventElapsedTime(&elapsedRecords[0], start_i, stop_i);
    cuda_check(err, "cudaElapsedTime");


    int chunkSize = 256;
    for (int i = 8; i < logn; i++) {
        //i start event
        err = cudaEventRecord(start_i);
        cuda_check(err, "cudaEventRecord");

        //launch kernel and wait
        mergeSortDesc << <numBlocks, blockSize >> > (arrDevice, tempDevice, size, chunkSize);
        chunkSize *= 2;
        err = cudaDeviceSynchronize();
        cuda_check(err, "cudaDeviceSynchronize");

        //swap pointers
        T* temp = arrDevice;
        arrDevice = tempDevice;
        tempDevice = temp;

        //i stop event
        err = cudaEventRecord(stop_i);
        cuda_check(err, "cudaEventRecord");
        err = cudaEventSynchronize(stop_i);
        cuda_check(err, "cudaEventSynchronize");

        //get elapsed
        err = cudaEventElapsedTime(&elapsedRecords[i - 7], start_i, stop_i);
        cuda_check(err, "cudaElapsedTime");
    }


    // Copy the array from the device to the host
    err = cudaMemcpy(arrHost, arrDevice, size * sizeof(T), cudaMemcpyDeviceToHost);
    cuda_check(err, "cudaMemcopy");

    printf(" [ITERATION 0 to %i] %.8f ms elapsed\n", std::min(logn - 1, 7), elapsedRecords[0]);

    float totalElapsed = elapsedRecords[0];
    for (int i = 1; i < logn - 7; i++) {
        totalElapsed += elapsedRecords[i];
        printf(" [ITERATION %i] %.8f ms elapsed\n", i + 7, elapsedRecords[i]);
    }

    printf(" [TOTAL] %.8f ms elapsed\n", totalElapsed);


    err = cudaFree(arrDevice);
    cuda_check(err, "cudaFree");
    err = cudaFree(tempDevice);
    cuda_check(err, "cudaFree");

    err = cudaEventDestroy(start_i);
    cuda_check(err, "cudaEventDestroy");
    err = cudaEventDestroy(stop_i);
    cuda_check(err, "cudaEventDestroy");

    return arrHost;
}




template<class T>
T* parallelMergeSortDescSM9(T* arr, int numels, float* elapsedRecords) {
    //allocation begin
    int size = smallestGreaterPowerOf2(numels);
    T* arrHost, * arrDevice, * tempDevice;

    arrHost = (T*)malloc(size * sizeof(T));
    cudaError_t err = cudaMalloc((void**)&arrDevice, size * sizeof(T));
    cuda_check(err, "cudaMalloc");
    err = cudaMalloc((void**)&tempDevice, size * sizeof(T));
    cuda_check(err, "cudaMalloc");

#pragma omp parallel for
    for (int i = 0; i < numels; ++i) {
        arrHost[i] = arr[i];
    }
    //add pad
    T min = std::numeric_limits<T>::min();
#pragma omp parallel for
    for (int i = numels; i < size; ++i) {
        arrHost[i] = min;
    }
    //allocation end





    // Launch the CUDA kernel
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    printf("Launching with blockSize: %d, numBlocks: %d, numels: %d, size: %d\n", blockSize, numBlocks, numels, size);

    // Copy the array from the host to the device
    err = cudaMemcpy(arrDevice, arrHost, size * sizeof(T), cudaMemcpyHostToDevice);
    cuda_check(err, "cudaMemcopy");

    int logn = int(log2(size));

    //create start and stop events
    cudaEvent_t start_i, stop_i;
    err = cudaEventCreate(&start_i);
    cuda_check(err, "cudaEventCreate");
    err = cudaEventCreate(&stop_i);
    cuda_check(err, "cudaEventCreate");

    // 0 to 7 iteration //
    //i start event
    err = cudaEventRecord(start_i);
    cuda_check(err, "cudaEventRecord");

    //launch kernel and wait
    mergeSortDescSM_Till128 << <numBlocks, blockSize >> > (arrDevice, size, logn);
    err = cudaDeviceSynchronize();
    cuda_check(err, "cudaDeviceSynchronize");

    //i stop event
    err = cudaEventRecord(stop_i);
    cuda_check(err, "cudaEventRecord");
    err = cudaEventSynchronize(stop_i);
    cuda_check(err, "cudaEventSynchronize");

    //get elapsed
    err = cudaEventElapsedTime(&elapsedRecords[0], start_i, stop_i);
    cuda_check(err, "cudaElapsedTime");


    //256 chunk optimization
    if (8 < logn) {
        //i start event
        err = cudaEventRecord(start_i);
        cuda_check(err, "cudaEventRecord");

        //launch kernel and wait
        mergeSortDescSM256 << <numBlocks, blockSize >> > (arrDevice, tempDevice, size);
        err = cudaDeviceSynchronize();
        cuda_check(err, "cudaDeviceSynchronize");

        //swap pointers
        T* temp = arrDevice;
        arrDevice = tempDevice;
        tempDevice = temp;

        //i stop event
        err = cudaEventRecord(stop_i);
        cuda_check(err, "cudaEventRecord");
        err = cudaEventSynchronize(stop_i);
        cuda_check(err, "cudaEventSynchronize");

        //get elapsed
        err = cudaEventElapsedTime(&elapsedRecords[1], start_i, stop_i);
        cuda_check(err, "cudaElapsedTime");
    }

    int chunkSize = 512;
    for (int i = 9; i < logn; i++) {
        //i start event
        err = cudaEventRecord(start_i);
        cuda_check(err, "cudaEventRecord");

        //launch kernel and wait
        mergeSortDesc << <numBlocks, blockSize >> > (arrDevice, tempDevice, size, chunkSize);
        chunkSize *= 2;
        err = cudaDeviceSynchronize();
        cuda_check(err, "cudaDeviceSynchronize");

        //swap pointers
        T* temp = arrDevice;
        arrDevice = tempDevice;
        tempDevice = temp;

        //i stop event
        err = cudaEventRecord(stop_i);
        cuda_check(err, "cudaEventRecord");
        err = cudaEventSynchronize(stop_i);
        cuda_check(err, "cudaEventSynchronize");

        //get elapsed
        err = cudaEventElapsedTime(&elapsedRecords[i - 7], start_i, stop_i);
        cuda_check(err, "cudaElapsedTime");
    }


    // Copy the array from the device to the host
    err = cudaMemcpy(arrHost, arrDevice, size * sizeof(T), cudaMemcpyDeviceToHost);
    cuda_check(err, "cudaMemcopy");

    printf(" [ITERATION 0 to %i] %.8f ms elapsed\n", std::min(logn - 1, 7), elapsedRecords[0]);

    float totalElapsed = elapsedRecords[0];
    for (int i = 1; i < logn - 7; i++) {
        totalElapsed += elapsedRecords[i];
        printf(" [ITERATION %i] %.8f ms elapsed\n", i + 7, elapsedRecords[i]);
    }

    printf(" [TOTAL] %.8f ms elapsed\n", totalElapsed);


    err = cudaFree(arrDevice);
    cuda_check(err, "cudaFree");
    err = cudaFree(tempDevice);
    cuda_check(err, "cudaFree");

    err = cudaEventDestroy(start_i);
    cuda_check(err, "cudaEventDestroy");
    err = cudaEventDestroy(stop_i);
    cuda_check(err, "cudaEventDestroy");

    return arrHost;
}




template<class T>
T* parallelMergeSortDescSM(T* arr, int numels, int sharedMemorySize, float* elapsedRecords) {
    int maxiter = log2(sharedMemorySize / sizeof(T));
    //allocation begin
    int size = smallestGreaterPowerOf2(numels);
    T* arrHost, * arrDevice, * tempDevice;

    arrHost = (T*)malloc(size * sizeof(T));
    cudaError_t err = cudaMalloc((void**)&arrDevice, size * sizeof(T));
    cuda_check(err, "cudaMalloc");
    err = cudaMalloc((void**)&tempDevice, size * sizeof(T));
    cuda_check(err, "cudaMalloc");

#pragma omp parallel for
    for (int i = 0; i < numels; ++i) {
        arrHost[i] = arr[i];
    }
    //add pad
    T min = std::numeric_limits<T>::min();
#pragma omp parallel for
    for (int i = numels; i < size; ++i) {
        arrHost[i] = min;
    }
    //allocation end





    // Launch the CUDA kernel
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    printf("Launching with blockSize: %d, numBlocks: %d, numels: %d, size: %d\n", blockSize, numBlocks, numels, size);

    // Copy the array from the host to the device
    err = cudaMemcpy(arrDevice, arrHost, size * sizeof(T), cudaMemcpyHostToDevice);
    cuda_check(err, "cudaMemcopy");

    int logn = int(log2(size));

    //create start and stop events
    cudaEvent_t start_i, stop_i;
    err = cudaEventCreate(&start_i);
    cuda_check(err, "cudaEventCreate");
    err = cudaEventCreate(&stop_i);
    cuda_check(err, "cudaEventCreate");

    // 0 to 7 iteration //
    //i start event
    err = cudaEventRecord(start_i);
    cuda_check(err, "cudaEventRecord");

    //launch kernel and wait
    mergeSortDescSM_Till128 << <numBlocks, blockSize >> > (arrDevice, size, logn);
    err = cudaDeviceSynchronize();
    cuda_check(err, "cudaDeviceSynchronize");

    //i stop event
    err = cudaEventRecord(stop_i);
    cuda_check(err, "cudaEventRecord");
    err = cudaEventSynchronize(stop_i);
    cuda_check(err, "cudaEventSynchronize");

    //get elapsed
    err = cudaEventElapsedTime(&elapsedRecords[0], start_i, stop_i);
    cuda_check(err, "cudaElapsedTime");




    int chunkSize = 256;
    for (int i = 8; i < logn && i < maxiter; i++) {
        //i start event
        err = cudaEventRecord(start_i);
        cuda_check(err, "cudaEventRecord");

        //launch kernel and wait
        numBlocks = size / chunkSize;
        mergeSortDescSM_From256 << <numBlocks, blockSize, chunkSize * sizeof(T) >> > (arrDevice, tempDevice, size, chunkSize);
        chunkSize *= 2;
        err = cudaDeviceSynchronize();
        cuda_check(err, "cudaDeviceSynchronize");

        //swap pointers
        T* temp = arrDevice;
        arrDevice = tempDevice;
        tempDevice = temp;

        //i stop event
        err = cudaEventRecord(stop_i);
        cuda_check(err, "cudaEventRecord");
        err = cudaEventSynchronize(stop_i);
        cuda_check(err, "cudaEventSynchronize");

        //get elapsed
        err = cudaEventElapsedTime(&elapsedRecords[i - 7], start_i, stop_i);
        cuda_check(err, "cudaElapsedTime");
    }

    numBlocks = (size + blockSize - 1) / blockSize;
    for (int i = maxiter; i < logn; i++) {
        //i start event
        err = cudaEventRecord(start_i);
        cuda_check(err, "cudaEventRecord");

        //launch kernel and wait
        mergeSortDesc << <numBlocks, blockSize >> > (arrDevice, tempDevice, size, chunkSize);
        chunkSize *= 2;
        err = cudaDeviceSynchronize();
        cuda_check(err, "cudaDeviceSynchronize");

        //swap pointers
        T* temp = arrDevice;
        arrDevice = tempDevice;
        tempDevice = temp;

        //i stop event
        err = cudaEventRecord(stop_i);
        cuda_check(err, "cudaEventRecord");
        err = cudaEventSynchronize(stop_i);
        cuda_check(err, "cudaEventSynchronize");

        //get elapsed
        err = cudaEventElapsedTime(&elapsedRecords[i - 7], start_i, stop_i);
        cuda_check(err, "cudaElapsedTime");
    }


    // Copy the array from the device to the host
    err = cudaMemcpy(arrHost, arrDevice, size * sizeof(T), cudaMemcpyDeviceToHost);
    cuda_check(err, "cudaMemcopy");

    printf(" [ITERATION 0 to %i] %.8f ms elapsed\n", std::min(logn - 1, 7), elapsedRecords[0]);

    float totalElapsed = elapsedRecords[0];
    for (int i = 1; i < logn - 7; i++) {
        totalElapsed += elapsedRecords[i];
        printf(" [ITERATION %i] %.8f ms elapsed\n", i + 7, elapsedRecords[i]);
    }

    printf(" [TOTAL] %.8f ms elapsed\n", totalElapsed);

    err = cudaFree(arrDevice);
    cuda_check(err, "cudaFree");
    err = cudaFree(tempDevice);
    cuda_check(err, "cudaFree");

    err = cudaEventDestroy(start_i);
    cuda_check(err, "cudaEventDestroy");
    err = cudaEventDestroy(stop_i);
    cuda_check(err, "cudaEventDestroy");

    return arrHost;
}


void saveElapsed(float* elapsedRecords, int size, char* filename) {
    std::ofstream myfile = std::ofstream(filename);
    if (myfile.is_open())
    {
        for (int i = 0; i < size; i++) {
            myfile << elapsedRecords[i] << " ";
        }
        myfile.close();
    }
}


int main(int argc, char** argv) {
    
    if (argc != 2) {
        printf("Usage: %s <integer>\n", argv[0]);
        return 1;
    }
    int numels = atoi(argv[1]);
    
    Entity* arr = (Entity*)malloc(numels * sizeof(Entity));

    int logn = log2(smallestGreaterPowerOf2(numels));
    float* elapsedRecords = (float*)malloc(logn * sizeof(float));

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("sharedMemorySize: %.i\n", deviceProp.sharedMemPerBlock);

    // Generate random floats in the range [0, 1000]
    time_t seed = time(NULL);
    printf("seed: %lld\n", seed);
    srand(seed);



    for (int i = 0; i < numels; i++) {
        arr[i] = Entity(0, 0, 0, 0, ((float)rand() / RAND_MAX) * 1000.0);
        //arrHost[i] = sin(i*100 %8) + 1.0;
    }


    Entity* res = parallelMergeSortDesc(arr, numels, elapsedRecords);
    printf("-result check %s\n", (checkResult(res, arr, numels) ? "was SUCCESSFUL" : "FAILED"));
    saveElapsed(elapsedRecords, logn, "ElapsedRecords/NOSM.txt");

    printf("\n--- with shared memory optimization (8 iterations (max)) ---\n");
    // Free memory first
    free(res);
    res = parallelMergeSortDescSM8(arr, numels, elapsedRecords);
    printf("-result check %s\n", (checkResult(res, arr, numels) ? "was SUCCESSFUL" : "FAILED"));
    saveElapsed(elapsedRecords, logn - 7, "ElapsedRecords/SM8.txt");

    printf("\n--- with shared memory optimization (9 iterations (max)) ---\n");
    // Free memory first
    free(res);
    res = parallelMergeSortDescSM9(arr, numels, elapsedRecords);
    printf("-result check %s\n", (checkResult(res, arr, numels) ? "was SUCCESSFUL" : "FAILED"));
    saveElapsed(elapsedRecords, logn - 7, "ElapsedRecords/SM9.txt");

    int maxiterSM = log2(deviceProp.sharedMemPerBlock / sizeof(Entity));
    printf("\n--- with shared memory optimization (till iteration %i) ---\n", maxiterSM - 1);
    // Free memory first
    free(res);
    res = parallelMergeSortDescSM(arr, numels, deviceProp.sharedMemPerBlock, elapsedRecords);
    printf("-result check %s\n", (checkResult(res, arr, numels) ? "was SUCCESSFUL" : "FAILED"));
    saveElapsed(elapsedRecords, logn - 7, "ElapsedRecords/SM.txt");

    //sequential sort
    std::chrono::high_resolution_clock::time_point start, stop;
    std::chrono::duration<int, std::milli> duration;
    start = std::chrono::high_resolution_clock::now();
    Entity* arrcopy = (Entity*)malloc(numels * sizeof(Entity));
    memcpy(arrcopy, arr, numels * sizeof(Entity));
    sequentialMergeSort(arrcopy, numels);

    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    printf("\n--- sequential merge sort ---\n", maxiterSM - 1);
    printf(" [sequential merge sort] %i ms elapsed\n", duration.count());
    printf("-result check %s\n", (checkResult(arrcopy, arr, numels) ? "was SUCCESSFUL" : "FAILED"));
    float sequentialElapsed = duration.count() / 1.0;
    saveElapsed(&sequentialElapsed, 1, "ElapsedRecords/SEQUENTIAL.txt");


    // Free memory
    free(arrcopy);
    free(arr);
    free(res);

    free(elapsedRecords);

    return 0;
}