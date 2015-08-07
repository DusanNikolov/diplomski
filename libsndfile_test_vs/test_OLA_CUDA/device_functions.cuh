
#ifndef DEVICE_FUNCTIONS_H
#define DEVICE_FUNCTIONS_H

// includes, project
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>

// Complex data type

static __device__ __host__ inline cufftComplex ComplexAdd(cufftComplex, cufftComplex);
static __device__ __host__ inline cufftComplex ComplexScale(cufftComplex, float);
static __device__ __host__ inline cufftComplex ComplexMul(cufftComplex, cufftComplex);
static __global__ void ComplexPointwiseMulAndScale(cufftComplex *, const cufftComplex *, int, float);

static __global__ void Normalize(cufftReal *, cufftReal, int size);

#endif