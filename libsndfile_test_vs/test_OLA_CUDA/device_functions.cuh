
#ifndef DEVICE_FUNCTIONS_H
#define DEVICE_FUNCTIONS_H

// includes, project
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>

// Complex data type

//static __device__ __host__ inline cufftComplex ComplexAdd(cufftComplex, cufftComplex);
//static __device__ __host__ inline cufftComplex ComplexScale(cufftComplex, float);
//static __device__ __host__ inline cufftComplex ComplexMul(cufftComplex, cufftComplex);

//static __global__ void ComplexPointwiseMulAndScale(cufftComplex *, const cufftComplex *, int, float);

static __global__ void ComplexMultiplyMono(cufftComplex *, const cufftComplex *, const cufftComplex *, int, int);
static __global__ void ComplexMultiplyStereo(cufftComplex *, const cufftComplex *, const cufftComplex *,
	cufftComplex *, const cufftComplex *, const cufftComplex *, int, int, int);

static __global__ void OverlapAdd(cufftReal *, int, const cufftReal *, int, int, int, int);
static __global__ void BackupCache(cufftReal *, cufftReal *, int, int);
static __global__ void Normalize(cufftReal *, cufftReal, int);

void ComplexMultiplyMono(dim3, dim3, cufftComplex *, const cufftComplex *, const cufftComplex *, int, int);
void ComplexMultiplyStereo(dim3, dim3, cufftComplex *, const cufftComplex *, const cufftComplex *,
	cufftComplex *, const cufftComplex *, const cufftComplex *, int, int, int);

void OverlapAdd(dim3, dim3, cufftReal *, int, const cufftReal *, int, int, int, int, cudaStream_t);
void BackupCache(dim3, dim3, cufftReal *, cufftReal *, int, int, cudaStream_t);
void Normalize(dim3, dim3, cufftReal *, cufftReal, int);

#endif