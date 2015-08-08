
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "device_functions.cuh"


////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////

// Complex addition
static __device__ __host__ inline cufftComplex ComplexAdd(cufftComplex a, cufftComplex b)
{
	cufftComplex c;
	c.x = a.x + b.x;
	c.y = a.y + b.y;
	return c;
}

// Complex scale
static __device__ __host__ inline cufftComplex ComplexScale(cufftComplex a, float s)
{
	cufftComplex c;
	c.x = s * a.x;
	c.y = s * a.y;
	return c;
}

// Complex multiplication
static __device__ __host__ inline cufftComplex ComplexMul(cufftComplex a, cufftComplex b)
{
	cufftComplex c;
	c.x = a.x * b.x - a.y * b.y;
	c.y = a.x * b.y + a.y * b.x;
	return c;
}

// Complex pointwise multiplication
static __global__ void ComplexPointwiseMulAndScale(cufftComplex *a, const cufftComplex *b, int size, float scale)
{
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = threadID; i < size; i += numThreads)
	{
		a[i] = ComplexScale(ComplexMul(a[i], b[i]), scale);
	}
}


static __global__ void ComplexMultiplyMono(cufftComplex *out, const cufftComplex *ir, const cufftComplex *in, int ir_sz, int in_sz) {

	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = threadID; i < ir_sz; i += numThreads) {
		out[i].x = ir[i].x * in[i % in_sz].x - ir[i].y * in[i % in_sz].y;
		out[i].y = ir[i].x * in[i % in_sz].y + ir[i].y * in[i % in_sz].x;
	}

}

static __global__ void ComplexMultiplyStereo(cufftComplex *out_l, const cufftComplex *ir_l, const cufftComplex *in_l,
	cufftComplex *out_r, const cufftComplex *ir_r, const cufftComplex *in_r, int ir_sz, int in_sz, int trueStereo) {

	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = threadID; i < ir_sz; i += numThreads) {

		//L-L
		out_l[i].x = ir_l[i].x * in_l[i % in_sz].x - ir_l[i].y * in_l[i % in_sz].y;
		out_l[i].y = ir_l[i].x * in_l[i % in_sz].y + ir_l[i].y * in_l[i % in_sz].x;
	
		if (trueStereo == 1) {
			//L-R
			out_l[i].x = ir_r[i].x * in_l[i % in_sz].x - ir_r[i].y * in_l[i % in_sz].y;
			out_l[i].y = ir_r[i].x * in_l[i % in_sz].y + ir_r[i].y * in_l[i % in_sz].x;

			out_l[i].x /= 2;
			out_l[i].y /= 2;

			//R-L
			out_r[i].x = ir_l[i].x * in_r[i % in_sz].x - ir_l[i].y * in_r[i % in_sz].y;
			out_r[i].y = ir_l[i].x * in_r[i % in_sz].y + ir_l[i].y * in_r[i % in_sz].x;

			//R-R
			out_r[i].x = ir_r[i].x * in_r[i % in_sz].x - ir_r[i].y * in_r[i % in_sz].y;
			out_r[i].y = ir_r[i].x * in_r[i % in_sz].y + ir_r[i].y * in_r[i % in_sz].x;

			out_r[i].x /= 2;
			out_r[i].y /= 2;

		}
		else {
			//Quasi stereo
			//R-L
			out_r[i].x = ir_l[i].x * in_r[i % in_sz].x - ir_l[i].y * in_r[i % in_sz].y;
			out_r[i].y = ir_l[i].x * in_r[i % in_sz].y + ir_l[i].y * in_r[i % in_sz].x;

		}
	}

}


//Overlap & Add
static __global__ void OverlapAdd(cufftReal *dst, int dst_sz, const cufftReal *src, int src_sz, int M, int N, int odd) {

	//odd == 1 then do odd overlap & add, else do even
	const int numThreads = blockDim.x * gridDim.x;
	//offset, only significant when odd = 1; because odd blocks are shifted to the right by size(offset)
	const int dst_offset = M * odd;
	const int blockID = blockIdx.x * blockDim.x;
	const int threadID = blockID + threadIdx.x;

	const int ind_dst = threadID + dst_offset;
	const int ind_src = (2 * (blockID / N) + odd) * N + threadID - (threadID / N) * N;
	
	
	//dst size  = (IR_blocks - 1) * M + N
	//srr size = IR_blocks * N
	
	//OLA with scaling
	if (threadID < src_sz / 2)
		dst[ind_dst] += src[ind_src] / N;
	
}

//Copy cache
static __global__ void BackupCache(cufftReal *dst, cufftReal *src, int count, int size) {

	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadID < size) {
		if (threadID < count) {
			dst[threadID] = src[threadID];
		}
		else {
			dst[threadID] = 0.0f;
		}
	}

}

static __global__ void Normalize(cufftReal *vector, cufftReal max, int size) {

	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = threadID; i < size; i += numThreads) {
		vector[i] /= max;
	}

}



void ComplexMultiplyMono(dim3 gridDim, dim3 blockDim, cufftComplex *out, const cufftComplex *ir, const cufftComplex *in, int ir_sz, int in_sz) {

	ComplexMultiplyMono << <gridDim, blockDim >> >(out, ir, in, ir_sz, in_sz);

}
void ComplexMultiplyStereo(dim3 gridDim, dim3 blockDim, cufftComplex *out_l, const cufftComplex *ir_l, const cufftComplex *in_l,
	cufftComplex *out_r, const cufftComplex *ir_r, const cufftComplex *in_r, int ir_sz, int in_sz, int trueStereo) {

	ComplexMultiplyStereo<<<gridDim, blockDim>>>(out_l, ir_l, in_l, out_r, ir_r, in_r, ir_sz, in_sz, trueStereo);

}

void OverlapAdd(dim3 gridDim, dim3 blockDim, cufftReal *dst, int dst_sz, const cufftReal *src, int src_sz, int M, int N, int odd) {

	OverlapAdd<<<gridDim, blockDim>>>(dst, dst_sz, src, src_sz, M, N, odd);

}
void BackupCache(dim3 gridDim, dim3 blockDim, cufftReal *dst, cufftReal *src, int count, int size) {

	BackupCache<<<gridDim, blockDim>>>(dst, src, count, size);

}
void Normalize(dim3 gridDim, dim3 blockDim, cufftReal *vector, cufftReal max, int size) {

	Normalize<<<gridDim, blockDim>>>(vector, max, size);

}