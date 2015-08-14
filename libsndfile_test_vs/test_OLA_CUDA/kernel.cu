
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
//static __device__ __host__ inline cufftComplex ComplexAdd(cufftComplex a, cufftComplex b)
//{
//	cufftComplex c;
//	c.x = a.x + b.x;
//	c.y = a.y + b.y;
//	return c;
//}

// Complex scale
//static __device__ __host__ inline cufftComplex ComplexScale(cufftComplex a, float s)
//{
//	cufftComplex c;
//	c.x = s * a.x;
//	c.y = s * a.y;
//	return c;
//}

// Complex multiplication
//static __device__ __host__ inline cufftComplex ComplexMul(cufftComplex a, cufftComplex b)
//{
//	cufftComplex c;
//	c.x = a.x * b.x - a.y * b.y;
//	c.y = a.x * b.y + a.y * b.x;
//	return c;
//}

// Complex pointwise multiplication
//static __global__ void ComplexPointwiseMulAndScale(cufftComplex *a, const cufftComplex *b, int size, float scale)
//{
//	const int numThreads = blockDim.x * gridDim.x;
//	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
//
//	for (int i = threadID; i < size; i += numThreads)
//	{
//		a[i] = ComplexScale(ComplexMul(a[i], b[i]), scale);
//	}
//}


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

	cuComplex local_in_l, local_in_r;
	cuComplex local_ir_l, local_ir_r;
	cuComplex local_out_l, local_out_r;

	for (int i = threadID; i < ir_sz; i += numThreads) {
		
		local_in_l = in_l[i % in_sz];
		local_in_r = in_r[i % in_sz];
		local_ir_l = ir_l[i];
		if (trueStereo == 1)
			local_ir_r = ir_r[i];

		//too much overhead due to GMEM accessing,
		//allocate local storage for in_l/in_r and out_l/out_r and after calculation perform writeout

		//L-L
		local_out_l.x = local_ir_l.x * local_in_l.x - local_ir_l.y * local_in_l.y;
		local_out_l.y = local_ir_l.x * local_in_l.y + local_ir_l.y * local_in_l.x;
	
		if (trueStereo == 1) {
			//L-R
			local_out_l.x = local_ir_r.x * local_in_l.x - local_ir_r.y * local_in_l.y;
			local_out_l.y = local_ir_r.x * local_in_l.y + local_ir_r.y * local_in_l.x;

			local_out_l.x /= 2;
			local_out_l.y /= 2;

			//R-L
			local_out_r.x = local_ir_l.x * local_in_r.x - local_ir_l.y * local_in_r.y;
			local_out_r.y = local_ir_l.x * local_in_r.y + local_ir_l.y * local_in_r.x;

			//R-R
			local_out_r.x = local_ir_r.x * local_in_r.x - local_ir_r.y * local_in_r.y;
			local_out_r.y = local_ir_r.x * local_in_r.y + local_ir_r.y * local_in_r.x;

			local_out_r.x /= 2;
			local_out_r.y /= 2;

		}
		else {
			//Quasi stereo
			//R-L
			local_out_r.x = local_ir_l.x * local_in_r.x - local_ir_l.y * local_in_r.y;
			local_out_r.y = local_ir_l.x * local_in_r.y + local_ir_l.y * local_in_r.x;

		}

		out_l[i] = local_out_l;
		out_r[i] = local_out_r;
	}

}


//Overlap & Add
static __global__ void OverlapAdd(cufftReal *dst, int dst_sz, const cufftReal *src, int src_sz, int M, int N, int odd) {

	//odd == 1 then do odd overlap & add, else do even
	//offset, only significant when odd = 1; because odd blocks are shifted to the right by size(offset)
	const int dst_offset = M * odd;
	const int blockID = blockIdx.x * blockDim.x;
	const int threadID = blockID + threadIdx.x;

	const int ind_dst = threadID + dst_offset;
	const int ind_src = (2 * (threadID / N) + odd) * N + threadID - (threadID / N) * N;
		
	//OLA with scaling
	if (ind_src < src_sz)
		dst[ind_dst] += src[ind_src] / N;
	
}

//Backup cache
static __global__ void BackupCache(cufftReal *dst, cufftReal *src, int count, int size) {

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

//static __global__ void Normalize(cufftReal *vector, cufftReal max, int size) {
//
//	const int numThreads = blockDim.x * gridDim.x;
//	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
//
//	for (int i = threadID; i < size; i += numThreads) {
//		vector[i] /= max;
//	}
//
//}



void ComplexMultiplyMono(dim3 gridDim, dim3 blockDim, cufftComplex *out, const cufftComplex *ir, const cufftComplex *in, int ir_sz, int in_sz) {

	ComplexMultiplyMono << <gridDim, blockDim >> >(out, ir, in, ir_sz, in_sz);

}
void ComplexMultiplyStereo(dim3 gridDim, dim3 blockDim, cufftComplex *out_l, const cufftComplex *ir_l, const cufftComplex *in_l,
	cufftComplex *out_r, const cufftComplex *ir_r, const cufftComplex *in_r, int ir_sz, int in_sz, int trueStereo) {

	ComplexMultiplyStereo<<<gridDim, blockDim>>>(out_l, ir_l, in_l, out_r, ir_r, in_r, ir_sz, in_sz, trueStereo);

}

void OverlapAdd(dim3 gridDim, dim3 blockDim, cufftReal *dst, int dst_sz, const cufftReal *src, int src_sz, int M, int N, int odd, cudaStream_t stream) {

	if (stream != NULL) {
		OverlapAdd<<<gridDim, blockDim, 0, stream>>>(dst, dst_sz, src, src_sz, M, N, odd);

	}
	else {
		OverlapAdd<<<gridDim, blockDim>>>(dst, dst_sz, src, src_sz, M, N, odd);
	}

}
void BackupCache(dim3 gridDim, dim3 blockDim, cufftReal *dst, cufftReal *src, int count, int size, cudaStream_t stream) {

	if (stream != NULL) {
		BackupCache<<<gridDim, blockDim, 0, stream>>>(dst, src, count, size);
	}
	else {
		BackupCache<<<gridDim, blockDim>>>(dst, src, count, size);
	}
}
//void Normalize(dim3 gridDim, dim3 blockDim, cufftReal *vector, cufftReal max, int size) {
//
//	Normalize<<<gridDim, blockDim>>>(vector, max, size);
//
//}