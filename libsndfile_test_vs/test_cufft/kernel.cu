//WORKS ONLY FOR MONO IN AND IR, for now...


//CUDA INCLUDES
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>
#include <cufft.h>
#include <cufftw.h>
#include <thrust\extrema.h>

#include <sndfile.hh>

#include <iostream>
#include <iomanip>

#include <algorithm>

using namespace std;

#define L 1024
#define M 1024
#define N 2048
#define ACTUAL_N (N / 2 + 1)

#define BLOCK_SIZE 256

// Complex data type
static __device__ __host__ inline cufftComplex ComplexAdd(cufftComplex, cufftComplex);
static __device__ __host__ inline cufftComplex ComplexScale(cufftComplex, float);
static __device__ __host__ inline cufftComplex ComplexMul(cufftComplex, cufftComplex);
static __global__ void ComplexPointwiseMulAndScale(cufftComplex *, const cufftComplex *, int, float);

// try to realize FindMax correctly, to avoid excess data transfers and use of std::max_element
static __global__ void FindMax(cufftReal *, cufftReal *, int);
static __global__ void NormalizeVector(cufftReal *, cufftReal, int);


int main()
{

	cudaEvent_t start, stop;
	float elapsed_time = 0.0f;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
#pragma region init_files
	SndfileHandle* in = new SndfileHandle("D:/Documents/GitHub/Diplomski/libsndfile_test_vs/test_cufft/wav_files/guitar_mono.wav");

	SndfileHandle* ir = new SndfileHandle("D:/Documents/GitHub/Diplomski/libsndfile_test_vs/test_cufft/wav_files/LongEchoHallIRMono.wav");

	SndfileHandle* out = new SndfileHandle("D:/Documents/GitHub/Diplomski/libsndfile_test_vs/test_cufft/wav_files/out_guit.wav",
		SFM_WRITE, SF_FORMAT_WAV | SF_FORMAT_PCM_16, 1, 44100);
	out->command(SFC_SET_UPDATE_HEADER_AUTO, NULL, SF_TRUE);

	sf_count_t in_sz = in->frames();
	sf_count_t ir_sz = ir->frames();
#pragma endregion

//simple convolution.. no OLA
// this will have to change, OLA is required for long in files...
// but unlike the sequential version, whole ir can be used at once!
#pragma region setFFTParameters
	int FFT_SIZE = ir_sz + in_sz - 1;
	FFT_SIZE--;
	FFT_SIZE |= FFT_SIZE >> 1;
	FFT_SIZE |= FFT_SIZE >> 2;
	FFT_SIZE |= FFT_SIZE >> 4;
	FFT_SIZE |= FFT_SIZE >> 8;
	FFT_SIZE |= FFT_SIZE >> 16;
	FFT_SIZE++;

	cout << "FFT_SIZE: " << FFT_SIZE << endl;
#pragma endregion


	cudaError_t cudaStatus;
	cufftResult_t cufftStatus;

	float *ir_mono, *in_mono;
	cufftReal *in_dev;
	cufftComplex *IR_DEV, *IN_DEV, *OUT_DEV;

	cufftHandle dfft, ifft;

	ir_mono = new float[FFT_SIZE];
	memset(ir_mono, 0, sizeof(float)* FFT_SIZE);
	ir->readf(ir_mono, ir_sz);

	in_mono = new float[FFT_SIZE];
	memset(in_mono, 0, sizeof(float)* FFT_SIZE);
	in->readf(in_mono, in_sz);

#pragma region deviceBufferAllocation
	cudaStatus = cudaMalloc((void**)&in_dev, sizeof(cufftReal)* FFT_SIZE);
	if (cudaStatus != cudaSuccess) {
		cerr << "cudaMalloc(in_dev) failed!" << endl;
		return -1;
	}

	cudaStatus = cudaMalloc((void**)&IR_DEV, sizeof(cufftComplex)* FFT_SIZE);
	if (cudaStatus != cudaSuccess) {
		cerr << "cudaMalloc(IR_DEV) failed!" << endl;
		return -1;
	}
	cudaStatus = cudaMalloc((void**)&IN_DEV, sizeof(cufftComplex)* FFT_SIZE);
	if (cudaStatus != cudaSuccess) {
		cerr << "cudaMalloc(IN_DEV) failed!" << endl;
		return -1;
	}
	cudaStatus = cudaMalloc((void**)&OUT_DEV, sizeof(cufftComplex)* FFT_SIZE);
	if (cudaStatus != cudaSuccess) {
		cerr << "cudaMalloc(OUT_DEV) failed!" << endl;
		return -1;
	}
#pragma endregion

#pragma region planMake
	cufftStatus = cufftPlan1d(&dfft, FFT_SIZE, CUFFT_R2C, 1);
	cufftStatus = cufftPlan1d(&ifft, FFT_SIZE, CUFFT_C2R, 1);
	if (cufftStatus != cudaSuccess) {
		cerr << "cufftPlan1d failed!" << endl;
		return -1;
	}
#pragma endregion
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsed_time, start, stop);
	cout << "Initialization time: " << setprecision(5) << elapsed_time << "[ms]" << endl;

	cudaEventRecord(start);
	//set IR_DEV
#pragma region set IR_DEV
	cudaStatus = cudaMemcpy(in_dev, ir_mono, sizeof(cufftReal)* FFT_SIZE, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		cerr << "cudaMemcpy failed!" << endl;
		return -1;
	}
	cufftStatus = cufftExecR2C(dfft, in_dev, IR_DEV);
	if (cufftStatus != cudaSuccess) {
		cerr << "cufftExecR2C failed!" << endl;
		return -1;
	}
#pragma endregion
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsed_time, start, stop);
	cout << "IR init time: " << setprecision(5) << elapsed_time << "[ms]" << endl;

	//set IN_DEV
#pragma region set IN_DEV
	cudaStatus = cudaMemcpy(in_dev, in_mono, sizeof(cufftReal)* FFT_SIZE, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		cerr << "cudaMemcpy failed!" << endl;
		return -1;
	}
	cufftStatus = cufftExecR2C(dfft, in_dev, IN_DEV);
	if (cufftStatus != cudaSuccess) {
		cerr << "cufftExecR2C failed!" << endl;
		return -1;
	}
#pragma endregion

	dim3 gridDim(FFT_SIZE / BLOCK_SIZE + 1);

	cudaEventRecord(start);
	ComplexPointwiseMulAndScale<<<gridDim, BLOCK_SIZE >>>(IN_DEV, IR_DEV, FFT_SIZE, 1.0f / FFT_SIZE);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsed_time, start, stop);
	cout << "Complex multiplication time: " << setprecision(5) << elapsed_time << "[ms]" << endl;


	// Transform signal back
	cufftStatus = cufftExecC2R(ifft, IN_DEV, in_dev);



	cudaEventRecord(start);
	
	float* out_mono = new float[FFT_SIZE];
	memset(out_mono, 0, sizeof(float)* FFT_SIZE);

	// Copy device memory to host
	cudaStatus = cudaMemcpy(out_mono, in_dev, sizeof(cufftReal)* FFT_SIZE, cudaMemcpyDeviceToHost);

	float *max_ind;
	max_ind = max_element(out_mono, out_mono + FFT_SIZE - 1);
	float max = *max_ind;
	
	NormalizeVector<<<gridDim, BLOCK_SIZE>>>(in_dev, max, FFT_SIZE);

	// Copy device memory to host
	cudaStatus = cudaMemcpy(out_mono, in_dev, sizeof(cufftReal)* FFT_SIZE, cudaMemcpyDeviceToHost);
	
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsed_time, start, stop);
	cout << "Normalization time: " << setprecision(5) << elapsed_time << "[ms]" << endl;

	out->writef(out_mono, in_sz + ir_sz - 1);

	cudaDeviceReset();

	return 0;
}


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

//Find maximum element
static __global__ void FindMax(cufftReal *vector, cufftReal *max, int size) {

	__shared__ cufftReal maxElements[BLOCK_SIZE];

	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadID < size)
		maxElements[threadIdx.x] = vector[threadID];
	else
		maxElements[threadIdx.x] = 0.0f;

	__syncthreads();

	for (int i = BLOCK_SIZE / 2; i > 0; i >>= 1) {
		if (threadIdx.x < i)
			maxElements[threadIdx.x] = maxElements[threadIdx.x] > maxElements[threadIdx.x + i]
									? maxElements[threadIdx.x] : maxElements[threadIdx.x + i];
	}

	__syncthreads();
	
	if (threadIdx.x == 0)
		max[blockIdx.x] = maxElements[0];

}

// Vector normalization
static __global__ void NormalizeVector(cufftReal *vector, cufftReal max, int size) {

	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = threadID; i < size; i += numThreads) {
		vector[i] = vector[i] / max;
	}


}