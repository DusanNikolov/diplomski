//Method definitions of CUDAReverbEffect class

#include "CUDAReverbEffect.h"
#include "MonoStereoConversion.h"
#include "device_functions.cuh"

#include <iostream>
using namespace std;

CUDAReverbEffect::CUDAReverbEffect(char *in_fn, char *ir_fn, char *out_fn) {

	QueryPerformanceFrequency(&frequency);

	initialize(in_fn, ir_fn, out_fn);

}

CUDAReverbEffect::~CUDAReverbEffect() {
	
	//delete cache; delete cache_l; delete cache_r;
	
	if (STEREO == channels) {
		delete in_stereo; delete in_l; delete in_r;
		delete out_stereo; delete out_l; delete out_r;
	}
	else {
		delete in_l;
		delete out_l;
	}

	if (STEREO == ir->channels()) {
		delete ir_stereo; delete ir_l; delete ir_r;

	}
	else {
		delete ir_l;
	}

	//fftwf_destroy_plan(DFFT);
	//fftwf_destroy_plan(IFFT);

	//fftwf_free(IR_R);
	//fftwf_free(IR_L);
	//fftwf_free(IN_R);
	//fftwf_free(IN_L);
	//fftwf_free(OUT_SRC_R);
	//fftwf_free(OUT_SRC_L);

	delete in_src_l; delete in_src_r;

	delete in; delete ir; delete out;

}

void CUDAReverbEffect::initialize(char *in_fn, char *ir_fn, char *out_fn) {
	LARGE_INTEGER start_init_files, end_init_files;
	LARGE_INTEGER start_init_fftws, end_init_fftws;
	LARGE_INTEGER start_init_inout, end_init_inout;
	LARGE_INTEGER start_init_ir, end_init_ir;
	double eTime_files, eTime_fftws, eTime_inout, eTime_ir;

	//QueryPerformanceFrequency(&frequency);

	QueryPerformanceCounter(&start_init_files);
	init_files(in_fn, ir_fn, out_fn);
	QueryPerformanceCounter(&end_init_files);
	
	QueryPerformanceCounter(&start_init_fftws);
	init_fftws();
	QueryPerformanceCounter(&end_init_fftws);
	
	QueryPerformanceCounter(&start_init_inout);
	//initialize input/output buffers
	if (STEREO == channels) {
		init_in_out_stereo();
	}
	else {
		init_in_out_mono();
	}
	QueryPerformanceCounter(&end_init_inout);

	QueryPerformanceCounter(&start_init_ir);
	//initialize ir buffers & perform FFT(ir)
	if (STEREO == ir->channels()) {
		init_ir_stereo();
	}
	else {
		init_ir_mono();
	}
	QueryPerformanceCounter(&end_init_ir);

	eTime_files = (end_init_files.QuadPart - start_init_files.QuadPart) * 1000.0 / frequency.QuadPart;
	eTime_fftws = (end_init_fftws.QuadPart - start_init_fftws.QuadPart) * 1000.0 / frequency.QuadPart;
	eTime_inout = (end_init_inout.QuadPart - start_init_inout.QuadPart) * 1000.0 / frequency.QuadPart;
	eTime_ir = (end_init_ir.QuadPart - start_init_ir.QuadPart) * 1000.0 / frequency.QuadPart;

	cout << "Files initialization time: " << eTime_files << "[ms]" << endl
		<< "FFTWS initialization time: " << eTime_fftws << "[ms]" << endl
		<< "In/Out initialization time: " << eTime_inout << "[ms]" << endl
		<< "IR initialization time: " << eTime_ir << "[ms]" << endl;


}

void CUDAReverbEffect::applyReverb() {

	if (STEREO == channels) {
		OLA_stereo();
	}
	else {
		OLA_mono();
	}

}

void CUDAReverbEffect::writeOutNormalized() {

	//needs some tidying-up

	if (STEREO == channels) {
		float scale_l = 1 / max_l,
			scale_r = 1 / max_r;
		for (long i = 0; i < in->frames() + ir->frames() - 1; i++) {
			out_l[i] *= scale_l;
			out_r[i] *= scale_r;
		}
		MonoStereoConversion::combine2Channels(out_l, out_r, out_stereo, in->frames() + ir->frames() - 1, &max);

		out->writef(out_stereo, in->frames() + ir->frames() - 1);
	}
	else {
		float scale = 1 / max;
		for (long i = 0; i < in->frames() + ir->frames() - 1; i++)
			out_l[i] *= scale;

		out->writef(out_l, in->frames() + ir->frames() - 1);
	}

}

bool CUDAReverbEffect::OLA_mono() {

	max = 0.0f;

	for (long i = 0; i < in->frames(); i += L) {
	
		//FFT input block
		if (i + L > in->frames()) {
			DFT(in_l + i, (in->frames() - i), in_dev_l, IN_L, N);
		}
		else {
			DFT(in_l + i, L, in_dev_l, IN_L, N);
		}
	
		//complex multiply whole IR with this IN block
		ComplexMultiplyMono(gridDim, BLOCK_SIZE, OUT_SRC_L, IR_L, IN_L, IR_blocks * (N / 2 + 1), (N / 2 + 1));
		//perform batched IFFT from OUT_SRC_L to cache_padded_l
		IFT();
		//move cache_padded_l to cache_l, return first L samples to host, and shift cache_l L samples to the left
		//do even blocks first
		OverlapAdd(gridDim, BLOCK_SIZE, cache_l, (IR_blocks - 1) * M + N, cache_padded_l, IR_blocks * N, M, N, 0);
		//do odd blocks next shifted by M to the right
		OverlapAdd(gridDim, BLOCK_SIZE, cache_l, (IR_blocks - 1) * M + N, cache_padded_l, IR_blocks * N, M, N, 1);
	
		//extract first L elements from the cache_l begining, shift cache_l << L

		if (i + L > in_sz) {
			cudaMemcpy(out_l + i, cache_l, sizeof(float)* (out_sz - i), cudaMemcpyDeviceToHost);
		}
		else {
			cudaMemcpy(out_l + i, cache_l, sizeof(float)* L, cudaMemcpyDeviceToHost);
			
			BackupCache(gridDim, BLOCK_SIZE, temp_cache_l, cache_l + L, (IR_blocks - 1) * M + N - L, (IR_blocks - 1) * M + N);
			BackupCache(gridDim, BLOCK_SIZE, cache_l, temp_cache_l, (IR_blocks - 1) * M + N, (IR_blocks - 1) * M + N);
		}

	}

	//this should also be parallelized (reduction on GPU?)
	max = 0.0f;
	for (int i = 0; i < out_sz; i++) {
		if (fabs(out_l[i]) > max)
			max = fabs(out_l[i]);
	}

	return true;
	
}

bool CUDAReverbEffect::OLA_stereo() {

	max_l = max_r = 0.0f;

	for (long i = 0; i < in->frames(); i += L) {

		//FFT input block
		if (i + L > in->frames()) {
			DFT(in_l + i, (in->frames() - i), in_dev_l, IN_L, N);
			DFT(in_r + i, (in->frames() - i), in_dev_r, IN_R, N);
		}
		else {
			DFT(in_l + i, L, in_dev_l, IN_L, N);
			DFT(in_r + i, L, in_dev_r, IN_R, N);
		}

		//complex multiply whole IR with this IN block
		if (STEREO == ir->channels()) {
			ComplexMultiplyStereo(gridDim, BLOCK_SIZE, OUT_SRC_L, IR_L, IN_L,
				OUT_SRC_R, IR_R, IN_R, IR_blocks * (N / 2 + 1), (N / 2 + 1), 1);
		}
		else {
			ComplexMultiplyStereo(gridDim, BLOCK_SIZE, OUT_SRC_L, IR_L, IN_L,
				NULL, NULL, NULL, IR_blocks * (N / 2 + 1), (N / 2 + 1), 0);
		}

		//perform batched IFFT from OUT_SRC to cache_padded
		IFT();
		//move cache_padded_l to cache_l, return first L samples to host, and shift cache_l L samples to the left
		//do even blocks first
		//left channel
		OverlapAdd(gridDim, BLOCK_SIZE, cache_l, (IR_blocks - 1) * M + N, cache_padded_l, IR_blocks * N, M, N, 0);
		//right channel
		OverlapAdd(gridDim, BLOCK_SIZE, cache_r, (IR_blocks - 1) * M + N, cache_padded_r, IR_blocks * N, M, N, 0);
		//do odd blocks next shifted by M to the right
		//left channel
		OverlapAdd(gridDim, BLOCK_SIZE, cache_l, (IR_blocks - 1) * M + N, cache_padded_l, IR_blocks * N, M, N, 1);
		//right channel
		OverlapAdd(gridDim, BLOCK_SIZE, cache_r, (IR_blocks - 1) * M + N, cache_padded_r, IR_blocks * N, M, N, 1);

		//extract first L elements from the cache_l begining, shift cache_l << L

		if (i + L > in_sz / 2) {
			cudaMemcpy(out_l + i, cache_l, sizeof(float)* (out_sz / 2 - i), cudaMemcpyDeviceToHost);
			cudaMemcpy(out_r + i, cache_r, sizeof(float)* (out_sz / 2 - i), cudaMemcpyDeviceToHost);
		}
		else {
			cudaMemcpy(out_l + i, cache_l, sizeof(float)* L, cudaMemcpyDeviceToHost);
			cudaMemcpy(out_r + i, cache_r, sizeof(float)* L, cudaMemcpyDeviceToHost);

			//shift left channel
			BackupCache(gridDim, BLOCK_SIZE, temp_cache_l, cache_l + L, (IR_blocks - 1) * M + N - L, (IR_blocks - 1) * M + N);
			BackupCache(gridDim, BLOCK_SIZE, cache_l, temp_cache_l, (IR_blocks - 1) * M + N, (IR_blocks - 1) * M + N);
		
			//shift right channel
			BackupCache(gridDim, BLOCK_SIZE, temp_cache_r, cache_r + L, (IR_blocks - 1) * M + N - L, (IR_blocks - 1) * M + N);
			BackupCache(gridDim, BLOCK_SIZE, cache_r, temp_cache_r, (IR_blocks - 1) * M + N, (IR_blocks - 1) * M + N);
		}

	}

	//this should also be parallelized (reduction on GPU?)
	for (int i = 0; i < out_sz / 2; i++) {
		if (fabs(out_l[i]) > max_l)
			max_l = fabs(out_l[i]);
		if (fabs(out_r[i]) > max_r)
			max_r = fabs(out_r[i]);
	}

	return true;

}

bool CUDAReverbEffect::DFT(float *in_host, long in_len, float *in_dev, cufftComplex *OUT_DEV, int fft_size) {

	//first: copy & pad data to the in_host buffer
	//second: transfer in_host to in_dev buffer via cudaMemcpy
	//third: perform fft to the desired dev_out buffer
	//REDUNDANT: first step. maybe better to memset in_dev to 0, and just cudaMemcpy to in_dev

	cudaError_t cudaStatus;
	cufftResult_t cufftStatus;

	cudaStatus = cudaMemset(in_dev, 0, sizeof(cufftReal)* fft_size);
	if (cudaStatus != cudaSuccess) {
		cerr << "cudaMemset(in_dev) failed!" << endl;
		return false;
	}
	cudaStatus = cudaMemcpy(in_dev, in_host, sizeof(cufftReal)* in_len, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		cerr << "cudaMemcpy(in_dev, in_host...) failed!" << endl;
		return false;
	}

	cufftStatus = cufftExecR2C(inDFFT, in_dev, OUT_DEV);
	if (cufftStatus != CUFFT_SUCCESS) {
		cerr << "cufftExecR2C failed!" << endl;
		return false;
	}

	return true;
}

bool CUDAReverbEffect::IFT() {

	cufftResult_t cufftStatus;

	cufftStatus = cufftExecC2R(IFFT, OUT_SRC_L, cache_padded_l);
	if (cufftStatus != CUFFT_SUCCESS) {
		cerr << "cufftExecC2R failed!" << endl;
		return false;
	}

	if (STEREO == channels) {
		cufftStatus = cufftExecC2R(IFFT, OUT_SRC_R, cache_padded_r);
		if (cufftStatus != CUFFT_SUCCESS) {
			cerr << "cufftExecC2R failed!" << endl;
			return false;
		}
	}

	return true;
}

void CUDAReverbEffect::init_files(char *in_fn, char *ir_fn, char *out_fn) {

	in = new SndfileHandle(in_fn);
	channels = in->channels();
	ir = new SndfileHandle(ir_fn);
	out = new SndfileHandle(out_fn, SFM_WRITE, format, channels, samplerate);
	out->command(SFC_SET_UPDATE_HEADER_AUTO, NULL, SF_TRUE);

}

int CUDAReverbEffect::init_fftws() {

	cudaError_t cudaStatus;

	IR_blocks = ir->frames() / M + 1;

	in_src_l = new float[N];
	in_src_r = new float[N];

	gridDim = IR_blocks * N / BLOCK_SIZE;

#pragma region init cuReal device buffers

#pragma region init in_dev
	cudaStatus = cudaMalloc((void**)&in_dev_l, sizeof(cufftReal)* N);
	if (cudaStatus != cudaSuccess) {
		cerr << "cudaMalloc(in_dev_l) failed!" << endl;
		delete in_src_l;
		delete in_src_r;
		cudaFree(in_dev_l);
		return -1;
	}

	cudaStatus = cudaMalloc((void**)&in_dev_r, sizeof(cufftReal)* N);
	if (cudaStatus != cudaSuccess) {
		cerr << "cudaMalloc(in_dev_r) failed!" << endl;
		delete in_src_l;
		delete in_src_r;
		cudaFree(in_dev_l);
		cudaFree(in_dev_r);
		return -1;
	}
#pragma endregion

#pragma region init cache_padded
	cudaStatus = cudaMalloc((void**)&cache_padded_l, sizeof(cufftReal)* IR_blocks * N);
	if (cudaStatus != cudaSuccess) {
		cerr << "cudaMalloc(cache_padded_l) failed!" << endl;
		delete in_src_l;
		delete in_src_r;
		cudaFree(in_dev_l);
		cudaFree(in_dev_r);
		cudaFree(cache_padded_l);
		return -1;
	}

	cudaStatus = cudaMalloc((void**)&cache_padded_r, sizeof(cufftReal)* IR_blocks * N);
	if (cudaStatus != cudaSuccess) {
		cerr << "cudaMalloc(cache_padded_r) failed!" << endl;
		delete in_src_l;
		delete in_src_r;
		cudaFree(in_dev_l);
		cudaFree(in_dev_r);
		cudaFree(cache_padded_l);
		cudaFree(cache_padded_r);
		return -1;
	}
#pragma endregion

#pragma region init cache
	cudaStatus = cudaMalloc((void**)&cache_l, sizeof(cufftReal)* ((IR_blocks - 1) * M + N));
	if (cudaStatus != cudaSuccess) {
		cerr << "cudaMalloc(cache_l) failed!" << endl;
		delete in_src_l;
		delete in_src_r;
		cudaFree(in_dev_l);
		cudaFree(in_dev_r);
		cudaFree(cache_padded_l);
		cudaFree(cache_padded_r);
		cudaFree(cache_l);
		return -1;
	}
	cudaStatus = cudaMemset(cache_l, 0, sizeof(cufftReal)* ((IR_blocks - 1) * M + N));
	if (cudaStatus != cudaSuccess) {
		cerr << "cudaMemset(cache_l) failed!" << endl;
		delete in_src_l;
		delete in_src_r;
		cudaFree(in_dev_l);
		cudaFree(in_dev_r);
		cudaFree(cache_padded_l);
		cudaFree(cache_padded_r);
		cudaFree(cache_l);
		return -1;
	}

	cudaStatus = cudaMalloc((void**)&cache_r, sizeof(cufftReal)* ((IR_blocks - 1) * M + N));
	if (cudaStatus != cudaSuccess) {
		cerr << "cudaMalloc(cache_r) failed!" << endl;
		delete in_src_l;
		delete in_src_r;
		cudaFree(in_dev_l);
		cudaFree(in_dev_r);
		cudaFree(cache_padded_l);
		cudaFree(cache_padded_r);
		cudaFree(cache_l);
		cudaFree(cache_r);
		return -1;
	}
	cudaStatus = cudaMemset(cache_r, 0, sizeof(cufftReal)* ((IR_blocks - 1) * M + N));
	if (cudaStatus != cudaSuccess) {
		cerr << "cudaMemset(cache_r) failed!" << endl;
		delete in_src_l;
		delete in_src_r;
		cudaFree(in_dev_l);
		cudaFree(in_dev_r);
		cudaFree(cache_padded_l);
		cudaFree(cache_padded_r);
		cudaFree(cache_l);
		cudaFree(cache_r);
		return -1;
	}
#pragma endregion

#pragma region init temp_cache
	cudaStatus = cudaMalloc((void**)&temp_cache_l, sizeof(cufftReal)* ((IR_blocks - 1) * M + N));
	if (cudaStatus != cudaSuccess) {
		cerr << "cudaMalloc(temp_cache_l) failed!" << endl;
		delete in_src_l;
		delete in_src_r;
		cudaFree(in_dev_l);
		cudaFree(in_dev_r);
		cudaFree(cache_padded_l);
		cudaFree(cache_padded_r);
		cudaFree(cache_l);
		cudaFree(cache_r);
		cudaFree(temp_cache_l);
		return -1;
	}
	cudaStatus = cudaMemset(temp_cache_l, 0, sizeof(cufftReal)* ((IR_blocks - 1) * M + N));
	if (cudaStatus != cudaSuccess) {
		cerr << "cudaMemset(temp_cache_l) failed!" << endl;
		delete in_src_l;
		delete in_src_r;
		cudaFree(in_dev_l);
		cudaFree(in_dev_r);
		cudaFree(cache_padded_l);
		cudaFree(cache_padded_r);
		cudaFree(cache_l);
		cudaFree(cache_r);
		cudaFree(temp_cache_l);
		return -1;
	}

	cudaStatus = cudaMalloc((void**)&temp_cache_r, sizeof(cufftReal)* ((IR_blocks - 1) * M + N));
	if (cudaStatus != cudaSuccess) {
		cerr << "cudaMalloc(temp_cache_r) failed!" << endl;
		delete in_src_l;
		delete in_src_r;
		cudaFree(in_dev_l);
		cudaFree(in_dev_r);
		cudaFree(cache_padded_l);
		cudaFree(cache_padded_r);
		cudaFree(cache_l);
		cudaFree(cache_r);
		cudaFree(temp_cache_l);
		cudaFree(temp_cache_r);
		return -1;
	}
	cudaStatus = cudaMemset(temp_cache_r, 0, sizeof(cufftReal)* ((IR_blocks - 1) * M + N));
	if (cudaStatus != cudaSuccess) {
		cerr << "cudaMemset(temp_cache_r) failed!" << endl;
		delete in_src_l;
		delete in_src_r;
		cudaFree(in_dev_l);
		cudaFree(in_dev_r);
		cudaFree(cache_padded_l);
		cudaFree(cache_padded_r);
		cudaFree(cache_l);
		cudaFree(cache_r);
		cudaFree(temp_cache_l);
		cudaFree(temp_cache_r);
		return -1;
	}
#pragma endregion

#pragma endregion

#pragma region init cuComplex device buffers

#pragma region init OUT_SRC
	cudaStatus = cudaMalloc((void**)&OUT_SRC_L, sizeof(cufftComplex)* IR_blocks * (N / 2 - 1));
	if (cudaStatus != cudaSuccess) {
		cerr << "cudaMalloc(OUT_SRC_L) failed!" << endl;
		delete in_src_l;
		delete in_src_r;
		cudaFree(in_dev_l);
		cudaFree(in_dev_r);
		cudaFree(cache_padded_l);
		cudaFree(cache_padded_r);
		cudaFree(cache_l);
		cudaFree(cache_r);
		cudaFree(temp_cache_l);
		cudaFree(temp_cache_r);
		cudaFree(OUT_SRC_L);
		return -1;
	}

	cudaStatus = cudaMalloc((void**)&OUT_SRC_R, sizeof(cufftComplex)* IR_blocks * (N / 2 - 1));
	if (cudaStatus != cudaSuccess) {
		cerr << "cudaMalloc(OUT_SRC_R) failed!" << endl;
		delete in_src_l;
		delete in_src_r;
		cudaFree(in_dev_l);
		cudaFree(in_dev_r);
		cudaFree(cache_padded_l);
		cudaFree(cache_padded_r);
		cudaFree(cache_l);
		cudaFree(cache_r);
		cudaFree(temp_cache_l);
		cudaFree(temp_cache_r);
		cudaFree(OUT_SRC_L);
		cudaFree(OUT_SRC_R);
		return -1;
	}
#pragma endregion

#pragma region init IN
	cudaStatus = cudaMalloc((void**)&IN_L, sizeof(cufftComplex)* (N / 2 - 1));
	if (cudaStatus != cudaSuccess) {
		cerr << "cudaMalloc(IN_L) failed!" << endl;
		delete in_src_l;
		delete in_src_r;
		cudaFree(in_dev_l);
		cudaFree(in_dev_r);
		cudaFree(cache_padded_l);
		cudaFree(cache_padded_r);
		cudaFree(cache_l);
		cudaFree(cache_r);
		cudaFree(temp_cache_l);
		cudaFree(temp_cache_r);
		cudaFree(OUT_SRC_L);
		cudaFree(OUT_SRC_R);
		cudaFree(IN_L);
		return -1;
	}

	cudaStatus = cudaMalloc((void**)&IN_R, sizeof(cufftComplex)* (N / 2 - 1));
	if (cudaStatus != cudaSuccess) {
		cerr << "cudaMalloc(IN_R) failed!" << endl;
		delete in_src_l;
		delete in_src_r;
		cudaFree(in_dev_l);
		cudaFree(in_dev_r);
		cudaFree(cache_padded_l);
		cudaFree(cache_padded_r);
		cudaFree(cache_l);
		cudaFree(cache_r);
		cudaFree(temp_cache_l);
		cudaFree(temp_cache_r);
		cudaFree(OUT_SRC_L);
		cudaFree(OUT_SRC_R);
		cudaFree(IN_L);
		cudaFree(IN_R);
		return -1;
	}
#pragma endregion

#pragma region init IR
	cudaStatus = cudaMalloc((void**)&IR_L, sizeof(cufftComplex)* IR_blocks * (N / 2 - 1));
	if (cudaStatus != cudaSuccess) {
		cerr << "cudaMalloc(IR_L) failed!" << endl;
		delete in_src_l;
		delete in_src_r;
		cudaFree(in_dev_l);
		cudaFree(in_dev_r);
		cudaFree(cache_padded_l);
		cudaFree(cache_padded_r);
		cudaFree(cache_l);
		cudaFree(cache_r);
		cudaFree(temp_cache_l);
		cudaFree(temp_cache_r);
		cudaFree(OUT_SRC_L);
		cudaFree(OUT_SRC_R);
		cudaFree(IN_L);
		cudaFree(IN_R);
		cudaFree(IR_L);
		return -1;
	}

	cudaStatus = cudaMalloc((void**)&IR_R, sizeof(cufftComplex)* IR_blocks * (N / 2 - 1));
	if (cudaStatus != cudaSuccess) {
		cerr << "cudaMalloc(IR_R) failed!" << endl;
		delete in_src_l;
		delete in_src_r;
		cudaFree(in_dev_l);
		cudaFree(in_dev_r);
		cudaFree(cache_padded_l);
		cudaFree(cache_padded_r);
		cudaFree(cache_l);
		cudaFree(cache_r);
		cudaFree(temp_cache_l);
		cudaFree(temp_cache_r);
		cudaFree(OUT_SRC_L);
		cudaFree(OUT_SRC_R);
		cudaFree(IN_L);
		cudaFree(IN_R);
		cudaFree(IR_L);
		cudaFree(IR_R);
		return -1;
	}
#pragma endregion


#pragma endregion

#pragma region init plans for dft/ift
	cufftResult_t cufftStatus;

	cufftStatus = cufftPlan1d(&inDFFT, N, CUFFT_R2C, 1);
	if (cufftStatus != CUFFT_SUCCESS) {
		cerr << "cufftPlan1d(inDFFT) failed!" << endl;
		delete in_src_l;
		delete in_src_r;
		cudaFree(in_dev_l);
		cudaFree(in_dev_r);
		cudaFree(cache_padded_l);
		cudaFree(cache_padded_r);
		cudaFree(cache_l);
		cudaFree(cache_r);
		cudaFree(temp_cache_l);
		cudaFree(temp_cache_r);
		cudaFree(OUT_SRC_L);
		cudaFree(OUT_SRC_R);
		cudaFree(IN_L);
		cudaFree(IN_R);
		cudaFree(IR_L);
		cudaFree(IR_R);
		cufftDestroy(inDFFT);
		return -1;
	}
	cufftStatus = cufftPlan1d(&irDFFT, N, CUFFT_R2C, IR_blocks);
	if (cufftStatus != CUFFT_SUCCESS) {
		cerr << "cufftPlan1d(irDFFT) failed!" << endl;
		delete in_src_l;
		delete in_src_r;
		cudaFree(in_dev_l);
		cudaFree(in_dev_r);
		cudaFree(cache_padded_l);
		cudaFree(cache_padded_r);
		cudaFree(cache_l);
		cudaFree(cache_r);
		cudaFree(temp_cache_l);
		cudaFree(temp_cache_r);
		cudaFree(OUT_SRC_L);
		cudaFree(OUT_SRC_R);
		cudaFree(IN_L);
		cudaFree(IN_R);
		cudaFree(IR_L);
		cudaFree(IR_R);
		cufftDestroy(inDFFT);
		cufftDestroy(irDFFT);
		return -1;
	}
	cufftStatus = cufftPlan1d(&IFFT, N, CUFFT_C2R, IR_blocks);
	if (cufftStatus != CUFFT_SUCCESS) {
		cerr << "cufftPlan1d(IFFT) failed!" << endl;
		delete in_src_l;
		delete in_src_r;
		cudaFree(in_dev_l);
		cudaFree(in_dev_r);
		cudaFree(cache_padded_l);
		cudaFree(cache_padded_r);
		cudaFree(cache_l);
		cudaFree(cache_r);
		cudaFree(temp_cache_l);
		cudaFree(temp_cache_r);
		cudaFree(OUT_SRC_L);
		cudaFree(OUT_SRC_R);
		cudaFree(IN_L);
		cudaFree(IN_R);
		cudaFree(IR_L);
		cudaFree(IR_R);
		cufftDestroy(inDFFT);
		cufftDestroy(irDFFT);
		cufftDestroy(IFFT);
		return -1;
	}
#pragma endregion

	return 0;
}

void CUDAReverbEffect::init_in_out_mono() {
	
	in_sz = in->frames();
	in_l = new float[in_sz];
	memset(in_l, 0, sizeof(in_sz));
	in->readf(in_l, in_sz);

	out_sz = in->frames() + ir->frames() - 1;
	out_l = new float[out_sz];
	memset(out_l, 0, sizeof(float)* out_sz);

}

void CUDAReverbEffect::init_in_out_stereo() {

	in_sz = in->frames() * 2;
	in_stereo = new float[in_sz];
	in_l = new float[in_sz / 2];
	in_r = new float[in_sz / 2];
	memset(in_stereo, 0, sizeof(float) * in_sz);
	memset(in_l, 0, sizeof(float)* in->frames());
	memset(in_r, 0, sizeof(float)* in->frames());
	in->readf(in_stereo, in->frames());
	MonoStereoConversion::extractBothChannels(in_stereo, in_l, in_r, in_sz);

	out_sz = (in->frames() + ir->frames() - 1) * 2;
	out_l = new float[out_sz / 2];
	out_r = new float[out_sz / 2];
	out_stereo = new float[out_sz];
	memset(out_l, 0, sizeof(float)* out_sz / 2);
	memset(out_r, 0, sizeof(float)* out_sz / 2);
	memset(out_stereo, 0, sizeof(float)* out_sz);

}

void CUDAReverbEffect::init_ir_mono() {

	ir_sz = ir->frames();
	ir_l = new float[ir_sz];
	memset(ir_l, 0, sizeof(float)* ir_sz);
	ir->readf(ir_l, ir->frames());


	//it would be better to do this whole thing in one batch
	//also one memcpyHostToDevice for whole ir would be better

	for (long i = 0; i < ir_sz; i += M) {
		if (i + M > ir->frames()) {
			DFT(ir_l + i, ir->frames() - i, in_dev_l, IR_L + (i / M) * (N / 2 + 1), N);
		}
		else {
			DFT(ir_l + i, M, in_dev_l, IR_L + (i / M) * (N / 2 + 1), N);
		}
	}

}

void CUDAReverbEffect::init_ir_stereo() {

	ir_sz = ir->frames() * 2;
	ir_stereo = new float[ir_sz];
	ir_l = new float[ir_sz / 2];
	ir_r = new float[ir_sz / 2];
	memset(ir_stereo, 0, sizeof(float)* ir_sz);
	memset(ir_l, 0, sizeof(float)* ir_sz / 2);
	memset(ir_r, 0, sizeof(float)* ir_sz / 2);
	ir->readf(ir_stereo, ir->frames());
	MonoStereoConversion::extractBothChannels(ir_stereo, ir_l, ir_r, ir_sz);

	for (long i = 0; i < ir_sz / 2; i += M) {
		if (i + M > ir->frames()) {
			DFT(ir_l + i, ir->frames() - i, in_dev_l, IR_L + (i / M) * (N / 2 + 1), N);
			DFT(ir_r + i, ir->frames() - i, in_dev_r, IR_R + (i / M) * (N / 2 + 1), N);
		}
		else {
			DFT(ir_l + i, M, in_dev_l, IR_L + (i / M) * (N / 2 + 1), N);
			DFT(ir_r + i, M, in_dev_r, IR_R + (i / M) * (N / 2 + 1), N);
		}
	}

}