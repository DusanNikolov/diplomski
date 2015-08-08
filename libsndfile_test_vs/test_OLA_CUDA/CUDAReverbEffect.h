//to measure time use QueryPerformanceFrequency
//LARGE_INTEGER frequency;        // ticks per second
//LARGE_INTEGER start, end;           // ticks
//double elapsedTime;
//
//QueryPerformanceFrequency(&frequency);
//
//
//QueryPerformanceCounter(&start);
//--CODE THAT IS BEING MEASURED--
//QueryPerformanceCounter(&end);
//
//elapsedTime = (end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
//cout << "left channel dfft time: " << elapsedTime << "[ms]" << endl;


//for MONO reverberation, only left buffers are used...
//that was done in order to simplify i/o operations and also to maintain more readible code

//TO-DO: Try to lower the big-O complexity of the OLA algo
//e.g. remove for loop, or reduce it only to the size of overlap, and use memcpy for the rest of the elements

//TO-DO2: If possible, implement the multithreaded approach for the fftw3 library... for now it's a sequential implementation


#ifndef CUDA_REVERB_EFFECT_
#define CUDA_REVERB_EFFECT_

//included for the time measuring purposes
//when CUDA gets implemented in the other version, use CUDA routines here also to provide better consistency
#include <Windows.h>

#include <sndfile.hh>

//CUDA includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>
#include <cufft.h>

#define MONO 1
#define STEREO 2

#define N 16384
#define M 8192
#define L 4096


#define BLOCK_SIZE 256


class CUDAReverbEffect {

public:
	CUDAReverbEffect(char *in_fn, char *ir_fn, char *out_fn);
	~CUDAReverbEffect();

	void initialize(char *in_fn, char *ir_fn, char *out_fn);

	void applyReverb();

	void writeOutNormalized();

private:
	void init_files(char *in_fn, char *ir_fn, char *out_fn);
	
	void init_in_out_mono();
	void init_in_out_stereo();
	void init_ir_mono();
	void init_ir_stereo();

	int init_fftws();

	bool OLA_mono();
	void OLA_stereo();

	bool DFT(float *in_host, long in_len, cufftReal *in_dev, cufftComplex *OUT_DEV, int fft_size);
	bool IFT();
	//remember! dst MUST be different from both src1 & src2
	void complexMul(cufftComplex *DST_L, cufftComplex *DST_R, cufftComplex *SRC1_L, cufftComplex *SRC1_R, long src1_off,
		cufftComplex *SRC2_L, cufftComplex *SRC_R, long src2_off);

private:
	int channels,
		format = SF_FORMAT_WAV | SF_FORMAT_PCM_16,
		//samplerate could be an issue if the IR or IN signal have different samplerate freqs...
		//it could be fixed if you implement a method for switching between different samplerate freqs 
		samplerate = 44100;

	//audio file handles
	SndfileHandle *in, *ir, *out;

	//fftw_handles (r2c/c2r fft is used here)
	long IR_blocks;
	float *in_src_l, *in_src_r;
	cufftReal *in_dev_l, *in_dev_r;
	cufftReal *cache, *cache_l, *cache_r, *temp_cache;
	cufftReal *cache_padded_l;
	cufftComplex *OUT_SRC_L, *OUT_SRC_R, *IN_L, *IN_R, *IR_L, *IR_R;

	cufftHandle inDFFT, irDFFT, IFFT;

	//buffers
	long ir_sz;
	float *ir_stereo, *ir_l, *ir_r;

	long in_sz;
	float *in_stereo, *in_l, *in_r;
	long out_sz;
	float *out_stereo, *out_l, *out_r;

	//used for normalization after reverb across all used channels (MONO/STEREO)
	float max, max_l, max_r;

	//for time measuring purposes
private:
	LARGE_INTEGER frequency;        // ticks per second
	LARGE_INTEGER start, end;           // ticks
	double elapsedTime;

	//CUDA
	dim3 gridDim;

};

#endif