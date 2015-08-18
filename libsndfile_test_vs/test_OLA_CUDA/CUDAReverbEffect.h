

//for MONO reverberation, only left buffers are used...
//that was done in order to simplify i/o operations and also to maintain more readible code

//TO-DO: Try to lower the big-O complexity of the OLA algo
//e.g. remove for loop, or reduce it only to the size of overlap, and use memcpy for the rest of the elements

//TO-DO2: If possible, implement the multithreaded approach for the fftw3 library... for now it's a sequential implementation


#ifndef CUDA_REVERB_EFFECT_
#define CUDA_REVERB_EFFECT_

#include <sndfile.hh>

//CUDA includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>
#include <cufft.h>

#define MONO 1
#define STEREO 2


//In order for OverlapAdd even and odd to work this condition must be true: (2 * M) >= N
#define N 16384
#define M 8192
#define L 4096


#define BLOCK_SIZE 256


class CUDAReverbEffect {

public:
	CUDAReverbEffect(SndfileHandle *in, SndfileHandle *ir, SndfileHandle *out);
	~CUDAReverbEffect();

	void initialize(SndfileHandle *in, SndfileHandle *ir, SndfileHandle *out);

	void applyReverb();

	void writeOutNormalized();

private:
	void init_files(SndfileHandle *in, SndfileHandle *ir, SndfileHandle *out);
	
	void init_in_out_mono();
	void init_in_out_stereo();
	void init_ir_mono();
	void init_ir_stereo();

	int init_fftws();

	bool OLA_mono();
	bool OLA_stereo();

	bool DFT(float *in_host, sf_count_t in_len, cufftReal *in_dev, cufftComplex *OUT_DEV, int fft_size);
	bool IFT();

private:
	int channels,
		format = SF_FORMAT_WAV | SF_FORMAT_PCM_16,
		//samplerate could be an issue if the IR or IN signal have different samplerate freqs...
		//it could be fixed if you implement a method for switching between different samplerate freqs 
		samplerate = 44100;

	//audio file handles
	SndfileHandle *in, *ir, *out;

	//cuda_handles (r2c/c2r fft is used here)
	sf_count_t IR_blocks;
	float *in_src_l, *in_src_r;
	cufftReal *in_dev_l, *in_dev_r;
	cufftReal *cache_l, *cache_r, *temp_cache_l, *temp_cache_r;
	cufftReal *cache_padded_l, *cache_padded_r;
	cufftComplex *OUT_SRC_L, *OUT_SRC_R, *IN_L, *IN_R, *IR_L, *IR_R;

	cufftHandle inDFFT, irDFFT, IFFT;

	dim3 gridDim;
	cudaStream_t stream_l, stream_r;

	//buffers
	sf_count_t ir_sz;
	float *ir_stereo, *ir_l, *ir_r;

	sf_count_t in_sz;
	float *in_stereo, *in_l, *in_r;
	sf_count_t out_sz;
	float *out_stereo, *out_l, *out_r;

	//used for normalization after reverb across all used channels (MONO/STEREO)
	float max, max_l, max_r;

	//time measuring
	cudaEvent_t start, stop;
	float time;

};

#endif