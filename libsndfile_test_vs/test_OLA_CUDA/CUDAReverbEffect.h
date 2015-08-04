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


#ifndef REVERB_EFFECT_
#define REVERB_EFFECT_

//included for the time measuring purposes
//when CUDA gets implemented in the other version, use CUDA routines here also to provide better consistency
#include <Windows.h>

#include <sndfile.hh>

#include <cuda_runtime_api.h>
#include <cufft.h>
#include <cufftw.h>

#include "device_functions.cuh"

#define MONO 1
#define STEREO 2

#define N 16384
#define M 8192
#define L 4096


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

	void init_fftws();

	void OLA_mono();
	void OLA_stereo();

	void DFT(float *in_buf, long in_len, float *in_fft, Complex *OUT_FFT);
	void IFT();
	//remember! dst MUST be different from both src1 & src2
	void complexMul(Complex *DST_L, Complex *DST_R, Complex *SRC1_L, Complex *SRC1_R, long src1_off,
		Complex *SRC2_L, Complex *SRC_R, long src2_off);

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
	Complex *OUT_SRC_L, *OUT_SRC_R, *IN_L, *IN_R, *IR_L, *IR_R;
	cufftHandle DFFT, IFFT;

	//buffers
	long ir_sz;
	float *ir_stereo, *ir_l, *ir_r;

	long in_sz;
	float *in_stereo, *in_l, *in_r;
	long out_sz;
	float *out_stereo, *out_l, *out_r;

	//currently not used, find a use or delete it!
	float *cache, *cache_l, *cache_r;

	//used for normalization after reverb across all used channels (MONO/STEREO)
	float max, max_l, max_r;

	//for time measuring purposes
private:
	LARGE_INTEGER frequency;        // ticks per second
	LARGE_INTEGER start, end;           // ticks
	double elapsedTime;

};

#endif