//class for implementing rounding-up methods required for convolution

#ifndef FILTER_H
#define FILTER_H


#include <fftw3.h>
#include <sndfile.hh>

//OLA2 stuff
#define L 4096   //size of input block
#define M 512    //size of ir block
#define N_ (L + M - 1)

#define MONO 1
#define STEREO 2

#define LEFT_CH 1
#define RIGHT_CH 2

#define INPUT 1
#define IR_ 2

#define MOD_OLA 1
#define MOD_SIMPLE 2

class Filter {
public:
	Filter(char* in_fname, char* IR_fname, char* out_fname, int mode);
	~Filter();

	void applyReverb();
	void applyReverb2();

	void DFFTChannel(int channelNo, int src);  // channelNO = {LEFT_CH,RIGHT_CH}, src = {INPUT, IR}
	void IFFTChannel(int channelNo);
	void convolveFFT(int channelNo);

	void recombineAndNormalize();

	void writeOut();

public:
	SndfileHandle *in, *ir, *out;

	float *in_lr, *in_l, *in_r;
	float *ir_lr, *ir_l, *ir_r;
	float *out_lr, *out_l, *out_r;

	fftw_complex *in_src, *out_src;
	fftw_complex *IN_L, *IN_R, *IR_L, *IR_R;
	fftw_plan plan_dfft, plan_ifft;

	long fft_size;  // N value: non real-time depends on sizes of  in and IR
	float maxValue; //will be used for scaling the output between [-1,1]

private:
	
	//overlap & add stuff:
	long in_frames, ir_frames, N, buf_sz;
	float *out_buf_lr, *out_buf_l, *out_buf_r, *ola_buf_l, *ola_buf_r;

	void initializeOLA();
	
	void IFFT_OLA(fftw_complex* in_src, fftw_complex* out_src, float* out_buf, float* ola_buf, long fft_size, long out_size, fftw_plan p_back);
	//end overlap & add stuff

	//overlap & add 2 (breaks the IR into pieces and performs OLA with a block of input, but also overlap-adds every consecutive block of output
	long cache_sz, ir_parts_cnt;
	//cache treba da cuva kompleksne vrednosti? zasto??
	float *cache_l, *cache_r;  //cache for storing konvolved data with overlapping segments

	void initializeOLA2();

	void IFFT_OLA2(fftw_complex* in_src, fftw_complex* out_src, float* cache, long index, long fft_size, fftw_plan p_back);
	//end overlap & add 2 stuff


	const int format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
	const int samplerate = 44100;
	int channels;	//out_file channels depend on in_file channels, mono -> mono / stereo -> stereo

	//helper functions
	long findNearestLargerPowerOfTwo(long number);
	void initialize();

	//simple convolution in time domain
	float* convolveSimple(float* A, float* B, long lenA, long lenB, long* lenC);

	//possibly return some value for logger, when you implement a logger
	void DFFT(fftw_complex* in_src, fftw_complex* out_src, float* input, long size, fftw_plan p_forw);
	void IFFT(fftw_complex* in_src, fftw_complex* out_src, float* out_buf, long fft_size, long out_size, fftw_plan p_back);
	void convolveFFT(fftw_complex* dst, fftw_complex* src1, fftw_complex* src2, long size);

};

#endif FILTER_H