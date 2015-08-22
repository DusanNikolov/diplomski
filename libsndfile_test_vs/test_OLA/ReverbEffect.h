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



#ifndef REVERB_EFFECT_
#define REVERB_EFFECT_

//included for the time measuring purposes
//when CUDA gets implemented in the other version, use CUDA routines here also to provide better consistency

#include "Timer.h"

#include <sndfile.hh>
#include <fftw3.h>

#define MONO 1
#define STEREO 2

#define N 16384
#define M 8192
#define L 4096


class ReverbEffect {

public:
	ReverbEffect(SndfileHandle *in, SndfileHandle *ir, SndfileHandle *out);
	~ReverbEffect();

	void initialize(SndfileHandle *in, SndfileHandle *ir, SndfileHandle *out);

	void applyReverb();

	void writeOutNormalized();

private:
	void init_files(SndfileHandle *in, SndfileHandle *ir, SndfileHandle *out);
	
	void init_in_out_mono();
	void init_in_out_stereo();
	void init_ir_mono();
	void init_ir_stereo();

	void init_fftws();

	void OLA_mono();
	void OLA_stereo();

	void DFT(float *in_buf, sf_count_t in_len, float *in_fft, fftwf_complex *OUT_FFT);
	void IFT();
	//remember! dst MUST be different from both src1 & src2
	void complexMul(fftwf_complex *DST_L, fftwf_complex *DST_R, fftwf_complex *SRC1_L, fftwf_complex *SRC1_R, long src1_off,
		fftwf_complex *SRC2_L, fftwf_complex *SRC_R, long src2_off);

private:
	int channels, ir_channels,
		format = SF_FORMAT_WAV | SF_FORMAT_PCM_16,
		//samplerate could be an issue if the IR or IN signal have different samplerate freqs...
		//it could be fixed if you implement a method for switching between different samplerate freqs 
		samplerate = 44100;

	//audio file handles
	SndfileHandle *in, *ir, *out;

	//fftw_handles (r2c/c2r fft is used here)
	long IR_blocks;
	float *in_src_l, *in_src_r;
	fftwf_complex *OUT_SRC_L, *OUT_SRC_R, *IN_L, *IN_R, *IR_L, *IR_R;
	fftwf_plan DFFT, IFFT;

	//buffers
	sf_count_t ir_sz;
	float *ir_stereo, *ir_l, *ir_r;

	sf_count_t in_sz;
	float *in_stereo, *in_l, *in_r;
	sf_count_t out_sz;
	float *out_stereo, *out_l, *out_r;

	//currently not used, find a use or delete it!
	float *cache, *cache_l, *cache_r;

	//used for normalization after reverb across all used channels (MONO/STEREO)
	float max, max_l, max_r;

	//for time measuring purposes
private:
	Timer timer;

};

#endif