// implementing Filter class

#include "Filter.h"
#include "MonoStereoConversion.h"

#include <cstdlib>
#include <iostream>

using namespace std;

Filter::Filter(char* in_fname, char* IR_fname, char* out_fname, int mode) {
	in = new SndfileHandle(in_fname);
	ir = new SndfileHandle(IR_fname);

	channels = in->channels();

	out = new SndfileHandle(out_fname, SFM_WRITE, format, channels, samplerate);
	//enable header updating after every write
	out->command(SFC_SET_UPDATE_HEADER_AUTO, NULL, SF_TRUE);

	MOD_OLA == mode ? initializeOLA2() : initialize();

}
Filter::~Filter() {

}


//OLA2
void Filter::initializeOLA2() {
	in_frames = in->frames();
	ir_frames = ir->frames();
	fft_size = findNearestLargerPowerOfTwo(N_);   //FFT size

	fft_size = N = N_;

	//size of cache is equal to the number of IR parts * size of input block (L) + M - 1 for overlap 
	ir_parts_cnt = (long)ceil((double)ir_frames / M);
	cache_sz = (L + (ir_parts_cnt * M) - 1);
	cache_l = new float[cache_sz];
	cache_r = new float[cache_sz];

	memset(cache_l, 0, sizeof(float)* cache_sz);
	memset(cache_r, 0, sizeof(float)* cache_sz);

#pragma region init_fftw_vars
	//N_ not power of two, but just testing...
	in_src = (fftw_complex*)malloc(sizeof(fftw_complex)* N_);
	out_src = (fftw_complex*)malloc(sizeof(fftw_complex)* N_);

	IN_L = (fftw_complex*)malloc(sizeof(fftw_complex)* N_);
	IN_R = (fftw_complex*)malloc(sizeof(fftw_complex)* N_);
	IR_L = (fftw_complex*)malloc(sizeof(fftw_complex)* N_ * ir_parts_cnt);
	IR_R = (fftw_complex*)malloc(sizeof(fftw_complex)* N_ * ir_parts_cnt);

	//kada budes implementirao FFT sa OVERLAP & ADD koristi FFTW_MEASURE
	plan_dfft = fftw_plan_dft_1d(N_, in_src, out_src, FFTW_FORWARD, FFTW_ESTIMATE);
	plan_ifft = fftw_plan_dft_1d(N_, in_src, out_src, FFTW_BACKWARD, FFTW_ESTIMATE);
#pragma endregion

#pragma region init_input
	if (STEREO == channels) {
		in_lr = new float[in_frames * STEREO];
		in_l = new float[in_frames];
		in_r = new float[in_frames];

		//read stereo audio data & separate left and right channels
		in->readf(in_lr, in_frames);
		MonoStereoConversion::extractBothChannels(in_lr, in_l, in_r, in_frames * STEREO);

	}
	else {
		in_lr = NULL; in_r = NULL;
		in_l = new float[in_frames];

		//read mono audio data
		in->readf(in_l, in_frames);

	}
#pragma endregion

#pragma region init_IR
	if (STEREO == ir->channels()) {
		ir_lr = new float[ir_frames * STEREO];
		ir_l = new float[ir_frames];
		ir_r = new float[ir_frames];

		//read stereo audio data & separate left and right channels
		ir->readf(ir_lr, ir->frames());
		MonoStereoConversion::extractBothChannels(ir_lr, ir_l, ir_r, ir_frames * STEREO);

		if (ir_frames % M) {
			for (long i = 0; i < ir_parts_cnt - 1; i++) {
				DFFT(in_src + i * M, IR_L + i * N_, ir_l, M, plan_dfft);
				DFFT(in_src + i * M, IR_R + i * N_, ir_r, M, plan_dfft);
			}
			DFFT(in_src + (ir_parts_cnt - 1) * M, IR_L + (ir_parts_cnt - 1) * N_, ir_l, (ir_frames % M), plan_dfft);
			DFFT(in_src + (ir_parts_cnt - 1) * M, IR_R + (ir_parts_cnt - 1) * N_, ir_r, (ir_frames % M), plan_dfft);

		}
		else {
			for (long i = 0; i < ir_parts_cnt; i++) {
				DFFT(in_src + i * M, IR_L + i * N_, ir_l, M, plan_dfft);
				DFFT(in_src + i * M, IR_R + i * N_, ir_r, M, plan_dfft);
			}
		}

	}
	else {
		ir_lr = NULL;
		ir_l = new float[ir->frames()];
		ir_r = NULL;

		//read mono audio data
		ir->readf(ir_l, ir->frames());

		if (ir_frames % M) {
			for (long i = 0; i < ir_parts_cnt - 1; i++) {
				DFFT(in_src + i * M, IR_L + i * N_, ir_l, M, plan_dfft);
			}
			DFFT(in_src + (ir_parts_cnt - 1) * M, IR_L + (ir_parts_cnt - 1) * N_, ir_l, (ir_frames % M), plan_dfft);

		}
		else {
			for (long i = 0; i < ir_parts_cnt; i++) {
				DFFT(in_src + i * M, IR_L + i * N_, ir_l, M, plan_dfft);
			}
		}

	}
#pragma endregion

#pragma region init_output
	if (STEREO == channels) {
		out_lr = new float[L * STEREO];
		out_l = new float[L];
		out_r = new float[L];
	}
	else {
		out_lr = new float[L];
		out_l = new float[L];
		out_r = NULL;
	}
#pragma endregion
}

void Filter::applyReverb2() {
	
	if (STEREO == channels) {

		for (long i = 0; i < in_frames; i += L) {

			if (i + L > in_frames) {
				DFFT(in_src, IN_L, (in_l + i), in_frames - i, plan_dfft);
				DFFT(in_src, IN_R, (in_r + i), in_frames - i, plan_dfft);

				for (long j = 0; j < ir_parts_cnt; j++) {

					convolveFFT(in_src, IN_L, IR_L + j * N_, N_);
					IFFT_OLA2(in_src, out_src, cache_l, j, N_, plan_ifft);

					convolveFFT(in_src, IN_R, IR_R + j * N_, N_);
					IFFT_OLA2(in_src, out_src, cache_r, j, N_, plan_ifft);

				}

				memcpy(out_l, cache_l, sizeof(float)* L);
				memmove(cache_l, cache_l + L, sizeof(float)* (cache_sz - L));
				memset(cache_l + cache_sz - L, 0, sizeof(float)* L);

				memcpy(out_r, cache_r, sizeof(float)* L);
				memmove(cache_r, cache_r + L, sizeof(float)* (cache_sz - L));
				memset(cache_r + cache_sz - L, 0, sizeof(float)* L);



				MonoStereoConversion::combine2Channels(out_l, out_r, out_lr, L, &maxValue);
				MonoStereoConversion::normalize(out_lr, 2 * L, maxValue);

				out->writef(out_lr, L);

			}
			else {
				DFFT(in_src, IN_L, (in_l + i), L, plan_dfft);

				for (long j = 0; j < ir_parts_cnt; j++) {

					convolveFFT(in_src, IN_L, IR_L + j * N_, N_);
					IFFT_OLA2(in_src, out_src, cache_l, j, N_, plan_ifft);

				}

				memcpy(out_l, cache_l, sizeof(float)* L);
				memmove(cache_l, cache_l + L, sizeof(float)* (cache_sz - L));
				memset(cache_l + cache_sz - L, 0, sizeof(float)* L);

				MonoStereoConversion::copy1Channel(out_l, out_lr, L, &maxValue);
				MonoStereoConversion::normalize(out_lr, L, maxValue);

				out->writef(out_lr, L);
			}

		}

	}
	else if (MONO == channels) {

		for (long i = 0; i < in_frames; i += L) {

			if (i + L > in_frames) {
				DFFT(in_src, IN_L, (in_l + i), in_frames - i, plan_dfft);

				for (long j = 0; j < ir_parts_cnt; j++) {

					convolveFFT(in_src, IN_L, IR_L + j * N_, N_);
					IFFT_OLA2(in_src, out_src, cache_l, j, N_, plan_ifft);

				}

				memcpy(out_l, cache_l, sizeof(float)* L);
				memmove(cache_l, cache_l + L, sizeof(float)* (cache_sz - L));
				memset(cache_l + cache_sz - L, 0, sizeof(float)* L);

				MonoStereoConversion::copy1Channel(out_l, out_lr, L, &maxValue);
				MonoStereoConversion::normalize(out_lr, L, maxValue);

				out->writef(out_lr, L);

			}
			else {
				DFFT(in_src, IN_L, (in_l + i), L, plan_dfft);

				for (long j = 0; j < ir_parts_cnt; j++) {

					convolveFFT(in_src, IN_L, IR_L + j * N_, N_);
					IFFT_OLA2(in_src, out_src, cache_l, j, N_, plan_ifft);
				
				}

				memcpy(out_l, cache_l, sizeof(float)* L);
				memmove(cache_l, cache_l + L, sizeof(float)* (cache_sz - L));
				memset(cache_l + cache_sz - L, 0, sizeof(float)* L);

				MonoStereoConversion::copy1Channel(out_l, out_lr, L, &maxValue);
				MonoStereoConversion::normalize(out_lr, L, maxValue);

				out->writef(out_lr, L);
			}

		}

	}
}

void Filter::IFFT_OLA2(fftw_complex* in_src, fftw_complex* out_src, float* cache, long index, long fft_size, fftw_plan p_back) {

	fftw_execute_dft(p_back, in_src, out_src);

	//next loop performs scaling-down, neccessary due to effects of fft/ifft
	for (long i = 0; i < fft_size; i++) {
		cache[index * L + i] += (out_src[i][0] / fft_size); //cisto sabiranje overlap-a, morace i nekakvo skaliranje da se radi...
		
	}
}

//OLA
void Filter::initializeOLA() {
	
	in_frames = in->frames();
	ir_frames = ir->frames();
	buf_sz = (ir_frames >= 2048 ? ir_frames : 2048);           //input buffer size
	N = findNearestLargerPowerOfTwo(buf_sz + ir_frames - 1);   //FFT size

	fft_size = N;

	out_buf_l = new float[N];
	ola_buf_l = new float[ir_frames - 1];
	out_buf_r = new float[N];
	ola_buf_r = new float[ir_frames - 1];

	out_buf_lr = new float[(buf_sz + ir_frames - 1) * 2];

	memset(out_buf_l, 0, sizeof(float)* N);
	memset(ola_buf_l, 0, sizeof(float)* ir_frames - 1);
	memset(out_buf_r, 0, sizeof(float)* N);
	memset(ola_buf_r, 0, sizeof(float)* ir_frames - 1);

	memset(out_buf_lr, 0, sizeof(float)* (buf_sz + ir_frames - 1) * 2);

#pragma region init_fftw_vars
	in_src = (fftw_complex*)malloc(sizeof(fftw_complex)* N);
	out_src = (fftw_complex*)malloc(sizeof(fftw_complex)* N);

	IN_L = (fftw_complex*)malloc(sizeof(fftw_complex)* N);
	IN_R = (fftw_complex*)malloc(sizeof(fftw_complex)* N);
	IR_L = (fftw_complex*)malloc(sizeof(fftw_complex)* N);
	IR_R = (fftw_complex*)malloc(sizeof(fftw_complex)* N);

	//kada budes implementirao FFT sa OVERLAP & ADD koristi FFTW_MEASURE
	plan_dfft = fftw_plan_dft_1d(N, in_src, out_src, FFTW_FORWARD, FFTW_ESTIMATE);
	plan_ifft = fftw_plan_dft_1d(N, in_src, out_src, FFTW_BACKWARD, FFTW_ESTIMATE);
#pragma endregion

#pragma region init_input
	if (STEREO == channels) {
		in_lr = new float[in_frames * STEREO];
		in_l = new float[in_frames];
		in_r = new float[in_frames];

		//read stereo audio data & separate left and right channels
		in->readf(in_lr, in_frames);
		MonoStereoConversion::extractBothChannels(in_lr, in_l, in_r, in_frames * STEREO);

	}
	else {
		in_lr = NULL; in_r = NULL;
		in_l = new float[in_frames];

		//read mono audio data
		in->readf(in_l, in_frames);

	}
#pragma endregion

#pragma region init_IR
	if (STEREO == ir->channels()) {
		ir_lr = new float[ir_frames * STEREO];
		ir_l = new float[ir_frames];
		ir_r = new float[ir_frames];

		//read stereo audio data & separate left and right channels
		ir->readf(ir_lr, ir->frames());
		MonoStereoConversion::extractBothChannels(ir_lr, ir_l, ir_r, ir_frames * STEREO);

		DFFT(in_src, IR_L, ir_l, ir_frames, plan_dfft);
		DFFT(in_src, IR_R, ir_r, ir_frames, plan_dfft);

	}
	else {
		ir_lr = NULL;
		ir_l = new float[ir->frames()];
		ir_r = NULL;

		//read mono audio data
		ir->readf(ir_l, ir->frames());

		DFFT(in_src, IR_L, ir_l, ir_frames, plan_dfft);

	}
#pragma endregion

#pragma region init_output
	if (STEREO == channels) {
		out_lr = new float[(in_frames + ir_frames - 1) * STEREO];
		out_l = new float[in_frames + ir_frames - 1];
		out_r = new float[in_frames + ir_frames - 1];
	}
	else {
		out_lr = new float[in_frames + ir_frames - 1];
		out_l = new float[in_frames + ir_frames - 1];
		out_r = NULL;
	}
#pragma endregion

}

void Filter::applyReverb() {

	if (STEREO == channels) {

		for (long i = 0; i < in_frames; i += buf_sz) {

			if (i + buf_sz > in_frames) {
				//left channel
				DFFT(in_src, IN_L, (in_l + i), in_frames - i, plan_dfft);
				convolveFFT(1);
				IFFT_OLA(in_src, out_src, out_buf_l, ola_buf_l, fft_size, buf_sz + ir_frames - 1, plan_ifft);
				memcpy(ola_buf_l, (out_buf_l + buf_sz), sizeof(float)* (ir_frames - 1));

				//right channel
				DFFT(in_src, IN_R, (in_r + i), in_frames - i, plan_dfft);
				convolveFFT(2);
				IFFT_OLA(in_src, out_src, out_buf_r, ola_buf_r, fft_size, buf_sz + ir_frames - 1, plan_ifft);
			
				MonoStereoConversion::combine2Channels(out_buf_l, out_buf_r, out_buf_lr, buf_sz + ir_frames - 1, &maxValue);
				MonoStereoConversion::normalize(out_buf_lr, buf_sz + ir_frames - 1, maxValue);

				out->writef(out_buf_lr, buf_sz + ir_frames - 1);
			}
			else {
				//left channel
				DFFT(in_src, IN_L, (in_l + i), buf_sz, plan_dfft);
				convolveFFT(1);
				IFFT_OLA(in_src, out_src, out_buf_l, ola_buf_l, fft_size, buf_sz + ir_frames - 1, plan_ifft);
				memcpy(ola_buf_l, (out_buf_l + buf_sz), sizeof(float)* (ir_frames - 1));

				//right channel
				DFFT(in_src, IN_R, (in_r + i), buf_sz, plan_dfft);
				convolveFFT(2);
				IFFT_OLA(in_src, out_src, out_buf_r, ola_buf_r, fft_size, buf_sz + ir_frames - 1, plan_ifft);
				memcpy(ola_buf_r, (out_buf_r + buf_sz), sizeof(float)* (ir_frames - 1));

				MonoStereoConversion::combine2Channels(out_buf_l, out_buf_r, out_buf_lr, buf_sz, &maxValue);
				MonoStereoConversion::normalize(out_buf_lr, buf_sz, maxValue);

				out->writef(out_buf_lr, buf_sz);
			}

		}

	}
	else {

		for (long i = 0; i < in_frames; i += buf_sz) {

			if (i + buf_sz > in_frames) {
				DFFT(in_src, IN_L, (in_l + i), in_frames - i, plan_dfft);

				convolveFFT(1);

				IFFT_OLA(in_src, out_src, out_buf_l, ola_buf_l, fft_size, buf_sz + ir_frames - 1, plan_ifft);

				out->writef(out_buf_l, buf_sz + ir_frames - 1);

			}

			else {
				DFFT(in_src, IN_L, (in_l + i), buf_sz, plan_dfft);

				convolveFFT(1);

				IFFT_OLA(in_src, out_src, out_buf_l, ola_buf_l, fft_size, buf_sz + ir_frames - 1, plan_ifft);

				memcpy(ola_buf_l, (out_buf_l + buf_sz), sizeof(float)* (ir_frames - 1));

				MonoStereoConversion::copy1Channel(out_buf_l, out_buf_lr, buf_sz + ir_frames - 1, &maxValue);
				MonoStereoConversion::normalize(out_buf_lr, buf_sz + ir_frames - 1, maxValue);

				out->writef(out_buf_l, buf_sz);
			}

		}

	}

}

void Filter::IFFT_OLA(fftw_complex* in_src, fftw_complex* out_src, float* out_buf, float* ola_buf, long fft_size, long out_size, fftw_plan p_back) {

	fftw_execute_dft(p_back, in_src, out_src);

	//next loop performs scaling-down, neccessary due to effects of fft/ifft
	for (long i = 0; i < out_size; i++) {
		if (i < ir_frames) {
			out_buf[i] = (out_src[i][0] / fft_size) * (1.0 - (i + 0.0) / (ir_frames - 1));
			out_buf[i] += ola_buf[i] * (i + 0.0) / (ir_frames - 1);
		}
		else {
			out_buf[i] = (out_src[i][0] / fft_size);
		}
	}
}



void Filter::initialize() {
	fft_size = findNearestLargerPowerOfTwo(in->frames() + ir->frames() - 1);

#pragma region init_input
	if (STEREO == channels) {
		in_lr = new float[in->frames() * STEREO];
		in_l = new float[fft_size];
		in_r = new float[fft_size];

		//read stereo audio data & separate left and right channels
		in->readf(in_lr, in->frames());
		MonoStereoConversion::extractBothChannels(in_lr, in_l, in_r, in->frames() * STEREO);

		//add zero padding to channel buffers
		for (long i = in->frames(); i < fft_size; i++) {
			in_l[i] = 0.0f;
			in_r[i] = 0.0f;
		}
	}
	else {
		in_lr = NULL; in_r = NULL;
		in_l = new float[fft_size];

		//read mono audio data
		in->readf(in_l, in->frames());

		//add zero padding
		for (long i = in->frames(); i < fft_size; i++)
			in_l[i] = 0.0f;
	}
#pragma endregion

#pragma region init_IR
	if (STEREO == ir->channels()) {
		ir_lr = new float[in->frames() * STEREO];
		ir_l = new float[fft_size];
		ir_r = new float[fft_size];

		//read stereo audio data & separate left and right channels
		ir->readf(ir_lr, ir->frames());
		MonoStereoConversion::extractBothChannels(ir_lr, ir_l, ir_r, ir->frames() * STEREO);

		//add zero padding to channel buffers
		for (long i = ir->frames(); i < fft_size; i++) {
			ir_l[i] = 0.0f;
			ir_r[i] = 0.0f;
		}
	}
	else {
		ir_lr = NULL;
		ir_l = new float[fft_size];
		ir_r = NULL;

		//read mono audio data
		ir->readf(ir_l, ir->frames());

		//add zero padding
		for (long i = ir->frames(); i < fft_size; i++)
			ir_l[i] = 0.0f;
	}
#pragma endregion

#pragma region init_output
	if (STEREO == channels) {
		out_lr = new float[(in->frames() + ir->frames() - 1) * STEREO];
		out_l = new float[in->frames() + ir->frames() - 1];
		out_r = new float[in->frames() + ir->frames() - 1];
	}
	else {
		out_lr = new float[in->frames() + ir->frames() - 1];
		out_l = new float[in->frames() + ir->frames() - 1];
		out_r = NULL;
	}
#pragma endregion

#pragma region init_fftw_vars
	in_src = (fftw_complex*)malloc(sizeof(fftw_complex)* fft_size);
	out_src = (fftw_complex*)malloc(sizeof(fftw_complex)* fft_size);
	IN_L = (fftw_complex*)malloc(sizeof(fftw_complex)* fft_size);
	IN_R = (fftw_complex*)malloc(sizeof(fftw_complex)* fft_size);
	IR_L = (fftw_complex*)malloc(sizeof(fftw_complex)* fft_size);
	IR_R = (fftw_complex*)malloc(sizeof(fftw_complex)* fft_size);

	//kada budes implementirao FFT sa OVERLAP & ADD koristi FFTW_MEASURE
	plan_dfft = fftw_plan_dft_1d(fft_size, in_src, out_src, FFTW_FORWARD, FFTW_ESTIMATE);
	plan_ifft = fftw_plan_dft_1d(fft_size, in_src, out_src, FFTW_BACKWARD, FFTW_ESTIMATE);
#pragma endregion
}

void Filter::DFFTChannel(int channelNo, int src) {

	if (LEFT_CH == channelNo) {
		if (INPUT == src)
			DFFT(in_src, IN_L, in_l, fft_size, plan_dfft);
		else
			DFFT(in_src, IR_L, ir_l, fft_size, plan_dfft);
	}
	else if (RIGHT_CH == channelNo) {
		if (INPUT == src)
			DFFT(in_src, IN_R, in_r, fft_size, plan_dfft);
		else if (IR_ == src) {
			if (STEREO == ir->channels())
				DFFT(in_src, IR_R, ir_r, fft_size, plan_dfft);
		}
	}

}
void Filter::IFFTChannel(int channelNo) {

	if (LEFT_CH == channelNo)
		IFFT(in_src, out_src, out_l, fft_size, in->frames() + ir->frames() - 1, plan_ifft);
	else if (RIGHT_CH == channelNo)
		IFFT(in_src, out_src, out_r, fft_size, in->frames() + ir->frames() - 1, plan_ifft);

}
void Filter::convolveFFT(int channelNo) {
	if (LEFT_CH == channelNo)
		convolveFFT(in_src, IN_L, IR_L, fft_size);
	else if (RIGHT_CH == channelNo) {
		if (MONO == ir->channels())
			convolveFFT(in_src, IN_R, IR_L, fft_size);
		else
			convolveFFT(in_src, IN_R, IR_R, fft_size);
	}
}

void Filter::writeOut() {
	out->writef(out_lr, in->frames() + ir->frames() - 1);
}

float* Filter::convolveSimple(float* A, float* B, long lenA, long lenB, long* lenC) {
	long nconv;
	long i, j, i1;
	float tmp;
	float *C;

	//allocated convolution array   
	nconv = lenA + lenB - 1;
	C = (float*)calloc(nconv, sizeof(float));

	//convolution process
	for (i = 0; i < nconv; i++) {
		i1 = i;
		tmp = 0.0;
		for (j = 0; j < lenB; j++) {
			if (i1 >= 0 && i1 < lenA)
				tmp = tmp + (A[i1] * B[j]);
			i1 = i1 - 1;
			C[i] = tmp;
		}
	}

	return C;
}

void Filter::DFFT(fftw_complex* in_src, fftw_complex* out_src, float* input, long size, fftw_plan p_forw) {

	memset(in_src, 0, sizeof(fftw_complex)* fft_size);

	//fill the complex input vector with real, sampled data
	for (long i = 0; i < size; i++) {
		in_src[i][0] = input[i];   //data is copied to the real part of complex data
		in_src[i][1] = 0.0f;       //imaginary part is filled with zeros
	}

	fftw_execute_dft(p_forw, in_src, out_src);

}

void Filter::IFFT(fftw_complex* in_src, fftw_complex* out_src, float* out_buf, long fft_size, long out_size, fftw_plan p_back) {

	fftw_execute_dft(p_back, in_src, out_src);

	//next loop performs scaling-down, neccessary due to effects of fft/ifft
	for (long i = 0; i < out_size; i++)
		out_buf[i] = out_src[i][0] / fft_size;   //scales down by the factor of fft_size

}

void Filter::convolveFFT(fftw_complex* dst, fftw_complex* src1, fftw_complex* src2, long size) {

	//complex multiplication
	for (long i = 0; i < size; i++) {
		dst[i][0] = src1[i][0] * src2[i][0] - src1[i][1] * src2[i][1];
		dst[i][1] = src1[i][0] * src2[i][1] + src1[i][1] * src2[i][0];
	}

}

void Filter::recombineAndNormalize() {
	if (STEREO == channels) {
		MonoStereoConversion::combine2Channels(out_l, out_r, out_lr, in->frames() + ir->frames() - 1, &maxValue);
		MonoStereoConversion::normalize(out_lr, (in->frames() + ir->frames() - 1) * 2, maxValue);
	}
	else {
		MonoStereoConversion::copy1Channel(out_l, out_lr, in->frames() + ir->frames() - 1, &maxValue);
		MonoStereoConversion::normalize(out_lr, in->frames() + ir->frames() - 1, maxValue);
	}
}

long Filter::findNearestLargerPowerOfTwo(long num) {
	long number = num;

	number--;
	number |= number >> 1;
	number |= number >> 2;
	number |= number >> 4;
	number |= number >> 8;
	number |= number >> 16;
	number++;

	return number;
}