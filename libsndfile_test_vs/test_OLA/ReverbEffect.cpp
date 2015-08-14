//Method definitions of ReverbEffect class

#include "ReverbEffect.h"
#include "MonoStereoConversion.h"

#include <omp.h>

#include <iostream>
using namespace std;

ReverbEffect::ReverbEffect(SndfileHandle *in, SndfileHandle *ir, SndfileHandle *out) {

	QueryPerformanceFrequency(&frequency);

	//better to set this up from command line!
	omp_set_num_threads(8);

	initialize(in, ir, out);

}

ReverbEffect::~ReverbEffect() {
	
	delete cache; delete cache_l; delete cache_r;
	
	if (STEREO == channels) {
		delete in_stereo; delete in_l; delete in_r;
		delete out_stereo; delete out_l; delete out_r;
	}
	else {
		delete in_l;
		delete out_l;
	}

	if (STEREO == ir_channels) {
		delete ir_stereo; delete ir_l; delete ir_r;

	}
	else {
		delete ir_l;
	}

	fftwf_destroy_plan(DFFT);
	fftwf_destroy_plan(IFFT);

	fftwf_free(IR_R);
	fftwf_free(IR_L);
	fftwf_free(IN_R);
	fftwf_free(IN_L);
	fftwf_free(OUT_SRC_R);
	fftwf_free(OUT_SRC_L);

	delete in_src_l; delete in_src_r;

	delete in; delete ir; delete out;

}

void ReverbEffect::initialize(SndfileHandle *in, SndfileHandle *ir, SndfileHandle *out) {
	LARGE_INTEGER start_init_files, end_init_files;
	LARGE_INTEGER start_init_fftws, end_init_fftws;
	LARGE_INTEGER start_init_inout, end_init_inout;
	LARGE_INTEGER start_init_ir, end_init_ir;
	double eTime_files, eTime_fftws, eTime_inout, eTime_ir;

	//QueryPerformanceFrequency(&frequency);

	QueryPerformanceCounter(&start_init_files);
	init_files(in, ir, out);
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

void ReverbEffect::applyReverb() {

	if (STEREO == channels) {
		OLA_stereo();
	}
	else {
		OLA_mono();
	}

}

//parallelised
void ReverbEffect::writeOutNormalized() {

	if (STEREO == channels) {
		float scale_l = 1 / max_l,
			scale_r = 1 / max_r;
#pragma omp parallel for schedule(static)\
	firstprivate(scale_l, scale_r)
		for (long i = 0; i < in->frames() + ir->frames() - 1; i++) {
			out_l[i] *= scale_l;
			out_r[i] *= scale_r;
		}
		MonoStereoConversion::combine2Channels(out_l, out_r, out_stereo, in->frames() + ir->frames() - 1, &max);

		out->writef(out_stereo, in->frames() + ir->frames() - 1);
	}
	else {
		float scale = 1 / max;
#pragma omp parallel for schedule(static)\
	firstprivate(scale)
		for (long i = 0; i < in->frames() + ir->frames() - 1; i++)
			out_l[i] *= scale;

		out->writef(out_l, in->frames() + ir->frames() - 1);
	}

}

//somewhat parallelised
void ReverbEffect::OLA_mono() {

	cache = new float[N * IR_blocks];
	memset(cache, 0, sizeof(float)* (N * IR_blocks));

	max = 0.0f;

	double avg_dft_time = 0.0f,
		  avg_ola_conv_time = 0.0f;

	omp_set_num_threads(8);

	for (long i = 0; i < in->frames(); i += L) {
	
		QueryPerformanceCounter(&start);
		//FFT input block
		if (i + L > in->frames()) {
			DFT(in_l + i, (in->frames() - i), in_src_l, IN_L);
		}
		else {
			DFT(in_l + i, L, in_src_l, IN_L);
		}
		QueryPerformanceCounter(&end);
		elapsedTime = (end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
		avg_dft_time += elapsedTime;

		//multiply & OLA with each piece of IR
		for (long j = 0; j < IR_blocks; j++) {

			complexMul(OUT_SRC_L, NULL, IN_L, NULL, 0, IR_L, NULL, j);
			IFT();

			//ovaj for loop pojede dosta vremena!!
			//valjalo bi da se izmeni ili iskoristi nekako memcpy...
			QueryPerformanceCounter(&start);
#pragma omp parallel for schedule(static)\
	firstprivate(i, j)
			for (long k = 0; k < N; k++) {
				if (i + j * M + k < out_sz) {
					out_l[i + j * M + k] += (in_src_l[k] / N);
					//perhaps a critical section for this if clause?
					if (fabs(out_l[i + j * M + k]) > max)
						max = fabs(out_l[i + j * M + k]);
				}
			}
			QueryPerformanceCounter(&end);
			
		}
		elapsedTime = (end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
		avg_ola_conv_time += elapsedTime;

	}

	cout << "Avg FFT(input) time: " << avg_dft_time / (ceil((double)in->frames() / L)) << "[ms]" << endl
		<< "Avg OLA Convolution for loop time per input block: " << avg_ola_conv_time / (ceil((double)in->frames() / (L * IR_blocks))) << "[ms]" << endl;

}
//somewhat parallelised
void ReverbEffect::OLA_stereo() {

	//redundant for now... perhaps use this to lower the complexity...
	cache = new float[N * IR_blocks];
	memset(cache, 0, sizeof(float)* (N * IR_blocks));

	max_l = max_r = 0.0f;

	double avg_dft_time = 0.0f,
		avg_ola_conv_time = 0.0f;

	for (long i = 0; i < in->frames(); i += L) {

		QueryPerformanceCounter(&start);
		//FFT input block
		if (i + L > in->frames()) {
			DFT(in_l + i, (in->frames() - i), in_src_l, IN_L);
			DFT(in_r + i, (in->frames() - i), in_src_r, IN_R);
		}
		else {
			DFT(in_l + i, L, in_src_l, IN_L);
			DFT(in_r + i, L, in_src_r, IN_R);
		}
		QueryPerformanceCounter(&end);
		elapsedTime = (end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
		avg_dft_time += elapsedTime;

		QueryPerformanceCounter(&start);
		//multiply & OLA with each piece of IR
		for (long j = 0; j < IR_blocks; j++) {

			complexMul(OUT_SRC_L, OUT_SRC_R, IN_L, IN_R, 0, IR_L, IR_R, j);
			IFT();
			
#pragma omp parallel for schedule(static)\
	firstprivate(i, j)
			for (long k = 0; k < N; k++) {
				if (i + j * M + k < out_sz / 2) {
					out_l[i + j * M + k] += (in_src_l[k] / N);
					out_r[i + j * M + k] += (in_src_r[k] / N);
					//in linux use openMP 3.1 and add reduction(max:max_l and max_r)
					// openMP 3.1 supports max reductions!
					if (out_l[i + j * M + k] < 0) {
						if (0 - out_l[i + j * M + k] > max_l)
							max_l = 0 - out_l[i + j * M + k];
					}
					else {
						if (out_l[i + j * M + k] > max_l)
							max_l = out_l[i + j * M + k];
					}
					if (out_r[i + j * M + k] < 0) {
						if (0 - out_r[i + j * M + k] > max_r)
							max_r = 0 - out_r[i + j * M + k];
					}
					else {
						if (out_r[i + j * M + k] > max_r)
							max_r = out_r[i + j * M + k];
					}
				}
			}
			

		}
		QueryPerformanceCounter(&end);

		elapsedTime = (end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
		avg_ola_conv_time += elapsedTime;
		
	}

	cout << "Avg FFT(input) time: " << avg_dft_time / (ceil((double)in->frames() / L)) << "[ms]" << endl
		<< "Avg OLA Convolution for loop time per input block: " << avg_ola_conv_time / (IR_blocks) << "[ms]" << endl;

}

void ReverbEffect::DFT(float *in_buf, sf_count_t in_len, float *in_fft, fftwf_complex *OUT_FFT) {

	memset(in_fft, 0, sizeof(float)* N);
	memcpy(in_fft, in_buf, sizeof(float)*in_len);
	memset(OUT_FFT, 0, sizeof(fftwf_complex)* N);
	fftwf_execute_dft_r2c(DFFT, in_fft, OUT_FFT);

}

void ReverbEffect::IFT() {

	memset(in_src_l, 0, sizeof(float)* N);
	fftwf_execute_dft_c2r(IFFT, OUT_SRC_L, in_src_l);

	if (STEREO == channels) {
		memset(in_src_r, 0, sizeof(float)* N);
		fftwf_execute_dft_c2r(IFFT, OUT_SRC_R, in_src_r);
	}

}

//parallelised
void ReverbEffect::complexMul(fftwf_complex *DST_L, fftwf_complex *DST_R, fftwf_complex *SRC1_L, fftwf_complex *SRC1_R, long src1_off,
	fftwf_complex *SRC2_L, fftwf_complex *SRC2_R, long src2_off) {
	
	fftwf_complex dst_l, dst_r, src1_l, src1_r, src2_l, src2_r;

#pragma omp parallel for schedule(static)\
	firstprivate(src1_off, src2_off)\
	private(dst_l, dst_r, src1_l, src1_r, src2_l, src2_r)
	for (long k = 0; k < N / 2 + 1; k++) {

		src1_l[0] = SRC1_L[src1_off * N + k][0]; src1_l[1] = SRC1_L[src1_off * N + k][1];
		if (SRC1_R != NULL) {
			src1_r[0] = SRC1_R[src1_off * N + k][0]; src1_r[1] = SRC1_R[src1_off * N + k][1];
		}
		src2_l[0] = SRC2_L[src2_off * N + k][0]; src2_l[1] = SRC2_L[src2_off * N + k][1];
		if (SRC2_R != NULL) {
			src2_r[0] = SRC2_R[src2_off * N + k][0]; src2_r[1] = SRC2_R[src2_off * N + k][1];
		}

		//L1L2
		dst_l[0] = (src1_l[0] * src2_l[0])
			- (src1_l[1] * src2_l[1]);

		dst_l[1] = (src1_l[1] * src2_l[0])
			+ (src1_l[0] * src2_l[1]);

		if (STEREO == channels) {
			if (STEREO == ir_channels) {
				//TrueStereo

				//L1R2
				dst_l[0] += (src1_l[0] * src2_r[0])
					- (src1_l[1] * src2_r[1]);

				dst_l[1] += (src1_l[1] * src2_r[0])
					+ (src1_l[0] * src2_r[1]);

				dst_l[0] /= 2;
				dst_l[1] /= 2;

				//R1L2
				dst_l[0] = (src1_r[0] * src2_l[0])
					- (src1_r[1] * src2_l[1]);

				dst_l[1] = (src1_r[1] * src2_l[0])
					+ (src1_r[0] * src2_l[1]);

				//R1R2
				dst_r[0] = (src1_r[0] * src2_r[0])
					- (src1_r[1] * src2_r[1]);

				dst_r[1] = (src1_r[1] * src2_r[0])
					+ (src1_r[0] * src2_r[1]);

				dst_r[0] /= 2;
				dst_r[1] /= 2;
			}
			else {
				//QuasiStereo
				//R1L2
				dst_r[0] = (src1_r[0] * src2_l[0])
					- (src1_r[1] * src2_l[1]);

				dst_r[1] = (src1_r[1] * src2_l[0])
					+ (src1_r[0] * src2_l[1]);
			}
		}

		DST_L[k][0] = dst_l[0]; DST_L[k][1] = dst_l[1];
		if (DST_R != NULL) {
			DST_R[k][0] = dst_r[0]; DST_R[k][1] = dst_r[1];
		}
	}
}

void ReverbEffect::init_files(SndfileHandle *in, SndfileHandle *ir, SndfileHandle *out) {

	this->in = in;
	this->ir = ir;
	this->out = out;

	channels = in->channels();

}

void ReverbEffect::init_fftws() {

	IR_blocks = (long)ceil((double)ir->frames() / M);

	in_src_l = new float[N];
	in_src_r = new float[N];
	OUT_SRC_L = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)* N);
	OUT_SRC_R = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)* N);
	IN_L = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)* N);
	IN_R = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)* N);
	IR_L = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)* IR_blocks * N);
	IR_R = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)* IR_blocks * N);


	memset(in_src_l, 0, sizeof(float)* N);
	memset(in_src_r, 0, sizeof(float)* N);
	memset(OUT_SRC_L, 0, sizeof(fftwf_complex)* N);
	memset(OUT_SRC_R, 0, sizeof(fftwf_complex)* N);
	memset(IN_L , 0, sizeof(fftwf_complex)* N);
	memset(IN_R, 0, sizeof(fftwf_complex)* N);
	memset(IR_L, 0, sizeof(fftwf_complex)* IR_blocks * N);
	memset(IR_R, 0, sizeof(fftwf_complex)* IR_blocks * N);

	DFFT = fftwf_plan_dft_r2c_1d(N, in_src_l, OUT_SRC_L, FFTW_ESTIMATE);
	IFFT = fftwf_plan_dft_c2r_1d(N, OUT_SRC_L, in_src_l, FFTW_ESTIMATE);

}

void ReverbEffect::init_in_out_mono() {
	
	in_sz = in->frames();
	in_l = new float[in_sz];
	memset(in_l, 0, sizeof(in_sz));
	in->readf(in_l, in_sz);

	out_sz = in->frames() + ir->frames() - 1;
	out_l = new float[out_sz];
	memset(out_l, 0, sizeof(float)* out_sz);

}

void ReverbEffect::init_in_out_stereo() {

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

//can't be parallelised.... something wierd with DFT
void ReverbEffect::init_ir_mono() {

	ir_sz = ir->frames();
	ir_channels = 1;
	ir_l = new float[ir_sz];
	memset(ir_l, 0, sizeof(float)* ir_sz);
	ir->readf(ir_l, ir->frames());

	for (long i = 0; i < ir_sz; i += M) {
		if (i + M > ir->frames()) {
			DFT(ir_l + i, ir->frames() - i, in_src_l, IR_L + (i / M) * N);
		}
		else {
			DFT(ir_l + i, M, in_src_l, IR_L + (i / M) * N);
		}
	}

}
//can't be parallelised.... something wierd with DFT
void ReverbEffect::init_ir_stereo() {

	ir_sz = ir->frames() * 2;
	ir_channels = 2;
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
			DFT(ir_l + i, ir->frames() - i, in_src_l, IR_L + (i / M) * N);
			DFT(ir_r + i, ir->frames() - i, in_src_r, IR_R + (i / M) * N);
		}
		else {
			DFT(ir_l + i, M, in_src_l, IR_L + (i / M) * N);
			DFT(ir_r + i, M, in_src_r, IR_R + (i / M) * N);
		}
	}

}