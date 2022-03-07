//test OLA - overlap & add
//mono files are ok...
//TO-DO: True stereo convolution (bineural IR & stereo input) requires 4 convolutions (Ll, Lr, Rl, Rr)!


#include <iostream>
using namespace std;

#include <sndfile.hh>
#include <fftw3.h>

#define N 2048
#define M 1024
#define L 512


int main(int* argc, char** argv) {

#pragma region init_files
	SndfileHandle *in, *ir, *out;
	int channels, format = SF_FORMAT_WAV | SF_FORMAT_PCM_16, samplerate = 44100;

	in = new SndfileHandle(argv[1]);
	channels = in->channels();
	ir = new SndfileHandle(argv[2]);
	out = new SndfileHandle(argv[3], SFM_WRITE, format, channels, samplerate);
	out->command(SFC_SET_UPDATE_HEADER_AUTO, NULL, SF_TRUE);
#pragma endregion

#pragma region init_fftw_vars
	float *in_src = new float[N];
	fftwf_complex *out_src, *IR, *IN;
	fftwf_plan plan_dfft, plan_ifft;

	long IR_blocks = (long)ceil((double)ir->frames() / M);

	out_src = (fftwf_complex*)malloc(sizeof(fftwf_complex)* N);
	IN = (fftwf_complex*)malloc(sizeof(fftwf_complex)* N);
	IR = (fftwf_complex*)malloc(sizeof(fftwf_complex)* IR_blocks * N);

	memset(in_src, 0, sizeof(float)* N);
	memset(out_src, 0, sizeof(fftwf_complex)* N);
	memset(IN, 0, sizeof(fftwf_complex)* N);
	memset(IR, 0, sizeof(fftwf_complex)* IR_blocks * N);


	plan_dfft = fftwf_plan_dft_r2c_1d(N, in_src, out_src, FFTW_ESTIMATE);
	plan_ifft = fftwf_plan_dft_c2r_1d(N, out_src, in_src, FFTW_ESTIMATE);
#pragma endregion

#pragma region init_IR
	long ir_sz = IR_blocks * M;
	float* ir_mono = new float[ir_sz];
	memset(ir_mono, 0, sizeof(float)* ir_sz);
	ir->readf(ir_mono, ir->frames());

	for (long i = 0; i < ir_sz; i += M) {
		if (i + M > ir->frames()) {
			memset(in_src, 0, sizeof(float)* N);
			memcpy(in_src, ir_mono + i, sizeof(float)* (ir->frames() - i));
			fftwf_execute_dft_r2c(plan_dfft, in_src, IR + (i / M) * N);
		}
		else {
			memset(in_src, 0, sizeof(float)* N);
			memcpy(in_src, ir_mono + i, sizeof(float)* M);
			fftwf_execute_dft_r2c(plan_dfft, in_src, IR + (i / M) * N);
		}
	}
#pragma endregion

#pragma region init_in_out
	//long in_sz = (long)ceil((double)in->frames() / L) * L;
	float *in_mono = new float[in->frames()];
	memset(in_mono, 0, sizeof(in->frames()));
	in->readf(in_mono, in->frames());

	long out_blocks = in->frames() + ir->frames() - 1;
	float* out_mono = new float[out_blocks];
	memset(out_mono, 0, sizeof(float)* out_blocks);
#pragma endregion

#pragma region OLA

	float *cache = new float[L + ir->frames() - 1];
	memset(cache, 0, sizeof(float)* (L + ir->frames() - 1));

	float max = 0.0f;

	for (long i = 0; i < in->frames(); i += L) {

#pragma region FFT(input_blk)
		memset(in_src, 0, sizeof(float)* N);
		
		if (i + L > in->frames()) {
			memcpy(in_src, in_mono + i, sizeof(float)* (in->frames() - i));
		}
		else {
			memcpy(in_src, in_mono + i, sizeof(float)* L);
		}

		memset(IN, 0, sizeof(fftwf_complex)* N);
		fftwf_execute_dft_r2c(plan_dfft, in_src, IN);
#pragma endregion

		for (long j = 0; j < IR_blocks; j++) {

#pragma region complex mul
			for (long k = 0; k < N / 2 + 1; k++) {
				out_src[k][0] = (IN[k][0] * IR[j * N + k][0]) - (IN[k][1] * IR[j * N + k][1]);
				out_src[k][1] = (IN[k][1] * IR[j * N + k][0]) + (IN[k][0] * IR[j * N + k][1]);
			}
#pragma endregion

#pragma region IFFT(result)
			memset(in_src, 0, sizeof(float)* N);
			fftwf_execute_dft_c2r(plan_ifft, out_src, in_src);
#pragma endregion
			
			for (long k = 0; k < N; k++) {
				if (i + j * M + k < in->frames() + ir->frames() - 1) {
					out_mono[i + j * M + k] += (in_src[k] / N);
					if (fabs(out_mono[i + j * M + k]) > max)
						max = fabs(out_mono[i + j * M + k]);
				}
			}



		}

	}

#pragma endregion

	//normalization / scaling down
	float scale = 1 / max;
	for (long i = 0; i < in->frames() + ir->frames() - 1; i++)
		out_mono[i] *= scale;
		

	out->writef(out_mono, in->frames() + ir->frames() - 1);


	cout << "KRAJ" << endl;
	int a;
	cin >> a;

	return 0;
}
