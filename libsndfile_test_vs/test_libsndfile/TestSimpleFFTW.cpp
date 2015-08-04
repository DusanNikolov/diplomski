//first test is to do DFFT and IFFT to return the same, unchanged input stream

#include <iostream>
#include <Windows.h>

using namespace std;

#include "MonoStereoConversion.h"
#include <sndfile.hh>
#include <fftw3.h>


int main() {

	const int format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
	const int sampleRate = 44100;

	// TO-DO: moraces da promenis nacin dohvatanja wav fajlova, ovako nece da bude fleksibilno!
	SndfileHandle* guit_cl = new SndfileHandle("D:/OneDrive/Fakultet/diplomski/development/testiranje_cli/libsndfile_test_vs/test_libsndfile/wav_files/guitar_clean.wav");

	SndfileHandle* guit_fftw = new SndfileHandle("D:/OneDrive/Fakultet/diplomski/development/testiranje_cli/libsndfile_test_vs/test_libsndfile/wav_files/guitar_fftw.wav",
		SFM_WRITE, format, 2, sampleRate);

	//neophodno podesiti kada se otvara fajl za upis, da bi se azurirao header file pri izmeni data sekcije fajla!
	guit_fftw->command(SFC_SET_UPDATE_HEADER_AUTO, NULL, SF_TRUE); // SF_FALSE - za prekid azuriranja headera nakon svakog upisa u fajl

	cout << "GUITAR_CLEAN" << endl
		<< "Samplerate: " << guit_cl->samplerate() << endl
		<< "Channels: " << guit_cl->channels() << endl
		<< "Frames: " << guit_cl->frames() << endl;

	//GUITAR:
	long N = guit_cl->frames();
	N--;
	N |= N >> 1;
	N |= N >> 2;
	N |= N >> 4;
	N |= N >> 8;
	N |= N >> 16;
	N++;

	float* guit_clean_lr = new float[guit_cl->frames() * guit_cl->channels()];
	float* guit_clean_l = new float[N];
	float* guit_clean_r = new float[N];
	//read guitar data stereo
	guit_cl->readf(guit_clean_lr, guit_cl->frames());
	//extract left and right channel from stereo
	MonoStereoConversion::extractBothChannels(guit_clean_lr, guit_clean_l, guit_clean_r, guit_cl->frames() * guit_cl->channels());
	//MonoStereoConversion::extractChannel(guit_clean_lr, guit_clean_l, 1, guit_cl->channels(), guit_cl->frames() * guit_cl->channels());
	//MonoStereoConversion::extractChannel(guit_clean_lr, guit_clean_r, 2, guit_cl->channels(), guit_cl->frames() * guit_cl->channels());


	//FFTW PART:
	float* guit_fft_l = new float[N];
	float* guit_fft_r = new float[N];
	float* guit_fft_lr = new float[guit_cl->frames()  * guit_cl->channels()];

	float maxValue = 0;

	//zero padding the input vectors
	for (long i = guit_cl->frames(); i < N; i++) {
		guit_clean_l[i] = 0;
		guit_clean_r[i] = 0;
	}

	fftw_complex *in_src, *out_src;
	fftw_plan p_forw, p_back;

	in_src = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)* N);
	out_src = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)* N);
	
	p_forw = fftw_plan_dft_1d(N, in_src, out_src, FFTW_FORWARD, FFTW_MEASURE);
	p_back = fftw_plan_dft_1d(N, out_src, in_src, FFTW_BACKWARD, FFTW_MEASURE);


	LARGE_INTEGER frequency;        // ticks per second
	LARGE_INTEGER start, end;           // ticks
	double elapsedTime;

	QueryPerformanceFrequency(&frequency);


	QueryPerformanceCounter(&start);
	//left channel dfft
	for (long i = 0; i < N; i++) {
		in_src[i][0] = guit_clean_l[i];
		in_src[i][1] = 0.0;
	}

	fftw_execute(p_forw);

	QueryPerformanceCounter(&end);
	elapsedTime = (end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
	cout << "left channel dfft time: " << elapsedTime << "[ms]" << endl;


	//reset memory buffer
	memset(in_src, 0, sizeof(fftw_complex)* N);


	QueryPerformanceCounter(&start);
	//left channel ifft
	fftw_execute(p_back);

	for (long i = 0; i < N; i++)
		guit_fft_l[i] = in_src[i][0] / N;

	QueryPerformanceCounter(&end);
	elapsedTime = (end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
	cout << "left channel ifft time: " << elapsedTime << "[ms]" << endl;


	QueryPerformanceCounter(&start);
	//right channel dfft
	for (long i = 0; i < N; i++) {
		in_src[i][0] = guit_clean_r[i];
		in_src[i][1] = 0.0;
	}

	fftw_execute(p_forw);

	QueryPerformanceCounter(&end);
	elapsedTime = (end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
	cout << "right channel dfft time: " << elapsedTime << "[ms]" << endl;

	//reset memory buffer
	memset(in_src, 0, sizeof(fftw_complex)* N);


	QueryPerformanceCounter(&start);
	//right channel ifft
	fftw_execute(p_back);

	for (long i = 0; i < N; i++)
		guit_fft_r[i] = in_src[i][0] / N;

	QueryPerformanceCounter(&end);
	elapsedTime = (end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
	cout << "right channel ifft time: " << elapsedTime << "[ms]" << endl;



	QueryPerformanceCounter(&start);
	//combine left and right channels to stereo
	MonoStereoConversion::combine2Channels(guit_fft_l, guit_fft_r, guit_fft_lr, guit_cl->frames(), &maxValue);
	// mozda ne najbolji nacin normalizacije! bez normalizacije se desava odsecanje zbog kog je zvuk distorziran
	MonoStereoConversion::normalize(guit_fft_lr, guit_cl->frames() * guit_cl->channels(), maxValue);

	QueryPerformanceCounter(&end);
	elapsedTime = (end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
	cout << "Recombining and normalization time: " << elapsedTime << "[ms]" << endl;



	cout << "No of frames writen to guitar_fftw: " << guit_fftw->writef(guit_fft_lr, guit_cl->frames()) << endl;


	

	delete guit_cl, guit_fftw;
	delete guit_clean_lr, guit_clean_l, guit_clean_r;
	delete guit_fft_lr, guit_fft_l, guit_fft_r;

	fftw_free(in_src); fftw_free(out_src);
	fftw_destroy_plan(p_forw); fftw_destroy_plan(p_back);

	//remove aferward...
	int a;
	cin >> a;

	return 0;
}