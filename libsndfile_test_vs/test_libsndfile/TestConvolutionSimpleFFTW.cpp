//first test is to do DFFT and IFFT to return the same, unchanged input stream

//to measure time use QueryPerformanceFrequency
/*	LARGE_INTEGER frequency;        // ticks per second
	LARGE_INTEGER start, end;           // ticks
	double elapsedTime;

	QueryPerformanceFrequency(&frequency);


	QueryPerformanceCounter(&start);
	--CODE THAT IS BEING MEASURED--
	QueryPerformanceCounter(&end);
	
	elapsedTime = (end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
	cout << "left channel dfft time: " << elapsedTime << "[ms]" << endl;
*/


#include <iostream>
#include <Windows.h>

using namespace std;

#include "MonoStereoConversion.h"
#include "Filter.h"

#include <sndfile.hh>
#include <fftw3.h>


int main(int* argc, char** argv) {
	
	SndfileHandle *in, *out;

	in = new SndfileHandle(argv[1]);

	out = new SndfileHandle(argv[2], SFM_WRITE, SF_FORMAT_WAV | SF_FORMAT_PCM_16, 1, 44100);
	//enable header updating after every write
	out->command(SFC_SET_UPDATE_HEADER_AUTO, NULL, SF_TRUE);

	float* in_lr, *in_mono;
	in_lr = new float[in->frames() * 2];
	in_mono = new float[in->frames()];

	in->readf(in_lr, in->frames());

	MonoStereoConversion::extractChannel(in_lr, in_mono, 1, 2, in->frames() * 2);

	out->writef(in_mono, in->frames());

}