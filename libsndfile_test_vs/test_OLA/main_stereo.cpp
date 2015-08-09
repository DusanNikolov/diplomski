//test OLA - overlap & add
//mono files are ok...
//TO-DO: True stereo convolution (bineural IR & stereo input) requires 4 convolutions (Ll, Lr, Rl, Rr)!


#include <iostream>
using namespace std;

#include <sndfile.hh>
#include <fftw3.h>

#include "ReverbEffect.h"

int main(int argc, char** argv) {

	if (argc < 4) {
		cerr << "Not enough parameters, see README for instructions!" << endl;
		return -1;
	}

	ReverbEffect* effect = new ReverbEffect(argv[1], argv[2], argv[3]);

	effect->applyReverb();
	effect->writeOutNormalized();

	cout << "KRAJ" << endl;
	int a;
	cin >> a;

	delete effect;

	return 0;
}
