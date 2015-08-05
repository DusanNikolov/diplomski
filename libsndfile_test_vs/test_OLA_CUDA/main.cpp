//test OLA CUDA


#include <iostream>
using namespace std;

#include <sndfile.hh>

#include "ReverbEffect.h"

int main(int* argc, char** argv) {

	ReverbEffect* effect = new ReverbEffect(argv[1], argv[2], argv[3]);

	effect->applyReverb();
	effect->writeOutNormalized();

	cout << "KRAJ" << endl;
	int a;
	cin >> a;

	delete effect;

	return 0;
}
