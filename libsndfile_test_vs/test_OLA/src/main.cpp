// test OLA - overlap & add
// mono files are ok...
// TO-DO: True stereo convolution (bineural IR & stereo input) requires 4 convolutions (Ll, Lr, Rl,
// Rr)!

#include "ReverbEffect.h"

#include <fftw3.h>
#include <sndfile.hh>

#include <cmath>
#include <cstring>
#include <iostream>

using namespace std;

int main(int argc, char **argv)
{

    if (argc < 3)
    {
        cerr << "Not enough parameters, see README for instructions!" << endl;
        return -1;
    }

    SndfileHandle *in, *ir, *out;

    in = new SndfileHandle(argv[1]);
    if (in->error() != SF_ERR_NO_ERROR)
    {
        cerr << "Input file not recognized!" << endl;

        delete in;
        return -1;
    }

    ir = new SndfileHandle(argv[2]);
    if (ir->error() != SF_ERR_NO_ERROR)
    {
        cerr << "IR file not recognized!" << endl;

        delete in;
        delete ir;
        return -1;
    }

    int in_fn_len = strlen(argv[1]);
    char *out_fn = new char[in_fn_len + 4];

    strncpy(out_fn, argv[1], sizeof(char) * (in_fn_len - 4));
    strcpy(out_fn + in_fn_len - 4, "_out.wav");

    out = new SndfileHandle(out_fn,
                            SFM_WRITE,
                            SF_FORMAT_WAV | SF_FORMAT_PCM_16,
                            in->channels(),
                            44100);
    if (out->error() != SF_ERR_NO_ERROR)
    {
        cerr << "Output file not formed!" << endl;

        delete in;
        delete ir;
        delete out;
        return -1;
    }

    out->command(SFC_SET_UPDATE_HEADER_AUTO, NULL, SF_TRUE);

    cout << "::::: " << argv[1] << " : " << argv[2] << " :::::" << endl;

    ReverbEffect *effect = new ReverbEffect(in, ir, out);

    effect->applyReverb();
    effect->writeOutNormalized();

    cout << endl;

    delete effect;

    return 0;
}
