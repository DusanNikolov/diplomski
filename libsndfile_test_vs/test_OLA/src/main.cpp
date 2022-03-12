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
#include <memory>

int main(int argc, char **argv)
{

    if (argc < 3)
    {
        std::cerr << "Not enough parameters, see README for instructions!" << std::endl;
        return -1;
    }

    auto in = std::make_unique<SndfileHandle>(argv[1]);
    if (in->error() != SF_ERR_NO_ERROR)
    {
        std::cerr << "Input file not recognized!" << std::endl;
        return -1;
    }

    auto ir = std::make_unique<SndfileHandle>(argv[2]);
    if (ir->error() != SF_ERR_NO_ERROR)
    {
        std::cerr << "IR file not recognized!" << std::endl;
        return -1;
    }

    int in_fn_len = strlen(argv[1]);
    char *out_fn = new char[in_fn_len + 4];

    strncpy(out_fn, argv[1], sizeof(char) * (in_fn_len - 4));
    strcpy(out_fn + in_fn_len - 4, "_out.wav");

    auto out = std::make_unique<SndfileHandle>(out_fn,
                                               SFM_WRITE,
                                               SF_FORMAT_WAV | SF_FORMAT_PCM_16,
                                               in->channels(),
                                               44100);
    if (out->error() != SF_ERR_NO_ERROR)
    {
        std::cerr << "Output file not formed!" << std::endl;
        return -1;
    }

    out->command(SFC_SET_UPDATE_HEADER_AUTO, NULL, SF_TRUE);

    std::cout << "::::: " << argv[1] << " : " << argv[2] << " :::::" << std::endl;

    auto effect = std::make_unique<ReverbEffect>(in.get(), ir.get(), out.get());

    effect->applyReverb();
    effect->writeOutNormalized();

    return 0;
}
