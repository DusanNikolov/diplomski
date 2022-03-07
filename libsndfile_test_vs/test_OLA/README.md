# Convolution reverberation implementation using FFTW

## Building

Instructions for building with clang (default version on Ubuntu 20.04):

### Prerequisites

1. Install clang: `sudo apt install clang`
2. Install pkgconf: `sudo apt install pkgconf` (this is useful to link libsndfile)
2. Install libsndfile: `sudo apt install libsndfile1-dev`
3. Install fftw: `sudo apt install libfftw3-dev`

### Actually building it
1. mkdir build && cd build
2. `clang++ -o main ../main.cpp ../ReverbEffect.cpp ../MonoStereoConversion.cpp ../Timer.cpp -fopenmp=libomp -lfftw3 -lfftw3f `pkgconf --libs sndfile``

## Running

Program has two arguments:
   1: input WAV filepath
   2: impulse response WAV filepath

Program will create new file named "<input_filename>_out.wav"
