#!/bin/bash

BUILD_DIR=build

if [ ! -z "$1" ] && [ "$1" == "--clean" ]; then
    echo "Cleaning up..."
    rm -rf "${BUILD_DIR}"
fi

if [ ! -d "${BUILD_DIR}" ]; then
    echo "Creating build directory..."
    mkdir "${BUILD_DIR}"
fi

pushd "${BUILD_DIR}"
echo "Building..."
clang++ -o main ../src/main.cpp ../src/ReverbEffect.cpp ../src/MonoStereoConversion.cpp ../src/Timer.cpp -fopenmp=libomp -lfftw3 -lfftw3f `pkgconf --libs sndfile`
popd

echo "Done!"