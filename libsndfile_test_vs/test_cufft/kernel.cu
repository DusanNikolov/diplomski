
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cufft.h>
#include <cufftw.h>

#include <sndfile.hh>

#include <iostream>
#include <iomanip>

using namespace std;

typedef float2 Complex;

cudaError_t fftWithCuda();

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

int main()
{

	SndfileHandle* in = new SndfileHandle("D:/Documents/GitHub/Diplomski/libsndfile_test_vs/test_cufft/wav_files/guitar_mono.wav");
	SndfileHandle* out = new SndfileHandle("D:/Documents/GitHub/Diplomski/libsndfile_test_vs/test_cufft/wav_files/out_guit.wav",
		SFM_WRITE, SF_FORMAT_WAV | SF_FORMAT_PCM_16, 1, 44100);
	out->command(SFC_SET_UPDATE_HEADER_AUTO, NULL, SF_TRUE);

	float *in_buf = new float[in->frames()];
	in->readf(in_buf, in->frames());

	long N = in->frames();

	N--;
	N |= N >> 1;
	N |= N >> 2;
	N |= N >> 4;
	N |= N >> 8;
	N |= N >> 16;
	N++;

	long FFT_SIZE = N;
	long ACTUAL_FFT_SIZE = (FFT_SIZE / 2 + 1);

	cout << "FFT_SIZE: " << FFT_SIZE << endl;

	long sig_mem_size = sizeof(cufftReal)* (ACTUAL_FFT_SIZE * 2);
	long fft_mem_size = sizeof(Complex)* ACTUAL_FFT_SIZE;

	cout << "sig_mem_size: " << sig_mem_size << endl
		<< "fft_mem_size: " << fft_mem_size << endl;

	cudaError_t cudaStatus;

	cufftReal *in_padded;
	cudaStatus = cudaMalloc((void**)&in_padded, sig_mem_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return 1;
	}

	cudaStatus = cudaMemset(in_padded, 0, sig_mem_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		return 2;
	}

	cudaStatus = cudaMemcpy(in_padded, in_buf, sizeof(cufftReal)*100, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return 3;
	}

	cufftHandle dfft, ifft;
	cufftPlan1d(&dfft, FFT_SIZE, CUFFT_R2C, 1);
	cufftPlan1d(&ifft, FFT_SIZE, CUFFT_C2R, 1);

	
	cufftExecR2C(dfft, (cufftReal*)in_padded, (Complex*)in_padded);
	cufftExecC2R(ifft, (Complex*)in_padded, (cufftReal*)in_padded);

	float* out_buf = new float[100];

	cudaStatus = cudaMemcpy(out_buf, in_padded, sizeof(cufftReal)*100, cudaMemcpyDeviceToHost);


	for (int i = 0; i < 10; i++)
		cout << "in[" << i << "]: " << setprecision(3) << in_buf[i] << '\t' << "out[" << i << "]: " << setprecision(3) << out_buf[i] / FFT_SIZE << endl;


	//out->writef(out_buf, in->frames());


    return 0;
}

cudaError_t fftWithCuda() {

	cudaError_t cudaStatus = cudaSuccess;

	

	return cudaStatus;
}

// Helper function for using CUDA to add vectors in parallel.
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
//{
//    int *dev_a = 0;
//    int *dev_b = 0;
//    int *dev_c = 0;
//    cudaError_t cudaStatus;
//
//    // Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//
//    // Allocate GPU buffers for three vectors (two input, one output)    .
//    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    // Copy input vectors from host memory to GPU buffers.
//    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    // Launch a kernel on the GPU with one thread for each element.
//    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        goto Error;
//    }
//    
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    // Copy output vector from GPU buffer to host memory.
//    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//Error:
//    cudaFree(dev_c);
//    cudaFree(dev_a);
//    cudaFree(dev_b);
//    
//    return cudaStatus;
//}
