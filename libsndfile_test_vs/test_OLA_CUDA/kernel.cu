
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "device_functions.cuh"

// Padding functions
int PadData(const Complex *, Complex **, int,
	const Complex *, Complex **, int);

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv)
{

	findCudaDevice(argc, (const char **)argv);

	// Allocate host memory for the signal
	Complex *h_signal = (Complex *)malloc(sizeof(Complex)* SIGNAL_SIZE);

	// Initalize the memory for the signal
	for (unsigned int i = 0; i < SIGNAL_SIZE; ++i)
	{
		h_signal[i].x = rand() / (float)RAND_MAX;
		h_signal[i].y = 0;
	}

	// Allocate host memory for the filter
	Complex *h_filter_kernel = (Complex *)malloc(sizeof(Complex)* FILTER_KERNEL_SIZE);

	// Initalize the memory for the filter
	for (unsigned int i = 0; i < FILTER_KERNEL_SIZE; ++i)
	{
		h_filter_kernel[i].x = rand() / (float)RAND_MAX;
		h_filter_kernel[i].y = 0;
	}

	// Pad signal and filter kernel
	Complex *h_padded_signal;
	Complex *h_padded_filter_kernel;
	int new_size = PadData(h_signal, &h_padded_signal, SIGNAL_SIZE,
		h_filter_kernel, &h_padded_filter_kernel, FILTER_KERNEL_SIZE);
	int mem_size = sizeof(Complex)* new_size;

	// Allocate device memory for signal
	Complex *d_signal;
	checkCudaErrors(cudaMalloc((void **)&d_signal, mem_size));
	// Copy host memory to device
	checkCudaErrors(cudaMemcpy(d_signal, h_padded_signal, mem_size,
		cudaMemcpyHostToDevice));

	// Allocate device memory for filter kernel
	Complex *d_filter_kernel;
	checkCudaErrors(cudaMalloc((void **)&d_filter_kernel, mem_size));

	// Copy host memory to device
	checkCudaErrors(cudaMemcpy(d_filter_kernel, h_padded_filter_kernel, mem_size,
		cudaMemcpyHostToDevice));

	// CUFFT plan
	cufftHandle plan;
	checkCudaErrors(cufftPlan1d(&plan, new_size, CUFFT_C2C, 1));

	// Transform signal and kernel
	printf("Transforming signal cufftExecC2C\n");
	checkCudaErrors(cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_FORWARD));
	checkCudaErrors(cufftExecC2C(plan, (cufftComplex *)d_filter_kernel, (cufftComplex *)d_filter_kernel, CUFFT_FORWARD));

	// Multiply the coefficients together and normalize the result
	printf("Launching ComplexPointwiseMulAndScale<<< >>>\n");
	ComplexPointwiseMulAndScale << <32, 256 >> >(d_signal, d_filter_kernel, new_size, 1.0f / new_size);

	// Check if kernel execution generated and error
	getLastCudaError("Kernel execution failed [ ComplexPointwiseMulAndScale ]");

	// Transform signal back
	printf("Transforming signal back cufftExecC2C\n");
	checkCudaErrors(cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_INVERSE));

	// Copy device memory to host
	Complex *h_convolved_signal = h_padded_signal;
	checkCudaErrors(cudaMemcpy(h_convolved_signal, d_signal, mem_size,
		cudaMemcpyDeviceToHost));

	// Allocate host memory for the convolution result
	Complex *h_convolved_signal_ref = (Complex *)malloc(sizeof(Complex)* SIGNAL_SIZE);

	// Convolve on the host
	Convolve(h_signal, SIGNAL_SIZE,
		h_filter_kernel, FILTER_KERNEL_SIZE,
		h_convolved_signal_ref);

	// check result
	bool bTestResult = sdkCompareL2fe((float *)h_convolved_signal_ref, (float *)h_convolved_signal, 2 * SIGNAL_SIZE, 1e-5f);

	//Destroy CUFFT context
	checkCudaErrors(cufftDestroy(plan));

	// cleanup memory
	free(h_signal);
	free(h_filter_kernel);
	free(h_padded_signal);
	free(h_padded_filter_kernel);
	free(h_convolved_signal_ref);
	checkCudaErrors(cudaFree(d_signal));
	checkCudaErrors(cudaFree(d_filter_kernel));

	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	cudaDeviceReset();
	exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

// Pad data
int PadData(const Complex *signal, Complex **padded_signal, int signal_size,
	const Complex *filter_kernel, Complex **padded_filter_kernel, int filter_kernel_size)
{
	int minRadius = filter_kernel_size / 2;
	int maxRadius = filter_kernel_size - minRadius;
	int new_size = signal_size + maxRadius;

	// Pad signal
	Complex *new_data = (Complex *)malloc(sizeof(Complex)* new_size);
	memcpy(new_data + 0, signal, signal_size * sizeof(Complex));
	memset(new_data + signal_size, 0, (new_size - signal_size) * sizeof(Complex));
	*padded_signal = new_data;

	// Pad filter
	new_data = (Complex *)malloc(sizeof(Complex)* new_size);
	memcpy(new_data + 0, filter_kernel + minRadius, maxRadius * sizeof(Complex));
	memset(new_data + maxRadius, 0, (new_size - filter_kernel_size) * sizeof(Complex));
	memcpy(new_data + new_size - minRadius, filter_kernel, minRadius * sizeof(Complex));
	*padded_filter_kernel = new_data;

	return new_size;
}


////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////

// Complex addition
static __device__ __host__ inline Complex ComplexAdd(Complex a, Complex b)
{
	Complex c;
	c.x = a.x + b.x;
	c.y = a.y + b.y;
	return c;
}

// Complex scale
static __device__ __host__ inline Complex ComplexScale(Complex a, float s)
{
	Complex c;
	c.x = s * a.x;
	c.y = s * a.y;
	return c;
}

// Complex multiplication
static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b)
{
	Complex c;
	c.x = a.x * b.x - a.y * b.y;
	c.y = a.x * b.y + a.y * b.x;
	return c;
}

// Complex pointwise multiplication
static __global__ void ComplexPointwiseMulAndScale(Complex *a, const Complex *b, int size, float scale)
{
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = threadID; i < size; i += numThreads)
	{
		a[i] = ComplexScale(ComplexMul(a[i], b[i]), scale);
	}
}
