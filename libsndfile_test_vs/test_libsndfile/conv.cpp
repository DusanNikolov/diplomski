//convolution code from the net... just for testing purposes

#include <cstdlib>

using namespace std;

//convolution algorithm
float *conv(float *A, float *B, int lenA, int lenB, int *lenC)
{
	int nconv;
	int i, j, i1;
	float tmp;
	float *C;

	//allocated convolution array   
	nconv = lenA + lenB - 1;
	C = (float*)calloc(nconv, sizeof(float));

	//convolution process
	for (i = 0; i<nconv; i++)
	{
		i1 = i;
		tmp = 0.0;
		for (j = 0; j<lenB; j++)
		{
			if (i1 >= 0 && i1<lenA)
				tmp = tmp + (A[i1] * B[j]);

			i1 = i1 - 1;
			C[i] = tmp;
		}
	}

	//get length of convolution array   
	(*lenC) = nconv;

	//return convolution array
	return(C);
}