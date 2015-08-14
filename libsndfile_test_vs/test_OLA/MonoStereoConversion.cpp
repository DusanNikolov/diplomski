

#include "MonoStereoConversion.h"
#include <cmath>

using namespace std;

void MonoStereoConversion::extractChannel(float* src, float* dst, int channelNo, int channelCnt, sf_count_t srcLen) {

	long dst_i = 0;
	for (long src_i = 0; src_i < srcLen; src_i += channelCnt) {
		dst[dst_i++] = src[src_i + (channelNo - 1)];
	}

}

void MonoStereoConversion::extractBothChannels(float* src, float* channel1, float* channel2, sf_count_t srcLen) {
	long dst_i = 0;
	for (long src_i = 0; src_i < srcLen; src_i += 2) {
		channel1[dst_i]   = src[src_i];
		channel2[dst_i++] = src[src_i + 1];
	}
}

//parallelised
void MonoStereoConversion::combine2Channels(float* src1, float* src2, float* dst, sf_count_t src_len, float* maxValue) {
	
	float tempMax = 0;

#pragma omp parallel for schedule(static)
	for (long src_i = 0; src_i < src_len; src_i++) {
		dst[2 * src_i] = src1[src_i];
		dst[2 * src_i + 1] = src2[src_i];
	}

	(*maxValue) = tempMax;

}

void MonoStereoConversion::copy1Channel(float* src, float* dst, sf_count_t src_len, float* maxValue) {

	float tempMax = 0;

	for (long i = 0; i < src_len; i++) {
		dst[i] = src[i];
		if (fabs(src[i]) > tempMax)
			tempMax = fabs(src[i]);
	}

	(*maxValue) = tempMax;

}

void MonoStereoConversion::normalize(float* buffer, sf_count_t size, float maxValue) {

	for (long i = 0; i < size; i++) {
		buffer[i] = buffer[i] / maxValue;
	}

}