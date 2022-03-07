#pragma once

#include <sndfile.h>

class MonoStereoConversion {
public:
	//channelNo starts from value 1
	static void extractChannel(float* src, float* dst, int channelNo, int channelCnt, sf_count_t src_len);
	static void extractBothChannels(float* src, float* channel1, float* channel2, sf_count_t src_len);

	static void combine2Channels(float* src1, float* src2, float* dst, sf_count_t src_len, float* maxValue);
	static void copy1Channel(float* src, float* dst, sf_count_t src_len, float* maxValue);
	
	static void normalize(float* buffer, sf_count_t size, float maxValue);
};
