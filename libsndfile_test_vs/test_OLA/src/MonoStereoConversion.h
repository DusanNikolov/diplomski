#pragma once

#include <sndfile.h>

namespace mono_stereo_conversion
{
// channelNo starts from value 1
void extractChannel(float *src, float *dst, int channelNo, int channelCnt, sf_count_t src_len);
void extractBothChannels(float *src, float *channel1, float *channel2, sf_count_t src_len);

void combine2Channels(float *src1, float *src2, float *dst, sf_count_t src_len, float *maxValue);
void copy1Channel(float *src, float *dst, sf_count_t src_len, float *maxValue);

void normalize(float *buffer, sf_count_t size, float maxValue);
} // namespace mono_stereo_conversion
