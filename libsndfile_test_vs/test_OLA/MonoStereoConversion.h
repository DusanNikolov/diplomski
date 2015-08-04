#ifndef MS_CONVERSION_H
#define MS_CONVERSION_H

class MonoStereoConversion {
public:
	//channelNo starts from value 1
	static void extractChannel(float* src, float* dst, int channelNo, int channelCnt, long src_len);
	static void extractBothChannels(float* src, float* channel1, float* channel2, long src_len);
	static void combine2Channels(float* src1, float* src2, float* dst, long src_len, float* maxValue);
	static void copy1Channel(float* src, float* dst, long src_len, float* maxValue);
	static void normalize(float* buffer, long size, float maxValue);
};

#endif