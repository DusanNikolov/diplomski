#include <iostream>
#include <sndfile.hh>

#include "MonoStereoConversion.h"
#include "conv.h"

using namespace std;

int main() {

	const int format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
	const int sampleRate = 44100;

	// TO-DO: moraces da promenis nacin dohvatanja wav fajlova, ovako nece da bude fleksibilno!
	SndfileHandle* guit_cl = new SndfileHandle("D:/OneDrive/Fakultet/diplomski/development/testiranje_cli/libsndfile_test_vs/test_libsndfile/wav_files/guitar_clean.wav");

	SndfileHandle* cab_ir = new SndfileHandle("D:/OneDrive/Fakultet/diplomski/development/testiranje_cli/libsndfile_test_vs/test_libsndfile/wav_files/cabinetIR.wav");

	SndfileHandle* hall_ir = new SndfileHandle("D:/OneDrive/Fakultet/diplomski/development/testiranje_cli/libsndfile_test_vs/test_libsndfile/wav_files/LongEchoHallIR.wav");

	SndfileHandle* guit_cab = new SndfileHandle("D:/OneDrive/Fakultet/diplomski/development/testiranje_cli/libsndfile_test_vs/test_libsndfile/wav_files/guitar_cab.wav",
		SFM_WRITE, format, 2, sampleRate);

	SndfileHandle* guit_hall = new SndfileHandle("D:/OneDrive/Fakultet/diplomski/development/testiranje_cli/libsndfile_test_vs/test_libsndfile/wav_files/guitar_hall.wav",
		SFM_WRITE, format, 2, sampleRate);

	//neophodno podesiti kada se otvara fajl za upis, da bi se azurirao header file pri izmeni data sekcije fajla!
	guit_cab->command(SFC_SET_UPDATE_HEADER_AUTO, NULL, SF_TRUE); // SF_TRUE - oznacava da je azuriranje aktivno
	guit_hall->command(SFC_SET_UPDATE_HEADER_AUTO, NULL, SF_TRUE); // SF_FALSE - za prekid azuriranja headera nakon svakog upisa u fajl

	cout << "GUITAR_CLEAN" << endl
		<< "Samplerate: "  << guit_cl->samplerate() << endl
		<< "Channels: "    << guit_cl->channels() << endl
		<< "Frames: "      << guit_cl->frames() << endl;

	cout << "CabinetIR"   << endl
		<< "Samplerate: " << cab_ir->samplerate() << endl
		<< "Channels: "   << cab_ir->channels() << endl
		<< "Frames: "     << cab_ir->frames() << endl;

	cout << "HallIR"      << endl
		<< "Samplerate: " << hall_ir->samplerate() << endl
		<< "Channels: "   << hall_ir->channels() << endl
		<< "Frames: "     << hall_ir->frames() << endl;

	//GUITAR:
	float* guit_clean_lr = new float[guit_cl->frames() * guit_cl->channels()];
	float* guit_clean_l  = new float[guit_cl->frames()];
	float* guit_clean_r  = new float[guit_cl->frames()];
	//read guitar data stereo
	guit_cl->readf(guit_clean_lr, guit_cl->frames());
	//extract left and right channel from stereo
	MonoStereoConversion::extractBothChannels(guit_clean_lr, guit_clean_l, guit_clean_r, guit_cl->frames() * guit_cl->channels());
	//MonoStereoConversion::extractChannel(guit_clean_lr, guit_clean_l, 1, guit_cl->channels(), guit_cl->frames() * guit_cl->channels());
	//MonoStereoConversion::extractChannel(guit_clean_lr, guit_clean_r, 2, guit_cl->channels(), guit_cl->frames() * guit_cl->channels());

	//CABINET:
	float* cab_mono = new float[cab_ir->frames() * cab_ir->channels()];
	//read cabinet data mono
	cab_ir->readf(cab_mono, cab_ir->frames());


	//HALL:
	float* hall_clear_lr = new float[hall_ir->frames() * hall_ir->channels()];
	float* hall_clear_l  = new float[hall_ir->frames()];
	float* hall_clear_r  = new float[hall_ir->frames()];
	//read hall data stereo
	hall_ir->readf(hall_clear_lr, hall_ir->frames());
	//extract left and right channel from stereo hall
	MonoStereoConversion::extractBothChannels(hall_clear_lr, hall_clear_l, hall_clear_r, hall_ir->frames() * hall_ir->channels());
	//MonoStereoConversion::extractChannel(hall_clear_lr, hall_clear_l, 1, hall_ir->channels(), hall_ir->frames() * hall_ir->channels());
	//MonoStereoConversion::extractChannel(hall_clear_lr, hall_clear_r, 1, hall_ir->channels(), hall_ir->frames() * hall_ir->channels());


	//convolve left and right channel separately for Cabinet:
	float *guit_cab_l, *guit_cab_r, *guit_cab_lr;
	int guit_cab_l_len, guit_cab_r_len;	
	float guit_cab_maxValue;

	guit_cab_l = conv(guit_clean_l, cab_mono, guit_cl->frames(), cab_ir->frames(), &guit_cab_l_len);
	guit_cab_r = conv(guit_clean_r, cab_mono, guit_cl->frames(), cab_ir->frames(), &guit_cab_r_len);

	guit_cab_lr = new float[guit_cab_l_len + guit_cab_r_len];

	//combine left and right channels to stereo
	MonoStereoConversion::combine2Channels(guit_cab_l, guit_cab_r, guit_cab_lr, guit_cab_l_len, &guit_cab_maxValue);
	// mozda ne najbolji nacin normalizacije! proveri da li klipuje fajl
	MonoStereoConversion::normalize(guit_cab_lr, guit_cab_l_len * 2, guit_cab_maxValue);

	cout << "No of frames writen to guitar_cab: " << guit_cab->writef(guit_cab_lr, guit_cab_l_len) << endl;


	//convolve left and right channel separately for Hall:
	float *guit_hall_l, *guit_hall_r, *guit_hall_lr;
	int guit_hall_l_len, guit_hall_r_len;
	float guit_hall_maxValue;

	guit_hall_l = conv(guit_clean_l, hall_clear_l, guit_cl->frames(), hall_ir->frames(), &guit_hall_l_len);
	guit_hall_r = conv(guit_clean_r, hall_clear_r, guit_cl->frames(), hall_ir->frames(), &guit_hall_r_len);

	guit_hall_lr = new float[guit_hall_l_len + guit_hall_r_len];

	//combine left and right channels to stereo
	MonoStereoConversion::combine2Channels(guit_hall_l, guit_hall_r, guit_hall_lr, guit_hall_l_len, &guit_hall_maxValue);
	//normalizacija...
	MonoStereoConversion::normalize(guit_hall_lr, guit_hall_l_len * 2, guit_hall_maxValue);

	cout << "No of frames writen to guitar_hall: " << guit_hall->writef(guit_hall_lr, guit_hall_l_len) << endl;



	delete guit_cl;
	delete cab_ir;
	delete guit_cab;

	delete guit_clean_l, guit_clean_r, guit_clean_lr;

	delete cab_mono;
	delete guit_cab_l, guit_cab_r, guit_cab_lr;

	delete hall_clear_l, hall_clear_r, hall_clear_lr;
	delete guit_hall_l, guit_hall_r, guit_hall_lr;

	//obrisi ovo... samo za testiranje
	int a;
	cin >> a;
	//

	return 0;
}