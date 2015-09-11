#################################################
###                                           ###
###   COMPILATION AND PROGRAM RUN ARGUMENTS   ###
###                                           ###
#################################################

I: COMPILATION USING VISUAL STUDIO ULTIMATE 2013
   
   1. C/C++\General      (Additional Include Directories): AdditionalLibraries\include
   2. C/C++\Preprocessor (Preprocessor Definitions):       append _CRT_SECURE_NO_WARNINGS
   3. Linker\General     (Additional Library Directories): AdditionalLibraries\lib
   4. Linker\Input       (Additional Dependencies):        append libsndfile-1.lib;libfftw3f-3.lib;
   5. C/C++\Language     (OpenMP Support):                 enable OpenMP support

II: PROGRAM ARGUMENTS
   
   Program has two arguments:
      1: input WAV filepath
      2: impulse response WAV filepath

   Program will create new file named "input_filename_out.wav"