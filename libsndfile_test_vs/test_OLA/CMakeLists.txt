cmake_minimum_required(VERSION 3.16.3)

project(reverberate)

find_package(OpenMP REQUIRED)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(SndFile REQUIRED)

add_executable(
  ${PROJECT_NAME}
  src/main.cpp
  src/MonoStereoConversion.cpp
  src/ReverbEffect.cpp
  src/Timer.cpp
)

set_target_properties(
  ${PROJECT_NAME} PROPERTIES CXX_STANDARD 17 CMAKE_CXX_STANDARD_REQUIRED ON
)

target_link_libraries(
  ${PROJECT_NAME}
  PRIVATE fftw3
          fftw3f
          OpenMP::OpenMP_CXX
          ${LIBSNDFILE_LIBRARIES}
)
