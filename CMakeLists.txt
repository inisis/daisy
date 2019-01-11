cmake_minimum_required(VERSION 2.8)
project(TensorRT-SSD)

SET(CMAKE_BUILD_TYPE "Debug")

set(CMAKE_CXX_FLAGS_DEBUG " $ENV{CXXFLAGS} -std=c++11 -DDEBUG -DUSE_CUDA -DDLIB_NO_GUI_SUPPORT -g -w")
set(CMAKE_CXX_FLAGS_RELEASE " $ENV{CXXFLAGS} -std=c++11 -DNDEBUG -DUSE_CUDA -DDLIB_NO_GUI_SUPPORT -O3 -w")

include_directories(include)

#SET(TENSORRT_DIR /home/desmond/Tensorrt5/TensorRT-5.0.2.6)
#FIND_PACKAGE(TensorRT REQUIRED)
set(TensorRT_PATH "/home/desmond/TensorRT-5.0.2.6")
include_directories(${TensorRT_PATH}/include)

SET(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-10.0)
find_package(CUDA REQUIRED)
# /usr/share/cmake/Modules/FindCUDA.cmake
# INCLUDE(FindCUDA)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the CUDAConfig.cmake file.
# Print some message showing some of them
message(STATUS "CUDA library status:")
message(STATUS "    toolkit root dir: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "    version: ${CUDA_VERSION}")
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

#Bring the headers, *.h into the project
include_directories(include ${CUDA_INCLUDE_DIRS})

link_directories(${LIBSDIR} "${TensorRT_PATH}/lib" ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

add_subdirectory(src)
add_subdirectory(test)