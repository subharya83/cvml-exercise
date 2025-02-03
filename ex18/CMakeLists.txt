cmake_minimum_required(VERSION 3.10)
project(VideoIntegral CUDA CXX)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)

include_directories(/usr/include/hdf5/serial)

cuda_add_executable(video_integral
    video_processor.cpp
    cuda_kernels.cu
)

target_link_libraries(video_integral
    ${OpenCV_LIBS}
    ${HDF5_LIBRARIES}
)