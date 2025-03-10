cmake_minimum_required(VERSION 3.10)
project(logoTracker)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIRS})

add_executable(logoTracker logoTracker.cpp sort.cpp sort.h)

target_link_libraries(logoTracker ${OpenCV_LIBS})