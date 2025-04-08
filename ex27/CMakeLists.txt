cmake_minimum_required(VERSION 3.10)
project(cv_minimal)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_SIFT "Enable SIFT-specific enhancements" ON)

if(USE_SIFT)
  add_definitions(-DUSE_SIFT)
  message(STATUS "Building with SIFT enhancements")
endif()

# Enable colored compiler output (for readability)
if(NOT WIN32)
  add_compile_options(-fdiagnostics-color=always)
endif()

# Find Eigen (header-only)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

# Add stb_image (header-only)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Glob all headers and sources
file(GLOB HEADERS "include/*.h")
file(GLOB SOURCES "src/*.cpp")

# Create executables for each source file
foreach(SOURCE ${SOURCES})
  get_filename_component(EXECUTABLE_NAME ${SOURCE} NAME_WE)
  add_executable(${EXECUTABLE_NAME} ${SOURCE} ${HEADERS})
  
  # Link Eigen only for files that need it
  if(${EXECUTABLE_NAME} MATCHES "optical_flow|homography|ransac|hog|feature_matching")
    target_link_libraries(${EXECUTABLE_NAME} Eigen3::Eigen)
  endif()
  
  # All executables use stb_image (via image_utils.h)
  target_include_directories(${EXECUTABLE_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
endforeach()

# Optional: Install stb_image if not present
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/stb_image.h")
  message(STATUS "Downloading stb_image.h...")
  file(DOWNLOAD 
    https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/stb_image.h
  )
endif()
