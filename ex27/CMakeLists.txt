# Modify the foreach loop to link additional dependencies for sift
foreach(SOURCE ${SOURCES})
  get_filename_component(EXECUTABLE_NAME ${SOURCE} NAME_WE)
  add_executable(${EXECUTABLE_NAME} ${SOURCE} ${HEADERS})
  
  # Link Eigen only for files that need it
  if(${EXECUTABLE_NAME} MATCHES "optical_flow|homography|ransac|hog|feature_matching|sift")
    target_link_libraries(${EXECUTABLE_NAME} Eigen3::Eigen)
  endif()
  
  # Add specific dependencies for sift
  if(${EXECUTABLE_NAME} STREQUAL "sift")
    # Assuming these functions are in harris.cpp and hog.cpp
    target_sources(${EXECUTABLE_NAME} PRIVATE 
      ${CMAKE_CURRENT_SOURCE_DIR}/src/harris.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/src/hog.cpp
    )
  endif()
  
  # All executables use stb_image (via image_utils.h)
  target_include_directories(${EXECUTABLE_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
endforeach()