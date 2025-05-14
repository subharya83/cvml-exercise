#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "GDAL::GDAL" for configuration "Release"
set_property(TARGET GDAL::GDAL APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(GDAL::GDAL PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libgdal.so.36.3.10.3"
  IMPORTED_SONAME_RELEASE "libgdal.so.36"
  )

list(APPEND _cmake_import_check_targets GDAL::GDAL )
list(APPEND _cmake_import_check_files_for_GDAL::GDAL "${_IMPORT_PREFIX}/lib/libgdal.so.36.3.10.3" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
