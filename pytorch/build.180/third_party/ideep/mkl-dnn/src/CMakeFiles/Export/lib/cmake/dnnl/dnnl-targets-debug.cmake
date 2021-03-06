#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "DNNL::dnnl" for configuration "Debug"
set_property(TARGET DNNL::dnnl APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(DNNL::dnnl PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C;CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libdnnl.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS DNNL::dnnl )
list(APPEND _IMPORT_CHECK_FILES_FOR_DNNL::dnnl "${_IMPORT_PREFIX}/lib/libdnnl.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
