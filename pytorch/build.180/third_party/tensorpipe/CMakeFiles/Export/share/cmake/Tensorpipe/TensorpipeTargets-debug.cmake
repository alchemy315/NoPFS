#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "tensorpipe_uv" for configuration "Debug"
set_property(TARGET tensorpipe_uv APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(tensorpipe_uv PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libtensorpipe_uv.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS tensorpipe_uv )
list(APPEND _IMPORT_CHECK_FILES_FOR_tensorpipe_uv "${_IMPORT_PREFIX}/lib/libtensorpipe_uv.a" )

# Import target "tensorpipe" for configuration "Debug"
set_property(TARGET tensorpipe APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(tensorpipe PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libtensorpipe.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS tensorpipe )
list(APPEND _IMPORT_CHECK_FILES_FOR_tensorpipe "${_IMPORT_PREFIX}/lib/libtensorpipe.a" )

# Import target "tensorpipe_cuda" for configuration "Debug"
set_property(TARGET tensorpipe_cuda APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(tensorpipe_cuda PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libtensorpipe_cuda.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS tensorpipe_cuda )
list(APPEND _IMPORT_CHECK_FILES_FOR_tensorpipe_cuda "${_IMPORT_PREFIX}/lib/libtensorpipe_cuda.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
