#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "gloo" for configuration "Debug"
set_property(TARGET gloo APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(gloo PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "/home/zzp/code/NoPFS/pytorch/torch/lib/libgloo.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS gloo )
list(APPEND _IMPORT_CHECK_FILES_FOR_gloo "/home/zzp/code/NoPFS/pytorch/torch/lib/libgloo.a" )

# Import target "gloo_cuda" for configuration "Debug"
set_property(TARGET gloo_cuda APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(gloo_cuda PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "/home/zzp/code/NoPFS/pytorch/torch/lib/libgloo_cuda.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS gloo_cuda )
list(APPEND _IMPORT_CHECK_FILES_FOR_gloo_cuda "/home/zzp/code/NoPFS/pytorch/torch/lib/libgloo_cuda.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
