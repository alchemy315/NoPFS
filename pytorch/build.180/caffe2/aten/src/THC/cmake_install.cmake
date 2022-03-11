# Install script for directory: /home/zzp/code/NoPFS/pytorch/aten/src/THC

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/zzp/code/NoPFS/pytorch/torch")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/THC" TYPE FILE MESSAGE_NEVER FILES
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THC.h"
    "/home/zzp/code/NoPFS/pytorch/build/caffe2/aten/src/THC/THCGeneral.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCGeneral.hpp"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCSleep.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCStorage.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCStorageCopy.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCTensor.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCTensorCopy.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCTensorCopy.hpp"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCTensorMathReduce.cuh"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCAsmUtils.cuh"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCAtomics.cuh"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCScanUtils.cuh"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCAllocator.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCCachingHostAllocator.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCDeviceUtils.cuh"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCDeviceTensor.cuh"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCDeviceTensor-inl.cuh"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCDeviceTensorUtils.cuh"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCDeviceTensorUtils-inl.cuh"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCGenerateAllTypes.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCGenerateBFloat16Type.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCGenerateBoolType.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCGenerateByteType.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCGenerateCharType.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCGenerateShortType.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCGenerateIntType.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCGenerateLongType.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCGenerateHalfType.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCGenerateFloatType.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCGenerateFloatTypes.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCGenerateDoubleType.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCGenerateComplexFloatType.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCGenerateComplexTypes.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCGenerateComplexDoubleType.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCIntegerDivider.cuh"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCNumerics.cuh"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCThrustAllocator.cuh"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCTensor.hpp"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/THCStorage.hpp"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/THC/generic" TYPE FILE MESSAGE_NEVER FILES
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/generic/THCStorage.cpp"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/generic/THCStorage.cu"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/generic/THCStorage.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/generic/THCTensor.cpp"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/generic/THCTensor.cu"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/generic/THCTensor.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/generic/THCStorageCopy.cpp"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/generic/THCStorageCopy.cu"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/generic/THCStorageCopy.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/generic/THCTensorCopy.cu"
    "/home/zzp/code/NoPFS/pytorch/aten/src/THC/generic/THCTensorCopy.h"
    )
endif()

