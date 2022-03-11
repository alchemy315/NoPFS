# Install script for directory: /home/zzp/code/NoPFS/pytorch/aten/src/TH

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/TH" TYPE FILE MESSAGE_NEVER FILES
    "/home/zzp/code/NoPFS/pytorch/aten/src/TH/TH.h"
    "/home/zzp/code/NoPFS/pytorch/build/caffe2/aten/src/TH/THGeneral.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/TH/THGenerateAllTypes.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/TH/THGenerateBFloat16Type.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/TH/THGenerateBoolType.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/TH/THGenerateDoubleType.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/TH/THGenerateFloatType.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/TH/THGenerateHalfType.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/TH/THGenerateComplexFloatType.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/TH/THGenerateComplexDoubleType.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/TH/THGenerateLongType.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/TH/THGenerateIntType.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/TH/THGenerateShortType.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/TH/THGenerateCharType.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/TH/THGenerateByteType.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/TH/THGenerateFloatTypes.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/TH/THGenerateComplexTypes.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/TH/THGenerateIntTypes.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/TH/THGenerateQUInt8Type.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/TH/THGenerateQUInt4x2Type.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/TH/THGenerateQInt8Type.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/TH/THGenerateQInt32Type.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/TH/THGenerateQTypes.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/TH/THStorage.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/TH/THStorageFunctions.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/TH/THTensor.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/TH/THHalf.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/TH/THTensor.hpp"
    "/home/zzp/code/NoPFS/pytorch/aten/src/TH/THStorageFunctions.hpp"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/TH/generic" TYPE FILE MESSAGE_NEVER FILES
    "/home/zzp/code/NoPFS/pytorch/aten/src/TH/generic/THStorage.cpp"
    "/home/zzp/code/NoPFS/pytorch/aten/src/TH/generic/THStorage.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/TH/generic/THStorageCopy.cpp"
    "/home/zzp/code/NoPFS/pytorch/aten/src/TH/generic/THStorageCopy.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/TH/generic/THTensor.cpp"
    "/home/zzp/code/NoPFS/pytorch/aten/src/TH/generic/THTensor.h"
    "/home/zzp/code/NoPFS/pytorch/aten/src/TH/generic/THTensor.hpp"
    )
endif()

