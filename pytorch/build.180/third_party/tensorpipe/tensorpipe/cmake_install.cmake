# Install script for directory: /home/zzp/code/NoPFS/pytorch/third_party/tensorpipe/tensorpipe

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/build/lib/libtensorpipe_uv.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/build/lib/libtensorpipe.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tensorpipe" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/tensorpipe/tensorpipe/tensorpipe.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tensorpipe/channel" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/tensorpipe/tensorpipe/channel/context.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tensorpipe/channel" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/tensorpipe/tensorpipe/channel/error.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tensorpipe/common" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/tensorpipe/tensorpipe/common/buffer.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tensorpipe/common" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/tensorpipe/tensorpipe/common/cpu_buffer.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tensorpipe/common" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/tensorpipe/tensorpipe/common/device.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tensorpipe/common" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/tensorpipe/tensorpipe/common/error.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tensorpipe/common" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/tensorpipe/tensorpipe/common/optional.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tensorpipe/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/tensorpipe/tensorpipe/core/context.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tensorpipe/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/tensorpipe/tensorpipe/core/error.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tensorpipe/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/tensorpipe/tensorpipe/core/listener.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tensorpipe/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/tensorpipe/tensorpipe/core/message.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tensorpipe/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/tensorpipe/tensorpipe/core/pipe.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tensorpipe/transport" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/tensorpipe/tensorpipe/transport/context.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tensorpipe/transport" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/tensorpipe/tensorpipe/transport/error.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tensorpipe/channel/basic" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/tensorpipe/tensorpipe/channel/basic/factory.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tensorpipe/channel/xth" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/tensorpipe/tensorpipe/channel/xth/factory.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tensorpipe/channel/cma" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/tensorpipe/tensorpipe/channel/cma/factory.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tensorpipe/channel/mpt" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/tensorpipe/tensorpipe/channel/mpt/factory.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tensorpipe/transport/uv" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/tensorpipe/tensorpipe/transport/uv/error.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tensorpipe/transport/uv" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/tensorpipe/tensorpipe/transport/uv/factory.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tensorpipe/transport/uv" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/tensorpipe/tensorpipe/transport/uv/utility.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tensorpipe/transport/shm" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/tensorpipe/tensorpipe/transport/shm/factory.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tensorpipe/transport/ibv" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/tensorpipe/tensorpipe/transport/ibv/error.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tensorpipe/transport/ibv" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/tensorpipe/tensorpipe/transport/ibv/factory.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tensorpipe/transport/ibv" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/tensorpipe/tensorpipe/transport/ibv/utility.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tensorpipe" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/build/third_party/tensorpipe/tensorpipe/config.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/build/lib/libtensorpipe_cuda.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tensorpipe" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/tensorpipe/tensorpipe/tensorpipe_cuda.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tensorpipe/common" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/tensorpipe/tensorpipe/common/cuda_buffer.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tensorpipe/channel/cuda_xth" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/tensorpipe/tensorpipe/channel/cuda_xth/factory.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tensorpipe/channel/cuda_basic" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/tensorpipe/tensorpipe/channel/cuda_basic/factory.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tensorpipe/channel/cuda_ipc" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/tensorpipe/tensorpipe/channel/cuda_ipc/factory.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tensorpipe/channel/cuda_gdr" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/tensorpipe/tensorpipe/channel/cuda_gdr/error.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tensorpipe/channel/cuda_gdr" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/tensorpipe/tensorpipe/channel/cuda_gdr/factory.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tensorpipe" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/build/third_party/tensorpipe/tensorpipe/config_cuda.h")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.

endif()

