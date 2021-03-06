# Install script for directory: /home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/build/lib/libasmjit.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/asmjit/asmjit-config.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/asmjit/asmjit-config.cmake"
         "/home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm/asmjit/CMakeFiles/Export/lib/cmake/asmjit/asmjit-config.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/asmjit/asmjit-config-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/asmjit/asmjit-config.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/asmjit" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm/asmjit/CMakeFiles/Export/lib/cmake/asmjit/asmjit-config.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/asmjit" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm/asmjit/CMakeFiles/Export/lib/cmake/asmjit/asmjit-config-release.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/asmjit.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/asmjit-scope-begin.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/asmjit-scope-end.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core/api-config.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core/archtraits.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core/archcommons.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core/assembler.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core/builder.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core/codebuffer.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core/codeholder.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core/compiler.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core/compilerdefs.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core/constpool.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core/cpuinfo.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core/datatypes.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core/emitter.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core/environment.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core/errorhandler.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core/features.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core/formatter.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core/func.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core/globals.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core/inst.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core/jitallocator.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core/jitruntime.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core/logger.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core/operand.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core/osutils.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core/string.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core/support.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core/target.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core/type.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core/virtmem.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core/zone.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core/zonehash.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core/zonelist.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core/zonestack.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core/zonestring.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core/zonetree.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/core" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/core/zonevector.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/x86.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/x86" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/x86/x86assembler.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/x86" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/x86/x86builder.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/x86" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/x86/x86compiler.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/x86" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/x86/x86emitter.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/x86" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/x86/x86features.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/x86" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/x86/x86globals.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/x86" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/x86/x86instdb.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/asmjit/x86" TYPE FILE MESSAGE_NEVER FILES "/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src/asmjit/x86/x86operand.h")
endif()

