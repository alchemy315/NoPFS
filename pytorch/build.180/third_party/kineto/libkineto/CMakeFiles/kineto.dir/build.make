# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zzp/code/NoPFS/pytorch

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zzp/code/NoPFS/pytorch/build

# Include any dependencies generated for this target.
include third_party/kineto/libkineto/CMakeFiles/kineto.dir/depend.make

# Include the progress variables for this target.
include third_party/kineto/libkineto/CMakeFiles/kineto.dir/progress.make

# Include the compile flags for this target's objects.
include third_party/kineto/libkineto/CMakeFiles/kineto.dir/flags.make

# Object files for target kineto
kineto_OBJECTS =

# External object files for target kineto
kineto_EXTERNAL_OBJECTS = \
"/home/zzp/code/NoPFS/pytorch/build/third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/AbstractConfig.cpp.o" \
"/home/zzp/code/NoPFS/pytorch/build/third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/ActivityProfiler.cpp.o" \
"/home/zzp/code/NoPFS/pytorch/build/third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/ActivityProfilerController.cpp.o" \
"/home/zzp/code/NoPFS/pytorch/build/third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/ActivityProfilerProxy.cpp.o" \
"/home/zzp/code/NoPFS/pytorch/build/third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/ActivityType.cpp.o" \
"/home/zzp/code/NoPFS/pytorch/build/third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/Config.cpp.o" \
"/home/zzp/code/NoPFS/pytorch/build/third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/ConfigLoader.cpp.o" \
"/home/zzp/code/NoPFS/pytorch/build/third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/CudaDeviceProperties.cpp.o" \
"/home/zzp/code/NoPFS/pytorch/build/third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/CuptiActivityInterface.cpp.o" \
"/home/zzp/code/NoPFS/pytorch/build/third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/CuptiActivityPlatform.cpp.o" \
"/home/zzp/code/NoPFS/pytorch/build/third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/CuptiEventInterface.cpp.o" \
"/home/zzp/code/NoPFS/pytorch/build/third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/CuptiMetricInterface.cpp.o" \
"/home/zzp/code/NoPFS/pytorch/build/third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/Demangle.cpp.o" \
"/home/zzp/code/NoPFS/pytorch/build/third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/EventProfiler.cpp.o" \
"/home/zzp/code/NoPFS/pytorch/build/third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/EventProfilerController.cpp.o" \
"/home/zzp/code/NoPFS/pytorch/build/third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/GenericTraceActivity.cpp.o" \
"/home/zzp/code/NoPFS/pytorch/build/third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/Logger.cpp.o" \
"/home/zzp/code/NoPFS/pytorch/build/third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/WeakSymbols.cpp.o" \
"/home/zzp/code/NoPFS/pytorch/build/third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/cupti_strings.cpp.o" \
"/home/zzp/code/NoPFS/pytorch/build/third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/init.cpp.o" \
"/home/zzp/code/NoPFS/pytorch/build/third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/output_csv.cpp.o" \
"/home/zzp/code/NoPFS/pytorch/build/third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/output_json.cpp.o" \
"/home/zzp/code/NoPFS/pytorch/build/third_party/kineto/libkineto/CMakeFiles/kineto_api.dir/src/ThreadUtil.cpp.o" \
"/home/zzp/code/NoPFS/pytorch/build/third_party/kineto/libkineto/CMakeFiles/kineto_api.dir/src/libkineto_api.cpp.o"

lib/libkineto.a: third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/AbstractConfig.cpp.o
lib/libkineto.a: third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/ActivityProfiler.cpp.o
lib/libkineto.a: third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/ActivityProfilerController.cpp.o
lib/libkineto.a: third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/ActivityProfilerProxy.cpp.o
lib/libkineto.a: third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/ActivityType.cpp.o
lib/libkineto.a: third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/Config.cpp.o
lib/libkineto.a: third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/ConfigLoader.cpp.o
lib/libkineto.a: third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/CudaDeviceProperties.cpp.o
lib/libkineto.a: third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/CuptiActivityInterface.cpp.o
lib/libkineto.a: third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/CuptiActivityPlatform.cpp.o
lib/libkineto.a: third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/CuptiEventInterface.cpp.o
lib/libkineto.a: third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/CuptiMetricInterface.cpp.o
lib/libkineto.a: third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/Demangle.cpp.o
lib/libkineto.a: third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/EventProfiler.cpp.o
lib/libkineto.a: third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/EventProfilerController.cpp.o
lib/libkineto.a: third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/GenericTraceActivity.cpp.o
lib/libkineto.a: third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/Logger.cpp.o
lib/libkineto.a: third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/WeakSymbols.cpp.o
lib/libkineto.a: third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/cupti_strings.cpp.o
lib/libkineto.a: third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/init.cpp.o
lib/libkineto.a: third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/output_csv.cpp.o
lib/libkineto.a: third_party/kineto/libkineto/CMakeFiles/kineto_base.dir/src/output_json.cpp.o
lib/libkineto.a: third_party/kineto/libkineto/CMakeFiles/kineto_api.dir/src/ThreadUtil.cpp.o
lib/libkineto.a: third_party/kineto/libkineto/CMakeFiles/kineto_api.dir/src/libkineto_api.cpp.o
lib/libkineto.a: third_party/kineto/libkineto/CMakeFiles/kineto.dir/build.make
lib/libkineto.a: third_party/kineto/libkineto/CMakeFiles/kineto.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Linking CXX static library ../../../lib/libkineto.a"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/kineto/libkineto && $(CMAKE_COMMAND) -P CMakeFiles/kineto.dir/cmake_clean_target.cmake
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/kineto/libkineto && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/kineto.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
third_party/kineto/libkineto/CMakeFiles/kineto.dir/build: lib/libkineto.a

.PHONY : third_party/kineto/libkineto/CMakeFiles/kineto.dir/build

third_party/kineto/libkineto/CMakeFiles/kineto.dir/requires:

.PHONY : third_party/kineto/libkineto/CMakeFiles/kineto.dir/requires

third_party/kineto/libkineto/CMakeFiles/kineto.dir/clean:
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/kineto/libkineto && $(CMAKE_COMMAND) -P CMakeFiles/kineto.dir/cmake_clean.cmake
.PHONY : third_party/kineto/libkineto/CMakeFiles/kineto.dir/clean

third_party/kineto/libkineto/CMakeFiles/kineto.dir/depend:
	cd /home/zzp/code/NoPFS/pytorch/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zzp/code/NoPFS/pytorch /home/zzp/code/NoPFS/pytorch/third_party/kineto/libkineto /home/zzp/code/NoPFS/pytorch/build /home/zzp/code/NoPFS/pytorch/build/third_party/kineto/libkineto /home/zzp/code/NoPFS/pytorch/build/third_party/kineto/libkineto/CMakeFiles/kineto.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : third_party/kineto/libkineto/CMakeFiles/kineto.dir/depend

