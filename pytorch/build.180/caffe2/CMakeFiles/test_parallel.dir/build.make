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
include caffe2/CMakeFiles/test_parallel.dir/depend.make

# Include the progress variables for this target.
include caffe2/CMakeFiles/test_parallel.dir/progress.make

# Include the compile flags for this target's objects.
include caffe2/CMakeFiles/test_parallel.dir/flags.make

caffe2/CMakeFiles/test_parallel.dir/__/aten/src/ATen/test/test_parallel.cpp.o: caffe2/CMakeFiles/test_parallel.dir/flags.make
caffe2/CMakeFiles/test_parallel.dir/__/aten/src/ATen/test/test_parallel.cpp.o: ../aten/src/ATen/test/test_parallel.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object caffe2/CMakeFiles/test_parallel.dir/__/aten/src/ATen/test/test_parallel.cpp.o"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2 && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_parallel.dir/__/aten/src/ATen/test/test_parallel.cpp.o -c /home/zzp/code/NoPFS/pytorch/aten/src/ATen/test/test_parallel.cpp

caffe2/CMakeFiles/test_parallel.dir/__/aten/src/ATen/test/test_parallel.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_parallel.dir/__/aten/src/ATen/test/test_parallel.cpp.i"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zzp/code/NoPFS/pytorch/aten/src/ATen/test/test_parallel.cpp > CMakeFiles/test_parallel.dir/__/aten/src/ATen/test/test_parallel.cpp.i

caffe2/CMakeFiles/test_parallel.dir/__/aten/src/ATen/test/test_parallel.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_parallel.dir/__/aten/src/ATen/test/test_parallel.cpp.s"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zzp/code/NoPFS/pytorch/aten/src/ATen/test/test_parallel.cpp -o CMakeFiles/test_parallel.dir/__/aten/src/ATen/test/test_parallel.cpp.s

caffe2/CMakeFiles/test_parallel.dir/__/aten/src/ATen/test/test_parallel.cpp.o.requires:

.PHONY : caffe2/CMakeFiles/test_parallel.dir/__/aten/src/ATen/test/test_parallel.cpp.o.requires

caffe2/CMakeFiles/test_parallel.dir/__/aten/src/ATen/test/test_parallel.cpp.o.provides: caffe2/CMakeFiles/test_parallel.dir/__/aten/src/ATen/test/test_parallel.cpp.o.requires
	$(MAKE) -f caffe2/CMakeFiles/test_parallel.dir/build.make caffe2/CMakeFiles/test_parallel.dir/__/aten/src/ATen/test/test_parallel.cpp.o.provides.build
.PHONY : caffe2/CMakeFiles/test_parallel.dir/__/aten/src/ATen/test/test_parallel.cpp.o.provides

caffe2/CMakeFiles/test_parallel.dir/__/aten/src/ATen/test/test_parallel.cpp.o.provides.build: caffe2/CMakeFiles/test_parallel.dir/__/aten/src/ATen/test/test_parallel.cpp.o


# Object files for target test_parallel
test_parallel_OBJECTS = \
"CMakeFiles/test_parallel.dir/__/aten/src/ATen/test/test_parallel.cpp.o"

# External object files for target test_parallel
test_parallel_EXTERNAL_OBJECTS =

bin/test_parallel: caffe2/CMakeFiles/test_parallel.dir/__/aten/src/ATen/test/test_parallel.cpp.o
bin/test_parallel: caffe2/CMakeFiles/test_parallel.dir/build.make
bin/test_parallel: lib/libgtest_main.a
bin/test_parallel: /usr/lib/gcc/x86_64-linux-gnu/7/libgomp.so
bin/test_parallel: /usr/lib/x86_64-linux-gnu/libpthread.so
bin/test_parallel: lib/libprotobuf.a
bin/test_parallel: lib/libdnnl.a
bin/test_parallel: lib/libc10_cuda.so
bin/test_parallel: lib/libc10.so
bin/test_parallel: /usr/local/cuda/lib64/libcudart.so
bin/test_parallel: /usr/local/cuda/lib64/libnvToolsExt.so
bin/test_parallel: /usr/local/cuda/lib64/libcufft.so
bin/test_parallel: /usr/local/cuda/lib64/libcurand.so
bin/test_parallel: /usr/local/cuda/lib64/libcublas.so
bin/test_parallel: /usr/lib/x86_64-linux-gnu/libcudnn.so
bin/test_parallel: lib/libgtest.a
bin/test_parallel: caffe2/CMakeFiles/test_parallel.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/test_parallel"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2 && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_parallel.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
caffe2/CMakeFiles/test_parallel.dir/build: bin/test_parallel

.PHONY : caffe2/CMakeFiles/test_parallel.dir/build

caffe2/CMakeFiles/test_parallel.dir/requires: caffe2/CMakeFiles/test_parallel.dir/__/aten/src/ATen/test/test_parallel.cpp.o.requires

.PHONY : caffe2/CMakeFiles/test_parallel.dir/requires

caffe2/CMakeFiles/test_parallel.dir/clean:
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2 && $(CMAKE_COMMAND) -P CMakeFiles/test_parallel.dir/cmake_clean.cmake
.PHONY : caffe2/CMakeFiles/test_parallel.dir/clean

caffe2/CMakeFiles/test_parallel.dir/depend:
	cd /home/zzp/code/NoPFS/pytorch/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zzp/code/NoPFS/pytorch /home/zzp/code/NoPFS/pytorch/caffe2 /home/zzp/code/NoPFS/pytorch/build /home/zzp/code/NoPFS/pytorch/build/caffe2 /home/zzp/code/NoPFS/pytorch/build/caffe2/CMakeFiles/test_parallel.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : caffe2/CMakeFiles/test_parallel.dir/depend
