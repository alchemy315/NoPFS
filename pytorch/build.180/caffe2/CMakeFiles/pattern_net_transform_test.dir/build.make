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
include caffe2/CMakeFiles/pattern_net_transform_test.dir/depend.make

# Include the progress variables for this target.
include caffe2/CMakeFiles/pattern_net_transform_test.dir/progress.make

# Include the compile flags for this target's objects.
include caffe2/CMakeFiles/pattern_net_transform_test.dir/flags.make

caffe2/CMakeFiles/pattern_net_transform_test.dir/transforms/pattern_net_transform_test.cc.o: caffe2/CMakeFiles/pattern_net_transform_test.dir/flags.make
caffe2/CMakeFiles/pattern_net_transform_test.dir/transforms/pattern_net_transform_test.cc.o: ../caffe2/transforms/pattern_net_transform_test.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object caffe2/CMakeFiles/pattern_net_transform_test.dir/transforms/pattern_net_transform_test.cc.o"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2 && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pattern_net_transform_test.dir/transforms/pattern_net_transform_test.cc.o -c /home/zzp/code/NoPFS/pytorch/caffe2/transforms/pattern_net_transform_test.cc

caffe2/CMakeFiles/pattern_net_transform_test.dir/transforms/pattern_net_transform_test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pattern_net_transform_test.dir/transforms/pattern_net_transform_test.cc.i"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zzp/code/NoPFS/pytorch/caffe2/transforms/pattern_net_transform_test.cc > CMakeFiles/pattern_net_transform_test.dir/transforms/pattern_net_transform_test.cc.i

caffe2/CMakeFiles/pattern_net_transform_test.dir/transforms/pattern_net_transform_test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pattern_net_transform_test.dir/transforms/pattern_net_transform_test.cc.s"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zzp/code/NoPFS/pytorch/caffe2/transforms/pattern_net_transform_test.cc -o CMakeFiles/pattern_net_transform_test.dir/transforms/pattern_net_transform_test.cc.s

caffe2/CMakeFiles/pattern_net_transform_test.dir/transforms/pattern_net_transform_test.cc.o.requires:

.PHONY : caffe2/CMakeFiles/pattern_net_transform_test.dir/transforms/pattern_net_transform_test.cc.o.requires

caffe2/CMakeFiles/pattern_net_transform_test.dir/transforms/pattern_net_transform_test.cc.o.provides: caffe2/CMakeFiles/pattern_net_transform_test.dir/transforms/pattern_net_transform_test.cc.o.requires
	$(MAKE) -f caffe2/CMakeFiles/pattern_net_transform_test.dir/build.make caffe2/CMakeFiles/pattern_net_transform_test.dir/transforms/pattern_net_transform_test.cc.o.provides.build
.PHONY : caffe2/CMakeFiles/pattern_net_transform_test.dir/transforms/pattern_net_transform_test.cc.o.provides

caffe2/CMakeFiles/pattern_net_transform_test.dir/transforms/pattern_net_transform_test.cc.o.provides.build: caffe2/CMakeFiles/pattern_net_transform_test.dir/transforms/pattern_net_transform_test.cc.o


# Object files for target pattern_net_transform_test
pattern_net_transform_test_OBJECTS = \
"CMakeFiles/pattern_net_transform_test.dir/transforms/pattern_net_transform_test.cc.o"

# External object files for target pattern_net_transform_test
pattern_net_transform_test_EXTERNAL_OBJECTS =

bin/pattern_net_transform_test: caffe2/CMakeFiles/pattern_net_transform_test.dir/transforms/pattern_net_transform_test.cc.o
bin/pattern_net_transform_test: caffe2/CMakeFiles/pattern_net_transform_test.dir/build.make
bin/pattern_net_transform_test: lib/libgtest_main.a
bin/pattern_net_transform_test: /usr/lib/gcc/x86_64-linux-gnu/7/libgomp.so
bin/pattern_net_transform_test: /usr/lib/x86_64-linux-gnu/libpthread.so
bin/pattern_net_transform_test: lib/libprotobuf.a
bin/pattern_net_transform_test: lib/libdnnl.a
bin/pattern_net_transform_test: lib/libc10_cuda.so
bin/pattern_net_transform_test: lib/libc10.so
bin/pattern_net_transform_test: /usr/local/cuda/lib64/libcudart.so
bin/pattern_net_transform_test: /usr/local/cuda/lib64/libnvToolsExt.so
bin/pattern_net_transform_test: /usr/local/cuda/lib64/libcufft.so
bin/pattern_net_transform_test: /usr/local/cuda/lib64/libcurand.so
bin/pattern_net_transform_test: /usr/local/cuda/lib64/libcublas.so
bin/pattern_net_transform_test: /usr/lib/x86_64-linux-gnu/libcudnn.so
bin/pattern_net_transform_test: lib/libgtest.a
bin/pattern_net_transform_test: caffe2/CMakeFiles/pattern_net_transform_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/pattern_net_transform_test"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2 && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pattern_net_transform_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
caffe2/CMakeFiles/pattern_net_transform_test.dir/build: bin/pattern_net_transform_test

.PHONY : caffe2/CMakeFiles/pattern_net_transform_test.dir/build

caffe2/CMakeFiles/pattern_net_transform_test.dir/requires: caffe2/CMakeFiles/pattern_net_transform_test.dir/transforms/pattern_net_transform_test.cc.o.requires

.PHONY : caffe2/CMakeFiles/pattern_net_transform_test.dir/requires

caffe2/CMakeFiles/pattern_net_transform_test.dir/clean:
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2 && $(CMAKE_COMMAND) -P CMakeFiles/pattern_net_transform_test.dir/cmake_clean.cmake
.PHONY : caffe2/CMakeFiles/pattern_net_transform_test.dir/clean

caffe2/CMakeFiles/pattern_net_transform_test.dir/depend:
	cd /home/zzp/code/NoPFS/pytorch/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zzp/code/NoPFS/pytorch /home/zzp/code/NoPFS/pytorch/caffe2 /home/zzp/code/NoPFS/pytorch/build /home/zzp/code/NoPFS/pytorch/build/caffe2 /home/zzp/code/NoPFS/pytorch/build/caffe2/CMakeFiles/pattern_net_transform_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : caffe2/CMakeFiles/pattern_net_transform_test.dir/depend

