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
include caffe2/CMakeFiles/TopoSortTest.dir/depend.make

# Include the progress variables for this target.
include caffe2/CMakeFiles/TopoSortTest.dir/progress.make

# Include the compile flags for this target's objects.
include caffe2/CMakeFiles/TopoSortTest.dir/flags.make

caffe2/CMakeFiles/TopoSortTest.dir/core/nomnigraph/tests/TopoSortTest.cc.o: caffe2/CMakeFiles/TopoSortTest.dir/flags.make
caffe2/CMakeFiles/TopoSortTest.dir/core/nomnigraph/tests/TopoSortTest.cc.o: ../caffe2/core/nomnigraph/tests/TopoSortTest.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object caffe2/CMakeFiles/TopoSortTest.dir/core/nomnigraph/tests/TopoSortTest.cc.o"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2 && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/TopoSortTest.dir/core/nomnigraph/tests/TopoSortTest.cc.o -c /home/zzp/code/NoPFS/pytorch/caffe2/core/nomnigraph/tests/TopoSortTest.cc

caffe2/CMakeFiles/TopoSortTest.dir/core/nomnigraph/tests/TopoSortTest.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TopoSortTest.dir/core/nomnigraph/tests/TopoSortTest.cc.i"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zzp/code/NoPFS/pytorch/caffe2/core/nomnigraph/tests/TopoSortTest.cc > CMakeFiles/TopoSortTest.dir/core/nomnigraph/tests/TopoSortTest.cc.i

caffe2/CMakeFiles/TopoSortTest.dir/core/nomnigraph/tests/TopoSortTest.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TopoSortTest.dir/core/nomnigraph/tests/TopoSortTest.cc.s"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zzp/code/NoPFS/pytorch/caffe2/core/nomnigraph/tests/TopoSortTest.cc -o CMakeFiles/TopoSortTest.dir/core/nomnigraph/tests/TopoSortTest.cc.s

caffe2/CMakeFiles/TopoSortTest.dir/core/nomnigraph/tests/TopoSortTest.cc.o.requires:

.PHONY : caffe2/CMakeFiles/TopoSortTest.dir/core/nomnigraph/tests/TopoSortTest.cc.o.requires

caffe2/CMakeFiles/TopoSortTest.dir/core/nomnigraph/tests/TopoSortTest.cc.o.provides: caffe2/CMakeFiles/TopoSortTest.dir/core/nomnigraph/tests/TopoSortTest.cc.o.requires
	$(MAKE) -f caffe2/CMakeFiles/TopoSortTest.dir/build.make caffe2/CMakeFiles/TopoSortTest.dir/core/nomnigraph/tests/TopoSortTest.cc.o.provides.build
.PHONY : caffe2/CMakeFiles/TopoSortTest.dir/core/nomnigraph/tests/TopoSortTest.cc.o.provides

caffe2/CMakeFiles/TopoSortTest.dir/core/nomnigraph/tests/TopoSortTest.cc.o.provides.build: caffe2/CMakeFiles/TopoSortTest.dir/core/nomnigraph/tests/TopoSortTest.cc.o


# Object files for target TopoSortTest
TopoSortTest_OBJECTS = \
"CMakeFiles/TopoSortTest.dir/core/nomnigraph/tests/TopoSortTest.cc.o"

# External object files for target TopoSortTest
TopoSortTest_EXTERNAL_OBJECTS =

bin/TopoSortTest: caffe2/CMakeFiles/TopoSortTest.dir/core/nomnigraph/tests/TopoSortTest.cc.o
bin/TopoSortTest: caffe2/CMakeFiles/TopoSortTest.dir/build.make
bin/TopoSortTest: lib/libgtest_main.a
bin/TopoSortTest: /usr/lib/gcc/x86_64-linux-gnu/7/libgomp.so
bin/TopoSortTest: /usr/lib/x86_64-linux-gnu/libpthread.so
bin/TopoSortTest: lib/libprotobuf.a
bin/TopoSortTest: lib/libdnnl.a
bin/TopoSortTest: lib/libc10_cuda.so
bin/TopoSortTest: lib/libc10.so
bin/TopoSortTest: /usr/local/cuda/lib64/libcudart.so
bin/TopoSortTest: /usr/local/cuda/lib64/libnvToolsExt.so
bin/TopoSortTest: /usr/local/cuda/lib64/libcufft.so
bin/TopoSortTest: /usr/local/cuda/lib64/libcurand.so
bin/TopoSortTest: /usr/local/cuda/lib64/libcublas.so
bin/TopoSortTest: /usr/lib/x86_64-linux-gnu/libcudnn.so
bin/TopoSortTest: lib/libgtest.a
bin/TopoSortTest: caffe2/CMakeFiles/TopoSortTest.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/TopoSortTest"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2 && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/TopoSortTest.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
caffe2/CMakeFiles/TopoSortTest.dir/build: bin/TopoSortTest

.PHONY : caffe2/CMakeFiles/TopoSortTest.dir/build

caffe2/CMakeFiles/TopoSortTest.dir/requires: caffe2/CMakeFiles/TopoSortTest.dir/core/nomnigraph/tests/TopoSortTest.cc.o.requires

.PHONY : caffe2/CMakeFiles/TopoSortTest.dir/requires

caffe2/CMakeFiles/TopoSortTest.dir/clean:
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2 && $(CMAKE_COMMAND) -P CMakeFiles/TopoSortTest.dir/cmake_clean.cmake
.PHONY : caffe2/CMakeFiles/TopoSortTest.dir/clean

caffe2/CMakeFiles/TopoSortTest.dir/depend:
	cd /home/zzp/code/NoPFS/pytorch/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zzp/code/NoPFS/pytorch /home/zzp/code/NoPFS/pytorch/caffe2 /home/zzp/code/NoPFS/pytorch/build /home/zzp/code/NoPFS/pytorch/build/caffe2 /home/zzp/code/NoPFS/pytorch/build/caffe2/CMakeFiles/TopoSortTest.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : caffe2/CMakeFiles/TopoSortTest.dir/depend

