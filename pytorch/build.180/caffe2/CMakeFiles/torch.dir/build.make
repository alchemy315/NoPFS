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
include caffe2/CMakeFiles/torch.dir/depend.make

# Include the progress variables for this target.
include caffe2/CMakeFiles/torch.dir/progress.make

# Include the compile flags for this target's objects.
include caffe2/CMakeFiles/torch.dir/flags.make

caffe2/CMakeFiles/torch.dir/__/empty.cpp.o: caffe2/CMakeFiles/torch.dir/flags.make
caffe2/CMakeFiles/torch.dir/__/empty.cpp.o: empty.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object caffe2/CMakeFiles/torch.dir/__/empty.cpp.o"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2 && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/torch.dir/__/empty.cpp.o -c /home/zzp/code/NoPFS/pytorch/build/empty.cpp

caffe2/CMakeFiles/torch.dir/__/empty.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/torch.dir/__/empty.cpp.i"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zzp/code/NoPFS/pytorch/build/empty.cpp > CMakeFiles/torch.dir/__/empty.cpp.i

caffe2/CMakeFiles/torch.dir/__/empty.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/torch.dir/__/empty.cpp.s"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zzp/code/NoPFS/pytorch/build/empty.cpp -o CMakeFiles/torch.dir/__/empty.cpp.s

caffe2/CMakeFiles/torch.dir/__/empty.cpp.o.requires:

.PHONY : caffe2/CMakeFiles/torch.dir/__/empty.cpp.o.requires

caffe2/CMakeFiles/torch.dir/__/empty.cpp.o.provides: caffe2/CMakeFiles/torch.dir/__/empty.cpp.o.requires
	$(MAKE) -f caffe2/CMakeFiles/torch.dir/build.make caffe2/CMakeFiles/torch.dir/__/empty.cpp.o.provides.build
.PHONY : caffe2/CMakeFiles/torch.dir/__/empty.cpp.o.provides

caffe2/CMakeFiles/torch.dir/__/empty.cpp.o.provides.build: caffe2/CMakeFiles/torch.dir/__/empty.cpp.o


# Object files for target torch
torch_OBJECTS = \
"CMakeFiles/torch.dir/__/empty.cpp.o"

# External object files for target torch
torch_EXTERNAL_OBJECTS =

lib/libtorch.so: caffe2/CMakeFiles/torch.dir/__/empty.cpp.o
lib/libtorch.so: caffe2/CMakeFiles/torch.dir/build.make
lib/libtorch.so: lib/libprotobuf.a
lib/libtorch.so: lib/libdnnl.a
lib/libtorch.so: lib/libc10_cuda.so
lib/libtorch.so: lib/libc10.so
lib/libtorch.so: /usr/local/cuda/lib64/libcudart.so
lib/libtorch.so: /usr/local/cuda/lib64/libnvToolsExt.so
lib/libtorch.so: /usr/local/cuda/lib64/libcufft.so
lib/libtorch.so: /usr/local/cuda/lib64/libcurand.so
lib/libtorch.so: /usr/local/cuda/lib64/libcublas.so
lib/libtorch.so: /usr/lib/x86_64-linux-gnu/libcudnn.so
lib/libtorch.so: caffe2/CMakeFiles/torch.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library ../lib/libtorch.so"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2 && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/torch.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
caffe2/CMakeFiles/torch.dir/build: lib/libtorch.so

.PHONY : caffe2/CMakeFiles/torch.dir/build

caffe2/CMakeFiles/torch.dir/requires: caffe2/CMakeFiles/torch.dir/__/empty.cpp.o.requires

.PHONY : caffe2/CMakeFiles/torch.dir/requires

caffe2/CMakeFiles/torch.dir/clean:
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2 && $(CMAKE_COMMAND) -P CMakeFiles/torch.dir/cmake_clean.cmake
.PHONY : caffe2/CMakeFiles/torch.dir/clean

caffe2/CMakeFiles/torch.dir/depend:
	cd /home/zzp/code/NoPFS/pytorch/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zzp/code/NoPFS/pytorch /home/zzp/code/NoPFS/pytorch/caffe2 /home/zzp/code/NoPFS/pytorch/build /home/zzp/code/NoPFS/pytorch/build/caffe2 /home/zzp/code/NoPFS/pytorch/build/caffe2/CMakeFiles/torch.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : caffe2/CMakeFiles/torch.dir/depend
