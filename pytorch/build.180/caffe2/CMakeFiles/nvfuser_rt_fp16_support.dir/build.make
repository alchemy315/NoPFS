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

# Utility rule file for nvfuser_rt_fp16_support.

# Include the progress variables for this target.
include caffe2/CMakeFiles/nvfuser_rt_fp16_support.dir/progress.make

caffe2/CMakeFiles/nvfuser_rt_fp16_support: include/nvfuser_resources/fp16_support.h


include/nvfuser_resources/fp16_support.h: ../torch/csrc/jit/codegen/cuda/runtime/fp16_support.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Stringify NVFUSER runtime source file"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2 && /opt/conda/bin/python /home/zzp/code/NoPFS/pytorch/torch/csrc/jit/codegen/cuda/tools/stringify_file.py -i /home/zzp/code/NoPFS/pytorch/torch/csrc/jit/codegen/cuda/runtime/fp16_support.cu -o /home/zzp/code/NoPFS/pytorch/build/include/nvfuser_resources/fp16_support.h

nvfuser_rt_fp16_support: caffe2/CMakeFiles/nvfuser_rt_fp16_support
nvfuser_rt_fp16_support: include/nvfuser_resources/fp16_support.h
nvfuser_rt_fp16_support: caffe2/CMakeFiles/nvfuser_rt_fp16_support.dir/build.make

.PHONY : nvfuser_rt_fp16_support

# Rule to build all files generated by this target.
caffe2/CMakeFiles/nvfuser_rt_fp16_support.dir/build: nvfuser_rt_fp16_support

.PHONY : caffe2/CMakeFiles/nvfuser_rt_fp16_support.dir/build

caffe2/CMakeFiles/nvfuser_rt_fp16_support.dir/clean:
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2 && $(CMAKE_COMMAND) -P CMakeFiles/nvfuser_rt_fp16_support.dir/cmake_clean.cmake
.PHONY : caffe2/CMakeFiles/nvfuser_rt_fp16_support.dir/clean

caffe2/CMakeFiles/nvfuser_rt_fp16_support.dir/depend:
	cd /home/zzp/code/NoPFS/pytorch/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zzp/code/NoPFS/pytorch /home/zzp/code/NoPFS/pytorch/caffe2 /home/zzp/code/NoPFS/pytorch/build /home/zzp/code/NoPFS/pytorch/build/caffe2 /home/zzp/code/NoPFS/pytorch/build/caffe2/CMakeFiles/nvfuser_rt_fp16_support.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : caffe2/CMakeFiles/nvfuser_rt_fp16_support.dir/depend
