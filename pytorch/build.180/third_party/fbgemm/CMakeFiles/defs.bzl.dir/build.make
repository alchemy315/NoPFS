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

# Utility rule file for defs.bzl.

# Include the progress variables for this target.
include third_party/fbgemm/CMakeFiles/defs.bzl.dir/progress.make

third_party/fbgemm/CMakeFiles/defs.bzl:


defs.bzl: third_party/fbgemm/CMakeFiles/defs.bzl
defs.bzl: third_party/fbgemm/CMakeFiles/defs.bzl.dir/build.make

.PHONY : defs.bzl

# Rule to build all files generated by this target.
third_party/fbgemm/CMakeFiles/defs.bzl.dir/build: defs.bzl

.PHONY : third_party/fbgemm/CMakeFiles/defs.bzl.dir/build

third_party/fbgemm/CMakeFiles/defs.bzl.dir/clean:
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm && $(CMAKE_COMMAND) -P CMakeFiles/defs.bzl.dir/cmake_clean.cmake
.PHONY : third_party/fbgemm/CMakeFiles/defs.bzl.dir/clean

third_party/fbgemm/CMakeFiles/defs.bzl.dir/depend:
	cd /home/zzp/code/NoPFS/pytorch/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zzp/code/NoPFS/pytorch /home/zzp/code/NoPFS/pytorch/third_party/fbgemm /home/zzp/code/NoPFS/pytorch/build /home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm /home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm/CMakeFiles/defs.bzl.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : third_party/fbgemm/CMakeFiles/defs.bzl.dir/depend

