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

# Utility rule file for renamePURECFMA_SCALAR.h_generated.

# Include the progress variables for this target.
include sleef/src/libm/CMakeFiles/renamePURECFMA_SCALAR.h_generated.dir/progress.make

sleef/src/libm/CMakeFiles/renamePURECFMA_SCALAR.h_generated: sleef/src/libm/include/renamepurecfma_scalar.h


sleef/src/libm/include/renamepurecfma_scalar.h: sleef/bin/mkrename
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating include/renamepurecfma_scalar.h"
	cd /home/zzp/code/NoPFS/pytorch/build/sleef/src/libm && echo Generating renamepurecfma_scalar.h: mkrename "finz_" "1" "1" "purecfma"
	cd /home/zzp/code/NoPFS/pytorch/build/sleef/src/libm && ../../bin/mkrename "finz_" "1" "1" "purecfma" > /home/zzp/code/NoPFS/pytorch/build/sleef/src/libm/include/renamepurecfma_scalar.h

renamePURECFMA_SCALAR.h_generated: sleef/src/libm/CMakeFiles/renamePURECFMA_SCALAR.h_generated
renamePURECFMA_SCALAR.h_generated: sleef/src/libm/include/renamepurecfma_scalar.h
renamePURECFMA_SCALAR.h_generated: sleef/src/libm/CMakeFiles/renamePURECFMA_SCALAR.h_generated.dir/build.make

.PHONY : renamePURECFMA_SCALAR.h_generated

# Rule to build all files generated by this target.
sleef/src/libm/CMakeFiles/renamePURECFMA_SCALAR.h_generated.dir/build: renamePURECFMA_SCALAR.h_generated

.PHONY : sleef/src/libm/CMakeFiles/renamePURECFMA_SCALAR.h_generated.dir/build

sleef/src/libm/CMakeFiles/renamePURECFMA_SCALAR.h_generated.dir/clean:
	cd /home/zzp/code/NoPFS/pytorch/build/sleef/src/libm && $(CMAKE_COMMAND) -P CMakeFiles/renamePURECFMA_SCALAR.h_generated.dir/cmake_clean.cmake
.PHONY : sleef/src/libm/CMakeFiles/renamePURECFMA_SCALAR.h_generated.dir/clean

sleef/src/libm/CMakeFiles/renamePURECFMA_SCALAR.h_generated.dir/depend:
	cd /home/zzp/code/NoPFS/pytorch/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zzp/code/NoPFS/pytorch /home/zzp/code/NoPFS/pytorch/third_party/sleef/src/libm /home/zzp/code/NoPFS/pytorch/build /home/zzp/code/NoPFS/pytorch/build/sleef/src/libm /home/zzp/code/NoPFS/pytorch/build/sleef/src/libm/CMakeFiles/renamePURECFMA_SCALAR.h_generated.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : sleef/src/libm/CMakeFiles/renamePURECFMA_SCALAR.h_generated.dir/depend

