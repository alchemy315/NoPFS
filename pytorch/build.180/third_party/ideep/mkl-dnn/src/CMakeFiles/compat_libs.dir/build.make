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

# Utility rule file for compat_libs.

# Include the progress variables for this target.
include third_party/ideep/mkl-dnn/src/CMakeFiles/compat_libs.dir/progress.make

third_party/ideep/mkl-dnn/src/CMakeFiles/compat_libs: lib/libmkldnn.a


lib/libmkldnn.a: lib/libdnnl.a
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating ../../../../lib/libmkldnn.a"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/ideep/mkl-dnn/src && /usr/bin/cmake -E remove -f /home/zzp/code/NoPFS/pytorch/build/lib/libmkldnn.a
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/ideep/mkl-dnn/src && /usr/bin/cmake -E create_symlink libdnnl.a /home/zzp/code/NoPFS/pytorch/build/lib/libmkldnn.a

compat_libs: third_party/ideep/mkl-dnn/src/CMakeFiles/compat_libs
compat_libs: lib/libmkldnn.a
compat_libs: third_party/ideep/mkl-dnn/src/CMakeFiles/compat_libs.dir/build.make

.PHONY : compat_libs

# Rule to build all files generated by this target.
third_party/ideep/mkl-dnn/src/CMakeFiles/compat_libs.dir/build: compat_libs

.PHONY : third_party/ideep/mkl-dnn/src/CMakeFiles/compat_libs.dir/build

third_party/ideep/mkl-dnn/src/CMakeFiles/compat_libs.dir/clean:
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/ideep/mkl-dnn/src && $(CMAKE_COMMAND) -P CMakeFiles/compat_libs.dir/cmake_clean.cmake
.PHONY : third_party/ideep/mkl-dnn/src/CMakeFiles/compat_libs.dir/clean

third_party/ideep/mkl-dnn/src/CMakeFiles/compat_libs.dir/depend:
	cd /home/zzp/code/NoPFS/pytorch/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zzp/code/NoPFS/pytorch /home/zzp/code/NoPFS/pytorch/third_party/ideep/mkl-dnn/src /home/zzp/code/NoPFS/pytorch/build /home/zzp/code/NoPFS/pytorch/build/third_party/ideep/mkl-dnn/src /home/zzp/code/NoPFS/pytorch/build/third_party/ideep/mkl-dnn/src/CMakeFiles/compat_libs.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : third_party/ideep/mkl-dnn/src/CMakeFiles/compat_libs.dir/depend

