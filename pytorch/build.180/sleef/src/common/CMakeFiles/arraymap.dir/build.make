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
include sleef/src/common/CMakeFiles/arraymap.dir/depend.make

# Include the progress variables for this target.
include sleef/src/common/CMakeFiles/arraymap.dir/progress.make

# Include the compile flags for this target's objects.
include sleef/src/common/CMakeFiles/arraymap.dir/flags.make

sleef/src/common/CMakeFiles/arraymap.dir/arraymap.c.o: sleef/src/common/CMakeFiles/arraymap.dir/flags.make
sleef/src/common/CMakeFiles/arraymap.dir/arraymap.c.o: ../third_party/sleef/src/common/arraymap.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object sleef/src/common/CMakeFiles/arraymap.dir/arraymap.c.o"
	cd /home/zzp/code/NoPFS/pytorch/build/sleef/src/common && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/arraymap.dir/arraymap.c.o   -c /home/zzp/code/NoPFS/pytorch/third_party/sleef/src/common/arraymap.c

sleef/src/common/CMakeFiles/arraymap.dir/arraymap.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/arraymap.dir/arraymap.c.i"
	cd /home/zzp/code/NoPFS/pytorch/build/sleef/src/common && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/zzp/code/NoPFS/pytorch/third_party/sleef/src/common/arraymap.c > CMakeFiles/arraymap.dir/arraymap.c.i

sleef/src/common/CMakeFiles/arraymap.dir/arraymap.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/arraymap.dir/arraymap.c.s"
	cd /home/zzp/code/NoPFS/pytorch/build/sleef/src/common && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/zzp/code/NoPFS/pytorch/third_party/sleef/src/common/arraymap.c -o CMakeFiles/arraymap.dir/arraymap.c.s

sleef/src/common/CMakeFiles/arraymap.dir/arraymap.c.o.requires:

.PHONY : sleef/src/common/CMakeFiles/arraymap.dir/arraymap.c.o.requires

sleef/src/common/CMakeFiles/arraymap.dir/arraymap.c.o.provides: sleef/src/common/CMakeFiles/arraymap.dir/arraymap.c.o.requires
	$(MAKE) -f sleef/src/common/CMakeFiles/arraymap.dir/build.make sleef/src/common/CMakeFiles/arraymap.dir/arraymap.c.o.provides.build
.PHONY : sleef/src/common/CMakeFiles/arraymap.dir/arraymap.c.o.provides

sleef/src/common/CMakeFiles/arraymap.dir/arraymap.c.o.provides.build: sleef/src/common/CMakeFiles/arraymap.dir/arraymap.c.o


arraymap: sleef/src/common/CMakeFiles/arraymap.dir/arraymap.c.o
arraymap: sleef/src/common/CMakeFiles/arraymap.dir/build.make

.PHONY : arraymap

# Rule to build all files generated by this target.
sleef/src/common/CMakeFiles/arraymap.dir/build: arraymap

.PHONY : sleef/src/common/CMakeFiles/arraymap.dir/build

sleef/src/common/CMakeFiles/arraymap.dir/requires: sleef/src/common/CMakeFiles/arraymap.dir/arraymap.c.o.requires

.PHONY : sleef/src/common/CMakeFiles/arraymap.dir/requires

sleef/src/common/CMakeFiles/arraymap.dir/clean:
	cd /home/zzp/code/NoPFS/pytorch/build/sleef/src/common && $(CMAKE_COMMAND) -P CMakeFiles/arraymap.dir/cmake_clean.cmake
.PHONY : sleef/src/common/CMakeFiles/arraymap.dir/clean

sleef/src/common/CMakeFiles/arraymap.dir/depend:
	cd /home/zzp/code/NoPFS/pytorch/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zzp/code/NoPFS/pytorch /home/zzp/code/NoPFS/pytorch/third_party/sleef/src/common /home/zzp/code/NoPFS/pytorch/build /home/zzp/code/NoPFS/pytorch/build/sleef/src/common /home/zzp/code/NoPFS/pytorch/build/sleef/src/common/CMakeFiles/arraymap.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : sleef/src/common/CMakeFiles/arraymap.dir/depend

