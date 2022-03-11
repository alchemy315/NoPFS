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
include sleef/src/libm/CMakeFiles/dispsse_obj.dir/depend.make

# Include the progress variables for this target.
include sleef/src/libm/CMakeFiles/dispsse_obj.dir/progress.make

# Include the compile flags for this target's objects.
include sleef/src/libm/CMakeFiles/dispsse_obj.dir/flags.make

sleef/src/libm/dispsse.c: sleef/bin/mkdisp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating dispsse.c"
	cd /home/zzp/code/NoPFS/pytorch/build/sleef/src/libm && /usr/bin/cmake -E copy /home/zzp/code/NoPFS/pytorch/third_party/sleef/src/libm/dispsse.c.org /home/zzp/code/NoPFS/pytorch/build/sleef/src/libm/dispsse.c
	cd /home/zzp/code/NoPFS/pytorch/build/sleef/src/libm && ../../bin/mkdisp 2 4 __m128d __m128 __m128i sse2 sse4 avx2128 >> /home/zzp/code/NoPFS/pytorch/build/sleef/src/libm/dispsse.c

sleef/src/libm/CMakeFiles/dispsse_obj.dir/dispsse.c.o: sleef/src/libm/CMakeFiles/dispsse_obj.dir/flags.make
sleef/src/libm/CMakeFiles/dispsse_obj.dir/dispsse.c.o: sleef/src/libm/dispsse.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object sleef/src/libm/CMakeFiles/dispsse_obj.dir/dispsse.c.o"
	cd /home/zzp/code/NoPFS/pytorch/build/sleef/src/libm && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/dispsse_obj.dir/dispsse.c.o   -c /home/zzp/code/NoPFS/pytorch/build/sleef/src/libm/dispsse.c

sleef/src/libm/CMakeFiles/dispsse_obj.dir/dispsse.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/dispsse_obj.dir/dispsse.c.i"
	cd /home/zzp/code/NoPFS/pytorch/build/sleef/src/libm && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/zzp/code/NoPFS/pytorch/build/sleef/src/libm/dispsse.c > CMakeFiles/dispsse_obj.dir/dispsse.c.i

sleef/src/libm/CMakeFiles/dispsse_obj.dir/dispsse.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/dispsse_obj.dir/dispsse.c.s"
	cd /home/zzp/code/NoPFS/pytorch/build/sleef/src/libm && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/zzp/code/NoPFS/pytorch/build/sleef/src/libm/dispsse.c -o CMakeFiles/dispsse_obj.dir/dispsse.c.s

sleef/src/libm/CMakeFiles/dispsse_obj.dir/dispsse.c.o.requires:

.PHONY : sleef/src/libm/CMakeFiles/dispsse_obj.dir/dispsse.c.o.requires

sleef/src/libm/CMakeFiles/dispsse_obj.dir/dispsse.c.o.provides: sleef/src/libm/CMakeFiles/dispsse_obj.dir/dispsse.c.o.requires
	$(MAKE) -f sleef/src/libm/CMakeFiles/dispsse_obj.dir/build.make sleef/src/libm/CMakeFiles/dispsse_obj.dir/dispsse.c.o.provides.build
.PHONY : sleef/src/libm/CMakeFiles/dispsse_obj.dir/dispsse.c.o.provides

sleef/src/libm/CMakeFiles/dispsse_obj.dir/dispsse.c.o.provides.build: sleef/src/libm/CMakeFiles/dispsse_obj.dir/dispsse.c.o


dispsse_obj: sleef/src/libm/CMakeFiles/dispsse_obj.dir/dispsse.c.o
dispsse_obj: sleef/src/libm/CMakeFiles/dispsse_obj.dir/build.make

.PHONY : dispsse_obj

# Rule to build all files generated by this target.
sleef/src/libm/CMakeFiles/dispsse_obj.dir/build: dispsse_obj

.PHONY : sleef/src/libm/CMakeFiles/dispsse_obj.dir/build

sleef/src/libm/CMakeFiles/dispsse_obj.dir/requires: sleef/src/libm/CMakeFiles/dispsse_obj.dir/dispsse.c.o.requires

.PHONY : sleef/src/libm/CMakeFiles/dispsse_obj.dir/requires

sleef/src/libm/CMakeFiles/dispsse_obj.dir/clean:
	cd /home/zzp/code/NoPFS/pytorch/build/sleef/src/libm && $(CMAKE_COMMAND) -P CMakeFiles/dispsse_obj.dir/cmake_clean.cmake
.PHONY : sleef/src/libm/CMakeFiles/dispsse_obj.dir/clean

sleef/src/libm/CMakeFiles/dispsse_obj.dir/depend: sleef/src/libm/dispsse.c
	cd /home/zzp/code/NoPFS/pytorch/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zzp/code/NoPFS/pytorch /home/zzp/code/NoPFS/pytorch/third_party/sleef/src/libm /home/zzp/code/NoPFS/pytorch/build /home/zzp/code/NoPFS/pytorch/build/sleef/src/libm /home/zzp/code/NoPFS/pytorch/build/sleef/src/libm/CMakeFiles/dispsse_obj.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : sleef/src/libm/CMakeFiles/dispsse_obj.dir/depend
