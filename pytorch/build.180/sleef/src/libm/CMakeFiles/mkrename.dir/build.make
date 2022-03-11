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
include sleef/src/libm/CMakeFiles/mkrename.dir/depend.make

# Include the progress variables for this target.
include sleef/src/libm/CMakeFiles/mkrename.dir/progress.make

# Include the compile flags for this target's objects.
include sleef/src/libm/CMakeFiles/mkrename.dir/flags.make

sleef/src/libm/CMakeFiles/mkrename.dir/mkrename.c.o: sleef/src/libm/CMakeFiles/mkrename.dir/flags.make
sleef/src/libm/CMakeFiles/mkrename.dir/mkrename.c.o: ../third_party/sleef/src/libm/mkrename.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object sleef/src/libm/CMakeFiles/mkrename.dir/mkrename.c.o"
	cd /home/zzp/code/NoPFS/pytorch/build/sleef/src/libm && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/mkrename.dir/mkrename.c.o   -c /home/zzp/code/NoPFS/pytorch/third_party/sleef/src/libm/mkrename.c

sleef/src/libm/CMakeFiles/mkrename.dir/mkrename.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/mkrename.dir/mkrename.c.i"
	cd /home/zzp/code/NoPFS/pytorch/build/sleef/src/libm && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/zzp/code/NoPFS/pytorch/third_party/sleef/src/libm/mkrename.c > CMakeFiles/mkrename.dir/mkrename.c.i

sleef/src/libm/CMakeFiles/mkrename.dir/mkrename.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/mkrename.dir/mkrename.c.s"
	cd /home/zzp/code/NoPFS/pytorch/build/sleef/src/libm && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/zzp/code/NoPFS/pytorch/third_party/sleef/src/libm/mkrename.c -o CMakeFiles/mkrename.dir/mkrename.c.s

sleef/src/libm/CMakeFiles/mkrename.dir/mkrename.c.o.requires:

.PHONY : sleef/src/libm/CMakeFiles/mkrename.dir/mkrename.c.o.requires

sleef/src/libm/CMakeFiles/mkrename.dir/mkrename.c.o.provides: sleef/src/libm/CMakeFiles/mkrename.dir/mkrename.c.o.requires
	$(MAKE) -f sleef/src/libm/CMakeFiles/mkrename.dir/build.make sleef/src/libm/CMakeFiles/mkrename.dir/mkrename.c.o.provides.build
.PHONY : sleef/src/libm/CMakeFiles/mkrename.dir/mkrename.c.o.provides

sleef/src/libm/CMakeFiles/mkrename.dir/mkrename.c.o.provides.build: sleef/src/libm/CMakeFiles/mkrename.dir/mkrename.c.o


# Object files for target mkrename
mkrename_OBJECTS = \
"CMakeFiles/mkrename.dir/mkrename.c.o"

# External object files for target mkrename
mkrename_EXTERNAL_OBJECTS =

sleef/bin/mkrename: sleef/src/libm/CMakeFiles/mkrename.dir/mkrename.c.o
sleef/bin/mkrename: sleef/src/libm/CMakeFiles/mkrename.dir/build.make
sleef/bin/mkrename: sleef/src/libm/CMakeFiles/mkrename.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable ../../bin/mkrename"
	cd /home/zzp/code/NoPFS/pytorch/build/sleef/src/libm && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mkrename.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
sleef/src/libm/CMakeFiles/mkrename.dir/build: sleef/bin/mkrename

.PHONY : sleef/src/libm/CMakeFiles/mkrename.dir/build

sleef/src/libm/CMakeFiles/mkrename.dir/requires: sleef/src/libm/CMakeFiles/mkrename.dir/mkrename.c.o.requires

.PHONY : sleef/src/libm/CMakeFiles/mkrename.dir/requires

sleef/src/libm/CMakeFiles/mkrename.dir/clean:
	cd /home/zzp/code/NoPFS/pytorch/build/sleef/src/libm && $(CMAKE_COMMAND) -P CMakeFiles/mkrename.dir/cmake_clean.cmake
.PHONY : sleef/src/libm/CMakeFiles/mkrename.dir/clean

sleef/src/libm/CMakeFiles/mkrename.dir/depend:
	cd /home/zzp/code/NoPFS/pytorch/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zzp/code/NoPFS/pytorch /home/zzp/code/NoPFS/pytorch/third_party/sleef/src/libm /home/zzp/code/NoPFS/pytorch/build /home/zzp/code/NoPFS/pytorch/build/sleef/src/libm /home/zzp/code/NoPFS/pytorch/build/sleef/src/libm/CMakeFiles/mkrename.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : sleef/src/libm/CMakeFiles/mkrename.dir/depend

