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
include sleef/src/common/CMakeFiles/addSuffix.dir/depend.make

# Include the progress variables for this target.
include sleef/src/common/CMakeFiles/addSuffix.dir/progress.make

# Include the compile flags for this target's objects.
include sleef/src/common/CMakeFiles/addSuffix.dir/flags.make

sleef/src/common/CMakeFiles/addSuffix.dir/addSuffix.c.o: sleef/src/common/CMakeFiles/addSuffix.dir/flags.make
sleef/src/common/CMakeFiles/addSuffix.dir/addSuffix.c.o: ../third_party/sleef/src/common/addSuffix.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object sleef/src/common/CMakeFiles/addSuffix.dir/addSuffix.c.o"
	cd /home/zzp/code/NoPFS/pytorch/build/sleef/src/common && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/addSuffix.dir/addSuffix.c.o   -c /home/zzp/code/NoPFS/pytorch/third_party/sleef/src/common/addSuffix.c

sleef/src/common/CMakeFiles/addSuffix.dir/addSuffix.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/addSuffix.dir/addSuffix.c.i"
	cd /home/zzp/code/NoPFS/pytorch/build/sleef/src/common && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/zzp/code/NoPFS/pytorch/third_party/sleef/src/common/addSuffix.c > CMakeFiles/addSuffix.dir/addSuffix.c.i

sleef/src/common/CMakeFiles/addSuffix.dir/addSuffix.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/addSuffix.dir/addSuffix.c.s"
	cd /home/zzp/code/NoPFS/pytorch/build/sleef/src/common && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/zzp/code/NoPFS/pytorch/third_party/sleef/src/common/addSuffix.c -o CMakeFiles/addSuffix.dir/addSuffix.c.s

sleef/src/common/CMakeFiles/addSuffix.dir/addSuffix.c.o.requires:

.PHONY : sleef/src/common/CMakeFiles/addSuffix.dir/addSuffix.c.o.requires

sleef/src/common/CMakeFiles/addSuffix.dir/addSuffix.c.o.provides: sleef/src/common/CMakeFiles/addSuffix.dir/addSuffix.c.o.requires
	$(MAKE) -f sleef/src/common/CMakeFiles/addSuffix.dir/build.make sleef/src/common/CMakeFiles/addSuffix.dir/addSuffix.c.o.provides.build
.PHONY : sleef/src/common/CMakeFiles/addSuffix.dir/addSuffix.c.o.provides

sleef/src/common/CMakeFiles/addSuffix.dir/addSuffix.c.o.provides.build: sleef/src/common/CMakeFiles/addSuffix.dir/addSuffix.c.o


# Object files for target addSuffix
addSuffix_OBJECTS = \
"CMakeFiles/addSuffix.dir/addSuffix.c.o"

# External object files for target addSuffix
addSuffix_EXTERNAL_OBJECTS =

sleef/bin/addSuffix: sleef/src/common/CMakeFiles/addSuffix.dir/addSuffix.c.o
sleef/bin/addSuffix: sleef/src/common/CMakeFiles/addSuffix.dir/build.make
sleef/bin/addSuffix: sleef/src/common/CMakeFiles/addSuffix.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable ../../bin/addSuffix"
	cd /home/zzp/code/NoPFS/pytorch/build/sleef/src/common && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/addSuffix.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
sleef/src/common/CMakeFiles/addSuffix.dir/build: sleef/bin/addSuffix

.PHONY : sleef/src/common/CMakeFiles/addSuffix.dir/build

sleef/src/common/CMakeFiles/addSuffix.dir/requires: sleef/src/common/CMakeFiles/addSuffix.dir/addSuffix.c.o.requires

.PHONY : sleef/src/common/CMakeFiles/addSuffix.dir/requires

sleef/src/common/CMakeFiles/addSuffix.dir/clean:
	cd /home/zzp/code/NoPFS/pytorch/build/sleef/src/common && $(CMAKE_COMMAND) -P CMakeFiles/addSuffix.dir/cmake_clean.cmake
.PHONY : sleef/src/common/CMakeFiles/addSuffix.dir/clean

sleef/src/common/CMakeFiles/addSuffix.dir/depend:
	cd /home/zzp/code/NoPFS/pytorch/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zzp/code/NoPFS/pytorch /home/zzp/code/NoPFS/pytorch/third_party/sleef/src/common /home/zzp/code/NoPFS/pytorch/build /home/zzp/code/NoPFS/pytorch/build/sleef/src/common /home/zzp/code/NoPFS/pytorch/build/sleef/src/common/CMakeFiles/addSuffix.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : sleef/src/common/CMakeFiles/addSuffix.dir/depend

