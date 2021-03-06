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
include confu-deps/cpuinfo/deps/clog/CMakeFiles/clog.dir/depend.make

# Include the progress variables for this target.
include confu-deps/cpuinfo/deps/clog/CMakeFiles/clog.dir/progress.make

# Include the compile flags for this target's objects.
include confu-deps/cpuinfo/deps/clog/CMakeFiles/clog.dir/flags.make

confu-deps/cpuinfo/deps/clog/CMakeFiles/clog.dir/src/clog.c.o: confu-deps/cpuinfo/deps/clog/CMakeFiles/clog.dir/flags.make
confu-deps/cpuinfo/deps/clog/CMakeFiles/clog.dir/src/clog.c.o: ../third_party/cpuinfo/deps/clog/src/clog.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object confu-deps/cpuinfo/deps/clog/CMakeFiles/clog.dir/src/clog.c.o"
	cd /home/zzp/code/NoPFS/pytorch/build/confu-deps/cpuinfo/deps/clog && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/clog.dir/src/clog.c.o   -c /home/zzp/code/NoPFS/pytorch/third_party/cpuinfo/deps/clog/src/clog.c

confu-deps/cpuinfo/deps/clog/CMakeFiles/clog.dir/src/clog.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/clog.dir/src/clog.c.i"
	cd /home/zzp/code/NoPFS/pytorch/build/confu-deps/cpuinfo/deps/clog && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/zzp/code/NoPFS/pytorch/third_party/cpuinfo/deps/clog/src/clog.c > CMakeFiles/clog.dir/src/clog.c.i

confu-deps/cpuinfo/deps/clog/CMakeFiles/clog.dir/src/clog.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/clog.dir/src/clog.c.s"
	cd /home/zzp/code/NoPFS/pytorch/build/confu-deps/cpuinfo/deps/clog && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/zzp/code/NoPFS/pytorch/third_party/cpuinfo/deps/clog/src/clog.c -o CMakeFiles/clog.dir/src/clog.c.s

confu-deps/cpuinfo/deps/clog/CMakeFiles/clog.dir/src/clog.c.o.requires:

.PHONY : confu-deps/cpuinfo/deps/clog/CMakeFiles/clog.dir/src/clog.c.o.requires

confu-deps/cpuinfo/deps/clog/CMakeFiles/clog.dir/src/clog.c.o.provides: confu-deps/cpuinfo/deps/clog/CMakeFiles/clog.dir/src/clog.c.o.requires
	$(MAKE) -f confu-deps/cpuinfo/deps/clog/CMakeFiles/clog.dir/build.make confu-deps/cpuinfo/deps/clog/CMakeFiles/clog.dir/src/clog.c.o.provides.build
.PHONY : confu-deps/cpuinfo/deps/clog/CMakeFiles/clog.dir/src/clog.c.o.provides

confu-deps/cpuinfo/deps/clog/CMakeFiles/clog.dir/src/clog.c.o.provides.build: confu-deps/cpuinfo/deps/clog/CMakeFiles/clog.dir/src/clog.c.o


# Object files for target clog
clog_OBJECTS = \
"CMakeFiles/clog.dir/src/clog.c.o"

# External object files for target clog
clog_EXTERNAL_OBJECTS =

lib/libclog.a: confu-deps/cpuinfo/deps/clog/CMakeFiles/clog.dir/src/clog.c.o
lib/libclog.a: confu-deps/cpuinfo/deps/clog/CMakeFiles/clog.dir/build.make
lib/libclog.a: confu-deps/cpuinfo/deps/clog/CMakeFiles/clog.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C static library ../../../../lib/libclog.a"
	cd /home/zzp/code/NoPFS/pytorch/build/confu-deps/cpuinfo/deps/clog && $(CMAKE_COMMAND) -P CMakeFiles/clog.dir/cmake_clean_target.cmake
	cd /home/zzp/code/NoPFS/pytorch/build/confu-deps/cpuinfo/deps/clog && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/clog.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
confu-deps/cpuinfo/deps/clog/CMakeFiles/clog.dir/build: lib/libclog.a

.PHONY : confu-deps/cpuinfo/deps/clog/CMakeFiles/clog.dir/build

confu-deps/cpuinfo/deps/clog/CMakeFiles/clog.dir/requires: confu-deps/cpuinfo/deps/clog/CMakeFiles/clog.dir/src/clog.c.o.requires

.PHONY : confu-deps/cpuinfo/deps/clog/CMakeFiles/clog.dir/requires

confu-deps/cpuinfo/deps/clog/CMakeFiles/clog.dir/clean:
	cd /home/zzp/code/NoPFS/pytorch/build/confu-deps/cpuinfo/deps/clog && $(CMAKE_COMMAND) -P CMakeFiles/clog.dir/cmake_clean.cmake
.PHONY : confu-deps/cpuinfo/deps/clog/CMakeFiles/clog.dir/clean

confu-deps/cpuinfo/deps/clog/CMakeFiles/clog.dir/depend:
	cd /home/zzp/code/NoPFS/pytorch/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zzp/code/NoPFS/pytorch /home/zzp/code/NoPFS/pytorch/third_party/cpuinfo/deps/clog /home/zzp/code/NoPFS/pytorch/build /home/zzp/code/NoPFS/pytorch/build/confu-deps/cpuinfo/deps/clog /home/zzp/code/NoPFS/pytorch/build/confu-deps/cpuinfo/deps/clog/CMakeFiles/clog.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : confu-deps/cpuinfo/deps/clog/CMakeFiles/clog.dir/depend

