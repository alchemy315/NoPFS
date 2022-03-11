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
include third_party/foxi/CMakeFiles/foxi_wrapper.dir/depend.make

# Include the progress variables for this target.
include third_party/foxi/CMakeFiles/foxi_wrapper.dir/progress.make

# Include the compile flags for this target's objects.
include third_party/foxi/CMakeFiles/foxi_wrapper.dir/flags.make

third_party/foxi/CMakeFiles/foxi_wrapper.dir/foxi/onnxifi_wrapper.c.o: third_party/foxi/CMakeFiles/foxi_wrapper.dir/flags.make
third_party/foxi/CMakeFiles/foxi_wrapper.dir/foxi/onnxifi_wrapper.c.o: ../third_party/foxi/foxi/onnxifi_wrapper.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object third_party/foxi/CMakeFiles/foxi_wrapper.dir/foxi/onnxifi_wrapper.c.o"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/foxi && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/foxi_wrapper.dir/foxi/onnxifi_wrapper.c.o   -c /home/zzp/code/NoPFS/pytorch/third_party/foxi/foxi/onnxifi_wrapper.c

third_party/foxi/CMakeFiles/foxi_wrapper.dir/foxi/onnxifi_wrapper.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/foxi_wrapper.dir/foxi/onnxifi_wrapper.c.i"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/foxi && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/zzp/code/NoPFS/pytorch/third_party/foxi/foxi/onnxifi_wrapper.c > CMakeFiles/foxi_wrapper.dir/foxi/onnxifi_wrapper.c.i

third_party/foxi/CMakeFiles/foxi_wrapper.dir/foxi/onnxifi_wrapper.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/foxi_wrapper.dir/foxi/onnxifi_wrapper.c.s"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/foxi && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/zzp/code/NoPFS/pytorch/third_party/foxi/foxi/onnxifi_wrapper.c -o CMakeFiles/foxi_wrapper.dir/foxi/onnxifi_wrapper.c.s

third_party/foxi/CMakeFiles/foxi_wrapper.dir/foxi/onnxifi_wrapper.c.o.requires:

.PHONY : third_party/foxi/CMakeFiles/foxi_wrapper.dir/foxi/onnxifi_wrapper.c.o.requires

third_party/foxi/CMakeFiles/foxi_wrapper.dir/foxi/onnxifi_wrapper.c.o.provides: third_party/foxi/CMakeFiles/foxi_wrapper.dir/foxi/onnxifi_wrapper.c.o.requires
	$(MAKE) -f third_party/foxi/CMakeFiles/foxi_wrapper.dir/build.make third_party/foxi/CMakeFiles/foxi_wrapper.dir/foxi/onnxifi_wrapper.c.o.provides.build
.PHONY : third_party/foxi/CMakeFiles/foxi_wrapper.dir/foxi/onnxifi_wrapper.c.o.provides

third_party/foxi/CMakeFiles/foxi_wrapper.dir/foxi/onnxifi_wrapper.c.o.provides.build: third_party/foxi/CMakeFiles/foxi_wrapper.dir/foxi/onnxifi_wrapper.c.o


# Object files for target foxi_wrapper
foxi_wrapper_OBJECTS = \
"CMakeFiles/foxi_wrapper.dir/foxi/onnxifi_wrapper.c.o"

# External object files for target foxi_wrapper
foxi_wrapper_EXTERNAL_OBJECTS =

lib/libfoxi.so: third_party/foxi/CMakeFiles/foxi_wrapper.dir/foxi/onnxifi_wrapper.c.o
lib/libfoxi.so: third_party/foxi/CMakeFiles/foxi_wrapper.dir/build.make
lib/libfoxi.so: lib/libfoxi_loader.a
lib/libfoxi.so: third_party/foxi/CMakeFiles/foxi_wrapper.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C shared module ../../lib/libfoxi.so"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/foxi && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/foxi_wrapper.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
third_party/foxi/CMakeFiles/foxi_wrapper.dir/build: lib/libfoxi.so

.PHONY : third_party/foxi/CMakeFiles/foxi_wrapper.dir/build

third_party/foxi/CMakeFiles/foxi_wrapper.dir/requires: third_party/foxi/CMakeFiles/foxi_wrapper.dir/foxi/onnxifi_wrapper.c.o.requires

.PHONY : third_party/foxi/CMakeFiles/foxi_wrapper.dir/requires

third_party/foxi/CMakeFiles/foxi_wrapper.dir/clean:
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/foxi && $(CMAKE_COMMAND) -P CMakeFiles/foxi_wrapper.dir/cmake_clean.cmake
.PHONY : third_party/foxi/CMakeFiles/foxi_wrapper.dir/clean

third_party/foxi/CMakeFiles/foxi_wrapper.dir/depend:
	cd /home/zzp/code/NoPFS/pytorch/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zzp/code/NoPFS/pytorch /home/zzp/code/NoPFS/pytorch/third_party/foxi /home/zzp/code/NoPFS/pytorch/build /home/zzp/code/NoPFS/pytorch/build/third_party/foxi /home/zzp/code/NoPFS/pytorch/build/third_party/foxi/CMakeFiles/foxi_wrapper.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : third_party/foxi/CMakeFiles/foxi_wrapper.dir/depend
