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
include third_party/protobuf/cmake/CMakeFiles/protoc.dir/depend.make

# Include the progress variables for this target.
include third_party/protobuf/cmake/CMakeFiles/protoc.dir/progress.make

# Include the compile flags for this target's objects.
include third_party/protobuf/cmake/CMakeFiles/protoc.dir/flags.make

third_party/protobuf/cmake/CMakeFiles/protoc.dir/__/src/google/protobuf/compiler/main.cc.o: third_party/protobuf/cmake/CMakeFiles/protoc.dir/flags.make
third_party/protobuf/cmake/CMakeFiles/protoc.dir/__/src/google/protobuf/compiler/main.cc.o: ../third_party/protobuf/src/google/protobuf/compiler/main.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object third_party/protobuf/cmake/CMakeFiles/protoc.dir/__/src/google/protobuf/compiler/main.cc.o"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/protobuf/cmake && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/protoc.dir/__/src/google/protobuf/compiler/main.cc.o -c /home/zzp/code/NoPFS/pytorch/third_party/protobuf/src/google/protobuf/compiler/main.cc

third_party/protobuf/cmake/CMakeFiles/protoc.dir/__/src/google/protobuf/compiler/main.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/protoc.dir/__/src/google/protobuf/compiler/main.cc.i"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/protobuf/cmake && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zzp/code/NoPFS/pytorch/third_party/protobuf/src/google/protobuf/compiler/main.cc > CMakeFiles/protoc.dir/__/src/google/protobuf/compiler/main.cc.i

third_party/protobuf/cmake/CMakeFiles/protoc.dir/__/src/google/protobuf/compiler/main.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/protoc.dir/__/src/google/protobuf/compiler/main.cc.s"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/protobuf/cmake && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zzp/code/NoPFS/pytorch/third_party/protobuf/src/google/protobuf/compiler/main.cc -o CMakeFiles/protoc.dir/__/src/google/protobuf/compiler/main.cc.s

third_party/protobuf/cmake/CMakeFiles/protoc.dir/__/src/google/protobuf/compiler/main.cc.o.requires:

.PHONY : third_party/protobuf/cmake/CMakeFiles/protoc.dir/__/src/google/protobuf/compiler/main.cc.o.requires

third_party/protobuf/cmake/CMakeFiles/protoc.dir/__/src/google/protobuf/compiler/main.cc.o.provides: third_party/protobuf/cmake/CMakeFiles/protoc.dir/__/src/google/protobuf/compiler/main.cc.o.requires
	$(MAKE) -f third_party/protobuf/cmake/CMakeFiles/protoc.dir/build.make third_party/protobuf/cmake/CMakeFiles/protoc.dir/__/src/google/protobuf/compiler/main.cc.o.provides.build
.PHONY : third_party/protobuf/cmake/CMakeFiles/protoc.dir/__/src/google/protobuf/compiler/main.cc.o.provides

third_party/protobuf/cmake/CMakeFiles/protoc.dir/__/src/google/protobuf/compiler/main.cc.o.provides.build: third_party/protobuf/cmake/CMakeFiles/protoc.dir/__/src/google/protobuf/compiler/main.cc.o


# Object files for target protoc
protoc_OBJECTS = \
"CMakeFiles/protoc.dir/__/src/google/protobuf/compiler/main.cc.o"

# External object files for target protoc
protoc_EXTERNAL_OBJECTS =

bin/protoc-3.13.0.0: third_party/protobuf/cmake/CMakeFiles/protoc.dir/__/src/google/protobuf/compiler/main.cc.o
bin/protoc-3.13.0.0: third_party/protobuf/cmake/CMakeFiles/protoc.dir/build.make
bin/protoc-3.13.0.0: lib/libprotoc.a
bin/protoc-3.13.0.0: lib/libprotobuf.a
bin/protoc-3.13.0.0: third_party/protobuf/cmake/CMakeFiles/protoc.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../../bin/protoc"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/protobuf/cmake && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/protoc.dir/link.txt --verbose=$(VERBOSE)
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/protobuf/cmake && $(CMAKE_COMMAND) -E cmake_symlink_executable ../../../bin/protoc-3.13.0.0 ../../../bin/protoc

bin/protoc: bin/protoc-3.13.0.0


# Rule to build all files generated by this target.
third_party/protobuf/cmake/CMakeFiles/protoc.dir/build: bin/protoc

.PHONY : third_party/protobuf/cmake/CMakeFiles/protoc.dir/build

third_party/protobuf/cmake/CMakeFiles/protoc.dir/requires: third_party/protobuf/cmake/CMakeFiles/protoc.dir/__/src/google/protobuf/compiler/main.cc.o.requires

.PHONY : third_party/protobuf/cmake/CMakeFiles/protoc.dir/requires

third_party/protobuf/cmake/CMakeFiles/protoc.dir/clean:
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/protobuf/cmake && $(CMAKE_COMMAND) -P CMakeFiles/protoc.dir/cmake_clean.cmake
.PHONY : third_party/protobuf/cmake/CMakeFiles/protoc.dir/clean

third_party/protobuf/cmake/CMakeFiles/protoc.dir/depend:
	cd /home/zzp/code/NoPFS/pytorch/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zzp/code/NoPFS/pytorch /home/zzp/code/NoPFS/pytorch/third_party/protobuf/cmake /home/zzp/code/NoPFS/pytorch/build /home/zzp/code/NoPFS/pytorch/build/third_party/protobuf/cmake /home/zzp/code/NoPFS/pytorch/build/third_party/protobuf/cmake/CMakeFiles/protoc.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : third_party/protobuf/cmake/CMakeFiles/protoc.dir/depend

