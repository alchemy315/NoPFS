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
include c10/test/CMakeFiles/c10_ordered_preserving_dict_test.dir/depend.make

# Include the progress variables for this target.
include c10/test/CMakeFiles/c10_ordered_preserving_dict_test.dir/progress.make

# Include the compile flags for this target's objects.
include c10/test/CMakeFiles/c10_ordered_preserving_dict_test.dir/flags.make

c10/test/CMakeFiles/c10_ordered_preserving_dict_test.dir/util/ordered_preserving_dict_test.cpp.o: c10/test/CMakeFiles/c10_ordered_preserving_dict_test.dir/flags.make
c10/test/CMakeFiles/c10_ordered_preserving_dict_test.dir/util/ordered_preserving_dict_test.cpp.o: ../c10/test/util/ordered_preserving_dict_test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object c10/test/CMakeFiles/c10_ordered_preserving_dict_test.dir/util/ordered_preserving_dict_test.cpp.o"
	cd /home/zzp/code/NoPFS/pytorch/build/c10/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/c10_ordered_preserving_dict_test.dir/util/ordered_preserving_dict_test.cpp.o -c /home/zzp/code/NoPFS/pytorch/c10/test/util/ordered_preserving_dict_test.cpp

c10/test/CMakeFiles/c10_ordered_preserving_dict_test.dir/util/ordered_preserving_dict_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/c10_ordered_preserving_dict_test.dir/util/ordered_preserving_dict_test.cpp.i"
	cd /home/zzp/code/NoPFS/pytorch/build/c10/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zzp/code/NoPFS/pytorch/c10/test/util/ordered_preserving_dict_test.cpp > CMakeFiles/c10_ordered_preserving_dict_test.dir/util/ordered_preserving_dict_test.cpp.i

c10/test/CMakeFiles/c10_ordered_preserving_dict_test.dir/util/ordered_preserving_dict_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/c10_ordered_preserving_dict_test.dir/util/ordered_preserving_dict_test.cpp.s"
	cd /home/zzp/code/NoPFS/pytorch/build/c10/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zzp/code/NoPFS/pytorch/c10/test/util/ordered_preserving_dict_test.cpp -o CMakeFiles/c10_ordered_preserving_dict_test.dir/util/ordered_preserving_dict_test.cpp.s

c10/test/CMakeFiles/c10_ordered_preserving_dict_test.dir/util/ordered_preserving_dict_test.cpp.o.requires:

.PHONY : c10/test/CMakeFiles/c10_ordered_preserving_dict_test.dir/util/ordered_preserving_dict_test.cpp.o.requires

c10/test/CMakeFiles/c10_ordered_preserving_dict_test.dir/util/ordered_preserving_dict_test.cpp.o.provides: c10/test/CMakeFiles/c10_ordered_preserving_dict_test.dir/util/ordered_preserving_dict_test.cpp.o.requires
	$(MAKE) -f c10/test/CMakeFiles/c10_ordered_preserving_dict_test.dir/build.make c10/test/CMakeFiles/c10_ordered_preserving_dict_test.dir/util/ordered_preserving_dict_test.cpp.o.provides.build
.PHONY : c10/test/CMakeFiles/c10_ordered_preserving_dict_test.dir/util/ordered_preserving_dict_test.cpp.o.provides

c10/test/CMakeFiles/c10_ordered_preserving_dict_test.dir/util/ordered_preserving_dict_test.cpp.o.provides.build: c10/test/CMakeFiles/c10_ordered_preserving_dict_test.dir/util/ordered_preserving_dict_test.cpp.o


# Object files for target c10_ordered_preserving_dict_test
c10_ordered_preserving_dict_test_OBJECTS = \
"CMakeFiles/c10_ordered_preserving_dict_test.dir/util/ordered_preserving_dict_test.cpp.o"

# External object files for target c10_ordered_preserving_dict_test
c10_ordered_preserving_dict_test_EXTERNAL_OBJECTS =

bin/c10_ordered_preserving_dict_test: c10/test/CMakeFiles/c10_ordered_preserving_dict_test.dir/util/ordered_preserving_dict_test.cpp.o
bin/c10_ordered_preserving_dict_test: c10/test/CMakeFiles/c10_ordered_preserving_dict_test.dir/build.make
bin/c10_ordered_preserving_dict_test: lib/libc10.so
bin/c10_ordered_preserving_dict_test: lib/libgmock.a
bin/c10_ordered_preserving_dict_test: lib/libgtest.a
bin/c10_ordered_preserving_dict_test: lib/libgtest_main.a
bin/c10_ordered_preserving_dict_test: lib/libgtest.a
bin/c10_ordered_preserving_dict_test: c10/test/CMakeFiles/c10_ordered_preserving_dict_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/c10_ordered_preserving_dict_test"
	cd /home/zzp/code/NoPFS/pytorch/build/c10/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/c10_ordered_preserving_dict_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
c10/test/CMakeFiles/c10_ordered_preserving_dict_test.dir/build: bin/c10_ordered_preserving_dict_test

.PHONY : c10/test/CMakeFiles/c10_ordered_preserving_dict_test.dir/build

c10/test/CMakeFiles/c10_ordered_preserving_dict_test.dir/requires: c10/test/CMakeFiles/c10_ordered_preserving_dict_test.dir/util/ordered_preserving_dict_test.cpp.o.requires

.PHONY : c10/test/CMakeFiles/c10_ordered_preserving_dict_test.dir/requires

c10/test/CMakeFiles/c10_ordered_preserving_dict_test.dir/clean:
	cd /home/zzp/code/NoPFS/pytorch/build/c10/test && $(CMAKE_COMMAND) -P CMakeFiles/c10_ordered_preserving_dict_test.dir/cmake_clean.cmake
.PHONY : c10/test/CMakeFiles/c10_ordered_preserving_dict_test.dir/clean

c10/test/CMakeFiles/c10_ordered_preserving_dict_test.dir/depend:
	cd /home/zzp/code/NoPFS/pytorch/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zzp/code/NoPFS/pytorch /home/zzp/code/NoPFS/pytorch/c10/test /home/zzp/code/NoPFS/pytorch/build /home/zzp/code/NoPFS/pytorch/build/c10/test /home/zzp/code/NoPFS/pytorch/build/c10/test/CMakeFiles/c10_ordered_preserving_dict_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : c10/test/CMakeFiles/c10_ordered_preserving_dict_test.dir/depend

