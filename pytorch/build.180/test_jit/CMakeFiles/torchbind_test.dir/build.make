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
include test_jit/CMakeFiles/torchbind_test.dir/depend.make

# Include the progress variables for this target.
include test_jit/CMakeFiles/torchbind_test.dir/progress.make

# Include the compile flags for this target's objects.
include test_jit/CMakeFiles/torchbind_test.dir/flags.make

test_jit/CMakeFiles/torchbind_test.dir/test_custom_class_registrations.cpp.o: test_jit/CMakeFiles/torchbind_test.dir/flags.make
test_jit/CMakeFiles/torchbind_test.dir/test_custom_class_registrations.cpp.o: ../test/cpp/jit/test_custom_class_registrations.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test_jit/CMakeFiles/torchbind_test.dir/test_custom_class_registrations.cpp.o"
	cd /home/zzp/code/NoPFS/pytorch/build/test_jit && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/torchbind_test.dir/test_custom_class_registrations.cpp.o -c /home/zzp/code/NoPFS/pytorch/test/cpp/jit/test_custom_class_registrations.cpp

test_jit/CMakeFiles/torchbind_test.dir/test_custom_class_registrations.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/torchbind_test.dir/test_custom_class_registrations.cpp.i"
	cd /home/zzp/code/NoPFS/pytorch/build/test_jit && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zzp/code/NoPFS/pytorch/test/cpp/jit/test_custom_class_registrations.cpp > CMakeFiles/torchbind_test.dir/test_custom_class_registrations.cpp.i

test_jit/CMakeFiles/torchbind_test.dir/test_custom_class_registrations.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/torchbind_test.dir/test_custom_class_registrations.cpp.s"
	cd /home/zzp/code/NoPFS/pytorch/build/test_jit && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zzp/code/NoPFS/pytorch/test/cpp/jit/test_custom_class_registrations.cpp -o CMakeFiles/torchbind_test.dir/test_custom_class_registrations.cpp.s

test_jit/CMakeFiles/torchbind_test.dir/test_custom_class_registrations.cpp.o.requires:

.PHONY : test_jit/CMakeFiles/torchbind_test.dir/test_custom_class_registrations.cpp.o.requires

test_jit/CMakeFiles/torchbind_test.dir/test_custom_class_registrations.cpp.o.provides: test_jit/CMakeFiles/torchbind_test.dir/test_custom_class_registrations.cpp.o.requires
	$(MAKE) -f test_jit/CMakeFiles/torchbind_test.dir/build.make test_jit/CMakeFiles/torchbind_test.dir/test_custom_class_registrations.cpp.o.provides.build
.PHONY : test_jit/CMakeFiles/torchbind_test.dir/test_custom_class_registrations.cpp.o.provides

test_jit/CMakeFiles/torchbind_test.dir/test_custom_class_registrations.cpp.o.provides.build: test_jit/CMakeFiles/torchbind_test.dir/test_custom_class_registrations.cpp.o


# Object files for target torchbind_test
torchbind_test_OBJECTS = \
"CMakeFiles/torchbind_test.dir/test_custom_class_registrations.cpp.o"

# External object files for target torchbind_test
torchbind_test_EXTERNAL_OBJECTS =

lib/libtorchbind_test.so: test_jit/CMakeFiles/torchbind_test.dir/test_custom_class_registrations.cpp.o
lib/libtorchbind_test.so: test_jit/CMakeFiles/torchbind_test.dir/build.make
lib/libtorchbind_test.so: lib/libtorch.so
lib/libtorchbind_test.so: lib/libprotobuf.a
lib/libtorchbind_test.so: lib/libdnnl.a
lib/libtorchbind_test.so: lib/libc10_cuda.so
lib/libtorchbind_test.so: lib/libc10.so
lib/libtorchbind_test.so: /usr/local/cuda/lib64/libcudart.so
lib/libtorchbind_test.so: /usr/local/cuda/lib64/libnvToolsExt.so
lib/libtorchbind_test.so: /usr/local/cuda/lib64/libcufft.so
lib/libtorchbind_test.so: /usr/local/cuda/lib64/libcurand.so
lib/libtorchbind_test.so: /usr/local/cuda/lib64/libcublas.so
lib/libtorchbind_test.so: /usr/lib/x86_64-linux-gnu/libcudnn.so
lib/libtorchbind_test.so: test_jit/CMakeFiles/torchbind_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library ../lib/libtorchbind_test.so"
	cd /home/zzp/code/NoPFS/pytorch/build/test_jit && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/torchbind_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test_jit/CMakeFiles/torchbind_test.dir/build: lib/libtorchbind_test.so

.PHONY : test_jit/CMakeFiles/torchbind_test.dir/build

test_jit/CMakeFiles/torchbind_test.dir/requires: test_jit/CMakeFiles/torchbind_test.dir/test_custom_class_registrations.cpp.o.requires

.PHONY : test_jit/CMakeFiles/torchbind_test.dir/requires

test_jit/CMakeFiles/torchbind_test.dir/clean:
	cd /home/zzp/code/NoPFS/pytorch/build/test_jit && $(CMAKE_COMMAND) -P CMakeFiles/torchbind_test.dir/cmake_clean.cmake
.PHONY : test_jit/CMakeFiles/torchbind_test.dir/clean

test_jit/CMakeFiles/torchbind_test.dir/depend:
	cd /home/zzp/code/NoPFS/pytorch/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zzp/code/NoPFS/pytorch /home/zzp/code/NoPFS/pytorch/test/cpp/jit /home/zzp/code/NoPFS/pytorch/build /home/zzp/code/NoPFS/pytorch/build/test_jit /home/zzp/code/NoPFS/pytorch/build/test_jit/CMakeFiles/torchbind_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test_jit/CMakeFiles/torchbind_test.dir/depend

