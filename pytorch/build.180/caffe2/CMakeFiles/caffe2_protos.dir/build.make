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
include caffe2/CMakeFiles/caffe2_protos.dir/depend.make

# Include the progress variables for this target.
include caffe2/CMakeFiles/caffe2_protos.dir/progress.make

# Include the compile flags for this target's objects.
include caffe2/CMakeFiles/caffe2_protos.dir/flags.make

# Object files for target caffe2_protos
caffe2_protos_OBJECTS =

# External object files for target caffe2_protos
caffe2_protos_EXTERNAL_OBJECTS = \
"/home/zzp/code/NoPFS/pytorch/build/caffe2/proto/CMakeFiles/Caffe2_PROTO.dir/caffe2.pb.cc.o" \
"/home/zzp/code/NoPFS/pytorch/build/caffe2/proto/CMakeFiles/Caffe2_PROTO.dir/caffe2_legacy.pb.cc.o" \
"/home/zzp/code/NoPFS/pytorch/build/caffe2/proto/CMakeFiles/Caffe2_PROTO.dir/hsm.pb.cc.o" \
"/home/zzp/code/NoPFS/pytorch/build/caffe2/proto/CMakeFiles/Caffe2_PROTO.dir/metanet.pb.cc.o" \
"/home/zzp/code/NoPFS/pytorch/build/caffe2/proto/CMakeFiles/Caffe2_PROTO.dir/predictor_consts.pb.cc.o" \
"/home/zzp/code/NoPFS/pytorch/build/caffe2/proto/CMakeFiles/Caffe2_PROTO.dir/prof_dag.pb.cc.o" \
"/home/zzp/code/NoPFS/pytorch/build/caffe2/proto/CMakeFiles/Caffe2_PROTO.dir/torch.pb.cc.o"

lib/libcaffe2_protos.a: caffe2/proto/CMakeFiles/Caffe2_PROTO.dir/caffe2.pb.cc.o
lib/libcaffe2_protos.a: caffe2/proto/CMakeFiles/Caffe2_PROTO.dir/caffe2_legacy.pb.cc.o
lib/libcaffe2_protos.a: caffe2/proto/CMakeFiles/Caffe2_PROTO.dir/hsm.pb.cc.o
lib/libcaffe2_protos.a: caffe2/proto/CMakeFiles/Caffe2_PROTO.dir/metanet.pb.cc.o
lib/libcaffe2_protos.a: caffe2/proto/CMakeFiles/Caffe2_PROTO.dir/predictor_consts.pb.cc.o
lib/libcaffe2_protos.a: caffe2/proto/CMakeFiles/Caffe2_PROTO.dir/prof_dag.pb.cc.o
lib/libcaffe2_protos.a: caffe2/proto/CMakeFiles/Caffe2_PROTO.dir/torch.pb.cc.o
lib/libcaffe2_protos.a: caffe2/CMakeFiles/caffe2_protos.dir/build.make
lib/libcaffe2_protos.a: caffe2/CMakeFiles/caffe2_protos.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Linking CXX static library ../lib/libcaffe2_protos.a"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2 && $(CMAKE_COMMAND) -P CMakeFiles/caffe2_protos.dir/cmake_clean_target.cmake
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2 && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/caffe2_protos.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
caffe2/CMakeFiles/caffe2_protos.dir/build: lib/libcaffe2_protos.a

.PHONY : caffe2/CMakeFiles/caffe2_protos.dir/build

caffe2/CMakeFiles/caffe2_protos.dir/requires:

.PHONY : caffe2/CMakeFiles/caffe2_protos.dir/requires

caffe2/CMakeFiles/caffe2_protos.dir/clean:
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2 && $(CMAKE_COMMAND) -P CMakeFiles/caffe2_protos.dir/cmake_clean.cmake
.PHONY : caffe2/CMakeFiles/caffe2_protos.dir/clean

caffe2/CMakeFiles/caffe2_protos.dir/depend:
	cd /home/zzp/code/NoPFS/pytorch/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zzp/code/NoPFS/pytorch /home/zzp/code/NoPFS/pytorch/caffe2 /home/zzp/code/NoPFS/pytorch/build /home/zzp/code/NoPFS/pytorch/build/caffe2 /home/zzp/code/NoPFS/pytorch/build/caffe2/CMakeFiles/caffe2_protos.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : caffe2/CMakeFiles/caffe2_protos.dir/depend

