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
include caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/depend.make

# Include the progress variables for this target.
include caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/progress.make

# Include the compile flags for this target's objects.
include caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/flags.make

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/adagrad_avx2.cc.o: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/flags.make
caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/adagrad_avx2.cc.o: ../caffe2/perfkernels/adagrad_avx2.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/adagrad_avx2.cc.o"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2/perfkernels && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Caffe2_perfkernels_avx2.dir/adagrad_avx2.cc.o -c /home/zzp/code/NoPFS/pytorch/caffe2/perfkernels/adagrad_avx2.cc

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/adagrad_avx2.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Caffe2_perfkernels_avx2.dir/adagrad_avx2.cc.i"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2/perfkernels && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zzp/code/NoPFS/pytorch/caffe2/perfkernels/adagrad_avx2.cc > CMakeFiles/Caffe2_perfkernels_avx2.dir/adagrad_avx2.cc.i

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/adagrad_avx2.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Caffe2_perfkernels_avx2.dir/adagrad_avx2.cc.s"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2/perfkernels && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zzp/code/NoPFS/pytorch/caffe2/perfkernels/adagrad_avx2.cc -o CMakeFiles/Caffe2_perfkernels_avx2.dir/adagrad_avx2.cc.s

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/adagrad_avx2.cc.o.requires:

.PHONY : caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/adagrad_avx2.cc.o.requires

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/adagrad_avx2.cc.o.provides: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/adagrad_avx2.cc.o.requires
	$(MAKE) -f caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/build.make caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/adagrad_avx2.cc.o.provides.build
.PHONY : caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/adagrad_avx2.cc.o.provides

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/adagrad_avx2.cc.o.provides.build: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/adagrad_avx2.cc.o


caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/common_avx2.cc.o: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/flags.make
caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/common_avx2.cc.o: ../caffe2/perfkernels/common_avx2.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/common_avx2.cc.o"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2/perfkernels && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Caffe2_perfkernels_avx2.dir/common_avx2.cc.o -c /home/zzp/code/NoPFS/pytorch/caffe2/perfkernels/common_avx2.cc

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/common_avx2.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Caffe2_perfkernels_avx2.dir/common_avx2.cc.i"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2/perfkernels && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zzp/code/NoPFS/pytorch/caffe2/perfkernels/common_avx2.cc > CMakeFiles/Caffe2_perfkernels_avx2.dir/common_avx2.cc.i

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/common_avx2.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Caffe2_perfkernels_avx2.dir/common_avx2.cc.s"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2/perfkernels && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zzp/code/NoPFS/pytorch/caffe2/perfkernels/common_avx2.cc -o CMakeFiles/Caffe2_perfkernels_avx2.dir/common_avx2.cc.s

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/common_avx2.cc.o.requires:

.PHONY : caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/common_avx2.cc.o.requires

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/common_avx2.cc.o.provides: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/common_avx2.cc.o.requires
	$(MAKE) -f caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/build.make caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/common_avx2.cc.o.provides.build
.PHONY : caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/common_avx2.cc.o.provides

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/common_avx2.cc.o.provides.build: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/common_avx2.cc.o


caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_avx2.cc.o: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/flags.make
caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_avx2.cc.o: ../caffe2/perfkernels/embedding_lookup_avx2.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_avx2.cc.o"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2/perfkernels && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_avx2.cc.o -c /home/zzp/code/NoPFS/pytorch/caffe2/perfkernels/embedding_lookup_avx2.cc

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_avx2.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_avx2.cc.i"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2/perfkernels && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zzp/code/NoPFS/pytorch/caffe2/perfkernels/embedding_lookup_avx2.cc > CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_avx2.cc.i

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_avx2.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_avx2.cc.s"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2/perfkernels && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zzp/code/NoPFS/pytorch/caffe2/perfkernels/embedding_lookup_avx2.cc -o CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_avx2.cc.s

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_avx2.cc.o.requires:

.PHONY : caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_avx2.cc.o.requires

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_avx2.cc.o.provides: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_avx2.cc.o.requires
	$(MAKE) -f caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/build.make caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_avx2.cc.o.provides.build
.PHONY : caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_avx2.cc.o.provides

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_avx2.cc.o.provides.build: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_avx2.cc.o


caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_avx2.cc.o: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/flags.make
caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_avx2.cc.o: ../caffe2/perfkernels/embedding_lookup_fused_8bit_rowwise_avx2.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_avx2.cc.o"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2/perfkernels && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_avx2.cc.o -c /home/zzp/code/NoPFS/pytorch/caffe2/perfkernels/embedding_lookup_fused_8bit_rowwise_avx2.cc

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_avx2.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_avx2.cc.i"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2/perfkernels && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zzp/code/NoPFS/pytorch/caffe2/perfkernels/embedding_lookup_fused_8bit_rowwise_avx2.cc > CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_avx2.cc.i

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_avx2.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_avx2.cc.s"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2/perfkernels && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zzp/code/NoPFS/pytorch/caffe2/perfkernels/embedding_lookup_fused_8bit_rowwise_avx2.cc -o CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_avx2.cc.s

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_avx2.cc.o.requires:

.PHONY : caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_avx2.cc.o.requires

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_avx2.cc.o.provides: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_avx2.cc.o.requires
	$(MAKE) -f caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/build.make caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_avx2.cc.o.provides.build
.PHONY : caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_avx2.cc.o.provides

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_avx2.cc.o.provides.build: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_avx2.cc.o


caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_idx_avx2.cc.o: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/flags.make
caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_idx_avx2.cc.o: ../caffe2/perfkernels/embedding_lookup_fused_8bit_rowwise_idx_avx2.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_idx_avx2.cc.o"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2/perfkernels && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_idx_avx2.cc.o -c /home/zzp/code/NoPFS/pytorch/caffe2/perfkernels/embedding_lookup_fused_8bit_rowwise_idx_avx2.cc

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_idx_avx2.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_idx_avx2.cc.i"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2/perfkernels && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zzp/code/NoPFS/pytorch/caffe2/perfkernels/embedding_lookup_fused_8bit_rowwise_idx_avx2.cc > CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_idx_avx2.cc.i

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_idx_avx2.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_idx_avx2.cc.s"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2/perfkernels && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zzp/code/NoPFS/pytorch/caffe2/perfkernels/embedding_lookup_fused_8bit_rowwise_idx_avx2.cc -o CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_idx_avx2.cc.s

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_idx_avx2.cc.o.requires:

.PHONY : caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_idx_avx2.cc.o.requires

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_idx_avx2.cc.o.provides: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_idx_avx2.cc.o.requires
	$(MAKE) -f caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/build.make caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_idx_avx2.cc.o.provides.build
.PHONY : caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_idx_avx2.cc.o.provides

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_idx_avx2.cc.o.provides.build: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_idx_avx2.cc.o


caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_idx_avx2.cc.o: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/flags.make
caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_idx_avx2.cc.o: ../caffe2/perfkernels/embedding_lookup_idx_avx2.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_idx_avx2.cc.o"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2/perfkernels && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_idx_avx2.cc.o -c /home/zzp/code/NoPFS/pytorch/caffe2/perfkernels/embedding_lookup_idx_avx2.cc

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_idx_avx2.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_idx_avx2.cc.i"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2/perfkernels && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zzp/code/NoPFS/pytorch/caffe2/perfkernels/embedding_lookup_idx_avx2.cc > CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_idx_avx2.cc.i

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_idx_avx2.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_idx_avx2.cc.s"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2/perfkernels && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zzp/code/NoPFS/pytorch/caffe2/perfkernels/embedding_lookup_idx_avx2.cc -o CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_idx_avx2.cc.s

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_idx_avx2.cc.o.requires:

.PHONY : caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_idx_avx2.cc.o.requires

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_idx_avx2.cc.o.provides: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_idx_avx2.cc.o.requires
	$(MAKE) -f caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/build.make caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_idx_avx2.cc.o.provides.build
.PHONY : caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_idx_avx2.cc.o.provides

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_idx_avx2.cc.o.provides.build: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_idx_avx2.cc.o


caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/lstm_unit_cpu_avx2.cc.o: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/flags.make
caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/lstm_unit_cpu_avx2.cc.o: ../caffe2/perfkernels/lstm_unit_cpu_avx2.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/lstm_unit_cpu_avx2.cc.o"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2/perfkernels && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Caffe2_perfkernels_avx2.dir/lstm_unit_cpu_avx2.cc.o -c /home/zzp/code/NoPFS/pytorch/caffe2/perfkernels/lstm_unit_cpu_avx2.cc

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/lstm_unit_cpu_avx2.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Caffe2_perfkernels_avx2.dir/lstm_unit_cpu_avx2.cc.i"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2/perfkernels && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zzp/code/NoPFS/pytorch/caffe2/perfkernels/lstm_unit_cpu_avx2.cc > CMakeFiles/Caffe2_perfkernels_avx2.dir/lstm_unit_cpu_avx2.cc.i

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/lstm_unit_cpu_avx2.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Caffe2_perfkernels_avx2.dir/lstm_unit_cpu_avx2.cc.s"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2/perfkernels && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zzp/code/NoPFS/pytorch/caffe2/perfkernels/lstm_unit_cpu_avx2.cc -o CMakeFiles/Caffe2_perfkernels_avx2.dir/lstm_unit_cpu_avx2.cc.s

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/lstm_unit_cpu_avx2.cc.o.requires:

.PHONY : caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/lstm_unit_cpu_avx2.cc.o.requires

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/lstm_unit_cpu_avx2.cc.o.provides: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/lstm_unit_cpu_avx2.cc.o.requires
	$(MAKE) -f caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/build.make caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/lstm_unit_cpu_avx2.cc.o.provides.build
.PHONY : caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/lstm_unit_cpu_avx2.cc.o.provides

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/lstm_unit_cpu_avx2.cc.o.provides.build: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/lstm_unit_cpu_avx2.cc.o


caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/math_cpu_avx2.cc.o: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/flags.make
caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/math_cpu_avx2.cc.o: ../caffe2/perfkernels/math_cpu_avx2.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/math_cpu_avx2.cc.o"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2/perfkernels && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Caffe2_perfkernels_avx2.dir/math_cpu_avx2.cc.o -c /home/zzp/code/NoPFS/pytorch/caffe2/perfkernels/math_cpu_avx2.cc

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/math_cpu_avx2.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Caffe2_perfkernels_avx2.dir/math_cpu_avx2.cc.i"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2/perfkernels && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zzp/code/NoPFS/pytorch/caffe2/perfkernels/math_cpu_avx2.cc > CMakeFiles/Caffe2_perfkernels_avx2.dir/math_cpu_avx2.cc.i

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/math_cpu_avx2.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Caffe2_perfkernels_avx2.dir/math_cpu_avx2.cc.s"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2/perfkernels && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zzp/code/NoPFS/pytorch/caffe2/perfkernels/math_cpu_avx2.cc -o CMakeFiles/Caffe2_perfkernels_avx2.dir/math_cpu_avx2.cc.s

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/math_cpu_avx2.cc.o.requires:

.PHONY : caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/math_cpu_avx2.cc.o.requires

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/math_cpu_avx2.cc.o.provides: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/math_cpu_avx2.cc.o.requires
	$(MAKE) -f caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/build.make caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/math_cpu_avx2.cc.o.provides.build
.PHONY : caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/math_cpu_avx2.cc.o.provides

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/math_cpu_avx2.cc.o.provides.build: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/math_cpu_avx2.cc.o


caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/typed_axpy_avx2.cc.o: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/flags.make
caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/typed_axpy_avx2.cc.o: ../caffe2/perfkernels/typed_axpy_avx2.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/typed_axpy_avx2.cc.o"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2/perfkernels && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Caffe2_perfkernels_avx2.dir/typed_axpy_avx2.cc.o -c /home/zzp/code/NoPFS/pytorch/caffe2/perfkernels/typed_axpy_avx2.cc

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/typed_axpy_avx2.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Caffe2_perfkernels_avx2.dir/typed_axpy_avx2.cc.i"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2/perfkernels && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zzp/code/NoPFS/pytorch/caffe2/perfkernels/typed_axpy_avx2.cc > CMakeFiles/Caffe2_perfkernels_avx2.dir/typed_axpy_avx2.cc.i

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/typed_axpy_avx2.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Caffe2_perfkernels_avx2.dir/typed_axpy_avx2.cc.s"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2/perfkernels && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zzp/code/NoPFS/pytorch/caffe2/perfkernels/typed_axpy_avx2.cc -o CMakeFiles/Caffe2_perfkernels_avx2.dir/typed_axpy_avx2.cc.s

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/typed_axpy_avx2.cc.o.requires:

.PHONY : caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/typed_axpy_avx2.cc.o.requires

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/typed_axpy_avx2.cc.o.provides: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/typed_axpy_avx2.cc.o.requires
	$(MAKE) -f caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/build.make caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/typed_axpy_avx2.cc.o.provides.build
.PHONY : caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/typed_axpy_avx2.cc.o.provides

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/typed_axpy_avx2.cc.o.provides.build: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/typed_axpy_avx2.cc.o


# Object files for target Caffe2_perfkernels_avx2
Caffe2_perfkernels_avx2_OBJECTS = \
"CMakeFiles/Caffe2_perfkernels_avx2.dir/adagrad_avx2.cc.o" \
"CMakeFiles/Caffe2_perfkernels_avx2.dir/common_avx2.cc.o" \
"CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_avx2.cc.o" \
"CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_avx2.cc.o" \
"CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_idx_avx2.cc.o" \
"CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_idx_avx2.cc.o" \
"CMakeFiles/Caffe2_perfkernels_avx2.dir/lstm_unit_cpu_avx2.cc.o" \
"CMakeFiles/Caffe2_perfkernels_avx2.dir/math_cpu_avx2.cc.o" \
"CMakeFiles/Caffe2_perfkernels_avx2.dir/typed_axpy_avx2.cc.o"

# External object files for target Caffe2_perfkernels_avx2
Caffe2_perfkernels_avx2_EXTERNAL_OBJECTS =

lib/libCaffe2_perfkernels_avx2.a: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/adagrad_avx2.cc.o
lib/libCaffe2_perfkernels_avx2.a: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/common_avx2.cc.o
lib/libCaffe2_perfkernels_avx2.a: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_avx2.cc.o
lib/libCaffe2_perfkernels_avx2.a: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_avx2.cc.o
lib/libCaffe2_perfkernels_avx2.a: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_idx_avx2.cc.o
lib/libCaffe2_perfkernels_avx2.a: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_idx_avx2.cc.o
lib/libCaffe2_perfkernels_avx2.a: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/lstm_unit_cpu_avx2.cc.o
lib/libCaffe2_perfkernels_avx2.a: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/math_cpu_avx2.cc.o
lib/libCaffe2_perfkernels_avx2.a: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/typed_axpy_avx2.cc.o
lib/libCaffe2_perfkernels_avx2.a: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/build.make
lib/libCaffe2_perfkernels_avx2.a: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Linking CXX static library ../../lib/libCaffe2_perfkernels_avx2.a"
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2/perfkernels && $(CMAKE_COMMAND) -P CMakeFiles/Caffe2_perfkernels_avx2.dir/cmake_clean_target.cmake
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2/perfkernels && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Caffe2_perfkernels_avx2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/build: lib/libCaffe2_perfkernels_avx2.a

.PHONY : caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/build

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/requires: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/adagrad_avx2.cc.o.requires
caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/requires: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/common_avx2.cc.o.requires
caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/requires: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_avx2.cc.o.requires
caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/requires: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_avx2.cc.o.requires
caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/requires: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_fused_8bit_rowwise_idx_avx2.cc.o.requires
caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/requires: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/embedding_lookup_idx_avx2.cc.o.requires
caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/requires: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/lstm_unit_cpu_avx2.cc.o.requires
caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/requires: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/math_cpu_avx2.cc.o.requires
caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/requires: caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/typed_axpy_avx2.cc.o.requires

.PHONY : caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/requires

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/clean:
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2/perfkernels && $(CMAKE_COMMAND) -P CMakeFiles/Caffe2_perfkernels_avx2.dir/cmake_clean.cmake
.PHONY : caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/clean

caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/depend:
	cd /home/zzp/code/NoPFS/pytorch/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zzp/code/NoPFS/pytorch /home/zzp/code/NoPFS/pytorch/caffe2/perfkernels /home/zzp/code/NoPFS/pytorch/build /home/zzp/code/NoPFS/pytorch/build/caffe2/perfkernels /home/zzp/code/NoPFS/pytorch/build/caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : caffe2/perfkernels/CMakeFiles/Caffe2_perfkernels_avx2.dir/depend

