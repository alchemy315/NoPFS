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
include third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/depend.make

# Include the progress variables for this target.
include third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/progress.make

# Include the compile flags for this target's objects.
include third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/flags.make

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmBfloat16ConvertAvx512.cc.o: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/flags.make
third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmBfloat16ConvertAvx512.cc.o: ../third_party/fbgemm/src/FbgemmBfloat16ConvertAvx512.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmBfloat16ConvertAvx512.cc.o"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fbgemm_avx512.dir/src/FbgemmBfloat16ConvertAvx512.cc.o -c /home/zzp/code/NoPFS/pytorch/third_party/fbgemm/src/FbgemmBfloat16ConvertAvx512.cc

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmBfloat16ConvertAvx512.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fbgemm_avx512.dir/src/FbgemmBfloat16ConvertAvx512.cc.i"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zzp/code/NoPFS/pytorch/third_party/fbgemm/src/FbgemmBfloat16ConvertAvx512.cc > CMakeFiles/fbgemm_avx512.dir/src/FbgemmBfloat16ConvertAvx512.cc.i

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmBfloat16ConvertAvx512.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fbgemm_avx512.dir/src/FbgemmBfloat16ConvertAvx512.cc.s"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zzp/code/NoPFS/pytorch/third_party/fbgemm/src/FbgemmBfloat16ConvertAvx512.cc -o CMakeFiles/fbgemm_avx512.dir/src/FbgemmBfloat16ConvertAvx512.cc.s

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmBfloat16ConvertAvx512.cc.o.requires:

.PHONY : third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmBfloat16ConvertAvx512.cc.o.requires

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmBfloat16ConvertAvx512.cc.o.provides: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmBfloat16ConvertAvx512.cc.o.requires
	$(MAKE) -f third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/build.make third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmBfloat16ConvertAvx512.cc.o.provides.build
.PHONY : third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmBfloat16ConvertAvx512.cc.o.provides

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmBfloat16ConvertAvx512.cc.o.provides.build: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmBfloat16ConvertAvx512.cc.o


third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/EmbeddingSpMDMAvx512.cc.o: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/flags.make
third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/EmbeddingSpMDMAvx512.cc.o: ../third_party/fbgemm/src/EmbeddingSpMDMAvx512.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/EmbeddingSpMDMAvx512.cc.o"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fbgemm_avx512.dir/src/EmbeddingSpMDMAvx512.cc.o -c /home/zzp/code/NoPFS/pytorch/third_party/fbgemm/src/EmbeddingSpMDMAvx512.cc

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/EmbeddingSpMDMAvx512.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fbgemm_avx512.dir/src/EmbeddingSpMDMAvx512.cc.i"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zzp/code/NoPFS/pytorch/third_party/fbgemm/src/EmbeddingSpMDMAvx512.cc > CMakeFiles/fbgemm_avx512.dir/src/EmbeddingSpMDMAvx512.cc.i

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/EmbeddingSpMDMAvx512.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fbgemm_avx512.dir/src/EmbeddingSpMDMAvx512.cc.s"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zzp/code/NoPFS/pytorch/third_party/fbgemm/src/EmbeddingSpMDMAvx512.cc -o CMakeFiles/fbgemm_avx512.dir/src/EmbeddingSpMDMAvx512.cc.s

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/EmbeddingSpMDMAvx512.cc.o.requires:

.PHONY : third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/EmbeddingSpMDMAvx512.cc.o.requires

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/EmbeddingSpMDMAvx512.cc.o.provides: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/EmbeddingSpMDMAvx512.cc.o.requires
	$(MAKE) -f third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/build.make third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/EmbeddingSpMDMAvx512.cc.o.provides.build
.PHONY : third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/EmbeddingSpMDMAvx512.cc.o.provides

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/EmbeddingSpMDMAvx512.cc.o.provides.build: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/EmbeddingSpMDMAvx512.cc.o


third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFloat16ConvertAvx512.cc.o: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/flags.make
third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFloat16ConvertAvx512.cc.o: ../third_party/fbgemm/src/FbgemmFloat16ConvertAvx512.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFloat16ConvertAvx512.cc.o"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fbgemm_avx512.dir/src/FbgemmFloat16ConvertAvx512.cc.o -c /home/zzp/code/NoPFS/pytorch/third_party/fbgemm/src/FbgemmFloat16ConvertAvx512.cc

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFloat16ConvertAvx512.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fbgemm_avx512.dir/src/FbgemmFloat16ConvertAvx512.cc.i"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zzp/code/NoPFS/pytorch/third_party/fbgemm/src/FbgemmFloat16ConvertAvx512.cc > CMakeFiles/fbgemm_avx512.dir/src/FbgemmFloat16ConvertAvx512.cc.i

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFloat16ConvertAvx512.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fbgemm_avx512.dir/src/FbgemmFloat16ConvertAvx512.cc.s"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zzp/code/NoPFS/pytorch/third_party/fbgemm/src/FbgemmFloat16ConvertAvx512.cc -o CMakeFiles/fbgemm_avx512.dir/src/FbgemmFloat16ConvertAvx512.cc.s

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFloat16ConvertAvx512.cc.o.requires:

.PHONY : third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFloat16ConvertAvx512.cc.o.requires

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFloat16ConvertAvx512.cc.o.provides: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFloat16ConvertAvx512.cc.o.requires
	$(MAKE) -f third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/build.make third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFloat16ConvertAvx512.cc.o.provides.build
.PHONY : third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFloat16ConvertAvx512.cc.o.provides

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFloat16ConvertAvx512.cc.o.provides.build: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFloat16ConvertAvx512.cc.o


third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseAvx512.cc.o: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/flags.make
third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseAvx512.cc.o: ../third_party/fbgemm/src/FbgemmSparseDenseAvx512.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseAvx512.cc.o"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseAvx512.cc.o -c /home/zzp/code/NoPFS/pytorch/third_party/fbgemm/src/FbgemmSparseDenseAvx512.cc

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseAvx512.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseAvx512.cc.i"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zzp/code/NoPFS/pytorch/third_party/fbgemm/src/FbgemmSparseDenseAvx512.cc > CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseAvx512.cc.i

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseAvx512.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseAvx512.cc.s"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zzp/code/NoPFS/pytorch/third_party/fbgemm/src/FbgemmSparseDenseAvx512.cc -o CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseAvx512.cc.s

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseAvx512.cc.o.requires:

.PHONY : third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseAvx512.cc.o.requires

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseAvx512.cc.o.provides: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseAvx512.cc.o.requires
	$(MAKE) -f third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/build.make third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseAvx512.cc.o.provides.build
.PHONY : third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseAvx512.cc.o.provides

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseAvx512.cc.o.provides.build: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseAvx512.cc.o


third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseInt8Avx512.cc.o: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/flags.make
third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseInt8Avx512.cc.o: ../third_party/fbgemm/src/FbgemmSparseDenseInt8Avx512.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseInt8Avx512.cc.o"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseInt8Avx512.cc.o -c /home/zzp/code/NoPFS/pytorch/third_party/fbgemm/src/FbgemmSparseDenseInt8Avx512.cc

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseInt8Avx512.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseInt8Avx512.cc.i"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zzp/code/NoPFS/pytorch/third_party/fbgemm/src/FbgemmSparseDenseInt8Avx512.cc > CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseInt8Avx512.cc.i

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseInt8Avx512.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseInt8Avx512.cc.s"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zzp/code/NoPFS/pytorch/third_party/fbgemm/src/FbgemmSparseDenseInt8Avx512.cc -o CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseInt8Avx512.cc.s

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseInt8Avx512.cc.o.requires:

.PHONY : third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseInt8Avx512.cc.o.requires

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseInt8Avx512.cc.o.provides: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseInt8Avx512.cc.o.requires
	$(MAKE) -f third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/build.make third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseInt8Avx512.cc.o.provides.build
.PHONY : third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseInt8Avx512.cc.o.provides

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseInt8Avx512.cc.o.provides.build: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseInt8Avx512.cc.o


third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseVectorInt8Avx512.cc.o: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/flags.make
third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseVectorInt8Avx512.cc.o: ../third_party/fbgemm/src/FbgemmSparseDenseVectorInt8Avx512.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseVectorInt8Avx512.cc.o"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseVectorInt8Avx512.cc.o -c /home/zzp/code/NoPFS/pytorch/third_party/fbgemm/src/FbgemmSparseDenseVectorInt8Avx512.cc

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseVectorInt8Avx512.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseVectorInt8Avx512.cc.i"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zzp/code/NoPFS/pytorch/third_party/fbgemm/src/FbgemmSparseDenseVectorInt8Avx512.cc > CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseVectorInt8Avx512.cc.i

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseVectorInt8Avx512.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseVectorInt8Avx512.cc.s"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zzp/code/NoPFS/pytorch/third_party/fbgemm/src/FbgemmSparseDenseVectorInt8Avx512.cc -o CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseVectorInt8Avx512.cc.s

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseVectorInt8Avx512.cc.o.requires:

.PHONY : third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseVectorInt8Avx512.cc.o.requires

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseVectorInt8Avx512.cc.o.provides: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseVectorInt8Avx512.cc.o.requires
	$(MAKE) -f third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/build.make third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseVectorInt8Avx512.cc.o.provides.build
.PHONY : third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseVectorInt8Avx512.cc.o.provides

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseVectorInt8Avx512.cc.o.provides.build: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseVectorInt8Avx512.cc.o


third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/QuantUtilsAvx512.cc.o: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/flags.make
third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/QuantUtilsAvx512.cc.o: ../third_party/fbgemm/src/QuantUtilsAvx512.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/QuantUtilsAvx512.cc.o"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fbgemm_avx512.dir/src/QuantUtilsAvx512.cc.o -c /home/zzp/code/NoPFS/pytorch/third_party/fbgemm/src/QuantUtilsAvx512.cc

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/QuantUtilsAvx512.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fbgemm_avx512.dir/src/QuantUtilsAvx512.cc.i"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zzp/code/NoPFS/pytorch/third_party/fbgemm/src/QuantUtilsAvx512.cc > CMakeFiles/fbgemm_avx512.dir/src/QuantUtilsAvx512.cc.i

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/QuantUtilsAvx512.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fbgemm_avx512.dir/src/QuantUtilsAvx512.cc.s"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zzp/code/NoPFS/pytorch/third_party/fbgemm/src/QuantUtilsAvx512.cc -o CMakeFiles/fbgemm_avx512.dir/src/QuantUtilsAvx512.cc.s

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/QuantUtilsAvx512.cc.o.requires:

.PHONY : third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/QuantUtilsAvx512.cc.o.requires

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/QuantUtilsAvx512.cc.o.provides: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/QuantUtilsAvx512.cc.o.requires
	$(MAKE) -f third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/build.make third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/QuantUtilsAvx512.cc.o.provides.build
.PHONY : third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/QuantUtilsAvx512.cc.o.provides

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/QuantUtilsAvx512.cc.o.provides.build: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/QuantUtilsAvx512.cc.o


third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/UtilsAvx512.cc.o: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/flags.make
third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/UtilsAvx512.cc.o: ../third_party/fbgemm/src/UtilsAvx512.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/UtilsAvx512.cc.o"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fbgemm_avx512.dir/src/UtilsAvx512.cc.o -c /home/zzp/code/NoPFS/pytorch/third_party/fbgemm/src/UtilsAvx512.cc

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/UtilsAvx512.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fbgemm_avx512.dir/src/UtilsAvx512.cc.i"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zzp/code/NoPFS/pytorch/third_party/fbgemm/src/UtilsAvx512.cc > CMakeFiles/fbgemm_avx512.dir/src/UtilsAvx512.cc.i

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/UtilsAvx512.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fbgemm_avx512.dir/src/UtilsAvx512.cc.s"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zzp/code/NoPFS/pytorch/third_party/fbgemm/src/UtilsAvx512.cc -o CMakeFiles/fbgemm_avx512.dir/src/UtilsAvx512.cc.s

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/UtilsAvx512.cc.o.requires:

.PHONY : third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/UtilsAvx512.cc.o.requires

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/UtilsAvx512.cc.o.provides: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/UtilsAvx512.cc.o.requires
	$(MAKE) -f third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/build.make third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/UtilsAvx512.cc.o.provides.build
.PHONY : third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/UtilsAvx512.cc.o.provides

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/UtilsAvx512.cc.o.provides.build: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/UtilsAvx512.cc.o


third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512.cc.o: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/flags.make
third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512.cc.o: ../third_party/fbgemm/src/FbgemmFP16UKernelsAvx512.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512.cc.o"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -masm=intel -o CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512.cc.o -c /home/zzp/code/NoPFS/pytorch/third_party/fbgemm/src/FbgemmFP16UKernelsAvx512.cc

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512.cc.i"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -masm=intel -E /home/zzp/code/NoPFS/pytorch/third_party/fbgemm/src/FbgemmFP16UKernelsAvx512.cc > CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512.cc.i

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512.cc.s"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -masm=intel -S /home/zzp/code/NoPFS/pytorch/third_party/fbgemm/src/FbgemmFP16UKernelsAvx512.cc -o CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512.cc.s

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512.cc.o.requires:

.PHONY : third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512.cc.o.requires

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512.cc.o.provides: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512.cc.o.requires
	$(MAKE) -f third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/build.make third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512.cc.o.provides.build
.PHONY : third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512.cc.o.provides

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512.cc.o.provides.build: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512.cc.o


third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512_256.cc.o: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/flags.make
third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512_256.cc.o: ../third_party/fbgemm/src/FbgemmFP16UKernelsAvx512_256.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512_256.cc.o"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -masm=intel -o CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512_256.cc.o -c /home/zzp/code/NoPFS/pytorch/third_party/fbgemm/src/FbgemmFP16UKernelsAvx512_256.cc

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512_256.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512_256.cc.i"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -masm=intel -E /home/zzp/code/NoPFS/pytorch/third_party/fbgemm/src/FbgemmFP16UKernelsAvx512_256.cc > CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512_256.cc.i

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512_256.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512_256.cc.s"
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -masm=intel -S /home/zzp/code/NoPFS/pytorch/third_party/fbgemm/src/FbgemmFP16UKernelsAvx512_256.cc -o CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512_256.cc.s

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512_256.cc.o.requires:

.PHONY : third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512_256.cc.o.requires

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512_256.cc.o.provides: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512_256.cc.o.requires
	$(MAKE) -f third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/build.make third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512_256.cc.o.provides.build
.PHONY : third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512_256.cc.o.provides

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512_256.cc.o.provides.build: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512_256.cc.o


fbgemm_avx512: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmBfloat16ConvertAvx512.cc.o
fbgemm_avx512: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/EmbeddingSpMDMAvx512.cc.o
fbgemm_avx512: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFloat16ConvertAvx512.cc.o
fbgemm_avx512: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseAvx512.cc.o
fbgemm_avx512: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseInt8Avx512.cc.o
fbgemm_avx512: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseVectorInt8Avx512.cc.o
fbgemm_avx512: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/QuantUtilsAvx512.cc.o
fbgemm_avx512: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/UtilsAvx512.cc.o
fbgemm_avx512: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512.cc.o
fbgemm_avx512: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512_256.cc.o
fbgemm_avx512: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/build.make

.PHONY : fbgemm_avx512

# Rule to build all files generated by this target.
third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/build: fbgemm_avx512

.PHONY : third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/build

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/requires: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmBfloat16ConvertAvx512.cc.o.requires
third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/requires: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/EmbeddingSpMDMAvx512.cc.o.requires
third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/requires: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFloat16ConvertAvx512.cc.o.requires
third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/requires: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseAvx512.cc.o.requires
third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/requires: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseInt8Avx512.cc.o.requires
third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/requires: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmSparseDenseVectorInt8Avx512.cc.o.requires
third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/requires: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/QuantUtilsAvx512.cc.o.requires
third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/requires: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/UtilsAvx512.cc.o.requires
third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/requires: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512.cc.o.requires
third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/requires: third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/src/FbgemmFP16UKernelsAvx512_256.cc.o.requires

.PHONY : third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/requires

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/clean:
	cd /home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm && $(CMAKE_COMMAND) -P CMakeFiles/fbgemm_avx512.dir/cmake_clean.cmake
.PHONY : third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/clean

third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/depend:
	cd /home/zzp/code/NoPFS/pytorch/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zzp/code/NoPFS/pytorch /home/zzp/code/NoPFS/pytorch/third_party/fbgemm /home/zzp/code/NoPFS/pytorch/build /home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm /home/zzp/code/NoPFS/pytorch/build/third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : third_party/fbgemm/CMakeFiles/fbgemm_avx512.dir/depend

