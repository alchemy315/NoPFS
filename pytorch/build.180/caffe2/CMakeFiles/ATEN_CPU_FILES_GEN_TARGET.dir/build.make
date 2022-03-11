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

# Utility rule file for ATEN_CPU_FILES_GEN_TARGET.

# Include the progress variables for this target.
include caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET.dir/progress.make

caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/CPUFunctions.h
caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/CPUFunctions_inl.h
caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/CompositeExplicitAutogradFunctions.h
caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/CompositeExplicitAutogradFunctions_inl.h
caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/CompositeImplicitAutogradFunctions.h
caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/CompositeImplicitAutogradFunctions_inl.h
caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/Declarations.yaml
caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/Functions.cpp
caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/Functions.h
caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/MetaFunctions.h
caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/MetaFunctions_inl.h
caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/NativeFunctions.h
caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/NativeMetaFunctions.h
caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/Operators.h
caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/Operators_0.cpp
caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/Operators_1.cpp
caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/Operators_2.cpp
caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/Operators_3.cpp
caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/Operators_4.cpp
caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/RedispatchFunctions.h
caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/RegisterBackendSelect.cpp
caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/RegisterCPU.cpp
caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/RegisterCompositeExplicitAutograd.cpp
caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/RegisterCompositeImplicitAutograd.cpp
caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/RegisterMeta.cpp
caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/RegisterMkldnnCPU.cpp
caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/RegisterQuantizedCPU.cpp
caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/RegisterSchema.cpp
caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/RegisterSparseCPU.cpp
caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/RegisterSparseCsrCPU.cpp
caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/RegistrationDeclarations.h
caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/core/ATenOpList.cpp
caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/core/TensorBody.h
caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/core/TensorMethods.cpp


aten/src/ATen/CPUFunctions.h: ../tools/codegen/__init__.py
aten/src/ATen/CPUFunctions.h: ../tools/codegen/api/__init__.py
aten/src/ATen/CPUFunctions.h: ../tools/codegen/api/autograd.py
aten/src/ATen/CPUFunctions.h: ../tools/codegen/api/cpp.py
aten/src/ATen/CPUFunctions.h: ../tools/codegen/api/dispatcher.py
aten/src/ATen/CPUFunctions.h: ../tools/codegen/api/meta.py
aten/src/ATen/CPUFunctions.h: ../tools/codegen/api/native.py
aten/src/ATen/CPUFunctions.h: ../tools/codegen/api/python.py
aten/src/ATen/CPUFunctions.h: ../tools/codegen/api/structured.py
aten/src/ATen/CPUFunctions.h: ../tools/codegen/api/translate.py
aten/src/ATen/CPUFunctions.h: ../tools/codegen/api/types.py
aten/src/ATen/CPUFunctions.h: ../tools/codegen/code_template.py
aten/src/ATen/CPUFunctions.h: ../tools/codegen/context.py
aten/src/ATen/CPUFunctions.h: ../tools/codegen/dest/__init__.py
aten/src/ATen/CPUFunctions.h: ../tools/codegen/dest/native_functions.py
aten/src/ATen/CPUFunctions.h: ../tools/codegen/dest/register_dispatch_key.py
aten/src/ATen/CPUFunctions.h: ../tools/codegen/gen.py
aten/src/ATen/CPUFunctions.h: ../tools/codegen/gen_backend_stubs.py
aten/src/ATen/CPUFunctions.h: ../tools/codegen/local.py
aten/src/ATen/CPUFunctions.h: ../tools/codegen/model.py
aten/src/ATen/CPUFunctions.h: ../tools/codegen/selective_build/__init__.py
aten/src/ATen/CPUFunctions.h: ../tools/codegen/selective_build/operator.py
aten/src/ATen/CPUFunctions.h: ../tools/codegen/selective_build/selector.py
aten/src/ATen/CPUFunctions.h: ../tools/codegen/utils.py
aten/src/ATen/CPUFunctions.h: ../aten/src/ATen/templates/ATenOpList.cpp
aten/src/ATen/CPUFunctions.h: ../aten/src/ATen/templates/DispatchKeyFunctions.h
aten/src/ATen/CPUFunctions.h: ../aten/src/ATen/templates/DispatchKeyFunctions_inl.h
aten/src/ATen/CPUFunctions.h: ../aten/src/ATen/templates/DispatchKeyNativeFunctions.h
aten/src/ATen/CPUFunctions.h: ../aten/src/ATen/templates/Functions.cpp
aten/src/ATen/CPUFunctions.h: ../aten/src/ATen/templates/Functions.h
aten/src/ATen/CPUFunctions.h: ../aten/src/ATen/templates/NativeFunctions.h
aten/src/ATen/CPUFunctions.h: ../aten/src/ATen/templates/NativeMetaFunctions.h
aten/src/ATen/CPUFunctions.h: ../aten/src/ATen/templates/Operators.cpp
aten/src/ATen/CPUFunctions.h: ../aten/src/ATen/templates/Operators.h
aten/src/ATen/CPUFunctions.h: ../aten/src/ATen/templates/RedispatchFunctions.cpp
aten/src/ATen/CPUFunctions.h: ../aten/src/ATen/templates/RedispatchFunctions.h
aten/src/ATen/CPUFunctions.h: ../aten/src/ATen/templates/RegisterBackendSelect.cpp
aten/src/ATen/CPUFunctions.h: ../aten/src/ATen/templates/RegisterDispatchKey.cpp
aten/src/ATen/CPUFunctions.h: ../aten/src/ATen/templates/RegisterSchema.cpp
aten/src/ATen/CPUFunctions.h: ../aten/src/ATen/templates/RegistrationDeclarations.h
aten/src/ATen/CPUFunctions.h: ../aten/src/ATen/templates/TensorBody.h
aten/src/ATen/CPUFunctions.h: ../aten/src/ATen/templates/TensorMethods.cpp
aten/src/ATen/CPUFunctions.h: ../aten/src/ATen/native/native_functions.yaml
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/zzp/code/NoPFS/pytorch/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating ../aten/src/ATen/CPUFunctions.h, ../aten/src/ATen/CPUFunctions_inl.h, ../aten/src/ATen/CompositeExplicitAutogradFunctions.h, ../aten/src/ATen/CompositeExplicitAutogradFunctions_inl.h, ../aten/src/ATen/CompositeImplicitAutogradFunctions.h, ../aten/src/ATen/CompositeImplicitAutogradFunctions_inl.h, ../aten/src/ATen/Declarations.yaml, ../aten/src/ATen/Functions.cpp, ../aten/src/ATen/Functions.h, ../aten/src/ATen/MetaFunctions.h, ../aten/src/ATen/MetaFunctions_inl.h, ../aten/src/ATen/NativeFunctions.h, ../aten/src/ATen/NativeMetaFunctions.h, ../aten/src/ATen/Operators.h, ../aten/src/ATen/Operators_0.cpp, ../aten/src/ATen/Operators_1.cpp, ../aten/src/ATen/Operators_2.cpp, ../aten/src/ATen/Operators_3.cpp, ../aten/src/ATen/Operators_4.cpp, ../aten/src/ATen/RedispatchFunctions.h, ../aten/src/ATen/RegisterBackendSelect.cpp, ../aten/src/ATen/RegisterCPU.cpp, ../aten/src/ATen/RegisterCompositeExplicitAutograd.cpp, ../aten/src/ATen/RegisterCompositeImplicitAutograd.cpp, ../aten/src/ATen/RegisterMeta.cpp, ../aten/src/ATen/RegisterMkldnnCPU.cpp, ../aten/src/ATen/RegisterQuantizedCPU.cpp, ../aten/src/ATen/RegisterSchema.cpp, ../aten/src/ATen/RegisterSparseCPU.cpp, ../aten/src/ATen/RegisterSparseCsrCPU.cpp, ../aten/src/ATen/RegistrationDeclarations.h, ../aten/src/ATen/CUDAFunctions.h, ../aten/src/ATen/CUDAFunctions_inl.h, ../aten/src/ATen/RegisterCUDA.cpp, ../aten/src/ATen/RegisterQuantizedCUDA.cpp, ../aten/src/ATen/RegisterSparseCUDA.cpp, ../aten/src/ATen/RegisterSparseCsrCUDA.cpp, ../aten/src/ATen/core/ATenOpList.cpp, ../aten/src/ATen/core/TensorBody.h, ../aten/src/ATen/core/TensorMethods.cpp"
	cd /home/zzp/code/NoPFS/pytorch && /opt/conda/bin/python -m tools.codegen.gen --source-path /home/zzp/code/NoPFS/pytorch/cmake/../aten/src/ATen --install_dir /home/zzp/code/NoPFS/pytorch/build/aten/src/ATen

aten/src/ATen/CPUFunctions_inl.h: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/CPUFunctions_inl.h

aten/src/ATen/CompositeExplicitAutogradFunctions.h: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/CompositeExplicitAutogradFunctions.h

aten/src/ATen/CompositeExplicitAutogradFunctions_inl.h: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/CompositeExplicitAutogradFunctions_inl.h

aten/src/ATen/CompositeImplicitAutogradFunctions.h: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/CompositeImplicitAutogradFunctions.h

aten/src/ATen/CompositeImplicitAutogradFunctions_inl.h: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/CompositeImplicitAutogradFunctions_inl.h

aten/src/ATen/Declarations.yaml: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/Declarations.yaml

aten/src/ATen/Functions.cpp: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/Functions.cpp

aten/src/ATen/Functions.h: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/Functions.h

aten/src/ATen/MetaFunctions.h: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/MetaFunctions.h

aten/src/ATen/MetaFunctions_inl.h: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/MetaFunctions_inl.h

aten/src/ATen/NativeFunctions.h: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/NativeFunctions.h

aten/src/ATen/NativeMetaFunctions.h: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/NativeMetaFunctions.h

aten/src/ATen/Operators.h: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/Operators.h

aten/src/ATen/Operators_0.cpp: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/Operators_0.cpp

aten/src/ATen/Operators_1.cpp: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/Operators_1.cpp

aten/src/ATen/Operators_2.cpp: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/Operators_2.cpp

aten/src/ATen/Operators_3.cpp: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/Operators_3.cpp

aten/src/ATen/Operators_4.cpp: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/Operators_4.cpp

aten/src/ATen/RedispatchFunctions.h: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/RedispatchFunctions.h

aten/src/ATen/RegisterBackendSelect.cpp: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/RegisterBackendSelect.cpp

aten/src/ATen/RegisterCPU.cpp: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/RegisterCPU.cpp

aten/src/ATen/RegisterCompositeExplicitAutograd.cpp: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/RegisterCompositeExplicitAutograd.cpp

aten/src/ATen/RegisterCompositeImplicitAutograd.cpp: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/RegisterCompositeImplicitAutograd.cpp

aten/src/ATen/RegisterMeta.cpp: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/RegisterMeta.cpp

aten/src/ATen/RegisterMkldnnCPU.cpp: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/RegisterMkldnnCPU.cpp

aten/src/ATen/RegisterQuantizedCPU.cpp: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/RegisterQuantizedCPU.cpp

aten/src/ATen/RegisterSchema.cpp: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/RegisterSchema.cpp

aten/src/ATen/RegisterSparseCPU.cpp: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/RegisterSparseCPU.cpp

aten/src/ATen/RegisterSparseCsrCPU.cpp: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/RegisterSparseCsrCPU.cpp

aten/src/ATen/RegistrationDeclarations.h: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/RegistrationDeclarations.h

aten/src/ATen/CUDAFunctions.h: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/CUDAFunctions.h

aten/src/ATen/CUDAFunctions_inl.h: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/CUDAFunctions_inl.h

aten/src/ATen/RegisterCUDA.cpp: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/RegisterCUDA.cpp

aten/src/ATen/RegisterQuantizedCUDA.cpp: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/RegisterQuantizedCUDA.cpp

aten/src/ATen/RegisterSparseCUDA.cpp: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/RegisterSparseCUDA.cpp

aten/src/ATen/RegisterSparseCsrCUDA.cpp: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/RegisterSparseCsrCUDA.cpp

aten/src/ATen/core/ATenOpList.cpp: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/core/ATenOpList.cpp

aten/src/ATen/core/TensorBody.h: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/core/TensorBody.h

aten/src/ATen/core/TensorMethods.cpp: aten/src/ATen/CPUFunctions.h
	@$(CMAKE_COMMAND) -E touch_nocreate aten/src/ATen/core/TensorMethods.cpp

ATEN_CPU_FILES_GEN_TARGET: caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/CPUFunctions.h
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/CPUFunctions_inl.h
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/CompositeExplicitAutogradFunctions.h
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/CompositeExplicitAutogradFunctions_inl.h
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/CompositeImplicitAutogradFunctions.h
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/CompositeImplicitAutogradFunctions_inl.h
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/Declarations.yaml
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/Functions.cpp
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/Functions.h
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/MetaFunctions.h
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/MetaFunctions_inl.h
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/NativeFunctions.h
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/NativeMetaFunctions.h
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/Operators.h
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/Operators_0.cpp
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/Operators_1.cpp
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/Operators_2.cpp
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/Operators_3.cpp
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/Operators_4.cpp
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/RedispatchFunctions.h
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/RegisterBackendSelect.cpp
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/RegisterCPU.cpp
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/RegisterCompositeExplicitAutograd.cpp
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/RegisterCompositeImplicitAutograd.cpp
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/RegisterMeta.cpp
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/RegisterMkldnnCPU.cpp
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/RegisterQuantizedCPU.cpp
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/RegisterSchema.cpp
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/RegisterSparseCPU.cpp
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/RegisterSparseCsrCPU.cpp
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/RegistrationDeclarations.h
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/CUDAFunctions.h
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/CUDAFunctions_inl.h
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/RegisterCUDA.cpp
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/RegisterQuantizedCUDA.cpp
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/RegisterSparseCUDA.cpp
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/RegisterSparseCsrCUDA.cpp
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/core/ATenOpList.cpp
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/core/TensorBody.h
ATEN_CPU_FILES_GEN_TARGET: aten/src/ATen/core/TensorMethods.cpp
ATEN_CPU_FILES_GEN_TARGET: caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET.dir/build.make

.PHONY : ATEN_CPU_FILES_GEN_TARGET

# Rule to build all files generated by this target.
caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET.dir/build: ATEN_CPU_FILES_GEN_TARGET

.PHONY : caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET.dir/build

caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET.dir/clean:
	cd /home/zzp/code/NoPFS/pytorch/build/caffe2 && $(CMAKE_COMMAND) -P CMakeFiles/ATEN_CPU_FILES_GEN_TARGET.dir/cmake_clean.cmake
.PHONY : caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET.dir/clean

caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET.dir/depend:
	cd /home/zzp/code/NoPFS/pytorch/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zzp/code/NoPFS/pytorch /home/zzp/code/NoPFS/pytorch/caffe2 /home/zzp/code/NoPFS/pytorch/build /home/zzp/code/NoPFS/pytorch/build/caffe2 /home/zzp/code/NoPFS/pytorch/build/caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : caffe2/CMakeFiles/ATEN_CPU_FILES_GEN_TARGET.dir/depend
