# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# compile C with /usr/bin/cc
# compile CXX with /usr/bin/c++
C_FLAGS =  -fopenmp -DNDEBUG -O3 -DNDEBUG -DNDEBUG -fPIC   -DCAFFE2_USE_GLOO -DCUDA_HAS_FP16=1 -DHAVE_GCC_GET_CPUID -DUSE_AVX -DUSE_AVX2 -DTH_HAVE_THREAD -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -Wno-write-strings -Wno-unknown-pragmas -Wno-missing-braces -Wno-maybe-uninitialized -fvisibility=hidden -O2 -fopenmp -DCAFFE2_BUILD_MAIN_LIB -pthread -DASMJIT_STATIC -std=gnu11

C_DEFINES = -DADD_BREAKPAD_SIGNAL_HANDLER -DCPUINFO_SUPPORTED_PLATFORM=1 -DFMT_HEADER_ONLY=1 -DFXDIV_USE_INLINE_ASSEMBLY=0 -DHAVE_MALLOC_USABLE_SIZE=1 -DHAVE_MMAP=1 -DHAVE_SHM_OPEN=1 -DHAVE_SHM_UNLINK=1 -DMINIZ_DISABLE_ZIP_READER_CRC32_CHECKS -DNNP_CONVOLUTION_ONLY=0 -DNNP_INFERENCE_ONLY=0 -DONNXIFI_ENABLE_EXT=1 -DONNX_ML=1 -DONNX_NAMESPACE=onnx_torch -DUSE_C10D_GLOO -DUSE_C10D_MPI -DUSE_DISTRIBUTED -DUSE_EXTERNAL_MZCRC -DUSE_RPC -DUSE_TENSORPIPE -D_FILE_OFFSET_BITS=64 -Dtorch_cpu_EXPORTS

C_INCLUDES = -I/home/zzp/code/NoPFS/pytorch/build/aten/src -I/home/zzp/code/NoPFS/pytorch/aten/src -I/home/zzp/code/NoPFS/pytorch/build -I/home/zzp/code/NoPFS/pytorch -isystem /home/zzp/code/NoPFS/pytorch/build/third_party/gloo -isystem /home/zzp/code/NoPFS/pytorch/cmake/../third_party/gloo -isystem /home/zzp/code/NoPFS/pytorch/cmake/../third_party/googletest/googlemock/include -isystem /home/zzp/code/NoPFS/pytorch/cmake/../third_party/googletest/googletest/include -isystem /home/zzp/code/NoPFS/pytorch/third_party/protobuf/src -isystem /home/zzp/code/NoPFS/pytorch/third_party/gemmlowp -isystem /home/zzp/code/NoPFS/pytorch/third_party/neon2sse -isystem /home/zzp/code/NoPFS/pytorch/third_party/XNNPACK/include -I/home/zzp/code/NoPFS/pytorch/cmake/../third_party/benchmark/include -isystem /home/zzp/code/NoPFS/pytorch/third_party -isystem /home/zzp/code/NoPFS/pytorch/cmake/../third_party/eigen -isystem /opt/conda/include/python3.7m -isystem /opt/conda/lib/python3.7/site-packages/numpy/core/include -isystem /home/zzp/code/NoPFS/pytorch/cmake/../third_party/pybind11/include -isystem /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -isystem /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent -isystem /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent/include -isystem /usr/lib/x86_64-linux-gnu/openmpi/include -I/home/zzp/code/NoPFS/pytorch/cmake/../third_party/cudnn_frontend/include -isystem /usr/local/cuda/include -I/home/zzp/code/NoPFS/pytorch/build/caffe2/contrib/aten -I/home/zzp/code/NoPFS/pytorch/third_party/onnx -I/home/zzp/code/NoPFS/pytorch/build/third_party/onnx -I/home/zzp/code/NoPFS/pytorch/third_party/foxi -I/home/zzp/code/NoPFS/pytorch/build/third_party/foxi -isystem /home/zzp/code/NoPFS/pytorch/torch/include -isystem /home/zzp/code/NoPFS/pytorch/third_party/ideep/include -I/home/zzp/code/NoPFS/pytorch/third_party/pocketfft -I/home/zzp/code/NoPFS/pytorch/torch/csrc/api -I/home/zzp/code/NoPFS/pytorch/torch/csrc/api/include -I/home/zzp/code/NoPFS/pytorch/caffe2/aten/src/TH -I/home/zzp/code/NoPFS/pytorch/build/caffe2/aten/src/TH -I/home/zzp/code/NoPFS/pytorch/build/caffe2/aten/src -I/home/zzp/code/NoPFS/pytorch/caffe2/../third_party -I/home/zzp/code/NoPFS/pytorch/caffe2/../third_party/breakpad/src -I/home/zzp/code/NoPFS/pytorch/build/caffe2/../aten/src -I/home/zzp/code/NoPFS/pytorch/build/caffe2/../aten/src/ATen -I/home/zzp/code/NoPFS/pytorch/torch/csrc -I/home/zzp/code/NoPFS/pytorch/third_party/miniz-2.0.8 -I/home/zzp/code/NoPFS/pytorch/third_party/kineto/libkineto/include -I/home/zzp/code/NoPFS/pytorch/third_party/kineto/libkineto/src -I/home/zzp/code/NoPFS/pytorch/torch/csrc/distributed -I/home/zzp/code/NoPFS/pytorch/aten/src/TH -I/home/zzp/code/NoPFS/pytorch/aten/../third_party/catch/single_include -I/home/zzp/code/NoPFS/pytorch/aten/src/ATen/.. -I/home/zzp/code/NoPFS/pytorch/build/caffe2/aten/src/ATen -I/home/zzp/code/NoPFS/pytorch/caffe2/core/nomnigraph/include -isystem /home/zzp/code/NoPFS/pytorch/build/include -I/home/zzp/code/NoPFS/pytorch/third_party/FXdiv/include -I/home/zzp/code/NoPFS/pytorch/c10/.. -I/home/zzp/code/NoPFS/pytorch/build/third_party/ideep/mkl-dnn/include -I/home/zzp/code/NoPFS/pytorch/third_party/ideep/mkl-dnn/src/../include -I/home/zzp/code/NoPFS/pytorch/third_party/pthreadpool/include -I/home/zzp/code/NoPFS/pytorch/third_party/cpuinfo/include -I/home/zzp/code/NoPFS/pytorch/third_party/QNNPACK/include -I/home/zzp/code/NoPFS/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/include -I/home/zzp/code/NoPFS/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src -I/home/zzp/code/NoPFS/pytorch/third_party/cpuinfo/deps/clog/include -I/home/zzp/code/NoPFS/pytorch/third_party/NNPACK/include -I/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/include -I/home/zzp/code/NoPFS/pytorch/third_party/fbgemm -I/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src -I/home/zzp/code/NoPFS/pytorch/third_party/FP16/include -I/home/zzp/code/NoPFS/pytorch/third_party/tensorpipe -I/home/zzp/code/NoPFS/pytorch/build/third_party/tensorpipe -I/home/zzp/code/NoPFS/pytorch/third_party/tensorpipe/third_party/libnop/include -I/home/zzp/code/NoPFS/pytorch/third_party/fmt/include 

CXX_FLAGS =  -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_KINETO -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -DEDGE_PROFILER_USE_KINETO -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-stringop-overflow -DHAVE_AVX2_CPU_DEFINITION -O3 -DNDEBUG -DNDEBUG -fPIC   -DCAFFE2_USE_GLOO -DCUDA_HAS_FP16=1 -DHAVE_GCC_GET_CPUID -DUSE_AVX -DUSE_AVX2 -DTH_HAVE_THREAD -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -Wno-write-strings -Wno-unknown-pragmas -Wno-missing-braces -Wno-maybe-uninitialized -fvisibility=hidden -O2 -fopenmp -DCAFFE2_BUILD_MAIN_LIB -pthread -DASMJIT_STATIC -std=gnu++14

CXX_DEFINES = -DADD_BREAKPAD_SIGNAL_HANDLER -DCPUINFO_SUPPORTED_PLATFORM=1 -DFMT_HEADER_ONLY=1 -DFXDIV_USE_INLINE_ASSEMBLY=0 -DHAVE_MALLOC_USABLE_SIZE=1 -DHAVE_MMAP=1 -DHAVE_SHM_OPEN=1 -DHAVE_SHM_UNLINK=1 -DMINIZ_DISABLE_ZIP_READER_CRC32_CHECKS -DNNP_CONVOLUTION_ONLY=0 -DNNP_INFERENCE_ONLY=0 -DONNXIFI_ENABLE_EXT=1 -DONNX_ML=1 -DONNX_NAMESPACE=onnx_torch -DUSE_C10D_GLOO -DUSE_C10D_MPI -DUSE_DISTRIBUTED -DUSE_EXTERNAL_MZCRC -DUSE_RPC -DUSE_TENSORPIPE -D_FILE_OFFSET_BITS=64 -Dtorch_cpu_EXPORTS

CXX_INCLUDES = -I/home/zzp/code/NoPFS/pytorch/build/aten/src -I/home/zzp/code/NoPFS/pytorch/aten/src -I/home/zzp/code/NoPFS/pytorch/build -I/home/zzp/code/NoPFS/pytorch -isystem /home/zzp/code/NoPFS/pytorch/build/third_party/gloo -isystem /home/zzp/code/NoPFS/pytorch/cmake/../third_party/gloo -isystem /home/zzp/code/NoPFS/pytorch/cmake/../third_party/googletest/googlemock/include -isystem /home/zzp/code/NoPFS/pytorch/cmake/../third_party/googletest/googletest/include -isystem /home/zzp/code/NoPFS/pytorch/third_party/protobuf/src -isystem /home/zzp/code/NoPFS/pytorch/third_party/gemmlowp -isystem /home/zzp/code/NoPFS/pytorch/third_party/neon2sse -isystem /home/zzp/code/NoPFS/pytorch/third_party/XNNPACK/include -I/home/zzp/code/NoPFS/pytorch/cmake/../third_party/benchmark/include -isystem /home/zzp/code/NoPFS/pytorch/third_party -isystem /home/zzp/code/NoPFS/pytorch/cmake/../third_party/eigen -isystem /opt/conda/include/python3.7m -isystem /opt/conda/lib/python3.7/site-packages/numpy/core/include -isystem /home/zzp/code/NoPFS/pytorch/cmake/../third_party/pybind11/include -isystem /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -isystem /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent -isystem /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent/include -isystem /usr/lib/x86_64-linux-gnu/openmpi/include -I/home/zzp/code/NoPFS/pytorch/cmake/../third_party/cudnn_frontend/include -isystem /usr/local/cuda/include -I/home/zzp/code/NoPFS/pytorch/build/caffe2/contrib/aten -I/home/zzp/code/NoPFS/pytorch/third_party/onnx -I/home/zzp/code/NoPFS/pytorch/build/third_party/onnx -I/home/zzp/code/NoPFS/pytorch/third_party/foxi -I/home/zzp/code/NoPFS/pytorch/build/third_party/foxi -isystem /home/zzp/code/NoPFS/pytorch/torch/include -isystem /home/zzp/code/NoPFS/pytorch/third_party/ideep/include -I/home/zzp/code/NoPFS/pytorch/third_party/pocketfft -I/home/zzp/code/NoPFS/pytorch/torch/csrc/api -I/home/zzp/code/NoPFS/pytorch/torch/csrc/api/include -I/home/zzp/code/NoPFS/pytorch/caffe2/aten/src/TH -I/home/zzp/code/NoPFS/pytorch/build/caffe2/aten/src/TH -I/home/zzp/code/NoPFS/pytorch/build/caffe2/aten/src -I/home/zzp/code/NoPFS/pytorch/caffe2/../third_party -I/home/zzp/code/NoPFS/pytorch/caffe2/../third_party/breakpad/src -I/home/zzp/code/NoPFS/pytorch/build/caffe2/../aten/src -I/home/zzp/code/NoPFS/pytorch/build/caffe2/../aten/src/ATen -I/home/zzp/code/NoPFS/pytorch/torch/csrc -I/home/zzp/code/NoPFS/pytorch/third_party/miniz-2.0.8 -I/home/zzp/code/NoPFS/pytorch/third_party/kineto/libkineto/include -I/home/zzp/code/NoPFS/pytorch/third_party/kineto/libkineto/src -I/home/zzp/code/NoPFS/pytorch/torch/csrc/distributed -I/home/zzp/code/NoPFS/pytorch/aten/src/TH -I/home/zzp/code/NoPFS/pytorch/aten/../third_party/catch/single_include -I/home/zzp/code/NoPFS/pytorch/aten/src/ATen/.. -I/home/zzp/code/NoPFS/pytorch/build/caffe2/aten/src/ATen -I/home/zzp/code/NoPFS/pytorch/caffe2/core/nomnigraph/include -isystem /home/zzp/code/NoPFS/pytorch/build/include -I/home/zzp/code/NoPFS/pytorch/third_party/FXdiv/include -I/home/zzp/code/NoPFS/pytorch/c10/.. -I/home/zzp/code/NoPFS/pytorch/build/third_party/ideep/mkl-dnn/include -I/home/zzp/code/NoPFS/pytorch/third_party/ideep/mkl-dnn/src/../include -I/home/zzp/code/NoPFS/pytorch/third_party/pthreadpool/include -I/home/zzp/code/NoPFS/pytorch/third_party/cpuinfo/include -I/home/zzp/code/NoPFS/pytorch/third_party/QNNPACK/include -I/home/zzp/code/NoPFS/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/include -I/home/zzp/code/NoPFS/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src -I/home/zzp/code/NoPFS/pytorch/third_party/cpuinfo/deps/clog/include -I/home/zzp/code/NoPFS/pytorch/third_party/NNPACK/include -I/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/include -I/home/zzp/code/NoPFS/pytorch/third_party/fbgemm -I/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/third_party/asmjit/src -I/home/zzp/code/NoPFS/pytorch/third_party/FP16/include -I/home/zzp/code/NoPFS/pytorch/third_party/tensorpipe -I/home/zzp/code/NoPFS/pytorch/build/third_party/tensorpipe -I/home/zzp/code/NoPFS/pytorch/third_party/tensorpipe/third_party/libnop/include -I/home/zzp/code/NoPFS/pytorch/third_party/fmt/include 

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/MapAllocator.cpp.o_FLAGS = -fno-openmp

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/EmbeddingBag.cpp.o_FLAGS = -Wno-attributes

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/QuantizedLinear.cpp.o_FLAGS = -Wno-deprecated-declarations

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/RNN.cpp.o_FLAGS = -Wno-deprecated-declarations

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/quantized/cpu/qlinear_prepack.cpp.o_FLAGS = -Wno-deprecated-declarations

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/quantized/cpu/qlinear_unpack.cpp.o_FLAGS = -Wno-deprecated-declarations

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/mkldnn/Pooling.cpp.o_FLAGS = -Werror

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/layer_norm_kernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/group_norm_kernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/batch_norm_kernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/UpSampleMoreKernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/UpSampleKernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/UnfoldBackwardKernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/Unfold2d.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/UnaryOpsKernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/TensorCompareKernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/SumKernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/StackKernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/SortingKernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/SoftMaxKernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/ScatterGatherKernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/RenormKernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/ReduceOpsKernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/ReduceAllOpsKernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/RangeFactoriesKernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/PowKernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/PointwiseOpsKernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/MultinomialKernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/MaxUnpoolKernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/MaxPooling.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/MaxPoolKernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/LinearAlgebraKernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/LerpKernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/IndexKernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/HistogramKernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/GridSamplerKernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/FunctionOfAMatrixUtilsKernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/FillKernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/DistanceOpsKernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/DepthwiseConvKernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/CrossKernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/CopyKernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/ComplexKernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/CatKernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/BlasKernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/BinaryOpsKernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/AvgPoolKernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/AdaptiveMaxPoolKernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/AdaptiveAvgPoolKernel.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/Activation.cpp.DEFAULT.cpp.o_FLAGS = -O3  -DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/operators/box_with_nms_limit_op.cc.o_FLAGS = -Wno-attributes

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/torch/csrc/autograd/record_function_ops.cpp.o_FLAGS = -Wno-deprecated-declarations

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/torch/csrc/jit/passes/frozen_conv_add_relu_fusion.cpp.o_FLAGS = -DUSE_CUDA=1

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/torch/csrc/jit/tensorexpr/llvm_jit.cpp.o_FLAGS = -Wno-noexcept-type

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/torch/csrc/jit/serialization/export.cpp.o_FLAGS = -Wno-deprecated-declarations

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/torch/csrc/distributed/c10d/ProcessGroupMPI.cpp.o_FLAGS = -Wno-deprecated-declarations

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/layer_norm_kernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/group_norm_kernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/batch_norm_kernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/UpSampleMoreKernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/UpSampleKernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/UnfoldBackwardKernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/Unfold2d.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/UnaryOpsKernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/TensorCompareKernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/SumKernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/StackKernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/SortingKernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/SoftMaxKernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/ScatterGatherKernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/RenormKernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/ReduceOpsKernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/ReduceAllOpsKernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/RangeFactoriesKernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/PowKernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/PointwiseOpsKernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/MultinomialKernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/MaxUnpoolKernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/MaxPooling.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/MaxPoolKernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/LinearAlgebraKernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/LerpKernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/IndexKernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/HistogramKernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/GridSamplerKernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/FunctionOfAMatrixUtilsKernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/FillKernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/DistanceOpsKernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/DepthwiseConvKernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/CrossKernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/CopyKernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/ComplexKernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/CatKernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/BlasKernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/BinaryOpsKernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/AvgPoolKernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/AdaptiveMaxPoolKernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/AdaptiveAvgPoolKernel.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

# Custom flags: caffe2/CMakeFiles/torch_cpu.dir/__/aten/src/ATen/native/cpu/Activation.cpp.AVX2.cpp.o_FLAGS = -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2

