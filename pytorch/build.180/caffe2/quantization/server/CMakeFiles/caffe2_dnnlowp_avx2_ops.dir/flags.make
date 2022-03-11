# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# compile CXX with /usr/bin/c++
CXX_FLAGS =  -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_KINETO -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -DEDGE_PROFILER_USE_KINETO -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-stringop-overflow -DHAVE_AVX2_CPU_DEFINITION -O3 -DNDEBUG -DNDEBUG -fPIC   -DCAFFE2_USE_GLOO -DCUDA_HAS_FP16=1 -DHAVE_GCC_GET_CPUID -DUSE_AVX -DUSE_AVX2 -DTH_HAVE_THREAD -std=gnu++14

CXX_DEFINES = -DHAVE_MALLOC_USABLE_SIZE=1 -DHAVE_MMAP=1 -DHAVE_SHM_OPEN=1 -DHAVE_SHM_UNLINK=1 -DMINIZ_DISABLE_ZIP_READER_CRC32_CHECKS -DONNXIFI_ENABLE_EXT=1 -DONNX_ML=1 -DONNX_NAMESPACE=onnx_torch -DUSE_EXTERNAL_MZCRC -D_FILE_OFFSET_BITS=64

CXX_INCLUDES = -I/home/zzp/code/NoPFS/pytorch/third_party/fbgemm/include -I/home/zzp/code/NoPFS/pytorch/build/aten/src -I/home/zzp/code/NoPFS/pytorch/aten/src -I/home/zzp/code/NoPFS/pytorch/build -I/home/zzp/code/NoPFS/pytorch -isystem /home/zzp/code/NoPFS/pytorch/build/third_party/gloo -isystem /home/zzp/code/NoPFS/pytorch/cmake/../third_party/gloo -isystem /home/zzp/code/NoPFS/pytorch/cmake/../third_party/googletest/googlemock/include -isystem /home/zzp/code/NoPFS/pytorch/cmake/../third_party/googletest/googletest/include -isystem /home/zzp/code/NoPFS/pytorch/third_party/protobuf/src -isystem /home/zzp/code/NoPFS/pytorch/third_party/gemmlowp -isystem /home/zzp/code/NoPFS/pytorch/third_party/neon2sse -isystem /home/zzp/code/NoPFS/pytorch/third_party/XNNPACK/include -I/home/zzp/code/NoPFS/pytorch/cmake/../third_party/benchmark/include -isystem /home/zzp/code/NoPFS/pytorch/third_party -isystem /home/zzp/code/NoPFS/pytorch/cmake/../third_party/eigen -isystem /opt/conda/include/python3.7m -isystem /opt/conda/lib/python3.7/site-packages/numpy/core/include -isystem /home/zzp/code/NoPFS/pytorch/cmake/../third_party/pybind11/include -isystem /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -isystem /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent -isystem /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent/include -isystem /usr/lib/x86_64-linux-gnu/openmpi/include -I/home/zzp/code/NoPFS/pytorch/cmake/../third_party/cudnn_frontend/include -isystem /usr/local/cuda/include -I/home/zzp/code/NoPFS/pytorch/build/caffe2/contrib/aten -I/home/zzp/code/NoPFS/pytorch/third_party/onnx -I/home/zzp/code/NoPFS/pytorch/build/third_party/onnx -I/home/zzp/code/NoPFS/pytorch/third_party/foxi -I/home/zzp/code/NoPFS/pytorch/build/third_party/foxi -isystem /home/zzp/code/NoPFS/pytorch/torch/include -isystem /home/zzp/code/NoPFS/pytorch/third_party/ideep/include 

# Custom flags: caffe2/quantization/server/CMakeFiles/caffe2_dnnlowp_avx2_ops.dir/elementwise_sum_dnnlowp_op_avx2.cc.o_FLAGS =  -mavx2 -mfma -mf16c -mxsave 

# Custom flags: caffe2/quantization/server/CMakeFiles/caffe2_dnnlowp_avx2_ops.dir/fully_connected_fake_lowp_op_avx2.cc.o_FLAGS =  -mavx2 -mfma -mf16c -mxsave 

# Custom flags: caffe2/quantization/server/CMakeFiles/caffe2_dnnlowp_avx2_ops.dir/group_norm_dnnlowp_op_avx2.cc.o_FLAGS =  -mavx2 -mfma -mf16c -mxsave 

# Custom flags: caffe2/quantization/server/CMakeFiles/caffe2_dnnlowp_avx2_ops.dir/pool_dnnlowp_op_avx2.cc.o_FLAGS =  -mavx2 -mfma -mf16c -mxsave 

# Custom flags: caffe2/quantization/server/CMakeFiles/caffe2_dnnlowp_avx2_ops.dir/relu_dnnlowp_op_avx2.cc.o_FLAGS =  -mavx2 -mfma -mf16c -mxsave 

# Custom flags: caffe2/quantization/server/CMakeFiles/caffe2_dnnlowp_avx2_ops.dir/spatial_batch_norm_dnnlowp_op_avx2.cc.o_FLAGS =  -mavx2 -mfma -mf16c -mxsave 

# Custom flags: caffe2/quantization/server/CMakeFiles/caffe2_dnnlowp_avx2_ops.dir/transpose.cc.o_FLAGS =  -mavx2 -mfma -mf16c -mxsave 

# Custom flags: caffe2/quantization/server/CMakeFiles/caffe2_dnnlowp_avx2_ops.dir/norm_minimization_avx2.cc.o_FLAGS =  -mavx2 -mfma -mf16c -mxsave 

