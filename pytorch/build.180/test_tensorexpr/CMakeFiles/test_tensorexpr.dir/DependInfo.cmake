# The set of languages for which implicit dependencies are needed:
set(CMAKE_DEPENDS_LANGUAGES
  "CXX"
  )
# The set of files for implicit dependencies of each language:
set(CMAKE_DEPENDS_CHECK_CXX
  "/home/zzp/code/NoPFS/pytorch/test/cpp/common/main.cpp" "/home/zzp/code/NoPFS/pytorch/build/test_tensorexpr/CMakeFiles/test_tensorexpr.dir/__/common/main.cpp.o"
  "/home/zzp/code/NoPFS/pytorch/test/cpp/tensorexpr/padded_buffer.cpp" "/home/zzp/code/NoPFS/pytorch/build/test_tensorexpr/CMakeFiles/test_tensorexpr.dir/padded_buffer.cpp.o"
  "/home/zzp/code/NoPFS/pytorch/test/cpp/tensorexpr/test_approx.cpp" "/home/zzp/code/NoPFS/pytorch/build/test_tensorexpr/CMakeFiles/test_tensorexpr.dir/test_approx.cpp.o"
  "/home/zzp/code/NoPFS/pytorch/test/cpp/tensorexpr/test_aten.cpp" "/home/zzp/code/NoPFS/pytorch/build/test_tensorexpr/CMakeFiles/test_tensorexpr.dir/test_aten.cpp.o"
  "/home/zzp/code/NoPFS/pytorch/test/cpp/tensorexpr/test_boundsinference.cpp" "/home/zzp/code/NoPFS/pytorch/build/test_tensorexpr/CMakeFiles/test_tensorexpr.dir/test_boundsinference.cpp.o"
  "/home/zzp/code/NoPFS/pytorch/test/cpp/tensorexpr/test_conv.cpp" "/home/zzp/code/NoPFS/pytorch/build/test_tensorexpr/CMakeFiles/test_tensorexpr.dir/test_conv.cpp.o"
  "/home/zzp/code/NoPFS/pytorch/test/cpp/tensorexpr/test_cpp_codegen.cpp" "/home/zzp/code/NoPFS/pytorch/build/test_tensorexpr/CMakeFiles/test_tensorexpr.dir/test_cpp_codegen.cpp.o"
  "/home/zzp/code/NoPFS/pytorch/test/cpp/tensorexpr/test_cuda.cpp" "/home/zzp/code/NoPFS/pytorch/build/test_tensorexpr/CMakeFiles/test_tensorexpr.dir/test_cuda.cpp.o"
  "/home/zzp/code/NoPFS/pytorch/test/cpp/tensorexpr/test_expr.cpp" "/home/zzp/code/NoPFS/pytorch/build/test_tensorexpr/CMakeFiles/test_tensorexpr.dir/test_expr.cpp.o"
  "/home/zzp/code/NoPFS/pytorch/test/cpp/tensorexpr/test_external_calls.cpp" "/home/zzp/code/NoPFS/pytorch/build/test_tensorexpr/CMakeFiles/test_tensorexpr.dir/test_external_calls.cpp.o"
  "/home/zzp/code/NoPFS/pytorch/test/cpp/tensorexpr/test_graph_opt.cpp" "/home/zzp/code/NoPFS/pytorch/build/test_tensorexpr/CMakeFiles/test_tensorexpr.dir/test_graph_opt.cpp.o"
  "/home/zzp/code/NoPFS/pytorch/test/cpp/tensorexpr/test_ir_printer.cpp" "/home/zzp/code/NoPFS/pytorch/build/test_tensorexpr/CMakeFiles/test_tensorexpr.dir/test_ir_printer.cpp.o"
  "/home/zzp/code/NoPFS/pytorch/test/cpp/tensorexpr/test_ir_verifier.cpp" "/home/zzp/code/NoPFS/pytorch/build/test_tensorexpr/CMakeFiles/test_tensorexpr.dir/test_ir_verifier.cpp.o"
  "/home/zzp/code/NoPFS/pytorch/test/cpp/tensorexpr/test_kernel.cpp" "/home/zzp/code/NoPFS/pytorch/build/test_tensorexpr/CMakeFiles/test_tensorexpr.dir/test_kernel.cpp.o"
  "/home/zzp/code/NoPFS/pytorch/test/cpp/tensorexpr/test_loopnest.cpp" "/home/zzp/code/NoPFS/pytorch/build/test_tensorexpr/CMakeFiles/test_tensorexpr.dir/test_loopnest.cpp.o"
  "/home/zzp/code/NoPFS/pytorch/test/cpp/tensorexpr/test_memdependency.cpp" "/home/zzp/code/NoPFS/pytorch/build/test_tensorexpr/CMakeFiles/test_tensorexpr.dir/test_memdependency.cpp.o"
  "/home/zzp/code/NoPFS/pytorch/test/cpp/tensorexpr/test_reductions.cpp" "/home/zzp/code/NoPFS/pytorch/build/test_tensorexpr/CMakeFiles/test_tensorexpr.dir/test_reductions.cpp.o"
  "/home/zzp/code/NoPFS/pytorch/test/cpp/tensorexpr/test_registerizer.cpp" "/home/zzp/code/NoPFS/pytorch/build/test_tensorexpr/CMakeFiles/test_tensorexpr.dir/test_registerizer.cpp.o"
  "/home/zzp/code/NoPFS/pytorch/test/cpp/tensorexpr/test_simplify.cpp" "/home/zzp/code/NoPFS/pytorch/build/test_tensorexpr/CMakeFiles/test_tensorexpr.dir/test_simplify.cpp.o"
  "/home/zzp/code/NoPFS/pytorch/test/cpp/tensorexpr/test_te_fuser_pass.cpp" "/home/zzp/code/NoPFS/pytorch/build/test_tensorexpr/CMakeFiles/test_tensorexpr.dir/test_te_fuser_pass.cpp.o"
  "/home/zzp/code/NoPFS/pytorch/test/cpp/tensorexpr/test_type.cpp" "/home/zzp/code/NoPFS/pytorch/build/test_tensorexpr/CMakeFiles/test_tensorexpr.dir/test_type.cpp.o"
  )
set(CMAKE_CXX_COMPILER_ID "GNU")

# Preprocessor definitions for this target.
set(CMAKE_TARGET_DEFINITIONS_CXX
  "HAVE_MALLOC_USABLE_SIZE=1"
  "HAVE_MMAP=1"
  "HAVE_SHM_OPEN=1"
  "HAVE_SHM_UNLINK=1"
  "MINIZ_DISABLE_ZIP_READER_CRC32_CHECKS"
  "ONNXIFI_ENABLE_EXT=1"
  "ONNX_ML=1"
  "ONNX_NAMESPACE=onnx_torch"
  "USE_C10D_GLOO"
  "USE_C10D_MPI"
  "USE_C10D_NCCL"
  "USE_CUDA"
  "USE_DISTRIBUTED"
  "USE_EXTERNAL_MZCRC"
  "USE_GTEST"
  "USE_RPC"
  "USE_TENSORPIPE"
  "_FILE_OFFSET_BITS=64"
  )

# The include file search paths:
set(CMAKE_CXX_TARGET_INCLUDE_PATH
  "aten/src"
  "../aten/src"
  "."
  "../"
  "third_party/gloo"
  "../cmake/../third_party/gloo"
  "../cmake/../third_party/googletest/googlemock/include"
  "../cmake/../third_party/googletest/googletest/include"
  "../third_party/protobuf/src"
  "../third_party/gemmlowp"
  "../third_party/neon2sse"
  "../third_party/XNNPACK/include"
  "../cmake/../third_party/benchmark/include"
  "../third_party"
  "../cmake/../third_party/eigen"
  "/opt/conda/include/python3.7m"
  "/opt/conda/lib/python3.7/site-packages/numpy/core/include"
  "../cmake/../third_party/pybind11/include"
  "/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi"
  "/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent"
  "/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent/include"
  "/usr/lib/x86_64-linux-gnu/openmpi/include"
  "../cmake/../third_party/cudnn_frontend/include"
  "/usr/local/cuda/include"
  "caffe2/contrib/aten"
  "../third_party/onnx"
  "third_party/onnx"
  "../third_party/foxi"
  "third_party/foxi"
  "../torch/include"
  "../third_party/ideep/include"
  "caffe2/../aten/src"
  "caffe2/../aten/src/ATen"
  "../torch/csrc/api"
  "../torch/csrc/api/include"
  "../c10/.."
  "third_party/ideep/mkl-dnn/include"
  "../third_party/ideep/mkl-dnn/src/../include"
  "../c10/cuda/../.."
  "../third_party/googletest/googletest/include"
  "../third_party/googletest/googletest"
  "../third_party/pthreadpool/include"
  )

# Targets to which this target links.
set(CMAKE_TARGET_LINKED_INFO_FILES
  "/home/zzp/code/NoPFS/pytorch/build/caffe2/CMakeFiles/torch.dir/DependInfo.cmake"
  "/home/zzp/code/NoPFS/pytorch/build/third_party/googletest/googletest/CMakeFiles/gtest.dir/DependInfo.cmake"
  "/home/zzp/code/NoPFS/pytorch/build/third_party/protobuf/cmake/CMakeFiles/libprotobuf.dir/DependInfo.cmake"
  "/home/zzp/code/NoPFS/pytorch/build/third_party/ideep/mkl-dnn/src/CMakeFiles/dnnl.dir/DependInfo.cmake"
  "/home/zzp/code/NoPFS/pytorch/build/c10/cuda/CMakeFiles/c10_cuda.dir/DependInfo.cmake"
  "/home/zzp/code/NoPFS/pytorch/build/c10/CMakeFiles/c10.dir/DependInfo.cmake"
  )

# Fortran module output directory.
set(CMAKE_Fortran_TARGET_MODULE_DIR "")
