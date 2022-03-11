# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# compile C with /usr/bin/cc
# compile CXX with /usr/bin/c++
C_FLAGS =  -fopenmp -DNDEBUG -fopenmp  -Wall -Wno-unknown-pragmas -fvisibility=internal -msse4 -fPIC -Wformat -Wformat-security -fstack-protector-strong -std=c99  -Wmissing-field-initializers  -Wno-strict-overflow  -O3 -DNDEBUG -DNDEBUG -D_FORTIFY_SOURCE=2 -fPIC   -DCAFFE2_USE_GLOO -DCUDA_HAS_FP16=1 -DHAVE_GCC_GET_CPUID -DUSE_AVX -DUSE_AVX2 -std=gnu11

C_DEFINES = -DDNNL_ENABLE_CONCURRENT_EXEC -DDNNL_ENABLE_CPU_ISA_HINTS -DDNNL_ENABLE_ITT_TASKS -DDNNL_ENABLE_MAX_CPU_ISA -DDNNL_X64=1 -DONNXIFI_ENABLE_EXT=1 -DONNX_ML=1 -DONNX_NAMESPACE=onnx_torch -D__STDC_CONSTANT_MACROS -D__STDC_LIMIT_MACROS

C_INCLUDES = -isystem /home/zzp/code/NoPFS/pytorch/build/third_party/gloo -isystem /home/zzp/code/NoPFS/pytorch/cmake/../third_party/gloo -isystem /home/zzp/code/NoPFS/pytorch/cmake/../third_party/googletest/googlemock/include -isystem /home/zzp/code/NoPFS/pytorch/cmake/../third_party/googletest/googletest/include -isystem /home/zzp/code/NoPFS/pytorch/third_party/protobuf/src -isystem /home/zzp/code/NoPFS/pytorch/third_party/gemmlowp -isystem /home/zzp/code/NoPFS/pytorch/third_party/neon2sse -isystem /home/zzp/code/NoPFS/pytorch/third_party/XNNPACK/include -I/home/zzp/code/NoPFS/pytorch/cmake/../third_party/benchmark/include -isystem /home/zzp/code/NoPFS/pytorch/third_party -isystem /home/zzp/code/NoPFS/pytorch/cmake/../third_party/eigen -isystem /opt/conda/include/python3.7m -isystem /opt/conda/lib/python3.7/site-packages/numpy/core/include -isystem /home/zzp/code/NoPFS/pytorch/cmake/../third_party/pybind11/include -isystem /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -isystem /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent -isystem /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent/include -isystem /usr/lib/x86_64-linux-gnu/openmpi/include -I/home/zzp/code/NoPFS/pytorch/cmake/../third_party/cudnn_frontend/include -isystem /usr/local/cuda/include -I/home/zzp/code/NoPFS/pytorch/build/caffe2/contrib/aten -I/home/zzp/code/NoPFS/pytorch/third_party/onnx -I/home/zzp/code/NoPFS/pytorch/build/third_party/onnx -I/home/zzp/code/NoPFS/pytorch/third_party/foxi -I/home/zzp/code/NoPFS/pytorch/build/third_party/foxi -I/home/zzp/code/NoPFS/pytorch/third_party/ideep/mkl-dnn/include -I/home/zzp/code/NoPFS/pytorch/build/third_party/ideep/mkl-dnn/include -I/home/zzp/code/NoPFS/pytorch/third_party/ideep/mkl-dnn/src 

CXX_FLAGS =  -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -fopenmp -fvisibility-inlines-hidden  -Wall -Wno-unknown-pragmas -fvisibility=internal -msse4 -fPIC -Wformat -Wformat-security -fstack-protector-strong -std=c++11  -Wmissing-field-initializers  -Wno-strict-overflow  -O3 -DNDEBUG -DNDEBUG -D_FORTIFY_SOURCE=2 -fPIC   -DCAFFE2_USE_GLOO -DCUDA_HAS_FP16=1 -DHAVE_GCC_GET_CPUID -DUSE_AVX -DUSE_AVX2 -std=gnu++14

CXX_DEFINES = -DDNNL_ENABLE_CONCURRENT_EXEC -DDNNL_ENABLE_CPU_ISA_HINTS -DDNNL_ENABLE_ITT_TASKS -DDNNL_ENABLE_MAX_CPU_ISA -DDNNL_X64=1 -DONNXIFI_ENABLE_EXT=1 -DONNX_ML=1 -DONNX_NAMESPACE=onnx_torch -D__STDC_CONSTANT_MACROS -D__STDC_LIMIT_MACROS

CXX_INCLUDES = -isystem /home/zzp/code/NoPFS/pytorch/build/third_party/gloo -isystem /home/zzp/code/NoPFS/pytorch/cmake/../third_party/gloo -isystem /home/zzp/code/NoPFS/pytorch/cmake/../third_party/googletest/googlemock/include -isystem /home/zzp/code/NoPFS/pytorch/cmake/../third_party/googletest/googletest/include -isystem /home/zzp/code/NoPFS/pytorch/third_party/protobuf/src -isystem /home/zzp/code/NoPFS/pytorch/third_party/gemmlowp -isystem /home/zzp/code/NoPFS/pytorch/third_party/neon2sse -isystem /home/zzp/code/NoPFS/pytorch/third_party/XNNPACK/include -I/home/zzp/code/NoPFS/pytorch/cmake/../third_party/benchmark/include -isystem /home/zzp/code/NoPFS/pytorch/third_party -isystem /home/zzp/code/NoPFS/pytorch/cmake/../third_party/eigen -isystem /opt/conda/include/python3.7m -isystem /opt/conda/lib/python3.7/site-packages/numpy/core/include -isystem /home/zzp/code/NoPFS/pytorch/cmake/../third_party/pybind11/include -isystem /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -isystem /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent -isystem /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent/include -isystem /usr/lib/x86_64-linux-gnu/openmpi/include -I/home/zzp/code/NoPFS/pytorch/cmake/../third_party/cudnn_frontend/include -isystem /usr/local/cuda/include -I/home/zzp/code/NoPFS/pytorch/build/caffe2/contrib/aten -I/home/zzp/code/NoPFS/pytorch/third_party/onnx -I/home/zzp/code/NoPFS/pytorch/build/third_party/onnx -I/home/zzp/code/NoPFS/pytorch/third_party/foxi -I/home/zzp/code/NoPFS/pytorch/build/third_party/foxi -I/home/zzp/code/NoPFS/pytorch/third_party/ideep/mkl-dnn/include -I/home/zzp/code/NoPFS/pytorch/build/third_party/ideep/mkl-dnn/include -I/home/zzp/code/NoPFS/pytorch/third_party/ideep/mkl-dnn/src 

