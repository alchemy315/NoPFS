// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif

// @generated by tools/codegen/gen.py from RegisterDispatchKey.cpp

#include <c10/core/TensorImpl.h>
#include <c10/core/Allocator.h>
#include <ATen/DeviceGuard.h>
#include <ATen/NativeFunctions.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/Dispatch.h>
#include <c10/util/ExclusivelyOwned.h>
#include <c10/util/Half.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/util/Optional.h>
#include <ATen/Tensor.h>
#include <ATen/Functions.h>
#include <ATen/native/Resize.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

#include <ATen/Config.h>
#include <ATen/core/op_registration/adaption.h>
#include <torch/library.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/ATenCUDAGeneral.h>
#include <ATen/cuda/CUDADevice.h>
#include <ATen/cuda/CUDAContext.h>



namespace at {

// NB: TORCH_LIBRARY_IMPL must be in an anonymous namespace to avoid
// ambiguity with conflicting identifiers that may have been defined in
// at namespace already.
namespace {


void resize_out(const Tensor &out, IntArrayRef sizes, IntArrayRef strides, const TensorOptions &options) {
  TORCH_CHECK(options.dtype() == out.dtype(),
      "Expected out tensor to have dtype ", options.dtype(), ", but got ", out.dtype(), " instead");
  TORCH_CHECK(options.device() == out.device(),
      "Expected out tensor to have device ", options.device(), ", but got ", out.device(), " instead");
  const bool resized = at::native::resize_output(out, sizes);
  // Only restride if a resize occurred; otherwise we ignore the (advisory)
  // strides from the meta function and directly use the output tensor's
  // preexisting strides
  if (resized) {
    if (!strides.empty()) {
      TORCH_INTERNAL_ASSERT(!options.memory_format_opt().has_value());
      at::native::as_strided_(out, sizes, strides);
    } else if (options.memory_format_opt().has_value()) {
      out.unsafeGetTensorImpl()->empty_tensor_restride(*options.memory_format_opt());
    }
  }
}

namespace {

at::Tensor wrapper__as_strided(const at::Tensor & self, at::IntArrayRef size, at::IntArrayRef stride, c10::optional<int64_t> storage_offset) {
    // No device check


  // DeviceGuard omitted
  return at::native::as_strided_qtensorimpl(self, size, stride, storage_offset);
}

} // anonymous namespace
namespace {

at::Tensor wrapper___empty_affine_quantized(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, double scale, int64_t zero_point, c10::optional<at::MemoryFormat> memory_format) {
  c10::optional<Device> common_device = nullopt;
(void)common_device; // Suppress unused variable warning


  globalContext().lazyInitCUDA();
  const DeviceGuard device_guard(device_or_default(device));
  return at::native::empty_affine_quantized(size, dtype, layout, device, pin_memory, scale, zero_point, memory_format);
}

} // anonymous namespace
namespace {

at::Tensor wrapper___empty_per_channel_affine_quantized(at::IntArrayRef size, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  c10::optional<Device> common_device = nullopt;
(void)common_device; // Suppress unused variable warning


  globalContext().lazyInitCUDA();
  const DeviceGuard device_guard(device_or_default(device));
  return at::native::empty_per_channel_affine_quantized(size, scales, zero_points, axis, dtype, layout, device, pin_memory, memory_format);
}

} // anonymous namespace
namespace {

at::Tensor wrapper__empty_quantized(at::IntArrayRef size, const at::Tensor & qtensor, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  c10::optional<Device> common_device = nullopt;
(void)common_device; // Suppress unused variable warning

  c10::impl::check_and_update_common_device(common_device, qtensor, "wrapper__empty_quantized", "qtensor");

  globalContext().lazyInitCUDA();
  const DeviceGuard device_guard(device_or_default(device));
  return at::native::empty_quantized(size, qtensor, dtype, layout, device, pin_memory, memory_format);
}

} // anonymous namespace
namespace {

at::Tensor & wrapper_Scalar_fill__Scalar(at::Tensor & self, const at::Scalar & value) {
    // No device check


  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::fill_(self, value);
}

} // anonymous namespace
namespace {

at::Tensor & wrapper_Tensor_fill__Tensor(at::Tensor & self, const at::Tensor & value) {
    // No device check


  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::fill_(self, value);
}

} // anonymous namespace
namespace {

::std::tuple<at::Tensor,at::Tensor> wrapper_dim_max_dim(const at::Tensor & self, int64_t dim, bool keepdim) {
    // No device check


  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::max(self, dim, keepdim);
}

} // anonymous namespace
namespace {

::std::tuple<at::Tensor,at::Tensor> wrapper_dim_min_dim(const at::Tensor & self, int64_t dim, bool keepdim) {
    // No device check


  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::min(self, dim, keepdim);
}

} // anonymous namespace
namespace {

at::Tensor wrapper___reshape_alias(const at::Tensor & self, at::IntArrayRef size, at::IntArrayRef stride) {
    // No device check


  // DeviceGuard omitted
  return at::native::_reshape_alias(self, size, stride);
}

} // anonymous namespace
namespace {

at::Tensor wrapper__flip(const at::Tensor & self, at::IntArrayRef dims) {
  c10::optional<Device> common_device = nullopt;
(void)common_device; // Suppress unused variable warning

  c10::impl::check_and_update_common_device(common_device, self, "wrapper__flip", "self");

  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::flip(self, dims);
}

} // anonymous namespace
namespace {

at::Tensor wrapper__clone(const at::Tensor & self, c10::optional<at::MemoryFormat> memory_format) {
  c10::optional<Device> common_device = nullopt;
(void)common_device; // Suppress unused variable warning

  c10::impl::check_and_update_common_device(common_device, self, "wrapper__clone", "self");

  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::quantized_clone(self, memory_format);
}

} // anonymous namespace
namespace {

at::Tensor wrapper_self_dequantize_self(const at::Tensor & self) {
  c10::optional<Device> common_device = nullopt;
(void)common_device; // Suppress unused variable warning

  c10::impl::check_and_update_common_device(common_device, self, "wrapper_self_dequantize_self", "self");

  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::dequantize_quantized(self);
}

} // anonymous namespace
namespace {

double wrapper__q_scale(const at::Tensor & self) {
  c10::optional<Device> common_device = nullopt;
(void)common_device; // Suppress unused variable warning

  c10::impl::check_and_update_common_device(common_device, self, "wrapper__q_scale", "self");

  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::q_scale_quant(self);
}

} // anonymous namespace
namespace {

int64_t wrapper__q_zero_point(const at::Tensor & self) {
  c10::optional<Device> common_device = nullopt;
(void)common_device; // Suppress unused variable warning

  c10::impl::check_and_update_common_device(common_device, self, "wrapper__q_zero_point", "self");

  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::q_zero_point_quant(self);
}

} // anonymous namespace
namespace {

at::Tensor wrapper__q_per_channel_scales(const at::Tensor & self) {
  c10::optional<Device> common_device = nullopt;
(void)common_device; // Suppress unused variable warning

  c10::impl::check_and_update_common_device(common_device, self, "wrapper__q_per_channel_scales", "self");

  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::q_per_channel_scales(self);
}

} // anonymous namespace
namespace {

at::Tensor wrapper__q_per_channel_zero_points(const at::Tensor & self) {
  c10::optional<Device> common_device = nullopt;
(void)common_device; // Suppress unused variable warning

  c10::impl::check_and_update_common_device(common_device, self, "wrapper__q_per_channel_zero_points", "self");

  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::q_per_channel_zero_points(self);
}

} // anonymous namespace
namespace {

int64_t wrapper__q_per_channel_axis(const at::Tensor & self) {
  c10::optional<Device> common_device = nullopt;
(void)common_device; // Suppress unused variable warning

  c10::impl::check_and_update_common_device(common_device, self, "wrapper__q_per_channel_axis", "self");

  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::q_per_channel_axis(self);
}

} // anonymous namespace
namespace {

at::Tensor wrapper__int_repr(const at::Tensor & self) {
    // No device check


  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::int_repr_quantized_cuda(self);
}

} // anonymous namespace
namespace {

at::QScheme wrapper__qscheme(const at::Tensor & self) {
  c10::optional<Device> common_device = nullopt;
(void)common_device; // Suppress unused variable warning

  c10::impl::check_and_update_common_device(common_device, self, "wrapper__qscheme", "self");

  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::qscheme_quant(self);
}

} // anonymous namespace
namespace {

at::Tensor & wrapper_source_Storage_storage_offset_set__source_Storage_storage_offset(at::Tensor & self, at::Storage source, int64_t storage_offset, at::IntArrayRef size, at::IntArrayRef stride) {
    // No device check


  // DeviceGuard omitted
  return at::native::set_storage_quantized_(self, source, storage_offset, size, stride);
}

} // anonymous namespace
namespace {

at::Tensor wrapper__view(const at::Tensor & self, at::IntArrayRef size) {
    // No device check


  // DeviceGuard omitted
  return at::native::view(self, size);
}

} // anonymous namespace
namespace {

at::Tensor wrapper__index_select(const at::Tensor & self, int64_t dim, const at::Tensor & index) {
  c10::optional<Device> common_device = nullopt;
(void)common_device; // Suppress unused variable warning

  c10::impl::check_and_update_common_device(common_device, self, "wrapper__index_select", "self");
  c10::impl::check_and_update_common_device(common_device, index, "wrapper__index_select", "index");

  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::index_select_cuda(self, dim, index);
}

} // anonymous namespace
namespace {

at::Tensor & wrapper_out_index_select_out_out(const at::Tensor & self, int64_t dim, const at::Tensor & index, at::Tensor & out) {
  c10::optional<Device> common_device = nullopt;
(void)common_device; // Suppress unused variable warning

  c10::impl::check_and_update_common_device(common_device, out, "wrapper_out_index_select_out_out", "out");
  c10::impl::check_and_update_common_device(common_device, self, "wrapper_out_index_select_out_out", "self");
  c10::impl::check_and_update_common_device(common_device, index, "wrapper_out_index_select_out_out", "index");

  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::index_select_out_cuda(self, dim, index, out);
}

} // anonymous namespace
namespace {

at::Tensor wrapper__unfold(const at::Tensor & self, int64_t dimension, int64_t size, int64_t step) {
    // No device check


  // DeviceGuard omitted
  return at::native::unfold(self, dimension, size, step);
}

} // anonymous namespace

TORCH_LIBRARY_IMPL(aten, QuantizedCUDA, m) {
  m.impl("as_strided",
  TORCH_FN(wrapper__as_strided));
  m.impl("_empty_affine_quantized",
  TORCH_FN(wrapper___empty_affine_quantized));
  m.impl("_empty_per_channel_affine_quantized",
  TORCH_FN(wrapper___empty_per_channel_affine_quantized));
  m.impl("empty_quantized",
  TORCH_FN(wrapper__empty_quantized));
  m.impl("fill_.Scalar",
  TORCH_FN(wrapper_Scalar_fill__Scalar));
  m.impl("fill_.Tensor",
  TORCH_FN(wrapper_Tensor_fill__Tensor));
  m.impl("max.dim",
  TORCH_FN(wrapper_dim_max_dim));
  m.impl("min.dim",
  TORCH_FN(wrapper_dim_min_dim));
  m.impl("_reshape_alias",
  TORCH_FN(wrapper___reshape_alias));
  m.impl("flip",
  TORCH_FN(wrapper__flip));
  m.impl("clone",
  TORCH_FN(wrapper__clone));
  m.impl("dequantize.self",
  TORCH_FN(wrapper_self_dequantize_self));
  m.impl("q_scale",
  TORCH_FN(wrapper__q_scale));
  m.impl("q_zero_point",
  TORCH_FN(wrapper__q_zero_point));
  m.impl("q_per_channel_scales",
  TORCH_FN(wrapper__q_per_channel_scales));
  m.impl("q_per_channel_zero_points",
  TORCH_FN(wrapper__q_per_channel_zero_points));
  m.impl("q_per_channel_axis",
  TORCH_FN(wrapper__q_per_channel_axis));
  m.impl("int_repr",
  TORCH_FN(wrapper__int_repr));
  m.impl("qscheme",
  TORCH_FN(wrapper__qscheme));
  m.impl("set_.source_Storage_storage_offset",
  TORCH_FN(wrapper_source_Storage_storage_offset_set__source_Storage_storage_offset));
  m.impl("view",
  TORCH_FN(wrapper__view));
  m.impl("index_select",
  TORCH_FN(wrapper__index_select));
  m.impl("index_select.out",
  TORCH_FN(wrapper_out_index_select_out_out));
  m.impl("unfold",
  TORCH_FN(wrapper__unfold));
}

} // anonymous namespace

namespace quantizedcuda {


at::Tensor as_strided(const at::Tensor & self, at::IntArrayRef size, at::IntArrayRef stride, c10::optional<int64_t> storage_offset) {
return wrapper__as_strided(self, size, stride, storage_offset);
}

at::Tensor _empty_affine_quantized(at::IntArrayRef size, at::TensorOptions options, double scale, int64_t zero_point, c10::optional<at::MemoryFormat> memory_format) {
return wrapper___empty_affine_quantized(size, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt(), scale, zero_point, c10::impl::check_tensor_options_and_extract_memory_format(options, memory_format));
}

at::Tensor _empty_affine_quantized(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, double scale, int64_t zero_point, c10::optional<at::MemoryFormat> memory_format) {
return wrapper___empty_affine_quantized(size, dtype, layout, device, pin_memory, scale, zero_point, memory_format);
}

at::Tensor _empty_per_channel_affine_quantized(at::IntArrayRef size, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, at::TensorOptions options, c10::optional<at::MemoryFormat> memory_format) {
return wrapper___empty_per_channel_affine_quantized(size, scales, zero_points, axis, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt(), c10::impl::check_tensor_options_and_extract_memory_format(options, memory_format));
}

at::Tensor _empty_per_channel_affine_quantized(at::IntArrayRef size, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
return wrapper___empty_per_channel_affine_quantized(size, scales, zero_points, axis, dtype, layout, device, pin_memory, memory_format);
}

at::Tensor empty_quantized(at::IntArrayRef size, const at::Tensor & qtensor, at::TensorOptions options, c10::optional<at::MemoryFormat> memory_format) {
return wrapper__empty_quantized(size, qtensor, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt(), c10::impl::check_tensor_options_and_extract_memory_format(options, memory_format));
}

at::Tensor empty_quantized(at::IntArrayRef size, const at::Tensor & qtensor, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
return wrapper__empty_quantized(size, qtensor, dtype, layout, device, pin_memory, memory_format);
}

at::Tensor & fill_(at::Tensor & self, const at::Scalar & value) {
return wrapper_Scalar_fill__Scalar(self, value);
}

at::Tensor & fill_(at::Tensor & self, const at::Tensor & value) {
return wrapper_Tensor_fill__Tensor(self, value);
}

::std::tuple<at::Tensor,at::Tensor> max(const at::Tensor & self, int64_t dim, bool keepdim) {
return wrapper_dim_max_dim(self, dim, keepdim);
}

::std::tuple<at::Tensor,at::Tensor> min(const at::Tensor & self, int64_t dim, bool keepdim) {
return wrapper_dim_min_dim(self, dim, keepdim);
}

at::Tensor _reshape_alias(const at::Tensor & self, at::IntArrayRef size, at::IntArrayRef stride) {
return wrapper___reshape_alias(self, size, stride);
}

at::Tensor flip(const at::Tensor & self, at::IntArrayRef dims) {
return wrapper__flip(self, dims);
}

at::Tensor clone(const at::Tensor & self, c10::optional<at::MemoryFormat> memory_format) {
return wrapper__clone(self, memory_format);
}

at::Tensor dequantize(const at::Tensor & self) {
return wrapper_self_dequantize_self(self);
}

double q_scale(const at::Tensor & self) {
return wrapper__q_scale(self);
}

int64_t q_zero_point(const at::Tensor & self) {
return wrapper__q_zero_point(self);
}

at::Tensor q_per_channel_scales(const at::Tensor & self) {
return wrapper__q_per_channel_scales(self);
}

at::Tensor q_per_channel_zero_points(const at::Tensor & self) {
return wrapper__q_per_channel_zero_points(self);
}

int64_t q_per_channel_axis(const at::Tensor & self) {
return wrapper__q_per_channel_axis(self);
}

at::Tensor int_repr(const at::Tensor & self) {
return wrapper__int_repr(self);
}

at::QScheme qscheme(const at::Tensor & self) {
return wrapper__qscheme(self);
}

at::Tensor & set_(at::Tensor & self, at::Storage source, int64_t storage_offset, at::IntArrayRef size, at::IntArrayRef stride) {
return wrapper_source_Storage_storage_offset_set__source_Storage_storage_offset(self, source, storage_offset, size, stride);
}

at::Tensor view(const at::Tensor & self, at::IntArrayRef size) {
return wrapper__view(self, size);
}

at::Tensor index_select(const at::Tensor & self, int64_t dim, const at::Tensor & index) {
return wrapper__index_select(self, dim, index);
}

at::Tensor & index_select_out(at::Tensor & out, const at::Tensor & self, int64_t dim, const at::Tensor & index) {
return wrapper_out_index_select_out_out(self, dim, index, out);
}

at::Tensor & index_select_outf(const at::Tensor & self, int64_t dim, const at::Tensor & index, at::Tensor & out) {
return wrapper_out_index_select_out_out(self, dim, index, out);
}

at::Tensor unfold(const at::Tensor & self, int64_t dimension, int64_t size, int64_t step) {
return wrapper__unfold(self, dimension, size, step);
}

} // namespace quantizedcuda

} // namespace at
