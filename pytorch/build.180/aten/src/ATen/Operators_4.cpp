#include <ATen/Operators.h>
#include <ATen/Tensor.h>
#include <ATen/core/dispatch/Dispatcher.h>

// NOTE See [Sharded File] comment in VariableType

namespace at { namespace _ops {


STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(set_data, name, "aten::set_data")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(set_data, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(set_data, schema_str, "set_data(Tensor(a!) self, Tensor new_data) -> ()")

// aten::set_data(Tensor(a!) self, Tensor new_data) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<set_data::schema> create_set_data_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(set_data::name, set_data::overload_name)
      .typed<set_data::schema>();
}

// aten::set_data(Tensor(a!) self, Tensor new_data) -> ()
void set_data::call(at::Tensor & self, const at::Tensor & new_data) {
    static auto op = create_set_data_typed_handle();
    return op.call(self, new_data);
}

// aten::set_data(Tensor(a!) self, Tensor new_data) -> ()
void set_data::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & new_data) {
    static auto op = create_set_data_typed_handle();
    return op.redispatch(dispatchKeySet, self, new_data);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fw_primal, name, "aten::_fw_primal")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fw_primal, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fw_primal, schema_str, "_fw_primal(Tensor(a) self, int level) -> Tensor(a)")

// aten::_fw_primal(Tensor(a) self, int level) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<_fw_primal::schema> create__fw_primal_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_fw_primal::name, _fw_primal::overload_name)
      .typed<_fw_primal::schema>();
}

// aten::_fw_primal(Tensor(a) self, int level) -> Tensor(a)
at::Tensor _fw_primal::call(const at::Tensor & self, int64_t level) {
    static auto op = create__fw_primal_typed_handle();
    return op.call(self, level);
}

// aten::_fw_primal(Tensor(a) self, int level) -> Tensor(a)
at::Tensor _fw_primal::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t level) {
    static auto op = create__fw_primal_typed_handle();
    return op.redispatch(dispatchKeySet, self, level);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(align_tensors, name, "aten::align_tensors")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(align_tensors, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(align_tensors, schema_str, "align_tensors(Tensor[] tensors) -> Tensor[]")

// aten::align_tensors(Tensor[] tensors) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<align_tensors::schema> create_align_tensors_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(align_tensors::name, align_tensors::overload_name)
      .typed<align_tensors::schema>();
}

// aten::align_tensors(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> align_tensors::call(at::TensorList tensors) {
    static auto op = create_align_tensors_typed_handle();
    return op.call(tensors);
}

// aten::align_tensors(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> align_tensors::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create_align_tensors_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_assert_async, name, "aten::_assert_async")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_assert_async, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_assert_async, schema_str, "_assert_async(Tensor self) -> ()")

// aten::_assert_async(Tensor self) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_assert_async::schema> create__assert_async_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_assert_async::name, _assert_async::overload_name)
      .typed<_assert_async::schema>();
}

// aten::_assert_async(Tensor self) -> ()
void _assert_async::call(const at::Tensor & self) {
    static auto op = create__assert_async_typed_handle();
    return op.call(self);
}

// aten::_assert_async(Tensor self) -> ()
void _assert_async::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create__assert_async_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_masked_scale, name, "aten::_masked_scale")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_masked_scale, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_masked_scale, schema_str, "_masked_scale(Tensor self, Tensor mask, float scale) -> Tensor")

// aten::_masked_scale(Tensor self, Tensor mask, float scale) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_masked_scale::schema> create__masked_scale_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_masked_scale::name, _masked_scale::overload_name)
      .typed<_masked_scale::schema>();
}

// aten::_masked_scale(Tensor self, Tensor mask, float scale) -> Tensor
at::Tensor _masked_scale::call(const at::Tensor & self, const at::Tensor & mask, double scale) {
    static auto op = create__masked_scale_typed_handle();
    return op.call(self, mask, scale);
}

// aten::_masked_scale(Tensor self, Tensor mask, float scale) -> Tensor
at::Tensor _masked_scale::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mask, double scale) {
    static auto op = create__masked_scale_typed_handle();
    return op.redispatch(dispatchKeySet, self, mask, scale);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sobol_engine_draw, name, "aten::_sobol_engine_draw")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sobol_engine_draw, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sobol_engine_draw, schema_str, "_sobol_engine_draw(Tensor quasi, int n, Tensor sobolstate, int dimension, int num_generated, ScalarType? dtype) -> (Tensor, Tensor)")

// aten::_sobol_engine_draw(Tensor quasi, int n, Tensor sobolstate, int dimension, int num_generated, ScalarType? dtype) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<_sobol_engine_draw::schema> create__sobol_engine_draw_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_sobol_engine_draw::name, _sobol_engine_draw::overload_name)
      .typed<_sobol_engine_draw::schema>();
}

// aten::_sobol_engine_draw(Tensor quasi, int n, Tensor sobolstate, int dimension, int num_generated, ScalarType? dtype) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> _sobol_engine_draw::call(const at::Tensor & quasi, int64_t n, const at::Tensor & sobolstate, int64_t dimension, int64_t num_generated, c10::optional<at::ScalarType> dtype) {
    static auto op = create__sobol_engine_draw_typed_handle();
    return op.call(quasi, n, sobolstate, dimension, num_generated, dtype);
}

// aten::_sobol_engine_draw(Tensor quasi, int n, Tensor sobolstate, int dimension, int num_generated, ScalarType? dtype) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> _sobol_engine_draw::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & quasi, int64_t n, const at::Tensor & sobolstate, int64_t dimension, int64_t num_generated, c10::optional<at::ScalarType> dtype) {
    static auto op = create__sobol_engine_draw_typed_handle();
    return op.redispatch(dispatchKeySet, quasi, n, sobolstate, dimension, num_generated, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sobol_engine_initialize_state_, name, "aten::_sobol_engine_initialize_state_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sobol_engine_initialize_state_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sobol_engine_initialize_state_, schema_str, "_sobol_engine_initialize_state_(Tensor(a!) self, int dimension) -> Tensor(a!)")

// aten::_sobol_engine_initialize_state_(Tensor(a!) self, int dimension) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<_sobol_engine_initialize_state_::schema> create__sobol_engine_initialize_state__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_sobol_engine_initialize_state_::name, _sobol_engine_initialize_state_::overload_name)
      .typed<_sobol_engine_initialize_state_::schema>();
}

// aten::_sobol_engine_initialize_state_(Tensor(a!) self, int dimension) -> Tensor(a!)
at::Tensor & _sobol_engine_initialize_state_::call(at::Tensor & self, int64_t dimension) {
    static auto op = create__sobol_engine_initialize_state__typed_handle();
    return op.call(self, dimension);
}

// aten::_sobol_engine_initialize_state_(Tensor(a!) self, int dimension) -> Tensor(a!)
at::Tensor & _sobol_engine_initialize_state_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t dimension) {
    static auto op = create__sobol_engine_initialize_state__typed_handle();
    return op.redispatch(dispatchKeySet, self, dimension);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_reshape_from_tensor, name, "aten::_reshape_from_tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_reshape_from_tensor, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_reshape_from_tensor, schema_str, "_reshape_from_tensor(Tensor self, Tensor shape) -> Tensor")

// aten::_reshape_from_tensor(Tensor self, Tensor shape) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_reshape_from_tensor::schema> create__reshape_from_tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_reshape_from_tensor::name, _reshape_from_tensor::overload_name)
      .typed<_reshape_from_tensor::schema>();
}

// aten::_reshape_from_tensor(Tensor self, Tensor shape) -> Tensor
at::Tensor _reshape_from_tensor::call(const at::Tensor & self, const at::Tensor & shape) {
    static auto op = create__reshape_from_tensor_typed_handle();
    return op.call(self, shape);
}

// aten::_reshape_from_tensor(Tensor self, Tensor shape) -> Tensor
at::Tensor _reshape_from_tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & shape) {
    static auto op = create__reshape_from_tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, shape);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(dropout_, name, "aten::dropout_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(dropout_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(dropout_, schema_str, "dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)")

// aten::dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<dropout_::schema> create_dropout__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(dropout_::name, dropout_::overload_name)
      .typed<dropout_::schema>();
}

// aten::dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)
at::Tensor & dropout_::call(at::Tensor & self, double p, bool train) {
    static auto op = create_dropout__typed_handle();
    return op.call(self, p, train);
}

// aten::dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)
at::Tensor & dropout_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, double p, bool train) {
    static auto op = create_dropout__typed_handle();
    return op.redispatch(dispatchKeySet, self, p, train);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(alpha_dropout, name, "aten::alpha_dropout")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(alpha_dropout, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(alpha_dropout, schema_str, "alpha_dropout(Tensor input, float p, bool train) -> Tensor")

// aten::alpha_dropout(Tensor input, float p, bool train) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<alpha_dropout::schema> create_alpha_dropout_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(alpha_dropout::name, alpha_dropout::overload_name)
      .typed<alpha_dropout::schema>();
}

// aten::alpha_dropout(Tensor input, float p, bool train) -> Tensor
at::Tensor alpha_dropout::call(const at::Tensor & input, double p, bool train) {
    static auto op = create_alpha_dropout_typed_handle();
    return op.call(input, p, train);
}

// aten::alpha_dropout(Tensor input, float p, bool train) -> Tensor
at::Tensor alpha_dropout::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, double p, bool train) {
    static auto op = create_alpha_dropout_typed_handle();
    return op.redispatch(dispatchKeySet, input, p, train);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(absolute_out, name, "aten::absolute")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(absolute_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(absolute_out, schema_str, "absolute.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::absolute.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<absolute_out::schema> create_absolute_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(absolute_out::name, absolute_out::overload_name)
      .typed<absolute_out::schema>();
}

// aten::absolute.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & absolute_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_absolute_out_typed_handle();
    return op.call(self, out);
}

// aten::absolute.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & absolute_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_absolute_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(view_as_real, name, "aten::view_as_real")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(view_as_real, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(view_as_real, schema_str, "view_as_real(Tensor(a) self) -> Tensor(a)")

// aten::view_as_real(Tensor(a) self) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<view_as_real::schema> create_view_as_real_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(view_as_real::name, view_as_real::overload_name)
      .typed<view_as_real::schema>();
}

// aten::view_as_real(Tensor(a) self) -> Tensor(a)
at::Tensor view_as_real::call(const at::Tensor & self) {
    static auto op = create_view_as_real_typed_handle();
    return op.call(self);
}

// aten::view_as_real(Tensor(a) self) -> Tensor(a)
at::Tensor view_as_real::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_view_as_real_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(view_as_complex, name, "aten::view_as_complex")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(view_as_complex, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(view_as_complex, schema_str, "view_as_complex(Tensor(a) self) -> Tensor(a)")

// aten::view_as_complex(Tensor(a) self) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<view_as_complex::schema> create_view_as_complex_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(view_as_complex::name, view_as_complex::overload_name)
      .typed<view_as_complex::schema>();
}

// aten::view_as_complex(Tensor(a) self) -> Tensor(a)
at::Tensor view_as_complex::call(const at::Tensor & self) {
    static auto op = create_view_as_complex_typed_handle();
    return op.call(self);
}

// aten::view_as_complex(Tensor(a) self) -> Tensor(a)
at::Tensor view_as_complex::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_view_as_complex_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conj_physical, name, "aten::conj_physical")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conj_physical, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conj_physical, schema_str, "conj_physical(Tensor self) -> Tensor")

// aten::conj_physical(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<conj_physical::schema> create_conj_physical_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(conj_physical::name, conj_physical::overload_name)
      .typed<conj_physical::schema>();
}

// aten::conj_physical(Tensor self) -> Tensor
at::Tensor conj_physical::call(const at::Tensor & self) {
    static auto op = create_conj_physical_typed_handle();
    return op.call(self);
}

// aten::conj_physical(Tensor self) -> Tensor
at::Tensor conj_physical::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_conj_physical_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(acos, name, "aten::acos")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(acos, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(acos, schema_str, "acos(Tensor self) -> Tensor")

// aten::acos(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<acos::schema> create_acos_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(acos::name, acos::overload_name)
      .typed<acos::schema>();
}

// aten::acos(Tensor self) -> Tensor
at::Tensor acos::call(const at::Tensor & self) {
    static auto op = create_acos_typed_handle();
    return op.call(self);
}

// aten::acos(Tensor self) -> Tensor
at::Tensor acos::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_acos_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arccos, name, "aten::arccos")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arccos, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arccos, schema_str, "arccos(Tensor self) -> Tensor")

// aten::arccos(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<arccos::schema> create_arccos_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(arccos::name, arccos::overload_name)
      .typed<arccos::schema>();
}

// aten::arccos(Tensor self) -> Tensor
at::Tensor arccos::call(const at::Tensor & self) {
    static auto op = create_arccos_typed_handle();
    return op.call(self);
}

// aten::arccos(Tensor self) -> Tensor
at::Tensor arccos::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_arccos_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(add_out, name, "aten::add")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(add_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(add_out, schema_str, "add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)")

// aten::add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<add_out::schema> create_add_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(add_out::name, add_out::overload_name)
      .typed<add_out::schema>();
}

// aten::add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
at::Tensor & add_out::call(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
    static auto op = create_add_out_typed_handle();
    return op.call(self, other, alpha, out);
}

// aten::add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
at::Tensor & add_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
    static auto op = create_add_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, alpha, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addr_out, name, "aten::addr")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addr_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addr_out, schema_str, "addr.out(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)")

// aten::addr.out(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<addr_out::schema> create_addr_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(addr_out::name, addr_out::overload_name)
      .typed<addr_out::schema>();
}

// aten::addr.out(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
at::Tensor & addr_out::call(const at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
    static auto op = create_addr_out_typed_handle();
    return op.call(self, vec1, vec2, beta, alpha, out);
}

// aten::addr.out(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
at::Tensor & addr_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
    static auto op = create_addr_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, vec1, vec2, beta, alpha, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(all_out, name, "aten::all")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(all_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(all_out, schema_str, "all.out(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)")

// aten::all.out(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<all_out::schema> create_all_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(all_out::name, all_out::overload_name)
      .typed<all_out::schema>();
}

// aten::all.out(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & all_out::call(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & out) {
    static auto op = create_all_out_typed_handle();
    return op.call(self, dim, keepdim, out);
}

// aten::all.out(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & all_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & out) {
    static auto op = create_all_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(all_dimname_out, name, "aten::all")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(all_dimname_out, overload_name, "dimname_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(all_dimname_out, schema_str, "all.dimname_out(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)")

// aten::all.dimname_out(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<all_dimname_out::schema> create_all_dimname_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(all_dimname_out::name, all_dimname_out::overload_name)
      .typed<all_dimname_out::schema>();
}

// aten::all.dimname_out(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & all_dimname_out::call(const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & out) {
    static auto op = create_all_dimname_out_typed_handle();
    return op.call(self, dim, keepdim, out);
}

// aten::all.dimname_out(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & all_dimname_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & out) {
    static auto op = create_all_dimname_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arange_start_step, name, "aten::arange")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arange_start_step, overload_name, "start_step")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arange_start_step, schema_str, "arange.start_step(Scalar start, Scalar end, Scalar step, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::arange.start_step(Scalar start, Scalar end, Scalar step, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<arange_start_step::schema> create_arange_start_step_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(arange_start_step::name, arange_start_step::overload_name)
      .typed<arange_start_step::schema>();
}

// aten::arange.start_step(Scalar start, Scalar end, Scalar step, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor arange_start_step::call(const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_arange_start_step_typed_handle();
    return op.call(start, end, step, dtype, layout, device, pin_memory);
}

// aten::arange.start_step(Scalar start, Scalar end, Scalar step, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor arange_start_step::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_arange_start_step_typed_handle();
    return op.redispatch(dispatchKeySet, start, end, step, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(argmin_out, name, "aten::argmin")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(argmin_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(argmin_out, schema_str, "argmin.out(Tensor self, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)")

// aten::argmin.out(Tensor self, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<argmin_out::schema> create_argmin_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(argmin_out::name, argmin_out::overload_name)
      .typed<argmin_out::schema>();
}

// aten::argmin.out(Tensor self, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & argmin_out::call(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out) {
    static auto op = create_argmin_out_typed_handle();
    return op.call(self, dim, keepdim, out);
}

// aten::argmin.out(Tensor self, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & argmin_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out) {
    static auto op = create_argmin_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arccosh, name, "aten::arccosh")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arccosh, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arccosh, schema_str, "arccosh(Tensor self) -> Tensor")

// aten::arccosh(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<arccosh::schema> create_arccosh_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(arccosh::name, arccosh::overload_name)
      .typed<arccosh::schema>();
}

// aten::arccosh(Tensor self) -> Tensor
at::Tensor arccosh::call(const at::Tensor & self) {
    static auto op = create_arccosh_typed_handle();
    return op.call(self);
}

// aten::arccosh(Tensor self) -> Tensor
at::Tensor arccosh::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_arccosh_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(asinh_out, name, "aten::asinh")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(asinh_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(asinh_out, schema_str, "asinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::asinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<asinh_out::schema> create_asinh_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(asinh_out::name, asinh_out::overload_name)
      .typed<asinh_out::schema>();
}

// aten::asinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & asinh_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_asinh_out_typed_handle();
    return op.call(self, out);
}

// aten::asinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & asinh_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_asinh_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arcsinh_out, name, "aten::arcsinh")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arcsinh_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arcsinh_out, schema_str, "arcsinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::arcsinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<arcsinh_out::schema> create_arcsinh_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(arcsinh_out::name, arcsinh_out::overload_name)
      .typed<arcsinh_out::schema>();
}

// aten::arcsinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & arcsinh_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_arcsinh_out_typed_handle();
    return op.call(self, out);
}

// aten::arcsinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & arcsinh_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_arcsinh_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(asin, name, "aten::asin")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(asin, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(asin, schema_str, "asin(Tensor self) -> Tensor")

// aten::asin(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<asin::schema> create_asin_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(asin::name, asin::overload_name)
      .typed<asin::schema>();
}

// aten::asin(Tensor self) -> Tensor
at::Tensor asin::call(const at::Tensor & self) {
    static auto op = create_asin_typed_handle();
    return op.call(self);
}

// aten::asin(Tensor self) -> Tensor
at::Tensor asin::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_asin_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(asin_out, name, "aten::asin")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(asin_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(asin_out, schema_str, "asin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::asin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<asin_out::schema> create_asin_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(asin_out::name, asin_out::overload_name)
      .typed<asin_out::schema>();
}

// aten::asin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & asin_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_asin_out_typed_handle();
    return op.call(self, out);
}

// aten::asin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & asin_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_asin_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arctan_, name, "aten::arctan_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arctan_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arctan_, schema_str, "arctan_(Tensor(a!) self) -> Tensor(a!)")

// aten::arctan_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<arctan_::schema> create_arctan__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(arctan_::name, arctan_::overload_name)
      .typed<arctan_::schema>();
}

// aten::arctan_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & arctan_::call(at::Tensor & self) {
    static auto op = create_arctan__typed_handle();
    return op.call(self);
}

// aten::arctan_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & arctan_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_arctan__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atleast_1d, name, "aten::atleast_1d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atleast_1d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atleast_1d, schema_str, "atleast_1d(Tensor self) -> Tensor")

// aten::atleast_1d(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<atleast_1d::schema> create_atleast_1d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(atleast_1d::name, atleast_1d::overload_name)
      .typed<atleast_1d::schema>();
}

// aten::atleast_1d(Tensor self) -> Tensor
at::Tensor atleast_1d::call(const at::Tensor & self) {
    static auto op = create_atleast_1d_typed_handle();
    return op.call(self);
}

// aten::atleast_1d(Tensor self) -> Tensor
at::Tensor atleast_1d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_atleast_1d_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atleast_2d_Sequence, name, "aten::atleast_2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atleast_2d_Sequence, overload_name, "Sequence")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atleast_2d_Sequence, schema_str, "atleast_2d.Sequence(Tensor[] tensors) -> Tensor[]")

// aten::atleast_2d.Sequence(Tensor[] tensors) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<atleast_2d_Sequence::schema> create_atleast_2d_Sequence_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(atleast_2d_Sequence::name, atleast_2d_Sequence::overload_name)
      .typed<atleast_2d_Sequence::schema>();
}

// aten::atleast_2d.Sequence(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> atleast_2d_Sequence::call(at::TensorList tensors) {
    static auto op = create_atleast_2d_Sequence_typed_handle();
    return op.call(tensors);
}

// aten::atleast_2d.Sequence(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> atleast_2d_Sequence::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create_atleast_2d_Sequence_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_not_, name, "aten::bitwise_not_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_not_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_not_, schema_str, "bitwise_not_(Tensor(a!) self) -> Tensor(a!)")

// aten::bitwise_not_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<bitwise_not_::schema> create_bitwise_not__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bitwise_not_::name, bitwise_not_::overload_name)
      .typed<bitwise_not_::schema>();
}

// aten::bitwise_not_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & bitwise_not_::call(at::Tensor & self) {
    static auto op = create_bitwise_not__typed_handle();
    return op.call(self);
}

// aten::bitwise_not_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & bitwise_not_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_bitwise_not__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(copysign_out, name, "aten::copysign")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(copysign_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(copysign_out, schema_str, "copysign.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::copysign.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<copysign_out::schema> create_copysign_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(copysign_out::name, copysign_out::overload_name)
      .typed<copysign_out::schema>();
}

// aten::copysign.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & copysign_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_copysign_out_typed_handle();
    return op.call(self, other, out);
}

// aten::copysign.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & copysign_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_copysign_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(copysign_Scalar, name, "aten::copysign")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(copysign_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(copysign_Scalar, schema_str, "copysign.Scalar(Tensor self, Scalar other) -> Tensor")

// aten::copysign.Scalar(Tensor self, Scalar other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<copysign_Scalar::schema> create_copysign_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(copysign_Scalar::name, copysign_Scalar::overload_name)
      .typed<copysign_Scalar::schema>();
}

// aten::copysign.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor copysign_Scalar::call(const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_copysign_Scalar_typed_handle();
    return op.call(self, other);
}

// aten::copysign.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor copysign_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_copysign_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logical_xor, name, "aten::logical_xor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logical_xor, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logical_xor, schema_str, "logical_xor(Tensor self, Tensor other) -> Tensor")

// aten::logical_xor(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<logical_xor::schema> create_logical_xor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(logical_xor::name, logical_xor::overload_name)
      .typed<logical_xor::schema>();
}

// aten::logical_xor(Tensor self, Tensor other) -> Tensor
at::Tensor logical_xor::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_logical_xor_typed_handle();
    return op.call(self, other);
}

// aten::logical_xor(Tensor self, Tensor other) -> Tensor
at::Tensor logical_xor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_logical_xor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(broadcast_to, name, "aten::broadcast_to")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(broadcast_to, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(broadcast_to, schema_str, "broadcast_to(Tensor(a) self, int[] size) -> Tensor(a)")

// aten::broadcast_to(Tensor(a) self, int[] size) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<broadcast_to::schema> create_broadcast_to_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(broadcast_to::name, broadcast_to::overload_name)
      .typed<broadcast_to::schema>();
}

// aten::broadcast_to(Tensor(a) self, int[] size) -> Tensor(a)
at::Tensor broadcast_to::call(const at::Tensor & self, at::IntArrayRef size) {
    static auto op = create_broadcast_to_typed_handle();
    return op.call(self, size);
}

// aten::broadcast_to(Tensor(a) self, int[] size) -> Tensor(a)
at::Tensor broadcast_to::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef size) {
    static auto op = create_broadcast_to_typed_handle();
    return op.redispatch(dispatchKeySet, self, size);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(concat_out, name, "aten::concat")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(concat_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(concat_out, schema_str, "concat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)")

// aten::concat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<concat_out::schema> create_concat_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(concat_out::name, concat_out::overload_name)
      .typed<concat_out::schema>();
}

// aten::concat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & concat_out::call(at::TensorList tensors, int64_t dim, at::Tensor & out) {
    static auto op = create_concat_out_typed_handle();
    return op.call(tensors, dim, out);
}

// aten::concat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & concat_out::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, int64_t dim, at::Tensor & out) {
    static auto op = create_concat_out_typed_handle();
    return op.redispatch(dispatchKeySet, tensors, dim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(concat_names_out, name, "aten::concat")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(concat_names_out, overload_name, "names_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(concat_names_out, schema_str, "concat.names_out(Tensor[] tensors, Dimname dim, *, Tensor(a!) out) -> Tensor(a!)")

// aten::concat.names_out(Tensor[] tensors, Dimname dim, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<concat_names_out::schema> create_concat_names_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(concat_names_out::name, concat_names_out::overload_name)
      .typed<concat_names_out::schema>();
}

// aten::concat.names_out(Tensor[] tensors, Dimname dim, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & concat_names_out::call(at::TensorList tensors, at::Dimname dim, at::Tensor & out) {
    static auto op = create_concat_names_out_typed_handle();
    return op.call(tensors, dim, out);
}

// aten::concat.names_out(Tensor[] tensors, Dimname dim, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & concat_names_out::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, at::Dimname dim, at::Tensor & out) {
    static auto op = create_concat_names_out_typed_handle();
    return op.redispatch(dispatchKeySet, tensors, dim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ceil_out, name, "aten::ceil")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ceil_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ceil_out, schema_str, "ceil.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::ceil.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<ceil_out::schema> create_ceil_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(ceil_out::name, ceil_out::overload_name)
      .typed<ceil_out::schema>();
}

// aten::ceil.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & ceil_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_ceil_out_typed_handle();
    return op.call(self, out);
}

// aten::ceil.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & ceil_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_ceil_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_out, name, "aten::clamp")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_out, schema_str, "clamp.out(Tensor self, Scalar? min=None, Scalar? max=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::clamp.out(Tensor self, Scalar? min=None, Scalar? max=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<clamp_out::schema> create_clamp_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(clamp_out::name, clamp_out::overload_name)
      .typed<clamp_out::schema>();
}

// aten::clamp.out(Tensor self, Scalar? min=None, Scalar? max=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & clamp_out::call(const at::Tensor & self, const c10::optional<at::Scalar> & min, const c10::optional<at::Scalar> & max, at::Tensor & out) {
    static auto op = create_clamp_out_typed_handle();
    return op.call(self, min, max, out);
}

// aten::clamp.out(Tensor self, Scalar? min=None, Scalar? max=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & clamp_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Scalar> & min, const c10::optional<at::Scalar> & max, at::Tensor & out) {
    static auto op = create_clamp_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, min, max, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_min_Tensor, name, "aten::clamp_min")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_min_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_min_Tensor, schema_str, "clamp_min.Tensor(Tensor self, Tensor min) -> Tensor")

// aten::clamp_min.Tensor(Tensor self, Tensor min) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<clamp_min_Tensor::schema> create_clamp_min_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(clamp_min_Tensor::name, clamp_min_Tensor::overload_name)
      .typed<clamp_min_Tensor::schema>();
}

// aten::clamp_min.Tensor(Tensor self, Tensor min) -> Tensor
at::Tensor clamp_min_Tensor::call(const at::Tensor & self, const at::Tensor & min) {
    static auto op = create_clamp_min_Tensor_typed_handle();
    return op.call(self, min);
}

// aten::clamp_min.Tensor(Tensor self, Tensor min) -> Tensor
at::Tensor clamp_min_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & min) {
    static auto op = create_clamp_min_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, min);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_min__Tensor, name, "aten::clamp_min_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_min__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_min__Tensor, schema_str, "clamp_min_.Tensor(Tensor(a!) self, Tensor min) -> Tensor(a!)")

// aten::clamp_min_.Tensor(Tensor(a!) self, Tensor min) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<clamp_min__Tensor::schema> create_clamp_min__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(clamp_min__Tensor::name, clamp_min__Tensor::overload_name)
      .typed<clamp_min__Tensor::schema>();
}

// aten::clamp_min_.Tensor(Tensor(a!) self, Tensor min) -> Tensor(a!)
at::Tensor & clamp_min__Tensor::call(at::Tensor & self, const at::Tensor & min) {
    static auto op = create_clamp_min__Tensor_typed_handle();
    return op.call(self, min);
}

// aten::clamp_min_.Tensor(Tensor(a!) self, Tensor min) -> Tensor(a!)
at::Tensor & clamp_min__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & min) {
    static auto op = create_clamp_min__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, min);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(constant_pad_nd, name, "aten::constant_pad_nd")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(constant_pad_nd, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(constant_pad_nd, schema_str, "constant_pad_nd(Tensor self, int[] pad, Scalar value=0) -> Tensor")

// aten::constant_pad_nd(Tensor self, int[] pad, Scalar value=0) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<constant_pad_nd::schema> create_constant_pad_nd_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(constant_pad_nd::name, constant_pad_nd::overload_name)
      .typed<constant_pad_nd::schema>();
}

// aten::constant_pad_nd(Tensor self, int[] pad, Scalar value=0) -> Tensor
at::Tensor constant_pad_nd::call(const at::Tensor & self, at::IntArrayRef pad, const at::Scalar & value) {
    static auto op = create_constant_pad_nd_typed_handle();
    return op.call(self, pad, value);
}

// aten::constant_pad_nd(Tensor self, int[] pad, Scalar value=0) -> Tensor
at::Tensor constant_pad_nd::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef pad, const at::Scalar & value) {
    static auto op = create_constant_pad_nd_typed_handle();
    return op.redispatch(dispatchKeySet, self, pad, value);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(contiguous, name, "aten::contiguous")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(contiguous, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(contiguous, schema_str, "contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format) -> Tensor(a)")

// aten::contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<contiguous::schema> create_contiguous_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(contiguous::name, contiguous::overload_name)
      .typed<contiguous::schema>();
}

// aten::contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format) -> Tensor(a)
at::Tensor contiguous::call(const at::Tensor & self, at::MemoryFormat memory_format) {
    static auto op = create_contiguous_typed_handle();
    return op.call(self, memory_format);
}

// aten::contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format) -> Tensor(a)
at::Tensor contiguous::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::MemoryFormat memory_format) {
    static auto op = create_contiguous_typed_handle();
    return op.redispatch(dispatchKeySet, self, memory_format);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(convolution_overrideable, name, "aten::convolution_overrideable")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(convolution_overrideable, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(convolution_overrideable, schema_str, "convolution_overrideable(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor")

// aten::convolution_overrideable(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<convolution_overrideable::schema> create_convolution_overrideable_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(convolution_overrideable::name, convolution_overrideable::overload_name)
      .typed<convolution_overrideable::schema>();
}

// aten::convolution_overrideable(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor
at::Tensor convolution_overrideable::call(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups) {
    static auto op = create_convolution_overrideable_typed_handle();
    return op.call(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
}

// aten::convolution_overrideable(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor
at::Tensor convolution_overrideable::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups) {
    static auto op = create_convolution_overrideable_typed_handle();
    return op.redispatch(dispatchKeySet, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_convolution_deprecated, name, "aten::_convolution")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_convolution_deprecated, overload_name, "deprecated")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_convolution_deprecated, schema_str, "_convolution.deprecated(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled) -> Tensor")

// aten::_convolution.deprecated(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_convolution_deprecated::schema> create__convolution_deprecated_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_convolution_deprecated::name, _convolution_deprecated::overload_name)
      .typed<_convolution_deprecated::schema>();
}

// aten::_convolution.deprecated(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled) -> Tensor
at::Tensor _convolution_deprecated::call(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled) {
    static auto op = create__convolution_deprecated_typed_handle();
    return op.call(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled);
}

// aten::_convolution.deprecated(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled) -> Tensor
at::Tensor _convolution_deprecated::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled) {
    static auto op = create__convolution_deprecated_typed_handle();
    return op.redispatch(dispatchKeySet, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_convolution_double_backward, name, "aten::_convolution_double_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_convolution_double_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_convolution_double_backward, schema_str, "_convolution_double_backward(Tensor? ggI, Tensor? ggW, Tensor? ggb, Tensor gO, Tensor weight, Tensor self, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32, bool[3] output_mask) -> (Tensor, Tensor, Tensor)")

// aten::_convolution_double_backward(Tensor? ggI, Tensor? ggW, Tensor? ggb, Tensor gO, Tensor weight, Tensor self, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<_convolution_double_backward::schema> create__convolution_double_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_convolution_double_backward::name, _convolution_double_backward::overload_name)
      .typed<_convolution_double_backward::schema>();
}

// aten::_convolution_double_backward(Tensor? ggI, Tensor? ggW, Tensor? ggb, Tensor gO, Tensor weight, Tensor self, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _convolution_double_backward::call(const c10::optional<at::Tensor> & ggI, const c10::optional<at::Tensor> & ggW, const c10::optional<at::Tensor> & ggb, const at::Tensor & gO, const at::Tensor & weight, const at::Tensor & self, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32, ::std::array<bool,3> output_mask) {
    static auto op = create__convolution_double_backward_typed_handle();
    return op.call(ggI, ggW, ggb, gO, weight, self, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32, output_mask);
}

// aten::_convolution_double_backward(Tensor? ggI, Tensor? ggW, Tensor? ggb, Tensor gO, Tensor weight, Tensor self, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _convolution_double_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const c10::optional<at::Tensor> & ggI, const c10::optional<at::Tensor> & ggW, const c10::optional<at::Tensor> & ggb, const at::Tensor & gO, const at::Tensor & weight, const at::Tensor & self, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32, ::std::array<bool,3> output_mask) {
    static auto op = create__convolution_double_backward_typed_handle();
    return op.redispatch(dispatchKeySet, ggI, ggW, ggb, gO, weight, self, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32, output_mask);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv2d, name, "aten::conv2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv2d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv2d, schema_str, "conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor")

// aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<conv2d::schema> create_conv2d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(conv2d::name, conv2d::overload_name)
      .typed<conv2d::schema>();
}

// aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor
at::Tensor conv2d::call(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
    static auto op = create_conv2d_typed_handle();
    return op.call(input, weight, bias, stride, padding, dilation, groups);
}

// aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor
at::Tensor conv2d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
    static auto op = create_conv2d_typed_handle();
    return op.redispatch(dispatchKeySet, input, weight, bias, stride, padding, dilation, groups);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_copy_from, name, "aten::_copy_from")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_copy_from, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_copy_from, schema_str, "_copy_from(Tensor self, Tensor dst, bool non_blocking=False) -> Tensor")

// aten::_copy_from(Tensor self, Tensor dst, bool non_blocking=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_copy_from::schema> create__copy_from_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_copy_from::name, _copy_from::overload_name)
      .typed<_copy_from::schema>();
}

// aten::_copy_from(Tensor self, Tensor dst, bool non_blocking=False) -> Tensor
at::Tensor _copy_from::call(const at::Tensor & self, const at::Tensor & dst, bool non_blocking) {
    static auto op = create__copy_from_typed_handle();
    return op.call(self, dst, non_blocking);
}

// aten::_copy_from(Tensor self, Tensor dst, bool non_blocking=False) -> Tensor
at::Tensor _copy_from::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & dst, bool non_blocking) {
    static auto op = create__copy_from_typed_handle();
    return op.redispatch(dispatchKeySet, self, dst, non_blocking);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(corrcoef, name, "aten::corrcoef")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(corrcoef, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(corrcoef, schema_str, "corrcoef(Tensor self) -> Tensor")

// aten::corrcoef(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<corrcoef::schema> create_corrcoef_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(corrcoef::name, corrcoef::overload_name)
      .typed<corrcoef::schema>();
}

// aten::corrcoef(Tensor self) -> Tensor
at::Tensor corrcoef::call(const at::Tensor & self) {
    static auto op = create_corrcoef_typed_handle();
    return op.call(self);
}

// aten::corrcoef(Tensor self) -> Tensor
at::Tensor corrcoef::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_corrcoef_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_batch_norm, name, "aten::cudnn_batch_norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_batch_norm, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_batch_norm, schema_str, "cudnn_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float exponential_average_factor, float epsilon) -> (Tensor, Tensor, Tensor, Tensor)")

// aten::cudnn_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float exponential_average_factor, float epsilon) -> (Tensor, Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<cudnn_batch_norm::schema> create_cudnn_batch_norm_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cudnn_batch_norm::name, cudnn_batch_norm::overload_name)
      .typed<cudnn_batch_norm::schema>();
}

// aten::cudnn_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float exponential_average_factor, float epsilon) -> (Tensor, Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> cudnn_batch_norm::call(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double exponential_average_factor, double epsilon) {
    static auto op = create_cudnn_batch_norm_typed_handle();
    return op.call(input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
}

// aten::cudnn_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float exponential_average_factor, float epsilon) -> (Tensor, Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> cudnn_batch_norm::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double exponential_average_factor, double epsilon) {
    static auto op = create_cudnn_batch_norm_typed_handle();
    return op.redispatch(dispatchKeySet, input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_transpose_backward, name, "aten::cudnn_convolution_transpose_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_transpose_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_transpose_backward, schema_str, "cudnn_convolution_transpose_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool allow_tf32, bool[2] output_mask) -> (Tensor, Tensor)")

// aten::cudnn_convolution_transpose_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool allow_tf32, bool[2] output_mask) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<cudnn_convolution_transpose_backward::schema> create_cudnn_convolution_transpose_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cudnn_convolution_transpose_backward::name, cudnn_convolution_transpose_backward::overload_name)
      .typed<cudnn_convolution_transpose_backward::schema>();
}

// aten::cudnn_convolution_transpose_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool allow_tf32, bool[2] output_mask) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> cudnn_convolution_transpose_backward::call(const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32, ::std::array<bool,2> output_mask) {
    static auto op = create_cudnn_convolution_transpose_backward_typed_handle();
    return op.call(self, grad_output, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, allow_tf32, output_mask);
}

// aten::cudnn_convolution_transpose_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool allow_tf32, bool[2] output_mask) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> cudnn_convolution_transpose_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32, ::std::array<bool,2> output_mask) {
    static auto op = create_cudnn_convolution_transpose_backward_typed_handle();
    return op.redispatch(dispatchKeySet, self, grad_output, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, allow_tf32, output_mask);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_transpose_backward_input, name, "aten::cudnn_convolution_transpose_backward_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_transpose_backward_input, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_transpose_backward_input, schema_str, "cudnn_convolution_transpose_backward_input(Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool allow_tf32) -> Tensor")

// aten::cudnn_convolution_transpose_backward_input(Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool allow_tf32) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<cudnn_convolution_transpose_backward_input::schema> create_cudnn_convolution_transpose_backward_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cudnn_convolution_transpose_backward_input::name, cudnn_convolution_transpose_backward_input::overload_name)
      .typed<cudnn_convolution_transpose_backward_input::schema>();
}

// aten::cudnn_convolution_transpose_backward_input(Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool allow_tf32) -> Tensor
at::Tensor cudnn_convolution_transpose_backward_input::call(const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
    static auto op = create_cudnn_convolution_transpose_backward_input_typed_handle();
    return op.call(grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
}

// aten::cudnn_convolution_transpose_backward_input(Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool allow_tf32) -> Tensor
at::Tensor cudnn_convolution_transpose_backward_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
    static auto op = create_cudnn_convolution_transpose_backward_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cummin_out, name, "aten::cummin")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cummin_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cummin_out, schema_str, "cummin.out(Tensor self, int dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)")

// aten::cummin.out(Tensor self, int dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
static C10_NOINLINE c10::TypedOperatorHandle<cummin_out::schema> create_cummin_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cummin_out::name, cummin_out::overload_name)
      .typed<cummin_out::schema>();
}

// aten::cummin.out(Tensor self, int dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> cummin_out::call(const at::Tensor & self, int64_t dim, at::Tensor & values, at::Tensor & indices) {
    static auto op = create_cummin_out_typed_handle();
    return op.call(self, dim, values, indices);
}

// aten::cummin.out(Tensor self, int dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> cummin_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, at::Tensor & values, at::Tensor & indices) {
    static auto op = create_cummin_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, values, indices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cummaxmin_backward, name, "aten::cummaxmin_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cummaxmin_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cummaxmin_backward, schema_str, "cummaxmin_backward(Tensor grad, Tensor input, Tensor indices, int dim) -> Tensor")

// aten::cummaxmin_backward(Tensor grad, Tensor input, Tensor indices, int dim) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<cummaxmin_backward::schema> create_cummaxmin_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cummaxmin_backward::name, cummaxmin_backward::overload_name)
      .typed<cummaxmin_backward::schema>();
}

// aten::cummaxmin_backward(Tensor grad, Tensor input, Tensor indices, int dim) -> Tensor
at::Tensor cummaxmin_backward::call(const at::Tensor & grad, const at::Tensor & input, const at::Tensor & indices, int64_t dim) {
    static auto op = create_cummaxmin_backward_typed_handle();
    return op.call(grad, input, indices, dim);
}

// aten::cummaxmin_backward(Tensor grad, Tensor input, Tensor indices, int dim) -> Tensor
at::Tensor cummaxmin_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & input, const at::Tensor & indices, int64_t dim) {
    static auto op = create_cummaxmin_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad, input, indices, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumprod_dimname_out, name, "aten::cumprod")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumprod_dimname_out, overload_name, "dimname_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumprod_dimname_out, schema_str, "cumprod.dimname_out(Tensor self, Dimname dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)")

// aten::cumprod.dimname_out(Tensor self, Dimname dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<cumprod_dimname_out::schema> create_cumprod_dimname_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cumprod_dimname_out::name, cumprod_dimname_out::overload_name)
      .typed<cumprod_dimname_out::schema>();
}

// aten::cumprod.dimname_out(Tensor self, Dimname dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & cumprod_dimname_out::call(const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
    static auto op = create_cumprod_dimname_out_typed_handle();
    return op.call(self, dim, dtype, out);
}

// aten::cumprod.dimname_out(Tensor self, Dimname dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & cumprod_dimname_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
    static auto op = create_cumprod_dimname_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, dtype, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumprod_backward, name, "aten::cumprod_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumprod_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumprod_backward, schema_str, "cumprod_backward(Tensor grad, Tensor input, int dim, Tensor output) -> Tensor")

// aten::cumprod_backward(Tensor grad, Tensor input, int dim, Tensor output) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<cumprod_backward::schema> create_cumprod_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cumprod_backward::name, cumprod_backward::overload_name)
      .typed<cumprod_backward::schema>();
}

// aten::cumprod_backward(Tensor grad, Tensor input, int dim, Tensor output) -> Tensor
at::Tensor cumprod_backward::call(const at::Tensor & grad, const at::Tensor & input, int64_t dim, const at::Tensor & output) {
    static auto op = create_cumprod_backward_typed_handle();
    return op.call(grad, input, dim, output);
}

// aten::cumprod_backward(Tensor grad, Tensor input, int dim, Tensor output) -> Tensor
at::Tensor cumprod_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & input, int64_t dim, const at::Tensor & output) {
    static auto op = create_cumprod_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad, input, dim, output);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumulative_trapezoid_dx, name, "aten::cumulative_trapezoid")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumulative_trapezoid_dx, overload_name, "dx")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumulative_trapezoid_dx, schema_str, "cumulative_trapezoid.dx(Tensor y, *, Scalar dx=1, int dim=-1) -> Tensor")

// aten::cumulative_trapezoid.dx(Tensor y, *, Scalar dx=1, int dim=-1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<cumulative_trapezoid_dx::schema> create_cumulative_trapezoid_dx_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cumulative_trapezoid_dx::name, cumulative_trapezoid_dx::overload_name)
      .typed<cumulative_trapezoid_dx::schema>();
}

// aten::cumulative_trapezoid.dx(Tensor y, *, Scalar dx=1, int dim=-1) -> Tensor
at::Tensor cumulative_trapezoid_dx::call(const at::Tensor & y, const at::Scalar & dx, int64_t dim) {
    static auto op = create_cumulative_trapezoid_dx_typed_handle();
    return op.call(y, dx, dim);
}

// aten::cumulative_trapezoid.dx(Tensor y, *, Scalar dx=1, int dim=-1) -> Tensor
at::Tensor cumulative_trapezoid_dx::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & y, const at::Scalar & dx, int64_t dim) {
    static auto op = create_cumulative_trapezoid_dx_typed_handle();
    return op.redispatch(dispatchKeySet, y, dx, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(diagonal_Dimname, name, "aten::diagonal")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(diagonal_Dimname, overload_name, "Dimname")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(diagonal_Dimname, schema_str, "diagonal.Dimname(Tensor(a) self, *, Dimname outdim, Dimname dim1, Dimname dim2, int offset=0) -> Tensor(a)")

// aten::diagonal.Dimname(Tensor(a) self, *, Dimname outdim, Dimname dim1, Dimname dim2, int offset=0) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<diagonal_Dimname::schema> create_diagonal_Dimname_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(diagonal_Dimname::name, diagonal_Dimname::overload_name)
      .typed<diagonal_Dimname::schema>();
}

// aten::diagonal.Dimname(Tensor(a) self, *, Dimname outdim, Dimname dim1, Dimname dim2, int offset=0) -> Tensor(a)
at::Tensor diagonal_Dimname::call(const at::Tensor & self, at::Dimname outdim, at::Dimname dim1, at::Dimname dim2, int64_t offset) {
    static auto op = create_diagonal_Dimname_typed_handle();
    return op.call(self, outdim, dim1, dim2, offset);
}

// aten::diagonal.Dimname(Tensor(a) self, *, Dimname outdim, Dimname dim1, Dimname dim2, int offset=0) -> Tensor(a)
at::Tensor diagonal_Dimname::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname outdim, at::Dimname dim1, at::Dimname dim2, int64_t offset) {
    static auto op = create_diagonal_Dimname_typed_handle();
    return op.redispatch(dispatchKeySet, self, outdim, dim1, dim2, offset);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(div_Tensor, name, "aten::div")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(div_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(div_Tensor, schema_str, "div.Tensor(Tensor self, Tensor other) -> Tensor")

// aten::div.Tensor(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<div_Tensor::schema> create_div_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(div_Tensor::name, div_Tensor::overload_name)
      .typed<div_Tensor::schema>();
}

// aten::div.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor div_Tensor::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_div_Tensor_typed_handle();
    return op.call(self, other);
}

// aten::div.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor div_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_div_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(divide__Tensor, name, "aten::divide_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(divide__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(divide__Tensor, schema_str, "divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<divide__Tensor::schema> create_divide__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(divide__Tensor::name, divide__Tensor::overload_name)
      .typed<divide__Tensor::schema>();
}

// aten::divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & divide__Tensor::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create_divide__Tensor_typed_handle();
    return op.call(self, other);
}

// aten::divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & divide__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create_divide__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(divide_Tensor_mode, name, "aten::divide")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(divide_Tensor_mode, overload_name, "Tensor_mode")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(divide_Tensor_mode, schema_str, "divide.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor")

// aten::divide.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<divide_Tensor_mode::schema> create_divide_Tensor_mode_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(divide_Tensor_mode::name, divide_Tensor_mode::overload_name)
      .typed<divide_Tensor_mode::schema>();
}

// aten::divide.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor
at::Tensor divide_Tensor_mode::call(const at::Tensor & self, const at::Tensor & other, c10::optional<c10::string_view> rounding_mode) {
    static auto op = create_divide_Tensor_mode_typed_handle();
    return op.call(self, other, rounding_mode);
}

// aten::divide.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor
at::Tensor divide_Tensor_mode::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, c10::optional<c10::string_view> rounding_mode) {
    static auto op = create_divide_Tensor_mode_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, rounding_mode);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(divide__Tensor_mode, name, "aten::divide_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(divide__Tensor_mode, overload_name, "Tensor_mode")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(divide__Tensor_mode, schema_str, "divide_.Tensor_mode(Tensor(a!) self, Tensor other, *, str? rounding_mode) -> Tensor(a!)")

// aten::divide_.Tensor_mode(Tensor(a!) self, Tensor other, *, str? rounding_mode) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<divide__Tensor_mode::schema> create_divide__Tensor_mode_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(divide__Tensor_mode::name, divide__Tensor_mode::overload_name)
      .typed<divide__Tensor_mode::schema>();
}

// aten::divide_.Tensor_mode(Tensor(a!) self, Tensor other, *, str? rounding_mode) -> Tensor(a!)
at::Tensor & divide__Tensor_mode::call(at::Tensor & self, const at::Tensor & other, c10::optional<c10::string_view> rounding_mode) {
    static auto op = create_divide__Tensor_mode_typed_handle();
    return op.call(self, other, rounding_mode);
}

// aten::divide_.Tensor_mode(Tensor(a!) self, Tensor other, *, str? rounding_mode) -> Tensor(a!)
at::Tensor & divide__Tensor_mode::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other, c10::optional<c10::string_view> rounding_mode) {
    static auto op = create_divide__Tensor_mode_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, rounding_mode);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(true_divide__Scalar, name, "aten::true_divide_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(true_divide__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(true_divide__Scalar, schema_str, "true_divide_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)")

// aten::true_divide_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<true_divide__Scalar::schema> create_true_divide__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(true_divide__Scalar::name, true_divide__Scalar::overload_name)
      .typed<true_divide__Scalar::schema>();
}

// aten::true_divide_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & true_divide__Scalar::call(at::Tensor & self, const at::Scalar & other) {
    static auto op = create_true_divide__Scalar_typed_handle();
    return op.call(self, other);
}

// aten::true_divide_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & true_divide__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
    static auto op = create_true_divide__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(dot_out, name, "aten::dot")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(dot_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(dot_out, schema_str, "dot.out(Tensor self, Tensor tensor, *, Tensor(a!) out) -> Tensor(a!)")

// aten::dot.out(Tensor self, Tensor tensor, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<dot_out::schema> create_dot_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(dot_out::name, dot_out::overload_name)
      .typed<dot_out::schema>();
}

// aten::dot.out(Tensor self, Tensor tensor, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & dot_out::call(const at::Tensor & self, const at::Tensor & tensor, at::Tensor & out) {
    static auto op = create_dot_out_typed_handle();
    return op.call(self, tensor, out);
}

// aten::dot.out(Tensor self, Tensor tensor, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & dot_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & tensor, at::Tensor & out) {
    static auto op = create_dot_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, tensor, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(vdot_out, name, "aten::vdot")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(vdot_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(vdot_out, schema_str, "vdot.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::vdot.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<vdot_out::schema> create_vdot_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(vdot_out::name, vdot_out::overload_name)
      .typed<vdot_out::schema>();
}

// aten::vdot.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & vdot_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_vdot_out_typed_handle();
    return op.call(self, other, out);
}

// aten::vdot.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & vdot_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_vdot_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(embedding, name, "aten::embedding")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(embedding, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(embedding, schema_str, "embedding(Tensor weight, Tensor indices, int padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor")

// aten::embedding(Tensor weight, Tensor indices, int padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<embedding::schema> create_embedding_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(embedding::name, embedding::overload_name)
      .typed<embedding::schema>();
}

// aten::embedding(Tensor weight, Tensor indices, int padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor
at::Tensor embedding::call(const at::Tensor & weight, const at::Tensor & indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
    static auto op = create_embedding_typed_handle();
    return op.call(weight, indices, padding_idx, scale_grad_by_freq, sparse);
}

// aten::embedding(Tensor weight, Tensor indices, int padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor
at::Tensor embedding::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & weight, const at::Tensor & indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
    static auto op = create_embedding_typed_handle();
    return op.redispatch(dispatchKeySet, weight, indices, padding_idx, scale_grad_by_freq, sparse);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_rowwise_prune, name, "aten::_rowwise_prune")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_rowwise_prune, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_rowwise_prune, schema_str, "_rowwise_prune(Tensor weight, Tensor mask, ScalarType compressed_indices_dtype) -> (Tensor, Tensor)")

// aten::_rowwise_prune(Tensor weight, Tensor mask, ScalarType compressed_indices_dtype) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<_rowwise_prune::schema> create__rowwise_prune_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_rowwise_prune::name, _rowwise_prune::overload_name)
      .typed<_rowwise_prune::schema>();
}

// aten::_rowwise_prune(Tensor weight, Tensor mask, ScalarType compressed_indices_dtype) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> _rowwise_prune::call(const at::Tensor & weight, const at::Tensor & mask, at::ScalarType compressed_indices_dtype) {
    static auto op = create__rowwise_prune_typed_handle();
    return op.call(weight, mask, compressed_indices_dtype);
}

// aten::_rowwise_prune(Tensor weight, Tensor mask, ScalarType compressed_indices_dtype) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> _rowwise_prune::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & weight, const at::Tensor & mask, at::ScalarType compressed_indices_dtype) {
    static auto op = create__rowwise_prune_typed_handle();
    return op.redispatch(dispatchKeySet, weight, mask, compressed_indices_dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(row_stack, name, "aten::row_stack")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(row_stack, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(row_stack, schema_str, "row_stack(Tensor[] tensors) -> Tensor")

// aten::row_stack(Tensor[] tensors) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<row_stack::schema> create_row_stack_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(row_stack::name, row_stack::overload_name)
      .typed<row_stack::schema>();
}

// aten::row_stack(Tensor[] tensors) -> Tensor
at::Tensor row_stack::call(at::TensorList tensors) {
    static auto op = create_row_stack_typed_handle();
    return op.call(tensors);
}

// aten::row_stack(Tensor[] tensors) -> Tensor
at::Tensor row_stack::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create_row_stack_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_embedding_bag_backward, name, "aten::_embedding_bag_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_embedding_bag_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_embedding_bag_backward, schema_str, "_embedding_bag_backward(Tensor grad, Tensor indices, Tensor offsets, Tensor offset2bag, Tensor bag_size, Tensor maximum_indices, int num_weights, bool scale_grad_by_freq, int mode, bool sparse, Tensor? per_sample_weights, int padding_idx=-1) -> Tensor")

// aten::_embedding_bag_backward(Tensor grad, Tensor indices, Tensor offsets, Tensor offset2bag, Tensor bag_size, Tensor maximum_indices, int num_weights, bool scale_grad_by_freq, int mode, bool sparse, Tensor? per_sample_weights, int padding_idx=-1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_embedding_bag_backward::schema> create__embedding_bag_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_embedding_bag_backward::name, _embedding_bag_backward::overload_name)
      .typed<_embedding_bag_backward::schema>();
}

// aten::_embedding_bag_backward(Tensor grad, Tensor indices, Tensor offsets, Tensor offset2bag, Tensor bag_size, Tensor maximum_indices, int num_weights, bool scale_grad_by_freq, int mode, bool sparse, Tensor? per_sample_weights, int padding_idx=-1) -> Tensor
at::Tensor _embedding_bag_backward::call(const at::Tensor & grad, const at::Tensor & indices, const at::Tensor & offsets, const at::Tensor & offset2bag, const at::Tensor & bag_size, const at::Tensor & maximum_indices, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<at::Tensor> & per_sample_weights, int64_t padding_idx) {
    static auto op = create__embedding_bag_backward_typed_handle();
    return op.call(grad, indices, offsets, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, sparse, per_sample_weights, padding_idx);
}

// aten::_embedding_bag_backward(Tensor grad, Tensor indices, Tensor offsets, Tensor offset2bag, Tensor bag_size, Tensor maximum_indices, int num_weights, bool scale_grad_by_freq, int mode, bool sparse, Tensor? per_sample_weights, int padding_idx=-1) -> Tensor
at::Tensor _embedding_bag_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & indices, const at::Tensor & offsets, const at::Tensor & offset2bag, const at::Tensor & bag_size, const at::Tensor & maximum_indices, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<at::Tensor> & per_sample_weights, int64_t padding_idx) {
    static auto op = create__embedding_bag_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad, indices, offsets, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, sparse, per_sample_weights, padding_idx);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_embedding_bag_dense_backward, name, "aten::_embedding_bag_dense_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_embedding_bag_dense_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_embedding_bag_dense_backward, schema_str, "_embedding_bag_dense_backward(Tensor grad, Tensor indices, Tensor offset2bag, Tensor bag_size, Tensor maximum_indices, int num_weights, bool scale_grad_by_freq, int mode, Tensor? per_sample_weights, int padding_idx=-1) -> Tensor")

// aten::_embedding_bag_dense_backward(Tensor grad, Tensor indices, Tensor offset2bag, Tensor bag_size, Tensor maximum_indices, int num_weights, bool scale_grad_by_freq, int mode, Tensor? per_sample_weights, int padding_idx=-1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_embedding_bag_dense_backward::schema> create__embedding_bag_dense_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_embedding_bag_dense_backward::name, _embedding_bag_dense_backward::overload_name)
      .typed<_embedding_bag_dense_backward::schema>();
}

// aten::_embedding_bag_dense_backward(Tensor grad, Tensor indices, Tensor offset2bag, Tensor bag_size, Tensor maximum_indices, int num_weights, bool scale_grad_by_freq, int mode, Tensor? per_sample_weights, int padding_idx=-1) -> Tensor
at::Tensor _embedding_bag_dense_backward::call(const at::Tensor & grad, const at::Tensor & indices, const at::Tensor & offset2bag, const at::Tensor & bag_size, const at::Tensor & maximum_indices, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, const c10::optional<at::Tensor> & per_sample_weights, int64_t padding_idx) {
    static auto op = create__embedding_bag_dense_backward_typed_handle();
    return op.call(grad, indices, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, per_sample_weights, padding_idx);
}

// aten::_embedding_bag_dense_backward(Tensor grad, Tensor indices, Tensor offset2bag, Tensor bag_size, Tensor maximum_indices, int num_weights, bool scale_grad_by_freq, int mode, Tensor? per_sample_weights, int padding_idx=-1) -> Tensor
at::Tensor _embedding_bag_dense_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & indices, const at::Tensor & offset2bag, const at::Tensor & bag_size, const at::Tensor & maximum_indices, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, const c10::optional<at::Tensor> & per_sample_weights, int64_t padding_idx) {
    static auto op = create__embedding_bag_dense_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad, indices, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, per_sample_weights, padding_idx);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(erf_out, name, "aten::erf")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(erf_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(erf_out, schema_str, "erf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::erf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<erf_out::schema> create_erf_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(erf_out::name, erf_out::overload_name)
      .typed<erf_out::schema>();
}

// aten::erf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & erf_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_erf_out_typed_handle();
    return op.call(self, out);
}

// aten::erf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & erf_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_erf_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(erfc, name, "aten::erfc")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(erfc, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(erfc, schema_str, "erfc(Tensor self) -> Tensor")

// aten::erfc(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<erfc::schema> create_erfc_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(erfc::name, erfc::overload_name)
      .typed<erfc::schema>();
}

// aten::erfc(Tensor self) -> Tensor
at::Tensor erfc::call(const at::Tensor & self) {
    static auto op = create_erfc_typed_handle();
    return op.call(self);
}

// aten::erfc(Tensor self) -> Tensor
at::Tensor erfc::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_erfc_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(erfc_out, name, "aten::erfc")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(erfc_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(erfc_out, schema_str, "erfc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::erfc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<erfc_out::schema> create_erfc_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(erfc_out::name, erfc_out::overload_name)
      .typed<erfc_out::schema>();
}

// aten::erfc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & erfc_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_erfc_out_typed_handle();
    return op.call(self, out);
}

// aten::erfc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & erfc_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_erfc_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(eye_m_out, name, "aten::eye")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(eye_m_out, overload_name, "m_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(eye_m_out, schema_str, "eye.m_out(int n, int m, *, Tensor(a!) out) -> Tensor(a!)")

// aten::eye.m_out(int n, int m, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<eye_m_out::schema> create_eye_m_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(eye_m_out::name, eye_m_out::overload_name)
      .typed<eye_m_out::schema>();
}

// aten::eye.m_out(int n, int m, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & eye_m_out::call(int64_t n, int64_t m, at::Tensor & out) {
    static auto op = create_eye_m_out_typed_handle();
    return op.call(n, m, out);
}

// aten::eye.m_out(int n, int m, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & eye_m_out::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t n, int64_t m, at::Tensor & out) {
    static auto op = create_eye_m_out_typed_handle();
    return op.redispatch(dispatchKeySet, n, m, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(flatten_using_names, name, "aten::flatten")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(flatten_using_names, overload_name, "using_names")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(flatten_using_names, schema_str, "flatten.using_names(Tensor(a) self, Dimname start_dim, Dimname end_dim, Dimname out_dim) -> Tensor(a)")

// aten::flatten.using_names(Tensor(a) self, Dimname start_dim, Dimname end_dim, Dimname out_dim) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<flatten_using_names::schema> create_flatten_using_names_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(flatten_using_names::name, flatten_using_names::overload_name)
      .typed<flatten_using_names::schema>();
}

// aten::flatten.using_names(Tensor(a) self, Dimname start_dim, Dimname end_dim, Dimname out_dim) -> Tensor(a)
at::Tensor flatten_using_names::call(const at::Tensor & self, at::Dimname start_dim, at::Dimname end_dim, at::Dimname out_dim) {
    static auto op = create_flatten_using_names_typed_handle();
    return op.call(self, start_dim, end_dim, out_dim);
}

// aten::flatten.using_names(Tensor(a) self, Dimname start_dim, Dimname end_dim, Dimname out_dim) -> Tensor(a)
at::Tensor flatten_using_names::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname start_dim, at::Dimname end_dim, at::Dimname out_dim) {
    static auto op = create_flatten_using_names_typed_handle();
    return op.redispatch(dispatchKeySet, self, start_dim, end_dim, out_dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(floor_divide, name, "aten::floor_divide")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(floor_divide, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(floor_divide, schema_str, "floor_divide(Tensor self, Tensor other) -> Tensor")

// aten::floor_divide(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<floor_divide::schema> create_floor_divide_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(floor_divide::name, floor_divide::overload_name)
      .typed<floor_divide::schema>();
}

// aten::floor_divide(Tensor self, Tensor other) -> Tensor
at::Tensor floor_divide::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_floor_divide_typed_handle();
    return op.call(self, other);
}

// aten::floor_divide(Tensor self, Tensor other) -> Tensor
at::Tensor floor_divide::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_floor_divide_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(floor_divide__Tensor, name, "aten::floor_divide_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(floor_divide__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(floor_divide__Tensor, schema_str, "floor_divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::floor_divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<floor_divide__Tensor::schema> create_floor_divide__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(floor_divide__Tensor::name, floor_divide__Tensor::overload_name)
      .typed<floor_divide__Tensor::schema>();
}

// aten::floor_divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & floor_divide__Tensor::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create_floor_divide__Tensor_typed_handle();
    return op.call(self, other);
}

// aten::floor_divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & floor_divide__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create_floor_divide__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(floor_divide_out, name, "aten::floor_divide")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(floor_divide_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(floor_divide_out, schema_str, "floor_divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::floor_divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<floor_divide_out::schema> create_floor_divide_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(floor_divide_out::name, floor_divide_out::overload_name)
      .typed<floor_divide_out::schema>();
}

// aten::floor_divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & floor_divide_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_floor_divide_out_typed_handle();
    return op.call(self, other, out);
}

// aten::floor_divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & floor_divide_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_floor_divide_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(floor_divide__Scalar, name, "aten::floor_divide_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(floor_divide__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(floor_divide__Scalar, schema_str, "floor_divide_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)")

// aten::floor_divide_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<floor_divide__Scalar::schema> create_floor_divide__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(floor_divide__Scalar::name, floor_divide__Scalar::overload_name)
      .typed<floor_divide__Scalar::schema>();
}

// aten::floor_divide_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & floor_divide__Scalar::call(at::Tensor & self, const at::Scalar & other) {
    static auto op = create_floor_divide__Scalar_typed_handle();
    return op.call(self, other);
}

// aten::floor_divide_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & floor_divide__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
    static auto op = create_floor_divide__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(full, name, "aten::full")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(full, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(full, schema_str, "full(int[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::full(int[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<full::schema> create_full_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(full::name, full::overload_name)
      .typed<full::schema>();
}

// aten::full(int[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor full::call(at::IntArrayRef size, const at::Scalar & fill_value, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_full_typed_handle();
    return op.call(size, fill_value, dtype, layout, device, pin_memory);
}

// aten::full(int[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor full::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, const at::Scalar & fill_value, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_full_typed_handle();
    return op.redispatch(dispatchKeySet, size, fill_value, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(full_like, name, "aten::full_like")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(full_like, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(full_like, schema_str, "full_like(Tensor self, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor")

// aten::full_like(Tensor self, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<full_like::schema> create_full_like_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(full_like::name, full_like::overload_name)
      .typed<full_like::schema>();
}

// aten::full_like(Tensor self, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
at::Tensor full_like::call(const at::Tensor & self, const at::Scalar & fill_value, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create_full_like_typed_handle();
    return op.call(self, fill_value, dtype, layout, device, pin_memory, memory_format);
}

// aten::full_like(Tensor self, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
at::Tensor full_like::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & fill_value, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create_full_like_typed_handle();
    return op.redispatch(dispatchKeySet, self, fill_value, dtype, layout, device, pin_memory, memory_format);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(grid_sampler_2d, name, "aten::grid_sampler_2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(grid_sampler_2d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(grid_sampler_2d, schema_str, "grid_sampler_2d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor")

// aten::grid_sampler_2d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<grid_sampler_2d::schema> create_grid_sampler_2d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(grid_sampler_2d::name, grid_sampler_2d::overload_name)
      .typed<grid_sampler_2d::schema>();
}

// aten::grid_sampler_2d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor
at::Tensor grid_sampler_2d::call(const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
    static auto op = create_grid_sampler_2d_typed_handle();
    return op.call(input, grid, interpolation_mode, padding_mode, align_corners);
}

// aten::grid_sampler_2d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor
at::Tensor grid_sampler_2d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
    static auto op = create_grid_sampler_2d_typed_handle();
    return op.redispatch(dispatchKeySet, input, grid, interpolation_mode, padding_mode, align_corners);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_grid_sampler_2d_cpu_fallback_backward, name, "aten::_grid_sampler_2d_cpu_fallback_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_grid_sampler_2d_cpu_fallback_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_grid_sampler_2d_cpu_fallback_backward, schema_str, "_grid_sampler_2d_cpu_fallback_backward(Tensor grad_output, Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> (Tensor, Tensor)")

// aten::_grid_sampler_2d_cpu_fallback_backward(Tensor grad_output, Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<_grid_sampler_2d_cpu_fallback_backward::schema> create__grid_sampler_2d_cpu_fallback_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_grid_sampler_2d_cpu_fallback_backward::name, _grid_sampler_2d_cpu_fallback_backward::overload_name)
      .typed<_grid_sampler_2d_cpu_fallback_backward::schema>();
}

// aten::_grid_sampler_2d_cpu_fallback_backward(Tensor grad_output, Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> _grid_sampler_2d_cpu_fallback_backward::call(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
    static auto op = create__grid_sampler_2d_cpu_fallback_backward_typed_handle();
    return op.call(grad_output, input, grid, interpolation_mode, padding_mode, align_corners);
}

// aten::_grid_sampler_2d_cpu_fallback_backward(Tensor grad_output, Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> _grid_sampler_2d_cpu_fallback_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
    static auto op = create__grid_sampler_2d_cpu_fallback_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, input, grid, interpolation_mode, padding_mode, align_corners);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hamming_window_periodic_alpha, name, "aten::hamming_window")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hamming_window_periodic_alpha, overload_name, "periodic_alpha")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hamming_window_periodic_alpha, schema_str, "hamming_window.periodic_alpha(int window_length, bool periodic, float alpha, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::hamming_window.periodic_alpha(int window_length, bool periodic, float alpha, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<hamming_window_periodic_alpha::schema> create_hamming_window_periodic_alpha_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(hamming_window_periodic_alpha::name, hamming_window_periodic_alpha::overload_name)
      .typed<hamming_window_periodic_alpha::schema>();
}

// aten::hamming_window.periodic_alpha(int window_length, bool periodic, float alpha, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor hamming_window_periodic_alpha::call(int64_t window_length, bool periodic, double alpha, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_hamming_window_periodic_alpha_typed_handle();
    return op.call(window_length, periodic, alpha, dtype, layout, device, pin_memory);
}

// aten::hamming_window.periodic_alpha(int window_length, bool periodic, float alpha, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor hamming_window_periodic_alpha::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t window_length, bool periodic, double alpha, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_hamming_window_periodic_alpha_typed_handle();
    return op.redispatch(dispatchKeySet, window_length, periodic, alpha, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(kaiser_window, name, "aten::kaiser_window")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(kaiser_window, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(kaiser_window, schema_str, "kaiser_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::kaiser_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<kaiser_window::schema> create_kaiser_window_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(kaiser_window::name, kaiser_window::overload_name)
      .typed<kaiser_window::schema>();
}

// aten::kaiser_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor kaiser_window::call(int64_t window_length, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_kaiser_window_typed_handle();
    return op.call(window_length, dtype, layout, device, pin_memory);
}

// aten::kaiser_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor kaiser_window::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t window_length, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_kaiser_window_typed_handle();
    return op.redispatch(dispatchKeySet, window_length, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fft_c2r, name, "aten::_fft_c2r")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fft_c2r, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fft_c2r, schema_str, "_fft_c2r(Tensor self, int[] dim, int normalization, int last_dim_size) -> Tensor")

// aten::_fft_c2r(Tensor self, int[] dim, int normalization, int last_dim_size) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_fft_c2r::schema> create__fft_c2r_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_fft_c2r::name, _fft_c2r::overload_name)
      .typed<_fft_c2r::schema>();
}

// aten::_fft_c2r(Tensor self, int[] dim, int normalization, int last_dim_size) -> Tensor
at::Tensor _fft_c2r::call(const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, int64_t last_dim_size) {
    static auto op = create__fft_c2r_typed_handle();
    return op.call(self, dim, normalization, last_dim_size);
}

// aten::_fft_c2r(Tensor self, int[] dim, int normalization, int last_dim_size) -> Tensor
at::Tensor _fft_c2r::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, int64_t last_dim_size) {
    static auto op = create__fft_c2r_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, normalization, last_dim_size);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cufft_set_plan_cache_max_size, name, "aten::_cufft_set_plan_cache_max_size")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cufft_set_plan_cache_max_size, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cufft_set_plan_cache_max_size, schema_str, "_cufft_set_plan_cache_max_size(int device_index, int max_size) -> ()")

// aten::_cufft_set_plan_cache_max_size(int device_index, int max_size) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_cufft_set_plan_cache_max_size::schema> create__cufft_set_plan_cache_max_size_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_cufft_set_plan_cache_max_size::name, _cufft_set_plan_cache_max_size::overload_name)
      .typed<_cufft_set_plan_cache_max_size::schema>();
}

// aten::_cufft_set_plan_cache_max_size(int device_index, int max_size) -> ()
void _cufft_set_plan_cache_max_size::call(int64_t device_index, int64_t max_size) {
    static auto op = create__cufft_set_plan_cache_max_size_typed_handle();
    return op.call(device_index, max_size);
}

// aten::_cufft_set_plan_cache_max_size(int device_index, int max_size) -> ()
void _cufft_set_plan_cache_max_size::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t device_index, int64_t max_size) {
    static auto op = create__cufft_set_plan_cache_max_size_typed_handle();
    return op.redispatch(dispatchKeySet, device_index, max_size);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_put, name, "aten::index_put")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_put, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_put, schema_str, "index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor")

// aten::index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<index_put::schema> create_index_put_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(index_put::name, index_put::overload_name)
      .typed<index_put::schema>();
}

// aten::index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor
at::Tensor index_put::call(const at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate) {
    static auto op = create_index_put_typed_handle();
    return op.call(self, indices, values, accumulate);
}

// aten::index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor
at::Tensor index_put::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate) {
    static auto op = create_index_put_typed_handle();
    return op.redispatch(dispatchKeySet, self, indices, values, accumulate);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(instance_norm, name, "aten::instance_norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(instance_norm, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(instance_norm, schema_str, "instance_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool use_input_stats, float momentum, float eps, bool cudnn_enabled) -> Tensor")

// aten::instance_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool use_input_stats, float momentum, float eps, bool cudnn_enabled) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<instance_norm::schema> create_instance_norm_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(instance_norm::name, instance_norm::overload_name)
      .typed<instance_norm::schema>();
}

// aten::instance_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool use_input_stats, float momentum, float eps, bool cudnn_enabled) -> Tensor
at::Tensor instance_norm::call(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool use_input_stats, double momentum, double eps, bool cudnn_enabled) {
    static auto op = create_instance_norm_typed_handle();
    return op.call(input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled);
}

// aten::instance_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool use_input_stats, float momentum, float eps, bool cudnn_enabled) -> Tensor
at::Tensor instance_norm::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool use_input_stats, double momentum, double eps, bool cudnn_enabled) {
    static auto op = create_instance_norm_typed_handle();
    return op.redispatch(dispatchKeySet, input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_inverse_helper, name, "aten::_inverse_helper")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_inverse_helper, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_inverse_helper, schema_str, "_inverse_helper(Tensor self) -> Tensor")

// aten::_inverse_helper(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_inverse_helper::schema> create__inverse_helper_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_inverse_helper::name, _inverse_helper::overload_name)
      .typed<_inverse_helper::schema>();
}

// aten::_inverse_helper(Tensor self) -> Tensor
at::Tensor _inverse_helper::call(const at::Tensor & self) {
    static auto op = create__inverse_helper_typed_handle();
    return op.call(self);
}

// aten::_inverse_helper(Tensor self) -> Tensor
at::Tensor _inverse_helper::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create__inverse_helper_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isclose, name, "aten::isclose")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isclose, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isclose, schema_str, "isclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> Tensor")

// aten::isclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<isclose::schema> create_isclose_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(isclose::name, isclose::overload_name)
      .typed<isclose::schema>();
}

// aten::isclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> Tensor
at::Tensor isclose::call(const at::Tensor & self, const at::Tensor & other, double rtol, double atol, bool equal_nan) {
    static auto op = create_isclose_typed_handle();
    return op.call(self, other, rtol, atol, equal_nan);
}

// aten::isclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> Tensor
at::Tensor isclose::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, double rtol, double atol, bool equal_nan) {
    static auto op = create_isclose_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, rtol, atol, equal_nan);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_floating_point, name, "aten::is_floating_point")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_floating_point, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_floating_point, schema_str, "is_floating_point(Tensor self) -> bool")

// aten::is_floating_point(Tensor self) -> bool
static C10_NOINLINE c10::TypedOperatorHandle<is_floating_point::schema> create_is_floating_point_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(is_floating_point::name, is_floating_point::overload_name)
      .typed<is_floating_point::schema>();
}

// aten::is_floating_point(Tensor self) -> bool
bool is_floating_point::call(const at::Tensor & self) {
    static auto op = create_is_floating_point_typed_handle();
    return op.call(self);
}

// aten::is_floating_point(Tensor self) -> bool
bool is_floating_point::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_is_floating_point_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_complex, name, "aten::is_complex")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_complex, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_complex, schema_str, "is_complex(Tensor self) -> bool")

// aten::is_complex(Tensor self) -> bool
static C10_NOINLINE c10::TypedOperatorHandle<is_complex::schema> create_is_complex_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(is_complex::name, is_complex::overload_name)
      .typed<is_complex::schema>();
}

// aten::is_complex(Tensor self) -> bool
bool is_complex::call(const at::Tensor & self) {
    static auto op = create_is_complex_typed_handle();
    return op.call(self);
}

// aten::is_complex(Tensor self) -> bool
bool is_complex::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_is_complex_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_same_size, name, "aten::is_same_size")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_same_size, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_same_size, schema_str, "is_same_size(Tensor self, Tensor other) -> bool")

// aten::is_same_size(Tensor self, Tensor other) -> bool
static C10_NOINLINE c10::TypedOperatorHandle<is_same_size::schema> create_is_same_size_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(is_same_size::name, is_same_size::overload_name)
      .typed<is_same_size::schema>();
}

// aten::is_same_size(Tensor self, Tensor other) -> bool
bool is_same_size::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_is_same_size_typed_handle();
    return op.call(self, other);
}

// aten::is_same_size(Tensor self, Tensor other) -> bool
bool is_same_size::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_is_same_size_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(kl_div, name, "aten::kl_div")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(kl_div, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(kl_div, schema_str, "kl_div(Tensor self, Tensor target, int reduction=Mean, *, bool log_target=False) -> Tensor")

// aten::kl_div(Tensor self, Tensor target, int reduction=Mean, *, bool log_target=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<kl_div::schema> create_kl_div_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(kl_div::name, kl_div::overload_name)
      .typed<kl_div::schema>();
}

// aten::kl_div(Tensor self, Tensor target, int reduction=Mean, *, bool log_target=False) -> Tensor
at::Tensor kl_div::call(const at::Tensor & self, const at::Tensor & target, int64_t reduction, bool log_target) {
    static auto op = create_kl_div_typed_handle();
    return op.call(self, target, reduction, log_target);
}

// aten::kl_div(Tensor self, Tensor target, int reduction=Mean, *, bool log_target=False) -> Tensor
at::Tensor kl_div::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, int64_t reduction, bool log_target) {
    static auto op = create_kl_div_typed_handle();
    return op.redispatch(dispatchKeySet, self, target, reduction, log_target);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(kl_div_backward, name, "aten::kl_div_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(kl_div_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(kl_div_backward, schema_str, "kl_div_backward(Tensor grad_output, Tensor self, Tensor target, int reduction=Mean, *, bool log_target=False) -> Tensor")

// aten::kl_div_backward(Tensor grad_output, Tensor self, Tensor target, int reduction=Mean, *, bool log_target=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<kl_div_backward::schema> create_kl_div_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(kl_div_backward::name, kl_div_backward::overload_name)
      .typed<kl_div_backward::schema>();
}

// aten::kl_div_backward(Tensor grad_output, Tensor self, Tensor target, int reduction=Mean, *, bool log_target=False) -> Tensor
at::Tensor kl_div_backward::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, bool log_target) {
    static auto op = create_kl_div_backward_typed_handle();
    return op.call(grad_output, self, target, reduction, log_target);
}

// aten::kl_div_backward(Tensor grad_output, Tensor self, Tensor target, int reduction=Mean, *, bool log_target=False) -> Tensor
at::Tensor kl_div_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, bool log_target) {
    static auto op = create_kl_div_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, target, reduction, log_target);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nan_to_num_, name, "aten::nan_to_num_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nan_to_num_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nan_to_num_, schema_str, "nan_to_num_(Tensor(a!) self, float? nan=None, float? posinf=None, float? neginf=None) -> Tensor(a!)")

// aten::nan_to_num_(Tensor(a!) self, float? nan=None, float? posinf=None, float? neginf=None) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<nan_to_num_::schema> create_nan_to_num__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nan_to_num_::name, nan_to_num_::overload_name)
      .typed<nan_to_num_::schema>();
}

// aten::nan_to_num_(Tensor(a!) self, float? nan=None, float? posinf=None, float? neginf=None) -> Tensor(a!)
at::Tensor & nan_to_num_::call(at::Tensor & self, c10::optional<double> nan, c10::optional<double> posinf, c10::optional<double> neginf) {
    static auto op = create_nan_to_num__typed_handle();
    return op.call(self, nan, posinf, neginf);
}

// aten::nan_to_num_(Tensor(a!) self, float? nan=None, float? posinf=None, float? neginf=None) -> Tensor(a!)
at::Tensor & nan_to_num_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, c10::optional<double> nan, c10::optional<double> posinf, c10::optional<double> neginf) {
    static auto op = create_nan_to_num__typed_handle();
    return op.redispatch(dispatchKeySet, self, nan, posinf, neginf);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fbgemm_pack_gemm_matrix_fp16, name, "aten::fbgemm_pack_gemm_matrix_fp16")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fbgemm_pack_gemm_matrix_fp16, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fbgemm_pack_gemm_matrix_fp16, schema_str, "fbgemm_pack_gemm_matrix_fp16(Tensor input) -> Tensor")

// aten::fbgemm_pack_gemm_matrix_fp16(Tensor input) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fbgemm_pack_gemm_matrix_fp16::schema> create_fbgemm_pack_gemm_matrix_fp16_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fbgemm_pack_gemm_matrix_fp16::name, fbgemm_pack_gemm_matrix_fp16::overload_name)
      .typed<fbgemm_pack_gemm_matrix_fp16::schema>();
}

// aten::fbgemm_pack_gemm_matrix_fp16(Tensor input) -> Tensor
at::Tensor fbgemm_pack_gemm_matrix_fp16::call(const at::Tensor & input) {
    static auto op = create_fbgemm_pack_gemm_matrix_fp16_typed_handle();
    return op.call(input);
}

// aten::fbgemm_pack_gemm_matrix_fp16(Tensor input) -> Tensor
at::Tensor fbgemm_pack_gemm_matrix_fp16::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input) {
    static auto op = create_fbgemm_pack_gemm_matrix_fp16_typed_handle();
    return op.redispatch(dispatchKeySet, input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log1p_, name, "aten::log1p_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log1p_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log1p_, schema_str, "log1p_(Tensor(a!) self) -> Tensor(a!)")

// aten::log1p_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<log1p_::schema> create_log1p__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(log1p_::name, log1p_::overload_name)
      .typed<log1p_::schema>();
}

// aten::log1p_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & log1p_::call(at::Tensor & self) {
    static auto op = create_log1p__typed_handle();
    return op.call(self);
}

// aten::log1p_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & log1p_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_log1p__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log2_, name, "aten::log2_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log2_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log2_, schema_str, "log2_(Tensor(a!) self) -> Tensor(a!)")

// aten::log2_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<log2_::schema> create_log2__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(log2_::name, log2_::overload_name)
      .typed<log2_::schema>();
}

// aten::log2_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & log2_::call(at::Tensor & self) {
    static auto op = create_log2__typed_handle();
    return op.call(self);
}

// aten::log2_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & log2_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_log2__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log2_out, name, "aten::log2")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log2_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log2_out, schema_str, "log2.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::log2.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<log2_out::schema> create_log2_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(log2_out::name, log2_out::overload_name)
      .typed<log2_out::schema>();
}

// aten::log2.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & log2_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_log2_out_typed_handle();
    return op.call(self, out);
}

// aten::log2.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & log2_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_log2_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log_softmax_Dimname, name, "aten::log_softmax")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log_softmax_Dimname, overload_name, "Dimname")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log_softmax_Dimname, schema_str, "log_softmax.Dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor")

// aten::log_softmax.Dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<log_softmax_Dimname::schema> create_log_softmax_Dimname_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(log_softmax_Dimname::name, log_softmax_Dimname::overload_name)
      .typed<log_softmax_Dimname::schema>();
}

// aten::log_softmax.Dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
at::Tensor log_softmax_Dimname::call(const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_log_softmax_Dimname_typed_handle();
    return op.call(self, dim, dtype);
}

// aten::log_softmax.Dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
at::Tensor log_softmax_Dimname::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_log_softmax_Dimname_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logsumexp_names, name, "aten::logsumexp")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logsumexp_names, overload_name, "names")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logsumexp_names, schema_str, "logsumexp.names(Tensor self, Dimname[1] dim, bool keepdim=False) -> Tensor")

// aten::logsumexp.names(Tensor self, Dimname[1] dim, bool keepdim=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<logsumexp_names::schema> create_logsumexp_names_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(logsumexp_names::name, logsumexp_names::overload_name)
      .typed<logsumexp_names::schema>();
}

// aten::logsumexp.names(Tensor self, Dimname[1] dim, bool keepdim=False) -> Tensor
at::Tensor logsumexp_names::call(const at::Tensor & self, at::DimnameList dim, bool keepdim) {
    static auto op = create_logsumexp_names_typed_handle();
    return op.call(self, dim, keepdim);
}

// aten::logsumexp.names(Tensor self, Dimname[1] dim, bool keepdim=False) -> Tensor
at::Tensor logsumexp_names::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList dim, bool keepdim) {
    static auto op = create_logsumexp_names_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logsumexp_names_out, name, "aten::logsumexp")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logsumexp_names_out, overload_name, "names_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logsumexp_names_out, schema_str, "logsumexp.names_out(Tensor self, Dimname[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)")

// aten::logsumexp.names_out(Tensor self, Dimname[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<logsumexp_names_out::schema> create_logsumexp_names_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(logsumexp_names_out::name, logsumexp_names_out::overload_name)
      .typed<logsumexp_names_out::schema>();
}

// aten::logsumexp.names_out(Tensor self, Dimname[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & logsumexp_names_out::call(const at::Tensor & self, at::DimnameList dim, bool keepdim, at::Tensor & out) {
    static auto op = create_logsumexp_names_out_typed_handle();
    return op.call(self, dim, keepdim, out);
}

// aten::logsumexp.names_out(Tensor self, Dimname[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & logsumexp_names_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList dim, bool keepdim, at::Tensor & out) {
    static auto op = create_logsumexp_names_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(margin_ranking_loss, name, "aten::margin_ranking_loss")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(margin_ranking_loss, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(margin_ranking_loss, schema_str, "margin_ranking_loss(Tensor input1, Tensor input2, Tensor target, float margin=0.0, int reduction=Mean) -> Tensor")

// aten::margin_ranking_loss(Tensor input1, Tensor input2, Tensor target, float margin=0.0, int reduction=Mean) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<margin_ranking_loss::schema> create_margin_ranking_loss_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(margin_ranking_loss::name, margin_ranking_loss::overload_name)
      .typed<margin_ranking_loss::schema>();
}

// aten::margin_ranking_loss(Tensor input1, Tensor input2, Tensor target, float margin=0.0, int reduction=Mean) -> Tensor
at::Tensor margin_ranking_loss::call(const at::Tensor & input1, const at::Tensor & input2, const at::Tensor & target, double margin, int64_t reduction) {
    static auto op = create_margin_ranking_loss_typed_handle();
    return op.call(input1, input2, target, margin, reduction);
}

// aten::margin_ranking_loss(Tensor input1, Tensor input2, Tensor target, float margin=0.0, int reduction=Mean) -> Tensor
at::Tensor margin_ranking_loss::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input1, const at::Tensor & input2, const at::Tensor & target, double margin, int64_t reduction) {
    static auto op = create_margin_ranking_loss_typed_handle();
    return op.redispatch(dispatchKeySet, input1, input2, target, margin, reduction);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(matmul, name, "aten::matmul")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(matmul, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(matmul, schema_str, "matmul(Tensor self, Tensor other) -> Tensor")

// aten::matmul(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<matmul::schema> create_matmul_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(matmul::name, matmul::overload_name)
      .typed<matmul::schema>();
}

// aten::matmul(Tensor self, Tensor other) -> Tensor
at::Tensor matmul::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_matmul_typed_handle();
    return op.call(self, other);
}

// aten::matmul(Tensor self, Tensor other) -> Tensor
at::Tensor matmul::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_matmul_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(matrix_rank, name, "aten::matrix_rank")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(matrix_rank, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(matrix_rank, schema_str, "matrix_rank(Tensor self, bool symmetric=False) -> Tensor")

// aten::matrix_rank(Tensor self, bool symmetric=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<matrix_rank::schema> create_matrix_rank_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(matrix_rank::name, matrix_rank::overload_name)
      .typed<matrix_rank::schema>();
}

// aten::matrix_rank(Tensor self, bool symmetric=False) -> Tensor
at::Tensor matrix_rank::call(const at::Tensor & self, bool symmetric) {
    static auto op = create_matrix_rank_typed_handle();
    return op.call(self, symmetric);
}

// aten::matrix_rank(Tensor self, bool symmetric=False) -> Tensor
at::Tensor matrix_rank::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool symmetric) {
    static auto op = create_matrix_rank_typed_handle();
    return op.redispatch(dispatchKeySet, self, symmetric);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(matrix_exp, name, "aten::matrix_exp")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(matrix_exp, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(matrix_exp, schema_str, "matrix_exp(Tensor self) -> Tensor")

// aten::matrix_exp(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<matrix_exp::schema> create_matrix_exp_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(matrix_exp::name, matrix_exp::overload_name)
      .typed<matrix_exp::schema>();
}

// aten::matrix_exp(Tensor self) -> Tensor
at::Tensor matrix_exp::call(const at::Tensor & self) {
    static auto op = create_matrix_exp_typed_handle();
    return op.call(self);
}

// aten::matrix_exp(Tensor self) -> Tensor
at::Tensor matrix_exp::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_matrix_exp_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_compute_linear_combination, name, "aten::_compute_linear_combination")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_compute_linear_combination, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_compute_linear_combination, schema_str, "_compute_linear_combination(Tensor input, Tensor coefficients) -> Tensor")

// aten::_compute_linear_combination(Tensor input, Tensor coefficients) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_compute_linear_combination::schema> create__compute_linear_combination_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_compute_linear_combination::name, _compute_linear_combination::overload_name)
      .typed<_compute_linear_combination::schema>();
}

// aten::_compute_linear_combination(Tensor input, Tensor coefficients) -> Tensor
at::Tensor _compute_linear_combination::call(const at::Tensor & input, const at::Tensor & coefficients) {
    static auto op = create__compute_linear_combination_typed_handle();
    return op.call(input, coefficients);
}

// aten::_compute_linear_combination(Tensor input, Tensor coefficients) -> Tensor
at::Tensor _compute_linear_combination::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & coefficients) {
    static auto op = create__compute_linear_combination_typed_handle();
    return op.redispatch(dispatchKeySet, input, coefficients);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_names_dim, name, "aten::max")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_names_dim, overload_name, "names_dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_names_dim, schema_str, "max.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)")

// aten::max.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
static C10_NOINLINE c10::TypedOperatorHandle<max_names_dim::schema> create_max_names_dim_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(max_names_dim::name, max_names_dim::overload_name)
      .typed<max_names_dim::schema>();
}

// aten::max.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> max_names_dim::call(const at::Tensor & self, at::Dimname dim, bool keepdim) {
    static auto op = create_max_names_dim_typed_handle();
    return op.call(self, dim, keepdim);
}

// aten::max.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> max_names_dim::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, bool keepdim) {
    static auto op = create_max_names_dim_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_max_pool2d_backward, name, "aten::mkldnn_max_pool2d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_max_pool2d_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_max_pool2d_backward, schema_str, "mkldnn_max_pool2d_backward(Tensor grad_output, Tensor output, Tensor input, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor")

// aten::mkldnn_max_pool2d_backward(Tensor grad_output, Tensor output, Tensor input, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<mkldnn_max_pool2d_backward::schema> create_mkldnn_max_pool2d_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mkldnn_max_pool2d_backward::name, mkldnn_max_pool2d_backward::overload_name)
      .typed<mkldnn_max_pool2d_backward::schema>();
}

// aten::mkldnn_max_pool2d_backward(Tensor grad_output, Tensor output, Tensor input, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor
at::Tensor mkldnn_max_pool2d_backward::call(const at::Tensor & grad_output, const at::Tensor & output, const at::Tensor & input, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
    static auto op = create_mkldnn_max_pool2d_backward_typed_handle();
    return op.call(grad_output, output, input, kernel_size, stride, padding, dilation, ceil_mode);
}

// aten::mkldnn_max_pool2d_backward(Tensor grad_output, Tensor output, Tensor input, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor
at::Tensor mkldnn_max_pool2d_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & output, const at::Tensor & input, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
    static auto op = create_mkldnn_max_pool2d_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, output, input, kernel_size, stride, padding, dilation, ceil_mode);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_pool3d, name, "aten::max_pool3d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_pool3d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_pool3d, schema_str, "max_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> Tensor")

// aten::max_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<max_pool3d::schema> create_max_pool3d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(max_pool3d::name, max_pool3d::overload_name)
      .typed<max_pool3d::schema>();
}

// aten::max_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> Tensor
at::Tensor max_pool3d::call(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
    static auto op = create_max_pool3d_typed_handle();
    return op.call(self, kernel_size, stride, padding, dilation, ceil_mode);
}

// aten::max_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> Tensor
at::Tensor max_pool3d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
    static auto op = create_max_pool3d_typed_handle();
    return op.redispatch(dispatchKeySet, self, kernel_size, stride, padding, dilation, ceil_mode);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mean_out, name, "aten::mean")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mean_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mean_out, schema_str, "mean.out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)")

// aten::mean.out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<mean_out::schema> create_mean_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mean_out::name, mean_out::overload_name)
      .typed<mean_out::schema>();
}

// aten::mean.out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & mean_out::call(const at::Tensor & self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
    static auto op = create_mean_out_typed_handle();
    return op.call(self, dim, keepdim, dtype, out);
}

// aten::mean.out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & mean_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
    static auto op = create_mean_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, dtype, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mean_names_dim, name, "aten::mean")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mean_names_dim, overload_name, "names_dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mean_names_dim, schema_str, "mean.names_dim(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor")

// aten::mean.names_dim(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<mean_names_dim::schema> create_mean_names_dim_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mean_names_dim::name, mean_names_dim::overload_name)
      .typed<mean_names_dim::schema>();
}

// aten::mean.names_dim(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
at::Tensor mean_names_dim::call(const at::Tensor & self, at::DimnameList dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_mean_names_dim_typed_handle();
    return op.call(self, dim, keepdim, dtype);
}

// aten::mean.names_dim(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
at::Tensor mean_names_dim::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_mean_names_dim_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(median, name, "aten::median")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(median, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(median, schema_str, "median(Tensor self) -> Tensor")

// aten::median(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<median::schema> create_median_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(median::name, median::overload_name)
      .typed<median::schema>();
}

// aten::median(Tensor self) -> Tensor
at::Tensor median::call(const at::Tensor & self) {
    static auto op = create_median_typed_handle();
    return op.call(self);
}

// aten::median(Tensor self) -> Tensor
at::Tensor median::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_median_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanmedian, name, "aten::nanmedian")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanmedian, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanmedian, schema_str, "nanmedian(Tensor self) -> Tensor")

// aten::nanmedian(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<nanmedian::schema> create_nanmedian_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nanmedian::name, nanmedian::overload_name)
      .typed<nanmedian::schema>();
}

// aten::nanmedian(Tensor self) -> Tensor
at::Tensor nanmedian::call(const at::Tensor & self) {
    static auto op = create_nanmedian_typed_handle();
    return op.call(self);
}

// aten::nanmedian(Tensor self) -> Tensor
at::Tensor nanmedian::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_nanmedian_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanmedian_dim, name, "aten::nanmedian")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanmedian_dim, overload_name, "dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanmedian_dim, schema_str, "nanmedian.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)")

// aten::nanmedian.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
static C10_NOINLINE c10::TypedOperatorHandle<nanmedian_dim::schema> create_nanmedian_dim_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nanmedian_dim::name, nanmedian_dim::overload_name)
      .typed<nanmedian_dim::schema>();
}

// aten::nanmedian.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> nanmedian_dim::call(const at::Tensor & self, int64_t dim, bool keepdim) {
    static auto op = create_nanmedian_dim_typed_handle();
    return op.call(self, dim, keepdim);
}

// aten::nanmedian.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> nanmedian_dim::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool keepdim) {
    static auto op = create_nanmedian_dim_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(min_dim, name, "aten::min")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(min_dim, overload_name, "dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(min_dim, schema_str, "min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)")

// aten::min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
static C10_NOINLINE c10::TypedOperatorHandle<min_dim::schema> create_min_dim_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(min_dim::name, min_dim::overload_name)
      .typed<min_dim::schema>();
}

// aten::min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> min_dim::call(const at::Tensor & self, int64_t dim, bool keepdim) {
    static auto op = create_min_dim_typed_handle();
    return op.call(self, dim, keepdim);
}

// aten::min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> min_dim::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool keepdim) {
    static auto op = create_min_dim_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_batch_norm, name, "aten::miopen_batch_norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_batch_norm, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_batch_norm, schema_str, "miopen_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float exponential_average_factor, float epsilon) -> (Tensor, Tensor, Tensor)")

// aten::miopen_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float exponential_average_factor, float epsilon) -> (Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<miopen_batch_norm::schema> create_miopen_batch_norm_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(miopen_batch_norm::name, miopen_batch_norm::overload_name)
      .typed<miopen_batch_norm::schema>();
}

// aten::miopen_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float exponential_average_factor, float epsilon) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_batch_norm::call(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double exponential_average_factor, double epsilon) {
    static auto op = create_miopen_batch_norm_typed_handle();
    return op.call(input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
}

// aten::miopen_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float exponential_average_factor, float epsilon) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_batch_norm::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double exponential_average_factor, double epsilon) {
    static auto op = create_miopen_batch_norm_typed_handle();
    return op.redispatch(dispatchKeySet, input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_convolution_backward, name, "aten::miopen_convolution_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_convolution_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_convolution_backward, schema_str, "miopen_convolution_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool[3] output_mask) -> (Tensor, Tensor, Tensor)")

// aten::miopen_convolution_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<miopen_convolution_backward::schema> create_miopen_convolution_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(miopen_convolution_backward::name, miopen_convolution_backward::overload_name)
      .typed<miopen_convolution_backward::schema>();
}

// aten::miopen_convolution_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_convolution_backward::call(const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, ::std::array<bool,3> output_mask) {
    static auto op = create_miopen_convolution_backward_typed_handle();
    return op.call(self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, output_mask);
}

// aten::miopen_convolution_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_convolution_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, ::std::array<bool,3> output_mask) {
    static auto op = create_miopen_convolution_backward_typed_handle();
    return op.redispatch(dispatchKeySet, self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, output_mask);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_convolution_backward_bias, name, "aten::miopen_convolution_backward_bias")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_convolution_backward_bias, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_convolution_backward_bias, schema_str, "miopen_convolution_backward_bias(Tensor grad_output) -> Tensor")

// aten::miopen_convolution_backward_bias(Tensor grad_output) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<miopen_convolution_backward_bias::schema> create_miopen_convolution_backward_bias_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(miopen_convolution_backward_bias::name, miopen_convolution_backward_bias::overload_name)
      .typed<miopen_convolution_backward_bias::schema>();
}

// aten::miopen_convolution_backward_bias(Tensor grad_output) -> Tensor
at::Tensor miopen_convolution_backward_bias::call(const at::Tensor & grad_output) {
    static auto op = create_miopen_convolution_backward_bias_typed_handle();
    return op.call(grad_output);
}

// aten::miopen_convolution_backward_bias(Tensor grad_output) -> Tensor
at::Tensor miopen_convolution_backward_bias::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output) {
    static auto op = create_miopen_convolution_backward_bias_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_convolution_transpose, name, "aten::miopen_convolution_transpose")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_convolution_transpose, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_convolution_transpose, schema_str, "miopen_convolution_transpose(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor")

// aten::miopen_convolution_transpose(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<miopen_convolution_transpose::schema> create_miopen_convolution_transpose_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(miopen_convolution_transpose::name, miopen_convolution_transpose::overload_name)
      .typed<miopen_convolution_transpose::schema>();
}

// aten::miopen_convolution_transpose(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
at::Tensor miopen_convolution_transpose::call(const at::Tensor & self, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
    static auto op = create_miopen_convolution_transpose_typed_handle();
    return op.call(self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
}

// aten::miopen_convolution_transpose(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
at::Tensor miopen_convolution_transpose::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
    static auto op = create_miopen_convolution_transpose_typed_handle();
    return op.redispatch(dispatchKeySet, self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_convolution_transpose_backward_input, name, "aten::miopen_convolution_transpose_backward_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_convolution_transpose_backward_input, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_convolution_transpose_backward_input, schema_str, "miopen_convolution_transpose_backward_input(Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor")

// aten::miopen_convolution_transpose_backward_input(Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<miopen_convolution_transpose_backward_input::schema> create_miopen_convolution_transpose_backward_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(miopen_convolution_transpose_backward_input::name, miopen_convolution_transpose_backward_input::overload_name)
      .typed<miopen_convolution_transpose_backward_input::schema>();
}

// aten::miopen_convolution_transpose_backward_input(Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
at::Tensor miopen_convolution_transpose_backward_input::call(const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
    static auto op = create_miopen_convolution_transpose_backward_input_typed_handle();
    return op.call(grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
}

// aten::miopen_convolution_transpose_backward_input(Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
at::Tensor miopen_convolution_transpose_backward_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
    static auto op = create_miopen_convolution_transpose_backward_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_depthwise_convolution_backward_weight, name, "aten::miopen_depthwise_convolution_backward_weight")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_depthwise_convolution_backward_weight, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_depthwise_convolution_backward_weight, schema_str, "miopen_depthwise_convolution_backward_weight(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor")

// aten::miopen_depthwise_convolution_backward_weight(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<miopen_depthwise_convolution_backward_weight::schema> create_miopen_depthwise_convolution_backward_weight_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(miopen_depthwise_convolution_backward_weight::name, miopen_depthwise_convolution_backward_weight::overload_name)
      .typed<miopen_depthwise_convolution_backward_weight::schema>();
}

// aten::miopen_depthwise_convolution_backward_weight(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
at::Tensor miopen_depthwise_convolution_backward_weight::call(at::IntArrayRef weight_size, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
    static auto op = create_miopen_depthwise_convolution_backward_weight_typed_handle();
    return op.call(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
}

// aten::miopen_depthwise_convolution_backward_weight(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
at::Tensor miopen_depthwise_convolution_backward_weight::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef weight_size, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
    static auto op = create_miopen_depthwise_convolution_backward_weight_typed_handle();
    return op.redispatch(dispatchKeySet, weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_rnn_backward, name, "aten::miopen_rnn_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_rnn_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_rnn_backward, schema_str, "miopen_rnn_backward(Tensor input, Tensor[] weight, int weight_stride0, Tensor weight_buf, Tensor hx, Tensor? cx, Tensor output, Tensor? grad_output, Tensor? grad_hy, Tensor? grad_cy, int mode, int hidden_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state, Tensor reserve, bool[4] output_mask) -> (Tensor, Tensor, Tensor, Tensor[])")

// aten::miopen_rnn_backward(Tensor input, Tensor[] weight, int weight_stride0, Tensor weight_buf, Tensor hx, Tensor? cx, Tensor output, Tensor? grad_output, Tensor? grad_hy, Tensor? grad_cy, int mode, int hidden_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state, Tensor reserve, bool[4] output_mask) -> (Tensor, Tensor, Tensor, Tensor[])
static C10_NOINLINE c10::TypedOperatorHandle<miopen_rnn_backward::schema> create_miopen_rnn_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(miopen_rnn_backward::name, miopen_rnn_backward::overload_name)
      .typed<miopen_rnn_backward::schema>();
}

// aten::miopen_rnn_backward(Tensor input, Tensor[] weight, int weight_stride0, Tensor weight_buf, Tensor hx, Tensor? cx, Tensor output, Tensor? grad_output, Tensor? grad_hy, Tensor? grad_cy, int mode, int hidden_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state, Tensor reserve, bool[4] output_mask) -> (Tensor, Tensor, Tensor, Tensor[])
::std::tuple<at::Tensor,at::Tensor,at::Tensor,::std::vector<at::Tensor>> miopen_rnn_backward::call(const at::Tensor & input, at::TensorList weight, int64_t weight_stride0, const at::Tensor & weight_buf, const at::Tensor & hx, const c10::optional<at::Tensor> & cx, const at::Tensor & output, const c10::optional<at::Tensor> & grad_output, const c10::optional<at::Tensor> & grad_hy, const c10::optional<at::Tensor> & grad_cy, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, at::IntArrayRef batch_sizes, const c10::optional<at::Tensor> & dropout_state, const at::Tensor & reserve, ::std::array<bool,4> output_mask) {
    static auto op = create_miopen_rnn_backward_typed_handle();
    return op.call(input, weight, weight_stride0, weight_buf, hx, cx, output, grad_output, grad_hy, grad_cy, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, reserve, output_mask);
}

// aten::miopen_rnn_backward(Tensor input, Tensor[] weight, int weight_stride0, Tensor weight_buf, Tensor hx, Tensor? cx, Tensor output, Tensor? grad_output, Tensor? grad_hy, Tensor? grad_cy, int mode, int hidden_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state, Tensor reserve, bool[4] output_mask) -> (Tensor, Tensor, Tensor, Tensor[])
::std::tuple<at::Tensor,at::Tensor,at::Tensor,::std::vector<at::Tensor>> miopen_rnn_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, at::TensorList weight, int64_t weight_stride0, const at::Tensor & weight_buf, const at::Tensor & hx, const c10::optional<at::Tensor> & cx, const at::Tensor & output, const c10::optional<at::Tensor> & grad_output, const c10::optional<at::Tensor> & grad_hy, const c10::optional<at::Tensor> & grad_cy, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, at::IntArrayRef batch_sizes, const c10::optional<at::Tensor> & dropout_state, const at::Tensor & reserve, ::std::array<bool,4> output_mask) {
    static auto op = create_miopen_rnn_backward_typed_handle();
    return op.redispatch(dispatchKeySet, input, weight, weight_stride0, weight_buf, hx, cx, output, grad_output, grad_hy, grad_cy, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, reserve, output_mask);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mul_Scalar, name, "aten::mul")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mul_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mul_Scalar, schema_str, "mul.Scalar(Tensor self, Scalar other) -> Tensor")

// aten::mul.Scalar(Tensor self, Scalar other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<mul_Scalar::schema> create_mul_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mul_Scalar::name, mul_Scalar::overload_name)
      .typed<mul_Scalar::schema>();
}

// aten::mul.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor mul_Scalar::call(const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_mul_Scalar_typed_handle();
    return op.call(self, other);
}

// aten::mul.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor mul_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_mul_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multiply_Tensor, name, "aten::multiply")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multiply_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multiply_Tensor, schema_str, "multiply.Tensor(Tensor self, Tensor other) -> Tensor")

// aten::multiply.Tensor(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<multiply_Tensor::schema> create_multiply_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(multiply_Tensor::name, multiply_Tensor::overload_name)
      .typed<multiply_Tensor::schema>();
}

// aten::multiply.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor multiply_Tensor::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_multiply_Tensor_typed_handle();
    return op.call(self, other);
}

// aten::multiply.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor multiply_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_multiply_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(batch_norm_elemt, name, "aten::batch_norm_elemt")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(batch_norm_elemt, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(batch_norm_elemt, schema_str, "batch_norm_elemt(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor invstd, float eps) -> Tensor")

// aten::batch_norm_elemt(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor invstd, float eps) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<batch_norm_elemt::schema> create_batch_norm_elemt_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(batch_norm_elemt::name, batch_norm_elemt::overload_name)
      .typed<batch_norm_elemt::schema>();
}

// aten::batch_norm_elemt(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor invstd, float eps) -> Tensor
at::Tensor batch_norm_elemt::call(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const at::Tensor & mean, const at::Tensor & invstd, double eps) {
    static auto op = create_batch_norm_elemt_typed_handle();
    return op.call(input, weight, bias, mean, invstd, eps);
}

// aten::batch_norm_elemt(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor invstd, float eps) -> Tensor
at::Tensor batch_norm_elemt::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const at::Tensor & mean, const at::Tensor & invstd, double eps) {
    static auto op = create_batch_norm_elemt_typed_handle();
    return op.redispatch(dispatchKeySet, input, weight, bias, mean, invstd, eps);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cdist, name, "aten::cdist")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cdist, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cdist, schema_str, "cdist(Tensor x1, Tensor x2, float p=2, int? compute_mode=None) -> Tensor")

// aten::cdist(Tensor x1, Tensor x2, float p=2, int? compute_mode=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<cdist::schema> create_cdist_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cdist::name, cdist::overload_name)
      .typed<cdist::schema>();
}

// aten::cdist(Tensor x1, Tensor x2, float p=2, int? compute_mode=None) -> Tensor
at::Tensor cdist::call(const at::Tensor & x1, const at::Tensor & x2, double p, c10::optional<int64_t> compute_mode) {
    static auto op = create_cdist_typed_handle();
    return op.call(x1, x2, p, compute_mode);
}

// aten::cdist(Tensor x1, Tensor x2, float p=2, int? compute_mode=None) -> Tensor
at::Tensor cdist::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & x1, const at::Tensor & x2, double p, c10::optional<int64_t> compute_mode) {
    static auto op = create_cdist_typed_handle();
    return op.redispatch(dispatchKeySet, x1, x2, p, compute_mode);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(moveaxis_int, name, "aten::moveaxis")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(moveaxis_int, overload_name, "int")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(moveaxis_int, schema_str, "moveaxis.int(Tensor(a) self, int source, int destination) -> Tensor(a)")

// aten::moveaxis.int(Tensor(a) self, int source, int destination) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<moveaxis_int::schema> create_moveaxis_int_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(moveaxis_int::name, moveaxis_int::overload_name)
      .typed<moveaxis_int::schema>();
}

// aten::moveaxis.int(Tensor(a) self, int source, int destination) -> Tensor(a)
at::Tensor moveaxis_int::call(const at::Tensor & self, int64_t source, int64_t destination) {
    static auto op = create_moveaxis_int_typed_handle();
    return op.call(self, source, destination);
}

// aten::moveaxis.int(Tensor(a) self, int source, int destination) -> Tensor(a)
at::Tensor moveaxis_int::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t source, int64_t destination) {
    static auto op = create_moveaxis_int_typed_handle();
    return op.redispatch(dispatchKeySet, self, source, destination);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(channel_shuffle, name, "aten::channel_shuffle")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(channel_shuffle, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(channel_shuffle, schema_str, "channel_shuffle(Tensor self, int groups) -> Tensor")

// aten::channel_shuffle(Tensor self, int groups) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<channel_shuffle::schema> create_channel_shuffle_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(channel_shuffle::name, channel_shuffle::overload_name)
      .typed<channel_shuffle::schema>();
}

// aten::channel_shuffle(Tensor self, int groups) -> Tensor
at::Tensor channel_shuffle::call(const at::Tensor & self, int64_t groups) {
    static auto op = create_channel_shuffle_typed_handle();
    return op.call(self, groups);
}

// aten::channel_shuffle(Tensor self, int groups) -> Tensor
at::Tensor channel_shuffle::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t groups) {
    static auto op = create_channel_shuffle_typed_handle();
    return op.redispatch(dispatchKeySet, self, groups);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(poisson_nll_loss, name, "aten::poisson_nll_loss")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(poisson_nll_loss, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(poisson_nll_loss, schema_str, "poisson_nll_loss(Tensor input, Tensor target, bool log_input, bool full, float eps, int reduction) -> Tensor")

// aten::poisson_nll_loss(Tensor input, Tensor target, bool log_input, bool full, float eps, int reduction) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<poisson_nll_loss::schema> create_poisson_nll_loss_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(poisson_nll_loss::name, poisson_nll_loss::overload_name)
      .typed<poisson_nll_loss::schema>();
}

// aten::poisson_nll_loss(Tensor input, Tensor target, bool log_input, bool full, float eps, int reduction) -> Tensor
at::Tensor poisson_nll_loss::call(const at::Tensor & input, const at::Tensor & target, bool log_input, bool full, double eps, int64_t reduction) {
    static auto op = create_poisson_nll_loss_typed_handle();
    return op.call(input, target, log_input, full, eps, reduction);
}

// aten::poisson_nll_loss(Tensor input, Tensor target, bool log_input, bool full, float eps, int reduction) -> Tensor
at::Tensor poisson_nll_loss::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & target, bool log_input, bool full, double eps, int64_t reduction) {
    static auto op = create_poisson_nll_loss_typed_handle();
    return op.redispatch(dispatchKeySet, input, target, log_input, full, eps, reduction);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rad2deg_, name, "aten::rad2deg_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rad2deg_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rad2deg_, schema_str, "rad2deg_(Tensor(a!) self) -> Tensor(a!)")

// aten::rad2deg_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<rad2deg_::schema> create_rad2deg__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(rad2deg_::name, rad2deg_::overload_name)
      .typed<rad2deg_::schema>();
}

// aten::rad2deg_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & rad2deg_::call(at::Tensor & self) {
    static auto op = create_rad2deg__typed_handle();
    return op.call(self);
}

// aten::rad2deg_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & rad2deg_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_rad2deg__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(deg2rad, name, "aten::deg2rad")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(deg2rad, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(deg2rad, schema_str, "deg2rad(Tensor self) -> Tensor")

// aten::deg2rad(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<deg2rad::schema> create_deg2rad_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(deg2rad::name, deg2rad::overload_name)
      .typed<deg2rad::schema>();
}

// aten::deg2rad(Tensor self) -> Tensor
at::Tensor deg2rad::call(const at::Tensor & self) {
    static auto op = create_deg2rad_typed_handle();
    return op.call(self);
}

// aten::deg2rad(Tensor self) -> Tensor
at::Tensor deg2rad::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_deg2rad_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(deg2rad_, name, "aten::deg2rad_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(deg2rad_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(deg2rad_, schema_str, "deg2rad_(Tensor(a!) self) -> Tensor(a!)")

// aten::deg2rad_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<deg2rad_::schema> create_deg2rad__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(deg2rad_::name, deg2rad_::overload_name)
      .typed<deg2rad_::schema>();
}

// aten::deg2rad_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & deg2rad_::call(at::Tensor & self) {
    static auto op = create_deg2rad__typed_handle();
    return op.call(self);
}

// aten::deg2rad_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & deg2rad_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_deg2rad__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(deg2rad_out, name, "aten::deg2rad")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(deg2rad_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(deg2rad_out, schema_str, "deg2rad.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::deg2rad.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<deg2rad_out::schema> create_deg2rad_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(deg2rad_out::name, deg2rad_out::overload_name)
      .typed<deg2rad_out::schema>();
}

// aten::deg2rad.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & deg2rad_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_deg2rad_out_typed_handle();
    return op.call(self, out);
}

// aten::deg2rad.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & deg2rad_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_deg2rad_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rand_generator, name, "aten::rand")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rand_generator, overload_name, "generator")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rand_generator, schema_str, "rand.generator(int[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::rand.generator(int[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<rand_generator::schema> create_rand_generator_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(rand_generator::name, rand_generator::overload_name)
      .typed<rand_generator::schema>();
}

// aten::rand.generator(int[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor rand_generator::call(at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_rand_generator_typed_handle();
    return op.call(size, generator, dtype, layout, device, pin_memory);
}

// aten::rand.generator(int[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor rand_generator::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_rand_generator_typed_handle();
    return op.redispatch(dispatchKeySet, size, generator, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rand_out, name, "aten::rand")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rand_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rand_out, schema_str, "rand.out(int[] size, *, Tensor(a!) out) -> Tensor(a!)")

// aten::rand.out(int[] size, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<rand_out::schema> create_rand_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(rand_out::name, rand_out::overload_name)
      .typed<rand_out::schema>();
}

// aten::rand.out(int[] size, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & rand_out::call(at::IntArrayRef size, at::Tensor & out) {
    static auto op = create_rand_out_typed_handle();
    return op.call(size, out);
}

// aten::rand.out(int[] size, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & rand_out::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, at::Tensor & out) {
    static auto op = create_rand_out_typed_handle();
    return op.redispatch(dispatchKeySet, size, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rand_generator_out, name, "aten::rand")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rand_generator_out, overload_name, "generator_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rand_generator_out, schema_str, "rand.generator_out(int[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)")

// aten::rand.generator_out(int[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<rand_generator_out::schema> create_rand_generator_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(rand_generator_out::name, rand_generator_out::overload_name)
      .typed<rand_generator_out::schema>();
}

// aten::rand.generator_out(int[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
at::Tensor & rand_generator_out::call(at::IntArrayRef size, c10::optional<at::Generator> generator, at::Tensor & out) {
    static auto op = create_rand_generator_out_typed_handle();
    return op.call(size, generator, out);
}

// aten::rand.generator_out(int[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
at::Tensor & rand_generator_out::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::Generator> generator, at::Tensor & out) {
    static auto op = create_rand_generator_out_typed_handle();
    return op.redispatch(dispatchKeySet, size, generator, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randint_low_generator, name, "aten::randint")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randint_low_generator, overload_name, "low_generator")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randint_low_generator, schema_str, "randint.low_generator(int low, int high, int[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::randint.low_generator(int low, int high, int[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<randint_low_generator::schema> create_randint_low_generator_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(randint_low_generator::name, randint_low_generator::overload_name)
      .typed<randint_low_generator::schema>();
}

// aten::randint.low_generator(int low, int high, int[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor randint_low_generator::call(int64_t low, int64_t high, at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_randint_low_generator_typed_handle();
    return op.call(low, high, size, generator, dtype, layout, device, pin_memory);
}

// aten::randint.low_generator(int low, int high, int[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor randint_low_generator::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t low, int64_t high, at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_randint_low_generator_typed_handle();
    return op.redispatch(dispatchKeySet, low, high, size, generator, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randint_out, name, "aten::randint")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randint_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randint_out, schema_str, "randint.out(int high, int[] size, *, Tensor(a!) out) -> Tensor(a!)")

// aten::randint.out(int high, int[] size, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<randint_out::schema> create_randint_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(randint_out::name, randint_out::overload_name)
      .typed<randint_out::schema>();
}

// aten::randint.out(int high, int[] size, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & randint_out::call(int64_t high, at::IntArrayRef size, at::Tensor & out) {
    static auto op = create_randint_out_typed_handle();
    return op.call(high, size, out);
}

// aten::randint.out(int high, int[] size, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & randint_out::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t high, at::IntArrayRef size, at::Tensor & out) {
    static auto op = create_randint_out_typed_handle();
    return op.redispatch(dispatchKeySet, high, size, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randn_generator_with_names, name, "aten::randn")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randn_generator_with_names, overload_name, "generator_with_names")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randn_generator_with_names, schema_str, "randn.generator_with_names(int[] size, *, Generator? generator, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::randn.generator_with_names(int[] size, *, Generator? generator, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<randn_generator_with_names::schema> create_randn_generator_with_names_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(randn_generator_with_names::name, randn_generator_with_names::overload_name)
      .typed<randn_generator_with_names::schema>();
}

// aten::randn.generator_with_names(int[] size, *, Generator? generator, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor randn_generator_with_names::call(at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_randn_generator_with_names_typed_handle();
    return op.call(size, generator, names, dtype, layout, device, pin_memory);
}

// aten::randn.generator_with_names(int[] size, *, Generator? generator, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor randn_generator_with_names::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_randn_generator_with_names_typed_handle();
    return op.redispatch(dispatchKeySet, size, generator, names, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randperm, name, "aten::randperm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randperm, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randperm, schema_str, "randperm(int n, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::randperm(int n, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<randperm::schema> create_randperm_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(randperm::name, randperm::overload_name)
      .typed<randperm::schema>();
}

// aten::randperm(int n, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor randperm::call(int64_t n, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_randperm_typed_handle();
    return op.call(n, dtype, layout, device, pin_memory);
}

// aten::randperm(int n, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor randperm::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t n, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_randperm_typed_handle();
    return op.redispatch(dispatchKeySet, n, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(range_out, name, "aten::range")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(range_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(range_out, schema_str, "range.out(Scalar start, Scalar end, Scalar step=1, *, Tensor(a!) out) -> Tensor(a!)")

// aten::range.out(Scalar start, Scalar end, Scalar step=1, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<range_out::schema> create_range_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(range_out::name, range_out::overload_name)
      .typed<range_out::schema>();
}

// aten::range.out(Scalar start, Scalar end, Scalar step=1, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & range_out::call(const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, at::Tensor & out) {
    static auto op = create_range_out_typed_handle();
    return op.call(start, end, step, out);
}

// aten::range.out(Scalar start, Scalar end, Scalar step=1, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & range_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, at::Tensor & out) {
    static auto op = create_range_out_typed_handle();
    return op.redispatch(dispatchKeySet, start, end, step, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(neg_, name, "aten::neg_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(neg_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(neg_, schema_str, "neg_(Tensor(a!) self) -> Tensor(a!)")

// aten::neg_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<neg_::schema> create_neg__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(neg_::name, neg_::overload_name)
      .typed<neg_::schema>();
}

// aten::neg_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & neg_::call(at::Tensor & self) {
    static auto op = create_neg__typed_handle();
    return op.call(self);
}

// aten::neg_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & neg_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_neg__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(negative, name, "aten::negative")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(negative, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(negative, schema_str, "negative(Tensor self) -> Tensor")

// aten::negative(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<negative::schema> create_negative_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(negative::name, negative::overload_name)
      .typed<negative::schema>();
}

// aten::negative(Tensor self) -> Tensor
at::Tensor negative::call(const at::Tensor & self) {
    static auto op = create_negative_typed_handle();
    return op.call(self);
}

// aten::negative(Tensor self) -> Tensor
at::Tensor negative::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_negative_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(negative_, name, "aten::negative_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(negative_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(negative_, schema_str, "negative_(Tensor(a!) self) -> Tensor(a!)")

// aten::negative_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<negative_::schema> create_negative__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(negative_::name, negative_::overload_name)
      .typed<negative_::schema>();
}

// aten::negative_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & negative_::call(at::Tensor & self) {
    static auto op = create_negative__typed_handle();
    return op.call(self);
}

// aten::negative_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & negative_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_negative__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(repeat_interleave_Tensor, name, "aten::repeat_interleave")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(repeat_interleave_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(repeat_interleave_Tensor, schema_str, "repeat_interleave.Tensor(Tensor repeats, *, int? output_size=None) -> Tensor")

// aten::repeat_interleave.Tensor(Tensor repeats, *, int? output_size=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<repeat_interleave_Tensor::schema> create_repeat_interleave_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(repeat_interleave_Tensor::name, repeat_interleave_Tensor::overload_name)
      .typed<repeat_interleave_Tensor::schema>();
}

// aten::repeat_interleave.Tensor(Tensor repeats, *, int? output_size=None) -> Tensor
at::Tensor repeat_interleave_Tensor::call(const at::Tensor & repeats, c10::optional<int64_t> output_size) {
    static auto op = create_repeat_interleave_Tensor_typed_handle();
    return op.call(repeats, output_size);
}

// aten::repeat_interleave.Tensor(Tensor repeats, *, int? output_size=None) -> Tensor
at::Tensor repeat_interleave_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & repeats, c10::optional<int64_t> output_size) {
    static auto op = create_repeat_interleave_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, repeats, output_size);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(repeat_interleave_self_int, name, "aten::repeat_interleave")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(repeat_interleave_self_int, overload_name, "self_int")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(repeat_interleave_self_int, schema_str, "repeat_interleave.self_int(Tensor self, int repeats, int? dim=None, *, int? output_size=None) -> Tensor")

// aten::repeat_interleave.self_int(Tensor self, int repeats, int? dim=None, *, int? output_size=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<repeat_interleave_self_int::schema> create_repeat_interleave_self_int_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(repeat_interleave_self_int::name, repeat_interleave_self_int::overload_name)
      .typed<repeat_interleave_self_int::schema>();
}

// aten::repeat_interleave.self_int(Tensor self, int repeats, int? dim=None, *, int? output_size=None) -> Tensor
at::Tensor repeat_interleave_self_int::call(const at::Tensor & self, int64_t repeats, c10::optional<int64_t> dim, c10::optional<int64_t> output_size) {
    static auto op = create_repeat_interleave_self_int_typed_handle();
    return op.call(self, repeats, dim, output_size);
}

// aten::repeat_interleave.self_int(Tensor self, int repeats, int? dim=None, *, int? output_size=None) -> Tensor
at::Tensor repeat_interleave_self_int::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t repeats, c10::optional<int64_t> dim, c10::optional<int64_t> output_size) {
    static auto op = create_repeat_interleave_self_int_typed_handle();
    return op.redispatch(dispatchKeySet, self, repeats, dim, output_size);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(round_, name, "aten::round_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(round_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(round_, schema_str, "round_(Tensor(a!) self) -> Tensor(a!)")

// aten::round_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<round_::schema> create_round__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(round_::name, round_::overload_name)
      .typed<round_::schema>();
}

// aten::round_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & round_::call(at::Tensor & self) {
    static auto op = create_round__typed_handle();
    return op.call(self);
}

// aten::round_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & round_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_round__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(relu, name, "aten::relu")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(relu, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(relu, schema_str, "relu(Tensor self) -> Tensor")

// aten::relu(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<relu::schema> create_relu_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(relu::name, relu::overload_name)
      .typed<relu::schema>();
}

// aten::relu(Tensor self) -> Tensor
at::Tensor relu::call(const at::Tensor & self) {
    static auto op = create_relu_typed_handle();
    return op.call(self);
}

// aten::relu(Tensor self) -> Tensor
at::Tensor relu::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_relu_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(infinitely_differentiable_gelu_backward, name, "aten::infinitely_differentiable_gelu_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(infinitely_differentiable_gelu_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(infinitely_differentiable_gelu_backward, schema_str, "infinitely_differentiable_gelu_backward(Tensor grad, Tensor self) -> Tensor")

// aten::infinitely_differentiable_gelu_backward(Tensor grad, Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<infinitely_differentiable_gelu_backward::schema> create_infinitely_differentiable_gelu_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(infinitely_differentiable_gelu_backward::name, infinitely_differentiable_gelu_backward::overload_name)
      .typed<infinitely_differentiable_gelu_backward::schema>();
}

// aten::infinitely_differentiable_gelu_backward(Tensor grad, Tensor self) -> Tensor
at::Tensor infinitely_differentiable_gelu_backward::call(const at::Tensor & grad, const at::Tensor & self) {
    static auto op = create_infinitely_differentiable_gelu_backward_typed_handle();
    return op.call(grad, self);
}

// aten::infinitely_differentiable_gelu_backward(Tensor grad, Tensor self) -> Tensor
at::Tensor infinitely_differentiable_gelu_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & self) {
    static auto op = create_infinitely_differentiable_gelu_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardshrink_backward_grad_input, name, "aten::hardshrink_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardshrink_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardshrink_backward_grad_input, schema_str, "hardshrink_backward.grad_input(Tensor grad_out, Tensor self, Scalar lambd, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::hardshrink_backward.grad_input(Tensor grad_out, Tensor self, Scalar lambd, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<hardshrink_backward_grad_input::schema> create_hardshrink_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(hardshrink_backward_grad_input::name, hardshrink_backward_grad_input::overload_name)
      .typed<hardshrink_backward_grad_input::schema>();
}

// aten::hardshrink_backward.grad_input(Tensor grad_out, Tensor self, Scalar lambd, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & hardshrink_backward_grad_input::call(const at::Tensor & grad_out, const at::Tensor & self, const at::Scalar & lambd, at::Tensor & grad_input) {
    static auto op = create_hardshrink_backward_grad_input_typed_handle();
    return op.call(grad_out, self, lambd, grad_input);
}

// aten::hardshrink_backward.grad_input(Tensor grad_out, Tensor self, Scalar lambd, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & hardshrink_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_out, const at::Tensor & self, const at::Scalar & lambd, at::Tensor & grad_input) {
    static auto op = create_hardshrink_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_out, self, lambd, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardshrink_backward, name, "aten::hardshrink_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardshrink_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardshrink_backward, schema_str, "hardshrink_backward(Tensor grad_out, Tensor self, Scalar lambd) -> Tensor")

// aten::hardshrink_backward(Tensor grad_out, Tensor self, Scalar lambd) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<hardshrink_backward::schema> create_hardshrink_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(hardshrink_backward::name, hardshrink_backward::overload_name)
      .typed<hardshrink_backward::schema>();
}

// aten::hardshrink_backward(Tensor grad_out, Tensor self, Scalar lambd) -> Tensor
at::Tensor hardshrink_backward::call(const at::Tensor & grad_out, const at::Tensor & self, const at::Scalar & lambd) {
    static auto op = create_hardshrink_backward_typed_handle();
    return op.call(grad_out, self, lambd);
}

// aten::hardshrink_backward(Tensor grad_out, Tensor self, Scalar lambd) -> Tensor
at::Tensor hardshrink_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_out, const at::Tensor & self, const at::Scalar & lambd) {
    static auto op = create_hardshrink_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_out, self, lambd);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(select_int, name, "aten::select")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(select_int, overload_name, "int")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(select_int, schema_str, "select.int(Tensor(a) self, int dim, int index) -> Tensor(a)")

// aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<select_int::schema> create_select_int_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(select_int::name, select_int::overload_name)
      .typed<select_int::schema>();
}

// aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)
at::Tensor select_int::call(const at::Tensor & self, int64_t dim, int64_t index) {
    static auto op = create_select_int_typed_handle();
    return op.call(self, dim, index);
}

// aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)
at::Tensor select_int::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, int64_t index) {
    static auto op = create_select_int_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(silu_, name, "aten::silu_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(silu_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(silu_, schema_str, "silu_(Tensor(a!) self) -> Tensor(a!)")

// aten::silu_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<silu_::schema> create_silu__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(silu_::name, silu_::overload_name)
      .typed<silu_::schema>();
}

// aten::silu_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & silu_::call(at::Tensor & self) {
    static auto op = create_silu__typed_handle();
    return op.call(self);
}

// aten::silu_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & silu_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_silu__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logit_, name, "aten::logit_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logit_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logit_, schema_str, "logit_(Tensor(a!) self, float? eps=None) -> Tensor(a!)")

// aten::logit_(Tensor(a!) self, float? eps=None) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<logit_::schema> create_logit__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(logit_::name, logit_::overload_name)
      .typed<logit_::schema>();
}

// aten::logit_(Tensor(a!) self, float? eps=None) -> Tensor(a!)
at::Tensor & logit_::call(at::Tensor & self, c10::optional<double> eps) {
    static auto op = create_logit__typed_handle();
    return op.call(self, eps);
}

// aten::logit_(Tensor(a!) self, float? eps=None) -> Tensor(a!)
at::Tensor & logit_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, c10::optional<double> eps) {
    static auto op = create_logit__typed_handle();
    return op.redispatch(dispatchKeySet, self, eps);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logit_out, name, "aten::logit")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logit_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logit_out, schema_str, "logit.out(Tensor self, float? eps=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::logit.out(Tensor self, float? eps=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<logit_out::schema> create_logit_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(logit_out::name, logit_out::overload_name)
      .typed<logit_out::schema>();
}

// aten::logit.out(Tensor self, float? eps=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & logit_out::call(const at::Tensor & self, c10::optional<double> eps, at::Tensor & out) {
    static auto op = create_logit_out_typed_handle();
    return op.call(self, eps, out);
}

// aten::logit.out(Tensor self, float? eps=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & logit_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<double> eps, at::Tensor & out) {
    static auto op = create_logit_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, eps, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sinc, name, "aten::sinc")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sinc, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sinc, schema_str, "sinc(Tensor self) -> Tensor")

// aten::sinc(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<sinc::schema> create_sinc_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sinc::name, sinc::overload_name)
      .typed<sinc::schema>();
}

// aten::sinc(Tensor self) -> Tensor
at::Tensor sinc::call(const at::Tensor & self) {
    static auto op = create_sinc_typed_handle();
    return op.call(self);
}

// aten::sinc(Tensor self) -> Tensor
at::Tensor sinc::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_sinc_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sinc_, name, "aten::sinc_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sinc_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sinc_, schema_str, "sinc_(Tensor(a!) self) -> Tensor(a!)")

// aten::sinc_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<sinc_::schema> create_sinc__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sinc_::name, sinc_::overload_name)
      .typed<sinc_::schema>();
}

// aten::sinc_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & sinc_::call(at::Tensor & self) {
    static auto op = create_sinc__typed_handle();
    return op.call(self);
}

// aten::sinc_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & sinc_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_sinc__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(smm, name, "aten::smm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(smm, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(smm, schema_str, "smm(Tensor self, Tensor mat2) -> Tensor")

// aten::smm(Tensor self, Tensor mat2) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<smm::schema> create_smm_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(smm::name, smm::overload_name)
      .typed<smm::schema>();
}

// aten::smm(Tensor self, Tensor mat2) -> Tensor
at::Tensor smm::call(const at::Tensor & self, const at::Tensor & mat2) {
    static auto op = create_smm_typed_handle();
    return op.call(self, mat2);
}

// aten::smm(Tensor self, Tensor mat2) -> Tensor
at::Tensor smm::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mat2) {
    static auto op = create_smm_typed_handle();
    return op.redispatch(dispatchKeySet, self, mat2);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unsafe_split_with_sizes, name, "aten::unsafe_split_with_sizes")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unsafe_split_with_sizes, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unsafe_split_with_sizes, schema_str, "unsafe_split_with_sizes(Tensor self, int[] split_sizes, int dim=0) -> Tensor[]")

// aten::unsafe_split_with_sizes(Tensor self, int[] split_sizes, int dim=0) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<unsafe_split_with_sizes::schema> create_unsafe_split_with_sizes_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(unsafe_split_with_sizes::name, unsafe_split_with_sizes::overload_name)
      .typed<unsafe_split_with_sizes::schema>();
}

// aten::unsafe_split_with_sizes(Tensor self, int[] split_sizes, int dim=0) -> Tensor[]
::std::vector<at::Tensor> unsafe_split_with_sizes::call(const at::Tensor & self, at::IntArrayRef split_sizes, int64_t dim) {
    static auto op = create_unsafe_split_with_sizes_typed_handle();
    return op.call(self, split_sizes, dim);
}

// aten::unsafe_split_with_sizes(Tensor self, int[] split_sizes, int dim=0) -> Tensor[]
::std::vector<at::Tensor> unsafe_split_with_sizes::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef split_sizes, int64_t dim) {
    static auto op = create_unsafe_split_with_sizes_typed_handle();
    return op.redispatch(dispatchKeySet, self, split_sizes, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hsplit_array, name, "aten::hsplit")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hsplit_array, overload_name, "array")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hsplit_array, schema_str, "hsplit.array(Tensor(a) self, int[] indices) -> Tensor(a)[]")

// aten::hsplit.array(Tensor(a) self, int[] indices) -> Tensor(a)[]
static C10_NOINLINE c10::TypedOperatorHandle<hsplit_array::schema> create_hsplit_array_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(hsplit_array::name, hsplit_array::overload_name)
      .typed<hsplit_array::schema>();
}

// aten::hsplit.array(Tensor(a) self, int[] indices) -> Tensor(a)[]
::std::vector<at::Tensor> hsplit_array::call(const at::Tensor & self, at::IntArrayRef indices) {
    static auto op = create_hsplit_array_typed_handle();
    return op.call(self, indices);
}

// aten::hsplit.array(Tensor(a) self, int[] indices) -> Tensor(a)[]
::std::vector<at::Tensor> hsplit_array::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef indices) {
    static auto op = create_hsplit_array_typed_handle();
    return op.redispatch(dispatchKeySet, self, indices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(squeeze__dimname, name, "aten::squeeze_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(squeeze__dimname, overload_name, "dimname")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(squeeze__dimname, schema_str, "squeeze_.dimname(Tensor(a!) self, Dimname dim) -> Tensor(a!)")

// aten::squeeze_.dimname(Tensor(a!) self, Dimname dim) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<squeeze__dimname::schema> create_squeeze__dimname_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(squeeze__dimname::name, squeeze__dimname::overload_name)
      .typed<squeeze__dimname::schema>();
}

// aten::squeeze_.dimname(Tensor(a!) self, Dimname dim) -> Tensor(a!)
at::Tensor & squeeze__dimname::call(at::Tensor & self, at::Dimname dim) {
    static auto op = create_squeeze__dimname_typed_handle();
    return op.call(self, dim);
}

// aten::squeeze_.dimname(Tensor(a!) self, Dimname dim) -> Tensor(a!)
at::Tensor & squeeze__dimname::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, at::Dimname dim) {
    static auto op = create_squeeze__dimname_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sspaddmm_out, name, "aten::sspaddmm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sspaddmm_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sspaddmm_out, schema_str, "sspaddmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)")

// aten::sspaddmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<sspaddmm_out::schema> create_sspaddmm_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sspaddmm_out::name, sspaddmm_out::overload_name)
      .typed<sspaddmm_out::schema>();
}

// aten::sspaddmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
at::Tensor & sspaddmm_out::call(const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
    static auto op = create_sspaddmm_out_typed_handle();
    return op.call(self, mat1, mat2, beta, alpha, out);
}

// aten::sspaddmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
at::Tensor & sspaddmm_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
    static auto op = create_sspaddmm_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, mat1, mat2, beta, alpha, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_stack_out, name, "aten::_stack")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_stack_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_stack_out, schema_str, "_stack.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)")

// aten::_stack.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<_stack_out::schema> create__stack_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_stack_out::name, _stack_out::overload_name)
      .typed<_stack_out::schema>();
}

// aten::_stack.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & _stack_out::call(at::TensorList tensors, int64_t dim, at::Tensor & out) {
    static auto op = create__stack_out_typed_handle();
    return op.call(tensors, dim, out);
}

// aten::_stack.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & _stack_out::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, int64_t dim, at::Tensor & out) {
    static auto op = create__stack_out_typed_handle();
    return op.redispatch(dispatchKeySet, tensors, dim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hstack_out, name, "aten::hstack")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hstack_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hstack_out, schema_str, "hstack.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)")

// aten::hstack.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<hstack_out::schema> create_hstack_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(hstack_out::name, hstack_out::overload_name)
      .typed<hstack_out::schema>();
}

// aten::hstack.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & hstack_out::call(at::TensorList tensors, at::Tensor & out) {
    static auto op = create_hstack_out_typed_handle();
    return op.call(tensors, out);
}

// aten::hstack.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & hstack_out::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, at::Tensor & out) {
    static auto op = create_hstack_out_typed_handle();
    return op.redispatch(dispatchKeySet, tensors, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(dstack, name, "aten::dstack")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(dstack, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(dstack, schema_str, "dstack(Tensor[] tensors) -> Tensor")

// aten::dstack(Tensor[] tensors) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<dstack::schema> create_dstack_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(dstack::name, dstack::overload_name)
      .typed<dstack::schema>();
}

// aten::dstack(Tensor[] tensors) -> Tensor
at::Tensor dstack::call(at::TensorList tensors) {
    static auto op = create_dstack_typed_handle();
    return op.call(tensors);
}

// aten::dstack(Tensor[] tensors) -> Tensor
at::Tensor dstack::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create_dstack_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(prod, name, "aten::prod")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(prod, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(prod, schema_str, "prod(Tensor self, *, ScalarType? dtype=None) -> Tensor")

// aten::prod(Tensor self, *, ScalarType? dtype=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<prod::schema> create_prod_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(prod::name, prod::overload_name)
      .typed<prod::schema>();
}

// aten::prod(Tensor self, *, ScalarType? dtype=None) -> Tensor
at::Tensor prod::call(const at::Tensor & self, c10::optional<at::ScalarType> dtype) {
    static auto op = create_prod_typed_handle();
    return op.call(self, dtype);
}

// aten::prod(Tensor self, *, ScalarType? dtype=None) -> Tensor
at::Tensor prod::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::ScalarType> dtype) {
    static auto op = create_prod_typed_handle();
    return op.redispatch(dispatchKeySet, self, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(prod_dim_int, name, "aten::prod")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(prod_dim_int, overload_name, "dim_int")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(prod_dim_int, schema_str, "prod.dim_int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor")

// aten::prod.dim_int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<prod_dim_int::schema> create_prod_dim_int_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(prod_dim_int::name, prod_dim_int::overload_name)
      .typed<prod_dim_int::schema>();
}

// aten::prod.dim_int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
at::Tensor prod_dim_int::call(const at::Tensor & self, int64_t dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_prod_dim_int_typed_handle();
    return op.call(self, dim, keepdim, dtype);
}

// aten::prod.dim_int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
at::Tensor prod_dim_int::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_prod_dim_int_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tan, name, "aten::tan")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tan, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tan, schema_str, "tan(Tensor self) -> Tensor")

// aten::tan(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<tan::schema> create_tan_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(tan::name, tan::overload_name)
      .typed<tan::schema>();
}

// aten::tan(Tensor self) -> Tensor
at::Tensor tan::call(const at::Tensor & self) {
    static auto op = create_tan_typed_handle();
    return op.call(self);
}

// aten::tan(Tensor self) -> Tensor
at::Tensor tan::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_tan_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tan_out, name, "aten::tan")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tan_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tan_out, schema_str, "tan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::tan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<tan_out::schema> create_tan_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(tan_out::name, tan_out::overload_name)
      .typed<tan_out::schema>();
}

// aten::tan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & tan_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_tan_out_typed_handle();
    return op.call(self, out);
}

// aten::tan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & tan_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_tan_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tensordot_out, name, "aten::tensordot")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tensordot_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tensordot_out, schema_str, "tensordot.out(Tensor self, Tensor other, int[] dims_self, int[] dims_other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::tensordot.out(Tensor self, Tensor other, int[] dims_self, int[] dims_other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<tensordot_out::schema> create_tensordot_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(tensordot_out::name, tensordot_out::overload_name)
      .typed<tensordot_out::schema>();
}

// aten::tensordot.out(Tensor self, Tensor other, int[] dims_self, int[] dims_other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & tensordot_out::call(const at::Tensor & self, const at::Tensor & other, at::IntArrayRef dims_self, at::IntArrayRef dims_other, at::Tensor & out) {
    static auto op = create_tensordot_out_typed_handle();
    return op.call(self, other, dims_self, dims_other, out);
}

// aten::tensordot.out(Tensor self, Tensor other, int[] dims_self, int[] dims_other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & tensordot_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::IntArrayRef dims_self, at::IntArrayRef dims_other, at::Tensor & out) {
    static auto op = create_tensordot_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, dims_self, dims_other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_mkldnn_transpose_, name, "aten::_mkldnn_transpose_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_mkldnn_transpose_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_mkldnn_transpose_, schema_str, "_mkldnn_transpose_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)")

// aten::_mkldnn_transpose_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<_mkldnn_transpose_::schema> create__mkldnn_transpose__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_mkldnn_transpose_::name, _mkldnn_transpose_::overload_name)
      .typed<_mkldnn_transpose_::schema>();
}

// aten::_mkldnn_transpose_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)
at::Tensor & _mkldnn_transpose_::call(at::Tensor & self, int64_t dim0, int64_t dim1) {
    static auto op = create__mkldnn_transpose__typed_handle();
    return op.call(self, dim0, dim1);
}

// aten::_mkldnn_transpose_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)
at::Tensor & _mkldnn_transpose_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t dim0, int64_t dim1) {
    static auto op = create__mkldnn_transpose__typed_handle();
    return op.redispatch(dispatchKeySet, self, dim0, dim1);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(trunc_out, name, "aten::trunc")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(trunc_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(trunc_out, schema_str, "trunc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::trunc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<trunc_out::schema> create_trunc_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(trunc_out::name, trunc_out::overload_name)
      .typed<trunc_out::schema>();
}

// aten::trunc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & trunc_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_trunc_out_typed_handle();
    return op.call(self, out);
}

// aten::trunc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & trunc_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_trunc_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unique_dim_consecutive, name, "aten::unique_dim_consecutive")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unique_dim_consecutive, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unique_dim_consecutive, schema_str, "unique_dim_consecutive(Tensor self, int dim, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)")

// aten::unique_dim_consecutive(Tensor self, int dim, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<unique_dim_consecutive::schema> create_unique_dim_consecutive_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(unique_dim_consecutive::name, unique_dim_consecutive::overload_name)
      .typed<unique_dim_consecutive::schema>();
}

// aten::unique_dim_consecutive(Tensor self, int dim, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> unique_dim_consecutive::call(const at::Tensor & self, int64_t dim, bool return_inverse, bool return_counts) {
    static auto op = create_unique_dim_consecutive_typed_handle();
    return op.call(self, dim, return_inverse, return_counts);
}

// aten::unique_dim_consecutive(Tensor self, int dim, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> unique_dim_consecutive::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool return_inverse, bool return_counts) {
    static auto op = create_unique_dim_consecutive_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, return_inverse, return_counts);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_unsafe_view, name, "aten::_unsafe_view")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_unsafe_view, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_unsafe_view, schema_str, "_unsafe_view(Tensor self, int[] size) -> Tensor")

// aten::_unsafe_view(Tensor self, int[] size) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_unsafe_view::schema> create__unsafe_view_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_unsafe_view::name, _unsafe_view::overload_name)
      .typed<_unsafe_view::schema>();
}

// aten::_unsafe_view(Tensor self, int[] size) -> Tensor
at::Tensor _unsafe_view::call(const at::Tensor & self, at::IntArrayRef size) {
    static auto op = create__unsafe_view_typed_handle();
    return op.call(self, size);
}

// aten::_unsafe_view(Tensor self, int[] size) -> Tensor
at::Tensor _unsafe_view::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef size) {
    static auto op = create__unsafe_view_typed_handle();
    return op.redispatch(dispatchKeySet, self, size);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unsqueeze, name, "aten::unsqueeze")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unsqueeze, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unsqueeze, schema_str, "unsqueeze(Tensor(a) self, int dim) -> Tensor(a)")

// aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<unsqueeze::schema> create_unsqueeze_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(unsqueeze::name, unsqueeze::overload_name)
      .typed<unsqueeze::schema>();
}

// aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)
at::Tensor unsqueeze::call(const at::Tensor & self, int64_t dim) {
    static auto op = create_unsqueeze_typed_handle();
    return op.call(self, dim);
}

// aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)
at::Tensor unsqueeze::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim) {
    static auto op = create_unsqueeze_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unsqueeze_, name, "aten::unsqueeze_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unsqueeze_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unsqueeze_, schema_str, "unsqueeze_(Tensor(a!) self, int dim) -> Tensor(a!)")

// aten::unsqueeze_(Tensor(a!) self, int dim) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<unsqueeze_::schema> create_unsqueeze__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(unsqueeze_::name, unsqueeze_::overload_name)
      .typed<unsqueeze_::schema>();
}

// aten::unsqueeze_(Tensor(a!) self, int dim) -> Tensor(a!)
at::Tensor & unsqueeze_::call(at::Tensor & self, int64_t dim) {
    static auto op = create_unsqueeze__typed_handle();
    return op.call(self, dim);
}

// aten::unsqueeze_(Tensor(a!) self, int dim) -> Tensor(a!)
at::Tensor & unsqueeze_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t dim) {
    static auto op = create_unsqueeze__typed_handle();
    return op.redispatch(dispatchKeySet, self, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_correction, name, "aten::var")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_correction, overload_name, "correction")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_correction, schema_str, "var.correction(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False) -> Tensor")

// aten::var.correction(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<var_correction::schema> create_var_correction_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(var_correction::name, var_correction::overload_name)
      .typed<var_correction::schema>();
}

// aten::var.correction(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False) -> Tensor
at::Tensor var_correction::call(const at::Tensor & self, c10::optional<at::IntArrayRef> dim, c10::optional<int64_t> correction, bool keepdim) {
    static auto op = create_var_correction_typed_handle();
    return op.call(self, dim, correction, keepdim);
}

// aten::var.correction(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False) -> Tensor
at::Tensor var_correction::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> dim, c10::optional<int64_t> correction, bool keepdim) {
    static auto op = create_var_correction_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, correction, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_s_where, name, "aten::_s_where")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_s_where, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_s_where, schema_str, "_s_where(Tensor condition, Tensor self, Tensor other) -> Tensor")

// aten::_s_where(Tensor condition, Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_s_where::schema> create__s_where_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_s_where::name, _s_where::overload_name)
      .typed<_s_where::schema>();
}

// aten::_s_where(Tensor condition, Tensor self, Tensor other) -> Tensor
at::Tensor _s_where::call(const at::Tensor & condition, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create__s_where_typed_handle();
    return op.call(condition, self, other);
}

// aten::_s_where(Tensor condition, Tensor self, Tensor other) -> Tensor
at::Tensor _s_where::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & condition, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create__s_where_typed_handle();
    return op.redispatch(dispatchKeySet, condition, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(zeros_out, name, "aten::zeros")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(zeros_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(zeros_out, schema_str, "zeros.out(int[] size, *, Tensor(a!) out) -> Tensor(a!)")

// aten::zeros.out(int[] size, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<zeros_out::schema> create_zeros_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(zeros_out::name, zeros_out::overload_name)
      .typed<zeros_out::schema>();
}

// aten::zeros.out(int[] size, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & zeros_out::call(at::IntArrayRef size, at::Tensor & out) {
    static auto op = create_zeros_out_typed_handle();
    return op.call(size, out);
}

// aten::zeros.out(int[] size, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & zeros_out::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, at::Tensor & out) {
    static auto op = create_zeros_out_typed_handle();
    return op.redispatch(dispatchKeySet, size, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(poisson, name, "aten::poisson")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(poisson, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(poisson, schema_str, "poisson(Tensor self, Generator? generator=None) -> Tensor")

// aten::poisson(Tensor self, Generator? generator=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<poisson::schema> create_poisson_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(poisson::name, poisson::overload_name)
      .typed<poisson::schema>();
}

// aten::poisson(Tensor self, Generator? generator=None) -> Tensor
at::Tensor poisson::call(const at::Tensor & self, c10::optional<at::Generator> generator) {
    static auto op = create_poisson_typed_handle();
    return op.call(self, generator);
}

// aten::poisson(Tensor self, Generator? generator=None) -> Tensor
at::Tensor poisson::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::Generator> generator) {
    static auto op = create_poisson_typed_handle();
    return op.redispatch(dispatchKeySet, self, generator);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_softmax_int, name, "aten::_sparse_softmax")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_softmax_int, overload_name, "int")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_softmax_int, schema_str, "_sparse_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor")

// aten::_sparse_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_sparse_softmax_int::schema> create__sparse_softmax_int_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_sparse_softmax_int::name, _sparse_softmax_int::overload_name)
      .typed<_sparse_softmax_int::schema>();
}

// aten::_sparse_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
at::Tensor _sparse_softmax_int::call(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
    static auto op = create__sparse_softmax_int_typed_handle();
    return op.call(self, dim, dtype);
}

// aten::_sparse_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
at::Tensor _sparse_softmax_int::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
    static auto op = create__sparse_softmax_int_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_softmax_Dimname, name, "aten::_sparse_softmax")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_softmax_Dimname, overload_name, "Dimname")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_softmax_Dimname, schema_str, "_sparse_softmax.Dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor")

// aten::_sparse_softmax.Dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_sparse_softmax_Dimname::schema> create__sparse_softmax_Dimname_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_sparse_softmax_Dimname::name, _sparse_softmax_Dimname::overload_name)
      .typed<_sparse_softmax_Dimname::schema>();
}

// aten::_sparse_softmax.Dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
at::Tensor _sparse_softmax_Dimname::call(const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
    static auto op = create__sparse_softmax_Dimname_typed_handle();
    return op.call(self, dim, dtype);
}

// aten::_sparse_softmax.Dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
at::Tensor _sparse_softmax_Dimname::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
    static auto op = create__sparse_softmax_Dimname_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_log_softmax_Dimname, name, "aten::_sparse_log_softmax")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_log_softmax_Dimname, overload_name, "Dimname")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_log_softmax_Dimname, schema_str, "_sparse_log_softmax.Dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor")

// aten::_sparse_log_softmax.Dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_sparse_log_softmax_Dimname::schema> create__sparse_log_softmax_Dimname_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_sparse_log_softmax_Dimname::name, _sparse_log_softmax_Dimname::overload_name)
      .typed<_sparse_log_softmax_Dimname::schema>();
}

// aten::_sparse_log_softmax.Dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
at::Tensor _sparse_log_softmax_Dimname::call(const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
    static auto op = create__sparse_log_softmax_Dimname_typed_handle();
    return op.call(self, dim, dtype);
}

// aten::_sparse_log_softmax.Dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
at::Tensor _sparse_log_softmax_Dimname::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
    static auto op = create__sparse_log_softmax_Dimname_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(norm_names_out, name, "aten::norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(norm_names_out, overload_name, "names_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(norm_names_out, schema_str, "norm.names_out(Tensor self, Scalar? p, Dimname[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)")

// aten::norm.names_out(Tensor self, Scalar? p, Dimname[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<norm_names_out::schema> create_norm_names_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(norm_names_out::name, norm_names_out::overload_name)
      .typed<norm_names_out::schema>();
}

// aten::norm.names_out(Tensor self, Scalar? p, Dimname[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & norm_names_out::call(const at::Tensor & self, const c10::optional<at::Scalar> & p, at::DimnameList dim, bool keepdim, at::Tensor & out) {
    static auto op = create_norm_names_out_typed_handle();
    return op.call(self, p, dim, keepdim, out);
}

// aten::norm.names_out(Tensor self, Scalar? p, Dimname[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & norm_names_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Scalar> & p, at::DimnameList dim, bool keepdim, at::Tensor & out) {
    static auto op = create_norm_names_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, p, dim, keepdim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sub__Tensor, name, "aten::sub_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sub__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sub__Tensor, schema_str, "sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)")

// aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<sub__Tensor::schema> create_sub__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sub__Tensor::name, sub__Tensor::overload_name)
      .typed<sub__Tensor::schema>();
}

// aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
at::Tensor & sub__Tensor::call(at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
    static auto op = create_sub__Tensor_typed_handle();
    return op.call(self, other, alpha);
}

// aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
at::Tensor & sub__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
    static auto op = create_sub__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(subtract__Tensor, name, "aten::subtract_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(subtract__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(subtract__Tensor, schema_str, "subtract_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)")

// aten::subtract_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<subtract__Tensor::schema> create_subtract__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(subtract__Tensor::name, subtract__Tensor::overload_name)
      .typed<subtract__Tensor::schema>();
}

// aten::subtract_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
at::Tensor & subtract__Tensor::call(at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
    static auto op = create_subtract__Tensor_typed_handle();
    return op.call(self, other, alpha);
}

// aten::subtract_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
at::Tensor & subtract__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
    static auto op = create_subtract__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(subtract_Scalar, name, "aten::subtract")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(subtract_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(subtract_Scalar, schema_str, "subtract.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor")

// aten::subtract.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<subtract_Scalar::schema> create_subtract_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(subtract_Scalar::name, subtract_Scalar::overload_name)
      .typed<subtract_Scalar::schema>();
}

// aten::subtract.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
at::Tensor subtract_Scalar::call(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
    static auto op = create_subtract_Scalar_typed_handle();
    return op.call(self, other, alpha);
}

// aten::subtract.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
at::Tensor subtract_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
    static auto op = create_subtract_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(subtract__Scalar, name, "aten::subtract_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(subtract__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(subtract__Scalar, schema_str, "subtract_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)")

// aten::subtract_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<subtract__Scalar::schema> create_subtract__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(subtract__Scalar::name, subtract__Scalar::overload_name)
      .typed<subtract__Scalar::schema>();
}

// aten::subtract_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
at::Tensor & subtract__Scalar::call(at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
    static auto op = create_subtract__Scalar_typed_handle();
    return op.call(self, other, alpha);
}

// aten::subtract_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
at::Tensor & subtract__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
    static auto op = create_subtract__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(heaviside, name, "aten::heaviside")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(heaviside, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(heaviside, schema_str, "heaviside(Tensor self, Tensor values) -> Tensor")

// aten::heaviside(Tensor self, Tensor values) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<heaviside::schema> create_heaviside_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(heaviside::name, heaviside::overload_name)
      .typed<heaviside::schema>();
}

// aten::heaviside(Tensor self, Tensor values) -> Tensor
at::Tensor heaviside::call(const at::Tensor & self, const at::Tensor & values) {
    static auto op = create_heaviside_typed_handle();
    return op.call(self, values);
}

// aten::heaviside(Tensor self, Tensor values) -> Tensor
at::Tensor heaviside::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & values) {
    static auto op = create_heaviside_typed_handle();
    return op.redispatch(dispatchKeySet, self, values);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(to_dense, name, "aten::to_dense")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(to_dense, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(to_dense, schema_str, "to_dense(Tensor self, ScalarType? dtype=None) -> Tensor")

// aten::to_dense(Tensor self, ScalarType? dtype=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<to_dense::schema> create_to_dense_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(to_dense::name, to_dense::overload_name)
      .typed<to_dense::schema>();
}

// aten::to_dense(Tensor self, ScalarType? dtype=None) -> Tensor
at::Tensor to_dense::call(const at::Tensor & self, c10::optional<at::ScalarType> dtype) {
    static auto op = create_to_dense_typed_handle();
    return op.call(self, dtype);
}

// aten::to_dense(Tensor self, ScalarType? dtype=None) -> Tensor
at::Tensor to_dense::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::ScalarType> dtype) {
    static auto op = create_to_dense_typed_handle();
    return op.redispatch(dispatchKeySet, self, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sparse_dim, name, "aten::sparse_dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sparse_dim, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sparse_dim, schema_str, "sparse_dim(Tensor self) -> int")

// aten::sparse_dim(Tensor self) -> int
static C10_NOINLINE c10::TypedOperatorHandle<sparse_dim::schema> create_sparse_dim_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sparse_dim::name, sparse_dim::overload_name)
      .typed<sparse_dim::schema>();
}

// aten::sparse_dim(Tensor self) -> int
int64_t sparse_dim::call(const at::Tensor & self) {
    static auto op = create_sparse_dim_typed_handle();
    return op.call(self);
}

// aten::sparse_dim(Tensor self) -> int
int64_t sparse_dim::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_sparse_dim_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_dimI, name, "aten::_dimI")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_dimI, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_dimI, schema_str, "_dimI(Tensor self) -> int")

// aten::_dimI(Tensor self) -> int
static C10_NOINLINE c10::TypedOperatorHandle<_dimI::schema> create__dimI_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_dimI::name, _dimI::overload_name)
      .typed<_dimI::schema>();
}

// aten::_dimI(Tensor self) -> int
int64_t _dimI::call(const at::Tensor & self) {
    static auto op = create__dimI_typed_handle();
    return op.call(self);
}

// aten::_dimI(Tensor self) -> int
int64_t _dimI::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create__dimI_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_nnz, name, "aten::_nnz")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_nnz, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_nnz, schema_str, "_nnz(Tensor self) -> int")

// aten::_nnz(Tensor self) -> int
static C10_NOINLINE c10::TypedOperatorHandle<_nnz::schema> create__nnz_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_nnz::name, _nnz::overload_name)
      .typed<_nnz::schema>();
}

// aten::_nnz(Tensor self) -> int
int64_t _nnz::call(const at::Tensor & self) {
    static auto op = create__nnz_typed_handle();
    return op.call(self);
}

// aten::_nnz(Tensor self) -> int
int64_t _nnz::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create__nnz_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(to_sparse_sparse_dim, name, "aten::to_sparse")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(to_sparse_sparse_dim, overload_name, "sparse_dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(to_sparse_sparse_dim, schema_str, "to_sparse.sparse_dim(Tensor self, int sparse_dim) -> Tensor")

// aten::to_sparse.sparse_dim(Tensor self, int sparse_dim) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<to_sparse_sparse_dim::schema> create_to_sparse_sparse_dim_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(to_sparse_sparse_dim::name, to_sparse_sparse_dim::overload_name)
      .typed<to_sparse_sparse_dim::schema>();
}

// aten::to_sparse.sparse_dim(Tensor self, int sparse_dim) -> Tensor
at::Tensor to_sparse_sparse_dim::call(const at::Tensor & self, int64_t sparse_dim) {
    static auto op = create_to_sparse_sparse_dim_typed_handle();
    return op.call(self, sparse_dim);
}

// aten::to_sparse.sparse_dim(Tensor self, int sparse_dim) -> Tensor
at::Tensor to_sparse_sparse_dim::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t sparse_dim) {
    static auto op = create_to_sparse_sparse_dim_typed_handle();
    return op.redispatch(dispatchKeySet, self, sparse_dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_reorder_conv3d_weight, name, "aten::mkldnn_reorder_conv3d_weight")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_reorder_conv3d_weight, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_reorder_conv3d_weight, schema_str, "mkldnn_reorder_conv3d_weight(Tensor self, int[3] padding=0, int[3] stride=1, int[3] dilation=1, int groups=1) -> Tensor")

// aten::mkldnn_reorder_conv3d_weight(Tensor self, int[3] padding=0, int[3] stride=1, int[3] dilation=1, int groups=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<mkldnn_reorder_conv3d_weight::schema> create_mkldnn_reorder_conv3d_weight_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mkldnn_reorder_conv3d_weight::name, mkldnn_reorder_conv3d_weight::overload_name)
      .typed<mkldnn_reorder_conv3d_weight::schema>();
}

// aten::mkldnn_reorder_conv3d_weight(Tensor self, int[3] padding=0, int[3] stride=1, int[3] dilation=1, int groups=1) -> Tensor
at::Tensor mkldnn_reorder_conv3d_weight::call(const at::Tensor & self, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups) {
    static auto op = create_mkldnn_reorder_conv3d_weight_typed_handle();
    return op.call(self, padding, stride, dilation, groups);
}

// aten::mkldnn_reorder_conv3d_weight(Tensor self, int[3] padding=0, int[3] stride=1, int[3] dilation=1, int groups=1) -> Tensor
at::Tensor mkldnn_reorder_conv3d_weight::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups) {
    static auto op = create_mkldnn_reorder_conv3d_weight_typed_handle();
    return op.redispatch(dispatchKeySet, self, padding, stride, dilation, groups);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(dequantize_self, name, "aten::dequantize")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(dequantize_self, overload_name, "self")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(dequantize_self, schema_str, "dequantize.self(Tensor self) -> Tensor")

// aten::dequantize.self(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<dequantize_self::schema> create_dequantize_self_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(dequantize_self::name, dequantize_self::overload_name)
      .typed<dequantize_self::schema>();
}

// aten::dequantize.self(Tensor self) -> Tensor
at::Tensor dequantize_self::call(const at::Tensor & self) {
    static auto op = create_dequantize_self_typed_handle();
    return op.call(self);
}

// aten::dequantize.self(Tensor self) -> Tensor
at::Tensor dequantize_self::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_dequantize_self_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(dequantize_tensors, name, "aten::dequantize")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(dequantize_tensors, overload_name, "tensors")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(dequantize_tensors, schema_str, "dequantize.tensors(Tensor[] tensors) -> Tensor[]")

// aten::dequantize.tensors(Tensor[] tensors) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<dequantize_tensors::schema> create_dequantize_tensors_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(dequantize_tensors::name, dequantize_tensors::overload_name)
      .typed<dequantize_tensors::schema>();
}

// aten::dequantize.tensors(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> dequantize_tensors::call(at::TensorList tensors) {
    static auto op = create_dequantize_tensors_typed_handle();
    return op.call(tensors);
}

// aten::dequantize.tensors(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> dequantize_tensors::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create_dequantize_tensors_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(q_scale, name, "aten::q_scale")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(q_scale, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(q_scale, schema_str, "q_scale(Tensor self) -> float")

// aten::q_scale(Tensor self) -> float
static C10_NOINLINE c10::TypedOperatorHandle<q_scale::schema> create_q_scale_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(q_scale::name, q_scale::overload_name)
      .typed<q_scale::schema>();
}

// aten::q_scale(Tensor self) -> float
double q_scale::call(const at::Tensor & self) {
    static auto op = create_q_scale_typed_handle();
    return op.call(self);
}

// aten::q_scale(Tensor self) -> float
double q_scale::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_q_scale_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(q_per_channel_axis, name, "aten::q_per_channel_axis")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(q_per_channel_axis, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(q_per_channel_axis, schema_str, "q_per_channel_axis(Tensor self) -> int")

// aten::q_per_channel_axis(Tensor self) -> int
static C10_NOINLINE c10::TypedOperatorHandle<q_per_channel_axis::schema> create_q_per_channel_axis_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(q_per_channel_axis::name, q_per_channel_axis::overload_name)
      .typed<q_per_channel_axis::schema>();
}

// aten::q_per_channel_axis(Tensor self) -> int
int64_t q_per_channel_axis::call(const at::Tensor & self) {
    static auto op = create_q_per_channel_axis_typed_handle();
    return op.call(self);
}

// aten::q_per_channel_axis(Tensor self) -> int
int64_t q_per_channel_axis::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_q_per_channel_axis_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_make_per_tensor_quantized_tensor, name, "aten::_make_per_tensor_quantized_tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_make_per_tensor_quantized_tensor, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_make_per_tensor_quantized_tensor, schema_str, "_make_per_tensor_quantized_tensor(Tensor self, float scale, int zero_point) -> Tensor")

// aten::_make_per_tensor_quantized_tensor(Tensor self, float scale, int zero_point) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_make_per_tensor_quantized_tensor::schema> create__make_per_tensor_quantized_tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_make_per_tensor_quantized_tensor::name, _make_per_tensor_quantized_tensor::overload_name)
      .typed<_make_per_tensor_quantized_tensor::schema>();
}

// aten::_make_per_tensor_quantized_tensor(Tensor self, float scale, int zero_point) -> Tensor
at::Tensor _make_per_tensor_quantized_tensor::call(const at::Tensor & self, double scale, int64_t zero_point) {
    static auto op = create__make_per_tensor_quantized_tensor_typed_handle();
    return op.call(self, scale, zero_point);
}

// aten::_make_per_tensor_quantized_tensor(Tensor self, float scale, int zero_point) -> Tensor
at::Tensor _make_per_tensor_quantized_tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double scale, int64_t zero_point) {
    static auto op = create__make_per_tensor_quantized_tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, scale, zero_point);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_make_per_channel_quantized_tensor, name, "aten::_make_per_channel_quantized_tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_make_per_channel_quantized_tensor, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_make_per_channel_quantized_tensor, schema_str, "_make_per_channel_quantized_tensor(Tensor self, Tensor scale, Tensor zero_point, int axis) -> Tensor")

// aten::_make_per_channel_quantized_tensor(Tensor self, Tensor scale, Tensor zero_point, int axis) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_make_per_channel_quantized_tensor::schema> create__make_per_channel_quantized_tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_make_per_channel_quantized_tensor::name, _make_per_channel_quantized_tensor::overload_name)
      .typed<_make_per_channel_quantized_tensor::schema>();
}

// aten::_make_per_channel_quantized_tensor(Tensor self, Tensor scale, Tensor zero_point, int axis) -> Tensor
at::Tensor _make_per_channel_quantized_tensor::call(const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis) {
    static auto op = create__make_per_channel_quantized_tensor_typed_handle();
    return op.call(self, scale, zero_point, axis);
}

// aten::_make_per_channel_quantized_tensor(Tensor self, Tensor scale, Tensor zero_point, int axis) -> Tensor
at::Tensor _make_per_channel_quantized_tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis) {
    static auto op = create__make_per_channel_quantized_tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, scale, zero_point, axis);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fake_quantize_per_tensor_affine_cachemask_backward, name, "aten::fake_quantize_per_tensor_affine_cachemask_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fake_quantize_per_tensor_affine_cachemask_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fake_quantize_per_tensor_affine_cachemask_backward, schema_str, "fake_quantize_per_tensor_affine_cachemask_backward(Tensor grad, Tensor mask) -> Tensor")

// aten::fake_quantize_per_tensor_affine_cachemask_backward(Tensor grad, Tensor mask) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fake_quantize_per_tensor_affine_cachemask_backward::schema> create_fake_quantize_per_tensor_affine_cachemask_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fake_quantize_per_tensor_affine_cachemask_backward::name, fake_quantize_per_tensor_affine_cachemask_backward::overload_name)
      .typed<fake_quantize_per_tensor_affine_cachemask_backward::schema>();
}

// aten::fake_quantize_per_tensor_affine_cachemask_backward(Tensor grad, Tensor mask) -> Tensor
at::Tensor fake_quantize_per_tensor_affine_cachemask_backward::call(const at::Tensor & grad, const at::Tensor & mask) {
    static auto op = create_fake_quantize_per_tensor_affine_cachemask_backward_typed_handle();
    return op.call(grad, mask);
}

// aten::fake_quantize_per_tensor_affine_cachemask_backward(Tensor grad, Tensor mask) -> Tensor
at::Tensor fake_quantize_per_tensor_affine_cachemask_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & mask) {
    static auto op = create_fake_quantize_per_tensor_affine_cachemask_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad, mask);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fake_quantize_per_channel_affine_cachemask_backward, name, "aten::fake_quantize_per_channel_affine_cachemask_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fake_quantize_per_channel_affine_cachemask_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fake_quantize_per_channel_affine_cachemask_backward, schema_str, "fake_quantize_per_channel_affine_cachemask_backward(Tensor grad, Tensor mask) -> Tensor")

// aten::fake_quantize_per_channel_affine_cachemask_backward(Tensor grad, Tensor mask) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fake_quantize_per_channel_affine_cachemask_backward::schema> create_fake_quantize_per_channel_affine_cachemask_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fake_quantize_per_channel_affine_cachemask_backward::name, fake_quantize_per_channel_affine_cachemask_backward::overload_name)
      .typed<fake_quantize_per_channel_affine_cachemask_backward::schema>();
}

// aten::fake_quantize_per_channel_affine_cachemask_backward(Tensor grad, Tensor mask) -> Tensor
at::Tensor fake_quantize_per_channel_affine_cachemask_backward::call(const at::Tensor & grad, const at::Tensor & mask) {
    static auto op = create_fake_quantize_per_channel_affine_cachemask_backward_typed_handle();
    return op.call(grad, mask);
}

// aten::fake_quantize_per_channel_affine_cachemask_backward(Tensor grad, Tensor mask) -> Tensor
at::Tensor fake_quantize_per_channel_affine_cachemask_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & mask) {
    static auto op = create_fake_quantize_per_channel_affine_cachemask_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad, mask);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_saturate_weight_to_fp16, name, "aten::_saturate_weight_to_fp16")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_saturate_weight_to_fp16, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_saturate_weight_to_fp16, schema_str, "_saturate_weight_to_fp16(Tensor weight) -> Tensor")

// aten::_saturate_weight_to_fp16(Tensor weight) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_saturate_weight_to_fp16::schema> create__saturate_weight_to_fp16_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_saturate_weight_to_fp16::name, _saturate_weight_to_fp16::overload_name)
      .typed<_saturate_weight_to_fp16::schema>();
}

// aten::_saturate_weight_to_fp16(Tensor weight) -> Tensor
at::Tensor _saturate_weight_to_fp16::call(const at::Tensor & weight) {
    static auto op = create__saturate_weight_to_fp16_typed_handle();
    return op.call(weight);
}

// aten::_saturate_weight_to_fp16(Tensor weight) -> Tensor
at::Tensor _saturate_weight_to_fp16::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & weight) {
    static auto op = create__saturate_weight_to_fp16_typed_handle();
    return op.redispatch(dispatchKeySet, weight);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_thnn_fused_lstm_cell_backward, name, "aten::_thnn_fused_lstm_cell_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_thnn_fused_lstm_cell_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_thnn_fused_lstm_cell_backward, schema_str, "_thnn_fused_lstm_cell_backward(Tensor? grad_hy, Tensor? grad_cy, Tensor cx, Tensor cy, Tensor workspace, bool has_bias) -> (Tensor, Tensor, Tensor, Tensor, Tensor)")

// aten::_thnn_fused_lstm_cell_backward(Tensor? grad_hy, Tensor? grad_cy, Tensor cx, Tensor cy, Tensor workspace, bool has_bias) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<_thnn_fused_lstm_cell_backward::schema> create__thnn_fused_lstm_cell_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_thnn_fused_lstm_cell_backward::name, _thnn_fused_lstm_cell_backward::overload_name)
      .typed<_thnn_fused_lstm_cell_backward::schema>();
}

// aten::_thnn_fused_lstm_cell_backward(Tensor? grad_hy, Tensor? grad_cy, Tensor cx, Tensor cy, Tensor workspace, bool has_bias) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> _thnn_fused_lstm_cell_backward::call(const c10::optional<at::Tensor> & grad_hy, const c10::optional<at::Tensor> & grad_cy, const at::Tensor & cx, const at::Tensor & cy, const at::Tensor & workspace, bool has_bias) {
    static auto op = create__thnn_fused_lstm_cell_backward_typed_handle();
    return op.call(grad_hy, grad_cy, cx, cy, workspace, has_bias);
}

// aten::_thnn_fused_lstm_cell_backward(Tensor? grad_hy, Tensor? grad_cy, Tensor cx, Tensor cy, Tensor workspace, bool has_bias) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> _thnn_fused_lstm_cell_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const c10::optional<at::Tensor> & grad_hy, const c10::optional<at::Tensor> & grad_cy, const at::Tensor & cx, const at::Tensor & cy, const at::Tensor & workspace, bool has_bias) {
    static auto op = create__thnn_fused_lstm_cell_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_hy, grad_cy, cx, cy, workspace, has_bias);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lstm_data, name, "aten::lstm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lstm_data, overload_name, "data")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lstm_data, schema_str, "lstm.data(Tensor data, Tensor batch_sizes, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor, Tensor)")

// aten::lstm.data(Tensor data, Tensor batch_sizes, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<lstm_data::schema> create_lstm_data_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(lstm_data::name, lstm_data::overload_name)
      .typed<lstm_data::schema>();
}

// aten::lstm.data(Tensor data, Tensor batch_sizes, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> lstm_data::call(const at::Tensor & data, const at::Tensor & batch_sizes, at::TensorList hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {
    static auto op = create_lstm_data_typed_handle();
    return op.call(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
}

// aten::lstm.data(Tensor data, Tensor batch_sizes, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> lstm_data::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & data, const at::Tensor & batch_sizes, at::TensorList hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {
    static auto op = create_lstm_data_typed_handle();
    return op.redispatch(dispatchKeySet, data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rnn_relu_data, name, "aten::rnn_relu")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rnn_relu_data, overload_name, "data")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rnn_relu_data, schema_str, "rnn_relu.data(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)")

// aten::rnn_relu.data(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<rnn_relu_data::schema> create_rnn_relu_data_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(rnn_relu_data::name, rnn_relu_data::overload_name)
      .typed<rnn_relu_data::schema>();
}

// aten::rnn_relu.data(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> rnn_relu_data::call(const at::Tensor & data, const at::Tensor & batch_sizes, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {
    static auto op = create_rnn_relu_data_typed_handle();
    return op.call(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
}

// aten::rnn_relu.data(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> rnn_relu_data::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & data, const at::Tensor & batch_sizes, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {
    static auto op = create_rnn_relu_data_typed_handle();
    return op.redispatch(dispatchKeySet, data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lstm_cell, name, "aten::lstm_cell")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lstm_cell, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lstm_cell, schema_str, "lstm_cell(Tensor input, Tensor[] hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> (Tensor, Tensor)")

// aten::lstm_cell(Tensor input, Tensor[] hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<lstm_cell::schema> create_lstm_cell_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(lstm_cell::name, lstm_cell::overload_name)
      .typed<lstm_cell::schema>();
}

// aten::lstm_cell(Tensor input, Tensor[] hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> lstm_cell::call(const at::Tensor & input, at::TensorList hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const c10::optional<at::Tensor> & b_ih, const c10::optional<at::Tensor> & b_hh) {
    static auto op = create_lstm_cell_typed_handle();
    return op.call(input, hx, w_ih, w_hh, b_ih, b_hh);
}

// aten::lstm_cell(Tensor input, Tensor[] hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> lstm_cell::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, at::TensorList hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const c10::optional<at::Tensor> & b_ih, const c10::optional<at::Tensor> & b_hh) {
    static auto op = create_lstm_cell_typed_handle();
    return op.redispatch(dispatchKeySet, input, hx, w_ih, w_hh, b_ih, b_hh);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantized_rnn_relu_cell, name, "aten::quantized_rnn_relu_cell")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantized_rnn_relu_cell, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantized_rnn_relu_cell, schema_str, "quantized_rnn_relu_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor")

// aten::quantized_rnn_relu_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<quantized_rnn_relu_cell::schema> create_quantized_rnn_relu_cell_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(quantized_rnn_relu_cell::name, quantized_rnn_relu_cell::overload_name)
      .typed<quantized_rnn_relu_cell::schema>();
}

// aten::quantized_rnn_relu_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor
at::Tensor quantized_rnn_relu_cell::call(const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const at::Tensor & b_ih, const at::Tensor & b_hh, const at::Tensor & packed_ih, const at::Tensor & packed_hh, const at::Tensor & col_offsets_ih, const at::Tensor & col_offsets_hh, const at::Scalar & scale_ih, const at::Scalar & scale_hh, const at::Scalar & zero_point_ih, const at::Scalar & zero_point_hh) {
    static auto op = create_quantized_rnn_relu_cell_typed_handle();
    return op.call(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
}

// aten::quantized_rnn_relu_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor
at::Tensor quantized_rnn_relu_cell::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const at::Tensor & b_ih, const at::Tensor & b_hh, const at::Tensor & packed_ih, const at::Tensor & packed_hh, const at::Tensor & col_offsets_ih, const at::Tensor & col_offsets_hh, const at::Scalar & scale_ih, const at::Scalar & scale_hh, const at::Scalar & zero_point_ih, const at::Scalar & zero_point_hh) {
    static auto op = create_quantized_rnn_relu_cell_typed_handle();
    return op.redispatch(dispatchKeySet, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(masked_scatter_, name, "aten::masked_scatter_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(masked_scatter_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(masked_scatter_, schema_str, "masked_scatter_(Tensor(a!) self, Tensor mask, Tensor source) -> Tensor(a!)")

// aten::masked_scatter_(Tensor(a!) self, Tensor mask, Tensor source) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<masked_scatter_::schema> create_masked_scatter__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(masked_scatter_::name, masked_scatter_::overload_name)
      .typed<masked_scatter_::schema>();
}

// aten::masked_scatter_(Tensor(a!) self, Tensor mask, Tensor source) -> Tensor(a!)
at::Tensor & masked_scatter_::call(at::Tensor & self, const at::Tensor & mask, const at::Tensor & source) {
    static auto op = create_masked_scatter__typed_handle();
    return op.call(self, mask, source);
}

// aten::masked_scatter_(Tensor(a!) self, Tensor mask, Tensor source) -> Tensor(a!)
at::Tensor & masked_scatter_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & mask, const at::Tensor & source) {
    static auto op = create_masked_scatter__typed_handle();
    return op.redispatch(dispatchKeySet, self, mask, source);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(masked_scatter, name, "aten::masked_scatter")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(masked_scatter, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(masked_scatter, schema_str, "masked_scatter(Tensor self, Tensor mask, Tensor source) -> Tensor")

// aten::masked_scatter(Tensor self, Tensor mask, Tensor source) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<masked_scatter::schema> create_masked_scatter_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(masked_scatter::name, masked_scatter::overload_name)
      .typed<masked_scatter::schema>();
}

// aten::masked_scatter(Tensor self, Tensor mask, Tensor source) -> Tensor
at::Tensor masked_scatter::call(const at::Tensor & self, const at::Tensor & mask, const at::Tensor & source) {
    static auto op = create_masked_scatter_typed_handle();
    return op.call(self, mask, source);
}

// aten::masked_scatter(Tensor self, Tensor mask, Tensor source) -> Tensor
at::Tensor masked_scatter::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mask, const at::Tensor & source) {
    static auto op = create_masked_scatter_typed_handle();
    return op.redispatch(dispatchKeySet, self, mask, source);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(put_, name, "aten::put_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(put_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(put_, schema_str, "put_(Tensor(a!) self, Tensor index, Tensor source, bool accumulate=False) -> Tensor(a!)")

// aten::put_(Tensor(a!) self, Tensor index, Tensor source, bool accumulate=False) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<put_::schema> create_put__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(put_::name, put_::overload_name)
      .typed<put_::schema>();
}

// aten::put_(Tensor(a!) self, Tensor index, Tensor source, bool accumulate=False) -> Tensor(a!)
at::Tensor & put_::call(at::Tensor & self, const at::Tensor & index, const at::Tensor & source, bool accumulate) {
    static auto op = create_put__typed_handle();
    return op.call(self, index, source, accumulate);
}

// aten::put_(Tensor(a!) self, Tensor index, Tensor source, bool accumulate=False) -> Tensor(a!)
at::Tensor & put_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & index, const at::Tensor & source, bool accumulate) {
    static auto op = create_put__typed_handle();
    return op.redispatch(dispatchKeySet, self, index, source, accumulate);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_add_, name, "aten::index_add_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_add_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_add_, schema_str, "index_add_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)")

// aten::index_add_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<index_add_::schema> create_index_add__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(index_add_::name, index_add_::overload_name)
      .typed<index_add_::schema>();
}

// aten::index_add_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)
at::Tensor & index_add_::call(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source) {
    static auto op = create_index_add__typed_handle();
    return op.call(self, dim, index, source);
}

// aten::index_add_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)
at::Tensor & index_add_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source) {
    static auto op = create_index_add__typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, source);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_add, name, "aten::index_add")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_add, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_add, schema_str, "index_add(Tensor self, int dim, Tensor index, Tensor source) -> Tensor")

// aten::index_add(Tensor self, int dim, Tensor index, Tensor source) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<index_add::schema> create_index_add_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(index_add::name, index_add::overload_name)
      .typed<index_add::schema>();
}

// aten::index_add(Tensor self, int dim, Tensor index, Tensor source) -> Tensor
at::Tensor index_add::call(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source) {
    static auto op = create_index_add_typed_handle();
    return op.call(self, dim, index, source);
}

// aten::index_add(Tensor self, int dim, Tensor index, Tensor source) -> Tensor
at::Tensor index_add::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source) {
    static auto op = create_index_add_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, source);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_fill__Dimname_Scalar, name, "aten::index_fill_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_fill__Dimname_Scalar, overload_name, "Dimname_Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_fill__Dimname_Scalar, schema_str, "index_fill_.Dimname_Scalar(Tensor(a!) self, Dimname dim, Tensor index, Scalar value) -> Tensor(a!)")

// aten::index_fill_.Dimname_Scalar(Tensor(a!) self, Dimname dim, Tensor index, Scalar value) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<index_fill__Dimname_Scalar::schema> create_index_fill__Dimname_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(index_fill__Dimname_Scalar::name, index_fill__Dimname_Scalar::overload_name)
      .typed<index_fill__Dimname_Scalar::schema>();
}

// aten::index_fill_.Dimname_Scalar(Tensor(a!) self, Dimname dim, Tensor index, Scalar value) -> Tensor(a!)
at::Tensor & index_fill__Dimname_Scalar::call(at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Scalar & value) {
    static auto op = create_index_fill__Dimname_Scalar_typed_handle();
    return op.call(self, dim, index, value);
}

// aten::index_fill_.Dimname_Scalar(Tensor(a!) self, Dimname dim, Tensor index, Scalar value) -> Tensor(a!)
at::Tensor & index_fill__Dimname_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Scalar & value) {
    static auto op = create_index_fill__Dimname_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, value);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter__value, name, "aten::scatter_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter__value, overload_name, "value")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter__value, schema_str, "scatter_.value(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)")

// aten::scatter_.value(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<scatter__value::schema> create_scatter__value_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(scatter__value::name, scatter__value::overload_name)
      .typed<scatter__value::schema>();
}

// aten::scatter_.value(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)
at::Tensor & scatter__value::call(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) {
    static auto op = create_scatter__value_typed_handle();
    return op.call(self, dim, index, value);
}

// aten::scatter_.value(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)
at::Tensor & scatter__value::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) {
    static auto op = create_scatter__value_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, value);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter__reduce, name, "aten::scatter_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter__reduce, overload_name, "reduce")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter__reduce, schema_str, "scatter_.reduce(Tensor(a!) self, int dim, Tensor index, Tensor src, *, str reduce) -> Tensor(a!)")

// aten::scatter_.reduce(Tensor(a!) self, int dim, Tensor index, Tensor src, *, str reduce) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<scatter__reduce::schema> create_scatter__reduce_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(scatter__reduce::name, scatter__reduce::overload_name)
      .typed<scatter__reduce::schema>();
}

// aten::scatter_.reduce(Tensor(a!) self, int dim, Tensor index, Tensor src, *, str reduce) -> Tensor(a!)
at::Tensor & scatter__reduce::call(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce) {
    static auto op = create_scatter__reduce_typed_handle();
    return op.call(self, dim, index, src, reduce);
}

// aten::scatter_.reduce(Tensor(a!) self, int dim, Tensor index, Tensor src, *, str reduce) -> Tensor(a!)
at::Tensor & scatter__reduce::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce) {
    static auto op = create_scatter__reduce_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, src, reduce);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_reduce_out, name, "aten::scatter")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_reduce_out, overload_name, "reduce_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_reduce_out, schema_str, "scatter.reduce_out(Tensor self, int dim, Tensor index, Tensor src, *, str reduce, Tensor(a!) out) -> Tensor(a!)")

// aten::scatter.reduce_out(Tensor self, int dim, Tensor index, Tensor src, *, str reduce, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<scatter_reduce_out::schema> create_scatter_reduce_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(scatter_reduce_out::name, scatter_reduce_out::overload_name)
      .typed<scatter_reduce_out::schema>();
}

// aten::scatter.reduce_out(Tensor self, int dim, Tensor index, Tensor src, *, str reduce, Tensor(a!) out) -> Tensor(a!)
at::Tensor & scatter_reduce_out::call(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce, at::Tensor & out) {
    static auto op = create_scatter_reduce_out_typed_handle();
    return op.call(self, dim, index, src, reduce, out);
}

// aten::scatter.reduce_out(Tensor self, int dim, Tensor index, Tensor src, *, str reduce, Tensor(a!) out) -> Tensor(a!)
at::Tensor & scatter_reduce_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce, at::Tensor & out) {
    static auto op = create_scatter_reduce_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, src, reduce, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_value_reduce_out, name, "aten::scatter")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_value_reduce_out, overload_name, "value_reduce_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_value_reduce_out, schema_str, "scatter.value_reduce_out(Tensor self, int dim, Tensor index, Scalar value, *, str reduce, Tensor(a!) out) -> Tensor(a!)")

// aten::scatter.value_reduce_out(Tensor self, int dim, Tensor index, Scalar value, *, str reduce, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<scatter_value_reduce_out::schema> create_scatter_value_reduce_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(scatter_value_reduce_out::name, scatter_value_reduce_out::overload_name)
      .typed<scatter_value_reduce_out::schema>();
}

// aten::scatter.value_reduce_out(Tensor self, int dim, Tensor index, Scalar value, *, str reduce, Tensor(a!) out) -> Tensor(a!)
at::Tensor & scatter_value_reduce_out::call(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, c10::string_view reduce, at::Tensor & out) {
    static auto op = create_scatter_value_reduce_out_typed_handle();
    return op.call(self, dim, index, value, reduce, out);
}

// aten::scatter.value_reduce_out(Tensor self, int dim, Tensor index, Scalar value, *, str reduce, Tensor(a!) out) -> Tensor(a!)
at::Tensor & scatter_value_reduce_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, c10::string_view reduce, at::Tensor & out) {
    static auto op = create_scatter_value_reduce_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, value, reduce, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_dimname_value, name, "aten::scatter")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_dimname_value, overload_name, "dimname_value")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_dimname_value, schema_str, "scatter.dimname_value(Tensor self, Dimname dim, Tensor index, Scalar value) -> Tensor")

// aten::scatter.dimname_value(Tensor self, Dimname dim, Tensor index, Scalar value) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<scatter_dimname_value::schema> create_scatter_dimname_value_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(scatter_dimname_value::name, scatter_dimname_value::overload_name)
      .typed<scatter_dimname_value::schema>();
}

// aten::scatter.dimname_value(Tensor self, Dimname dim, Tensor index, Scalar value) -> Tensor
at::Tensor scatter_dimname_value::call(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Scalar & value) {
    static auto op = create_scatter_dimname_value_typed_handle();
    return op.call(self, dim, index, value);
}

// aten::scatter.dimname_value(Tensor self, Dimname dim, Tensor index, Scalar value) -> Tensor
at::Tensor scatter_dimname_value::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Scalar & value) {
    static auto op = create_scatter_dimname_value_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, value);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(eq__Scalar, name, "aten::eq_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(eq__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(eq__Scalar, schema_str, "eq_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)")

// aten::eq_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<eq__Scalar::schema> create_eq__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(eq__Scalar::name, eq__Scalar::overload_name)
      .typed<eq__Scalar::schema>();
}

// aten::eq_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & eq__Scalar::call(at::Tensor & self, const at::Scalar & other) {
    static auto op = create_eq__Scalar_typed_handle();
    return op.call(self, other);
}

// aten::eq_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & eq__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
    static auto op = create_eq__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_and_Scalar, name, "aten::bitwise_and")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_and_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_and_Scalar, schema_str, "bitwise_and.Scalar(Tensor self, Scalar other) -> Tensor")

// aten::bitwise_and.Scalar(Tensor self, Scalar other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<bitwise_and_Scalar::schema> create_bitwise_and_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bitwise_and_Scalar::name, bitwise_and_Scalar::overload_name)
      .typed<bitwise_and_Scalar::schema>();
}

// aten::bitwise_and.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor bitwise_and_Scalar::call(const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_bitwise_and_Scalar_typed_handle();
    return op.call(self, other);
}

// aten::bitwise_and.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor bitwise_and_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_bitwise_and_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_and__Tensor, name, "aten::bitwise_and_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_and__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_and__Tensor, schema_str, "bitwise_and_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::bitwise_and_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<bitwise_and__Tensor::schema> create_bitwise_and__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bitwise_and__Tensor::name, bitwise_and__Tensor::overload_name)
      .typed<bitwise_and__Tensor::schema>();
}

// aten::bitwise_and_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & bitwise_and__Tensor::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create_bitwise_and__Tensor_typed_handle();
    return op.call(self, other);
}

// aten::bitwise_and_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & bitwise_and__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create_bitwise_and__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_or_Tensor, name, "aten::bitwise_or")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_or_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_or_Tensor, schema_str, "bitwise_or.Tensor(Tensor self, Tensor other) -> Tensor")

// aten::bitwise_or.Tensor(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<bitwise_or_Tensor::schema> create_bitwise_or_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bitwise_or_Tensor::name, bitwise_or_Tensor::overload_name)
      .typed<bitwise_or_Tensor::schema>();
}

// aten::bitwise_or.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor bitwise_or_Tensor::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_bitwise_or_Tensor_typed_handle();
    return op.call(self, other);
}

// aten::bitwise_or.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor bitwise_or_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_bitwise_or_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_or__Scalar, name, "aten::bitwise_or_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_or__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_or__Scalar, schema_str, "bitwise_or_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)")

// aten::bitwise_or_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<bitwise_or__Scalar::schema> create_bitwise_or__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bitwise_or__Scalar::name, bitwise_or__Scalar::overload_name)
      .typed<bitwise_or__Scalar::schema>();
}

// aten::bitwise_or_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & bitwise_or__Scalar::call(at::Tensor & self, const at::Scalar & other) {
    static auto op = create_bitwise_or__Scalar_typed_handle();
    return op.call(self, other);
}

// aten::bitwise_or_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & bitwise_or__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
    static auto op = create_bitwise_or__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_or__Tensor, name, "aten::bitwise_or_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_or__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_or__Tensor, schema_str, "bitwise_or_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::bitwise_or_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<bitwise_or__Tensor::schema> create_bitwise_or__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bitwise_or__Tensor::name, bitwise_or__Tensor::overload_name)
      .typed<bitwise_or__Tensor::schema>();
}

// aten::bitwise_or_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & bitwise_or__Tensor::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create_bitwise_or__Tensor_typed_handle();
    return op.call(self, other);
}

// aten::bitwise_or_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & bitwise_or__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create_bitwise_or__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__ior___Tensor, name, "aten::__ior__")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__ior___Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__ior___Tensor, schema_str, "__ior__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::__ior__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<__ior___Tensor::schema> create___ior___Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(__ior___Tensor::name, __ior___Tensor::overload_name)
      .typed<__ior___Tensor::schema>();
}

// aten::__ior__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & __ior___Tensor::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create___ior___Tensor_typed_handle();
    return op.call(self, other);
}

// aten::__ior__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & __ior___Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create___ior___Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_xor_Tensor, name, "aten::bitwise_xor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_xor_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_xor_Tensor, schema_str, "bitwise_xor.Tensor(Tensor self, Tensor other) -> Tensor")

// aten::bitwise_xor.Tensor(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<bitwise_xor_Tensor::schema> create_bitwise_xor_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bitwise_xor_Tensor::name, bitwise_xor_Tensor::overload_name)
      .typed<bitwise_xor_Tensor::schema>();
}

// aten::bitwise_xor.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor bitwise_xor_Tensor::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_bitwise_xor_Tensor_typed_handle();
    return op.call(self, other);
}

// aten::bitwise_xor.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor bitwise_xor_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_bitwise_xor_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_xor__Scalar, name, "aten::bitwise_xor_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_xor__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_xor__Scalar, schema_str, "bitwise_xor_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)")

// aten::bitwise_xor_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<bitwise_xor__Scalar::schema> create_bitwise_xor__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bitwise_xor__Scalar::name, bitwise_xor__Scalar::overload_name)
      .typed<bitwise_xor__Scalar::schema>();
}

// aten::bitwise_xor_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & bitwise_xor__Scalar::call(at::Tensor & self, const at::Scalar & other) {
    static auto op = create_bitwise_xor__Scalar_typed_handle();
    return op.call(self, other);
}

// aten::bitwise_xor_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & bitwise_xor__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
    static auto op = create_bitwise_xor__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__ixor___Tensor, name, "aten::__ixor__")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__ixor___Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__ixor___Tensor, schema_str, "__ixor__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::__ixor__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<__ixor___Tensor::schema> create___ixor___Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(__ixor___Tensor::name, __ixor___Tensor::overload_name)
      .typed<__ixor___Tensor::schema>();
}

// aten::__ixor__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & __ixor___Tensor::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create___ixor___Tensor_typed_handle();
    return op.call(self, other);
}

// aten::__ixor__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & __ixor___Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create___ixor___Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__rshift___Scalar, name, "aten::__rshift__")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__rshift___Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__rshift___Scalar, schema_str, "__rshift__.Scalar(Tensor self, Scalar other) -> Tensor")

// aten::__rshift__.Scalar(Tensor self, Scalar other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<__rshift___Scalar::schema> create___rshift___Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(__rshift___Scalar::name, __rshift___Scalar::overload_name)
      .typed<__rshift___Scalar::schema>();
}

// aten::__rshift__.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor __rshift___Scalar::call(const at::Tensor & self, const at::Scalar & other) {
    static auto op = create___rshift___Scalar_typed_handle();
    return op.call(self, other);
}

// aten::__rshift__.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor __rshift___Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
    static auto op = create___rshift___Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_right_shift_Tensor_Scalar_out, name, "aten::bitwise_right_shift")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_right_shift_Tensor_Scalar_out, overload_name, "Tensor_Scalar_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_right_shift_Tensor_Scalar_out, schema_str, "bitwise_right_shift.Tensor_Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::bitwise_right_shift.Tensor_Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<bitwise_right_shift_Tensor_Scalar_out::schema> create_bitwise_right_shift_Tensor_Scalar_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bitwise_right_shift_Tensor_Scalar_out::name, bitwise_right_shift_Tensor_Scalar_out::overload_name)
      .typed<bitwise_right_shift_Tensor_Scalar_out::schema>();
}

// aten::bitwise_right_shift.Tensor_Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & bitwise_right_shift_Tensor_Scalar_out::call(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_bitwise_right_shift_Tensor_Scalar_out_typed_handle();
    return op.call(self, other, out);
}

// aten::bitwise_right_shift.Tensor_Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & bitwise_right_shift_Tensor_Scalar_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_bitwise_right_shift_Tensor_Scalar_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_right_shift_Scalar_Tensor, name, "aten::bitwise_right_shift")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_right_shift_Scalar_Tensor, overload_name, "Scalar_Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_right_shift_Scalar_Tensor, schema_str, "bitwise_right_shift.Scalar_Tensor(Scalar self, Tensor other) -> Tensor")

// aten::bitwise_right_shift.Scalar_Tensor(Scalar self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<bitwise_right_shift_Scalar_Tensor::schema> create_bitwise_right_shift_Scalar_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bitwise_right_shift_Scalar_Tensor::name, bitwise_right_shift_Scalar_Tensor::overload_name)
      .typed<bitwise_right_shift_Scalar_Tensor::schema>();
}

// aten::bitwise_right_shift.Scalar_Tensor(Scalar self, Tensor other) -> Tensor
at::Tensor bitwise_right_shift_Scalar_Tensor::call(const at::Scalar & self, const at::Tensor & other) {
    static auto op = create_bitwise_right_shift_Scalar_Tensor_typed_handle();
    return op.call(self, other);
}

// aten::bitwise_right_shift.Scalar_Tensor(Scalar self, Tensor other) -> Tensor
at::Tensor bitwise_right_shift_Scalar_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Scalar & self, const at::Tensor & other) {
    static auto op = create_bitwise_right_shift_Scalar_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(random__to, name, "aten::random_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(random__to, overload_name, "to")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(random__to, schema_str, "random_.to(Tensor(a!) self, int to, *, Generator? generator=None) -> Tensor(a!)")

// aten::random_.to(Tensor(a!) self, int to, *, Generator? generator=None) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<random__to::schema> create_random__to_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(random__to::name, random__to::overload_name)
      .typed<random__to::schema>();
}

// aten::random_.to(Tensor(a!) self, int to, *, Generator? generator=None) -> Tensor(a!)
at::Tensor & random__to::call(at::Tensor & self, int64_t to, c10::optional<at::Generator> generator) {
    static auto op = create_random__to_typed_handle();
    return op.call(self, to, generator);
}

// aten::random_.to(Tensor(a!) self, int to, *, Generator? generator=None) -> Tensor(a!)
at::Tensor & random__to::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t to, c10::optional<at::Generator> generator) {
    static auto op = create_random__to_typed_handle();
    return op.redispatch(dispatchKeySet, self, to, generator);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log_normal_, name, "aten::log_normal_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log_normal_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log_normal_, schema_str, "log_normal_(Tensor(a!) self, float mean=1, float std=2, *, Generator? generator=None) -> Tensor(a!)")

// aten::log_normal_(Tensor(a!) self, float mean=1, float std=2, *, Generator? generator=None) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<log_normal_::schema> create_log_normal__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(log_normal_::name, log_normal_::overload_name)
      .typed<log_normal_::schema>();
}

// aten::log_normal_(Tensor(a!) self, float mean=1, float std=2, *, Generator? generator=None) -> Tensor(a!)
at::Tensor & log_normal_::call(at::Tensor & self, double mean, double std, c10::optional<at::Generator> generator) {
    static auto op = create_log_normal__typed_handle();
    return op.call(self, mean, std, generator);
}

// aten::log_normal_(Tensor(a!) self, float mean=1, float std=2, *, Generator? generator=None) -> Tensor(a!)
at::Tensor & log_normal_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, double mean, double std, c10::optional<at::Generator> generator) {
    static auto op = create_log_normal__typed_handle();
    return op.redispatch(dispatchKeySet, self, mean, std, generator);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(diag, name, "aten::diag")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(diag, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(diag, schema_str, "diag(Tensor self, int diagonal=0) -> Tensor")

// aten::diag(Tensor self, int diagonal=0) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<diag::schema> create_diag_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(diag::name, diag::overload_name)
      .typed<diag::schema>();
}

// aten::diag(Tensor self, int diagonal=0) -> Tensor
at::Tensor diag::call(const at::Tensor & self, int64_t diagonal) {
    static auto op = create_diag_typed_handle();
    return op.call(self, diagonal);
}

// aten::diag(Tensor self, int diagonal=0) -> Tensor
at::Tensor diag::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t diagonal) {
    static auto op = create_diag_typed_handle();
    return op.redispatch(dispatchKeySet, self, diagonal);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(triu_indices, name, "aten::triu_indices")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(triu_indices, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(triu_indices, schema_str, "triu_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::triu_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<triu_indices::schema> create_triu_indices_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(triu_indices::name, triu_indices::overload_name)
      .typed<triu_indices::schema>();
}

// aten::triu_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor triu_indices::call(int64_t row, int64_t col, int64_t offset, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_triu_indices_typed_handle();
    return op.call(row, col, offset, dtype, layout, device, pin_memory);
}

// aten::triu_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor triu_indices::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t row, int64_t col, int64_t offset, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_triu_indices_typed_handle();
    return op.redispatch(dispatchKeySet, row, col, offset, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(trace, name, "aten::trace")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(trace, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(trace, schema_str, "trace(Tensor self) -> Tensor")

// aten::trace(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<trace::schema> create_trace_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(trace::name, trace::overload_name)
      .typed<trace::schema>();
}

// aten::trace(Tensor self) -> Tensor
at::Tensor trace::call(const at::Tensor & self) {
    static auto op = create_trace_typed_handle();
    return op.call(self);
}

// aten::trace(Tensor self) -> Tensor
at::Tensor trace::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_trace_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ne__Scalar, name, "aten::ne_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ne__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ne__Scalar, schema_str, "ne_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)")

// aten::ne_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<ne__Scalar::schema> create_ne__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(ne__Scalar::name, ne__Scalar::overload_name)
      .typed<ne__Scalar::schema>();
}

// aten::ne_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & ne__Scalar::call(at::Tensor & self, const at::Scalar & other) {
    static auto op = create_ne__Scalar_typed_handle();
    return op.call(self, other);
}

// aten::ne_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & ne__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
    static auto op = create_ne__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(not_equal_Scalar, name, "aten::not_equal")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(not_equal_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(not_equal_Scalar, schema_str, "not_equal.Scalar(Tensor self, Scalar other) -> Tensor")

// aten::not_equal.Scalar(Tensor self, Scalar other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<not_equal_Scalar::schema> create_not_equal_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(not_equal_Scalar::name, not_equal_Scalar::overload_name)
      .typed<not_equal_Scalar::schema>();
}

// aten::not_equal.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor not_equal_Scalar::call(const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_not_equal_Scalar_typed_handle();
    return op.call(self, other);
}

// aten::not_equal.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor not_equal_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_not_equal_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(eq_Scalar_out, name, "aten::eq")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(eq_Scalar_out, overload_name, "Scalar_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(eq_Scalar_out, schema_str, "eq.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::eq.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<eq_Scalar_out::schema> create_eq_Scalar_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(eq_Scalar_out::name, eq_Scalar_out::overload_name)
      .typed<eq_Scalar_out::schema>();
}

// aten::eq.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & eq_Scalar_out::call(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_eq_Scalar_out_typed_handle();
    return op.call(self, other, out);
}

// aten::eq.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & eq_Scalar_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_eq_Scalar_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ge__Tensor, name, "aten::ge_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ge__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ge__Tensor, schema_str, "ge_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::ge_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<ge__Tensor::schema> create_ge__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(ge__Tensor::name, ge__Tensor::overload_name)
      .typed<ge__Tensor::schema>();
}

// aten::ge_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & ge__Tensor::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create_ge__Tensor_typed_handle();
    return op.call(self, other);
}

// aten::ge_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & ge__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create_ge__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(le_Scalar, name, "aten::le")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(le_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(le_Scalar, schema_str, "le.Scalar(Tensor self, Scalar other) -> Tensor")

// aten::le.Scalar(Tensor self, Scalar other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<le_Scalar::schema> create_le_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(le_Scalar::name, le_Scalar::overload_name)
      .typed<le_Scalar::schema>();
}

// aten::le.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor le_Scalar::call(const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_le_Scalar_typed_handle();
    return op.call(self, other);
}

// aten::le.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor le_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_le_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(le_Tensor_out, name, "aten::le")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(le_Tensor_out, overload_name, "Tensor_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(le_Tensor_out, schema_str, "le.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::le.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<le_Tensor_out::schema> create_le_Tensor_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(le_Tensor_out::name, le_Tensor_out::overload_name)
      .typed<le_Tensor_out::schema>();
}

// aten::le.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & le_Tensor_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_le_Tensor_out_typed_handle();
    return op.call(self, other, out);
}

// aten::le.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & le_Tensor_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_le_Tensor_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(less_equal_Scalar_out, name, "aten::less_equal")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(less_equal_Scalar_out, overload_name, "Scalar_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(less_equal_Scalar_out, schema_str, "less_equal.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::less_equal.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<less_equal_Scalar_out::schema> create_less_equal_Scalar_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(less_equal_Scalar_out::name, less_equal_Scalar_out::overload_name)
      .typed<less_equal_Scalar_out::schema>();
}

// aten::less_equal.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & less_equal_Scalar_out::call(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_less_equal_Scalar_out_typed_handle();
    return op.call(self, other, out);
}

// aten::less_equal.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & less_equal_Scalar_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_less_equal_Scalar_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(less_equal_Tensor_out, name, "aten::less_equal")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(less_equal_Tensor_out, overload_name, "Tensor_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(less_equal_Tensor_out, schema_str, "less_equal.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::less_equal.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<less_equal_Tensor_out::schema> create_less_equal_Tensor_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(less_equal_Tensor_out::name, less_equal_Tensor_out::overload_name)
      .typed<less_equal_Tensor_out::schema>();
}

// aten::less_equal.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & less_equal_Tensor_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_less_equal_Tensor_out_typed_handle();
    return op.call(self, other, out);
}

// aten::less_equal.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & less_equal_Tensor_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_less_equal_Tensor_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gt_Scalar, name, "aten::gt")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gt_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gt_Scalar, schema_str, "gt.Scalar(Tensor self, Scalar other) -> Tensor")

// aten::gt.Scalar(Tensor self, Scalar other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<gt_Scalar::schema> create_gt_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(gt_Scalar::name, gt_Scalar::overload_name)
      .typed<gt_Scalar::schema>();
}

// aten::gt.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor gt_Scalar::call(const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_gt_Scalar_typed_handle();
    return op.call(self, other);
}

// aten::gt.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor gt_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_gt_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gt_Tensor_out, name, "aten::gt")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gt_Tensor_out, overload_name, "Tensor_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gt_Tensor_out, schema_str, "gt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::gt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<gt_Tensor_out::schema> create_gt_Tensor_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(gt_Tensor_out::name, gt_Tensor_out::overload_name)
      .typed<gt_Tensor_out::schema>();
}

// aten::gt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & gt_Tensor_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_gt_Tensor_out_typed_handle();
    return op.call(self, other, out);
}

// aten::gt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & gt_Tensor_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_gt_Tensor_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gt__Tensor, name, "aten::gt_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gt__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gt__Tensor, schema_str, "gt_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::gt_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<gt__Tensor::schema> create_gt__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(gt__Tensor::name, gt__Tensor::overload_name)
      .typed<gt__Tensor::schema>();
}

// aten::gt_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & gt__Tensor::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create_gt__Tensor_typed_handle();
    return op.call(self, other);
}

// aten::gt_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & gt__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create_gt__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(greater_Tensor_out, name, "aten::greater")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(greater_Tensor_out, overload_name, "Tensor_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(greater_Tensor_out, schema_str, "greater.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::greater.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<greater_Tensor_out::schema> create_greater_Tensor_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(greater_Tensor_out::name, greater_Tensor_out::overload_name)
      .typed<greater_Tensor_out::schema>();
}

// aten::greater.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & greater_Tensor_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_greater_Tensor_out_typed_handle();
    return op.call(self, other, out);
}

// aten::greater.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & greater_Tensor_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_greater_Tensor_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(greater__Tensor, name, "aten::greater_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(greater__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(greater__Tensor, schema_str, "greater_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::greater_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<greater__Tensor::schema> create_greater__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(greater__Tensor::name, greater__Tensor::overload_name)
      .typed<greater__Tensor::schema>();
}

// aten::greater_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & greater__Tensor::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create_greater__Tensor_typed_handle();
    return op.call(self, other);
}

// aten::greater_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & greater__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create_greater__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(less_Tensor_out, name, "aten::less")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(less_Tensor_out, overload_name, "Tensor_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(less_Tensor_out, schema_str, "less.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::less.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<less_Tensor_out::schema> create_less_Tensor_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(less_Tensor_out::name, less_Tensor_out::overload_name)
      .typed<less_Tensor_out::schema>();
}

// aten::less.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & less_Tensor_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_less_Tensor_out_typed_handle();
    return op.call(self, other, out);
}

// aten::less.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & less_Tensor_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_less_Tensor_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(take, name, "aten::take")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(take, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(take, schema_str, "take(Tensor self, Tensor index) -> Tensor")

// aten::take(Tensor self, Tensor index) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<take::schema> create_take_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(take::name, take::overload_name)
      .typed<take::schema>();
}

// aten::take(Tensor self, Tensor index) -> Tensor
at::Tensor take::call(const at::Tensor & self, const at::Tensor & index) {
    static auto op = create_take_typed_handle();
    return op.call(self, index);
}

// aten::take(Tensor self, Tensor index) -> Tensor
at::Tensor take::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & index) {
    static auto op = create_take_typed_handle();
    return op.redispatch(dispatchKeySet, self, index);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(take_along_dim_out, name, "aten::take_along_dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(take_along_dim_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(take_along_dim_out, schema_str, "take_along_dim.out(Tensor self, Tensor indices, int? dim=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::take_along_dim.out(Tensor self, Tensor indices, int? dim=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<take_along_dim_out::schema> create_take_along_dim_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(take_along_dim_out::name, take_along_dim_out::overload_name)
      .typed<take_along_dim_out::schema>();
}

// aten::take_along_dim.out(Tensor self, Tensor indices, int? dim=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & take_along_dim_out::call(const at::Tensor & self, const at::Tensor & indices, c10::optional<int64_t> dim, at::Tensor & out) {
    static auto op = create_take_along_dim_out_typed_handle();
    return op.call(self, indices, dim, out);
}

// aten::take_along_dim.out(Tensor self, Tensor indices, int? dim=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & take_along_dim_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & indices, c10::optional<int64_t> dim, at::Tensor & out) {
    static auto op = create_take_along_dim_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, indices, dim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_select_backward, name, "aten::index_select_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_select_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_select_backward, schema_str, "index_select_backward(Tensor grad, int[] self_sizes, int dim, Tensor index) -> Tensor")

// aten::index_select_backward(Tensor grad, int[] self_sizes, int dim, Tensor index) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<index_select_backward::schema> create_index_select_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(index_select_backward::name, index_select_backward::overload_name)
      .typed<index_select_backward::schema>();
}

// aten::index_select_backward(Tensor grad, int[] self_sizes, int dim, Tensor index) -> Tensor
at::Tensor index_select_backward::call(const at::Tensor & grad, at::IntArrayRef self_sizes, int64_t dim, const at::Tensor & index) {
    static auto op = create_index_select_backward_typed_handle();
    return op.call(grad, self_sizes, dim, index);
}

// aten::index_select_backward(Tensor grad, int[] self_sizes, int dim, Tensor index) -> Tensor
at::Tensor index_select_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, at::IntArrayRef self_sizes, int64_t dim, const at::Tensor & index) {
    static auto op = create_index_select_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad, self_sizes, dim, index);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(eig, name, "aten::eig")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(eig, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(eig, schema_str, "eig(Tensor self, bool eigenvectors=False) -> (Tensor eigenvalues, Tensor eigenvectors)")

// aten::eig(Tensor self, bool eigenvectors=False) -> (Tensor eigenvalues, Tensor eigenvectors)
static C10_NOINLINE c10::TypedOperatorHandle<eig::schema> create_eig_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(eig::name, eig::overload_name)
      .typed<eig::schema>();
}

// aten::eig(Tensor self, bool eigenvectors=False) -> (Tensor eigenvalues, Tensor eigenvectors)
::std::tuple<at::Tensor,at::Tensor> eig::call(const at::Tensor & self, bool eigenvectors) {
    static auto op = create_eig_typed_handle();
    return op.call(self, eigenvectors);
}

// aten::eig(Tensor self, bool eigenvectors=False) -> (Tensor eigenvalues, Tensor eigenvectors)
::std::tuple<at::Tensor,at::Tensor> eig::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool eigenvectors) {
    static auto op = create_eig_typed_handle();
    return op.redispatch(dispatchKeySet, self, eigenvectors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(svd, name, "aten::svd")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(svd, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(svd, schema_str, "svd(Tensor self, bool some=True, bool compute_uv=True) -> (Tensor U, Tensor S, Tensor V)")

// aten::svd(Tensor self, bool some=True, bool compute_uv=True) -> (Tensor U, Tensor S, Tensor V)
static C10_NOINLINE c10::TypedOperatorHandle<svd::schema> create_svd_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(svd::name, svd::overload_name)
      .typed<svd::schema>();
}

// aten::svd(Tensor self, bool some=True, bool compute_uv=True) -> (Tensor U, Tensor S, Tensor V)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> svd::call(const at::Tensor & self, bool some, bool compute_uv) {
    static auto op = create_svd_typed_handle();
    return op.call(self, some, compute_uv);
}

// aten::svd(Tensor self, bool some=True, bool compute_uv=True) -> (Tensor U, Tensor S, Tensor V)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> svd::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool some, bool compute_uv) {
    static auto op = create_svd_typed_handle();
    return op.redispatch(dispatchKeySet, self, some, compute_uv);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cholesky_solve_out, name, "aten::cholesky_solve")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cholesky_solve_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cholesky_solve_out, schema_str, "cholesky_solve.out(Tensor self, Tensor input2, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)")

// aten::cholesky_solve.out(Tensor self, Tensor input2, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<cholesky_solve_out::schema> create_cholesky_solve_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cholesky_solve_out::name, cholesky_solve_out::overload_name)
      .typed<cholesky_solve_out::schema>();
}

// aten::cholesky_solve.out(Tensor self, Tensor input2, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & cholesky_solve_out::call(const at::Tensor & self, const at::Tensor & input2, bool upper, at::Tensor & out) {
    static auto op = create_cholesky_solve_out_typed_handle();
    return op.call(self, input2, upper, out);
}

// aten::cholesky_solve.out(Tensor self, Tensor input2, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & cholesky_solve_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & input2, bool upper, at::Tensor & out) {
    static auto op = create_cholesky_solve_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, input2, upper, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(solve_solution, name, "aten::solve")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(solve_solution, overload_name, "solution")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(solve_solution, schema_str, "solve.solution(Tensor self, Tensor A, *, Tensor(a!) solution, Tensor(b!) lu) -> (Tensor(a!) solution, Tensor(b!) LU)")

// aten::solve.solution(Tensor self, Tensor A, *, Tensor(a!) solution, Tensor(b!) lu) -> (Tensor(a!) solution, Tensor(b!) LU)
static C10_NOINLINE c10::TypedOperatorHandle<solve_solution::schema> create_solve_solution_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(solve_solution::name, solve_solution::overload_name)
      .typed<solve_solution::schema>();
}

// aten::solve.solution(Tensor self, Tensor A, *, Tensor(a!) solution, Tensor(b!) lu) -> (Tensor(a!) solution, Tensor(b!) LU)
::std::tuple<at::Tensor &,at::Tensor &> solve_solution::call(const at::Tensor & self, const at::Tensor & A, at::Tensor & solution, at::Tensor & lu) {
    static auto op = create_solve_solution_typed_handle();
    return op.call(self, A, solution, lu);
}

// aten::solve.solution(Tensor self, Tensor A, *, Tensor(a!) solution, Tensor(b!) lu) -> (Tensor(a!) solution, Tensor(b!) LU)
::std::tuple<at::Tensor &,at::Tensor &> solve_solution::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & A, at::Tensor & solution, at::Tensor & lu) {
    static auto op = create_solve_solution_typed_handle();
    return op.redispatch(dispatchKeySet, self, A, solution, lu);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(geqrf, name, "aten::geqrf")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(geqrf, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(geqrf, schema_str, "geqrf(Tensor self) -> (Tensor a, Tensor tau)")

// aten::geqrf(Tensor self) -> (Tensor a, Tensor tau)
static C10_NOINLINE c10::TypedOperatorHandle<geqrf::schema> create_geqrf_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(geqrf::name, geqrf::overload_name)
      .typed<geqrf::schema>();
}

// aten::geqrf(Tensor self) -> (Tensor a, Tensor tau)
::std::tuple<at::Tensor,at::Tensor> geqrf::call(const at::Tensor & self) {
    static auto op = create_geqrf_typed_handle();
    return op.call(self);
}

// aten::geqrf(Tensor self) -> (Tensor a, Tensor tau)
::std::tuple<at::Tensor,at::Tensor> geqrf::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_geqrf_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(orgqr, name, "aten::orgqr")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(orgqr, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(orgqr, schema_str, "orgqr(Tensor self, Tensor input2) -> Tensor")

// aten::orgqr(Tensor self, Tensor input2) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<orgqr::schema> create_orgqr_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(orgqr::name, orgqr::overload_name)
      .typed<orgqr::schema>();
}

// aten::orgqr(Tensor self, Tensor input2) -> Tensor
at::Tensor orgqr::call(const at::Tensor & self, const at::Tensor & input2) {
    static auto op = create_orgqr_typed_handle();
    return op.call(self, input2);
}

// aten::orgqr(Tensor self, Tensor input2) -> Tensor
at::Tensor orgqr::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & input2) {
    static auto op = create_orgqr_typed_handle();
    return op.redispatch(dispatchKeySet, self, input2);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(orgqr_out, name, "aten::orgqr")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(orgqr_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(orgqr_out, schema_str, "orgqr.out(Tensor self, Tensor input2, *, Tensor(a!) out) -> Tensor(a!)")

// aten::orgqr.out(Tensor self, Tensor input2, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<orgqr_out::schema> create_orgqr_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(orgqr_out::name, orgqr_out::overload_name)
      .typed<orgqr_out::schema>();
}

// aten::orgqr.out(Tensor self, Tensor input2, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & orgqr_out::call(const at::Tensor & self, const at::Tensor & input2, at::Tensor & out) {
    static auto op = create_orgqr_out_typed_handle();
    return op.call(self, input2, out);
}

// aten::orgqr.out(Tensor self, Tensor input2, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & orgqr_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & input2, at::Tensor & out) {
    static auto op = create_orgqr_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, input2, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lu_unpack_out, name, "aten::lu_unpack")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lu_unpack_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lu_unpack_out, schema_str, "lu_unpack.out(Tensor LU_data, Tensor LU_pivots, bool unpack_data=True, bool unpack_pivots=True, *, Tensor(a!) P, Tensor(b!) L, Tensor(c!) U) -> (Tensor(a!) P, Tensor(b!) L, Tensor(c!) U)")

// aten::lu_unpack.out(Tensor LU_data, Tensor LU_pivots, bool unpack_data=True, bool unpack_pivots=True, *, Tensor(a!) P, Tensor(b!) L, Tensor(c!) U) -> (Tensor(a!) P, Tensor(b!) L, Tensor(c!) U)
static C10_NOINLINE c10::TypedOperatorHandle<lu_unpack_out::schema> create_lu_unpack_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(lu_unpack_out::name, lu_unpack_out::overload_name)
      .typed<lu_unpack_out::schema>();
}

// aten::lu_unpack.out(Tensor LU_data, Tensor LU_pivots, bool unpack_data=True, bool unpack_pivots=True, *, Tensor(a!) P, Tensor(b!) L, Tensor(c!) U) -> (Tensor(a!) P, Tensor(b!) L, Tensor(c!) U)
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> lu_unpack_out::call(const at::Tensor & LU_data, const at::Tensor & LU_pivots, bool unpack_data, bool unpack_pivots, at::Tensor & P, at::Tensor & L, at::Tensor & U) {
    static auto op = create_lu_unpack_out_typed_handle();
    return op.call(LU_data, LU_pivots, unpack_data, unpack_pivots, P, L, U);
}

// aten::lu_unpack.out(Tensor LU_data, Tensor LU_pivots, bool unpack_data=True, bool unpack_pivots=True, *, Tensor(a!) P, Tensor(b!) L, Tensor(c!) U) -> (Tensor(a!) P, Tensor(b!) L, Tensor(c!) U)
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> lu_unpack_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & LU_data, const at::Tensor & LU_pivots, bool unpack_data, bool unpack_pivots, at::Tensor & P, at::Tensor & L, at::Tensor & U) {
    static auto op = create_lu_unpack_out_typed_handle();
    return op.redispatch(dispatchKeySet, LU_data, LU_pivots, unpack_data, unpack_pivots, P, L, U);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(digamma_out, name, "aten::digamma")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(digamma_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(digamma_out, schema_str, "digamma.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::digamma.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<digamma_out::schema> create_digamma_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(digamma_out::name, digamma_out::overload_name)
      .typed<digamma_out::schema>();
}

// aten::digamma.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & digamma_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_digamma_out_typed_handle();
    return op.call(self, out);
}

// aten::digamma.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & digamma_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_digamma_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(erfinv, name, "aten::erfinv")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(erfinv, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(erfinv, schema_str, "erfinv(Tensor self) -> Tensor")

// aten::erfinv(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<erfinv::schema> create_erfinv_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(erfinv::name, erfinv::overload_name)
      .typed<erfinv::schema>();
}

// aten::erfinv(Tensor self) -> Tensor
at::Tensor erfinv::call(const at::Tensor & self) {
    static auto op = create_erfinv_typed_handle();
    return op.call(self);
}

// aten::erfinv(Tensor self) -> Tensor
at::Tensor erfinv::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_erfinv_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(erfinv_, name, "aten::erfinv_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(erfinv_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(erfinv_, schema_str, "erfinv_(Tensor(a!) self) -> Tensor(a!)")

// aten::erfinv_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<erfinv_::schema> create_erfinv__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(erfinv_::name, erfinv_::overload_name)
      .typed<erfinv_::schema>();
}

// aten::erfinv_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & erfinv_::call(at::Tensor & self) {
    static auto op = create_erfinv__typed_handle();
    return op.call(self);
}

// aten::erfinv_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & erfinv_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_erfinv__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(i0_, name, "aten::i0_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(i0_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(i0_, schema_str, "i0_(Tensor(a!) self) -> Tensor(a!)")

// aten::i0_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<i0_::schema> create_i0__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(i0_::name, i0_::overload_name)
      .typed<i0_::schema>();
}

// aten::i0_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & i0_::call(at::Tensor & self) {
    static auto op = create_i0__typed_handle();
    return op.call(self);
}

// aten::i0_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & i0_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_i0__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(signbit, name, "aten::signbit")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(signbit, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(signbit, schema_str, "signbit(Tensor self) -> Tensor")

// aten::signbit(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<signbit::schema> create_signbit_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(signbit::name, signbit::overload_name)
      .typed<signbit::schema>();
}

// aten::signbit(Tensor self) -> Tensor
at::Tensor signbit::call(const at::Tensor & self) {
    static auto op = create_signbit_typed_handle();
    return op.call(self);
}

// aten::signbit(Tensor self) -> Tensor
at::Tensor signbit::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_signbit_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(dist, name, "aten::dist")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(dist, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(dist, schema_str, "dist(Tensor self, Tensor other, Scalar p=2) -> Tensor")

// aten::dist(Tensor self, Tensor other, Scalar p=2) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<dist::schema> create_dist_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(dist::name, dist::overload_name)
      .typed<dist::schema>();
}

// aten::dist(Tensor self, Tensor other, Scalar p=2) -> Tensor
at::Tensor dist::call(const at::Tensor & self, const at::Tensor & other, const at::Scalar & p) {
    static auto op = create_dist_typed_handle();
    return op.call(self, other, p);
}

// aten::dist(Tensor self, Tensor other, Scalar p=2) -> Tensor
at::Tensor dist::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, const at::Scalar & p) {
    static auto op = create_dist_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, p);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lerp_Tensor, name, "aten::lerp")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lerp_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lerp_Tensor, schema_str, "lerp.Tensor(Tensor self, Tensor end, Tensor weight) -> Tensor")

// aten::lerp.Tensor(Tensor self, Tensor end, Tensor weight) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<lerp_Tensor::schema> create_lerp_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(lerp_Tensor::name, lerp_Tensor::overload_name)
      .typed<lerp_Tensor::schema>();
}

// aten::lerp.Tensor(Tensor self, Tensor end, Tensor weight) -> Tensor
at::Tensor lerp_Tensor::call(const at::Tensor & self, const at::Tensor & end, const at::Tensor & weight) {
    static auto op = create_lerp_Tensor_typed_handle();
    return op.call(self, end, weight);
}

// aten::lerp.Tensor(Tensor self, Tensor end, Tensor weight) -> Tensor
at::Tensor lerp_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & end, const at::Tensor & weight) {
    static auto op = create_lerp_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, end, weight);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(igamma_out, name, "aten::igamma")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(igamma_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(igamma_out, schema_str, "igamma.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::igamma.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<igamma_out::schema> create_igamma_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(igamma_out::name, igamma_out::overload_name)
      .typed<igamma_out::schema>();
}

// aten::igamma.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & igamma_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_igamma_out_typed_handle();
    return op.call(self, other, out);
}

// aten::igamma.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & igamma_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_igamma_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fmax_out, name, "aten::fmax")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fmax_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fmax_out, schema_str, "fmax.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::fmax.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<fmax_out::schema> create_fmax_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fmax_out::name, fmax_out::overload_name)
      .typed<fmax_out::schema>();
}

// aten::fmax.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fmax_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_fmax_out_typed_handle();
    return op.call(self, other, out);
}

// aten::fmax.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fmax_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_fmax_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantile_scalar, name, "aten::quantile")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantile_scalar, overload_name, "scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantile_scalar, schema_str, "quantile.scalar(Tensor self, float q, int? dim=None, bool keepdim=False) -> Tensor")

// aten::quantile.scalar(Tensor self, float q, int? dim=None, bool keepdim=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<quantile_scalar::schema> create_quantile_scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(quantile_scalar::name, quantile_scalar::overload_name)
      .typed<quantile_scalar::schema>();
}

// aten::quantile.scalar(Tensor self, float q, int? dim=None, bool keepdim=False) -> Tensor
at::Tensor quantile_scalar::call(const at::Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim) {
    static auto op = create_quantile_scalar_typed_handle();
    return op.call(self, q, dim, keepdim);
}

// aten::quantile.scalar(Tensor self, float q, int? dim=None, bool keepdim=False) -> Tensor
at::Tensor quantile_scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim) {
    static auto op = create_quantile_scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, q, dim, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanquantile_scalar_out, name, "aten::nanquantile")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanquantile_scalar_out, overload_name, "scalar_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanquantile_scalar_out, schema_str, "nanquantile.scalar_out(Tensor self, float q, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)")

// aten::nanquantile.scalar_out(Tensor self, float q, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<nanquantile_scalar_out::schema> create_nanquantile_scalar_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nanquantile_scalar_out::name, nanquantile_scalar_out::overload_name)
      .typed<nanquantile_scalar_out::schema>();
}

// aten::nanquantile.scalar_out(Tensor self, float q, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & nanquantile_scalar_out::call(const at::Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out) {
    static auto op = create_nanquantile_scalar_out_typed_handle();
    return op.call(self, q, dim, keepdim, out);
}

// aten::nanquantile.scalar_out(Tensor self, float q, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & nanquantile_scalar_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out) {
    static auto op = create_nanquantile_scalar_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, q, dim, keepdim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanquantile, name, "aten::nanquantile")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanquantile, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanquantile, schema_str, "nanquantile(Tensor self, Tensor q, int? dim=None, bool keepdim=False) -> Tensor")

// aten::nanquantile(Tensor self, Tensor q, int? dim=None, bool keepdim=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<nanquantile::schema> create_nanquantile_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nanquantile::name, nanquantile::overload_name)
      .typed<nanquantile::schema>();
}

// aten::nanquantile(Tensor self, Tensor q, int? dim=None, bool keepdim=False) -> Tensor
at::Tensor nanquantile::call(const at::Tensor & self, const at::Tensor & q, c10::optional<int64_t> dim, bool keepdim) {
    static auto op = create_nanquantile_typed_handle();
    return op.call(self, q, dim, keepdim);
}

// aten::nanquantile(Tensor self, Tensor q, int? dim=None, bool keepdim=False) -> Tensor
at::Tensor nanquantile::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & q, c10::optional<int64_t> dim, bool keepdim) {
    static auto op = create_nanquantile_typed_handle();
    return op.redispatch(dispatchKeySet, self, q, dim, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanquantile_new_scalar, name, "aten::nanquantile")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanquantile_new_scalar, overload_name, "new_scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanquantile_new_scalar, schema_str, "nanquantile.new_scalar(Tensor self, float q, int? dim, bool keepdim, *, str interpolation) -> Tensor")

// aten::nanquantile.new_scalar(Tensor self, float q, int? dim, bool keepdim, *, str interpolation) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<nanquantile_new_scalar::schema> create_nanquantile_new_scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nanquantile_new_scalar::name, nanquantile_new_scalar::overload_name)
      .typed<nanquantile_new_scalar::schema>();
}

// aten::nanquantile.new_scalar(Tensor self, float q, int? dim, bool keepdim, *, str interpolation) -> Tensor
at::Tensor nanquantile_new_scalar::call(const at::Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) {
    static auto op = create_nanquantile_new_scalar_typed_handle();
    return op.call(self, q, dim, keepdim, interpolation);
}

// aten::nanquantile.new_scalar(Tensor self, float q, int? dim, bool keepdim, *, str interpolation) -> Tensor
at::Tensor nanquantile_new_scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) {
    static auto op = create_nanquantile_new_scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, q, dim, keepdim, interpolation);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(any, name, "aten::any")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(any, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(any, schema_str, "any(Tensor self) -> Tensor")

// aten::any(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<any::schema> create_any_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(any::name, any::overload_name)
      .typed<any::schema>();
}

// aten::any(Tensor self) -> Tensor
at::Tensor any::call(const at::Tensor & self) {
    static auto op = create_any_typed_handle();
    return op.call(self);
}

// aten::any(Tensor self) -> Tensor
at::Tensor any::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_any_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(renorm, name, "aten::renorm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(renorm, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(renorm, schema_str, "renorm(Tensor self, Scalar p, int dim, Scalar maxnorm) -> Tensor")

// aten::renorm(Tensor self, Scalar p, int dim, Scalar maxnorm) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<renorm::schema> create_renorm_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(renorm::name, renorm::overload_name)
      .typed<renorm::schema>();
}

// aten::renorm(Tensor self, Scalar p, int dim, Scalar maxnorm) -> Tensor
at::Tensor renorm::call(const at::Tensor & self, const at::Scalar & p, int64_t dim, const at::Scalar & maxnorm) {
    static auto op = create_renorm_typed_handle();
    return op.call(self, p, dim, maxnorm);
}

// aten::renorm(Tensor self, Scalar p, int dim, Scalar maxnorm) -> Tensor
at::Tensor renorm::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & p, int64_t dim, const at::Scalar & maxnorm) {
    static auto op = create_renorm_typed_handle();
    return op.redispatch(dispatchKeySet, self, p, dim, maxnorm);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(renorm_, name, "aten::renorm_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(renorm_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(renorm_, schema_str, "renorm_(Tensor(a!) self, Scalar p, int dim, Scalar maxnorm) -> Tensor(a!)")

// aten::renorm_(Tensor(a!) self, Scalar p, int dim, Scalar maxnorm) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<renorm_::schema> create_renorm__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(renorm_::name, renorm_::overload_name)
      .typed<renorm_::schema>();
}

// aten::renorm_(Tensor(a!) self, Scalar p, int dim, Scalar maxnorm) -> Tensor(a!)
at::Tensor & renorm_::call(at::Tensor & self, const at::Scalar & p, int64_t dim, const at::Scalar & maxnorm) {
    static auto op = create_renorm__typed_handle();
    return op.call(self, p, dim, maxnorm);
}

// aten::renorm_(Tensor(a!) self, Scalar p, int dim, Scalar maxnorm) -> Tensor(a!)
at::Tensor & renorm_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & p, int64_t dim, const at::Scalar & maxnorm) {
    static auto op = create_renorm__typed_handle();
    return op.redispatch(dispatchKeySet, self, p, dim, maxnorm);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unfold, name, "aten::unfold")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unfold, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unfold, schema_str, "unfold(Tensor(a) self, int dimension, int size, int step) -> Tensor(a)")

// aten::unfold(Tensor(a) self, int dimension, int size, int step) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<unfold::schema> create_unfold_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(unfold::name, unfold::overload_name)
      .typed<unfold::schema>();
}

// aten::unfold(Tensor(a) self, int dimension, int size, int step) -> Tensor(a)
at::Tensor unfold::call(const at::Tensor & self, int64_t dimension, int64_t size, int64_t step) {
    static auto op = create_unfold_typed_handle();
    return op.call(self, dimension, size, step);
}

// aten::unfold(Tensor(a) self, int dimension, int size, int step) -> Tensor(a)
at::Tensor unfold::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dimension, int64_t size, int64_t step) {
    static auto op = create_unfold_typed_handle();
    return op.redispatch(dispatchKeySet, self, dimension, size, step);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pow__Tensor, name, "aten::pow_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pow__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pow__Tensor, schema_str, "pow_.Tensor(Tensor(a!) self, Tensor exponent) -> Tensor(a!)")

// aten::pow_.Tensor(Tensor(a!) self, Tensor exponent) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<pow__Tensor::schema> create_pow__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(pow__Tensor::name, pow__Tensor::overload_name)
      .typed<pow__Tensor::schema>();
}

// aten::pow_.Tensor(Tensor(a!) self, Tensor exponent) -> Tensor(a!)
at::Tensor & pow__Tensor::call(at::Tensor & self, const at::Tensor & exponent) {
    static auto op = create_pow__Tensor_typed_handle();
    return op.call(self, exponent);
}

// aten::pow_.Tensor(Tensor(a!) self, Tensor exponent) -> Tensor(a!)
at::Tensor & pow__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & exponent) {
    static auto op = create_pow__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, exponent);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(float_power_Scalar, name, "aten::float_power")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(float_power_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(float_power_Scalar, schema_str, "float_power.Scalar(Scalar self, Tensor exponent) -> Tensor")

// aten::float_power.Scalar(Scalar self, Tensor exponent) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<float_power_Scalar::schema> create_float_power_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(float_power_Scalar::name, float_power_Scalar::overload_name)
      .typed<float_power_Scalar::schema>();
}

// aten::float_power.Scalar(Scalar self, Tensor exponent) -> Tensor
at::Tensor float_power_Scalar::call(const at::Scalar & self, const at::Tensor & exponent) {
    static auto op = create_float_power_Scalar_typed_handle();
    return op.call(self, exponent);
}

// aten::float_power.Scalar(Scalar self, Tensor exponent) -> Tensor
at::Tensor float_power_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Scalar & self, const at::Tensor & exponent) {
    static auto op = create_float_power_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, exponent);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(float_power_Tensor_Scalar, name, "aten::float_power")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(float_power_Tensor_Scalar, overload_name, "Tensor_Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(float_power_Tensor_Scalar, schema_str, "float_power.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor")

// aten::float_power.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<float_power_Tensor_Scalar::schema> create_float_power_Tensor_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(float_power_Tensor_Scalar::name, float_power_Tensor_Scalar::overload_name)
      .typed<float_power_Tensor_Scalar::schema>();
}

// aten::float_power.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor
at::Tensor float_power_Tensor_Scalar::call(const at::Tensor & self, const at::Scalar & exponent) {
    static auto op = create_float_power_Tensor_Scalar_typed_handle();
    return op.call(self, exponent);
}

// aten::float_power.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor
at::Tensor float_power_Tensor_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & exponent) {
    static auto op = create_float_power_Tensor_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, exponent);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(normal_Tensor_float_out, name, "aten::normal")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(normal_Tensor_float_out, overload_name, "Tensor_float_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(normal_Tensor_float_out, schema_str, "normal.Tensor_float_out(Tensor mean, float std=1, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)")

// aten::normal.Tensor_float_out(Tensor mean, float std=1, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<normal_Tensor_float_out::schema> create_normal_Tensor_float_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(normal_Tensor_float_out::name, normal_Tensor_float_out::overload_name)
      .typed<normal_Tensor_float_out::schema>();
}

// aten::normal.Tensor_float_out(Tensor mean, float std=1, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & normal_Tensor_float_out::call(const at::Tensor & mean, double std, c10::optional<at::Generator> generator, at::Tensor & out) {
    static auto op = create_normal_Tensor_float_out_typed_handle();
    return op.call(mean, std, generator, out);
}

// aten::normal.Tensor_float_out(Tensor mean, float std=1, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & normal_Tensor_float_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & mean, double std, c10::optional<at::Generator> generator, at::Tensor & out) {
    static auto op = create_normal_Tensor_float_out_typed_handle();
    return op.redispatch(dispatchKeySet, mean, std, generator, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(normal_Tensor_float, name, "aten::normal")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(normal_Tensor_float, overload_name, "Tensor_float")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(normal_Tensor_float, schema_str, "normal.Tensor_float(Tensor mean, float std=1, *, Generator? generator=None) -> Tensor")

// aten::normal.Tensor_float(Tensor mean, float std=1, *, Generator? generator=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<normal_Tensor_float::schema> create_normal_Tensor_float_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(normal_Tensor_float::name, normal_Tensor_float::overload_name)
      .typed<normal_Tensor_float::schema>();
}

// aten::normal.Tensor_float(Tensor mean, float std=1, *, Generator? generator=None) -> Tensor
at::Tensor normal_Tensor_float::call(const at::Tensor & mean, double std, c10::optional<at::Generator> generator) {
    static auto op = create_normal_Tensor_float_typed_handle();
    return op.call(mean, std, generator);
}

// aten::normal.Tensor_float(Tensor mean, float std=1, *, Generator? generator=None) -> Tensor
at::Tensor normal_Tensor_float::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & mean, double std, c10::optional<at::Generator> generator) {
    static auto op = create_normal_Tensor_float_typed_handle();
    return op.redispatch(dispatchKeySet, mean, std, generator);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(normal_float_Tensor_out, name, "aten::normal")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(normal_float_Tensor_out, overload_name, "float_Tensor_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(normal_float_Tensor_out, schema_str, "normal.float_Tensor_out(float mean, Tensor std, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)")

// aten::normal.float_Tensor_out(float mean, Tensor std, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<normal_float_Tensor_out::schema> create_normal_float_Tensor_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(normal_float_Tensor_out::name, normal_float_Tensor_out::overload_name)
      .typed<normal_float_Tensor_out::schema>();
}

// aten::normal.float_Tensor_out(float mean, Tensor std, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & normal_float_Tensor_out::call(double mean, const at::Tensor & std, c10::optional<at::Generator> generator, at::Tensor & out) {
    static auto op = create_normal_float_Tensor_out_typed_handle();
    return op.call(mean, std, generator, out);
}

// aten::normal.float_Tensor_out(float mean, Tensor std, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & normal_float_Tensor_out::redispatch(c10::DispatchKeySet dispatchKeySet, double mean, const at::Tensor & std, c10::optional<at::Generator> generator, at::Tensor & out) {
    static auto op = create_normal_float_Tensor_out_typed_handle();
    return op.redispatch(dispatchKeySet, mean, std, generator, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(normal_Tensor_Tensor_out, name, "aten::normal")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(normal_Tensor_Tensor_out, overload_name, "Tensor_Tensor_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(normal_Tensor_Tensor_out, schema_str, "normal.Tensor_Tensor_out(Tensor mean, Tensor std, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)")

// aten::normal.Tensor_Tensor_out(Tensor mean, Tensor std, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<normal_Tensor_Tensor_out::schema> create_normal_Tensor_Tensor_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(normal_Tensor_Tensor_out::name, normal_Tensor_Tensor_out::overload_name)
      .typed<normal_Tensor_Tensor_out::schema>();
}

// aten::normal.Tensor_Tensor_out(Tensor mean, Tensor std, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & normal_Tensor_Tensor_out::call(const at::Tensor & mean, const at::Tensor & std, c10::optional<at::Generator> generator, at::Tensor & out) {
    static auto op = create_normal_Tensor_Tensor_out_typed_handle();
    return op.call(mean, std, generator, out);
}

// aten::normal.Tensor_Tensor_out(Tensor mean, Tensor std, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & normal_Tensor_Tensor_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & mean, const at::Tensor & std, c10::optional<at::Generator> generator, at::Tensor & out) {
    static auto op = create_normal_Tensor_Tensor_out_typed_handle();
    return op.redispatch(dispatchKeySet, mean, std, generator, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cat_out, name, "aten::_cat")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cat_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cat_out, schema_str, "_cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)")

// aten::_cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<_cat_out::schema> create__cat_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_cat_out::name, _cat_out::overload_name)
      .typed<_cat_out::schema>();
}

// aten::_cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & _cat_out::call(at::TensorList tensors, int64_t dim, at::Tensor & out) {
    static auto op = create__cat_out_typed_handle();
    return op.call(tensors, dim, out);
}

// aten::_cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & _cat_out::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, int64_t dim, at::Tensor & out) {
    static auto op = create__cat_out_typed_handle();
    return op.redispatch(dispatchKeySet, tensors, dim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_mul__Scalar, name, "aten::_foreach_mul_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_mul__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_mul__Scalar, schema_str, "_foreach_mul_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()")

// aten::_foreach_mul_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_mul__Scalar::schema> create__foreach_mul__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_mul__Scalar::name, _foreach_mul__Scalar::overload_name)
      .typed<_foreach_mul__Scalar::schema>();
}

// aten::_foreach_mul_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()
void _foreach_mul__Scalar::call(at::TensorList self, const at::Scalar & scalar) {
    static auto op = create__foreach_mul__Scalar_typed_handle();
    return op.call(self, scalar);
}

// aten::_foreach_mul_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()
void _foreach_mul__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self, const at::Scalar & scalar) {
    static auto op = create__foreach_mul__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, scalar);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sub__List, name, "aten::_foreach_sub_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sub__List, overload_name, "List")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sub__List, schema_str, "_foreach_sub_.List(Tensor(a!)[] self, Tensor[] other, *, Scalar alpha=1) -> ()")

// aten::_foreach_sub_.List(Tensor(a!)[] self, Tensor[] other, *, Scalar alpha=1) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_sub__List::schema> create__foreach_sub__List_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_sub__List::name, _foreach_sub__List::overload_name)
      .typed<_foreach_sub__List::schema>();
}

// aten::_foreach_sub_.List(Tensor(a!)[] self, Tensor[] other, *, Scalar alpha=1) -> ()
void _foreach_sub__List::call(at::TensorList self, at::TensorList other, const at::Scalar & alpha) {
    static auto op = create__foreach_sub__List_typed_handle();
    return op.call(self, other, alpha);
}

// aten::_foreach_sub_.List(Tensor(a!)[] self, Tensor[] other, *, Scalar alpha=1) -> ()
void _foreach_sub__List::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self, at::TensorList other, const at::Scalar & alpha) {
    static auto op = create__foreach_sub__List_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sub__ScalarList, name, "aten::_foreach_sub_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sub__ScalarList, overload_name, "ScalarList")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sub__ScalarList, schema_str, "_foreach_sub_.ScalarList(Tensor(a!)[] self, Scalar[] scalars) -> ()")

// aten::_foreach_sub_.ScalarList(Tensor(a!)[] self, Scalar[] scalars) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_sub__ScalarList::schema> create__foreach_sub__ScalarList_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_sub__ScalarList::name, _foreach_sub__ScalarList::overload_name)
      .typed<_foreach_sub__ScalarList::schema>();
}

// aten::_foreach_sub_.ScalarList(Tensor(a!)[] self, Scalar[] scalars) -> ()
void _foreach_sub__ScalarList::call(at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
    static auto op = create__foreach_sub__ScalarList_typed_handle();
    return op.call(self, scalars);
}

// aten::_foreach_sub_.ScalarList(Tensor(a!)[] self, Scalar[] scalars) -> ()
void _foreach_sub__ScalarList::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
    static auto op = create__foreach_sub__ScalarList_typed_handle();
    return op.redispatch(dispatchKeySet, self, scalars);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_exp_, name, "aten::_foreach_exp_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_exp_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_exp_, schema_str, "_foreach_exp_(Tensor(a!)[] self) -> ()")

// aten::_foreach_exp_(Tensor(a!)[] self) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_exp_::schema> create__foreach_exp__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_exp_::name, _foreach_exp_::overload_name)
      .typed<_foreach_exp_::schema>();
}

// aten::_foreach_exp_(Tensor(a!)[] self) -> ()
void _foreach_exp_::call(at::TensorList self) {
    static auto op = create__foreach_exp__typed_handle();
    return op.call(self);
}

// aten::_foreach_exp_(Tensor(a!)[] self) -> ()
void _foreach_exp_::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self) {
    static auto op = create__foreach_exp__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sqrt_, name, "aten::_foreach_sqrt_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sqrt_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sqrt_, schema_str, "_foreach_sqrt_(Tensor(a!)[] self) -> ()")

// aten::_foreach_sqrt_(Tensor(a!)[] self) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_sqrt_::schema> create__foreach_sqrt__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_sqrt_::name, _foreach_sqrt_::overload_name)
      .typed<_foreach_sqrt_::schema>();
}

// aten::_foreach_sqrt_(Tensor(a!)[] self) -> ()
void _foreach_sqrt_::call(at::TensorList self) {
    static auto op = create__foreach_sqrt__typed_handle();
    return op.call(self);
}

// aten::_foreach_sqrt_(Tensor(a!)[] self) -> ()
void _foreach_sqrt_::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self) {
    static auto op = create__foreach_sqrt__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_abs, name, "aten::_foreach_abs")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_abs, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_abs, schema_str, "_foreach_abs(Tensor[] tensors) -> Tensor[]")

// aten::_foreach_abs(Tensor[] tensors) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_abs::schema> create__foreach_abs_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_abs::name, _foreach_abs::overload_name)
      .typed<_foreach_abs::schema>();
}

// aten::_foreach_abs(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_abs::call(at::TensorList tensors) {
    static auto op = create__foreach_abs_typed_handle();
    return op.call(tensors);
}

// aten::_foreach_abs(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_abs::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create__foreach_abs_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_atan_, name, "aten::_foreach_atan_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_atan_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_atan_, schema_str, "_foreach_atan_(Tensor(a!)[] self) -> ()")

// aten::_foreach_atan_(Tensor(a!)[] self) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_atan_::schema> create__foreach_atan__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_atan_::name, _foreach_atan_::overload_name)
      .typed<_foreach_atan_::schema>();
}

// aten::_foreach_atan_(Tensor(a!)[] self) -> ()
void _foreach_atan_::call(at::TensorList self) {
    static auto op = create__foreach_atan__typed_handle();
    return op.call(self);
}

// aten::_foreach_atan_(Tensor(a!)[] self) -> ()
void _foreach_atan_::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self) {
    static auto op = create__foreach_atan__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_erf_, name, "aten::_foreach_erf_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_erf_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_erf_, schema_str, "_foreach_erf_(Tensor(a!)[] self) -> ()")

// aten::_foreach_erf_(Tensor(a!)[] self) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_erf_::schema> create__foreach_erf__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_erf_::name, _foreach_erf_::overload_name)
      .typed<_foreach_erf_::schema>();
}

// aten::_foreach_erf_(Tensor(a!)[] self) -> ()
void _foreach_erf_::call(at::TensorList self) {
    static auto op = create__foreach_erf__typed_handle();
    return op.call(self);
}

// aten::_foreach_erf_(Tensor(a!)[] self) -> ()
void _foreach_erf_::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self) {
    static auto op = create__foreach_erf__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_erfc_, name, "aten::_foreach_erfc_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_erfc_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_erfc_, schema_str, "_foreach_erfc_(Tensor(a!)[] self) -> ()")

// aten::_foreach_erfc_(Tensor(a!)[] self) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_erfc_::schema> create__foreach_erfc__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_erfc_::name, _foreach_erfc_::overload_name)
      .typed<_foreach_erfc_::schema>();
}

// aten::_foreach_erfc_(Tensor(a!)[] self) -> ()
void _foreach_erfc_::call(at::TensorList self) {
    static auto op = create__foreach_erfc__typed_handle();
    return op.call(self);
}

// aten::_foreach_erfc_(Tensor(a!)[] self) -> ()
void _foreach_erfc_::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self) {
    static auto op = create__foreach_erfc__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_expm1, name, "aten::_foreach_expm1")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_expm1, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_expm1, schema_str, "_foreach_expm1(Tensor[] tensors) -> Tensor[]")

// aten::_foreach_expm1(Tensor[] tensors) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_expm1::schema> create__foreach_expm1_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_expm1::name, _foreach_expm1::overload_name)
      .typed<_foreach_expm1::schema>();
}

// aten::_foreach_expm1(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_expm1::call(at::TensorList tensors) {
    static auto op = create__foreach_expm1_typed_handle();
    return op.call(tensors);
}

// aten::_foreach_expm1(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_expm1::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create__foreach_expm1_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_log10, name, "aten::_foreach_log10")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_log10, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_log10, schema_str, "_foreach_log10(Tensor[] tensors) -> Tensor[]")

// aten::_foreach_log10(Tensor[] tensors) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_log10::schema> create__foreach_log10_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_log10::name, _foreach_log10::overload_name)
      .typed<_foreach_log10::schema>();
}

// aten::_foreach_log10(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_log10::call(at::TensorList tensors) {
    static auto op = create__foreach_log10_typed_handle();
    return op.call(tensors);
}

// aten::_foreach_log10(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_log10::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create__foreach_log10_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_tan, name, "aten::_foreach_tan")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_tan, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_tan, schema_str, "_foreach_tan(Tensor[] tensors) -> Tensor[]")

// aten::_foreach_tan(Tensor[] tensors) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_tan::schema> create__foreach_tan_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_tan::name, _foreach_tan::overload_name)
      .typed<_foreach_tan::schema>();
}

// aten::_foreach_tan(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_tan::call(at::TensorList tensors) {
    static auto op = create__foreach_tan_typed_handle();
    return op.call(tensors);
}

// aten::_foreach_tan(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_tan::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create__foreach_tan_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sinh, name, "aten::_foreach_sinh")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sinh, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sinh, schema_str, "_foreach_sinh(Tensor[] tensors) -> Tensor[]")

// aten::_foreach_sinh(Tensor[] tensors) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_sinh::schema> create__foreach_sinh_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_sinh::name, _foreach_sinh::overload_name)
      .typed<_foreach_sinh::schema>();
}

// aten::_foreach_sinh(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_sinh::call(at::TensorList tensors) {
    static auto op = create__foreach_sinh_typed_handle();
    return op.call(tensors);
}

// aten::_foreach_sinh(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_sinh::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create__foreach_sinh_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sigmoid_, name, "aten::_foreach_sigmoid_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sigmoid_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sigmoid_, schema_str, "_foreach_sigmoid_(Tensor(a!)[] self) -> ()")

// aten::_foreach_sigmoid_(Tensor(a!)[] self) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_sigmoid_::schema> create__foreach_sigmoid__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_sigmoid_::name, _foreach_sigmoid_::overload_name)
      .typed<_foreach_sigmoid_::schema>();
}

// aten::_foreach_sigmoid_(Tensor(a!)[] self) -> ()
void _foreach_sigmoid_::call(at::TensorList self) {
    static auto op = create__foreach_sigmoid__typed_handle();
    return op.call(self);
}

// aten::_foreach_sigmoid_(Tensor(a!)[] self) -> ()
void _foreach_sigmoid_::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self) {
    static auto op = create__foreach_sigmoid__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_addcdiv_ScalarList, name, "aten::_foreach_addcdiv")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_addcdiv_ScalarList, overload_name, "ScalarList")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_addcdiv_ScalarList, schema_str, "_foreach_addcdiv.ScalarList(Tensor[] input, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars) -> Tensor[]")

// aten::_foreach_addcdiv.ScalarList(Tensor[] input, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_addcdiv_ScalarList::schema> create__foreach_addcdiv_ScalarList_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_addcdiv_ScalarList::name, _foreach_addcdiv_ScalarList::overload_name)
      .typed<_foreach_addcdiv_ScalarList::schema>();
}

// aten::_foreach_addcdiv.ScalarList(Tensor[] input, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars) -> Tensor[]
::std::vector<at::Tensor> _foreach_addcdiv_ScalarList::call(at::TensorList input, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars) {
    static auto op = create__foreach_addcdiv_ScalarList_typed_handle();
    return op.call(input, tensor1, tensor2, scalars);
}

// aten::_foreach_addcdiv.ScalarList(Tensor[] input, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars) -> Tensor[]
::std::vector<at::Tensor> _foreach_addcdiv_ScalarList::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList input, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars) {
    static auto op = create__foreach_addcdiv_ScalarList_typed_handle();
    return op.redispatch(dispatchKeySet, input, tensor1, tensor2, scalars);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(searchsorted_Tensor, name, "aten::searchsorted")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(searchsorted_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(searchsorted_Tensor, schema_str, "searchsorted.Tensor(Tensor sorted_sequence, Tensor self, *, bool out_int32=False, bool right=False) -> Tensor")

// aten::searchsorted.Tensor(Tensor sorted_sequence, Tensor self, *, bool out_int32=False, bool right=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<searchsorted_Tensor::schema> create_searchsorted_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(searchsorted_Tensor::name, searchsorted_Tensor::overload_name)
      .typed<searchsorted_Tensor::schema>();
}

// aten::searchsorted.Tensor(Tensor sorted_sequence, Tensor self, *, bool out_int32=False, bool right=False) -> Tensor
at::Tensor searchsorted_Tensor::call(const at::Tensor & sorted_sequence, const at::Tensor & self, bool out_int32, bool right) {
    static auto op = create_searchsorted_Tensor_typed_handle();
    return op.call(sorted_sequence, self, out_int32, right);
}

// aten::searchsorted.Tensor(Tensor sorted_sequence, Tensor self, *, bool out_int32=False, bool right=False) -> Tensor
at::Tensor searchsorted_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & sorted_sequence, const at::Tensor & self, bool out_int32, bool right) {
    static auto op = create_searchsorted_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, sorted_sequence, self, out_int32, right);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(searchsorted_Scalar, name, "aten::searchsorted")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(searchsorted_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(searchsorted_Scalar, schema_str, "searchsorted.Scalar(Tensor sorted_sequence, Scalar self, *, bool out_int32=False, bool right=False) -> Tensor")

// aten::searchsorted.Scalar(Tensor sorted_sequence, Scalar self, *, bool out_int32=False, bool right=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<searchsorted_Scalar::schema> create_searchsorted_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(searchsorted_Scalar::name, searchsorted_Scalar::overload_name)
      .typed<searchsorted_Scalar::schema>();
}

// aten::searchsorted.Scalar(Tensor sorted_sequence, Scalar self, *, bool out_int32=False, bool right=False) -> Tensor
at::Tensor searchsorted_Scalar::call(const at::Tensor & sorted_sequence, const at::Scalar & self, bool out_int32, bool right) {
    static auto op = create_searchsorted_Scalar_typed_handle();
    return op.call(sorted_sequence, self, out_int32, right);
}

// aten::searchsorted.Scalar(Tensor sorted_sequence, Scalar self, *, bool out_int32=False, bool right=False) -> Tensor
at::Tensor searchsorted_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & sorted_sequence, const at::Scalar & self, bool out_int32, bool right) {
    static auto op = create_searchsorted_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, sorted_sequence, self, out_int32, right);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mse_loss_backward_grad_input, name, "aten::mse_loss_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mse_loss_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mse_loss_backward_grad_input, schema_str, "mse_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::mse_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<mse_loss_backward_grad_input::schema> create_mse_loss_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mse_loss_backward_grad_input::name, mse_loss_backward_grad_input::overload_name)
      .typed<mse_loss_backward_grad_input::schema>();
}

// aten::mse_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & mse_loss_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & grad_input) {
    static auto op = create_mse_loss_backward_grad_input_typed_handle();
    return op.call(grad_output, self, target, reduction, grad_input);
}

// aten::mse_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & mse_loss_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & grad_input) {
    static auto op = create_mse_loss_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, target, reduction, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multi_margin_loss_backward_grad_input, name, "aten::multi_margin_loss_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multi_margin_loss_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multi_margin_loss_backward_grad_input, schema_str, "multi_margin_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Scalar p, Scalar margin, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::multi_margin_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Scalar p, Scalar margin, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<multi_margin_loss_backward_grad_input::schema> create_multi_margin_loss_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(multi_margin_loss_backward_grad_input::name, multi_margin_loss_backward_grad_input::overload_name)
      .typed<multi_margin_loss_backward_grad_input::schema>();
}

// aten::multi_margin_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Scalar p, Scalar margin, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & multi_margin_loss_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const at::Scalar & p, const at::Scalar & margin, const c10::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & grad_input) {
    static auto op = create_multi_margin_loss_backward_grad_input_typed_handle();
    return op.call(grad_output, self, target, p, margin, weight, reduction, grad_input);
}

// aten::multi_margin_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Scalar p, Scalar margin, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & multi_margin_loss_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const at::Scalar & p, const at::Scalar & margin, const c10::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & grad_input) {
    static auto op = create_multi_margin_loss_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, target, p, margin, weight, reduction, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss_backward_grad_input, name, "aten::nll_loss_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss_backward_grad_input, schema_str, "nll_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::nll_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<nll_loss_backward_grad_input::schema> create_nll_loss_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nll_loss_backward_grad_input::name, nll_loss_backward_grad_input::overload_name)
      .typed<nll_loss_backward_grad_input::schema>();
}

// aten::nll_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & nll_loss_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight, at::Tensor & grad_input) {
    static auto op = create_nll_loss_backward_grad_input_typed_handle();
    return op.call(grad_output, self, target, weight, reduction, ignore_index, total_weight, grad_input);
}

// aten::nll_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & nll_loss_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight, at::Tensor & grad_input) {
    static auto op = create_nll_loss_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, target, weight, reduction, ignore_index, total_weight, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(smooth_l1_loss, name, "aten::smooth_l1_loss")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(smooth_l1_loss, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(smooth_l1_loss, schema_str, "smooth_l1_loss(Tensor self, Tensor target, int reduction=Mean, float beta=1.0) -> Tensor")

// aten::smooth_l1_loss(Tensor self, Tensor target, int reduction=Mean, float beta=1.0) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<smooth_l1_loss::schema> create_smooth_l1_loss_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(smooth_l1_loss::name, smooth_l1_loss::overload_name)
      .typed<smooth_l1_loss::schema>();
}

// aten::smooth_l1_loss(Tensor self, Tensor target, int reduction=Mean, float beta=1.0) -> Tensor
at::Tensor smooth_l1_loss::call(const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta) {
    static auto op = create_smooth_l1_loss_typed_handle();
    return op.call(self, target, reduction, beta);
}

// aten::smooth_l1_loss(Tensor self, Tensor target, int reduction=Mean, float beta=1.0) -> Tensor
at::Tensor smooth_l1_loss::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta) {
    static auto op = create_smooth_l1_loss_typed_handle();
    return op.redispatch(dispatchKeySet, self, target, reduction, beta);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(huber_loss_out, name, "aten::huber_loss")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(huber_loss_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(huber_loss_out, schema_str, "huber_loss.out(Tensor self, Tensor target, int reduction=Mean, float delta=1.0, *, Tensor(a!) out) -> Tensor(a!)")

// aten::huber_loss.out(Tensor self, Tensor target, int reduction=Mean, float delta=1.0, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<huber_loss_out::schema> create_huber_loss_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(huber_loss_out::name, huber_loss_out::overload_name)
      .typed<huber_loss_out::schema>();
}

// aten::huber_loss.out(Tensor self, Tensor target, int reduction=Mean, float delta=1.0, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & huber_loss_out::call(const at::Tensor & self, const at::Tensor & target, int64_t reduction, double delta, at::Tensor & out) {
    static auto op = create_huber_loss_out_typed_handle();
    return op.call(self, target, reduction, delta, out);
}

// aten::huber_loss.out(Tensor self, Tensor target, int reduction=Mean, float delta=1.0, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & huber_loss_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double delta, at::Tensor & out) {
    static auto op = create_huber_loss_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, target, reduction, delta, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(huber_loss_backward_out, name, "aten::huber_loss_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(huber_loss_backward_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(huber_loss_backward_out, schema_str, "huber_loss_backward.out(Tensor grad_output, Tensor self, Tensor target, int reduction, float delta, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::huber_loss_backward.out(Tensor grad_output, Tensor self, Tensor target, int reduction, float delta, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<huber_loss_backward_out::schema> create_huber_loss_backward_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(huber_loss_backward_out::name, huber_loss_backward_out::overload_name)
      .typed<huber_loss_backward_out::schema>();
}

// aten::huber_loss_backward.out(Tensor grad_output, Tensor self, Tensor target, int reduction, float delta, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & huber_loss_backward_out::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double delta, at::Tensor & grad_input) {
    static auto op = create_huber_loss_backward_out_typed_handle();
    return op.call(grad_output, self, target, reduction, delta, grad_input);
}

// aten::huber_loss_backward.out(Tensor grad_output, Tensor self, Tensor target, int reduction, float delta, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & huber_loss_backward_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double delta, at::Tensor & grad_input) {
    static auto op = create_huber_loss_backward_out_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, target, reduction, delta, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(elu, name, "aten::elu")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(elu, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(elu, schema_str, "elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor")

// aten::elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<elu::schema> create_elu_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(elu::name, elu::overload_name)
      .typed<elu::schema>();
}

// aten::elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor
at::Tensor elu::call(const at::Tensor & self, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale) {
    static auto op = create_elu_typed_handle();
    return op.call(self, alpha, scale, input_scale);
}

// aten::elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor
at::Tensor elu::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale) {
    static auto op = create_elu_typed_handle();
    return op.redispatch(dispatchKeySet, self, alpha, scale, input_scale);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(elu_, name, "aten::elu_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(elu_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(elu_, schema_str, "elu_(Tensor(a!) self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor(a!)")

// aten::elu_(Tensor(a!) self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<elu_::schema> create_elu__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(elu_::name, elu_::overload_name)
      .typed<elu_::schema>();
}

// aten::elu_(Tensor(a!) self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor(a!)
at::Tensor & elu_::call(at::Tensor & self, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale) {
    static auto op = create_elu__typed_handle();
    return op.call(self, alpha, scale, input_scale);
}

// aten::elu_(Tensor(a!) self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor(a!)
at::Tensor & elu_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale) {
    static auto op = create_elu__typed_handle();
    return op.redispatch(dispatchKeySet, self, alpha, scale, input_scale);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(glu_backward, name, "aten::glu_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(glu_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(glu_backward, schema_str, "glu_backward(Tensor grad_output, Tensor self, int dim) -> Tensor")

// aten::glu_backward(Tensor grad_output, Tensor self, int dim) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<glu_backward::schema> create_glu_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(glu_backward::name, glu_backward::overload_name)
      .typed<glu_backward::schema>();
}

// aten::glu_backward(Tensor grad_output, Tensor self, int dim) -> Tensor
at::Tensor glu_backward::call(const at::Tensor & grad_output, const at::Tensor & self, int64_t dim) {
    static auto op = create_glu_backward_typed_handle();
    return op.call(grad_output, self, dim);
}

// aten::glu_backward(Tensor grad_output, Tensor self, int dim) -> Tensor
at::Tensor glu_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, int64_t dim) {
    static auto op = create_glu_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardtanh_backward, name, "aten::hardtanh_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardtanh_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardtanh_backward, schema_str, "hardtanh_backward(Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val) -> Tensor")

// aten::hardtanh_backward(Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<hardtanh_backward::schema> create_hardtanh_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(hardtanh_backward::name, hardtanh_backward::overload_name)
      .typed<hardtanh_backward::schema>();
}

// aten::hardtanh_backward(Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val) -> Tensor
at::Tensor hardtanh_backward::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val) {
    static auto op = create_hardtanh_backward_typed_handle();
    return op.call(grad_output, self, min_val, max_val);
}

// aten::hardtanh_backward(Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val) -> Tensor
at::Tensor hardtanh_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val) {
    static auto op = create_hardtanh_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, min_val, max_val);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardtanh_, name, "aten::hardtanh_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardtanh_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardtanh_, schema_str, "hardtanh_(Tensor(a!) self, Scalar min_val=-1, Scalar max_val=1) -> Tensor(a!)")

// aten::hardtanh_(Tensor(a!) self, Scalar min_val=-1, Scalar max_val=1) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<hardtanh_::schema> create_hardtanh__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(hardtanh_::name, hardtanh_::overload_name)
      .typed<hardtanh_::schema>();
}

// aten::hardtanh_(Tensor(a!) self, Scalar min_val=-1, Scalar max_val=1) -> Tensor(a!)
at::Tensor & hardtanh_::call(at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val) {
    static auto op = create_hardtanh__typed_handle();
    return op.call(self, min_val, max_val);
}

// aten::hardtanh_(Tensor(a!) self, Scalar min_val=-1, Scalar max_val=1) -> Tensor(a!)
at::Tensor & hardtanh_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val) {
    static auto op = create_hardtanh__typed_handle();
    return op.redispatch(dispatchKeySet, self, min_val, max_val);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardswish_, name, "aten::hardswish_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardswish_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardswish_, schema_str, "hardswish_(Tensor(a!) self) -> Tensor(a!)")

// aten::hardswish_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<hardswish_::schema> create_hardswish__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(hardswish_::name, hardswish_::overload_name)
      .typed<hardswish_::schema>();
}

// aten::hardswish_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & hardswish_::call(at::Tensor & self) {
    static auto op = create_hardswish__typed_handle();
    return op.call(self);
}

// aten::hardswish_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & hardswish_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_hardswish__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(leaky_relu_backward_grad_input, name, "aten::leaky_relu_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(leaky_relu_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(leaky_relu_backward_grad_input, schema_str, "leaky_relu_backward.grad_input(Tensor grad_output, Tensor self, Scalar negative_slope, bool self_is_result, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::leaky_relu_backward.grad_input(Tensor grad_output, Tensor self, Scalar negative_slope, bool self_is_result, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<leaky_relu_backward_grad_input::schema> create_leaky_relu_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(leaky_relu_backward_grad_input::name, leaky_relu_backward_grad_input::overload_name)
      .typed<leaky_relu_backward_grad_input::schema>();
}

// aten::leaky_relu_backward.grad_input(Tensor grad_output, Tensor self, Scalar negative_slope, bool self_is_result, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & leaky_relu_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & negative_slope, bool self_is_result, at::Tensor & grad_input) {
    static auto op = create_leaky_relu_backward_grad_input_typed_handle();
    return op.call(grad_output, self, negative_slope, self_is_result, grad_input);
}

// aten::leaky_relu_backward.grad_input(Tensor grad_output, Tensor self, Scalar negative_slope, bool self_is_result, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & leaky_relu_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & negative_slope, bool self_is_result, at::Tensor & grad_input) {
    static auto op = create_leaky_relu_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, negative_slope, self_is_result, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(leaky_relu_backward, name, "aten::leaky_relu_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(leaky_relu_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(leaky_relu_backward, schema_str, "leaky_relu_backward(Tensor grad_output, Tensor self, Scalar negative_slope, bool self_is_result) -> Tensor")

// aten::leaky_relu_backward(Tensor grad_output, Tensor self, Scalar negative_slope, bool self_is_result) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<leaky_relu_backward::schema> create_leaky_relu_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(leaky_relu_backward::name, leaky_relu_backward::overload_name)
      .typed<leaky_relu_backward::schema>();
}

// aten::leaky_relu_backward(Tensor grad_output, Tensor self, Scalar negative_slope, bool self_is_result) -> Tensor
at::Tensor leaky_relu_backward::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & negative_slope, bool self_is_result) {
    static auto op = create_leaky_relu_backward_typed_handle();
    return op.call(grad_output, self, negative_slope, self_is_result);
}

// aten::leaky_relu_backward(Tensor grad_output, Tensor self, Scalar negative_slope, bool self_is_result) -> Tensor
at::Tensor leaky_relu_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & negative_slope, bool self_is_result) {
    static auto op = create_leaky_relu_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, negative_slope, self_is_result);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(leaky_relu_, name, "aten::leaky_relu_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(leaky_relu_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(leaky_relu_, schema_str, "leaky_relu_(Tensor(a!) self, Scalar negative_slope=0.01) -> Tensor(a!)")

// aten::leaky_relu_(Tensor(a!) self, Scalar negative_slope=0.01) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<leaky_relu_::schema> create_leaky_relu__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(leaky_relu_::name, leaky_relu_::overload_name)
      .typed<leaky_relu_::schema>();
}

// aten::leaky_relu_(Tensor(a!) self, Scalar negative_slope=0.01) -> Tensor(a!)
at::Tensor & leaky_relu_::call(at::Tensor & self, const at::Scalar & negative_slope) {
    static auto op = create_leaky_relu__typed_handle();
    return op.call(self, negative_slope);
}

// aten::leaky_relu_(Tensor(a!) self, Scalar negative_slope=0.01) -> Tensor(a!)
at::Tensor & leaky_relu_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & negative_slope) {
    static auto op = create_leaky_relu__typed_handle();
    return op.redispatch(dispatchKeySet, self, negative_slope);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(softplus_out, name, "aten::softplus")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(softplus_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(softplus_out, schema_str, "softplus.out(Tensor self, Scalar beta=1, Scalar threshold=20, *, Tensor(a!) out) -> Tensor(a!)")

// aten::softplus.out(Tensor self, Scalar beta=1, Scalar threshold=20, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<softplus_out::schema> create_softplus_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(softplus_out::name, softplus_out::overload_name)
      .typed<softplus_out::schema>();
}

// aten::softplus.out(Tensor self, Scalar beta=1, Scalar threshold=20, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & softplus_out::call(const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold, at::Tensor & out) {
    static auto op = create_softplus_out_typed_handle();
    return op.call(self, beta, threshold, out);
}

// aten::softplus.out(Tensor self, Scalar beta=1, Scalar threshold=20, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & softplus_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold, at::Tensor & out) {
    static auto op = create_softplus_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, beta, threshold, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(softplus, name, "aten::softplus")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(softplus, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(softplus, schema_str, "softplus(Tensor self, Scalar beta=1, Scalar threshold=20) -> Tensor")

// aten::softplus(Tensor self, Scalar beta=1, Scalar threshold=20) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<softplus::schema> create_softplus_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(softplus::name, softplus::overload_name)
      .typed<softplus::schema>();
}

// aten::softplus(Tensor self, Scalar beta=1, Scalar threshold=20) -> Tensor
at::Tensor softplus::call(const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold) {
    static auto op = create_softplus_typed_handle();
    return op.call(self, beta, threshold);
}

// aten::softplus(Tensor self, Scalar beta=1, Scalar threshold=20) -> Tensor
at::Tensor softplus::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold) {
    static auto op = create_softplus_typed_handle();
    return op.redispatch(dispatchKeySet, self, beta, threshold);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(softshrink_out, name, "aten::softshrink")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(softshrink_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(softshrink_out, schema_str, "softshrink.out(Tensor self, Scalar lambd=0.5, *, Tensor(a!) out) -> Tensor(a!)")

// aten::softshrink.out(Tensor self, Scalar lambd=0.5, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<softshrink_out::schema> create_softshrink_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(softshrink_out::name, softshrink_out::overload_name)
      .typed<softshrink_out::schema>();
}

// aten::softshrink.out(Tensor self, Scalar lambd=0.5, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & softshrink_out::call(const at::Tensor & self, const at::Scalar & lambd, at::Tensor & out) {
    static auto op = create_softshrink_out_typed_handle();
    return op.call(self, lambd, out);
}

// aten::softshrink.out(Tensor self, Scalar lambd=0.5, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & softshrink_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & lambd, at::Tensor & out) {
    static auto op = create_softshrink_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, lambd, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_adaptive_avg_pool2d, name, "aten::mkldnn_adaptive_avg_pool2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_adaptive_avg_pool2d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_adaptive_avg_pool2d, schema_str, "mkldnn_adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor")

// aten::mkldnn_adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<mkldnn_adaptive_avg_pool2d::schema> create_mkldnn_adaptive_avg_pool2d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mkldnn_adaptive_avg_pool2d::name, mkldnn_adaptive_avg_pool2d::overload_name)
      .typed<mkldnn_adaptive_avg_pool2d::schema>();
}

// aten::mkldnn_adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor
at::Tensor mkldnn_adaptive_avg_pool2d::call(const at::Tensor & self, at::IntArrayRef output_size) {
    static auto op = create_mkldnn_adaptive_avg_pool2d_typed_handle();
    return op.call(self, output_size);
}

// aten::mkldnn_adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor
at::Tensor mkldnn_adaptive_avg_pool2d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size) {
    static auto op = create_mkldnn_adaptive_avg_pool2d_typed_handle();
    return op.redispatch(dispatchKeySet, self, output_size);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_adaptive_avg_pool2d, name, "aten::_adaptive_avg_pool2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_adaptive_avg_pool2d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_adaptive_avg_pool2d, schema_str, "_adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor")

// aten::_adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_adaptive_avg_pool2d::schema> create__adaptive_avg_pool2d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_adaptive_avg_pool2d::name, _adaptive_avg_pool2d::overload_name)
      .typed<_adaptive_avg_pool2d::schema>();
}

// aten::_adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor
at::Tensor _adaptive_avg_pool2d::call(const at::Tensor & self, at::IntArrayRef output_size) {
    static auto op = create__adaptive_avg_pool2d_typed_handle();
    return op.call(self, output_size);
}

// aten::_adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor
at::Tensor _adaptive_avg_pool2d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size) {
    static auto op = create__adaptive_avg_pool2d_typed_handle();
    return op.redispatch(dispatchKeySet, self, output_size);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(avg_pool2d_backward_grad_input, name, "aten::avg_pool2d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(avg_pool2d_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(avg_pool2d_backward_grad_input, schema_str, "avg_pool2d_backward.grad_input(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::avg_pool2d_backward.grad_input(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<avg_pool2d_backward_grad_input::schema> create_avg_pool2d_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(avg_pool2d_backward_grad_input::name, avg_pool2d_backward_grad_input::overload_name)
      .typed<avg_pool2d_backward_grad_input::schema>();
}

// aten::avg_pool2d_backward.grad_input(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & avg_pool2d_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, at::Tensor & grad_input) {
    static auto op = create_avg_pool2d_backward_grad_input_typed_handle();
    return op.call(grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, grad_input);
}

// aten::avg_pool2d_backward.grad_input(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & avg_pool2d_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, at::Tensor & grad_input) {
    static auto op = create_avg_pool2d_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(avg_pool3d, name, "aten::avg_pool3d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(avg_pool3d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(avg_pool3d, schema_str, "avg_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor")

// aten::avg_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<avg_pool3d::schema> create_avg_pool3d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(avg_pool3d::name, avg_pool3d::overload_name)
      .typed<avg_pool3d::schema>();
}

// aten::avg_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
at::Tensor avg_pool3d::call(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
    static auto op = create_avg_pool3d_typed_handle();
    return op.call(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

// aten::avg_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
at::Tensor avg_pool3d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
    static auto op = create_avg_pool3d_typed_handle();
    return op.redispatch(dispatchKeySet, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(avg_pool3d_backward, name, "aten::avg_pool3d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(avg_pool3d_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(avg_pool3d_backward, schema_str, "avg_pool3d_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor")

// aten::avg_pool3d_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<avg_pool3d_backward::schema> create_avg_pool3d_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(avg_pool3d_backward::name, avg_pool3d_backward::overload_name)
      .typed<avg_pool3d_backward::schema>();
}

// aten::avg_pool3d_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor
at::Tensor avg_pool3d_backward::call(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
    static auto op = create_avg_pool3d_backward_typed_handle();
    return op.call(grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

// aten::avg_pool3d_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor
at::Tensor avg_pool3d_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
    static auto op = create_avg_pool3d_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fractional_max_pool2d_backward_grad_input, name, "aten::fractional_max_pool2d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fractional_max_pool2d_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fractional_max_pool2d_backward_grad_input, schema_str, "fractional_max_pool2d_backward.grad_input(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] output_size, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::fractional_max_pool2d_backward.grad_input(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] output_size, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<fractional_max_pool2d_backward_grad_input::schema> create_fractional_max_pool2d_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fractional_max_pool2d_backward_grad_input::name, fractional_max_pool2d_backward_grad_input::overload_name)
      .typed<fractional_max_pool2d_backward_grad_input::schema>();
}

// aten::fractional_max_pool2d_backward.grad_input(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] output_size, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & fractional_max_pool2d_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & indices, at::Tensor & grad_input) {
    static auto op = create_fractional_max_pool2d_backward_grad_input_typed_handle();
    return op.call(grad_output, self, kernel_size, output_size, indices, grad_input);
}

// aten::fractional_max_pool2d_backward.grad_input(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] output_size, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & fractional_max_pool2d_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & indices, at::Tensor & grad_input) {
    static auto op = create_fractional_max_pool2d_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, kernel_size, output_size, indices, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_pool2d_with_indices_backward, name, "aten::max_pool2d_with_indices_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_pool2d_with_indices_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_pool2d_with_indices_backward, schema_str, "max_pool2d_with_indices_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices) -> Tensor")

// aten::max_pool2d_with_indices_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<max_pool2d_with_indices_backward::schema> create_max_pool2d_with_indices_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(max_pool2d_with_indices_backward::name, max_pool2d_with_indices_backward::overload_name)
      .typed<max_pool2d_with_indices_backward::schema>();
}

// aten::max_pool2d_with_indices_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices) -> Tensor
at::Tensor max_pool2d_with_indices_backward::call(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices) {
    static auto op = create_max_pool2d_with_indices_backward_typed_handle();
    return op.call(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}

// aten::max_pool2d_with_indices_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices) -> Tensor
at::Tensor max_pool2d_with_indices_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices) {
    static auto op = create_max_pool2d_with_indices_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_pool3d_with_indices, name, "aten::max_pool3d_with_indices")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_pool3d_with_indices, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_pool3d_with_indices, schema_str, "max_pool3d_with_indices(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)")

// aten::max_pool3d_with_indices(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<max_pool3d_with_indices::schema> create_max_pool3d_with_indices_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(max_pool3d_with_indices::name, max_pool3d_with_indices::overload_name)
      .typed<max_pool3d_with_indices::schema>();
}

// aten::max_pool3d_with_indices(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> max_pool3d_with_indices::call(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
    static auto op = create_max_pool3d_with_indices_typed_handle();
    return op.call(self, kernel_size, stride, padding, dilation, ceil_mode);
}

// aten::max_pool3d_with_indices(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> max_pool3d_with_indices::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
    static auto op = create_max_pool3d_with_indices_typed_handle();
    return op.redispatch(dispatchKeySet, self, kernel_size, stride, padding, dilation, ceil_mode);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_unpool2d_backward_grad_input, name, "aten::max_unpool2d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_unpool2d_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_unpool2d_backward_grad_input, schema_str, "max_unpool2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor indices, int[2] output_size, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::max_unpool2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor indices, int[2] output_size, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<max_unpool2d_backward_grad_input::schema> create_max_unpool2d_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(max_unpool2d_backward_grad_input::name, max_unpool2d_backward_grad_input::overload_name)
      .typed<max_unpool2d_backward_grad_input::schema>();
}

// aten::max_unpool2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor indices, int[2] output_size, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & max_unpool2d_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::Tensor & grad_input) {
    static auto op = create_max_unpool2d_backward_grad_input_typed_handle();
    return op.call(grad_output, self, indices, output_size, grad_input);
}

// aten::max_unpool2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor indices, int[2] output_size, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & max_unpool2d_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::Tensor & grad_input) {
    static auto op = create_max_unpool2d_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, indices, output_size, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reflection_pad2d, name, "aten::reflection_pad2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reflection_pad2d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reflection_pad2d, schema_str, "reflection_pad2d(Tensor self, int[4] padding) -> Tensor")

// aten::reflection_pad2d(Tensor self, int[4] padding) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<reflection_pad2d::schema> create_reflection_pad2d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(reflection_pad2d::name, reflection_pad2d::overload_name)
      .typed<reflection_pad2d::schema>();
}

// aten::reflection_pad2d(Tensor self, int[4] padding) -> Tensor
at::Tensor reflection_pad2d::call(const at::Tensor & self, at::IntArrayRef padding) {
    static auto op = create_reflection_pad2d_typed_handle();
    return op.call(self, padding);
}

// aten::reflection_pad2d(Tensor self, int[4] padding) -> Tensor
at::Tensor reflection_pad2d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef padding) {
    static auto op = create_reflection_pad2d_typed_handle();
    return op.redispatch(dispatchKeySet, self, padding);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_linear1d_backward_vec, name, "aten::upsample_linear1d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_linear1d_backward_vec, overload_name, "vec")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_linear1d_backward_vec, schema_str, "upsample_linear1d_backward.vec(Tensor grad_output, int[]? output_size, int[] input_size, bool align_corners, float[]? scale_factors) -> Tensor")

// aten::upsample_linear1d_backward.vec(Tensor grad_output, int[]? output_size, int[] input_size, bool align_corners, float[]? scale_factors) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<upsample_linear1d_backward_vec::schema> create_upsample_linear1d_backward_vec_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_linear1d_backward_vec::name, upsample_linear1d_backward_vec::overload_name)
      .typed<upsample_linear1d_backward_vec::schema>();
}

// aten::upsample_linear1d_backward.vec(Tensor grad_output, int[]? output_size, int[] input_size, bool align_corners, float[]? scale_factors) -> Tensor
at::Tensor upsample_linear1d_backward_vec::call(const at::Tensor & grad_output, c10::optional<at::IntArrayRef> output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
    static auto op = create_upsample_linear1d_backward_vec_typed_handle();
    return op.call(grad_output, output_size, input_size, align_corners, scale_factors);
}

// aten::upsample_linear1d_backward.vec(Tensor grad_output, int[]? output_size, int[] input_size, bool align_corners, float[]? scale_factors) -> Tensor
at::Tensor upsample_linear1d_backward_vec::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, c10::optional<at::IntArrayRef> output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
    static auto op = create_upsample_linear1d_backward_vec_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, output_size, input_size, align_corners, scale_factors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_bilinear2d_backward_vec, name, "aten::upsample_bilinear2d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_bilinear2d_backward_vec, overload_name, "vec")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_bilinear2d_backward_vec, schema_str, "upsample_bilinear2d_backward.vec(Tensor grad_output, int[]? output_size, int[] input_size, bool align_corners, float[]? scale_factors) -> Tensor")

// aten::upsample_bilinear2d_backward.vec(Tensor grad_output, int[]? output_size, int[] input_size, bool align_corners, float[]? scale_factors) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<upsample_bilinear2d_backward_vec::schema> create_upsample_bilinear2d_backward_vec_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_bilinear2d_backward_vec::name, upsample_bilinear2d_backward_vec::overload_name)
      .typed<upsample_bilinear2d_backward_vec::schema>();
}

// aten::upsample_bilinear2d_backward.vec(Tensor grad_output, int[]? output_size, int[] input_size, bool align_corners, float[]? scale_factors) -> Tensor
at::Tensor upsample_bilinear2d_backward_vec::call(const at::Tensor & grad_output, c10::optional<at::IntArrayRef> output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
    static auto op = create_upsample_bilinear2d_backward_vec_typed_handle();
    return op.call(grad_output, output_size, input_size, align_corners, scale_factors);
}

// aten::upsample_bilinear2d_backward.vec(Tensor grad_output, int[]? output_size, int[] input_size, bool align_corners, float[]? scale_factors) -> Tensor
at::Tensor upsample_bilinear2d_backward_vec::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, c10::optional<at::IntArrayRef> output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
    static auto op = create_upsample_bilinear2d_backward_vec_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, output_size, input_size, align_corners, scale_factors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_bicubic2d_vec, name, "aten::upsample_bicubic2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_bicubic2d_vec, overload_name, "vec")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_bicubic2d_vec, schema_str, "upsample_bicubic2d.vec(Tensor input, int[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor")

// aten::upsample_bicubic2d.vec(Tensor input, int[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<upsample_bicubic2d_vec::schema> create_upsample_bicubic2d_vec_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_bicubic2d_vec::name, upsample_bicubic2d_vec::overload_name)
      .typed<upsample_bicubic2d_vec::schema>();
}

// aten::upsample_bicubic2d.vec(Tensor input, int[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
at::Tensor upsample_bicubic2d_vec::call(const at::Tensor & input, c10::optional<at::IntArrayRef> output_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
    static auto op = create_upsample_bicubic2d_vec_typed_handle();
    return op.call(input, output_size, align_corners, scale_factors);
}

// aten::upsample_bicubic2d.vec(Tensor input, int[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
at::Tensor upsample_bicubic2d_vec::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, c10::optional<at::IntArrayRef> output_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
    static auto op = create_upsample_bicubic2d_vec_typed_handle();
    return op.redispatch(dispatchKeySet, input, output_size, align_corners, scale_factors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_bicubic2d_backward_vec, name, "aten::upsample_bicubic2d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_bicubic2d_backward_vec, overload_name, "vec")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_bicubic2d_backward_vec, schema_str, "upsample_bicubic2d_backward.vec(Tensor grad_output, int[]? output_size, int[] input_size, bool align_corners, float[]? scale_factors) -> Tensor")

// aten::upsample_bicubic2d_backward.vec(Tensor grad_output, int[]? output_size, int[] input_size, bool align_corners, float[]? scale_factors) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<upsample_bicubic2d_backward_vec::schema> create_upsample_bicubic2d_backward_vec_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_bicubic2d_backward_vec::name, upsample_bicubic2d_backward_vec::overload_name)
      .typed<upsample_bicubic2d_backward_vec::schema>();
}

// aten::upsample_bicubic2d_backward.vec(Tensor grad_output, int[]? output_size, int[] input_size, bool align_corners, float[]? scale_factors) -> Tensor
at::Tensor upsample_bicubic2d_backward_vec::call(const at::Tensor & grad_output, c10::optional<at::IntArrayRef> output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
    static auto op = create_upsample_bicubic2d_backward_vec_typed_handle();
    return op.call(grad_output, output_size, input_size, align_corners, scale_factors);
}

// aten::upsample_bicubic2d_backward.vec(Tensor grad_output, int[]? output_size, int[] input_size, bool align_corners, float[]? scale_factors) -> Tensor
at::Tensor upsample_bicubic2d_backward_vec::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, c10::optional<at::IntArrayRef> output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
    static auto op = create_upsample_bicubic2d_backward_vec_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, output_size, input_size, align_corners, scale_factors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_linear1d_backward, name, "aten::upsample_linear1d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_linear1d_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_linear1d_backward, schema_str, "upsample_linear1d_backward(Tensor grad_output, int[1] output_size, int[3] input_size, bool align_corners, float? scales=None) -> Tensor")

// aten::upsample_linear1d_backward(Tensor grad_output, int[1] output_size, int[3] input_size, bool align_corners, float? scales=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<upsample_linear1d_backward::schema> create_upsample_linear1d_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_linear1d_backward::name, upsample_linear1d_backward::overload_name)
      .typed<upsample_linear1d_backward::schema>();
}

// aten::upsample_linear1d_backward(Tensor grad_output, int[1] output_size, int[3] input_size, bool align_corners, float? scales=None) -> Tensor
at::Tensor upsample_linear1d_backward::call(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales) {
    static auto op = create_upsample_linear1d_backward_typed_handle();
    return op.call(grad_output, output_size, input_size, align_corners, scales);
}

// aten::upsample_linear1d_backward(Tensor grad_output, int[1] output_size, int[3] input_size, bool align_corners, float? scales=None) -> Tensor
at::Tensor upsample_linear1d_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales) {
    static auto op = create_upsample_linear1d_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, output_size, input_size, align_corners, scales);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_bicubic2d_backward_grad_input, name, "aten::upsample_bicubic2d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_bicubic2d_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_bicubic2d_backward_grad_input, schema_str, "upsample_bicubic2d_backward.grad_input(Tensor grad_output, int[2] output_size, int[4] input_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::upsample_bicubic2d_backward.grad_input(Tensor grad_output, int[2] output_size, int[4] input_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<upsample_bicubic2d_backward_grad_input::schema> create_upsample_bicubic2d_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_bicubic2d_backward_grad_input::name, upsample_bicubic2d_backward_grad_input::overload_name)
      .typed<upsample_bicubic2d_backward_grad_input::schema>();
}

// aten::upsample_bicubic2d_backward.grad_input(Tensor grad_output, int[2] output_size, int[4] input_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & upsample_bicubic2d_backward_grad_input::call(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input) {
    static auto op = create_upsample_bicubic2d_backward_grad_input_typed_handle();
    return op.call(grad_output, output_size, input_size, align_corners, scales_h, scales_w, grad_input);
}

// aten::upsample_bicubic2d_backward.grad_input(Tensor grad_output, int[2] output_size, int[4] input_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & upsample_bicubic2d_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input) {
    static auto op = create_upsample_bicubic2d_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, output_size, input_size, align_corners, scales_h, scales_w, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_trilinear3d_out, name, "aten::upsample_trilinear3d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_trilinear3d_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_trilinear3d_out, schema_str, "upsample_trilinear3d.out(Tensor self, int[3] output_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::upsample_trilinear3d.out(Tensor self, int[3] output_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<upsample_trilinear3d_out::schema> create_upsample_trilinear3d_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_trilinear3d_out::name, upsample_trilinear3d_out::overload_name)
      .typed<upsample_trilinear3d_out::schema>();
}

// aten::upsample_trilinear3d.out(Tensor self, int[3] output_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & upsample_trilinear3d_out::call(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out) {
    static auto op = create_upsample_trilinear3d_out_typed_handle();
    return op.call(self, output_size, align_corners, scales_d, scales_h, scales_w, out);
}

// aten::upsample_trilinear3d.out(Tensor self, int[3] output_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & upsample_trilinear3d_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out) {
    static auto op = create_upsample_trilinear3d_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, output_size, align_corners, scales_d, scales_h, scales_w, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_trilinear3d_backward_grad_input, name, "aten::upsample_trilinear3d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_trilinear3d_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_trilinear3d_backward_grad_input, schema_str, "upsample_trilinear3d_backward.grad_input(Tensor grad_output, int[3] output_size, int[5] input_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::upsample_trilinear3d_backward.grad_input(Tensor grad_output, int[3] output_size, int[5] input_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<upsample_trilinear3d_backward_grad_input::schema> create_upsample_trilinear3d_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_trilinear3d_backward_grad_input::name, upsample_trilinear3d_backward_grad_input::overload_name)
      .typed<upsample_trilinear3d_backward_grad_input::schema>();
}

// aten::upsample_trilinear3d_backward.grad_input(Tensor grad_output, int[3] output_size, int[5] input_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & upsample_trilinear3d_backward_grad_input::call(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input) {
    static auto op = create_upsample_trilinear3d_backward_grad_input_typed_handle();
    return op.call(grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w, grad_input);
}

// aten::upsample_trilinear3d_backward.grad_input(Tensor grad_output, int[3] output_size, int[5] input_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & upsample_trilinear3d_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input) {
    static auto op = create_upsample_trilinear3d_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest1d_backward, name, "aten::upsample_nearest1d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest1d_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest1d_backward, schema_str, "upsample_nearest1d_backward(Tensor grad_output, int[1] output_size, int[3] input_size, float? scales=None) -> Tensor")

// aten::upsample_nearest1d_backward(Tensor grad_output, int[1] output_size, int[3] input_size, float? scales=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<upsample_nearest1d_backward::schema> create_upsample_nearest1d_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_nearest1d_backward::name, upsample_nearest1d_backward::overload_name)
      .typed<upsample_nearest1d_backward::schema>();
}

// aten::upsample_nearest1d_backward(Tensor grad_output, int[1] output_size, int[3] input_size, float? scales=None) -> Tensor
at::Tensor upsample_nearest1d_backward::call(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales) {
    static auto op = create_upsample_nearest1d_backward_typed_handle();
    return op.call(grad_output, output_size, input_size, scales);
}

// aten::upsample_nearest1d_backward(Tensor grad_output, int[1] output_size, int[3] input_size, float? scales=None) -> Tensor
at::Tensor upsample_nearest1d_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales) {
    static auto op = create_upsample_nearest1d_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, output_size, input_size, scales);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest2d_backward, name, "aten::upsample_nearest2d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest2d_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest2d_backward, schema_str, "upsample_nearest2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size, float? scales_h=None, float? scales_w=None) -> Tensor")

// aten::upsample_nearest2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size, float? scales_h=None, float? scales_w=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<upsample_nearest2d_backward::schema> create_upsample_nearest2d_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_nearest2d_backward::name, upsample_nearest2d_backward::overload_name)
      .typed<upsample_nearest2d_backward::schema>();
}

// aten::upsample_nearest2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size, float? scales_h=None, float? scales_w=None) -> Tensor
at::Tensor upsample_nearest2d_backward::call(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales_h, c10::optional<double> scales_w) {
    static auto op = create_upsample_nearest2d_backward_typed_handle();
    return op.call(grad_output, output_size, input_size, scales_h, scales_w);
}

// aten::upsample_nearest2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size, float? scales_h=None, float? scales_w=None) -> Tensor
at::Tensor upsample_nearest2d_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales_h, c10::optional<double> scales_w) {
    static auto op = create_upsample_nearest2d_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, output_size, input_size, scales_h, scales_w);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest3d_backward_grad_input, name, "aten::upsample_nearest3d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest3d_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest3d_backward_grad_input, schema_str, "upsample_nearest3d_backward.grad_input(Tensor grad_output, int[3] output_size, int[5] input_size, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::upsample_nearest3d_backward.grad_input(Tensor grad_output, int[3] output_size, int[5] input_size, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<upsample_nearest3d_backward_grad_input::schema> create_upsample_nearest3d_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_nearest3d_backward_grad_input::name, upsample_nearest3d_backward_grad_input::overload_name)
      .typed<upsample_nearest3d_backward_grad_input::schema>();
}

// aten::upsample_nearest3d_backward.grad_input(Tensor grad_output, int[3] output_size, int[5] input_size, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & upsample_nearest3d_backward_grad_input::call(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input) {
    static auto op = create_upsample_nearest3d_backward_grad_input_typed_handle();
    return op.call(grad_output, output_size, input_size, scales_d, scales_h, scales_w, grad_input);
}

// aten::upsample_nearest3d_backward.grad_input(Tensor grad_output, int[3] output_size, int[5] input_size, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & upsample_nearest3d_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input) {
    static auto op = create_upsample_nearest3d_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, output_size, input_size, scales_d, scales_h, scales_w, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv_transpose2d_backward_output_mask, name, "aten::slow_conv_transpose2d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv_transpose2d_backward_output_mask, overload_name, "output_mask")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv_transpose2d_backward_output_mask, schema_str, "slow_conv_transpose2d_backward.output_mask(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] output_padding, int[2] dilation, Tensor columns, Tensor ones, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)")

// aten::slow_conv_transpose2d_backward.output_mask(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] output_padding, int[2] dilation, Tensor columns, Tensor ones, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)
static C10_NOINLINE c10::TypedOperatorHandle<slow_conv_transpose2d_backward_output_mask::schema> create_slow_conv_transpose2d_backward_output_mask_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(slow_conv_transpose2d_backward_output_mask::name, slow_conv_transpose2d_backward_output_mask::overload_name)
      .typed<slow_conv_transpose2d_backward_output_mask::schema>();
}

// aten::slow_conv_transpose2d_backward.output_mask(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] output_padding, int[2] dilation, Tensor columns, Tensor ones, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> slow_conv_transpose2d_backward_output_mask::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, const at::Tensor & columns, const at::Tensor & ones, ::std::array<bool,3> output_mask) {
    static auto op = create_slow_conv_transpose2d_backward_output_mask_typed_handle();
    return op.call(grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, columns, ones, output_mask);
}

// aten::slow_conv_transpose2d_backward.output_mask(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] output_padding, int[2] dilation, Tensor columns, Tensor ones, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> slow_conv_transpose2d_backward_output_mask::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, const at::Tensor & columns, const at::Tensor & ones, ::std::array<bool,3> output_mask) {
    static auto op = create_slow_conv_transpose2d_backward_output_mask_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, columns, ones, output_mask);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv_transpose3d, name, "aten::slow_conv_transpose3d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv_transpose3d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv_transpose3d, schema_str, "slow_conv_transpose3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] output_padding=0, int[3] dilation=1) -> Tensor")

// aten::slow_conv_transpose3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] output_padding=0, int[3] dilation=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<slow_conv_transpose3d::schema> create_slow_conv_transpose3d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(slow_conv_transpose3d::name, slow_conv_transpose3d::overload_name)
      .typed<slow_conv_transpose3d::schema>();
}

// aten::slow_conv_transpose3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] output_padding=0, int[3] dilation=1) -> Tensor
at::Tensor slow_conv_transpose3d::call(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation) {
    static auto op = create_slow_conv_transpose3d_typed_handle();
    return op.call(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}

// aten::slow_conv_transpose3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] output_padding=0, int[3] dilation=1) -> Tensor
at::Tensor slow_conv_transpose3d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation) {
    static auto op = create_slow_conv_transpose3d_typed_handle();
    return op.redispatch(dispatchKeySet, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv_transpose3d_backward_grad_output, name, "aten::slow_conv_transpose3d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv_transpose3d_backward_grad_output, overload_name, "grad_output")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv_transpose3d_backward_grad_output, schema_str, "slow_conv_transpose3d_backward.grad_output(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, int[3] output_padding, int[3] dilation, Tensor finput, Tensor fgrad_input, *, Tensor(a!) grad_input, Tensor(b!) grad_weight, Tensor(c!) grad_bias) -> (Tensor(a!), Tensor(b!), Tensor(c!))")

// aten::slow_conv_transpose3d_backward.grad_output(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, int[3] output_padding, int[3] dilation, Tensor finput, Tensor fgrad_input, *, Tensor(a!) grad_input, Tensor(b!) grad_weight, Tensor(c!) grad_bias) -> (Tensor(a!), Tensor(b!), Tensor(c!))
static C10_NOINLINE c10::TypedOperatorHandle<slow_conv_transpose3d_backward_grad_output::schema> create_slow_conv_transpose3d_backward_grad_output_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(slow_conv_transpose3d_backward_grad_output::name, slow_conv_transpose3d_backward_grad_output::overload_name)
      .typed<slow_conv_transpose3d_backward_grad_output::schema>();
}

// aten::slow_conv_transpose3d_backward.grad_output(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, int[3] output_padding, int[3] dilation, Tensor finput, Tensor fgrad_input, *, Tensor(a!) grad_input, Tensor(b!) grad_weight, Tensor(c!) grad_bias) -> (Tensor(a!), Tensor(b!), Tensor(c!))
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> slow_conv_transpose3d_backward_grad_output::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, const at::Tensor & finput, const at::Tensor & fgrad_input, at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias) {
    static auto op = create_slow_conv_transpose3d_backward_grad_output_typed_handle();
    return op.call(grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, finput, fgrad_input, grad_input, grad_weight, grad_bias);
}

// aten::slow_conv_transpose3d_backward.grad_output(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, int[3] output_padding, int[3] dilation, Tensor finput, Tensor fgrad_input, *, Tensor(a!) grad_input, Tensor(b!) grad_weight, Tensor(c!) grad_bias) -> (Tensor(a!), Tensor(b!), Tensor(c!))
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> slow_conv_transpose3d_backward_grad_output::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, const at::Tensor & finput, const at::Tensor & fgrad_input, at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias) {
    static auto op = create_slow_conv_transpose3d_backward_grad_output_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, finput, fgrad_input, grad_input, grad_weight, grad_bias);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(thnn_conv2d_out, name, "aten::thnn_conv2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(thnn_conv2d_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(thnn_conv2d_out, schema_str, "thnn_conv2d.out(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, *, Tensor(a!) out) -> Tensor(a!)")

// aten::thnn_conv2d.out(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<thnn_conv2d_out::schema> create_thnn_conv2d_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(thnn_conv2d_out::name, thnn_conv2d_out::overload_name)
      .typed<thnn_conv2d_out::schema>();
}

// aten::thnn_conv2d.out(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & thnn_conv2d_out::call(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & out) {
    static auto op = create_thnn_conv2d_out_typed_handle();
    return op.call(self, weight, kernel_size, bias, stride, padding, out);
}

// aten::thnn_conv2d.out(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & thnn_conv2d_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & out) {
    static auto op = create_thnn_conv2d_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, weight, kernel_size, bias, stride, padding, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv3d_forward, name, "aten::slow_conv3d_forward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv3d_forward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv3d_forward, schema_str, "slow_conv3d_forward(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias, int[3] stride, int[3] padding) -> (Tensor output, Tensor finput, Tensor fgrad_input)")

// aten::slow_conv3d_forward(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias, int[3] stride, int[3] padding) -> (Tensor output, Tensor finput, Tensor fgrad_input)
static C10_NOINLINE c10::TypedOperatorHandle<slow_conv3d_forward::schema> create_slow_conv3d_forward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(slow_conv3d_forward::name, slow_conv3d_forward::overload_name)
      .typed<slow_conv3d_forward::schema>();
}

// aten::slow_conv3d_forward(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias, int[3] stride, int[3] padding) -> (Tensor output, Tensor finput, Tensor fgrad_input)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> slow_conv3d_forward::call(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding) {
    static auto op = create_slow_conv3d_forward_typed_handle();
    return op.call(self, weight, kernel_size, bias, stride, padding);
}

// aten::slow_conv3d_forward(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias, int[3] stride, int[3] padding) -> (Tensor output, Tensor finput, Tensor fgrad_input)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> slow_conv3d_forward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding) {
    static auto op = create_slow_conv3d_forward_typed_handle();
    return op.redispatch(dispatchKeySet, self, weight, kernel_size, bias, stride, padding);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(im2col, name, "aten::im2col")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(im2col, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(im2col, schema_str, "im2col(Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor")

// aten::im2col(Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<im2col::schema> create_im2col_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(im2col::name, im2col::overload_name)
      .typed<im2col::schema>();
}

// aten::im2col(Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor
at::Tensor im2col::call(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride) {
    static auto op = create_im2col_typed_handle();
    return op.call(self, kernel_size, dilation, padding, stride);
}

// aten::im2col(Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor
at::Tensor im2col::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride) {
    static auto op = create_im2col_typed_handle();
    return op.redispatch(dispatchKeySet, self, kernel_size, dilation, padding, stride);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(im2col_backward, name, "aten::im2col_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(im2col_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(im2col_backward, schema_str, "im2col_backward(Tensor grad_output, int[2] input_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor")

// aten::im2col_backward(Tensor grad_output, int[2] input_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<im2col_backward::schema> create_im2col_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(im2col_backward::name, im2col_backward::overload_name)
      .typed<im2col_backward::schema>();
}

// aten::im2col_backward(Tensor grad_output, int[2] input_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor
at::Tensor im2col_backward::call(const at::Tensor & grad_output, at::IntArrayRef input_size, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride) {
    static auto op = create_im2col_backward_typed_handle();
    return op.call(grad_output, input_size, kernel_size, dilation, padding, stride);
}

// aten::im2col_backward(Tensor grad_output, int[2] input_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor
at::Tensor im2col_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, at::IntArrayRef input_size, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride) {
    static auto op = create_im2col_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, input_size, kernel_size, dilation, padding, stride);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isneginf, name, "aten::isneginf")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isneginf, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isneginf, schema_str, "isneginf(Tensor self) -> Tensor")

// aten::isneginf(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<isneginf::schema> create_isneginf_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(isneginf::name, isneginf::overload_name)
      .typed<isneginf::schema>();
}

// aten::isneginf(Tensor self) -> Tensor
at::Tensor isneginf::call(const at::Tensor & self) {
    static auto op = create_isneginf_typed_handle();
    return op.call(self);
}

// aten::isneginf(Tensor self) -> Tensor
at::Tensor isneginf::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_isneginf_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_add_batch_dim, name, "aten::_add_batch_dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_add_batch_dim, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_add_batch_dim, schema_str, "_add_batch_dim(Tensor self, int batch_dim, int level) -> Tensor")

// aten::_add_batch_dim(Tensor self, int batch_dim, int level) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_add_batch_dim::schema> create__add_batch_dim_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_add_batch_dim::name, _add_batch_dim::overload_name)
      .typed<_add_batch_dim::schema>();
}

// aten::_add_batch_dim(Tensor self, int batch_dim, int level) -> Tensor
at::Tensor _add_batch_dim::call(const at::Tensor & self, int64_t batch_dim, int64_t level) {
    static auto op = create__add_batch_dim_typed_handle();
    return op.call(self, batch_dim, level);
}

// aten::_add_batch_dim(Tensor self, int batch_dim, int level) -> Tensor
at::Tensor _add_batch_dim::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t batch_dim, int64_t level) {
    static auto op = create__add_batch_dim_typed_handle();
    return op.redispatch(dispatchKeySet, self, batch_dim, level);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_psi, name, "aten::special_psi")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_psi, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_psi, schema_str, "special_psi(Tensor self) -> Tensor")

// aten::special_psi(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<special_psi::schema> create_special_psi_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_psi::name, special_psi::overload_name)
      .typed<special_psi::schema>();
}

// aten::special_psi(Tensor self) -> Tensor
at::Tensor special_psi::call(const at::Tensor & self) {
    static auto op = create_special_psi_typed_handle();
    return op.call(self);
}

// aten::special_psi(Tensor self) -> Tensor
at::Tensor special_psi::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_special_psi_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_erf_out, name, "aten::special_erf")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_erf_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_erf_out, schema_str, "special_erf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::special_erf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<special_erf_out::schema> create_special_erf_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_erf_out::name, special_erf_out::overload_name)
      .typed<special_erf_out::schema>();
}

// aten::special_erf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_erf_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_erf_out_typed_handle();
    return op.call(self, out);
}

// aten::special_erf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_erf_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_erf_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_erfcx, name, "aten::special_erfcx")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_erfcx, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_erfcx, schema_str, "special_erfcx(Tensor self) -> Tensor")

// aten::special_erfcx(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<special_erfcx::schema> create_special_erfcx_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_erfcx::name, special_erfcx::overload_name)
      .typed<special_erfcx::schema>();
}

// aten::special_erfcx(Tensor self) -> Tensor
at::Tensor special_erfcx::call(const at::Tensor & self) {
    static auto op = create_special_erfcx_typed_handle();
    return op.call(self);
}

// aten::special_erfcx(Tensor self) -> Tensor
at::Tensor special_erfcx::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_special_erfcx_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_erfcx_out, name, "aten::special_erfcx")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_erfcx_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_erfcx_out, schema_str, "special_erfcx.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::special_erfcx.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<special_erfcx_out::schema> create_special_erfcx_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_erfcx_out::name, special_erfcx_out::overload_name)
      .typed<special_erfcx_out::schema>();
}

// aten::special_erfcx.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_erfcx_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_erfcx_out_typed_handle();
    return op.call(self, out);
}

// aten::special_erfcx.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_erfcx_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_erfcx_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_zeta_self_scalar_out, name, "aten::special_zeta")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_zeta_self_scalar_out, overload_name, "self_scalar_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_zeta_self_scalar_out, schema_str, "special_zeta.self_scalar_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::special_zeta.self_scalar_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<special_zeta_self_scalar_out::schema> create_special_zeta_self_scalar_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_zeta_self_scalar_out::name, special_zeta_self_scalar_out::overload_name)
      .typed<special_zeta_self_scalar_out::schema>();
}

// aten::special_zeta.self_scalar_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_zeta_self_scalar_out::call(const at::Scalar & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_special_zeta_self_scalar_out_typed_handle();
    return op.call(self, other, out);
}

// aten::special_zeta.self_scalar_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_zeta_self_scalar_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Scalar & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_special_zeta_self_scalar_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_zeta_other_scalar_out, name, "aten::special_zeta")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_zeta_other_scalar_out, overload_name, "other_scalar_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_zeta_other_scalar_out, schema_str, "special_zeta.other_scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::special_zeta.other_scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<special_zeta_other_scalar_out::schema> create_special_zeta_other_scalar_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_zeta_other_scalar_out::name, special_zeta_other_scalar_out::overload_name)
      .typed<special_zeta_other_scalar_out::schema>();
}

// aten::special_zeta.other_scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_zeta_other_scalar_out::call(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_special_zeta_other_scalar_out_typed_handle();
    return op.call(self, other, out);
}

// aten::special_zeta.other_scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_zeta_other_scalar_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_special_zeta_other_scalar_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_i0e, name, "aten::special_i0e")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_i0e, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_i0e, schema_str, "special_i0e(Tensor self) -> Tensor")

// aten::special_i0e(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<special_i0e::schema> create_special_i0e_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_i0e::name, special_i0e::overload_name)
      .typed<special_i0e::schema>();
}

// aten::special_i0e(Tensor self) -> Tensor
at::Tensor special_i0e::call(const at::Tensor & self) {
    static auto op = create_special_i0e_typed_handle();
    return op.call(self);
}

// aten::special_i0e(Tensor self) -> Tensor
at::Tensor special_i0e::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_special_i0e_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_i1, name, "aten::special_i1")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_i1, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_i1, schema_str, "special_i1(Tensor self) -> Tensor")

// aten::special_i1(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<special_i1::schema> create_special_i1_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_i1::name, special_i1::overload_name)
      .typed<special_i1::schema>();
}

// aten::special_i1(Tensor self) -> Tensor
at::Tensor special_i1::call(const at::Tensor & self) {
    static auto op = create_special_i1_typed_handle();
    return op.call(self);
}

// aten::special_i1(Tensor self) -> Tensor
at::Tensor special_i1::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_special_i1_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_logit, name, "aten::special_logit")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_logit, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_logit, schema_str, "special_logit(Tensor self, float? eps=None) -> Tensor")

// aten::special_logit(Tensor self, float? eps=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<special_logit::schema> create_special_logit_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_logit::name, special_logit::overload_name)
      .typed<special_logit::schema>();
}

// aten::special_logit(Tensor self, float? eps=None) -> Tensor
at::Tensor special_logit::call(const at::Tensor & self, c10::optional<double> eps) {
    static auto op = create_special_logit_typed_handle();
    return op.call(self, eps);
}

// aten::special_logit(Tensor self, float? eps=None) -> Tensor
at::Tensor special_logit::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<double> eps) {
    static auto op = create_special_logit_typed_handle();
    return op.redispatch(dispatchKeySet, self, eps);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_logsumexp_out, name, "aten::special_logsumexp")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_logsumexp_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_logsumexp_out, schema_str, "special_logsumexp.out(Tensor self, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)")

// aten::special_logsumexp.out(Tensor self, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<special_logsumexp_out::schema> create_special_logsumexp_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_logsumexp_out::name, special_logsumexp_out::overload_name)
      .typed<special_logsumexp_out::schema>();
}

// aten::special_logsumexp.out(Tensor self, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_logsumexp_out::call(const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
    static auto op = create_special_logsumexp_out_typed_handle();
    return op.call(self, dim, keepdim, out);
}

// aten::special_logsumexp.out(Tensor self, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_logsumexp_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
    static auto op = create_special_logsumexp_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_log1p_out, name, "aten::special_log1p")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_log1p_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_log1p_out, schema_str, "special_log1p.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::special_log1p.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<special_log1p_out::schema> create_special_log1p_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_log1p_out::name, special_log1p_out::overload_name)
      .typed<special_log1p_out::schema>();
}

// aten::special_log1p.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_log1p_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_log1p_out_typed_handle();
    return op.call(self, out);
}

// aten::special_log1p.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_log1p_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_log1p_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_log_softmax, name, "aten::special_log_softmax")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_log_softmax, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_log_softmax, schema_str, "special_log_softmax(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor")

// aten::special_log_softmax(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<special_log_softmax::schema> create_special_log_softmax_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_log_softmax::name, special_log_softmax::overload_name)
      .typed<special_log_softmax::schema>();
}

// aten::special_log_softmax(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
at::Tensor special_log_softmax::call(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_special_log_softmax_typed_handle();
    return op.call(self, dim, dtype);
}

// aten::special_log_softmax(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
at::Tensor special_log_softmax::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_special_log_softmax_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_gammainc_out, name, "aten::special_gammainc")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_gammainc_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_gammainc_out, schema_str, "special_gammainc.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::special_gammainc.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<special_gammainc_out::schema> create_special_gammainc_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_gammainc_out::name, special_gammainc_out::overload_name)
      .typed<special_gammainc_out::schema>();
}

// aten::special_gammainc.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_gammainc_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_special_gammainc_out_typed_handle();
    return op.call(self, other, out);
}

// aten::special_gammainc.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_gammainc_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_special_gammainc_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_gammaincc_out, name, "aten::special_gammaincc")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_gammaincc_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_gammaincc_out, schema_str, "special_gammaincc.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::special_gammaincc.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<special_gammaincc_out::schema> create_special_gammaincc_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_gammaincc_out::name, special_gammaincc_out::overload_name)
      .typed<special_gammaincc_out::schema>();
}

// aten::special_gammaincc.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_gammaincc_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_special_gammaincc_out_typed_handle();
    return op.call(self, other, out);
}

// aten::special_gammaincc.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_gammaincc_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_special_gammaincc_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_gammaincc, name, "aten::special_gammaincc")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_gammaincc, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_gammaincc, schema_str, "special_gammaincc(Tensor self, Tensor other) -> Tensor")

// aten::special_gammaincc(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<special_gammaincc::schema> create_special_gammaincc_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_gammaincc::name, special_gammaincc::overload_name)
      .typed<special_gammaincc::schema>();
}

// aten::special_gammaincc(Tensor self, Tensor other) -> Tensor
at::Tensor special_gammaincc::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_special_gammaincc_typed_handle();
    return op.call(self, other);
}

// aten::special_gammaincc(Tensor self, Tensor other) -> Tensor
at::Tensor special_gammaincc::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_special_gammaincc_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_multigammaln, name, "aten::special_multigammaln")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_multigammaln, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_multigammaln, schema_str, "special_multigammaln(Tensor self, int p) -> Tensor")

// aten::special_multigammaln(Tensor self, int p) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<special_multigammaln::schema> create_special_multigammaln_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_multigammaln::name, special_multigammaln::overload_name)
      .typed<special_multigammaln::schema>();
}

// aten::special_multigammaln(Tensor self, int p) -> Tensor
at::Tensor special_multigammaln::call(const at::Tensor & self, int64_t p) {
    static auto op = create_special_multigammaln_typed_handle();
    return op.call(self, p);
}

// aten::special_multigammaln(Tensor self, int p) -> Tensor
at::Tensor special_multigammaln::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t p) {
    static auto op = create_special_multigammaln_typed_handle();
    return op.redispatch(dispatchKeySet, self, p);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_fft2, name, "aten::fft_fft2")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_fft2, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_fft2, schema_str, "fft_fft2(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None) -> Tensor")

// aten::fft_fft2(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fft_fft2::schema> create_fft_fft2_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fft_fft2::name, fft_fft2::overload_name)
      .typed<fft_fft2::schema>();
}

// aten::fft_fft2(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None) -> Tensor
at::Tensor fft_fft2::call(const at::Tensor & self, c10::optional<at::IntArrayRef> s, at::IntArrayRef dim, c10::optional<c10::string_view> norm) {
    static auto op = create_fft_fft2_typed_handle();
    return op.call(self, s, dim, norm);
}

// aten::fft_fft2(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None) -> Tensor
at::Tensor fft_fft2::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> s, at::IntArrayRef dim, c10::optional<c10::string_view> norm) {
    static auto op = create_fft_fft2_typed_handle();
    return op.redispatch(dispatchKeySet, self, s, dim, norm);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_ifft2_out, name, "aten::fft_ifft2")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_ifft2_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_ifft2_out, schema_str, "fft_ifft2.out(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::fft_ifft2.out(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<fft_ifft2_out::schema> create_fft_ifft2_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fft_ifft2_out::name, fft_ifft2_out::overload_name)
      .typed<fft_ifft2_out::schema>();
}

// aten::fft_ifft2.out(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fft_ifft2_out::call(const at::Tensor & self, c10::optional<at::IntArrayRef> s, at::IntArrayRef dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
    static auto op = create_fft_ifft2_out_typed_handle();
    return op.call(self, s, dim, norm, out);
}

// aten::fft_ifft2.out(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fft_ifft2_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> s, at::IntArrayRef dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
    static auto op = create_fft_ifft2_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, s, dim, norm, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_fftn, name, "aten::fft_fftn")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_fftn, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_fftn, schema_str, "fft_fftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor")

// aten::fft_fftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fft_fftn::schema> create_fft_fftn_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fft_fftn::name, fft_fftn::overload_name)
      .typed<fft_fftn::schema>();
}

// aten::fft_fftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor
at::Tensor fft_fftn::call(const at::Tensor & self, c10::optional<at::IntArrayRef> s, c10::optional<at::IntArrayRef> dim, c10::optional<c10::string_view> norm) {
    static auto op = create_fft_fftn_typed_handle();
    return op.call(self, s, dim, norm);
}

// aten::fft_fftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor
at::Tensor fft_fftn::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> s, c10::optional<at::IntArrayRef> dim, c10::optional<c10::string_view> norm) {
    static auto op = create_fft_fftn_typed_handle();
    return op.redispatch(dispatchKeySet, self, s, dim, norm);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_ifftn_out, name, "aten::fft_ifftn")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_ifftn_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_ifftn_out, schema_str, "fft_ifftn.out(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::fft_ifftn.out(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<fft_ifftn_out::schema> create_fft_ifftn_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fft_ifftn_out::name, fft_ifftn_out::overload_name)
      .typed<fft_ifftn_out::schema>();
}

// aten::fft_ifftn.out(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fft_ifftn_out::call(const at::Tensor & self, c10::optional<at::IntArrayRef> s, c10::optional<at::IntArrayRef> dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
    static auto op = create_fft_ifftn_out_typed_handle();
    return op.call(self, s, dim, norm, out);
}

// aten::fft_ifftn.out(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fft_ifftn_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> s, c10::optional<at::IntArrayRef> dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
    static auto op = create_fft_ifftn_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, s, dim, norm, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_fftshift, name, "aten::fft_fftshift")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_fftshift, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_fftshift, schema_str, "fft_fftshift(Tensor self, int[1]? dim=None) -> Tensor")

// aten::fft_fftshift(Tensor self, int[1]? dim=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fft_fftshift::schema> create_fft_fftshift_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fft_fftshift::name, fft_fftshift::overload_name)
      .typed<fft_fftshift::schema>();
}

// aten::fft_fftshift(Tensor self, int[1]? dim=None) -> Tensor
at::Tensor fft_fftshift::call(const at::Tensor & self, c10::optional<at::IntArrayRef> dim) {
    static auto op = create_fft_fftshift_typed_handle();
    return op.call(self, dim);
}

// aten::fft_fftshift(Tensor self, int[1]? dim=None) -> Tensor
at::Tensor fft_fftshift::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> dim) {
    static auto op = create_fft_fftshift_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_det, name, "aten::linalg_det")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_det, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_det, schema_str, "linalg_det(Tensor self) -> Tensor")

// aten::linalg_det(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<linalg_det::schema> create_linalg_det_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_det::name, linalg_det::overload_name)
      .typed<linalg_det::schema>();
}

// aten::linalg_det(Tensor self) -> Tensor
at::Tensor linalg_det::call(const at::Tensor & self) {
    static auto op = create_linalg_det_typed_handle();
    return op.call(self);
}

// aten::linalg_det(Tensor self) -> Tensor
at::Tensor linalg_det::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_linalg_det_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_inv_ex_inverse, name, "aten::linalg_inv_ex")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_inv_ex_inverse, overload_name, "inverse")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_inv_ex_inverse, schema_str, "linalg_inv_ex.inverse(Tensor self, *, bool check_errors=False, Tensor(a!) inverse, Tensor(b!) info) -> (Tensor(a!) inverse, Tensor(b!) info)")

// aten::linalg_inv_ex.inverse(Tensor self, *, bool check_errors=False, Tensor(a!) inverse, Tensor(b!) info) -> (Tensor(a!) inverse, Tensor(b!) info)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_inv_ex_inverse::schema> create_linalg_inv_ex_inverse_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_inv_ex_inverse::name, linalg_inv_ex_inverse::overload_name)
      .typed<linalg_inv_ex_inverse::schema>();
}

// aten::linalg_inv_ex.inverse(Tensor self, *, bool check_errors=False, Tensor(a!) inverse, Tensor(b!) info) -> (Tensor(a!) inverse, Tensor(b!) info)
::std::tuple<at::Tensor &,at::Tensor &> linalg_inv_ex_inverse::call(const at::Tensor & self, bool check_errors, at::Tensor & inverse, at::Tensor & info) {
    static auto op = create_linalg_inv_ex_inverse_typed_handle();
    return op.call(self, check_errors, inverse, info);
}

// aten::linalg_inv_ex.inverse(Tensor self, *, bool check_errors=False, Tensor(a!) inverse, Tensor(b!) info) -> (Tensor(a!) inverse, Tensor(b!) info)
::std::tuple<at::Tensor &,at::Tensor &> linalg_inv_ex_inverse::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool check_errors, at::Tensor & inverse, at::Tensor & info) {
    static auto op = create_linalg_inv_ex_inverse_typed_handle();
    return op.redispatch(dispatchKeySet, self, check_errors, inverse, info);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_inv, name, "aten::linalg_inv")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_inv, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_inv, schema_str, "linalg_inv(Tensor self) -> Tensor")

// aten::linalg_inv(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<linalg_inv::schema> create_linalg_inv_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_inv::name, linalg_inv::overload_name)
      .typed<linalg_inv::schema>();
}

// aten::linalg_inv(Tensor self) -> Tensor
at::Tensor linalg_inv::call(const at::Tensor & self) {
    static auto op = create_linalg_inv_typed_handle();
    return op.call(self);
}

// aten::linalg_inv(Tensor self) -> Tensor
at::Tensor linalg_inv::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_linalg_inv_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(outer, name, "aten::outer")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(outer, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(outer, schema_str, "outer(Tensor self, Tensor vec2) -> Tensor")

// aten::outer(Tensor self, Tensor vec2) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<outer::schema> create_outer_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(outer::name, outer::overload_name)
      .typed<outer::schema>();
}

// aten::outer(Tensor self, Tensor vec2) -> Tensor
at::Tensor outer::call(const at::Tensor & self, const at::Tensor & vec2) {
    static auto op = create_outer_typed_handle();
    return op.call(self, vec2);
}

// aten::outer(Tensor self, Tensor vec2) -> Tensor
at::Tensor outer::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & vec2) {
    static auto op = create_outer_typed_handle();
    return op.redispatch(dispatchKeySet, self, vec2);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ger, name, "aten::ger")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ger, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ger, schema_str, "ger(Tensor self, Tensor vec2) -> Tensor")

// aten::ger(Tensor self, Tensor vec2) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<ger::schema> create_ger_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(ger::name, ger::overload_name)
      .typed<ger::schema>();
}

// aten::ger(Tensor self, Tensor vec2) -> Tensor
at::Tensor ger::call(const at::Tensor & self, const at::Tensor & vec2) {
    static auto op = create_ger_typed_handle();
    return op.call(self, vec2);
}

// aten::ger(Tensor self, Tensor vec2) -> Tensor
at::Tensor ger::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & vec2) {
    static auto op = create_ger_typed_handle();
    return op.redispatch(dispatchKeySet, self, vec2);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_vector_norm_out, name, "aten::linalg_vector_norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_vector_norm_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_vector_norm_out, schema_str, "linalg_vector_norm.out(Tensor self, Scalar ord=2, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)")

// aten::linalg_vector_norm.out(Tensor self, Scalar ord=2, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_vector_norm_out::schema> create_linalg_vector_norm_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_vector_norm_out::name, linalg_vector_norm_out::overload_name)
      .typed<linalg_vector_norm_out::schema>();
}

// aten::linalg_vector_norm.out(Tensor self, Scalar ord=2, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_vector_norm_out::call(const at::Tensor & self, const at::Scalar & ord, c10::optional<at::IntArrayRef> dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
    static auto op = create_linalg_vector_norm_out_typed_handle();
    return op.call(self, ord, dim, keepdim, dtype, out);
}

// aten::linalg_vector_norm.out(Tensor self, Scalar ord=2, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_vector_norm_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & ord, c10::optional<at::IntArrayRef> dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
    static auto op = create_linalg_vector_norm_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, ord, dim, keepdim, dtype, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_matrix_norm_str_ord, name, "aten::linalg_matrix_norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_matrix_norm_str_ord, overload_name, "str_ord")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_matrix_norm_str_ord, schema_str, "linalg_matrix_norm.str_ord(Tensor self, str ord='fro', int[] dim=[-2,-1], bool keepdim=False, *, ScalarType? dtype=None) -> Tensor")

// aten::linalg_matrix_norm.str_ord(Tensor self, str ord='fro', int[] dim=[-2,-1], bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<linalg_matrix_norm_str_ord::schema> create_linalg_matrix_norm_str_ord_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_matrix_norm_str_ord::name, linalg_matrix_norm_str_ord::overload_name)
      .typed<linalg_matrix_norm_str_ord::schema>();
}

// aten::linalg_matrix_norm.str_ord(Tensor self, str ord='fro', int[] dim=[-2,-1], bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
at::Tensor linalg_matrix_norm_str_ord::call(const at::Tensor & self, c10::string_view ord, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_linalg_matrix_norm_str_ord_typed_handle();
    return op.call(self, ord, dim, keepdim, dtype);
}

// aten::linalg_matrix_norm.str_ord(Tensor self, str ord='fro', int[] dim=[-2,-1], bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
at::Tensor linalg_matrix_norm_str_ord::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::string_view ord, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_linalg_matrix_norm_str_ord_typed_handle();
    return op.redispatch(dispatchKeySet, self, ord, dim, keepdim, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_svd_U, name, "aten::linalg_svd")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_svd_U, overload_name, "U")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_svd_U, schema_str, "linalg_svd.U(Tensor self, bool full_matrices=True, *, Tensor(a!) U, Tensor(b!) S, Tensor(c!) Vh) -> (Tensor(a!) U, Tensor(b!) S, Tensor(c!) Vh)")

// aten::linalg_svd.U(Tensor self, bool full_matrices=True, *, Tensor(a!) U, Tensor(b!) S, Tensor(c!) Vh) -> (Tensor(a!) U, Tensor(b!) S, Tensor(c!) Vh)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_svd_U::schema> create_linalg_svd_U_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_svd_U::name, linalg_svd_U::overload_name)
      .typed<linalg_svd_U::schema>();
}

// aten::linalg_svd.U(Tensor self, bool full_matrices=True, *, Tensor(a!) U, Tensor(b!) S, Tensor(c!) Vh) -> (Tensor(a!) U, Tensor(b!) S, Tensor(c!) Vh)
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> linalg_svd_U::call(const at::Tensor & self, bool full_matrices, at::Tensor & U, at::Tensor & S, at::Tensor & Vh) {
    static auto op = create_linalg_svd_U_typed_handle();
    return op.call(self, full_matrices, U, S, Vh);
}

// aten::linalg_svd.U(Tensor self, bool full_matrices=True, *, Tensor(a!) U, Tensor(b!) S, Tensor(c!) Vh) -> (Tensor(a!) U, Tensor(b!) S, Tensor(c!) Vh)
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> linalg_svd_U::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool full_matrices, at::Tensor & U, at::Tensor & S, at::Tensor & Vh) {
    static auto op = create_linalg_svd_U_typed_handle();
    return op.redispatch(dispatchKeySet, self, full_matrices, U, S, Vh);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_qr, name, "aten::linalg_qr")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_qr, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_qr, schema_str, "linalg_qr(Tensor self, str mode='reduced') -> (Tensor Q, Tensor R)")

// aten::linalg_qr(Tensor self, str mode='reduced') -> (Tensor Q, Tensor R)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_qr::schema> create_linalg_qr_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_qr::name, linalg_qr::overload_name)
      .typed<linalg_qr::schema>();
}

// aten::linalg_qr(Tensor self, str mode='reduced') -> (Tensor Q, Tensor R)
::std::tuple<at::Tensor,at::Tensor> linalg_qr::call(const at::Tensor & self, c10::string_view mode) {
    static auto op = create_linalg_qr_typed_handle();
    return op.call(self, mode);
}

// aten::linalg_qr(Tensor self, str mode='reduced') -> (Tensor Q, Tensor R)
::std::tuple<at::Tensor,at::Tensor> linalg_qr::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::string_view mode) {
    static auto op = create_linalg_qr_typed_handle();
    return op.redispatch(dispatchKeySet, self, mode);
}

}} // namespace at::_ops
