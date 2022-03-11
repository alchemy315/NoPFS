#include <ATen/Operators.h>
#include <ATen/Tensor.h>
#include <ATen/core/dispatch/Dispatcher.h>

// NOTE See [Sharded File] comment in VariableType

namespace at { namespace _ops {


STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cast_Short, name, "aten::_cast_Short")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cast_Short, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cast_Short, schema_str, "_cast_Short(Tensor self, bool non_blocking=False) -> Tensor")

// aten::_cast_Short(Tensor self, bool non_blocking=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_cast_Short::schema> create__cast_Short_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_cast_Short::name, _cast_Short::overload_name)
      .typed<_cast_Short::schema>();
}

// aten::_cast_Short(Tensor self, bool non_blocking=False) -> Tensor
at::Tensor _cast_Short::call(const at::Tensor & self, bool non_blocking) {
    static auto op = create__cast_Short_typed_handle();
    return op.call(self, non_blocking);
}

// aten::_cast_Short(Tensor self, bool non_blocking=False) -> Tensor
at::Tensor _cast_Short::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool non_blocking) {
    static auto op = create__cast_Short_typed_handle();
    return op.redispatch(dispatchKeySet, self, non_blocking);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(output_nr, name, "aten::output_nr")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(output_nr, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(output_nr, schema_str, "output_nr(Tensor self) -> int")

// aten::output_nr(Tensor self) -> int
static C10_NOINLINE c10::TypedOperatorHandle<output_nr::schema> create_output_nr_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(output_nr::name, output_nr::overload_name)
      .typed<output_nr::schema>();
}

// aten::output_nr(Tensor self) -> int
int64_t output_nr::call(const at::Tensor & self) {
    static auto op = create_output_nr_typed_handle();
    return op.call(self);
}

// aten::output_nr(Tensor self) -> int
int64_t output_nr::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_output_nr_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cudnn_init_dropout_state, name, "aten::_cudnn_init_dropout_state")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cudnn_init_dropout_state, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cudnn_init_dropout_state, schema_str, "_cudnn_init_dropout_state(float dropout, bool train, int dropout_seed, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor")

// aten::_cudnn_init_dropout_state(float dropout, bool train, int dropout_seed, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_cudnn_init_dropout_state::schema> create__cudnn_init_dropout_state_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_cudnn_init_dropout_state::name, _cudnn_init_dropout_state::overload_name)
      .typed<_cudnn_init_dropout_state::schema>();
}

// aten::_cudnn_init_dropout_state(float dropout, bool train, int dropout_seed, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
at::Tensor _cudnn_init_dropout_state::call(double dropout, bool train, int64_t dropout_seed, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create__cudnn_init_dropout_state_typed_handle();
    return op.call(dropout, train, dropout_seed, dtype, layout, device, pin_memory);
}

// aten::_cudnn_init_dropout_state(float dropout, bool train, int dropout_seed, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
at::Tensor _cudnn_init_dropout_state::redispatch(c10::DispatchKeySet dispatchKeySet, double dropout, bool train, int64_t dropout_seed, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create__cudnn_init_dropout_state_typed_handle();
    return op.redispatch(dispatchKeySet, dropout, train, dropout_seed, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sobol_engine_ff_, name, "aten::_sobol_engine_ff_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sobol_engine_ff_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sobol_engine_ff_, schema_str, "_sobol_engine_ff_(Tensor(a!) self, int n, Tensor sobolstate, int dimension, int num_generated) -> Tensor(a!)")

// aten::_sobol_engine_ff_(Tensor(a!) self, int n, Tensor sobolstate, int dimension, int num_generated) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<_sobol_engine_ff_::schema> create__sobol_engine_ff__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_sobol_engine_ff_::name, _sobol_engine_ff_::overload_name)
      .typed<_sobol_engine_ff_::schema>();
}

// aten::_sobol_engine_ff_(Tensor(a!) self, int n, Tensor sobolstate, int dimension, int num_generated) -> Tensor(a!)
at::Tensor & _sobol_engine_ff_::call(at::Tensor & self, int64_t n, const at::Tensor & sobolstate, int64_t dimension, int64_t num_generated) {
    static auto op = create__sobol_engine_ff__typed_handle();
    return op.call(self, n, sobolstate, dimension, num_generated);
}

// aten::_sobol_engine_ff_(Tensor(a!) self, int n, Tensor sobolstate, int dimension, int num_generated) -> Tensor(a!)
at::Tensor & _sobol_engine_ff_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t n, const at::Tensor & sobolstate, int64_t dimension, int64_t num_generated) {
    static auto op = create__sobol_engine_ff__typed_handle();
    return op.redispatch(dispatchKeySet, self, n, sobolstate, dimension, num_generated);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(abs_, name, "aten::abs_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(abs_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(abs_, schema_str, "abs_(Tensor(a!) self) -> Tensor(a!)")

// aten::abs_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<abs_::schema> create_abs__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(abs_::name, abs_::overload_name)
      .typed<abs_::schema>();
}

// aten::abs_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & abs_::call(at::Tensor & self) {
    static auto op = create_abs__typed_handle();
    return op.call(self);
}

// aten::abs_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & abs_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_abs__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(absolute, name, "aten::absolute")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(absolute, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(absolute, schema_str, "absolute(Tensor self) -> Tensor")

// aten::absolute(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<absolute::schema> create_absolute_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(absolute::name, absolute::overload_name)
      .typed<absolute::schema>();
}

// aten::absolute(Tensor self) -> Tensor
at::Tensor absolute::call(const at::Tensor & self) {
    static auto op = create_absolute_typed_handle();
    return op.call(self);
}

// aten::absolute(Tensor self) -> Tensor
at::Tensor absolute::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_absolute_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(angle, name, "aten::angle")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(angle, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(angle, schema_str, "angle(Tensor self) -> Tensor")

// aten::angle(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<angle::schema> create_angle_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(angle::name, angle::overload_name)
      .typed<angle::schema>();
}

// aten::angle(Tensor self) -> Tensor
at::Tensor angle::call(const at::Tensor & self) {
    static auto op = create_angle_typed_handle();
    return op.call(self);
}

// aten::angle(Tensor self) -> Tensor
at::Tensor angle::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_angle_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sgn_, name, "aten::sgn_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sgn_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sgn_, schema_str, "sgn_(Tensor(a!) self) -> Tensor(a!)")

// aten::sgn_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<sgn_::schema> create_sgn__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sgn_::name, sgn_::overload_name)
      .typed<sgn_::schema>();
}

// aten::sgn_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & sgn_::call(at::Tensor & self) {
    static auto op = create_sgn__typed_handle();
    return op.call(self);
}

// aten::sgn_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & sgn_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_sgn__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sgn_out, name, "aten::sgn")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sgn_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sgn_out, schema_str, "sgn.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::sgn.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<sgn_out::schema> create_sgn_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sgn_out::name, sgn_out::overload_name)
      .typed<sgn_out::schema>();
}

// aten::sgn.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & sgn_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_sgn_out_typed_handle();
    return op.call(self, out);
}

// aten::sgn.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & sgn_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_sgn_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(acos_, name, "aten::acos_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(acos_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(acos_, schema_str, "acos_(Tensor(a!) self) -> Tensor(a!)")

// aten::acos_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<acos_::schema> create_acos__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(acos_::name, acos_::overload_name)
      .typed<acos_::schema>();
}

// aten::acos_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & acos_::call(at::Tensor & self) {
    static auto op = create_acos__typed_handle();
    return op.call(self);
}

// aten::acos_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & acos_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_acos__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_add_relu__Tensor, name, "aten::_add_relu_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_add_relu__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_add_relu__Tensor, schema_str, "_add_relu_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)")

// aten::_add_relu_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<_add_relu__Tensor::schema> create__add_relu__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_add_relu__Tensor::name, _add_relu__Tensor::overload_name)
      .typed<_add_relu__Tensor::schema>();
}

// aten::_add_relu_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
at::Tensor & _add_relu__Tensor::call(at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
    static auto op = create__add_relu__Tensor_typed_handle();
    return op.call(self, other, alpha);
}

// aten::_add_relu_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
at::Tensor & _add_relu__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
    static auto op = create__add_relu__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(add__Scalar, name, "aten::add_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(add__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(add__Scalar, schema_str, "add_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)")

// aten::add_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<add__Scalar::schema> create_add__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(add__Scalar::name, add__Scalar::overload_name)
      .typed<add__Scalar::schema>();
}

// aten::add_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
at::Tensor & add__Scalar::call(at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
    static auto op = create_add__Scalar_typed_handle();
    return op.call(self, other, alpha);
}

// aten::add_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
at::Tensor & add__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
    static auto op = create_add__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addmv_out, name, "aten::addmv")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addmv_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addmv_out, schema_str, "addmv.out(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)")

// aten::addmv.out(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<addmv_out::schema> create_addmv_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(addmv_out::name, addmv_out::overload_name)
      .typed<addmv_out::schema>();
}

// aten::addmv.out(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
at::Tensor & addmv_out::call(const at::Tensor & self, const at::Tensor & mat, const at::Tensor & vec, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
    static auto op = create_addmv_out_typed_handle();
    return op.call(self, mat, vec, beta, alpha, out);
}

// aten::addmv.out(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
at::Tensor & addmv_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mat, const at::Tensor & vec, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
    static auto op = create_addmv_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, mat, vec, beta, alpha, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arange_out, name, "aten::arange")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arange_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arange_out, schema_str, "arange.out(Scalar end, *, Tensor(a!) out) -> Tensor(a!)")

// aten::arange.out(Scalar end, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<arange_out::schema> create_arange_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(arange_out::name, arange_out::overload_name)
      .typed<arange_out::schema>();
}

// aten::arange.out(Scalar end, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & arange_out::call(const at::Scalar & end, at::Tensor & out) {
    static auto op = create_arange_out_typed_handle();
    return op.call(end, out);
}

// aten::arange.out(Scalar end, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & arange_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Scalar & end, at::Tensor & out) {
    static auto op = create_arange_out_typed_handle();
    return op.redispatch(dispatchKeySet, end, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(argmax_out, name, "aten::argmax")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(argmax_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(argmax_out, schema_str, "argmax.out(Tensor self, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)")

// aten::argmax.out(Tensor self, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<argmax_out::schema> create_argmax_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(argmax_out::name, argmax_out::overload_name)
      .typed<argmax_out::schema>();
}

// aten::argmax.out(Tensor self, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & argmax_out::call(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out) {
    static auto op = create_argmax_out_typed_handle();
    return op.call(self, dim, keepdim, out);
}

// aten::argmax.out(Tensor self, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & argmax_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out) {
    static auto op = create_argmax_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(acosh_, name, "aten::acosh_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(acosh_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(acosh_, schema_str, "acosh_(Tensor(a!) self) -> Tensor(a!)")

// aten::acosh_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<acosh_::schema> create_acosh__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(acosh_::name, acosh_::overload_name)
      .typed<acosh_::schema>();
}

// aten::acosh_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & acosh_::call(at::Tensor & self) {
    static auto op = create_acosh__typed_handle();
    return op.call(self);
}

// aten::acosh_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & acosh_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_acosh__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(asinh, name, "aten::asinh")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(asinh, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(asinh, schema_str, "asinh(Tensor self) -> Tensor")

// aten::asinh(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<asinh::schema> create_asinh_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(asinh::name, asinh::overload_name)
      .typed<asinh::schema>();
}

// aten::asinh(Tensor self) -> Tensor
at::Tensor asinh::call(const at::Tensor & self) {
    static auto op = create_asinh_typed_handle();
    return op.call(self);
}

// aten::asinh(Tensor self) -> Tensor
at::Tensor asinh::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_asinh_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(asinh_, name, "aten::asinh_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(asinh_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(asinh_, schema_str, "asinh_(Tensor(a!) self) -> Tensor(a!)")

// aten::asinh_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<asinh_::schema> create_asinh__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(asinh_::name, asinh_::overload_name)
      .typed<asinh_::schema>();
}

// aten::asinh_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & asinh_::call(at::Tensor & self) {
    static auto op = create_asinh__typed_handle();
    return op.call(self);
}

// aten::asinh_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & asinh_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_asinh__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(as_strided_, name, "aten::as_strided_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(as_strided_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(as_strided_, schema_str, "as_strided_(Tensor(a!) self, int[] size, int[] stride, int? storage_offset=None) -> Tensor(a!)")

// aten::as_strided_(Tensor(a!) self, int[] size, int[] stride, int? storage_offset=None) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<as_strided_::schema> create_as_strided__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(as_strided_::name, as_strided_::overload_name)
      .typed<as_strided_::schema>();
}

// aten::as_strided_(Tensor(a!) self, int[] size, int[] stride, int? storage_offset=None) -> Tensor(a!)
const at::Tensor & as_strided_::call(const at::Tensor & self, at::IntArrayRef size, at::IntArrayRef stride, c10::optional<int64_t> storage_offset) {
    static auto op = create_as_strided__typed_handle();
    return op.call(self, size, stride, storage_offset);
}

// aten::as_strided_(Tensor(a!) self, int[] size, int[] stride, int? storage_offset=None) -> Tensor(a!)
const at::Tensor & as_strided_::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef size, at::IntArrayRef stride, c10::optional<int64_t> storage_offset) {
    static auto op = create_as_strided__typed_handle();
    return op.redispatch(dispatchKeySet, self, size, stride, storage_offset);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atan_, name, "aten::atan_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atan_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atan_, schema_str, "atan_(Tensor(a!) self) -> Tensor(a!)")

// aten::atan_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<atan_::schema> create_atan__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(atan_::name, atan_::overload_name)
      .typed<atan_::schema>();
}

// aten::atan_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & atan_::call(at::Tensor & self) {
    static auto op = create_atan__typed_handle();
    return op.call(self);
}

// aten::atan_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & atan_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_atan__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atleast_2d, name, "aten::atleast_2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atleast_2d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atleast_2d, schema_str, "atleast_2d(Tensor self) -> Tensor")

// aten::atleast_2d(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<atleast_2d::schema> create_atleast_2d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(atleast_2d::name, atleast_2d::overload_name)
      .typed<atleast_2d::schema>();
}

// aten::atleast_2d(Tensor self) -> Tensor
at::Tensor atleast_2d::call(const at::Tensor & self) {
    static auto op = create_atleast_2d_typed_handle();
    return op.call(self);
}

// aten::atleast_2d(Tensor self) -> Tensor
at::Tensor atleast_2d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_atleast_2d_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(baddbmm, name, "aten::baddbmm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(baddbmm, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(baddbmm, schema_str, "baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor")

// aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<baddbmm::schema> create_baddbmm_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(baddbmm::name, baddbmm::overload_name)
      .typed<baddbmm::schema>();
}

// aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
at::Tensor baddbmm::call(const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha) {
    static auto op = create_baddbmm_typed_handle();
    return op.call(self, batch1, batch2, beta, alpha);
}

// aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
at::Tensor baddbmm::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha) {
    static auto op = create_baddbmm_typed_handle();
    return op.redispatch(dispatchKeySet, self, batch1, batch2, beta, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(baddbmm_, name, "aten::baddbmm_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(baddbmm_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(baddbmm_, schema_str, "baddbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)")

// aten::baddbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<baddbmm_::schema> create_baddbmm__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(baddbmm_::name, baddbmm_::overload_name)
      .typed<baddbmm_::schema>();
}

// aten::baddbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
at::Tensor & baddbmm_::call(at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha) {
    static auto op = create_baddbmm__typed_handle();
    return op.call(self, batch1, batch2, beta, alpha);
}

// aten::baddbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
at::Tensor & baddbmm_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha) {
    static auto op = create_baddbmm__typed_handle();
    return op.redispatch(dispatchKeySet, self, batch1, batch2, beta, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(batch_norm, name, "aten::batch_norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(batch_norm, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(batch_norm, schema_str, "batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor")

// aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<batch_norm::schema> create_batch_norm_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(batch_norm::name, batch_norm::overload_name)
      .typed<batch_norm::schema>();
}

// aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor
at::Tensor batch_norm::call(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double momentum, double eps, bool cudnn_enabled) {
    static auto op = create_batch_norm_typed_handle();
    return op.call(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
}

// aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor
at::Tensor batch_norm::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double momentum, double eps, bool cudnn_enabled) {
    static auto op = create_batch_norm_typed_handle();
    return op.redispatch(dispatchKeySet, input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bernoulli, name, "aten::bernoulli")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bernoulli, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bernoulli, schema_str, "bernoulli(Tensor self, *, Generator? generator=None) -> Tensor")

// aten::bernoulli(Tensor self, *, Generator? generator=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<bernoulli::schema> create_bernoulli_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bernoulli::name, bernoulli::overload_name)
      .typed<bernoulli::schema>();
}

// aten::bernoulli(Tensor self, *, Generator? generator=None) -> Tensor
at::Tensor bernoulli::call(const at::Tensor & self, c10::optional<at::Generator> generator) {
    static auto op = create_bernoulli_typed_handle();
    return op.call(self, generator);
}

// aten::bernoulli(Tensor self, *, Generator? generator=None) -> Tensor
at::Tensor bernoulli::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::Generator> generator) {
    static auto op = create_bernoulli_typed_handle();
    return op.redispatch(dispatchKeySet, self, generator);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bernoulli__Tensor, name, "aten::bernoulli_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bernoulli__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bernoulli__Tensor, schema_str, "bernoulli_.Tensor(Tensor(a!) self, Tensor p, *, Generator? generator=None) -> Tensor(a!)")

// aten::bernoulli_.Tensor(Tensor(a!) self, Tensor p, *, Generator? generator=None) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<bernoulli__Tensor::schema> create_bernoulli__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bernoulli__Tensor::name, bernoulli__Tensor::overload_name)
      .typed<bernoulli__Tensor::schema>();
}

// aten::bernoulli_.Tensor(Tensor(a!) self, Tensor p, *, Generator? generator=None) -> Tensor(a!)
at::Tensor & bernoulli__Tensor::call(at::Tensor & self, const at::Tensor & p, c10::optional<at::Generator> generator) {
    static auto op = create_bernoulli__Tensor_typed_handle();
    return op.call(self, p, generator);
}

// aten::bernoulli_.Tensor(Tensor(a!) self, Tensor p, *, Generator? generator=None) -> Tensor(a!)
at::Tensor & bernoulli__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & p, c10::optional<at::Generator> generator) {
    static auto op = create_bernoulli__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, p, generator);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bilinear, name, "aten::bilinear")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bilinear, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bilinear, schema_str, "bilinear(Tensor input1, Tensor input2, Tensor weight, Tensor? bias) -> Tensor")

// aten::bilinear(Tensor input1, Tensor input2, Tensor weight, Tensor? bias) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<bilinear::schema> create_bilinear_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bilinear::name, bilinear::overload_name)
      .typed<bilinear::schema>();
}

// aten::bilinear(Tensor input1, Tensor input2, Tensor weight, Tensor? bias) -> Tensor
at::Tensor bilinear::call(const at::Tensor & input1, const at::Tensor & input2, const at::Tensor & weight, const c10::optional<at::Tensor> & bias) {
    static auto op = create_bilinear_typed_handle();
    return op.call(input1, input2, weight, bias);
}

// aten::bilinear(Tensor input1, Tensor input2, Tensor weight, Tensor? bias) -> Tensor
at::Tensor bilinear::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input1, const at::Tensor & input2, const at::Tensor & weight, const c10::optional<at::Tensor> & bias) {
    static auto op = create_bilinear_typed_handle();
    return op.redispatch(dispatchKeySet, input1, input2, weight, bias);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(binary_cross_entropy_with_logits, name, "aten::binary_cross_entropy_with_logits")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(binary_cross_entropy_with_logits, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(binary_cross_entropy_with_logits, schema_str, "binary_cross_entropy_with_logits(Tensor self, Tensor target, Tensor? weight=None, Tensor? pos_weight=None, int reduction=Mean) -> Tensor")

// aten::binary_cross_entropy_with_logits(Tensor self, Tensor target, Tensor? weight=None, Tensor? pos_weight=None, int reduction=Mean) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<binary_cross_entropy_with_logits::schema> create_binary_cross_entropy_with_logits_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(binary_cross_entropy_with_logits::name, binary_cross_entropy_with_logits::overload_name)
      .typed<binary_cross_entropy_with_logits::schema>();
}

// aten::binary_cross_entropy_with_logits(Tensor self, Tensor target, Tensor? weight=None, Tensor? pos_weight=None, int reduction=Mean) -> Tensor
at::Tensor binary_cross_entropy_with_logits::call(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & pos_weight, int64_t reduction) {
    static auto op = create_binary_cross_entropy_with_logits_typed_handle();
    return op.call(self, target, weight, pos_weight, reduction);
}

// aten::binary_cross_entropy_with_logits(Tensor self, Tensor target, Tensor? weight=None, Tensor? pos_weight=None, int reduction=Mean) -> Tensor
at::Tensor binary_cross_entropy_with_logits::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & pos_weight, int64_t reduction) {
    static auto op = create_binary_cross_entropy_with_logits_typed_handle();
    return op.redispatch(dispatchKeySet, self, target, weight, pos_weight, reduction);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bincount, name, "aten::bincount")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bincount, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bincount, schema_str, "bincount(Tensor self, Tensor? weights=None, int minlength=0) -> Tensor")

// aten::bincount(Tensor self, Tensor? weights=None, int minlength=0) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<bincount::schema> create_bincount_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bincount::name, bincount::overload_name)
      .typed<bincount::schema>();
}

// aten::bincount(Tensor self, Tensor? weights=None, int minlength=0) -> Tensor
at::Tensor bincount::call(const at::Tensor & self, const c10::optional<at::Tensor> & weights, int64_t minlength) {
    static auto op = create_bincount_typed_handle();
    return op.call(self, weights, minlength);
}

// aten::bincount(Tensor self, Tensor? weights=None, int minlength=0) -> Tensor
at::Tensor bincount::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Tensor> & weights, int64_t minlength) {
    static auto op = create_bincount_typed_handle();
    return op.redispatch(dispatchKeySet, self, weights, minlength);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logical_and, name, "aten::logical_and")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logical_and, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logical_and, schema_str, "logical_and(Tensor self, Tensor other) -> Tensor")

// aten::logical_and(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<logical_and::schema> create_logical_and_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(logical_and::name, logical_and::overload_name)
      .typed<logical_and::schema>();
}

// aten::logical_and(Tensor self, Tensor other) -> Tensor
at::Tensor logical_and::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_logical_and_typed_handle();
    return op.call(self, other);
}

// aten::logical_and(Tensor self, Tensor other) -> Tensor
at::Tensor logical_and::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_logical_and_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logical_or_out, name, "aten::logical_or")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logical_or_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logical_or_out, schema_str, "logical_or.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::logical_or.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<logical_or_out::schema> create_logical_or_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(logical_or_out::name, logical_or_out::overload_name)
      .typed<logical_or_out::schema>();
}

// aten::logical_or.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & logical_or_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_logical_or_out_typed_handle();
    return op.call(self, other, out);
}

// aten::logical_or.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & logical_or_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_logical_or_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cat_names_out, name, "aten::cat")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cat_names_out, overload_name, "names_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cat_names_out, schema_str, "cat.names_out(Tensor[] tensors, Dimname dim, *, Tensor(a!) out) -> Tensor(a!)")

// aten::cat.names_out(Tensor[] tensors, Dimname dim, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<cat_names_out::schema> create_cat_names_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cat_names_out::name, cat_names_out::overload_name)
      .typed<cat_names_out::schema>();
}

// aten::cat.names_out(Tensor[] tensors, Dimname dim, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & cat_names_out::call(at::TensorList tensors, at::Dimname dim, at::Tensor & out) {
    static auto op = create_cat_names_out_typed_handle();
    return op.call(tensors, dim, out);
}

// aten::cat.names_out(Tensor[] tensors, Dimname dim, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & cat_names_out::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, at::Dimname dim, at::Tensor & out) {
    static auto op = create_cat_names_out_typed_handle();
    return op.redispatch(dispatchKeySet, tensors, dim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(block_diag, name, "aten::block_diag")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(block_diag, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(block_diag, schema_str, "block_diag(Tensor[] tensors) -> Tensor")

// aten::block_diag(Tensor[] tensors) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<block_diag::schema> create_block_diag_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(block_diag::name, block_diag::overload_name)
      .typed<block_diag::schema>();
}

// aten::block_diag(Tensor[] tensors) -> Tensor
at::Tensor block_diag::call(at::TensorList tensors) {
    static auto op = create_block_diag_typed_handle();
    return op.call(tensors);
}

// aten::block_diag(Tensor[] tensors) -> Tensor
at::Tensor block_diag::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create_block_diag_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unsafe_chunk, name, "aten::unsafe_chunk")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unsafe_chunk, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unsafe_chunk, schema_str, "unsafe_chunk(Tensor self, int chunks, int dim=0) -> Tensor[]")

// aten::unsafe_chunk(Tensor self, int chunks, int dim=0) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<unsafe_chunk::schema> create_unsafe_chunk_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(unsafe_chunk::name, unsafe_chunk::overload_name)
      .typed<unsafe_chunk::schema>();
}

// aten::unsafe_chunk(Tensor self, int chunks, int dim=0) -> Tensor[]
::std::vector<at::Tensor> unsafe_chunk::call(const at::Tensor & self, int64_t chunks, int64_t dim) {
    static auto op = create_unsafe_chunk_typed_handle();
    return op.call(self, chunks, dim);
}

// aten::unsafe_chunk(Tensor self, int chunks, int dim=0) -> Tensor[]
::std::vector<at::Tensor> unsafe_chunk::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t chunks, int64_t dim) {
    static auto op = create_unsafe_chunk_typed_handle();
    return op.redispatch(dispatchKeySet, self, chunks, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(chunk, name, "aten::chunk")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(chunk, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(chunk, schema_str, "chunk(Tensor(a) self, int chunks, int dim=0) -> Tensor(a)[]")

// aten::chunk(Tensor(a) self, int chunks, int dim=0) -> Tensor(a)[]
static C10_NOINLINE c10::TypedOperatorHandle<chunk::schema> create_chunk_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(chunk::name, chunk::overload_name)
      .typed<chunk::schema>();
}

// aten::chunk(Tensor(a) self, int chunks, int dim=0) -> Tensor(a)[]
::std::vector<at::Tensor> chunk::call(const at::Tensor & self, int64_t chunks, int64_t dim) {
    static auto op = create_chunk_typed_handle();
    return op.call(self, chunks, dim);
}

// aten::chunk(Tensor(a) self, int chunks, int dim=0) -> Tensor(a)[]
::std::vector<at::Tensor> chunk::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t chunks, int64_t dim) {
    static auto op = create_chunk_typed_handle();
    return op.redispatch(dispatchKeySet, self, chunks, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tensor_split_sections, name, "aten::tensor_split")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tensor_split_sections, overload_name, "sections")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tensor_split_sections, schema_str, "tensor_split.sections(Tensor(a) self, int sections, int dim=0) -> Tensor(a)[]")

// aten::tensor_split.sections(Tensor(a) self, int sections, int dim=0) -> Tensor(a)[]
static C10_NOINLINE c10::TypedOperatorHandle<tensor_split_sections::schema> create_tensor_split_sections_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(tensor_split_sections::name, tensor_split_sections::overload_name)
      .typed<tensor_split_sections::schema>();
}

// aten::tensor_split.sections(Tensor(a) self, int sections, int dim=0) -> Tensor(a)[]
::std::vector<at::Tensor> tensor_split_sections::call(const at::Tensor & self, int64_t sections, int64_t dim) {
    static auto op = create_tensor_split_sections_typed_handle();
    return op.call(self, sections, dim);
}

// aten::tensor_split.sections(Tensor(a) self, int sections, int dim=0) -> Tensor(a)[]
::std::vector<at::Tensor> tensor_split_sections::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t sections, int64_t dim) {
    static auto op = create_tensor_split_sections_typed_handle();
    return op.redispatch(dispatchKeySet, self, sections, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp, name, "aten::clamp")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp, schema_str, "clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor")

// aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<clamp::schema> create_clamp_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(clamp::name, clamp::overload_name)
      .typed<clamp::schema>();
}

// aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
at::Tensor clamp::call(const at::Tensor & self, const c10::optional<at::Scalar> & min, const c10::optional<at::Scalar> & max) {
    static auto op = create_clamp_typed_handle();
    return op.call(self, min, max);
}

// aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
at::Tensor clamp::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Scalar> & min, const c10::optional<at::Scalar> & max) {
    static auto op = create_clamp_typed_handle();
    return op.redispatch(dispatchKeySet, self, min, max);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_max, name, "aten::clamp_max")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_max, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_max, schema_str, "clamp_max(Tensor self, Scalar max) -> Tensor")

// aten::clamp_max(Tensor self, Scalar max) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<clamp_max::schema> create_clamp_max_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(clamp_max::name, clamp_max::overload_name)
      .typed<clamp_max::schema>();
}

// aten::clamp_max(Tensor self, Scalar max) -> Tensor
at::Tensor clamp_max::call(const at::Tensor & self, const at::Scalar & max) {
    static auto op = create_clamp_max_typed_handle();
    return op.call(self, max);
}

// aten::clamp_max(Tensor self, Scalar max) -> Tensor
at::Tensor clamp_max::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & max) {
    static auto op = create_clamp_max_typed_handle();
    return op.redispatch(dispatchKeySet, self, max);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_max_Tensor, name, "aten::clamp_max")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_max_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_max_Tensor, schema_str, "clamp_max.Tensor(Tensor self, Tensor max) -> Tensor")

// aten::clamp_max.Tensor(Tensor self, Tensor max) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<clamp_max_Tensor::schema> create_clamp_max_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(clamp_max_Tensor::name, clamp_max_Tensor::overload_name)
      .typed<clamp_max_Tensor::schema>();
}

// aten::clamp_max.Tensor(Tensor self, Tensor max) -> Tensor
at::Tensor clamp_max_Tensor::call(const at::Tensor & self, const at::Tensor & max) {
    static auto op = create_clamp_max_Tensor_typed_handle();
    return op.call(self, max);
}

// aten::clamp_max.Tensor(Tensor self, Tensor max) -> Tensor
at::Tensor clamp_max_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & max) {
    static auto op = create_clamp_max_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, max);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_max_, name, "aten::clamp_max_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_max_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_max_, schema_str, "clamp_max_(Tensor(a!) self, Scalar max) -> Tensor(a!)")

// aten::clamp_max_(Tensor(a!) self, Scalar max) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<clamp_max_::schema> create_clamp_max__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(clamp_max_::name, clamp_max_::overload_name)
      .typed<clamp_max_::schema>();
}

// aten::clamp_max_(Tensor(a!) self, Scalar max) -> Tensor(a!)
at::Tensor & clamp_max_::call(at::Tensor & self, const at::Scalar & max) {
    static auto op = create_clamp_max__typed_handle();
    return op.call(self, max);
}

// aten::clamp_max_(Tensor(a!) self, Scalar max) -> Tensor(a!)
at::Tensor & clamp_max_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & max) {
    static auto op = create_clamp_max__typed_handle();
    return op.redispatch(dispatchKeySet, self, max);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_max__Tensor, name, "aten::clamp_max_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_max__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_max__Tensor, schema_str, "clamp_max_.Tensor(Tensor(a!) self, Tensor max) -> Tensor(a!)")

// aten::clamp_max_.Tensor(Tensor(a!) self, Tensor max) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<clamp_max__Tensor::schema> create_clamp_max__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(clamp_max__Tensor::name, clamp_max__Tensor::overload_name)
      .typed<clamp_max__Tensor::schema>();
}

// aten::clamp_max_.Tensor(Tensor(a!) self, Tensor max) -> Tensor(a!)
at::Tensor & clamp_max__Tensor::call(at::Tensor & self, const at::Tensor & max) {
    static auto op = create_clamp_max__Tensor_typed_handle();
    return op.call(self, max);
}

// aten::clamp_max_.Tensor(Tensor(a!) self, Tensor max) -> Tensor(a!)
at::Tensor & clamp_max__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & max) {
    static auto op = create_clamp_max__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, max);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_max_out, name, "aten::clamp_max")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_max_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_max_out, schema_str, "clamp_max.out(Tensor self, Scalar max, *, Tensor(a!) out) -> Tensor(a!)")

// aten::clamp_max.out(Tensor self, Scalar max, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<clamp_max_out::schema> create_clamp_max_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(clamp_max_out::name, clamp_max_out::overload_name)
      .typed<clamp_max_out::schema>();
}

// aten::clamp_max.out(Tensor self, Scalar max, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & clamp_max_out::call(const at::Tensor & self, const at::Scalar & max, at::Tensor & out) {
    static auto op = create_clamp_max_out_typed_handle();
    return op.call(self, max, out);
}

// aten::clamp_max.out(Tensor self, Scalar max, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & clamp_max_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & max, at::Tensor & out) {
    static auto op = create_clamp_max_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, max, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_min_Tensor_out, name, "aten::clamp_min")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_min_Tensor_out, overload_name, "Tensor_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_min_Tensor_out, schema_str, "clamp_min.Tensor_out(Tensor self, Tensor min, *, Tensor(a!) out) -> Tensor(a!)")

// aten::clamp_min.Tensor_out(Tensor self, Tensor min, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<clamp_min_Tensor_out::schema> create_clamp_min_Tensor_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(clamp_min_Tensor_out::name, clamp_min_Tensor_out::overload_name)
      .typed<clamp_min_Tensor_out::schema>();
}

// aten::clamp_min.Tensor_out(Tensor self, Tensor min, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & clamp_min_Tensor_out::call(const at::Tensor & self, const at::Tensor & min, at::Tensor & out) {
    static auto op = create_clamp_min_Tensor_out_typed_handle();
    return op.call(self, min, out);
}

// aten::clamp_min.Tensor_out(Tensor self, Tensor min, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & clamp_min_Tensor_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & min, at::Tensor & out) {
    static auto op = create_clamp_min_Tensor_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, min, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clip, name, "aten::clip")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clip, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clip, schema_str, "clip(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor")

// aten::clip(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<clip::schema> create_clip_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(clip::name, clip::overload_name)
      .typed<clip::schema>();
}

// aten::clip(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
at::Tensor clip::call(const at::Tensor & self, const c10::optional<at::Scalar> & min, const c10::optional<at::Scalar> & max) {
    static auto op = create_clip_typed_handle();
    return op.call(self, min, max);
}

// aten::clip(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
at::Tensor clip::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Scalar> & min, const c10::optional<at::Scalar> & max) {
    static auto op = create_clip_typed_handle();
    return op.redispatch(dispatchKeySet, self, min, max);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_is_acceptable, name, "aten::cudnn_is_acceptable")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_is_acceptable, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_is_acceptable, schema_str, "cudnn_is_acceptable(Tensor self) -> bool")

// aten::cudnn_is_acceptable(Tensor self) -> bool
static C10_NOINLINE c10::TypedOperatorHandle<cudnn_is_acceptable::schema> create_cudnn_is_acceptable_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cudnn_is_acceptable::name, cudnn_is_acceptable::overload_name)
      .typed<cudnn_is_acceptable::schema>();
}

// aten::cudnn_is_acceptable(Tensor self) -> bool
bool cudnn_is_acceptable::call(const at::Tensor & self) {
    static auto op = create_cudnn_is_acceptable_typed_handle();
    return op.call(self);
}

// aten::cudnn_is_acceptable(Tensor self) -> bool
bool cudnn_is_acceptable::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_cudnn_is_acceptable_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(complex, name, "aten::complex")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(complex, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(complex, schema_str, "complex(Tensor real, Tensor imag) -> Tensor")

// aten::complex(Tensor real, Tensor imag) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<complex::schema> create_complex_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(complex::name, complex::overload_name)
      .typed<complex::schema>();
}

// aten::complex(Tensor real, Tensor imag) -> Tensor
at::Tensor complex::call(const at::Tensor & real, const at::Tensor & imag) {
    static auto op = create_complex_typed_handle();
    return op.call(real, imag);
}

// aten::complex(Tensor real, Tensor imag) -> Tensor
at::Tensor complex::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & real, const at::Tensor & imag) {
    static auto op = create_complex_typed_handle();
    return op.redispatch(dispatchKeySet, real, imag);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(complex_out, name, "aten::complex")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(complex_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(complex_out, schema_str, "complex.out(Tensor real, Tensor imag, *, Tensor(a!) out) -> Tensor(a!)")

// aten::complex.out(Tensor real, Tensor imag, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<complex_out::schema> create_complex_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(complex_out::name, complex_out::overload_name)
      .typed<complex_out::schema>();
}

// aten::complex.out(Tensor real, Tensor imag, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & complex_out::call(const at::Tensor & real, const at::Tensor & imag, at::Tensor & out) {
    static auto op = create_complex_out_typed_handle();
    return op.call(real, imag, out);
}

// aten::complex.out(Tensor real, Tensor imag, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & complex_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & real, const at::Tensor & imag, at::Tensor & out) {
    static auto op = create_complex_out_typed_handle();
    return op.redispatch(dispatchKeySet, real, imag, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(polar, name, "aten::polar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(polar, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(polar, schema_str, "polar(Tensor abs, Tensor angle) -> Tensor")

// aten::polar(Tensor abs, Tensor angle) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<polar::schema> create_polar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(polar::name, polar::overload_name)
      .typed<polar::schema>();
}

// aten::polar(Tensor abs, Tensor angle) -> Tensor
at::Tensor polar::call(const at::Tensor & abs, const at::Tensor & angle) {
    static auto op = create_polar_typed_handle();
    return op.call(abs, angle);
}

// aten::polar(Tensor abs, Tensor angle) -> Tensor
at::Tensor polar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & abs, const at::Tensor & angle) {
    static auto op = create_polar_typed_handle();
    return op.redispatch(dispatchKeySet, abs, angle);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv2d_padding, name, "aten::conv2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv2d_padding, overload_name, "padding")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv2d_padding, schema_str, "conv2d.padding(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, str padding=\"valid\", int[2] dilation=1, int groups=1) -> Tensor")

// aten::conv2d.padding(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, str padding="valid", int[2] dilation=1, int groups=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<conv2d_padding::schema> create_conv2d_padding_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(conv2d_padding::name, conv2d_padding::overload_name)
      .typed<conv2d_padding::schema>();
}

// aten::conv2d.padding(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, str padding="valid", int[2] dilation=1, int groups=1) -> Tensor
at::Tensor conv2d_padding::call(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, c10::string_view padding, at::IntArrayRef dilation, int64_t groups) {
    static auto op = create_conv2d_padding_typed_handle();
    return op.call(input, weight, bias, stride, padding, dilation, groups);
}

// aten::conv2d.padding(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, str padding="valid", int[2] dilation=1, int groups=1) -> Tensor
at::Tensor conv2d_padding::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, c10::string_view padding, at::IntArrayRef dilation, int64_t groups) {
    static auto op = create_conv2d_padding_typed_handle();
    return op.redispatch(dispatchKeySet, input, weight, bias, stride, padding, dilation, groups);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv_transpose3d_input, name, "aten::conv_transpose3d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv_transpose3d_input, overload_name, "input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv_transpose3d_input, schema_str, "conv_transpose3d.input(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] output_padding=0, int groups=1, int[3] dilation=1) -> Tensor")

// aten::conv_transpose3d.input(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] output_padding=0, int groups=1, int[3] dilation=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<conv_transpose3d_input::schema> create_conv_transpose3d_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(conv_transpose3d_input::name, conv_transpose3d_input::overload_name)
      .typed<conv_transpose3d_input::schema>();
}

// aten::conv_transpose3d.input(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] output_padding=0, int groups=1, int[3] dilation=1) -> Tensor
at::Tensor conv_transpose3d_input::call(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, int64_t groups, at::IntArrayRef dilation) {
    static auto op = create_conv_transpose3d_input_typed_handle();
    return op.call(input, weight, bias, stride, padding, output_padding, groups, dilation);
}

// aten::conv_transpose3d.input(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] output_padding=0, int groups=1, int[3] dilation=1) -> Tensor
at::Tensor conv_transpose3d_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, int64_t groups, at::IntArrayRef dilation) {
    static auto op = create_conv_transpose3d_input_typed_handle();
    return op.redispatch(dispatchKeySet, input, weight, bias, stride, padding, output_padding, groups, dilation);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cos_, name, "aten::cos_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cos_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cos_, schema_str, "cos_(Tensor(a!) self) -> Tensor(a!)")

// aten::cos_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<cos_::schema> create_cos__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cos_::name, cos_::overload_name)
      .typed<cos_::schema>();
}

// aten::cos_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & cos_::call(at::Tensor & self) {
    static auto op = create_cos__typed_handle();
    return op.call(self);
}

// aten::cos_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & cos_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_cos__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(count_nonzero, name, "aten::count_nonzero")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(count_nonzero, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(count_nonzero, schema_str, "count_nonzero(Tensor self, int? dim=None) -> Tensor")

// aten::count_nonzero(Tensor self, int? dim=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<count_nonzero::schema> create_count_nonzero_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(count_nonzero::name, count_nonzero::overload_name)
      .typed<count_nonzero::schema>();
}

// aten::count_nonzero(Tensor self, int? dim=None) -> Tensor
at::Tensor count_nonzero::call(const at::Tensor & self, c10::optional<int64_t> dim) {
    static auto op = create_count_nonzero_typed_handle();
    return op.call(self, dim);
}

// aten::count_nonzero(Tensor self, int? dim=None) -> Tensor
at::Tensor count_nonzero::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<int64_t> dim) {
    static auto op = create_count_nonzero_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cov, name, "aten::cov")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cov, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cov, schema_str, "cov(Tensor self, *, int correction=1, Tensor? fweights=None, Tensor? aweights=None) -> Tensor")

// aten::cov(Tensor self, *, int correction=1, Tensor? fweights=None, Tensor? aweights=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<cov::schema> create_cov_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cov::name, cov::overload_name)
      .typed<cov::schema>();
}

// aten::cov(Tensor self, *, int correction=1, Tensor? fweights=None, Tensor? aweights=None) -> Tensor
at::Tensor cov::call(const at::Tensor & self, int64_t correction, const c10::optional<at::Tensor> & fweights, const c10::optional<at::Tensor> & aweights) {
    static auto op = create_cov_typed_handle();
    return op.call(self, correction, fweights, aweights);
}

// aten::cov(Tensor self, *, int correction=1, Tensor? fweights=None, Tensor? aweights=None) -> Tensor
at::Tensor cov::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t correction, const c10::optional<at::Tensor> & fweights, const c10::optional<at::Tensor> & aweights) {
    static auto op = create_cov_typed_handle();
    return op.redispatch(dispatchKeySet, self, correction, fweights, aweights);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_transpose_deprecated2, name, "aten::cudnn_convolution_transpose")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_transpose_deprecated2, overload_name, "deprecated2")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_transpose_deprecated2, schema_str, "cudnn_convolution_transpose.deprecated2(Tensor self, Tensor weight, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor")

// aten::cudnn_convolution_transpose.deprecated2(Tensor self, Tensor weight, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<cudnn_convolution_transpose_deprecated2::schema> create_cudnn_convolution_transpose_deprecated2_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cudnn_convolution_transpose_deprecated2::name, cudnn_convolution_transpose_deprecated2::overload_name)
      .typed<cudnn_convolution_transpose_deprecated2::schema>();
}

// aten::cudnn_convolution_transpose.deprecated2(Tensor self, Tensor weight, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
at::Tensor cudnn_convolution_transpose_deprecated2::call(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
    static auto op = create_cudnn_convolution_transpose_deprecated2_typed_handle();
    return op.call(self, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
}

// aten::cudnn_convolution_transpose.deprecated2(Tensor self, Tensor weight, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
at::Tensor cudnn_convolution_transpose_deprecated2::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
    static auto op = create_cudnn_convolution_transpose_deprecated2_typed_handle();
    return op.redispatch(dispatchKeySet, self, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_add_relu, name, "aten::cudnn_convolution_add_relu")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_add_relu, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_add_relu, schema_str, "cudnn_convolution_add_relu(Tensor self, Tensor weight, Tensor z, Scalar? alpha, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor")

// aten::cudnn_convolution_add_relu(Tensor self, Tensor weight, Tensor z, Scalar? alpha, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<cudnn_convolution_add_relu::schema> create_cudnn_convolution_add_relu_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cudnn_convolution_add_relu::name, cudnn_convolution_add_relu::overload_name)
      .typed<cudnn_convolution_add_relu::schema>();
}

// aten::cudnn_convolution_add_relu(Tensor self, Tensor weight, Tensor z, Scalar? alpha, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor
at::Tensor cudnn_convolution_add_relu::call(const at::Tensor & self, const at::Tensor & weight, const at::Tensor & z, const c10::optional<at::Scalar> & alpha, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
    static auto op = create_cudnn_convolution_add_relu_typed_handle();
    return op.call(self, weight, z, alpha, bias, stride, padding, dilation, groups);
}

// aten::cudnn_convolution_add_relu(Tensor self, Tensor weight, Tensor z, Scalar? alpha, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor
at::Tensor cudnn_convolution_add_relu::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, const at::Tensor & z, const c10::optional<at::Scalar> & alpha, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
    static auto op = create_cudnn_convolution_add_relu_typed_handle();
    return op.redispatch(dispatchKeySet, self, weight, z, alpha, bias, stride, padding, dilation, groups);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cummax, name, "aten::cummax")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cummax, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cummax, schema_str, "cummax(Tensor self, int dim) -> (Tensor values, Tensor indices)")

// aten::cummax(Tensor self, int dim) -> (Tensor values, Tensor indices)
static C10_NOINLINE c10::TypedOperatorHandle<cummax::schema> create_cummax_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cummax::name, cummax::overload_name)
      .typed<cummax::schema>();
}

// aten::cummax(Tensor self, int dim) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> cummax::call(const at::Tensor & self, int64_t dim) {
    static auto op = create_cummax_typed_handle();
    return op.call(self, dim);
}

// aten::cummax(Tensor self, int dim) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> cummax::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim) {
    static auto op = create_cummax_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cummax_helper, name, "aten::_cummax_helper")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cummax_helper, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cummax_helper, schema_str, "_cummax_helper(Tensor self, Tensor(a!) values, Tensor(b!) indices, int dim) -> ()")

// aten::_cummax_helper(Tensor self, Tensor(a!) values, Tensor(b!) indices, int dim) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_cummax_helper::schema> create__cummax_helper_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_cummax_helper::name, _cummax_helper::overload_name)
      .typed<_cummax_helper::schema>();
}

// aten::_cummax_helper(Tensor self, Tensor(a!) values, Tensor(b!) indices, int dim) -> ()
void _cummax_helper::call(const at::Tensor & self, at::Tensor & values, at::Tensor & indices, int64_t dim) {
    static auto op = create__cummax_helper_typed_handle();
    return op.call(self, values, indices, dim);
}

// aten::_cummax_helper(Tensor self, Tensor(a!) values, Tensor(b!) indices, int dim) -> ()
void _cummax_helper::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & values, at::Tensor & indices, int64_t dim) {
    static auto op = create__cummax_helper_typed_handle();
    return op.redispatch(dispatchKeySet, self, values, indices, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumprod_, name, "aten::cumprod_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumprod_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumprod_, schema_str, "cumprod_(Tensor(a!) self, int dim, *, ScalarType? dtype=None) -> Tensor(a!)")

// aten::cumprod_(Tensor(a!) self, int dim, *, ScalarType? dtype=None) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<cumprod_::schema> create_cumprod__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cumprod_::name, cumprod_::overload_name)
      .typed<cumprod_::schema>();
}

// aten::cumprod_(Tensor(a!) self, int dim, *, ScalarType? dtype=None) -> Tensor(a!)
at::Tensor & cumprod_::call(at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_cumprod__typed_handle();
    return op.call(self, dim, dtype);
}

// aten::cumprod_(Tensor(a!) self, int dim, *, ScalarType? dtype=None) -> Tensor(a!)
at::Tensor & cumprod_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_cumprod__typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumprod_dimname, name, "aten::cumprod")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumprod_dimname, overload_name, "dimname")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumprod_dimname, schema_str, "cumprod.dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor")

// aten::cumprod.dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<cumprod_dimname::schema> create_cumprod_dimname_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cumprod_dimname::name, cumprod_dimname::overload_name)
      .typed<cumprod_dimname::schema>();
}

// aten::cumprod.dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
at::Tensor cumprod_dimname::call(const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_cumprod_dimname_typed_handle();
    return op.call(self, dim, dtype);
}

// aten::cumprod.dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
at::Tensor cumprod_dimname::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_cumprod_dimname_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumsum_, name, "aten::cumsum_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumsum_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumsum_, schema_str, "cumsum_(Tensor(a!) self, int dim, *, ScalarType? dtype=None) -> Tensor(a!)")

// aten::cumsum_(Tensor(a!) self, int dim, *, ScalarType? dtype=None) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<cumsum_::schema> create_cumsum__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cumsum_::name, cumsum_::overload_name)
      .typed<cumsum_::schema>();
}

// aten::cumsum_(Tensor(a!) self, int dim, *, ScalarType? dtype=None) -> Tensor(a!)
at::Tensor & cumsum_::call(at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_cumsum__typed_handle();
    return op.call(self, dim, dtype);
}

// aten::cumsum_(Tensor(a!) self, int dim, *, ScalarType? dtype=None) -> Tensor(a!)
at::Tensor & cumsum_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_cumsum__typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumsum_out, name, "aten::cumsum")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumsum_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumsum_out, schema_str, "cumsum.out(Tensor self, int dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)")

// aten::cumsum.out(Tensor self, int dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<cumsum_out::schema> create_cumsum_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cumsum_out::name, cumsum_out::overload_name)
      .typed<cumsum_out::schema>();
}

// aten::cumsum.out(Tensor self, int dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & cumsum_out::call(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
    static auto op = create_cumsum_out_typed_handle();
    return op.call(self, dim, dtype, out);
}

// aten::cumsum.out(Tensor self, int dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & cumsum_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
    static auto op = create_cumsum_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, dtype, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ctc_loss_IntList, name, "aten::ctc_loss")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ctc_loss_IntList, overload_name, "IntList")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ctc_loss_IntList, schema_str, "ctc_loss.IntList(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank=0, int reduction=Mean, bool zero_infinity=False) -> Tensor")

// aten::ctc_loss.IntList(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank=0, int reduction=Mean, bool zero_infinity=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<ctc_loss_IntList::schema> create_ctc_loss_IntList_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(ctc_loss_IntList::name, ctc_loss_IntList::overload_name)
      .typed<ctc_loss_IntList::schema>();
}

// aten::ctc_loss.IntList(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank=0, int reduction=Mean, bool zero_infinity=False) -> Tensor
at::Tensor ctc_loss_IntList::call(const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, int64_t blank, int64_t reduction, bool zero_infinity) {
    static auto op = create_ctc_loss_IntList_typed_handle();
    return op.call(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
}

// aten::ctc_loss.IntList(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank=0, int reduction=Mean, bool zero_infinity=False) -> Tensor
at::Tensor ctc_loss_IntList::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, int64_t blank, int64_t reduction, bool zero_infinity) {
    static auto op = create_ctc_loss_IntList_typed_handle();
    return op.redispatch(dispatchKeySet, log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_ctc_loss_backward, name, "aten::_ctc_loss_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_ctc_loss_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_ctc_loss_backward, schema_str, "_ctc_loss_backward(Tensor grad, Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, Tensor neg_log_likelihood, Tensor log_alpha, int blank, bool zero_infinity=False) -> Tensor")

// aten::_ctc_loss_backward(Tensor grad, Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, Tensor neg_log_likelihood, Tensor log_alpha, int blank, bool zero_infinity=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_ctc_loss_backward::schema> create__ctc_loss_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_ctc_loss_backward::name, _ctc_loss_backward::overload_name)
      .typed<_ctc_loss_backward::schema>();
}

// aten::_ctc_loss_backward(Tensor grad, Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, Tensor neg_log_likelihood, Tensor log_alpha, int blank, bool zero_infinity=False) -> Tensor
at::Tensor _ctc_loss_backward::call(const at::Tensor & grad, const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, const at::Tensor & neg_log_likelihood, const at::Tensor & log_alpha, int64_t blank, bool zero_infinity) {
    static auto op = create__ctc_loss_backward_typed_handle();
    return op.call(grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank, zero_infinity);
}

// aten::_ctc_loss_backward(Tensor grad, Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, Tensor neg_log_likelihood, Tensor log_alpha, int blank, bool zero_infinity=False) -> Tensor
at::Tensor _ctc_loss_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, const at::Tensor & neg_log_likelihood, const at::Tensor & log_alpha, int64_t blank, bool zero_infinity) {
    static auto op = create__ctc_loss_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank, zero_infinity);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(diagonal_backward, name, "aten::diagonal_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(diagonal_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(diagonal_backward, schema_str, "diagonal_backward(Tensor grad_output, int[] input_sizes, int offset, int dim1, int dim2) -> Tensor")

// aten::diagonal_backward(Tensor grad_output, int[] input_sizes, int offset, int dim1, int dim2) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<diagonal_backward::schema> create_diagonal_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(diagonal_backward::name, diagonal_backward::overload_name)
      .typed<diagonal_backward::schema>();
}

// aten::diagonal_backward(Tensor grad_output, int[] input_sizes, int offset, int dim1, int dim2) -> Tensor
at::Tensor diagonal_backward::call(const at::Tensor & grad_output, at::IntArrayRef input_sizes, int64_t offset, int64_t dim1, int64_t dim2) {
    static auto op = create_diagonal_backward_typed_handle();
    return op.call(grad_output, input_sizes, offset, dim1, dim2);
}

// aten::diagonal_backward(Tensor grad_output, int[] input_sizes, int offset, int dim1, int dim2) -> Tensor
at::Tensor diagonal_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, at::IntArrayRef input_sizes, int64_t offset, int64_t dim1, int64_t dim2) {
    static auto op = create_diagonal_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, input_sizes, offset, dim1, dim2);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(diff, name, "aten::diff")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(diff, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(diff, schema_str, "diff(Tensor self, int n=1, int dim=-1, Tensor? prepend=None, Tensor? append=None) -> Tensor")

// aten::diff(Tensor self, int n=1, int dim=-1, Tensor? prepend=None, Tensor? append=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<diff::schema> create_diff_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(diff::name, diff::overload_name)
      .typed<diff::schema>();
}

// aten::diff(Tensor self, int n=1, int dim=-1, Tensor? prepend=None, Tensor? append=None) -> Tensor
at::Tensor diff::call(const at::Tensor & self, int64_t n, int64_t dim, const c10::optional<at::Tensor> & prepend, const c10::optional<at::Tensor> & append) {
    static auto op = create_diff_typed_handle();
    return op.call(self, n, dim, prepend, append);
}

// aten::diff(Tensor self, int n=1, int dim=-1, Tensor? prepend=None, Tensor? append=None) -> Tensor
at::Tensor diff::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t n, int64_t dim, const c10::optional<at::Tensor> & prepend, const c10::optional<at::Tensor> & append) {
    static auto op = create_diff_typed_handle();
    return op.redispatch(dispatchKeySet, self, n, dim, prepend, append);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gradient_scalarint, name, "aten::gradient")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gradient_scalarint, overload_name, "scalarint")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gradient_scalarint, schema_str, "gradient.scalarint(Tensor self, *, Scalar? spacing=None, int? dim=None, int edge_order=1) -> Tensor[]")

// aten::gradient.scalarint(Tensor self, *, Scalar? spacing=None, int? dim=None, int edge_order=1) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<gradient_scalarint::schema> create_gradient_scalarint_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(gradient_scalarint::name, gradient_scalarint::overload_name)
      .typed<gradient_scalarint::schema>();
}

// aten::gradient.scalarint(Tensor self, *, Scalar? spacing=None, int? dim=None, int edge_order=1) -> Tensor[]
::std::vector<at::Tensor> gradient_scalarint::call(const at::Tensor & self, const c10::optional<at::Scalar> & spacing, c10::optional<int64_t> dim, int64_t edge_order) {
    static auto op = create_gradient_scalarint_typed_handle();
    return op.call(self, spacing, dim, edge_order);
}

// aten::gradient.scalarint(Tensor self, *, Scalar? spacing=None, int? dim=None, int edge_order=1) -> Tensor[]
::std::vector<at::Tensor> gradient_scalarint::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Scalar> & spacing, c10::optional<int64_t> dim, int64_t edge_order) {
    static auto op = create_gradient_scalarint_typed_handle();
    return op.redispatch(dispatchKeySet, self, spacing, dim, edge_order);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gradient_scalararray, name, "aten::gradient")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gradient_scalararray, overload_name, "scalararray")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gradient_scalararray, schema_str, "gradient.scalararray(Tensor self, *, Scalar spacing, int[] dim, int edge_order=1) -> Tensor[]")

// aten::gradient.scalararray(Tensor self, *, Scalar spacing, int[] dim, int edge_order=1) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<gradient_scalararray::schema> create_gradient_scalararray_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(gradient_scalararray::name, gradient_scalararray::overload_name)
      .typed<gradient_scalararray::schema>();
}

// aten::gradient.scalararray(Tensor self, *, Scalar spacing, int[] dim, int edge_order=1) -> Tensor[]
::std::vector<at::Tensor> gradient_scalararray::call(const at::Tensor & self, const at::Scalar & spacing, at::IntArrayRef dim, int64_t edge_order) {
    static auto op = create_gradient_scalararray_typed_handle();
    return op.call(self, spacing, dim, edge_order);
}

// aten::gradient.scalararray(Tensor self, *, Scalar spacing, int[] dim, int edge_order=1) -> Tensor[]
::std::vector<at::Tensor> gradient_scalararray::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & spacing, at::IntArrayRef dim, int64_t edge_order) {
    static auto op = create_gradient_scalararray_typed_handle();
    return op.redispatch(dispatchKeySet, self, spacing, dim, edge_order);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gradient_tensorarrayint, name, "aten::gradient")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gradient_tensorarrayint, overload_name, "tensorarrayint")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gradient_tensorarrayint, schema_str, "gradient.tensorarrayint(Tensor self, *, Tensor[] spacing, int? dim=None, int edge_order=1) -> Tensor[]")

// aten::gradient.tensorarrayint(Tensor self, *, Tensor[] spacing, int? dim=None, int edge_order=1) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<gradient_tensorarrayint::schema> create_gradient_tensorarrayint_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(gradient_tensorarrayint::name, gradient_tensorarrayint::overload_name)
      .typed<gradient_tensorarrayint::schema>();
}

// aten::gradient.tensorarrayint(Tensor self, *, Tensor[] spacing, int? dim=None, int edge_order=1) -> Tensor[]
::std::vector<at::Tensor> gradient_tensorarrayint::call(const at::Tensor & self, at::TensorList spacing, c10::optional<int64_t> dim, int64_t edge_order) {
    static auto op = create_gradient_tensorarrayint_typed_handle();
    return op.call(self, spacing, dim, edge_order);
}

// aten::gradient.tensorarrayint(Tensor self, *, Tensor[] spacing, int? dim=None, int edge_order=1) -> Tensor[]
::std::vector<at::Tensor> gradient_tensorarrayint::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::TensorList spacing, c10::optional<int64_t> dim, int64_t edge_order) {
    static auto op = create_gradient_tensorarrayint_typed_handle();
    return op.redispatch(dispatchKeySet, self, spacing, dim, edge_order);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(divide_out, name, "aten::divide")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(divide_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(divide_out, schema_str, "divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<divide_out::schema> create_divide_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(divide_out::name, divide_out::overload_name)
      .typed<divide_out::schema>();
}

// aten::divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & divide_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_divide_out_typed_handle();
    return op.call(self, other, out);
}

// aten::divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & divide_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_divide_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(divide_Scalar_mode, name, "aten::divide")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(divide_Scalar_mode, overload_name, "Scalar_mode")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(divide_Scalar_mode, schema_str, "divide.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> Tensor")

// aten::divide.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<divide_Scalar_mode::schema> create_divide_Scalar_mode_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(divide_Scalar_mode::name, divide_Scalar_mode::overload_name)
      .typed<divide_Scalar_mode::schema>();
}

// aten::divide.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> Tensor
at::Tensor divide_Scalar_mode::call(const at::Tensor & self, const at::Scalar & other, c10::optional<c10::string_view> rounding_mode) {
    static auto op = create_divide_Scalar_mode_typed_handle();
    return op.call(self, other, rounding_mode);
}

// aten::divide.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> Tensor
at::Tensor divide_Scalar_mode::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, c10::optional<c10::string_view> rounding_mode) {
    static auto op = create_divide_Scalar_mode_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, rounding_mode);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(dot, name, "aten::dot")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(dot, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(dot, schema_str, "dot(Tensor self, Tensor tensor) -> Tensor")

// aten::dot(Tensor self, Tensor tensor) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<dot::schema> create_dot_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(dot::name, dot::overload_name)
      .typed<dot::schema>();
}

// aten::dot(Tensor self, Tensor tensor) -> Tensor
at::Tensor dot::call(const at::Tensor & self, const at::Tensor & tensor) {
    static auto op = create_dot_typed_handle();
    return op.call(self, tensor);
}

// aten::dot(Tensor self, Tensor tensor) -> Tensor
at::Tensor dot::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & tensor) {
    static auto op = create_dot_typed_handle();
    return op.redispatch(dispatchKeySet, self, tensor);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(einsum, name, "aten::einsum")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(einsum, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(einsum, schema_str, "einsum(str equation, Tensor[] tensors) -> Tensor")

// aten::einsum(str equation, Tensor[] tensors) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<einsum::schema> create_einsum_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(einsum::name, einsum::overload_name)
      .typed<einsum::schema>();
}

// aten::einsum(str equation, Tensor[] tensors) -> Tensor
at::Tensor einsum::call(c10::string_view equation, at::TensorList tensors) {
    static auto op = create_einsum_typed_handle();
    return op.call(equation, tensors);
}

// aten::einsum(str equation, Tensor[] tensors) -> Tensor
at::Tensor einsum::redispatch(c10::DispatchKeySet dispatchKeySet, c10::string_view equation, at::TensorList tensors) {
    static auto op = create_einsum_typed_handle();
    return op.redispatch(dispatchKeySet, equation, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(embedding_sparse_backward, name, "aten::embedding_sparse_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(embedding_sparse_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(embedding_sparse_backward, schema_str, "embedding_sparse_backward(Tensor grad, Tensor indices, int num_weights, int padding_idx, bool scale_grad_by_freq) -> Tensor")

// aten::embedding_sparse_backward(Tensor grad, Tensor indices, int num_weights, int padding_idx, bool scale_grad_by_freq) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<embedding_sparse_backward::schema> create_embedding_sparse_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(embedding_sparse_backward::name, embedding_sparse_backward::overload_name)
      .typed<embedding_sparse_backward::schema>();
}

// aten::embedding_sparse_backward(Tensor grad, Tensor indices, int num_weights, int padding_idx, bool scale_grad_by_freq) -> Tensor
at::Tensor embedding_sparse_backward::call(const at::Tensor & grad, const at::Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {
    static auto op = create_embedding_sparse_backward_typed_handle();
    return op.call(grad, indices, num_weights, padding_idx, scale_grad_by_freq);
}

// aten::embedding_sparse_backward(Tensor grad, Tensor indices, int num_weights, int padding_idx, bool scale_grad_by_freq) -> Tensor
at::Tensor embedding_sparse_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {
    static auto op = create_embedding_sparse_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad, indices, num_weights, padding_idx, scale_grad_by_freq);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_embedding_bag_per_sample_weights_backward, name, "aten::_embedding_bag_per_sample_weights_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_embedding_bag_per_sample_weights_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_embedding_bag_per_sample_weights_backward, schema_str, "_embedding_bag_per_sample_weights_backward(Tensor grad, Tensor weight, Tensor indices, Tensor offsets, Tensor offset2bag, int mode, int padding_idx=-1) -> Tensor")

// aten::_embedding_bag_per_sample_weights_backward(Tensor grad, Tensor weight, Tensor indices, Tensor offsets, Tensor offset2bag, int mode, int padding_idx=-1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_embedding_bag_per_sample_weights_backward::schema> create__embedding_bag_per_sample_weights_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_embedding_bag_per_sample_weights_backward::name, _embedding_bag_per_sample_weights_backward::overload_name)
      .typed<_embedding_bag_per_sample_weights_backward::schema>();
}

// aten::_embedding_bag_per_sample_weights_backward(Tensor grad, Tensor weight, Tensor indices, Tensor offsets, Tensor offset2bag, int mode, int padding_idx=-1) -> Tensor
at::Tensor _embedding_bag_per_sample_weights_backward::call(const at::Tensor & grad, const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, const at::Tensor & offset2bag, int64_t mode, int64_t padding_idx) {
    static auto op = create__embedding_bag_per_sample_weights_backward_typed_handle();
    return op.call(grad, weight, indices, offsets, offset2bag, mode, padding_idx);
}

// aten::_embedding_bag_per_sample_weights_backward(Tensor grad, Tensor weight, Tensor indices, Tensor offsets, Tensor offset2bag, int mode, int padding_idx=-1) -> Tensor
at::Tensor _embedding_bag_per_sample_weights_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, const at::Tensor & offset2bag, int64_t mode, int64_t padding_idx) {
    static auto op = create__embedding_bag_per_sample_weights_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad, weight, indices, offsets, offset2bag, mode, padding_idx);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(empty_names, name, "aten::empty")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(empty_names, overload_name, "names")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(empty_names, schema_str, "empty.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor")

// aten::empty.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<empty_names::schema> create_empty_names_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(empty_names::name, empty_names::overload_name)
      .typed<empty_names::schema>();
}

// aten::empty.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
at::Tensor empty_names::call(at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create_empty_names_typed_handle();
    return op.call(size, names, dtype, layout, device, pin_memory, memory_format);
}

// aten::empty.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
at::Tensor empty_names::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create_empty_names_typed_handle();
    return op.redispatch(dispatchKeySet, size, names, dtype, layout, device, pin_memory, memory_format);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(new_empty_strided, name, "aten::new_empty_strided")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(new_empty_strided, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(new_empty_strided, schema_str, "new_empty_strided(Tensor self, int[] size, int[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::new_empty_strided(Tensor self, int[] size, int[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<new_empty_strided::schema> create_new_empty_strided_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(new_empty_strided::name, new_empty_strided::overload_name)
      .typed<new_empty_strided::schema>();
}

// aten::new_empty_strided(Tensor self, int[] size, int[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor new_empty_strided::call(const at::Tensor & self, at::IntArrayRef size, at::IntArrayRef stride, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_new_empty_strided_typed_handle();
    return op.call(self, size, stride, dtype, layout, device, pin_memory);
}

// aten::new_empty_strided(Tensor self, int[] size, int[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor new_empty_strided::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef size, at::IntArrayRef stride, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_new_empty_strided_typed_handle();
    return op.redispatch(dispatchKeySet, self, size, stride, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(new_full, name, "aten::new_full")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(new_full, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(new_full, schema_str, "new_full(Tensor self, int[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::new_full(Tensor self, int[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<new_full::schema> create_new_full_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(new_full::name, new_full::overload_name)
      .typed<new_full::schema>();
}

// aten::new_full(Tensor self, int[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor new_full::call(const at::Tensor & self, at::IntArrayRef size, const at::Scalar & fill_value, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_new_full_typed_handle();
    return op.call(self, size, fill_value, dtype, layout, device, pin_memory);
}

// aten::new_full(Tensor self, int[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor new_full::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef size, const at::Scalar & fill_value, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_new_full_typed_handle();
    return op.redispatch(dispatchKeySet, self, size, fill_value, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(new_ones, name, "aten::new_ones")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(new_ones, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(new_ones, schema_str, "new_ones(Tensor self, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::new_ones(Tensor self, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<new_ones::schema> create_new_ones_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(new_ones::name, new_ones::overload_name)
      .typed<new_ones::schema>();
}

// aten::new_ones(Tensor self, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor new_ones::call(const at::Tensor & self, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_new_ones_typed_handle();
    return op.call(self, size, dtype, layout, device, pin_memory);
}

// aten::new_ones(Tensor self, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor new_ones::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_new_ones_typed_handle();
    return op.redispatch(dispatchKeySet, self, size, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_empty_per_channel_affine_quantized, name, "aten::_empty_per_channel_affine_quantized")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_empty_per_channel_affine_quantized, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_empty_per_channel_affine_quantized, schema_str, "_empty_per_channel_affine_quantized(int[] size, *, Tensor scales, Tensor zero_points, int axis, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=contiguous_format) -> Tensor")

// aten::_empty_per_channel_affine_quantized(int[] size, *, Tensor scales, Tensor zero_points, int axis, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=contiguous_format) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_empty_per_channel_affine_quantized::schema> create__empty_per_channel_affine_quantized_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_empty_per_channel_affine_quantized::name, _empty_per_channel_affine_quantized::overload_name)
      .typed<_empty_per_channel_affine_quantized::schema>();
}

// aten::_empty_per_channel_affine_quantized(int[] size, *, Tensor scales, Tensor zero_points, int axis, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=contiguous_format) -> Tensor
at::Tensor _empty_per_channel_affine_quantized::call(at::IntArrayRef size, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create__empty_per_channel_affine_quantized_typed_handle();
    return op.call(size, scales, zero_points, axis, dtype, layout, device, pin_memory, memory_format);
}

// aten::_empty_per_channel_affine_quantized(int[] size, *, Tensor scales, Tensor zero_points, int axis, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=contiguous_format) -> Tensor
at::Tensor _empty_per_channel_affine_quantized::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create__empty_per_channel_affine_quantized_typed_handle();
    return op.redispatch(dispatchKeySet, size, scales, zero_points, axis, dtype, layout, device, pin_memory, memory_format);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(empty_quantized, name, "aten::empty_quantized")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(empty_quantized, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(empty_quantized, schema_str, "empty_quantized(int[] size, Tensor qtensor, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor")

// aten::empty_quantized(int[] size, Tensor qtensor, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<empty_quantized::schema> create_empty_quantized_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(empty_quantized::name, empty_quantized::overload_name)
      .typed<empty_quantized::schema>();
}

// aten::empty_quantized(int[] size, Tensor qtensor, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
at::Tensor empty_quantized::call(at::IntArrayRef size, const at::Tensor & qtensor, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create_empty_quantized_typed_handle();
    return op.call(size, qtensor, dtype, layout, device, pin_memory, memory_format);
}

// aten::empty_quantized(int[] size, Tensor qtensor, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
at::Tensor empty_quantized::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, const at::Tensor & qtensor, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create_empty_quantized_typed_handle();
    return op.redispatch(dispatchKeySet, size, qtensor, dtype, layout, device, pin_memory, memory_format);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(empty_out, name, "aten::empty")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(empty_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(empty_out, schema_str, "empty.out(int[] size, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)")

// aten::empty.out(int[] size, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<empty_out::schema> create_empty_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(empty_out::name, empty_out::overload_name)
      .typed<empty_out::schema>();
}

// aten::empty.out(int[] size, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & empty_out::call(at::IntArrayRef size, c10::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
    static auto op = create_empty_out_typed_handle();
    return op.call(size, memory_format, out);
}

// aten::empty.out(int[] size, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & empty_out::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
    static auto op = create_empty_out_typed_handle();
    return op.redispatch(dispatchKeySet, size, memory_format, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(empty_strided, name, "aten::empty_strided")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(empty_strided, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(empty_strided, schema_str, "empty_strided(int[] size, int[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::empty_strided(int[] size, int[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<empty_strided::schema> create_empty_strided_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(empty_strided::name, empty_strided::overload_name)
      .typed<empty_strided::schema>();
}

// aten::empty_strided(int[] size, int[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor empty_strided::call(at::IntArrayRef size, at::IntArrayRef stride, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_empty_strided_typed_handle();
    return op.call(size, stride, dtype, layout, device, pin_memory);
}

// aten::empty_strided(int[] size, int[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor empty_strided::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, at::IntArrayRef stride, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_empty_strided_typed_handle();
    return op.redispatch(dispatchKeySet, size, stride, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(exp, name, "aten::exp")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(exp, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(exp, schema_str, "exp(Tensor self) -> Tensor")

// aten::exp(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<exp::schema> create_exp_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(exp::name, exp::overload_name)
      .typed<exp::schema>();
}

// aten::exp(Tensor self) -> Tensor
at::Tensor exp::call(const at::Tensor & self) {
    static auto op = create_exp_typed_handle();
    return op.call(self);
}

// aten::exp(Tensor self) -> Tensor
at::Tensor exp::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_exp_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(exp2, name, "aten::exp2")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(exp2, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(exp2, schema_str, "exp2(Tensor self) -> Tensor")

// aten::exp2(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<exp2::schema> create_exp2_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(exp2::name, exp2::overload_name)
      .typed<exp2::schema>();
}

// aten::exp2(Tensor self) -> Tensor
at::Tensor exp2::call(const at::Tensor & self) {
    static auto op = create_exp2_typed_handle();
    return op.call(self);
}

// aten::exp2(Tensor self) -> Tensor
at::Tensor exp2::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_exp2_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(expm1_, name, "aten::expm1_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(expm1_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(expm1_, schema_str, "expm1_(Tensor(a!) self) -> Tensor(a!)")

// aten::expm1_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<expm1_::schema> create_expm1__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(expm1_::name, expm1_::overload_name)
      .typed<expm1_::schema>();
}

// aten::expm1_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & expm1_::call(at::Tensor & self) {
    static auto op = create_expm1__typed_handle();
    return op.call(self);
}

// aten::expm1_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & expm1_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_expm1__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(eye, name, "aten::eye")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(eye, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(eye, schema_str, "eye(int n, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::eye(int n, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<eye::schema> create_eye_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(eye::name, eye::overload_name)
      .typed<eye::schema>();
}

// aten::eye(int n, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor eye::call(int64_t n, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_eye_typed_handle();
    return op.call(n, dtype, layout, device, pin_memory);
}

// aten::eye(int n, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor eye::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t n, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_eye_typed_handle();
    return op.redispatch(dispatchKeySet, n, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(frac, name, "aten::frac")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(frac, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(frac, schema_str, "frac(Tensor self) -> Tensor")

// aten::frac(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<frac::schema> create_frac_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(frac::name, frac::overload_name)
      .typed<frac::schema>();
}

// aten::frac(Tensor self) -> Tensor
at::Tensor frac::call(const at::Tensor & self) {
    static auto op = create_frac_typed_handle();
    return op.call(self);
}

// aten::frac(Tensor self) -> Tensor
at::Tensor frac::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_frac_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(from_file, name, "aten::from_file")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(from_file, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(from_file, schema_str, "from_file(str filename, bool? shared=None, int? size=0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::from_file(str filename, bool? shared=None, int? size=0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<from_file::schema> create_from_file_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(from_file::name, from_file::overload_name)
      .typed<from_file::schema>();
}

// aten::from_file(str filename, bool? shared=None, int? size=0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor from_file::call(c10::string_view filename, c10::optional<bool> shared, c10::optional<int64_t> size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_from_file_typed_handle();
    return op.call(filename, shared, size, dtype, layout, device, pin_memory);
}

// aten::from_file(str filename, bool? shared=None, int? size=0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor from_file::redispatch(c10::DispatchKeySet dispatchKeySet, c10::string_view filename, c10::optional<bool> shared, c10::optional<int64_t> size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_from_file_typed_handle();
    return op.redispatch(dispatchKeySet, filename, shared, size, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gcd, name, "aten::gcd")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gcd, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gcd, schema_str, "gcd(Tensor self, Tensor other) -> Tensor")

// aten::gcd(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<gcd::schema> create_gcd_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(gcd::name, gcd::overload_name)
      .typed<gcd::schema>();
}

// aten::gcd(Tensor self, Tensor other) -> Tensor
at::Tensor gcd::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_gcd_typed_handle();
    return op.call(self, other);
}

// aten::gcd(Tensor self, Tensor other) -> Tensor
at::Tensor gcd::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_gcd_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gcd_, name, "aten::gcd_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gcd_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gcd_, schema_str, "gcd_(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::gcd_(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<gcd_::schema> create_gcd__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(gcd_::name, gcd_::overload_name)
      .typed<gcd_::schema>();
}

// aten::gcd_(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & gcd_::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create_gcd__typed_handle();
    return op.call(self, other);
}

// aten::gcd_(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & gcd_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create_gcd__typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hamming_window_periodic_alpha_beta, name, "aten::hamming_window")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hamming_window_periodic_alpha_beta, overload_name, "periodic_alpha_beta")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hamming_window_periodic_alpha_beta, schema_str, "hamming_window.periodic_alpha_beta(int window_length, bool periodic, float alpha, float beta, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::hamming_window.periodic_alpha_beta(int window_length, bool periodic, float alpha, float beta, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<hamming_window_periodic_alpha_beta::schema> create_hamming_window_periodic_alpha_beta_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(hamming_window_periodic_alpha_beta::name, hamming_window_periodic_alpha_beta::overload_name)
      .typed<hamming_window_periodic_alpha_beta::schema>();
}

// aten::hamming_window.periodic_alpha_beta(int window_length, bool periodic, float alpha, float beta, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor hamming_window_periodic_alpha_beta::call(int64_t window_length, bool periodic, double alpha, double beta, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_hamming_window_periodic_alpha_beta_typed_handle();
    return op.call(window_length, periodic, alpha, beta, dtype, layout, device, pin_memory);
}

// aten::hamming_window.periodic_alpha_beta(int window_length, bool periodic, float alpha, float beta, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor hamming_window_periodic_alpha_beta::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t window_length, bool periodic, double alpha, double beta, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_hamming_window_periodic_alpha_beta_typed_handle();
    return op.redispatch(dispatchKeySet, window_length, periodic, alpha, beta, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fft_c2r_out, name, "aten::_fft_c2r")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fft_c2r_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fft_c2r_out, schema_str, "_fft_c2r.out(Tensor self, int[] dim, int normalization, int last_dim_size, *, Tensor(a!) out) -> Tensor(a!)")

// aten::_fft_c2r.out(Tensor self, int[] dim, int normalization, int last_dim_size, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<_fft_c2r_out::schema> create__fft_c2r_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_fft_c2r_out::name, _fft_c2r_out::overload_name)
      .typed<_fft_c2r_out::schema>();
}

// aten::_fft_c2r.out(Tensor self, int[] dim, int normalization, int last_dim_size, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & _fft_c2r_out::call(const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, int64_t last_dim_size, at::Tensor & out) {
    static auto op = create__fft_c2r_out_typed_handle();
    return op.call(self, dim, normalization, last_dim_size, out);
}

// aten::_fft_c2r.out(Tensor self, int[] dim, int normalization, int last_dim_size, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & _fft_c2r_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, int64_t last_dim_size, at::Tensor & out) {
    static auto op = create__fft_c2r_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, normalization, last_dim_size, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fft_c2c_out, name, "aten::_fft_c2c")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fft_c2c_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fft_c2c_out, schema_str, "_fft_c2c.out(Tensor self, int[] dim, int normalization, bool forward, *, Tensor(a!) out) -> Tensor(a!)")

// aten::_fft_c2c.out(Tensor self, int[] dim, int normalization, bool forward, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<_fft_c2c_out::schema> create__fft_c2c_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_fft_c2c_out::name, _fft_c2c_out::overload_name)
      .typed<_fft_c2c_out::schema>();
}

// aten::_fft_c2c.out(Tensor self, int[] dim, int normalization, bool forward, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & _fft_c2c_out::call(const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, bool forward, at::Tensor & out) {
    static auto op = create__fft_c2c_out_typed_handle();
    return op.call(self, dim, normalization, forward, out);
}

// aten::_fft_c2c.out(Tensor self, int[] dim, int normalization, bool forward, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & _fft_c2c_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, bool forward, at::Tensor & out) {
    static auto op = create__fft_c2c_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, normalization, forward, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cufft_clear_plan_cache, name, "aten::_cufft_clear_plan_cache")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cufft_clear_plan_cache, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cufft_clear_plan_cache, schema_str, "_cufft_clear_plan_cache(int device_index) -> ()")

// aten::_cufft_clear_plan_cache(int device_index) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_cufft_clear_plan_cache::schema> create__cufft_clear_plan_cache_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_cufft_clear_plan_cache::name, _cufft_clear_plan_cache::overload_name)
      .typed<_cufft_clear_plan_cache::schema>();
}

// aten::_cufft_clear_plan_cache(int device_index) -> ()
void _cufft_clear_plan_cache::call(int64_t device_index) {
    static auto op = create__cufft_clear_plan_cache_typed_handle();
    return op.call(device_index);
}

// aten::_cufft_clear_plan_cache(int device_index) -> ()
void _cufft_clear_plan_cache::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t device_index) {
    static auto op = create__cufft_clear_plan_cache_typed_handle();
    return op.redispatch(dispatchKeySet, device_index);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_Tensor, name, "aten::index")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_Tensor, schema_str, "index.Tensor(Tensor self, Tensor?[] indices) -> Tensor")

// aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<index_Tensor::schema> create_index_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(index_Tensor::name, index_Tensor::overload_name)
      .typed<index_Tensor::schema>();
}

// aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor
at::Tensor index_Tensor::call(const at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices) {
    static auto op = create_index_Tensor_typed_handle();
    return op.call(self, indices);
}

// aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor
at::Tensor index_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices) {
    static auto op = create_index_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, indices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_copy_dimname, name, "aten::index_copy")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_copy_dimname, overload_name, "dimname")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_copy_dimname, schema_str, "index_copy.dimname(Tensor self, Dimname dim, Tensor index, Tensor source) -> Tensor")

// aten::index_copy.dimname(Tensor self, Dimname dim, Tensor index, Tensor source) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<index_copy_dimname::schema> create_index_copy_dimname_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(index_copy_dimname::name, index_copy_dimname::overload_name)
      .typed<index_copy_dimname::schema>();
}

// aten::index_copy.dimname(Tensor self, Dimname dim, Tensor index, Tensor source) -> Tensor
at::Tensor index_copy_dimname::call(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & source) {
    static auto op = create_index_copy_dimname_typed_handle();
    return op.call(self, dim, index, source);
}

// aten::index_copy.dimname(Tensor self, Dimname dim, Tensor index, Tensor source) -> Tensor
at::Tensor index_copy_dimname::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & source) {
    static auto op = create_index_copy_dimname_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, source);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isin_Tensor_Tensor_out, name, "aten::isin")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isin_Tensor_Tensor_out, overload_name, "Tensor_Tensor_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isin_Tensor_Tensor_out, schema_str, "isin.Tensor_Tensor_out(Tensor elements, Tensor test_elements, *, bool assume_unique=False, bool invert=False, Tensor(a!) out) -> Tensor(a!)")

// aten::isin.Tensor_Tensor_out(Tensor elements, Tensor test_elements, *, bool assume_unique=False, bool invert=False, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<isin_Tensor_Tensor_out::schema> create_isin_Tensor_Tensor_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(isin_Tensor_Tensor_out::name, isin_Tensor_Tensor_out::overload_name)
      .typed<isin_Tensor_Tensor_out::schema>();
}

// aten::isin.Tensor_Tensor_out(Tensor elements, Tensor test_elements, *, bool assume_unique=False, bool invert=False, Tensor(a!) out) -> Tensor(a!)
at::Tensor & isin_Tensor_Tensor_out::call(const at::Tensor & elements, const at::Tensor & test_elements, bool assume_unique, bool invert, at::Tensor & out) {
    static auto op = create_isin_Tensor_Tensor_out_typed_handle();
    return op.call(elements, test_elements, assume_unique, invert, out);
}

// aten::isin.Tensor_Tensor_out(Tensor elements, Tensor test_elements, *, bool assume_unique=False, bool invert=False, Tensor(a!) out) -> Tensor(a!)
at::Tensor & isin_Tensor_Tensor_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & elements, const at::Tensor & test_elements, bool assume_unique, bool invert, at::Tensor & out) {
    static auto op = create_isin_Tensor_Tensor_out_typed_handle();
    return op.redispatch(dispatchKeySet, elements, test_elements, assume_unique, invert, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isin_Scalar_Tensor, name, "aten::isin")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isin_Scalar_Tensor, overload_name, "Scalar_Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isin_Scalar_Tensor, schema_str, "isin.Scalar_Tensor(Scalar element, Tensor test_elements, *, bool assume_unique=False, bool invert=False) -> Tensor")

// aten::isin.Scalar_Tensor(Scalar element, Tensor test_elements, *, bool assume_unique=False, bool invert=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<isin_Scalar_Tensor::schema> create_isin_Scalar_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(isin_Scalar_Tensor::name, isin_Scalar_Tensor::overload_name)
      .typed<isin_Scalar_Tensor::schema>();
}

// aten::isin.Scalar_Tensor(Scalar element, Tensor test_elements, *, bool assume_unique=False, bool invert=False) -> Tensor
at::Tensor isin_Scalar_Tensor::call(const at::Scalar & element, const at::Tensor & test_elements, bool assume_unique, bool invert) {
    static auto op = create_isin_Scalar_Tensor_typed_handle();
    return op.call(element, test_elements, assume_unique, invert);
}

// aten::isin.Scalar_Tensor(Scalar element, Tensor test_elements, *, bool assume_unique=False, bool invert=False) -> Tensor
at::Tensor isin_Scalar_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Scalar & element, const at::Tensor & test_elements, bool assume_unique, bool invert) {
    static auto op = create_isin_Scalar_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, element, test_elements, assume_unique, invert);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_conj, name, "aten::is_conj")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_conj, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_conj, schema_str, "is_conj(Tensor self) -> bool")

// aten::is_conj(Tensor self) -> bool
static C10_NOINLINE c10::TypedOperatorHandle<is_conj::schema> create_is_conj_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(is_conj::name, is_conj::overload_name)
      .typed<is_conj::schema>();
}

// aten::is_conj(Tensor self) -> bool
bool is_conj::call(const at::Tensor & self) {
    static auto op = create_is_conj_typed_handle();
    return op.call(self);
}

// aten::is_conj(Tensor self) -> bool
bool is_conj::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_is_conj_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_nonzero, name, "aten::is_nonzero")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_nonzero, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_nonzero, schema_str, "is_nonzero(Tensor self) -> bool")

// aten::is_nonzero(Tensor self) -> bool
static C10_NOINLINE c10::TypedOperatorHandle<is_nonzero::schema> create_is_nonzero_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(is_nonzero::name, is_nonzero::overload_name)
      .typed<is_nonzero::schema>();
}

// aten::is_nonzero(Tensor self) -> bool
bool is_nonzero::call(const at::Tensor & self) {
    static auto op = create_is_nonzero_typed_handle();
    return op.call(self);
}

// aten::is_nonzero(Tensor self) -> bool
bool is_nonzero::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_is_nonzero_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_signed, name, "aten::is_signed")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_signed, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_signed, schema_str, "is_signed(Tensor self) -> bool")

// aten::is_signed(Tensor self) -> bool
static C10_NOINLINE c10::TypedOperatorHandle<is_signed::schema> create_is_signed_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(is_signed::name, is_signed::overload_name)
      .typed<is_signed::schema>();
}

// aten::is_signed(Tensor self) -> bool
bool is_signed::call(const at::Tensor & self) {
    static auto op = create_is_signed_typed_handle();
    return op.call(self);
}

// aten::is_signed(Tensor self) -> bool
bool is_signed::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_is_signed_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(layer_norm, name, "aten::layer_norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(layer_norm, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(layer_norm, schema_str, "layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor")

// aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<layer_norm::schema> create_layer_norm_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(layer_norm::name, layer_norm::overload_name)
      .typed<layer_norm::schema>();
}

// aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor
at::Tensor layer_norm::call(const at::Tensor & input, at::IntArrayRef normalized_shape, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, double eps, bool cudnn_enable) {
    static auto op = create_layer_norm_typed_handle();
    return op.call(input, normalized_shape, weight, bias, eps, cudnn_enable);
}

// aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor
at::Tensor layer_norm::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, at::IntArrayRef normalized_shape, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, double eps, bool cudnn_enable) {
    static auto op = create_layer_norm_typed_handle();
    return op.redispatch(dispatchKeySet, input, normalized_shape, weight, bias, eps, cudnn_enable);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(native_layer_norm_backward, name, "aten::native_layer_norm_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(native_layer_norm_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(native_layer_norm_backward, schema_str, "native_layer_norm_backward(Tensor grad_out, Tensor input, int[] normalized_shape, Tensor mean, Tensor rstd, Tensor? weight, Tensor? bias, bool[3] output_mask) -> (Tensor, Tensor, Tensor)")

// aten::native_layer_norm_backward(Tensor grad_out, Tensor input, int[] normalized_shape, Tensor mean, Tensor rstd, Tensor? weight, Tensor? bias, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<native_layer_norm_backward::schema> create_native_layer_norm_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(native_layer_norm_backward::name, native_layer_norm_backward::overload_name)
      .typed<native_layer_norm_backward::schema>();
}

// aten::native_layer_norm_backward(Tensor grad_out, Tensor input, int[] normalized_shape, Tensor mean, Tensor rstd, Tensor? weight, Tensor? bias, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_layer_norm_backward::call(const at::Tensor & grad_out, const at::Tensor & input, at::IntArrayRef normalized_shape, const at::Tensor & mean, const at::Tensor & rstd, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, ::std::array<bool,3> output_mask) {
    static auto op = create_native_layer_norm_backward_typed_handle();
    return op.call(grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask);
}

// aten::native_layer_norm_backward(Tensor grad_out, Tensor input, int[] normalized_shape, Tensor mean, Tensor rstd, Tensor? weight, Tensor? bias, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_layer_norm_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_out, const at::Tensor & input, at::IntArrayRef normalized_shape, const at::Tensor & mean, const at::Tensor & rstd, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, ::std::array<bool,3> output_mask) {
    static auto op = create_native_layer_norm_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nan_to_num_out, name, "aten::nan_to_num")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nan_to_num_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nan_to_num_out, schema_str, "nan_to_num.out(Tensor self, float? nan=None, float? posinf=None, float? neginf=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::nan_to_num.out(Tensor self, float? nan=None, float? posinf=None, float? neginf=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<nan_to_num_out::schema> create_nan_to_num_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nan_to_num_out::name, nan_to_num_out::overload_name)
      .typed<nan_to_num_out::schema>();
}

// aten::nan_to_num.out(Tensor self, float? nan=None, float? posinf=None, float? neginf=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & nan_to_num_out::call(const at::Tensor & self, c10::optional<double> nan, c10::optional<double> posinf, c10::optional<double> neginf, at::Tensor & out) {
    static auto op = create_nan_to_num_out_typed_handle();
    return op.call(self, nan, posinf, neginf, out);
}

// aten::nan_to_num.out(Tensor self, float? nan=None, float? posinf=None, float? neginf=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & nan_to_num_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<double> nan, c10::optional<double> posinf, c10::optional<double> neginf, at::Tensor & out) {
    static auto op = create_nan_to_num_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, nan, posinf, neginf, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fbgemm_linear_fp16_weight, name, "aten::fbgemm_linear_fp16_weight")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fbgemm_linear_fp16_weight, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fbgemm_linear_fp16_weight, schema_str, "fbgemm_linear_fp16_weight(Tensor input, Tensor packed_weight, Tensor bias) -> Tensor")

// aten::fbgemm_linear_fp16_weight(Tensor input, Tensor packed_weight, Tensor bias) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fbgemm_linear_fp16_weight::schema> create_fbgemm_linear_fp16_weight_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fbgemm_linear_fp16_weight::name, fbgemm_linear_fp16_weight::overload_name)
      .typed<fbgemm_linear_fp16_weight::schema>();
}

// aten::fbgemm_linear_fp16_weight(Tensor input, Tensor packed_weight, Tensor bias) -> Tensor
at::Tensor fbgemm_linear_fp16_weight::call(const at::Tensor & input, const at::Tensor & packed_weight, const at::Tensor & bias) {
    static auto op = create_fbgemm_linear_fp16_weight_typed_handle();
    return op.call(input, packed_weight, bias);
}

// aten::fbgemm_linear_fp16_weight(Tensor input, Tensor packed_weight, Tensor bias) -> Tensor
at::Tensor fbgemm_linear_fp16_weight::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & packed_weight, const at::Tensor & bias) {
    static auto op = create_fbgemm_linear_fp16_weight_typed_handle();
    return op.redispatch(dispatchKeySet, input, packed_weight, bias);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fbgemm_pack_quantized_matrix, name, "aten::fbgemm_pack_quantized_matrix")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fbgemm_pack_quantized_matrix, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fbgemm_pack_quantized_matrix, schema_str, "fbgemm_pack_quantized_matrix(Tensor input) -> Tensor")

// aten::fbgemm_pack_quantized_matrix(Tensor input) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fbgemm_pack_quantized_matrix::schema> create_fbgemm_pack_quantized_matrix_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fbgemm_pack_quantized_matrix::name, fbgemm_pack_quantized_matrix::overload_name)
      .typed<fbgemm_pack_quantized_matrix::schema>();
}

// aten::fbgemm_pack_quantized_matrix(Tensor input) -> Tensor
at::Tensor fbgemm_pack_quantized_matrix::call(const at::Tensor & input) {
    static auto op = create_fbgemm_pack_quantized_matrix_typed_handle();
    return op.call(input);
}

// aten::fbgemm_pack_quantized_matrix(Tensor input) -> Tensor
at::Tensor fbgemm_pack_quantized_matrix::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input) {
    static auto op = create_fbgemm_pack_quantized_matrix_typed_handle();
    return op.redispatch(dispatchKeySet, input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fbgemm_pack_quantized_matrix_KN, name, "aten::fbgemm_pack_quantized_matrix")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fbgemm_pack_quantized_matrix_KN, overload_name, "KN")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fbgemm_pack_quantized_matrix_KN, schema_str, "fbgemm_pack_quantized_matrix.KN(Tensor input, int K, int N) -> Tensor")

// aten::fbgemm_pack_quantized_matrix.KN(Tensor input, int K, int N) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fbgemm_pack_quantized_matrix_KN::schema> create_fbgemm_pack_quantized_matrix_KN_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fbgemm_pack_quantized_matrix_KN::name, fbgemm_pack_quantized_matrix_KN::overload_name)
      .typed<fbgemm_pack_quantized_matrix_KN::schema>();
}

// aten::fbgemm_pack_quantized_matrix.KN(Tensor input, int K, int N) -> Tensor
at::Tensor fbgemm_pack_quantized_matrix_KN::call(const at::Tensor & input, int64_t K, int64_t N) {
    static auto op = create_fbgemm_pack_quantized_matrix_KN_typed_handle();
    return op.call(input, K, N);
}

// aten::fbgemm_pack_quantized_matrix.KN(Tensor input, int K, int N) -> Tensor
at::Tensor fbgemm_pack_quantized_matrix_KN::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, int64_t K, int64_t N) {
    static auto op = create_fbgemm_pack_quantized_matrix_KN_typed_handle();
    return op.redispatch(dispatchKeySet, input, K, N);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linspace_out, name, "aten::linspace")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linspace_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linspace_out, schema_str, "linspace.out(Scalar start, Scalar end, int? steps=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::linspace.out(Scalar start, Scalar end, int? steps=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<linspace_out::schema> create_linspace_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linspace_out::name, linspace_out::overload_name)
      .typed<linspace_out::schema>();
}

// aten::linspace.out(Scalar start, Scalar end, int? steps=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linspace_out::call(const at::Scalar & start, const at::Scalar & end, c10::optional<int64_t> steps, at::Tensor & out) {
    static auto op = create_linspace_out_typed_handle();
    return op.call(start, end, steps, out);
}

// aten::linspace.out(Scalar start, Scalar end, int? steps=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linspace_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Scalar & start, const at::Scalar & end, c10::optional<int64_t> steps, at::Tensor & out) {
    static auto op = create_linspace_out_typed_handle();
    return op.redispatch(dispatchKeySet, start, end, steps, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log, name, "aten::log")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log, schema_str, "log(Tensor self) -> Tensor")

// aten::log(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<log::schema> create_log_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(log::name, log::overload_name)
      .typed<log::schema>();
}

// aten::log(Tensor self) -> Tensor
at::Tensor log::call(const at::Tensor & self) {
    static auto op = create_log_typed_handle();
    return op.call(self);
}

// aten::log(Tensor self) -> Tensor
at::Tensor log::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_log_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log2, name, "aten::log2")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log2, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log2, schema_str, "log2(Tensor self) -> Tensor")

// aten::log2(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<log2::schema> create_log2_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(log2::name, log2::overload_name)
      .typed<log2::schema>();
}

// aten::log2(Tensor self) -> Tensor
at::Tensor log2::call(const at::Tensor & self) {
    static auto op = create_log2_typed_handle();
    return op.call(self);
}

// aten::log2(Tensor self) -> Tensor
at::Tensor log2::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_log2_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logaddexp, name, "aten::logaddexp")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logaddexp, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logaddexp, schema_str, "logaddexp(Tensor self, Tensor other) -> Tensor")

// aten::logaddexp(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<logaddexp::schema> create_logaddexp_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(logaddexp::name, logaddexp::overload_name)
      .typed<logaddexp::schema>();
}

// aten::logaddexp(Tensor self, Tensor other) -> Tensor
at::Tensor logaddexp::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_logaddexp_typed_handle();
    return op.call(self, other);
}

// aten::logaddexp(Tensor self, Tensor other) -> Tensor
at::Tensor logaddexp::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_logaddexp_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(xlogy__Tensor, name, "aten::xlogy_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(xlogy__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(xlogy__Tensor, schema_str, "xlogy_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::xlogy_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<xlogy__Tensor::schema> create_xlogy__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(xlogy__Tensor::name, xlogy__Tensor::overload_name)
      .typed<xlogy__Tensor::schema>();
}

// aten::xlogy_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & xlogy__Tensor::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create_xlogy__Tensor_typed_handle();
    return op.call(self, other);
}

// aten::xlogy_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & xlogy__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create_xlogy__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(xlogy_OutTensor, name, "aten::xlogy")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(xlogy_OutTensor, overload_name, "OutTensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(xlogy_OutTensor, schema_str, "xlogy.OutTensor(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::xlogy.OutTensor(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<xlogy_OutTensor::schema> create_xlogy_OutTensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(xlogy_OutTensor::name, xlogy_OutTensor::overload_name)
      .typed<xlogy_OutTensor::schema>();
}

// aten::xlogy.OutTensor(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & xlogy_OutTensor::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_xlogy_OutTensor_typed_handle();
    return op.call(self, other, out);
}

// aten::xlogy.OutTensor(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & xlogy_OutTensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_xlogy_OutTensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(xlogy_OutScalar_Self, name, "aten::xlogy")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(xlogy_OutScalar_Self, overload_name, "OutScalar_Self")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(xlogy_OutScalar_Self, schema_str, "xlogy.OutScalar_Self(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::xlogy.OutScalar_Self(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<xlogy_OutScalar_Self::schema> create_xlogy_OutScalar_Self_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(xlogy_OutScalar_Self::name, xlogy_OutScalar_Self::overload_name)
      .typed<xlogy_OutScalar_Self::schema>();
}

// aten::xlogy.OutScalar_Self(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & xlogy_OutScalar_Self::call(const at::Scalar & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_xlogy_OutScalar_Self_typed_handle();
    return op.call(self, other, out);
}

// aten::xlogy.OutScalar_Self(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & xlogy_OutScalar_Self::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Scalar & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_xlogy_OutScalar_Self_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(xlogy_OutScalar_Other, name, "aten::xlogy")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(xlogy_OutScalar_Other, overload_name, "OutScalar_Other")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(xlogy_OutScalar_Other, schema_str, "xlogy.OutScalar_Other(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::xlogy.OutScalar_Other(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<xlogy_OutScalar_Other::schema> create_xlogy_OutScalar_Other_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(xlogy_OutScalar_Other::name, xlogy_OutScalar_Other::overload_name)
      .typed<xlogy_OutScalar_Other::schema>();
}

// aten::xlogy.OutScalar_Other(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & xlogy_OutScalar_Other::call(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_xlogy_OutScalar_Other_typed_handle();
    return op.call(self, other, out);
}

// aten::xlogy.OutScalar_Other(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & xlogy_OutScalar_Other::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_xlogy_OutScalar_Other_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logspace, name, "aten::logspace")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logspace, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logspace, schema_str, "logspace(Scalar start, Scalar end, int? steps=None, float base=10.0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::logspace(Scalar start, Scalar end, int? steps=None, float base=10.0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<logspace::schema> create_logspace_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(logspace::name, logspace::overload_name)
      .typed<logspace::schema>();
}

// aten::logspace(Scalar start, Scalar end, int? steps=None, float base=10.0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor logspace::call(const at::Scalar & start, const at::Scalar & end, c10::optional<int64_t> steps, double base, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_logspace_typed_handle();
    return op.call(start, end, steps, base, dtype, layout, device, pin_memory);
}

// aten::logspace(Scalar start, Scalar end, int? steps=None, float base=10.0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor logspace::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Scalar & start, const at::Scalar & end, c10::optional<int64_t> steps, double base, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_logspace_typed_handle();
    return op.redispatch(dispatchKeySet, start, end, steps, base, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logspace_out, name, "aten::logspace")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logspace_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logspace_out, schema_str, "logspace.out(Scalar start, Scalar end, int? steps=None, float base=10.0, *, Tensor(a!) out) -> Tensor(a!)")

// aten::logspace.out(Scalar start, Scalar end, int? steps=None, float base=10.0, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<logspace_out::schema> create_logspace_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(logspace_out::name, logspace_out::overload_name)
      .typed<logspace_out::schema>();
}

// aten::logspace.out(Scalar start, Scalar end, int? steps=None, float base=10.0, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & logspace_out::call(const at::Scalar & start, const at::Scalar & end, c10::optional<int64_t> steps, double base, at::Tensor & out) {
    static auto op = create_logspace_out_typed_handle();
    return op.call(start, end, steps, base, out);
}

// aten::logspace.out(Scalar start, Scalar end, int? steps=None, float base=10.0, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & logspace_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Scalar & start, const at::Scalar & end, c10::optional<int64_t> steps, double base, at::Tensor & out) {
    static auto op = create_logspace_out_typed_handle();
    return op.redispatch(dispatchKeySet, start, end, steps, base, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logcumsumexp_dimname, name, "aten::logcumsumexp")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logcumsumexp_dimname, overload_name, "dimname")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logcumsumexp_dimname, schema_str, "logcumsumexp.dimname(Tensor self, Dimname dim) -> Tensor")

// aten::logcumsumexp.dimname(Tensor self, Dimname dim) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<logcumsumexp_dimname::schema> create_logcumsumexp_dimname_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(logcumsumexp_dimname::name, logcumsumexp_dimname::overload_name)
      .typed<logcumsumexp_dimname::schema>();
}

// aten::logcumsumexp.dimname(Tensor self, Dimname dim) -> Tensor
at::Tensor logcumsumexp_dimname::call(const at::Tensor & self, at::Dimname dim) {
    static auto op = create_logcumsumexp_dimname_typed_handle();
    return op.call(self, dim);
}

// aten::logcumsumexp.dimname(Tensor self, Dimname dim) -> Tensor
at::Tensor logcumsumexp_dimname::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim) {
    static auto op = create_logcumsumexp_dimname_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(matrix_rank_tol, name, "aten::matrix_rank")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(matrix_rank_tol, overload_name, "tol")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(matrix_rank_tol, schema_str, "matrix_rank.tol(Tensor self, float tol, bool symmetric=False) -> Tensor")

// aten::matrix_rank.tol(Tensor self, float tol, bool symmetric=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<matrix_rank_tol::schema> create_matrix_rank_tol_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(matrix_rank_tol::name, matrix_rank_tol::overload_name)
      .typed<matrix_rank_tol::schema>();
}

// aten::matrix_rank.tol(Tensor self, float tol, bool symmetric=False) -> Tensor
at::Tensor matrix_rank_tol::call(const at::Tensor & self, double tol, bool symmetric) {
    static auto op = create_matrix_rank_tol_typed_handle();
    return op.call(self, tol, symmetric);
}

// aten::matrix_rank.tol(Tensor self, float tol, bool symmetric=False) -> Tensor
at::Tensor matrix_rank_tol::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double tol, bool symmetric) {
    static auto op = create_matrix_rank_tol_typed_handle();
    return op.redispatch(dispatchKeySet, self, tol, symmetric);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(matrix_power, name, "aten::matrix_power")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(matrix_power, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(matrix_power, schema_str, "matrix_power(Tensor self, int n) -> Tensor")

// aten::matrix_power(Tensor self, int n) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<matrix_power::schema> create_matrix_power_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(matrix_power::name, matrix_power::overload_name)
      .typed<matrix_power::schema>();
}

// aten::matrix_power(Tensor self, int n) -> Tensor
at::Tensor matrix_power::call(const at::Tensor & self, int64_t n) {
    static auto op = create_matrix_power_typed_handle();
    return op.call(self, n);
}

// aten::matrix_power(Tensor self, int n) -> Tensor
at::Tensor matrix_power::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t n) {
    static auto op = create_matrix_power_typed_handle();
    return op.redispatch(dispatchKeySet, self, n);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_dim, name, "aten::max")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_dim, overload_name, "dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_dim, schema_str, "max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)")

// aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
static C10_NOINLINE c10::TypedOperatorHandle<max_dim::schema> create_max_dim_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(max_dim::name, max_dim::overload_name)
      .typed<max_dim::schema>();
}

// aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> max_dim::call(const at::Tensor & self, int64_t dim, bool keepdim) {
    static auto op = create_max_dim_typed_handle();
    return op.call(self, dim, keepdim);
}

// aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> max_dim::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool keepdim) {
    static auto op = create_max_dim_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_dim_max, name, "aten::max")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_dim_max, overload_name, "dim_max")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_dim_max, schema_str, "max.dim_max(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) -> (Tensor(a!) values, Tensor(b!) indices)")

// aten::max.dim_max(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) -> (Tensor(a!) values, Tensor(b!) indices)
static C10_NOINLINE c10::TypedOperatorHandle<max_dim_max::schema> create_max_dim_max_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(max_dim_max::name, max_dim_max::overload_name)
      .typed<max_dim_max::schema>();
}

// aten::max.dim_max(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> max_dim_max::call(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & max, at::Tensor & max_values) {
    static auto op = create_max_dim_max_typed_handle();
    return op.call(self, dim, keepdim, max, max_values);
}

// aten::max.dim_max(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> max_dim_max::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & max, at::Tensor & max_values) {
    static auto op = create_max_dim_max_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, max, max_values);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_max_pool3d, name, "aten::mkldnn_max_pool3d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_max_pool3d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_max_pool3d, schema_str, "mkldnn_max_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> Tensor")

// aten::mkldnn_max_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<mkldnn_max_pool3d::schema> create_mkldnn_max_pool3d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mkldnn_max_pool3d::name, mkldnn_max_pool3d::overload_name)
      .typed<mkldnn_max_pool3d::schema>();
}

// aten::mkldnn_max_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> Tensor
at::Tensor mkldnn_max_pool3d::call(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
    static auto op = create_mkldnn_max_pool3d_typed_handle();
    return op.call(self, kernel_size, stride, padding, dilation, ceil_mode);
}

// aten::mkldnn_max_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> Tensor
at::Tensor mkldnn_max_pool3d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
    static auto op = create_mkldnn_max_pool3d_typed_handle();
    return op.redispatch(dispatchKeySet, self, kernel_size, stride, padding, dilation, ceil_mode);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(median_names_dim, name, "aten::median")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(median_names_dim, overload_name, "names_dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(median_names_dim, schema_str, "median.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)")

// aten::median.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
static C10_NOINLINE c10::TypedOperatorHandle<median_names_dim::schema> create_median_names_dim_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(median_names_dim::name, median_names_dim::overload_name)
      .typed<median_names_dim::schema>();
}

// aten::median.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> median_names_dim::call(const at::Tensor & self, at::Dimname dim, bool keepdim) {
    static auto op = create_median_names_dim_typed_handle();
    return op.call(self, dim, keepdim);
}

// aten::median.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> median_names_dim::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, bool keepdim) {
    static auto op = create_median_names_dim_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(median_names_dim_values, name, "aten::median")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(median_names_dim_values, overload_name, "names_dim_values")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(median_names_dim_values, schema_str, "median.names_dim_values(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)")

// aten::median.names_dim_values(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
static C10_NOINLINE c10::TypedOperatorHandle<median_names_dim_values::schema> create_median_names_dim_values_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(median_names_dim_values::name, median_names_dim_values::overload_name)
      .typed<median_names_dim_values::schema>();
}

// aten::median.names_dim_values(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> median_names_dim_values::call(const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
    static auto op = create_median_names_dim_values_typed_handle();
    return op.call(self, dim, keepdim, values, indices);
}

// aten::median.names_dim_values(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> median_names_dim_values::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
    static auto op = create_median_names_dim_values_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, values, indices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanmedian_dim_values, name, "aten::nanmedian")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanmedian_dim_values, overload_name, "dim_values")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanmedian_dim_values, schema_str, "nanmedian.dim_values(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)")

// aten::nanmedian.dim_values(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
static C10_NOINLINE c10::TypedOperatorHandle<nanmedian_dim_values::schema> create_nanmedian_dim_values_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nanmedian_dim_values::name, nanmedian_dim_values::overload_name)
      .typed<nanmedian_dim_values::schema>();
}

// aten::nanmedian.dim_values(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> nanmedian_dim_values::call(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
    static auto op = create_nanmedian_dim_values_typed_handle();
    return op.call(self, dim, keepdim, values, indices);
}

// aten::nanmedian.dim_values(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> nanmedian_dim_values::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
    static auto op = create_nanmedian_dim_values_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, values, indices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_convolution_backward_input, name, "aten::mkldnn_convolution_backward_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_convolution_backward_input, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_convolution_backward_input, schema_str, "mkldnn_convolution_backward_input(int[] self_size, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool bias_defined) -> Tensor")

// aten::mkldnn_convolution_backward_input(int[] self_size, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool bias_defined) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<mkldnn_convolution_backward_input::schema> create_mkldnn_convolution_backward_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mkldnn_convolution_backward_input::name, mkldnn_convolution_backward_input::overload_name)
      .typed<mkldnn_convolution_backward_input::schema>();
}

// aten::mkldnn_convolution_backward_input(int[] self_size, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool bias_defined) -> Tensor
at::Tensor mkldnn_convolution_backward_input::call(at::IntArrayRef self_size, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool bias_defined) {
    static auto op = create_mkldnn_convolution_backward_input_typed_handle();
    return op.call(self_size, grad_output, weight, padding, stride, dilation, groups, bias_defined);
}

// aten::mkldnn_convolution_backward_input(int[] self_size, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool bias_defined) -> Tensor
at::Tensor mkldnn_convolution_backward_input::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef self_size, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool bias_defined) {
    static auto op = create_mkldnn_convolution_backward_input_typed_handle();
    return op.redispatch(dispatchKeySet, self_size, grad_output, weight, padding, stride, dilation, groups, bias_defined);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_convolution, name, "aten::miopen_convolution")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_convolution, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_convolution, schema_str, "miopen_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor")

// aten::miopen_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<miopen_convolution::schema> create_miopen_convolution_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(miopen_convolution::name, miopen_convolution::overload_name)
      .typed<miopen_convolution::schema>();
}

// aten::miopen_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
at::Tensor miopen_convolution::call(const at::Tensor & self, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
    static auto op = create_miopen_convolution_typed_handle();
    return op.call(self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
}

// aten::miopen_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
at::Tensor miopen_convolution::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
    static auto op = create_miopen_convolution_typed_handle();
    return op.redispatch(dispatchKeySet, self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_depthwise_convolution_backward_input, name, "aten::miopen_depthwise_convolution_backward_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_depthwise_convolution_backward_input, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_depthwise_convolution_backward_input, schema_str, "miopen_depthwise_convolution_backward_input(int[] self_size, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor")

// aten::miopen_depthwise_convolution_backward_input(int[] self_size, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<miopen_depthwise_convolution_backward_input::schema> create_miopen_depthwise_convolution_backward_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(miopen_depthwise_convolution_backward_input::name, miopen_depthwise_convolution_backward_input::overload_name)
      .typed<miopen_depthwise_convolution_backward_input::schema>();
}

// aten::miopen_depthwise_convolution_backward_input(int[] self_size, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
at::Tensor miopen_depthwise_convolution_backward_input::call(at::IntArrayRef self_size, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
    static auto op = create_miopen_depthwise_convolution_backward_input_typed_handle();
    return op.call(self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
}

// aten::miopen_depthwise_convolution_backward_input(int[] self_size, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
at::Tensor miopen_depthwise_convolution_backward_input::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef self_size, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
    static auto op = create_miopen_depthwise_convolution_backward_input_typed_handle();
    return op.redispatch(dispatchKeySet, self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_rnn, name, "aten::miopen_rnn")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_rnn, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_rnn, schema_str, "miopen_rnn(Tensor input, Tensor[] weight, int weight_stride0, Tensor hx, Tensor? cx, int mode, int hidden_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state) -> (Tensor, Tensor, Tensor, Tensor, Tensor)")

// aten::miopen_rnn(Tensor input, Tensor[] weight, int weight_stride0, Tensor hx, Tensor? cx, int mode, int hidden_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<miopen_rnn::schema> create_miopen_rnn_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(miopen_rnn::name, miopen_rnn::overload_name)
      .typed<miopen_rnn::schema>();
}

// aten::miopen_rnn(Tensor input, Tensor[] weight, int weight_stride0, Tensor hx, Tensor? cx, int mode, int hidden_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> miopen_rnn::call(const at::Tensor & input, at::TensorList weight, int64_t weight_stride0, const at::Tensor & hx, const c10::optional<at::Tensor> & cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, at::IntArrayRef batch_sizes, const c10::optional<at::Tensor> & dropout_state) {
    static auto op = create_miopen_rnn_typed_handle();
    return op.call(input, weight, weight_stride0, hx, cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state);
}

// aten::miopen_rnn(Tensor input, Tensor[] weight, int weight_stride0, Tensor hx, Tensor? cx, int mode, int hidden_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> miopen_rnn::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, at::TensorList weight, int64_t weight_stride0, const at::Tensor & hx, const c10::optional<at::Tensor> & cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, at::IntArrayRef batch_sizes, const c10::optional<at::Tensor> & dropout_state) {
    static auto op = create_miopen_rnn_typed_handle();
    return op.redispatch(dispatchKeySet, input, weight, weight_stride0, hx, cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_mm, name, "aten::_sparse_mm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_mm, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_mm, schema_str, "_sparse_mm(Tensor sparse, Tensor dense) -> Tensor")

// aten::_sparse_mm(Tensor sparse, Tensor dense) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_sparse_mm::schema> create__sparse_mm_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_sparse_mm::name, _sparse_mm::overload_name)
      .typed<_sparse_mm::schema>();
}

// aten::_sparse_mm(Tensor sparse, Tensor dense) -> Tensor
at::Tensor _sparse_mm::call(const at::Tensor & sparse, const at::Tensor & dense) {
    static auto op = create__sparse_mm_typed_handle();
    return op.call(sparse, dense);
}

// aten::_sparse_mm(Tensor sparse, Tensor dense) -> Tensor
at::Tensor _sparse_mm::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & sparse, const at::Tensor & dense) {
    static auto op = create__sparse_mm_typed_handle();
    return op.redispatch(dispatchKeySet, sparse, dense);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_sparse_matmul, name, "aten::_sparse_sparse_matmul")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_sparse_matmul, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_sparse_matmul, schema_str, "_sparse_sparse_matmul(Tensor self, Tensor other) -> Tensor")

// aten::_sparse_sparse_matmul(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_sparse_sparse_matmul::schema> create__sparse_sparse_matmul_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_sparse_sparse_matmul::name, _sparse_sparse_matmul::overload_name)
      .typed<_sparse_sparse_matmul::schema>();
}

// aten::_sparse_sparse_matmul(Tensor self, Tensor other) -> Tensor
at::Tensor _sparse_sparse_matmul::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create__sparse_sparse_matmul_typed_handle();
    return op.call(self, other);
}

// aten::_sparse_sparse_matmul(Tensor self, Tensor other) -> Tensor
at::Tensor _sparse_sparse_matmul::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create__sparse_sparse_matmul_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mode_dimname, name, "aten::mode")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mode_dimname, overload_name, "dimname")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mode_dimname, schema_str, "mode.dimname(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)")

// aten::mode.dimname(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
static C10_NOINLINE c10::TypedOperatorHandle<mode_dimname::schema> create_mode_dimname_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mode_dimname::name, mode_dimname::overload_name)
      .typed<mode_dimname::schema>();
}

// aten::mode.dimname(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> mode_dimname::call(const at::Tensor & self, at::Dimname dim, bool keepdim) {
    static auto op = create_mode_dimname_typed_handle();
    return op.call(self, dim, keepdim);
}

// aten::mode.dimname(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> mode_dimname::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, bool keepdim) {
    static auto op = create_mode_dimname_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mul_out, name, "aten::mul")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mul_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mul_out, schema_str, "mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<mul_out::schema> create_mul_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mul_out::name, mul_out::overload_name)
      .typed<mul_out::schema>();
}

// aten::mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & mul_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_mul_out_typed_handle();
    return op.call(self, other, out);
}

// aten::mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & mul_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_mul_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mul__Scalar, name, "aten::mul_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mul__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mul__Scalar, schema_str, "mul_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)")

// aten::mul_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<mul__Scalar::schema> create_mul__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mul__Scalar::name, mul__Scalar::overload_name)
      .typed<mul__Scalar::schema>();
}

// aten::mul_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & mul__Scalar::call(at::Tensor & self, const at::Scalar & other) {
    static auto op = create_mul__Scalar_typed_handle();
    return op.call(self, other);
}

// aten::mul_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & mul__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
    static auto op = create_mul__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multiply_Scalar, name, "aten::multiply")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multiply_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multiply_Scalar, schema_str, "multiply.Scalar(Tensor self, Scalar other) -> Tensor")

// aten::multiply.Scalar(Tensor self, Scalar other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<multiply_Scalar::schema> create_multiply_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(multiply_Scalar::name, multiply_Scalar::overload_name)
      .typed<multiply_Scalar::schema>();
}

// aten::multiply.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor multiply_Scalar::call(const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_multiply_Scalar_typed_handle();
    return op.call(self, other);
}

// aten::multiply.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor multiply_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_multiply_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mv_out, name, "aten::mv")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mv_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mv_out, schema_str, "mv.out(Tensor self, Tensor vec, *, Tensor(a!) out) -> Tensor(a!)")

// aten::mv.out(Tensor self, Tensor vec, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<mv_out::schema> create_mv_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mv_out::name, mv_out::overload_name)
      .typed<mv_out::schema>();
}

// aten::mv.out(Tensor self, Tensor vec, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & mv_out::call(const at::Tensor & self, const at::Tensor & vec, at::Tensor & out) {
    static auto op = create_mv_out_typed_handle();
    return op.call(self, vec, out);
}

// aten::mv.out(Tensor self, Tensor vec, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & mv_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & vec, at::Tensor & out) {
    static auto op = create_mv_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, vec, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(native_batch_norm_out, name, "aten::native_batch_norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(native_batch_norm_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(native_batch_norm_out, schema_str, "native_batch_norm.out(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, *, Tensor(a!) out, Tensor(b!) save_mean, Tensor(c!) save_invstd) -> (Tensor(a!), Tensor(b!), Tensor(c!))")

// aten::native_batch_norm.out(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, *, Tensor(a!) out, Tensor(b!) save_mean, Tensor(c!) save_invstd) -> (Tensor(a!), Tensor(b!), Tensor(c!))
static C10_NOINLINE c10::TypedOperatorHandle<native_batch_norm_out::schema> create_native_batch_norm_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(native_batch_norm_out::name, native_batch_norm_out::overload_name)
      .typed<native_batch_norm_out::schema>();
}

// aten::native_batch_norm.out(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, *, Tensor(a!) out, Tensor(b!) save_mean, Tensor(c!) save_invstd) -> (Tensor(a!), Tensor(b!), Tensor(c!))
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> native_batch_norm_out::call(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double momentum, double eps, at::Tensor & out, at::Tensor & save_mean, at::Tensor & save_invstd) {
    static auto op = create_native_batch_norm_out_typed_handle();
    return op.call(input, weight, bias, running_mean, running_var, training, momentum, eps, out, save_mean, save_invstd);
}

// aten::native_batch_norm.out(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, *, Tensor(a!) out, Tensor(b!) save_mean, Tensor(c!) save_invstd) -> (Tensor(a!), Tensor(b!), Tensor(c!))
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> native_batch_norm_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double momentum, double eps, at::Tensor & out, at::Tensor & save_mean, at::Tensor & save_invstd) {
    static auto op = create_native_batch_norm_out_typed_handle();
    return op.redispatch(dispatchKeySet, input, weight, bias, running_mean, running_var, training, momentum, eps, out, save_mean, save_invstd);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(batch_norm_elemt_out, name, "aten::batch_norm_elemt")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(batch_norm_elemt_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(batch_norm_elemt_out, schema_str, "batch_norm_elemt.out(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor invstd, float eps, *, Tensor(a!) out) -> Tensor(a!)")

// aten::batch_norm_elemt.out(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor invstd, float eps, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<batch_norm_elemt_out::schema> create_batch_norm_elemt_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(batch_norm_elemt_out::name, batch_norm_elemt_out::overload_name)
      .typed<batch_norm_elemt_out::schema>();
}

// aten::batch_norm_elemt.out(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor invstd, float eps, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & batch_norm_elemt_out::call(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const at::Tensor & mean, const at::Tensor & invstd, double eps, at::Tensor & out) {
    static auto op = create_batch_norm_elemt_out_typed_handle();
    return op.call(input, weight, bias, mean, invstd, eps, out);
}

// aten::batch_norm_elemt.out(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor invstd, float eps, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & batch_norm_elemt_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const at::Tensor & mean, const at::Tensor & invstd, double eps, at::Tensor & out) {
    static auto op = create_batch_norm_elemt_out_typed_handle();
    return op.redispatch(dispatchKeySet, input, weight, bias, mean, invstd, eps, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(batch_norm_update_stats, name, "aten::batch_norm_update_stats")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(batch_norm_update_stats, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(batch_norm_update_stats, schema_str, "batch_norm_update_stats(Tensor input, Tensor? running_mean, Tensor? running_var, float momentum) -> (Tensor, Tensor)")

// aten::batch_norm_update_stats(Tensor input, Tensor? running_mean, Tensor? running_var, float momentum) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<batch_norm_update_stats::schema> create_batch_norm_update_stats_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(batch_norm_update_stats::name, batch_norm_update_stats::overload_name)
      .typed<batch_norm_update_stats::schema>();
}

// aten::batch_norm_update_stats(Tensor input, Tensor? running_mean, Tensor? running_var, float momentum) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> batch_norm_update_stats::call(const at::Tensor & input, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, double momentum) {
    static auto op = create_batch_norm_update_stats_typed_handle();
    return op.call(input, running_mean, running_var, momentum);
}

// aten::batch_norm_update_stats(Tensor input, Tensor? running_mean, Tensor? running_var, float momentum) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> batch_norm_update_stats::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, double momentum) {
    static auto op = create_batch_norm_update_stats_typed_handle();
    return op.redispatch(dispatchKeySet, input, running_mean, running_var, momentum);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_nnpack_available, name, "aten::_nnpack_available")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_nnpack_available, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_nnpack_available, schema_str, "_nnpack_available() -> bool")

// aten::_nnpack_available() -> bool
static C10_NOINLINE c10::TypedOperatorHandle<_nnpack_available::schema> create__nnpack_available_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_nnpack_available::name, _nnpack_available::overload_name)
      .typed<_nnpack_available::schema>();
}

// aten::_nnpack_available() -> bool
bool _nnpack_available::call() {
    static auto op = create__nnpack_available_typed_handle();
    return op.call();
}

// aten::_nnpack_available() -> bool
bool _nnpack_available::redispatch(c10::DispatchKeySet dispatchKeySet) {
    static auto op = create__nnpack_available_typed_handle();
    return op.redispatch(dispatchKeySet);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_nnpack_spatial_convolution_backward_input, name, "aten::_nnpack_spatial_convolution_backward_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_nnpack_spatial_convolution_backward_input, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_nnpack_spatial_convolution_backward_input, schema_str, "_nnpack_spatial_convolution_backward_input(Tensor input, Tensor grad_output, Tensor weight, int[2] padding) -> Tensor")

// aten::_nnpack_spatial_convolution_backward_input(Tensor input, Tensor grad_output, Tensor weight, int[2] padding) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_nnpack_spatial_convolution_backward_input::schema> create__nnpack_spatial_convolution_backward_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_nnpack_spatial_convolution_backward_input::name, _nnpack_spatial_convolution_backward_input::overload_name)
      .typed<_nnpack_spatial_convolution_backward_input::schema>();
}

// aten::_nnpack_spatial_convolution_backward_input(Tensor input, Tensor grad_output, Tensor weight, int[2] padding) -> Tensor
at::Tensor _nnpack_spatial_convolution_backward_input::call(const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding) {
    static auto op = create__nnpack_spatial_convolution_backward_input_typed_handle();
    return op.call(input, grad_output, weight, padding);
}

// aten::_nnpack_spatial_convolution_backward_input(Tensor input, Tensor grad_output, Tensor weight, int[2] padding) -> Tensor
at::Tensor _nnpack_spatial_convolution_backward_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding) {
    static auto op = create__nnpack_spatial_convolution_backward_input_typed_handle();
    return op.redispatch(dispatchKeySet, input, grad_output, weight, padding);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_nnpack_spatial_convolution_backward_weight, name, "aten::_nnpack_spatial_convolution_backward_weight")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_nnpack_spatial_convolution_backward_weight, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_nnpack_spatial_convolution_backward_weight, schema_str, "_nnpack_spatial_convolution_backward_weight(Tensor input, int[] weightsize, Tensor grad_output, int[2] padding) -> Tensor")

// aten::_nnpack_spatial_convolution_backward_weight(Tensor input, int[] weightsize, Tensor grad_output, int[2] padding) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_nnpack_spatial_convolution_backward_weight::schema> create__nnpack_spatial_convolution_backward_weight_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_nnpack_spatial_convolution_backward_weight::name, _nnpack_spatial_convolution_backward_weight::overload_name)
      .typed<_nnpack_spatial_convolution_backward_weight::schema>();
}

// aten::_nnpack_spatial_convolution_backward_weight(Tensor input, int[] weightsize, Tensor grad_output, int[2] padding) -> Tensor
at::Tensor _nnpack_spatial_convolution_backward_weight::call(const at::Tensor & input, at::IntArrayRef weightsize, const at::Tensor & grad_output, at::IntArrayRef padding) {
    static auto op = create__nnpack_spatial_convolution_backward_weight_typed_handle();
    return op.call(input, weightsize, grad_output, padding);
}

// aten::_nnpack_spatial_convolution_backward_weight(Tensor input, int[] weightsize, Tensor grad_output, int[2] padding) -> Tensor
at::Tensor _nnpack_spatial_convolution_backward_weight::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, at::IntArrayRef weightsize, const at::Tensor & grad_output, at::IntArrayRef padding) {
    static auto op = create__nnpack_spatial_convolution_backward_weight_typed_handle();
    return op.redispatch(dispatchKeySet, input, weightsize, grad_output, padding);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ones_like, name, "aten::ones_like")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ones_like, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ones_like, schema_str, "ones_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor")

// aten::ones_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<ones_like::schema> create_ones_like_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(ones_like::name, ones_like::overload_name)
      .typed<ones_like::schema>();
}

// aten::ones_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
at::Tensor ones_like::call(const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create_ones_like_typed_handle();
    return op.call(self, dtype, layout, device, pin_memory, memory_format);
}

// aten::ones_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
at::Tensor ones_like::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create_ones_like_typed_handle();
    return op.redispatch(dispatchKeySet, self, dtype, layout, device, pin_memory, memory_format);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_euclidean_dist, name, "aten::_euclidean_dist")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_euclidean_dist, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_euclidean_dist, schema_str, "_euclidean_dist(Tensor x1, Tensor x2) -> Tensor")

// aten::_euclidean_dist(Tensor x1, Tensor x2) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_euclidean_dist::schema> create__euclidean_dist_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_euclidean_dist::name, _euclidean_dist::overload_name)
      .typed<_euclidean_dist::schema>();
}

// aten::_euclidean_dist(Tensor x1, Tensor x2) -> Tensor
at::Tensor _euclidean_dist::call(const at::Tensor & x1, const at::Tensor & x2) {
    static auto op = create__euclidean_dist_typed_handle();
    return op.call(x1, x2);
}

// aten::_euclidean_dist(Tensor x1, Tensor x2) -> Tensor
at::Tensor _euclidean_dist::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & x1, const at::Tensor & x2) {
    static auto op = create__euclidean_dist_typed_handle();
    return op.redispatch(dispatchKeySet, x1, x2);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cdist_backward, name, "aten::_cdist_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cdist_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cdist_backward, schema_str, "_cdist_backward(Tensor grad, Tensor x1, Tensor x2, float p, Tensor cdist) -> Tensor")

// aten::_cdist_backward(Tensor grad, Tensor x1, Tensor x2, float p, Tensor cdist) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_cdist_backward::schema> create__cdist_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_cdist_backward::name, _cdist_backward::overload_name)
      .typed<_cdist_backward::schema>();
}

// aten::_cdist_backward(Tensor grad, Tensor x1, Tensor x2, float p, Tensor cdist) -> Tensor
at::Tensor _cdist_backward::call(const at::Tensor & grad, const at::Tensor & x1, const at::Tensor & x2, double p, const at::Tensor & cdist) {
    static auto op = create__cdist_backward_typed_handle();
    return op.call(grad, x1, x2, p, cdist);
}

// aten::_cdist_backward(Tensor grad, Tensor x1, Tensor x2, float p, Tensor cdist) -> Tensor
at::Tensor _cdist_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & x1, const at::Tensor & x2, double p, const at::Tensor & cdist) {
    static auto op = create__cdist_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad, x1, x2, p, cdist);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_pdist_forward, name, "aten::_pdist_forward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_pdist_forward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_pdist_forward, schema_str, "_pdist_forward(Tensor self, float p=2) -> Tensor")

// aten::_pdist_forward(Tensor self, float p=2) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_pdist_forward::schema> create__pdist_forward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_pdist_forward::name, _pdist_forward::overload_name)
      .typed<_pdist_forward::schema>();
}

// aten::_pdist_forward(Tensor self, float p=2) -> Tensor
at::Tensor _pdist_forward::call(const at::Tensor & self, double p) {
    static auto op = create__pdist_forward_typed_handle();
    return op.call(self, p);
}

// aten::_pdist_forward(Tensor self, float p=2) -> Tensor
at::Tensor _pdist_forward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double p) {
    static auto op = create__pdist_forward_typed_handle();
    return op.redispatch(dispatchKeySet, self, p);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rad2deg, name, "aten::rad2deg")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rad2deg, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rad2deg, schema_str, "rad2deg(Tensor self) -> Tensor")

// aten::rad2deg(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<rad2deg::schema> create_rad2deg_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(rad2deg::name, rad2deg::overload_name)
      .typed<rad2deg::schema>();
}

// aten::rad2deg(Tensor self) -> Tensor
at::Tensor rad2deg::call(const at::Tensor & self) {
    static auto op = create_rad2deg_typed_handle();
    return op.call(self);
}

// aten::rad2deg(Tensor self) -> Tensor
at::Tensor rad2deg::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_rad2deg_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scalar_tensor, name, "aten::scalar_tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scalar_tensor, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scalar_tensor, schema_str, "scalar_tensor(Scalar s, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::scalar_tensor(Scalar s, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<scalar_tensor::schema> create_scalar_tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(scalar_tensor::name, scalar_tensor::overload_name)
      .typed<scalar_tensor::schema>();
}

// aten::scalar_tensor(Scalar s, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor scalar_tensor::call(const at::Scalar & s, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_scalar_tensor_typed_handle();
    return op.call(s, dtype, layout, device, pin_memory);
}

// aten::scalar_tensor(Scalar s, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor scalar_tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Scalar & s, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_scalar_tensor_typed_handle();
    return op.redispatch(dispatchKeySet, s, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rand_generator_with_names, name, "aten::rand")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rand_generator_with_names, overload_name, "generator_with_names")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rand_generator_with_names, schema_str, "rand.generator_with_names(int[] size, *, Generator? generator, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::rand.generator_with_names(int[] size, *, Generator? generator, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<rand_generator_with_names::schema> create_rand_generator_with_names_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(rand_generator_with_names::name, rand_generator_with_names::overload_name)
      .typed<rand_generator_with_names::schema>();
}

// aten::rand.generator_with_names(int[] size, *, Generator? generator, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor rand_generator_with_names::call(at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_rand_generator_with_names_typed_handle();
    return op.call(size, generator, names, dtype, layout, device, pin_memory);
}

// aten::rand.generator_with_names(int[] size, *, Generator? generator, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor rand_generator_with_names::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_rand_generator_with_names_typed_handle();
    return op.redispatch(dispatchKeySet, size, generator, names, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rand, name, "aten::rand")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rand, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rand, schema_str, "rand(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::rand(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<rand::schema> create_rand_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(rand::name, rand::overload_name)
      .typed<rand::schema>();
}

// aten::rand(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor rand::call(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_rand_typed_handle();
    return op.call(size, dtype, layout, device, pin_memory);
}

// aten::rand(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor rand::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_rand_typed_handle();
    return op.redispatch(dispatchKeySet, size, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randint, name, "aten::randint")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randint, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randint, schema_str, "randint(int high, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::randint(int high, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<randint::schema> create_randint_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(randint::name, randint::overload_name)
      .typed<randint::schema>();
}

// aten::randint(int high, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor randint::call(int64_t high, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_randint_typed_handle();
    return op.call(high, size, dtype, layout, device, pin_memory);
}

// aten::randint(int high, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor randint::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t high, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_randint_typed_handle();
    return op.redispatch(dispatchKeySet, high, size, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randint_like_low_dtype, name, "aten::randint_like")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randint_like_low_dtype, overload_name, "low_dtype")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randint_like_low_dtype, schema_str, "randint_like.low_dtype(Tensor self, int low, int high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor")

// aten::randint_like.low_dtype(Tensor self, int low, int high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<randint_like_low_dtype::schema> create_randint_like_low_dtype_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(randint_like_low_dtype::name, randint_like_low_dtype::overload_name)
      .typed<randint_like_low_dtype::schema>();
}

// aten::randint_like.low_dtype(Tensor self, int low, int high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
at::Tensor randint_like_low_dtype::call(const at::Tensor & self, int64_t low, int64_t high, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create_randint_like_low_dtype_typed_handle();
    return op.call(self, low, high, dtype, layout, device, pin_memory, memory_format);
}

// aten::randint_like.low_dtype(Tensor self, int low, int high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
at::Tensor randint_like_low_dtype::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t low, int64_t high, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create_randint_like_low_dtype_typed_handle();
    return op.redispatch(dispatchKeySet, self, low, high, dtype, layout, device, pin_memory, memory_format);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randn_out, name, "aten::randn")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randn_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randn_out, schema_str, "randn.out(int[] size, *, Tensor(a!) out) -> Tensor(a!)")

// aten::randn.out(int[] size, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<randn_out::schema> create_randn_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(randn_out::name, randn_out::overload_name)
      .typed<randn_out::schema>();
}

// aten::randn.out(int[] size, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & randn_out::call(at::IntArrayRef size, at::Tensor & out) {
    static auto op = create_randn_out_typed_handle();
    return op.call(size, out);
}

// aten::randn.out(int[] size, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & randn_out::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, at::Tensor & out) {
    static auto op = create_randn_out_typed_handle();
    return op.redispatch(dispatchKeySet, size, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randn_like, name, "aten::randn_like")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randn_like, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randn_like, schema_str, "randn_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor")

// aten::randn_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<randn_like::schema> create_randn_like_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(randn_like::name, randn_like::overload_name)
      .typed<randn_like::schema>();
}

// aten::randn_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
at::Tensor randn_like::call(const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create_randn_like_typed_handle();
    return op.call(self, dtype, layout, device, pin_memory, memory_format);
}

// aten::randn_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
at::Tensor randn_like::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create_randn_like_typed_handle();
    return op.redispatch(dispatchKeySet, self, dtype, layout, device, pin_memory, memory_format);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randperm_out, name, "aten::randperm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randperm_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randperm_out, schema_str, "randperm.out(int n, *, Tensor(a!) out) -> Tensor(a!)")

// aten::randperm.out(int n, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<randperm_out::schema> create_randperm_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(randperm_out::name, randperm_out::overload_name)
      .typed<randperm_out::schema>();
}

// aten::randperm.out(int n, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & randperm_out::call(int64_t n, at::Tensor & out) {
    static auto op = create_randperm_out_typed_handle();
    return op.call(n, out);
}

// aten::randperm.out(int n, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & randperm_out::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t n, at::Tensor & out) {
    static auto op = create_randperm_out_typed_handle();
    return op.redispatch(dispatchKeySet, n, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randperm_generator_out, name, "aten::randperm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randperm_generator_out, overload_name, "generator_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randperm_generator_out, schema_str, "randperm.generator_out(int n, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)")

// aten::randperm.generator_out(int n, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<randperm_generator_out::schema> create_randperm_generator_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(randperm_generator_out::name, randperm_generator_out::overload_name)
      .typed<randperm_generator_out::schema>();
}

// aten::randperm.generator_out(int n, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
at::Tensor & randperm_generator_out::call(int64_t n, c10::optional<at::Generator> generator, at::Tensor & out) {
    static auto op = create_randperm_generator_out_typed_handle();
    return op.call(n, generator, out);
}

// aten::randperm.generator_out(int n, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
at::Tensor & randperm_generator_out::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t n, c10::optional<at::Generator> generator, at::Tensor & out) {
    static auto op = create_randperm_generator_out_typed_handle();
    return op.redispatch(dispatchKeySet, n, generator, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(repeat, name, "aten::repeat")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(repeat, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(repeat, schema_str, "repeat(Tensor self, int[] repeats) -> Tensor")

// aten::repeat(Tensor self, int[] repeats) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<repeat::schema> create_repeat_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(repeat::name, repeat::overload_name)
      .typed<repeat::schema>();
}

// aten::repeat(Tensor self, int[] repeats) -> Tensor
at::Tensor repeat::call(const at::Tensor & self, at::IntArrayRef repeats) {
    static auto op = create_repeat_typed_handle();
    return op.call(self, repeats);
}

// aten::repeat(Tensor self, int[] repeats) -> Tensor
at::Tensor repeat::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef repeats) {
    static auto op = create_repeat_typed_handle();
    return op.redispatch(dispatchKeySet, self, repeats);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_mkldnn_reshape, name, "aten::_mkldnn_reshape")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_mkldnn_reshape, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_mkldnn_reshape, schema_str, "_mkldnn_reshape(Tensor self, int[] shape) -> Tensor")

// aten::_mkldnn_reshape(Tensor self, int[] shape) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_mkldnn_reshape::schema> create__mkldnn_reshape_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_mkldnn_reshape::name, _mkldnn_reshape::overload_name)
      .typed<_mkldnn_reshape::schema>();
}

// aten::_mkldnn_reshape(Tensor self, int[] shape) -> Tensor
at::Tensor _mkldnn_reshape::call(const at::Tensor & self, at::IntArrayRef shape) {
    static auto op = create__mkldnn_reshape_typed_handle();
    return op.call(self, shape);
}

// aten::_mkldnn_reshape(Tensor self, int[] shape) -> Tensor
at::Tensor _mkldnn_reshape::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef shape) {
    static auto op = create__mkldnn_reshape_typed_handle();
    return op.redispatch(dispatchKeySet, self, shape);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(relu_, name, "aten::relu_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(relu_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(relu_, schema_str, "relu_(Tensor(a!) self) -> Tensor(a!)")

// aten::relu_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<relu_::schema> create_relu__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(relu_::name, relu_::overload_name)
      .typed<relu_::schema>();
}

// aten::relu_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & relu_::call(at::Tensor & self) {
    static auto op = create_relu__typed_handle();
    return op.call(self);
}

// aten::relu_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & relu_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_relu__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(prelu_backward, name, "aten::prelu_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(prelu_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(prelu_backward, schema_str, "prelu_backward(Tensor grad_output, Tensor self, Tensor weight) -> (Tensor, Tensor)")

// aten::prelu_backward(Tensor grad_output, Tensor self, Tensor weight) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<prelu_backward::schema> create_prelu_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(prelu_backward::name, prelu_backward::overload_name)
      .typed<prelu_backward::schema>();
}

// aten::prelu_backward(Tensor grad_output, Tensor self, Tensor weight) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> prelu_backward::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight) {
    static auto op = create_prelu_backward_typed_handle();
    return op.call(grad_output, self, weight);
}

// aten::prelu_backward(Tensor grad_output, Tensor self, Tensor weight) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> prelu_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight) {
    static auto op = create_prelu_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, weight);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gelu_backward_grad_input, name, "aten::gelu_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gelu_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gelu_backward_grad_input, schema_str, "gelu_backward.grad_input(Tensor grad, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::gelu_backward.grad_input(Tensor grad, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<gelu_backward_grad_input::schema> create_gelu_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(gelu_backward_grad_input::name, gelu_backward_grad_input::overload_name)
      .typed<gelu_backward_grad_input::schema>();
}

// aten::gelu_backward.grad_input(Tensor grad, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & gelu_backward_grad_input::call(const at::Tensor & grad, const at::Tensor & self, at::Tensor & grad_input) {
    static auto op = create_gelu_backward_grad_input_typed_handle();
    return op.call(grad, self, grad_input);
}

// aten::gelu_backward.grad_input(Tensor grad, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & gelu_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & self, at::Tensor & grad_input) {
    static auto op = create_gelu_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad, self, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardshrink_out, name, "aten::hardshrink")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardshrink_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardshrink_out, schema_str, "hardshrink.out(Tensor self, Scalar lambd=0.5, *, Tensor(a!) out) -> Tensor(a!)")

// aten::hardshrink.out(Tensor self, Scalar lambd=0.5, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<hardshrink_out::schema> create_hardshrink_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(hardshrink_out::name, hardshrink_out::overload_name)
      .typed<hardshrink_out::schema>();
}

// aten::hardshrink.out(Tensor self, Scalar lambd=0.5, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & hardshrink_out::call(const at::Tensor & self, const at::Scalar & lambd, at::Tensor & out) {
    static auto op = create_hardshrink_out_typed_handle();
    return op.call(self, lambd, out);
}

// aten::hardshrink.out(Tensor self, Scalar lambd=0.5, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & hardshrink_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & lambd, at::Tensor & out) {
    static auto op = create_hardshrink_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, lambd, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rsqrt, name, "aten::rsqrt")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rsqrt, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rsqrt, schema_str, "rsqrt(Tensor self) -> Tensor")

// aten::rsqrt(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<rsqrt::schema> create_rsqrt_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(rsqrt::name, rsqrt::overload_name)
      .typed<rsqrt::schema>();
}

// aten::rsqrt(Tensor self) -> Tensor
at::Tensor rsqrt::call(const at::Tensor & self) {
    static auto op = create_rsqrt_typed_handle();
    return op.call(self);
}

// aten::rsqrt(Tensor self) -> Tensor
at::Tensor rsqrt::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_rsqrt_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mish_out, name, "aten::mish")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mish_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mish_out, schema_str, "mish.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::mish.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<mish_out::schema> create_mish_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mish_out::name, mish_out::overload_name)
      .typed<mish_out::schema>();
}

// aten::mish.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & mish_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_mish_out_typed_handle();
    return op.call(self, out);
}

// aten::mish.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & mish_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_mish_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sigmoid_out, name, "aten::sigmoid")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sigmoid_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sigmoid_out, schema_str, "sigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::sigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<sigmoid_out::schema> create_sigmoid_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sigmoid_out::name, sigmoid_out::overload_name)
      .typed<sigmoid_out::schema>();
}

// aten::sigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & sigmoid_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_sigmoid_out_typed_handle();
    return op.call(self, out);
}

// aten::sigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & sigmoid_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_sigmoid_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sin_, name, "aten::sin_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sin_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sin_, schema_str, "sin_(Tensor(a!) self) -> Tensor(a!)")

// aten::sin_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<sin_::schema> create_sin__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sin_::name, sin_::overload_name)
      .typed<sin_::schema>();
}

// aten::sin_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & sin_::call(at::Tensor & self) {
    static auto op = create_sin__typed_handle();
    return op.call(self);
}

// aten::sin_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & sin_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_sin__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sinc_out, name, "aten::sinc")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sinc_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sinc_out, schema_str, "sinc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::sinc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<sinc_out::schema> create_sinc_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sinc_out::name, sinc_out::overload_name)
      .typed<sinc_out::schema>();
}

// aten::sinc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & sinc_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_sinc_out_typed_handle();
    return op.call(self, out);
}

// aten::sinc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & sinc_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_sinc_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(size_int, name, "aten::size")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(size_int, overload_name, "int")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(size_int, schema_str, "size.int(Tensor self, int dim) -> int")

// aten::size.int(Tensor self, int dim) -> int
static C10_NOINLINE c10::TypedOperatorHandle<size_int::schema> create_size_int_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(size_int::name, size_int::overload_name)
      .typed<size_int::schema>();
}

// aten::size.int(Tensor self, int dim) -> int
int64_t size_int::call(const at::Tensor & self, int64_t dim) {
    static auto op = create_size_int_typed_handle();
    return op.call(self, dim);
}

// aten::size.int(Tensor self, int dim) -> int
int64_t size_int::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim) {
    static auto op = create_size_int_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slogdet, name, "aten::slogdet")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slogdet, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slogdet, schema_str, "slogdet(Tensor self) -> (Tensor sign, Tensor logabsdet)")

// aten::slogdet(Tensor self) -> (Tensor sign, Tensor logabsdet)
static C10_NOINLINE c10::TypedOperatorHandle<slogdet::schema> create_slogdet_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(slogdet::name, slogdet::overload_name)
      .typed<slogdet::schema>();
}

// aten::slogdet(Tensor self) -> (Tensor sign, Tensor logabsdet)
::std::tuple<at::Tensor,at::Tensor> slogdet::call(const at::Tensor & self) {
    static auto op = create_slogdet_typed_handle();
    return op.call(self);
}

// aten::slogdet(Tensor self) -> (Tensor sign, Tensor logabsdet)
::std::tuple<at::Tensor,at::Tensor> slogdet::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_slogdet_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(softmax_int, name, "aten::softmax")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(softmax_int, overload_name, "int")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(softmax_int, schema_str, "softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor")

// aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<softmax_int::schema> create_softmax_int_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(softmax_int::name, softmax_int::overload_name)
      .typed<softmax_int::schema>();
}

// aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
at::Tensor softmax_int::call(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_softmax_int_typed_handle();
    return op.call(self, dim, dtype);
}

// aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
at::Tensor softmax_int::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_softmax_int_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_softmax_out, name, "aten::_softmax")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_softmax_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_softmax_out, schema_str, "_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)")

// aten::_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<_softmax_out::schema> create__softmax_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_softmax_out::name, _softmax_out::overload_name)
      .typed<_softmax_out::schema>();
}

// aten::_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & _softmax_out::call(const at::Tensor & self, int64_t dim, bool half_to_float, at::Tensor & out) {
    static auto op = create__softmax_out_typed_handle();
    return op.call(self, dim, half_to_float, out);
}

// aten::_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & _softmax_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool half_to_float, at::Tensor & out) {
    static auto op = create__softmax_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, half_to_float, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hsplit_int, name, "aten::hsplit")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hsplit_int, overload_name, "int")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hsplit_int, schema_str, "hsplit.int(Tensor(a) self, int sections) -> Tensor(a)[]")

// aten::hsplit.int(Tensor(a) self, int sections) -> Tensor(a)[]
static C10_NOINLINE c10::TypedOperatorHandle<hsplit_int::schema> create_hsplit_int_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(hsplit_int::name, hsplit_int::overload_name)
      .typed<hsplit_int::schema>();
}

// aten::hsplit.int(Tensor(a) self, int sections) -> Tensor(a)[]
::std::vector<at::Tensor> hsplit_int::call(const at::Tensor & self, int64_t sections) {
    static auto op = create_hsplit_int_typed_handle();
    return op.call(self, sections);
}

// aten::hsplit.int(Tensor(a) self, int sections) -> Tensor(a)[]
::std::vector<at::Tensor> hsplit_int::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t sections) {
    static auto op = create_hsplit_int_typed_handle();
    return op.redispatch(dispatchKeySet, self, sections);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(dsplit_array, name, "aten::dsplit")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(dsplit_array, overload_name, "array")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(dsplit_array, schema_str, "dsplit.array(Tensor(a) self, int[] indices) -> Tensor(a)[]")

// aten::dsplit.array(Tensor(a) self, int[] indices) -> Tensor(a)[]
static C10_NOINLINE c10::TypedOperatorHandle<dsplit_array::schema> create_dsplit_array_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(dsplit_array::name, dsplit_array::overload_name)
      .typed<dsplit_array::schema>();
}

// aten::dsplit.array(Tensor(a) self, int[] indices) -> Tensor(a)[]
::std::vector<at::Tensor> dsplit_array::call(const at::Tensor & self, at::IntArrayRef indices) {
    static auto op = create_dsplit_array_typed_handle();
    return op.call(self, indices);
}

// aten::dsplit.array(Tensor(a) self, int[] indices) -> Tensor(a)[]
::std::vector<at::Tensor> dsplit_array::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef indices) {
    static auto op = create_dsplit_array_typed_handle();
    return op.redispatch(dispatchKeySet, self, indices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(squeeze_dim, name, "aten::squeeze")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(squeeze_dim, overload_name, "dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(squeeze_dim, schema_str, "squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)")

// aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<squeeze_dim::schema> create_squeeze_dim_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(squeeze_dim::name, squeeze_dim::overload_name)
      .typed<squeeze_dim::schema>();
}

// aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)
at::Tensor squeeze_dim::call(const at::Tensor & self, int64_t dim) {
    static auto op = create_squeeze_dim_typed_handle();
    return op.call(self, dim);
}

// aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)
at::Tensor squeeze_dim::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim) {
    static auto op = create_squeeze_dim_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(squeeze_dimname, name, "aten::squeeze")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(squeeze_dimname, overload_name, "dimname")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(squeeze_dimname, schema_str, "squeeze.dimname(Tensor(a) self, Dimname dim) -> Tensor(a)")

// aten::squeeze.dimname(Tensor(a) self, Dimname dim) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<squeeze_dimname::schema> create_squeeze_dimname_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(squeeze_dimname::name, squeeze_dimname::overload_name)
      .typed<squeeze_dimname::schema>();
}

// aten::squeeze.dimname(Tensor(a) self, Dimname dim) -> Tensor(a)
at::Tensor squeeze_dimname::call(const at::Tensor & self, at::Dimname dim) {
    static auto op = create_squeeze_dimname_typed_handle();
    return op.call(self, dim);
}

// aten::squeeze.dimname(Tensor(a) self, Dimname dim) -> Tensor(a)
at::Tensor squeeze_dimname::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim) {
    static auto op = create_squeeze_dimname_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(squeeze_, name, "aten::squeeze_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(squeeze_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(squeeze_, schema_str, "squeeze_(Tensor(a!) self) -> Tensor(a!)")

// aten::squeeze_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<squeeze_::schema> create_squeeze__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(squeeze_::name, squeeze_::overload_name)
      .typed<squeeze_::schema>();
}

// aten::squeeze_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & squeeze_::call(at::Tensor & self) {
    static auto op = create_squeeze__typed_handle();
    return op.call(self);
}

// aten::squeeze_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & squeeze_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_squeeze__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(squeeze__dim, name, "aten::squeeze_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(squeeze__dim, overload_name, "dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(squeeze__dim, schema_str, "squeeze_.dim(Tensor(a!) self, int dim) -> Tensor(a!)")

// aten::squeeze_.dim(Tensor(a!) self, int dim) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<squeeze__dim::schema> create_squeeze__dim_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(squeeze__dim::name, squeeze__dim::overload_name)
      .typed<squeeze__dim::schema>();
}

// aten::squeeze_.dim(Tensor(a!) self, int dim) -> Tensor(a!)
at::Tensor & squeeze__dim::call(at::Tensor & self, int64_t dim) {
    static auto op = create_squeeze__dim_typed_handle();
    return op.call(self, dim);
}

// aten::squeeze_.dim(Tensor(a!) self, int dim) -> Tensor(a!)
at::Tensor & squeeze__dim::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t dim) {
    static auto op = create_squeeze__dim_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hstack, name, "aten::hstack")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hstack, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hstack, schema_str, "hstack(Tensor[] tensors) -> Tensor")

// aten::hstack(Tensor[] tensors) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<hstack::schema> create_hstack_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(hstack::name, hstack::overload_name)
      .typed<hstack::schema>();
}

// aten::hstack(Tensor[] tensors) -> Tensor
at::Tensor hstack::call(at::TensorList tensors) {
    static auto op = create_hstack_typed_handle();
    return op.call(tensors);
}

// aten::hstack(Tensor[] tensors) -> Tensor
at::Tensor hstack::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create_hstack_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(dstack_out, name, "aten::dstack")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(dstack_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(dstack_out, schema_str, "dstack.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)")

// aten::dstack.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<dstack_out::schema> create_dstack_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(dstack_out::name, dstack_out::overload_name)
      .typed<dstack_out::schema>();
}

// aten::dstack.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & dstack_out::call(at::TensorList tensors, at::Tensor & out) {
    static auto op = create_dstack_out_typed_handle();
    return op.call(tensors, out);
}

// aten::dstack.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & dstack_out::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, at::Tensor & out) {
    static auto op = create_dstack_out_typed_handle();
    return op.redispatch(dispatchKeySet, tensors, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(istft, name, "aten::istft")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(istft, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(istft, schema_str, "istft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool center=True, bool normalized=False, bool? onesided=None, int? length=None, bool return_complex=False) -> Tensor")

// aten::istft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool center=True, bool normalized=False, bool? onesided=None, int? length=None, bool return_complex=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<istft::schema> create_istft_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(istft::name, istft::overload_name)
      .typed<istft::schema>();
}

// aten::istft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool center=True, bool normalized=False, bool? onesided=None, int? length=None, bool return_complex=False) -> Tensor
at::Tensor istft::call(const at::Tensor & self, int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const c10::optional<at::Tensor> & window, bool center, bool normalized, c10::optional<bool> onesided, c10::optional<int64_t> length, bool return_complex) {
    static auto op = create_istft_typed_handle();
    return op.call(self, n_fft, hop_length, win_length, window, center, normalized, onesided, length, return_complex);
}

// aten::istft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool center=True, bool normalized=False, bool? onesided=None, int? length=None, bool return_complex=False) -> Tensor
at::Tensor istft::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const c10::optional<at::Tensor> & window, bool center, bool normalized, c10::optional<bool> onesided, c10::optional<int64_t> length, bool return_complex) {
    static auto op = create_istft_typed_handle();
    return op.redispatch(dispatchKeySet, self, n_fft, hop_length, win_length, window, center, normalized, onesided, length, return_complex);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sum, name, "aten::sum")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sum, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sum, schema_str, "sum(Tensor self, *, ScalarType? dtype=None) -> Tensor")

// aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<sum::schema> create_sum_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sum::name, sum::overload_name)
      .typed<sum::schema>();
}

// aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor
at::Tensor sum::call(const at::Tensor & self, c10::optional<at::ScalarType> dtype) {
    static auto op = create_sum_typed_handle();
    return op.call(self, dtype);
}

// aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor
at::Tensor sum::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::ScalarType> dtype) {
    static auto op = create_sum_typed_handle();
    return op.redispatch(dispatchKeySet, self, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sum_IntList_out, name, "aten::sum")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sum_IntList_out, overload_name, "IntList_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sum_IntList_out, schema_str, "sum.IntList_out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)")

// aten::sum.IntList_out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<sum_IntList_out::schema> create_sum_IntList_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sum_IntList_out::name, sum_IntList_out::overload_name)
      .typed<sum_IntList_out::schema>();
}

// aten::sum.IntList_out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & sum_IntList_out::call(const at::Tensor & self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
    static auto op = create_sum_IntList_out_typed_handle();
    return op.call(self, dim, keepdim, dtype, out);
}

// aten::sum.IntList_out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & sum_IntList_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
    static auto op = create_sum_IntList_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, dtype, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sum_DimnameList_out, name, "aten::sum")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sum_DimnameList_out, overload_name, "DimnameList_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sum_DimnameList_out, schema_str, "sum.DimnameList_out(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)")

// aten::sum.DimnameList_out(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<sum_DimnameList_out::schema> create_sum_DimnameList_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sum_DimnameList_out::name, sum_DimnameList_out::overload_name)
      .typed<sum_DimnameList_out::schema>();
}

// aten::sum.DimnameList_out(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & sum_DimnameList_out::call(const at::Tensor & self, at::DimnameList dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
    static auto op = create_sum_DimnameList_out_typed_handle();
    return op.call(self, dim, keepdim, dtype, out);
}

// aten::sum.DimnameList_out(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & sum_DimnameList_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
    static auto op = create_sum_DimnameList_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, dtype, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nansum, name, "aten::nansum")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nansum, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nansum, schema_str, "nansum(Tensor self, *, ScalarType? dtype=None) -> Tensor")

// aten::nansum(Tensor self, *, ScalarType? dtype=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<nansum::schema> create_nansum_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nansum::name, nansum::overload_name)
      .typed<nansum::schema>();
}

// aten::nansum(Tensor self, *, ScalarType? dtype=None) -> Tensor
at::Tensor nansum::call(const at::Tensor & self, c10::optional<at::ScalarType> dtype) {
    static auto op = create_nansum_typed_handle();
    return op.call(self, dtype);
}

// aten::nansum(Tensor self, *, ScalarType? dtype=None) -> Tensor
at::Tensor nansum::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::ScalarType> dtype) {
    static auto op = create_nansum_typed_handle();
    return op.redispatch(dispatchKeySet, self, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sqrt_out, name, "aten::sqrt")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sqrt_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sqrt_out, schema_str, "sqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::sqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<sqrt_out::schema> create_sqrt_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sqrt_out::name, sqrt_out::overload_name)
      .typed<sqrt_out::schema>();
}

// aten::sqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & sqrt_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_sqrt_out_typed_handle();
    return op.call(self, out);
}

// aten::sqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & sqrt_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_sqrt_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_mean_correction, name, "aten::std_mean")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_mean_correction, overload_name, "correction")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_mean_correction, schema_str, "std_mean.correction(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False) -> (Tensor, Tensor)")

// aten::std_mean.correction(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<std_mean_correction::schema> create_std_mean_correction_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(std_mean_correction::name, std_mean_correction::overload_name)
      .typed<std_mean_correction::schema>();
}

// aten::std_mean.correction(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> std_mean_correction::call(const at::Tensor & self, c10::optional<at::IntArrayRef> dim, c10::optional<int64_t> correction, bool keepdim) {
    static auto op = create_std_mean_correction_typed_handle();
    return op.call(self, dim, correction, keepdim);
}

// aten::std_mean.correction(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> std_mean_correction::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> dim, c10::optional<int64_t> correction, bool keepdim) {
    static auto op = create_std_mean_correction_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, correction, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_out, name, "aten::std")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_out, schema_str, "std.out(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)")

// aten::std.out(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<std_out::schema> create_std_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(std_out::name, std_out::overload_name)
      .typed<std_out::schema>();
}

// aten::std.out(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & std_out::call(const at::Tensor & self, at::IntArrayRef dim, bool unbiased, bool keepdim, at::Tensor & out) {
    static auto op = create_std_out_typed_handle();
    return op.call(self, dim, unbiased, keepdim, out);
}

// aten::std.out(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & std_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool unbiased, bool keepdim, at::Tensor & out) {
    static auto op = create_std_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, unbiased, keepdim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_correction_names_out, name, "aten::std")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_correction_names_out, overload_name, "correction_names_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_correction_names_out, schema_str, "std.correction_names_out(Tensor self, Dimname[1] dim, *, int? correction, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)")

// aten::std.correction_names_out(Tensor self, Dimname[1] dim, *, int? correction, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<std_correction_names_out::schema> create_std_correction_names_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(std_correction_names_out::name, std_correction_names_out::overload_name)
      .typed<std_correction_names_out::schema>();
}

// aten::std.correction_names_out(Tensor self, Dimname[1] dim, *, int? correction, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)
at::Tensor & std_correction_names_out::call(const at::Tensor & self, at::DimnameList dim, c10::optional<int64_t> correction, bool keepdim, at::Tensor & out) {
    static auto op = create_std_correction_names_out_typed_handle();
    return op.call(self, dim, correction, keepdim, out);
}

// aten::std.correction_names_out(Tensor self, Dimname[1] dim, *, int? correction, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)
at::Tensor & std_correction_names_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList dim, c10::optional<int64_t> correction, bool keepdim, at::Tensor & out) {
    static auto op = create_std_correction_names_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, correction, keepdim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tanh_out, name, "aten::tanh")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tanh_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tanh_out, schema_str, "tanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::tanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<tanh_out::schema> create_tanh_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(tanh_out::name, tanh_out::overload_name)
      .typed<tanh_out::schema>();
}

// aten::tanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & tanh_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_tanh_out_typed_handle();
    return op.call(self, out);
}

// aten::tanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & tanh_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_tanh_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(threshold_, name, "aten::threshold_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(threshold_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(threshold_, schema_str, "threshold_(Tensor(a!) self, Scalar threshold, Scalar value) -> Tensor(a!)")

// aten::threshold_(Tensor(a!) self, Scalar threshold, Scalar value) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<threshold_::schema> create_threshold__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(threshold_::name, threshold_::overload_name)
      .typed<threshold_::schema>();
}

// aten::threshold_(Tensor(a!) self, Scalar threshold, Scalar value) -> Tensor(a!)
at::Tensor & threshold_::call(at::Tensor & self, const at::Scalar & threshold, const at::Scalar & value) {
    static auto op = create_threshold__typed_handle();
    return op.call(self, threshold, value);
}

// aten::threshold_(Tensor(a!) self, Scalar threshold, Scalar value) -> Tensor(a!)
at::Tensor & threshold_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & threshold, const at::Scalar & value) {
    static auto op = create_threshold__typed_handle();
    return op.redispatch(dispatchKeySet, self, threshold, value);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(flipud, name, "aten::flipud")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(flipud, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(flipud, schema_str, "flipud(Tensor self) -> Tensor")

// aten::flipud(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<flipud::schema> create_flipud_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(flipud::name, flipud::overload_name)
      .typed<flipud::schema>();
}

// aten::flipud(Tensor self) -> Tensor
at::Tensor flipud::call(const at::Tensor & self) {
    static auto op = create_flipud_typed_handle();
    return op.call(self);
}

// aten::flipud(Tensor self) -> Tensor
at::Tensor flipud::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_flipud_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rot90, name, "aten::rot90")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rot90, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rot90, schema_str, "rot90(Tensor self, int k=1, int[] dims=[0,1]) -> Tensor")

// aten::rot90(Tensor self, int k=1, int[] dims=[0,1]) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<rot90::schema> create_rot90_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(rot90::name, rot90::overload_name)
      .typed<rot90::schema>();
}

// aten::rot90(Tensor self, int k=1, int[] dims=[0,1]) -> Tensor
at::Tensor rot90::call(const at::Tensor & self, int64_t k, at::IntArrayRef dims) {
    static auto op = create_rot90_typed_handle();
    return op.call(self, k, dims);
}

// aten::rot90(Tensor self, int k=1, int[] dims=[0,1]) -> Tensor
at::Tensor rot90::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t k, at::IntArrayRef dims) {
    static auto op = create_rot90_typed_handle();
    return op.redispatch(dispatchKeySet, self, k, dims);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(trapezoid_x, name, "aten::trapezoid")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(trapezoid_x, overload_name, "x")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(trapezoid_x, schema_str, "trapezoid.x(Tensor y, Tensor x, *, int dim=-1) -> Tensor")

// aten::trapezoid.x(Tensor y, Tensor x, *, int dim=-1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<trapezoid_x::schema> create_trapezoid_x_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(trapezoid_x::name, trapezoid_x::overload_name)
      .typed<trapezoid_x::schema>();
}

// aten::trapezoid.x(Tensor y, Tensor x, *, int dim=-1) -> Tensor
at::Tensor trapezoid_x::call(const at::Tensor & y, const at::Tensor & x, int64_t dim) {
    static auto op = create_trapezoid_x_typed_handle();
    return op.call(y, x, dim);
}

// aten::trapezoid.x(Tensor y, Tensor x, *, int dim=-1) -> Tensor
at::Tensor trapezoid_x::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & y, const at::Tensor & x, int64_t dim) {
    static auto op = create_trapezoid_x_typed_handle();
    return op.redispatch(dispatchKeySet, y, x, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(triplet_margin_loss, name, "aten::triplet_margin_loss")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(triplet_margin_loss, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(triplet_margin_loss, schema_str, "triplet_margin_loss(Tensor anchor, Tensor positive, Tensor negative, float margin=1.0, float p=2, float eps=1e-06, bool swap=False, int reduction=Mean) -> Tensor")

// aten::triplet_margin_loss(Tensor anchor, Tensor positive, Tensor negative, float margin=1.0, float p=2, float eps=1e-06, bool swap=False, int reduction=Mean) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<triplet_margin_loss::schema> create_triplet_margin_loss_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(triplet_margin_loss::name, triplet_margin_loss::overload_name)
      .typed<triplet_margin_loss::schema>();
}

// aten::triplet_margin_loss(Tensor anchor, Tensor positive, Tensor negative, float margin=1.0, float p=2, float eps=1e-06, bool swap=False, int reduction=Mean) -> Tensor
at::Tensor triplet_margin_loss::call(const at::Tensor & anchor, const at::Tensor & positive, const at::Tensor & negative, double margin, double p, double eps, bool swap, int64_t reduction) {
    static auto op = create_triplet_margin_loss_typed_handle();
    return op.call(anchor, positive, negative, margin, p, eps, swap, reduction);
}

// aten::triplet_margin_loss(Tensor anchor, Tensor positive, Tensor negative, float margin=1.0, float p=2, float eps=1e-06, bool swap=False, int reduction=Mean) -> Tensor
at::Tensor triplet_margin_loss::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & anchor, const at::Tensor & positive, const at::Tensor & negative, double margin, double p, double eps, bool swap, int64_t reduction) {
    static auto op = create_triplet_margin_loss_typed_handle();
    return op.redispatch(dispatchKeySet, anchor, positive, negative, margin, p, eps, swap, reduction);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(trunc, name, "aten::trunc")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(trunc, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(trunc, schema_str, "trunc(Tensor self) -> Tensor")

// aten::trunc(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<trunc::schema> create_trunc_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(trunc::name, trunc::overload_name)
      .typed<trunc::schema>();
}

// aten::trunc(Tensor self) -> Tensor
at::Tensor trunc::call(const at::Tensor & self) {
    static auto op = create_trunc_typed_handle();
    return op.call(self);
}

// aten::trunc(Tensor self) -> Tensor
at::Tensor trunc::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_trunc_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(trunc_, name, "aten::trunc_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(trunc_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(trunc_, schema_str, "trunc_(Tensor(a!) self) -> Tensor(a!)")

// aten::trunc_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<trunc_::schema> create_trunc__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(trunc_::name, trunc_::overload_name)
      .typed<trunc_::schema>();
}

// aten::trunc_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & trunc_::call(at::Tensor & self) {
    static auto op = create_trunc__typed_handle();
    return op.call(self);
}

// aten::trunc_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & trunc_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_trunc__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var, name, "aten::var")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var, schema_str, "var(Tensor self, bool unbiased=True) -> Tensor")

// aten::var(Tensor self, bool unbiased=True) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<var::schema> create_var_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(var::name, var::overload_name)
      .typed<var::schema>();
}

// aten::var(Tensor self, bool unbiased=True) -> Tensor
at::Tensor var::call(const at::Tensor & self, bool unbiased) {
    static auto op = create_var_typed_handle();
    return op.call(self, unbiased);
}

// aten::var(Tensor self, bool unbiased=True) -> Tensor
at::Tensor var::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool unbiased) {
    static auto op = create_var_typed_handle();
    return op.redispatch(dispatchKeySet, self, unbiased);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_correction_out, name, "aten::var")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_correction_out, overload_name, "correction_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_correction_out, schema_str, "var.correction_out(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)")

// aten::var.correction_out(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<var_correction_out::schema> create_var_correction_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(var_correction_out::name, var_correction_out::overload_name)
      .typed<var_correction_out::schema>();
}

// aten::var.correction_out(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)
at::Tensor & var_correction_out::call(const at::Tensor & self, c10::optional<at::IntArrayRef> dim, c10::optional<int64_t> correction, bool keepdim, at::Tensor & out) {
    static auto op = create_var_correction_out_typed_handle();
    return op.call(self, dim, correction, keepdim, out);
}

// aten::var.correction_out(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)
at::Tensor & var_correction_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> dim, c10::optional<int64_t> correction, bool keepdim, at::Tensor & out) {
    static auto op = create_var_correction_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, correction, keepdim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_names_out, name, "aten::var")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_names_out, overload_name, "names_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_names_out, schema_str, "var.names_out(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)")

// aten::var.names_out(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<var_names_out::schema> create_var_names_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(var_names_out::name, var_names_out::overload_name)
      .typed<var_names_out::schema>();
}

// aten::var.names_out(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & var_names_out::call(const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim, at::Tensor & out) {
    static auto op = create_var_names_out_typed_handle();
    return op.call(self, dim, unbiased, keepdim, out);
}

// aten::var.names_out(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & var_names_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim, at::Tensor & out) {
    static auto op = create_var_names_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, unbiased, keepdim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_mean, name, "aten::var_mean")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_mean, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_mean, schema_str, "var_mean(Tensor self, bool unbiased=True) -> (Tensor, Tensor)")

// aten::var_mean(Tensor self, bool unbiased=True) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<var_mean::schema> create_var_mean_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(var_mean::name, var_mean::overload_name)
      .typed<var_mean::schema>();
}

// aten::var_mean(Tensor self, bool unbiased=True) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> var_mean::call(const at::Tensor & self, bool unbiased) {
    static auto op = create_var_mean_typed_handle();
    return op.call(self, unbiased);
}

// aten::var_mean(Tensor self, bool unbiased=True) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> var_mean::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool unbiased) {
    static auto op = create_var_mean_typed_handle();
    return op.redispatch(dispatchKeySet, self, unbiased);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(where_ScalarSelf, name, "aten::where")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(where_ScalarSelf, overload_name, "ScalarSelf")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(where_ScalarSelf, schema_str, "where.ScalarSelf(Tensor condition, Scalar self, Tensor other) -> Tensor")

// aten::where.ScalarSelf(Tensor condition, Scalar self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<where_ScalarSelf::schema> create_where_ScalarSelf_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(where_ScalarSelf::name, where_ScalarSelf::overload_name)
      .typed<where_ScalarSelf::schema>();
}

// aten::where.ScalarSelf(Tensor condition, Scalar self, Tensor other) -> Tensor
at::Tensor where_ScalarSelf::call(const at::Tensor & condition, const at::Scalar & self, const at::Tensor & other) {
    static auto op = create_where_ScalarSelf_typed_handle();
    return op.call(condition, self, other);
}

// aten::where.ScalarSelf(Tensor condition, Scalar self, Tensor other) -> Tensor
at::Tensor where_ScalarSelf::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & condition, const at::Scalar & self, const at::Tensor & other) {
    static auto op = create_where_ScalarSelf_typed_handle();
    return op.redispatch(dispatchKeySet, condition, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(where_ScalarOther, name, "aten::where")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(where_ScalarOther, overload_name, "ScalarOther")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(where_ScalarOther, schema_str, "where.ScalarOther(Tensor condition, Tensor self, Scalar other) -> Tensor")

// aten::where.ScalarOther(Tensor condition, Tensor self, Scalar other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<where_ScalarOther::schema> create_where_ScalarOther_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(where_ScalarOther::name, where_ScalarOther::overload_name)
      .typed<where_ScalarOther::schema>();
}

// aten::where.ScalarOther(Tensor condition, Tensor self, Scalar other) -> Tensor
at::Tensor where_ScalarOther::call(const at::Tensor & condition, const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_where_ScalarOther_typed_handle();
    return op.call(condition, self, other);
}

// aten::where.ScalarOther(Tensor condition, Tensor self, Scalar other) -> Tensor
at::Tensor where_ScalarOther::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & condition, const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_where_ScalarOther_typed_handle();
    return op.redispatch(dispatchKeySet, condition, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(norm_except_dim, name, "aten::norm_except_dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(norm_except_dim, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(norm_except_dim, schema_str, "norm_except_dim(Tensor v, int pow=2, int dim=0) -> Tensor")

// aten::norm_except_dim(Tensor v, int pow=2, int dim=0) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<norm_except_dim::schema> create_norm_except_dim_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(norm_except_dim::name, norm_except_dim::overload_name)
      .typed<norm_except_dim::schema>();
}

// aten::norm_except_dim(Tensor v, int pow=2, int dim=0) -> Tensor
at::Tensor norm_except_dim::call(const at::Tensor & v, int64_t pow, int64_t dim) {
    static auto op = create_norm_except_dim_typed_handle();
    return op.call(v, pow, dim);
}

// aten::norm_except_dim(Tensor v, int pow=2, int dim=0) -> Tensor
at::Tensor norm_except_dim::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & v, int64_t pow, int64_t dim) {
    static auto op = create_norm_except_dim_typed_handle();
    return op.redispatch(dispatchKeySet, v, pow, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_standard_gamma_grad, name, "aten::_standard_gamma_grad")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_standard_gamma_grad, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_standard_gamma_grad, schema_str, "_standard_gamma_grad(Tensor self, Tensor output) -> Tensor")

// aten::_standard_gamma_grad(Tensor self, Tensor output) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_standard_gamma_grad::schema> create__standard_gamma_grad_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_standard_gamma_grad::name, _standard_gamma_grad::overload_name)
      .typed<_standard_gamma_grad::schema>();
}

// aten::_standard_gamma_grad(Tensor self, Tensor output) -> Tensor
at::Tensor _standard_gamma_grad::call(const at::Tensor & self, const at::Tensor & output) {
    static auto op = create__standard_gamma_grad_typed_handle();
    return op.call(self, output);
}

// aten::_standard_gamma_grad(Tensor self, Tensor output) -> Tensor
at::Tensor _standard_gamma_grad::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & output) {
    static auto op = create__standard_gamma_grad_typed_handle();
    return op.redispatch(dispatchKeySet, self, output);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(native_norm, name, "aten::native_norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(native_norm, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(native_norm, schema_str, "native_norm(Tensor self, Scalar p=2) -> Tensor")

// aten::native_norm(Tensor self, Scalar p=2) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<native_norm::schema> create_native_norm_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(native_norm::name, native_norm::overload_name)
      .typed<native_norm::schema>();
}

// aten::native_norm(Tensor self, Scalar p=2) -> Tensor
at::Tensor native_norm::call(const at::Tensor & self, const at::Scalar & p) {
    static auto op = create_native_norm_typed_handle();
    return op.call(self, p);
}

// aten::native_norm(Tensor self, Scalar p=2) -> Tensor
at::Tensor native_norm::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & p) {
    static auto op = create_native_norm_typed_handle();
    return op.redispatch(dispatchKeySet, self, p);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(native_norm_ScalarOpt_dim_dtype, name, "aten::native_norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(native_norm_ScalarOpt_dim_dtype, overload_name, "ScalarOpt_dim_dtype")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(native_norm_ScalarOpt_dim_dtype, schema_str, "native_norm.ScalarOpt_dim_dtype(Tensor self, Scalar? p, int[1] dim, bool keepdim, ScalarType? dtype) -> Tensor")

// aten::native_norm.ScalarOpt_dim_dtype(Tensor self, Scalar? p, int[1] dim, bool keepdim, ScalarType? dtype) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<native_norm_ScalarOpt_dim_dtype::schema> create_native_norm_ScalarOpt_dim_dtype_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(native_norm_ScalarOpt_dim_dtype::name, native_norm_ScalarOpt_dim_dtype::overload_name)
      .typed<native_norm_ScalarOpt_dim_dtype::schema>();
}

// aten::native_norm.ScalarOpt_dim_dtype(Tensor self, Scalar? p, int[1] dim, bool keepdim, ScalarType? dtype) -> Tensor
at::Tensor native_norm_ScalarOpt_dim_dtype::call(const at::Tensor & self, const c10::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_native_norm_ScalarOpt_dim_dtype_typed_handle();
    return op.call(self, p, dim, keepdim, dtype);
}

// aten::native_norm.ScalarOpt_dim_dtype(Tensor self, Scalar? p, int[1] dim, bool keepdim, ScalarType? dtype) -> Tensor
at::Tensor native_norm_ScalarOpt_dim_dtype::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_native_norm_ScalarOpt_dim_dtype_typed_handle();
    return op.redispatch(dispatchKeySet, self, p, dim, keepdim, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_sum_dim_dtype, name, "aten::_sparse_sum")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_sum_dim_dtype, overload_name, "dim_dtype")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_sum_dim_dtype, schema_str, "_sparse_sum.dim_dtype(Tensor self, int[1] dim, *, ScalarType dtype) -> Tensor")

// aten::_sparse_sum.dim_dtype(Tensor self, int[1] dim, *, ScalarType dtype) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_sparse_sum_dim_dtype::schema> create__sparse_sum_dim_dtype_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_sparse_sum_dim_dtype::name, _sparse_sum_dim_dtype::overload_name)
      .typed<_sparse_sum_dim_dtype::schema>();
}

// aten::_sparse_sum.dim_dtype(Tensor self, int[1] dim, *, ScalarType dtype) -> Tensor
at::Tensor _sparse_sum_dim_dtype::call(const at::Tensor & self, at::IntArrayRef dim, at::ScalarType dtype) {
    static auto op = create__sparse_sum_dim_dtype_typed_handle();
    return op.call(self, dim, dtype);
}

// aten::_sparse_sum.dim_dtype(Tensor self, int[1] dim, *, ScalarType dtype) -> Tensor
at::Tensor _sparse_sum_dim_dtype::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, at::ScalarType dtype) {
    static auto op = create__sparse_sum_dim_dtype_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_sum_backward, name, "aten::_sparse_sum_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_sum_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_sum_backward, schema_str, "_sparse_sum_backward(Tensor grad, Tensor self, int[] dim) -> Tensor")

// aten::_sparse_sum_backward(Tensor grad, Tensor self, int[] dim) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_sparse_sum_backward::schema> create__sparse_sum_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_sparse_sum_backward::name, _sparse_sum_backward::overload_name)
      .typed<_sparse_sum_backward::schema>();
}

// aten::_sparse_sum_backward(Tensor grad, Tensor self, int[] dim) -> Tensor
at::Tensor _sparse_sum_backward::call(const at::Tensor & grad, const at::Tensor & self, at::IntArrayRef dim) {
    static auto op = create__sparse_sum_backward_typed_handle();
    return op.call(grad, self, dim);
}

// aten::_sparse_sum_backward(Tensor grad, Tensor self, int[] dim) -> Tensor
at::Tensor _sparse_sum_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & self, at::IntArrayRef dim) {
    static auto op = create__sparse_sum_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad, self, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_softmax, name, "aten::_sparse_softmax")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_softmax, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_softmax, schema_str, "_sparse_softmax(Tensor self, int dim, bool half_to_float) -> Tensor")

// aten::_sparse_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_sparse_softmax::schema> create__sparse_softmax_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_sparse_softmax::name, _sparse_softmax::overload_name)
      .typed<_sparse_softmax::schema>();
}

// aten::_sparse_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
at::Tensor _sparse_softmax::call(const at::Tensor & self, int64_t dim, bool half_to_float) {
    static auto op = create__sparse_softmax_typed_handle();
    return op.call(self, dim, half_to_float);
}

// aten::_sparse_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
at::Tensor _sparse_softmax::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool half_to_float) {
    static auto op = create__sparse_softmax_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, half_to_float);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(norm_Scalar, name, "aten::norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(norm_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(norm_Scalar, schema_str, "norm.Scalar(Tensor self, Scalar p=2) -> Tensor")

// aten::norm.Scalar(Tensor self, Scalar p=2) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<norm_Scalar::schema> create_norm_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(norm_Scalar::name, norm_Scalar::overload_name)
      .typed<norm_Scalar::schema>();
}

// aten::norm.Scalar(Tensor self, Scalar p=2) -> Tensor
at::Tensor norm_Scalar::call(const at::Tensor & self, const at::Scalar & p) {
    static auto op = create_norm_Scalar_typed_handle();
    return op.call(self, p);
}

// aten::norm.Scalar(Tensor self, Scalar p=2) -> Tensor
at::Tensor norm_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & p) {
    static auto op = create_norm_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, p);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(norm_ScalarOpt_dim_dtype, name, "aten::norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(norm_ScalarOpt_dim_dtype, overload_name, "ScalarOpt_dim_dtype")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(norm_ScalarOpt_dim_dtype, schema_str, "norm.ScalarOpt_dim_dtype(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor")

// aten::norm.ScalarOpt_dim_dtype(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<norm_ScalarOpt_dim_dtype::schema> create_norm_ScalarOpt_dim_dtype_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(norm_ScalarOpt_dim_dtype::name, norm_ScalarOpt_dim_dtype::overload_name)
      .typed<norm_ScalarOpt_dim_dtype::schema>();
}

// aten::norm.ScalarOpt_dim_dtype(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor
at::Tensor norm_ScalarOpt_dim_dtype::call(const at::Tensor & self, const c10::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, at::ScalarType dtype) {
    static auto op = create_norm_ScalarOpt_dim_dtype_typed_handle();
    return op.call(self, p, dim, keepdim, dtype);
}

// aten::norm.ScalarOpt_dim_dtype(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor
at::Tensor norm_ScalarOpt_dim_dtype::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, at::ScalarType dtype) {
    static auto op = create_norm_ScalarOpt_dim_dtype_typed_handle();
    return op.redispatch(dispatchKeySet, self, p, dim, keepdim, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(norm_names_dtype_out, name, "aten::norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(norm_names_dtype_out, overload_name, "names_dtype_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(norm_names_dtype_out, schema_str, "norm.names_dtype_out(Tensor self, Scalar? p, Dimname[1] dim, bool keepdim, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)")

// aten::norm.names_dtype_out(Tensor self, Scalar? p, Dimname[1] dim, bool keepdim, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<norm_names_dtype_out::schema> create_norm_names_dtype_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(norm_names_dtype_out::name, norm_names_dtype_out::overload_name)
      .typed<norm_names_dtype_out::schema>();
}

// aten::norm.names_dtype_out(Tensor self, Scalar? p, Dimname[1] dim, bool keepdim, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)
at::Tensor & norm_names_dtype_out::call(const at::Tensor & self, const c10::optional<at::Scalar> & p, at::DimnameList dim, bool keepdim, at::ScalarType dtype, at::Tensor & out) {
    static auto op = create_norm_names_dtype_out_typed_handle();
    return op.call(self, p, dim, keepdim, dtype, out);
}

// aten::norm.names_dtype_out(Tensor self, Scalar? p, Dimname[1] dim, bool keepdim, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)
at::Tensor & norm_names_dtype_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Scalar> & p, at::DimnameList dim, bool keepdim, at::ScalarType dtype, at::Tensor & out) {
    static auto op = create_norm_names_dtype_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, p, dim, keepdim, dtype, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nuclear_norm, name, "aten::nuclear_norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nuclear_norm, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nuclear_norm, schema_str, "nuclear_norm(Tensor self, bool keepdim=False) -> Tensor")

// aten::nuclear_norm(Tensor self, bool keepdim=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<nuclear_norm::schema> create_nuclear_norm_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nuclear_norm::name, nuclear_norm::overload_name)
      .typed<nuclear_norm::schema>();
}

// aten::nuclear_norm(Tensor self, bool keepdim=False) -> Tensor
at::Tensor nuclear_norm::call(const at::Tensor & self, bool keepdim) {
    static auto op = create_nuclear_norm_typed_handle();
    return op.call(self, keepdim);
}

// aten::nuclear_norm(Tensor self, bool keepdim=False) -> Tensor
at::Tensor nuclear_norm::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool keepdim) {
    static auto op = create_nuclear_norm_typed_handle();
    return op.redispatch(dispatchKeySet, self, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sub_Scalar, name, "aten::sub")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sub_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sub_Scalar, schema_str, "sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor")

// aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<sub_Scalar::schema> create_sub_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sub_Scalar::name, sub_Scalar::overload_name)
      .typed<sub_Scalar::schema>();
}

// aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
at::Tensor sub_Scalar::call(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
    static auto op = create_sub_Scalar_typed_handle();
    return op.call(self, other, alpha);
}

// aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
at::Tensor sub_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
    static auto op = create_sub_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rsub_Tensor, name, "aten::rsub")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rsub_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rsub_Tensor, schema_str, "rsub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor")

// aten::rsub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<rsub_Tensor::schema> create_rsub_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(rsub_Tensor::name, rsub_Tensor::overload_name)
      .typed<rsub_Tensor::schema>();
}

// aten::rsub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
at::Tensor rsub_Tensor::call(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
    static auto op = create_rsub_Tensor_typed_handle();
    return op.call(self, other, alpha);
}

// aten::rsub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
at::Tensor rsub_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
    static auto op = create_rsub_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(heaviside_out, name, "aten::heaviside")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(heaviside_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(heaviside_out, schema_str, "heaviside.out(Tensor self, Tensor values, *, Tensor(a!) out) -> Tensor(a!)")

// aten::heaviside.out(Tensor self, Tensor values, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<heaviside_out::schema> create_heaviside_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(heaviside_out::name, heaviside_out::overload_name)
      .typed<heaviside_out::schema>();
}

// aten::heaviside.out(Tensor self, Tensor values, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & heaviside_out::call(const at::Tensor & self, const at::Tensor & values, at::Tensor & out) {
    static auto op = create_heaviside_out_typed_handle();
    return op.call(self, values, out);
}

// aten::heaviside.out(Tensor self, Tensor values, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & heaviside_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & values, at::Tensor & out) {
    static auto op = create_heaviside_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, values, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(heaviside_, name, "aten::heaviside_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(heaviside_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(heaviside_, schema_str, "heaviside_(Tensor(a!) self, Tensor values) -> Tensor(a!)")

// aten::heaviside_(Tensor(a!) self, Tensor values) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<heaviside_::schema> create_heaviside__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(heaviside_::name, heaviside_::overload_name)
      .typed<heaviside_::schema>();
}

// aten::heaviside_(Tensor(a!) self, Tensor values) -> Tensor(a!)
at::Tensor & heaviside_::call(at::Tensor & self, const at::Tensor & values) {
    static auto op = create_heaviside__typed_handle();
    return op.call(self, values);
}

// aten::heaviside_(Tensor(a!) self, Tensor values) -> Tensor(a!)
at::Tensor & heaviside_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & values) {
    static auto op = create_heaviside__typed_handle();
    return op.redispatch(dispatchKeySet, self, values);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addmm_out, name, "aten::addmm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addmm_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addmm_out, schema_str, "addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)")

// aten::addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<addmm_out::schema> create_addmm_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(addmm_out::name, addmm_out::overload_name)
      .typed<addmm_out::schema>();
}

// aten::addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
at::Tensor & addmm_out::call(const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
    static auto op = create_addmm_out_typed_handle();
    return op.call(self, mat1, mat2, beta, alpha, out);
}

// aten::addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
at::Tensor & addmm_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
    static auto op = create_addmm_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, mat1, mat2, beta, alpha, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sparse_coo_tensor_indices, name, "aten::sparse_coo_tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sparse_coo_tensor_indices, overload_name, "indices")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sparse_coo_tensor_indices, schema_str, "sparse_coo_tensor.indices(Tensor indices, Tensor values, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::sparse_coo_tensor.indices(Tensor indices, Tensor values, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<sparse_coo_tensor_indices::schema> create_sparse_coo_tensor_indices_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sparse_coo_tensor_indices::name, sparse_coo_tensor_indices::overload_name)
      .typed<sparse_coo_tensor_indices::schema>();
}

// aten::sparse_coo_tensor.indices(Tensor indices, Tensor values, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor sparse_coo_tensor_indices::call(const at::Tensor & indices, const at::Tensor & values, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_sparse_coo_tensor_indices_typed_handle();
    return op.call(indices, values, dtype, layout, device, pin_memory);
}

// aten::sparse_coo_tensor.indices(Tensor indices, Tensor values, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor sparse_coo_tensor_indices::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & indices, const at::Tensor & values, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_sparse_coo_tensor_indices_typed_handle();
    return op.redispatch(dispatchKeySet, indices, values, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_validate_sparse_coo_tensor_args, name, "aten::_validate_sparse_coo_tensor_args")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_validate_sparse_coo_tensor_args, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_validate_sparse_coo_tensor_args, schema_str, "_validate_sparse_coo_tensor_args(Tensor indices, Tensor values, int[] size) -> ()")

// aten::_validate_sparse_coo_tensor_args(Tensor indices, Tensor values, int[] size) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_validate_sparse_coo_tensor_args::schema> create__validate_sparse_coo_tensor_args_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_validate_sparse_coo_tensor_args::name, _validate_sparse_coo_tensor_args::overload_name)
      .typed<_validate_sparse_coo_tensor_args::schema>();
}

// aten::_validate_sparse_coo_tensor_args(Tensor indices, Tensor values, int[] size) -> ()
void _validate_sparse_coo_tensor_args::call(const at::Tensor & indices, const at::Tensor & values, at::IntArrayRef size) {
    static auto op = create__validate_sparse_coo_tensor_args_typed_handle();
    return op.call(indices, values, size);
}

// aten::_validate_sparse_coo_tensor_args(Tensor indices, Tensor values, int[] size) -> ()
void _validate_sparse_coo_tensor_args::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & indices, const at::Tensor & values, at::IntArrayRef size) {
    static auto op = create__validate_sparse_coo_tensor_args_typed_handle();
    return op.redispatch(dispatchKeySet, indices, values, size);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_coo_tensor_with_dims_and_tensors, name, "aten::_sparse_coo_tensor_with_dims_and_tensors")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_coo_tensor_with_dims_and_tensors, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_coo_tensor_with_dims_and_tensors, schema_str, "_sparse_coo_tensor_with_dims_and_tensors(int sparse_dim, int dense_dim, int[] size, Tensor indices, Tensor values, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor")

// aten::_sparse_coo_tensor_with_dims_and_tensors(int sparse_dim, int dense_dim, int[] size, Tensor indices, Tensor values, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_sparse_coo_tensor_with_dims_and_tensors::schema> create__sparse_coo_tensor_with_dims_and_tensors_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_sparse_coo_tensor_with_dims_and_tensors::name, _sparse_coo_tensor_with_dims_and_tensors::overload_name)
      .typed<_sparse_coo_tensor_with_dims_and_tensors::schema>();
}

// aten::_sparse_coo_tensor_with_dims_and_tensors(int sparse_dim, int dense_dim, int[] size, Tensor indices, Tensor values, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
at::Tensor _sparse_coo_tensor_with_dims_and_tensors::call(int64_t sparse_dim, int64_t dense_dim, at::IntArrayRef size, const at::Tensor & indices, const at::Tensor & values, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create__sparse_coo_tensor_with_dims_and_tensors_typed_handle();
    return op.call(sparse_dim, dense_dim, size, indices, values, dtype, layout, device, pin_memory);
}

// aten::_sparse_coo_tensor_with_dims_and_tensors(int sparse_dim, int dense_dim, int[] size, Tensor indices, Tensor values, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
at::Tensor _sparse_coo_tensor_with_dims_and_tensors::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t sparse_dim, int64_t dense_dim, at::IntArrayRef size, const at::Tensor & indices, const at::Tensor & values, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create__sparse_coo_tensor_with_dims_and_tensors_typed_handle();
    return op.redispatch(dispatchKeySet, sparse_dim, dense_dim, size, indices, values, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sparse_resize_, name, "aten::sparse_resize_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sparse_resize_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sparse_resize_, schema_str, "sparse_resize_(Tensor(a!) self, int[] size, int sparse_dim, int dense_dim) -> Tensor(a!)")

// aten::sparse_resize_(Tensor(a!) self, int[] size, int sparse_dim, int dense_dim) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<sparse_resize_::schema> create_sparse_resize__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sparse_resize_::name, sparse_resize_::overload_name)
      .typed<sparse_resize_::schema>();
}

// aten::sparse_resize_(Tensor(a!) self, int[] size, int sparse_dim, int dense_dim) -> Tensor(a!)
const at::Tensor & sparse_resize_::call(const at::Tensor & self, at::IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) {
    static auto op = create_sparse_resize__typed_handle();
    return op.call(self, size, sparse_dim, dense_dim);
}

// aten::sparse_resize_(Tensor(a!) self, int[] size, int sparse_dim, int dense_dim) -> Tensor(a!)
const at::Tensor & sparse_resize_::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) {
    static auto op = create_sparse_resize__typed_handle();
    return op.redispatch(dispatchKeySet, self, size, sparse_dim, dense_dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_coalesced, name, "aten::is_coalesced")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_coalesced, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_coalesced, schema_str, "is_coalesced(Tensor self) -> bool")

// aten::is_coalesced(Tensor self) -> bool
static C10_NOINLINE c10::TypedOperatorHandle<is_coalesced::schema> create_is_coalesced_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(is_coalesced::name, is_coalesced::overload_name)
      .typed<is_coalesced::schema>();
}

// aten::is_coalesced(Tensor self) -> bool
bool is_coalesced::call(const at::Tensor & self) {
    static auto op = create_is_coalesced_typed_handle();
    return op.call(self);
}

// aten::is_coalesced(Tensor self) -> bool
bool is_coalesced::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_is_coalesced_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(indices, name, "aten::indices")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(indices, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(indices, schema_str, "indices(Tensor(a) self) -> Tensor(a)")

// aten::indices(Tensor(a) self) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<indices::schema> create_indices_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(indices::name, indices::overload_name)
      .typed<indices::schema>();
}

// aten::indices(Tensor(a) self) -> Tensor(a)
at::Tensor indices::call(const at::Tensor & self) {
    static auto op = create_indices_typed_handle();
    return op.call(self);
}

// aten::indices(Tensor(a) self) -> Tensor(a)
at::Tensor indices::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_indices_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(col_indices, name, "aten::col_indices")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(col_indices, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(col_indices, schema_str, "col_indices(Tensor(a) self) -> Tensor(a)")

// aten::col_indices(Tensor(a) self) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<col_indices::schema> create_col_indices_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(col_indices::name, col_indices::overload_name)
      .typed<col_indices::schema>();
}

// aten::col_indices(Tensor(a) self) -> Tensor(a)
at::Tensor col_indices::call(const at::Tensor & self) {
    static auto op = create_col_indices_typed_handle();
    return op.call(self);
}

// aten::col_indices(Tensor(a) self) -> Tensor(a)
at::Tensor col_indices::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_col_indices_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hspmm, name, "aten::hspmm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hspmm, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hspmm, schema_str, "hspmm(Tensor mat1, Tensor mat2) -> Tensor")

// aten::hspmm(Tensor mat1, Tensor mat2) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<hspmm::schema> create_hspmm_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(hspmm::name, hspmm::overload_name)
      .typed<hspmm::schema>();
}

// aten::hspmm(Tensor mat1, Tensor mat2) -> Tensor
at::Tensor hspmm::call(const at::Tensor & mat1, const at::Tensor & mat2) {
    static auto op = create_hspmm_typed_handle();
    return op.call(mat1, mat2);
}

// aten::hspmm(Tensor mat1, Tensor mat2) -> Tensor
at::Tensor hspmm::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & mat1, const at::Tensor & mat2) {
    static auto op = create_hspmm_typed_handle();
    return op.redispatch(dispatchKeySet, mat1, mat2);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unbind_int, name, "aten::unbind")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unbind_int, overload_name, "int")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unbind_int, schema_str, "unbind.int(Tensor(a) self, int dim=0) -> Tensor(a)[]")

// aten::unbind.int(Tensor(a) self, int dim=0) -> Tensor(a)[]
static C10_NOINLINE c10::TypedOperatorHandle<unbind_int::schema> create_unbind_int_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(unbind_int::name, unbind_int::overload_name)
      .typed<unbind_int::schema>();
}

// aten::unbind.int(Tensor(a) self, int dim=0) -> Tensor(a)[]
::std::vector<at::Tensor> unbind_int::call(const at::Tensor & self, int64_t dim) {
    static auto op = create_unbind_int_typed_handle();
    return op.call(self, dim);
}

// aten::unbind.int(Tensor(a) self, int dim=0) -> Tensor(a)[]
::std::vector<at::Tensor> unbind_int::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim) {
    static auto op = create_unbind_int_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantize_per_tensor, name, "aten::quantize_per_tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantize_per_tensor, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantize_per_tensor, schema_str, "quantize_per_tensor(Tensor self, float scale, int zero_point, ScalarType dtype) -> Tensor")

// aten::quantize_per_tensor(Tensor self, float scale, int zero_point, ScalarType dtype) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<quantize_per_tensor::schema> create_quantize_per_tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(quantize_per_tensor::name, quantize_per_tensor::overload_name)
      .typed<quantize_per_tensor::schema>();
}

// aten::quantize_per_tensor(Tensor self, float scale, int zero_point, ScalarType dtype) -> Tensor
at::Tensor quantize_per_tensor::call(const at::Tensor & self, double scale, int64_t zero_point, at::ScalarType dtype) {
    static auto op = create_quantize_per_tensor_typed_handle();
    return op.call(self, scale, zero_point, dtype);
}

// aten::quantize_per_tensor(Tensor self, float scale, int zero_point, ScalarType dtype) -> Tensor
at::Tensor quantize_per_tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double scale, int64_t zero_point, at::ScalarType dtype) {
    static auto op = create_quantize_per_tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, scale, zero_point, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fake_quantize_per_tensor_affine_cachemask, name, "aten::fake_quantize_per_tensor_affine_cachemask")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fake_quantize_per_tensor_affine_cachemask, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fake_quantize_per_tensor_affine_cachemask, schema_str, "fake_quantize_per_tensor_affine_cachemask(Tensor self, float scale, int zero_point, int quant_min, int quant_max) -> (Tensor output, Tensor mask)")

// aten::fake_quantize_per_tensor_affine_cachemask(Tensor self, float scale, int zero_point, int quant_min, int quant_max) -> (Tensor output, Tensor mask)
static C10_NOINLINE c10::TypedOperatorHandle<fake_quantize_per_tensor_affine_cachemask::schema> create_fake_quantize_per_tensor_affine_cachemask_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fake_quantize_per_tensor_affine_cachemask::name, fake_quantize_per_tensor_affine_cachemask::overload_name)
      .typed<fake_quantize_per_tensor_affine_cachemask::schema>();
}

// aten::fake_quantize_per_tensor_affine_cachemask(Tensor self, float scale, int zero_point, int quant_min, int quant_max) -> (Tensor output, Tensor mask)
::std::tuple<at::Tensor,at::Tensor> fake_quantize_per_tensor_affine_cachemask::call(const at::Tensor & self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max) {
    static auto op = create_fake_quantize_per_tensor_affine_cachemask_typed_handle();
    return op.call(self, scale, zero_point, quant_min, quant_max);
}

// aten::fake_quantize_per_tensor_affine_cachemask(Tensor self, float scale, int zero_point, int quant_min, int quant_max) -> (Tensor output, Tensor mask)
::std::tuple<at::Tensor,at::Tensor> fake_quantize_per_tensor_affine_cachemask::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max) {
    static auto op = create_fake_quantize_per_tensor_affine_cachemask_typed_handle();
    return op.redispatch(dispatchKeySet, self, scale, zero_point, quant_min, quant_max);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fake_quantize_per_tensor_affine_cachemask_tensor_qparams, name, "aten::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fake_quantize_per_tensor_affine_cachemask_tensor_qparams, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fake_quantize_per_tensor_affine_cachemask_tensor_qparams, schema_str, "_fake_quantize_per_tensor_affine_cachemask_tensor_qparams(Tensor self, Tensor scale, Tensor zero_point, Tensor fake_quant_enabled, int quant_min, int quant_max) -> (Tensor output, Tensor mask)")

// aten::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams(Tensor self, Tensor scale, Tensor zero_point, Tensor fake_quant_enabled, int quant_min, int quant_max) -> (Tensor output, Tensor mask)
static C10_NOINLINE c10::TypedOperatorHandle<_fake_quantize_per_tensor_affine_cachemask_tensor_qparams::schema> create__fake_quantize_per_tensor_affine_cachemask_tensor_qparams_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_fake_quantize_per_tensor_affine_cachemask_tensor_qparams::name, _fake_quantize_per_tensor_affine_cachemask_tensor_qparams::overload_name)
      .typed<_fake_quantize_per_tensor_affine_cachemask_tensor_qparams::schema>();
}

// aten::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams(Tensor self, Tensor scale, Tensor zero_point, Tensor fake_quant_enabled, int quant_min, int quant_max) -> (Tensor output, Tensor mask)
::std::tuple<at::Tensor,at::Tensor> _fake_quantize_per_tensor_affine_cachemask_tensor_qparams::call(const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, const at::Tensor & fake_quant_enabled, int64_t quant_min, int64_t quant_max) {
    static auto op = create__fake_quantize_per_tensor_affine_cachemask_tensor_qparams_typed_handle();
    return op.call(self, scale, zero_point, fake_quant_enabled, quant_min, quant_max);
}

// aten::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams(Tensor self, Tensor scale, Tensor zero_point, Tensor fake_quant_enabled, int quant_min, int quant_max) -> (Tensor output, Tensor mask)
::std::tuple<at::Tensor,at::Tensor> _fake_quantize_per_tensor_affine_cachemask_tensor_qparams::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, const at::Tensor & fake_quant_enabled, int64_t quant_min, int64_t quant_max) {
    static auto op = create__fake_quantize_per_tensor_affine_cachemask_tensor_qparams_typed_handle();
    return op.redispatch(dispatchKeySet, self, scale, zero_point, fake_quant_enabled, quant_min, quant_max);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fake_quantize_learnable_per_tensor_affine, name, "aten::_fake_quantize_learnable_per_tensor_affine")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fake_quantize_learnable_per_tensor_affine, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fake_quantize_learnable_per_tensor_affine, schema_str, "_fake_quantize_learnable_per_tensor_affine(Tensor self, Tensor scale, Tensor zero_point, int quant_min, int quant_max, float grad_factor=1.0) -> Tensor")

// aten::_fake_quantize_learnable_per_tensor_affine(Tensor self, Tensor scale, Tensor zero_point, int quant_min, int quant_max, float grad_factor=1.0) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_fake_quantize_learnable_per_tensor_affine::schema> create__fake_quantize_learnable_per_tensor_affine_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_fake_quantize_learnable_per_tensor_affine::name, _fake_quantize_learnable_per_tensor_affine::overload_name)
      .typed<_fake_quantize_learnable_per_tensor_affine::schema>();
}

// aten::_fake_quantize_learnable_per_tensor_affine(Tensor self, Tensor scale, Tensor zero_point, int quant_min, int quant_max, float grad_factor=1.0) -> Tensor
at::Tensor _fake_quantize_learnable_per_tensor_affine::call(const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t quant_min, int64_t quant_max, double grad_factor) {
    static auto op = create__fake_quantize_learnable_per_tensor_affine_typed_handle();
    return op.call(self, scale, zero_point, quant_min, quant_max, grad_factor);
}

// aten::_fake_quantize_learnable_per_tensor_affine(Tensor self, Tensor scale, Tensor zero_point, int quant_min, int quant_max, float grad_factor=1.0) -> Tensor
at::Tensor _fake_quantize_learnable_per_tensor_affine::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t quant_min, int64_t quant_max, double grad_factor) {
    static auto op = create__fake_quantize_learnable_per_tensor_affine_typed_handle();
    return op.redispatch(dispatchKeySet, self, scale, zero_point, quant_min, quant_max, grad_factor);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(choose_qparams_optimized, name, "aten::choose_qparams_optimized")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(choose_qparams_optimized, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(choose_qparams_optimized, schema_str, "choose_qparams_optimized(Tensor input, int numel, int n_bins, float ratio, int bit_width) -> (Tensor, Tensor)")

// aten::choose_qparams_optimized(Tensor input, int numel, int n_bins, float ratio, int bit_width) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<choose_qparams_optimized::schema> create_choose_qparams_optimized_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(choose_qparams_optimized::name, choose_qparams_optimized::overload_name)
      .typed<choose_qparams_optimized::schema>();
}

// aten::choose_qparams_optimized(Tensor input, int numel, int n_bins, float ratio, int bit_width) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> choose_qparams_optimized::call(const at::Tensor & input, int64_t numel, int64_t n_bins, double ratio, int64_t bit_width) {
    static auto op = create_choose_qparams_optimized_typed_handle();
    return op.call(input, numel, n_bins, ratio, bit_width);
}

// aten::choose_qparams_optimized(Tensor input, int numel, int n_bins, float ratio, int bit_width) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> choose_qparams_optimized::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, int64_t numel, int64_t n_bins, double ratio, int64_t bit_width) {
    static auto op = create_choose_qparams_optimized_typed_handle();
    return op.redispatch(dispatchKeySet, input, numel, n_bins, ratio, bit_width);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(to_dtype, name, "aten::to")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(to_dtype, overload_name, "dtype")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(to_dtype, schema_str, "to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)")

// aten::to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<to_dtype::schema> create_to_dtype_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(to_dtype::name, to_dtype::overload_name)
      .typed<to_dtype::schema>();
}

// aten::to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)
at::Tensor to_dtype::call(const at::Tensor & self, at::ScalarType dtype, bool non_blocking, bool copy, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create_to_dtype_typed_handle();
    return op.call(self, dtype, non_blocking, copy, memory_format);
}

// aten::to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)
at::Tensor to_dtype::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::ScalarType dtype, bool non_blocking, bool copy, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create_to_dtype_typed_handle();
    return op.redispatch(dispatchKeySet, self, dtype, non_blocking, copy, memory_format);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cartesian_prod, name, "aten::cartesian_prod")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cartesian_prod, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cartesian_prod, schema_str, "cartesian_prod(Tensor[] tensors) -> Tensor")

// aten::cartesian_prod(Tensor[] tensors) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<cartesian_prod::schema> create_cartesian_prod_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cartesian_prod::name, cartesian_prod::overload_name)
      .typed<cartesian_prod::schema>();
}

// aten::cartesian_prod(Tensor[] tensors) -> Tensor
at::Tensor cartesian_prod::call(at::TensorList tensors) {
    static auto op = create_cartesian_prod_typed_handle();
    return op.call(tensors);
}

// aten::cartesian_prod(Tensor[] tensors) -> Tensor
at::Tensor cartesian_prod::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create_cartesian_prod_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(result_type_Tensor, name, "aten::result_type")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(result_type_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(result_type_Tensor, schema_str, "result_type.Tensor(Tensor tensor, Tensor other) -> ScalarType")

// aten::result_type.Tensor(Tensor tensor, Tensor other) -> ScalarType
static C10_NOINLINE c10::TypedOperatorHandle<result_type_Tensor::schema> create_result_type_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(result_type_Tensor::name, result_type_Tensor::overload_name)
      .typed<result_type_Tensor::schema>();
}

// aten::result_type.Tensor(Tensor tensor, Tensor other) -> ScalarType
at::ScalarType result_type_Tensor::call(const at::Tensor & tensor, const at::Tensor & other) {
    static auto op = create_result_type_Tensor_typed_handle();
    return op.call(tensor, other);
}

// aten::result_type.Tensor(Tensor tensor, Tensor other) -> ScalarType
at::ScalarType result_type_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & tensor, const at::Tensor & other) {
    static auto op = create_result_type_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, tensor, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(result_type_Scalar, name, "aten::result_type")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(result_type_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(result_type_Scalar, schema_str, "result_type.Scalar(Tensor tensor, Scalar other) -> ScalarType")

// aten::result_type.Scalar(Tensor tensor, Scalar other) -> ScalarType
static C10_NOINLINE c10::TypedOperatorHandle<result_type_Scalar::schema> create_result_type_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(result_type_Scalar::name, result_type_Scalar::overload_name)
      .typed<result_type_Scalar::schema>();
}

// aten::result_type.Scalar(Tensor tensor, Scalar other) -> ScalarType
at::ScalarType result_type_Scalar::call(const at::Tensor & tensor, const at::Scalar & other) {
    static auto op = create_result_type_Scalar_typed_handle();
    return op.call(tensor, other);
}

// aten::result_type.Scalar(Tensor tensor, Scalar other) -> ScalarType
at::ScalarType result_type_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & tensor, const at::Scalar & other) {
    static auto op = create_result_type_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, tensor, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(result_type_Scalar_Scalar, name, "aten::result_type")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(result_type_Scalar_Scalar, overload_name, "Scalar_Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(result_type_Scalar_Scalar, schema_str, "result_type.Scalar_Scalar(Scalar scalar1, Scalar scalar2) -> ScalarType")

// aten::result_type.Scalar_Scalar(Scalar scalar1, Scalar scalar2) -> ScalarType
static C10_NOINLINE c10::TypedOperatorHandle<result_type_Scalar_Scalar::schema> create_result_type_Scalar_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(result_type_Scalar_Scalar::name, result_type_Scalar_Scalar::overload_name)
      .typed<result_type_Scalar_Scalar::schema>();
}

// aten::result_type.Scalar_Scalar(Scalar scalar1, Scalar scalar2) -> ScalarType
at::ScalarType result_type_Scalar_Scalar::call(const at::Scalar & scalar1, const at::Scalar & scalar2) {
    static auto op = create_result_type_Scalar_Scalar_typed_handle();
    return op.call(scalar1, scalar2);
}

// aten::result_type.Scalar_Scalar(Scalar scalar1, Scalar scalar2) -> ScalarType
at::ScalarType result_type_Scalar_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Scalar & scalar1, const at::Scalar & scalar2) {
    static auto op = create_result_type_Scalar_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, scalar1, scalar2);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(promote_types, name, "aten::promote_types")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(promote_types, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(promote_types, schema_str, "promote_types(ScalarType type1, ScalarType type2) -> ScalarType")

// aten::promote_types(ScalarType type1, ScalarType type2) -> ScalarType
static C10_NOINLINE c10::TypedOperatorHandle<promote_types::schema> create_promote_types_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(promote_types::name, promote_types::overload_name)
      .typed<promote_types::schema>();
}

// aten::promote_types(ScalarType type1, ScalarType type2) -> ScalarType
at::ScalarType promote_types::call(at::ScalarType type1, at::ScalarType type2) {
    static auto op = create_promote_types_typed_handle();
    return op.call(type1, type2);
}

// aten::promote_types(ScalarType type1, ScalarType type2) -> ScalarType
at::ScalarType promote_types::redispatch(c10::DispatchKeySet dispatchKeySet, at::ScalarType type1, at::ScalarType type2) {
    static auto op = create_promote_types_typed_handle();
    return op.redispatch(dispatchKeySet, type1, type2);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_local_scalar_dense, name, "aten::_local_scalar_dense")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_local_scalar_dense, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_local_scalar_dense, schema_str, "_local_scalar_dense(Tensor self) -> Scalar")

// aten::_local_scalar_dense(Tensor self) -> Scalar
static C10_NOINLINE c10::TypedOperatorHandle<_local_scalar_dense::schema> create__local_scalar_dense_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_local_scalar_dense::name, _local_scalar_dense::overload_name)
      .typed<_local_scalar_dense::schema>();
}

// aten::_local_scalar_dense(Tensor self) -> Scalar
at::Scalar _local_scalar_dense::call(const at::Tensor & self) {
    static auto op = create__local_scalar_dense_typed_handle();
    return op.call(self);
}

// aten::_local_scalar_dense(Tensor self) -> Scalar
at::Scalar _local_scalar_dense::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create__local_scalar_dense_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_thnn_fused_gru_cell_backward, name, "aten::_thnn_fused_gru_cell_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_thnn_fused_gru_cell_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_thnn_fused_gru_cell_backward, schema_str, "_thnn_fused_gru_cell_backward(Tensor grad_hy, Tensor workspace, bool has_bias) -> (Tensor, Tensor, Tensor, Tensor, Tensor)")

// aten::_thnn_fused_gru_cell_backward(Tensor grad_hy, Tensor workspace, bool has_bias) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<_thnn_fused_gru_cell_backward::schema> create__thnn_fused_gru_cell_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_thnn_fused_gru_cell_backward::name, _thnn_fused_gru_cell_backward::overload_name)
      .typed<_thnn_fused_gru_cell_backward::schema>();
}

// aten::_thnn_fused_gru_cell_backward(Tensor grad_hy, Tensor workspace, bool has_bias) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> _thnn_fused_gru_cell_backward::call(const at::Tensor & grad_hy, const at::Tensor & workspace, bool has_bias) {
    static auto op = create__thnn_fused_gru_cell_backward_typed_handle();
    return op.call(grad_hy, workspace, has_bias);
}

// aten::_thnn_fused_gru_cell_backward(Tensor grad_hy, Tensor workspace, bool has_bias) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> _thnn_fused_gru_cell_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_hy, const at::Tensor & workspace, bool has_bias) {
    static auto op = create__thnn_fused_gru_cell_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_hy, workspace, has_bias);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lstm_input, name, "aten::lstm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lstm_input, overload_name, "input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lstm_input, schema_str, "lstm.input(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor, Tensor)")

// aten::lstm.input(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<lstm_input::schema> create_lstm_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(lstm_input::name, lstm_input::overload_name)
      .typed<lstm_input::schema>();
}

// aten::lstm.input(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> lstm_input::call(const at::Tensor & input, at::TensorList hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
    static auto op = create_lstm_input_typed_handle();
    return op.call(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}

// aten::lstm.input(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> lstm_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, at::TensorList hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
    static auto op = create_lstm_input_typed_handle();
    return op.redispatch(dispatchKeySet, input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gru_data, name, "aten::gru")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gru_data, overload_name, "data")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gru_data, schema_str, "gru.data(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)")

// aten::gru.data(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<gru_data::schema> create_gru_data_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(gru_data::name, gru_data::overload_name)
      .typed<gru_data::schema>();
}

// aten::gru.data(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> gru_data::call(const at::Tensor & data, const at::Tensor & batch_sizes, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {
    static auto op = create_gru_data_typed_handle();
    return op.call(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
}

// aten::gru.data(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> gru_data::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & data, const at::Tensor & batch_sizes, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {
    static auto op = create_gru_data_typed_handle();
    return op.redispatch(dispatchKeySet, data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gru_cell, name, "aten::gru_cell")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gru_cell, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gru_cell, schema_str, "gru_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor")

// aten::gru_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<gru_cell::schema> create_gru_cell_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(gru_cell::name, gru_cell::overload_name)
      .typed<gru_cell::schema>();
}

// aten::gru_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor
at::Tensor gru_cell::call(const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const c10::optional<at::Tensor> & b_ih, const c10::optional<at::Tensor> & b_hh) {
    static auto op = create_gru_cell_typed_handle();
    return op.call(input, hx, w_ih, w_hh, b_ih, b_hh);
}

// aten::gru_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor
at::Tensor gru_cell::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const c10::optional<at::Tensor> & b_ih, const c10::optional<at::Tensor> & b_hh) {
    static auto op = create_gru_cell_typed_handle();
    return op.redispatch(dispatchKeySet, input, hx, w_ih, w_hh, b_ih, b_hh);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantized_lstm_cell, name, "aten::quantized_lstm_cell")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantized_lstm_cell, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantized_lstm_cell, schema_str, "quantized_lstm_cell(Tensor input, Tensor[] hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> (Tensor, Tensor)")

// aten::quantized_lstm_cell(Tensor input, Tensor[] hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<quantized_lstm_cell::schema> create_quantized_lstm_cell_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(quantized_lstm_cell::name, quantized_lstm_cell::overload_name)
      .typed<quantized_lstm_cell::schema>();
}

// aten::quantized_lstm_cell(Tensor input, Tensor[] hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> quantized_lstm_cell::call(const at::Tensor & input, at::TensorList hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const at::Tensor & b_ih, const at::Tensor & b_hh, const at::Tensor & packed_ih, const at::Tensor & packed_hh, const at::Tensor & col_offsets_ih, const at::Tensor & col_offsets_hh, const at::Scalar & scale_ih, const at::Scalar & scale_hh, const at::Scalar & zero_point_ih, const at::Scalar & zero_point_hh) {
    static auto op = create_quantized_lstm_cell_typed_handle();
    return op.call(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
}

// aten::quantized_lstm_cell(Tensor input, Tensor[] hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> quantized_lstm_cell::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, at::TensorList hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const at::Tensor & b_ih, const at::Tensor & b_hh, const at::Tensor & packed_ih, const at::Tensor & packed_hh, const at::Tensor & col_offsets_ih, const at::Tensor & col_offsets_hh, const at::Scalar & scale_ih, const at::Scalar & scale_hh, const at::Scalar & zero_point_ih, const at::Scalar & zero_point_hh) {
    static auto op = create_quantized_lstm_cell_typed_handle();
    return op.redispatch(dispatchKeySet, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(set__source_Storage_storage_offset, name, "aten::set_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(set__source_Storage_storage_offset, overload_name, "source_Storage_storage_offset")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(set__source_Storage_storage_offset, schema_str, "set_.source_Storage_storage_offset(Tensor(a!) self, Storage source, int storage_offset, int[] size, int[] stride=[]) -> Tensor(a!)")

// aten::set_.source_Storage_storage_offset(Tensor(a!) self, Storage source, int storage_offset, int[] size, int[] stride=[]) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<set__source_Storage_storage_offset::schema> create_set__source_Storage_storage_offset_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(set__source_Storage_storage_offset::name, set__source_Storage_storage_offset::overload_name)
      .typed<set__source_Storage_storage_offset::schema>();
}

// aten::set_.source_Storage_storage_offset(Tensor(a!) self, Storage source, int storage_offset, int[] size, int[] stride=[]) -> Tensor(a!)
at::Tensor & set__source_Storage_storage_offset::call(at::Tensor & self, at::Storage source, int64_t storage_offset, at::IntArrayRef size, at::IntArrayRef stride) {
    static auto op = create_set__source_Storage_storage_offset_typed_handle();
    return op.call(self, source, storage_offset, size, stride);
}

// aten::set_.source_Storage_storage_offset(Tensor(a!) self, Storage source, int storage_offset, int[] size, int[] stride=[]) -> Tensor(a!)
at::Tensor & set__source_Storage_storage_offset::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, at::Storage source, int64_t storage_offset, at::IntArrayRef size, at::IntArrayRef stride) {
    static auto op = create_set__source_Storage_storage_offset_typed_handle();
    return op.redispatch(dispatchKeySet, self, source, storage_offset, size, stride);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(masked_fill__Scalar, name, "aten::masked_fill_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(masked_fill__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(masked_fill__Scalar, schema_str, "masked_fill_.Scalar(Tensor(a!) self, Tensor mask, Scalar value) -> Tensor(a!)")

// aten::masked_fill_.Scalar(Tensor(a!) self, Tensor mask, Scalar value) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<masked_fill__Scalar::schema> create_masked_fill__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(masked_fill__Scalar::name, masked_fill__Scalar::overload_name)
      .typed<masked_fill__Scalar::schema>();
}

// aten::masked_fill_.Scalar(Tensor(a!) self, Tensor mask, Scalar value) -> Tensor(a!)
at::Tensor & masked_fill__Scalar::call(at::Tensor & self, const at::Tensor & mask, const at::Scalar & value) {
    static auto op = create_masked_fill__Scalar_typed_handle();
    return op.call(self, mask, value);
}

// aten::masked_fill_.Scalar(Tensor(a!) self, Tensor mask, Scalar value) -> Tensor(a!)
at::Tensor & masked_fill__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & mask, const at::Scalar & value) {
    static auto op = create_masked_fill__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, mask, value);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(masked_fill__Tensor, name, "aten::masked_fill_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(masked_fill__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(masked_fill__Tensor, schema_str, "masked_fill_.Tensor(Tensor(a!) self, Tensor mask, Tensor value) -> Tensor(a!)")

// aten::masked_fill_.Tensor(Tensor(a!) self, Tensor mask, Tensor value) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<masked_fill__Tensor::schema> create_masked_fill__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(masked_fill__Tensor::name, masked_fill__Tensor::overload_name)
      .typed<masked_fill__Tensor::schema>();
}

// aten::masked_fill_.Tensor(Tensor(a!) self, Tensor mask, Tensor value) -> Tensor(a!)
at::Tensor & masked_fill__Tensor::call(at::Tensor & self, const at::Tensor & mask, const at::Tensor & value) {
    static auto op = create_masked_fill__Tensor_typed_handle();
    return op.call(self, mask, value);
}

// aten::masked_fill_.Tensor(Tensor(a!) self, Tensor mask, Tensor value) -> Tensor(a!)
at::Tensor & masked_fill__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & mask, const at::Tensor & value) {
    static auto op = create_masked_fill__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, mask, value);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(put, name, "aten::put")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(put, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(put, schema_str, "put(Tensor self, Tensor index, Tensor source, bool accumulate=False) -> Tensor")

// aten::put(Tensor self, Tensor index, Tensor source, bool accumulate=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<put::schema> create_put_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(put::name, put::overload_name)
      .typed<put::schema>();
}

// aten::put(Tensor self, Tensor index, Tensor source, bool accumulate=False) -> Tensor
at::Tensor put::call(const at::Tensor & self, const at::Tensor & index, const at::Tensor & source, bool accumulate) {
    static auto op = create_put_typed_handle();
    return op.call(self, index, source, accumulate);
}

// aten::put(Tensor self, Tensor index, Tensor source, bool accumulate=False) -> Tensor
at::Tensor put::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & index, const at::Tensor & source, bool accumulate) {
    static auto op = create_put_typed_handle();
    return op.redispatch(dispatchKeySet, self, index, source, accumulate);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_add_alpha, name, "aten::index_add")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_add_alpha, overload_name, "alpha")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_add_alpha, schema_str, "index_add.alpha(Tensor self, int dim, Tensor index, Tensor source, *, Scalar alpha) -> Tensor")

// aten::index_add.alpha(Tensor self, int dim, Tensor index, Tensor source, *, Scalar alpha) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<index_add_alpha::schema> create_index_add_alpha_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(index_add_alpha::name, index_add_alpha::overload_name)
      .typed<index_add_alpha::schema>();
}

// aten::index_add.alpha(Tensor self, int dim, Tensor index, Tensor source, *, Scalar alpha) -> Tensor
at::Tensor index_add_alpha::call(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha) {
    static auto op = create_index_add_alpha_typed_handle();
    return op.call(self, dim, index, source, alpha);
}

// aten::index_add.alpha(Tensor self, int dim, Tensor index, Tensor source, *, Scalar alpha) -> Tensor
at::Tensor index_add_alpha::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha) {
    static auto op = create_index_add_alpha_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, source, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_fill__int_Scalar, name, "aten::index_fill_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_fill__int_Scalar, overload_name, "int_Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_fill__int_Scalar, schema_str, "index_fill_.int_Scalar(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)")

// aten::index_fill_.int_Scalar(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<index_fill__int_Scalar::schema> create_index_fill__int_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(index_fill__int_Scalar::name, index_fill__int_Scalar::overload_name)
      .typed<index_fill__int_Scalar::schema>();
}

// aten::index_fill_.int_Scalar(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)
at::Tensor & index_fill__int_Scalar::call(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) {
    static auto op = create_index_fill__int_Scalar_typed_handle();
    return op.call(self, dim, index, value);
}

// aten::index_fill_.int_Scalar(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)
at::Tensor & index_fill__int_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) {
    static auto op = create_index_fill__int_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, value);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_fill__int_Tensor, name, "aten::index_fill_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_fill__int_Tensor, overload_name, "int_Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_fill__int_Tensor, schema_str, "index_fill_.int_Tensor(Tensor(a!) self, int dim, Tensor index, Tensor value) -> Tensor(a!)")

// aten::index_fill_.int_Tensor(Tensor(a!) self, int dim, Tensor index, Tensor value) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<index_fill__int_Tensor::schema> create_index_fill__int_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(index_fill__int_Tensor::name, index_fill__int_Tensor::overload_name)
      .typed<index_fill__int_Tensor::schema>();
}

// aten::index_fill_.int_Tensor(Tensor(a!) self, int dim, Tensor index, Tensor value) -> Tensor(a!)
at::Tensor & index_fill__int_Tensor::call(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & value) {
    static auto op = create_index_fill__int_Tensor_typed_handle();
    return op.call(self, dim, index, value);
}

// aten::index_fill_.int_Tensor(Tensor(a!) self, int dim, Tensor index, Tensor value) -> Tensor(a!)
at::Tensor & index_fill__int_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & value) {
    static auto op = create_index_fill__int_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, value);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_fill_Dimname_Tensor, name, "aten::index_fill")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_fill_Dimname_Tensor, overload_name, "Dimname_Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_fill_Dimname_Tensor, schema_str, "index_fill.Dimname_Tensor(Tensor self, Dimname dim, Tensor index, Tensor value) -> Tensor")

// aten::index_fill.Dimname_Tensor(Tensor self, Dimname dim, Tensor index, Tensor value) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<index_fill_Dimname_Tensor::schema> create_index_fill_Dimname_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(index_fill_Dimname_Tensor::name, index_fill_Dimname_Tensor::overload_name)
      .typed<index_fill_Dimname_Tensor::schema>();
}

// aten::index_fill.Dimname_Tensor(Tensor self, Dimname dim, Tensor index, Tensor value) -> Tensor
at::Tensor index_fill_Dimname_Tensor::call(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & value) {
    static auto op = create_index_fill_Dimname_Tensor_typed_handle();
    return op.call(self, dim, index, value);
}

// aten::index_fill.Dimname_Tensor(Tensor self, Dimname dim, Tensor index, Tensor value) -> Tensor
at::Tensor index_fill_Dimname_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & value) {
    static auto op = create_index_fill_Dimname_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, value);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_src, name, "aten::scatter")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_src, overload_name, "src")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_src, schema_str, "scatter.src(Tensor self, int dim, Tensor index, Tensor src) -> Tensor")

// aten::scatter.src(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<scatter_src::schema> create_scatter_src_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(scatter_src::name, scatter_src::overload_name)
      .typed<scatter_src::schema>();
}

// aten::scatter.src(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
at::Tensor scatter_src::call(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src) {
    static auto op = create_scatter_src_typed_handle();
    return op.call(self, dim, index, src);
}

// aten::scatter.src(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
at::Tensor scatter_src::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src) {
    static auto op = create_scatter_src_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, src);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter__src, name, "aten::scatter_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter__src, overload_name, "src")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter__src, schema_str, "scatter_.src(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)")

// aten::scatter_.src(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<scatter__src::schema> create_scatter__src_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(scatter__src::name, scatter__src::overload_name)
      .typed<scatter__src::schema>();
}

// aten::scatter_.src(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)
at::Tensor & scatter__src::call(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src) {
    static auto op = create_scatter__src_typed_handle();
    return op.call(self, dim, index, src);
}

// aten::scatter_.src(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)
at::Tensor & scatter__src::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src) {
    static auto op = create_scatter__src_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, src);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_value, name, "aten::scatter")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_value, overload_name, "value")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_value, schema_str, "scatter.value(Tensor self, int dim, Tensor index, Scalar value) -> Tensor")

// aten::scatter.value(Tensor self, int dim, Tensor index, Scalar value) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<scatter_value::schema> create_scatter_value_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(scatter_value::name, scatter_value::overload_name)
      .typed<scatter_value::schema>();
}

// aten::scatter.value(Tensor self, int dim, Tensor index, Scalar value) -> Tensor
at::Tensor scatter_value::call(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) {
    static auto op = create_scatter_value_typed_handle();
    return op.call(self, dim, index, value);
}

// aten::scatter.value(Tensor self, int dim, Tensor index, Scalar value) -> Tensor
at::Tensor scatter_value::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) {
    static auto op = create_scatter_value_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, value);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_reduce, name, "aten::scatter")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_reduce, overload_name, "reduce")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_reduce, schema_str, "scatter.reduce(Tensor self, int dim, Tensor index, Tensor src, *, str reduce) -> Tensor")

// aten::scatter.reduce(Tensor self, int dim, Tensor index, Tensor src, *, str reduce) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<scatter_reduce::schema> create_scatter_reduce_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(scatter_reduce::name, scatter_reduce::overload_name)
      .typed<scatter_reduce::schema>();
}

// aten::scatter.reduce(Tensor self, int dim, Tensor index, Tensor src, *, str reduce) -> Tensor
at::Tensor scatter_reduce::call(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce) {
    static auto op = create_scatter_reduce_typed_handle();
    return op.call(self, dim, index, src, reduce);
}

// aten::scatter.reduce(Tensor self, int dim, Tensor index, Tensor src, *, str reduce) -> Tensor
at::Tensor scatter_reduce::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce) {
    static auto op = create_scatter_reduce_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, src, reduce);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter__value_reduce, name, "aten::scatter_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter__value_reduce, overload_name, "value_reduce")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter__value_reduce, schema_str, "scatter_.value_reduce(Tensor(a!) self, int dim, Tensor index, Scalar value, *, str reduce) -> Tensor(a!)")

// aten::scatter_.value_reduce(Tensor(a!) self, int dim, Tensor index, Scalar value, *, str reduce) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<scatter__value_reduce::schema> create_scatter__value_reduce_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(scatter__value_reduce::name, scatter__value_reduce::overload_name)
      .typed<scatter__value_reduce::schema>();
}

// aten::scatter_.value_reduce(Tensor(a!) self, int dim, Tensor index, Scalar value, *, str reduce) -> Tensor(a!)
at::Tensor & scatter__value_reduce::call(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, c10::string_view reduce) {
    static auto op = create_scatter__value_reduce_typed_handle();
    return op.call(self, dim, index, value, reduce);
}

// aten::scatter_.value_reduce(Tensor(a!) self, int dim, Tensor index, Scalar value, *, str reduce) -> Tensor(a!)
at::Tensor & scatter__value_reduce::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, c10::string_view reduce) {
    static auto op = create_scatter__value_reduce_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, value, reduce);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_dimname_src, name, "aten::scatter")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_dimname_src, overload_name, "dimname_src")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_dimname_src, schema_str, "scatter.dimname_src(Tensor self, Dimname dim, Tensor index, Tensor src) -> Tensor")

// aten::scatter.dimname_src(Tensor self, Dimname dim, Tensor index, Tensor src) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<scatter_dimname_src::schema> create_scatter_dimname_src_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(scatter_dimname_src::name, scatter_dimname_src::overload_name)
      .typed<scatter_dimname_src::schema>();
}

// aten::scatter.dimname_src(Tensor self, Dimname dim, Tensor index, Tensor src) -> Tensor
at::Tensor scatter_dimname_src::call(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & src) {
    static auto op = create_scatter_dimname_src_typed_handle();
    return op.call(self, dim, index, src);
}

// aten::scatter.dimname_src(Tensor self, Dimname dim, Tensor index, Tensor src) -> Tensor
at::Tensor scatter_dimname_src::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & src) {
    static auto op = create_scatter_dimname_src_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, src);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_add_out, name, "aten::scatter_add")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_add_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_add_out, schema_str, "scatter_add.out(Tensor self, int dim, Tensor index, Tensor src, *, Tensor(a!) out) -> Tensor(a!)")

// aten::scatter_add.out(Tensor self, int dim, Tensor index, Tensor src, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<scatter_add_out::schema> create_scatter_add_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(scatter_add_out::name, scatter_add_out::overload_name)
      .typed<scatter_add_out::schema>();
}

// aten::scatter_add.out(Tensor self, int dim, Tensor index, Tensor src, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & scatter_add_out::call(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, at::Tensor & out) {
    static auto op = create_scatter_add_out_typed_handle();
    return op.call(self, dim, index, src, out);
}

// aten::scatter_add.out(Tensor self, int dim, Tensor index, Tensor src, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & scatter_add_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, at::Tensor & out) {
    static auto op = create_scatter_add_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, src, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_add_dimname, name, "aten::scatter_add")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_add_dimname, overload_name, "dimname")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_add_dimname, schema_str, "scatter_add.dimname(Tensor self, Dimname dim, Tensor index, Tensor src) -> Tensor")

// aten::scatter_add.dimname(Tensor self, Dimname dim, Tensor index, Tensor src) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<scatter_add_dimname::schema> create_scatter_add_dimname_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(scatter_add_dimname::name, scatter_add_dimname::overload_name)
      .typed<scatter_add_dimname::schema>();
}

// aten::scatter_add.dimname(Tensor self, Dimname dim, Tensor index, Tensor src) -> Tensor
at::Tensor scatter_add_dimname::call(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & src) {
    static auto op = create_scatter_add_dimname_typed_handle();
    return op.call(self, dim, index, src);
}

// aten::scatter_add.dimname(Tensor self, Dimname dim, Tensor index, Tensor src) -> Tensor
at::Tensor scatter_add_dimname::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & src) {
    static auto op = create_scatter_add_dimname_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, src);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(eq__Tensor, name, "aten::eq_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(eq__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(eq__Tensor, schema_str, "eq_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::eq_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<eq__Tensor::schema> create_eq__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(eq__Tensor::name, eq__Tensor::overload_name)
      .typed<eq__Tensor::schema>();
}

// aten::eq_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & eq__Tensor::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create_eq__Tensor_typed_handle();
    return op.call(self, other);
}

// aten::eq_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & eq__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create_eq__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_and_Scalar_out, name, "aten::bitwise_and")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_and_Scalar_out, overload_name, "Scalar_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_and_Scalar_out, schema_str, "bitwise_and.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::bitwise_and.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<bitwise_and_Scalar_out::schema> create_bitwise_and_Scalar_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bitwise_and_Scalar_out::name, bitwise_and_Scalar_out::overload_name)
      .typed<bitwise_and_Scalar_out::schema>();
}

// aten::bitwise_and.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & bitwise_and_Scalar_out::call(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_bitwise_and_Scalar_out_typed_handle();
    return op.call(self, other, out);
}

// aten::bitwise_and.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & bitwise_and_Scalar_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_bitwise_and_Scalar_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__iand___Scalar, name, "aten::__iand__")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__iand___Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__iand___Scalar, schema_str, "__iand__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)")

// aten::__iand__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<__iand___Scalar::schema> create___iand___Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(__iand___Scalar::name, __iand___Scalar::overload_name)
      .typed<__iand___Scalar::schema>();
}

// aten::__iand__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & __iand___Scalar::call(at::Tensor & self, const at::Scalar & other) {
    static auto op = create___iand___Scalar_typed_handle();
    return op.call(self, other);
}

// aten::__iand__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & __iand___Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
    static auto op = create___iand___Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_or_Scalar_out, name, "aten::bitwise_or")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_or_Scalar_out, overload_name, "Scalar_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_or_Scalar_out, schema_str, "bitwise_or.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::bitwise_or.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<bitwise_or_Scalar_out::schema> create_bitwise_or_Scalar_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bitwise_or_Scalar_out::name, bitwise_or_Scalar_out::overload_name)
      .typed<bitwise_or_Scalar_out::schema>();
}

// aten::bitwise_or.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & bitwise_or_Scalar_out::call(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_bitwise_or_Scalar_out_typed_handle();
    return op.call(self, other, out);
}

// aten::bitwise_or.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & bitwise_or_Scalar_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_bitwise_or_Scalar_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__or___Scalar, name, "aten::__or__")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__or___Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__or___Scalar, schema_str, "__or__.Scalar(Tensor self, Scalar other) -> Tensor")

// aten::__or__.Scalar(Tensor self, Scalar other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<__or___Scalar::schema> create___or___Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(__or___Scalar::name, __or___Scalar::overload_name)
      .typed<__or___Scalar::schema>();
}

// aten::__or__.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor __or___Scalar::call(const at::Tensor & self, const at::Scalar & other) {
    static auto op = create___or___Scalar_typed_handle();
    return op.call(self, other);
}

// aten::__or__.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor __or___Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
    static auto op = create___or___Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_xor__Tensor, name, "aten::bitwise_xor_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_xor__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_xor__Tensor, schema_str, "bitwise_xor_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::bitwise_xor_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<bitwise_xor__Tensor::schema> create_bitwise_xor__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bitwise_xor__Tensor::name, bitwise_xor__Tensor::overload_name)
      .typed<bitwise_xor__Tensor::schema>();
}

// aten::bitwise_xor_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & bitwise_xor__Tensor::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create_bitwise_xor__Tensor_typed_handle();
    return op.call(self, other);
}

// aten::bitwise_xor_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & bitwise_xor__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create_bitwise_xor__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__xor___Scalar, name, "aten::__xor__")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__xor___Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__xor___Scalar, schema_str, "__xor__.Scalar(Tensor self, Scalar other) -> Tensor")

// aten::__xor__.Scalar(Tensor self, Scalar other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<__xor___Scalar::schema> create___xor___Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(__xor___Scalar::name, __xor___Scalar::overload_name)
      .typed<__xor___Scalar::schema>();
}

// aten::__xor__.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor __xor___Scalar::call(const at::Tensor & self, const at::Scalar & other) {
    static auto op = create___xor___Scalar_typed_handle();
    return op.call(self, other);
}

// aten::__xor__.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor __xor___Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
    static auto op = create___xor___Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__xor___Tensor, name, "aten::__xor__")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__xor___Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__xor___Tensor, schema_str, "__xor__.Tensor(Tensor self, Tensor other) -> Tensor")

// aten::__xor__.Tensor(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<__xor___Tensor::schema> create___xor___Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(__xor___Tensor::name, __xor___Tensor::overload_name)
      .typed<__xor___Tensor::schema>();
}

// aten::__xor__.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor __xor___Tensor::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create___xor___Tensor_typed_handle();
    return op.call(self, other);
}

// aten::__xor__.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor __xor___Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create___xor___Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__lshift___Tensor, name, "aten::__lshift__")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__lshift___Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__lshift___Tensor, schema_str, "__lshift__.Tensor(Tensor self, Tensor other) -> Tensor")

// aten::__lshift__.Tensor(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<__lshift___Tensor::schema> create___lshift___Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(__lshift___Tensor::name, __lshift___Tensor::overload_name)
      .typed<__lshift___Tensor::schema>();
}

// aten::__lshift__.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor __lshift___Tensor::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create___lshift___Tensor_typed_handle();
    return op.call(self, other);
}

// aten::__lshift__.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor __lshift___Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create___lshift___Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_left_shift__Tensor, name, "aten::bitwise_left_shift_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_left_shift__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_left_shift__Tensor, schema_str, "bitwise_left_shift_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::bitwise_left_shift_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<bitwise_left_shift__Tensor::schema> create_bitwise_left_shift__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bitwise_left_shift__Tensor::name, bitwise_left_shift__Tensor::overload_name)
      .typed<bitwise_left_shift__Tensor::schema>();
}

// aten::bitwise_left_shift_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & bitwise_left_shift__Tensor::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create_bitwise_left_shift__Tensor_typed_handle();
    return op.call(self, other);
}

// aten::bitwise_left_shift_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & bitwise_left_shift__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create_bitwise_left_shift__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_left_shift_Tensor_out, name, "aten::bitwise_left_shift")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_left_shift_Tensor_out, overload_name, "Tensor_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_left_shift_Tensor_out, schema_str, "bitwise_left_shift.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::bitwise_left_shift.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<bitwise_left_shift_Tensor_out::schema> create_bitwise_left_shift_Tensor_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bitwise_left_shift_Tensor_out::name, bitwise_left_shift_Tensor_out::overload_name)
      .typed<bitwise_left_shift_Tensor_out::schema>();
}

// aten::bitwise_left_shift.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & bitwise_left_shift_Tensor_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_bitwise_left_shift_Tensor_out_typed_handle();
    return op.call(self, other, out);
}

// aten::bitwise_left_shift.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & bitwise_left_shift_Tensor_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_bitwise_left_shift_Tensor_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__irshift___Scalar, name, "aten::__irshift__")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__irshift___Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__irshift___Scalar, schema_str, "__irshift__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)")

// aten::__irshift__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<__irshift___Scalar::schema> create___irshift___Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(__irshift___Scalar::name, __irshift___Scalar::overload_name)
      .typed<__irshift___Scalar::schema>();
}

// aten::__irshift__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & __irshift___Scalar::call(at::Tensor & self, const at::Scalar & other) {
    static auto op = create___irshift___Scalar_typed_handle();
    return op.call(self, other);
}

// aten::__irshift__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & __irshift___Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
    static auto op = create___irshift___Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_right_shift__Tensor, name, "aten::bitwise_right_shift_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_right_shift__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_right_shift__Tensor, schema_str, "bitwise_right_shift_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::bitwise_right_shift_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<bitwise_right_shift__Tensor::schema> create_bitwise_right_shift__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bitwise_right_shift__Tensor::name, bitwise_right_shift__Tensor::overload_name)
      .typed<bitwise_right_shift__Tensor::schema>();
}

// aten::bitwise_right_shift_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & bitwise_right_shift__Tensor::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create_bitwise_right_shift__Tensor_typed_handle();
    return op.call(self, other);
}

// aten::bitwise_right_shift_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & bitwise_right_shift__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create_bitwise_right_shift__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_right_shift__Tensor_Scalar, name, "aten::bitwise_right_shift_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_right_shift__Tensor_Scalar, overload_name, "Tensor_Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_right_shift__Tensor_Scalar, schema_str, "bitwise_right_shift_.Tensor_Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)")

// aten::bitwise_right_shift_.Tensor_Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<bitwise_right_shift__Tensor_Scalar::schema> create_bitwise_right_shift__Tensor_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bitwise_right_shift__Tensor_Scalar::name, bitwise_right_shift__Tensor_Scalar::overload_name)
      .typed<bitwise_right_shift__Tensor_Scalar::schema>();
}

// aten::bitwise_right_shift_.Tensor_Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & bitwise_right_shift__Tensor_Scalar::call(at::Tensor & self, const at::Scalar & other) {
    static auto op = create_bitwise_right_shift__Tensor_Scalar_typed_handle();
    return op.call(self, other);
}

// aten::bitwise_right_shift_.Tensor_Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & bitwise_right_shift__Tensor_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
    static auto op = create_bitwise_right_shift__Tensor_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lerp__Scalar, name, "aten::lerp_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lerp__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lerp__Scalar, schema_str, "lerp_.Scalar(Tensor(a!) self, Tensor end, Scalar weight) -> Tensor(a!)")

// aten::lerp_.Scalar(Tensor(a!) self, Tensor end, Scalar weight) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<lerp__Scalar::schema> create_lerp__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(lerp__Scalar::name, lerp__Scalar::overload_name)
      .typed<lerp__Scalar::schema>();
}

// aten::lerp_.Scalar(Tensor(a!) self, Tensor end, Scalar weight) -> Tensor(a!)
at::Tensor & lerp__Scalar::call(at::Tensor & self, const at::Tensor & end, const at::Scalar & weight) {
    static auto op = create_lerp__Scalar_typed_handle();
    return op.call(self, end, weight);
}

// aten::lerp_.Scalar(Tensor(a!) self, Tensor end, Scalar weight) -> Tensor(a!)
at::Tensor & lerp__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & end, const at::Scalar & weight) {
    static auto op = create_lerp__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, end, weight);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addbmm_, name, "aten::addbmm_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addbmm_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addbmm_, schema_str, "addbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)")

// aten::addbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<addbmm_::schema> create_addbmm__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(addbmm_::name, addbmm_::overload_name)
      .typed<addbmm_::schema>();
}

// aten::addbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
at::Tensor & addbmm_::call(at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha) {
    static auto op = create_addbmm__typed_handle();
    return op.call(self, batch1, batch2, beta, alpha);
}

// aten::addbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
at::Tensor & addbmm_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha) {
    static auto op = create_addbmm__typed_handle();
    return op.redispatch(dispatchKeySet, self, batch1, batch2, beta, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(random__from, name, "aten::random_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(random__from, overload_name, "from")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(random__from, schema_str, "random_.from(Tensor(a!) self, int from, int? to, *, Generator? generator=None) -> Tensor(a!)")

// aten::random_.from(Tensor(a!) self, int from, int? to, *, Generator? generator=None) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<random__from::schema> create_random__from_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(random__from::name, random__from::overload_name)
      .typed<random__from::schema>();
}

// aten::random_.from(Tensor(a!) self, int from, int? to, *, Generator? generator=None) -> Tensor(a!)
at::Tensor & random__from::call(at::Tensor & self, int64_t from, c10::optional<int64_t> to, c10::optional<at::Generator> generator) {
    static auto op = create_random__from_typed_handle();
    return op.call(self, from, to, generator);
}

// aten::random_.from(Tensor(a!) self, int from, int? to, *, Generator? generator=None) -> Tensor(a!)
at::Tensor & random__from::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t from, c10::optional<int64_t> to, c10::optional<at::Generator> generator) {
    static auto op = create_random__from_typed_handle();
    return op.redispatch(dispatchKeySet, self, from, to, generator);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(random_, name, "aten::random_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(random_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(random_, schema_str, "random_(Tensor(a!) self, *, Generator? generator=None) -> Tensor(a!)")

// aten::random_(Tensor(a!) self, *, Generator? generator=None) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<random_::schema> create_random__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(random_::name, random_::overload_name)
      .typed<random_::schema>();
}

// aten::random_(Tensor(a!) self, *, Generator? generator=None) -> Tensor(a!)
at::Tensor & random_::call(at::Tensor & self, c10::optional<at::Generator> generator) {
    static auto op = create_random__typed_handle();
    return op.call(self, generator);
}

// aten::random_(Tensor(a!) self, *, Generator? generator=None) -> Tensor(a!)
at::Tensor & random_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, c10::optional<at::Generator> generator) {
    static auto op = create_random__typed_handle();
    return op.redispatch(dispatchKeySet, self, generator);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(diag_out, name, "aten::diag")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(diag_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(diag_out, schema_str, "diag.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)")

// aten::diag.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<diag_out::schema> create_diag_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(diag_out::name, diag_out::overload_name)
      .typed<diag_out::schema>();
}

// aten::diag.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & diag_out::call(const at::Tensor & self, int64_t diagonal, at::Tensor & out) {
    static auto op = create_diag_out_typed_handle();
    return op.call(self, diagonal, out);
}

// aten::diag.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & diag_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t diagonal, at::Tensor & out) {
    static auto op = create_diag_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, diagonal, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(diag_backward, name, "aten::diag_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(diag_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(diag_backward, schema_str, "diag_backward(Tensor grad, int[] input_sizes, int diagonal) -> Tensor")

// aten::diag_backward(Tensor grad, int[] input_sizes, int diagonal) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<diag_backward::schema> create_diag_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(diag_backward::name, diag_backward::overload_name)
      .typed<diag_backward::schema>();
}

// aten::diag_backward(Tensor grad, int[] input_sizes, int diagonal) -> Tensor
at::Tensor diag_backward::call(const at::Tensor & grad, at::IntArrayRef input_sizes, int64_t diagonal) {
    static auto op = create_diag_backward_typed_handle();
    return op.call(grad, input_sizes, diagonal);
}

// aten::diag_backward(Tensor grad, int[] input_sizes, int diagonal) -> Tensor
at::Tensor diag_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, at::IntArrayRef input_sizes, int64_t diagonal) {
    static auto op = create_diag_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad, input_sizes, diagonal);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tril, name, "aten::tril")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tril, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tril, schema_str, "tril(Tensor self, int diagonal=0) -> Tensor")

// aten::tril(Tensor self, int diagonal=0) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<tril::schema> create_tril_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(tril::name, tril::overload_name)
      .typed<tril::schema>();
}

// aten::tril(Tensor self, int diagonal=0) -> Tensor
at::Tensor tril::call(const at::Tensor & self, int64_t diagonal) {
    static auto op = create_tril_typed_handle();
    return op.call(self, diagonal);
}

// aten::tril(Tensor self, int diagonal=0) -> Tensor
at::Tensor tril::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t diagonal) {
    static auto op = create_tril_typed_handle();
    return op.redispatch(dispatchKeySet, self, diagonal);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tril_indices, name, "aten::tril_indices")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tril_indices, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tril_indices, schema_str, "tril_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::tril_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<tril_indices::schema> create_tril_indices_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(tril_indices::name, tril_indices::overload_name)
      .typed<tril_indices::schema>();
}

// aten::tril_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor tril_indices::call(int64_t row, int64_t col, int64_t offset, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_tril_indices_typed_handle();
    return op.call(row, col, offset, dtype, layout, device, pin_memory);
}

// aten::tril_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor tril_indices::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t row, int64_t col, int64_t offset, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_tril_indices_typed_handle();
    return op.redispatch(dispatchKeySet, row, col, offset, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ne_Tensor, name, "aten::ne")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ne_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ne_Tensor, schema_str, "ne.Tensor(Tensor self, Tensor other) -> Tensor")

// aten::ne.Tensor(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<ne_Tensor::schema> create_ne_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(ne_Tensor::name, ne_Tensor::overload_name)
      .typed<ne_Tensor::schema>();
}

// aten::ne.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor ne_Tensor::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_ne_Tensor_typed_handle();
    return op.call(self, other);
}

// aten::ne.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor ne_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_ne_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(not_equal_Tensor, name, "aten::not_equal")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(not_equal_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(not_equal_Tensor, schema_str, "not_equal.Tensor(Tensor self, Tensor other) -> Tensor")

// aten::not_equal.Tensor(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<not_equal_Tensor::schema> create_not_equal_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(not_equal_Tensor::name, not_equal_Tensor::overload_name)
      .typed<not_equal_Tensor::schema>();
}

// aten::not_equal.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor not_equal_Tensor::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_not_equal_Tensor_typed_handle();
    return op.call(self, other);
}

// aten::not_equal.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor not_equal_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_not_equal_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(greater_equal_Scalar_out, name, "aten::greater_equal")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(greater_equal_Scalar_out, overload_name, "Scalar_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(greater_equal_Scalar_out, schema_str, "greater_equal.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::greater_equal.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<greater_equal_Scalar_out::schema> create_greater_equal_Scalar_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(greater_equal_Scalar_out::name, greater_equal_Scalar_out::overload_name)
      .typed<greater_equal_Scalar_out::schema>();
}

// aten::greater_equal.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & greater_equal_Scalar_out::call(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_greater_equal_Scalar_out_typed_handle();
    return op.call(self, other, out);
}

// aten::greater_equal.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & greater_equal_Scalar_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_greater_equal_Scalar_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(greater_equal__Scalar, name, "aten::greater_equal_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(greater_equal__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(greater_equal__Scalar, schema_str, "greater_equal_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)")

// aten::greater_equal_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<greater_equal__Scalar::schema> create_greater_equal__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(greater_equal__Scalar::name, greater_equal__Scalar::overload_name)
      .typed<greater_equal__Scalar::schema>();
}

// aten::greater_equal_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & greater_equal__Scalar::call(at::Tensor & self, const at::Scalar & other) {
    static auto op = create_greater_equal__Scalar_typed_handle();
    return op.call(self, other);
}

// aten::greater_equal_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & greater_equal__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
    static auto op = create_greater_equal__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(le_Tensor, name, "aten::le")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(le_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(le_Tensor, schema_str, "le.Tensor(Tensor self, Tensor other) -> Tensor")

// aten::le.Tensor(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<le_Tensor::schema> create_le_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(le_Tensor::name, le_Tensor::overload_name)
      .typed<le_Tensor::schema>();
}

// aten::le.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor le_Tensor::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_le_Tensor_typed_handle();
    return op.call(self, other);
}

// aten::le.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor le_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_le_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(le__Scalar, name, "aten::le_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(le__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(le__Scalar, schema_str, "le_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)")

// aten::le_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<le__Scalar::schema> create_le__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(le__Scalar::name, le__Scalar::overload_name)
      .typed<le__Scalar::schema>();
}

// aten::le_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & le__Scalar::call(at::Tensor & self, const at::Scalar & other) {
    static auto op = create_le__Scalar_typed_handle();
    return op.call(self, other);
}

// aten::le_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & le__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
    static auto op = create_le__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(less_equal_Scalar, name, "aten::less_equal")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(less_equal_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(less_equal_Scalar, schema_str, "less_equal.Scalar(Tensor self, Scalar other) -> Tensor")

// aten::less_equal.Scalar(Tensor self, Scalar other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<less_equal_Scalar::schema> create_less_equal_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(less_equal_Scalar::name, less_equal_Scalar::overload_name)
      .typed<less_equal_Scalar::schema>();
}

// aten::less_equal.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor less_equal_Scalar::call(const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_less_equal_Scalar_typed_handle();
    return op.call(self, other);
}

// aten::less_equal.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor less_equal_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_less_equal_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(less_equal_Tensor, name, "aten::less_equal")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(less_equal_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(less_equal_Tensor, schema_str, "less_equal.Tensor(Tensor self, Tensor other) -> Tensor")

// aten::less_equal.Tensor(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<less_equal_Tensor::schema> create_less_equal_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(less_equal_Tensor::name, less_equal_Tensor::overload_name)
      .typed<less_equal_Tensor::schema>();
}

// aten::less_equal.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor less_equal_Tensor::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_less_equal_Tensor_typed_handle();
    return op.call(self, other);
}

// aten::less_equal.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor less_equal_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_less_equal_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gt__Scalar, name, "aten::gt_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gt__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gt__Scalar, schema_str, "gt_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)")

// aten::gt_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<gt__Scalar::schema> create_gt__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(gt__Scalar::name, gt__Scalar::overload_name)
      .typed<gt__Scalar::schema>();
}

// aten::gt_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & gt__Scalar::call(at::Tensor & self, const at::Scalar & other) {
    static auto op = create_gt__Scalar_typed_handle();
    return op.call(self, other);
}

// aten::gt_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & gt__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
    static auto op = create_gt__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(greater__Scalar, name, "aten::greater_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(greater__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(greater__Scalar, schema_str, "greater_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)")

// aten::greater_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<greater__Scalar::schema> create_greater__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(greater__Scalar::name, greater__Scalar::overload_name)
      .typed<greater__Scalar::schema>();
}

// aten::greater_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & greater__Scalar::call(at::Tensor & self, const at::Scalar & other) {
    static auto op = create_greater__Scalar_typed_handle();
    return op.call(self, other);
}

// aten::greater_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & greater__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
    static auto op = create_greater__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lt__Tensor, name, "aten::lt_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lt__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lt__Tensor, schema_str, "lt_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::lt_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<lt__Tensor::schema> create_lt__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(lt__Tensor::name, lt__Tensor::overload_name)
      .typed<lt__Tensor::schema>();
}

// aten::lt_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & lt__Tensor::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create_lt__Tensor_typed_handle();
    return op.call(self, other);
}

// aten::lt_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & lt__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create_lt__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(masked_select, name, "aten::masked_select")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(masked_select, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(masked_select, schema_str, "masked_select(Tensor self, Tensor mask) -> Tensor")

// aten::masked_select(Tensor self, Tensor mask) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<masked_select::schema> create_masked_select_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(masked_select::name, masked_select::overload_name)
      .typed<masked_select::schema>();
}

// aten::masked_select(Tensor self, Tensor mask) -> Tensor
at::Tensor masked_select::call(const at::Tensor & self, const at::Tensor & mask) {
    static auto op = create_masked_select_typed_handle();
    return op.call(self, mask);
}

// aten::masked_select(Tensor self, Tensor mask) -> Tensor
at::Tensor masked_select::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mask) {
    static auto op = create_masked_select_typed_handle();
    return op.redispatch(dispatchKeySet, self, mask);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addcdiv, name, "aten::addcdiv")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addcdiv, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addcdiv, schema_str, "addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor")

// aten::addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<addcdiv::schema> create_addcdiv_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(addcdiv::name, addcdiv::overload_name)
      .typed<addcdiv::schema>();
}

// aten::addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor
at::Tensor addcdiv::call(const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value) {
    static auto op = create_addcdiv_typed_handle();
    return op.call(self, tensor1, tensor2, value);
}

// aten::addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor
at::Tensor addcdiv::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value) {
    static auto op = create_addcdiv_typed_handle();
    return op.redispatch(dispatchKeySet, self, tensor1, tensor2, value);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addcdiv_, name, "aten::addcdiv_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addcdiv_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addcdiv_, schema_str, "addcdiv_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)")

// aten::addcdiv_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<addcdiv_::schema> create_addcdiv__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(addcdiv_::name, addcdiv_::overload_name)
      .typed<addcdiv_::schema>();
}

// aten::addcdiv_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)
at::Tensor & addcdiv_::call(at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value) {
    static auto op = create_addcdiv__typed_handle();
    return op.call(self, tensor1, tensor2, value);
}

// aten::addcdiv_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)
at::Tensor & addcdiv_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value) {
    static auto op = create_addcdiv__typed_handle();
    return op.redispatch(dispatchKeySet, self, tensor1, tensor2, value);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(triangular_solve_X, name, "aten::triangular_solve")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(triangular_solve_X, overload_name, "X")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(triangular_solve_X, schema_str, "triangular_solve.X(Tensor self, Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False, *, Tensor(a!) X, Tensor(b!) M) -> (Tensor(a!) solution, Tensor(b!) cloned_coefficient)")

// aten::triangular_solve.X(Tensor self, Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False, *, Tensor(a!) X, Tensor(b!) M) -> (Tensor(a!) solution, Tensor(b!) cloned_coefficient)
static C10_NOINLINE c10::TypedOperatorHandle<triangular_solve_X::schema> create_triangular_solve_X_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(triangular_solve_X::name, triangular_solve_X::overload_name)
      .typed<triangular_solve_X::schema>();
}

// aten::triangular_solve.X(Tensor self, Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False, *, Tensor(a!) X, Tensor(b!) M) -> (Tensor(a!) solution, Tensor(b!) cloned_coefficient)
::std::tuple<at::Tensor &,at::Tensor &> triangular_solve_X::call(const at::Tensor & self, const at::Tensor & A, bool upper, bool transpose, bool unitriangular, at::Tensor & X, at::Tensor & M) {
    static auto op = create_triangular_solve_X_typed_handle();
    return op.call(self, A, upper, transpose, unitriangular, X, M);
}

// aten::triangular_solve.X(Tensor self, Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False, *, Tensor(a!) X, Tensor(b!) M) -> (Tensor(a!) solution, Tensor(b!) cloned_coefficient)
::std::tuple<at::Tensor &,at::Tensor &> triangular_solve_X::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & A, bool upper, bool transpose, bool unitriangular, at::Tensor & X, at::Tensor & M) {
    static auto op = create_triangular_solve_X_typed_handle();
    return op.redispatch(dispatchKeySet, self, A, upper, transpose, unitriangular, X, M);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(symeig_e, name, "aten::symeig")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(symeig_e, overload_name, "e")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(symeig_e, schema_str, "symeig.e(Tensor self, bool eigenvectors=False, bool upper=True, *, Tensor(a!) e, Tensor(b!) V) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)")

// aten::symeig.e(Tensor self, bool eigenvectors=False, bool upper=True, *, Tensor(a!) e, Tensor(b!) V) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)
static C10_NOINLINE c10::TypedOperatorHandle<symeig_e::schema> create_symeig_e_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(symeig_e::name, symeig_e::overload_name)
      .typed<symeig_e::schema>();
}

// aten::symeig.e(Tensor self, bool eigenvectors=False, bool upper=True, *, Tensor(a!) e, Tensor(b!) V) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)
::std::tuple<at::Tensor &,at::Tensor &> symeig_e::call(const at::Tensor & self, bool eigenvectors, bool upper, at::Tensor & e, at::Tensor & V) {
    static auto op = create_symeig_e_typed_handle();
    return op.call(self, eigenvectors, upper, e, V);
}

// aten::symeig.e(Tensor self, bool eigenvectors=False, bool upper=True, *, Tensor(a!) e, Tensor(b!) V) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)
::std::tuple<at::Tensor &,at::Tensor &> symeig_e::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool eigenvectors, bool upper, at::Tensor & e, at::Tensor & V) {
    static auto op = create_symeig_e_typed_handle();
    return op.redispatch(dispatchKeySet, self, eigenvectors, upper, e, V);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(symeig, name, "aten::symeig")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(symeig, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(symeig, schema_str, "symeig(Tensor self, bool eigenvectors=False, bool upper=True) -> (Tensor eigenvalues, Tensor eigenvectors)")

// aten::symeig(Tensor self, bool eigenvectors=False, bool upper=True) -> (Tensor eigenvalues, Tensor eigenvectors)
static C10_NOINLINE c10::TypedOperatorHandle<symeig::schema> create_symeig_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(symeig::name, symeig::overload_name)
      .typed<symeig::schema>();
}

// aten::symeig(Tensor self, bool eigenvectors=False, bool upper=True) -> (Tensor eigenvalues, Tensor eigenvectors)
::std::tuple<at::Tensor,at::Tensor> symeig::call(const at::Tensor & self, bool eigenvectors, bool upper) {
    static auto op = create_symeig_typed_handle();
    return op.call(self, eigenvectors, upper);
}

// aten::symeig(Tensor self, bool eigenvectors=False, bool upper=True) -> (Tensor eigenvalues, Tensor eigenvectors)
::std::tuple<at::Tensor,at::Tensor> symeig::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool eigenvectors, bool upper) {
    static auto op = create_symeig_typed_handle();
    return op.redispatch(dispatchKeySet, self, eigenvectors, upper);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_symeig_helper, name, "aten::_symeig_helper")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_symeig_helper, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_symeig_helper, schema_str, "_symeig_helper(Tensor self, bool eigenvectors, bool upper) -> (Tensor, Tensor)")

// aten::_symeig_helper(Tensor self, bool eigenvectors, bool upper) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<_symeig_helper::schema> create__symeig_helper_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_symeig_helper::name, _symeig_helper::overload_name)
      .typed<_symeig_helper::schema>();
}

// aten::_symeig_helper(Tensor self, bool eigenvectors, bool upper) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> _symeig_helper::call(const at::Tensor & self, bool eigenvectors, bool upper) {
    static auto op = create__symeig_helper_typed_handle();
    return op.call(self, eigenvectors, upper);
}

// aten::_symeig_helper(Tensor self, bool eigenvectors, bool upper) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> _symeig_helper::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool eigenvectors, bool upper) {
    static auto op = create__symeig_helper_typed_handle();
    return op.redispatch(dispatchKeySet, self, eigenvectors, upper);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cholesky_out, name, "aten::cholesky")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cholesky_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cholesky_out, schema_str, "cholesky.out(Tensor self, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)")

// aten::cholesky.out(Tensor self, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<cholesky_out::schema> create_cholesky_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cholesky_out::name, cholesky_out::overload_name)
      .typed<cholesky_out::schema>();
}

// aten::cholesky.out(Tensor self, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & cholesky_out::call(const at::Tensor & self, bool upper, at::Tensor & out) {
    static auto op = create_cholesky_out_typed_handle();
    return op.call(self, upper, out);
}

// aten::cholesky.out(Tensor self, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & cholesky_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool upper, at::Tensor & out) {
    static auto op = create_cholesky_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, upper, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cholesky_solve_helper, name, "aten::_cholesky_solve_helper")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cholesky_solve_helper, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cholesky_solve_helper, schema_str, "_cholesky_solve_helper(Tensor self, Tensor A, bool upper) -> Tensor")

// aten::_cholesky_solve_helper(Tensor self, Tensor A, bool upper) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_cholesky_solve_helper::schema> create__cholesky_solve_helper_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_cholesky_solve_helper::name, _cholesky_solve_helper::overload_name)
      .typed<_cholesky_solve_helper::schema>();
}

// aten::_cholesky_solve_helper(Tensor self, Tensor A, bool upper) -> Tensor
at::Tensor _cholesky_solve_helper::call(const at::Tensor & self, const at::Tensor & A, bool upper) {
    static auto op = create__cholesky_solve_helper_typed_handle();
    return op.call(self, A, upper);
}

// aten::_cholesky_solve_helper(Tensor self, Tensor A, bool upper) -> Tensor
at::Tensor _cholesky_solve_helper::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & A, bool upper) {
    static auto op = create__cholesky_solve_helper_typed_handle();
    return op.redispatch(dispatchKeySet, self, A, upper);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cholesky_inverse, name, "aten::cholesky_inverse")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cholesky_inverse, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cholesky_inverse, schema_str, "cholesky_inverse(Tensor self, bool upper=False) -> Tensor")

// aten::cholesky_inverse(Tensor self, bool upper=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<cholesky_inverse::schema> create_cholesky_inverse_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cholesky_inverse::name, cholesky_inverse::overload_name)
      .typed<cholesky_inverse::schema>();
}

// aten::cholesky_inverse(Tensor self, bool upper=False) -> Tensor
at::Tensor cholesky_inverse::call(const at::Tensor & self, bool upper) {
    static auto op = create_cholesky_inverse_typed_handle();
    return op.call(self, upper);
}

// aten::cholesky_inverse(Tensor self, bool upper=False) -> Tensor
at::Tensor cholesky_inverse::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool upper) {
    static auto op = create_cholesky_inverse_typed_handle();
    return op.redispatch(dispatchKeySet, self, upper);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(geqrf_a, name, "aten::geqrf")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(geqrf_a, overload_name, "a")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(geqrf_a, schema_str, "geqrf.a(Tensor self, *, Tensor(a!) a, Tensor(b!) tau) -> (Tensor(a!) a, Tensor(b!) tau)")

// aten::geqrf.a(Tensor self, *, Tensor(a!) a, Tensor(b!) tau) -> (Tensor(a!) a, Tensor(b!) tau)
static C10_NOINLINE c10::TypedOperatorHandle<geqrf_a::schema> create_geqrf_a_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(geqrf_a::name, geqrf_a::overload_name)
      .typed<geqrf_a::schema>();
}

// aten::geqrf.a(Tensor self, *, Tensor(a!) a, Tensor(b!) tau) -> (Tensor(a!) a, Tensor(b!) tau)
::std::tuple<at::Tensor &,at::Tensor &> geqrf_a::call(const at::Tensor & self, at::Tensor & a, at::Tensor & tau) {
    static auto op = create_geqrf_a_typed_handle();
    return op.call(self, a, tau);
}

// aten::geqrf.a(Tensor self, *, Tensor(a!) a, Tensor(b!) tau) -> (Tensor(a!) a, Tensor(b!) tau)
::std::tuple<at::Tensor &,at::Tensor &> geqrf_a::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & a, at::Tensor & tau) {
    static auto op = create_geqrf_a_typed_handle();
    return op.redispatch(dispatchKeySet, self, a, tau);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_lu_with_info, name, "aten::_lu_with_info")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_lu_with_info, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_lu_with_info, schema_str, "_lu_with_info(Tensor self, bool pivot=True, bool check_errors=True) -> (Tensor LU, Tensor pivots, Tensor info)")

// aten::_lu_with_info(Tensor self, bool pivot=True, bool check_errors=True) -> (Tensor LU, Tensor pivots, Tensor info)
static C10_NOINLINE c10::TypedOperatorHandle<_lu_with_info::schema> create__lu_with_info_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_lu_with_info::name, _lu_with_info::overload_name)
      .typed<_lu_with_info::schema>();
}

// aten::_lu_with_info(Tensor self, bool pivot=True, bool check_errors=True) -> (Tensor LU, Tensor pivots, Tensor info)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _lu_with_info::call(const at::Tensor & self, bool pivot, bool check_errors) {
    static auto op = create__lu_with_info_typed_handle();
    return op.call(self, pivot, check_errors);
}

// aten::_lu_with_info(Tensor self, bool pivot=True, bool check_errors=True) -> (Tensor LU, Tensor pivots, Tensor info)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _lu_with_info::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool pivot, bool check_errors) {
    static auto op = create__lu_with_info_typed_handle();
    return op.redispatch(dispatchKeySet, self, pivot, check_errors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(polygamma_out, name, "aten::polygamma")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(polygamma_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(polygamma_out, schema_str, "polygamma.out(int n, Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::polygamma.out(int n, Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<polygamma_out::schema> create_polygamma_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(polygamma_out::name, polygamma_out::overload_name)
      .typed<polygamma_out::schema>();
}

// aten::polygamma.out(int n, Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & polygamma_out::call(int64_t n, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_polygamma_out_typed_handle();
    return op.call(n, self, out);
}

// aten::polygamma.out(int n, Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & polygamma_out::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t n, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_polygamma_out_typed_handle();
    return op.redispatch(dispatchKeySet, n, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atan2_out, name, "aten::atan2")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atan2_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atan2_out, schema_str, "atan2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::atan2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<atan2_out::schema> create_atan2_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(atan2_out::name, atan2_out::overload_name)
      .typed<atan2_out::schema>();
}

// aten::atan2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & atan2_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_atan2_out_typed_handle();
    return op.call(self, other, out);
}

// aten::atan2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & atan2_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_atan2_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atan2_, name, "aten::atan2_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atan2_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atan2_, schema_str, "atan2_(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::atan2_(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<atan2_::schema> create_atan2__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(atan2_::name, atan2_::overload_name)
      .typed<atan2_::schema>();
}

// aten::atan2_(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & atan2_::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create_atan2__typed_handle();
    return op.call(self, other);
}

// aten::atan2_(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & atan2_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create_atan2__typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atan2, name, "aten::atan2")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atan2, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atan2, schema_str, "atan2(Tensor self, Tensor other) -> Tensor")

// aten::atan2(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<atan2::schema> create_atan2_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(atan2::name, atan2::overload_name)
      .typed<atan2::schema>();
}

// aten::atan2(Tensor self, Tensor other) -> Tensor
at::Tensor atan2::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_atan2_typed_handle();
    return op.call(self, other);
}

// aten::atan2(Tensor self, Tensor other) -> Tensor
at::Tensor atan2::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_atan2_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lerp_Scalar, name, "aten::lerp")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lerp_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lerp_Scalar, schema_str, "lerp.Scalar(Tensor self, Tensor end, Scalar weight) -> Tensor")

// aten::lerp.Scalar(Tensor self, Tensor end, Scalar weight) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<lerp_Scalar::schema> create_lerp_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(lerp_Scalar::name, lerp_Scalar::overload_name)
      .typed<lerp_Scalar::schema>();
}

// aten::lerp.Scalar(Tensor self, Tensor end, Scalar weight) -> Tensor
at::Tensor lerp_Scalar::call(const at::Tensor & self, const at::Tensor & end, const at::Scalar & weight) {
    static auto op = create_lerp_Scalar_typed_handle();
    return op.call(self, end, weight);
}

// aten::lerp.Scalar(Tensor self, Tensor end, Scalar weight) -> Tensor
at::Tensor lerp_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & end, const at::Scalar & weight) {
    static auto op = create_lerp_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, end, weight);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(histc_out, name, "aten::histc")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(histc_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(histc_out, schema_str, "histc.out(Tensor self, int bins=100, Scalar min=0, Scalar max=0, *, Tensor(a!) out) -> Tensor(a!)")

// aten::histc.out(Tensor self, int bins=100, Scalar min=0, Scalar max=0, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<histc_out::schema> create_histc_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(histc_out::name, histc_out::overload_name)
      .typed<histc_out::schema>();
}

// aten::histc.out(Tensor self, int bins=100, Scalar min=0, Scalar max=0, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & histc_out::call(const at::Tensor & self, int64_t bins, const at::Scalar & min, const at::Scalar & max, at::Tensor & out) {
    static auto op = create_histc_out_typed_handle();
    return op.call(self, bins, min, max, out);
}

// aten::histc.out(Tensor self, int bins=100, Scalar min=0, Scalar max=0, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & histc_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t bins, const at::Scalar & min, const at::Scalar & max, at::Tensor & out) {
    static auto op = create_histc_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, bins, min, max, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fmod_Scalar_out, name, "aten::fmod")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fmod_Scalar_out, overload_name, "Scalar_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fmod_Scalar_out, schema_str, "fmod.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::fmod.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<fmod_Scalar_out::schema> create_fmod_Scalar_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fmod_Scalar_out::name, fmod_Scalar_out::overload_name)
      .typed<fmod_Scalar_out::schema>();
}

// aten::fmod.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fmod_Scalar_out::call(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_fmod_Scalar_out_typed_handle();
    return op.call(self, other, out);
}

// aten::fmod.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fmod_Scalar_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_fmod_Scalar_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hypot, name, "aten::hypot")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hypot, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hypot, schema_str, "hypot(Tensor self, Tensor other) -> Tensor")

// aten::hypot(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<hypot::schema> create_hypot_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(hypot::name, hypot::overload_name)
      .typed<hypot::schema>();
}

// aten::hypot(Tensor self, Tensor other) -> Tensor
at::Tensor hypot::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_hypot_typed_handle();
    return op.call(self, other);
}

// aten::hypot(Tensor self, Tensor other) -> Tensor
at::Tensor hypot::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_hypot_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(igammac, name, "aten::igammac")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(igammac, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(igammac, schema_str, "igammac(Tensor self, Tensor other) -> Tensor")

// aten::igammac(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<igammac::schema> create_igammac_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(igammac::name, igammac::overload_name)
      .typed<igammac::schema>();
}

// aten::igammac(Tensor self, Tensor other) -> Tensor
at::Tensor igammac::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_igammac_typed_handle();
    return op.call(self, other);
}

// aten::igammac(Tensor self, Tensor other) -> Tensor
at::Tensor igammac::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_igammac_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nextafter_out, name, "aten::nextafter")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nextafter_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nextafter_out, schema_str, "nextafter.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::nextafter.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<nextafter_out::schema> create_nextafter_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nextafter_out::name, nextafter_out::overload_name)
      .typed<nextafter_out::schema>();
}

// aten::nextafter.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & nextafter_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_nextafter_out_typed_handle();
    return op.call(self, other, out);
}

// aten::nextafter.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & nextafter_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_nextafter_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(remainder_Scalar_out, name, "aten::remainder")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(remainder_Scalar_out, overload_name, "Scalar_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(remainder_Scalar_out, schema_str, "remainder.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::remainder.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<remainder_Scalar_out::schema> create_remainder_Scalar_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(remainder_Scalar_out::name, remainder_Scalar_out::overload_name)
      .typed<remainder_Scalar_out::schema>();
}

// aten::remainder.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & remainder_Scalar_out::call(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_remainder_Scalar_out_typed_handle();
    return op.call(self, other, out);
}

// aten::remainder.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & remainder_Scalar_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_remainder_Scalar_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(remainder_Scalar_Tensor, name, "aten::remainder")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(remainder_Scalar_Tensor, overload_name, "Scalar_Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(remainder_Scalar_Tensor, schema_str, "remainder.Scalar_Tensor(Scalar self, Tensor other) -> Tensor")

// aten::remainder.Scalar_Tensor(Scalar self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<remainder_Scalar_Tensor::schema> create_remainder_Scalar_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(remainder_Scalar_Tensor::name, remainder_Scalar_Tensor::overload_name)
      .typed<remainder_Scalar_Tensor::schema>();
}

// aten::remainder.Scalar_Tensor(Scalar self, Tensor other) -> Tensor
at::Tensor remainder_Scalar_Tensor::call(const at::Scalar & self, const at::Tensor & other) {
    static auto op = create_remainder_Scalar_Tensor_typed_handle();
    return op.call(self, other);
}

// aten::remainder.Scalar_Tensor(Scalar self, Tensor other) -> Tensor
at::Tensor remainder_Scalar_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Scalar & self, const at::Tensor & other) {
    static auto op = create_remainder_Scalar_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fmax, name, "aten::fmax")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fmax, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fmax, schema_str, "fmax(Tensor self, Tensor other) -> Tensor")

// aten::fmax(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fmax::schema> create_fmax_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fmax::name, fmax::overload_name)
      .typed<fmax::schema>();
}

// aten::fmax(Tensor self, Tensor other) -> Tensor
at::Tensor fmax::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_fmax_typed_handle();
    return op.call(self, other);
}

// aten::fmax(Tensor self, Tensor other) -> Tensor
at::Tensor fmax::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_fmax_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(min_other, name, "aten::min")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(min_other, overload_name, "other")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(min_other, schema_str, "min.other(Tensor self, Tensor other) -> Tensor")

// aten::min.other(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<min_other::schema> create_min_other_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(min_other::name, min_other::overload_name)
      .typed<min_other::schema>();
}

// aten::min.other(Tensor self, Tensor other) -> Tensor
at::Tensor min_other::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_min_other_typed_handle();
    return op.call(self, other);
}

// aten::min.other(Tensor self, Tensor other) -> Tensor
at::Tensor min_other::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_min_other_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanquantile_out, name, "aten::nanquantile")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanquantile_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanquantile_out, schema_str, "nanquantile.out(Tensor self, Tensor q, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)")

// aten::nanquantile.out(Tensor self, Tensor q, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<nanquantile_out::schema> create_nanquantile_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nanquantile_out::name, nanquantile_out::overload_name)
      .typed<nanquantile_out::schema>();
}

// aten::nanquantile.out(Tensor self, Tensor q, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & nanquantile_out::call(const at::Tensor & self, const at::Tensor & q, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out) {
    static auto op = create_nanquantile_out_typed_handle();
    return op.call(self, q, dim, keepdim, out);
}

// aten::nanquantile.out(Tensor self, Tensor q, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & nanquantile_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & q, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out) {
    static auto op = create_nanquantile_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, q, dim, keepdim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantile_new_scalar_out, name, "aten::quantile")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantile_new_scalar_out, overload_name, "new_scalar_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantile_new_scalar_out, schema_str, "quantile.new_scalar_out(Tensor self, float q, int? dim, bool keepdim, *, str interpolation, Tensor(a!) out) -> Tensor(a!)")

// aten::quantile.new_scalar_out(Tensor self, float q, int? dim, bool keepdim, *, str interpolation, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<quantile_new_scalar_out::schema> create_quantile_new_scalar_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(quantile_new_scalar_out::name, quantile_new_scalar_out::overload_name)
      .typed<quantile_new_scalar_out::schema>();
}

// aten::quantile.new_scalar_out(Tensor self, float q, int? dim, bool keepdim, *, str interpolation, Tensor(a!) out) -> Tensor(a!)
at::Tensor & quantile_new_scalar_out::call(const at::Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim, c10::string_view interpolation, at::Tensor & out) {
    static auto op = create_quantile_new_scalar_out_typed_handle();
    return op.call(self, q, dim, keepdim, interpolation, out);
}

// aten::quantile.new_scalar_out(Tensor self, float q, int? dim, bool keepdim, *, str interpolation, Tensor(a!) out) -> Tensor(a!)
at::Tensor & quantile_new_scalar_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim, c10::string_view interpolation, at::Tensor & out) {
    static auto op = create_quantile_new_scalar_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, q, dim, keepdim, interpolation, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantile_new_scalar, name, "aten::quantile")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantile_new_scalar, overload_name, "new_scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantile_new_scalar, schema_str, "quantile.new_scalar(Tensor self, float q, int? dim, bool keepdim, *, str interpolation) -> Tensor")

// aten::quantile.new_scalar(Tensor self, float q, int? dim, bool keepdim, *, str interpolation) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<quantile_new_scalar::schema> create_quantile_new_scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(quantile_new_scalar::name, quantile_new_scalar::overload_name)
      .typed<quantile_new_scalar::schema>();
}

// aten::quantile.new_scalar(Tensor self, float q, int? dim, bool keepdim, *, str interpolation) -> Tensor
at::Tensor quantile_new_scalar::call(const at::Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) {
    static auto op = create_quantile_new_scalar_typed_handle();
    return op.call(self, q, dim, keepdim, interpolation);
}

// aten::quantile.new_scalar(Tensor self, float q, int? dim, bool keepdim, *, str interpolation) -> Tensor
at::Tensor quantile_new_scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) {
    static auto op = create_quantile_new_scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, q, dim, keepdim, interpolation);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanquantile_new_scalar_out, name, "aten::nanquantile")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanquantile_new_scalar_out, overload_name, "new_scalar_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanquantile_new_scalar_out, schema_str, "nanquantile.new_scalar_out(Tensor self, float q, int? dim, bool keepdim, *, str interpolation, Tensor(a!) out) -> Tensor(a!)")

// aten::nanquantile.new_scalar_out(Tensor self, float q, int? dim, bool keepdim, *, str interpolation, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<nanquantile_new_scalar_out::schema> create_nanquantile_new_scalar_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nanquantile_new_scalar_out::name, nanquantile_new_scalar_out::overload_name)
      .typed<nanquantile_new_scalar_out::schema>();
}

// aten::nanquantile.new_scalar_out(Tensor self, float q, int? dim, bool keepdim, *, str interpolation, Tensor(a!) out) -> Tensor(a!)
at::Tensor & nanquantile_new_scalar_out::call(const at::Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim, c10::string_view interpolation, at::Tensor & out) {
    static auto op = create_nanquantile_new_scalar_out_typed_handle();
    return op.call(self, q, dim, keepdim, interpolation, out);
}

// aten::nanquantile.new_scalar_out(Tensor self, float q, int? dim, bool keepdim, *, str interpolation, Tensor(a!) out) -> Tensor(a!)
at::Tensor & nanquantile_new_scalar_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim, c10::string_view interpolation, at::Tensor & out) {
    static auto op = create_nanquantile_new_scalar_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, q, dim, keepdim, interpolation, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanquantile_new, name, "aten::nanquantile")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanquantile_new, overload_name, "new")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanquantile_new, schema_str, "nanquantile.new(Tensor self, Tensor q, int? dim, bool keepdim, *, str interpolation) -> Tensor")

// aten::nanquantile.new(Tensor self, Tensor q, int? dim, bool keepdim, *, str interpolation) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<nanquantile_new::schema> create_nanquantile_new_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nanquantile_new::name, nanquantile_new::overload_name)
      .typed<nanquantile_new::schema>();
}

// aten::nanquantile.new(Tensor self, Tensor q, int? dim, bool keepdim, *, str interpolation) -> Tensor
at::Tensor nanquantile_new::call(const at::Tensor & self, const at::Tensor & q, c10::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) {
    static auto op = create_nanquantile_new_typed_handle();
    return op.call(self, q, dim, keepdim, interpolation);
}

// aten::nanquantile.new(Tensor self, Tensor q, int? dim, bool keepdim, *, str interpolation) -> Tensor
at::Tensor nanquantile_new::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & q, c10::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) {
    static auto op = create_nanquantile_new_typed_handle();
    return op.redispatch(dispatchKeySet, self, q, dim, keepdim, interpolation);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sort, name, "aten::sort")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sort, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sort, schema_str, "sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)")

// aten::sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)
static C10_NOINLINE c10::TypedOperatorHandle<sort::schema> create_sort_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sort::name, sort::overload_name)
      .typed<sort::schema>();
}

// aten::sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> sort::call(const at::Tensor & self, int64_t dim, bool descending) {
    static auto op = create_sort_typed_handle();
    return op.call(self, dim, descending);
}

// aten::sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> sort::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool descending) {
    static auto op = create_sort_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, descending);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sort_dimname_values_stable, name, "aten::sort")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sort_dimname_values_stable, overload_name, "dimname_values_stable")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sort_dimname_values_stable, schema_str, "sort.dimname_values_stable(Tensor self, *, bool? stable, Dimname dim, bool descending=False, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)")

// aten::sort.dimname_values_stable(Tensor self, *, bool? stable, Dimname dim, bool descending=False, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
static C10_NOINLINE c10::TypedOperatorHandle<sort_dimname_values_stable::schema> create_sort_dimname_values_stable_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sort_dimname_values_stable::name, sort_dimname_values_stable::overload_name)
      .typed<sort_dimname_values_stable::schema>();
}

// aten::sort.dimname_values_stable(Tensor self, *, bool? stable, Dimname dim, bool descending=False, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> sort_dimname_values_stable::call(const at::Tensor & self, c10::optional<bool> stable, at::Dimname dim, bool descending, at::Tensor & values, at::Tensor & indices) {
    static auto op = create_sort_dimname_values_stable_typed_handle();
    return op.call(self, stable, dim, descending, values, indices);
}

// aten::sort.dimname_values_stable(Tensor self, *, bool? stable, Dimname dim, bool descending=False, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> sort_dimname_values_stable::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<bool> stable, at::Dimname dim, bool descending, at::Tensor & values, at::Tensor & indices) {
    static auto op = create_sort_dimname_values_stable_typed_handle();
    return op.redispatch(dispatchKeySet, self, stable, dim, descending, values, indices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(msort_out, name, "aten::msort")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(msort_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(msort_out, schema_str, "msort.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::msort.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<msort_out::schema> create_msort_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(msort_out::name, msort_out::overload_name)
      .typed<msort_out::schema>();
}

// aten::msort.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & msort_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_msort_out_typed_handle();
    return op.call(self, out);
}

// aten::msort.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & msort_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_msort_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(argsort_dimname, name, "aten::argsort")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(argsort_dimname, overload_name, "dimname")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(argsort_dimname, schema_str, "argsort.dimname(Tensor self, Dimname dim, bool descending=False) -> Tensor")

// aten::argsort.dimname(Tensor self, Dimname dim, bool descending=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<argsort_dimname::schema> create_argsort_dimname_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(argsort_dimname::name, argsort_dimname::overload_name)
      .typed<argsort_dimname::schema>();
}

// aten::argsort.dimname(Tensor self, Dimname dim, bool descending=False) -> Tensor
at::Tensor argsort_dimname::call(const at::Tensor & self, at::Dimname dim, bool descending) {
    static auto op = create_argsort_dimname_typed_handle();
    return op.call(self, dim, descending);
}

// aten::argsort.dimname(Tensor self, Dimname dim, bool descending=False) -> Tensor
at::Tensor argsort_dimname::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, bool descending) {
    static auto op = create_argsort_dimname_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, descending);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(all, name, "aten::all")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(all, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(all, schema_str, "all(Tensor self) -> Tensor")

// aten::all(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<all::schema> create_all_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(all::name, all::overload_name)
      .typed<all::schema>();
}

// aten::all(Tensor self) -> Tensor
at::Tensor all::call(const at::Tensor & self) {
    static auto op = create_all_typed_handle();
    return op.call(self);
}

// aten::all(Tensor self) -> Tensor
at::Tensor all::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_all_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(renorm_out, name, "aten::renorm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(renorm_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(renorm_out, schema_str, "renorm.out(Tensor self, Scalar p, int dim, Scalar maxnorm, *, Tensor(a!) out) -> Tensor(a!)")

// aten::renorm.out(Tensor self, Scalar p, int dim, Scalar maxnorm, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<renorm_out::schema> create_renorm_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(renorm_out::name, renorm_out::overload_name)
      .typed<renorm_out::schema>();
}

// aten::renorm.out(Tensor self, Scalar p, int dim, Scalar maxnorm, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & renorm_out::call(const at::Tensor & self, const at::Scalar & p, int64_t dim, const at::Scalar & maxnorm, at::Tensor & out) {
    static auto op = create_renorm_out_typed_handle();
    return op.call(self, p, dim, maxnorm, out);
}

// aten::renorm.out(Tensor self, Scalar p, int dim, Scalar maxnorm, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & renorm_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & p, int64_t dim, const at::Scalar & maxnorm, at::Tensor & out) {
    static auto op = create_renorm_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, p, dim, maxnorm, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pow_Tensor_Scalar_out, name, "aten::pow")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pow_Tensor_Scalar_out, overload_name, "Tensor_Scalar_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pow_Tensor_Scalar_out, schema_str, "pow.Tensor_Scalar_out(Tensor self, Scalar exponent, *, Tensor(a!) out) -> Tensor(a!)")

// aten::pow.Tensor_Scalar_out(Tensor self, Scalar exponent, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<pow_Tensor_Scalar_out::schema> create_pow_Tensor_Scalar_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(pow_Tensor_Scalar_out::name, pow_Tensor_Scalar_out::overload_name)
      .typed<pow_Tensor_Scalar_out::schema>();
}

// aten::pow.Tensor_Scalar_out(Tensor self, Scalar exponent, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & pow_Tensor_Scalar_out::call(const at::Tensor & self, const at::Scalar & exponent, at::Tensor & out) {
    static auto op = create_pow_Tensor_Scalar_out_typed_handle();
    return op.call(self, exponent, out);
}

// aten::pow.Tensor_Scalar_out(Tensor self, Scalar exponent, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & pow_Tensor_Scalar_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & exponent, at::Tensor & out) {
    static auto op = create_pow_Tensor_Scalar_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, exponent, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(float_power_Tensor_Tensor_out, name, "aten::float_power")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(float_power_Tensor_Tensor_out, overload_name, "Tensor_Tensor_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(float_power_Tensor_Tensor_out, schema_str, "float_power.Tensor_Tensor_out(Tensor self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)")

// aten::float_power.Tensor_Tensor_out(Tensor self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<float_power_Tensor_Tensor_out::schema> create_float_power_Tensor_Tensor_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(float_power_Tensor_Tensor_out::name, float_power_Tensor_Tensor_out::overload_name)
      .typed<float_power_Tensor_Tensor_out::schema>();
}

// aten::float_power.Tensor_Tensor_out(Tensor self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & float_power_Tensor_Tensor_out::call(const at::Tensor & self, const at::Tensor & exponent, at::Tensor & out) {
    static auto op = create_float_power_Tensor_Tensor_out_typed_handle();
    return op.call(self, exponent, out);
}

// aten::float_power.Tensor_Tensor_out(Tensor self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & float_power_Tensor_Tensor_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & exponent, at::Tensor & out) {
    static auto op = create_float_power_Tensor_Tensor_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, exponent, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(float_power_Scalar_out, name, "aten::float_power")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(float_power_Scalar_out, overload_name, "Scalar_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(float_power_Scalar_out, schema_str, "float_power.Scalar_out(Scalar self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)")

// aten::float_power.Scalar_out(Scalar self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<float_power_Scalar_out::schema> create_float_power_Scalar_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(float_power_Scalar_out::name, float_power_Scalar_out::overload_name)
      .typed<float_power_Scalar_out::schema>();
}

// aten::float_power.Scalar_out(Scalar self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & float_power_Scalar_out::call(const at::Scalar & self, const at::Tensor & exponent, at::Tensor & out) {
    static auto op = create_float_power_Scalar_out_typed_handle();
    return op.call(self, exponent, out);
}

// aten::float_power.Scalar_out(Scalar self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & float_power_Scalar_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Scalar & self, const at::Tensor & exponent, at::Tensor & out) {
    static auto op = create_float_power_Scalar_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, exponent, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(float_power_Tensor_Scalar_out, name, "aten::float_power")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(float_power_Tensor_Scalar_out, overload_name, "Tensor_Scalar_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(float_power_Tensor_Scalar_out, schema_str, "float_power.Tensor_Scalar_out(Tensor self, Scalar exponent, *, Tensor(a!) out) -> Tensor(a!)")

// aten::float_power.Tensor_Scalar_out(Tensor self, Scalar exponent, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<float_power_Tensor_Scalar_out::schema> create_float_power_Tensor_Scalar_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(float_power_Tensor_Scalar_out::name, float_power_Tensor_Scalar_out::overload_name)
      .typed<float_power_Tensor_Scalar_out::schema>();
}

// aten::float_power.Tensor_Scalar_out(Tensor self, Scalar exponent, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & float_power_Tensor_Scalar_out::call(const at::Tensor & self, const at::Scalar & exponent, at::Tensor & out) {
    static auto op = create_float_power_Tensor_Scalar_out_typed_handle();
    return op.call(self, exponent, out);
}

// aten::float_power.Tensor_Scalar_out(Tensor self, Scalar exponent, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & float_power_Tensor_Scalar_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & exponent, at::Tensor & out) {
    static auto op = create_float_power_Tensor_Scalar_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, exponent, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(normal_float_float, name, "aten::normal")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(normal_float_float, overload_name, "float_float")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(normal_float_float, schema_str, "normal.float_float(float mean, float std, int[] size, *, Generator? generator=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::normal.float_float(float mean, float std, int[] size, *, Generator? generator=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<normal_float_float::schema> create_normal_float_float_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(normal_float_float::name, normal_float_float::overload_name)
      .typed<normal_float_float::schema>();
}

// aten::normal.float_float(float mean, float std, int[] size, *, Generator? generator=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor normal_float_float::call(double mean, double std, at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_normal_float_float_typed_handle();
    return op.call(mean, std, size, generator, dtype, layout, device, pin_memory);
}

// aten::normal.float_float(float mean, float std, int[] size, *, Generator? generator=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor normal_float_float::redispatch(c10::DispatchKeySet dispatchKeySet, double mean, double std, at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_normal_float_float_typed_handle();
    return op.redispatch(dispatchKeySet, mean, std, size, generator, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(normal_float_float_out, name, "aten::normal")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(normal_float_float_out, overload_name, "float_float_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(normal_float_float_out, schema_str, "normal.float_float_out(float mean, float std, int[] size, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)")

// aten::normal.float_float_out(float mean, float std, int[] size, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<normal_float_float_out::schema> create_normal_float_float_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(normal_float_float_out::name, normal_float_float_out::overload_name)
      .typed<normal_float_float_out::schema>();
}

// aten::normal.float_float_out(float mean, float std, int[] size, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & normal_float_float_out::call(double mean, double std, at::IntArrayRef size, c10::optional<at::Generator> generator, at::Tensor & out) {
    static auto op = create_normal_float_float_out_typed_handle();
    return op.call(mean, std, size, generator, out);
}

// aten::normal.float_float_out(float mean, float std, int[] size, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & normal_float_float_out::redispatch(c10::DispatchKeySet dispatchKeySet, double mean, double std, at::IntArrayRef size, c10::optional<at::Generator> generator, at::Tensor & out) {
    static auto op = create_normal_float_float_out_typed_handle();
    return op.redispatch(dispatchKeySet, mean, std, size, generator, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_index_copy_, name, "aten::_index_copy_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_index_copy_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_index_copy_, schema_str, "_index_copy_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)")

// aten::_index_copy_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<_index_copy_::schema> create__index_copy__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_index_copy_::name, _index_copy_::overload_name)
      .typed<_index_copy_::schema>();
}

// aten::_index_copy_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)
at::Tensor & _index_copy_::call(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source) {
    static auto op = create__index_copy__typed_handle();
    return op.call(self, dim, index, source);
}

// aten::_index_copy_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)
at::Tensor & _index_copy_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source) {
    static auto op = create__index_copy__typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, source);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_add__Scalar, name, "aten::_foreach_add_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_add__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_add__Scalar, schema_str, "_foreach_add_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()")

// aten::_foreach_add_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_add__Scalar::schema> create__foreach_add__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_add__Scalar::name, _foreach_add__Scalar::overload_name)
      .typed<_foreach_add__Scalar::schema>();
}

// aten::_foreach_add_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()
void _foreach_add__Scalar::call(at::TensorList self, const at::Scalar & scalar) {
    static auto op = create__foreach_add__Scalar_typed_handle();
    return op.call(self, scalar);
}

// aten::_foreach_add_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()
void _foreach_add__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self, const at::Scalar & scalar) {
    static auto op = create__foreach_add__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, scalar);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sub_Scalar, name, "aten::_foreach_sub")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sub_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sub_Scalar, schema_str, "_foreach_sub.Scalar(Tensor[] tensors, Scalar scalar) -> Tensor[]")

// aten::_foreach_sub.Scalar(Tensor[] tensors, Scalar scalar) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_sub_Scalar::schema> create__foreach_sub_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_sub_Scalar::name, _foreach_sub_Scalar::overload_name)
      .typed<_foreach_sub_Scalar::schema>();
}

// aten::_foreach_sub.Scalar(Tensor[] tensors, Scalar scalar) -> Tensor[]
::std::vector<at::Tensor> _foreach_sub_Scalar::call(at::TensorList tensors, const at::Scalar & scalar) {
    static auto op = create__foreach_sub_Scalar_typed_handle();
    return op.call(tensors, scalar);
}

// aten::_foreach_sub.Scalar(Tensor[] tensors, Scalar scalar) -> Tensor[]
::std::vector<at::Tensor> _foreach_sub_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, const at::Scalar & scalar) {
    static auto op = create__foreach_sub_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, tensors, scalar);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sub__Scalar, name, "aten::_foreach_sub_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sub__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sub__Scalar, schema_str, "_foreach_sub_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()")

// aten::_foreach_sub_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_sub__Scalar::schema> create__foreach_sub__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_sub__Scalar::name, _foreach_sub__Scalar::overload_name)
      .typed<_foreach_sub__Scalar::schema>();
}

// aten::_foreach_sub_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()
void _foreach_sub__Scalar::call(at::TensorList self, const at::Scalar & scalar) {
    static auto op = create__foreach_sub__Scalar_typed_handle();
    return op.call(self, scalar);
}

// aten::_foreach_sub_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()
void _foreach_sub__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self, const at::Scalar & scalar) {
    static auto op = create__foreach_sub__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, scalar);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_add_List, name, "aten::_foreach_add")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_add_List, overload_name, "List")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_add_List, schema_str, "_foreach_add.List(Tensor[] tensors1, Tensor[] tensors2, *, Scalar alpha=1) -> Tensor[]")

// aten::_foreach_add.List(Tensor[] tensors1, Tensor[] tensors2, *, Scalar alpha=1) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_add_List::schema> create__foreach_add_List_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_add_List::name, _foreach_add_List::overload_name)
      .typed<_foreach_add_List::schema>();
}

// aten::_foreach_add.List(Tensor[] tensors1, Tensor[] tensors2, *, Scalar alpha=1) -> Tensor[]
::std::vector<at::Tensor> _foreach_add_List::call(at::TensorList tensors1, at::TensorList tensors2, const at::Scalar & alpha) {
    static auto op = create__foreach_add_List_typed_handle();
    return op.call(tensors1, tensors2, alpha);
}

// aten::_foreach_add.List(Tensor[] tensors1, Tensor[] tensors2, *, Scalar alpha=1) -> Tensor[]
::std::vector<at::Tensor> _foreach_add_List::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors1, at::TensorList tensors2, const at::Scalar & alpha) {
    static auto op = create__foreach_add_List_typed_handle();
    return op.redispatch(dispatchKeySet, tensors1, tensors2, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_add__List, name, "aten::_foreach_add_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_add__List, overload_name, "List")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_add__List, schema_str, "_foreach_add_.List(Tensor(a!)[] self, Tensor[] other, *, Scalar alpha=1) -> ()")

// aten::_foreach_add_.List(Tensor(a!)[] self, Tensor[] other, *, Scalar alpha=1) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_add__List::schema> create__foreach_add__List_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_add__List::name, _foreach_add__List::overload_name)
      .typed<_foreach_add__List::schema>();
}

// aten::_foreach_add_.List(Tensor(a!)[] self, Tensor[] other, *, Scalar alpha=1) -> ()
void _foreach_add__List::call(at::TensorList self, at::TensorList other, const at::Scalar & alpha) {
    static auto op = create__foreach_add__List_typed_handle();
    return op.call(self, other, alpha);
}

// aten::_foreach_add_.List(Tensor(a!)[] self, Tensor[] other, *, Scalar alpha=1) -> ()
void _foreach_add__List::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self, at::TensorList other, const at::Scalar & alpha) {
    static auto op = create__foreach_add__List_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_div__List, name, "aten::_foreach_div_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_div__List, overload_name, "List")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_div__List, schema_str, "_foreach_div_.List(Tensor(a!)[] self, Tensor[] other) -> ()")

// aten::_foreach_div_.List(Tensor(a!)[] self, Tensor[] other) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_div__List::schema> create__foreach_div__List_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_div__List::name, _foreach_div__List::overload_name)
      .typed<_foreach_div__List::schema>();
}

// aten::_foreach_div_.List(Tensor(a!)[] self, Tensor[] other) -> ()
void _foreach_div__List::call(at::TensorList self, at::TensorList other) {
    static auto op = create__foreach_div__List_typed_handle();
    return op.call(self, other);
}

// aten::_foreach_div_.List(Tensor(a!)[] self, Tensor[] other) -> ()
void _foreach_div__List::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self, at::TensorList other) {
    static auto op = create__foreach_div__List_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sub_ScalarList, name, "aten::_foreach_sub")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sub_ScalarList, overload_name, "ScalarList")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sub_ScalarList, schema_str, "_foreach_sub.ScalarList(Tensor[] tensors, Scalar[] scalars) -> Tensor[]")

// aten::_foreach_sub.ScalarList(Tensor[] tensors, Scalar[] scalars) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_sub_ScalarList::schema> create__foreach_sub_ScalarList_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_sub_ScalarList::name, _foreach_sub_ScalarList::overload_name)
      .typed<_foreach_sub_ScalarList::schema>();
}

// aten::_foreach_sub.ScalarList(Tensor[] tensors, Scalar[] scalars) -> Tensor[]
::std::vector<at::Tensor> _foreach_sub_ScalarList::call(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars) {
    static auto op = create__foreach_sub_ScalarList_typed_handle();
    return op.call(tensors, scalars);
}

// aten::_foreach_sub.ScalarList(Tensor[] tensors, Scalar[] scalars) -> Tensor[]
::std::vector<at::Tensor> _foreach_sub_ScalarList::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, at::ArrayRef<at::Scalar> scalars) {
    static auto op = create__foreach_sub_ScalarList_typed_handle();
    return op.redispatch(dispatchKeySet, tensors, scalars);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_div_ScalarList, name, "aten::_foreach_div")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_div_ScalarList, overload_name, "ScalarList")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_div_ScalarList, schema_str, "_foreach_div.ScalarList(Tensor[] tensors, Scalar[] scalars) -> Tensor[]")

// aten::_foreach_div.ScalarList(Tensor[] tensors, Scalar[] scalars) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_div_ScalarList::schema> create__foreach_div_ScalarList_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_div_ScalarList::name, _foreach_div_ScalarList::overload_name)
      .typed<_foreach_div_ScalarList::schema>();
}

// aten::_foreach_div.ScalarList(Tensor[] tensors, Scalar[] scalars) -> Tensor[]
::std::vector<at::Tensor> _foreach_div_ScalarList::call(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars) {
    static auto op = create__foreach_div_ScalarList_typed_handle();
    return op.call(tensors, scalars);
}

// aten::_foreach_div.ScalarList(Tensor[] tensors, Scalar[] scalars) -> Tensor[]
::std::vector<at::Tensor> _foreach_div_ScalarList::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, at::ArrayRef<at::Scalar> scalars) {
    static auto op = create__foreach_div_ScalarList_typed_handle();
    return op.redispatch(dispatchKeySet, tensors, scalars);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_mul__ScalarList, name, "aten::_foreach_mul_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_mul__ScalarList, overload_name, "ScalarList")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_mul__ScalarList, schema_str, "_foreach_mul_.ScalarList(Tensor(a!)[] self, Scalar[] scalars) -> ()")

// aten::_foreach_mul_.ScalarList(Tensor(a!)[] self, Scalar[] scalars) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_mul__ScalarList::schema> create__foreach_mul__ScalarList_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_mul__ScalarList::name, _foreach_mul__ScalarList::overload_name)
      .typed<_foreach_mul__ScalarList::schema>();
}

// aten::_foreach_mul_.ScalarList(Tensor(a!)[] self, Scalar[] scalars) -> ()
void _foreach_mul__ScalarList::call(at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
    static auto op = create__foreach_mul__ScalarList_typed_handle();
    return op.call(self, scalars);
}

// aten::_foreach_mul_.ScalarList(Tensor(a!)[] self, Scalar[] scalars) -> ()
void _foreach_mul__ScalarList::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
    static auto op = create__foreach_mul__ScalarList_typed_handle();
    return op.redispatch(dispatchKeySet, self, scalars);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_exp, name, "aten::_foreach_exp")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_exp, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_exp, schema_str, "_foreach_exp(Tensor[] tensors) -> Tensor[]")

// aten::_foreach_exp(Tensor[] tensors) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_exp::schema> create__foreach_exp_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_exp::name, _foreach_exp::overload_name)
      .typed<_foreach_exp::schema>();
}

// aten::_foreach_exp(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_exp::call(at::TensorList tensors) {
    static auto op = create__foreach_exp_typed_handle();
    return op.call(tensors);
}

// aten::_foreach_exp(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_exp::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create__foreach_exp_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sqrt, name, "aten::_foreach_sqrt")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sqrt, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sqrt, schema_str, "_foreach_sqrt(Tensor[] tensors) -> Tensor[]")

// aten::_foreach_sqrt(Tensor[] tensors) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_sqrt::schema> create__foreach_sqrt_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_sqrt::name, _foreach_sqrt::overload_name)
      .typed<_foreach_sqrt::schema>();
}

// aten::_foreach_sqrt(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_sqrt::call(at::TensorList tensors) {
    static auto op = create__foreach_sqrt_typed_handle();
    return op.call(tensors);
}

// aten::_foreach_sqrt(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_sqrt::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create__foreach_sqrt_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_log, name, "aten::_foreach_log")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_log, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_log, schema_str, "_foreach_log(Tensor[] tensors) -> Tensor[]")

// aten::_foreach_log(Tensor[] tensors) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_log::schema> create__foreach_log_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_log::name, _foreach_log::overload_name)
      .typed<_foreach_log::schema>();
}

// aten::_foreach_log(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_log::call(at::TensorList tensors) {
    static auto op = create__foreach_log_typed_handle();
    return op.call(tensors);
}

// aten::_foreach_log(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_log::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create__foreach_log_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_log1p, name, "aten::_foreach_log1p")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_log1p, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_log1p, schema_str, "_foreach_log1p(Tensor[] tensors) -> Tensor[]")

// aten::_foreach_log1p(Tensor[] tensors) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_log1p::schema> create__foreach_log1p_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_log1p::name, _foreach_log1p::overload_name)
      .typed<_foreach_log1p::schema>();
}

// aten::_foreach_log1p(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_log1p::call(at::TensorList tensors) {
    static auto op = create__foreach_log1p_typed_handle();
    return op.call(tensors);
}

// aten::_foreach_log1p(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_log1p::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create__foreach_log1p_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_neg, name, "aten::_foreach_neg")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_neg, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_neg, schema_str, "_foreach_neg(Tensor[] tensors) -> Tensor[]")

// aten::_foreach_neg(Tensor[] tensors) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_neg::schema> create__foreach_neg_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_neg::name, _foreach_neg::overload_name)
      .typed<_foreach_neg::schema>();
}

// aten::_foreach_neg(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_neg::call(at::TensorList tensors) {
    static auto op = create__foreach_neg_typed_handle();
    return op.call(tensors);
}

// aten::_foreach_neg(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_neg::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create__foreach_neg_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_tanh_, name, "aten::_foreach_tanh_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_tanh_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_tanh_, schema_str, "_foreach_tanh_(Tensor(a!)[] self) -> ()")

// aten::_foreach_tanh_(Tensor(a!)[] self) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_tanh_::schema> create__foreach_tanh__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_tanh_::name, _foreach_tanh_::overload_name)
      .typed<_foreach_tanh_::schema>();
}

// aten::_foreach_tanh_(Tensor(a!)[] self) -> ()
void _foreach_tanh_::call(at::TensorList self) {
    static auto op = create__foreach_tanh__typed_handle();
    return op.call(self);
}

// aten::_foreach_tanh_(Tensor(a!)[] self) -> ()
void _foreach_tanh_::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self) {
    static auto op = create__foreach_tanh__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sin, name, "aten::_foreach_sin")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sin, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sin, schema_str, "_foreach_sin(Tensor[] tensors) -> Tensor[]")

// aten::_foreach_sin(Tensor[] tensors) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_sin::schema> create__foreach_sin_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_sin::name, _foreach_sin::overload_name)
      .typed<_foreach_sin::schema>();
}

// aten::_foreach_sin(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_sin::call(at::TensorList tensors) {
    static auto op = create__foreach_sin_typed_handle();
    return op.call(tensors);
}

// aten::_foreach_sin(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_sin::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create__foreach_sin_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sinh_, name, "aten::_foreach_sinh_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sinh_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sinh_, schema_str, "_foreach_sinh_(Tensor(a!)[] self) -> ()")

// aten::_foreach_sinh_(Tensor(a!)[] self) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_sinh_::schema> create__foreach_sinh__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_sinh_::name, _foreach_sinh_::overload_name)
      .typed<_foreach_sinh_::schema>();
}

// aten::_foreach_sinh_(Tensor(a!)[] self) -> ()
void _foreach_sinh_::call(at::TensorList self) {
    static auto op = create__foreach_sinh__typed_handle();
    return op.call(self);
}

// aten::_foreach_sinh_(Tensor(a!)[] self) -> ()
void _foreach_sinh_::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self) {
    static auto op = create__foreach_sinh__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_reciprocal, name, "aten::_foreach_reciprocal")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_reciprocal, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_reciprocal, schema_str, "_foreach_reciprocal(Tensor[] tensors) -> Tensor[]")

// aten::_foreach_reciprocal(Tensor[] tensors) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_reciprocal::schema> create__foreach_reciprocal_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_reciprocal::name, _foreach_reciprocal::overload_name)
      .typed<_foreach_reciprocal::schema>();
}

// aten::_foreach_reciprocal(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_reciprocal::call(at::TensorList tensors) {
    static auto op = create__foreach_reciprocal_typed_handle();
    return op.call(tensors);
}

// aten::_foreach_reciprocal(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_reciprocal::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create__foreach_reciprocal_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_reciprocal_, name, "aten::_foreach_reciprocal_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_reciprocal_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_reciprocal_, schema_str, "_foreach_reciprocal_(Tensor(a!)[] self) -> ()")

// aten::_foreach_reciprocal_(Tensor(a!)[] self) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_reciprocal_::schema> create__foreach_reciprocal__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_reciprocal_::name, _foreach_reciprocal_::overload_name)
      .typed<_foreach_reciprocal_::schema>();
}

// aten::_foreach_reciprocal_(Tensor(a!)[] self) -> ()
void _foreach_reciprocal_::call(at::TensorList self) {
    static auto op = create__foreach_reciprocal__typed_handle();
    return op.call(self);
}

// aten::_foreach_reciprocal_(Tensor(a!)[] self) -> ()
void _foreach_reciprocal_::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self) {
    static auto op = create__foreach_reciprocal__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sigmoid, name, "aten::_foreach_sigmoid")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sigmoid, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sigmoid, schema_str, "_foreach_sigmoid(Tensor[] tensors) -> Tensor[]")

// aten::_foreach_sigmoid(Tensor[] tensors) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_sigmoid::schema> create__foreach_sigmoid_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_sigmoid::name, _foreach_sigmoid::overload_name)
      .typed<_foreach_sigmoid::schema>();
}

// aten::_foreach_sigmoid(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_sigmoid::call(at::TensorList tensors) {
    static auto op = create__foreach_sigmoid_typed_handle();
    return op.call(tensors);
}

// aten::_foreach_sigmoid(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_sigmoid::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create__foreach_sigmoid_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_addcdiv__Scalar, name, "aten::_foreach_addcdiv_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_addcdiv__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_addcdiv__Scalar, schema_str, "_foreach_addcdiv_.Scalar(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1) -> ()")

// aten::_foreach_addcdiv_.Scalar(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_addcdiv__Scalar::schema> create__foreach_addcdiv__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_addcdiv__Scalar::name, _foreach_addcdiv__Scalar::overload_name)
      .typed<_foreach_addcdiv__Scalar::schema>();
}

// aten::_foreach_addcdiv_.Scalar(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1) -> ()
void _foreach_addcdiv__Scalar::call(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value) {
    static auto op = create__foreach_addcdiv__Scalar_typed_handle();
    return op.call(self, tensor1, tensor2, value);
}

// aten::_foreach_addcdiv_.Scalar(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1) -> ()
void _foreach_addcdiv__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value) {
    static auto op = create__foreach_addcdiv__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, tensor1, tensor2, value);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_addcmul_ScalarList, name, "aten::_foreach_addcmul")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_addcmul_ScalarList, overload_name, "ScalarList")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_addcmul_ScalarList, schema_str, "_foreach_addcmul.ScalarList(Tensor[] input, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars) -> Tensor[]")

// aten::_foreach_addcmul.ScalarList(Tensor[] input, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_addcmul_ScalarList::schema> create__foreach_addcmul_ScalarList_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_addcmul_ScalarList::name, _foreach_addcmul_ScalarList::overload_name)
      .typed<_foreach_addcmul_ScalarList::schema>();
}

// aten::_foreach_addcmul.ScalarList(Tensor[] input, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars) -> Tensor[]
::std::vector<at::Tensor> _foreach_addcmul_ScalarList::call(at::TensorList input, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars) {
    static auto op = create__foreach_addcmul_ScalarList_typed_handle();
    return op.call(input, tensor1, tensor2, scalars);
}

// aten::_foreach_addcmul.ScalarList(Tensor[] input, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars) -> Tensor[]
::std::vector<at::Tensor> _foreach_addcmul_ScalarList::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList input, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars) {
    static auto op = create__foreach_addcmul_ScalarList_typed_handle();
    return op.redispatch(dispatchKeySet, input, tensor1, tensor2, scalars);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_minimum_List, name, "aten::_foreach_minimum")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_minimum_List, overload_name, "List")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_minimum_List, schema_str, "_foreach_minimum.List(Tensor[] tensors1, Tensor[] tensors2) -> Tensor[]")

// aten::_foreach_minimum.List(Tensor[] tensors1, Tensor[] tensors2) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_minimum_List::schema> create__foreach_minimum_List_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_minimum_List::name, _foreach_minimum_List::overload_name)
      .typed<_foreach_minimum_List::schema>();
}

// aten::_foreach_minimum.List(Tensor[] tensors1, Tensor[] tensors2) -> Tensor[]
::std::vector<at::Tensor> _foreach_minimum_List::call(at::TensorList tensors1, at::TensorList tensors2) {
    static auto op = create__foreach_minimum_List_typed_handle();
    return op.call(tensors1, tensors2);
}

// aten::_foreach_minimum.List(Tensor[] tensors1, Tensor[] tensors2) -> Tensor[]
::std::vector<at::Tensor> _foreach_minimum_List::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors1, at::TensorList tensors2) {
    static auto op = create__foreach_minimum_List_typed_handle();
    return op.redispatch(dispatchKeySet, tensors1, tensors2);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_convert_indices_from_coo_to_csr, name, "aten::_convert_indices_from_coo_to_csr")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_convert_indices_from_coo_to_csr, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_convert_indices_from_coo_to_csr, schema_str, "_convert_indices_from_coo_to_csr(Tensor self, int size, *, bool out_int32=False) -> Tensor")

// aten::_convert_indices_from_coo_to_csr(Tensor self, int size, *, bool out_int32=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_convert_indices_from_coo_to_csr::schema> create__convert_indices_from_coo_to_csr_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_convert_indices_from_coo_to_csr::name, _convert_indices_from_coo_to_csr::overload_name)
      .typed<_convert_indices_from_coo_to_csr::schema>();
}

// aten::_convert_indices_from_coo_to_csr(Tensor self, int size, *, bool out_int32=False) -> Tensor
at::Tensor _convert_indices_from_coo_to_csr::call(const at::Tensor & self, int64_t size, bool out_int32) {
    static auto op = create__convert_indices_from_coo_to_csr_typed_handle();
    return op.call(self, size, out_int32);
}

// aten::_convert_indices_from_coo_to_csr(Tensor self, int size, *, bool out_int32=False) -> Tensor
at::Tensor _convert_indices_from_coo_to_csr::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t size, bool out_int32) {
    static auto op = create__convert_indices_from_coo_to_csr_typed_handle();
    return op.redispatch(dispatchKeySet, self, size, out_int32);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mse_loss_out, name, "aten::mse_loss")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mse_loss_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mse_loss_out, schema_str, "mse_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)")

// aten::mse_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<mse_loss_out::schema> create_mse_loss_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mse_loss_out::name, mse_loss_out::overload_name)
      .typed<mse_loss_out::schema>();
}

// aten::mse_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & mse_loss_out::call(const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & out) {
    static auto op = create_mse_loss_out_typed_handle();
    return op.call(self, target, reduction, out);
}

// aten::mse_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & mse_loss_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & out) {
    static auto op = create_mse_loss_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, target, reduction, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(l1_loss_backward, name, "aten::l1_loss_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(l1_loss_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(l1_loss_backward, schema_str, "l1_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor")

// aten::l1_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<l1_loss_backward::schema> create_l1_loss_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(l1_loss_backward::name, l1_loss_backward::overload_name)
      .typed<l1_loss_backward::schema>();
}

// aten::l1_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor
at::Tensor l1_loss_backward::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
    static auto op = create_l1_loss_backward_typed_handle();
    return op.call(grad_output, self, target, reduction);
}

// aten::l1_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor
at::Tensor l1_loss_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
    static auto op = create_l1_loss_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, target, reduction);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multi_margin_loss, name, "aten::multi_margin_loss")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multi_margin_loss, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multi_margin_loss, schema_str, "multi_margin_loss(Tensor self, Tensor target, Scalar p=1, Scalar margin=1, Tensor? weight=None, int reduction=Mean) -> Tensor")

// aten::multi_margin_loss(Tensor self, Tensor target, Scalar p=1, Scalar margin=1, Tensor? weight=None, int reduction=Mean) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<multi_margin_loss::schema> create_multi_margin_loss_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(multi_margin_loss::name, multi_margin_loss::overload_name)
      .typed<multi_margin_loss::schema>();
}

// aten::multi_margin_loss(Tensor self, Tensor target, Scalar p=1, Scalar margin=1, Tensor? weight=None, int reduction=Mean) -> Tensor
at::Tensor multi_margin_loss::call(const at::Tensor & self, const at::Tensor & target, const at::Scalar & p, const at::Scalar & margin, const c10::optional<at::Tensor> & weight, int64_t reduction) {
    static auto op = create_multi_margin_loss_typed_handle();
    return op.call(self, target, p, margin, weight, reduction);
}

// aten::multi_margin_loss(Tensor self, Tensor target, Scalar p=1, Scalar margin=1, Tensor? weight=None, int reduction=Mean) -> Tensor
at::Tensor multi_margin_loss::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, const at::Scalar & p, const at::Scalar & margin, const c10::optional<at::Tensor> & weight, int64_t reduction) {
    static auto op = create_multi_margin_loss_typed_handle();
    return op.redispatch(dispatchKeySet, self, target, p, margin, weight, reduction);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multilabel_margin_loss_out, name, "aten::multilabel_margin_loss")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multilabel_margin_loss_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multilabel_margin_loss_out, schema_str, "multilabel_margin_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)")

// aten::multilabel_margin_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<multilabel_margin_loss_out::schema> create_multilabel_margin_loss_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(multilabel_margin_loss_out::name, multilabel_margin_loss_out::overload_name)
      .typed<multilabel_margin_loss_out::schema>();
}

// aten::multilabel_margin_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & multilabel_margin_loss_out::call(const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & out) {
    static auto op = create_multilabel_margin_loss_out_typed_handle();
    return op.call(self, target, reduction, out);
}

// aten::multilabel_margin_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & multilabel_margin_loss_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & out) {
    static auto op = create_multilabel_margin_loss_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, target, reduction, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multilabel_margin_loss, name, "aten::multilabel_margin_loss")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multilabel_margin_loss, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multilabel_margin_loss, schema_str, "multilabel_margin_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor")

// aten::multilabel_margin_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<multilabel_margin_loss::schema> create_multilabel_margin_loss_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(multilabel_margin_loss::name, multilabel_margin_loss::overload_name)
      .typed<multilabel_margin_loss::schema>();
}

// aten::multilabel_margin_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor
at::Tensor multilabel_margin_loss::call(const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
    static auto op = create_multilabel_margin_loss_typed_handle();
    return op.call(self, target, reduction);
}

// aten::multilabel_margin_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor
at::Tensor multilabel_margin_loss::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
    static auto op = create_multilabel_margin_loss_typed_handle();
    return op.redispatch(dispatchKeySet, self, target, reduction);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multilabel_margin_loss_forward, name, "aten::multilabel_margin_loss_forward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multilabel_margin_loss_forward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multilabel_margin_loss_forward, schema_str, "multilabel_margin_loss_forward(Tensor self, Tensor target, int reduction) -> (Tensor output, Tensor is_target)")

// aten::multilabel_margin_loss_forward(Tensor self, Tensor target, int reduction) -> (Tensor output, Tensor is_target)
static C10_NOINLINE c10::TypedOperatorHandle<multilabel_margin_loss_forward::schema> create_multilabel_margin_loss_forward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(multilabel_margin_loss_forward::name, multilabel_margin_loss_forward::overload_name)
      .typed<multilabel_margin_loss_forward::schema>();
}

// aten::multilabel_margin_loss_forward(Tensor self, Tensor target, int reduction) -> (Tensor output, Tensor is_target)
::std::tuple<at::Tensor,at::Tensor> multilabel_margin_loss_forward::call(const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
    static auto op = create_multilabel_margin_loss_forward_typed_handle();
    return op.call(self, target, reduction);
}

// aten::multilabel_margin_loss_forward(Tensor self, Tensor target, int reduction) -> (Tensor output, Tensor is_target)
::std::tuple<at::Tensor,at::Tensor> multilabel_margin_loss_forward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
    static auto op = create_multilabel_margin_loss_forward_typed_handle();
    return op.redispatch(dispatchKeySet, self, target, reduction);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss_out, name, "aten::nll_loss")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss_out, schema_str, "nll_loss.out(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100, *, Tensor(a!) out) -> Tensor(a!)")

// aten::nll_loss.out(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<nll_loss_out::schema> create_nll_loss_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nll_loss_out::name, nll_loss_out::overload_name)
      .typed<nll_loss_out::schema>();
}

// aten::nll_loss.out(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & nll_loss_out::call(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, at::Tensor & out) {
    static auto op = create_nll_loss_out_typed_handle();
    return op.call(self, target, weight, reduction, ignore_index, out);
}

// aten::nll_loss.out(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & nll_loss_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, at::Tensor & out) {
    static auto op = create_nll_loss_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, target, weight, reduction, ignore_index, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss_forward, name, "aten::nll_loss_forward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss_forward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss_forward, schema_str, "nll_loss_forward(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index) -> (Tensor output, Tensor total_weight)")

// aten::nll_loss_forward(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index) -> (Tensor output, Tensor total_weight)
static C10_NOINLINE c10::TypedOperatorHandle<nll_loss_forward::schema> create_nll_loss_forward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nll_loss_forward::name, nll_loss_forward::overload_name)
      .typed<nll_loss_forward::schema>();
}

// aten::nll_loss_forward(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index) -> (Tensor output, Tensor total_weight)
::std::tuple<at::Tensor,at::Tensor> nll_loss_forward::call(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index) {
    static auto op = create_nll_loss_forward_typed_handle();
    return op.call(self, target, weight, reduction, ignore_index);
}

// aten::nll_loss_forward(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index) -> (Tensor output, Tensor total_weight)
::std::tuple<at::Tensor,at::Tensor> nll_loss_forward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index) {
    static auto op = create_nll_loss_forward_typed_handle();
    return op.redispatch(dispatchKeySet, self, target, weight, reduction, ignore_index);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss2d_forward_output, name, "aten::nll_loss2d_forward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss2d_forward_output, overload_name, "output")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss2d_forward_output, schema_str, "nll_loss2d_forward.output(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, *, Tensor(a!) output, Tensor(b!) total_weight) -> (Tensor(a!), Tensor(b!))")

// aten::nll_loss2d_forward.output(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, *, Tensor(a!) output, Tensor(b!) total_weight) -> (Tensor(a!), Tensor(b!))
static C10_NOINLINE c10::TypedOperatorHandle<nll_loss2d_forward_output::schema> create_nll_loss2d_forward_output_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nll_loss2d_forward_output::name, nll_loss2d_forward_output::overload_name)
      .typed<nll_loss2d_forward_output::schema>();
}

// aten::nll_loss2d_forward.output(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, *, Tensor(a!) output, Tensor(b!) total_weight) -> (Tensor(a!), Tensor(b!))
::std::tuple<at::Tensor &,at::Tensor &> nll_loss2d_forward_output::call(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, at::Tensor & output, at::Tensor & total_weight) {
    static auto op = create_nll_loss2d_forward_output_typed_handle();
    return op.call(self, target, weight, reduction, ignore_index, output, total_weight);
}

// aten::nll_loss2d_forward.output(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, *, Tensor(a!) output, Tensor(b!) total_weight) -> (Tensor(a!), Tensor(b!))
::std::tuple<at::Tensor &,at::Tensor &> nll_loss2d_forward_output::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, at::Tensor & output, at::Tensor & total_weight) {
    static auto op = create_nll_loss2d_forward_output_typed_handle();
    return op.redispatch(dispatchKeySet, self, target, weight, reduction, ignore_index, output, total_weight);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(soft_margin_loss_out, name, "aten::soft_margin_loss")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(soft_margin_loss_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(soft_margin_loss_out, schema_str, "soft_margin_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)")

// aten::soft_margin_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<soft_margin_loss_out::schema> create_soft_margin_loss_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(soft_margin_loss_out::name, soft_margin_loss_out::overload_name)
      .typed<soft_margin_loss_out::schema>();
}

// aten::soft_margin_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & soft_margin_loss_out::call(const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & out) {
    static auto op = create_soft_margin_loss_out_typed_handle();
    return op.call(self, target, reduction, out);
}

// aten::soft_margin_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & soft_margin_loss_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & out) {
    static auto op = create_soft_margin_loss_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, target, reduction, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(soft_margin_loss_backward, name, "aten::soft_margin_loss_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(soft_margin_loss_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(soft_margin_loss_backward, schema_str, "soft_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor")

// aten::soft_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<soft_margin_loss_backward::schema> create_soft_margin_loss_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(soft_margin_loss_backward::name, soft_margin_loss_backward::overload_name)
      .typed<soft_margin_loss_backward::schema>();
}

// aten::soft_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor
at::Tensor soft_margin_loss_backward::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
    static auto op = create_soft_margin_loss_backward_typed_handle();
    return op.call(grad_output, self, target, reduction);
}

// aten::soft_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor
at::Tensor soft_margin_loss_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
    static auto op = create_soft_margin_loss_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, target, reduction);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(elu_out, name, "aten::elu")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(elu_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(elu_out, schema_str, "elu.out(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1, *, Tensor(a!) out) -> Tensor(a!)")

// aten::elu.out(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<elu_out::schema> create_elu_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(elu_out::name, elu_out::overload_name)
      .typed<elu_out::schema>();
}

// aten::elu.out(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & elu_out::call(const at::Tensor & self, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale, at::Tensor & out) {
    static auto op = create_elu_out_typed_handle();
    return op.call(self, alpha, scale, input_scale, out);
}

// aten::elu.out(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & elu_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale, at::Tensor & out) {
    static auto op = create_elu_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, alpha, scale, input_scale, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardsigmoid_out, name, "aten::hardsigmoid")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardsigmoid_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardsigmoid_out, schema_str, "hardsigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::hardsigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<hardsigmoid_out::schema> create_hardsigmoid_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(hardsigmoid_out::name, hardsigmoid_out::overload_name)
      .typed<hardsigmoid_out::schema>();
}

// aten::hardsigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & hardsigmoid_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_hardsigmoid_out_typed_handle();
    return op.call(self, out);
}

// aten::hardsigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & hardsigmoid_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_hardsigmoid_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardsigmoid_backward_grad_input, name, "aten::hardsigmoid_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardsigmoid_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardsigmoid_backward_grad_input, schema_str, "hardsigmoid_backward.grad_input(Tensor grad_output, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::hardsigmoid_backward.grad_input(Tensor grad_output, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<hardsigmoid_backward_grad_input::schema> create_hardsigmoid_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(hardsigmoid_backward_grad_input::name, hardsigmoid_backward_grad_input::overload_name)
      .typed<hardsigmoid_backward_grad_input::schema>();
}

// aten::hardsigmoid_backward.grad_input(Tensor grad_output, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & hardsigmoid_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, at::Tensor & grad_input) {
    static auto op = create_hardsigmoid_backward_grad_input_typed_handle();
    return op.call(grad_output, self, grad_input);
}

// aten::hardsigmoid_backward.grad_input(Tensor grad_output, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & hardsigmoid_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::Tensor & grad_input) {
    static auto op = create_hardsigmoid_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardswish, name, "aten::hardswish")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardswish, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardswish, schema_str, "hardswish(Tensor self) -> Tensor")

// aten::hardswish(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<hardswish::schema> create_hardswish_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(hardswish::name, hardswish::overload_name)
      .typed<hardswish::schema>();
}

// aten::hardswish(Tensor self) -> Tensor
at::Tensor hardswish::call(const at::Tensor & self) {
    static auto op = create_hardswish_typed_handle();
    return op.call(self);
}

// aten::hardswish(Tensor self) -> Tensor
at::Tensor hardswish::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_hardswish_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log_sigmoid_backward_grad_input, name, "aten::log_sigmoid_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log_sigmoid_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log_sigmoid_backward_grad_input, schema_str, "log_sigmoid_backward.grad_input(Tensor grad_output, Tensor self, Tensor buffer, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::log_sigmoid_backward.grad_input(Tensor grad_output, Tensor self, Tensor buffer, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<log_sigmoid_backward_grad_input::schema> create_log_sigmoid_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(log_sigmoid_backward_grad_input::name, log_sigmoid_backward_grad_input::overload_name)
      .typed<log_sigmoid_backward_grad_input::schema>();
}

// aten::log_sigmoid_backward.grad_input(Tensor grad_output, Tensor self, Tensor buffer, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & log_sigmoid_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & buffer, at::Tensor & grad_input) {
    static auto op = create_log_sigmoid_backward_grad_input_typed_handle();
    return op.call(grad_output, self, buffer, grad_input);
}

// aten::log_sigmoid_backward.grad_input(Tensor grad_output, Tensor self, Tensor buffer, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & log_sigmoid_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & buffer, at::Tensor & grad_input) {
    static auto op = create_log_sigmoid_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, buffer, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rrelu_with_noise, name, "aten::rrelu_with_noise")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rrelu_with_noise, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rrelu_with_noise, schema_str, "rrelu_with_noise(Tensor self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor")

// aten::rrelu_with_noise(Tensor self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<rrelu_with_noise::schema> create_rrelu_with_noise_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(rrelu_with_noise::name, rrelu_with_noise::overload_name)
      .typed<rrelu_with_noise::schema>();
}

// aten::rrelu_with_noise(Tensor self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor
at::Tensor rrelu_with_noise::call(const at::Tensor & self, const at::Tensor & noise, const at::Scalar & lower, const at::Scalar & upper, bool training, c10::optional<at::Generator> generator) {
    static auto op = create_rrelu_with_noise_typed_handle();
    return op.call(self, noise, lower, upper, training, generator);
}

// aten::rrelu_with_noise(Tensor self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor
at::Tensor rrelu_with_noise::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & noise, const at::Scalar & lower, const at::Scalar & upper, bool training, c10::optional<at::Generator> generator) {
    static auto op = create_rrelu_with_noise_typed_handle();
    return op.redispatch(dispatchKeySet, self, noise, lower, upper, training, generator);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(softshrink_backward_grad_input, name, "aten::softshrink_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(softshrink_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(softshrink_backward_grad_input, schema_str, "softshrink_backward.grad_input(Tensor grad_output, Tensor self, Scalar lambd, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::softshrink_backward.grad_input(Tensor grad_output, Tensor self, Scalar lambd, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<softshrink_backward_grad_input::schema> create_softshrink_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(softshrink_backward_grad_input::name, softshrink_backward_grad_input::overload_name)
      .typed<softshrink_backward_grad_input::schema>();
}

// aten::softshrink_backward.grad_input(Tensor grad_output, Tensor self, Scalar lambd, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & softshrink_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & lambd, at::Tensor & grad_input) {
    static auto op = create_softshrink_backward_grad_input_typed_handle();
    return op.call(grad_output, self, lambd, grad_input);
}

// aten::softshrink_backward.grad_input(Tensor grad_output, Tensor self, Scalar lambd, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & softshrink_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & lambd, at::Tensor & grad_input) {
    static auto op = create_softshrink_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, lambd, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(softshrink_backward, name, "aten::softshrink_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(softshrink_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(softshrink_backward, schema_str, "softshrink_backward(Tensor grad_output, Tensor self, Scalar lambd) -> Tensor")

// aten::softshrink_backward(Tensor grad_output, Tensor self, Scalar lambd) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<softshrink_backward::schema> create_softshrink_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(softshrink_backward::name, softshrink_backward::overload_name)
      .typed<softshrink_backward::schema>();
}

// aten::softshrink_backward(Tensor grad_output, Tensor self, Scalar lambd) -> Tensor
at::Tensor softshrink_backward::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & lambd) {
    static auto op = create_softshrink_backward_typed_handle();
    return op.call(grad_output, self, lambd);
}

// aten::softshrink_backward(Tensor grad_output, Tensor self, Scalar lambd) -> Tensor
at::Tensor softshrink_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & lambd) {
    static auto op = create_softshrink_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, lambd);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_adaptive_avg_pool2d_backward, name, "aten::_adaptive_avg_pool2d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_adaptive_avg_pool2d_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_adaptive_avg_pool2d_backward, schema_str, "_adaptive_avg_pool2d_backward(Tensor grad_output, Tensor self) -> Tensor")

// aten::_adaptive_avg_pool2d_backward(Tensor grad_output, Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_adaptive_avg_pool2d_backward::schema> create__adaptive_avg_pool2d_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_adaptive_avg_pool2d_backward::name, _adaptive_avg_pool2d_backward::overload_name)
      .typed<_adaptive_avg_pool2d_backward::schema>();
}

// aten::_adaptive_avg_pool2d_backward(Tensor grad_output, Tensor self) -> Tensor
at::Tensor _adaptive_avg_pool2d_backward::call(const at::Tensor & grad_output, const at::Tensor & self) {
    static auto op = create__adaptive_avg_pool2d_backward_typed_handle();
    return op.call(grad_output, self);
}

// aten::_adaptive_avg_pool2d_backward(Tensor grad_output, Tensor self) -> Tensor
at::Tensor _adaptive_avg_pool2d_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self) {
    static auto op = create__adaptive_avg_pool2d_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_max_pool3d_backward_grad_input, name, "aten::adaptive_max_pool3d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_max_pool3d_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_max_pool3d_backward_grad_input, schema_str, "adaptive_max_pool3d_backward.grad_input(Tensor grad_output, Tensor self, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::adaptive_max_pool3d_backward.grad_input(Tensor grad_output, Tensor self, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<adaptive_max_pool3d_backward_grad_input::schema> create_adaptive_max_pool3d_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(adaptive_max_pool3d_backward_grad_input::name, adaptive_max_pool3d_backward_grad_input::overload_name)
      .typed<adaptive_max_pool3d_backward_grad_input::schema>();
}

// aten::adaptive_max_pool3d_backward.grad_input(Tensor grad_output, Tensor self, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & adaptive_max_pool3d_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::Tensor & grad_input) {
    static auto op = create_adaptive_max_pool3d_backward_grad_input_typed_handle();
    return op.call(grad_output, self, indices, grad_input);
}

// aten::adaptive_max_pool3d_backward.grad_input(Tensor grad_output, Tensor self, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & adaptive_max_pool3d_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::Tensor & grad_input) {
    static auto op = create_adaptive_max_pool3d_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, indices, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(avg_pool2d, name, "aten::avg_pool2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(avg_pool2d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(avg_pool2d, schema_str, "avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor")

// aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<avg_pool2d::schema> create_avg_pool2d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(avg_pool2d::name, avg_pool2d::overload_name)
      .typed<avg_pool2d::schema>();
}

// aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
at::Tensor avg_pool2d::call(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
    static auto op = create_avg_pool2d_typed_handle();
    return op.call(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

// aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
at::Tensor avg_pool2d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
    static auto op = create_avg_pool2d_typed_handle();
    return op.redispatch(dispatchKeySet, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fractional_max_pool2d_backward, name, "aten::fractional_max_pool2d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fractional_max_pool2d_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fractional_max_pool2d_backward, schema_str, "fractional_max_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] output_size, Tensor indices) -> Tensor")

// aten::fractional_max_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] output_size, Tensor indices) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fractional_max_pool2d_backward::schema> create_fractional_max_pool2d_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fractional_max_pool2d_backward::name, fractional_max_pool2d_backward::overload_name)
      .typed<fractional_max_pool2d_backward::schema>();
}

// aten::fractional_max_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] output_size, Tensor indices) -> Tensor
at::Tensor fractional_max_pool2d_backward::call(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & indices) {
    static auto op = create_fractional_max_pool2d_backward_typed_handle();
    return op.call(grad_output, self, kernel_size, output_size, indices);
}

// aten::fractional_max_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] output_size, Tensor indices) -> Tensor
at::Tensor fractional_max_pool2d_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & indices) {
    static auto op = create_fractional_max_pool2d_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, kernel_size, output_size, indices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_pool2d_with_indices_out, name, "aten::max_pool2d_with_indices")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_pool2d_with_indices_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_pool2d_with_indices_out, schema_str, "max_pool2d_with_indices.out(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))")

// aten::max_pool2d_with_indices.out(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
static C10_NOINLINE c10::TypedOperatorHandle<max_pool2d_with_indices_out::schema> create_max_pool2d_with_indices_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(max_pool2d_with_indices_out::name, max_pool2d_with_indices_out::overload_name)
      .typed<max_pool2d_with_indices_out::schema>();
}

// aten::max_pool2d_with_indices.out(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
::std::tuple<at::Tensor &,at::Tensor &> max_pool2d_with_indices_out::call(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out, at::Tensor & indices) {
    static auto op = create_max_pool2d_with_indices_out_typed_handle();
    return op.call(self, kernel_size, stride, padding, dilation, ceil_mode, out, indices);
}

// aten::max_pool2d_with_indices.out(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
::std::tuple<at::Tensor &,at::Tensor &> max_pool2d_with_indices_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out, at::Tensor & indices) {
    static auto op = create_max_pool2d_with_indices_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, kernel_size, stride, padding, dilation, ceil_mode, out, indices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_pool3d_with_indices_out, name, "aten::max_pool3d_with_indices")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_pool3d_with_indices_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_pool3d_with_indices_out, schema_str, "max_pool3d_with_indices.out(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))")

// aten::max_pool3d_with_indices.out(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
static C10_NOINLINE c10::TypedOperatorHandle<max_pool3d_with_indices_out::schema> create_max_pool3d_with_indices_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(max_pool3d_with_indices_out::name, max_pool3d_with_indices_out::overload_name)
      .typed<max_pool3d_with_indices_out::schema>();
}

// aten::max_pool3d_with_indices.out(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
::std::tuple<at::Tensor &,at::Tensor &> max_pool3d_with_indices_out::call(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out, at::Tensor & indices) {
    static auto op = create_max_pool3d_with_indices_out_typed_handle();
    return op.call(self, kernel_size, stride, padding, dilation, ceil_mode, out, indices);
}

// aten::max_pool3d_with_indices.out(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
::std::tuple<at::Tensor &,at::Tensor &> max_pool3d_with_indices_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out, at::Tensor & indices) {
    static auto op = create_max_pool3d_with_indices_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, kernel_size, stride, padding, dilation, ceil_mode, out, indices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_pool3d_with_indices_backward, name, "aten::max_pool3d_with_indices_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_pool3d_with_indices_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_pool3d_with_indices_backward, schema_str, "max_pool3d_with_indices_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool ceil_mode, Tensor indices) -> Tensor")

// aten::max_pool3d_with_indices_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool ceil_mode, Tensor indices) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<max_pool3d_with_indices_backward::schema> create_max_pool3d_with_indices_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(max_pool3d_with_indices_backward::name, max_pool3d_with_indices_backward::overload_name)
      .typed<max_pool3d_with_indices_backward::schema>();
}

// aten::max_pool3d_with_indices_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool ceil_mode, Tensor indices) -> Tensor
at::Tensor max_pool3d_with_indices_backward::call(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices) {
    static auto op = create_max_pool3d_with_indices_backward_typed_handle();
    return op.call(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}

// aten::max_pool3d_with_indices_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool ceil_mode, Tensor indices) -> Tensor
at::Tensor max_pool3d_with_indices_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices) {
    static auto op = create_max_pool3d_with_indices_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_unpool2d_out, name, "aten::max_unpool2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_unpool2d_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_unpool2d_out, schema_str, "max_unpool2d.out(Tensor self, Tensor indices, int[2] output_size, *, Tensor(a!) out) -> Tensor(a!)")

// aten::max_unpool2d.out(Tensor self, Tensor indices, int[2] output_size, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<max_unpool2d_out::schema> create_max_unpool2d_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(max_unpool2d_out::name, max_unpool2d_out::overload_name)
      .typed<max_unpool2d_out::schema>();
}

// aten::max_unpool2d.out(Tensor self, Tensor indices, int[2] output_size, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & max_unpool2d_out::call(const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::Tensor & out) {
    static auto op = create_max_unpool2d_out_typed_handle();
    return op.call(self, indices, output_size, out);
}

// aten::max_unpool2d.out(Tensor self, Tensor indices, int[2] output_size, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & max_unpool2d_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::Tensor & out) {
    static auto op = create_max_unpool2d_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, indices, output_size, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_unpool3d_backward_grad_input, name, "aten::max_unpool3d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_unpool3d_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_unpool3d_backward_grad_input, schema_str, "max_unpool3d_backward.grad_input(Tensor grad_output, Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::max_unpool3d_backward.grad_input(Tensor grad_output, Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<max_unpool3d_backward_grad_input::schema> create_max_unpool3d_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(max_unpool3d_backward_grad_input::name, max_unpool3d_backward_grad_input::overload_name)
      .typed<max_unpool3d_backward_grad_input::schema>();
}

// aten::max_unpool3d_backward.grad_input(Tensor grad_output, Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & max_unpool3d_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & grad_input) {
    static auto op = create_max_unpool3d_backward_grad_input_typed_handle();
    return op.call(grad_output, self, indices, output_size, stride, padding, grad_input);
}

// aten::max_unpool3d_backward.grad_input(Tensor grad_output, Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & max_unpool3d_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & grad_input) {
    static auto op = create_max_unpool3d_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, indices, output_size, stride, padding, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_unpool3d_backward, name, "aten::max_unpool3d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_unpool3d_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_unpool3d_backward, schema_str, "max_unpool3d_backward(Tensor grad_output, Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding) -> Tensor")

// aten::max_unpool3d_backward(Tensor grad_output, Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<max_unpool3d_backward::schema> create_max_unpool3d_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(max_unpool3d_backward::name, max_unpool3d_backward::overload_name)
      .typed<max_unpool3d_backward::schema>();
}

// aten::max_unpool3d_backward(Tensor grad_output, Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding) -> Tensor
at::Tensor max_unpool3d_backward::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::IntArrayRef stride, at::IntArrayRef padding) {
    static auto op = create_max_unpool3d_backward_typed_handle();
    return op.call(grad_output, self, indices, output_size, stride, padding);
}

// aten::max_unpool3d_backward(Tensor grad_output, Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding) -> Tensor
at::Tensor max_unpool3d_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::IntArrayRef stride, at::IntArrayRef padding) {
    static auto op = create_max_unpool3d_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, indices, output_size, stride, padding);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reflection_pad1d_backward, name, "aten::reflection_pad1d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reflection_pad1d_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reflection_pad1d_backward, schema_str, "reflection_pad1d_backward(Tensor grad_output, Tensor self, int[2] padding) -> Tensor")

// aten::reflection_pad1d_backward(Tensor grad_output, Tensor self, int[2] padding) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<reflection_pad1d_backward::schema> create_reflection_pad1d_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(reflection_pad1d_backward::name, reflection_pad1d_backward::overload_name)
      .typed<reflection_pad1d_backward::schema>();
}

// aten::reflection_pad1d_backward(Tensor grad_output, Tensor self, int[2] padding) -> Tensor
at::Tensor reflection_pad1d_backward::call(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding) {
    static auto op = create_reflection_pad1d_backward_typed_handle();
    return op.call(grad_output, self, padding);
}

// aten::reflection_pad1d_backward(Tensor grad_output, Tensor self, int[2] padding) -> Tensor
at::Tensor reflection_pad1d_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding) {
    static auto op = create_reflection_pad1d_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, padding);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reflection_pad2d_backward_grad_input, name, "aten::reflection_pad2d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reflection_pad2d_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reflection_pad2d_backward_grad_input, schema_str, "reflection_pad2d_backward.grad_input(Tensor grad_output, Tensor self, int[4] padding, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::reflection_pad2d_backward.grad_input(Tensor grad_output, Tensor self, int[4] padding, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<reflection_pad2d_backward_grad_input::schema> create_reflection_pad2d_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(reflection_pad2d_backward_grad_input::name, reflection_pad2d_backward_grad_input::overload_name)
      .typed<reflection_pad2d_backward_grad_input::schema>();
}

// aten::reflection_pad2d_backward.grad_input(Tensor grad_output, Tensor self, int[4] padding, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & reflection_pad2d_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input) {
    static auto op = create_reflection_pad2d_backward_grad_input_typed_handle();
    return op.call(grad_output, self, padding, grad_input);
}

// aten::reflection_pad2d_backward.grad_input(Tensor grad_output, Tensor self, int[4] padding, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & reflection_pad2d_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input) {
    static auto op = create_reflection_pad2d_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, padding, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reflection_pad2d_backward, name, "aten::reflection_pad2d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reflection_pad2d_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reflection_pad2d_backward, schema_str, "reflection_pad2d_backward(Tensor grad_output, Tensor self, int[4] padding) -> Tensor")

// aten::reflection_pad2d_backward(Tensor grad_output, Tensor self, int[4] padding) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<reflection_pad2d_backward::schema> create_reflection_pad2d_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(reflection_pad2d_backward::name, reflection_pad2d_backward::overload_name)
      .typed<reflection_pad2d_backward::schema>();
}

// aten::reflection_pad2d_backward(Tensor grad_output, Tensor self, int[4] padding) -> Tensor
at::Tensor reflection_pad2d_backward::call(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding) {
    static auto op = create_reflection_pad2d_backward_typed_handle();
    return op.call(grad_output, self, padding);
}

// aten::reflection_pad2d_backward(Tensor grad_output, Tensor self, int[4] padding) -> Tensor
at::Tensor reflection_pad2d_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding) {
    static auto op = create_reflection_pad2d_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, padding);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reflection_pad3d_backward_grad_input, name, "aten::reflection_pad3d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reflection_pad3d_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reflection_pad3d_backward_grad_input, schema_str, "reflection_pad3d_backward.grad_input(Tensor grad_output, Tensor self, int[6] padding, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::reflection_pad3d_backward.grad_input(Tensor grad_output, Tensor self, int[6] padding, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<reflection_pad3d_backward_grad_input::schema> create_reflection_pad3d_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(reflection_pad3d_backward_grad_input::name, reflection_pad3d_backward_grad_input::overload_name)
      .typed<reflection_pad3d_backward_grad_input::schema>();
}

// aten::reflection_pad3d_backward.grad_input(Tensor grad_output, Tensor self, int[6] padding, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & reflection_pad3d_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input) {
    static auto op = create_reflection_pad3d_backward_grad_input_typed_handle();
    return op.call(grad_output, self, padding, grad_input);
}

// aten::reflection_pad3d_backward.grad_input(Tensor grad_output, Tensor self, int[6] padding, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & reflection_pad3d_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input) {
    static auto op = create_reflection_pad3d_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, padding, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(replication_pad1d_backward, name, "aten::replication_pad1d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(replication_pad1d_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(replication_pad1d_backward, schema_str, "replication_pad1d_backward(Tensor grad_output, Tensor self, int[2] padding) -> Tensor")

// aten::replication_pad1d_backward(Tensor grad_output, Tensor self, int[2] padding) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<replication_pad1d_backward::schema> create_replication_pad1d_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(replication_pad1d_backward::name, replication_pad1d_backward::overload_name)
      .typed<replication_pad1d_backward::schema>();
}

// aten::replication_pad1d_backward(Tensor grad_output, Tensor self, int[2] padding) -> Tensor
at::Tensor replication_pad1d_backward::call(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding) {
    static auto op = create_replication_pad1d_backward_typed_handle();
    return op.call(grad_output, self, padding);
}

// aten::replication_pad1d_backward(Tensor grad_output, Tensor self, int[2] padding) -> Tensor
at::Tensor replication_pad1d_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding) {
    static auto op = create_replication_pad1d_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, padding);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(replication_pad3d_backward_grad_input, name, "aten::replication_pad3d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(replication_pad3d_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(replication_pad3d_backward_grad_input, schema_str, "replication_pad3d_backward.grad_input(Tensor grad_output, Tensor self, int[6] padding, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::replication_pad3d_backward.grad_input(Tensor grad_output, Tensor self, int[6] padding, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<replication_pad3d_backward_grad_input::schema> create_replication_pad3d_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(replication_pad3d_backward_grad_input::name, replication_pad3d_backward_grad_input::overload_name)
      .typed<replication_pad3d_backward_grad_input::schema>();
}

// aten::replication_pad3d_backward.grad_input(Tensor grad_output, Tensor self, int[6] padding, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & replication_pad3d_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input) {
    static auto op = create_replication_pad3d_backward_grad_input_typed_handle();
    return op.call(grad_output, self, padding, grad_input);
}

// aten::replication_pad3d_backward.grad_input(Tensor grad_output, Tensor self, int[6] padding, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & replication_pad3d_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input) {
    static auto op = create_replication_pad3d_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, padding, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(replication_pad3d_backward, name, "aten::replication_pad3d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(replication_pad3d_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(replication_pad3d_backward, schema_str, "replication_pad3d_backward(Tensor grad_output, Tensor self, int[6] padding) -> Tensor")

// aten::replication_pad3d_backward(Tensor grad_output, Tensor self, int[6] padding) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<replication_pad3d_backward::schema> create_replication_pad3d_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(replication_pad3d_backward::name, replication_pad3d_backward::overload_name)
      .typed<replication_pad3d_backward::schema>();
}

// aten::replication_pad3d_backward(Tensor grad_output, Tensor self, int[6] padding) -> Tensor
at::Tensor replication_pad3d_backward::call(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding) {
    static auto op = create_replication_pad3d_backward_typed_handle();
    return op.call(grad_output, self, padding);
}

// aten::replication_pad3d_backward(Tensor grad_output, Tensor self, int[6] padding) -> Tensor
at::Tensor replication_pad3d_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding) {
    static auto op = create_replication_pad3d_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, padding);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_trilinear3d_vec, name, "aten::upsample_trilinear3d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_trilinear3d_vec, overload_name, "vec")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_trilinear3d_vec, schema_str, "upsample_trilinear3d.vec(Tensor input, int[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor")

// aten::upsample_trilinear3d.vec(Tensor input, int[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<upsample_trilinear3d_vec::schema> create_upsample_trilinear3d_vec_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_trilinear3d_vec::name, upsample_trilinear3d_vec::overload_name)
      .typed<upsample_trilinear3d_vec::schema>();
}

// aten::upsample_trilinear3d.vec(Tensor input, int[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
at::Tensor upsample_trilinear3d_vec::call(const at::Tensor & input, c10::optional<at::IntArrayRef> output_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
    static auto op = create_upsample_trilinear3d_vec_typed_handle();
    return op.call(input, output_size, align_corners, scale_factors);
}

// aten::upsample_trilinear3d.vec(Tensor input, int[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
at::Tensor upsample_trilinear3d_vec::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, c10::optional<at::IntArrayRef> output_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
    static auto op = create_upsample_trilinear3d_vec_typed_handle();
    return op.redispatch(dispatchKeySet, input, output_size, align_corners, scale_factors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest2d_backward_vec, name, "aten::upsample_nearest2d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest2d_backward_vec, overload_name, "vec")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest2d_backward_vec, schema_str, "upsample_nearest2d_backward.vec(Tensor grad_output, int[]? output_size, int[] input_size, float[]? scale_factors) -> Tensor")

// aten::upsample_nearest2d_backward.vec(Tensor grad_output, int[]? output_size, int[] input_size, float[]? scale_factors) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<upsample_nearest2d_backward_vec::schema> create_upsample_nearest2d_backward_vec_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_nearest2d_backward_vec::name, upsample_nearest2d_backward_vec::overload_name)
      .typed<upsample_nearest2d_backward_vec::schema>();
}

// aten::upsample_nearest2d_backward.vec(Tensor grad_output, int[]? output_size, int[] input_size, float[]? scale_factors) -> Tensor
at::Tensor upsample_nearest2d_backward_vec::call(const at::Tensor & grad_output, c10::optional<at::IntArrayRef> output_size, at::IntArrayRef input_size, c10::optional<at::ArrayRef<double>> scale_factors) {
    static auto op = create_upsample_nearest2d_backward_vec_typed_handle();
    return op.call(grad_output, output_size, input_size, scale_factors);
}

// aten::upsample_nearest2d_backward.vec(Tensor grad_output, int[]? output_size, int[] input_size, float[]? scale_factors) -> Tensor
at::Tensor upsample_nearest2d_backward_vec::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, c10::optional<at::IntArrayRef> output_size, at::IntArrayRef input_size, c10::optional<at::ArrayRef<double>> scale_factors) {
    static auto op = create_upsample_nearest2d_backward_vec_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, output_size, input_size, scale_factors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest3d_backward_vec, name, "aten::upsample_nearest3d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest3d_backward_vec, overload_name, "vec")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest3d_backward_vec, schema_str, "upsample_nearest3d_backward.vec(Tensor grad_output, int[]? output_size, int[] input_size, float[]? scale_factors) -> Tensor")

// aten::upsample_nearest3d_backward.vec(Tensor grad_output, int[]? output_size, int[] input_size, float[]? scale_factors) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<upsample_nearest3d_backward_vec::schema> create_upsample_nearest3d_backward_vec_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_nearest3d_backward_vec::name, upsample_nearest3d_backward_vec::overload_name)
      .typed<upsample_nearest3d_backward_vec::schema>();
}

// aten::upsample_nearest3d_backward.vec(Tensor grad_output, int[]? output_size, int[] input_size, float[]? scale_factors) -> Tensor
at::Tensor upsample_nearest3d_backward_vec::call(const at::Tensor & grad_output, c10::optional<at::IntArrayRef> output_size, at::IntArrayRef input_size, c10::optional<at::ArrayRef<double>> scale_factors) {
    static auto op = create_upsample_nearest3d_backward_vec_typed_handle();
    return op.call(grad_output, output_size, input_size, scale_factors);
}

// aten::upsample_nearest3d_backward.vec(Tensor grad_output, int[]? output_size, int[] input_size, float[]? scale_factors) -> Tensor
at::Tensor upsample_nearest3d_backward_vec::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, c10::optional<at::IntArrayRef> output_size, at::IntArrayRef input_size, c10::optional<at::ArrayRef<double>> scale_factors) {
    static auto op = create_upsample_nearest3d_backward_vec_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, output_size, input_size, scale_factors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_linear1d_out, name, "aten::upsample_linear1d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_linear1d_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_linear1d_out, schema_str, "upsample_linear1d.out(Tensor self, int[1] output_size, bool align_corners, float? scales=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::upsample_linear1d.out(Tensor self, int[1] output_size, bool align_corners, float? scales=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<upsample_linear1d_out::schema> create_upsample_linear1d_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_linear1d_out::name, upsample_linear1d_out::overload_name)
      .typed<upsample_linear1d_out::schema>();
}

// aten::upsample_linear1d.out(Tensor self, int[1] output_size, bool align_corners, float? scales=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & upsample_linear1d_out::call(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales, at::Tensor & out) {
    static auto op = create_upsample_linear1d_out_typed_handle();
    return op.call(self, output_size, align_corners, scales, out);
}

// aten::upsample_linear1d.out(Tensor self, int[1] output_size, bool align_corners, float? scales=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & upsample_linear1d_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales, at::Tensor & out) {
    static auto op = create_upsample_linear1d_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, output_size, align_corners, scales, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_linear1d_backward_grad_input, name, "aten::upsample_linear1d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_linear1d_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_linear1d_backward_grad_input, schema_str, "upsample_linear1d_backward.grad_input(Tensor grad_output, int[1] output_size, int[3] input_size, bool align_corners, float? scales=None, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::upsample_linear1d_backward.grad_input(Tensor grad_output, int[1] output_size, int[3] input_size, bool align_corners, float? scales=None, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<upsample_linear1d_backward_grad_input::schema> create_upsample_linear1d_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_linear1d_backward_grad_input::name, upsample_linear1d_backward_grad_input::overload_name)
      .typed<upsample_linear1d_backward_grad_input::schema>();
}

// aten::upsample_linear1d_backward.grad_input(Tensor grad_output, int[1] output_size, int[3] input_size, bool align_corners, float? scales=None, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & upsample_linear1d_backward_grad_input::call(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales, at::Tensor & grad_input) {
    static auto op = create_upsample_linear1d_backward_grad_input_typed_handle();
    return op.call(grad_output, output_size, input_size, align_corners, scales, grad_input);
}

// aten::upsample_linear1d_backward.grad_input(Tensor grad_output, int[1] output_size, int[3] input_size, bool align_corners, float? scales=None, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & upsample_linear1d_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales, at::Tensor & grad_input) {
    static auto op = create_upsample_linear1d_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, output_size, input_size, align_corners, scales, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_bilinear2d_backward, name, "aten::upsample_bilinear2d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_bilinear2d_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_bilinear2d_backward, schema_str, "upsample_bilinear2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor")

// aten::upsample_bilinear2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<upsample_bilinear2d_backward::schema> create_upsample_bilinear2d_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_bilinear2d_backward::name, upsample_bilinear2d_backward::overload_name)
      .typed<upsample_bilinear2d_backward::schema>();
}

// aten::upsample_bilinear2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor
at::Tensor upsample_bilinear2d_backward::call(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
    static auto op = create_upsample_bilinear2d_backward_typed_handle();
    return op.call(grad_output, output_size, input_size, align_corners, scales_h, scales_w);
}

// aten::upsample_bilinear2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor
at::Tensor upsample_bilinear2d_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
    static auto op = create_upsample_bilinear2d_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_bicubic2d_out, name, "aten::upsample_bicubic2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_bicubic2d_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_bicubic2d_out, schema_str, "upsample_bicubic2d.out(Tensor self, int[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::upsample_bicubic2d.out(Tensor self, int[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<upsample_bicubic2d_out::schema> create_upsample_bicubic2d_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_bicubic2d_out::name, upsample_bicubic2d_out::overload_name)
      .typed<upsample_bicubic2d_out::schema>();
}

// aten::upsample_bicubic2d.out(Tensor self, int[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & upsample_bicubic2d_out::call(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out) {
    static auto op = create_upsample_bicubic2d_out_typed_handle();
    return op.call(self, output_size, align_corners, scales_h, scales_w, out);
}

// aten::upsample_bicubic2d.out(Tensor self, int[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & upsample_bicubic2d_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out) {
    static auto op = create_upsample_bicubic2d_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, output_size, align_corners, scales_h, scales_w, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sigmoid_backward_grad_input, name, "aten::sigmoid_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sigmoid_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sigmoid_backward_grad_input, schema_str, "sigmoid_backward.grad_input(Tensor grad_output, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::sigmoid_backward.grad_input(Tensor grad_output, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<sigmoid_backward_grad_input::schema> create_sigmoid_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sigmoid_backward_grad_input::name, sigmoid_backward_grad_input::overload_name)
      .typed<sigmoid_backward_grad_input::schema>();
}

// aten::sigmoid_backward.grad_input(Tensor grad_output, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & sigmoid_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & output, at::Tensor & grad_input) {
    static auto op = create_sigmoid_backward_grad_input_typed_handle();
    return op.call(grad_output, output, grad_input);
}

// aten::sigmoid_backward.grad_input(Tensor grad_output, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & sigmoid_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & output, at::Tensor & grad_input) {
    static auto op = create_sigmoid_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, output, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv_transpose3d_out, name, "aten::slow_conv_transpose3d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv_transpose3d_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv_transpose3d_out, schema_str, "slow_conv_transpose3d.out(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] output_padding=0, int[3] dilation=1, *, Tensor(a!) out) -> Tensor(a!)")

// aten::slow_conv_transpose3d.out(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] output_padding=0, int[3] dilation=1, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<slow_conv_transpose3d_out::schema> create_slow_conv_transpose3d_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(slow_conv_transpose3d_out::name, slow_conv_transpose3d_out::overload_name)
      .typed<slow_conv_transpose3d_out::schema>();
}

// aten::slow_conv_transpose3d.out(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] output_padding=0, int[3] dilation=1, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & slow_conv_transpose3d_out::call(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, at::Tensor & out) {
    static auto op = create_slow_conv_transpose3d_out_typed_handle();
    return op.call(self, weight, kernel_size, bias, stride, padding, output_padding, dilation, out);
}

// aten::slow_conv_transpose3d.out(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] output_padding=0, int[3] dilation=1, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & slow_conv_transpose3d_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, at::Tensor & out) {
    static auto op = create_slow_conv_transpose3d_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, weight, kernel_size, bias, stride, padding, output_padding, dilation, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_conv_depthwise2d_backward_grad_input, name, "aten::_conv_depthwise2d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_conv_depthwise2d_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_conv_depthwise2d_backward_grad_input, schema_str, "_conv_depthwise2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, *, Tensor(a!) grad_input, Tensor(b!) grad_weight) -> (Tensor(a!), Tensor(b!))")

// aten::_conv_depthwise2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, *, Tensor(a!) grad_input, Tensor(b!) grad_weight) -> (Tensor(a!), Tensor(b!))
static C10_NOINLINE c10::TypedOperatorHandle<_conv_depthwise2d_backward_grad_input::schema> create__conv_depthwise2d_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_conv_depthwise2d_backward_grad_input::name, _conv_depthwise2d_backward_grad_input::overload_name)
      .typed<_conv_depthwise2d_backward_grad_input::schema>();
}

// aten::_conv_depthwise2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, *, Tensor(a!) grad_input, Tensor(b!) grad_weight) -> (Tensor(a!), Tensor(b!))
::std::tuple<at::Tensor &,at::Tensor &> _conv_depthwise2d_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, at::Tensor & grad_input, at::Tensor & grad_weight) {
    static auto op = create__conv_depthwise2d_backward_grad_input_typed_handle();
    return op.call(grad_output, self, weight, kernel_size, stride, padding, dilation, grad_input, grad_weight);
}

// aten::_conv_depthwise2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, *, Tensor(a!) grad_input, Tensor(b!) grad_weight) -> (Tensor(a!), Tensor(b!))
::std::tuple<at::Tensor &,at::Tensor &> _conv_depthwise2d_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, at::Tensor & grad_input, at::Tensor & grad_weight) {
    static auto op = create__conv_depthwise2d_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, weight, kernel_size, stride, padding, dilation, grad_input, grad_weight);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv_depthwise3d_backward_output_mask, name, "aten::conv_depthwise3d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv_depthwise3d_backward_output_mask, overload_name, "output_mask")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv_depthwise3d_backward_output_mask, schema_str, "conv_depthwise3d_backward.output_mask(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)")

// aten::conv_depthwise3d_backward.output_mask(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)
static C10_NOINLINE c10::TypedOperatorHandle<conv_depthwise3d_backward_output_mask::schema> create_conv_depthwise3d_backward_output_mask_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(conv_depthwise3d_backward_output_mask::name, conv_depthwise3d_backward_output_mask::overload_name)
      .typed<conv_depthwise3d_backward_output_mask::schema>();
}

// aten::conv_depthwise3d_backward.output_mask(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> conv_depthwise3d_backward_output_mask::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, ::std::array<bool,3> output_mask) {
    static auto op = create_conv_depthwise3d_backward_output_mask_typed_handle();
    return op.call(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
}

// aten::conv_depthwise3d_backward.output_mask(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> conv_depthwise3d_backward_output_mask::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, ::std::array<bool,3> output_mask) {
    static auto op = create_conv_depthwise3d_backward_output_mask_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv3d_backward_grad_input, name, "aten::slow_conv3d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv3d_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv3d_backward_grad_input, schema_str, "slow_conv3d_backward.grad_input(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, Tensor finput, Tensor fgrad_input, *, Tensor(a!) grad_input, Tensor(b!) grad_weight, Tensor(c!) grad_bias) -> (Tensor(a!), Tensor(b!), Tensor(c!))")

// aten::slow_conv3d_backward.grad_input(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, Tensor finput, Tensor fgrad_input, *, Tensor(a!) grad_input, Tensor(b!) grad_weight, Tensor(c!) grad_bias) -> (Tensor(a!), Tensor(b!), Tensor(c!))
static C10_NOINLINE c10::TypedOperatorHandle<slow_conv3d_backward_grad_input::schema> create_slow_conv3d_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(slow_conv3d_backward_grad_input::name, slow_conv3d_backward_grad_input::overload_name)
      .typed<slow_conv3d_backward_grad_input::schema>();
}

// aten::slow_conv3d_backward.grad_input(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, Tensor finput, Tensor fgrad_input, *, Tensor(a!) grad_input, Tensor(b!) grad_weight, Tensor(c!) grad_bias) -> (Tensor(a!), Tensor(b!), Tensor(c!))
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> slow_conv3d_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, const at::Tensor & finput, const at::Tensor & fgrad_input, at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias) {
    static auto op = create_slow_conv3d_backward_grad_input_typed_handle();
    return op.call(grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, grad_input, grad_weight, grad_bias);
}

// aten::slow_conv3d_backward.grad_input(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, Tensor finput, Tensor fgrad_input, *, Tensor(a!) grad_input, Tensor(b!) grad_weight, Tensor(c!) grad_bias) -> (Tensor(a!), Tensor(b!), Tensor(c!))
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> slow_conv3d_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, const at::Tensor & finput, const at::Tensor & fgrad_input, at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias) {
    static auto op = create_slow_conv3d_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, grad_input, grad_weight, grad_bias);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv_dilated3d, name, "aten::slow_conv_dilated3d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv_dilated3d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv_dilated3d, schema_str, "slow_conv_dilated3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] dilation=1) -> Tensor")

// aten::slow_conv_dilated3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] dilation=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<slow_conv_dilated3d::schema> create_slow_conv_dilated3d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(slow_conv_dilated3d::name, slow_conv_dilated3d::overload_name)
      .typed<slow_conv_dilated3d::schema>();
}

// aten::slow_conv_dilated3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] dilation=1) -> Tensor
at::Tensor slow_conv_dilated3d::call(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation) {
    static auto op = create_slow_conv_dilated3d_typed_handle();
    return op.call(self, weight, kernel_size, bias, stride, padding, dilation);
}

// aten::slow_conv_dilated3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] dilation=1) -> Tensor
at::Tensor slow_conv_dilated3d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation) {
    static auto op = create_slow_conv_dilated3d_typed_handle();
    return op.redispatch(dispatchKeySet, self, weight, kernel_size, bias, stride, padding, dilation);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv_dilated3d_backward, name, "aten::slow_conv_dilated3d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv_dilated3d_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv_dilated3d_backward, schema_str, "slow_conv_dilated3d_backward(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)")

// aten::slow_conv_dilated3d_backward(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)
static C10_NOINLINE c10::TypedOperatorHandle<slow_conv_dilated3d_backward::schema> create_slow_conv_dilated3d_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(slow_conv_dilated3d_backward::name, slow_conv_dilated3d_backward::overload_name)
      .typed<slow_conv_dilated3d_backward::schema>();
}

// aten::slow_conv_dilated3d_backward(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> slow_conv_dilated3d_backward::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, ::std::array<bool,3> output_mask) {
    static auto op = create_slow_conv_dilated3d_backward_typed_handle();
    return op.call(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
}

// aten::slow_conv_dilated3d_backward(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> slow_conv_dilated3d_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, ::std::array<bool,3> output_mask) {
    static auto op = create_slow_conv_dilated3d_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(col2im_out, name, "aten::col2im")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(col2im_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(col2im_out, schema_str, "col2im.out(Tensor self, int[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) out) -> Tensor(a!)")

// aten::col2im.out(Tensor self, int[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<col2im_out::schema> create_col2im_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(col2im_out::name, col2im_out::overload_name)
      .typed<col2im_out::schema>();
}

// aten::col2im.out(Tensor self, int[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & col2im_out::call(const at::Tensor & self, at::IntArrayRef output_size, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride, at::Tensor & out) {
    static auto op = create_col2im_out_typed_handle();
    return op.call(self, output_size, kernel_size, dilation, padding, stride, out);
}

// aten::col2im.out(Tensor self, int[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & col2im_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride, at::Tensor & out) {
    static auto op = create_col2im_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, output_size, kernel_size, dilation, padding, stride, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(col2im_backward, name, "aten::col2im_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(col2im_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(col2im_backward, schema_str, "col2im_backward(Tensor grad_output, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor")

// aten::col2im_backward(Tensor grad_output, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<col2im_backward::schema> create_col2im_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(col2im_backward::name, col2im_backward::overload_name)
      .typed<col2im_backward::schema>();
}

// aten::col2im_backward(Tensor grad_output, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor
at::Tensor col2im_backward::call(const at::Tensor & grad_output, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride) {
    static auto op = create_col2im_backward_typed_handle();
    return op.call(grad_output, kernel_size, dilation, padding, stride);
}

// aten::col2im_backward(Tensor grad_output, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor
at::Tensor col2im_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride) {
    static auto op = create_col2im_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, kernel_size, dilation, padding, stride);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(im2col_out, name, "aten::im2col")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(im2col_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(im2col_out, schema_str, "im2col.out(Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) out) -> Tensor(a!)")

// aten::im2col.out(Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<im2col_out::schema> create_im2col_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(im2col_out::name, im2col_out::overload_name)
      .typed<im2col_out::schema>();
}

// aten::im2col.out(Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & im2col_out::call(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride, at::Tensor & out) {
    static auto op = create_im2col_out_typed_handle();
    return op.call(self, kernel_size, dilation, padding, stride, out);
}

// aten::im2col.out(Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & im2col_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride, at::Tensor & out) {
    static auto op = create_im2col_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, kernel_size, dilation, padding, stride, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isinf, name, "aten::isinf")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isinf, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isinf, schema_str, "isinf(Tensor self) -> Tensor")

// aten::isinf(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<isinf::schema> create_isinf_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(isinf::name, isinf::overload_name)
      .typed<isinf::schema>();
}

// aten::isinf(Tensor self) -> Tensor
at::Tensor isinf::call(const at::Tensor & self) {
    static auto op = create_isinf_typed_handle();
    return op.call(self);
}

// aten::isinf(Tensor self) -> Tensor
at::Tensor isinf::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_isinf_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isneginf_out, name, "aten::isneginf")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isneginf_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isneginf_out, schema_str, "isneginf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::isneginf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<isneginf_out::schema> create_isneginf_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(isneginf_out::name, isneginf_out::overload_name)
      .typed<isneginf_out::schema>();
}

// aten::isneginf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & isneginf_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_isneginf_out_typed_handle();
    return op.call(self, out);
}

// aten::isneginf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & isneginf_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_isneginf_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_entr_out, name, "aten::special_entr")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_entr_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_entr_out, schema_str, "special_entr.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::special_entr.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<special_entr_out::schema> create_special_entr_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_entr_out::name, special_entr_out::overload_name)
      .typed<special_entr_out::schema>();
}

// aten::special_entr.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_entr_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_entr_out_typed_handle();
    return op.call(self, out);
}

// aten::special_entr.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_entr_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_entr_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_expm1_out, name, "aten::special_expm1")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_expm1_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_expm1_out, schema_str, "special_expm1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::special_expm1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<special_expm1_out::schema> create_special_expm1_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_expm1_out::name, special_expm1_out::overload_name)
      .typed<special_expm1_out::schema>();
}

// aten::special_expm1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_expm1_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_expm1_out_typed_handle();
    return op.call(self, out);
}

// aten::special_expm1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_expm1_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_expm1_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_psi_out, name, "aten::special_psi")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_psi_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_psi_out, schema_str, "special_psi.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::special_psi.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<special_psi_out::schema> create_special_psi_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_psi_out::name, special_psi_out::overload_name)
      .typed<special_psi_out::schema>();
}

// aten::special_psi.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_psi_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_psi_out_typed_handle();
    return op.call(self, out);
}

// aten::special_psi.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_psi_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_psi_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_digamma, name, "aten::special_digamma")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_digamma, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_digamma, schema_str, "special_digamma(Tensor self) -> Tensor")

// aten::special_digamma(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<special_digamma::schema> create_special_digamma_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_digamma::name, special_digamma::overload_name)
      .typed<special_digamma::schema>();
}

// aten::special_digamma(Tensor self) -> Tensor
at::Tensor special_digamma::call(const at::Tensor & self) {
    static auto op = create_special_digamma_typed_handle();
    return op.call(self);
}

// aten::special_digamma(Tensor self) -> Tensor
at::Tensor special_digamma::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_special_digamma_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_ndtr, name, "aten::special_ndtr")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_ndtr, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_ndtr, schema_str, "special_ndtr(Tensor self) -> Tensor")

// aten::special_ndtr(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<special_ndtr::schema> create_special_ndtr_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_ndtr::name, special_ndtr::overload_name)
      .typed<special_ndtr::schema>();
}

// aten::special_ndtr(Tensor self) -> Tensor
at::Tensor special_ndtr::call(const at::Tensor & self) {
    static auto op = create_special_ndtr_typed_handle();
    return op.call(self);
}

// aten::special_ndtr(Tensor self) -> Tensor
at::Tensor special_ndtr::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_special_ndtr_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_xlog1py_other_scalar, name, "aten::special_xlog1py")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_xlog1py_other_scalar, overload_name, "other_scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_xlog1py_other_scalar, schema_str, "special_xlog1py.other_scalar(Tensor self, Scalar other) -> Tensor")

// aten::special_xlog1py.other_scalar(Tensor self, Scalar other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<special_xlog1py_other_scalar::schema> create_special_xlog1py_other_scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_xlog1py_other_scalar::name, special_xlog1py_other_scalar::overload_name)
      .typed<special_xlog1py_other_scalar::schema>();
}

// aten::special_xlog1py.other_scalar(Tensor self, Scalar other) -> Tensor
at::Tensor special_xlog1py_other_scalar::call(const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_special_xlog1py_other_scalar_typed_handle();
    return op.call(self, other);
}

// aten::special_xlog1py.other_scalar(Tensor self, Scalar other) -> Tensor
at::Tensor special_xlog1py_other_scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_special_xlog1py_other_scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_xlog1py_other_scalar_out, name, "aten::special_xlog1py")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_xlog1py_other_scalar_out, overload_name, "other_scalar_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_xlog1py_other_scalar_out, schema_str, "special_xlog1py.other_scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::special_xlog1py.other_scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<special_xlog1py_other_scalar_out::schema> create_special_xlog1py_other_scalar_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_xlog1py_other_scalar_out::name, special_xlog1py_other_scalar_out::overload_name)
      .typed<special_xlog1py_other_scalar_out::schema>();
}

// aten::special_xlog1py.other_scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_xlog1py_other_scalar_out::call(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_special_xlog1py_other_scalar_out_typed_handle();
    return op.call(self, other, out);
}

// aten::special_xlog1py.other_scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_xlog1py_other_scalar_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_special_xlog1py_other_scalar_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_xlogy_self_scalar, name, "aten::special_xlogy")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_xlogy_self_scalar, overload_name, "self_scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_xlogy_self_scalar, schema_str, "special_xlogy.self_scalar(Scalar self, Tensor other) -> Tensor")

// aten::special_xlogy.self_scalar(Scalar self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<special_xlogy_self_scalar::schema> create_special_xlogy_self_scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_xlogy_self_scalar::name, special_xlogy_self_scalar::overload_name)
      .typed<special_xlogy_self_scalar::schema>();
}

// aten::special_xlogy.self_scalar(Scalar self, Tensor other) -> Tensor
at::Tensor special_xlogy_self_scalar::call(const at::Scalar & self, const at::Tensor & other) {
    static auto op = create_special_xlogy_self_scalar_typed_handle();
    return op.call(self, other);
}

// aten::special_xlogy.self_scalar(Scalar self, Tensor other) -> Tensor
at::Tensor special_xlogy_self_scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Scalar & self, const at::Tensor & other) {
    static auto op = create_special_xlogy_self_scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_xlogy_out, name, "aten::special_xlogy")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_xlogy_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_xlogy_out, schema_str, "special_xlogy.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::special_xlogy.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<special_xlogy_out::schema> create_special_xlogy_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_xlogy_out::name, special_xlogy_out::overload_name)
      .typed<special_xlogy_out::schema>();
}

// aten::special_xlogy.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_xlogy_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_special_xlogy_out_typed_handle();
    return op.call(self, other, out);
}

// aten::special_xlogy.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_xlogy_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_special_xlogy_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_zeta, name, "aten::special_zeta")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_zeta, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_zeta, schema_str, "special_zeta(Tensor self, Tensor other) -> Tensor")

// aten::special_zeta(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<special_zeta::schema> create_special_zeta_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_zeta::name, special_zeta::overload_name)
      .typed<special_zeta::schema>();
}

// aten::special_zeta(Tensor self, Tensor other) -> Tensor
at::Tensor special_zeta::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_special_zeta_typed_handle();
    return op.call(self, other);
}

// aten::special_zeta(Tensor self, Tensor other) -> Tensor
at::Tensor special_zeta::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_special_zeta_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_zeta_self_scalar, name, "aten::special_zeta")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_zeta_self_scalar, overload_name, "self_scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_zeta_self_scalar, schema_str, "special_zeta.self_scalar(Scalar self, Tensor other) -> Tensor")

// aten::special_zeta.self_scalar(Scalar self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<special_zeta_self_scalar::schema> create_special_zeta_self_scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_zeta_self_scalar::name, special_zeta_self_scalar::overload_name)
      .typed<special_zeta_self_scalar::schema>();
}

// aten::special_zeta.self_scalar(Scalar self, Tensor other) -> Tensor
at::Tensor special_zeta_self_scalar::call(const at::Scalar & self, const at::Tensor & other) {
    static auto op = create_special_zeta_self_scalar_typed_handle();
    return op.call(self, other);
}

// aten::special_zeta.self_scalar(Scalar self, Tensor other) -> Tensor
at::Tensor special_zeta_self_scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Scalar & self, const at::Tensor & other) {
    static auto op = create_special_zeta_self_scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_zeta_other_scalar, name, "aten::special_zeta")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_zeta_other_scalar, overload_name, "other_scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_zeta_other_scalar, schema_str, "special_zeta.other_scalar(Tensor self, Scalar other) -> Tensor")

// aten::special_zeta.other_scalar(Tensor self, Scalar other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<special_zeta_other_scalar::schema> create_special_zeta_other_scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_zeta_other_scalar::name, special_zeta_other_scalar::overload_name)
      .typed<special_zeta_other_scalar::schema>();
}

// aten::special_zeta.other_scalar(Tensor self, Scalar other) -> Tensor
at::Tensor special_zeta_other_scalar::call(const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_special_zeta_other_scalar_typed_handle();
    return op.call(self, other);
}

// aten::special_zeta.other_scalar(Tensor self, Scalar other) -> Tensor
at::Tensor special_zeta_other_scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_special_zeta_other_scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_zeta_out, name, "aten::special_zeta")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_zeta_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_zeta_out, schema_str, "special_zeta.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::special_zeta.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<special_zeta_out::schema> create_special_zeta_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_zeta_out::name, special_zeta_out::overload_name)
      .typed<special_zeta_out::schema>();
}

// aten::special_zeta.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_zeta_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_special_zeta_out_typed_handle();
    return op.call(self, other, out);
}

// aten::special_zeta.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_zeta_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_special_zeta_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_i0_out, name, "aten::special_i0")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_i0_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_i0_out, schema_str, "special_i0.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::special_i0.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<special_i0_out::schema> create_special_i0_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_i0_out::name, special_i0_out::overload_name)
      .typed<special_i0_out::schema>();
}

// aten::special_i0.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_i0_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_i0_out_typed_handle();
    return op.call(self, out);
}

// aten::special_i0.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_i0_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_i0_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_i1_out, name, "aten::special_i1")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_i1_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_i1_out, schema_str, "special_i1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::special_i1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<special_i1_out::schema> create_special_i1_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_i1_out::name, special_i1_out::overload_name)
      .typed<special_i1_out::schema>();
}

// aten::special_i1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_i1_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_i1_out_typed_handle();
    return op.call(self, out);
}

// aten::special_i1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_i1_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_i1_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_logit_out, name, "aten::special_logit")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_logit_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_logit_out, schema_str, "special_logit.out(Tensor self, float? eps=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::special_logit.out(Tensor self, float? eps=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<special_logit_out::schema> create_special_logit_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_logit_out::name, special_logit_out::overload_name)
      .typed<special_logit_out::schema>();
}

// aten::special_logit.out(Tensor self, float? eps=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_logit_out::call(const at::Tensor & self, c10::optional<double> eps, at::Tensor & out) {
    static auto op = create_special_logit_out_typed_handle();
    return op.call(self, eps, out);
}

// aten::special_logit.out(Tensor self, float? eps=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_logit_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<double> eps, at::Tensor & out) {
    static auto op = create_special_logit_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, eps, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_sinc_out, name, "aten::special_sinc")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_sinc_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_sinc_out, schema_str, "special_sinc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::special_sinc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<special_sinc_out::schema> create_special_sinc_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_sinc_out::name, special_sinc_out::overload_name)
      .typed<special_sinc_out::schema>();
}

// aten::special_sinc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_sinc_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_sinc_out_typed_handle();
    return op.call(self, out);
}

// aten::special_sinc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_sinc_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_sinc_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_round, name, "aten::special_round")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_round, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_round, schema_str, "special_round(Tensor self) -> Tensor")

// aten::special_round(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<special_round::schema> create_special_round_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_round::name, special_round::overload_name)
      .typed<special_round::schema>();
}

// aten::special_round(Tensor self) -> Tensor
at::Tensor special_round::call(const at::Tensor & self) {
    static auto op = create_special_round_typed_handle();
    return op.call(self);
}

// aten::special_round(Tensor self) -> Tensor
at::Tensor special_round::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_special_round_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_ifft, name, "aten::fft_ifft")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_ifft, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_ifft, schema_str, "fft_ifft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> Tensor")

// aten::fft_ifft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fft_ifft::schema> create_fft_ifft_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fft_ifft::name, fft_ifft::overload_name)
      .typed<fft_ifft::schema>();
}

// aten::fft_ifft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> Tensor
at::Tensor fft_ifft::call(const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm) {
    static auto op = create_fft_ifft_typed_handle();
    return op.call(self, n, dim, norm);
}

// aten::fft_ifft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> Tensor
at::Tensor fft_ifft::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm) {
    static auto op = create_fft_ifft_typed_handle();
    return op.redispatch(dispatchKeySet, self, n, dim, norm);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_ifft_out, name, "aten::fft_ifft")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_ifft_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_ifft_out, schema_str, "fft_ifft.out(Tensor self, int? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::fft_ifft.out(Tensor self, int? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<fft_ifft_out::schema> create_fft_ifft_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fft_ifft_out::name, fft_ifft_out::overload_name)
      .typed<fft_ifft_out::schema>();
}

// aten::fft_ifft.out(Tensor self, int? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fft_ifft_out::call(const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
    static auto op = create_fft_ifft_out_typed_handle();
    return op.call(self, n, dim, norm, out);
}

// aten::fft_ifft.out(Tensor self, int? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fft_ifft_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
    static auto op = create_fft_ifft_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, n, dim, norm, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_hfft, name, "aten::fft_hfft")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_hfft, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_hfft, schema_str, "fft_hfft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> Tensor")

// aten::fft_hfft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fft_hfft::schema> create_fft_hfft_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fft_hfft::name, fft_hfft::overload_name)
      .typed<fft_hfft::schema>();
}

// aten::fft_hfft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> Tensor
at::Tensor fft_hfft::call(const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm) {
    static auto op = create_fft_hfft_typed_handle();
    return op.call(self, n, dim, norm);
}

// aten::fft_hfft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> Tensor
at::Tensor fft_hfft::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm) {
    static auto op = create_fft_hfft_typed_handle();
    return op.redispatch(dispatchKeySet, self, n, dim, norm);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_ihfft, name, "aten::fft_ihfft")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_ihfft, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_ihfft, schema_str, "fft_ihfft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> Tensor")

// aten::fft_ihfft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fft_ihfft::schema> create_fft_ihfft_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fft_ihfft::name, fft_ihfft::overload_name)
      .typed<fft_ihfft::schema>();
}

// aten::fft_ihfft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> Tensor
at::Tensor fft_ihfft::call(const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm) {
    static auto op = create_fft_ihfft_typed_handle();
    return op.call(self, n, dim, norm);
}

// aten::fft_ihfft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> Tensor
at::Tensor fft_ihfft::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm) {
    static auto op = create_fft_ihfft_typed_handle();
    return op.redispatch(dispatchKeySet, self, n, dim, norm);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_irfft2_out, name, "aten::fft_irfft2")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_irfft2_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_irfft2_out, schema_str, "fft_irfft2.out(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::fft_irfft2.out(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<fft_irfft2_out::schema> create_fft_irfft2_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fft_irfft2_out::name, fft_irfft2_out::overload_name)
      .typed<fft_irfft2_out::schema>();
}

// aten::fft_irfft2.out(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fft_irfft2_out::call(const at::Tensor & self, c10::optional<at::IntArrayRef> s, at::IntArrayRef dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
    static auto op = create_fft_irfft2_out_typed_handle();
    return op.call(self, s, dim, norm, out);
}

// aten::fft_irfft2.out(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fft_irfft2_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> s, at::IntArrayRef dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
    static auto op = create_fft_irfft2_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, s, dim, norm, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_rfftn_out, name, "aten::fft_rfftn")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_rfftn_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_rfftn_out, schema_str, "fft_rfftn.out(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::fft_rfftn.out(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<fft_rfftn_out::schema> create_fft_rfftn_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fft_rfftn_out::name, fft_rfftn_out::overload_name)
      .typed<fft_rfftn_out::schema>();
}

// aten::fft_rfftn.out(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fft_rfftn_out::call(const at::Tensor & self, c10::optional<at::IntArrayRef> s, c10::optional<at::IntArrayRef> dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
    static auto op = create_fft_rfftn_out_typed_handle();
    return op.call(self, s, dim, norm, out);
}

// aten::fft_rfftn.out(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fft_rfftn_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> s, c10::optional<at::IntArrayRef> dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
    static auto op = create_fft_rfftn_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, s, dim, norm, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_irfftn, name, "aten::fft_irfftn")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_irfftn, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_irfftn, schema_str, "fft_irfftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor")

// aten::fft_irfftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fft_irfftn::schema> create_fft_irfftn_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fft_irfftn::name, fft_irfftn::overload_name)
      .typed<fft_irfftn::schema>();
}

// aten::fft_irfftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor
at::Tensor fft_irfftn::call(const at::Tensor & self, c10::optional<at::IntArrayRef> s, c10::optional<at::IntArrayRef> dim, c10::optional<c10::string_view> norm) {
    static auto op = create_fft_irfftn_typed_handle();
    return op.call(self, s, dim, norm);
}

// aten::fft_irfftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor
at::Tensor fft_irfftn::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> s, c10::optional<at::IntArrayRef> dim, c10::optional<c10::string_view> norm) {
    static auto op = create_fft_irfftn_typed_handle();
    return op.redispatch(dispatchKeySet, self, s, dim, norm);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_fftfreq_out, name, "aten::fft_fftfreq")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_fftfreq_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_fftfreq_out, schema_str, "fft_fftfreq.out(int n, float d=1.0, *, Tensor(a!) out) -> Tensor(a!)")

// aten::fft_fftfreq.out(int n, float d=1.0, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<fft_fftfreq_out::schema> create_fft_fftfreq_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fft_fftfreq_out::name, fft_fftfreq_out::overload_name)
      .typed<fft_fftfreq_out::schema>();
}

// aten::fft_fftfreq.out(int n, float d=1.0, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fft_fftfreq_out::call(int64_t n, double d, at::Tensor & out) {
    static auto op = create_fft_fftfreq_out_typed_handle();
    return op.call(n, d, out);
}

// aten::fft_fftfreq.out(int n, float d=1.0, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fft_fftfreq_out::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t n, double d, at::Tensor & out) {
    static auto op = create_fft_fftfreq_out_typed_handle();
    return op.redispatch(dispatchKeySet, n, d, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_rfftfreq_out, name, "aten::fft_rfftfreq")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_rfftfreq_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_rfftfreq_out, schema_str, "fft_rfftfreq.out(int n, float d=1.0, *, Tensor(a!) out) -> Tensor(a!)")

// aten::fft_rfftfreq.out(int n, float d=1.0, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<fft_rfftfreq_out::schema> create_fft_rfftfreq_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fft_rfftfreq_out::name, fft_rfftfreq_out::overload_name)
      .typed<fft_rfftfreq_out::schema>();
}

// aten::fft_rfftfreq.out(int n, float d=1.0, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fft_rfftfreq_out::call(int64_t n, double d, at::Tensor & out) {
    static auto op = create_fft_rfftfreq_out_typed_handle();
    return op.call(n, d, out);
}

// aten::fft_rfftfreq.out(int n, float d=1.0, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fft_rfftfreq_out::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t n, double d, at::Tensor & out) {
    static auto op = create_fft_rfftfreq_out_typed_handle();
    return op.redispatch(dispatchKeySet, n, d, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_ifftshift, name, "aten::fft_ifftshift")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_ifftshift, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_ifftshift, schema_str, "fft_ifftshift(Tensor self, int[1]? dim=None) -> Tensor")

// aten::fft_ifftshift(Tensor self, int[1]? dim=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fft_ifftshift::schema> create_fft_ifftshift_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fft_ifftshift::name, fft_ifftshift::overload_name)
      .typed<fft_ifftshift::schema>();
}

// aten::fft_ifftshift(Tensor self, int[1]? dim=None) -> Tensor
at::Tensor fft_ifftshift::call(const at::Tensor & self, c10::optional<at::IntArrayRef> dim) {
    static auto op = create_fft_ifftshift_typed_handle();
    return op.call(self, dim);
}

// aten::fft_ifftshift(Tensor self, int[1]? dim=None) -> Tensor
at::Tensor fft_ifftshift::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> dim) {
    static auto op = create_fft_ifftshift_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_det_lu_based_helper, name, "aten::_det_lu_based_helper")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_det_lu_based_helper, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_det_lu_based_helper, schema_str, "_det_lu_based_helper(Tensor self) -> (Tensor det, Tensor lu, Tensor pivs)")

// aten::_det_lu_based_helper(Tensor self) -> (Tensor det, Tensor lu, Tensor pivs)
static C10_NOINLINE c10::TypedOperatorHandle<_det_lu_based_helper::schema> create__det_lu_based_helper_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_det_lu_based_helper::name, _det_lu_based_helper::overload_name)
      .typed<_det_lu_based_helper::schema>();
}

// aten::_det_lu_based_helper(Tensor self) -> (Tensor det, Tensor lu, Tensor pivs)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _det_lu_based_helper::call(const at::Tensor & self) {
    static auto op = create__det_lu_based_helper_typed_handle();
    return op.call(self);
}

// aten::_det_lu_based_helper(Tensor self) -> (Tensor det, Tensor lu, Tensor pivs)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _det_lu_based_helper::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create__det_lu_based_helper_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_det_lu_based_helper_backward_helper, name, "aten::_det_lu_based_helper_backward_helper")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_det_lu_based_helper_backward_helper, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_det_lu_based_helper_backward_helper, schema_str, "_det_lu_based_helper_backward_helper(Tensor det_grad, Tensor det, Tensor self, Tensor lu, Tensor pivs) -> Tensor")

// aten::_det_lu_based_helper_backward_helper(Tensor det_grad, Tensor det, Tensor self, Tensor lu, Tensor pivs) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_det_lu_based_helper_backward_helper::schema> create__det_lu_based_helper_backward_helper_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_det_lu_based_helper_backward_helper::name, _det_lu_based_helper_backward_helper::overload_name)
      .typed<_det_lu_based_helper_backward_helper::schema>();
}

// aten::_det_lu_based_helper_backward_helper(Tensor det_grad, Tensor det, Tensor self, Tensor lu, Tensor pivs) -> Tensor
at::Tensor _det_lu_based_helper_backward_helper::call(const at::Tensor & det_grad, const at::Tensor & det, const at::Tensor & self, const at::Tensor & lu, const at::Tensor & pivs) {
    static auto op = create__det_lu_based_helper_backward_helper_typed_handle();
    return op.call(det_grad, det, self, lu, pivs);
}

// aten::_det_lu_based_helper_backward_helper(Tensor det_grad, Tensor det, Tensor self, Tensor lu, Tensor pivs) -> Tensor
at::Tensor _det_lu_based_helper_backward_helper::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & det_grad, const at::Tensor & det, const at::Tensor & self, const at::Tensor & lu, const at::Tensor & pivs) {
    static auto op = create__det_lu_based_helper_backward_helper_typed_handle();
    return op.redispatch(dispatchKeySet, det_grad, det, self, lu, pivs);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_eig, name, "aten::linalg_eig")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_eig, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_eig, schema_str, "linalg_eig(Tensor self) -> (Tensor eigenvalues, Tensor eigenvectors)")

// aten::linalg_eig(Tensor self) -> (Tensor eigenvalues, Tensor eigenvectors)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_eig::schema> create_linalg_eig_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_eig::name, linalg_eig::overload_name)
      .typed<linalg_eig::schema>();
}

// aten::linalg_eig(Tensor self) -> (Tensor eigenvalues, Tensor eigenvectors)
::std::tuple<at::Tensor,at::Tensor> linalg_eig::call(const at::Tensor & self) {
    static auto op = create_linalg_eig_typed_handle();
    return op.call(self);
}

// aten::linalg_eig(Tensor self) -> (Tensor eigenvalues, Tensor eigenvectors)
::std::tuple<at::Tensor,at::Tensor> linalg_eig::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_linalg_eig_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_eig_out, name, "aten::linalg_eig")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_eig_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_eig_out, schema_str, "linalg_eig.out(Tensor self, *, Tensor(a!) eigenvalues, Tensor(b!) eigenvectors) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)")

// aten::linalg_eig.out(Tensor self, *, Tensor(a!) eigenvalues, Tensor(b!) eigenvectors) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_eig_out::schema> create_linalg_eig_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_eig_out::name, linalg_eig_out::overload_name)
      .typed<linalg_eig_out::schema>();
}

// aten::linalg_eig.out(Tensor self, *, Tensor(a!) eigenvalues, Tensor(b!) eigenvectors) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)
::std::tuple<at::Tensor &,at::Tensor &> linalg_eig_out::call(const at::Tensor & self, at::Tensor & eigenvalues, at::Tensor & eigenvectors) {
    static auto op = create_linalg_eig_out_typed_handle();
    return op.call(self, eigenvalues, eigenvectors);
}

// aten::linalg_eig.out(Tensor self, *, Tensor(a!) eigenvalues, Tensor(b!) eigenvectors) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)
::std::tuple<at::Tensor &,at::Tensor &> linalg_eig_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & eigenvalues, at::Tensor & eigenvectors) {
    static auto op = create_linalg_eig_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, eigenvalues, eigenvectors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_eigh, name, "aten::linalg_eigh")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_eigh, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_eigh, schema_str, "linalg_eigh(Tensor self, str UPLO=\"L\") -> (Tensor eigenvalues, Tensor eigenvectors)")

// aten::linalg_eigh(Tensor self, str UPLO="L") -> (Tensor eigenvalues, Tensor eigenvectors)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_eigh::schema> create_linalg_eigh_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_eigh::name, linalg_eigh::overload_name)
      .typed<linalg_eigh::schema>();
}

// aten::linalg_eigh(Tensor self, str UPLO="L") -> (Tensor eigenvalues, Tensor eigenvectors)
::std::tuple<at::Tensor,at::Tensor> linalg_eigh::call(const at::Tensor & self, c10::string_view UPLO) {
    static auto op = create_linalg_eigh_typed_handle();
    return op.call(self, UPLO);
}

// aten::linalg_eigh(Tensor self, str UPLO="L") -> (Tensor eigenvalues, Tensor eigenvectors)
::std::tuple<at::Tensor,at::Tensor> linalg_eigh::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::string_view UPLO) {
    static auto op = create_linalg_eigh_typed_handle();
    return op.redispatch(dispatchKeySet, self, UPLO);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_eigvalsh, name, "aten::linalg_eigvalsh")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_eigvalsh, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_eigvalsh, schema_str, "linalg_eigvalsh(Tensor self, str UPLO=\"L\") -> Tensor")

// aten::linalg_eigvalsh(Tensor self, str UPLO="L") -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<linalg_eigvalsh::schema> create_linalg_eigvalsh_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_eigvalsh::name, linalg_eigvalsh::overload_name)
      .typed<linalg_eigvalsh::schema>();
}

// aten::linalg_eigvalsh(Tensor self, str UPLO="L") -> Tensor
at::Tensor linalg_eigvalsh::call(const at::Tensor & self, c10::string_view UPLO) {
    static auto op = create_linalg_eigvalsh_typed_handle();
    return op.call(self, UPLO);
}

// aten::linalg_eigvalsh(Tensor self, str UPLO="L") -> Tensor
at::Tensor linalg_eigvalsh::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::string_view UPLO) {
    static auto op = create_linalg_eigvalsh_typed_handle();
    return op.redispatch(dispatchKeySet, self, UPLO);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_eigvalsh_out, name, "aten::linalg_eigvalsh")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_eigvalsh_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_eigvalsh_out, schema_str, "linalg_eigvalsh.out(Tensor self, str UPLO='L', *, Tensor(a!) out) -> Tensor(a!)")

// aten::linalg_eigvalsh.out(Tensor self, str UPLO='L', *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_eigvalsh_out::schema> create_linalg_eigvalsh_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_eigvalsh_out::name, linalg_eigvalsh_out::overload_name)
      .typed<linalg_eigvalsh_out::schema>();
}

// aten::linalg_eigvalsh.out(Tensor self, str UPLO='L', *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_eigvalsh_out::call(const at::Tensor & self, c10::string_view UPLO, at::Tensor & out) {
    static auto op = create_linalg_eigvalsh_out_typed_handle();
    return op.call(self, UPLO, out);
}

// aten::linalg_eigvalsh.out(Tensor self, str UPLO='L', *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_eigvalsh_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::string_view UPLO, at::Tensor & out) {
    static auto op = create_linalg_eigvalsh_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, UPLO, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_householder_product, name, "aten::linalg_householder_product")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_householder_product, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_householder_product, schema_str, "linalg_householder_product(Tensor input, Tensor tau) -> Tensor")

// aten::linalg_householder_product(Tensor input, Tensor tau) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<linalg_householder_product::schema> create_linalg_householder_product_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_householder_product::name, linalg_householder_product::overload_name)
      .typed<linalg_householder_product::schema>();
}

// aten::linalg_householder_product(Tensor input, Tensor tau) -> Tensor
at::Tensor linalg_householder_product::call(const at::Tensor & input, const at::Tensor & tau) {
    static auto op = create_linalg_householder_product_typed_handle();
    return op.call(input, tau);
}

// aten::linalg_householder_product(Tensor input, Tensor tau) -> Tensor
at::Tensor linalg_householder_product::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & tau) {
    static auto op = create_linalg_householder_product_typed_handle();
    return op.redispatch(dispatchKeySet, input, tau);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(inner_out, name, "aten::inner")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(inner_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(inner_out, schema_str, "inner.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::inner.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<inner_out::schema> create_inner_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(inner_out::name, inner_out::overload_name)
      .typed<inner_out::schema>();
}

// aten::inner.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & inner_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_inner_out_typed_handle();
    return op.call(self, other, out);
}

// aten::inner.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & inner_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_inner_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_matrix_norm, name, "aten::linalg_matrix_norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_matrix_norm, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_matrix_norm, schema_str, "linalg_matrix_norm(Tensor self, Scalar ord, int[] dim=[-2,-1], bool keepdim=False, *, ScalarType? dtype=None) -> Tensor")

// aten::linalg_matrix_norm(Tensor self, Scalar ord, int[] dim=[-2,-1], bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<linalg_matrix_norm::schema> create_linalg_matrix_norm_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_matrix_norm::name, linalg_matrix_norm::overload_name)
      .typed<linalg_matrix_norm::schema>();
}

// aten::linalg_matrix_norm(Tensor self, Scalar ord, int[] dim=[-2,-1], bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
at::Tensor linalg_matrix_norm::call(const at::Tensor & self, const at::Scalar & ord, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_linalg_matrix_norm_typed_handle();
    return op.call(self, ord, dim, keepdim, dtype);
}

// aten::linalg_matrix_norm(Tensor self, Scalar ord, int[] dim=[-2,-1], bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
at::Tensor linalg_matrix_norm::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & ord, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_linalg_matrix_norm_typed_handle();
    return op.redispatch(dispatchKeySet, self, ord, dim, keepdim, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_svd, name, "aten::linalg_svd")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_svd, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_svd, schema_str, "linalg_svd(Tensor self, bool full_matrices=True) -> (Tensor U, Tensor S, Tensor Vh)")

// aten::linalg_svd(Tensor self, bool full_matrices=True) -> (Tensor U, Tensor S, Tensor Vh)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_svd::schema> create_linalg_svd_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_svd::name, linalg_svd::overload_name)
      .typed<linalg_svd::schema>();
}

// aten::linalg_svd(Tensor self, bool full_matrices=True) -> (Tensor U, Tensor S, Tensor Vh)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> linalg_svd::call(const at::Tensor & self, bool full_matrices) {
    static auto op = create_linalg_svd_typed_handle();
    return op.call(self, full_matrices);
}

// aten::linalg_svd(Tensor self, bool full_matrices=True) -> (Tensor U, Tensor S, Tensor Vh)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> linalg_svd::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool full_matrices) {
    static auto op = create_linalg_svd_typed_handle();
    return op.redispatch(dispatchKeySet, self, full_matrices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_svdvals_out, name, "aten::linalg_svdvals")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_svdvals_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_svdvals_out, schema_str, "linalg_svdvals.out(Tensor input, *, Tensor(a!) out) -> Tensor(a!)")

// aten::linalg_svdvals.out(Tensor input, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_svdvals_out::schema> create_linalg_svdvals_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_svdvals_out::name, linalg_svdvals_out::overload_name)
      .typed<linalg_svdvals_out::schema>();
}

// aten::linalg_svdvals.out(Tensor input, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_svdvals_out::call(const at::Tensor & input, at::Tensor & out) {
    static auto op = create_linalg_svdvals_out_typed_handle();
    return op.call(input, out);
}

// aten::linalg_svdvals.out(Tensor input, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_svdvals_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, at::Tensor & out) {
    static auto op = create_linalg_svdvals_out_typed_handle();
    return op.redispatch(dispatchKeySet, input, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_qr_out, name, "aten::linalg_qr")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_qr_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_qr_out, schema_str, "linalg_qr.out(Tensor self, str mode='reduced', *, Tensor(a!) Q, Tensor(b!) R) -> (Tensor(a!) Q, Tensor(b!) R)")

// aten::linalg_qr.out(Tensor self, str mode='reduced', *, Tensor(a!) Q, Tensor(b!) R) -> (Tensor(a!) Q, Tensor(b!) R)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_qr_out::schema> create_linalg_qr_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_qr_out::name, linalg_qr_out::overload_name)
      .typed<linalg_qr_out::schema>();
}

// aten::linalg_qr.out(Tensor self, str mode='reduced', *, Tensor(a!) Q, Tensor(b!) R) -> (Tensor(a!) Q, Tensor(b!) R)
::std::tuple<at::Tensor &,at::Tensor &> linalg_qr_out::call(const at::Tensor & self, c10::string_view mode, at::Tensor & Q, at::Tensor & R) {
    static auto op = create_linalg_qr_out_typed_handle();
    return op.call(self, mode, Q, R);
}

// aten::linalg_qr.out(Tensor self, str mode='reduced', *, Tensor(a!) Q, Tensor(b!) R) -> (Tensor(a!) Q, Tensor(b!) R)
::std::tuple<at::Tensor &,at::Tensor &> linalg_qr_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::string_view mode, at::Tensor & Q, at::Tensor & R) {
    static auto op = create_linalg_qr_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, mode, Q, R);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_matrix_rank_tol_tensor, name, "aten::linalg_matrix_rank")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_matrix_rank_tol_tensor, overload_name, "tol_tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_matrix_rank_tol_tensor, schema_str, "linalg_matrix_rank.tol_tensor(Tensor input, Tensor tol, bool hermitian=False) -> Tensor")

// aten::linalg_matrix_rank.tol_tensor(Tensor input, Tensor tol, bool hermitian=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<linalg_matrix_rank_tol_tensor::schema> create_linalg_matrix_rank_tol_tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_matrix_rank_tol_tensor::name, linalg_matrix_rank_tol_tensor::overload_name)
      .typed<linalg_matrix_rank_tol_tensor::schema>();
}

// aten::linalg_matrix_rank.tol_tensor(Tensor input, Tensor tol, bool hermitian=False) -> Tensor
at::Tensor linalg_matrix_rank_tol_tensor::call(const at::Tensor & input, const at::Tensor & tol, bool hermitian) {
    static auto op = create_linalg_matrix_rank_tol_tensor_typed_handle();
    return op.call(input, tol, hermitian);
}

// aten::linalg_matrix_rank.tol_tensor(Tensor input, Tensor tol, bool hermitian=False) -> Tensor
at::Tensor linalg_matrix_rank_tol_tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & tol, bool hermitian) {
    static auto op = create_linalg_matrix_rank_tol_tensor_typed_handle();
    return op.redispatch(dispatchKeySet, input, tol, hermitian);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_matrix_rank_out_tol_tensor, name, "aten::linalg_matrix_rank")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_matrix_rank_out_tol_tensor, overload_name, "out_tol_tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_matrix_rank_out_tol_tensor, schema_str, "linalg_matrix_rank.out_tol_tensor(Tensor input, Tensor tol, bool hermitian=False, *, Tensor(a!) out) -> Tensor(a!)")

// aten::linalg_matrix_rank.out_tol_tensor(Tensor input, Tensor tol, bool hermitian=False, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_matrix_rank_out_tol_tensor::schema> create_linalg_matrix_rank_out_tol_tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_matrix_rank_out_tol_tensor::name, linalg_matrix_rank_out_tol_tensor::overload_name)
      .typed<linalg_matrix_rank_out_tol_tensor::schema>();
}

// aten::linalg_matrix_rank.out_tol_tensor(Tensor input, Tensor tol, bool hermitian=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_matrix_rank_out_tol_tensor::call(const at::Tensor & input, const at::Tensor & tol, bool hermitian, at::Tensor & out) {
    static auto op = create_linalg_matrix_rank_out_tol_tensor_typed_handle();
    return op.call(input, tol, hermitian, out);
}

// aten::linalg_matrix_rank.out_tol_tensor(Tensor input, Tensor tol, bool hermitian=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_matrix_rank_out_tol_tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & tol, bool hermitian, at::Tensor & out) {
    static auto op = create_linalg_matrix_rank_out_tol_tensor_typed_handle();
    return op.redispatch(dispatchKeySet, input, tol, hermitian, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_test_optional_floatlist, name, "aten::_test_optional_floatlist")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_test_optional_floatlist, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_test_optional_floatlist, schema_str, "_test_optional_floatlist(Tensor values, float[]? addends) -> Tensor")

// aten::_test_optional_floatlist(Tensor values, float[]? addends) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_test_optional_floatlist::schema> create__test_optional_floatlist_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_test_optional_floatlist::name, _test_optional_floatlist::overload_name)
      .typed<_test_optional_floatlist::schema>();
}

// aten::_test_optional_floatlist(Tensor values, float[]? addends) -> Tensor
at::Tensor _test_optional_floatlist::call(const at::Tensor & values, c10::optional<at::ArrayRef<double>> addends) {
    static auto op = create__test_optional_floatlist_typed_handle();
    return op.call(values, addends);
}

// aten::_test_optional_floatlist(Tensor values, float[]? addends) -> Tensor
at::Tensor _test_optional_floatlist::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & values, c10::optional<at::ArrayRef<double>> addends) {
    static auto op = create__test_optional_floatlist_typed_handle();
    return op.redispatch(dispatchKeySet, values, addends);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unflatten_dense_tensors, name, "aten::unflatten_dense_tensors")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unflatten_dense_tensors, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unflatten_dense_tensors, schema_str, "unflatten_dense_tensors(Tensor flat, Tensor[] tensors) -> Tensor[]")

// aten::unflatten_dense_tensors(Tensor flat, Tensor[] tensors) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<unflatten_dense_tensors::schema> create_unflatten_dense_tensors_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(unflatten_dense_tensors::name, unflatten_dense_tensors::overload_name)
      .typed<unflatten_dense_tensors::schema>();
}

// aten::unflatten_dense_tensors(Tensor flat, Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> unflatten_dense_tensors::call(const at::Tensor & flat, at::TensorList tensors) {
    static auto op = create_unflatten_dense_tensors_typed_handle();
    return op.call(flat, tensors);
}

// aten::unflatten_dense_tensors(Tensor flat, Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> unflatten_dense_tensors::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & flat, at::TensorList tensors) {
    static auto op = create_unflatten_dense_tensors_typed_handle();
    return op.redispatch(dispatchKeySet, flat, tensors);
}

}} // namespace at::_ops
