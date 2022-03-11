#include <ATen/Operators.h>
#include <ATen/Tensor.h>
#include <ATen/core/dispatch/Dispatcher.h>

// NOTE See [Sharded File] comment in VariableType

namespace at { namespace _ops {


STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cast_Double, name, "aten::_cast_Double")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cast_Double, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cast_Double, schema_str, "_cast_Double(Tensor self, bool non_blocking=False) -> Tensor")

// aten::_cast_Double(Tensor self, bool non_blocking=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_cast_Double::schema> create__cast_Double_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_cast_Double::name, _cast_Double::overload_name)
      .typed<_cast_Double::schema>();
}

// aten::_cast_Double(Tensor self, bool non_blocking=False) -> Tensor
at::Tensor _cast_Double::call(const at::Tensor & self, bool non_blocking) {
    static auto op = create__cast_Double_typed_handle();
    return op.call(self, non_blocking);
}

// aten::_cast_Double(Tensor self, bool non_blocking=False) -> Tensor
at::Tensor _cast_Double::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool non_blocking) {
    static auto op = create__cast_Double_typed_handle();
    return op.redispatch(dispatchKeySet, self, non_blocking);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cast_Float, name, "aten::_cast_Float")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cast_Float, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cast_Float, schema_str, "_cast_Float(Tensor self, bool non_blocking=False) -> Tensor")

// aten::_cast_Float(Tensor self, bool non_blocking=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_cast_Float::schema> create__cast_Float_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_cast_Float::name, _cast_Float::overload_name)
      .typed<_cast_Float::schema>();
}

// aten::_cast_Float(Tensor self, bool non_blocking=False) -> Tensor
at::Tensor _cast_Float::call(const at::Tensor & self, bool non_blocking) {
    static auto op = create__cast_Float_typed_handle();
    return op.call(self, non_blocking);
}

// aten::_cast_Float(Tensor self, bool non_blocking=False) -> Tensor
at::Tensor _cast_Float::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool non_blocking) {
    static auto op = create__cast_Float_typed_handle();
    return op.redispatch(dispatchKeySet, self, non_blocking);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cast_Half, name, "aten::_cast_Half")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cast_Half, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cast_Half, schema_str, "_cast_Half(Tensor self, bool non_blocking=False) -> Tensor")

// aten::_cast_Half(Tensor self, bool non_blocking=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_cast_Half::schema> create__cast_Half_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_cast_Half::name, _cast_Half::overload_name)
      .typed<_cast_Half::schema>();
}

// aten::_cast_Half(Tensor self, bool non_blocking=False) -> Tensor
at::Tensor _cast_Half::call(const at::Tensor & self, bool non_blocking) {
    static auto op = create__cast_Half_typed_handle();
    return op.call(self, non_blocking);
}

// aten::_cast_Half(Tensor self, bool non_blocking=False) -> Tensor
at::Tensor _cast_Half::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool non_blocking) {
    static auto op = create__cast_Half_typed_handle();
    return op.redispatch(dispatchKeySet, self, non_blocking);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_version, name, "aten::_version")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_version, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_version, schema_str, "_version(Tensor self) -> int")

// aten::_version(Tensor self) -> int
static C10_NOINLINE c10::TypedOperatorHandle<_version::schema> create__version_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_version::name, _version::overload_name)
      .typed<_version::schema>();
}

// aten::_version(Tensor self) -> int
int64_t _version::call(const at::Tensor & self) {
    static auto op = create__version_typed_handle();
    return op.call(self);
}

// aten::_version(Tensor self) -> int
int64_t _version::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create__version_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(requires_grad_, name, "aten::requires_grad_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(requires_grad_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(requires_grad_, schema_str, "requires_grad_(Tensor(a!) self, bool requires_grad=True) -> Tensor(a!)")

// aten::requires_grad_(Tensor(a!) self, bool requires_grad=True) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<requires_grad_::schema> create_requires_grad__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(requires_grad_::name, requires_grad_::overload_name)
      .typed<requires_grad_::schema>();
}

// aten::requires_grad_(Tensor(a!) self, bool requires_grad=True) -> Tensor(a!)
at::Tensor & requires_grad_::call(at::Tensor & self, bool requires_grad) {
    static auto op = create_requires_grad__typed_handle();
    return op.call(self, requires_grad);
}

// aten::requires_grad_(Tensor(a!) self, bool requires_grad=True) -> Tensor(a!)
at::Tensor & requires_grad_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, bool requires_grad) {
    static auto op = create_requires_grad__typed_handle();
    return op.redispatch(dispatchKeySet, self, requires_grad);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_make_dual, name, "aten::_make_dual")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_make_dual, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_make_dual, schema_str, "_make_dual(Tensor(a) primal, Tensor tangent, int level) -> Tensor(a)")

// aten::_make_dual(Tensor(a) primal, Tensor tangent, int level) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<_make_dual::schema> create__make_dual_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_make_dual::name, _make_dual::overload_name)
      .typed<_make_dual::schema>();
}

// aten::_make_dual(Tensor(a) primal, Tensor tangent, int level) -> Tensor(a)
at::Tensor _make_dual::call(const at::Tensor & primal, const at::Tensor & tangent, int64_t level) {
    static auto op = create__make_dual_typed_handle();
    return op.call(primal, tangent, level);
}

// aten::_make_dual(Tensor(a) primal, Tensor tangent, int level) -> Tensor(a)
at::Tensor _make_dual::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & primal, const at::Tensor & tangent, int64_t level) {
    static auto op = create__make_dual_typed_handle();
    return op.redispatch(dispatchKeySet, primal, tangent, level);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(align_to_ellipsis_idx, name, "aten::align_to")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(align_to_ellipsis_idx, overload_name, "ellipsis_idx")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(align_to_ellipsis_idx, schema_str, "align_to.ellipsis_idx(Tensor(a) self, Dimname[] order, int ellipsis_idx) -> Tensor(a)")

// aten::align_to.ellipsis_idx(Tensor(a) self, Dimname[] order, int ellipsis_idx) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<align_to_ellipsis_idx::schema> create_align_to_ellipsis_idx_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(align_to_ellipsis_idx::name, align_to_ellipsis_idx::overload_name)
      .typed<align_to_ellipsis_idx::schema>();
}

// aten::align_to.ellipsis_idx(Tensor(a) self, Dimname[] order, int ellipsis_idx) -> Tensor(a)
at::Tensor align_to_ellipsis_idx::call(const at::Tensor & self, at::DimnameList order, int64_t ellipsis_idx) {
    static auto op = create_align_to_ellipsis_idx_typed_handle();
    return op.call(self, order, ellipsis_idx);
}

// aten::align_to.ellipsis_idx(Tensor(a) self, Dimname[] order, int ellipsis_idx) -> Tensor(a)
at::Tensor align_to_ellipsis_idx::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList order, int64_t ellipsis_idx) {
    static auto op = create_align_to_ellipsis_idx_typed_handle();
    return op.redispatch(dispatchKeySet, self, order, ellipsis_idx);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(align_as, name, "aten::align_as")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(align_as, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(align_as, schema_str, "align_as(Tensor self, Tensor other) -> Tensor")

// aten::align_as(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<align_as::schema> create_align_as_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(align_as::name, align_as::overload_name)
      .typed<align_as::schema>();
}

// aten::align_as(Tensor self, Tensor other) -> Tensor
at::Tensor align_as::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_align_as_typed_handle();
    return op.call(self, other);
}

// aten::align_as(Tensor self, Tensor other) -> Tensor
at::Tensor align_as::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_align_as_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(refine_names, name, "aten::refine_names")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(refine_names, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(refine_names, schema_str, "refine_names(Tensor(a) self, Dimname[] names) -> Tensor(a)")

// aten::refine_names(Tensor(a) self, Dimname[] names) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<refine_names::schema> create_refine_names_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(refine_names::name, refine_names::overload_name)
      .typed<refine_names::schema>();
}

// aten::refine_names(Tensor(a) self, Dimname[] names) -> Tensor(a)
at::Tensor refine_names::call(const at::Tensor & self, at::DimnameList names) {
    static auto op = create_refine_names_typed_handle();
    return op.call(self, names);
}

// aten::refine_names(Tensor(a) self, Dimname[] names) -> Tensor(a)
at::Tensor refine_names::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList names) {
    static auto op = create_refine_names_typed_handle();
    return op.redispatch(dispatchKeySet, self, names);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_use_cudnn_rnn_flatten_weight, name, "aten::_use_cudnn_rnn_flatten_weight")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_use_cudnn_rnn_flatten_weight, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_use_cudnn_rnn_flatten_weight, schema_str, "_use_cudnn_rnn_flatten_weight() -> bool")

// aten::_use_cudnn_rnn_flatten_weight() -> bool
static C10_NOINLINE c10::TypedOperatorHandle<_use_cudnn_rnn_flatten_weight::schema> create__use_cudnn_rnn_flatten_weight_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_use_cudnn_rnn_flatten_weight::name, _use_cudnn_rnn_flatten_weight::overload_name)
      .typed<_use_cudnn_rnn_flatten_weight::schema>();
}

// aten::_use_cudnn_rnn_flatten_weight() -> bool
bool _use_cudnn_rnn_flatten_weight::call() {
    static auto op = create__use_cudnn_rnn_flatten_weight_typed_handle();
    return op.call();
}

// aten::_use_cudnn_rnn_flatten_weight() -> bool
bool _use_cudnn_rnn_flatten_weight::redispatch(c10::DispatchKeySet dispatchKeySet) {
    static auto op = create__use_cudnn_rnn_flatten_weight_typed_handle();
    return op.redispatch(dispatchKeySet);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cudnn_rnn_flatten_weight, name, "aten::_cudnn_rnn_flatten_weight")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cudnn_rnn_flatten_weight, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cudnn_rnn_flatten_weight, schema_str, "_cudnn_rnn_flatten_weight(Tensor[] weight_arr, int weight_stride0, int input_size, int mode, int hidden_size, int proj_size, int num_layers, bool batch_first, bool bidirectional) -> Tensor")

// aten::_cudnn_rnn_flatten_weight(Tensor[] weight_arr, int weight_stride0, int input_size, int mode, int hidden_size, int proj_size, int num_layers, bool batch_first, bool bidirectional) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_cudnn_rnn_flatten_weight::schema> create__cudnn_rnn_flatten_weight_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_cudnn_rnn_flatten_weight::name, _cudnn_rnn_flatten_weight::overload_name)
      .typed<_cudnn_rnn_flatten_weight::schema>();
}

// aten::_cudnn_rnn_flatten_weight(Tensor[] weight_arr, int weight_stride0, int input_size, int mode, int hidden_size, int proj_size, int num_layers, bool batch_first, bool bidirectional) -> Tensor
at::Tensor _cudnn_rnn_flatten_weight::call(at::TensorList weight_arr, int64_t weight_stride0, int64_t input_size, int64_t mode, int64_t hidden_size, int64_t proj_size, int64_t num_layers, bool batch_first, bool bidirectional) {
    static auto op = create__cudnn_rnn_flatten_weight_typed_handle();
    return op.call(weight_arr, weight_stride0, input_size, mode, hidden_size, proj_size, num_layers, batch_first, bidirectional);
}

// aten::_cudnn_rnn_flatten_weight(Tensor[] weight_arr, int weight_stride0, int input_size, int mode, int hidden_size, int proj_size, int num_layers, bool batch_first, bool bidirectional) -> Tensor
at::Tensor _cudnn_rnn_flatten_weight::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList weight_arr, int64_t weight_stride0, int64_t input_size, int64_t mode, int64_t hidden_size, int64_t proj_size, int64_t num_layers, bool batch_first, bool bidirectional) {
    static auto op = create__cudnn_rnn_flatten_weight_typed_handle();
    return op.redispatch(dispatchKeySet, weight_arr, weight_stride0, input_size, mode, hidden_size, proj_size, num_layers, batch_first, bidirectional);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(feature_alpha_dropout, name, "aten::feature_alpha_dropout")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(feature_alpha_dropout, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(feature_alpha_dropout, schema_str, "feature_alpha_dropout(Tensor input, float p, bool train) -> Tensor")

// aten::feature_alpha_dropout(Tensor input, float p, bool train) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<feature_alpha_dropout::schema> create_feature_alpha_dropout_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(feature_alpha_dropout::name, feature_alpha_dropout::overload_name)
      .typed<feature_alpha_dropout::schema>();
}

// aten::feature_alpha_dropout(Tensor input, float p, bool train) -> Tensor
at::Tensor feature_alpha_dropout::call(const at::Tensor & input, double p, bool train) {
    static auto op = create_feature_alpha_dropout_typed_handle();
    return op.call(input, p, train);
}

// aten::feature_alpha_dropout(Tensor input, float p, bool train) -> Tensor
at::Tensor feature_alpha_dropout::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, double p, bool train) {
    static auto op = create_feature_alpha_dropout_typed_handle();
    return op.redispatch(dispatchKeySet, input, p, train);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(feature_alpha_dropout_, name, "aten::feature_alpha_dropout_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(feature_alpha_dropout_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(feature_alpha_dropout_, schema_str, "feature_alpha_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)")

// aten::feature_alpha_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<feature_alpha_dropout_::schema> create_feature_alpha_dropout__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(feature_alpha_dropout_::name, feature_alpha_dropout_::overload_name)
      .typed<feature_alpha_dropout_::schema>();
}

// aten::feature_alpha_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)
at::Tensor & feature_alpha_dropout_::call(at::Tensor & self, double p, bool train) {
    static auto op = create_feature_alpha_dropout__typed_handle();
    return op.call(self, p, train);
}

// aten::feature_alpha_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)
at::Tensor & feature_alpha_dropout_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, double p, bool train) {
    static auto op = create_feature_alpha_dropout__typed_handle();
    return op.redispatch(dispatchKeySet, self, p, train);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(abs, name, "aten::abs")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(abs, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(abs, schema_str, "abs(Tensor self) -> Tensor")

// aten::abs(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<abs::schema> create_abs_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(abs::name, abs::overload_name)
      .typed<abs::schema>();
}

// aten::abs(Tensor self) -> Tensor
at::Tensor abs::call(const at::Tensor & self) {
    static auto op = create_abs_typed_handle();
    return op.call(self);
}

// aten::abs(Tensor self) -> Tensor
at::Tensor abs::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_abs_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(imag, name, "aten::imag")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(imag, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(imag, schema_str, "imag(Tensor(a) self) -> Tensor(a)")

// aten::imag(Tensor(a) self) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<imag::schema> create_imag_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(imag::name, imag::overload_name)
      .typed<imag::schema>();
}

// aten::imag(Tensor(a) self) -> Tensor(a)
at::Tensor imag::call(const at::Tensor & self) {
    static auto op = create_imag_typed_handle();
    return op.call(self);
}

// aten::imag(Tensor(a) self) -> Tensor(a)
at::Tensor imag::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_imag_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(resolve_conj, name, "aten::resolve_conj")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(resolve_conj, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(resolve_conj, schema_str, "resolve_conj(Tensor(a) self) -> Tensor(a)")

// aten::resolve_conj(Tensor(a) self) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<resolve_conj::schema> create_resolve_conj_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(resolve_conj::name, resolve_conj::overload_name)
      .typed<resolve_conj::schema>();
}

// aten::resolve_conj(Tensor(a) self) -> Tensor(a)
at::Tensor resolve_conj::call(const at::Tensor & self) {
    static auto op = create_resolve_conj_typed_handle();
    return op.call(self);
}

// aten::resolve_conj(Tensor(a) self) -> Tensor(a)
at::Tensor resolve_conj::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_resolve_conj_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(resolve_neg, name, "aten::resolve_neg")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(resolve_neg, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(resolve_neg, schema_str, "resolve_neg(Tensor(a) self) -> Tensor(a)")

// aten::resolve_neg(Tensor(a) self) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<resolve_neg::schema> create_resolve_neg_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(resolve_neg::name, resolve_neg::overload_name)
      .typed<resolve_neg::schema>();
}

// aten::resolve_neg(Tensor(a) self) -> Tensor(a)
at::Tensor resolve_neg::call(const at::Tensor & self) {
    static auto op = create_resolve_neg_typed_handle();
    return op.call(self);
}

// aten::resolve_neg(Tensor(a) self) -> Tensor(a)
at::Tensor resolve_neg::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_resolve_neg_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(acos_out, name, "aten::acos")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(acos_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(acos_out, schema_str, "acos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::acos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<acos_out::schema> create_acos_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(acos_out::name, acos_out::overload_name)
      .typed<acos_out::schema>();
}

// aten::acos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & acos_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_acos_out_typed_handle();
    return op.call(self, out);
}

// aten::acos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & acos_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_acos_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_max_pool1d, name, "aten::adaptive_max_pool1d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_max_pool1d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_max_pool1d, schema_str, "adaptive_max_pool1d(Tensor self, int[1] output_size) -> (Tensor, Tensor)")

// aten::adaptive_max_pool1d(Tensor self, int[1] output_size) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<adaptive_max_pool1d::schema> create_adaptive_max_pool1d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(adaptive_max_pool1d::name, adaptive_max_pool1d::overload_name)
      .typed<adaptive_max_pool1d::schema>();
}

// aten::adaptive_max_pool1d(Tensor self, int[1] output_size) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> adaptive_max_pool1d::call(const at::Tensor & self, at::IntArrayRef output_size) {
    static auto op = create_adaptive_max_pool1d_typed_handle();
    return op.call(self, output_size);
}

// aten::adaptive_max_pool1d(Tensor self, int[1] output_size) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> adaptive_max_pool1d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size) {
    static auto op = create_adaptive_max_pool1d_typed_handle();
    return op.redispatch(dispatchKeySet, self, output_size);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(add__Tensor, name, "aten::add_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(add__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(add__Tensor, schema_str, "add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)")

// aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<add__Tensor::schema> create_add__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(add__Tensor::name, add__Tensor::overload_name)
      .typed<add__Tensor::schema>();
}

// aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
at::Tensor & add__Tensor::call(at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
    static auto op = create_add__Tensor_typed_handle();
    return op.call(self, other, alpha);
}

// aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
at::Tensor & add__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
    static auto op = create_add__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_add_relu_out, name, "aten::_add_relu")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_add_relu_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_add_relu_out, schema_str, "_add_relu.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)")

// aten::_add_relu.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<_add_relu_out::schema> create__add_relu_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_add_relu_out::name, _add_relu_out::overload_name)
      .typed<_add_relu_out::schema>();
}

// aten::_add_relu.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
at::Tensor & _add_relu_out::call(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
    static auto op = create__add_relu_out_typed_handle();
    return op.call(self, other, alpha, out);
}

// aten::_add_relu.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
at::Tensor & _add_relu_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
    static auto op = create__add_relu_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, alpha, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addmv, name, "aten::addmv")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addmv, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addmv, schema_str, "addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor")

// aten::addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<addmv::schema> create_addmv_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(addmv::name, addmv::overload_name)
      .typed<addmv::schema>();
}

// aten::addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor
at::Tensor addmv::call(const at::Tensor & self, const at::Tensor & mat, const at::Tensor & vec, const at::Scalar & beta, const at::Scalar & alpha) {
    static auto op = create_addmv_typed_handle();
    return op.call(self, mat, vec, beta, alpha);
}

// aten::addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor
at::Tensor addmv::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mat, const at::Tensor & vec, const at::Scalar & beta, const at::Scalar & alpha) {
    static auto op = create_addmv_typed_handle();
    return op.redispatch(dispatchKeySet, self, mat, vec, beta, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addr, name, "aten::addr")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addr, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addr, schema_str, "addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor")

// aten::addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<addr::schema> create_addr_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(addr::name, addr::overload_name)
      .typed<addr::schema>();
}

// aten::addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
at::Tensor addr::call(const at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2, const at::Scalar & beta, const at::Scalar & alpha) {
    static auto op = create_addr_typed_handle();
    return op.call(self, vec1, vec2, beta, alpha);
}

// aten::addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
at::Tensor addr::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2, const at::Scalar & beta, const at::Scalar & alpha) {
    static auto op = create_addr_typed_handle();
    return op.redispatch(dispatchKeySet, self, vec1, vec2, beta, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addr_, name, "aten::addr_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addr_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addr_, schema_str, "addr_(Tensor(a!) self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)")

// aten::addr_(Tensor(a!) self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<addr_::schema> create_addr__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(addr_::name, addr_::overload_name)
      .typed<addr_::schema>();
}

// aten::addr_(Tensor(a!) self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
at::Tensor & addr_::call(at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2, const at::Scalar & beta, const at::Scalar & alpha) {
    static auto op = create_addr__typed_handle();
    return op.call(self, vec1, vec2, beta, alpha);
}

// aten::addr_(Tensor(a!) self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
at::Tensor & addr_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2, const at::Scalar & beta, const at::Scalar & alpha) {
    static auto op = create_addr__typed_handle();
    return op.redispatch(dispatchKeySet, self, vec1, vec2, beta, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(affine_grid_generator_backward, name, "aten::affine_grid_generator_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(affine_grid_generator_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(affine_grid_generator_backward, schema_str, "affine_grid_generator_backward(Tensor grad, int[] size, bool align_corners) -> Tensor")

// aten::affine_grid_generator_backward(Tensor grad, int[] size, bool align_corners) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<affine_grid_generator_backward::schema> create_affine_grid_generator_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(affine_grid_generator_backward::name, affine_grid_generator_backward::overload_name)
      .typed<affine_grid_generator_backward::schema>();
}

// aten::affine_grid_generator_backward(Tensor grad, int[] size, bool align_corners) -> Tensor
at::Tensor affine_grid_generator_backward::call(const at::Tensor & grad, at::IntArrayRef size, bool align_corners) {
    static auto op = create_affine_grid_generator_backward_typed_handle();
    return op.call(grad, size, align_corners);
}

// aten::affine_grid_generator_backward(Tensor grad, int[] size, bool align_corners) -> Tensor
at::Tensor affine_grid_generator_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, at::IntArrayRef size, bool align_corners) {
    static auto op = create_affine_grid_generator_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad, size, align_corners);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(any_dimname, name, "aten::any")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(any_dimname, overload_name, "dimname")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(any_dimname, schema_str, "any.dimname(Tensor self, Dimname dim, bool keepdim=False) -> Tensor")

// aten::any.dimname(Tensor self, Dimname dim, bool keepdim=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<any_dimname::schema> create_any_dimname_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(any_dimname::name, any_dimname::overload_name)
      .typed<any_dimname::schema>();
}

// aten::any.dimname(Tensor self, Dimname dim, bool keepdim=False) -> Tensor
at::Tensor any_dimname::call(const at::Tensor & self, at::Dimname dim, bool keepdim) {
    static auto op = create_any_dimname_typed_handle();
    return op.call(self, dim, keepdim);
}

// aten::any.dimname(Tensor self, Dimname dim, bool keepdim=False) -> Tensor
at::Tensor any_dimname::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, bool keepdim) {
    static auto op = create_any_dimname_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(argmin, name, "aten::argmin")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(argmin, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(argmin, schema_str, "argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor")

// aten::argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<argmin::schema> create_argmin_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(argmin::name, argmin::overload_name)
      .typed<argmin::schema>();
}

// aten::argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
at::Tensor argmin::call(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim) {
    static auto op = create_argmin_typed_handle();
    return op.call(self, dim, keepdim);
}

// aten::argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
at::Tensor argmin::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim) {
    static auto op = create_argmin_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(acosh_out, name, "aten::acosh")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(acosh_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(acosh_out, schema_str, "acosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::acosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<acosh_out::schema> create_acosh_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(acosh_out::name, acosh_out::overload_name)
      .typed<acosh_out::schema>();
}

// aten::acosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & acosh_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_acosh_out_typed_handle();
    return op.call(self, out);
}

// aten::acosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & acosh_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_acosh_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arcsinh_, name, "aten::arcsinh_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arcsinh_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arcsinh_, schema_str, "arcsinh_(Tensor(a!) self) -> Tensor(a!)")

// aten::arcsinh_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<arcsinh_::schema> create_arcsinh__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(arcsinh_::name, arcsinh_::overload_name)
      .typed<arcsinh_::schema>();
}

// aten::arcsinh_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & arcsinh_::call(at::Tensor & self) {
    static auto op = create_arcsinh__typed_handle();
    return op.call(self);
}

// aten::arcsinh_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & arcsinh_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_arcsinh__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atanh_out, name, "aten::atanh")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atanh_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atanh_out, schema_str, "atanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::atanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<atanh_out::schema> create_atanh_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(atanh_out::name, atanh_out::overload_name)
      .typed<atanh_out::schema>();
}

// aten::atanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & atanh_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_atanh_out_typed_handle();
    return op.call(self, out);
}

// aten::atanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & atanh_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_atanh_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arctanh_out, name, "aten::arctanh")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arctanh_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arctanh_out, schema_str, "arctanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::arctanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<arctanh_out::schema> create_arctanh_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(arctanh_out::name, arctanh_out::overload_name)
      .typed<arctanh_out::schema>();
}

// aten::arctanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & arctanh_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_arctanh_out_typed_handle();
    return op.call(self, out);
}

// aten::arctanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & arctanh_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_arctanh_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(asin_, name, "aten::asin_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(asin_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(asin_, schema_str, "asin_(Tensor(a!) self) -> Tensor(a!)")

// aten::asin_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<asin_::schema> create_asin__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(asin_::name, asin_::overload_name)
      .typed<asin_::schema>();
}

// aten::asin_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & asin_::call(at::Tensor & self) {
    static auto op = create_asin__typed_handle();
    return op.call(self);
}

// aten::asin_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & asin_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_asin__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arcsin_out, name, "aten::arcsin")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arcsin_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arcsin_out, schema_str, "arcsin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::arcsin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<arcsin_out::schema> create_arcsin_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(arcsin_out::name, arcsin_out::overload_name)
      .typed<arcsin_out::schema>();
}

// aten::arcsin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & arcsin_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_arcsin_out_typed_handle();
    return op.call(self, out);
}

// aten::arcsin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & arcsin_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_arcsin_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atan, name, "aten::atan")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atan, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atan, schema_str, "atan(Tensor self) -> Tensor")

// aten::atan(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<atan::schema> create_atan_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(atan::name, atan::overload_name)
      .typed<atan::schema>();
}

// aten::atan(Tensor self) -> Tensor
at::Tensor atan::call(const at::Tensor & self) {
    static auto op = create_atan_typed_handle();
    return op.call(self);
}

// aten::atan(Tensor self) -> Tensor
at::Tensor atan::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_atan_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arctan, name, "aten::arctan")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arctan, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arctan, schema_str, "arctan(Tensor self) -> Tensor")

// aten::arctan(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<arctan::schema> create_arctan_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(arctan::name, arctan::overload_name)
      .typed<arctan::schema>();
}

// aten::arctan(Tensor self) -> Tensor
at::Tensor arctan::call(const at::Tensor & self) {
    static auto op = create_arctan_typed_handle();
    return op.call(self);
}

// aten::arctan(Tensor self) -> Tensor
at::Tensor arctan::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_arctan_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arctan_out, name, "aten::arctan")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arctan_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arctan_out, schema_str, "arctan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::arctan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<arctan_out::schema> create_arctan_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(arctan_out::name, arctan_out::overload_name)
      .typed<arctan_out::schema>();
}

// aten::arctan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & arctan_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_arctan_out_typed_handle();
    return op.call(self, out);
}

// aten::arctan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & arctan_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_arctan_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atleast_1d_Sequence, name, "aten::atleast_1d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atleast_1d_Sequence, overload_name, "Sequence")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atleast_1d_Sequence, schema_str, "atleast_1d.Sequence(Tensor[] tensors) -> Tensor[]")

// aten::atleast_1d.Sequence(Tensor[] tensors) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<atleast_1d_Sequence::schema> create_atleast_1d_Sequence_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(atleast_1d_Sequence::name, atleast_1d_Sequence::overload_name)
      .typed<atleast_1d_Sequence::schema>();
}

// aten::atleast_1d.Sequence(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> atleast_1d_Sequence::call(at::TensorList tensors) {
    static auto op = create_atleast_1d_Sequence_typed_handle();
    return op.call(tensors);
}

// aten::atleast_1d.Sequence(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> atleast_1d_Sequence::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create_atleast_1d_Sequence_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atleast_3d_Sequence, name, "aten::atleast_3d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atleast_3d_Sequence, overload_name, "Sequence")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atleast_3d_Sequence, schema_str, "atleast_3d.Sequence(Tensor[] tensors) -> Tensor[]")

// aten::atleast_3d.Sequence(Tensor[] tensors) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<atleast_3d_Sequence::schema> create_atleast_3d_Sequence_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(atleast_3d_Sequence::name, atleast_3d_Sequence::overload_name)
      .typed<atleast_3d_Sequence::schema>();
}

// aten::atleast_3d.Sequence(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> atleast_3d_Sequence::call(at::TensorList tensors) {
    static auto op = create_atleast_3d_Sequence_typed_handle();
    return op.call(tensors);
}

// aten::atleast_3d.Sequence(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> atleast_3d_Sequence::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create_atleast_3d_Sequence_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(baddbmm_out, name, "aten::baddbmm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(baddbmm_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(baddbmm_out, schema_str, "baddbmm.out(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)")

// aten::baddbmm.out(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<baddbmm_out::schema> create_baddbmm_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(baddbmm_out::name, baddbmm_out::overload_name)
      .typed<baddbmm_out::schema>();
}

// aten::baddbmm.out(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
at::Tensor & baddbmm_out::call(const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
    static auto op = create_baddbmm_out_typed_handle();
    return op.call(self, batch1, batch2, beta, alpha, out);
}

// aten::baddbmm.out(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
at::Tensor & baddbmm_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
    static auto op = create_baddbmm_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, batch1, batch2, beta, alpha, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantized_batch_norm, name, "aten::quantized_batch_norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantized_batch_norm, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantized_batch_norm, schema_str, "quantized_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> Tensor")

// aten::quantized_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<quantized_batch_norm::schema> create_quantized_batch_norm_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(quantized_batch_norm::name, quantized_batch_norm::overload_name)
      .typed<quantized_batch_norm::schema>();
}

// aten::quantized_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> Tensor
at::Tensor quantized_batch_norm::call(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const at::Tensor & mean, const at::Tensor & var, double eps, double output_scale, int64_t output_zero_point) {
    static auto op = create_quantized_batch_norm_typed_handle();
    return op.call(input, weight, bias, mean, var, eps, output_scale, output_zero_point);
}

// aten::quantized_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> Tensor
at::Tensor quantized_batch_norm::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const at::Tensor & mean, const at::Tensor & var, double eps, double output_scale, int64_t output_zero_point) {
    static auto op = create_quantized_batch_norm_typed_handle();
    return op.redispatch(dispatchKeySet, input, weight, bias, mean, var, eps, output_scale, output_zero_point);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bernoulli_out, name, "aten::bernoulli")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bernoulli_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bernoulli_out, schema_str, "bernoulli.out(Tensor self, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)")

// aten::bernoulli.out(Tensor self, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<bernoulli_out::schema> create_bernoulli_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bernoulli_out::name, bernoulli_out::overload_name)
      .typed<bernoulli_out::schema>();
}

// aten::bernoulli.out(Tensor self, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & bernoulli_out::call(const at::Tensor & self, c10::optional<at::Generator> generator, at::Tensor & out) {
    static auto op = create_bernoulli_out_typed_handle();
    return op.call(self, generator, out);
}

// aten::bernoulli.out(Tensor self, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & bernoulli_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::Generator> generator, at::Tensor & out) {
    static auto op = create_bernoulli_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, generator, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(binary_cross_entropy_backward, name, "aten::binary_cross_entropy_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(binary_cross_entropy_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(binary_cross_entropy_backward, schema_str, "binary_cross_entropy_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor")

// aten::binary_cross_entropy_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<binary_cross_entropy_backward::schema> create_binary_cross_entropy_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(binary_cross_entropy_backward::name, binary_cross_entropy_backward::overload_name)
      .typed<binary_cross_entropy_backward::schema>();
}

// aten::binary_cross_entropy_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor
at::Tensor binary_cross_entropy_backward::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction) {
    static auto op = create_binary_cross_entropy_backward_typed_handle();
    return op.call(grad_output, self, target, weight, reduction);
}

// aten::binary_cross_entropy_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor
at::Tensor binary_cross_entropy_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction) {
    static auto op = create_binary_cross_entropy_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, target, weight, reduction);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(binary_cross_entropy_backward_grad_input, name, "aten::binary_cross_entropy_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(binary_cross_entropy_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(binary_cross_entropy_backward_grad_input, schema_str, "binary_cross_entropy_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::binary_cross_entropy_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<binary_cross_entropy_backward_grad_input::schema> create_binary_cross_entropy_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(binary_cross_entropy_backward_grad_input::name, binary_cross_entropy_backward_grad_input::overload_name)
      .typed<binary_cross_entropy_backward_grad_input::schema>();
}

// aten::binary_cross_entropy_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & binary_cross_entropy_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & grad_input) {
    static auto op = create_binary_cross_entropy_backward_grad_input_typed_handle();
    return op.call(grad_output, self, target, weight, reduction, grad_input);
}

// aten::binary_cross_entropy_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & binary_cross_entropy_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & grad_input) {
    static auto op = create_binary_cross_entropy_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, target, weight, reduction, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_not, name, "aten::bitwise_not")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_not, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_not, schema_str, "bitwise_not(Tensor self) -> Tensor")

// aten::bitwise_not(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<bitwise_not::schema> create_bitwise_not_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bitwise_not::name, bitwise_not::overload_name)
      .typed<bitwise_not::schema>();
}

// aten::bitwise_not(Tensor self) -> Tensor
at::Tensor bitwise_not::call(const at::Tensor & self) {
    static auto op = create_bitwise_not_typed_handle();
    return op.call(self);
}

// aten::bitwise_not(Tensor self) -> Tensor
at::Tensor bitwise_not::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_bitwise_not_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(copysign__Tensor, name, "aten::copysign_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(copysign__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(copysign__Tensor, schema_str, "copysign_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::copysign_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<copysign__Tensor::schema> create_copysign__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(copysign__Tensor::name, copysign__Tensor::overload_name)
      .typed<copysign__Tensor::schema>();
}

// aten::copysign_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & copysign__Tensor::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create_copysign__Tensor_typed_handle();
    return op.call(self, other);
}

// aten::copysign_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & copysign__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create_copysign__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logical_not, name, "aten::logical_not")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logical_not, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logical_not, schema_str, "logical_not(Tensor self) -> Tensor")

// aten::logical_not(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<logical_not::schema> create_logical_not_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(logical_not::name, logical_not::overload_name)
      .typed<logical_not::schema>();
}

// aten::logical_not(Tensor self) -> Tensor
at::Tensor logical_not::call(const at::Tensor & self) {
    static auto op = create_logical_not_typed_handle();
    return op.call(self);
}

// aten::logical_not(Tensor self) -> Tensor
at::Tensor logical_not::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_logical_not_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logical_not_, name, "aten::logical_not_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logical_not_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logical_not_, schema_str, "logical_not_(Tensor(a!) self) -> Tensor(a!)")

// aten::logical_not_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<logical_not_::schema> create_logical_not__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(logical_not_::name, logical_not_::overload_name)
      .typed<logical_not_::schema>();
}

// aten::logical_not_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & logical_not_::call(at::Tensor & self) {
    static auto op = create_logical_not__typed_handle();
    return op.call(self);
}

// aten::logical_not_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & logical_not_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_logical_not__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logical_not_out, name, "aten::logical_not")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logical_not_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logical_not_out, schema_str, "logical_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::logical_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<logical_not_out::schema> create_logical_not_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(logical_not_out::name, logical_not_out::overload_name)
      .typed<logical_not_out::schema>();
}

// aten::logical_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & logical_not_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_logical_not_out_typed_handle();
    return op.call(self, out);
}

// aten::logical_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & logical_not_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_logical_not_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logical_and_out, name, "aten::logical_and")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logical_and_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logical_and_out, schema_str, "logical_and.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::logical_and.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<logical_and_out::schema> create_logical_and_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(logical_and_out::name, logical_and_out::overload_name)
      .typed<logical_and_out::schema>();
}

// aten::logical_and.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & logical_and_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_logical_and_out_typed_handle();
    return op.call(self, other, out);
}

// aten::logical_and.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & logical_and_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_logical_and_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(blackman_window_periodic, name, "aten::blackman_window")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(blackman_window_periodic, overload_name, "periodic")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(blackman_window_periodic, schema_str, "blackman_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::blackman_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<blackman_window_periodic::schema> create_blackman_window_periodic_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(blackman_window_periodic::name, blackman_window_periodic::overload_name)
      .typed<blackman_window_periodic::schema>();
}

// aten::blackman_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor blackman_window_periodic::call(int64_t window_length, bool periodic, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_blackman_window_periodic_typed_handle();
    return op.call(window_length, periodic, dtype, layout, device, pin_memory);
}

// aten::blackman_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor blackman_window_periodic::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t window_length, bool periodic, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_blackman_window_periodic_typed_handle();
    return op.redispatch(dispatchKeySet, window_length, periodic, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cat_out, name, "aten::cat")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cat_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cat_out, schema_str, "cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)")

// aten::cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<cat_out::schema> create_cat_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cat_out::name, cat_out::overload_name)
      .typed<cat_out::schema>();
}

// aten::cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & cat_out::call(at::TensorList tensors, int64_t dim, at::Tensor & out) {
    static auto op = create_cat_out_typed_handle();
    return op.call(tensors, dim, out);
}

// aten::cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & cat_out::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, int64_t dim, at::Tensor & out) {
    static auto op = create_cat_out_typed_handle();
    return op.redispatch(dispatchKeySet, tensors, dim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(concat_names, name, "aten::concat")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(concat_names, overload_name, "names")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(concat_names, schema_str, "concat.names(Tensor[] tensors, Dimname dim) -> Tensor")

// aten::concat.names(Tensor[] tensors, Dimname dim) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<concat_names::schema> create_concat_names_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(concat_names::name, concat_names::overload_name)
      .typed<concat_names::schema>();
}

// aten::concat.names(Tensor[] tensors, Dimname dim) -> Tensor
at::Tensor concat_names::call(at::TensorList tensors, at::Dimname dim) {
    static auto op = create_concat_names_typed_handle();
    return op.call(tensors, dim);
}

// aten::concat.names(Tensor[] tensors, Dimname dim) -> Tensor
at::Tensor concat_names::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, at::Dimname dim) {
    static auto op = create_concat_names_typed_handle();
    return op.redispatch(dispatchKeySet, tensors, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ceil, name, "aten::ceil")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ceil, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ceil, schema_str, "ceil(Tensor self) -> Tensor")

// aten::ceil(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<ceil::schema> create_ceil_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(ceil::name, ceil::overload_name)
      .typed<ceil::schema>();
}

// aten::ceil(Tensor self) -> Tensor
at::Tensor ceil::call(const at::Tensor & self) {
    static auto op = create_ceil_typed_handle();
    return op.call(self);
}

// aten::ceil(Tensor self) -> Tensor
at::Tensor ceil::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_ceil_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ceil_, name, "aten::ceil_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ceil_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ceil_, schema_str, "ceil_(Tensor(a!) self) -> Tensor(a!)")

// aten::ceil_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<ceil_::schema> create_ceil__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(ceil_::name, ceil_::overload_name)
      .typed<ceil_::schema>();
}

// aten::ceil_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & ceil_::call(at::Tensor & self) {
    static auto op = create_ceil__typed_handle();
    return op.call(self);
}

// aten::ceil_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & ceil_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_ceil__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tensor_split_tensor_indices_or_sections, name, "aten::tensor_split")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tensor_split_tensor_indices_or_sections, overload_name, "tensor_indices_or_sections")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tensor_split_tensor_indices_or_sections, schema_str, "tensor_split.tensor_indices_or_sections(Tensor(a) self, Tensor tensor_indices_or_sections, int dim=0) -> Tensor(a)[]")

// aten::tensor_split.tensor_indices_or_sections(Tensor(a) self, Tensor tensor_indices_or_sections, int dim=0) -> Tensor(a)[]
static C10_NOINLINE c10::TypedOperatorHandle<tensor_split_tensor_indices_or_sections::schema> create_tensor_split_tensor_indices_or_sections_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(tensor_split_tensor_indices_or_sections::name, tensor_split_tensor_indices_or_sections::overload_name)
      .typed<tensor_split_tensor_indices_or_sections::schema>();
}

// aten::tensor_split.tensor_indices_or_sections(Tensor(a) self, Tensor tensor_indices_or_sections, int dim=0) -> Tensor(a)[]
::std::vector<at::Tensor> tensor_split_tensor_indices_or_sections::call(const at::Tensor & self, const at::Tensor & tensor_indices_or_sections, int64_t dim) {
    static auto op = create_tensor_split_tensor_indices_or_sections_typed_handle();
    return op.call(self, tensor_indices_or_sections, dim);
}

// aten::tensor_split.tensor_indices_or_sections(Tensor(a) self, Tensor tensor_indices_or_sections, int dim=0) -> Tensor(a)[]
::std::vector<at::Tensor> tensor_split_tensor_indices_or_sections::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & tensor_indices_or_sections, int64_t dim) {
    static auto op = create_tensor_split_tensor_indices_or_sections_typed_handle();
    return op.redispatch(dispatchKeySet, self, tensor_indices_or_sections, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clip_, name, "aten::clip_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clip_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clip_, schema_str, "clip_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor(a!)")

// aten::clip_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<clip_::schema> create_clip__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(clip_::name, clip_::overload_name)
      .typed<clip_::schema>();
}

// aten::clip_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor(a!)
at::Tensor & clip_::call(at::Tensor & self, const c10::optional<at::Scalar> & min, const c10::optional<at::Scalar> & max) {
    static auto op = create_clip__typed_handle();
    return op.call(self, min, max);
}

// aten::clip_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor(a!)
at::Tensor & clip_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const c10::optional<at::Scalar> & min, const c10::optional<at::Scalar> & max) {
    static auto op = create_clip__typed_handle();
    return op.redispatch(dispatchKeySet, self, min, max);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(polar_out, name, "aten::polar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(polar_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(polar_out, schema_str, "polar.out(Tensor abs, Tensor angle, *, Tensor(a!) out) -> Tensor(a!)")

// aten::polar.out(Tensor abs, Tensor angle, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<polar_out::schema> create_polar_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(polar_out::name, polar_out::overload_name)
      .typed<polar_out::schema>();
}

// aten::polar.out(Tensor abs, Tensor angle, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & polar_out::call(const at::Tensor & abs, const at::Tensor & angle, at::Tensor & out) {
    static auto op = create_polar_out_typed_handle();
    return op.call(abs, angle, out);
}

// aten::polar.out(Tensor abs, Tensor angle, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & polar_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & abs, const at::Tensor & angle, at::Tensor & out) {
    static auto op = create_polar_out_typed_handle();
    return op.redispatch(dispatchKeySet, abs, angle, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv_tbc, name, "aten::conv_tbc")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv_tbc, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv_tbc, schema_str, "conv_tbc(Tensor self, Tensor weight, Tensor bias, int pad=0) -> Tensor")

// aten::conv_tbc(Tensor self, Tensor weight, Tensor bias, int pad=0) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<conv_tbc::schema> create_conv_tbc_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(conv_tbc::name, conv_tbc::overload_name)
      .typed<conv_tbc::schema>();
}

// aten::conv_tbc(Tensor self, Tensor weight, Tensor bias, int pad=0) -> Tensor
at::Tensor conv_tbc::call(const at::Tensor & self, const at::Tensor & weight, const at::Tensor & bias, int64_t pad) {
    static auto op = create_conv_tbc_typed_handle();
    return op.call(self, weight, bias, pad);
}

// aten::conv_tbc(Tensor self, Tensor weight, Tensor bias, int pad=0) -> Tensor
at::Tensor conv_tbc::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, const at::Tensor & bias, int64_t pad) {
    static auto op = create_conv_tbc_typed_handle();
    return op.redispatch(dispatchKeySet, self, weight, bias, pad);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv_transpose2d_input, name, "aten::conv_transpose2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv_transpose2d_input, overload_name, "input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv_transpose2d_input, schema_str, "conv_transpose2d.input(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] output_padding=0, int groups=1, int[2] dilation=1) -> Tensor")

// aten::conv_transpose2d.input(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] output_padding=0, int groups=1, int[2] dilation=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<conv_transpose2d_input::schema> create_conv_transpose2d_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(conv_transpose2d_input::name, conv_transpose2d_input::overload_name)
      .typed<conv_transpose2d_input::schema>();
}

// aten::conv_transpose2d.input(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] output_padding=0, int groups=1, int[2] dilation=1) -> Tensor
at::Tensor conv_transpose2d_input::call(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, int64_t groups, at::IntArrayRef dilation) {
    static auto op = create_conv_transpose2d_input_typed_handle();
    return op.call(input, weight, bias, stride, padding, output_padding, groups, dilation);
}

// aten::conv_transpose2d.input(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] output_padding=0, int groups=1, int[2] dilation=1) -> Tensor
at::Tensor conv_transpose2d_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, int64_t groups, at::IntArrayRef dilation) {
    static auto op = create_conv_transpose2d_input_typed_handle();
    return op.redispatch(dispatchKeySet, input, weight, bias, stride, padding, output_padding, groups, dilation);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(copy_, name, "aten::copy_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(copy_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(copy_, schema_str, "copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)")

// aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<copy_::schema> create_copy__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(copy_::name, copy_::overload_name)
      .typed<copy_::schema>();
}

// aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)
at::Tensor & copy_::call(at::Tensor & self, const at::Tensor & src, bool non_blocking) {
    static auto op = create_copy__typed_handle();
    return op.call(self, src, non_blocking);
}

// aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)
at::Tensor & copy_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & src, bool non_blocking) {
    static auto op = create_copy__typed_handle();
    return op.redispatch(dispatchKeySet, self, src, non_blocking);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cosh, name, "aten::cosh")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cosh, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cosh, schema_str, "cosh(Tensor self) -> Tensor")

// aten::cosh(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<cosh::schema> create_cosh_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cosh::name, cosh::overload_name)
      .typed<cosh::schema>();
}

// aten::cosh(Tensor self) -> Tensor
at::Tensor cosh::call(const at::Tensor & self) {
    static auto op = create_cosh_typed_handle();
    return op.call(self);
}

// aten::cosh(Tensor self) -> Tensor
at::Tensor cosh::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_cosh_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cosine_embedding_loss, name, "aten::cosine_embedding_loss")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cosine_embedding_loss, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cosine_embedding_loss, schema_str, "cosine_embedding_loss(Tensor input1, Tensor input2, Tensor target, float margin=0.0, int reduction=Mean) -> Tensor")

// aten::cosine_embedding_loss(Tensor input1, Tensor input2, Tensor target, float margin=0.0, int reduction=Mean) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<cosine_embedding_loss::schema> create_cosine_embedding_loss_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cosine_embedding_loss::name, cosine_embedding_loss::overload_name)
      .typed<cosine_embedding_loss::schema>();
}

// aten::cosine_embedding_loss(Tensor input1, Tensor input2, Tensor target, float margin=0.0, int reduction=Mean) -> Tensor
at::Tensor cosine_embedding_loss::call(const at::Tensor & input1, const at::Tensor & input2, const at::Tensor & target, double margin, int64_t reduction) {
    static auto op = create_cosine_embedding_loss_typed_handle();
    return op.call(input1, input2, target, margin, reduction);
}

// aten::cosine_embedding_loss(Tensor input1, Tensor input2, Tensor target, float margin=0.0, int reduction=Mean) -> Tensor
at::Tensor cosine_embedding_loss::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input1, const at::Tensor & input2, const at::Tensor & target, double margin, int64_t reduction) {
    static auto op = create_cosine_embedding_loss_typed_handle();
    return op.redispatch(dispatchKeySet, input1, input2, target, margin, reduction);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(count_nonzero_dim_IntList, name, "aten::count_nonzero")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(count_nonzero_dim_IntList, overload_name, "dim_IntList")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(count_nonzero_dim_IntList, schema_str, "count_nonzero.dim_IntList(Tensor self, int[] dim) -> Tensor")

// aten::count_nonzero.dim_IntList(Tensor self, int[] dim) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<count_nonzero_dim_IntList::schema> create_count_nonzero_dim_IntList_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(count_nonzero_dim_IntList::name, count_nonzero_dim_IntList::overload_name)
      .typed<count_nonzero_dim_IntList::schema>();
}

// aten::count_nonzero.dim_IntList(Tensor self, int[] dim) -> Tensor
at::Tensor count_nonzero_dim_IntList::call(const at::Tensor & self, at::IntArrayRef dim) {
    static auto op = create_count_nonzero_dim_IntList_typed_handle();
    return op.call(self, dim);
}

// aten::count_nonzero.dim_IntList(Tensor self, int[] dim) -> Tensor
at::Tensor count_nonzero_dim_IntList::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim) {
    static auto op = create_count_nonzero_dim_IntList_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_affine_grid_generator_backward, name, "aten::cudnn_affine_grid_generator_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_affine_grid_generator_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_affine_grid_generator_backward, schema_str, "cudnn_affine_grid_generator_backward(Tensor grad, int N, int C, int H, int W) -> Tensor grad_theta")

// aten::cudnn_affine_grid_generator_backward(Tensor grad, int N, int C, int H, int W) -> Tensor grad_theta
static C10_NOINLINE c10::TypedOperatorHandle<cudnn_affine_grid_generator_backward::schema> create_cudnn_affine_grid_generator_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cudnn_affine_grid_generator_backward::name, cudnn_affine_grid_generator_backward::overload_name)
      .typed<cudnn_affine_grid_generator_backward::schema>();
}

// aten::cudnn_affine_grid_generator_backward(Tensor grad, int N, int C, int H, int W) -> Tensor grad_theta
at::Tensor cudnn_affine_grid_generator_backward::call(const at::Tensor & grad, int64_t N, int64_t C, int64_t H, int64_t W) {
    static auto op = create_cudnn_affine_grid_generator_backward_typed_handle();
    return op.call(grad, N, C, H, W);
}

// aten::cudnn_affine_grid_generator_backward(Tensor grad, int N, int C, int H, int W) -> Tensor grad_theta
at::Tensor cudnn_affine_grid_generator_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, int64_t N, int64_t C, int64_t H, int64_t W) {
    static auto op = create_cudnn_affine_grid_generator_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad, N, C, H, W);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_deprecated2, name, "aten::cudnn_convolution")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_deprecated2, overload_name, "deprecated2")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_deprecated2, schema_str, "cudnn_convolution.deprecated2(Tensor self, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor")

// aten::cudnn_convolution.deprecated2(Tensor self, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<cudnn_convolution_deprecated2::schema> create_cudnn_convolution_deprecated2_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cudnn_convolution_deprecated2::name, cudnn_convolution_deprecated2::overload_name)
      .typed<cudnn_convolution_deprecated2::schema>();
}

// aten::cudnn_convolution.deprecated2(Tensor self, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
at::Tensor cudnn_convolution_deprecated2::call(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
    static auto op = create_cudnn_convolution_deprecated2_typed_handle();
    return op.call(self, weight, padding, stride, dilation, groups, benchmark, deterministic);
}

// aten::cudnn_convolution.deprecated2(Tensor self, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
at::Tensor cudnn_convolution_deprecated2::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
    static auto op = create_cudnn_convolution_deprecated2_typed_handle();
    return op.redispatch(dispatchKeySet, self, weight, padding, stride, dilation, groups, benchmark, deterministic);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_backward_input, name, "aten::cudnn_convolution_backward_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_backward_input, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_backward_input, schema_str, "cudnn_convolution_backward_input(int[] self_size, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool allow_tf32) -> Tensor")

// aten::cudnn_convolution_backward_input(int[] self_size, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool allow_tf32) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<cudnn_convolution_backward_input::schema> create_cudnn_convolution_backward_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cudnn_convolution_backward_input::name, cudnn_convolution_backward_input::overload_name)
      .typed<cudnn_convolution_backward_input::schema>();
}

// aten::cudnn_convolution_backward_input(int[] self_size, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool allow_tf32) -> Tensor
at::Tensor cudnn_convolution_backward_input::call(at::IntArrayRef self_size, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
    static auto op = create_cudnn_convolution_backward_input_typed_handle();
    return op.call(self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
}

// aten::cudnn_convolution_backward_input(int[] self_size, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool allow_tf32) -> Tensor
at::Tensor cudnn_convolution_backward_input::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef self_size, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
    static auto op = create_cudnn_convolution_backward_input_typed_handle();
    return op.redispatch(dispatchKeySet, self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_backward, name, "aten::cudnn_convolution_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_backward, schema_str, "cudnn_convolution_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool allow_tf32, bool[2] output_mask) -> (Tensor, Tensor)")

// aten::cudnn_convolution_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool allow_tf32, bool[2] output_mask) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<cudnn_convolution_backward::schema> create_cudnn_convolution_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cudnn_convolution_backward::name, cudnn_convolution_backward::overload_name)
      .typed<cudnn_convolution_backward::schema>();
}

// aten::cudnn_convolution_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool allow_tf32, bool[2] output_mask) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> cudnn_convolution_backward::call(const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32, ::std::array<bool,2> output_mask) {
    static auto op = create_cudnn_convolution_backward_typed_handle();
    return op.call(self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32, output_mask);
}

// aten::cudnn_convolution_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool allow_tf32, bool[2] output_mask) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> cudnn_convolution_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32, ::std::array<bool,2> output_mask) {
    static auto op = create_cudnn_convolution_backward_typed_handle();
    return op.redispatch(dispatchKeySet, self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32, output_mask);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_grid_sampler, name, "aten::cudnn_grid_sampler")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_grid_sampler, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_grid_sampler, schema_str, "cudnn_grid_sampler(Tensor self, Tensor grid) -> Tensor output")

// aten::cudnn_grid_sampler(Tensor self, Tensor grid) -> Tensor output
static C10_NOINLINE c10::TypedOperatorHandle<cudnn_grid_sampler::schema> create_cudnn_grid_sampler_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cudnn_grid_sampler::name, cudnn_grid_sampler::overload_name)
      .typed<cudnn_grid_sampler::schema>();
}

// aten::cudnn_grid_sampler(Tensor self, Tensor grid) -> Tensor output
at::Tensor cudnn_grid_sampler::call(const at::Tensor & self, const at::Tensor & grid) {
    static auto op = create_cudnn_grid_sampler_typed_handle();
    return op.call(self, grid);
}

// aten::cudnn_grid_sampler(Tensor self, Tensor grid) -> Tensor output
at::Tensor cudnn_grid_sampler::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & grid) {
    static auto op = create_cudnn_grid_sampler_typed_handle();
    return op.redispatch(dispatchKeySet, self, grid);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cummin, name, "aten::cummin")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cummin, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cummin, schema_str, "cummin(Tensor self, int dim) -> (Tensor values, Tensor indices)")

// aten::cummin(Tensor self, int dim) -> (Tensor values, Tensor indices)
static C10_NOINLINE c10::TypedOperatorHandle<cummin::schema> create_cummin_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cummin::name, cummin::overload_name)
      .typed<cummin::schema>();
}

// aten::cummin(Tensor self, int dim) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> cummin::call(const at::Tensor & self, int64_t dim) {
    static auto op = create_cummin_typed_handle();
    return op.call(self, dim);
}

// aten::cummin(Tensor self, int dim) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> cummin::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim) {
    static auto op = create_cummin_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cummin_helper, name, "aten::_cummin_helper")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cummin_helper, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cummin_helper, schema_str, "_cummin_helper(Tensor self, Tensor(a!) values, Tensor(b!) indices, int dim) -> ()")

// aten::_cummin_helper(Tensor self, Tensor(a!) values, Tensor(b!) indices, int dim) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_cummin_helper::schema> create__cummin_helper_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_cummin_helper::name, _cummin_helper::overload_name)
      .typed<_cummin_helper::schema>();
}

// aten::_cummin_helper(Tensor self, Tensor(a!) values, Tensor(b!) indices, int dim) -> ()
void _cummin_helper::call(const at::Tensor & self, at::Tensor & values, at::Tensor & indices, int64_t dim) {
    static auto op = create__cummin_helper_typed_handle();
    return op.call(self, values, indices, dim);
}

// aten::_cummin_helper(Tensor self, Tensor(a!) values, Tensor(b!) indices, int dim) -> ()
void _cummin_helper::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & values, at::Tensor & indices, int64_t dim) {
    static auto op = create__cummin_helper_typed_handle();
    return op.redispatch(dispatchKeySet, self, values, indices, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumprod_out, name, "aten::cumprod")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumprod_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumprod_out, schema_str, "cumprod.out(Tensor self, int dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)")

// aten::cumprod.out(Tensor self, int dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<cumprod_out::schema> create_cumprod_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cumprod_out::name, cumprod_out::overload_name)
      .typed<cumprod_out::schema>();
}

// aten::cumprod.out(Tensor self, int dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & cumprod_out::call(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
    static auto op = create_cumprod_out_typed_handle();
    return op.call(self, dim, dtype, out);
}

// aten::cumprod.out(Tensor self, int dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & cumprod_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
    static auto op = create_cumprod_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, dtype, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(diff_out, name, "aten::diff")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(diff_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(diff_out, schema_str, "diff.out(Tensor self, int n=1, int dim=-1, Tensor? prepend=None, Tensor? append=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::diff.out(Tensor self, int n=1, int dim=-1, Tensor? prepend=None, Tensor? append=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<diff_out::schema> create_diff_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(diff_out::name, diff_out::overload_name)
      .typed<diff_out::schema>();
}

// aten::diff.out(Tensor self, int n=1, int dim=-1, Tensor? prepend=None, Tensor? append=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & diff_out::call(const at::Tensor & self, int64_t n, int64_t dim, const c10::optional<at::Tensor> & prepend, const c10::optional<at::Tensor> & append, at::Tensor & out) {
    static auto op = create_diff_out_typed_handle();
    return op.call(self, n, dim, prepend, append, out);
}

// aten::diff.out(Tensor self, int n=1, int dim=-1, Tensor? prepend=None, Tensor? append=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & diff_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t n, int64_t dim, const c10::optional<at::Tensor> & prepend, const c10::optional<at::Tensor> & append, at::Tensor & out) {
    static auto op = create_diff_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, n, dim, prepend, append, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gradient_array, name, "aten::gradient")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gradient_array, overload_name, "array")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gradient_array, schema_str, "gradient.array(Tensor self, *, int[] dim, int edge_order=1) -> Tensor[]")

// aten::gradient.array(Tensor self, *, int[] dim, int edge_order=1) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<gradient_array::schema> create_gradient_array_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(gradient_array::name, gradient_array::overload_name)
      .typed<gradient_array::schema>();
}

// aten::gradient.array(Tensor self, *, int[] dim, int edge_order=1) -> Tensor[]
::std::vector<at::Tensor> gradient_array::call(const at::Tensor & self, at::IntArrayRef dim, int64_t edge_order) {
    static auto op = create_gradient_array_typed_handle();
    return op.call(self, dim, edge_order);
}

// aten::gradient.array(Tensor self, *, int[] dim, int edge_order=1) -> Tensor[]
::std::vector<at::Tensor> gradient_array::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, int64_t edge_order) {
    static auto op = create_gradient_array_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, edge_order);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gradient_scalarrayarray, name, "aten::gradient")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gradient_scalarrayarray, overload_name, "scalarrayarray")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gradient_scalarrayarray, schema_str, "gradient.scalarrayarray(Tensor self, *, Scalar[] spacing, int[] dim, int edge_order=1) -> Tensor[]")

// aten::gradient.scalarrayarray(Tensor self, *, Scalar[] spacing, int[] dim, int edge_order=1) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<gradient_scalarrayarray::schema> create_gradient_scalarrayarray_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(gradient_scalarrayarray::name, gradient_scalarrayarray::overload_name)
      .typed<gradient_scalarrayarray::schema>();
}

// aten::gradient.scalarrayarray(Tensor self, *, Scalar[] spacing, int[] dim, int edge_order=1) -> Tensor[]
::std::vector<at::Tensor> gradient_scalarrayarray::call(const at::Tensor & self, at::ArrayRef<at::Scalar> spacing, at::IntArrayRef dim, int64_t edge_order) {
    static auto op = create_gradient_scalarrayarray_typed_handle();
    return op.call(self, spacing, dim, edge_order);
}

// aten::gradient.scalarrayarray(Tensor self, *, Scalar[] spacing, int[] dim, int edge_order=1) -> Tensor[]
::std::vector<at::Tensor> gradient_scalarrayarray::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::ArrayRef<at::Scalar> spacing, at::IntArrayRef dim, int64_t edge_order) {
    static auto op = create_gradient_scalarrayarray_typed_handle();
    return op.redispatch(dispatchKeySet, self, spacing, dim, edge_order);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gradient_tensorarray, name, "aten::gradient")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gradient_tensorarray, overload_name, "tensorarray")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gradient_tensorarray, schema_str, "gradient.tensorarray(Tensor self, *, Tensor[] spacing, int[] dim, int edge_order=1) -> Tensor[]")

// aten::gradient.tensorarray(Tensor self, *, Tensor[] spacing, int[] dim, int edge_order=1) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<gradient_tensorarray::schema> create_gradient_tensorarray_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(gradient_tensorarray::name, gradient_tensorarray::overload_name)
      .typed<gradient_tensorarray::schema>();
}

// aten::gradient.tensorarray(Tensor self, *, Tensor[] spacing, int[] dim, int edge_order=1) -> Tensor[]
::std::vector<at::Tensor> gradient_tensorarray::call(const at::Tensor & self, at::TensorList spacing, at::IntArrayRef dim, int64_t edge_order) {
    static auto op = create_gradient_tensorarray_typed_handle();
    return op.call(self, spacing, dim, edge_order);
}

// aten::gradient.tensorarray(Tensor self, *, Tensor[] spacing, int[] dim, int edge_order=1) -> Tensor[]
::std::vector<at::Tensor> gradient_tensorarray::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::TensorList spacing, at::IntArrayRef dim, int64_t edge_order) {
    static auto op = create_gradient_tensorarray_typed_handle();
    return op.redispatch(dispatchKeySet, self, spacing, dim, edge_order);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(div__Tensor, name, "aten::div_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(div__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(div__Tensor, schema_str, "div_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::div_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<div__Tensor::schema> create_div__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(div__Tensor::name, div__Tensor::overload_name)
      .typed<div__Tensor::schema>();
}

// aten::div_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & div__Tensor::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create_div__Tensor_typed_handle();
    return op.call(self, other);
}

// aten::div_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & div__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create_div__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(div_Scalar, name, "aten::div")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(div_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(div_Scalar, schema_str, "div.Scalar(Tensor self, Scalar other) -> Tensor")

// aten::div.Scalar(Tensor self, Scalar other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<div_Scalar::schema> create_div_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(div_Scalar::name, div_Scalar::overload_name)
      .typed<div_Scalar::schema>();
}

// aten::div.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor div_Scalar::call(const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_div_Scalar_typed_handle();
    return op.call(self, other);
}

// aten::div.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor div_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_div_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(div_Scalar_mode, name, "aten::div")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(div_Scalar_mode, overload_name, "Scalar_mode")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(div_Scalar_mode, schema_str, "div.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> Tensor")

// aten::div.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<div_Scalar_mode::schema> create_div_Scalar_mode_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(div_Scalar_mode::name, div_Scalar_mode::overload_name)
      .typed<div_Scalar_mode::schema>();
}

// aten::div.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> Tensor
at::Tensor div_Scalar_mode::call(const at::Tensor & self, const at::Scalar & other, c10::optional<c10::string_view> rounding_mode) {
    static auto op = create_div_Scalar_mode_typed_handle();
    return op.call(self, other, rounding_mode);
}

// aten::div.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> Tensor
at::Tensor div_Scalar_mode::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, c10::optional<c10::string_view> rounding_mode) {
    static auto op = create_div_Scalar_mode_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, rounding_mode);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(div__Scalar_mode, name, "aten::div_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(div__Scalar_mode, overload_name, "Scalar_mode")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(div__Scalar_mode, schema_str, "div_.Scalar_mode(Tensor(a!) self, Scalar other, *, str? rounding_mode) -> Tensor(a!)")

// aten::div_.Scalar_mode(Tensor(a!) self, Scalar other, *, str? rounding_mode) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<div__Scalar_mode::schema> create_div__Scalar_mode_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(div__Scalar_mode::name, div__Scalar_mode::overload_name)
      .typed<div__Scalar_mode::schema>();
}

// aten::div_.Scalar_mode(Tensor(a!) self, Scalar other, *, str? rounding_mode) -> Tensor(a!)
at::Tensor & div__Scalar_mode::call(at::Tensor & self, const at::Scalar & other, c10::optional<c10::string_view> rounding_mode) {
    static auto op = create_div__Scalar_mode_typed_handle();
    return op.call(self, other, rounding_mode);
}

// aten::div_.Scalar_mode(Tensor(a!) self, Scalar other, *, str? rounding_mode) -> Tensor(a!)
at::Tensor & div__Scalar_mode::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other, c10::optional<c10::string_view> rounding_mode) {
    static auto op = create_div__Scalar_mode_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, rounding_mode);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(divide_Tensor, name, "aten::divide")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(divide_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(divide_Tensor, schema_str, "divide.Tensor(Tensor self, Tensor other) -> Tensor")

// aten::divide.Tensor(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<divide_Tensor::schema> create_divide_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(divide_Tensor::name, divide_Tensor::overload_name)
      .typed<divide_Tensor::schema>();
}

// aten::divide.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor divide_Tensor::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_divide_Tensor_typed_handle();
    return op.call(self, other);
}

// aten::divide.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor divide_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_divide_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(divide_out_mode, name, "aten::divide")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(divide_out_mode, overload_name, "out_mode")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(divide_out_mode, schema_str, "divide.out_mode(Tensor self, Tensor other, *, str? rounding_mode, Tensor(a!) out) -> Tensor(a!)")

// aten::divide.out_mode(Tensor self, Tensor other, *, str? rounding_mode, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<divide_out_mode::schema> create_divide_out_mode_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(divide_out_mode::name, divide_out_mode::overload_name)
      .typed<divide_out_mode::schema>();
}

// aten::divide.out_mode(Tensor self, Tensor other, *, str? rounding_mode, Tensor(a!) out) -> Tensor(a!)
at::Tensor & divide_out_mode::call(const at::Tensor & self, const at::Tensor & other, c10::optional<c10::string_view> rounding_mode, at::Tensor & out) {
    static auto op = create_divide_out_mode_typed_handle();
    return op.call(self, other, rounding_mode, out);
}

// aten::divide.out_mode(Tensor self, Tensor other, *, str? rounding_mode, Tensor(a!) out) -> Tensor(a!)
at::Tensor & divide_out_mode::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, c10::optional<c10::string_view> rounding_mode, at::Tensor & out) {
    static auto op = create_divide_out_mode_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, rounding_mode, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(true_divide_Tensor, name, "aten::true_divide")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(true_divide_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(true_divide_Tensor, schema_str, "true_divide.Tensor(Tensor self, Tensor other) -> Tensor")

// aten::true_divide.Tensor(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<true_divide_Tensor::schema> create_true_divide_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(true_divide_Tensor::name, true_divide_Tensor::overload_name)
      .typed<true_divide_Tensor::schema>();
}

// aten::true_divide.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor true_divide_Tensor::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_true_divide_Tensor_typed_handle();
    return op.call(self, other);
}

// aten::true_divide.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor true_divide_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_true_divide_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(true_divide_Scalar, name, "aten::true_divide")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(true_divide_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(true_divide_Scalar, schema_str, "true_divide.Scalar(Tensor self, Scalar other) -> Tensor")

// aten::true_divide.Scalar(Tensor self, Scalar other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<true_divide_Scalar::schema> create_true_divide_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(true_divide_Scalar::name, true_divide_Scalar::overload_name)
      .typed<true_divide_Scalar::schema>();
}

// aten::true_divide.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor true_divide_Scalar::call(const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_true_divide_Scalar_typed_handle();
    return op.call(self, other);
}

// aten::true_divide.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor true_divide_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_true_divide_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_embedding_bag_forward_only, name, "aten::_embedding_bag_forward_only")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_embedding_bag_forward_only, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_embedding_bag_forward_only, schema_str, "_embedding_bag_forward_only(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int padding_idx=-1) -> (Tensor, Tensor, Tensor, Tensor)")

// aten::_embedding_bag_forward_only(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int padding_idx=-1) -> (Tensor, Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<_embedding_bag_forward_only::schema> create__embedding_bag_forward_only_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_embedding_bag_forward_only::name, _embedding_bag_forward_only::overload_name)
      .typed<_embedding_bag_forward_only::schema>();
}

// aten::_embedding_bag_forward_only(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int padding_idx=-1) -> (Tensor, Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> _embedding_bag_forward_only::call(const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<at::Tensor> & per_sample_weights, bool include_last_offset, int64_t padding_idx) {
    static auto op = create__embedding_bag_forward_only_typed_handle();
    return op.call(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx);
}

// aten::_embedding_bag_forward_only(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int padding_idx=-1) -> (Tensor, Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> _embedding_bag_forward_only::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<at::Tensor> & per_sample_weights, bool include_last_offset, int64_t padding_idx) {
    static auto op = create__embedding_bag_forward_only_typed_handle();
    return op.redispatch(dispatchKeySet, weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(embedding_bag, name, "aten::embedding_bag")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(embedding_bag, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(embedding_bag, schema_str, "embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False) -> (Tensor, Tensor, Tensor, Tensor)")

// aten::embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False) -> (Tensor, Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<embedding_bag::schema> create_embedding_bag_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(embedding_bag::name, embedding_bag::overload_name)
      .typed<embedding_bag::schema>();
}

// aten::embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False) -> (Tensor, Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> embedding_bag::call(const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<at::Tensor> & per_sample_weights, bool include_last_offset) {
    static auto op = create_embedding_bag_typed_handle();
    return op.call(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset);
}

// aten::embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False) -> (Tensor, Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> embedding_bag::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<at::Tensor> & per_sample_weights, bool include_last_offset) {
    static auto op = create_embedding_bag_typed_handle();
    return op.redispatch(dispatchKeySet, weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(empty_memory_format, name, "aten::empty")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(empty_memory_format, overload_name, "memory_format")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(empty_memory_format, schema_str, "empty.memory_format(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor")

// aten::empty.memory_format(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<empty_memory_format::schema> create_empty_memory_format_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(empty_memory_format::name, empty_memory_format::overload_name)
      .typed<empty_memory_format::schema>();
}

// aten::empty.memory_format(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
at::Tensor empty_memory_format::call(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create_empty_memory_format_typed_handle();
    return op.call(size, dtype, layout, device, pin_memory, memory_format);
}

// aten::empty.memory_format(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
at::Tensor empty_memory_format::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create_empty_memory_format_typed_handle();
    return op.redispatch(dispatchKeySet, size, dtype, layout, device, pin_memory, memory_format);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(new_zeros, name, "aten::new_zeros")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(new_zeros, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(new_zeros, schema_str, "new_zeros(Tensor self, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::new_zeros(Tensor self, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<new_zeros::schema> create_new_zeros_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(new_zeros::name, new_zeros::overload_name)
      .typed<new_zeros::schema>();
}

// aten::new_zeros(Tensor self, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor new_zeros::call(const at::Tensor & self, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_new_zeros_typed_handle();
    return op.call(self, size, dtype, layout, device, pin_memory);
}

// aten::new_zeros(Tensor self, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor new_zeros::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_new_zeros_typed_handle();
    return op.redispatch(dispatchKeySet, self, size, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(erf, name, "aten::erf")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(erf, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(erf, schema_str, "erf(Tensor self) -> Tensor")

// aten::erf(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<erf::schema> create_erf_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(erf::name, erf::overload_name)
      .typed<erf::schema>();
}

// aten::erf(Tensor self) -> Tensor
at::Tensor erf::call(const at::Tensor & self) {
    static auto op = create_erf_typed_handle();
    return op.call(self);
}

// aten::erf(Tensor self) -> Tensor
at::Tensor erf::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_erf_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(erfc_, name, "aten::erfc_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(erfc_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(erfc_, schema_str, "erfc_(Tensor(a!) self) -> Tensor(a!)")

// aten::erfc_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<erfc_::schema> create_erfc__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(erfc_::name, erfc_::overload_name)
      .typed<erfc_::schema>();
}

// aten::erfc_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & erfc_::call(at::Tensor & self) {
    static auto op = create_erfc__typed_handle();
    return op.call(self);
}

// aten::erfc_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & erfc_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_erfc__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(expm1_out, name, "aten::expm1")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(expm1_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(expm1_out, schema_str, "expm1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::expm1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<expm1_out::schema> create_expm1_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(expm1_out::name, expm1_out::overload_name)
      .typed<expm1_out::schema>();
}

// aten::expm1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & expm1_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_expm1_out_typed_handle();
    return op.call(self, out);
}

// aten::expm1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & expm1_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_expm1_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(eye_m, name, "aten::eye")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(eye_m, overload_name, "m")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(eye_m, schema_str, "eye.m(int n, int m, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::eye.m(int n, int m, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<eye_m::schema> create_eye_m_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(eye_m::name, eye_m::overload_name)
      .typed<eye_m::schema>();
}

// aten::eye.m(int n, int m, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor eye_m::call(int64_t n, int64_t m, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_eye_m_typed_handle();
    return op.call(n, m, dtype, layout, device, pin_memory);
}

// aten::eye.m(int n, int m, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor eye_m::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t n, int64_t m, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_eye_m_typed_handle();
    return op.redispatch(dispatchKeySet, n, m, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(flatten_using_ints, name, "aten::flatten")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(flatten_using_ints, overload_name, "using_ints")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(flatten_using_ints, schema_str, "flatten.using_ints(Tensor(a) self, int start_dim=0, int end_dim=-1) -> Tensor(a)")

// aten::flatten.using_ints(Tensor(a) self, int start_dim=0, int end_dim=-1) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<flatten_using_ints::schema> create_flatten_using_ints_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(flatten_using_ints::name, flatten_using_ints::overload_name)
      .typed<flatten_using_ints::schema>();
}

// aten::flatten.using_ints(Tensor(a) self, int start_dim=0, int end_dim=-1) -> Tensor(a)
at::Tensor flatten_using_ints::call(const at::Tensor & self, int64_t start_dim, int64_t end_dim) {
    static auto op = create_flatten_using_ints_typed_handle();
    return op.call(self, start_dim, end_dim);
}

// aten::flatten.using_ints(Tensor(a) self, int start_dim=0, int end_dim=-1) -> Tensor(a)
at::Tensor flatten_using_ints::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t start_dim, int64_t end_dim) {
    static auto op = create_flatten_using_ints_typed_handle();
    return op.redispatch(dispatchKeySet, self, start_dim, end_dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unflatten_int, name, "aten::unflatten")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unflatten_int, overload_name, "int")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unflatten_int, schema_str, "unflatten.int(Tensor(a) self, int dim, int[] sizes, Dimname[]? names=None) -> Tensor(a)")

// aten::unflatten.int(Tensor(a) self, int dim, int[] sizes, Dimname[]? names=None) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<unflatten_int::schema> create_unflatten_int_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(unflatten_int::name, unflatten_int::overload_name)
      .typed<unflatten_int::schema>();
}

// aten::unflatten.int(Tensor(a) self, int dim, int[] sizes, Dimname[]? names=None) -> Tensor(a)
at::Tensor unflatten_int::call(const at::Tensor & self, int64_t dim, at::IntArrayRef sizes, c10::optional<at::DimnameList> names) {
    static auto op = create_unflatten_int_typed_handle();
    return op.call(self, dim, sizes, names);
}

// aten::unflatten.int(Tensor(a) self, int dim, int[] sizes, Dimname[]? names=None) -> Tensor(a)
at::Tensor unflatten_int::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, at::IntArrayRef sizes, c10::optional<at::DimnameList> names) {
    static auto op = create_unflatten_int_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, sizes, names);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unflatten_Dimname, name, "aten::unflatten")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unflatten_Dimname, overload_name, "Dimname")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unflatten_Dimname, schema_str, "unflatten.Dimname(Tensor(a) self, Dimname dim, int[] sizes, Dimname[] names) -> Tensor(a)")

// aten::unflatten.Dimname(Tensor(a) self, Dimname dim, int[] sizes, Dimname[] names) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<unflatten_Dimname::schema> create_unflatten_Dimname_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(unflatten_Dimname::name, unflatten_Dimname::overload_name)
      .typed<unflatten_Dimname::schema>();
}

// aten::unflatten.Dimname(Tensor(a) self, Dimname dim, int[] sizes, Dimname[] names) -> Tensor(a)
at::Tensor unflatten_Dimname::call(const at::Tensor & self, at::Dimname dim, at::IntArrayRef sizes, at::DimnameList names) {
    static auto op = create_unflatten_Dimname_typed_handle();
    return op.call(self, dim, sizes, names);
}

// aten::unflatten.Dimname(Tensor(a) self, Dimname dim, int[] sizes, Dimname[] names) -> Tensor(a)
at::Tensor unflatten_Dimname::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, at::IntArrayRef sizes, at::DimnameList names) {
    static auto op = create_unflatten_Dimname_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, sizes, names);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(frac_, name, "aten::frac_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(frac_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(frac_, schema_str, "frac_(Tensor(a!) self) -> Tensor(a!)")

// aten::frac_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<frac_::schema> create_frac__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(frac_::name, frac_::overload_name)
      .typed<frac_::schema>();
}

// aten::frac_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & frac_::call(at::Tensor & self) {
    static auto op = create_frac__typed_handle();
    return op.call(self);
}

// aten::frac_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & frac_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_frac__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(full_names, name, "aten::full")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(full_names, overload_name, "names")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(full_names, schema_str, "full.names(int[] size, Scalar fill_value, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::full.names(int[] size, Scalar fill_value, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<full_names::schema> create_full_names_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(full_names::name, full_names::overload_name)
      .typed<full_names::schema>();
}

// aten::full.names(int[] size, Scalar fill_value, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor full_names::call(at::IntArrayRef size, const at::Scalar & fill_value, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_full_names_typed_handle();
    return op.call(size, fill_value, names, dtype, layout, device, pin_memory);
}

// aten::full.names(int[] size, Scalar fill_value, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor full_names::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, const at::Scalar & fill_value, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_full_names_typed_handle();
    return op.redispatch(dispatchKeySet, size, fill_value, names, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gcd_out, name, "aten::gcd")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gcd_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gcd_out, schema_str, "gcd.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::gcd.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<gcd_out::schema> create_gcd_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(gcd_out::name, gcd_out::overload_name)
      .typed<gcd_out::schema>();
}

// aten::gcd.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & gcd_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_gcd_out_typed_handle();
    return op.call(self, other, out);
}

// aten::gcd.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & gcd_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_gcd_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lcm_out, name, "aten::lcm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lcm_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lcm_out, schema_str, "lcm.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::lcm.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<lcm_out::schema> create_lcm_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(lcm_out::name, lcm_out::overload_name)
      .typed<lcm_out::schema>();
}

// aten::lcm.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & lcm_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_lcm_out_typed_handle();
    return op.call(self, other, out);
}

// aten::lcm.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & lcm_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_lcm_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(grid_sampler, name, "aten::grid_sampler")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(grid_sampler, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(grid_sampler, schema_str, "grid_sampler(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor")

// aten::grid_sampler(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<grid_sampler::schema> create_grid_sampler_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(grid_sampler::name, grid_sampler::overload_name)
      .typed<grid_sampler::schema>();
}

// aten::grid_sampler(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor
at::Tensor grid_sampler::call(const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
    static auto op = create_grid_sampler_typed_handle();
    return op.call(input, grid, interpolation_mode, padding_mode, align_corners);
}

// aten::grid_sampler(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor
at::Tensor grid_sampler::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
    static auto op = create_grid_sampler_typed_handle();
    return op.redispatch(dispatchKeySet, input, grid, interpolation_mode, padding_mode, align_corners);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_grid_sampler_2d_cpu_fallback, name, "aten::_grid_sampler_2d_cpu_fallback")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_grid_sampler_2d_cpu_fallback, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_grid_sampler_2d_cpu_fallback, schema_str, "_grid_sampler_2d_cpu_fallback(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor")

// aten::_grid_sampler_2d_cpu_fallback(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_grid_sampler_2d_cpu_fallback::schema> create__grid_sampler_2d_cpu_fallback_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_grid_sampler_2d_cpu_fallback::name, _grid_sampler_2d_cpu_fallback::overload_name)
      .typed<_grid_sampler_2d_cpu_fallback::schema>();
}

// aten::_grid_sampler_2d_cpu_fallback(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor
at::Tensor _grid_sampler_2d_cpu_fallback::call(const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
    static auto op = create__grid_sampler_2d_cpu_fallback_typed_handle();
    return op.call(input, grid, interpolation_mode, padding_mode, align_corners);
}

// aten::_grid_sampler_2d_cpu_fallback(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor
at::Tensor _grid_sampler_2d_cpu_fallback::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
    static auto op = create__grid_sampler_2d_cpu_fallback_typed_handle();
    return op.redispatch(dispatchKeySet, input, grid, interpolation_mode, padding_mode, align_corners);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(grid_sampler_3d, name, "aten::grid_sampler_3d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(grid_sampler_3d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(grid_sampler_3d, schema_str, "grid_sampler_3d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor")

// aten::grid_sampler_3d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<grid_sampler_3d::schema> create_grid_sampler_3d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(grid_sampler_3d::name, grid_sampler_3d::overload_name)
      .typed<grid_sampler_3d::schema>();
}

// aten::grid_sampler_3d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor
at::Tensor grid_sampler_3d::call(const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
    static auto op = create_grid_sampler_3d_typed_handle();
    return op.call(input, grid, interpolation_mode, padding_mode, align_corners);
}

// aten::grid_sampler_3d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor
at::Tensor grid_sampler_3d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
    static auto op = create_grid_sampler_3d_typed_handle();
    return op.redispatch(dispatchKeySet, input, grid, interpolation_mode, padding_mode, align_corners);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hann_window, name, "aten::hann_window")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hann_window, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hann_window, schema_str, "hann_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::hann_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<hann_window::schema> create_hann_window_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(hann_window::name, hann_window::overload_name)
      .typed<hann_window::schema>();
}

// aten::hann_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor hann_window::call(int64_t window_length, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_hann_window_typed_handle();
    return op.call(window_length, dtype, layout, device, pin_memory);
}

// aten::hann_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor hann_window::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t window_length, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_hann_window_typed_handle();
    return op.redispatch(dispatchKeySet, window_length, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hann_window_periodic, name, "aten::hann_window")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hann_window_periodic, overload_name, "periodic")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hann_window_periodic, schema_str, "hann_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::hann_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<hann_window_periodic::schema> create_hann_window_periodic_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(hann_window_periodic::name, hann_window_periodic::overload_name)
      .typed<hann_window_periodic::schema>();
}

// aten::hann_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor hann_window_periodic::call(int64_t window_length, bool periodic, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_hann_window_periodic_typed_handle();
    return op.call(window_length, periodic, dtype, layout, device, pin_memory);
}

// aten::hann_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor hann_window_periodic::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t window_length, bool periodic, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_hann_window_periodic_typed_handle();
    return op.redispatch(dispatchKeySet, window_length, periodic, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hamming_window, name, "aten::hamming_window")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hamming_window, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hamming_window, schema_str, "hamming_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::hamming_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<hamming_window::schema> create_hamming_window_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(hamming_window::name, hamming_window::overload_name)
      .typed<hamming_window::schema>();
}

// aten::hamming_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor hamming_window::call(int64_t window_length, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_hamming_window_typed_handle();
    return op.call(window_length, dtype, layout, device, pin_memory);
}

// aten::hamming_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor hamming_window::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t window_length, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_hamming_window_typed_handle();
    return op.redispatch(dispatchKeySet, window_length, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(native_group_norm_backward, name, "aten::native_group_norm_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(native_group_norm_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(native_group_norm_backward, schema_str, "native_group_norm_backward(Tensor grad_out, Tensor input, Tensor mean, Tensor rstd, Tensor? weight, int N, int C, int HxW, int group, bool[3] output_mask) -> (Tensor, Tensor, Tensor)")

// aten::native_group_norm_backward(Tensor grad_out, Tensor input, Tensor mean, Tensor rstd, Tensor? weight, int N, int C, int HxW, int group, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<native_group_norm_backward::schema> create_native_group_norm_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(native_group_norm_backward::name, native_group_norm_backward::overload_name)
      .typed<native_group_norm_backward::schema>();
}

// aten::native_group_norm_backward(Tensor grad_out, Tensor input, Tensor mean, Tensor rstd, Tensor? weight, int N, int C, int HxW, int group, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_group_norm_backward::call(const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & rstd, const c10::optional<at::Tensor> & weight, int64_t N, int64_t C, int64_t HxW, int64_t group, ::std::array<bool,3> output_mask) {
    static auto op = create_native_group_norm_backward_typed_handle();
    return op.call(grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask);
}

// aten::native_group_norm_backward(Tensor grad_out, Tensor input, Tensor mean, Tensor rstd, Tensor? weight, int N, int C, int HxW, int group, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_group_norm_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & rstd, const c10::optional<at::Tensor> & weight, int64_t N, int64_t C, int64_t HxW, int64_t group, ::std::array<bool,3> output_mask) {
    static auto op = create_native_group_norm_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fft_c2c, name, "aten::_fft_c2c")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fft_c2c, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fft_c2c, schema_str, "_fft_c2c(Tensor self, int[] dim, int normalization, bool forward) -> Tensor")

// aten::_fft_c2c(Tensor self, int[] dim, int normalization, bool forward) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_fft_c2c::schema> create__fft_c2c_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_fft_c2c::name, _fft_c2c::overload_name)
      .typed<_fft_c2c::schema>();
}

// aten::_fft_c2c(Tensor self, int[] dim, int normalization, bool forward) -> Tensor
at::Tensor _fft_c2c::call(const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, bool forward) {
    static auto op = create__fft_c2c_typed_handle();
    return op.call(self, dim, normalization, forward);
}

// aten::_fft_c2c(Tensor self, int[] dim, int normalization, bool forward) -> Tensor
at::Tensor _fft_c2c::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, bool forward) {
    static auto op = create__fft_c2c_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, normalization, forward);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cufft_get_plan_cache_size, name, "aten::_cufft_get_plan_cache_size")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cufft_get_plan_cache_size, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cufft_get_plan_cache_size, schema_str, "_cufft_get_plan_cache_size(int device_index) -> int")

// aten::_cufft_get_plan_cache_size(int device_index) -> int
static C10_NOINLINE c10::TypedOperatorHandle<_cufft_get_plan_cache_size::schema> create__cufft_get_plan_cache_size_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_cufft_get_plan_cache_size::name, _cufft_get_plan_cache_size::overload_name)
      .typed<_cufft_get_plan_cache_size::schema>();
}

// aten::_cufft_get_plan_cache_size(int device_index) -> int
int64_t _cufft_get_plan_cache_size::call(int64_t device_index) {
    static auto op = create__cufft_get_plan_cache_size_typed_handle();
    return op.call(device_index);
}

// aten::_cufft_get_plan_cache_size(int device_index) -> int
int64_t _cufft_get_plan_cache_size::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t device_index) {
    static auto op = create__cufft_get_plan_cache_size_typed_handle();
    return op.redispatch(dispatchKeySet, device_index);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cufft_get_plan_cache_max_size, name, "aten::_cufft_get_plan_cache_max_size")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cufft_get_plan_cache_max_size, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cufft_get_plan_cache_max_size, schema_str, "_cufft_get_plan_cache_max_size(int device_index) -> int")

// aten::_cufft_get_plan_cache_max_size(int device_index) -> int
static C10_NOINLINE c10::TypedOperatorHandle<_cufft_get_plan_cache_max_size::schema> create__cufft_get_plan_cache_max_size_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_cufft_get_plan_cache_max_size::name, _cufft_get_plan_cache_max_size::overload_name)
      .typed<_cufft_get_plan_cache_max_size::schema>();
}

// aten::_cufft_get_plan_cache_max_size(int device_index) -> int
int64_t _cufft_get_plan_cache_max_size::call(int64_t device_index) {
    static auto op = create__cufft_get_plan_cache_max_size_typed_handle();
    return op.call(device_index);
}

// aten::_cufft_get_plan_cache_max_size(int device_index) -> int
int64_t _cufft_get_plan_cache_max_size::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t device_index) {
    static auto op = create__cufft_get_plan_cache_max_size_typed_handle();
    return op.redispatch(dispatchKeySet, device_index);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_copy__dimname, name, "aten::index_copy_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_copy__dimname, overload_name, "dimname")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_copy__dimname, schema_str, "index_copy_.dimname(Tensor(a!) self, Dimname dim, Tensor index, Tensor source) -> Tensor(a!)")

// aten::index_copy_.dimname(Tensor(a!) self, Dimname dim, Tensor index, Tensor source) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<index_copy__dimname::schema> create_index_copy__dimname_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(index_copy__dimname::name, index_copy__dimname::overload_name)
      .typed<index_copy__dimname::schema>();
}

// aten::index_copy_.dimname(Tensor(a!) self, Dimname dim, Tensor index, Tensor source) -> Tensor(a!)
at::Tensor & index_copy__dimname::call(at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & source) {
    static auto op = create_index_copy__dimname_typed_handle();
    return op.call(self, dim, index, source);
}

// aten::index_copy_.dimname(Tensor(a!) self, Dimname dim, Tensor index, Tensor source) -> Tensor(a!)
at::Tensor & index_copy__dimname::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & source) {
    static auto op = create_index_copy__dimname_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, source);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_put_, name, "aten::index_put_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_put_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_put_, schema_str, "index_put_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor(a!)")

// aten::index_put_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<index_put_::schema> create_index_put__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(index_put_::name, index_put_::overload_name)
      .typed<index_put_::schema>();
}

// aten::index_put_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor(a!)
at::Tensor & index_put_::call(at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate) {
    static auto op = create_index_put__typed_handle();
    return op.call(self, indices, values, accumulate);
}

// aten::index_put_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor(a!)
at::Tensor & index_put_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate) {
    static auto op = create_index_put__typed_handle();
    return op.redispatch(dispatchKeySet, self, indices, values, accumulate);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_index_put_impl_, name, "aten::_index_put_impl_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_index_put_impl_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_index_put_impl_, schema_str, "_index_put_impl_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False, bool unsafe=False) -> Tensor(a!)")

// aten::_index_put_impl_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False, bool unsafe=False) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<_index_put_impl_::schema> create__index_put_impl__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_index_put_impl_::name, _index_put_impl_::overload_name)
      .typed<_index_put_impl_::schema>();
}

// aten::_index_put_impl_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False, bool unsafe=False) -> Tensor(a!)
at::Tensor & _index_put_impl_::call(at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate, bool unsafe) {
    static auto op = create__index_put_impl__typed_handle();
    return op.call(self, indices, values, accumulate, unsafe);
}

// aten::_index_put_impl_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False, bool unsafe=False) -> Tensor(a!)
at::Tensor & _index_put_impl_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate, bool unsafe) {
    static auto op = create__index_put_impl__typed_handle();
    return op.redispatch(dispatchKeySet, self, indices, values, accumulate, unsafe);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(inverse, name, "aten::inverse")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(inverse, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(inverse, schema_str, "inverse(Tensor self) -> Tensor")

// aten::inverse(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<inverse::schema> create_inverse_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(inverse::name, inverse::overload_name)
      .typed<inverse::schema>();
}

// aten::inverse(Tensor self) -> Tensor
at::Tensor inverse::call(const at::Tensor & self) {
    static auto op = create_inverse_typed_handle();
    return op.call(self);
}

// aten::inverse(Tensor self) -> Tensor
at::Tensor inverse::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_inverse_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(inverse_out, name, "aten::inverse")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(inverse_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(inverse_out, schema_str, "inverse.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::inverse.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<inverse_out::schema> create_inverse_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(inverse_out::name, inverse_out::overload_name)
      .typed<inverse_out::schema>();
}

// aten::inverse.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & inverse_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_inverse_out_typed_handle();
    return op.call(self, out);
}

// aten::inverse.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & inverse_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_inverse_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isnan, name, "aten::isnan")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isnan, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isnan, schema_str, "isnan(Tensor self) -> Tensor")

// aten::isnan(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<isnan::schema> create_isnan_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(isnan::name, isnan::overload_name)
      .typed<isnan::schema>();
}

// aten::isnan(Tensor self) -> Tensor
at::Tensor isnan::call(const at::Tensor & self) {
    static auto op = create_isnan_typed_handle();
    return op.call(self);
}

// aten::isnan(Tensor self) -> Tensor
at::Tensor isnan::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_isnan_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(kthvalue, name, "aten::kthvalue")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(kthvalue, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(kthvalue, schema_str, "kthvalue(Tensor self, int k, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)")

// aten::kthvalue(Tensor self, int k, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)
static C10_NOINLINE c10::TypedOperatorHandle<kthvalue::schema> create_kthvalue_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(kthvalue::name, kthvalue::overload_name)
      .typed<kthvalue::schema>();
}

// aten::kthvalue(Tensor self, int k, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> kthvalue::call(const at::Tensor & self, int64_t k, int64_t dim, bool keepdim) {
    static auto op = create_kthvalue_typed_handle();
    return op.call(self, k, dim, keepdim);
}

// aten::kthvalue(Tensor self, int k, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> kthvalue::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t k, int64_t dim, bool keepdim) {
    static auto op = create_kthvalue_typed_handle();
    return op.redispatch(dispatchKeySet, self, k, dim, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(kthvalue_dimname_out, name, "aten::kthvalue")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(kthvalue_dimname_out, overload_name, "dimname_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(kthvalue_dimname_out, schema_str, "kthvalue.dimname_out(Tensor self, int k, Dimname dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)")

// aten::kthvalue.dimname_out(Tensor self, int k, Dimname dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
static C10_NOINLINE c10::TypedOperatorHandle<kthvalue_dimname_out::schema> create_kthvalue_dimname_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(kthvalue_dimname_out::name, kthvalue_dimname_out::overload_name)
      .typed<kthvalue_dimname_out::schema>();
}

// aten::kthvalue.dimname_out(Tensor self, int k, Dimname dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> kthvalue_dimname_out::call(const at::Tensor & self, int64_t k, at::Dimname dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
    static auto op = create_kthvalue_dimname_out_typed_handle();
    return op.call(self, k, dim, keepdim, values, indices);
}

// aten::kthvalue.dimname_out(Tensor self, int k, Dimname dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> kthvalue_dimname_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t k, at::Dimname dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
    static auto op = create_kthvalue_dimname_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, k, dim, keepdim, values, indices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(native_layer_norm, name, "aten::native_layer_norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(native_layer_norm, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(native_layer_norm, schema_str, "native_layer_norm(Tensor input, int[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)")

// aten::native_layer_norm(Tensor input, int[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<native_layer_norm::schema> create_native_layer_norm_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(native_layer_norm::name, native_layer_norm::overload_name)
      .typed<native_layer_norm::schema>();
}

// aten::native_layer_norm(Tensor input, int[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_layer_norm::call(const at::Tensor & input, at::IntArrayRef normalized_shape, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, double eps) {
    static auto op = create_native_layer_norm_typed_handle();
    return op.call(input, normalized_shape, weight, bias, eps);
}

// aten::native_layer_norm(Tensor input, int[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_layer_norm::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, at::IntArrayRef normalized_shape, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, double eps) {
    static auto op = create_native_layer_norm_typed_handle();
    return op.redispatch(dispatchKeySet, input, normalized_shape, weight, bias, eps);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nan_to_num, name, "aten::nan_to_num")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nan_to_num, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nan_to_num, schema_str, "nan_to_num(Tensor self, float? nan=None, float? posinf=None, float? neginf=None) -> Tensor")

// aten::nan_to_num(Tensor self, float? nan=None, float? posinf=None, float? neginf=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<nan_to_num::schema> create_nan_to_num_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nan_to_num::name, nan_to_num::overload_name)
      .typed<nan_to_num::schema>();
}

// aten::nan_to_num(Tensor self, float? nan=None, float? posinf=None, float? neginf=None) -> Tensor
at::Tensor nan_to_num::call(const at::Tensor & self, c10::optional<double> nan, c10::optional<double> posinf, c10::optional<double> neginf) {
    static auto op = create_nan_to_num_typed_handle();
    return op.call(self, nan, posinf, neginf);
}

// aten::nan_to_num(Tensor self, float? nan=None, float? posinf=None, float? neginf=None) -> Tensor
at::Tensor nan_to_num::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<double> nan, c10::optional<double> posinf, c10::optional<double> neginf) {
    static auto op = create_nan_to_num_typed_handle();
    return op.redispatch(dispatchKeySet, self, nan, posinf, neginf);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fbgemm_linear_int8_weight_fp32_activation, name, "aten::fbgemm_linear_int8_weight_fp32_activation")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fbgemm_linear_int8_weight_fp32_activation, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fbgemm_linear_int8_weight_fp32_activation, schema_str, "fbgemm_linear_int8_weight_fp32_activation(Tensor input, Tensor weight, Tensor packed, Tensor col_offsets, Scalar weight_scale, Scalar weight_zero_point, Tensor bias) -> Tensor")

// aten::fbgemm_linear_int8_weight_fp32_activation(Tensor input, Tensor weight, Tensor packed, Tensor col_offsets, Scalar weight_scale, Scalar weight_zero_point, Tensor bias) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fbgemm_linear_int8_weight_fp32_activation::schema> create_fbgemm_linear_int8_weight_fp32_activation_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fbgemm_linear_int8_weight_fp32_activation::name, fbgemm_linear_int8_weight_fp32_activation::overload_name)
      .typed<fbgemm_linear_int8_weight_fp32_activation::schema>();
}

// aten::fbgemm_linear_int8_weight_fp32_activation(Tensor input, Tensor weight, Tensor packed, Tensor col_offsets, Scalar weight_scale, Scalar weight_zero_point, Tensor bias) -> Tensor
at::Tensor fbgemm_linear_int8_weight_fp32_activation::call(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & packed, const at::Tensor & col_offsets, const at::Scalar & weight_scale, const at::Scalar & weight_zero_point, const at::Tensor & bias) {
    static auto op = create_fbgemm_linear_int8_weight_fp32_activation_typed_handle();
    return op.call(input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias);
}

// aten::fbgemm_linear_int8_weight_fp32_activation(Tensor input, Tensor weight, Tensor packed, Tensor col_offsets, Scalar weight_scale, Scalar weight_zero_point, Tensor bias) -> Tensor
at::Tensor fbgemm_linear_int8_weight_fp32_activation::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const at::Tensor & packed, const at::Tensor & col_offsets, const at::Scalar & weight_scale, const at::Scalar & weight_zero_point, const at::Tensor & bias) {
    static auto op = create_fbgemm_linear_int8_weight_fp32_activation_typed_handle();
    return op.redispatch(dispatchKeySet, input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fbgemm_linear_int8_weight, name, "aten::fbgemm_linear_int8_weight")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fbgemm_linear_int8_weight, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fbgemm_linear_int8_weight, schema_str, "fbgemm_linear_int8_weight(Tensor input, Tensor weight, Tensor packed, Tensor col_offsets, Scalar weight_scale, Scalar weight_zero_point, Tensor bias) -> Tensor")

// aten::fbgemm_linear_int8_weight(Tensor input, Tensor weight, Tensor packed, Tensor col_offsets, Scalar weight_scale, Scalar weight_zero_point, Tensor bias) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fbgemm_linear_int8_weight::schema> create_fbgemm_linear_int8_weight_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fbgemm_linear_int8_weight::name, fbgemm_linear_int8_weight::overload_name)
      .typed<fbgemm_linear_int8_weight::schema>();
}

// aten::fbgemm_linear_int8_weight(Tensor input, Tensor weight, Tensor packed, Tensor col_offsets, Scalar weight_scale, Scalar weight_zero_point, Tensor bias) -> Tensor
at::Tensor fbgemm_linear_int8_weight::call(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & packed, const at::Tensor & col_offsets, const at::Scalar & weight_scale, const at::Scalar & weight_zero_point, const at::Tensor & bias) {
    static auto op = create_fbgemm_linear_int8_weight_typed_handle();
    return op.call(input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias);
}

// aten::fbgemm_linear_int8_weight(Tensor input, Tensor weight, Tensor packed, Tensor col_offsets, Scalar weight_scale, Scalar weight_zero_point, Tensor bias) -> Tensor
at::Tensor fbgemm_linear_int8_weight::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const at::Tensor & packed, const at::Tensor & col_offsets, const at::Scalar & weight_scale, const at::Scalar & weight_zero_point, const at::Tensor & bias) {
    static auto op = create_fbgemm_linear_int8_weight_typed_handle();
    return op.redispatch(dispatchKeySet, input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fbgemm_linear_fp16_weight_fp32_activation, name, "aten::fbgemm_linear_fp16_weight_fp32_activation")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fbgemm_linear_fp16_weight_fp32_activation, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fbgemm_linear_fp16_weight_fp32_activation, schema_str, "fbgemm_linear_fp16_weight_fp32_activation(Tensor input, Tensor packed_weight, Tensor bias) -> Tensor")

// aten::fbgemm_linear_fp16_weight_fp32_activation(Tensor input, Tensor packed_weight, Tensor bias) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fbgemm_linear_fp16_weight_fp32_activation::schema> create_fbgemm_linear_fp16_weight_fp32_activation_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fbgemm_linear_fp16_weight_fp32_activation::name, fbgemm_linear_fp16_weight_fp32_activation::overload_name)
      .typed<fbgemm_linear_fp16_weight_fp32_activation::schema>();
}

// aten::fbgemm_linear_fp16_weight_fp32_activation(Tensor input, Tensor packed_weight, Tensor bias) -> Tensor
at::Tensor fbgemm_linear_fp16_weight_fp32_activation::call(const at::Tensor & input, const at::Tensor & packed_weight, const at::Tensor & bias) {
    static auto op = create_fbgemm_linear_fp16_weight_fp32_activation_typed_handle();
    return op.call(input, packed_weight, bias);
}

// aten::fbgemm_linear_fp16_weight_fp32_activation(Tensor input, Tensor packed_weight, Tensor bias) -> Tensor
at::Tensor fbgemm_linear_fp16_weight_fp32_activation::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & packed_weight, const at::Tensor & bias) {
    static auto op = create_fbgemm_linear_fp16_weight_fp32_activation_typed_handle();
    return op.redispatch(dispatchKeySet, input, packed_weight, bias);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ldexp_, name, "aten::ldexp_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ldexp_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ldexp_, schema_str, "ldexp_(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::ldexp_(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<ldexp_::schema> create_ldexp__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(ldexp_::name, ldexp_::overload_name)
      .typed<ldexp_::schema>();
}

// aten::ldexp_(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & ldexp_::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create_ldexp__typed_handle();
    return op.call(self, other);
}

// aten::ldexp_(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & ldexp_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create_ldexp__typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ldexp_out, name, "aten::ldexp")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ldexp_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ldexp_out, schema_str, "ldexp.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::ldexp.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<ldexp_out::schema> create_ldexp_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(ldexp_out::name, ldexp_out::overload_name)
      .typed<ldexp_out::schema>();
}

// aten::ldexp.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & ldexp_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_ldexp_out_typed_handle();
    return op.call(self, other, out);
}

// aten::ldexp.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & ldexp_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_ldexp_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log10_out, name, "aten::log10")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log10_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log10_out, schema_str, "log10.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::log10.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<log10_out::schema> create_log10_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(log10_out::name, log10_out::overload_name)
      .typed<log10_out::schema>();
}

// aten::log10.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & log10_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_log10_out_typed_handle();
    return op.call(self, out);
}

// aten::log10.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & log10_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_log10_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(xlogy_Tensor, name, "aten::xlogy")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(xlogy_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(xlogy_Tensor, schema_str, "xlogy.Tensor(Tensor self, Tensor other) -> Tensor")

// aten::xlogy.Tensor(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<xlogy_Tensor::schema> create_xlogy_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(xlogy_Tensor::name, xlogy_Tensor::overload_name)
      .typed<xlogy_Tensor::schema>();
}

// aten::xlogy.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor xlogy_Tensor::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_xlogy_Tensor_typed_handle();
    return op.call(self, other);
}

// aten::xlogy.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor xlogy_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_xlogy_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_log_softmax_backward_data, name, "aten::_log_softmax_backward_data")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_log_softmax_backward_data, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_log_softmax_backward_data, schema_str, "_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor")

// aten::_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_log_softmax_backward_data::schema> create__log_softmax_backward_data_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_log_softmax_backward_data::name, _log_softmax_backward_data::overload_name)
      .typed<_log_softmax_backward_data::schema>();
}

// aten::_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor
at::Tensor _log_softmax_backward_data::call(const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, const at::Tensor & self) {
    static auto op = create__log_softmax_backward_data_typed_handle();
    return op.call(grad_output, output, dim, self);
}

// aten::_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor
at::Tensor _log_softmax_backward_data::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, const at::Tensor & self) {
    static auto op = create__log_softmax_backward_data_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, output, dim, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_logcumsumexp_out, name, "aten::_logcumsumexp")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_logcumsumexp_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_logcumsumexp_out, schema_str, "_logcumsumexp.out(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)")

// aten::_logcumsumexp.out(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<_logcumsumexp_out::schema> create__logcumsumexp_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_logcumsumexp_out::name, _logcumsumexp_out::overload_name)
      .typed<_logcumsumexp_out::schema>();
}

// aten::_logcumsumexp.out(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & _logcumsumexp_out::call(const at::Tensor & self, int64_t dim, at::Tensor & out) {
    static auto op = create__logcumsumexp_out_typed_handle();
    return op.call(self, dim, out);
}

// aten::_logcumsumexp.out(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & _logcumsumexp_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, at::Tensor & out) {
    static auto op = create__logcumsumexp_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logcumsumexp, name, "aten::logcumsumexp")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logcumsumexp, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logcumsumexp, schema_str, "logcumsumexp(Tensor self, int dim) -> Tensor")

// aten::logcumsumexp(Tensor self, int dim) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<logcumsumexp::schema> create_logcumsumexp_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(logcumsumexp::name, logcumsumexp::overload_name)
      .typed<logcumsumexp::schema>();
}

// aten::logcumsumexp(Tensor self, int dim) -> Tensor
at::Tensor logcumsumexp::call(const at::Tensor & self, int64_t dim) {
    static auto op = create_logcumsumexp_typed_handle();
    return op.call(self, dim);
}

// aten::logcumsumexp(Tensor self, int dim) -> Tensor
at::Tensor logcumsumexp::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim) {
    static auto op = create_logcumsumexp_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(matmul_out, name, "aten::matmul")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(matmul_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(matmul_out, schema_str, "matmul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::matmul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<matmul_out::schema> create_matmul_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(matmul_out::name, matmul_out::overload_name)
      .typed<matmul_out::schema>();
}

// aten::matmul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & matmul_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_matmul_out_typed_handle();
    return op.call(self, other, out);
}

// aten::matmul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & matmul_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_matmul_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(matrix_power_out, name, "aten::matrix_power")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(matrix_power_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(matrix_power_out, schema_str, "matrix_power.out(Tensor self, int n, *, Tensor(a!) out) -> Tensor(a!)")

// aten::matrix_power.out(Tensor self, int n, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<matrix_power_out::schema> create_matrix_power_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(matrix_power_out::name, matrix_power_out::overload_name)
      .typed<matrix_power_out::schema>();
}

// aten::matrix_power.out(Tensor self, int n, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & matrix_power_out::call(const at::Tensor & self, int64_t n, at::Tensor & out) {
    static auto op = create_matrix_power_out_typed_handle();
    return op.call(self, n, out);
}

// aten::matrix_power.out(Tensor self, int n, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & matrix_power_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t n, at::Tensor & out) {
    static auto op = create_matrix_power_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, n, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(matrix_exp_backward, name, "aten::matrix_exp_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(matrix_exp_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(matrix_exp_backward, schema_str, "matrix_exp_backward(Tensor self, Tensor grad) -> Tensor")

// aten::matrix_exp_backward(Tensor self, Tensor grad) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<matrix_exp_backward::schema> create_matrix_exp_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(matrix_exp_backward::name, matrix_exp_backward::overload_name)
      .typed<matrix_exp_backward::schema>();
}

// aten::matrix_exp_backward(Tensor self, Tensor grad) -> Tensor
at::Tensor matrix_exp_backward::call(const at::Tensor & self, const at::Tensor & grad) {
    static auto op = create_matrix_exp_backward_typed_handle();
    return op.call(self, grad);
}

// aten::matrix_exp_backward(Tensor self, Tensor grad) -> Tensor
at::Tensor matrix_exp_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & grad) {
    static auto op = create_matrix_exp_backward_typed_handle();
    return op.redispatch(dispatchKeySet, self, grad);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_aminmax_dim, name, "aten::_aminmax")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_aminmax_dim, overload_name, "dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_aminmax_dim, schema_str, "_aminmax.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor, Tensor)")

// aten::_aminmax.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<_aminmax_dim::schema> create__aminmax_dim_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_aminmax_dim::name, _aminmax_dim::overload_name)
      .typed<_aminmax_dim::schema>();
}

// aten::_aminmax.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> _aminmax_dim::call(const at::Tensor & self, int64_t dim, bool keepdim) {
    static auto op = create__aminmax_dim_typed_handle();
    return op.call(self, dim, keepdim);
}

// aten::_aminmax.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> _aminmax_dim::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool keepdim) {
    static auto op = create__aminmax_dim_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(aminmax_out, name, "aten::aminmax")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(aminmax_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(aminmax_out, schema_str, "aminmax.out(Tensor self, *, int? dim=None, bool keepdim=False, Tensor(a!) min, Tensor(b!) max) -> (Tensor(a!) min, Tensor(b!) max)")

// aten::aminmax.out(Tensor self, *, int? dim=None, bool keepdim=False, Tensor(a!) min, Tensor(b!) max) -> (Tensor(a!) min, Tensor(b!) max)
static C10_NOINLINE c10::TypedOperatorHandle<aminmax_out::schema> create_aminmax_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(aminmax_out::name, aminmax_out::overload_name)
      .typed<aminmax_out::schema>();
}

// aten::aminmax.out(Tensor self, *, int? dim=None, bool keepdim=False, Tensor(a!) min, Tensor(b!) max) -> (Tensor(a!) min, Tensor(b!) max)
::std::tuple<at::Tensor &,at::Tensor &> aminmax_out::call(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim, at::Tensor & min, at::Tensor & max) {
    static auto op = create_aminmax_out_typed_handle();
    return op.call(self, dim, keepdim, min, max);
}

// aten::aminmax.out(Tensor self, *, int? dim=None, bool keepdim=False, Tensor(a!) min, Tensor(b!) max) -> (Tensor(a!) min, Tensor(b!) max)
::std::tuple<at::Tensor &,at::Tensor &> aminmax_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim, at::Tensor & min, at::Tensor & max) {
    static auto op = create_aminmax_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, min, max);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_compute_linear_combination_out, name, "aten::_compute_linear_combination")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_compute_linear_combination_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_compute_linear_combination_out, schema_str, "_compute_linear_combination.out(Tensor input, Tensor coefficients, *, Tensor(a!) out) -> Tensor(a!)")

// aten::_compute_linear_combination.out(Tensor input, Tensor coefficients, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<_compute_linear_combination_out::schema> create__compute_linear_combination_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_compute_linear_combination_out::name, _compute_linear_combination_out::overload_name)
      .typed<_compute_linear_combination_out::schema>();
}

// aten::_compute_linear_combination.out(Tensor input, Tensor coefficients, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & _compute_linear_combination_out::call(const at::Tensor & input, const at::Tensor & coefficients, at::Tensor & out) {
    static auto op = create__compute_linear_combination_out_typed_handle();
    return op.call(input, coefficients, out);
}

// aten::_compute_linear_combination.out(Tensor input, Tensor coefficients, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & _compute_linear_combination_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & coefficients, at::Tensor & out) {
    static auto op = create__compute_linear_combination_out_typed_handle();
    return op.redispatch(dispatchKeySet, input, coefficients, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(amax, name, "aten::amax")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(amax, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(amax, schema_str, "amax(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor")

// aten::amax(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<amax::schema> create_amax_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(amax::name, amax::overload_name)
      .typed<amax::schema>();
}

// aten::amax(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor
at::Tensor amax::call(const at::Tensor & self, at::IntArrayRef dim, bool keepdim) {
    static auto op = create_amax_typed_handle();
    return op.call(self, dim, keepdim);
}

// aten::amax(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor
at::Tensor amax::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool keepdim) {
    static auto op = create_amax_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(amax_out, name, "aten::amax")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(amax_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(amax_out, schema_str, "amax.out(Tensor self, int[1] dim=[], bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)")

// aten::amax.out(Tensor self, int[1] dim=[], bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<amax_out::schema> create_amax_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(amax_out::name, amax_out::overload_name)
      .typed<amax_out::schema>();
}

// aten::amax.out(Tensor self, int[1] dim=[], bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & amax_out::call(const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
    static auto op = create_amax_out_typed_handle();
    return op.call(self, dim, keepdim, out);
}

// aten::amax.out(Tensor self, int[1] dim=[], bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & amax_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
    static auto op = create_amax_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_max_pool2d, name, "aten::mkldnn_max_pool2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_max_pool2d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_max_pool2d, schema_str, "mkldnn_max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor")

// aten::mkldnn_max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<mkldnn_max_pool2d::schema> create_mkldnn_max_pool2d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mkldnn_max_pool2d::name, mkldnn_max_pool2d::overload_name)
      .typed<mkldnn_max_pool2d::schema>();
}

// aten::mkldnn_max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor
at::Tensor mkldnn_max_pool2d::call(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
    static auto op = create_mkldnn_max_pool2d_typed_handle();
    return op.call(self, kernel_size, stride, padding, dilation, ceil_mode);
}

// aten::mkldnn_max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor
at::Tensor mkldnn_max_pool2d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
    static auto op = create_mkldnn_max_pool2d_typed_handle();
    return op.redispatch(dispatchKeySet, self, kernel_size, stride, padding, dilation, ceil_mode);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantized_max_pool2d, name, "aten::quantized_max_pool2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantized_max_pool2d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantized_max_pool2d, schema_str, "quantized_max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor")

// aten::quantized_max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<quantized_max_pool2d::schema> create_quantized_max_pool2d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(quantized_max_pool2d::name, quantized_max_pool2d::overload_name)
      .typed<quantized_max_pool2d::schema>();
}

// aten::quantized_max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor
at::Tensor quantized_max_pool2d::call(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
    static auto op = create_quantized_max_pool2d_typed_handle();
    return op.call(self, kernel_size, stride, padding, dilation, ceil_mode);
}

// aten::quantized_max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor
at::Tensor quantized_max_pool2d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
    static auto op = create_quantized_max_pool2d_typed_handle();
    return op.redispatch(dispatchKeySet, self, kernel_size, stride, padding, dilation, ceil_mode);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mean_names_out, name, "aten::mean")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mean_names_out, overload_name, "names_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mean_names_out, schema_str, "mean.names_out(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)")

// aten::mean.names_out(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<mean_names_out::schema> create_mean_names_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mean_names_out::name, mean_names_out::overload_name)
      .typed<mean_names_out::schema>();
}

// aten::mean.names_out(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & mean_names_out::call(const at::Tensor & self, at::DimnameList dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
    static auto op = create_mean_names_out_typed_handle();
    return op.call(self, dim, keepdim, dtype, out);
}

// aten::mean.names_out(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & mean_names_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
    static auto op = create_mean_names_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, dtype, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanmean_out, name, "aten::nanmean")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanmean_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanmean_out, schema_str, "nanmean.out(Tensor self, int[1] dim=[], bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)")

// aten::nanmean.out(Tensor self, int[1] dim=[], bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<nanmean_out::schema> create_nanmean_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nanmean_out::name, nanmean_out::overload_name)
      .typed<nanmean_out::schema>();
}

// aten::nanmean.out(Tensor self, int[1] dim=[], bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & nanmean_out::call(const at::Tensor & self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
    static auto op = create_nanmean_out_typed_handle();
    return op.call(self, dim, keepdim, dtype, out);
}

// aten::nanmean.out(Tensor self, int[1] dim=[], bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & nanmean_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
    static auto op = create_nanmean_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, dtype, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(amin, name, "aten::amin")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(amin, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(amin, schema_str, "amin(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor")

// aten::amin(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<amin::schema> create_amin_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(amin::name, amin::overload_name)
      .typed<amin::schema>();
}

// aten::amin(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor
at::Tensor amin::call(const at::Tensor & self, at::IntArrayRef dim, bool keepdim) {
    static auto op = create_amin_typed_handle();
    return op.call(self, dim, keepdim);
}

// aten::amin(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor
at::Tensor amin::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool keepdim) {
    static auto op = create_amin_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_convolution_transpose_backward, name, "aten::miopen_convolution_transpose_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_convolution_transpose_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_convolution_transpose_backward, schema_str, "miopen_convolution_transpose_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool[3] output_mask) -> (Tensor, Tensor, Tensor)")

// aten::miopen_convolution_transpose_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<miopen_convolution_transpose_backward::schema> create_miopen_convolution_transpose_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(miopen_convolution_transpose_backward::name, miopen_convolution_transpose_backward::overload_name)
      .typed<miopen_convolution_transpose_backward::schema>();
}

// aten::miopen_convolution_transpose_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_convolution_transpose_backward::call(const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, ::std::array<bool,3> output_mask) {
    static auto op = create_miopen_convolution_transpose_backward_typed_handle();
    return op.call(self, grad_output, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, output_mask);
}

// aten::miopen_convolution_transpose_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_convolution_transpose_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, ::std::array<bool,3> output_mask) {
    static auto op = create_miopen_convolution_transpose_backward_typed_handle();
    return op.redispatch(dispatchKeySet, self, grad_output, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, output_mask);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_convolution_transpose_backward_weight, name, "aten::miopen_convolution_transpose_backward_weight")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_convolution_transpose_backward_weight, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_convolution_transpose_backward_weight, schema_str, "miopen_convolution_transpose_backward_weight(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor")

// aten::miopen_convolution_transpose_backward_weight(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<miopen_convolution_transpose_backward_weight::schema> create_miopen_convolution_transpose_backward_weight_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(miopen_convolution_transpose_backward_weight::name, miopen_convolution_transpose_backward_weight::overload_name)
      .typed<miopen_convolution_transpose_backward_weight::schema>();
}

// aten::miopen_convolution_transpose_backward_weight(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
at::Tensor miopen_convolution_transpose_backward_weight::call(at::IntArrayRef weight_size, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
    static auto op = create_miopen_convolution_transpose_backward_weight_typed_handle();
    return op.call(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
}

// aten::miopen_convolution_transpose_backward_weight(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
at::Tensor miopen_convolution_transpose_backward_weight::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef weight_size, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
    static auto op = create_miopen_convolution_transpose_backward_weight_typed_handle();
    return op.redispatch(dispatchKeySet, weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_depthwise_convolution, name, "aten::miopen_depthwise_convolution")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_depthwise_convolution, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_depthwise_convolution, schema_str, "miopen_depthwise_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor")

// aten::miopen_depthwise_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<miopen_depthwise_convolution::schema> create_miopen_depthwise_convolution_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(miopen_depthwise_convolution::name, miopen_depthwise_convolution::overload_name)
      .typed<miopen_depthwise_convolution::schema>();
}

// aten::miopen_depthwise_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
at::Tensor miopen_depthwise_convolution::call(const at::Tensor & self, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
    static auto op = create_miopen_depthwise_convolution_typed_handle();
    return op.call(self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
}

// aten::miopen_depthwise_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
at::Tensor miopen_depthwise_convolution::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
    static auto op = create_miopen_depthwise_convolution_typed_handle();
    return op.redispatch(dispatchKeySet, self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mm_out, name, "aten::mm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mm_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mm_out, schema_str, "mm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)")

// aten::mm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<mm_out::schema> create_mm_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mm_out::name, mm_out::overload_name)
      .typed<mm_out::schema>();
}

// aten::mm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & mm_out::call(const at::Tensor & self, const at::Tensor & mat2, at::Tensor & out) {
    static auto op = create_mm_out_typed_handle();
    return op.call(self, mat2, out);
}

// aten::mm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & mm_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mat2, at::Tensor & out) {
    static auto op = create_mm_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, mat2, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_mask_helper, name, "aten::_sparse_mask_helper")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_mask_helper, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_mask_helper, schema_str, "_sparse_mask_helper(Tensor t, Tensor mask_indices) -> Tensor")

// aten::_sparse_mask_helper(Tensor t, Tensor mask_indices) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_sparse_mask_helper::schema> create__sparse_mask_helper_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_sparse_mask_helper::name, _sparse_mask_helper::overload_name)
      .typed<_sparse_mask_helper::schema>();
}

// aten::_sparse_mask_helper(Tensor t, Tensor mask_indices) -> Tensor
at::Tensor _sparse_mask_helper::call(const at::Tensor & t, const at::Tensor & mask_indices) {
    static auto op = create__sparse_mask_helper_typed_handle();
    return op.call(t, mask_indices);
}

// aten::_sparse_mask_helper(Tensor t, Tensor mask_indices) -> Tensor
at::Tensor _sparse_mask_helper::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & t, const at::Tensor & mask_indices) {
    static auto op = create__sparse_mask_helper_typed_handle();
    return op.redispatch(dispatchKeySet, t, mask_indices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mode_dimname_out, name, "aten::mode")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mode_dimname_out, overload_name, "dimname_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mode_dimname_out, schema_str, "mode.dimname_out(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)")

// aten::mode.dimname_out(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
static C10_NOINLINE c10::TypedOperatorHandle<mode_dimname_out::schema> create_mode_dimname_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mode_dimname_out::name, mode_dimname_out::overload_name)
      .typed<mode_dimname_out::schema>();
}

// aten::mode.dimname_out(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> mode_dimname_out::call(const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
    static auto op = create_mode_dimname_out_typed_handle();
    return op.call(self, dim, keepdim, values, indices);
}

// aten::mode.dimname_out(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> mode_dimname_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
    static auto op = create_mode_dimname_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, values, indices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multiply__Tensor, name, "aten::multiply_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multiply__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multiply__Tensor, schema_str, "multiply_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::multiply_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<multiply__Tensor::schema> create_multiply__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(multiply__Tensor::name, multiply__Tensor::overload_name)
      .typed<multiply__Tensor::schema>();
}

// aten::multiply_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & multiply__Tensor::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create_multiply__Tensor_typed_handle();
    return op.call(self, other);
}

// aten::multiply_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & multiply__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create_multiply__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(narrow_Tensor, name, "aten::narrow")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(narrow_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(narrow_Tensor, schema_str, "narrow.Tensor(Tensor(a) self, int dim, Tensor start, int length) -> Tensor(a)")

// aten::narrow.Tensor(Tensor(a) self, int dim, Tensor start, int length) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<narrow_Tensor::schema> create_narrow_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(narrow_Tensor::name, narrow_Tensor::overload_name)
      .typed<narrow_Tensor::schema>();
}

// aten::narrow.Tensor(Tensor(a) self, int dim, Tensor start, int length) -> Tensor(a)
at::Tensor narrow_Tensor::call(const at::Tensor & self, int64_t dim, const at::Tensor & start, int64_t length) {
    static auto op = create_narrow_Tensor_typed_handle();
    return op.call(self, dim, start, length);
}

// aten::narrow.Tensor(Tensor(a) self, int dim, Tensor start, int length) -> Tensor(a)
at::Tensor narrow_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, const at::Tensor & start, int64_t length) {
    static auto op = create_narrow_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, start, length);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(native_batch_norm, name, "aten::native_batch_norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(native_batch_norm, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(native_batch_norm, schema_str, "native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)")

// aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<native_batch_norm::schema> create_native_batch_norm_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(native_batch_norm::name, native_batch_norm::overload_name)
      .typed<native_batch_norm::schema>();
}

// aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_batch_norm::call(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double momentum, double eps) {
    static auto op = create_native_batch_norm_typed_handle();
    return op.call(input, weight, bias, running_mean, running_var, training, momentum, eps);
}

// aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_batch_norm::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double momentum, double eps) {
    static auto op = create_native_batch_norm_typed_handle();
    return op.redispatch(dispatchKeySet, input, weight, bias, running_mean, running_var, training, momentum, eps);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(batch_norm_stats, name, "aten::batch_norm_stats")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(batch_norm_stats, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(batch_norm_stats, schema_str, "batch_norm_stats(Tensor input, float eps) -> (Tensor, Tensor)")

// aten::batch_norm_stats(Tensor input, float eps) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<batch_norm_stats::schema> create_batch_norm_stats_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(batch_norm_stats::name, batch_norm_stats::overload_name)
      .typed<batch_norm_stats::schema>();
}

// aten::batch_norm_stats(Tensor input, float eps) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> batch_norm_stats::call(const at::Tensor & input, double eps) {
    static auto op = create_batch_norm_stats_typed_handle();
    return op.call(input, eps);
}

// aten::batch_norm_stats(Tensor input, float eps) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> batch_norm_stats::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, double eps) {
    static auto op = create_batch_norm_stats_typed_handle();
    return op.redispatch(dispatchKeySet, input, eps);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(batch_norm_gather_stats, name, "aten::batch_norm_gather_stats")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(batch_norm_gather_stats, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(batch_norm_gather_stats, schema_str, "batch_norm_gather_stats(Tensor input, Tensor mean, Tensor invstd, Tensor? running_mean, Tensor? running_var, float momentum, float eps, int count) -> (Tensor, Tensor)")

// aten::batch_norm_gather_stats(Tensor input, Tensor mean, Tensor invstd, Tensor? running_mean, Tensor? running_var, float momentum, float eps, int count) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<batch_norm_gather_stats::schema> create_batch_norm_gather_stats_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(batch_norm_gather_stats::name, batch_norm_gather_stats::overload_name)
      .typed<batch_norm_gather_stats::schema>();
}

// aten::batch_norm_gather_stats(Tensor input, Tensor mean, Tensor invstd, Tensor? running_mean, Tensor? running_var, float momentum, float eps, int count) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> batch_norm_gather_stats::call(const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, double momentum, double eps, int64_t count) {
    static auto op = create_batch_norm_gather_stats_typed_handle();
    return op.call(input, mean, invstd, running_mean, running_var, momentum, eps, count);
}

// aten::batch_norm_gather_stats(Tensor input, Tensor mean, Tensor invstd, Tensor? running_mean, Tensor? running_var, float momentum, float eps, int count) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> batch_norm_gather_stats::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, double momentum, double eps, int64_t count) {
    static auto op = create_batch_norm_gather_stats_typed_handle();
    return op.redispatch(dispatchKeySet, input, mean, invstd, running_mean, running_var, momentum, eps, count);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(native_batch_norm_backward, name, "aten::native_batch_norm_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(native_batch_norm_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(native_batch_norm_backward, schema_str, "native_batch_norm_backward(Tensor grad_out, Tensor input, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_invstd, bool train, float eps, bool[3] output_mask) -> (Tensor, Tensor, Tensor)")

// aten::native_batch_norm_backward(Tensor grad_out, Tensor input, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_invstd, bool train, float eps, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<native_batch_norm_backward::schema> create_native_batch_norm_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(native_batch_norm_backward::name, native_batch_norm_backward::overload_name)
      .typed<native_batch_norm_backward::schema>();
}

// aten::native_batch_norm_backward(Tensor grad_out, Tensor input, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_invstd, bool train, float eps, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_batch_norm_backward::call(const at::Tensor & grad_out, const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, const c10::optional<at::Tensor> & save_mean, const c10::optional<at::Tensor> & save_invstd, bool train, double eps, ::std::array<bool,3> output_mask) {
    static auto op = create_native_batch_norm_backward_typed_handle();
    return op.call(grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask);
}

// aten::native_batch_norm_backward(Tensor grad_out, Tensor input, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_invstd, bool train, float eps, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_batch_norm_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_out, const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, const c10::optional<at::Tensor> & save_mean, const c10::optional<at::Tensor> & save_invstd, bool train, double eps, ::std::array<bool,3> output_mask) {
    static auto op = create_native_batch_norm_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(batch_norm_backward_reduce, name, "aten::batch_norm_backward_reduce")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(batch_norm_backward_reduce, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(batch_norm_backward_reduce, schema_str, "batch_norm_backward_reduce(Tensor grad_out, Tensor input, Tensor mean, Tensor invstd, Tensor? weight, bool input_g, bool weight_g, bool bias_g) -> (Tensor, Tensor, Tensor, Tensor)")

// aten::batch_norm_backward_reduce(Tensor grad_out, Tensor input, Tensor mean, Tensor invstd, Tensor? weight, bool input_g, bool weight_g, bool bias_g) -> (Tensor, Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<batch_norm_backward_reduce::schema> create_batch_norm_backward_reduce_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(batch_norm_backward_reduce::name, batch_norm_backward_reduce::overload_name)
      .typed<batch_norm_backward_reduce::schema>();
}

// aten::batch_norm_backward_reduce(Tensor grad_out, Tensor input, Tensor mean, Tensor invstd, Tensor? weight, bool input_g, bool weight_g, bool bias_g) -> (Tensor, Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> batch_norm_backward_reduce::call(const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const c10::optional<at::Tensor> & weight, bool input_g, bool weight_g, bool bias_g) {
    static auto op = create_batch_norm_backward_reduce_typed_handle();
    return op.call(grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g);
}

// aten::batch_norm_backward_reduce(Tensor grad_out, Tensor input, Tensor mean, Tensor invstd, Tensor? weight, bool input_g, bool weight_g, bool bias_g) -> (Tensor, Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> batch_norm_backward_reduce::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const c10::optional<at::Tensor> & weight, bool input_g, bool weight_g, bool bias_g) {
    static auto op = create_batch_norm_backward_reduce_typed_handle();
    return op.redispatch(dispatchKeySet, grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_vulkan_available, name, "aten::is_vulkan_available")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_vulkan_available, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_vulkan_available, schema_str, "is_vulkan_available() -> bool")

// aten::is_vulkan_available() -> bool
static C10_NOINLINE c10::TypedOperatorHandle<is_vulkan_available::schema> create_is_vulkan_available_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(is_vulkan_available::name, is_vulkan_available::overload_name)
      .typed<is_vulkan_available::schema>();
}

// aten::is_vulkan_available() -> bool
bool is_vulkan_available::call() {
    static auto op = create_is_vulkan_available_typed_handle();
    return op.call();
}

// aten::is_vulkan_available() -> bool
bool is_vulkan_available::redispatch(c10::DispatchKeySet dispatchKeySet) {
    static auto op = create_is_vulkan_available_typed_handle();
    return op.redispatch(dispatchKeySet);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_nnpack_spatial_convolution, name, "aten::_nnpack_spatial_convolution")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_nnpack_spatial_convolution, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_nnpack_spatial_convolution, schema_str, "_nnpack_spatial_convolution(Tensor input, Tensor weight, Tensor? bias, int[2] padding, int[2] stride=1) -> Tensor")

// aten::_nnpack_spatial_convolution(Tensor input, Tensor weight, Tensor? bias, int[2] padding, int[2] stride=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_nnpack_spatial_convolution::schema> create__nnpack_spatial_convolution_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_nnpack_spatial_convolution::name, _nnpack_spatial_convolution::overload_name)
      .typed<_nnpack_spatial_convolution::schema>();
}

// aten::_nnpack_spatial_convolution(Tensor input, Tensor weight, Tensor? bias, int[2] padding, int[2] stride=1) -> Tensor
at::Tensor _nnpack_spatial_convolution::call(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef stride) {
    static auto op = create__nnpack_spatial_convolution_typed_handle();
    return op.call(input, weight, bias, padding, stride);
}

// aten::_nnpack_spatial_convolution(Tensor input, Tensor weight, Tensor? bias, int[2] padding, int[2] stride=1) -> Tensor
at::Tensor _nnpack_spatial_convolution::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef stride) {
    static auto op = create__nnpack_spatial_convolution_typed_handle();
    return op.redispatch(dispatchKeySet, input, weight, bias, padding, stride);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ones_names, name, "aten::ones")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ones_names, overload_name, "names")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ones_names, schema_str, "ones.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::ones.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<ones_names::schema> create_ones_names_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(ones_names::name, ones_names::overload_name)
      .typed<ones_names::schema>();
}

// aten::ones.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor ones_names::call(at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_ones_names_typed_handle();
    return op.call(size, names, dtype, layout, device, pin_memory);
}

// aten::ones.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor ones_names::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_ones_names_typed_handle();
    return op.redispatch(dispatchKeySet, size, names, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ones, name, "aten::ones")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ones, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ones, schema_str, "ones(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::ones(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<ones::schema> create_ones_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(ones::name, ones::overload_name)
      .typed<ones::schema>();
}

// aten::ones(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor ones::call(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_ones_typed_handle();
    return op.call(size, dtype, layout, device, pin_memory);
}

// aten::ones(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor ones::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_ones_typed_handle();
    return op.redispatch(dispatchKeySet, size, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cdist_forward, name, "aten::_cdist_forward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cdist_forward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cdist_forward, schema_str, "_cdist_forward(Tensor x1, Tensor x2, float p, int? compute_mode) -> Tensor")

// aten::_cdist_forward(Tensor x1, Tensor x2, float p, int? compute_mode) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_cdist_forward::schema> create__cdist_forward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_cdist_forward::name, _cdist_forward::overload_name)
      .typed<_cdist_forward::schema>();
}

// aten::_cdist_forward(Tensor x1, Tensor x2, float p, int? compute_mode) -> Tensor
at::Tensor _cdist_forward::call(const at::Tensor & x1, const at::Tensor & x2, double p, c10::optional<int64_t> compute_mode) {
    static auto op = create__cdist_forward_typed_handle();
    return op.call(x1, x2, p, compute_mode);
}

// aten::_cdist_forward(Tensor x1, Tensor x2, float p, int? compute_mode) -> Tensor
at::Tensor _cdist_forward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & x1, const at::Tensor & x2, double p, c10::optional<int64_t> compute_mode) {
    static auto op = create__cdist_forward_typed_handle();
    return op.redispatch(dispatchKeySet, x1, x2, p, compute_mode);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cosine_similarity, name, "aten::cosine_similarity")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cosine_similarity, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cosine_similarity, schema_str, "cosine_similarity(Tensor x1, Tensor x2, int dim=1, float eps=1e-08) -> Tensor")

// aten::cosine_similarity(Tensor x1, Tensor x2, int dim=1, float eps=1e-08) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<cosine_similarity::schema> create_cosine_similarity_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cosine_similarity::name, cosine_similarity::overload_name)
      .typed<cosine_similarity::schema>();
}

// aten::cosine_similarity(Tensor x1, Tensor x2, int dim=1, float eps=1e-08) -> Tensor
at::Tensor cosine_similarity::call(const at::Tensor & x1, const at::Tensor & x2, int64_t dim, double eps) {
    static auto op = create_cosine_similarity_typed_handle();
    return op.call(x1, x2, dim, eps);
}

// aten::cosine_similarity(Tensor x1, Tensor x2, int dim=1, float eps=1e-08) -> Tensor
at::Tensor cosine_similarity::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & x1, const at::Tensor & x2, int64_t dim, double eps) {
    static auto op = create_cosine_similarity_typed_handle();
    return op.redispatch(dispatchKeySet, x1, x2, dim, eps);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(numpy_T, name, "aten::numpy_T")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(numpy_T, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(numpy_T, schema_str, "numpy_T(Tensor(a) self) -> Tensor(a)")

// aten::numpy_T(Tensor(a) self) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<numpy_T::schema> create_numpy_T_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(numpy_T::name, numpy_T::overload_name)
      .typed<numpy_T::schema>();
}

// aten::numpy_T(Tensor(a) self) -> Tensor(a)
at::Tensor numpy_T::call(const at::Tensor & self) {
    static auto op = create_numpy_T_typed_handle();
    return op.call(self);
}

// aten::numpy_T(Tensor(a) self) -> Tensor(a)
at::Tensor numpy_T::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_numpy_T_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rand_like, name, "aten::rand_like")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rand_like, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rand_like, schema_str, "rand_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor")

// aten::rand_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<rand_like::schema> create_rand_like_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(rand_like::name, rand_like::overload_name)
      .typed<rand_like::schema>();
}

// aten::rand_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
at::Tensor rand_like::call(const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create_rand_like_typed_handle();
    return op.call(self, dtype, layout, device, pin_memory, memory_format);
}

// aten::rand_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
at::Tensor rand_like::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create_rand_like_typed_handle();
    return op.redispatch(dispatchKeySet, self, dtype, layout, device, pin_memory, memory_format);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randint_generator, name, "aten::randint")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randint_generator, overload_name, "generator")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randint_generator, schema_str, "randint.generator(int high, int[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::randint.generator(int high, int[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<randint_generator::schema> create_randint_generator_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(randint_generator::name, randint_generator::overload_name)
      .typed<randint_generator::schema>();
}

// aten::randint.generator(int high, int[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor randint_generator::call(int64_t high, at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_randint_generator_typed_handle();
    return op.call(high, size, generator, dtype, layout, device, pin_memory);
}

// aten::randint.generator(int high, int[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor randint_generator::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t high, at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_randint_generator_typed_handle();
    return op.redispatch(dispatchKeySet, high, size, generator, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randint_low, name, "aten::randint")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randint_low, overload_name, "low")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randint_low, schema_str, "randint.low(int low, int high, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::randint.low(int low, int high, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<randint_low::schema> create_randint_low_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(randint_low::name, randint_low::overload_name)
      .typed<randint_low::schema>();
}

// aten::randint.low(int low, int high, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor randint_low::call(int64_t low, int64_t high, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_randint_low_typed_handle();
    return op.call(low, high, size, dtype, layout, device, pin_memory);
}

// aten::randint.low(int low, int high, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor randint_low::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t low, int64_t high, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_randint_low_typed_handle();
    return op.redispatch(dispatchKeySet, low, high, size, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randint_low_out, name, "aten::randint")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randint_low_out, overload_name, "low_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randint_low_out, schema_str, "randint.low_out(int low, int high, int[] size, *, Tensor(a!) out) -> Tensor(a!)")

// aten::randint.low_out(int low, int high, int[] size, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<randint_low_out::schema> create_randint_low_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(randint_low_out::name, randint_low_out::overload_name)
      .typed<randint_low_out::schema>();
}

// aten::randint.low_out(int low, int high, int[] size, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & randint_low_out::call(int64_t low, int64_t high, at::IntArrayRef size, at::Tensor & out) {
    static auto op = create_randint_low_out_typed_handle();
    return op.call(low, high, size, out);
}

// aten::randint.low_out(int low, int high, int[] size, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & randint_low_out::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t low, int64_t high, at::IntArrayRef size, at::Tensor & out) {
    static auto op = create_randint_low_out_typed_handle();
    return op.redispatch(dispatchKeySet, low, high, size, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randint_like, name, "aten::randint_like")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randint_like, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randint_like, schema_str, "randint_like(Tensor self, int high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor")

// aten::randint_like(Tensor self, int high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<randint_like::schema> create_randint_like_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(randint_like::name, randint_like::overload_name)
      .typed<randint_like::schema>();
}

// aten::randint_like(Tensor self, int high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
at::Tensor randint_like::call(const at::Tensor & self, int64_t high, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create_randint_like_typed_handle();
    return op.call(self, high, dtype, layout, device, pin_memory, memory_format);
}

// aten::randint_like(Tensor self, int high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
at::Tensor randint_like::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t high, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create_randint_like_typed_handle();
    return op.redispatch(dispatchKeySet, self, high, dtype, layout, device, pin_memory, memory_format);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randn_names, name, "aten::randn")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randn_names, overload_name, "names")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randn_names, schema_str, "randn.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::randn.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<randn_names::schema> create_randn_names_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(randn_names::name, randn_names::overload_name)
      .typed<randn_names::schema>();
}

// aten::randn.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor randn_names::call(at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_randn_names_typed_handle();
    return op.call(size, names, dtype, layout, device, pin_memory);
}

// aten::randn.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor randn_names::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_randn_names_typed_handle();
    return op.redispatch(dispatchKeySet, size, names, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(range_step, name, "aten::range")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(range_step, overload_name, "step")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(range_step, schema_str, "range.step(Scalar start, Scalar end, Scalar step=1, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::range.step(Scalar start, Scalar end, Scalar step=1, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<range_step::schema> create_range_step_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(range_step::name, range_step::overload_name)
      .typed<range_step::schema>();
}

// aten::range.step(Scalar start, Scalar end, Scalar step=1, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor range_step::call(const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_range_step_typed_handle();
    return op.call(start, end, step, dtype, layout, device, pin_memory);
}

// aten::range.step(Scalar start, Scalar end, Scalar step=1, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor range_step::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_range_step_typed_handle();
    return op.redispatch(dispatchKeySet, start, end, step, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(neg_out, name, "aten::neg")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(neg_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(neg_out, schema_str, "neg.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::neg.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<neg_out::schema> create_neg_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(neg_out::name, neg_out::overload_name)
      .typed<neg_out::schema>();
}

// aten::neg.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & neg_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_neg_out_typed_handle();
    return op.call(self, out);
}

// aten::neg.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & neg_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_neg_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(round, name, "aten::round")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(round, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(round, schema_str, "round(Tensor self) -> Tensor")

// aten::round(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<round::schema> create_round_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(round::name, round::overload_name)
      .typed<round::schema>();
}

// aten::round(Tensor self) -> Tensor
at::Tensor round::call(const at::Tensor & self) {
    static auto op = create_round_typed_handle();
    return op.call(self);
}

// aten::round(Tensor self) -> Tensor
at::Tensor round::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_round_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(round_out, name, "aten::round")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(round_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(round_out, schema_str, "round.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::round.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<round_out::schema> create_round_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(round_out::name, round_out::overload_name)
      .typed<round_out::schema>();
}

// aten::round.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & round_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_round_out_typed_handle();
    return op.call(self, out);
}

// aten::round.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & round_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_round_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rrelu_, name, "aten::rrelu_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rrelu_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rrelu_, schema_str, "rrelu_(Tensor(a!) self, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor(a!)")

// aten::rrelu_(Tensor(a!) self, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<rrelu_::schema> create_rrelu__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(rrelu_::name, rrelu_::overload_name)
      .typed<rrelu_::schema>();
}

// aten::rrelu_(Tensor(a!) self, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor(a!)
at::Tensor & rrelu_::call(at::Tensor & self, const at::Scalar & lower, const at::Scalar & upper, bool training, c10::optional<at::Generator> generator) {
    static auto op = create_rrelu__typed_handle();
    return op.call(self, lower, upper, training, generator);
}

// aten::rrelu_(Tensor(a!) self, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor(a!)
at::Tensor & rrelu_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & lower, const at::Scalar & upper, bool training, c10::optional<at::Generator> generator) {
    static auto op = create_rrelu__typed_handle();
    return op.redispatch(dispatchKeySet, self, lower, upper, training, generator);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gelu, name, "aten::gelu")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gelu, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gelu, schema_str, "gelu(Tensor self) -> Tensor")

// aten::gelu(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<gelu::schema> create_gelu_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(gelu::name, gelu::overload_name)
      .typed<gelu::schema>();
}

// aten::gelu(Tensor self) -> Tensor
at::Tensor gelu::call(const at::Tensor & self) {
    static auto op = create_gelu_typed_handle();
    return op.call(self);
}

// aten::gelu(Tensor self) -> Tensor
at::Tensor gelu::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_gelu_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardshrink, name, "aten::hardshrink")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardshrink, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardshrink, schema_str, "hardshrink(Tensor self, Scalar lambd=0.5) -> Tensor")

// aten::hardshrink(Tensor self, Scalar lambd=0.5) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<hardshrink::schema> create_hardshrink_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(hardshrink::name, hardshrink::overload_name)
      .typed<hardshrink::schema>();
}

// aten::hardshrink(Tensor self, Scalar lambd=0.5) -> Tensor
at::Tensor hardshrink::call(const at::Tensor & self, const at::Scalar & lambd) {
    static auto op = create_hardshrink_typed_handle();
    return op.call(self, lambd);
}

// aten::hardshrink(Tensor self, Scalar lambd=0.5) -> Tensor
at::Tensor hardshrink::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & lambd) {
    static auto op = create_hardshrink_typed_handle();
    return op.redispatch(dispatchKeySet, self, lambd);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rsqrt_, name, "aten::rsqrt_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rsqrt_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rsqrt_, schema_str, "rsqrt_(Tensor(a!) self) -> Tensor(a!)")

// aten::rsqrt_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<rsqrt_::schema> create_rsqrt__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(rsqrt_::name, rsqrt_::overload_name)
      .typed<rsqrt_::schema>();
}

// aten::rsqrt_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & rsqrt_::call(at::Tensor & self) {
    static auto op = create_rsqrt__typed_handle();
    return op.call(self);
}

// aten::rsqrt_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & rsqrt_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_rsqrt__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rsqrt_out, name, "aten::rsqrt")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rsqrt_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rsqrt_out, schema_str, "rsqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::rsqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<rsqrt_out::schema> create_rsqrt_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(rsqrt_out::name, rsqrt_out::overload_name)
      .typed<rsqrt_out::schema>();
}

// aten::rsqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & rsqrt_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_rsqrt_out_typed_handle();
    return op.call(self, out);
}

// aten::rsqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & rsqrt_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_rsqrt_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(select_backward, name, "aten::select_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(select_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(select_backward, schema_str, "select_backward(Tensor grad_output, int[] input_sizes, int dim, int index) -> Tensor")

// aten::select_backward(Tensor grad_output, int[] input_sizes, int dim, int index) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<select_backward::schema> create_select_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(select_backward::name, select_backward::overload_name)
      .typed<select_backward::schema>();
}

// aten::select_backward(Tensor grad_output, int[] input_sizes, int dim, int index) -> Tensor
at::Tensor select_backward::call(const at::Tensor & grad_output, at::IntArrayRef input_sizes, int64_t dim, int64_t index) {
    static auto op = create_select_backward_typed_handle();
    return op.call(grad_output, input_sizes, dim, index);
}

// aten::select_backward(Tensor grad_output, int[] input_sizes, int dim, int index) -> Tensor
at::Tensor select_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, at::IntArrayRef input_sizes, int64_t dim, int64_t index) {
    static auto op = create_select_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, input_sizes, dim, index);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mish, name, "aten::mish")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mish, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mish, schema_str, "mish(Tensor self) -> Tensor")

// aten::mish(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<mish::schema> create_mish_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mish::name, mish::overload_name)
      .typed<mish::schema>();
}

// aten::mish(Tensor self) -> Tensor
at::Tensor mish::call(const at::Tensor & self) {
    static auto op = create_mish_typed_handle();
    return op.call(self);
}

// aten::mish(Tensor self) -> Tensor
at::Tensor mish::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_mish_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sigmoid, name, "aten::sigmoid")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sigmoid, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sigmoid, schema_str, "sigmoid(Tensor self) -> Tensor")

// aten::sigmoid(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<sigmoid::schema> create_sigmoid_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sigmoid::name, sigmoid::overload_name)
      .typed<sigmoid::schema>();
}

// aten::sigmoid(Tensor self) -> Tensor
at::Tensor sigmoid::call(const at::Tensor & self) {
    static auto op = create_sigmoid_typed_handle();
    return op.call(self);
}

// aten::sigmoid(Tensor self) -> Tensor
at::Tensor sigmoid::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_sigmoid_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sinh_out, name, "aten::sinh")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sinh_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sinh_out, schema_str, "sinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::sinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<sinh_out::schema> create_sinh_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sinh_out::name, sinh_out::overload_name)
      .typed<sinh_out::schema>();
}

// aten::sinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & sinh_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_sinh_out_typed_handle();
    return op.call(self, out);
}

// aten::sinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & sinh_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_sinh_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(detach, name, "aten::detach")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(detach, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(detach, schema_str, "detach(Tensor(a) self) -> Tensor(a)")

// aten::detach(Tensor(a) self) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<detach::schema> create_detach_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(detach::name, detach::overload_name)
      .typed<detach::schema>();
}

// aten::detach(Tensor(a) self) -> Tensor(a)
at::Tensor detach::call(const at::Tensor & self) {
    static auto op = create_detach_typed_handle();
    return op.call(self);
}

// aten::detach(Tensor(a) self) -> Tensor(a)
at::Tensor detach::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_detach_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(detach_, name, "aten::detach_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(detach_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(detach_, schema_str, "detach_(Tensor(a!) self) -> Tensor(a!)")

// aten::detach_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<detach_::schema> create_detach__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(detach_::name, detach_::overload_name)
      .typed<detach_::schema>();
}

// aten::detach_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & detach_::call(at::Tensor & self) {
    static auto op = create_detach__typed_handle();
    return op.call(self);
}

// aten::detach_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & detach_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_detach__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slice_Tensor, name, "aten::slice")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slice_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slice_Tensor, schema_str, "slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)")

// aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<slice_Tensor::schema> create_slice_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(slice_Tensor::name, slice_Tensor::overload_name)
      .typed<slice_Tensor::schema>();
}

// aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)
at::Tensor slice_Tensor::call(const at::Tensor & self, int64_t dim, c10::optional<int64_t> start, c10::optional<int64_t> end, int64_t step) {
    static auto op = create_slice_Tensor_typed_handle();
    return op.call(self, dim, start, end, step);
}

// aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)
at::Tensor slice_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, c10::optional<int64_t> start, c10::optional<int64_t> end, int64_t step) {
    static auto op = create_slice_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, start, end, step);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_softmax_backward_data, name, "aten::_softmax_backward_data")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_softmax_backward_data, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_softmax_backward_data, schema_str, "_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor")

// aten::_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_softmax_backward_data::schema> create__softmax_backward_data_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_softmax_backward_data::name, _softmax_backward_data::overload_name)
      .typed<_softmax_backward_data::schema>();
}

// aten::_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor
at::Tensor _softmax_backward_data::call(const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, const at::Tensor & self) {
    static auto op = create__softmax_backward_data_typed_handle();
    return op.call(grad_output, output, dim, self);
}

// aten::_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor
at::Tensor _softmax_backward_data::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, const at::Tensor & self) {
    static auto op = create__softmax_backward_data_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, output, dim, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(split_with_sizes, name, "aten::split_with_sizes")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(split_with_sizes, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(split_with_sizes, schema_str, "split_with_sizes(Tensor(a) self, int[] split_sizes, int dim=0) -> Tensor(a)[]")

// aten::split_with_sizes(Tensor(a) self, int[] split_sizes, int dim=0) -> Tensor(a)[]
static C10_NOINLINE c10::TypedOperatorHandle<split_with_sizes::schema> create_split_with_sizes_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(split_with_sizes::name, split_with_sizes::overload_name)
      .typed<split_with_sizes::schema>();
}

// aten::split_with_sizes(Tensor(a) self, int[] split_sizes, int dim=0) -> Tensor(a)[]
::std::vector<at::Tensor> split_with_sizes::call(const at::Tensor & self, at::IntArrayRef split_sizes, int64_t dim) {
    static auto op = create_split_with_sizes_typed_handle();
    return op.call(self, split_sizes, dim);
}

// aten::split_with_sizes(Tensor(a) self, int[] split_sizes, int dim=0) -> Tensor(a)[]
::std::vector<at::Tensor> split_with_sizes::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef split_sizes, int64_t dim) {
    static auto op = create_split_with_sizes_typed_handle();
    return op.redispatch(dispatchKeySet, self, split_sizes, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(vsplit_int, name, "aten::vsplit")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(vsplit_int, overload_name, "int")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(vsplit_int, schema_str, "vsplit.int(Tensor(a) self, int sections) -> Tensor(a)[]")

// aten::vsplit.int(Tensor(a) self, int sections) -> Tensor(a)[]
static C10_NOINLINE c10::TypedOperatorHandle<vsplit_int::schema> create_vsplit_int_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(vsplit_int::name, vsplit_int::overload_name)
      .typed<vsplit_int::schema>();
}

// aten::vsplit.int(Tensor(a) self, int sections) -> Tensor(a)[]
::std::vector<at::Tensor> vsplit_int::call(const at::Tensor & self, int64_t sections) {
    static auto op = create_vsplit_int_typed_handle();
    return op.call(self, sections);
}

// aten::vsplit.int(Tensor(a) self, int sections) -> Tensor(a)[]
::std::vector<at::Tensor> vsplit_int::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t sections) {
    static auto op = create_vsplit_int_typed_handle();
    return op.redispatch(dispatchKeySet, self, sections);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(dsplit_int, name, "aten::dsplit")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(dsplit_int, overload_name, "int")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(dsplit_int, schema_str, "dsplit.int(Tensor(a) self, int sections) -> Tensor(a)[]")

// aten::dsplit.int(Tensor(a) self, int sections) -> Tensor(a)[]
static C10_NOINLINE c10::TypedOperatorHandle<dsplit_int::schema> create_dsplit_int_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(dsplit_int::name, dsplit_int::overload_name)
      .typed<dsplit_int::schema>();
}

// aten::dsplit.int(Tensor(a) self, int sections) -> Tensor(a)[]
::std::vector<at::Tensor> dsplit_int::call(const at::Tensor & self, int64_t sections) {
    static auto op = create_dsplit_int_typed_handle();
    return op.call(self, sections);
}

// aten::dsplit.int(Tensor(a) self, int sections) -> Tensor(a)[]
::std::vector<at::Tensor> dsplit_int::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t sections) {
    static auto op = create_dsplit_int_typed_handle();
    return op.redispatch(dispatchKeySet, self, sections);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(stack, name, "aten::stack")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(stack, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(stack, schema_str, "stack(Tensor[] tensors, int dim=0) -> Tensor")

// aten::stack(Tensor[] tensors, int dim=0) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<stack::schema> create_stack_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(stack::name, stack::overload_name)
      .typed<stack::schema>();
}

// aten::stack(Tensor[] tensors, int dim=0) -> Tensor
at::Tensor stack::call(at::TensorList tensors, int64_t dim) {
    static auto op = create_stack_typed_handle();
    return op.call(tensors, dim);
}

// aten::stack(Tensor[] tensors, int dim=0) -> Tensor
at::Tensor stack::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, int64_t dim) {
    static auto op = create_stack_typed_handle();
    return op.redispatch(dispatchKeySet, tensors, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_stack, name, "aten::_stack")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_stack, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_stack, schema_str, "_stack(Tensor[] tensors, int dim=0) -> Tensor")

// aten::_stack(Tensor[] tensors, int dim=0) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_stack::schema> create__stack_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_stack::name, _stack::overload_name)
      .typed<_stack::schema>();
}

// aten::_stack(Tensor[] tensors, int dim=0) -> Tensor
at::Tensor _stack::call(at::TensorList tensors, int64_t dim) {
    static auto op = create__stack_typed_handle();
    return op.call(tensors, dim);
}

// aten::_stack(Tensor[] tensors, int dim=0) -> Tensor
at::Tensor _stack::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, int64_t dim) {
    static auto op = create__stack_typed_handle();
    return op.redispatch(dispatchKeySet, tensors, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nansum_dim_IntList, name, "aten::nansum")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nansum_dim_IntList, overload_name, "dim_IntList")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nansum_dim_IntList, schema_str, "nansum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor")

// aten::nansum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<nansum_dim_IntList::schema> create_nansum_dim_IntList_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nansum_dim_IntList::name, nansum_dim_IntList::overload_name)
      .typed<nansum_dim_IntList::schema>();
}

// aten::nansum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
at::Tensor nansum_dim_IntList::call(const at::Tensor & self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_nansum_dim_IntList_typed_handle();
    return op.call(self, dim, keepdim, dtype);
}

// aten::nansum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
at::Tensor nansum_dim_IntList::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_nansum_dim_IntList_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nansum_IntList_out, name, "aten::nansum")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nansum_IntList_out, overload_name, "IntList_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nansum_IntList_out, schema_str, "nansum.IntList_out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)")

// aten::nansum.IntList_out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<nansum_IntList_out::schema> create_nansum_IntList_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nansum_IntList_out::name, nansum_IntList_out::overload_name)
      .typed<nansum_IntList_out::schema>();
}

// aten::nansum.IntList_out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & nansum_IntList_out::call(const at::Tensor & self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
    static auto op = create_nansum_IntList_out_typed_handle();
    return op.call(self, dim, keepdim, dtype, out);
}

// aten::nansum.IntList_out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & nansum_IntList_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
    static auto op = create_nansum_IntList_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, dtype, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(square, name, "aten::square")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(square, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(square, schema_str, "square(Tensor self) -> Tensor")

// aten::square(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<square::schema> create_square_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(square::name, square::overload_name)
      .typed<square::schema>();
}

// aten::square(Tensor self) -> Tensor
at::Tensor square::call(const at::Tensor & self) {
    static auto op = create_square_typed_handle();
    return op.call(self);
}

// aten::square(Tensor self) -> Tensor
at::Tensor square::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_square_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(square_out, name, "aten::square")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(square_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(square_out, schema_str, "square.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::square.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<square_out::schema> create_square_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(square_out::name, square_out::overload_name)
      .typed<square_out::schema>();
}

// aten::square.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & square_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_square_out_typed_handle();
    return op.call(self, out);
}

// aten::square.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & square_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_square_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_correction, name, "aten::std")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_correction, overload_name, "correction")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_correction, schema_str, "std.correction(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False) -> Tensor")

// aten::std.correction(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<std_correction::schema> create_std_correction_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(std_correction::name, std_correction::overload_name)
      .typed<std_correction::schema>();
}

// aten::std.correction(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False) -> Tensor
at::Tensor std_correction::call(const at::Tensor & self, c10::optional<at::IntArrayRef> dim, c10::optional<int64_t> correction, bool keepdim) {
    static auto op = create_std_correction_typed_handle();
    return op.call(self, dim, correction, keepdim);
}

// aten::std.correction(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False) -> Tensor
at::Tensor std_correction::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> dim, c10::optional<int64_t> correction, bool keepdim) {
    static auto op = create_std_correction_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, correction, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_names_dim, name, "aten::std")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_names_dim, overload_name, "names_dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_names_dim, schema_str, "std.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor")

// aten::std.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<std_names_dim::schema> create_std_names_dim_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(std_names_dim::name, std_names_dim::overload_name)
      .typed<std_names_dim::schema>();
}

// aten::std.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
at::Tensor std_names_dim::call(const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim) {
    static auto op = create_std_names_dim_typed_handle();
    return op.call(self, dim, unbiased, keepdim);
}

// aten::std.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
at::Tensor std_names_dim::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim) {
    static auto op = create_std_names_dim_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, unbiased, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(t_, name, "aten::t_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(t_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(t_, schema_str, "t_(Tensor(a!) self) -> Tensor(a!)")

// aten::t_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<t_::schema> create_t__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(t_::name, t_::overload_name)
      .typed<t_::schema>();
}

// aten::t_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & t_::call(at::Tensor & self) {
    static auto op = create_t__typed_handle();
    return op.call(self);
}

// aten::t_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & t_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_t__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tan_, name, "aten::tan_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tan_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tan_, schema_str, "tan_(Tensor(a!) self) -> Tensor(a!)")

// aten::tan_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<tan_::schema> create_tan__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(tan_::name, tan_::overload_name)
      .typed<tan_::schema>();
}

// aten::tan_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & tan_::call(at::Tensor & self) {
    static auto op = create_tan__typed_handle();
    return op.call(self);
}

// aten::tan_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & tan_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_tan__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tanh, name, "aten::tanh")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tanh, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tanh, schema_str, "tanh(Tensor self) -> Tensor")

// aten::tanh(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<tanh::schema> create_tanh_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(tanh::name, tanh::overload_name)
      .typed<tanh::schema>();
}

// aten::tanh(Tensor self) -> Tensor
at::Tensor tanh::call(const at::Tensor & self) {
    static auto op = create_tanh_typed_handle();
    return op.call(self);
}

// aten::tanh(Tensor self) -> Tensor
at::Tensor tanh::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_tanh_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tensordot, name, "aten::tensordot")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tensordot, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tensordot, schema_str, "tensordot(Tensor self, Tensor other, int[] dims_self, int[] dims_other) -> Tensor")

// aten::tensordot(Tensor self, Tensor other, int[] dims_self, int[] dims_other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<tensordot::schema> create_tensordot_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(tensordot::name, tensordot::overload_name)
      .typed<tensordot::schema>();
}

// aten::tensordot(Tensor self, Tensor other, int[] dims_self, int[] dims_other) -> Tensor
at::Tensor tensordot::call(const at::Tensor & self, const at::Tensor & other, at::IntArrayRef dims_self, at::IntArrayRef dims_other) {
    static auto op = create_tensordot_typed_handle();
    return op.call(self, other, dims_self, dims_other);
}

// aten::tensordot(Tensor self, Tensor other, int[] dims_self, int[] dims_other) -> Tensor
at::Tensor tensordot::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::IntArrayRef dims_self, at::IntArrayRef dims_other) {
    static auto op = create_tensordot_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, dims_self, dims_other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tile, name, "aten::tile")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tile, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tile, schema_str, "tile(Tensor self, int[] dims) -> Tensor")

// aten::tile(Tensor self, int[] dims) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<tile::schema> create_tile_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(tile::name, tile::overload_name)
      .typed<tile::schema>();
}

// aten::tile(Tensor self, int[] dims) -> Tensor
at::Tensor tile::call(const at::Tensor & self, at::IntArrayRef dims) {
    static auto op = create_tile_typed_handle();
    return op.call(self, dims);
}

// aten::tile(Tensor self, int[] dims) -> Tensor
at::Tensor tile::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dims) {
    static auto op = create_tile_typed_handle();
    return op.redispatch(dispatchKeySet, self, dims);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(transpose_int, name, "aten::transpose")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(transpose_int, overload_name, "int")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(transpose_int, schema_str, "transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)")

// aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<transpose_int::schema> create_transpose_int_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(transpose_int::name, transpose_int::overload_name)
      .typed<transpose_int::schema>();
}

// aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)
at::Tensor transpose_int::call(const at::Tensor & self, int64_t dim0, int64_t dim1) {
    static auto op = create_transpose_int_typed_handle();
    return op.call(self, dim0, dim1);
}

// aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)
at::Tensor transpose_int::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim0, int64_t dim1) {
    static auto op = create_transpose_int_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim0, dim1);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(transpose_Dimname, name, "aten::transpose")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(transpose_Dimname, overload_name, "Dimname")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(transpose_Dimname, schema_str, "transpose.Dimname(Tensor(a) self, Dimname dim0, Dimname dim1) -> Tensor(a)")

// aten::transpose.Dimname(Tensor(a) self, Dimname dim0, Dimname dim1) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<transpose_Dimname::schema> create_transpose_Dimname_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(transpose_Dimname::name, transpose_Dimname::overload_name)
      .typed<transpose_Dimname::schema>();
}

// aten::transpose.Dimname(Tensor(a) self, Dimname dim0, Dimname dim1) -> Tensor(a)
at::Tensor transpose_Dimname::call(const at::Tensor & self, at::Dimname dim0, at::Dimname dim1) {
    static auto op = create_transpose_Dimname_typed_handle();
    return op.call(self, dim0, dim1);
}

// aten::transpose.Dimname(Tensor(a) self, Dimname dim0, Dimname dim1) -> Tensor(a)
at::Tensor transpose_Dimname::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim0, at::Dimname dim1) {
    static auto op = create_transpose_Dimname_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim0, dim1);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_mkldnn_transpose, name, "aten::_mkldnn_transpose")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_mkldnn_transpose, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_mkldnn_transpose, schema_str, "_mkldnn_transpose(Tensor self, int dim0, int dim1) -> Tensor")

// aten::_mkldnn_transpose(Tensor self, int dim0, int dim1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_mkldnn_transpose::schema> create__mkldnn_transpose_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_mkldnn_transpose::name, _mkldnn_transpose::overload_name)
      .typed<_mkldnn_transpose::schema>();
}

// aten::_mkldnn_transpose(Tensor self, int dim0, int dim1) -> Tensor
at::Tensor _mkldnn_transpose::call(const at::Tensor & self, int64_t dim0, int64_t dim1) {
    static auto op = create__mkldnn_transpose_typed_handle();
    return op.call(self, dim0, dim1);
}

// aten::_mkldnn_transpose(Tensor self, int dim0, int dim1) -> Tensor
at::Tensor _mkldnn_transpose::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim0, int64_t dim1) {
    static auto op = create__mkldnn_transpose_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim0, dim1);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fliplr, name, "aten::fliplr")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fliplr, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fliplr, schema_str, "fliplr(Tensor self) -> Tensor")

// aten::fliplr(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fliplr::schema> create_fliplr_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fliplr::name, fliplr::overload_name)
      .typed<fliplr::schema>();
}

// aten::fliplr(Tensor self) -> Tensor
at::Tensor fliplr::call(const at::Tensor & self) {
    static auto op = create_fliplr_typed_handle();
    return op.call(self);
}

// aten::fliplr(Tensor self) -> Tensor
at::Tensor fliplr::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_fliplr_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(trapz_x, name, "aten::trapz")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(trapz_x, overload_name, "x")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(trapz_x, schema_str, "trapz.x(Tensor y, Tensor x, *, int dim=-1) -> Tensor")

// aten::trapz.x(Tensor y, Tensor x, *, int dim=-1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<trapz_x::schema> create_trapz_x_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(trapz_x::name, trapz_x::overload_name)
      .typed<trapz_x::schema>();
}

// aten::trapz.x(Tensor y, Tensor x, *, int dim=-1) -> Tensor
at::Tensor trapz_x::call(const at::Tensor & y, const at::Tensor & x, int64_t dim) {
    static auto op = create_trapz_x_typed_handle();
    return op.call(y, x, dim);
}

// aten::trapz.x(Tensor y, Tensor x, *, int dim=-1) -> Tensor
at::Tensor trapz_x::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & y, const at::Tensor & x, int64_t dim) {
    static auto op = create_trapz_x_typed_handle();
    return op.redispatch(dispatchKeySet, y, x, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(trapz_dx, name, "aten::trapz")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(trapz_dx, overload_name, "dx")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(trapz_dx, schema_str, "trapz.dx(Tensor y, *, float dx=1, int dim=-1) -> Tensor")

// aten::trapz.dx(Tensor y, *, float dx=1, int dim=-1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<trapz_dx::schema> create_trapz_dx_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(trapz_dx::name, trapz_dx::overload_name)
      .typed<trapz_dx::schema>();
}

// aten::trapz.dx(Tensor y, *, float dx=1, int dim=-1) -> Tensor
at::Tensor trapz_dx::call(const at::Tensor & y, double dx, int64_t dim) {
    static auto op = create_trapz_dx_typed_handle();
    return op.call(y, dx, dim);
}

// aten::trapz.dx(Tensor y, *, float dx=1, int dim=-1) -> Tensor
at::Tensor trapz_dx::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & y, double dx, int64_t dim) {
    static auto op = create_trapz_dx_typed_handle();
    return op.redispatch(dispatchKeySet, y, dx, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fix, name, "aten::fix")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fix, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fix, schema_str, "fix(Tensor self) -> Tensor")

// aten::fix(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fix::schema> create_fix_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fix::name, fix::overload_name)
      .typed<fix::schema>();
}

// aten::fix(Tensor self) -> Tensor
at::Tensor fix::call(const at::Tensor & self) {
    static auto op = create_fix_typed_handle();
    return op.call(self);
}

// aten::fix(Tensor self) -> Tensor
at::Tensor fix::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_fix_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unique_dim, name, "aten::unique_dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unique_dim, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unique_dim, schema_str, "unique_dim(Tensor self, int dim, bool sorted=True, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)")

// aten::unique_dim(Tensor self, int dim, bool sorted=True, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<unique_dim::schema> create_unique_dim_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(unique_dim::name, unique_dim::overload_name)
      .typed<unique_dim::schema>();
}

// aten::unique_dim(Tensor self, int dim, bool sorted=True, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> unique_dim::call(const at::Tensor & self, int64_t dim, bool sorted, bool return_inverse, bool return_counts) {
    static auto op = create_unique_dim_typed_handle();
    return op.call(self, dim, sorted, return_inverse, return_counts);
}

// aten::unique_dim(Tensor self, int dim, bool sorted=True, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> unique_dim::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool sorted, bool return_inverse, bool return_counts) {
    static auto op = create_unique_dim_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, sorted, return_inverse, return_counts);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unique_consecutive, name, "aten::unique_consecutive")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unique_consecutive, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unique_consecutive, schema_str, "unique_consecutive(Tensor self, bool return_inverse=False, bool return_counts=False, int? dim=None) -> (Tensor, Tensor, Tensor)")

// aten::unique_consecutive(Tensor self, bool return_inverse=False, bool return_counts=False, int? dim=None) -> (Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<unique_consecutive::schema> create_unique_consecutive_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(unique_consecutive::name, unique_consecutive::overload_name)
      .typed<unique_consecutive::schema>();
}

// aten::unique_consecutive(Tensor self, bool return_inverse=False, bool return_counts=False, int? dim=None) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> unique_consecutive::call(const at::Tensor & self, bool return_inverse, bool return_counts, c10::optional<int64_t> dim) {
    static auto op = create_unique_consecutive_typed_handle();
    return op.call(self, return_inverse, return_counts, dim);
}

// aten::unique_consecutive(Tensor self, bool return_inverse=False, bool return_counts=False, int? dim=None) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> unique_consecutive::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool return_inverse, bool return_counts, c10::optional<int64_t> dim) {
    static auto op = create_unique_consecutive_typed_handle();
    return op.redispatch(dispatchKeySet, self, return_inverse, return_counts, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(vander, name, "aten::vander")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(vander, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(vander, schema_str, "vander(Tensor x, int? N=None, bool increasing=False) -> Tensor")

// aten::vander(Tensor x, int? N=None, bool increasing=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<vander::schema> create_vander_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(vander::name, vander::overload_name)
      .typed<vander::schema>();
}

// aten::vander(Tensor x, int? N=None, bool increasing=False) -> Tensor
at::Tensor vander::call(const at::Tensor & x, c10::optional<int64_t> N, bool increasing) {
    static auto op = create_vander_typed_handle();
    return op.call(x, N, increasing);
}

// aten::vander(Tensor x, int? N=None, bool increasing=False) -> Tensor
at::Tensor vander::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & x, c10::optional<int64_t> N, bool increasing) {
    static auto op = create_vander_typed_handle();
    return op.redispatch(dispatchKeySet, x, N, increasing);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_mean_correction, name, "aten::var_mean")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_mean_correction, overload_name, "correction")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_mean_correction, schema_str, "var_mean.correction(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False) -> (Tensor, Tensor)")

// aten::var_mean.correction(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<var_mean_correction::schema> create_var_mean_correction_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(var_mean_correction::name, var_mean_correction::overload_name)
      .typed<var_mean_correction::schema>();
}

// aten::var_mean.correction(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> var_mean_correction::call(const at::Tensor & self, c10::optional<at::IntArrayRef> dim, c10::optional<int64_t> correction, bool keepdim) {
    static auto op = create_var_mean_correction_typed_handle();
    return op.call(self, dim, correction, keepdim);
}

// aten::var_mean.correction(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> var_mean_correction::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> dim, c10::optional<int64_t> correction, bool keepdim) {
    static auto op = create_var_mean_correction_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, correction, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_mean_names_dim, name, "aten::var_mean")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_mean_names_dim, overload_name, "names_dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_mean_names_dim, schema_str, "var_mean.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)")

// aten::var_mean.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<var_mean_names_dim::schema> create_var_mean_names_dim_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(var_mean_names_dim::name, var_mean_names_dim::overload_name)
      .typed<var_mean_names_dim::schema>();
}

// aten::var_mean.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> var_mean_names_dim::call(const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim) {
    static auto op = create_var_mean_names_dim_typed_handle();
    return op.call(self, dim, unbiased, keepdim);
}

// aten::var_mean.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> var_mean_names_dim::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim) {
    static auto op = create_var_mean_names_dim_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, unbiased, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(view_as, name, "aten::view_as")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(view_as, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(view_as, schema_str, "view_as(Tensor(a) self, Tensor other) -> Tensor(a)")

// aten::view_as(Tensor(a) self, Tensor other) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<view_as::schema> create_view_as_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(view_as::name, view_as::overload_name)
      .typed<view_as::schema>();
}

// aten::view_as(Tensor(a) self, Tensor other) -> Tensor(a)
at::Tensor view_as::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_view_as_typed_handle();
    return op.call(self, other);
}

// aten::view_as(Tensor(a) self, Tensor other) -> Tensor(a)
at::Tensor view_as::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_view_as_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(where_Scalar, name, "aten::where")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(where_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(where_Scalar, schema_str, "where.Scalar(Tensor condition, Scalar self, Scalar other) -> Tensor")

// aten::where.Scalar(Tensor condition, Scalar self, Scalar other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<where_Scalar::schema> create_where_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(where_Scalar::name, where_Scalar::overload_name)
      .typed<where_Scalar::schema>();
}

// aten::where.Scalar(Tensor condition, Scalar self, Scalar other) -> Tensor
at::Tensor where_Scalar::call(const at::Tensor & condition, const at::Scalar & self, const at::Scalar & other) {
    static auto op = create_where_Scalar_typed_handle();
    return op.call(condition, self, other);
}

// aten::where.Scalar(Tensor condition, Scalar self, Scalar other) -> Tensor
at::Tensor where_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & condition, const at::Scalar & self, const at::Scalar & other) {
    static auto op = create_where_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, condition, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_weight_norm_cuda_interface, name, "aten::_weight_norm_cuda_interface")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_weight_norm_cuda_interface, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_weight_norm_cuda_interface, schema_str, "_weight_norm_cuda_interface(Tensor v, Tensor g, int dim=0) -> (Tensor, Tensor)")

// aten::_weight_norm_cuda_interface(Tensor v, Tensor g, int dim=0) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<_weight_norm_cuda_interface::schema> create__weight_norm_cuda_interface_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_weight_norm_cuda_interface::name, _weight_norm_cuda_interface::overload_name)
      .typed<_weight_norm_cuda_interface::schema>();
}

// aten::_weight_norm_cuda_interface(Tensor v, Tensor g, int dim=0) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> _weight_norm_cuda_interface::call(const at::Tensor & v, const at::Tensor & g, int64_t dim) {
    static auto op = create__weight_norm_cuda_interface_typed_handle();
    return op.call(v, g, dim);
}

// aten::_weight_norm_cuda_interface(Tensor v, Tensor g, int dim=0) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> _weight_norm_cuda_interface::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & v, const at::Tensor & g, int64_t dim) {
    static auto op = create__weight_norm_cuda_interface_typed_handle();
    return op.redispatch(dispatchKeySet, v, g, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_dirichlet_grad, name, "aten::_dirichlet_grad")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_dirichlet_grad, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_dirichlet_grad, schema_str, "_dirichlet_grad(Tensor x, Tensor alpha, Tensor total) -> Tensor")

// aten::_dirichlet_grad(Tensor x, Tensor alpha, Tensor total) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_dirichlet_grad::schema> create__dirichlet_grad_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_dirichlet_grad::name, _dirichlet_grad::overload_name)
      .typed<_dirichlet_grad::schema>();
}

// aten::_dirichlet_grad(Tensor x, Tensor alpha, Tensor total) -> Tensor
at::Tensor _dirichlet_grad::call(const at::Tensor & x, const at::Tensor & alpha, const at::Tensor & total) {
    static auto op = create__dirichlet_grad_typed_handle();
    return op.call(x, alpha, total);
}

// aten::_dirichlet_grad(Tensor x, Tensor alpha, Tensor total) -> Tensor
at::Tensor _dirichlet_grad::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & x, const at::Tensor & alpha, const at::Tensor & total) {
    static auto op = create__dirichlet_grad_typed_handle();
    return op.redispatch(dispatchKeySet, x, alpha, total);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(norm_ScalarOpt_dim, name, "aten::norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(norm_ScalarOpt_dim, overload_name, "ScalarOpt_dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(norm_ScalarOpt_dim, schema_str, "norm.ScalarOpt_dim(Tensor self, Scalar? p, int[1] dim, bool keepdim=False) -> Tensor")

// aten::norm.ScalarOpt_dim(Tensor self, Scalar? p, int[1] dim, bool keepdim=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<norm_ScalarOpt_dim::schema> create_norm_ScalarOpt_dim_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(norm_ScalarOpt_dim::name, norm_ScalarOpt_dim::overload_name)
      .typed<norm_ScalarOpt_dim::schema>();
}

// aten::norm.ScalarOpt_dim(Tensor self, Scalar? p, int[1] dim, bool keepdim=False) -> Tensor
at::Tensor norm_ScalarOpt_dim::call(const at::Tensor & self, const c10::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim) {
    static auto op = create_norm_ScalarOpt_dim_typed_handle();
    return op.call(self, p, dim, keepdim);
}

// aten::norm.ScalarOpt_dim(Tensor self, Scalar? p, int[1] dim, bool keepdim=False) -> Tensor
at::Tensor norm_ScalarOpt_dim::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim) {
    static auto op = create_norm_ScalarOpt_dim_typed_handle();
    return op.redispatch(dispatchKeySet, self, p, dim, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(norm_dtype_out, name, "aten::norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(norm_dtype_out, overload_name, "dtype_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(norm_dtype_out, schema_str, "norm.dtype_out(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)")

// aten::norm.dtype_out(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<norm_dtype_out::schema> create_norm_dtype_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(norm_dtype_out::name, norm_dtype_out::overload_name)
      .typed<norm_dtype_out::schema>();
}

// aten::norm.dtype_out(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)
at::Tensor & norm_dtype_out::call(const at::Tensor & self, const c10::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, at::ScalarType dtype, at::Tensor & out) {
    static auto op = create_norm_dtype_out_typed_handle();
    return op.call(self, p, dim, keepdim, dtype, out);
}

// aten::norm.dtype_out(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)
at::Tensor & norm_dtype_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, at::ScalarType dtype, at::Tensor & out) {
    static auto op = create_norm_dtype_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, p, dim, keepdim, dtype, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(frexp_Tensor, name, "aten::frexp")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(frexp_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(frexp_Tensor, schema_str, "frexp.Tensor(Tensor self) -> (Tensor mantissa, Tensor exponent)")

// aten::frexp.Tensor(Tensor self) -> (Tensor mantissa, Tensor exponent)
static C10_NOINLINE c10::TypedOperatorHandle<frexp_Tensor::schema> create_frexp_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(frexp_Tensor::name, frexp_Tensor::overload_name)
      .typed<frexp_Tensor::schema>();
}

// aten::frexp.Tensor(Tensor self) -> (Tensor mantissa, Tensor exponent)
::std::tuple<at::Tensor,at::Tensor> frexp_Tensor::call(const at::Tensor & self) {
    static auto op = create_frexp_Tensor_typed_handle();
    return op.call(self);
}

// aten::frexp.Tensor(Tensor self) -> (Tensor mantissa, Tensor exponent)
::std::tuple<at::Tensor,at::Tensor> frexp_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_frexp_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(frexp_Tensor_out, name, "aten::frexp")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(frexp_Tensor_out, overload_name, "Tensor_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(frexp_Tensor_out, schema_str, "frexp.Tensor_out(Tensor self, *, Tensor(a!) mantissa, Tensor(b!) exponent) -> (Tensor(a!) mantissa, Tensor(b!) exponent)")

// aten::frexp.Tensor_out(Tensor self, *, Tensor(a!) mantissa, Tensor(b!) exponent) -> (Tensor(a!) mantissa, Tensor(b!) exponent)
static C10_NOINLINE c10::TypedOperatorHandle<frexp_Tensor_out::schema> create_frexp_Tensor_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(frexp_Tensor_out::name, frexp_Tensor_out::overload_name)
      .typed<frexp_Tensor_out::schema>();
}

// aten::frexp.Tensor_out(Tensor self, *, Tensor(a!) mantissa, Tensor(b!) exponent) -> (Tensor(a!) mantissa, Tensor(b!) exponent)
::std::tuple<at::Tensor &,at::Tensor &> frexp_Tensor_out::call(const at::Tensor & self, at::Tensor & mantissa, at::Tensor & exponent) {
    static auto op = create_frexp_Tensor_out_typed_handle();
    return op.call(self, mantissa, exponent);
}

// aten::frexp.Tensor_out(Tensor self, *, Tensor(a!) mantissa, Tensor(b!) exponent) -> (Tensor(a!) mantissa, Tensor(b!) exponent)
::std::tuple<at::Tensor &,at::Tensor &> frexp_Tensor_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & mantissa, at::Tensor & exponent) {
    static auto op = create_frexp_Tensor_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, mantissa, exponent);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(frobenius_norm, name, "aten::frobenius_norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(frobenius_norm, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(frobenius_norm, schema_str, "frobenius_norm(Tensor self) -> Tensor")

// aten::frobenius_norm(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<frobenius_norm::schema> create_frobenius_norm_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(frobenius_norm::name, frobenius_norm::overload_name)
      .typed<frobenius_norm::schema>();
}

// aten::frobenius_norm(Tensor self) -> Tensor
at::Tensor frobenius_norm::call(const at::Tensor & self) {
    static auto op = create_frobenius_norm_typed_handle();
    return op.call(self);
}

// aten::frobenius_norm(Tensor self) -> Tensor
at::Tensor frobenius_norm::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_frobenius_norm_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(frobenius_norm_out, name, "aten::frobenius_norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(frobenius_norm_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(frobenius_norm_out, schema_str, "frobenius_norm.out(Tensor self, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)")

// aten::frobenius_norm.out(Tensor self, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<frobenius_norm_out::schema> create_frobenius_norm_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(frobenius_norm_out::name, frobenius_norm_out::overload_name)
      .typed<frobenius_norm_out::schema>();
}

// aten::frobenius_norm.out(Tensor self, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & frobenius_norm_out::call(const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
    static auto op = create_frobenius_norm_out_typed_handle();
    return op.call(self, dim, keepdim, out);
}

// aten::frobenius_norm.out(Tensor self, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & frobenius_norm_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
    static auto op = create_frobenius_norm_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clone, name, "aten::clone")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clone, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clone, schema_str, "clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor")

// aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<clone::schema> create_clone_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(clone::name, clone::overload_name)
      .typed<clone::schema>();
}

// aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor
at::Tensor clone::call(const at::Tensor & self, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create_clone_typed_handle();
    return op.call(self, memory_format);
}

// aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor
at::Tensor clone::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create_clone_typed_handle();
    return op.redispatch(dispatchKeySet, self, memory_format);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(positive, name, "aten::positive")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(positive, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(positive, schema_str, "positive(Tensor(a) self) -> Tensor(a)")

// aten::positive(Tensor(a) self) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<positive::schema> create_positive_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(positive::name, positive::overload_name)
      .typed<positive::schema>();
}

// aten::positive(Tensor(a) self) -> Tensor(a)
at::Tensor positive::call(const at::Tensor & self) {
    static auto op = create_positive_typed_handle();
    return op.call(self);
}

// aten::positive(Tensor(a) self) -> Tensor(a)
at::Tensor positive::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_positive_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(resize_as_, name, "aten::resize_as_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(resize_as_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(resize_as_, schema_str, "resize_as_(Tensor(a!) self, Tensor the_template, *, MemoryFormat? memory_format=None) -> Tensor(a!)")

// aten::resize_as_(Tensor(a!) self, Tensor the_template, *, MemoryFormat? memory_format=None) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<resize_as_::schema> create_resize_as__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(resize_as_::name, resize_as_::overload_name)
      .typed<resize_as_::schema>();
}

// aten::resize_as_(Tensor(a!) self, Tensor the_template, *, MemoryFormat? memory_format=None) -> Tensor(a!)
const at::Tensor & resize_as_::call(const at::Tensor & self, const at::Tensor & the_template, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create_resize_as__typed_handle();
    return op.call(self, the_template, memory_format);
}

// aten::resize_as_(Tensor(a!) self, Tensor the_template, *, MemoryFormat? memory_format=None) -> Tensor(a!)
const at::Tensor & resize_as_::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & the_template, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create_resize_as__typed_handle();
    return op.redispatch(dispatchKeySet, self, the_template, memory_format);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sub_Tensor, name, "aten::sub")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sub_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sub_Tensor, schema_str, "sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor")

// aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<sub_Tensor::schema> create_sub_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sub_Tensor::name, sub_Tensor::overload_name)
      .typed<sub_Tensor::schema>();
}

// aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
at::Tensor sub_Tensor::call(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
    static auto op = create_sub_Tensor_typed_handle();
    return op.call(self, other, alpha);
}

// aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
at::Tensor sub_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
    static auto op = create_sub_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sub__Scalar, name, "aten::sub_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sub__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sub__Scalar, schema_str, "sub_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)")

// aten::sub_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<sub__Scalar::schema> create_sub__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sub__Scalar::name, sub__Scalar::overload_name)
      .typed<sub__Scalar::schema>();
}

// aten::sub_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
at::Tensor & sub__Scalar::call(at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
    static auto op = create_sub__Scalar_typed_handle();
    return op.call(self, other, alpha);
}

// aten::sub_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
at::Tensor & sub__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
    static auto op = create_sub__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(subtract_Tensor, name, "aten::subtract")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(subtract_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(subtract_Tensor, schema_str, "subtract.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor")

// aten::subtract.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<subtract_Tensor::schema> create_subtract_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(subtract_Tensor::name, subtract_Tensor::overload_name)
      .typed<subtract_Tensor::schema>();
}

// aten::subtract.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
at::Tensor subtract_Tensor::call(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
    static auto op = create_subtract_Tensor_typed_handle();
    return op.call(self, other, alpha);
}

// aten::subtract.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
at::Tensor subtract_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
    static auto op = create_subtract_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addmm_, name, "aten::addmm_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addmm_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addmm_, schema_str, "addmm_(Tensor(a!) self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)")

// aten::addmm_(Tensor(a!) self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<addmm_::schema> create_addmm__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(addmm_::name, addmm_::overload_name)
      .typed<addmm_::schema>();
}

// aten::addmm_(Tensor(a!) self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
at::Tensor & addmm_::call(at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha) {
    static auto op = create_addmm__typed_handle();
    return op.call(self, mat1, mat2, beta, alpha);
}

// aten::addmm_(Tensor(a!) self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
at::Tensor & addmm_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha) {
    static auto op = create_addmm__typed_handle();
    return op.redispatch(dispatchKeySet, self, mat1, mat2, beta, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sparse_csr_tensor_crow_col_value, name, "aten::sparse_csr_tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sparse_csr_tensor_crow_col_value, overload_name, "crow_col_value")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sparse_csr_tensor_crow_col_value, schema_str, "sparse_csr_tensor.crow_col_value(Tensor crow_indices, Tensor col_indices, Tensor values, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor")

// aten::sparse_csr_tensor.crow_col_value(Tensor crow_indices, Tensor col_indices, Tensor values, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<sparse_csr_tensor_crow_col_value::schema> create_sparse_csr_tensor_crow_col_value_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sparse_csr_tensor_crow_col_value::name, sparse_csr_tensor_crow_col_value::overload_name)
      .typed<sparse_csr_tensor_crow_col_value::schema>();
}

// aten::sparse_csr_tensor.crow_col_value(Tensor crow_indices, Tensor col_indices, Tensor values, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
at::Tensor sparse_csr_tensor_crow_col_value::call(const at::Tensor & crow_indices, const at::Tensor & col_indices, const at::Tensor & values, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_sparse_csr_tensor_crow_col_value_typed_handle();
    return op.call(crow_indices, col_indices, values, dtype, layout, device, pin_memory);
}

// aten::sparse_csr_tensor.crow_col_value(Tensor crow_indices, Tensor col_indices, Tensor values, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
at::Tensor sparse_csr_tensor_crow_col_value::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & crow_indices, const at::Tensor & col_indices, const at::Tensor & values, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_sparse_csr_tensor_crow_col_value_typed_handle();
    return op.redispatch(dispatchKeySet, crow_indices, col_indices, values, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sparse_coo_tensor_indices_size, name, "aten::sparse_coo_tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sparse_coo_tensor_indices_size, overload_name, "indices_size")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sparse_coo_tensor_indices_size, schema_str, "sparse_coo_tensor.indices_size(Tensor indices, Tensor values, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::sparse_coo_tensor.indices_size(Tensor indices, Tensor values, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<sparse_coo_tensor_indices_size::schema> create_sparse_coo_tensor_indices_size_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sparse_coo_tensor_indices_size::name, sparse_coo_tensor_indices_size::overload_name)
      .typed<sparse_coo_tensor_indices_size::schema>();
}

// aten::sparse_coo_tensor.indices_size(Tensor indices, Tensor values, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor sparse_coo_tensor_indices_size::call(const at::Tensor & indices, const at::Tensor & values, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_sparse_coo_tensor_indices_size_typed_handle();
    return op.call(indices, values, size, dtype, layout, device, pin_memory);
}

// aten::sparse_coo_tensor.indices_size(Tensor indices, Tensor values, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor sparse_coo_tensor_indices_size::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & indices, const at::Tensor & values, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_sparse_coo_tensor_indices_size_typed_handle();
    return op.redispatch(dispatchKeySet, indices, values, size, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(dense_dim, name, "aten::dense_dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(dense_dim, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(dense_dim, schema_str, "dense_dim(Tensor self) -> int")

// aten::dense_dim(Tensor self) -> int
static C10_NOINLINE c10::TypedOperatorHandle<dense_dim::schema> create_dense_dim_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(dense_dim::name, dense_dim::overload_name)
      .typed<dense_dim::schema>();
}

// aten::dense_dim(Tensor self) -> int
int64_t dense_dim::call(const at::Tensor & self) {
    static auto op = create_dense_dim_typed_handle();
    return op.call(self);
}

// aten::dense_dim(Tensor self) -> int
int64_t dense_dim::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_dense_dim_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_dimV, name, "aten::_dimV")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_dimV, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_dimV, schema_str, "_dimV(Tensor self) -> int")

// aten::_dimV(Tensor self) -> int
static C10_NOINLINE c10::TypedOperatorHandle<_dimV::schema> create__dimV_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_dimV::name, _dimV::overload_name)
      .typed<_dimV::schema>();
}

// aten::_dimV(Tensor self) -> int
int64_t _dimV::call(const at::Tensor & self) {
    static auto op = create__dimV_typed_handle();
    return op.call(self);
}

// aten::_dimV(Tensor self) -> int
int64_t _dimV::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create__dimV_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(coalesce, name, "aten::coalesce")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(coalesce, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(coalesce, schema_str, "coalesce(Tensor(a) self) -> Tensor(a)")

// aten::coalesce(Tensor(a) self) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<coalesce::schema> create_coalesce_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(coalesce::name, coalesce::overload_name)
      .typed<coalesce::schema>();
}

// aten::coalesce(Tensor(a) self) -> Tensor(a)
at::Tensor coalesce::call(const at::Tensor & self) {
    static auto op = create_coalesce_typed_handle();
    return op.call(self);
}

// aten::coalesce(Tensor(a) self) -> Tensor(a)
at::Tensor coalesce::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_coalesce_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_indices, name, "aten::_indices")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_indices, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_indices, schema_str, "_indices(Tensor(a) self) -> Tensor(a)")

// aten::_indices(Tensor(a) self) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<_indices::schema> create__indices_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_indices::name, _indices::overload_name)
      .typed<_indices::schema>();
}

// aten::_indices(Tensor(a) self) -> Tensor(a)
at::Tensor _indices::call(const at::Tensor & self) {
    static auto op = create__indices_typed_handle();
    return op.call(self);
}

// aten::_indices(Tensor(a) self) -> Tensor(a)
at::Tensor _indices::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create__indices_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_coalesced_, name, "aten::_coalesced_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_coalesced_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_coalesced_, schema_str, "_coalesced_(Tensor(a!) self, bool coalesced) -> Tensor(a!)")

// aten::_coalesced_(Tensor(a!) self, bool coalesced) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<_coalesced_::schema> create__coalesced__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_coalesced_::name, _coalesced_::overload_name)
      .typed<_coalesced_::schema>();
}

// aten::_coalesced_(Tensor(a!) self, bool coalesced) -> Tensor(a!)
at::Tensor & _coalesced_::call(at::Tensor & self, bool coalesced) {
    static auto op = create__coalesced__typed_handle();
    return op.call(self, coalesced);
}

// aten::_coalesced_(Tensor(a!) self, bool coalesced) -> Tensor(a!)
at::Tensor & _coalesced_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, bool coalesced) {
    static auto op = create__coalesced__typed_handle();
    return op.redispatch(dispatchKeySet, self, coalesced);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_reorder_conv2d_weight, name, "aten::mkldnn_reorder_conv2d_weight")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_reorder_conv2d_weight, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_reorder_conv2d_weight, schema_str, "mkldnn_reorder_conv2d_weight(Tensor self, int[2] padding=0, int[2] stride=1, int[2] dilation=1, int groups=1) -> Tensor")

// aten::mkldnn_reorder_conv2d_weight(Tensor self, int[2] padding=0, int[2] stride=1, int[2] dilation=1, int groups=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<mkldnn_reorder_conv2d_weight::schema> create_mkldnn_reorder_conv2d_weight_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mkldnn_reorder_conv2d_weight::name, mkldnn_reorder_conv2d_weight::overload_name)
      .typed<mkldnn_reorder_conv2d_weight::schema>();
}

// aten::mkldnn_reorder_conv2d_weight(Tensor self, int[2] padding=0, int[2] stride=1, int[2] dilation=1, int groups=1) -> Tensor
at::Tensor mkldnn_reorder_conv2d_weight::call(const at::Tensor & self, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups) {
    static auto op = create_mkldnn_reorder_conv2d_weight_typed_handle();
    return op.call(self, padding, stride, dilation, groups);
}

// aten::mkldnn_reorder_conv2d_weight(Tensor self, int[2] padding=0, int[2] stride=1, int[2] dilation=1, int groups=1) -> Tensor
at::Tensor mkldnn_reorder_conv2d_weight::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups) {
    static auto op = create_mkldnn_reorder_conv2d_weight_typed_handle();
    return op.redispatch(dispatchKeySet, self, padding, stride, dilation, groups);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantize_per_channel, name, "aten::quantize_per_channel")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantize_per_channel, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantize_per_channel, schema_str, "quantize_per_channel(Tensor self, Tensor scales, Tensor zero_points, int axis, ScalarType dtype) -> Tensor")

// aten::quantize_per_channel(Tensor self, Tensor scales, Tensor zero_points, int axis, ScalarType dtype) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<quantize_per_channel::schema> create_quantize_per_channel_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(quantize_per_channel::name, quantize_per_channel::overload_name)
      .typed<quantize_per_channel::schema>();
}

// aten::quantize_per_channel(Tensor self, Tensor scales, Tensor zero_points, int axis, ScalarType dtype) -> Tensor
at::Tensor quantize_per_channel::call(const at::Tensor & self, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, at::ScalarType dtype) {
    static auto op = create_quantize_per_channel_typed_handle();
    return op.call(self, scales, zero_points, axis, dtype);
}

// aten::quantize_per_channel(Tensor self, Tensor scales, Tensor zero_points, int axis, ScalarType dtype) -> Tensor
at::Tensor quantize_per_channel::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, at::ScalarType dtype) {
    static auto op = create_quantize_per_channel_typed_handle();
    return op.redispatch(dispatchKeySet, self, scales, zero_points, axis, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(q_per_channel_zero_points, name, "aten::q_per_channel_zero_points")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(q_per_channel_zero_points, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(q_per_channel_zero_points, schema_str, "q_per_channel_zero_points(Tensor self) -> Tensor")

// aten::q_per_channel_zero_points(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<q_per_channel_zero_points::schema> create_q_per_channel_zero_points_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(q_per_channel_zero_points::name, q_per_channel_zero_points::overload_name)
      .typed<q_per_channel_zero_points::schema>();
}

// aten::q_per_channel_zero_points(Tensor self) -> Tensor
at::Tensor q_per_channel_zero_points::call(const at::Tensor & self) {
    static auto op = create_q_per_channel_zero_points_typed_handle();
    return op.call(self);
}

// aten::q_per_channel_zero_points(Tensor self) -> Tensor
at::Tensor q_per_channel_zero_points::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_q_per_channel_zero_points_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fake_quantize_per_tensor_affine, name, "aten::fake_quantize_per_tensor_affine")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fake_quantize_per_tensor_affine, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fake_quantize_per_tensor_affine, schema_str, "fake_quantize_per_tensor_affine(Tensor self, float scale, int zero_point, int quant_min, int quant_max) -> Tensor")

// aten::fake_quantize_per_tensor_affine(Tensor self, float scale, int zero_point, int quant_min, int quant_max) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fake_quantize_per_tensor_affine::schema> create_fake_quantize_per_tensor_affine_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fake_quantize_per_tensor_affine::name, fake_quantize_per_tensor_affine::overload_name)
      .typed<fake_quantize_per_tensor_affine::schema>();
}

// aten::fake_quantize_per_tensor_affine(Tensor self, float scale, int zero_point, int quant_min, int quant_max) -> Tensor
at::Tensor fake_quantize_per_tensor_affine::call(const at::Tensor & self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max) {
    static auto op = create_fake_quantize_per_tensor_affine_typed_handle();
    return op.call(self, scale, zero_point, quant_min, quant_max);
}

// aten::fake_quantize_per_tensor_affine(Tensor self, float scale, int zero_point, int quant_min, int quant_max) -> Tensor
at::Tensor fake_quantize_per_tensor_affine::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max) {
    static auto op = create_fake_quantize_per_tensor_affine_typed_handle();
    return op.redispatch(dispatchKeySet, self, scale, zero_point, quant_min, quant_max);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fake_quantize_learnable_per_channel_affine, name, "aten::_fake_quantize_learnable_per_channel_affine")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fake_quantize_learnable_per_channel_affine, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fake_quantize_learnable_per_channel_affine, schema_str, "_fake_quantize_learnable_per_channel_affine(Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max, float grad_factor=1.0) -> Tensor")

// aten::_fake_quantize_learnable_per_channel_affine(Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max, float grad_factor=1.0) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_fake_quantize_learnable_per_channel_affine::schema> create__fake_quantize_learnable_per_channel_affine_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_fake_quantize_learnable_per_channel_affine::name, _fake_quantize_learnable_per_channel_affine::overload_name)
      .typed<_fake_quantize_learnable_per_channel_affine::schema>();
}

// aten::_fake_quantize_learnable_per_channel_affine(Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max, float grad_factor=1.0) -> Tensor
at::Tensor _fake_quantize_learnable_per_channel_affine::call(const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max, double grad_factor) {
    static auto op = create__fake_quantize_learnable_per_channel_affine_typed_handle();
    return op.call(self, scale, zero_point, axis, quant_min, quant_max, grad_factor);
}

// aten::_fake_quantize_learnable_per_channel_affine(Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max, float grad_factor=1.0) -> Tensor
at::Tensor _fake_quantize_learnable_per_channel_affine::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max, double grad_factor) {
    static auto op = create__fake_quantize_learnable_per_channel_affine_typed_handle();
    return op.redispatch(dispatchKeySet, self, scale, zero_point, axis, quant_min, quant_max, grad_factor);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(to_device, name, "aten::to")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(to_device, overload_name, "device")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(to_device, schema_str, "to.device(Tensor(a) self, Device device, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)")

// aten::to.device(Tensor(a) self, Device device, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<to_device::schema> create_to_device_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(to_device::name, to_device::overload_name)
      .typed<to_device::schema>();
}

// aten::to.device(Tensor(a) self, Device device, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)
at::Tensor to_device::call(const at::Tensor & self, at::Device device, at::ScalarType dtype, bool non_blocking, bool copy, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create_to_device_typed_handle();
    return op.call(self, device, dtype, non_blocking, copy, memory_format);
}

// aten::to.device(Tensor(a) self, Device device, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)
at::Tensor to_device::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Device device, at::ScalarType dtype, bool non_blocking, bool copy, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create_to_device_typed_handle();
    return op.redispatch(dispatchKeySet, self, device, dtype, non_blocking, copy, memory_format);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(combinations, name, "aten::combinations")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(combinations, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(combinations, schema_str, "combinations(Tensor self, int r=2, bool with_replacement=False) -> Tensor")

// aten::combinations(Tensor self, int r=2, bool with_replacement=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<combinations::schema> create_combinations_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(combinations::name, combinations::overload_name)
      .typed<combinations::schema>();
}

// aten::combinations(Tensor self, int r=2, bool with_replacement=False) -> Tensor
at::Tensor combinations::call(const at::Tensor & self, int64_t r, bool with_replacement) {
    static auto op = create_combinations_typed_handle();
    return op.call(self, r, with_replacement);
}

// aten::combinations(Tensor self, int r=2, bool with_replacement=False) -> Tensor
at::Tensor combinations::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t r, bool with_replacement) {
    static auto op = create_combinations_typed_handle();
    return op.redispatch(dispatchKeySet, self, r, with_replacement);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(item, name, "aten::item")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(item, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(item, schema_str, "item(Tensor self) -> Scalar")

// aten::item(Tensor self) -> Scalar
static C10_NOINLINE c10::TypedOperatorHandle<item::schema> create_item_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(item::name, item::overload_name)
      .typed<item::schema>();
}

// aten::item(Tensor self) -> Scalar
at::Scalar item::call(const at::Tensor & self) {
    static auto op = create_item_typed_handle();
    return op.call(self);
}

// aten::item(Tensor self) -> Scalar
at::Scalar item::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_item_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(result_type_Scalar_Tensor, name, "aten::result_type")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(result_type_Scalar_Tensor, overload_name, "Scalar_Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(result_type_Scalar_Tensor, schema_str, "result_type.Scalar_Tensor(Scalar scalar, Tensor tensor) -> ScalarType")

// aten::result_type.Scalar_Tensor(Scalar scalar, Tensor tensor) -> ScalarType
static C10_NOINLINE c10::TypedOperatorHandle<result_type_Scalar_Tensor::schema> create_result_type_Scalar_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(result_type_Scalar_Tensor::name, result_type_Scalar_Tensor::overload_name)
      .typed<result_type_Scalar_Tensor::schema>();
}

// aten::result_type.Scalar_Tensor(Scalar scalar, Tensor tensor) -> ScalarType
at::ScalarType result_type_Scalar_Tensor::call(const at::Scalar & scalar, const at::Tensor & tensor) {
    static auto op = create_result_type_Scalar_Tensor_typed_handle();
    return op.call(scalar, tensor);
}

// aten::result_type.Scalar_Tensor(Scalar scalar, Tensor tensor) -> ScalarType
at::ScalarType result_type_Scalar_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Scalar & scalar, const at::Tensor & tensor) {
    static auto op = create_result_type_Scalar_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, scalar, tensor);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_thnn_fused_lstm_cell, name, "aten::_thnn_fused_lstm_cell")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_thnn_fused_lstm_cell, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_thnn_fused_lstm_cell, schema_str, "_thnn_fused_lstm_cell(Tensor input_gates, Tensor hidden_gates, Tensor cx, Tensor? input_bias=None, Tensor? hidden_bias=None) -> (Tensor, Tensor, Tensor)")

// aten::_thnn_fused_lstm_cell(Tensor input_gates, Tensor hidden_gates, Tensor cx, Tensor? input_bias=None, Tensor? hidden_bias=None) -> (Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<_thnn_fused_lstm_cell::schema> create__thnn_fused_lstm_cell_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_thnn_fused_lstm_cell::name, _thnn_fused_lstm_cell::overload_name)
      .typed<_thnn_fused_lstm_cell::schema>();
}

// aten::_thnn_fused_lstm_cell(Tensor input_gates, Tensor hidden_gates, Tensor cx, Tensor? input_bias=None, Tensor? hidden_bias=None) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _thnn_fused_lstm_cell::call(const at::Tensor & input_gates, const at::Tensor & hidden_gates, const at::Tensor & cx, const c10::optional<at::Tensor> & input_bias, const c10::optional<at::Tensor> & hidden_bias) {
    static auto op = create__thnn_fused_lstm_cell_typed_handle();
    return op.call(input_gates, hidden_gates, cx, input_bias, hidden_bias);
}

// aten::_thnn_fused_lstm_cell(Tensor input_gates, Tensor hidden_gates, Tensor cx, Tensor? input_bias=None, Tensor? hidden_bias=None) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _thnn_fused_lstm_cell::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input_gates, const at::Tensor & hidden_gates, const at::Tensor & cx, const c10::optional<at::Tensor> & input_bias, const c10::optional<at::Tensor> & hidden_bias) {
    static auto op = create__thnn_fused_lstm_cell_typed_handle();
    return op.redispatch(dispatchKeySet, input_gates, hidden_gates, cx, input_bias, hidden_bias);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rnn_tanh_data, name, "aten::rnn_tanh")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rnn_tanh_data, overload_name, "data")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rnn_tanh_data, schema_str, "rnn_tanh.data(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)")

// aten::rnn_tanh.data(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<rnn_tanh_data::schema> create_rnn_tanh_data_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(rnn_tanh_data::name, rnn_tanh_data::overload_name)
      .typed<rnn_tanh_data::schema>();
}

// aten::rnn_tanh.data(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> rnn_tanh_data::call(const at::Tensor & data, const at::Tensor & batch_sizes, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {
    static auto op = create_rnn_tanh_data_typed_handle();
    return op.call(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
}

// aten::rnn_tanh.data(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> rnn_tanh_data::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & data, const at::Tensor & batch_sizes, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {
    static auto op = create_rnn_tanh_data_typed_handle();
    return op.redispatch(dispatchKeySet, data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rnn_relu_input, name, "aten::rnn_relu")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rnn_relu_input, overload_name, "input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rnn_relu_input, schema_str, "rnn_relu.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)")

// aten::rnn_relu.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<rnn_relu_input::schema> create_rnn_relu_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(rnn_relu_input::name, rnn_relu_input::overload_name)
      .typed<rnn_relu_input::schema>();
}

// aten::rnn_relu.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> rnn_relu_input::call(const at::Tensor & input, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
    static auto op = create_rnn_relu_input_typed_handle();
    return op.call(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}

// aten::rnn_relu.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> rnn_relu_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
    static auto op = create_rnn_relu_input_typed_handle();
    return op.redispatch(dispatchKeySet, input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rnn_relu_cell, name, "aten::rnn_relu_cell")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rnn_relu_cell, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rnn_relu_cell, schema_str, "rnn_relu_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor")

// aten::rnn_relu_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<rnn_relu_cell::schema> create_rnn_relu_cell_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(rnn_relu_cell::name, rnn_relu_cell::overload_name)
      .typed<rnn_relu_cell::schema>();
}

// aten::rnn_relu_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor
at::Tensor rnn_relu_cell::call(const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const c10::optional<at::Tensor> & b_ih, const c10::optional<at::Tensor> & b_hh) {
    static auto op = create_rnn_relu_cell_typed_handle();
    return op.call(input, hx, w_ih, w_hh, b_ih, b_hh);
}

// aten::rnn_relu_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor
at::Tensor rnn_relu_cell::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const c10::optional<at::Tensor> & b_ih, const c10::optional<at::Tensor> & b_hh) {
    static auto op = create_rnn_relu_cell_typed_handle();
    return op.redispatch(dispatchKeySet, input, hx, w_ih, w_hh, b_ih, b_hh);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_pad_packed_sequence, name, "aten::_pad_packed_sequence")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_pad_packed_sequence, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_pad_packed_sequence, schema_str, "_pad_packed_sequence(Tensor data, Tensor batch_sizes, bool batch_first, Scalar padding_value, int total_length) -> (Tensor, Tensor)")

// aten::_pad_packed_sequence(Tensor data, Tensor batch_sizes, bool batch_first, Scalar padding_value, int total_length) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<_pad_packed_sequence::schema> create__pad_packed_sequence_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_pad_packed_sequence::name, _pad_packed_sequence::overload_name)
      .typed<_pad_packed_sequence::schema>();
}

// aten::_pad_packed_sequence(Tensor data, Tensor batch_sizes, bool batch_first, Scalar padding_value, int total_length) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> _pad_packed_sequence::call(const at::Tensor & data, const at::Tensor & batch_sizes, bool batch_first, const at::Scalar & padding_value, int64_t total_length) {
    static auto op = create__pad_packed_sequence_typed_handle();
    return op.call(data, batch_sizes, batch_first, padding_value, total_length);
}

// aten::_pad_packed_sequence(Tensor data, Tensor batch_sizes, bool batch_first, Scalar padding_value, int total_length) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> _pad_packed_sequence::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & data, const at::Tensor & batch_sizes, bool batch_first, const at::Scalar & padding_value, int64_t total_length) {
    static auto op = create__pad_packed_sequence_typed_handle();
    return op.redispatch(dispatchKeySet, data, batch_sizes, batch_first, padding_value, total_length);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(set__source_Tensor, name, "aten::set_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(set__source_Tensor, overload_name, "source_Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(set__source_Tensor, schema_str, "set_.source_Tensor(Tensor(a!) self, Tensor source) -> Tensor(a!)")

// aten::set_.source_Tensor(Tensor(a!) self, Tensor source) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<set__source_Tensor::schema> create_set__source_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(set__source_Tensor::name, set__source_Tensor::overload_name)
      .typed<set__source_Tensor::schema>();
}

// aten::set_.source_Tensor(Tensor(a!) self, Tensor source) -> Tensor(a!)
at::Tensor & set__source_Tensor::call(at::Tensor & self, const at::Tensor & source) {
    static auto op = create_set__source_Tensor_typed_handle();
    return op.call(self, source);
}

// aten::set_.source_Tensor(Tensor(a!) self, Tensor source) -> Tensor(a!)
at::Tensor & set__source_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & source) {
    static auto op = create_set__source_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, source);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(masked_fill_Tensor, name, "aten::masked_fill")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(masked_fill_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(masked_fill_Tensor, schema_str, "masked_fill.Tensor(Tensor self, Tensor mask, Tensor value) -> Tensor")

// aten::masked_fill.Tensor(Tensor self, Tensor mask, Tensor value) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<masked_fill_Tensor::schema> create_masked_fill_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(masked_fill_Tensor::name, masked_fill_Tensor::overload_name)
      .typed<masked_fill_Tensor::schema>();
}

// aten::masked_fill.Tensor(Tensor self, Tensor mask, Tensor value) -> Tensor
at::Tensor masked_fill_Tensor::call(const at::Tensor & self, const at::Tensor & mask, const at::Tensor & value) {
    static auto op = create_masked_fill_Tensor_typed_handle();
    return op.call(self, mask, value);
}

// aten::masked_fill.Tensor(Tensor self, Tensor mask, Tensor value) -> Tensor
at::Tensor masked_fill_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mask, const at::Tensor & value) {
    static auto op = create_masked_fill_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, mask, value);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(view_dtype, name, "aten::view")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(view_dtype, overload_name, "dtype")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(view_dtype, schema_str, "view.dtype(Tensor(a) self, ScalarType dtype) -> Tensor(a)")

// aten::view.dtype(Tensor(a) self, ScalarType dtype) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<view_dtype::schema> create_view_dtype_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(view_dtype::name, view_dtype::overload_name)
      .typed<view_dtype::schema>();
}

// aten::view.dtype(Tensor(a) self, ScalarType dtype) -> Tensor(a)
at::Tensor view_dtype::call(const at::Tensor & self, at::ScalarType dtype) {
    static auto op = create_view_dtype_typed_handle();
    return op.call(self, dtype);
}

// aten::view.dtype(Tensor(a) self, ScalarType dtype) -> Tensor(a)
at::Tensor view_dtype::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::ScalarType dtype) {
    static auto op = create_view_dtype_typed_handle();
    return op.redispatch(dispatchKeySet, self, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_fill_int_Tensor, name, "aten::index_fill")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_fill_int_Tensor, overload_name, "int_Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_fill_int_Tensor, schema_str, "index_fill.int_Tensor(Tensor self, int dim, Tensor index, Tensor value) -> Tensor")

// aten::index_fill.int_Tensor(Tensor self, int dim, Tensor index, Tensor value) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<index_fill_int_Tensor::schema> create_index_fill_int_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(index_fill_int_Tensor::name, index_fill_int_Tensor::overload_name)
      .typed<index_fill_int_Tensor::schema>();
}

// aten::index_fill.int_Tensor(Tensor self, int dim, Tensor index, Tensor value) -> Tensor
at::Tensor index_fill_int_Tensor::call(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & value) {
    static auto op = create_index_fill_int_Tensor_typed_handle();
    return op.call(self, dim, index, value);
}

// aten::index_fill.int_Tensor(Tensor self, int dim, Tensor index, Tensor value) -> Tensor
at::Tensor index_fill_int_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & value) {
    static auto op = create_index_fill_int_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, value);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_fill__Dimname_Tensor, name, "aten::index_fill_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_fill__Dimname_Tensor, overload_name, "Dimname_Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_fill__Dimname_Tensor, schema_str, "index_fill_.Dimname_Tensor(Tensor(a!) self, Dimname dim, Tensor index, Tensor value) -> Tensor(a!)")

// aten::index_fill_.Dimname_Tensor(Tensor(a!) self, Dimname dim, Tensor index, Tensor value) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<index_fill__Dimname_Tensor::schema> create_index_fill__Dimname_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(index_fill__Dimname_Tensor::name, index_fill__Dimname_Tensor::overload_name)
      .typed<index_fill__Dimname_Tensor::schema>();
}

// aten::index_fill_.Dimname_Tensor(Tensor(a!) self, Dimname dim, Tensor index, Tensor value) -> Tensor(a!)
at::Tensor & index_fill__Dimname_Tensor::call(at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & value) {
    static auto op = create_index_fill__Dimname_Tensor_typed_handle();
    return op.call(self, dim, index, value);
}

// aten::index_fill_.Dimname_Tensor(Tensor(a!) self, Dimname dim, Tensor index, Tensor value) -> Tensor(a!)
at::Tensor & index_fill__Dimname_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & value) {
    static auto op = create_index_fill__Dimname_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, value);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_add, name, "aten::scatter_add")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_add, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_add, schema_str, "scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor")

// aten::scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<scatter_add::schema> create_scatter_add_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(scatter_add::name, scatter_add::overload_name)
      .typed<scatter_add::schema>();
}

// aten::scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
at::Tensor scatter_add::call(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src) {
    static auto op = create_scatter_add_typed_handle();
    return op.call(self, dim, index, src);
}

// aten::scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
at::Tensor scatter_add::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src) {
    static auto op = create_scatter_add_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, src);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_and_Tensor_out, name, "aten::bitwise_and")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_and_Tensor_out, overload_name, "Tensor_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_and_Tensor_out, schema_str, "bitwise_and.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::bitwise_and.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<bitwise_and_Tensor_out::schema> create_bitwise_and_Tensor_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bitwise_and_Tensor_out::name, bitwise_and_Tensor_out::overload_name)
      .typed<bitwise_and_Tensor_out::schema>();
}

// aten::bitwise_and.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & bitwise_and_Tensor_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_bitwise_and_Tensor_out_typed_handle();
    return op.call(self, other, out);
}

// aten::bitwise_and.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & bitwise_and_Tensor_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_bitwise_and_Tensor_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_and__Scalar, name, "aten::bitwise_and_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_and__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_and__Scalar, schema_str, "bitwise_and_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)")

// aten::bitwise_and_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<bitwise_and__Scalar::schema> create_bitwise_and__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bitwise_and__Scalar::name, bitwise_and__Scalar::overload_name)
      .typed<bitwise_and__Scalar::schema>();
}

// aten::bitwise_and_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & bitwise_and__Scalar::call(at::Tensor & self, const at::Scalar & other) {
    static auto op = create_bitwise_and__Scalar_typed_handle();
    return op.call(self, other);
}

// aten::bitwise_and_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & bitwise_and__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
    static auto op = create_bitwise_and__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__ior___Scalar, name, "aten::__ior__")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__ior___Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__ior___Scalar, schema_str, "__ior__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)")

// aten::__ior__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<__ior___Scalar::schema> create___ior___Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(__ior___Scalar::name, __ior___Scalar::overload_name)
      .typed<__ior___Scalar::schema>();
}

// aten::__ior__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & __ior___Scalar::call(at::Tensor & self, const at::Scalar & other) {
    static auto op = create___ior___Scalar_typed_handle();
    return op.call(self, other);
}

// aten::__ior__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & __ior___Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
    static auto op = create___ior___Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_xor_Scalar_out, name, "aten::bitwise_xor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_xor_Scalar_out, overload_name, "Scalar_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_xor_Scalar_out, schema_str, "bitwise_xor.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::bitwise_xor.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<bitwise_xor_Scalar_out::schema> create_bitwise_xor_Scalar_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bitwise_xor_Scalar_out::name, bitwise_xor_Scalar_out::overload_name)
      .typed<bitwise_xor_Scalar_out::schema>();
}

// aten::bitwise_xor.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & bitwise_xor_Scalar_out::call(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_bitwise_xor_Scalar_out_typed_handle();
    return op.call(self, other, out);
}

// aten::bitwise_xor.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & bitwise_xor_Scalar_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_bitwise_xor_Scalar_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_left_shift__Tensor_Scalar, name, "aten::bitwise_left_shift_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_left_shift__Tensor_Scalar, overload_name, "Tensor_Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_left_shift__Tensor_Scalar, schema_str, "bitwise_left_shift_.Tensor_Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)")

// aten::bitwise_left_shift_.Tensor_Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<bitwise_left_shift__Tensor_Scalar::schema> create_bitwise_left_shift__Tensor_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bitwise_left_shift__Tensor_Scalar::name, bitwise_left_shift__Tensor_Scalar::overload_name)
      .typed<bitwise_left_shift__Tensor_Scalar::schema>();
}

// aten::bitwise_left_shift_.Tensor_Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & bitwise_left_shift__Tensor_Scalar::call(at::Tensor & self, const at::Scalar & other) {
    static auto op = create_bitwise_left_shift__Tensor_Scalar_typed_handle();
    return op.call(self, other);
}

// aten::bitwise_left_shift_.Tensor_Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & bitwise_left_shift__Tensor_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
    static auto op = create_bitwise_left_shift__Tensor_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tril_, name, "aten::tril_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tril_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tril_, schema_str, "tril_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)")

// aten::tril_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<tril_::schema> create_tril__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(tril_::name, tril_::overload_name)
      .typed<tril_::schema>();
}

// aten::tril_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)
at::Tensor & tril_::call(at::Tensor & self, int64_t diagonal) {
    static auto op = create_tril__typed_handle();
    return op.call(self, diagonal);
}

// aten::tril_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)
at::Tensor & tril_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t diagonal) {
    static auto op = create_tril__typed_handle();
    return op.redispatch(dispatchKeySet, self, diagonal);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lerp__Tensor, name, "aten::lerp_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lerp__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lerp__Tensor, schema_str, "lerp_.Tensor(Tensor(a!) self, Tensor end, Tensor weight) -> Tensor(a!)")

// aten::lerp_.Tensor(Tensor(a!) self, Tensor end, Tensor weight) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<lerp__Tensor::schema> create_lerp__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(lerp__Tensor::name, lerp__Tensor::overload_name)
      .typed<lerp__Tensor::schema>();
}

// aten::lerp_.Tensor(Tensor(a!) self, Tensor end, Tensor weight) -> Tensor(a!)
at::Tensor & lerp__Tensor::call(at::Tensor & self, const at::Tensor & end, const at::Tensor & weight) {
    static auto op = create_lerp__Tensor_typed_handle();
    return op.call(self, end, weight);
}

// aten::lerp_.Tensor(Tensor(a!) self, Tensor end, Tensor weight) -> Tensor(a!)
at::Tensor & lerp__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & end, const at::Tensor & weight) {
    static auto op = create_lerp__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, end, weight);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addbmm_out, name, "aten::addbmm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addbmm_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addbmm_out, schema_str, "addbmm.out(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)")

// aten::addbmm.out(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<addbmm_out::schema> create_addbmm_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(addbmm_out::name, addbmm_out::overload_name)
      .typed<addbmm_out::schema>();
}

// aten::addbmm.out(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
at::Tensor & addbmm_out::call(const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
    static auto op = create_addbmm_out_typed_handle();
    return op.call(self, batch1, batch2, beta, alpha, out);
}

// aten::addbmm.out(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
at::Tensor & addbmm_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
    static auto op = create_addbmm_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, batch1, batch2, beta, alpha, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(uniform_, name, "aten::uniform_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(uniform_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(uniform_, schema_str, "uniform_(Tensor(a!) self, float from=0, float to=1, *, Generator? generator=None) -> Tensor(a!)")

// aten::uniform_(Tensor(a!) self, float from=0, float to=1, *, Generator? generator=None) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<uniform_::schema> create_uniform__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(uniform_::name, uniform_::overload_name)
      .typed<uniform_::schema>();
}

// aten::uniform_(Tensor(a!) self, float from=0, float to=1, *, Generator? generator=None) -> Tensor(a!)
at::Tensor & uniform_::call(at::Tensor & self, double from, double to, c10::optional<at::Generator> generator) {
    static auto op = create_uniform__typed_handle();
    return op.call(self, from, to, generator);
}

// aten::uniform_(Tensor(a!) self, float from=0, float to=1, *, Generator? generator=None) -> Tensor(a!)
at::Tensor & uniform_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, double from, double to, c10::optional<at::Generator> generator) {
    static auto op = create_uniform__typed_handle();
    return op.redispatch(dispatchKeySet, self, from, to, generator);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cross, name, "aten::cross")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cross, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cross, schema_str, "cross(Tensor self, Tensor other, int? dim=None) -> Tensor")

// aten::cross(Tensor self, Tensor other, int? dim=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<cross::schema> create_cross_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cross::name, cross::overload_name)
      .typed<cross::schema>();
}

// aten::cross(Tensor self, Tensor other, int? dim=None) -> Tensor
at::Tensor cross::call(const at::Tensor & self, const at::Tensor & other, c10::optional<int64_t> dim) {
    static auto op = create_cross_typed_handle();
    return op.call(self, other, dim);
}

// aten::cross(Tensor self, Tensor other, int? dim=None) -> Tensor
at::Tensor cross::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, c10::optional<int64_t> dim) {
    static auto op = create_cross_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ne__Tensor, name, "aten::ne_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ne__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ne__Tensor, schema_str, "ne_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::ne_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<ne__Tensor::schema> create_ne__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(ne__Tensor::name, ne__Tensor::overload_name)
      .typed<ne__Tensor::schema>();
}

// aten::ne_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & ne__Tensor::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create_ne__Tensor_typed_handle();
    return op.call(self, other);
}

// aten::ne_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & ne__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create_ne__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(eq_Scalar, name, "aten::eq")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(eq_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(eq_Scalar, schema_str, "eq.Scalar(Tensor self, Scalar other) -> Tensor")

// aten::eq.Scalar(Tensor self, Scalar other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<eq_Scalar::schema> create_eq_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(eq_Scalar::name, eq_Scalar::overload_name)
      .typed<eq_Scalar::schema>();
}

// aten::eq.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor eq_Scalar::call(const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_eq_Scalar_typed_handle();
    return op.call(self, other);
}

// aten::eq.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor eq_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_eq_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(eq_Tensor, name, "aten::eq")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(eq_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(eq_Tensor, schema_str, "eq.Tensor(Tensor self, Tensor other) -> Tensor")

// aten::eq.Tensor(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<eq_Tensor::schema> create_eq_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(eq_Tensor::name, eq_Tensor::overload_name)
      .typed<eq_Tensor::schema>();
}

// aten::eq.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor eq_Tensor::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_eq_Tensor_typed_handle();
    return op.call(self, other);
}

// aten::eq.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor eq_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_eq_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ge_Scalar, name, "aten::ge")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ge_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ge_Scalar, schema_str, "ge.Scalar(Tensor self, Scalar other) -> Tensor")

// aten::ge.Scalar(Tensor self, Scalar other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<ge_Scalar::schema> create_ge_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(ge_Scalar::name, ge_Scalar::overload_name)
      .typed<ge_Scalar::schema>();
}

// aten::ge.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor ge_Scalar::call(const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_ge_Scalar_typed_handle();
    return op.call(self, other);
}

// aten::ge.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor ge_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_ge_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(greater_equal_Tensor_out, name, "aten::greater_equal")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(greater_equal_Tensor_out, overload_name, "Tensor_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(greater_equal_Tensor_out, schema_str, "greater_equal.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::greater_equal.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<greater_equal_Tensor_out::schema> create_greater_equal_Tensor_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(greater_equal_Tensor_out::name, greater_equal_Tensor_out::overload_name)
      .typed<greater_equal_Tensor_out::schema>();
}

// aten::greater_equal.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & greater_equal_Tensor_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_greater_equal_Tensor_out_typed_handle();
    return op.call(self, other, out);
}

// aten::greater_equal.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & greater_equal_Tensor_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_greater_equal_Tensor_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(le_Scalar_out, name, "aten::le")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(le_Scalar_out, overload_name, "Scalar_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(le_Scalar_out, schema_str, "le.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::le.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<le_Scalar_out::schema> create_le_Scalar_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(le_Scalar_out::name, le_Scalar_out::overload_name)
      .typed<le_Scalar_out::schema>();
}

// aten::le.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & le_Scalar_out::call(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_le_Scalar_out_typed_handle();
    return op.call(self, other, out);
}

// aten::le.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & le_Scalar_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_le_Scalar_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(greater_Scalar_out, name, "aten::greater")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(greater_Scalar_out, overload_name, "Scalar_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(greater_Scalar_out, schema_str, "greater.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::greater.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<greater_Scalar_out::schema> create_greater_Scalar_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(greater_Scalar_out::name, greater_Scalar_out::overload_name)
      .typed<greater_Scalar_out::schema>();
}

// aten::greater.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & greater_Scalar_out::call(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_greater_Scalar_out_typed_handle();
    return op.call(self, other, out);
}

// aten::greater.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & greater_Scalar_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_greater_Scalar_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(greater_Tensor, name, "aten::greater")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(greater_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(greater_Tensor, schema_str, "greater.Tensor(Tensor self, Tensor other) -> Tensor")

// aten::greater.Tensor(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<greater_Tensor::schema> create_greater_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(greater_Tensor::name, greater_Tensor::overload_name)
      .typed<greater_Tensor::schema>();
}

// aten::greater.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor greater_Tensor::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_greater_Tensor_typed_handle();
    return op.call(self, other);
}

// aten::greater.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor greater_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_greater_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lt_Scalar, name, "aten::lt")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lt_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lt_Scalar, schema_str, "lt.Scalar(Tensor self, Scalar other) -> Tensor")

// aten::lt.Scalar(Tensor self, Scalar other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<lt_Scalar::schema> create_lt_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(lt_Scalar::name, lt_Scalar::overload_name)
      .typed<lt_Scalar::schema>();
}

// aten::lt.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor lt_Scalar::call(const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_lt_Scalar_typed_handle();
    return op.call(self, other);
}

// aten::lt.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor lt_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_lt_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lt_Tensor_out, name, "aten::lt")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lt_Tensor_out, overload_name, "Tensor_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lt_Tensor_out, schema_str, "lt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::lt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<lt_Tensor_out::schema> create_lt_Tensor_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(lt_Tensor_out::name, lt_Tensor_out::overload_name)
      .typed<lt_Tensor_out::schema>();
}

// aten::lt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & lt_Tensor_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_lt_Tensor_out_typed_handle();
    return op.call(self, other, out);
}

// aten::lt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & lt_Tensor_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_lt_Tensor_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(less_Tensor, name, "aten::less")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(less_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(less_Tensor, schema_str, "less.Tensor(Tensor self, Tensor other) -> Tensor")

// aten::less.Tensor(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<less_Tensor::schema> create_less_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(less_Tensor::name, less_Tensor::overload_name)
      .typed<less_Tensor::schema>();
}

// aten::less.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor less_Tensor::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_less_Tensor_typed_handle();
    return op.call(self, other);
}

// aten::less.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor less_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_less_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_select_out, name, "aten::index_select")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_select_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_select_out, schema_str, "index_select.out(Tensor self, int dim, Tensor index, *, Tensor(a!) out) -> Tensor(a!)")

// aten::index_select.out(Tensor self, int dim, Tensor index, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<index_select_out::schema> create_index_select_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(index_select_out::name, index_select_out::overload_name)
      .typed<index_select_out::schema>();
}

// aten::index_select.out(Tensor self, int dim, Tensor index, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & index_select_out::call(const at::Tensor & self, int64_t dim, const at::Tensor & index, at::Tensor & out) {
    static auto op = create_index_select_out_typed_handle();
    return op.call(self, dim, index, out);
}

// aten::index_select.out(Tensor self, int dim, Tensor index, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & index_select_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, const at::Tensor & index, at::Tensor & out) {
    static auto op = create_index_select_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_select_dimname, name, "aten::index_select")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_select_dimname, overload_name, "dimname")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_select_dimname, schema_str, "index_select.dimname(Tensor self, Dimname dim, Tensor index) -> Tensor")

// aten::index_select.dimname(Tensor self, Dimname dim, Tensor index) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<index_select_dimname::schema> create_index_select_dimname_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(index_select_dimname::name, index_select_dimname::overload_name)
      .typed<index_select_dimname::schema>();
}

// aten::index_select.dimname(Tensor self, Dimname dim, Tensor index) -> Tensor
at::Tensor index_select_dimname::call(const at::Tensor & self, at::Dimname dim, const at::Tensor & index) {
    static auto op = create_index_select_dimname_typed_handle();
    return op.call(self, dim, index);
}

// aten::index_select.dimname(Tensor self, Dimname dim, Tensor index) -> Tensor
at::Tensor index_select_dimname::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, const at::Tensor & index) {
    static auto op = create_index_select_dimname_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(masked_select_out, name, "aten::masked_select")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(masked_select_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(masked_select_out, schema_str, "masked_select.out(Tensor self, Tensor mask, *, Tensor(a!) out) -> Tensor(a!)")

// aten::masked_select.out(Tensor self, Tensor mask, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<masked_select_out::schema> create_masked_select_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(masked_select_out::name, masked_select_out::overload_name)
      .typed<masked_select_out::schema>();
}

// aten::masked_select.out(Tensor self, Tensor mask, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & masked_select_out::call(const at::Tensor & self, const at::Tensor & mask, at::Tensor & out) {
    static auto op = create_masked_select_out_typed_handle();
    return op.call(self, mask, out);
}

// aten::masked_select.out(Tensor self, Tensor mask, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & masked_select_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mask, at::Tensor & out) {
    static auto op = create_masked_select_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, mask, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_gather_sparse_backward, name, "aten::_gather_sparse_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_gather_sparse_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_gather_sparse_backward, schema_str, "_gather_sparse_backward(Tensor self, int dim, Tensor index, Tensor grad) -> Tensor")

// aten::_gather_sparse_backward(Tensor self, int dim, Tensor index, Tensor grad) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_gather_sparse_backward::schema> create__gather_sparse_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_gather_sparse_backward::name, _gather_sparse_backward::overload_name)
      .typed<_gather_sparse_backward::schema>();
}

// aten::_gather_sparse_backward(Tensor self, int dim, Tensor index, Tensor grad) -> Tensor
at::Tensor _gather_sparse_backward::call(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & grad) {
    static auto op = create__gather_sparse_backward_typed_handle();
    return op.call(self, dim, index, grad);
}

// aten::_gather_sparse_backward(Tensor self, int dim, Tensor index, Tensor grad) -> Tensor
at::Tensor _gather_sparse_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & grad) {
    static auto op = create__gather_sparse_backward_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, grad);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addcdiv_out, name, "aten::addcdiv")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addcdiv_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addcdiv_out, schema_str, "addcdiv.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)")

// aten::addcdiv.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<addcdiv_out::schema> create_addcdiv_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(addcdiv_out::name, addcdiv_out::overload_name)
      .typed<addcdiv_out::schema>();
}

// aten::addcdiv.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)
at::Tensor & addcdiv_out::call(const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value, at::Tensor & out) {
    static auto op = create_addcdiv_out_typed_handle();
    return op.call(self, tensor1, tensor2, value, out);
}

// aten::addcdiv.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)
at::Tensor & addcdiv_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value, at::Tensor & out) {
    static auto op = create_addcdiv_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, tensor1, tensor2, value, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lstsq_X, name, "aten::lstsq")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lstsq_X, overload_name, "X")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lstsq_X, schema_str, "lstsq.X(Tensor self, Tensor A, *, Tensor(a!) X, Tensor(b!) qr) -> (Tensor(a!) solution, Tensor(b!) QR)")

// aten::lstsq.X(Tensor self, Tensor A, *, Tensor(a!) X, Tensor(b!) qr) -> (Tensor(a!) solution, Tensor(b!) QR)
static C10_NOINLINE c10::TypedOperatorHandle<lstsq_X::schema> create_lstsq_X_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(lstsq_X::name, lstsq_X::overload_name)
      .typed<lstsq_X::schema>();
}

// aten::lstsq.X(Tensor self, Tensor A, *, Tensor(a!) X, Tensor(b!) qr) -> (Tensor(a!) solution, Tensor(b!) QR)
::std::tuple<at::Tensor &,at::Tensor &> lstsq_X::call(const at::Tensor & self, const at::Tensor & A, at::Tensor & X, at::Tensor & qr) {
    static auto op = create_lstsq_X_typed_handle();
    return op.call(self, A, X, qr);
}

// aten::lstsq.X(Tensor self, Tensor A, *, Tensor(a!) X, Tensor(b!) qr) -> (Tensor(a!) solution, Tensor(b!) QR)
::std::tuple<at::Tensor &,at::Tensor &> lstsq_X::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & A, at::Tensor & X, at::Tensor & qr) {
    static auto op = create_lstsq_X_typed_handle();
    return op.redispatch(dispatchKeySet, self, A, X, qr);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(eig_e, name, "aten::eig")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(eig_e, overload_name, "e")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(eig_e, schema_str, "eig.e(Tensor self, bool eigenvectors=False, *, Tensor(a!) e, Tensor(b!) v) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)")

// aten::eig.e(Tensor self, bool eigenvectors=False, *, Tensor(a!) e, Tensor(b!) v) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)
static C10_NOINLINE c10::TypedOperatorHandle<eig_e::schema> create_eig_e_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(eig_e::name, eig_e::overload_name)
      .typed<eig_e::schema>();
}

// aten::eig.e(Tensor self, bool eigenvectors=False, *, Tensor(a!) e, Tensor(b!) v) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)
::std::tuple<at::Tensor &,at::Tensor &> eig_e::call(const at::Tensor & self, bool eigenvectors, at::Tensor & e, at::Tensor & v) {
    static auto op = create_eig_e_typed_handle();
    return op.call(self, eigenvectors, e, v);
}

// aten::eig.e(Tensor self, bool eigenvectors=False, *, Tensor(a!) e, Tensor(b!) v) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)
::std::tuple<at::Tensor &,at::Tensor &> eig_e::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool eigenvectors, at::Tensor & e, at::Tensor & v) {
    static auto op = create_eig_e_typed_handle();
    return op.redispatch(dispatchKeySet, self, eigenvectors, e, v);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(swapaxes, name, "aten::swapaxes")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(swapaxes, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(swapaxes, schema_str, "swapaxes(Tensor(a) self, int axis0, int axis1) -> Tensor(a)")

// aten::swapaxes(Tensor(a) self, int axis0, int axis1) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<swapaxes::schema> create_swapaxes_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(swapaxes::name, swapaxes::overload_name)
      .typed<swapaxes::schema>();
}

// aten::swapaxes(Tensor(a) self, int axis0, int axis1) -> Tensor(a)
at::Tensor swapaxes::call(const at::Tensor & self, int64_t axis0, int64_t axis1) {
    static auto op = create_swapaxes_typed_handle();
    return op.call(self, axis0, axis1);
}

// aten::swapaxes(Tensor(a) self, int axis0, int axis1) -> Tensor(a)
at::Tensor swapaxes::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t axis0, int64_t axis1) {
    static auto op = create_swapaxes_typed_handle();
    return op.redispatch(dispatchKeySet, self, axis0, axis1);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cholesky_solve, name, "aten::cholesky_solve")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cholesky_solve, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cholesky_solve, schema_str, "cholesky_solve(Tensor self, Tensor input2, bool upper=False) -> Tensor")

// aten::cholesky_solve(Tensor self, Tensor input2, bool upper=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<cholesky_solve::schema> create_cholesky_solve_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cholesky_solve::name, cholesky_solve::overload_name)
      .typed<cholesky_solve::schema>();
}

// aten::cholesky_solve(Tensor self, Tensor input2, bool upper=False) -> Tensor
at::Tensor cholesky_solve::call(const at::Tensor & self, const at::Tensor & input2, bool upper) {
    static auto op = create_cholesky_solve_typed_handle();
    return op.call(self, input2, upper);
}

// aten::cholesky_solve(Tensor self, Tensor input2, bool upper=False) -> Tensor
at::Tensor cholesky_solve::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & input2, bool upper) {
    static auto op = create_cholesky_solve_typed_handle();
    return op.redispatch(dispatchKeySet, self, input2, upper);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_solve_helper, name, "aten::_solve_helper")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_solve_helper, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_solve_helper, schema_str, "_solve_helper(Tensor self, Tensor A) -> (Tensor, Tensor)")

// aten::_solve_helper(Tensor self, Tensor A) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<_solve_helper::schema> create__solve_helper_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_solve_helper::name, _solve_helper::overload_name)
      .typed<_solve_helper::schema>();
}

// aten::_solve_helper(Tensor self, Tensor A) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> _solve_helper::call(const at::Tensor & self, const at::Tensor & A) {
    static auto op = create__solve_helper_typed_handle();
    return op.call(self, A);
}

// aten::_solve_helper(Tensor self, Tensor A) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> _solve_helper::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & A) {
    static auto op = create__solve_helper_typed_handle();
    return op.redispatch(dispatchKeySet, self, A);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cholesky_inverse_out, name, "aten::cholesky_inverse")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cholesky_inverse_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cholesky_inverse_out, schema_str, "cholesky_inverse.out(Tensor self, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)")

// aten::cholesky_inverse.out(Tensor self, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<cholesky_inverse_out::schema> create_cholesky_inverse_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cholesky_inverse_out::name, cholesky_inverse_out::overload_name)
      .typed<cholesky_inverse_out::schema>();
}

// aten::cholesky_inverse.out(Tensor self, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & cholesky_inverse_out::call(const at::Tensor & self, bool upper, at::Tensor & out) {
    static auto op = create_cholesky_inverse_out_typed_handle();
    return op.call(self, upper, out);
}

// aten::cholesky_inverse.out(Tensor self, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & cholesky_inverse_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool upper, at::Tensor & out) {
    static auto op = create_cholesky_inverse_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, upper, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(qr_Q, name, "aten::qr")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(qr_Q, overload_name, "Q")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(qr_Q, schema_str, "qr.Q(Tensor self, bool some=True, *, Tensor(a!) Q, Tensor(b!) R) -> (Tensor(a!) Q, Tensor(b!) R)")

// aten::qr.Q(Tensor self, bool some=True, *, Tensor(a!) Q, Tensor(b!) R) -> (Tensor(a!) Q, Tensor(b!) R)
static C10_NOINLINE c10::TypedOperatorHandle<qr_Q::schema> create_qr_Q_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(qr_Q::name, qr_Q::overload_name)
      .typed<qr_Q::schema>();
}

// aten::qr.Q(Tensor self, bool some=True, *, Tensor(a!) Q, Tensor(b!) R) -> (Tensor(a!) Q, Tensor(b!) R)
::std::tuple<at::Tensor &,at::Tensor &> qr_Q::call(const at::Tensor & self, bool some, at::Tensor & Q, at::Tensor & R) {
    static auto op = create_qr_Q_typed_handle();
    return op.call(self, some, Q, R);
}

// aten::qr.Q(Tensor self, bool some=True, *, Tensor(a!) Q, Tensor(b!) R) -> (Tensor(a!) Q, Tensor(b!) R)
::std::tuple<at::Tensor &,at::Tensor &> qr_Q::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool some, at::Tensor & Q, at::Tensor & R) {
    static auto op = create_qr_Q_typed_handle();
    return op.redispatch(dispatchKeySet, self, some, Q, R);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(qr, name, "aten::qr")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(qr, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(qr, schema_str, "qr(Tensor self, bool some=True) -> (Tensor Q, Tensor R)")

// aten::qr(Tensor self, bool some=True) -> (Tensor Q, Tensor R)
static C10_NOINLINE c10::TypedOperatorHandle<qr::schema> create_qr_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(qr::name, qr::overload_name)
      .typed<qr::schema>();
}

// aten::qr(Tensor self, bool some=True) -> (Tensor Q, Tensor R)
::std::tuple<at::Tensor,at::Tensor> qr::call(const at::Tensor & self, bool some) {
    static auto op = create_qr_typed_handle();
    return op.call(self, some);
}

// aten::qr(Tensor self, bool some=True) -> (Tensor Q, Tensor R)
::std::tuple<at::Tensor,at::Tensor> qr::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool some) {
    static auto op = create_qr_typed_handle();
    return op.redispatch(dispatchKeySet, self, some);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(digamma, name, "aten::digamma")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(digamma, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(digamma, schema_str, "digamma(Tensor self) -> Tensor")

// aten::digamma(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<digamma::schema> create_digamma_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(digamma::name, digamma::overload_name)
      .typed<digamma::schema>();
}

// aten::digamma(Tensor self) -> Tensor
at::Tensor digamma::call(const at::Tensor & self) {
    static auto op = create_digamma_typed_handle();
    return op.call(self);
}

// aten::digamma(Tensor self) -> Tensor
at::Tensor digamma::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_digamma_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(polygamma, name, "aten::polygamma")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(polygamma, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(polygamma, schema_str, "polygamma(int n, Tensor self) -> Tensor")

// aten::polygamma(int n, Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<polygamma::schema> create_polygamma_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(polygamma::name, polygamma::overload_name)
      .typed<polygamma::schema>();
}

// aten::polygamma(int n, Tensor self) -> Tensor
at::Tensor polygamma::call(int64_t n, const at::Tensor & self) {
    static auto op = create_polygamma_typed_handle();
    return op.call(n, self);
}

// aten::polygamma(int n, Tensor self) -> Tensor
at::Tensor polygamma::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t n, const at::Tensor & self) {
    static auto op = create_polygamma_typed_handle();
    return op.redispatch(dispatchKeySet, n, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(polygamma_, name, "aten::polygamma_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(polygamma_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(polygamma_, schema_str, "polygamma_(Tensor(a!) self, int n) -> Tensor(a!)")

// aten::polygamma_(Tensor(a!) self, int n) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<polygamma_::schema> create_polygamma__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(polygamma_::name, polygamma_::overload_name)
      .typed<polygamma_::schema>();
}

// aten::polygamma_(Tensor(a!) self, int n) -> Tensor(a!)
at::Tensor & polygamma_::call(at::Tensor & self, int64_t n) {
    static auto op = create_polygamma__typed_handle();
    return op.call(self, n);
}

// aten::polygamma_(Tensor(a!) self, int n) -> Tensor(a!)
at::Tensor & polygamma_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t n) {
    static auto op = create_polygamma__typed_handle();
    return op.redispatch(dispatchKeySet, self, n);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(i0_out, name, "aten::i0")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(i0_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(i0_out, schema_str, "i0.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::i0.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<i0_out::schema> create_i0_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(i0_out::name, i0_out::overload_name)
      .typed<i0_out::schema>();
}

// aten::i0.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & i0_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_i0_out_typed_handle();
    return op.call(self, out);
}

// aten::i0.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & i0_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_i0_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sign_, name, "aten::sign_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sign_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sign_, schema_str, "sign_(Tensor(a!) self) -> Tensor(a!)")

// aten::sign_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<sign_::schema> create_sign__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sign_::name, sign_::overload_name)
      .typed<sign_::schema>();
}

// aten::sign_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & sign_::call(at::Tensor & self) {
    static auto op = create_sign__typed_handle();
    return op.call(self);
}

// aten::sign_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & sign_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_sign__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sign_out, name, "aten::sign")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sign_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sign_out, schema_str, "sign.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::sign.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<sign_out::schema> create_sign_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sign_out::name, sign_out::overload_name)
      .typed<sign_out::schema>();
}

// aten::sign.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & sign_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_sign_out_typed_handle();
    return op.call(self, out);
}

// aten::sign.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & sign_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_sign_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(histc, name, "aten::histc")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(histc, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(histc, schema_str, "histc(Tensor self, int bins=100, Scalar min=0, Scalar max=0) -> Tensor")

// aten::histc(Tensor self, int bins=100, Scalar min=0, Scalar max=0) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<histc::schema> create_histc_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(histc::name, histc::overload_name)
      .typed<histc::schema>();
}

// aten::histc(Tensor self, int bins=100, Scalar min=0, Scalar max=0) -> Tensor
at::Tensor histc::call(const at::Tensor & self, int64_t bins, const at::Scalar & min, const at::Scalar & max) {
    static auto op = create_histc_typed_handle();
    return op.call(self, bins, min, max);
}

// aten::histc(Tensor self, int bins=100, Scalar min=0, Scalar max=0) -> Tensor
at::Tensor histc::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t bins, const at::Scalar & min, const at::Scalar & max) {
    static auto op = create_histc_typed_handle();
    return op.redispatch(dispatchKeySet, self, bins, min, max);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(histogram_bin_ct_out, name, "aten::histogram")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(histogram_bin_ct_out, overload_name, "bin_ct_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(histogram_bin_ct_out, schema_str, "histogram.bin_ct_out(Tensor self, int bins=100, *, float[]? range=None, Tensor? weight=None, bool density=False, Tensor(a!) hist, Tensor(b!) bin_edges) -> (Tensor(a!) hist, Tensor(b!) bin_edges)")

// aten::histogram.bin_ct_out(Tensor self, int bins=100, *, float[]? range=None, Tensor? weight=None, bool density=False, Tensor(a!) hist, Tensor(b!) bin_edges) -> (Tensor(a!) hist, Tensor(b!) bin_edges)
static C10_NOINLINE c10::TypedOperatorHandle<histogram_bin_ct_out::schema> create_histogram_bin_ct_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(histogram_bin_ct_out::name, histogram_bin_ct_out::overload_name)
      .typed<histogram_bin_ct_out::schema>();
}

// aten::histogram.bin_ct_out(Tensor self, int bins=100, *, float[]? range=None, Tensor? weight=None, bool density=False, Tensor(a!) hist, Tensor(b!) bin_edges) -> (Tensor(a!) hist, Tensor(b!) bin_edges)
::std::tuple<at::Tensor &,at::Tensor &> histogram_bin_ct_out::call(const at::Tensor & self, int64_t bins, c10::optional<at::ArrayRef<double>> range, const c10::optional<at::Tensor> & weight, bool density, at::Tensor & hist, at::Tensor & bin_edges) {
    static auto op = create_histogram_bin_ct_out_typed_handle();
    return op.call(self, bins, range, weight, density, hist, bin_edges);
}

// aten::histogram.bin_ct_out(Tensor self, int bins=100, *, float[]? range=None, Tensor? weight=None, bool density=False, Tensor(a!) hist, Tensor(b!) bin_edges) -> (Tensor(a!) hist, Tensor(b!) bin_edges)
::std::tuple<at::Tensor &,at::Tensor &> histogram_bin_ct_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t bins, c10::optional<at::ArrayRef<double>> range, const c10::optional<at::Tensor> & weight, bool density, at::Tensor & hist, at::Tensor & bin_edges) {
    static auto op = create_histogram_bin_ct_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, bins, range, weight, density, hist, bin_edges);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fmod_Scalar, name, "aten::fmod")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fmod_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fmod_Scalar, schema_str, "fmod.Scalar(Tensor self, Scalar other) -> Tensor")

// aten::fmod.Scalar(Tensor self, Scalar other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fmod_Scalar::schema> create_fmod_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fmod_Scalar::name, fmod_Scalar::overload_name)
      .typed<fmod_Scalar::schema>();
}

// aten::fmod.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor fmod_Scalar::call(const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_fmod_Scalar_typed_handle();
    return op.call(self, other);
}

// aten::fmod.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor fmod_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_fmod_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fmod__Scalar, name, "aten::fmod_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fmod__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fmod__Scalar, schema_str, "fmod_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)")

// aten::fmod_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<fmod__Scalar::schema> create_fmod__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fmod__Scalar::name, fmod__Scalar::overload_name)
      .typed<fmod__Scalar::schema>();
}

// aten::fmod_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & fmod__Scalar::call(at::Tensor & self, const at::Scalar & other) {
    static auto op = create_fmod__Scalar_typed_handle();
    return op.call(self, other);
}

// aten::fmod_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & fmod__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
    static auto op = create_fmod__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fmod_Tensor_out, name, "aten::fmod")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fmod_Tensor_out, overload_name, "Tensor_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fmod_Tensor_out, schema_str, "fmod.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::fmod.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<fmod_Tensor_out::schema> create_fmod_Tensor_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fmod_Tensor_out::name, fmod_Tensor_out::overload_name)
      .typed<fmod_Tensor_out::schema>();
}

// aten::fmod.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fmod_Tensor_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_fmod_Tensor_out_typed_handle();
    return op.call(self, other, out);
}

// aten::fmod.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fmod_Tensor_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_fmod_Tensor_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fmod_Tensor, name, "aten::fmod")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fmod_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fmod_Tensor, schema_str, "fmod.Tensor(Tensor self, Tensor other) -> Tensor")

// aten::fmod.Tensor(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fmod_Tensor::schema> create_fmod_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fmod_Tensor::name, fmod_Tensor::overload_name)
      .typed<fmod_Tensor::schema>();
}

// aten::fmod.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor fmod_Tensor::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_fmod_Tensor_typed_handle();
    return op.call(self, other);
}

// aten::fmod.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor fmod_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_fmod_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fmod__Tensor, name, "aten::fmod_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fmod__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fmod__Tensor, schema_str, "fmod_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::fmod_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<fmod__Tensor::schema> create_fmod__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fmod__Tensor::name, fmod__Tensor::overload_name)
      .typed<fmod__Tensor::schema>();
}

// aten::fmod_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & fmod__Tensor::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create_fmod__Tensor_typed_handle();
    return op.call(self, other);
}

// aten::fmod_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & fmod__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create_fmod__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(igamma_, name, "aten::igamma_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(igamma_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(igamma_, schema_str, "igamma_(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::igamma_(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<igamma_::schema> create_igamma__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(igamma_::name, igamma_::overload_name)
      .typed<igamma_::schema>();
}

// aten::igamma_(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & igamma_::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create_igamma__typed_handle();
    return op.call(self, other);
}

// aten::igamma_(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & igamma_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create_igamma__typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(igammac_out, name, "aten::igammac")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(igammac_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(igammac_out, schema_str, "igammac.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::igammac.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<igammac_out::schema> create_igammac_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(igammac_out::name, igammac_out::overload_name)
      .typed<igammac_out::schema>();
}

// aten::igammac.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & igammac_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_igammac_out_typed_handle();
    return op.call(self, other, out);
}

// aten::igammac.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & igammac_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_igammac_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(igammac_, name, "aten::igammac_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(igammac_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(igammac_, schema_str, "igammac_(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::igammac_(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<igammac_::schema> create_igammac__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(igammac_::name, igammac_::overload_name)
      .typed<igammac_::schema>();
}

// aten::igammac_(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & igammac_::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create_igammac__typed_handle();
    return op.call(self, other);
}

// aten::igammac_(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & igammac_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create_igammac__typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nextafter, name, "aten::nextafter")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nextafter, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nextafter, schema_str, "nextafter(Tensor self, Tensor other) -> Tensor")

// aten::nextafter(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<nextafter::schema> create_nextafter_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nextafter::name, nextafter::overload_name)
      .typed<nextafter::schema>();
}

// aten::nextafter(Tensor self, Tensor other) -> Tensor
at::Tensor nextafter::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_nextafter_typed_handle();
    return op.call(self, other);
}

// aten::nextafter(Tensor self, Tensor other) -> Tensor
at::Tensor nextafter::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_nextafter_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fmin_out, name, "aten::fmin")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fmin_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fmin_out, schema_str, "fmin.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::fmin.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<fmin_out::schema> create_fmin_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fmin_out::name, fmin_out::overload_name)
      .typed<fmin_out::schema>();
}

// aten::fmin.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fmin_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_fmin_out_typed_handle();
    return op.call(self, other, out);
}

// aten::fmin.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fmin_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_fmin_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(maximum, name, "aten::maximum")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(maximum, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(maximum, schema_str, "maximum(Tensor self, Tensor other) -> Tensor")

// aten::maximum(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<maximum::schema> create_maximum_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(maximum::name, maximum::overload_name)
      .typed<maximum::schema>();
}

// aten::maximum(Tensor self, Tensor other) -> Tensor
at::Tensor maximum::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_maximum_typed_handle();
    return op.call(self, other);
}

// aten::maximum(Tensor self, Tensor other) -> Tensor
at::Tensor maximum::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_maximum_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_other, name, "aten::max")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_other, overload_name, "other")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_other, schema_str, "max.other(Tensor self, Tensor other) -> Tensor")

// aten::max.other(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<max_other::schema> create_max_other_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(max_other::name, max_other::overload_name)
      .typed<max_other::schema>();
}

// aten::max.other(Tensor self, Tensor other) -> Tensor
at::Tensor max_other::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_max_other_typed_handle();
    return op.call(self, other);
}

// aten::max.other(Tensor self, Tensor other) -> Tensor
at::Tensor max_other::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_max_other_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_out, name, "aten::max")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_out, schema_str, "max.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::max.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<max_out::schema> create_max_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(max_out::name, max_out::overload_name)
      .typed<max_out::schema>();
}

// aten::max.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & max_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_max_out_typed_handle();
    return op.call(self, other, out);
}

// aten::max.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & max_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_max_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(minimum, name, "aten::minimum")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(minimum, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(minimum, schema_str, "minimum(Tensor self, Tensor other) -> Tensor")

// aten::minimum(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<minimum::schema> create_minimum_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(minimum::name, minimum::overload_name)
      .typed<minimum::schema>();
}

// aten::minimum(Tensor self, Tensor other) -> Tensor
at::Tensor minimum::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_minimum_typed_handle();
    return op.call(self, other);
}

// aten::minimum(Tensor self, Tensor other) -> Tensor
at::Tensor minimum::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_minimum_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantile, name, "aten::quantile")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantile, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantile, schema_str, "quantile(Tensor self, Tensor q, int? dim=None, bool keepdim=False) -> Tensor")

// aten::quantile(Tensor self, Tensor q, int? dim=None, bool keepdim=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<quantile::schema> create_quantile_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(quantile::name, quantile::overload_name)
      .typed<quantile::schema>();
}

// aten::quantile(Tensor self, Tensor q, int? dim=None, bool keepdim=False) -> Tensor
at::Tensor quantile::call(const at::Tensor & self, const at::Tensor & q, c10::optional<int64_t> dim, bool keepdim) {
    static auto op = create_quantile_typed_handle();
    return op.call(self, q, dim, keepdim);
}

// aten::quantile(Tensor self, Tensor q, int? dim=None, bool keepdim=False) -> Tensor
at::Tensor quantile::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & q, c10::optional<int64_t> dim, bool keepdim) {
    static auto op = create_quantile_typed_handle();
    return op.redispatch(dispatchKeySet, self, q, dim, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanquantile_scalar, name, "aten::nanquantile")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanquantile_scalar, overload_name, "scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanquantile_scalar, schema_str, "nanquantile.scalar(Tensor self, float q, int? dim=None, bool keepdim=False) -> Tensor")

// aten::nanquantile.scalar(Tensor self, float q, int? dim=None, bool keepdim=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<nanquantile_scalar::schema> create_nanquantile_scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nanquantile_scalar::name, nanquantile_scalar::overload_name)
      .typed<nanquantile_scalar::schema>();
}

// aten::nanquantile.scalar(Tensor self, float q, int? dim=None, bool keepdim=False) -> Tensor
at::Tensor nanquantile_scalar::call(const at::Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim) {
    static auto op = create_nanquantile_scalar_typed_handle();
    return op.call(self, q, dim, keepdim);
}

// aten::nanquantile.scalar(Tensor self, float q, int? dim=None, bool keepdim=False) -> Tensor
at::Tensor nanquantile_scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim) {
    static auto op = create_nanquantile_scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, q, dim, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sort_values_stable, name, "aten::sort")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sort_values_stable, overload_name, "values_stable")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sort_values_stable, schema_str, "sort.values_stable(Tensor self, *, bool? stable, int dim=-1, bool descending=False, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)")

// aten::sort.values_stable(Tensor self, *, bool? stable, int dim=-1, bool descending=False, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
static C10_NOINLINE c10::TypedOperatorHandle<sort_values_stable::schema> create_sort_values_stable_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sort_values_stable::name, sort_values_stable::overload_name)
      .typed<sort_values_stable::schema>();
}

// aten::sort.values_stable(Tensor self, *, bool? stable, int dim=-1, bool descending=False, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> sort_values_stable::call(const at::Tensor & self, c10::optional<bool> stable, int64_t dim, bool descending, at::Tensor & values, at::Tensor & indices) {
    static auto op = create_sort_values_stable_typed_handle();
    return op.call(self, stable, dim, descending, values, indices);
}

// aten::sort.values_stable(Tensor self, *, bool? stable, int dim=-1, bool descending=False, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> sort_values_stable::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<bool> stable, int64_t dim, bool descending, at::Tensor & values, at::Tensor & indices) {
    static auto op = create_sort_values_stable_typed_handle();
    return op.redispatch(dispatchKeySet, self, stable, dim, descending, values, indices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sort_stable, name, "aten::sort")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sort_stable, overload_name, "stable")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sort_stable, schema_str, "sort.stable(Tensor self, *, bool? stable, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)")

// aten::sort.stable(Tensor self, *, bool? stable, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)
static C10_NOINLINE c10::TypedOperatorHandle<sort_stable::schema> create_sort_stable_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sort_stable::name, sort_stable::overload_name)
      .typed<sort_stable::schema>();
}

// aten::sort.stable(Tensor self, *, bool? stable, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> sort_stable::call(const at::Tensor & self, c10::optional<bool> stable, int64_t dim, bool descending) {
    static auto op = create_sort_stable_typed_handle();
    return op.call(self, stable, dim, descending);
}

// aten::sort.stable(Tensor self, *, bool? stable, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> sort_stable::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<bool> stable, int64_t dim, bool descending) {
    static auto op = create_sort_stable_typed_handle();
    return op.redispatch(dispatchKeySet, self, stable, dim, descending);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sort_dimname, name, "aten::sort")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sort_dimname, overload_name, "dimname")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sort_dimname, schema_str, "sort.dimname(Tensor self, Dimname dim, bool descending=False) -> (Tensor values, Tensor indices)")

// aten::sort.dimname(Tensor self, Dimname dim, bool descending=False) -> (Tensor values, Tensor indices)
static C10_NOINLINE c10::TypedOperatorHandle<sort_dimname::schema> create_sort_dimname_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sort_dimname::name, sort_dimname::overload_name)
      .typed<sort_dimname::schema>();
}

// aten::sort.dimname(Tensor self, Dimname dim, bool descending=False) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> sort_dimname::call(const at::Tensor & self, at::Dimname dim, bool descending) {
    static auto op = create_sort_dimname_typed_handle();
    return op.call(self, dim, descending);
}

// aten::sort.dimname(Tensor self, Dimname dim, bool descending=False) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> sort_dimname::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, bool descending) {
    static auto op = create_sort_dimname_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, descending);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(msort, name, "aten::msort")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(msort, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(msort, schema_str, "msort(Tensor self) -> Tensor")

// aten::msort(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<msort::schema> create_msort_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(msort::name, msort::overload_name)
      .typed<msort::schema>();
}

// aten::msort(Tensor self) -> Tensor
at::Tensor msort::call(const at::Tensor & self) {
    static auto op = create_msort_typed_handle();
    return op.call(self);
}

// aten::msort(Tensor self) -> Tensor
at::Tensor msort::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_msort_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(argsort, name, "aten::argsort")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(argsort, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(argsort, schema_str, "argsort(Tensor self, int dim=-1, bool descending=False) -> Tensor")

// aten::argsort(Tensor self, int dim=-1, bool descending=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<argsort::schema> create_argsort_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(argsort::name, argsort::overload_name)
      .typed<argsort::schema>();
}

// aten::argsort(Tensor self, int dim=-1, bool descending=False) -> Tensor
at::Tensor argsort::call(const at::Tensor & self, int64_t dim, bool descending) {
    static auto op = create_argsort_typed_handle();
    return op.call(self, dim, descending);
}

// aten::argsort(Tensor self, int dim=-1, bool descending=False) -> Tensor
at::Tensor argsort::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool descending) {
    static auto op = create_argsort_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, descending);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(topk_values, name, "aten::topk")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(topk_values, overload_name, "values")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(topk_values, schema_str, "topk.values(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)")

// aten::topk.values(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
static C10_NOINLINE c10::TypedOperatorHandle<topk_values::schema> create_topk_values_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(topk_values::name, topk_values::overload_name)
      .typed<topk_values::schema>();
}

// aten::topk.values(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> topk_values::call(const at::Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted, at::Tensor & values, at::Tensor & indices) {
    static auto op = create_topk_values_typed_handle();
    return op.call(self, k, dim, largest, sorted, values, indices);
}

// aten::topk.values(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> topk_values::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted, at::Tensor & values, at::Tensor & indices) {
    static auto op = create_topk_values_typed_handle();
    return op.redispatch(dispatchKeySet, self, k, dim, largest, sorted, values, indices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(topk, name, "aten::topk")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(topk, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(topk, schema_str, "topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)")

// aten::topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)
static C10_NOINLINE c10::TypedOperatorHandle<topk::schema> create_topk_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(topk::name, topk::overload_name)
      .typed<topk::schema>();
}

// aten::topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> topk::call(const at::Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) {
    static auto op = create_topk_typed_handle();
    return op.call(self, k, dim, largest, sorted);
}

// aten::topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> topk::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) {
    static auto op = create_topk_typed_handle();
    return op.redispatch(dispatchKeySet, self, k, dim, largest, sorted);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unfold_backward, name, "aten::unfold_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unfold_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unfold_backward, schema_str, "unfold_backward(Tensor grad_in, int[] input_sizes, int dim, int size, int step) -> Tensor")

// aten::unfold_backward(Tensor grad_in, int[] input_sizes, int dim, int size, int step) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<unfold_backward::schema> create_unfold_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(unfold_backward::name, unfold_backward::overload_name)
      .typed<unfold_backward::schema>();
}

// aten::unfold_backward(Tensor grad_in, int[] input_sizes, int dim, int size, int step) -> Tensor
at::Tensor unfold_backward::call(const at::Tensor & grad_in, at::IntArrayRef input_sizes, int64_t dim, int64_t size, int64_t step) {
    static auto op = create_unfold_backward_typed_handle();
    return op.call(grad_in, input_sizes, dim, size, step);
}

// aten::unfold_backward(Tensor grad_in, int[] input_sizes, int dim, int size, int step) -> Tensor
at::Tensor unfold_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_in, at::IntArrayRef input_sizes, int64_t dim, int64_t size, int64_t step) {
    static auto op = create_unfold_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_in, input_sizes, dim, size, step);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pow_Scalar, name, "aten::pow")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pow_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pow_Scalar, schema_str, "pow.Scalar(Scalar self, Tensor exponent) -> Tensor")

// aten::pow.Scalar(Scalar self, Tensor exponent) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<pow_Scalar::schema> create_pow_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(pow_Scalar::name, pow_Scalar::overload_name)
      .typed<pow_Scalar::schema>();
}

// aten::pow.Scalar(Scalar self, Tensor exponent) -> Tensor
at::Tensor pow_Scalar::call(const at::Scalar & self, const at::Tensor & exponent) {
    static auto op = create_pow_Scalar_typed_handle();
    return op.call(self, exponent);
}

// aten::pow.Scalar(Scalar self, Tensor exponent) -> Tensor
at::Tensor pow_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Scalar & self, const at::Tensor & exponent) {
    static auto op = create_pow_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, exponent);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pow_Tensor_Scalar, name, "aten::pow")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pow_Tensor_Scalar, overload_name, "Tensor_Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pow_Tensor_Scalar, schema_str, "pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor")

// aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<pow_Tensor_Scalar::schema> create_pow_Tensor_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(pow_Tensor_Scalar::name, pow_Tensor_Scalar::overload_name)
      .typed<pow_Tensor_Scalar::schema>();
}

// aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor
at::Tensor pow_Tensor_Scalar::call(const at::Tensor & self, const at::Scalar & exponent) {
    static auto op = create_pow_Tensor_Scalar_typed_handle();
    return op.call(self, exponent);
}

// aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor
at::Tensor pow_Tensor_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & exponent) {
    static auto op = create_pow_Tensor_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, exponent);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(float_power_Tensor_Tensor, name, "aten::float_power")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(float_power_Tensor_Tensor, overload_name, "Tensor_Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(float_power_Tensor_Tensor, schema_str, "float_power.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor")

// aten::float_power.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<float_power_Tensor_Tensor::schema> create_float_power_Tensor_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(float_power_Tensor_Tensor::name, float_power_Tensor_Tensor::overload_name)
      .typed<float_power_Tensor_Tensor::schema>();
}

// aten::float_power.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor
at::Tensor float_power_Tensor_Tensor::call(const at::Tensor & self, const at::Tensor & exponent) {
    static auto op = create_float_power_Tensor_Tensor_typed_handle();
    return op.call(self, exponent);
}

// aten::float_power.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor
at::Tensor float_power_Tensor_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & exponent) {
    static auto op = create_float_power_Tensor_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, exponent);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(alias, name, "aten::alias")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(alias, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(alias, schema_str, "alias(Tensor(a) self) -> Tensor(a)")

// aten::alias(Tensor(a) self) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<alias::schema> create_alias_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(alias::name, alias::overload_name)
      .typed<alias::schema>();
}

// aten::alias(Tensor(a) self) -> Tensor(a)
at::Tensor alias::call(const at::Tensor & self) {
    static auto op = create_alias_typed_handle();
    return op.call(self);
}

// aten::alias(Tensor(a) self) -> Tensor(a)
at::Tensor alias::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_alias_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_div_Scalar, name, "aten::_foreach_div")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_div_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_div_Scalar, schema_str, "_foreach_div.Scalar(Tensor[] tensors, Scalar scalar) -> Tensor[]")

// aten::_foreach_div.Scalar(Tensor[] tensors, Scalar scalar) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_div_Scalar::schema> create__foreach_div_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_div_Scalar::name, _foreach_div_Scalar::overload_name)
      .typed<_foreach_div_Scalar::schema>();
}

// aten::_foreach_div.Scalar(Tensor[] tensors, Scalar scalar) -> Tensor[]
::std::vector<at::Tensor> _foreach_div_Scalar::call(at::TensorList tensors, const at::Scalar & scalar) {
    static auto op = create__foreach_div_Scalar_typed_handle();
    return op.call(tensors, scalar);
}

// aten::_foreach_div.Scalar(Tensor[] tensors, Scalar scalar) -> Tensor[]
::std::vector<at::Tensor> _foreach_div_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, const at::Scalar & scalar) {
    static auto op = create__foreach_div_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, tensors, scalar);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_abs_, name, "aten::_foreach_abs_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_abs_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_abs_, schema_str, "_foreach_abs_(Tensor(a!)[] self) -> ()")

// aten::_foreach_abs_(Tensor(a!)[] self) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_abs_::schema> create__foreach_abs__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_abs_::name, _foreach_abs_::overload_name)
      .typed<_foreach_abs_::schema>();
}

// aten::_foreach_abs_(Tensor(a!)[] self) -> ()
void _foreach_abs_::call(at::TensorList self) {
    static auto op = create__foreach_abs__typed_handle();
    return op.call(self);
}

// aten::_foreach_abs_(Tensor(a!)[] self) -> ()
void _foreach_abs_::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self) {
    static auto op = create__foreach_abs__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_acos, name, "aten::_foreach_acos")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_acos, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_acos, schema_str, "_foreach_acos(Tensor[] tensors) -> Tensor[]")

// aten::_foreach_acos(Tensor[] tensors) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_acos::schema> create__foreach_acos_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_acos::name, _foreach_acos::overload_name)
      .typed<_foreach_acos::schema>();
}

// aten::_foreach_acos(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_acos::call(at::TensorList tensors) {
    static auto op = create__foreach_acos_typed_handle();
    return op.call(tensors);
}

// aten::_foreach_acos(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_acos::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create__foreach_acos_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_atan, name, "aten::_foreach_atan")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_atan, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_atan, schema_str, "_foreach_atan(Tensor[] tensors) -> Tensor[]")

// aten::_foreach_atan(Tensor[] tensors) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_atan::schema> create__foreach_atan_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_atan::name, _foreach_atan::overload_name)
      .typed<_foreach_atan::schema>();
}

// aten::_foreach_atan(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_atan::call(at::TensorList tensors) {
    static auto op = create__foreach_atan_typed_handle();
    return op.call(tensors);
}

// aten::_foreach_atan(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_atan::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create__foreach_atan_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_ceil, name, "aten::_foreach_ceil")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_ceil, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_ceil, schema_str, "_foreach_ceil(Tensor[] tensors) -> Tensor[]")

// aten::_foreach_ceil(Tensor[] tensors) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_ceil::schema> create__foreach_ceil_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_ceil::name, _foreach_ceil::overload_name)
      .typed<_foreach_ceil::schema>();
}

// aten::_foreach_ceil(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_ceil::call(at::TensorList tensors) {
    static auto op = create__foreach_ceil_typed_handle();
    return op.call(tensors);
}

// aten::_foreach_ceil(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_ceil::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create__foreach_ceil_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_erf, name, "aten::_foreach_erf")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_erf, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_erf, schema_str, "_foreach_erf(Tensor[] tensors) -> Tensor[]")

// aten::_foreach_erf(Tensor[] tensors) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_erf::schema> create__foreach_erf_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_erf::name, _foreach_erf::overload_name)
      .typed<_foreach_erf::schema>();
}

// aten::_foreach_erf(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_erf::call(at::TensorList tensors) {
    static auto op = create__foreach_erf_typed_handle();
    return op.call(tensors);
}

// aten::_foreach_erf(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_erf::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create__foreach_erf_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_log10_, name, "aten::_foreach_log10_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_log10_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_log10_, schema_str, "_foreach_log10_(Tensor(a!)[] self) -> ()")

// aten::_foreach_log10_(Tensor(a!)[] self) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_log10_::schema> create__foreach_log10__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_log10_::name, _foreach_log10_::overload_name)
      .typed<_foreach_log10_::schema>();
}

// aten::_foreach_log10_(Tensor(a!)[] self) -> ()
void _foreach_log10_::call(at::TensorList self) {
    static auto op = create__foreach_log10__typed_handle();
    return op.call(self);
}

// aten::_foreach_log10_(Tensor(a!)[] self) -> ()
void _foreach_log10_::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self) {
    static auto op = create__foreach_log10__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_log1p_, name, "aten::_foreach_log1p_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_log1p_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_log1p_, schema_str, "_foreach_log1p_(Tensor(a!)[] self) -> ()")

// aten::_foreach_log1p_(Tensor(a!)[] self) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_log1p_::schema> create__foreach_log1p__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_log1p_::name, _foreach_log1p_::overload_name)
      .typed<_foreach_log1p_::schema>();
}

// aten::_foreach_log1p_(Tensor(a!)[] self) -> ()
void _foreach_log1p_::call(at::TensorList self) {
    static auto op = create__foreach_log1p__typed_handle();
    return op.call(self);
}

// aten::_foreach_log1p_(Tensor(a!)[] self) -> ()
void _foreach_log1p_::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self) {
    static auto op = create__foreach_log1p__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_log2, name, "aten::_foreach_log2")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_log2, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_log2, schema_str, "_foreach_log2(Tensor[] tensors) -> Tensor[]")

// aten::_foreach_log2(Tensor[] tensors) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_log2::schema> create__foreach_log2_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_log2::name, _foreach_log2::overload_name)
      .typed<_foreach_log2::schema>();
}

// aten::_foreach_log2(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_log2::call(at::TensorList tensors) {
    static auto op = create__foreach_log2_typed_handle();
    return op.call(tensors);
}

// aten::_foreach_log2(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_log2::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create__foreach_log2_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_neg_, name, "aten::_foreach_neg_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_neg_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_neg_, schema_str, "_foreach_neg_(Tensor(a!)[] self) -> ()")

// aten::_foreach_neg_(Tensor(a!)[] self) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_neg_::schema> create__foreach_neg__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_neg_::name, _foreach_neg_::overload_name)
      .typed<_foreach_neg_::schema>();
}

// aten::_foreach_neg_(Tensor(a!)[] self) -> ()
void _foreach_neg_::call(at::TensorList self) {
    static auto op = create__foreach_neg__typed_handle();
    return op.call(self);
}

// aten::_foreach_neg_(Tensor(a!)[] self) -> ()
void _foreach_neg_::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self) {
    static auto op = create__foreach_neg__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_round_, name, "aten::_foreach_round_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_round_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_round_, schema_str, "_foreach_round_(Tensor(a!)[] self) -> ()")

// aten::_foreach_round_(Tensor(a!)[] self) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_round_::schema> create__foreach_round__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_round_::name, _foreach_round_::overload_name)
      .typed<_foreach_round_::schema>();
}

// aten::_foreach_round_(Tensor(a!)[] self) -> ()
void _foreach_round_::call(at::TensorList self) {
    static auto op = create__foreach_round__typed_handle();
    return op.call(self);
}

// aten::_foreach_round_(Tensor(a!)[] self) -> ()
void _foreach_round_::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self) {
    static auto op = create__foreach_round__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_frac_, name, "aten::_foreach_frac_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_frac_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_frac_, schema_str, "_foreach_frac_(Tensor(a!)[] self) -> ()")

// aten::_foreach_frac_(Tensor(a!)[] self) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_frac_::schema> create__foreach_frac__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_frac_::name, _foreach_frac_::overload_name)
      .typed<_foreach_frac_::schema>();
}

// aten::_foreach_frac_(Tensor(a!)[] self) -> ()
void _foreach_frac_::call(at::TensorList self) {
    static auto op = create__foreach_frac__typed_handle();
    return op.call(self);
}

// aten::_foreach_frac_(Tensor(a!)[] self) -> ()
void _foreach_frac_::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self) {
    static auto op = create__foreach_frac__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_addcmul_Scalar, name, "aten::_foreach_addcmul")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_addcmul_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_addcmul_Scalar, schema_str, "_foreach_addcmul.Scalar(Tensor[] input, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1) -> Tensor[]")

// aten::_foreach_addcmul.Scalar(Tensor[] input, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_addcmul_Scalar::schema> create__foreach_addcmul_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_addcmul_Scalar::name, _foreach_addcmul_Scalar::overload_name)
      .typed<_foreach_addcmul_Scalar::schema>();
}

// aten::_foreach_addcmul.Scalar(Tensor[] input, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1) -> Tensor[]
::std::vector<at::Tensor> _foreach_addcmul_Scalar::call(at::TensorList input, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value) {
    static auto op = create__foreach_addcmul_Scalar_typed_handle();
    return op.call(input, tensor1, tensor2, value);
}

// aten::_foreach_addcmul.Scalar(Tensor[] input, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1) -> Tensor[]
::std::vector<at::Tensor> _foreach_addcmul_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList input, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value) {
    static auto op = create__foreach_addcmul_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, input, tensor1, tensor2, value);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mse_loss, name, "aten::mse_loss")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mse_loss, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mse_loss, schema_str, "mse_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor")

// aten::mse_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<mse_loss::schema> create_mse_loss_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mse_loss::name, mse_loss::overload_name)
      .typed<mse_loss::schema>();
}

// aten::mse_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor
at::Tensor mse_loss::call(const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
    static auto op = create_mse_loss_typed_handle();
    return op.call(self, target, reduction);
}

// aten::mse_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor
at::Tensor mse_loss::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
    static auto op = create_mse_loss_typed_handle();
    return op.redispatch(dispatchKeySet, self, target, reduction);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(l1_loss, name, "aten::l1_loss")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(l1_loss, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(l1_loss, schema_str, "l1_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor")

// aten::l1_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<l1_loss::schema> create_l1_loss_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(l1_loss::name, l1_loss::overload_name)
      .typed<l1_loss::schema>();
}

// aten::l1_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor
at::Tensor l1_loss::call(const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
    static auto op = create_l1_loss_typed_handle();
    return op.call(self, target, reduction);
}

// aten::l1_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor
at::Tensor l1_loss::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
    static auto op = create_l1_loss_typed_handle();
    return op.redispatch(dispatchKeySet, self, target, reduction);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multilabel_margin_loss_forward_output, name, "aten::multilabel_margin_loss_forward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multilabel_margin_loss_forward_output, overload_name, "output")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multilabel_margin_loss_forward_output, schema_str, "multilabel_margin_loss_forward.output(Tensor self, Tensor target, int reduction, *, Tensor(a!) output, Tensor(b!) is_target) -> (Tensor(a!), Tensor(b!))")

// aten::multilabel_margin_loss_forward.output(Tensor self, Tensor target, int reduction, *, Tensor(a!) output, Tensor(b!) is_target) -> (Tensor(a!), Tensor(b!))
static C10_NOINLINE c10::TypedOperatorHandle<multilabel_margin_loss_forward_output::schema> create_multilabel_margin_loss_forward_output_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(multilabel_margin_loss_forward_output::name, multilabel_margin_loss_forward_output::overload_name)
      .typed<multilabel_margin_loss_forward_output::schema>();
}

// aten::multilabel_margin_loss_forward.output(Tensor self, Tensor target, int reduction, *, Tensor(a!) output, Tensor(b!) is_target) -> (Tensor(a!), Tensor(b!))
::std::tuple<at::Tensor &,at::Tensor &> multilabel_margin_loss_forward_output::call(const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & output, at::Tensor & is_target) {
    static auto op = create_multilabel_margin_loss_forward_output_typed_handle();
    return op.call(self, target, reduction, output, is_target);
}

// aten::multilabel_margin_loss_forward.output(Tensor self, Tensor target, int reduction, *, Tensor(a!) output, Tensor(b!) is_target) -> (Tensor(a!), Tensor(b!))
::std::tuple<at::Tensor &,at::Tensor &> multilabel_margin_loss_forward_output::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & output, at::Tensor & is_target) {
    static auto op = create_multilabel_margin_loss_forward_output_typed_handle();
    return op.redispatch(dispatchKeySet, self, target, reduction, output, is_target);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss_nd, name, "aten::nll_loss_nd")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss_nd, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss_nd, schema_str, "nll_loss_nd(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100) -> Tensor")

// aten::nll_loss_nd(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<nll_loss_nd::schema> create_nll_loss_nd_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nll_loss_nd::name, nll_loss_nd::overload_name)
      .typed<nll_loss_nd::schema>();
}

// aten::nll_loss_nd(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100) -> Tensor
at::Tensor nll_loss_nd::call(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index) {
    static auto op = create_nll_loss_nd_typed_handle();
    return op.call(self, target, weight, reduction, ignore_index);
}

// aten::nll_loss_nd(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100) -> Tensor
at::Tensor nll_loss_nd::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index) {
    static auto op = create_nll_loss_nd_typed_handle();
    return op.redispatch(dispatchKeySet, self, target, weight, reduction, ignore_index);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss_forward_output, name, "aten::nll_loss_forward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss_forward_output, overload_name, "output")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss_forward_output, schema_str, "nll_loss_forward.output(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, *, Tensor(a!) output, Tensor(b!) total_weight) -> (Tensor(a!), Tensor(b!))")

// aten::nll_loss_forward.output(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, *, Tensor(a!) output, Tensor(b!) total_weight) -> (Tensor(a!), Tensor(b!))
static C10_NOINLINE c10::TypedOperatorHandle<nll_loss_forward_output::schema> create_nll_loss_forward_output_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nll_loss_forward_output::name, nll_loss_forward_output::overload_name)
      .typed<nll_loss_forward_output::schema>();
}

// aten::nll_loss_forward.output(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, *, Tensor(a!) output, Tensor(b!) total_weight) -> (Tensor(a!), Tensor(b!))
::std::tuple<at::Tensor &,at::Tensor &> nll_loss_forward_output::call(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, at::Tensor & output, at::Tensor & total_weight) {
    static auto op = create_nll_loss_forward_output_typed_handle();
    return op.call(self, target, weight, reduction, ignore_index, output, total_weight);
}

// aten::nll_loss_forward.output(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, *, Tensor(a!) output, Tensor(b!) total_weight) -> (Tensor(a!), Tensor(b!))
::std::tuple<at::Tensor &,at::Tensor &> nll_loss_forward_output::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, at::Tensor & output, at::Tensor & total_weight) {
    static auto op = create_nll_loss_forward_output_typed_handle();
    return op.redispatch(dispatchKeySet, self, target, weight, reduction, ignore_index, output, total_weight);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss2d_out, name, "aten::nll_loss2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss2d_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss2d_out, schema_str, "nll_loss2d.out(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100, *, Tensor(a!) out) -> Tensor(a!)")

// aten::nll_loss2d.out(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<nll_loss2d_out::schema> create_nll_loss2d_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nll_loss2d_out::name, nll_loss2d_out::overload_name)
      .typed<nll_loss2d_out::schema>();
}

// aten::nll_loss2d.out(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & nll_loss2d_out::call(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, at::Tensor & out) {
    static auto op = create_nll_loss2d_out_typed_handle();
    return op.call(self, target, weight, reduction, ignore_index, out);
}

// aten::nll_loss2d.out(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & nll_loss2d_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, at::Tensor & out) {
    static auto op = create_nll_loss2d_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, target, weight, reduction, ignore_index, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss2d, name, "aten::nll_loss2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss2d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss2d, schema_str, "nll_loss2d(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100) -> Tensor")

// aten::nll_loss2d(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<nll_loss2d::schema> create_nll_loss2d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nll_loss2d::name, nll_loss2d::overload_name)
      .typed<nll_loss2d::schema>();
}

// aten::nll_loss2d(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100) -> Tensor
at::Tensor nll_loss2d::call(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index) {
    static auto op = create_nll_loss2d_typed_handle();
    return op.call(self, target, weight, reduction, ignore_index);
}

// aten::nll_loss2d(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100) -> Tensor
at::Tensor nll_loss2d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index) {
    static auto op = create_nll_loss2d_typed_handle();
    return op.redispatch(dispatchKeySet, self, target, weight, reduction, ignore_index);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss2d_forward, name, "aten::nll_loss2d_forward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss2d_forward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss2d_forward, schema_str, "nll_loss2d_forward(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index) -> (Tensor output, Tensor total_weight)")

// aten::nll_loss2d_forward(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index) -> (Tensor output, Tensor total_weight)
static C10_NOINLINE c10::TypedOperatorHandle<nll_loss2d_forward::schema> create_nll_loss2d_forward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nll_loss2d_forward::name, nll_loss2d_forward::overload_name)
      .typed<nll_loss2d_forward::schema>();
}

// aten::nll_loss2d_forward(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index) -> (Tensor output, Tensor total_weight)
::std::tuple<at::Tensor,at::Tensor> nll_loss2d_forward::call(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index) {
    static auto op = create_nll_loss2d_forward_typed_handle();
    return op.call(self, target, weight, reduction, ignore_index);
}

// aten::nll_loss2d_forward(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index) -> (Tensor output, Tensor total_weight)
::std::tuple<at::Tensor,at::Tensor> nll_loss2d_forward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index) {
    static auto op = create_nll_loss2d_forward_typed_handle();
    return op.redispatch(dispatchKeySet, self, target, weight, reduction, ignore_index);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss2d_backward_grad_input, name, "aten::nll_loss2d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss2d_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss2d_backward_grad_input, schema_str, "nll_loss2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::nll_loss2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<nll_loss2d_backward_grad_input::schema> create_nll_loss2d_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nll_loss2d_backward_grad_input::name, nll_loss2d_backward_grad_input::overload_name)
      .typed<nll_loss2d_backward_grad_input::schema>();
}

// aten::nll_loss2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & nll_loss2d_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight, at::Tensor & grad_input) {
    static auto op = create_nll_loss2d_backward_grad_input_typed_handle();
    return op.call(grad_output, self, target, weight, reduction, ignore_index, total_weight, grad_input);
}

// aten::nll_loss2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & nll_loss2d_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight, at::Tensor & grad_input) {
    static auto op = create_nll_loss2d_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, target, weight, reduction, ignore_index, total_weight, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss2d_backward, name, "aten::nll_loss2d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss2d_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss2d_backward, schema_str, "nll_loss2d_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight) -> Tensor")

// aten::nll_loss2d_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<nll_loss2d_backward::schema> create_nll_loss2d_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nll_loss2d_backward::name, nll_loss2d_backward::overload_name)
      .typed<nll_loss2d_backward::schema>();
}

// aten::nll_loss2d_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight) -> Tensor
at::Tensor nll_loss2d_backward::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight) {
    static auto op = create_nll_loss2d_backward_typed_handle();
    return op.call(grad_output, self, target, weight, reduction, ignore_index, total_weight);
}

// aten::nll_loss2d_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight) -> Tensor
at::Tensor nll_loss2d_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight) {
    static auto op = create_nll_loss2d_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, target, weight, reduction, ignore_index, total_weight);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(smooth_l1_loss_out, name, "aten::smooth_l1_loss")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(smooth_l1_loss_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(smooth_l1_loss_out, schema_str, "smooth_l1_loss.out(Tensor self, Tensor target, int reduction=Mean, float beta=1.0, *, Tensor(a!) out) -> Tensor(a!)")

// aten::smooth_l1_loss.out(Tensor self, Tensor target, int reduction=Mean, float beta=1.0, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<smooth_l1_loss_out::schema> create_smooth_l1_loss_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(smooth_l1_loss_out::name, smooth_l1_loss_out::overload_name)
      .typed<smooth_l1_loss_out::schema>();
}

// aten::smooth_l1_loss.out(Tensor self, Tensor target, int reduction=Mean, float beta=1.0, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & smooth_l1_loss_out::call(const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta, at::Tensor & out) {
    static auto op = create_smooth_l1_loss_out_typed_handle();
    return op.call(self, target, reduction, beta, out);
}

// aten::smooth_l1_loss.out(Tensor self, Tensor target, int reduction=Mean, float beta=1.0, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & smooth_l1_loss_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta, at::Tensor & out) {
    static auto op = create_smooth_l1_loss_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, target, reduction, beta, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(soft_margin_loss, name, "aten::soft_margin_loss")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(soft_margin_loss, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(soft_margin_loss, schema_str, "soft_margin_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor")

// aten::soft_margin_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<soft_margin_loss::schema> create_soft_margin_loss_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(soft_margin_loss::name, soft_margin_loss::overload_name)
      .typed<soft_margin_loss::schema>();
}

// aten::soft_margin_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor
at::Tensor soft_margin_loss::call(const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
    static auto op = create_soft_margin_loss_typed_handle();
    return op.call(self, target, reduction);
}

// aten::soft_margin_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor
at::Tensor soft_margin_loss::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
    static auto op = create_soft_margin_loss_typed_handle();
    return op.redispatch(dispatchKeySet, self, target, reduction);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(glu, name, "aten::glu")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(glu, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(glu, schema_str, "glu(Tensor self, int dim=-1) -> Tensor")

// aten::glu(Tensor self, int dim=-1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<glu::schema> create_glu_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(glu::name, glu::overload_name)
      .typed<glu::schema>();
}

// aten::glu(Tensor self, int dim=-1) -> Tensor
at::Tensor glu::call(const at::Tensor & self, int64_t dim) {
    static auto op = create_glu_typed_handle();
    return op.call(self, dim);
}

// aten::glu(Tensor self, int dim=-1) -> Tensor
at::Tensor glu::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim) {
    static auto op = create_glu_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardsigmoid_, name, "aten::hardsigmoid_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardsigmoid_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardsigmoid_, schema_str, "hardsigmoid_(Tensor(a!) self) -> Tensor(a!)")

// aten::hardsigmoid_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<hardsigmoid_::schema> create_hardsigmoid__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(hardsigmoid_::name, hardsigmoid_::overload_name)
      .typed<hardsigmoid_::schema>();
}

// aten::hardsigmoid_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & hardsigmoid_::call(at::Tensor & self) {
    static auto op = create_hardsigmoid__typed_handle();
    return op.call(self);
}

// aten::hardsigmoid_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & hardsigmoid_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_hardsigmoid__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardtanh_out, name, "aten::hardtanh")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardtanh_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardtanh_out, schema_str, "hardtanh.out(Tensor self, Scalar min_val=-1, Scalar max_val=1, *, Tensor(a!) out) -> Tensor(a!)")

// aten::hardtanh.out(Tensor self, Scalar min_val=-1, Scalar max_val=1, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<hardtanh_out::schema> create_hardtanh_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(hardtanh_out::name, hardtanh_out::overload_name)
      .typed<hardtanh_out::schema>();
}

// aten::hardtanh.out(Tensor self, Scalar min_val=-1, Scalar max_val=1, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & hardtanh_out::call(const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val, at::Tensor & out) {
    static auto op = create_hardtanh_out_typed_handle();
    return op.call(self, min_val, max_val, out);
}

// aten::hardtanh.out(Tensor self, Scalar min_val=-1, Scalar max_val=1, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & hardtanh_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val, at::Tensor & out) {
    static auto op = create_hardtanh_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, min_val, max_val, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardtanh, name, "aten::hardtanh")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardtanh, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardtanh, schema_str, "hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor")

// aten::hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<hardtanh::schema> create_hardtanh_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(hardtanh::name, hardtanh::overload_name)
      .typed<hardtanh::schema>();
}

// aten::hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor
at::Tensor hardtanh::call(const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val) {
    static auto op = create_hardtanh_typed_handle();
    return op.call(self, min_val, max_val);
}

// aten::hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor
at::Tensor hardtanh::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val) {
    static auto op = create_hardtanh_typed_handle();
    return op.redispatch(dispatchKeySet, self, min_val, max_val);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardswish_out, name, "aten::hardswish")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardswish_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardswish_out, schema_str, "hardswish.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::hardswish.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<hardswish_out::schema> create_hardswish_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(hardswish_out::name, hardswish_out::overload_name)
      .typed<hardswish_out::schema>();
}

// aten::hardswish.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & hardswish_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_hardswish_out_typed_handle();
    return op.call(self, out);
}

// aten::hardswish.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & hardswish_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_hardswish_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardswish_backward, name, "aten::hardswish_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardswish_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardswish_backward, schema_str, "hardswish_backward(Tensor grad_output, Tensor self) -> Tensor")

// aten::hardswish_backward(Tensor grad_output, Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<hardswish_backward::schema> create_hardswish_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(hardswish_backward::name, hardswish_backward::overload_name)
      .typed<hardswish_backward::schema>();
}

// aten::hardswish_backward(Tensor grad_output, Tensor self) -> Tensor
at::Tensor hardswish_backward::call(const at::Tensor & grad_output, const at::Tensor & self) {
    static auto op = create_hardswish_backward_typed_handle();
    return op.call(grad_output, self);
}

// aten::hardswish_backward(Tensor grad_output, Tensor self) -> Tensor
at::Tensor hardswish_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self) {
    static auto op = create_hardswish_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(leaky_relu_out, name, "aten::leaky_relu")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(leaky_relu_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(leaky_relu_out, schema_str, "leaky_relu.out(Tensor self, Scalar negative_slope=0.01, *, Tensor(a!) out) -> Tensor(a!)")

// aten::leaky_relu.out(Tensor self, Scalar negative_slope=0.01, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<leaky_relu_out::schema> create_leaky_relu_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(leaky_relu_out::name, leaky_relu_out::overload_name)
      .typed<leaky_relu_out::schema>();
}

// aten::leaky_relu.out(Tensor self, Scalar negative_slope=0.01, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & leaky_relu_out::call(const at::Tensor & self, const at::Scalar & negative_slope, at::Tensor & out) {
    static auto op = create_leaky_relu_out_typed_handle();
    return op.call(self, negative_slope, out);
}

// aten::leaky_relu.out(Tensor self, Scalar negative_slope=0.01, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & leaky_relu_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & negative_slope, at::Tensor & out) {
    static auto op = create_leaky_relu_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, negative_slope, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(leaky_relu, name, "aten::leaky_relu")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(leaky_relu, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(leaky_relu, schema_str, "leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor")

// aten::leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<leaky_relu::schema> create_leaky_relu_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(leaky_relu::name, leaky_relu::overload_name)
      .typed<leaky_relu::schema>();
}

// aten::leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor
at::Tensor leaky_relu::call(const at::Tensor & self, const at::Scalar & negative_slope) {
    static auto op = create_leaky_relu_typed_handle();
    return op.call(self, negative_slope);
}

// aten::leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor
at::Tensor leaky_relu::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & negative_slope) {
    static auto op = create_leaky_relu_typed_handle();
    return op.redispatch(dispatchKeySet, self, negative_slope);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log_sigmoid_forward, name, "aten::log_sigmoid_forward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log_sigmoid_forward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log_sigmoid_forward, schema_str, "log_sigmoid_forward(Tensor self) -> (Tensor output, Tensor buffer)")

// aten::log_sigmoid_forward(Tensor self) -> (Tensor output, Tensor buffer)
static C10_NOINLINE c10::TypedOperatorHandle<log_sigmoid_forward::schema> create_log_sigmoid_forward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(log_sigmoid_forward::name, log_sigmoid_forward::overload_name)
      .typed<log_sigmoid_forward::schema>();
}

// aten::log_sigmoid_forward(Tensor self) -> (Tensor output, Tensor buffer)
::std::tuple<at::Tensor,at::Tensor> log_sigmoid_forward::call(const at::Tensor & self) {
    static auto op = create_log_sigmoid_forward_typed_handle();
    return op.call(self);
}

// aten::log_sigmoid_forward(Tensor self) -> (Tensor output, Tensor buffer)
::std::tuple<at::Tensor,at::Tensor> log_sigmoid_forward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_log_sigmoid_forward_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log_sigmoid_backward, name, "aten::log_sigmoid_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log_sigmoid_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log_sigmoid_backward, schema_str, "log_sigmoid_backward(Tensor grad_output, Tensor self, Tensor buffer) -> Tensor")

// aten::log_sigmoid_backward(Tensor grad_output, Tensor self, Tensor buffer) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<log_sigmoid_backward::schema> create_log_sigmoid_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(log_sigmoid_backward::name, log_sigmoid_backward::overload_name)
      .typed<log_sigmoid_backward::schema>();
}

// aten::log_sigmoid_backward(Tensor grad_output, Tensor self, Tensor buffer) -> Tensor
at::Tensor log_sigmoid_backward::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & buffer) {
    static auto op = create_log_sigmoid_backward_typed_handle();
    return op.call(grad_output, self, buffer);
}

// aten::log_sigmoid_backward(Tensor grad_output, Tensor self, Tensor buffer) -> Tensor
at::Tensor log_sigmoid_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & buffer) {
    static auto op = create_log_sigmoid_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, buffer);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rrelu_with_noise_out, name, "aten::rrelu_with_noise")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rrelu_with_noise_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rrelu_with_noise_out, schema_str, "rrelu_with_noise.out(Tensor self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::rrelu_with_noise.out(Tensor self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<rrelu_with_noise_out::schema> create_rrelu_with_noise_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(rrelu_with_noise_out::name, rrelu_with_noise_out::overload_name)
      .typed<rrelu_with_noise_out::schema>();
}

// aten::rrelu_with_noise.out(Tensor self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & rrelu_with_noise_out::call(const at::Tensor & self, const at::Tensor & noise, const at::Scalar & lower, const at::Scalar & upper, bool training, c10::optional<at::Generator> generator, at::Tensor & out) {
    static auto op = create_rrelu_with_noise_out_typed_handle();
    return op.call(self, noise, lower, upper, training, generator, out);
}

// aten::rrelu_with_noise.out(Tensor self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & rrelu_with_noise_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & noise, const at::Scalar & lower, const at::Scalar & upper, bool training, c10::optional<at::Generator> generator, at::Tensor & out) {
    static auto op = create_rrelu_with_noise_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, noise, lower, upper, training, generator, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rrelu_with_noise_, name, "aten::rrelu_with_noise_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rrelu_with_noise_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rrelu_with_noise_, schema_str, "rrelu_with_noise_(Tensor(a!) self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor(a!)")

// aten::rrelu_with_noise_(Tensor(a!) self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<rrelu_with_noise_::schema> create_rrelu_with_noise__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(rrelu_with_noise_::name, rrelu_with_noise_::overload_name)
      .typed<rrelu_with_noise_::schema>();
}

// aten::rrelu_with_noise_(Tensor(a!) self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor(a!)
at::Tensor & rrelu_with_noise_::call(at::Tensor & self, const at::Tensor & noise, const at::Scalar & lower, const at::Scalar & upper, bool training, c10::optional<at::Generator> generator) {
    static auto op = create_rrelu_with_noise__typed_handle();
    return op.call(self, noise, lower, upper, training, generator);
}

// aten::rrelu_with_noise_(Tensor(a!) self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor(a!)
at::Tensor & rrelu_with_noise_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & noise, const at::Scalar & lower, const at::Scalar & upper, bool training, c10::optional<at::Generator> generator) {
    static auto op = create_rrelu_with_noise__typed_handle();
    return op.redispatch(dispatchKeySet, self, noise, lower, upper, training, generator);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(softplus_backward_grad_input, name, "aten::softplus_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(softplus_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(softplus_backward_grad_input, schema_str, "softplus_backward.grad_input(Tensor grad_output, Tensor self, Scalar beta, Scalar threshold, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::softplus_backward.grad_input(Tensor grad_output, Tensor self, Scalar beta, Scalar threshold, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<softplus_backward_grad_input::schema> create_softplus_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(softplus_backward_grad_input::name, softplus_backward_grad_input::overload_name)
      .typed<softplus_backward_grad_input::schema>();
}

// aten::softplus_backward.grad_input(Tensor grad_output, Tensor self, Scalar beta, Scalar threshold, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & softplus_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold, const at::Tensor & output, at::Tensor & grad_input) {
    static auto op = create_softplus_backward_grad_input_typed_handle();
    return op.call(grad_output, self, beta, threshold, output, grad_input);
}

// aten::softplus_backward.grad_input(Tensor grad_output, Tensor self, Scalar beta, Scalar threshold, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & softplus_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold, const at::Tensor & output, at::Tensor & grad_input) {
    static auto op = create_softplus_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, beta, threshold, output, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(softshrink, name, "aten::softshrink")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(softshrink, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(softshrink, schema_str, "softshrink(Tensor self, Scalar lambd=0.5) -> Tensor")

// aten::softshrink(Tensor self, Scalar lambd=0.5) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<softshrink::schema> create_softshrink_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(softshrink::name, softshrink::overload_name)
      .typed<softshrink::schema>();
}

// aten::softshrink(Tensor self, Scalar lambd=0.5) -> Tensor
at::Tensor softshrink::call(const at::Tensor & self, const at::Scalar & lambd) {
    static auto op = create_softshrink_typed_handle();
    return op.call(self, lambd);
}

// aten::softshrink(Tensor self, Scalar lambd=0.5) -> Tensor
at::Tensor softshrink::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & lambd) {
    static auto op = create_softshrink_typed_handle();
    return op.redispatch(dispatchKeySet, self, lambd);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_adaptive_avg_pool3d_backward, name, "aten::_adaptive_avg_pool3d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_adaptive_avg_pool3d_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_adaptive_avg_pool3d_backward, schema_str, "_adaptive_avg_pool3d_backward(Tensor grad_output, Tensor self) -> Tensor")

// aten::_adaptive_avg_pool3d_backward(Tensor grad_output, Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_adaptive_avg_pool3d_backward::schema> create__adaptive_avg_pool3d_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_adaptive_avg_pool3d_backward::name, _adaptive_avg_pool3d_backward::overload_name)
      .typed<_adaptive_avg_pool3d_backward::schema>();
}

// aten::_adaptive_avg_pool3d_backward(Tensor grad_output, Tensor self) -> Tensor
at::Tensor _adaptive_avg_pool3d_backward::call(const at::Tensor & grad_output, const at::Tensor & self) {
    static auto op = create__adaptive_avg_pool3d_backward_typed_handle();
    return op.call(grad_output, self);
}

// aten::_adaptive_avg_pool3d_backward(Tensor grad_output, Tensor self) -> Tensor
at::Tensor _adaptive_avg_pool3d_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self) {
    static auto op = create__adaptive_avg_pool3d_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_max_pool2d_out, name, "aten::adaptive_max_pool2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_max_pool2d_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_max_pool2d_out, schema_str, "adaptive_max_pool2d.out(Tensor self, int[2] output_size, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))")

// aten::adaptive_max_pool2d.out(Tensor self, int[2] output_size, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
static C10_NOINLINE c10::TypedOperatorHandle<adaptive_max_pool2d_out::schema> create_adaptive_max_pool2d_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(adaptive_max_pool2d_out::name, adaptive_max_pool2d_out::overload_name)
      .typed<adaptive_max_pool2d_out::schema>();
}

// aten::adaptive_max_pool2d.out(Tensor self, int[2] output_size, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
::std::tuple<at::Tensor &,at::Tensor &> adaptive_max_pool2d_out::call(const at::Tensor & self, at::IntArrayRef output_size, at::Tensor & out, at::Tensor & indices) {
    static auto op = create_adaptive_max_pool2d_out_typed_handle();
    return op.call(self, output_size, out, indices);
}

// aten::adaptive_max_pool2d.out(Tensor self, int[2] output_size, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
::std::tuple<at::Tensor &,at::Tensor &> adaptive_max_pool2d_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size, at::Tensor & out, at::Tensor & indices) {
    static auto op = create_adaptive_max_pool2d_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, output_size, out, indices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_max_pool2d_backward, name, "aten::adaptive_max_pool2d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_max_pool2d_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_max_pool2d_backward, schema_str, "adaptive_max_pool2d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor")

// aten::adaptive_max_pool2d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<adaptive_max_pool2d_backward::schema> create_adaptive_max_pool2d_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(adaptive_max_pool2d_backward::name, adaptive_max_pool2d_backward::overload_name)
      .typed<adaptive_max_pool2d_backward::schema>();
}

// aten::adaptive_max_pool2d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor
at::Tensor adaptive_max_pool2d_backward::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices) {
    static auto op = create_adaptive_max_pool2d_backward_typed_handle();
    return op.call(grad_output, self, indices);
}

// aten::adaptive_max_pool2d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor
at::Tensor adaptive_max_pool2d_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices) {
    static auto op = create_adaptive_max_pool2d_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, indices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_max_pool3d_out, name, "aten::adaptive_max_pool3d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_max_pool3d_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_max_pool3d_out, schema_str, "adaptive_max_pool3d.out(Tensor self, int[3] output_size, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))")

// aten::adaptive_max_pool3d.out(Tensor self, int[3] output_size, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
static C10_NOINLINE c10::TypedOperatorHandle<adaptive_max_pool3d_out::schema> create_adaptive_max_pool3d_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(adaptive_max_pool3d_out::name, adaptive_max_pool3d_out::overload_name)
      .typed<adaptive_max_pool3d_out::schema>();
}

// aten::adaptive_max_pool3d.out(Tensor self, int[3] output_size, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
::std::tuple<at::Tensor &,at::Tensor &> adaptive_max_pool3d_out::call(const at::Tensor & self, at::IntArrayRef output_size, at::Tensor & out, at::Tensor & indices) {
    static auto op = create_adaptive_max_pool3d_out_typed_handle();
    return op.call(self, output_size, out, indices);
}

// aten::adaptive_max_pool3d.out(Tensor self, int[3] output_size, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
::std::tuple<at::Tensor &,at::Tensor &> adaptive_max_pool3d_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size, at::Tensor & out, at::Tensor & indices) {
    static auto op = create_adaptive_max_pool3d_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, output_size, out, indices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_max_pool3d_backward, name, "aten::adaptive_max_pool3d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_max_pool3d_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_max_pool3d_backward, schema_str, "adaptive_max_pool3d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor")

// aten::adaptive_max_pool3d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<adaptive_max_pool3d_backward::schema> create_adaptive_max_pool3d_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(adaptive_max_pool3d_backward::name, adaptive_max_pool3d_backward::overload_name)
      .typed<adaptive_max_pool3d_backward::schema>();
}

// aten::adaptive_max_pool3d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor
at::Tensor adaptive_max_pool3d_backward::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices) {
    static auto op = create_adaptive_max_pool3d_backward_typed_handle();
    return op.call(grad_output, self, indices);
}

// aten::adaptive_max_pool3d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor
at::Tensor adaptive_max_pool3d_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices) {
    static auto op = create_adaptive_max_pool3d_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, indices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fractional_max_pool2d_output, name, "aten::fractional_max_pool2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fractional_max_pool2d_output, overload_name, "output")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fractional_max_pool2d_output, schema_str, "fractional_max_pool2d.output(Tensor self, int[2] kernel_size, int[2] output_size, Tensor random_samples, *, Tensor(a!) output, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))")

// aten::fractional_max_pool2d.output(Tensor self, int[2] kernel_size, int[2] output_size, Tensor random_samples, *, Tensor(a!) output, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
static C10_NOINLINE c10::TypedOperatorHandle<fractional_max_pool2d_output::schema> create_fractional_max_pool2d_output_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fractional_max_pool2d_output::name, fractional_max_pool2d_output::overload_name)
      .typed<fractional_max_pool2d_output::schema>();
}

// aten::fractional_max_pool2d.output(Tensor self, int[2] kernel_size, int[2] output_size, Tensor random_samples, *, Tensor(a!) output, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
::std::tuple<at::Tensor &,at::Tensor &> fractional_max_pool2d_output::call(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & random_samples, at::Tensor & output, at::Tensor & indices) {
    static auto op = create_fractional_max_pool2d_output_typed_handle();
    return op.call(self, kernel_size, output_size, random_samples, output, indices);
}

// aten::fractional_max_pool2d.output(Tensor self, int[2] kernel_size, int[2] output_size, Tensor random_samples, *, Tensor(a!) output, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
::std::tuple<at::Tensor &,at::Tensor &> fractional_max_pool2d_output::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & random_samples, at::Tensor & output, at::Tensor & indices) {
    static auto op = create_fractional_max_pool2d_output_typed_handle();
    return op.redispatch(dispatchKeySet, self, kernel_size, output_size, random_samples, output, indices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fractional_max_pool3d, name, "aten::fractional_max_pool3d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fractional_max_pool3d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fractional_max_pool3d, schema_str, "fractional_max_pool3d(Tensor self, int[3] kernel_size, int[3] output_size, Tensor random_samples) -> (Tensor, Tensor)")

// aten::fractional_max_pool3d(Tensor self, int[3] kernel_size, int[3] output_size, Tensor random_samples) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<fractional_max_pool3d::schema> create_fractional_max_pool3d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fractional_max_pool3d::name, fractional_max_pool3d::overload_name)
      .typed<fractional_max_pool3d::schema>();
}

// aten::fractional_max_pool3d(Tensor self, int[3] kernel_size, int[3] output_size, Tensor random_samples) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> fractional_max_pool3d::call(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & random_samples) {
    static auto op = create_fractional_max_pool3d_typed_handle();
    return op.call(self, kernel_size, output_size, random_samples);
}

// aten::fractional_max_pool3d(Tensor self, int[3] kernel_size, int[3] output_size, Tensor random_samples) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> fractional_max_pool3d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & random_samples) {
    static auto op = create_fractional_max_pool3d_typed_handle();
    return op.redispatch(dispatchKeySet, self, kernel_size, output_size, random_samples);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_pool2d_with_indices_backward_grad_input, name, "aten::max_pool2d_with_indices_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_pool2d_with_indices_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_pool2d_with_indices_backward_grad_input, schema_str, "max_pool2d_with_indices_backward.grad_input(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::max_pool2d_with_indices_backward.grad_input(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<max_pool2d_with_indices_backward_grad_input::schema> create_max_pool2d_with_indices_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(max_pool2d_with_indices_backward_grad_input::name, max_pool2d_with_indices_backward_grad_input::overload_name)
      .typed<max_pool2d_with_indices_backward_grad_input::schema>();
}

// aten::max_pool2d_with_indices_backward.grad_input(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & max_pool2d_with_indices_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices, at::Tensor & grad_input) {
    static auto op = create_max_pool2d_with_indices_backward_grad_input_typed_handle();
    return op.call(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices, grad_input);
}

// aten::max_pool2d_with_indices_backward.grad_input(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & max_pool2d_with_indices_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices, at::Tensor & grad_input) {
    static auto op = create_max_pool2d_with_indices_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_pool3d_with_indices_backward_grad_input, name, "aten::max_pool3d_with_indices_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_pool3d_with_indices_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_pool3d_with_indices_backward_grad_input, schema_str, "max_pool3d_with_indices_backward.grad_input(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool ceil_mode, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::max_pool3d_with_indices_backward.grad_input(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool ceil_mode, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<max_pool3d_with_indices_backward_grad_input::schema> create_max_pool3d_with_indices_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(max_pool3d_with_indices_backward_grad_input::name, max_pool3d_with_indices_backward_grad_input::overload_name)
      .typed<max_pool3d_with_indices_backward_grad_input::schema>();
}

// aten::max_pool3d_with_indices_backward.grad_input(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool ceil_mode, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & max_pool3d_with_indices_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices, at::Tensor & grad_input) {
    static auto op = create_max_pool3d_with_indices_backward_grad_input_typed_handle();
    return op.call(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices, grad_input);
}

// aten::max_pool3d_with_indices_backward.grad_input(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool ceil_mode, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & max_pool3d_with_indices_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices, at::Tensor & grad_input) {
    static auto op = create_max_pool3d_with_indices_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reflection_pad1d_backward_grad_input, name, "aten::reflection_pad1d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reflection_pad1d_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reflection_pad1d_backward_grad_input, schema_str, "reflection_pad1d_backward.grad_input(Tensor grad_output, Tensor self, int[2] padding, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::reflection_pad1d_backward.grad_input(Tensor grad_output, Tensor self, int[2] padding, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<reflection_pad1d_backward_grad_input::schema> create_reflection_pad1d_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(reflection_pad1d_backward_grad_input::name, reflection_pad1d_backward_grad_input::overload_name)
      .typed<reflection_pad1d_backward_grad_input::schema>();
}

// aten::reflection_pad1d_backward.grad_input(Tensor grad_output, Tensor self, int[2] padding, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & reflection_pad1d_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input) {
    static auto op = create_reflection_pad1d_backward_grad_input_typed_handle();
    return op.call(grad_output, self, padding, grad_input);
}

// aten::reflection_pad1d_backward.grad_input(Tensor grad_output, Tensor self, int[2] padding, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & reflection_pad1d_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input) {
    static auto op = create_reflection_pad1d_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, padding, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reflection_pad3d_out, name, "aten::reflection_pad3d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reflection_pad3d_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reflection_pad3d_out, schema_str, "reflection_pad3d.out(Tensor self, int[6] padding, *, Tensor(a!) out) -> Tensor(a!)")

// aten::reflection_pad3d.out(Tensor self, int[6] padding, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<reflection_pad3d_out::schema> create_reflection_pad3d_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(reflection_pad3d_out::name, reflection_pad3d_out::overload_name)
      .typed<reflection_pad3d_out::schema>();
}

// aten::reflection_pad3d.out(Tensor self, int[6] padding, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & reflection_pad3d_out::call(const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out) {
    static auto op = create_reflection_pad3d_out_typed_handle();
    return op.call(self, padding, out);
}

// aten::reflection_pad3d.out(Tensor self, int[6] padding, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & reflection_pad3d_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out) {
    static auto op = create_reflection_pad3d_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, padding, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reflection_pad3d, name, "aten::reflection_pad3d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reflection_pad3d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reflection_pad3d, schema_str, "reflection_pad3d(Tensor self, int[6] padding) -> Tensor")

// aten::reflection_pad3d(Tensor self, int[6] padding) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<reflection_pad3d::schema> create_reflection_pad3d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(reflection_pad3d::name, reflection_pad3d::overload_name)
      .typed<reflection_pad3d::schema>();
}

// aten::reflection_pad3d(Tensor self, int[6] padding) -> Tensor
at::Tensor reflection_pad3d::call(const at::Tensor & self, at::IntArrayRef padding) {
    static auto op = create_reflection_pad3d_typed_handle();
    return op.call(self, padding);
}

// aten::reflection_pad3d(Tensor self, int[6] padding) -> Tensor
at::Tensor reflection_pad3d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef padding) {
    static auto op = create_reflection_pad3d_typed_handle();
    return op.redispatch(dispatchKeySet, self, padding);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(replication_pad1d_out, name, "aten::replication_pad1d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(replication_pad1d_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(replication_pad1d_out, schema_str, "replication_pad1d.out(Tensor self, int[2] padding, *, Tensor(a!) out) -> Tensor(a!)")

// aten::replication_pad1d.out(Tensor self, int[2] padding, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<replication_pad1d_out::schema> create_replication_pad1d_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(replication_pad1d_out::name, replication_pad1d_out::overload_name)
      .typed<replication_pad1d_out::schema>();
}

// aten::replication_pad1d.out(Tensor self, int[2] padding, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & replication_pad1d_out::call(const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out) {
    static auto op = create_replication_pad1d_out_typed_handle();
    return op.call(self, padding, out);
}

// aten::replication_pad1d.out(Tensor self, int[2] padding, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & replication_pad1d_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out) {
    static auto op = create_replication_pad1d_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, padding, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(replication_pad1d, name, "aten::replication_pad1d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(replication_pad1d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(replication_pad1d, schema_str, "replication_pad1d(Tensor self, int[2] padding) -> Tensor")

// aten::replication_pad1d(Tensor self, int[2] padding) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<replication_pad1d::schema> create_replication_pad1d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(replication_pad1d::name, replication_pad1d::overload_name)
      .typed<replication_pad1d::schema>();
}

// aten::replication_pad1d(Tensor self, int[2] padding) -> Tensor
at::Tensor replication_pad1d::call(const at::Tensor & self, at::IntArrayRef padding) {
    static auto op = create_replication_pad1d_typed_handle();
    return op.call(self, padding);
}

// aten::replication_pad1d(Tensor self, int[2] padding) -> Tensor
at::Tensor replication_pad1d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef padding) {
    static auto op = create_replication_pad1d_typed_handle();
    return op.redispatch(dispatchKeySet, self, padding);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(replication_pad2d, name, "aten::replication_pad2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(replication_pad2d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(replication_pad2d, schema_str, "replication_pad2d(Tensor self, int[4] padding) -> Tensor")

// aten::replication_pad2d(Tensor self, int[4] padding) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<replication_pad2d::schema> create_replication_pad2d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(replication_pad2d::name, replication_pad2d::overload_name)
      .typed<replication_pad2d::schema>();
}

// aten::replication_pad2d(Tensor self, int[4] padding) -> Tensor
at::Tensor replication_pad2d::call(const at::Tensor & self, at::IntArrayRef padding) {
    static auto op = create_replication_pad2d_typed_handle();
    return op.call(self, padding);
}

// aten::replication_pad2d(Tensor self, int[4] padding) -> Tensor
at::Tensor replication_pad2d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef padding) {
    static auto op = create_replication_pad2d_typed_handle();
    return op.redispatch(dispatchKeySet, self, padding);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(replication_pad3d_out, name, "aten::replication_pad3d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(replication_pad3d_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(replication_pad3d_out, schema_str, "replication_pad3d.out(Tensor self, int[6] padding, *, Tensor(a!) out) -> Tensor(a!)")

// aten::replication_pad3d.out(Tensor self, int[6] padding, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<replication_pad3d_out::schema> create_replication_pad3d_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(replication_pad3d_out::name, replication_pad3d_out::overload_name)
      .typed<replication_pad3d_out::schema>();
}

// aten::replication_pad3d.out(Tensor self, int[6] padding, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & replication_pad3d_out::call(const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out) {
    static auto op = create_replication_pad3d_out_typed_handle();
    return op.call(self, padding, out);
}

// aten::replication_pad3d.out(Tensor self, int[6] padding, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & replication_pad3d_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out) {
    static auto op = create_replication_pad3d_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, padding, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_linear1d_vec, name, "aten::upsample_linear1d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_linear1d_vec, overload_name, "vec")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_linear1d_vec, schema_str, "upsample_linear1d.vec(Tensor input, int[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor")

// aten::upsample_linear1d.vec(Tensor input, int[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<upsample_linear1d_vec::schema> create_upsample_linear1d_vec_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_linear1d_vec::name, upsample_linear1d_vec::overload_name)
      .typed<upsample_linear1d_vec::schema>();
}

// aten::upsample_linear1d.vec(Tensor input, int[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
at::Tensor upsample_linear1d_vec::call(const at::Tensor & input, c10::optional<at::IntArrayRef> output_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
    static auto op = create_upsample_linear1d_vec_typed_handle();
    return op.call(input, output_size, align_corners, scale_factors);
}

// aten::upsample_linear1d.vec(Tensor input, int[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
at::Tensor upsample_linear1d_vec::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, c10::optional<at::IntArrayRef> output_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
    static auto op = create_upsample_linear1d_vec_typed_handle();
    return op.redispatch(dispatchKeySet, input, output_size, align_corners, scale_factors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_bilinear2d_vec, name, "aten::upsample_bilinear2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_bilinear2d_vec, overload_name, "vec")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_bilinear2d_vec, schema_str, "upsample_bilinear2d.vec(Tensor input, int[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor")

// aten::upsample_bilinear2d.vec(Tensor input, int[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<upsample_bilinear2d_vec::schema> create_upsample_bilinear2d_vec_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_bilinear2d_vec::name, upsample_bilinear2d_vec::overload_name)
      .typed<upsample_bilinear2d_vec::schema>();
}

// aten::upsample_bilinear2d.vec(Tensor input, int[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
at::Tensor upsample_bilinear2d_vec::call(const at::Tensor & input, c10::optional<at::IntArrayRef> output_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
    static auto op = create_upsample_bilinear2d_vec_typed_handle();
    return op.call(input, output_size, align_corners, scale_factors);
}

// aten::upsample_bilinear2d.vec(Tensor input, int[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
at::Tensor upsample_bilinear2d_vec::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, c10::optional<at::IntArrayRef> output_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
    static auto op = create_upsample_bilinear2d_vec_typed_handle();
    return op.redispatch(dispatchKeySet, input, output_size, align_corners, scale_factors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_trilinear3d_backward_vec, name, "aten::upsample_trilinear3d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_trilinear3d_backward_vec, overload_name, "vec")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_trilinear3d_backward_vec, schema_str, "upsample_trilinear3d_backward.vec(Tensor grad_output, int[]? output_size, int[] input_size, bool align_corners, float[]? scale_factors) -> Tensor")

// aten::upsample_trilinear3d_backward.vec(Tensor grad_output, int[]? output_size, int[] input_size, bool align_corners, float[]? scale_factors) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<upsample_trilinear3d_backward_vec::schema> create_upsample_trilinear3d_backward_vec_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_trilinear3d_backward_vec::name, upsample_trilinear3d_backward_vec::overload_name)
      .typed<upsample_trilinear3d_backward_vec::schema>();
}

// aten::upsample_trilinear3d_backward.vec(Tensor grad_output, int[]? output_size, int[] input_size, bool align_corners, float[]? scale_factors) -> Tensor
at::Tensor upsample_trilinear3d_backward_vec::call(const at::Tensor & grad_output, c10::optional<at::IntArrayRef> output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
    static auto op = create_upsample_trilinear3d_backward_vec_typed_handle();
    return op.call(grad_output, output_size, input_size, align_corners, scale_factors);
}

// aten::upsample_trilinear3d_backward.vec(Tensor grad_output, int[]? output_size, int[] input_size, bool align_corners, float[]? scale_factors) -> Tensor
at::Tensor upsample_trilinear3d_backward_vec::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, c10::optional<at::IntArrayRef> output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
    static auto op = create_upsample_trilinear3d_backward_vec_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, output_size, input_size, align_corners, scale_factors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest1d_vec, name, "aten::upsample_nearest1d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest1d_vec, overload_name, "vec")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest1d_vec, schema_str, "upsample_nearest1d.vec(Tensor input, int[]? output_size, float[]? scale_factors) -> Tensor")

// aten::upsample_nearest1d.vec(Tensor input, int[]? output_size, float[]? scale_factors) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<upsample_nearest1d_vec::schema> create_upsample_nearest1d_vec_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_nearest1d_vec::name, upsample_nearest1d_vec::overload_name)
      .typed<upsample_nearest1d_vec::schema>();
}

// aten::upsample_nearest1d.vec(Tensor input, int[]? output_size, float[]? scale_factors) -> Tensor
at::Tensor upsample_nearest1d_vec::call(const at::Tensor & input, c10::optional<at::IntArrayRef> output_size, c10::optional<at::ArrayRef<double>> scale_factors) {
    static auto op = create_upsample_nearest1d_vec_typed_handle();
    return op.call(input, output_size, scale_factors);
}

// aten::upsample_nearest1d.vec(Tensor input, int[]? output_size, float[]? scale_factors) -> Tensor
at::Tensor upsample_nearest1d_vec::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, c10::optional<at::IntArrayRef> output_size, c10::optional<at::ArrayRef<double>> scale_factors) {
    static auto op = create_upsample_nearest1d_vec_typed_handle();
    return op.redispatch(dispatchKeySet, input, output_size, scale_factors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest1d_backward_vec, name, "aten::upsample_nearest1d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest1d_backward_vec, overload_name, "vec")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest1d_backward_vec, schema_str, "upsample_nearest1d_backward.vec(Tensor grad_output, int[]? output_size, int[] input_size, float[]? scale_factors) -> Tensor")

// aten::upsample_nearest1d_backward.vec(Tensor grad_output, int[]? output_size, int[] input_size, float[]? scale_factors) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<upsample_nearest1d_backward_vec::schema> create_upsample_nearest1d_backward_vec_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_nearest1d_backward_vec::name, upsample_nearest1d_backward_vec::overload_name)
      .typed<upsample_nearest1d_backward_vec::schema>();
}

// aten::upsample_nearest1d_backward.vec(Tensor grad_output, int[]? output_size, int[] input_size, float[]? scale_factors) -> Tensor
at::Tensor upsample_nearest1d_backward_vec::call(const at::Tensor & grad_output, c10::optional<at::IntArrayRef> output_size, at::IntArrayRef input_size, c10::optional<at::ArrayRef<double>> scale_factors) {
    static auto op = create_upsample_nearest1d_backward_vec_typed_handle();
    return op.call(grad_output, output_size, input_size, scale_factors);
}

// aten::upsample_nearest1d_backward.vec(Tensor grad_output, int[]? output_size, int[] input_size, float[]? scale_factors) -> Tensor
at::Tensor upsample_nearest1d_backward_vec::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, c10::optional<at::IntArrayRef> output_size, at::IntArrayRef input_size, c10::optional<at::ArrayRef<double>> scale_factors) {
    static auto op = create_upsample_nearest1d_backward_vec_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, output_size, input_size, scale_factors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest3d_vec, name, "aten::upsample_nearest3d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest3d_vec, overload_name, "vec")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest3d_vec, schema_str, "upsample_nearest3d.vec(Tensor input, int[]? output_size, float[]? scale_factors) -> Tensor")

// aten::upsample_nearest3d.vec(Tensor input, int[]? output_size, float[]? scale_factors) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<upsample_nearest3d_vec::schema> create_upsample_nearest3d_vec_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_nearest3d_vec::name, upsample_nearest3d_vec::overload_name)
      .typed<upsample_nearest3d_vec::schema>();
}

// aten::upsample_nearest3d.vec(Tensor input, int[]? output_size, float[]? scale_factors) -> Tensor
at::Tensor upsample_nearest3d_vec::call(const at::Tensor & input, c10::optional<at::IntArrayRef> output_size, c10::optional<at::ArrayRef<double>> scale_factors) {
    static auto op = create_upsample_nearest3d_vec_typed_handle();
    return op.call(input, output_size, scale_factors);
}

// aten::upsample_nearest3d.vec(Tensor input, int[]? output_size, float[]? scale_factors) -> Tensor
at::Tensor upsample_nearest3d_vec::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, c10::optional<at::IntArrayRef> output_size, c10::optional<at::ArrayRef<double>> scale_factors) {
    static auto op = create_upsample_nearest3d_vec_typed_handle();
    return op.redispatch(dispatchKeySet, input, output_size, scale_factors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest1d_out, name, "aten::upsample_nearest1d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest1d_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest1d_out, schema_str, "upsample_nearest1d.out(Tensor self, int[1] output_size, float? scales=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::upsample_nearest1d.out(Tensor self, int[1] output_size, float? scales=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<upsample_nearest1d_out::schema> create_upsample_nearest1d_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_nearest1d_out::name, upsample_nearest1d_out::overload_name)
      .typed<upsample_nearest1d_out::schema>();
}

// aten::upsample_nearest1d.out(Tensor self, int[1] output_size, float? scales=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & upsample_nearest1d_out::call(const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales, at::Tensor & out) {
    static auto op = create_upsample_nearest1d_out_typed_handle();
    return op.call(self, output_size, scales, out);
}

// aten::upsample_nearest1d.out(Tensor self, int[1] output_size, float? scales=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & upsample_nearest1d_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales, at::Tensor & out) {
    static auto op = create_upsample_nearest1d_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, output_size, scales, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest1d, name, "aten::upsample_nearest1d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest1d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest1d, schema_str, "upsample_nearest1d(Tensor self, int[1] output_size, float? scales=None) -> Tensor")

// aten::upsample_nearest1d(Tensor self, int[1] output_size, float? scales=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<upsample_nearest1d::schema> create_upsample_nearest1d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_nearest1d::name, upsample_nearest1d::overload_name)
      .typed<upsample_nearest1d::schema>();
}

// aten::upsample_nearest1d(Tensor self, int[1] output_size, float? scales=None) -> Tensor
at::Tensor upsample_nearest1d::call(const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales) {
    static auto op = create_upsample_nearest1d_typed_handle();
    return op.call(self, output_size, scales);
}

// aten::upsample_nearest1d(Tensor self, int[1] output_size, float? scales=None) -> Tensor
at::Tensor upsample_nearest1d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales) {
    static auto op = create_upsample_nearest1d_typed_handle();
    return op.redispatch(dispatchKeySet, self, output_size, scales);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logit_backward_grad_input, name, "aten::logit_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logit_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logit_backward_grad_input, schema_str, "logit_backward.grad_input(Tensor grad_output, Tensor self, float? eps=None, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::logit_backward.grad_input(Tensor grad_output, Tensor self, float? eps=None, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<logit_backward_grad_input::schema> create_logit_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(logit_backward_grad_input::name, logit_backward_grad_input::overload_name)
      .typed<logit_backward_grad_input::schema>();
}

// aten::logit_backward.grad_input(Tensor grad_output, Tensor self, float? eps=None, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & logit_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, c10::optional<double> eps, at::Tensor & grad_input) {
    static auto op = create_logit_backward_grad_input_typed_handle();
    return op.call(grad_output, self, eps, grad_input);
}

// aten::logit_backward.grad_input(Tensor grad_output, Tensor self, float? eps=None, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & logit_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, c10::optional<double> eps, at::Tensor & grad_input) {
    static auto op = create_logit_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, eps, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv_transpose2d_out, name, "aten::slow_conv_transpose2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv_transpose2d_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv_transpose2d_out, schema_str, "slow_conv_transpose2d.out(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] output_padding=0, int[2] dilation=1, *, Tensor(a!) out) -> Tensor(a!)")

// aten::slow_conv_transpose2d.out(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] output_padding=0, int[2] dilation=1, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<slow_conv_transpose2d_out::schema> create_slow_conv_transpose2d_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(slow_conv_transpose2d_out::name, slow_conv_transpose2d_out::overload_name)
      .typed<slow_conv_transpose2d_out::schema>();
}

// aten::slow_conv_transpose2d.out(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] output_padding=0, int[2] dilation=1, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & slow_conv_transpose2d_out::call(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, at::Tensor & out) {
    static auto op = create_slow_conv_transpose2d_out_typed_handle();
    return op.call(self, weight, kernel_size, bias, stride, padding, output_padding, dilation, out);
}

// aten::slow_conv_transpose2d.out(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] output_padding=0, int[2] dilation=1, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & slow_conv_transpose2d_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, at::Tensor & out) {
    static auto op = create_slow_conv_transpose2d_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, weight, kernel_size, bias, stride, padding, output_padding, dilation, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv_transpose2d_backward_grad_output, name, "aten::slow_conv_transpose2d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv_transpose2d_backward_grad_output, overload_name, "grad_output")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv_transpose2d_backward_grad_output, schema_str, "slow_conv_transpose2d_backward.grad_output(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] output_padding, int[2] dilation, Tensor columns, Tensor ones, *, Tensor(a!) grad_input, Tensor(b!) grad_weight, Tensor(c!) grad_bias) -> (Tensor(a!), Tensor(b!), Tensor(c!))")

// aten::slow_conv_transpose2d_backward.grad_output(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] output_padding, int[2] dilation, Tensor columns, Tensor ones, *, Tensor(a!) grad_input, Tensor(b!) grad_weight, Tensor(c!) grad_bias) -> (Tensor(a!), Tensor(b!), Tensor(c!))
static C10_NOINLINE c10::TypedOperatorHandle<slow_conv_transpose2d_backward_grad_output::schema> create_slow_conv_transpose2d_backward_grad_output_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(slow_conv_transpose2d_backward_grad_output::name, slow_conv_transpose2d_backward_grad_output::overload_name)
      .typed<slow_conv_transpose2d_backward_grad_output::schema>();
}

// aten::slow_conv_transpose2d_backward.grad_output(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] output_padding, int[2] dilation, Tensor columns, Tensor ones, *, Tensor(a!) grad_input, Tensor(b!) grad_weight, Tensor(c!) grad_bias) -> (Tensor(a!), Tensor(b!), Tensor(c!))
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> slow_conv_transpose2d_backward_grad_output::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, const at::Tensor & columns, const at::Tensor & ones, at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias) {
    static auto op = create_slow_conv_transpose2d_backward_grad_output_typed_handle();
    return op.call(grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, columns, ones, grad_input, grad_weight, grad_bias);
}

// aten::slow_conv_transpose2d_backward.grad_output(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] output_padding, int[2] dilation, Tensor columns, Tensor ones, *, Tensor(a!) grad_input, Tensor(b!) grad_weight, Tensor(c!) grad_bias) -> (Tensor(a!), Tensor(b!), Tensor(c!))
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> slow_conv_transpose2d_backward_grad_output::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, const at::Tensor & columns, const at::Tensor & ones, at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias) {
    static auto op = create_slow_conv_transpose2d_backward_grad_output_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, columns, ones, grad_input, grad_weight, grad_bias);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv_transpose3d_backward_output_mask, name, "aten::slow_conv_transpose3d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv_transpose3d_backward_output_mask, overload_name, "output_mask")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv_transpose3d_backward_output_mask, schema_str, "slow_conv_transpose3d_backward.output_mask(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, int[3] output_padding, int[3] dilation, Tensor finput, Tensor fgrad_input, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)")

// aten::slow_conv_transpose3d_backward.output_mask(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, int[3] output_padding, int[3] dilation, Tensor finput, Tensor fgrad_input, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)
static C10_NOINLINE c10::TypedOperatorHandle<slow_conv_transpose3d_backward_output_mask::schema> create_slow_conv_transpose3d_backward_output_mask_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(slow_conv_transpose3d_backward_output_mask::name, slow_conv_transpose3d_backward_output_mask::overload_name)
      .typed<slow_conv_transpose3d_backward_output_mask::schema>();
}

// aten::slow_conv_transpose3d_backward.output_mask(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, int[3] output_padding, int[3] dilation, Tensor finput, Tensor fgrad_input, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> slow_conv_transpose3d_backward_output_mask::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, const at::Tensor & finput, const at::Tensor & fgrad_input, ::std::array<bool,3> output_mask) {
    static auto op = create_slow_conv_transpose3d_backward_output_mask_typed_handle();
    return op.call(grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, finput, fgrad_input, output_mask);
}

// aten::slow_conv_transpose3d_backward.output_mask(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, int[3] output_padding, int[3] dilation, Tensor finput, Tensor fgrad_input, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> slow_conv_transpose3d_backward_output_mask::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, const at::Tensor & finput, const at::Tensor & fgrad_input, ::std::array<bool,3> output_mask) {
    static auto op = create_slow_conv_transpose3d_backward_output_mask_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, finput, fgrad_input, output_mask);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(thnn_conv2d_forward, name, "aten::thnn_conv2d_forward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(thnn_conv2d_forward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(thnn_conv2d_forward, schema_str, "thnn_conv2d_forward(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding) -> (Tensor output, Tensor finput, Tensor fgrad_input)")

// aten::thnn_conv2d_forward(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding) -> (Tensor output, Tensor finput, Tensor fgrad_input)
static C10_NOINLINE c10::TypedOperatorHandle<thnn_conv2d_forward::schema> create_thnn_conv2d_forward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(thnn_conv2d_forward::name, thnn_conv2d_forward::overload_name)
      .typed<thnn_conv2d_forward::schema>();
}

// aten::thnn_conv2d_forward(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding) -> (Tensor output, Tensor finput, Tensor fgrad_input)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> thnn_conv2d_forward::call(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding) {
    static auto op = create_thnn_conv2d_forward_typed_handle();
    return op.call(self, weight, kernel_size, bias, stride, padding);
}

// aten::thnn_conv2d_forward(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding) -> (Tensor output, Tensor finput, Tensor fgrad_input)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> thnn_conv2d_forward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding) {
    static auto op = create_thnn_conv2d_forward_typed_handle();
    return op.redispatch(dispatchKeySet, self, weight, kernel_size, bias, stride, padding);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_conv_depthwise2d, name, "aten::_conv_depthwise2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_conv_depthwise2d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_conv_depthwise2d, schema_str, "_conv_depthwise2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding, int[2] dilation) -> Tensor")

// aten::_conv_depthwise2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding, int[2] dilation) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_conv_depthwise2d::schema> create__conv_depthwise2d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_conv_depthwise2d::name, _conv_depthwise2d::overload_name)
      .typed<_conv_depthwise2d::schema>();
}

// aten::_conv_depthwise2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding, int[2] dilation) -> Tensor
at::Tensor _conv_depthwise2d::call(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation) {
    static auto op = create__conv_depthwise2d_typed_handle();
    return op.call(self, weight, kernel_size, bias, stride, padding, dilation);
}

// aten::_conv_depthwise2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding, int[2] dilation) -> Tensor
at::Tensor _conv_depthwise2d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation) {
    static auto op = create__conv_depthwise2d_typed_handle();
    return op.redispatch(dispatchKeySet, self, weight, kernel_size, bias, stride, padding, dilation);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv3d_out, name, "aten::slow_conv3d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv3d_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv3d_out, schema_str, "slow_conv3d.out(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0, *, Tensor(a!) out) -> Tensor(a!)")

// aten::slow_conv3d.out(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<slow_conv3d_out::schema> create_slow_conv3d_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(slow_conv3d_out::name, slow_conv3d_out::overload_name)
      .typed<slow_conv3d_out::schema>();
}

// aten::slow_conv3d.out(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & slow_conv3d_out::call(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & out) {
    static auto op = create_slow_conv3d_out_typed_handle();
    return op.call(self, weight, kernel_size, bias, stride, padding, out);
}

// aten::slow_conv3d.out(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & slow_conv3d_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & out) {
    static auto op = create_slow_conv3d_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, weight, kernel_size, bias, stride, padding, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv3d, name, "aten::slow_conv3d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv3d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv3d, schema_str, "slow_conv3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0) -> Tensor")

// aten::slow_conv3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<slow_conv3d::schema> create_slow_conv3d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(slow_conv3d::name, slow_conv3d::overload_name)
      .typed<slow_conv3d::schema>();
}

// aten::slow_conv3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0) -> Tensor
at::Tensor slow_conv3d::call(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding) {
    static auto op = create_slow_conv3d_typed_handle();
    return op.call(self, weight, kernel_size, bias, stride, padding);
}

// aten::slow_conv3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0) -> Tensor
at::Tensor slow_conv3d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding) {
    static auto op = create_slow_conv3d_typed_handle();
    return op.redispatch(dispatchKeySet, self, weight, kernel_size, bias, stride, padding);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv3d_backward_output_mask, name, "aten::slow_conv3d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv3d_backward_output_mask, overload_name, "output_mask")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv3d_backward_output_mask, schema_str, "slow_conv3d_backward.output_mask(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, Tensor finput, Tensor fgrad_input, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)")

// aten::slow_conv3d_backward.output_mask(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, Tensor finput, Tensor fgrad_input, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)
static C10_NOINLINE c10::TypedOperatorHandle<slow_conv3d_backward_output_mask::schema> create_slow_conv3d_backward_output_mask_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(slow_conv3d_backward_output_mask::name, slow_conv3d_backward_output_mask::overload_name)
      .typed<slow_conv3d_backward_output_mask::schema>();
}

// aten::slow_conv3d_backward.output_mask(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, Tensor finput, Tensor fgrad_input, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> slow_conv3d_backward_output_mask::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, const at::Tensor & finput, const at::Tensor & fgrad_input, ::std::array<bool,3> output_mask) {
    static auto op = create_slow_conv3d_backward_output_mask_typed_handle();
    return op.call(grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, output_mask);
}

// aten::slow_conv3d_backward.output_mask(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, Tensor finput, Tensor fgrad_input, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> slow_conv3d_backward_output_mask::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, const at::Tensor & finput, const at::Tensor & fgrad_input, ::std::array<bool,3> output_mask) {
    static auto op = create_slow_conv3d_backward_output_mask_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, output_mask);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_remove_batch_dim, name, "aten::_remove_batch_dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_remove_batch_dim, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_remove_batch_dim, schema_str, "_remove_batch_dim(Tensor self, int level, int batch_size, int out_dim) -> Tensor")

// aten::_remove_batch_dim(Tensor self, int level, int batch_size, int out_dim) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_remove_batch_dim::schema> create__remove_batch_dim_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_remove_batch_dim::name, _remove_batch_dim::overload_name)
      .typed<_remove_batch_dim::schema>();
}

// aten::_remove_batch_dim(Tensor self, int level, int batch_size, int out_dim) -> Tensor
at::Tensor _remove_batch_dim::call(const at::Tensor & self, int64_t level, int64_t batch_size, int64_t out_dim) {
    static auto op = create__remove_batch_dim_typed_handle();
    return op.call(self, level, batch_size, out_dim);
}

// aten::_remove_batch_dim(Tensor self, int level, int batch_size, int out_dim) -> Tensor
at::Tensor _remove_batch_dim::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t level, int64_t batch_size, int64_t out_dim) {
    static auto op = create__remove_batch_dim_typed_handle();
    return op.redispatch(dispatchKeySet, self, level, batch_size, out_dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_exp2_out, name, "aten::special_exp2")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_exp2_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_exp2_out, schema_str, "special_exp2.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::special_exp2.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<special_exp2_out::schema> create_special_exp2_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_exp2_out::name, special_exp2_out::overload_name)
      .typed<special_exp2_out::schema>();
}

// aten::special_exp2.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_exp2_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_exp2_out_typed_handle();
    return op.call(self, out);
}

// aten::special_exp2.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_exp2_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_exp2_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_erf, name, "aten::special_erf")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_erf, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_erf, schema_str, "special_erf(Tensor self) -> Tensor")

// aten::special_erf(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<special_erf::schema> create_special_erf_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_erf::name, special_erf::overload_name)
      .typed<special_erf::schema>();
}

// aten::special_erf(Tensor self) -> Tensor
at::Tensor special_erf::call(const at::Tensor & self) {
    static auto op = create_special_erf_typed_handle();
    return op.call(self);
}

// aten::special_erf(Tensor self) -> Tensor
at::Tensor special_erf::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_special_erf_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_erfc_out, name, "aten::special_erfc")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_erfc_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_erfc_out, schema_str, "special_erfc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::special_erfc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<special_erfc_out::schema> create_special_erfc_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_erfc_out::name, special_erfc_out::overload_name)
      .typed<special_erfc_out::schema>();
}

// aten::special_erfc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_erfc_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_erfc_out_typed_handle();
    return op.call(self, out);
}

// aten::special_erfc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_erfc_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_erfc_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_xlog1py_self_scalar_out, name, "aten::special_xlog1py")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_xlog1py_self_scalar_out, overload_name, "self_scalar_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_xlog1py_self_scalar_out, schema_str, "special_xlog1py.self_scalar_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::special_xlog1py.self_scalar_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<special_xlog1py_self_scalar_out::schema> create_special_xlog1py_self_scalar_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_xlog1py_self_scalar_out::name, special_xlog1py_self_scalar_out::overload_name)
      .typed<special_xlog1py_self_scalar_out::schema>();
}

// aten::special_xlog1py.self_scalar_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_xlog1py_self_scalar_out::call(const at::Scalar & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_special_xlog1py_self_scalar_out_typed_handle();
    return op.call(self, other, out);
}

// aten::special_xlog1py.self_scalar_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_xlog1py_self_scalar_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Scalar & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_special_xlog1py_self_scalar_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_xlogy, name, "aten::special_xlogy")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_xlogy, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_xlogy, schema_str, "special_xlogy(Tensor self, Tensor other) -> Tensor")

// aten::special_xlogy(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<special_xlogy::schema> create_special_xlogy_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_xlogy::name, special_xlogy::overload_name)
      .typed<special_xlogy::schema>();
}

// aten::special_xlogy(Tensor self, Tensor other) -> Tensor
at::Tensor special_xlogy::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_special_xlogy_typed_handle();
    return op.call(self, other);
}

// aten::special_xlogy(Tensor self, Tensor other) -> Tensor
at::Tensor special_xlogy::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_special_xlogy_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_expit, name, "aten::special_expit")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_expit, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_expit, schema_str, "special_expit(Tensor self) -> Tensor")

// aten::special_expit(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<special_expit::schema> create_special_expit_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_expit::name, special_expit::overload_name)
      .typed<special_expit::schema>();
}

// aten::special_expit(Tensor self) -> Tensor
at::Tensor special_expit::call(const at::Tensor & self) {
    static auto op = create_special_expit_typed_handle();
    return op.call(self);
}

// aten::special_expit(Tensor self) -> Tensor
at::Tensor special_expit::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_special_expit_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_sinc, name, "aten::special_sinc")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_sinc, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_sinc, schema_str, "special_sinc(Tensor self) -> Tensor")

// aten::special_sinc(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<special_sinc::schema> create_special_sinc_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_sinc::name, special_sinc::overload_name)
      .typed<special_sinc::schema>();
}

// aten::special_sinc(Tensor self) -> Tensor
at::Tensor special_sinc::call(const at::Tensor & self) {
    static auto op = create_special_sinc_typed_handle();
    return op.call(self);
}

// aten::special_sinc(Tensor self) -> Tensor
at::Tensor special_sinc::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_special_sinc_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_round_out, name, "aten::special_round")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_round_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_round_out, schema_str, "special_round.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::special_round.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<special_round_out::schema> create_special_round_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_round_out::name, special_round_out::overload_name)
      .typed<special_round_out::schema>();
}

// aten::special_round.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_round_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_round_out_typed_handle();
    return op.call(self, out);
}

// aten::special_round.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_round_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_round_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_fft, name, "aten::fft_fft")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_fft, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_fft, schema_str, "fft_fft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> Tensor")

// aten::fft_fft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fft_fft::schema> create_fft_fft_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fft_fft::name, fft_fft::overload_name)
      .typed<fft_fft::schema>();
}

// aten::fft_fft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> Tensor
at::Tensor fft_fft::call(const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm) {
    static auto op = create_fft_fft_typed_handle();
    return op.call(self, n, dim, norm);
}

// aten::fft_fft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> Tensor
at::Tensor fft_fft::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm) {
    static auto op = create_fft_fft_typed_handle();
    return op.redispatch(dispatchKeySet, self, n, dim, norm);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_fft_out, name, "aten::fft_fft")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_fft_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_fft_out, schema_str, "fft_fft.out(Tensor self, int? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::fft_fft.out(Tensor self, int? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<fft_fft_out::schema> create_fft_fft_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fft_fft_out::name, fft_fft_out::overload_name)
      .typed<fft_fft_out::schema>();
}

// aten::fft_fft.out(Tensor self, int? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fft_fft_out::call(const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
    static auto op = create_fft_fft_out_typed_handle();
    return op.call(self, n, dim, norm, out);
}

// aten::fft_fft.out(Tensor self, int? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fft_fft_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
    static auto op = create_fft_fft_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, n, dim, norm, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_rfft, name, "aten::fft_rfft")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_rfft, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_rfft, schema_str, "fft_rfft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> Tensor")

// aten::fft_rfft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fft_rfft::schema> create_fft_rfft_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fft_rfft::name, fft_rfft::overload_name)
      .typed<fft_rfft::schema>();
}

// aten::fft_rfft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> Tensor
at::Tensor fft_rfft::call(const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm) {
    static auto op = create_fft_rfft_typed_handle();
    return op.call(self, n, dim, norm);
}

// aten::fft_rfft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> Tensor
at::Tensor fft_rfft::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm) {
    static auto op = create_fft_rfft_typed_handle();
    return op.redispatch(dispatchKeySet, self, n, dim, norm);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_fft2_out, name, "aten::fft_fft2")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_fft2_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_fft2_out, schema_str, "fft_fft2.out(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::fft_fft2.out(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<fft_fft2_out::schema> create_fft_fft2_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fft_fft2_out::name, fft_fft2_out::overload_name)
      .typed<fft_fft2_out::schema>();
}

// aten::fft_fft2.out(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fft_fft2_out::call(const at::Tensor & self, c10::optional<at::IntArrayRef> s, at::IntArrayRef dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
    static auto op = create_fft_fft2_out_typed_handle();
    return op.call(self, s, dim, norm, out);
}

// aten::fft_fft2.out(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fft_fft2_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> s, at::IntArrayRef dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
    static auto op = create_fft_fft2_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, s, dim, norm, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_fftn_out, name, "aten::fft_fftn")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_fftn_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_fftn_out, schema_str, "fft_fftn.out(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::fft_fftn.out(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<fft_fftn_out::schema> create_fft_fftn_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fft_fftn_out::name, fft_fftn_out::overload_name)
      .typed<fft_fftn_out::schema>();
}

// aten::fft_fftn.out(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fft_fftn_out::call(const at::Tensor & self, c10::optional<at::IntArrayRef> s, c10::optional<at::IntArrayRef> dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
    static auto op = create_fft_fftn_out_typed_handle();
    return op.call(self, s, dim, norm, out);
}

// aten::fft_fftn.out(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fft_fftn_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> s, c10::optional<at::IntArrayRef> dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
    static auto op = create_fft_fftn_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, s, dim, norm, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_ifftn, name, "aten::fft_ifftn")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_ifftn, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_ifftn, schema_str, "fft_ifftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor")

// aten::fft_ifftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fft_ifftn::schema> create_fft_ifftn_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fft_ifftn::name, fft_ifftn::overload_name)
      .typed<fft_ifftn::schema>();
}

// aten::fft_ifftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor
at::Tensor fft_ifftn::call(const at::Tensor & self, c10::optional<at::IntArrayRef> s, c10::optional<at::IntArrayRef> dim, c10::optional<c10::string_view> norm) {
    static auto op = create_fft_ifftn_typed_handle();
    return op.call(self, s, dim, norm);
}

// aten::fft_ifftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor
at::Tensor fft_ifftn::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> s, c10::optional<at::IntArrayRef> dim, c10::optional<c10::string_view> norm) {
    static auto op = create_fft_ifftn_typed_handle();
    return op.redispatch(dispatchKeySet, self, s, dim, norm);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_fftfreq, name, "aten::fft_fftfreq")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_fftfreq, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_fftfreq, schema_str, "fft_fftfreq(int n, float d=1.0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::fft_fftfreq(int n, float d=1.0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fft_fftfreq::schema> create_fft_fftfreq_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fft_fftfreq::name, fft_fftfreq::overload_name)
      .typed<fft_fftfreq::schema>();
}

// aten::fft_fftfreq(int n, float d=1.0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor fft_fftfreq::call(int64_t n, double d, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_fft_fftfreq_typed_handle();
    return op.call(n, d, dtype, layout, device, pin_memory);
}

// aten::fft_fftfreq(int n, float d=1.0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor fft_fftfreq::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t n, double d, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_fft_fftfreq_typed_handle();
    return op.redispatch(dispatchKeySet, n, d, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_rfftfreq, name, "aten::fft_rfftfreq")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_rfftfreq, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_rfftfreq, schema_str, "fft_rfftfreq(int n, float d=1.0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::fft_rfftfreq(int n, float d=1.0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fft_rfftfreq::schema> create_fft_rfftfreq_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fft_rfftfreq::name, fft_rfftfreq::overload_name)
      .typed<fft_rfftfreq::schema>();
}

// aten::fft_rfftfreq(int n, float d=1.0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor fft_rfftfreq::call(int64_t n, double d, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_fft_rfftfreq_typed_handle();
    return op.call(n, d, dtype, layout, device, pin_memory);
}

// aten::fft_rfftfreq(int n, float d=1.0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor fft_rfftfreq::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t n, double d, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_fft_rfftfreq_typed_handle();
    return op.redispatch(dispatchKeySet, n, d, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_cholesky_ex, name, "aten::linalg_cholesky_ex")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_cholesky_ex, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_cholesky_ex, schema_str, "linalg_cholesky_ex(Tensor self, *, bool upper=False, bool check_errors=False) -> (Tensor L, Tensor info)")

// aten::linalg_cholesky_ex(Tensor self, *, bool upper=False, bool check_errors=False) -> (Tensor L, Tensor info)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_cholesky_ex::schema> create_linalg_cholesky_ex_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_cholesky_ex::name, linalg_cholesky_ex::overload_name)
      .typed<linalg_cholesky_ex::schema>();
}

// aten::linalg_cholesky_ex(Tensor self, *, bool upper=False, bool check_errors=False) -> (Tensor L, Tensor info)
::std::tuple<at::Tensor,at::Tensor> linalg_cholesky_ex::call(const at::Tensor & self, bool upper, bool check_errors) {
    static auto op = create_linalg_cholesky_ex_typed_handle();
    return op.call(self, upper, check_errors);
}

// aten::linalg_cholesky_ex(Tensor self, *, bool upper=False, bool check_errors=False) -> (Tensor L, Tensor info)
::std::tuple<at::Tensor,at::Tensor> linalg_cholesky_ex::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool upper, bool check_errors) {
    static auto op = create_linalg_cholesky_ex_typed_handle();
    return op.redispatch(dispatchKeySet, self, upper, check_errors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_cholesky_out, name, "aten::linalg_cholesky")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_cholesky_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_cholesky_out, schema_str, "linalg_cholesky.out(Tensor self, *, bool upper=False, Tensor(a!) out) -> Tensor(a!)")

// aten::linalg_cholesky.out(Tensor self, *, bool upper=False, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_cholesky_out::schema> create_linalg_cholesky_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_cholesky_out::name, linalg_cholesky_out::overload_name)
      .typed<linalg_cholesky_out::schema>();
}

// aten::linalg_cholesky.out(Tensor self, *, bool upper=False, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_cholesky_out::call(const at::Tensor & self, bool upper, at::Tensor & out) {
    static auto op = create_linalg_cholesky_out_typed_handle();
    return op.call(self, upper, out);
}

// aten::linalg_cholesky.out(Tensor self, *, bool upper=False, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_cholesky_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool upper, at::Tensor & out) {
    static auto op = create_linalg_cholesky_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, upper, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_det_out, name, "aten::linalg_det")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_det_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_det_out, schema_str, "linalg_det.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::linalg_det.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_det_out::schema> create_linalg_det_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_det_out::name, linalg_det_out::overload_name)
      .typed<linalg_det_out::schema>();
}

// aten::linalg_det.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_det_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_linalg_det_out_typed_handle();
    return op.call(self, out);
}

// aten::linalg_det.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_det_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_linalg_det_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(det, name, "aten::det")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(det, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(det, schema_str, "det(Tensor self) -> Tensor")

// aten::det(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<det::schema> create_det_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(det::name, det::overload_name)
      .typed<det::schema>();
}

// aten::det(Tensor self) -> Tensor
at::Tensor det::call(const at::Tensor & self) {
    static auto op = create_det_typed_handle();
    return op.call(self);
}

// aten::det(Tensor self) -> Tensor
at::Tensor det::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_det_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_eigh_eigvals, name, "aten::linalg_eigh")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_eigh_eigvals, overload_name, "eigvals")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_eigh_eigvals, schema_str, "linalg_eigh.eigvals(Tensor self, str UPLO=\"L\", *, Tensor(a!) eigvals, Tensor(b!) eigvecs) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)")

// aten::linalg_eigh.eigvals(Tensor self, str UPLO="L", *, Tensor(a!) eigvals, Tensor(b!) eigvecs) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_eigh_eigvals::schema> create_linalg_eigh_eigvals_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_eigh_eigvals::name, linalg_eigh_eigvals::overload_name)
      .typed<linalg_eigh_eigvals::schema>();
}

// aten::linalg_eigh.eigvals(Tensor self, str UPLO="L", *, Tensor(a!) eigvals, Tensor(b!) eigvecs) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)
::std::tuple<at::Tensor &,at::Tensor &> linalg_eigh_eigvals::call(const at::Tensor & self, c10::string_view UPLO, at::Tensor & eigvals, at::Tensor & eigvecs) {
    static auto op = create_linalg_eigh_eigvals_typed_handle();
    return op.call(self, UPLO, eigvals, eigvecs);
}

// aten::linalg_eigh.eigvals(Tensor self, str UPLO="L", *, Tensor(a!) eigvals, Tensor(b!) eigvecs) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)
::std::tuple<at::Tensor &,at::Tensor &> linalg_eigh_eigvals::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::string_view UPLO, at::Tensor & eigvals, at::Tensor & eigvecs) {
    static auto op = create_linalg_eigh_eigvals_typed_handle();
    return op.redispatch(dispatchKeySet, self, UPLO, eigvals, eigvecs);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_householder_product_out, name, "aten::linalg_householder_product")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_householder_product_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_householder_product_out, schema_str, "linalg_householder_product.out(Tensor input, Tensor tau, *, Tensor(a!) out) -> Tensor(a!)")

// aten::linalg_householder_product.out(Tensor input, Tensor tau, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_householder_product_out::schema> create_linalg_householder_product_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_householder_product_out::name, linalg_householder_product_out::overload_name)
      .typed<linalg_householder_product_out::schema>();
}

// aten::linalg_householder_product.out(Tensor input, Tensor tau, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_householder_product_out::call(const at::Tensor & input, const at::Tensor & tau, at::Tensor & out) {
    static auto op = create_linalg_householder_product_out_typed_handle();
    return op.call(input, tau, out);
}

// aten::linalg_householder_product.out(Tensor input, Tensor tau, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_householder_product_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & tau, at::Tensor & out) {
    static auto op = create_linalg_householder_product_out_typed_handle();
    return op.redispatch(dispatchKeySet, input, tau, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_norm_out, name, "aten::linalg_norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_norm_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_norm_out, schema_str, "linalg_norm.out(Tensor self, Scalar? ord=None, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)")

// aten::linalg_norm.out(Tensor self, Scalar? ord=None, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_norm_out::schema> create_linalg_norm_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_norm_out::name, linalg_norm_out::overload_name)
      .typed<linalg_norm_out::schema>();
}

// aten::linalg_norm.out(Tensor self, Scalar? ord=None, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_norm_out::call(const at::Tensor & self, const c10::optional<at::Scalar> & ord, c10::optional<at::IntArrayRef> dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
    static auto op = create_linalg_norm_out_typed_handle();
    return op.call(self, ord, dim, keepdim, dtype, out);
}

// aten::linalg_norm.out(Tensor self, Scalar? ord=None, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_norm_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Scalar> & ord, c10::optional<at::IntArrayRef> dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
    static auto op = create_linalg_norm_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, ord, dim, keepdim, dtype, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_matrix_norm_out, name, "aten::linalg_matrix_norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_matrix_norm_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_matrix_norm_out, schema_str, "linalg_matrix_norm.out(Tensor self, Scalar ord, int[] dim=[-2,-1], bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)")

// aten::linalg_matrix_norm.out(Tensor self, Scalar ord, int[] dim=[-2,-1], bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_matrix_norm_out::schema> create_linalg_matrix_norm_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_matrix_norm_out::name, linalg_matrix_norm_out::overload_name)
      .typed<linalg_matrix_norm_out::schema>();
}

// aten::linalg_matrix_norm.out(Tensor self, Scalar ord, int[] dim=[-2,-1], bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_matrix_norm_out::call(const at::Tensor & self, const at::Scalar & ord, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
    static auto op = create_linalg_matrix_norm_out_typed_handle();
    return op.call(self, ord, dim, keepdim, dtype, out);
}

// aten::linalg_matrix_norm.out(Tensor self, Scalar ord, int[] dim=[-2,-1], bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_matrix_norm_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & ord, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
    static auto op = create_linalg_matrix_norm_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, ord, dim, keepdim, dtype, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_cond, name, "aten::linalg_cond")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_cond, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_cond, schema_str, "linalg_cond(Tensor self, Scalar? p=None) -> Tensor")

// aten::linalg_cond(Tensor self, Scalar? p=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<linalg_cond::schema> create_linalg_cond_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_cond::name, linalg_cond::overload_name)
      .typed<linalg_cond::schema>();
}

// aten::linalg_cond(Tensor self, Scalar? p=None) -> Tensor
at::Tensor linalg_cond::call(const at::Tensor & self, const c10::optional<at::Scalar> & p) {
    static auto op = create_linalg_cond_typed_handle();
    return op.call(self, p);
}

// aten::linalg_cond(Tensor self, Scalar? p=None) -> Tensor
at::Tensor linalg_cond::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Scalar> & p) {
    static auto op = create_linalg_cond_typed_handle();
    return op.redispatch(dispatchKeySet, self, p);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_cond_p_str, name, "aten::linalg_cond")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_cond_p_str, overload_name, "p_str")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_cond_p_str, schema_str, "linalg_cond.p_str(Tensor self, str p) -> Tensor")

// aten::linalg_cond.p_str(Tensor self, str p) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<linalg_cond_p_str::schema> create_linalg_cond_p_str_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_cond_p_str::name, linalg_cond_p_str::overload_name)
      .typed<linalg_cond_p_str::schema>();
}

// aten::linalg_cond.p_str(Tensor self, str p) -> Tensor
at::Tensor linalg_cond_p_str::call(const at::Tensor & self, c10::string_view p) {
    static auto op = create_linalg_cond_p_str_typed_handle();
    return op.call(self, p);
}

// aten::linalg_cond.p_str(Tensor self, str p) -> Tensor
at::Tensor linalg_cond_p_str::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::string_view p) {
    static auto op = create_linalg_cond_p_str_typed_handle();
    return op.redispatch(dispatchKeySet, self, p);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_cond_p_str_out, name, "aten::linalg_cond")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_cond_p_str_out, overload_name, "p_str_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_cond_p_str_out, schema_str, "linalg_cond.p_str_out(Tensor self, str p, *, Tensor(a!) out) -> Tensor(a!)")

// aten::linalg_cond.p_str_out(Tensor self, str p, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_cond_p_str_out::schema> create_linalg_cond_p_str_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_cond_p_str_out::name, linalg_cond_p_str_out::overload_name)
      .typed<linalg_cond_p_str_out::schema>();
}

// aten::linalg_cond.p_str_out(Tensor self, str p, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_cond_p_str_out::call(const at::Tensor & self, c10::string_view p, at::Tensor & out) {
    static auto op = create_linalg_cond_p_str_out_typed_handle();
    return op.call(self, p, out);
}

// aten::linalg_cond.p_str_out(Tensor self, str p, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_cond_p_str_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::string_view p, at::Tensor & out) {
    static auto op = create_linalg_cond_p_str_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, p, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_pinv, name, "aten::linalg_pinv")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_pinv, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_pinv, schema_str, "linalg_pinv(Tensor self, float rcond=1e-15, bool hermitian=False) -> Tensor")

// aten::linalg_pinv(Tensor self, float rcond=1e-15, bool hermitian=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<linalg_pinv::schema> create_linalg_pinv_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_pinv::name, linalg_pinv::overload_name)
      .typed<linalg_pinv::schema>();
}

// aten::linalg_pinv(Tensor self, float rcond=1e-15, bool hermitian=False) -> Tensor
at::Tensor linalg_pinv::call(const at::Tensor & self, double rcond, bool hermitian) {
    static auto op = create_linalg_pinv_typed_handle();
    return op.call(self, rcond, hermitian);
}

// aten::linalg_pinv(Tensor self, float rcond=1e-15, bool hermitian=False) -> Tensor
at::Tensor linalg_pinv::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double rcond, bool hermitian) {
    static auto op = create_linalg_pinv_typed_handle();
    return op.redispatch(dispatchKeySet, self, rcond, hermitian);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_pinv_rcond_tensor, name, "aten::linalg_pinv")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_pinv_rcond_tensor, overload_name, "rcond_tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_pinv_rcond_tensor, schema_str, "linalg_pinv.rcond_tensor(Tensor self, Tensor rcond, bool hermitian=False) -> Tensor")

// aten::linalg_pinv.rcond_tensor(Tensor self, Tensor rcond, bool hermitian=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<linalg_pinv_rcond_tensor::schema> create_linalg_pinv_rcond_tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_pinv_rcond_tensor::name, linalg_pinv_rcond_tensor::overload_name)
      .typed<linalg_pinv_rcond_tensor::schema>();
}

// aten::linalg_pinv.rcond_tensor(Tensor self, Tensor rcond, bool hermitian=False) -> Tensor
at::Tensor linalg_pinv_rcond_tensor::call(const at::Tensor & self, const at::Tensor & rcond, bool hermitian) {
    static auto op = create_linalg_pinv_rcond_tensor_typed_handle();
    return op.call(self, rcond, hermitian);
}

// aten::linalg_pinv.rcond_tensor(Tensor self, Tensor rcond, bool hermitian=False) -> Tensor
at::Tensor linalg_pinv_rcond_tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & rcond, bool hermitian) {
    static auto op = create_linalg_pinv_rcond_tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, rcond, hermitian);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_pinv_out, name, "aten::linalg_pinv")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_pinv_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_pinv_out, schema_str, "linalg_pinv.out(Tensor self, float rcond=1e-15, bool hermitian=False, *, Tensor(a!) out) -> Tensor(a!)")

// aten::linalg_pinv.out(Tensor self, float rcond=1e-15, bool hermitian=False, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_pinv_out::schema> create_linalg_pinv_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_pinv_out::name, linalg_pinv_out::overload_name)
      .typed<linalg_pinv_out::schema>();
}

// aten::linalg_pinv.out(Tensor self, float rcond=1e-15, bool hermitian=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_pinv_out::call(const at::Tensor & self, double rcond, bool hermitian, at::Tensor & out) {
    static auto op = create_linalg_pinv_out_typed_handle();
    return op.call(self, rcond, hermitian, out);
}

// aten::linalg_pinv.out(Tensor self, float rcond=1e-15, bool hermitian=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_pinv_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double rcond, bool hermitian, at::Tensor & out) {
    static auto op = create_linalg_pinv_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, rcond, hermitian, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_tensorsolve, name, "aten::linalg_tensorsolve")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_tensorsolve, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_tensorsolve, schema_str, "linalg_tensorsolve(Tensor self, Tensor other, int[]? dims=None) -> Tensor")

// aten::linalg_tensorsolve(Tensor self, Tensor other, int[]? dims=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<linalg_tensorsolve::schema> create_linalg_tensorsolve_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_tensorsolve::name, linalg_tensorsolve::overload_name)
      .typed<linalg_tensorsolve::schema>();
}

// aten::linalg_tensorsolve(Tensor self, Tensor other, int[]? dims=None) -> Tensor
at::Tensor linalg_tensorsolve::call(const at::Tensor & self, const at::Tensor & other, c10::optional<at::IntArrayRef> dims) {
    static auto op = create_linalg_tensorsolve_typed_handle();
    return op.call(self, other, dims);
}

// aten::linalg_tensorsolve(Tensor self, Tensor other, int[]? dims=None) -> Tensor
at::Tensor linalg_tensorsolve::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, c10::optional<at::IntArrayRef> dims) {
    static auto op = create_linalg_tensorsolve_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, dims);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_multi_dot, name, "aten::linalg_multi_dot")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_multi_dot, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_multi_dot, schema_str, "linalg_multi_dot(Tensor[] tensors) -> Tensor")

// aten::linalg_multi_dot(Tensor[] tensors) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<linalg_multi_dot::schema> create_linalg_multi_dot_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_multi_dot::name, linalg_multi_dot::overload_name)
      .typed<linalg_multi_dot::schema>();
}

// aten::linalg_multi_dot(Tensor[] tensors) -> Tensor
at::Tensor linalg_multi_dot::call(at::TensorList tensors) {
    static auto op = create_linalg_multi_dot_typed_handle();
    return op.call(tensors);
}

// aten::linalg_multi_dot(Tensor[] tensors) -> Tensor
at::Tensor linalg_multi_dot::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create_linalg_multi_dot_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_multi_dot_out, name, "aten::linalg_multi_dot")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_multi_dot_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_multi_dot_out, schema_str, "linalg_multi_dot.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)")

// aten::linalg_multi_dot.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_multi_dot_out::schema> create_linalg_multi_dot_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_multi_dot_out::name, linalg_multi_dot_out::overload_name)
      .typed<linalg_multi_dot_out::schema>();
}

// aten::linalg_multi_dot.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_multi_dot_out::call(at::TensorList tensors, at::Tensor & out) {
    static auto op = create_linalg_multi_dot_out_typed_handle();
    return op.call(tensors, out);
}

// aten::linalg_multi_dot.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_multi_dot_out::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, at::Tensor & out) {
    static auto op = create_linalg_multi_dot_out_typed_handle();
    return op.redispatch(dispatchKeySet, tensors, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_test_string_default, name, "aten::_test_string_default")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_test_string_default, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_test_string_default, schema_str, "_test_string_default(Tensor dummy, str a=\"\\\"'\\\\\", str b='\"\\'\\\\') -> Tensor")

// aten::_test_string_default(Tensor dummy, str a="\"'\\", str b='"\'\\') -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_test_string_default::schema> create__test_string_default_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_test_string_default::name, _test_string_default::overload_name)
      .typed<_test_string_default::schema>();
}

// aten::_test_string_default(Tensor dummy, str a="\"'\\", str b='"\'\\') -> Tensor
at::Tensor _test_string_default::call(const at::Tensor & dummy, c10::string_view a, c10::string_view b) {
    static auto op = create__test_string_default_typed_handle();
    return op.call(dummy, a, b);
}

// aten::_test_string_default(Tensor dummy, str a="\"'\\", str b='"\'\\') -> Tensor
at::Tensor _test_string_default::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & dummy, c10::string_view a, c10::string_view b) {
    static auto op = create__test_string_default_typed_handle();
    return op.redispatch(dispatchKeySet, dummy, a, b);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_test_ambiguous_defaults_a, name, "aten::_test_ambiguous_defaults")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_test_ambiguous_defaults_a, overload_name, "a")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_test_ambiguous_defaults_a, schema_str, "_test_ambiguous_defaults.a(Tensor dummy, int a=1, int b=1) -> Tensor")

// aten::_test_ambiguous_defaults.a(Tensor dummy, int a=1, int b=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_test_ambiguous_defaults_a::schema> create__test_ambiguous_defaults_a_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_test_ambiguous_defaults_a::name, _test_ambiguous_defaults_a::overload_name)
      .typed<_test_ambiguous_defaults_a::schema>();
}

// aten::_test_ambiguous_defaults.a(Tensor dummy, int a=1, int b=1) -> Tensor
at::Tensor _test_ambiguous_defaults_a::call(const at::Tensor & dummy, int64_t a, int64_t b) {
    static auto op = create__test_ambiguous_defaults_a_typed_handle();
    return op.call(dummy, a, b);
}

// aten::_test_ambiguous_defaults.a(Tensor dummy, int a=1, int b=1) -> Tensor
at::Tensor _test_ambiguous_defaults_a::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & dummy, int64_t a, int64_t b) {
    static auto op = create__test_ambiguous_defaults_a_typed_handle();
    return op.redispatch(dispatchKeySet, dummy, a, b);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(flatten_dense_tensors, name, "aten::flatten_dense_tensors")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(flatten_dense_tensors, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(flatten_dense_tensors, schema_str, "flatten_dense_tensors(Tensor[] tensors) -> Tensor")

// aten::flatten_dense_tensors(Tensor[] tensors) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<flatten_dense_tensors::schema> create_flatten_dense_tensors_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(flatten_dense_tensors::name, flatten_dense_tensors::overload_name)
      .typed<flatten_dense_tensors::schema>();
}

// aten::flatten_dense_tensors(Tensor[] tensors) -> Tensor
at::Tensor flatten_dense_tensors::call(at::TensorList tensors) {
    static auto op = create_flatten_dense_tensors_typed_handle();
    return op.call(tensors);
}

// aten::flatten_dense_tensors(Tensor[] tensors) -> Tensor
at::Tensor flatten_dense_tensors::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create_flatten_dense_tensors_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

}} // namespace at::_ops
