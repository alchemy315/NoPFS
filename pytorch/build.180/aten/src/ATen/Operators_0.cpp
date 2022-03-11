#include <ATen/Operators.h>
#include <ATen/Tensor.h>
#include <ATen/core/dispatch/Dispatcher.h>

// NOTE See [Sharded File] comment in VariableType

namespace at { namespace _ops {


STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cast_Byte, name, "aten::_cast_Byte")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cast_Byte, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cast_Byte, schema_str, "_cast_Byte(Tensor self, bool non_blocking=False) -> Tensor")

// aten::_cast_Byte(Tensor self, bool non_blocking=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_cast_Byte::schema> create__cast_Byte_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_cast_Byte::name, _cast_Byte::overload_name)
      .typed<_cast_Byte::schema>();
}

// aten::_cast_Byte(Tensor self, bool non_blocking=False) -> Tensor
at::Tensor _cast_Byte::call(const at::Tensor & self, bool non_blocking) {
    static auto op = create__cast_Byte_typed_handle();
    return op.call(self, non_blocking);
}

// aten::_cast_Byte(Tensor self, bool non_blocking=False) -> Tensor
at::Tensor _cast_Byte::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool non_blocking) {
    static auto op = create__cast_Byte_typed_handle();
    return op.redispatch(dispatchKeySet, self, non_blocking);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cast_Char, name, "aten::_cast_Char")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cast_Char, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cast_Char, schema_str, "_cast_Char(Tensor self, bool non_blocking=False) -> Tensor")

// aten::_cast_Char(Tensor self, bool non_blocking=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_cast_Char::schema> create__cast_Char_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_cast_Char::name, _cast_Char::overload_name)
      .typed<_cast_Char::schema>();
}

// aten::_cast_Char(Tensor self, bool non_blocking=False) -> Tensor
at::Tensor _cast_Char::call(const at::Tensor & self, bool non_blocking) {
    static auto op = create__cast_Char_typed_handle();
    return op.call(self, non_blocking);
}

// aten::_cast_Char(Tensor self, bool non_blocking=False) -> Tensor
at::Tensor _cast_Char::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool non_blocking) {
    static auto op = create__cast_Char_typed_handle();
    return op.redispatch(dispatchKeySet, self, non_blocking);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_leaf, name, "aten::is_leaf")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_leaf, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_leaf, schema_str, "is_leaf(Tensor self) -> bool")

// aten::is_leaf(Tensor self) -> bool
static C10_NOINLINE c10::TypedOperatorHandle<is_leaf::schema> create_is_leaf_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(is_leaf::name, is_leaf::overload_name)
      .typed<is_leaf::schema>();
}

// aten::is_leaf(Tensor self) -> bool
bool is_leaf::call(const at::Tensor & self) {
    static auto op = create_is_leaf_typed_handle();
    return op.call(self);
}

// aten::is_leaf(Tensor self) -> bool
bool is_leaf::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_is_leaf_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(retains_grad, name, "aten::retains_grad")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(retains_grad, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(retains_grad, schema_str, "retains_grad(Tensor self) -> bool")

// aten::retains_grad(Tensor self) -> bool
static C10_NOINLINE c10::TypedOperatorHandle<retains_grad::schema> create_retains_grad_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(retains_grad::name, retains_grad::overload_name)
      .typed<retains_grad::schema>();
}

// aten::retains_grad(Tensor self) -> bool
bool retains_grad::call(const at::Tensor & self) {
    static auto op = create_retains_grad_typed_handle();
    return op.call(self);
}

// aten::retains_grad(Tensor self) -> bool
bool retains_grad::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_retains_grad_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_unpack_dual, name, "aten::_unpack_dual")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_unpack_dual, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_unpack_dual, schema_str, "_unpack_dual(Tensor(a) dual, int level) -> (Tensor(a) primal, Tensor tangent)")

// aten::_unpack_dual(Tensor(a) dual, int level) -> (Tensor(a) primal, Tensor tangent)
static C10_NOINLINE c10::TypedOperatorHandle<_unpack_dual::schema> create__unpack_dual_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_unpack_dual::name, _unpack_dual::overload_name)
      .typed<_unpack_dual::schema>();
}

// aten::_unpack_dual(Tensor(a) dual, int level) -> (Tensor(a) primal, Tensor tangent)
::std::tuple<at::Tensor,at::Tensor> _unpack_dual::call(const at::Tensor & dual, int64_t level) {
    static auto op = create__unpack_dual_typed_handle();
    return op.call(dual, level);
}

// aten::_unpack_dual(Tensor(a) dual, int level) -> (Tensor(a) primal, Tensor tangent)
::std::tuple<at::Tensor,at::Tensor> _unpack_dual::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & dual, int64_t level) {
    static auto op = create__unpack_dual_typed_handle();
    return op.redispatch(dispatchKeySet, dual, level);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(align_to, name, "aten::align_to")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(align_to, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(align_to, schema_str, "align_to(Tensor(a) self, Dimname[] names) -> Tensor(a)")

// aten::align_to(Tensor(a) self, Dimname[] names) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<align_to::schema> create_align_to_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(align_to::name, align_to::overload_name)
      .typed<align_to::schema>();
}

// aten::align_to(Tensor(a) self, Dimname[] names) -> Tensor(a)
at::Tensor align_to::call(const at::Tensor & self, at::DimnameList names) {
    static auto op = create_align_to_typed_handle();
    return op.call(self, names);
}

// aten::align_to(Tensor(a) self, Dimname[] names) -> Tensor(a)
at::Tensor align_to::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList names) {
    static auto op = create_align_to_typed_handle();
    return op.redispatch(dispatchKeySet, self, names);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_use_cudnn_ctc_loss, name, "aten::_use_cudnn_ctc_loss")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_use_cudnn_ctc_loss, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_use_cudnn_ctc_loss, schema_str, "_use_cudnn_ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank) -> bool")

// aten::_use_cudnn_ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank) -> bool
static C10_NOINLINE c10::TypedOperatorHandle<_use_cudnn_ctc_loss::schema> create__use_cudnn_ctc_loss_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_use_cudnn_ctc_loss::name, _use_cudnn_ctc_loss::overload_name)
      .typed<_use_cudnn_ctc_loss::schema>();
}

// aten::_use_cudnn_ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank) -> bool
bool _use_cudnn_ctc_loss::call(const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, int64_t blank) {
    static auto op = create__use_cudnn_ctc_loss_typed_handle();
    return op.call(log_probs, targets, input_lengths, target_lengths, blank);
}

// aten::_use_cudnn_ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank) -> bool
bool _use_cudnn_ctc_loss::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, int64_t blank) {
    static auto op = create__use_cudnn_ctc_loss_typed_handle();
    return op.redispatch(dispatchKeySet, log_probs, targets, input_lengths, target_lengths, blank);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cudnn_ctc_loss, name, "aten::_cudnn_ctc_loss")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cudnn_ctc_loss, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cudnn_ctc_loss, schema_str, "_cudnn_ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank, bool deterministic, bool zero_infinity) -> (Tensor, Tensor)")

// aten::_cudnn_ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank, bool deterministic, bool zero_infinity) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<_cudnn_ctc_loss::schema> create__cudnn_ctc_loss_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_cudnn_ctc_loss::name, _cudnn_ctc_loss::overload_name)
      .typed<_cudnn_ctc_loss::schema>();
}

// aten::_cudnn_ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank, bool deterministic, bool zero_infinity) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> _cudnn_ctc_loss::call(const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, int64_t blank, bool deterministic, bool zero_infinity) {
    static auto op = create__cudnn_ctc_loss_typed_handle();
    return op.call(log_probs, targets, input_lengths, target_lengths, blank, deterministic, zero_infinity);
}

// aten::_cudnn_ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank, bool deterministic, bool zero_infinity) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> _cudnn_ctc_loss::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, int64_t blank, bool deterministic, bool zero_infinity) {
    static auto op = create__cudnn_ctc_loss_typed_handle();
    return op.redispatch(dispatchKeySet, log_probs, targets, input_lengths, target_lengths, blank, deterministic, zero_infinity);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cudnn_rnn, name, "aten::_cudnn_rnn")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cudnn_rnn, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cudnn_rnn, schema_str, "_cudnn_rnn(Tensor input, Tensor[] weight, int weight_stride0, Tensor? weight_buf, Tensor hx, Tensor? cx, int mode, int hidden_size, int proj_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state) -> (Tensor, Tensor, Tensor, Tensor, Tensor)")

// aten::_cudnn_rnn(Tensor input, Tensor[] weight, int weight_stride0, Tensor? weight_buf, Tensor hx, Tensor? cx, int mode, int hidden_size, int proj_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<_cudnn_rnn::schema> create__cudnn_rnn_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_cudnn_rnn::name, _cudnn_rnn::overload_name)
      .typed<_cudnn_rnn::schema>();
}

// aten::_cudnn_rnn(Tensor input, Tensor[] weight, int weight_stride0, Tensor? weight_buf, Tensor hx, Tensor? cx, int mode, int hidden_size, int proj_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> _cudnn_rnn::call(const at::Tensor & input, at::TensorList weight, int64_t weight_stride0, const c10::optional<at::Tensor> & weight_buf, const at::Tensor & hx, const c10::optional<at::Tensor> & cx, int64_t mode, int64_t hidden_size, int64_t proj_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, at::IntArrayRef batch_sizes, const c10::optional<at::Tensor> & dropout_state) {
    static auto op = create__cudnn_rnn_typed_handle();
    return op.call(input, weight, weight_stride0, weight_buf, hx, cx, mode, hidden_size, proj_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state);
}

// aten::_cudnn_rnn(Tensor input, Tensor[] weight, int weight_stride0, Tensor? weight_buf, Tensor hx, Tensor? cx, int mode, int hidden_size, int proj_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> _cudnn_rnn::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, at::TensorList weight, int64_t weight_stride0, const c10::optional<at::Tensor> & weight_buf, const at::Tensor & hx, const c10::optional<at::Tensor> & cx, int64_t mode, int64_t hidden_size, int64_t proj_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, at::IntArrayRef batch_sizes, const c10::optional<at::Tensor> & dropout_state) {
    static auto op = create__cudnn_rnn_typed_handle();
    return op.redispatch(dispatchKeySet, input, weight, weight_stride0, weight_buf, hx, cx, mode, hidden_size, proj_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_debug_has_internal_overlap, name, "aten::_debug_has_internal_overlap")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_debug_has_internal_overlap, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_debug_has_internal_overlap, schema_str, "_debug_has_internal_overlap(Tensor self) -> int")

// aten::_debug_has_internal_overlap(Tensor self) -> int
static C10_NOINLINE c10::TypedOperatorHandle<_debug_has_internal_overlap::schema> create__debug_has_internal_overlap_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_debug_has_internal_overlap::name, _debug_has_internal_overlap::overload_name)
      .typed<_debug_has_internal_overlap::schema>();
}

// aten::_debug_has_internal_overlap(Tensor self) -> int
int64_t _debug_has_internal_overlap::call(const at::Tensor & self) {
    static auto op = create__debug_has_internal_overlap_typed_handle();
    return op.call(self);
}

// aten::_debug_has_internal_overlap(Tensor self) -> int
int64_t _debug_has_internal_overlap::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create__debug_has_internal_overlap_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fused_dropout, name, "aten::_fused_dropout")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fused_dropout, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fused_dropout, schema_str, "_fused_dropout(Tensor self, float p, Generator? generator=None) -> (Tensor, Tensor)")

// aten::_fused_dropout(Tensor self, float p, Generator? generator=None) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<_fused_dropout::schema> create__fused_dropout_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_fused_dropout::name, _fused_dropout::overload_name)
      .typed<_fused_dropout::schema>();
}

// aten::_fused_dropout(Tensor self, float p, Generator? generator=None) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> _fused_dropout::call(const at::Tensor & self, double p, c10::optional<at::Generator> generator) {
    static auto op = create__fused_dropout_typed_handle();
    return op.call(self, p, generator);
}

// aten::_fused_dropout(Tensor self, float p, Generator? generator=None) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> _fused_dropout::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double p, c10::optional<at::Generator> generator) {
    static auto op = create__fused_dropout_typed_handle();
    return op.redispatch(dispatchKeySet, self, p, generator);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_shape_as_tensor, name, "aten::_shape_as_tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_shape_as_tensor, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_shape_as_tensor, schema_str, "_shape_as_tensor(Tensor self) -> Tensor")

// aten::_shape_as_tensor(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_shape_as_tensor::schema> create__shape_as_tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_shape_as_tensor::name, _shape_as_tensor::overload_name)
      .typed<_shape_as_tensor::schema>();
}

// aten::_shape_as_tensor(Tensor self) -> Tensor
at::Tensor _shape_as_tensor::call(const at::Tensor & self) {
    static auto op = create__shape_as_tensor_typed_handle();
    return op.call(self);
}

// aten::_shape_as_tensor(Tensor self) -> Tensor
at::Tensor _shape_as_tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create__shape_as_tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(dropout, name, "aten::dropout")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(dropout, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(dropout, schema_str, "dropout(Tensor input, float p, bool train) -> Tensor")

// aten::dropout(Tensor input, float p, bool train) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<dropout::schema> create_dropout_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(dropout::name, dropout::overload_name)
      .typed<dropout::schema>();
}

// aten::dropout(Tensor input, float p, bool train) -> Tensor
at::Tensor dropout::call(const at::Tensor & input, double p, bool train) {
    static auto op = create_dropout_typed_handle();
    return op.call(input, p, train);
}

// aten::dropout(Tensor input, float p, bool train) -> Tensor
at::Tensor dropout::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, double p, bool train) {
    static auto op = create_dropout_typed_handle();
    return op.redispatch(dispatchKeySet, input, p, train);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(abs_out, name, "aten::abs")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(abs_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(abs_out, schema_str, "abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<abs_out::schema> create_abs_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(abs_out::name, abs_out::overload_name)
      .typed<abs_out::schema>();
}

// aten::abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & abs_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_abs_out_typed_handle();
    return op.call(self, out);
}

// aten::abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & abs_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_abs_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(angle_out, name, "aten::angle")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(angle_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(angle_out, schema_str, "angle.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::angle.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<angle_out::schema> create_angle_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(angle_out::name, angle_out::overload_name)
      .typed<angle_out::schema>();
}

// aten::angle.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & angle_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_angle_out_typed_handle();
    return op.call(self, out);
}

// aten::angle.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & angle_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_angle_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sgn, name, "aten::sgn")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sgn, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sgn, schema_str, "sgn(Tensor self) -> Tensor")

// aten::sgn(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<sgn::schema> create_sgn_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sgn::name, sgn::overload_name)
      .typed<sgn::schema>();
}

// aten::sgn(Tensor self) -> Tensor
at::Tensor sgn::call(const at::Tensor & self) {
    static auto op = create_sgn_typed_handle();
    return op.call(self);
}

// aten::sgn(Tensor self) -> Tensor
at::Tensor sgn::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_sgn_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(real, name, "aten::real")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(real, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(real, schema_str, "real(Tensor(a) self) -> Tensor(a)")

// aten::real(Tensor(a) self) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<real::schema> create_real_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(real::name, real::overload_name)
      .typed<real::schema>();
}

// aten::real(Tensor(a) self) -> Tensor(a)
at::Tensor real::call(const at::Tensor & self) {
    static auto op = create_real_typed_handle();
    return op.call(self);
}

// aten::real(Tensor(a) self) -> Tensor(a)
at::Tensor real::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_real_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_conj, name, "aten::_conj")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_conj, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_conj, schema_str, "_conj(Tensor(a) self) -> Tensor(a)")

// aten::_conj(Tensor(a) self) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<_conj::schema> create__conj_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_conj::name, _conj::overload_name)
      .typed<_conj::schema>();
}

// aten::_conj(Tensor(a) self) -> Tensor(a)
at::Tensor _conj::call(const at::Tensor & self) {
    static auto op = create__conj_typed_handle();
    return op.call(self);
}

// aten::_conj(Tensor(a) self) -> Tensor(a)
at::Tensor _conj::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create__conj_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_conj_physical, name, "aten::_conj_physical")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_conj_physical, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_conj_physical, schema_str, "_conj_physical(Tensor self) -> Tensor")

// aten::_conj_physical(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_conj_physical::schema> create__conj_physical_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_conj_physical::name, _conj_physical::overload_name)
      .typed<_conj_physical::schema>();
}

// aten::_conj_physical(Tensor self) -> Tensor
at::Tensor _conj_physical::call(const at::Tensor & self) {
    static auto op = create__conj_physical_typed_handle();
    return op.call(self);
}

// aten::_conj_physical(Tensor self) -> Tensor
at::Tensor _conj_physical::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create__conj_physical_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conj_physical_out, name, "aten::conj_physical")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conj_physical_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conj_physical_out, schema_str, "conj_physical.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::conj_physical.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<conj_physical_out::schema> create_conj_physical_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(conj_physical_out::name, conj_physical_out::overload_name)
      .typed<conj_physical_out::schema>();
}

// aten::conj_physical.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & conj_physical_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_conj_physical_out_typed_handle();
    return op.call(self, out);
}

// aten::conj_physical.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & conj_physical_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_conj_physical_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_neg_view, name, "aten::_neg_view")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_neg_view, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_neg_view, schema_str, "_neg_view(Tensor(a) self) -> Tensor(a)")

// aten::_neg_view(Tensor(a) self) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<_neg_view::schema> create__neg_view_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_neg_view::name, _neg_view::overload_name)
      .typed<_neg_view::schema>();
}

// aten::_neg_view(Tensor(a) self) -> Tensor(a)
at::Tensor _neg_view::call(const at::Tensor & self) {
    static auto op = create__neg_view_typed_handle();
    return op.call(self);
}

// aten::_neg_view(Tensor(a) self) -> Tensor(a)
at::Tensor _neg_view::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create__neg_view_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arccos_, name, "aten::arccos_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arccos_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arccos_, schema_str, "arccos_(Tensor(a!) self) -> Tensor(a!)")

// aten::arccos_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<arccos_::schema> create_arccos__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(arccos_::name, arccos_::overload_name)
      .typed<arccos_::schema>();
}

// aten::arccos_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & arccos_::call(at::Tensor & self) {
    static auto op = create_arccos__typed_handle();
    return op.call(self);
}

// aten::arccos_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & arccos_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_arccos__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arccos_out, name, "aten::arccos")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arccos_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arccos_out, schema_str, "arccos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::arccos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<arccos_out::schema> create_arccos_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(arccos_out::name, arccos_out::overload_name)
      .typed<arccos_out::schema>();
}

// aten::arccos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & arccos_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_arccos_out_typed_handle();
    return op.call(self, out);
}

// aten::arccos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & arccos_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_arccos_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(avg_pool1d, name, "aten::avg_pool1d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(avg_pool1d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(avg_pool1d, schema_str, "avg_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, bool ceil_mode=False, bool count_include_pad=True) -> Tensor")

// aten::avg_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, bool ceil_mode=False, bool count_include_pad=True) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<avg_pool1d::schema> create_avg_pool1d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(avg_pool1d::name, avg_pool1d::overload_name)
      .typed<avg_pool1d::schema>();
}

// aten::avg_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, bool ceil_mode=False, bool count_include_pad=True) -> Tensor
at::Tensor avg_pool1d::call(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad) {
    static auto op = create_avg_pool1d_typed_handle();
    return op.call(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}

// aten::avg_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, bool ceil_mode=False, bool count_include_pad=True) -> Tensor
at::Tensor avg_pool1d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad) {
    static auto op = create_avg_pool1d_typed_handle();
    return op.redispatch(dispatchKeySet, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_avg_pool1d, name, "aten::adaptive_avg_pool1d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_avg_pool1d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_avg_pool1d, schema_str, "adaptive_avg_pool1d(Tensor self, int[1] output_size) -> Tensor")

// aten::adaptive_avg_pool1d(Tensor self, int[1] output_size) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<adaptive_avg_pool1d::schema> create_adaptive_avg_pool1d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(adaptive_avg_pool1d::name, adaptive_avg_pool1d::overload_name)
      .typed<adaptive_avg_pool1d::schema>();
}

// aten::adaptive_avg_pool1d(Tensor self, int[1] output_size) -> Tensor
at::Tensor adaptive_avg_pool1d::call(const at::Tensor & self, at::IntArrayRef output_size) {
    static auto op = create_adaptive_avg_pool1d_typed_handle();
    return op.call(self, output_size);
}

// aten::adaptive_avg_pool1d(Tensor self, int[1] output_size) -> Tensor
at::Tensor adaptive_avg_pool1d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size) {
    static auto op = create_adaptive_avg_pool1d_typed_handle();
    return op.redispatch(dispatchKeySet, self, output_size);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_add_relu__Scalar, name, "aten::_add_relu_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_add_relu__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_add_relu__Scalar, schema_str, "_add_relu_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)")

// aten::_add_relu_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<_add_relu__Scalar::schema> create__add_relu__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_add_relu__Scalar::name, _add_relu__Scalar::overload_name)
      .typed<_add_relu__Scalar::schema>();
}

// aten::_add_relu_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
at::Tensor & _add_relu__Scalar::call(at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
    static auto op = create__add_relu__Scalar_typed_handle();
    return op.call(self, other, alpha);
}

// aten::_add_relu_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
at::Tensor & _add_relu__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
    static auto op = create__add_relu__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addmv_, name, "aten::addmv_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addmv_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addmv_, schema_str, "addmv_(Tensor(a!) self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)")

// aten::addmv_(Tensor(a!) self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<addmv_::schema> create_addmv__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(addmv_::name, addmv_::overload_name)
      .typed<addmv_::schema>();
}

// aten::addmv_(Tensor(a!) self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
at::Tensor & addmv_::call(at::Tensor & self, const at::Tensor & mat, const at::Tensor & vec, const at::Scalar & beta, const at::Scalar & alpha) {
    static auto op = create_addmv__typed_handle();
    return op.call(self, mat, vec, beta, alpha);
}

// aten::addmv_(Tensor(a!) self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
at::Tensor & addmv_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & mat, const at::Tensor & vec, const at::Scalar & beta, const at::Scalar & alpha) {
    static auto op = create_addmv__typed_handle();
    return op.redispatch(dispatchKeySet, self, mat, vec, beta, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(all_dim, name, "aten::all")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(all_dim, overload_name, "dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(all_dim, schema_str, "all.dim(Tensor self, int dim, bool keepdim=False) -> Tensor")

// aten::all.dim(Tensor self, int dim, bool keepdim=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<all_dim::schema> create_all_dim_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(all_dim::name, all_dim::overload_name)
      .typed<all_dim::schema>();
}

// aten::all.dim(Tensor self, int dim, bool keepdim=False) -> Tensor
at::Tensor all_dim::call(const at::Tensor & self, int64_t dim, bool keepdim) {
    static auto op = create_all_dim_typed_handle();
    return op.call(self, dim, keepdim);
}

// aten::all.dim(Tensor self, int dim, bool keepdim=False) -> Tensor
at::Tensor all_dim::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool keepdim) {
    static auto op = create_all_dim_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(allclose, name, "aten::allclose")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(allclose, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(allclose, schema_str, "allclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> bool")

// aten::allclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> bool
static C10_NOINLINE c10::TypedOperatorHandle<allclose::schema> create_allclose_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(allclose::name, allclose::overload_name)
      .typed<allclose::schema>();
}

// aten::allclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> bool
bool allclose::call(const at::Tensor & self, const at::Tensor & other, double rtol, double atol, bool equal_nan) {
    static auto op = create_allclose_typed_handle();
    return op.call(self, other, rtol, atol, equal_nan);
}

// aten::allclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> bool
bool allclose::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, double rtol, double atol, bool equal_nan) {
    static auto op = create_allclose_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, rtol, atol, equal_nan);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(any_dimname_out, name, "aten::any")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(any_dimname_out, overload_name, "dimname_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(any_dimname_out, schema_str, "any.dimname_out(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)")

// aten::any.dimname_out(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<any_dimname_out::schema> create_any_dimname_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(any_dimname_out::name, any_dimname_out::overload_name)
      .typed<any_dimname_out::schema>();
}

// aten::any.dimname_out(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & any_dimname_out::call(const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & out) {
    static auto op = create_any_dimname_out_typed_handle();
    return op.call(self, dim, keepdim, out);
}

// aten::any.dimname_out(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & any_dimname_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & out) {
    static auto op = create_any_dimname_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arange_start, name, "aten::arange")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arange_start, overload_name, "start")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arange_start, schema_str, "arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<arange_start::schema> create_arange_start_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(arange_start::name, arange_start::overload_name)
      .typed<arange_start::schema>();
}

// aten::arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor arange_start::call(const at::Scalar & start, const at::Scalar & end, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_arange_start_typed_handle();
    return op.call(start, end, dtype, layout, device, pin_memory);
}

// aten::arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor arange_start::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Scalar & start, const at::Scalar & end, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_arange_start_typed_handle();
    return op.redispatch(dispatchKeySet, start, end, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(argmax, name, "aten::argmax")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(argmax, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(argmax, schema_str, "argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor")

// aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<argmax::schema> create_argmax_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(argmax::name, argmax::overload_name)
      .typed<argmax::schema>();
}

// aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
at::Tensor argmax::call(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim) {
    static auto op = create_argmax_typed_handle();
    return op.call(self, dim, keepdim);
}

// aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
at::Tensor argmax::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim) {
    static auto op = create_argmax_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(acosh, name, "aten::acosh")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(acosh, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(acosh, schema_str, "acosh(Tensor self) -> Tensor")

// aten::acosh(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<acosh::schema> create_acosh_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(acosh::name, acosh::overload_name)
      .typed<acosh::schema>();
}

// aten::acosh(Tensor self) -> Tensor
at::Tensor acosh::call(const at::Tensor & self) {
    static auto op = create_acosh_typed_handle();
    return op.call(self);
}

// aten::acosh(Tensor self) -> Tensor
at::Tensor acosh::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_acosh_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arccosh_out, name, "aten::arccosh")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arccosh_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arccosh_out, schema_str, "arccosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::arccosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<arccosh_out::schema> create_arccosh_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(arccosh_out::name, arccosh_out::overload_name)
      .typed<arccosh_out::schema>();
}

// aten::arccosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & arccosh_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_arccosh_out_typed_handle();
    return op.call(self, out);
}

// aten::arccosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & arccosh_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_arccosh_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atanh_, name, "aten::atanh_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atanh_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atanh_, schema_str, "atanh_(Tensor(a!) self) -> Tensor(a!)")

// aten::atanh_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<atanh_::schema> create_atanh__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(atanh_::name, atanh_::overload_name)
      .typed<atanh_::schema>();
}

// aten::atanh_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & atanh_::call(at::Tensor & self) {
    static auto op = create_atanh__typed_handle();
    return op.call(self);
}

// aten::atanh_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & atanh_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_atanh__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arctanh, name, "aten::arctanh")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arctanh, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arctanh, schema_str, "arctanh(Tensor self) -> Tensor")

// aten::arctanh(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<arctanh::schema> create_arctanh_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(arctanh::name, arctanh::overload_name)
      .typed<arctanh::schema>();
}

// aten::arctanh(Tensor self) -> Tensor
at::Tensor arctanh::call(const at::Tensor & self) {
    static auto op = create_arctanh_typed_handle();
    return op.call(self);
}

// aten::arctanh(Tensor self) -> Tensor
at::Tensor arctanh::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_arctanh_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(as_strided, name, "aten::as_strided")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(as_strided, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(as_strided, schema_str, "as_strided(Tensor(a) self, int[] size, int[] stride, int? storage_offset=None) -> Tensor(a)")

// aten::as_strided(Tensor(a) self, int[] size, int[] stride, int? storage_offset=None) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<as_strided::schema> create_as_strided_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(as_strided::name, as_strided::overload_name)
      .typed<as_strided::schema>();
}

// aten::as_strided(Tensor(a) self, int[] size, int[] stride, int? storage_offset=None) -> Tensor(a)
at::Tensor as_strided::call(const at::Tensor & self, at::IntArrayRef size, at::IntArrayRef stride, c10::optional<int64_t> storage_offset) {
    static auto op = create_as_strided_typed_handle();
    return op.call(self, size, stride, storage_offset);
}

// aten::as_strided(Tensor(a) self, int[] size, int[] stride, int? storage_offset=None) -> Tensor(a)
at::Tensor as_strided::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef size, at::IntArrayRef stride, c10::optional<int64_t> storage_offset) {
    static auto op = create_as_strided_typed_handle();
    return op.redispatch(dispatchKeySet, self, size, stride, storage_offset);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arcsin_, name, "aten::arcsin_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arcsin_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arcsin_, schema_str, "arcsin_(Tensor(a!) self) -> Tensor(a!)")

// aten::arcsin_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<arcsin_::schema> create_arcsin__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(arcsin_::name, arcsin_::overload_name)
      .typed<arcsin_::schema>();
}

// aten::arcsin_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & arcsin_::call(at::Tensor & self) {
    static auto op = create_arcsin__typed_handle();
    return op.call(self);
}

// aten::arcsin_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & arcsin_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_arcsin__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atleast_3d, name, "aten::atleast_3d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atleast_3d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atleast_3d, schema_str, "atleast_3d(Tensor self) -> Tensor")

// aten::atleast_3d(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<atleast_3d::schema> create_atleast_3d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(atleast_3d::name, atleast_3d::overload_name)
      .typed<atleast_3d::schema>();
}

// aten::atleast_3d(Tensor self) -> Tensor
at::Tensor atleast_3d::call(const at::Tensor & self) {
    static auto op = create_atleast_3d_typed_handle();
    return op.call(self);
}

// aten::atleast_3d(Tensor self) -> Tensor
at::Tensor atleast_3d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_atleast_3d_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_baddbmm_mkl_, name, "aten::_baddbmm_mkl_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_baddbmm_mkl_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_baddbmm_mkl_, schema_str, "_baddbmm_mkl_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)")

// aten::_baddbmm_mkl_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<_baddbmm_mkl_::schema> create__baddbmm_mkl__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_baddbmm_mkl_::name, _baddbmm_mkl_::overload_name)
      .typed<_baddbmm_mkl_::schema>();
}

// aten::_baddbmm_mkl_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
at::Tensor & _baddbmm_mkl_::call(at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha) {
    static auto op = create__baddbmm_mkl__typed_handle();
    return op.call(self, batch1, batch2, beta, alpha);
}

// aten::_baddbmm_mkl_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
at::Tensor & _baddbmm_mkl_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha) {
    static auto op = create__baddbmm_mkl__typed_handle();
    return op.redispatch(dispatchKeySet, self, batch1, batch2, beta, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bartlett_window_periodic, name, "aten::bartlett_window")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bartlett_window_periodic, overload_name, "periodic")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bartlett_window_periodic, schema_str, "bartlett_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::bartlett_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<bartlett_window_periodic::schema> create_bartlett_window_periodic_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bartlett_window_periodic::name, bartlett_window_periodic::overload_name)
      .typed<bartlett_window_periodic::schema>();
}

// aten::bartlett_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor bartlett_window_periodic::call(int64_t window_length, bool periodic, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_bartlett_window_periodic_typed_handle();
    return op.call(window_length, periodic, dtype, layout, device, pin_memory);
}

// aten::bartlett_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor bartlett_window_periodic::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t window_length, bool periodic, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_bartlett_window_periodic_typed_handle();
    return op.redispatch(dispatchKeySet, window_length, periodic, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_batch_norm_impl_index, name, "aten::_batch_norm_impl_index")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_batch_norm_impl_index, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_batch_norm_impl_index, schema_str, "_batch_norm_impl_index(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> (Tensor, Tensor, Tensor, Tensor, int)")

// aten::_batch_norm_impl_index(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> (Tensor, Tensor, Tensor, Tensor, int)
static C10_NOINLINE c10::TypedOperatorHandle<_batch_norm_impl_index::schema> create__batch_norm_impl_index_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_batch_norm_impl_index::name, _batch_norm_impl_index::overload_name)
      .typed<_batch_norm_impl_index::schema>();
}

// aten::_batch_norm_impl_index(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> (Tensor, Tensor, Tensor, Tensor, int)
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t> _batch_norm_impl_index::call(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double momentum, double eps, bool cudnn_enabled) {
    static auto op = create__batch_norm_impl_index_typed_handle();
    return op.call(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
}

// aten::_batch_norm_impl_index(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> (Tensor, Tensor, Tensor, Tensor, int)
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t> _batch_norm_impl_index::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double momentum, double eps, bool cudnn_enabled) {
    static auto op = create__batch_norm_impl_index_typed_handle();
    return op.redispatch(dispatchKeySet, input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_batch_norm_impl_index_backward, name, "aten::_batch_norm_impl_index_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_batch_norm_impl_index_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_batch_norm_impl_index_backward, schema_str, "_batch_norm_impl_index_backward(int impl_index, Tensor input, Tensor grad_output, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var_transform, bool train, float eps, bool[3] output_mask, Tensor reservedSpace) -> (Tensor, Tensor, Tensor)")

// aten::_batch_norm_impl_index_backward(int impl_index, Tensor input, Tensor grad_output, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var_transform, bool train, float eps, bool[3] output_mask, Tensor reservedSpace) -> (Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<_batch_norm_impl_index_backward::schema> create__batch_norm_impl_index_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_batch_norm_impl_index_backward::name, _batch_norm_impl_index_backward::overload_name)
      .typed<_batch_norm_impl_index_backward::schema>();
}

// aten::_batch_norm_impl_index_backward(int impl_index, Tensor input, Tensor grad_output, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var_transform, bool train, float eps, bool[3] output_mask, Tensor reservedSpace) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _batch_norm_impl_index_backward::call(int64_t impl_index, const at::Tensor & input, const at::Tensor & grad_output, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, const c10::optional<at::Tensor> & save_mean, const c10::optional<at::Tensor> & save_var_transform, bool train, double eps, ::std::array<bool,3> output_mask, const at::Tensor & reservedSpace) {
    static auto op = create__batch_norm_impl_index_backward_typed_handle();
    return op.call(impl_index, input, grad_output, weight, running_mean, running_var, save_mean, save_var_transform, train, eps, output_mask, reservedSpace);
}

// aten::_batch_norm_impl_index_backward(int impl_index, Tensor input, Tensor grad_output, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var_transform, bool train, float eps, bool[3] output_mask, Tensor reservedSpace) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _batch_norm_impl_index_backward::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t impl_index, const at::Tensor & input, const at::Tensor & grad_output, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, const c10::optional<at::Tensor> & save_mean, const c10::optional<at::Tensor> & save_var_transform, bool train, double eps, ::std::array<bool,3> output_mask, const at::Tensor & reservedSpace) {
    static auto op = create__batch_norm_impl_index_backward_typed_handle();
    return op.redispatch(dispatchKeySet, impl_index, input, grad_output, weight, running_mean, running_var, save_mean, save_var_transform, train, eps, output_mask, reservedSpace);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bernoulli_p, name, "aten::bernoulli")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bernoulli_p, overload_name, "p")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bernoulli_p, schema_str, "bernoulli.p(Tensor self, float p, *, Generator? generator=None) -> Tensor")

// aten::bernoulli.p(Tensor self, float p, *, Generator? generator=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<bernoulli_p::schema> create_bernoulli_p_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bernoulli_p::name, bernoulli_p::overload_name)
      .typed<bernoulli_p::schema>();
}

// aten::bernoulli.p(Tensor self, float p, *, Generator? generator=None) -> Tensor
at::Tensor bernoulli_p::call(const at::Tensor & self, double p, c10::optional<at::Generator> generator) {
    static auto op = create_bernoulli_p_typed_handle();
    return op.call(self, p, generator);
}

// aten::bernoulli.p(Tensor self, float p, *, Generator? generator=None) -> Tensor
at::Tensor bernoulli_p::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double p, c10::optional<at::Generator> generator) {
    static auto op = create_bernoulli_p_typed_handle();
    return op.redispatch(dispatchKeySet, self, p, generator);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(binary_cross_entropy_out, name, "aten::binary_cross_entropy")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(binary_cross_entropy_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(binary_cross_entropy_out, schema_str, "binary_cross_entropy.out(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)")

// aten::binary_cross_entropy.out(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<binary_cross_entropy_out::schema> create_binary_cross_entropy_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(binary_cross_entropy_out::name, binary_cross_entropy_out::overload_name)
      .typed<binary_cross_entropy_out::schema>();
}

// aten::binary_cross_entropy.out(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & binary_cross_entropy_out::call(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & out) {
    static auto op = create_binary_cross_entropy_out_typed_handle();
    return op.call(self, target, weight, reduction, out);
}

// aten::binary_cross_entropy.out(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & binary_cross_entropy_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & out) {
    static auto op = create_binary_cross_entropy_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, target, weight, reduction, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(copysign__Scalar, name, "aten::copysign_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(copysign__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(copysign__Scalar, schema_str, "copysign_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)")

// aten::copysign_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<copysign__Scalar::schema> create_copysign__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(copysign__Scalar::name, copysign__Scalar::overload_name)
      .typed<copysign__Scalar::schema>();
}

// aten::copysign_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & copysign__Scalar::call(at::Tensor & self, const at::Scalar & other) {
    static auto op = create_copysign__Scalar_typed_handle();
    return op.call(self, other);
}

// aten::copysign_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & copysign__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
    static auto op = create_copysign__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logical_and_, name, "aten::logical_and_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logical_and_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logical_and_, schema_str, "logical_and_(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::logical_and_(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<logical_and_::schema> create_logical_and__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(logical_and_::name, logical_and_::overload_name)
      .typed<logical_and_::schema>();
}

// aten::logical_and_(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & logical_and_::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create_logical_and__typed_handle();
    return op.call(self, other);
}

// aten::logical_and_(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & logical_and_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create_logical_and__typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logical_or, name, "aten::logical_or")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logical_or, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logical_or, schema_str, "logical_or(Tensor self, Tensor other) -> Tensor")

// aten::logical_or(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<logical_or::schema> create_logical_or_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(logical_or::name, logical_or::overload_name)
      .typed<logical_or::schema>();
}

// aten::logical_or(Tensor self, Tensor other) -> Tensor
at::Tensor logical_or::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_logical_or_typed_handle();
    return op.call(self, other);
}

// aten::logical_or(Tensor self, Tensor other) -> Tensor
at::Tensor logical_or::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_logical_or_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logical_or_, name, "aten::logical_or_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logical_or_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logical_or_, schema_str, "logical_or_(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::logical_or_(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<logical_or_::schema> create_logical_or__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(logical_or_::name, logical_or_::overload_name)
      .typed<logical_or_::schema>();
}

// aten::logical_or_(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & logical_or_::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create_logical_or__typed_handle();
    return op.call(self, other);
}

// aten::logical_or_(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & logical_or_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create_logical_or__typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(blackman_window, name, "aten::blackman_window")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(blackman_window, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(blackman_window, schema_str, "blackman_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::blackman_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<blackman_window::schema> create_blackman_window_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(blackman_window::name, blackman_window::overload_name)
      .typed<blackman_window::schema>();
}

// aten::blackman_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor blackman_window::call(int64_t window_length, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_blackman_window_typed_handle();
    return op.call(window_length, dtype, layout, device, pin_memory);
}

// aten::blackman_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor blackman_window::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t window_length, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_blackman_window_typed_handle();
    return op.redispatch(dispatchKeySet, window_length, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(broadcast_tensors, name, "aten::broadcast_tensors")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(broadcast_tensors, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(broadcast_tensors, schema_str, "broadcast_tensors(Tensor[] tensors) -> Tensor[]")

// aten::broadcast_tensors(Tensor[] tensors) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<broadcast_tensors::schema> create_broadcast_tensors_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(broadcast_tensors::name, broadcast_tensors::overload_name)
      .typed<broadcast_tensors::schema>();
}

// aten::broadcast_tensors(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> broadcast_tensors::call(at::TensorList tensors) {
    static auto op = create_broadcast_tensors_typed_handle();
    return op.call(tensors);
}

// aten::broadcast_tensors(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> broadcast_tensors::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create_broadcast_tensors_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cat, name, "aten::cat")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cat, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cat, schema_str, "cat(Tensor[] tensors, int dim=0) -> Tensor")

// aten::cat(Tensor[] tensors, int dim=0) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<cat::schema> create_cat_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cat::name, cat::overload_name)
      .typed<cat::schema>();
}

// aten::cat(Tensor[] tensors, int dim=0) -> Tensor
at::Tensor cat::call(at::TensorList tensors, int64_t dim) {
    static auto op = create_cat_typed_handle();
    return op.call(tensors, dim);
}

// aten::cat(Tensor[] tensors, int dim=0) -> Tensor
at::Tensor cat::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, int64_t dim) {
    static auto op = create_cat_typed_handle();
    return op.redispatch(dispatchKeySet, tensors, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tensor_split_indices, name, "aten::tensor_split")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tensor_split_indices, overload_name, "indices")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tensor_split_indices, schema_str, "tensor_split.indices(Tensor(a) self, int[] indices, int dim=0) -> Tensor(a)[]")

// aten::tensor_split.indices(Tensor(a) self, int[] indices, int dim=0) -> Tensor(a)[]
static C10_NOINLINE c10::TypedOperatorHandle<tensor_split_indices::schema> create_tensor_split_indices_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(tensor_split_indices::name, tensor_split_indices::overload_name)
      .typed<tensor_split_indices::schema>();
}

// aten::tensor_split.indices(Tensor(a) self, int[] indices, int dim=0) -> Tensor(a)[]
::std::vector<at::Tensor> tensor_split_indices::call(const at::Tensor & self, at::IntArrayRef indices, int64_t dim) {
    static auto op = create_tensor_split_indices_typed_handle();
    return op.call(self, indices, dim);
}

// aten::tensor_split.indices(Tensor(a) self, int[] indices, int dim=0) -> Tensor(a)[]
::std::vector<at::Tensor> tensor_split_indices::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef indices, int64_t dim) {
    static auto op = create_tensor_split_indices_typed_handle();
    return op.redispatch(dispatchKeySet, self, indices, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_Tensor, name, "aten::clamp")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_Tensor, schema_str, "clamp.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor")

// aten::clamp.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<clamp_Tensor::schema> create_clamp_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(clamp_Tensor::name, clamp_Tensor::overload_name)
      .typed<clamp_Tensor::schema>();
}

// aten::clamp.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor
at::Tensor clamp_Tensor::call(const at::Tensor & self, const c10::optional<at::Tensor> & min, const c10::optional<at::Tensor> & max) {
    static auto op = create_clamp_Tensor_typed_handle();
    return op.call(self, min, max);
}

// aten::clamp.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor
at::Tensor clamp_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Tensor> & min, const c10::optional<at::Tensor> & max) {
    static auto op = create_clamp_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, min, max);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_, name, "aten::clamp_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_, schema_str, "clamp_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor(a!)")

// aten::clamp_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<clamp_::schema> create_clamp__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(clamp_::name, clamp_::overload_name)
      .typed<clamp_::schema>();
}

// aten::clamp_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor(a!)
at::Tensor & clamp_::call(at::Tensor & self, const c10::optional<at::Scalar> & min, const c10::optional<at::Scalar> & max) {
    static auto op = create_clamp__typed_handle();
    return op.call(self, min, max);
}

// aten::clamp_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor(a!)
at::Tensor & clamp_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const c10::optional<at::Scalar> & min, const c10::optional<at::Scalar> & max) {
    static auto op = create_clamp__typed_handle();
    return op.redispatch(dispatchKeySet, self, min, max);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp__Tensor, name, "aten::clamp_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp__Tensor, schema_str, "clamp_.Tensor(Tensor(a!) self, Tensor? min=None, Tensor? max=None) -> Tensor(a!)")

// aten::clamp_.Tensor(Tensor(a!) self, Tensor? min=None, Tensor? max=None) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<clamp__Tensor::schema> create_clamp__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(clamp__Tensor::name, clamp__Tensor::overload_name)
      .typed<clamp__Tensor::schema>();
}

// aten::clamp_.Tensor(Tensor(a!) self, Tensor? min=None, Tensor? max=None) -> Tensor(a!)
at::Tensor & clamp__Tensor::call(at::Tensor & self, const c10::optional<at::Tensor> & min, const c10::optional<at::Tensor> & max) {
    static auto op = create_clamp__Tensor_typed_handle();
    return op.call(self, min, max);
}

// aten::clamp_.Tensor(Tensor(a!) self, Tensor? min=None, Tensor? max=None) -> Tensor(a!)
at::Tensor & clamp__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const c10::optional<at::Tensor> & min, const c10::optional<at::Tensor> & max) {
    static auto op = create_clamp__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, min, max);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_min_, name, "aten::clamp_min_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_min_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_min_, schema_str, "clamp_min_(Tensor(a!) self, Scalar min) -> Tensor(a!)")

// aten::clamp_min_(Tensor(a!) self, Scalar min) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<clamp_min_::schema> create_clamp_min__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(clamp_min_::name, clamp_min_::overload_name)
      .typed<clamp_min_::schema>();
}

// aten::clamp_min_(Tensor(a!) self, Scalar min) -> Tensor(a!)
at::Tensor & clamp_min_::call(at::Tensor & self, const at::Scalar & min) {
    static auto op = create_clamp_min__typed_handle();
    return op.call(self, min);
}

// aten::clamp_min_(Tensor(a!) self, Scalar min) -> Tensor(a!)
at::Tensor & clamp_min_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & min) {
    static auto op = create_clamp_min__typed_handle();
    return op.redispatch(dispatchKeySet, self, min);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clip_Tensor, name, "aten::clip")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clip_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clip_Tensor, schema_str, "clip.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor")

// aten::clip.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<clip_Tensor::schema> create_clip_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(clip_Tensor::name, clip_Tensor::overload_name)
      .typed<clip_Tensor::schema>();
}

// aten::clip.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor
at::Tensor clip_Tensor::call(const at::Tensor & self, const c10::optional<at::Tensor> & min, const c10::optional<at::Tensor> & max) {
    static auto op = create_clip_Tensor_typed_handle();
    return op.call(self, min, max);
}

// aten::clip.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor
at::Tensor clip_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Tensor> & min, const c10::optional<at::Tensor> & max) {
    static auto op = create_clip_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, min, max);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clip_Tensor_out, name, "aten::clip")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clip_Tensor_out, overload_name, "Tensor_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clip_Tensor_out, schema_str, "clip.Tensor_out(Tensor self, Tensor? min=None, Tensor? max=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::clip.Tensor_out(Tensor self, Tensor? min=None, Tensor? max=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<clip_Tensor_out::schema> create_clip_Tensor_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(clip_Tensor_out::name, clip_Tensor_out::overload_name)
      .typed<clip_Tensor_out::schema>();
}

// aten::clip.Tensor_out(Tensor self, Tensor? min=None, Tensor? max=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & clip_Tensor_out::call(const at::Tensor & self, const c10::optional<at::Tensor> & min, const c10::optional<at::Tensor> & max, at::Tensor & out) {
    static auto op = create_clip_Tensor_out_typed_handle();
    return op.call(self, min, max, out);
}

// aten::clip.Tensor_out(Tensor self, Tensor? min=None, Tensor? max=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & clip_Tensor_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Tensor> & min, const c10::optional<at::Tensor> & max, at::Tensor & out) {
    static auto op = create_clip_Tensor_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, min, max, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(convolution, name, "aten::convolution")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(convolution, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(convolution, schema_str, "convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor")

// aten::convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<convolution::schema> create_convolution_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(convolution::name, convolution::overload_name)
      .typed<convolution::schema>();
}

// aten::convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor
at::Tensor convolution::call(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups) {
    static auto op = create_convolution_typed_handle();
    return op.call(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
}

// aten::convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor
at::Tensor convolution::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups) {
    static auto op = create_convolution_typed_handle();
    return op.redispatch(dispatchKeySet, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(convolution_backward_overrideable, name, "aten::convolution_backward_overrideable")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(convolution_backward_overrideable, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(convolution_backward_overrideable, schema_str, "convolution_backward_overrideable(Tensor grad_output, Tensor input, Tensor weight, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)")

// aten::convolution_backward_overrideable(Tensor grad_output, Tensor input, Tensor weight, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)
static C10_NOINLINE c10::TypedOperatorHandle<convolution_backward_overrideable::schema> create_convolution_backward_overrideable_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(convolution_backward_overrideable::name, convolution_backward_overrideable::overload_name)
      .typed<convolution_backward_overrideable::schema>();
}

// aten::convolution_backward_overrideable(Tensor grad_output, Tensor input, Tensor weight, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> convolution_backward_overrideable::call(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, ::std::array<bool,3> output_mask) {
    static auto op = create_convolution_backward_overrideable_typed_handle();
    return op.call(grad_output, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask);
}

// aten::convolution_backward_overrideable(Tensor grad_output, Tensor input, Tensor weight, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> convolution_backward_overrideable::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, ::std::array<bool,3> output_mask) {
    static auto op = create_convolution_backward_overrideable_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_convolution, name, "aten::_convolution")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_convolution, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_convolution, schema_str, "_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) -> Tensor")

// aten::_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_convolution::schema> create__convolution_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_convolution::name, _convolution::overload_name)
      .typed<_convolution::schema>();
}

// aten::_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) -> Tensor
at::Tensor _convolution::call(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) {
    static auto op = create__convolution_typed_handle();
    return op.call(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32);
}

// aten::_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) -> Tensor
at::Tensor _convolution::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) {
    static auto op = create__convolution_typed_handle();
    return op.redispatch(dispatchKeySet, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv_transpose1d, name, "aten::conv_transpose1d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv_transpose1d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv_transpose1d, schema_str, "conv_transpose1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] output_padding=0, int groups=1, int[1] dilation=1) -> Tensor")

// aten::conv_transpose1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] output_padding=0, int groups=1, int[1] dilation=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<conv_transpose1d::schema> create_conv_transpose1d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(conv_transpose1d::name, conv_transpose1d::overload_name)
      .typed<conv_transpose1d::schema>();
}

// aten::conv_transpose1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] output_padding=0, int groups=1, int[1] dilation=1) -> Tensor
at::Tensor conv_transpose1d::call(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, int64_t groups, at::IntArrayRef dilation) {
    static auto op = create_conv_transpose1d_typed_handle();
    return op.call(input, weight, bias, stride, padding, output_padding, groups, dilation);
}

// aten::conv_transpose1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] output_padding=0, int groups=1, int[1] dilation=1) -> Tensor
at::Tensor conv_transpose1d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, int64_t groups, at::IntArrayRef dilation) {
    static auto op = create_conv_transpose1d_typed_handle();
    return op.redispatch(dispatchKeySet, input, weight, bias, stride, padding, output_padding, groups, dilation);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cos, name, "aten::cos")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cos, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cos, schema_str, "cos(Tensor self) -> Tensor")

// aten::cos(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<cos::schema> create_cos_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cos::name, cos::overload_name)
      .typed<cos::schema>();
}

// aten::cos(Tensor self) -> Tensor
at::Tensor cos::call(const at::Tensor & self) {
    static auto op = create_cos_typed_handle();
    return op.call(self);
}

// aten::cos(Tensor self) -> Tensor
at::Tensor cos::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_cos_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cos_out, name, "aten::cos")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cos_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cos_out, schema_str, "cos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::cos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<cos_out::schema> create_cos_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cos_out::name, cos_out::overload_name)
      .typed<cos_out::schema>();
}

// aten::cos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & cos_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_cos_out_typed_handle();
    return op.call(self, out);
}

// aten::cos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & cos_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_cos_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_affine_grid_generator, name, "aten::cudnn_affine_grid_generator")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_affine_grid_generator, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_affine_grid_generator, schema_str, "cudnn_affine_grid_generator(Tensor theta, int N, int C, int H, int W) -> Tensor grid")

// aten::cudnn_affine_grid_generator(Tensor theta, int N, int C, int H, int W) -> Tensor grid
static C10_NOINLINE c10::TypedOperatorHandle<cudnn_affine_grid_generator::schema> create_cudnn_affine_grid_generator_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cudnn_affine_grid_generator::name, cudnn_affine_grid_generator::overload_name)
      .typed<cudnn_affine_grid_generator::schema>();
}

// aten::cudnn_affine_grid_generator(Tensor theta, int N, int C, int H, int W) -> Tensor grid
at::Tensor cudnn_affine_grid_generator::call(const at::Tensor & theta, int64_t N, int64_t C, int64_t H, int64_t W) {
    static auto op = create_cudnn_affine_grid_generator_typed_handle();
    return op.call(theta, N, C, H, W);
}

// aten::cudnn_affine_grid_generator(Tensor theta, int N, int C, int H, int W) -> Tensor grid
at::Tensor cudnn_affine_grid_generator::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & theta, int64_t N, int64_t C, int64_t H, int64_t W) {
    static auto op = create_cudnn_affine_grid_generator_typed_handle();
    return op.redispatch(dispatchKeySet, theta, N, C, H, W);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_batch_norm_backward, name, "aten::cudnn_batch_norm_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_batch_norm_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_batch_norm_backward, schema_str, "cudnn_batch_norm_backward(Tensor input, Tensor grad_output, Tensor weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var, float epsilon, Tensor reserveSpace) -> (Tensor, Tensor, Tensor)")

// aten::cudnn_batch_norm_backward(Tensor input, Tensor grad_output, Tensor weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var, float epsilon, Tensor reserveSpace) -> (Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<cudnn_batch_norm_backward::schema> create_cudnn_batch_norm_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cudnn_batch_norm_backward::name, cudnn_batch_norm_backward::overload_name)
      .typed<cudnn_batch_norm_backward::schema>();
}

// aten::cudnn_batch_norm_backward(Tensor input, Tensor grad_output, Tensor weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var, float epsilon, Tensor reserveSpace) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> cudnn_batch_norm_backward::call(const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, const c10::optional<at::Tensor> & save_mean, const c10::optional<at::Tensor> & save_var, double epsilon, const at::Tensor & reserveSpace) {
    static auto op = create_cudnn_batch_norm_backward_typed_handle();
    return op.call(input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon, reserveSpace);
}

// aten::cudnn_batch_norm_backward(Tensor input, Tensor grad_output, Tensor weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var, float epsilon, Tensor reserveSpace) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> cudnn_batch_norm_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, const c10::optional<at::Tensor> & save_mean, const c10::optional<at::Tensor> & save_var, double epsilon, const at::Tensor & reserveSpace) {
    static auto op = create_cudnn_batch_norm_backward_typed_handle();
    return op.redispatch(dispatchKeySet, input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon, reserveSpace);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_deprecated, name, "aten::cudnn_convolution")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_deprecated, overload_name, "deprecated")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_deprecated, schema_str, "cudnn_convolution.deprecated(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor")

// aten::cudnn_convolution.deprecated(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<cudnn_convolution_deprecated::schema> create_cudnn_convolution_deprecated_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cudnn_convolution_deprecated::name, cudnn_convolution_deprecated::overload_name)
      .typed<cudnn_convolution_deprecated::schema>();
}

// aten::cudnn_convolution.deprecated(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
at::Tensor cudnn_convolution_deprecated::call(const at::Tensor & self, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
    static auto op = create_cudnn_convolution_deprecated_typed_handle();
    return op.call(self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
}

// aten::cudnn_convolution.deprecated(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
at::Tensor cudnn_convolution_deprecated::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
    static auto op = create_cudnn_convolution_deprecated_typed_handle();
    return op.redispatch(dispatchKeySet, self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_transpose_deprecated, name, "aten::cudnn_convolution_transpose")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_transpose_deprecated, overload_name, "deprecated")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_transpose_deprecated, schema_str, "cudnn_convolution_transpose.deprecated(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor")

// aten::cudnn_convolution_transpose.deprecated(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<cudnn_convolution_transpose_deprecated::schema> create_cudnn_convolution_transpose_deprecated_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cudnn_convolution_transpose_deprecated::name, cudnn_convolution_transpose_deprecated::overload_name)
      .typed<cudnn_convolution_transpose_deprecated::schema>();
}

// aten::cudnn_convolution_transpose.deprecated(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
at::Tensor cudnn_convolution_transpose_deprecated::call(const at::Tensor & self, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
    static auto op = create_cudnn_convolution_transpose_deprecated_typed_handle();
    return op.call(self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
}

// aten::cudnn_convolution_transpose.deprecated(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
at::Tensor cudnn_convolution_transpose_deprecated::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
    static auto op = create_cudnn_convolution_transpose_deprecated_typed_handle();
    return op.redispatch(dispatchKeySet, self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_transpose, name, "aten::cudnn_convolution_transpose")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_transpose, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_transpose, schema_str, "cudnn_convolution_transpose(Tensor self, Tensor weight, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool allow_tf32) -> Tensor")

// aten::cudnn_convolution_transpose(Tensor self, Tensor weight, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool allow_tf32) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<cudnn_convolution_transpose::schema> create_cudnn_convolution_transpose_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cudnn_convolution_transpose::name, cudnn_convolution_transpose::overload_name)
      .typed<cudnn_convolution_transpose::schema>();
}

// aten::cudnn_convolution_transpose(Tensor self, Tensor weight, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool allow_tf32) -> Tensor
at::Tensor cudnn_convolution_transpose::call(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
    static auto op = create_cudnn_convolution_transpose_typed_handle();
    return op.call(self, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
}

// aten::cudnn_convolution_transpose(Tensor self, Tensor weight, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool allow_tf32) -> Tensor
at::Tensor cudnn_convolution_transpose::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
    static auto op = create_cudnn_convolution_transpose_typed_handle();
    return op.redispatch(dispatchKeySet, self, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_grid_sampler_backward, name, "aten::cudnn_grid_sampler_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_grid_sampler_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_grid_sampler_backward, schema_str, "cudnn_grid_sampler_backward(Tensor self, Tensor grid, Tensor grad_output) -> (Tensor grad_self, Tensor grad_grid)")

// aten::cudnn_grid_sampler_backward(Tensor self, Tensor grid, Tensor grad_output) -> (Tensor grad_self, Tensor grad_grid)
static C10_NOINLINE c10::TypedOperatorHandle<cudnn_grid_sampler_backward::schema> create_cudnn_grid_sampler_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cudnn_grid_sampler_backward::name, cudnn_grid_sampler_backward::overload_name)
      .typed<cudnn_grid_sampler_backward::schema>();
}

// aten::cudnn_grid_sampler_backward(Tensor self, Tensor grid, Tensor grad_output) -> (Tensor grad_self, Tensor grad_grid)
::std::tuple<at::Tensor,at::Tensor> cudnn_grid_sampler_backward::call(const at::Tensor & self, const at::Tensor & grid, const at::Tensor & grad_output) {
    static auto op = create_cudnn_grid_sampler_backward_typed_handle();
    return op.call(self, grid, grad_output);
}

// aten::cudnn_grid_sampler_backward(Tensor self, Tensor grid, Tensor grad_output) -> (Tensor grad_self, Tensor grad_grid)
::std::tuple<at::Tensor,at::Tensor> cudnn_grid_sampler_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & grid, const at::Tensor & grad_output) {
    static auto op = create_cudnn_grid_sampler_backward_typed_handle();
    return op.redispatch(dispatchKeySet, self, grid, grad_output);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cummin_dimname_out, name, "aten::cummin")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cummin_dimname_out, overload_name, "dimname_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cummin_dimname_out, schema_str, "cummin.dimname_out(Tensor self, Dimname dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)")

// aten::cummin.dimname_out(Tensor self, Dimname dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
static C10_NOINLINE c10::TypedOperatorHandle<cummin_dimname_out::schema> create_cummin_dimname_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cummin_dimname_out::name, cummin_dimname_out::overload_name)
      .typed<cummin_dimname_out::schema>();
}

// aten::cummin.dimname_out(Tensor self, Dimname dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> cummin_dimname_out::call(const at::Tensor & self, at::Dimname dim, at::Tensor & values, at::Tensor & indices) {
    static auto op = create_cummin_dimname_out_typed_handle();
    return op.call(self, dim, values, indices);
}

// aten::cummin.dimname_out(Tensor self, Dimname dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> cummin_dimname_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, at::Tensor & values, at::Tensor & indices) {
    static auto op = create_cummin_dimname_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, values, indices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumprod__dimname, name, "aten::cumprod_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumprod__dimname, overload_name, "dimname")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumprod__dimname, schema_str, "cumprod_.dimname(Tensor(a!) self, Dimname dim, *, ScalarType? dtype=None) -> Tensor(a!)")

// aten::cumprod_.dimname(Tensor(a!) self, Dimname dim, *, ScalarType? dtype=None) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<cumprod__dimname::schema> create_cumprod__dimname_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cumprod__dimname::name, cumprod__dimname::overload_name)
      .typed<cumprod__dimname::schema>();
}

// aten::cumprod_.dimname(Tensor(a!) self, Dimname dim, *, ScalarType? dtype=None) -> Tensor(a!)
at::Tensor & cumprod__dimname::call(at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_cumprod__dimname_typed_handle();
    return op.call(self, dim, dtype);
}

// aten::cumprod_.dimname(Tensor(a!) self, Dimname dim, *, ScalarType? dtype=None) -> Tensor(a!)
at::Tensor & cumprod__dimname::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_cumprod__dimname_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumsum, name, "aten::cumsum")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumsum, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumsum, schema_str, "cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor")

// aten::cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<cumsum::schema> create_cumsum_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cumsum::name, cumsum::overload_name)
      .typed<cumsum::schema>();
}

// aten::cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
at::Tensor cumsum::call(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_cumsum_typed_handle();
    return op.call(self, dim, dtype);
}

// aten::cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
at::Tensor cumsum::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_cumsum_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumsum_dimname_out, name, "aten::cumsum")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumsum_dimname_out, overload_name, "dimname_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumsum_dimname_out, schema_str, "cumsum.dimname_out(Tensor self, Dimname dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)")

// aten::cumsum.dimname_out(Tensor self, Dimname dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<cumsum_dimname_out::schema> create_cumsum_dimname_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cumsum_dimname_out::name, cumsum_dimname_out::overload_name)
      .typed<cumsum_dimname_out::schema>();
}

// aten::cumsum.dimname_out(Tensor self, Dimname dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & cumsum_dimname_out::call(const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
    static auto op = create_cumsum_dimname_out_typed_handle();
    return op.call(self, dim, dtype, out);
}

// aten::cumsum.dimname_out(Tensor self, Dimname dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & cumsum_dimname_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
    static auto op = create_cumsum_dimname_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, dtype, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ctc_loss_Tensor, name, "aten::ctc_loss")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ctc_loss_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ctc_loss_Tensor, schema_str, "ctc_loss.Tensor(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, int blank=0, int reduction=Mean, bool zero_infinity=False) -> Tensor")

// aten::ctc_loss.Tensor(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, int blank=0, int reduction=Mean, bool zero_infinity=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<ctc_loss_Tensor::schema> create_ctc_loss_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(ctc_loss_Tensor::name, ctc_loss_Tensor::overload_name)
      .typed<ctc_loss_Tensor::schema>();
}

// aten::ctc_loss.Tensor(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, int blank=0, int reduction=Mean, bool zero_infinity=False) -> Tensor
at::Tensor ctc_loss_Tensor::call(const at::Tensor & log_probs, const at::Tensor & targets, const at::Tensor & input_lengths, const at::Tensor & target_lengths, int64_t blank, int64_t reduction, bool zero_infinity) {
    static auto op = create_ctc_loss_Tensor_typed_handle();
    return op.call(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
}

// aten::ctc_loss.Tensor(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, int blank=0, int reduction=Mean, bool zero_infinity=False) -> Tensor
at::Tensor ctc_loss_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & log_probs, const at::Tensor & targets, const at::Tensor & input_lengths, const at::Tensor & target_lengths, int64_t blank, int64_t reduction, bool zero_infinity) {
    static auto op = create_ctc_loss_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_ctc_loss, name, "aten::_ctc_loss")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_ctc_loss, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_ctc_loss, schema_str, "_ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank=0, bool zero_infinity=False) -> (Tensor, Tensor)")

// aten::_ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank=0, bool zero_infinity=False) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<_ctc_loss::schema> create__ctc_loss_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_ctc_loss::name, _ctc_loss::overload_name)
      .typed<_ctc_loss::schema>();
}

// aten::_ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank=0, bool zero_infinity=False) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> _ctc_loss::call(const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, int64_t blank, bool zero_infinity) {
    static auto op = create__ctc_loss_typed_handle();
    return op.call(log_probs, targets, input_lengths, target_lengths, blank, zero_infinity);
}

// aten::_ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank=0, bool zero_infinity=False) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> _ctc_loss::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, int64_t blank, bool zero_infinity) {
    static auto op = create__ctc_loss_typed_handle();
    return op.redispatch(dispatchKeySet, log_probs, targets, input_lengths, target_lengths, blank, zero_infinity);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(diagflat, name, "aten::diagflat")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(diagflat, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(diagflat, schema_str, "diagflat(Tensor self, int offset=0) -> Tensor")

// aten::diagflat(Tensor self, int offset=0) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<diagflat::schema> create_diagflat_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(diagflat::name, diagflat::overload_name)
      .typed<diagflat::schema>();
}

// aten::diagflat(Tensor self, int offset=0) -> Tensor
at::Tensor diagflat::call(const at::Tensor & self, int64_t offset) {
    static auto op = create_diagflat_typed_handle();
    return op.call(self, offset);
}

// aten::diagflat(Tensor self, int offset=0) -> Tensor
at::Tensor diagflat::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t offset) {
    static auto op = create_diagflat_typed_handle();
    return op.redispatch(dispatchKeySet, self, offset);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(div_out, name, "aten::div")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(div_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(div_out, schema_str, "div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<div_out::schema> create_div_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(div_out::name, div_out::overload_name)
      .typed<div_out::schema>();
}

// aten::div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & div_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_div_out_typed_handle();
    return op.call(self, other, out);
}

// aten::div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & div_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_div_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(div_Tensor_mode, name, "aten::div")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(div_Tensor_mode, overload_name, "Tensor_mode")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(div_Tensor_mode, schema_str, "div.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor")

// aten::div.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<div_Tensor_mode::schema> create_div_Tensor_mode_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(div_Tensor_mode::name, div_Tensor_mode::overload_name)
      .typed<div_Tensor_mode::schema>();
}

// aten::div.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor
at::Tensor div_Tensor_mode::call(const at::Tensor & self, const at::Tensor & other, c10::optional<c10::string_view> rounding_mode) {
    static auto op = create_div_Tensor_mode_typed_handle();
    return op.call(self, other, rounding_mode);
}

// aten::div.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor
at::Tensor div_Tensor_mode::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, c10::optional<c10::string_view> rounding_mode) {
    static auto op = create_div_Tensor_mode_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, rounding_mode);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(div__Tensor_mode, name, "aten::div_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(div__Tensor_mode, overload_name, "Tensor_mode")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(div__Tensor_mode, schema_str, "div_.Tensor_mode(Tensor(a!) self, Tensor other, *, str? rounding_mode) -> Tensor(a!)")

// aten::div_.Tensor_mode(Tensor(a!) self, Tensor other, *, str? rounding_mode) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<div__Tensor_mode::schema> create_div__Tensor_mode_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(div__Tensor_mode::name, div__Tensor_mode::overload_name)
      .typed<div__Tensor_mode::schema>();
}

// aten::div_.Tensor_mode(Tensor(a!) self, Tensor other, *, str? rounding_mode) -> Tensor(a!)
at::Tensor & div__Tensor_mode::call(at::Tensor & self, const at::Tensor & other, c10::optional<c10::string_view> rounding_mode) {
    static auto op = create_div__Tensor_mode_typed_handle();
    return op.call(self, other, rounding_mode);
}

// aten::div_.Tensor_mode(Tensor(a!) self, Tensor other, *, str? rounding_mode) -> Tensor(a!)
at::Tensor & div__Tensor_mode::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other, c10::optional<c10::string_view> rounding_mode) {
    static auto op = create_div__Tensor_mode_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, rounding_mode);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(div_out_mode, name, "aten::div")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(div_out_mode, overload_name, "out_mode")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(div_out_mode, schema_str, "div.out_mode(Tensor self, Tensor other, *, str? rounding_mode, Tensor(a!) out) -> Tensor(a!)")

// aten::div.out_mode(Tensor self, Tensor other, *, str? rounding_mode, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<div_out_mode::schema> create_div_out_mode_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(div_out_mode::name, div_out_mode::overload_name)
      .typed<div_out_mode::schema>();
}

// aten::div.out_mode(Tensor self, Tensor other, *, str? rounding_mode, Tensor(a!) out) -> Tensor(a!)
at::Tensor & div_out_mode::call(const at::Tensor & self, const at::Tensor & other, c10::optional<c10::string_view> rounding_mode, at::Tensor & out) {
    static auto op = create_div_out_mode_typed_handle();
    return op.call(self, other, rounding_mode, out);
}

// aten::div.out_mode(Tensor self, Tensor other, *, str? rounding_mode, Tensor(a!) out) -> Tensor(a!)
at::Tensor & div_out_mode::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, c10::optional<c10::string_view> rounding_mode, at::Tensor & out) {
    static auto op = create_div_out_mode_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, rounding_mode, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(divide_Scalar, name, "aten::divide")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(divide_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(divide_Scalar, schema_str, "divide.Scalar(Tensor self, Scalar other) -> Tensor")

// aten::divide.Scalar(Tensor self, Scalar other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<divide_Scalar::schema> create_divide_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(divide_Scalar::name, divide_Scalar::overload_name)
      .typed<divide_Scalar::schema>();
}

// aten::divide.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor divide_Scalar::call(const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_divide_Scalar_typed_handle();
    return op.call(self, other);
}

// aten::divide.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor divide_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_divide_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(divide__Scalar_mode, name, "aten::divide_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(divide__Scalar_mode, overload_name, "Scalar_mode")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(divide__Scalar_mode, schema_str, "divide_.Scalar_mode(Tensor(a!) self, Scalar other, *, str? rounding_mode) -> Tensor(a!)")

// aten::divide_.Scalar_mode(Tensor(a!) self, Scalar other, *, str? rounding_mode) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<divide__Scalar_mode::schema> create_divide__Scalar_mode_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(divide__Scalar_mode::name, divide__Scalar_mode::overload_name)
      .typed<divide__Scalar_mode::schema>();
}

// aten::divide_.Scalar_mode(Tensor(a!) self, Scalar other, *, str? rounding_mode) -> Tensor(a!)
at::Tensor & divide__Scalar_mode::call(at::Tensor & self, const at::Scalar & other, c10::optional<c10::string_view> rounding_mode) {
    static auto op = create_divide__Scalar_mode_typed_handle();
    return op.call(self, other, rounding_mode);
}

// aten::divide_.Scalar_mode(Tensor(a!) self, Scalar other, *, str? rounding_mode) -> Tensor(a!)
at::Tensor & divide__Scalar_mode::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other, c10::optional<c10::string_view> rounding_mode) {
    static auto op = create_divide__Scalar_mode_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, rounding_mode);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(true_divide__Tensor, name, "aten::true_divide_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(true_divide__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(true_divide__Tensor, schema_str, "true_divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::true_divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<true_divide__Tensor::schema> create_true_divide__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(true_divide__Tensor::name, true_divide__Tensor::overload_name)
      .typed<true_divide__Tensor::schema>();
}

// aten::true_divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & true_divide__Tensor::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create_true_divide__Tensor_typed_handle();
    return op.call(self, other);
}

// aten::true_divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & true_divide__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create_true_divide__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(vdot, name, "aten::vdot")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(vdot, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(vdot, schema_str, "vdot(Tensor self, Tensor other) -> Tensor")

// aten::vdot(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<vdot::schema> create_vdot_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(vdot::name, vdot::overload_name)
      .typed<vdot::schema>();
}

// aten::vdot(Tensor self, Tensor other) -> Tensor
at::Tensor vdot::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_vdot_typed_handle();
    return op.call(self, other);
}

// aten::vdot(Tensor self, Tensor other) -> Tensor
at::Tensor vdot::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_vdot_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(embedding_backward, name, "aten::embedding_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(embedding_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(embedding_backward, schema_str, "embedding_backward(Tensor grad, Tensor indices, int num_weights, int padding_idx, bool scale_grad_by_freq, bool sparse) -> Tensor")

// aten::embedding_backward(Tensor grad, Tensor indices, int num_weights, int padding_idx, bool scale_grad_by_freq, bool sparse) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<embedding_backward::schema> create_embedding_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(embedding_backward::name, embedding_backward::overload_name)
      .typed<embedding_backward::schema>();
}

// aten::embedding_backward(Tensor grad, Tensor indices, int num_weights, int padding_idx, bool scale_grad_by_freq, bool sparse) -> Tensor
at::Tensor embedding_backward::call(const at::Tensor & grad, const at::Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
    static auto op = create_embedding_backward_typed_handle();
    return op.call(grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse);
}

// aten::embedding_backward(Tensor grad, Tensor indices, int num_weights, int padding_idx, bool scale_grad_by_freq, bool sparse) -> Tensor
at::Tensor embedding_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
    static auto op = create_embedding_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(embedding_dense_backward, name, "aten::embedding_dense_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(embedding_dense_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(embedding_dense_backward, schema_str, "embedding_dense_backward(Tensor grad_output, Tensor indices, int num_weights, int padding_idx, bool scale_grad_by_freq) -> Tensor")

// aten::embedding_dense_backward(Tensor grad_output, Tensor indices, int num_weights, int padding_idx, bool scale_grad_by_freq) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<embedding_dense_backward::schema> create_embedding_dense_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(embedding_dense_backward::name, embedding_dense_backward::overload_name)
      .typed<embedding_dense_backward::schema>();
}

// aten::embedding_dense_backward(Tensor grad_output, Tensor indices, int num_weights, int padding_idx, bool scale_grad_by_freq) -> Tensor
at::Tensor embedding_dense_backward::call(const at::Tensor & grad_output, const at::Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {
    static auto op = create_embedding_dense_backward_typed_handle();
    return op.call(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq);
}

// aten::embedding_dense_backward(Tensor grad_output, Tensor indices, int num_weights, int padding_idx, bool scale_grad_by_freq) -> Tensor
at::Tensor embedding_dense_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {
    static auto op = create_embedding_dense_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, indices, num_weights, padding_idx, scale_grad_by_freq);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(row_stack_out, name, "aten::row_stack")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(row_stack_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(row_stack_out, schema_str, "row_stack.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)")

// aten::row_stack.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<row_stack_out::schema> create_row_stack_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(row_stack_out::name, row_stack_out::overload_name)
      .typed<row_stack_out::schema>();
}

// aten::row_stack.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & row_stack_out::call(at::TensorList tensors, at::Tensor & out) {
    static auto op = create_row_stack_out_typed_handle();
    return op.call(tensors, out);
}

// aten::row_stack.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & row_stack_out::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, at::Tensor & out) {
    static auto op = create_row_stack_out_typed_handle();
    return op.redispatch(dispatchKeySet, tensors, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_embedding_bag, name, "aten::_embedding_bag")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_embedding_bag, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_embedding_bag, schema_str, "_embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int padding_idx=-1) -> (Tensor, Tensor, Tensor, Tensor)")

// aten::_embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int padding_idx=-1) -> (Tensor, Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<_embedding_bag::schema> create__embedding_bag_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_embedding_bag::name, _embedding_bag::overload_name)
      .typed<_embedding_bag::schema>();
}

// aten::_embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int padding_idx=-1) -> (Tensor, Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> _embedding_bag::call(const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<at::Tensor> & per_sample_weights, bool include_last_offset, int64_t padding_idx) {
    static auto op = create__embedding_bag_typed_handle();
    return op.call(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx);
}

// aten::_embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int padding_idx=-1) -> (Tensor, Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> _embedding_bag::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<at::Tensor> & per_sample_weights, bool include_last_offset, int64_t padding_idx) {
    static auto op = create__embedding_bag_typed_handle();
    return op.redispatch(dispatchKeySet, weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_embedding_bag_sparse_backward, name, "aten::_embedding_bag_sparse_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_embedding_bag_sparse_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_embedding_bag_sparse_backward, schema_str, "_embedding_bag_sparse_backward(Tensor grad, Tensor indices, Tensor offsets, Tensor offset2bag, Tensor bag_size, int num_weights, bool scale_grad_by_freq, int mode, Tensor? per_sample_weights, int padding_idx=-1) -> Tensor")

// aten::_embedding_bag_sparse_backward(Tensor grad, Tensor indices, Tensor offsets, Tensor offset2bag, Tensor bag_size, int num_weights, bool scale_grad_by_freq, int mode, Tensor? per_sample_weights, int padding_idx=-1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_embedding_bag_sparse_backward::schema> create__embedding_bag_sparse_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_embedding_bag_sparse_backward::name, _embedding_bag_sparse_backward::overload_name)
      .typed<_embedding_bag_sparse_backward::schema>();
}

// aten::_embedding_bag_sparse_backward(Tensor grad, Tensor indices, Tensor offsets, Tensor offset2bag, Tensor bag_size, int num_weights, bool scale_grad_by_freq, int mode, Tensor? per_sample_weights, int padding_idx=-1) -> Tensor
at::Tensor _embedding_bag_sparse_backward::call(const at::Tensor & grad, const at::Tensor & indices, const at::Tensor & offsets, const at::Tensor & offset2bag, const at::Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, const c10::optional<at::Tensor> & per_sample_weights, int64_t padding_idx) {
    static auto op = create__embedding_bag_sparse_backward_typed_handle();
    return op.call(grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode, per_sample_weights, padding_idx);
}

// aten::_embedding_bag_sparse_backward(Tensor grad, Tensor indices, Tensor offsets, Tensor offset2bag, Tensor bag_size, int num_weights, bool scale_grad_by_freq, int mode, Tensor? per_sample_weights, int padding_idx=-1) -> Tensor
at::Tensor _embedding_bag_sparse_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & indices, const at::Tensor & offsets, const at::Tensor & offset2bag, const at::Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, const c10::optional<at::Tensor> & per_sample_weights, int64_t padding_idx) {
    static auto op = create__embedding_bag_sparse_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode, per_sample_weights, padding_idx);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(new_empty, name, "aten::new_empty")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(new_empty, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(new_empty, schema_str, "new_empty(Tensor self, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::new_empty(Tensor self, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<new_empty::schema> create_new_empty_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(new_empty::name, new_empty::overload_name)
      .typed<new_empty::schema>();
}

// aten::new_empty(Tensor self, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor new_empty::call(const at::Tensor & self, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_new_empty_typed_handle();
    return op.call(self, size, dtype, layout, device, pin_memory);
}

// aten::new_empty(Tensor self, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor new_empty::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_new_empty_typed_handle();
    return op.redispatch(dispatchKeySet, self, size, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(resize_, name, "aten::resize_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(resize_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(resize_, schema_str, "resize_(Tensor(a!) self, int[] size, *, MemoryFormat? memory_format=None) -> Tensor(a!)")

// aten::resize_(Tensor(a!) self, int[] size, *, MemoryFormat? memory_format=None) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<resize_::schema> create_resize__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(resize_::name, resize_::overload_name)
      .typed<resize_::schema>();
}

// aten::resize_(Tensor(a!) self, int[] size, *, MemoryFormat? memory_format=None) -> Tensor(a!)
const at::Tensor & resize_::call(const at::Tensor & self, at::IntArrayRef size, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create_resize__typed_handle();
    return op.call(self, size, memory_format);
}

// aten::resize_(Tensor(a!) self, int[] size, *, MemoryFormat? memory_format=None) -> Tensor(a!)
const at::Tensor & resize_::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef size, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create_resize__typed_handle();
    return op.redispatch(dispatchKeySet, self, size, memory_format);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(exp_, name, "aten::exp_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(exp_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(exp_, schema_str, "exp_(Tensor(a!) self) -> Tensor(a!)")

// aten::exp_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<exp_::schema> create_exp__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(exp_::name, exp_::overload_name)
      .typed<exp_::schema>();
}

// aten::exp_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & exp_::call(at::Tensor & self) {
    static auto op = create_exp__typed_handle();
    return op.call(self);
}

// aten::exp_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & exp_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_exp__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(expm1, name, "aten::expm1")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(expm1, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(expm1, schema_str, "expm1(Tensor self) -> Tensor")

// aten::expm1(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<expm1::schema> create_expm1_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(expm1::name, expm1::overload_name)
      .typed<expm1::schema>();
}

// aten::expm1(Tensor self) -> Tensor
at::Tensor expm1::call(const at::Tensor & self) {
    static auto op = create_expm1_typed_handle();
    return op.call(self);
}

// aten::expm1(Tensor self) -> Tensor
at::Tensor expm1::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_expm1_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(expand_as, name, "aten::expand_as")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(expand_as, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(expand_as, schema_str, "expand_as(Tensor(a) self, Tensor other) -> Tensor(a)")

// aten::expand_as(Tensor(a) self, Tensor other) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<expand_as::schema> create_expand_as_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(expand_as::name, expand_as::overload_name)
      .typed<expand_as::schema>();
}

// aten::expand_as(Tensor(a) self, Tensor other) -> Tensor(a)
at::Tensor expand_as::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_expand_as_typed_handle();
    return op.call(self, other);
}

// aten::expand_as(Tensor(a) self, Tensor other) -> Tensor(a)
at::Tensor expand_as::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_expand_as_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(flatten_DimnameList, name, "aten::flatten")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(flatten_DimnameList, overload_name, "DimnameList")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(flatten_DimnameList, schema_str, "flatten.DimnameList(Tensor(a) self, Dimname[] dims, Dimname out_dim) -> Tensor(a)")

// aten::flatten.DimnameList(Tensor(a) self, Dimname[] dims, Dimname out_dim) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<flatten_DimnameList::schema> create_flatten_DimnameList_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(flatten_DimnameList::name, flatten_DimnameList::overload_name)
      .typed<flatten_DimnameList::schema>();
}

// aten::flatten.DimnameList(Tensor(a) self, Dimname[] dims, Dimname out_dim) -> Tensor(a)
at::Tensor flatten_DimnameList::call(const at::Tensor & self, at::DimnameList dims, at::Dimname out_dim) {
    static auto op = create_flatten_DimnameList_typed_handle();
    return op.call(self, dims, out_dim);
}

// aten::flatten.DimnameList(Tensor(a) self, Dimname[] dims, Dimname out_dim) -> Tensor(a)
at::Tensor flatten_DimnameList::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList dims, at::Dimname out_dim) {
    static auto op = create_flatten_DimnameList_typed_handle();
    return op.redispatch(dispatchKeySet, self, dims, out_dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fill__Scalar, name, "aten::fill_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fill__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fill__Scalar, schema_str, "fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)")

// aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<fill__Scalar::schema> create_fill__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fill__Scalar::name, fill__Scalar::overload_name)
      .typed<fill__Scalar::schema>();
}

// aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)
at::Tensor & fill__Scalar::call(at::Tensor & self, const at::Scalar & value) {
    static auto op = create_fill__Scalar_typed_handle();
    return op.call(self, value);
}

// aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)
at::Tensor & fill__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & value) {
    static auto op = create_fill__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, value);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fill__Tensor, name, "aten::fill_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fill__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fill__Tensor, schema_str, "fill_.Tensor(Tensor(a!) self, Tensor value) -> Tensor(a!)")

// aten::fill_.Tensor(Tensor(a!) self, Tensor value) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<fill__Tensor::schema> create_fill__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fill__Tensor::name, fill__Tensor::overload_name)
      .typed<fill__Tensor::schema>();
}

// aten::fill_.Tensor(Tensor(a!) self, Tensor value) -> Tensor(a!)
at::Tensor & fill__Tensor::call(at::Tensor & self, const at::Tensor & value) {
    static auto op = create_fill__Tensor_typed_handle();
    return op.call(self, value);
}

// aten::fill_.Tensor(Tensor(a!) self, Tensor value) -> Tensor(a!)
at::Tensor & fill__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & value) {
    static auto op = create_fill__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, value);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(floor_, name, "aten::floor_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(floor_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(floor_, schema_str, "floor_(Tensor(a!) self) -> Tensor(a!)")

// aten::floor_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<floor_::schema> create_floor__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(floor_::name, floor_::overload_name)
      .typed<floor_::schema>();
}

// aten::floor_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & floor_::call(at::Tensor & self) {
    static auto op = create_floor__typed_handle();
    return op.call(self);
}

// aten::floor_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & floor_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_floor__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(floor_divide_Scalar, name, "aten::floor_divide")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(floor_divide_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(floor_divide_Scalar, schema_str, "floor_divide.Scalar(Tensor self, Scalar other) -> Tensor")

// aten::floor_divide.Scalar(Tensor self, Scalar other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<floor_divide_Scalar::schema> create_floor_divide_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(floor_divide_Scalar::name, floor_divide_Scalar::overload_name)
      .typed<floor_divide_Scalar::schema>();
}

// aten::floor_divide.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor floor_divide_Scalar::call(const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_floor_divide_Scalar_typed_handle();
    return op.call(self, other);
}

// aten::floor_divide.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor floor_divide_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_floor_divide_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(full_out, name, "aten::full")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(full_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(full_out, schema_str, "full.out(int[] size, Scalar fill_value, *, Tensor(a!) out) -> Tensor(a!)")

// aten::full.out(int[] size, Scalar fill_value, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<full_out::schema> create_full_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(full_out::name, full_out::overload_name)
      .typed<full_out::schema>();
}

// aten::full.out(int[] size, Scalar fill_value, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & full_out::call(at::IntArrayRef size, const at::Scalar & fill_value, at::Tensor & out) {
    static auto op = create_full_out_typed_handle();
    return op.call(size, fill_value, out);
}

// aten::full.out(int[] size, Scalar fill_value, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & full_out::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, const at::Scalar & fill_value, at::Tensor & out) {
    static auto op = create_full_out_typed_handle();
    return op.redispatch(dispatchKeySet, size, fill_value, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lcm, name, "aten::lcm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lcm, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lcm, schema_str, "lcm(Tensor self, Tensor other) -> Tensor")

// aten::lcm(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<lcm::schema> create_lcm_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(lcm::name, lcm::overload_name)
      .typed<lcm::schema>();
}

// aten::lcm(Tensor self, Tensor other) -> Tensor
at::Tensor lcm::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_lcm_typed_handle();
    return op.call(self, other);
}

// aten::lcm(Tensor self, Tensor other) -> Tensor
at::Tensor lcm::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_lcm_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lcm_, name, "aten::lcm_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lcm_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lcm_, schema_str, "lcm_(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::lcm_(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<lcm_::schema> create_lcm__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(lcm_::name, lcm_::overload_name)
      .typed<lcm_::schema>();
}

// aten::lcm_(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & lcm_::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create_lcm__typed_handle();
    return op.call(self, other);
}

// aten::lcm_(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & lcm_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create_lcm__typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(grid_sampler_2d_backward, name, "aten::grid_sampler_2d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(grid_sampler_2d_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(grid_sampler_2d_backward, schema_str, "grid_sampler_2d_backward(Tensor grad_output, Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> (Tensor, Tensor)")

// aten::grid_sampler_2d_backward(Tensor grad_output, Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<grid_sampler_2d_backward::schema> create_grid_sampler_2d_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(grid_sampler_2d_backward::name, grid_sampler_2d_backward::overload_name)
      .typed<grid_sampler_2d_backward::schema>();
}

// aten::grid_sampler_2d_backward(Tensor grad_output, Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> grid_sampler_2d_backward::call(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
    static auto op = create_grid_sampler_2d_backward_typed_handle();
    return op.call(grad_output, input, grid, interpolation_mode, padding_mode, align_corners);
}

// aten::grid_sampler_2d_backward(Tensor grad_output, Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> grid_sampler_2d_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
    static auto op = create_grid_sampler_2d_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, input, grid, interpolation_mode, padding_mode, align_corners);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(kaiser_window_beta, name, "aten::kaiser_window")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(kaiser_window_beta, overload_name, "beta")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(kaiser_window_beta, schema_str, "kaiser_window.beta(int window_length, bool periodic, float beta, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::kaiser_window.beta(int window_length, bool periodic, float beta, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<kaiser_window_beta::schema> create_kaiser_window_beta_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(kaiser_window_beta::name, kaiser_window_beta::overload_name)
      .typed<kaiser_window_beta::schema>();
}

// aten::kaiser_window.beta(int window_length, bool periodic, float beta, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor kaiser_window_beta::call(int64_t window_length, bool periodic, double beta, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_kaiser_window_beta_typed_handle();
    return op.call(window_length, periodic, beta, dtype, layout, device, pin_memory);
}

// aten::kaiser_window.beta(int window_length, bool periodic, float beta, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor kaiser_window_beta::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t window_length, bool periodic, double beta, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_kaiser_window_beta_typed_handle();
    return op.redispatch(dispatchKeySet, window_length, periodic, beta, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(group_norm, name, "aten::group_norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(group_norm, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(group_norm, schema_str, "group_norm(Tensor input, int num_groups, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enabled=True) -> Tensor")

// aten::group_norm(Tensor input, int num_groups, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enabled=True) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<group_norm::schema> create_group_norm_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(group_norm::name, group_norm::overload_name)
      .typed<group_norm::schema>();
}

// aten::group_norm(Tensor input, int num_groups, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enabled=True) -> Tensor
at::Tensor group_norm::call(const at::Tensor & input, int64_t num_groups, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, double eps, bool cudnn_enabled) {
    static auto op = create_group_norm_typed_handle();
    return op.call(input, num_groups, weight, bias, eps, cudnn_enabled);
}

// aten::group_norm(Tensor input, int num_groups, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enabled=True) -> Tensor
at::Tensor group_norm::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, int64_t num_groups, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, double eps, bool cudnn_enabled) {
    static auto op = create_group_norm_typed_handle();
    return op.redispatch(dispatchKeySet, input, num_groups, weight, bias, eps, cudnn_enabled);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fft_r2c_out, name, "aten::_fft_r2c")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fft_r2c_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fft_r2c_out, schema_str, "_fft_r2c.out(Tensor self, int[] dim, int normalization, bool onesided, *, Tensor(a!) out) -> Tensor(a!)")

// aten::_fft_r2c.out(Tensor self, int[] dim, int normalization, bool onesided, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<_fft_r2c_out::schema> create__fft_r2c_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_fft_r2c_out::name, _fft_r2c_out::overload_name)
      .typed<_fft_r2c_out::schema>();
}

// aten::_fft_r2c.out(Tensor self, int[] dim, int normalization, bool onesided, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & _fft_r2c_out::call(const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, bool onesided, at::Tensor & out) {
    static auto op = create__fft_r2c_out_typed_handle();
    return op.call(self, dim, normalization, onesided, out);
}

// aten::_fft_r2c.out(Tensor self, int[] dim, int normalization, bool onesided, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & _fft_r2c_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, bool onesided, at::Tensor & out) {
    static auto op = create__fft_r2c_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, normalization, onesided, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_copy_, name, "aten::index_copy_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_copy_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_copy_, schema_str, "index_copy_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)")

// aten::index_copy_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<index_copy_::schema> create_index_copy__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(index_copy_::name, index_copy_::overload_name)
      .typed<index_copy_::schema>();
}

// aten::index_copy_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)
at::Tensor & index_copy_::call(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source) {
    static auto op = create_index_copy__typed_handle();
    return op.call(self, dim, index, source);
}

// aten::index_copy_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)
at::Tensor & index_copy_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source) {
    static auto op = create_index_copy__typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, source);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_copy, name, "aten::index_copy")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_copy, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_copy, schema_str, "index_copy(Tensor self, int dim, Tensor index, Tensor source) -> Tensor")

// aten::index_copy(Tensor self, int dim, Tensor index, Tensor source) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<index_copy::schema> create_index_copy_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(index_copy::name, index_copy::overload_name)
      .typed<index_copy::schema>();
}

// aten::index_copy(Tensor self, int dim, Tensor index, Tensor source) -> Tensor
at::Tensor index_copy::call(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source) {
    static auto op = create_index_copy_typed_handle();
    return op.call(self, dim, index, source);
}

// aten::index_copy(Tensor self, int dim, Tensor index, Tensor source) -> Tensor
at::Tensor index_copy::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source) {
    static auto op = create_index_copy_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, source);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isin_Tensor_Tensor, name, "aten::isin")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isin_Tensor_Tensor, overload_name, "Tensor_Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isin_Tensor_Tensor, schema_str, "isin.Tensor_Tensor(Tensor elements, Tensor test_elements, *, bool assume_unique=False, bool invert=False) -> Tensor")

// aten::isin.Tensor_Tensor(Tensor elements, Tensor test_elements, *, bool assume_unique=False, bool invert=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<isin_Tensor_Tensor::schema> create_isin_Tensor_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(isin_Tensor_Tensor::name, isin_Tensor_Tensor::overload_name)
      .typed<isin_Tensor_Tensor::schema>();
}

// aten::isin.Tensor_Tensor(Tensor elements, Tensor test_elements, *, bool assume_unique=False, bool invert=False) -> Tensor
at::Tensor isin_Tensor_Tensor::call(const at::Tensor & elements, const at::Tensor & test_elements, bool assume_unique, bool invert) {
    static auto op = create_isin_Tensor_Tensor_typed_handle();
    return op.call(elements, test_elements, assume_unique, invert);
}

// aten::isin.Tensor_Tensor(Tensor elements, Tensor test_elements, *, bool assume_unique=False, bool invert=False) -> Tensor
at::Tensor isin_Tensor_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & elements, const at::Tensor & test_elements, bool assume_unique, bool invert) {
    static auto op = create_isin_Tensor_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, elements, test_elements, assume_unique, invert);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isin_Tensor_Scalar_out, name, "aten::isin")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isin_Tensor_Scalar_out, overload_name, "Tensor_Scalar_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isin_Tensor_Scalar_out, schema_str, "isin.Tensor_Scalar_out(Tensor elements, Scalar test_element, *, bool assume_unique=False, bool invert=False, Tensor(a!) out) -> Tensor(a!)")

// aten::isin.Tensor_Scalar_out(Tensor elements, Scalar test_element, *, bool assume_unique=False, bool invert=False, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<isin_Tensor_Scalar_out::schema> create_isin_Tensor_Scalar_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(isin_Tensor_Scalar_out::name, isin_Tensor_Scalar_out::overload_name)
      .typed<isin_Tensor_Scalar_out::schema>();
}

// aten::isin.Tensor_Scalar_out(Tensor elements, Scalar test_element, *, bool assume_unique=False, bool invert=False, Tensor(a!) out) -> Tensor(a!)
at::Tensor & isin_Tensor_Scalar_out::call(const at::Tensor & elements, const at::Scalar & test_element, bool assume_unique, bool invert, at::Tensor & out) {
    static auto op = create_isin_Tensor_Scalar_out_typed_handle();
    return op.call(elements, test_element, assume_unique, invert, out);
}

// aten::isin.Tensor_Scalar_out(Tensor elements, Scalar test_element, *, bool assume_unique=False, bool invert=False, Tensor(a!) out) -> Tensor(a!)
at::Tensor & isin_Tensor_Scalar_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & elements, const at::Scalar & test_element, bool assume_unique, bool invert, at::Tensor & out) {
    static auto op = create_isin_Tensor_Scalar_out_typed_handle();
    return op.redispatch(dispatchKeySet, elements, test_element, assume_unique, invert, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isin_Scalar_Tensor_out, name, "aten::isin")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isin_Scalar_Tensor_out, overload_name, "Scalar_Tensor_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isin_Scalar_Tensor_out, schema_str, "isin.Scalar_Tensor_out(Scalar element, Tensor test_elements, *, bool assume_unique=False, bool invert=False, Tensor(a!) out) -> Tensor(a!)")

// aten::isin.Scalar_Tensor_out(Scalar element, Tensor test_elements, *, bool assume_unique=False, bool invert=False, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<isin_Scalar_Tensor_out::schema> create_isin_Scalar_Tensor_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(isin_Scalar_Tensor_out::name, isin_Scalar_Tensor_out::overload_name)
      .typed<isin_Scalar_Tensor_out::schema>();
}

// aten::isin.Scalar_Tensor_out(Scalar element, Tensor test_elements, *, bool assume_unique=False, bool invert=False, Tensor(a!) out) -> Tensor(a!)
at::Tensor & isin_Scalar_Tensor_out::call(const at::Scalar & element, const at::Tensor & test_elements, bool assume_unique, bool invert, at::Tensor & out) {
    static auto op = create_isin_Scalar_Tensor_out_typed_handle();
    return op.call(element, test_elements, assume_unique, invert, out);
}

// aten::isin.Scalar_Tensor_out(Scalar element, Tensor test_elements, *, bool assume_unique=False, bool invert=False, Tensor(a!) out) -> Tensor(a!)
at::Tensor & isin_Scalar_Tensor_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Scalar & element, const at::Tensor & test_elements, bool assume_unique, bool invert, at::Tensor & out) {
    static auto op = create_isin_Scalar_Tensor_out_typed_handle();
    return op.redispatch(dispatchKeySet, element, test_elements, assume_unique, invert, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_distributed, name, "aten::is_distributed")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_distributed, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_distributed, schema_str, "is_distributed(Tensor self) -> bool")

// aten::is_distributed(Tensor self) -> bool
static C10_NOINLINE c10::TypedOperatorHandle<is_distributed::schema> create_is_distributed_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(is_distributed::name, is_distributed::overload_name)
      .typed<is_distributed::schema>();
}

// aten::is_distributed(Tensor self) -> bool
bool is_distributed::call(const at::Tensor & self) {
    static auto op = create_is_distributed_typed_handle();
    return op.call(self);
}

// aten::is_distributed(Tensor self) -> bool
bool is_distributed::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_is_distributed_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_inference, name, "aten::is_inference")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_inference, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_inference, schema_str, "is_inference(Tensor self) -> bool")

// aten::is_inference(Tensor self) -> bool
static C10_NOINLINE c10::TypedOperatorHandle<is_inference::schema> create_is_inference_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(is_inference::name, is_inference::overload_name)
      .typed<is_inference::schema>();
}

// aten::is_inference(Tensor self) -> bool
bool is_inference::call(const at::Tensor & self) {
    static auto op = create_is_inference_typed_handle();
    return op.call(self);
}

// aten::is_inference(Tensor self) -> bool
bool is_inference::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_is_inference_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(kron, name, "aten::kron")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(kron, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(kron, schema_str, "kron(Tensor self, Tensor other) -> Tensor")

// aten::kron(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<kron::schema> create_kron_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(kron::name, kron::overload_name)
      .typed<kron::schema>();
}

// aten::kron(Tensor self, Tensor other) -> Tensor
at::Tensor kron::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_kron_typed_handle();
    return op.call(self, other);
}

// aten::kron(Tensor self, Tensor other) -> Tensor
at::Tensor kron::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_kron_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(kron_out, name, "aten::kron")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(kron_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(kron_out, schema_str, "kron.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::kron.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<kron_out::schema> create_kron_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(kron_out::name, kron_out::overload_name)
      .typed<kron_out::schema>();
}

// aten::kron.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & kron_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_kron_out_typed_handle();
    return op.call(self, other, out);
}

// aten::kron.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & kron_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_kron_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linear, name, "aten::linear")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linear, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linear, schema_str, "linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor")

// aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<linear::schema> create_linear_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linear::name, linear::overload_name)
      .typed<linear::schema>();
}

// aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor
at::Tensor linear::call(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias) {
    static auto op = create_linear_typed_handle();
    return op.call(input, weight, bias);
}

// aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor
at::Tensor linear::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias) {
    static auto op = create_linear_typed_handle();
    return op.redispatch(dispatchKeySet, input, weight, bias);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_linear, name, "aten::mkldnn_linear")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_linear, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_linear, schema_str, "mkldnn_linear(Tensor self, Tensor weight, Tensor? bias=None) -> Tensor")

// aten::mkldnn_linear(Tensor self, Tensor weight, Tensor? bias=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<mkldnn_linear::schema> create_mkldnn_linear_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mkldnn_linear::name, mkldnn_linear::overload_name)
      .typed<mkldnn_linear::schema>();
}

// aten::mkldnn_linear(Tensor self, Tensor weight, Tensor? bias=None) -> Tensor
at::Tensor mkldnn_linear::call(const at::Tensor & self, const at::Tensor & weight, const c10::optional<at::Tensor> & bias) {
    static auto op = create_mkldnn_linear_typed_handle();
    return op.call(self, weight, bias);
}

// aten::mkldnn_linear(Tensor self, Tensor weight, Tensor? bias=None) -> Tensor
at::Tensor mkldnn_linear::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, const c10::optional<at::Tensor> & bias) {
    static auto op = create_mkldnn_linear_typed_handle();
    return op.redispatch(dispatchKeySet, self, weight, bias);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_linear_backward_weights, name, "aten::mkldnn_linear_backward_weights")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_linear_backward_weights, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_linear_backward_weights, schema_str, "mkldnn_linear_backward_weights(Tensor grad_output, Tensor input, Tensor weight, bool bias_defined) -> (Tensor, Tensor)")

// aten::mkldnn_linear_backward_weights(Tensor grad_output, Tensor input, Tensor weight, bool bias_defined) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<mkldnn_linear_backward_weights::schema> create_mkldnn_linear_backward_weights_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mkldnn_linear_backward_weights::name, mkldnn_linear_backward_weights::overload_name)
      .typed<mkldnn_linear_backward_weights::schema>();
}

// aten::mkldnn_linear_backward_weights(Tensor grad_output, Tensor input, Tensor weight, bool bias_defined) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> mkldnn_linear_backward_weights::call(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, bool bias_defined) {
    static auto op = create_mkldnn_linear_backward_weights_typed_handle();
    return op.call(grad_output, input, weight, bias_defined);
}

// aten::mkldnn_linear_backward_weights(Tensor grad_output, Tensor input, Tensor weight, bool bias_defined) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> mkldnn_linear_backward_weights::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, bool bias_defined) {
    static auto op = create_mkldnn_linear_backward_weights_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, input, weight, bias_defined);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fbgemm_linear_quantize_weight, name, "aten::fbgemm_linear_quantize_weight")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fbgemm_linear_quantize_weight, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fbgemm_linear_quantize_weight, schema_str, "fbgemm_linear_quantize_weight(Tensor input) -> (Tensor, Tensor, float, int)")

// aten::fbgemm_linear_quantize_weight(Tensor input) -> (Tensor, Tensor, float, int)
static C10_NOINLINE c10::TypedOperatorHandle<fbgemm_linear_quantize_weight::schema> create_fbgemm_linear_quantize_weight_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fbgemm_linear_quantize_weight::name, fbgemm_linear_quantize_weight::overload_name)
      .typed<fbgemm_linear_quantize_weight::schema>();
}

// aten::fbgemm_linear_quantize_weight(Tensor input) -> (Tensor, Tensor, float, int)
::std::tuple<at::Tensor,at::Tensor,double,int64_t> fbgemm_linear_quantize_weight::call(const at::Tensor & input) {
    static auto op = create_fbgemm_linear_quantize_weight_typed_handle();
    return op.call(input);
}

// aten::fbgemm_linear_quantize_weight(Tensor input) -> (Tensor, Tensor, float, int)
::std::tuple<at::Tensor,at::Tensor,double,int64_t> fbgemm_linear_quantize_weight::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input) {
    static auto op = create_fbgemm_linear_quantize_weight_typed_handle();
    return op.redispatch(dispatchKeySet, input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ldexp_Tensor, name, "aten::ldexp")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ldexp_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ldexp_Tensor, schema_str, "ldexp.Tensor(Tensor self, Tensor other) -> Tensor")

// aten::ldexp.Tensor(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<ldexp_Tensor::schema> create_ldexp_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(ldexp_Tensor::name, ldexp_Tensor::overload_name)
      .typed<ldexp_Tensor::schema>();
}

// aten::ldexp.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor ldexp_Tensor::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_ldexp_Tensor_typed_handle();
    return op.call(self, other);
}

// aten::ldexp.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor ldexp_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_ldexp_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linspace, name, "aten::linspace")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linspace, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linspace, schema_str, "linspace(Scalar start, Scalar end, int? steps=None, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::linspace(Scalar start, Scalar end, int? steps=None, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<linspace::schema> create_linspace_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linspace::name, linspace::overload_name)
      .typed<linspace::schema>();
}

// aten::linspace(Scalar start, Scalar end, int? steps=None, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor linspace::call(const at::Scalar & start, const at::Scalar & end, c10::optional<int64_t> steps, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_linspace_typed_handle();
    return op.call(start, end, steps, dtype, layout, device, pin_memory);
}

// aten::linspace(Scalar start, Scalar end, int? steps=None, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor linspace::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Scalar & start, const at::Scalar & end, c10::optional<int64_t> steps, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_linspace_typed_handle();
    return op.redispatch(dispatchKeySet, start, end, steps, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log_, name, "aten::log_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log_, schema_str, "log_(Tensor(a!) self) -> Tensor(a!)")

// aten::log_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<log_::schema> create_log__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(log_::name, log_::overload_name)
      .typed<log_::schema>();
}

// aten::log_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & log_::call(at::Tensor & self) {
    static auto op = create_log__typed_handle();
    return op.call(self);
}

// aten::log_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & log_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_log__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log10, name, "aten::log10")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log10, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log10, schema_str, "log10(Tensor self) -> Tensor")

// aten::log10(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<log10::schema> create_log10_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(log10::name, log10::overload_name)
      .typed<log10::schema>();
}

// aten::log10(Tensor self) -> Tensor
at::Tensor log10::call(const at::Tensor & self) {
    static auto op = create_log10_typed_handle();
    return op.call(self);
}

// aten::log10(Tensor self) -> Tensor
at::Tensor log10::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_log10_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log10_, name, "aten::log10_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log10_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log10_, schema_str, "log10_(Tensor(a!) self) -> Tensor(a!)")

// aten::log10_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<log10_::schema> create_log10__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(log10_::name, log10_::overload_name)
      .typed<log10_::schema>();
}

// aten::log10_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & log10_::call(at::Tensor & self) {
    static auto op = create_log10__typed_handle();
    return op.call(self);
}

// aten::log10_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & log10_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_log10__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log1p, name, "aten::log1p")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log1p, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log1p, schema_str, "log1p(Tensor self) -> Tensor")

// aten::log1p(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<log1p::schema> create_log1p_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(log1p::name, log1p::overload_name)
      .typed<log1p::schema>();
}

// aten::log1p(Tensor self) -> Tensor
at::Tensor log1p::call(const at::Tensor & self) {
    static auto op = create_log1p_typed_handle();
    return op.call(self);
}

// aten::log1p(Tensor self) -> Tensor
at::Tensor log1p::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_log1p_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log1p_out, name, "aten::log1p")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log1p_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log1p_out, schema_str, "log1p.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::log1p.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<log1p_out::schema> create_log1p_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(log1p_out::name, log1p_out::overload_name)
      .typed<log1p_out::schema>();
}

// aten::log1p.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & log1p_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_log1p_out_typed_handle();
    return op.call(self, out);
}

// aten::log1p.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & log1p_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_log1p_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logaddexp2_out, name, "aten::logaddexp2")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logaddexp2_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logaddexp2_out, schema_str, "logaddexp2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::logaddexp2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<logaddexp2_out::schema> create_logaddexp2_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(logaddexp2_out::name, logaddexp2_out::overload_name)
      .typed<logaddexp2_out::schema>();
}

// aten::logaddexp2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & logaddexp2_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_logaddexp2_out_typed_handle();
    return op.call(self, other, out);
}

// aten::logaddexp2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & logaddexp2_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_logaddexp2_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logaddexp2, name, "aten::logaddexp2")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logaddexp2, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logaddexp2, schema_str, "logaddexp2(Tensor self, Tensor other) -> Tensor")

// aten::logaddexp2(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<logaddexp2::schema> create_logaddexp2_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(logaddexp2::name, logaddexp2::overload_name)
      .typed<logaddexp2::schema>();
}

// aten::logaddexp2(Tensor self, Tensor other) -> Tensor
at::Tensor logaddexp2::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_logaddexp2_typed_handle();
    return op.call(self, other);
}

// aten::logaddexp2(Tensor self, Tensor other) -> Tensor
at::Tensor logaddexp2::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_logaddexp2_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(xlogy_Scalar_Self, name, "aten::xlogy")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(xlogy_Scalar_Self, overload_name, "Scalar_Self")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(xlogy_Scalar_Self, schema_str, "xlogy.Scalar_Self(Scalar self, Tensor other) -> Tensor")

// aten::xlogy.Scalar_Self(Scalar self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<xlogy_Scalar_Self::schema> create_xlogy_Scalar_Self_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(xlogy_Scalar_Self::name, xlogy_Scalar_Self::overload_name)
      .typed<xlogy_Scalar_Self::schema>();
}

// aten::xlogy.Scalar_Self(Scalar self, Tensor other) -> Tensor
at::Tensor xlogy_Scalar_Self::call(const at::Scalar & self, const at::Tensor & other) {
    static auto op = create_xlogy_Scalar_Self_typed_handle();
    return op.call(self, other);
}

// aten::xlogy.Scalar_Self(Scalar self, Tensor other) -> Tensor
at::Tensor xlogy_Scalar_Self::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Scalar & self, const at::Tensor & other) {
    static auto op = create_xlogy_Scalar_Self_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(xlogy_Scalar_Other, name, "aten::xlogy")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(xlogy_Scalar_Other, overload_name, "Scalar_Other")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(xlogy_Scalar_Other, schema_str, "xlogy.Scalar_Other(Tensor self, Scalar other) -> Tensor")

// aten::xlogy.Scalar_Other(Tensor self, Scalar other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<xlogy_Scalar_Other::schema> create_xlogy_Scalar_Other_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(xlogy_Scalar_Other::name, xlogy_Scalar_Other::overload_name)
      .typed<xlogy_Scalar_Other::schema>();
}

// aten::xlogy.Scalar_Other(Tensor self, Scalar other) -> Tensor
at::Tensor xlogy_Scalar_Other::call(const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_xlogy_Scalar_Other_typed_handle();
    return op.call(self, other);
}

// aten::xlogy.Scalar_Other(Tensor self, Scalar other) -> Tensor
at::Tensor xlogy_Scalar_Other::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_xlogy_Scalar_Other_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(xlogy__Scalar_Other, name, "aten::xlogy_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(xlogy__Scalar_Other, overload_name, "Scalar_Other")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(xlogy__Scalar_Other, schema_str, "xlogy_.Scalar_Other(Tensor(a!) self, Scalar other) -> Tensor(a!)")

// aten::xlogy_.Scalar_Other(Tensor(a!) self, Scalar other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<xlogy__Scalar_Other::schema> create_xlogy__Scalar_Other_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(xlogy__Scalar_Other::name, xlogy__Scalar_Other::overload_name)
      .typed<xlogy__Scalar_Other::schema>();
}

// aten::xlogy_.Scalar_Other(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & xlogy__Scalar_Other::call(at::Tensor & self, const at::Scalar & other) {
    static auto op = create_xlogy__Scalar_Other_typed_handle();
    return op.call(self, other);
}

// aten::xlogy_.Scalar_Other(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & xlogy__Scalar_Other::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
    static auto op = create_xlogy__Scalar_Other_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_log_softmax, name, "aten::_log_softmax")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_log_softmax, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_log_softmax, schema_str, "_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor")

// aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_log_softmax::schema> create__log_softmax_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_log_softmax::name, _log_softmax::overload_name)
      .typed<_log_softmax::schema>();
}

// aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
at::Tensor _log_softmax::call(const at::Tensor & self, int64_t dim, bool half_to_float) {
    static auto op = create__log_softmax_typed_handle();
    return op.call(self, dim, half_to_float);
}

// aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
at::Tensor _log_softmax::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool half_to_float) {
    static auto op = create__log_softmax_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, half_to_float);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_log_softmax_out, name, "aten::_log_softmax")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_log_softmax_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_log_softmax_out, schema_str, "_log_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)")

// aten::_log_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<_log_softmax_out::schema> create__log_softmax_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_log_softmax_out::name, _log_softmax_out::overload_name)
      .typed<_log_softmax_out::schema>();
}

// aten::_log_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & _log_softmax_out::call(const at::Tensor & self, int64_t dim, bool half_to_float, at::Tensor & out) {
    static auto op = create__log_softmax_out_typed_handle();
    return op.call(self, dim, half_to_float, out);
}

// aten::_log_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & _log_softmax_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool half_to_float, at::Tensor & out) {
    static auto op = create__log_softmax_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, half_to_float, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logcumsumexp_dimname_out, name, "aten::logcumsumexp")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logcumsumexp_dimname_out, overload_name, "dimname_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logcumsumexp_dimname_out, schema_str, "logcumsumexp.dimname_out(Tensor self, Dimname dim, *, Tensor(a!) out) -> Tensor(a!)")

// aten::logcumsumexp.dimname_out(Tensor self, Dimname dim, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<logcumsumexp_dimname_out::schema> create_logcumsumexp_dimname_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(logcumsumexp_dimname_out::name, logcumsumexp_dimname_out::overload_name)
      .typed<logcumsumexp_dimname_out::schema>();
}

// aten::logcumsumexp.dimname_out(Tensor self, Dimname dim, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & logcumsumexp_dimname_out::call(const at::Tensor & self, at::Dimname dim, at::Tensor & out) {
    static auto op = create_logcumsumexp_dimname_out_typed_handle();
    return op.call(self, dim, out);
}

// aten::logcumsumexp.dimname_out(Tensor self, Dimname dim, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & logcumsumexp_dimname_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, at::Tensor & out) {
    static auto op = create_logcumsumexp_dimname_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logsumexp, name, "aten::logsumexp")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logsumexp, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logsumexp, schema_str, "logsumexp(Tensor self, int[1] dim, bool keepdim=False) -> Tensor")

// aten::logsumexp(Tensor self, int[1] dim, bool keepdim=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<logsumexp::schema> create_logsumexp_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(logsumexp::name, logsumexp::overload_name)
      .typed<logsumexp::schema>();
}

// aten::logsumexp(Tensor self, int[1] dim, bool keepdim=False) -> Tensor
at::Tensor logsumexp::call(const at::Tensor & self, at::IntArrayRef dim, bool keepdim) {
    static auto op = create_logsumexp_typed_handle();
    return op.call(self, dim, keepdim);
}

// aten::logsumexp(Tensor self, int[1] dim, bool keepdim=False) -> Tensor
at::Tensor logsumexp::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool keepdim) {
    static auto op = create_logsumexp_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_aminmax, name, "aten::_aminmax")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_aminmax, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_aminmax, schema_str, "_aminmax(Tensor self) -> (Tensor, Tensor)")

// aten::_aminmax(Tensor self) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<_aminmax::schema> create__aminmax_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_aminmax::name, _aminmax::overload_name)
      .typed<_aminmax::schema>();
}

// aten::_aminmax(Tensor self) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> _aminmax::call(const at::Tensor & self) {
    static auto op = create__aminmax_typed_handle();
    return op.call(self);
}

// aten::_aminmax(Tensor self) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> _aminmax::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create__aminmax_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(aminmax, name, "aten::aminmax")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(aminmax, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(aminmax, schema_str, "aminmax(Tensor self, *, int? dim=None, bool keepdim=False) -> (Tensor min, Tensor max)")

// aten::aminmax(Tensor self, *, int? dim=None, bool keepdim=False) -> (Tensor min, Tensor max)
static C10_NOINLINE c10::TypedOperatorHandle<aminmax::schema> create_aminmax_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(aminmax::name, aminmax::overload_name)
      .typed<aminmax::schema>();
}

// aten::aminmax(Tensor self, *, int? dim=None, bool keepdim=False) -> (Tensor min, Tensor max)
::std::tuple<at::Tensor,at::Tensor> aminmax::call(const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim) {
    static auto op = create_aminmax_typed_handle();
    return op.call(self, dim, keepdim);
}

// aten::aminmax(Tensor self, *, int? dim=None, bool keepdim=False) -> (Tensor min, Tensor max)
::std::tuple<at::Tensor,at::Tensor> aminmax::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim) {
    static auto op = create_aminmax_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_pool1d_with_indices, name, "aten::max_pool1d_with_indices")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_pool1d_with_indices, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_pool1d_with_indices, schema_str, "max_pool1d_with_indices(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)")

// aten::max_pool1d_with_indices(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<max_pool1d_with_indices::schema> create_max_pool1d_with_indices_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(max_pool1d_with_indices::name, max_pool1d_with_indices::overload_name)
      .typed<max_pool1d_with_indices::schema>();
}

// aten::max_pool1d_with_indices(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> max_pool1d_with_indices::call(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
    static auto op = create_max_pool1d_with_indices_typed_handle();
    return op.call(self, kernel_size, stride, padding, dilation, ceil_mode);
}

// aten::max_pool1d_with_indices(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> max_pool1d_with_indices::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
    static auto op = create_max_pool1d_with_indices_typed_handle();
    return op.redispatch(dispatchKeySet, self, kernel_size, stride, padding, dilation, ceil_mode);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_max_pool3d_backward, name, "aten::mkldnn_max_pool3d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_max_pool3d_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_max_pool3d_backward, schema_str, "mkldnn_max_pool3d_backward(Tensor grad_output, Tensor output, Tensor input, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> Tensor")

// aten::mkldnn_max_pool3d_backward(Tensor grad_output, Tensor output, Tensor input, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<mkldnn_max_pool3d_backward::schema> create_mkldnn_max_pool3d_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mkldnn_max_pool3d_backward::name, mkldnn_max_pool3d_backward::overload_name)
      .typed<mkldnn_max_pool3d_backward::schema>();
}

// aten::mkldnn_max_pool3d_backward(Tensor grad_output, Tensor output, Tensor input, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> Tensor
at::Tensor mkldnn_max_pool3d_backward::call(const at::Tensor & grad_output, const at::Tensor & output, const at::Tensor & input, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
    static auto op = create_mkldnn_max_pool3d_backward_typed_handle();
    return op.call(grad_output, output, input, kernel_size, stride, padding, dilation, ceil_mode);
}

// aten::mkldnn_max_pool3d_backward(Tensor grad_output, Tensor output, Tensor input, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> Tensor
at::Tensor mkldnn_max_pool3d_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & output, const at::Tensor & input, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
    static auto op = create_mkldnn_max_pool3d_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, output, input, kernel_size, stride, padding, dilation, ceil_mode);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantized_max_pool1d, name, "aten::quantized_max_pool1d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantized_max_pool1d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantized_max_pool1d, schema_str, "quantized_max_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> Tensor")

// aten::quantized_max_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<quantized_max_pool1d::schema> create_quantized_max_pool1d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(quantized_max_pool1d::name, quantized_max_pool1d::overload_name)
      .typed<quantized_max_pool1d::schema>();
}

// aten::quantized_max_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> Tensor
at::Tensor quantized_max_pool1d::call(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
    static auto op = create_quantized_max_pool1d_typed_handle();
    return op.call(self, kernel_size, stride, padding, dilation, ceil_mode);
}

// aten::quantized_max_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> Tensor
at::Tensor quantized_max_pool1d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
    static auto op = create_quantized_max_pool1d_typed_handle();
    return op.redispatch(dispatchKeySet, self, kernel_size, stride, padding, dilation, ceil_mode);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(median_dim_values, name, "aten::median")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(median_dim_values, overload_name, "dim_values")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(median_dim_values, schema_str, "median.dim_values(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)")

// aten::median.dim_values(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
static C10_NOINLINE c10::TypedOperatorHandle<median_dim_values::schema> create_median_dim_values_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(median_dim_values::name, median_dim_values::overload_name)
      .typed<median_dim_values::schema>();
}

// aten::median.dim_values(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> median_dim_values::call(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
    static auto op = create_median_dim_values_typed_handle();
    return op.call(self, dim, keepdim, values, indices);
}

// aten::median.dim_values(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> median_dim_values::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
    static auto op = create_median_dim_values_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, values, indices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanmedian_names_dim_values, name, "aten::nanmedian")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanmedian_names_dim_values, overload_name, "names_dim_values")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanmedian_names_dim_values, schema_str, "nanmedian.names_dim_values(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)")

// aten::nanmedian.names_dim_values(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
static C10_NOINLINE c10::TypedOperatorHandle<nanmedian_names_dim_values::schema> create_nanmedian_names_dim_values_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nanmedian_names_dim_values::name, nanmedian_names_dim_values::overload_name)
      .typed<nanmedian_names_dim_values::schema>();
}

// aten::nanmedian.names_dim_values(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> nanmedian_names_dim_values::call(const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
    static auto op = create_nanmedian_names_dim_values_typed_handle();
    return op.call(self, dim, keepdim, values, indices);
}

// aten::nanmedian.names_dim_values(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> nanmedian_names_dim_values::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
    static auto op = create_nanmedian_names_dim_values_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, values, indices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(min_dim_min, name, "aten::min")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(min_dim_min, overload_name, "dim_min")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(min_dim_min, schema_str, "min.dim_min(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices) -> (Tensor(a!) values, Tensor(b!) indices)")

// aten::min.dim_min(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices) -> (Tensor(a!) values, Tensor(b!) indices)
static C10_NOINLINE c10::TypedOperatorHandle<min_dim_min::schema> create_min_dim_min_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(min_dim_min::name, min_dim_min::overload_name)
      .typed<min_dim_min::schema>();
}

// aten::min.dim_min(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> min_dim_min::call(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & min, at::Tensor & min_indices) {
    static auto op = create_min_dim_min_typed_handle();
    return op.call(self, dim, keepdim, min, min_indices);
}

// aten::min.dim_min(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> min_dim_min::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & min, at::Tensor & min_indices) {
    static auto op = create_min_dim_min_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, min, min_indices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(min_names_dim, name, "aten::min")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(min_names_dim, overload_name, "names_dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(min_names_dim, schema_str, "min.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)")

// aten::min.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
static C10_NOINLINE c10::TypedOperatorHandle<min_names_dim::schema> create_min_names_dim_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(min_names_dim::name, min_names_dim::overload_name)
      .typed<min_names_dim::schema>();
}

// aten::min.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> min_names_dim::call(const at::Tensor & self, at::Dimname dim, bool keepdim) {
    static auto op = create_min_names_dim_typed_handle();
    return op.call(self, dim, keepdim);
}

// aten::min.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> min_names_dim::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, bool keepdim) {
    static auto op = create_min_names_dim_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(amin_out, name, "aten::amin")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(amin_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(amin_out, schema_str, "amin.out(Tensor self, int[1] dim=[], bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)")

// aten::amin.out(Tensor self, int[1] dim=[], bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<amin_out::schema> create_amin_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(amin_out::name, amin_out::overload_name)
      .typed<amin_out::schema>();
}

// aten::amin.out(Tensor self, int[1] dim=[], bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & amin_out::call(const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
    static auto op = create_amin_out_typed_handle();
    return op.call(self, dim, keepdim, out);
}

// aten::amin.out(Tensor self, int[1] dim=[], bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & amin_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
    static auto op = create_amin_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_convolution, name, "aten::mkldnn_convolution")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_convolution, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_convolution, schema_str, "mkldnn_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups) -> Tensor")

// aten::mkldnn_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<mkldnn_convolution::schema> create_mkldnn_convolution_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mkldnn_convolution::name, mkldnn_convolution::overload_name)
      .typed<mkldnn_convolution::schema>();
}

// aten::mkldnn_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups) -> Tensor
at::Tensor mkldnn_convolution::call(const at::Tensor & self, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups) {
    static auto op = create_mkldnn_convolution_typed_handle();
    return op.call(self, weight, bias, padding, stride, dilation, groups);
}

// aten::mkldnn_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups) -> Tensor
at::Tensor mkldnn_convolution::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups) {
    static auto op = create_mkldnn_convolution_typed_handle();
    return op.redispatch(dispatchKeySet, self, weight, bias, padding, stride, dilation, groups);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_batch_norm_backward, name, "aten::miopen_batch_norm_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_batch_norm_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_batch_norm_backward, schema_str, "miopen_batch_norm_backward(Tensor input, Tensor grad_output, Tensor weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var, float epsilon) -> (Tensor, Tensor, Tensor)")

// aten::miopen_batch_norm_backward(Tensor input, Tensor grad_output, Tensor weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var, float epsilon) -> (Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<miopen_batch_norm_backward::schema> create_miopen_batch_norm_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(miopen_batch_norm_backward::name, miopen_batch_norm_backward::overload_name)
      .typed<miopen_batch_norm_backward::schema>();
}

// aten::miopen_batch_norm_backward(Tensor input, Tensor grad_output, Tensor weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var, float epsilon) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_batch_norm_backward::call(const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, const c10::optional<at::Tensor> & save_mean, const c10::optional<at::Tensor> & save_var, double epsilon) {
    static auto op = create_miopen_batch_norm_backward_typed_handle();
    return op.call(input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon);
}

// aten::miopen_batch_norm_backward(Tensor input, Tensor grad_output, Tensor weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var, float epsilon) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_batch_norm_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, const c10::optional<at::Tensor> & save_mean, const c10::optional<at::Tensor> & save_var, double epsilon) {
    static auto op = create_miopen_batch_norm_backward_typed_handle();
    return op.redispatch(dispatchKeySet, input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_convolution_backward_input, name, "aten::miopen_convolution_backward_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_convolution_backward_input, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_convolution_backward_input, schema_str, "miopen_convolution_backward_input(int[] self_size, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor")

// aten::miopen_convolution_backward_input(int[] self_size, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<miopen_convolution_backward_input::schema> create_miopen_convolution_backward_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(miopen_convolution_backward_input::name, miopen_convolution_backward_input::overload_name)
      .typed<miopen_convolution_backward_input::schema>();
}

// aten::miopen_convolution_backward_input(int[] self_size, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
at::Tensor miopen_convolution_backward_input::call(at::IntArrayRef self_size, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
    static auto op = create_miopen_convolution_backward_input_typed_handle();
    return op.call(self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
}

// aten::miopen_convolution_backward_input(int[] self_size, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
at::Tensor miopen_convolution_backward_input::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef self_size, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
    static auto op = create_miopen_convolution_backward_input_typed_handle();
    return op.redispatch(dispatchKeySet, self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_convolution_backward_weight, name, "aten::miopen_convolution_backward_weight")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_convolution_backward_weight, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_convolution_backward_weight, schema_str, "miopen_convolution_backward_weight(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor")

// aten::miopen_convolution_backward_weight(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<miopen_convolution_backward_weight::schema> create_miopen_convolution_backward_weight_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(miopen_convolution_backward_weight::name, miopen_convolution_backward_weight::overload_name)
      .typed<miopen_convolution_backward_weight::schema>();
}

// aten::miopen_convolution_backward_weight(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
at::Tensor miopen_convolution_backward_weight::call(at::IntArrayRef weight_size, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
    static auto op = create_miopen_convolution_backward_weight_typed_handle();
    return op.call(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
}

// aten::miopen_convolution_backward_weight(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
at::Tensor miopen_convolution_backward_weight::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef weight_size, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
    static auto op = create_miopen_convolution_backward_weight_typed_handle();
    return op.redispatch(dispatchKeySet, weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_depthwise_convolution_backward, name, "aten::miopen_depthwise_convolution_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_depthwise_convolution_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(miopen_depthwise_convolution_backward, schema_str, "miopen_depthwise_convolution_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool[3] output_mask) -> (Tensor, Tensor, Tensor)")

// aten::miopen_depthwise_convolution_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<miopen_depthwise_convolution_backward::schema> create_miopen_depthwise_convolution_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(miopen_depthwise_convolution_backward::name, miopen_depthwise_convolution_backward::overload_name)
      .typed<miopen_depthwise_convolution_backward::schema>();
}

// aten::miopen_depthwise_convolution_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_depthwise_convolution_backward::call(const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, ::std::array<bool,3> output_mask) {
    static auto op = create_miopen_depthwise_convolution_backward_typed_handle();
    return op.call(self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, output_mask);
}

// aten::miopen_depthwise_convolution_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_depthwise_convolution_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, ::std::array<bool,3> output_mask) {
    static auto op = create_miopen_depthwise_convolution_backward_typed_handle();
    return op.redispatch(dispatchKeySet, self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, output_mask);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mode, name, "aten::mode")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mode, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mode, schema_str, "mode(Tensor self, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)")

// aten::mode(Tensor self, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)
static C10_NOINLINE c10::TypedOperatorHandle<mode::schema> create_mode_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mode::name, mode::overload_name)
      .typed<mode::schema>();
}

// aten::mode(Tensor self, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> mode::call(const at::Tensor & self, int64_t dim, bool keepdim) {
    static auto op = create_mode_typed_handle();
    return op.call(self, dim, keepdim);
}

// aten::mode(Tensor self, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> mode::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool keepdim) {
    static auto op = create_mode_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mode_values, name, "aten::mode")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mode_values, overload_name, "values")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mode_values, schema_str, "mode.values(Tensor self, int dim=-1, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)")

// aten::mode.values(Tensor self, int dim=-1, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
static C10_NOINLINE c10::TypedOperatorHandle<mode_values::schema> create_mode_values_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mode_values::name, mode_values::overload_name)
      .typed<mode_values::schema>();
}

// aten::mode.values(Tensor self, int dim=-1, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> mode_values::call(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
    static auto op = create_mode_values_typed_handle();
    return op.call(self, dim, keepdim, values, indices);
}

// aten::mode.values(Tensor self, int dim=-1, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> mode_values::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
    static auto op = create_mode_values_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, values, indices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mul_Tensor, name, "aten::mul")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mul_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mul_Tensor, schema_str, "mul.Tensor(Tensor self, Tensor other) -> Tensor")

// aten::mul.Tensor(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<mul_Tensor::schema> create_mul_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mul_Tensor::name, mul_Tensor::overload_name)
      .typed<mul_Tensor::schema>();
}

// aten::mul.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor mul_Tensor::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_mul_Tensor_typed_handle();
    return op.call(self, other);
}

// aten::mul.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor mul_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_mul_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mul__Tensor, name, "aten::mul_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mul__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mul__Tensor, schema_str, "mul_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::mul_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<mul__Tensor::schema> create_mul__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mul__Tensor::name, mul__Tensor::overload_name)
      .typed<mul__Tensor::schema>();
}

// aten::mul_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & mul__Tensor::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create_mul__Tensor_typed_handle();
    return op.call(self, other);
}

// aten::mul_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & mul__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create_mul__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multiply__Scalar, name, "aten::multiply_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multiply__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multiply__Scalar, schema_str, "multiply_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)")

// aten::multiply_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<multiply__Scalar::schema> create_multiply__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(multiply__Scalar::name, multiply__Scalar::overload_name)
      .typed<multiply__Scalar::schema>();
}

// aten::multiply_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & multiply__Scalar::call(at::Tensor & self, const at::Scalar & other) {
    static auto op = create_multiply__Scalar_typed_handle();
    return op.call(self, other);
}

// aten::multiply_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & multiply__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
    static auto op = create_multiply__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mvlgamma_out, name, "aten::mvlgamma")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mvlgamma_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mvlgamma_out, schema_str, "mvlgamma.out(Tensor self, int p, *, Tensor(a!) out) -> Tensor(a!)")

// aten::mvlgamma.out(Tensor self, int p, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<mvlgamma_out::schema> create_mvlgamma_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mvlgamma_out::name, mvlgamma_out::overload_name)
      .typed<mvlgamma_out::schema>();
}

// aten::mvlgamma.out(Tensor self, int p, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & mvlgamma_out::call(const at::Tensor & self, int64_t p, at::Tensor & out) {
    static auto op = create_mvlgamma_out_typed_handle();
    return op.call(self, p, out);
}

// aten::mvlgamma.out(Tensor self, int p, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & mvlgamma_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t p, at::Tensor & out) {
    static auto op = create_mvlgamma_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, p, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mvlgamma, name, "aten::mvlgamma")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mvlgamma, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mvlgamma, schema_str, "mvlgamma(Tensor self, int p) -> Tensor")

// aten::mvlgamma(Tensor self, int p) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<mvlgamma::schema> create_mvlgamma_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mvlgamma::name, mvlgamma::overload_name)
      .typed<mvlgamma::schema>();
}

// aten::mvlgamma(Tensor self, int p) -> Tensor
at::Tensor mvlgamma::call(const at::Tensor & self, int64_t p) {
    static auto op = create_mvlgamma_typed_handle();
    return op.call(self, p);
}

// aten::mvlgamma(Tensor self, int p) -> Tensor
at::Tensor mvlgamma::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t p) {
    static auto op = create_mvlgamma_typed_handle();
    return op.redispatch(dispatchKeySet, self, p);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(narrow_copy_out, name, "aten::narrow_copy")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(narrow_copy_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(narrow_copy_out, schema_str, "narrow_copy.out(Tensor self, int dim, int start, int length, *, Tensor(a!) out) -> Tensor(a!)")

// aten::narrow_copy.out(Tensor self, int dim, int start, int length, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<narrow_copy_out::schema> create_narrow_copy_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(narrow_copy_out::name, narrow_copy_out::overload_name)
      .typed<narrow_copy_out::schema>();
}

// aten::narrow_copy.out(Tensor self, int dim, int start, int length, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & narrow_copy_out::call(const at::Tensor & self, int64_t dim, int64_t start, int64_t length, at::Tensor & out) {
    static auto op = create_narrow_copy_out_typed_handle();
    return op.call(self, dim, start, length, out);
}

// aten::narrow_copy.out(Tensor self, int dim, int start, int length, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & narrow_copy_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, int64_t start, int64_t length, at::Tensor & out) {
    static auto op = create_narrow_copy_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, start, length, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(narrow, name, "aten::narrow")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(narrow, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(narrow, schema_str, "narrow(Tensor(a) self, int dim, int start, int length) -> Tensor(a)")

// aten::narrow(Tensor(a) self, int dim, int start, int length) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<narrow::schema> create_narrow_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(narrow::name, narrow::overload_name)
      .typed<narrow::schema>();
}

// aten::narrow(Tensor(a) self, int dim, int start, int length) -> Tensor(a)
at::Tensor narrow::call(const at::Tensor & self, int64_t dim, int64_t start, int64_t length) {
    static auto op = create_narrow_typed_handle();
    return op.call(self, dim, start, length);
}

// aten::narrow(Tensor(a) self, int dim, int start, int length) -> Tensor(a)
at::Tensor narrow::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, int64_t start, int64_t length) {
    static auto op = create_narrow_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, start, length);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(batch_norm_backward_elemt, name, "aten::batch_norm_backward_elemt")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(batch_norm_backward_elemt, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(batch_norm_backward_elemt, schema_str, "batch_norm_backward_elemt(Tensor grad_out, Tensor input, Tensor mean, Tensor invstd, Tensor? weight, Tensor mean_dy, Tensor mean_dy_xmu, Tensor count) -> Tensor")

// aten::batch_norm_backward_elemt(Tensor grad_out, Tensor input, Tensor mean, Tensor invstd, Tensor? weight, Tensor mean_dy, Tensor mean_dy_xmu, Tensor count) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<batch_norm_backward_elemt::schema> create_batch_norm_backward_elemt_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(batch_norm_backward_elemt::name, batch_norm_backward_elemt::overload_name)
      .typed<batch_norm_backward_elemt::schema>();
}

// aten::batch_norm_backward_elemt(Tensor grad_out, Tensor input, Tensor mean, Tensor invstd, Tensor? weight, Tensor mean_dy, Tensor mean_dy_xmu, Tensor count) -> Tensor
at::Tensor batch_norm_backward_elemt::call(const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const c10::optional<at::Tensor> & weight, const at::Tensor & mean_dy, const at::Tensor & mean_dy_xmu, const at::Tensor & count) {
    static auto op = create_batch_norm_backward_elemt_typed_handle();
    return op.call(grad_out, input, mean, invstd, weight, mean_dy, mean_dy_xmu, count);
}

// aten::batch_norm_backward_elemt(Tensor grad_out, Tensor input, Tensor mean, Tensor invstd, Tensor? weight, Tensor mean_dy, Tensor mean_dy_xmu, Tensor count) -> Tensor
at::Tensor batch_norm_backward_elemt::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const c10::optional<at::Tensor> & weight, const at::Tensor & mean_dy, const at::Tensor & mean_dy_xmu, const at::Tensor & count) {
    static auto op = create_batch_norm_backward_elemt_typed_handle();
    return op.redispatch(dispatchKeySet, grad_out, input, mean, invstd, weight, mean_dy, mean_dy_xmu, count);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pdist, name, "aten::pdist")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pdist, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pdist, schema_str, "pdist(Tensor self, float p=2) -> Tensor")

// aten::pdist(Tensor self, float p=2) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<pdist::schema> create_pdist_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(pdist::name, pdist::overload_name)
      .typed<pdist::schema>();
}

// aten::pdist(Tensor self, float p=2) -> Tensor
at::Tensor pdist::call(const at::Tensor & self, double p) {
    static auto op = create_pdist_typed_handle();
    return op.call(self, p);
}

// aten::pdist(Tensor self, float p=2) -> Tensor
at::Tensor pdist::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double p) {
    static auto op = create_pdist_typed_handle();
    return op.redispatch(dispatchKeySet, self, p);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(movedim_intlist, name, "aten::movedim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(movedim_intlist, overload_name, "intlist")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(movedim_intlist, schema_str, "movedim.intlist(Tensor(a) self, int[] source, int[] destination) -> Tensor(a)")

// aten::movedim.intlist(Tensor(a) self, int[] source, int[] destination) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<movedim_intlist::schema> create_movedim_intlist_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(movedim_intlist::name, movedim_intlist::overload_name)
      .typed<movedim_intlist::schema>();
}

// aten::movedim.intlist(Tensor(a) self, int[] source, int[] destination) -> Tensor(a)
at::Tensor movedim_intlist::call(const at::Tensor & self, at::IntArrayRef source, at::IntArrayRef destination) {
    static auto op = create_movedim_intlist_typed_handle();
    return op.call(self, source, destination);
}

// aten::movedim.intlist(Tensor(a) self, int[] source, int[] destination) -> Tensor(a)
at::Tensor movedim_intlist::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef source, at::IntArrayRef destination) {
    static auto op = create_movedim_intlist_typed_handle();
    return op.redispatch(dispatchKeySet, self, source, destination);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(moveaxis_intlist, name, "aten::moveaxis")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(moveaxis_intlist, overload_name, "intlist")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(moveaxis_intlist, schema_str, "moveaxis.intlist(Tensor(a) self, int[] source, int[] destination) -> Tensor(a)")

// aten::moveaxis.intlist(Tensor(a) self, int[] source, int[] destination) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<moveaxis_intlist::schema> create_moveaxis_intlist_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(moveaxis_intlist::name, moveaxis_intlist::overload_name)
      .typed<moveaxis_intlist::schema>();
}

// aten::moveaxis.intlist(Tensor(a) self, int[] source, int[] destination) -> Tensor(a)
at::Tensor moveaxis_intlist::call(const at::Tensor & self, at::IntArrayRef source, at::IntArrayRef destination) {
    static auto op = create_moveaxis_intlist_typed_handle();
    return op.call(self, source, destination);
}

// aten::moveaxis.intlist(Tensor(a) self, int[] source, int[] destination) -> Tensor(a)
at::Tensor moveaxis_intlist::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef source, at::IntArrayRef destination) {
    static auto op = create_moveaxis_intlist_typed_handle();
    return op.redispatch(dispatchKeySet, self, source, destination);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pixel_unshuffle, name, "aten::pixel_unshuffle")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pixel_unshuffle, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pixel_unshuffle, schema_str, "pixel_unshuffle(Tensor self, int downscale_factor) -> Tensor")

// aten::pixel_unshuffle(Tensor self, int downscale_factor) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<pixel_unshuffle::schema> create_pixel_unshuffle_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(pixel_unshuffle::name, pixel_unshuffle::overload_name)
      .typed<pixel_unshuffle::schema>();
}

// aten::pixel_unshuffle(Tensor self, int downscale_factor) -> Tensor
at::Tensor pixel_unshuffle::call(const at::Tensor & self, int64_t downscale_factor) {
    static auto op = create_pixel_unshuffle_typed_handle();
    return op.call(self, downscale_factor);
}

// aten::pixel_unshuffle(Tensor self, int downscale_factor) -> Tensor
at::Tensor pixel_unshuffle::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t downscale_factor) {
    static auto op = create_pixel_unshuffle_typed_handle();
    return op.redispatch(dispatchKeySet, self, downscale_factor);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_pinned, name, "aten::is_pinned")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_pinned, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_pinned, schema_str, "is_pinned(Tensor self, Device? device=None) -> bool")

// aten::is_pinned(Tensor self, Device? device=None) -> bool
static C10_NOINLINE c10::TypedOperatorHandle<is_pinned::schema> create_is_pinned_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(is_pinned::name, is_pinned::overload_name)
      .typed<is_pinned::schema>();
}

// aten::is_pinned(Tensor self, Device? device=None) -> bool
bool is_pinned::call(const at::Tensor & self, c10::optional<at::Device> device) {
    static auto op = create_is_pinned_typed_handle();
    return op.call(self, device);
}

// aten::is_pinned(Tensor self, Device? device=None) -> bool
bool is_pinned::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::Device> device) {
    static auto op = create_is_pinned_typed_handle();
    return op.redispatch(dispatchKeySet, self, device);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pin_memory, name, "aten::pin_memory")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pin_memory, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pin_memory, schema_str, "pin_memory(Tensor(a) self, Device? device=None) -> Tensor(a)")

// aten::pin_memory(Tensor(a) self, Device? device=None) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<pin_memory::schema> create_pin_memory_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(pin_memory::name, pin_memory::overload_name)
      .typed<pin_memory::schema>();
}

// aten::pin_memory(Tensor(a) self, Device? device=None) -> Tensor(a)
at::Tensor pin_memory::call(const at::Tensor & self, c10::optional<at::Device> device) {
    static auto op = create_pin_memory_typed_handle();
    return op.call(self, device);
}

// aten::pin_memory(Tensor(a) self, Device? device=None) -> Tensor(a)
at::Tensor pin_memory::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::Device> device) {
    static auto op = create_pin_memory_typed_handle();
    return op.redispatch(dispatchKeySet, self, device);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_pin_memory, name, "aten::_pin_memory")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_pin_memory, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_pin_memory, schema_str, "_pin_memory(Tensor self, Device? device=None) -> Tensor")

// aten::_pin_memory(Tensor self, Device? device=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_pin_memory::schema> create__pin_memory_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_pin_memory::name, _pin_memory::overload_name)
      .typed<_pin_memory::schema>();
}

// aten::_pin_memory(Tensor self, Device? device=None) -> Tensor
at::Tensor _pin_memory::call(const at::Tensor & self, c10::optional<at::Device> device) {
    static auto op = create__pin_memory_typed_handle();
    return op.call(self, device);
}

// aten::_pin_memory(Tensor self, Device? device=None) -> Tensor
at::Tensor _pin_memory::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::Device> device) {
    static auto op = create__pin_memory_typed_handle();
    return op.redispatch(dispatchKeySet, self, device);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rad2deg_out, name, "aten::rad2deg")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rad2deg_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rad2deg_out, schema_str, "rad2deg.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::rad2deg.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<rad2deg_out::schema> create_rad2deg_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(rad2deg_out::name, rad2deg_out::overload_name)
      .typed<rad2deg_out::schema>();
}

// aten::rad2deg.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & rad2deg_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_rad2deg_out_typed_handle();
    return op.call(self, out);
}

// aten::rad2deg.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & rad2deg_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_rad2deg_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randint_low_generator_out, name, "aten::randint")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randint_low_generator_out, overload_name, "low_generator_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randint_low_generator_out, schema_str, "randint.low_generator_out(int low, int high, int[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)")

// aten::randint.low_generator_out(int low, int high, int[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<randint_low_generator_out::schema> create_randint_low_generator_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(randint_low_generator_out::name, randint_low_generator_out::overload_name)
      .typed<randint_low_generator_out::schema>();
}

// aten::randint.low_generator_out(int low, int high, int[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
at::Tensor & randint_low_generator_out::call(int64_t low, int64_t high, at::IntArrayRef size, c10::optional<at::Generator> generator, at::Tensor & out) {
    static auto op = create_randint_low_generator_out_typed_handle();
    return op.call(low, high, size, generator, out);
}

// aten::randint.low_generator_out(int low, int high, int[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
at::Tensor & randint_low_generator_out::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t low, int64_t high, at::IntArrayRef size, c10::optional<at::Generator> generator, at::Tensor & out) {
    static auto op = create_randint_low_generator_out_typed_handle();
    return op.redispatch(dispatchKeySet, low, high, size, generator, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randn, name, "aten::randn")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randn, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randn, schema_str, "randn(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::randn(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<randn::schema> create_randn_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(randn::name, randn::overload_name)
      .typed<randn::schema>();
}

// aten::randn(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor randn::call(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_randn_typed_handle();
    return op.call(size, dtype, layout, device, pin_memory);
}

// aten::randn(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor randn::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_randn_typed_handle();
    return op.redispatch(dispatchKeySet, size, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randn_generator, name, "aten::randn")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randn_generator, overload_name, "generator")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randn_generator, schema_str, "randn.generator(int[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::randn.generator(int[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<randn_generator::schema> create_randn_generator_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(randn_generator::name, randn_generator::overload_name)
      .typed<randn_generator::schema>();
}

// aten::randn.generator(int[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor randn_generator::call(at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_randn_generator_typed_handle();
    return op.call(size, generator, dtype, layout, device, pin_memory);
}

// aten::randn.generator(int[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor randn_generator::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_randn_generator_typed_handle();
    return op.redispatch(dispatchKeySet, size, generator, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randn_generator_out, name, "aten::randn")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randn_generator_out, overload_name, "generator_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randn_generator_out, schema_str, "randn.generator_out(int[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)")

// aten::randn.generator_out(int[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<randn_generator_out::schema> create_randn_generator_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(randn_generator_out::name, randn_generator_out::overload_name)
      .typed<randn_generator_out::schema>();
}

// aten::randn.generator_out(int[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
at::Tensor & randn_generator_out::call(at::IntArrayRef size, c10::optional<at::Generator> generator, at::Tensor & out) {
    static auto op = create_randn_generator_out_typed_handle();
    return op.call(size, generator, out);
}

// aten::randn.generator_out(int[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
at::Tensor & randn_generator_out::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::Generator> generator, at::Tensor & out) {
    static auto op = create_randn_generator_out_typed_handle();
    return op.redispatch(dispatchKeySet, size, generator, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(range, name, "aten::range")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(range, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(range, schema_str, "range(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::range(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<range::schema> create_range_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(range::name, range::overload_name)
      .typed<range::schema>();
}

// aten::range(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor range::call(const at::Scalar & start, const at::Scalar & end, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_range_typed_handle();
    return op.call(start, end, dtype, layout, device, pin_memory);
}

// aten::range(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor range::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Scalar & start, const at::Scalar & end, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_range_typed_handle();
    return op.redispatch(dispatchKeySet, start, end, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ravel, name, "aten::ravel")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ravel, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ravel, schema_str, "ravel(Tensor(a) self) -> Tensor(a)")

// aten::ravel(Tensor(a) self) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<ravel::schema> create_ravel_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(ravel::name, ravel::overload_name)
      .typed<ravel::schema>();
}

// aten::ravel(Tensor(a) self) -> Tensor(a)
at::Tensor ravel::call(const at::Tensor & self) {
    static auto op = create_ravel_typed_handle();
    return op.call(self);
}

// aten::ravel(Tensor(a) self) -> Tensor(a)
at::Tensor ravel::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_ravel_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reciprocal, name, "aten::reciprocal")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reciprocal, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reciprocal, schema_str, "reciprocal(Tensor self) -> Tensor")

// aten::reciprocal(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<reciprocal::schema> create_reciprocal_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(reciprocal::name, reciprocal::overload_name)
      .typed<reciprocal::schema>();
}

// aten::reciprocal(Tensor self) -> Tensor
at::Tensor reciprocal::call(const at::Tensor & self) {
    static auto op = create_reciprocal_typed_handle();
    return op.call(self);
}

// aten::reciprocal(Tensor self) -> Tensor
at::Tensor reciprocal::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_reciprocal_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(neg, name, "aten::neg")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(neg, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(neg, schema_str, "neg(Tensor self) -> Tensor")

// aten::neg(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<neg::schema> create_neg_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(neg::name, neg::overload_name)
      .typed<neg::schema>();
}

// aten::neg(Tensor self) -> Tensor
at::Tensor neg::call(const at::Tensor & self) {
    static auto op = create_neg_typed_handle();
    return op.call(self);
}

// aten::neg(Tensor self) -> Tensor
at::Tensor neg::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_neg_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(negative_out, name, "aten::negative")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(negative_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(negative_out, schema_str, "negative.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::negative.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<negative_out::schema> create_negative_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(negative_out::name, negative_out::overload_name)
      .typed<negative_out::schema>();
}

// aten::negative.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & negative_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_negative_out_typed_handle();
    return op.call(self, out);
}

// aten::negative.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & negative_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_negative_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(repeat_interleave_self_Tensor, name, "aten::repeat_interleave")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(repeat_interleave_self_Tensor, overload_name, "self_Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(repeat_interleave_self_Tensor, schema_str, "repeat_interleave.self_Tensor(Tensor self, Tensor repeats, int? dim=None, *, int? output_size=None) -> Tensor")

// aten::repeat_interleave.self_Tensor(Tensor self, Tensor repeats, int? dim=None, *, int? output_size=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<repeat_interleave_self_Tensor::schema> create_repeat_interleave_self_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(repeat_interleave_self_Tensor::name, repeat_interleave_self_Tensor::overload_name)
      .typed<repeat_interleave_self_Tensor::schema>();
}

// aten::repeat_interleave.self_Tensor(Tensor self, Tensor repeats, int? dim=None, *, int? output_size=None) -> Tensor
at::Tensor repeat_interleave_self_Tensor::call(const at::Tensor & self, const at::Tensor & repeats, c10::optional<int64_t> dim, c10::optional<int64_t> output_size) {
    static auto op = create_repeat_interleave_self_Tensor_typed_handle();
    return op.call(self, repeats, dim, output_size);
}

// aten::repeat_interleave.self_Tensor(Tensor self, Tensor repeats, int? dim=None, *, int? output_size=None) -> Tensor
at::Tensor repeat_interleave_self_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & repeats, c10::optional<int64_t> dim, c10::optional<int64_t> output_size) {
    static auto op = create_repeat_interleave_self_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, repeats, dim, output_size);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reshape_as, name, "aten::reshape_as")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reshape_as, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reshape_as, schema_str, "reshape_as(Tensor(a) self, Tensor other) -> Tensor(a)")

// aten::reshape_as(Tensor(a) self, Tensor other) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<reshape_as::schema> create_reshape_as_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(reshape_as::name, reshape_as::overload_name)
      .typed<reshape_as::schema>();
}

// aten::reshape_as(Tensor(a) self, Tensor other) -> Tensor(a)
at::Tensor reshape_as::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_reshape_as_typed_handle();
    return op.call(self, other);
}

// aten::reshape_as(Tensor(a) self, Tensor other) -> Tensor(a)
at::Tensor reshape_as::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_reshape_as_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rrelu, name, "aten::rrelu")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rrelu, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rrelu, schema_str, "rrelu(Tensor self, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor")

// aten::rrelu(Tensor self, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<rrelu::schema> create_rrelu_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(rrelu::name, rrelu::overload_name)
      .typed<rrelu::schema>();
}

// aten::rrelu(Tensor self, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor
at::Tensor rrelu::call(const at::Tensor & self, const at::Scalar & lower, const at::Scalar & upper, bool training, c10::optional<at::Generator> generator) {
    static auto op = create_rrelu_typed_handle();
    return op.call(self, lower, upper, training, generator);
}

// aten::rrelu(Tensor self, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor
at::Tensor rrelu::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & lower, const at::Scalar & upper, bool training, c10::optional<at::Generator> generator) {
    static auto op = create_rrelu_typed_handle();
    return op.redispatch(dispatchKeySet, self, lower, upper, training, generator);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(relu6, name, "aten::relu6")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(relu6, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(relu6, schema_str, "relu6(Tensor self) -> Tensor")

// aten::relu6(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<relu6::schema> create_relu6_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(relu6::name, relu6::overload_name)
      .typed<relu6::schema>();
}

// aten::relu6(Tensor self) -> Tensor
at::Tensor relu6::call(const at::Tensor & self) {
    static auto op = create_relu6_typed_handle();
    return op.call(self);
}

// aten::relu6(Tensor self) -> Tensor
at::Tensor relu6::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_relu6_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(prelu, name, "aten::prelu")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(prelu, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(prelu, schema_str, "prelu(Tensor self, Tensor weight) -> Tensor")

// aten::prelu(Tensor self, Tensor weight) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<prelu::schema> create_prelu_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(prelu::name, prelu::overload_name)
      .typed<prelu::schema>();
}

// aten::prelu(Tensor self, Tensor weight) -> Tensor
at::Tensor prelu::call(const at::Tensor & self, const at::Tensor & weight) {
    static auto op = create_prelu_typed_handle();
    return op.call(self, weight);
}

// aten::prelu(Tensor self, Tensor weight) -> Tensor
at::Tensor prelu::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight) {
    static auto op = create_prelu_typed_handle();
    return op.redispatch(dispatchKeySet, self, weight);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gelu_backward, name, "aten::gelu_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gelu_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gelu_backward, schema_str, "gelu_backward(Tensor grad, Tensor self) -> Tensor")

// aten::gelu_backward(Tensor grad, Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<gelu_backward::schema> create_gelu_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(gelu_backward::name, gelu_backward::overload_name)
      .typed<gelu_backward::schema>();
}

// aten::gelu_backward(Tensor grad, Tensor self) -> Tensor
at::Tensor gelu_backward::call(const at::Tensor & grad, const at::Tensor & self) {
    static auto op = create_gelu_backward_typed_handle();
    return op.call(grad, self);
}

// aten::gelu_backward(Tensor grad, Tensor self) -> Tensor
at::Tensor gelu_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & self) {
    static auto op = create_gelu_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(select_Dimname, name, "aten::select")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(select_Dimname, overload_name, "Dimname")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(select_Dimname, schema_str, "select.Dimname(Tensor(a) self, Dimname dim, int index) -> Tensor(a)")

// aten::select.Dimname(Tensor(a) self, Dimname dim, int index) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<select_Dimname::schema> create_select_Dimname_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(select_Dimname::name, select_Dimname::overload_name)
      .typed<select_Dimname::schema>();
}

// aten::select.Dimname(Tensor(a) self, Dimname dim, int index) -> Tensor(a)
at::Tensor select_Dimname::call(const at::Tensor & self, at::Dimname dim, int64_t index) {
    static auto op = create_select_Dimname_typed_handle();
    return op.call(self, dim, index);
}

// aten::select.Dimname(Tensor(a) self, Dimname dim, int index) -> Tensor(a)
at::Tensor select_Dimname::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, int64_t index) {
    static auto op = create_select_Dimname_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(selu, name, "aten::selu")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(selu, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(selu, schema_str, "selu(Tensor self) -> Tensor")

// aten::selu(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<selu::schema> create_selu_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(selu::name, selu::overload_name)
      .typed<selu::schema>();
}

// aten::selu(Tensor self) -> Tensor
at::Tensor selu::call(const at::Tensor & self) {
    static auto op = create_selu_typed_handle();
    return op.call(self);
}

// aten::selu(Tensor self) -> Tensor
at::Tensor selu::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_selu_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(silu_backward_grad_input, name, "aten::silu_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(silu_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(silu_backward_grad_input, schema_str, "silu_backward.grad_input(Tensor grad_output, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::silu_backward.grad_input(Tensor grad_output, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<silu_backward_grad_input::schema> create_silu_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(silu_backward_grad_input::name, silu_backward_grad_input::overload_name)
      .typed<silu_backward_grad_input::schema>();
}

// aten::silu_backward.grad_input(Tensor grad_output, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & silu_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, at::Tensor & grad_input) {
    static auto op = create_silu_backward_grad_input_typed_handle();
    return op.call(grad_output, self, grad_input);
}

// aten::silu_backward.grad_input(Tensor grad_output, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & silu_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::Tensor & grad_input) {
    static auto op = create_silu_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(silu_backward, name, "aten::silu_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(silu_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(silu_backward, schema_str, "silu_backward(Tensor grad_output, Tensor self) -> Tensor")

// aten::silu_backward(Tensor grad_output, Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<silu_backward::schema> create_silu_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(silu_backward::name, silu_backward::overload_name)
      .typed<silu_backward::schema>();
}

// aten::silu_backward(Tensor grad_output, Tensor self) -> Tensor
at::Tensor silu_backward::call(const at::Tensor & grad_output, const at::Tensor & self) {
    static auto op = create_silu_backward_typed_handle();
    return op.call(grad_output, self);
}

// aten::silu_backward(Tensor grad_output, Tensor self) -> Tensor
at::Tensor silu_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self) {
    static auto op = create_silu_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sigmoid_, name, "aten::sigmoid_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sigmoid_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sigmoid_, schema_str, "sigmoid_(Tensor(a!) self) -> Tensor(a!)")

// aten::sigmoid_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<sigmoid_::schema> create_sigmoid__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sigmoid_::name, sigmoid_::overload_name)
      .typed<sigmoid_::schema>();
}

// aten::sigmoid_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & sigmoid_::call(at::Tensor & self) {
    static auto op = create_sigmoid__typed_handle();
    return op.call(self);
}

// aten::sigmoid_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & sigmoid_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_sigmoid__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sin, name, "aten::sin")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sin, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sin, schema_str, "sin(Tensor self) -> Tensor")

// aten::sin(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<sin::schema> create_sin_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sin::name, sin::overload_name)
      .typed<sin::schema>();
}

// aten::sin(Tensor self) -> Tensor
at::Tensor sin::call(const at::Tensor & self) {
    static auto op = create_sin_typed_handle();
    return op.call(self);
}

// aten::sin(Tensor self) -> Tensor
at::Tensor sin::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_sin_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sin_out, name, "aten::sin")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sin_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sin_out, schema_str, "sin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::sin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<sin_out::schema> create_sin_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sin_out::name, sin_out::overload_name)
      .typed<sin_out::schema>();
}

// aten::sin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & sin_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_sin_out_typed_handle();
    return op.call(self, out);
}

// aten::sin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & sin_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_sin_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(size_Dimname, name, "aten::size")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(size_Dimname, overload_name, "Dimname")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(size_Dimname, schema_str, "size.Dimname(Tensor self, Dimname dim) -> int")

// aten::size.Dimname(Tensor self, Dimname dim) -> int
static C10_NOINLINE c10::TypedOperatorHandle<size_Dimname::schema> create_size_Dimname_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(size_Dimname::name, size_Dimname::overload_name)
      .typed<size_Dimname::schema>();
}

// aten::size.Dimname(Tensor self, Dimname dim) -> int
int64_t size_Dimname::call(const at::Tensor & self, at::Dimname dim) {
    static auto op = create_size_Dimname_typed_handle();
    return op.call(self, dim);
}

// aten::size.Dimname(Tensor self, Dimname dim) -> int
int64_t size_Dimname::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim) {
    static auto op = create_size_Dimname_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(softmax_Dimname, name, "aten::softmax")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(softmax_Dimname, overload_name, "Dimname")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(softmax_Dimname, schema_str, "softmax.Dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor")

// aten::softmax.Dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<softmax_Dimname::schema> create_softmax_Dimname_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(softmax_Dimname::name, softmax_Dimname::overload_name)
      .typed<softmax_Dimname::schema>();
}

// aten::softmax.Dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
at::Tensor softmax_Dimname::call(const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_softmax_Dimname_typed_handle();
    return op.call(self, dim, dtype);
}

// aten::softmax.Dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
at::Tensor softmax_Dimname::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_softmax_Dimname_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_softmax_backward_data_out, name, "aten::_softmax_backward_data")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_softmax_backward_data_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_softmax_backward_data_out, schema_str, "_softmax_backward_data.out(Tensor grad_output, Tensor output, int dim, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::_softmax_backward_data.out(Tensor grad_output, Tensor output, int dim, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<_softmax_backward_data_out::schema> create__softmax_backward_data_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_softmax_backward_data_out::name, _softmax_backward_data_out::overload_name)
      .typed<_softmax_backward_data_out::schema>();
}

// aten::_softmax_backward_data.out(Tensor grad_output, Tensor output, int dim, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & _softmax_backward_data_out::call(const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, const at::Tensor & self, at::Tensor & grad_input) {
    static auto op = create__softmax_backward_data_out_typed_handle();
    return op.call(grad_output, output, dim, self, grad_input);
}

// aten::_softmax_backward_data.out(Tensor grad_output, Tensor output, int dim, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & _softmax_backward_data_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, const at::Tensor & self, at::Tensor & grad_input) {
    static auto op = create__softmax_backward_data_out_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, output, dim, self, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unsafe_split_Tensor, name, "aten::unsafe_split")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unsafe_split_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unsafe_split_Tensor, schema_str, "unsafe_split.Tensor(Tensor self, int split_size, int dim=0) -> Tensor[]")

// aten::unsafe_split.Tensor(Tensor self, int split_size, int dim=0) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<unsafe_split_Tensor::schema> create_unsafe_split_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(unsafe_split_Tensor::name, unsafe_split_Tensor::overload_name)
      .typed<unsafe_split_Tensor::schema>();
}

// aten::unsafe_split.Tensor(Tensor self, int split_size, int dim=0) -> Tensor[]
::std::vector<at::Tensor> unsafe_split_Tensor::call(const at::Tensor & self, int64_t split_size, int64_t dim) {
    static auto op = create_unsafe_split_Tensor_typed_handle();
    return op.call(self, split_size, dim);
}

// aten::unsafe_split.Tensor(Tensor self, int split_size, int dim=0) -> Tensor[]
::std::vector<at::Tensor> unsafe_split_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t split_size, int64_t dim) {
    static auto op = create_unsafe_split_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, split_size, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(squeeze, name, "aten::squeeze")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(squeeze, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(squeeze, schema_str, "squeeze(Tensor(a) self) -> Tensor(a)")

// aten::squeeze(Tensor(a) self) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<squeeze::schema> create_squeeze_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(squeeze::name, squeeze::overload_name)
      .typed<squeeze::schema>();
}

// aten::squeeze(Tensor(a) self) -> Tensor(a)
at::Tensor squeeze::call(const at::Tensor & self) {
    static auto op = create_squeeze_typed_handle();
    return op.call(self);
}

// aten::squeeze(Tensor(a) self) -> Tensor(a)
at::Tensor squeeze::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_squeeze_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sspaddmm, name, "aten::sspaddmm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sspaddmm, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sspaddmm, schema_str, "sspaddmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor")

// aten::sspaddmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<sspaddmm::schema> create_sspaddmm_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sspaddmm::name, sspaddmm::overload_name)
      .typed<sspaddmm::schema>();
}

// aten::sspaddmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
at::Tensor sspaddmm::call(const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha) {
    static auto op = create_sspaddmm_typed_handle();
    return op.call(self, mat1, mat2, beta, alpha);
}

// aten::sspaddmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
at::Tensor sspaddmm::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha) {
    static auto op = create_sspaddmm_typed_handle();
    return op.redispatch(dispatchKeySet, self, mat1, mat2, beta, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(vstack_out, name, "aten::vstack")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(vstack_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(vstack_out, schema_str, "vstack.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)")

// aten::vstack.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<vstack_out::schema> create_vstack_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(vstack_out::name, vstack_out::overload_name)
      .typed<vstack_out::schema>();
}

// aten::vstack.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & vstack_out::call(at::TensorList tensors, at::Tensor & out) {
    static auto op = create_vstack_out_typed_handle();
    return op.call(tensors, out);
}

// aten::vstack.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & vstack_out::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, at::Tensor & out) {
    static auto op = create_vstack_out_typed_handle();
    return op.redispatch(dispatchKeySet, tensors, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sqrt_, name, "aten::sqrt_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sqrt_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sqrt_, schema_str, "sqrt_(Tensor(a!) self) -> Tensor(a!)")

// aten::sqrt_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<sqrt_::schema> create_sqrt__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sqrt_::name, sqrt_::overload_name)
      .typed<sqrt_::schema>();
}

// aten::sqrt_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & sqrt_::call(at::Tensor & self) {
    static auto op = create_sqrt__typed_handle();
    return op.call(self);
}

// aten::sqrt_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & sqrt_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_sqrt__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_dim, name, "aten::std")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_dim, overload_name, "dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_dim, schema_str, "std.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor")

// aten::std.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<std_dim::schema> create_std_dim_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(std_dim::name, std_dim::overload_name)
      .typed<std_dim::schema>();
}

// aten::std.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
at::Tensor std_dim::call(const at::Tensor & self, at::IntArrayRef dim, bool unbiased, bool keepdim) {
    static auto op = create_std_dim_typed_handle();
    return op.call(self, dim, unbiased, keepdim);
}

// aten::std.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
at::Tensor std_dim::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool unbiased, bool keepdim) {
    static auto op = create_std_dim_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, unbiased, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_mean_dim, name, "aten::std_mean")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_mean_dim, overload_name, "dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_mean_dim, schema_str, "std_mean.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)")

// aten::std_mean.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<std_mean_dim::schema> create_std_mean_dim_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(std_mean_dim::name, std_mean_dim::overload_name)
      .typed<std_mean_dim::schema>();
}

// aten::std_mean.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> std_mean_dim::call(const at::Tensor & self, at::IntArrayRef dim, bool unbiased, bool keepdim) {
    static auto op = create_std_mean_dim_typed_handle();
    return op.call(self, dim, unbiased, keepdim);
}

// aten::std_mean.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> std_mean_dim::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool unbiased, bool keepdim) {
    static auto op = create_std_mean_dim_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, unbiased, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(prod_int_out, name, "aten::prod")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(prod_int_out, overload_name, "int_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(prod_int_out, schema_str, "prod.int_out(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)")

// aten::prod.int_out(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<prod_int_out::schema> create_prod_int_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(prod_int_out::name, prod_int_out::overload_name)
      .typed<prod_int_out::schema>();
}

// aten::prod.int_out(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & prod_int_out::call(const at::Tensor & self, int64_t dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
    static auto op = create_prod_int_out_typed_handle();
    return op.call(self, dim, keepdim, dtype, out);
}

// aten::prod.int_out(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & prod_int_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
    static auto op = create_prod_int_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, dtype, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(prod_dim_Dimname, name, "aten::prod")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(prod_dim_Dimname, overload_name, "dim_Dimname")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(prod_dim_Dimname, schema_str, "prod.dim_Dimname(Tensor self, Dimname dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor")

// aten::prod.dim_Dimname(Tensor self, Dimname dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<prod_dim_Dimname::schema> create_prod_dim_Dimname_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(prod_dim_Dimname::name, prod_dim_Dimname::overload_name)
      .typed<prod_dim_Dimname::schema>();
}

// aten::prod.dim_Dimname(Tensor self, Dimname dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
at::Tensor prod_dim_Dimname::call(const at::Tensor & self, at::Dimname dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_prod_dim_Dimname_typed_handle();
    return op.call(self, dim, keepdim, dtype);
}

// aten::prod.dim_Dimname(Tensor self, Dimname dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
at::Tensor prod_dim_Dimname::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_prod_dim_Dimname_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(prod_Dimname_out, name, "aten::prod")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(prod_Dimname_out, overload_name, "Dimname_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(prod_Dimname_out, schema_str, "prod.Dimname_out(Tensor self, Dimname dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)")

// aten::prod.Dimname_out(Tensor self, Dimname dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<prod_Dimname_out::schema> create_prod_Dimname_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(prod_Dimname_out::name, prod_Dimname_out::overload_name)
      .typed<prod_Dimname_out::schema>();
}

// aten::prod.Dimname_out(Tensor self, Dimname dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & prod_Dimname_out::call(const at::Tensor & self, at::Dimname dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
    static auto op = create_prod_Dimname_out_typed_handle();
    return op.call(self, dim, keepdim, dtype, out);
}

// aten::prod.Dimname_out(Tensor self, Dimname dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & prod_Dimname_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
    static auto op = create_prod_Dimname_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, dtype, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(threshold_backward_grad_input, name, "aten::threshold_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(threshold_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(threshold_backward_grad_input, schema_str, "threshold_backward.grad_input(Tensor grad_output, Tensor self, Scalar threshold, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::threshold_backward.grad_input(Tensor grad_output, Tensor self, Scalar threshold, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<threshold_backward_grad_input::schema> create_threshold_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(threshold_backward_grad_input::name, threshold_backward_grad_input::overload_name)
      .typed<threshold_backward_grad_input::schema>();
}

// aten::threshold_backward.grad_input(Tensor grad_output, Tensor self, Scalar threshold, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & threshold_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & threshold, at::Tensor & grad_input) {
    static auto op = create_threshold_backward_grad_input_typed_handle();
    return op.call(grad_output, self, threshold, grad_input);
}

// aten::threshold_backward.grad_input(Tensor grad_output, Tensor self, Scalar threshold, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & threshold_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & threshold, at::Tensor & grad_input) {
    static auto op = create_threshold_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, threshold, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(threshold_backward, name, "aten::threshold_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(threshold_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(threshold_backward, schema_str, "threshold_backward(Tensor grad_output, Tensor self, Scalar threshold) -> Tensor")

// aten::threshold_backward(Tensor grad_output, Tensor self, Scalar threshold) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<threshold_backward::schema> create_threshold_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(threshold_backward::name, threshold_backward::overload_name)
      .typed<threshold_backward::schema>();
}

// aten::threshold_backward(Tensor grad_output, Tensor self, Scalar threshold) -> Tensor
at::Tensor threshold_backward::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & threshold) {
    static auto op = create_threshold_backward_typed_handle();
    return op.call(grad_output, self, threshold);
}

// aten::threshold_backward(Tensor grad_output, Tensor self, Scalar threshold) -> Tensor
at::Tensor threshold_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & threshold) {
    static auto op = create_threshold_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, threshold);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(one_hot, name, "aten::one_hot")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(one_hot, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(one_hot, schema_str, "one_hot(Tensor self, int num_classes=-1) -> Tensor")

// aten::one_hot(Tensor self, int num_classes=-1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<one_hot::schema> create_one_hot_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(one_hot::name, one_hot::overload_name)
      .typed<one_hot::schema>();
}

// aten::one_hot(Tensor self, int num_classes=-1) -> Tensor
at::Tensor one_hot::call(const at::Tensor & self, int64_t num_classes) {
    static auto op = create_one_hot_typed_handle();
    return op.call(self, num_classes);
}

// aten::one_hot(Tensor self, int num_classes=-1) -> Tensor
at::Tensor one_hot::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t num_classes) {
    static auto op = create_one_hot_typed_handle();
    return op.redispatch(dispatchKeySet, self, num_classes);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(trapezoid_dx, name, "aten::trapezoid")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(trapezoid_dx, overload_name, "dx")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(trapezoid_dx, schema_str, "trapezoid.dx(Tensor y, *, Scalar dx=1, int dim=-1) -> Tensor")

// aten::trapezoid.dx(Tensor y, *, Scalar dx=1, int dim=-1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<trapezoid_dx::schema> create_trapezoid_dx_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(trapezoid_dx::name, trapezoid_dx::overload_name)
      .typed<trapezoid_dx::schema>();
}

// aten::trapezoid.dx(Tensor y, *, Scalar dx=1, int dim=-1) -> Tensor
at::Tensor trapezoid_dx::call(const at::Tensor & y, const at::Scalar & dx, int64_t dim) {
    static auto op = create_trapezoid_dx_typed_handle();
    return op.call(y, dx, dim);
}

// aten::trapezoid.dx(Tensor y, *, Scalar dx=1, int dim=-1) -> Tensor
at::Tensor trapezoid_dx::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & y, const at::Scalar & dx, int64_t dim) {
    static auto op = create_trapezoid_dx_typed_handle();
    return op.redispatch(dispatchKeySet, y, dx, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fix_, name, "aten::fix_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fix_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fix_, schema_str, "fix_(Tensor(a!) self) -> Tensor(a!)")

// aten::fix_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<fix_::schema> create_fix__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fix_::name, fix_::overload_name)
      .typed<fix_::schema>();
}

// aten::fix_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & fix_::call(at::Tensor & self) {
    static auto op = create_fix__typed_handle();
    return op.call(self);
}

// aten::fix_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & fix_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_fix__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_unique, name, "aten::_unique")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_unique, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_unique, schema_str, "_unique(Tensor self, bool sorted=True, bool return_inverse=False) -> (Tensor, Tensor)")

// aten::_unique(Tensor self, bool sorted=True, bool return_inverse=False) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<_unique::schema> create__unique_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_unique::name, _unique::overload_name)
      .typed<_unique::schema>();
}

// aten::_unique(Tensor self, bool sorted=True, bool return_inverse=False) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> _unique::call(const at::Tensor & self, bool sorted, bool return_inverse) {
    static auto op = create__unique_typed_handle();
    return op.call(self, sorted, return_inverse);
}

// aten::_unique(Tensor self, bool sorted=True, bool return_inverse=False) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> _unique::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool sorted, bool return_inverse) {
    static auto op = create__unique_typed_handle();
    return op.redispatch(dispatchKeySet, self, sorted, return_inverse);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_out, name, "aten::var")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_out, schema_str, "var.out(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)")

// aten::var.out(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<var_out::schema> create_var_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(var_out::name, var_out::overload_name)
      .typed<var_out::schema>();
}

// aten::var.out(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & var_out::call(const at::Tensor & self, at::IntArrayRef dim, bool unbiased, bool keepdim, at::Tensor & out) {
    static auto op = create_var_out_typed_handle();
    return op.call(self, dim, unbiased, keepdim, out);
}

// aten::var.out(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & var_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool unbiased, bool keepdim, at::Tensor & out) {
    static auto op = create_var_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, unbiased, keepdim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_names_dim, name, "aten::var")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_names_dim, overload_name, "names_dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_names_dim, schema_str, "var.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor")

// aten::var.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<var_names_dim::schema> create_var_names_dim_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(var_names_dim::name, var_names_dim::overload_name)
      .typed<var_names_dim::schema>();
}

// aten::var.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
at::Tensor var_names_dim::call(const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim) {
    static auto op = create_var_names_dim_typed_handle();
    return op.call(self, dim, unbiased, keepdim);
}

// aten::var.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
at::Tensor var_names_dim::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim) {
    static auto op = create_var_names_dim_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, unbiased, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_correction_names_out, name, "aten::var")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_correction_names_out, overload_name, "correction_names_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_correction_names_out, schema_str, "var.correction_names_out(Tensor self, Dimname[1] dim, *, int? correction, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)")

// aten::var.correction_names_out(Tensor self, Dimname[1] dim, *, int? correction, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<var_correction_names_out::schema> create_var_correction_names_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(var_correction_names_out::name, var_correction_names_out::overload_name)
      .typed<var_correction_names_out::schema>();
}

// aten::var.correction_names_out(Tensor self, Dimname[1] dim, *, int? correction, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)
at::Tensor & var_correction_names_out::call(const at::Tensor & self, at::DimnameList dim, c10::optional<int64_t> correction, bool keepdim, at::Tensor & out) {
    static auto op = create_var_correction_names_out_typed_handle();
    return op.call(self, dim, correction, keepdim, out);
}

// aten::var.correction_names_out(Tensor self, Dimname[1] dim, *, int? correction, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)
at::Tensor & var_correction_names_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList dim, c10::optional<int64_t> correction, bool keepdim, at::Tensor & out) {
    static auto op = create_var_correction_names_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, correction, keepdim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_mean_correction_names, name, "aten::var_mean")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_mean_correction_names, overload_name, "correction_names")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_mean_correction_names, schema_str, "var_mean.correction_names(Tensor self, Dimname[1] dim, *, int? correction, bool keepdim=False) -> (Tensor, Tensor)")

// aten::var_mean.correction_names(Tensor self, Dimname[1] dim, *, int? correction, bool keepdim=False) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<var_mean_correction_names::schema> create_var_mean_correction_names_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(var_mean_correction_names::name, var_mean_correction_names::overload_name)
      .typed<var_mean_correction_names::schema>();
}

// aten::var_mean.correction_names(Tensor self, Dimname[1] dim, *, int? correction, bool keepdim=False) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> var_mean_correction_names::call(const at::Tensor & self, at::DimnameList dim, c10::optional<int64_t> correction, bool keepdim) {
    static auto op = create_var_mean_correction_names_typed_handle();
    return op.call(self, dim, correction, keepdim);
}

// aten::var_mean.correction_names(Tensor self, Dimname[1] dim, *, int? correction, bool keepdim=False) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> var_mean_correction_names::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList dim, c10::optional<int64_t> correction, bool keepdim) {
    static auto op = create_var_mean_correction_names_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, correction, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(where_self, name, "aten::where")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(where_self, overload_name, "self")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(where_self, schema_str, "where.self(Tensor condition, Tensor self, Tensor other) -> Tensor")

// aten::where.self(Tensor condition, Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<where_self::schema> create_where_self_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(where_self::name, where_self::overload_name)
      .typed<where_self::schema>();
}

// aten::where.self(Tensor condition, Tensor self, Tensor other) -> Tensor
at::Tensor where_self::call(const at::Tensor & condition, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_where_self_typed_handle();
    return op.call(condition, self, other);
}

// aten::where.self(Tensor condition, Tensor self, Tensor other) -> Tensor
at::Tensor where_self::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & condition, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_where_self_typed_handle();
    return op.redispatch(dispatchKeySet, condition, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(where, name, "aten::where")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(where, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(where, schema_str, "where(Tensor condition) -> Tensor[]")

// aten::where(Tensor condition) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<where::schema> create_where_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(where::name, where::overload_name)
      .typed<where::schema>();
}

// aten::where(Tensor condition) -> Tensor[]
::std::vector<at::Tensor> where::call(const at::Tensor & condition) {
    static auto op = create_where_typed_handle();
    return op.call(condition);
}

// aten::where(Tensor condition) -> Tensor[]
::std::vector<at::Tensor> where::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & condition) {
    static auto op = create_where_typed_handle();
    return op.redispatch(dispatchKeySet, condition);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_weight_norm, name, "aten::_weight_norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_weight_norm, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_weight_norm, schema_str, "_weight_norm(Tensor v, Tensor g, int dim=0) -> Tensor")

// aten::_weight_norm(Tensor v, Tensor g, int dim=0) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_weight_norm::schema> create__weight_norm_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_weight_norm::name, _weight_norm::overload_name)
      .typed<_weight_norm::schema>();
}

// aten::_weight_norm(Tensor v, Tensor g, int dim=0) -> Tensor
at::Tensor _weight_norm::call(const at::Tensor & v, const at::Tensor & g, int64_t dim) {
    static auto op = create__weight_norm_typed_handle();
    return op.call(v, g, dim);
}

// aten::_weight_norm(Tensor v, Tensor g, int dim=0) -> Tensor
at::Tensor _weight_norm::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & v, const at::Tensor & g, int64_t dim) {
    static auto op = create__weight_norm_typed_handle();
    return op.redispatch(dispatchKeySet, v, g, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_weight_norm_differentiable_backward, name, "aten::_weight_norm_differentiable_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_weight_norm_differentiable_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_weight_norm_differentiable_backward, schema_str, "_weight_norm_differentiable_backward(Tensor grad_w, Tensor saved_v, Tensor saved_g, Tensor saved_norms, int dim) -> (Tensor, Tensor)")

// aten::_weight_norm_differentiable_backward(Tensor grad_w, Tensor saved_v, Tensor saved_g, Tensor saved_norms, int dim) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<_weight_norm_differentiable_backward::schema> create__weight_norm_differentiable_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_weight_norm_differentiable_backward::name, _weight_norm_differentiable_backward::overload_name)
      .typed<_weight_norm_differentiable_backward::schema>();
}

// aten::_weight_norm_differentiable_backward(Tensor grad_w, Tensor saved_v, Tensor saved_g, Tensor saved_norms, int dim) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> _weight_norm_differentiable_backward::call(const at::Tensor & grad_w, const at::Tensor & saved_v, const at::Tensor & saved_g, const at::Tensor & saved_norms, int64_t dim) {
    static auto op = create__weight_norm_differentiable_backward_typed_handle();
    return op.call(grad_w, saved_v, saved_g, saved_norms, dim);
}

// aten::_weight_norm_differentiable_backward(Tensor grad_w, Tensor saved_v, Tensor saved_g, Tensor saved_norms, int dim) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> _weight_norm_differentiable_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_w, const at::Tensor & saved_v, const at::Tensor & saved_g, const at::Tensor & saved_norms, int64_t dim) {
    static auto op = create__weight_norm_differentiable_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_w, saved_v, saved_g, saved_norms, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(zeros, name, "aten::zeros")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(zeros, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(zeros, schema_str, "zeros(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::zeros(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<zeros::schema> create_zeros_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(zeros::name, zeros::overload_name)
      .typed<zeros::schema>();
}

// aten::zeros(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor zeros::call(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_zeros_typed_handle();
    return op.call(size, dtype, layout, device, pin_memory);
}

// aten::zeros(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor zeros::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_zeros_typed_handle();
    return op.redispatch(dispatchKeySet, size, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_standard_gamma, name, "aten::_standard_gamma")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_standard_gamma, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_standard_gamma, schema_str, "_standard_gamma(Tensor self, Generator? generator=None) -> Tensor")

// aten::_standard_gamma(Tensor self, Generator? generator=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_standard_gamma::schema> create__standard_gamma_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_standard_gamma::name, _standard_gamma::overload_name)
      .typed<_standard_gamma::schema>();
}

// aten::_standard_gamma(Tensor self, Generator? generator=None) -> Tensor
at::Tensor _standard_gamma::call(const at::Tensor & self, c10::optional<at::Generator> generator) {
    static auto op = create__standard_gamma_typed_handle();
    return op.call(self, generator);
}

// aten::_standard_gamma(Tensor self, Generator? generator=None) -> Tensor
at::Tensor _standard_gamma::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::Generator> generator) {
    static auto op = create__standard_gamma_typed_handle();
    return op.redispatch(dispatchKeySet, self, generator);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sample_dirichlet, name, "aten::_sample_dirichlet")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sample_dirichlet, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sample_dirichlet, schema_str, "_sample_dirichlet(Tensor self, Generator? generator=None) -> Tensor")

// aten::_sample_dirichlet(Tensor self, Generator? generator=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_sample_dirichlet::schema> create__sample_dirichlet_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_sample_dirichlet::name, _sample_dirichlet::overload_name)
      .typed<_sample_dirichlet::schema>();
}

// aten::_sample_dirichlet(Tensor self, Generator? generator=None) -> Tensor
at::Tensor _sample_dirichlet::call(const at::Tensor & self, c10::optional<at::Generator> generator) {
    static auto op = create__sample_dirichlet_typed_handle();
    return op.call(self, generator);
}

// aten::_sample_dirichlet(Tensor self, Generator? generator=None) -> Tensor
at::Tensor _sample_dirichlet::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::Generator> generator) {
    static auto op = create__sample_dirichlet_typed_handle();
    return op.redispatch(dispatchKeySet, self, generator);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(binomial, name, "aten::binomial")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(binomial, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(binomial, schema_str, "binomial(Tensor count, Tensor prob, Generator? generator=None) -> Tensor")

// aten::binomial(Tensor count, Tensor prob, Generator? generator=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<binomial::schema> create_binomial_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(binomial::name, binomial::overload_name)
      .typed<binomial::schema>();
}

// aten::binomial(Tensor count, Tensor prob, Generator? generator=None) -> Tensor
at::Tensor binomial::call(const at::Tensor & count, const at::Tensor & prob, c10::optional<at::Generator> generator) {
    static auto op = create_binomial_typed_handle();
    return op.call(count, prob, generator);
}

// aten::binomial(Tensor count, Tensor prob, Generator? generator=None) -> Tensor
at::Tensor binomial::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & count, const at::Tensor & prob, c10::optional<at::Generator> generator) {
    static auto op = create_binomial_typed_handle();
    return op.redispatch(dispatchKeySet, count, prob, generator);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_sum, name, "aten::_sparse_sum")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_sum, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_sum, schema_str, "_sparse_sum(Tensor self) -> Tensor")

// aten::_sparse_sum(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_sparse_sum::schema> create__sparse_sum_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_sparse_sum::name, _sparse_sum::overload_name)
      .typed<_sparse_sum::schema>();
}

// aten::_sparse_sum(Tensor self) -> Tensor
at::Tensor _sparse_sum::call(const at::Tensor & self) {
    static auto op = create__sparse_sum_typed_handle();
    return op.call(self);
}

// aten::_sparse_sum(Tensor self) -> Tensor
at::Tensor _sparse_sum::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create__sparse_sum_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_sum_dim, name, "aten::_sparse_sum")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_sum_dim, overload_name, "dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_sum_dim, schema_str, "_sparse_sum.dim(Tensor self, int[1] dim) -> Tensor")

// aten::_sparse_sum.dim(Tensor self, int[1] dim) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_sparse_sum_dim::schema> create__sparse_sum_dim_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_sparse_sum_dim::name, _sparse_sum_dim::overload_name)
      .typed<_sparse_sum_dim::schema>();
}

// aten::_sparse_sum.dim(Tensor self, int[1] dim) -> Tensor
at::Tensor _sparse_sum_dim::call(const at::Tensor & self, at::IntArrayRef dim) {
    static auto op = create__sparse_sum_dim_typed_handle();
    return op.call(self, dim);
}

// aten::_sparse_sum.dim(Tensor self, int[1] dim) -> Tensor
at::Tensor _sparse_sum_dim::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim) {
    static auto op = create__sparse_sum_dim_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(norm_names_ScalarOpt_dim, name, "aten::norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(norm_names_ScalarOpt_dim, overload_name, "names_ScalarOpt_dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(norm_names_ScalarOpt_dim, schema_str, "norm.names_ScalarOpt_dim(Tensor self, Scalar? p, Dimname[1] dim, bool keepdim=False) -> Tensor")

// aten::norm.names_ScalarOpt_dim(Tensor self, Scalar? p, Dimname[1] dim, bool keepdim=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<norm_names_ScalarOpt_dim::schema> create_norm_names_ScalarOpt_dim_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(norm_names_ScalarOpt_dim::name, norm_names_ScalarOpt_dim::overload_name)
      .typed<norm_names_ScalarOpt_dim::schema>();
}

// aten::norm.names_ScalarOpt_dim(Tensor self, Scalar? p, Dimname[1] dim, bool keepdim=False) -> Tensor
at::Tensor norm_names_ScalarOpt_dim::call(const at::Tensor & self, const c10::optional<at::Scalar> & p, at::DimnameList dim, bool keepdim) {
    static auto op = create_norm_names_ScalarOpt_dim_typed_handle();
    return op.call(self, p, dim, keepdim);
}

// aten::norm.names_ScalarOpt_dim(Tensor self, Scalar? p, Dimname[1] dim, bool keepdim=False) -> Tensor
at::Tensor norm_names_ScalarOpt_dim::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Scalar> & p, at::DimnameList dim, bool keepdim) {
    static auto op = create_norm_names_ScalarOpt_dim_typed_handle();
    return op.redispatch(dispatchKeySet, self, p, dim, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(zero_, name, "aten::zero_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(zero_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(zero_, schema_str, "zero_(Tensor(a!) self) -> Tensor(a!)")

// aten::zero_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<zero_::schema> create_zero__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(zero_::name, zero_::overload_name)
      .typed<zero_::schema>();
}

// aten::zero_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & zero_::call(at::Tensor & self) {
    static auto op = create_zero__typed_handle();
    return op.call(self);
}

// aten::zero_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & zero_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_zero__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rsub_Scalar, name, "aten::rsub")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rsub_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rsub_Scalar, schema_str, "rsub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor")

// aten::rsub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<rsub_Scalar::schema> create_rsub_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(rsub_Scalar::name, rsub_Scalar::overload_name)
      .typed<rsub_Scalar::schema>();
}

// aten::rsub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
at::Tensor rsub_Scalar::call(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
    static auto op = create_rsub_Scalar_typed_handle();
    return op.call(self, other, alpha);
}

// aten::rsub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
at::Tensor rsub_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
    static auto op = create_rsub_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_addmm, name, "aten::_sparse_addmm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_addmm, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_addmm, schema_str, "_sparse_addmm(Tensor self, Tensor sparse, Tensor dense, *, Scalar beta=1, Scalar alpha=1) -> Tensor")

// aten::_sparse_addmm(Tensor self, Tensor sparse, Tensor dense, *, Scalar beta=1, Scalar alpha=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_sparse_addmm::schema> create__sparse_addmm_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_sparse_addmm::name, _sparse_addmm::overload_name)
      .typed<_sparse_addmm::schema>();
}

// aten::_sparse_addmm(Tensor self, Tensor sparse, Tensor dense, *, Scalar beta=1, Scalar alpha=1) -> Tensor
at::Tensor _sparse_addmm::call(const at::Tensor & self, const at::Tensor & sparse, const at::Tensor & dense, const at::Scalar & beta, const at::Scalar & alpha) {
    static auto op = create__sparse_addmm_typed_handle();
    return op.call(self, sparse, dense, beta, alpha);
}

// aten::_sparse_addmm(Tensor self, Tensor sparse, Tensor dense, *, Scalar beta=1, Scalar alpha=1) -> Tensor
at::Tensor _sparse_addmm::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & sparse, const at::Tensor & dense, const at::Scalar & beta, const at::Scalar & alpha) {
    static auto op = create__sparse_addmm_typed_handle();
    return op.redispatch(dispatchKeySet, self, sparse, dense, beta, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addmm, name, "aten::addmm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addmm, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addmm, schema_str, "addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor")

// aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<addmm::schema> create_addmm_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(addmm::name, addmm::overload_name)
      .typed<addmm::schema>();
}

// aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
at::Tensor addmm::call(const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha) {
    static auto op = create_addmm_typed_handle();
    return op.call(self, mat1, mat2, beta, alpha);
}

// aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
at::Tensor addmm::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha) {
    static auto op = create_addmm_typed_handle();
    return op.redispatch(dispatchKeySet, self, mat1, mat2, beta, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_csr_tensor_unsafe, name, "aten::_sparse_csr_tensor_unsafe")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_csr_tensor_unsafe, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_csr_tensor_unsafe, schema_str, "_sparse_csr_tensor_unsafe(Tensor crow_indices, Tensor col_indices, Tensor values, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::_sparse_csr_tensor_unsafe(Tensor crow_indices, Tensor col_indices, Tensor values, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_sparse_csr_tensor_unsafe::schema> create__sparse_csr_tensor_unsafe_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_sparse_csr_tensor_unsafe::name, _sparse_csr_tensor_unsafe::overload_name)
      .typed<_sparse_csr_tensor_unsafe::schema>();
}

// aten::_sparse_csr_tensor_unsafe(Tensor crow_indices, Tensor col_indices, Tensor values, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor _sparse_csr_tensor_unsafe::call(const at::Tensor & crow_indices, const at::Tensor & col_indices, const at::Tensor & values, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create__sparse_csr_tensor_unsafe_typed_handle();
    return op.call(crow_indices, col_indices, values, size, dtype, layout, device, pin_memory);
}

// aten::_sparse_csr_tensor_unsafe(Tensor crow_indices, Tensor col_indices, Tensor values, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor _sparse_csr_tensor_unsafe::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & crow_indices, const at::Tensor & col_indices, const at::Tensor & values, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create__sparse_csr_tensor_unsafe_typed_handle();
    return op.redispatch(dispatchKeySet, crow_indices, col_indices, values, size, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_coo_tensor_unsafe, name, "aten::_sparse_coo_tensor_unsafe")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_coo_tensor_unsafe, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_coo_tensor_unsafe, schema_str, "_sparse_coo_tensor_unsafe(Tensor indices, Tensor values, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::_sparse_coo_tensor_unsafe(Tensor indices, Tensor values, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_sparse_coo_tensor_unsafe::schema> create__sparse_coo_tensor_unsafe_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_sparse_coo_tensor_unsafe::name, _sparse_coo_tensor_unsafe::overload_name)
      .typed<_sparse_coo_tensor_unsafe::schema>();
}

// aten::_sparse_coo_tensor_unsafe(Tensor indices, Tensor values, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor _sparse_coo_tensor_unsafe::call(const at::Tensor & indices, const at::Tensor & values, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create__sparse_coo_tensor_unsafe_typed_handle();
    return op.call(indices, values, size, dtype, layout, device, pin_memory);
}

// aten::_sparse_coo_tensor_unsafe(Tensor indices, Tensor values, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor _sparse_coo_tensor_unsafe::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & indices, const at::Tensor & values, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create__sparse_coo_tensor_unsafe_typed_handle();
    return op.redispatch(dispatchKeySet, indices, values, size, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_validate_sparse_csr_tensor_args, name, "aten::_validate_sparse_csr_tensor_args")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_validate_sparse_csr_tensor_args, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_validate_sparse_csr_tensor_args, schema_str, "_validate_sparse_csr_tensor_args(Tensor crow_indices, Tensor col_indices, Tensor values, int[] size) -> ()")

// aten::_validate_sparse_csr_tensor_args(Tensor crow_indices, Tensor col_indices, Tensor values, int[] size) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_validate_sparse_csr_tensor_args::schema> create__validate_sparse_csr_tensor_args_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_validate_sparse_csr_tensor_args::name, _validate_sparse_csr_tensor_args::overload_name)
      .typed<_validate_sparse_csr_tensor_args::schema>();
}

// aten::_validate_sparse_csr_tensor_args(Tensor crow_indices, Tensor col_indices, Tensor values, int[] size) -> ()
void _validate_sparse_csr_tensor_args::call(const at::Tensor & crow_indices, const at::Tensor & col_indices, const at::Tensor & values, at::IntArrayRef size) {
    static auto op = create__validate_sparse_csr_tensor_args_typed_handle();
    return op.call(crow_indices, col_indices, values, size);
}

// aten::_validate_sparse_csr_tensor_args(Tensor crow_indices, Tensor col_indices, Tensor values, int[] size) -> ()
void _validate_sparse_csr_tensor_args::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & crow_indices, const at::Tensor & col_indices, const at::Tensor & values, at::IntArrayRef size) {
    static auto op = create__validate_sparse_csr_tensor_args_typed_handle();
    return op.redispatch(dispatchKeySet, crow_indices, col_indices, values, size);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sparse_resize_and_clear_, name, "aten::sparse_resize_and_clear_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sparse_resize_and_clear_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sparse_resize_and_clear_, schema_str, "sparse_resize_and_clear_(Tensor(a!) self, int[] size, int sparse_dim, int dense_dim) -> Tensor(a!)")

// aten::sparse_resize_and_clear_(Tensor(a!) self, int[] size, int sparse_dim, int dense_dim) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<sparse_resize_and_clear_::schema> create_sparse_resize_and_clear__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sparse_resize_and_clear_::name, sparse_resize_and_clear_::overload_name)
      .typed<sparse_resize_and_clear_::schema>();
}

// aten::sparse_resize_and_clear_(Tensor(a!) self, int[] size, int sparse_dim, int dense_dim) -> Tensor(a!)
const at::Tensor & sparse_resize_and_clear_::call(const at::Tensor & self, at::IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) {
    static auto op = create_sparse_resize_and_clear__typed_handle();
    return op.call(self, size, sparse_dim, dense_dim);
}

// aten::sparse_resize_and_clear_(Tensor(a!) self, int[] size, int sparse_dim, int dense_dim) -> Tensor(a!)
const at::Tensor & sparse_resize_and_clear_::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) {
    static auto op = create_sparse_resize_and_clear__typed_handle();
    return op.redispatch(dispatchKeySet, self, size, sparse_dim, dense_dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sparse_mask, name, "aten::sparse_mask")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sparse_mask, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sparse_mask, schema_str, "sparse_mask(Tensor self, Tensor mask) -> Tensor")

// aten::sparse_mask(Tensor self, Tensor mask) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<sparse_mask::schema> create_sparse_mask_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sparse_mask::name, sparse_mask::overload_name)
      .typed<sparse_mask::schema>();
}

// aten::sparse_mask(Tensor self, Tensor mask) -> Tensor
at::Tensor sparse_mask::call(const at::Tensor & self, const at::Tensor & mask) {
    static auto op = create_sparse_mask_typed_handle();
    return op.call(self, mask);
}

// aten::sparse_mask(Tensor self, Tensor mask) -> Tensor
at::Tensor sparse_mask::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mask) {
    static auto op = create_sparse_mask_typed_handle();
    return op.redispatch(dispatchKeySet, self, mask);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_to_cpu, name, "aten::_to_cpu")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_to_cpu, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_to_cpu, schema_str, "_to_cpu(Tensor[] tensors) -> Tensor[]")

// aten::_to_cpu(Tensor[] tensors) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_to_cpu::schema> create__to_cpu_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_to_cpu::name, _to_cpu::overload_name)
      .typed<_to_cpu::schema>();
}

// aten::_to_cpu(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _to_cpu::call(at::TensorList tensors) {
    static auto op = create__to_cpu_typed_handle();
    return op.call(tensors);
}

// aten::_to_cpu(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _to_cpu::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create__to_cpu_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(values, name, "aten::values")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(values, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(values, schema_str, "values(Tensor(a) self) -> Tensor(a)")

// aten::values(Tensor(a) self) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<values::schema> create_values_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(values::name, values::overload_name)
      .typed<values::schema>();
}

// aten::values(Tensor(a) self) -> Tensor(a)
at::Tensor values::call(const at::Tensor & self) {
    static auto op = create_values_typed_handle();
    return op.call(self);
}

// aten::values(Tensor(a) self) -> Tensor(a)
at::Tensor values::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_values_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hspmm_out, name, "aten::hspmm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hspmm_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hspmm_out, schema_str, "hspmm.out(Tensor mat1, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)")

// aten::hspmm.out(Tensor mat1, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<hspmm_out::schema> create_hspmm_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(hspmm_out::name, hspmm_out::overload_name)
      .typed<hspmm_out::schema>();
}

// aten::hspmm.out(Tensor mat1, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & hspmm_out::call(const at::Tensor & mat1, const at::Tensor & mat2, at::Tensor & out) {
    static auto op = create_hspmm_out_typed_handle();
    return op.call(mat1, mat2, out);
}

// aten::hspmm.out(Tensor mat1, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & hspmm_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & mat1, const at::Tensor & mat2, at::Tensor & out) {
    static auto op = create_hspmm_out_typed_handle();
    return op.redispatch(dispatchKeySet, mat1, mat2, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(copy_sparse_to_sparse_, name, "aten::copy_sparse_to_sparse_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(copy_sparse_to_sparse_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(copy_sparse_to_sparse_, schema_str, "copy_sparse_to_sparse_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)")

// aten::copy_sparse_to_sparse_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<copy_sparse_to_sparse_::schema> create_copy_sparse_to_sparse__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(copy_sparse_to_sparse_::name, copy_sparse_to_sparse_::overload_name)
      .typed<copy_sparse_to_sparse_::schema>();
}

// aten::copy_sparse_to_sparse_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)
at::Tensor & copy_sparse_to_sparse_::call(at::Tensor & self, const at::Tensor & src, bool non_blocking) {
    static auto op = create_copy_sparse_to_sparse__typed_handle();
    return op.call(self, src, non_blocking);
}

// aten::copy_sparse_to_sparse_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)
at::Tensor & copy_sparse_to_sparse_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & src, bool non_blocking) {
    static auto op = create_copy_sparse_to_sparse__typed_handle();
    return op.redispatch(dispatchKeySet, self, src, non_blocking);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(to_sparse, name, "aten::to_sparse")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(to_sparse, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(to_sparse, schema_str, "to_sparse(Tensor self) -> Tensor")

// aten::to_sparse(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<to_sparse::schema> create_to_sparse_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(to_sparse::name, to_sparse::overload_name)
      .typed<to_sparse::schema>();
}

// aten::to_sparse(Tensor self) -> Tensor
at::Tensor to_sparse::call(const at::Tensor & self) {
    static auto op = create_to_sparse_typed_handle();
    return op.call(self);
}

// aten::to_sparse(Tensor self) -> Tensor
at::Tensor to_sparse::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_to_sparse_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(to_mkldnn, name, "aten::to_mkldnn")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(to_mkldnn, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(to_mkldnn, schema_str, "to_mkldnn(Tensor self, ScalarType? dtype=None) -> Tensor")

// aten::to_mkldnn(Tensor self, ScalarType? dtype=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<to_mkldnn::schema> create_to_mkldnn_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(to_mkldnn::name, to_mkldnn::overload_name)
      .typed<to_mkldnn::schema>();
}

// aten::to_mkldnn(Tensor self, ScalarType? dtype=None) -> Tensor
at::Tensor to_mkldnn::call(const at::Tensor & self, c10::optional<at::ScalarType> dtype) {
    static auto op = create_to_mkldnn_typed_handle();
    return op.call(self, dtype);
}

// aten::to_mkldnn(Tensor self, ScalarType? dtype=None) -> Tensor
at::Tensor to_mkldnn::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::ScalarType> dtype) {
    static auto op = create_to_mkldnn_typed_handle();
    return op.redispatch(dispatchKeySet, self, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(to_mkldnn_backward, name, "aten::to_mkldnn_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(to_mkldnn_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(to_mkldnn_backward, schema_str, "to_mkldnn_backward(Tensor grad, Tensor input) -> Tensor")

// aten::to_mkldnn_backward(Tensor grad, Tensor input) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<to_mkldnn_backward::schema> create_to_mkldnn_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(to_mkldnn_backward::name, to_mkldnn_backward::overload_name)
      .typed<to_mkldnn_backward::schema>();
}

// aten::to_mkldnn_backward(Tensor grad, Tensor input) -> Tensor
at::Tensor to_mkldnn_backward::call(const at::Tensor & grad, const at::Tensor & input) {
    static auto op = create_to_mkldnn_backward_typed_handle();
    return op.call(grad, input);
}

// aten::to_mkldnn_backward(Tensor grad, Tensor input) -> Tensor
at::Tensor to_mkldnn_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & input) {
    static auto op = create_to_mkldnn_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad, input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantize_per_tensor_tensors, name, "aten::quantize_per_tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantize_per_tensor_tensors, overload_name, "tensors")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantize_per_tensor_tensors, schema_str, "quantize_per_tensor.tensors(Tensor[] tensors, Tensor scales, Tensor zero_points, ScalarType dtype) -> Tensor[]")

// aten::quantize_per_tensor.tensors(Tensor[] tensors, Tensor scales, Tensor zero_points, ScalarType dtype) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<quantize_per_tensor_tensors::schema> create_quantize_per_tensor_tensors_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(quantize_per_tensor_tensors::name, quantize_per_tensor_tensors::overload_name)
      .typed<quantize_per_tensor_tensors::schema>();
}

// aten::quantize_per_tensor.tensors(Tensor[] tensors, Tensor scales, Tensor zero_points, ScalarType dtype) -> Tensor[]
::std::vector<at::Tensor> quantize_per_tensor_tensors::call(at::TensorList tensors, const at::Tensor & scales, const at::Tensor & zero_points, at::ScalarType dtype) {
    static auto op = create_quantize_per_tensor_tensors_typed_handle();
    return op.call(tensors, scales, zero_points, dtype);
}

// aten::quantize_per_tensor.tensors(Tensor[] tensors, Tensor scales, Tensor zero_points, ScalarType dtype) -> Tensor[]
::std::vector<at::Tensor> quantize_per_tensor_tensors::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, const at::Tensor & scales, const at::Tensor & zero_points, at::ScalarType dtype) {
    static auto op = create_quantize_per_tensor_tensors_typed_handle();
    return op.redispatch(dispatchKeySet, tensors, scales, zero_points, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(int_repr, name, "aten::int_repr")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(int_repr, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(int_repr, schema_str, "int_repr(Tensor self) -> Tensor")

// aten::int_repr(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<int_repr::schema> create_int_repr_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(int_repr::name, int_repr::overload_name)
      .typed<int_repr::schema>();
}

// aten::int_repr(Tensor self) -> Tensor
at::Tensor int_repr::call(const at::Tensor & self) {
    static auto op = create_int_repr_typed_handle();
    return op.call(self);
}

// aten::int_repr(Tensor self) -> Tensor
at::Tensor int_repr::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_int_repr_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(qscheme, name, "aten::qscheme")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(qscheme, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(qscheme, schema_str, "qscheme(Tensor self) -> QScheme")

// aten::qscheme(Tensor self) -> QScheme
static C10_NOINLINE c10::TypedOperatorHandle<qscheme::schema> create_qscheme_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(qscheme::name, qscheme::overload_name)
      .typed<qscheme::schema>();
}

// aten::qscheme(Tensor self) -> QScheme
at::QScheme qscheme::call(const at::Tensor & self) {
    static auto op = create_qscheme_typed_handle();
    return op.call(self);
}

// aten::qscheme(Tensor self) -> QScheme
at::QScheme qscheme::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_qscheme_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fake_quantize_per_channel_affine, name, "aten::fake_quantize_per_channel_affine")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fake_quantize_per_channel_affine, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fake_quantize_per_channel_affine, schema_str, "fake_quantize_per_channel_affine(Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max) -> Tensor")

// aten::fake_quantize_per_channel_affine(Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fake_quantize_per_channel_affine::schema> create_fake_quantize_per_channel_affine_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fake_quantize_per_channel_affine::name, fake_quantize_per_channel_affine::overload_name)
      .typed<fake_quantize_per_channel_affine::schema>();
}

// aten::fake_quantize_per_channel_affine(Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max) -> Tensor
at::Tensor fake_quantize_per_channel_affine::call(const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max) {
    static auto op = create_fake_quantize_per_channel_affine_typed_handle();
    return op.call(self, scale, zero_point, axis, quant_min, quant_max);
}

// aten::fake_quantize_per_channel_affine(Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max) -> Tensor
at::Tensor fake_quantize_per_channel_affine::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max) {
    static auto op = create_fake_quantize_per_channel_affine_typed_handle();
    return op.redispatch(dispatchKeySet, self, scale, zero_point, axis, quant_min, quant_max);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fake_quantize_per_channel_affine_cachemask, name, "aten::fake_quantize_per_channel_affine_cachemask")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fake_quantize_per_channel_affine_cachemask, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fake_quantize_per_channel_affine_cachemask, schema_str, "fake_quantize_per_channel_affine_cachemask(Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max) -> (Tensor output, Tensor mask)")

// aten::fake_quantize_per_channel_affine_cachemask(Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max) -> (Tensor output, Tensor mask)
static C10_NOINLINE c10::TypedOperatorHandle<fake_quantize_per_channel_affine_cachemask::schema> create_fake_quantize_per_channel_affine_cachemask_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fake_quantize_per_channel_affine_cachemask::name, fake_quantize_per_channel_affine_cachemask::overload_name)
      .typed<fake_quantize_per_channel_affine_cachemask::schema>();
}

// aten::fake_quantize_per_channel_affine_cachemask(Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max) -> (Tensor output, Tensor mask)
::std::tuple<at::Tensor,at::Tensor> fake_quantize_per_channel_affine_cachemask::call(const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max) {
    static auto op = create_fake_quantize_per_channel_affine_cachemask_typed_handle();
    return op.call(self, scale, zero_point, axis, quant_min, quant_max);
}

// aten::fake_quantize_per_channel_affine_cachemask(Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max) -> (Tensor output, Tensor mask)
::std::tuple<at::Tensor,at::Tensor> fake_quantize_per_channel_affine_cachemask::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max) {
    static auto op = create_fake_quantize_per_channel_affine_cachemask_typed_handle();
    return op.redispatch(dispatchKeySet, self, scale, zero_point, axis, quant_min, quant_max);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fused_moving_avg_obs_fq_helper, name, "aten::_fused_moving_avg_obs_fq_helper")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fused_moving_avg_obs_fq_helper, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fused_moving_avg_obs_fq_helper, schema_str, "_fused_moving_avg_obs_fq_helper(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor(a!) running_min, Tensor(b!) running_max, Tensor(c!) scale, Tensor(d!) zero_point, float averaging_const, int quant_min, int quant_max, int ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False) -> (Tensor output, Tensor mask)")

// aten::_fused_moving_avg_obs_fq_helper(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor(a!) running_min, Tensor(b!) running_max, Tensor(c!) scale, Tensor(d!) zero_point, float averaging_const, int quant_min, int quant_max, int ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False) -> (Tensor output, Tensor mask)
static C10_NOINLINE c10::TypedOperatorHandle<_fused_moving_avg_obs_fq_helper::schema> create__fused_moving_avg_obs_fq_helper_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_fused_moving_avg_obs_fq_helper::name, _fused_moving_avg_obs_fq_helper::overload_name)
      .typed<_fused_moving_avg_obs_fq_helper::schema>();
}

// aten::_fused_moving_avg_obs_fq_helper(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor(a!) running_min, Tensor(b!) running_max, Tensor(c!) scale, Tensor(d!) zero_point, float averaging_const, int quant_min, int quant_max, int ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False) -> (Tensor output, Tensor mask)
::std::tuple<at::Tensor,at::Tensor> _fused_moving_avg_obs_fq_helper::call(const at::Tensor & self, const at::Tensor & observer_on, const at::Tensor & fake_quant_on, at::Tensor & running_min, at::Tensor & running_max, at::Tensor & scale, at::Tensor & zero_point, double averaging_const, int64_t quant_min, int64_t quant_max, int64_t ch_axis, bool per_row_fake_quant, bool symmetric_quant) {
    static auto op = create__fused_moving_avg_obs_fq_helper_typed_handle();
    return op.call(self, observer_on, fake_quant_on, running_min, running_max, scale, zero_point, averaging_const, quant_min, quant_max, ch_axis, per_row_fake_quant, symmetric_quant);
}

// aten::_fused_moving_avg_obs_fq_helper(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor(a!) running_min, Tensor(b!) running_max, Tensor(c!) scale, Tensor(d!) zero_point, float averaging_const, int quant_min, int quant_max, int ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False) -> (Tensor output, Tensor mask)
::std::tuple<at::Tensor,at::Tensor> _fused_moving_avg_obs_fq_helper::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & observer_on, const at::Tensor & fake_quant_on, at::Tensor & running_min, at::Tensor & running_max, at::Tensor & scale, at::Tensor & zero_point, double averaging_const, int64_t quant_min, int64_t quant_max, int64_t ch_axis, bool per_row_fake_quant, bool symmetric_quant) {
    static auto op = create__fused_moving_avg_obs_fq_helper_typed_handle();
    return op.redispatch(dispatchKeySet, self, observer_on, fake_quant_on, running_min, running_max, scale, zero_point, averaging_const, quant_min, quant_max, ch_axis, per_row_fake_quant, symmetric_quant);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_to_copy, name, "aten::_to_copy")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_to_copy, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_to_copy, schema_str, "_to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, MemoryFormat? memory_format=None) -> Tensor")

// aten::_to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, MemoryFormat? memory_format=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_to_copy::schema> create__to_copy_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_to_copy::name, _to_copy::overload_name)
      .typed<_to_copy::schema>();
}

// aten::_to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, MemoryFormat? memory_format=None) -> Tensor
at::Tensor _to_copy::call(const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, bool non_blocking, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create__to_copy_typed_handle();
    return op.call(self, dtype, layout, device, pin_memory, non_blocking, memory_format);
}

// aten::_to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, MemoryFormat? memory_format=None) -> Tensor
at::Tensor _to_copy::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, bool non_blocking, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create__to_copy_typed_handle();
    return op.redispatch(dispatchKeySet, self, dtype, layout, device, pin_memory, non_blocking, memory_format);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_thnn_differentiable_lstm_cell_backward, name, "aten::_thnn_differentiable_lstm_cell_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_thnn_differentiable_lstm_cell_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_thnn_differentiable_lstm_cell_backward, schema_str, "_thnn_differentiable_lstm_cell_backward(Tensor? grad_hy, Tensor? grad_cy, Tensor input_gates, Tensor hidden_gates, Tensor? input_bias, Tensor? hidden_bias, Tensor cx, Tensor cy) -> (Tensor, Tensor, Tensor, Tensor, Tensor)")

// aten::_thnn_differentiable_lstm_cell_backward(Tensor? grad_hy, Tensor? grad_cy, Tensor input_gates, Tensor hidden_gates, Tensor? input_bias, Tensor? hidden_bias, Tensor cx, Tensor cy) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<_thnn_differentiable_lstm_cell_backward::schema> create__thnn_differentiable_lstm_cell_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_thnn_differentiable_lstm_cell_backward::name, _thnn_differentiable_lstm_cell_backward::overload_name)
      .typed<_thnn_differentiable_lstm_cell_backward::schema>();
}

// aten::_thnn_differentiable_lstm_cell_backward(Tensor? grad_hy, Tensor? grad_cy, Tensor input_gates, Tensor hidden_gates, Tensor? input_bias, Tensor? hidden_bias, Tensor cx, Tensor cy) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> _thnn_differentiable_lstm_cell_backward::call(const c10::optional<at::Tensor> & grad_hy, const c10::optional<at::Tensor> & grad_cy, const at::Tensor & input_gates, const at::Tensor & hidden_gates, const c10::optional<at::Tensor> & input_bias, const c10::optional<at::Tensor> & hidden_bias, const at::Tensor & cx, const at::Tensor & cy) {
    static auto op = create__thnn_differentiable_lstm_cell_backward_typed_handle();
    return op.call(grad_hy, grad_cy, input_gates, hidden_gates, input_bias, hidden_bias, cx, cy);
}

// aten::_thnn_differentiable_lstm_cell_backward(Tensor? grad_hy, Tensor? grad_cy, Tensor input_gates, Tensor hidden_gates, Tensor? input_bias, Tensor? hidden_bias, Tensor cx, Tensor cy) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> _thnn_differentiable_lstm_cell_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const c10::optional<at::Tensor> & grad_hy, const c10::optional<at::Tensor> & grad_cy, const at::Tensor & input_gates, const at::Tensor & hidden_gates, const c10::optional<at::Tensor> & input_bias, const c10::optional<at::Tensor> & hidden_bias, const at::Tensor & cx, const at::Tensor & cy) {
    static auto op = create__thnn_differentiable_lstm_cell_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_hy, grad_cy, input_gates, hidden_gates, input_bias, hidden_bias, cx, cy);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_thnn_differentiable_gru_cell_backward, name, "aten::_thnn_differentiable_gru_cell_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_thnn_differentiable_gru_cell_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_thnn_differentiable_gru_cell_backward, schema_str, "_thnn_differentiable_gru_cell_backward(Tensor grad_hy, Tensor input_gates, Tensor hidden_gates, Tensor hx, Tensor? input_bias, Tensor? hidden_bias) -> (Tensor, Tensor, Tensor, Tensor, Tensor)")

// aten::_thnn_differentiable_gru_cell_backward(Tensor grad_hy, Tensor input_gates, Tensor hidden_gates, Tensor hx, Tensor? input_bias, Tensor? hidden_bias) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<_thnn_differentiable_gru_cell_backward::schema> create__thnn_differentiable_gru_cell_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_thnn_differentiable_gru_cell_backward::name, _thnn_differentiable_gru_cell_backward::overload_name)
      .typed<_thnn_differentiable_gru_cell_backward::schema>();
}

// aten::_thnn_differentiable_gru_cell_backward(Tensor grad_hy, Tensor input_gates, Tensor hidden_gates, Tensor hx, Tensor? input_bias, Tensor? hidden_bias) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> _thnn_differentiable_gru_cell_backward::call(const at::Tensor & grad_hy, const at::Tensor & input_gates, const at::Tensor & hidden_gates, const at::Tensor & hx, const c10::optional<at::Tensor> & input_bias, const c10::optional<at::Tensor> & hidden_bias) {
    static auto op = create__thnn_differentiable_gru_cell_backward_typed_handle();
    return op.call(grad_hy, input_gates, hidden_gates, hx, input_bias, hidden_bias);
}

// aten::_thnn_differentiable_gru_cell_backward(Tensor grad_hy, Tensor input_gates, Tensor hidden_gates, Tensor hx, Tensor? input_bias, Tensor? hidden_bias) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> _thnn_differentiable_gru_cell_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_hy, const at::Tensor & input_gates, const at::Tensor & hidden_gates, const at::Tensor & hx, const c10::optional<at::Tensor> & input_bias, const c10::optional<at::Tensor> & hidden_bias) {
    static auto op = create__thnn_differentiable_gru_cell_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_hy, input_gates, hidden_gates, hx, input_bias, hidden_bias);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rnn_tanh_input, name, "aten::rnn_tanh")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rnn_tanh_input, overload_name, "input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rnn_tanh_input, schema_str, "rnn_tanh.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)")

// aten::rnn_tanh.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<rnn_tanh_input::schema> create_rnn_tanh_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(rnn_tanh_input::name, rnn_tanh_input::overload_name)
      .typed<rnn_tanh_input::schema>();
}

// aten::rnn_tanh.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> rnn_tanh_input::call(const at::Tensor & input, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
    static auto op = create_rnn_tanh_input_typed_handle();
    return op.call(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}

// aten::rnn_tanh.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> rnn_tanh_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
    static auto op = create_rnn_tanh_input_typed_handle();
    return op.redispatch(dispatchKeySet, input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rnn_tanh_cell, name, "aten::rnn_tanh_cell")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rnn_tanh_cell, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rnn_tanh_cell, schema_str, "rnn_tanh_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor")

// aten::rnn_tanh_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<rnn_tanh_cell::schema> create_rnn_tanh_cell_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(rnn_tanh_cell::name, rnn_tanh_cell::overload_name)
      .typed<rnn_tanh_cell::schema>();
}

// aten::rnn_tanh_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor
at::Tensor rnn_tanh_cell::call(const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const c10::optional<at::Tensor> & b_ih, const c10::optional<at::Tensor> & b_hh) {
    static auto op = create_rnn_tanh_cell_typed_handle();
    return op.call(input, hx, w_ih, w_hh, b_ih, b_hh);
}

// aten::rnn_tanh_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor
at::Tensor rnn_tanh_cell::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const c10::optional<at::Tensor> & b_ih, const c10::optional<at::Tensor> & b_hh) {
    static auto op = create_rnn_tanh_cell_typed_handle();
    return op.redispatch(dispatchKeySet, input, hx, w_ih, w_hh, b_ih, b_hh);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantized_gru_cell, name, "aten::quantized_gru_cell")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantized_gru_cell, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantized_gru_cell, schema_str, "quantized_gru_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor")

// aten::quantized_gru_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<quantized_gru_cell::schema> create_quantized_gru_cell_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(quantized_gru_cell::name, quantized_gru_cell::overload_name)
      .typed<quantized_gru_cell::schema>();
}

// aten::quantized_gru_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor
at::Tensor quantized_gru_cell::call(const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const at::Tensor & b_ih, const at::Tensor & b_hh, const at::Tensor & packed_ih, const at::Tensor & packed_hh, const at::Tensor & col_offsets_ih, const at::Tensor & col_offsets_hh, const at::Scalar & scale_ih, const at::Scalar & scale_hh, const at::Scalar & zero_point_ih, const at::Scalar & zero_point_hh) {
    static auto op = create_quantized_gru_cell_typed_handle();
    return op.call(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
}

// aten::quantized_gru_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor
at::Tensor quantized_gru_cell::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const at::Tensor & b_ih, const at::Tensor & b_hh, const at::Tensor & packed_ih, const at::Tensor & packed_hh, const at::Tensor & col_offsets_ih, const at::Tensor & col_offsets_hh, const at::Scalar & scale_ih, const at::Scalar & scale_hh, const at::Scalar & zero_point_ih, const at::Scalar & zero_point_hh) {
    static auto op = create_quantized_gru_cell_typed_handle();
    return op.redispatch(dispatchKeySet, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_pack_padded_sequence_backward, name, "aten::_pack_padded_sequence_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_pack_padded_sequence_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_pack_padded_sequence_backward, schema_str, "_pack_padded_sequence_backward(Tensor grad, int[] input_size, Tensor batch_sizes, bool batch_first) -> Tensor")

// aten::_pack_padded_sequence_backward(Tensor grad, int[] input_size, Tensor batch_sizes, bool batch_first) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_pack_padded_sequence_backward::schema> create__pack_padded_sequence_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_pack_padded_sequence_backward::name, _pack_padded_sequence_backward::overload_name)
      .typed<_pack_padded_sequence_backward::schema>();
}

// aten::_pack_padded_sequence_backward(Tensor grad, int[] input_size, Tensor batch_sizes, bool batch_first) -> Tensor
at::Tensor _pack_padded_sequence_backward::call(const at::Tensor & grad, at::IntArrayRef input_size, const at::Tensor & batch_sizes, bool batch_first) {
    static auto op = create__pack_padded_sequence_backward_typed_handle();
    return op.call(grad, input_size, batch_sizes, batch_first);
}

// aten::_pack_padded_sequence_backward(Tensor grad, int[] input_size, Tensor batch_sizes, bool batch_first) -> Tensor
at::Tensor _pack_padded_sequence_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, at::IntArrayRef input_size, const at::Tensor & batch_sizes, bool batch_first) {
    static auto op = create__pack_padded_sequence_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad, input_size, batch_sizes, batch_first);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(set__source_Storage, name, "aten::set_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(set__source_Storage, overload_name, "source_Storage")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(set__source_Storage, schema_str, "set_.source_Storage(Tensor(a!) self, Storage source) -> Tensor(a!)")

// aten::set_.source_Storage(Tensor(a!) self, Storage source) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<set__source_Storage::schema> create_set__source_Storage_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(set__source_Storage::name, set__source_Storage::overload_name)
      .typed<set__source_Storage::schema>();
}

// aten::set_.source_Storage(Tensor(a!) self, Storage source) -> Tensor(a!)
at::Tensor & set__source_Storage::call(at::Tensor & self, at::Storage source) {
    static auto op = create_set__source_Storage_typed_handle();
    return op.call(self, source);
}

// aten::set_.source_Storage(Tensor(a!) self, Storage source) -> Tensor(a!)
at::Tensor & set__source_Storage::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, at::Storage source) {
    static auto op = create_set__source_Storage_typed_handle();
    return op.redispatch(dispatchKeySet, self, source);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(set_, name, "aten::set_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(set_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(set_, schema_str, "set_(Tensor(a!) self) -> Tensor(a!)")

// aten::set_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<set_::schema> create_set__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(set_::name, set_::overload_name)
      .typed<set_::schema>();
}

// aten::set_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & set_::call(at::Tensor & self) {
    static auto op = create_set__typed_handle();
    return op.call(self);
}

// aten::set_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & set_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_set__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(masked_fill_Scalar, name, "aten::masked_fill")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(masked_fill_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(masked_fill_Scalar, schema_str, "masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor")

// aten::masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<masked_fill_Scalar::schema> create_masked_fill_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(masked_fill_Scalar::name, masked_fill_Scalar::overload_name)
      .typed<masked_fill_Scalar::schema>();
}

// aten::masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor
at::Tensor masked_fill_Scalar::call(const at::Tensor & self, const at::Tensor & mask, const at::Scalar & value) {
    static auto op = create_masked_fill_Scalar_typed_handle();
    return op.call(self, mask, value);
}

// aten::masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor
at::Tensor masked_fill_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mask, const at::Scalar & value) {
    static auto op = create_masked_fill_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, mask, value);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_add__alpha, name, "aten::index_add_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_add__alpha, overload_name, "alpha")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_add__alpha, schema_str, "index_add_.alpha(Tensor(a!) self, int dim, Tensor index, Tensor source, *, Scalar alpha) -> Tensor(a!)")

// aten::index_add_.alpha(Tensor(a!) self, int dim, Tensor index, Tensor source, *, Scalar alpha) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<index_add__alpha::schema> create_index_add__alpha_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(index_add__alpha::name, index_add__alpha::overload_name)
      .typed<index_add__alpha::schema>();
}

// aten::index_add_.alpha(Tensor(a!) self, int dim, Tensor index, Tensor source, *, Scalar alpha) -> Tensor(a!)
at::Tensor & index_add__alpha::call(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha) {
    static auto op = create_index_add__alpha_typed_handle();
    return op.call(self, dim, index, source, alpha);
}

// aten::index_add_.alpha(Tensor(a!) self, int dim, Tensor index, Tensor source, *, Scalar alpha) -> Tensor(a!)
at::Tensor & index_add__alpha::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha) {
    static auto op = create_index_add__alpha_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, source, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_fill_int_Scalar, name, "aten::index_fill")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_fill_int_Scalar, overload_name, "int_Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_fill_int_Scalar, schema_str, "index_fill.int_Scalar(Tensor self, int dim, Tensor index, Scalar value) -> Tensor")

// aten::index_fill.int_Scalar(Tensor self, int dim, Tensor index, Scalar value) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<index_fill_int_Scalar::schema> create_index_fill_int_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(index_fill_int_Scalar::name, index_fill_int_Scalar::overload_name)
      .typed<index_fill_int_Scalar::schema>();
}

// aten::index_fill.int_Scalar(Tensor self, int dim, Tensor index, Scalar value) -> Tensor
at::Tensor index_fill_int_Scalar::call(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) {
    static auto op = create_index_fill_int_Scalar_typed_handle();
    return op.call(self, dim, index, value);
}

// aten::index_fill.int_Scalar(Tensor self, int dim, Tensor index, Scalar value) -> Tensor
at::Tensor index_fill_int_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) {
    static auto op = create_index_fill_int_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, value);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_fill_Dimname_Scalar, name, "aten::index_fill")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_fill_Dimname_Scalar, overload_name, "Dimname_Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_fill_Dimname_Scalar, schema_str, "index_fill.Dimname_Scalar(Tensor self, Dimname dim, Tensor index, Scalar value) -> Tensor")

// aten::index_fill.Dimname_Scalar(Tensor self, Dimname dim, Tensor index, Scalar value) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<index_fill_Dimname_Scalar::schema> create_index_fill_Dimname_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(index_fill_Dimname_Scalar::name, index_fill_Dimname_Scalar::overload_name)
      .typed<index_fill_Dimname_Scalar::schema>();
}

// aten::index_fill.Dimname_Scalar(Tensor self, Dimname dim, Tensor index, Scalar value) -> Tensor
at::Tensor index_fill_Dimname_Scalar::call(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Scalar & value) {
    static auto op = create_index_fill_Dimname_Scalar_typed_handle();
    return op.call(self, dim, index, value);
}

// aten::index_fill.Dimname_Scalar(Tensor self, Dimname dim, Tensor index, Scalar value) -> Tensor
at::Tensor index_fill_Dimname_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Scalar & value) {
    static auto op = create_index_fill_Dimname_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, value);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_src_out, name, "aten::scatter")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_src_out, overload_name, "src_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_src_out, schema_str, "scatter.src_out(Tensor self, int dim, Tensor index, Tensor src, *, Tensor(a!) out) -> Tensor(a!)")

// aten::scatter.src_out(Tensor self, int dim, Tensor index, Tensor src, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<scatter_src_out::schema> create_scatter_src_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(scatter_src_out::name, scatter_src_out::overload_name)
      .typed<scatter_src_out::schema>();
}

// aten::scatter.src_out(Tensor self, int dim, Tensor index, Tensor src, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & scatter_src_out::call(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, at::Tensor & out) {
    static auto op = create_scatter_src_out_typed_handle();
    return op.call(self, dim, index, src, out);
}

// aten::scatter.src_out(Tensor self, int dim, Tensor index, Tensor src, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & scatter_src_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, at::Tensor & out) {
    static auto op = create_scatter_src_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, src, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_value_reduce, name, "aten::scatter")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_value_reduce, overload_name, "value_reduce")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_value_reduce, schema_str, "scatter.value_reduce(Tensor self, int dim, Tensor index, Scalar value, *, str reduce) -> Tensor")

// aten::scatter.value_reduce(Tensor self, int dim, Tensor index, Scalar value, *, str reduce) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<scatter_value_reduce::schema> create_scatter_value_reduce_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(scatter_value_reduce::name, scatter_value_reduce::overload_name)
      .typed<scatter_value_reduce::schema>();
}

// aten::scatter.value_reduce(Tensor self, int dim, Tensor index, Scalar value, *, str reduce) -> Tensor
at::Tensor scatter_value_reduce::call(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, c10::string_view reduce) {
    static auto op = create_scatter_value_reduce_typed_handle();
    return op.call(self, dim, index, value, reduce);
}

// aten::scatter.value_reduce(Tensor self, int dim, Tensor index, Scalar value, *, str reduce) -> Tensor
at::Tensor scatter_value_reduce::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, c10::string_view reduce) {
    static auto op = create_scatter_value_reduce_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, value, reduce);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__iand___Tensor, name, "aten::__iand__")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__iand___Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__iand___Tensor, schema_str, "__iand__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::__iand__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<__iand___Tensor::schema> create___iand___Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(__iand___Tensor::name, __iand___Tensor::overload_name)
      .typed<__iand___Tensor::schema>();
}

// aten::__iand__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & __iand___Tensor::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create___iand___Tensor_typed_handle();
    return op.call(self, other);
}

// aten::__iand__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & __iand___Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create___iand___Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_or_Scalar, name, "aten::bitwise_or")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_or_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_or_Scalar, schema_str, "bitwise_or.Scalar(Tensor self, Scalar other) -> Tensor")

// aten::bitwise_or.Scalar(Tensor self, Scalar other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<bitwise_or_Scalar::schema> create_bitwise_or_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bitwise_or_Scalar::name, bitwise_or_Scalar::overload_name)
      .typed<bitwise_or_Scalar::schema>();
}

// aten::bitwise_or.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor bitwise_or_Scalar::call(const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_bitwise_or_Scalar_typed_handle();
    return op.call(self, other);
}

// aten::bitwise_or.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor bitwise_or_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_bitwise_or_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_xor_Scalar, name, "aten::bitwise_xor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_xor_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_xor_Scalar, schema_str, "bitwise_xor.Scalar(Tensor self, Scalar other) -> Tensor")

// aten::bitwise_xor.Scalar(Tensor self, Scalar other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<bitwise_xor_Scalar::schema> create_bitwise_xor_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bitwise_xor_Scalar::name, bitwise_xor_Scalar::overload_name)
      .typed<bitwise_xor_Scalar::schema>();
}

// aten::bitwise_xor.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor bitwise_xor_Scalar::call(const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_bitwise_xor_Scalar_typed_handle();
    return op.call(self, other);
}

// aten::bitwise_xor.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor bitwise_xor_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_bitwise_xor_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__ilshift___Tensor, name, "aten::__ilshift__")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__ilshift___Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__ilshift___Tensor, schema_str, "__ilshift__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::__ilshift__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<__ilshift___Tensor::schema> create___ilshift___Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(__ilshift___Tensor::name, __ilshift___Tensor::overload_name)
      .typed<__ilshift___Tensor::schema>();
}

// aten::__ilshift__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & __ilshift___Tensor::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create___ilshift___Tensor_typed_handle();
    return op.call(self, other);
}

// aten::__ilshift__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & __ilshift___Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create___ilshift___Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_left_shift_Tensor, name, "aten::bitwise_left_shift")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_left_shift_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_left_shift_Tensor, schema_str, "bitwise_left_shift.Tensor(Tensor self, Tensor other) -> Tensor")

// aten::bitwise_left_shift.Tensor(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<bitwise_left_shift_Tensor::schema> create_bitwise_left_shift_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bitwise_left_shift_Tensor::name, bitwise_left_shift_Tensor::overload_name)
      .typed<bitwise_left_shift_Tensor::schema>();
}

// aten::bitwise_left_shift.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor bitwise_left_shift_Tensor::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_bitwise_left_shift_Tensor_typed_handle();
    return op.call(self, other);
}

// aten::bitwise_left_shift.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor bitwise_left_shift_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_bitwise_left_shift_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_left_shift_Scalar_Tensor, name, "aten::bitwise_left_shift")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_left_shift_Scalar_Tensor, overload_name, "Scalar_Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_left_shift_Scalar_Tensor, schema_str, "bitwise_left_shift.Scalar_Tensor(Scalar self, Tensor other) -> Tensor")

// aten::bitwise_left_shift.Scalar_Tensor(Scalar self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<bitwise_left_shift_Scalar_Tensor::schema> create_bitwise_left_shift_Scalar_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bitwise_left_shift_Scalar_Tensor::name, bitwise_left_shift_Scalar_Tensor::overload_name)
      .typed<bitwise_left_shift_Scalar_Tensor::schema>();
}

// aten::bitwise_left_shift.Scalar_Tensor(Scalar self, Tensor other) -> Tensor
at::Tensor bitwise_left_shift_Scalar_Tensor::call(const at::Scalar & self, const at::Tensor & other) {
    static auto op = create_bitwise_left_shift_Scalar_Tensor_typed_handle();
    return op.call(self, other);
}

// aten::bitwise_left_shift.Scalar_Tensor(Scalar self, Tensor other) -> Tensor
at::Tensor bitwise_left_shift_Scalar_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Scalar & self, const at::Tensor & other) {
    static auto op = create_bitwise_left_shift_Scalar_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__rshift___Tensor, name, "aten::__rshift__")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__rshift___Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__rshift___Tensor, schema_str, "__rshift__.Tensor(Tensor self, Tensor other) -> Tensor")

// aten::__rshift__.Tensor(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<__rshift___Tensor::schema> create___rshift___Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(__rshift___Tensor::name, __rshift___Tensor::overload_name)
      .typed<__rshift___Tensor::schema>();
}

// aten::__rshift__.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor __rshift___Tensor::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create___rshift___Tensor_typed_handle();
    return op.call(self, other);
}

// aten::__rshift__.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor __rshift___Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create___rshift___Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_right_shift_Tensor, name, "aten::bitwise_right_shift")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_right_shift_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_right_shift_Tensor, schema_str, "bitwise_right_shift.Tensor(Tensor self, Tensor other) -> Tensor")

// aten::bitwise_right_shift.Tensor(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<bitwise_right_shift_Tensor::schema> create_bitwise_right_shift_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bitwise_right_shift_Tensor::name, bitwise_right_shift_Tensor::overload_name)
      .typed<bitwise_right_shift_Tensor::schema>();
}

// aten::bitwise_right_shift.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor bitwise_right_shift_Tensor::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_bitwise_right_shift_Tensor_typed_handle();
    return op.call(self, other);
}

// aten::bitwise_right_shift.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor bitwise_right_shift_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_bitwise_right_shift_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_right_shift_Tensor_out, name, "aten::bitwise_right_shift")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_right_shift_Tensor_out, overload_name, "Tensor_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_right_shift_Tensor_out, schema_str, "bitwise_right_shift.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::bitwise_right_shift.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<bitwise_right_shift_Tensor_out::schema> create_bitwise_right_shift_Tensor_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bitwise_right_shift_Tensor_out::name, bitwise_right_shift_Tensor_out::overload_name)
      .typed<bitwise_right_shift_Tensor_out::schema>();
}

// aten::bitwise_right_shift.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & bitwise_right_shift_Tensor_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_bitwise_right_shift_Tensor_out_typed_handle();
    return op.call(self, other, out);
}

// aten::bitwise_right_shift.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & bitwise_right_shift_Tensor_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_bitwise_right_shift_Tensor_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_right_shift_Tensor_Scalar, name, "aten::bitwise_right_shift")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_right_shift_Tensor_Scalar, overload_name, "Tensor_Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_right_shift_Tensor_Scalar, schema_str, "bitwise_right_shift.Tensor_Scalar(Tensor self, Scalar other) -> Tensor")

// aten::bitwise_right_shift.Tensor_Scalar(Tensor self, Scalar other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<bitwise_right_shift_Tensor_Scalar::schema> create_bitwise_right_shift_Tensor_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bitwise_right_shift_Tensor_Scalar::name, bitwise_right_shift_Tensor_Scalar::overload_name)
      .typed<bitwise_right_shift_Tensor_Scalar::schema>();
}

// aten::bitwise_right_shift.Tensor_Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor bitwise_right_shift_Tensor_Scalar::call(const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_bitwise_right_shift_Tensor_Scalar_typed_handle();
    return op.call(self, other);
}

// aten::bitwise_right_shift.Tensor_Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor bitwise_right_shift_Tensor_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_bitwise_right_shift_Tensor_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(digamma_, name, "aten::digamma_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(digamma_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(digamma_, schema_str, "digamma_(Tensor(a!) self) -> Tensor(a!)")

// aten::digamma_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<digamma_::schema> create_digamma__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(digamma_::name, digamma_::overload_name)
      .typed<digamma_::schema>();
}

// aten::digamma_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & digamma_::call(at::Tensor & self) {
    static auto op = create_digamma__typed_handle();
    return op.call(self);
}

// aten::digamma_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & digamma_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_digamma__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(exponential_, name, "aten::exponential_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(exponential_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(exponential_, schema_str, "exponential_(Tensor(a!) self, float lambd=1, *, Generator? generator=None) -> Tensor(a!)")

// aten::exponential_(Tensor(a!) self, float lambd=1, *, Generator? generator=None) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<exponential_::schema> create_exponential__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(exponential_::name, exponential_::overload_name)
      .typed<exponential_::schema>();
}

// aten::exponential_(Tensor(a!) self, float lambd=1, *, Generator? generator=None) -> Tensor(a!)
at::Tensor & exponential_::call(at::Tensor & self, double lambd, c10::optional<at::Generator> generator) {
    static auto op = create_exponential__typed_handle();
    return op.call(self, lambd, generator);
}

// aten::exponential_(Tensor(a!) self, float lambd=1, *, Generator? generator=None) -> Tensor(a!)
at::Tensor & exponential_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, double lambd, c10::optional<at::Generator> generator) {
    static auto op = create_exponential__typed_handle();
    return op.redispatch(dispatchKeySet, self, lambd, generator);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(geometric_, name, "aten::geometric_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(geometric_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(geometric_, schema_str, "geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)")

// aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<geometric_::schema> create_geometric__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(geometric_::name, geometric_::overload_name)
      .typed<geometric_::schema>();
}

// aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
at::Tensor & geometric_::call(at::Tensor & self, double p, c10::optional<at::Generator> generator) {
    static auto op = create_geometric__typed_handle();
    return op.call(self, p, generator);
}

// aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
at::Tensor & geometric_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, double p, c10::optional<at::Generator> generator) {
    static auto op = create_geometric__typed_handle();
    return op.redispatch(dispatchKeySet, self, p, generator);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cross_out, name, "aten::cross")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cross_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cross_out, schema_str, "cross.out(Tensor self, Tensor other, int? dim=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::cross.out(Tensor self, Tensor other, int? dim=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<cross_out::schema> create_cross_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cross_out::name, cross_out::overload_name)
      .typed<cross_out::schema>();
}

// aten::cross.out(Tensor self, Tensor other, int? dim=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & cross_out::call(const at::Tensor & self, const at::Tensor & other, c10::optional<int64_t> dim, at::Tensor & out) {
    static auto op = create_cross_out_typed_handle();
    return op.call(self, other, dim, out);
}

// aten::cross.out(Tensor self, Tensor other, int? dim=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & cross_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, c10::optional<int64_t> dim, at::Tensor & out) {
    static auto op = create_cross_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, dim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tril_out, name, "aten::tril")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tril_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tril_out, schema_str, "tril.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)")

// aten::tril.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<tril_out::schema> create_tril_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(tril_out::name, tril_out::overload_name)
      .typed<tril_out::schema>();
}

// aten::tril.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & tril_out::call(const at::Tensor & self, int64_t diagonal, at::Tensor & out) {
    static auto op = create_tril_out_typed_handle();
    return op.call(self, diagonal, out);
}

// aten::tril.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & tril_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t diagonal, at::Tensor & out) {
    static auto op = create_tril_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, diagonal, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(trace_backward, name, "aten::trace_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(trace_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(trace_backward, schema_str, "trace_backward(Tensor grad, int[] sizes) -> Tensor")

// aten::trace_backward(Tensor grad, int[] sizes) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<trace_backward::schema> create_trace_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(trace_backward::name, trace_backward::overload_name)
      .typed<trace_backward::schema>();
}

// aten::trace_backward(Tensor grad, int[] sizes) -> Tensor
at::Tensor trace_backward::call(const at::Tensor & grad, at::IntArrayRef sizes) {
    static auto op = create_trace_backward_typed_handle();
    return op.call(grad, sizes);
}

// aten::trace_backward(Tensor grad, int[] sizes) -> Tensor
at::Tensor trace_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, at::IntArrayRef sizes) {
    static auto op = create_trace_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad, sizes);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ne_Scalar, name, "aten::ne")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ne_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ne_Scalar, schema_str, "ne.Scalar(Tensor self, Scalar other) -> Tensor")

// aten::ne.Scalar(Tensor self, Scalar other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<ne_Scalar::schema> create_ne_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(ne_Scalar::name, ne_Scalar::overload_name)
      .typed<ne_Scalar::schema>();
}

// aten::ne.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor ne_Scalar::call(const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_ne_Scalar_typed_handle();
    return op.call(self, other);
}

// aten::ne.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor ne_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_ne_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ne_Tensor_out, name, "aten::ne")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ne_Tensor_out, overload_name, "Tensor_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ne_Tensor_out, schema_str, "ne.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::ne.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<ne_Tensor_out::schema> create_ne_Tensor_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(ne_Tensor_out::name, ne_Tensor_out::overload_name)
      .typed<ne_Tensor_out::schema>();
}

// aten::ne.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & ne_Tensor_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_ne_Tensor_out_typed_handle();
    return op.call(self, other, out);
}

// aten::ne.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & ne_Tensor_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_ne_Tensor_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(not_equal_Tensor_out, name, "aten::not_equal")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(not_equal_Tensor_out, overload_name, "Tensor_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(not_equal_Tensor_out, schema_str, "not_equal.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::not_equal.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<not_equal_Tensor_out::schema> create_not_equal_Tensor_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(not_equal_Tensor_out::name, not_equal_Tensor_out::overload_name)
      .typed<not_equal_Tensor_out::schema>();
}

// aten::not_equal.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & not_equal_Tensor_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_not_equal_Tensor_out_typed_handle();
    return op.call(self, other, out);
}

// aten::not_equal.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & not_equal_Tensor_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_not_equal_Tensor_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(not_equal__Tensor, name, "aten::not_equal_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(not_equal__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(not_equal__Tensor, schema_str, "not_equal_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::not_equal_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<not_equal__Tensor::schema> create_not_equal__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(not_equal__Tensor::name, not_equal__Tensor::overload_name)
      .typed<not_equal__Tensor::schema>();
}

// aten::not_equal_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & not_equal__Tensor::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create_not_equal__Tensor_typed_handle();
    return op.call(self, other);
}

// aten::not_equal_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & not_equal__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create_not_equal__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ge_Scalar_out, name, "aten::ge")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ge_Scalar_out, overload_name, "Scalar_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ge_Scalar_out, schema_str, "ge.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::ge.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<ge_Scalar_out::schema> create_ge_Scalar_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(ge_Scalar_out::name, ge_Scalar_out::overload_name)
      .typed<ge_Scalar_out::schema>();
}

// aten::ge.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & ge_Scalar_out::call(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_ge_Scalar_out_typed_handle();
    return op.call(self, other, out);
}

// aten::ge.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & ge_Scalar_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_ge_Scalar_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ge_Tensor_out, name, "aten::ge")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ge_Tensor_out, overload_name, "Tensor_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ge_Tensor_out, schema_str, "ge.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::ge.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<ge_Tensor_out::schema> create_ge_Tensor_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(ge_Tensor_out::name, ge_Tensor_out::overload_name)
      .typed<ge_Tensor_out::schema>();
}

// aten::ge.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & ge_Tensor_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_ge_Tensor_out_typed_handle();
    return op.call(self, other, out);
}

// aten::ge.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & ge_Tensor_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_ge_Tensor_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ge_Tensor, name, "aten::ge")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ge_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ge_Tensor, schema_str, "ge.Tensor(Tensor self, Tensor other) -> Tensor")

// aten::ge.Tensor(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<ge_Tensor::schema> create_ge_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(ge_Tensor::name, ge_Tensor::overload_name)
      .typed<ge_Tensor::schema>();
}

// aten::ge.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor ge_Tensor::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_ge_Tensor_typed_handle();
    return op.call(self, other);
}

// aten::ge.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor ge_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_ge_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ge__Scalar, name, "aten::ge_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ge__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ge__Scalar, schema_str, "ge_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)")

// aten::ge_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<ge__Scalar::schema> create_ge__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(ge__Scalar::name, ge__Scalar::overload_name)
      .typed<ge__Scalar::schema>();
}

// aten::ge_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & ge__Scalar::call(at::Tensor & self, const at::Scalar & other) {
    static auto op = create_ge__Scalar_typed_handle();
    return op.call(self, other);
}

// aten::ge_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & ge__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
    static auto op = create_ge__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(greater_equal_Scalar, name, "aten::greater_equal")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(greater_equal_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(greater_equal_Scalar, schema_str, "greater_equal.Scalar(Tensor self, Scalar other) -> Tensor")

// aten::greater_equal.Scalar(Tensor self, Scalar other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<greater_equal_Scalar::schema> create_greater_equal_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(greater_equal_Scalar::name, greater_equal_Scalar::overload_name)
      .typed<greater_equal_Scalar::schema>();
}

// aten::greater_equal.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor greater_equal_Scalar::call(const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_greater_equal_Scalar_typed_handle();
    return op.call(self, other);
}

// aten::greater_equal.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor greater_equal_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_greater_equal_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(greater_equal_Tensor, name, "aten::greater_equal")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(greater_equal_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(greater_equal_Tensor, schema_str, "greater_equal.Tensor(Tensor self, Tensor other) -> Tensor")

// aten::greater_equal.Tensor(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<greater_equal_Tensor::schema> create_greater_equal_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(greater_equal_Tensor::name, greater_equal_Tensor::overload_name)
      .typed<greater_equal_Tensor::schema>();
}

// aten::greater_equal.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor greater_equal_Tensor::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_greater_equal_Tensor_typed_handle();
    return op.call(self, other);
}

// aten::greater_equal.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor greater_equal_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_greater_equal_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(less_equal__Tensor, name, "aten::less_equal_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(less_equal__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(less_equal__Tensor, schema_str, "less_equal_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::less_equal_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<less_equal__Tensor::schema> create_less_equal__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(less_equal__Tensor::name, less_equal__Tensor::overload_name)
      .typed<less_equal__Tensor::schema>();
}

// aten::less_equal_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & less_equal__Tensor::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create_less_equal__Tensor_typed_handle();
    return op.call(self, other);
}

// aten::less_equal_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & less_equal__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create_less_equal__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(less_Scalar_out, name, "aten::less")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(less_Scalar_out, overload_name, "Scalar_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(less_Scalar_out, schema_str, "less.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::less.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<less_Scalar_out::schema> create_less_Scalar_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(less_Scalar_out::name, less_Scalar_out::overload_name)
      .typed<less_Scalar_out::schema>();
}

// aten::less.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & less_Scalar_out::call(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_less_Scalar_out_typed_handle();
    return op.call(self, other, out);
}

// aten::less.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & less_Scalar_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_less_Scalar_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(less_Scalar, name, "aten::less")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(less_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(less_Scalar, schema_str, "less.Scalar(Tensor self, Scalar other) -> Tensor")

// aten::less.Scalar(Tensor self, Scalar other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<less_Scalar::schema> create_less_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(less_Scalar::name, less_Scalar::overload_name)
      .typed<less_Scalar::schema>();
}

// aten::less.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor less_Scalar::call(const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_less_Scalar_typed_handle();
    return op.call(self, other);
}

// aten::less.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor less_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_less_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(less__Tensor, name, "aten::less_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(less__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(less__Tensor, schema_str, "less_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::less_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<less__Tensor::schema> create_less__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(less__Tensor::name, less__Tensor::overload_name)
      .typed<less__Tensor::schema>();
}

// aten::less_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & less__Tensor::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create_less__Tensor_typed_handle();
    return op.call(self, other);
}

// aten::less_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & less__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create_less__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(take_out, name, "aten::take")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(take_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(take_out, schema_str, "take.out(Tensor self, Tensor index, *, Tensor(a!) out) -> Tensor(a!)")

// aten::take.out(Tensor self, Tensor index, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<take_out::schema> create_take_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(take_out::name, take_out::overload_name)
      .typed<take_out::schema>();
}

// aten::take.out(Tensor self, Tensor index, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & take_out::call(const at::Tensor & self, const at::Tensor & index, at::Tensor & out) {
    static auto op = create_take_out_typed_handle();
    return op.call(self, index, out);
}

// aten::take.out(Tensor self, Tensor index, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & take_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & index, at::Tensor & out) {
    static auto op = create_take_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, index, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(take_along_dim, name, "aten::take_along_dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(take_along_dim, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(take_along_dim, schema_str, "take_along_dim(Tensor self, Tensor indices, int? dim=None) -> Tensor")

// aten::take_along_dim(Tensor self, Tensor indices, int? dim=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<take_along_dim::schema> create_take_along_dim_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(take_along_dim::name, take_along_dim::overload_name)
      .typed<take_along_dim::schema>();
}

// aten::take_along_dim(Tensor self, Tensor indices, int? dim=None) -> Tensor
at::Tensor take_along_dim::call(const at::Tensor & self, const at::Tensor & indices, c10::optional<int64_t> dim) {
    static auto op = create_take_along_dim_typed_handle();
    return op.call(self, indices, dim);
}

// aten::take_along_dim(Tensor self, Tensor indices, int? dim=None) -> Tensor
at::Tensor take_along_dim::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & indices, c10::optional<int64_t> dim) {
    static auto op = create_take_along_dim_typed_handle();
    return op.redispatch(dispatchKeySet, self, indices, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_select, name, "aten::index_select")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_select, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_select, schema_str, "index_select(Tensor self, int dim, Tensor index) -> Tensor")

// aten::index_select(Tensor self, int dim, Tensor index) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<index_select::schema> create_index_select_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(index_select::name, index_select::overload_name)
      .typed<index_select::schema>();
}

// aten::index_select(Tensor self, int dim, Tensor index) -> Tensor
at::Tensor index_select::call(const at::Tensor & self, int64_t dim, const at::Tensor & index) {
    static auto op = create_index_select_typed_handle();
    return op.call(self, dim, index);
}

// aten::index_select(Tensor self, int dim, Tensor index) -> Tensor
at::Tensor index_select::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, const at::Tensor & index) {
    static auto op = create_index_select_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_select_dimname_out, name, "aten::index_select")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_select_dimname_out, overload_name, "dimname_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_select_dimname_out, schema_str, "index_select.dimname_out(Tensor self, Dimname dim, Tensor index, *, Tensor(a!) out) -> Tensor(a!)")

// aten::index_select.dimname_out(Tensor self, Dimname dim, Tensor index, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<index_select_dimname_out::schema> create_index_select_dimname_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(index_select_dimname_out::name, index_select_dimname_out::overload_name)
      .typed<index_select_dimname_out::schema>();
}

// aten::index_select.dimname_out(Tensor self, Dimname dim, Tensor index, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & index_select_dimname_out::call(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, at::Tensor & out) {
    static auto op = create_index_select_dimname_out_typed_handle();
    return op.call(self, dim, index, out);
}

// aten::index_select.dimname_out(Tensor self, Dimname dim, Tensor index, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & index_select_dimname_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, const at::Tensor & index, at::Tensor & out) {
    static auto op = create_index_select_dimname_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(masked_select_backward, name, "aten::masked_select_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(masked_select_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(masked_select_backward, schema_str, "masked_select_backward(Tensor grad, Tensor input, Tensor mask) -> Tensor")

// aten::masked_select_backward(Tensor grad, Tensor input, Tensor mask) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<masked_select_backward::schema> create_masked_select_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(masked_select_backward::name, masked_select_backward::overload_name)
      .typed<masked_select_backward::schema>();
}

// aten::masked_select_backward(Tensor grad, Tensor input, Tensor mask) -> Tensor
at::Tensor masked_select_backward::call(const at::Tensor & grad, const at::Tensor & input, const at::Tensor & mask) {
    static auto op = create_masked_select_backward_typed_handle();
    return op.call(grad, input, mask);
}

// aten::masked_select_backward(Tensor grad, Tensor input, Tensor mask) -> Tensor
at::Tensor masked_select_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & input, const at::Tensor & mask) {
    static auto op = create_masked_select_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad, input, mask);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nonzero, name, "aten::nonzero")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nonzero, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nonzero, schema_str, "nonzero(Tensor self) -> Tensor")

// aten::nonzero(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<nonzero::schema> create_nonzero_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nonzero::name, nonzero::overload_name)
      .typed<nonzero::schema>();
}

// aten::nonzero(Tensor self) -> Tensor
at::Tensor nonzero::call(const at::Tensor & self) {
    static auto op = create_nonzero_typed_handle();
    return op.call(self);
}

// aten::nonzero(Tensor self) -> Tensor
at::Tensor nonzero::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_nonzero_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nonzero_numpy, name, "aten::nonzero_numpy")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nonzero_numpy, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nonzero_numpy, schema_str, "nonzero_numpy(Tensor self) -> Tensor[]")

// aten::nonzero_numpy(Tensor self) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<nonzero_numpy::schema> create_nonzero_numpy_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nonzero_numpy::name, nonzero_numpy::overload_name)
      .typed<nonzero_numpy::schema>();
}

// aten::nonzero_numpy(Tensor self) -> Tensor[]
::std::vector<at::Tensor> nonzero_numpy::call(const at::Tensor & self) {
    static auto op = create_nonzero_numpy_typed_handle();
    return op.call(self);
}

// aten::nonzero_numpy(Tensor self) -> Tensor[]
::std::vector<at::Tensor> nonzero_numpy::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_nonzero_numpy_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gather_dimname_out, name, "aten::gather")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gather_dimname_out, overload_name, "dimname_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gather_dimname_out, schema_str, "gather.dimname_out(Tensor self, Dimname dim, Tensor index, *, bool sparse_grad=False, Tensor(a!) out) -> Tensor(a!)")

// aten::gather.dimname_out(Tensor self, Dimname dim, Tensor index, *, bool sparse_grad=False, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<gather_dimname_out::schema> create_gather_dimname_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(gather_dimname_out::name, gather_dimname_out::overload_name)
      .typed<gather_dimname_out::schema>();
}

// aten::gather.dimname_out(Tensor self, Dimname dim, Tensor index, *, bool sparse_grad=False, Tensor(a!) out) -> Tensor(a!)
at::Tensor & gather_dimname_out::call(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, bool sparse_grad, at::Tensor & out) {
    static auto op = create_gather_dimname_out_typed_handle();
    return op.call(self, dim, index, sparse_grad, out);
}

// aten::gather.dimname_out(Tensor self, Dimname dim, Tensor index, *, bool sparse_grad=False, Tensor(a!) out) -> Tensor(a!)
at::Tensor & gather_dimname_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, const at::Tensor & index, bool sparse_grad, at::Tensor & out) {
    static auto op = create_gather_dimname_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, sparse_grad, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gather_dimname, name, "aten::gather")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gather_dimname, overload_name, "dimname")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gather_dimname, schema_str, "gather.dimname(Tensor self, Dimname dim, Tensor index, *, bool sparse_grad=False) -> Tensor")

// aten::gather.dimname(Tensor self, Dimname dim, Tensor index, *, bool sparse_grad=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<gather_dimname::schema> create_gather_dimname_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(gather_dimname::name, gather_dimname::overload_name)
      .typed<gather_dimname::schema>();
}

// aten::gather.dimname(Tensor self, Dimname dim, Tensor index, *, bool sparse_grad=False) -> Tensor
at::Tensor gather_dimname::call(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, bool sparse_grad) {
    static auto op = create_gather_dimname_typed_handle();
    return op.call(self, dim, index, sparse_grad);
}

// aten::gather.dimname(Tensor self, Dimname dim, Tensor index, *, bool sparse_grad=False) -> Tensor
at::Tensor gather_dimname::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, const at::Tensor & index, bool sparse_grad) {
    static auto op = create_gather_dimname_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, sparse_grad);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addcmul, name, "aten::addcmul")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addcmul, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addcmul, schema_str, "addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor")

// aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<addcmul::schema> create_addcmul_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(addcmul::name, addcmul::overload_name)
      .typed<addcmul::schema>();
}

// aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor
at::Tensor addcmul::call(const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value) {
    static auto op = create_addcmul_typed_handle();
    return op.call(self, tensor1, tensor2, value);
}

// aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor
at::Tensor addcmul::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value) {
    static auto op = create_addcmul_typed_handle();
    return op.redispatch(dispatchKeySet, self, tensor1, tensor2, value);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addcmul_, name, "aten::addcmul_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addcmul_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addcmul_, schema_str, "addcmul_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)")

// aten::addcmul_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<addcmul_::schema> create_addcmul__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(addcmul_::name, addcmul_::overload_name)
      .typed<addcmul_::schema>();
}

// aten::addcmul_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)
at::Tensor & addcmul_::call(at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value) {
    static auto op = create_addcmul__typed_handle();
    return op.call(self, tensor1, tensor2, value);
}

// aten::addcmul_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)
at::Tensor & addcmul_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value) {
    static auto op = create_addcmul__typed_handle();
    return op.redispatch(dispatchKeySet, self, tensor1, tensor2, value);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lstsq, name, "aten::lstsq")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lstsq, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lstsq, schema_str, "lstsq(Tensor self, Tensor A) -> (Tensor solution, Tensor QR)")

// aten::lstsq(Tensor self, Tensor A) -> (Tensor solution, Tensor QR)
static C10_NOINLINE c10::TypedOperatorHandle<lstsq::schema> create_lstsq_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(lstsq::name, lstsq::overload_name)
      .typed<lstsq::schema>();
}

// aten::lstsq(Tensor self, Tensor A) -> (Tensor solution, Tensor QR)
::std::tuple<at::Tensor,at::Tensor> lstsq::call(const at::Tensor & self, const at::Tensor & A) {
    static auto op = create_lstsq_typed_handle();
    return op.call(self, A);
}

// aten::lstsq(Tensor self, Tensor A) -> (Tensor solution, Tensor QR)
::std::tuple<at::Tensor,at::Tensor> lstsq::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & A) {
    static auto op = create_lstsq_typed_handle();
    return op.redispatch(dispatchKeySet, self, A);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(svd_U, name, "aten::svd")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(svd_U, overload_name, "U")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(svd_U, schema_str, "svd.U(Tensor self, bool some=True, bool compute_uv=True, *, Tensor(a!) U, Tensor(b!) S, Tensor(c!) V) -> (Tensor(a!) U, Tensor(b!) S, Tensor(c!) V)")

// aten::svd.U(Tensor self, bool some=True, bool compute_uv=True, *, Tensor(a!) U, Tensor(b!) S, Tensor(c!) V) -> (Tensor(a!) U, Tensor(b!) S, Tensor(c!) V)
static C10_NOINLINE c10::TypedOperatorHandle<svd_U::schema> create_svd_U_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(svd_U::name, svd_U::overload_name)
      .typed<svd_U::schema>();
}

// aten::svd.U(Tensor self, bool some=True, bool compute_uv=True, *, Tensor(a!) U, Tensor(b!) S, Tensor(c!) V) -> (Tensor(a!) U, Tensor(b!) S, Tensor(c!) V)
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> svd_U::call(const at::Tensor & self, bool some, bool compute_uv, at::Tensor & U, at::Tensor & S, at::Tensor & V) {
    static auto op = create_svd_U_typed_handle();
    return op.call(self, some, compute_uv, U, S, V);
}

// aten::svd.U(Tensor self, bool some=True, bool compute_uv=True, *, Tensor(a!) U, Tensor(b!) S, Tensor(c!) V) -> (Tensor(a!) U, Tensor(b!) S, Tensor(c!) V)
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> svd_U::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool some, bool compute_uv, at::Tensor & U, at::Tensor & S, at::Tensor & V) {
    static auto op = create_svd_U_typed_handle();
    return op.redispatch(dispatchKeySet, self, some, compute_uv, U, S, V);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(swapaxes_, name, "aten::swapaxes_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(swapaxes_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(swapaxes_, schema_str, "swapaxes_(Tensor(a!) self, int axis0, int axis1) -> Tensor(a!)")

// aten::swapaxes_(Tensor(a!) self, int axis0, int axis1) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<swapaxes_::schema> create_swapaxes__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(swapaxes_::name, swapaxes_::overload_name)
      .typed<swapaxes_::schema>();
}

// aten::swapaxes_(Tensor(a!) self, int axis0, int axis1) -> Tensor(a!)
at::Tensor & swapaxes_::call(at::Tensor & self, int64_t axis0, int64_t axis1) {
    static auto op = create_swapaxes__typed_handle();
    return op.call(self, axis0, axis1);
}

// aten::swapaxes_(Tensor(a!) self, int axis0, int axis1) -> Tensor(a!)
at::Tensor & swapaxes_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t axis0, int64_t axis1) {
    static auto op = create_swapaxes__typed_handle();
    return op.redispatch(dispatchKeySet, self, axis0, axis1);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(swapdims, name, "aten::swapdims")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(swapdims, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(swapdims, schema_str, "swapdims(Tensor(a) self, int dim0, int dim1) -> Tensor(a)")

// aten::swapdims(Tensor(a) self, int dim0, int dim1) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<swapdims::schema> create_swapdims_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(swapdims::name, swapdims::overload_name)
      .typed<swapdims::schema>();
}

// aten::swapdims(Tensor(a) self, int dim0, int dim1) -> Tensor(a)
at::Tensor swapdims::call(const at::Tensor & self, int64_t dim0, int64_t dim1) {
    static auto op = create_swapdims_typed_handle();
    return op.call(self, dim0, dim1);
}

// aten::swapdims(Tensor(a) self, int dim0, int dim1) -> Tensor(a)
at::Tensor swapdims::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim0, int64_t dim1) {
    static auto op = create_swapdims_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim0, dim1);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(swapdims_, name, "aten::swapdims_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(swapdims_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(swapdims_, schema_str, "swapdims_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)")

// aten::swapdims_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<swapdims_::schema> create_swapdims__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(swapdims_::name, swapdims_::overload_name)
      .typed<swapdims_::schema>();
}

// aten::swapdims_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)
at::Tensor & swapdims_::call(at::Tensor & self, int64_t dim0, int64_t dim1) {
    static auto op = create_swapdims__typed_handle();
    return op.call(self, dim0, dim1);
}

// aten::swapdims_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)
at::Tensor & swapdims_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t dim0, int64_t dim1) {
    static auto op = create_swapdims__typed_handle();
    return op.redispatch(dispatchKeySet, self, dim0, dim1);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cholesky, name, "aten::cholesky")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cholesky, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cholesky, schema_str, "cholesky(Tensor self, bool upper=False) -> Tensor")

// aten::cholesky(Tensor self, bool upper=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<cholesky::schema> create_cholesky_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cholesky::name, cholesky::overload_name)
      .typed<cholesky::schema>();
}

// aten::cholesky(Tensor self, bool upper=False) -> Tensor
at::Tensor cholesky::call(const at::Tensor & self, bool upper) {
    static auto op = create_cholesky_typed_handle();
    return op.call(self, upper);
}

// aten::cholesky(Tensor self, bool upper=False) -> Tensor
at::Tensor cholesky::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool upper) {
    static auto op = create_cholesky_typed_handle();
    return op.redispatch(dispatchKeySet, self, upper);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lu_solve, name, "aten::lu_solve")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lu_solve, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lu_solve, schema_str, "lu_solve(Tensor self, Tensor LU_data, Tensor LU_pivots) -> Tensor")

// aten::lu_solve(Tensor self, Tensor LU_data, Tensor LU_pivots) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<lu_solve::schema> create_lu_solve_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(lu_solve::name, lu_solve::overload_name)
      .typed<lu_solve::schema>();
}

// aten::lu_solve(Tensor self, Tensor LU_data, Tensor LU_pivots) -> Tensor
at::Tensor lu_solve::call(const at::Tensor & self, const at::Tensor & LU_data, const at::Tensor & LU_pivots) {
    static auto op = create_lu_solve_typed_handle();
    return op.call(self, LU_data, LU_pivots);
}

// aten::lu_solve(Tensor self, Tensor LU_data, Tensor LU_pivots) -> Tensor
at::Tensor lu_solve::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & LU_data, const at::Tensor & LU_pivots) {
    static auto op = create_lu_solve_typed_handle();
    return op.redispatch(dispatchKeySet, self, LU_data, LU_pivots);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lu_unpack, name, "aten::lu_unpack")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lu_unpack, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lu_unpack, schema_str, "lu_unpack(Tensor LU_data, Tensor LU_pivots, bool unpack_data=True, bool unpack_pivots=True) -> (Tensor P, Tensor L, Tensor U)")

// aten::lu_unpack(Tensor LU_data, Tensor LU_pivots, bool unpack_data=True, bool unpack_pivots=True) -> (Tensor P, Tensor L, Tensor U)
static C10_NOINLINE c10::TypedOperatorHandle<lu_unpack::schema> create_lu_unpack_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(lu_unpack::name, lu_unpack::overload_name)
      .typed<lu_unpack::schema>();
}

// aten::lu_unpack(Tensor LU_data, Tensor LU_pivots, bool unpack_data=True, bool unpack_pivots=True) -> (Tensor P, Tensor L, Tensor U)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> lu_unpack::call(const at::Tensor & LU_data, const at::Tensor & LU_pivots, bool unpack_data, bool unpack_pivots) {
    static auto op = create_lu_unpack_typed_handle();
    return op.call(LU_data, LU_pivots, unpack_data, unpack_pivots);
}

// aten::lu_unpack(Tensor LU_data, Tensor LU_pivots, bool unpack_data=True, bool unpack_pivots=True) -> (Tensor P, Tensor L, Tensor U)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> lu_unpack::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & LU_data, const at::Tensor & LU_pivots, bool unpack_data, bool unpack_pivots) {
    static auto op = create_lu_unpack_typed_handle();
    return op.redispatch(dispatchKeySet, LU_data, LU_pivots, unpack_data, unpack_pivots);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multinomial_out, name, "aten::multinomial")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multinomial_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multinomial_out, schema_str, "multinomial.out(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)")

// aten::multinomial.out(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<multinomial_out::schema> create_multinomial_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(multinomial_out::name, multinomial_out::overload_name)
      .typed<multinomial_out::schema>();
}

// aten::multinomial.out(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & multinomial_out::call(const at::Tensor & self, int64_t num_samples, bool replacement, c10::optional<at::Generator> generator, at::Tensor & out) {
    static auto op = create_multinomial_out_typed_handle();
    return op.call(self, num_samples, replacement, generator, out);
}

// aten::multinomial.out(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & multinomial_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t num_samples, bool replacement, c10::optional<at::Generator> generator, at::Tensor & out) {
    static auto op = create_multinomial_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, num_samples, replacement, generator, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multinomial, name, "aten::multinomial")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multinomial, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multinomial, schema_str, "multinomial(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None) -> Tensor")

// aten::multinomial(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<multinomial::schema> create_multinomial_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(multinomial::name, multinomial::overload_name)
      .typed<multinomial::schema>();
}

// aten::multinomial(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None) -> Tensor
at::Tensor multinomial::call(const at::Tensor & self, int64_t num_samples, bool replacement, c10::optional<at::Generator> generator) {
    static auto op = create_multinomial_typed_handle();
    return op.call(self, num_samples, replacement, generator);
}

// aten::multinomial(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None) -> Tensor
at::Tensor multinomial::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t num_samples, bool replacement, c10::optional<at::Generator> generator) {
    static auto op = create_multinomial_typed_handle();
    return op.redispatch(dispatchKeySet, self, num_samples, replacement, generator);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lgamma_out, name, "aten::lgamma")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lgamma_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lgamma_out, schema_str, "lgamma.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::lgamma.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<lgamma_out::schema> create_lgamma_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(lgamma_out::name, lgamma_out::overload_name)
      .typed<lgamma_out::schema>();
}

// aten::lgamma.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & lgamma_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_lgamma_out_typed_handle();
    return op.call(self, out);
}

// aten::lgamma.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & lgamma_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_lgamma_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lgamma_, name, "aten::lgamma_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lgamma_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lgamma_, schema_str, "lgamma_(Tensor(a!) self) -> Tensor(a!)")

// aten::lgamma_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<lgamma_::schema> create_lgamma__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(lgamma_::name, lgamma_::overload_name)
      .typed<lgamma_::schema>();
}

// aten::lgamma_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & lgamma_::call(at::Tensor & self) {
    static auto op = create_lgamma__typed_handle();
    return op.call(self);
}

// aten::lgamma_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & lgamma_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_lgamma__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lgamma, name, "aten::lgamma")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lgamma, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lgamma, schema_str, "lgamma(Tensor self) -> Tensor")

// aten::lgamma(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<lgamma::schema> create_lgamma_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(lgamma::name, lgamma::overload_name)
      .typed<lgamma::schema>();
}

// aten::lgamma(Tensor self) -> Tensor
at::Tensor lgamma::call(const at::Tensor & self) {
    static auto op = create_lgamma_typed_handle();
    return op.call(self);
}

// aten::lgamma(Tensor self) -> Tensor
at::Tensor lgamma::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_lgamma_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(erfinv_out, name, "aten::erfinv")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(erfinv_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(erfinv_out, schema_str, "erfinv.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::erfinv.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<erfinv_out::schema> create_erfinv_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(erfinv_out::name, erfinv_out::overload_name)
      .typed<erfinv_out::schema>();
}

// aten::erfinv.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & erfinv_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_erfinv_out_typed_handle();
    return op.call(self, out);
}

// aten::erfinv.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & erfinv_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_erfinv_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(signbit_out, name, "aten::signbit")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(signbit_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(signbit_out, schema_str, "signbit.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::signbit.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<signbit_out::schema> create_signbit_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(signbit_out::name, signbit_out::overload_name)
      .typed<signbit_out::schema>();
}

// aten::signbit.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & signbit_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_signbit_out_typed_handle();
    return op.call(self, out);
}

// aten::signbit.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & signbit_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_signbit_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lerp_Scalar_out, name, "aten::lerp")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lerp_Scalar_out, overload_name, "Scalar_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lerp_Scalar_out, schema_str, "lerp.Scalar_out(Tensor self, Tensor end, Scalar weight, *, Tensor(a!) out) -> Tensor(a!)")

// aten::lerp.Scalar_out(Tensor self, Tensor end, Scalar weight, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<lerp_Scalar_out::schema> create_lerp_Scalar_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(lerp_Scalar_out::name, lerp_Scalar_out::overload_name)
      .typed<lerp_Scalar_out::schema>();
}

// aten::lerp.Scalar_out(Tensor self, Tensor end, Scalar weight, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & lerp_Scalar_out::call(const at::Tensor & self, const at::Tensor & end, const at::Scalar & weight, at::Tensor & out) {
    static auto op = create_lerp_Scalar_out_typed_handle();
    return op.call(self, end, weight, out);
}

// aten::lerp.Scalar_out(Tensor self, Tensor end, Scalar weight, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & lerp_Scalar_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & end, const at::Scalar & weight, at::Tensor & out) {
    static auto op = create_lerp_Scalar_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, end, weight, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(histogram_bins_tensor_out, name, "aten::histogram")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(histogram_bins_tensor_out, overload_name, "bins_tensor_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(histogram_bins_tensor_out, schema_str, "histogram.bins_tensor_out(Tensor self, Tensor bins, *, Tensor? weight=None, bool density=False, Tensor(a!) hist, Tensor(b!) bin_edges) -> (Tensor(a!) hist, Tensor(b!) bin_edges)")

// aten::histogram.bins_tensor_out(Tensor self, Tensor bins, *, Tensor? weight=None, bool density=False, Tensor(a!) hist, Tensor(b!) bin_edges) -> (Tensor(a!) hist, Tensor(b!) bin_edges)
static C10_NOINLINE c10::TypedOperatorHandle<histogram_bins_tensor_out::schema> create_histogram_bins_tensor_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(histogram_bins_tensor_out::name, histogram_bins_tensor_out::overload_name)
      .typed<histogram_bins_tensor_out::schema>();
}

// aten::histogram.bins_tensor_out(Tensor self, Tensor bins, *, Tensor? weight=None, bool density=False, Tensor(a!) hist, Tensor(b!) bin_edges) -> (Tensor(a!) hist, Tensor(b!) bin_edges)
::std::tuple<at::Tensor &,at::Tensor &> histogram_bins_tensor_out::call(const at::Tensor & self, const at::Tensor & bins, const c10::optional<at::Tensor> & weight, bool density, at::Tensor & hist, at::Tensor & bin_edges) {
    static auto op = create_histogram_bins_tensor_out_typed_handle();
    return op.call(self, bins, weight, density, hist, bin_edges);
}

// aten::histogram.bins_tensor_out(Tensor self, Tensor bins, *, Tensor? weight=None, bool density=False, Tensor(a!) hist, Tensor(b!) bin_edges) -> (Tensor(a!) hist, Tensor(b!) bin_edges)
::std::tuple<at::Tensor &,at::Tensor &> histogram_bins_tensor_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & bins, const c10::optional<at::Tensor> & weight, bool density, at::Tensor & hist, at::Tensor & bin_edges) {
    static auto op = create_histogram_bins_tensor_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, bins, weight, density, hist, bin_edges);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(histogram_bins_tensor, name, "aten::histogram")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(histogram_bins_tensor, overload_name, "bins_tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(histogram_bins_tensor, schema_str, "histogram.bins_tensor(Tensor self, Tensor bins, *, Tensor? weight=None, bool density=False) -> (Tensor hist, Tensor bin_edges)")

// aten::histogram.bins_tensor(Tensor self, Tensor bins, *, Tensor? weight=None, bool density=False) -> (Tensor hist, Tensor bin_edges)
static C10_NOINLINE c10::TypedOperatorHandle<histogram_bins_tensor::schema> create_histogram_bins_tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(histogram_bins_tensor::name, histogram_bins_tensor::overload_name)
      .typed<histogram_bins_tensor::schema>();
}

// aten::histogram.bins_tensor(Tensor self, Tensor bins, *, Tensor? weight=None, bool density=False) -> (Tensor hist, Tensor bin_edges)
::std::tuple<at::Tensor,at::Tensor> histogram_bins_tensor::call(const at::Tensor & self, const at::Tensor & bins, const c10::optional<at::Tensor> & weight, bool density) {
    static auto op = create_histogram_bins_tensor_typed_handle();
    return op.call(self, bins, weight, density);
}

// aten::histogram.bins_tensor(Tensor self, Tensor bins, *, Tensor? weight=None, bool density=False) -> (Tensor hist, Tensor bin_edges)
::std::tuple<at::Tensor,at::Tensor> histogram_bins_tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & bins, const c10::optional<at::Tensor> & weight, bool density) {
    static auto op = create_histogram_bins_tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, bins, weight, density);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(igamma, name, "aten::igamma")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(igamma, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(igamma, schema_str, "igamma(Tensor self, Tensor other) -> Tensor")

// aten::igamma(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<igamma::schema> create_igamma_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(igamma::name, igamma::overload_name)
      .typed<igamma::schema>();
}

// aten::igamma(Tensor self, Tensor other) -> Tensor
at::Tensor igamma::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_igamma_typed_handle();
    return op.call(self, other);
}

// aten::igamma(Tensor self, Tensor other) -> Tensor
at::Tensor igamma::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_igamma_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nextafter_, name, "aten::nextafter_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nextafter_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nextafter_, schema_str, "nextafter_(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::nextafter_(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<nextafter_::schema> create_nextafter__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nextafter_::name, nextafter_::overload_name)
      .typed<nextafter_::schema>();
}

// aten::nextafter_(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & nextafter_::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create_nextafter__typed_handle();
    return op.call(self, other);
}

// aten::nextafter_(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & nextafter_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create_nextafter__typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(remainder_Tensor_out, name, "aten::remainder")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(remainder_Tensor_out, overload_name, "Tensor_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(remainder_Tensor_out, schema_str, "remainder.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::remainder.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<remainder_Tensor_out::schema> create_remainder_Tensor_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(remainder_Tensor_out::name, remainder_Tensor_out::overload_name)
      .typed<remainder_Tensor_out::schema>();
}

// aten::remainder.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & remainder_Tensor_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_remainder_Tensor_out_typed_handle();
    return op.call(self, other, out);
}

// aten::remainder.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & remainder_Tensor_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_remainder_Tensor_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(remainder_Tensor, name, "aten::remainder")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(remainder_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(remainder_Tensor, schema_str, "remainder.Tensor(Tensor self, Tensor other) -> Tensor")

// aten::remainder.Tensor(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<remainder_Tensor::schema> create_remainder_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(remainder_Tensor::name, remainder_Tensor::overload_name)
      .typed<remainder_Tensor::schema>();
}

// aten::remainder.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor remainder_Tensor::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_remainder_Tensor_typed_handle();
    return op.call(self, other);
}

// aten::remainder.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor remainder_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_remainder_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(remainder__Tensor, name, "aten::remainder_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(remainder__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(remainder__Tensor, schema_str, "remainder_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::remainder_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<remainder__Tensor::schema> create_remainder__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(remainder__Tensor::name, remainder__Tensor::overload_name)
      .typed<remainder__Tensor::schema>();
}

// aten::remainder_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & remainder__Tensor::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create_remainder__Tensor_typed_handle();
    return op.call(self, other);
}

// aten::remainder_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & remainder__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create_remainder__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max, name, "aten::max")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max, schema_str, "max(Tensor self) -> Tensor")

// aten::max(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<max::schema> create_max_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(max::name, max::overload_name)
      .typed<max::schema>();
}

// aten::max(Tensor self) -> Tensor
at::Tensor max::call(const at::Tensor & self) {
    static auto op = create_max_typed_handle();
    return op.call(self);
}

// aten::max(Tensor self) -> Tensor
at::Tensor max::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_max_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(maximum_out, name, "aten::maximum")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(maximum_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(maximum_out, schema_str, "maximum.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::maximum.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<maximum_out::schema> create_maximum_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(maximum_out::name, maximum_out::overload_name)
      .typed<maximum_out::schema>();
}

// aten::maximum.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & maximum_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_maximum_out_typed_handle();
    return op.call(self, other, out);
}

// aten::maximum.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & maximum_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_maximum_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(minimum_out, name, "aten::minimum")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(minimum_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(minimum_out, schema_str, "minimum.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::minimum.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<minimum_out::schema> create_minimum_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(minimum_out::name, minimum_out::overload_name)
      .typed<minimum_out::schema>();
}

// aten::minimum.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & minimum_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_minimum_out_typed_handle();
    return op.call(self, other, out);
}

// aten::minimum.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & minimum_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_minimum_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantile_new, name, "aten::quantile")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantile_new, overload_name, "new")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantile_new, schema_str, "quantile.new(Tensor self, Tensor q, int? dim, bool keepdim, *, str interpolation) -> Tensor")

// aten::quantile.new(Tensor self, Tensor q, int? dim, bool keepdim, *, str interpolation) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<quantile_new::schema> create_quantile_new_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(quantile_new::name, quantile_new::overload_name)
      .typed<quantile_new::schema>();
}

// aten::quantile.new(Tensor self, Tensor q, int? dim, bool keepdim, *, str interpolation) -> Tensor
at::Tensor quantile_new::call(const at::Tensor & self, const at::Tensor & q, c10::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) {
    static auto op = create_quantile_new_typed_handle();
    return op.call(self, q, dim, keepdim, interpolation);
}

// aten::quantile.new(Tensor self, Tensor q, int? dim, bool keepdim, *, str interpolation) -> Tensor
at::Tensor quantile_new::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & q, c10::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) {
    static auto op = create_quantile_new_typed_handle();
    return op.redispatch(dispatchKeySet, self, q, dim, keepdim, interpolation);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sort_values, name, "aten::sort")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sort_values, overload_name, "values")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sort_values, schema_str, "sort.values(Tensor self, int dim=-1, bool descending=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)")

// aten::sort.values(Tensor self, int dim=-1, bool descending=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
static C10_NOINLINE c10::TypedOperatorHandle<sort_values::schema> create_sort_values_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sort_values::name, sort_values::overload_name)
      .typed<sort_values::schema>();
}

// aten::sort.values(Tensor self, int dim=-1, bool descending=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> sort_values::call(const at::Tensor & self, int64_t dim, bool descending, at::Tensor & values, at::Tensor & indices) {
    static auto op = create_sort_values_typed_handle();
    return op.call(self, dim, descending, values, indices);
}

// aten::sort.values(Tensor self, int dim=-1, bool descending=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> sort_values::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool descending, at::Tensor & values, at::Tensor & indices) {
    static auto op = create_sort_values_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, descending, values, indices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sort_dimname_values, name, "aten::sort")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sort_dimname_values, overload_name, "dimname_values")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sort_dimname_values, schema_str, "sort.dimname_values(Tensor self, Dimname dim, bool descending=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)")

// aten::sort.dimname_values(Tensor self, Dimname dim, bool descending=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
static C10_NOINLINE c10::TypedOperatorHandle<sort_dimname_values::schema> create_sort_dimname_values_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sort_dimname_values::name, sort_dimname_values::overload_name)
      .typed<sort_dimname_values::schema>();
}

// aten::sort.dimname_values(Tensor self, Dimname dim, bool descending=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> sort_dimname_values::call(const at::Tensor & self, at::Dimname dim, bool descending, at::Tensor & values, at::Tensor & indices) {
    static auto op = create_sort_dimname_values_typed_handle();
    return op.call(self, dim, descending, values, indices);
}

// aten::sort.dimname_values(Tensor self, Dimname dim, bool descending=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> sort_dimname_values::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, bool descending, at::Tensor & values, at::Tensor & indices) {
    static auto op = create_sort_dimname_values_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, descending, values, indices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(any_all_out, name, "aten::any")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(any_all_out, overload_name, "all_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(any_all_out, schema_str, "any.all_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::any.all_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<any_all_out::schema> create_any_all_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(any_all_out::name, any_all_out::overload_name)
      .typed<any_all_out::schema>();
}

// aten::any.all_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & any_all_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_any_all_out_typed_handle();
    return op.call(self, out);
}

// aten::any.all_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & any_all_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_any_all_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pow__Scalar, name, "aten::pow_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pow__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pow__Scalar, schema_str, "pow_.Scalar(Tensor(a!) self, Scalar exponent) -> Tensor(a!)")

// aten::pow_.Scalar(Tensor(a!) self, Scalar exponent) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<pow__Scalar::schema> create_pow__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(pow__Scalar::name, pow__Scalar::overload_name)
      .typed<pow__Scalar::schema>();
}

// aten::pow_.Scalar(Tensor(a!) self, Scalar exponent) -> Tensor(a!)
at::Tensor & pow__Scalar::call(at::Tensor & self, const at::Scalar & exponent) {
    static auto op = create_pow__Scalar_typed_handle();
    return op.call(self, exponent);
}

// aten::pow_.Scalar(Tensor(a!) self, Scalar exponent) -> Tensor(a!)
at::Tensor & pow__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & exponent) {
    static auto op = create_pow__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, exponent);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(float_power__Scalar, name, "aten::float_power_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(float_power__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(float_power__Scalar, schema_str, "float_power_.Scalar(Tensor(a!) self, Scalar exponent) -> Tensor(a!)")

// aten::float_power_.Scalar(Tensor(a!) self, Scalar exponent) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<float_power__Scalar::schema> create_float_power__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(float_power__Scalar::name, float_power__Scalar::overload_name)
      .typed<float_power__Scalar::schema>();
}

// aten::float_power_.Scalar(Tensor(a!) self, Scalar exponent) -> Tensor(a!)
at::Tensor & float_power__Scalar::call(at::Tensor & self, const at::Scalar & exponent) {
    static auto op = create_float_power__Scalar_typed_handle();
    return op.call(self, exponent);
}

// aten::float_power_.Scalar(Tensor(a!) self, Scalar exponent) -> Tensor(a!)
at::Tensor & float_power__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & exponent) {
    static auto op = create_float_power__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, exponent);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(float_power__Tensor, name, "aten::float_power_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(float_power__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(float_power__Tensor, schema_str, "float_power_.Tensor(Tensor(a!) self, Tensor exponent) -> Tensor(a!)")

// aten::float_power_.Tensor(Tensor(a!) self, Tensor exponent) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<float_power__Tensor::schema> create_float_power__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(float_power__Tensor::name, float_power__Tensor::overload_name)
      .typed<float_power__Tensor::schema>();
}

// aten::float_power_.Tensor(Tensor(a!) self, Tensor exponent) -> Tensor(a!)
at::Tensor & float_power__Tensor::call(at::Tensor & self, const at::Tensor & exponent) {
    static auto op = create_float_power__Tensor_typed_handle();
    return op.call(self, exponent);
}

// aten::float_power_.Tensor(Tensor(a!) self, Tensor exponent) -> Tensor(a!)
at::Tensor & float_power__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & exponent) {
    static auto op = create_float_power__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, exponent);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(normal_, name, "aten::normal_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(normal_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(normal_, schema_str, "normal_(Tensor(a!) self, float mean=0, float std=1, *, Generator? generator=None) -> Tensor(a!)")

// aten::normal_(Tensor(a!) self, float mean=0, float std=1, *, Generator? generator=None) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<normal_::schema> create_normal__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(normal_::name, normal_::overload_name)
      .typed<normal_::schema>();
}

// aten::normal_(Tensor(a!) self, float mean=0, float std=1, *, Generator? generator=None) -> Tensor(a!)
at::Tensor & normal_::call(at::Tensor & self, double mean, double std, c10::optional<at::Generator> generator) {
    static auto op = create_normal__typed_handle();
    return op.call(self, mean, std, generator);
}

// aten::normal_(Tensor(a!) self, float mean=0, float std=1, *, Generator? generator=None) -> Tensor(a!)
at::Tensor & normal_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, double mean, double std, c10::optional<at::Generator> generator) {
    static auto op = create_normal__typed_handle();
    return op.redispatch(dispatchKeySet, self, mean, std, generator);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(normal_float_Tensor, name, "aten::normal")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(normal_float_Tensor, overload_name, "float_Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(normal_float_Tensor, schema_str, "normal.float_Tensor(float mean, Tensor std, *, Generator? generator=None) -> Tensor")

// aten::normal.float_Tensor(float mean, Tensor std, *, Generator? generator=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<normal_float_Tensor::schema> create_normal_float_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(normal_float_Tensor::name, normal_float_Tensor::overload_name)
      .typed<normal_float_Tensor::schema>();
}

// aten::normal.float_Tensor(float mean, Tensor std, *, Generator? generator=None) -> Tensor
at::Tensor normal_float_Tensor::call(double mean, const at::Tensor & std, c10::optional<at::Generator> generator) {
    static auto op = create_normal_float_Tensor_typed_handle();
    return op.call(mean, std, generator);
}

// aten::normal.float_Tensor(float mean, Tensor std, *, Generator? generator=None) -> Tensor
at::Tensor normal_float_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, double mean, const at::Tensor & std, c10::optional<at::Generator> generator) {
    static auto op = create_normal_float_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, mean, std, generator);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_amp_foreach_non_finite_check_and_unscale_, name, "aten::_amp_foreach_non_finite_check_and_unscale_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_amp_foreach_non_finite_check_and_unscale_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_amp_foreach_non_finite_check_and_unscale_, schema_str, "_amp_foreach_non_finite_check_and_unscale_(Tensor(a!)[] self, Tensor(b!) found_inf, Tensor inv_scale) -> ()")

// aten::_amp_foreach_non_finite_check_and_unscale_(Tensor(a!)[] self, Tensor(b!) found_inf, Tensor inv_scale) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_amp_foreach_non_finite_check_and_unscale_::schema> create__amp_foreach_non_finite_check_and_unscale__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_amp_foreach_non_finite_check_and_unscale_::name, _amp_foreach_non_finite_check_and_unscale_::overload_name)
      .typed<_amp_foreach_non_finite_check_and_unscale_::schema>();
}

// aten::_amp_foreach_non_finite_check_and_unscale_(Tensor(a!)[] self, Tensor(b!) found_inf, Tensor inv_scale) -> ()
void _amp_foreach_non_finite_check_and_unscale_::call(at::TensorList self, at::Tensor & found_inf, const at::Tensor & inv_scale) {
    static auto op = create__amp_foreach_non_finite_check_and_unscale__typed_handle();
    return op.call(self, found_inf, inv_scale);
}

// aten::_amp_foreach_non_finite_check_and_unscale_(Tensor(a!)[] self, Tensor(b!) found_inf, Tensor inv_scale) -> ()
void _amp_foreach_non_finite_check_and_unscale_::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self, at::Tensor & found_inf, const at::Tensor & inv_scale) {
    static auto op = create__amp_foreach_non_finite_check_and_unscale__typed_handle();
    return op.redispatch(dispatchKeySet, self, found_inf, inv_scale);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_mul_Scalar, name, "aten::_foreach_mul")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_mul_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_mul_Scalar, schema_str, "_foreach_mul.Scalar(Tensor[] tensors, Scalar scalar) -> Tensor[]")

// aten::_foreach_mul.Scalar(Tensor[] tensors, Scalar scalar) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_mul_Scalar::schema> create__foreach_mul_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_mul_Scalar::name, _foreach_mul_Scalar::overload_name)
      .typed<_foreach_mul_Scalar::schema>();
}

// aten::_foreach_mul.Scalar(Tensor[] tensors, Scalar scalar) -> Tensor[]
::std::vector<at::Tensor> _foreach_mul_Scalar::call(at::TensorList tensors, const at::Scalar & scalar) {
    static auto op = create__foreach_mul_Scalar_typed_handle();
    return op.call(tensors, scalar);
}

// aten::_foreach_mul.Scalar(Tensor[] tensors, Scalar scalar) -> Tensor[]
::std::vector<at::Tensor> _foreach_mul_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, const at::Scalar & scalar) {
    static auto op = create__foreach_mul_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, tensors, scalar);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_add_ScalarList, name, "aten::_foreach_add")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_add_ScalarList, overload_name, "ScalarList")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_add_ScalarList, schema_str, "_foreach_add.ScalarList(Tensor[] tensors, Scalar[] scalars) -> Tensor[]")

// aten::_foreach_add.ScalarList(Tensor[] tensors, Scalar[] scalars) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_add_ScalarList::schema> create__foreach_add_ScalarList_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_add_ScalarList::name, _foreach_add_ScalarList::overload_name)
      .typed<_foreach_add_ScalarList::schema>();
}

// aten::_foreach_add.ScalarList(Tensor[] tensors, Scalar[] scalars) -> Tensor[]
::std::vector<at::Tensor> _foreach_add_ScalarList::call(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars) {
    static auto op = create__foreach_add_ScalarList_typed_handle();
    return op.call(tensors, scalars);
}

// aten::_foreach_add.ScalarList(Tensor[] tensors, Scalar[] scalars) -> Tensor[]
::std::vector<at::Tensor> _foreach_add_ScalarList::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, at::ArrayRef<at::Scalar> scalars) {
    static auto op = create__foreach_add_ScalarList_typed_handle();
    return op.redispatch(dispatchKeySet, tensors, scalars);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_div__ScalarList, name, "aten::_foreach_div_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_div__ScalarList, overload_name, "ScalarList")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_div__ScalarList, schema_str, "_foreach_div_.ScalarList(Tensor(a!)[] self, Scalar[] scalars) -> ()")

// aten::_foreach_div_.ScalarList(Tensor(a!)[] self, Scalar[] scalars) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_div__ScalarList::schema> create__foreach_div__ScalarList_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_div__ScalarList::name, _foreach_div__ScalarList::overload_name)
      .typed<_foreach_div__ScalarList::schema>();
}

// aten::_foreach_div_.ScalarList(Tensor(a!)[] self, Scalar[] scalars) -> ()
void _foreach_div__ScalarList::call(at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
    static auto op = create__foreach_div__ScalarList_typed_handle();
    return op.call(self, scalars);
}

// aten::_foreach_div_.ScalarList(Tensor(a!)[] self, Scalar[] scalars) -> ()
void _foreach_div__ScalarList::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
    static auto op = create__foreach_div__ScalarList_typed_handle();
    return op.redispatch(dispatchKeySet, self, scalars);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_mul_ScalarList, name, "aten::_foreach_mul")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_mul_ScalarList, overload_name, "ScalarList")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_mul_ScalarList, schema_str, "_foreach_mul.ScalarList(Tensor[] tensors, Scalar[] scalars) -> Tensor[]")

// aten::_foreach_mul.ScalarList(Tensor[] tensors, Scalar[] scalars) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_mul_ScalarList::schema> create__foreach_mul_ScalarList_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_mul_ScalarList::name, _foreach_mul_ScalarList::overload_name)
      .typed<_foreach_mul_ScalarList::schema>();
}

// aten::_foreach_mul.ScalarList(Tensor[] tensors, Scalar[] scalars) -> Tensor[]
::std::vector<at::Tensor> _foreach_mul_ScalarList::call(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars) {
    static auto op = create__foreach_mul_ScalarList_typed_handle();
    return op.call(tensors, scalars);
}

// aten::_foreach_mul.ScalarList(Tensor[] tensors, Scalar[] scalars) -> Tensor[]
::std::vector<at::Tensor> _foreach_mul_ScalarList::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, at::ArrayRef<at::Scalar> scalars) {
    static auto op = create__foreach_mul_ScalarList_typed_handle();
    return op.redispatch(dispatchKeySet, tensors, scalars);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_acos_, name, "aten::_foreach_acos_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_acos_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_acos_, schema_str, "_foreach_acos_(Tensor(a!)[] self) -> ()")

// aten::_foreach_acos_(Tensor(a!)[] self) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_acos_::schema> create__foreach_acos__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_acos_::name, _foreach_acos_::overload_name)
      .typed<_foreach_acos_::schema>();
}

// aten::_foreach_acos_(Tensor(a!)[] self) -> ()
void _foreach_acos_::call(at::TensorList self) {
    static auto op = create__foreach_acos__typed_handle();
    return op.call(self);
}

// aten::_foreach_acos_(Tensor(a!)[] self) -> ()
void _foreach_acos_::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self) {
    static auto op = create__foreach_acos__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_asin_, name, "aten::_foreach_asin_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_asin_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_asin_, schema_str, "_foreach_asin_(Tensor(a!)[] self) -> ()")

// aten::_foreach_asin_(Tensor(a!)[] self) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_asin_::schema> create__foreach_asin__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_asin_::name, _foreach_asin_::overload_name)
      .typed<_foreach_asin_::schema>();
}

// aten::_foreach_asin_(Tensor(a!)[] self) -> ()
void _foreach_asin_::call(at::TensorList self) {
    static auto op = create__foreach_asin__typed_handle();
    return op.call(self);
}

// aten::_foreach_asin_(Tensor(a!)[] self) -> ()
void _foreach_asin_::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self) {
    static auto op = create__foreach_asin__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_cosh, name, "aten::_foreach_cosh")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_cosh, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_cosh, schema_str, "_foreach_cosh(Tensor[] tensors) -> Tensor[]")

// aten::_foreach_cosh(Tensor[] tensors) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_cosh::schema> create__foreach_cosh_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_cosh::name, _foreach_cosh::overload_name)
      .typed<_foreach_cosh::schema>();
}

// aten::_foreach_cosh(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_cosh::call(at::TensorList tensors) {
    static auto op = create__foreach_cosh_typed_handle();
    return op.call(tensors);
}

// aten::_foreach_cosh(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_cosh::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create__foreach_cosh_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_cosh_, name, "aten::_foreach_cosh_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_cosh_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_cosh_, schema_str, "_foreach_cosh_(Tensor(a!)[] self) -> ()")

// aten::_foreach_cosh_(Tensor(a!)[] self) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_cosh_::schema> create__foreach_cosh__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_cosh_::name, _foreach_cosh_::overload_name)
      .typed<_foreach_cosh_::schema>();
}

// aten::_foreach_cosh_(Tensor(a!)[] self) -> ()
void _foreach_cosh_::call(at::TensorList self) {
    static auto op = create__foreach_cosh__typed_handle();
    return op.call(self);
}

// aten::_foreach_cosh_(Tensor(a!)[] self) -> ()
void _foreach_cosh_::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self) {
    static auto op = create__foreach_cosh__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_erfc, name, "aten::_foreach_erfc")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_erfc, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_erfc, schema_str, "_foreach_erfc(Tensor[] tensors) -> Tensor[]")

// aten::_foreach_erfc(Tensor[] tensors) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_erfc::schema> create__foreach_erfc_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_erfc::name, _foreach_erfc::overload_name)
      .typed<_foreach_erfc::schema>();
}

// aten::_foreach_erfc(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_erfc::call(at::TensorList tensors) {
    static auto op = create__foreach_erfc_typed_handle();
    return op.call(tensors);
}

// aten::_foreach_erfc(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_erfc::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create__foreach_erfc_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_expm1_, name, "aten::_foreach_expm1_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_expm1_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_expm1_, schema_str, "_foreach_expm1_(Tensor(a!)[] self) -> ()")

// aten::_foreach_expm1_(Tensor(a!)[] self) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_expm1_::schema> create__foreach_expm1__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_expm1_::name, _foreach_expm1_::overload_name)
      .typed<_foreach_expm1_::schema>();
}

// aten::_foreach_expm1_(Tensor(a!)[] self) -> ()
void _foreach_expm1_::call(at::TensorList self) {
    static auto op = create__foreach_expm1__typed_handle();
    return op.call(self);
}

// aten::_foreach_expm1_(Tensor(a!)[] self) -> ()
void _foreach_expm1_::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self) {
    static auto op = create__foreach_expm1__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_log_, name, "aten::_foreach_log_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_log_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_log_, schema_str, "_foreach_log_(Tensor(a!)[] self) -> ()")

// aten::_foreach_log_(Tensor(a!)[] self) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_log_::schema> create__foreach_log__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_log_::name, _foreach_log_::overload_name)
      .typed<_foreach_log_::schema>();
}

// aten::_foreach_log_(Tensor(a!)[] self) -> ()
void _foreach_log_::call(at::TensorList self) {
    static auto op = create__foreach_log__typed_handle();
    return op.call(self);
}

// aten::_foreach_log_(Tensor(a!)[] self) -> ()
void _foreach_log_::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self) {
    static auto op = create__foreach_log__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_tan_, name, "aten::_foreach_tan_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_tan_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_tan_, schema_str, "_foreach_tan_(Tensor(a!)[] self) -> ()")

// aten::_foreach_tan_(Tensor(a!)[] self) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_tan_::schema> create__foreach_tan__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_tan_::name, _foreach_tan_::overload_name)
      .typed<_foreach_tan_::schema>();
}

// aten::_foreach_tan_(Tensor(a!)[] self) -> ()
void _foreach_tan_::call(at::TensorList self) {
    static auto op = create__foreach_tan__typed_handle();
    return op.call(self);
}

// aten::_foreach_tan_(Tensor(a!)[] self) -> ()
void _foreach_tan_::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self) {
    static auto op = create__foreach_tan__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_round, name, "aten::_foreach_round")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_round, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_round, schema_str, "_foreach_round(Tensor[] tensors) -> Tensor[]")

// aten::_foreach_round(Tensor[] tensors) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_round::schema> create__foreach_round_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_round::name, _foreach_round::overload_name)
      .typed<_foreach_round::schema>();
}

// aten::_foreach_round(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_round::call(at::TensorList tensors) {
    static auto op = create__foreach_round_typed_handle();
    return op.call(tensors);
}

// aten::_foreach_round(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_round::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create__foreach_round_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_lgamma, name, "aten::_foreach_lgamma")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_lgamma, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_lgamma, schema_str, "_foreach_lgamma(Tensor[] tensors) -> Tensor[]")

// aten::_foreach_lgamma(Tensor[] tensors) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_lgamma::schema> create__foreach_lgamma_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_lgamma::name, _foreach_lgamma::overload_name)
      .typed<_foreach_lgamma::schema>();
}

// aten::_foreach_lgamma(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_lgamma::call(at::TensorList tensors) {
    static auto op = create__foreach_lgamma_typed_handle();
    return op.call(tensors);
}

// aten::_foreach_lgamma(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_lgamma::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create__foreach_lgamma_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_frac, name, "aten::_foreach_frac")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_frac, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_frac, schema_str, "_foreach_frac(Tensor[] tensors) -> Tensor[]")

// aten::_foreach_frac(Tensor[] tensors) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_frac::schema> create__foreach_frac_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_frac::name, _foreach_frac::overload_name)
      .typed<_foreach_frac::schema>();
}

// aten::_foreach_frac(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_frac::call(at::TensorList tensors) {
    static auto op = create__foreach_frac_typed_handle();
    return op.call(tensors);
}

// aten::_foreach_frac(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_frac::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create__foreach_frac_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_trunc, name, "aten::_foreach_trunc")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_trunc, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_trunc, schema_str, "_foreach_trunc(Tensor[] tensors) -> Tensor[]")

// aten::_foreach_trunc(Tensor[] tensors) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_trunc::schema> create__foreach_trunc_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_trunc::name, _foreach_trunc::overload_name)
      .typed<_foreach_trunc::schema>();
}

// aten::_foreach_trunc(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_trunc::call(at::TensorList tensors) {
    static auto op = create__foreach_trunc_typed_handle();
    return op.call(tensors);
}

// aten::_foreach_trunc(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_trunc::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create__foreach_trunc_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_trunc_, name, "aten::_foreach_trunc_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_trunc_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_trunc_, schema_str, "_foreach_trunc_(Tensor(a!)[] self) -> ()")

// aten::_foreach_trunc_(Tensor(a!)[] self) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_trunc_::schema> create__foreach_trunc__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_trunc_::name, _foreach_trunc_::overload_name)
      .typed<_foreach_trunc_::schema>();
}

// aten::_foreach_trunc_(Tensor(a!)[] self) -> ()
void _foreach_trunc_::call(at::TensorList self) {
    static auto op = create__foreach_trunc__typed_handle();
    return op.call(self);
}

// aten::_foreach_trunc_(Tensor(a!)[] self) -> ()
void _foreach_trunc_::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self) {
    static auto op = create__foreach_trunc__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_addcmul__ScalarList, name, "aten::_foreach_addcmul_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_addcmul__ScalarList, overload_name, "ScalarList")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_addcmul__ScalarList, schema_str, "_foreach_addcmul_.ScalarList(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars) -> ()")

// aten::_foreach_addcmul_.ScalarList(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_addcmul__ScalarList::schema> create__foreach_addcmul__ScalarList_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_addcmul__ScalarList::name, _foreach_addcmul__ScalarList::overload_name)
      .typed<_foreach_addcmul__ScalarList::schema>();
}

// aten::_foreach_addcmul_.ScalarList(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars) -> ()
void _foreach_addcmul__ScalarList::call(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars) {
    static auto op = create__foreach_addcmul__ScalarList_typed_handle();
    return op.call(self, tensor1, tensor2, scalars);
}

// aten::_foreach_addcmul_.ScalarList(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars) -> ()
void _foreach_addcmul__ScalarList::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars) {
    static auto op = create__foreach_addcmul__ScalarList_typed_handle();
    return op.redispatch(dispatchKeySet, self, tensor1, tensor2, scalars);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_addcdiv_Scalar, name, "aten::_foreach_addcdiv")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_addcdiv_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_addcdiv_Scalar, schema_str, "_foreach_addcdiv.Scalar(Tensor[] input, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1) -> Tensor[]")

// aten::_foreach_addcdiv.Scalar(Tensor[] input, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_addcdiv_Scalar::schema> create__foreach_addcdiv_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_addcdiv_Scalar::name, _foreach_addcdiv_Scalar::overload_name)
      .typed<_foreach_addcdiv_Scalar::schema>();
}

// aten::_foreach_addcdiv.Scalar(Tensor[] input, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1) -> Tensor[]
::std::vector<at::Tensor> _foreach_addcdiv_Scalar::call(at::TensorList input, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value) {
    static auto op = create__foreach_addcdiv_Scalar_typed_handle();
    return op.call(input, tensor1, tensor2, value);
}

// aten::_foreach_addcdiv.Scalar(Tensor[] input, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1) -> Tensor[]
::std::vector<at::Tensor> _foreach_addcdiv_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList input, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value) {
    static auto op = create__foreach_addcdiv_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, input, tensor1, tensor2, value);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bucketize_Tensor, name, "aten::bucketize")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bucketize_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bucketize_Tensor, schema_str, "bucketize.Tensor(Tensor self, Tensor boundaries, *, bool out_int32=False, bool right=False) -> Tensor")

// aten::bucketize.Tensor(Tensor self, Tensor boundaries, *, bool out_int32=False, bool right=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<bucketize_Tensor::schema> create_bucketize_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bucketize_Tensor::name, bucketize_Tensor::overload_name)
      .typed<bucketize_Tensor::schema>();
}

// aten::bucketize.Tensor(Tensor self, Tensor boundaries, *, bool out_int32=False, bool right=False) -> Tensor
at::Tensor bucketize_Tensor::call(const at::Tensor & self, const at::Tensor & boundaries, bool out_int32, bool right) {
    static auto op = create_bucketize_Tensor_typed_handle();
    return op.call(self, boundaries, out_int32, right);
}

// aten::bucketize.Tensor(Tensor self, Tensor boundaries, *, bool out_int32=False, bool right=False) -> Tensor
at::Tensor bucketize_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & boundaries, bool out_int32, bool right) {
    static auto op = create_bucketize_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, boundaries, out_int32, right);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bucketize_Scalar, name, "aten::bucketize")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bucketize_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bucketize_Scalar, schema_str, "bucketize.Scalar(Scalar self, Tensor boundaries, *, bool out_int32=False, bool right=False) -> Tensor")

// aten::bucketize.Scalar(Scalar self, Tensor boundaries, *, bool out_int32=False, bool right=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<bucketize_Scalar::schema> create_bucketize_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bucketize_Scalar::name, bucketize_Scalar::overload_name)
      .typed<bucketize_Scalar::schema>();
}

// aten::bucketize.Scalar(Scalar self, Tensor boundaries, *, bool out_int32=False, bool right=False) -> Tensor
at::Tensor bucketize_Scalar::call(const at::Scalar & self, const at::Tensor & boundaries, bool out_int32, bool right) {
    static auto op = create_bucketize_Scalar_typed_handle();
    return op.call(self, boundaries, out_int32, right);
}

// aten::bucketize.Scalar(Scalar self, Tensor boundaries, *, bool out_int32=False, bool right=False) -> Tensor
at::Tensor bucketize_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Scalar & self, const at::Tensor & boundaries, bool out_int32, bool right) {
    static auto op = create_bucketize_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, boundaries, out_int32, right);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(searchsorted_Tensor_out, name, "aten::searchsorted")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(searchsorted_Tensor_out, overload_name, "Tensor_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(searchsorted_Tensor_out, schema_str, "searchsorted.Tensor_out(Tensor sorted_sequence, Tensor self, *, bool out_int32=False, bool right=False, Tensor(a!) out) -> Tensor(a!)")

// aten::searchsorted.Tensor_out(Tensor sorted_sequence, Tensor self, *, bool out_int32=False, bool right=False, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<searchsorted_Tensor_out::schema> create_searchsorted_Tensor_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(searchsorted_Tensor_out::name, searchsorted_Tensor_out::overload_name)
      .typed<searchsorted_Tensor_out::schema>();
}

// aten::searchsorted.Tensor_out(Tensor sorted_sequence, Tensor self, *, bool out_int32=False, bool right=False, Tensor(a!) out) -> Tensor(a!)
at::Tensor & searchsorted_Tensor_out::call(const at::Tensor & sorted_sequence, const at::Tensor & self, bool out_int32, bool right, at::Tensor & out) {
    static auto op = create_searchsorted_Tensor_out_typed_handle();
    return op.call(sorted_sequence, self, out_int32, right, out);
}

// aten::searchsorted.Tensor_out(Tensor sorted_sequence, Tensor self, *, bool out_int32=False, bool right=False, Tensor(a!) out) -> Tensor(a!)
at::Tensor & searchsorted_Tensor_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & sorted_sequence, const at::Tensor & self, bool out_int32, bool right, at::Tensor & out) {
    static auto op = create_searchsorted_Tensor_out_typed_handle();
    return op.redispatch(dispatchKeySet, sorted_sequence, self, out_int32, right, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mse_loss_backward, name, "aten::mse_loss_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mse_loss_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mse_loss_backward, schema_str, "mse_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor")

// aten::mse_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<mse_loss_backward::schema> create_mse_loss_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mse_loss_backward::name, mse_loss_backward::overload_name)
      .typed<mse_loss_backward::schema>();
}

// aten::mse_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor
at::Tensor mse_loss_backward::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
    static auto op = create_mse_loss_backward_typed_handle();
    return op.call(grad_output, self, target, reduction);
}

// aten::mse_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor
at::Tensor mse_loss_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
    static auto op = create_mse_loss_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, target, reduction);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(l1_loss_out, name, "aten::l1_loss")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(l1_loss_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(l1_loss_out, schema_str, "l1_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)")

// aten::l1_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<l1_loss_out::schema> create_l1_loss_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(l1_loss_out::name, l1_loss_out::overload_name)
      .typed<l1_loss_out::schema>();
}

// aten::l1_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & l1_loss_out::call(const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & out) {
    static auto op = create_l1_loss_out_typed_handle();
    return op.call(self, target, reduction, out);
}

// aten::l1_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & l1_loss_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & out) {
    static auto op = create_l1_loss_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, target, reduction, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multi_margin_loss_out, name, "aten::multi_margin_loss")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multi_margin_loss_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multi_margin_loss_out, schema_str, "multi_margin_loss.out(Tensor self, Tensor target, Scalar p=1, Scalar margin=1, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)")

// aten::multi_margin_loss.out(Tensor self, Tensor target, Scalar p=1, Scalar margin=1, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<multi_margin_loss_out::schema> create_multi_margin_loss_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(multi_margin_loss_out::name, multi_margin_loss_out::overload_name)
      .typed<multi_margin_loss_out::schema>();
}

// aten::multi_margin_loss.out(Tensor self, Tensor target, Scalar p=1, Scalar margin=1, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & multi_margin_loss_out::call(const at::Tensor & self, const at::Tensor & target, const at::Scalar & p, const at::Scalar & margin, const c10::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & out) {
    static auto op = create_multi_margin_loss_out_typed_handle();
    return op.call(self, target, p, margin, weight, reduction, out);
}

// aten::multi_margin_loss.out(Tensor self, Tensor target, Scalar p=1, Scalar margin=1, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & multi_margin_loss_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, const at::Scalar & p, const at::Scalar & margin, const c10::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & out) {
    static auto op = create_multi_margin_loss_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, target, p, margin, weight, reduction, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multi_margin_loss_backward, name, "aten::multi_margin_loss_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multi_margin_loss_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multi_margin_loss_backward, schema_str, "multi_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, Scalar p, Scalar margin, Tensor? weight=None, int reduction=Mean) -> Tensor")

// aten::multi_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, Scalar p, Scalar margin, Tensor? weight=None, int reduction=Mean) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<multi_margin_loss_backward::schema> create_multi_margin_loss_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(multi_margin_loss_backward::name, multi_margin_loss_backward::overload_name)
      .typed<multi_margin_loss_backward::schema>();
}

// aten::multi_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, Scalar p, Scalar margin, Tensor? weight=None, int reduction=Mean) -> Tensor
at::Tensor multi_margin_loss_backward::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const at::Scalar & p, const at::Scalar & margin, const c10::optional<at::Tensor> & weight, int64_t reduction) {
    static auto op = create_multi_margin_loss_backward_typed_handle();
    return op.call(grad_output, self, target, p, margin, weight, reduction);
}

// aten::multi_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, Scalar p, Scalar margin, Tensor? weight=None, int reduction=Mean) -> Tensor
at::Tensor multi_margin_loss_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const at::Scalar & p, const at::Scalar & margin, const c10::optional<at::Tensor> & weight, int64_t reduction) {
    static auto op = create_multi_margin_loss_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, target, p, margin, weight, reduction);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multilabel_margin_loss_backward_grad_input, name, "aten::multilabel_margin_loss_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multilabel_margin_loss_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multilabel_margin_loss_backward_grad_input, schema_str, "multilabel_margin_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, Tensor is_target, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::multilabel_margin_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, Tensor is_target, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<multilabel_margin_loss_backward_grad_input::schema> create_multilabel_margin_loss_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(multilabel_margin_loss_backward_grad_input::name, multilabel_margin_loss_backward_grad_input::overload_name)
      .typed<multilabel_margin_loss_backward_grad_input::schema>();
}

// aten::multilabel_margin_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, Tensor is_target, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & multilabel_margin_loss_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, const at::Tensor & is_target, at::Tensor & grad_input) {
    static auto op = create_multilabel_margin_loss_backward_grad_input_typed_handle();
    return op.call(grad_output, self, target, reduction, is_target, grad_input);
}

// aten::multilabel_margin_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, Tensor is_target, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & multilabel_margin_loss_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, const at::Tensor & is_target, at::Tensor & grad_input) {
    static auto op = create_multilabel_margin_loss_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, target, reduction, is_target, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multilabel_margin_loss_backward, name, "aten::multilabel_margin_loss_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multilabel_margin_loss_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multilabel_margin_loss_backward, schema_str, "multilabel_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, Tensor is_target) -> Tensor")

// aten::multilabel_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, Tensor is_target) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<multilabel_margin_loss_backward::schema> create_multilabel_margin_loss_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(multilabel_margin_loss_backward::name, multilabel_margin_loss_backward::overload_name)
      .typed<multilabel_margin_loss_backward::schema>();
}

// aten::multilabel_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, Tensor is_target) -> Tensor
at::Tensor multilabel_margin_loss_backward::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, const at::Tensor & is_target) {
    static auto op = create_multilabel_margin_loss_backward_typed_handle();
    return op.call(grad_output, self, target, reduction, is_target);
}

// aten::multilabel_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, Tensor is_target) -> Tensor
at::Tensor multilabel_margin_loss_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, const at::Tensor & is_target) {
    static auto op = create_multilabel_margin_loss_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, target, reduction, is_target);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(smooth_l1_loss_backward_grad_input, name, "aten::smooth_l1_loss_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(smooth_l1_loss_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(smooth_l1_loss_backward_grad_input, schema_str, "smooth_l1_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, float beta, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::smooth_l1_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, float beta, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<smooth_l1_loss_backward_grad_input::schema> create_smooth_l1_loss_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(smooth_l1_loss_backward_grad_input::name, smooth_l1_loss_backward_grad_input::overload_name)
      .typed<smooth_l1_loss_backward_grad_input::schema>();
}

// aten::smooth_l1_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, float beta, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & smooth_l1_loss_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta, at::Tensor & grad_input) {
    static auto op = create_smooth_l1_loss_backward_grad_input_typed_handle();
    return op.call(grad_output, self, target, reduction, beta, grad_input);
}

// aten::smooth_l1_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, float beta, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & smooth_l1_loss_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta, at::Tensor & grad_input) {
    static auto op = create_smooth_l1_loss_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, target, reduction, beta, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(soft_margin_loss_backward_grad_input, name, "aten::soft_margin_loss_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(soft_margin_loss_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(soft_margin_loss_backward_grad_input, schema_str, "soft_margin_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::soft_margin_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<soft_margin_loss_backward_grad_input::schema> create_soft_margin_loss_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(soft_margin_loss_backward_grad_input::name, soft_margin_loss_backward_grad_input::overload_name)
      .typed<soft_margin_loss_backward_grad_input::schema>();
}

// aten::soft_margin_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & soft_margin_loss_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & grad_input) {
    static auto op = create_soft_margin_loss_backward_grad_input_typed_handle();
    return op.call(grad_output, self, target, reduction, grad_input);
}

// aten::soft_margin_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & soft_margin_loss_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & grad_input) {
    static auto op = create_soft_margin_loss_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, target, reduction, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(elu_backward, name, "aten::elu_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(elu_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(elu_backward, schema_str, "elu_backward(Tensor grad_output, Scalar alpha, Scalar scale, Scalar input_scale, bool is_result, Tensor self_or_result) -> Tensor")

// aten::elu_backward(Tensor grad_output, Scalar alpha, Scalar scale, Scalar input_scale, bool is_result, Tensor self_or_result) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<elu_backward::schema> create_elu_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(elu_backward::name, elu_backward::overload_name)
      .typed<elu_backward::schema>();
}

// aten::elu_backward(Tensor grad_output, Scalar alpha, Scalar scale, Scalar input_scale, bool is_result, Tensor self_or_result) -> Tensor
at::Tensor elu_backward::call(const at::Tensor & grad_output, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale, bool is_result, const at::Tensor & self_or_result) {
    static auto op = create_elu_backward_typed_handle();
    return op.call(grad_output, alpha, scale, input_scale, is_result, self_or_result);
}

// aten::elu_backward(Tensor grad_output, Scalar alpha, Scalar scale, Scalar input_scale, bool is_result, Tensor self_or_result) -> Tensor
at::Tensor elu_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale, bool is_result, const at::Tensor & self_or_result) {
    static auto op = create_elu_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, alpha, scale, input_scale, is_result, self_or_result);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(glu_out, name, "aten::glu")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(glu_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(glu_out, schema_str, "glu.out(Tensor self, int dim=-1, *, Tensor(a!) out) -> Tensor(a!)")

// aten::glu.out(Tensor self, int dim=-1, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<glu_out::schema> create_glu_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(glu_out::name, glu_out::overload_name)
      .typed<glu_out::schema>();
}

// aten::glu.out(Tensor self, int dim=-1, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & glu_out::call(const at::Tensor & self, int64_t dim, at::Tensor & out) {
    static auto op = create_glu_out_typed_handle();
    return op.call(self, dim, out);
}

// aten::glu.out(Tensor self, int dim=-1, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & glu_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, at::Tensor & out) {
    static auto op = create_glu_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(glu_backward_grad_input, name, "aten::glu_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(glu_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(glu_backward_grad_input, schema_str, "glu_backward.grad_input(Tensor grad_output, Tensor self, int dim, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::glu_backward.grad_input(Tensor grad_output, Tensor self, int dim, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<glu_backward_grad_input::schema> create_glu_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(glu_backward_grad_input::name, glu_backward_grad_input::overload_name)
      .typed<glu_backward_grad_input::schema>();
}

// aten::glu_backward.grad_input(Tensor grad_output, Tensor self, int dim, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & glu_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, int64_t dim, at::Tensor & grad_input) {
    static auto op = create_glu_backward_grad_input_typed_handle();
    return op.call(grad_output, self, dim, grad_input);
}

// aten::glu_backward.grad_input(Tensor grad_output, Tensor self, int dim, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & glu_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, int64_t dim, at::Tensor & grad_input) {
    static auto op = create_glu_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, dim, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardsigmoid_backward, name, "aten::hardsigmoid_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardsigmoid_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardsigmoid_backward, schema_str, "hardsigmoid_backward(Tensor grad_output, Tensor self) -> Tensor")

// aten::hardsigmoid_backward(Tensor grad_output, Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<hardsigmoid_backward::schema> create_hardsigmoid_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(hardsigmoid_backward::name, hardsigmoid_backward::overload_name)
      .typed<hardsigmoid_backward::schema>();
}

// aten::hardsigmoid_backward(Tensor grad_output, Tensor self) -> Tensor
at::Tensor hardsigmoid_backward::call(const at::Tensor & grad_output, const at::Tensor & self) {
    static auto op = create_hardsigmoid_backward_typed_handle();
    return op.call(grad_output, self);
}

// aten::hardsigmoid_backward(Tensor grad_output, Tensor self) -> Tensor
at::Tensor hardsigmoid_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self) {
    static auto op = create_hardsigmoid_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rrelu_with_noise_backward, name, "aten::rrelu_with_noise_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rrelu_with_noise_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rrelu_with_noise_backward, schema_str, "rrelu_with_noise_backward(Tensor grad_output, Tensor self, Tensor noise, Scalar lower, Scalar upper, bool training, bool self_is_result) -> Tensor")

// aten::rrelu_with_noise_backward(Tensor grad_output, Tensor self, Tensor noise, Scalar lower, Scalar upper, bool training, bool self_is_result) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<rrelu_with_noise_backward::schema> create_rrelu_with_noise_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(rrelu_with_noise_backward::name, rrelu_with_noise_backward::overload_name)
      .typed<rrelu_with_noise_backward::schema>();
}

// aten::rrelu_with_noise_backward(Tensor grad_output, Tensor self, Tensor noise, Scalar lower, Scalar upper, bool training, bool self_is_result) -> Tensor
at::Tensor rrelu_with_noise_backward::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & noise, const at::Scalar & lower, const at::Scalar & upper, bool training, bool self_is_result) {
    static auto op = create_rrelu_with_noise_backward_typed_handle();
    return op.call(grad_output, self, noise, lower, upper, training, self_is_result);
}

// aten::rrelu_with_noise_backward(Tensor grad_output, Tensor self, Tensor noise, Scalar lower, Scalar upper, bool training, bool self_is_result) -> Tensor
at::Tensor rrelu_with_noise_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & noise, const at::Scalar & lower, const at::Scalar & upper, bool training, bool self_is_result) {
    static auto op = create_rrelu_with_noise_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, noise, lower, upper, training, self_is_result);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(softplus_backward, name, "aten::softplus_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(softplus_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(softplus_backward, schema_str, "softplus_backward(Tensor grad_output, Tensor self, Scalar beta, Scalar threshold, Tensor output) -> Tensor")

// aten::softplus_backward(Tensor grad_output, Tensor self, Scalar beta, Scalar threshold, Tensor output) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<softplus_backward::schema> create_softplus_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(softplus_backward::name, softplus_backward::overload_name)
      .typed<softplus_backward::schema>();
}

// aten::softplus_backward(Tensor grad_output, Tensor self, Scalar beta, Scalar threshold, Tensor output) -> Tensor
at::Tensor softplus_backward::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold, const at::Tensor & output) {
    static auto op = create_softplus_backward_typed_handle();
    return op.call(grad_output, self, beta, threshold, output);
}

// aten::softplus_backward(Tensor grad_output, Tensor self, Scalar beta, Scalar threshold, Tensor output) -> Tensor
at::Tensor softplus_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold, const at::Tensor & output) {
    static auto op = create_softplus_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, beta, threshold, output);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_adaptive_avg_pool2d_backward, name, "aten::mkldnn_adaptive_avg_pool2d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_adaptive_avg_pool2d_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_adaptive_avg_pool2d_backward, schema_str, "mkldnn_adaptive_avg_pool2d_backward(Tensor grad_output, Tensor self) -> Tensor")

// aten::mkldnn_adaptive_avg_pool2d_backward(Tensor grad_output, Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<mkldnn_adaptive_avg_pool2d_backward::schema> create_mkldnn_adaptive_avg_pool2d_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mkldnn_adaptive_avg_pool2d_backward::name, mkldnn_adaptive_avg_pool2d_backward::overload_name)
      .typed<mkldnn_adaptive_avg_pool2d_backward::schema>();
}

// aten::mkldnn_adaptive_avg_pool2d_backward(Tensor grad_output, Tensor self) -> Tensor
at::Tensor mkldnn_adaptive_avg_pool2d_backward::call(const at::Tensor & grad_output, const at::Tensor & self) {
    static auto op = create_mkldnn_adaptive_avg_pool2d_backward_typed_handle();
    return op.call(grad_output, self);
}

// aten::mkldnn_adaptive_avg_pool2d_backward(Tensor grad_output, Tensor self) -> Tensor
at::Tensor mkldnn_adaptive_avg_pool2d_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self) {
    static auto op = create_mkldnn_adaptive_avg_pool2d_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_avg_pool3d_out, name, "aten::adaptive_avg_pool3d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_avg_pool3d_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_avg_pool3d_out, schema_str, "adaptive_avg_pool3d.out(Tensor self, int[3] output_size, *, Tensor(a!) out) -> Tensor(a!)")

// aten::adaptive_avg_pool3d.out(Tensor self, int[3] output_size, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<adaptive_avg_pool3d_out::schema> create_adaptive_avg_pool3d_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(adaptive_avg_pool3d_out::name, adaptive_avg_pool3d_out::overload_name)
      .typed<adaptive_avg_pool3d_out::schema>();
}

// aten::adaptive_avg_pool3d.out(Tensor self, int[3] output_size, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & adaptive_avg_pool3d_out::call(const at::Tensor & self, at::IntArrayRef output_size, at::Tensor & out) {
    static auto op = create_adaptive_avg_pool3d_out_typed_handle();
    return op.call(self, output_size, out);
}

// aten::adaptive_avg_pool3d.out(Tensor self, int[3] output_size, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & adaptive_avg_pool3d_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size, at::Tensor & out) {
    static auto op = create_adaptive_avg_pool3d_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, output_size, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_max_pool2d_backward_grad_input, name, "aten::adaptive_max_pool2d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_max_pool2d_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_max_pool2d_backward_grad_input, schema_str, "adaptive_max_pool2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::adaptive_max_pool2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<adaptive_max_pool2d_backward_grad_input::schema> create_adaptive_max_pool2d_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(adaptive_max_pool2d_backward_grad_input::name, adaptive_max_pool2d_backward_grad_input::overload_name)
      .typed<adaptive_max_pool2d_backward_grad_input::schema>();
}

// aten::adaptive_max_pool2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & adaptive_max_pool2d_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::Tensor & grad_input) {
    static auto op = create_adaptive_max_pool2d_backward_grad_input_typed_handle();
    return op.call(grad_output, self, indices, grad_input);
}

// aten::adaptive_max_pool2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & adaptive_max_pool2d_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::Tensor & grad_input) {
    static auto op = create_adaptive_max_pool2d_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, indices, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(avg_pool2d_out, name, "aten::avg_pool2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(avg_pool2d_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(avg_pool2d_out, schema_str, "avg_pool2d.out(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::avg_pool2d.out(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<avg_pool2d_out::schema> create_avg_pool2d_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(avg_pool2d_out::name, avg_pool2d_out::overload_name)
      .typed<avg_pool2d_out::schema>();
}

// aten::avg_pool2d.out(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & avg_pool2d_out::call(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, at::Tensor & out) {
    static auto op = create_avg_pool2d_out_typed_handle();
    return op.call(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, out);
}

// aten::avg_pool2d.out(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & avg_pool2d_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, at::Tensor & out) {
    static auto op = create_avg_pool2d_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(avg_pool3d_backward_grad_input, name, "aten::avg_pool3d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(avg_pool3d_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(avg_pool3d_backward_grad_input, schema_str, "avg_pool3d_backward.grad_input(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, bool ceil_mode, bool count_include_pad, int? divisor_override, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::avg_pool3d_backward.grad_input(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, bool ceil_mode, bool count_include_pad, int? divisor_override, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<avg_pool3d_backward_grad_input::schema> create_avg_pool3d_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(avg_pool3d_backward_grad_input::name, avg_pool3d_backward_grad_input::overload_name)
      .typed<avg_pool3d_backward_grad_input::schema>();
}

// aten::avg_pool3d_backward.grad_input(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, bool ceil_mode, bool count_include_pad, int? divisor_override, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & avg_pool3d_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, at::Tensor & grad_input) {
    static auto op = create_avg_pool3d_backward_grad_input_typed_handle();
    return op.call(grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, grad_input);
}

// aten::avg_pool3d_backward.grad_input(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, bool ceil_mode, bool count_include_pad, int? divisor_override, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & avg_pool3d_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, at::Tensor & grad_input) {
    static auto op = create_avg_pool3d_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fractional_max_pool3d_backward, name, "aten::fractional_max_pool3d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fractional_max_pool3d_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fractional_max_pool3d_backward, schema_str, "fractional_max_pool3d_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] output_size, Tensor indices) -> Tensor")

// aten::fractional_max_pool3d_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] output_size, Tensor indices) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fractional_max_pool3d_backward::schema> create_fractional_max_pool3d_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fractional_max_pool3d_backward::name, fractional_max_pool3d_backward::overload_name)
      .typed<fractional_max_pool3d_backward::schema>();
}

// aten::fractional_max_pool3d_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] output_size, Tensor indices) -> Tensor
at::Tensor fractional_max_pool3d_backward::call(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & indices) {
    static auto op = create_fractional_max_pool3d_backward_typed_handle();
    return op.call(grad_output, self, kernel_size, output_size, indices);
}

// aten::fractional_max_pool3d_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] output_size, Tensor indices) -> Tensor
at::Tensor fractional_max_pool3d_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & indices) {
    static auto op = create_fractional_max_pool3d_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, kernel_size, output_size, indices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_pool2d_with_indices, name, "aten::max_pool2d_with_indices")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_pool2d_with_indices, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_pool2d_with_indices, schema_str, "max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)")

// aten::max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<max_pool2d_with_indices::schema> create_max_pool2d_with_indices_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(max_pool2d_with_indices::name, max_pool2d_with_indices::overload_name)
      .typed<max_pool2d_with_indices::schema>();
}

// aten::max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> max_pool2d_with_indices::call(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
    static auto op = create_max_pool2d_with_indices_typed_handle();
    return op.call(self, kernel_size, stride, padding, dilation, ceil_mode);
}

// aten::max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> max_pool2d_with_indices::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
    static auto op = create_max_pool2d_with_indices_typed_handle();
    return op.redispatch(dispatchKeySet, self, kernel_size, stride, padding, dilation, ceil_mode);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_unpool2d_backward, name, "aten::max_unpool2d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_unpool2d_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_unpool2d_backward, schema_str, "max_unpool2d_backward(Tensor grad_output, Tensor self, Tensor indices, int[2] output_size) -> Tensor")

// aten::max_unpool2d_backward(Tensor grad_output, Tensor self, Tensor indices, int[2] output_size) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<max_unpool2d_backward::schema> create_max_unpool2d_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(max_unpool2d_backward::name, max_unpool2d_backward::overload_name)
      .typed<max_unpool2d_backward::schema>();
}

// aten::max_unpool2d_backward(Tensor grad_output, Tensor self, Tensor indices, int[2] output_size) -> Tensor
at::Tensor max_unpool2d_backward::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size) {
    static auto op = create_max_unpool2d_backward_typed_handle();
    return op.call(grad_output, self, indices, output_size);
}

// aten::max_unpool2d_backward(Tensor grad_output, Tensor self, Tensor indices, int[2] output_size) -> Tensor
at::Tensor max_unpool2d_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size) {
    static auto op = create_max_unpool2d_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, indices, output_size);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_unpool3d_out, name, "aten::max_unpool3d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_unpool3d_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_unpool3d_out, schema_str, "max_unpool3d.out(Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding, *, Tensor(a!) out) -> Tensor(a!)")

// aten::max_unpool3d.out(Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<max_unpool3d_out::schema> create_max_unpool3d_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(max_unpool3d_out::name, max_unpool3d_out::overload_name)
      .typed<max_unpool3d_out::schema>();
}

// aten::max_unpool3d.out(Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & max_unpool3d_out::call(const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & out) {
    static auto op = create_max_unpool3d_out_typed_handle();
    return op.call(self, indices, output_size, stride, padding, out);
}

// aten::max_unpool3d.out(Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & max_unpool3d_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & out) {
    static auto op = create_max_unpool3d_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, indices, output_size, stride, padding, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reflection_pad1d, name, "aten::reflection_pad1d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reflection_pad1d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reflection_pad1d, schema_str, "reflection_pad1d(Tensor self, int[2] padding) -> Tensor")

// aten::reflection_pad1d(Tensor self, int[2] padding) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<reflection_pad1d::schema> create_reflection_pad1d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(reflection_pad1d::name, reflection_pad1d::overload_name)
      .typed<reflection_pad1d::schema>();
}

// aten::reflection_pad1d(Tensor self, int[2] padding) -> Tensor
at::Tensor reflection_pad1d::call(const at::Tensor & self, at::IntArrayRef padding) {
    static auto op = create_reflection_pad1d_typed_handle();
    return op.call(self, padding);
}

// aten::reflection_pad1d(Tensor self, int[2] padding) -> Tensor
at::Tensor reflection_pad1d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef padding) {
    static auto op = create_reflection_pad1d_typed_handle();
    return op.redispatch(dispatchKeySet, self, padding);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reflection_pad2d_out, name, "aten::reflection_pad2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reflection_pad2d_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reflection_pad2d_out, schema_str, "reflection_pad2d.out(Tensor self, int[4] padding, *, Tensor(a!) out) -> Tensor(a!)")

// aten::reflection_pad2d.out(Tensor self, int[4] padding, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<reflection_pad2d_out::schema> create_reflection_pad2d_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(reflection_pad2d_out::name, reflection_pad2d_out::overload_name)
      .typed<reflection_pad2d_out::schema>();
}

// aten::reflection_pad2d.out(Tensor self, int[4] padding, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & reflection_pad2d_out::call(const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out) {
    static auto op = create_reflection_pad2d_out_typed_handle();
    return op.call(self, padding, out);
}

// aten::reflection_pad2d.out(Tensor self, int[4] padding, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & reflection_pad2d_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out) {
    static auto op = create_reflection_pad2d_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, padding, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(replication_pad1d_backward_grad_input, name, "aten::replication_pad1d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(replication_pad1d_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(replication_pad1d_backward_grad_input, schema_str, "replication_pad1d_backward.grad_input(Tensor grad_output, Tensor self, int[2] padding, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::replication_pad1d_backward.grad_input(Tensor grad_output, Tensor self, int[2] padding, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<replication_pad1d_backward_grad_input::schema> create_replication_pad1d_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(replication_pad1d_backward_grad_input::name, replication_pad1d_backward_grad_input::overload_name)
      .typed<replication_pad1d_backward_grad_input::schema>();
}

// aten::replication_pad1d_backward.grad_input(Tensor grad_output, Tensor self, int[2] padding, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & replication_pad1d_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input) {
    static auto op = create_replication_pad1d_backward_grad_input_typed_handle();
    return op.call(grad_output, self, padding, grad_input);
}

// aten::replication_pad1d_backward.grad_input(Tensor grad_output, Tensor self, int[2] padding, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & replication_pad1d_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input) {
    static auto op = create_replication_pad1d_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, padding, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(replication_pad2d_out, name, "aten::replication_pad2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(replication_pad2d_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(replication_pad2d_out, schema_str, "replication_pad2d.out(Tensor self, int[4] padding, *, Tensor(a!) out) -> Tensor(a!)")

// aten::replication_pad2d.out(Tensor self, int[4] padding, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<replication_pad2d_out::schema> create_replication_pad2d_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(replication_pad2d_out::name, replication_pad2d_out::overload_name)
      .typed<replication_pad2d_out::schema>();
}

// aten::replication_pad2d.out(Tensor self, int[4] padding, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & replication_pad2d_out::call(const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out) {
    static auto op = create_replication_pad2d_out_typed_handle();
    return op.call(self, padding, out);
}

// aten::replication_pad2d.out(Tensor self, int[4] padding, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & replication_pad2d_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out) {
    static auto op = create_replication_pad2d_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, padding, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(replication_pad2d_backward_grad_input, name, "aten::replication_pad2d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(replication_pad2d_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(replication_pad2d_backward_grad_input, schema_str, "replication_pad2d_backward.grad_input(Tensor grad_output, Tensor self, int[4] padding, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::replication_pad2d_backward.grad_input(Tensor grad_output, Tensor self, int[4] padding, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<replication_pad2d_backward_grad_input::schema> create_replication_pad2d_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(replication_pad2d_backward_grad_input::name, replication_pad2d_backward_grad_input::overload_name)
      .typed<replication_pad2d_backward_grad_input::schema>();
}

// aten::replication_pad2d_backward.grad_input(Tensor grad_output, Tensor self, int[4] padding, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & replication_pad2d_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input) {
    static auto op = create_replication_pad2d_backward_grad_input_typed_handle();
    return op.call(grad_output, self, padding, grad_input);
}

// aten::replication_pad2d_backward.grad_input(Tensor grad_output, Tensor self, int[4] padding, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & replication_pad2d_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input) {
    static auto op = create_replication_pad2d_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, padding, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest2d_vec, name, "aten::upsample_nearest2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest2d_vec, overload_name, "vec")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest2d_vec, schema_str, "upsample_nearest2d.vec(Tensor input, int[]? output_size, float[]? scale_factors) -> Tensor")

// aten::upsample_nearest2d.vec(Tensor input, int[]? output_size, float[]? scale_factors) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<upsample_nearest2d_vec::schema> create_upsample_nearest2d_vec_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_nearest2d_vec::name, upsample_nearest2d_vec::overload_name)
      .typed<upsample_nearest2d_vec::schema>();
}

// aten::upsample_nearest2d.vec(Tensor input, int[]? output_size, float[]? scale_factors) -> Tensor
at::Tensor upsample_nearest2d_vec::call(const at::Tensor & input, c10::optional<at::IntArrayRef> output_size, c10::optional<at::ArrayRef<double>> scale_factors) {
    static auto op = create_upsample_nearest2d_vec_typed_handle();
    return op.call(input, output_size, scale_factors);
}

// aten::upsample_nearest2d.vec(Tensor input, int[]? output_size, float[]? scale_factors) -> Tensor
at::Tensor upsample_nearest2d_vec::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, c10::optional<at::IntArrayRef> output_size, c10::optional<at::ArrayRef<double>> scale_factors) {
    static auto op = create_upsample_nearest2d_vec_typed_handle();
    return op.redispatch(dispatchKeySet, input, output_size, scale_factors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_bilinear2d_out, name, "aten::upsample_bilinear2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_bilinear2d_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_bilinear2d_out, schema_str, "upsample_bilinear2d.out(Tensor self, int[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::upsample_bilinear2d.out(Tensor self, int[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<upsample_bilinear2d_out::schema> create_upsample_bilinear2d_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_bilinear2d_out::name, upsample_bilinear2d_out::overload_name)
      .typed<upsample_bilinear2d_out::schema>();
}

// aten::upsample_bilinear2d.out(Tensor self, int[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & upsample_bilinear2d_out::call(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out) {
    static auto op = create_upsample_bilinear2d_out_typed_handle();
    return op.call(self, output_size, align_corners, scales_h, scales_w, out);
}

// aten::upsample_bilinear2d.out(Tensor self, int[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & upsample_bilinear2d_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out) {
    static auto op = create_upsample_bilinear2d_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, output_size, align_corners, scales_h, scales_w, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_trilinear3d, name, "aten::upsample_trilinear3d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_trilinear3d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_trilinear3d, schema_str, "upsample_trilinear3d(Tensor self, int[3] output_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor")

// aten::upsample_trilinear3d(Tensor self, int[3] output_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<upsample_trilinear3d::schema> create_upsample_trilinear3d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_trilinear3d::name, upsample_trilinear3d::overload_name)
      .typed<upsample_trilinear3d::schema>();
}

// aten::upsample_trilinear3d(Tensor self, int[3] output_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor
at::Tensor upsample_trilinear3d::call(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
    static auto op = create_upsample_trilinear3d_typed_handle();
    return op.call(self, output_size, align_corners, scales_d, scales_h, scales_w);
}

// aten::upsample_trilinear3d(Tensor self, int[3] output_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor
at::Tensor upsample_trilinear3d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
    static auto op = create_upsample_trilinear3d_typed_handle();
    return op.redispatch(dispatchKeySet, self, output_size, align_corners, scales_d, scales_h, scales_w);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest3d, name, "aten::upsample_nearest3d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest3d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest3d, schema_str, "upsample_nearest3d(Tensor self, int[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor")

// aten::upsample_nearest3d(Tensor self, int[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<upsample_nearest3d::schema> create_upsample_nearest3d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_nearest3d::name, upsample_nearest3d::overload_name)
      .typed<upsample_nearest3d::schema>();
}

// aten::upsample_nearest3d(Tensor self, int[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor
at::Tensor upsample_nearest3d::call(const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
    static auto op = create_upsample_nearest3d_typed_handle();
    return op.call(self, output_size, scales_d, scales_h, scales_w);
}

// aten::upsample_nearest3d(Tensor self, int[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor
at::Tensor upsample_nearest3d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
    static auto op = create_upsample_nearest3d_typed_handle();
    return op.redispatch(dispatchKeySet, self, output_size, scales_d, scales_h, scales_w);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sigmoid_backward, name, "aten::sigmoid_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sigmoid_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sigmoid_backward, schema_str, "sigmoid_backward(Tensor grad_output, Tensor output) -> Tensor")

// aten::sigmoid_backward(Tensor grad_output, Tensor output) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<sigmoid_backward::schema> create_sigmoid_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sigmoid_backward::name, sigmoid_backward::overload_name)
      .typed<sigmoid_backward::schema>();
}

// aten::sigmoid_backward(Tensor grad_output, Tensor output) -> Tensor
at::Tensor sigmoid_backward::call(const at::Tensor & grad_output, const at::Tensor & output) {
    static auto op = create_sigmoid_backward_typed_handle();
    return op.call(grad_output, output);
}

// aten::sigmoid_backward(Tensor grad_output, Tensor output) -> Tensor
at::Tensor sigmoid_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & output) {
    static auto op = create_sigmoid_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, output);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tanh_backward, name, "aten::tanh_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tanh_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tanh_backward, schema_str, "tanh_backward(Tensor grad_output, Tensor output) -> Tensor")

// aten::tanh_backward(Tensor grad_output, Tensor output) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<tanh_backward::schema> create_tanh_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(tanh_backward::name, tanh_backward::overload_name)
      .typed<tanh_backward::schema>();
}

// aten::tanh_backward(Tensor grad_output, Tensor output) -> Tensor
at::Tensor tanh_backward::call(const at::Tensor & grad_output, const at::Tensor & output) {
    static auto op = create_tanh_backward_typed_handle();
    return op.call(grad_output, output);
}

// aten::tanh_backward(Tensor grad_output, Tensor output) -> Tensor
at::Tensor tanh_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & output) {
    static auto op = create_tanh_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, output);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(thnn_conv2d, name, "aten::thnn_conv2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(thnn_conv2d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(thnn_conv2d, schema_str, "thnn_conv2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0) -> Tensor")

// aten::thnn_conv2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<thnn_conv2d::schema> create_thnn_conv2d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(thnn_conv2d::name, thnn_conv2d::overload_name)
      .typed<thnn_conv2d::schema>();
}

// aten::thnn_conv2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0) -> Tensor
at::Tensor thnn_conv2d::call(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding) {
    static auto op = create_thnn_conv2d_typed_handle();
    return op.call(self, weight, kernel_size, bias, stride, padding);
}

// aten::thnn_conv2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0) -> Tensor
at::Tensor thnn_conv2d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding) {
    static auto op = create_thnn_conv2d_typed_handle();
    return op.redispatch(dispatchKeySet, self, weight, kernel_size, bias, stride, padding);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(thnn_conv2d_backward_grad_input, name, "aten::thnn_conv2d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(thnn_conv2d_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(thnn_conv2d_backward_grad_input, schema_str, "thnn_conv2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, Tensor finput, Tensor fgrad_input, *, Tensor(a!) grad_input, Tensor(b!) grad_weight, Tensor(c!) grad_bias) -> (Tensor(a!), Tensor(b!), Tensor(c!))")

// aten::thnn_conv2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, Tensor finput, Tensor fgrad_input, *, Tensor(a!) grad_input, Tensor(b!) grad_weight, Tensor(c!) grad_bias) -> (Tensor(a!), Tensor(b!), Tensor(c!))
static C10_NOINLINE c10::TypedOperatorHandle<thnn_conv2d_backward_grad_input::schema> create_thnn_conv2d_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(thnn_conv2d_backward_grad_input::name, thnn_conv2d_backward_grad_input::overload_name)
      .typed<thnn_conv2d_backward_grad_input::schema>();
}

// aten::thnn_conv2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, Tensor finput, Tensor fgrad_input, *, Tensor(a!) grad_input, Tensor(b!) grad_weight, Tensor(c!) grad_bias) -> (Tensor(a!), Tensor(b!), Tensor(c!))
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> thnn_conv2d_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, const at::Tensor & finput, const at::Tensor & fgrad_input, at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias) {
    static auto op = create_thnn_conv2d_backward_grad_input_typed_handle();
    return op.call(grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, grad_input, grad_weight, grad_bias);
}

// aten::thnn_conv2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, Tensor finput, Tensor fgrad_input, *, Tensor(a!) grad_input, Tensor(b!) grad_weight, Tensor(c!) grad_bias) -> (Tensor(a!), Tensor(b!), Tensor(c!))
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> thnn_conv2d_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, const at::Tensor & finput, const at::Tensor & fgrad_input, at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias) {
    static auto op = create_thnn_conv2d_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, grad_input, grad_weight, grad_bias);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_conv_depthwise2d_out, name, "aten::_conv_depthwise2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_conv_depthwise2d_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_conv_depthwise2d_out, schema_str, "_conv_depthwise2d.out(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding, int[2] dilation, *, Tensor(a!) out) -> Tensor(a!)")

// aten::_conv_depthwise2d.out(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding, int[2] dilation, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<_conv_depthwise2d_out::schema> create__conv_depthwise2d_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_conv_depthwise2d_out::name, _conv_depthwise2d_out::overload_name)
      .typed<_conv_depthwise2d_out::schema>();
}

// aten::_conv_depthwise2d.out(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding, int[2] dilation, *, Tensor(a!) out) -> Tensor(a!)
const at::Tensor & _conv_depthwise2d_out::call(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, const at::Tensor & out) {
    static auto op = create__conv_depthwise2d_out_typed_handle();
    return op.call(self, weight, kernel_size, bias, stride, padding, dilation, out);
}

// aten::_conv_depthwise2d.out(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding, int[2] dilation, *, Tensor(a!) out) -> Tensor(a!)
const at::Tensor & _conv_depthwise2d_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, const at::Tensor & out) {
    static auto op = create__conv_depthwise2d_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, weight, kernel_size, bias, stride, padding, dilation, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_conv_depthwise2d_backward_output_mask, name, "aten::_conv_depthwise2d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_conv_depthwise2d_backward_output_mask, overload_name, "output_mask")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_conv_depthwise2d_backward_output_mask, schema_str, "_conv_depthwise2d_backward.output_mask(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool[2] output_mask) -> (Tensor grad_input, Tensor grad_weight)")

// aten::_conv_depthwise2d_backward.output_mask(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool[2] output_mask) -> (Tensor grad_input, Tensor grad_weight)
static C10_NOINLINE c10::TypedOperatorHandle<_conv_depthwise2d_backward_output_mask::schema> create__conv_depthwise2d_backward_output_mask_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_conv_depthwise2d_backward_output_mask::name, _conv_depthwise2d_backward_output_mask::overload_name)
      .typed<_conv_depthwise2d_backward_output_mask::schema>();
}

// aten::_conv_depthwise2d_backward.output_mask(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool[2] output_mask) -> (Tensor grad_input, Tensor grad_weight)
::std::tuple<at::Tensor,at::Tensor> _conv_depthwise2d_backward_output_mask::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, ::std::array<bool,2> output_mask) {
    static auto op = create__conv_depthwise2d_backward_output_mask_typed_handle();
    return op.call(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
}

// aten::_conv_depthwise2d_backward.output_mask(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool[2] output_mask) -> (Tensor grad_input, Tensor grad_weight)
::std::tuple<at::Tensor,at::Tensor> _conv_depthwise2d_backward_output_mask::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, ::std::array<bool,2> output_mask) {
    static auto op = create__conv_depthwise2d_backward_output_mask_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(column_stack, name, "aten::column_stack")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(column_stack, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(column_stack, schema_str, "column_stack(Tensor[] tensors) -> Tensor")

// aten::column_stack(Tensor[] tensors) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<column_stack::schema> create_column_stack_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(column_stack::name, column_stack::overload_name)
      .typed<column_stack::schema>();
}

// aten::column_stack(Tensor[] tensors) -> Tensor
at::Tensor column_stack::call(at::TensorList tensors) {
    static auto op = create_column_stack_typed_handle();
    return op.call(tensors);
}

// aten::column_stack(Tensor[] tensors) -> Tensor
at::Tensor column_stack::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create_column_stack_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_entr, name, "aten::special_entr")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_entr, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_entr, schema_str, "special_entr(Tensor self) -> Tensor")

// aten::special_entr(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<special_entr::schema> create_special_entr_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_entr::name, special_entr::overload_name)
      .typed<special_entr::schema>();
}

// aten::special_entr(Tensor self) -> Tensor
at::Tensor special_entr::call(const at::Tensor & self) {
    static auto op = create_special_entr_typed_handle();
    return op.call(self);
}

// aten::special_entr(Tensor self) -> Tensor
at::Tensor special_entr::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_special_entr_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_ndtri, name, "aten::special_ndtri")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_ndtri, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_ndtri, schema_str, "special_ndtri(Tensor self) -> Tensor")

// aten::special_ndtri(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<special_ndtri::schema> create_special_ndtri_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_ndtri::name, special_ndtri::overload_name)
      .typed<special_ndtri::schema>();
}

// aten::special_ndtri(Tensor self) -> Tensor
at::Tensor special_ndtri::call(const at::Tensor & self) {
    static auto op = create_special_ndtri_typed_handle();
    return op.call(self);
}

// aten::special_ndtri(Tensor self) -> Tensor
at::Tensor special_ndtri::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_special_ndtri_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_digamma_out, name, "aten::special_digamma")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_digamma_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_digamma_out, schema_str, "special_digamma.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::special_digamma.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<special_digamma_out::schema> create_special_digamma_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_digamma_out::name, special_digamma_out::overload_name)
      .typed<special_digamma_out::schema>();
}

// aten::special_digamma.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_digamma_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_digamma_out_typed_handle();
    return op.call(self, out);
}

// aten::special_digamma.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_digamma_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_digamma_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_erfc, name, "aten::special_erfc")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_erfc, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_erfc, schema_str, "special_erfc(Tensor self) -> Tensor")

// aten::special_erfc(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<special_erfc::schema> create_special_erfc_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_erfc::name, special_erfc::overload_name)
      .typed<special_erfc::schema>();
}

// aten::special_erfc(Tensor self) -> Tensor
at::Tensor special_erfc::call(const at::Tensor & self) {
    static auto op = create_special_erfc_typed_handle();
    return op.call(self);
}

// aten::special_erfc(Tensor self) -> Tensor
at::Tensor special_erfc::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_special_erfc_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_erfinv_out, name, "aten::special_erfinv")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_erfinv_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_erfinv_out, schema_str, "special_erfinv.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::special_erfinv.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<special_erfinv_out::schema> create_special_erfinv_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_erfinv_out::name, special_erfinv_out::overload_name)
      .typed<special_erfinv_out::schema>();
}

// aten::special_erfinv.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_erfinv_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_erfinv_out_typed_handle();
    return op.call(self, out);
}

// aten::special_erfinv.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_erfinv_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_erfinv_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_xlog1py_self_scalar, name, "aten::special_xlog1py")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_xlog1py_self_scalar, overload_name, "self_scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_xlog1py_self_scalar, schema_str, "special_xlog1py.self_scalar(Scalar self, Tensor other) -> Tensor")

// aten::special_xlog1py.self_scalar(Scalar self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<special_xlog1py_self_scalar::schema> create_special_xlog1py_self_scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_xlog1py_self_scalar::name, special_xlog1py_self_scalar::overload_name)
      .typed<special_xlog1py_self_scalar::schema>();
}

// aten::special_xlog1py.self_scalar(Scalar self, Tensor other) -> Tensor
at::Tensor special_xlog1py_self_scalar::call(const at::Scalar & self, const at::Tensor & other) {
    static auto op = create_special_xlog1py_self_scalar_typed_handle();
    return op.call(self, other);
}

// aten::special_xlog1py.self_scalar(Scalar self, Tensor other) -> Tensor
at::Tensor special_xlog1py_self_scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Scalar & self, const at::Tensor & other) {
    static auto op = create_special_xlog1py_self_scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_xlogy_other_scalar, name, "aten::special_xlogy")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_xlogy_other_scalar, overload_name, "other_scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_xlogy_other_scalar, schema_str, "special_xlogy.other_scalar(Tensor self, Scalar other) -> Tensor")

// aten::special_xlogy.other_scalar(Tensor self, Scalar other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<special_xlogy_other_scalar::schema> create_special_xlogy_other_scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_xlogy_other_scalar::name, special_xlogy_other_scalar::overload_name)
      .typed<special_xlogy_other_scalar::schema>();
}

// aten::special_xlogy.other_scalar(Tensor self, Scalar other) -> Tensor
at::Tensor special_xlogy_other_scalar::call(const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_special_xlogy_other_scalar_typed_handle();
    return op.call(self, other);
}

// aten::special_xlogy.other_scalar(Tensor self, Scalar other) -> Tensor
at::Tensor special_xlogy_other_scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_special_xlogy_other_scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_xlogy_self_scalar_out, name, "aten::special_xlogy")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_xlogy_self_scalar_out, overload_name, "self_scalar_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_xlogy_self_scalar_out, schema_str, "special_xlogy.self_scalar_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::special_xlogy.self_scalar_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<special_xlogy_self_scalar_out::schema> create_special_xlogy_self_scalar_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_xlogy_self_scalar_out::name, special_xlogy_self_scalar_out::overload_name)
      .typed<special_xlogy_self_scalar_out::schema>();
}

// aten::special_xlogy.self_scalar_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_xlogy_self_scalar_out::call(const at::Scalar & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_special_xlogy_self_scalar_out_typed_handle();
    return op.call(self, other, out);
}

// aten::special_xlogy.self_scalar_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_xlogy_self_scalar_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Scalar & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_special_xlogy_self_scalar_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_i0e_out, name, "aten::special_i0e")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_i0e_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_i0e_out, schema_str, "special_i0e.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::special_i0e.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<special_i0e_out::schema> create_special_i0e_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_i0e_out::name, special_i0e_out::overload_name)
      .typed<special_i0e_out::schema>();
}

// aten::special_i0e.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_i0e_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_i0e_out_typed_handle();
    return op.call(self, out);
}

// aten::special_i0e.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_i0e_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_i0e_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_i1e, name, "aten::special_i1e")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_i1e, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_i1e, schema_str, "special_i1e(Tensor self) -> Tensor")

// aten::special_i1e(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<special_i1e::schema> create_special_i1e_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_i1e::name, special_i1e::overload_name)
      .typed<special_i1e::schema>();
}

// aten::special_i1e(Tensor self) -> Tensor
at::Tensor special_i1e::call(const at::Tensor & self) {
    static auto op = create_special_i1e_typed_handle();
    return op.call(self);
}

// aten::special_i1e(Tensor self) -> Tensor
at::Tensor special_i1e::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_special_i1e_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_polygamma_out, name, "aten::special_polygamma")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_polygamma_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_polygamma_out, schema_str, "special_polygamma.out(int n, Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::special_polygamma.out(int n, Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<special_polygamma_out::schema> create_special_polygamma_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_polygamma_out::name, special_polygamma_out::overload_name)
      .typed<special_polygamma_out::schema>();
}

// aten::special_polygamma.out(int n, Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_polygamma_out::call(int64_t n, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_polygamma_out_typed_handle();
    return op.call(n, self, out);
}

// aten::special_polygamma.out(int n, Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_polygamma_out::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t n, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_polygamma_out_typed_handle();
    return op.redispatch(dispatchKeySet, n, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_logsumexp, name, "aten::special_logsumexp")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_logsumexp, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_logsumexp, schema_str, "special_logsumexp(Tensor self, int[1] dim, bool keepdim=False) -> Tensor")

// aten::special_logsumexp(Tensor self, int[1] dim, bool keepdim=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<special_logsumexp::schema> create_special_logsumexp_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_logsumexp::name, special_logsumexp::overload_name)
      .typed<special_logsumexp::schema>();
}

// aten::special_logsumexp(Tensor self, int[1] dim, bool keepdim=False) -> Tensor
at::Tensor special_logsumexp::call(const at::Tensor & self, at::IntArrayRef dim, bool keepdim) {
    static auto op = create_special_logsumexp_typed_handle();
    return op.call(self, dim, keepdim);
}

// aten::special_logsumexp(Tensor self, int[1] dim, bool keepdim=False) -> Tensor
at::Tensor special_logsumexp::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool keepdim) {
    static auto op = create_special_logsumexp_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_expit_out, name, "aten::special_expit")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_expit_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_expit_out, schema_str, "special_expit.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::special_expit.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<special_expit_out::schema> create_special_expit_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_expit_out::name, special_expit_out::overload_name)
      .typed<special_expit_out::schema>();
}

// aten::special_expit.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_expit_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_expit_out_typed_handle();
    return op.call(self, out);
}

// aten::special_expit.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_expit_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_expit_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_gammainc, name, "aten::special_gammainc")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_gammainc, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_gammainc, schema_str, "special_gammainc(Tensor self, Tensor other) -> Tensor")

// aten::special_gammainc(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<special_gammainc::schema> create_special_gammainc_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_gammainc::name, special_gammainc::overload_name)
      .typed<special_gammainc::schema>();
}

// aten::special_gammainc(Tensor self, Tensor other) -> Tensor
at::Tensor special_gammainc::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_special_gammainc_typed_handle();
    return op.call(self, other);
}

// aten::special_gammainc(Tensor self, Tensor other) -> Tensor
at::Tensor special_gammainc::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_special_gammainc_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_hfft_out, name, "aten::fft_hfft")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_hfft_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_hfft_out, schema_str, "fft_hfft.out(Tensor self, int? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::fft_hfft.out(Tensor self, int? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<fft_hfft_out::schema> create_fft_hfft_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fft_hfft_out::name, fft_hfft_out::overload_name)
      .typed<fft_hfft_out::schema>();
}

// aten::fft_hfft.out(Tensor self, int? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fft_hfft_out::call(const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
    static auto op = create_fft_hfft_out_typed_handle();
    return op.call(self, n, dim, norm, out);
}

// aten::fft_hfft.out(Tensor self, int? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fft_hfft_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
    static auto op = create_fft_hfft_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, n, dim, norm, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_rfft2, name, "aten::fft_rfft2")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_rfft2, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_rfft2, schema_str, "fft_rfft2(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None) -> Tensor")

// aten::fft_rfft2(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fft_rfft2::schema> create_fft_rfft2_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fft_rfft2::name, fft_rfft2::overload_name)
      .typed<fft_rfft2::schema>();
}

// aten::fft_rfft2(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None) -> Tensor
at::Tensor fft_rfft2::call(const at::Tensor & self, c10::optional<at::IntArrayRef> s, at::IntArrayRef dim, c10::optional<c10::string_view> norm) {
    static auto op = create_fft_rfft2_typed_handle();
    return op.call(self, s, dim, norm);
}

// aten::fft_rfft2(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None) -> Tensor
at::Tensor fft_rfft2::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> s, at::IntArrayRef dim, c10::optional<c10::string_view> norm) {
    static auto op = create_fft_rfft2_typed_handle();
    return op.redispatch(dispatchKeySet, self, s, dim, norm);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_lstsq, name, "aten::linalg_lstsq")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_lstsq, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_lstsq, schema_str, "linalg_lstsq(Tensor self, Tensor b, float? rcond=None, *, str? driver=None) -> (Tensor solution, Tensor residuals, Tensor rank, Tensor singular_values)")

// aten::linalg_lstsq(Tensor self, Tensor b, float? rcond=None, *, str? driver=None) -> (Tensor solution, Tensor residuals, Tensor rank, Tensor singular_values)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_lstsq::schema> create_linalg_lstsq_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_lstsq::name, linalg_lstsq::overload_name)
      .typed<linalg_lstsq::schema>();
}

// aten::linalg_lstsq(Tensor self, Tensor b, float? rcond=None, *, str? driver=None) -> (Tensor solution, Tensor residuals, Tensor rank, Tensor singular_values)
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> linalg_lstsq::call(const at::Tensor & self, const at::Tensor & b, c10::optional<double> rcond, c10::optional<c10::string_view> driver) {
    static auto op = create_linalg_lstsq_typed_handle();
    return op.call(self, b, rcond, driver);
}

// aten::linalg_lstsq(Tensor self, Tensor b, float? rcond=None, *, str? driver=None) -> (Tensor solution, Tensor residuals, Tensor rank, Tensor singular_values)
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> linalg_lstsq::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & b, c10::optional<double> rcond, c10::optional<c10::string_view> driver) {
    static auto op = create_linalg_lstsq_typed_handle();
    return op.redispatch(dispatchKeySet, self, b, rcond, driver);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_lstsq_out, name, "aten::linalg_lstsq")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_lstsq_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_lstsq_out, schema_str, "linalg_lstsq.out(Tensor self, Tensor b, float? rcond=None, *, str? driver=None, Tensor(a!) solution, Tensor(b!) residuals, Tensor(c!) rank, Tensor(d!) singular_values) -> (Tensor(a!) solution, Tensor(b!) residuals, Tensor(c!) rank, Tensor(d!) singular_values)")

// aten::linalg_lstsq.out(Tensor self, Tensor b, float? rcond=None, *, str? driver=None, Tensor(a!) solution, Tensor(b!) residuals, Tensor(c!) rank, Tensor(d!) singular_values) -> (Tensor(a!) solution, Tensor(b!) residuals, Tensor(c!) rank, Tensor(d!) singular_values)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_lstsq_out::schema> create_linalg_lstsq_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_lstsq_out::name, linalg_lstsq_out::overload_name)
      .typed<linalg_lstsq_out::schema>();
}

// aten::linalg_lstsq.out(Tensor self, Tensor b, float? rcond=None, *, str? driver=None, Tensor(a!) solution, Tensor(b!) residuals, Tensor(c!) rank, Tensor(d!) singular_values) -> (Tensor(a!) solution, Tensor(b!) residuals, Tensor(c!) rank, Tensor(d!) singular_values)
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> linalg_lstsq_out::call(const at::Tensor & self, const at::Tensor & b, c10::optional<double> rcond, c10::optional<c10::string_view> driver, at::Tensor & solution, at::Tensor & residuals, at::Tensor & rank, at::Tensor & singular_values) {
    static auto op = create_linalg_lstsq_out_typed_handle();
    return op.call(self, b, rcond, driver, solution, residuals, rank, singular_values);
}

// aten::linalg_lstsq.out(Tensor self, Tensor b, float? rcond=None, *, str? driver=None, Tensor(a!) solution, Tensor(b!) residuals, Tensor(c!) rank, Tensor(d!) singular_values) -> (Tensor(a!) solution, Tensor(b!) residuals, Tensor(c!) rank, Tensor(d!) singular_values)
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> linalg_lstsq_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & b, c10::optional<double> rcond, c10::optional<c10::string_view> driver, at::Tensor & solution, at::Tensor & residuals, at::Tensor & rank, at::Tensor & singular_values) {
    static auto op = create_linalg_lstsq_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, b, rcond, driver, solution, residuals, rank, singular_values);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_matmul_out, name, "aten::linalg_matmul")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_matmul_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_matmul_out, schema_str, "linalg_matmul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::linalg_matmul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_matmul_out::schema> create_linalg_matmul_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_matmul_out::name, linalg_matmul_out::overload_name)
      .typed<linalg_matmul_out::schema>();
}

// aten::linalg_matmul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_matmul_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_linalg_matmul_out_typed_handle();
    return op.call(self, other, out);
}

// aten::linalg_matmul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_matmul_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_linalg_matmul_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_slogdet_out, name, "aten::linalg_slogdet")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_slogdet_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_slogdet_out, schema_str, "linalg_slogdet.out(Tensor self, *, Tensor(a!) sign, Tensor(b!) logabsdet) -> (Tensor(a!) sign, Tensor(b!) logabsdet)")

// aten::linalg_slogdet.out(Tensor self, *, Tensor(a!) sign, Tensor(b!) logabsdet) -> (Tensor(a!) sign, Tensor(b!) logabsdet)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_slogdet_out::schema> create_linalg_slogdet_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_slogdet_out::name, linalg_slogdet_out::overload_name)
      .typed<linalg_slogdet_out::schema>();
}

// aten::linalg_slogdet.out(Tensor self, *, Tensor(a!) sign, Tensor(b!) logabsdet) -> (Tensor(a!) sign, Tensor(b!) logabsdet)
::std::tuple<at::Tensor &,at::Tensor &> linalg_slogdet_out::call(const at::Tensor & self, at::Tensor & sign, at::Tensor & logabsdet) {
    static auto op = create_linalg_slogdet_out_typed_handle();
    return op.call(self, sign, logabsdet);
}

// aten::linalg_slogdet.out(Tensor self, *, Tensor(a!) sign, Tensor(b!) logabsdet) -> (Tensor(a!) sign, Tensor(b!) logabsdet)
::std::tuple<at::Tensor &,at::Tensor &> linalg_slogdet_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & sign, at::Tensor & logabsdet) {
    static auto op = create_linalg_slogdet_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, sign, logabsdet);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_eigvals_out, name, "aten::linalg_eigvals")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_eigvals_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_eigvals_out, schema_str, "linalg_eigvals.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::linalg_eigvals.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_eigvals_out::schema> create_linalg_eigvals_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_eigvals_out::name, linalg_eigvals_out::overload_name)
      .typed<linalg_eigvals_out::schema>();
}

// aten::linalg_eigvals.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_eigvals_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_linalg_eigvals_out_typed_handle();
    return op.call(self, out);
}

// aten::linalg_eigvals.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_eigvals_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_linalg_eigvals_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(outer_out, name, "aten::outer")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(outer_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(outer_out, schema_str, "outer.out(Tensor self, Tensor vec2, *, Tensor(a!) out) -> Tensor(a!)")

// aten::outer.out(Tensor self, Tensor vec2, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<outer_out::schema> create_outer_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(outer_out::name, outer_out::overload_name)
      .typed<outer_out::schema>();
}

// aten::outer.out(Tensor self, Tensor vec2, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & outer_out::call(const at::Tensor & self, const at::Tensor & vec2, at::Tensor & out) {
    static auto op = create_outer_out_typed_handle();
    return op.call(self, vec2, out);
}

// aten::outer.out(Tensor self, Tensor vec2, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & outer_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & vec2, at::Tensor & out) {
    static auto op = create_outer_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, vec2, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_norm, name, "aten::linalg_norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_norm, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_norm, schema_str, "linalg_norm(Tensor self, Scalar? ord=None, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor")

// aten::linalg_norm(Tensor self, Scalar? ord=None, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<linalg_norm::schema> create_linalg_norm_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_norm::name, linalg_norm::overload_name)
      .typed<linalg_norm::schema>();
}

// aten::linalg_norm(Tensor self, Scalar? ord=None, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
at::Tensor linalg_norm::call(const at::Tensor & self, const c10::optional<at::Scalar> & ord, c10::optional<at::IntArrayRef> dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_linalg_norm_typed_handle();
    return op.call(self, ord, dim, keepdim, dtype);
}

// aten::linalg_norm(Tensor self, Scalar? ord=None, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
at::Tensor linalg_norm::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Scalar> & ord, c10::optional<at::IntArrayRef> dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_linalg_norm_typed_handle();
    return op.redispatch(dispatchKeySet, self, ord, dim, keepdim, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_svdvals, name, "aten::linalg_svdvals")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_svdvals, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_svdvals, schema_str, "linalg_svdvals(Tensor input) -> Tensor")

// aten::linalg_svdvals(Tensor input) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<linalg_svdvals::schema> create_linalg_svdvals_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_svdvals::name, linalg_svdvals::overload_name)
      .typed<linalg_svdvals::schema>();
}

// aten::linalg_svdvals(Tensor input) -> Tensor
at::Tensor linalg_svdvals::call(const at::Tensor & input) {
    static auto op = create_linalg_svdvals_typed_handle();
    return op.call(input);
}

// aten::linalg_svdvals(Tensor input) -> Tensor
at::Tensor linalg_svdvals::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input) {
    static auto op = create_linalg_svdvals_typed_handle();
    return op.redispatch(dispatchKeySet, input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_solve_out, name, "aten::linalg_solve")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_solve_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_solve_out, schema_str, "linalg_solve.out(Tensor input, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::linalg_solve.out(Tensor input, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_solve_out::schema> create_linalg_solve_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_solve_out::name, linalg_solve_out::overload_name)
      .typed<linalg_solve_out::schema>();
}

// aten::linalg_solve.out(Tensor input, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_solve_out::call(const at::Tensor & input, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_linalg_solve_out_typed_handle();
    return op.call(input, other, out);
}

// aten::linalg_solve.out(Tensor input, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_solve_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_linalg_solve_out_typed_handle();
    return op.redispatch(dispatchKeySet, input, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_tensorsolve_out, name, "aten::linalg_tensorsolve")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_tensorsolve_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_tensorsolve_out, schema_str, "linalg_tensorsolve.out(Tensor self, Tensor other, int[]? dims=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::linalg_tensorsolve.out(Tensor self, Tensor other, int[]? dims=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_tensorsolve_out::schema> create_linalg_tensorsolve_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_tensorsolve_out::name, linalg_tensorsolve_out::overload_name)
      .typed<linalg_tensorsolve_out::schema>();
}

// aten::linalg_tensorsolve.out(Tensor self, Tensor other, int[]? dims=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_tensorsolve_out::call(const at::Tensor & self, const at::Tensor & other, c10::optional<at::IntArrayRef> dims, at::Tensor & out) {
    static auto op = create_linalg_tensorsolve_out_typed_handle();
    return op.call(self, other, dims, out);
}

// aten::linalg_tensorsolve.out(Tensor self, Tensor other, int[]? dims=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_tensorsolve_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, c10::optional<at::IntArrayRef> dims, at::Tensor & out) {
    static auto op = create_linalg_tensorsolve_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, dims, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_matrix_power, name, "aten::linalg_matrix_power")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_matrix_power, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_matrix_power, schema_str, "linalg_matrix_power(Tensor self, int n) -> Tensor")

// aten::linalg_matrix_power(Tensor self, int n) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<linalg_matrix_power::schema> create_linalg_matrix_power_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_matrix_power::name, linalg_matrix_power::overload_name)
      .typed<linalg_matrix_power::schema>();
}

// aten::linalg_matrix_power(Tensor self, int n) -> Tensor
at::Tensor linalg_matrix_power::call(const at::Tensor & self, int64_t n) {
    static auto op = create_linalg_matrix_power_typed_handle();
    return op.call(self, n);
}

// aten::linalg_matrix_power(Tensor self, int n) -> Tensor
at::Tensor linalg_matrix_power::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t n) {
    static auto op = create_linalg_matrix_power_typed_handle();
    return op.redispatch(dispatchKeySet, self, n);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_matrix_power_out, name, "aten::linalg_matrix_power")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_matrix_power_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_matrix_power_out, schema_str, "linalg_matrix_power.out(Tensor self, int n, *, Tensor(a!) out) -> Tensor(a!)")

// aten::linalg_matrix_power.out(Tensor self, int n, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_matrix_power_out::schema> create_linalg_matrix_power_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_matrix_power_out::name, linalg_matrix_power_out::overload_name)
      .typed<linalg_matrix_power_out::schema>();
}

// aten::linalg_matrix_power.out(Tensor self, int n, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_matrix_power_out::call(const at::Tensor & self, int64_t n, at::Tensor & out) {
    static auto op = create_linalg_matrix_power_out_typed_handle();
    return op.call(self, n, out);
}

// aten::linalg_matrix_power.out(Tensor self, int n, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_matrix_power_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t n, at::Tensor & out) {
    static auto op = create_linalg_matrix_power_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, n, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_matrix_rank_out, name, "aten::linalg_matrix_rank")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_matrix_rank_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_matrix_rank_out, schema_str, "linalg_matrix_rank.out(Tensor self, float? tol=None, bool hermitian=False, *, Tensor(a!) out) -> Tensor(a!)")

// aten::linalg_matrix_rank.out(Tensor self, float? tol=None, bool hermitian=False, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_matrix_rank_out::schema> create_linalg_matrix_rank_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_matrix_rank_out::name, linalg_matrix_rank_out::overload_name)
      .typed<linalg_matrix_rank_out::schema>();
}

// aten::linalg_matrix_rank.out(Tensor self, float? tol=None, bool hermitian=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_matrix_rank_out::call(const at::Tensor & self, c10::optional<double> tol, bool hermitian, at::Tensor & out) {
    static auto op = create_linalg_matrix_rank_out_typed_handle();
    return op.call(self, tol, hermitian, out);
}

// aten::linalg_matrix_rank.out(Tensor self, float? tol=None, bool hermitian=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_matrix_rank_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<double> tol, bool hermitian, at::Tensor & out) {
    static auto op = create_linalg_matrix_rank_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, tol, hermitian, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_test_serialization_subcmul, name, "aten::_test_serialization_subcmul")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_test_serialization_subcmul, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_test_serialization_subcmul, schema_str, "_test_serialization_subcmul(Tensor self, Tensor other, Scalar alpha=1) -> Tensor")

// aten::_test_serialization_subcmul(Tensor self, Tensor other, Scalar alpha=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_test_serialization_subcmul::schema> create__test_serialization_subcmul_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_test_serialization_subcmul::name, _test_serialization_subcmul::overload_name)
      .typed<_test_serialization_subcmul::schema>();
}

// aten::_test_serialization_subcmul(Tensor self, Tensor other, Scalar alpha=1) -> Tensor
at::Tensor _test_serialization_subcmul::call(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
    static auto op = create__test_serialization_subcmul_typed_handle();
    return op.call(self, other, alpha);
}

// aten::_test_serialization_subcmul(Tensor self, Tensor other, Scalar alpha=1) -> Tensor
at::Tensor _test_serialization_subcmul::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
    static auto op = create__test_serialization_subcmul_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_test_optional_intlist, name, "aten::_test_optional_intlist")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_test_optional_intlist, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_test_optional_intlist, schema_str, "_test_optional_intlist(Tensor values, int[]? addends) -> Tensor")

// aten::_test_optional_intlist(Tensor values, int[]? addends) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_test_optional_intlist::schema> create__test_optional_intlist_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_test_optional_intlist::name, _test_optional_intlist::overload_name)
      .typed<_test_optional_intlist::schema>();
}

// aten::_test_optional_intlist(Tensor values, int[]? addends) -> Tensor
at::Tensor _test_optional_intlist::call(const at::Tensor & values, c10::optional<at::IntArrayRef> addends) {
    static auto op = create__test_optional_intlist_typed_handle();
    return op.call(values, addends);
}

// aten::_test_optional_intlist(Tensor values, int[]? addends) -> Tensor
at::Tensor _test_optional_intlist::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & values, c10::optional<at::IntArrayRef> addends) {
    static auto op = create__test_optional_intlist_typed_handle();
    return op.redispatch(dispatchKeySet, values, addends);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_test_ambiguous_defaults_b, name, "aten::_test_ambiguous_defaults")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_test_ambiguous_defaults_b, overload_name, "b")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_test_ambiguous_defaults_b, schema_str, "_test_ambiguous_defaults.b(Tensor dummy, int a=2, str b=\"2\") -> Tensor")

// aten::_test_ambiguous_defaults.b(Tensor dummy, int a=2, str b="2") -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_test_ambiguous_defaults_b::schema> create__test_ambiguous_defaults_b_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_test_ambiguous_defaults_b::name, _test_ambiguous_defaults_b::overload_name)
      .typed<_test_ambiguous_defaults_b::schema>();
}

// aten::_test_ambiguous_defaults.b(Tensor dummy, int a=2, str b="2") -> Tensor
at::Tensor _test_ambiguous_defaults_b::call(const at::Tensor & dummy, int64_t a, c10::string_view b) {
    static auto op = create__test_ambiguous_defaults_b_typed_handle();
    return op.call(dummy, a, b);
}

// aten::_test_ambiguous_defaults.b(Tensor dummy, int a=2, str b="2") -> Tensor
at::Tensor _test_ambiguous_defaults_b::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & dummy, int64_t a, c10::string_view b) {
    static auto op = create__test_ambiguous_defaults_b_typed_handle();
    return op.redispatch(dispatchKeySet, dummy, a, b);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(segment_reduce, name, "aten::segment_reduce")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(segment_reduce, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(segment_reduce, schema_str, "segment_reduce(Tensor data, str reduce, *, Tensor? lengths=None, Tensor? indices=None, int axis=0, bool unsafe=False, Scalar? initial=None) -> Tensor")

// aten::segment_reduce(Tensor data, str reduce, *, Tensor? lengths=None, Tensor? indices=None, int axis=0, bool unsafe=False, Scalar? initial=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<segment_reduce::schema> create_segment_reduce_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(segment_reduce::name, segment_reduce::overload_name)
      .typed<segment_reduce::schema>();
}

// aten::segment_reduce(Tensor data, str reduce, *, Tensor? lengths=None, Tensor? indices=None, int axis=0, bool unsafe=False, Scalar? initial=None) -> Tensor
at::Tensor segment_reduce::call(const at::Tensor & data, c10::string_view reduce, const c10::optional<at::Tensor> & lengths, const c10::optional<at::Tensor> & indices, int64_t axis, bool unsafe, const c10::optional<at::Scalar> & initial) {
    static auto op = create_segment_reduce_typed_handle();
    return op.call(data, reduce, lengths, indices, axis, unsafe, initial);
}

// aten::segment_reduce(Tensor data, str reduce, *, Tensor? lengths=None, Tensor? indices=None, int axis=0, bool unsafe=False, Scalar? initial=None) -> Tensor
at::Tensor segment_reduce::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & data, c10::string_view reduce, const c10::optional<at::Tensor> & lengths, const c10::optional<at::Tensor> & indices, int64_t axis, bool unsafe, const c10::optional<at::Scalar> & initial) {
    static auto op = create_segment_reduce_typed_handle();
    return op.redispatch(dispatchKeySet, data, reduce, lengths, indices, axis, unsafe, initial);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_segment_reduce_backward, name, "aten::_segment_reduce_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_segment_reduce_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_segment_reduce_backward, schema_str, "_segment_reduce_backward(Tensor grad, Tensor output, Tensor data, str reduce, *, Tensor? lengths=None, int axis=0) -> Tensor")

// aten::_segment_reduce_backward(Tensor grad, Tensor output, Tensor data, str reduce, *, Tensor? lengths=None, int axis=0) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_segment_reduce_backward::schema> create__segment_reduce_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_segment_reduce_backward::name, _segment_reduce_backward::overload_name)
      .typed<_segment_reduce_backward::schema>();
}

// aten::_segment_reduce_backward(Tensor grad, Tensor output, Tensor data, str reduce, *, Tensor? lengths=None, int axis=0) -> Tensor
at::Tensor _segment_reduce_backward::call(const at::Tensor & grad, const at::Tensor & output, const at::Tensor & data, c10::string_view reduce, const c10::optional<at::Tensor> & lengths, int64_t axis) {
    static auto op = create__segment_reduce_backward_typed_handle();
    return op.call(grad, output, data, reduce, lengths, axis);
}

// aten::_segment_reduce_backward(Tensor grad, Tensor output, Tensor data, str reduce, *, Tensor? lengths=None, int axis=0) -> Tensor
at::Tensor _segment_reduce_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & output, const at::Tensor & data, c10::string_view reduce, const c10::optional<at::Tensor> & lengths, int64_t axis) {
    static auto op = create__segment_reduce_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad, output, data, reduce, lengths, axis);
}

}} // namespace at::_ops
