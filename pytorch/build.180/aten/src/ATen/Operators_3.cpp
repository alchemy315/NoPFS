#include <ATen/Operators.h>
#include <ATen/Tensor.h>
#include <ATen/core/dispatch/Dispatcher.h>

// NOTE See [Sharded File] comment in VariableType

namespace at { namespace _ops {


STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cast_Int, name, "aten::_cast_Int")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cast_Int, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cast_Int, schema_str, "_cast_Int(Tensor self, bool non_blocking=False) -> Tensor")

// aten::_cast_Int(Tensor self, bool non_blocking=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_cast_Int::schema> create__cast_Int_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_cast_Int::name, _cast_Int::overload_name)
      .typed<_cast_Int::schema>();
}

// aten::_cast_Int(Tensor self, bool non_blocking=False) -> Tensor
at::Tensor _cast_Int::call(const at::Tensor & self, bool non_blocking) {
    static auto op = create__cast_Int_typed_handle();
    return op.call(self, non_blocking);
}

// aten::_cast_Int(Tensor self, bool non_blocking=False) -> Tensor
at::Tensor _cast_Int::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool non_blocking) {
    static auto op = create__cast_Int_typed_handle();
    return op.redispatch(dispatchKeySet, self, non_blocking);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cast_Long, name, "aten::_cast_Long")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cast_Long, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cast_Long, schema_str, "_cast_Long(Tensor self, bool non_blocking=False) -> Tensor")

// aten::_cast_Long(Tensor self, bool non_blocking=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_cast_Long::schema> create__cast_Long_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_cast_Long::name, _cast_Long::overload_name)
      .typed<_cast_Long::schema>();
}

// aten::_cast_Long(Tensor self, bool non_blocking=False) -> Tensor
at::Tensor _cast_Long::call(const at::Tensor & self, bool non_blocking) {
    static auto op = create__cast_Long_typed_handle();
    return op.call(self, non_blocking);
}

// aten::_cast_Long(Tensor self, bool non_blocking=False) -> Tensor
at::Tensor _cast_Long::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool non_blocking) {
    static auto op = create__cast_Long_typed_handle();
    return op.redispatch(dispatchKeySet, self, non_blocking);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_backward, name, "aten::_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_backward, schema_str, "_backward(Tensor self, Tensor[] inputs, Tensor? gradient=None, bool? retain_graph=None, bool create_graph=False) -> ()")

// aten::_backward(Tensor self, Tensor[] inputs, Tensor? gradient=None, bool? retain_graph=None, bool create_graph=False) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_backward::schema> create__backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_backward::name, _backward::overload_name)
      .typed<_backward::schema>();
}

// aten::_backward(Tensor self, Tensor[] inputs, Tensor? gradient=None, bool? retain_graph=None, bool create_graph=False) -> ()
void _backward::call(const at::Tensor & self, at::TensorList inputs, const c10::optional<at::Tensor> & gradient, c10::optional<bool> retain_graph, bool create_graph) {
    static auto op = create__backward_typed_handle();
    return op.call(self, inputs, gradient, retain_graph, create_graph);
}

// aten::_backward(Tensor self, Tensor[] inputs, Tensor? gradient=None, bool? retain_graph=None, bool create_graph=False) -> ()
void _backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::TensorList inputs, const c10::optional<at::Tensor> & gradient, c10::optional<bool> retain_graph, bool create_graph) {
    static auto op = create__backward_typed_handle();
    return op.redispatch(dispatchKeySet, self, inputs, gradient, retain_graph, create_graph);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(data, name, "aten::data")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(data, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(data, schema_str, "data(Tensor self) -> Tensor")

// aten::data(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<data::schema> create_data_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(data::name, data::overload_name)
      .typed<data::schema>();
}

// aten::data(Tensor self) -> Tensor
at::Tensor data::call(const at::Tensor & self) {
    static auto op = create_data_typed_handle();
    return op.call(self);
}

// aten::data(Tensor self) -> Tensor
at::Tensor data::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_data_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(retain_grad, name, "aten::retain_grad")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(retain_grad, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(retain_grad, schema_str, "retain_grad(Tensor(a!) self) -> ()")

// aten::retain_grad(Tensor(a!) self) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<retain_grad::schema> create_retain_grad_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(retain_grad::name, retain_grad::overload_name)
      .typed<retain_grad::schema>();
}

// aten::retain_grad(Tensor(a!) self) -> ()
void retain_grad::call(at::Tensor & self) {
    static auto op = create_retain_grad_typed_handle();
    return op.call(self);
}

// aten::retain_grad(Tensor(a!) self) -> ()
void retain_grad::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_retain_grad_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rename_, name, "aten::rename_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rename_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rename_, schema_str, "rename_(Tensor(a!) self, Dimname[]? names) -> Tensor(a!)")

// aten::rename_(Tensor(a!) self, Dimname[]? names) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<rename_::schema> create_rename__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(rename_::name, rename_::overload_name)
      .typed<rename_::schema>();
}

// aten::rename_(Tensor(a!) self, Dimname[]? names) -> Tensor(a!)
at::Tensor & rename_::call(at::Tensor & self, c10::optional<at::DimnameList> names) {
    static auto op = create_rename__typed_handle();
    return op.call(self, names);
}

// aten::rename_(Tensor(a!) self, Dimname[]? names) -> Tensor(a!)
at::Tensor & rename_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, c10::optional<at::DimnameList> names) {
    static auto op = create_rename__typed_handle();
    return op.redispatch(dispatchKeySet, self, names);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rename, name, "aten::rename")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rename, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rename, schema_str, "rename(Tensor(a) self, Dimname[]? names) -> Tensor(a)")

// aten::rename(Tensor(a) self, Dimname[]? names) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<rename::schema> create_rename_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(rename::name, rename::overload_name)
      .typed<rename::schema>();
}

// aten::rename(Tensor(a) self, Dimname[]? names) -> Tensor(a)
at::Tensor rename::call(const at::Tensor & self, c10::optional<at::DimnameList> names) {
    static auto op = create_rename_typed_handle();
    return op.call(self, names);
}

// aten::rename(Tensor(a) self, Dimname[]? names) -> Tensor(a)
at::Tensor rename::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::DimnameList> names) {
    static auto op = create_rename_typed_handle();
    return op.redispatch(dispatchKeySet, self, names);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cudnn_rnn_backward, name, "aten::_cudnn_rnn_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cudnn_rnn_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cudnn_rnn_backward, schema_str, "_cudnn_rnn_backward(Tensor input, Tensor[] weight, int weight_stride0, Tensor weight_buf, Tensor hx, Tensor? cx, Tensor output, Tensor? grad_output, Tensor? grad_hy, Tensor? grad_cy, int mode, int hidden_size, int proj_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state, Tensor reserve, bool[4] output_mask) -> (Tensor, Tensor, Tensor, Tensor[])")

// aten::_cudnn_rnn_backward(Tensor input, Tensor[] weight, int weight_stride0, Tensor weight_buf, Tensor hx, Tensor? cx, Tensor output, Tensor? grad_output, Tensor? grad_hy, Tensor? grad_cy, int mode, int hidden_size, int proj_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state, Tensor reserve, bool[4] output_mask) -> (Tensor, Tensor, Tensor, Tensor[])
static C10_NOINLINE c10::TypedOperatorHandle<_cudnn_rnn_backward::schema> create__cudnn_rnn_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_cudnn_rnn_backward::name, _cudnn_rnn_backward::overload_name)
      .typed<_cudnn_rnn_backward::schema>();
}

// aten::_cudnn_rnn_backward(Tensor input, Tensor[] weight, int weight_stride0, Tensor weight_buf, Tensor hx, Tensor? cx, Tensor output, Tensor? grad_output, Tensor? grad_hy, Tensor? grad_cy, int mode, int hidden_size, int proj_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state, Tensor reserve, bool[4] output_mask) -> (Tensor, Tensor, Tensor, Tensor[])
::std::tuple<at::Tensor,at::Tensor,at::Tensor,::std::vector<at::Tensor>> _cudnn_rnn_backward::call(const at::Tensor & input, at::TensorList weight, int64_t weight_stride0, const at::Tensor & weight_buf, const at::Tensor & hx, const c10::optional<at::Tensor> & cx, const at::Tensor & output, const c10::optional<at::Tensor> & grad_output, const c10::optional<at::Tensor> & grad_hy, const c10::optional<at::Tensor> & grad_cy, int64_t mode, int64_t hidden_size, int64_t proj_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, at::IntArrayRef batch_sizes, const c10::optional<at::Tensor> & dropout_state, const at::Tensor & reserve, ::std::array<bool,4> output_mask) {
    static auto op = create__cudnn_rnn_backward_typed_handle();
    return op.call(input, weight, weight_stride0, weight_buf, hx, cx, output, grad_output, grad_hy, grad_cy, mode, hidden_size, proj_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, reserve, output_mask);
}

// aten::_cudnn_rnn_backward(Tensor input, Tensor[] weight, int weight_stride0, Tensor weight_buf, Tensor hx, Tensor? cx, Tensor output, Tensor? grad_output, Tensor? grad_hy, Tensor? grad_cy, int mode, int hidden_size, int proj_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state, Tensor reserve, bool[4] output_mask) -> (Tensor, Tensor, Tensor, Tensor[])
::std::tuple<at::Tensor,at::Tensor,at::Tensor,::std::vector<at::Tensor>> _cudnn_rnn_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, at::TensorList weight, int64_t weight_stride0, const at::Tensor & weight_buf, const at::Tensor & hx, const c10::optional<at::Tensor> & cx, const at::Tensor & output, const c10::optional<at::Tensor> & grad_output, const c10::optional<at::Tensor> & grad_hy, const c10::optional<at::Tensor> & grad_cy, int64_t mode, int64_t hidden_size, int64_t proj_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, at::IntArrayRef batch_sizes, const c10::optional<at::Tensor> & dropout_state, const at::Tensor & reserve, ::std::array<bool,4> output_mask) {
    static auto op = create__cudnn_rnn_backward_typed_handle();
    return op.redispatch(dispatchKeySet, input, weight, weight_stride0, weight_buf, hx, cx, output, grad_output, grad_hy, grad_cy, mode, hidden_size, proj_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, reserve, output_mask);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sobol_engine_scramble_, name, "aten::_sobol_engine_scramble_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sobol_engine_scramble_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sobol_engine_scramble_, schema_str, "_sobol_engine_scramble_(Tensor(a!) self, Tensor ltm, int dimension) -> Tensor(a!)")

// aten::_sobol_engine_scramble_(Tensor(a!) self, Tensor ltm, int dimension) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<_sobol_engine_scramble_::schema> create__sobol_engine_scramble__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_sobol_engine_scramble_::name, _sobol_engine_scramble_::overload_name)
      .typed<_sobol_engine_scramble_::schema>();
}

// aten::_sobol_engine_scramble_(Tensor(a!) self, Tensor ltm, int dimension) -> Tensor(a!)
at::Tensor & _sobol_engine_scramble_::call(at::Tensor & self, const at::Tensor & ltm, int64_t dimension) {
    static auto op = create__sobol_engine_scramble__typed_handle();
    return op.call(self, ltm, dimension);
}

// aten::_sobol_engine_scramble_(Tensor(a!) self, Tensor ltm, int dimension) -> Tensor(a!)
at::Tensor & _sobol_engine_scramble_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & ltm, int64_t dimension) {
    static auto op = create__sobol_engine_scramble__typed_handle();
    return op.redispatch(dispatchKeySet, self, ltm, dimension);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(feature_dropout, name, "aten::feature_dropout")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(feature_dropout, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(feature_dropout, schema_str, "feature_dropout(Tensor input, float p, bool train) -> Tensor")

// aten::feature_dropout(Tensor input, float p, bool train) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<feature_dropout::schema> create_feature_dropout_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(feature_dropout::name, feature_dropout::overload_name)
      .typed<feature_dropout::schema>();
}

// aten::feature_dropout(Tensor input, float p, bool train) -> Tensor
at::Tensor feature_dropout::call(const at::Tensor & input, double p, bool train) {
    static auto op = create_feature_dropout_typed_handle();
    return op.call(input, p, train);
}

// aten::feature_dropout(Tensor input, float p, bool train) -> Tensor
at::Tensor feature_dropout::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, double p, bool train) {
    static auto op = create_feature_dropout_typed_handle();
    return op.redispatch(dispatchKeySet, input, p, train);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(feature_dropout_, name, "aten::feature_dropout_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(feature_dropout_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(feature_dropout_, schema_str, "feature_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)")

// aten::feature_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<feature_dropout_::schema> create_feature_dropout__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(feature_dropout_::name, feature_dropout_::overload_name)
      .typed<feature_dropout_::schema>();
}

// aten::feature_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)
at::Tensor & feature_dropout_::call(at::Tensor & self, double p, bool train) {
    static auto op = create_feature_dropout__typed_handle();
    return op.call(self, p, train);
}

// aten::feature_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)
at::Tensor & feature_dropout_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, double p, bool train) {
    static auto op = create_feature_dropout__typed_handle();
    return op.redispatch(dispatchKeySet, self, p, train);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(alpha_dropout_, name, "aten::alpha_dropout_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(alpha_dropout_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(alpha_dropout_, schema_str, "alpha_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)")

// aten::alpha_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<alpha_dropout_::schema> create_alpha_dropout__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(alpha_dropout_::name, alpha_dropout_::overload_name)
      .typed<alpha_dropout_::schema>();
}

// aten::alpha_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)
at::Tensor & alpha_dropout_::call(at::Tensor & self, double p, bool train) {
    static auto op = create_alpha_dropout__typed_handle();
    return op.call(self, p, train);
}

// aten::alpha_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)
at::Tensor & alpha_dropout_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, double p, bool train) {
    static auto op = create_alpha_dropout__typed_handle();
    return op.redispatch(dispatchKeySet, self, p, train);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(absolute_, name, "aten::absolute_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(absolute_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(absolute_, schema_str, "absolute_(Tensor(a!) self) -> Tensor(a!)")

// aten::absolute_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<absolute_::schema> create_absolute__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(absolute_::name, absolute_::overload_name)
      .typed<absolute_::schema>();
}

// aten::absolute_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & absolute_::call(at::Tensor & self) {
    static auto op = create_absolute__typed_handle();
    return op.call(self);
}

// aten::absolute_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & absolute_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_absolute__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conj, name, "aten::conj")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conj, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conj, schema_str, "conj(Tensor(a) self) -> Tensor(a)")

// aten::conj(Tensor(a) self) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<conj::schema> create_conj_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(conj::name, conj::overload_name)
      .typed<conj::schema>();
}

// aten::conj(Tensor(a) self) -> Tensor(a)
at::Tensor conj::call(const at::Tensor & self) {
    static auto op = create_conj_typed_handle();
    return op.call(self);
}

// aten::conj(Tensor(a) self) -> Tensor(a)
at::Tensor conj::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_conj_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conj_physical_, name, "aten::conj_physical_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conj_physical_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conj_physical_, schema_str, "conj_physical_(Tensor(a!) self) -> Tensor(a!)")

// aten::conj_physical_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<conj_physical_::schema> create_conj_physical__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(conj_physical_::name, conj_physical_::overload_name)
      .typed<conj_physical_::schema>();
}

// aten::conj_physical_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & conj_physical_::call(at::Tensor & self) {
    static auto op = create_conj_physical__typed_handle();
    return op.call(self);
}

// aten::conj_physical_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & conj_physical_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_conj_physical__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(add_Tensor, name, "aten::add")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(add_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(add_Tensor, schema_str, "add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor")

// aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<add_Tensor::schema> create_add_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(add_Tensor::name, add_Tensor::overload_name)
      .typed<add_Tensor::schema>();
}

// aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
at::Tensor add_Tensor::call(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
    static auto op = create_add_Tensor_typed_handle();
    return op.call(self, other, alpha);
}

// aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
at::Tensor add_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
    static auto op = create_add_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_add_relu_Tensor, name, "aten::_add_relu")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_add_relu_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_add_relu_Tensor, schema_str, "_add_relu.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor")

// aten::_add_relu.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_add_relu_Tensor::schema> create__add_relu_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_add_relu_Tensor::name, _add_relu_Tensor::overload_name)
      .typed<_add_relu_Tensor::schema>();
}

// aten::_add_relu.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
at::Tensor _add_relu_Tensor::call(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
    static auto op = create__add_relu_Tensor_typed_handle();
    return op.call(self, other, alpha);
}

// aten::_add_relu.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
at::Tensor _add_relu_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
    static auto op = create__add_relu_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_add_relu_Scalar, name, "aten::_add_relu")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_add_relu_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_add_relu_Scalar, schema_str, "_add_relu.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor")

// aten::_add_relu.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_add_relu_Scalar::schema> create__add_relu_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_add_relu_Scalar::name, _add_relu_Scalar::overload_name)
      .typed<_add_relu_Scalar::schema>();
}

// aten::_add_relu.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
at::Tensor _add_relu_Scalar::call(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
    static auto op = create__add_relu_Scalar_typed_handle();
    return op.call(self, other, alpha);
}

// aten::_add_relu.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
at::Tensor _add_relu_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
    static auto op = create__add_relu_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(add_Scalar, name, "aten::add")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(add_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(add_Scalar, schema_str, "add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor")

// aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<add_Scalar::schema> create_add_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(add_Scalar::name, add_Scalar::overload_name)
      .typed<add_Scalar::schema>();
}

// aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
at::Tensor add_Scalar::call(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
    static auto op = create_add_Scalar_typed_handle();
    return op.call(self, other, alpha);
}

// aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
at::Tensor add_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
    static auto op = create_add_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(affine_grid_generator, name, "aten::affine_grid_generator")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(affine_grid_generator, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(affine_grid_generator, schema_str, "affine_grid_generator(Tensor theta, int[] size, bool align_corners) -> Tensor")

// aten::affine_grid_generator(Tensor theta, int[] size, bool align_corners) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<affine_grid_generator::schema> create_affine_grid_generator_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(affine_grid_generator::name, affine_grid_generator::overload_name)
      .typed<affine_grid_generator::schema>();
}

// aten::affine_grid_generator(Tensor theta, int[] size, bool align_corners) -> Tensor
at::Tensor affine_grid_generator::call(const at::Tensor & theta, at::IntArrayRef size, bool align_corners) {
    static auto op = create_affine_grid_generator_typed_handle();
    return op.call(theta, size, align_corners);
}

// aten::affine_grid_generator(Tensor theta, int[] size, bool align_corners) -> Tensor
at::Tensor affine_grid_generator::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & theta, at::IntArrayRef size, bool align_corners) {
    static auto op = create_affine_grid_generator_typed_handle();
    return op.redispatch(dispatchKeySet, theta, size, align_corners);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(all_dimname, name, "aten::all")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(all_dimname, overload_name, "dimname")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(all_dimname, schema_str, "all.dimname(Tensor self, Dimname dim, bool keepdim=False) -> Tensor")

// aten::all.dimname(Tensor self, Dimname dim, bool keepdim=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<all_dimname::schema> create_all_dimname_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(all_dimname::name, all_dimname::overload_name)
      .typed<all_dimname::schema>();
}

// aten::all.dimname(Tensor self, Dimname dim, bool keepdim=False) -> Tensor
at::Tensor all_dimname::call(const at::Tensor & self, at::Dimname dim, bool keepdim) {
    static auto op = create_all_dimname_typed_handle();
    return op.call(self, dim, keepdim);
}

// aten::all.dimname(Tensor self, Dimname dim, bool keepdim=False) -> Tensor
at::Tensor all_dimname::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, bool keepdim) {
    static auto op = create_all_dimname_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(any_dim, name, "aten::any")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(any_dim, overload_name, "dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(any_dim, schema_str, "any.dim(Tensor self, int dim, bool keepdim=False) -> Tensor")

// aten::any.dim(Tensor self, int dim, bool keepdim=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<any_dim::schema> create_any_dim_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(any_dim::name, any_dim::overload_name)
      .typed<any_dim::schema>();
}

// aten::any.dim(Tensor self, int dim, bool keepdim=False) -> Tensor
at::Tensor any_dim::call(const at::Tensor & self, int64_t dim, bool keepdim) {
    static auto op = create_any_dim_typed_handle();
    return op.call(self, dim, keepdim);
}

// aten::any.dim(Tensor self, int dim, bool keepdim=False) -> Tensor
at::Tensor any_dim::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool keepdim) {
    static auto op = create_any_dim_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(any_out, name, "aten::any")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(any_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(any_out, schema_str, "any.out(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)")

// aten::any.out(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<any_out::schema> create_any_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(any_out::name, any_out::overload_name)
      .typed<any_out::schema>();
}

// aten::any.out(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & any_out::call(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & out) {
    static auto op = create_any_out_typed_handle();
    return op.call(self, dim, keepdim, out);
}

// aten::any.out(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & any_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & out) {
    static auto op = create_any_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arange, name, "aten::arange")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arange, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arange, schema_str, "arange(Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::arange(Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<arange::schema> create_arange_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(arange::name, arange::overload_name)
      .typed<arange::schema>();
}

// aten::arange(Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor arange::call(const at::Scalar & end, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_arange_typed_handle();
    return op.call(end, dtype, layout, device, pin_memory);
}

// aten::arange(Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor arange::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Scalar & end, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_arange_typed_handle();
    return op.redispatch(dispatchKeySet, end, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arange_start_out, name, "aten::arange")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arange_start_out, overload_name, "start_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arange_start_out, schema_str, "arange.start_out(Scalar start, Scalar end, Scalar step=1, *, Tensor(a!) out) -> Tensor(a!)")

// aten::arange.start_out(Scalar start, Scalar end, Scalar step=1, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<arange_start_out::schema> create_arange_start_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(arange_start_out::name, arange_start_out::overload_name)
      .typed<arange_start_out::schema>();
}

// aten::arange.start_out(Scalar start, Scalar end, Scalar step=1, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & arange_start_out::call(const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, at::Tensor & out) {
    static auto op = create_arange_start_out_typed_handle();
    return op.call(start, end, step, out);
}

// aten::arange.start_out(Scalar start, Scalar end, Scalar step=1, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & arange_start_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, at::Tensor & out) {
    static auto op = create_arange_start_out_typed_handle();
    return op.redispatch(dispatchKeySet, start, end, step, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_dim_arange, name, "aten::_dim_arange")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_dim_arange, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_dim_arange, schema_str, "_dim_arange(Tensor like, int dim) -> Tensor")

// aten::_dim_arange(Tensor like, int dim) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_dim_arange::schema> create__dim_arange_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_dim_arange::name, _dim_arange::overload_name)
      .typed<_dim_arange::schema>();
}

// aten::_dim_arange(Tensor like, int dim) -> Tensor
at::Tensor _dim_arange::call(const at::Tensor & like, int64_t dim) {
    static auto op = create__dim_arange_typed_handle();
    return op.call(like, dim);
}

// aten::_dim_arange(Tensor like, int dim) -> Tensor
at::Tensor _dim_arange::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & like, int64_t dim) {
    static auto op = create__dim_arange_typed_handle();
    return op.redispatch(dispatchKeySet, like, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arccosh_, name, "aten::arccosh_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arccosh_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arccosh_, schema_str, "arccosh_(Tensor(a!) self) -> Tensor(a!)")

// aten::arccosh_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<arccosh_::schema> create_arccosh__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(arccosh_::name, arccosh_::overload_name)
      .typed<arccosh_::schema>();
}

// aten::arccosh_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & arccosh_::call(at::Tensor & self) {
    static auto op = create_arccosh__typed_handle();
    return op.call(self);
}

// aten::arccosh_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & arccosh_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_arccosh__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arcsinh, name, "aten::arcsinh")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arcsinh, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arcsinh, schema_str, "arcsinh(Tensor self) -> Tensor")

// aten::arcsinh(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<arcsinh::schema> create_arcsinh_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(arcsinh::name, arcsinh::overload_name)
      .typed<arcsinh::schema>();
}

// aten::arcsinh(Tensor self) -> Tensor
at::Tensor arcsinh::call(const at::Tensor & self) {
    static auto op = create_arcsinh_typed_handle();
    return op.call(self);
}

// aten::arcsinh(Tensor self) -> Tensor
at::Tensor arcsinh::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_arcsinh_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atanh, name, "aten::atanh")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atanh, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atanh, schema_str, "atanh(Tensor self) -> Tensor")

// aten::atanh(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<atanh::schema> create_atanh_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(atanh::name, atanh::overload_name)
      .typed<atanh::schema>();
}

// aten::atanh(Tensor self) -> Tensor
at::Tensor atanh::call(const at::Tensor & self) {
    static auto op = create_atanh_typed_handle();
    return op.call(self);
}

// aten::atanh(Tensor self) -> Tensor
at::Tensor atanh::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_atanh_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arctanh_, name, "aten::arctanh_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arctanh_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arctanh_, schema_str, "arctanh_(Tensor(a!) self) -> Tensor(a!)")

// aten::arctanh_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<arctanh_::schema> create_arctanh__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(arctanh_::name, arctanh_::overload_name)
      .typed<arctanh_::schema>();
}

// aten::arctanh_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & arctanh_::call(at::Tensor & self) {
    static auto op = create_arctanh__typed_handle();
    return op.call(self);
}

// aten::arctanh_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & arctanh_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_arctanh__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arcsin, name, "aten::arcsin")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arcsin, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(arcsin, schema_str, "arcsin(Tensor self) -> Tensor")

// aten::arcsin(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<arcsin::schema> create_arcsin_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(arcsin::name, arcsin::overload_name)
      .typed<arcsin::schema>();
}

// aten::arcsin(Tensor self) -> Tensor
at::Tensor arcsin::call(const at::Tensor & self) {
    static auto op = create_arcsin_typed_handle();
    return op.call(self);
}

// aten::arcsin(Tensor self) -> Tensor
at::Tensor arcsin::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_arcsin_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atan_out, name, "aten::atan")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atan_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(atan_out, schema_str, "atan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::atan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<atan_out::schema> create_atan_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(atan_out::name, atan_out::overload_name)
      .typed<atan_out::schema>();
}

// aten::atan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & atan_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_atan_out_typed_handle();
    return op.call(self, out);
}

// aten::atan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & atan_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_atan_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bartlett_window, name, "aten::bartlett_window")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bartlett_window, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bartlett_window, schema_str, "bartlett_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::bartlett_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<bartlett_window::schema> create_bartlett_window_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bartlett_window::name, bartlett_window::overload_name)
      .typed<bartlett_window::schema>();
}

// aten::bartlett_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor bartlett_window::call(int64_t window_length, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_bartlett_window_typed_handle();
    return op.call(window_length, dtype, layout, device, pin_memory);
}

// aten::bartlett_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor bartlett_window::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t window_length, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_bartlett_window_typed_handle();
    return op.redispatch(dispatchKeySet, window_length, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bernoulli__float, name, "aten::bernoulli_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bernoulli__float, overload_name, "float")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bernoulli__float, schema_str, "bernoulli_.float(Tensor(a!) self, float p=0.5, *, Generator? generator=None) -> Tensor(a!)")

// aten::bernoulli_.float(Tensor(a!) self, float p=0.5, *, Generator? generator=None) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<bernoulli__float::schema> create_bernoulli__float_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bernoulli__float::name, bernoulli__float::overload_name)
      .typed<bernoulli__float::schema>();
}

// aten::bernoulli_.float(Tensor(a!) self, float p=0.5, *, Generator? generator=None) -> Tensor(a!)
at::Tensor & bernoulli__float::call(at::Tensor & self, double p, c10::optional<at::Generator> generator) {
    static auto op = create_bernoulli__float_typed_handle();
    return op.call(self, p, generator);
}

// aten::bernoulli_.float(Tensor(a!) self, float p=0.5, *, Generator? generator=None) -> Tensor(a!)
at::Tensor & bernoulli__float::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, double p, c10::optional<at::Generator> generator) {
    static auto op = create_bernoulli__float_typed_handle();
    return op.redispatch(dispatchKeySet, self, p, generator);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(binary_cross_entropy, name, "aten::binary_cross_entropy")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(binary_cross_entropy, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(binary_cross_entropy, schema_str, "binary_cross_entropy(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor")

// aten::binary_cross_entropy(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<binary_cross_entropy::schema> create_binary_cross_entropy_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(binary_cross_entropy::name, binary_cross_entropy::overload_name)
      .typed<binary_cross_entropy::schema>();
}

// aten::binary_cross_entropy(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor
at::Tensor binary_cross_entropy::call(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction) {
    static auto op = create_binary_cross_entropy_typed_handle();
    return op.call(self, target, weight, reduction);
}

// aten::binary_cross_entropy(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor
at::Tensor binary_cross_entropy::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction) {
    static auto op = create_binary_cross_entropy_typed_handle();
    return op.redispatch(dispatchKeySet, self, target, weight, reduction);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(binary_cross_entropy_with_logits_backward, name, "aten::binary_cross_entropy_with_logits_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(binary_cross_entropy_with_logits_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(binary_cross_entropy_with_logits_backward, schema_str, "binary_cross_entropy_with_logits_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, Tensor? pos_weight=None, int reduction=Mean) -> Tensor")

// aten::binary_cross_entropy_with_logits_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, Tensor? pos_weight=None, int reduction=Mean) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<binary_cross_entropy_with_logits_backward::schema> create_binary_cross_entropy_with_logits_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(binary_cross_entropy_with_logits_backward::name, binary_cross_entropy_with_logits_backward::overload_name)
      .typed<binary_cross_entropy_with_logits_backward::schema>();
}

// aten::binary_cross_entropy_with_logits_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, Tensor? pos_weight=None, int reduction=Mean) -> Tensor
at::Tensor binary_cross_entropy_with_logits_backward::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & pos_weight, int64_t reduction) {
    static auto op = create_binary_cross_entropy_with_logits_backward_typed_handle();
    return op.call(grad_output, self, target, weight, pos_weight, reduction);
}

// aten::binary_cross_entropy_with_logits_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, Tensor? pos_weight=None, int reduction=Mean) -> Tensor
at::Tensor binary_cross_entropy_with_logits_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & pos_weight, int64_t reduction) {
    static auto op = create_binary_cross_entropy_with_logits_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, target, weight, pos_weight, reduction);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_not_out, name, "aten::bitwise_not")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_not_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_not_out, schema_str, "bitwise_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::bitwise_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<bitwise_not_out::schema> create_bitwise_not_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bitwise_not_out::name, bitwise_not_out::overload_name)
      .typed<bitwise_not_out::schema>();
}

// aten::bitwise_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & bitwise_not_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_bitwise_not_out_typed_handle();
    return op.call(self, out);
}

// aten::bitwise_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & bitwise_not_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_bitwise_not_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(copysign_Tensor, name, "aten::copysign")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(copysign_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(copysign_Tensor, schema_str, "copysign.Tensor(Tensor self, Tensor other) -> Tensor")

// aten::copysign.Tensor(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<copysign_Tensor::schema> create_copysign_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(copysign_Tensor::name, copysign_Tensor::overload_name)
      .typed<copysign_Tensor::schema>();
}

// aten::copysign.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor copysign_Tensor::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_copysign_Tensor_typed_handle();
    return op.call(self, other);
}

// aten::copysign.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor copysign_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_copysign_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(copysign_Scalar_out, name, "aten::copysign")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(copysign_Scalar_out, overload_name, "Scalar_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(copysign_Scalar_out, schema_str, "copysign.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::copysign.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<copysign_Scalar_out::schema> create_copysign_Scalar_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(copysign_Scalar_out::name, copysign_Scalar_out::overload_name)
      .typed<copysign_Scalar_out::schema>();
}

// aten::copysign.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & copysign_Scalar_out::call(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_copysign_Scalar_out_typed_handle();
    return op.call(self, other, out);
}

// aten::copysign.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & copysign_Scalar_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_copysign_Scalar_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logical_xor_, name, "aten::logical_xor_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logical_xor_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logical_xor_, schema_str, "logical_xor_(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::logical_xor_(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<logical_xor_::schema> create_logical_xor__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(logical_xor_::name, logical_xor_::overload_name)
      .typed<logical_xor_::schema>();
}

// aten::logical_xor_(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & logical_xor_::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create_logical_xor__typed_handle();
    return op.call(self, other);
}

// aten::logical_xor_(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & logical_xor_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create_logical_xor__typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logical_xor_out, name, "aten::logical_xor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logical_xor_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logical_xor_out, schema_str, "logical_xor.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::logical_xor.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<logical_xor_out::schema> create_logical_xor_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(logical_xor_out::name, logical_xor_out::overload_name)
      .typed<logical_xor_out::schema>();
}

// aten::logical_xor.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & logical_xor_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_logical_xor_out_typed_handle();
    return op.call(self, other, out);
}

// aten::logical_xor.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & logical_xor_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_logical_xor_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bmm, name, "aten::bmm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bmm, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bmm, schema_str, "bmm(Tensor self, Tensor mat2) -> Tensor")

// aten::bmm(Tensor self, Tensor mat2) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<bmm::schema> create_bmm_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bmm::name, bmm::overload_name)
      .typed<bmm::schema>();
}

// aten::bmm(Tensor self, Tensor mat2) -> Tensor
at::Tensor bmm::call(const at::Tensor & self, const at::Tensor & mat2) {
    static auto op = create_bmm_typed_handle();
    return op.call(self, mat2);
}

// aten::bmm(Tensor self, Tensor mat2) -> Tensor
at::Tensor bmm::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mat2) {
    static auto op = create_bmm_typed_handle();
    return op.redispatch(dispatchKeySet, self, mat2);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bmm_out, name, "aten::bmm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bmm_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bmm_out, schema_str, "bmm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)")

// aten::bmm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<bmm_out::schema> create_bmm_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bmm_out::name, bmm_out::overload_name)
      .typed<bmm_out::schema>();
}

// aten::bmm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & bmm_out::call(const at::Tensor & self, const at::Tensor & mat2, at::Tensor & out) {
    static auto op = create_bmm_out_typed_handle();
    return op.call(self, mat2, out);
}

// aten::bmm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & bmm_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mat2, at::Tensor & out) {
    static auto op = create_bmm_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, mat2, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cat_names, name, "aten::cat")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cat_names, overload_name, "names")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cat_names, schema_str, "cat.names(Tensor[] tensors, Dimname dim) -> Tensor")

// aten::cat.names(Tensor[] tensors, Dimname dim) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<cat_names::schema> create_cat_names_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cat_names::name, cat_names::overload_name)
      .typed<cat_names::schema>();
}

// aten::cat.names(Tensor[] tensors, Dimname dim) -> Tensor
at::Tensor cat_names::call(at::TensorList tensors, at::Dimname dim) {
    static auto op = create_cat_names_typed_handle();
    return op.call(tensors, dim);
}

// aten::cat.names(Tensor[] tensors, Dimname dim) -> Tensor
at::Tensor cat_names::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, at::Dimname dim) {
    static auto op = create_cat_names_typed_handle();
    return op.redispatch(dispatchKeySet, tensors, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(concat, name, "aten::concat")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(concat, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(concat, schema_str, "concat(Tensor[] tensors, int dim=0) -> Tensor")

// aten::concat(Tensor[] tensors, int dim=0) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<concat::schema> create_concat_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(concat::name, concat::overload_name)
      .typed<concat::schema>();
}

// aten::concat(Tensor[] tensors, int dim=0) -> Tensor
at::Tensor concat::call(at::TensorList tensors, int64_t dim) {
    static auto op = create_concat_typed_handle();
    return op.call(tensors, dim);
}

// aten::concat(Tensor[] tensors, int dim=0) -> Tensor
at::Tensor concat::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, int64_t dim) {
    static auto op = create_concat_typed_handle();
    return op.redispatch(dispatchKeySet, tensors, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(chain_matmul, name, "aten::chain_matmul")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(chain_matmul, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(chain_matmul, schema_str, "chain_matmul(Tensor[] matrices) -> Tensor")

// aten::chain_matmul(Tensor[] matrices) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<chain_matmul::schema> create_chain_matmul_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(chain_matmul::name, chain_matmul::overload_name)
      .typed<chain_matmul::schema>();
}

// aten::chain_matmul(Tensor[] matrices) -> Tensor
at::Tensor chain_matmul::call(at::TensorList matrices) {
    static auto op = create_chain_matmul_typed_handle();
    return op.call(matrices);
}

// aten::chain_matmul(Tensor[] matrices) -> Tensor
at::Tensor chain_matmul::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList matrices) {
    static auto op = create_chain_matmul_typed_handle();
    return op.redispatch(dispatchKeySet, matrices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(chain_matmul_out, name, "aten::chain_matmul")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(chain_matmul_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(chain_matmul_out, schema_str, "chain_matmul.out(Tensor[] matrices, *, Tensor(a!) out) -> Tensor(a!)")

// aten::chain_matmul.out(Tensor[] matrices, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<chain_matmul_out::schema> create_chain_matmul_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(chain_matmul_out::name, chain_matmul_out::overload_name)
      .typed<chain_matmul_out::schema>();
}

// aten::chain_matmul.out(Tensor[] matrices, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & chain_matmul_out::call(at::TensorList matrices, at::Tensor & out) {
    static auto op = create_chain_matmul_out_typed_handle();
    return op.call(matrices, out);
}

// aten::chain_matmul.out(Tensor[] matrices, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & chain_matmul_out::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList matrices, at::Tensor & out) {
    static auto op = create_chain_matmul_out_typed_handle();
    return op.redispatch(dispatchKeySet, matrices, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_Tensor_out, name, "aten::clamp")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_Tensor_out, overload_name, "Tensor_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_Tensor_out, schema_str, "clamp.Tensor_out(Tensor self, Tensor? min=None, Tensor? max=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::clamp.Tensor_out(Tensor self, Tensor? min=None, Tensor? max=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<clamp_Tensor_out::schema> create_clamp_Tensor_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(clamp_Tensor_out::name, clamp_Tensor_out::overload_name)
      .typed<clamp_Tensor_out::schema>();
}

// aten::clamp.Tensor_out(Tensor self, Tensor? min=None, Tensor? max=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & clamp_Tensor_out::call(const at::Tensor & self, const c10::optional<at::Tensor> & min, const c10::optional<at::Tensor> & max, at::Tensor & out) {
    static auto op = create_clamp_Tensor_out_typed_handle();
    return op.call(self, min, max, out);
}

// aten::clamp.Tensor_out(Tensor self, Tensor? min=None, Tensor? max=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & clamp_Tensor_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Tensor> & min, const c10::optional<at::Tensor> & max, at::Tensor & out) {
    static auto op = create_clamp_Tensor_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, min, max, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_max_Tensor_out, name, "aten::clamp_max")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_max_Tensor_out, overload_name, "Tensor_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_max_Tensor_out, schema_str, "clamp_max.Tensor_out(Tensor self, Tensor max, *, Tensor(a!) out) -> Tensor(a!)")

// aten::clamp_max.Tensor_out(Tensor self, Tensor max, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<clamp_max_Tensor_out::schema> create_clamp_max_Tensor_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(clamp_max_Tensor_out::name, clamp_max_Tensor_out::overload_name)
      .typed<clamp_max_Tensor_out::schema>();
}

// aten::clamp_max.Tensor_out(Tensor self, Tensor max, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & clamp_max_Tensor_out::call(const at::Tensor & self, const at::Tensor & max, at::Tensor & out) {
    static auto op = create_clamp_max_Tensor_out_typed_handle();
    return op.call(self, max, out);
}

// aten::clamp_max.Tensor_out(Tensor self, Tensor max, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & clamp_max_Tensor_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & max, at::Tensor & out) {
    static auto op = create_clamp_max_Tensor_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, max, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_min, name, "aten::clamp_min")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_min, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_min, schema_str, "clamp_min(Tensor self, Scalar min) -> Tensor")

// aten::clamp_min(Tensor self, Scalar min) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<clamp_min::schema> create_clamp_min_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(clamp_min::name, clamp_min::overload_name)
      .typed<clamp_min::schema>();
}

// aten::clamp_min(Tensor self, Scalar min) -> Tensor
at::Tensor clamp_min::call(const at::Tensor & self, const at::Scalar & min) {
    static auto op = create_clamp_min_typed_handle();
    return op.call(self, min);
}

// aten::clamp_min(Tensor self, Scalar min) -> Tensor
at::Tensor clamp_min::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & min) {
    static auto op = create_clamp_min_typed_handle();
    return op.redispatch(dispatchKeySet, self, min);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_min_out, name, "aten::clamp_min")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_min_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clamp_min_out, schema_str, "clamp_min.out(Tensor self, Scalar min, *, Tensor(a!) out) -> Tensor(a!)")

// aten::clamp_min.out(Tensor self, Scalar min, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<clamp_min_out::schema> create_clamp_min_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(clamp_min_out::name, clamp_min_out::overload_name)
      .typed<clamp_min_out::schema>();
}

// aten::clamp_min.out(Tensor self, Scalar min, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & clamp_min_out::call(const at::Tensor & self, const at::Scalar & min, at::Tensor & out) {
    static auto op = create_clamp_min_out_typed_handle();
    return op.call(self, min, out);
}

// aten::clamp_min.out(Tensor self, Scalar min, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & clamp_min_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & min, at::Tensor & out) {
    static auto op = create_clamp_min_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, min, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clip__Tensor, name, "aten::clip_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clip__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clip__Tensor, schema_str, "clip_.Tensor(Tensor(a!) self, Tensor? min=None, Tensor? max=None) -> Tensor(a!)")

// aten::clip_.Tensor(Tensor(a!) self, Tensor? min=None, Tensor? max=None) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<clip__Tensor::schema> create_clip__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(clip__Tensor::name, clip__Tensor::overload_name)
      .typed<clip__Tensor::schema>();
}

// aten::clip_.Tensor(Tensor(a!) self, Tensor? min=None, Tensor? max=None) -> Tensor(a!)
at::Tensor & clip__Tensor::call(at::Tensor & self, const c10::optional<at::Tensor> & min, const c10::optional<at::Tensor> & max) {
    static auto op = create_clip__Tensor_typed_handle();
    return op.call(self, min, max);
}

// aten::clip_.Tensor(Tensor(a!) self, Tensor? min=None, Tensor? max=None) -> Tensor(a!)
at::Tensor & clip__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const c10::optional<at::Tensor> & min, const c10::optional<at::Tensor> & max) {
    static auto op = create_clip__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, min, max);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clip_out, name, "aten::clip")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clip_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(clip_out, schema_str, "clip.out(Tensor self, Scalar? min=None, Scalar? max=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::clip.out(Tensor self, Scalar? min=None, Scalar? max=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<clip_out::schema> create_clip_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(clip_out::name, clip_out::overload_name)
      .typed<clip_out::schema>();
}

// aten::clip.out(Tensor self, Scalar? min=None, Scalar? max=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & clip_out::call(const at::Tensor & self, const c10::optional<at::Scalar> & min, const c10::optional<at::Scalar> & max, at::Tensor & out) {
    static auto op = create_clip_out_typed_handle();
    return op.call(self, min, max, out);
}

// aten::clip.out(Tensor self, Scalar? min=None, Scalar? max=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & clip_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Scalar> & min, const c10::optional<at::Scalar> & max, at::Tensor & out) {
    static auto op = create_clip_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, min, max, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_convolution_mode, name, "aten::_convolution_mode")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_convolution_mode, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_convolution_mode, schema_str, "_convolution_mode(Tensor input, Tensor weight, Tensor? bias, int[] stride, str padding, int[] dilation, int groups) -> Tensor")

// aten::_convolution_mode(Tensor input, Tensor weight, Tensor? bias, int[] stride, str padding, int[] dilation, int groups) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_convolution_mode::schema> create__convolution_mode_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_convolution_mode::name, _convolution_mode::overload_name)
      .typed<_convolution_mode::schema>();
}

// aten::_convolution_mode(Tensor input, Tensor weight, Tensor? bias, int[] stride, str padding, int[] dilation, int groups) -> Tensor
at::Tensor _convolution_mode::call(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, c10::string_view padding, at::IntArrayRef dilation, int64_t groups) {
    static auto op = create__convolution_mode_typed_handle();
    return op.call(input, weight, bias, stride, padding, dilation, groups);
}

// aten::_convolution_mode(Tensor input, Tensor weight, Tensor? bias, int[] stride, str padding, int[] dilation, int groups) -> Tensor
at::Tensor _convolution_mode::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, c10::string_view padding, at::IntArrayRef dilation, int64_t groups) {
    static auto op = create__convolution_mode_typed_handle();
    return op.redispatch(dispatchKeySet, input, weight, bias, stride, padding, dilation, groups);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_convolution_nogroup, name, "aten::_convolution_nogroup")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_convolution_nogroup, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_convolution_nogroup, schema_str, "_convolution_nogroup(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding) -> Tensor")

// aten::_convolution_nogroup(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_convolution_nogroup::schema> create__convolution_nogroup_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_convolution_nogroup::name, _convolution_nogroup::overload_name)
      .typed<_convolution_nogroup::schema>();
}

// aten::_convolution_nogroup(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding) -> Tensor
at::Tensor _convolution_nogroup::call(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding) {
    static auto op = create__convolution_nogroup_typed_handle();
    return op.call(input, weight, bias, stride, padding, dilation, transposed, output_padding);
}

// aten::_convolution_nogroup(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding) -> Tensor
at::Tensor _convolution_nogroup::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding) {
    static auto op = create__convolution_nogroup_typed_handle();
    return op.redispatch(dispatchKeySet, input, weight, bias, stride, padding, dilation, transposed, output_padding);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv1d, name, "aten::conv1d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv1d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv1d, schema_str, "conv1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] dilation=1, int groups=1) -> Tensor")

// aten::conv1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] dilation=1, int groups=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<conv1d::schema> create_conv1d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(conv1d::name, conv1d::overload_name)
      .typed<conv1d::schema>();
}

// aten::conv1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] dilation=1, int groups=1) -> Tensor
at::Tensor conv1d::call(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
    static auto op = create_conv1d_typed_handle();
    return op.call(input, weight, bias, stride, padding, dilation, groups);
}

// aten::conv1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] dilation=1, int groups=1) -> Tensor
at::Tensor conv1d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
    static auto op = create_conv1d_typed_handle();
    return op.redispatch(dispatchKeySet, input, weight, bias, stride, padding, dilation, groups);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv3d, name, "aten::conv3d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv3d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv3d, schema_str, "conv3d(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] dilation=1, int groups=1) -> Tensor")

// aten::conv3d(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] dilation=1, int groups=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<conv3d::schema> create_conv3d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(conv3d::name, conv3d::overload_name)
      .typed<conv3d::schema>();
}

// aten::conv3d(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] dilation=1, int groups=1) -> Tensor
at::Tensor conv3d::call(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
    static auto op = create_conv3d_typed_handle();
    return op.call(input, weight, bias, stride, padding, dilation, groups);
}

// aten::conv3d(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] dilation=1, int groups=1) -> Tensor
at::Tensor conv3d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
    static auto op = create_conv3d_typed_handle();
    return op.redispatch(dispatchKeySet, input, weight, bias, stride, padding, dilation, groups);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv1d_padding, name, "aten::conv1d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv1d_padding, overload_name, "padding")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv1d_padding, schema_str, "conv1d.padding(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, str padding=\"valid\", int[1] dilation=1, int groups=1) -> Tensor")

// aten::conv1d.padding(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, str padding="valid", int[1] dilation=1, int groups=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<conv1d_padding::schema> create_conv1d_padding_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(conv1d_padding::name, conv1d_padding::overload_name)
      .typed<conv1d_padding::schema>();
}

// aten::conv1d.padding(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, str padding="valid", int[1] dilation=1, int groups=1) -> Tensor
at::Tensor conv1d_padding::call(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, c10::string_view padding, at::IntArrayRef dilation, int64_t groups) {
    static auto op = create_conv1d_padding_typed_handle();
    return op.call(input, weight, bias, stride, padding, dilation, groups);
}

// aten::conv1d.padding(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, str padding="valid", int[1] dilation=1, int groups=1) -> Tensor
at::Tensor conv1d_padding::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, c10::string_view padding, at::IntArrayRef dilation, int64_t groups) {
    static auto op = create_conv1d_padding_typed_handle();
    return op.redispatch(dispatchKeySet, input, weight, bias, stride, padding, dilation, groups);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv3d_padding, name, "aten::conv3d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv3d_padding, overload_name, "padding")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv3d_padding, schema_str, "conv3d.padding(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, str padding=\"valid\", int[3] dilation=1, int groups=1) -> Tensor")

// aten::conv3d.padding(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, str padding="valid", int[3] dilation=1, int groups=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<conv3d_padding::schema> create_conv3d_padding_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(conv3d_padding::name, conv3d_padding::overload_name)
      .typed<conv3d_padding::schema>();
}

// aten::conv3d.padding(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, str padding="valid", int[3] dilation=1, int groups=1) -> Tensor
at::Tensor conv3d_padding::call(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, c10::string_view padding, at::IntArrayRef dilation, int64_t groups) {
    static auto op = create_conv3d_padding_typed_handle();
    return op.call(input, weight, bias, stride, padding, dilation, groups);
}

// aten::conv3d.padding(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, str padding="valid", int[3] dilation=1, int groups=1) -> Tensor
at::Tensor conv3d_padding::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, c10::string_view padding, at::IntArrayRef dilation, int64_t groups) {
    static auto op = create_conv3d_padding_typed_handle();
    return op.redispatch(dispatchKeySet, input, weight, bias, stride, padding, dilation, groups);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv_tbc_backward, name, "aten::conv_tbc_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv_tbc_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv_tbc_backward, schema_str, "conv_tbc_backward(Tensor self, Tensor input, Tensor weight, Tensor bias, int pad) -> (Tensor, Tensor, Tensor)")

// aten::conv_tbc_backward(Tensor self, Tensor input, Tensor weight, Tensor bias, int pad) -> (Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<conv_tbc_backward::schema> create_conv_tbc_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(conv_tbc_backward::name, conv_tbc_backward::overload_name)
      .typed<conv_tbc_backward::schema>();
}

// aten::conv_tbc_backward(Tensor self, Tensor input, Tensor weight, Tensor bias, int pad) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> conv_tbc_backward::call(const at::Tensor & self, const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias, int64_t pad) {
    static auto op = create_conv_tbc_backward_typed_handle();
    return op.call(self, input, weight, bias, pad);
}

// aten::conv_tbc_backward(Tensor self, Tensor input, Tensor weight, Tensor bias, int pad) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> conv_tbc_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias, int64_t pad) {
    static auto op = create_conv_tbc_backward_typed_handle();
    return op.redispatch(dispatchKeySet, self, input, weight, bias, pad);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_copy_from_and_resize, name, "aten::_copy_from_and_resize")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_copy_from_and_resize, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_copy_from_and_resize, schema_str, "_copy_from_and_resize(Tensor self, Tensor dst) -> Tensor")

// aten::_copy_from_and_resize(Tensor self, Tensor dst) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_copy_from_and_resize::schema> create__copy_from_and_resize_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_copy_from_and_resize::name, _copy_from_and_resize::overload_name)
      .typed<_copy_from_and_resize::schema>();
}

// aten::_copy_from_and_resize(Tensor self, Tensor dst) -> Tensor
at::Tensor _copy_from_and_resize::call(const at::Tensor & self, const at::Tensor & dst) {
    static auto op = create__copy_from_and_resize_typed_handle();
    return op.call(self, dst);
}

// aten::_copy_from_and_resize(Tensor self, Tensor dst) -> Tensor
at::Tensor _copy_from_and_resize::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & dst) {
    static auto op = create__copy_from_and_resize_typed_handle();
    return op.redispatch(dispatchKeySet, self, dst);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cosh_, name, "aten::cosh_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cosh_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cosh_, schema_str, "cosh_(Tensor(a!) self) -> Tensor(a!)")

// aten::cosh_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<cosh_::schema> create_cosh__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cosh_::name, cosh_::overload_name)
      .typed<cosh_::schema>();
}

// aten::cosh_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & cosh_::call(at::Tensor & self) {
    static auto op = create_cosh__typed_handle();
    return op.call(self);
}

// aten::cosh_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & cosh_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_cosh__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cosh_out, name, "aten::cosh")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cosh_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cosh_out, schema_str, "cosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::cosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<cosh_out::schema> create_cosh_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cosh_out::name, cosh_out::overload_name)
      .typed<cosh_out::schema>();
}

// aten::cosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & cosh_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_cosh_out_typed_handle();
    return op.call(self, out);
}

// aten::cosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & cosh_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_cosh_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution, name, "aten::cudnn_convolution")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution, schema_str, "cudnn_convolution(Tensor self, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool allow_tf32) -> Tensor")

// aten::cudnn_convolution(Tensor self, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool allow_tf32) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<cudnn_convolution::schema> create_cudnn_convolution_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cudnn_convolution::name, cudnn_convolution::overload_name)
      .typed<cudnn_convolution::schema>();
}

// aten::cudnn_convolution(Tensor self, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool allow_tf32) -> Tensor
at::Tensor cudnn_convolution::call(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
    static auto op = create_cudnn_convolution_typed_handle();
    return op.call(self, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
}

// aten::cudnn_convolution(Tensor self, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool allow_tf32) -> Tensor
at::Tensor cudnn_convolution::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
    static auto op = create_cudnn_convolution_typed_handle();
    return op.redispatch(dispatchKeySet, self, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_backward_weight, name, "aten::cudnn_convolution_backward_weight")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_backward_weight, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_backward_weight, schema_str, "cudnn_convolution_backward_weight(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool allow_tf32) -> Tensor")

// aten::cudnn_convolution_backward_weight(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool allow_tf32) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<cudnn_convolution_backward_weight::schema> create_cudnn_convolution_backward_weight_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cudnn_convolution_backward_weight::name, cudnn_convolution_backward_weight::overload_name)
      .typed<cudnn_convolution_backward_weight::schema>();
}

// aten::cudnn_convolution_backward_weight(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool allow_tf32) -> Tensor
at::Tensor cudnn_convolution_backward_weight::call(at::IntArrayRef weight_size, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
    static auto op = create_cudnn_convolution_backward_weight_typed_handle();
    return op.call(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
}

// aten::cudnn_convolution_backward_weight(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool allow_tf32) -> Tensor
at::Tensor cudnn_convolution_backward_weight::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef weight_size, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
    static auto op = create_cudnn_convolution_backward_weight_typed_handle();
    return op.redispatch(dispatchKeySet, weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_transpose_backward_weight, name, "aten::cudnn_convolution_transpose_backward_weight")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_transpose_backward_weight, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_transpose_backward_weight, schema_str, "cudnn_convolution_transpose_backward_weight(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool allow_tf32) -> Tensor")

// aten::cudnn_convolution_transpose_backward_weight(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool allow_tf32) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<cudnn_convolution_transpose_backward_weight::schema> create_cudnn_convolution_transpose_backward_weight_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cudnn_convolution_transpose_backward_weight::name, cudnn_convolution_transpose_backward_weight::overload_name)
      .typed<cudnn_convolution_transpose_backward_weight::schema>();
}

// aten::cudnn_convolution_transpose_backward_weight(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool allow_tf32) -> Tensor
at::Tensor cudnn_convolution_transpose_backward_weight::call(at::IntArrayRef weight_size, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
    static auto op = create_cudnn_convolution_transpose_backward_weight_typed_handle();
    return op.call(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
}

// aten::cudnn_convolution_transpose_backward_weight(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool allow_tf32) -> Tensor
at::Tensor cudnn_convolution_transpose_backward_weight::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef weight_size, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
    static auto op = create_cudnn_convolution_transpose_backward_weight_typed_handle();
    return op.redispatch(dispatchKeySet, weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_relu, name, "aten::cudnn_convolution_relu")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_relu, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cudnn_convolution_relu, schema_str, "cudnn_convolution_relu(Tensor self, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor")

// aten::cudnn_convolution_relu(Tensor self, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<cudnn_convolution_relu::schema> create_cudnn_convolution_relu_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cudnn_convolution_relu::name, cudnn_convolution_relu::overload_name)
      .typed<cudnn_convolution_relu::schema>();
}

// aten::cudnn_convolution_relu(Tensor self, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor
at::Tensor cudnn_convolution_relu::call(const at::Tensor & self, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
    static auto op = create_cudnn_convolution_relu_typed_handle();
    return op.call(self, weight, bias, stride, padding, dilation, groups);
}

// aten::cudnn_convolution_relu(Tensor self, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor
at::Tensor cudnn_convolution_relu::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
    static auto op = create_cudnn_convolution_relu_typed_handle();
    return op.redispatch(dispatchKeySet, self, weight, bias, stride, padding, dilation, groups);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cummax_out, name, "aten::cummax")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cummax_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cummax_out, schema_str, "cummax.out(Tensor self, int dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)")

// aten::cummax.out(Tensor self, int dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
static C10_NOINLINE c10::TypedOperatorHandle<cummax_out::schema> create_cummax_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cummax_out::name, cummax_out::overload_name)
      .typed<cummax_out::schema>();
}

// aten::cummax.out(Tensor self, int dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> cummax_out::call(const at::Tensor & self, int64_t dim, at::Tensor & values, at::Tensor & indices) {
    static auto op = create_cummax_out_typed_handle();
    return op.call(self, dim, values, indices);
}

// aten::cummax.out(Tensor self, int dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> cummax_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, at::Tensor & values, at::Tensor & indices) {
    static auto op = create_cummax_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, values, indices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cummax_dimname, name, "aten::cummax")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cummax_dimname, overload_name, "dimname")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cummax_dimname, schema_str, "cummax.dimname(Tensor self, Dimname dim) -> (Tensor values, Tensor indices)")

// aten::cummax.dimname(Tensor self, Dimname dim) -> (Tensor values, Tensor indices)
static C10_NOINLINE c10::TypedOperatorHandle<cummax_dimname::schema> create_cummax_dimname_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cummax_dimname::name, cummax_dimname::overload_name)
      .typed<cummax_dimname::schema>();
}

// aten::cummax.dimname(Tensor self, Dimname dim) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> cummax_dimname::call(const at::Tensor & self, at::Dimname dim) {
    static auto op = create_cummax_dimname_typed_handle();
    return op.call(self, dim);
}

// aten::cummax.dimname(Tensor self, Dimname dim) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> cummax_dimname::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim) {
    static auto op = create_cummax_dimname_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cummax_dimname_out, name, "aten::cummax")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cummax_dimname_out, overload_name, "dimname_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cummax_dimname_out, schema_str, "cummax.dimname_out(Tensor self, Dimname dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)")

// aten::cummax.dimname_out(Tensor self, Dimname dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
static C10_NOINLINE c10::TypedOperatorHandle<cummax_dimname_out::schema> create_cummax_dimname_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cummax_dimname_out::name, cummax_dimname_out::overload_name)
      .typed<cummax_dimname_out::schema>();
}

// aten::cummax.dimname_out(Tensor self, Dimname dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> cummax_dimname_out::call(const at::Tensor & self, at::Dimname dim, at::Tensor & values, at::Tensor & indices) {
    static auto op = create_cummax_dimname_out_typed_handle();
    return op.call(self, dim, values, indices);
}

// aten::cummax.dimname_out(Tensor self, Dimname dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> cummax_dimname_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, at::Tensor & values, at::Tensor & indices) {
    static auto op = create_cummax_dimname_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, values, indices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cummin_dimname, name, "aten::cummin")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cummin_dimname, overload_name, "dimname")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cummin_dimname, schema_str, "cummin.dimname(Tensor self, Dimname dim) -> (Tensor values, Tensor indices)")

// aten::cummin.dimname(Tensor self, Dimname dim) -> (Tensor values, Tensor indices)
static C10_NOINLINE c10::TypedOperatorHandle<cummin_dimname::schema> create_cummin_dimname_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cummin_dimname::name, cummin_dimname::overload_name)
      .typed<cummin_dimname::schema>();
}

// aten::cummin.dimname(Tensor self, Dimname dim) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> cummin_dimname::call(const at::Tensor & self, at::Dimname dim) {
    static auto op = create_cummin_dimname_typed_handle();
    return op.call(self, dim);
}

// aten::cummin.dimname(Tensor self, Dimname dim) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> cummin_dimname::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim) {
    static auto op = create_cummin_dimname_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumprod, name, "aten::cumprod")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumprod, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumprod, schema_str, "cumprod(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor")

// aten::cumprod(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<cumprod::schema> create_cumprod_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cumprod::name, cumprod::overload_name)
      .typed<cumprod::schema>();
}

// aten::cumprod(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
at::Tensor cumprod::call(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_cumprod_typed_handle();
    return op.call(self, dim, dtype);
}

// aten::cumprod(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
at::Tensor cumprod::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_cumprod_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumsum_dimname, name, "aten::cumsum")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumsum_dimname, overload_name, "dimname")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumsum_dimname, schema_str, "cumsum.dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor")

// aten::cumsum.dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<cumsum_dimname::schema> create_cumsum_dimname_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cumsum_dimname::name, cumsum_dimname::overload_name)
      .typed<cumsum_dimname::schema>();
}

// aten::cumsum.dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
at::Tensor cumsum_dimname::call(const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_cumsum_dimname_typed_handle();
    return op.call(self, dim, dtype);
}

// aten::cumsum.dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
at::Tensor cumsum_dimname::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_cumsum_dimname_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumsum__dimname, name, "aten::cumsum_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumsum__dimname, overload_name, "dimname")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumsum__dimname, schema_str, "cumsum_.dimname(Tensor(a!) self, Dimname dim, *, ScalarType? dtype=None) -> Tensor(a!)")

// aten::cumsum_.dimname(Tensor(a!) self, Dimname dim, *, ScalarType? dtype=None) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<cumsum__dimname::schema> create_cumsum__dimname_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cumsum__dimname::name, cumsum__dimname::overload_name)
      .typed<cumsum__dimname::schema>();
}

// aten::cumsum_.dimname(Tensor(a!) self, Dimname dim, *, ScalarType? dtype=None) -> Tensor(a!)
at::Tensor & cumsum__dimname::call(at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_cumsum__dimname_typed_handle();
    return op.call(self, dim, dtype);
}

// aten::cumsum_.dimname(Tensor(a!) self, Dimname dim, *, ScalarType? dtype=None) -> Tensor(a!)
at::Tensor & cumsum__dimname::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, at::Dimname dim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_cumsum__dimname_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumulative_trapezoid_x, name, "aten::cumulative_trapezoid")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumulative_trapezoid_x, overload_name, "x")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cumulative_trapezoid_x, schema_str, "cumulative_trapezoid.x(Tensor y, Tensor x, *, int dim=-1) -> Tensor")

// aten::cumulative_trapezoid.x(Tensor y, Tensor x, *, int dim=-1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<cumulative_trapezoid_x::schema> create_cumulative_trapezoid_x_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cumulative_trapezoid_x::name, cumulative_trapezoid_x::overload_name)
      .typed<cumulative_trapezoid_x::schema>();
}

// aten::cumulative_trapezoid.x(Tensor y, Tensor x, *, int dim=-1) -> Tensor
at::Tensor cumulative_trapezoid_x::call(const at::Tensor & y, const at::Tensor & x, int64_t dim) {
    static auto op = create_cumulative_trapezoid_x_typed_handle();
    return op.call(y, x, dim);
}

// aten::cumulative_trapezoid.x(Tensor y, Tensor x, *, int dim=-1) -> Tensor
at::Tensor cumulative_trapezoid_x::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & y, const at::Tensor & x, int64_t dim) {
    static auto op = create_cumulative_trapezoid_x_typed_handle();
    return op.redispatch(dispatchKeySet, y, x, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(diag_embed, name, "aten::diag_embed")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(diag_embed, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(diag_embed, schema_str, "diag_embed(Tensor self, int offset=0, int dim1=-2, int dim2=-1) -> Tensor")

// aten::diag_embed(Tensor self, int offset=0, int dim1=-2, int dim2=-1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<diag_embed::schema> create_diag_embed_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(diag_embed::name, diag_embed::overload_name)
      .typed<diag_embed::schema>();
}

// aten::diag_embed(Tensor self, int offset=0, int dim1=-2, int dim2=-1) -> Tensor
at::Tensor diag_embed::call(const at::Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) {
    static auto op = create_diag_embed_typed_handle();
    return op.call(self, offset, dim1, dim2);
}

// aten::diag_embed(Tensor self, int offset=0, int dim1=-2, int dim2=-1) -> Tensor
at::Tensor diag_embed::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) {
    static auto op = create_diag_embed_typed_handle();
    return op.redispatch(dispatchKeySet, self, offset, dim1, dim2);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(diagonal, name, "aten::diagonal")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(diagonal, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(diagonal, schema_str, "diagonal(Tensor(a) self, int offset=0, int dim1=0, int dim2=1) -> Tensor(a)")

// aten::diagonal(Tensor(a) self, int offset=0, int dim1=0, int dim2=1) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<diagonal::schema> create_diagonal_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(diagonal::name, diagonal::overload_name)
      .typed<diagonal::schema>();
}

// aten::diagonal(Tensor(a) self, int offset=0, int dim1=0, int dim2=1) -> Tensor(a)
at::Tensor diagonal::call(const at::Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) {
    static auto op = create_diagonal_typed_handle();
    return op.call(self, offset, dim1, dim2);
}

// aten::diagonal(Tensor(a) self, int offset=0, int dim1=0, int dim2=1) -> Tensor(a)
at::Tensor diagonal::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) {
    static auto op = create_diagonal_typed_handle();
    return op.redispatch(dispatchKeySet, self, offset, dim1, dim2);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fill_diagonal_, name, "aten::fill_diagonal_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fill_diagonal_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fill_diagonal_, schema_str, "fill_diagonal_(Tensor(a!) self, Scalar fill_value, bool wrap=False) -> Tensor(a!)")

// aten::fill_diagonal_(Tensor(a!) self, Scalar fill_value, bool wrap=False) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<fill_diagonal_::schema> create_fill_diagonal__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fill_diagonal_::name, fill_diagonal_::overload_name)
      .typed<fill_diagonal_::schema>();
}

// aten::fill_diagonal_(Tensor(a!) self, Scalar fill_value, bool wrap=False) -> Tensor(a!)
at::Tensor & fill_diagonal_::call(at::Tensor & self, const at::Scalar & fill_value, bool wrap) {
    static auto op = create_fill_diagonal__typed_handle();
    return op.call(self, fill_value, wrap);
}

// aten::fill_diagonal_(Tensor(a!) self, Scalar fill_value, bool wrap=False) -> Tensor(a!)
at::Tensor & fill_diagonal_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & fill_value, bool wrap) {
    static auto op = create_fill_diagonal__typed_handle();
    return op.redispatch(dispatchKeySet, self, fill_value, wrap);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gradient_scalarrayint, name, "aten::gradient")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gradient_scalarrayint, overload_name, "scalarrayint")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gradient_scalarrayint, schema_str, "gradient.scalarrayint(Tensor self, *, Scalar[] spacing, int? dim=None, int edge_order=1) -> Tensor[]")

// aten::gradient.scalarrayint(Tensor self, *, Scalar[] spacing, int? dim=None, int edge_order=1) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<gradient_scalarrayint::schema> create_gradient_scalarrayint_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(gradient_scalarrayint::name, gradient_scalarrayint::overload_name)
      .typed<gradient_scalarrayint::schema>();
}

// aten::gradient.scalarrayint(Tensor self, *, Scalar[] spacing, int? dim=None, int edge_order=1) -> Tensor[]
::std::vector<at::Tensor> gradient_scalarrayint::call(const at::Tensor & self, at::ArrayRef<at::Scalar> spacing, c10::optional<int64_t> dim, int64_t edge_order) {
    static auto op = create_gradient_scalarrayint_typed_handle();
    return op.call(self, spacing, dim, edge_order);
}

// aten::gradient.scalarrayint(Tensor self, *, Scalar[] spacing, int? dim=None, int edge_order=1) -> Tensor[]
::std::vector<at::Tensor> gradient_scalarrayint::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::ArrayRef<at::Scalar> spacing, c10::optional<int64_t> dim, int64_t edge_order) {
    static auto op = create_gradient_scalarrayint_typed_handle();
    return op.redispatch(dispatchKeySet, self, spacing, dim, edge_order);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(div__Scalar, name, "aten::div_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(div__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(div__Scalar, schema_str, "div_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)")

// aten::div_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<div__Scalar::schema> create_div__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(div__Scalar::name, div__Scalar::overload_name)
      .typed<div__Scalar::schema>();
}

// aten::div_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & div__Scalar::call(at::Tensor & self, const at::Scalar & other) {
    static auto op = create_div__Scalar_typed_handle();
    return op.call(self, other);
}

// aten::div_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & div__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
    static auto op = create_div__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(divide__Scalar, name, "aten::divide_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(divide__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(divide__Scalar, schema_str, "divide_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)")

// aten::divide_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<divide__Scalar::schema> create_divide__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(divide__Scalar::name, divide__Scalar::overload_name)
      .typed<divide__Scalar::schema>();
}

// aten::divide_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & divide__Scalar::call(at::Tensor & self, const at::Scalar & other) {
    static auto op = create_divide__Scalar_typed_handle();
    return op.call(self, other);
}

// aten::divide_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & divide__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
    static auto op = create_divide__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(true_divide_out, name, "aten::true_divide")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(true_divide_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(true_divide_out, schema_str, "true_divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::true_divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<true_divide_out::schema> create_true_divide_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(true_divide_out::name, true_divide_out::overload_name)
      .typed<true_divide_out::schema>();
}

// aten::true_divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & true_divide_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_true_divide_out_typed_handle();
    return op.call(self, other, out);
}

// aten::true_divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & true_divide_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_true_divide_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(embedding_renorm_, name, "aten::embedding_renorm_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(embedding_renorm_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(embedding_renorm_, schema_str, "embedding_renorm_(Tensor(a!) self, Tensor indices, float max_norm, float norm_type) -> Tensor(a!)")

// aten::embedding_renorm_(Tensor(a!) self, Tensor indices, float max_norm, float norm_type) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<embedding_renorm_::schema> create_embedding_renorm__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(embedding_renorm_::name, embedding_renorm_::overload_name)
      .typed<embedding_renorm_::schema>();
}

// aten::embedding_renorm_(Tensor(a!) self, Tensor indices, float max_norm, float norm_type) -> Tensor(a!)
at::Tensor & embedding_renorm_::call(at::Tensor & self, const at::Tensor & indices, double max_norm, double norm_type) {
    static auto op = create_embedding_renorm__typed_handle();
    return op.call(self, indices, max_norm, norm_type);
}

// aten::embedding_renorm_(Tensor(a!) self, Tensor indices, float max_norm, float norm_type) -> Tensor(a!)
at::Tensor & embedding_renorm_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & indices, double max_norm, double norm_type) {
    static auto op = create_embedding_renorm__typed_handle();
    return op.redispatch(dispatchKeySet, self, indices, max_norm, norm_type);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(embedding_bag_padding_idx, name, "aten::embedding_bag")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(embedding_bag_padding_idx, overload_name, "padding_idx")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(embedding_bag_padding_idx, schema_str, "embedding_bag.padding_idx(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq, int mode, bool sparse, Tensor? per_sample_weights, bool include_last_offset, int? padding_idx) -> (Tensor, Tensor, Tensor, Tensor)")

// aten::embedding_bag.padding_idx(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq, int mode, bool sparse, Tensor? per_sample_weights, bool include_last_offset, int? padding_idx) -> (Tensor, Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<embedding_bag_padding_idx::schema> create_embedding_bag_padding_idx_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(embedding_bag_padding_idx::name, embedding_bag_padding_idx::overload_name)
      .typed<embedding_bag_padding_idx::schema>();
}

// aten::embedding_bag.padding_idx(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq, int mode, bool sparse, Tensor? per_sample_weights, bool include_last_offset, int? padding_idx) -> (Tensor, Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> embedding_bag_padding_idx::call(const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<at::Tensor> & per_sample_weights, bool include_last_offset, c10::optional<int64_t> padding_idx) {
    static auto op = create_embedding_bag_padding_idx_typed_handle();
    return op.call(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx);
}

// aten::embedding_bag.padding_idx(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq, int mode, bool sparse, Tensor? per_sample_weights, bool include_last_offset, int? padding_idx) -> (Tensor, Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> embedding_bag_padding_idx::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<at::Tensor> & per_sample_weights, bool include_last_offset, c10::optional<int64_t> padding_idx) {
    static auto op = create_embedding_bag_padding_idx_typed_handle();
    return op.redispatch(dispatchKeySet, weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_empty_affine_quantized, name, "aten::_empty_affine_quantized")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_empty_affine_quantized, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_empty_affine_quantized, schema_str, "_empty_affine_quantized(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, float scale=1, int zero_point=0, MemoryFormat? memory_format=contiguous_format) -> Tensor")

// aten::_empty_affine_quantized(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, float scale=1, int zero_point=0, MemoryFormat? memory_format=contiguous_format) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_empty_affine_quantized::schema> create__empty_affine_quantized_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_empty_affine_quantized::name, _empty_affine_quantized::overload_name)
      .typed<_empty_affine_quantized::schema>();
}

// aten::_empty_affine_quantized(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, float scale=1, int zero_point=0, MemoryFormat? memory_format=contiguous_format) -> Tensor
at::Tensor _empty_affine_quantized::call(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, double scale, int64_t zero_point, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create__empty_affine_quantized_typed_handle();
    return op.call(size, dtype, layout, device, pin_memory, scale, zero_point, memory_format);
}

// aten::_empty_affine_quantized(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, float scale=1, int zero_point=0, MemoryFormat? memory_format=contiguous_format) -> Tensor
at::Tensor _empty_affine_quantized::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, double scale, int64_t zero_point, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create__empty_affine_quantized_typed_handle();
    return op.redispatch(dispatchKeySet, size, dtype, layout, device, pin_memory, scale, zero_point, memory_format);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(empty_like, name, "aten::empty_like")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(empty_like, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(empty_like, schema_str, "empty_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor")

// aten::empty_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<empty_like::schema> create_empty_like_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(empty_like::name, empty_like::overload_name)
      .typed<empty_like::schema>();
}

// aten::empty_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
at::Tensor empty_like::call(const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create_empty_like_typed_handle();
    return op.call(self, dtype, layout, device, pin_memory, memory_format);
}

// aten::empty_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
at::Tensor empty_like::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create_empty_like_typed_handle();
    return op.redispatch(dispatchKeySet, self, dtype, layout, device, pin_memory, memory_format);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(erf_, name, "aten::erf_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(erf_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(erf_, schema_str, "erf_(Tensor(a!) self) -> Tensor(a!)")

// aten::erf_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<erf_::schema> create_erf__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(erf_::name, erf_::overload_name)
      .typed<erf_::schema>();
}

// aten::erf_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & erf_::call(at::Tensor & self) {
    static auto op = create_erf__typed_handle();
    return op.call(self);
}

// aten::erf_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & erf_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_erf__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(exp_out, name, "aten::exp")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(exp_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(exp_out, schema_str, "exp.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::exp.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<exp_out::schema> create_exp_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(exp_out::name, exp_out::overload_name)
      .typed<exp_out::schema>();
}

// aten::exp.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & exp_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_exp_out_typed_handle();
    return op.call(self, out);
}

// aten::exp.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & exp_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_exp_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(exp2_, name, "aten::exp2_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(exp2_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(exp2_, schema_str, "exp2_(Tensor(a!) self) -> Tensor(a!)")

// aten::exp2_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<exp2_::schema> create_exp2__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(exp2_::name, exp2_::overload_name)
      .typed<exp2_::schema>();
}

// aten::exp2_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & exp2_::call(at::Tensor & self) {
    static auto op = create_exp2__typed_handle();
    return op.call(self);
}

// aten::exp2_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & exp2_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_exp2__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(exp2_out, name, "aten::exp2")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(exp2_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(exp2_out, schema_str, "exp2.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::exp2.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<exp2_out::schema> create_exp2_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(exp2_out::name, exp2_out::overload_name)
      .typed<exp2_out::schema>();
}

// aten::exp2.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & exp2_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_exp2_out_typed_handle();
    return op.call(self, out);
}

// aten::exp2.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & exp2_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_exp2_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(expand, name, "aten::expand")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(expand, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(expand, schema_str, "expand(Tensor(a) self, int[] size, *, bool implicit=False) -> Tensor(a)")

// aten::expand(Tensor(a) self, int[] size, *, bool implicit=False) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<expand::schema> create_expand_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(expand::name, expand::overload_name)
      .typed<expand::schema>();
}

// aten::expand(Tensor(a) self, int[] size, *, bool implicit=False) -> Tensor(a)
at::Tensor expand::call(const at::Tensor & self, at::IntArrayRef size, bool implicit) {
    static auto op = create_expand_typed_handle();
    return op.call(self, size, implicit);
}

// aten::expand(Tensor(a) self, int[] size, *, bool implicit=False) -> Tensor(a)
at::Tensor expand::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef size, bool implicit) {
    static auto op = create_expand_typed_handle();
    return op.redispatch(dispatchKeySet, self, size, implicit);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(eye_out, name, "aten::eye")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(eye_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(eye_out, schema_str, "eye.out(int n, *, Tensor(a!) out) -> Tensor(a!)")

// aten::eye.out(int n, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<eye_out::schema> create_eye_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(eye_out::name, eye_out::overload_name)
      .typed<eye_out::schema>();
}

// aten::eye.out(int n, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & eye_out::call(int64_t n, at::Tensor & out) {
    static auto op = create_eye_out_typed_handle();
    return op.call(n, out);
}

// aten::eye.out(int n, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & eye_out::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t n, at::Tensor & out) {
    static auto op = create_eye_out_typed_handle();
    return op.redispatch(dispatchKeySet, n, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(flatten_named_out_dim, name, "aten::flatten")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(flatten_named_out_dim, overload_name, "named_out_dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(flatten_named_out_dim, schema_str, "flatten.named_out_dim(Tensor(a) self, int start_dim, int end_dim, Dimname out_dim) -> Tensor(a)")

// aten::flatten.named_out_dim(Tensor(a) self, int start_dim, int end_dim, Dimname out_dim) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<flatten_named_out_dim::schema> create_flatten_named_out_dim_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(flatten_named_out_dim::name, flatten_named_out_dim::overload_name)
      .typed<flatten_named_out_dim::schema>();
}

// aten::flatten.named_out_dim(Tensor(a) self, int start_dim, int end_dim, Dimname out_dim) -> Tensor(a)
at::Tensor flatten_named_out_dim::call(const at::Tensor & self, int64_t start_dim, int64_t end_dim, at::Dimname out_dim) {
    static auto op = create_flatten_named_out_dim_typed_handle();
    return op.call(self, start_dim, end_dim, out_dim);
}

// aten::flatten.named_out_dim(Tensor(a) self, int start_dim, int end_dim, Dimname out_dim) -> Tensor(a)
at::Tensor flatten_named_out_dim::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t start_dim, int64_t end_dim, at::Dimname out_dim) {
    static auto op = create_flatten_named_out_dim_typed_handle();
    return op.redispatch(dispatchKeySet, self, start_dim, end_dim, out_dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(floor, name, "aten::floor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(floor, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(floor, schema_str, "floor(Tensor self) -> Tensor")

// aten::floor(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<floor::schema> create_floor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(floor::name, floor::overload_name)
      .typed<floor::schema>();
}

// aten::floor(Tensor self) -> Tensor
at::Tensor floor::call(const at::Tensor & self) {
    static auto op = create_floor_typed_handle();
    return op.call(self);
}

// aten::floor(Tensor self) -> Tensor
at::Tensor floor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_floor_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(floor_out, name, "aten::floor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(floor_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(floor_out, schema_str, "floor.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::floor.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<floor_out::schema> create_floor_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(floor_out::name, floor_out::overload_name)
      .typed<floor_out::schema>();
}

// aten::floor.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & floor_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_floor_out_typed_handle();
    return op.call(self, out);
}

// aten::floor.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & floor_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_floor_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(frac_out, name, "aten::frac")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(frac_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(frac_out, schema_str, "frac.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::frac.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<frac_out::schema> create_frac_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(frac_out::name, frac_out::overload_name)
      .typed<frac_out::schema>();
}

// aten::frac.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & frac_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_frac_out_typed_handle();
    return op.call(self, out);
}

// aten::frac.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & frac_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_frac_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(grid_sampler_3d_backward, name, "aten::grid_sampler_3d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(grid_sampler_3d_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(grid_sampler_3d_backward, schema_str, "grid_sampler_3d_backward(Tensor grad_output, Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> (Tensor, Tensor)")

// aten::grid_sampler_3d_backward(Tensor grad_output, Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<grid_sampler_3d_backward::schema> create_grid_sampler_3d_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(grid_sampler_3d_backward::name, grid_sampler_3d_backward::overload_name)
      .typed<grid_sampler_3d_backward::schema>();
}

// aten::grid_sampler_3d_backward(Tensor grad_output, Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> grid_sampler_3d_backward::call(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
    static auto op = create_grid_sampler_3d_backward_typed_handle();
    return op.call(grad_output, input, grid, interpolation_mode, padding_mode, align_corners);
}

// aten::grid_sampler_3d_backward(Tensor grad_output, Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> grid_sampler_3d_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
    static auto op = create_grid_sampler_3d_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, input, grid, interpolation_mode, padding_mode, align_corners);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hamming_window_periodic, name, "aten::hamming_window")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hamming_window_periodic, overload_name, "periodic")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hamming_window_periodic, schema_str, "hamming_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::hamming_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<hamming_window_periodic::schema> create_hamming_window_periodic_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(hamming_window_periodic::name, hamming_window_periodic::overload_name)
      .typed<hamming_window_periodic::schema>();
}

// aten::hamming_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor hamming_window_periodic::call(int64_t window_length, bool periodic, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_hamming_window_periodic_typed_handle();
    return op.call(window_length, periodic, dtype, layout, device, pin_memory);
}

// aten::hamming_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor hamming_window_periodic::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t window_length, bool periodic, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_hamming_window_periodic_typed_handle();
    return op.redispatch(dispatchKeySet, window_length, periodic, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(kaiser_window_periodic, name, "aten::kaiser_window")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(kaiser_window_periodic, overload_name, "periodic")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(kaiser_window_periodic, schema_str, "kaiser_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::kaiser_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<kaiser_window_periodic::schema> create_kaiser_window_periodic_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(kaiser_window_periodic::name, kaiser_window_periodic::overload_name)
      .typed<kaiser_window_periodic::schema>();
}

// aten::kaiser_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor kaiser_window_periodic::call(int64_t window_length, bool periodic, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_kaiser_window_periodic_typed_handle();
    return op.call(window_length, periodic, dtype, layout, device, pin_memory);
}

// aten::kaiser_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor kaiser_window_periodic::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t window_length, bool periodic, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_kaiser_window_periodic_typed_handle();
    return op.redispatch(dispatchKeySet, window_length, periodic, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hinge_embedding_loss, name, "aten::hinge_embedding_loss")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hinge_embedding_loss, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hinge_embedding_loss, schema_str, "hinge_embedding_loss(Tensor self, Tensor target, float margin=1.0, int reduction=Mean) -> Tensor")

// aten::hinge_embedding_loss(Tensor self, Tensor target, float margin=1.0, int reduction=Mean) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<hinge_embedding_loss::schema> create_hinge_embedding_loss_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(hinge_embedding_loss::name, hinge_embedding_loss::overload_name)
      .typed<hinge_embedding_loss::schema>();
}

// aten::hinge_embedding_loss(Tensor self, Tensor target, float margin=1.0, int reduction=Mean) -> Tensor
at::Tensor hinge_embedding_loss::call(const at::Tensor & self, const at::Tensor & target, double margin, int64_t reduction) {
    static auto op = create_hinge_embedding_loss_typed_handle();
    return op.call(self, target, margin, reduction);
}

// aten::hinge_embedding_loss(Tensor self, Tensor target, float margin=1.0, int reduction=Mean) -> Tensor
at::Tensor hinge_embedding_loss::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, double margin, int64_t reduction) {
    static auto op = create_hinge_embedding_loss_typed_handle();
    return op.redispatch(dispatchKeySet, self, target, margin, reduction);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(native_group_norm, name, "aten::native_group_norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(native_group_norm, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(native_group_norm, schema_str, "native_group_norm(Tensor input, Tensor? weight, Tensor? bias, int N, int C, int HxW, int group, float eps) -> (Tensor, Tensor, Tensor)")

// aten::native_group_norm(Tensor input, Tensor? weight, Tensor? bias, int N, int C, int HxW, int group, float eps) -> (Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<native_group_norm::schema> create_native_group_norm_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(native_group_norm::name, native_group_norm::overload_name)
      .typed<native_group_norm::schema>();
}

// aten::native_group_norm(Tensor input, Tensor? weight, Tensor? bias, int N, int C, int HxW, int group, float eps) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_group_norm::call(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, int64_t N, int64_t C, int64_t HxW, int64_t group, double eps) {
    static auto op = create_native_group_norm_typed_handle();
    return op.call(input, weight, bias, N, C, HxW, group, eps);
}

// aten::native_group_norm(Tensor input, Tensor? weight, Tensor? bias, int N, int C, int HxW, int group, float eps) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_group_norm::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, int64_t N, int64_t C, int64_t HxW, int64_t group, double eps) {
    static auto op = create_native_group_norm_typed_handle();
    return op.redispatch(dispatchKeySet, input, weight, bias, N, C, HxW, group, eps);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fft_r2c, name, "aten::_fft_r2c")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fft_r2c, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fft_r2c, schema_str, "_fft_r2c(Tensor self, int[] dim, int normalization, bool onesided) -> Tensor")

// aten::_fft_r2c(Tensor self, int[] dim, int normalization, bool onesided) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_fft_r2c::schema> create__fft_r2c_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_fft_r2c::name, _fft_r2c::overload_name)
      .typed<_fft_r2c::schema>();
}

// aten::_fft_r2c(Tensor self, int[] dim, int normalization, bool onesided) -> Tensor
at::Tensor _fft_r2c::call(const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, bool onesided) {
    static auto op = create__fft_r2c_typed_handle();
    return op.call(self, dim, normalization, onesided);
}

// aten::_fft_r2c(Tensor self, int[] dim, int normalization, bool onesided) -> Tensor
at::Tensor _fft_r2c::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, bool onesided) {
    static auto op = create__fft_r2c_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, normalization, onesided);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isin_Tensor_Scalar, name, "aten::isin")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isin_Tensor_Scalar, overload_name, "Tensor_Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isin_Tensor_Scalar, schema_str, "isin.Tensor_Scalar(Tensor elements, Scalar test_element, *, bool assume_unique=False, bool invert=False) -> Tensor")

// aten::isin.Tensor_Scalar(Tensor elements, Scalar test_element, *, bool assume_unique=False, bool invert=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<isin_Tensor_Scalar::schema> create_isin_Tensor_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(isin_Tensor_Scalar::name, isin_Tensor_Scalar::overload_name)
      .typed<isin_Tensor_Scalar::schema>();
}

// aten::isin.Tensor_Scalar(Tensor elements, Scalar test_element, *, bool assume_unique=False, bool invert=False) -> Tensor
at::Tensor isin_Tensor_Scalar::call(const at::Tensor & elements, const at::Scalar & test_element, bool assume_unique, bool invert) {
    static auto op = create_isin_Tensor_Scalar_typed_handle();
    return op.call(elements, test_element, assume_unique, invert);
}

// aten::isin.Tensor_Scalar(Tensor elements, Scalar test_element, *, bool assume_unique=False, bool invert=False) -> Tensor
at::Tensor isin_Tensor_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & elements, const at::Scalar & test_element, bool assume_unique, bool invert) {
    static auto op = create_isin_Tensor_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, elements, test_element, assume_unique, invert);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_neg, name, "aten::is_neg")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_neg, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_neg, schema_str, "is_neg(Tensor self) -> bool")

// aten::is_neg(Tensor self) -> bool
static C10_NOINLINE c10::TypedOperatorHandle<is_neg::schema> create_is_neg_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(is_neg::name, is_neg::overload_name)
      .typed<is_neg::schema>();
}

// aten::is_neg(Tensor self) -> bool
bool is_neg::call(const at::Tensor & self) {
    static auto op = create_is_neg_typed_handle();
    return op.call(self);
}

// aten::is_neg(Tensor self) -> bool
bool is_neg::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_is_neg_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isreal, name, "aten::isreal")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isreal, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isreal, schema_str, "isreal(Tensor self) -> Tensor")

// aten::isreal(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<isreal::schema> create_isreal_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(isreal::name, isreal::overload_name)
      .typed<isreal::schema>();
}

// aten::isreal(Tensor self) -> Tensor
at::Tensor isreal::call(const at::Tensor & self) {
    static auto op = create_isreal_typed_handle();
    return op.call(self);
}

// aten::isreal(Tensor self) -> Tensor
at::Tensor isreal::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_isreal_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(kthvalue_values, name, "aten::kthvalue")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(kthvalue_values, overload_name, "values")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(kthvalue_values, schema_str, "kthvalue.values(Tensor self, int k, int dim=-1, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)")

// aten::kthvalue.values(Tensor self, int k, int dim=-1, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
static C10_NOINLINE c10::TypedOperatorHandle<kthvalue_values::schema> create_kthvalue_values_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(kthvalue_values::name, kthvalue_values::overload_name)
      .typed<kthvalue_values::schema>();
}

// aten::kthvalue.values(Tensor self, int k, int dim=-1, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> kthvalue_values::call(const at::Tensor & self, int64_t k, int64_t dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
    static auto op = create_kthvalue_values_typed_handle();
    return op.call(self, k, dim, keepdim, values, indices);
}

// aten::kthvalue.values(Tensor self, int k, int dim=-1, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> kthvalue_values::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t k, int64_t dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
    static auto op = create_kthvalue_values_typed_handle();
    return op.redispatch(dispatchKeySet, self, k, dim, keepdim, values, indices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(kthvalue_dimname, name, "aten::kthvalue")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(kthvalue_dimname, overload_name, "dimname")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(kthvalue_dimname, schema_str, "kthvalue.dimname(Tensor self, int k, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)")

// aten::kthvalue.dimname(Tensor self, int k, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
static C10_NOINLINE c10::TypedOperatorHandle<kthvalue_dimname::schema> create_kthvalue_dimname_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(kthvalue_dimname::name, kthvalue_dimname::overload_name)
      .typed<kthvalue_dimname::schema>();
}

// aten::kthvalue.dimname(Tensor self, int k, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> kthvalue_dimname::call(const at::Tensor & self, int64_t k, at::Dimname dim, bool keepdim) {
    static auto op = create_kthvalue_dimname_typed_handle();
    return op.call(self, k, dim, keepdim);
}

// aten::kthvalue.dimname(Tensor self, int k, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> kthvalue_dimname::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t k, at::Dimname dim, bool keepdim) {
    static auto op = create_kthvalue_dimname_typed_handle();
    return op.redispatch(dispatchKeySet, self, k, dim, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linear_out, name, "aten::linear")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linear_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linear_out, schema_str, "linear.out(Tensor input, Tensor weight, Tensor? bias=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::linear.out(Tensor input, Tensor weight, Tensor? bias=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<linear_out::schema> create_linear_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linear_out::name, linear_out::overload_name)
      .typed<linear_out::schema>();
}

// aten::linear.out(Tensor input, Tensor weight, Tensor? bias=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linear_out::call(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::Tensor & out) {
    static auto op = create_linear_out_typed_handle();
    return op.call(input, weight, bias, out);
}

// aten::linear.out(Tensor input, Tensor weight, Tensor? bias=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linear_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::Tensor & out) {
    static auto op = create_linear_out_typed_handle();
    return op.redispatch(dispatchKeySet, input, weight, bias, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_linear_backward_input, name, "aten::mkldnn_linear_backward_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_linear_backward_input, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_linear_backward_input, schema_str, "mkldnn_linear_backward_input(int[] input_size, Tensor grad_output, Tensor weight) -> Tensor")

// aten::mkldnn_linear_backward_input(int[] input_size, Tensor grad_output, Tensor weight) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<mkldnn_linear_backward_input::schema> create_mkldnn_linear_backward_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mkldnn_linear_backward_input::name, mkldnn_linear_backward_input::overload_name)
      .typed<mkldnn_linear_backward_input::schema>();
}

// aten::mkldnn_linear_backward_input(int[] input_size, Tensor grad_output, Tensor weight) -> Tensor
at::Tensor mkldnn_linear_backward_input::call(at::IntArrayRef input_size, const at::Tensor & grad_output, const at::Tensor & weight) {
    static auto op = create_mkldnn_linear_backward_input_typed_handle();
    return op.call(input_size, grad_output, weight);
}

// aten::mkldnn_linear_backward_input(int[] input_size, Tensor grad_output, Tensor weight) -> Tensor
at::Tensor mkldnn_linear_backward_input::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef input_size, const at::Tensor & grad_output, const at::Tensor & weight) {
    static auto op = create_mkldnn_linear_backward_input_typed_handle();
    return op.redispatch(dispatchKeySet, input_size, grad_output, weight);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_linear_backward, name, "aten::mkldnn_linear_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_linear_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_linear_backward, schema_str, "mkldnn_linear_backward(Tensor self, Tensor grad_output, Tensor weight, bool[3] output_mask) -> (Tensor, Tensor, Tensor)")

// aten::mkldnn_linear_backward(Tensor self, Tensor grad_output, Tensor weight, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<mkldnn_linear_backward::schema> create_mkldnn_linear_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mkldnn_linear_backward::name, mkldnn_linear_backward::overload_name)
      .typed<mkldnn_linear_backward::schema>();
}

// aten::mkldnn_linear_backward(Tensor self, Tensor grad_output, Tensor weight, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> mkldnn_linear_backward::call(const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, ::std::array<bool,3> output_mask) {
    static auto op = create_mkldnn_linear_backward_typed_handle();
    return op.call(self, grad_output, weight, output_mask);
}

// aten::mkldnn_linear_backward(Tensor self, Tensor grad_output, Tensor weight, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> mkldnn_linear_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, ::std::array<bool,3> output_mask) {
    static auto op = create_mkldnn_linear_backward_typed_handle();
    return op.redispatch(dispatchKeySet, self, grad_output, weight, output_mask);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log_out, name, "aten::log")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log_out, schema_str, "log.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::log.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<log_out::schema> create_log_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(log_out::name, log_out::overload_name)
      .typed<log_out::schema>();
}

// aten::log.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & log_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_log_out_typed_handle();
    return op.call(self, out);
}

// aten::log.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & log_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_log_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logaddexp_out, name, "aten::logaddexp")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logaddexp_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logaddexp_out, schema_str, "logaddexp.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::logaddexp.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<logaddexp_out::schema> create_logaddexp_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(logaddexp_out::name, logaddexp_out::overload_name)
      .typed<logaddexp_out::schema>();
}

// aten::logaddexp.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & logaddexp_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_logaddexp_out_typed_handle();
    return op.call(self, other, out);
}

// aten::logaddexp.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & logaddexp_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_logaddexp_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logdet, name, "aten::logdet")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logdet, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logdet, schema_str, "logdet(Tensor self) -> Tensor")

// aten::logdet(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<logdet::schema> create_logdet_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(logdet::name, logdet::overload_name)
      .typed<logdet::schema>();
}

// aten::logdet(Tensor self) -> Tensor
at::Tensor logdet::call(const at::Tensor & self) {
    static auto op = create_logdet_typed_handle();
    return op.call(self);
}

// aten::logdet(Tensor self) -> Tensor
at::Tensor logdet::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_logdet_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log_softmax_int, name, "aten::log_softmax")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log_softmax_int, overload_name, "int")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log_softmax_int, schema_str, "log_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor")

// aten::log_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<log_softmax_int::schema> create_log_softmax_int_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(log_softmax_int::name, log_softmax_int::overload_name)
      .typed<log_softmax_int::schema>();
}

// aten::log_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
at::Tensor log_softmax_int::call(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_log_softmax_int_typed_handle();
    return op.call(self, dim, dtype);
}

// aten::log_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
at::Tensor log_softmax_int::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_log_softmax_int_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_log_softmax_backward_data_out, name, "aten::_log_softmax_backward_data")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_log_softmax_backward_data_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_log_softmax_backward_data_out, schema_str, "_log_softmax_backward_data.out(Tensor grad_output, Tensor output, int dim, Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::_log_softmax_backward_data.out(Tensor grad_output, Tensor output, int dim, Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<_log_softmax_backward_data_out::schema> create__log_softmax_backward_data_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_log_softmax_backward_data_out::name, _log_softmax_backward_data_out::overload_name)
      .typed<_log_softmax_backward_data_out::schema>();
}

// aten::_log_softmax_backward_data.out(Tensor grad_output, Tensor output, int dim, Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & _log_softmax_backward_data_out::call(const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, const at::Tensor & self, at::Tensor & out) {
    static auto op = create__log_softmax_backward_data_out_typed_handle();
    return op.call(grad_output, output, dim, self, out);
}

// aten::_log_softmax_backward_data.out(Tensor grad_output, Tensor output, int dim, Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & _log_softmax_backward_data_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, const at::Tensor & self, at::Tensor & out) {
    static auto op = create__log_softmax_backward_data_out_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, output, dim, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_logcumsumexp, name, "aten::_logcumsumexp")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_logcumsumexp, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_logcumsumexp, schema_str, "_logcumsumexp(Tensor self, int dim) -> Tensor")

// aten::_logcumsumexp(Tensor self, int dim) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_logcumsumexp::schema> create__logcumsumexp_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_logcumsumexp::name, _logcumsumexp::overload_name)
      .typed<_logcumsumexp::schema>();
}

// aten::_logcumsumexp(Tensor self, int dim) -> Tensor
at::Tensor _logcumsumexp::call(const at::Tensor & self, int64_t dim) {
    static auto op = create__logcumsumexp_typed_handle();
    return op.call(self, dim);
}

// aten::_logcumsumexp(Tensor self, int dim) -> Tensor
at::Tensor _logcumsumexp::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim) {
    static auto op = create__logcumsumexp_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logcumsumexp_out, name, "aten::logcumsumexp")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logcumsumexp_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logcumsumexp_out, schema_str, "logcumsumexp.out(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)")

// aten::logcumsumexp.out(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<logcumsumexp_out::schema> create_logcumsumexp_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(logcumsumexp_out::name, logcumsumexp_out::overload_name)
      .typed<logcumsumexp_out::schema>();
}

// aten::logcumsumexp.out(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & logcumsumexp_out::call(const at::Tensor & self, int64_t dim, at::Tensor & out) {
    static auto op = create_logcumsumexp_out_typed_handle();
    return op.call(self, dim, out);
}

// aten::logcumsumexp.out(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & logcumsumexp_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, at::Tensor & out) {
    static auto op = create_logcumsumexp_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logsumexp_out, name, "aten::logsumexp")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logsumexp_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logsumexp_out, schema_str, "logsumexp.out(Tensor self, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)")

// aten::logsumexp.out(Tensor self, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<logsumexp_out::schema> create_logsumexp_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(logsumexp_out::name, logsumexp_out::overload_name)
      .typed<logsumexp_out::schema>();
}

// aten::logsumexp.out(Tensor self, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & logsumexp_out::call(const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
    static auto op = create_logsumexp_out_typed_handle();
    return op.call(self, dim, keepdim, out);
}

// aten::logsumexp.out(Tensor self, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & logsumexp_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
    static auto op = create_logsumexp_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_names_dim_max, name, "aten::max")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_names_dim_max, overload_name, "names_dim_max")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_names_dim_max, schema_str, "max.names_dim_max(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) -> (Tensor(a!) values, Tensor(b!) indices)")

// aten::max.names_dim_max(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) -> (Tensor(a!) values, Tensor(b!) indices)
static C10_NOINLINE c10::TypedOperatorHandle<max_names_dim_max::schema> create_max_names_dim_max_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(max_names_dim_max::name, max_names_dim_max::overload_name)
      .typed<max_names_dim_max::schema>();
}

// aten::max.names_dim_max(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> max_names_dim_max::call(const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & max, at::Tensor & max_values) {
    static auto op = create_max_names_dim_max_typed_handle();
    return op.call(self, dim, keepdim, max, max_values);
}

// aten::max.names_dim_max(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> max_names_dim_max::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & max, at::Tensor & max_values) {
    static auto op = create_max_names_dim_max_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, max, max_values);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(value_selecting_reduction_backward, name, "aten::value_selecting_reduction_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(value_selecting_reduction_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(value_selecting_reduction_backward, schema_str, "value_selecting_reduction_backward(Tensor grad, int dim, Tensor indices, int[] sizes, bool keepdim) -> Tensor")

// aten::value_selecting_reduction_backward(Tensor grad, int dim, Tensor indices, int[] sizes, bool keepdim) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<value_selecting_reduction_backward::schema> create_value_selecting_reduction_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(value_selecting_reduction_backward::name, value_selecting_reduction_backward::overload_name)
      .typed<value_selecting_reduction_backward::schema>();
}

// aten::value_selecting_reduction_backward(Tensor grad, int dim, Tensor indices, int[] sizes, bool keepdim) -> Tensor
at::Tensor value_selecting_reduction_backward::call(const at::Tensor & grad, int64_t dim, const at::Tensor & indices, at::IntArrayRef sizes, bool keepdim) {
    static auto op = create_value_selecting_reduction_backward_typed_handle();
    return op.call(grad, dim, indices, sizes, keepdim);
}

// aten::value_selecting_reduction_backward(Tensor grad, int dim, Tensor indices, int[] sizes, bool keepdim) -> Tensor
at::Tensor value_selecting_reduction_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, int64_t dim, const at::Tensor & indices, at::IntArrayRef sizes, bool keepdim) {
    static auto op = create_value_selecting_reduction_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad, dim, indices, sizes, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_pool1d, name, "aten::max_pool1d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_pool1d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_pool1d, schema_str, "max_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> Tensor")

// aten::max_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<max_pool1d::schema> create_max_pool1d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(max_pool1d::name, max_pool1d::overload_name)
      .typed<max_pool1d::schema>();
}

// aten::max_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> Tensor
at::Tensor max_pool1d::call(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
    static auto op = create_max_pool1d_typed_handle();
    return op.call(self, kernel_size, stride, padding, dilation, ceil_mode);
}

// aten::max_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> Tensor
at::Tensor max_pool1d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
    static auto op = create_max_pool1d_typed_handle();
    return op.redispatch(dispatchKeySet, self, kernel_size, stride, padding, dilation, ceil_mode);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_pool2d, name, "aten::max_pool2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_pool2d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_pool2d, schema_str, "max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor")

// aten::max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<max_pool2d::schema> create_max_pool2d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(max_pool2d::name, max_pool2d::overload_name)
      .typed<max_pool2d::schema>();
}

// aten::max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor
at::Tensor max_pool2d::call(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
    static auto op = create_max_pool2d_typed_handle();
    return op.call(self, kernel_size, stride, padding, dilation, ceil_mode);
}

// aten::max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor
at::Tensor max_pool2d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
    static auto op = create_max_pool2d_typed_handle();
    return op.redispatch(dispatchKeySet, self, kernel_size, stride, padding, dilation, ceil_mode);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mean, name, "aten::mean")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mean, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mean, schema_str, "mean(Tensor self, *, ScalarType? dtype=None) -> Tensor")

// aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<mean::schema> create_mean_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mean::name, mean::overload_name)
      .typed<mean::schema>();
}

// aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor
at::Tensor mean::call(const at::Tensor & self, c10::optional<at::ScalarType> dtype) {
    static auto op = create_mean_typed_handle();
    return op.call(self, dtype);
}

// aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor
at::Tensor mean::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::ScalarType> dtype) {
    static auto op = create_mean_typed_handle();
    return op.redispatch(dispatchKeySet, self, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mean_dim, name, "aten::mean")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mean_dim, overload_name, "dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mean_dim, schema_str, "mean.dim(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor")

// aten::mean.dim(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<mean_dim::schema> create_mean_dim_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mean_dim::name, mean_dim::overload_name)
      .typed<mean_dim::schema>();
}

// aten::mean.dim(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
at::Tensor mean_dim::call(const at::Tensor & self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_mean_dim_typed_handle();
    return op.call(self, dim, keepdim, dtype);
}

// aten::mean.dim(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
at::Tensor mean_dim::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_mean_dim_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanmean, name, "aten::nanmean")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanmean, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanmean, schema_str, "nanmean(Tensor self, int[1] dim=[], bool keepdim=False, *, ScalarType? dtype=None) -> Tensor")

// aten::nanmean(Tensor self, int[1] dim=[], bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<nanmean::schema> create_nanmean_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nanmean::name, nanmean::overload_name)
      .typed<nanmean::schema>();
}

// aten::nanmean(Tensor self, int[1] dim=[], bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
at::Tensor nanmean::call(const at::Tensor & self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_nanmean_typed_handle();
    return op.call(self, dim, keepdim, dtype);
}

// aten::nanmean(Tensor self, int[1] dim=[], bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
at::Tensor nanmean::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_nanmean_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(median_dim, name, "aten::median")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(median_dim, overload_name, "dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(median_dim, schema_str, "median.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)")

// aten::median.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
static C10_NOINLINE c10::TypedOperatorHandle<median_dim::schema> create_median_dim_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(median_dim::name, median_dim::overload_name)
      .typed<median_dim::schema>();
}

// aten::median.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> median_dim::call(const at::Tensor & self, int64_t dim, bool keepdim) {
    static auto op = create_median_dim_typed_handle();
    return op.call(self, dim, keepdim);
}

// aten::median.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> median_dim::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool keepdim) {
    static auto op = create_median_dim_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanmedian_names_dim, name, "aten::nanmedian")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanmedian_names_dim, overload_name, "names_dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanmedian_names_dim, schema_str, "nanmedian.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)")

// aten::nanmedian.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
static C10_NOINLINE c10::TypedOperatorHandle<nanmedian_names_dim::schema> create_nanmedian_names_dim_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nanmedian_names_dim::name, nanmedian_names_dim::overload_name)
      .typed<nanmedian_names_dim::schema>();
}

// aten::nanmedian.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> nanmedian_names_dim::call(const at::Tensor & self, at::Dimname dim, bool keepdim) {
    static auto op = create_nanmedian_names_dim_typed_handle();
    return op.call(self, dim, keepdim);
}

// aten::nanmedian.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> nanmedian_names_dim::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, bool keepdim) {
    static auto op = create_nanmedian_names_dim_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(min_names_dim_min, name, "aten::min")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(min_names_dim_min, overload_name, "names_dim_min")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(min_names_dim_min, schema_str, "min.names_dim_min(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices) -> (Tensor(a!) values, Tensor(b!) indices)")

// aten::min.names_dim_min(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices) -> (Tensor(a!) values, Tensor(b!) indices)
static C10_NOINLINE c10::TypedOperatorHandle<min_names_dim_min::schema> create_min_names_dim_min_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(min_names_dim_min::name, min_names_dim_min::overload_name)
      .typed<min_names_dim_min::schema>();
}

// aten::min.names_dim_min(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> min_names_dim_min::call(const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & min, at::Tensor & min_indices) {
    static auto op = create_min_names_dim_min_typed_handle();
    return op.call(self, dim, keepdim, min, min_indices);
}

// aten::min.names_dim_min(Tensor self, Dimname dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices) -> (Tensor(a!) values, Tensor(b!) indices)
::std::tuple<at::Tensor &,at::Tensor &> min_names_dim_min::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & min, at::Tensor & min_indices) {
    static auto op = create_min_names_dim_min_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, min, min_indices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_convolution_backward_weights, name, "aten::mkldnn_convolution_backward_weights")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_convolution_backward_weights, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_convolution_backward_weights, schema_str, "mkldnn_convolution_backward_weights(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool bias_defined) -> (Tensor, Tensor)")

// aten::mkldnn_convolution_backward_weights(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool bias_defined) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<mkldnn_convolution_backward_weights::schema> create_mkldnn_convolution_backward_weights_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mkldnn_convolution_backward_weights::name, mkldnn_convolution_backward_weights::overload_name)
      .typed<mkldnn_convolution_backward_weights::schema>();
}

// aten::mkldnn_convolution_backward_weights(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool bias_defined) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> mkldnn_convolution_backward_weights::call(at::IntArrayRef weight_size, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool bias_defined) {
    static auto op = create_mkldnn_convolution_backward_weights_typed_handle();
    return op.call(weight_size, grad_output, self, padding, stride, dilation, groups, bias_defined);
}

// aten::mkldnn_convolution_backward_weights(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool bias_defined) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> mkldnn_convolution_backward_weights::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef weight_size, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool bias_defined) {
    static auto op = create_mkldnn_convolution_backward_weights_typed_handle();
    return op.redispatch(dispatchKeySet, weight_size, grad_output, self, padding, stride, dilation, groups, bias_defined);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_convolution_backward, name, "aten::mkldnn_convolution_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_convolution_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mkldnn_convolution_backward, schema_str, "mkldnn_convolution_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)")

// aten::mkldnn_convolution_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<mkldnn_convolution_backward::schema> create_mkldnn_convolution_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mkldnn_convolution_backward::name, mkldnn_convolution_backward::overload_name)
      .typed<mkldnn_convolution_backward::schema>();
}

// aten::mkldnn_convolution_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> mkldnn_convolution_backward::call(const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, ::std::array<bool,3> output_mask) {
    static auto op = create_mkldnn_convolution_backward_typed_handle();
    return op.call(self, grad_output, weight, padding, stride, dilation, groups, output_mask);
}

// aten::mkldnn_convolution_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> mkldnn_convolution_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, ::std::array<bool,3> output_mask) {
    static auto op = create_mkldnn_convolution_backward_typed_handle();
    return op.redispatch(dispatchKeySet, self, grad_output, weight, padding, stride, dilation, groups, output_mask);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mm, name, "aten::mm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mm, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mm, schema_str, "mm(Tensor self, Tensor mat2) -> Tensor")

// aten::mm(Tensor self, Tensor mat2) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<mm::schema> create_mm_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mm::name, mm::overload_name)
      .typed<mm::schema>();
}

// aten::mm(Tensor self, Tensor mat2) -> Tensor
at::Tensor mm::call(const at::Tensor & self, const at::Tensor & mat2) {
    static auto op = create_mm_typed_handle();
    return op.call(self, mat2);
}

// aten::mm(Tensor self, Tensor mat2) -> Tensor
at::Tensor mm::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & mat2) {
    static auto op = create_mm_typed_handle();
    return op.redispatch(dispatchKeySet, self, mat2);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multiply_out, name, "aten::multiply")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multiply_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(multiply_out, schema_str, "multiply.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::multiply.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<multiply_out::schema> create_multiply_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(multiply_out::name, multiply_out::overload_name)
      .typed<multiply_out::schema>();
}

// aten::multiply.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & multiply_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_multiply_out_typed_handle();
    return op.call(self, other, out);
}

// aten::multiply.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & multiply_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_multiply_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mv, name, "aten::mv")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mv, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mv, schema_str, "mv(Tensor self, Tensor vec) -> Tensor")

// aten::mv(Tensor self, Tensor vec) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<mv::schema> create_mv_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mv::name, mv::overload_name)
      .typed<mv::schema>();
}

// aten::mv(Tensor self, Tensor vec) -> Tensor
at::Tensor mv::call(const at::Tensor & self, const at::Tensor & vec) {
    static auto op = create_mv_typed_handle();
    return op.call(self, vec);
}

// aten::mv(Tensor self, Tensor vec) -> Tensor
at::Tensor mv::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & vec) {
    static auto op = create_mv_typed_handle();
    return op.redispatch(dispatchKeySet, self, vec);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mvlgamma_, name, "aten::mvlgamma_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mvlgamma_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mvlgamma_, schema_str, "mvlgamma_(Tensor(a!) self, int p) -> Tensor(a!)")

// aten::mvlgamma_(Tensor(a!) self, int p) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<mvlgamma_::schema> create_mvlgamma__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mvlgamma_::name, mvlgamma_::overload_name)
      .typed<mvlgamma_::schema>();
}

// aten::mvlgamma_(Tensor(a!) self, int p) -> Tensor(a!)
at::Tensor & mvlgamma_::call(at::Tensor & self, int64_t p) {
    static auto op = create_mvlgamma__typed_handle();
    return op.call(self, p);
}

// aten::mvlgamma_(Tensor(a!) self, int p) -> Tensor(a!)
at::Tensor & mvlgamma_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t p) {
    static auto op = create_mvlgamma__typed_handle();
    return op.redispatch(dispatchKeySet, self, p);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(narrow_copy, name, "aten::narrow_copy")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(narrow_copy, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(narrow_copy, schema_str, "narrow_copy(Tensor self, int dim, int start, int length) -> Tensor")

// aten::narrow_copy(Tensor self, int dim, int start, int length) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<narrow_copy::schema> create_narrow_copy_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(narrow_copy::name, narrow_copy::overload_name)
      .typed<narrow_copy::schema>();
}

// aten::narrow_copy(Tensor self, int dim, int start, int length) -> Tensor
at::Tensor narrow_copy::call(const at::Tensor & self, int64_t dim, int64_t start, int64_t length) {
    static auto op = create_narrow_copy_typed_handle();
    return op.call(self, dim, start, length);
}

// aten::narrow_copy(Tensor self, int dim, int start, int length) -> Tensor
at::Tensor narrow_copy::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, int64_t start, int64_t length) {
    static auto op = create_narrow_copy_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, start, length);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(batch_norm_gather_stats_with_counts, name, "aten::batch_norm_gather_stats_with_counts")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(batch_norm_gather_stats_with_counts, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(batch_norm_gather_stats_with_counts, schema_str, "batch_norm_gather_stats_with_counts(Tensor input, Tensor mean, Tensor invstd, Tensor? running_mean, Tensor? running_var, float momentum, float eps, Tensor counts) -> (Tensor, Tensor)")

// aten::batch_norm_gather_stats_with_counts(Tensor input, Tensor mean, Tensor invstd, Tensor? running_mean, Tensor? running_var, float momentum, float eps, Tensor counts) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<batch_norm_gather_stats_with_counts::schema> create_batch_norm_gather_stats_with_counts_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(batch_norm_gather_stats_with_counts::name, batch_norm_gather_stats_with_counts::overload_name)
      .typed<batch_norm_gather_stats_with_counts::schema>();
}

// aten::batch_norm_gather_stats_with_counts(Tensor input, Tensor mean, Tensor invstd, Tensor? running_mean, Tensor? running_var, float momentum, float eps, Tensor counts) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> batch_norm_gather_stats_with_counts::call(const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, double momentum, double eps, const at::Tensor & counts) {
    static auto op = create_batch_norm_gather_stats_with_counts_typed_handle();
    return op.call(input, mean, invstd, running_mean, running_var, momentum, eps, counts);
}

// aten::batch_norm_gather_stats_with_counts(Tensor input, Tensor mean, Tensor invstd, Tensor? running_mean, Tensor? running_var, float momentum, float eps, Tensor counts) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> batch_norm_gather_stats_with_counts::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, double momentum, double eps, const at::Tensor & counts) {
    static auto op = create_batch_norm_gather_stats_with_counts_typed_handle();
    return op.redispatch(dispatchKeySet, input, mean, invstd, running_mean, running_var, momentum, eps, counts);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_nnpack_spatial_convolution_backward, name, "aten::_nnpack_spatial_convolution_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_nnpack_spatial_convolution_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_nnpack_spatial_convolution_backward, schema_str, "_nnpack_spatial_convolution_backward(Tensor input, Tensor grad_output, Tensor weight, int[2] padding, bool[3] output_mask) -> (Tensor, Tensor, Tensor)")

// aten::_nnpack_spatial_convolution_backward(Tensor input, Tensor grad_output, Tensor weight, int[2] padding, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<_nnpack_spatial_convolution_backward::schema> create__nnpack_spatial_convolution_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_nnpack_spatial_convolution_backward::name, _nnpack_spatial_convolution_backward::overload_name)
      .typed<_nnpack_spatial_convolution_backward::schema>();
}

// aten::_nnpack_spatial_convolution_backward(Tensor input, Tensor grad_output, Tensor weight, int[2] padding, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _nnpack_spatial_convolution_backward::call(const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, ::std::array<bool,3> output_mask) {
    static auto op = create__nnpack_spatial_convolution_backward_typed_handle();
    return op.call(input, grad_output, weight, padding, output_mask);
}

// aten::_nnpack_spatial_convolution_backward(Tensor input, Tensor grad_output, Tensor weight, int[2] padding, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _nnpack_spatial_convolution_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, ::std::array<bool,3> output_mask) {
    static auto op = create__nnpack_spatial_convolution_backward_typed_handle();
    return op.redispatch(dispatchKeySet, input, grad_output, weight, padding, output_mask);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ones_out, name, "aten::ones")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ones_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ones_out, schema_str, "ones.out(int[] size, *, Tensor(a!) out) -> Tensor(a!)")

// aten::ones.out(int[] size, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<ones_out::schema> create_ones_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(ones_out::name, ones_out::overload_name)
      .typed<ones_out::schema>();
}

// aten::ones.out(int[] size, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & ones_out::call(at::IntArrayRef size, at::Tensor & out) {
    static auto op = create_ones_out_typed_handle();
    return op.call(size, out);
}

// aten::ones.out(int[] size, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & ones_out::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, at::Tensor & out) {
    static auto op = create_ones_out_typed_handle();
    return op.redispatch(dispatchKeySet, size, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pairwise_distance, name, "aten::pairwise_distance")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pairwise_distance, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pairwise_distance, schema_str, "pairwise_distance(Tensor x1, Tensor x2, float p=2, float eps=1e-06, bool keepdim=False) -> Tensor")

// aten::pairwise_distance(Tensor x1, Tensor x2, float p=2, float eps=1e-06, bool keepdim=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<pairwise_distance::schema> create_pairwise_distance_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(pairwise_distance::name, pairwise_distance::overload_name)
      .typed<pairwise_distance::schema>();
}

// aten::pairwise_distance(Tensor x1, Tensor x2, float p=2, float eps=1e-06, bool keepdim=False) -> Tensor
at::Tensor pairwise_distance::call(const at::Tensor & x1, const at::Tensor & x2, double p, double eps, bool keepdim) {
    static auto op = create_pairwise_distance_typed_handle();
    return op.call(x1, x2, p, eps, keepdim);
}

// aten::pairwise_distance(Tensor x1, Tensor x2, float p=2, float eps=1e-06, bool keepdim=False) -> Tensor
at::Tensor pairwise_distance::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & x1, const at::Tensor & x2, double p, double eps, bool keepdim) {
    static auto op = create_pairwise_distance_typed_handle();
    return op.redispatch(dispatchKeySet, x1, x2, p, eps, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_pdist_backward, name, "aten::_pdist_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_pdist_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_pdist_backward, schema_str, "_pdist_backward(Tensor grad, Tensor self, float p, Tensor pdist) -> Tensor")

// aten::_pdist_backward(Tensor grad, Tensor self, float p, Tensor pdist) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_pdist_backward::schema> create__pdist_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_pdist_backward::name, _pdist_backward::overload_name)
      .typed<_pdist_backward::schema>();
}

// aten::_pdist_backward(Tensor grad, Tensor self, float p, Tensor pdist) -> Tensor
at::Tensor _pdist_backward::call(const at::Tensor & grad, const at::Tensor & self, double p, const at::Tensor & pdist) {
    static auto op = create__pdist_backward_typed_handle();
    return op.call(grad, self, p, pdist);
}

// aten::_pdist_backward(Tensor grad, Tensor self, float p, Tensor pdist) -> Tensor
at::Tensor _pdist_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & self, double p, const at::Tensor & pdist) {
    static auto op = create__pdist_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad, self, p, pdist);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(permute, name, "aten::permute")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(permute, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(permute, schema_str, "permute(Tensor(a) self, int[] dims) -> Tensor(a)")

// aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<permute::schema> create_permute_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(permute::name, permute::overload_name)
      .typed<permute::schema>();
}

// aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)
at::Tensor permute::call(const at::Tensor & self, at::IntArrayRef dims) {
    static auto op = create_permute_typed_handle();
    return op.call(self, dims);
}

// aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)
at::Tensor permute::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dims) {
    static auto op = create_permute_typed_handle();
    return op.redispatch(dispatchKeySet, self, dims);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(movedim_int, name, "aten::movedim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(movedim_int, overload_name, "int")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(movedim_int, schema_str, "movedim.int(Tensor(a) self, int source, int destination) -> Tensor(a)")

// aten::movedim.int(Tensor(a) self, int source, int destination) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<movedim_int::schema> create_movedim_int_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(movedim_int::name, movedim_int::overload_name)
      .typed<movedim_int::schema>();
}

// aten::movedim.int(Tensor(a) self, int source, int destination) -> Tensor(a)
at::Tensor movedim_int::call(const at::Tensor & self, int64_t source, int64_t destination) {
    static auto op = create_movedim_int_typed_handle();
    return op.call(self, source, destination);
}

// aten::movedim.int(Tensor(a) self, int source, int destination) -> Tensor(a)
at::Tensor movedim_int::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t source, int64_t destination) {
    static auto op = create_movedim_int_typed_handle();
    return op.redispatch(dispatchKeySet, self, source, destination);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pixel_shuffle, name, "aten::pixel_shuffle")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pixel_shuffle, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pixel_shuffle, schema_str, "pixel_shuffle(Tensor self, int upscale_factor) -> Tensor")

// aten::pixel_shuffle(Tensor self, int upscale_factor) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<pixel_shuffle::schema> create_pixel_shuffle_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(pixel_shuffle::name, pixel_shuffle::overload_name)
      .typed<pixel_shuffle::schema>();
}

// aten::pixel_shuffle(Tensor self, int upscale_factor) -> Tensor
at::Tensor pixel_shuffle::call(const at::Tensor & self, int64_t upscale_factor) {
    static auto op = create_pixel_shuffle_typed_handle();
    return op.call(self, upscale_factor);
}

// aten::pixel_shuffle(Tensor self, int upscale_factor) -> Tensor
at::Tensor pixel_shuffle::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t upscale_factor) {
    static auto op = create_pixel_shuffle_typed_handle();
    return op.redispatch(dispatchKeySet, self, upscale_factor);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pinverse, name, "aten::pinverse")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pinverse, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pinverse, schema_str, "pinverse(Tensor self, float rcond=1e-15) -> Tensor")

// aten::pinverse(Tensor self, float rcond=1e-15) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<pinverse::schema> create_pinverse_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(pinverse::name, pinverse::overload_name)
      .typed<pinverse::schema>();
}

// aten::pinverse(Tensor self, float rcond=1e-15) -> Tensor
at::Tensor pinverse::call(const at::Tensor & self, double rcond) {
    static auto op = create_pinverse_typed_handle();
    return op.call(self, rcond);
}

// aten::pinverse(Tensor self, float rcond=1e-15) -> Tensor
at::Tensor pinverse::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double rcond) {
    static auto op = create_pinverse_typed_handle();
    return op.redispatch(dispatchKeySet, self, rcond);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rand_names, name, "aten::rand")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rand_names, overload_name, "names")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(rand_names, schema_str, "rand.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::rand.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<rand_names::schema> create_rand_names_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(rand_names::name, rand_names::overload_name)
      .typed<rand_names::schema>();
}

// aten::rand.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor rand_names::call(at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_rand_names_typed_handle();
    return op.call(size, names, dtype, layout, device, pin_memory);
}

// aten::rand.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor rand_names::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_rand_names_typed_handle();
    return op.redispatch(dispatchKeySet, size, names, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randint_generator_out, name, "aten::randint")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randint_generator_out, overload_name, "generator_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randint_generator_out, schema_str, "randint.generator_out(int high, int[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)")

// aten::randint.generator_out(int high, int[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<randint_generator_out::schema> create_randint_generator_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(randint_generator_out::name, randint_generator_out::overload_name)
      .typed<randint_generator_out::schema>();
}

// aten::randint.generator_out(int high, int[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
at::Tensor & randint_generator_out::call(int64_t high, at::IntArrayRef size, c10::optional<at::Generator> generator, at::Tensor & out) {
    static auto op = create_randint_generator_out_typed_handle();
    return op.call(high, size, generator, out);
}

// aten::randint.generator_out(int high, int[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
at::Tensor & randint_generator_out::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t high, at::IntArrayRef size, c10::optional<at::Generator> generator, at::Tensor & out) {
    static auto op = create_randint_generator_out_typed_handle();
    return op.redispatch(dispatchKeySet, high, size, generator, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randperm_generator, name, "aten::randperm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randperm_generator, overload_name, "generator")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(randperm_generator, schema_str, "randperm.generator(int n, *, Generator? generator, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::randperm.generator(int n, *, Generator? generator, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<randperm_generator::schema> create_randperm_generator_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(randperm_generator::name, randperm_generator::overload_name)
      .typed<randperm_generator::schema>();
}

// aten::randperm.generator(int n, *, Generator? generator, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor randperm_generator::call(int64_t n, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_randperm_generator_typed_handle();
    return op.call(n, generator, dtype, layout, device, pin_memory);
}

// aten::randperm.generator(int n, *, Generator? generator, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor randperm_generator::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t n, c10::optional<at::Generator> generator, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_randperm_generator_typed_handle();
    return op.redispatch(dispatchKeySet, n, generator, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reciprocal_, name, "aten::reciprocal_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reciprocal_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reciprocal_, schema_str, "reciprocal_(Tensor(a!) self) -> Tensor(a!)")

// aten::reciprocal_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<reciprocal_::schema> create_reciprocal__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(reciprocal_::name, reciprocal_::overload_name)
      .typed<reciprocal_::schema>();
}

// aten::reciprocal_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & reciprocal_::call(at::Tensor & self) {
    static auto op = create_reciprocal__typed_handle();
    return op.call(self);
}

// aten::reciprocal_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & reciprocal_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_reciprocal__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reciprocal_out, name, "aten::reciprocal")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reciprocal_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reciprocal_out, schema_str, "reciprocal.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::reciprocal.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<reciprocal_out::schema> create_reciprocal_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(reciprocal_out::name, reciprocal_out::overload_name)
      .typed<reciprocal_out::schema>();
}

// aten::reciprocal.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & reciprocal_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_reciprocal_out_typed_handle();
    return op.call(self, out);
}

// aten::reciprocal.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & reciprocal_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_reciprocal_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reshape, name, "aten::reshape")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reshape, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reshape, schema_str, "reshape(Tensor(a) self, int[] shape) -> Tensor(a)")

// aten::reshape(Tensor(a) self, int[] shape) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<reshape::schema> create_reshape_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(reshape::name, reshape::overload_name)
      .typed<reshape::schema>();
}

// aten::reshape(Tensor(a) self, int[] shape) -> Tensor(a)
at::Tensor reshape::call(const at::Tensor & self, at::IntArrayRef shape) {
    static auto op = create_reshape_typed_handle();
    return op.call(self, shape);
}

// aten::reshape(Tensor(a) self, int[] shape) -> Tensor(a)
at::Tensor reshape::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef shape) {
    static auto op = create_reshape_typed_handle();
    return op.redispatch(dispatchKeySet, self, shape);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_reshape_alias, name, "aten::_reshape_alias")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_reshape_alias, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_reshape_alias, schema_str, "_reshape_alias(Tensor(a) self, int[] size, int[] stride) -> Tensor(a)")

// aten::_reshape_alias(Tensor(a) self, int[] size, int[] stride) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<_reshape_alias::schema> create__reshape_alias_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_reshape_alias::name, _reshape_alias::overload_name)
      .typed<_reshape_alias::schema>();
}

// aten::_reshape_alias(Tensor(a) self, int[] size, int[] stride) -> Tensor(a)
at::Tensor _reshape_alias::call(const at::Tensor & self, at::IntArrayRef size, at::IntArrayRef stride) {
    static auto op = create__reshape_alias_typed_handle();
    return op.call(self, size, stride);
}

// aten::_reshape_alias(Tensor(a) self, int[] size, int[] stride) -> Tensor(a)
at::Tensor _reshape_alias::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef size, at::IntArrayRef stride) {
    static auto op = create__reshape_alias_typed_handle();
    return op.redispatch(dispatchKeySet, self, size, stride);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(relu6_, name, "aten::relu6_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(relu6_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(relu6_, schema_str, "relu6_(Tensor(a!) self) -> Tensor(a!)")

// aten::relu6_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<relu6_::schema> create_relu6__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(relu6_::name, relu6_::overload_name)
      .typed<relu6_::schema>();
}

// aten::relu6_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & relu6_::call(at::Tensor & self) {
    static auto op = create_relu6__typed_handle();
    return op.call(self);
}

// aten::relu6_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & relu6_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_relu6__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gelu_out, name, "aten::gelu")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gelu_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gelu_out, schema_str, "gelu.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::gelu.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<gelu_out::schema> create_gelu_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(gelu_out::name, gelu_out::overload_name)
      .typed<gelu_out::schema>();
}

// aten::gelu.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & gelu_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_gelu_out_typed_handle();
    return op.call(self, out);
}

// aten::gelu.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & gelu_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_gelu_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(selu_, name, "aten::selu_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(selu_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(selu_, schema_str, "selu_(Tensor(a!) self) -> Tensor(a!)")

// aten::selu_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<selu_::schema> create_selu__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(selu_::name, selu_::overload_name)
      .typed<selu_::schema>();
}

// aten::selu_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & selu_::call(at::Tensor & self) {
    static auto op = create_selu__typed_handle();
    return op.call(self);
}

// aten::selu_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & selu_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_selu__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(celu, name, "aten::celu")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(celu, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(celu, schema_str, "celu(Tensor self, Scalar alpha=1.0) -> Tensor")

// aten::celu(Tensor self, Scalar alpha=1.0) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<celu::schema> create_celu_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(celu::name, celu::overload_name)
      .typed<celu::schema>();
}

// aten::celu(Tensor self, Scalar alpha=1.0) -> Tensor
at::Tensor celu::call(const at::Tensor & self, const at::Scalar & alpha) {
    static auto op = create_celu_typed_handle();
    return op.call(self, alpha);
}

// aten::celu(Tensor self, Scalar alpha=1.0) -> Tensor
at::Tensor celu::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & alpha) {
    static auto op = create_celu_typed_handle();
    return op.redispatch(dispatchKeySet, self, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(celu_, name, "aten::celu_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(celu_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(celu_, schema_str, "celu_(Tensor(a!) self, Scalar alpha=1.0) -> Tensor(a!)")

// aten::celu_(Tensor(a!) self, Scalar alpha=1.0) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<celu_::schema> create_celu__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(celu_::name, celu_::overload_name)
      .typed<celu_::schema>();
}

// aten::celu_(Tensor(a!) self, Scalar alpha=1.0) -> Tensor(a!)
at::Tensor & celu_::call(at::Tensor & self, const at::Scalar & alpha) {
    static auto op = create_celu__typed_handle();
    return op.call(self, alpha);
}

// aten::celu_(Tensor(a!) self, Scalar alpha=1.0) -> Tensor(a!)
at::Tensor & celu_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & alpha) {
    static auto op = create_celu__typed_handle();
    return op.redispatch(dispatchKeySet, self, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(silu, name, "aten::silu")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(silu, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(silu, schema_str, "silu(Tensor self) -> Tensor")

// aten::silu(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<silu::schema> create_silu_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(silu::name, silu::overload_name)
      .typed<silu::schema>();
}

// aten::silu(Tensor self) -> Tensor
at::Tensor silu::call(const at::Tensor & self) {
    static auto op = create_silu_typed_handle();
    return op.call(self);
}

// aten::silu(Tensor self) -> Tensor
at::Tensor silu::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_silu_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(silu_out, name, "aten::silu")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(silu_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(silu_out, schema_str, "silu.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::silu.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<silu_out::schema> create_silu_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(silu_out::name, silu_out::overload_name)
      .typed<silu_out::schema>();
}

// aten::silu.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & silu_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_silu_out_typed_handle();
    return op.call(self, out);
}

// aten::silu.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & silu_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_silu_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mish_, name, "aten::mish_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mish_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mish_, schema_str, "mish_(Tensor(a!) self) -> Tensor(a!)")

// aten::mish_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<mish_::schema> create_mish__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mish_::name, mish_::overload_name)
      .typed<mish_::schema>();
}

// aten::mish_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & mish_::call(at::Tensor & self) {
    static auto op = create_mish__typed_handle();
    return op.call(self);
}

// aten::mish_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & mish_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_mish__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mish_backward, name, "aten::mish_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mish_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(mish_backward, schema_str, "mish_backward(Tensor grad_output, Tensor self) -> Tensor")

// aten::mish_backward(Tensor grad_output, Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<mish_backward::schema> create_mish_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(mish_backward::name, mish_backward::overload_name)
      .typed<mish_backward::schema>();
}

// aten::mish_backward(Tensor grad_output, Tensor self) -> Tensor
at::Tensor mish_backward::call(const at::Tensor & grad_output, const at::Tensor & self) {
    static auto op = create_mish_backward_typed_handle();
    return op.call(grad_output, self);
}

// aten::mish_backward(Tensor grad_output, Tensor self) -> Tensor
at::Tensor mish_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self) {
    static auto op = create_mish_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logit, name, "aten::logit")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logit, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logit, schema_str, "logit(Tensor self, float? eps=None) -> Tensor")

// aten::logit(Tensor self, float? eps=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<logit::schema> create_logit_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(logit::name, logit::overload_name)
      .typed<logit::schema>();
}

// aten::logit(Tensor self, float? eps=None) -> Tensor
at::Tensor logit::call(const at::Tensor & self, c10::optional<double> eps) {
    static auto op = create_logit_typed_handle();
    return op.call(self, eps);
}

// aten::logit(Tensor self, float? eps=None) -> Tensor
at::Tensor logit::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<double> eps) {
    static auto op = create_logit_typed_handle();
    return op.redispatch(dispatchKeySet, self, eps);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sinh, name, "aten::sinh")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sinh, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sinh, schema_str, "sinh(Tensor self) -> Tensor")

// aten::sinh(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<sinh::schema> create_sinh_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sinh::name, sinh::overload_name)
      .typed<sinh::schema>();
}

// aten::sinh(Tensor self) -> Tensor
at::Tensor sinh::call(const at::Tensor & self) {
    static auto op = create_sinh_typed_handle();
    return op.call(self);
}

// aten::sinh(Tensor self) -> Tensor
at::Tensor sinh::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_sinh_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sinh_, name, "aten::sinh_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sinh_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sinh_, schema_str, "sinh_(Tensor(a!) self) -> Tensor(a!)")

// aten::sinh_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<sinh_::schema> create_sinh__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sinh_::name, sinh_::overload_name)
      .typed<sinh_::schema>();
}

// aten::sinh_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & sinh_::call(at::Tensor & self) {
    static auto op = create_sinh__typed_handle();
    return op.call(self);
}

// aten::sinh_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & sinh_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_sinh__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slice_backward, name, "aten::slice_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slice_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slice_backward, schema_str, "slice_backward(Tensor grad_output, int[] input_sizes, int dim, int start, int end, int step) -> Tensor")

// aten::slice_backward(Tensor grad_output, int[] input_sizes, int dim, int start, int end, int step) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<slice_backward::schema> create_slice_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(slice_backward::name, slice_backward::overload_name)
      .typed<slice_backward::schema>();
}

// aten::slice_backward(Tensor grad_output, int[] input_sizes, int dim, int start, int end, int step) -> Tensor
at::Tensor slice_backward::call(const at::Tensor & grad_output, at::IntArrayRef input_sizes, int64_t dim, int64_t start, int64_t end, int64_t step) {
    static auto op = create_slice_backward_typed_handle();
    return op.call(grad_output, input_sizes, dim, start, end, step);
}

// aten::slice_backward(Tensor grad_output, int[] input_sizes, int dim, int start, int end, int step) -> Tensor
at::Tensor slice_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, at::IntArrayRef input_sizes, int64_t dim, int64_t start, int64_t end, int64_t step) {
    static auto op = create_slice_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, input_sizes, dim, start, end, step);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_softmax, name, "aten::_softmax")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_softmax, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_softmax, schema_str, "_softmax(Tensor self, int dim, bool half_to_float) -> Tensor")

// aten::_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_softmax::schema> create__softmax_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_softmax::name, _softmax::overload_name)
      .typed<_softmax::schema>();
}

// aten::_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
at::Tensor _softmax::call(const at::Tensor & self, int64_t dim, bool half_to_float) {
    static auto op = create__softmax_typed_handle();
    return op.call(self, dim, half_to_float);
}

// aten::_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
at::Tensor _softmax::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool half_to_float) {
    static auto op = create__softmax_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, half_to_float);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(split_Tensor, name, "aten::split")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(split_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(split_Tensor, schema_str, "split.Tensor(Tensor(a) self, int split_size, int dim=0) -> Tensor(a)[]")

// aten::split.Tensor(Tensor(a) self, int split_size, int dim=0) -> Tensor(a)[]
static C10_NOINLINE c10::TypedOperatorHandle<split_Tensor::schema> create_split_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(split_Tensor::name, split_Tensor::overload_name)
      .typed<split_Tensor::schema>();
}

// aten::split.Tensor(Tensor(a) self, int split_size, int dim=0) -> Tensor(a)[]
::std::vector<at::Tensor> split_Tensor::call(const at::Tensor & self, int64_t split_size, int64_t dim) {
    static auto op = create_split_Tensor_typed_handle();
    return op.call(self, split_size, dim);
}

// aten::split.Tensor(Tensor(a) self, int split_size, int dim=0) -> Tensor(a)[]
::std::vector<at::Tensor> split_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t split_size, int64_t dim) {
    static auto op = create_split_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, split_size, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(vsplit_array, name, "aten::vsplit")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(vsplit_array, overload_name, "array")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(vsplit_array, schema_str, "vsplit.array(Tensor(a) self, int[] indices) -> Tensor(a)[]")

// aten::vsplit.array(Tensor(a) self, int[] indices) -> Tensor(a)[]
static C10_NOINLINE c10::TypedOperatorHandle<vsplit_array::schema> create_vsplit_array_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(vsplit_array::name, vsplit_array::overload_name)
      .typed<vsplit_array::schema>();
}

// aten::vsplit.array(Tensor(a) self, int[] indices) -> Tensor(a)[]
::std::vector<at::Tensor> vsplit_array::call(const at::Tensor & self, at::IntArrayRef indices) {
    static auto op = create_vsplit_array_typed_handle();
    return op.call(self, indices);
}

// aten::vsplit.array(Tensor(a) self, int[] indices) -> Tensor(a)[]
::std::vector<at::Tensor> vsplit_array::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef indices) {
    static auto op = create_vsplit_array_typed_handle();
    return op.redispatch(dispatchKeySet, self, indices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(stack_out, name, "aten::stack")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(stack_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(stack_out, schema_str, "stack.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)")

// aten::stack.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<stack_out::schema> create_stack_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(stack_out::name, stack_out::overload_name)
      .typed<stack_out::schema>();
}

// aten::stack.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & stack_out::call(at::TensorList tensors, int64_t dim, at::Tensor & out) {
    static auto op = create_stack_out_typed_handle();
    return op.call(tensors, dim, out);
}

// aten::stack.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & stack_out::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, int64_t dim, at::Tensor & out) {
    static auto op = create_stack_out_typed_handle();
    return op.redispatch(dispatchKeySet, tensors, dim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(vstack, name, "aten::vstack")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(vstack, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(vstack, schema_str, "vstack(Tensor[] tensors) -> Tensor")

// aten::vstack(Tensor[] tensors) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<vstack::schema> create_vstack_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(vstack::name, vstack::overload_name)
      .typed<vstack::schema>();
}

// aten::vstack(Tensor[] tensors) -> Tensor
at::Tensor vstack::call(at::TensorList tensors) {
    static auto op = create_vstack_typed_handle();
    return op.call(tensors);
}

// aten::vstack(Tensor[] tensors) -> Tensor
at::Tensor vstack::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create_vstack_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(stft, name, "aten::stft")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(stft, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(stft, schema_str, "stft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool normalized=False, bool? onesided=None, bool? return_complex=None) -> Tensor")

// aten::stft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool normalized=False, bool? onesided=None, bool? return_complex=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<stft::schema> create_stft_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(stft::name, stft::overload_name)
      .typed<stft::schema>();
}

// aten::stft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool normalized=False, bool? onesided=None, bool? return_complex=None) -> Tensor
at::Tensor stft::call(const at::Tensor & self, int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const c10::optional<at::Tensor> & window, bool normalized, c10::optional<bool> onesided, c10::optional<bool> return_complex) {
    static auto op = create_stft_typed_handle();
    return op.call(self, n_fft, hop_length, win_length, window, normalized, onesided, return_complex);
}

// aten::stft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool normalized=False, bool? onesided=None, bool? return_complex=None) -> Tensor
at::Tensor stft::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const c10::optional<at::Tensor> & window, bool normalized, c10::optional<bool> onesided, c10::optional<bool> return_complex) {
    static auto op = create_stft_typed_handle();
    return op.redispatch(dispatchKeySet, self, n_fft, hop_length, win_length, window, normalized, onesided, return_complex);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(stride_int, name, "aten::stride")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(stride_int, overload_name, "int")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(stride_int, schema_str, "stride.int(Tensor self, int dim) -> int")

// aten::stride.int(Tensor self, int dim) -> int
static C10_NOINLINE c10::TypedOperatorHandle<stride_int::schema> create_stride_int_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(stride_int::name, stride_int::overload_name)
      .typed<stride_int::schema>();
}

// aten::stride.int(Tensor self, int dim) -> int
int64_t stride_int::call(const at::Tensor & self, int64_t dim) {
    static auto op = create_stride_int_typed_handle();
    return op.call(self, dim);
}

// aten::stride.int(Tensor self, int dim) -> int
int64_t stride_int::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim) {
    static auto op = create_stride_int_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(stride_Dimname, name, "aten::stride")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(stride_Dimname, overload_name, "Dimname")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(stride_Dimname, schema_str, "stride.Dimname(Tensor self, Dimname dim) -> int")

// aten::stride.Dimname(Tensor self, Dimname dim) -> int
static C10_NOINLINE c10::TypedOperatorHandle<stride_Dimname::schema> create_stride_Dimname_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(stride_Dimname::name, stride_Dimname::overload_name)
      .typed<stride_Dimname::schema>();
}

// aten::stride.Dimname(Tensor self, Dimname dim) -> int
int64_t stride_Dimname::call(const at::Tensor & self, at::Dimname dim) {
    static auto op = create_stride_Dimname_typed_handle();
    return op.call(self, dim);
}

// aten::stride.Dimname(Tensor self, Dimname dim) -> int
int64_t stride_Dimname::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim) {
    static auto op = create_stride_Dimname_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sum_dim_IntList, name, "aten::sum")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sum_dim_IntList, overload_name, "dim_IntList")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sum_dim_IntList, schema_str, "sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor")

// aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<sum_dim_IntList::schema> create_sum_dim_IntList_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sum_dim_IntList::name, sum_dim_IntList::overload_name)
      .typed<sum_dim_IntList::schema>();
}

// aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
at::Tensor sum_dim_IntList::call(const at::Tensor & self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_sum_dim_IntList_typed_handle();
    return op.call(self, dim, keepdim, dtype);
}

// aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
at::Tensor sum_dim_IntList::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_sum_dim_IntList_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sum_dim_DimnameList, name, "aten::sum")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sum_dim_DimnameList, overload_name, "dim_DimnameList")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sum_dim_DimnameList, schema_str, "sum.dim_DimnameList(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor")

// aten::sum.dim_DimnameList(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<sum_dim_DimnameList::schema> create_sum_dim_DimnameList_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sum_dim_DimnameList::name, sum_dim_DimnameList::overload_name)
      .typed<sum_dim_DimnameList::schema>();
}

// aten::sum.dim_DimnameList(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
at::Tensor sum_dim_DimnameList::call(const at::Tensor & self, at::DimnameList dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_sum_dim_DimnameList_typed_handle();
    return op.call(self, dim, keepdim, dtype);
}

// aten::sum.dim_DimnameList(Tensor self, Dimname[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
at::Tensor sum_dim_DimnameList::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_sum_dim_DimnameList_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sum_to_size, name, "aten::sum_to_size")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sum_to_size, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sum_to_size, schema_str, "sum_to_size(Tensor self, int[] size) -> Tensor")

// aten::sum_to_size(Tensor self, int[] size) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<sum_to_size::schema> create_sum_to_size_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sum_to_size::name, sum_to_size::overload_name)
      .typed<sum_to_size::schema>();
}

// aten::sum_to_size(Tensor self, int[] size) -> Tensor
at::Tensor sum_to_size::call(const at::Tensor & self, at::IntArrayRef size) {
    static auto op = create_sum_to_size_typed_handle();
    return op.call(self, size);
}

// aten::sum_to_size(Tensor self, int[] size) -> Tensor
at::Tensor sum_to_size::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef size) {
    static auto op = create_sum_to_size_typed_handle();
    return op.redispatch(dispatchKeySet, self, size);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sqrt, name, "aten::sqrt")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sqrt, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sqrt, schema_str, "sqrt(Tensor self) -> Tensor")

// aten::sqrt(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<sqrt::schema> create_sqrt_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sqrt::name, sqrt::overload_name)
      .typed<sqrt::schema>();
}

// aten::sqrt(Tensor self) -> Tensor
at::Tensor sqrt::call(const at::Tensor & self) {
    static auto op = create_sqrt_typed_handle();
    return op.call(self);
}

// aten::sqrt(Tensor self) -> Tensor
at::Tensor sqrt::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_sqrt_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(square_, name, "aten::square_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(square_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(square_, schema_str, "square_(Tensor(a!) self) -> Tensor(a!)")

// aten::square_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<square_::schema> create_square__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(square_::name, square_::overload_name)
      .typed<square_::schema>();
}

// aten::square_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & square_::call(at::Tensor & self) {
    static auto op = create_square__typed_handle();
    return op.call(self);
}

// aten::square_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & square_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_square__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std, name, "aten::std")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std, schema_str, "std(Tensor self, bool unbiased=True) -> Tensor")

// aten::std(Tensor self, bool unbiased=True) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<std::schema> create_std_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(std::name, std::overload_name)
      .typed<std::schema>();
}

// aten::std(Tensor self, bool unbiased=True) -> Tensor
at::Tensor std::call(const at::Tensor & self, bool unbiased) {
    static auto op = create_std_typed_handle();
    return op.call(self, unbiased);
}

// aten::std(Tensor self, bool unbiased=True) -> Tensor
at::Tensor std::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool unbiased) {
    static auto op = create_std_typed_handle();
    return op.redispatch(dispatchKeySet, self, unbiased);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_mean, name, "aten::std_mean")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_mean, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_mean, schema_str, "std_mean(Tensor self, bool unbiased=True) -> (Tensor, Tensor)")

// aten::std_mean(Tensor self, bool unbiased=True) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<std_mean::schema> create_std_mean_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(std_mean::name, std_mean::overload_name)
      .typed<std_mean::schema>();
}

// aten::std_mean(Tensor self, bool unbiased=True) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> std_mean::call(const at::Tensor & self, bool unbiased) {
    static auto op = create_std_mean_typed_handle();
    return op.call(self, unbiased);
}

// aten::std_mean(Tensor self, bool unbiased=True) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> std_mean::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool unbiased) {
    static auto op = create_std_mean_typed_handle();
    return op.redispatch(dispatchKeySet, self, unbiased);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_mean_names_dim, name, "aten::std_mean")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_mean_names_dim, overload_name, "names_dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_mean_names_dim, schema_str, "std_mean.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)")

// aten::std_mean.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<std_mean_names_dim::schema> create_std_mean_names_dim_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(std_mean_names_dim::name, std_mean_names_dim::overload_name)
      .typed<std_mean_names_dim::schema>();
}

// aten::std_mean.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> std_mean_names_dim::call(const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim) {
    static auto op = create_std_mean_names_dim_typed_handle();
    return op.call(self, dim, unbiased, keepdim);
}

// aten::std_mean.names_dim(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> std_mean_names_dim::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim) {
    static auto op = create_std_mean_names_dim_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, unbiased, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_mean_correction_names, name, "aten::std_mean")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_mean_correction_names, overload_name, "correction_names")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_mean_correction_names, schema_str, "std_mean.correction_names(Tensor self, Dimname[1] dim, *, int? correction, bool keepdim=False) -> (Tensor, Tensor)")

// aten::std_mean.correction_names(Tensor self, Dimname[1] dim, *, int? correction, bool keepdim=False) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<std_mean_correction_names::schema> create_std_mean_correction_names_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(std_mean_correction_names::name, std_mean_correction_names::overload_name)
      .typed<std_mean_correction_names::schema>();
}

// aten::std_mean.correction_names(Tensor self, Dimname[1] dim, *, int? correction, bool keepdim=False) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> std_mean_correction_names::call(const at::Tensor & self, at::DimnameList dim, c10::optional<int64_t> correction, bool keepdim) {
    static auto op = create_std_mean_correction_names_typed_handle();
    return op.call(self, dim, correction, keepdim);
}

// aten::std_mean.correction_names(Tensor self, Dimname[1] dim, *, int? correction, bool keepdim=False) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> std_mean_correction_names::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList dim, c10::optional<int64_t> correction, bool keepdim) {
    static auto op = create_std_mean_correction_names_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, correction, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_correction_out, name, "aten::std")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_correction_out, overload_name, "correction_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_correction_out, schema_str, "std.correction_out(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)")

// aten::std.correction_out(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<std_correction_out::schema> create_std_correction_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(std_correction_out::name, std_correction_out::overload_name)
      .typed<std_correction_out::schema>();
}

// aten::std.correction_out(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)
at::Tensor & std_correction_out::call(const at::Tensor & self, c10::optional<at::IntArrayRef> dim, c10::optional<int64_t> correction, bool keepdim, at::Tensor & out) {
    static auto op = create_std_correction_out_typed_handle();
    return op.call(self, dim, correction, keepdim, out);
}

// aten::std.correction_out(Tensor self, int[1]? dim, *, int? correction, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)
at::Tensor & std_correction_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> dim, c10::optional<int64_t> correction, bool keepdim, at::Tensor & out) {
    static auto op = create_std_correction_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, correction, keepdim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_names_out, name, "aten::std")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_names_out, overload_name, "names_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_names_out, schema_str, "std.names_out(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)")

// aten::std.names_out(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<std_names_out::schema> create_std_names_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(std_names_out::name, std_names_out::overload_name)
      .typed<std_names_out::schema>();
}

// aten::std.names_out(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & std_names_out::call(const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim, at::Tensor & out) {
    static auto op = create_std_names_out_typed_handle();
    return op.call(self, dim, unbiased, keepdim, out);
}

// aten::std.names_out(Tensor self, Dimname[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & std_names_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim, at::Tensor & out) {
    static auto op = create_std_names_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, unbiased, keepdim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_correction_names, name, "aten::std")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_correction_names, overload_name, "correction_names")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(std_correction_names, schema_str, "std.correction_names(Tensor self, Dimname[1] dim, *, int? correction, bool keepdim=False) -> Tensor")

// aten::std.correction_names(Tensor self, Dimname[1] dim, *, int? correction, bool keepdim=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<std_correction_names::schema> create_std_correction_names_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(std_correction_names::name, std_correction_names::overload_name)
      .typed<std_correction_names::schema>();
}

// aten::std.correction_names(Tensor self, Dimname[1] dim, *, int? correction, bool keepdim=False) -> Tensor
at::Tensor std_correction_names::call(const at::Tensor & self, at::DimnameList dim, c10::optional<int64_t> correction, bool keepdim) {
    static auto op = create_std_correction_names_typed_handle();
    return op.call(self, dim, correction, keepdim);
}

// aten::std.correction_names(Tensor self, Dimname[1] dim, *, int? correction, bool keepdim=False) -> Tensor
at::Tensor std_correction_names::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList dim, c10::optional<int64_t> correction, bool keepdim) {
    static auto op = create_std_correction_names_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, correction, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(t, name, "aten::t")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(t, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(t, schema_str, "t(Tensor(a) self) -> Tensor(a)")

// aten::t(Tensor(a) self) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<t::schema> create_t_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(t::name, t::overload_name)
      .typed<t::schema>();
}

// aten::t(Tensor(a) self) -> Tensor(a)
at::Tensor t::call(const at::Tensor & self) {
    static auto op = create_t_typed_handle();
    return op.call(self);
}

// aten::t(Tensor(a) self) -> Tensor(a)
at::Tensor t::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_t_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tanh_, name, "aten::tanh_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tanh_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tanh_, schema_str, "tanh_(Tensor(a!) self) -> Tensor(a!)")

// aten::tanh_(Tensor(a!) self) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<tanh_::schema> create_tanh__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(tanh_::name, tanh_::overload_name)
      .typed<tanh_::schema>();
}

// aten::tanh_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & tanh_::call(at::Tensor & self) {
    static auto op = create_tanh__typed_handle();
    return op.call(self);
}

// aten::tanh_(Tensor(a!) self) -> Tensor(a!)
at::Tensor & tanh_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self) {
    static auto op = create_tanh__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(threshold, name, "aten::threshold")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(threshold, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(threshold, schema_str, "threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor")

// aten::threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<threshold::schema> create_threshold_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(threshold::name, threshold::overload_name)
      .typed<threshold::schema>();
}

// aten::threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor
at::Tensor threshold::call(const at::Tensor & self, const at::Scalar & threshold, const at::Scalar & value) {
    static auto op = create_threshold_typed_handle();
    return op.call(self, threshold, value);
}

// aten::threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor
at::Tensor threshold::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & threshold, const at::Scalar & value) {
    static auto op = create_threshold_typed_handle();
    return op.redispatch(dispatchKeySet, self, threshold, value);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(threshold_out, name, "aten::threshold")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(threshold_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(threshold_out, schema_str, "threshold.out(Tensor self, Scalar threshold, Scalar value, *, Tensor(a!) out) -> Tensor(a!)")

// aten::threshold.out(Tensor self, Scalar threshold, Scalar value, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<threshold_out::schema> create_threshold_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(threshold_out::name, threshold_out::overload_name)
      .typed<threshold_out::schema>();
}

// aten::threshold.out(Tensor self, Scalar threshold, Scalar value, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & threshold_out::call(const at::Tensor & self, const at::Scalar & threshold, const at::Scalar & value, at::Tensor & out) {
    static auto op = create_threshold_out_typed_handle();
    return op.call(self, threshold, value, out);
}

// aten::threshold.out(Tensor self, Scalar threshold, Scalar value, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & threshold_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & threshold, const at::Scalar & value, at::Tensor & out) {
    static auto op = create_threshold_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, threshold, value, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(transpose_, name, "aten::transpose_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(transpose_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(transpose_, schema_str, "transpose_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)")

// aten::transpose_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<transpose_::schema> create_transpose__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(transpose_::name, transpose_::overload_name)
      .typed<transpose_::schema>();
}

// aten::transpose_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)
at::Tensor & transpose_::call(at::Tensor & self, int64_t dim0, int64_t dim1) {
    static auto op = create_transpose__typed_handle();
    return op.call(self, dim0, dim1);
}

// aten::transpose_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)
at::Tensor & transpose_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t dim0, int64_t dim1) {
    static auto op = create_transpose__typed_handle();
    return op.redispatch(dispatchKeySet, self, dim0, dim1);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(flip, name, "aten::flip")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(flip, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(flip, schema_str, "flip(Tensor self, int[] dims) -> Tensor")

// aten::flip(Tensor self, int[] dims) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<flip::schema> create_flip_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(flip::name, flip::overload_name)
      .typed<flip::schema>();
}

// aten::flip(Tensor self, int[] dims) -> Tensor
at::Tensor flip::call(const at::Tensor & self, at::IntArrayRef dims) {
    static auto op = create_flip_typed_handle();
    return op.call(self, dims);
}

// aten::flip(Tensor self, int[] dims) -> Tensor
at::Tensor flip::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dims) {
    static auto op = create_flip_typed_handle();
    return op.redispatch(dispatchKeySet, self, dims);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(roll, name, "aten::roll")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(roll, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(roll, schema_str, "roll(Tensor self, int[1] shifts, int[1] dims=[]) -> Tensor")

// aten::roll(Tensor self, int[1] shifts, int[1] dims=[]) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<roll::schema> create_roll_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(roll::name, roll::overload_name)
      .typed<roll::schema>();
}

// aten::roll(Tensor self, int[1] shifts, int[1] dims=[]) -> Tensor
at::Tensor roll::call(const at::Tensor & self, at::IntArrayRef shifts, at::IntArrayRef dims) {
    static auto op = create_roll_typed_handle();
    return op.call(self, shifts, dims);
}

// aten::roll(Tensor self, int[1] shifts, int[1] dims=[]) -> Tensor
at::Tensor roll::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef shifts, at::IntArrayRef dims) {
    static auto op = create_roll_typed_handle();
    return op.redispatch(dispatchKeySet, self, shifts, dims);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_trilinear, name, "aten::_trilinear")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_trilinear, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_trilinear, schema_str, "_trilinear(Tensor i1, Tensor i2, Tensor i3, int[] expand1, int[] expand2, int[] expand3, int[] sumdim, int unroll_dim=1) -> Tensor")

// aten::_trilinear(Tensor i1, Tensor i2, Tensor i3, int[] expand1, int[] expand2, int[] expand3, int[] sumdim, int unroll_dim=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_trilinear::schema> create__trilinear_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_trilinear::name, _trilinear::overload_name)
      .typed<_trilinear::schema>();
}

// aten::_trilinear(Tensor i1, Tensor i2, Tensor i3, int[] expand1, int[] expand2, int[] expand3, int[] sumdim, int unroll_dim=1) -> Tensor
at::Tensor _trilinear::call(const at::Tensor & i1, const at::Tensor & i2, const at::Tensor & i3, at::IntArrayRef expand1, at::IntArrayRef expand2, at::IntArrayRef expand3, at::IntArrayRef sumdim, int64_t unroll_dim) {
    static auto op = create__trilinear_typed_handle();
    return op.call(i1, i2, i3, expand1, expand2, expand3, sumdim, unroll_dim);
}

// aten::_trilinear(Tensor i1, Tensor i2, Tensor i3, int[] expand1, int[] expand2, int[] expand3, int[] sumdim, int unroll_dim=1) -> Tensor
at::Tensor _trilinear::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & i1, const at::Tensor & i2, const at::Tensor & i3, at::IntArrayRef expand1, at::IntArrayRef expand2, at::IntArrayRef expand3, at::IntArrayRef sumdim, int64_t unroll_dim) {
    static auto op = create__trilinear_typed_handle();
    return op.redispatch(dispatchKeySet, i1, i2, i3, expand1, expand2, expand3, sumdim, unroll_dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fix_out, name, "aten::fix")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fix_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fix_out, schema_str, "fix.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::fix.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<fix_out::schema> create_fix_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fix_out::name, fix_out::overload_name)
      .typed<fix_out::schema>();
}

// aten::fix.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fix_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_fix_out_typed_handle();
    return op.call(self, out);
}

// aten::fix.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fix_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_fix_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(type_as, name, "aten::type_as")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(type_as, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(type_as, schema_str, "type_as(Tensor self, Tensor other) -> Tensor")

// aten::type_as(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<type_as::schema> create_type_as_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(type_as::name, type_as::overload_name)
      .typed<type_as::schema>();
}

// aten::type_as(Tensor self, Tensor other) -> Tensor
at::Tensor type_as::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_type_as_typed_handle();
    return op.call(self, other);
}

// aten::type_as(Tensor self, Tensor other) -> Tensor
at::Tensor type_as::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_type_as_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_has_compatible_shallow_copy_type, name, "aten::_has_compatible_shallow_copy_type")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_has_compatible_shallow_copy_type, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_has_compatible_shallow_copy_type, schema_str, "_has_compatible_shallow_copy_type(Tensor self, Tensor from) -> bool")

// aten::_has_compatible_shallow_copy_type(Tensor self, Tensor from) -> bool
static C10_NOINLINE c10::TypedOperatorHandle<_has_compatible_shallow_copy_type::schema> create__has_compatible_shallow_copy_type_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_has_compatible_shallow_copy_type::name, _has_compatible_shallow_copy_type::overload_name)
      .typed<_has_compatible_shallow_copy_type::schema>();
}

// aten::_has_compatible_shallow_copy_type(Tensor self, Tensor from) -> bool
bool _has_compatible_shallow_copy_type::call(const at::Tensor & self, const at::Tensor & from) {
    static auto op = create__has_compatible_shallow_copy_type_typed_handle();
    return op.call(self, from);
}

// aten::_has_compatible_shallow_copy_type(Tensor self, Tensor from) -> bool
bool _has_compatible_shallow_copy_type::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & from) {
    static auto op = create__has_compatible_shallow_copy_type_typed_handle();
    return op.redispatch(dispatchKeySet, self, from);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_unique2, name, "aten::_unique2")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_unique2, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_unique2, schema_str, "_unique2(Tensor self, bool sorted=True, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)")

// aten::_unique2(Tensor self, bool sorted=True, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<_unique2::schema> create__unique2_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_unique2::name, _unique2::overload_name)
      .typed<_unique2::schema>();
}

// aten::_unique2(Tensor self, bool sorted=True, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _unique2::call(const at::Tensor & self, bool sorted, bool return_inverse, bool return_counts) {
    static auto op = create__unique2_typed_handle();
    return op.call(self, sorted, return_inverse, return_counts);
}

// aten::_unique2(Tensor self, bool sorted=True, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _unique2::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool sorted, bool return_inverse, bool return_counts) {
    static auto op = create__unique2_typed_handle();
    return op.redispatch(dispatchKeySet, self, sorted, return_inverse, return_counts);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_dim, name, "aten::var")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_dim, overload_name, "dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_dim, schema_str, "var.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor")

// aten::var.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<var_dim::schema> create_var_dim_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(var_dim::name, var_dim::overload_name)
      .typed<var_dim::schema>();
}

// aten::var.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
at::Tensor var_dim::call(const at::Tensor & self, at::IntArrayRef dim, bool unbiased, bool keepdim) {
    static auto op = create_var_dim_typed_handle();
    return op.call(self, dim, unbiased, keepdim);
}

// aten::var.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
at::Tensor var_dim::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool unbiased, bool keepdim) {
    static auto op = create_var_dim_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, unbiased, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_correction_names, name, "aten::var")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_correction_names, overload_name, "correction_names")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_correction_names, schema_str, "var.correction_names(Tensor self, Dimname[1] dim, *, int? correction, bool keepdim=False) -> Tensor")

// aten::var.correction_names(Tensor self, Dimname[1] dim, *, int? correction, bool keepdim=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<var_correction_names::schema> create_var_correction_names_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(var_correction_names::name, var_correction_names::overload_name)
      .typed<var_correction_names::schema>();
}

// aten::var.correction_names(Tensor self, Dimname[1] dim, *, int? correction, bool keepdim=False) -> Tensor
at::Tensor var_correction_names::call(const at::Tensor & self, at::DimnameList dim, c10::optional<int64_t> correction, bool keepdim) {
    static auto op = create_var_correction_names_typed_handle();
    return op.call(self, dim, correction, keepdim);
}

// aten::var.correction_names(Tensor self, Dimname[1] dim, *, int? correction, bool keepdim=False) -> Tensor
at::Tensor var_correction_names::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::DimnameList dim, c10::optional<int64_t> correction, bool keepdim) {
    static auto op = create_var_correction_names_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, correction, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_mean_dim, name, "aten::var_mean")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_mean_dim, overload_name, "dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(var_mean_dim, schema_str, "var_mean.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)")

// aten::var_mean.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<var_mean_dim::schema> create_var_mean_dim_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(var_mean_dim::name, var_mean_dim::overload_name)
      .typed<var_mean_dim::schema>();
}

// aten::var_mean.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> var_mean_dim::call(const at::Tensor & self, at::IntArrayRef dim, bool unbiased, bool keepdim) {
    static auto op = create_var_mean_dim_typed_handle();
    return op.call(self, dim, unbiased, keepdim);
}

// aten::var_mean.dim(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> var_mean_dim::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool unbiased, bool keepdim) {
    static auto op = create_var_mean_dim_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, unbiased, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_weight_norm_cuda_interface_backward, name, "aten::_weight_norm_cuda_interface_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_weight_norm_cuda_interface_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_weight_norm_cuda_interface_backward, schema_str, "_weight_norm_cuda_interface_backward(Tensor grad_w, Tensor saved_v, Tensor saved_g, Tensor saved_norms, int dim) -> (Tensor, Tensor)")

// aten::_weight_norm_cuda_interface_backward(Tensor grad_w, Tensor saved_v, Tensor saved_g, Tensor saved_norms, int dim) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<_weight_norm_cuda_interface_backward::schema> create__weight_norm_cuda_interface_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_weight_norm_cuda_interface_backward::name, _weight_norm_cuda_interface_backward::overload_name)
      .typed<_weight_norm_cuda_interface_backward::schema>();
}

// aten::_weight_norm_cuda_interface_backward(Tensor grad_w, Tensor saved_v, Tensor saved_g, Tensor saved_norms, int dim) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> _weight_norm_cuda_interface_backward::call(const at::Tensor & grad_w, const at::Tensor & saved_v, const at::Tensor & saved_g, const at::Tensor & saved_norms, int64_t dim) {
    static auto op = create__weight_norm_cuda_interface_backward_typed_handle();
    return op.call(grad_w, saved_v, saved_g, saved_norms, dim);
}

// aten::_weight_norm_cuda_interface_backward(Tensor grad_w, Tensor saved_v, Tensor saved_g, Tensor saved_norms, int dim) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> _weight_norm_cuda_interface_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_w, const at::Tensor & saved_v, const at::Tensor & saved_g, const at::Tensor & saved_norms, int64_t dim) {
    static auto op = create__weight_norm_cuda_interface_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_w, saved_v, saved_g, saved_norms, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(zeros_names, name, "aten::zeros")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(zeros_names, overload_name, "names")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(zeros_names, schema_str, "zeros.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor")

// aten::zeros.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<zeros_names::schema> create_zeros_names_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(zeros_names::name, zeros_names::overload_name)
      .typed<zeros_names::schema>();
}

// aten::zeros.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor zeros_names::call(at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_zeros_names_typed_handle();
    return op.call(size, names, dtype, layout, device, pin_memory);
}

// aten::zeros.names(int[] size, *, Dimname[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
at::Tensor zeros_names::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_zeros_names_typed_handle();
    return op.redispatch(dispatchKeySet, size, names, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(zeros_like, name, "aten::zeros_like")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(zeros_like, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(zeros_like, schema_str, "zeros_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor")

// aten::zeros_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<zeros_like::schema> create_zeros_like_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(zeros_like::name, zeros_like::overload_name)
      .typed<zeros_like::schema>();
}

// aten::zeros_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
at::Tensor zeros_like::call(const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create_zeros_like_typed_handle();
    return op.call(self, dtype, layout, device, pin_memory, memory_format);
}

// aten::zeros_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
at::Tensor zeros_like::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create_zeros_like_typed_handle();
    return op.redispatch(dispatchKeySet, self, dtype, layout, device, pin_memory, memory_format);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_sum_dtype, name, "aten::_sparse_sum")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_sum_dtype, overload_name, "dtype")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_sum_dtype, schema_str, "_sparse_sum.dtype(Tensor self, *, ScalarType dtype) -> Tensor")

// aten::_sparse_sum.dtype(Tensor self, *, ScalarType dtype) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_sparse_sum_dtype::schema> create__sparse_sum_dtype_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_sparse_sum_dtype::name, _sparse_sum_dtype::overload_name)
      .typed<_sparse_sum_dtype::schema>();
}

// aten::_sparse_sum.dtype(Tensor self, *, ScalarType dtype) -> Tensor
at::Tensor _sparse_sum_dtype::call(const at::Tensor & self, at::ScalarType dtype) {
    static auto op = create__sparse_sum_dtype_typed_handle();
    return op.call(self, dtype);
}

// aten::_sparse_sum.dtype(Tensor self, *, ScalarType dtype) -> Tensor
at::Tensor _sparse_sum_dtype::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::ScalarType dtype) {
    static auto op = create__sparse_sum_dtype_typed_handle();
    return op.redispatch(dispatchKeySet, self, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_softmax_backward_data, name, "aten::_sparse_softmax_backward_data")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_softmax_backward_data, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_softmax_backward_data, schema_str, "_sparse_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor")

// aten::_sparse_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_sparse_softmax_backward_data::schema> create__sparse_softmax_backward_data_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_sparse_softmax_backward_data::name, _sparse_softmax_backward_data::overload_name)
      .typed<_sparse_softmax_backward_data::schema>();
}

// aten::_sparse_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor
at::Tensor _sparse_softmax_backward_data::call(const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, const at::Tensor & self) {
    static auto op = create__sparse_softmax_backward_data_typed_handle();
    return op.call(grad_output, output, dim, self);
}

// aten::_sparse_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor
at::Tensor _sparse_softmax_backward_data::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, const at::Tensor & self) {
    static auto op = create__sparse_softmax_backward_data_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, output, dim, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_log_softmax_int, name, "aten::_sparse_log_softmax")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_log_softmax_int, overload_name, "int")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_log_softmax_int, schema_str, "_sparse_log_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor")

// aten::_sparse_log_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_sparse_log_softmax_int::schema> create__sparse_log_softmax_int_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_sparse_log_softmax_int::name, _sparse_log_softmax_int::overload_name)
      .typed<_sparse_log_softmax_int::schema>();
}

// aten::_sparse_log_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
at::Tensor _sparse_log_softmax_int::call(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
    static auto op = create__sparse_log_softmax_int_typed_handle();
    return op.call(self, dim, dtype);
}

// aten::_sparse_log_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
at::Tensor _sparse_log_softmax_int::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
    static auto op = create__sparse_log_softmax_int_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_log_softmax, name, "aten::_sparse_log_softmax")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_log_softmax, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_log_softmax, schema_str, "_sparse_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor")

// aten::_sparse_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_sparse_log_softmax::schema> create__sparse_log_softmax_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_sparse_log_softmax::name, _sparse_log_softmax::overload_name)
      .typed<_sparse_log_softmax::schema>();
}

// aten::_sparse_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
at::Tensor _sparse_log_softmax::call(const at::Tensor & self, int64_t dim, bool half_to_float) {
    static auto op = create__sparse_log_softmax_typed_handle();
    return op.call(self, dim, half_to_float);
}

// aten::_sparse_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
at::Tensor _sparse_log_softmax::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, bool half_to_float) {
    static auto op = create__sparse_log_softmax_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, half_to_float);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_log_softmax_backward_data, name, "aten::_sparse_log_softmax_backward_data")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_log_softmax_backward_data, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_log_softmax_backward_data, schema_str, "_sparse_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor")

// aten::_sparse_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_sparse_log_softmax_backward_data::schema> create__sparse_log_softmax_backward_data_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_sparse_log_softmax_backward_data::name, _sparse_log_softmax_backward_data::overload_name)
      .typed<_sparse_log_softmax_backward_data::schema>();
}

// aten::_sparse_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor
at::Tensor _sparse_log_softmax_backward_data::call(const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, const at::Tensor & self) {
    static auto op = create__sparse_log_softmax_backward_data_typed_handle();
    return op.call(grad_output, output, dim, self);
}

// aten::_sparse_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor
at::Tensor _sparse_log_softmax_backward_data::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, const at::Tensor & self) {
    static auto op = create__sparse_log_softmax_backward_data_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, output, dim, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(norm_ScalarOpt_dtype, name, "aten::norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(norm_ScalarOpt_dtype, overload_name, "ScalarOpt_dtype")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(norm_ScalarOpt_dtype, schema_str, "norm.ScalarOpt_dtype(Tensor self, Scalar? p, *, ScalarType dtype) -> Tensor")

// aten::norm.ScalarOpt_dtype(Tensor self, Scalar? p, *, ScalarType dtype) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<norm_ScalarOpt_dtype::schema> create_norm_ScalarOpt_dtype_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(norm_ScalarOpt_dtype::name, norm_ScalarOpt_dtype::overload_name)
      .typed<norm_ScalarOpt_dtype::schema>();
}

// aten::norm.ScalarOpt_dtype(Tensor self, Scalar? p, *, ScalarType dtype) -> Tensor
at::Tensor norm_ScalarOpt_dtype::call(const at::Tensor & self, const c10::optional<at::Scalar> & p, at::ScalarType dtype) {
    static auto op = create_norm_ScalarOpt_dtype_typed_handle();
    return op.call(self, p, dtype);
}

// aten::norm.ScalarOpt_dtype(Tensor self, Scalar? p, *, ScalarType dtype) -> Tensor
at::Tensor norm_ScalarOpt_dtype::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Scalar> & p, at::ScalarType dtype) {
    static auto op = create_norm_ScalarOpt_dtype_typed_handle();
    return op.redispatch(dispatchKeySet, self, p, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(norm_out, name, "aten::norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(norm_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(norm_out, schema_str, "norm.out(Tensor self, Scalar? p, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)")

// aten::norm.out(Tensor self, Scalar? p, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<norm_out::schema> create_norm_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(norm_out::name, norm_out::overload_name)
      .typed<norm_out::schema>();
}

// aten::norm.out(Tensor self, Scalar? p, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & norm_out::call(const at::Tensor & self, const c10::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
    static auto op = create_norm_out_typed_handle();
    return op.call(self, p, dim, keepdim, out);
}

// aten::norm.out(Tensor self, Scalar? p, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & norm_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
    static auto op = create_norm_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, p, dim, keepdim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(norm_names_ScalarOpt_dim_dtype, name, "aten::norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(norm_names_ScalarOpt_dim_dtype, overload_name, "names_ScalarOpt_dim_dtype")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(norm_names_ScalarOpt_dim_dtype, schema_str, "norm.names_ScalarOpt_dim_dtype(Tensor self, Scalar? p, Dimname[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor")

// aten::norm.names_ScalarOpt_dim_dtype(Tensor self, Scalar? p, Dimname[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<norm_names_ScalarOpt_dim_dtype::schema> create_norm_names_ScalarOpt_dim_dtype_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(norm_names_ScalarOpt_dim_dtype::name, norm_names_ScalarOpt_dim_dtype::overload_name)
      .typed<norm_names_ScalarOpt_dim_dtype::schema>();
}

// aten::norm.names_ScalarOpt_dim_dtype(Tensor self, Scalar? p, Dimname[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor
at::Tensor norm_names_ScalarOpt_dim_dtype::call(const at::Tensor & self, const c10::optional<at::Scalar> & p, at::DimnameList dim, bool keepdim, at::ScalarType dtype) {
    static auto op = create_norm_names_ScalarOpt_dim_dtype_typed_handle();
    return op.call(self, p, dim, keepdim, dtype);
}

// aten::norm.names_ScalarOpt_dim_dtype(Tensor self, Scalar? p, Dimname[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor
at::Tensor norm_names_ScalarOpt_dim_dtype::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Scalar> & p, at::DimnameList dim, bool keepdim, at::ScalarType dtype) {
    static auto op = create_norm_names_ScalarOpt_dim_dtype_typed_handle();
    return op.redispatch(dispatchKeySet, self, p, dim, keepdim, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(frobenius_norm_dim, name, "aten::frobenius_norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(frobenius_norm_dim, overload_name, "dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(frobenius_norm_dim, schema_str, "frobenius_norm.dim(Tensor self, int[1] dim, bool keepdim=False) -> Tensor")

// aten::frobenius_norm.dim(Tensor self, int[1] dim, bool keepdim=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<frobenius_norm_dim::schema> create_frobenius_norm_dim_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(frobenius_norm_dim::name, frobenius_norm_dim::overload_name)
      .typed<frobenius_norm_dim::schema>();
}

// aten::frobenius_norm.dim(Tensor self, int[1] dim, bool keepdim=False) -> Tensor
at::Tensor frobenius_norm_dim::call(const at::Tensor & self, at::IntArrayRef dim, bool keepdim) {
    static auto op = create_frobenius_norm_dim_typed_handle();
    return op.call(self, dim, keepdim);
}

// aten::frobenius_norm.dim(Tensor self, int[1] dim, bool keepdim=False) -> Tensor
at::Tensor frobenius_norm_dim::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool keepdim) {
    static auto op = create_frobenius_norm_dim_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nuclear_norm_out, name, "aten::nuclear_norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nuclear_norm_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nuclear_norm_out, schema_str, "nuclear_norm.out(Tensor self, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)")

// aten::nuclear_norm.out(Tensor self, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<nuclear_norm_out::schema> create_nuclear_norm_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nuclear_norm_out::name, nuclear_norm_out::overload_name)
      .typed<nuclear_norm_out::schema>();
}

// aten::nuclear_norm.out(Tensor self, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & nuclear_norm_out::call(const at::Tensor & self, bool keepdim, at::Tensor & out) {
    static auto op = create_nuclear_norm_out_typed_handle();
    return op.call(self, keepdim, out);
}

// aten::nuclear_norm.out(Tensor self, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & nuclear_norm_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool keepdim, at::Tensor & out) {
    static auto op = create_nuclear_norm_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, keepdim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nuclear_norm_dim, name, "aten::nuclear_norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nuclear_norm_dim, overload_name, "dim")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nuclear_norm_dim, schema_str, "nuclear_norm.dim(Tensor self, int[2] dim, bool keepdim=False) -> Tensor")

// aten::nuclear_norm.dim(Tensor self, int[2] dim, bool keepdim=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<nuclear_norm_dim::schema> create_nuclear_norm_dim_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nuclear_norm_dim::name, nuclear_norm_dim::overload_name)
      .typed<nuclear_norm_dim::schema>();
}

// aten::nuclear_norm.dim(Tensor self, int[2] dim, bool keepdim=False) -> Tensor
at::Tensor nuclear_norm_dim::call(const at::Tensor & self, at::IntArrayRef dim, bool keepdim) {
    static auto op = create_nuclear_norm_dim_typed_handle();
    return op.call(self, dim, keepdim);
}

// aten::nuclear_norm.dim(Tensor self, int[2] dim, bool keepdim=False) -> Tensor
at::Tensor nuclear_norm_dim::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool keepdim) {
    static auto op = create_nuclear_norm_dim_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nuclear_norm_dim_out, name, "aten::nuclear_norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nuclear_norm_dim_out, overload_name, "dim_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nuclear_norm_dim_out, schema_str, "nuclear_norm.dim_out(Tensor self, int[2] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)")

// aten::nuclear_norm.dim_out(Tensor self, int[2] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<nuclear_norm_dim_out::schema> create_nuclear_norm_dim_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nuclear_norm_dim_out::name, nuclear_norm_dim_out::overload_name)
      .typed<nuclear_norm_dim_out::schema>();
}

// aten::nuclear_norm.dim_out(Tensor self, int[2] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & nuclear_norm_dim_out::call(const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
    static auto op = create_nuclear_norm_dim_out_typed_handle();
    return op.call(self, dim, keepdim, out);
}

// aten::nuclear_norm.dim_out(Tensor self, int[2] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & nuclear_norm_dim_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
    static auto op = create_nuclear_norm_dim_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, keepdim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(resize_as_sparse_, name, "aten::resize_as_sparse_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(resize_as_sparse_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(resize_as_sparse_, schema_str, "resize_as_sparse_(Tensor(a!) self, Tensor the_template) -> Tensor(a!)")

// aten::resize_as_sparse_(Tensor(a!) self, Tensor the_template) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<resize_as_sparse_::schema> create_resize_as_sparse__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(resize_as_sparse_::name, resize_as_sparse_::overload_name)
      .typed<resize_as_sparse_::schema>();
}

// aten::resize_as_sparse_(Tensor(a!) self, Tensor the_template) -> Tensor(a!)
const at::Tensor & resize_as_sparse_::call(const at::Tensor & self, const at::Tensor & the_template) {
    static auto op = create_resize_as_sparse__typed_handle();
    return op.call(self, the_template);
}

// aten::resize_as_sparse_(Tensor(a!) self, Tensor the_template) -> Tensor(a!)
const at::Tensor & resize_as_sparse_::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & the_template) {
    static auto op = create_resize_as_sparse__typed_handle();
    return op.redispatch(dispatchKeySet, self, the_template);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sub_out, name, "aten::sub")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sub_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sub_out, schema_str, "sub.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)")

// aten::sub.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<sub_out::schema> create_sub_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sub_out::name, sub_out::overload_name)
      .typed<sub_out::schema>();
}

// aten::sub.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
at::Tensor & sub_out::call(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
    static auto op = create_sub_out_typed_handle();
    return op.call(self, other, alpha, out);
}

// aten::sub.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
at::Tensor & sub_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
    static auto op = create_sub_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, alpha, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(subtract_out, name, "aten::subtract")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(subtract_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(subtract_out, schema_str, "subtract.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)")

// aten::subtract.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<subtract_out::schema> create_subtract_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(subtract_out::name, subtract_out::overload_name)
      .typed<subtract_out::schema>();
}

// aten::subtract.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
at::Tensor & subtract_out::call(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
    static auto op = create_subtract_out_typed_handle();
    return op.call(self, other, alpha, out);
}

// aten::subtract.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
at::Tensor & subtract_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
    static auto op = create_subtract_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, alpha, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sparse_csr_tensor_crow_col_value_size, name, "aten::sparse_csr_tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sparse_csr_tensor_crow_col_value_size, overload_name, "crow_col_value_size")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sparse_csr_tensor_crow_col_value_size, schema_str, "sparse_csr_tensor.crow_col_value_size(Tensor crow_indices, Tensor col_indices, Tensor values, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor")

// aten::sparse_csr_tensor.crow_col_value_size(Tensor crow_indices, Tensor col_indices, Tensor values, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<sparse_csr_tensor_crow_col_value_size::schema> create_sparse_csr_tensor_crow_col_value_size_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sparse_csr_tensor_crow_col_value_size::name, sparse_csr_tensor_crow_col_value_size::overload_name)
      .typed<sparse_csr_tensor_crow_col_value_size::schema>();
}

// aten::sparse_csr_tensor.crow_col_value_size(Tensor crow_indices, Tensor col_indices, Tensor values, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
at::Tensor sparse_csr_tensor_crow_col_value_size::call(const at::Tensor & crow_indices, const at::Tensor & col_indices, const at::Tensor & values, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_sparse_csr_tensor_crow_col_value_size_typed_handle();
    return op.call(crow_indices, col_indices, values, size, dtype, layout, device, pin_memory);
}

// aten::sparse_csr_tensor.crow_col_value_size(Tensor crow_indices, Tensor col_indices, Tensor values, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
at::Tensor sparse_csr_tensor_crow_col_value_size::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & crow_indices, const at::Tensor & col_indices, const at::Tensor & values, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_sparse_csr_tensor_crow_col_value_size_typed_handle();
    return op.redispatch(dispatchKeySet, crow_indices, col_indices, values, size, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sparse_coo_tensor_size, name, "aten::sparse_coo_tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sparse_coo_tensor_size, overload_name, "size")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sparse_coo_tensor_size, schema_str, "sparse_coo_tensor.size(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor")

// aten::sparse_coo_tensor.size(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<sparse_coo_tensor_size::schema> create_sparse_coo_tensor_size_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sparse_coo_tensor_size::name, sparse_coo_tensor_size::overload_name)
      .typed<sparse_coo_tensor_size::schema>();
}

// aten::sparse_coo_tensor.size(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
at::Tensor sparse_coo_tensor_size::call(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_sparse_coo_tensor_size_typed_handle();
    return op.call(size, dtype, layout, device, pin_memory);
}

// aten::sparse_coo_tensor.size(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
at::Tensor sparse_coo_tensor_size::redispatch(c10::DispatchKeySet dispatchKeySet, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create_sparse_coo_tensor_size_typed_handle();
    return op.redispatch(dispatchKeySet, size, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_coo_tensor_with_dims, name, "aten::_sparse_coo_tensor_with_dims")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_coo_tensor_with_dims, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_sparse_coo_tensor_with_dims, schema_str, "_sparse_coo_tensor_with_dims(int sparse_dim, int dense_dim, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor")

// aten::_sparse_coo_tensor_with_dims(int sparse_dim, int dense_dim, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_sparse_coo_tensor_with_dims::schema> create__sparse_coo_tensor_with_dims_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_sparse_coo_tensor_with_dims::name, _sparse_coo_tensor_with_dims::overload_name)
      .typed<_sparse_coo_tensor_with_dims::schema>();
}

// aten::_sparse_coo_tensor_with_dims(int sparse_dim, int dense_dim, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
at::Tensor _sparse_coo_tensor_with_dims::call(int64_t sparse_dim, int64_t dense_dim, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create__sparse_coo_tensor_with_dims_typed_handle();
    return op.call(sparse_dim, dense_dim, size, dtype, layout, device, pin_memory);
}

// aten::_sparse_coo_tensor_with_dims(int sparse_dim, int dense_dim, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
at::Tensor _sparse_coo_tensor_with_dims::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t sparse_dim, int64_t dense_dim, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = create__sparse_coo_tensor_with_dims_typed_handle();
    return op.redispatch(dispatchKeySet, sparse_dim, dense_dim, size, dtype, layout, device, pin_memory);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(to_dense_backward, name, "aten::to_dense_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(to_dense_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(to_dense_backward, schema_str, "to_dense_backward(Tensor grad, Tensor input) -> Tensor")

// aten::to_dense_backward(Tensor grad, Tensor input) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<to_dense_backward::schema> create_to_dense_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(to_dense_backward::name, to_dense_backward::overload_name)
      .typed<to_dense_backward::schema>();
}

// aten::to_dense_backward(Tensor grad, Tensor input) -> Tensor
at::Tensor to_dense_backward::call(const at::Tensor & grad, const at::Tensor & input) {
    static auto op = create_to_dense_backward_typed_handle();
    return op.call(grad, input);
}

// aten::to_dense_backward(Tensor grad, Tensor input) -> Tensor
at::Tensor to_dense_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & input) {
    static auto op = create_to_dense_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad, input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_coalesce, name, "aten::_coalesce")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_coalesce, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_coalesce, schema_str, "_coalesce(Tensor self) -> Tensor")

// aten::_coalesce(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_coalesce::schema> create__coalesce_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_coalesce::name, _coalesce::overload_name)
      .typed<_coalesce::schema>();
}

// aten::_coalesce(Tensor self) -> Tensor
at::Tensor _coalesce::call(const at::Tensor & self) {
    static auto op = create__coalesce_typed_handle();
    return op.call(self);
}

// aten::_coalesce(Tensor self) -> Tensor
at::Tensor _coalesce::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create__coalesce_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_values, name, "aten::_values")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_values, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_values, schema_str, "_values(Tensor(a) self) -> Tensor(a)")

// aten::_values(Tensor(a) self) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<_values::schema> create__values_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_values::name, _values::overload_name)
      .typed<_values::schema>();
}

// aten::_values(Tensor(a) self) -> Tensor(a)
at::Tensor _values::call(const at::Tensor & self) {
    static auto op = create__values_typed_handle();
    return op.call(self);
}

// aten::_values(Tensor(a) self) -> Tensor(a)
at::Tensor _values::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create__values_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(crow_indices, name, "aten::crow_indices")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(crow_indices, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(crow_indices, schema_str, "crow_indices(Tensor(a) self) -> Tensor(a)")

// aten::crow_indices(Tensor(a) self) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<crow_indices::schema> create_crow_indices_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(crow_indices::name, crow_indices::overload_name)
      .typed<crow_indices::schema>();
}

// aten::crow_indices(Tensor(a) self) -> Tensor(a)
at::Tensor crow_indices::call(const at::Tensor & self) {
    static auto op = create_crow_indices_typed_handle();
    return op.call(self);
}

// aten::crow_indices(Tensor(a) self) -> Tensor(a)
at::Tensor crow_indices::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_crow_indices_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unbind_Dimname, name, "aten::unbind")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unbind_Dimname, overload_name, "Dimname")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(unbind_Dimname, schema_str, "unbind.Dimname(Tensor(a) self, Dimname dim) -> Tensor(a)[]")

// aten::unbind.Dimname(Tensor(a) self, Dimname dim) -> Tensor(a)[]
static C10_NOINLINE c10::TypedOperatorHandle<unbind_Dimname::schema> create_unbind_Dimname_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(unbind_Dimname::name, unbind_Dimname::overload_name)
      .typed<unbind_Dimname::schema>();
}

// aten::unbind.Dimname(Tensor(a) self, Dimname dim) -> Tensor(a)[]
::std::vector<at::Tensor> unbind_Dimname::call(const at::Tensor & self, at::Dimname dim) {
    static auto op = create_unbind_Dimname_typed_handle();
    return op.call(self, dim);
}

// aten::unbind.Dimname(Tensor(a) self, Dimname dim) -> Tensor(a)[]
::std::vector<at::Tensor> unbind_Dimname::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim) {
    static auto op = create_unbind_Dimname_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantize_per_tensor_tensor_qparams, name, "aten::quantize_per_tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantize_per_tensor_tensor_qparams, overload_name, "tensor_qparams")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantize_per_tensor_tensor_qparams, schema_str, "quantize_per_tensor.tensor_qparams(Tensor self, Tensor scale, Tensor zero_point, ScalarType dtype) -> Tensor")

// aten::quantize_per_tensor.tensor_qparams(Tensor self, Tensor scale, Tensor zero_point, ScalarType dtype) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<quantize_per_tensor_tensor_qparams::schema> create_quantize_per_tensor_tensor_qparams_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(quantize_per_tensor_tensor_qparams::name, quantize_per_tensor_tensor_qparams::overload_name)
      .typed<quantize_per_tensor_tensor_qparams::schema>();
}

// aten::quantize_per_tensor.tensor_qparams(Tensor self, Tensor scale, Tensor zero_point, ScalarType dtype) -> Tensor
at::Tensor quantize_per_tensor_tensor_qparams::call(const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, at::ScalarType dtype) {
    static auto op = create_quantize_per_tensor_tensor_qparams_typed_handle();
    return op.call(self, scale, zero_point, dtype);
}

// aten::quantize_per_tensor.tensor_qparams(Tensor self, Tensor scale, Tensor zero_point, ScalarType dtype) -> Tensor
at::Tensor quantize_per_tensor_tensor_qparams::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, at::ScalarType dtype) {
    static auto op = create_quantize_per_tensor_tensor_qparams_typed_handle();
    return op.redispatch(dispatchKeySet, self, scale, zero_point, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(q_zero_point, name, "aten::q_zero_point")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(q_zero_point, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(q_zero_point, schema_str, "q_zero_point(Tensor self) -> int")

// aten::q_zero_point(Tensor self) -> int
static C10_NOINLINE c10::TypedOperatorHandle<q_zero_point::schema> create_q_zero_point_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(q_zero_point::name, q_zero_point::overload_name)
      .typed<q_zero_point::schema>();
}

// aten::q_zero_point(Tensor self) -> int
int64_t q_zero_point::call(const at::Tensor & self) {
    static auto op = create_q_zero_point_typed_handle();
    return op.call(self);
}

// aten::q_zero_point(Tensor self) -> int
int64_t q_zero_point::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_q_zero_point_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(q_per_channel_scales, name, "aten::q_per_channel_scales")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(q_per_channel_scales, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(q_per_channel_scales, schema_str, "q_per_channel_scales(Tensor self) -> Tensor")

// aten::q_per_channel_scales(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<q_per_channel_scales::schema> create_q_per_channel_scales_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(q_per_channel_scales::name, q_per_channel_scales::overload_name)
      .typed<q_per_channel_scales::schema>();
}

// aten::q_per_channel_scales(Tensor self) -> Tensor
at::Tensor q_per_channel_scales::call(const at::Tensor & self) {
    static auto op = create_q_per_channel_scales_typed_handle();
    return op.call(self);
}

// aten::q_per_channel_scales(Tensor self) -> Tensor
at::Tensor q_per_channel_scales::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_q_per_channel_scales_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fake_quantize_per_tensor_affine_tensor_qparams, name, "aten::fake_quantize_per_tensor_affine")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fake_quantize_per_tensor_affine_tensor_qparams, overload_name, "tensor_qparams")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fake_quantize_per_tensor_affine_tensor_qparams, schema_str, "fake_quantize_per_tensor_affine.tensor_qparams(Tensor self, Tensor scale, Tensor zero_point, int quant_min, int quant_max) -> Tensor")

// aten::fake_quantize_per_tensor_affine.tensor_qparams(Tensor self, Tensor scale, Tensor zero_point, int quant_min, int quant_max) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fake_quantize_per_tensor_affine_tensor_qparams::schema> create_fake_quantize_per_tensor_affine_tensor_qparams_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fake_quantize_per_tensor_affine_tensor_qparams::name, fake_quantize_per_tensor_affine_tensor_qparams::overload_name)
      .typed<fake_quantize_per_tensor_affine_tensor_qparams::schema>();
}

// aten::fake_quantize_per_tensor_affine.tensor_qparams(Tensor self, Tensor scale, Tensor zero_point, int quant_min, int quant_max) -> Tensor
at::Tensor fake_quantize_per_tensor_affine_tensor_qparams::call(const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t quant_min, int64_t quant_max) {
    static auto op = create_fake_quantize_per_tensor_affine_tensor_qparams_typed_handle();
    return op.call(self, scale, zero_point, quant_min, quant_max);
}

// aten::fake_quantize_per_tensor_affine.tensor_qparams(Tensor self, Tensor scale, Tensor zero_point, int quant_min, int quant_max) -> Tensor
at::Tensor fake_quantize_per_tensor_affine_tensor_qparams::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t quant_min, int64_t quant_max) {
    static auto op = create_fake_quantize_per_tensor_affine_tensor_qparams_typed_handle();
    return op.redispatch(dispatchKeySet, self, scale, zero_point, quant_min, quant_max);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fake_quantize_learnable_per_tensor_affine_backward, name, "aten::_fake_quantize_learnable_per_tensor_affine_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fake_quantize_learnable_per_tensor_affine_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fake_quantize_learnable_per_tensor_affine_backward, schema_str, "_fake_quantize_learnable_per_tensor_affine_backward(Tensor grad, Tensor self, Tensor scale, Tensor zero_point, int quant_min, int quant_max, float grad_factor=1.0) -> (Tensor, Tensor, Tensor)")

// aten::_fake_quantize_learnable_per_tensor_affine_backward(Tensor grad, Tensor self, Tensor scale, Tensor zero_point, int quant_min, int quant_max, float grad_factor=1.0) -> (Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<_fake_quantize_learnable_per_tensor_affine_backward::schema> create__fake_quantize_learnable_per_tensor_affine_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_fake_quantize_learnable_per_tensor_affine_backward::name, _fake_quantize_learnable_per_tensor_affine_backward::overload_name)
      .typed<_fake_quantize_learnable_per_tensor_affine_backward::schema>();
}

// aten::_fake_quantize_learnable_per_tensor_affine_backward(Tensor grad, Tensor self, Tensor scale, Tensor zero_point, int quant_min, int quant_max, float grad_factor=1.0) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _fake_quantize_learnable_per_tensor_affine_backward::call(const at::Tensor & grad, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t quant_min, int64_t quant_max, double grad_factor) {
    static auto op = create__fake_quantize_learnable_per_tensor_affine_backward_typed_handle();
    return op.call(grad, self, scale, zero_point, quant_min, quant_max, grad_factor);
}

// aten::_fake_quantize_learnable_per_tensor_affine_backward(Tensor grad, Tensor self, Tensor scale, Tensor zero_point, int quant_min, int quant_max, float grad_factor=1.0) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _fake_quantize_learnable_per_tensor_affine_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t quant_min, int64_t quant_max, double grad_factor) {
    static auto op = create__fake_quantize_learnable_per_tensor_affine_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad, self, scale, zero_point, quant_min, quant_max, grad_factor);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fake_quantize_learnable_per_channel_affine_backward, name, "aten::_fake_quantize_learnable_per_channel_affine_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fake_quantize_learnable_per_channel_affine_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_fake_quantize_learnable_per_channel_affine_backward, schema_str, "_fake_quantize_learnable_per_channel_affine_backward(Tensor grad, Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max, float grad_factor=1.0) -> (Tensor, Tensor, Tensor)")

// aten::_fake_quantize_learnable_per_channel_affine_backward(Tensor grad, Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max, float grad_factor=1.0) -> (Tensor, Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<_fake_quantize_learnable_per_channel_affine_backward::schema> create__fake_quantize_learnable_per_channel_affine_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_fake_quantize_learnable_per_channel_affine_backward::name, _fake_quantize_learnable_per_channel_affine_backward::overload_name)
      .typed<_fake_quantize_learnable_per_channel_affine_backward::schema>();
}

// aten::_fake_quantize_learnable_per_channel_affine_backward(Tensor grad, Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max, float grad_factor=1.0) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _fake_quantize_learnable_per_channel_affine_backward::call(const at::Tensor & grad, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max, double grad_factor) {
    static auto op = create__fake_quantize_learnable_per_channel_affine_backward_typed_handle();
    return op.call(grad, self, scale, zero_point, axis, quant_min, quant_max, grad_factor);
}

// aten::_fake_quantize_learnable_per_channel_affine_backward(Tensor grad, Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max, float grad_factor=1.0) -> (Tensor, Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _fake_quantize_learnable_per_channel_affine_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max, double grad_factor) {
    static auto op = create__fake_quantize_learnable_per_channel_affine_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad, self, scale, zero_point, axis, quant_min, quant_max, grad_factor);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fused_moving_avg_obs_fake_quant, name, "aten::fused_moving_avg_obs_fake_quant")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fused_moving_avg_obs_fake_quant, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fused_moving_avg_obs_fake_quant, schema_str, "fused_moving_avg_obs_fake_quant(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor(a!) running_min, Tensor(b!) running_max, Tensor(c!) scale, Tensor(d!) zero_point, float averaging_const, int quant_min, int quant_max, int ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False) -> Tensor")

// aten::fused_moving_avg_obs_fake_quant(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor(a!) running_min, Tensor(b!) running_max, Tensor(c!) scale, Tensor(d!) zero_point, float averaging_const, int quant_min, int quant_max, int ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fused_moving_avg_obs_fake_quant::schema> create_fused_moving_avg_obs_fake_quant_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fused_moving_avg_obs_fake_quant::name, fused_moving_avg_obs_fake_quant::overload_name)
      .typed<fused_moving_avg_obs_fake_quant::schema>();
}

// aten::fused_moving_avg_obs_fake_quant(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor(a!) running_min, Tensor(b!) running_max, Tensor(c!) scale, Tensor(d!) zero_point, float averaging_const, int quant_min, int quant_max, int ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False) -> Tensor
at::Tensor fused_moving_avg_obs_fake_quant::call(const at::Tensor & self, const at::Tensor & observer_on, const at::Tensor & fake_quant_on, at::Tensor & running_min, at::Tensor & running_max, at::Tensor & scale, at::Tensor & zero_point, double averaging_const, int64_t quant_min, int64_t quant_max, int64_t ch_axis, bool per_row_fake_quant, bool symmetric_quant) {
    static auto op = create_fused_moving_avg_obs_fake_quant_typed_handle();
    return op.call(self, observer_on, fake_quant_on, running_min, running_max, scale, zero_point, averaging_const, quant_min, quant_max, ch_axis, per_row_fake_quant, symmetric_quant);
}

// aten::fused_moving_avg_obs_fake_quant(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor(a!) running_min, Tensor(b!) running_max, Tensor(c!) scale, Tensor(d!) zero_point, float averaging_const, int quant_min, int quant_max, int ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False) -> Tensor
at::Tensor fused_moving_avg_obs_fake_quant::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & observer_on, const at::Tensor & fake_quant_on, at::Tensor & running_min, at::Tensor & running_max, at::Tensor & scale, at::Tensor & zero_point, double averaging_const, int64_t quant_min, int64_t quant_max, int64_t ch_axis, bool per_row_fake_quant, bool symmetric_quant) {
    static auto op = create_fused_moving_avg_obs_fake_quant_typed_handle();
    return op.redispatch(dispatchKeySet, self, observer_on, fake_quant_on, running_min, running_max, scale, zero_point, averaging_const, quant_min, quant_max, ch_axis, per_row_fake_quant, symmetric_quant);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_choose_qparams_per_tensor, name, "aten::_choose_qparams_per_tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_choose_qparams_per_tensor, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_choose_qparams_per_tensor, schema_str, "_choose_qparams_per_tensor(Tensor self, bool reduce_range=False) -> (float, int)")

// aten::_choose_qparams_per_tensor(Tensor self, bool reduce_range=False) -> (float, int)
static C10_NOINLINE c10::TypedOperatorHandle<_choose_qparams_per_tensor::schema> create__choose_qparams_per_tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_choose_qparams_per_tensor::name, _choose_qparams_per_tensor::overload_name)
      .typed<_choose_qparams_per_tensor::schema>();
}

// aten::_choose_qparams_per_tensor(Tensor self, bool reduce_range=False) -> (float, int)
::std::tuple<double,int64_t> _choose_qparams_per_tensor::call(const at::Tensor & self, bool reduce_range) {
    static auto op = create__choose_qparams_per_tensor_typed_handle();
    return op.call(self, reduce_range);
}

// aten::_choose_qparams_per_tensor(Tensor self, bool reduce_range=False) -> (float, int)
::std::tuple<double,int64_t> _choose_qparams_per_tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool reduce_range) {
    static auto op = create__choose_qparams_per_tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, reduce_range);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(to_dtype_layout, name, "aten::to")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(to_dtype_layout, overload_name, "dtype_layout")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(to_dtype_layout, schema_str, "to.dtype_layout(Tensor(a) self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)")

// aten::to.dtype_layout(Tensor(a) self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<to_dtype_layout::schema> create_to_dtype_layout_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(to_dtype_layout::name, to_dtype_layout::overload_name)
      .typed<to_dtype_layout::schema>();
}

// aten::to.dtype_layout(Tensor(a) self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)
at::Tensor to_dtype_layout::call(const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, bool non_blocking, bool copy, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create_to_dtype_layout_typed_handle();
    return op.call(self, dtype, layout, device, pin_memory, non_blocking, copy, memory_format);
}

// aten::to.dtype_layout(Tensor(a) self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)
at::Tensor to_dtype_layout::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, bool non_blocking, bool copy, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create_to_dtype_layout_typed_handle();
    return op.redispatch(dispatchKeySet, self, dtype, layout, device, pin_memory, non_blocking, copy, memory_format);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(to_other, name, "aten::to")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(to_other, overload_name, "other")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(to_other, schema_str, "to.other(Tensor(a) self, Tensor other, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)")

// aten::to.other(Tensor(a) self, Tensor other, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<to_other::schema> create_to_other_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(to_other::name, to_other::overload_name)
      .typed<to_other::schema>();
}

// aten::to.other(Tensor(a) self, Tensor other, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)
at::Tensor to_other::call(const at::Tensor & self, const at::Tensor & other, bool non_blocking, bool copy, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create_to_other_typed_handle();
    return op.call(self, other, non_blocking, copy, memory_format);
}

// aten::to.other(Tensor(a) self, Tensor other, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)
at::Tensor to_other::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, bool non_blocking, bool copy, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create_to_other_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, non_blocking, copy, memory_format);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(meshgrid, name, "aten::meshgrid")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(meshgrid, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(meshgrid, schema_str, "meshgrid(Tensor[] tensors) -> Tensor[]")

// aten::meshgrid(Tensor[] tensors) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<meshgrid::schema> create_meshgrid_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(meshgrid::name, meshgrid::overload_name)
      .typed<meshgrid::schema>();
}

// aten::meshgrid(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> meshgrid::call(at::TensorList tensors) {
    static auto op = create_meshgrid_typed_handle();
    return op.call(tensors);
}

// aten::meshgrid(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> meshgrid::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create_meshgrid_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(meshgrid_indexing, name, "aten::meshgrid")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(meshgrid_indexing, overload_name, "indexing")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(meshgrid_indexing, schema_str, "meshgrid.indexing(Tensor[] tensors, *, str indexing) -> Tensor[]")

// aten::meshgrid.indexing(Tensor[] tensors, *, str indexing) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<meshgrid_indexing::schema> create_meshgrid_indexing_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(meshgrid_indexing::name, meshgrid_indexing::overload_name)
      .typed<meshgrid_indexing::schema>();
}

// aten::meshgrid.indexing(Tensor[] tensors, *, str indexing) -> Tensor[]
::std::vector<at::Tensor> meshgrid_indexing::call(at::TensorList tensors, c10::string_view indexing) {
    static auto op = create_meshgrid_indexing_typed_handle();
    return op.call(tensors, indexing);
}

// aten::meshgrid.indexing(Tensor[] tensors, *, str indexing) -> Tensor[]
::std::vector<at::Tensor> meshgrid_indexing::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, c10::string_view indexing) {
    static auto op = create_meshgrid_indexing_typed_handle();
    return op.redispatch(dispatchKeySet, tensors, indexing);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(can_cast, name, "aten::can_cast")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(can_cast, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(can_cast, schema_str, "can_cast(ScalarType from, ScalarType to) -> bool")

// aten::can_cast(ScalarType from, ScalarType to) -> bool
static C10_NOINLINE c10::TypedOperatorHandle<can_cast::schema> create_can_cast_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(can_cast::name, can_cast::overload_name)
      .typed<can_cast::schema>();
}

// aten::can_cast(ScalarType from, ScalarType to) -> bool
bool can_cast::call(at::ScalarType from, at::ScalarType to) {
    static auto op = create_can_cast_typed_handle();
    return op.call(from, to);
}

// aten::can_cast(ScalarType from, ScalarType to) -> bool
bool can_cast::redispatch(c10::DispatchKeySet dispatchKeySet, at::ScalarType from, at::ScalarType to) {
    static auto op = create_can_cast_typed_handle();
    return op.redispatch(dispatchKeySet, from, to);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_thnn_fused_gru_cell, name, "aten::_thnn_fused_gru_cell")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_thnn_fused_gru_cell, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_thnn_fused_gru_cell, schema_str, "_thnn_fused_gru_cell(Tensor input_gates, Tensor hidden_gates, Tensor hx, Tensor? input_bias=None, Tensor? hidden_bias=None) -> (Tensor, Tensor)")

// aten::_thnn_fused_gru_cell(Tensor input_gates, Tensor hidden_gates, Tensor hx, Tensor? input_bias=None, Tensor? hidden_bias=None) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<_thnn_fused_gru_cell::schema> create__thnn_fused_gru_cell_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_thnn_fused_gru_cell::name, _thnn_fused_gru_cell::overload_name)
      .typed<_thnn_fused_gru_cell::schema>();
}

// aten::_thnn_fused_gru_cell(Tensor input_gates, Tensor hidden_gates, Tensor hx, Tensor? input_bias=None, Tensor? hidden_bias=None) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> _thnn_fused_gru_cell::call(const at::Tensor & input_gates, const at::Tensor & hidden_gates, const at::Tensor & hx, const c10::optional<at::Tensor> & input_bias, const c10::optional<at::Tensor> & hidden_bias) {
    static auto op = create__thnn_fused_gru_cell_typed_handle();
    return op.call(input_gates, hidden_gates, hx, input_bias, hidden_bias);
}

// aten::_thnn_fused_gru_cell(Tensor input_gates, Tensor hidden_gates, Tensor hx, Tensor? input_bias=None, Tensor? hidden_bias=None) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> _thnn_fused_gru_cell::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input_gates, const at::Tensor & hidden_gates, const at::Tensor & hx, const c10::optional<at::Tensor> & input_bias, const c10::optional<at::Tensor> & hidden_bias) {
    static auto op = create__thnn_fused_gru_cell_typed_handle();
    return op.redispatch(dispatchKeySet, input_gates, hidden_gates, hx, input_bias, hidden_bias);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gru_input, name, "aten::gru")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gru_input, overload_name, "input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gru_input, schema_str, "gru.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)")

// aten::gru.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<gru_input::schema> create_gru_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(gru_input::name, gru_input::overload_name)
      .typed<gru_input::schema>();
}

// aten::gru.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> gru_input::call(const at::Tensor & input, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
    static auto op = create_gru_input_typed_handle();
    return op.call(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}

// aten::gru.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> gru_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
    static auto op = create_gru_input_typed_handle();
    return op.redispatch(dispatchKeySet, input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantized_rnn_tanh_cell, name, "aten::quantized_rnn_tanh_cell")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantized_rnn_tanh_cell, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantized_rnn_tanh_cell, schema_str, "quantized_rnn_tanh_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor")

// aten::quantized_rnn_tanh_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<quantized_rnn_tanh_cell::schema> create_quantized_rnn_tanh_cell_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(quantized_rnn_tanh_cell::name, quantized_rnn_tanh_cell::overload_name)
      .typed<quantized_rnn_tanh_cell::schema>();
}

// aten::quantized_rnn_tanh_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor
at::Tensor quantized_rnn_tanh_cell::call(const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const at::Tensor & b_ih, const at::Tensor & b_hh, const at::Tensor & packed_ih, const at::Tensor & packed_hh, const at::Tensor & col_offsets_ih, const at::Tensor & col_offsets_hh, const at::Scalar & scale_ih, const at::Scalar & scale_hh, const at::Scalar & zero_point_ih, const at::Scalar & zero_point_hh) {
    static auto op = create_quantized_rnn_tanh_cell_typed_handle();
    return op.call(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
}

// aten::quantized_rnn_tanh_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor
at::Tensor quantized_rnn_tanh_cell::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const at::Tensor & b_ih, const at::Tensor & b_hh, const at::Tensor & packed_ih, const at::Tensor & packed_hh, const at::Tensor & col_offsets_ih, const at::Tensor & col_offsets_hh, const at::Scalar & scale_ih, const at::Scalar & scale_hh, const at::Scalar & zero_point_ih, const at::Scalar & zero_point_hh) {
    static auto op = create_quantized_rnn_tanh_cell_typed_handle();
    return op.redispatch(dispatchKeySet, input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_pack_padded_sequence, name, "aten::_pack_padded_sequence")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_pack_padded_sequence, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_pack_padded_sequence, schema_str, "_pack_padded_sequence(Tensor input, Tensor lengths, bool batch_first) -> (Tensor, Tensor)")

// aten::_pack_padded_sequence(Tensor input, Tensor lengths, bool batch_first) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<_pack_padded_sequence::schema> create__pack_padded_sequence_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_pack_padded_sequence::name, _pack_padded_sequence::overload_name)
      .typed<_pack_padded_sequence::schema>();
}

// aten::_pack_padded_sequence(Tensor input, Tensor lengths, bool batch_first) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> _pack_padded_sequence::call(const at::Tensor & input, const at::Tensor & lengths, bool batch_first) {
    static auto op = create__pack_padded_sequence_typed_handle();
    return op.call(input, lengths, batch_first);
}

// aten::_pack_padded_sequence(Tensor input, Tensor lengths, bool batch_first) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> _pack_padded_sequence::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & lengths, bool batch_first) {
    static auto op = create__pack_padded_sequence_typed_handle();
    return op.redispatch(dispatchKeySet, input, lengths, batch_first);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_set_to, name, "aten::is_set_to")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_set_to, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(is_set_to, schema_str, "is_set_to(Tensor self, Tensor tensor) -> bool")

// aten::is_set_to(Tensor self, Tensor tensor) -> bool
static C10_NOINLINE c10::TypedOperatorHandle<is_set_to::schema> create_is_set_to_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(is_set_to::name, is_set_to::overload_name)
      .typed<is_set_to::schema>();
}

// aten::is_set_to(Tensor self, Tensor tensor) -> bool
bool is_set_to::call(const at::Tensor & self, const at::Tensor & tensor) {
    static auto op = create_is_set_to_typed_handle();
    return op.call(self, tensor);
}

// aten::is_set_to(Tensor self, Tensor tensor) -> bool
bool is_set_to::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & tensor) {
    static auto op = create_is_set_to_typed_handle();
    return op.redispatch(dispatchKeySet, self, tensor);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(view, name, "aten::view")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(view, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(view, schema_str, "view(Tensor(a) self, int[] size) -> Tensor(a)")

// aten::view(Tensor(a) self, int[] size) -> Tensor(a)
static C10_NOINLINE c10::TypedOperatorHandle<view::schema> create_view_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(view::name, view::overload_name)
      .typed<view::schema>();
}

// aten::view(Tensor(a) self, int[] size) -> Tensor(a)
at::Tensor view::call(const at::Tensor & self, at::IntArrayRef size) {
    static auto op = create_view_typed_handle();
    return op.call(self, size);
}

// aten::view(Tensor(a) self, int[] size) -> Tensor(a)
at::Tensor view::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef size) {
    static auto op = create_view_typed_handle();
    return op.redispatch(dispatchKeySet, self, size);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_add_dimname, name, "aten::index_add")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_add_dimname, overload_name, "dimname")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(index_add_dimname, schema_str, "index_add.dimname(Tensor self, Dimname dim, Tensor index, Tensor source, *, Scalar alpha=1) -> Tensor")

// aten::index_add.dimname(Tensor self, Dimname dim, Tensor index, Tensor source, *, Scalar alpha=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<index_add_dimname::schema> create_index_add_dimname_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(index_add_dimname::name, index_add_dimname::overload_name)
      .typed<index_add_dimname::schema>();
}

// aten::index_add.dimname(Tensor self, Dimname dim, Tensor index, Tensor source, *, Scalar alpha=1) -> Tensor
at::Tensor index_add_dimname::call(const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha) {
    static auto op = create_index_add_dimname_typed_handle();
    return op.call(self, dim, index, source, alpha);
}

// aten::index_add.dimname(Tensor self, Dimname dim, Tensor index, Tensor source, *, Scalar alpha=1) -> Tensor
at::Tensor index_add_dimname::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha) {
    static auto op = create_index_add_dimname_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, source, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_value_out, name, "aten::scatter")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_value_out, overload_name, "value_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_value_out, schema_str, "scatter.value_out(Tensor self, int dim, Tensor index, Scalar value, *, Tensor(a!) out) -> Tensor(a!)")

// aten::scatter.value_out(Tensor self, int dim, Tensor index, Scalar value, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<scatter_value_out::schema> create_scatter_value_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(scatter_value_out::name, scatter_value_out::overload_name)
      .typed<scatter_value_out::schema>();
}

// aten::scatter.value_out(Tensor self, int dim, Tensor index, Scalar value, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & scatter_value_out::call(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, at::Tensor & out) {
    static auto op = create_scatter_value_out_typed_handle();
    return op.call(self, dim, index, value, out);
}

// aten::scatter.value_out(Tensor self, int dim, Tensor index, Scalar value, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & scatter_value_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, at::Tensor & out) {
    static auto op = create_scatter_value_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, value, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_add_, name, "aten::scatter_add_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_add_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(scatter_add_, schema_str, "scatter_add_(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)")

// aten::scatter_add_(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<scatter_add_::schema> create_scatter_add__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(scatter_add_::name, scatter_add_::overload_name)
      .typed<scatter_add_::schema>();
}

// aten::scatter_add_(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)
at::Tensor & scatter_add_::call(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src) {
    static auto op = create_scatter_add__typed_handle();
    return op.call(self, dim, index, src);
}

// aten::scatter_add_(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)
at::Tensor & scatter_add_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src) {
    static auto op = create_scatter_add__typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, src);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_and_Tensor, name, "aten::bitwise_and")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_and_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_and_Tensor, schema_str, "bitwise_and.Tensor(Tensor self, Tensor other) -> Tensor")

// aten::bitwise_and.Tensor(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<bitwise_and_Tensor::schema> create_bitwise_and_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bitwise_and_Tensor::name, bitwise_and_Tensor::overload_name)
      .typed<bitwise_and_Tensor::schema>();
}

// aten::bitwise_and.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor bitwise_and_Tensor::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_bitwise_and_Tensor_typed_handle();
    return op.call(self, other);
}

// aten::bitwise_and.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor bitwise_and_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_bitwise_and_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__and___Scalar, name, "aten::__and__")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__and___Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__and___Scalar, schema_str, "__and__.Scalar(Tensor self, Scalar other) -> Tensor")

// aten::__and__.Scalar(Tensor self, Scalar other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<__and___Scalar::schema> create___and___Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(__and___Scalar::name, __and___Scalar::overload_name)
      .typed<__and___Scalar::schema>();
}

// aten::__and__.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor __and___Scalar::call(const at::Tensor & self, const at::Scalar & other) {
    static auto op = create___and___Scalar_typed_handle();
    return op.call(self, other);
}

// aten::__and__.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor __and___Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
    static auto op = create___and___Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__and___Tensor, name, "aten::__and__")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__and___Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__and___Tensor, schema_str, "__and__.Tensor(Tensor self, Tensor other) -> Tensor")

// aten::__and__.Tensor(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<__and___Tensor::schema> create___and___Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(__and___Tensor::name, __and___Tensor::overload_name)
      .typed<__and___Tensor::schema>();
}

// aten::__and__.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor __and___Tensor::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create___and___Tensor_typed_handle();
    return op.call(self, other);
}

// aten::__and__.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor __and___Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create___and___Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_or_Tensor_out, name, "aten::bitwise_or")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_or_Tensor_out, overload_name, "Tensor_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_or_Tensor_out, schema_str, "bitwise_or.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::bitwise_or.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<bitwise_or_Tensor_out::schema> create_bitwise_or_Tensor_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bitwise_or_Tensor_out::name, bitwise_or_Tensor_out::overload_name)
      .typed<bitwise_or_Tensor_out::schema>();
}

// aten::bitwise_or.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & bitwise_or_Tensor_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_bitwise_or_Tensor_out_typed_handle();
    return op.call(self, other, out);
}

// aten::bitwise_or.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & bitwise_or_Tensor_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_bitwise_or_Tensor_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__or___Tensor, name, "aten::__or__")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__or___Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__or___Tensor, schema_str, "__or__.Tensor(Tensor self, Tensor other) -> Tensor")

// aten::__or__.Tensor(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<__or___Tensor::schema> create___or___Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(__or___Tensor::name, __or___Tensor::overload_name)
      .typed<__or___Tensor::schema>();
}

// aten::__or__.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor __or___Tensor::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create___or___Tensor_typed_handle();
    return op.call(self, other);
}

// aten::__or__.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor __or___Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create___or___Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_xor_Tensor_out, name, "aten::bitwise_xor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_xor_Tensor_out, overload_name, "Tensor_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_xor_Tensor_out, schema_str, "bitwise_xor.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::bitwise_xor.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<bitwise_xor_Tensor_out::schema> create_bitwise_xor_Tensor_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bitwise_xor_Tensor_out::name, bitwise_xor_Tensor_out::overload_name)
      .typed<bitwise_xor_Tensor_out::schema>();
}

// aten::bitwise_xor.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & bitwise_xor_Tensor_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_bitwise_xor_Tensor_out_typed_handle();
    return op.call(self, other, out);
}

// aten::bitwise_xor.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & bitwise_xor_Tensor_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_bitwise_xor_Tensor_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__ixor___Scalar, name, "aten::__ixor__")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__ixor___Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__ixor___Scalar, schema_str, "__ixor__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)")

// aten::__ixor__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<__ixor___Scalar::schema> create___ixor___Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(__ixor___Scalar::name, __ixor___Scalar::overload_name)
      .typed<__ixor___Scalar::schema>();
}

// aten::__ixor__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & __ixor___Scalar::call(at::Tensor & self, const at::Scalar & other) {
    static auto op = create___ixor___Scalar_typed_handle();
    return op.call(self, other);
}

// aten::__ixor__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & __ixor___Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
    static auto op = create___ixor___Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__lshift___Scalar, name, "aten::__lshift__")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__lshift___Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__lshift___Scalar, schema_str, "__lshift__.Scalar(Tensor self, Scalar other) -> Tensor")

// aten::__lshift__.Scalar(Tensor self, Scalar other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<__lshift___Scalar::schema> create___lshift___Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(__lshift___Scalar::name, __lshift___Scalar::overload_name)
      .typed<__lshift___Scalar::schema>();
}

// aten::__lshift__.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor __lshift___Scalar::call(const at::Tensor & self, const at::Scalar & other) {
    static auto op = create___lshift___Scalar_typed_handle();
    return op.call(self, other);
}

// aten::__lshift__.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor __lshift___Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
    static auto op = create___lshift___Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__ilshift___Scalar, name, "aten::__ilshift__")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__ilshift___Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__ilshift___Scalar, schema_str, "__ilshift__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)")

// aten::__ilshift__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<__ilshift___Scalar::schema> create___ilshift___Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(__ilshift___Scalar::name, __ilshift___Scalar::overload_name)
      .typed<__ilshift___Scalar::schema>();
}

// aten::__ilshift__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & __ilshift___Scalar::call(at::Tensor & self, const at::Scalar & other) {
    static auto op = create___ilshift___Scalar_typed_handle();
    return op.call(self, other);
}

// aten::__ilshift__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & __ilshift___Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
    static auto op = create___ilshift___Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_left_shift_Tensor_Scalar, name, "aten::bitwise_left_shift")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_left_shift_Tensor_Scalar, overload_name, "Tensor_Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_left_shift_Tensor_Scalar, schema_str, "bitwise_left_shift.Tensor_Scalar(Tensor self, Scalar other) -> Tensor")

// aten::bitwise_left_shift.Tensor_Scalar(Tensor self, Scalar other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<bitwise_left_shift_Tensor_Scalar::schema> create_bitwise_left_shift_Tensor_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bitwise_left_shift_Tensor_Scalar::name, bitwise_left_shift_Tensor_Scalar::overload_name)
      .typed<bitwise_left_shift_Tensor_Scalar::schema>();
}

// aten::bitwise_left_shift.Tensor_Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor bitwise_left_shift_Tensor_Scalar::call(const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_bitwise_left_shift_Tensor_Scalar_typed_handle();
    return op.call(self, other);
}

// aten::bitwise_left_shift.Tensor_Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor bitwise_left_shift_Tensor_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_bitwise_left_shift_Tensor_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_left_shift_Tensor_Scalar_out, name, "aten::bitwise_left_shift")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_left_shift_Tensor_Scalar_out, overload_name, "Tensor_Scalar_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bitwise_left_shift_Tensor_Scalar_out, schema_str, "bitwise_left_shift.Tensor_Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::bitwise_left_shift.Tensor_Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<bitwise_left_shift_Tensor_Scalar_out::schema> create_bitwise_left_shift_Tensor_Scalar_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bitwise_left_shift_Tensor_Scalar_out::name, bitwise_left_shift_Tensor_Scalar_out::overload_name)
      .typed<bitwise_left_shift_Tensor_Scalar_out::schema>();
}

// aten::bitwise_left_shift.Tensor_Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & bitwise_left_shift_Tensor_Scalar_out::call(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_bitwise_left_shift_Tensor_Scalar_out_typed_handle();
    return op.call(self, other, out);
}

// aten::bitwise_left_shift.Tensor_Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & bitwise_left_shift_Tensor_Scalar_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_bitwise_left_shift_Tensor_Scalar_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__irshift___Tensor, name, "aten::__irshift__")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__irshift___Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(__irshift___Tensor, schema_str, "__irshift__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::__irshift__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<__irshift___Tensor::schema> create___irshift___Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(__irshift___Tensor::name, __irshift___Tensor::overload_name)
      .typed<__irshift___Tensor::schema>();
}

// aten::__irshift__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & __irshift___Tensor::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create___irshift___Tensor_typed_handle();
    return op.call(self, other);
}

// aten::__irshift__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & __irshift___Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create___irshift___Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(triu_, name, "aten::triu_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(triu_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(triu_, schema_str, "triu_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)")

// aten::triu_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<triu_::schema> create_triu__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(triu_::name, triu_::overload_name)
      .typed<triu_::schema>();
}

// aten::triu_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)
at::Tensor & triu_::call(at::Tensor & self, int64_t diagonal) {
    static auto op = create_triu__typed_handle();
    return op.call(self, diagonal);
}

// aten::triu_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)
at::Tensor & triu_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, int64_t diagonal) {
    static auto op = create_triu__typed_handle();
    return op.redispatch(dispatchKeySet, self, diagonal);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addbmm, name, "aten::addbmm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addbmm, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addbmm, schema_str, "addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor")

// aten::addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<addbmm::schema> create_addbmm_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(addbmm::name, addbmm::overload_name)
      .typed<addbmm::schema>();
}

// aten::addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
at::Tensor addbmm::call(const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha) {
    static auto op = create_addbmm_typed_handle();
    return op.call(self, batch1, batch2, beta, alpha);
}

// aten::addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
at::Tensor addbmm::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha) {
    static auto op = create_addbmm_typed_handle();
    return op.redispatch(dispatchKeySet, self, batch1, batch2, beta, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cauchy_, name, "aten::cauchy_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cauchy_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cauchy_, schema_str, "cauchy_(Tensor(a!) self, float median=0, float sigma=1, *, Generator? generator=None) -> Tensor(a!)")

// aten::cauchy_(Tensor(a!) self, float median=0, float sigma=1, *, Generator? generator=None) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<cauchy_::schema> create_cauchy__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cauchy_::name, cauchy_::overload_name)
      .typed<cauchy_::schema>();
}

// aten::cauchy_(Tensor(a!) self, float median=0, float sigma=1, *, Generator? generator=None) -> Tensor(a!)
at::Tensor & cauchy_::call(at::Tensor & self, double median, double sigma, c10::optional<at::Generator> generator) {
    static auto op = create_cauchy__typed_handle();
    return op.call(self, median, sigma, generator);
}

// aten::cauchy_(Tensor(a!) self, float median=0, float sigma=1, *, Generator? generator=None) -> Tensor(a!)
at::Tensor & cauchy_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, double median, double sigma, c10::optional<at::Generator> generator) {
    static auto op = create_cauchy__typed_handle();
    return op.redispatch(dispatchKeySet, self, median, sigma, generator);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(triu_out, name, "aten::triu")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(triu_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(triu_out, schema_str, "triu.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)")

// aten::triu.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<triu_out::schema> create_triu_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(triu_out::name, triu_out::overload_name)
      .typed<triu_out::schema>();
}

// aten::triu.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & triu_out::call(const at::Tensor & self, int64_t diagonal, at::Tensor & out) {
    static auto op = create_triu_out_typed_handle();
    return op.call(self, diagonal, out);
}

// aten::triu.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & triu_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t diagonal, at::Tensor & out) {
    static auto op = create_triu_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, diagonal, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(triu, name, "aten::triu")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(triu, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(triu, schema_str, "triu(Tensor self, int diagonal=0) -> Tensor")

// aten::triu(Tensor self, int diagonal=0) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<triu::schema> create_triu_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(triu::name, triu::overload_name)
      .typed<triu::schema>();
}

// aten::triu(Tensor self, int diagonal=0) -> Tensor
at::Tensor triu::call(const at::Tensor & self, int64_t diagonal) {
    static auto op = create_triu_typed_handle();
    return op.call(self, diagonal);
}

// aten::triu(Tensor self, int diagonal=0) -> Tensor
at::Tensor triu::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t diagonal) {
    static auto op = create_triu_typed_handle();
    return op.redispatch(dispatchKeySet, self, diagonal);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ne_Scalar_out, name, "aten::ne")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ne_Scalar_out, overload_name, "Scalar_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ne_Scalar_out, schema_str, "ne.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::ne.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<ne_Scalar_out::schema> create_ne_Scalar_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(ne_Scalar_out::name, ne_Scalar_out::overload_name)
      .typed<ne_Scalar_out::schema>();
}

// aten::ne.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & ne_Scalar_out::call(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_ne_Scalar_out_typed_handle();
    return op.call(self, other, out);
}

// aten::ne.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & ne_Scalar_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_ne_Scalar_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(not_equal_Scalar_out, name, "aten::not_equal")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(not_equal_Scalar_out, overload_name, "Scalar_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(not_equal_Scalar_out, schema_str, "not_equal.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::not_equal.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<not_equal_Scalar_out::schema> create_not_equal_Scalar_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(not_equal_Scalar_out::name, not_equal_Scalar_out::overload_name)
      .typed<not_equal_Scalar_out::schema>();
}

// aten::not_equal.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & not_equal_Scalar_out::call(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_not_equal_Scalar_out_typed_handle();
    return op.call(self, other, out);
}

// aten::not_equal.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & not_equal_Scalar_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_not_equal_Scalar_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(not_equal__Scalar, name, "aten::not_equal_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(not_equal__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(not_equal__Scalar, schema_str, "not_equal_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)")

// aten::not_equal_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<not_equal__Scalar::schema> create_not_equal__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(not_equal__Scalar::name, not_equal__Scalar::overload_name)
      .typed<not_equal__Scalar::schema>();
}

// aten::not_equal_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & not_equal__Scalar::call(at::Tensor & self, const at::Scalar & other) {
    static auto op = create_not_equal__Scalar_typed_handle();
    return op.call(self, other);
}

// aten::not_equal_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & not_equal__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
    static auto op = create_not_equal__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(eq_Tensor_out, name, "aten::eq")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(eq_Tensor_out, overload_name, "Tensor_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(eq_Tensor_out, schema_str, "eq.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::eq.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<eq_Tensor_out::schema> create_eq_Tensor_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(eq_Tensor_out::name, eq_Tensor_out::overload_name)
      .typed<eq_Tensor_out::schema>();
}

// aten::eq.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & eq_Tensor_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_eq_Tensor_out_typed_handle();
    return op.call(self, other, out);
}

// aten::eq.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & eq_Tensor_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_eq_Tensor_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(greater_equal__Tensor, name, "aten::greater_equal_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(greater_equal__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(greater_equal__Tensor, schema_str, "greater_equal_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::greater_equal_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<greater_equal__Tensor::schema> create_greater_equal__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(greater_equal__Tensor::name, greater_equal__Tensor::overload_name)
      .typed<greater_equal__Tensor::schema>();
}

// aten::greater_equal_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & greater_equal__Tensor::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create_greater_equal__Tensor_typed_handle();
    return op.call(self, other);
}

// aten::greater_equal_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & greater_equal__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create_greater_equal__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(le__Tensor, name, "aten::le_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(le__Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(le__Tensor, schema_str, "le_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::le_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<le__Tensor::schema> create_le__Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(le__Tensor::name, le__Tensor::overload_name)
      .typed<le__Tensor::schema>();
}

// aten::le_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & le__Tensor::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create_le__Tensor_typed_handle();
    return op.call(self, other);
}

// aten::le_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & le__Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create_le__Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(less_equal__Scalar, name, "aten::less_equal_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(less_equal__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(less_equal__Scalar, schema_str, "less_equal_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)")

// aten::less_equal_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<less_equal__Scalar::schema> create_less_equal__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(less_equal__Scalar::name, less_equal__Scalar::overload_name)
      .typed<less_equal__Scalar::schema>();
}

// aten::less_equal_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & less_equal__Scalar::call(at::Tensor & self, const at::Scalar & other) {
    static auto op = create_less_equal__Scalar_typed_handle();
    return op.call(self, other);
}

// aten::less_equal_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & less_equal__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
    static auto op = create_less_equal__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gt_Scalar_out, name, "aten::gt")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gt_Scalar_out, overload_name, "Scalar_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gt_Scalar_out, schema_str, "gt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::gt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<gt_Scalar_out::schema> create_gt_Scalar_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(gt_Scalar_out::name, gt_Scalar_out::overload_name)
      .typed<gt_Scalar_out::schema>();
}

// aten::gt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & gt_Scalar_out::call(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_gt_Scalar_out_typed_handle();
    return op.call(self, other, out);
}

// aten::gt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & gt_Scalar_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_gt_Scalar_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gt_Tensor, name, "aten::gt")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gt_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gt_Tensor, schema_str, "gt.Tensor(Tensor self, Tensor other) -> Tensor")

// aten::gt.Tensor(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<gt_Tensor::schema> create_gt_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(gt_Tensor::name, gt_Tensor::overload_name)
      .typed<gt_Tensor::schema>();
}

// aten::gt.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor gt_Tensor::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_gt_Tensor_typed_handle();
    return op.call(self, other);
}

// aten::gt.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor gt_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_gt_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(greater_Scalar, name, "aten::greater")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(greater_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(greater_Scalar, schema_str, "greater.Scalar(Tensor self, Scalar other) -> Tensor")

// aten::greater.Scalar(Tensor self, Scalar other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<greater_Scalar::schema> create_greater_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(greater_Scalar::name, greater_Scalar::overload_name)
      .typed<greater_Scalar::schema>();
}

// aten::greater.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor greater_Scalar::call(const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_greater_Scalar_typed_handle();
    return op.call(self, other);
}

// aten::greater.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor greater_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_greater_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lt_Scalar_out, name, "aten::lt")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lt_Scalar_out, overload_name, "Scalar_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lt_Scalar_out, schema_str, "lt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::lt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<lt_Scalar_out::schema> create_lt_Scalar_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(lt_Scalar_out::name, lt_Scalar_out::overload_name)
      .typed<lt_Scalar_out::schema>();
}

// aten::lt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & lt_Scalar_out::call(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_lt_Scalar_out_typed_handle();
    return op.call(self, other, out);
}

// aten::lt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & lt_Scalar_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_lt_Scalar_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lt_Tensor, name, "aten::lt")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lt_Tensor, overload_name, "Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lt_Tensor, schema_str, "lt.Tensor(Tensor self, Tensor other) -> Tensor")

// aten::lt.Tensor(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<lt_Tensor::schema> create_lt_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(lt_Tensor::name, lt_Tensor::overload_name)
      .typed<lt_Tensor::schema>();
}

// aten::lt.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor lt_Tensor::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_lt_Tensor_typed_handle();
    return op.call(self, other);
}

// aten::lt.Tensor(Tensor self, Tensor other) -> Tensor
at::Tensor lt_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_lt_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lt__Scalar, name, "aten::lt_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lt__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lt__Scalar, schema_str, "lt_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)")

// aten::lt_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<lt__Scalar::schema> create_lt__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(lt__Scalar::name, lt__Scalar::overload_name)
      .typed<lt__Scalar::schema>();
}

// aten::lt_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & lt__Scalar::call(at::Tensor & self, const at::Scalar & other) {
    static auto op = create_lt__Scalar_typed_handle();
    return op.call(self, other);
}

// aten::lt_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & lt__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
    static auto op = create_lt__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(less__Scalar, name, "aten::less_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(less__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(less__Scalar, schema_str, "less_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)")

// aten::less_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<less__Scalar::schema> create_less__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(less__Scalar::name, less__Scalar::overload_name)
      .typed<less__Scalar::schema>();
}

// aten::less_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & less__Scalar::call(at::Tensor & self, const at::Scalar & other) {
    static auto op = create_less__Scalar_typed_handle();
    return op.call(self, other);
}

// aten::less_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & less__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
    static auto op = create_less__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nonzero_out, name, "aten::nonzero")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nonzero_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nonzero_out, schema_str, "nonzero.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::nonzero.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<nonzero_out::schema> create_nonzero_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nonzero_out::name, nonzero_out::overload_name)
      .typed<nonzero_out::schema>();
}

// aten::nonzero.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & nonzero_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_nonzero_out_typed_handle();
    return op.call(self, out);
}

// aten::nonzero.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & nonzero_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_nonzero_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gather_out, name, "aten::gather")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gather_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gather_out, schema_str, "gather.out(Tensor self, int dim, Tensor index, *, bool sparse_grad=False, Tensor(a!) out) -> Tensor(a!)")

// aten::gather.out(Tensor self, int dim, Tensor index, *, bool sparse_grad=False, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<gather_out::schema> create_gather_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(gather_out::name, gather_out::overload_name)
      .typed<gather_out::schema>();
}

// aten::gather.out(Tensor self, int dim, Tensor index, *, bool sparse_grad=False, Tensor(a!) out) -> Tensor(a!)
at::Tensor & gather_out::call(const at::Tensor & self, int64_t dim, const at::Tensor & index, bool sparse_grad, at::Tensor & out) {
    static auto op = create_gather_out_typed_handle();
    return op.call(self, dim, index, sparse_grad, out);
}

// aten::gather.out(Tensor self, int dim, Tensor index, *, bool sparse_grad=False, Tensor(a!) out) -> Tensor(a!)
at::Tensor & gather_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, const at::Tensor & index, bool sparse_grad, at::Tensor & out) {
    static auto op = create_gather_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, sparse_grad, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gather, name, "aten::gather")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gather, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gather, schema_str, "gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor")

// aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<gather::schema> create_gather_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(gather::name, gather::overload_name)
      .typed<gather::schema>();
}

// aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor
at::Tensor gather::call(const at::Tensor & self, int64_t dim, const at::Tensor & index, bool sparse_grad) {
    static auto op = create_gather_typed_handle();
    return op.call(self, dim, index, sparse_grad);
}

// aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor
at::Tensor gather::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t dim, const at::Tensor & index, bool sparse_grad) {
    static auto op = create_gather_typed_handle();
    return op.redispatch(dispatchKeySet, self, dim, index, sparse_grad);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gather_backward, name, "aten::gather_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gather_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(gather_backward, schema_str, "gather_backward(Tensor grad, Tensor self, int dim, Tensor index, bool sparse_grad) -> Tensor")

// aten::gather_backward(Tensor grad, Tensor self, int dim, Tensor index, bool sparse_grad) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<gather_backward::schema> create_gather_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(gather_backward::name, gather_backward::overload_name)
      .typed<gather_backward::schema>();
}

// aten::gather_backward(Tensor grad, Tensor self, int dim, Tensor index, bool sparse_grad) -> Tensor
at::Tensor gather_backward::call(const at::Tensor & grad, const at::Tensor & self, int64_t dim, const at::Tensor & index, bool sparse_grad) {
    static auto op = create_gather_backward_typed_handle();
    return op.call(grad, self, dim, index, sparse_grad);
}

// aten::gather_backward(Tensor grad, Tensor self, int dim, Tensor index, bool sparse_grad) -> Tensor
at::Tensor gather_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad, const at::Tensor & self, int64_t dim, const at::Tensor & index, bool sparse_grad) {
    static auto op = create_gather_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad, self, dim, index, sparse_grad);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addcmul_out, name, "aten::addcmul")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addcmul_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(addcmul_out, schema_str, "addcmul.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)")

// aten::addcmul.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<addcmul_out::schema> create_addcmul_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(addcmul_out::name, addcmul_out::overload_name)
      .typed<addcmul_out::schema>();
}

// aten::addcmul.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)
at::Tensor & addcmul_out::call(const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value, at::Tensor & out) {
    static auto op = create_addcmul_out_typed_handle();
    return op.call(self, tensor1, tensor2, value, out);
}

// aten::addcmul.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)
at::Tensor & addcmul_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value, at::Tensor & out) {
    static auto op = create_addcmul_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, tensor1, tensor2, value, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cross_entropy_loss, name, "aten::cross_entropy_loss")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cross_entropy_loss, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(cross_entropy_loss, schema_str, "cross_entropy_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100, float label_smoothing=0.0) -> Tensor")

// aten::cross_entropy_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100, float label_smoothing=0.0) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<cross_entropy_loss::schema> create_cross_entropy_loss_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(cross_entropy_loss::name, cross_entropy_loss::overload_name)
      .typed<cross_entropy_loss::schema>();
}

// aten::cross_entropy_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100, float label_smoothing=0.0) -> Tensor
at::Tensor cross_entropy_loss::call(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, double label_smoothing) {
    static auto op = create_cross_entropy_loss_typed_handle();
    return op.call(self, target, weight, reduction, ignore_index, label_smoothing);
}

// aten::cross_entropy_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100, float label_smoothing=0.0) -> Tensor
at::Tensor cross_entropy_loss::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, double label_smoothing) {
    static auto op = create_cross_entropy_loss_typed_handle();
    return op.redispatch(dispatchKeySet, self, target, weight, reduction, ignore_index, label_smoothing);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(triangular_solve, name, "aten::triangular_solve")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(triangular_solve, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(triangular_solve, schema_str, "triangular_solve(Tensor self, Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False) -> (Tensor solution, Tensor cloned_coefficient)")

// aten::triangular_solve(Tensor self, Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False) -> (Tensor solution, Tensor cloned_coefficient)
static C10_NOINLINE c10::TypedOperatorHandle<triangular_solve::schema> create_triangular_solve_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(triangular_solve::name, triangular_solve::overload_name)
      .typed<triangular_solve::schema>();
}

// aten::triangular_solve(Tensor self, Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False) -> (Tensor solution, Tensor cloned_coefficient)
::std::tuple<at::Tensor,at::Tensor> triangular_solve::call(const at::Tensor & self, const at::Tensor & A, bool upper, bool transpose, bool unitriangular) {
    static auto op = create_triangular_solve_typed_handle();
    return op.call(self, A, upper, transpose, unitriangular);
}

// aten::triangular_solve(Tensor self, Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False) -> (Tensor solution, Tensor cloned_coefficient)
::std::tuple<at::Tensor,at::Tensor> triangular_solve::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & A, bool upper, bool transpose, bool unitriangular) {
    static auto op = create_triangular_solve_typed_handle();
    return op.redispatch(dispatchKeySet, self, A, upper, transpose, unitriangular);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_svd_helper, name, "aten::_svd_helper")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_svd_helper, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_svd_helper, schema_str, "_svd_helper(Tensor self, bool some, bool compute_uv) -> (Tensor U, Tensor S, Tensor V)")

// aten::_svd_helper(Tensor self, bool some, bool compute_uv) -> (Tensor U, Tensor S, Tensor V)
static C10_NOINLINE c10::TypedOperatorHandle<_svd_helper::schema> create__svd_helper_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_svd_helper::name, _svd_helper::overload_name)
      .typed<_svd_helper::schema>();
}

// aten::_svd_helper(Tensor self, bool some, bool compute_uv) -> (Tensor U, Tensor S, Tensor V)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _svd_helper::call(const at::Tensor & self, bool some, bool compute_uv) {
    static auto op = create__svd_helper_typed_handle();
    return op.call(self, some, compute_uv);
}

// aten::_svd_helper(Tensor self, bool some, bool compute_uv) -> (Tensor U, Tensor S, Tensor V)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _svd_helper::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool some, bool compute_uv) {
    static auto op = create__svd_helper_typed_handle();
    return op.redispatch(dispatchKeySet, self, some, compute_uv);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(solve, name, "aten::solve")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(solve, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(solve, schema_str, "solve(Tensor self, Tensor A) -> (Tensor solution, Tensor LU)")

// aten::solve(Tensor self, Tensor A) -> (Tensor solution, Tensor LU)
static C10_NOINLINE c10::TypedOperatorHandle<solve::schema> create_solve_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(solve::name, solve::overload_name)
      .typed<solve::schema>();
}

// aten::solve(Tensor self, Tensor A) -> (Tensor solution, Tensor LU)
::std::tuple<at::Tensor,at::Tensor> solve::call(const at::Tensor & self, const at::Tensor & A) {
    static auto op = create_solve_typed_handle();
    return op.call(self, A);
}

// aten::solve(Tensor self, Tensor A) -> (Tensor solution, Tensor LU)
::std::tuple<at::Tensor,at::Tensor> solve::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & A) {
    static auto op = create_solve_typed_handle();
    return op.redispatch(dispatchKeySet, self, A);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ormqr_out, name, "aten::ormqr")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ormqr_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ormqr_out, schema_str, "ormqr.out(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False, *, Tensor(a!) out) -> Tensor(a!)")

// aten::ormqr.out(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<ormqr_out::schema> create_ormqr_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(ormqr_out::name, ormqr_out::overload_name)
      .typed<ormqr_out::schema>();
}

// aten::ormqr.out(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & ormqr_out::call(const at::Tensor & self, const at::Tensor & input2, const at::Tensor & input3, bool left, bool transpose, at::Tensor & out) {
    static auto op = create_ormqr_out_typed_handle();
    return op.call(self, input2, input3, left, transpose, out);
}

// aten::ormqr.out(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & ormqr_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & input2, const at::Tensor & input3, bool left, bool transpose, at::Tensor & out) {
    static auto op = create_ormqr_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, input2, input3, left, transpose, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ormqr, name, "aten::ormqr")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ormqr, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ormqr, schema_str, "ormqr(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False) -> Tensor")

// aten::ormqr(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<ormqr::schema> create_ormqr_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(ormqr::name, ormqr::overload_name)
      .typed<ormqr::schema>();
}

// aten::ormqr(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False) -> Tensor
at::Tensor ormqr::call(const at::Tensor & self, const at::Tensor & input2, const at::Tensor & input3, bool left, bool transpose) {
    static auto op = create_ormqr_typed_handle();
    return op.call(self, input2, input3, left, transpose);
}

// aten::ormqr(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False) -> Tensor
at::Tensor ormqr::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & input2, const at::Tensor & input3, bool left, bool transpose) {
    static auto op = create_ormqr_typed_handle();
    return op.redispatch(dispatchKeySet, self, input2, input3, left, transpose);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lu_solve_out, name, "aten::lu_solve")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lu_solve_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lu_solve_out, schema_str, "lu_solve.out(Tensor self, Tensor LU_data, Tensor LU_pivots, *, Tensor(a!) out) -> Tensor(a!)")

// aten::lu_solve.out(Tensor self, Tensor LU_data, Tensor LU_pivots, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<lu_solve_out::schema> create_lu_solve_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(lu_solve_out::name, lu_solve_out::overload_name)
      .typed<lu_solve_out::schema>();
}

// aten::lu_solve.out(Tensor self, Tensor LU_data, Tensor LU_pivots, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & lu_solve_out::call(const at::Tensor & self, const at::Tensor & LU_data, const at::Tensor & LU_pivots, at::Tensor & out) {
    static auto op = create_lu_solve_out_typed_handle();
    return op.call(self, LU_data, LU_pivots, out);
}

// aten::lu_solve.out(Tensor self, Tensor LU_data, Tensor LU_pivots, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & lu_solve_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & LU_data, const at::Tensor & LU_pivots, at::Tensor & out) {
    static auto op = create_lu_solve_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, LU_data, LU_pivots, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(i0, name, "aten::i0")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(i0, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(i0, schema_str, "i0(Tensor self) -> Tensor")

// aten::i0(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<i0::schema> create_i0_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(i0::name, i0::overload_name)
      .typed<i0::schema>();
}

// aten::i0(Tensor self) -> Tensor
at::Tensor i0::call(const at::Tensor & self) {
    static auto op = create_i0_typed_handle();
    return op.call(self);
}

// aten::i0(Tensor self) -> Tensor
at::Tensor i0::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_i0_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sign, name, "aten::sign")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sign, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sign, schema_str, "sign(Tensor self) -> Tensor")

// aten::sign(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<sign::schema> create_sign_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sign::name, sign::overload_name)
      .typed<sign::schema>();
}

// aten::sign(Tensor self) -> Tensor
at::Tensor sign::call(const at::Tensor & self) {
    static auto op = create_sign_typed_handle();
    return op.call(self);
}

// aten::sign(Tensor self) -> Tensor
at::Tensor sign::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_sign_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lerp_Tensor_out, name, "aten::lerp")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lerp_Tensor_out, overload_name, "Tensor_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(lerp_Tensor_out, schema_str, "lerp.Tensor_out(Tensor self, Tensor end, Tensor weight, *, Tensor(a!) out) -> Tensor(a!)")

// aten::lerp.Tensor_out(Tensor self, Tensor end, Tensor weight, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<lerp_Tensor_out::schema> create_lerp_Tensor_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(lerp_Tensor_out::name, lerp_Tensor_out::overload_name)
      .typed<lerp_Tensor_out::schema>();
}

// aten::lerp.Tensor_out(Tensor self, Tensor end, Tensor weight, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & lerp_Tensor_out::call(const at::Tensor & self, const at::Tensor & end, const at::Tensor & weight, at::Tensor & out) {
    static auto op = create_lerp_Tensor_out_typed_handle();
    return op.call(self, end, weight, out);
}

// aten::lerp.Tensor_out(Tensor self, Tensor end, Tensor weight, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & lerp_Tensor_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & end, const at::Tensor & weight, at::Tensor & out) {
    static auto op = create_lerp_Tensor_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, end, weight, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(histogram_bin_ct, name, "aten::histogram")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(histogram_bin_ct, overload_name, "bin_ct")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(histogram_bin_ct, schema_str, "histogram.bin_ct(Tensor self, int bins=100, *, float[]? range=None, Tensor? weight=None, bool density=False) -> (Tensor hist, Tensor bin_edges)")

// aten::histogram.bin_ct(Tensor self, int bins=100, *, float[]? range=None, Tensor? weight=None, bool density=False) -> (Tensor hist, Tensor bin_edges)
static C10_NOINLINE c10::TypedOperatorHandle<histogram_bin_ct::schema> create_histogram_bin_ct_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(histogram_bin_ct::name, histogram_bin_ct::overload_name)
      .typed<histogram_bin_ct::schema>();
}

// aten::histogram.bin_ct(Tensor self, int bins=100, *, float[]? range=None, Tensor? weight=None, bool density=False) -> (Tensor hist, Tensor bin_edges)
::std::tuple<at::Tensor,at::Tensor> histogram_bin_ct::call(const at::Tensor & self, int64_t bins, c10::optional<at::ArrayRef<double>> range, const c10::optional<at::Tensor> & weight, bool density) {
    static auto op = create_histogram_bin_ct_typed_handle();
    return op.call(self, bins, range, weight, density);
}

// aten::histogram.bin_ct(Tensor self, int bins=100, *, float[]? range=None, Tensor? weight=None, bool density=False) -> (Tensor hist, Tensor bin_edges)
::std::tuple<at::Tensor,at::Tensor> histogram_bin_ct::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t bins, c10::optional<at::ArrayRef<double>> range, const c10::optional<at::Tensor> & weight, bool density) {
    static auto op = create_histogram_bin_ct_typed_handle();
    return op.redispatch(dispatchKeySet, self, bins, range, weight, density);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hypot_out, name, "aten::hypot")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hypot_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hypot_out, schema_str, "hypot.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::hypot.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<hypot_out::schema> create_hypot_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(hypot_out::name, hypot_out::overload_name)
      .typed<hypot_out::schema>();
}

// aten::hypot.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & hypot_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_hypot_out_typed_handle();
    return op.call(self, other, out);
}

// aten::hypot.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & hypot_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_hypot_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hypot_, name, "aten::hypot_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hypot_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hypot_, schema_str, "hypot_(Tensor(a!) self, Tensor other) -> Tensor(a!)")

// aten::hypot_(Tensor(a!) self, Tensor other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<hypot_::schema> create_hypot__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(hypot_::name, hypot_::overload_name)
      .typed<hypot_::schema>();
}

// aten::hypot_(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & hypot_::call(at::Tensor & self, const at::Tensor & other) {
    static auto op = create_hypot__typed_handle();
    return op.call(self, other);
}

// aten::hypot_(Tensor(a!) self, Tensor other) -> Tensor(a!)
at::Tensor & hypot_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Tensor & other) {
    static auto op = create_hypot__typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(remainder_Scalar, name, "aten::remainder")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(remainder_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(remainder_Scalar, schema_str, "remainder.Scalar(Tensor self, Scalar other) -> Tensor")

// aten::remainder.Scalar(Tensor self, Scalar other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<remainder_Scalar::schema> create_remainder_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(remainder_Scalar::name, remainder_Scalar::overload_name)
      .typed<remainder_Scalar::schema>();
}

// aten::remainder.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor remainder_Scalar::call(const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_remainder_Scalar_typed_handle();
    return op.call(self, other);
}

// aten::remainder.Scalar(Tensor self, Scalar other) -> Tensor
at::Tensor remainder_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other) {
    static auto op = create_remainder_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(remainder__Scalar, name, "aten::remainder_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(remainder__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(remainder__Scalar, schema_str, "remainder_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)")

// aten::remainder_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<remainder__Scalar::schema> create_remainder__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(remainder__Scalar::name, remainder__Scalar::overload_name)
      .typed<remainder__Scalar::schema>();
}

// aten::remainder_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & remainder__Scalar::call(at::Tensor & self, const at::Scalar & other) {
    static auto op = create_remainder__Scalar_typed_handle();
    return op.call(self, other);
}

// aten::remainder_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
at::Tensor & remainder__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, const at::Scalar & other) {
    static auto op = create_remainder__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(min, name, "aten::min")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(min, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(min, schema_str, "min(Tensor self) -> Tensor")

// aten::min(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<min::schema> create_min_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(min::name, min::overload_name)
      .typed<min::schema>();
}

// aten::min(Tensor self) -> Tensor
at::Tensor min::call(const at::Tensor & self) {
    static auto op = create_min_typed_handle();
    return op.call(self);
}

// aten::min(Tensor self) -> Tensor
at::Tensor min::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_min_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fmin, name, "aten::fmin")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fmin, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fmin, schema_str, "fmin(Tensor self, Tensor other) -> Tensor")

// aten::fmin(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fmin::schema> create_fmin_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fmin::name, fmin::overload_name)
      .typed<fmin::schema>();
}

// aten::fmin(Tensor self, Tensor other) -> Tensor
at::Tensor fmin::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_fmin_typed_handle();
    return op.call(self, other);
}

// aten::fmin(Tensor self, Tensor other) -> Tensor
at::Tensor fmin::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_fmin_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(min_out, name, "aten::min")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(min_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(min_out, schema_str, "min.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::min.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<min_out::schema> create_min_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(min_out::name, min_out::overload_name)
      .typed<min_out::schema>();
}

// aten::min.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & min_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_min_out_typed_handle();
    return op.call(self, other, out);
}

// aten::min.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & min_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_min_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantile_scalar_out, name, "aten::quantile")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantile_scalar_out, overload_name, "scalar_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantile_scalar_out, schema_str, "quantile.scalar_out(Tensor self, float q, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)")

// aten::quantile.scalar_out(Tensor self, float q, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<quantile_scalar_out::schema> create_quantile_scalar_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(quantile_scalar_out::name, quantile_scalar_out::overload_name)
      .typed<quantile_scalar_out::schema>();
}

// aten::quantile.scalar_out(Tensor self, float q, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & quantile_scalar_out::call(const at::Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out) {
    static auto op = create_quantile_scalar_out_typed_handle();
    return op.call(self, q, dim, keepdim, out);
}

// aten::quantile.scalar_out(Tensor self, float q, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & quantile_scalar_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, double q, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out) {
    static auto op = create_quantile_scalar_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, q, dim, keepdim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantile_out, name, "aten::quantile")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantile_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantile_out, schema_str, "quantile.out(Tensor self, Tensor q, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)")

// aten::quantile.out(Tensor self, Tensor q, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<quantile_out::schema> create_quantile_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(quantile_out::name, quantile_out::overload_name)
      .typed<quantile_out::schema>();
}

// aten::quantile.out(Tensor self, Tensor q, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & quantile_out::call(const at::Tensor & self, const at::Tensor & q, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out) {
    static auto op = create_quantile_out_typed_handle();
    return op.call(self, q, dim, keepdim, out);
}

// aten::quantile.out(Tensor self, Tensor q, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & quantile_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & q, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out) {
    static auto op = create_quantile_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, q, dim, keepdim, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantile_new_out, name, "aten::quantile")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantile_new_out, overload_name, "new_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(quantile_new_out, schema_str, "quantile.new_out(Tensor self, Tensor q, int? dim, bool keepdim, *, str interpolation, Tensor(a!) out) -> Tensor(a!)")

// aten::quantile.new_out(Tensor self, Tensor q, int? dim, bool keepdim, *, str interpolation, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<quantile_new_out::schema> create_quantile_new_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(quantile_new_out::name, quantile_new_out::overload_name)
      .typed<quantile_new_out::schema>();
}

// aten::quantile.new_out(Tensor self, Tensor q, int? dim, bool keepdim, *, str interpolation, Tensor(a!) out) -> Tensor(a!)
at::Tensor & quantile_new_out::call(const at::Tensor & self, const at::Tensor & q, c10::optional<int64_t> dim, bool keepdim, c10::string_view interpolation, at::Tensor & out) {
    static auto op = create_quantile_new_out_typed_handle();
    return op.call(self, q, dim, keepdim, interpolation, out);
}

// aten::quantile.new_out(Tensor self, Tensor q, int? dim, bool keepdim, *, str interpolation, Tensor(a!) out) -> Tensor(a!)
at::Tensor & quantile_new_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & q, c10::optional<int64_t> dim, bool keepdim, c10::string_view interpolation, at::Tensor & out) {
    static auto op = create_quantile_new_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, q, dim, keepdim, interpolation, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanquantile_new_out, name, "aten::nanquantile")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanquantile_new_out, overload_name, "new_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nanquantile_new_out, schema_str, "nanquantile.new_out(Tensor self, Tensor q, int? dim, bool keepdim, *, str interpolation, Tensor(a!) out) -> Tensor(a!)")

// aten::nanquantile.new_out(Tensor self, Tensor q, int? dim, bool keepdim, *, str interpolation, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<nanquantile_new_out::schema> create_nanquantile_new_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nanquantile_new_out::name, nanquantile_new_out::overload_name)
      .typed<nanquantile_new_out::schema>();
}

// aten::nanquantile.new_out(Tensor self, Tensor q, int? dim, bool keepdim, *, str interpolation, Tensor(a!) out) -> Tensor(a!)
at::Tensor & nanquantile_new_out::call(const at::Tensor & self, const at::Tensor & q, c10::optional<int64_t> dim, bool keepdim, c10::string_view interpolation, at::Tensor & out) {
    static auto op = create_nanquantile_new_out_typed_handle();
    return op.call(self, q, dim, keepdim, interpolation, out);
}

// aten::nanquantile.new_out(Tensor self, Tensor q, int? dim, bool keepdim, *, str interpolation, Tensor(a!) out) -> Tensor(a!)
at::Tensor & nanquantile_new_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & q, c10::optional<int64_t> dim, bool keepdim, c10::string_view interpolation, at::Tensor & out) {
    static auto op = create_nanquantile_new_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, q, dim, keepdim, interpolation, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sort_dimname_stable, name, "aten::sort")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sort_dimname_stable, overload_name, "dimname_stable")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(sort_dimname_stable, schema_str, "sort.dimname_stable(Tensor self, *, bool? stable, Dimname dim, bool descending=False) -> (Tensor values, Tensor indices)")

// aten::sort.dimname_stable(Tensor self, *, bool? stable, Dimname dim, bool descending=False) -> (Tensor values, Tensor indices)
static C10_NOINLINE c10::TypedOperatorHandle<sort_dimname_stable::schema> create_sort_dimname_stable_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(sort_dimname_stable::name, sort_dimname_stable::overload_name)
      .typed<sort_dimname_stable::schema>();
}

// aten::sort.dimname_stable(Tensor self, *, bool? stable, Dimname dim, bool descending=False) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> sort_dimname_stable::call(const at::Tensor & self, c10::optional<bool> stable, at::Dimname dim, bool descending) {
    static auto op = create_sort_dimname_stable_typed_handle();
    return op.call(self, stable, dim, descending);
}

// aten::sort.dimname_stable(Tensor self, *, bool? stable, Dimname dim, bool descending=False) -> (Tensor values, Tensor indices)
::std::tuple<at::Tensor,at::Tensor> sort_dimname_stable::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<bool> stable, at::Dimname dim, bool descending) {
    static auto op = create_sort_dimname_stable_typed_handle();
    return op.redispatch(dispatchKeySet, self, stable, dim, descending);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(all_all_out, name, "aten::all")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(all_all_out, overload_name, "all_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(all_all_out, schema_str, "all.all_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::all.all_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<all_all_out::schema> create_all_all_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(all_all_out::name, all_all_out::overload_name)
      .typed<all_all_out::schema>();
}

// aten::all.all_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & all_all_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_all_all_out_typed_handle();
    return op.call(self, out);
}

// aten::all.all_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & all_all_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_all_all_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(equal, name, "aten::equal")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(equal, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(equal, schema_str, "equal(Tensor self, Tensor other) -> bool")

// aten::equal(Tensor self, Tensor other) -> bool
static C10_NOINLINE c10::TypedOperatorHandle<equal::schema> create_equal_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(equal::name, equal::overload_name)
      .typed<equal::schema>();
}

// aten::equal(Tensor self, Tensor other) -> bool
bool equal::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_equal_typed_handle();
    return op.call(self, other);
}

// aten::equal(Tensor self, Tensor other) -> bool
bool equal::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_equal_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pow_Tensor_Tensor_out, name, "aten::pow")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pow_Tensor_Tensor_out, overload_name, "Tensor_Tensor_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pow_Tensor_Tensor_out, schema_str, "pow.Tensor_Tensor_out(Tensor self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)")

// aten::pow.Tensor_Tensor_out(Tensor self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<pow_Tensor_Tensor_out::schema> create_pow_Tensor_Tensor_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(pow_Tensor_Tensor_out::name, pow_Tensor_Tensor_out::overload_name)
      .typed<pow_Tensor_Tensor_out::schema>();
}

// aten::pow.Tensor_Tensor_out(Tensor self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & pow_Tensor_Tensor_out::call(const at::Tensor & self, const at::Tensor & exponent, at::Tensor & out) {
    static auto op = create_pow_Tensor_Tensor_out_typed_handle();
    return op.call(self, exponent, out);
}

// aten::pow.Tensor_Tensor_out(Tensor self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & pow_Tensor_Tensor_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & exponent, at::Tensor & out) {
    static auto op = create_pow_Tensor_Tensor_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, exponent, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pow_Tensor_Tensor, name, "aten::pow")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pow_Tensor_Tensor, overload_name, "Tensor_Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pow_Tensor_Tensor, schema_str, "pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor")

// aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<pow_Tensor_Tensor::schema> create_pow_Tensor_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(pow_Tensor_Tensor::name, pow_Tensor_Tensor::overload_name)
      .typed<pow_Tensor_Tensor::schema>();
}

// aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor
at::Tensor pow_Tensor_Tensor::call(const at::Tensor & self, const at::Tensor & exponent) {
    static auto op = create_pow_Tensor_Tensor_typed_handle();
    return op.call(self, exponent);
}

// aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor
at::Tensor pow_Tensor_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & exponent) {
    static auto op = create_pow_Tensor_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, exponent);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pow_Scalar_out, name, "aten::pow")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pow_Scalar_out, overload_name, "Scalar_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pow_Scalar_out, schema_str, "pow.Scalar_out(Scalar self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)")

// aten::pow.Scalar_out(Scalar self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<pow_Scalar_out::schema> create_pow_Scalar_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(pow_Scalar_out::name, pow_Scalar_out::overload_name)
      .typed<pow_Scalar_out::schema>();
}

// aten::pow.Scalar_out(Scalar self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & pow_Scalar_out::call(const at::Scalar & self, const at::Tensor & exponent, at::Tensor & out) {
    static auto op = create_pow_Scalar_out_typed_handle();
    return op.call(self, exponent, out);
}

// aten::pow.Scalar_out(Scalar self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & pow_Scalar_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Scalar & self, const at::Tensor & exponent, at::Tensor & out) {
    static auto op = create_pow_Scalar_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, exponent, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(normal_Tensor_Tensor, name, "aten::normal")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(normal_Tensor_Tensor, overload_name, "Tensor_Tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(normal_Tensor_Tensor, schema_str, "normal.Tensor_Tensor(Tensor mean, Tensor std, *, Generator? generator=None) -> Tensor")

// aten::normal.Tensor_Tensor(Tensor mean, Tensor std, *, Generator? generator=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<normal_Tensor_Tensor::schema> create_normal_Tensor_Tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(normal_Tensor_Tensor::name, normal_Tensor_Tensor::overload_name)
      .typed<normal_Tensor_Tensor::schema>();
}

// aten::normal.Tensor_Tensor(Tensor mean, Tensor std, *, Generator? generator=None) -> Tensor
at::Tensor normal_Tensor_Tensor::call(const at::Tensor & mean, const at::Tensor & std, c10::optional<at::Generator> generator) {
    static auto op = create_normal_Tensor_Tensor_typed_handle();
    return op.call(mean, std, generator);
}

// aten::normal.Tensor_Tensor(Tensor mean, Tensor std, *, Generator? generator=None) -> Tensor
at::Tensor normal_Tensor_Tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & mean, const at::Tensor & std, c10::optional<at::Generator> generator) {
    static auto op = create_normal_Tensor_Tensor_typed_handle();
    return op.redispatch(dispatchKeySet, mean, std, generator);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_amp_update_scale_, name, "aten::_amp_update_scale_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_amp_update_scale_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_amp_update_scale_, schema_str, "_amp_update_scale_(Tensor(a!) self, Tensor(b!) growth_tracker, Tensor found_inf, float scale_growth_factor, float scale_backoff_factor, int growth_interval) -> Tensor(a!)")

// aten::_amp_update_scale_(Tensor(a!) self, Tensor(b!) growth_tracker, Tensor found_inf, float scale_growth_factor, float scale_backoff_factor, int growth_interval) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<_amp_update_scale_::schema> create__amp_update_scale__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_amp_update_scale_::name, _amp_update_scale_::overload_name)
      .typed<_amp_update_scale_::schema>();
}

// aten::_amp_update_scale_(Tensor(a!) self, Tensor(b!) growth_tracker, Tensor found_inf, float scale_growth_factor, float scale_backoff_factor, int growth_interval) -> Tensor(a!)
at::Tensor & _amp_update_scale_::call(at::Tensor & self, at::Tensor & growth_tracker, const at::Tensor & found_inf, double scale_growth_factor, double scale_backoff_factor, int64_t growth_interval) {
    static auto op = create__amp_update_scale__typed_handle();
    return op.call(self, growth_tracker, found_inf, scale_growth_factor, scale_backoff_factor, growth_interval);
}

// aten::_amp_update_scale_(Tensor(a!) self, Tensor(b!) growth_tracker, Tensor found_inf, float scale_growth_factor, float scale_backoff_factor, int growth_interval) -> Tensor(a!)
at::Tensor & _amp_update_scale_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, at::Tensor & growth_tracker, const at::Tensor & found_inf, double scale_growth_factor, double scale_backoff_factor, int64_t growth_interval) {
    static auto op = create__amp_update_scale__typed_handle();
    return op.redispatch(dispatchKeySet, self, growth_tracker, found_inf, scale_growth_factor, scale_backoff_factor, growth_interval);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cat, name, "aten::_cat")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cat, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_cat, schema_str, "_cat(Tensor[] tensors, int dim=0) -> Tensor")

// aten::_cat(Tensor[] tensors, int dim=0) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_cat::schema> create__cat_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_cat::name, _cat::overload_name)
      .typed<_cat::schema>();
}

// aten::_cat(Tensor[] tensors, int dim=0) -> Tensor
at::Tensor _cat::call(at::TensorList tensors, int64_t dim) {
    static auto op = create__cat_typed_handle();
    return op.call(tensors, dim);
}

// aten::_cat(Tensor[] tensors, int dim=0) -> Tensor
at::Tensor _cat::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, int64_t dim) {
    static auto op = create__cat_typed_handle();
    return op.redispatch(dispatchKeySet, tensors, dim);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_add_Scalar, name, "aten::_foreach_add")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_add_Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_add_Scalar, schema_str, "_foreach_add.Scalar(Tensor[] tensors, Scalar scalar) -> Tensor[]")

// aten::_foreach_add.Scalar(Tensor[] tensors, Scalar scalar) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_add_Scalar::schema> create__foreach_add_Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_add_Scalar::name, _foreach_add_Scalar::overload_name)
      .typed<_foreach_add_Scalar::schema>();
}

// aten::_foreach_add.Scalar(Tensor[] tensors, Scalar scalar) -> Tensor[]
::std::vector<at::Tensor> _foreach_add_Scalar::call(at::TensorList tensors, const at::Scalar & scalar) {
    static auto op = create__foreach_add_Scalar_typed_handle();
    return op.call(tensors, scalar);
}

// aten::_foreach_add.Scalar(Tensor[] tensors, Scalar scalar) -> Tensor[]
::std::vector<at::Tensor> _foreach_add_Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, const at::Scalar & scalar) {
    static auto op = create__foreach_add_Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, tensors, scalar);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_div__Scalar, name, "aten::_foreach_div_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_div__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_div__Scalar, schema_str, "_foreach_div_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()")

// aten::_foreach_div_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_div__Scalar::schema> create__foreach_div__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_div__Scalar::name, _foreach_div__Scalar::overload_name)
      .typed<_foreach_div__Scalar::schema>();
}

// aten::_foreach_div_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()
void _foreach_div__Scalar::call(at::TensorList self, const at::Scalar & scalar) {
    static auto op = create__foreach_div__Scalar_typed_handle();
    return op.call(self, scalar);
}

// aten::_foreach_div_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()
void _foreach_div__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self, const at::Scalar & scalar) {
    static auto op = create__foreach_div__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, scalar);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sub_List, name, "aten::_foreach_sub")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sub_List, overload_name, "List")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sub_List, schema_str, "_foreach_sub.List(Tensor[] tensors1, Tensor[] tensors2, *, Scalar alpha=1) -> Tensor[]")

// aten::_foreach_sub.List(Tensor[] tensors1, Tensor[] tensors2, *, Scalar alpha=1) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_sub_List::schema> create__foreach_sub_List_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_sub_List::name, _foreach_sub_List::overload_name)
      .typed<_foreach_sub_List::schema>();
}

// aten::_foreach_sub.List(Tensor[] tensors1, Tensor[] tensors2, *, Scalar alpha=1) -> Tensor[]
::std::vector<at::Tensor> _foreach_sub_List::call(at::TensorList tensors1, at::TensorList tensors2, const at::Scalar & alpha) {
    static auto op = create__foreach_sub_List_typed_handle();
    return op.call(tensors1, tensors2, alpha);
}

// aten::_foreach_sub.List(Tensor[] tensors1, Tensor[] tensors2, *, Scalar alpha=1) -> Tensor[]
::std::vector<at::Tensor> _foreach_sub_List::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors1, at::TensorList tensors2, const at::Scalar & alpha) {
    static auto op = create__foreach_sub_List_typed_handle();
    return op.redispatch(dispatchKeySet, tensors1, tensors2, alpha);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_mul_List, name, "aten::_foreach_mul")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_mul_List, overload_name, "List")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_mul_List, schema_str, "_foreach_mul.List(Tensor[] tensors1, Tensor[] tensors2) -> Tensor[]")

// aten::_foreach_mul.List(Tensor[] tensors1, Tensor[] tensors2) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_mul_List::schema> create__foreach_mul_List_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_mul_List::name, _foreach_mul_List::overload_name)
      .typed<_foreach_mul_List::schema>();
}

// aten::_foreach_mul.List(Tensor[] tensors1, Tensor[] tensors2) -> Tensor[]
::std::vector<at::Tensor> _foreach_mul_List::call(at::TensorList tensors1, at::TensorList tensors2) {
    static auto op = create__foreach_mul_List_typed_handle();
    return op.call(tensors1, tensors2);
}

// aten::_foreach_mul.List(Tensor[] tensors1, Tensor[] tensors2) -> Tensor[]
::std::vector<at::Tensor> _foreach_mul_List::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors1, at::TensorList tensors2) {
    static auto op = create__foreach_mul_List_typed_handle();
    return op.redispatch(dispatchKeySet, tensors1, tensors2);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_mul__List, name, "aten::_foreach_mul_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_mul__List, overload_name, "List")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_mul__List, schema_str, "_foreach_mul_.List(Tensor(a!)[] self, Tensor[] other) -> ()")

// aten::_foreach_mul_.List(Tensor(a!)[] self, Tensor[] other) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_mul__List::schema> create__foreach_mul__List_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_mul__List::name, _foreach_mul__List::overload_name)
      .typed<_foreach_mul__List::schema>();
}

// aten::_foreach_mul_.List(Tensor(a!)[] self, Tensor[] other) -> ()
void _foreach_mul__List::call(at::TensorList self, at::TensorList other) {
    static auto op = create__foreach_mul__List_typed_handle();
    return op.call(self, other);
}

// aten::_foreach_mul_.List(Tensor(a!)[] self, Tensor[] other) -> ()
void _foreach_mul__List::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self, at::TensorList other) {
    static auto op = create__foreach_mul__List_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_div_List, name, "aten::_foreach_div")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_div_List, overload_name, "List")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_div_List, schema_str, "_foreach_div.List(Tensor[] tensors1, Tensor[] tensors2) -> Tensor[]")

// aten::_foreach_div.List(Tensor[] tensors1, Tensor[] tensors2) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_div_List::schema> create__foreach_div_List_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_div_List::name, _foreach_div_List::overload_name)
      .typed<_foreach_div_List::schema>();
}

// aten::_foreach_div.List(Tensor[] tensors1, Tensor[] tensors2) -> Tensor[]
::std::vector<at::Tensor> _foreach_div_List::call(at::TensorList tensors1, at::TensorList tensors2) {
    static auto op = create__foreach_div_List_typed_handle();
    return op.call(tensors1, tensors2);
}

// aten::_foreach_div.List(Tensor[] tensors1, Tensor[] tensors2) -> Tensor[]
::std::vector<at::Tensor> _foreach_div_List::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors1, at::TensorList tensors2) {
    static auto op = create__foreach_div_List_typed_handle();
    return op.redispatch(dispatchKeySet, tensors1, tensors2);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_add__ScalarList, name, "aten::_foreach_add_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_add__ScalarList, overload_name, "ScalarList")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_add__ScalarList, schema_str, "_foreach_add_.ScalarList(Tensor(a!)[] self, Scalar[] scalars) -> ()")

// aten::_foreach_add_.ScalarList(Tensor(a!)[] self, Scalar[] scalars) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_add__ScalarList::schema> create__foreach_add__ScalarList_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_add__ScalarList::name, _foreach_add__ScalarList::overload_name)
      .typed<_foreach_add__ScalarList::schema>();
}

// aten::_foreach_add_.ScalarList(Tensor(a!)[] self, Scalar[] scalars) -> ()
void _foreach_add__ScalarList::call(at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
    static auto op = create__foreach_add__ScalarList_typed_handle();
    return op.call(self, scalars);
}

// aten::_foreach_add_.ScalarList(Tensor(a!)[] self, Scalar[] scalars) -> ()
void _foreach_add__ScalarList::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
    static auto op = create__foreach_add__ScalarList_typed_handle();
    return op.redispatch(dispatchKeySet, self, scalars);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_zero_, name, "aten::_foreach_zero_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_zero_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_zero_, schema_str, "_foreach_zero_(Tensor(a!)[] self) -> ()")

// aten::_foreach_zero_(Tensor(a!)[] self) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_zero_::schema> create__foreach_zero__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_zero_::name, _foreach_zero_::overload_name)
      .typed<_foreach_zero_::schema>();
}

// aten::_foreach_zero_(Tensor(a!)[] self) -> ()
void _foreach_zero_::call(at::TensorList self) {
    static auto op = create__foreach_zero__typed_handle();
    return op.call(self);
}

// aten::_foreach_zero_(Tensor(a!)[] self) -> ()
void _foreach_zero_::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self) {
    static auto op = create__foreach_zero__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_asin, name, "aten::_foreach_asin")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_asin, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_asin, schema_str, "_foreach_asin(Tensor[] tensors) -> Tensor[]")

// aten::_foreach_asin(Tensor[] tensors) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_asin::schema> create__foreach_asin_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_asin::name, _foreach_asin::overload_name)
      .typed<_foreach_asin::schema>();
}

// aten::_foreach_asin(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_asin::call(at::TensorList tensors) {
    static auto op = create__foreach_asin_typed_handle();
    return op.call(tensors);
}

// aten::_foreach_asin(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_asin::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create__foreach_asin_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_ceil_, name, "aten::_foreach_ceil_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_ceil_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_ceil_, schema_str, "_foreach_ceil_(Tensor(a!)[] self) -> ()")

// aten::_foreach_ceil_(Tensor(a!)[] self) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_ceil_::schema> create__foreach_ceil__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_ceil_::name, _foreach_ceil_::overload_name)
      .typed<_foreach_ceil_::schema>();
}

// aten::_foreach_ceil_(Tensor(a!)[] self) -> ()
void _foreach_ceil_::call(at::TensorList self) {
    static auto op = create__foreach_ceil__typed_handle();
    return op.call(self);
}

// aten::_foreach_ceil_(Tensor(a!)[] self) -> ()
void _foreach_ceil_::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self) {
    static auto op = create__foreach_ceil__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_cos, name, "aten::_foreach_cos")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_cos, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_cos, schema_str, "_foreach_cos(Tensor[] tensors) -> Tensor[]")

// aten::_foreach_cos(Tensor[] tensors) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_cos::schema> create__foreach_cos_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_cos::name, _foreach_cos::overload_name)
      .typed<_foreach_cos::schema>();
}

// aten::_foreach_cos(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_cos::call(at::TensorList tensors) {
    static auto op = create__foreach_cos_typed_handle();
    return op.call(tensors);
}

// aten::_foreach_cos(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_cos::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create__foreach_cos_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_cos_, name, "aten::_foreach_cos_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_cos_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_cos_, schema_str, "_foreach_cos_(Tensor(a!)[] self) -> ()")

// aten::_foreach_cos_(Tensor(a!)[] self) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_cos_::schema> create__foreach_cos__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_cos_::name, _foreach_cos_::overload_name)
      .typed<_foreach_cos_::schema>();
}

// aten::_foreach_cos_(Tensor(a!)[] self) -> ()
void _foreach_cos_::call(at::TensorList self) {
    static auto op = create__foreach_cos__typed_handle();
    return op.call(self);
}

// aten::_foreach_cos_(Tensor(a!)[] self) -> ()
void _foreach_cos_::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self) {
    static auto op = create__foreach_cos__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_floor, name, "aten::_foreach_floor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_floor, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_floor, schema_str, "_foreach_floor(Tensor[] tensors) -> Tensor[]")

// aten::_foreach_floor(Tensor[] tensors) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_floor::schema> create__foreach_floor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_floor::name, _foreach_floor::overload_name)
      .typed<_foreach_floor::schema>();
}

// aten::_foreach_floor(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_floor::call(at::TensorList tensors) {
    static auto op = create__foreach_floor_typed_handle();
    return op.call(tensors);
}

// aten::_foreach_floor(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_floor::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create__foreach_floor_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_floor_, name, "aten::_foreach_floor_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_floor_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_floor_, schema_str, "_foreach_floor_(Tensor(a!)[] self) -> ()")

// aten::_foreach_floor_(Tensor(a!)[] self) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_floor_::schema> create__foreach_floor__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_floor_::name, _foreach_floor_::overload_name)
      .typed<_foreach_floor_::schema>();
}

// aten::_foreach_floor_(Tensor(a!)[] self) -> ()
void _foreach_floor_::call(at::TensorList self) {
    static auto op = create__foreach_floor__typed_handle();
    return op.call(self);
}

// aten::_foreach_floor_(Tensor(a!)[] self) -> ()
void _foreach_floor_::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self) {
    static auto op = create__foreach_floor__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_log2_, name, "aten::_foreach_log2_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_log2_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_log2_, schema_str, "_foreach_log2_(Tensor(a!)[] self) -> ()")

// aten::_foreach_log2_(Tensor(a!)[] self) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_log2_::schema> create__foreach_log2__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_log2_::name, _foreach_log2_::overload_name)
      .typed<_foreach_log2_::schema>();
}

// aten::_foreach_log2_(Tensor(a!)[] self) -> ()
void _foreach_log2_::call(at::TensorList self) {
    static auto op = create__foreach_log2__typed_handle();
    return op.call(self);
}

// aten::_foreach_log2_(Tensor(a!)[] self) -> ()
void _foreach_log2_::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self) {
    static auto op = create__foreach_log2__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_tanh, name, "aten::_foreach_tanh")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_tanh, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_tanh, schema_str, "_foreach_tanh(Tensor[] tensors) -> Tensor[]")

// aten::_foreach_tanh(Tensor[] tensors) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_tanh::schema> create__foreach_tanh_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_tanh::name, _foreach_tanh::overload_name)
      .typed<_foreach_tanh::schema>();
}

// aten::_foreach_tanh(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_tanh::call(at::TensorList tensors) {
    static auto op = create__foreach_tanh_typed_handle();
    return op.call(tensors);
}

// aten::_foreach_tanh(Tensor[] tensors) -> Tensor[]
::std::vector<at::Tensor> _foreach_tanh::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors) {
    static auto op = create__foreach_tanh_typed_handle();
    return op.redispatch(dispatchKeySet, tensors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sin_, name, "aten::_foreach_sin_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sin_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_sin_, schema_str, "_foreach_sin_(Tensor(a!)[] self) -> ()")

// aten::_foreach_sin_(Tensor(a!)[] self) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_sin_::schema> create__foreach_sin__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_sin_::name, _foreach_sin_::overload_name)
      .typed<_foreach_sin_::schema>();
}

// aten::_foreach_sin_(Tensor(a!)[] self) -> ()
void _foreach_sin_::call(at::TensorList self) {
    static auto op = create__foreach_sin__typed_handle();
    return op.call(self);
}

// aten::_foreach_sin_(Tensor(a!)[] self) -> ()
void _foreach_sin_::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self) {
    static auto op = create__foreach_sin__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_lgamma_, name, "aten::_foreach_lgamma_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_lgamma_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_lgamma_, schema_str, "_foreach_lgamma_(Tensor(a!)[] self) -> ()")

// aten::_foreach_lgamma_(Tensor(a!)[] self) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_lgamma_::schema> create__foreach_lgamma__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_lgamma_::name, _foreach_lgamma_::overload_name)
      .typed<_foreach_lgamma_::schema>();
}

// aten::_foreach_lgamma_(Tensor(a!)[] self) -> ()
void _foreach_lgamma_::call(at::TensorList self) {
    static auto op = create__foreach_lgamma__typed_handle();
    return op.call(self);
}

// aten::_foreach_lgamma_(Tensor(a!)[] self) -> ()
void _foreach_lgamma_::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self) {
    static auto op = create__foreach_lgamma__typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_addcmul__Scalar, name, "aten::_foreach_addcmul_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_addcmul__Scalar, overload_name, "Scalar")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_addcmul__Scalar, schema_str, "_foreach_addcmul_.Scalar(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1) -> ()")

// aten::_foreach_addcmul_.Scalar(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_addcmul__Scalar::schema> create__foreach_addcmul__Scalar_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_addcmul__Scalar::name, _foreach_addcmul__Scalar::overload_name)
      .typed<_foreach_addcmul__Scalar::schema>();
}

// aten::_foreach_addcmul_.Scalar(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1) -> ()
void _foreach_addcmul__Scalar::call(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value) {
    static auto op = create__foreach_addcmul__Scalar_typed_handle();
    return op.call(self, tensor1, tensor2, value);
}

// aten::_foreach_addcmul_.Scalar(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1) -> ()
void _foreach_addcmul__Scalar::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value) {
    static auto op = create__foreach_addcmul__Scalar_typed_handle();
    return op.redispatch(dispatchKeySet, self, tensor1, tensor2, value);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_addcdiv__ScalarList, name, "aten::_foreach_addcdiv_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_addcdiv__ScalarList, overload_name, "ScalarList")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_addcdiv__ScalarList, schema_str, "_foreach_addcdiv_.ScalarList(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars) -> ()")

// aten::_foreach_addcdiv_.ScalarList(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_addcdiv__ScalarList::schema> create__foreach_addcdiv__ScalarList_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_addcdiv__ScalarList::name, _foreach_addcdiv__ScalarList::overload_name)
      .typed<_foreach_addcdiv__ScalarList::schema>();
}

// aten::_foreach_addcdiv_.ScalarList(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars) -> ()
void _foreach_addcdiv__ScalarList::call(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars) {
    static auto op = create__foreach_addcdiv__ScalarList_typed_handle();
    return op.call(self, tensor1, tensor2, scalars);
}

// aten::_foreach_addcdiv_.ScalarList(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars) -> ()
void _foreach_addcdiv__ScalarList::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars) {
    static auto op = create__foreach_addcdiv__ScalarList_typed_handle();
    return op.redispatch(dispatchKeySet, self, tensor1, tensor2, scalars);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_maximum_List, name, "aten::_foreach_maximum")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_maximum_List, overload_name, "List")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_foreach_maximum_List, schema_str, "_foreach_maximum.List(Tensor[] tensors1, Tensor[] tensors2) -> Tensor[]")

// aten::_foreach_maximum.List(Tensor[] tensors1, Tensor[] tensors2) -> Tensor[]
static C10_NOINLINE c10::TypedOperatorHandle<_foreach_maximum_List::schema> create__foreach_maximum_List_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_foreach_maximum_List::name, _foreach_maximum_List::overload_name)
      .typed<_foreach_maximum_List::schema>();
}

// aten::_foreach_maximum.List(Tensor[] tensors1, Tensor[] tensors2) -> Tensor[]
::std::vector<at::Tensor> _foreach_maximum_List::call(at::TensorList tensors1, at::TensorList tensors2) {
    static auto op = create__foreach_maximum_List_typed_handle();
    return op.call(tensors1, tensors2);
}

// aten::_foreach_maximum.List(Tensor[] tensors1, Tensor[] tensors2) -> Tensor[]
::std::vector<at::Tensor> _foreach_maximum_List::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors1, at::TensorList tensors2) {
    static auto op = create__foreach_maximum_List_typed_handle();
    return op.redispatch(dispatchKeySet, tensors1, tensors2);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bucketize_Tensor_out, name, "aten::bucketize")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bucketize_Tensor_out, overload_name, "Tensor_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(bucketize_Tensor_out, schema_str, "bucketize.Tensor_out(Tensor self, Tensor boundaries, *, bool out_int32=False, bool right=False, Tensor(a!) out) -> Tensor(a!)")

// aten::bucketize.Tensor_out(Tensor self, Tensor boundaries, *, bool out_int32=False, bool right=False, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<bucketize_Tensor_out::schema> create_bucketize_Tensor_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(bucketize_Tensor_out::name, bucketize_Tensor_out::overload_name)
      .typed<bucketize_Tensor_out::schema>();
}

// aten::bucketize.Tensor_out(Tensor self, Tensor boundaries, *, bool out_int32=False, bool right=False, Tensor(a!) out) -> Tensor(a!)
at::Tensor & bucketize_Tensor_out::call(const at::Tensor & self, const at::Tensor & boundaries, bool out_int32, bool right, at::Tensor & out) {
    static auto op = create_bucketize_Tensor_out_typed_handle();
    return op.call(self, boundaries, out_int32, right, out);
}

// aten::bucketize.Tensor_out(Tensor self, Tensor boundaries, *, bool out_int32=False, bool right=False, Tensor(a!) out) -> Tensor(a!)
at::Tensor & bucketize_Tensor_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & boundaries, bool out_int32, bool right, at::Tensor & out) {
    static auto op = create_bucketize_Tensor_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, boundaries, out_int32, right, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_convert_indices_from_coo_to_csr_out, name, "aten::_convert_indices_from_coo_to_csr")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_convert_indices_from_coo_to_csr_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_convert_indices_from_coo_to_csr_out, schema_str, "_convert_indices_from_coo_to_csr.out(Tensor self, int size, *, bool out_int32=False, Tensor(a!) out) -> Tensor(a!)")

// aten::_convert_indices_from_coo_to_csr.out(Tensor self, int size, *, bool out_int32=False, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<_convert_indices_from_coo_to_csr_out::schema> create__convert_indices_from_coo_to_csr_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_convert_indices_from_coo_to_csr_out::name, _convert_indices_from_coo_to_csr_out::overload_name)
      .typed<_convert_indices_from_coo_to_csr_out::schema>();
}

// aten::_convert_indices_from_coo_to_csr.out(Tensor self, int size, *, bool out_int32=False, Tensor(a!) out) -> Tensor(a!)
at::Tensor & _convert_indices_from_coo_to_csr_out::call(const at::Tensor & self, int64_t size, bool out_int32, at::Tensor & out) {
    static auto op = create__convert_indices_from_coo_to_csr_out_typed_handle();
    return op.call(self, size, out_int32, out);
}

// aten::_convert_indices_from_coo_to_csr.out(Tensor self, int size, *, bool out_int32=False, Tensor(a!) out) -> Tensor(a!)
at::Tensor & _convert_indices_from_coo_to_csr_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t size, bool out_int32, at::Tensor & out) {
    static auto op = create__convert_indices_from_coo_to_csr_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, size, out_int32, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(l1_loss_backward_grad_input, name, "aten::l1_loss_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(l1_loss_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(l1_loss_backward_grad_input, schema_str, "l1_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::l1_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<l1_loss_backward_grad_input::schema> create_l1_loss_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(l1_loss_backward_grad_input::name, l1_loss_backward_grad_input::overload_name)
      .typed<l1_loss_backward_grad_input::schema>();
}

// aten::l1_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & l1_loss_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & grad_input) {
    static auto op = create_l1_loss_backward_grad_input_typed_handle();
    return op.call(grad_output, self, target, reduction, grad_input);
}

// aten::l1_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & l1_loss_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & grad_input) {
    static auto op = create_l1_loss_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, target, reduction, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss, name, "aten::nll_loss")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss, schema_str, "nll_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100) -> Tensor")

// aten::nll_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<nll_loss::schema> create_nll_loss_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nll_loss::name, nll_loss::overload_name)
      .typed<nll_loss::schema>();
}

// aten::nll_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100) -> Tensor
at::Tensor nll_loss::call(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index) {
    static auto op = create_nll_loss_typed_handle();
    return op.call(self, target, weight, reduction, ignore_index);
}

// aten::nll_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100) -> Tensor
at::Tensor nll_loss::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index) {
    static auto op = create_nll_loss_typed_handle();
    return op.redispatch(dispatchKeySet, self, target, weight, reduction, ignore_index);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss_backward, name, "aten::nll_loss_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(nll_loss_backward, schema_str, "nll_loss_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight) -> Tensor")

// aten::nll_loss_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<nll_loss_backward::schema> create_nll_loss_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(nll_loss_backward::name, nll_loss_backward::overload_name)
      .typed<nll_loss_backward::schema>();
}

// aten::nll_loss_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight) -> Tensor
at::Tensor nll_loss_backward::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight) {
    static auto op = create_nll_loss_backward_typed_handle();
    return op.call(grad_output, self, target, weight, reduction, ignore_index, total_weight);
}

// aten::nll_loss_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight) -> Tensor
at::Tensor nll_loss_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight) {
    static auto op = create_nll_loss_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, target, weight, reduction, ignore_index, total_weight);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(smooth_l1_loss_backward, name, "aten::smooth_l1_loss_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(smooth_l1_loss_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(smooth_l1_loss_backward, schema_str, "smooth_l1_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, float beta) -> Tensor")

// aten::smooth_l1_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, float beta) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<smooth_l1_loss_backward::schema> create_smooth_l1_loss_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(smooth_l1_loss_backward::name, smooth_l1_loss_backward::overload_name)
      .typed<smooth_l1_loss_backward::schema>();
}

// aten::smooth_l1_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, float beta) -> Tensor
at::Tensor smooth_l1_loss_backward::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta) {
    static auto op = create_smooth_l1_loss_backward_typed_handle();
    return op.call(grad_output, self, target, reduction, beta);
}

// aten::smooth_l1_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, float beta) -> Tensor
at::Tensor smooth_l1_loss_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta) {
    static auto op = create_smooth_l1_loss_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, target, reduction, beta);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(huber_loss, name, "aten::huber_loss")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(huber_loss, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(huber_loss, schema_str, "huber_loss(Tensor self, Tensor target, int reduction=Mean, float delta=1.0) -> Tensor")

// aten::huber_loss(Tensor self, Tensor target, int reduction=Mean, float delta=1.0) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<huber_loss::schema> create_huber_loss_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(huber_loss::name, huber_loss::overload_name)
      .typed<huber_loss::schema>();
}

// aten::huber_loss(Tensor self, Tensor target, int reduction=Mean, float delta=1.0) -> Tensor
at::Tensor huber_loss::call(const at::Tensor & self, const at::Tensor & target, int64_t reduction, double delta) {
    static auto op = create_huber_loss_typed_handle();
    return op.call(self, target, reduction, delta);
}

// aten::huber_loss(Tensor self, Tensor target, int reduction=Mean, float delta=1.0) -> Tensor
at::Tensor huber_loss::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double delta) {
    static auto op = create_huber_loss_typed_handle();
    return op.redispatch(dispatchKeySet, self, target, reduction, delta);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(huber_loss_backward, name, "aten::huber_loss_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(huber_loss_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(huber_loss_backward, schema_str, "huber_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, float delta) -> Tensor")

// aten::huber_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, float delta) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<huber_loss_backward::schema> create_huber_loss_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(huber_loss_backward::name, huber_loss_backward::overload_name)
      .typed<huber_loss_backward::schema>();
}

// aten::huber_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, float delta) -> Tensor
at::Tensor huber_loss_backward::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double delta) {
    static auto op = create_huber_loss_backward_typed_handle();
    return op.call(grad_output, self, target, reduction, delta);
}

// aten::huber_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, float delta) -> Tensor
at::Tensor huber_loss_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double delta) {
    static auto op = create_huber_loss_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, target, reduction, delta);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(elu_backward_grad_input, name, "aten::elu_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(elu_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(elu_backward_grad_input, schema_str, "elu_backward.grad_input(Tensor grad_output, Scalar alpha, Scalar scale, Scalar input_scale, bool is_result, Tensor self_or_result, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::elu_backward.grad_input(Tensor grad_output, Scalar alpha, Scalar scale, Scalar input_scale, bool is_result, Tensor self_or_result, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<elu_backward_grad_input::schema> create_elu_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(elu_backward_grad_input::name, elu_backward_grad_input::overload_name)
      .typed<elu_backward_grad_input::schema>();
}

// aten::elu_backward.grad_input(Tensor grad_output, Scalar alpha, Scalar scale, Scalar input_scale, bool is_result, Tensor self_or_result, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & elu_backward_grad_input::call(const at::Tensor & grad_output, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale, bool is_result, const at::Tensor & self_or_result, at::Tensor & grad_input) {
    static auto op = create_elu_backward_grad_input_typed_handle();
    return op.call(grad_output, alpha, scale, input_scale, is_result, self_or_result, grad_input);
}

// aten::elu_backward.grad_input(Tensor grad_output, Scalar alpha, Scalar scale, Scalar input_scale, bool is_result, Tensor self_or_result, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & elu_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale, bool is_result, const at::Tensor & self_or_result, at::Tensor & grad_input) {
    static auto op = create_elu_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, alpha, scale, input_scale, is_result, self_or_result, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardsigmoid, name, "aten::hardsigmoid")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardsigmoid, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardsigmoid, schema_str, "hardsigmoid(Tensor self) -> Tensor")

// aten::hardsigmoid(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<hardsigmoid::schema> create_hardsigmoid_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(hardsigmoid::name, hardsigmoid::overload_name)
      .typed<hardsigmoid::schema>();
}

// aten::hardsigmoid(Tensor self) -> Tensor
at::Tensor hardsigmoid::call(const at::Tensor & self) {
    static auto op = create_hardsigmoid_typed_handle();
    return op.call(self);
}

// aten::hardsigmoid(Tensor self) -> Tensor
at::Tensor hardsigmoid::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_hardsigmoid_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardtanh_backward_grad_input, name, "aten::hardtanh_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardtanh_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(hardtanh_backward_grad_input, schema_str, "hardtanh_backward.grad_input(Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::hardtanh_backward.grad_input(Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<hardtanh_backward_grad_input::schema> create_hardtanh_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(hardtanh_backward_grad_input::name, hardtanh_backward_grad_input::overload_name)
      .typed<hardtanh_backward_grad_input::schema>();
}

// aten::hardtanh_backward.grad_input(Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & hardtanh_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val, at::Tensor & grad_input) {
    static auto op = create_hardtanh_backward_grad_input_typed_handle();
    return op.call(grad_output, self, min_val, max_val, grad_input);
}

// aten::hardtanh_backward.grad_input(Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & hardtanh_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val, at::Tensor & grad_input) {
    static auto op = create_hardtanh_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, min_val, max_val, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log_sigmoid_out, name, "aten::log_sigmoid")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log_sigmoid_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log_sigmoid_out, schema_str, "log_sigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::log_sigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<log_sigmoid_out::schema> create_log_sigmoid_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(log_sigmoid_out::name, log_sigmoid_out::overload_name)
      .typed<log_sigmoid_out::schema>();
}

// aten::log_sigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & log_sigmoid_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_log_sigmoid_out_typed_handle();
    return op.call(self, out);
}

// aten::log_sigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & log_sigmoid_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_log_sigmoid_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log_sigmoid, name, "aten::log_sigmoid")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log_sigmoid, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log_sigmoid, schema_str, "log_sigmoid(Tensor self) -> Tensor")

// aten::log_sigmoid(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<log_sigmoid::schema> create_log_sigmoid_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(log_sigmoid::name, log_sigmoid::overload_name)
      .typed<log_sigmoid::schema>();
}

// aten::log_sigmoid(Tensor self) -> Tensor
at::Tensor log_sigmoid::call(const at::Tensor & self) {
    static auto op = create_log_sigmoid_typed_handle();
    return op.call(self);
}

// aten::log_sigmoid(Tensor self) -> Tensor
at::Tensor log_sigmoid::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_log_sigmoid_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log_sigmoid_forward_output, name, "aten::log_sigmoid_forward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log_sigmoid_forward_output, overload_name, "output")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(log_sigmoid_forward_output, schema_str, "log_sigmoid_forward.output(Tensor self, *, Tensor(a!) output, Tensor(b!) buffer) -> (Tensor(a!), Tensor(b!))")

// aten::log_sigmoid_forward.output(Tensor self, *, Tensor(a!) output, Tensor(b!) buffer) -> (Tensor(a!), Tensor(b!))
static C10_NOINLINE c10::TypedOperatorHandle<log_sigmoid_forward_output::schema> create_log_sigmoid_forward_output_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(log_sigmoid_forward_output::name, log_sigmoid_forward_output::overload_name)
      .typed<log_sigmoid_forward_output::schema>();
}

// aten::log_sigmoid_forward.output(Tensor self, *, Tensor(a!) output, Tensor(b!) buffer) -> (Tensor(a!), Tensor(b!))
::std::tuple<at::Tensor &,at::Tensor &> log_sigmoid_forward_output::call(const at::Tensor & self, at::Tensor & output, at::Tensor & buffer) {
    static auto op = create_log_sigmoid_forward_output_typed_handle();
    return op.call(self, output, buffer);
}

// aten::log_sigmoid_forward.output(Tensor self, *, Tensor(a!) output, Tensor(b!) buffer) -> (Tensor(a!), Tensor(b!))
::std::tuple<at::Tensor &,at::Tensor &> log_sigmoid_forward_output::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & output, at::Tensor & buffer) {
    static auto op = create_log_sigmoid_forward_output_typed_handle();
    return op.redispatch(dispatchKeySet, self, output, buffer);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_avg_pool2d_out, name, "aten::adaptive_avg_pool2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_avg_pool2d_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_avg_pool2d_out, schema_str, "adaptive_avg_pool2d.out(Tensor self, int[2] output_size, *, Tensor(a!) out) -> Tensor(a!)")

// aten::adaptive_avg_pool2d.out(Tensor self, int[2] output_size, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<adaptive_avg_pool2d_out::schema> create_adaptive_avg_pool2d_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(adaptive_avg_pool2d_out::name, adaptive_avg_pool2d_out::overload_name)
      .typed<adaptive_avg_pool2d_out::schema>();
}

// aten::adaptive_avg_pool2d.out(Tensor self, int[2] output_size, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & adaptive_avg_pool2d_out::call(const at::Tensor & self, at::IntArrayRef output_size, at::Tensor & out) {
    static auto op = create_adaptive_avg_pool2d_out_typed_handle();
    return op.call(self, output_size, out);
}

// aten::adaptive_avg_pool2d.out(Tensor self, int[2] output_size, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & adaptive_avg_pool2d_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size, at::Tensor & out) {
    static auto op = create_adaptive_avg_pool2d_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, output_size, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_avg_pool2d, name, "aten::adaptive_avg_pool2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_avg_pool2d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_avg_pool2d, schema_str, "adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor")

// aten::adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<adaptive_avg_pool2d::schema> create_adaptive_avg_pool2d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(adaptive_avg_pool2d::name, adaptive_avg_pool2d::overload_name)
      .typed<adaptive_avg_pool2d::schema>();
}

// aten::adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor
at::Tensor adaptive_avg_pool2d::call(const at::Tensor & self, at::IntArrayRef output_size) {
    static auto op = create_adaptive_avg_pool2d_typed_handle();
    return op.call(self, output_size);
}

// aten::adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor
at::Tensor adaptive_avg_pool2d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size) {
    static auto op = create_adaptive_avg_pool2d_typed_handle();
    return op.redispatch(dispatchKeySet, self, output_size);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_avg_pool3d, name, "aten::adaptive_avg_pool3d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_avg_pool3d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_avg_pool3d, schema_str, "adaptive_avg_pool3d(Tensor self, int[3] output_size) -> Tensor")

// aten::adaptive_avg_pool3d(Tensor self, int[3] output_size) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<adaptive_avg_pool3d::schema> create_adaptive_avg_pool3d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(adaptive_avg_pool3d::name, adaptive_avg_pool3d::overload_name)
      .typed<adaptive_avg_pool3d::schema>();
}

// aten::adaptive_avg_pool3d(Tensor self, int[3] output_size) -> Tensor
at::Tensor adaptive_avg_pool3d::call(const at::Tensor & self, at::IntArrayRef output_size) {
    static auto op = create_adaptive_avg_pool3d_typed_handle();
    return op.call(self, output_size);
}

// aten::adaptive_avg_pool3d(Tensor self, int[3] output_size) -> Tensor
at::Tensor adaptive_avg_pool3d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size) {
    static auto op = create_adaptive_avg_pool3d_typed_handle();
    return op.redispatch(dispatchKeySet, self, output_size);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_adaptive_avg_pool3d, name, "aten::_adaptive_avg_pool3d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_adaptive_avg_pool3d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_adaptive_avg_pool3d, schema_str, "_adaptive_avg_pool3d(Tensor self, int[3] output_size) -> Tensor")

// aten::_adaptive_avg_pool3d(Tensor self, int[3] output_size) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_adaptive_avg_pool3d::schema> create__adaptive_avg_pool3d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_adaptive_avg_pool3d::name, _adaptive_avg_pool3d::overload_name)
      .typed<_adaptive_avg_pool3d::schema>();
}

// aten::_adaptive_avg_pool3d(Tensor self, int[3] output_size) -> Tensor
at::Tensor _adaptive_avg_pool3d::call(const at::Tensor & self, at::IntArrayRef output_size) {
    static auto op = create__adaptive_avg_pool3d_typed_handle();
    return op.call(self, output_size);
}

// aten::_adaptive_avg_pool3d(Tensor self, int[3] output_size) -> Tensor
at::Tensor _adaptive_avg_pool3d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size) {
    static auto op = create__adaptive_avg_pool3d_typed_handle();
    return op.redispatch(dispatchKeySet, self, output_size);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_avg_pool3d_backward_grad_input, name, "aten::adaptive_avg_pool3d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_avg_pool3d_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_avg_pool3d_backward_grad_input, schema_str, "adaptive_avg_pool3d_backward.grad_input(Tensor grad_output, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::adaptive_avg_pool3d_backward.grad_input(Tensor grad_output, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<adaptive_avg_pool3d_backward_grad_input::schema> create_adaptive_avg_pool3d_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(adaptive_avg_pool3d_backward_grad_input::name, adaptive_avg_pool3d_backward_grad_input::overload_name)
      .typed<adaptive_avg_pool3d_backward_grad_input::schema>();
}

// aten::adaptive_avg_pool3d_backward.grad_input(Tensor grad_output, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & adaptive_avg_pool3d_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, at::Tensor & grad_input) {
    static auto op = create_adaptive_avg_pool3d_backward_grad_input_typed_handle();
    return op.call(grad_output, self, grad_input);
}

// aten::adaptive_avg_pool3d_backward.grad_input(Tensor grad_output, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & adaptive_avg_pool3d_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::Tensor & grad_input) {
    static auto op = create_adaptive_avg_pool3d_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_max_pool2d, name, "aten::adaptive_max_pool2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_max_pool2d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_max_pool2d, schema_str, "adaptive_max_pool2d(Tensor self, int[2] output_size) -> (Tensor, Tensor)")

// aten::adaptive_max_pool2d(Tensor self, int[2] output_size) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<adaptive_max_pool2d::schema> create_adaptive_max_pool2d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(adaptive_max_pool2d::name, adaptive_max_pool2d::overload_name)
      .typed<adaptive_max_pool2d::schema>();
}

// aten::adaptive_max_pool2d(Tensor self, int[2] output_size) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> adaptive_max_pool2d::call(const at::Tensor & self, at::IntArrayRef output_size) {
    static auto op = create_adaptive_max_pool2d_typed_handle();
    return op.call(self, output_size);
}

// aten::adaptive_max_pool2d(Tensor self, int[2] output_size) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> adaptive_max_pool2d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size) {
    static auto op = create_adaptive_max_pool2d_typed_handle();
    return op.redispatch(dispatchKeySet, self, output_size);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_max_pool3d, name, "aten::adaptive_max_pool3d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_max_pool3d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(adaptive_max_pool3d, schema_str, "adaptive_max_pool3d(Tensor self, int[3] output_size) -> (Tensor, Tensor)")

// aten::adaptive_max_pool3d(Tensor self, int[3] output_size) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<adaptive_max_pool3d::schema> create_adaptive_max_pool3d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(adaptive_max_pool3d::name, adaptive_max_pool3d::overload_name)
      .typed<adaptive_max_pool3d::schema>();
}

// aten::adaptive_max_pool3d(Tensor self, int[3] output_size) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> adaptive_max_pool3d::call(const at::Tensor & self, at::IntArrayRef output_size) {
    static auto op = create_adaptive_max_pool3d_typed_handle();
    return op.call(self, output_size);
}

// aten::adaptive_max_pool3d(Tensor self, int[3] output_size) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> adaptive_max_pool3d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size) {
    static auto op = create_adaptive_max_pool3d_typed_handle();
    return op.redispatch(dispatchKeySet, self, output_size);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(avg_pool2d_backward, name, "aten::avg_pool2d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(avg_pool2d_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(avg_pool2d_backward, schema_str, "avg_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor")

// aten::avg_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<avg_pool2d_backward::schema> create_avg_pool2d_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(avg_pool2d_backward::name, avg_pool2d_backward::overload_name)
      .typed<avg_pool2d_backward::schema>();
}

// aten::avg_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor
at::Tensor avg_pool2d_backward::call(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
    static auto op = create_avg_pool2d_backward_typed_handle();
    return op.call(grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

// aten::avg_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor
at::Tensor avg_pool2d_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
    static auto op = create_avg_pool2d_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(avg_pool3d_out, name, "aten::avg_pool3d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(avg_pool3d_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(avg_pool3d_out, schema_str, "avg_pool3d.out(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::avg_pool3d.out(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<avg_pool3d_out::schema> create_avg_pool3d_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(avg_pool3d_out::name, avg_pool3d_out::overload_name)
      .typed<avg_pool3d_out::schema>();
}

// aten::avg_pool3d.out(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & avg_pool3d_out::call(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, at::Tensor & out) {
    static auto op = create_avg_pool3d_out_typed_handle();
    return op.call(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, out);
}

// aten::avg_pool3d.out(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & avg_pool3d_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, at::Tensor & out) {
    static auto op = create_avg_pool3d_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fractional_max_pool2d, name, "aten::fractional_max_pool2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fractional_max_pool2d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fractional_max_pool2d, schema_str, "fractional_max_pool2d(Tensor self, int[2] kernel_size, int[2] output_size, Tensor random_samples) -> (Tensor, Tensor)")

// aten::fractional_max_pool2d(Tensor self, int[2] kernel_size, int[2] output_size, Tensor random_samples) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<fractional_max_pool2d::schema> create_fractional_max_pool2d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fractional_max_pool2d::name, fractional_max_pool2d::overload_name)
      .typed<fractional_max_pool2d::schema>();
}

// aten::fractional_max_pool2d(Tensor self, int[2] kernel_size, int[2] output_size, Tensor random_samples) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> fractional_max_pool2d::call(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & random_samples) {
    static auto op = create_fractional_max_pool2d_typed_handle();
    return op.call(self, kernel_size, output_size, random_samples);
}

// aten::fractional_max_pool2d(Tensor self, int[2] kernel_size, int[2] output_size, Tensor random_samples) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> fractional_max_pool2d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & random_samples) {
    static auto op = create_fractional_max_pool2d_typed_handle();
    return op.redispatch(dispatchKeySet, self, kernel_size, output_size, random_samples);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fractional_max_pool3d_output, name, "aten::fractional_max_pool3d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fractional_max_pool3d_output, overload_name, "output")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fractional_max_pool3d_output, schema_str, "fractional_max_pool3d.output(Tensor self, int[3] kernel_size, int[3] output_size, Tensor random_samples, *, Tensor(a!) output, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))")

// aten::fractional_max_pool3d.output(Tensor self, int[3] kernel_size, int[3] output_size, Tensor random_samples, *, Tensor(a!) output, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
static C10_NOINLINE c10::TypedOperatorHandle<fractional_max_pool3d_output::schema> create_fractional_max_pool3d_output_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fractional_max_pool3d_output::name, fractional_max_pool3d_output::overload_name)
      .typed<fractional_max_pool3d_output::schema>();
}

// aten::fractional_max_pool3d.output(Tensor self, int[3] kernel_size, int[3] output_size, Tensor random_samples, *, Tensor(a!) output, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
::std::tuple<at::Tensor &,at::Tensor &> fractional_max_pool3d_output::call(const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & random_samples, at::Tensor & output, at::Tensor & indices) {
    static auto op = create_fractional_max_pool3d_output_typed_handle();
    return op.call(self, kernel_size, output_size, random_samples, output, indices);
}

// aten::fractional_max_pool3d.output(Tensor self, int[3] kernel_size, int[3] output_size, Tensor random_samples, *, Tensor(a!) output, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
::std::tuple<at::Tensor &,at::Tensor &> fractional_max_pool3d_output::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & random_samples, at::Tensor & output, at::Tensor & indices) {
    static auto op = create_fractional_max_pool3d_output_typed_handle();
    return op.redispatch(dispatchKeySet, self, kernel_size, output_size, random_samples, output, indices);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fractional_max_pool3d_backward_grad_input, name, "aten::fractional_max_pool3d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fractional_max_pool3d_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fractional_max_pool3d_backward_grad_input, schema_str, "fractional_max_pool3d_backward.grad_input(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] output_size, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::fractional_max_pool3d_backward.grad_input(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] output_size, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<fractional_max_pool3d_backward_grad_input::schema> create_fractional_max_pool3d_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fractional_max_pool3d_backward_grad_input::name, fractional_max_pool3d_backward_grad_input::overload_name)
      .typed<fractional_max_pool3d_backward_grad_input::schema>();
}

// aten::fractional_max_pool3d_backward.grad_input(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] output_size, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & fractional_max_pool3d_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & indices, at::Tensor & grad_input) {
    static auto op = create_fractional_max_pool3d_backward_grad_input_typed_handle();
    return op.call(grad_output, self, kernel_size, output_size, indices, grad_input);
}

// aten::fractional_max_pool3d_backward.grad_input(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] output_size, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & fractional_max_pool3d_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & indices, at::Tensor & grad_input) {
    static auto op = create_fractional_max_pool3d_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, kernel_size, output_size, indices, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_unpool2d, name, "aten::max_unpool2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_unpool2d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_unpool2d, schema_str, "max_unpool2d(Tensor self, Tensor indices, int[2] output_size) -> Tensor")

// aten::max_unpool2d(Tensor self, Tensor indices, int[2] output_size) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<max_unpool2d::schema> create_max_unpool2d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(max_unpool2d::name, max_unpool2d::overload_name)
      .typed<max_unpool2d::schema>();
}

// aten::max_unpool2d(Tensor self, Tensor indices, int[2] output_size) -> Tensor
at::Tensor max_unpool2d::call(const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size) {
    static auto op = create_max_unpool2d_typed_handle();
    return op.call(self, indices, output_size);
}

// aten::max_unpool2d(Tensor self, Tensor indices, int[2] output_size) -> Tensor
at::Tensor max_unpool2d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size) {
    static auto op = create_max_unpool2d_typed_handle();
    return op.redispatch(dispatchKeySet, self, indices, output_size);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_unpool3d, name, "aten::max_unpool3d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_unpool3d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(max_unpool3d, schema_str, "max_unpool3d(Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding) -> Tensor")

// aten::max_unpool3d(Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<max_unpool3d::schema> create_max_unpool3d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(max_unpool3d::name, max_unpool3d::overload_name)
      .typed<max_unpool3d::schema>();
}

// aten::max_unpool3d(Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding) -> Tensor
at::Tensor max_unpool3d::call(const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::IntArrayRef stride, at::IntArrayRef padding) {
    static auto op = create_max_unpool3d_typed_handle();
    return op.call(self, indices, output_size, stride, padding);
}

// aten::max_unpool3d(Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding) -> Tensor
at::Tensor max_unpool3d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::IntArrayRef stride, at::IntArrayRef padding) {
    static auto op = create_max_unpool3d_typed_handle();
    return op.redispatch(dispatchKeySet, self, indices, output_size, stride, padding);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reflection_pad1d_out, name, "aten::reflection_pad1d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reflection_pad1d_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reflection_pad1d_out, schema_str, "reflection_pad1d.out(Tensor self, int[2] padding, *, Tensor(a!) out) -> Tensor(a!)")

// aten::reflection_pad1d.out(Tensor self, int[2] padding, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<reflection_pad1d_out::schema> create_reflection_pad1d_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(reflection_pad1d_out::name, reflection_pad1d_out::overload_name)
      .typed<reflection_pad1d_out::schema>();
}

// aten::reflection_pad1d.out(Tensor self, int[2] padding, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & reflection_pad1d_out::call(const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out) {
    static auto op = create_reflection_pad1d_out_typed_handle();
    return op.call(self, padding, out);
}

// aten::reflection_pad1d.out(Tensor self, int[2] padding, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & reflection_pad1d_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out) {
    static auto op = create_reflection_pad1d_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, padding, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reflection_pad3d_backward, name, "aten::reflection_pad3d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reflection_pad3d_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(reflection_pad3d_backward, schema_str, "reflection_pad3d_backward(Tensor grad_output, Tensor self, int[6] padding) -> Tensor")

// aten::reflection_pad3d_backward(Tensor grad_output, Tensor self, int[6] padding) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<reflection_pad3d_backward::schema> create_reflection_pad3d_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(reflection_pad3d_backward::name, reflection_pad3d_backward::overload_name)
      .typed<reflection_pad3d_backward::schema>();
}

// aten::reflection_pad3d_backward(Tensor grad_output, Tensor self, int[6] padding) -> Tensor
at::Tensor reflection_pad3d_backward::call(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding) {
    static auto op = create_reflection_pad3d_backward_typed_handle();
    return op.call(grad_output, self, padding);
}

// aten::reflection_pad3d_backward(Tensor grad_output, Tensor self, int[6] padding) -> Tensor
at::Tensor reflection_pad3d_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding) {
    static auto op = create_reflection_pad3d_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, padding);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(replication_pad2d_backward, name, "aten::replication_pad2d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(replication_pad2d_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(replication_pad2d_backward, schema_str, "replication_pad2d_backward(Tensor grad_output, Tensor self, int[4] padding) -> Tensor")

// aten::replication_pad2d_backward(Tensor grad_output, Tensor self, int[4] padding) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<replication_pad2d_backward::schema> create_replication_pad2d_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(replication_pad2d_backward::name, replication_pad2d_backward::overload_name)
      .typed<replication_pad2d_backward::schema>();
}

// aten::replication_pad2d_backward(Tensor grad_output, Tensor self, int[4] padding) -> Tensor
at::Tensor replication_pad2d_backward::call(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding) {
    static auto op = create_replication_pad2d_backward_typed_handle();
    return op.call(grad_output, self, padding);
}

// aten::replication_pad2d_backward(Tensor grad_output, Tensor self, int[4] padding) -> Tensor
at::Tensor replication_pad2d_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding) {
    static auto op = create_replication_pad2d_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, padding);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(replication_pad3d, name, "aten::replication_pad3d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(replication_pad3d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(replication_pad3d, schema_str, "replication_pad3d(Tensor self, int[6] padding) -> Tensor")

// aten::replication_pad3d(Tensor self, int[6] padding) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<replication_pad3d::schema> create_replication_pad3d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(replication_pad3d::name, replication_pad3d::overload_name)
      .typed<replication_pad3d::schema>();
}

// aten::replication_pad3d(Tensor self, int[6] padding) -> Tensor
at::Tensor replication_pad3d::call(const at::Tensor & self, at::IntArrayRef padding) {
    static auto op = create_replication_pad3d_typed_handle();
    return op.call(self, padding);
}

// aten::replication_pad3d(Tensor self, int[6] padding) -> Tensor
at::Tensor replication_pad3d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef padding) {
    static auto op = create_replication_pad3d_typed_handle();
    return op.redispatch(dispatchKeySet, self, padding);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_linear1d, name, "aten::upsample_linear1d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_linear1d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_linear1d, schema_str, "upsample_linear1d(Tensor self, int[1] output_size, bool align_corners, float? scales=None) -> Tensor")

// aten::upsample_linear1d(Tensor self, int[1] output_size, bool align_corners, float? scales=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<upsample_linear1d::schema> create_upsample_linear1d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_linear1d::name, upsample_linear1d::overload_name)
      .typed<upsample_linear1d::schema>();
}

// aten::upsample_linear1d(Tensor self, int[1] output_size, bool align_corners, float? scales=None) -> Tensor
at::Tensor upsample_linear1d::call(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales) {
    static auto op = create_upsample_linear1d_typed_handle();
    return op.call(self, output_size, align_corners, scales);
}

// aten::upsample_linear1d(Tensor self, int[1] output_size, bool align_corners, float? scales=None) -> Tensor
at::Tensor upsample_linear1d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales) {
    static auto op = create_upsample_linear1d_typed_handle();
    return op.redispatch(dispatchKeySet, self, output_size, align_corners, scales);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_bilinear2d, name, "aten::upsample_bilinear2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_bilinear2d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_bilinear2d, schema_str, "upsample_bilinear2d(Tensor self, int[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor")

// aten::upsample_bilinear2d(Tensor self, int[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<upsample_bilinear2d::schema> create_upsample_bilinear2d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_bilinear2d::name, upsample_bilinear2d::overload_name)
      .typed<upsample_bilinear2d::schema>();
}

// aten::upsample_bilinear2d(Tensor self, int[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor
at::Tensor upsample_bilinear2d::call(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
    static auto op = create_upsample_bilinear2d_typed_handle();
    return op.call(self, output_size, align_corners, scales_h, scales_w);
}

// aten::upsample_bilinear2d(Tensor self, int[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor
at::Tensor upsample_bilinear2d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
    static auto op = create_upsample_bilinear2d_typed_handle();
    return op.redispatch(dispatchKeySet, self, output_size, align_corners, scales_h, scales_w);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_bilinear2d_backward_grad_input, name, "aten::upsample_bilinear2d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_bilinear2d_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_bilinear2d_backward_grad_input, schema_str, "upsample_bilinear2d_backward.grad_input(Tensor grad_output, int[2] output_size, int[4] input_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::upsample_bilinear2d_backward.grad_input(Tensor grad_output, int[2] output_size, int[4] input_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<upsample_bilinear2d_backward_grad_input::schema> create_upsample_bilinear2d_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_bilinear2d_backward_grad_input::name, upsample_bilinear2d_backward_grad_input::overload_name)
      .typed<upsample_bilinear2d_backward_grad_input::schema>();
}

// aten::upsample_bilinear2d_backward.grad_input(Tensor grad_output, int[2] output_size, int[4] input_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & upsample_bilinear2d_backward_grad_input::call(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input) {
    static auto op = create_upsample_bilinear2d_backward_grad_input_typed_handle();
    return op.call(grad_output, output_size, input_size, align_corners, scales_h, scales_w, grad_input);
}

// aten::upsample_bilinear2d_backward.grad_input(Tensor grad_output, int[2] output_size, int[4] input_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & upsample_bilinear2d_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input) {
    static auto op = create_upsample_bilinear2d_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, output_size, input_size, align_corners, scales_h, scales_w, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_bicubic2d, name, "aten::upsample_bicubic2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_bicubic2d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_bicubic2d, schema_str, "upsample_bicubic2d(Tensor self, int[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor")

// aten::upsample_bicubic2d(Tensor self, int[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<upsample_bicubic2d::schema> create_upsample_bicubic2d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_bicubic2d::name, upsample_bicubic2d::overload_name)
      .typed<upsample_bicubic2d::schema>();
}

// aten::upsample_bicubic2d(Tensor self, int[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor
at::Tensor upsample_bicubic2d::call(const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
    static auto op = create_upsample_bicubic2d_typed_handle();
    return op.call(self, output_size, align_corners, scales_h, scales_w);
}

// aten::upsample_bicubic2d(Tensor self, int[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor
at::Tensor upsample_bicubic2d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
    static auto op = create_upsample_bicubic2d_typed_handle();
    return op.redispatch(dispatchKeySet, self, output_size, align_corners, scales_h, scales_w);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_bicubic2d_backward, name, "aten::upsample_bicubic2d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_bicubic2d_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_bicubic2d_backward, schema_str, "upsample_bicubic2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor")

// aten::upsample_bicubic2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<upsample_bicubic2d_backward::schema> create_upsample_bicubic2d_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_bicubic2d_backward::name, upsample_bicubic2d_backward::overload_name)
      .typed<upsample_bicubic2d_backward::schema>();
}

// aten::upsample_bicubic2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor
at::Tensor upsample_bicubic2d_backward::call(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
    static auto op = create_upsample_bicubic2d_backward_typed_handle();
    return op.call(grad_output, output_size, input_size, align_corners, scales_h, scales_w);
}

// aten::upsample_bicubic2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor
at::Tensor upsample_bicubic2d_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
    static auto op = create_upsample_bicubic2d_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_trilinear3d_backward, name, "aten::upsample_trilinear3d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_trilinear3d_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_trilinear3d_backward, schema_str, "upsample_trilinear3d_backward(Tensor grad_output, int[3] output_size, int[5] input_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor")

// aten::upsample_trilinear3d_backward(Tensor grad_output, int[3] output_size, int[5] input_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<upsample_trilinear3d_backward::schema> create_upsample_trilinear3d_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_trilinear3d_backward::name, upsample_trilinear3d_backward::overload_name)
      .typed<upsample_trilinear3d_backward::schema>();
}

// aten::upsample_trilinear3d_backward(Tensor grad_output, int[3] output_size, int[5] input_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor
at::Tensor upsample_trilinear3d_backward::call(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
    static auto op = create_upsample_trilinear3d_backward_typed_handle();
    return op.call(grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w);
}

// aten::upsample_trilinear3d_backward(Tensor grad_output, int[3] output_size, int[5] input_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor
at::Tensor upsample_trilinear3d_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
    static auto op = create_upsample_trilinear3d_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest1d_backward_grad_input, name, "aten::upsample_nearest1d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest1d_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest1d_backward_grad_input, schema_str, "upsample_nearest1d_backward.grad_input(Tensor grad_output, int[1] output_size, int[3] input_size, float? scales=None, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::upsample_nearest1d_backward.grad_input(Tensor grad_output, int[1] output_size, int[3] input_size, float? scales=None, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<upsample_nearest1d_backward_grad_input::schema> create_upsample_nearest1d_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_nearest1d_backward_grad_input::name, upsample_nearest1d_backward_grad_input::overload_name)
      .typed<upsample_nearest1d_backward_grad_input::schema>();
}

// aten::upsample_nearest1d_backward.grad_input(Tensor grad_output, int[1] output_size, int[3] input_size, float? scales=None, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & upsample_nearest1d_backward_grad_input::call(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales, at::Tensor & grad_input) {
    static auto op = create_upsample_nearest1d_backward_grad_input_typed_handle();
    return op.call(grad_output, output_size, input_size, scales, grad_input);
}

// aten::upsample_nearest1d_backward.grad_input(Tensor grad_output, int[1] output_size, int[3] input_size, float? scales=None, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & upsample_nearest1d_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales, at::Tensor & grad_input) {
    static auto op = create_upsample_nearest1d_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, output_size, input_size, scales, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest2d_out, name, "aten::upsample_nearest2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest2d_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest2d_out, schema_str, "upsample_nearest2d.out(Tensor self, int[2] output_size, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::upsample_nearest2d.out(Tensor self, int[2] output_size, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<upsample_nearest2d_out::schema> create_upsample_nearest2d_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_nearest2d_out::name, upsample_nearest2d_out::overload_name)
      .typed<upsample_nearest2d_out::schema>();
}

// aten::upsample_nearest2d.out(Tensor self, int[2] output_size, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & upsample_nearest2d_out::call(const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out) {
    static auto op = create_upsample_nearest2d_out_typed_handle();
    return op.call(self, output_size, scales_h, scales_w, out);
}

// aten::upsample_nearest2d.out(Tensor self, int[2] output_size, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & upsample_nearest2d_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out) {
    static auto op = create_upsample_nearest2d_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, output_size, scales_h, scales_w, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest2d, name, "aten::upsample_nearest2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest2d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest2d, schema_str, "upsample_nearest2d(Tensor self, int[2] output_size, float? scales_h=None, float? scales_w=None) -> Tensor")

// aten::upsample_nearest2d(Tensor self, int[2] output_size, float? scales_h=None, float? scales_w=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<upsample_nearest2d::schema> create_upsample_nearest2d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_nearest2d::name, upsample_nearest2d::overload_name)
      .typed<upsample_nearest2d::schema>();
}

// aten::upsample_nearest2d(Tensor self, int[2] output_size, float? scales_h=None, float? scales_w=None) -> Tensor
at::Tensor upsample_nearest2d::call(const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w) {
    static auto op = create_upsample_nearest2d_typed_handle();
    return op.call(self, output_size, scales_h, scales_w);
}

// aten::upsample_nearest2d(Tensor self, int[2] output_size, float? scales_h=None, float? scales_w=None) -> Tensor
at::Tensor upsample_nearest2d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w) {
    static auto op = create_upsample_nearest2d_typed_handle();
    return op.redispatch(dispatchKeySet, self, output_size, scales_h, scales_w);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest2d_backward_grad_input, name, "aten::upsample_nearest2d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest2d_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest2d_backward_grad_input, schema_str, "upsample_nearest2d_backward.grad_input(Tensor grad_output, int[2] output_size, int[4] input_size, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::upsample_nearest2d_backward.grad_input(Tensor grad_output, int[2] output_size, int[4] input_size, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<upsample_nearest2d_backward_grad_input::schema> create_upsample_nearest2d_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_nearest2d_backward_grad_input::name, upsample_nearest2d_backward_grad_input::overload_name)
      .typed<upsample_nearest2d_backward_grad_input::schema>();
}

// aten::upsample_nearest2d_backward.grad_input(Tensor grad_output, int[2] output_size, int[4] input_size, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & upsample_nearest2d_backward_grad_input::call(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input) {
    static auto op = create_upsample_nearest2d_backward_grad_input_typed_handle();
    return op.call(grad_output, output_size, input_size, scales_h, scales_w, grad_input);
}

// aten::upsample_nearest2d_backward.grad_input(Tensor grad_output, int[2] output_size, int[4] input_size, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & upsample_nearest2d_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input) {
    static auto op = create_upsample_nearest2d_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, output_size, input_size, scales_h, scales_w, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest3d_out, name, "aten::upsample_nearest3d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest3d_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest3d_out, schema_str, "upsample_nearest3d.out(Tensor self, int[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::upsample_nearest3d.out(Tensor self, int[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<upsample_nearest3d_out::schema> create_upsample_nearest3d_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_nearest3d_out::name, upsample_nearest3d_out::overload_name)
      .typed<upsample_nearest3d_out::schema>();
}

// aten::upsample_nearest3d.out(Tensor self, int[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & upsample_nearest3d_out::call(const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out) {
    static auto op = create_upsample_nearest3d_out_typed_handle();
    return op.call(self, output_size, scales_d, scales_h, scales_w, out);
}

// aten::upsample_nearest3d.out(Tensor self, int[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & upsample_nearest3d_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out) {
    static auto op = create_upsample_nearest3d_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, output_size, scales_d, scales_h, scales_w, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest3d_backward, name, "aten::upsample_nearest3d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest3d_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(upsample_nearest3d_backward, schema_str, "upsample_nearest3d_backward(Tensor grad_output, int[3] output_size, int[5] input_size, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor")

// aten::upsample_nearest3d_backward(Tensor grad_output, int[3] output_size, int[5] input_size, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<upsample_nearest3d_backward::schema> create_upsample_nearest3d_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(upsample_nearest3d_backward::name, upsample_nearest3d_backward::overload_name)
      .typed<upsample_nearest3d_backward::schema>();
}

// aten::upsample_nearest3d_backward(Tensor grad_output, int[3] output_size, int[5] input_size, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor
at::Tensor upsample_nearest3d_backward::call(const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
    static auto op = create_upsample_nearest3d_backward_typed_handle();
    return op.call(grad_output, output_size, input_size, scales_d, scales_h, scales_w);
}

// aten::upsample_nearest3d_backward(Tensor grad_output, int[3] output_size, int[5] input_size, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor
at::Tensor upsample_nearest3d_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
    static auto op = create_upsample_nearest3d_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, output_size, input_size, scales_d, scales_h, scales_w);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logit_backward, name, "aten::logit_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logit_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(logit_backward, schema_str, "logit_backward(Tensor grad_output, Tensor self, float? eps=None) -> Tensor")

// aten::logit_backward(Tensor grad_output, Tensor self, float? eps=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<logit_backward::schema> create_logit_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(logit_backward::name, logit_backward::overload_name)
      .typed<logit_backward::schema>();
}

// aten::logit_backward(Tensor grad_output, Tensor self, float? eps=None) -> Tensor
at::Tensor logit_backward::call(const at::Tensor & grad_output, const at::Tensor & self, c10::optional<double> eps) {
    static auto op = create_logit_backward_typed_handle();
    return op.call(grad_output, self, eps);
}

// aten::logit_backward(Tensor grad_output, Tensor self, float? eps=None) -> Tensor
at::Tensor logit_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, c10::optional<double> eps) {
    static auto op = create_logit_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, eps);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tanh_backward_grad_input, name, "aten::tanh_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tanh_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(tanh_backward_grad_input, schema_str, "tanh_backward.grad_input(Tensor grad_output, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::tanh_backward.grad_input(Tensor grad_output, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<tanh_backward_grad_input::schema> create_tanh_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(tanh_backward_grad_input::name, tanh_backward_grad_input::overload_name)
      .typed<tanh_backward_grad_input::schema>();
}

// aten::tanh_backward.grad_input(Tensor grad_output, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & tanh_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & output, at::Tensor & grad_input) {
    static auto op = create_tanh_backward_grad_input_typed_handle();
    return op.call(grad_output, output, grad_input);
}

// aten::tanh_backward.grad_input(Tensor grad_output, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & tanh_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & output, at::Tensor & grad_input) {
    static auto op = create_tanh_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, output, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv_transpose2d, name, "aten::slow_conv_transpose2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv_transpose2d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv_transpose2d, schema_str, "slow_conv_transpose2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] output_padding=0, int[2] dilation=1) -> Tensor")

// aten::slow_conv_transpose2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] output_padding=0, int[2] dilation=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<slow_conv_transpose2d::schema> create_slow_conv_transpose2d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(slow_conv_transpose2d::name, slow_conv_transpose2d::overload_name)
      .typed<slow_conv_transpose2d::schema>();
}

// aten::slow_conv_transpose2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] output_padding=0, int[2] dilation=1) -> Tensor
at::Tensor slow_conv_transpose2d::call(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation) {
    static auto op = create_slow_conv_transpose2d_typed_handle();
    return op.call(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}

// aten::slow_conv_transpose2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] output_padding=0, int[2] dilation=1) -> Tensor
at::Tensor slow_conv_transpose2d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation) {
    static auto op = create_slow_conv_transpose2d_typed_handle();
    return op.redispatch(dispatchKeySet, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(thnn_conv2d_forward_output, name, "aten::thnn_conv2d_forward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(thnn_conv2d_forward_output, overload_name, "output")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(thnn_conv2d_forward_output, schema_str, "thnn_conv2d_forward.output(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding, *, Tensor(a!) output, Tensor(b!) finput, Tensor(c!) fgrad_input) -> (Tensor(a!), Tensor(b!), Tensor(c!))")

// aten::thnn_conv2d_forward.output(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding, *, Tensor(a!) output, Tensor(b!) finput, Tensor(c!) fgrad_input) -> (Tensor(a!), Tensor(b!), Tensor(c!))
static C10_NOINLINE c10::TypedOperatorHandle<thnn_conv2d_forward_output::schema> create_thnn_conv2d_forward_output_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(thnn_conv2d_forward_output::name, thnn_conv2d_forward_output::overload_name)
      .typed<thnn_conv2d_forward_output::schema>();
}

// aten::thnn_conv2d_forward.output(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding, *, Tensor(a!) output, Tensor(b!) finput, Tensor(c!) fgrad_input) -> (Tensor(a!), Tensor(b!), Tensor(c!))
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> thnn_conv2d_forward_output::call(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & output, at::Tensor & finput, at::Tensor & fgrad_input) {
    static auto op = create_thnn_conv2d_forward_output_typed_handle();
    return op.call(self, weight, kernel_size, bias, stride, padding, output, finput, fgrad_input);
}

// aten::thnn_conv2d_forward.output(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding, *, Tensor(a!) output, Tensor(b!) finput, Tensor(c!) fgrad_input) -> (Tensor(a!), Tensor(b!), Tensor(c!))
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> thnn_conv2d_forward_output::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & output, at::Tensor & finput, at::Tensor & fgrad_input) {
    static auto op = create_thnn_conv2d_forward_output_typed_handle();
    return op.redispatch(dispatchKeySet, self, weight, kernel_size, bias, stride, padding, output, finput, fgrad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(thnn_conv2d_backward_output_mask, name, "aten::thnn_conv2d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(thnn_conv2d_backward_output_mask, overload_name, "output_mask")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(thnn_conv2d_backward_output_mask, schema_str, "thnn_conv2d_backward.output_mask(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, Tensor finput, Tensor fgrad_input, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)")

// aten::thnn_conv2d_backward.output_mask(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, Tensor finput, Tensor fgrad_input, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)
static C10_NOINLINE c10::TypedOperatorHandle<thnn_conv2d_backward_output_mask::schema> create_thnn_conv2d_backward_output_mask_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(thnn_conv2d_backward_output_mask::name, thnn_conv2d_backward_output_mask::overload_name)
      .typed<thnn_conv2d_backward_output_mask::schema>();
}

// aten::thnn_conv2d_backward.output_mask(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, Tensor finput, Tensor fgrad_input, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> thnn_conv2d_backward_output_mask::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, const at::Tensor & finput, const at::Tensor & fgrad_input, ::std::array<bool,3> output_mask) {
    static auto op = create_thnn_conv2d_backward_output_mask_typed_handle();
    return op.call(grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, output_mask);
}

// aten::thnn_conv2d_backward.output_mask(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, Tensor finput, Tensor fgrad_input, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> thnn_conv2d_backward_output_mask::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, const at::Tensor & finput, const at::Tensor & fgrad_input, ::std::array<bool,3> output_mask) {
    static auto op = create_thnn_conv2d_backward_output_mask_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, output_mask);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv_depthwise3d, name, "aten::conv_depthwise3d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv_depthwise3d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv_depthwise3d, schema_str, "conv_depthwise3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias, int[3] stride, int[3] padding, int[3] dilation) -> Tensor")

// aten::conv_depthwise3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias, int[3] stride, int[3] padding, int[3] dilation) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<conv_depthwise3d::schema> create_conv_depthwise3d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(conv_depthwise3d::name, conv_depthwise3d::overload_name)
      .typed<conv_depthwise3d::schema>();
}

// aten::conv_depthwise3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias, int[3] stride, int[3] padding, int[3] dilation) -> Tensor
at::Tensor conv_depthwise3d::call(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation) {
    static auto op = create_conv_depthwise3d_typed_handle();
    return op.call(self, weight, kernel_size, bias, stride, padding, dilation);
}

// aten::conv_depthwise3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias, int[3] stride, int[3] padding, int[3] dilation) -> Tensor
at::Tensor conv_depthwise3d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation) {
    static auto op = create_conv_depthwise3d_typed_handle();
    return op.redispatch(dispatchKeySet, self, weight, kernel_size, bias, stride, padding, dilation);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv_depthwise3d_backward_grad_input, name, "aten::conv_depthwise3d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv_depthwise3d_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(conv_depthwise3d_backward_grad_input, schema_str, "conv_depthwise3d_backward.grad_input(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, *, Tensor(a!) grad_input, Tensor(b!) grad_weight, Tensor(c!) grad_bias) -> (Tensor(a!), Tensor(b!), Tensor(c!))")

// aten::conv_depthwise3d_backward.grad_input(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, *, Tensor(a!) grad_input, Tensor(b!) grad_weight, Tensor(c!) grad_bias) -> (Tensor(a!), Tensor(b!), Tensor(c!))
static C10_NOINLINE c10::TypedOperatorHandle<conv_depthwise3d_backward_grad_input::schema> create_conv_depthwise3d_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(conv_depthwise3d_backward_grad_input::name, conv_depthwise3d_backward_grad_input::overload_name)
      .typed<conv_depthwise3d_backward_grad_input::schema>();
}

// aten::conv_depthwise3d_backward.grad_input(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, *, Tensor(a!) grad_input, Tensor(b!) grad_weight, Tensor(c!) grad_bias) -> (Tensor(a!), Tensor(b!), Tensor(c!))
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> conv_depthwise3d_backward_grad_input::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias) {
    static auto op = create_conv_depthwise3d_backward_grad_input_typed_handle();
    return op.call(grad_output, self, weight, kernel_size, stride, padding, dilation, grad_input, grad_weight, grad_bias);
}

// aten::conv_depthwise3d_backward.grad_input(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, *, Tensor(a!) grad_input, Tensor(b!) grad_weight, Tensor(c!) grad_bias) -> (Tensor(a!), Tensor(b!), Tensor(c!))
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> conv_depthwise3d_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias) {
    static auto op = create_conv_depthwise3d_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, weight, kernel_size, stride, padding, dilation, grad_input, grad_weight, grad_bias);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv3d_forward_output, name, "aten::slow_conv3d_forward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv3d_forward_output, overload_name, "output")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv3d_forward_output, schema_str, "slow_conv3d_forward.output(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias, int[3] stride, int[3] padding, *, Tensor(a!) output, Tensor(b!) finput, Tensor(c!) fgrad_input) -> (Tensor(a!), Tensor(b!), Tensor(c!))")

// aten::slow_conv3d_forward.output(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias, int[3] stride, int[3] padding, *, Tensor(a!) output, Tensor(b!) finput, Tensor(c!) fgrad_input) -> (Tensor(a!), Tensor(b!), Tensor(c!))
static C10_NOINLINE c10::TypedOperatorHandle<slow_conv3d_forward_output::schema> create_slow_conv3d_forward_output_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(slow_conv3d_forward_output::name, slow_conv3d_forward_output::overload_name)
      .typed<slow_conv3d_forward_output::schema>();
}

// aten::slow_conv3d_forward.output(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias, int[3] stride, int[3] padding, *, Tensor(a!) output, Tensor(b!) finput, Tensor(c!) fgrad_input) -> (Tensor(a!), Tensor(b!), Tensor(c!))
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> slow_conv3d_forward_output::call(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & output, at::Tensor & finput, at::Tensor & fgrad_input) {
    static auto op = create_slow_conv3d_forward_output_typed_handle();
    return op.call(self, weight, kernel_size, bias, stride, padding, output, finput, fgrad_input);
}

// aten::slow_conv3d_forward.output(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias, int[3] stride, int[3] padding, *, Tensor(a!) output, Tensor(b!) finput, Tensor(c!) fgrad_input) -> (Tensor(a!), Tensor(b!), Tensor(c!))
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> slow_conv3d_forward_output::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & output, at::Tensor & finput, at::Tensor & fgrad_input) {
    static auto op = create_slow_conv3d_forward_output_typed_handle();
    return op.redispatch(dispatchKeySet, self, weight, kernel_size, bias, stride, padding, output, finput, fgrad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv_dilated2d, name, "aten::slow_conv_dilated2d")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv_dilated2d, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv_dilated2d, schema_str, "slow_conv_dilated2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1) -> Tensor")

// aten::slow_conv_dilated2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<slow_conv_dilated2d::schema> create_slow_conv_dilated2d_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(slow_conv_dilated2d::name, slow_conv_dilated2d::overload_name)
      .typed<slow_conv_dilated2d::schema>();
}

// aten::slow_conv_dilated2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1) -> Tensor
at::Tensor slow_conv_dilated2d::call(const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation) {
    static auto op = create_slow_conv_dilated2d_typed_handle();
    return op.call(self, weight, kernel_size, bias, stride, padding, dilation);
}

// aten::slow_conv_dilated2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1) -> Tensor
at::Tensor slow_conv_dilated2d::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation) {
    static auto op = create_slow_conv_dilated2d_typed_handle();
    return op.redispatch(dispatchKeySet, self, weight, kernel_size, bias, stride, padding, dilation);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv_dilated2d_backward, name, "aten::slow_conv_dilated2d_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv_dilated2d_backward, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(slow_conv_dilated2d_backward, schema_str, "slow_conv_dilated2d_backward(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)")

// aten::slow_conv_dilated2d_backward(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)
static C10_NOINLINE c10::TypedOperatorHandle<slow_conv_dilated2d_backward::schema> create_slow_conv_dilated2d_backward_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(slow_conv_dilated2d_backward::name, slow_conv_dilated2d_backward::overload_name)
      .typed<slow_conv_dilated2d_backward::schema>();
}

// aten::slow_conv_dilated2d_backward(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> slow_conv_dilated2d_backward::call(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, ::std::array<bool,3> output_mask) {
    static auto op = create_slow_conv_dilated2d_backward_typed_handle();
    return op.call(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
}

// aten::slow_conv_dilated2d_backward(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)
::std::tuple<at::Tensor,at::Tensor,at::Tensor> slow_conv_dilated2d_backward::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, ::std::array<bool,3> output_mask) {
    static auto op = create_slow_conv_dilated2d_backward_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(col2im, name, "aten::col2im")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(col2im, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(col2im, schema_str, "col2im(Tensor self, int[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor")

// aten::col2im(Tensor self, int[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<col2im::schema> create_col2im_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(col2im::name, col2im::overload_name)
      .typed<col2im::schema>();
}

// aten::col2im(Tensor self, int[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor
at::Tensor col2im::call(const at::Tensor & self, at::IntArrayRef output_size, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride) {
    static auto op = create_col2im_typed_handle();
    return op.call(self, output_size, kernel_size, dilation, padding, stride);
}

// aten::col2im(Tensor self, int[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor
at::Tensor col2im::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::IntArrayRef output_size, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride) {
    static auto op = create_col2im_typed_handle();
    return op.redispatch(dispatchKeySet, self, output_size, kernel_size, dilation, padding, stride);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(col2im_backward_grad_input, name, "aten::col2im_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(col2im_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(col2im_backward_grad_input, schema_str, "col2im_backward.grad_input(Tensor grad_output, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::col2im_backward.grad_input(Tensor grad_output, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<col2im_backward_grad_input::schema> create_col2im_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(col2im_backward_grad_input::name, col2im_backward_grad_input::overload_name)
      .typed<col2im_backward_grad_input::schema>();
}

// aten::col2im_backward.grad_input(Tensor grad_output, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & col2im_backward_grad_input::call(const at::Tensor & grad_output, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride, at::Tensor & grad_input) {
    static auto op = create_col2im_backward_grad_input_typed_handle();
    return op.call(grad_output, kernel_size, dilation, padding, stride, grad_input);
}

// aten::col2im_backward.grad_input(Tensor grad_output, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & col2im_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride, at::Tensor & grad_input) {
    static auto op = create_col2im_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, kernel_size, dilation, padding, stride, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(column_stack_out, name, "aten::column_stack")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(column_stack_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(column_stack_out, schema_str, "column_stack.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)")

// aten::column_stack.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<column_stack_out::schema> create_column_stack_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(column_stack_out::name, column_stack_out::overload_name)
      .typed<column_stack_out::schema>();
}

// aten::column_stack.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & column_stack_out::call(at::TensorList tensors, at::Tensor & out) {
    static auto op = create_column_stack_out_typed_handle();
    return op.call(tensors, out);
}

// aten::column_stack.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & column_stack_out::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList tensors, at::Tensor & out) {
    static auto op = create_column_stack_out_typed_handle();
    return op.redispatch(dispatchKeySet, tensors, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(im2col_backward_grad_input, name, "aten::im2col_backward")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(im2col_backward_grad_input, overload_name, "grad_input")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(im2col_backward_grad_input, schema_str, "im2col_backward.grad_input(Tensor grad_output, int[2] input_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) grad_input) -> Tensor(a!)")

// aten::im2col_backward.grad_input(Tensor grad_output, int[2] input_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) grad_input) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<im2col_backward_grad_input::schema> create_im2col_backward_grad_input_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(im2col_backward_grad_input::name, im2col_backward_grad_input::overload_name)
      .typed<im2col_backward_grad_input::schema>();
}

// aten::im2col_backward.grad_input(Tensor grad_output, int[2] input_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & im2col_backward_grad_input::call(const at::Tensor & grad_output, at::IntArrayRef input_size, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride, at::Tensor & grad_input) {
    static auto op = create_im2col_backward_grad_input_typed_handle();
    return op.call(grad_output, input_size, kernel_size, dilation, padding, stride, grad_input);
}

// aten::im2col_backward.grad_input(Tensor grad_output, int[2] input_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) grad_input) -> Tensor(a!)
at::Tensor & im2col_backward_grad_input::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & grad_output, at::IntArrayRef input_size, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride, at::Tensor & grad_input) {
    static auto op = create_im2col_backward_grad_input_typed_handle();
    return op.redispatch(dispatchKeySet, grad_output, input_size, kernel_size, dilation, padding, stride, grad_input);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isfinite, name, "aten::isfinite")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isfinite, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isfinite, schema_str, "isfinite(Tensor self) -> Tensor")

// aten::isfinite(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<isfinite::schema> create_isfinite_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(isfinite::name, isfinite::overload_name)
      .typed<isfinite::schema>();
}

// aten::isfinite(Tensor self) -> Tensor
at::Tensor isfinite::call(const at::Tensor & self) {
    static auto op = create_isfinite_typed_handle();
    return op.call(self);
}

// aten::isfinite(Tensor self) -> Tensor
at::Tensor isfinite::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_isfinite_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(record_stream, name, "aten::record_stream")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(record_stream, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(record_stream, schema_str, "record_stream(Tensor(a!) self, Stream s) -> ()")

// aten::record_stream(Tensor(a!) self, Stream s) -> ()
static C10_NOINLINE c10::TypedOperatorHandle<record_stream::schema> create_record_stream_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(record_stream::name, record_stream::overload_name)
      .typed<record_stream::schema>();
}

// aten::record_stream(Tensor(a!) self, Stream s) -> ()
void record_stream::call(at::Tensor & self, at::Stream s) {
    static auto op = create_record_stream_typed_handle();
    return op.call(self, s);
}

// aten::record_stream(Tensor(a!) self, Stream s) -> ()
void record_stream::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, at::Stream s) {
    static auto op = create_record_stream_typed_handle();
    return op.redispatch(dispatchKeySet, self, s);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isposinf, name, "aten::isposinf")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isposinf, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isposinf, schema_str, "isposinf(Tensor self) -> Tensor")

// aten::isposinf(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<isposinf::schema> create_isposinf_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(isposinf::name, isposinf::overload_name)
      .typed<isposinf::schema>();
}

// aten::isposinf(Tensor self) -> Tensor
at::Tensor isposinf::call(const at::Tensor & self) {
    static auto op = create_isposinf_typed_handle();
    return op.call(self);
}

// aten::isposinf(Tensor self) -> Tensor
at::Tensor isposinf::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_isposinf_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isposinf_out, name, "aten::isposinf")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isposinf_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(isposinf_out, schema_str, "isposinf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::isposinf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<isposinf_out::schema> create_isposinf_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(isposinf_out::name, isposinf_out::overload_name)
      .typed<isposinf_out::schema>();
}

// aten::isposinf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & isposinf_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_isposinf_out_typed_handle();
    return op.call(self, out);
}

// aten::isposinf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & isposinf_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_isposinf_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_ndtri_out, name, "aten::special_ndtri")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_ndtri_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_ndtri_out, schema_str, "special_ndtri.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::special_ndtri.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<special_ndtri_out::schema> create_special_ndtri_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_ndtri_out::name, special_ndtri_out::overload_name)
      .typed<special_ndtri_out::schema>();
}

// aten::special_ndtri.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_ndtri_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_ndtri_out_typed_handle();
    return op.call(self, out);
}

// aten::special_ndtri.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_ndtri_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_ndtri_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_expm1, name, "aten::special_expm1")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_expm1, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_expm1, schema_str, "special_expm1(Tensor self) -> Tensor")

// aten::special_expm1(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<special_expm1::schema> create_special_expm1_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_expm1::name, special_expm1::overload_name)
      .typed<special_expm1::schema>();
}

// aten::special_expm1(Tensor self) -> Tensor
at::Tensor special_expm1::call(const at::Tensor & self) {
    static auto op = create_special_expm1_typed_handle();
    return op.call(self);
}

// aten::special_expm1(Tensor self) -> Tensor
at::Tensor special_expm1::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_special_expm1_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_exp2, name, "aten::special_exp2")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_exp2, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_exp2, schema_str, "special_exp2(Tensor self) -> Tensor")

// aten::special_exp2(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<special_exp2::schema> create_special_exp2_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_exp2::name, special_exp2::overload_name)
      .typed<special_exp2::schema>();
}

// aten::special_exp2(Tensor self) -> Tensor
at::Tensor special_exp2::call(const at::Tensor & self) {
    static auto op = create_special_exp2_typed_handle();
    return op.call(self);
}

// aten::special_exp2(Tensor self) -> Tensor
at::Tensor special_exp2::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_special_exp2_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_gammaln, name, "aten::special_gammaln")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_gammaln, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_gammaln, schema_str, "special_gammaln(Tensor self) -> Tensor")

// aten::special_gammaln(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<special_gammaln::schema> create_special_gammaln_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_gammaln::name, special_gammaln::overload_name)
      .typed<special_gammaln::schema>();
}

// aten::special_gammaln(Tensor self) -> Tensor
at::Tensor special_gammaln::call(const at::Tensor & self) {
    static auto op = create_special_gammaln_typed_handle();
    return op.call(self);
}

// aten::special_gammaln(Tensor self) -> Tensor
at::Tensor special_gammaln::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_special_gammaln_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_gammaln_out, name, "aten::special_gammaln")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_gammaln_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_gammaln_out, schema_str, "special_gammaln.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::special_gammaln.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<special_gammaln_out::schema> create_special_gammaln_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_gammaln_out::name, special_gammaln_out::overload_name)
      .typed<special_gammaln_out::schema>();
}

// aten::special_gammaln.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_gammaln_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_gammaln_out_typed_handle();
    return op.call(self, out);
}

// aten::special_gammaln.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_gammaln_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_gammaln_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_erfinv, name, "aten::special_erfinv")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_erfinv, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_erfinv, schema_str, "special_erfinv(Tensor self) -> Tensor")

// aten::special_erfinv(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<special_erfinv::schema> create_special_erfinv_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_erfinv::name, special_erfinv::overload_name)
      .typed<special_erfinv::schema>();
}

// aten::special_erfinv(Tensor self) -> Tensor
at::Tensor special_erfinv::call(const at::Tensor & self) {
    static auto op = create_special_erfinv_typed_handle();
    return op.call(self);
}

// aten::special_erfinv(Tensor self) -> Tensor
at::Tensor special_erfinv::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_special_erfinv_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_ndtr_out, name, "aten::special_ndtr")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_ndtr_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_ndtr_out, schema_str, "special_ndtr.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::special_ndtr.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<special_ndtr_out::schema> create_special_ndtr_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_ndtr_out::name, special_ndtr_out::overload_name)
      .typed<special_ndtr_out::schema>();
}

// aten::special_ndtr.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_ndtr_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_ndtr_out_typed_handle();
    return op.call(self, out);
}

// aten::special_ndtr.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_ndtr_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_ndtr_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_xlog1py, name, "aten::special_xlog1py")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_xlog1py, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_xlog1py, schema_str, "special_xlog1py(Tensor self, Tensor other) -> Tensor")

// aten::special_xlog1py(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<special_xlog1py::schema> create_special_xlog1py_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_xlog1py::name, special_xlog1py::overload_name)
      .typed<special_xlog1py::schema>();
}

// aten::special_xlog1py(Tensor self, Tensor other) -> Tensor
at::Tensor special_xlog1py::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_special_xlog1py_typed_handle();
    return op.call(self, other);
}

// aten::special_xlog1py(Tensor self, Tensor other) -> Tensor
at::Tensor special_xlog1py::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_special_xlog1py_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_xlog1py_out, name, "aten::special_xlog1py")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_xlog1py_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_xlog1py_out, schema_str, "special_xlog1py.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::special_xlog1py.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<special_xlog1py_out::schema> create_special_xlog1py_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_xlog1py_out::name, special_xlog1py_out::overload_name)
      .typed<special_xlog1py_out::schema>();
}

// aten::special_xlog1py.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_xlog1py_out::call(const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_special_xlog1py_out_typed_handle();
    return op.call(self, other, out);
}

// aten::special_xlog1py.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_xlog1py_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
    static auto op = create_special_xlog1py_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_xlogy_other_scalar_out, name, "aten::special_xlogy")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_xlogy_other_scalar_out, overload_name, "other_scalar_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_xlogy_other_scalar_out, schema_str, "special_xlogy.other_scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)")

// aten::special_xlogy.other_scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<special_xlogy_other_scalar_out::schema> create_special_xlogy_other_scalar_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_xlogy_other_scalar_out::name, special_xlogy_other_scalar_out::overload_name)
      .typed<special_xlogy_other_scalar_out::schema>();
}

// aten::special_xlogy.other_scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_xlogy_other_scalar_out::call(const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_special_xlogy_other_scalar_out_typed_handle();
    return op.call(self, other, out);
}

// aten::special_xlogy.other_scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_xlogy_other_scalar_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
    static auto op = create_special_xlogy_other_scalar_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, other, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_i0, name, "aten::special_i0")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_i0, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_i0, schema_str, "special_i0(Tensor self) -> Tensor")

// aten::special_i0(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<special_i0::schema> create_special_i0_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_i0::name, special_i0::overload_name)
      .typed<special_i0::schema>();
}

// aten::special_i0(Tensor self) -> Tensor
at::Tensor special_i0::call(const at::Tensor & self) {
    static auto op = create_special_i0_typed_handle();
    return op.call(self);
}

// aten::special_i0(Tensor self) -> Tensor
at::Tensor special_i0::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_special_i0_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_i1e_out, name, "aten::special_i1e")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_i1e_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_i1e_out, schema_str, "special_i1e.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::special_i1e.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<special_i1e_out::schema> create_special_i1e_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_i1e_out::name, special_i1e_out::overload_name)
      .typed<special_i1e_out::schema>();
}

// aten::special_i1e.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_i1e_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_i1e_out_typed_handle();
    return op.call(self, out);
}

// aten::special_i1e.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_i1e_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_special_i1e_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_polygamma, name, "aten::special_polygamma")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_polygamma, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_polygamma, schema_str, "special_polygamma(int n, Tensor self) -> Tensor")

// aten::special_polygamma(int n, Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<special_polygamma::schema> create_special_polygamma_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_polygamma::name, special_polygamma::overload_name)
      .typed<special_polygamma::schema>();
}

// aten::special_polygamma(int n, Tensor self) -> Tensor
at::Tensor special_polygamma::call(int64_t n, const at::Tensor & self) {
    static auto op = create_special_polygamma_typed_handle();
    return op.call(n, self);
}

// aten::special_polygamma(int n, Tensor self) -> Tensor
at::Tensor special_polygamma::redispatch(c10::DispatchKeySet dispatchKeySet, int64_t n, const at::Tensor & self) {
    static auto op = create_special_polygamma_typed_handle();
    return op.redispatch(dispatchKeySet, n, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_log1p, name, "aten::special_log1p")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_log1p, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_log1p, schema_str, "special_log1p(Tensor self) -> Tensor")

// aten::special_log1p(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<special_log1p::schema> create_special_log1p_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_log1p::name, special_log1p::overload_name)
      .typed<special_log1p::schema>();
}

// aten::special_log1p(Tensor self) -> Tensor
at::Tensor special_log1p::call(const at::Tensor & self) {
    static auto op = create_special_log1p_typed_handle();
    return op.call(self);
}

// aten::special_log1p(Tensor self) -> Tensor
at::Tensor special_log1p::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_special_log1p_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_multigammaln_out, name, "aten::special_multigammaln")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_multigammaln_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(special_multigammaln_out, schema_str, "special_multigammaln.out(Tensor self, int p, *, Tensor(a!) out) -> Tensor(a!)")

// aten::special_multigammaln.out(Tensor self, int p, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<special_multigammaln_out::schema> create_special_multigammaln_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(special_multigammaln_out::name, special_multigammaln_out::overload_name)
      .typed<special_multigammaln_out::schema>();
}

// aten::special_multigammaln.out(Tensor self, int p, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_multigammaln_out::call(const at::Tensor & self, int64_t p, at::Tensor & out) {
    static auto op = create_special_multigammaln_out_typed_handle();
    return op.call(self, p, out);
}

// aten::special_multigammaln.out(Tensor self, int p, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & special_multigammaln_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t p, at::Tensor & out) {
    static auto op = create_special_multigammaln_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, p, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_rfft_out, name, "aten::fft_rfft")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_rfft_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_rfft_out, schema_str, "fft_rfft.out(Tensor self, int? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::fft_rfft.out(Tensor self, int? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<fft_rfft_out::schema> create_fft_rfft_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fft_rfft_out::name, fft_rfft_out::overload_name)
      .typed<fft_rfft_out::schema>();
}

// aten::fft_rfft.out(Tensor self, int? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fft_rfft_out::call(const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
    static auto op = create_fft_rfft_out_typed_handle();
    return op.call(self, n, dim, norm, out);
}

// aten::fft_rfft.out(Tensor self, int? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fft_rfft_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
    static auto op = create_fft_rfft_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, n, dim, norm, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_irfft, name, "aten::fft_irfft")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_irfft, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_irfft, schema_str, "fft_irfft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> Tensor")

// aten::fft_irfft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fft_irfft::schema> create_fft_irfft_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fft_irfft::name, fft_irfft::overload_name)
      .typed<fft_irfft::schema>();
}

// aten::fft_irfft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> Tensor
at::Tensor fft_irfft::call(const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm) {
    static auto op = create_fft_irfft_typed_handle();
    return op.call(self, n, dim, norm);
}

// aten::fft_irfft(Tensor self, int? n=None, int dim=-1, str? norm=None) -> Tensor
at::Tensor fft_irfft::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm) {
    static auto op = create_fft_irfft_typed_handle();
    return op.redispatch(dispatchKeySet, self, n, dim, norm);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_irfft_out, name, "aten::fft_irfft")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_irfft_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_irfft_out, schema_str, "fft_irfft.out(Tensor self, int? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::fft_irfft.out(Tensor self, int? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<fft_irfft_out::schema> create_fft_irfft_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fft_irfft_out::name, fft_irfft_out::overload_name)
      .typed<fft_irfft_out::schema>();
}

// aten::fft_irfft.out(Tensor self, int? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fft_irfft_out::call(const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
    static auto op = create_fft_irfft_out_typed_handle();
    return op.call(self, n, dim, norm, out);
}

// aten::fft_irfft.out(Tensor self, int? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fft_irfft_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
    static auto op = create_fft_irfft_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, n, dim, norm, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_ihfft_out, name, "aten::fft_ihfft")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_ihfft_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_ihfft_out, schema_str, "fft_ihfft.out(Tensor self, int? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::fft_ihfft.out(Tensor self, int? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<fft_ihfft_out::schema> create_fft_ihfft_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fft_ihfft_out::name, fft_ihfft_out::overload_name)
      .typed<fft_ihfft_out::schema>();
}

// aten::fft_ihfft.out(Tensor self, int? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fft_ihfft_out::call(const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
    static auto op = create_fft_ihfft_out_typed_handle();
    return op.call(self, n, dim, norm, out);
}

// aten::fft_ihfft.out(Tensor self, int? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fft_ihfft_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<int64_t> n, int64_t dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
    static auto op = create_fft_ihfft_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, n, dim, norm, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_ifft2, name, "aten::fft_ifft2")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_ifft2, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_ifft2, schema_str, "fft_ifft2(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None) -> Tensor")

// aten::fft_ifft2(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fft_ifft2::schema> create_fft_ifft2_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fft_ifft2::name, fft_ifft2::overload_name)
      .typed<fft_ifft2::schema>();
}

// aten::fft_ifft2(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None) -> Tensor
at::Tensor fft_ifft2::call(const at::Tensor & self, c10::optional<at::IntArrayRef> s, at::IntArrayRef dim, c10::optional<c10::string_view> norm) {
    static auto op = create_fft_ifft2_typed_handle();
    return op.call(self, s, dim, norm);
}

// aten::fft_ifft2(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None) -> Tensor
at::Tensor fft_ifft2::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> s, at::IntArrayRef dim, c10::optional<c10::string_view> norm) {
    static auto op = create_fft_ifft2_typed_handle();
    return op.redispatch(dispatchKeySet, self, s, dim, norm);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_rfft2_out, name, "aten::fft_rfft2")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_rfft2_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_rfft2_out, schema_str, "fft_rfft2.out(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::fft_rfft2.out(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<fft_rfft2_out::schema> create_fft_rfft2_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fft_rfft2_out::name, fft_rfft2_out::overload_name)
      .typed<fft_rfft2_out::schema>();
}

// aten::fft_rfft2.out(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fft_rfft2_out::call(const at::Tensor & self, c10::optional<at::IntArrayRef> s, at::IntArrayRef dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
    static auto op = create_fft_rfft2_out_typed_handle();
    return op.call(self, s, dim, norm, out);
}

// aten::fft_rfft2.out(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fft_rfft2_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> s, at::IntArrayRef dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
    static auto op = create_fft_rfft2_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, s, dim, norm, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_irfft2, name, "aten::fft_irfft2")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_irfft2, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_irfft2, schema_str, "fft_irfft2(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None) -> Tensor")

// aten::fft_irfft2(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fft_irfft2::schema> create_fft_irfft2_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fft_irfft2::name, fft_irfft2::overload_name)
      .typed<fft_irfft2::schema>();
}

// aten::fft_irfft2(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None) -> Tensor
at::Tensor fft_irfft2::call(const at::Tensor & self, c10::optional<at::IntArrayRef> s, at::IntArrayRef dim, c10::optional<c10::string_view> norm) {
    static auto op = create_fft_irfft2_typed_handle();
    return op.call(self, s, dim, norm);
}

// aten::fft_irfft2(Tensor self, int[1]? s=None, int[1] dim=[-2,-1], str? norm=None) -> Tensor
at::Tensor fft_irfft2::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> s, at::IntArrayRef dim, c10::optional<c10::string_view> norm) {
    static auto op = create_fft_irfft2_typed_handle();
    return op.redispatch(dispatchKeySet, self, s, dim, norm);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_rfftn, name, "aten::fft_rfftn")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_rfftn, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_rfftn, schema_str, "fft_rfftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor")

// aten::fft_rfftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<fft_rfftn::schema> create_fft_rfftn_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fft_rfftn::name, fft_rfftn::overload_name)
      .typed<fft_rfftn::schema>();
}

// aten::fft_rfftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor
at::Tensor fft_rfftn::call(const at::Tensor & self, c10::optional<at::IntArrayRef> s, c10::optional<at::IntArrayRef> dim, c10::optional<c10::string_view> norm) {
    static auto op = create_fft_rfftn_typed_handle();
    return op.call(self, s, dim, norm);
}

// aten::fft_rfftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor
at::Tensor fft_rfftn::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> s, c10::optional<at::IntArrayRef> dim, c10::optional<c10::string_view> norm) {
    static auto op = create_fft_rfftn_typed_handle();
    return op.redispatch(dispatchKeySet, self, s, dim, norm);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_irfftn_out, name, "aten::fft_irfftn")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_irfftn_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(fft_irfftn_out, schema_str, "fft_irfftn.out(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::fft_irfftn.out(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<fft_irfftn_out::schema> create_fft_irfftn_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(fft_irfftn_out::name, fft_irfftn_out::overload_name)
      .typed<fft_irfftn_out::schema>();
}

// aten::fft_irfftn.out(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fft_irfftn_out::call(const at::Tensor & self, c10::optional<at::IntArrayRef> s, c10::optional<at::IntArrayRef> dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
    static auto op = create_fft_irfftn_out_typed_handle();
    return op.call(self, s, dim, norm, out);
}

// aten::fft_irfftn.out(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & fft_irfftn_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<at::IntArrayRef> s, c10::optional<at::IntArrayRef> dim, c10::optional<c10::string_view> norm, at::Tensor & out) {
    static auto op = create_fft_irfftn_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, s, dim, norm, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_cholesky_ex_L, name, "aten::linalg_cholesky_ex")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_cholesky_ex_L, overload_name, "L")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_cholesky_ex_L, schema_str, "linalg_cholesky_ex.L(Tensor self, *, bool upper=False, bool check_errors=False, Tensor(a!) L, Tensor(b!) info) -> (Tensor(a!) L, Tensor(b!) info)")

// aten::linalg_cholesky_ex.L(Tensor self, *, bool upper=False, bool check_errors=False, Tensor(a!) L, Tensor(b!) info) -> (Tensor(a!) L, Tensor(b!) info)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_cholesky_ex_L::schema> create_linalg_cholesky_ex_L_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_cholesky_ex_L::name, linalg_cholesky_ex_L::overload_name)
      .typed<linalg_cholesky_ex_L::schema>();
}

// aten::linalg_cholesky_ex.L(Tensor self, *, bool upper=False, bool check_errors=False, Tensor(a!) L, Tensor(b!) info) -> (Tensor(a!) L, Tensor(b!) info)
::std::tuple<at::Tensor &,at::Tensor &> linalg_cholesky_ex_L::call(const at::Tensor & self, bool upper, bool check_errors, at::Tensor & L, at::Tensor & info) {
    static auto op = create_linalg_cholesky_ex_L_typed_handle();
    return op.call(self, upper, check_errors, L, info);
}

// aten::linalg_cholesky_ex.L(Tensor self, *, bool upper=False, bool check_errors=False, Tensor(a!) L, Tensor(b!) info) -> (Tensor(a!) L, Tensor(b!) info)
::std::tuple<at::Tensor &,at::Tensor &> linalg_cholesky_ex_L::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool upper, bool check_errors, at::Tensor & L, at::Tensor & info) {
    static auto op = create_linalg_cholesky_ex_L_typed_handle();
    return op.redispatch(dispatchKeySet, self, upper, check_errors, L, info);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_cholesky, name, "aten::linalg_cholesky")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_cholesky, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_cholesky, schema_str, "linalg_cholesky(Tensor self, *, bool upper=False) -> Tensor")

// aten::linalg_cholesky(Tensor self, *, bool upper=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<linalg_cholesky::schema> create_linalg_cholesky_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_cholesky::name, linalg_cholesky::overload_name)
      .typed<linalg_cholesky::schema>();
}

// aten::linalg_cholesky(Tensor self, *, bool upper=False) -> Tensor
at::Tensor linalg_cholesky::call(const at::Tensor & self, bool upper) {
    static auto op = create_linalg_cholesky_typed_handle();
    return op.call(self, upper);
}

// aten::linalg_cholesky(Tensor self, *, bool upper=False) -> Tensor
at::Tensor linalg_cholesky::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool upper) {
    static auto op = create_linalg_cholesky_typed_handle();
    return op.redispatch(dispatchKeySet, self, upper);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_matmul, name, "aten::linalg_matmul")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_matmul, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_matmul, schema_str, "linalg_matmul(Tensor self, Tensor other) -> Tensor")

// aten::linalg_matmul(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<linalg_matmul::schema> create_linalg_matmul_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_matmul::name, linalg_matmul::overload_name)
      .typed<linalg_matmul::schema>();
}

// aten::linalg_matmul(Tensor self, Tensor other) -> Tensor
at::Tensor linalg_matmul::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_linalg_matmul_typed_handle();
    return op.call(self, other);
}

// aten::linalg_matmul(Tensor self, Tensor other) -> Tensor
at::Tensor linalg_matmul::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_linalg_matmul_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_slogdet, name, "aten::linalg_slogdet")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_slogdet, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_slogdet, schema_str, "linalg_slogdet(Tensor self) -> (Tensor sign, Tensor logabsdet)")

// aten::linalg_slogdet(Tensor self) -> (Tensor sign, Tensor logabsdet)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_slogdet::schema> create_linalg_slogdet_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_slogdet::name, linalg_slogdet::overload_name)
      .typed<linalg_slogdet::schema>();
}

// aten::linalg_slogdet(Tensor self) -> (Tensor sign, Tensor logabsdet)
::std::tuple<at::Tensor,at::Tensor> linalg_slogdet::call(const at::Tensor & self) {
    static auto op = create_linalg_slogdet_typed_handle();
    return op.call(self);
}

// aten::linalg_slogdet(Tensor self) -> (Tensor sign, Tensor logabsdet)
::std::tuple<at::Tensor,at::Tensor> linalg_slogdet::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_linalg_slogdet_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_eigvals, name, "aten::linalg_eigvals")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_eigvals, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_eigvals, schema_str, "linalg_eigvals(Tensor self) -> Tensor")

// aten::linalg_eigvals(Tensor self) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<linalg_eigvals::schema> create_linalg_eigvals_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_eigvals::name, linalg_eigvals::overload_name)
      .typed<linalg_eigvals::schema>();
}

// aten::linalg_eigvals(Tensor self) -> Tensor
at::Tensor linalg_eigvals::call(const at::Tensor & self) {
    static auto op = create_linalg_eigvals_typed_handle();
    return op.call(self);
}

// aten::linalg_eigvals(Tensor self) -> Tensor
at::Tensor linalg_eigvals::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self) {
    static auto op = create_linalg_eigvals_typed_handle();
    return op.redispatch(dispatchKeySet, self);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_linalg_inv_out_helper_, name, "aten::_linalg_inv_out_helper_")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_linalg_inv_out_helper_, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_linalg_inv_out_helper_, schema_str, "_linalg_inv_out_helper_(Tensor(a!) self, Tensor(b!) infos_lu, Tensor(c!) infos_getri) -> Tensor(a!)")

// aten::_linalg_inv_out_helper_(Tensor(a!) self, Tensor(b!) infos_lu, Tensor(c!) infos_getri) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<_linalg_inv_out_helper_::schema> create__linalg_inv_out_helper__typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_linalg_inv_out_helper_::name, _linalg_inv_out_helper_::overload_name)
      .typed<_linalg_inv_out_helper_::schema>();
}

// aten::_linalg_inv_out_helper_(Tensor(a!) self, Tensor(b!) infos_lu, Tensor(c!) infos_getri) -> Tensor(a!)
at::Tensor & _linalg_inv_out_helper_::call(at::Tensor & self, at::Tensor & infos_lu, at::Tensor & infos_getri) {
    static auto op = create__linalg_inv_out_helper__typed_handle();
    return op.call(self, infos_lu, infos_getri);
}

// aten::_linalg_inv_out_helper_(Tensor(a!) self, Tensor(b!) infos_lu, Tensor(c!) infos_getri) -> Tensor(a!)
at::Tensor & _linalg_inv_out_helper_::redispatch(c10::DispatchKeySet dispatchKeySet, at::Tensor & self, at::Tensor & infos_lu, at::Tensor & infos_getri) {
    static auto op = create__linalg_inv_out_helper__typed_handle();
    return op.redispatch(dispatchKeySet, self, infos_lu, infos_getri);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_inv_ex, name, "aten::linalg_inv_ex")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_inv_ex, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_inv_ex, schema_str, "linalg_inv_ex(Tensor self, *, bool check_errors=False) -> (Tensor inverse, Tensor info)")

// aten::linalg_inv_ex(Tensor self, *, bool check_errors=False) -> (Tensor inverse, Tensor info)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_inv_ex::schema> create_linalg_inv_ex_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_inv_ex::name, linalg_inv_ex::overload_name)
      .typed<linalg_inv_ex::schema>();
}

// aten::linalg_inv_ex(Tensor self, *, bool check_errors=False) -> (Tensor inverse, Tensor info)
::std::tuple<at::Tensor,at::Tensor> linalg_inv_ex::call(const at::Tensor & self, bool check_errors) {
    static auto op = create_linalg_inv_ex_typed_handle();
    return op.call(self, check_errors);
}

// aten::linalg_inv_ex(Tensor self, *, bool check_errors=False) -> (Tensor inverse, Tensor info)
::std::tuple<at::Tensor,at::Tensor> linalg_inv_ex::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, bool check_errors) {
    static auto op = create_linalg_inv_ex_typed_handle();
    return op.redispatch(dispatchKeySet, self, check_errors);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_inv_out, name, "aten::linalg_inv")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_inv_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_inv_out, schema_str, "linalg_inv.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

// aten::linalg_inv.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_inv_out::schema> create_linalg_inv_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_inv_out::name, linalg_inv_out::overload_name)
      .typed<linalg_inv_out::schema>();
}

// aten::linalg_inv.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_inv_out::call(const at::Tensor & self, at::Tensor & out) {
    static auto op = create_linalg_inv_out_typed_handle();
    return op.call(self, out);
}

// aten::linalg_inv.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_inv_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::Tensor & out) {
    static auto op = create_linalg_inv_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(inner, name, "aten::inner")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(inner, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(inner, schema_str, "inner(Tensor self, Tensor other) -> Tensor")

// aten::inner(Tensor self, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<inner::schema> create_inner_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(inner::name, inner::overload_name)
      .typed<inner::schema>();
}

// aten::inner(Tensor self, Tensor other) -> Tensor
at::Tensor inner::call(const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_inner_typed_handle();
    return op.call(self, other);
}

// aten::inner(Tensor self, Tensor other) -> Tensor
at::Tensor inner::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & other) {
    static auto op = create_inner_typed_handle();
    return op.redispatch(dispatchKeySet, self, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ger_out, name, "aten::ger")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ger_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(ger_out, schema_str, "ger.out(Tensor self, Tensor vec2, *, Tensor(a!) out) -> Tensor(a!)")

// aten::ger.out(Tensor self, Tensor vec2, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<ger_out::schema> create_ger_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(ger_out::name, ger_out::overload_name)
      .typed<ger_out::schema>();
}

// aten::ger.out(Tensor self, Tensor vec2, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & ger_out::call(const at::Tensor & self, const at::Tensor & vec2, at::Tensor & out) {
    static auto op = create_ger_out_typed_handle();
    return op.call(self, vec2, out);
}

// aten::ger.out(Tensor self, Tensor vec2, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & ger_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & vec2, at::Tensor & out) {
    static auto op = create_ger_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, vec2, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_norm_ord_str, name, "aten::linalg_norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_norm_ord_str, overload_name, "ord_str")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_norm_ord_str, schema_str, "linalg_norm.ord_str(Tensor self, str ord, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor")

// aten::linalg_norm.ord_str(Tensor self, str ord, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<linalg_norm_ord_str::schema> create_linalg_norm_ord_str_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_norm_ord_str::name, linalg_norm_ord_str::overload_name)
      .typed<linalg_norm_ord_str::schema>();
}

// aten::linalg_norm.ord_str(Tensor self, str ord, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
at::Tensor linalg_norm_ord_str::call(const at::Tensor & self, c10::string_view ord, c10::optional<at::IntArrayRef> dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_linalg_norm_ord_str_typed_handle();
    return op.call(self, ord, dim, keepdim, dtype);
}

// aten::linalg_norm.ord_str(Tensor self, str ord, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
at::Tensor linalg_norm_ord_str::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::string_view ord, c10::optional<at::IntArrayRef> dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_linalg_norm_ord_str_typed_handle();
    return op.redispatch(dispatchKeySet, self, ord, dim, keepdim, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_norm_ord_str_out, name, "aten::linalg_norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_norm_ord_str_out, overload_name, "ord_str_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_norm_ord_str_out, schema_str, "linalg_norm.ord_str_out(Tensor self, str ord, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)")

// aten::linalg_norm.ord_str_out(Tensor self, str ord, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_norm_ord_str_out::schema> create_linalg_norm_ord_str_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_norm_ord_str_out::name, linalg_norm_ord_str_out::overload_name)
      .typed<linalg_norm_ord_str_out::schema>();
}

// aten::linalg_norm.ord_str_out(Tensor self, str ord, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_norm_ord_str_out::call(const at::Tensor & self, c10::string_view ord, c10::optional<at::IntArrayRef> dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
    static auto op = create_linalg_norm_ord_str_out_typed_handle();
    return op.call(self, ord, dim, keepdim, dtype, out);
}

// aten::linalg_norm.ord_str_out(Tensor self, str ord, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_norm_ord_str_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::string_view ord, c10::optional<at::IntArrayRef> dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
    static auto op = create_linalg_norm_ord_str_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, ord, dim, keepdim, dtype, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_vector_norm, name, "aten::linalg_vector_norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_vector_norm, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_vector_norm, schema_str, "linalg_vector_norm(Tensor self, Scalar ord=2, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor")

// aten::linalg_vector_norm(Tensor self, Scalar ord=2, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<linalg_vector_norm::schema> create_linalg_vector_norm_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_vector_norm::name, linalg_vector_norm::overload_name)
      .typed<linalg_vector_norm::schema>();
}

// aten::linalg_vector_norm(Tensor self, Scalar ord=2, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
at::Tensor linalg_vector_norm::call(const at::Tensor & self, const at::Scalar & ord, c10::optional<at::IntArrayRef> dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_linalg_vector_norm_typed_handle();
    return op.call(self, ord, dim, keepdim, dtype);
}

// aten::linalg_vector_norm(Tensor self, Scalar ord=2, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
at::Tensor linalg_vector_norm::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Scalar & ord, c10::optional<at::IntArrayRef> dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
    static auto op = create_linalg_vector_norm_typed_handle();
    return op.redispatch(dispatchKeySet, self, ord, dim, keepdim, dtype);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_matrix_norm_str_ord_out, name, "aten::linalg_matrix_norm")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_matrix_norm_str_ord_out, overload_name, "str_ord_out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_matrix_norm_str_ord_out, schema_str, "linalg_matrix_norm.str_ord_out(Tensor self, str ord='fro', int[] dim=[-2,-1], bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)")

// aten::linalg_matrix_norm.str_ord_out(Tensor self, str ord='fro', int[] dim=[-2,-1], bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_matrix_norm_str_ord_out::schema> create_linalg_matrix_norm_str_ord_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_matrix_norm_str_ord_out::name, linalg_matrix_norm_str_ord_out::overload_name)
      .typed<linalg_matrix_norm_str_ord_out::schema>();
}

// aten::linalg_matrix_norm.str_ord_out(Tensor self, str ord='fro', int[] dim=[-2,-1], bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_matrix_norm_str_ord_out::call(const at::Tensor & self, c10::string_view ord, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
    static auto op = create_linalg_matrix_norm_str_ord_out_typed_handle();
    return op.call(self, ord, dim, keepdim, dtype, out);
}

// aten::linalg_matrix_norm.str_ord_out(Tensor self, str ord='fro', int[] dim=[-2,-1], bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_matrix_norm_str_ord_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::string_view ord, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
    static auto op = create_linalg_matrix_norm_str_ord_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, ord, dim, keepdim, dtype, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_cond_out, name, "aten::linalg_cond")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_cond_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_cond_out, schema_str, "linalg_cond.out(Tensor self, Scalar? p=None, *, Tensor(a!) out) -> Tensor(a!)")

// aten::linalg_cond.out(Tensor self, Scalar? p=None, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_cond_out::schema> create_linalg_cond_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_cond_out::name, linalg_cond_out::overload_name)
      .typed<linalg_cond_out::schema>();
}

// aten::linalg_cond.out(Tensor self, Scalar? p=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_cond_out::call(const at::Tensor & self, const c10::optional<at::Scalar> & p, at::Tensor & out) {
    static auto op = create_linalg_cond_out_typed_handle();
    return op.call(self, p, out);
}

// aten::linalg_cond.out(Tensor self, Scalar? p=None, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_cond_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const c10::optional<at::Scalar> & p, at::Tensor & out) {
    static auto op = create_linalg_cond_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, p, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_pinv_out_rcond_tensor, name, "aten::linalg_pinv")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_pinv_out_rcond_tensor, overload_name, "out_rcond_tensor")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_pinv_out_rcond_tensor, schema_str, "linalg_pinv.out_rcond_tensor(Tensor self, Tensor rcond, bool hermitian=False, *, Tensor(a!) out) -> Tensor(a!)")

// aten::linalg_pinv.out_rcond_tensor(Tensor self, Tensor rcond, bool hermitian=False, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_pinv_out_rcond_tensor::schema> create_linalg_pinv_out_rcond_tensor_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_pinv_out_rcond_tensor::name, linalg_pinv_out_rcond_tensor::overload_name)
      .typed<linalg_pinv_out_rcond_tensor::schema>();
}

// aten::linalg_pinv.out_rcond_tensor(Tensor self, Tensor rcond, bool hermitian=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_pinv_out_rcond_tensor::call(const at::Tensor & self, const at::Tensor & rcond, bool hermitian, at::Tensor & out) {
    static auto op = create_linalg_pinv_out_rcond_tensor_typed_handle();
    return op.call(self, rcond, hermitian, out);
}

// aten::linalg_pinv.out_rcond_tensor(Tensor self, Tensor rcond, bool hermitian=False, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_pinv_out_rcond_tensor::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & rcond, bool hermitian, at::Tensor & out) {
    static auto op = create_linalg_pinv_out_rcond_tensor_typed_handle();
    return op.redispatch(dispatchKeySet, self, rcond, hermitian, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_solve, name, "aten::linalg_solve")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_solve, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_solve, schema_str, "linalg_solve(Tensor input, Tensor other) -> Tensor")

// aten::linalg_solve(Tensor input, Tensor other) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<linalg_solve::schema> create_linalg_solve_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_solve::name, linalg_solve::overload_name)
      .typed<linalg_solve::schema>();
}

// aten::linalg_solve(Tensor input, Tensor other) -> Tensor
at::Tensor linalg_solve::call(const at::Tensor & input, const at::Tensor & other) {
    static auto op = create_linalg_solve_typed_handle();
    return op.call(input, other);
}

// aten::linalg_solve(Tensor input, Tensor other) -> Tensor
at::Tensor linalg_solve::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & other) {
    static auto op = create_linalg_solve_typed_handle();
    return op.redispatch(dispatchKeySet, input, other);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_tensorinv, name, "aten::linalg_tensorinv")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_tensorinv, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_tensorinv, schema_str, "linalg_tensorinv(Tensor self, int ind=2) -> Tensor")

// aten::linalg_tensorinv(Tensor self, int ind=2) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<linalg_tensorinv::schema> create_linalg_tensorinv_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_tensorinv::name, linalg_tensorinv::overload_name)
      .typed<linalg_tensorinv::schema>();
}

// aten::linalg_tensorinv(Tensor self, int ind=2) -> Tensor
at::Tensor linalg_tensorinv::call(const at::Tensor & self, int64_t ind) {
    static auto op = create_linalg_tensorinv_typed_handle();
    return op.call(self, ind);
}

// aten::linalg_tensorinv(Tensor self, int ind=2) -> Tensor
at::Tensor linalg_tensorinv::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t ind) {
    static auto op = create_linalg_tensorinv_typed_handle();
    return op.redispatch(dispatchKeySet, self, ind);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_tensorinv_out, name, "aten::linalg_tensorinv")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_tensorinv_out, overload_name, "out")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_tensorinv_out, schema_str, "linalg_tensorinv.out(Tensor self, int ind=2, *, Tensor(a!) out) -> Tensor(a!)")

// aten::linalg_tensorinv.out(Tensor self, int ind=2, *, Tensor(a!) out) -> Tensor(a!)
static C10_NOINLINE c10::TypedOperatorHandle<linalg_tensorinv_out::schema> create_linalg_tensorinv_out_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_tensorinv_out::name, linalg_tensorinv_out::overload_name)
      .typed<linalg_tensorinv_out::schema>();
}

// aten::linalg_tensorinv.out(Tensor self, int ind=2, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_tensorinv_out::call(const at::Tensor & self, int64_t ind, at::Tensor & out) {
    static auto op = create_linalg_tensorinv_out_typed_handle();
    return op.call(self, ind, out);
}

// aten::linalg_tensorinv.out(Tensor self, int ind=2, *, Tensor(a!) out) -> Tensor(a!)
at::Tensor & linalg_tensorinv_out::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, int64_t ind, at::Tensor & out) {
    static auto op = create_linalg_tensorinv_out_typed_handle();
    return op.redispatch(dispatchKeySet, self, ind, out);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_linalg_qr_helper, name, "aten::_linalg_qr_helper")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_linalg_qr_helper, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_linalg_qr_helper, schema_str, "_linalg_qr_helper(Tensor self, str mode) -> (Tensor, Tensor)")

// aten::_linalg_qr_helper(Tensor self, str mode) -> (Tensor, Tensor)
static C10_NOINLINE c10::TypedOperatorHandle<_linalg_qr_helper::schema> create__linalg_qr_helper_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_linalg_qr_helper::name, _linalg_qr_helper::overload_name)
      .typed<_linalg_qr_helper::schema>();
}

// aten::_linalg_qr_helper(Tensor self, str mode) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> _linalg_qr_helper::call(const at::Tensor & self, c10::string_view mode) {
    static auto op = create__linalg_qr_helper_typed_handle();
    return op.call(self, mode);
}

// aten::_linalg_qr_helper(Tensor self, str mode) -> (Tensor, Tensor)
::std::tuple<at::Tensor,at::Tensor> _linalg_qr_helper::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::string_view mode) {
    static auto op = create__linalg_qr_helper_typed_handle();
    return op.redispatch(dispatchKeySet, self, mode);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_matrix_rank, name, "aten::linalg_matrix_rank")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_matrix_rank, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(linalg_matrix_rank, schema_str, "linalg_matrix_rank(Tensor self, float? tol=None, bool hermitian=False) -> Tensor")

// aten::linalg_matrix_rank(Tensor self, float? tol=None, bool hermitian=False) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<linalg_matrix_rank::schema> create_linalg_matrix_rank_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(linalg_matrix_rank::name, linalg_matrix_rank::overload_name)
      .typed<linalg_matrix_rank::schema>();
}

// aten::linalg_matrix_rank(Tensor self, float? tol=None, bool hermitian=False) -> Tensor
at::Tensor linalg_matrix_rank::call(const at::Tensor & self, c10::optional<double> tol, bool hermitian) {
    static auto op = create_linalg_matrix_rank_typed_handle();
    return op.call(self, tol, hermitian);
}

// aten::linalg_matrix_rank(Tensor self, float? tol=None, bool hermitian=False) -> Tensor
at::Tensor linalg_matrix_rank::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, c10::optional<double> tol, bool hermitian) {
    static auto op = create_linalg_matrix_rank_typed_handle();
    return op.redispatch(dispatchKeySet, self, tol, hermitian);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_test_optional_filled_intlist, name, "aten::_test_optional_filled_intlist")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_test_optional_filled_intlist, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(_test_optional_filled_intlist, schema_str, "_test_optional_filled_intlist(Tensor values, int[2]? addends) -> Tensor")

// aten::_test_optional_filled_intlist(Tensor values, int[2]? addends) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<_test_optional_filled_intlist::schema> create__test_optional_filled_intlist_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(_test_optional_filled_intlist::name, _test_optional_filled_intlist::overload_name)
      .typed<_test_optional_filled_intlist::schema>();
}

// aten::_test_optional_filled_intlist(Tensor values, int[2]? addends) -> Tensor
at::Tensor _test_optional_filled_intlist::call(const at::Tensor & values, c10::optional<at::IntArrayRef> addends) {
    static auto op = create__test_optional_filled_intlist_typed_handle();
    return op.call(values, addends);
}

// aten::_test_optional_filled_intlist(Tensor values, int[2]? addends) -> Tensor
at::Tensor _test_optional_filled_intlist::redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & values, c10::optional<at::IntArrayRef> addends) {
    static auto op = create__test_optional_filled_intlist_typed_handle();
    return op.redispatch(dispatchKeySet, values, addends);
}

STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pad_sequence, name, "aten::pad_sequence")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pad_sequence, overload_name, "")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA(pad_sequence, schema_str, "pad_sequence(Tensor[] sequences, bool batch_first=False, float padding_value=0.0) -> Tensor")

// aten::pad_sequence(Tensor[] sequences, bool batch_first=False, float padding_value=0.0) -> Tensor
static C10_NOINLINE c10::TypedOperatorHandle<pad_sequence::schema> create_pad_sequence_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(pad_sequence::name, pad_sequence::overload_name)
      .typed<pad_sequence::schema>();
}

// aten::pad_sequence(Tensor[] sequences, bool batch_first=False, float padding_value=0.0) -> Tensor
at::Tensor pad_sequence::call(at::TensorList sequences, bool batch_first, double padding_value) {
    static auto op = create_pad_sequence_typed_handle();
    return op.call(sequences, batch_first, padding_value);
}

// aten::pad_sequence(Tensor[] sequences, bool batch_first=False, float padding_value=0.0) -> Tensor
at::Tensor pad_sequence::redispatch(c10::DispatchKeySet dispatchKeySet, at::TensorList sequences, bool batch_first, double padding_value) {
    static auto op = create_pad_sequence_typed_handle();
    return op.redispatch(dispatchKeySet, sequences, batch_first, padding_value);
}

}} // namespace at::_ops
