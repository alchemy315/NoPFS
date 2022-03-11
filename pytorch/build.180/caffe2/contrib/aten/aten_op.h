#pragma once
#include <unordered_map>
#include <string>
#include <ATen/ATen.h>
#include <c10/macros/Macros.h>
#include <caffe2/core/context.h>
#include <caffe2/core/operator.h>
#include <caffe2/utils/math.h>
#include <iostream>

// a map from descriptor strings (see [DESCRIPTORS])
// to the key in the switch statement that implements them
static std::unordered_map<std::string, int> op_to_key = {
  { "_cast_Byte-non_blocking-1", 0 },
  { "_cast_Byte-1", 1 },
  { "_cast_Char-non_blocking-1", 2 },
  { "_cast_Char-1", 3 },
  { "_cast_Double-non_blocking-1", 4 },
  { "_cast_Double-1", 5 },
  { "_cast_Float-non_blocking-1", 6 },
  { "_cast_Float-1", 7 },
  { "_cast_Int-non_blocking-1", 8 },
  { "_cast_Int-1", 9 },
  { "_cast_Long-non_blocking-1", 10 },
  { "_cast_Long-1", 11 },
  { "_cast_Short-non_blocking-1", 12 },
  { "_cast_Short-1", 13 },
  { "_cast_Half-non_blocking-1", 14 },
  { "_cast_Half-1", 15 },
  { "data-1", 16 },
  { "is_leaf-1", 17 },
  { "output_nr-1", 18 },
  { "_version-1", 19 },
  { "retains_grad-1", 20 },
  { "_fw_primal-level-1", 21 },
  { "_make_dual-level-2", 22 },
  { "_unpack_dual-level-1", 23 },
  { "align_as-2", 24 },
  { "align_tensors-*", 25 },
  { "_use_cudnn_ctc_loss-blank-input_lengths-target_lengths-2", 26 },
  { "_cudnn_ctc_loss-blank-deterministic-input_lengths-target_lengths-zero_infinity-2", 27 },
  { "_use_cudnn_rnn_flatten_weight-0", 28 },
  { "_cudnn_rnn_flatten_weight-batch_first-bidirectional-hidden_size-input_size-mode-num_layers-proj_size-weight_stride0-*", 29 },
  { "_cudnn_rnn-batch_first-batch_sizes-bidirectional-dropout-hidden_size-mode-num_layers-proj_size-train-weight_stride0-*", 30 },
  { "_debug_has_internal_overlap-1", 31 },
  { "_fused_dropout-p-1", 32 },
  { "_masked_scale-scale-2", 33 },
  { "_reshape_from_tensor-2", 34 },
  { "_shape_as_tensor-1", 35 },
  { "dropout-p-train-1", 36 },
  { "feature_dropout-p-train-1", 37 },
  { "alpha_dropout-p-train-1", 38 },
  { "feature_alpha_dropout-p-train-1", 39 },
  { "abs-1", 40 },
  { "absolute-1", 41 },
  { "angle-1", 42 },
  { "view_as_real-1", 43 },
  { "view_as_complex-1", 44 },
  { "sgn-1", 45 },
  { "real-1", 46 },
  { "imag-1", 47 },
  { "_conj-1", 48 },
  { "conj-1", 49 },
  { "_conj_physical-1", 50 },
  { "conj_physical-1", 51 },
  { "resolve_conj-1", 52 },
  { "resolve_neg-1", 53 },
  { "_neg_view-1", 54 },
  { "acos-1", 55 },
  { "arccos-1", 56 },
  { "avg_pool1d-ceil_mode-count_include_pad-kernel_size-padding-stride-1", 57 },
  { "avg_pool1d-ceil_mode-kernel_size-padding-stride-1", 58 },
  { "avg_pool1d-kernel_size-padding-stride-1", 59 },
  { "avg_pool1d-kernel_size-stride-1", 60 },
  { "avg_pool1d-kernel_size-1", 61 },
  { "adaptive_avg_pool1d-output_size-1", 62 },
  { "adaptive_max_pool1d-output_size-1", 63 },
  { "add-alpha-2", 64 },
  { "add-2", 65 },
  { "_add_relu-alpha-2", 66 },
  { "_add_relu-2", 67 },
  { "_add_relu-alpha-other-1", 68 },
  { "_add_relu-other-1", 69 },
  { "add-alpha-other-1", 70 },
  { "add-other-1", 71 },
  { "addmv-alpha-beta-3", 72 },
  { "addmv-beta-3", 73 },
  { "addmv-3", 74 },
  { "addr-alpha-beta-3", 75 },
  { "addr-beta-3", 76 },
  { "addr-3", 77 },
  { "affine_grid_generator-align_corners-size-1", 78 },
  { "affine_grid_generator_backward-align_corners-size-1", 79 },
  { "all-dim-keepdim-1", 80 },
  { "all-dim-1", 81 },
  { "allclose-atol-equal_nan-rtol-2", 82 },
  { "allclose-atol-rtol-2", 83 },
  { "allclose-rtol-2", 84 },
  { "allclose-2", 85 },
  { "any-dim-keepdim-1", 86 },
  { "any-dim-1", 87 },
  { "_dim_arange-dim-1", 88 },
  { "argmax-1", 89 },
  { "argmin-1", 90 },
  { "acosh-1", 91 },
  { "arccosh-1", 92 },
  { "asinh-1", 93 },
  { "arcsinh-1", 94 },
  { "atanh-1", 95 },
  { "arctanh-1", 96 },
  { "as_strided-size-stride-1", 97 },
  { "asin-1", 98 },
  { "arcsin-1", 99 },
  { "atan-1", 100 },
  { "arctan-1", 101 },
  { "atleast_1d-1", 102 },
  { "atleast_1d-*", 103 },
  { "atleast_2d-1", 104 },
  { "atleast_2d-*", 105 },
  { "atleast_3d-1", 106 },
  { "atleast_3d-*", 107 },
  { "baddbmm-alpha-beta-3", 108 },
  { "baddbmm-beta-3", 109 },
  { "baddbmm-3", 110 },
  { "batch_norm-cudnn_enabled-eps-momentum-training-5", 111 },
  { "quantized_batch_norm-eps-output_scale-output_zero_point-5", 112 },
  { "_batch_norm_impl_index-cudnn_enabled-eps-momentum-training-5", 113 },
  { "_batch_norm_impl_index_backward-eps-impl_index-output_mask-train-8", 114 },
  { "bernoulli-1", 115 },
  { "bernoulli-p-1", 116 },
  { "bilinear-4", 117 },
  { "binary_cross_entropy-reduction-3", 118 },
  { "binary_cross_entropy-3", 119 },
  { "binary_cross_entropy-2", 120 },
  { "binary_cross_entropy_backward-reduction-4", 121 },
  { "binary_cross_entropy_backward-4", 122 },
  { "binary_cross_entropy_backward-3", 123 },
  { "binary_cross_entropy_with_logits-reduction-4", 124 },
  { "binary_cross_entropy_with_logits-4", 125 },
  { "binary_cross_entropy_with_logits-3", 126 },
  { "binary_cross_entropy_with_logits-2", 127 },
  { "binary_cross_entropy_with_logits_backward-reduction-5", 128 },
  { "binary_cross_entropy_with_logits_backward-5", 129 },
  { "binary_cross_entropy_with_logits_backward-4", 130 },
  { "binary_cross_entropy_with_logits_backward-3", 131 },
  { "bincount-minlength-2", 132 },
  { "bincount-2", 133 },
  { "bincount-1", 134 },
  { "bitwise_not-1", 135 },
  { "copysign-2", 136 },
  { "copysign-other-1", 137 },
  { "logical_not-1", 138 },
  { "logical_xor-2", 139 },
  { "logical_and-2", 140 },
  { "logical_or-2", 141 },
  { "bmm-2", 142 },
  { "broadcast_tensors-*", 143 },
  { "broadcast_to-size-1", 144 },
  { "cat-dim-*", 145 },
  { "cat-*", 146 },
  { "concat-dim-*", 147 },
  { "concat-*", 148 },
  { "block_diag-*", 149 },
  { "ceil-1", 150 },
  { "chain_matmul-*", 151 },
  { "unsafe_chunk-chunks-dim-1", 152 },
  { "unsafe_chunk-chunks-1", 153 },
  { "chunk-chunks-dim-1", 154 },
  { "chunk-chunks-1", 155 },
  { "tensor_split-dim-sections-1", 156 },
  { "tensor_split-sections-1", 157 },
  { "tensor_split-dim-indices-1", 158 },
  { "tensor_split-indices-1", 159 },
  { "tensor_split-dim-2", 160 },
  { "tensor_split-2", 161 },
  { "clamp-1", 162 },
  { "clamp-3", 163 },
  { "clamp-2", 164 },
  { "clamp_max-max-1", 165 },
  { "clamp_max-2", 166 },
  { "clamp_min-min-1", 167 },
  { "clamp_min-2", 168 },
  { "clip-1", 169 },
  { "clip-3", 170 },
  { "clip-2", 171 },
  { "cudnn_is_acceptable-1", 172 },
  { "complex-2", 173 },
  { "polar-2", 174 },
  { "constant_pad_nd-pad-value-1", 175 },
  { "constant_pad_nd-pad-1", 176 },
  { "contiguous-1", 177 },
  { "convolution-dilation-groups-output_padding-padding-stride-transposed-3", 178 },
  { "convolution_overrideable-dilation-groups-output_padding-padding-stride-transposed-3", 179 },
  { "convolution_backward_overrideable-dilation-groups-output_mask-output_padding-padding-stride-transposed-3", 180 },
  { "_convolution-allow_tf32-benchmark-cudnn_enabled-deterministic-dilation-groups-output_padding-padding-stride-transposed-3", 181 },
  { "_convolution-benchmark-cudnn_enabled-deterministic-dilation-groups-output_padding-padding-stride-transposed-3", 182 },
  { "_convolution_nogroup-dilation-output_padding-padding-stride-transposed-3", 183 },
  { "_convolution_double_backward-allow_tf32-benchmark-cudnn_enabled-deterministic-dilation-groups-output_mask-output_padding-padding-stride-transposed-6", 184 },
  { "conv1d-dilation-groups-padding-stride-3", 185 },
  { "conv1d-dilation-padding-stride-3", 186 },
  { "conv1d-padding-stride-3", 187 },
  { "conv1d-stride-3", 188 },
  { "conv1d-3", 189 },
  { "conv1d-2", 190 },
  { "conv2d-dilation-groups-padding-stride-3", 191 },
  { "conv2d-dilation-padding-stride-3", 192 },
  { "conv2d-padding-stride-3", 193 },
  { "conv2d-stride-3", 194 },
  { "conv2d-3", 195 },
  { "conv2d-2", 196 },
  { "conv3d-dilation-groups-padding-stride-3", 197 },
  { "conv3d-dilation-padding-stride-3", 198 },
  { "conv3d-padding-stride-3", 199 },
  { "conv3d-stride-3", 200 },
  { "conv3d-3", 201 },
  { "conv3d-2", 202 },
  { "conv_tbc-pad-3", 203 },
  { "conv_tbc-3", 204 },
  { "conv_tbc_backward-pad-4", 205 },
  { "conv_transpose1d-dilation-groups-output_padding-padding-stride-3", 206 },
  { "conv_transpose1d-groups-output_padding-padding-stride-3", 207 },
  { "conv_transpose1d-output_padding-padding-stride-3", 208 },
  { "conv_transpose1d-padding-stride-3", 209 },
  { "conv_transpose1d-stride-3", 210 },
  { "conv_transpose1d-3", 211 },
  { "conv_transpose1d-2", 212 },
  { "conv_transpose2d-dilation-groups-output_padding-padding-stride-3", 213 },
  { "conv_transpose2d-groups-output_padding-padding-stride-3", 214 },
  { "conv_transpose2d-output_padding-padding-stride-3", 215 },
  { "conv_transpose2d-padding-stride-3", 216 },
  { "conv_transpose2d-stride-3", 217 },
  { "conv_transpose2d-3", 218 },
  { "conv_transpose2d-2", 219 },
  { "conv_transpose3d-dilation-groups-output_padding-padding-stride-3", 220 },
  { "conv_transpose3d-groups-output_padding-padding-stride-3", 221 },
  { "conv_transpose3d-output_padding-padding-stride-3", 222 },
  { "conv_transpose3d-padding-stride-3", 223 },
  { "conv_transpose3d-stride-3", 224 },
  { "conv_transpose3d-3", 225 },
  { "conv_transpose3d-2", 226 },
  { "_copy_from-non_blocking-2", 227 },
  { "_copy_from-2", 228 },
  { "_copy_from_and_resize-2", 229 },
  { "cos-1", 230 },
  { "cosh-1", 231 },
  { "cosine_embedding_loss-margin-reduction-3", 232 },
  { "cosine_embedding_loss-margin-3", 233 },
  { "cosine_embedding_loss-3", 234 },
  { "count_nonzero-dim-1", 235 },
  { "count_nonzero-1", 236 },
  { "cov-correction-3", 237 },
  { "cov-correction-2", 238 },
  { "cov-correction-1", 239 },
  { "cov-1", 240 },
  { "corrcoef-1", 241 },
  { "cudnn_affine_grid_generator-C-H-N-W-1", 242 },
  { "cudnn_affine_grid_generator_backward-C-H-N-W-1", 243 },
  { "cudnn_batch_norm-epsilon-exponential_average_factor-training-5", 244 },
  { "cudnn_batch_norm_backward-epsilon-8", 245 },
  { "cudnn_convolution-benchmark-deterministic-dilation-groups-padding-stride-3", 246 },
  { "cudnn_convolution-benchmark-deterministic-dilation-groups-padding-stride-2", 247 },
  { "cudnn_convolution-allow_tf32-benchmark-deterministic-dilation-groups-padding-stride-2", 248 },
  { "cudnn_convolution_backward_input-allow_tf32-benchmark-deterministic-dilation-groups-padding-self_size-stride-2", 249 },
  { "cudnn_convolution_backward-allow_tf32-benchmark-deterministic-dilation-groups-output_mask-padding-stride-3", 250 },
  { "cudnn_convolution_backward_weight-allow_tf32-benchmark-deterministic-dilation-groups-padding-stride-weight_size-2", 251 },
  { "cudnn_convolution_transpose-benchmark-deterministic-dilation-groups-output_padding-padding-stride-3", 252 },
  { "cudnn_convolution_transpose-benchmark-deterministic-dilation-groups-output_padding-padding-stride-2", 253 },
  { "cudnn_convolution_transpose-allow_tf32-benchmark-deterministic-dilation-groups-output_padding-padding-stride-2", 254 },
  { "cudnn_convolution_transpose_backward-allow_tf32-benchmark-deterministic-dilation-groups-output_mask-output_padding-padding-stride-3", 255 },
  { "cudnn_convolution_transpose_backward_input-allow_tf32-benchmark-deterministic-dilation-groups-padding-stride-2", 256 },
  { "cudnn_convolution_transpose_backward_weight-allow_tf32-benchmark-deterministic-dilation-groups-padding-stride-weight_size-2", 257 },
  { "cudnn_convolution_relu-dilation-groups-padding-stride-3", 258 },
  { "cudnn_grid_sampler-2", 259 },
  { "cudnn_grid_sampler_backward-3", 260 },
  { "cummax-dim-1", 261 },
  { "cummin-dim-1", 262 },
  { "cummaxmin_backward-dim-3", 263 },
  { "cumprod-dim-1", 264 },
  { "cumprod_backward-dim-3", 265 },
  { "cumsum-dim-1", 266 },
  { "cumulative_trapezoid-dim-2", 267 },
  { "cumulative_trapezoid-2", 268 },
  { "cumulative_trapezoid-dim-dx-1", 269 },
  { "cumulative_trapezoid-dx-1", 270 },
  { "cumulative_trapezoid-1", 271 },
  { "ctc_loss-blank-input_lengths-reduction-target_lengths-zero_infinity-2", 272 },
  { "ctc_loss-blank-input_lengths-reduction-target_lengths-2", 273 },
  { "ctc_loss-blank-input_lengths-target_lengths-2", 274 },
  { "ctc_loss-input_lengths-target_lengths-2", 275 },
  { "ctc_loss-blank-reduction-zero_infinity-4", 276 },
  { "ctc_loss-blank-reduction-4", 277 },
  { "ctc_loss-blank-4", 278 },
  { "ctc_loss-4", 279 },
  { "_ctc_loss-blank-input_lengths-target_lengths-zero_infinity-2", 280 },
  { "_ctc_loss-blank-input_lengths-target_lengths-2", 281 },
  { "_ctc_loss-input_lengths-target_lengths-2", 282 },
  { "_ctc_loss_backward-blank-input_lengths-target_lengths-zero_infinity-5", 283 },
  { "_ctc_loss_backward-blank-input_lengths-target_lengths-5", 284 },
  { "diag_embed-dim1-dim2-offset-1", 285 },
  { "diag_embed-dim1-offset-1", 286 },
  { "diag_embed-offset-1", 287 },
  { "diag_embed-1", 288 },
  { "diagflat-offset-1", 289 },
  { "diagflat-1", 290 },
  { "diagonal-dim1-dim2-offset-1", 291 },
  { "diagonal-dim1-offset-1", 292 },
  { "diagonal-offset-1", 293 },
  { "diagonal-1", 294 },
  { "diagonal_backward-dim1-dim2-input_sizes-offset-1", 295 },
  { "diff-dim-n-3", 296 },
  { "diff-dim-n-2", 297 },
  { "diff-dim-n-1", 298 },
  { "diff-n-1", 299 },
  { "diff-1", 300 },
  { "gradient-1", 301 },
  { "gradient-dim-edge_order-spacing-1", 302 },
  { "gradient-dim-spacing-1", 303 },
  { "gradient-dim-edge_order-1", 304 },
  { "gradient-dim-1", 305 },
  { "gradient-*", 306 },
  { "gradient-dim-edge_order-*", 307 },
  { "gradient-dim-*", 308 },
  { "div-2", 309 },
  { "div-other-1", 310 },
  { "divide-2", 311 },
  { "divide-other-1", 312 },
  { "true_divide-2", 313 },
  { "true_divide-other-1", 314 },
  { "dot-2", 315 },
  { "vdot-2", 316 },
  { "embedding-padding_idx-scale_grad_by_freq-sparse-2", 317 },
  { "embedding-padding_idx-scale_grad_by_freq-2", 318 },
  { "embedding-padding_idx-2", 319 },
  { "embedding-2", 320 },
  { "embedding_backward-num_weights-padding_idx-scale_grad_by_freq-sparse-2", 321 },
  { "embedding_dense_backward-num_weights-padding_idx-scale_grad_by_freq-2", 322 },
  { "embedding_sparse_backward-num_weights-padding_idx-scale_grad_by_freq-2", 323 },
  { "_embedding_bag_forward_only-include_last_offset-mode-padding_idx-scale_grad_by_freq-sparse-4", 324 },
  { "_embedding_bag_forward_only-include_last_offset-mode-scale_grad_by_freq-sparse-4", 325 },
  { "_embedding_bag_forward_only-mode-scale_grad_by_freq-sparse-4", 326 },
  { "_embedding_bag_forward_only-mode-scale_grad_by_freq-sparse-3", 327 },
  { "_embedding_bag_forward_only-mode-scale_grad_by_freq-3", 328 },
  { "_embedding_bag_forward_only-scale_grad_by_freq-3", 329 },
  { "_embedding_bag_forward_only-3", 330 },
  { "row_stack-*", 331 },
  { "embedding_bag-include_last_offset-mode-scale_grad_by_freq-sparse-4", 332 },
  { "embedding_bag-mode-scale_grad_by_freq-sparse-4", 333 },
  { "embedding_bag-mode-scale_grad_by_freq-sparse-3", 334 },
  { "embedding_bag-mode-scale_grad_by_freq-3", 335 },
  { "embedding_bag-scale_grad_by_freq-3", 336 },
  { "embedding_bag-3", 337 },
  { "_embedding_bag-include_last_offset-mode-padding_idx-scale_grad_by_freq-sparse-4", 338 },
  { "_embedding_bag-include_last_offset-mode-scale_grad_by_freq-sparse-4", 339 },
  { "_embedding_bag-mode-scale_grad_by_freq-sparse-4", 340 },
  { "_embedding_bag-mode-scale_grad_by_freq-sparse-3", 341 },
  { "_embedding_bag-mode-scale_grad_by_freq-3", 342 },
  { "_embedding_bag-scale_grad_by_freq-3", 343 },
  { "_embedding_bag-3", 344 },
  { "_embedding_bag_backward-mode-num_weights-padding_idx-scale_grad_by_freq-sparse-7", 345 },
  { "_embedding_bag_backward-mode-num_weights-scale_grad_by_freq-sparse-7", 346 },
  { "_embedding_bag_sparse_backward-mode-num_weights-padding_idx-scale_grad_by_freq-6", 347 },
  { "_embedding_bag_sparse_backward-mode-num_weights-scale_grad_by_freq-6", 348 },
  { "_embedding_bag_dense_backward-mode-num_weights-padding_idx-scale_grad_by_freq-6", 349 },
  { "_embedding_bag_dense_backward-mode-num_weights-scale_grad_by_freq-6", 350 },
  { "_embedding_bag_per_sample_weights_backward-mode-padding_idx-5", 351 },
  { "_embedding_bag_per_sample_weights_backward-mode-5", 352 },
  { "erf-1", 353 },
  { "erfc-1", 354 },
  { "exp-1", 355 },
  { "exp2-1", 356 },
  { "expm1-1", 357 },
  { "expand-implicit-size-1", 358 },
  { "expand-size-1", 359 },
  { "expand_as-2", 360 },
  { "flatten-end_dim-start_dim-1", 361 },
  { "flatten-start_dim-1", 362 },
  { "flatten-1", 363 },
  { "unflatten-dim-sizes-1", 364 },
  { "floor-1", 365 },
  { "floor_divide-2", 366 },
  { "floor_divide-other-1", 367 },
  { "frac-1", 368 },
  { "gcd-2", 369 },
  { "lcm-2", 370 },
  { "grid_sampler-align_corners-interpolation_mode-padding_mode-2", 371 },
  { "grid_sampler_2d-align_corners-interpolation_mode-padding_mode-2", 372 },
  { "grid_sampler_2d_backward-align_corners-interpolation_mode-padding_mode-3", 373 },
  { "_grid_sampler_2d_cpu_fallback-align_corners-interpolation_mode-padding_mode-2", 374 },
  { "_grid_sampler_2d_cpu_fallback_backward-align_corners-interpolation_mode-padding_mode-3", 375 },
  { "grid_sampler_3d-align_corners-interpolation_mode-padding_mode-2", 376 },
  { "grid_sampler_3d_backward-align_corners-interpolation_mode-padding_mode-3", 377 },
  { "hinge_embedding_loss-margin-reduction-2", 378 },
  { "hinge_embedding_loss-margin-2", 379 },
  { "hinge_embedding_loss-2", 380 },
  { "group_norm-cudnn_enabled-eps-num_groups-3", 381 },
  { "group_norm-eps-num_groups-3", 382 },
  { "group_norm-num_groups-3", 383 },
  { "group_norm-num_groups-2", 384 },
  { "group_norm-num_groups-1", 385 },
  { "native_group_norm-C-HxW-N-eps-group-3", 386 },
  { "native_group_norm_backward-C-HxW-N-group-output_mask-5", 387 },
  { "_fft_r2c-dim-normalization-onesided-1", 388 },
  { "_fft_c2r-dim-last_dim_size-normalization-1", 389 },
  { "_fft_c2c-dim-forward-normalization-1", 390 },
  { "_cufft_get_plan_cache_size-device_index-0", 391 },
  { "_cufft_get_plan_cache_max_size-device_index-0", 392 },
  { "index-*", 393 },
  { "index_copy-dim-3", 394 },
  { "index_put-accumulate-*", 395 },
  { "index_put-*", 396 },
  { "instance_norm-cudnn_enabled-eps-momentum-use_input_stats-5", 397 },
  { "inverse-1", 398 },
  { "_inverse_helper-1", 399 },
  { "isclose-atol-equal_nan-rtol-2", 400 },
  { "isclose-atol-rtol-2", 401 },
  { "isclose-rtol-2", 402 },
  { "isclose-2", 403 },
  { "isin-assume_unique-invert-2", 404 },
  { "isin-assume_unique-2", 405 },
  { "isin-2", 406 },
  { "isin-assume_unique-invert-test_element-1", 407 },
  { "isin-assume_unique-test_element-1", 408 },
  { "isin-test_element-1", 409 },
  { "isin-assume_unique-element-invert-1", 410 },
  { "isin-assume_unique-element-1", 411 },
  { "isin-element-1", 412 },
  { "isnan-1", 413 },
  { "is_distributed-1", 414 },
  { "is_floating_point-1", 415 },
  { "is_complex-1", 416 },
  { "is_conj-1", 417 },
  { "is_neg-1", 418 },
  { "isreal-1", 419 },
  { "is_nonzero-1", 420 },
  { "is_same_size-2", 421 },
  { "is_signed-1", 422 },
  { "is_inference-1", 423 },
  { "kl_div-log_target-reduction-2", 424 },
  { "kl_div-reduction-2", 425 },
  { "kl_div-2", 426 },
  { "kl_div_backward-log_target-reduction-3", 427 },
  { "kl_div_backward-reduction-3", 428 },
  { "kl_div_backward-3", 429 },
  { "kron-2", 430 },
  { "kthvalue-dim-k-keepdim-1", 431 },
  { "kthvalue-dim-k-1", 432 },
  { "kthvalue-k-1", 433 },
  { "layer_norm-cudnn_enable-eps-normalized_shape-3", 434 },
  { "layer_norm-eps-normalized_shape-3", 435 },
  { "layer_norm-normalized_shape-3", 436 },
  { "layer_norm-normalized_shape-2", 437 },
  { "layer_norm-normalized_shape-1", 438 },
  { "native_layer_norm-eps-normalized_shape-3", 439 },
  { "native_layer_norm_backward-normalized_shape-output_mask-6", 440 },
  { "nan_to_num-1", 441 },
  { "linear-3", 442 },
  { "linear-2", 443 },
  { "mkldnn_linear-3", 444 },
  { "mkldnn_linear-2", 445 },
  { "mkldnn_linear_backward_input-input_size-2", 446 },
  { "mkldnn_linear_backward_weights-bias_defined-3", 447 },
  { "mkldnn_linear_backward-output_mask-3", 448 },
  { "fbgemm_linear_int8_weight_fp32_activation-weight_scale-weight_zero_point-5", 449 },
  { "fbgemm_linear_int8_weight-weight_scale-weight_zero_point-5", 450 },
  { "fbgemm_pack_gemm_matrix_fp16-1", 451 },
  { "fbgemm_linear_fp16_weight_fp32_activation-3", 452 },
  { "fbgemm_linear_fp16_weight-3", 453 },
  { "fbgemm_pack_quantized_matrix-1", 454 },
  { "fbgemm_pack_quantized_matrix-K-N-1", 455 },
  { "ldexp-2", 456 },
  { "log-1", 457 },
  { "log10-1", 458 },
  { "log1p-1", 459 },
  { "log2-1", 460 },
  { "logaddexp-2", 461 },
  { "logaddexp2-2", 462 },
  { "xlogy-2", 463 },
  { "xlogy-self-1", 464 },
  { "xlogy-other-1", 465 },
  { "logdet-1", 466 },
  { "log_softmax-dim-1", 467 },
  { "_log_softmax-dim-half_to_float-1", 468 },
  { "_log_softmax_backward_data-dim-3", 469 },
  { "_logcumsumexp-dim-1", 470 },
  { "logcumsumexp-dim-1", 471 },
  { "logsumexp-dim-keepdim-1", 472 },
  { "logsumexp-dim-1", 473 },
  { "margin_ranking_loss-margin-reduction-3", 474 },
  { "margin_ranking_loss-margin-3", 475 },
  { "margin_ranking_loss-3", 476 },
  { "matmul-2", 477 },
  { "matrix_rank-symmetric-tol-1", 478 },
  { "matrix_rank-tol-1", 479 },
  { "matrix_rank-symmetric-1", 480 },
  { "matrix_rank-1", 481 },
  { "matrix_power-n-1", 482 },
  { "matrix_exp-1", 483 },
  { "matrix_exp_backward-2", 484 },
  { "_aminmax-1", 485 },
  { "_aminmax-dim-keepdim-1", 486 },
  { "_aminmax-dim-1", 487 },
  { "aminmax-1", 488 },
  { "_compute_linear_combination-2", 489 },
  { "max-dim-keepdim-1", 490 },
  { "max-dim-1", 491 },
  { "value_selecting_reduction_backward-dim-keepdim-sizes-2", 492 },
  { "amax-dim-keepdim-1", 493 },
  { "amax-dim-1", 494 },
  { "amax-1", 495 },
  { "max_pool1d_with_indices-ceil_mode-dilation-kernel_size-padding-stride-1", 496 },
  { "max_pool1d_with_indices-dilation-kernel_size-padding-stride-1", 497 },
  { "max_pool1d_with_indices-kernel_size-padding-stride-1", 498 },
  { "max_pool1d_with_indices-kernel_size-stride-1", 499 },
  { "max_pool1d_with_indices-kernel_size-1", 500 },
  { "max_pool1d-ceil_mode-dilation-kernel_size-padding-stride-1", 501 },
  { "max_pool1d-dilation-kernel_size-padding-stride-1", 502 },
  { "max_pool1d-kernel_size-padding-stride-1", 503 },
  { "max_pool1d-kernel_size-stride-1", 504 },
  { "max_pool1d-kernel_size-1", 505 },
  { "max_pool2d-ceil_mode-dilation-kernel_size-padding-stride-1", 506 },
  { "max_pool2d-dilation-kernel_size-padding-stride-1", 507 },
  { "max_pool2d-kernel_size-padding-stride-1", 508 },
  { "max_pool2d-kernel_size-stride-1", 509 },
  { "max_pool2d-kernel_size-1", 510 },
  { "mkldnn_max_pool2d-ceil_mode-dilation-kernel_size-padding-stride-1", 511 },
  { "mkldnn_max_pool2d-dilation-kernel_size-padding-stride-1", 512 },
  { "mkldnn_max_pool2d-kernel_size-padding-stride-1", 513 },
  { "mkldnn_max_pool2d-kernel_size-stride-1", 514 },
  { "mkldnn_max_pool2d-kernel_size-1", 515 },
  { "mkldnn_max_pool2d_backward-ceil_mode-dilation-kernel_size-padding-stride-3", 516 },
  { "mkldnn_max_pool2d_backward-dilation-kernel_size-padding-stride-3", 517 },
  { "mkldnn_max_pool2d_backward-kernel_size-padding-stride-3", 518 },
  { "mkldnn_max_pool2d_backward-kernel_size-stride-3", 519 },
  { "mkldnn_max_pool2d_backward-kernel_size-3", 520 },
  { "mkldnn_max_pool3d-ceil_mode-dilation-kernel_size-padding-stride-1", 521 },
  { "mkldnn_max_pool3d-dilation-kernel_size-padding-stride-1", 522 },
  { "mkldnn_max_pool3d-kernel_size-padding-stride-1", 523 },
  { "mkldnn_max_pool3d-kernel_size-stride-1", 524 },
  { "mkldnn_max_pool3d-kernel_size-1", 525 },
  { "mkldnn_max_pool3d_backward-ceil_mode-dilation-kernel_size-padding-stride-3", 526 },
  { "mkldnn_max_pool3d_backward-dilation-kernel_size-padding-stride-3", 527 },
  { "mkldnn_max_pool3d_backward-kernel_size-padding-stride-3", 528 },
  { "mkldnn_max_pool3d_backward-kernel_size-stride-3", 529 },
  { "mkldnn_max_pool3d_backward-kernel_size-3", 530 },
  { "quantized_max_pool1d-ceil_mode-dilation-kernel_size-padding-stride-1", 531 },
  { "quantized_max_pool1d-dilation-kernel_size-padding-stride-1", 532 },
  { "quantized_max_pool1d-kernel_size-padding-stride-1", 533 },
  { "quantized_max_pool1d-kernel_size-stride-1", 534 },
  { "quantized_max_pool1d-kernel_size-1", 535 },
  { "quantized_max_pool2d-ceil_mode-dilation-kernel_size-padding-stride-1", 536 },
  { "quantized_max_pool2d-dilation-kernel_size-padding-stride-1", 537 },
  { "quantized_max_pool2d-kernel_size-padding-stride-1", 538 },
  { "quantized_max_pool2d-kernel_size-stride-1", 539 },
  { "quantized_max_pool2d-kernel_size-1", 540 },
  { "max_pool3d-ceil_mode-dilation-kernel_size-padding-stride-1", 541 },
  { "max_pool3d-dilation-kernel_size-padding-stride-1", 542 },
  { "max_pool3d-kernel_size-padding-stride-1", 543 },
  { "max_pool3d-kernel_size-stride-1", 544 },
  { "max_pool3d-kernel_size-1", 545 },
  { "mean-1", 546 },
  { "mean-dim-keepdim-1", 547 },
  { "mean-dim-1", 548 },
  { "nanmean-dim-keepdim-1", 549 },
  { "nanmean-dim-1", 550 },
  { "nanmean-1", 551 },
  { "median-1", 552 },
  { "median-dim-keepdim-1", 553 },
  { "median-dim-1", 554 },
  { "nanmedian-1", 555 },
  { "nanmedian-dim-keepdim-1", 556 },
  { "nanmedian-dim-1", 557 },
  { "min-dim-keepdim-1", 558 },
  { "min-dim-1", 559 },
  { "amin-dim-keepdim-1", 560 },
  { "amin-dim-1", 561 },
  { "amin-1", 562 },
  { "mkldnn_convolution-dilation-groups-padding-stride-3", 563 },
  { "mkldnn_convolution_backward_input-bias_defined-dilation-groups-padding-self_size-stride-2", 564 },
  { "mkldnn_convolution_backward_weights-bias_defined-dilation-groups-padding-stride-weight_size-2", 565 },
  { "mkldnn_convolution_backward-dilation-groups-output_mask-padding-stride-3", 566 },
  { "miopen_batch_norm-epsilon-exponential_average_factor-training-5", 567 },
  { "miopen_batch_norm_backward-epsilon-7", 568 },
  { "miopen_convolution-benchmark-deterministic-dilation-groups-padding-stride-3", 569 },
  { "miopen_convolution_backward_input-benchmark-deterministic-dilation-groups-padding-self_size-stride-2", 570 },
  { "miopen_convolution_backward-benchmark-deterministic-dilation-groups-output_mask-padding-stride-3", 571 },
  { "miopen_convolution_backward_bias-1", 572 },
  { "miopen_convolution_backward_weight-benchmark-deterministic-dilation-groups-padding-stride-weight_size-2", 573 },
  { "miopen_convolution_transpose-benchmark-deterministic-dilation-groups-output_padding-padding-stride-3", 574 },
  { "miopen_convolution_transpose_backward-benchmark-deterministic-dilation-groups-output_mask-output_padding-padding-stride-3", 575 },
  { "miopen_convolution_transpose_backward_input-benchmark-deterministic-dilation-groups-padding-stride-2", 576 },
  { "miopen_convolution_transpose_backward_weight-benchmark-deterministic-dilation-groups-padding-stride-weight_size-2", 577 },
  { "miopen_depthwise_convolution-benchmark-deterministic-dilation-groups-padding-stride-3", 578 },
  { "miopen_depthwise_convolution_backward_input-benchmark-deterministic-dilation-groups-padding-self_size-stride-2", 579 },
  { "miopen_depthwise_convolution_backward-benchmark-deterministic-dilation-groups-output_mask-padding-stride-3", 580 },
  { "miopen_depthwise_convolution_backward_weight-benchmark-deterministic-dilation-groups-padding-stride-weight_size-2", 581 },
  { "miopen_rnn-batch_first-batch_sizes-bidirectional-dropout-hidden_size-mode-num_layers-train-weight_stride0-*", 582 },
  { "mm-2", 583 },
  { "_sparse_mm-2", 584 },
  { "_sparse_sparse_matmul-2", 585 },
  { "_sparse_mask_helper-2", 586 },
  { "mode-dim-keepdim-1", 587 },
  { "mode-dim-1", 588 },
  { "mode-1", 589 },
  { "mul-2", 590 },
  { "mul-other-1", 591 },
  { "multiply-2", 592 },
  { "multiply-other-1", 593 },
  { "mv-2", 594 },
  { "mvlgamma-p-1", 595 },
  { "narrow_copy-dim-length-start-1", 596 },
  { "narrow-dim-length-start-1", 597 },
  { "narrow-dim-length-2", 598 },
  { "native_batch_norm-eps-momentum-training-5", 599 },
  { "batch_norm_stats-eps-1", 600 },
  { "batch_norm_elemt-eps-5", 601 },
  { "batch_norm_gather_stats-count-eps-momentum-5", 602 },
  { "batch_norm_gather_stats_with_counts-eps-momentum-6", 603 },
  { "native_batch_norm_backward-eps-output_mask-train-7", 604 },
  { "batch_norm_backward_reduce-bias_g-input_g-weight_g-5", 605 },
  { "batch_norm_backward_elemt-8", 606 },
  { "batch_norm_update_stats-momentum-3", 607 },
  { "is_vulkan_available-0", 608 },
  { "_nnpack_available-0", 609 },
  { "_nnpack_spatial_convolution-padding-stride-3", 610 },
  { "_nnpack_spatial_convolution-padding-3", 611 },
  { "_nnpack_spatial_convolution_backward-output_mask-padding-3", 612 },
  { "_nnpack_spatial_convolution_backward_input-padding-3", 613 },
  { "_nnpack_spatial_convolution_backward_weight-padding-weightsize-2", 614 },
  { "pairwise_distance-eps-keepdim-p-2", 615 },
  { "pairwise_distance-eps-p-2", 616 },
  { "pairwise_distance-p-2", 617 },
  { "pairwise_distance-2", 618 },
  { "cdist-p-2", 619 },
  { "cdist-2", 620 },
  { "_euclidean_dist-2", 621 },
  { "_cdist_backward-p-4", 622 },
  { "pdist-p-1", 623 },
  { "pdist-1", 624 },
  { "_pdist_forward-p-1", 625 },
  { "_pdist_forward-1", 626 },
  { "_pdist_backward-p-3", 627 },
  { "cosine_similarity-dim-eps-2", 628 },
  { "cosine_similarity-dim-2", 629 },
  { "cosine_similarity-2", 630 },
  { "permute-dims-1", 631 },
  { "movedim-destination-source-1", 632 },
  { "moveaxis-destination-source-1", 633 },
  { "numpy_T-1", 634 },
  { "pixel_shuffle-upscale_factor-1", 635 },
  { "pixel_unshuffle-downscale_factor-1", 636 },
  { "channel_shuffle-groups-1", 637 },
  { "is_pinned-1", 638 },
  { "pin_memory-1", 639 },
  { "_pin_memory-1", 640 },
  { "pinverse-rcond-1", 641 },
  { "pinverse-1", 642 },
  { "poisson_nll_loss-eps-full-log_input-reduction-2", 643 },
  { "rad2deg-1", 644 },
  { "deg2rad-1", 645 },
  { "ravel-1", 646 },
  { "reciprocal-1", 647 },
  { "neg-1", 648 },
  { "negative-1", 649 },
  { "repeat-repeats-1", 650 },
  { "repeat_interleave-1", 651 },
  { "repeat_interleave-2", 652 },
  { "repeat_interleave-repeats-1", 653 },
  { "reshape-shape-1", 654 },
  { "_reshape_alias-size-stride-1", 655 },
  { "_mkldnn_reshape-shape-1", 656 },
  { "reshape_as-2", 657 },
  { "round-1", 658 },
  { "rrelu-lower-training-upper-1", 659 },
  { "rrelu-lower-upper-1", 660 },
  { "rrelu-lower-1", 661 },
  { "rrelu-1", 662 },
  { "relu-1", 663 },
  { "relu6-1", 664 },
  { "prelu-2", 665 },
  { "prelu_backward-3", 666 },
  { "gelu-1", 667 },
  { "gelu_backward-2", 668 },
  { "infinitely_differentiable_gelu_backward-2", 669 },
  { "hardshrink-lambd-1", 670 },
  { "hardshrink-1", 671 },
  { "hardshrink_backward-lambd-2", 672 },
  { "rsqrt-1", 673 },
  { "select-dim-index-1", 674 },
  { "select_backward-dim-index-input_sizes-1", 675 },
  { "selu-1", 676 },
  { "celu-alpha-1", 677 },
  { "celu-1", 678 },
  { "silu-1", 679 },
  { "silu_backward-2", 680 },
  { "mish-1", 681 },
  { "mish_backward-2", 682 },
  { "sigmoid-1", 683 },
  { "logit-1", 684 },
  { "sin-1", 685 },
  { "sinc-1", 686 },
  { "sinh-1", 687 },
  { "detach-1", 688 },
  { "size-dim-1", 689 },
  { "slice-dim-1", 690 },
  { "slice-1", 691 },
  { "slice_backward-dim-end-input_sizes-start-step-1", 692 },
  { "slogdet-1", 693 },
  { "smm-2", 694 },
  { "softmax-dim-1", 695 },
  { "_softmax-dim-half_to_float-1", 696 },
  { "_softmax_backward_data-dim-3", 697 },
  { "unsafe_split-dim-split_size-1", 698 },
  { "unsafe_split-split_size-1", 699 },
  { "split-dim-split_size-1", 700 },
  { "split-split_size-1", 701 },
  { "unsafe_split_with_sizes-dim-split_sizes-1", 702 },
  { "unsafe_split_with_sizes-split_sizes-1", 703 },
  { "split_with_sizes-dim-split_sizes-1", 704 },
  { "split_with_sizes-split_sizes-1", 705 },
  { "hsplit-sections-1", 706 },
  { "hsplit-indices-1", 707 },
  { "vsplit-sections-1", 708 },
  { "vsplit-indices-1", 709 },
  { "dsplit-sections-1", 710 },
  { "dsplit-indices-1", 711 },
  { "squeeze-1", 712 },
  { "squeeze-dim-1", 713 },
  { "sspaddmm-alpha-beta-3", 714 },
  { "sspaddmm-beta-3", 715 },
  { "sspaddmm-3", 716 },
  { "stack-dim-*", 717 },
  { "stack-*", 718 },
  { "_stack-dim-*", 719 },
  { "_stack-*", 720 },
  { "hstack-*", 721 },
  { "vstack-*", 722 },
  { "dstack-*", 723 },
  { "stft-n_fft-1", 724 },
  { "istft-n_fft-1", 725 },
  { "stride-dim-1", 726 },
  { "sum-1", 727 },
  { "sum-dim-keepdim-1", 728 },
  { "sum-dim-1", 729 },
  { "nansum-1", 730 },
  { "nansum-dim-keepdim-1", 731 },
  { "nansum-dim-1", 732 },
  { "sum_to_size-size-1", 733 },
  { "sqrt-1", 734 },
  { "square-1", 735 },
  { "std-unbiased-1", 736 },
  { "std-1", 737 },
  { "std-dim-keepdim-unbiased-1", 738 },
  { "std-dim-unbiased-1", 739 },
  { "std-dim-1", 740 },
  { "std_mean-unbiased-1", 741 },
  { "std_mean-1", 742 },
  { "std_mean-dim-keepdim-unbiased-1", 743 },
  { "std_mean-dim-unbiased-1", 744 },
  { "std_mean-dim-1", 745 },
  { "prod-1", 746 },
  { "prod-dim-keepdim-1", 747 },
  { "prod-dim-1", 748 },
  { "t-1", 749 },
  { "tan-1", 750 },
  { "tanh-1", 751 },
  { "tensordot-dims_other-dims_self-2", 752 },
  { "threshold-threshold-value-1", 753 },
  { "threshold_backward-threshold-2", 754 },
  { "tile-dims-1", 755 },
  { "transpose-dim0-dim1-1", 756 },
  { "_mkldnn_transpose-dim0-dim1-1", 757 },
  { "one_hot-num_classes-1", 758 },
  { "one_hot-1", 759 },
  { "flip-dims-1", 760 },
  { "fliplr-1", 761 },
  { "flipud-1", 762 },
  { "roll-dims-shifts-1", 763 },
  { "roll-shifts-1", 764 },
  { "rot90-dims-k-1", 765 },
  { "rot90-k-1", 766 },
  { "rot90-1", 767 },
  { "trapezoid-dim-2", 768 },
  { "trapezoid-2", 769 },
  { "trapezoid-dim-dx-1", 770 },
  { "trapezoid-dx-1", 771 },
  { "trapezoid-1", 772 },
  { "trapz-dim-2", 773 },
  { "trapz-2", 774 },
  { "trapz-dim-dx-1", 775 },
  { "trapz-dx-1", 776 },
  { "trapz-1", 777 },
  { "_trilinear-expand1-expand2-expand3-sumdim-unroll_dim-3", 778 },
  { "_trilinear-expand1-expand2-expand3-sumdim-3", 779 },
  { "triplet_margin_loss-eps-margin-p-reduction-swap-3", 780 },
  { "triplet_margin_loss-eps-margin-p-swap-3", 781 },
  { "triplet_margin_loss-eps-margin-p-3", 782 },
  { "triplet_margin_loss-margin-p-3", 783 },
  { "triplet_margin_loss-margin-3", 784 },
  { "triplet_margin_loss-3", 785 },
  { "trunc-1", 786 },
  { "fix-1", 787 },
  { "type_as-2", 788 },
  { "_has_compatible_shallow_copy_type-2", 789 },
  { "_unique-return_inverse-sorted-1", 790 },
  { "_unique-sorted-1", 791 },
  { "_unique-1", 792 },
  { "unique_dim-dim-return_counts-return_inverse-sorted-1", 793 },
  { "unique_dim-dim-return_inverse-sorted-1", 794 },
  { "unique_dim-dim-sorted-1", 795 },
  { "unique_dim-dim-1", 796 },
  { "unique_consecutive-return_counts-return_inverse-1", 797 },
  { "unique_consecutive-return_inverse-1", 798 },
  { "unique_consecutive-1", 799 },
  { "unique_dim_consecutive-dim-return_counts-return_inverse-1", 800 },
  { "unique_dim_consecutive-dim-return_inverse-1", 801 },
  { "unique_dim_consecutive-dim-1", 802 },
  { "_unique2-return_counts-return_inverse-sorted-1", 803 },
  { "_unique2-return_inverse-sorted-1", 804 },
  { "_unique2-sorted-1", 805 },
  { "_unique2-1", 806 },
  { "_unsafe_view-size-1", 807 },
  { "unsqueeze-dim-1", 808 },
  { "vander-1", 809 },
  { "var-unbiased-1", 810 },
  { "var-1", 811 },
  { "var-dim-keepdim-unbiased-1", 812 },
  { "var-dim-unbiased-1", 813 },
  { "var-dim-1", 814 },
  { "var_mean-unbiased-1", 815 },
  { "var_mean-1", 816 },
  { "var_mean-dim-keepdim-unbiased-1", 817 },
  { "var_mean-dim-unbiased-1", 818 },
  { "var_mean-dim-1", 819 },
  { "view_as-2", 820 },
  { "where-3", 821 },
  { "where-self-2", 822 },
  { "where-other-2", 823 },
  { "where-other-self-1", 824 },
  { "where-1", 825 },
  { "_s_where-3", 826 },
  { "norm_except_dim-dim-pow-1", 827 },
  { "norm_except_dim-pow-1", 828 },
  { "norm_except_dim-1", 829 },
  { "_weight_norm-dim-2", 830 },
  { "_weight_norm-2", 831 },
  { "_weight_norm_cuda_interface-dim-2", 832 },
  { "_weight_norm_cuda_interface-2", 833 },
  { "_weight_norm_cuda_interface_backward-dim-4", 834 },
  { "_weight_norm_differentiable_backward-dim-4", 835 },
  { "_standard_gamma_grad-2", 836 },
  { "_standard_gamma-1", 837 },
  { "_dirichlet_grad-3", 838 },
  { "_sample_dirichlet-1", 839 },
  { "poisson-1", 840 },
  { "binomial-2", 841 },
  { "native_norm-p-1", 842 },
  { "native_norm-1", 843 },
  { "_sparse_sum-1", 844 },
  { "_sparse_sum-dim-1", 845 },
  { "_sparse_sum_backward-dim-2", 846 },
  { "_sparse_softmax-dim-1", 847 },
  { "_sparse_softmax-dim-half_to_float-1", 848 },
  { "_sparse_softmax_backward_data-dim-3", 849 },
  { "_sparse_log_softmax-dim-1", 850 },
  { "_sparse_log_softmax-dim-half_to_float-1", 851 },
  { "_sparse_log_softmax_backward_data-dim-3", 852 },
  { "norm-p-1", 853 },
  { "norm-1", 854 },
  { "frexp-1", 855 },
  { "frobenius_norm-1", 856 },
  { "frobenius_norm-dim-keepdim-1", 857 },
  { "frobenius_norm-dim-1", 858 },
  { "nuclear_norm-keepdim-1", 859 },
  { "nuclear_norm-1", 860 },
  { "nuclear_norm-dim-keepdim-1", 861 },
  { "nuclear_norm-dim-1", 862 },
  { "clone-1", 863 },
  { "positive-1", 864 },
  { "sub-alpha-2", 865 },
  { "sub-2", 866 },
  { "sub-alpha-other-1", 867 },
  { "sub-other-1", 868 },
  { "subtract-alpha-2", 869 },
  { "subtract-2", 870 },
  { "subtract-alpha-other-1", 871 },
  { "subtract-other-1", 872 },
  { "rsub-alpha-2", 873 },
  { "rsub-2", 874 },
  { "heaviside-2", 875 },
  { "rsub-alpha-other-1", 876 },
  { "rsub-other-1", 877 },
  { "_sparse_addmm-alpha-beta-3", 878 },
  { "_sparse_addmm-beta-3", 879 },
  { "_sparse_addmm-3", 880 },
  { "addmm-alpha-beta-3", 881 },
  { "addmm-beta-3", 882 },
  { "addmm-3", 883 },
  { "sparse_mask-2", 884 },
  { "_to_cpu-*", 885 },
  { "to_dense-1", 886 },
  { "to_dense_backward-2", 887 },
  { "sparse_dim-1", 888 },
  { "_dimI-1", 889 },
  { "dense_dim-1", 890 },
  { "_dimV-1", 891 },
  { "_nnz-1", 892 },
  { "coalesce-1", 893 },
  { "_coalesce-1", 894 },
  { "is_coalesced-1", 895 },
  { "_indices-1", 896 },
  { "_values-1", 897 },
  { "indices-1", 898 },
  { "values-1", 899 },
  { "crow_indices-1", 900 },
  { "col_indices-1", 901 },
  { "hspmm-2", 902 },
  { "unbind-dim-1", 903 },
  { "unbind-1", 904 },
  { "to_sparse-sparse_dim-1", 905 },
  { "to_sparse-1", 906 },
  { "to_mkldnn-1", 907 },
  { "mkldnn_reorder_conv2d_weight-dilation-groups-padding-stride-1", 908 },
  { "mkldnn_reorder_conv2d_weight-dilation-padding-stride-1", 909 },
  { "mkldnn_reorder_conv2d_weight-padding-stride-1", 910 },
  { "mkldnn_reorder_conv2d_weight-padding-1", 911 },
  { "mkldnn_reorder_conv2d_weight-1", 912 },
  { "mkldnn_reorder_conv3d_weight-dilation-groups-padding-stride-1", 913 },
  { "mkldnn_reorder_conv3d_weight-dilation-padding-stride-1", 914 },
  { "mkldnn_reorder_conv3d_weight-padding-stride-1", 915 },
  { "mkldnn_reorder_conv3d_weight-padding-1", 916 },
  { "mkldnn_reorder_conv3d_weight-1", 917 },
  { "to_mkldnn_backward-2", 918 },
  { "dequantize-1", 919 },
  { "dequantize-*", 920 },
  { "q_zero_point-1", 921 },
  { "q_per_channel_scales-1", 922 },
  { "q_per_channel_zero_points-1", 923 },
  { "q_per_channel_axis-1", 924 },
  { "int_repr-1", 925 },
  { "_make_per_tensor_quantized_tensor-scale-zero_point-1", 926 },
  { "_make_per_channel_quantized_tensor-axis-3", 927 },
  { "fake_quantize_per_tensor_affine-quant_max-quant_min-scale-zero_point-1", 928 },
  { "fake_quantize_per_tensor_affine-quant_max-quant_min-3", 929 },
  { "fake_quantize_per_tensor_affine_cachemask-quant_max-quant_min-scale-zero_point-1", 930 },
  { "_fake_quantize_per_tensor_affine_cachemask_tensor_qparams-quant_max-quant_min-4", 931 },
  { "fake_quantize_per_tensor_affine_cachemask_backward-2", 932 },
  { "_fake_quantize_learnable_per_tensor_affine-grad_factor-quant_max-quant_min-3", 933 },
  { "_fake_quantize_learnable_per_tensor_affine-quant_max-quant_min-3", 934 },
  { "_fake_quantize_learnable_per_tensor_affine_backward-grad_factor-quant_max-quant_min-4", 935 },
  { "_fake_quantize_learnable_per_tensor_affine_backward-quant_max-quant_min-4", 936 },
  { "fake_quantize_per_channel_affine-axis-quant_max-quant_min-3", 937 },
  { "fake_quantize_per_channel_affine_cachemask-axis-quant_max-quant_min-3", 938 },
  { "fake_quantize_per_channel_affine_cachemask_backward-2", 939 },
  { "_fake_quantize_learnable_per_channel_affine-axis-grad_factor-quant_max-quant_min-3", 940 },
  { "_fake_quantize_learnable_per_channel_affine-axis-quant_max-quant_min-3", 941 },
  { "_fake_quantize_learnable_per_channel_affine_backward-axis-grad_factor-quant_max-quant_min-4", 942 },
  { "_fake_quantize_learnable_per_channel_affine_backward-axis-quant_max-quant_min-4", 943 },
  { "fused_moving_avg_obs_fake_quant-averaging_const-ch_axis-per_row_fake_quant-quant_max-quant_min-symmetric_quant-7", 944 },
  { "fused_moving_avg_obs_fake_quant-averaging_const-ch_axis-per_row_fake_quant-quant_max-quant_min-7", 945 },
  { "fused_moving_avg_obs_fake_quant-averaging_const-ch_axis-quant_max-quant_min-7", 946 },
  { "_fused_moving_avg_obs_fq_helper-averaging_const-ch_axis-per_row_fake_quant-quant_max-quant_min-symmetric_quant-7", 947 },
  { "_fused_moving_avg_obs_fq_helper-averaging_const-ch_axis-per_row_fake_quant-quant_max-quant_min-7", 948 },
  { "_fused_moving_avg_obs_fq_helper-averaging_const-ch_axis-quant_max-quant_min-7", 949 },
  { "_saturate_weight_to_fp16-1", 950 },
  { "choose_qparams_optimized-bit_width-n_bins-numel-ratio-1", 951 },
  { "meshgrid-*", 952 },
  { "cartesian_prod-*", 953 },
  { "combinations-r-with_replacement-1", 954 },
  { "combinations-r-1", 955 },
  { "combinations-1", 956 },
  { "item-1", 957 },
  { "_local_scalar_dense-1", 958 },
  { "_thnn_fused_lstm_cell-5", 959 },
  { "_thnn_fused_lstm_cell-4", 960 },
  { "_thnn_fused_lstm_cell-3", 961 },
  { "_thnn_fused_lstm_cell_backward-has_bias-5", 962 },
  { "_thnn_differentiable_lstm_cell_backward-8", 963 },
  { "_thnn_fused_gru_cell-5", 964 },
  { "_thnn_fused_gru_cell-4", 965 },
  { "_thnn_fused_gru_cell-3", 966 },
  { "_thnn_fused_gru_cell_backward-has_bias-2", 967 },
  { "_thnn_differentiable_gru_cell_backward-6", 968 },
  { "lstm-batch_first-bidirectional-dropout-has_biases-num_layers-train-*", 969 },
  { "lstm-bidirectional-dropout-has_biases-num_layers-train-*", 970 },
  { "gru-batch_first-bidirectional-dropout-has_biases-num_layers-train-*", 971 },
  { "gru-bidirectional-dropout-has_biases-num_layers-train-*", 972 },
  { "rnn_tanh-batch_first-bidirectional-dropout-has_biases-num_layers-train-*", 973 },
  { "rnn_tanh-bidirectional-dropout-has_biases-num_layers-train-*", 974 },
  { "rnn_relu-batch_first-bidirectional-dropout-has_biases-num_layers-train-*", 975 },
  { "rnn_relu-bidirectional-dropout-has_biases-num_layers-train-*", 976 },
  { "lstm_cell-*", 977 },
  { "gru_cell-6", 978 },
  { "gru_cell-5", 979 },
  { "gru_cell-4", 980 },
  { "rnn_tanh_cell-6", 981 },
  { "rnn_tanh_cell-5", 982 },
  { "rnn_tanh_cell-4", 983 },
  { "rnn_relu_cell-6", 984 },
  { "rnn_relu_cell-5", 985 },
  { "rnn_relu_cell-4", 986 },
  { "quantized_lstm_cell-scale_hh-scale_ih-zero_point_hh-zero_point_ih-*", 987 },
  { "quantized_gru_cell-scale_hh-scale_ih-zero_point_hh-zero_point_ih-10", 988 },
  { "quantized_rnn_relu_cell-scale_hh-scale_ih-zero_point_hh-zero_point_ih-10", 989 },
  { "quantized_rnn_tanh_cell-scale_hh-scale_ih-zero_point_hh-zero_point_ih-10", 990 },
  { "_pack_padded_sequence-batch_first-2", 991 },
  { "_pack_padded_sequence_backward-batch_first-input_size-2", 992 },
  { "_pad_packed_sequence-batch_first-padding_value-total_length-2", 993 },
  { "is_set_to-2", 994 },
  { "masked_fill-value-2", 995 },
  { "masked_fill-3", 996 },
  { "masked_scatter-3", 997 },
  { "view-size-1", 998 },
  { "put-accumulate-3", 999 },
  { "put-3", 1000 },
  { "index_add-dim-3", 1001 },
  { "index_add-alpha-dim-3", 1002 },
  { "index_fill-dim-value-2", 1003 },
  { "index_fill-dim-3", 1004 },
  { "scatter-dim-3", 1005 },
  { "scatter-dim-value-2", 1006 },
  { "scatter_add-dim-3", 1007 },
  { "bitwise_and-other-1", 1008 },
  { "bitwise_and-2", 1009 },
  { "__and__-other-1", 1010 },
  { "__and__-2", 1011 },
  { "bitwise_or-other-1", 1012 },
  { "bitwise_or-2", 1013 },
  { "__or__-other-1", 1014 },
  { "__or__-2", 1015 },
  { "bitwise_xor-other-1", 1016 },
  { "bitwise_xor-2", 1017 },
  { "__xor__-other-1", 1018 },
  { "__xor__-2", 1019 },
  { "__lshift__-other-1", 1020 },
  { "__lshift__-2", 1021 },
  { "bitwise_left_shift-2", 1022 },
  { "bitwise_left_shift-other-1", 1023 },
  { "bitwise_left_shift-self-1", 1024 },
  { "__rshift__-other-1", 1025 },
  { "__rshift__-2", 1026 },
  { "bitwise_right_shift-2", 1027 },
  { "bitwise_right_shift-other-1", 1028 },
  { "bitwise_right_shift-self-1", 1029 },
  { "addbmm-alpha-beta-3", 1030 },
  { "addbmm-beta-3", 1031 },
  { "addbmm-3", 1032 },
  { "diag-diagonal-1", 1033 },
  { "diag-1", 1034 },
  { "diag_backward-diagonal-input_sizes-1", 1035 },
  { "cross-2", 1036 },
  { "triu-diagonal-1", 1037 },
  { "triu-1", 1038 },
  { "tril-diagonal-1", 1039 },
  { "tril-1", 1040 },
  { "trace-1", 1041 },
  { "trace_backward-sizes-1", 1042 },
  { "ne-other-1", 1043 },
  { "ne-2", 1044 },
  { "not_equal-other-1", 1045 },
  { "not_equal-2", 1046 },
  { "eq-other-1", 1047 },
  { "eq-2", 1048 },
  { "ge-other-1", 1049 },
  { "ge-2", 1050 },
  { "greater_equal-other-1", 1051 },
  { "greater_equal-2", 1052 },
  { "le-other-1", 1053 },
  { "le-2", 1054 },
  { "less_equal-other-1", 1055 },
  { "less_equal-2", 1056 },
  { "gt-other-1", 1057 },
  { "gt-2", 1058 },
  { "greater-other-1", 1059 },
  { "greater-2", 1060 },
  { "lt-other-1", 1061 },
  { "lt-2", 1062 },
  { "less-other-1", 1063 },
  { "less-2", 1064 },
  { "take-2", 1065 },
  { "take_along_dim-2", 1066 },
  { "index_select-dim-2", 1067 },
  { "index_select_backward-dim-self_sizes-2", 1068 },
  { "masked_select-2", 1069 },
  { "masked_select_backward-3", 1070 },
  { "nonzero-1", 1071 },
  { "nonzero_numpy-1", 1072 },
  { "gather-dim-sparse_grad-2", 1073 },
  { "gather-dim-2", 1074 },
  { "gather_backward-dim-sparse_grad-3", 1075 },
  { "_gather_sparse_backward-dim-3", 1076 },
  { "addcmul-value-3", 1077 },
  { "addcmul-3", 1078 },
  { "addcdiv-value-3", 1079 },
  { "addcdiv-3", 1080 },
  { "cross_entropy_loss-ignore_index-label_smoothing-reduction-3", 1081 },
  { "cross_entropy_loss-ignore_index-reduction-3", 1082 },
  { "cross_entropy_loss-reduction-3", 1083 },
  { "cross_entropy_loss-3", 1084 },
  { "cross_entropy_loss-2", 1085 },
  { "lstsq-2", 1086 },
  { "triangular_solve-transpose-unitriangular-upper-2", 1087 },
  { "triangular_solve-transpose-upper-2", 1088 },
  { "triangular_solve-upper-2", 1089 },
  { "triangular_solve-2", 1090 },
  { "symeig-eigenvectors-upper-1", 1091 },
  { "symeig-eigenvectors-1", 1092 },
  { "symeig-1", 1093 },
  { "_symeig_helper-eigenvectors-upper-1", 1094 },
  { "eig-eigenvectors-1", 1095 },
  { "eig-1", 1096 },
  { "svd-compute_uv-some-1", 1097 },
  { "svd-some-1", 1098 },
  { "svd-1", 1099 },
  { "_svd_helper-compute_uv-some-1", 1100 },
  { "swapaxes-axis0-axis1-1", 1101 },
  { "swapdims-dim0-dim1-1", 1102 },
  { "cholesky-upper-1", 1103 },
  { "cholesky-1", 1104 },
  { "cholesky_solve-upper-2", 1105 },
  { "cholesky_solve-2", 1106 },
  { "_cholesky_solve_helper-upper-2", 1107 },
  { "solve-2", 1108 },
  { "_solve_helper-2", 1109 },
  { "cholesky_inverse-upper-1", 1110 },
  { "cholesky_inverse-1", 1111 },
  { "qr-some-1", 1112 },
  { "qr-1", 1113 },
  { "geqrf-1", 1114 },
  { "orgqr-2", 1115 },
  { "ormqr-left-transpose-3", 1116 },
  { "ormqr-left-3", 1117 },
  { "ormqr-3", 1118 },
  { "_lu_with_info-check_errors-pivot-1", 1119 },
  { "_lu_with_info-pivot-1", 1120 },
  { "_lu_with_info-1", 1121 },
  { "lu_solve-3", 1122 },
  { "lu_unpack-unpack_data-unpack_pivots-2", 1123 },
  { "lu_unpack-unpack_data-2", 1124 },
  { "lu_unpack-2", 1125 },
  { "multinomial-num_samples-replacement-1", 1126 },
  { "multinomial-num_samples-1", 1127 },
  { "lgamma-1", 1128 },
  { "digamma-1", 1129 },
  { "polygamma-n-1", 1130 },
  { "erfinv-1", 1131 },
  { "i0-1", 1132 },
  { "sign-1", 1133 },
  { "signbit-1", 1134 },
  { "dist-p-2", 1135 },
  { "dist-2", 1136 },
  { "atan2-2", 1137 },
  { "lerp-weight-2", 1138 },
  { "lerp-3", 1139 },
  { "histc-bins-max-min-1", 1140 },
  { "histc-bins-min-1", 1141 },
  { "histc-bins-1", 1142 },
  { "histc-1", 1143 },
  { "histogram-density-3", 1144 },
  { "histogram-3", 1145 },
  { "histogram-2", 1146 },
  { "histogram-bins-1", 1147 },
  { "histogram-1", 1148 },
  { "fmod-other-1", 1149 },
  { "fmod-2", 1150 },
  { "hypot-2", 1151 },
  { "igamma-2", 1152 },
  { "igammac-2", 1153 },
  { "nextafter-2", 1154 },
  { "remainder-other-1", 1155 },
  { "remainder-2", 1156 },
  { "remainder-self-1", 1157 },
  { "min-1", 1158 },
  { "fmin-2", 1159 },
  { "max-1", 1160 },
  { "fmax-2", 1161 },
  { "maximum-2", 1162 },
  { "max-2", 1163 },
  { "minimum-2", 1164 },
  { "min-2", 1165 },
  { "quantile-q-1", 1166 },
  { "quantile-2", 1167 },
  { "nanquantile-q-1", 1168 },
  { "nanquantile-2", 1169 },
  { "sort-descending-dim-1", 1170 },
  { "sort-dim-1", 1171 },
  { "sort-1", 1172 },
  { "msort-1", 1173 },
  { "argsort-descending-dim-1", 1174 },
  { "argsort-dim-1", 1175 },
  { "argsort-1", 1176 },
  { "topk-dim-k-largest-sorted-1", 1177 },
  { "topk-dim-k-largest-1", 1178 },
  { "topk-dim-k-1", 1179 },
  { "topk-k-1", 1180 },
  { "all-1", 1181 },
  { "any-1", 1182 },
  { "renorm-dim-maxnorm-p-1", 1183 },
  { "unfold-dimension-size-step-1", 1184 },
  { "unfold_backward-dim-input_sizes-size-step-1", 1185 },
  { "equal-2", 1186 },
  { "pow-2", 1187 },
  { "pow-self-1", 1188 },
  { "pow-exponent-1", 1189 },
  { "float_power-2", 1190 },
  { "float_power-self-1", 1191 },
  { "float_power-exponent-1", 1192 },
  { "alias-1", 1193 },
  { "_cat-dim-*", 1194 },
  { "_cat-*", 1195 },
  { "_foreach_add-scalar-*", 1196 },
  { "_foreach_sub-scalar-*", 1197 },
  { "_foreach_mul-scalar-*", 1198 },
  { "_foreach_div-scalar-*", 1199 },
  { "_foreach_add-alpha-*", 1200 },
  { "_foreach_add-*", 1201 },
  { "_foreach_sub-alpha-*", 1202 },
  { "_foreach_sub-*", 1203 },
  { "_foreach_mul-*", 1204 },
  { "_foreach_div-*", 1205 },
  { "_foreach_exp-*", 1206 },
  { "_foreach_sqrt-*", 1207 },
  { "_foreach_abs-*", 1208 },
  { "_foreach_acos-*", 1209 },
  { "_foreach_asin-*", 1210 },
  { "_foreach_atan-*", 1211 },
  { "_foreach_ceil-*", 1212 },
  { "_foreach_cos-*", 1213 },
  { "_foreach_cosh-*", 1214 },
  { "_foreach_erf-*", 1215 },
  { "_foreach_erfc-*", 1216 },
  { "_foreach_expm1-*", 1217 },
  { "_foreach_floor-*", 1218 },
  { "_foreach_log-*", 1219 },
  { "_foreach_log10-*", 1220 },
  { "_foreach_log1p-*", 1221 },
  { "_foreach_log2-*", 1222 },
  { "_foreach_neg-*", 1223 },
  { "_foreach_tan-*", 1224 },
  { "_foreach_tanh-*", 1225 },
  { "_foreach_sin-*", 1226 },
  { "_foreach_sinh-*", 1227 },
  { "_foreach_round-*", 1228 },
  { "_foreach_lgamma-*", 1229 },
  { "_foreach_frac-*", 1230 },
  { "_foreach_reciprocal-*", 1231 },
  { "_foreach_sigmoid-*", 1232 },
  { "_foreach_trunc-*", 1233 },
  { "_foreach_addcdiv-value-*", 1234 },
  { "_foreach_addcdiv-*", 1235 },
  { "_foreach_addcmul-value-*", 1236 },
  { "_foreach_addcmul-*", 1237 },
  { "_foreach_maximum-*", 1238 },
  { "_foreach_minimum-*", 1239 },
  { "bucketize-out_int32-right-2", 1240 },
  { "bucketize-out_int32-2", 1241 },
  { "bucketize-2", 1242 },
  { "bucketize-out_int32-right-self-1", 1243 },
  { "bucketize-out_int32-self-1", 1244 },
  { "bucketize-self-1", 1245 },
  { "searchsorted-out_int32-right-2", 1246 },
  { "searchsorted-out_int32-2", 1247 },
  { "searchsorted-2", 1248 },
  { "searchsorted-out_int32-right-self-1", 1249 },
  { "searchsorted-out_int32-self-1", 1250 },
  { "searchsorted-self-1", 1251 },
  { "_convert_indices_from_coo_to_csr-out_int32-size-1", 1252 },
  { "_convert_indices_from_coo_to_csr-size-1", 1253 },
  { "mse_loss-reduction-2", 1254 },
  { "mse_loss-2", 1255 },
  { "mse_loss_backward-reduction-3", 1256 },
  { "l1_loss-reduction-2", 1257 },
  { "l1_loss-2", 1258 },
  { "l1_loss_backward-reduction-3", 1259 },
  { "multi_margin_loss-margin-p-reduction-3", 1260 },
  { "multi_margin_loss-margin-p-3", 1261 },
  { "multi_margin_loss-margin-p-2", 1262 },
  { "multi_margin_loss-p-2", 1263 },
  { "multi_margin_loss-2", 1264 },
  { "multi_margin_loss_backward-margin-p-reduction-4", 1265 },
  { "multi_margin_loss_backward-margin-p-4", 1266 },
  { "multi_margin_loss_backward-margin-p-3", 1267 },
  { "multilabel_margin_loss-reduction-2", 1268 },
  { "multilabel_margin_loss-2", 1269 },
  { "multilabel_margin_loss_forward-reduction-2", 1270 },
  { "multilabel_margin_loss_backward-reduction-4", 1271 },
  { "nll_loss_nd-ignore_index-reduction-3", 1272 },
  { "nll_loss_nd-reduction-3", 1273 },
  { "nll_loss_nd-3", 1274 },
  { "nll_loss_nd-2", 1275 },
  { "nll_loss-ignore_index-reduction-3", 1276 },
  { "nll_loss-reduction-3", 1277 },
  { "nll_loss-3", 1278 },
  { "nll_loss-2", 1279 },
  { "nll_loss_forward-ignore_index-reduction-3", 1280 },
  { "nll_loss_backward-ignore_index-reduction-5", 1281 },
  { "nll_loss2d-ignore_index-reduction-3", 1282 },
  { "nll_loss2d-reduction-3", 1283 },
  { "nll_loss2d-3", 1284 },
  { "nll_loss2d-2", 1285 },
  { "nll_loss2d_forward-ignore_index-reduction-3", 1286 },
  { "nll_loss2d_backward-ignore_index-reduction-5", 1287 },
  { "smooth_l1_loss-beta-reduction-2", 1288 },
  { "smooth_l1_loss-reduction-2", 1289 },
  { "smooth_l1_loss-2", 1290 },
  { "smooth_l1_loss_backward-beta-reduction-3", 1291 },
  { "huber_loss-delta-reduction-2", 1292 },
  { "huber_loss-reduction-2", 1293 },
  { "huber_loss-2", 1294 },
  { "huber_loss_backward-delta-reduction-3", 1295 },
  { "soft_margin_loss-reduction-2", 1296 },
  { "soft_margin_loss-2", 1297 },
  { "soft_margin_loss_backward-reduction-3", 1298 },
  { "elu-alpha-input_scale-scale-1", 1299 },
  { "elu-alpha-scale-1", 1300 },
  { "elu-alpha-1", 1301 },
  { "elu-1", 1302 },
  { "elu_backward-alpha-input_scale-is_result-scale-2", 1303 },
  { "glu-dim-1", 1304 },
  { "glu-1", 1305 },
  { "glu_backward-dim-2", 1306 },
  { "hardsigmoid-1", 1307 },
  { "hardsigmoid_backward-2", 1308 },
  { "hardtanh-max_val-min_val-1", 1309 },
  { "hardtanh-min_val-1", 1310 },
  { "hardtanh-1", 1311 },
  { "hardtanh_backward-max_val-min_val-2", 1312 },
  { "hardswish-1", 1313 },
  { "hardswish_backward-2", 1314 },
  { "leaky_relu-negative_slope-1", 1315 },
  { "leaky_relu-1", 1316 },
  { "leaky_relu_backward-negative_slope-self_is_result-2", 1317 },
  { "log_sigmoid-1", 1318 },
  { "log_sigmoid_forward-1", 1319 },
  { "log_sigmoid_backward-3", 1320 },
  { "rrelu_with_noise-lower-training-upper-2", 1321 },
  { "rrelu_with_noise-lower-upper-2", 1322 },
  { "rrelu_with_noise-lower-2", 1323 },
  { "rrelu_with_noise-2", 1324 },
  { "rrelu_with_noise_backward-lower-self_is_result-training-upper-3", 1325 },
  { "softplus-beta-threshold-1", 1326 },
  { "softplus-beta-1", 1327 },
  { "softplus-1", 1328 },
  { "softplus_backward-beta-threshold-3", 1329 },
  { "softshrink-lambd-1", 1330 },
  { "softshrink-1", 1331 },
  { "softshrink_backward-lambd-2", 1332 },
  { "adaptive_avg_pool2d-output_size-1", 1333 },
  { "mkldnn_adaptive_avg_pool2d-output_size-1", 1334 },
  { "mkldnn_adaptive_avg_pool2d_backward-2", 1335 },
  { "_adaptive_avg_pool2d-output_size-1", 1336 },
  { "_adaptive_avg_pool2d_backward-2", 1337 },
  { "adaptive_avg_pool3d-output_size-1", 1338 },
  { "_adaptive_avg_pool3d-output_size-1", 1339 },
  { "_adaptive_avg_pool3d_backward-2", 1340 },
  { "adaptive_max_pool2d-output_size-1", 1341 },
  { "adaptive_max_pool2d_backward-3", 1342 },
  { "adaptive_max_pool3d-output_size-1", 1343 },
  { "adaptive_max_pool3d_backward-3", 1344 },
  { "avg_pool2d-ceil_mode-count_include_pad-kernel_size-padding-stride-1", 1345 },
  { "avg_pool2d-ceil_mode-kernel_size-padding-stride-1", 1346 },
  { "avg_pool2d-kernel_size-padding-stride-1", 1347 },
  { "avg_pool2d-kernel_size-stride-1", 1348 },
  { "avg_pool2d-kernel_size-1", 1349 },
  { "avg_pool3d-ceil_mode-count_include_pad-kernel_size-padding-stride-1", 1350 },
  { "avg_pool3d-ceil_mode-kernel_size-padding-stride-1", 1351 },
  { "avg_pool3d-kernel_size-padding-stride-1", 1352 },
  { "avg_pool3d-kernel_size-stride-1", 1353 },
  { "avg_pool3d-kernel_size-1", 1354 },
  { "fractional_max_pool2d-kernel_size-output_size-2", 1355 },
  { "fractional_max_pool2d_backward-kernel_size-output_size-3", 1356 },
  { "fractional_max_pool3d-kernel_size-output_size-2", 1357 },
  { "fractional_max_pool3d_backward-kernel_size-output_size-3", 1358 },
  { "max_pool2d_with_indices-ceil_mode-dilation-kernel_size-padding-stride-1", 1359 },
  { "max_pool2d_with_indices-dilation-kernel_size-padding-stride-1", 1360 },
  { "max_pool2d_with_indices-kernel_size-padding-stride-1", 1361 },
  { "max_pool2d_with_indices-kernel_size-stride-1", 1362 },
  { "max_pool2d_with_indices-kernel_size-1", 1363 },
  { "max_pool2d_with_indices_backward-ceil_mode-dilation-kernel_size-padding-stride-3", 1364 },
  { "max_pool3d_with_indices-ceil_mode-dilation-kernel_size-padding-stride-1", 1365 },
  { "max_pool3d_with_indices-dilation-kernel_size-padding-stride-1", 1366 },
  { "max_pool3d_with_indices-kernel_size-padding-stride-1", 1367 },
  { "max_pool3d_with_indices-kernel_size-stride-1", 1368 },
  { "max_pool3d_with_indices-kernel_size-1", 1369 },
  { "max_pool3d_with_indices_backward-ceil_mode-dilation-kernel_size-padding-stride-3", 1370 },
  { "max_unpool2d-output_size-2", 1371 },
  { "max_unpool2d_backward-output_size-3", 1372 },
  { "max_unpool3d-output_size-padding-stride-2", 1373 },
  { "max_unpool3d_backward-output_size-padding-stride-3", 1374 },
  { "reflection_pad1d-padding-1", 1375 },
  { "reflection_pad1d_backward-padding-2", 1376 },
  { "reflection_pad2d-padding-1", 1377 },
  { "reflection_pad2d_backward-padding-2", 1378 },
  { "reflection_pad3d-padding-1", 1379 },
  { "reflection_pad3d_backward-padding-2", 1380 },
  { "replication_pad1d-padding-1", 1381 },
  { "replication_pad1d_backward-padding-2", 1382 },
  { "replication_pad2d-padding-1", 1383 },
  { "replication_pad2d_backward-padding-2", 1384 },
  { "replication_pad3d-padding-1", 1385 },
  { "replication_pad3d_backward-padding-2", 1386 },
  { "upsample_linear1d-align_corners-output_size-1", 1387 },
  { "upsample_linear1d_backward-align_corners-input_size-output_size-1", 1388 },
  { "upsample_bilinear2d-align_corners-output_size-1", 1389 },
  { "upsample_bilinear2d_backward-align_corners-input_size-output_size-1", 1390 },
  { "upsample_bicubic2d-align_corners-output_size-1", 1391 },
  { "upsample_bicubic2d_backward-align_corners-input_size-output_size-1", 1392 },
  { "upsample_trilinear3d-align_corners-output_size-1", 1393 },
  { "upsample_trilinear3d_backward-align_corners-input_size-output_size-1", 1394 },
  { "upsample_nearest1d-output_size-1", 1395 },
  { "upsample_nearest1d_backward-input_size-output_size-1", 1396 },
  { "upsample_nearest2d-output_size-1", 1397 },
  { "upsample_nearest2d_backward-input_size-output_size-1", 1398 },
  { "upsample_nearest3d-output_size-1", 1399 },
  { "upsample_nearest3d_backward-input_size-output_size-1", 1400 },
  { "sigmoid_backward-2", 1401 },
  { "logit_backward-2", 1402 },
  { "tanh_backward-2", 1403 },
  { "slow_conv_transpose2d-dilation-kernel_size-output_padding-padding-stride-3", 1404 },
  { "slow_conv_transpose2d-kernel_size-output_padding-padding-stride-3", 1405 },
  { "slow_conv_transpose2d-kernel_size-padding-stride-3", 1406 },
  { "slow_conv_transpose2d-kernel_size-stride-3", 1407 },
  { "slow_conv_transpose2d-kernel_size-3", 1408 },
  { "slow_conv_transpose2d-kernel_size-2", 1409 },
  { "slow_conv_transpose2d_backward-dilation-kernel_size-output_mask-output_padding-padding-stride-5", 1410 },
  { "slow_conv_transpose3d-dilation-kernel_size-output_padding-padding-stride-3", 1411 },
  { "slow_conv_transpose3d-kernel_size-output_padding-padding-stride-3", 1412 },
  { "slow_conv_transpose3d-kernel_size-padding-stride-3", 1413 },
  { "slow_conv_transpose3d-kernel_size-stride-3", 1414 },
  { "slow_conv_transpose3d-kernel_size-3", 1415 },
  { "slow_conv_transpose3d-kernel_size-2", 1416 },
  { "slow_conv_transpose3d_backward-dilation-kernel_size-output_mask-output_padding-padding-stride-5", 1417 },
  { "thnn_conv2d-kernel_size-padding-stride-3", 1418 },
  { "thnn_conv2d-kernel_size-stride-3", 1419 },
  { "thnn_conv2d-kernel_size-3", 1420 },
  { "thnn_conv2d-kernel_size-2", 1421 },
  { "thnn_conv2d_forward-kernel_size-padding-stride-3", 1422 },
  { "thnn_conv2d_backward-kernel_size-output_mask-padding-stride-5", 1423 },
  { "_conv_depthwise2d-dilation-kernel_size-padding-stride-3", 1424 },
  { "_conv_depthwise2d_backward-dilation-kernel_size-output_mask-padding-stride-3", 1425 },
  { "conv_depthwise3d-dilation-kernel_size-padding-stride-3", 1426 },
  { "conv_depthwise3d_backward-dilation-kernel_size-output_mask-padding-stride-3", 1427 },
  { "slow_conv3d-kernel_size-padding-stride-3", 1428 },
  { "slow_conv3d-kernel_size-stride-3", 1429 },
  { "slow_conv3d-kernel_size-3", 1430 },
  { "slow_conv3d-kernel_size-2", 1431 },
  { "slow_conv3d_forward-kernel_size-padding-stride-3", 1432 },
  { "slow_conv3d_backward-kernel_size-output_mask-padding-stride-5", 1433 },
  { "slow_conv_dilated2d-dilation-kernel_size-padding-stride-3", 1434 },
  { "slow_conv_dilated2d-kernel_size-padding-stride-3", 1435 },
  { "slow_conv_dilated2d-kernel_size-stride-3", 1436 },
  { "slow_conv_dilated2d-kernel_size-3", 1437 },
  { "slow_conv_dilated2d-kernel_size-2", 1438 },
  { "slow_conv_dilated2d_backward-dilation-kernel_size-output_mask-padding-stride-3", 1439 },
  { "slow_conv_dilated3d-dilation-kernel_size-padding-stride-3", 1440 },
  { "slow_conv_dilated3d-kernel_size-padding-stride-3", 1441 },
  { "slow_conv_dilated3d-kernel_size-stride-3", 1442 },
  { "slow_conv_dilated3d-kernel_size-3", 1443 },
  { "slow_conv_dilated3d-kernel_size-2", 1444 },
  { "slow_conv_dilated3d_backward-dilation-kernel_size-output_mask-padding-stride-3", 1445 },
  { "col2im-dilation-kernel_size-output_size-padding-stride-1", 1446 },
  { "col2im_backward-dilation-kernel_size-padding-stride-1", 1447 },
  { "column_stack-*", 1448 },
  { "im2col-dilation-kernel_size-padding-stride-1", 1449 },
  { "im2col_backward-dilation-input_size-kernel_size-padding-stride-1", 1450 },
  { "isfinite-1", 1451 },
  { "isinf-1", 1452 },
  { "isposinf-1", 1453 },
  { "isneginf-1", 1454 },
  { "_add_batch_dim-batch_dim-level-1", 1455 },
  { "_remove_batch_dim-batch_size-level-out_dim-1", 1456 },
  { "special_entr-1", 1457 },
  { "special_ndtri-1", 1458 },
  { "special_expm1-1", 1459 },
  { "special_exp2-1", 1460 },
  { "special_psi-1", 1461 },
  { "special_digamma-1", 1462 },
  { "special_gammaln-1", 1463 },
  { "special_erf-1", 1464 },
  { "special_erfc-1", 1465 },
  { "special_erfcx-1", 1466 },
  { "special_erfinv-1", 1467 },
  { "special_ndtr-1", 1468 },
  { "special_xlog1py-2", 1469 },
  { "special_xlog1py-self-1", 1470 },
  { "special_xlog1py-other-1", 1471 },
  { "special_xlogy-2", 1472 },
  { "special_xlogy-self-1", 1473 },
  { "special_xlogy-other-1", 1474 },
  { "special_zeta-2", 1475 },
  { "special_zeta-self-1", 1476 },
  { "special_zeta-other-1", 1477 },
  { "special_i0-1", 1478 },
  { "special_i0e-1", 1479 },
  { "special_i1-1", 1480 },
  { "special_i1e-1", 1481 },
  { "special_logit-1", 1482 },
  { "special_polygamma-n-1", 1483 },
  { "special_logsumexp-dim-keepdim-1", 1484 },
  { "special_logsumexp-dim-1", 1485 },
  { "special_expit-1", 1486 },
  { "special_sinc-1", 1487 },
  { "special_round-1", 1488 },
  { "special_log1p-1", 1489 },
  { "special_log_softmax-dim-1", 1490 },
  { "special_gammainc-2", 1491 },
  { "special_gammaincc-2", 1492 },
  { "special_multigammaln-p-1", 1493 },
  { "fft_fft-1", 1494 },
  { "fft_ifft-1", 1495 },
  { "fft_rfft-1", 1496 },
  { "fft_irfft-1", 1497 },
  { "fft_hfft-1", 1498 },
  { "fft_ihfft-1", 1499 },
  { "fft_fft2-1", 1500 },
  { "fft_ifft2-1", 1501 },
  { "fft_rfft2-1", 1502 },
  { "fft_irfft2-1", 1503 },
  { "fft_fftn-1", 1504 },
  { "fft_ifftn-1", 1505 },
  { "fft_rfftn-1", 1506 },
  { "fft_irfftn-1", 1507 },
  { "fft_fftshift-1", 1508 },
  { "fft_ifftshift-1", 1509 },
  { "linalg_cholesky_ex-check_errors-upper-1", 1510 },
  { "linalg_cholesky_ex-upper-1", 1511 },
  { "linalg_cholesky_ex-1", 1512 },
  { "linalg_cholesky-upper-1", 1513 },
  { "linalg_cholesky-1", 1514 },
  { "linalg_det-1", 1515 },
  { "det-1", 1516 },
  { "_det_lu_based_helper-1", 1517 },
  { "_det_lu_based_helper_backward_helper-5", 1518 },
  { "linalg_lstsq-2", 1519 },
  { "linalg_matmul-2", 1520 },
  { "linalg_slogdet-1", 1521 },
  { "linalg_eig-1", 1522 },
  { "linalg_eigvals-1", 1523 },
  { "linalg_eigh-1", 1524 },
  { "linalg_eigvalsh-1", 1525 },
  { "linalg_householder_product-2", 1526 },
  { "linalg_inv_ex-check_errors-1", 1527 },
  { "linalg_inv_ex-1", 1528 },
  { "linalg_inv-1", 1529 },
  { "inner-2", 1530 },
  { "outer-2", 1531 },
  { "ger-2", 1532 },
  { "linalg_norm-1", 1533 },
  { "linalg_vector_norm-ord-1", 1534 },
  { "linalg_vector_norm-1", 1535 },
  { "linalg_matrix_norm-dim-keepdim-ord-1", 1536 },
  { "linalg_matrix_norm-dim-ord-1", 1537 },
  { "linalg_matrix_norm-ord-1", 1538 },
  { "linalg_matrix_norm-1", 1539 },
  { "linalg_svd-full_matrices-1", 1540 },
  { "linalg_svd-1", 1541 },
  { "linalg_svdvals-1", 1542 },
  { "linalg_cond-1", 1543 },
  { "linalg_pinv-hermitian-rcond-1", 1544 },
  { "linalg_pinv-rcond-1", 1545 },
  { "linalg_pinv-1", 1546 },
  { "linalg_pinv-hermitian-2", 1547 },
  { "linalg_pinv-2", 1548 },
  { "linalg_solve-2", 1549 },
  { "linalg_tensorinv-ind-1", 1550 },
  { "linalg_tensorinv-1", 1551 },
  { "linalg_tensorsolve-2", 1552 },
  { "linalg_qr-1", 1553 },
  { "linalg_matrix_power-n-1", 1554 },
  { "linalg_matrix_rank-1", 1555 },
  { "linalg_matrix_rank-hermitian-2", 1556 },
  { "linalg_matrix_rank-2", 1557 },
  { "linalg_multi_dot-*", 1558 },
  { "_test_serialization_subcmul-alpha-2", 1559 },
  { "_test_serialization_subcmul-2", 1560 },
  { "_test_string_default-1", 1561 },
  { "_test_ambiguous_defaults-a-b-1", 1562 },
  { "_test_ambiguous_defaults-a-1", 1563 },
  { "_test_ambiguous_defaults-1", 1564 },
  { "pad_sequence-batch_first-padding_value-*", 1565 },
  { "pad_sequence-batch_first-*", 1566 },
  { "pad_sequence-*", 1567 },
  { "flatten_dense_tensors-*", 1568 },
  { "unflatten_dense_tensors-*", 1569 },
};

namespace caffe2 {

using at::Half; // for AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, ...)

namespace internal {
TORCH_API at::Tensor index_with_uint8_handling(
    const at::Tensor& self,
    const torch::List<c10::optional<at::Tensor>>& indices);
}

template <class Context>
class ATenOp : public Operator<Context> {
 public:
  ATenOp(const OperatorDef& operator_def, Workspace* ws)
  : Operator<Context>(operator_def, ws) {
    VLOG(2) << "ATen OpDef: " << ProtoDebugString(operator_def) << "\n";
    switch(findImplementation(operator_def)) {
      case 0: // _cast_Byte
        implementation_0();
        break;
      case 1: // _cast_Byte
        implementation_1();
        break;
      case 2: // _cast_Char
        implementation_2();
        break;
      case 3: // _cast_Char
        implementation_3();
        break;
      case 4: // _cast_Double
        implementation_4();
        break;
      case 5: // _cast_Double
        implementation_5();
        break;
      case 6: // _cast_Float
        implementation_6();
        break;
      case 7: // _cast_Float
        implementation_7();
        break;
      case 8: // _cast_Int
        implementation_8();
        break;
      case 9: // _cast_Int
        implementation_9();
        break;
      case 10: // _cast_Long
        implementation_10();
        break;
      case 11: // _cast_Long
        implementation_11();
        break;
      case 12: // _cast_Short
        implementation_12();
        break;
      case 13: // _cast_Short
        implementation_13();
        break;
      case 14: // _cast_Half
        implementation_14();
        break;
      case 15: // _cast_Half
        implementation_15();
        break;
      case 16: // data
        implementation_16();
        break;
      case 17: // is_leaf
        implementation_17();
        break;
      case 18: // output_nr
        implementation_18();
        break;
      case 19: // _version
        implementation_19();
        break;
      case 20: // retains_grad
        implementation_20();
        break;
      case 21: // _fw_primal
        implementation_21();
        break;
      case 22: // _make_dual
        implementation_22();
        break;
      case 23: // _unpack_dual
        implementation_23();
        break;
      case 24: // align_as
        implementation_24();
        break;
      case 25: // align_tensors
        implementation_25();
        break;
      case 26: // _use_cudnn_ctc_loss
        implementation_26();
        break;
      case 27: // _cudnn_ctc_loss
        implementation_27();
        break;
      case 28: // _use_cudnn_rnn_flatten_weight
        implementation_28();
        break;
      case 29: // _cudnn_rnn_flatten_weight
        implementation_29();
        break;
      case 30: // _cudnn_rnn
        implementation_30();
        break;
      case 31: // _debug_has_internal_overlap
        implementation_31();
        break;
      case 32: // _fused_dropout
        implementation_32();
        break;
      case 33: // _masked_scale
        implementation_33();
        break;
      case 34: // _reshape_from_tensor
        implementation_34();
        break;
      case 35: // _shape_as_tensor
        implementation_35();
        break;
      case 36: // dropout
        implementation_36();
        break;
      case 37: // feature_dropout
        implementation_37();
        break;
      case 38: // alpha_dropout
        implementation_38();
        break;
      case 39: // feature_alpha_dropout
        implementation_39();
        break;
      case 40: // abs
        implementation_40();
        break;
      case 41: // absolute
        implementation_41();
        break;
      case 42: // angle
        implementation_42();
        break;
      case 43: // view_as_real
        implementation_43();
        break;
      case 44: // view_as_complex
        implementation_44();
        break;
      case 45: // sgn
        implementation_45();
        break;
      case 46: // real
        implementation_46();
        break;
      case 47: // imag
        implementation_47();
        break;
      case 48: // _conj
        implementation_48();
        break;
      case 49: // conj
        implementation_49();
        break;
      case 50: // _conj_physical
        implementation_50();
        break;
      case 51: // conj_physical
        implementation_51();
        break;
      case 52: // resolve_conj
        implementation_52();
        break;
      case 53: // resolve_neg
        implementation_53();
        break;
      case 54: // _neg_view
        implementation_54();
        break;
      case 55: // acos
        implementation_55();
        break;
      case 56: // arccos
        implementation_56();
        break;
      case 57: // avg_pool1d
        implementation_57();
        break;
      case 58: // avg_pool1d
        implementation_58();
        break;
      case 59: // avg_pool1d
        implementation_59();
        break;
      case 60: // avg_pool1d
        implementation_60();
        break;
      case 61: // avg_pool1d
        implementation_61();
        break;
      case 62: // adaptive_avg_pool1d
        implementation_62();
        break;
      case 63: // adaptive_max_pool1d
        implementation_63();
        break;
      case 64: // add
        implementation_64();
        break;
      case 65: // add
        implementation_65();
        break;
      case 66: // _add_relu
        implementation_66();
        break;
      case 67: // _add_relu
        implementation_67();
        break;
      case 68: // _add_relu
        implementation_68();
        break;
      case 69: // _add_relu
        implementation_69();
        break;
      case 70: // add
        implementation_70();
        break;
      case 71: // add
        implementation_71();
        break;
      case 72: // addmv
        implementation_72();
        break;
      case 73: // addmv
        implementation_73();
        break;
      case 74: // addmv
        implementation_74();
        break;
      case 75: // addr
        implementation_75();
        break;
      case 76: // addr
        implementation_76();
        break;
      case 77: // addr
        implementation_77();
        break;
      case 78: // affine_grid_generator
        implementation_78();
        break;
      case 79: // affine_grid_generator_backward
        implementation_79();
        break;
      case 80: // all
        implementation_80();
        break;
      case 81: // all
        implementation_81();
        break;
      case 82: // allclose
        implementation_82();
        break;
      case 83: // allclose
        implementation_83();
        break;
      case 84: // allclose
        implementation_84();
        break;
      case 85: // allclose
        implementation_85();
        break;
      case 86: // any
        implementation_86();
        break;
      case 87: // any
        implementation_87();
        break;
      case 88: // _dim_arange
        implementation_88();
        break;
      case 89: // argmax
        implementation_89();
        break;
      case 90: // argmin
        implementation_90();
        break;
      case 91: // acosh
        implementation_91();
        break;
      case 92: // arccosh
        implementation_92();
        break;
      case 93: // asinh
        implementation_93();
        break;
      case 94: // arcsinh
        implementation_94();
        break;
      case 95: // atanh
        implementation_95();
        break;
      case 96: // arctanh
        implementation_96();
        break;
      case 97: // as_strided
        implementation_97();
        break;
      case 98: // asin
        implementation_98();
        break;
      case 99: // arcsin
        implementation_99();
        break;
      case 100: // atan
        implementation_100();
        break;
      case 101: // arctan
        implementation_101();
        break;
      case 102: // atleast_1d
        implementation_102();
        break;
      case 103: // atleast_1d
        implementation_103();
        break;
      case 104: // atleast_2d
        implementation_104();
        break;
      case 105: // atleast_2d
        implementation_105();
        break;
      case 106: // atleast_3d
        implementation_106();
        break;
      case 107: // atleast_3d
        implementation_107();
        break;
      case 108: // baddbmm
        implementation_108();
        break;
      case 109: // baddbmm
        implementation_109();
        break;
      case 110: // baddbmm
        implementation_110();
        break;
      case 111: // batch_norm
        implementation_111();
        break;
      case 112: // quantized_batch_norm
        implementation_112();
        break;
      case 113: // _batch_norm_impl_index
        implementation_113();
        break;
      case 114: // _batch_norm_impl_index_backward
        implementation_114();
        break;
      case 115: // bernoulli
        implementation_115();
        break;
      case 116: // bernoulli
        implementation_116();
        break;
      case 117: // bilinear
        implementation_117();
        break;
      case 118: // binary_cross_entropy
        implementation_118();
        break;
      case 119: // binary_cross_entropy
        implementation_119();
        break;
      case 120: // binary_cross_entropy
        implementation_120();
        break;
      case 121: // binary_cross_entropy_backward
        implementation_121();
        break;
      case 122: // binary_cross_entropy_backward
        implementation_122();
        break;
      case 123: // binary_cross_entropy_backward
        implementation_123();
        break;
      case 124: // binary_cross_entropy_with_logits
        implementation_124();
        break;
      case 125: // binary_cross_entropy_with_logits
        implementation_125();
        break;
      case 126: // binary_cross_entropy_with_logits
        implementation_126();
        break;
      case 127: // binary_cross_entropy_with_logits
        implementation_127();
        break;
      case 128: // binary_cross_entropy_with_logits_backward
        implementation_128();
        break;
      case 129: // binary_cross_entropy_with_logits_backward
        implementation_129();
        break;
      case 130: // binary_cross_entropy_with_logits_backward
        implementation_130();
        break;
      case 131: // binary_cross_entropy_with_logits_backward
        implementation_131();
        break;
      case 132: // bincount
        implementation_132();
        break;
      case 133: // bincount
        implementation_133();
        break;
      case 134: // bincount
        implementation_134();
        break;
      case 135: // bitwise_not
        implementation_135();
        break;
      case 136: // copysign
        implementation_136();
        break;
      case 137: // copysign
        implementation_137();
        break;
      case 138: // logical_not
        implementation_138();
        break;
      case 139: // logical_xor
        implementation_139();
        break;
      case 140: // logical_and
        implementation_140();
        break;
      case 141: // logical_or
        implementation_141();
        break;
      case 142: // bmm
        implementation_142();
        break;
      case 143: // broadcast_tensors
        implementation_143();
        break;
      case 144: // broadcast_to
        implementation_144();
        break;
      case 145: // cat
        implementation_145();
        break;
      case 146: // cat
        implementation_146();
        break;
      case 147: // concat
        implementation_147();
        break;
      case 148: // concat
        implementation_148();
        break;
      case 149: // block_diag
        implementation_149();
        break;
      case 150: // ceil
        implementation_150();
        break;
      case 151: // chain_matmul
        implementation_151();
        break;
      case 152: // unsafe_chunk
        implementation_152();
        break;
      case 153: // unsafe_chunk
        implementation_153();
        break;
      case 154: // chunk
        implementation_154();
        break;
      case 155: // chunk
        implementation_155();
        break;
      case 156: // tensor_split
        implementation_156();
        break;
      case 157: // tensor_split
        implementation_157();
        break;
      case 158: // tensor_split
        implementation_158();
        break;
      case 159: // tensor_split
        implementation_159();
        break;
      case 160: // tensor_split
        implementation_160();
        break;
      case 161: // tensor_split
        implementation_161();
        break;
      case 162: // clamp
        implementation_162();
        break;
      case 163: // clamp
        implementation_163();
        break;
      case 164: // clamp
        implementation_164();
        break;
      case 165: // clamp_max
        implementation_165();
        break;
      case 166: // clamp_max
        implementation_166();
        break;
      case 167: // clamp_min
        implementation_167();
        break;
      case 168: // clamp_min
        implementation_168();
        break;
      case 169: // clip
        implementation_169();
        break;
      case 170: // clip
        implementation_170();
        break;
      case 171: // clip
        implementation_171();
        break;
      case 172: // cudnn_is_acceptable
        implementation_172();
        break;
      case 173: // complex
        implementation_173();
        break;
      case 174: // polar
        implementation_174();
        break;
      case 175: // constant_pad_nd
        implementation_175();
        break;
      case 176: // constant_pad_nd
        implementation_176();
        break;
      case 177: // contiguous
        implementation_177();
        break;
      case 178: // convolution
        implementation_178();
        break;
      case 179: // convolution_overrideable
        implementation_179();
        break;
      case 180: // convolution_backward_overrideable
        implementation_180();
        break;
      case 181: // _convolution
        implementation_181();
        break;
      case 182: // _convolution
        implementation_182();
        break;
      case 183: // _convolution_nogroup
        implementation_183();
        break;
      case 184: // _convolution_double_backward
        implementation_184();
        break;
      case 185: // conv1d
        implementation_185();
        break;
      case 186: // conv1d
        implementation_186();
        break;
      case 187: // conv1d
        implementation_187();
        break;
      case 188: // conv1d
        implementation_188();
        break;
      case 189: // conv1d
        implementation_189();
        break;
      case 190: // conv1d
        implementation_190();
        break;
      case 191: // conv2d
        implementation_191();
        break;
      case 192: // conv2d
        implementation_192();
        break;
      case 193: // conv2d
        implementation_193();
        break;
      case 194: // conv2d
        implementation_194();
        break;
      case 195: // conv2d
        implementation_195();
        break;
      case 196: // conv2d
        implementation_196();
        break;
      case 197: // conv3d
        implementation_197();
        break;
      case 198: // conv3d
        implementation_198();
        break;
      case 199: // conv3d
        implementation_199();
        break;
      case 200: // conv3d
        implementation_200();
        break;
      case 201: // conv3d
        implementation_201();
        break;
      case 202: // conv3d
        implementation_202();
        break;
      case 203: // conv_tbc
        implementation_203();
        break;
      case 204: // conv_tbc
        implementation_204();
        break;
      case 205: // conv_tbc_backward
        implementation_205();
        break;
      case 206: // conv_transpose1d
        implementation_206();
        break;
      case 207: // conv_transpose1d
        implementation_207();
        break;
      case 208: // conv_transpose1d
        implementation_208();
        break;
      case 209: // conv_transpose1d
        implementation_209();
        break;
      case 210: // conv_transpose1d
        implementation_210();
        break;
      case 211: // conv_transpose1d
        implementation_211();
        break;
      case 212: // conv_transpose1d
        implementation_212();
        break;
      case 213: // conv_transpose2d
        implementation_213();
        break;
      case 214: // conv_transpose2d
        implementation_214();
        break;
      case 215: // conv_transpose2d
        implementation_215();
        break;
      case 216: // conv_transpose2d
        implementation_216();
        break;
      case 217: // conv_transpose2d
        implementation_217();
        break;
      case 218: // conv_transpose2d
        implementation_218();
        break;
      case 219: // conv_transpose2d
        implementation_219();
        break;
      case 220: // conv_transpose3d
        implementation_220();
        break;
      case 221: // conv_transpose3d
        implementation_221();
        break;
      case 222: // conv_transpose3d
        implementation_222();
        break;
      case 223: // conv_transpose3d
        implementation_223();
        break;
      case 224: // conv_transpose3d
        implementation_224();
        break;
      case 225: // conv_transpose3d
        implementation_225();
        break;
      case 226: // conv_transpose3d
        implementation_226();
        break;
      case 227: // _copy_from
        implementation_227();
        break;
      case 228: // _copy_from
        implementation_228();
        break;
      case 229: // _copy_from_and_resize
        implementation_229();
        break;
      case 230: // cos
        implementation_230();
        break;
      case 231: // cosh
        implementation_231();
        break;
      case 232: // cosine_embedding_loss
        implementation_232();
        break;
      case 233: // cosine_embedding_loss
        implementation_233();
        break;
      case 234: // cosine_embedding_loss
        implementation_234();
        break;
      case 235: // count_nonzero
        implementation_235();
        break;
      case 236: // count_nonzero
        implementation_236();
        break;
      case 237: // cov
        implementation_237();
        break;
      case 238: // cov
        implementation_238();
        break;
      case 239: // cov
        implementation_239();
        break;
      case 240: // cov
        implementation_240();
        break;
      case 241: // corrcoef
        implementation_241();
        break;
      case 242: // cudnn_affine_grid_generator
        implementation_242();
        break;
      case 243: // cudnn_affine_grid_generator_backward
        implementation_243();
        break;
      case 244: // cudnn_batch_norm
        implementation_244();
        break;
      case 245: // cudnn_batch_norm_backward
        implementation_245();
        break;
      case 246: // cudnn_convolution
        implementation_246();
        break;
      case 247: // cudnn_convolution
        implementation_247();
        break;
      case 248: // cudnn_convolution
        implementation_248();
        break;
      case 249: // cudnn_convolution_backward_input
        implementation_249();
        break;
      case 250: // cudnn_convolution_backward
        implementation_250();
        break;
      case 251: // cudnn_convolution_backward_weight
        implementation_251();
        break;
      case 252: // cudnn_convolution_transpose
        implementation_252();
        break;
      case 253: // cudnn_convolution_transpose
        implementation_253();
        break;
      case 254: // cudnn_convolution_transpose
        implementation_254();
        break;
      case 255: // cudnn_convolution_transpose_backward
        implementation_255();
        break;
      case 256: // cudnn_convolution_transpose_backward_input
        implementation_256();
        break;
      case 257: // cudnn_convolution_transpose_backward_weight
        implementation_257();
        break;
      case 258: // cudnn_convolution_relu
        implementation_258();
        break;
      case 259: // cudnn_grid_sampler
        implementation_259();
        break;
      case 260: // cudnn_grid_sampler_backward
        implementation_260();
        break;
      case 261: // cummax
        implementation_261();
        break;
      case 262: // cummin
        implementation_262();
        break;
      case 263: // cummaxmin_backward
        implementation_263();
        break;
      case 264: // cumprod
        implementation_264();
        break;
      case 265: // cumprod_backward
        implementation_265();
        break;
      case 266: // cumsum
        implementation_266();
        break;
      case 267: // cumulative_trapezoid
        implementation_267();
        break;
      case 268: // cumulative_trapezoid
        implementation_268();
        break;
      case 269: // cumulative_trapezoid
        implementation_269();
        break;
      case 270: // cumulative_trapezoid
        implementation_270();
        break;
      case 271: // cumulative_trapezoid
        implementation_271();
        break;
      case 272: // ctc_loss
        implementation_272();
        break;
      case 273: // ctc_loss
        implementation_273();
        break;
      case 274: // ctc_loss
        implementation_274();
        break;
      case 275: // ctc_loss
        implementation_275();
        break;
      case 276: // ctc_loss
        implementation_276();
        break;
      case 277: // ctc_loss
        implementation_277();
        break;
      case 278: // ctc_loss
        implementation_278();
        break;
      case 279: // ctc_loss
        implementation_279();
        break;
      case 280: // _ctc_loss
        implementation_280();
        break;
      case 281: // _ctc_loss
        implementation_281();
        break;
      case 282: // _ctc_loss
        implementation_282();
        break;
      case 283: // _ctc_loss_backward
        implementation_283();
        break;
      case 284: // _ctc_loss_backward
        implementation_284();
        break;
      case 285: // diag_embed
        implementation_285();
        break;
      case 286: // diag_embed
        implementation_286();
        break;
      case 287: // diag_embed
        implementation_287();
        break;
      case 288: // diag_embed
        implementation_288();
        break;
      case 289: // diagflat
        implementation_289();
        break;
      case 290: // diagflat
        implementation_290();
        break;
      case 291: // diagonal
        implementation_291();
        break;
      case 292: // diagonal
        implementation_292();
        break;
      case 293: // diagonal
        implementation_293();
        break;
      case 294: // diagonal
        implementation_294();
        break;
      case 295: // diagonal_backward
        implementation_295();
        break;
      case 296: // diff
        implementation_296();
        break;
      case 297: // diff
        implementation_297();
        break;
      case 298: // diff
        implementation_298();
        break;
      case 299: // diff
        implementation_299();
        break;
      case 300: // diff
        implementation_300();
        break;
      case 301: // gradient
        implementation_301();
        break;
      case 302: // gradient
        implementation_302();
        break;
      case 303: // gradient
        implementation_303();
        break;
      case 304: // gradient
        implementation_304();
        break;
      case 305: // gradient
        implementation_305();
        break;
      case 306: // gradient
        implementation_306();
        break;
      case 307: // gradient
        implementation_307();
        break;
      case 308: // gradient
        implementation_308();
        break;
      case 309: // div
        implementation_309();
        break;
      case 310: // div
        implementation_310();
        break;
      case 311: // divide
        implementation_311();
        break;
      case 312: // divide
        implementation_312();
        break;
      case 313: // true_divide
        implementation_313();
        break;
      case 314: // true_divide
        implementation_314();
        break;
      case 315: // dot
        implementation_315();
        break;
      case 316: // vdot
        implementation_316();
        break;
      case 317: // embedding
        implementation_317();
        break;
      case 318: // embedding
        implementation_318();
        break;
      case 319: // embedding
        implementation_319();
        break;
      case 320: // embedding
        implementation_320();
        break;
      case 321: // embedding_backward
        implementation_321();
        break;
      case 322: // embedding_dense_backward
        implementation_322();
        break;
      case 323: // embedding_sparse_backward
        implementation_323();
        break;
      case 324: // _embedding_bag_forward_only
        implementation_324();
        break;
      case 325: // _embedding_bag_forward_only
        implementation_325();
        break;
      case 326: // _embedding_bag_forward_only
        implementation_326();
        break;
      case 327: // _embedding_bag_forward_only
        implementation_327();
        break;
      case 328: // _embedding_bag_forward_only
        implementation_328();
        break;
      case 329: // _embedding_bag_forward_only
        implementation_329();
        break;
      case 330: // _embedding_bag_forward_only
        implementation_330();
        break;
      case 331: // row_stack
        implementation_331();
        break;
      case 332: // embedding_bag
        implementation_332();
        break;
      case 333: // embedding_bag
        implementation_333();
        break;
      case 334: // embedding_bag
        implementation_334();
        break;
      case 335: // embedding_bag
        implementation_335();
        break;
      case 336: // embedding_bag
        implementation_336();
        break;
      case 337: // embedding_bag
        implementation_337();
        break;
      case 338: // _embedding_bag
        implementation_338();
        break;
      case 339: // _embedding_bag
        implementation_339();
        break;
      case 340: // _embedding_bag
        implementation_340();
        break;
      case 341: // _embedding_bag
        implementation_341();
        break;
      case 342: // _embedding_bag
        implementation_342();
        break;
      case 343: // _embedding_bag
        implementation_343();
        break;
      case 344: // _embedding_bag
        implementation_344();
        break;
      case 345: // _embedding_bag_backward
        implementation_345();
        break;
      case 346: // _embedding_bag_backward
        implementation_346();
        break;
      case 347: // _embedding_bag_sparse_backward
        implementation_347();
        break;
      case 348: // _embedding_bag_sparse_backward
        implementation_348();
        break;
      case 349: // _embedding_bag_dense_backward
        implementation_349();
        break;
      case 350: // _embedding_bag_dense_backward
        implementation_350();
        break;
      case 351: // _embedding_bag_per_sample_weights_backward
        implementation_351();
        break;
      case 352: // _embedding_bag_per_sample_weights_backward
        implementation_352();
        break;
      case 353: // erf
        implementation_353();
        break;
      case 354: // erfc
        implementation_354();
        break;
      case 355: // exp
        implementation_355();
        break;
      case 356: // exp2
        implementation_356();
        break;
      case 357: // expm1
        implementation_357();
        break;
      case 358: // expand
        implementation_358();
        break;
      case 359: // expand
        implementation_359();
        break;
      case 360: // expand_as
        implementation_360();
        break;
      case 361: // flatten
        implementation_361();
        break;
      case 362: // flatten
        implementation_362();
        break;
      case 363: // flatten
        implementation_363();
        break;
      case 364: // unflatten
        implementation_364();
        break;
      case 365: // floor
        implementation_365();
        break;
      case 366: // floor_divide
        implementation_366();
        break;
      case 367: // floor_divide
        implementation_367();
        break;
      case 368: // frac
        implementation_368();
        break;
      case 369: // gcd
        implementation_369();
        break;
      case 370: // lcm
        implementation_370();
        break;
      case 371: // grid_sampler
        implementation_371();
        break;
      case 372: // grid_sampler_2d
        implementation_372();
        break;
      case 373: // grid_sampler_2d_backward
        implementation_373();
        break;
      case 374: // _grid_sampler_2d_cpu_fallback
        implementation_374();
        break;
      case 375: // _grid_sampler_2d_cpu_fallback_backward
        implementation_375();
        break;
      case 376: // grid_sampler_3d
        implementation_376();
        break;
      case 377: // grid_sampler_3d_backward
        implementation_377();
        break;
      case 378: // hinge_embedding_loss
        implementation_378();
        break;
      case 379: // hinge_embedding_loss
        implementation_379();
        break;
      case 380: // hinge_embedding_loss
        implementation_380();
        break;
      case 381: // group_norm
        implementation_381();
        break;
      case 382: // group_norm
        implementation_382();
        break;
      case 383: // group_norm
        implementation_383();
        break;
      case 384: // group_norm
        implementation_384();
        break;
      case 385: // group_norm
        implementation_385();
        break;
      case 386: // native_group_norm
        implementation_386();
        break;
      case 387: // native_group_norm_backward
        implementation_387();
        break;
      case 388: // _fft_r2c
        implementation_388();
        break;
      case 389: // _fft_c2r
        implementation_389();
        break;
      case 390: // _fft_c2c
        implementation_390();
        break;
      case 391: // _cufft_get_plan_cache_size
        implementation_391();
        break;
      case 392: // _cufft_get_plan_cache_max_size
        implementation_392();
        break;
      case 393: // index
        implementation_393();
        break;
      case 394: // index_copy
        implementation_394();
        break;
      case 395: // index_put
        implementation_395();
        break;
      case 396: // index_put
        implementation_396();
        break;
      case 397: // instance_norm
        implementation_397();
        break;
      case 398: // inverse
        implementation_398();
        break;
      case 399: // _inverse_helper
        implementation_399();
        break;
      case 400: // isclose
        implementation_400();
        break;
      case 401: // isclose
        implementation_401();
        break;
      case 402: // isclose
        implementation_402();
        break;
      case 403: // isclose
        implementation_403();
        break;
      case 404: // isin
        implementation_404();
        break;
      case 405: // isin
        implementation_405();
        break;
      case 406: // isin
        implementation_406();
        break;
      case 407: // isin
        implementation_407();
        break;
      case 408: // isin
        implementation_408();
        break;
      case 409: // isin
        implementation_409();
        break;
      case 410: // isin
        implementation_410();
        break;
      case 411: // isin
        implementation_411();
        break;
      case 412: // isin
        implementation_412();
        break;
      case 413: // isnan
        implementation_413();
        break;
      case 414: // is_distributed
        implementation_414();
        break;
      case 415: // is_floating_point
        implementation_415();
        break;
      case 416: // is_complex
        implementation_416();
        break;
      case 417: // is_conj
        implementation_417();
        break;
      case 418: // is_neg
        implementation_418();
        break;
      case 419: // isreal
        implementation_419();
        break;
      case 420: // is_nonzero
        implementation_420();
        break;
      case 421: // is_same_size
        implementation_421();
        break;
      case 422: // is_signed
        implementation_422();
        break;
      case 423: // is_inference
        implementation_423();
        break;
      case 424: // kl_div
        implementation_424();
        break;
      case 425: // kl_div
        implementation_425();
        break;
      case 426: // kl_div
        implementation_426();
        break;
      case 427: // kl_div_backward
        implementation_427();
        break;
      case 428: // kl_div_backward
        implementation_428();
        break;
      case 429: // kl_div_backward
        implementation_429();
        break;
      case 430: // kron
        implementation_430();
        break;
      case 431: // kthvalue
        implementation_431();
        break;
      case 432: // kthvalue
        implementation_432();
        break;
      case 433: // kthvalue
        implementation_433();
        break;
      case 434: // layer_norm
        implementation_434();
        break;
      case 435: // layer_norm
        implementation_435();
        break;
      case 436: // layer_norm
        implementation_436();
        break;
      case 437: // layer_norm
        implementation_437();
        break;
      case 438: // layer_norm
        implementation_438();
        break;
      case 439: // native_layer_norm
        implementation_439();
        break;
      case 440: // native_layer_norm_backward
        implementation_440();
        break;
      case 441: // nan_to_num
        implementation_441();
        break;
      case 442: // linear
        implementation_442();
        break;
      case 443: // linear
        implementation_443();
        break;
      case 444: // mkldnn_linear
        implementation_444();
        break;
      case 445: // mkldnn_linear
        implementation_445();
        break;
      case 446: // mkldnn_linear_backward_input
        implementation_446();
        break;
      case 447: // mkldnn_linear_backward_weights
        implementation_447();
        break;
      case 448: // mkldnn_linear_backward
        implementation_448();
        break;
      case 449: // fbgemm_linear_int8_weight_fp32_activation
        implementation_449();
        break;
      case 450: // fbgemm_linear_int8_weight
        implementation_450();
        break;
      case 451: // fbgemm_pack_gemm_matrix_fp16
        implementation_451();
        break;
      case 452: // fbgemm_linear_fp16_weight_fp32_activation
        implementation_452();
        break;
      case 453: // fbgemm_linear_fp16_weight
        implementation_453();
        break;
      case 454: // fbgemm_pack_quantized_matrix
        implementation_454();
        break;
      case 455: // fbgemm_pack_quantized_matrix
        implementation_455();
        break;
      case 456: // ldexp
        implementation_456();
        break;
      case 457: // log
        implementation_457();
        break;
      case 458: // log10
        implementation_458();
        break;
      case 459: // log1p
        implementation_459();
        break;
      case 460: // log2
        implementation_460();
        break;
      case 461: // logaddexp
        implementation_461();
        break;
      case 462: // logaddexp2
        implementation_462();
        break;
      case 463: // xlogy
        implementation_463();
        break;
      case 464: // xlogy
        implementation_464();
        break;
      case 465: // xlogy
        implementation_465();
        break;
      case 466: // logdet
        implementation_466();
        break;
      case 467: // log_softmax
        implementation_467();
        break;
      case 468: // _log_softmax
        implementation_468();
        break;
      case 469: // _log_softmax_backward_data
        implementation_469();
        break;
      case 470: // _logcumsumexp
        implementation_470();
        break;
      case 471: // logcumsumexp
        implementation_471();
        break;
      case 472: // logsumexp
        implementation_472();
        break;
      case 473: // logsumexp
        implementation_473();
        break;
      case 474: // margin_ranking_loss
        implementation_474();
        break;
      case 475: // margin_ranking_loss
        implementation_475();
        break;
      case 476: // margin_ranking_loss
        implementation_476();
        break;
      case 477: // matmul
        implementation_477();
        break;
      case 478: // matrix_rank
        implementation_478();
        break;
      case 479: // matrix_rank
        implementation_479();
        break;
      case 480: // matrix_rank
        implementation_480();
        break;
      case 481: // matrix_rank
        implementation_481();
        break;
      case 482: // matrix_power
        implementation_482();
        break;
      case 483: // matrix_exp
        implementation_483();
        break;
      case 484: // matrix_exp_backward
        implementation_484();
        break;
      case 485: // _aminmax
        implementation_485();
        break;
      case 486: // _aminmax
        implementation_486();
        break;
      case 487: // _aminmax
        implementation_487();
        break;
      case 488: // aminmax
        implementation_488();
        break;
      case 489: // _compute_linear_combination
        implementation_489();
        break;
      case 490: // max
        implementation_490();
        break;
      case 491: // max
        implementation_491();
        break;
      case 492: // value_selecting_reduction_backward
        implementation_492();
        break;
      case 493: // amax
        implementation_493();
        break;
      case 494: // amax
        implementation_494();
        break;
      case 495: // amax
        implementation_495();
        break;
      case 496: // max_pool1d_with_indices
        implementation_496();
        break;
      case 497: // max_pool1d_with_indices
        implementation_497();
        break;
      case 498: // max_pool1d_with_indices
        implementation_498();
        break;
      case 499: // max_pool1d_with_indices
        implementation_499();
        break;
      case 500: // max_pool1d_with_indices
        implementation_500();
        break;
      case 501: // max_pool1d
        implementation_501();
        break;
      case 502: // max_pool1d
        implementation_502();
        break;
      case 503: // max_pool1d
        implementation_503();
        break;
      case 504: // max_pool1d
        implementation_504();
        break;
      case 505: // max_pool1d
        implementation_505();
        break;
      case 506: // max_pool2d
        implementation_506();
        break;
      case 507: // max_pool2d
        implementation_507();
        break;
      case 508: // max_pool2d
        implementation_508();
        break;
      case 509: // max_pool2d
        implementation_509();
        break;
      case 510: // max_pool2d
        implementation_510();
        break;
      case 511: // mkldnn_max_pool2d
        implementation_511();
        break;
      case 512: // mkldnn_max_pool2d
        implementation_512();
        break;
      case 513: // mkldnn_max_pool2d
        implementation_513();
        break;
      case 514: // mkldnn_max_pool2d
        implementation_514();
        break;
      case 515: // mkldnn_max_pool2d
        implementation_515();
        break;
      case 516: // mkldnn_max_pool2d_backward
        implementation_516();
        break;
      case 517: // mkldnn_max_pool2d_backward
        implementation_517();
        break;
      case 518: // mkldnn_max_pool2d_backward
        implementation_518();
        break;
      case 519: // mkldnn_max_pool2d_backward
        implementation_519();
        break;
      case 520: // mkldnn_max_pool2d_backward
        implementation_520();
        break;
      case 521: // mkldnn_max_pool3d
        implementation_521();
        break;
      case 522: // mkldnn_max_pool3d
        implementation_522();
        break;
      case 523: // mkldnn_max_pool3d
        implementation_523();
        break;
      case 524: // mkldnn_max_pool3d
        implementation_524();
        break;
      case 525: // mkldnn_max_pool3d
        implementation_525();
        break;
      case 526: // mkldnn_max_pool3d_backward
        implementation_526();
        break;
      case 527: // mkldnn_max_pool3d_backward
        implementation_527();
        break;
      case 528: // mkldnn_max_pool3d_backward
        implementation_528();
        break;
      case 529: // mkldnn_max_pool3d_backward
        implementation_529();
        break;
      case 530: // mkldnn_max_pool3d_backward
        implementation_530();
        break;
      case 531: // quantized_max_pool1d
        implementation_531();
        break;
      case 532: // quantized_max_pool1d
        implementation_532();
        break;
      case 533: // quantized_max_pool1d
        implementation_533();
        break;
      case 534: // quantized_max_pool1d
        implementation_534();
        break;
      case 535: // quantized_max_pool1d
        implementation_535();
        break;
      case 536: // quantized_max_pool2d
        implementation_536();
        break;
      case 537: // quantized_max_pool2d
        implementation_537();
        break;
      case 538: // quantized_max_pool2d
        implementation_538();
        break;
      case 539: // quantized_max_pool2d
        implementation_539();
        break;
      case 540: // quantized_max_pool2d
        implementation_540();
        break;
      case 541: // max_pool3d
        implementation_541();
        break;
      case 542: // max_pool3d
        implementation_542();
        break;
      case 543: // max_pool3d
        implementation_543();
        break;
      case 544: // max_pool3d
        implementation_544();
        break;
      case 545: // max_pool3d
        implementation_545();
        break;
      case 546: // mean
        implementation_546();
        break;
      case 547: // mean
        implementation_547();
        break;
      case 548: // mean
        implementation_548();
        break;
      case 549: // nanmean
        implementation_549();
        break;
      case 550: // nanmean
        implementation_550();
        break;
      case 551: // nanmean
        implementation_551();
        break;
      case 552: // median
        implementation_552();
        break;
      case 553: // median
        implementation_553();
        break;
      case 554: // median
        implementation_554();
        break;
      case 555: // nanmedian
        implementation_555();
        break;
      case 556: // nanmedian
        implementation_556();
        break;
      case 557: // nanmedian
        implementation_557();
        break;
      case 558: // min
        implementation_558();
        break;
      case 559: // min
        implementation_559();
        break;
      case 560: // amin
        implementation_560();
        break;
      case 561: // amin
        implementation_561();
        break;
      case 562: // amin
        implementation_562();
        break;
      case 563: // mkldnn_convolution
        implementation_563();
        break;
      case 564: // mkldnn_convolution_backward_input
        implementation_564();
        break;
      case 565: // mkldnn_convolution_backward_weights
        implementation_565();
        break;
      case 566: // mkldnn_convolution_backward
        implementation_566();
        break;
      case 567: // miopen_batch_norm
        implementation_567();
        break;
      case 568: // miopen_batch_norm_backward
        implementation_568();
        break;
      case 569: // miopen_convolution
        implementation_569();
        break;
      case 570: // miopen_convolution_backward_input
        implementation_570();
        break;
      case 571: // miopen_convolution_backward
        implementation_571();
        break;
      case 572: // miopen_convolution_backward_bias
        implementation_572();
        break;
      case 573: // miopen_convolution_backward_weight
        implementation_573();
        break;
      case 574: // miopen_convolution_transpose
        implementation_574();
        break;
      case 575: // miopen_convolution_transpose_backward
        implementation_575();
        break;
      case 576: // miopen_convolution_transpose_backward_input
        implementation_576();
        break;
      case 577: // miopen_convolution_transpose_backward_weight
        implementation_577();
        break;
      case 578: // miopen_depthwise_convolution
        implementation_578();
        break;
      case 579: // miopen_depthwise_convolution_backward_input
        implementation_579();
        break;
      case 580: // miopen_depthwise_convolution_backward
        implementation_580();
        break;
      case 581: // miopen_depthwise_convolution_backward_weight
        implementation_581();
        break;
      case 582: // miopen_rnn
        implementation_582();
        break;
      case 583: // mm
        implementation_583();
        break;
      case 584: // _sparse_mm
        implementation_584();
        break;
      case 585: // _sparse_sparse_matmul
        implementation_585();
        break;
      case 586: // _sparse_mask_helper
        implementation_586();
        break;
      case 587: // mode
        implementation_587();
        break;
      case 588: // mode
        implementation_588();
        break;
      case 589: // mode
        implementation_589();
        break;
      case 590: // mul
        implementation_590();
        break;
      case 591: // mul
        implementation_591();
        break;
      case 592: // multiply
        implementation_592();
        break;
      case 593: // multiply
        implementation_593();
        break;
      case 594: // mv
        implementation_594();
        break;
      case 595: // mvlgamma
        implementation_595();
        break;
      case 596: // narrow_copy
        implementation_596();
        break;
      case 597: // narrow
        implementation_597();
        break;
      case 598: // narrow
        implementation_598();
        break;
      case 599: // native_batch_norm
        implementation_599();
        break;
      case 600: // batch_norm_stats
        implementation_600();
        break;
      case 601: // batch_norm_elemt
        implementation_601();
        break;
      case 602: // batch_norm_gather_stats
        implementation_602();
        break;
      case 603: // batch_norm_gather_stats_with_counts
        implementation_603();
        break;
      case 604: // native_batch_norm_backward
        implementation_604();
        break;
      case 605: // batch_norm_backward_reduce
        implementation_605();
        break;
      case 606: // batch_norm_backward_elemt
        implementation_606();
        break;
      case 607: // batch_norm_update_stats
        implementation_607();
        break;
      case 608: // is_vulkan_available
        implementation_608();
        break;
      case 609: // _nnpack_available
        implementation_609();
        break;
      case 610: // _nnpack_spatial_convolution
        implementation_610();
        break;
      case 611: // _nnpack_spatial_convolution
        implementation_611();
        break;
      case 612: // _nnpack_spatial_convolution_backward
        implementation_612();
        break;
      case 613: // _nnpack_spatial_convolution_backward_input
        implementation_613();
        break;
      case 614: // _nnpack_spatial_convolution_backward_weight
        implementation_614();
        break;
      case 615: // pairwise_distance
        implementation_615();
        break;
      case 616: // pairwise_distance
        implementation_616();
        break;
      case 617: // pairwise_distance
        implementation_617();
        break;
      case 618: // pairwise_distance
        implementation_618();
        break;
      case 619: // cdist
        implementation_619();
        break;
      case 620: // cdist
        implementation_620();
        break;
      case 621: // _euclidean_dist
        implementation_621();
        break;
      case 622: // _cdist_backward
        implementation_622();
        break;
      case 623: // pdist
        implementation_623();
        break;
      case 624: // pdist
        implementation_624();
        break;
      case 625: // _pdist_forward
        implementation_625();
        break;
      case 626: // _pdist_forward
        implementation_626();
        break;
      case 627: // _pdist_backward
        implementation_627();
        break;
      case 628: // cosine_similarity
        implementation_628();
        break;
      case 629: // cosine_similarity
        implementation_629();
        break;
      case 630: // cosine_similarity
        implementation_630();
        break;
      case 631: // permute
        implementation_631();
        break;
      case 632: // movedim
        implementation_632();
        break;
      case 633: // moveaxis
        implementation_633();
        break;
      case 634: // numpy_T
        implementation_634();
        break;
      case 635: // pixel_shuffle
        implementation_635();
        break;
      case 636: // pixel_unshuffle
        implementation_636();
        break;
      case 637: // channel_shuffle
        implementation_637();
        break;
      case 638: // is_pinned
        implementation_638();
        break;
      case 639: // pin_memory
        implementation_639();
        break;
      case 640: // _pin_memory
        implementation_640();
        break;
      case 641: // pinverse
        implementation_641();
        break;
      case 642: // pinverse
        implementation_642();
        break;
      case 643: // poisson_nll_loss
        implementation_643();
        break;
      case 644: // rad2deg
        implementation_644();
        break;
      case 645: // deg2rad
        implementation_645();
        break;
      case 646: // ravel
        implementation_646();
        break;
      case 647: // reciprocal
        implementation_647();
        break;
      case 648: // neg
        implementation_648();
        break;
      case 649: // negative
        implementation_649();
        break;
      case 650: // repeat
        implementation_650();
        break;
      case 651: // repeat_interleave
        implementation_651();
        break;
      case 652: // repeat_interleave
        implementation_652();
        break;
      case 653: // repeat_interleave
        implementation_653();
        break;
      case 654: // reshape
        implementation_654();
        break;
      case 655: // _reshape_alias
        implementation_655();
        break;
      case 656: // _mkldnn_reshape
        implementation_656();
        break;
      case 657: // reshape_as
        implementation_657();
        break;
      case 658: // round
        implementation_658();
        break;
      case 659: // rrelu
        implementation_659();
        break;
      case 660: // rrelu
        implementation_660();
        break;
      case 661: // rrelu
        implementation_661();
        break;
      case 662: // rrelu
        implementation_662();
        break;
      case 663: // relu
        implementation_663();
        break;
      case 664: // relu6
        implementation_664();
        break;
      case 665: // prelu
        implementation_665();
        break;
      case 666: // prelu_backward
        implementation_666();
        break;
      case 667: // gelu
        implementation_667();
        break;
      case 668: // gelu_backward
        implementation_668();
        break;
      case 669: // infinitely_differentiable_gelu_backward
        implementation_669();
        break;
      case 670: // hardshrink
        implementation_670();
        break;
      case 671: // hardshrink
        implementation_671();
        break;
      case 672: // hardshrink_backward
        implementation_672();
        break;
      case 673: // rsqrt
        implementation_673();
        break;
      case 674: // select
        implementation_674();
        break;
      case 675: // select_backward
        implementation_675();
        break;
      case 676: // selu
        implementation_676();
        break;
      case 677: // celu
        implementation_677();
        break;
      case 678: // celu
        implementation_678();
        break;
      case 679: // silu
        implementation_679();
        break;
      case 680: // silu_backward
        implementation_680();
        break;
      case 681: // mish
        implementation_681();
        break;
      case 682: // mish_backward
        implementation_682();
        break;
      case 683: // sigmoid
        implementation_683();
        break;
      case 684: // logit
        implementation_684();
        break;
      case 685: // sin
        implementation_685();
        break;
      case 686: // sinc
        implementation_686();
        break;
      case 687: // sinh
        implementation_687();
        break;
      case 688: // detach
        implementation_688();
        break;
      case 689: // size
        implementation_689();
        break;
      case 690: // slice
        implementation_690();
        break;
      case 691: // slice
        implementation_691();
        break;
      case 692: // slice_backward
        implementation_692();
        break;
      case 693: // slogdet
        implementation_693();
        break;
      case 694: // smm
        implementation_694();
        break;
      case 695: // softmax
        implementation_695();
        break;
      case 696: // _softmax
        implementation_696();
        break;
      case 697: // _softmax_backward_data
        implementation_697();
        break;
      case 698: // unsafe_split
        implementation_698();
        break;
      case 699: // unsafe_split
        implementation_699();
        break;
      case 700: // split
        implementation_700();
        break;
      case 701: // split
        implementation_701();
        break;
      case 702: // unsafe_split_with_sizes
        implementation_702();
        break;
      case 703: // unsafe_split_with_sizes
        implementation_703();
        break;
      case 704: // split_with_sizes
        implementation_704();
        break;
      case 705: // split_with_sizes
        implementation_705();
        break;
      case 706: // hsplit
        implementation_706();
        break;
      case 707: // hsplit
        implementation_707();
        break;
      case 708: // vsplit
        implementation_708();
        break;
      case 709: // vsplit
        implementation_709();
        break;
      case 710: // dsplit
        implementation_710();
        break;
      case 711: // dsplit
        implementation_711();
        break;
      case 712: // squeeze
        implementation_712();
        break;
      case 713: // squeeze
        implementation_713();
        break;
      case 714: // sspaddmm
        implementation_714();
        break;
      case 715: // sspaddmm
        implementation_715();
        break;
      case 716: // sspaddmm
        implementation_716();
        break;
      case 717: // stack
        implementation_717();
        break;
      case 718: // stack
        implementation_718();
        break;
      case 719: // _stack
        implementation_719();
        break;
      case 720: // _stack
        implementation_720();
        break;
      case 721: // hstack
        implementation_721();
        break;
      case 722: // vstack
        implementation_722();
        break;
      case 723: // dstack
        implementation_723();
        break;
      case 724: // stft
        implementation_724();
        break;
      case 725: // istft
        implementation_725();
        break;
      case 726: // stride
        implementation_726();
        break;
      case 727: // sum
        implementation_727();
        break;
      case 728: // sum
        implementation_728();
        break;
      case 729: // sum
        implementation_729();
        break;
      case 730: // nansum
        implementation_730();
        break;
      case 731: // nansum
        implementation_731();
        break;
      case 732: // nansum
        implementation_732();
        break;
      case 733: // sum_to_size
        implementation_733();
        break;
      case 734: // sqrt
        implementation_734();
        break;
      case 735: // square
        implementation_735();
        break;
      case 736: // std
        implementation_736();
        break;
      case 737: // std
        implementation_737();
        break;
      case 738: // std
        implementation_738();
        break;
      case 739: // std
        implementation_739();
        break;
      case 740: // std
        implementation_740();
        break;
      case 741: // std_mean
        implementation_741();
        break;
      case 742: // std_mean
        implementation_742();
        break;
      case 743: // std_mean
        implementation_743();
        break;
      case 744: // std_mean
        implementation_744();
        break;
      case 745: // std_mean
        implementation_745();
        break;
      case 746: // prod
        implementation_746();
        break;
      case 747: // prod
        implementation_747();
        break;
      case 748: // prod
        implementation_748();
        break;
      case 749: // t
        implementation_749();
        break;
      case 750: // tan
        implementation_750();
        break;
      case 751: // tanh
        implementation_751();
        break;
      case 752: // tensordot
        implementation_752();
        break;
      case 753: // threshold
        implementation_753();
        break;
      case 754: // threshold_backward
        implementation_754();
        break;
      case 755: // tile
        implementation_755();
        break;
      case 756: // transpose
        implementation_756();
        break;
      case 757: // _mkldnn_transpose
        implementation_757();
        break;
      case 758: // one_hot
        implementation_758();
        break;
      case 759: // one_hot
        implementation_759();
        break;
      case 760: // flip
        implementation_760();
        break;
      case 761: // fliplr
        implementation_761();
        break;
      case 762: // flipud
        implementation_762();
        break;
      case 763: // roll
        implementation_763();
        break;
      case 764: // roll
        implementation_764();
        break;
      case 765: // rot90
        implementation_765();
        break;
      case 766: // rot90
        implementation_766();
        break;
      case 767: // rot90
        implementation_767();
        break;
      case 768: // trapezoid
        implementation_768();
        break;
      case 769: // trapezoid
        implementation_769();
        break;
      case 770: // trapezoid
        implementation_770();
        break;
      case 771: // trapezoid
        implementation_771();
        break;
      case 772: // trapezoid
        implementation_772();
        break;
      case 773: // trapz
        implementation_773();
        break;
      case 774: // trapz
        implementation_774();
        break;
      case 775: // trapz
        implementation_775();
        break;
      case 776: // trapz
        implementation_776();
        break;
      case 777: // trapz
        implementation_777();
        break;
      case 778: // _trilinear
        implementation_778();
        break;
      case 779: // _trilinear
        implementation_779();
        break;
      case 780: // triplet_margin_loss
        implementation_780();
        break;
      case 781: // triplet_margin_loss
        implementation_781();
        break;
      case 782: // triplet_margin_loss
        implementation_782();
        break;
      case 783: // triplet_margin_loss
        implementation_783();
        break;
      case 784: // triplet_margin_loss
        implementation_784();
        break;
      case 785: // triplet_margin_loss
        implementation_785();
        break;
      case 786: // trunc
        implementation_786();
        break;
      case 787: // fix
        implementation_787();
        break;
      case 788: // type_as
        implementation_788();
        break;
      case 789: // _has_compatible_shallow_copy_type
        implementation_789();
        break;
      case 790: // _unique
        implementation_790();
        break;
      case 791: // _unique
        implementation_791();
        break;
      case 792: // _unique
        implementation_792();
        break;
      case 793: // unique_dim
        implementation_793();
        break;
      case 794: // unique_dim
        implementation_794();
        break;
      case 795: // unique_dim
        implementation_795();
        break;
      case 796: // unique_dim
        implementation_796();
        break;
      case 797: // unique_consecutive
        implementation_797();
        break;
      case 798: // unique_consecutive
        implementation_798();
        break;
      case 799: // unique_consecutive
        implementation_799();
        break;
      case 800: // unique_dim_consecutive
        implementation_800();
        break;
      case 801: // unique_dim_consecutive
        implementation_801();
        break;
      case 802: // unique_dim_consecutive
        implementation_802();
        break;
      case 803: // _unique2
        implementation_803();
        break;
      case 804: // _unique2
        implementation_804();
        break;
      case 805: // _unique2
        implementation_805();
        break;
      case 806: // _unique2
        implementation_806();
        break;
      case 807: // _unsafe_view
        implementation_807();
        break;
      case 808: // unsqueeze
        implementation_808();
        break;
      case 809: // vander
        implementation_809();
        break;
      case 810: // var
        implementation_810();
        break;
      case 811: // var
        implementation_811();
        break;
      case 812: // var
        implementation_812();
        break;
      case 813: // var
        implementation_813();
        break;
      case 814: // var
        implementation_814();
        break;
      case 815: // var_mean
        implementation_815();
        break;
      case 816: // var_mean
        implementation_816();
        break;
      case 817: // var_mean
        implementation_817();
        break;
      case 818: // var_mean
        implementation_818();
        break;
      case 819: // var_mean
        implementation_819();
        break;
      case 820: // view_as
        implementation_820();
        break;
      case 821: // where
        implementation_821();
        break;
      case 822: // where
        implementation_822();
        break;
      case 823: // where
        implementation_823();
        break;
      case 824: // where
        implementation_824();
        break;
      case 825: // where
        implementation_825();
        break;
      case 826: // _s_where
        implementation_826();
        break;
      case 827: // norm_except_dim
        implementation_827();
        break;
      case 828: // norm_except_dim
        implementation_828();
        break;
      case 829: // norm_except_dim
        implementation_829();
        break;
      case 830: // _weight_norm
        implementation_830();
        break;
      case 831: // _weight_norm
        implementation_831();
        break;
      case 832: // _weight_norm_cuda_interface
        implementation_832();
        break;
      case 833: // _weight_norm_cuda_interface
        implementation_833();
        break;
      case 834: // _weight_norm_cuda_interface_backward
        implementation_834();
        break;
      case 835: // _weight_norm_differentiable_backward
        implementation_835();
        break;
      case 836: // _standard_gamma_grad
        implementation_836();
        break;
      case 837: // _standard_gamma
        implementation_837();
        break;
      case 838: // _dirichlet_grad
        implementation_838();
        break;
      case 839: // _sample_dirichlet
        implementation_839();
        break;
      case 840: // poisson
        implementation_840();
        break;
      case 841: // binomial
        implementation_841();
        break;
      case 842: // native_norm
        implementation_842();
        break;
      case 843: // native_norm
        implementation_843();
        break;
      case 844: // _sparse_sum
        implementation_844();
        break;
      case 845: // _sparse_sum
        implementation_845();
        break;
      case 846: // _sparse_sum_backward
        implementation_846();
        break;
      case 847: // _sparse_softmax
        implementation_847();
        break;
      case 848: // _sparse_softmax
        implementation_848();
        break;
      case 849: // _sparse_softmax_backward_data
        implementation_849();
        break;
      case 850: // _sparse_log_softmax
        implementation_850();
        break;
      case 851: // _sparse_log_softmax
        implementation_851();
        break;
      case 852: // _sparse_log_softmax_backward_data
        implementation_852();
        break;
      case 853: // norm
        implementation_853();
        break;
      case 854: // norm
        implementation_854();
        break;
      case 855: // frexp
        implementation_855();
        break;
      case 856: // frobenius_norm
        implementation_856();
        break;
      case 857: // frobenius_norm
        implementation_857();
        break;
      case 858: // frobenius_norm
        implementation_858();
        break;
      case 859: // nuclear_norm
        implementation_859();
        break;
      case 860: // nuclear_norm
        implementation_860();
        break;
      case 861: // nuclear_norm
        implementation_861();
        break;
      case 862: // nuclear_norm
        implementation_862();
        break;
      case 863: // clone
        implementation_863();
        break;
      case 864: // positive
        implementation_864();
        break;
      case 865: // sub
        implementation_865();
        break;
      case 866: // sub
        implementation_866();
        break;
      case 867: // sub
        implementation_867();
        break;
      case 868: // sub
        implementation_868();
        break;
      case 869: // subtract
        implementation_869();
        break;
      case 870: // subtract
        implementation_870();
        break;
      case 871: // subtract
        implementation_871();
        break;
      case 872: // subtract
        implementation_872();
        break;
      case 873: // rsub
        implementation_873();
        break;
      case 874: // rsub
        implementation_874();
        break;
      case 875: // heaviside
        implementation_875();
        break;
      case 876: // rsub
        implementation_876();
        break;
      case 877: // rsub
        implementation_877();
        break;
      case 878: // _sparse_addmm
        implementation_878();
        break;
      case 879: // _sparse_addmm
        implementation_879();
        break;
      case 880: // _sparse_addmm
        implementation_880();
        break;
      case 881: // addmm
        implementation_881();
        break;
      case 882: // addmm
        implementation_882();
        break;
      case 883: // addmm
        implementation_883();
        break;
      case 884: // sparse_mask
        implementation_884();
        break;
      case 885: // _to_cpu
        implementation_885();
        break;
      case 886: // to_dense
        implementation_886();
        break;
      case 887: // to_dense_backward
        implementation_887();
        break;
      case 888: // sparse_dim
        implementation_888();
        break;
      case 889: // _dimI
        implementation_889();
        break;
      case 890: // dense_dim
        implementation_890();
        break;
      case 891: // _dimV
        implementation_891();
        break;
      case 892: // _nnz
        implementation_892();
        break;
      case 893: // coalesce
        implementation_893();
        break;
      case 894: // _coalesce
        implementation_894();
        break;
      case 895: // is_coalesced
        implementation_895();
        break;
      case 896: // _indices
        implementation_896();
        break;
      case 897: // _values
        implementation_897();
        break;
      case 898: // indices
        implementation_898();
        break;
      case 899: // values
        implementation_899();
        break;
      case 900: // crow_indices
        implementation_900();
        break;
      case 901: // col_indices
        implementation_901();
        break;
      case 902: // hspmm
        implementation_902();
        break;
      case 903: // unbind
        implementation_903();
        break;
      case 904: // unbind
        implementation_904();
        break;
      case 905: // to_sparse
        implementation_905();
        break;
      case 906: // to_sparse
        implementation_906();
        break;
      case 907: // to_mkldnn
        implementation_907();
        break;
      case 908: // mkldnn_reorder_conv2d_weight
        implementation_908();
        break;
      case 909: // mkldnn_reorder_conv2d_weight
        implementation_909();
        break;
      case 910: // mkldnn_reorder_conv2d_weight
        implementation_910();
        break;
      case 911: // mkldnn_reorder_conv2d_weight
        implementation_911();
        break;
      case 912: // mkldnn_reorder_conv2d_weight
        implementation_912();
        break;
      case 913: // mkldnn_reorder_conv3d_weight
        implementation_913();
        break;
      case 914: // mkldnn_reorder_conv3d_weight
        implementation_914();
        break;
      case 915: // mkldnn_reorder_conv3d_weight
        implementation_915();
        break;
      case 916: // mkldnn_reorder_conv3d_weight
        implementation_916();
        break;
      case 917: // mkldnn_reorder_conv3d_weight
        implementation_917();
        break;
      case 918: // to_mkldnn_backward
        implementation_918();
        break;
      case 919: // dequantize
        implementation_919();
        break;
      case 920: // dequantize
        implementation_920();
        break;
      case 921: // q_zero_point
        implementation_921();
        break;
      case 922: // q_per_channel_scales
        implementation_922();
        break;
      case 923: // q_per_channel_zero_points
        implementation_923();
        break;
      case 924: // q_per_channel_axis
        implementation_924();
        break;
      case 925: // int_repr
        implementation_925();
        break;
      case 926: // _make_per_tensor_quantized_tensor
        implementation_926();
        break;
      case 927: // _make_per_channel_quantized_tensor
        implementation_927();
        break;
      case 928: // fake_quantize_per_tensor_affine
        implementation_928();
        break;
      case 929: // fake_quantize_per_tensor_affine
        implementation_929();
        break;
      case 930: // fake_quantize_per_tensor_affine_cachemask
        implementation_930();
        break;
      case 931: // _fake_quantize_per_tensor_affine_cachemask_tensor_qparams
        implementation_931();
        break;
      case 932: // fake_quantize_per_tensor_affine_cachemask_backward
        implementation_932();
        break;
      case 933: // _fake_quantize_learnable_per_tensor_affine
        implementation_933();
        break;
      case 934: // _fake_quantize_learnable_per_tensor_affine
        implementation_934();
        break;
      case 935: // _fake_quantize_learnable_per_tensor_affine_backward
        implementation_935();
        break;
      case 936: // _fake_quantize_learnable_per_tensor_affine_backward
        implementation_936();
        break;
      case 937: // fake_quantize_per_channel_affine
        implementation_937();
        break;
      case 938: // fake_quantize_per_channel_affine_cachemask
        implementation_938();
        break;
      case 939: // fake_quantize_per_channel_affine_cachemask_backward
        implementation_939();
        break;
      case 940: // _fake_quantize_learnable_per_channel_affine
        implementation_940();
        break;
      case 941: // _fake_quantize_learnable_per_channel_affine
        implementation_941();
        break;
      case 942: // _fake_quantize_learnable_per_channel_affine_backward
        implementation_942();
        break;
      case 943: // _fake_quantize_learnable_per_channel_affine_backward
        implementation_943();
        break;
      case 944: // fused_moving_avg_obs_fake_quant
        implementation_944();
        break;
      case 945: // fused_moving_avg_obs_fake_quant
        implementation_945();
        break;
      case 946: // fused_moving_avg_obs_fake_quant
        implementation_946();
        break;
      case 947: // _fused_moving_avg_obs_fq_helper
        implementation_947();
        break;
      case 948: // _fused_moving_avg_obs_fq_helper
        implementation_948();
        break;
      case 949: // _fused_moving_avg_obs_fq_helper
        implementation_949();
        break;
      case 950: // _saturate_weight_to_fp16
        implementation_950();
        break;
      case 951: // choose_qparams_optimized
        implementation_951();
        break;
      case 952: // meshgrid
        implementation_952();
        break;
      case 953: // cartesian_prod
        implementation_953();
        break;
      case 954: // combinations
        implementation_954();
        break;
      case 955: // combinations
        implementation_955();
        break;
      case 956: // combinations
        implementation_956();
        break;
      case 957: // item
        implementation_957();
        break;
      case 958: // _local_scalar_dense
        implementation_958();
        break;
      case 959: // _thnn_fused_lstm_cell
        implementation_959();
        break;
      case 960: // _thnn_fused_lstm_cell
        implementation_960();
        break;
      case 961: // _thnn_fused_lstm_cell
        implementation_961();
        break;
      case 962: // _thnn_fused_lstm_cell_backward
        implementation_962();
        break;
      case 963: // _thnn_differentiable_lstm_cell_backward
        implementation_963();
        break;
      case 964: // _thnn_fused_gru_cell
        implementation_964();
        break;
      case 965: // _thnn_fused_gru_cell
        implementation_965();
        break;
      case 966: // _thnn_fused_gru_cell
        implementation_966();
        break;
      case 967: // _thnn_fused_gru_cell_backward
        implementation_967();
        break;
      case 968: // _thnn_differentiable_gru_cell_backward
        implementation_968();
        break;
      case 969: // lstm
        implementation_969();
        break;
      case 970: // lstm
        implementation_970();
        break;
      case 971: // gru
        implementation_971();
        break;
      case 972: // gru
        implementation_972();
        break;
      case 973: // rnn_tanh
        implementation_973();
        break;
      case 974: // rnn_tanh
        implementation_974();
        break;
      case 975: // rnn_relu
        implementation_975();
        break;
      case 976: // rnn_relu
        implementation_976();
        break;
      case 977: // lstm_cell
        implementation_977();
        break;
      case 978: // gru_cell
        implementation_978();
        break;
      case 979: // gru_cell
        implementation_979();
        break;
      case 980: // gru_cell
        implementation_980();
        break;
      case 981: // rnn_tanh_cell
        implementation_981();
        break;
      case 982: // rnn_tanh_cell
        implementation_982();
        break;
      case 983: // rnn_tanh_cell
        implementation_983();
        break;
      case 984: // rnn_relu_cell
        implementation_984();
        break;
      case 985: // rnn_relu_cell
        implementation_985();
        break;
      case 986: // rnn_relu_cell
        implementation_986();
        break;
      case 987: // quantized_lstm_cell
        implementation_987();
        break;
      case 988: // quantized_gru_cell
        implementation_988();
        break;
      case 989: // quantized_rnn_relu_cell
        implementation_989();
        break;
      case 990: // quantized_rnn_tanh_cell
        implementation_990();
        break;
      case 991: // _pack_padded_sequence
        implementation_991();
        break;
      case 992: // _pack_padded_sequence_backward
        implementation_992();
        break;
      case 993: // _pad_packed_sequence
        implementation_993();
        break;
      case 994: // is_set_to
        implementation_994();
        break;
      case 995: // masked_fill
        implementation_995();
        break;
      case 996: // masked_fill
        implementation_996();
        break;
      case 997: // masked_scatter
        implementation_997();
        break;
      case 998: // view
        implementation_998();
        break;
      case 999: // put
        implementation_999();
        break;
      case 1000: // put
        implementation_1000();
        break;
      case 1001: // index_add
        implementation_1001();
        break;
      case 1002: // index_add
        implementation_1002();
        break;
      case 1003: // index_fill
        implementation_1003();
        break;
      case 1004: // index_fill
        implementation_1004();
        break;
      case 1005: // scatter
        implementation_1005();
        break;
      case 1006: // scatter
        implementation_1006();
        break;
      case 1007: // scatter_add
        implementation_1007();
        break;
      case 1008: // bitwise_and
        implementation_1008();
        break;
      case 1009: // bitwise_and
        implementation_1009();
        break;
      case 1010: // __and__
        implementation_1010();
        break;
      case 1011: // __and__
        implementation_1011();
        break;
      case 1012: // bitwise_or
        implementation_1012();
        break;
      case 1013: // bitwise_or
        implementation_1013();
        break;
      case 1014: // __or__
        implementation_1014();
        break;
      case 1015: // __or__
        implementation_1015();
        break;
      case 1016: // bitwise_xor
        implementation_1016();
        break;
      case 1017: // bitwise_xor
        implementation_1017();
        break;
      case 1018: // __xor__
        implementation_1018();
        break;
      case 1019: // __xor__
        implementation_1019();
        break;
      case 1020: // __lshift__
        implementation_1020();
        break;
      case 1021: // __lshift__
        implementation_1021();
        break;
      case 1022: // bitwise_left_shift
        implementation_1022();
        break;
      case 1023: // bitwise_left_shift
        implementation_1023();
        break;
      case 1024: // bitwise_left_shift
        implementation_1024();
        break;
      case 1025: // __rshift__
        implementation_1025();
        break;
      case 1026: // __rshift__
        implementation_1026();
        break;
      case 1027: // bitwise_right_shift
        implementation_1027();
        break;
      case 1028: // bitwise_right_shift
        implementation_1028();
        break;
      case 1029: // bitwise_right_shift
        implementation_1029();
        break;
      case 1030: // addbmm
        implementation_1030();
        break;
      case 1031: // addbmm
        implementation_1031();
        break;
      case 1032: // addbmm
        implementation_1032();
        break;
      case 1033: // diag
        implementation_1033();
        break;
      case 1034: // diag
        implementation_1034();
        break;
      case 1035: // diag_backward
        implementation_1035();
        break;
      case 1036: // cross
        implementation_1036();
        break;
      case 1037: // triu
        implementation_1037();
        break;
      case 1038: // triu
        implementation_1038();
        break;
      case 1039: // tril
        implementation_1039();
        break;
      case 1040: // tril
        implementation_1040();
        break;
      case 1041: // trace
        implementation_1041();
        break;
      case 1042: // trace_backward
        implementation_1042();
        break;
      case 1043: // ne
        implementation_1043();
        break;
      case 1044: // ne
        implementation_1044();
        break;
      case 1045: // not_equal
        implementation_1045();
        break;
      case 1046: // not_equal
        implementation_1046();
        break;
      case 1047: // eq
        implementation_1047();
        break;
      case 1048: // eq
        implementation_1048();
        break;
      case 1049: // ge
        implementation_1049();
        break;
      case 1050: // ge
        implementation_1050();
        break;
      case 1051: // greater_equal
        implementation_1051();
        break;
      case 1052: // greater_equal
        implementation_1052();
        break;
      case 1053: // le
        implementation_1053();
        break;
      case 1054: // le
        implementation_1054();
        break;
      case 1055: // less_equal
        implementation_1055();
        break;
      case 1056: // less_equal
        implementation_1056();
        break;
      case 1057: // gt
        implementation_1057();
        break;
      case 1058: // gt
        implementation_1058();
        break;
      case 1059: // greater
        implementation_1059();
        break;
      case 1060: // greater
        implementation_1060();
        break;
      case 1061: // lt
        implementation_1061();
        break;
      case 1062: // lt
        implementation_1062();
        break;
      case 1063: // less
        implementation_1063();
        break;
      case 1064: // less
        implementation_1064();
        break;
      case 1065: // take
        implementation_1065();
        break;
      case 1066: // take_along_dim
        implementation_1066();
        break;
      case 1067: // index_select
        implementation_1067();
        break;
      case 1068: // index_select_backward
        implementation_1068();
        break;
      case 1069: // masked_select
        implementation_1069();
        break;
      case 1070: // masked_select_backward
        implementation_1070();
        break;
      case 1071: // nonzero
        implementation_1071();
        break;
      case 1072: // nonzero_numpy
        implementation_1072();
        break;
      case 1073: // gather
        implementation_1073();
        break;
      case 1074: // gather
        implementation_1074();
        break;
      case 1075: // gather_backward
        implementation_1075();
        break;
      case 1076: // _gather_sparse_backward
        implementation_1076();
        break;
      case 1077: // addcmul
        implementation_1077();
        break;
      case 1078: // addcmul
        implementation_1078();
        break;
      case 1079: // addcdiv
        implementation_1079();
        break;
      case 1080: // addcdiv
        implementation_1080();
        break;
      case 1081: // cross_entropy_loss
        implementation_1081();
        break;
      case 1082: // cross_entropy_loss
        implementation_1082();
        break;
      case 1083: // cross_entropy_loss
        implementation_1083();
        break;
      case 1084: // cross_entropy_loss
        implementation_1084();
        break;
      case 1085: // cross_entropy_loss
        implementation_1085();
        break;
      case 1086: // lstsq
        implementation_1086();
        break;
      case 1087: // triangular_solve
        implementation_1087();
        break;
      case 1088: // triangular_solve
        implementation_1088();
        break;
      case 1089: // triangular_solve
        implementation_1089();
        break;
      case 1090: // triangular_solve
        implementation_1090();
        break;
      case 1091: // symeig
        implementation_1091();
        break;
      case 1092: // symeig
        implementation_1092();
        break;
      case 1093: // symeig
        implementation_1093();
        break;
      case 1094: // _symeig_helper
        implementation_1094();
        break;
      case 1095: // eig
        implementation_1095();
        break;
      case 1096: // eig
        implementation_1096();
        break;
      case 1097: // svd
        implementation_1097();
        break;
      case 1098: // svd
        implementation_1098();
        break;
      case 1099: // svd
        implementation_1099();
        break;
      case 1100: // _svd_helper
        implementation_1100();
        break;
      case 1101: // swapaxes
        implementation_1101();
        break;
      case 1102: // swapdims
        implementation_1102();
        break;
      case 1103: // cholesky
        implementation_1103();
        break;
      case 1104: // cholesky
        implementation_1104();
        break;
      case 1105: // cholesky_solve
        implementation_1105();
        break;
      case 1106: // cholesky_solve
        implementation_1106();
        break;
      case 1107: // _cholesky_solve_helper
        implementation_1107();
        break;
      case 1108: // solve
        implementation_1108();
        break;
      case 1109: // _solve_helper
        implementation_1109();
        break;
      case 1110: // cholesky_inverse
        implementation_1110();
        break;
      case 1111: // cholesky_inverse
        implementation_1111();
        break;
      case 1112: // qr
        implementation_1112();
        break;
      case 1113: // qr
        implementation_1113();
        break;
      case 1114: // geqrf
        implementation_1114();
        break;
      case 1115: // orgqr
        implementation_1115();
        break;
      case 1116: // ormqr
        implementation_1116();
        break;
      case 1117: // ormqr
        implementation_1117();
        break;
      case 1118: // ormqr
        implementation_1118();
        break;
      case 1119: // _lu_with_info
        implementation_1119();
        break;
      case 1120: // _lu_with_info
        implementation_1120();
        break;
      case 1121: // _lu_with_info
        implementation_1121();
        break;
      case 1122: // lu_solve
        implementation_1122();
        break;
      case 1123: // lu_unpack
        implementation_1123();
        break;
      case 1124: // lu_unpack
        implementation_1124();
        break;
      case 1125: // lu_unpack
        implementation_1125();
        break;
      case 1126: // multinomial
        implementation_1126();
        break;
      case 1127: // multinomial
        implementation_1127();
        break;
      case 1128: // lgamma
        implementation_1128();
        break;
      case 1129: // digamma
        implementation_1129();
        break;
      case 1130: // polygamma
        implementation_1130();
        break;
      case 1131: // erfinv
        implementation_1131();
        break;
      case 1132: // i0
        implementation_1132();
        break;
      case 1133: // sign
        implementation_1133();
        break;
      case 1134: // signbit
        implementation_1134();
        break;
      case 1135: // dist
        implementation_1135();
        break;
      case 1136: // dist
        implementation_1136();
        break;
      case 1137: // atan2
        implementation_1137();
        break;
      case 1138: // lerp
        implementation_1138();
        break;
      case 1139: // lerp
        implementation_1139();
        break;
      case 1140: // histc
        implementation_1140();
        break;
      case 1141: // histc
        implementation_1141();
        break;
      case 1142: // histc
        implementation_1142();
        break;
      case 1143: // histc
        implementation_1143();
        break;
      case 1144: // histogram
        implementation_1144();
        break;
      case 1145: // histogram
        implementation_1145();
        break;
      case 1146: // histogram
        implementation_1146();
        break;
      case 1147: // histogram
        implementation_1147();
        break;
      case 1148: // histogram
        implementation_1148();
        break;
      case 1149: // fmod
        implementation_1149();
        break;
      case 1150: // fmod
        implementation_1150();
        break;
      case 1151: // hypot
        implementation_1151();
        break;
      case 1152: // igamma
        implementation_1152();
        break;
      case 1153: // igammac
        implementation_1153();
        break;
      case 1154: // nextafter
        implementation_1154();
        break;
      case 1155: // remainder
        implementation_1155();
        break;
      case 1156: // remainder
        implementation_1156();
        break;
      case 1157: // remainder
        implementation_1157();
        break;
      case 1158: // min
        implementation_1158();
        break;
      case 1159: // fmin
        implementation_1159();
        break;
      case 1160: // max
        implementation_1160();
        break;
      case 1161: // fmax
        implementation_1161();
        break;
      case 1162: // maximum
        implementation_1162();
        break;
      case 1163: // max
        implementation_1163();
        break;
      case 1164: // minimum
        implementation_1164();
        break;
      case 1165: // min
        implementation_1165();
        break;
      case 1166: // quantile
        implementation_1166();
        break;
      case 1167: // quantile
        implementation_1167();
        break;
      case 1168: // nanquantile
        implementation_1168();
        break;
      case 1169: // nanquantile
        implementation_1169();
        break;
      case 1170: // sort
        implementation_1170();
        break;
      case 1171: // sort
        implementation_1171();
        break;
      case 1172: // sort
        implementation_1172();
        break;
      case 1173: // msort
        implementation_1173();
        break;
      case 1174: // argsort
        implementation_1174();
        break;
      case 1175: // argsort
        implementation_1175();
        break;
      case 1176: // argsort
        implementation_1176();
        break;
      case 1177: // topk
        implementation_1177();
        break;
      case 1178: // topk
        implementation_1178();
        break;
      case 1179: // topk
        implementation_1179();
        break;
      case 1180: // topk
        implementation_1180();
        break;
      case 1181: // all
        implementation_1181();
        break;
      case 1182: // any
        implementation_1182();
        break;
      case 1183: // renorm
        implementation_1183();
        break;
      case 1184: // unfold
        implementation_1184();
        break;
      case 1185: // unfold_backward
        implementation_1185();
        break;
      case 1186: // equal
        implementation_1186();
        break;
      case 1187: // pow
        implementation_1187();
        break;
      case 1188: // pow
        implementation_1188();
        break;
      case 1189: // pow
        implementation_1189();
        break;
      case 1190: // float_power
        implementation_1190();
        break;
      case 1191: // float_power
        implementation_1191();
        break;
      case 1192: // float_power
        implementation_1192();
        break;
      case 1193: // alias
        implementation_1193();
        break;
      case 1194: // _cat
        implementation_1194();
        break;
      case 1195: // _cat
        implementation_1195();
        break;
      case 1196: // _foreach_add
        implementation_1196();
        break;
      case 1197: // _foreach_sub
        implementation_1197();
        break;
      case 1198: // _foreach_mul
        implementation_1198();
        break;
      case 1199: // _foreach_div
        implementation_1199();
        break;
      case 1200: // _foreach_add
        implementation_1200();
        break;
      case 1201: // _foreach_add
        implementation_1201();
        break;
      case 1202: // _foreach_sub
        implementation_1202();
        break;
      case 1203: // _foreach_sub
        implementation_1203();
        break;
      case 1204: // _foreach_mul
        implementation_1204();
        break;
      case 1205: // _foreach_div
        implementation_1205();
        break;
      case 1206: // _foreach_exp
        implementation_1206();
        break;
      case 1207: // _foreach_sqrt
        implementation_1207();
        break;
      case 1208: // _foreach_abs
        implementation_1208();
        break;
      case 1209: // _foreach_acos
        implementation_1209();
        break;
      case 1210: // _foreach_asin
        implementation_1210();
        break;
      case 1211: // _foreach_atan
        implementation_1211();
        break;
      case 1212: // _foreach_ceil
        implementation_1212();
        break;
      case 1213: // _foreach_cos
        implementation_1213();
        break;
      case 1214: // _foreach_cosh
        implementation_1214();
        break;
      case 1215: // _foreach_erf
        implementation_1215();
        break;
      case 1216: // _foreach_erfc
        implementation_1216();
        break;
      case 1217: // _foreach_expm1
        implementation_1217();
        break;
      case 1218: // _foreach_floor
        implementation_1218();
        break;
      case 1219: // _foreach_log
        implementation_1219();
        break;
      case 1220: // _foreach_log10
        implementation_1220();
        break;
      case 1221: // _foreach_log1p
        implementation_1221();
        break;
      case 1222: // _foreach_log2
        implementation_1222();
        break;
      case 1223: // _foreach_neg
        implementation_1223();
        break;
      case 1224: // _foreach_tan
        implementation_1224();
        break;
      case 1225: // _foreach_tanh
        implementation_1225();
        break;
      case 1226: // _foreach_sin
        implementation_1226();
        break;
      case 1227: // _foreach_sinh
        implementation_1227();
        break;
      case 1228: // _foreach_round
        implementation_1228();
        break;
      case 1229: // _foreach_lgamma
        implementation_1229();
        break;
      case 1230: // _foreach_frac
        implementation_1230();
        break;
      case 1231: // _foreach_reciprocal
        implementation_1231();
        break;
      case 1232: // _foreach_sigmoid
        implementation_1232();
        break;
      case 1233: // _foreach_trunc
        implementation_1233();
        break;
      case 1234: // _foreach_addcdiv
        implementation_1234();
        break;
      case 1235: // _foreach_addcdiv
        implementation_1235();
        break;
      case 1236: // _foreach_addcmul
        implementation_1236();
        break;
      case 1237: // _foreach_addcmul
        implementation_1237();
        break;
      case 1238: // _foreach_maximum
        implementation_1238();
        break;
      case 1239: // _foreach_minimum
        implementation_1239();
        break;
      case 1240: // bucketize
        implementation_1240();
        break;
      case 1241: // bucketize
        implementation_1241();
        break;
      case 1242: // bucketize
        implementation_1242();
        break;
      case 1243: // bucketize
        implementation_1243();
        break;
      case 1244: // bucketize
        implementation_1244();
        break;
      case 1245: // bucketize
        implementation_1245();
        break;
      case 1246: // searchsorted
        implementation_1246();
        break;
      case 1247: // searchsorted
        implementation_1247();
        break;
      case 1248: // searchsorted
        implementation_1248();
        break;
      case 1249: // searchsorted
        implementation_1249();
        break;
      case 1250: // searchsorted
        implementation_1250();
        break;
      case 1251: // searchsorted
        implementation_1251();
        break;
      case 1252: // _convert_indices_from_coo_to_csr
        implementation_1252();
        break;
      case 1253: // _convert_indices_from_coo_to_csr
        implementation_1253();
        break;
      case 1254: // mse_loss
        implementation_1254();
        break;
      case 1255: // mse_loss
        implementation_1255();
        break;
      case 1256: // mse_loss_backward
        implementation_1256();
        break;
      case 1257: // l1_loss
        implementation_1257();
        break;
      case 1258: // l1_loss
        implementation_1258();
        break;
      case 1259: // l1_loss_backward
        implementation_1259();
        break;
      case 1260: // multi_margin_loss
        implementation_1260();
        break;
      case 1261: // multi_margin_loss
        implementation_1261();
        break;
      case 1262: // multi_margin_loss
        implementation_1262();
        break;
      case 1263: // multi_margin_loss
        implementation_1263();
        break;
      case 1264: // multi_margin_loss
        implementation_1264();
        break;
      case 1265: // multi_margin_loss_backward
        implementation_1265();
        break;
      case 1266: // multi_margin_loss_backward
        implementation_1266();
        break;
      case 1267: // multi_margin_loss_backward
        implementation_1267();
        break;
      case 1268: // multilabel_margin_loss
        implementation_1268();
        break;
      case 1269: // multilabel_margin_loss
        implementation_1269();
        break;
      case 1270: // multilabel_margin_loss_forward
        implementation_1270();
        break;
      case 1271: // multilabel_margin_loss_backward
        implementation_1271();
        break;
      case 1272: // nll_loss_nd
        implementation_1272();
        break;
      case 1273: // nll_loss_nd
        implementation_1273();
        break;
      case 1274: // nll_loss_nd
        implementation_1274();
        break;
      case 1275: // nll_loss_nd
        implementation_1275();
        break;
      case 1276: // nll_loss
        implementation_1276();
        break;
      case 1277: // nll_loss
        implementation_1277();
        break;
      case 1278: // nll_loss
        implementation_1278();
        break;
      case 1279: // nll_loss
        implementation_1279();
        break;
      case 1280: // nll_loss_forward
        implementation_1280();
        break;
      case 1281: // nll_loss_backward
        implementation_1281();
        break;
      case 1282: // nll_loss2d
        implementation_1282();
        break;
      case 1283: // nll_loss2d
        implementation_1283();
        break;
      case 1284: // nll_loss2d
        implementation_1284();
        break;
      case 1285: // nll_loss2d
        implementation_1285();
        break;
      case 1286: // nll_loss2d_forward
        implementation_1286();
        break;
      case 1287: // nll_loss2d_backward
        implementation_1287();
        break;
      case 1288: // smooth_l1_loss
        implementation_1288();
        break;
      case 1289: // smooth_l1_loss
        implementation_1289();
        break;
      case 1290: // smooth_l1_loss
        implementation_1290();
        break;
      case 1291: // smooth_l1_loss_backward
        implementation_1291();
        break;
      case 1292: // huber_loss
        implementation_1292();
        break;
      case 1293: // huber_loss
        implementation_1293();
        break;
      case 1294: // huber_loss
        implementation_1294();
        break;
      case 1295: // huber_loss_backward
        implementation_1295();
        break;
      case 1296: // soft_margin_loss
        implementation_1296();
        break;
      case 1297: // soft_margin_loss
        implementation_1297();
        break;
      case 1298: // soft_margin_loss_backward
        implementation_1298();
        break;
      case 1299: // elu
        implementation_1299();
        break;
      case 1300: // elu
        implementation_1300();
        break;
      case 1301: // elu
        implementation_1301();
        break;
      case 1302: // elu
        implementation_1302();
        break;
      case 1303: // elu_backward
        implementation_1303();
        break;
      case 1304: // glu
        implementation_1304();
        break;
      case 1305: // glu
        implementation_1305();
        break;
      case 1306: // glu_backward
        implementation_1306();
        break;
      case 1307: // hardsigmoid
        implementation_1307();
        break;
      case 1308: // hardsigmoid_backward
        implementation_1308();
        break;
      case 1309: // hardtanh
        implementation_1309();
        break;
      case 1310: // hardtanh
        implementation_1310();
        break;
      case 1311: // hardtanh
        implementation_1311();
        break;
      case 1312: // hardtanh_backward
        implementation_1312();
        break;
      case 1313: // hardswish
        implementation_1313();
        break;
      case 1314: // hardswish_backward
        implementation_1314();
        break;
      case 1315: // leaky_relu
        implementation_1315();
        break;
      case 1316: // leaky_relu
        implementation_1316();
        break;
      case 1317: // leaky_relu_backward
        implementation_1317();
        break;
      case 1318: // log_sigmoid
        implementation_1318();
        break;
      case 1319: // log_sigmoid_forward
        implementation_1319();
        break;
      case 1320: // log_sigmoid_backward
        implementation_1320();
        break;
      case 1321: // rrelu_with_noise
        implementation_1321();
        break;
      case 1322: // rrelu_with_noise
        implementation_1322();
        break;
      case 1323: // rrelu_with_noise
        implementation_1323();
        break;
      case 1324: // rrelu_with_noise
        implementation_1324();
        break;
      case 1325: // rrelu_with_noise_backward
        implementation_1325();
        break;
      case 1326: // softplus
        implementation_1326();
        break;
      case 1327: // softplus
        implementation_1327();
        break;
      case 1328: // softplus
        implementation_1328();
        break;
      case 1329: // softplus_backward
        implementation_1329();
        break;
      case 1330: // softshrink
        implementation_1330();
        break;
      case 1331: // softshrink
        implementation_1331();
        break;
      case 1332: // softshrink_backward
        implementation_1332();
        break;
      case 1333: // adaptive_avg_pool2d
        implementation_1333();
        break;
      case 1334: // mkldnn_adaptive_avg_pool2d
        implementation_1334();
        break;
      case 1335: // mkldnn_adaptive_avg_pool2d_backward
        implementation_1335();
        break;
      case 1336: // _adaptive_avg_pool2d
        implementation_1336();
        break;
      case 1337: // _adaptive_avg_pool2d_backward
        implementation_1337();
        break;
      case 1338: // adaptive_avg_pool3d
        implementation_1338();
        break;
      case 1339: // _adaptive_avg_pool3d
        implementation_1339();
        break;
      case 1340: // _adaptive_avg_pool3d_backward
        implementation_1340();
        break;
      case 1341: // adaptive_max_pool2d
        implementation_1341();
        break;
      case 1342: // adaptive_max_pool2d_backward
        implementation_1342();
        break;
      case 1343: // adaptive_max_pool3d
        implementation_1343();
        break;
      case 1344: // adaptive_max_pool3d_backward
        implementation_1344();
        break;
      case 1345: // avg_pool2d
        implementation_1345();
        break;
      case 1346: // avg_pool2d
        implementation_1346();
        break;
      case 1347: // avg_pool2d
        implementation_1347();
        break;
      case 1348: // avg_pool2d
        implementation_1348();
        break;
      case 1349: // avg_pool2d
        implementation_1349();
        break;
      case 1350: // avg_pool3d
        implementation_1350();
        break;
      case 1351: // avg_pool3d
        implementation_1351();
        break;
      case 1352: // avg_pool3d
        implementation_1352();
        break;
      case 1353: // avg_pool3d
        implementation_1353();
        break;
      case 1354: // avg_pool3d
        implementation_1354();
        break;
      case 1355: // fractional_max_pool2d
        implementation_1355();
        break;
      case 1356: // fractional_max_pool2d_backward
        implementation_1356();
        break;
      case 1357: // fractional_max_pool3d
        implementation_1357();
        break;
      case 1358: // fractional_max_pool3d_backward
        implementation_1358();
        break;
      case 1359: // max_pool2d_with_indices
        implementation_1359();
        break;
      case 1360: // max_pool2d_with_indices
        implementation_1360();
        break;
      case 1361: // max_pool2d_with_indices
        implementation_1361();
        break;
      case 1362: // max_pool2d_with_indices
        implementation_1362();
        break;
      case 1363: // max_pool2d_with_indices
        implementation_1363();
        break;
      case 1364: // max_pool2d_with_indices_backward
        implementation_1364();
        break;
      case 1365: // max_pool3d_with_indices
        implementation_1365();
        break;
      case 1366: // max_pool3d_with_indices
        implementation_1366();
        break;
      case 1367: // max_pool3d_with_indices
        implementation_1367();
        break;
      case 1368: // max_pool3d_with_indices
        implementation_1368();
        break;
      case 1369: // max_pool3d_with_indices
        implementation_1369();
        break;
      case 1370: // max_pool3d_with_indices_backward
        implementation_1370();
        break;
      case 1371: // max_unpool2d
        implementation_1371();
        break;
      case 1372: // max_unpool2d_backward
        implementation_1372();
        break;
      case 1373: // max_unpool3d
        implementation_1373();
        break;
      case 1374: // max_unpool3d_backward
        implementation_1374();
        break;
      case 1375: // reflection_pad1d
        implementation_1375();
        break;
      case 1376: // reflection_pad1d_backward
        implementation_1376();
        break;
      case 1377: // reflection_pad2d
        implementation_1377();
        break;
      case 1378: // reflection_pad2d_backward
        implementation_1378();
        break;
      case 1379: // reflection_pad3d
        implementation_1379();
        break;
      case 1380: // reflection_pad3d_backward
        implementation_1380();
        break;
      case 1381: // replication_pad1d
        implementation_1381();
        break;
      case 1382: // replication_pad1d_backward
        implementation_1382();
        break;
      case 1383: // replication_pad2d
        implementation_1383();
        break;
      case 1384: // replication_pad2d_backward
        implementation_1384();
        break;
      case 1385: // replication_pad3d
        implementation_1385();
        break;
      case 1386: // replication_pad3d_backward
        implementation_1386();
        break;
      case 1387: // upsample_linear1d
        implementation_1387();
        break;
      case 1388: // upsample_linear1d_backward
        implementation_1388();
        break;
      case 1389: // upsample_bilinear2d
        implementation_1389();
        break;
      case 1390: // upsample_bilinear2d_backward
        implementation_1390();
        break;
      case 1391: // upsample_bicubic2d
        implementation_1391();
        break;
      case 1392: // upsample_bicubic2d_backward
        implementation_1392();
        break;
      case 1393: // upsample_trilinear3d
        implementation_1393();
        break;
      case 1394: // upsample_trilinear3d_backward
        implementation_1394();
        break;
      case 1395: // upsample_nearest1d
        implementation_1395();
        break;
      case 1396: // upsample_nearest1d_backward
        implementation_1396();
        break;
      case 1397: // upsample_nearest2d
        implementation_1397();
        break;
      case 1398: // upsample_nearest2d_backward
        implementation_1398();
        break;
      case 1399: // upsample_nearest3d
        implementation_1399();
        break;
      case 1400: // upsample_nearest3d_backward
        implementation_1400();
        break;
      case 1401: // sigmoid_backward
        implementation_1401();
        break;
      case 1402: // logit_backward
        implementation_1402();
        break;
      case 1403: // tanh_backward
        implementation_1403();
        break;
      case 1404: // slow_conv_transpose2d
        implementation_1404();
        break;
      case 1405: // slow_conv_transpose2d
        implementation_1405();
        break;
      case 1406: // slow_conv_transpose2d
        implementation_1406();
        break;
      case 1407: // slow_conv_transpose2d
        implementation_1407();
        break;
      case 1408: // slow_conv_transpose2d
        implementation_1408();
        break;
      case 1409: // slow_conv_transpose2d
        implementation_1409();
        break;
      case 1410: // slow_conv_transpose2d_backward
        implementation_1410();
        break;
      case 1411: // slow_conv_transpose3d
        implementation_1411();
        break;
      case 1412: // slow_conv_transpose3d
        implementation_1412();
        break;
      case 1413: // slow_conv_transpose3d
        implementation_1413();
        break;
      case 1414: // slow_conv_transpose3d
        implementation_1414();
        break;
      case 1415: // slow_conv_transpose3d
        implementation_1415();
        break;
      case 1416: // slow_conv_transpose3d
        implementation_1416();
        break;
      case 1417: // slow_conv_transpose3d_backward
        implementation_1417();
        break;
      case 1418: // thnn_conv2d
        implementation_1418();
        break;
      case 1419: // thnn_conv2d
        implementation_1419();
        break;
      case 1420: // thnn_conv2d
        implementation_1420();
        break;
      case 1421: // thnn_conv2d
        implementation_1421();
        break;
      case 1422: // thnn_conv2d_forward
        implementation_1422();
        break;
      case 1423: // thnn_conv2d_backward
        implementation_1423();
        break;
      case 1424: // _conv_depthwise2d
        implementation_1424();
        break;
      case 1425: // _conv_depthwise2d_backward
        implementation_1425();
        break;
      case 1426: // conv_depthwise3d
        implementation_1426();
        break;
      case 1427: // conv_depthwise3d_backward
        implementation_1427();
        break;
      case 1428: // slow_conv3d
        implementation_1428();
        break;
      case 1429: // slow_conv3d
        implementation_1429();
        break;
      case 1430: // slow_conv3d
        implementation_1430();
        break;
      case 1431: // slow_conv3d
        implementation_1431();
        break;
      case 1432: // slow_conv3d_forward
        implementation_1432();
        break;
      case 1433: // slow_conv3d_backward
        implementation_1433();
        break;
      case 1434: // slow_conv_dilated2d
        implementation_1434();
        break;
      case 1435: // slow_conv_dilated2d
        implementation_1435();
        break;
      case 1436: // slow_conv_dilated2d
        implementation_1436();
        break;
      case 1437: // slow_conv_dilated2d
        implementation_1437();
        break;
      case 1438: // slow_conv_dilated2d
        implementation_1438();
        break;
      case 1439: // slow_conv_dilated2d_backward
        implementation_1439();
        break;
      case 1440: // slow_conv_dilated3d
        implementation_1440();
        break;
      case 1441: // slow_conv_dilated3d
        implementation_1441();
        break;
      case 1442: // slow_conv_dilated3d
        implementation_1442();
        break;
      case 1443: // slow_conv_dilated3d
        implementation_1443();
        break;
      case 1444: // slow_conv_dilated3d
        implementation_1444();
        break;
      case 1445: // slow_conv_dilated3d_backward
        implementation_1445();
        break;
      case 1446: // col2im
        implementation_1446();
        break;
      case 1447: // col2im_backward
        implementation_1447();
        break;
      case 1448: // column_stack
        implementation_1448();
        break;
      case 1449: // im2col
        implementation_1449();
        break;
      case 1450: // im2col_backward
        implementation_1450();
        break;
      case 1451: // isfinite
        implementation_1451();
        break;
      case 1452: // isinf
        implementation_1452();
        break;
      case 1453: // isposinf
        implementation_1453();
        break;
      case 1454: // isneginf
        implementation_1454();
        break;
      case 1455: // _add_batch_dim
        implementation_1455();
        break;
      case 1456: // _remove_batch_dim
        implementation_1456();
        break;
      case 1457: // special_entr
        implementation_1457();
        break;
      case 1458: // special_ndtri
        implementation_1458();
        break;
      case 1459: // special_expm1
        implementation_1459();
        break;
      case 1460: // special_exp2
        implementation_1460();
        break;
      case 1461: // special_psi
        implementation_1461();
        break;
      case 1462: // special_digamma
        implementation_1462();
        break;
      case 1463: // special_gammaln
        implementation_1463();
        break;
      case 1464: // special_erf
        implementation_1464();
        break;
      case 1465: // special_erfc
        implementation_1465();
        break;
      case 1466: // special_erfcx
        implementation_1466();
        break;
      case 1467: // special_erfinv
        implementation_1467();
        break;
      case 1468: // special_ndtr
        implementation_1468();
        break;
      case 1469: // special_xlog1py
        implementation_1469();
        break;
      case 1470: // special_xlog1py
        implementation_1470();
        break;
      case 1471: // special_xlog1py
        implementation_1471();
        break;
      case 1472: // special_xlogy
        implementation_1472();
        break;
      case 1473: // special_xlogy
        implementation_1473();
        break;
      case 1474: // special_xlogy
        implementation_1474();
        break;
      case 1475: // special_zeta
        implementation_1475();
        break;
      case 1476: // special_zeta
        implementation_1476();
        break;
      case 1477: // special_zeta
        implementation_1477();
        break;
      case 1478: // special_i0
        implementation_1478();
        break;
      case 1479: // special_i0e
        implementation_1479();
        break;
      case 1480: // special_i1
        implementation_1480();
        break;
      case 1481: // special_i1e
        implementation_1481();
        break;
      case 1482: // special_logit
        implementation_1482();
        break;
      case 1483: // special_polygamma
        implementation_1483();
        break;
      case 1484: // special_logsumexp
        implementation_1484();
        break;
      case 1485: // special_logsumexp
        implementation_1485();
        break;
      case 1486: // special_expit
        implementation_1486();
        break;
      case 1487: // special_sinc
        implementation_1487();
        break;
      case 1488: // special_round
        implementation_1488();
        break;
      case 1489: // special_log1p
        implementation_1489();
        break;
      case 1490: // special_log_softmax
        implementation_1490();
        break;
      case 1491: // special_gammainc
        implementation_1491();
        break;
      case 1492: // special_gammaincc
        implementation_1492();
        break;
      case 1493: // special_multigammaln
        implementation_1493();
        break;
      case 1494: // fft_fft
        implementation_1494();
        break;
      case 1495: // fft_ifft
        implementation_1495();
        break;
      case 1496: // fft_rfft
        implementation_1496();
        break;
      case 1497: // fft_irfft
        implementation_1497();
        break;
      case 1498: // fft_hfft
        implementation_1498();
        break;
      case 1499: // fft_ihfft
        implementation_1499();
        break;
      case 1500: // fft_fft2
        implementation_1500();
        break;
      case 1501: // fft_ifft2
        implementation_1501();
        break;
      case 1502: // fft_rfft2
        implementation_1502();
        break;
      case 1503: // fft_irfft2
        implementation_1503();
        break;
      case 1504: // fft_fftn
        implementation_1504();
        break;
      case 1505: // fft_ifftn
        implementation_1505();
        break;
      case 1506: // fft_rfftn
        implementation_1506();
        break;
      case 1507: // fft_irfftn
        implementation_1507();
        break;
      case 1508: // fft_fftshift
        implementation_1508();
        break;
      case 1509: // fft_ifftshift
        implementation_1509();
        break;
      case 1510: // linalg_cholesky_ex
        implementation_1510();
        break;
      case 1511: // linalg_cholesky_ex
        implementation_1511();
        break;
      case 1512: // linalg_cholesky_ex
        implementation_1512();
        break;
      case 1513: // linalg_cholesky
        implementation_1513();
        break;
      case 1514: // linalg_cholesky
        implementation_1514();
        break;
      case 1515: // linalg_det
        implementation_1515();
        break;
      case 1516: // det
        implementation_1516();
        break;
      case 1517: // _det_lu_based_helper
        implementation_1517();
        break;
      case 1518: // _det_lu_based_helper_backward_helper
        implementation_1518();
        break;
      case 1519: // linalg_lstsq
        implementation_1519();
        break;
      case 1520: // linalg_matmul
        implementation_1520();
        break;
      case 1521: // linalg_slogdet
        implementation_1521();
        break;
      case 1522: // linalg_eig
        implementation_1522();
        break;
      case 1523: // linalg_eigvals
        implementation_1523();
        break;
      case 1524: // linalg_eigh
        implementation_1524();
        break;
      case 1525: // linalg_eigvalsh
        implementation_1525();
        break;
      case 1526: // linalg_householder_product
        implementation_1526();
        break;
      case 1527: // linalg_inv_ex
        implementation_1527();
        break;
      case 1528: // linalg_inv_ex
        implementation_1528();
        break;
      case 1529: // linalg_inv
        implementation_1529();
        break;
      case 1530: // inner
        implementation_1530();
        break;
      case 1531: // outer
        implementation_1531();
        break;
      case 1532: // ger
        implementation_1532();
        break;
      case 1533: // linalg_norm
        implementation_1533();
        break;
      case 1534: // linalg_vector_norm
        implementation_1534();
        break;
      case 1535: // linalg_vector_norm
        implementation_1535();
        break;
      case 1536: // linalg_matrix_norm
        implementation_1536();
        break;
      case 1537: // linalg_matrix_norm
        implementation_1537();
        break;
      case 1538: // linalg_matrix_norm
        implementation_1538();
        break;
      case 1539: // linalg_matrix_norm
        implementation_1539();
        break;
      case 1540: // linalg_svd
        implementation_1540();
        break;
      case 1541: // linalg_svd
        implementation_1541();
        break;
      case 1542: // linalg_svdvals
        implementation_1542();
        break;
      case 1543: // linalg_cond
        implementation_1543();
        break;
      case 1544: // linalg_pinv
        implementation_1544();
        break;
      case 1545: // linalg_pinv
        implementation_1545();
        break;
      case 1546: // linalg_pinv
        implementation_1546();
        break;
      case 1547: // linalg_pinv
        implementation_1547();
        break;
      case 1548: // linalg_pinv
        implementation_1548();
        break;
      case 1549: // linalg_solve
        implementation_1549();
        break;
      case 1550: // linalg_tensorinv
        implementation_1550();
        break;
      case 1551: // linalg_tensorinv
        implementation_1551();
        break;
      case 1552: // linalg_tensorsolve
        implementation_1552();
        break;
      case 1553: // linalg_qr
        implementation_1553();
        break;
      case 1554: // linalg_matrix_power
        implementation_1554();
        break;
      case 1555: // linalg_matrix_rank
        implementation_1555();
        break;
      case 1556: // linalg_matrix_rank
        implementation_1556();
        break;
      case 1557: // linalg_matrix_rank
        implementation_1557();
        break;
      case 1558: // linalg_multi_dot
        implementation_1558();
        break;
      case 1559: // _test_serialization_subcmul
        implementation_1559();
        break;
      case 1560: // _test_serialization_subcmul
        implementation_1560();
        break;
      case 1561: // _test_string_default
        implementation_1561();
        break;
      case 1562: // _test_ambiguous_defaults
        implementation_1562();
        break;
      case 1563: // _test_ambiguous_defaults
        implementation_1563();
        break;
      case 1564: // _test_ambiguous_defaults
        implementation_1564();
        break;
      case 1565: // pad_sequence
        implementation_1565();
        break;
      case 1566: // pad_sequence
        implementation_1566();
        break;
      case 1567: // pad_sequence
        implementation_1567();
        break;
      case 1568: // flatten_dense_tensors
        implementation_1568();
        break;
      case 1569: // unflatten_dense_tensors
        implementation_1569();
        break;
      default:
        CAFFE_THROW("Unexpected key value for aten operator");
    }
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    return run_op();
  }
private:
  // actual operator implementation is initialized in ctor.
  std::function<bool()> run_op;
  at::Backend backend() const;

  TypeMeta typeMetaFor(const at::Tensor & t) {
    return typeMetaFor(t.scalar_type());
  }
  TypeMeta typeMetaFor(at::ScalarType st) {
    #define DEFINE_CASE(ctype,aten_name) \
      case at::k##aten_name: \
        return TypeMeta::Make<ctype>();
    switch(st) {
      AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, DEFINE_CASE)
    default:
      CAFFE_THROW("Unknown ATen Type");
    }
    #undef DEFINE_CASE
  }

  at::TensorOptions optionsFor(const Tensor& ten) {
    at::Device device = ten.GetDevice();
#ifdef __HIP_PLATFORM_HCC__
    if (backend() == at::Backend::HIP) {
      device = at::Device(kCUDA, device.index());
    }
#endif
    return at::TensorOptions(device).dtype(ten.dtype());
  }

  at::Tensor tensorWrapping(const Tensor& ten_) {
    auto& ten = const_cast<Tensor&>(ten_);
    return at::from_blob(
        ten.raw_mutable_data(),
        ten.sizes(),
        optionsFor(ten));
  }

  at::Tensor peek(size_t i, size_t N) {
    auto real_idx = InputSize() - N + i;
    return tensorWrapping(Input(real_idx));
  }

  std::vector<at::Tensor> peekSlice(size_t i, size_t len, size_t N) {
    std::vector<at::Tensor> results;
    results.reserve(len);
    for (size_t ii = i; ii < i + len; ++ii) {
      results.push_back(peek(ii, N));
    }
    return results;
  }

  torch::List<c10::optional<at::Tensor>> peekSliceOptionals(size_t i, size_t len, size_t N) {
    torch::List<c10::optional<at::Tensor>> results;
    results.reserve(len);
    for (size_t ii = i; ii < i + len; ++ii) {
      results.push_back(peek(ii, N));
    }
    return results;
  }

  void assignTo(Tensor* dst, const at::Tensor& src_) {
    at::Tensor src = src_.contiguous();
    auto at_sizes = src.sizes();
    caffe2::TypeMeta type_meta = typeMetaFor(src);
    at::Device device = src.device();
#ifdef __HIP_PLATFORM_HCC__
    if (device.is_cuda()) {
      device = at::Device(at::DeviceType::HIP, device.index());
    }
#endif
    at::TensorImpl* src_impl = src.unsafeReleaseTensorImpl();
    std::vector<int64_t> dims(at_sizes.begin(), at_sizes.end());
    dst->Resize(dims);
    dst->ShareExternalPointer(
        at::DataPtr(
            src_impl->data(),
            static_cast<void*>(src_impl),
            [](void* t_ptr) -> void {
              at::TensorImpl* local_impl = static_cast<at::TensorImpl*>(t_ptr);
              c10::raw::intrusive_ptr::decref(local_impl);
            },
            device),
        type_meta,
        0);
  }
  void assignListStartingAt(
      size_t offset,
      const std::vector<at::Tensor>& tensors) {
    for (size_t i = 0; i < tensors.size(); i++) {
      assignTo(Output(offset + i), tensors[i]);
    }
  }

  template<typename T,
          typename std::enable_if<std::numeric_limits<T>::is_integer, bool>::type* =
              nullptr>
  int64_t extract(const at::Scalar &s) {
    return s.toLong();
  }

  template<typename T,
          typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type* =
              nullptr>
  int64_t extract(const at::Scalar &s) {
    return s.toDouble();
  }

  void assignTo(Tensor* dst, at::ScalarType scalar_type, const at::Scalar& scalar) {
    switch(scalar_type) {
      #define DEFINE_CASE(ctype,aten_name) \
        case at::k##aten_name: { \
          auto value = extract<ctype>(scalar); \
          assignToValue<ctype>(dst, at::convert<ctype,decltype(value)>(value)); \
        } break;
      AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, DEFINE_CASE)
#undef DEFINE_CASE
      default:
        CAFFE_THROW("Unknown ATen Type");
    }
  }
  template <typename T>
  void assignToValue(Tensor* dst, T v) {
    dst->Resize(std::vector<int64_t>());
    math::Set(1, v, dst->template mutable_data<T>(), &context_);
  }
  int findImplementation(const OperatorDef& operator_def) {
    CAFFE_ENFORCE(HasArgument("operator"));
    std::string op = OperatorBase::GetSingleArgument<std::string>("operator", "");
    // construct descriptor string ([DESCRIPTORS]) given the attributes
    // and inputs of this operator_def, and look up the implementation key
    // for this variant
    std::stringstream descriptor;
    descriptor << op;
    std::vector<std::string> attrs;
    for(size_t i = 0; i < operator_def.arg_size(); i++) {
      auto & attr = operator_def.arg(i);
      if(attr.name() == "operator" || attr.name() == "type" )
        continue;
      attrs.push_back(attr.name());
    }
    std::sort(attrs.begin(), attrs.end());
    for(auto & a : attrs)
      descriptor << "-" << a;

    std::string descriptor_sized =
        descriptor.str() + "-" + c10::to_string(InputSize());
    std::string descriptor_var_args = descriptor.str() + "-*";
    if (op_to_key.count(descriptor_sized) > 0) {
      return op_to_key[descriptor_sized];
    }
    if (op_to_key.count(descriptor_var_args) > 0) {
      return op_to_key[descriptor_var_args];
    }
    std::stringstream ss;
    ss << "Attempting to run unknown ATen operator configuration: "
       << descriptor_sized;
    CAFFE_THROW(ss.str());
  }
  at::Scalar readScalarAttribute(const std::string & name) {
    if(OperatorBase::HasSingleArgumentOfType<int64_t>(name)) {
      return OperatorBase::GetSingleArgument<int64_t>(name, 0);
    } else {
      CAFFE_ENFORCE(OperatorBase::HasSingleArgumentOfType<float>(name));
      return OperatorBase::GetSingleArgument<float>(name, 0);
    }
  }
  template<typename T>
  T readAttribute(const std::string & name) {
    CAFFE_ENFORCE(OperatorBase::HasSingleArgumentOfType<T>(name));
    return OperatorBase::GetSingleArgument<T>(name, 0);
  }
  std::vector<int64_t> readIntArrayRef(const std::string & name) {
    CAFFE_ENFORCE(OperatorBase::HasArgument(name));
    return OperatorBase::GetRepeatedArgument<int64_t>(name, {});
  }
  template <int N>
  std::array<bool, N> readBoolMask(const std::string& name) {
    CAFFE_ENFORCE(OperatorBase::HasArgument(name));
    std::vector<int64_t> ints =
        OperatorBase::GetRepeatedArgument<int64_t>(name, {});
    std::array<bool, N> result;
    for (size_t i = 0; i < N; ++i) {
      result[i] = ints.at(i);
    }
    return result;
  }

  C10_NOINLINE void implementation_0() { // _cast_Byte
      bool non_blocking = readAttribute<int64_t>("non_blocking");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_cast_Byte(self, non_blocking);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1() { // _cast_Byte
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_cast_Byte(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_2() { // _cast_Char
      bool non_blocking = readAttribute<int64_t>("non_blocking");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_cast_Char(self, non_blocking);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_3() { // _cast_Char
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_cast_Char(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_4() { // _cast_Double
      bool non_blocking = readAttribute<int64_t>("non_blocking");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_cast_Double(self, non_blocking);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_5() { // _cast_Double
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_cast_Double(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_6() { // _cast_Float
      bool non_blocking = readAttribute<int64_t>("non_blocking");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_cast_Float(self, non_blocking);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_7() { // _cast_Float
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_cast_Float(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_8() { // _cast_Int
      bool non_blocking = readAttribute<int64_t>("non_blocking");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_cast_Int(self, non_blocking);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_9() { // _cast_Int
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_cast_Int(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_10() { // _cast_Long
      bool non_blocking = readAttribute<int64_t>("non_blocking");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_cast_Long(self, non_blocking);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_11() { // _cast_Long
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_cast_Long(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_12() { // _cast_Short
      bool non_blocking = readAttribute<int64_t>("non_blocking");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_cast_Short(self, non_blocking);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_13() { // _cast_Short
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_cast_Short(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_14() { // _cast_Half
      bool non_blocking = readAttribute<int64_t>("non_blocking");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_cast_Half(self, non_blocking);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_15() { // _cast_Half
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_cast_Half(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_16() { // data
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = self.data();
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_17() { // is_leaf
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = self.is_leaf();
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_18() { // output_nr
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = self.output_nr();
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_19() { // _version
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = self._version();
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_20() { // retains_grad
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = self.retains_grad();
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_21() { // _fw_primal
      int64_t level = readAttribute<int64_t>("level");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = self._fw_primal(level);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_22() { // _make_dual
      int64_t level = readAttribute<int64_t>("level");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto primal = peek(0, 2);
          auto tangent = peek(1, 2);
          auto the_result = at::_make_dual(primal, tangent, level);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_23() { // _unpack_dual
      int64_t level = readAttribute<int64_t>("level");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto dual = peek(0, 1);
          auto the_result = at::_unpack_dual(dual, level);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_24() { // align_as
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = self.align_as(other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_25() { // align_tensors
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::align_tensors(tensors);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_26() { // _use_cudnn_ctc_loss
      auto input_lengths = readIntArrayRef("input_lengths");
      auto target_lengths = readIntArrayRef("target_lengths");
      int64_t blank = readAttribute<int64_t>("blank");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto log_probs = peek(0, 2);
          auto targets = peek(1, 2);
          auto the_result = at::_use_cudnn_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank);
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_27() { // _cudnn_ctc_loss
      auto input_lengths = readIntArrayRef("input_lengths");
      auto target_lengths = readIntArrayRef("target_lengths");
      int64_t blank = readAttribute<int64_t>("blank");
      bool deterministic = readAttribute<int64_t>("deterministic");
      bool zero_infinity = readAttribute<int64_t>("zero_infinity");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto log_probs = peek(0, 2);
          auto targets = peek(1, 2);
          auto the_result = at::_cudnn_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, deterministic, zero_infinity);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_28() { // _use_cudnn_rnn_flatten_weight
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
  
          auto the_result = at::_use_cudnn_rnn_flatten_weight();
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_29() { // _cudnn_rnn_flatten_weight
      int64_t weight_stride0 = readAttribute<int64_t>("weight_stride0");
      int64_t input_size = readAttribute<int64_t>("input_size");
      int64_t mode = readAttribute<int64_t>("mode");
      int64_t hidden_size = readAttribute<int64_t>("hidden_size");
      int64_t proj_size = readAttribute<int64_t>("proj_size");
      int64_t num_layers = readAttribute<int64_t>("num_layers");
      bool batch_first = readAttribute<int64_t>("batch_first");
      bool bidirectional = readAttribute<int64_t>("bidirectional");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto weight_arr = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_cudnn_rnn_flatten_weight(weight_arr, weight_stride0, input_size, mode, hidden_size, proj_size, num_layers, batch_first, bidirectional);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_30() { // _cudnn_rnn
      int64_t weight_stride0 = readAttribute<int64_t>("weight_stride0");
      int64_t mode = readAttribute<int64_t>("mode");
      int64_t hidden_size = readAttribute<int64_t>("hidden_size");
      int64_t proj_size = readAttribute<int64_t>("proj_size");
      int64_t num_layers = readAttribute<int64_t>("num_layers");
      bool batch_first = readAttribute<int64_t>("batch_first");
      double dropout = readAttribute<float>("dropout");
      bool train = readAttribute<int64_t>("train");
      bool bidirectional = readAttribute<int64_t>("bidirectional");
      auto batch_sizes = readIntArrayRef("batch_sizes");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, InputSize());
          auto weight = peekSlice(1, InputSize() - 5, InputSize());
          auto weight_buf = peek(1, 5);
          auto hx = peek(2, 5);
          auto cx = peek(3, 5);
          auto dropout_state = peek(4, 5);
          auto the_result = at::_cudnn_rnn(input, weight, weight_stride0, weight_buf, hx, cx, mode, hidden_size, proj_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
            if(OutputSize() > 3) {assignTo(Output(3),::std::get<3>(the_result));}
            if(OutputSize() > 4) {assignTo(Output(4),::std::get<4>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_31() { // _debug_has_internal_overlap
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_debug_has_internal_overlap(self);
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_32() { // _fused_dropout
      double p = readAttribute<float>("p");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_fused_dropout(self, p);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_33() { // _masked_scale
      double scale = readAttribute<float>("scale");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto mask = peek(1, 2);
          auto the_result = at::_masked_scale(self, mask, scale);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_34() { // _reshape_from_tensor
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto shape = peek(1, 2);
          auto the_result = at::_reshape_from_tensor(self, shape);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_35() { // _shape_as_tensor
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_shape_as_tensor(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_36() { // dropout
      double p = readAttribute<float>("p");
      bool train = readAttribute<int64_t>("train");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 1);
          auto the_result = at::dropout(input, p, train);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_37() { // feature_dropout
      double p = readAttribute<float>("p");
      bool train = readAttribute<int64_t>("train");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 1);
          auto the_result = at::feature_dropout(input, p, train);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_38() { // alpha_dropout
      double p = readAttribute<float>("p");
      bool train = readAttribute<int64_t>("train");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 1);
          auto the_result = at::alpha_dropout(input, p, train);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_39() { // feature_alpha_dropout
      double p = readAttribute<float>("p");
      bool train = readAttribute<int64_t>("train");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 1);
          auto the_result = at::feature_alpha_dropout(input, p, train);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_40() { // abs
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::abs(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_41() { // absolute
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::absolute(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_42() { // angle
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::angle(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_43() { // view_as_real
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::view_as_real(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_44() { // view_as_complex
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::view_as_complex(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_45() { // sgn
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::sgn(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_46() { // real
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::real(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_47() { // imag
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::imag(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_48() { // _conj
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_conj(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_49() { // conj
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::conj(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_50() { // _conj_physical
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_conj_physical(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_51() { // conj_physical
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::conj_physical(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_52() { // resolve_conj
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::resolve_conj(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_53() { // resolve_neg
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::resolve_neg(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_54() { // _neg_view
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_neg_view(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_55() { // acos
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::acos(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_56() { // arccos
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::arccos(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_57() { // avg_pool1d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      bool ceil_mode = readAttribute<int64_t>("ceil_mode");
      bool count_include_pad = readAttribute<int64_t>("count_include_pad");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::avg_pool1d(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_58() { // avg_pool1d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      bool ceil_mode = readAttribute<int64_t>("ceil_mode");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::avg_pool1d(self, kernel_size, stride, padding, ceil_mode);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_59() { // avg_pool1d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::avg_pool1d(self, kernel_size, stride, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_60() { // avg_pool1d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::avg_pool1d(self, kernel_size, stride);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_61() { // avg_pool1d
      auto kernel_size = readIntArrayRef("kernel_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::avg_pool1d(self, kernel_size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_62() { // adaptive_avg_pool1d
      auto output_size = readIntArrayRef("output_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::adaptive_avg_pool1d(self, output_size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_63() { // adaptive_max_pool1d
      auto output_size = readIntArrayRef("output_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::adaptive_max_pool1d(self, output_size);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_64() { // add
      at::Scalar alpha = readScalarAttribute("alpha");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::add(self, other, alpha);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_65() { // add
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::add(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_66() { // _add_relu
      at::Scalar alpha = readScalarAttribute("alpha");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::_add_relu(self, other, alpha);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_67() { // _add_relu
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::_add_relu(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_68() { // _add_relu
      at::Scalar other = readScalarAttribute("other");
      at::Scalar alpha = readScalarAttribute("alpha");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_add_relu(self, other, alpha);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_69() { // _add_relu
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_add_relu(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_70() { // add
      at::Scalar other = readScalarAttribute("other");
      at::Scalar alpha = readScalarAttribute("alpha");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::add(self, other, alpha);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_71() { // add
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::add(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_72() { // addmv
      at::Scalar beta = readScalarAttribute("beta");
      at::Scalar alpha = readScalarAttribute("alpha");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto mat = peek(1, 3);
          auto vec = peek(2, 3);
          auto the_result = at::addmv(self, mat, vec, beta, alpha);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_73() { // addmv
      at::Scalar beta = readScalarAttribute("beta");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto mat = peek(1, 3);
          auto vec = peek(2, 3);
          auto the_result = at::addmv(self, mat, vec, beta);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_74() { // addmv
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto mat = peek(1, 3);
          auto vec = peek(2, 3);
          auto the_result = at::addmv(self, mat, vec);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_75() { // addr
      at::Scalar beta = readScalarAttribute("beta");
      at::Scalar alpha = readScalarAttribute("alpha");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto vec1 = peek(1, 3);
          auto vec2 = peek(2, 3);
          auto the_result = at::addr(self, vec1, vec2, beta, alpha);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_76() { // addr
      at::Scalar beta = readScalarAttribute("beta");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto vec1 = peek(1, 3);
          auto vec2 = peek(2, 3);
          auto the_result = at::addr(self, vec1, vec2, beta);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_77() { // addr
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto vec1 = peek(1, 3);
          auto vec2 = peek(2, 3);
          auto the_result = at::addr(self, vec1, vec2);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_78() { // affine_grid_generator
      auto size = readIntArrayRef("size");
      bool align_corners = readAttribute<int64_t>("align_corners");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto theta = peek(0, 1);
          auto the_result = at::affine_grid_generator(theta, size, align_corners);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_79() { // affine_grid_generator_backward
      auto size = readIntArrayRef("size");
      bool align_corners = readAttribute<int64_t>("align_corners");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad = peek(0, 1);
          auto the_result = at::affine_grid_generator_backward(grad, size, align_corners);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_80() { // all
      int64_t dim = readAttribute<int64_t>("dim");
      bool keepdim = readAttribute<int64_t>("keepdim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::all(self, dim, keepdim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_81() { // all
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::all(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_82() { // allclose
      double rtol = readAttribute<float>("rtol");
      double atol = readAttribute<float>("atol");
      bool equal_nan = readAttribute<int64_t>("equal_nan");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::allclose(self, other, rtol, atol, equal_nan);
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_83() { // allclose
      double rtol = readAttribute<float>("rtol");
      double atol = readAttribute<float>("atol");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::allclose(self, other, rtol, atol);
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_84() { // allclose
      double rtol = readAttribute<float>("rtol");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::allclose(self, other, rtol);
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_85() { // allclose
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::allclose(self, other);
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_86() { // any
      int64_t dim = readAttribute<int64_t>("dim");
      bool keepdim = readAttribute<int64_t>("keepdim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::any(self, dim, keepdim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_87() { // any
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::any(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_88() { // _dim_arange
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto like = peek(0, 1);
          auto the_result = at::_dim_arange(like, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_89() { // argmax
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::argmax(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_90() { // argmin
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::argmin(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_91() { // acosh
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::acosh(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_92() { // arccosh
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::arccosh(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_93() { // asinh
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::asinh(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_94() { // arcsinh
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::arcsinh(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_95() { // atanh
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::atanh(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_96() { // arctanh
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::arctanh(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_97() { // as_strided
      auto size = readIntArrayRef("size");
      auto stride = readIntArrayRef("stride");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::as_strided(self, size, stride);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_98() { // asin
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::asin(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_99() { // arcsin
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::arcsin(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_100() { // atan
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::atan(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_101() { // arctan
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::arctan(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_102() { // atleast_1d
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::atleast_1d(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_103() { // atleast_1d
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::atleast_1d(tensors);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_104() { // atleast_2d
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::atleast_2d(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_105() { // atleast_2d
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::atleast_2d(tensors);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_106() { // atleast_3d
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::atleast_3d(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_107() { // atleast_3d
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::atleast_3d(tensors);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_108() { // baddbmm
      at::Scalar beta = readScalarAttribute("beta");
      at::Scalar alpha = readScalarAttribute("alpha");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto batch1 = peek(1, 3);
          auto batch2 = peek(2, 3);
          auto the_result = at::baddbmm(self, batch1, batch2, beta, alpha);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_109() { // baddbmm
      at::Scalar beta = readScalarAttribute("beta");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto batch1 = peek(1, 3);
          auto batch2 = peek(2, 3);
          auto the_result = at::baddbmm(self, batch1, batch2, beta);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_110() { // baddbmm
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto batch1 = peek(1, 3);
          auto batch2 = peek(2, 3);
          auto the_result = at::baddbmm(self, batch1, batch2);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_111() { // batch_norm
      bool training = readAttribute<int64_t>("training");
      double momentum = readAttribute<float>("momentum");
      double eps = readAttribute<float>("eps");
      bool cudnn_enabled = readAttribute<int64_t>("cudnn_enabled");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 5);
          auto weight = peek(1, 5);
          auto bias = peek(2, 5);
          auto running_mean = peek(3, 5);
          auto running_var = peek(4, 5);
          auto the_result = at::batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_112() { // quantized_batch_norm
      double eps = readAttribute<float>("eps");
      double output_scale = readAttribute<float>("output_scale");
      int64_t output_zero_point = readAttribute<int64_t>("output_zero_point");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 5);
          auto weight = peek(1, 5);
          auto bias = peek(2, 5);
          auto mean = peek(3, 5);
          auto var = peek(4, 5);
          auto the_result = at::quantized_batch_norm(input, weight, bias, mean, var, eps, output_scale, output_zero_point);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_113() { // _batch_norm_impl_index
      bool training = readAttribute<int64_t>("training");
      double momentum = readAttribute<float>("momentum");
      double eps = readAttribute<float>("eps");
      bool cudnn_enabled = readAttribute<int64_t>("cudnn_enabled");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 5);
          auto weight = peek(1, 5);
          auto bias = peek(2, 5);
          auto running_mean = peek(3, 5);
          auto running_var = peek(4, 5);
          auto the_result = at::_batch_norm_impl_index(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
            if(OutputSize() > 3) {assignTo(Output(3),::std::get<3>(the_result));}
            if(OutputSize() > 4) {assignToValue<int64_t>(Output(4),::std::get<4>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_114() { // _batch_norm_impl_index_backward
      int64_t impl_index = readAttribute<int64_t>("impl_index");
      bool train = readAttribute<int64_t>("train");
      double eps = readAttribute<float>("eps");
      auto output_mask = readBoolMask<3>("output_mask");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 8);
          auto grad_output = peek(1, 8);
          auto weight = peek(2, 8);
          auto running_mean = peek(3, 8);
          auto running_var = peek(4, 8);
          auto save_mean = peek(5, 8);
          auto save_var_transform = peek(6, 8);
          auto reservedSpace = peek(7, 8);
          auto the_result = at::_batch_norm_impl_index_backward(impl_index, input, grad_output, weight, running_mean, running_var, save_mean, save_var_transform, train, eps, output_mask, reservedSpace);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_115() { // bernoulli
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::bernoulli(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_116() { // bernoulli
      double p = readAttribute<float>("p");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::bernoulli(self, p);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_117() { // bilinear
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input1 = peek(0, 4);
          auto input2 = peek(1, 4);
          auto weight = peek(2, 4);
          auto bias = peek(3, 4);
          auto the_result = at::bilinear(input1, input2, weight, bias);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_118() { // binary_cross_entropy
      int64_t reduction = readAttribute<int64_t>("reduction");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto target = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::binary_cross_entropy(self, target, weight, reduction);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_119() { // binary_cross_entropy
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto target = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::binary_cross_entropy(self, target, weight);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_120() { // binary_cross_entropy
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto target = peek(1, 2);
          auto the_result = at::binary_cross_entropy(self, target);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_121() { // binary_cross_entropy_backward
      int64_t reduction = readAttribute<int64_t>("reduction");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 4);
          auto self = peek(1, 4);
          auto target = peek(2, 4);
          auto weight = peek(3, 4);
          auto the_result = at::binary_cross_entropy_backward(grad_output, self, target, weight, reduction);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_122() { // binary_cross_entropy_backward
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 4);
          auto self = peek(1, 4);
          auto target = peek(2, 4);
          auto weight = peek(3, 4);
          auto the_result = at::binary_cross_entropy_backward(grad_output, self, target, weight);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_123() { // binary_cross_entropy_backward
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto self = peek(1, 3);
          auto target = peek(2, 3);
          auto the_result = at::binary_cross_entropy_backward(grad_output, self, target);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_124() { // binary_cross_entropy_with_logits
      int64_t reduction = readAttribute<int64_t>("reduction");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 4);
          auto target = peek(1, 4);
          auto weight = peek(2, 4);
          auto pos_weight = peek(3, 4);
          auto the_result = at::binary_cross_entropy_with_logits(self, target, weight, pos_weight, reduction);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_125() { // binary_cross_entropy_with_logits
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 4);
          auto target = peek(1, 4);
          auto weight = peek(2, 4);
          auto pos_weight = peek(3, 4);
          auto the_result = at::binary_cross_entropy_with_logits(self, target, weight, pos_weight);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_126() { // binary_cross_entropy_with_logits
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto target = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::binary_cross_entropy_with_logits(self, target, weight);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_127() { // binary_cross_entropy_with_logits
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto target = peek(1, 2);
          auto the_result = at::binary_cross_entropy_with_logits(self, target);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_128() { // binary_cross_entropy_with_logits_backward
      int64_t reduction = readAttribute<int64_t>("reduction");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 5);
          auto self = peek(1, 5);
          auto target = peek(2, 5);
          auto weight = peek(3, 5);
          auto pos_weight = peek(4, 5);
          auto the_result = at::binary_cross_entropy_with_logits_backward(grad_output, self, target, weight, pos_weight, reduction);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_129() { // binary_cross_entropy_with_logits_backward
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 5);
          auto self = peek(1, 5);
          auto target = peek(2, 5);
          auto weight = peek(3, 5);
          auto pos_weight = peek(4, 5);
          auto the_result = at::binary_cross_entropy_with_logits_backward(grad_output, self, target, weight, pos_weight);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_130() { // binary_cross_entropy_with_logits_backward
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 4);
          auto self = peek(1, 4);
          auto target = peek(2, 4);
          auto weight = peek(3, 4);
          auto the_result = at::binary_cross_entropy_with_logits_backward(grad_output, self, target, weight);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_131() { // binary_cross_entropy_with_logits_backward
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto self = peek(1, 3);
          auto target = peek(2, 3);
          auto the_result = at::binary_cross_entropy_with_logits_backward(grad_output, self, target);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_132() { // bincount
      int64_t minlength = readAttribute<int64_t>("minlength");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto weights = peek(1, 2);
          auto the_result = at::bincount(self, weights, minlength);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_133() { // bincount
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto weights = peek(1, 2);
          auto the_result = at::bincount(self, weights);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_134() { // bincount
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::bincount(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_135() { // bitwise_not
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::bitwise_not(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_136() { // copysign
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::copysign(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_137() { // copysign
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::copysign(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_138() { // logical_not
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::logical_not(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_139() { // logical_xor
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::logical_xor(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_140() { // logical_and
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::logical_and(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_141() { // logical_or
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::logical_or(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_142() { // bmm
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto mat2 = peek(1, 2);
          auto the_result = at::bmm(self, mat2);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_143() { // broadcast_tensors
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::broadcast_tensors(tensors);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_144() { // broadcast_to
      auto size = readIntArrayRef("size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::broadcast_to(self, size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_145() { // cat
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::cat(tensors, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_146() { // cat
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::cat(tensors);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_147() { // concat
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::concat(tensors, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_148() { // concat
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::concat(tensors);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_149() { // block_diag
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::block_diag(tensors);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_150() { // ceil
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::ceil(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_151() { // chain_matmul
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto matrices = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::chain_matmul(matrices);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_152() { // unsafe_chunk
      int64_t chunks = readAttribute<int64_t>("chunks");
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::unsafe_chunk(self, chunks, dim);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_153() { // unsafe_chunk
      int64_t chunks = readAttribute<int64_t>("chunks");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::unsafe_chunk(self, chunks);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_154() { // chunk
      int64_t chunks = readAttribute<int64_t>("chunks");
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::chunk(self, chunks, dim);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_155() { // chunk
      int64_t chunks = readAttribute<int64_t>("chunks");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::chunk(self, chunks);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_156() { // tensor_split
      int64_t sections = readAttribute<int64_t>("sections");
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::tensor_split(self, sections, dim);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_157() { // tensor_split
      int64_t sections = readAttribute<int64_t>("sections");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::tensor_split(self, sections);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_158() { // tensor_split
      auto indices = readIntArrayRef("indices");
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::tensor_split(self, indices, dim);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_159() { // tensor_split
      auto indices = readIntArrayRef("indices");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::tensor_split(self, indices);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_160() { // tensor_split
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto tensor_indices_or_sections = peek(1, 2);
          auto the_result = at::tensor_split(self, tensor_indices_or_sections, dim);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_161() { // tensor_split
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto tensor_indices_or_sections = peek(1, 2);
          auto the_result = at::tensor_split(self, tensor_indices_or_sections);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_162() { // clamp
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::clamp(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_163() { // clamp
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto min = peek(1, 3);
          auto max = peek(2, 3);
          auto the_result = at::clamp(self, min, max);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_164() { // clamp
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto min = peek(1, 2);
          auto the_result = at::clamp(self, min);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_165() { // clamp_max
      at::Scalar max = readScalarAttribute("max");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::clamp_max(self, max);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_166() { // clamp_max
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto max = peek(1, 2);
          auto the_result = at::clamp_max(self, max);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_167() { // clamp_min
      at::Scalar min = readScalarAttribute("min");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::clamp_min(self, min);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_168() { // clamp_min
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto min = peek(1, 2);
          auto the_result = at::clamp_min(self, min);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_169() { // clip
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::clip(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_170() { // clip
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto min = peek(1, 3);
          auto max = peek(2, 3);
          auto the_result = at::clip(self, min, max);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_171() { // clip
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto min = peek(1, 2);
          auto the_result = at::clip(self, min);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_172() { // cudnn_is_acceptable
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::cudnn_is_acceptable(self);
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_173() { // complex
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto real = peek(0, 2);
          auto imag = peek(1, 2);
          auto the_result = at::complex(real, imag);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_174() { // polar
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto abs = peek(0, 2);
          auto angle = peek(1, 2);
          auto the_result = at::polar(abs, angle);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_175() { // constant_pad_nd
      auto pad = readIntArrayRef("pad");
      at::Scalar value = readScalarAttribute("value");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::constant_pad_nd(self, pad, value);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_176() { // constant_pad_nd
      auto pad = readIntArrayRef("pad");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::constant_pad_nd(self, pad);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_177() { // contiguous
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = self.contiguous();
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_178() { // convolution
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      bool transposed = readAttribute<int64_t>("transposed");
      auto output_padding = readIntArrayRef("output_padding");
      int64_t groups = readAttribute<int64_t>("groups");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_179() { // convolution_overrideable
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      bool transposed = readAttribute<int64_t>("transposed");
      auto output_padding = readIntArrayRef("output_padding");
      int64_t groups = readAttribute<int64_t>("groups");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::convolution_overrideable(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_180() { // convolution_backward_overrideable
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      bool transposed = readAttribute<int64_t>("transposed");
      auto output_padding = readIntArrayRef("output_padding");
      int64_t groups = readAttribute<int64_t>("groups");
      auto output_mask = readBoolMask<3>("output_mask");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto input = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::convolution_backward_overrideable(grad_output, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_181() { // _convolution
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      bool transposed = readAttribute<int64_t>("transposed");
      auto output_padding = readIntArrayRef("output_padding");
      int64_t groups = readAttribute<int64_t>("groups");
      bool benchmark = readAttribute<int64_t>("benchmark");
      bool deterministic = readAttribute<int64_t>("deterministic");
      bool cudnn_enabled = readAttribute<int64_t>("cudnn_enabled");
      bool allow_tf32 = readAttribute<int64_t>("allow_tf32");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_182() { // _convolution
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      bool transposed = readAttribute<int64_t>("transposed");
      auto output_padding = readIntArrayRef("output_padding");
      int64_t groups = readAttribute<int64_t>("groups");
      bool benchmark = readAttribute<int64_t>("benchmark");
      bool deterministic = readAttribute<int64_t>("deterministic");
      bool cudnn_enabled = readAttribute<int64_t>("cudnn_enabled");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_183() { // _convolution_nogroup
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      bool transposed = readAttribute<int64_t>("transposed");
      auto output_padding = readIntArrayRef("output_padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::_convolution_nogroup(input, weight, bias, stride, padding, dilation, transposed, output_padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_184() { // _convolution_double_backward
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      bool transposed = readAttribute<int64_t>("transposed");
      auto output_padding = readIntArrayRef("output_padding");
      int64_t groups = readAttribute<int64_t>("groups");
      bool benchmark = readAttribute<int64_t>("benchmark");
      bool deterministic = readAttribute<int64_t>("deterministic");
      bool cudnn_enabled = readAttribute<int64_t>("cudnn_enabled");
      bool allow_tf32 = readAttribute<int64_t>("allow_tf32");
      auto output_mask = readBoolMask<3>("output_mask");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto ggI = peek(0, 6);
          auto ggW = peek(1, 6);
          auto ggb = peek(2, 6);
          auto gO = peek(3, 6);
          auto weight = peek(4, 6);
          auto self = peek(5, 6);
          auto the_result = at::_convolution_double_backward(ggI, ggW, ggb, gO, weight, self, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32, output_mask);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_185() { // conv1d
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      int64_t groups = readAttribute<int64_t>("groups");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::conv1d(input, weight, bias, stride, padding, dilation, groups);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_186() { // conv1d
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::conv1d(input, weight, bias, stride, padding, dilation);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_187() { // conv1d
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::conv1d(input, weight, bias, stride, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_188() { // conv1d
      auto stride = readIntArrayRef("stride");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::conv1d(input, weight, bias, stride);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_189() { // conv1d
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::conv1d(input, weight, bias);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_190() { // conv1d
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 2);
          auto weight = peek(1, 2);
          auto the_result = at::conv1d(input, weight);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_191() { // conv2d
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      int64_t groups = readAttribute<int64_t>("groups");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::conv2d(input, weight, bias, stride, padding, dilation, groups);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_192() { // conv2d
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::conv2d(input, weight, bias, stride, padding, dilation);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_193() { // conv2d
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::conv2d(input, weight, bias, stride, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_194() { // conv2d
      auto stride = readIntArrayRef("stride");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::conv2d(input, weight, bias, stride);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_195() { // conv2d
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::conv2d(input, weight, bias);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_196() { // conv2d
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 2);
          auto weight = peek(1, 2);
          auto the_result = at::conv2d(input, weight);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_197() { // conv3d
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      int64_t groups = readAttribute<int64_t>("groups");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::conv3d(input, weight, bias, stride, padding, dilation, groups);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_198() { // conv3d
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::conv3d(input, weight, bias, stride, padding, dilation);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_199() { // conv3d
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::conv3d(input, weight, bias, stride, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_200() { // conv3d
      auto stride = readIntArrayRef("stride");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::conv3d(input, weight, bias, stride);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_201() { // conv3d
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::conv3d(input, weight, bias);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_202() { // conv3d
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 2);
          auto weight = peek(1, 2);
          auto the_result = at::conv3d(input, weight);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_203() { // conv_tbc
      int64_t pad = readAttribute<int64_t>("pad");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::conv_tbc(self, weight, bias, pad);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_204() { // conv_tbc
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::conv_tbc(self, weight, bias);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_205() { // conv_tbc_backward
      int64_t pad = readAttribute<int64_t>("pad");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 4);
          auto input = peek(1, 4);
          auto weight = peek(2, 4);
          auto bias = peek(3, 4);
          auto the_result = at::conv_tbc_backward(self, input, weight, bias, pad);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_206() { // conv_transpose1d
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto output_padding = readIntArrayRef("output_padding");
      int64_t groups = readAttribute<int64_t>("groups");
      auto dilation = readIntArrayRef("dilation");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::conv_transpose1d(input, weight, bias, stride, padding, output_padding, groups, dilation);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_207() { // conv_transpose1d
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto output_padding = readIntArrayRef("output_padding");
      int64_t groups = readAttribute<int64_t>("groups");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::conv_transpose1d(input, weight, bias, stride, padding, output_padding, groups);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_208() { // conv_transpose1d
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto output_padding = readIntArrayRef("output_padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::conv_transpose1d(input, weight, bias, stride, padding, output_padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_209() { // conv_transpose1d
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::conv_transpose1d(input, weight, bias, stride, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_210() { // conv_transpose1d
      auto stride = readIntArrayRef("stride");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::conv_transpose1d(input, weight, bias, stride);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_211() { // conv_transpose1d
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::conv_transpose1d(input, weight, bias);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_212() { // conv_transpose1d
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 2);
          auto weight = peek(1, 2);
          auto the_result = at::conv_transpose1d(input, weight);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_213() { // conv_transpose2d
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto output_padding = readIntArrayRef("output_padding");
      int64_t groups = readAttribute<int64_t>("groups");
      auto dilation = readIntArrayRef("dilation");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups, dilation);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_214() { // conv_transpose2d
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto output_padding = readIntArrayRef("output_padding");
      int64_t groups = readAttribute<int64_t>("groups");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_215() { // conv_transpose2d
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto output_padding = readIntArrayRef("output_padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::conv_transpose2d(input, weight, bias, stride, padding, output_padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_216() { // conv_transpose2d
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::conv_transpose2d(input, weight, bias, stride, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_217() { // conv_transpose2d
      auto stride = readIntArrayRef("stride");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::conv_transpose2d(input, weight, bias, stride);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_218() { // conv_transpose2d
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::conv_transpose2d(input, weight, bias);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_219() { // conv_transpose2d
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 2);
          auto weight = peek(1, 2);
          auto the_result = at::conv_transpose2d(input, weight);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_220() { // conv_transpose3d
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto output_padding = readIntArrayRef("output_padding");
      int64_t groups = readAttribute<int64_t>("groups");
      auto dilation = readIntArrayRef("dilation");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::conv_transpose3d(input, weight, bias, stride, padding, output_padding, groups, dilation);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_221() { // conv_transpose3d
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto output_padding = readIntArrayRef("output_padding");
      int64_t groups = readAttribute<int64_t>("groups");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::conv_transpose3d(input, weight, bias, stride, padding, output_padding, groups);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_222() { // conv_transpose3d
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto output_padding = readIntArrayRef("output_padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::conv_transpose3d(input, weight, bias, stride, padding, output_padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_223() { // conv_transpose3d
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::conv_transpose3d(input, weight, bias, stride, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_224() { // conv_transpose3d
      auto stride = readIntArrayRef("stride");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::conv_transpose3d(input, weight, bias, stride);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_225() { // conv_transpose3d
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::conv_transpose3d(input, weight, bias);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_226() { // conv_transpose3d
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 2);
          auto weight = peek(1, 2);
          auto the_result = at::conv_transpose3d(input, weight);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_227() { // _copy_from
      bool non_blocking = readAttribute<int64_t>("non_blocking");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto dst = peek(1, 2);
          auto the_result = at::_copy_from(self, dst, non_blocking);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_228() { // _copy_from
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto dst = peek(1, 2);
          auto the_result = at::_copy_from(self, dst);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_229() { // _copy_from_and_resize
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto dst = peek(1, 2);
          auto the_result = at::_copy_from_and_resize(self, dst);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_230() { // cos
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::cos(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_231() { // cosh
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::cosh(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_232() { // cosine_embedding_loss
      double margin = readAttribute<float>("margin");
      int64_t reduction = readAttribute<int64_t>("reduction");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input1 = peek(0, 3);
          auto input2 = peek(1, 3);
          auto target = peek(2, 3);
          auto the_result = at::cosine_embedding_loss(input1, input2, target, margin, reduction);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_233() { // cosine_embedding_loss
      double margin = readAttribute<float>("margin");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input1 = peek(0, 3);
          auto input2 = peek(1, 3);
          auto target = peek(2, 3);
          auto the_result = at::cosine_embedding_loss(input1, input2, target, margin);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_234() { // cosine_embedding_loss
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input1 = peek(0, 3);
          auto input2 = peek(1, 3);
          auto target = peek(2, 3);
          auto the_result = at::cosine_embedding_loss(input1, input2, target);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_235() { // count_nonzero
      auto dim = readIntArrayRef("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::count_nonzero(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_236() { // count_nonzero
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::count_nonzero(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_237() { // cov
      int64_t correction = readAttribute<int64_t>("correction");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto fweights = peek(1, 3);
          auto aweights = peek(2, 3);
          auto the_result = at::cov(self, correction, fweights, aweights);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_238() { // cov
      int64_t correction = readAttribute<int64_t>("correction");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto fweights = peek(1, 2);
          auto the_result = at::cov(self, correction, fweights);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_239() { // cov
      int64_t correction = readAttribute<int64_t>("correction");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::cov(self, correction);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_240() { // cov
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::cov(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_241() { // corrcoef
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::corrcoef(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_242() { // cudnn_affine_grid_generator
      int64_t N = readAttribute<int64_t>("N");
      int64_t C = readAttribute<int64_t>("C");
      int64_t H = readAttribute<int64_t>("H");
      int64_t W = readAttribute<int64_t>("W");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto theta = peek(0, 1);
          auto the_result = at::cudnn_affine_grid_generator(theta, N, C, H, W);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_243() { // cudnn_affine_grid_generator_backward
      int64_t N = readAttribute<int64_t>("N");
      int64_t C = readAttribute<int64_t>("C");
      int64_t H = readAttribute<int64_t>("H");
      int64_t W = readAttribute<int64_t>("W");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad = peek(0, 1);
          auto the_result = at::cudnn_affine_grid_generator_backward(grad, N, C, H, W);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_244() { // cudnn_batch_norm
      bool training = readAttribute<int64_t>("training");
      double exponential_average_factor = readAttribute<float>("exponential_average_factor");
      double epsilon = readAttribute<float>("epsilon");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 5);
          auto weight = peek(1, 5);
          auto bias = peek(2, 5);
          auto running_mean = peek(3, 5);
          auto running_var = peek(4, 5);
          auto the_result = at::cudnn_batch_norm(input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
            if(OutputSize() > 3) {assignTo(Output(3),::std::get<3>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_245() { // cudnn_batch_norm_backward
      double epsilon = readAttribute<float>("epsilon");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 8);
          auto grad_output = peek(1, 8);
          auto weight = peek(2, 8);
          auto running_mean = peek(3, 8);
          auto running_var = peek(4, 8);
          auto save_mean = peek(5, 8);
          auto save_var = peek(6, 8);
          auto reserveSpace = peek(7, 8);
          auto the_result = at::cudnn_batch_norm_backward(input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon, reserveSpace);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_246() { // cudnn_convolution
      auto padding = readIntArrayRef("padding");
      auto stride = readIntArrayRef("stride");
      auto dilation = readIntArrayRef("dilation");
      int64_t groups = readAttribute<int64_t>("groups");
      bool benchmark = readAttribute<int64_t>("benchmark");
      bool deterministic = readAttribute<int64_t>("deterministic");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::cudnn_convolution(self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_247() { // cudnn_convolution
      auto padding = readIntArrayRef("padding");
      auto stride = readIntArrayRef("stride");
      auto dilation = readIntArrayRef("dilation");
      int64_t groups = readAttribute<int64_t>("groups");
      bool benchmark = readAttribute<int64_t>("benchmark");
      bool deterministic = readAttribute<int64_t>("deterministic");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto weight = peek(1, 2);
          auto the_result = at::cudnn_convolution(self, weight, padding, stride, dilation, groups, benchmark, deterministic);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_248() { // cudnn_convolution
      auto padding = readIntArrayRef("padding");
      auto stride = readIntArrayRef("stride");
      auto dilation = readIntArrayRef("dilation");
      int64_t groups = readAttribute<int64_t>("groups");
      bool benchmark = readAttribute<int64_t>("benchmark");
      bool deterministic = readAttribute<int64_t>("deterministic");
      bool allow_tf32 = readAttribute<int64_t>("allow_tf32");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto weight = peek(1, 2);
          auto the_result = at::cudnn_convolution(self, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_249() { // cudnn_convolution_backward_input
      auto self_size = readIntArrayRef("self_size");
      auto padding = readIntArrayRef("padding");
      auto stride = readIntArrayRef("stride");
      auto dilation = readIntArrayRef("dilation");
      int64_t groups = readAttribute<int64_t>("groups");
      bool benchmark = readAttribute<int64_t>("benchmark");
      bool deterministic = readAttribute<int64_t>("deterministic");
      bool allow_tf32 = readAttribute<int64_t>("allow_tf32");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 2);
          auto weight = peek(1, 2);
          auto the_result = at::cudnn_convolution_backward_input(self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_250() { // cudnn_convolution_backward
      auto padding = readIntArrayRef("padding");
      auto stride = readIntArrayRef("stride");
      auto dilation = readIntArrayRef("dilation");
      int64_t groups = readAttribute<int64_t>("groups");
      bool benchmark = readAttribute<int64_t>("benchmark");
      bool deterministic = readAttribute<int64_t>("deterministic");
      bool allow_tf32 = readAttribute<int64_t>("allow_tf32");
      auto output_mask = readBoolMask<2>("output_mask");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto grad_output = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::cudnn_convolution_backward(self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32, output_mask);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_251() { // cudnn_convolution_backward_weight
      auto weight_size = readIntArrayRef("weight_size");
      auto padding = readIntArrayRef("padding");
      auto stride = readIntArrayRef("stride");
      auto dilation = readIntArrayRef("dilation");
      int64_t groups = readAttribute<int64_t>("groups");
      bool benchmark = readAttribute<int64_t>("benchmark");
      bool deterministic = readAttribute<int64_t>("deterministic");
      bool allow_tf32 = readAttribute<int64_t>("allow_tf32");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 2);
          auto self = peek(1, 2);
          auto the_result = at::cudnn_convolution_backward_weight(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_252() { // cudnn_convolution_transpose
      auto padding = readIntArrayRef("padding");
      auto output_padding = readIntArrayRef("output_padding");
      auto stride = readIntArrayRef("stride");
      auto dilation = readIntArrayRef("dilation");
      int64_t groups = readAttribute<int64_t>("groups");
      bool benchmark = readAttribute<int64_t>("benchmark");
      bool deterministic = readAttribute<int64_t>("deterministic");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::cudnn_convolution_transpose(self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_253() { // cudnn_convolution_transpose
      auto padding = readIntArrayRef("padding");
      auto output_padding = readIntArrayRef("output_padding");
      auto stride = readIntArrayRef("stride");
      auto dilation = readIntArrayRef("dilation");
      int64_t groups = readAttribute<int64_t>("groups");
      bool benchmark = readAttribute<int64_t>("benchmark");
      bool deterministic = readAttribute<int64_t>("deterministic");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto weight = peek(1, 2);
          auto the_result = at::cudnn_convolution_transpose(self, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_254() { // cudnn_convolution_transpose
      auto padding = readIntArrayRef("padding");
      auto output_padding = readIntArrayRef("output_padding");
      auto stride = readIntArrayRef("stride");
      auto dilation = readIntArrayRef("dilation");
      int64_t groups = readAttribute<int64_t>("groups");
      bool benchmark = readAttribute<int64_t>("benchmark");
      bool deterministic = readAttribute<int64_t>("deterministic");
      bool allow_tf32 = readAttribute<int64_t>("allow_tf32");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto weight = peek(1, 2);
          auto the_result = at::cudnn_convolution_transpose(self, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_255() { // cudnn_convolution_transpose_backward
      auto padding = readIntArrayRef("padding");
      auto output_padding = readIntArrayRef("output_padding");
      auto stride = readIntArrayRef("stride");
      auto dilation = readIntArrayRef("dilation");
      int64_t groups = readAttribute<int64_t>("groups");
      bool benchmark = readAttribute<int64_t>("benchmark");
      bool deterministic = readAttribute<int64_t>("deterministic");
      bool allow_tf32 = readAttribute<int64_t>("allow_tf32");
      auto output_mask = readBoolMask<2>("output_mask");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto grad_output = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::cudnn_convolution_transpose_backward(self, grad_output, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, allow_tf32, output_mask);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_256() { // cudnn_convolution_transpose_backward_input
      auto padding = readIntArrayRef("padding");
      auto stride = readIntArrayRef("stride");
      auto dilation = readIntArrayRef("dilation");
      int64_t groups = readAttribute<int64_t>("groups");
      bool benchmark = readAttribute<int64_t>("benchmark");
      bool deterministic = readAttribute<int64_t>("deterministic");
      bool allow_tf32 = readAttribute<int64_t>("allow_tf32");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 2);
          auto weight = peek(1, 2);
          auto the_result = at::cudnn_convolution_transpose_backward_input(grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_257() { // cudnn_convolution_transpose_backward_weight
      auto weight_size = readIntArrayRef("weight_size");
      auto padding = readIntArrayRef("padding");
      auto stride = readIntArrayRef("stride");
      auto dilation = readIntArrayRef("dilation");
      int64_t groups = readAttribute<int64_t>("groups");
      bool benchmark = readAttribute<int64_t>("benchmark");
      bool deterministic = readAttribute<int64_t>("deterministic");
      bool allow_tf32 = readAttribute<int64_t>("allow_tf32");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 2);
          auto self = peek(1, 2);
          auto the_result = at::cudnn_convolution_transpose_backward_weight(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_258() { // cudnn_convolution_relu
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      int64_t groups = readAttribute<int64_t>("groups");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::cudnn_convolution_relu(self, weight, bias, stride, padding, dilation, groups);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_259() { // cudnn_grid_sampler
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto grid = peek(1, 2);
          auto the_result = at::cudnn_grid_sampler(self, grid);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_260() { // cudnn_grid_sampler_backward
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto grid = peek(1, 3);
          auto grad_output = peek(2, 3);
          auto the_result = at::cudnn_grid_sampler_backward(self, grid, grad_output);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_261() { // cummax
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::cummax(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_262() { // cummin
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::cummin(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_263() { // cummaxmin_backward
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad = peek(0, 3);
          auto input = peek(1, 3);
          auto indices = peek(2, 3);
          auto the_result = at::cummaxmin_backward(grad, input, indices, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_264() { // cumprod
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::cumprod(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_265() { // cumprod_backward
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad = peek(0, 3);
          auto input = peek(1, 3);
          auto output = peek(2, 3);
          auto the_result = at::cumprod_backward(grad, input, dim, output);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_266() { // cumsum
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::cumsum(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_267() { // cumulative_trapezoid
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto y = peek(0, 2);
          auto x = peek(1, 2);
          auto the_result = at::cumulative_trapezoid(y, x, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_268() { // cumulative_trapezoid
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto y = peek(0, 2);
          auto x = peek(1, 2);
          auto the_result = at::cumulative_trapezoid(y, x);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_269() { // cumulative_trapezoid
      at::Scalar dx = readScalarAttribute("dx");
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto y = peek(0, 1);
          auto the_result = at::cumulative_trapezoid(y, dx, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_270() { // cumulative_trapezoid
      at::Scalar dx = readScalarAttribute("dx");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto y = peek(0, 1);
          auto the_result = at::cumulative_trapezoid(y, dx);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_271() { // cumulative_trapezoid
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto y = peek(0, 1);
          auto the_result = at::cumulative_trapezoid(y);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_272() { // ctc_loss
      auto input_lengths = readIntArrayRef("input_lengths");
      auto target_lengths = readIntArrayRef("target_lengths");
      int64_t blank = readAttribute<int64_t>("blank");
      int64_t reduction = readAttribute<int64_t>("reduction");
      bool zero_infinity = readAttribute<int64_t>("zero_infinity");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto log_probs = peek(0, 2);
          auto targets = peek(1, 2);
          auto the_result = at::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_273() { // ctc_loss
      auto input_lengths = readIntArrayRef("input_lengths");
      auto target_lengths = readIntArrayRef("target_lengths");
      int64_t blank = readAttribute<int64_t>("blank");
      int64_t reduction = readAttribute<int64_t>("reduction");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto log_probs = peek(0, 2);
          auto targets = peek(1, 2);
          auto the_result = at::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_274() { // ctc_loss
      auto input_lengths = readIntArrayRef("input_lengths");
      auto target_lengths = readIntArrayRef("target_lengths");
      int64_t blank = readAttribute<int64_t>("blank");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto log_probs = peek(0, 2);
          auto targets = peek(1, 2);
          auto the_result = at::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_275() { // ctc_loss
      auto input_lengths = readIntArrayRef("input_lengths");
      auto target_lengths = readIntArrayRef("target_lengths");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto log_probs = peek(0, 2);
          auto targets = peek(1, 2);
          auto the_result = at::ctc_loss(log_probs, targets, input_lengths, target_lengths);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_276() { // ctc_loss
      int64_t blank = readAttribute<int64_t>("blank");
      int64_t reduction = readAttribute<int64_t>("reduction");
      bool zero_infinity = readAttribute<int64_t>("zero_infinity");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto log_probs = peek(0, 4);
          auto targets = peek(1, 4);
          auto input_lengths = peek(2, 4);
          auto target_lengths = peek(3, 4);
          auto the_result = at::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_277() { // ctc_loss
      int64_t blank = readAttribute<int64_t>("blank");
      int64_t reduction = readAttribute<int64_t>("reduction");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto log_probs = peek(0, 4);
          auto targets = peek(1, 4);
          auto input_lengths = peek(2, 4);
          auto target_lengths = peek(3, 4);
          auto the_result = at::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_278() { // ctc_loss
      int64_t blank = readAttribute<int64_t>("blank");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto log_probs = peek(0, 4);
          auto targets = peek(1, 4);
          auto input_lengths = peek(2, 4);
          auto target_lengths = peek(3, 4);
          auto the_result = at::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_279() { // ctc_loss
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto log_probs = peek(0, 4);
          auto targets = peek(1, 4);
          auto input_lengths = peek(2, 4);
          auto target_lengths = peek(3, 4);
          auto the_result = at::ctc_loss(log_probs, targets, input_lengths, target_lengths);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_280() { // _ctc_loss
      auto input_lengths = readIntArrayRef("input_lengths");
      auto target_lengths = readIntArrayRef("target_lengths");
      int64_t blank = readAttribute<int64_t>("blank");
      bool zero_infinity = readAttribute<int64_t>("zero_infinity");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto log_probs = peek(0, 2);
          auto targets = peek(1, 2);
          auto the_result = at::_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, zero_infinity);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_281() { // _ctc_loss
      auto input_lengths = readIntArrayRef("input_lengths");
      auto target_lengths = readIntArrayRef("target_lengths");
      int64_t blank = readAttribute<int64_t>("blank");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto log_probs = peek(0, 2);
          auto targets = peek(1, 2);
          auto the_result = at::_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_282() { // _ctc_loss
      auto input_lengths = readIntArrayRef("input_lengths");
      auto target_lengths = readIntArrayRef("target_lengths");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto log_probs = peek(0, 2);
          auto targets = peek(1, 2);
          auto the_result = at::_ctc_loss(log_probs, targets, input_lengths, target_lengths);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_283() { // _ctc_loss_backward
      auto input_lengths = readIntArrayRef("input_lengths");
      auto target_lengths = readIntArrayRef("target_lengths");
      int64_t blank = readAttribute<int64_t>("blank");
      bool zero_infinity = readAttribute<int64_t>("zero_infinity");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad = peek(0, 5);
          auto log_probs = peek(1, 5);
          auto targets = peek(2, 5);
          auto neg_log_likelihood = peek(3, 5);
          auto log_alpha = peek(4, 5);
          auto the_result = at::_ctc_loss_backward(grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank, zero_infinity);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_284() { // _ctc_loss_backward
      auto input_lengths = readIntArrayRef("input_lengths");
      auto target_lengths = readIntArrayRef("target_lengths");
      int64_t blank = readAttribute<int64_t>("blank");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad = peek(0, 5);
          auto log_probs = peek(1, 5);
          auto targets = peek(2, 5);
          auto neg_log_likelihood = peek(3, 5);
          auto log_alpha = peek(4, 5);
          auto the_result = at::_ctc_loss_backward(grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_285() { // diag_embed
      int64_t offset = readAttribute<int64_t>("offset");
      int64_t dim1 = readAttribute<int64_t>("dim1");
      int64_t dim2 = readAttribute<int64_t>("dim2");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::diag_embed(self, offset, dim1, dim2);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_286() { // diag_embed
      int64_t offset = readAttribute<int64_t>("offset");
      int64_t dim1 = readAttribute<int64_t>("dim1");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::diag_embed(self, offset, dim1);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_287() { // diag_embed
      int64_t offset = readAttribute<int64_t>("offset");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::diag_embed(self, offset);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_288() { // diag_embed
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::diag_embed(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_289() { // diagflat
      int64_t offset = readAttribute<int64_t>("offset");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::diagflat(self, offset);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_290() { // diagflat
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::diagflat(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_291() { // diagonal
      int64_t offset = readAttribute<int64_t>("offset");
      int64_t dim1 = readAttribute<int64_t>("dim1");
      int64_t dim2 = readAttribute<int64_t>("dim2");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::diagonal(self, offset, dim1, dim2);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_292() { // diagonal
      int64_t offset = readAttribute<int64_t>("offset");
      int64_t dim1 = readAttribute<int64_t>("dim1");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::diagonal(self, offset, dim1);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_293() { // diagonal
      int64_t offset = readAttribute<int64_t>("offset");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::diagonal(self, offset);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_294() { // diagonal
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::diagonal(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_295() { // diagonal_backward
      auto input_sizes = readIntArrayRef("input_sizes");
      int64_t offset = readAttribute<int64_t>("offset");
      int64_t dim1 = readAttribute<int64_t>("dim1");
      int64_t dim2 = readAttribute<int64_t>("dim2");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 1);
          auto the_result = at::diagonal_backward(grad_output, input_sizes, offset, dim1, dim2);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_296() { // diff
      int64_t n = readAttribute<int64_t>("n");
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto prepend = peek(1, 3);
          auto append = peek(2, 3);
          auto the_result = at::diff(self, n, dim, prepend, append);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_297() { // diff
      int64_t n = readAttribute<int64_t>("n");
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto prepend = peek(1, 2);
          auto the_result = at::diff(self, n, dim, prepend);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_298() { // diff
      int64_t n = readAttribute<int64_t>("n");
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::diff(self, n, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_299() { // diff
      int64_t n = readAttribute<int64_t>("n");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::diff(self, n);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_300() { // diff
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::diff(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_301() { // gradient
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::gradient(self);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_302() { // gradient
      at::Scalar spacing = readScalarAttribute("spacing");
      auto dim = readIntArrayRef("dim");
      int64_t edge_order = readAttribute<int64_t>("edge_order");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::gradient(self, spacing, dim, edge_order);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_303() { // gradient
      at::Scalar spacing = readScalarAttribute("spacing");
      auto dim = readIntArrayRef("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::gradient(self, spacing, dim);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_304() { // gradient
      auto dim = readIntArrayRef("dim");
      int64_t edge_order = readAttribute<int64_t>("edge_order");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::gradient(self, dim, edge_order);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_305() { // gradient
      auto dim = readIntArrayRef("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::gradient(self, dim);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_306() { // gradient
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, InputSize());
          auto spacing = peekSlice(1, InputSize() - 1, InputSize());
          auto the_result = at::gradient(self, spacing);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_307() { // gradient
      auto dim = readIntArrayRef("dim");
      int64_t edge_order = readAttribute<int64_t>("edge_order");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, InputSize());
          auto spacing = peekSlice(1, InputSize() - 1, InputSize());
          auto the_result = at::gradient(self, spacing, dim, edge_order);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_308() { // gradient
      auto dim = readIntArrayRef("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, InputSize());
          auto spacing = peekSlice(1, InputSize() - 1, InputSize());
          auto the_result = at::gradient(self, spacing, dim);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_309() { // div
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::div(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_310() { // div
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::div(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_311() { // divide
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::divide(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_312() { // divide
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::divide(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_313() { // true_divide
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::true_divide(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_314() { // true_divide
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::true_divide(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_315() { // dot
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto tensor = peek(1, 2);
          auto the_result = at::dot(self, tensor);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_316() { // vdot
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::vdot(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_317() { // embedding
      int64_t padding_idx = readAttribute<int64_t>("padding_idx");
      bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
      bool sparse = readAttribute<int64_t>("sparse");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto weight = peek(0, 2);
          auto indices = peek(1, 2);
          auto the_result = at::embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_318() { // embedding
      int64_t padding_idx = readAttribute<int64_t>("padding_idx");
      bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto weight = peek(0, 2);
          auto indices = peek(1, 2);
          auto the_result = at::embedding(weight, indices, padding_idx, scale_grad_by_freq);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_319() { // embedding
      int64_t padding_idx = readAttribute<int64_t>("padding_idx");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto weight = peek(0, 2);
          auto indices = peek(1, 2);
          auto the_result = at::embedding(weight, indices, padding_idx);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_320() { // embedding
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto weight = peek(0, 2);
          auto indices = peek(1, 2);
          auto the_result = at::embedding(weight, indices);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_321() { // embedding_backward
      int64_t num_weights = readAttribute<int64_t>("num_weights");
      int64_t padding_idx = readAttribute<int64_t>("padding_idx");
      bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
      bool sparse = readAttribute<int64_t>("sparse");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad = peek(0, 2);
          auto indices = peek(1, 2);
          auto the_result = at::embedding_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_322() { // embedding_dense_backward
      int64_t num_weights = readAttribute<int64_t>("num_weights");
      int64_t padding_idx = readAttribute<int64_t>("padding_idx");
      bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 2);
          auto indices = peek(1, 2);
          auto the_result = at::embedding_dense_backward(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_323() { // embedding_sparse_backward
      int64_t num_weights = readAttribute<int64_t>("num_weights");
      int64_t padding_idx = readAttribute<int64_t>("padding_idx");
      bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad = peek(0, 2);
          auto indices = peek(1, 2);
          auto the_result = at::embedding_sparse_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_324() { // _embedding_bag_forward_only
      bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
      int64_t mode = readAttribute<int64_t>("mode");
      bool sparse = readAttribute<int64_t>("sparse");
      bool include_last_offset = readAttribute<int64_t>("include_last_offset");
      int64_t padding_idx = readAttribute<int64_t>("padding_idx");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto weight = peek(0, 4);
          auto indices = peek(1, 4);
          auto offsets = peek(2, 4);
          auto per_sample_weights = peek(3, 4);
          auto the_result = at::_embedding_bag_forward_only(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
            if(OutputSize() > 3) {assignTo(Output(3),::std::get<3>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_325() { // _embedding_bag_forward_only
      bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
      int64_t mode = readAttribute<int64_t>("mode");
      bool sparse = readAttribute<int64_t>("sparse");
      bool include_last_offset = readAttribute<int64_t>("include_last_offset");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto weight = peek(0, 4);
          auto indices = peek(1, 4);
          auto offsets = peek(2, 4);
          auto per_sample_weights = peek(3, 4);
          auto the_result = at::_embedding_bag_forward_only(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
            if(OutputSize() > 3) {assignTo(Output(3),::std::get<3>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_326() { // _embedding_bag_forward_only
      bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
      int64_t mode = readAttribute<int64_t>("mode");
      bool sparse = readAttribute<int64_t>("sparse");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto weight = peek(0, 4);
          auto indices = peek(1, 4);
          auto offsets = peek(2, 4);
          auto per_sample_weights = peek(3, 4);
          auto the_result = at::_embedding_bag_forward_only(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
            if(OutputSize() > 3) {assignTo(Output(3),::std::get<3>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_327() { // _embedding_bag_forward_only
      bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
      int64_t mode = readAttribute<int64_t>("mode");
      bool sparse = readAttribute<int64_t>("sparse");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto weight = peek(0, 3);
          auto indices = peek(1, 3);
          auto offsets = peek(2, 3);
          auto the_result = at::_embedding_bag_forward_only(weight, indices, offsets, scale_grad_by_freq, mode, sparse);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
            if(OutputSize() > 3) {assignTo(Output(3),::std::get<3>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_328() { // _embedding_bag_forward_only
      bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
      int64_t mode = readAttribute<int64_t>("mode");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto weight = peek(0, 3);
          auto indices = peek(1, 3);
          auto offsets = peek(2, 3);
          auto the_result = at::_embedding_bag_forward_only(weight, indices, offsets, scale_grad_by_freq, mode);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
            if(OutputSize() > 3) {assignTo(Output(3),::std::get<3>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_329() { // _embedding_bag_forward_only
      bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto weight = peek(0, 3);
          auto indices = peek(1, 3);
          auto offsets = peek(2, 3);
          auto the_result = at::_embedding_bag_forward_only(weight, indices, offsets, scale_grad_by_freq);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
            if(OutputSize() > 3) {assignTo(Output(3),::std::get<3>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_330() { // _embedding_bag_forward_only
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto weight = peek(0, 3);
          auto indices = peek(1, 3);
          auto offsets = peek(2, 3);
          auto the_result = at::_embedding_bag_forward_only(weight, indices, offsets);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
            if(OutputSize() > 3) {assignTo(Output(3),::std::get<3>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_331() { // row_stack
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::row_stack(tensors);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_332() { // embedding_bag
      bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
      int64_t mode = readAttribute<int64_t>("mode");
      bool sparse = readAttribute<int64_t>("sparse");
      bool include_last_offset = readAttribute<int64_t>("include_last_offset");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto weight = peek(0, 4);
          auto indices = peek(1, 4);
          auto offsets = peek(2, 4);
          auto per_sample_weights = peek(3, 4);
          auto the_result = at::embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
            if(OutputSize() > 3) {assignTo(Output(3),::std::get<3>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_333() { // embedding_bag
      bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
      int64_t mode = readAttribute<int64_t>("mode");
      bool sparse = readAttribute<int64_t>("sparse");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto weight = peek(0, 4);
          auto indices = peek(1, 4);
          auto offsets = peek(2, 4);
          auto per_sample_weights = peek(3, 4);
          auto the_result = at::embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
            if(OutputSize() > 3) {assignTo(Output(3),::std::get<3>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_334() { // embedding_bag
      bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
      int64_t mode = readAttribute<int64_t>("mode");
      bool sparse = readAttribute<int64_t>("sparse");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto weight = peek(0, 3);
          auto indices = peek(1, 3);
          auto offsets = peek(2, 3);
          auto the_result = at::embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
            if(OutputSize() > 3) {assignTo(Output(3),::std::get<3>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_335() { // embedding_bag
      bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
      int64_t mode = readAttribute<int64_t>("mode");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto weight = peek(0, 3);
          auto indices = peek(1, 3);
          auto offsets = peek(2, 3);
          auto the_result = at::embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
            if(OutputSize() > 3) {assignTo(Output(3),::std::get<3>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_336() { // embedding_bag
      bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto weight = peek(0, 3);
          auto indices = peek(1, 3);
          auto offsets = peek(2, 3);
          auto the_result = at::embedding_bag(weight, indices, offsets, scale_grad_by_freq);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
            if(OutputSize() > 3) {assignTo(Output(3),::std::get<3>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_337() { // embedding_bag
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto weight = peek(0, 3);
          auto indices = peek(1, 3);
          auto offsets = peek(2, 3);
          auto the_result = at::embedding_bag(weight, indices, offsets);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
            if(OutputSize() > 3) {assignTo(Output(3),::std::get<3>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_338() { // _embedding_bag
      bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
      int64_t mode = readAttribute<int64_t>("mode");
      bool sparse = readAttribute<int64_t>("sparse");
      bool include_last_offset = readAttribute<int64_t>("include_last_offset");
      int64_t padding_idx = readAttribute<int64_t>("padding_idx");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto weight = peek(0, 4);
          auto indices = peek(1, 4);
          auto offsets = peek(2, 4);
          auto per_sample_weights = peek(3, 4);
          auto the_result = at::_embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
            if(OutputSize() > 3) {assignTo(Output(3),::std::get<3>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_339() { // _embedding_bag
      bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
      int64_t mode = readAttribute<int64_t>("mode");
      bool sparse = readAttribute<int64_t>("sparse");
      bool include_last_offset = readAttribute<int64_t>("include_last_offset");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto weight = peek(0, 4);
          auto indices = peek(1, 4);
          auto offsets = peek(2, 4);
          auto per_sample_weights = peek(3, 4);
          auto the_result = at::_embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
            if(OutputSize() > 3) {assignTo(Output(3),::std::get<3>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_340() { // _embedding_bag
      bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
      int64_t mode = readAttribute<int64_t>("mode");
      bool sparse = readAttribute<int64_t>("sparse");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto weight = peek(0, 4);
          auto indices = peek(1, 4);
          auto offsets = peek(2, 4);
          auto per_sample_weights = peek(3, 4);
          auto the_result = at::_embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
            if(OutputSize() > 3) {assignTo(Output(3),::std::get<3>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_341() { // _embedding_bag
      bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
      int64_t mode = readAttribute<int64_t>("mode");
      bool sparse = readAttribute<int64_t>("sparse");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto weight = peek(0, 3);
          auto indices = peek(1, 3);
          auto offsets = peek(2, 3);
          auto the_result = at::_embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
            if(OutputSize() > 3) {assignTo(Output(3),::std::get<3>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_342() { // _embedding_bag
      bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
      int64_t mode = readAttribute<int64_t>("mode");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto weight = peek(0, 3);
          auto indices = peek(1, 3);
          auto offsets = peek(2, 3);
          auto the_result = at::_embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
            if(OutputSize() > 3) {assignTo(Output(3),::std::get<3>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_343() { // _embedding_bag
      bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto weight = peek(0, 3);
          auto indices = peek(1, 3);
          auto offsets = peek(2, 3);
          auto the_result = at::_embedding_bag(weight, indices, offsets, scale_grad_by_freq);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
            if(OutputSize() > 3) {assignTo(Output(3),::std::get<3>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_344() { // _embedding_bag
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto weight = peek(0, 3);
          auto indices = peek(1, 3);
          auto offsets = peek(2, 3);
          auto the_result = at::_embedding_bag(weight, indices, offsets);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
            if(OutputSize() > 3) {assignTo(Output(3),::std::get<3>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_345() { // _embedding_bag_backward
      int64_t num_weights = readAttribute<int64_t>("num_weights");
      bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
      int64_t mode = readAttribute<int64_t>("mode");
      bool sparse = readAttribute<int64_t>("sparse");
      int64_t padding_idx = readAttribute<int64_t>("padding_idx");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad = peek(0, 7);
          auto indices = peek(1, 7);
          auto offsets = peek(2, 7);
          auto offset2bag = peek(3, 7);
          auto bag_size = peek(4, 7);
          auto maximum_indices = peek(5, 7);
          auto per_sample_weights = peek(6, 7);
          auto the_result = at::_embedding_bag_backward(grad, indices, offsets, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, sparse, per_sample_weights, padding_idx);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_346() { // _embedding_bag_backward
      int64_t num_weights = readAttribute<int64_t>("num_weights");
      bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
      int64_t mode = readAttribute<int64_t>("mode");
      bool sparse = readAttribute<int64_t>("sparse");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad = peek(0, 7);
          auto indices = peek(1, 7);
          auto offsets = peek(2, 7);
          auto offset2bag = peek(3, 7);
          auto bag_size = peek(4, 7);
          auto maximum_indices = peek(5, 7);
          auto per_sample_weights = peek(6, 7);
          auto the_result = at::_embedding_bag_backward(grad, indices, offsets, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, sparse, per_sample_weights);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_347() { // _embedding_bag_sparse_backward
      int64_t num_weights = readAttribute<int64_t>("num_weights");
      bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
      int64_t mode = readAttribute<int64_t>("mode");
      int64_t padding_idx = readAttribute<int64_t>("padding_idx");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad = peek(0, 6);
          auto indices = peek(1, 6);
          auto offsets = peek(2, 6);
          auto offset2bag = peek(3, 6);
          auto bag_size = peek(4, 6);
          auto per_sample_weights = peek(5, 6);
          auto the_result = at::_embedding_bag_sparse_backward(grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode, per_sample_weights, padding_idx);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_348() { // _embedding_bag_sparse_backward
      int64_t num_weights = readAttribute<int64_t>("num_weights");
      bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
      int64_t mode = readAttribute<int64_t>("mode");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad = peek(0, 6);
          auto indices = peek(1, 6);
          auto offsets = peek(2, 6);
          auto offset2bag = peek(3, 6);
          auto bag_size = peek(4, 6);
          auto per_sample_weights = peek(5, 6);
          auto the_result = at::_embedding_bag_sparse_backward(grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode, per_sample_weights);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_349() { // _embedding_bag_dense_backward
      int64_t num_weights = readAttribute<int64_t>("num_weights");
      bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
      int64_t mode = readAttribute<int64_t>("mode");
      int64_t padding_idx = readAttribute<int64_t>("padding_idx");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad = peek(0, 6);
          auto indices = peek(1, 6);
          auto offset2bag = peek(2, 6);
          auto bag_size = peek(3, 6);
          auto maximum_indices = peek(4, 6);
          auto per_sample_weights = peek(5, 6);
          auto the_result = at::_embedding_bag_dense_backward(grad, indices, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, per_sample_weights, padding_idx);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_350() { // _embedding_bag_dense_backward
      int64_t num_weights = readAttribute<int64_t>("num_weights");
      bool scale_grad_by_freq = readAttribute<int64_t>("scale_grad_by_freq");
      int64_t mode = readAttribute<int64_t>("mode");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad = peek(0, 6);
          auto indices = peek(1, 6);
          auto offset2bag = peek(2, 6);
          auto bag_size = peek(3, 6);
          auto maximum_indices = peek(4, 6);
          auto per_sample_weights = peek(5, 6);
          auto the_result = at::_embedding_bag_dense_backward(grad, indices, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, per_sample_weights);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_351() { // _embedding_bag_per_sample_weights_backward
      int64_t mode = readAttribute<int64_t>("mode");
      int64_t padding_idx = readAttribute<int64_t>("padding_idx");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad = peek(0, 5);
          auto weight = peek(1, 5);
          auto indices = peek(2, 5);
          auto offsets = peek(3, 5);
          auto offset2bag = peek(4, 5);
          auto the_result = at::_embedding_bag_per_sample_weights_backward(grad, weight, indices, offsets, offset2bag, mode, padding_idx);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_352() { // _embedding_bag_per_sample_weights_backward
      int64_t mode = readAttribute<int64_t>("mode");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad = peek(0, 5);
          auto weight = peek(1, 5);
          auto indices = peek(2, 5);
          auto offsets = peek(3, 5);
          auto offset2bag = peek(4, 5);
          auto the_result = at::_embedding_bag_per_sample_weights_backward(grad, weight, indices, offsets, offset2bag, mode);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_353() { // erf
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::erf(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_354() { // erfc
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::erfc(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_355() { // exp
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::exp(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_356() { // exp2
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::exp2(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_357() { // expm1
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::expm1(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_358() { // expand
      auto size = readIntArrayRef("size");
      bool implicit = readAttribute<int64_t>("implicit");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = self.expand(size, implicit);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_359() { // expand
      auto size = readIntArrayRef("size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = self.expand(size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_360() { // expand_as
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = self.expand_as(other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_361() { // flatten
      int64_t start_dim = readAttribute<int64_t>("start_dim");
      int64_t end_dim = readAttribute<int64_t>("end_dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::flatten(self, start_dim, end_dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_362() { // flatten
      int64_t start_dim = readAttribute<int64_t>("start_dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::flatten(self, start_dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_363() { // flatten
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::flatten(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_364() { // unflatten
      int64_t dim = readAttribute<int64_t>("dim");
      auto sizes = readIntArrayRef("sizes");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = self.unflatten(dim, sizes);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_365() { // floor
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::floor(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_366() { // floor_divide
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::floor_divide(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_367() { // floor_divide
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::floor_divide(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_368() { // frac
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::frac(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_369() { // gcd
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::gcd(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_370() { // lcm
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::lcm(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_371() { // grid_sampler
      int64_t interpolation_mode = readAttribute<int64_t>("interpolation_mode");
      int64_t padding_mode = readAttribute<int64_t>("padding_mode");
      bool align_corners = readAttribute<int64_t>("align_corners");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 2);
          auto grid = peek(1, 2);
          auto the_result = at::grid_sampler(input, grid, interpolation_mode, padding_mode, align_corners);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_372() { // grid_sampler_2d
      int64_t interpolation_mode = readAttribute<int64_t>("interpolation_mode");
      int64_t padding_mode = readAttribute<int64_t>("padding_mode");
      bool align_corners = readAttribute<int64_t>("align_corners");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 2);
          auto grid = peek(1, 2);
          auto the_result = at::grid_sampler_2d(input, grid, interpolation_mode, padding_mode, align_corners);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_373() { // grid_sampler_2d_backward
      int64_t interpolation_mode = readAttribute<int64_t>("interpolation_mode");
      int64_t padding_mode = readAttribute<int64_t>("padding_mode");
      bool align_corners = readAttribute<int64_t>("align_corners");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto input = peek(1, 3);
          auto grid = peek(2, 3);
          auto the_result = at::grid_sampler_2d_backward(grad_output, input, grid, interpolation_mode, padding_mode, align_corners);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_374() { // _grid_sampler_2d_cpu_fallback
      int64_t interpolation_mode = readAttribute<int64_t>("interpolation_mode");
      int64_t padding_mode = readAttribute<int64_t>("padding_mode");
      bool align_corners = readAttribute<int64_t>("align_corners");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 2);
          auto grid = peek(1, 2);
          auto the_result = at::_grid_sampler_2d_cpu_fallback(input, grid, interpolation_mode, padding_mode, align_corners);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_375() { // _grid_sampler_2d_cpu_fallback_backward
      int64_t interpolation_mode = readAttribute<int64_t>("interpolation_mode");
      int64_t padding_mode = readAttribute<int64_t>("padding_mode");
      bool align_corners = readAttribute<int64_t>("align_corners");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto input = peek(1, 3);
          auto grid = peek(2, 3);
          auto the_result = at::_grid_sampler_2d_cpu_fallback_backward(grad_output, input, grid, interpolation_mode, padding_mode, align_corners);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_376() { // grid_sampler_3d
      int64_t interpolation_mode = readAttribute<int64_t>("interpolation_mode");
      int64_t padding_mode = readAttribute<int64_t>("padding_mode");
      bool align_corners = readAttribute<int64_t>("align_corners");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 2);
          auto grid = peek(1, 2);
          auto the_result = at::grid_sampler_3d(input, grid, interpolation_mode, padding_mode, align_corners);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_377() { // grid_sampler_3d_backward
      int64_t interpolation_mode = readAttribute<int64_t>("interpolation_mode");
      int64_t padding_mode = readAttribute<int64_t>("padding_mode");
      bool align_corners = readAttribute<int64_t>("align_corners");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto input = peek(1, 3);
          auto grid = peek(2, 3);
          auto the_result = at::grid_sampler_3d_backward(grad_output, input, grid, interpolation_mode, padding_mode, align_corners);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_378() { // hinge_embedding_loss
      double margin = readAttribute<float>("margin");
      int64_t reduction = readAttribute<int64_t>("reduction");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto target = peek(1, 2);
          auto the_result = at::hinge_embedding_loss(self, target, margin, reduction);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_379() { // hinge_embedding_loss
      double margin = readAttribute<float>("margin");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto target = peek(1, 2);
          auto the_result = at::hinge_embedding_loss(self, target, margin);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_380() { // hinge_embedding_loss
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto target = peek(1, 2);
          auto the_result = at::hinge_embedding_loss(self, target);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_381() { // group_norm
      int64_t num_groups = readAttribute<int64_t>("num_groups");
      double eps = readAttribute<float>("eps");
      bool cudnn_enabled = readAttribute<int64_t>("cudnn_enabled");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::group_norm(input, num_groups, weight, bias, eps, cudnn_enabled);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_382() { // group_norm
      int64_t num_groups = readAttribute<int64_t>("num_groups");
      double eps = readAttribute<float>("eps");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::group_norm(input, num_groups, weight, bias, eps);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_383() { // group_norm
      int64_t num_groups = readAttribute<int64_t>("num_groups");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::group_norm(input, num_groups, weight, bias);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_384() { // group_norm
      int64_t num_groups = readAttribute<int64_t>("num_groups");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 2);
          auto weight = peek(1, 2);
          auto the_result = at::group_norm(input, num_groups, weight);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_385() { // group_norm
      int64_t num_groups = readAttribute<int64_t>("num_groups");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 1);
          auto the_result = at::group_norm(input, num_groups);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_386() { // native_group_norm
      int64_t N = readAttribute<int64_t>("N");
      int64_t C = readAttribute<int64_t>("C");
      int64_t HxW = readAttribute<int64_t>("HxW");
      int64_t group = readAttribute<int64_t>("group");
      double eps = readAttribute<float>("eps");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::native_group_norm(input, weight, bias, N, C, HxW, group, eps);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_387() { // native_group_norm_backward
      int64_t N = readAttribute<int64_t>("N");
      int64_t C = readAttribute<int64_t>("C");
      int64_t HxW = readAttribute<int64_t>("HxW");
      int64_t group = readAttribute<int64_t>("group");
      auto output_mask = readBoolMask<3>("output_mask");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_out = peek(0, 5);
          auto input = peek(1, 5);
          auto mean = peek(2, 5);
          auto rstd = peek(3, 5);
          auto weight = peek(4, 5);
          auto the_result = at::native_group_norm_backward(grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_388() { // _fft_r2c
      auto dim = readIntArrayRef("dim");
      int64_t normalization = readAttribute<int64_t>("normalization");
      bool onesided = readAttribute<int64_t>("onesided");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_fft_r2c(self, dim, normalization, onesided);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_389() { // _fft_c2r
      auto dim = readIntArrayRef("dim");
      int64_t normalization = readAttribute<int64_t>("normalization");
      int64_t last_dim_size = readAttribute<int64_t>("last_dim_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_fft_c2r(self, dim, normalization, last_dim_size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_390() { // _fft_c2c
      auto dim = readIntArrayRef("dim");
      int64_t normalization = readAttribute<int64_t>("normalization");
      bool forward = readAttribute<int64_t>("forward");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_fft_c2c(self, dim, normalization, forward);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_391() { // _cufft_get_plan_cache_size
      int64_t device_index = readAttribute<int64_t>("device_index");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
  
          auto the_result = at::_cufft_get_plan_cache_size(device_index);
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_392() { // _cufft_get_plan_cache_max_size
      int64_t device_index = readAttribute<int64_t>("device_index");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
  
          auto the_result = at::_cufft_get_plan_cache_max_size(device_index);
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_393() { // index
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, InputSize());
          auto indices = peekSliceOptionals(1, InputSize() - 1, InputSize());
          auto the_result = internal::index_with_uint8_handling(self,indices);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_394() { // index_copy
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto index = peek(1, 3);
          auto source = peek(2, 3);
          auto the_result = at::index_copy(self, dim, index, source);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_395() { // index_put
      bool accumulate = readAttribute<int64_t>("accumulate");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, InputSize());
          auto indices = peekSliceOptionals(1, InputSize() - 2, InputSize());
          auto values = peek(1, 2);
          auto the_result = at::index_put(self, indices, values, accumulate);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_396() { // index_put
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, InputSize());
          auto indices = peekSliceOptionals(1, InputSize() - 2, InputSize());
          auto values = peek(1, 2);
          auto the_result = at::index_put(self, indices, values);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_397() { // instance_norm
      bool use_input_stats = readAttribute<int64_t>("use_input_stats");
      double momentum = readAttribute<float>("momentum");
      double eps = readAttribute<float>("eps");
      bool cudnn_enabled = readAttribute<int64_t>("cudnn_enabled");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 5);
          auto weight = peek(1, 5);
          auto bias = peek(2, 5);
          auto running_mean = peek(3, 5);
          auto running_var = peek(4, 5);
          auto the_result = at::instance_norm(input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_398() { // inverse
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::inverse(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_399() { // _inverse_helper
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_inverse_helper(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_400() { // isclose
      double rtol = readAttribute<float>("rtol");
      double atol = readAttribute<float>("atol");
      bool equal_nan = readAttribute<int64_t>("equal_nan");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::isclose(self, other, rtol, atol, equal_nan);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_401() { // isclose
      double rtol = readAttribute<float>("rtol");
      double atol = readAttribute<float>("atol");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::isclose(self, other, rtol, atol);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_402() { // isclose
      double rtol = readAttribute<float>("rtol");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::isclose(self, other, rtol);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_403() { // isclose
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::isclose(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_404() { // isin
      bool assume_unique = readAttribute<int64_t>("assume_unique");
      bool invert = readAttribute<int64_t>("invert");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto elements = peek(0, 2);
          auto test_elements = peek(1, 2);
          auto the_result = at::isin(elements, test_elements, assume_unique, invert);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_405() { // isin
      bool assume_unique = readAttribute<int64_t>("assume_unique");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto elements = peek(0, 2);
          auto test_elements = peek(1, 2);
          auto the_result = at::isin(elements, test_elements, assume_unique);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_406() { // isin
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto elements = peek(0, 2);
          auto test_elements = peek(1, 2);
          auto the_result = at::isin(elements, test_elements);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_407() { // isin
      at::Scalar test_element = readScalarAttribute("test_element");
      bool assume_unique = readAttribute<int64_t>("assume_unique");
      bool invert = readAttribute<int64_t>("invert");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto elements = peek(0, 1);
          auto the_result = at::isin(elements, test_element, assume_unique, invert);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_408() { // isin
      at::Scalar test_element = readScalarAttribute("test_element");
      bool assume_unique = readAttribute<int64_t>("assume_unique");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto elements = peek(0, 1);
          auto the_result = at::isin(elements, test_element, assume_unique);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_409() { // isin
      at::Scalar test_element = readScalarAttribute("test_element");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto elements = peek(0, 1);
          auto the_result = at::isin(elements, test_element);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_410() { // isin
      at::Scalar element = readScalarAttribute("element");
      bool assume_unique = readAttribute<int64_t>("assume_unique");
      bool invert = readAttribute<int64_t>("invert");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto test_elements = peek(0, 1);
          auto the_result = at::isin(element, test_elements, assume_unique, invert);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_411() { // isin
      at::Scalar element = readScalarAttribute("element");
      bool assume_unique = readAttribute<int64_t>("assume_unique");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto test_elements = peek(0, 1);
          auto the_result = at::isin(element, test_elements, assume_unique);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_412() { // isin
      at::Scalar element = readScalarAttribute("element");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto test_elements = peek(0, 1);
          auto the_result = at::isin(element, test_elements);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_413() { // isnan
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::isnan(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_414() { // is_distributed
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::is_distributed(self);
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_415() { // is_floating_point
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::is_floating_point(self);
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_416() { // is_complex
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::is_complex(self);
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_417() { // is_conj
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::is_conj(self);
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_418() { // is_neg
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::is_neg(self);
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_419() { // isreal
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::isreal(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_420() { // is_nonzero
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::is_nonzero(self);
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_421() { // is_same_size
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::is_same_size(self, other);
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_422() { // is_signed
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::is_signed(self);
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_423() { // is_inference
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::is_inference(self);
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_424() { // kl_div
      int64_t reduction = readAttribute<int64_t>("reduction");
      bool log_target = readAttribute<int64_t>("log_target");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto target = peek(1, 2);
          auto the_result = at::kl_div(self, target, reduction, log_target);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_425() { // kl_div
      int64_t reduction = readAttribute<int64_t>("reduction");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto target = peek(1, 2);
          auto the_result = at::kl_div(self, target, reduction);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_426() { // kl_div
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto target = peek(1, 2);
          auto the_result = at::kl_div(self, target);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_427() { // kl_div_backward
      int64_t reduction = readAttribute<int64_t>("reduction");
      bool log_target = readAttribute<int64_t>("log_target");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto self = peek(1, 3);
          auto target = peek(2, 3);
          auto the_result = at::kl_div_backward(grad_output, self, target, reduction, log_target);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_428() { // kl_div_backward
      int64_t reduction = readAttribute<int64_t>("reduction");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto self = peek(1, 3);
          auto target = peek(2, 3);
          auto the_result = at::kl_div_backward(grad_output, self, target, reduction);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_429() { // kl_div_backward
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto self = peek(1, 3);
          auto target = peek(2, 3);
          auto the_result = at::kl_div_backward(grad_output, self, target);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_430() { // kron
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::kron(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_431() { // kthvalue
      int64_t k = readAttribute<int64_t>("k");
      int64_t dim = readAttribute<int64_t>("dim");
      bool keepdim = readAttribute<int64_t>("keepdim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::kthvalue(self, k, dim, keepdim);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_432() { // kthvalue
      int64_t k = readAttribute<int64_t>("k");
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::kthvalue(self, k, dim);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_433() { // kthvalue
      int64_t k = readAttribute<int64_t>("k");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::kthvalue(self, k);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_434() { // layer_norm
      auto normalized_shape = readIntArrayRef("normalized_shape");
      double eps = readAttribute<float>("eps");
      bool cudnn_enable = readAttribute<int64_t>("cudnn_enable");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::layer_norm(input, normalized_shape, weight, bias, eps, cudnn_enable);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_435() { // layer_norm
      auto normalized_shape = readIntArrayRef("normalized_shape");
      double eps = readAttribute<float>("eps");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::layer_norm(input, normalized_shape, weight, bias, eps);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_436() { // layer_norm
      auto normalized_shape = readIntArrayRef("normalized_shape");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::layer_norm(input, normalized_shape, weight, bias);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_437() { // layer_norm
      auto normalized_shape = readIntArrayRef("normalized_shape");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 2);
          auto weight = peek(1, 2);
          auto the_result = at::layer_norm(input, normalized_shape, weight);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_438() { // layer_norm
      auto normalized_shape = readIntArrayRef("normalized_shape");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 1);
          auto the_result = at::layer_norm(input, normalized_shape);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_439() { // native_layer_norm
      auto normalized_shape = readIntArrayRef("normalized_shape");
      double eps = readAttribute<float>("eps");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::native_layer_norm(input, normalized_shape, weight, bias, eps);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_440() { // native_layer_norm_backward
      auto normalized_shape = readIntArrayRef("normalized_shape");
      auto output_mask = readBoolMask<3>("output_mask");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_out = peek(0, 6);
          auto input = peek(1, 6);
          auto mean = peek(2, 6);
          auto rstd = peek(3, 6);
          auto weight = peek(4, 6);
          auto bias = peek(5, 6);
          auto the_result = at::native_layer_norm_backward(grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_441() { // nan_to_num
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::nan_to_num(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_442() { // linear
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::linear(input, weight, bias);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_443() { // linear
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 2);
          auto weight = peek(1, 2);
          auto the_result = at::linear(input, weight);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_444() { // mkldnn_linear
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::mkldnn_linear(self, weight, bias);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_445() { // mkldnn_linear
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto weight = peek(1, 2);
          auto the_result = at::mkldnn_linear(self, weight);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_446() { // mkldnn_linear_backward_input
      auto input_size = readIntArrayRef("input_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 2);
          auto weight = peek(1, 2);
          auto the_result = at::mkldnn_linear_backward_input(input_size, grad_output, weight);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_447() { // mkldnn_linear_backward_weights
      bool bias_defined = readAttribute<int64_t>("bias_defined");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto input = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::mkldnn_linear_backward_weights(grad_output, input, weight, bias_defined);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_448() { // mkldnn_linear_backward
      auto output_mask = readBoolMask<3>("output_mask");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto grad_output = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::mkldnn_linear_backward(self, grad_output, weight, output_mask);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_449() { // fbgemm_linear_int8_weight_fp32_activation
      at::Scalar weight_scale = readScalarAttribute("weight_scale");
      at::Scalar weight_zero_point = readScalarAttribute("weight_zero_point");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 5);
          auto weight = peek(1, 5);
          auto packed = peek(2, 5);
          auto col_offsets = peek(3, 5);
          auto bias = peek(4, 5);
          auto the_result = at::fbgemm_linear_int8_weight_fp32_activation(input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_450() { // fbgemm_linear_int8_weight
      at::Scalar weight_scale = readScalarAttribute("weight_scale");
      at::Scalar weight_zero_point = readScalarAttribute("weight_zero_point");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 5);
          auto weight = peek(1, 5);
          auto packed = peek(2, 5);
          auto col_offsets = peek(3, 5);
          auto bias = peek(4, 5);
          auto the_result = at::fbgemm_linear_int8_weight(input, weight, packed, col_offsets, weight_scale, weight_zero_point, bias);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_451() { // fbgemm_pack_gemm_matrix_fp16
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 1);
          auto the_result = at::fbgemm_pack_gemm_matrix_fp16(input);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_452() { // fbgemm_linear_fp16_weight_fp32_activation
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto packed_weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::fbgemm_linear_fp16_weight_fp32_activation(input, packed_weight, bias);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_453() { // fbgemm_linear_fp16_weight
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto packed_weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::fbgemm_linear_fp16_weight(input, packed_weight, bias);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_454() { // fbgemm_pack_quantized_matrix
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 1);
          auto the_result = at::fbgemm_pack_quantized_matrix(input);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_455() { // fbgemm_pack_quantized_matrix
      int64_t K = readAttribute<int64_t>("K");
      int64_t N = readAttribute<int64_t>("N");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 1);
          auto the_result = at::fbgemm_pack_quantized_matrix(input, K, N);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_456() { // ldexp
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::ldexp(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_457() { // log
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::log(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_458() { // log10
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::log10(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_459() { // log1p
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::log1p(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_460() { // log2
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::log2(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_461() { // logaddexp
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::logaddexp(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_462() { // logaddexp2
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::logaddexp2(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_463() { // xlogy
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::xlogy(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_464() { // xlogy
      at::Scalar self = readScalarAttribute("self");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto other = peek(0, 1);
          auto the_result = at::xlogy(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_465() { // xlogy
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::xlogy(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_466() { // logdet
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::logdet(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_467() { // log_softmax
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::log_softmax(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_468() { // _log_softmax
      int64_t dim = readAttribute<int64_t>("dim");
      bool half_to_float = readAttribute<int64_t>("half_to_float");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_log_softmax(self, dim, half_to_float);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_469() { // _log_softmax_backward_data
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto output = peek(1, 3);
          auto self = peek(2, 3);
          auto the_result = at::_log_softmax_backward_data(grad_output, output, dim, self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_470() { // _logcumsumexp
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_logcumsumexp(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_471() { // logcumsumexp
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::logcumsumexp(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_472() { // logsumexp
      auto dim = readIntArrayRef("dim");
      bool keepdim = readAttribute<int64_t>("keepdim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::logsumexp(self, dim, keepdim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_473() { // logsumexp
      auto dim = readIntArrayRef("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::logsumexp(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_474() { // margin_ranking_loss
      double margin = readAttribute<float>("margin");
      int64_t reduction = readAttribute<int64_t>("reduction");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input1 = peek(0, 3);
          auto input2 = peek(1, 3);
          auto target = peek(2, 3);
          auto the_result = at::margin_ranking_loss(input1, input2, target, margin, reduction);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_475() { // margin_ranking_loss
      double margin = readAttribute<float>("margin");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input1 = peek(0, 3);
          auto input2 = peek(1, 3);
          auto target = peek(2, 3);
          auto the_result = at::margin_ranking_loss(input1, input2, target, margin);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_476() { // margin_ranking_loss
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input1 = peek(0, 3);
          auto input2 = peek(1, 3);
          auto target = peek(2, 3);
          auto the_result = at::margin_ranking_loss(input1, input2, target);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_477() { // matmul
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::matmul(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_478() { // matrix_rank
      double tol = readAttribute<float>("tol");
      bool symmetric = readAttribute<int64_t>("symmetric");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::matrix_rank(self, tol, symmetric);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_479() { // matrix_rank
      double tol = readAttribute<float>("tol");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::matrix_rank(self, tol);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_480() { // matrix_rank
      bool symmetric = readAttribute<int64_t>("symmetric");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::matrix_rank(self, symmetric);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_481() { // matrix_rank
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::matrix_rank(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_482() { // matrix_power
      int64_t n = readAttribute<int64_t>("n");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::matrix_power(self, n);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_483() { // matrix_exp
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::matrix_exp(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_484() { // matrix_exp_backward
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto grad = peek(1, 2);
          auto the_result = at::matrix_exp_backward(self, grad);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_485() { // _aminmax
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_aminmax(self);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_486() { // _aminmax
      int64_t dim = readAttribute<int64_t>("dim");
      bool keepdim = readAttribute<int64_t>("keepdim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_aminmax(self, dim, keepdim);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_487() { // _aminmax
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_aminmax(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_488() { // aminmax
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::aminmax(self);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_489() { // _compute_linear_combination
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 2);
          auto coefficients = peek(1, 2);
          auto the_result = at::_compute_linear_combination(input, coefficients);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_490() { // max
      int64_t dim = readAttribute<int64_t>("dim");
      bool keepdim = readAttribute<int64_t>("keepdim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::max(self, dim, keepdim);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_491() { // max
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::max(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_492() { // value_selecting_reduction_backward
      int64_t dim = readAttribute<int64_t>("dim");
      auto sizes = readIntArrayRef("sizes");
      bool keepdim = readAttribute<int64_t>("keepdim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad = peek(0, 2);
          auto indices = peek(1, 2);
          auto the_result = at::value_selecting_reduction_backward(grad, dim, indices, sizes, keepdim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_493() { // amax
      auto dim = readIntArrayRef("dim");
      bool keepdim = readAttribute<int64_t>("keepdim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::amax(self, dim, keepdim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_494() { // amax
      auto dim = readIntArrayRef("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::amax(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_495() { // amax
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::amax(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_496() { // max_pool1d_with_indices
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      bool ceil_mode = readAttribute<int64_t>("ceil_mode");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::max_pool1d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_497() { // max_pool1d_with_indices
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::max_pool1d_with_indices(self, kernel_size, stride, padding, dilation);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_498() { // max_pool1d_with_indices
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::max_pool1d_with_indices(self, kernel_size, stride, padding);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_499() { // max_pool1d_with_indices
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::max_pool1d_with_indices(self, kernel_size, stride);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_500() { // max_pool1d_with_indices
      auto kernel_size = readIntArrayRef("kernel_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::max_pool1d_with_indices(self, kernel_size);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_501() { // max_pool1d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      bool ceil_mode = readAttribute<int64_t>("ceil_mode");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::max_pool1d(self, kernel_size, stride, padding, dilation, ceil_mode);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_502() { // max_pool1d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::max_pool1d(self, kernel_size, stride, padding, dilation);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_503() { // max_pool1d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::max_pool1d(self, kernel_size, stride, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_504() { // max_pool1d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::max_pool1d(self, kernel_size, stride);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_505() { // max_pool1d
      auto kernel_size = readIntArrayRef("kernel_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::max_pool1d(self, kernel_size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_506() { // max_pool2d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      bool ceil_mode = readAttribute<int64_t>("ceil_mode");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_507() { // max_pool2d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::max_pool2d(self, kernel_size, stride, padding, dilation);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_508() { // max_pool2d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::max_pool2d(self, kernel_size, stride, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_509() { // max_pool2d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::max_pool2d(self, kernel_size, stride);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_510() { // max_pool2d
      auto kernel_size = readIntArrayRef("kernel_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::max_pool2d(self, kernel_size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_511() { // mkldnn_max_pool2d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      bool ceil_mode = readAttribute<int64_t>("ceil_mode");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::mkldnn_max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_512() { // mkldnn_max_pool2d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::mkldnn_max_pool2d(self, kernel_size, stride, padding, dilation);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_513() { // mkldnn_max_pool2d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::mkldnn_max_pool2d(self, kernel_size, stride, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_514() { // mkldnn_max_pool2d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::mkldnn_max_pool2d(self, kernel_size, stride);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_515() { // mkldnn_max_pool2d
      auto kernel_size = readIntArrayRef("kernel_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::mkldnn_max_pool2d(self, kernel_size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_516() { // mkldnn_max_pool2d_backward
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      bool ceil_mode = readAttribute<int64_t>("ceil_mode");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto output = peek(1, 3);
          auto input = peek(2, 3);
          auto the_result = at::mkldnn_max_pool2d_backward(grad_output, output, input, kernel_size, stride, padding, dilation, ceil_mode);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_517() { // mkldnn_max_pool2d_backward
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto output = peek(1, 3);
          auto input = peek(2, 3);
          auto the_result = at::mkldnn_max_pool2d_backward(grad_output, output, input, kernel_size, stride, padding, dilation);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_518() { // mkldnn_max_pool2d_backward
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto output = peek(1, 3);
          auto input = peek(2, 3);
          auto the_result = at::mkldnn_max_pool2d_backward(grad_output, output, input, kernel_size, stride, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_519() { // mkldnn_max_pool2d_backward
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto output = peek(1, 3);
          auto input = peek(2, 3);
          auto the_result = at::mkldnn_max_pool2d_backward(grad_output, output, input, kernel_size, stride);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_520() { // mkldnn_max_pool2d_backward
      auto kernel_size = readIntArrayRef("kernel_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto output = peek(1, 3);
          auto input = peek(2, 3);
          auto the_result = at::mkldnn_max_pool2d_backward(grad_output, output, input, kernel_size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_521() { // mkldnn_max_pool3d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      bool ceil_mode = readAttribute<int64_t>("ceil_mode");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::mkldnn_max_pool3d(self, kernel_size, stride, padding, dilation, ceil_mode);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_522() { // mkldnn_max_pool3d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::mkldnn_max_pool3d(self, kernel_size, stride, padding, dilation);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_523() { // mkldnn_max_pool3d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::mkldnn_max_pool3d(self, kernel_size, stride, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_524() { // mkldnn_max_pool3d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::mkldnn_max_pool3d(self, kernel_size, stride);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_525() { // mkldnn_max_pool3d
      auto kernel_size = readIntArrayRef("kernel_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::mkldnn_max_pool3d(self, kernel_size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_526() { // mkldnn_max_pool3d_backward
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      bool ceil_mode = readAttribute<int64_t>("ceil_mode");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto output = peek(1, 3);
          auto input = peek(2, 3);
          auto the_result = at::mkldnn_max_pool3d_backward(grad_output, output, input, kernel_size, stride, padding, dilation, ceil_mode);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_527() { // mkldnn_max_pool3d_backward
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto output = peek(1, 3);
          auto input = peek(2, 3);
          auto the_result = at::mkldnn_max_pool3d_backward(grad_output, output, input, kernel_size, stride, padding, dilation);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_528() { // mkldnn_max_pool3d_backward
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto output = peek(1, 3);
          auto input = peek(2, 3);
          auto the_result = at::mkldnn_max_pool3d_backward(grad_output, output, input, kernel_size, stride, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_529() { // mkldnn_max_pool3d_backward
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto output = peek(1, 3);
          auto input = peek(2, 3);
          auto the_result = at::mkldnn_max_pool3d_backward(grad_output, output, input, kernel_size, stride);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_530() { // mkldnn_max_pool3d_backward
      auto kernel_size = readIntArrayRef("kernel_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto output = peek(1, 3);
          auto input = peek(2, 3);
          auto the_result = at::mkldnn_max_pool3d_backward(grad_output, output, input, kernel_size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_531() { // quantized_max_pool1d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      bool ceil_mode = readAttribute<int64_t>("ceil_mode");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::quantized_max_pool1d(self, kernel_size, stride, padding, dilation, ceil_mode);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_532() { // quantized_max_pool1d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::quantized_max_pool1d(self, kernel_size, stride, padding, dilation);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_533() { // quantized_max_pool1d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::quantized_max_pool1d(self, kernel_size, stride, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_534() { // quantized_max_pool1d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::quantized_max_pool1d(self, kernel_size, stride);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_535() { // quantized_max_pool1d
      auto kernel_size = readIntArrayRef("kernel_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::quantized_max_pool1d(self, kernel_size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_536() { // quantized_max_pool2d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      bool ceil_mode = readAttribute<int64_t>("ceil_mode");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::quantized_max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_537() { // quantized_max_pool2d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::quantized_max_pool2d(self, kernel_size, stride, padding, dilation);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_538() { // quantized_max_pool2d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::quantized_max_pool2d(self, kernel_size, stride, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_539() { // quantized_max_pool2d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::quantized_max_pool2d(self, kernel_size, stride);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_540() { // quantized_max_pool2d
      auto kernel_size = readIntArrayRef("kernel_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::quantized_max_pool2d(self, kernel_size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_541() { // max_pool3d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      bool ceil_mode = readAttribute<int64_t>("ceil_mode");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::max_pool3d(self, kernel_size, stride, padding, dilation, ceil_mode);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_542() { // max_pool3d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::max_pool3d(self, kernel_size, stride, padding, dilation);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_543() { // max_pool3d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::max_pool3d(self, kernel_size, stride, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_544() { // max_pool3d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::max_pool3d(self, kernel_size, stride);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_545() { // max_pool3d
      auto kernel_size = readIntArrayRef("kernel_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::max_pool3d(self, kernel_size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_546() { // mean
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::mean(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_547() { // mean
      auto dim = readIntArrayRef("dim");
      bool keepdim = readAttribute<int64_t>("keepdim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::mean(self, dim, keepdim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_548() { // mean
      auto dim = readIntArrayRef("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::mean(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_549() { // nanmean
      auto dim = readIntArrayRef("dim");
      bool keepdim = readAttribute<int64_t>("keepdim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::nanmean(self, dim, keepdim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_550() { // nanmean
      auto dim = readIntArrayRef("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::nanmean(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_551() { // nanmean
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::nanmean(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_552() { // median
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::median(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_553() { // median
      int64_t dim = readAttribute<int64_t>("dim");
      bool keepdim = readAttribute<int64_t>("keepdim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::median(self, dim, keepdim);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_554() { // median
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::median(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_555() { // nanmedian
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::nanmedian(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_556() { // nanmedian
      int64_t dim = readAttribute<int64_t>("dim");
      bool keepdim = readAttribute<int64_t>("keepdim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::nanmedian(self, dim, keepdim);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_557() { // nanmedian
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::nanmedian(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_558() { // min
      int64_t dim = readAttribute<int64_t>("dim");
      bool keepdim = readAttribute<int64_t>("keepdim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::min(self, dim, keepdim);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_559() { // min
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::min(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_560() { // amin
      auto dim = readIntArrayRef("dim");
      bool keepdim = readAttribute<int64_t>("keepdim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::amin(self, dim, keepdim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_561() { // amin
      auto dim = readIntArrayRef("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::amin(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_562() { // amin
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::amin(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_563() { // mkldnn_convolution
      auto padding = readIntArrayRef("padding");
      auto stride = readIntArrayRef("stride");
      auto dilation = readIntArrayRef("dilation");
      int64_t groups = readAttribute<int64_t>("groups");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::mkldnn_convolution(self, weight, bias, padding, stride, dilation, groups);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_564() { // mkldnn_convolution_backward_input
      auto self_size = readIntArrayRef("self_size");
      auto padding = readIntArrayRef("padding");
      auto stride = readIntArrayRef("stride");
      auto dilation = readIntArrayRef("dilation");
      int64_t groups = readAttribute<int64_t>("groups");
      bool bias_defined = readAttribute<int64_t>("bias_defined");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 2);
          auto weight = peek(1, 2);
          auto the_result = at::mkldnn_convolution_backward_input(self_size, grad_output, weight, padding, stride, dilation, groups, bias_defined);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_565() { // mkldnn_convolution_backward_weights
      auto weight_size = readIntArrayRef("weight_size");
      auto padding = readIntArrayRef("padding");
      auto stride = readIntArrayRef("stride");
      auto dilation = readIntArrayRef("dilation");
      int64_t groups = readAttribute<int64_t>("groups");
      bool bias_defined = readAttribute<int64_t>("bias_defined");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 2);
          auto self = peek(1, 2);
          auto the_result = at::mkldnn_convolution_backward_weights(weight_size, grad_output, self, padding, stride, dilation, groups, bias_defined);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_566() { // mkldnn_convolution_backward
      auto padding = readIntArrayRef("padding");
      auto stride = readIntArrayRef("stride");
      auto dilation = readIntArrayRef("dilation");
      int64_t groups = readAttribute<int64_t>("groups");
      auto output_mask = readBoolMask<3>("output_mask");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto grad_output = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::mkldnn_convolution_backward(self, grad_output, weight, padding, stride, dilation, groups, output_mask);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_567() { // miopen_batch_norm
      bool training = readAttribute<int64_t>("training");
      double exponential_average_factor = readAttribute<float>("exponential_average_factor");
      double epsilon = readAttribute<float>("epsilon");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 5);
          auto weight = peek(1, 5);
          auto bias = peek(2, 5);
          auto running_mean = peek(3, 5);
          auto running_var = peek(4, 5);
          auto the_result = at::miopen_batch_norm(input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_568() { // miopen_batch_norm_backward
      double epsilon = readAttribute<float>("epsilon");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 7);
          auto grad_output = peek(1, 7);
          auto weight = peek(2, 7);
          auto running_mean = peek(3, 7);
          auto running_var = peek(4, 7);
          auto save_mean = peek(5, 7);
          auto save_var = peek(6, 7);
          auto the_result = at::miopen_batch_norm_backward(input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_569() { // miopen_convolution
      auto padding = readIntArrayRef("padding");
      auto stride = readIntArrayRef("stride");
      auto dilation = readIntArrayRef("dilation");
      int64_t groups = readAttribute<int64_t>("groups");
      bool benchmark = readAttribute<int64_t>("benchmark");
      bool deterministic = readAttribute<int64_t>("deterministic");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::miopen_convolution(self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_570() { // miopen_convolution_backward_input
      auto self_size = readIntArrayRef("self_size");
      auto padding = readIntArrayRef("padding");
      auto stride = readIntArrayRef("stride");
      auto dilation = readIntArrayRef("dilation");
      int64_t groups = readAttribute<int64_t>("groups");
      bool benchmark = readAttribute<int64_t>("benchmark");
      bool deterministic = readAttribute<int64_t>("deterministic");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 2);
          auto weight = peek(1, 2);
          auto the_result = at::miopen_convolution_backward_input(self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_571() { // miopen_convolution_backward
      auto padding = readIntArrayRef("padding");
      auto stride = readIntArrayRef("stride");
      auto dilation = readIntArrayRef("dilation");
      int64_t groups = readAttribute<int64_t>("groups");
      bool benchmark = readAttribute<int64_t>("benchmark");
      bool deterministic = readAttribute<int64_t>("deterministic");
      auto output_mask = readBoolMask<3>("output_mask");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto grad_output = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::miopen_convolution_backward(self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, output_mask);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_572() { // miopen_convolution_backward_bias
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 1);
          auto the_result = at::miopen_convolution_backward_bias(grad_output);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_573() { // miopen_convolution_backward_weight
      auto weight_size = readIntArrayRef("weight_size");
      auto padding = readIntArrayRef("padding");
      auto stride = readIntArrayRef("stride");
      auto dilation = readIntArrayRef("dilation");
      int64_t groups = readAttribute<int64_t>("groups");
      bool benchmark = readAttribute<int64_t>("benchmark");
      bool deterministic = readAttribute<int64_t>("deterministic");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 2);
          auto self = peek(1, 2);
          auto the_result = at::miopen_convolution_backward_weight(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_574() { // miopen_convolution_transpose
      auto padding = readIntArrayRef("padding");
      auto output_padding = readIntArrayRef("output_padding");
      auto stride = readIntArrayRef("stride");
      auto dilation = readIntArrayRef("dilation");
      int64_t groups = readAttribute<int64_t>("groups");
      bool benchmark = readAttribute<int64_t>("benchmark");
      bool deterministic = readAttribute<int64_t>("deterministic");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::miopen_convolution_transpose(self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_575() { // miopen_convolution_transpose_backward
      auto padding = readIntArrayRef("padding");
      auto output_padding = readIntArrayRef("output_padding");
      auto stride = readIntArrayRef("stride");
      auto dilation = readIntArrayRef("dilation");
      int64_t groups = readAttribute<int64_t>("groups");
      bool benchmark = readAttribute<int64_t>("benchmark");
      bool deterministic = readAttribute<int64_t>("deterministic");
      auto output_mask = readBoolMask<3>("output_mask");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto grad_output = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::miopen_convolution_transpose_backward(self, grad_output, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, output_mask);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_576() { // miopen_convolution_transpose_backward_input
      auto padding = readIntArrayRef("padding");
      auto stride = readIntArrayRef("stride");
      auto dilation = readIntArrayRef("dilation");
      int64_t groups = readAttribute<int64_t>("groups");
      bool benchmark = readAttribute<int64_t>("benchmark");
      bool deterministic = readAttribute<int64_t>("deterministic");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 2);
          auto weight = peek(1, 2);
          auto the_result = at::miopen_convolution_transpose_backward_input(grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_577() { // miopen_convolution_transpose_backward_weight
      auto weight_size = readIntArrayRef("weight_size");
      auto padding = readIntArrayRef("padding");
      auto stride = readIntArrayRef("stride");
      auto dilation = readIntArrayRef("dilation");
      int64_t groups = readAttribute<int64_t>("groups");
      bool benchmark = readAttribute<int64_t>("benchmark");
      bool deterministic = readAttribute<int64_t>("deterministic");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 2);
          auto self = peek(1, 2);
          auto the_result = at::miopen_convolution_transpose_backward_weight(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_578() { // miopen_depthwise_convolution
      auto padding = readIntArrayRef("padding");
      auto stride = readIntArrayRef("stride");
      auto dilation = readIntArrayRef("dilation");
      int64_t groups = readAttribute<int64_t>("groups");
      bool benchmark = readAttribute<int64_t>("benchmark");
      bool deterministic = readAttribute<int64_t>("deterministic");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::miopen_depthwise_convolution(self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_579() { // miopen_depthwise_convolution_backward_input
      auto self_size = readIntArrayRef("self_size");
      auto padding = readIntArrayRef("padding");
      auto stride = readIntArrayRef("stride");
      auto dilation = readIntArrayRef("dilation");
      int64_t groups = readAttribute<int64_t>("groups");
      bool benchmark = readAttribute<int64_t>("benchmark");
      bool deterministic = readAttribute<int64_t>("deterministic");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 2);
          auto weight = peek(1, 2);
          auto the_result = at::miopen_depthwise_convolution_backward_input(self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_580() { // miopen_depthwise_convolution_backward
      auto padding = readIntArrayRef("padding");
      auto stride = readIntArrayRef("stride");
      auto dilation = readIntArrayRef("dilation");
      int64_t groups = readAttribute<int64_t>("groups");
      bool benchmark = readAttribute<int64_t>("benchmark");
      bool deterministic = readAttribute<int64_t>("deterministic");
      auto output_mask = readBoolMask<3>("output_mask");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto grad_output = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::miopen_depthwise_convolution_backward(self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, output_mask);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_581() { // miopen_depthwise_convolution_backward_weight
      auto weight_size = readIntArrayRef("weight_size");
      auto padding = readIntArrayRef("padding");
      auto stride = readIntArrayRef("stride");
      auto dilation = readIntArrayRef("dilation");
      int64_t groups = readAttribute<int64_t>("groups");
      bool benchmark = readAttribute<int64_t>("benchmark");
      bool deterministic = readAttribute<int64_t>("deterministic");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 2);
          auto self = peek(1, 2);
          auto the_result = at::miopen_depthwise_convolution_backward_weight(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_582() { // miopen_rnn
      int64_t weight_stride0 = readAttribute<int64_t>("weight_stride0");
      int64_t mode = readAttribute<int64_t>("mode");
      int64_t hidden_size = readAttribute<int64_t>("hidden_size");
      int64_t num_layers = readAttribute<int64_t>("num_layers");
      bool batch_first = readAttribute<int64_t>("batch_first");
      double dropout = readAttribute<float>("dropout");
      bool train = readAttribute<int64_t>("train");
      bool bidirectional = readAttribute<int64_t>("bidirectional");
      auto batch_sizes = readIntArrayRef("batch_sizes");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, InputSize());
          auto weight = peekSlice(1, InputSize() - 4, InputSize());
          auto hx = peek(1, 4);
          auto cx = peek(2, 4);
          auto dropout_state = peek(3, 4);
          auto the_result = at::miopen_rnn(input, weight, weight_stride0, hx, cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
            if(OutputSize() > 3) {assignTo(Output(3),::std::get<3>(the_result));}
            if(OutputSize() > 4) {assignTo(Output(4),::std::get<4>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_583() { // mm
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto mat2 = peek(1, 2);
          auto the_result = at::mm(self, mat2);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_584() { // _sparse_mm
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto sparse = peek(0, 2);
          auto dense = peek(1, 2);
          auto the_result = at::_sparse_mm(sparse, dense);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_585() { // _sparse_sparse_matmul
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::_sparse_sparse_matmul(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_586() { // _sparse_mask_helper
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto t = peek(0, 2);
          auto mask_indices = peek(1, 2);
          auto the_result = at::_sparse_mask_helper(t, mask_indices);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_587() { // mode
      int64_t dim = readAttribute<int64_t>("dim");
      bool keepdim = readAttribute<int64_t>("keepdim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::mode(self, dim, keepdim);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_588() { // mode
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::mode(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_589() { // mode
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::mode(self);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_590() { // mul
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::mul(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_591() { // mul
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::mul(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_592() { // multiply
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::multiply(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_593() { // multiply
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::multiply(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_594() { // mv
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto vec = peek(1, 2);
          auto the_result = at::mv(self, vec);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_595() { // mvlgamma
      int64_t p = readAttribute<int64_t>("p");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::mvlgamma(self, p);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_596() { // narrow_copy
      int64_t dim = readAttribute<int64_t>("dim");
      int64_t start = readAttribute<int64_t>("start");
      int64_t length = readAttribute<int64_t>("length");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::narrow_copy(self, dim, start, length);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_597() { // narrow
      int64_t dim = readAttribute<int64_t>("dim");
      int64_t start = readAttribute<int64_t>("start");
      int64_t length = readAttribute<int64_t>("length");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::narrow(self, dim, start, length);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_598() { // narrow
      int64_t dim = readAttribute<int64_t>("dim");
      int64_t length = readAttribute<int64_t>("length");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto start = peek(1, 2);
          auto the_result = at::narrow(self, dim, start, length);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_599() { // native_batch_norm
      bool training = readAttribute<int64_t>("training");
      double momentum = readAttribute<float>("momentum");
      double eps = readAttribute<float>("eps");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 5);
          auto weight = peek(1, 5);
          auto bias = peek(2, 5);
          auto running_mean = peek(3, 5);
          auto running_var = peek(4, 5);
          auto the_result = at::native_batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_600() { // batch_norm_stats
      double eps = readAttribute<float>("eps");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 1);
          auto the_result = at::batch_norm_stats(input, eps);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_601() { // batch_norm_elemt
      double eps = readAttribute<float>("eps");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 5);
          auto weight = peek(1, 5);
          auto bias = peek(2, 5);
          auto mean = peek(3, 5);
          auto invstd = peek(4, 5);
          auto the_result = at::batch_norm_elemt(input, weight, bias, mean, invstd, eps);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_602() { // batch_norm_gather_stats
      double momentum = readAttribute<float>("momentum");
      double eps = readAttribute<float>("eps");
      int64_t count = readAttribute<int64_t>("count");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 5);
          auto mean = peek(1, 5);
          auto invstd = peek(2, 5);
          auto running_mean = peek(3, 5);
          auto running_var = peek(4, 5);
          auto the_result = at::batch_norm_gather_stats(input, mean, invstd, running_mean, running_var, momentum, eps, count);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_603() { // batch_norm_gather_stats_with_counts
      double momentum = readAttribute<float>("momentum");
      double eps = readAttribute<float>("eps");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 6);
          auto mean = peek(1, 6);
          auto invstd = peek(2, 6);
          auto running_mean = peek(3, 6);
          auto running_var = peek(4, 6);
          auto counts = peek(5, 6);
          auto the_result = at::batch_norm_gather_stats_with_counts(input, mean, invstd, running_mean, running_var, momentum, eps, counts);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_604() { // native_batch_norm_backward
      bool train = readAttribute<int64_t>("train");
      double eps = readAttribute<float>("eps");
      auto output_mask = readBoolMask<3>("output_mask");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_out = peek(0, 7);
          auto input = peek(1, 7);
          auto weight = peek(2, 7);
          auto running_mean = peek(3, 7);
          auto running_var = peek(4, 7);
          auto save_mean = peek(5, 7);
          auto save_invstd = peek(6, 7);
          auto the_result = at::native_batch_norm_backward(grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_605() { // batch_norm_backward_reduce
      bool input_g = readAttribute<int64_t>("input_g");
      bool weight_g = readAttribute<int64_t>("weight_g");
      bool bias_g = readAttribute<int64_t>("bias_g");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_out = peek(0, 5);
          auto input = peek(1, 5);
          auto mean = peek(2, 5);
          auto invstd = peek(3, 5);
          auto weight = peek(4, 5);
          auto the_result = at::batch_norm_backward_reduce(grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
            if(OutputSize() > 3) {assignTo(Output(3),::std::get<3>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_606() { // batch_norm_backward_elemt
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_out = peek(0, 8);
          auto input = peek(1, 8);
          auto mean = peek(2, 8);
          auto invstd = peek(3, 8);
          auto weight = peek(4, 8);
          auto mean_dy = peek(5, 8);
          auto mean_dy_xmu = peek(6, 8);
          auto count = peek(7, 8);
          auto the_result = at::batch_norm_backward_elemt(grad_out, input, mean, invstd, weight, mean_dy, mean_dy_xmu, count);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_607() { // batch_norm_update_stats
      double momentum = readAttribute<float>("momentum");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto running_mean = peek(1, 3);
          auto running_var = peek(2, 3);
          auto the_result = at::batch_norm_update_stats(input, running_mean, running_var, momentum);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_608() { // is_vulkan_available
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
  
          auto the_result = at::is_vulkan_available();
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_609() { // _nnpack_available
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
  
          auto the_result = at::_nnpack_available();
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_610() { // _nnpack_spatial_convolution
      auto padding = readIntArrayRef("padding");
      auto stride = readIntArrayRef("stride");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::_nnpack_spatial_convolution(input, weight, bias, padding, stride);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_611() { // _nnpack_spatial_convolution
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::_nnpack_spatial_convolution(input, weight, bias, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_612() { // _nnpack_spatial_convolution_backward
      auto padding = readIntArrayRef("padding");
      auto output_mask = readBoolMask<3>("output_mask");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto grad_output = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::_nnpack_spatial_convolution_backward(input, grad_output, weight, padding, output_mask);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_613() { // _nnpack_spatial_convolution_backward_input
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 3);
          auto grad_output = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::_nnpack_spatial_convolution_backward_input(input, grad_output, weight, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_614() { // _nnpack_spatial_convolution_backward_weight
      auto weightsize = readIntArrayRef("weightsize");
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 2);
          auto grad_output = peek(1, 2);
          auto the_result = at::_nnpack_spatial_convolution_backward_weight(input, weightsize, grad_output, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_615() { // pairwise_distance
      double p = readAttribute<float>("p");
      double eps = readAttribute<float>("eps");
      bool keepdim = readAttribute<int64_t>("keepdim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto x1 = peek(0, 2);
          auto x2 = peek(1, 2);
          auto the_result = at::pairwise_distance(x1, x2, p, eps, keepdim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_616() { // pairwise_distance
      double p = readAttribute<float>("p");
      double eps = readAttribute<float>("eps");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto x1 = peek(0, 2);
          auto x2 = peek(1, 2);
          auto the_result = at::pairwise_distance(x1, x2, p, eps);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_617() { // pairwise_distance
      double p = readAttribute<float>("p");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto x1 = peek(0, 2);
          auto x2 = peek(1, 2);
          auto the_result = at::pairwise_distance(x1, x2, p);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_618() { // pairwise_distance
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto x1 = peek(0, 2);
          auto x2 = peek(1, 2);
          auto the_result = at::pairwise_distance(x1, x2);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_619() { // cdist
      double p = readAttribute<float>("p");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto x1 = peek(0, 2);
          auto x2 = peek(1, 2);
          auto the_result = at::cdist(x1, x2, p);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_620() { // cdist
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto x1 = peek(0, 2);
          auto x2 = peek(1, 2);
          auto the_result = at::cdist(x1, x2);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_621() { // _euclidean_dist
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto x1 = peek(0, 2);
          auto x2 = peek(1, 2);
          auto the_result = at::_euclidean_dist(x1, x2);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_622() { // _cdist_backward
      double p = readAttribute<float>("p");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad = peek(0, 4);
          auto x1 = peek(1, 4);
          auto x2 = peek(2, 4);
          auto cdist = peek(3, 4);
          auto the_result = at::_cdist_backward(grad, x1, x2, p, cdist);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_623() { // pdist
      double p = readAttribute<float>("p");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::pdist(self, p);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_624() { // pdist
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::pdist(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_625() { // _pdist_forward
      double p = readAttribute<float>("p");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_pdist_forward(self, p);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_626() { // _pdist_forward
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_pdist_forward(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_627() { // _pdist_backward
      double p = readAttribute<float>("p");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad = peek(0, 3);
          auto self = peek(1, 3);
          auto pdist = peek(2, 3);
          auto the_result = at::_pdist_backward(grad, self, p, pdist);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_628() { // cosine_similarity
      int64_t dim = readAttribute<int64_t>("dim");
      double eps = readAttribute<float>("eps");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto x1 = peek(0, 2);
          auto x2 = peek(1, 2);
          auto the_result = at::cosine_similarity(x1, x2, dim, eps);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_629() { // cosine_similarity
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto x1 = peek(0, 2);
          auto x2 = peek(1, 2);
          auto the_result = at::cosine_similarity(x1, x2, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_630() { // cosine_similarity
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto x1 = peek(0, 2);
          auto x2 = peek(1, 2);
          auto the_result = at::cosine_similarity(x1, x2);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_631() { // permute
      auto dims = readIntArrayRef("dims");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::permute(self, dims);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_632() { // movedim
      auto source = readIntArrayRef("source");
      auto destination = readIntArrayRef("destination");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::movedim(self, source, destination);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_633() { // moveaxis
      auto source = readIntArrayRef("source");
      auto destination = readIntArrayRef("destination");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::moveaxis(self, source, destination);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_634() { // numpy_T
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = self.numpy_T();
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_635() { // pixel_shuffle
      int64_t upscale_factor = readAttribute<int64_t>("upscale_factor");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::pixel_shuffle(self, upscale_factor);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_636() { // pixel_unshuffle
      int64_t downscale_factor = readAttribute<int64_t>("downscale_factor");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::pixel_unshuffle(self, downscale_factor);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_637() { // channel_shuffle
      int64_t groups = readAttribute<int64_t>("groups");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::channel_shuffle(self, groups);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_638() { // is_pinned
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = self.is_pinned();
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_639() { // pin_memory
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = self.pin_memory();
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_640() { // _pin_memory
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_pin_memory(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_641() { // pinverse
      double rcond = readAttribute<float>("rcond");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::pinverse(self, rcond);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_642() { // pinverse
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::pinverse(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_643() { // poisson_nll_loss
      bool log_input = readAttribute<int64_t>("log_input");
      bool full = readAttribute<int64_t>("full");
      double eps = readAttribute<float>("eps");
      int64_t reduction = readAttribute<int64_t>("reduction");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 2);
          auto target = peek(1, 2);
          auto the_result = at::poisson_nll_loss(input, target, log_input, full, eps, reduction);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_644() { // rad2deg
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::rad2deg(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_645() { // deg2rad
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::deg2rad(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_646() { // ravel
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::ravel(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_647() { // reciprocal
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::reciprocal(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_648() { // neg
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::neg(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_649() { // negative
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::negative(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_650() { // repeat
      auto repeats = readIntArrayRef("repeats");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = self.repeat(repeats);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_651() { // repeat_interleave
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto repeats = peek(0, 1);
          auto the_result = at::repeat_interleave(repeats);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_652() { // repeat_interleave
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto repeats = peek(1, 2);
          auto the_result = at::repeat_interleave(self, repeats);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_653() { // repeat_interleave
      int64_t repeats = readAttribute<int64_t>("repeats");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::repeat_interleave(self, repeats);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_654() { // reshape
      auto shape = readIntArrayRef("shape");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::reshape(self, shape);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_655() { // _reshape_alias
      auto size = readIntArrayRef("size");
      auto stride = readIntArrayRef("stride");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_reshape_alias(self, size, stride);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_656() { // _mkldnn_reshape
      auto shape = readIntArrayRef("shape");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_mkldnn_reshape(self, shape);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_657() { // reshape_as
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = self.reshape_as(other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_658() { // round
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::round(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_659() { // rrelu
      at::Scalar lower = readScalarAttribute("lower");
      at::Scalar upper = readScalarAttribute("upper");
      bool training = readAttribute<int64_t>("training");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::rrelu(self, lower, upper, training);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_660() { // rrelu
      at::Scalar lower = readScalarAttribute("lower");
      at::Scalar upper = readScalarAttribute("upper");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::rrelu(self, lower, upper);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_661() { // rrelu
      at::Scalar lower = readScalarAttribute("lower");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::rrelu(self, lower);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_662() { // rrelu
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::rrelu(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_663() { // relu
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::relu(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_664() { // relu6
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::relu6(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_665() { // prelu
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto weight = peek(1, 2);
          auto the_result = at::prelu(self, weight);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_666() { // prelu_backward
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto self = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::prelu_backward(grad_output, self, weight);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_667() { // gelu
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::gelu(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_668() { // gelu_backward
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad = peek(0, 2);
          auto self = peek(1, 2);
          auto the_result = at::gelu_backward(grad, self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_669() { // infinitely_differentiable_gelu_backward
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad = peek(0, 2);
          auto self = peek(1, 2);
          auto the_result = at::infinitely_differentiable_gelu_backward(grad, self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_670() { // hardshrink
      at::Scalar lambd = readScalarAttribute("lambd");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::hardshrink(self, lambd);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_671() { // hardshrink
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::hardshrink(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_672() { // hardshrink_backward
      at::Scalar lambd = readScalarAttribute("lambd");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_out = peek(0, 2);
          auto self = peek(1, 2);
          auto the_result = at::hardshrink_backward(grad_out, self, lambd);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_673() { // rsqrt
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::rsqrt(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_674() { // select
      int64_t dim = readAttribute<int64_t>("dim");
      int64_t index = readAttribute<int64_t>("index");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::select(self, dim, index);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_675() { // select_backward
      auto input_sizes = readIntArrayRef("input_sizes");
      int64_t dim = readAttribute<int64_t>("dim");
      int64_t index = readAttribute<int64_t>("index");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 1);
          auto the_result = at::select_backward(grad_output, input_sizes, dim, index);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_676() { // selu
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::selu(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_677() { // celu
      at::Scalar alpha = readScalarAttribute("alpha");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::celu(self, alpha);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_678() { // celu
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::celu(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_679() { // silu
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::silu(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_680() { // silu_backward
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 2);
          auto self = peek(1, 2);
          auto the_result = at::silu_backward(grad_output, self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_681() { // mish
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::mish(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_682() { // mish_backward
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 2);
          auto self = peek(1, 2);
          auto the_result = at::mish_backward(grad_output, self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_683() { // sigmoid
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::sigmoid(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_684() { // logit
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::logit(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_685() { // sin
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::sin(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_686() { // sinc
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::sinc(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_687() { // sinh
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::sinh(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_688() { // detach
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::detach(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_689() { // size
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::size(self, dim);
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_690() { // slice
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::slice(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_691() { // slice
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::slice(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_692() { // slice_backward
      auto input_sizes = readIntArrayRef("input_sizes");
      int64_t dim = readAttribute<int64_t>("dim");
      int64_t start = readAttribute<int64_t>("start");
      int64_t end = readAttribute<int64_t>("end");
      int64_t step = readAttribute<int64_t>("step");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 1);
          auto the_result = at::slice_backward(grad_output, input_sizes, dim, start, end, step);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_693() { // slogdet
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::slogdet(self);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_694() { // smm
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto mat2 = peek(1, 2);
          auto the_result = at::smm(self, mat2);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_695() { // softmax
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::softmax(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_696() { // _softmax
      int64_t dim = readAttribute<int64_t>("dim");
      bool half_to_float = readAttribute<int64_t>("half_to_float");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_softmax(self, dim, half_to_float);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_697() { // _softmax_backward_data
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto output = peek(1, 3);
          auto self = peek(2, 3);
          auto the_result = at::_softmax_backward_data(grad_output, output, dim, self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_698() { // unsafe_split
      int64_t split_size = readAttribute<int64_t>("split_size");
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::unsafe_split(self, split_size, dim);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_699() { // unsafe_split
      int64_t split_size = readAttribute<int64_t>("split_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::unsafe_split(self, split_size);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_700() { // split
      int64_t split_size = readAttribute<int64_t>("split_size");
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::split(self, split_size, dim);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_701() { // split
      int64_t split_size = readAttribute<int64_t>("split_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::split(self, split_size);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_702() { // unsafe_split_with_sizes
      auto split_sizes = readIntArrayRef("split_sizes");
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::unsafe_split_with_sizes(self, split_sizes, dim);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_703() { // unsafe_split_with_sizes
      auto split_sizes = readIntArrayRef("split_sizes");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::unsafe_split_with_sizes(self, split_sizes);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_704() { // split_with_sizes
      auto split_sizes = readIntArrayRef("split_sizes");
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::split_with_sizes(self, split_sizes, dim);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_705() { // split_with_sizes
      auto split_sizes = readIntArrayRef("split_sizes");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::split_with_sizes(self, split_sizes);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_706() { // hsplit
      int64_t sections = readAttribute<int64_t>("sections");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::hsplit(self, sections);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_707() { // hsplit
      auto indices = readIntArrayRef("indices");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::hsplit(self, indices);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_708() { // vsplit
      int64_t sections = readAttribute<int64_t>("sections");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::vsplit(self, sections);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_709() { // vsplit
      auto indices = readIntArrayRef("indices");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::vsplit(self, indices);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_710() { // dsplit
      int64_t sections = readAttribute<int64_t>("sections");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::dsplit(self, sections);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_711() { // dsplit
      auto indices = readIntArrayRef("indices");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::dsplit(self, indices);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_712() { // squeeze
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::squeeze(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_713() { // squeeze
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::squeeze(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_714() { // sspaddmm
      at::Scalar beta = readScalarAttribute("beta");
      at::Scalar alpha = readScalarAttribute("alpha");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto mat1 = peek(1, 3);
          auto mat2 = peek(2, 3);
          auto the_result = at::sspaddmm(self, mat1, mat2, beta, alpha);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_715() { // sspaddmm
      at::Scalar beta = readScalarAttribute("beta");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto mat1 = peek(1, 3);
          auto mat2 = peek(2, 3);
          auto the_result = at::sspaddmm(self, mat1, mat2, beta);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_716() { // sspaddmm
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto mat1 = peek(1, 3);
          auto mat2 = peek(2, 3);
          auto the_result = at::sspaddmm(self, mat1, mat2);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_717() { // stack
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::stack(tensors, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_718() { // stack
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::stack(tensors);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_719() { // _stack
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_stack(tensors, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_720() { // _stack
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_stack(tensors);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_721() { // hstack
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::hstack(tensors);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_722() { // vstack
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::vstack(tensors);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_723() { // dstack
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::dstack(tensors);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_724() { // stft
      int64_t n_fft = readAttribute<int64_t>("n_fft");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::stft(self, n_fft);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_725() { // istft
      int64_t n_fft = readAttribute<int64_t>("n_fft");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::istft(self, n_fft);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_726() { // stride
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::stride(self, dim);
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_727() { // sum
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::sum(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_728() { // sum
      auto dim = readIntArrayRef("dim");
      bool keepdim = readAttribute<int64_t>("keepdim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::sum(self, dim, keepdim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_729() { // sum
      auto dim = readIntArrayRef("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::sum(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_730() { // nansum
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::nansum(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_731() { // nansum
      auto dim = readIntArrayRef("dim");
      bool keepdim = readAttribute<int64_t>("keepdim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::nansum(self, dim, keepdim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_732() { // nansum
      auto dim = readIntArrayRef("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::nansum(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_733() { // sum_to_size
      auto size = readIntArrayRef("size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = self.sum_to_size(size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_734() { // sqrt
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::sqrt(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_735() { // square
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::square(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_736() { // std
      bool unbiased = readAttribute<int64_t>("unbiased");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::std(self, unbiased);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_737() { // std
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::std(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_738() { // std
      auto dim = readIntArrayRef("dim");
      bool unbiased = readAttribute<int64_t>("unbiased");
      bool keepdim = readAttribute<int64_t>("keepdim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::std(self, dim, unbiased, keepdim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_739() { // std
      auto dim = readIntArrayRef("dim");
      bool unbiased = readAttribute<int64_t>("unbiased");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::std(self, dim, unbiased);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_740() { // std
      auto dim = readIntArrayRef("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::std(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_741() { // std_mean
      bool unbiased = readAttribute<int64_t>("unbiased");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::std_mean(self, unbiased);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_742() { // std_mean
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::std_mean(self);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_743() { // std_mean
      auto dim = readIntArrayRef("dim");
      bool unbiased = readAttribute<int64_t>("unbiased");
      bool keepdim = readAttribute<int64_t>("keepdim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::std_mean(self, dim, unbiased, keepdim);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_744() { // std_mean
      auto dim = readIntArrayRef("dim");
      bool unbiased = readAttribute<int64_t>("unbiased");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::std_mean(self, dim, unbiased);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_745() { // std_mean
      auto dim = readIntArrayRef("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::std_mean(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_746() { // prod
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::prod(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_747() { // prod
      int64_t dim = readAttribute<int64_t>("dim");
      bool keepdim = readAttribute<int64_t>("keepdim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::prod(self, dim, keepdim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_748() { // prod
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::prod(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_749() { // t
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::t(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_750() { // tan
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::tan(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_751() { // tanh
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::tanh(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_752() { // tensordot
      auto dims_self = readIntArrayRef("dims_self");
      auto dims_other = readIntArrayRef("dims_other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::tensordot(self, other, dims_self, dims_other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_753() { // threshold
      at::Scalar threshold = readScalarAttribute("threshold");
      at::Scalar value = readScalarAttribute("value");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::threshold(self, threshold, value);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_754() { // threshold_backward
      at::Scalar threshold = readScalarAttribute("threshold");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 2);
          auto self = peek(1, 2);
          auto the_result = at::threshold_backward(grad_output, self, threshold);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_755() { // tile
      auto dims = readIntArrayRef("dims");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::tile(self, dims);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_756() { // transpose
      int64_t dim0 = readAttribute<int64_t>("dim0");
      int64_t dim1 = readAttribute<int64_t>("dim1");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::transpose(self, dim0, dim1);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_757() { // _mkldnn_transpose
      int64_t dim0 = readAttribute<int64_t>("dim0");
      int64_t dim1 = readAttribute<int64_t>("dim1");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_mkldnn_transpose(self, dim0, dim1);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_758() { // one_hot
      int64_t num_classes = readAttribute<int64_t>("num_classes");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::one_hot(self, num_classes);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_759() { // one_hot
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::one_hot(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_760() { // flip
      auto dims = readIntArrayRef("dims");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::flip(self, dims);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_761() { // fliplr
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::fliplr(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_762() { // flipud
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::flipud(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_763() { // roll
      auto shifts = readIntArrayRef("shifts");
      auto dims = readIntArrayRef("dims");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::roll(self, shifts, dims);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_764() { // roll
      auto shifts = readIntArrayRef("shifts");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::roll(self, shifts);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_765() { // rot90
      int64_t k = readAttribute<int64_t>("k");
      auto dims = readIntArrayRef("dims");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::rot90(self, k, dims);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_766() { // rot90
      int64_t k = readAttribute<int64_t>("k");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::rot90(self, k);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_767() { // rot90
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::rot90(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_768() { // trapezoid
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto y = peek(0, 2);
          auto x = peek(1, 2);
          auto the_result = at::trapezoid(y, x, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_769() { // trapezoid
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto y = peek(0, 2);
          auto x = peek(1, 2);
          auto the_result = at::trapezoid(y, x);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_770() { // trapezoid
      at::Scalar dx = readScalarAttribute("dx");
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto y = peek(0, 1);
          auto the_result = at::trapezoid(y, dx, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_771() { // trapezoid
      at::Scalar dx = readScalarAttribute("dx");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto y = peek(0, 1);
          auto the_result = at::trapezoid(y, dx);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_772() { // trapezoid
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto y = peek(0, 1);
          auto the_result = at::trapezoid(y);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_773() { // trapz
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto y = peek(0, 2);
          auto x = peek(1, 2);
          auto the_result = at::trapz(y, x, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_774() { // trapz
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto y = peek(0, 2);
          auto x = peek(1, 2);
          auto the_result = at::trapz(y, x);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_775() { // trapz
      double dx = readAttribute<float>("dx");
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto y = peek(0, 1);
          auto the_result = at::trapz(y, dx, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_776() { // trapz
      double dx = readAttribute<float>("dx");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto y = peek(0, 1);
          auto the_result = at::trapz(y, dx);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_777() { // trapz
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto y = peek(0, 1);
          auto the_result = at::trapz(y);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_778() { // _trilinear
      auto expand1 = readIntArrayRef("expand1");
      auto expand2 = readIntArrayRef("expand2");
      auto expand3 = readIntArrayRef("expand3");
      auto sumdim = readIntArrayRef("sumdim");
      int64_t unroll_dim = readAttribute<int64_t>("unroll_dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto i1 = peek(0, 3);
          auto i2 = peek(1, 3);
          auto i3 = peek(2, 3);
          auto the_result = at::_trilinear(i1, i2, i3, expand1, expand2, expand3, sumdim, unroll_dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_779() { // _trilinear
      auto expand1 = readIntArrayRef("expand1");
      auto expand2 = readIntArrayRef("expand2");
      auto expand3 = readIntArrayRef("expand3");
      auto sumdim = readIntArrayRef("sumdim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto i1 = peek(0, 3);
          auto i2 = peek(1, 3);
          auto i3 = peek(2, 3);
          auto the_result = at::_trilinear(i1, i2, i3, expand1, expand2, expand3, sumdim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_780() { // triplet_margin_loss
      double margin = readAttribute<float>("margin");
      double p = readAttribute<float>("p");
      double eps = readAttribute<float>("eps");
      bool swap = readAttribute<int64_t>("swap");
      int64_t reduction = readAttribute<int64_t>("reduction");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto anchor = peek(0, 3);
          auto positive = peek(1, 3);
          auto negative = peek(2, 3);
          auto the_result = at::triplet_margin_loss(anchor, positive, negative, margin, p, eps, swap, reduction);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_781() { // triplet_margin_loss
      double margin = readAttribute<float>("margin");
      double p = readAttribute<float>("p");
      double eps = readAttribute<float>("eps");
      bool swap = readAttribute<int64_t>("swap");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto anchor = peek(0, 3);
          auto positive = peek(1, 3);
          auto negative = peek(2, 3);
          auto the_result = at::triplet_margin_loss(anchor, positive, negative, margin, p, eps, swap);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_782() { // triplet_margin_loss
      double margin = readAttribute<float>("margin");
      double p = readAttribute<float>("p");
      double eps = readAttribute<float>("eps");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto anchor = peek(0, 3);
          auto positive = peek(1, 3);
          auto negative = peek(2, 3);
          auto the_result = at::triplet_margin_loss(anchor, positive, negative, margin, p, eps);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_783() { // triplet_margin_loss
      double margin = readAttribute<float>("margin");
      double p = readAttribute<float>("p");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto anchor = peek(0, 3);
          auto positive = peek(1, 3);
          auto negative = peek(2, 3);
          auto the_result = at::triplet_margin_loss(anchor, positive, negative, margin, p);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_784() { // triplet_margin_loss
      double margin = readAttribute<float>("margin");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto anchor = peek(0, 3);
          auto positive = peek(1, 3);
          auto negative = peek(2, 3);
          auto the_result = at::triplet_margin_loss(anchor, positive, negative, margin);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_785() { // triplet_margin_loss
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto anchor = peek(0, 3);
          auto positive = peek(1, 3);
          auto negative = peek(2, 3);
          auto the_result = at::triplet_margin_loss(anchor, positive, negative);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_786() { // trunc
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::trunc(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_787() { // fix
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::fix(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_788() { // type_as
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = self.type_as(other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_789() { // _has_compatible_shallow_copy_type
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto from = peek(1, 2);
          auto the_result = at::_has_compatible_shallow_copy_type(self, from);
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_790() { // _unique
      bool sorted = readAttribute<int64_t>("sorted");
      bool return_inverse = readAttribute<int64_t>("return_inverse");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_unique(self, sorted, return_inverse);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_791() { // _unique
      bool sorted = readAttribute<int64_t>("sorted");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_unique(self, sorted);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_792() { // _unique
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_unique(self);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_793() { // unique_dim
      int64_t dim = readAttribute<int64_t>("dim");
      bool sorted = readAttribute<int64_t>("sorted");
      bool return_inverse = readAttribute<int64_t>("return_inverse");
      bool return_counts = readAttribute<int64_t>("return_counts");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::unique_dim(self, dim, sorted, return_inverse, return_counts);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_794() { // unique_dim
      int64_t dim = readAttribute<int64_t>("dim");
      bool sorted = readAttribute<int64_t>("sorted");
      bool return_inverse = readAttribute<int64_t>("return_inverse");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::unique_dim(self, dim, sorted, return_inverse);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_795() { // unique_dim
      int64_t dim = readAttribute<int64_t>("dim");
      bool sorted = readAttribute<int64_t>("sorted");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::unique_dim(self, dim, sorted);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_796() { // unique_dim
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::unique_dim(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_797() { // unique_consecutive
      bool return_inverse = readAttribute<int64_t>("return_inverse");
      bool return_counts = readAttribute<int64_t>("return_counts");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::unique_consecutive(self, return_inverse, return_counts);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_798() { // unique_consecutive
      bool return_inverse = readAttribute<int64_t>("return_inverse");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::unique_consecutive(self, return_inverse);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_799() { // unique_consecutive
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::unique_consecutive(self);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_800() { // unique_dim_consecutive
      int64_t dim = readAttribute<int64_t>("dim");
      bool return_inverse = readAttribute<int64_t>("return_inverse");
      bool return_counts = readAttribute<int64_t>("return_counts");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::unique_dim_consecutive(self, dim, return_inverse, return_counts);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_801() { // unique_dim_consecutive
      int64_t dim = readAttribute<int64_t>("dim");
      bool return_inverse = readAttribute<int64_t>("return_inverse");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::unique_dim_consecutive(self, dim, return_inverse);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_802() { // unique_dim_consecutive
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::unique_dim_consecutive(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_803() { // _unique2
      bool sorted = readAttribute<int64_t>("sorted");
      bool return_inverse = readAttribute<int64_t>("return_inverse");
      bool return_counts = readAttribute<int64_t>("return_counts");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_unique2(self, sorted, return_inverse, return_counts);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_804() { // _unique2
      bool sorted = readAttribute<int64_t>("sorted");
      bool return_inverse = readAttribute<int64_t>("return_inverse");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_unique2(self, sorted, return_inverse);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_805() { // _unique2
      bool sorted = readAttribute<int64_t>("sorted");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_unique2(self, sorted);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_806() { // _unique2
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_unique2(self);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_807() { // _unsafe_view
      auto size = readIntArrayRef("size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_unsafe_view(self, size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_808() { // unsqueeze
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::unsqueeze(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_809() { // vander
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto x = peek(0, 1);
          auto the_result = at::vander(x);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_810() { // var
      bool unbiased = readAttribute<int64_t>("unbiased");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::var(self, unbiased);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_811() { // var
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::var(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_812() { // var
      auto dim = readIntArrayRef("dim");
      bool unbiased = readAttribute<int64_t>("unbiased");
      bool keepdim = readAttribute<int64_t>("keepdim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::var(self, dim, unbiased, keepdim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_813() { // var
      auto dim = readIntArrayRef("dim");
      bool unbiased = readAttribute<int64_t>("unbiased");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::var(self, dim, unbiased);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_814() { // var
      auto dim = readIntArrayRef("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::var(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_815() { // var_mean
      bool unbiased = readAttribute<int64_t>("unbiased");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::var_mean(self, unbiased);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_816() { // var_mean
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::var_mean(self);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_817() { // var_mean
      auto dim = readIntArrayRef("dim");
      bool unbiased = readAttribute<int64_t>("unbiased");
      bool keepdim = readAttribute<int64_t>("keepdim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::var_mean(self, dim, unbiased, keepdim);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_818() { // var_mean
      auto dim = readIntArrayRef("dim");
      bool unbiased = readAttribute<int64_t>("unbiased");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::var_mean(self, dim, unbiased);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_819() { // var_mean
      auto dim = readIntArrayRef("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::var_mean(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_820() { // view_as
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = self.view_as(other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_821() { // where
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto condition = peek(0, 3);
          auto self = peek(1, 3);
          auto other = peek(2, 3);
          auto the_result = at::where(condition, self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_822() { // where
      at::Scalar self = readScalarAttribute("self");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto condition = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::where(condition, self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_823() { // where
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto condition = peek(0, 2);
          auto self = peek(1, 2);
          auto the_result = at::where(condition, self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_824() { // where
      at::Scalar self = readScalarAttribute("self");
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto condition = peek(0, 1);
          auto the_result = at::where(condition, self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_825() { // where
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto condition = peek(0, 1);
          auto the_result = at::where(condition);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_826() { // _s_where
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto condition = peek(0, 3);
          auto self = peek(1, 3);
          auto other = peek(2, 3);
          auto the_result = at::_s_where(condition, self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_827() { // norm_except_dim
      int64_t pow = readAttribute<int64_t>("pow");
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto v = peek(0, 1);
          auto the_result = at::norm_except_dim(v, pow, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_828() { // norm_except_dim
      int64_t pow = readAttribute<int64_t>("pow");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto v = peek(0, 1);
          auto the_result = at::norm_except_dim(v, pow);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_829() { // norm_except_dim
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto v = peek(0, 1);
          auto the_result = at::norm_except_dim(v);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_830() { // _weight_norm
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto v = peek(0, 2);
          auto g = peek(1, 2);
          auto the_result = at::_weight_norm(v, g, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_831() { // _weight_norm
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto v = peek(0, 2);
          auto g = peek(1, 2);
          auto the_result = at::_weight_norm(v, g);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_832() { // _weight_norm_cuda_interface
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto v = peek(0, 2);
          auto g = peek(1, 2);
          auto the_result = at::_weight_norm_cuda_interface(v, g, dim);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_833() { // _weight_norm_cuda_interface
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto v = peek(0, 2);
          auto g = peek(1, 2);
          auto the_result = at::_weight_norm_cuda_interface(v, g);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_834() { // _weight_norm_cuda_interface_backward
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_w = peek(0, 4);
          auto saved_v = peek(1, 4);
          auto saved_g = peek(2, 4);
          auto saved_norms = peek(3, 4);
          auto the_result = at::_weight_norm_cuda_interface_backward(grad_w, saved_v, saved_g, saved_norms, dim);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_835() { // _weight_norm_differentiable_backward
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_w = peek(0, 4);
          auto saved_v = peek(1, 4);
          auto saved_g = peek(2, 4);
          auto saved_norms = peek(3, 4);
          auto the_result = at::_weight_norm_differentiable_backward(grad_w, saved_v, saved_g, saved_norms, dim);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_836() { // _standard_gamma_grad
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto output = peek(1, 2);
          auto the_result = at::_standard_gamma_grad(self, output);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_837() { // _standard_gamma
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_standard_gamma(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_838() { // _dirichlet_grad
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto x = peek(0, 3);
          auto alpha = peek(1, 3);
          auto total = peek(2, 3);
          auto the_result = at::_dirichlet_grad(x, alpha, total);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_839() { // _sample_dirichlet
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_sample_dirichlet(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_840() { // poisson
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::poisson(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_841() { // binomial
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto count = peek(0, 2);
          auto prob = peek(1, 2);
          auto the_result = at::binomial(count, prob);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_842() { // native_norm
      at::Scalar p = readScalarAttribute("p");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::native_norm(self, p);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_843() { // native_norm
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::native_norm(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_844() { // _sparse_sum
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_sparse_sum(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_845() { // _sparse_sum
      auto dim = readIntArrayRef("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_sparse_sum(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_846() { // _sparse_sum_backward
      auto dim = readIntArrayRef("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad = peek(0, 2);
          auto self = peek(1, 2);
          auto the_result = at::_sparse_sum_backward(grad, self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_847() { // _sparse_softmax
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_sparse_softmax(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_848() { // _sparse_softmax
      int64_t dim = readAttribute<int64_t>("dim");
      bool half_to_float = readAttribute<int64_t>("half_to_float");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_sparse_softmax(self, dim, half_to_float);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_849() { // _sparse_softmax_backward_data
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto output = peek(1, 3);
          auto self = peek(2, 3);
          auto the_result = at::_sparse_softmax_backward_data(grad_output, output, dim, self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_850() { // _sparse_log_softmax
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_sparse_log_softmax(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_851() { // _sparse_log_softmax
      int64_t dim = readAttribute<int64_t>("dim");
      bool half_to_float = readAttribute<int64_t>("half_to_float");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_sparse_log_softmax(self, dim, half_to_float);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_852() { // _sparse_log_softmax_backward_data
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto output = peek(1, 3);
          auto self = peek(2, 3);
          auto the_result = at::_sparse_log_softmax_backward_data(grad_output, output, dim, self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_853() { // norm
      at::Scalar p = readScalarAttribute("p");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::norm(self, p);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_854() { // norm
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::norm(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_855() { // frexp
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::frexp(self);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_856() { // frobenius_norm
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::frobenius_norm(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_857() { // frobenius_norm
      auto dim = readIntArrayRef("dim");
      bool keepdim = readAttribute<int64_t>("keepdim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::frobenius_norm(self, dim, keepdim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_858() { // frobenius_norm
      auto dim = readIntArrayRef("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::frobenius_norm(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_859() { // nuclear_norm
      bool keepdim = readAttribute<int64_t>("keepdim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::nuclear_norm(self, keepdim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_860() { // nuclear_norm
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::nuclear_norm(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_861() { // nuclear_norm
      auto dim = readIntArrayRef("dim");
      bool keepdim = readAttribute<int64_t>("keepdim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::nuclear_norm(self, dim, keepdim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_862() { // nuclear_norm
      auto dim = readIntArrayRef("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::nuclear_norm(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_863() { // clone
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::clone(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_864() { // positive
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::positive(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_865() { // sub
      at::Scalar alpha = readScalarAttribute("alpha");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::sub(self, other, alpha);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_866() { // sub
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::sub(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_867() { // sub
      at::Scalar other = readScalarAttribute("other");
      at::Scalar alpha = readScalarAttribute("alpha");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::sub(self, other, alpha);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_868() { // sub
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::sub(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_869() { // subtract
      at::Scalar alpha = readScalarAttribute("alpha");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::subtract(self, other, alpha);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_870() { // subtract
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::subtract(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_871() { // subtract
      at::Scalar other = readScalarAttribute("other");
      at::Scalar alpha = readScalarAttribute("alpha");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::subtract(self, other, alpha);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_872() { // subtract
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::subtract(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_873() { // rsub
      at::Scalar alpha = readScalarAttribute("alpha");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::rsub(self, other, alpha);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_874() { // rsub
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::rsub(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_875() { // heaviside
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto values = peek(1, 2);
          auto the_result = at::heaviside(self, values);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_876() { // rsub
      at::Scalar other = readScalarAttribute("other");
      at::Scalar alpha = readScalarAttribute("alpha");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::rsub(self, other, alpha);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_877() { // rsub
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::rsub(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_878() { // _sparse_addmm
      at::Scalar beta = readScalarAttribute("beta");
      at::Scalar alpha = readScalarAttribute("alpha");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto sparse = peek(1, 3);
          auto dense = peek(2, 3);
          auto the_result = at::_sparse_addmm(self, sparse, dense, beta, alpha);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_879() { // _sparse_addmm
      at::Scalar beta = readScalarAttribute("beta");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto sparse = peek(1, 3);
          auto dense = peek(2, 3);
          auto the_result = at::_sparse_addmm(self, sparse, dense, beta);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_880() { // _sparse_addmm
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto sparse = peek(1, 3);
          auto dense = peek(2, 3);
          auto the_result = at::_sparse_addmm(self, sparse, dense);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_881() { // addmm
      at::Scalar beta = readScalarAttribute("beta");
      at::Scalar alpha = readScalarAttribute("alpha");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto mat1 = peek(1, 3);
          auto mat2 = peek(2, 3);
          auto the_result = at::addmm(self, mat1, mat2, beta, alpha);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_882() { // addmm
      at::Scalar beta = readScalarAttribute("beta");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto mat1 = peek(1, 3);
          auto mat2 = peek(2, 3);
          auto the_result = at::addmm(self, mat1, mat2, beta);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_883() { // addmm
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto mat1 = peek(1, 3);
          auto mat2 = peek(2, 3);
          auto the_result = at::addmm(self, mat1, mat2);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_884() { // sparse_mask
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto mask = peek(1, 2);
          auto the_result = self.sparse_mask(mask);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_885() { // _to_cpu
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_to_cpu(tensors);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_886() { // to_dense
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = self.to_dense();
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_887() { // to_dense_backward
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad = peek(0, 2);
          auto input = peek(1, 2);
          auto the_result = at::to_dense_backward(grad, input);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_888() { // sparse_dim
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = self.sparse_dim();
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_889() { // _dimI
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = self._dimI();
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_890() { // dense_dim
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = self.dense_dim();
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_891() { // _dimV
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = self._dimV();
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_892() { // _nnz
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = self._nnz();
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_893() { // coalesce
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = self.coalesce();
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_894() { // _coalesce
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_coalesce(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_895() { // is_coalesced
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = self.is_coalesced();
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_896() { // _indices
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = self._indices();
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_897() { // _values
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = self._values();
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_898() { // indices
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = self.indices();
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_899() { // values
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = self.values();
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_900() { // crow_indices
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = self.crow_indices();
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_901() { // col_indices
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = self.col_indices();
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_902() { // hspmm
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto mat1 = peek(0, 2);
          auto mat2 = peek(1, 2);
          auto the_result = at::hspmm(mat1, mat2);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_903() { // unbind
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::unbind(self, dim);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_904() { // unbind
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::unbind(self);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_905() { // to_sparse
      int64_t sparse_dim = readAttribute<int64_t>("sparse_dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = self.to_sparse(sparse_dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_906() { // to_sparse
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = self.to_sparse();
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_907() { // to_mkldnn
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = self.to_mkldnn();
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_908() { // mkldnn_reorder_conv2d_weight
      auto padding = readIntArrayRef("padding");
      auto stride = readIntArrayRef("stride");
      auto dilation = readIntArrayRef("dilation");
      int64_t groups = readAttribute<int64_t>("groups");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::mkldnn_reorder_conv2d_weight(self, padding, stride, dilation, groups);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_909() { // mkldnn_reorder_conv2d_weight
      auto padding = readIntArrayRef("padding");
      auto stride = readIntArrayRef("stride");
      auto dilation = readIntArrayRef("dilation");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::mkldnn_reorder_conv2d_weight(self, padding, stride, dilation);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_910() { // mkldnn_reorder_conv2d_weight
      auto padding = readIntArrayRef("padding");
      auto stride = readIntArrayRef("stride");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::mkldnn_reorder_conv2d_weight(self, padding, stride);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_911() { // mkldnn_reorder_conv2d_weight
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::mkldnn_reorder_conv2d_weight(self, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_912() { // mkldnn_reorder_conv2d_weight
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::mkldnn_reorder_conv2d_weight(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_913() { // mkldnn_reorder_conv3d_weight
      auto padding = readIntArrayRef("padding");
      auto stride = readIntArrayRef("stride");
      auto dilation = readIntArrayRef("dilation");
      int64_t groups = readAttribute<int64_t>("groups");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::mkldnn_reorder_conv3d_weight(self, padding, stride, dilation, groups);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_914() { // mkldnn_reorder_conv3d_weight
      auto padding = readIntArrayRef("padding");
      auto stride = readIntArrayRef("stride");
      auto dilation = readIntArrayRef("dilation");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::mkldnn_reorder_conv3d_weight(self, padding, stride, dilation);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_915() { // mkldnn_reorder_conv3d_weight
      auto padding = readIntArrayRef("padding");
      auto stride = readIntArrayRef("stride");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::mkldnn_reorder_conv3d_weight(self, padding, stride);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_916() { // mkldnn_reorder_conv3d_weight
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::mkldnn_reorder_conv3d_weight(self, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_917() { // mkldnn_reorder_conv3d_weight
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::mkldnn_reorder_conv3d_weight(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_918() { // to_mkldnn_backward
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad = peek(0, 2);
          auto input = peek(1, 2);
          auto the_result = at::to_mkldnn_backward(grad, input);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_919() { // dequantize
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::dequantize(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_920() { // dequantize
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::dequantize(tensors);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_921() { // q_zero_point
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::q_zero_point(self);
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_922() { // q_per_channel_scales
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::q_per_channel_scales(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_923() { // q_per_channel_zero_points
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::q_per_channel_zero_points(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_924() { // q_per_channel_axis
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::q_per_channel_axis(self);
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_925() { // int_repr
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::int_repr(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_926() { // _make_per_tensor_quantized_tensor
      double scale = readAttribute<float>("scale");
      int64_t zero_point = readAttribute<int64_t>("zero_point");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_make_per_tensor_quantized_tensor(self, scale, zero_point);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_927() { // _make_per_channel_quantized_tensor
      int64_t axis = readAttribute<int64_t>("axis");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto scale = peek(1, 3);
          auto zero_point = peek(2, 3);
          auto the_result = at::_make_per_channel_quantized_tensor(self, scale, zero_point, axis);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_928() { // fake_quantize_per_tensor_affine
      double scale = readAttribute<float>("scale");
      int64_t zero_point = readAttribute<int64_t>("zero_point");
      int64_t quant_min = readAttribute<int64_t>("quant_min");
      int64_t quant_max = readAttribute<int64_t>("quant_max");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::fake_quantize_per_tensor_affine(self, scale, zero_point, quant_min, quant_max);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_929() { // fake_quantize_per_tensor_affine
      int64_t quant_min = readAttribute<int64_t>("quant_min");
      int64_t quant_max = readAttribute<int64_t>("quant_max");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto scale = peek(1, 3);
          auto zero_point = peek(2, 3);
          auto the_result = at::fake_quantize_per_tensor_affine(self, scale, zero_point, quant_min, quant_max);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_930() { // fake_quantize_per_tensor_affine_cachemask
      double scale = readAttribute<float>("scale");
      int64_t zero_point = readAttribute<int64_t>("zero_point");
      int64_t quant_min = readAttribute<int64_t>("quant_min");
      int64_t quant_max = readAttribute<int64_t>("quant_max");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::fake_quantize_per_tensor_affine_cachemask(self, scale, zero_point, quant_min, quant_max);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_931() { // _fake_quantize_per_tensor_affine_cachemask_tensor_qparams
      int64_t quant_min = readAttribute<int64_t>("quant_min");
      int64_t quant_max = readAttribute<int64_t>("quant_max");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 4);
          auto scale = peek(1, 4);
          auto zero_point = peek(2, 4);
          auto fake_quant_enabled = peek(3, 4);
          auto the_result = at::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams(self, scale, zero_point, fake_quant_enabled, quant_min, quant_max);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_932() { // fake_quantize_per_tensor_affine_cachemask_backward
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad = peek(0, 2);
          auto mask = peek(1, 2);
          auto the_result = at::fake_quantize_per_tensor_affine_cachemask_backward(grad, mask);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_933() { // _fake_quantize_learnable_per_tensor_affine
      int64_t quant_min = readAttribute<int64_t>("quant_min");
      int64_t quant_max = readAttribute<int64_t>("quant_max");
      double grad_factor = readAttribute<float>("grad_factor");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto scale = peek(1, 3);
          auto zero_point = peek(2, 3);
          auto the_result = at::_fake_quantize_learnable_per_tensor_affine(self, scale, zero_point, quant_min, quant_max, grad_factor);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_934() { // _fake_quantize_learnable_per_tensor_affine
      int64_t quant_min = readAttribute<int64_t>("quant_min");
      int64_t quant_max = readAttribute<int64_t>("quant_max");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto scale = peek(1, 3);
          auto zero_point = peek(2, 3);
          auto the_result = at::_fake_quantize_learnable_per_tensor_affine(self, scale, zero_point, quant_min, quant_max);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_935() { // _fake_quantize_learnable_per_tensor_affine_backward
      int64_t quant_min = readAttribute<int64_t>("quant_min");
      int64_t quant_max = readAttribute<int64_t>("quant_max");
      double grad_factor = readAttribute<float>("grad_factor");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad = peek(0, 4);
          auto self = peek(1, 4);
          auto scale = peek(2, 4);
          auto zero_point = peek(3, 4);
          auto the_result = at::_fake_quantize_learnable_per_tensor_affine_backward(grad, self, scale, zero_point, quant_min, quant_max, grad_factor);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_936() { // _fake_quantize_learnable_per_tensor_affine_backward
      int64_t quant_min = readAttribute<int64_t>("quant_min");
      int64_t quant_max = readAttribute<int64_t>("quant_max");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad = peek(0, 4);
          auto self = peek(1, 4);
          auto scale = peek(2, 4);
          auto zero_point = peek(3, 4);
          auto the_result = at::_fake_quantize_learnable_per_tensor_affine_backward(grad, self, scale, zero_point, quant_min, quant_max);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_937() { // fake_quantize_per_channel_affine
      int64_t axis = readAttribute<int64_t>("axis");
      int64_t quant_min = readAttribute<int64_t>("quant_min");
      int64_t quant_max = readAttribute<int64_t>("quant_max");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto scale = peek(1, 3);
          auto zero_point = peek(2, 3);
          auto the_result = at::fake_quantize_per_channel_affine(self, scale, zero_point, axis, quant_min, quant_max);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_938() { // fake_quantize_per_channel_affine_cachemask
      int64_t axis = readAttribute<int64_t>("axis");
      int64_t quant_min = readAttribute<int64_t>("quant_min");
      int64_t quant_max = readAttribute<int64_t>("quant_max");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto scale = peek(1, 3);
          auto zero_point = peek(2, 3);
          auto the_result = at::fake_quantize_per_channel_affine_cachemask(self, scale, zero_point, axis, quant_min, quant_max);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_939() { // fake_quantize_per_channel_affine_cachemask_backward
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad = peek(0, 2);
          auto mask = peek(1, 2);
          auto the_result = at::fake_quantize_per_channel_affine_cachemask_backward(grad, mask);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_940() { // _fake_quantize_learnable_per_channel_affine
      int64_t axis = readAttribute<int64_t>("axis");
      int64_t quant_min = readAttribute<int64_t>("quant_min");
      int64_t quant_max = readAttribute<int64_t>("quant_max");
      double grad_factor = readAttribute<float>("grad_factor");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto scale = peek(1, 3);
          auto zero_point = peek(2, 3);
          auto the_result = at::_fake_quantize_learnable_per_channel_affine(self, scale, zero_point, axis, quant_min, quant_max, grad_factor);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_941() { // _fake_quantize_learnable_per_channel_affine
      int64_t axis = readAttribute<int64_t>("axis");
      int64_t quant_min = readAttribute<int64_t>("quant_min");
      int64_t quant_max = readAttribute<int64_t>("quant_max");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto scale = peek(1, 3);
          auto zero_point = peek(2, 3);
          auto the_result = at::_fake_quantize_learnable_per_channel_affine(self, scale, zero_point, axis, quant_min, quant_max);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_942() { // _fake_quantize_learnable_per_channel_affine_backward
      int64_t axis = readAttribute<int64_t>("axis");
      int64_t quant_min = readAttribute<int64_t>("quant_min");
      int64_t quant_max = readAttribute<int64_t>("quant_max");
      double grad_factor = readAttribute<float>("grad_factor");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad = peek(0, 4);
          auto self = peek(1, 4);
          auto scale = peek(2, 4);
          auto zero_point = peek(3, 4);
          auto the_result = at::_fake_quantize_learnable_per_channel_affine_backward(grad, self, scale, zero_point, axis, quant_min, quant_max, grad_factor);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_943() { // _fake_quantize_learnable_per_channel_affine_backward
      int64_t axis = readAttribute<int64_t>("axis");
      int64_t quant_min = readAttribute<int64_t>("quant_min");
      int64_t quant_max = readAttribute<int64_t>("quant_max");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad = peek(0, 4);
          auto self = peek(1, 4);
          auto scale = peek(2, 4);
          auto zero_point = peek(3, 4);
          auto the_result = at::_fake_quantize_learnable_per_channel_affine_backward(grad, self, scale, zero_point, axis, quant_min, quant_max);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_944() { // fused_moving_avg_obs_fake_quant
      double averaging_const = readAttribute<float>("averaging_const");
      int64_t quant_min = readAttribute<int64_t>("quant_min");
      int64_t quant_max = readAttribute<int64_t>("quant_max");
      int64_t ch_axis = readAttribute<int64_t>("ch_axis");
      bool per_row_fake_quant = readAttribute<int64_t>("per_row_fake_quant");
      bool symmetric_quant = readAttribute<int64_t>("symmetric_quant");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 7);
          auto observer_on = peek(1, 7);
          auto fake_quant_on = peek(2, 7);
          auto running_min = peek(3, 7);
          auto running_max = peek(4, 7);
          auto scale = peek(5, 7);
          auto zero_point = peek(6, 7);
          auto the_result = at::fused_moving_avg_obs_fake_quant(self, observer_on, fake_quant_on, running_min, running_max, scale, zero_point, averaging_const, quant_min, quant_max, ch_axis, per_row_fake_quant, symmetric_quant);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_945() { // fused_moving_avg_obs_fake_quant
      double averaging_const = readAttribute<float>("averaging_const");
      int64_t quant_min = readAttribute<int64_t>("quant_min");
      int64_t quant_max = readAttribute<int64_t>("quant_max");
      int64_t ch_axis = readAttribute<int64_t>("ch_axis");
      bool per_row_fake_quant = readAttribute<int64_t>("per_row_fake_quant");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 7);
          auto observer_on = peek(1, 7);
          auto fake_quant_on = peek(2, 7);
          auto running_min = peek(3, 7);
          auto running_max = peek(4, 7);
          auto scale = peek(5, 7);
          auto zero_point = peek(6, 7);
          auto the_result = at::fused_moving_avg_obs_fake_quant(self, observer_on, fake_quant_on, running_min, running_max, scale, zero_point, averaging_const, quant_min, quant_max, ch_axis, per_row_fake_quant);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_946() { // fused_moving_avg_obs_fake_quant
      double averaging_const = readAttribute<float>("averaging_const");
      int64_t quant_min = readAttribute<int64_t>("quant_min");
      int64_t quant_max = readAttribute<int64_t>("quant_max");
      int64_t ch_axis = readAttribute<int64_t>("ch_axis");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 7);
          auto observer_on = peek(1, 7);
          auto fake_quant_on = peek(2, 7);
          auto running_min = peek(3, 7);
          auto running_max = peek(4, 7);
          auto scale = peek(5, 7);
          auto zero_point = peek(6, 7);
          auto the_result = at::fused_moving_avg_obs_fake_quant(self, observer_on, fake_quant_on, running_min, running_max, scale, zero_point, averaging_const, quant_min, quant_max, ch_axis);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_947() { // _fused_moving_avg_obs_fq_helper
      double averaging_const = readAttribute<float>("averaging_const");
      int64_t quant_min = readAttribute<int64_t>("quant_min");
      int64_t quant_max = readAttribute<int64_t>("quant_max");
      int64_t ch_axis = readAttribute<int64_t>("ch_axis");
      bool per_row_fake_quant = readAttribute<int64_t>("per_row_fake_quant");
      bool symmetric_quant = readAttribute<int64_t>("symmetric_quant");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 7);
          auto observer_on = peek(1, 7);
          auto fake_quant_on = peek(2, 7);
          auto running_min = peek(3, 7);
          auto running_max = peek(4, 7);
          auto scale = peek(5, 7);
          auto zero_point = peek(6, 7);
          auto the_result = at::_fused_moving_avg_obs_fq_helper(self, observer_on, fake_quant_on, running_min, running_max, scale, zero_point, averaging_const, quant_min, quant_max, ch_axis, per_row_fake_quant, symmetric_quant);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_948() { // _fused_moving_avg_obs_fq_helper
      double averaging_const = readAttribute<float>("averaging_const");
      int64_t quant_min = readAttribute<int64_t>("quant_min");
      int64_t quant_max = readAttribute<int64_t>("quant_max");
      int64_t ch_axis = readAttribute<int64_t>("ch_axis");
      bool per_row_fake_quant = readAttribute<int64_t>("per_row_fake_quant");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 7);
          auto observer_on = peek(1, 7);
          auto fake_quant_on = peek(2, 7);
          auto running_min = peek(3, 7);
          auto running_max = peek(4, 7);
          auto scale = peek(5, 7);
          auto zero_point = peek(6, 7);
          auto the_result = at::_fused_moving_avg_obs_fq_helper(self, observer_on, fake_quant_on, running_min, running_max, scale, zero_point, averaging_const, quant_min, quant_max, ch_axis, per_row_fake_quant);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_949() { // _fused_moving_avg_obs_fq_helper
      double averaging_const = readAttribute<float>("averaging_const");
      int64_t quant_min = readAttribute<int64_t>("quant_min");
      int64_t quant_max = readAttribute<int64_t>("quant_max");
      int64_t ch_axis = readAttribute<int64_t>("ch_axis");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 7);
          auto observer_on = peek(1, 7);
          auto fake_quant_on = peek(2, 7);
          auto running_min = peek(3, 7);
          auto running_max = peek(4, 7);
          auto scale = peek(5, 7);
          auto zero_point = peek(6, 7);
          auto the_result = at::_fused_moving_avg_obs_fq_helper(self, observer_on, fake_quant_on, running_min, running_max, scale, zero_point, averaging_const, quant_min, quant_max, ch_axis);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_950() { // _saturate_weight_to_fp16
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto weight = peek(0, 1);
          auto the_result = at::_saturate_weight_to_fp16(weight);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_951() { // choose_qparams_optimized
      int64_t numel = readAttribute<int64_t>("numel");
      int64_t n_bins = readAttribute<int64_t>("n_bins");
      double ratio = readAttribute<float>("ratio");
      int64_t bit_width = readAttribute<int64_t>("bit_width");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 1);
          auto the_result = at::choose_qparams_optimized(input, numel, n_bins, ratio, bit_width);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_952() { // meshgrid
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::meshgrid(tensors);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_953() { // cartesian_prod
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::cartesian_prod(tensors);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_954() { // combinations
      int64_t r = readAttribute<int64_t>("r");
      bool with_replacement = readAttribute<int64_t>("with_replacement");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::combinations(self, r, with_replacement);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_955() { // combinations
      int64_t r = readAttribute<int64_t>("r");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::combinations(self, r);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_956() { // combinations
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::combinations(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_957() { // item
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = self.item();
            if(OutputSize() > 0) {assignTo(Output(0),the_result.type(), the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_958() { // _local_scalar_dense
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_local_scalar_dense(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result.type(), the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_959() { // _thnn_fused_lstm_cell
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input_gates = peek(0, 5);
          auto hidden_gates = peek(1, 5);
          auto cx = peek(2, 5);
          auto input_bias = peek(3, 5);
          auto hidden_bias = peek(4, 5);
          auto the_result = at::_thnn_fused_lstm_cell(input_gates, hidden_gates, cx, input_bias, hidden_bias);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_960() { // _thnn_fused_lstm_cell
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input_gates = peek(0, 4);
          auto hidden_gates = peek(1, 4);
          auto cx = peek(2, 4);
          auto input_bias = peek(3, 4);
          auto the_result = at::_thnn_fused_lstm_cell(input_gates, hidden_gates, cx, input_bias);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_961() { // _thnn_fused_lstm_cell
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input_gates = peek(0, 3);
          auto hidden_gates = peek(1, 3);
          auto cx = peek(2, 3);
          auto the_result = at::_thnn_fused_lstm_cell(input_gates, hidden_gates, cx);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_962() { // _thnn_fused_lstm_cell_backward
      bool has_bias = readAttribute<int64_t>("has_bias");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_hy = peek(0, 5);
          auto grad_cy = peek(1, 5);
          auto cx = peek(2, 5);
          auto cy = peek(3, 5);
          auto workspace = peek(4, 5);
          auto the_result = at::_thnn_fused_lstm_cell_backward(grad_hy, grad_cy, cx, cy, workspace, has_bias);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
            if(OutputSize() > 3) {assignTo(Output(3),::std::get<3>(the_result));}
            if(OutputSize() > 4) {assignTo(Output(4),::std::get<4>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_963() { // _thnn_differentiable_lstm_cell_backward
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_hy = peek(0, 8);
          auto grad_cy = peek(1, 8);
          auto input_gates = peek(2, 8);
          auto hidden_gates = peek(3, 8);
          auto input_bias = peek(4, 8);
          auto hidden_bias = peek(5, 8);
          auto cx = peek(6, 8);
          auto cy = peek(7, 8);
          auto the_result = at::_thnn_differentiable_lstm_cell_backward(grad_hy, grad_cy, input_gates, hidden_gates, input_bias, hidden_bias, cx, cy);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
            if(OutputSize() > 3) {assignTo(Output(3),::std::get<3>(the_result));}
            if(OutputSize() > 4) {assignTo(Output(4),::std::get<4>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_964() { // _thnn_fused_gru_cell
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input_gates = peek(0, 5);
          auto hidden_gates = peek(1, 5);
          auto hx = peek(2, 5);
          auto input_bias = peek(3, 5);
          auto hidden_bias = peek(4, 5);
          auto the_result = at::_thnn_fused_gru_cell(input_gates, hidden_gates, hx, input_bias, hidden_bias);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_965() { // _thnn_fused_gru_cell
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input_gates = peek(0, 4);
          auto hidden_gates = peek(1, 4);
          auto hx = peek(2, 4);
          auto input_bias = peek(3, 4);
          auto the_result = at::_thnn_fused_gru_cell(input_gates, hidden_gates, hx, input_bias);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_966() { // _thnn_fused_gru_cell
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input_gates = peek(0, 3);
          auto hidden_gates = peek(1, 3);
          auto hx = peek(2, 3);
          auto the_result = at::_thnn_fused_gru_cell(input_gates, hidden_gates, hx);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_967() { // _thnn_fused_gru_cell_backward
      bool has_bias = readAttribute<int64_t>("has_bias");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_hy = peek(0, 2);
          auto workspace = peek(1, 2);
          auto the_result = at::_thnn_fused_gru_cell_backward(grad_hy, workspace, has_bias);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
            if(OutputSize() > 3) {assignTo(Output(3),::std::get<3>(the_result));}
            if(OutputSize() > 4) {assignTo(Output(4),::std::get<4>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_968() { // _thnn_differentiable_gru_cell_backward
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_hy = peek(0, 6);
          auto input_gates = peek(1, 6);
          auto hidden_gates = peek(2, 6);
          auto hx = peek(3, 6);
          auto input_bias = peek(4, 6);
          auto hidden_bias = peek(5, 6);
          auto the_result = at::_thnn_differentiable_gru_cell_backward(grad_hy, input_gates, hidden_gates, hx, input_bias, hidden_bias);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
            if(OutputSize() > 3) {assignTo(Output(3),::std::get<3>(the_result));}
            if(OutputSize() > 4) {assignTo(Output(4),::std::get<4>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_969() { // lstm
      bool has_biases = readAttribute<int64_t>("has_biases");
      int64_t num_layers = readAttribute<int64_t>("num_layers");
      double dropout = readAttribute<float>("dropout");
      bool train = readAttribute<int64_t>("train");
      bool bidirectional = readAttribute<int64_t>("bidirectional");
      bool batch_first = readAttribute<int64_t>("batch_first");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, InputSize());
          auto hx = peekSlice(1, InputSize() - 1, InputSize());
          auto params = peekSlice(1, InputSize() - 1, InputSize());
          auto the_result = at::lstm(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_970() { // lstm
      bool has_biases = readAttribute<int64_t>("has_biases");
      int64_t num_layers = readAttribute<int64_t>("num_layers");
      double dropout = readAttribute<float>("dropout");
      bool train = readAttribute<int64_t>("train");
      bool bidirectional = readAttribute<int64_t>("bidirectional");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto data = peek(0, InputSize());
          auto batch_sizes = peek(1, InputSize());
          auto hx = peekSlice(2, InputSize() - 2, InputSize());
          auto params = peekSlice(2, InputSize() - 2, InputSize());
          auto the_result = at::lstm(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_971() { // gru
      bool has_biases = readAttribute<int64_t>("has_biases");
      int64_t num_layers = readAttribute<int64_t>("num_layers");
      double dropout = readAttribute<float>("dropout");
      bool train = readAttribute<int64_t>("train");
      bool bidirectional = readAttribute<int64_t>("bidirectional");
      bool batch_first = readAttribute<int64_t>("batch_first");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, InputSize());
          auto hx = peek(1, InputSize());
          auto params = peekSlice(2, InputSize() - 2, InputSize());
          auto the_result = at::gru(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_972() { // gru
      bool has_biases = readAttribute<int64_t>("has_biases");
      int64_t num_layers = readAttribute<int64_t>("num_layers");
      double dropout = readAttribute<float>("dropout");
      bool train = readAttribute<int64_t>("train");
      bool bidirectional = readAttribute<int64_t>("bidirectional");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto data = peek(0, InputSize());
          auto batch_sizes = peek(1, InputSize());
          auto hx = peek(2, InputSize());
          auto params = peekSlice(3, InputSize() - 3, InputSize());
          auto the_result = at::gru(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_973() { // rnn_tanh
      bool has_biases = readAttribute<int64_t>("has_biases");
      int64_t num_layers = readAttribute<int64_t>("num_layers");
      double dropout = readAttribute<float>("dropout");
      bool train = readAttribute<int64_t>("train");
      bool bidirectional = readAttribute<int64_t>("bidirectional");
      bool batch_first = readAttribute<int64_t>("batch_first");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, InputSize());
          auto hx = peek(1, InputSize());
          auto params = peekSlice(2, InputSize() - 2, InputSize());
          auto the_result = at::rnn_tanh(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_974() { // rnn_tanh
      bool has_biases = readAttribute<int64_t>("has_biases");
      int64_t num_layers = readAttribute<int64_t>("num_layers");
      double dropout = readAttribute<float>("dropout");
      bool train = readAttribute<int64_t>("train");
      bool bidirectional = readAttribute<int64_t>("bidirectional");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto data = peek(0, InputSize());
          auto batch_sizes = peek(1, InputSize());
          auto hx = peek(2, InputSize());
          auto params = peekSlice(3, InputSize() - 3, InputSize());
          auto the_result = at::rnn_tanh(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_975() { // rnn_relu
      bool has_biases = readAttribute<int64_t>("has_biases");
      int64_t num_layers = readAttribute<int64_t>("num_layers");
      double dropout = readAttribute<float>("dropout");
      bool train = readAttribute<int64_t>("train");
      bool bidirectional = readAttribute<int64_t>("bidirectional");
      bool batch_first = readAttribute<int64_t>("batch_first");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, InputSize());
          auto hx = peek(1, InputSize());
          auto params = peekSlice(2, InputSize() - 2, InputSize());
          auto the_result = at::rnn_relu(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_976() { // rnn_relu
      bool has_biases = readAttribute<int64_t>("has_biases");
      int64_t num_layers = readAttribute<int64_t>("num_layers");
      double dropout = readAttribute<float>("dropout");
      bool train = readAttribute<int64_t>("train");
      bool bidirectional = readAttribute<int64_t>("bidirectional");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto data = peek(0, InputSize());
          auto batch_sizes = peek(1, InputSize());
          auto hx = peek(2, InputSize());
          auto params = peekSlice(3, InputSize() - 3, InputSize());
          auto the_result = at::rnn_relu(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_977() { // lstm_cell
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, InputSize());
          auto hx = peekSlice(1, InputSize() - 5, InputSize());
          auto w_ih = peek(1, 5);
          auto w_hh = peek(2, 5);
          auto b_ih = peek(3, 5);
          auto b_hh = peek(4, 5);
          auto the_result = at::lstm_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_978() { // gru_cell
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 6);
          auto hx = peek(1, 6);
          auto w_ih = peek(2, 6);
          auto w_hh = peek(3, 6);
          auto b_ih = peek(4, 6);
          auto b_hh = peek(5, 6);
          auto the_result = at::gru_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_979() { // gru_cell
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 5);
          auto hx = peek(1, 5);
          auto w_ih = peek(2, 5);
          auto w_hh = peek(3, 5);
          auto b_ih = peek(4, 5);
          auto the_result = at::gru_cell(input, hx, w_ih, w_hh, b_ih);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_980() { // gru_cell
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 4);
          auto hx = peek(1, 4);
          auto w_ih = peek(2, 4);
          auto w_hh = peek(3, 4);
          auto the_result = at::gru_cell(input, hx, w_ih, w_hh);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_981() { // rnn_tanh_cell
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 6);
          auto hx = peek(1, 6);
          auto w_ih = peek(2, 6);
          auto w_hh = peek(3, 6);
          auto b_ih = peek(4, 6);
          auto b_hh = peek(5, 6);
          auto the_result = at::rnn_tanh_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_982() { // rnn_tanh_cell
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 5);
          auto hx = peek(1, 5);
          auto w_ih = peek(2, 5);
          auto w_hh = peek(3, 5);
          auto b_ih = peek(4, 5);
          auto the_result = at::rnn_tanh_cell(input, hx, w_ih, w_hh, b_ih);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_983() { // rnn_tanh_cell
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 4);
          auto hx = peek(1, 4);
          auto w_ih = peek(2, 4);
          auto w_hh = peek(3, 4);
          auto the_result = at::rnn_tanh_cell(input, hx, w_ih, w_hh);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_984() { // rnn_relu_cell
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 6);
          auto hx = peek(1, 6);
          auto w_ih = peek(2, 6);
          auto w_hh = peek(3, 6);
          auto b_ih = peek(4, 6);
          auto b_hh = peek(5, 6);
          auto the_result = at::rnn_relu_cell(input, hx, w_ih, w_hh, b_ih, b_hh);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_985() { // rnn_relu_cell
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 5);
          auto hx = peek(1, 5);
          auto w_ih = peek(2, 5);
          auto w_hh = peek(3, 5);
          auto b_ih = peek(4, 5);
          auto the_result = at::rnn_relu_cell(input, hx, w_ih, w_hh, b_ih);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_986() { // rnn_relu_cell
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 4);
          auto hx = peek(1, 4);
          auto w_ih = peek(2, 4);
          auto w_hh = peek(3, 4);
          auto the_result = at::rnn_relu_cell(input, hx, w_ih, w_hh);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_987() { // quantized_lstm_cell
      at::Scalar scale_ih = readScalarAttribute("scale_ih");
      at::Scalar scale_hh = readScalarAttribute("scale_hh");
      at::Scalar zero_point_ih = readScalarAttribute("zero_point_ih");
      at::Scalar zero_point_hh = readScalarAttribute("zero_point_hh");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, InputSize());
          auto hx = peekSlice(1, InputSize() - 9, InputSize());
          auto w_ih = peek(1, 9);
          auto w_hh = peek(2, 9);
          auto b_ih = peek(3, 9);
          auto b_hh = peek(4, 9);
          auto packed_ih = peek(5, 9);
          auto packed_hh = peek(6, 9);
          auto col_offsets_ih = peek(7, 9);
          auto col_offsets_hh = peek(8, 9);
          auto the_result = at::quantized_lstm_cell(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_988() { // quantized_gru_cell
      at::Scalar scale_ih = readScalarAttribute("scale_ih");
      at::Scalar scale_hh = readScalarAttribute("scale_hh");
      at::Scalar zero_point_ih = readScalarAttribute("zero_point_ih");
      at::Scalar zero_point_hh = readScalarAttribute("zero_point_hh");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 10);
          auto hx = peek(1, 10);
          auto w_ih = peek(2, 10);
          auto w_hh = peek(3, 10);
          auto b_ih = peek(4, 10);
          auto b_hh = peek(5, 10);
          auto packed_ih = peek(6, 10);
          auto packed_hh = peek(7, 10);
          auto col_offsets_ih = peek(8, 10);
          auto col_offsets_hh = peek(9, 10);
          auto the_result = at::quantized_gru_cell(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_989() { // quantized_rnn_relu_cell
      at::Scalar scale_ih = readScalarAttribute("scale_ih");
      at::Scalar scale_hh = readScalarAttribute("scale_hh");
      at::Scalar zero_point_ih = readScalarAttribute("zero_point_ih");
      at::Scalar zero_point_hh = readScalarAttribute("zero_point_hh");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 10);
          auto hx = peek(1, 10);
          auto w_ih = peek(2, 10);
          auto w_hh = peek(3, 10);
          auto b_ih = peek(4, 10);
          auto b_hh = peek(5, 10);
          auto packed_ih = peek(6, 10);
          auto packed_hh = peek(7, 10);
          auto col_offsets_ih = peek(8, 10);
          auto col_offsets_hh = peek(9, 10);
          auto the_result = at::quantized_rnn_relu_cell(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_990() { // quantized_rnn_tanh_cell
      at::Scalar scale_ih = readScalarAttribute("scale_ih");
      at::Scalar scale_hh = readScalarAttribute("scale_hh");
      at::Scalar zero_point_ih = readScalarAttribute("zero_point_ih");
      at::Scalar zero_point_hh = readScalarAttribute("zero_point_hh");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 10);
          auto hx = peek(1, 10);
          auto w_ih = peek(2, 10);
          auto w_hh = peek(3, 10);
          auto b_ih = peek(4, 10);
          auto b_hh = peek(5, 10);
          auto packed_ih = peek(6, 10);
          auto packed_hh = peek(7, 10);
          auto col_offsets_ih = peek(8, 10);
          auto col_offsets_hh = peek(9, 10);
          auto the_result = at::quantized_rnn_tanh_cell(input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_991() { // _pack_padded_sequence
      bool batch_first = readAttribute<int64_t>("batch_first");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 2);
          auto lengths = peek(1, 2);
          auto the_result = at::_pack_padded_sequence(input, lengths, batch_first);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_992() { // _pack_padded_sequence_backward
      auto input_size = readIntArrayRef("input_size");
      bool batch_first = readAttribute<int64_t>("batch_first");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad = peek(0, 2);
          auto batch_sizes = peek(1, 2);
          auto the_result = at::_pack_padded_sequence_backward(grad, input_size, batch_sizes, batch_first);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_993() { // _pad_packed_sequence
      bool batch_first = readAttribute<int64_t>("batch_first");
      at::Scalar padding_value = readScalarAttribute("padding_value");
      int64_t total_length = readAttribute<int64_t>("total_length");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto data = peek(0, 2);
          auto batch_sizes = peek(1, 2);
          auto the_result = at::_pad_packed_sequence(data, batch_sizes, batch_first, padding_value, total_length);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_994() { // is_set_to
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto tensor = peek(1, 2);
          auto the_result = self.is_set_to(tensor);
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_995() { // masked_fill
      at::Scalar value = readScalarAttribute("value");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto mask = peek(1, 2);
          auto the_result = at::masked_fill(self, mask, value);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_996() { // masked_fill
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto mask = peek(1, 3);
          auto value = peek(2, 3);
          auto the_result = at::masked_fill(self, mask, value);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_997() { // masked_scatter
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto mask = peek(1, 3);
          auto source = peek(2, 3);
          auto the_result = at::masked_scatter(self, mask, source);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_998() { // view
      auto size = readIntArrayRef("size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = self.view(size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_999() { // put
      bool accumulate = readAttribute<int64_t>("accumulate");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto index = peek(1, 3);
          auto source = peek(2, 3);
          auto the_result = at::put(self, index, source, accumulate);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1000() { // put
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto index = peek(1, 3);
          auto source = peek(2, 3);
          auto the_result = at::put(self, index, source);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1001() { // index_add
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto index = peek(1, 3);
          auto source = peek(2, 3);
          auto the_result = at::index_add(self, dim, index, source);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1002() { // index_add
      int64_t dim = readAttribute<int64_t>("dim");
      at::Scalar alpha = readScalarAttribute("alpha");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto index = peek(1, 3);
          auto source = peek(2, 3);
          auto the_result = at::index_add(self, dim, index, source, alpha);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1003() { // index_fill
      int64_t dim = readAttribute<int64_t>("dim");
      at::Scalar value = readScalarAttribute("value");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto index = peek(1, 2);
          auto the_result = at::index_fill(self, dim, index, value);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1004() { // index_fill
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto index = peek(1, 3);
          auto value = peek(2, 3);
          auto the_result = at::index_fill(self, dim, index, value);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1005() { // scatter
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto index = peek(1, 3);
          auto src = peek(2, 3);
          auto the_result = at::scatter(self, dim, index, src);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1006() { // scatter
      int64_t dim = readAttribute<int64_t>("dim");
      at::Scalar value = readScalarAttribute("value");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto index = peek(1, 2);
          auto the_result = at::scatter(self, dim, index, value);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1007() { // scatter_add
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto index = peek(1, 3);
          auto src = peek(2, 3);
          auto the_result = at::scatter_add(self, dim, index, src);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1008() { // bitwise_and
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::bitwise_and(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1009() { // bitwise_and
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::bitwise_and(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1010() { // __and__
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::__and__(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1011() { // __and__
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::__and__(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1012() { // bitwise_or
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::bitwise_or(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1013() { // bitwise_or
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::bitwise_or(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1014() { // __or__
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::__or__(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1015() { // __or__
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::__or__(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1016() { // bitwise_xor
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::bitwise_xor(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1017() { // bitwise_xor
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::bitwise_xor(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1018() { // __xor__
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::__xor__(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1019() { // __xor__
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::__xor__(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1020() { // __lshift__
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::__lshift__(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1021() { // __lshift__
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::__lshift__(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1022() { // bitwise_left_shift
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::bitwise_left_shift(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1023() { // bitwise_left_shift
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::bitwise_left_shift(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1024() { // bitwise_left_shift
      at::Scalar self = readScalarAttribute("self");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto other = peek(0, 1);
          auto the_result = at::bitwise_left_shift(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1025() { // __rshift__
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::__rshift__(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1026() { // __rshift__
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::__rshift__(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1027() { // bitwise_right_shift
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::bitwise_right_shift(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1028() { // bitwise_right_shift
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::bitwise_right_shift(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1029() { // bitwise_right_shift
      at::Scalar self = readScalarAttribute("self");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto other = peek(0, 1);
          auto the_result = at::bitwise_right_shift(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1030() { // addbmm
      at::Scalar beta = readScalarAttribute("beta");
      at::Scalar alpha = readScalarAttribute("alpha");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto batch1 = peek(1, 3);
          auto batch2 = peek(2, 3);
          auto the_result = at::addbmm(self, batch1, batch2, beta, alpha);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1031() { // addbmm
      at::Scalar beta = readScalarAttribute("beta");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto batch1 = peek(1, 3);
          auto batch2 = peek(2, 3);
          auto the_result = at::addbmm(self, batch1, batch2, beta);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1032() { // addbmm
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto batch1 = peek(1, 3);
          auto batch2 = peek(2, 3);
          auto the_result = at::addbmm(self, batch1, batch2);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1033() { // diag
      int64_t diagonal = readAttribute<int64_t>("diagonal");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::diag(self, diagonal);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1034() { // diag
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::diag(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1035() { // diag_backward
      auto input_sizes = readIntArrayRef("input_sizes");
      int64_t diagonal = readAttribute<int64_t>("diagonal");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad = peek(0, 1);
          auto the_result = at::diag_backward(grad, input_sizes, diagonal);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1036() { // cross
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::cross(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1037() { // triu
      int64_t diagonal = readAttribute<int64_t>("diagonal");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::triu(self, diagonal);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1038() { // triu
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::triu(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1039() { // tril
      int64_t diagonal = readAttribute<int64_t>("diagonal");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::tril(self, diagonal);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1040() { // tril
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::tril(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1041() { // trace
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::trace(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1042() { // trace_backward
      auto sizes = readIntArrayRef("sizes");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad = peek(0, 1);
          auto the_result = at::trace_backward(grad, sizes);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1043() { // ne
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::ne(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1044() { // ne
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::ne(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1045() { // not_equal
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::not_equal(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1046() { // not_equal
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::not_equal(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1047() { // eq
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::eq(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1048() { // eq
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::eq(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1049() { // ge
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::ge(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1050() { // ge
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::ge(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1051() { // greater_equal
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::greater_equal(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1052() { // greater_equal
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::greater_equal(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1053() { // le
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::le(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1054() { // le
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::le(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1055() { // less_equal
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::less_equal(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1056() { // less_equal
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::less_equal(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1057() { // gt
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::gt(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1058() { // gt
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::gt(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1059() { // greater
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::greater(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1060() { // greater
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::greater(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1061() { // lt
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::lt(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1062() { // lt
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::lt(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1063() { // less
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::less(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1064() { // less
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::less(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1065() { // take
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto index = peek(1, 2);
          auto the_result = at::take(self, index);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1066() { // take_along_dim
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto indices = peek(1, 2);
          auto the_result = at::take_along_dim(self, indices);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1067() { // index_select
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto index = peek(1, 2);
          auto the_result = at::index_select(self, dim, index);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1068() { // index_select_backward
      auto self_sizes = readIntArrayRef("self_sizes");
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad = peek(0, 2);
          auto index = peek(1, 2);
          auto the_result = at::index_select_backward(grad, self_sizes, dim, index);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1069() { // masked_select
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto mask = peek(1, 2);
          auto the_result = at::masked_select(self, mask);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1070() { // masked_select_backward
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad = peek(0, 3);
          auto input = peek(1, 3);
          auto mask = peek(2, 3);
          auto the_result = at::masked_select_backward(grad, input, mask);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1071() { // nonzero
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::nonzero(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1072() { // nonzero_numpy
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::nonzero_numpy(self);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1073() { // gather
      int64_t dim = readAttribute<int64_t>("dim");
      bool sparse_grad = readAttribute<int64_t>("sparse_grad");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto index = peek(1, 2);
          auto the_result = at::gather(self, dim, index, sparse_grad);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1074() { // gather
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto index = peek(1, 2);
          auto the_result = at::gather(self, dim, index);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1075() { // gather_backward
      int64_t dim = readAttribute<int64_t>("dim");
      bool sparse_grad = readAttribute<int64_t>("sparse_grad");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad = peek(0, 3);
          auto self = peek(1, 3);
          auto index = peek(2, 3);
          auto the_result = at::gather_backward(grad, self, dim, index, sparse_grad);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1076() { // _gather_sparse_backward
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto index = peek(1, 3);
          auto grad = peek(2, 3);
          auto the_result = at::_gather_sparse_backward(self, dim, index, grad);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1077() { // addcmul
      at::Scalar value = readScalarAttribute("value");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto tensor1 = peek(1, 3);
          auto tensor2 = peek(2, 3);
          auto the_result = at::addcmul(self, tensor1, tensor2, value);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1078() { // addcmul
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto tensor1 = peek(1, 3);
          auto tensor2 = peek(2, 3);
          auto the_result = at::addcmul(self, tensor1, tensor2);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1079() { // addcdiv
      at::Scalar value = readScalarAttribute("value");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto tensor1 = peek(1, 3);
          auto tensor2 = peek(2, 3);
          auto the_result = at::addcdiv(self, tensor1, tensor2, value);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1080() { // addcdiv
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto tensor1 = peek(1, 3);
          auto tensor2 = peek(2, 3);
          auto the_result = at::addcdiv(self, tensor1, tensor2);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1081() { // cross_entropy_loss
      int64_t reduction = readAttribute<int64_t>("reduction");
      int64_t ignore_index = readAttribute<int64_t>("ignore_index");
      double label_smoothing = readAttribute<float>("label_smoothing");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto target = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::cross_entropy_loss(self, target, weight, reduction, ignore_index, label_smoothing);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1082() { // cross_entropy_loss
      int64_t reduction = readAttribute<int64_t>("reduction");
      int64_t ignore_index = readAttribute<int64_t>("ignore_index");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto target = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::cross_entropy_loss(self, target, weight, reduction, ignore_index);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1083() { // cross_entropy_loss
      int64_t reduction = readAttribute<int64_t>("reduction");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto target = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::cross_entropy_loss(self, target, weight, reduction);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1084() { // cross_entropy_loss
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto target = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::cross_entropy_loss(self, target, weight);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1085() { // cross_entropy_loss
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto target = peek(1, 2);
          auto the_result = at::cross_entropy_loss(self, target);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1086() { // lstsq
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto A = peek(1, 2);
          auto the_result = at::lstsq(self, A);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1087() { // triangular_solve
      bool upper = readAttribute<int64_t>("upper");
      bool transpose = readAttribute<int64_t>("transpose");
      bool unitriangular = readAttribute<int64_t>("unitriangular");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto A = peek(1, 2);
          auto the_result = at::triangular_solve(self, A, upper, transpose, unitriangular);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1088() { // triangular_solve
      bool upper = readAttribute<int64_t>("upper");
      bool transpose = readAttribute<int64_t>("transpose");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto A = peek(1, 2);
          auto the_result = at::triangular_solve(self, A, upper, transpose);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1089() { // triangular_solve
      bool upper = readAttribute<int64_t>("upper");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto A = peek(1, 2);
          auto the_result = at::triangular_solve(self, A, upper);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1090() { // triangular_solve
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto A = peek(1, 2);
          auto the_result = at::triangular_solve(self, A);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1091() { // symeig
      bool eigenvectors = readAttribute<int64_t>("eigenvectors");
      bool upper = readAttribute<int64_t>("upper");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::symeig(self, eigenvectors, upper);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1092() { // symeig
      bool eigenvectors = readAttribute<int64_t>("eigenvectors");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::symeig(self, eigenvectors);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1093() { // symeig
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::symeig(self);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1094() { // _symeig_helper
      bool eigenvectors = readAttribute<int64_t>("eigenvectors");
      bool upper = readAttribute<int64_t>("upper");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_symeig_helper(self, eigenvectors, upper);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1095() { // eig
      bool eigenvectors = readAttribute<int64_t>("eigenvectors");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::eig(self, eigenvectors);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1096() { // eig
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::eig(self);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1097() { // svd
      bool some = readAttribute<int64_t>("some");
      bool compute_uv = readAttribute<int64_t>("compute_uv");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::svd(self, some, compute_uv);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1098() { // svd
      bool some = readAttribute<int64_t>("some");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::svd(self, some);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1099() { // svd
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::svd(self);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1100() { // _svd_helper
      bool some = readAttribute<int64_t>("some");
      bool compute_uv = readAttribute<int64_t>("compute_uv");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_svd_helper(self, some, compute_uv);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1101() { // swapaxes
      int64_t axis0 = readAttribute<int64_t>("axis0");
      int64_t axis1 = readAttribute<int64_t>("axis1");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::swapaxes(self, axis0, axis1);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1102() { // swapdims
      int64_t dim0 = readAttribute<int64_t>("dim0");
      int64_t dim1 = readAttribute<int64_t>("dim1");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::swapdims(self, dim0, dim1);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1103() { // cholesky
      bool upper = readAttribute<int64_t>("upper");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::cholesky(self, upper);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1104() { // cholesky
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::cholesky(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1105() { // cholesky_solve
      bool upper = readAttribute<int64_t>("upper");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto input2 = peek(1, 2);
          auto the_result = at::cholesky_solve(self, input2, upper);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1106() { // cholesky_solve
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto input2 = peek(1, 2);
          auto the_result = at::cholesky_solve(self, input2);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1107() { // _cholesky_solve_helper
      bool upper = readAttribute<int64_t>("upper");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto A = peek(1, 2);
          auto the_result = at::_cholesky_solve_helper(self, A, upper);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1108() { // solve
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto A = peek(1, 2);
          auto the_result = at::solve(self, A);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1109() { // _solve_helper
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto A = peek(1, 2);
          auto the_result = at::_solve_helper(self, A);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1110() { // cholesky_inverse
      bool upper = readAttribute<int64_t>("upper");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::cholesky_inverse(self, upper);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1111() { // cholesky_inverse
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::cholesky_inverse(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1112() { // qr
      bool some = readAttribute<int64_t>("some");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::qr(self, some);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1113() { // qr
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::qr(self);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1114() { // geqrf
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::geqrf(self);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1115() { // orgqr
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto input2 = peek(1, 2);
          auto the_result = at::orgqr(self, input2);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1116() { // ormqr
      bool left = readAttribute<int64_t>("left");
      bool transpose = readAttribute<int64_t>("transpose");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto input2 = peek(1, 3);
          auto input3 = peek(2, 3);
          auto the_result = at::ormqr(self, input2, input3, left, transpose);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1117() { // ormqr
      bool left = readAttribute<int64_t>("left");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto input2 = peek(1, 3);
          auto input3 = peek(2, 3);
          auto the_result = at::ormqr(self, input2, input3, left);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1118() { // ormqr
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto input2 = peek(1, 3);
          auto input3 = peek(2, 3);
          auto the_result = at::ormqr(self, input2, input3);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1119() { // _lu_with_info
      bool pivot = readAttribute<int64_t>("pivot");
      bool check_errors = readAttribute<int64_t>("check_errors");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_lu_with_info(self, pivot, check_errors);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1120() { // _lu_with_info
      bool pivot = readAttribute<int64_t>("pivot");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_lu_with_info(self, pivot);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1121() { // _lu_with_info
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_lu_with_info(self);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1122() { // lu_solve
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto LU_data = peek(1, 3);
          auto LU_pivots = peek(2, 3);
          auto the_result = at::lu_solve(self, LU_data, LU_pivots);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1123() { // lu_unpack
      bool unpack_data = readAttribute<int64_t>("unpack_data");
      bool unpack_pivots = readAttribute<int64_t>("unpack_pivots");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto LU_data = peek(0, 2);
          auto LU_pivots = peek(1, 2);
          auto the_result = at::lu_unpack(LU_data, LU_pivots, unpack_data, unpack_pivots);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1124() { // lu_unpack
      bool unpack_data = readAttribute<int64_t>("unpack_data");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto LU_data = peek(0, 2);
          auto LU_pivots = peek(1, 2);
          auto the_result = at::lu_unpack(LU_data, LU_pivots, unpack_data);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1125() { // lu_unpack
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto LU_data = peek(0, 2);
          auto LU_pivots = peek(1, 2);
          auto the_result = at::lu_unpack(LU_data, LU_pivots);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1126() { // multinomial
      int64_t num_samples = readAttribute<int64_t>("num_samples");
      bool replacement = readAttribute<int64_t>("replacement");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::multinomial(self, num_samples, replacement);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1127() { // multinomial
      int64_t num_samples = readAttribute<int64_t>("num_samples");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::multinomial(self, num_samples);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1128() { // lgamma
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::lgamma(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1129() { // digamma
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::digamma(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1130() { // polygamma
      int64_t n = readAttribute<int64_t>("n");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::polygamma(n, self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1131() { // erfinv
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::erfinv(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1132() { // i0
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::i0(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1133() { // sign
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::sign(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1134() { // signbit
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::signbit(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1135() { // dist
      at::Scalar p = readScalarAttribute("p");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::dist(self, other, p);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1136() { // dist
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::dist(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1137() { // atan2
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::atan2(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1138() { // lerp
      at::Scalar weight = readScalarAttribute("weight");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto end = peek(1, 2);
          auto the_result = at::lerp(self, end, weight);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1139() { // lerp
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto end = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::lerp(self, end, weight);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1140() { // histc
      int64_t bins = readAttribute<int64_t>("bins");
      at::Scalar min = readScalarAttribute("min");
      at::Scalar max = readScalarAttribute("max");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::histc(self, bins, min, max);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1141() { // histc
      int64_t bins = readAttribute<int64_t>("bins");
      at::Scalar min = readScalarAttribute("min");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::histc(self, bins, min);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1142() { // histc
      int64_t bins = readAttribute<int64_t>("bins");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::histc(self, bins);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1143() { // histc
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::histc(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1144() { // histogram
      bool density = readAttribute<int64_t>("density");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto bins = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::histogram(self, bins, weight, density);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1145() { // histogram
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto bins = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::histogram(self, bins, weight);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1146() { // histogram
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto bins = peek(1, 2);
          auto the_result = at::histogram(self, bins);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1147() { // histogram
      int64_t bins = readAttribute<int64_t>("bins");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::histogram(self, bins);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1148() { // histogram
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::histogram(self);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1149() { // fmod
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::fmod(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1150() { // fmod
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::fmod(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1151() { // hypot
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::hypot(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1152() { // igamma
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::igamma(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1153() { // igammac
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::igammac(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1154() { // nextafter
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::nextafter(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1155() { // remainder
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::remainder(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1156() { // remainder
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::remainder(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1157() { // remainder
      at::Scalar self = readScalarAttribute("self");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto other = peek(0, 1);
          auto the_result = at::remainder(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1158() { // min
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::min(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1159() { // fmin
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::fmin(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1160() { // max
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::max(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1161() { // fmax
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::fmax(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1162() { // maximum
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::maximum(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1163() { // max
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::max(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1164() { // minimum
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::minimum(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1165() { // min
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::min(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1166() { // quantile
      double q = readAttribute<float>("q");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::quantile(self, q);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1167() { // quantile
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto q = peek(1, 2);
          auto the_result = at::quantile(self, q);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1168() { // nanquantile
      double q = readAttribute<float>("q");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::nanquantile(self, q);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1169() { // nanquantile
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto q = peek(1, 2);
          auto the_result = at::nanquantile(self, q);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1170() { // sort
      int64_t dim = readAttribute<int64_t>("dim");
      bool descending = readAttribute<int64_t>("descending");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::sort(self, dim, descending);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1171() { // sort
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::sort(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1172() { // sort
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::sort(self);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1173() { // msort
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::msort(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1174() { // argsort
      int64_t dim = readAttribute<int64_t>("dim");
      bool descending = readAttribute<int64_t>("descending");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::argsort(self, dim, descending);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1175() { // argsort
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::argsort(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1176() { // argsort
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::argsort(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1177() { // topk
      int64_t k = readAttribute<int64_t>("k");
      int64_t dim = readAttribute<int64_t>("dim");
      bool largest = readAttribute<int64_t>("largest");
      bool sorted = readAttribute<int64_t>("sorted");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::topk(self, k, dim, largest, sorted);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1178() { // topk
      int64_t k = readAttribute<int64_t>("k");
      int64_t dim = readAttribute<int64_t>("dim");
      bool largest = readAttribute<int64_t>("largest");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::topk(self, k, dim, largest);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1179() { // topk
      int64_t k = readAttribute<int64_t>("k");
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::topk(self, k, dim);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1180() { // topk
      int64_t k = readAttribute<int64_t>("k");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::topk(self, k);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1181() { // all
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::all(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1182() { // any
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::any(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1183() { // renorm
      at::Scalar p = readScalarAttribute("p");
      int64_t dim = readAttribute<int64_t>("dim");
      at::Scalar maxnorm = readScalarAttribute("maxnorm");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::renorm(self, p, dim, maxnorm);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1184() { // unfold
      int64_t dimension = readAttribute<int64_t>("dimension");
      int64_t size = readAttribute<int64_t>("size");
      int64_t step = readAttribute<int64_t>("step");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = self.unfold(dimension, size, step);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1185() { // unfold_backward
      auto input_sizes = readIntArrayRef("input_sizes");
      int64_t dim = readAttribute<int64_t>("dim");
      int64_t size = readAttribute<int64_t>("size");
      int64_t step = readAttribute<int64_t>("step");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_in = peek(0, 1);
          auto the_result = at::unfold_backward(grad_in, input_sizes, dim, size, step);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1186() { // equal
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::equal(self, other);
            if(OutputSize() > 0) {assignToValue<int64_t>(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1187() { // pow
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto exponent = peek(1, 2);
          auto the_result = at::pow(self, exponent);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1188() { // pow
      at::Scalar self = readScalarAttribute("self");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto exponent = peek(0, 1);
          auto the_result = at::pow(self, exponent);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1189() { // pow
      at::Scalar exponent = readScalarAttribute("exponent");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::pow(self, exponent);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1190() { // float_power
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto exponent = peek(1, 2);
          auto the_result = at::float_power(self, exponent);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1191() { // float_power
      at::Scalar self = readScalarAttribute("self");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto exponent = peek(0, 1);
          auto the_result = at::float_power(self, exponent);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1192() { // float_power
      at::Scalar exponent = readScalarAttribute("exponent");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::float_power(self, exponent);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1193() { // alias
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::alias(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1194() { // _cat
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_cat(tensors, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1195() { // _cat
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_cat(tensors);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1196() { // _foreach_add
      at::Scalar scalar = readScalarAttribute("scalar");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_add(tensors, scalar);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1197() { // _foreach_sub
      at::Scalar scalar = readScalarAttribute("scalar");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_sub(tensors, scalar);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1198() { // _foreach_mul
      at::Scalar scalar = readScalarAttribute("scalar");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_mul(tensors, scalar);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1199() { // _foreach_div
      at::Scalar scalar = readScalarAttribute("scalar");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_div(tensors, scalar);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1200() { // _foreach_add
      at::Scalar alpha = readScalarAttribute("alpha");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors1 = peekSlice(0, InputSize() - 0, InputSize());
          auto tensors2 = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_add(tensors1, tensors2, alpha);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1201() { // _foreach_add
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors1 = peekSlice(0, InputSize() - 0, InputSize());
          auto tensors2 = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_add(tensors1, tensors2);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1202() { // _foreach_sub
      at::Scalar alpha = readScalarAttribute("alpha");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors1 = peekSlice(0, InputSize() - 0, InputSize());
          auto tensors2 = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_sub(tensors1, tensors2, alpha);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1203() { // _foreach_sub
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors1 = peekSlice(0, InputSize() - 0, InputSize());
          auto tensors2 = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_sub(tensors1, tensors2);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1204() { // _foreach_mul
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors1 = peekSlice(0, InputSize() - 0, InputSize());
          auto tensors2 = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_mul(tensors1, tensors2);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1205() { // _foreach_div
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors1 = peekSlice(0, InputSize() - 0, InputSize());
          auto tensors2 = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_div(tensors1, tensors2);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1206() { // _foreach_exp
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_exp(tensors);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1207() { // _foreach_sqrt
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_sqrt(tensors);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1208() { // _foreach_abs
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_abs(tensors);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1209() { // _foreach_acos
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_acos(tensors);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1210() { // _foreach_asin
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_asin(tensors);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1211() { // _foreach_atan
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_atan(tensors);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1212() { // _foreach_ceil
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_ceil(tensors);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1213() { // _foreach_cos
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_cos(tensors);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1214() { // _foreach_cosh
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_cosh(tensors);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1215() { // _foreach_erf
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_erf(tensors);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1216() { // _foreach_erfc
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_erfc(tensors);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1217() { // _foreach_expm1
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_expm1(tensors);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1218() { // _foreach_floor
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_floor(tensors);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1219() { // _foreach_log
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_log(tensors);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1220() { // _foreach_log10
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_log10(tensors);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1221() { // _foreach_log1p
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_log1p(tensors);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1222() { // _foreach_log2
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_log2(tensors);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1223() { // _foreach_neg
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_neg(tensors);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1224() { // _foreach_tan
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_tan(tensors);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1225() { // _foreach_tanh
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_tanh(tensors);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1226() { // _foreach_sin
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_sin(tensors);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1227() { // _foreach_sinh
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_sinh(tensors);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1228() { // _foreach_round
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_round(tensors);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1229() { // _foreach_lgamma
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_lgamma(tensors);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1230() { // _foreach_frac
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_frac(tensors);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1231() { // _foreach_reciprocal
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_reciprocal(tensors);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1232() { // _foreach_sigmoid
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_sigmoid(tensors);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1233() { // _foreach_trunc
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_trunc(tensors);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1234() { // _foreach_addcdiv
      at::Scalar value = readScalarAttribute("value");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peekSlice(0, InputSize() - 0, InputSize());
          auto tensor1 = peekSlice(0, InputSize() - 0, InputSize());
          auto tensor2 = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_addcdiv(input, tensor1, tensor2, value);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1235() { // _foreach_addcdiv
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peekSlice(0, InputSize() - 0, InputSize());
          auto tensor1 = peekSlice(0, InputSize() - 0, InputSize());
          auto tensor2 = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_addcdiv(input, tensor1, tensor2);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1236() { // _foreach_addcmul
      at::Scalar value = readScalarAttribute("value");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peekSlice(0, InputSize() - 0, InputSize());
          auto tensor1 = peekSlice(0, InputSize() - 0, InputSize());
          auto tensor2 = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_addcmul(input, tensor1, tensor2, value);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1237() { // _foreach_addcmul
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peekSlice(0, InputSize() - 0, InputSize());
          auto tensor1 = peekSlice(0, InputSize() - 0, InputSize());
          auto tensor2 = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_addcmul(input, tensor1, tensor2);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1238() { // _foreach_maximum
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors1 = peekSlice(0, InputSize() - 0, InputSize());
          auto tensors2 = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_maximum(tensors1, tensors2);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1239() { // _foreach_minimum
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors1 = peekSlice(0, InputSize() - 0, InputSize());
          auto tensors2 = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::_foreach_minimum(tensors1, tensors2);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1240() { // bucketize
      bool out_int32 = readAttribute<int64_t>("out_int32");
      bool right = readAttribute<int64_t>("right");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto boundaries = peek(1, 2);
          auto the_result = at::bucketize(self, boundaries, out_int32, right);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1241() { // bucketize
      bool out_int32 = readAttribute<int64_t>("out_int32");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto boundaries = peek(1, 2);
          auto the_result = at::bucketize(self, boundaries, out_int32);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1242() { // bucketize
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto boundaries = peek(1, 2);
          auto the_result = at::bucketize(self, boundaries);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1243() { // bucketize
      at::Scalar self = readScalarAttribute("self");
      bool out_int32 = readAttribute<int64_t>("out_int32");
      bool right = readAttribute<int64_t>("right");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto boundaries = peek(0, 1);
          auto the_result = at::bucketize(self, boundaries, out_int32, right);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1244() { // bucketize
      at::Scalar self = readScalarAttribute("self");
      bool out_int32 = readAttribute<int64_t>("out_int32");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto boundaries = peek(0, 1);
          auto the_result = at::bucketize(self, boundaries, out_int32);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1245() { // bucketize
      at::Scalar self = readScalarAttribute("self");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto boundaries = peek(0, 1);
          auto the_result = at::bucketize(self, boundaries);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1246() { // searchsorted
      bool out_int32 = readAttribute<int64_t>("out_int32");
      bool right = readAttribute<int64_t>("right");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto sorted_sequence = peek(0, 2);
          auto self = peek(1, 2);
          auto the_result = at::searchsorted(sorted_sequence, self, out_int32, right);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1247() { // searchsorted
      bool out_int32 = readAttribute<int64_t>("out_int32");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto sorted_sequence = peek(0, 2);
          auto self = peek(1, 2);
          auto the_result = at::searchsorted(sorted_sequence, self, out_int32);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1248() { // searchsorted
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto sorted_sequence = peek(0, 2);
          auto self = peek(1, 2);
          auto the_result = at::searchsorted(sorted_sequence, self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1249() { // searchsorted
      at::Scalar self = readScalarAttribute("self");
      bool out_int32 = readAttribute<int64_t>("out_int32");
      bool right = readAttribute<int64_t>("right");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto sorted_sequence = peek(0, 1);
          auto the_result = at::searchsorted(sorted_sequence, self, out_int32, right);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1250() { // searchsorted
      at::Scalar self = readScalarAttribute("self");
      bool out_int32 = readAttribute<int64_t>("out_int32");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto sorted_sequence = peek(0, 1);
          auto the_result = at::searchsorted(sorted_sequence, self, out_int32);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1251() { // searchsorted
      at::Scalar self = readScalarAttribute("self");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto sorted_sequence = peek(0, 1);
          auto the_result = at::searchsorted(sorted_sequence, self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1252() { // _convert_indices_from_coo_to_csr
      int64_t size = readAttribute<int64_t>("size");
      bool out_int32 = readAttribute<int64_t>("out_int32");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_convert_indices_from_coo_to_csr(self, size, out_int32);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1253() { // _convert_indices_from_coo_to_csr
      int64_t size = readAttribute<int64_t>("size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_convert_indices_from_coo_to_csr(self, size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1254() { // mse_loss
      int64_t reduction = readAttribute<int64_t>("reduction");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto target = peek(1, 2);
          auto the_result = at::mse_loss(self, target, reduction);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1255() { // mse_loss
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto target = peek(1, 2);
          auto the_result = at::mse_loss(self, target);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1256() { // mse_loss_backward
      int64_t reduction = readAttribute<int64_t>("reduction");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto self = peek(1, 3);
          auto target = peek(2, 3);
          auto the_result = at::mse_loss_backward(grad_output, self, target, reduction);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1257() { // l1_loss
      int64_t reduction = readAttribute<int64_t>("reduction");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto target = peek(1, 2);
          auto the_result = at::l1_loss(self, target, reduction);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1258() { // l1_loss
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto target = peek(1, 2);
          auto the_result = at::l1_loss(self, target);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1259() { // l1_loss_backward
      int64_t reduction = readAttribute<int64_t>("reduction");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto self = peek(1, 3);
          auto target = peek(2, 3);
          auto the_result = at::l1_loss_backward(grad_output, self, target, reduction);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1260() { // multi_margin_loss
      at::Scalar p = readScalarAttribute("p");
      at::Scalar margin = readScalarAttribute("margin");
      int64_t reduction = readAttribute<int64_t>("reduction");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto target = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::multi_margin_loss(self, target, p, margin, weight, reduction);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1261() { // multi_margin_loss
      at::Scalar p = readScalarAttribute("p");
      at::Scalar margin = readScalarAttribute("margin");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto target = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::multi_margin_loss(self, target, p, margin, weight);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1262() { // multi_margin_loss
      at::Scalar p = readScalarAttribute("p");
      at::Scalar margin = readScalarAttribute("margin");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto target = peek(1, 2);
          auto the_result = at::multi_margin_loss(self, target, p, margin);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1263() { // multi_margin_loss
      at::Scalar p = readScalarAttribute("p");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto target = peek(1, 2);
          auto the_result = at::multi_margin_loss(self, target, p);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1264() { // multi_margin_loss
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto target = peek(1, 2);
          auto the_result = at::multi_margin_loss(self, target);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1265() { // multi_margin_loss_backward
      at::Scalar p = readScalarAttribute("p");
      at::Scalar margin = readScalarAttribute("margin");
      int64_t reduction = readAttribute<int64_t>("reduction");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 4);
          auto self = peek(1, 4);
          auto target = peek(2, 4);
          auto weight = peek(3, 4);
          auto the_result = at::multi_margin_loss_backward(grad_output, self, target, p, margin, weight, reduction);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1266() { // multi_margin_loss_backward
      at::Scalar p = readScalarAttribute("p");
      at::Scalar margin = readScalarAttribute("margin");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 4);
          auto self = peek(1, 4);
          auto target = peek(2, 4);
          auto weight = peek(3, 4);
          auto the_result = at::multi_margin_loss_backward(grad_output, self, target, p, margin, weight);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1267() { // multi_margin_loss_backward
      at::Scalar p = readScalarAttribute("p");
      at::Scalar margin = readScalarAttribute("margin");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto self = peek(1, 3);
          auto target = peek(2, 3);
          auto the_result = at::multi_margin_loss_backward(grad_output, self, target, p, margin);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1268() { // multilabel_margin_loss
      int64_t reduction = readAttribute<int64_t>("reduction");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto target = peek(1, 2);
          auto the_result = at::multilabel_margin_loss(self, target, reduction);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1269() { // multilabel_margin_loss
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto target = peek(1, 2);
          auto the_result = at::multilabel_margin_loss(self, target);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1270() { // multilabel_margin_loss_forward
      int64_t reduction = readAttribute<int64_t>("reduction");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto target = peek(1, 2);
          auto the_result = at::multilabel_margin_loss_forward(self, target, reduction);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1271() { // multilabel_margin_loss_backward
      int64_t reduction = readAttribute<int64_t>("reduction");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 4);
          auto self = peek(1, 4);
          auto target = peek(2, 4);
          auto is_target = peek(3, 4);
          auto the_result = at::multilabel_margin_loss_backward(grad_output, self, target, reduction, is_target);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1272() { // nll_loss_nd
      int64_t reduction = readAttribute<int64_t>("reduction");
      int64_t ignore_index = readAttribute<int64_t>("ignore_index");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto target = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::nll_loss_nd(self, target, weight, reduction, ignore_index);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1273() { // nll_loss_nd
      int64_t reduction = readAttribute<int64_t>("reduction");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto target = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::nll_loss_nd(self, target, weight, reduction);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1274() { // nll_loss_nd
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto target = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::nll_loss_nd(self, target, weight);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1275() { // nll_loss_nd
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto target = peek(1, 2);
          auto the_result = at::nll_loss_nd(self, target);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1276() { // nll_loss
      int64_t reduction = readAttribute<int64_t>("reduction");
      int64_t ignore_index = readAttribute<int64_t>("ignore_index");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto target = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::nll_loss(self, target, weight, reduction, ignore_index);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1277() { // nll_loss
      int64_t reduction = readAttribute<int64_t>("reduction");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto target = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::nll_loss(self, target, weight, reduction);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1278() { // nll_loss
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto target = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::nll_loss(self, target, weight);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1279() { // nll_loss
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto target = peek(1, 2);
          auto the_result = at::nll_loss(self, target);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1280() { // nll_loss_forward
      int64_t reduction = readAttribute<int64_t>("reduction");
      int64_t ignore_index = readAttribute<int64_t>("ignore_index");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto target = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::nll_loss_forward(self, target, weight, reduction, ignore_index);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1281() { // nll_loss_backward
      int64_t reduction = readAttribute<int64_t>("reduction");
      int64_t ignore_index = readAttribute<int64_t>("ignore_index");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 5);
          auto self = peek(1, 5);
          auto target = peek(2, 5);
          auto weight = peek(3, 5);
          auto total_weight = peek(4, 5);
          auto the_result = at::nll_loss_backward(grad_output, self, target, weight, reduction, ignore_index, total_weight);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1282() { // nll_loss2d
      int64_t reduction = readAttribute<int64_t>("reduction");
      int64_t ignore_index = readAttribute<int64_t>("ignore_index");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto target = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::nll_loss2d(self, target, weight, reduction, ignore_index);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1283() { // nll_loss2d
      int64_t reduction = readAttribute<int64_t>("reduction");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto target = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::nll_loss2d(self, target, weight, reduction);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1284() { // nll_loss2d
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto target = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::nll_loss2d(self, target, weight);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1285() { // nll_loss2d
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto target = peek(1, 2);
          auto the_result = at::nll_loss2d(self, target);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1286() { // nll_loss2d_forward
      int64_t reduction = readAttribute<int64_t>("reduction");
      int64_t ignore_index = readAttribute<int64_t>("ignore_index");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto target = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::nll_loss2d_forward(self, target, weight, reduction, ignore_index);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1287() { // nll_loss2d_backward
      int64_t reduction = readAttribute<int64_t>("reduction");
      int64_t ignore_index = readAttribute<int64_t>("ignore_index");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 5);
          auto self = peek(1, 5);
          auto target = peek(2, 5);
          auto weight = peek(3, 5);
          auto total_weight = peek(4, 5);
          auto the_result = at::nll_loss2d_backward(grad_output, self, target, weight, reduction, ignore_index, total_weight);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1288() { // smooth_l1_loss
      int64_t reduction = readAttribute<int64_t>("reduction");
      double beta = readAttribute<float>("beta");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto target = peek(1, 2);
          auto the_result = at::smooth_l1_loss(self, target, reduction, beta);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1289() { // smooth_l1_loss
      int64_t reduction = readAttribute<int64_t>("reduction");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto target = peek(1, 2);
          auto the_result = at::smooth_l1_loss(self, target, reduction);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1290() { // smooth_l1_loss
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto target = peek(1, 2);
          auto the_result = at::smooth_l1_loss(self, target);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1291() { // smooth_l1_loss_backward
      int64_t reduction = readAttribute<int64_t>("reduction");
      double beta = readAttribute<float>("beta");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto self = peek(1, 3);
          auto target = peek(2, 3);
          auto the_result = at::smooth_l1_loss_backward(grad_output, self, target, reduction, beta);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1292() { // huber_loss
      int64_t reduction = readAttribute<int64_t>("reduction");
      double delta = readAttribute<float>("delta");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto target = peek(1, 2);
          auto the_result = at::huber_loss(self, target, reduction, delta);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1293() { // huber_loss
      int64_t reduction = readAttribute<int64_t>("reduction");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto target = peek(1, 2);
          auto the_result = at::huber_loss(self, target, reduction);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1294() { // huber_loss
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto target = peek(1, 2);
          auto the_result = at::huber_loss(self, target);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1295() { // huber_loss_backward
      int64_t reduction = readAttribute<int64_t>("reduction");
      double delta = readAttribute<float>("delta");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto self = peek(1, 3);
          auto target = peek(2, 3);
          auto the_result = at::huber_loss_backward(grad_output, self, target, reduction, delta);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1296() { // soft_margin_loss
      int64_t reduction = readAttribute<int64_t>("reduction");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto target = peek(1, 2);
          auto the_result = at::soft_margin_loss(self, target, reduction);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1297() { // soft_margin_loss
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto target = peek(1, 2);
          auto the_result = at::soft_margin_loss(self, target);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1298() { // soft_margin_loss_backward
      int64_t reduction = readAttribute<int64_t>("reduction");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto self = peek(1, 3);
          auto target = peek(2, 3);
          auto the_result = at::soft_margin_loss_backward(grad_output, self, target, reduction);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1299() { // elu
      at::Scalar alpha = readScalarAttribute("alpha");
      at::Scalar scale = readScalarAttribute("scale");
      at::Scalar input_scale = readScalarAttribute("input_scale");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::elu(self, alpha, scale, input_scale);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1300() { // elu
      at::Scalar alpha = readScalarAttribute("alpha");
      at::Scalar scale = readScalarAttribute("scale");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::elu(self, alpha, scale);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1301() { // elu
      at::Scalar alpha = readScalarAttribute("alpha");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::elu(self, alpha);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1302() { // elu
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::elu(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1303() { // elu_backward
      at::Scalar alpha = readScalarAttribute("alpha");
      at::Scalar scale = readScalarAttribute("scale");
      at::Scalar input_scale = readScalarAttribute("input_scale");
      bool is_result = readAttribute<int64_t>("is_result");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 2);
          auto self_or_result = peek(1, 2);
          auto the_result = at::elu_backward(grad_output, alpha, scale, input_scale, is_result, self_or_result);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1304() { // glu
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::glu(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1305() { // glu
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::glu(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1306() { // glu_backward
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 2);
          auto self = peek(1, 2);
          auto the_result = at::glu_backward(grad_output, self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1307() { // hardsigmoid
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::hardsigmoid(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1308() { // hardsigmoid_backward
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 2);
          auto self = peek(1, 2);
          auto the_result = at::hardsigmoid_backward(grad_output, self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1309() { // hardtanh
      at::Scalar min_val = readScalarAttribute("min_val");
      at::Scalar max_val = readScalarAttribute("max_val");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::hardtanh(self, min_val, max_val);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1310() { // hardtanh
      at::Scalar min_val = readScalarAttribute("min_val");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::hardtanh(self, min_val);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1311() { // hardtanh
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::hardtanh(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1312() { // hardtanh_backward
      at::Scalar min_val = readScalarAttribute("min_val");
      at::Scalar max_val = readScalarAttribute("max_val");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 2);
          auto self = peek(1, 2);
          auto the_result = at::hardtanh_backward(grad_output, self, min_val, max_val);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1313() { // hardswish
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::hardswish(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1314() { // hardswish_backward
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 2);
          auto self = peek(1, 2);
          auto the_result = at::hardswish_backward(grad_output, self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1315() { // leaky_relu
      at::Scalar negative_slope = readScalarAttribute("negative_slope");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::leaky_relu(self, negative_slope);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1316() { // leaky_relu
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::leaky_relu(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1317() { // leaky_relu_backward
      at::Scalar negative_slope = readScalarAttribute("negative_slope");
      bool self_is_result = readAttribute<int64_t>("self_is_result");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 2);
          auto self = peek(1, 2);
          auto the_result = at::leaky_relu_backward(grad_output, self, negative_slope, self_is_result);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1318() { // log_sigmoid
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::log_sigmoid(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1319() { // log_sigmoid_forward
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::log_sigmoid_forward(self);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1320() { // log_sigmoid_backward
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto self = peek(1, 3);
          auto buffer = peek(2, 3);
          auto the_result = at::log_sigmoid_backward(grad_output, self, buffer);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1321() { // rrelu_with_noise
      at::Scalar lower = readScalarAttribute("lower");
      at::Scalar upper = readScalarAttribute("upper");
      bool training = readAttribute<int64_t>("training");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto noise = peek(1, 2);
          auto the_result = at::rrelu_with_noise(self, noise, lower, upper, training);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1322() { // rrelu_with_noise
      at::Scalar lower = readScalarAttribute("lower");
      at::Scalar upper = readScalarAttribute("upper");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto noise = peek(1, 2);
          auto the_result = at::rrelu_with_noise(self, noise, lower, upper);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1323() { // rrelu_with_noise
      at::Scalar lower = readScalarAttribute("lower");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto noise = peek(1, 2);
          auto the_result = at::rrelu_with_noise(self, noise, lower);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1324() { // rrelu_with_noise
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto noise = peek(1, 2);
          auto the_result = at::rrelu_with_noise(self, noise);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1325() { // rrelu_with_noise_backward
      at::Scalar lower = readScalarAttribute("lower");
      at::Scalar upper = readScalarAttribute("upper");
      bool training = readAttribute<int64_t>("training");
      bool self_is_result = readAttribute<int64_t>("self_is_result");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto self = peek(1, 3);
          auto noise = peek(2, 3);
          auto the_result = at::rrelu_with_noise_backward(grad_output, self, noise, lower, upper, training, self_is_result);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1326() { // softplus
      at::Scalar beta = readScalarAttribute("beta");
      at::Scalar threshold = readScalarAttribute("threshold");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::softplus(self, beta, threshold);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1327() { // softplus
      at::Scalar beta = readScalarAttribute("beta");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::softplus(self, beta);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1328() { // softplus
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::softplus(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1329() { // softplus_backward
      at::Scalar beta = readScalarAttribute("beta");
      at::Scalar threshold = readScalarAttribute("threshold");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto self = peek(1, 3);
          auto output = peek(2, 3);
          auto the_result = at::softplus_backward(grad_output, self, beta, threshold, output);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1330() { // softshrink
      at::Scalar lambd = readScalarAttribute("lambd");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::softshrink(self, lambd);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1331() { // softshrink
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::softshrink(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1332() { // softshrink_backward
      at::Scalar lambd = readScalarAttribute("lambd");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 2);
          auto self = peek(1, 2);
          auto the_result = at::softshrink_backward(grad_output, self, lambd);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1333() { // adaptive_avg_pool2d
      auto output_size = readIntArrayRef("output_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::adaptive_avg_pool2d(self, output_size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1334() { // mkldnn_adaptive_avg_pool2d
      auto output_size = readIntArrayRef("output_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::mkldnn_adaptive_avg_pool2d(self, output_size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1335() { // mkldnn_adaptive_avg_pool2d_backward
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 2);
          auto self = peek(1, 2);
          auto the_result = at::mkldnn_adaptive_avg_pool2d_backward(grad_output, self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1336() { // _adaptive_avg_pool2d
      auto output_size = readIntArrayRef("output_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_adaptive_avg_pool2d(self, output_size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1337() { // _adaptive_avg_pool2d_backward
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 2);
          auto self = peek(1, 2);
          auto the_result = at::_adaptive_avg_pool2d_backward(grad_output, self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1338() { // adaptive_avg_pool3d
      auto output_size = readIntArrayRef("output_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::adaptive_avg_pool3d(self, output_size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1339() { // _adaptive_avg_pool3d
      auto output_size = readIntArrayRef("output_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_adaptive_avg_pool3d(self, output_size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1340() { // _adaptive_avg_pool3d_backward
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 2);
          auto self = peek(1, 2);
          auto the_result = at::_adaptive_avg_pool3d_backward(grad_output, self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1341() { // adaptive_max_pool2d
      auto output_size = readIntArrayRef("output_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::adaptive_max_pool2d(self, output_size);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1342() { // adaptive_max_pool2d_backward
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto self = peek(1, 3);
          auto indices = peek(2, 3);
          auto the_result = at::adaptive_max_pool2d_backward(grad_output, self, indices);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1343() { // adaptive_max_pool3d
      auto output_size = readIntArrayRef("output_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::adaptive_max_pool3d(self, output_size);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1344() { // adaptive_max_pool3d_backward
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto self = peek(1, 3);
          auto indices = peek(2, 3);
          auto the_result = at::adaptive_max_pool3d_backward(grad_output, self, indices);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1345() { // avg_pool2d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      bool ceil_mode = readAttribute<int64_t>("ceil_mode");
      bool count_include_pad = readAttribute<int64_t>("count_include_pad");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::avg_pool2d(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1346() { // avg_pool2d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      bool ceil_mode = readAttribute<int64_t>("ceil_mode");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::avg_pool2d(self, kernel_size, stride, padding, ceil_mode);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1347() { // avg_pool2d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::avg_pool2d(self, kernel_size, stride, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1348() { // avg_pool2d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::avg_pool2d(self, kernel_size, stride);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1349() { // avg_pool2d
      auto kernel_size = readIntArrayRef("kernel_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::avg_pool2d(self, kernel_size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1350() { // avg_pool3d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      bool ceil_mode = readAttribute<int64_t>("ceil_mode");
      bool count_include_pad = readAttribute<int64_t>("count_include_pad");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::avg_pool3d(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1351() { // avg_pool3d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      bool ceil_mode = readAttribute<int64_t>("ceil_mode");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::avg_pool3d(self, kernel_size, stride, padding, ceil_mode);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1352() { // avg_pool3d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::avg_pool3d(self, kernel_size, stride, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1353() { // avg_pool3d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::avg_pool3d(self, kernel_size, stride);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1354() { // avg_pool3d
      auto kernel_size = readIntArrayRef("kernel_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::avg_pool3d(self, kernel_size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1355() { // fractional_max_pool2d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto output_size = readIntArrayRef("output_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto random_samples = peek(1, 2);
          auto the_result = at::fractional_max_pool2d(self, kernel_size, output_size, random_samples);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1356() { // fractional_max_pool2d_backward
      auto kernel_size = readIntArrayRef("kernel_size");
      auto output_size = readIntArrayRef("output_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto self = peek(1, 3);
          auto indices = peek(2, 3);
          auto the_result = at::fractional_max_pool2d_backward(grad_output, self, kernel_size, output_size, indices);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1357() { // fractional_max_pool3d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto output_size = readIntArrayRef("output_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto random_samples = peek(1, 2);
          auto the_result = at::fractional_max_pool3d(self, kernel_size, output_size, random_samples);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1358() { // fractional_max_pool3d_backward
      auto kernel_size = readIntArrayRef("kernel_size");
      auto output_size = readIntArrayRef("output_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto self = peek(1, 3);
          auto indices = peek(2, 3);
          auto the_result = at::fractional_max_pool3d_backward(grad_output, self, kernel_size, output_size, indices);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1359() { // max_pool2d_with_indices
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      bool ceil_mode = readAttribute<int64_t>("ceil_mode");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::max_pool2d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1360() { // max_pool2d_with_indices
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::max_pool2d_with_indices(self, kernel_size, stride, padding, dilation);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1361() { // max_pool2d_with_indices
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::max_pool2d_with_indices(self, kernel_size, stride, padding);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1362() { // max_pool2d_with_indices
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::max_pool2d_with_indices(self, kernel_size, stride);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1363() { // max_pool2d_with_indices
      auto kernel_size = readIntArrayRef("kernel_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::max_pool2d_with_indices(self, kernel_size);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1364() { // max_pool2d_with_indices_backward
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      bool ceil_mode = readAttribute<int64_t>("ceil_mode");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto self = peek(1, 3);
          auto indices = peek(2, 3);
          auto the_result = at::max_pool2d_with_indices_backward(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1365() { // max_pool3d_with_indices
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      bool ceil_mode = readAttribute<int64_t>("ceil_mode");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::max_pool3d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1366() { // max_pool3d_with_indices
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::max_pool3d_with_indices(self, kernel_size, stride, padding, dilation);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1367() { // max_pool3d_with_indices
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::max_pool3d_with_indices(self, kernel_size, stride, padding);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1368() { // max_pool3d_with_indices
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::max_pool3d_with_indices(self, kernel_size, stride);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1369() { // max_pool3d_with_indices
      auto kernel_size = readIntArrayRef("kernel_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::max_pool3d_with_indices(self, kernel_size);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1370() { // max_pool3d_with_indices_backward
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      bool ceil_mode = readAttribute<int64_t>("ceil_mode");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto self = peek(1, 3);
          auto indices = peek(2, 3);
          auto the_result = at::max_pool3d_with_indices_backward(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1371() { // max_unpool2d
      auto output_size = readIntArrayRef("output_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto indices = peek(1, 2);
          auto the_result = at::max_unpool2d(self, indices, output_size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1372() { // max_unpool2d_backward
      auto output_size = readIntArrayRef("output_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto self = peek(1, 3);
          auto indices = peek(2, 3);
          auto the_result = at::max_unpool2d_backward(grad_output, self, indices, output_size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1373() { // max_unpool3d
      auto output_size = readIntArrayRef("output_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto indices = peek(1, 2);
          auto the_result = at::max_unpool3d(self, indices, output_size, stride, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1374() { // max_unpool3d_backward
      auto output_size = readIntArrayRef("output_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto self = peek(1, 3);
          auto indices = peek(2, 3);
          auto the_result = at::max_unpool3d_backward(grad_output, self, indices, output_size, stride, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1375() { // reflection_pad1d
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::reflection_pad1d(self, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1376() { // reflection_pad1d_backward
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 2);
          auto self = peek(1, 2);
          auto the_result = at::reflection_pad1d_backward(grad_output, self, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1377() { // reflection_pad2d
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::reflection_pad2d(self, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1378() { // reflection_pad2d_backward
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 2);
          auto self = peek(1, 2);
          auto the_result = at::reflection_pad2d_backward(grad_output, self, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1379() { // reflection_pad3d
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::reflection_pad3d(self, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1380() { // reflection_pad3d_backward
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 2);
          auto self = peek(1, 2);
          auto the_result = at::reflection_pad3d_backward(grad_output, self, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1381() { // replication_pad1d
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::replication_pad1d(self, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1382() { // replication_pad1d_backward
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 2);
          auto self = peek(1, 2);
          auto the_result = at::replication_pad1d_backward(grad_output, self, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1383() { // replication_pad2d
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::replication_pad2d(self, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1384() { // replication_pad2d_backward
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 2);
          auto self = peek(1, 2);
          auto the_result = at::replication_pad2d_backward(grad_output, self, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1385() { // replication_pad3d
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::replication_pad3d(self, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1386() { // replication_pad3d_backward
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 2);
          auto self = peek(1, 2);
          auto the_result = at::replication_pad3d_backward(grad_output, self, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1387() { // upsample_linear1d
      auto output_size = readIntArrayRef("output_size");
      bool align_corners = readAttribute<int64_t>("align_corners");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::upsample_linear1d(self, output_size, align_corners);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1388() { // upsample_linear1d_backward
      auto output_size = readIntArrayRef("output_size");
      auto input_size = readIntArrayRef("input_size");
      bool align_corners = readAttribute<int64_t>("align_corners");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 1);
          auto the_result = at::upsample_linear1d_backward(grad_output, output_size, input_size, align_corners);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1389() { // upsample_bilinear2d
      auto output_size = readIntArrayRef("output_size");
      bool align_corners = readAttribute<int64_t>("align_corners");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::upsample_bilinear2d(self, output_size, align_corners);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1390() { // upsample_bilinear2d_backward
      auto output_size = readIntArrayRef("output_size");
      auto input_size = readIntArrayRef("input_size");
      bool align_corners = readAttribute<int64_t>("align_corners");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 1);
          auto the_result = at::upsample_bilinear2d_backward(grad_output, output_size, input_size, align_corners);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1391() { // upsample_bicubic2d
      auto output_size = readIntArrayRef("output_size");
      bool align_corners = readAttribute<int64_t>("align_corners");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::upsample_bicubic2d(self, output_size, align_corners);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1392() { // upsample_bicubic2d_backward
      auto output_size = readIntArrayRef("output_size");
      auto input_size = readIntArrayRef("input_size");
      bool align_corners = readAttribute<int64_t>("align_corners");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 1);
          auto the_result = at::upsample_bicubic2d_backward(grad_output, output_size, input_size, align_corners);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1393() { // upsample_trilinear3d
      auto output_size = readIntArrayRef("output_size");
      bool align_corners = readAttribute<int64_t>("align_corners");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::upsample_trilinear3d(self, output_size, align_corners);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1394() { // upsample_trilinear3d_backward
      auto output_size = readIntArrayRef("output_size");
      auto input_size = readIntArrayRef("input_size");
      bool align_corners = readAttribute<int64_t>("align_corners");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 1);
          auto the_result = at::upsample_trilinear3d_backward(grad_output, output_size, input_size, align_corners);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1395() { // upsample_nearest1d
      auto output_size = readIntArrayRef("output_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::upsample_nearest1d(self, output_size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1396() { // upsample_nearest1d_backward
      auto output_size = readIntArrayRef("output_size");
      auto input_size = readIntArrayRef("input_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 1);
          auto the_result = at::upsample_nearest1d_backward(grad_output, output_size, input_size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1397() { // upsample_nearest2d
      auto output_size = readIntArrayRef("output_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::upsample_nearest2d(self, output_size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1398() { // upsample_nearest2d_backward
      auto output_size = readIntArrayRef("output_size");
      auto input_size = readIntArrayRef("input_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 1);
          auto the_result = at::upsample_nearest2d_backward(grad_output, output_size, input_size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1399() { // upsample_nearest3d
      auto output_size = readIntArrayRef("output_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::upsample_nearest3d(self, output_size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1400() { // upsample_nearest3d_backward
      auto output_size = readIntArrayRef("output_size");
      auto input_size = readIntArrayRef("input_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 1);
          auto the_result = at::upsample_nearest3d_backward(grad_output, output_size, input_size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1401() { // sigmoid_backward
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 2);
          auto output = peek(1, 2);
          auto the_result = at::sigmoid_backward(grad_output, output);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1402() { // logit_backward
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 2);
          auto self = peek(1, 2);
          auto the_result = at::logit_backward(grad_output, self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1403() { // tanh_backward
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 2);
          auto output = peek(1, 2);
          auto the_result = at::tanh_backward(grad_output, output);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1404() { // slow_conv_transpose2d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto output_padding = readIntArrayRef("output_padding");
      auto dilation = readIntArrayRef("dilation");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::slow_conv_transpose2d(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1405() { // slow_conv_transpose2d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto output_padding = readIntArrayRef("output_padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::slow_conv_transpose2d(self, weight, kernel_size, bias, stride, padding, output_padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1406() { // slow_conv_transpose2d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::slow_conv_transpose2d(self, weight, kernel_size, bias, stride, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1407() { // slow_conv_transpose2d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::slow_conv_transpose2d(self, weight, kernel_size, bias, stride);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1408() { // slow_conv_transpose2d
      auto kernel_size = readIntArrayRef("kernel_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::slow_conv_transpose2d(self, weight, kernel_size, bias);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1409() { // slow_conv_transpose2d
      auto kernel_size = readIntArrayRef("kernel_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto weight = peek(1, 2);
          auto the_result = at::slow_conv_transpose2d(self, weight, kernel_size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1410() { // slow_conv_transpose2d_backward
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto output_padding = readIntArrayRef("output_padding");
      auto dilation = readIntArrayRef("dilation");
      auto output_mask = readBoolMask<3>("output_mask");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 5);
          auto self = peek(1, 5);
          auto weight = peek(2, 5);
          auto columns = peek(3, 5);
          auto ones = peek(4, 5);
          auto the_result = at::slow_conv_transpose2d_backward(grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, columns, ones, output_mask);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1411() { // slow_conv_transpose3d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto output_padding = readIntArrayRef("output_padding");
      auto dilation = readIntArrayRef("dilation");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::slow_conv_transpose3d(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1412() { // slow_conv_transpose3d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto output_padding = readIntArrayRef("output_padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::slow_conv_transpose3d(self, weight, kernel_size, bias, stride, padding, output_padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1413() { // slow_conv_transpose3d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::slow_conv_transpose3d(self, weight, kernel_size, bias, stride, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1414() { // slow_conv_transpose3d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::slow_conv_transpose3d(self, weight, kernel_size, bias, stride);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1415() { // slow_conv_transpose3d
      auto kernel_size = readIntArrayRef("kernel_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::slow_conv_transpose3d(self, weight, kernel_size, bias);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1416() { // slow_conv_transpose3d
      auto kernel_size = readIntArrayRef("kernel_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto weight = peek(1, 2);
          auto the_result = at::slow_conv_transpose3d(self, weight, kernel_size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1417() { // slow_conv_transpose3d_backward
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto output_padding = readIntArrayRef("output_padding");
      auto dilation = readIntArrayRef("dilation");
      auto output_mask = readBoolMask<3>("output_mask");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 5);
          auto self = peek(1, 5);
          auto weight = peek(2, 5);
          auto finput = peek(3, 5);
          auto fgrad_input = peek(4, 5);
          auto the_result = at::slow_conv_transpose3d_backward(grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, finput, fgrad_input, output_mask);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1418() { // thnn_conv2d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::thnn_conv2d(self, weight, kernel_size, bias, stride, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1419() { // thnn_conv2d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::thnn_conv2d(self, weight, kernel_size, bias, stride);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1420() { // thnn_conv2d
      auto kernel_size = readIntArrayRef("kernel_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::thnn_conv2d(self, weight, kernel_size, bias);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1421() { // thnn_conv2d
      auto kernel_size = readIntArrayRef("kernel_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto weight = peek(1, 2);
          auto the_result = at::thnn_conv2d(self, weight, kernel_size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1422() { // thnn_conv2d_forward
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::thnn_conv2d_forward(self, weight, kernel_size, bias, stride, padding);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1423() { // thnn_conv2d_backward
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto output_mask = readBoolMask<3>("output_mask");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 5);
          auto self = peek(1, 5);
          auto weight = peek(2, 5);
          auto finput = peek(3, 5);
          auto fgrad_input = peek(4, 5);
          auto the_result = at::thnn_conv2d_backward(grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, output_mask);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1424() { // _conv_depthwise2d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::_conv_depthwise2d(self, weight, kernel_size, bias, stride, padding, dilation);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1425() { // _conv_depthwise2d_backward
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      auto output_mask = readBoolMask<2>("output_mask");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto self = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::_conv_depthwise2d_backward(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1426() { // conv_depthwise3d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::conv_depthwise3d(self, weight, kernel_size, bias, stride, padding, dilation);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1427() { // conv_depthwise3d_backward
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      auto output_mask = readBoolMask<3>("output_mask");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto self = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::conv_depthwise3d_backward(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1428() { // slow_conv3d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::slow_conv3d(self, weight, kernel_size, bias, stride, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1429() { // slow_conv3d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::slow_conv3d(self, weight, kernel_size, bias, stride);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1430() { // slow_conv3d
      auto kernel_size = readIntArrayRef("kernel_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::slow_conv3d(self, weight, kernel_size, bias);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1431() { // slow_conv3d
      auto kernel_size = readIntArrayRef("kernel_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto weight = peek(1, 2);
          auto the_result = at::slow_conv3d(self, weight, kernel_size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1432() { // slow_conv3d_forward
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::slow_conv3d_forward(self, weight, kernel_size, bias, stride, padding);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1433() { // slow_conv3d_backward
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto output_mask = readBoolMask<3>("output_mask");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 5);
          auto self = peek(1, 5);
          auto weight = peek(2, 5);
          auto finput = peek(3, 5);
          auto fgrad_input = peek(4, 5);
          auto the_result = at::slow_conv3d_backward(grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, output_mask);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1434() { // slow_conv_dilated2d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::slow_conv_dilated2d(self, weight, kernel_size, bias, stride, padding, dilation);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1435() { // slow_conv_dilated2d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::slow_conv_dilated2d(self, weight, kernel_size, bias, stride, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1436() { // slow_conv_dilated2d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::slow_conv_dilated2d(self, weight, kernel_size, bias, stride);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1437() { // slow_conv_dilated2d
      auto kernel_size = readIntArrayRef("kernel_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::slow_conv_dilated2d(self, weight, kernel_size, bias);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1438() { // slow_conv_dilated2d
      auto kernel_size = readIntArrayRef("kernel_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto weight = peek(1, 2);
          auto the_result = at::slow_conv_dilated2d(self, weight, kernel_size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1439() { // slow_conv_dilated2d_backward
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      auto output_mask = readBoolMask<3>("output_mask");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto self = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::slow_conv_dilated2d_backward(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1440() { // slow_conv_dilated3d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::slow_conv_dilated3d(self, weight, kernel_size, bias, stride, padding, dilation);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1441() { // slow_conv_dilated3d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::slow_conv_dilated3d(self, weight, kernel_size, bias, stride, padding);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1442() { // slow_conv_dilated3d
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::slow_conv_dilated3d(self, weight, kernel_size, bias, stride);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1443() { // slow_conv_dilated3d
      auto kernel_size = readIntArrayRef("kernel_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 3);
          auto weight = peek(1, 3);
          auto bias = peek(2, 3);
          auto the_result = at::slow_conv_dilated3d(self, weight, kernel_size, bias);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1444() { // slow_conv_dilated3d
      auto kernel_size = readIntArrayRef("kernel_size");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto weight = peek(1, 2);
          auto the_result = at::slow_conv_dilated3d(self, weight, kernel_size);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1445() { // slow_conv_dilated3d_backward
      auto kernel_size = readIntArrayRef("kernel_size");
      auto stride = readIntArrayRef("stride");
      auto padding = readIntArrayRef("padding");
      auto dilation = readIntArrayRef("dilation");
      auto output_mask = readBoolMask<3>("output_mask");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 3);
          auto self = peek(1, 3);
          auto weight = peek(2, 3);
          auto the_result = at::slow_conv_dilated3d_backward(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1446() { // col2im
      auto output_size = readIntArrayRef("output_size");
      auto kernel_size = readIntArrayRef("kernel_size");
      auto dilation = readIntArrayRef("dilation");
      auto padding = readIntArrayRef("padding");
      auto stride = readIntArrayRef("stride");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::col2im(self, output_size, kernel_size, dilation, padding, stride);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1447() { // col2im_backward
      auto kernel_size = readIntArrayRef("kernel_size");
      auto dilation = readIntArrayRef("dilation");
      auto padding = readIntArrayRef("padding");
      auto stride = readIntArrayRef("stride");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 1);
          auto the_result = at::col2im_backward(grad_output, kernel_size, dilation, padding, stride);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1448() { // column_stack
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::column_stack(tensors);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1449() { // im2col
      auto kernel_size = readIntArrayRef("kernel_size");
      auto dilation = readIntArrayRef("dilation");
      auto padding = readIntArrayRef("padding");
      auto stride = readIntArrayRef("stride");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::im2col(self, kernel_size, dilation, padding, stride);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1450() { // im2col_backward
      auto input_size = readIntArrayRef("input_size");
      auto kernel_size = readIntArrayRef("kernel_size");
      auto dilation = readIntArrayRef("dilation");
      auto padding = readIntArrayRef("padding");
      auto stride = readIntArrayRef("stride");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto grad_output = peek(0, 1);
          auto the_result = at::im2col_backward(grad_output, input_size, kernel_size, dilation, padding, stride);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1451() { // isfinite
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::isfinite(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1452() { // isinf
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::isinf(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1453() { // isposinf
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::isposinf(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1454() { // isneginf
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::isneginf(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1455() { // _add_batch_dim
      int64_t batch_dim = readAttribute<int64_t>("batch_dim");
      int64_t level = readAttribute<int64_t>("level");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_add_batch_dim(self, batch_dim, level);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1456() { // _remove_batch_dim
      int64_t level = readAttribute<int64_t>("level");
      int64_t batch_size = readAttribute<int64_t>("batch_size");
      int64_t out_dim = readAttribute<int64_t>("out_dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_remove_batch_dim(self, level, batch_size, out_dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1457() { // special_entr
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::special_entr(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1458() { // special_ndtri
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::special_ndtri(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1459() { // special_expm1
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::special_expm1(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1460() { // special_exp2
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::special_exp2(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1461() { // special_psi
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::special_psi(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1462() { // special_digamma
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::special_digamma(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1463() { // special_gammaln
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::special_gammaln(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1464() { // special_erf
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::special_erf(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1465() { // special_erfc
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::special_erfc(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1466() { // special_erfcx
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::special_erfcx(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1467() { // special_erfinv
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::special_erfinv(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1468() { // special_ndtr
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::special_ndtr(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1469() { // special_xlog1py
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::special_xlog1py(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1470() { // special_xlog1py
      at::Scalar self = readScalarAttribute("self");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto other = peek(0, 1);
          auto the_result = at::special_xlog1py(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1471() { // special_xlog1py
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::special_xlog1py(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1472() { // special_xlogy
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::special_xlogy(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1473() { // special_xlogy
      at::Scalar self = readScalarAttribute("self");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto other = peek(0, 1);
          auto the_result = at::special_xlogy(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1474() { // special_xlogy
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::special_xlogy(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1475() { // special_zeta
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::special_zeta(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1476() { // special_zeta
      at::Scalar self = readScalarAttribute("self");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto other = peek(0, 1);
          auto the_result = at::special_zeta(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1477() { // special_zeta
      at::Scalar other = readScalarAttribute("other");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::special_zeta(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1478() { // special_i0
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::special_i0(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1479() { // special_i0e
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::special_i0e(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1480() { // special_i1
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::special_i1(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1481() { // special_i1e
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::special_i1e(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1482() { // special_logit
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::special_logit(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1483() { // special_polygamma
      int64_t n = readAttribute<int64_t>("n");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::special_polygamma(n, self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1484() { // special_logsumexp
      auto dim = readIntArrayRef("dim");
      bool keepdim = readAttribute<int64_t>("keepdim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::special_logsumexp(self, dim, keepdim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1485() { // special_logsumexp
      auto dim = readIntArrayRef("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::special_logsumexp(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1486() { // special_expit
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::special_expit(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1487() { // special_sinc
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::special_sinc(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1488() { // special_round
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::special_round(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1489() { // special_log1p
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::special_log1p(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1490() { // special_log_softmax
      int64_t dim = readAttribute<int64_t>("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::special_log_softmax(self, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1491() { // special_gammainc
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::special_gammainc(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1492() { // special_gammaincc
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::special_gammaincc(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1493() { // special_multigammaln
      int64_t p = readAttribute<int64_t>("p");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::special_multigammaln(self, p);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1494() { // fft_fft
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::fft_fft(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1495() { // fft_ifft
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::fft_ifft(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1496() { // fft_rfft
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::fft_rfft(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1497() { // fft_irfft
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::fft_irfft(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1498() { // fft_hfft
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::fft_hfft(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1499() { // fft_ihfft
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::fft_ihfft(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1500() { // fft_fft2
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::fft_fft2(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1501() { // fft_ifft2
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::fft_ifft2(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1502() { // fft_rfft2
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::fft_rfft2(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1503() { // fft_irfft2
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::fft_irfft2(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1504() { // fft_fftn
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::fft_fftn(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1505() { // fft_ifftn
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::fft_ifftn(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1506() { // fft_rfftn
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::fft_rfftn(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1507() { // fft_irfftn
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::fft_irfftn(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1508() { // fft_fftshift
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::fft_fftshift(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1509() { // fft_ifftshift
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::fft_ifftshift(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1510() { // linalg_cholesky_ex
      bool upper = readAttribute<int64_t>("upper");
      bool check_errors = readAttribute<int64_t>("check_errors");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::linalg_cholesky_ex(self, upper, check_errors);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1511() { // linalg_cholesky_ex
      bool upper = readAttribute<int64_t>("upper");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::linalg_cholesky_ex(self, upper);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1512() { // linalg_cholesky_ex
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::linalg_cholesky_ex(self);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1513() { // linalg_cholesky
      bool upper = readAttribute<int64_t>("upper");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::linalg_cholesky(self, upper);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1514() { // linalg_cholesky
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::linalg_cholesky(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1515() { // linalg_det
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::linalg_det(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1516() { // det
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::det(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1517() { // _det_lu_based_helper
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::_det_lu_based_helper(self);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1518() { // _det_lu_based_helper_backward_helper
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto det_grad = peek(0, 5);
          auto det = peek(1, 5);
          auto self = peek(2, 5);
          auto lu = peek(3, 5);
          auto pivs = peek(4, 5);
          auto the_result = at::_det_lu_based_helper_backward_helper(det_grad, det, self, lu, pivs);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1519() { // linalg_lstsq
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto b = peek(1, 2);
          auto the_result = at::linalg_lstsq(self, b);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
            if(OutputSize() > 3) {assignTo(Output(3),::std::get<3>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1520() { // linalg_matmul
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::linalg_matmul(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1521() { // linalg_slogdet
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::linalg_slogdet(self);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1522() { // linalg_eig
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::linalg_eig(self);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1523() { // linalg_eigvals
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::linalg_eigvals(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1524() { // linalg_eigh
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::linalg_eigh(self);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1525() { // linalg_eigvalsh
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::linalg_eigvalsh(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1526() { // linalg_householder_product
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 2);
          auto tau = peek(1, 2);
          auto the_result = at::linalg_householder_product(input, tau);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1527() { // linalg_inv_ex
      bool check_errors = readAttribute<int64_t>("check_errors");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::linalg_inv_ex(self, check_errors);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1528() { // linalg_inv_ex
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::linalg_inv_ex(self);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1529() { // linalg_inv
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::linalg_inv(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1530() { // inner
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::inner(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1531() { // outer
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto vec2 = peek(1, 2);
          auto the_result = at::outer(self, vec2);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1532() { // ger
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto vec2 = peek(1, 2);
          auto the_result = at::ger(self, vec2);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1533() { // linalg_norm
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::linalg_norm(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1534() { // linalg_vector_norm
      at::Scalar ord = readScalarAttribute("ord");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::linalg_vector_norm(self, ord);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1535() { // linalg_vector_norm
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::linalg_vector_norm(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1536() { // linalg_matrix_norm
      at::Scalar ord = readScalarAttribute("ord");
      auto dim = readIntArrayRef("dim");
      bool keepdim = readAttribute<int64_t>("keepdim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::linalg_matrix_norm(self, ord, dim, keepdim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1537() { // linalg_matrix_norm
      at::Scalar ord = readScalarAttribute("ord");
      auto dim = readIntArrayRef("dim");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::linalg_matrix_norm(self, ord, dim);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1538() { // linalg_matrix_norm
      at::Scalar ord = readScalarAttribute("ord");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::linalg_matrix_norm(self, ord);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1539() { // linalg_matrix_norm
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::linalg_matrix_norm(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1540() { // linalg_svd
      bool full_matrices = readAttribute<int64_t>("full_matrices");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::linalg_svd(self, full_matrices);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1541() { // linalg_svd
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::linalg_svd(self);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
            if(OutputSize() > 2) {assignTo(Output(2),::std::get<2>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1542() { // linalg_svdvals
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 1);
          auto the_result = at::linalg_svdvals(input);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1543() { // linalg_cond
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::linalg_cond(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1544() { // linalg_pinv
      double rcond = readAttribute<float>("rcond");
      bool hermitian = readAttribute<int64_t>("hermitian");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::linalg_pinv(self, rcond, hermitian);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1545() { // linalg_pinv
      double rcond = readAttribute<float>("rcond");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::linalg_pinv(self, rcond);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1546() { // linalg_pinv
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::linalg_pinv(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1547() { // linalg_pinv
      bool hermitian = readAttribute<int64_t>("hermitian");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto rcond = peek(1, 2);
          auto the_result = at::linalg_pinv(self, rcond, hermitian);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1548() { // linalg_pinv
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto rcond = peek(1, 2);
          auto the_result = at::linalg_pinv(self, rcond);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1549() { // linalg_solve
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::linalg_solve(input, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1550() { // linalg_tensorinv
      int64_t ind = readAttribute<int64_t>("ind");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::linalg_tensorinv(self, ind);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1551() { // linalg_tensorinv
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::linalg_tensorinv(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1552() { // linalg_tensorsolve
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::linalg_tensorsolve(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1553() { // linalg_qr
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::linalg_qr(self);
            if(OutputSize() > 0) {assignTo(Output(0),::std::get<0>(the_result));}
            if(OutputSize() > 1) {assignTo(Output(1),::std::get<1>(the_result));}
          return true;
      };
  }
  C10_NOINLINE void implementation_1554() { // linalg_matrix_power
      int64_t n = readAttribute<int64_t>("n");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::linalg_matrix_power(self, n);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1555() { // linalg_matrix_rank
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 1);
          auto the_result = at::linalg_matrix_rank(self);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1556() { // linalg_matrix_rank
      bool hermitian = readAttribute<int64_t>("hermitian");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 2);
          auto tol = peek(1, 2);
          auto the_result = at::linalg_matrix_rank(input, tol, hermitian);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1557() { // linalg_matrix_rank
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto input = peek(0, 2);
          auto tol = peek(1, 2);
          auto the_result = at::linalg_matrix_rank(input, tol);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1558() { // linalg_multi_dot
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::linalg_multi_dot(tensors);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1559() { // _test_serialization_subcmul
      at::Scalar alpha = readScalarAttribute("alpha");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::_test_serialization_subcmul(self, other, alpha);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1560() { // _test_serialization_subcmul
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto self = peek(0, 2);
          auto other = peek(1, 2);
          auto the_result = at::_test_serialization_subcmul(self, other);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1561() { // _test_string_default
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto dummy = peek(0, 1);
          auto the_result = at::_test_string_default(dummy);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1562() { // _test_ambiguous_defaults
      int64_t a = readAttribute<int64_t>("a");
      int64_t b = readAttribute<int64_t>("b");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto dummy = peek(0, 1);
          auto the_result = at::_test_ambiguous_defaults(dummy, a, b);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1563() { // _test_ambiguous_defaults
      int64_t a = readAttribute<int64_t>("a");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto dummy = peek(0, 1);
          auto the_result = at::_test_ambiguous_defaults(dummy, a);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1564() { // _test_ambiguous_defaults
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto dummy = peek(0, 1);
          auto the_result = at::_test_ambiguous_defaults(dummy);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1565() { // pad_sequence
      bool batch_first = readAttribute<int64_t>("batch_first");
      double padding_value = readAttribute<float>("padding_value");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto sequences = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::pad_sequence(sequences, batch_first, padding_value);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1566() { // pad_sequence
      bool batch_first = readAttribute<int64_t>("batch_first");
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto sequences = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::pad_sequence(sequences, batch_first);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1567() { // pad_sequence
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto sequences = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::pad_sequence(sequences);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1568() { // flatten_dense_tensors
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto tensors = peekSlice(0, InputSize() - 0, InputSize());
          auto the_result = at::flatten_dense_tensors(tensors);
            if(OutputSize() > 0) {assignTo(Output(0),the_result);}
          return true;
      };
  }
  C10_NOINLINE void implementation_1569() { // unflatten_dense_tensors
  
      run_op = [=] {
          at::AutoDispatchBelowAutograd guard;
          auto flat = peek(0, InputSize());
          auto tensors = peekSlice(1, InputSize() - 1, InputSize());
          auto the_result = at::unflatten_dense_tensors(flat, tensors);
            if(OutputSize() > 0) {assignListStartingAt(0, the_result);}
          return true;
      };
  }
};

}
