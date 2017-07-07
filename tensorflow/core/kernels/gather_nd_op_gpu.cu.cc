/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/gather_nd_op.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename T, typename Index>
__global__ void GatherSliceOpKernel(
    const T* params, const Index* indices, T* out,
    const gtl::InlinedVector<int64, 8>& batch_strides,
    const gtl::InlinedVector<int64, 8>& batch_indices,
    const int64 indices_size, const int64 slice_size,
    const int64 out_size, const int ixdim) {
  // TODO(ebrevdo): reduce inner loop into two loops:
  // one over the number of locs, and one over the offsets inside the locs.
  CUDA_1D_KERNEL_LOOP(i, out_size) {
    const Index loc = i / slice_size;
    const auto indices_i = indices + ixdim * loc;
    bool out_of_bounds = false;
    Index offset = 0;
#pragma unroll
    for (int j = 0; j < ixdim; ++j) {
      const Index index_j = ldg(indices_i + j);
      out_of_bounds |= !FastBoundsCheck(index_j, batch_indices[j]);
      offset += batch_strides[j] * index_j;
    }
    // TODO(ebrevdo):
    // This is the only part that depends on the offset.  The part
    // above does not need to be executed for every index i.
    // Is there a way to break the outer loop into two loops?  One
    // that determines how many slice_size-length locs are iterated
    // over, and another that iterates over slice_size iterations for
    // the correct indices?
    const Index loc_offset = i - loc * slice_size;
    out[i] = (out_of_bounds) ? T(0) : ldg(params + offset + loc_offset);
  }
}

namespace functor {

template <typename T, typename Index>
struct GatherNdSlice<GPUDevice, T, Index> {
  Index operator()(const GPUDevice& d, const Index unused_N_result,
                   const Index unused_slice_size, int ixdim,
                   Tensor& scratch, const Tensor& params,
                   const Tensor& indices, Tensor* out) {
    const int64 indices_size = indices.dim_size(1);
    const int64 out_size = out->NumElements();
    int64 s_size = out->dim_size(1);
    gtl::InlinedVector<int64, 8> batch_strides(ixdim);
    gtl::InlinedVector<int64, 8> batch_indices(ixdim);
    if (ixdim > 0) {
      batch_strides[size_t(ixdim - 1)] = s_size;
      batch_indices[size_t(ixdim - 1)] = params.dim_size(ixdim - 1);
    }
    for (int i = ixdim - 1; i > 0; --i) {
      batch_indices[i - 1] = params.dim_size(i - 1);
      batch_strides[i - 1] = batch_strides[i] * params.dim_size(i);
    }
    CudaLaunchConfig config = GetCudaLaunchConfig(out_size, d);

    // clang-format off
    GatherSliceOpKernel<T, Index>
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
            params.flat<T>.data(), indices.flat<T>.data(),
            out->flat<T>.data(), batch_strides, batch_indices,
            indices_size, s_size, out_size, ixdim);
    // clang-format on

    // TODO(ebrevdo): enable indices validation on GPU.
    // Right now checking for indices out of bound in the kernel would
    // require copying code between GPU/CPU, and is too slow.
    return -1;
  }
};

}  // namespace functor

#define DEFINE_GPU_SPECS_INDEX(T, Index)    \
  template struct functor::GatherNdSlice<GPUDevice, T, Index>;

#define DEFINE_GPU_SPECS(T)         \
  DEFINE_GPU_SPECS_INDEX(T, int32); \
  DEFINE_GPU_SPECS_INDEX(T, int64);

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);

#undef DEFINE_GPU_SPECS
#undef DEFINE_GPU_SPECS_INDEX

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
