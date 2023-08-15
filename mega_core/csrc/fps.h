#pragma once
#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <THC/THC.h>

#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

int furthest_point_sampling(int b, int n, int m,
    at::Tensor points_tensor, at::Tensor temp_tensor, at::Tensor idx_tensor) {

  if (points_tensor.type().is_cuda()) {
#ifdef WITH_CUDA
    if (points_tensor.numel() == 0)
      return -1;
      //return at::empty({0}, points_tensor.options().dtype(at::kLong).device(at::kCPU));

    const float *points = points_tensor.data<float>();
    float *temp = temp_tensor.data<float>();
    int *idx = idx_tensor.data<int>();

    furthest_point_sampling_kernel_launcher(b, n, m, points, temp, idx);

    return 1;
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return -1;
}