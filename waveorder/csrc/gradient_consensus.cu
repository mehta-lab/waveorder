#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/complex.h>
#include <cufft.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>

namespace waveorder_gradient_consensus {
namespace {

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 4096;
constexpr float kEpsilon = 1e-12f;

void fft_check(cufftResult result, const char* label) {
  TORCH_CHECK(result == CUFFT_SUCCESS, label, " failed with cuFFT code ", static_cast<int>(result));
}

int blocks_for(int64_t n) {
  return static_cast<int>(std::min<int64_t>((n - 1) / kThreads + 1, kMaxBlocks));
}

int64_t product(int64_t a, int64_t b) {
  TORCH_CHECK(a > 0 && b > 0 && a <= std::numeric_limits<int64_t>::max() / b,
              "Volume dimensions overflow int64 addressing");
  return a * b;
}

void check_tensor(const at::Tensor& tensor, at::ScalarType type, const char* name) {
  TORCH_CHECK(tensor.defined() && tensor.is_cuda() && tensor.layout() == at::kStrided &&
              tensor.dim() == 3 && tensor.scalar_type() == type && tensor.is_contiguous() &&
              !tensor.is_conj() && !tensor.is_neg() && !tensor.requires_grad(),
              name, " must be contiguous CUDA ZYX with the expected dtype and no gradient");
}

bool overlaps(const at::Tensor& a, const at::Tensor& b) {
  const auto pa = reinterpret_cast<std::uintptr_t>(a.const_data_ptr());
  const auto pb = reinterpret_cast<std::uintptr_t>(b.const_data_ptr());
  return pa <= pb ? pb - pa < a.nbytes() : pa - pb < b.nbytes();
}

cufftComplex* complex_data(at::Tensor& tensor) {
  static_assert(sizeof(c10::complex<float>) == sizeof(cufftComplex));
  return reinterpret_cast<cufftComplex*>(tensor.data_ptr<c10::complex<float>>());
}

struct Resources {
  int device;
  cufftHandle forward = 0, backward = 0;
  cudaEvent_t finished = nullptr;

  explicit Resources(int index) : device(index) {}
  Resources(const Resources&) = delete;
  Resources& operator=(const Resources&) = delete;

  void initialize() {
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&finished, cudaEventDisableTiming));
    for (auto* plan : {&forward, &backward}) {
      fft_check(cufftCreate(plan), "cufftCreate");
      fft_check(cufftSetAutoAllocation(*plan, 0), "cufftSetAutoAllocation");
    }
  }

  ~Resources() noexcept {
    if (!finished && !forward && !backward) return;
    int old = -1;
    const bool had_device = cudaGetDevice(&old) == cudaSuccess;
    if (!had_device || old != device) cudaSetDevice(device);
    if (finished) cudaEventSynchronize(finished);
    if (forward) cufftDestroy(forward);
    if (backward) cufftDestroy(backward);
    if (finished) cudaEventDestroy(finished);
    if (had_device && old != device) cudaSetDevice(old);
  }
};

struct Completion {
  cudaEvent_t event;
  cudaStream_t stream;
  bool recorded = false;
  void record() {
    C10_CUDA_CHECK(cudaEventRecord(event, stream));
    recorded = true;
  }
  ~Completion() noexcept {
    if (!recorded && cudaEventRecord(event, stream) != cudaSuccess) cudaStreamSynchronize(stream);
  }
};

__global__ void fill_constant(float* output, int64_t count, float value) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += static_cast<int64_t>(blockDim.x) * gridDim.x) output[i] = value;
}


__global__ void spectral_product(cufftComplex* spectrum, const cufftComplex* transfer,
                                 int64_t count, bool transpose, float inverse_size) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const cufftComplex a = spectrum[i], h = transfer[i];
    const float im = transpose ? -h.y : h.y;
    spectrum[i] = {(a.x * h.x - a.y * im) * inverse_size,
                   (a.x * im + a.y * h.x) * inverse_size};
  }
}

__global__ void spectral_power_product(cufftComplex* spectrum, const cufftComplex* transfer,
                                       int64_t count, float inverse_size) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const cufftComplex h = transfer[i];
    const float power = (h.x * h.x + h.y * h.y) * inverse_size;
    spectrum[i].x *= power;
    spectrum[i].y *= power;
  }
}


__global__ void pack_ratio(const float* measured, const float* rates, float* output,
                           int64_t count, float background, float offset) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float rate = rates[i] + background;
    output[i] = measured[i] / (rate < kEpsilon ? kEpsilon : rate) - offset;
  }
}

__global__ void pack_local_product(const float* gradient, const float* heads_gradient,
                                   float* output, int64_t count) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float head = heads_gradient[i];
    output[i] = head * (gradient[i] - head);
  }
}

__global__ void update(const float* previous, const float* gradient,
                       const float* neighborhood, float* updated,
                       double* partials, int64_t count, float denominator) {
  double active = 0.0, change = 0.0, scale = 0.0;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float step = neighborhood[i] <= 0.0f ? 0.0f : previous[i] / denominator;
    const float raw = previous[i] + gradient[i] * step;
    const float next = raw < kEpsilon ? kEpsilon : raw;
    updated[i] = next;
    active += step != 0.0f ? 1.0 : 0.0;
    const double delta = static_cast<double>(next) - static_cast<double>(previous[i]);
    change += delta * delta;
    scale += static_cast<double>(previous[i]) * static_cast<double>(previous[i]);
  }
  __shared__ double sums[3][kThreads];
  sums[0][threadIdx.x] = active;
  sums[1][threadIdx.x] = change;
  sums[2][threadIdx.x] = scale;
  __syncthreads();
  for (int offset = kThreads / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset)
      for (int k = 0; k < 3; ++k) sums[k][threadIdx.x] += sums[k][threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    for (int k = 0; k < 3; ++k) partials[static_cast<int64_t>(blockIdx.x) * 3 + k] = sums[k][0];
}

__global__ void finish_reduction(const double* partials, double* summary, int blocks) {
  double local[3] = {0.0, 0.0, 0.0};
  for (int i = threadIdx.x; i < blocks; i += kThreads)
    for (int k = 0; k < 3; ++k) local[k] += partials[static_cast<int64_t>(i) * 3 + k];
  __shared__ double sums[3][kThreads];
  for (int k = 0; k < 3; ++k) sums[k][threadIdx.x] = local[k];
  __syncthreads();
  for (int offset = kThreads / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset)
      for (int k = 0; k < 3; ++k) sums[k][threadIdx.x] += sums[k][threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0) for (int k = 0; k < 3; ++k) summary[k] = sums[k][0];
}

__global__ void crop_output(const float* estimate, float* output, int64_t count, int64_t offset) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += static_cast<int64_t>(blockDim.x) * gridDim.x)
    output[i] = estimate[offset + i];
}

}  // namespace

enum class SpectralFilter { Forward, Adjoint, Power };

struct State {
  int device;
  int64_t z, y, x, half_x, count, half_count;
  int blocks;
  float inverse_size, normalization, forward_dc;
  bool first_step = true;
  at::Tensor transfer, spectrum, estimate, rates, gradient, heads_gradient;
  at::Tensor partials, summary, fft_workspace;
  Resources resources;

  State(const at::Tensor& compact, int64_t logical_x)
      : device(compact.defined() && compact.is_cuda() ? compact.get_device() : -1),
        resources(device) {
    check_tensor(compact, at::kComplexFloat, "transfer");
    z = compact.size(0); y = compact.size(1); x = logical_x;
    TORCH_CHECK(z > 0 && y > 0 && x > 0 && compact.size(2) == x / 2 + 1,
                "transfer half-spectrum does not match the logical shape");
    half_x = x / 2 + 1;
    count = product(product(z, y), x);
    half_count = product(product(z, y), half_x);
    TORCH_CHECK(half_count <= std::numeric_limits<int64_t>::max() / 8,
                "Compact spectrum exceeds int64 byte addressing");
    blocks = blocks_for(count);
    inverse_size = static_cast<float>(1.0 / static_cast<double>(count));
    const c10::cuda::CUDAGuard guard(compact.device());
    auto current = c10::cuda::getCurrentCUDAStream(device);
    const cudaStream_t stream = current.stream();
    resources.initialize();
    Completion completion{resources.finished, stream};
    auto options = at::TensorOptions().device(compact.device()).dtype(at::kFloat).requires_grad(false);
    transfer = compact;
    spectrum = at::empty({z, y, half_x}, options.dtype(at::kComplexFloat));
    estimate = at::empty({z, y, x}, options);
    rates = at::empty({z, y, x}, options);
    gradient = at::empty({z, y, x}, options);
    heads_gradient = at::empty({z, y, x}, options);
    partials = at::empty({blocks * 3}, options.dtype(at::kDouble));
    summary = at::empty({3}, options.dtype(at::kDouble));
    long long dimensions[3] = {z, y, x};
    long long real_embed[3] = {z, y, x};
    long long complex_embed[3] = {z, y, half_x};
    std::size_t forward_bytes = 0, backward_bytes = 0;
    fft_check(cufftMakePlanMany64(resources.forward, 3, dimensions, real_embed, 1,
                                  count, complex_embed, 1, half_count,
                                  CUFFT_R2C, 1, &forward_bytes), "R2C plan");
    fft_check(cufftMakePlanMany64(resources.backward, 3, dimensions, complex_embed, 1,
                                  half_count, real_embed, 1, count,
                                  CUFFT_C2R, 1, &backward_bytes), "C2R plan");
    const auto workspace_bytes = std::max(forward_bytes, backward_bytes);
    TORCH_CHECK(workspace_bytes <= static_cast<std::size_t>(std::numeric_limits<int64_t>::max()),
                "cuFFT workspace exceeds int64 tensor size");
    fft_workspace = at::empty({static_cast<int64_t>(workspace_bytes)}, options.dtype(at::kByte));
    for (const cufftHandle plan : {resources.forward, resources.backward}) {
      fft_check(cufftSetWorkArea(plan, fft_workspace.data_ptr()), "cufftSetWorkArea");
      fft_check(cufftSetStream(plan, stream), "cufftSetStream");
    }
    cufftComplex dc{};
    C10_CUDA_CHECK(cudaMemcpyAsync(&dc, complex_data(transfer), sizeof(dc), cudaMemcpyDeviceToHost, stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    forward_dc = dc.x;
    normalization = dc.x < kEpsilon ? kEpsilon : dc.x;
    completion.record();
  }

  void use_stream(cudaStream_t stream) {
    C10_CUDA_CHECK(cudaStreamWaitEvent(stream, resources.finished, 0));
    fft_check(cufftSetStream(resources.forward, stream), "cufftSetStream R2C");
    fft_check(cufftSetStream(resources.backward, stream), "cufftSetStream C2R");
  }

  void convolve(const at::Tensor& input, at::Tensor& output, SpectralFilter filter, cudaStream_t stream) {
    auto* fft_buffer = complex_data(spectrum);
    fft_check(cufftExecR2C(resources.forward, input.data_ptr<float>(), fft_buffer), "cufftExecR2C");
    if (filter == SpectralFilter::Power) {
      spectral_power_product<<<blocks_for(half_count), kThreads, 0, stream>>>(
          fft_buffer, complex_data(transfer), half_count, inverse_size);
    } else {
      spectral_product<<<blocks_for(half_count), kThreads, 0, stream>>>(
          fft_buffer, complex_data(transfer), half_count, filter == SpectralFilter::Adjoint, inverse_size);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    fft_check(cufftExecC2R(resources.backward, fft_buffer, output.data_ptr<float>()), "cufftExecC2R");
  }

  void convolve_ratio(const at::Tensor& input, at::Tensor& output,
                      float background, float offset, cudaStream_t stream) {
    pack_ratio<<<blocks, kThreads, 0, stream>>>(
        input.data_ptr<float>(), rates.data_ptr<float>(), output.data_ptr<float>(),
        count, background, offset);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    convolve(output, output, SpectralFilter::Adjoint, stream);
  }

  void reset() {
    const c10::cuda::CUDAGuard guard(transfer.device());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device).stream();
    use_stream(stream);
    Completion completion{resources.finished, stream};
    fill_constant<<<blocks, kThreads, 0, stream>>>(estimate.data_ptr<float>(), count, 1.0f);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    first_step = true;
    completion.record();
  }

  std::tuple<bool, double> step(const at::Tensor& measured, const at::Tensor& heads, double background) {
    check_tensor(measured, at::kFloat, "measured");
    check_tensor(heads, at::kFloat, "heads");
    for (const auto& input : {measured, heads})
      TORCH_CHECK(input.get_device() == device && input.size(0) == z &&
                  input.size(1) == y && input.size(2) == x, "measured and heads must match the Plan shape/device");
    TORCH_CHECK(std::isfinite(background) && std::abs(background) <= std::numeric_limits<float>::max(),
                "background must be a finite float32 scalar");
    const c10::cuda::CUDAGuard guard(transfer.device());
    const auto current = c10::cuda::getCurrentCUDAStream(device);
    const cudaStream_t stream = current.stream();
    use_stream(stream);
    Completion completion{resources.finished, stream};
    c10::cuda::CUDACachingAllocator::recordStream(measured.storage().data_ptr(), current);
    c10::cuda::CUDACachingAllocator::recordStream(heads.storage().data_ptr(), current);
    if (first_step) {
      fill_constant<<<blocks, kThreads, 0, stream>>>(rates.data_ptr<float>(), count, forward_dc);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      first_step = false;
    } else {
      convolve(estimate, rates, SpectralFilter::Forward, stream);
    }
    convolve_ratio(measured, gradient, static_cast<float>(background), 1.0f, stream);
    convolve_ratio(heads, heads_gradient, static_cast<float>(background), 0.5f, stream);
    pack_local_product<<<blocks, kThreads, 0, stream>>>(
        gradient.data_ptr<float>(), heads_gradient.data_ptr<float>(), rates.data_ptr<float>(), count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    // Both ratios are finished. Reuse rates for the consensus mask and then the next estimate.
    convolve(rates, rates, SpectralFilter::Power, stream);
    update<<<blocks, kThreads, 0, stream>>>(estimate.data_ptr<float>(), gradient.data_ptr<float>(),
                                            rates.data_ptr<float>(), rates.data_ptr<float>(),
                                            partials.data_ptr<double>(), count, normalization);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    finish_reduction<<<1, kThreads, 0, stream>>>(partials.data_ptr<double>(), summary.data_ptr<double>(), blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    double metrics[3];
    C10_CUDA_CHECK(cudaMemcpyAsync(metrics, summary.data_ptr<double>(), sizeof(metrics), cudaMemcpyDeviceToHost, stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    std::swap(estimate, rates);
    completion.record();
    return {metrics[0] == 0.0, metrics[2] > 0.0 ? std::sqrt(metrics[1] / metrics[2]) : std::numeric_limits<double>::infinity()};
  }

  void copy(const at::Tensor& output, int64_t padding) {
    check_tensor(output, at::kFloat, "output");
    TORCH_CHECK(padding >= 0 && padding <= (z - 1) / 2 && output.get_device() == device &&
                output.size(0) == z - 2 * padding && output.size(1) == y && output.size(2) == x,
                "output must match the unpadded Plan shape/device");
    for (const auto& private_buffer : {transfer, spectrum, estimate, rates, gradient,
                                       heads_gradient, partials, summary, fft_workspace})
      TORCH_CHECK(!overlaps(output, private_buffer), "output must not overlap Plan storage");
    const c10::cuda::CUDAGuard guard(transfer.device());
    const auto current = c10::cuda::getCurrentCUDAStream(device);
    const cudaStream_t stream = current.stream();
    use_stream(stream);
    Completion completion{resources.finished, stream};
    c10::cuda::CUDACachingAllocator::recordStream(output.storage().data_ptr(), current);
    const int64_t out_count = product(z - 2 * padding, product(y, x));
    crop_output<<<blocks_for(out_count), kThreads, 0, stream>>>(estimate.data_ptr<float>(),
        output.data_ptr<float>(), out_count, padding * y * x);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    completion.record();
  }
};

State* create_plan(const at::Tensor& transfer, int64_t logical_x) { return new State(transfer, logical_x); }
void destroy_plan(State* state) noexcept { delete state; }
void reset_plan(State* state) { state->reset(); }
std::tuple<bool, double> step_plan(State* state, const at::Tensor& measured,
                                   const at::Tensor& heads, double background) {
  return state->step(measured, heads, background);
}
void copy_output(State* state, const at::Tensor& output, int64_t padding) { state->copy(output, padding); }

}  // namespace waveorder_gradient_consensus
