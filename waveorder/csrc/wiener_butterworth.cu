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

namespace waveorder_wiener_butterworth {
namespace {
constexpr int kThreads = 256;
constexpr int kMaxBlocks = 4096;
constexpr float kEps = 1e-12f;

void fft_check(cufftResult status, const char* name) {
  TORCH_CHECK(status == CUFFT_SUCCESS, name, " failed (cuFFT code ", int(status), ")");
}
int blocks(int64_t count) {
  return static_cast<int>(std::min<int64_t>((count - 1) / kThreads + 1, kMaxBlocks));
}
int64_t product(int64_t a, int64_t b) {
  TORCH_CHECK(a > 0 && b > 0 && a <= std::numeric_limits<int64_t>::max() / b,
              "Volume dimensions overflow int64");
  return a * b;
}
void check_tensor(const at::Tensor& tensor, at::ScalarType type, const char* name) {
  TORCH_CHECK(tensor.defined() && tensor.is_cuda() && tensor.dim() == 3 &&
                  tensor.scalar_type() == type && tensor.layout() == at::kStrided &&
                  tensor.is_contiguous() && !tensor.is_conj() && !tensor.is_neg() &&
                  !tensor.requires_grad(), name, " must be physical contiguous CUDA ZYX without gradients");
}
bool overlaps(const at::Tensor& a, const at::Tensor& b) {
  auto x = reinterpret_cast<std::uintptr_t>(a.const_data_ptr());
  auto y = reinterpret_cast<std::uintptr_t>(b.const_data_ptr());
  return x <= y ? y - x < a.nbytes() : x - y < b.nbytes();
}
cufftComplex* complex_data(const at::Tensor& tensor) {
  static_assert(sizeof(c10::complex<float>) == sizeof(cufftComplex));
  return reinterpret_cast<cufftComplex*>(tensor.data_ptr<c10::complex<float>>());
}

// Native workspace uses out-of-place transforms: reusable work, plus
// measured and estimate volumes only when later iterations need them. One
// half spectrum and a shared cuFFT work area serve both plans. The first
// ratio uses H(ones) at DC without a forward transform.
__global__ void multiply(cufftComplex* spectrum, const cufftComplex* filter, int64_t count) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const cufftComplex a = spectrum[i], b = filter[i];
    spectrum[i] = {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
  }
}
__global__ void initial_ratio_kernel(const float* input, float* measured, float* estimate,
                                     float* work, const cufftComplex* h, int64_t count,
                                     int64_t z_size, int64_t plane, int64_t y_size, int64_t x_size,
                                     int64_t padding, float background) {
  // Circular H(ones) is constant, including when the forward PSF is not normalized.
  const float rate = h[0].x + background;
  const float denominator = rate < kEps ? kEps : rate;
  for (int64_t row = blockIdx.x; row < count / x_size; row += gridDim.x) {
    int64_t z = row / y_size - padding;
    const bool halo = z < 0 || z >= z_size;
    const bool zero_halo = halo && padding >= z_size;
    if (halo && !zero_halo) z = z < 0 ? -z - 1 : 2 * z_size - z - 1;
    const float* source = zero_halo ? nullptr : input + z * plane + (row % y_size) * x_size;
    const int64_t offset = row * x_size;
    for (int64_t column = threadIdx.x; column < x_size; column += blockDim.x) {
      const int64_t index = offset + column;
      const float value = source ? source[column] : 0.0f;
      const float clamped = value < 0.0f ? 0.0f : value;
      if (measured) measured[index] = clamped;
      if (estimate) estimate[index] = 1.0f;
      work[index] = clamped / denominator - 1.0f;
    }
  }
}

__global__ void ratio_kernel(float* work, const float* measured, int64_t count,
                             float inverse_size, float background) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float rate = work[i] * inverse_size + background;
    work[i] = measured[i] / (rate < kEps ? kEps : rate) - 1.0f;
  }
}
__global__ void update_kernel(float* estimate, const float* gradient, const cufftComplex* b,
                              int64_t count, float inverse_size) {
  // H_T(1) is spatially constant for circular convolution. With Hermitian B,
  // it is real(B[0]), floored exactly as rlgc.clip(transpose(ones)).
  const float dc = b[0].x;
  const float normalizer = dc < kEps ? kEps : dc;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float old = estimate[i];
    const float updated = old + (gradient[i] * inverse_size) * (old / normalizer);
    estimate[i] = updated < kEps ? kEps : updated;
  }
}
__global__ void update_crop_first_kernel(const float* gradient, float* output, const cufftComplex* b,
                                         int64_t count, int64_t offset, float inverse_size) {
  const float dc = b[0].x;
  const float normalizer = dc < kEps ? kEps : dc;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float updated = 1.0f + (gradient[i + offset] * inverse_size) * (1.0f / normalizer);
    output[i] = updated < kEps ? kEps : updated;
  }
}

__global__ void crop_kernel(const float* estimate, float* output, int64_t count, int64_t offset) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += static_cast<int64_t>(blockDim.x) * gridDim.x)
    output[i] = estimate[i + offset];
}

struct Resources {
  int device;
  cufftHandle forward = 0, backward = 0;
  cudaEvent_t finished = nullptr;
  explicit Resources(int d) : device(d) {}
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
    if (!forward && !backward && !finished) return;
    int previous = -1;
    const bool have_previous = cudaGetDevice(&previous) == cudaSuccess;
    if (!have_previous || previous != device) cudaSetDevice(device);
    if (finished) cudaEventSynchronize(finished);
    if (forward) cufftDestroy(forward);
    if (backward) cufftDestroy(backward);
    if (finished) cudaEventDestroy(finished);
    if (have_previous && previous != device) cudaSetDevice(previous);
  }
};
struct RecordCompletion {
  cudaEvent_t event;
  cudaStream_t stream;
  bool recorded = false;
  void record() {
    C10_CUDA_CHECK(cudaEventRecord(event, stream));
    recorded = true;
  }
  ~RecordCompletion() noexcept {
    if (!recorded && cudaEventRecord(event, stream) != cudaSuccess) cudaStreamSynchronize(stream);
  }
};
}  // namespace

struct State {
  int device;
  int64_t padded_z, z_size, y_size, x_size, padding, iterations;
  int64_t count, output_count, half_count, plane;
  float background, inverse_size;
  std::size_t fft_workspace_bytes = 0;
  at::Tensor h, b, measured, estimate, work, spectrum, fft_workspace;
  Resources resources;

  State(const at::Tensor& forward, const at::Tensor& transpose, int64_t logical_x,
        int64_t z_padding, int64_t num_iterations, float bg)
      : device(forward.defined() ? forward.get_device() : -1), resources(device) {
    check_tensor(forward, at::kComplexFloat, "h");
    check_tensor(transpose, at::kComplexFloat, "b");
    TORCH_CHECK(forward.sizes() == transpose.sizes() && forward.get_device() == transpose.get_device(),
                "h and b must have identical shape and device");
    padded_z = forward.size(0);
    y_size = forward.size(1);
    x_size = logical_x;
    padding = z_padding;
    iterations = num_iterations;
    background = bg;
    TORCH_CHECK(padded_z > 0 && y_size > 0 && x_size > 0 && padding >= 0 &&
                padding <= (padded_z - 1) / 2 && iterations >= 1 && std::isfinite(bg),
                "invalid plan dimensions, padding, iterations, or background");
    TORCH_CHECK(forward.size(2) == x_size / 2 + 1, "filter half-spectrum does not match logical_x");
    z_size = padded_z - 2 * padding;
    plane = product(y_size, x_size);
    count = product(padded_z, plane);
    output_count = product(z_size, plane);
    half_count = product(product(padded_z, y_size), x_size / 2 + 1);
    TORCH_CHECK(count <= std::numeric_limits<int64_t>::max() / 4 &&
                half_count <= std::numeric_limits<int64_t>::max() / 8, "workspace byte size overflow");
    inverse_size = static_cast<float>(1.0 / static_cast<double>(count));
    const c10::cuda::CUDAGuard guard(forward.device());
    const auto stream = c10::cuda::getCurrentCUDAStream(device).stream();
    resources.initialize();
    RecordCompletion completion{resources.finished, stream};
    const auto options = at::TensorOptions().device(forward.device()).dtype(at::kFloat).requires_grad(false);
    h = forward;
    b = transpose;
    if (iterations > 1) {
      measured = at::empty({padded_z, y_size, x_size}, options);
      estimate = at::empty({padded_z, y_size, x_size}, options);
    }
    work = at::empty({padded_z, y_size, x_size}, options);
    spectrum = at::empty({padded_z, y_size, x_size / 2 + 1}, options.dtype(at::kComplexFloat));
    long long dims[3] = {padded_z, y_size, x_size};
    long long real_embed[3] = {padded_z, y_size, x_size};
    long long complex_embed[3] = {padded_z, y_size, x_size / 2 + 1};
    std::size_t forward_bytes = 0, backward_bytes = 0;
    fft_check(cufftMakePlanMany64(resources.forward, 3, dims, real_embed, 1, count,
                                  complex_embed, 1, half_count, CUFFT_R2C, 1, &forward_bytes), "R2C plan");
    fft_check(cufftMakePlanMany64(resources.backward, 3, dims, complex_embed, 1, half_count,
                                  real_embed, 1, count, CUFFT_C2R, 1, &backward_bytes), "C2R plan");
    fft_workspace_bytes = std::max(forward_bytes, backward_bytes);
    TORCH_CHECK(fft_workspace_bytes <= static_cast<std::size_t>(std::numeric_limits<int64_t>::max()),
                "cuFFT workspace too large");
    fft_workspace = at::empty({static_cast<int64_t>(fft_workspace_bytes)}, options.dtype(at::kByte));
    for (auto plan : {resources.forward, resources.backward}) {
      fft_check(cufftSetWorkArea(plan, fft_workspace.data_ptr()), "cufftSetWorkArea");
      fft_check(cufftSetStream(plan, stream), "cufftSetStream");
    }
    completion.record();
    C10_CUDA_CHECK(cudaEventSynchronize(resources.finished));
  }

  at::Tensor run(const at::Tensor& input, const at::Tensor& output) {
    check_tensor(input, at::kFloat, "input");
    check_tensor(output, at::kFloat, "output");
    for (const auto& tensor : {input, output})
      TORCH_CHECK(tensor.get_device() == device && tensor.size(0) == z_size &&
                  tensor.size(1) == y_size && tensor.size(2) == x_size,
                  "input and output must match plan device and unpadded shape");
    TORCH_CHECK(!overlaps(input, output), "output must not overlap input");
    for (const auto& tensor : {h, b, measured, estimate, work, spectrum, fft_workspace})
      if (tensor.defined()) TORCH_CHECK(!overlaps(output, tensor), "output must not overlap private plan storage");
    const c10::cuda::CUDAGuard guard(input.device());
    const auto current = c10::cuda::getCurrentCUDAStream(device);
    const auto stream = current.stream();
    C10_CUDA_CHECK(cudaStreamWaitEvent(stream, resources.finished, 0));
    RecordCompletion completion{resources.finished, stream};
    c10::cuda::CUDACachingAllocator::recordStream(input.storage().data_ptr(), current);
    c10::cuda::CUDACachingAllocator::recordStream(output.storage().data_ptr(), current);
    fft_check(cufftSetStream(resources.forward, stream), "cufftSetStream R2C");
    fft_check(cufftSetStream(resources.backward, stream), "cufftSetStream C2R");
    initial_ratio_kernel<<<static_cast<int>(std::min<int64_t>(count / x_size, blocks(count))),
                           kThreads, 0, stream>>>(
        input.data_ptr<float>(), iterations > 1 ? measured.data_ptr<float>() : nullptr,
        iterations > 1 ? estimate.data_ptr<float>() : nullptr, work.data_ptr<float>(),
        complex_data(h), count, z_size, plane, y_size, x_size, padding, background);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto* frequency = complex_data(spectrum);
    for (int64_t n = 0; n < iterations; ++n) {
      if (n > 0) {
        fft_check(cufftExecR2C(resources.forward, estimate.data_ptr<float>(), frequency), "cufftExecR2C forward");
        multiply<<<blocks(half_count), kThreads, 0, stream>>>(frequency, complex_data(h), half_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        fft_check(cufftExecC2R(resources.backward, frequency, work.data_ptr<float>()), "cufftExecC2R forward");
        ratio_kernel<<<blocks(count), kThreads, 0, stream>>>(work.data_ptr<float>(), measured.data_ptr<float>(),
                                                             count, inverse_size, background);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
      fft_check(cufftExecR2C(resources.forward, work.data_ptr<float>(), frequency), "cufftExecR2C transpose");
      multiply<<<blocks(half_count), kThreads, 0, stream>>>(frequency, complex_data(b), half_count);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      fft_check(cufftExecC2R(resources.backward, frequency, work.data_ptr<float>()), "cufftExecC2R transpose");
      if (iterations == 1) {
        update_crop_first_kernel<<<blocks(output_count), kThreads, 0, stream>>>(
            work.data_ptr<float>(), output.data_ptr<float>(), complex_data(b), output_count,
            padding * plane, inverse_size);
      } else {
        update_kernel<<<blocks(count), kThreads, 0, stream>>>(estimate.data_ptr<float>(), work.data_ptr<float>(),
                                                              complex_data(b), count, inverse_size);
      }
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (iterations > 1) {
      crop_kernel<<<blocks(output_count), kThreads, 0, stream>>>(estimate.data_ptr<float>(),
                                                                 output.data_ptr<float>(), output_count,
                                                                 padding * plane);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    completion.record();
    return output;
  }
};

State* create_plan(const at::Tensor& h, const at::Tensor& b, int64_t x, int64_t p, int64_t n, float bg) {
  return new State(h, b, x, p, n, bg);
}
void destroy_plan(State* state) noexcept { delete state; }
at::Tensor run_plan(State* state, const at::Tensor& input, const at::Tensor& output) {
  return state->run(input, output);
}
std::size_t workspace_bytes(const State* state) { return state->fft_workspace_bytes; }
std::size_t persistent_bytes(const State* state) {
  return state->h.nbytes() + state->b.nbytes() +
         (state->measured.defined() ? state->measured.nbytes() : 0) +
         (state->estimate.defined() ? state->estimate.nbytes() : 0) + state->work.nbytes() +
         state->spectrum.nbytes() + state->fft_workspace.nbytes();
}
}  // namespace waveorder_wiener_butterworth
