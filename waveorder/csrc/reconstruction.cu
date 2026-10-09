#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/complex.h>
#include <cufft.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace waveorder_reconstruction {
namespace {

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 4096;

void check_fft(cufftResult result, const char* operation) {
  TORCH_CHECK(result == CUFFT_SUCCESS, operation, " failed with cuFFT code ", static_cast<int>(result));
}

int blocks_for(int64_t count) {
  return static_cast<int>(std::min<int64_t>((count - 1) / kThreads + 1, kMaxBlocks));
}

int64_t checked_product(int64_t a, int64_t b) {
  TORCH_CHECK(a > 0 && b > 0 && a <= std::numeric_limits<int64_t>::max() / b,
              "Volume dimensions overflow int64 addressing");
  return a * b;
}

std::size_t checked_bytes_sum(std::size_t a, std::size_t b) {
  TORCH_CHECK(a <= std::numeric_limits<std::size_t>::max() - b,
              "Persistent tensor byte sizes overflow size_t");
  return a + b;
}

void check_tensor(const at::Tensor& tensor, at::ScalarType dtype, const char* name) {
  TORCH_CHECK(tensor.defined() && tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.layout() == at::kStrided && tensor.dim() == 3 &&
                  tensor.scalar_type() == dtype && tensor.is_contiguous(),
              name, " has the wrong dtype, rank, or layout");
  TORCH_CHECK(!tensor.is_conj() && !tensor.is_neg(), name, " must not have lazy math bits");
  TORCH_CHECK(!tensor.requires_grad(), name, " must not require gradients in inference reconstruction");
}

bool overlaps(const at::Tensor& first, const at::Tensor& second) {
  const auto a = reinterpret_cast<std::uintptr_t>(first.const_data_ptr());
  const auto b = reinterpret_cast<std::uintptr_t>(second.const_data_ptr());
  return a <= b ? b - a < first.nbytes() : a - b < second.nbytes();
}

// Declared after tensors so queued work finishes before owned buffers die.
struct Resources {
  int device;
  cufftHandle forward = 0, backward = 0;
  cudaEvent_t finished = nullptr;

  explicit Resources(int device_index) : device(device_index) {}
  Resources(const Resources&) = delete;
  Resources& operator=(const Resources&) = delete;

  void initialize() {
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&finished, cudaEventDisableTiming));
    for (auto* plan : {&forward, &backward}) {
      check_fft(cufftCreate(plan), "cufftCreate");
      check_fft(cufftSetAutoAllocation(*plan, 0), "cufftSetAutoAllocation");
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

__device__ __forceinline__ float padded_value(
    const float* input, int64_t index, int64_t z_size, int64_t plane_size, int64_t padding) {
  int64_t z = index / plane_size - padding;
  if (z < 0 || z >= z_size) {
    // Edge-inclusive reflection, or a zero halo when padding is at least Z.
    if (padding >= z_size) return 0.0f;
    z = z < 0 ? -z - 1 : 2 * z_size - z - 1;
  }
  return input[z * plane_size + index % plane_size];
}

__global__ void partial_mean_kernel(
    const float* input, double* partials, int64_t count, int64_t z_size,
    int64_t plane_size, int64_t padding) {
  double sum = 0.0;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    sum += static_cast<double>(padded_value(input, index, z_size, plane_size, padding));
  }
  __shared__ double sums[kThreads];
  sums[threadIdx.x] = sum;
  __syncthreads();
  for (int offset = kThreads / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) sums[threadIdx.x] += sums[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0) partials[blockIdx.x] = sums[0];
}

__global__ void finish_mean_kernel(const double* partials, float* mean, int partial_count, int64_t count) {
  double sum = 0.0;
  for (int i = threadIdx.x; i < partial_count; i += blockDim.x) sum += partials[i];
  __shared__ double sums[kThreads];
  sums[threadIdx.x] = sum;
  __syncthreads();
  for (int offset = kThreads / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) sums[threadIdx.x] += sums[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float value = static_cast<float>(sums[0] / static_cast<double>(count));
    mean[0] = value < 1e-12f ? 1e-12f : value;
  }
}

__global__ void pack_kernel(
    const float* input, const float* mean, float* spatial, int64_t count,
    int64_t z_size, int64_t plane_size, int64_t padding, int64_t x_size, int64_t physical_x) {
  const float padded_mean = mean[0];
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float value = padded_value(input, index, z_size, plane_size, padding);
    spatial[(index / x_size) * physical_x + index % x_size] = value / padded_mean - 1.0f;
  }
}

__global__ void filter_kernel(cufftComplex* spectrum, const cufftComplex* inverse, int64_t count) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const cufftComplex value = spectrum[index];
    const cufftComplex filter = inverse[index];
    spectrum[index] = {value.x * filter.x - value.y * filter.y,
                       value.x * filter.y + value.y * filter.x};
  }
}

__global__ void output_kernel(
    const float* spatial, float* output, int64_t count, int64_t unpad_offset,
    int64_t x_size, int64_t physical_x, float inverse_size) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t source = index + unpad_offset;
    output[index] = __fmul_rn(spatial[(source / x_size) * physical_x + source % x_size], inverse_size);
  }
}

cufftComplex* complex_data(const at::Tensor& tensor) {
  static_assert(sizeof(c10::complex<float>) == sizeof(cufftComplex));
  return reinterpret_cast<cufftComplex*>(tensor.data_ptr<c10::complex<float>>());
}

}  // namespace

struct State {
  int device;
  int64_t padded_z, z_size, y_size, x_size, padding, plane_size;
  int64_t padded_count, output_count, unpad_offset, half_x, physical_x, half_count;
  int partial_count, output_blocks;
  float inverse_size;
  std::size_t fft_workspace_bytes = 0;
  std::size_t persistent_tensor_bytes = 0;
  at::Tensor spectrum, inverse, partials, mean, fft_workspace;
  Resources resources;

  State(const at::Tensor& compact_inverse, int64_t logical_x, int64_t z_padding)
      : device(compact_inverse.defined() ? compact_inverse.get_device() : -1), resources(device) {
    check_tensor(compact_inverse, at::kComplexFloat, "inverse");
    padded_z = compact_inverse.size(0);
    y_size = compact_inverse.size(1);
    x_size = logical_x;
    padding = z_padding;
    TORCH_CHECK(padded_z > 0 && y_size > 0 && x_size > 0, "logical dimensions must be positive");
    TORCH_CHECK(padding >= 0 && padding <= (padded_z - 1) / 2,
                "z_padding must leave a positive unpadded Z dimension");
    TORCH_CHECK(compact_inverse.size(2) == x_size / 2 + 1,
                "inverse half-spectrum size must match explicit logical_x");
    z_size = padded_z - 2 * padding;
    plane_size = checked_product(y_size, x_size);
    padded_count = checked_product(padded_z, plane_size);
    output_count = checked_product(z_size, plane_size);
    TORCH_CHECK(padded_count <= std::numeric_limits<int64_t>::max() / 8 &&
                    output_count <= std::numeric_limits<int64_t>::max() / 4,
                "Volume byte sizes overflow int64");
    half_x = x_size / 2 + 1;
    physical_x = 2 * half_x;
    half_count = checked_product(checked_product(padded_z, y_size), half_x);
    TORCH_CHECK(half_count <= std::numeric_limits<int64_t>::max() / 8, "R2C storage overflows int64");
    unpad_offset = padding * plane_size;
    partial_count = blocks_for(padded_count);
    output_blocks = blocks_for(output_count);
    inverse_size = static_cast<float>(1.0 / static_cast<double>(padded_count));
    const c10::cuda::CUDAGuard guard(compact_inverse.device());
    const auto current = c10::cuda::getCurrentCUDAStream(device);
    const cudaStream_t stream = current.stream();
    resources.initialize();
    RecordCompletion completion{resources.finished, stream};
    const auto options = at::TensorOptions().device(compact_inverse.device()).dtype(at::kFloat).requires_grad(false);
    spectrum = at::empty({padded_z, y_size, half_x}, options.dtype(at::kComplexFloat));
    inverse = compact_inverse;
    partials = at::empty({partial_count}, options.dtype(at::kDouble));
    mean = at::empty({1}, options);
    long long dimensions[3] = {padded_z, y_size, x_size};
    long long real_embed[3] = {padded_z, y_size, physical_x};
    long long complex_embed[3] = {padded_z, y_size, half_x};
    std::size_t forward_bytes = 0, backward_bytes = 0;
    check_fft(cufftMakePlanMany64(resources.forward, 3, dimensions, real_embed, 1,
                                2 * half_count, complex_embed, 1, half_count,
                                CUFFT_R2C, 1, &forward_bytes), "R2C plan");
    check_fft(cufftMakePlanMany64(resources.backward, 3, dimensions, complex_embed, 1,
                                half_count, real_embed, 1, 2 * half_count,
                                CUFFT_C2R, 1, &backward_bytes), "C2R plan");
    fft_workspace_bytes = std::max(forward_bytes, backward_bytes);
    TORCH_CHECK(fft_workspace_bytes <= static_cast<std::size_t>(std::numeric_limits<int64_t>::max()),
                "cuFFT workspace exceeds int64 tensor size");
    fft_workspace = at::empty({static_cast<int64_t>(fft_workspace_bytes)}, options.dtype(at::kByte));
    for (auto plan : {resources.forward, resources.backward}) {
      check_fft(cufftSetWorkArea(plan, fft_workspace.data_ptr()), "cufftSetWorkArea");
      check_fft(cufftSetStream(plan, stream), "cufftSetStream");
    }
    persistent_tensor_bytes = checked_bytes_sum(spectrum.nbytes(), inverse.nbytes());
    persistent_tensor_bytes = checked_bytes_sum(persistent_tensor_bytes, partials.nbytes());
    persistent_tensor_bytes = checked_bytes_sum(persistent_tensor_bytes, mean.nbytes());
    persistent_tensor_bytes = checked_bytes_sum(persistent_tensor_bytes, fft_workspace.nbytes());
    completion.record();
    // G's producer must be ordered on this stream before construction.
    C10_CUDA_CHECK(cudaEventSynchronize(resources.finished));
  }

  at::Tensor run(const at::Tensor& input, const at::Tensor& output) {
    check_tensor(input, at::kFloat, "input");
    check_tensor(output, at::kFloat, "output");
    for (const auto& tensor : {input, output}) {
      TORCH_CHECK(tensor.get_device() == device && tensor.size(0) == z_size &&
                      tensor.size(1) == y_size && tensor.size(2) == x_size,
                  "input and output must match the Plan device and shape");
    }
    TORCH_CHECK(!overlaps(input, output), "output must not overlap input");
    TORCH_CHECK(!overlaps(output, inverse) && !overlaps(output, spectrum),
                "output must not overlap private Plan storage");
    const c10::cuda::CUDAGuard guard(input.device());
    const auto current = c10::cuda::getCurrentCUDAStream(device);
    const cudaStream_t stream = current.stream();
    C10_CUDA_CHECK(cudaStreamWaitEvent(stream, resources.finished, 0));
    RecordCompletion completion{resources.finished, stream};
    c10::cuda::CUDACachingAllocator::recordStream(input.storage().data_ptr(), current);
    c10::cuda::CUDACachingAllocator::recordStream(output.storage().data_ptr(), current);
    check_fft(cufftSetStream(resources.forward, stream), "cufftSetStream forward");
    check_fft(cufftSetStream(resources.backward, stream), "cufftSetStream backward");
    auto* buffer = complex_data(spectrum);
    partial_mean_kernel<<<partial_count, kThreads, 0, stream>>>(
        input.data_ptr<float>(), partials.data_ptr<double>(), padded_count, z_size, plane_size, padding);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    finish_mean_kernel<<<1, kThreads, 0, stream>>>(partials.data_ptr<double>(), mean.data_ptr<float>(), partial_count, padded_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    pack_kernel<<<partial_count, kThreads, 0, stream>>>(
        input.data_ptr<float>(), mean.data_ptr<float>(), reinterpret_cast<float*>(buffer),
        padded_count, z_size, plane_size, padding, x_size, physical_x);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    check_fft(cufftExecR2C(resources.forward, reinterpret_cast<float*>(buffer), buffer), "cufftExecR2C");
    filter_kernel<<<partial_count, kThreads, 0, stream>>>(buffer, complex_data(inverse), half_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    check_fft(cufftExecC2R(resources.backward, buffer, reinterpret_cast<float*>(buffer)), "cufftExecC2R");
    output_kernel<<<output_blocks, kThreads, 0, stream>>>(
        reinterpret_cast<float*>(buffer), output.data_ptr<float>(), output_count, unpad_offset,
        x_size, physical_x, inverse_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    completion.record();
    return output;
  }
};

State* create_plan(const at::Tensor& inverse, int64_t logical_x, int64_t z_padding) {
  return new State(inverse, logical_x, z_padding);
}
void destroy_plan(State* state) noexcept { delete state; }
at::Tensor run_plan(State* state, const at::Tensor& input, const at::Tensor& output) {
  return state->run(input, output);
}
std::size_t workspace_bytes(const State* state) { return state->fft_workspace_bytes; }
std::size_t persistent_bytes(const State* state) { return state->persistent_tensor_bytes; }

}  // namespace waveorder_reconstruction
