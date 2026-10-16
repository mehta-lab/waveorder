#include <torch/extension.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

namespace waveorder_reconstruction {

struct State;
State* create_plan(const at::Tensor& inverse, int64_t logical_x, int64_t z_padding);
void destroy_plan(State* state) noexcept;
at::Tensor run_plan(State* state, const at::Tensor& input, const at::Tensor& output);
std::size_t workspace_bytes(const State* state);
std::size_t persistent_bytes(const State* state);

class Plan {
 public:
  Plan(const at::Tensor& inverse, int64_t logical_x, int64_t z_padding)
      : state_(create_plan(inverse, logical_x, z_padding), destroy_plan) {}

  at::Tensor run(const at::Tensor& input, const at::Tensor& output) {
    std::lock_guard<std::mutex> guard(mutex_);
    TORCH_CHECK(state_, "Reconstruction Plan is closed");
    return run_plan(state_.get(), input, output);
  }

  void close() {
    std::lock_guard<std::mutex> guard(mutex_);
    state_.reset();
  }

  std::size_t workspace_bytes() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return state_ ? waveorder_reconstruction::workspace_bytes(state_.get()) : 0;
  }

  std::size_t persistent_bytes() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return state_ ? waveorder_reconstruction::persistent_bytes(state_.get()) : 0;
  }

 private:
  mutable std::mutex mutex_;
  std::unique_ptr<State, void (*)(State*)> state_;
};

}  // namespace waveorder_reconstruction

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  pybind11::class_<waveorder_reconstruction::Plan>(module, "Plan")
      .def(pybind11::init<const at::Tensor&, int64_t, int64_t>(),
           pybind11::arg("inverse"), pybind11::arg("logical_x"), pybind11::arg("z_padding"),
           "Adopt immutable compact G produced on the constructor's current stream.")
      .def("run", &waveorder_reconstruction::Plan::run,
           pybind11::arg("input"), pybind11::arg("output"),
           "Enqueue on the current stream into caller-owned output. No output alias is retained.")
      .def("close", &waveorder_reconstruction::Plan::close)
      .def("workspace_bytes", &waveorder_reconstruction::Plan::workspace_bytes)
      .def("persistent_bytes", &waveorder_reconstruction::Plan::persistent_bytes);
}
