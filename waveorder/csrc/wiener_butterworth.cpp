#include <torch/extension.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

namespace waveorder_wiener_butterworth {
struct State;
State* create_plan(const at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t, float);
void destroy_plan(State*) noexcept;
at::Tensor run_plan(State*, const at::Tensor&, const at::Tensor&);
std::size_t workspace_bytes(const State*);
std::size_t persistent_bytes(const State*);

class Plan {
 public:
  Plan(const at::Tensor& h, const at::Tensor& b, int64_t logical_x,
       int64_t padding, int64_t iterations, float background)
      : state_(create_plan(h, b, logical_x, padding, iterations, background), destroy_plan) {}
  at::Tensor run(const at::Tensor& input, const at::Tensor& output) {
    std::lock_guard<std::mutex> guard(mutex_);
    TORCH_CHECK(state_, "WienerButterworthRL Plan is closed");
    return run_plan(state_.get(), input, output);
  }
  void close() {
    std::lock_guard<std::mutex> guard(mutex_);
    state_.reset();
  }
  std::size_t workspace_bytes() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return state_ ? waveorder_wiener_butterworth::workspace_bytes(state_.get()) : 0;
  }
  std::size_t persistent_bytes() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return state_ ? waveorder_wiener_butterworth::persistent_bytes(state_.get()) : 0;
  }
 private:
  mutable std::mutex mutex_;
  std::unique_ptr<State, void (*)(State*)> state_;
};
}  // namespace waveorder_wiener_butterworth

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  pybind11::class_<waveorder_wiener_butterworth::Plan>(module, "Plan")
      .def(pybind11::init<const at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t, float>(),
           pybind11::arg("h"), pybind11::arg("b"), pybind11::arg("logical_x"),
           pybind11::arg("padding"), pybind11::arg("iterations"), pybind11::arg("background"))
      .def("run", &waveorder_wiener_butterworth::Plan::run)
      .def("close", &waveorder_wiener_butterworth::Plan::close)
      .def("workspace_bytes", &waveorder_wiener_butterworth::Plan::workspace_bytes)
      .def("persistent_bytes", &waveorder_wiener_butterworth::Plan::persistent_bytes);
}
