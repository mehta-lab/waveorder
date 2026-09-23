#include <torch/extension.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <tuple>

namespace waveorder_gradient_consensus {

struct State;
State* create_plan(const at::Tensor& transfer, int64_t logical_x);
void destroy_plan(State* state) noexcept;
void reset_plan(State* state);
std::tuple<bool, double> step_plan(State* state, const at::Tensor& measured,
                                   const at::Tensor& heads, double background);
void copy_output(State* state, const at::Tensor& output, int64_t padding);

class Plan {
 public:
  Plan(const at::Tensor& transfer, int64_t logical_x)
      : state_(create_plan(transfer, logical_x), destroy_plan) {}

  void reset() {
    std::lock_guard<std::mutex> guard(mutex_);
    TORCH_CHECK(state_, "GradientConsensusRL Plan is closed");
    reset_plan(state_.get());
  }

  std::tuple<bool, double> step(const at::Tensor& measured, const at::Tensor& heads,
                                double background) {
    std::lock_guard<std::mutex> guard(mutex_);
    TORCH_CHECK(state_, "GradientConsensusRL Plan is closed");
    return step_plan(state_.get(), measured, heads, background);
  }

  void copy(const at::Tensor& output, int64_t padding) {
    std::lock_guard<std::mutex> guard(mutex_);
    TORCH_CHECK(state_, "GradientConsensusRL Plan is closed");
    copy_output(state_.get(), output, padding);
  }

  void close() {
    std::lock_guard<std::mutex> guard(mutex_);
    state_.reset();
  }

 private:
  std::mutex mutex_;
  std::unique_ptr<State, void (*)(State*)> state_;
};

}  // namespace waveorder_gradient_consensus

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  pybind11::class_<waveorder_gradient_consensus::Plan>(module, "Plan")
      .def(pybind11::init<const at::Tensor&, int64_t>(), pybind11::arg("transfer"), pybind11::arg("logical_x"))
      .def("reset", &waveorder_gradient_consensus::Plan::reset)
      .def("step", &waveorder_gradient_consensus::Plan::step,
           pybind11::arg("measured"), pybind11::arg("heads"), pybind11::arg("background"))
      .def("copy_output", &waveorder_gradient_consensus::Plan::copy,
           pybind11::arg("output"), pybind11::arg("padding"))
      .def("close", &waveorder_gradient_consensus::Plan::close);
}
