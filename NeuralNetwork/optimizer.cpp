#include "optimizer.h"

namespace nn {

//// Constant optimizer

ConstantOptimizer::ConstantOptimizer(std::vector<ParameterPack>&& parameters_pack, double lr)
    : params_(std::move(parameters_pack)), lr_(lr) {
}

void ConstantOptimizer::Step(void) {
    for (auto& param_pack : params_) {
        param_pack.w -= lr_ * param_pack.grad;
    }
}

////

//// Hyperbolic optimizer

HyperbolicOptimizer::HyperbolicOptimizer(std::vector<ParameterPack>&& parameters_pack, double lr)
    : params_(std::move(parameters_pack)), lr_(lr), initial_lr_(lr) {
}

void HyperbolicOptimizer::Step(void) {
    for (auto& param_pack : params_) {
        param_pack.w -= lr_ * param_pack.grad;
    }
    ++epoch_;
    lr_ = initial_lr_ / epoch_;
}

////

Optimizer Optimizer::ConstantOptimizer(std::vector<ParameterPack>&& training_params,
                                       double learning_rate) {
    class ConstantOptimizer optimizer(std::move(training_params), learning_rate);
    return Optimizer(std::move(optimizer));
}

Optimizer Optimizer::HyperbolicOptimizer(std::vector<ParameterPack>&& training_params,
                                         double learning_rate) {
    class ConstantOptimizer optimizer(std::move(training_params), learning_rate);
    return Optimizer(std::move(optimizer));
}

}  // namespace nn
