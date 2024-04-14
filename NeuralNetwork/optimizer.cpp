#include "optimizer.h"

#include <iostream>

namespace nn {

//// Constant optimizer

ConstantOptimizer::ConstantOptimizer(std::vector<ParameterPack>&& parameters_pack, double lr)
    : params_(std::move(parameters_pack)), lr_(lr) {
}

void ConstantOptimizer::Step(void) {
    for (auto& param_pack : params_) {
        param_pack.grad *= lr_;
        param_pack.w -= param_pack.grad;
    }
}

void ConstantOptimizer::ZeroGrad(void) {
    for (auto& param_pack : params_) {
        param_pack.grad.setZero();
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

void HyperbolicOptimizer::ZeroGrad(void) {
    for (auto& param_pack : params_) {
        param_pack.grad.setZero();
    }
}

////

}  // namespace nn
