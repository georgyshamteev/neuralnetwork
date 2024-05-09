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

HyperbolicOptimizer::HyperbolicOptimizer(std::vector<ParameterPack>&& parameters_pack, double lr,
                                         double decay)
    : params_(std::move(parameters_pack)), lr_(lr), decay_(decay) {
}

void HyperbolicOptimizer::Step(void) {
    for (auto& param_pack : params_) {
        param_pack.grad *= lr_;
        param_pack.w -= param_pack.grad;
    }
    double coef = decay_ * !(step_ % 100) + !!(step_ % 100);
    lr_ = coef * lr_;
    ++step_;
}

void HyperbolicOptimizer::ZeroGrad(void) {
    for (auto& param_pack : params_) {
        param_pack.grad.setZero();
    }
}

////

//// SGD with momentum

SGDOptimizer::SGDOptimizer(std::vector<ParameterPack>&& parameters_pack, double lr, double momentum,
                           double dampening, double weight_decay, bool nesterov)
    : params_(std::move(parameters_pack)),
      lr_(lr),
      momentum_(momentum),
      dampening_(dampening),
      weight_decay_(weight_decay),
      nesterov_(nesterov) {

    b_curr_.resize(params_.size());
    b_prev_.resize(params_.size());
}

void SGDOptimizer::Step(void) {
    size_t idx = 0;
    for (auto& param_pack : params_) {
        if (weight_decay_ != 0) {
            param_pack.grad += (weight_decay_ * param_pack.w).eval();
        }

        if (momentum_ != 0) {
            if (time_ > 1) {
                b_curr_[idx] = (momentum_ * b_prev_[idx]).eval() + ((1 - dampening_) * param_pack.grad).eval();
            } else {
                b_curr_[idx] = param_pack.grad;
            }

            b_prev_[idx] = b_curr_[idx];

            if (nesterov_) {
                param_pack.grad += (momentum_ * b_curr_[idx]).eval();
            } else {
                param_pack.grad = b_curr_[idx].eval();
            }
        }

        param_pack.w -= lr_ * param_pack.grad;

        ++idx;
    }

    ++time_;
}

void SGDOptimizer::ZeroGrad(void) {
    for (auto& param_pack : params_) {
        param_pack.grad.setZero();
    }
}

}  // namespace nn
