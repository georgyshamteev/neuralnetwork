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
                b_curr_[idx] =
                    (momentum_ * b_prev_[idx]).eval() + ((1 - dampening_) * param_pack.grad).eval();
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

////

//// AdamW

AdamW::AdamW(std::vector<ParameterPack>&& parameters_pack, double lr, double beta1, double beta2,
             double eps, double weight_decay, bool amsgrad)
    : params_(std::move(parameters_pack)),
      lr_(lr),
      beta1_(beta1),
      beta2_(beta2),
      eps_(eps),
      weight_decay_(weight_decay),
      amsgrad_(amsgrad) {
    m_.resize(params_.size());
    m_prev_.resize(params_.size());
    v_.resize(params_.size());
    v_prev_.resize(params_.size());
    m_hat_.resize(params_.size());
    v_hat_.resize(params_.size());
    v_hat_max_.resize(params_.size());

    for (size_t i = 0; i < params_.size(); ++i) {
        m_prev_[i].resize(params_[i].w.rows(), params_[i].w.cols());
        m_prev_[i].setZero();
    }
    for (size_t i = 0; i < params_.size(); ++i) {
        v_prev_[i].resize(params_[i].w.rows(), params_[i].w.cols());
        v_prev_[i].setZero();
    }
    for (size_t i = 0; i < params_.size(); ++i) {
        v_hat_max_[i].resize(params_[i].w.rows(), params_[i].w.cols());
        v_hat_max_[i].setZero();
    }
}

void AdamW::Step(void) {
    static auto square = [](double x) -> double { return std::pow(x, 2); };
    static auto sqrt_eps = [=](double x) -> double { return std::sqrt(x) + eps_; };

    size_t idx = 0;
    for (auto& param_pack : params_) {
        param_pack.w -= (weight_decay_ * lr_ * param_pack.w).eval();
        m_[idx] = (beta1_ * m_prev_[idx]).eval() + ((1 - beta1_) * param_pack.grad).eval();
        v_[idx] = (beta2_ * v_prev_[idx]).eval() +
                  ((1 - beta2_) * param_pack.grad.unaryExpr(square)).eval();
        m_hat_[idx] = m_[idx] / (1 - std::pow(beta1_, time_));
        v_hat_[idx] = v_[idx] / (1 - std::pow(beta2_, time_));

        if (amsgrad_) {
            v_hat_max_[idx] = v_hat_max_[idx].cwiseMax(v_hat_[idx]).eval();
            param_pack.w -=
                lr_ * (m_hat_[idx].cwiseProduct(v_hat_max_[idx].unaryExpr(sqrt_eps).cwiseInverse()))
                          .eval();
        } else {
            param_pack.w -=
                lr_ *
                (m_hat_[idx].cwiseProduct(v_hat_[idx].unaryExpr(sqrt_eps).cwiseInverse())).eval();
        }
        m_prev_[idx] = m_[idx];
        v_prev_[idx] = v_[idx];
        ++idx;
    }
    ++time_;
}

void AdamW::ZeroGrad(void) {
    for (auto& param_pack : params_) {
        param_pack.grad.setZero();
    }
}

}  // namespace nn
