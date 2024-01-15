#include "activation.h"

namespace nn {
ReLu::ReLu()
    : sigma_([](double x) { return x > 0 ? x : 0; }),
      dsigma_([](double x) { return x > 0 ? 1 : 0; }) {
}

ActivationFunction::Tensor2D ReLu::operator()(const ActivationFunction::Tensor2D& x) const {
    //// TODO: do we have to do x.array() for vectorization and does it return Tensor2D or not
    return x.array().unaryExpr(sigma_);
}

ActivationFunction::Tensor2D ReLu::Update(const ActivationFunction::Tensor2D& u) {
    //// Тут какой-то пиздец с умножением - нужно проверять что работает
    Tensor2D ret(u.cols(), u.rows());
    for (Index i = 0; i < u.rows(); ++i) {
        ret.col(i) = Eigen::Diagonal(u.row(i)).unaryExpr(dsigma_) * u.row(i).transpose();
    }
    return ret;
}

Sigmoid::Sigmoid()
    : sigma_([](double x) { return 1 / (1 + std::exp(-x)); }),
      dsigma_([](double x) { return (1 / (1 + std::exp(-x))) * (1 - 1 / (1 + std::exp(-x))); }) {
}

NeuralBase::Tensor2D Sigmoid::operator()(const NeuralBase::Tensor2D& x) const {
    //// TODO: do we have to do x.array() for vectorization and does it return Tensor2D or not
    return x.array().unaryExpr(sigma_);
}

NeuralBase::Tensor2D Sigmoid::Update(const NeuralBase::Tensor2D& u) {
    //// Тут какой-то пиздец с умножением - нужно проверять что работает
    Tensor2D ret(u.cols(), u.rows());
    for (Index i = 0; i < u.rows(); ++i) {
        ret.col(i) = Eigen::Diagonal(u.row(i)).unaryExpr(dsigma_) * u.row(i).transpose();
    }
    return ret;
}

}  // namespace nn
