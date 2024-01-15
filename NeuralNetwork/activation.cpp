#include "activation.h"

namespace nn {
ReLu::ReLu()
    : sigma_([](double x) { return x > 0 ? x : 0; }),
      dsigma_([](double x) { return x > 0 ? 1 : 0; }) {
}

ActivationFunction::Tensor2D ReLu::operator()(const ActivationFunction::Tensor2D& x) const {
    x.array().unaryExpr(sigma_);
}

ActivationFunction::Tensor2D ReLu::Update(const ActivationFunction::Tensor2D& u) {
    //// Тут какой-то пиздец с умножением - нужно проверять что работает
    Tensor2D ret(u.cols(), u.rows());
    for (Index i = 0; i < u.rows(); ++i) {
        ret.col(i) = Eigen::Diagonal(u.row(i)).unaryExpr(dsigma_) * u.row(i).transpose();
    }
    return ret;
}

}  // namespace nn
