#include "activation.h"

namespace nn {

ActivationFunction ActivationFunction::ReLU() {
    return ActivationFunction(([](double x) { return x > 0 ? x : 0; }),
                              ([](double x) { return x > 0 ? 1 : 0; }));
}

ActivationFunction ActivationFunction::Sigmoid() {
    return ActivationFunction(
        [](double x) { return 1 / (1 + std::exp(-x)); },
        [](double x) { return (1 / (1 + std::exp(-x))) * (1 - 1 / (1 + std::exp(-x))); });
}

ActivationFunction ActivationFunction::Tanh() {
    //    return ActivationFunction();
}

NeuralDefines::Tensor2D ActivationFunction::operator()(const NeuralDefines::Tensor2D& x) const {
    return x.unaryExpr(sigma_);
}

NeuralDefines::Tensor2D ActivationFunction::Update(const NeuralDefines::Tensor2D& u) {
    Tensor2D ret(u.cols(), u.rows());
    for (Index i = 0; i < u.rows(); ++i) {
        ret.col(i) = Eigen::Diagonal(u.row(i)).unaryExpr(dsigma_) * u.row(i).transpose();
    }
    return ret;
}

ActivationFunction::ActivationFunction(std::function<double(double)> sigma,
                                       std::function<double(double)> dsigma)
    : sigma_(sigma), dsigma_(dsigma) {
}

}  // namespace nn
