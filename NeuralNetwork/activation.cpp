#include "activation.h"

namespace nn {

ActivationFunction ActivationFunction::ReLU() {
    auto sigma = [](const Tensor2D& m) -> Tensor2D {
        return m.unaryExpr([](double x) -> double { return x > 0 ? x : 0; });
    };
    auto dsigma = [](const Tensor1D& m) -> Tensor2D {
        return m.unaryExpr([](double x) -> double { return x > 0 ? 1 : 0; }).asDiagonal();
    };

    return ActivationFunction(sigma, dsigma);
}

ActivationFunction ActivationFunction::Sigmoid() {
    auto sigma = [](const Tensor2D& m) -> Tensor2D {
        return m.unaryExpr([](double x) -> double { return 1 / (1 + std::exp(-x)); });
    };
    auto dsigma = [](const Tensor1D& m) -> Tensor2D {
        return m
            .unaryExpr([](double x) -> double {
                return (1 / (1 + std::exp(-x))) * (1 - 1 / (1 + std::exp(-x)));
            })
            .asDiagonal();
    };

    return ActivationFunction(sigma, dsigma);
}

// ActivationFunction ActivationFunction::Tanh() {
//         return ActivationFunction();
// }

NeuralDefines::Tensor2D ActivationFunction::operator()(const NeuralDefines::Tensor2D& x) {
    input_ = x;
    return sigma_(x);
}

NeuralDefines::Tensor2D ActivationFunction::Update(const NeuralDefines::Tensor2D& u) {
    Tensor2D ret(u.cols(), u.rows());
    for (Index i = 0; i < u.rows(); ++i) {
        ret.col(i) = dsigma_(input_.row(i)) * (u.row(i).transpose());
    }
    return ret;
}

ActivationFunction::ActivationFunction(std::function<Tensor2D(const Tensor2D&)> sigma,
                                       std::function<Tensor2D(const Tensor1D&)> dsigma)
    : sigma_(sigma), dsigma_(dsigma) {
}

}  // namespace nn
