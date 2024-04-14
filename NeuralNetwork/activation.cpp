#include "activation.h"

#include <iostream>

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

ActivationFunction ActivationFunction::Tanh() {
    auto tanh = [](double x) -> double {
        return (std::exp(x) - std::exp(-x)) / (std::exp(x) + std::exp(-x));
    };

    auto sigma = [tanh](const Tensor2D& m) -> Tensor2D { return m.unaryExpr(tanh); };

    auto dsigma = [tanh](const Tensor1D& m) -> Tensor2D {
        return m
            .unaryExpr([tanh](double x) -> double {
                double t = tanh(x);
                return 1 - t * t;
            })
            .asDiagonal();
    };

    return ActivationFunction(sigma, dsigma);
}

ActivationFunction ActivationFunction::Softmax() {
    auto sigma = [](const Tensor2D& m) -> Tensor2D {
        Tensor2D ret = m;
        for (Index i = 0; i < m.rows(); ++i) {
            double c = m.row(i).maxCoeff();
            double y = 0;
            for (Index j = 0; j < m.cols(); ++j) {
                y += std::exp(m(i, j) - c);
            }
            y = c + std::log(y);
            for (Index j = 0; j < m.cols(); ++j) {
                ret(i, j) = std::exp(m(i, j) - y);
            }
        }
        return ret;
    };

    auto dsigma = [](const Tensor1D& m) -> Tensor2D {
        Tensor2D ret(m.size(), m.size());
        double c = m.maxCoeff();
        double y = 0;
        for (Index j = 0; j < m.cols(); ++j) {
            y += std::exp(m(0, j) - c);
        }
        y = c + std::log(y);

        for (Index i = 0; i < m.cols(); ++i) {
            for (Index j = 0; j < m.cols(); ++j) {
                double si = std::exp(m(0, i) - y);
                double sj = std::exp(m(0, j) - y);

                if (i == j) {
                    ret(i, j) = si * (1 - sj);
                } else {
                    ret(i, j) = -si * sj;
                }
            }
        }
        return ret;
    };

    return ActivationFunction(sigma, dsigma);
}

NeuralDefines::Tensor2D ActivationFunction::operator()(const NeuralDefines::Tensor2D& x) {
    input_ = x;
    return sigma_(x).eval();
}

NeuralDefines::Tensor2D ActivationFunction::Update(const NeuralDefines::Tensor2D& u) {
    Tensor2D ret(u.cols(), u.rows());
//    std::cout << input_ << std::endl;
//    std::cout << "---------------";
    for (Index i = 0; i < u.rows(); ++i) {
        ret.col(i) = dsigma_(input_.row(i)).eval() * (u.row(i).transpose()).eval();
    }
    return ret;
}

ActivationFunction::ActivationFunction(std::function<Tensor2D(const Tensor2D&)> sigma,
                                       std::function<Tensor2D(const Tensor1D&)> dsigma)
    : sigma_(sigma), dsigma_(dsigma) {
}

std::vector<ParameterPack> ActivationFunction::TrainingParams() {
    return std::vector<ParameterPack>();
}

std::fstream& operator<<(std::fstream& in, const ActivationFunction& layer) {
    return in;
}

std::fstream& operator>>(std::fstream& in, ActivationFunction& layer) {
    return in;
}

}  // namespace nn
