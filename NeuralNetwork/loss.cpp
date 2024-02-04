#include "loss.h"

namespace nn {

Loss::Loss(std::function<double(const Tensor2D&, const Tensor2D&)> loss,
           std::function<Tensor2D(const Tensor2D&, const Tensor2D&)> grad)
    : loss_(loss), grad_(grad) {
}

double Loss::operator()(const nn::Tensor2D& x, const nn::Tensor2D& y) {
    assert(x.rows() == y.rows());
    assert(x.cols() == y.cols());
    input_ = x;
    label_ = y;
    return 1 / y.rows() * loss_(x, y);
}

Tensor2D Loss::Gradient() const {
    return grad_(input_, label_);
}

Loss Loss::MSE() {
    auto mse = [](const Tensor1D& x, const Tensor1D& y) -> double {
        double result = 0;
        for (Index i = 0; i < x.cols(); ++i) {
            result += (x[i] - y[i]) * (x[i] - y[i]);
        }
        return result;
    };

    auto mse_grad = [](const Tensor1D& x, const Tensor1D& y) -> Tensor1D {
        Tensor1D ret(x.cols());
        for (Index i = 0; i < x.cols(); ++i) {
            ret[i] = (x[i] - y[i]) * 2;
        }
        return ret;
    };

    auto loss = [mse](const Tensor2D& x, const Tensor2D& y) -> double {
        double result = 0;
        for (Index i = 0; i < x.rows(); ++i) {
            result += mse(x.row(i), y.row(i));
        }
        return result;
    };

    auto grad = [mse_grad](const Tensor2D& x, const Tensor2D& y) -> Tensor2D {
        Tensor2D ret(x.rows(), x.cols());
        for (Index i = 0; i < x.rows(); ++i) {
            ret.row(i) = mse_grad(x.row(i), y.row(i));
        }
        return ret;
    };

    return Loss(loss, grad);
}

}  // namespace nn
