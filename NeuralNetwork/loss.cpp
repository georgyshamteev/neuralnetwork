#include "loss.h"

namespace nn {

double Loss::operator()(const nn::Tensor2D& x, const nn::Tensor2D& y) const {
    return 0;
}

Tensor2D Loss::Gradient() const {
    return nn::NeuralDefines::Tensor2D();
}

}  // namespace nn
