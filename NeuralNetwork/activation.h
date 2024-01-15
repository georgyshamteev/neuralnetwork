#include "neuralbase.h"

namespace nn {
class ActivationFunction : protected NeuralBase {

public:
    virtual Tensor2D operator()(const Tensor2D& x) const = 0;
    virtual Tensor2D Update(const Tensor2D& u) = 0;
};

class ReLu : ActivationFunction {
public:
    ReLu();
    Tensor2D operator()(const Tensor2D& x) const;
    Tensor2D Update(const Tensor2D& u);

private:
    std::function<double(double)> sigma_;
    std::function<double(double)> dsigma_;
};

class Sigmoid : ActivationFunction {};

class Tanh : ActivationFunction {};
}  // namespace nn
