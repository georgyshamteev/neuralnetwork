#include "neuraldefines.h"

namespace nn {

class ActivationFunction : protected NeuralDefines {

public:
    static ActivationFunction ReLU();
    static ActivationFunction Sigmoid();
    static ActivationFunction Tanh();

    Tensor2D operator()(const Tensor2D& x) const;
    Tensor2D Update(const Tensor2D& u);

private:
    ActivationFunction(std::function<double(double)>, std::function<double(double)>);
    Tensor2D input_seq_;
    std::function<double(double)> sigma_;
    std::function<double(double)> dsigma_;
};

}  // namespace nn
