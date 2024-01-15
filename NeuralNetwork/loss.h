#include "neuralbase.h"

namespace nn {
class Loss : protected NeuralBase {

public:
    virtual double operator()(const Tensor2D& x, const Tensor2D& y) = 0;
    virtual Tensor2D CalculateGradient() = 0;
};

class MSELoss : Loss {};

class BCELoss : Loss {};
}  // namespace nn
