#include "neuraldefines.h"

namespace nn {

class Loss : protected NeuralDefines {

public:
    Loss();
    double operator()(const Tensor2D& x, const Tensor2D& y);
    Tensor2D CalculateGradient();

private:
};

}  // namespace nn
