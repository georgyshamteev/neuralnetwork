#include "neuraldefines.h"

namespace nn {

class Loss : protected NeuralDefines {

public:
    double operator()(const Tensor2D& x, const Tensor2D& y) const;
    Tensor2D Gradient() const;

private:
};

}  // namespace nn
