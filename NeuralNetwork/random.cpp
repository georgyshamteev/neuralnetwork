#include "random.h"

namespace nn_random {

Random::Tensor1D Random::GetUniformVector(double low, double high, Index cols) const {
    // TODO: function needs refactoring, because can only generate in [-1; 1]
    low = -1;
    high = 1;
    Eigen::Rand::UniformRealGen gen(low, high);
    return gen.generate<Tensor1D>(1, cols, urng);
}

Random::Tensor2D Random::GetUniformMatrix(double low, double high, Index rows, Index cols) const {
    // TODO: function needs refactoring, because can only generate in [-1; 1]
    low = -1;
    high = 1;
    Eigen::Rand::UniformRealGen gen(low, high);
    return gen.generate<Tensor2D>(rows, cols, urng);
}

}  // namespace nn_random
