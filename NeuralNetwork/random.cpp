#include "random.h"

namespace nn_random {
Random::Tensor1D Random::GetUniformVector(double low, double high, Index cols) const {
    Eigen::Rand::UniformRealGen gen(low, high);
    return gen.generate<Tensor1D>(1, cols, urng);
}

Random::Tensor2D Random::GetUniformMatrix(double low, double high, Index rows, Index cols) const {
    Eigen::Rand::UniformRealGen gen(low, high);
    return gen.generate<Tensor2D>(rows, cols, urng);
}
}  // namespace nn_random
