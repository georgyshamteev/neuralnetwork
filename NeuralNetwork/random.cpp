#include "random.h"

namespace nn_random {

Random::Tensor1D Random::GetGaussVector(Index cols) {
    Eigen::Rand::NormalGen<double> gen;
    return gen.generate<Tensor1D>(1, cols, Gen());
}

Random::Tensor2D Random::GetGaussMatrix(Index rows, Index cols) {
    Eigen::Rand::NormalGen<double> gen;
    return gen.generate<Tensor2D>(rows, cols, Gen());
}

}  // namespace nn_random
