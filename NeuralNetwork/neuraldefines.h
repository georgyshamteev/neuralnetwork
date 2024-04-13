#pragma once

#include <Eigen/Dense>
#include "EigenRand/EigenRand"
#include <cmath>
#include "random.h"
#include <cassert>

namespace nn {

using Index = Eigen::Index;
using Tensor1D = Eigen::Matrix<double, 1, Eigen::Dynamic>;
using Tensor2D = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
using Tensor3D = std::vector<Tensor2D>;

struct ParameterPack {
    Tensor2D& w;
    Tensor2D& grad;
};

class NeuralDefines {
protected:
    using Index = Eigen::Index;
    using Tensor1D = Eigen::Matrix<double, 1, Eigen::Dynamic>;
    using Tensor2D = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using Tensor3D = std::vector<Tensor2D>;
};

}  // namespace nn
