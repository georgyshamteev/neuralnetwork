#include <Eigen/Dense>
#include "EigenRand/EigenRand"

namespace nn {
class ActivationFunction {
protected:
    using Index = Eigen::Index;
    using Tensor1D = Eigen::Matrix<double, 1, Eigen::Dynamic>;
    using Tensor2D = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using Tensor3D = std::vector<Tensor2D>;

public:
    virtual Tensor2D operator()(const Tensor2D& x) const = 0;
    virtual void Update(Tensor2D u) = 0;
};

class ReLu : ActivationFunction {};

class Sigmoid : ActivationFunction {};

class Tanh : ActivationFunction {};
}  // namespace nn
