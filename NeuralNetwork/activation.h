#include <Eigen/Dense>
#include "../EigenRand/EigenRand/EigenRand"

namespace nn {
class ActivationFunction {
    ////
    using Tensor1D = Eigen::Matrix<double, 1, Eigen::Dynamic>;
    using Tensor2D = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using Tensor3D = std::vector<Tensor2D>;
    ////
public:
    Tensor2D operator()(Tensor2D x);
    void Update(Tensor2D u);
    Tensor2D RethrowGradient(Tensor2D u);

private:
    Tensor3D d_sigma_;
};
}  // namespace nn
