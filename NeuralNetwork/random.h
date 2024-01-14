#include <Eigen/Dense>
#include "EigenRand/EigenRand"

namespace nn_random {
class Random {
    using Index = Eigen::Index;
    using Tensor1D = Eigen::Matrix<double, 1, Eigen::Dynamic>;
    using Tensor2D = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using Tensor3D = std::vector<Tensor2D>;

public:
    Tensor1D GetUniformVector(double low, double high, Index cols) const;
    Tensor2D GetUniformMatrix(double low, double high, Index rows, Index cols) const;

private:
    inline static Eigen::Rand::Vmt19937_64 urng{42};
};
}  // namespace nn_random
