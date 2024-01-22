#include <Eigen/Dense>
#include "EigenRand/EigenRand"

namespace nn_random {

class Random {
    using Index = Eigen::Index;
    using Tensor1D = Eigen::Matrix<double, 1, Eigen::Dynamic>;
    using Tensor2D = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using Tensor3D = std::vector<Tensor2D>;

public:
    static Tensor1D GetGaussVector(Index cols);
    static Tensor2D GetGaussMatrix(Index rows, Index cols);

private:
    using Generator64 = Eigen::Rand::Vmt19937_64;
    static inline Generator64& Gen() {
        static Generator64 rng = 42;
        return rng;
    }
};

}  // namespace nn_random
