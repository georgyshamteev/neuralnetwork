#include <iostream>
#include <Eigen/Dense>
#include "../EigenRand/EigenRand/EigenRand"


class Linear {
    ////
    using Tensor1D = Eigen::Matrix<double, 1, Eigen::Dynamic>;
    using Tensor2D = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using Tensor3D = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic>;

    ////
public:
    Linear() = delete;
    Linear(size_t in_features, size_t out_features, bool bias = true);
    Tensor2D operator()(Tensor2D x);
    void Update(Tensor2D u);
    Tensor2D RethrowGradient(Tensor2D u);

private:
    Tensor2D input_seq_;
    Tensor2D output_seq_;
    Tensor2D weight_grad_;
    Tensor1D bias_grad_;
    Tensor2D weight_;
    Tensor1D bias_;
    inline static Eigen::Rand::Vmt19937_64 urng{42};
};

