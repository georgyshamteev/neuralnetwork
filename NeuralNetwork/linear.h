#include <iostream>
#include "Eigen/Dense"
#include "EigenRand/EigenRand"


class Linear {
    ////
    using Tensor1D = Eigen::Matrix<double, 1, Eigen::Dynamic>;
    using Tensor2D = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    ////
public:
    Linear(size_t in_features, size_t out_features, bool bias = true);

    Tensor2D operator()(Tensor2D x);

private:
    Tensor2D grad_;
    Tensor2D weight_;
    Tensor1D bias_;
};