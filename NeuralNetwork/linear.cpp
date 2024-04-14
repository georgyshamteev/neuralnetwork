#include "linear.h"

namespace nn {

Linear::Linear(Index in_features, Index out_features, Bias enable_bias)
    : model_state_(ModelState::train),
      bias_state_(enable_bias),
      weight_(InitializeWeights(in_features, out_features)),
      bias_(InitializeBias(enable_bias, in_features, out_features)),
      data_ptr_(std::make_unique<LinearData>()) {
    assert(data_ptr_ != nullptr);
    assert(weight_.rows() == in_features);
    assert(weight_.cols() == out_features);
    assert(bias_.rows() == 1);
    assert(bias_.cols() == out_features);
}

Linear::Tensor2D Linear::operator()(const Linear::Tensor2D& x) {
    if (model_state_ == ModelState::train) {
        data_ptr_->input_seq_ = x;
    }
    return x * weight_ + (Tensor1D(x.rows()).setOnes().transpose()) * bias_;
}

Linear::Tensor2D Linear::Update(Linear::Tensor2D& u) {
    /*
     * We expect that vector u is coming from non-linear layer.
     * Thus vector u coming for update is actually vector u` = d sigma * u,
     * where u is the gradient vector that was fed to non-linear layer.
     * The size of u` is expected to be (m, k), while weight size is (n, m).
     */
    assert(u.rows() == weight_.cols());

    Tensor2D rethrow_gradient_vector = (weight_ * u).transpose();
    data_ptr_->weight_grad_ = (u * data_ptr_->input_seq_).transpose();
    data_ptr_->bias_grad_ = u.rowwise().sum().transpose();

    return rethrow_gradient_vector;
}

Linear::Tensor2D Linear::InitializeWeights(int64_t in_features, int64_t out_features) {
    assert(in_features != 0);
    assert(out_features != 0);
    return nn_random::Random::GetGaussMatrix(in_features, out_features);
}

Linear::Tensor2D Linear::InitializeBias(Bias enable_bias, int64_t in_features,
                                        int64_t out_features) {
    assert(in_features != 0);
    assert(out_features != 0);
    if (enable_bias == Bias::enable) {
        return nn_random::Random::GetGaussVector(out_features);
    }
    return Tensor2D(1, out_features).setZero();
}

void Linear::Train() {
    model_state_ = ModelState::train;
    //    if (data_ptr_ == nullptr) {
    //        data_ptr_ = std::make_unique<LinearData>();
    //    }
}

void Linear::Eval() {
    model_state_ = ModelState::eval;
    //    if (data_ptr_ != nullptr) {
    //        data_ptr_.reset(nullptr);
    //    }
}

std::vector<ParameterPack> Linear::TrainingParams() {
    if (model_state_ == ModelState::eval) {
        throw std::runtime_error("Can not get training params in eval mode.");
    }
    if (bias_state_ == Bias::enable) {
        return std::vector<ParameterPack>{ParameterPack{weight_, data_ptr_->weight_grad_},
                                          {bias_, data_ptr_->bias_grad_}};
    }
    return std::vector<ParameterPack>{{weight_, data_ptr_->weight_grad_}};
}

std::fstream& operator<<(std::fstream& out, const Linear& layer) {
    if (layer.bias_state_ == Bias::enable) {
        out << 1 << '\n';
        out << layer.bias_.rows() << ' ' << layer.bias_.cols() << '\n';
        out << layer.bias_ << '\n';
    } else {
        out << 0 << '\n';
    }
    out << layer.weight_.rows() << ' ' << layer.weight_.cols() << '\n';
    out << layer.weight_ << '\n';
    return out;
}

std::fstream& operator>>(std::fstream& in, Linear& layer) {
    int bias;
    in >> bias;
    if (bias) {
        Index bias_rows;
        Index bias_cols;
        in >> bias_rows >> bias_cols;
        for (size_t i = 0; i < bias_rows; ++i) {
            for (size_t j = 0; j < bias_cols; ++j) {
                double w;
                in >> w;
                layer.bias_(i, j) = w;
            }
        }
    }
    Index rows;
    Index cols;
    in >> rows >> cols;

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            double w;
            in >> w;
            layer.weight_(i, j) = w;
        }
    }
    return in;
}

}  // namespace nn
