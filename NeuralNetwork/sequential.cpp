#include "sequential.h"

nn::Tensor2D nn::Sequential::operator()(const nn::Tensor2D& x) {
    Tensor2D data = x;
    for (size_t i = 0; i < layers_.size(); ++i) {
        data = (layers_[i])(data);
    }
    return data;
}

void nn::Sequential::Backward(const nn::Tensor2D& x) {
    Tensor2D grad = x;
    for (size_t i = layers_.size(); i != 0; --i) {
        grad = layers_[i - 1]->Update(grad);
    }
}

std::vector<nn::ParameterPack> nn::Sequential::TrainingParams() {
    std::vector<nn::ParameterPack> res;
    for (auto& l : layers_) {
        auto params = l->TrainingParams();
        if (!params.empty()) {
            for (auto parameter : params) {
                res.push_back(parameter);
            }
        }
    }
    return res;
}

//// TODO: if I make "nn::Sequential& layer" const then AnyLayer fails, why?

std::fstream& nn::operator<<(std::fstream& in, nn::Sequential& layer) {
    for (auto& l : layer.layers_) {
        l << in;
    }
    //// bad sintaxis for some reason, "l << in" does l.operator<<(in) { in << l }; Problem in
    ///AnyLayer?
    return in;
}

std::fstream& nn::operator>>(std::fstream& in, nn::Sequential& layer) {
    for (auto& l : layer.layers_) {
        l >> in;
    }
    //// same problem as for <<
    return in;
}
