#include "AnyOptimizer.h"
#include "neuraldefines.h"

namespace nn {

class ConstantOptimizer {
public:
    ConstantOptimizer(std::vector<ParameterPack>&& parameters_pack, double lr);
    void Step(void);
    void ZeroGrad(void);

private:
    std::vector<ParameterPack> params_;
    double lr_;
};

class HyperbolicOptimizer {
public:
    HyperbolicOptimizer(std::vector<ParameterPack>&& parameters_pack, double lr, double decay);
    void Step(void);
    void ZeroGrad(void);

private:
    std::vector<ParameterPack> params_;
    double lr_;
    double decay_;
    size_t step_ = 1;
};

class SGDOptimizer {
public:
    SGDOptimizer(std::vector<ParameterPack>&& parameters_pack, double lr, double momentum = 0, double dampening = 0, double weight_decay = 0, bool nesterov = false);
    void Step(void);
    void ZeroGrad(void);

private:
    std::vector<ParameterPack> params_;
    double lr_;
    double momentum_;
    double dampening_;
    double weight_decay_;
    bool nesterov_;

    Tensor3D b_curr_;
    Tensor3D b_prev_;
    size_t time_ = 1;
};

class AdamW {
public:

private:

};

}  // namespace nn
