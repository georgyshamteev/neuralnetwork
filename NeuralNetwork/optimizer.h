#include "AnyOptimizer.h"
#include "neuraldefines.h"

namespace nn {

//// Хочу спрятать класс от пользователя, чтобы его мог звать только класс Optimizer, как лучше
/// сделать?

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
    HyperbolicOptimizer(std::vector<ParameterPack>&& parameters_pack, double lr);
    void Step(void);
    void ZeroGrad(void);

private:
    std::vector<ParameterPack> params_;
    double initial_lr_;
    double lr_;
    size_t epoch_ = 1;
};

}  // namespace nn
