#include "neuraldefines.h"
#include "AnyOptimizer.h"

namespace nn {

//// Хочу спрятать класс от пользователя, чтобы его мог звать только класс Optimizer, как лучше
///сделать?

class ConstantOptimizer {
public:
    ConstantOptimizer(std::vector<ParameterPack>&& parameters_pack, double lr);
    void Step(void);

private:
    std::vector<ParameterPack> params_;
    double lr_;
};

class HyperbolicOptimizer {
public:
    HyperbolicOptimizer(std::vector<ParameterPack>&& parameters_pack, double lr);
    void Step(void);

private:
    std::vector<ParameterPack> params_;
    double initial_lr_;
    double lr_;
    size_t epoch_ = 1;
};

class Optimizer {
    friend class ConstantOptimizer;
    friend class HyperbolicOptimizer;

    static Optimizer ConstantOptimizer(std::vector<ParameterPack>&& training_params,
                                       double learning_rate = 0.01);
    static Optimizer HyperbolicOptimizer(std::vector<ParameterPack>&& training_params,
                                         double learning_rate = 0.01);

private:
    template <class Optim>
    Optimizer(Optim&& optim) : optim_(std::forward<Optim>(optim)) {
    }

    AnyOptimizer optim_;
};

}  // namespace nn
