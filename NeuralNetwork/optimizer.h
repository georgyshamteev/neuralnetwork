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
    SGDOptimizer(std::vector<ParameterPack>&& parameters_pack, double lr, double momentum = 0,
                 double dampening = 0, double weight_decay = 0, bool nesterov = false);
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
    AdamW(std::vector<ParameterPack>&& parameters_pack, double lr, double beta1 = 0.9,
          double beta2 = 0.999, double eps = 1e-08, double weight_decay = 0.01,
          bool amsgrad = false);
    void Step(void);
    void ZeroGrad(void);

private:
    std::vector<ParameterPack> params_;
    double lr_;
    double beta1_;
    double beta2_;
    double eps_;
    double weight_decay_;
    bool amsgrad_;

    Tensor3D m_;
    Tensor3D m_prev_;
    Tensor3D v_;
    Tensor3D v_prev_;

    Tensor3D m_hat_;
    Tensor3D v_hat_;
    Tensor3D v_hat_max_;

    size_t time_ = 1;
};

class RMSProp {
public:
    RMSProp(std::vector<ParameterPack>&& parameters_pack, double lr, double alpha = 0.99,
            double eps = 1e-08, double weight_decay = 0, double momentum = 0,
            bool centered = false);
    void Step(void);
    void ZeroGrad(void);

private:
    std::vector<ParameterPack> params_;
    double lr_;
    double alpha_;
    double eps_;
    double weight_decay_;
    double momentum_;
    bool centered_;

    Tensor3D v_;
    Tensor3D v_prev_;
    Tensor3D b_;
    Tensor3D b_prev_;
    Tensor3D g_ave_;
    Tensor3D g_ave_prev_;
    Tensor3D v_hat_;
};

}  // namespace nn
