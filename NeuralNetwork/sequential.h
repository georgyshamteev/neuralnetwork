#include "neuraldefines.h"

namespace nn {

template <typename... Args>
class Sequential {

public:
    Sequential(Args&&... args);

private:
    std::tuple<Args...> layers_;
};

}  // namespace nn
