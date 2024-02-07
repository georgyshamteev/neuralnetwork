#include "sequential.h"

namespace nn {

template <typename... Args>
Sequential<Args...>::Sequential(Args &&...args) : layers_(std::move(args...)) {
}

}  // namespace nn
