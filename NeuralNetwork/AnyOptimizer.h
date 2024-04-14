#pragma once

#include <Eigen/Dense>
#include <memory>

#include "neuraldefines.h"

class AnyOptimizer {
public:
    using Tensor2D = nn::Tensor2D;

    class InnerBase;
    AnyOptimizer() = default;

    template <typename T>
    AnyOptimizer(T &&object)
        : inner_(std::make_unique<Inner<std::remove_reference_t<T>>>(std::forward<T>(object))) {
    }

    AnyOptimizer(AnyOptimizer &&) noexcept = default;
    AnyOptimizer &operator=(AnyOptimizer &&) noexcept = default;

    const InnerBase *operator->() const {
        return inner_.get();
    }
    InnerBase *operator->() {
        return inner_.get();
    }

    class InnerBase {
    public:
        using Tensor2D = nn::Tensor2D;
        friend class AnyOptimizer;

        virtual ~InnerBase() = default;
        virtual void Step(void) = 0;
    };

private:
    template <typename T>
    class Inner : public InnerBase {
    public:
        Inner(T &&value) : value_(std::move(value)) {
        }

        void Step(void) override {
            value_.Step();
        }

    private:
        T value_;
    };

    std::unique_ptr<InnerBase> inner_;
};
