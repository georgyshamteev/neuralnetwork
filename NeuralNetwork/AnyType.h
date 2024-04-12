#pragma once

#include <Eigen/Dense>
#include <memory>
#include "neuraldefines.h"

class AnyObject {
public:
    using Tensor2D = nn::Tensor2D;

    class InnerBase;
    AnyObject() = default;

    template <typename T>
    AnyObject(T &&object)
        : inner_(std::make_unique<Inner<std::remove_reference_t<T>>>(std::forward<T>(object))) {
    }

    AnyObject(AnyObject &&) noexcept = default;
    AnyObject &operator=(AnyObject &&) noexcept = default;

    const InnerBase *operator->() const {
        return inner_.get();
    }
    InnerBase *operator->() {
        return inner_.get();
    }

    Tensor2D operator()(const Tensor2D& x) {
        return inner_->operator()(x);
    }

    class InnerBase {
    public:
        using Tensor2D = nn::Tensor2D;
        friend class AnyObject;

        virtual ~InnerBase() = default;
        virtual Tensor2D operator()(const Tensor2D &) = 0;
        virtual Tensor2D Update(Tensor2D &u) = 0;

    };

private:
    template <typename T>
    class Inner : public InnerBase {
    public:
        Inner(T &&value) : value_(std::move(value)) {
        }

        virtual Tensor2D operator()(const Tensor2D &x) override {
            return value_(x);
        }

        Tensor2D Update(Tensor2D &u) override {
            return value_.Update(u);
        }

    private:
        T value_;
    };

    std::unique_ptr<InnerBase> inner_;
};
