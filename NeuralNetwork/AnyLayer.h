#pragma once

#include <Eigen/Dense>
#include <memory>
#include "neuraldefines.h"
#include <fstream>

class AnyLayer {
public:
    using Tensor2D = nn::Tensor2D;

    class InnerBase;
    AnyLayer() = default;

    template <typename T>
    AnyLayer(T &&object)
        : inner_(std::make_unique<Inner<std::remove_reference_t<T>>>(std::forward<T>(object))) {
    }

    AnyLayer(AnyLayer &&) noexcept = default;
    AnyLayer &operator=(AnyLayer &&) noexcept = default;

    const InnerBase *operator->() const {
        return inner_.get();
    }

    InnerBase *operator->() {
        return inner_.get();
    }

    Tensor2D operator()(const Tensor2D &x) {
        return inner_->operator()(x);
    }

    std::fstream& operator<<(std::fstream& f) {
        return inner_->operator<<(f);
    }

    std::fstream& operator>>(std::fstream& f) {
        return inner_->operator>>(f);
    }

    class InnerBase {
    public:
        using Tensor2D = nn::Tensor2D;
        friend class AnyLayer;

        virtual ~InnerBase() = default;
        virtual Tensor2D operator()(const Tensor2D &) = 0;
        virtual Tensor2D Update(Tensor2D &u) = 0;
        virtual std::vector<nn::ParameterPack> TrainingParams() = 0;
        virtual std::fstream& operator<<(std::fstream&) = 0;
        virtual std::fstream& operator>>(std::fstream&) = 0;
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

        std::vector<nn::ParameterPack> TrainingParams() override {
            return value_.TrainingParams();
        }

        std::fstream& operator<<(std::fstream& f) override {
            return f << value_;
        }

        std::fstream& operator>>(std::fstream& f) override {
            return f >> value_;
        }

    private:
        T value_;
    };

    std::unique_ptr<InnerBase> inner_;
};
