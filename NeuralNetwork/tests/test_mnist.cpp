#include "../activation.h"
#include "../dataloader.h"
#include "../linear.h"
#include "../loss.h"
#include "../metrics.h"
#include "../optimizer.h"
#include "../sequential.h"

int main() {
    using Tensor2D = nn::Tensor2D;

    Tensor2D train_data = utils::ReadCSV("/home/georgyshamteev/archive/mnist_train.csv");
    Tensor2D test_data = utils::ReadCSV("/home/georgyshamteev/archive/mnist_test.csv");

    auto train_x = train_data.middleCols(1, train_data.cols() - 1);
    auto train_y_labels = train_data.col(0);
    Tensor2D train_y(train_y_labels.rows(), 10);
    train_y.setZero();

    for (nn::Index i = 0; i < train_y_labels.rows(); ++i) {
        nn::Index j = std::round(train_y_labels(i, 0));
        train_y(i, j) = 1.0;
    }

    train_x = train_x.unaryExpr([](double x) -> double { return x / 256; });

    auto test_x = test_data.middleCols(1, train_data.cols() - 1);
    auto test_y_labels = test_data.col(0);

    Tensor2D test_y(test_y_labels.rows(), 10);
    test_y.setZero();

    for (nn::Index i = 0; i < test_y_labels.rows(); ++i) {
        nn::Index j = std::round(test_y_labels(i, 0));
        test_y(i, j) = 1.0;
    }

    test_x = test_x.unaryExpr([](double x) -> double { return x / 256; });

    auto train_loader = utils::DataLoader(train_x, train_y, 100, utils::DataLoader::shuffle::True);
    auto val_loader = utils::DataLoader(test_x, test_y, 100, utils::DataLoader::shuffle::False);

    constexpr size_t kEpoch = 100;

    nn::Sequential model(nn::Linear(784, 256, nn::Bias::enable), nn::ActivationFunction::ReLU(),
                         nn::Linear(256, 128, nn::Bias::enable), nn::ActivationFunction::ReLU(),
                         nn::Linear(128, 32, nn::Bias::enable), nn::ActivationFunction::ReLU(),
                         nn::Linear(32, 10, nn::Bias::enable), nn::ActivationFunction::Softmax());

    auto loss_fn = nn::Loss::MSE();

    auto optimizer = nn::optimizer::RMSProp(model.TrainingParams(), 0.0001, 0.99, 1e-08, 0.001, 0.9, true);

    for (size_t epoch = 1; epoch < kEpoch; ++epoch) {
        size_t batch_number = 1;
        for (const auto& batch : train_loader) {
            auto inputs = batch.data;
            auto labels = batch.labels;

            optimizer.ZeroGrad();

            auto output = model(inputs);

            auto loss = loss_fn(output, labels);

            model.Backward(loss_fn.Gradient());

            optimizer.Step();

            if (batch_number % 10 == 0) {
                std::cout << "epoch" << ": " << epoch << ", batch: " << batch_number
                          << ", loss: " << loss << std::endl;
            }
            ++batch_number;
        }

        nn::Tensor1D pred_stacked(val_loader.Size());
        nn::Index pred_cnt = 0;
        nn::Tensor1D label_stacked(val_loader.Size());
        nn::Index label_cnt = 0;

        for (const auto& batch : val_loader) {
            auto inputs = batch.data;
            auto labels = batch.labels;

            auto output = model(inputs);

            Tensor2D out = utils::Argmax(output, 1);
            for (size_t i = 0; i < out.rows(); ++i) {
                pred_stacked(0, pred_cnt) = out(i, 0);
                ++pred_cnt;
            }

            Tensor2D label = utils::Argmax(labels, 1);
            for (size_t i = 0; i < label.rows(); ++i) {
                label_stacked(0, label_cnt) = label(i, 0);
                ++label_cnt;
            }
        }
        std::cout << "Accuracy: "
                  << metrics::Accuracy(pred_stacked, label_stacked, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
                  << "     F1: "
                  << metrics::F1(pred_stacked, label_stacked, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
                  << std::endl;
    }
}