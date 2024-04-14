#include "../activation.h"
#include "../dataloader.h"
#include "../linear.h"
#include "../loss.h"
#include "../metrics.h"
#include "../optimizer.h"
#include "../sequential.h"


int main()  {
    using Tensor2D = nn::Tensor2D;

    Tensor2D train_data = utils::ReadCSV("/home/georgyshamteev/archive/mnist_train.csv");
    Tensor2D test_data = utils::ReadCSV("/home/georgyshamteev/archive/mnist_test.csv");

    auto train_x = train_data.middleCols(1, train_data.cols() - 1);
    auto train_y = train_data.col(0);
    train_x = train_x.unaryExpr([](double x) -> double { return x / 256; });

    auto test_x = test_data.middleCols(1, train_data.cols() - 1);
    auto test_y = test_data.col(0);
    test_x = test_x.unaryExpr([](double x) -> double { return x / 256; });

    auto train_loader = utils::DataLoader(train_x, train_y, 100, utils::DataLoader::shuffle::True);
    auto val_loader = utils::DataLoader(test_x, test_y, 100, utils::DataLoader::shuffle::False);

    size_t EPOCH = 100;

    nn::Sequential model(nn::Linear(784, 196, nn::Bias::enable), nn::ActivationFunction::ReLU(),
                         nn::Linear(196, 49, nn::Bias::enable), nn::ActivationFunction::ReLU(),
                         nn::Linear(49, 10, nn::Bias::enable), nn::ActivationFunction::Softmax());

    auto loss_fn = nn::Loss::MSE();

    auto optimizer = nn::ConstantOptimizer(model.TrainingParams(), 0.001);

    for (size_t epoch = 1; epoch < 2; ++epoch) {
        size_t batch_number = 1;
        for (const auto& batch : train_loader) {
            auto inputs = batch.data;
            auto labels = batch.labels;

            optimizer.ZeroGrad();

            auto output = model(inputs);

            Tensor2D label_probs(labels.rows(), 10);
            label_probs.setZero();

            for (nn::Index i = 0; i < labels.rows(); ++i) {
                nn::Index j = std::round(labels(i, 0));
                label_probs(i, j) = 1.0;
            }

            auto loss = loss_fn(output, label_probs);

            model.Backward(loss_fn.Gradient());

            optimizer.Step();

            if (batch_number % 10 == 0) {
                std::cout << "epoch" << ": " << epoch << ", batch: " << batch_number << ", loss: " << loss
                          << std::endl;
            }
            ++batch_number;
        }
    }
}
