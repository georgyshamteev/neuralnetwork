#include "metrics.h"

namespace metrics {

double Precision(const Tensor1D& pred, const Tensor1D& labels, std::vector<int> classes) {
    Tensor1D correct_predictions(classes.size());
    Tensor1D class_predictions(classes.size());
    correct_predictions.setZero();
    class_predictions.setZero();

    for (Index i = 0; i < pred.cols(); ++i) {
        if (pred[i] == labels[i]) {
            ++correct_predictions[pred[i]];
        }
        ++class_predictions[pred[i]];
    }

    double precision = 0;

    for (Index i = 0; i < correct_predictions.cols(); ++i) {
        precision += correct_predictions[i] / class_predictions[i];
    }
    precision /= correct_predictions.cols();
    return precision;
}

double Recall(const Tensor1D& pred, const Tensor1D& labels, std::vector<int> classes) {
    Tensor1D correct_predictions(classes.size());
    Tensor1D class_instances(classes.size());
    correct_predictions.setZero();
    class_instances.setZero();

    for (Index i = 0; i < pred.cols(); ++i) {
        if (pred[i] == labels[i]) {
            ++correct_predictions[pred[i]];
        }
        ++class_instances[labels[i]];
    }

    double recall = 0;

    for (Index i = 0; i < correct_predictions.cols(); ++i) {
        recall += correct_predictions[i] / class_instances[i];
    }
    recall /= correct_predictions.cols();
    return recall;
}

double Accuracy(const Tensor1D& pred, const Tensor1D& labels, std::vector<int> classes) {
    Tensor1D correct_predictions(classes.size());
    correct_predictions.setZero();

    double c = 0;

    for (Index i = 0; i < pred.cols(); ++i) {
        if (pred[i] == labels[i]) {
            ++correct_predictions[pred[i]];
            ++c;
        }
    }

//    return static_cast<double>(correct_predictions.sum()) / pred.cols();
    return c / pred.cols();
}

double F1(const Tensor1D& pred, const Tensor1D& labels, std::vector<int> classes) {
    double precision = Precision(pred, labels, classes);
    double recall = Recall(pred, labels, classes);
    return 2 * precision * recall / (precision + recall);
}

}  // namespace metrics
