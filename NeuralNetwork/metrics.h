#include <Eigen/Dense>
#include <vector>

namespace metrics {

using Index = Eigen::Index;
using Tensor1D = Eigen::Matrix<double, 1, Eigen::Dynamic>;
using Tensor2D = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
using Tensor3D = std::vector<Tensor2D>;

double Precision(const Tensor1D& pred, const Tensor1D& labels, std::vector<int> classes);

double Recall(const Tensor1D& pred, const Tensor1D& labels, std::vector<int> classes);

double Accuracy(const Tensor1D& pred, const Tensor1D& labels, std::vector<int> classes);

double F1(const Tensor1D& pred, const Tensor1D& labels, std::vector<int> classes);

}  // namespace metrics
