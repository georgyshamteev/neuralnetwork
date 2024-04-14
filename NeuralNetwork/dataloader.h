#include <Eigen/Dense>
#include <fstream>
#include <string>

namespace utils {

using Index = Eigen::Index;
using Tensor1D = Eigen::Matrix<double, 1, Eigen::Dynamic>;
using Tensor2D = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
using Tensor3D = std::vector<Tensor2D>;

class DataLoader {
    using Permutations = Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>;

public:
    enum class shuffle {
        True,
        False,
    };
    DataLoader(Tensor2D data, Tensor2D labels, size_t batch_size, shuffle enable_shuffle);

    void Shuffle();
    void Restore();

    class DataLoaderIterator {
    public:
        struct Batch {
            Tensor2D data;
            Tensor2D labels;
        };
        using IteratorCategory = std::forward_iterator_tag;
        using DifferenceType = std::ptrdiff_t;
        using Reference = Batch;

    public:
        DataLoaderIterator(const Tensor2D* data_ptr, const Tensor2D* labels_ptr, size_t idx,
                           size_t batch);

        Reference operator*() const;

        DataLoaderIterator& operator++();
        DataLoaderIterator operator++(int);
        friend bool operator!=(const DataLoaderIterator& lhs, const DataLoaderIterator& rhs);

    private:
        const Tensor2D* data_ptr_;
        const Tensor2D* labels_ptr_;
        size_t index_ = 0;
        size_t batch_size_ = 1;
    };

    DataLoaderIterator begin();
    DataLoaderIterator end();

private:
    Tensor2D data_;
    Tensor2D labels_;
    size_t batch_size_ = 1;
    shuffle enable_shuffle_ = shuffle::True;
    Permutations permutations_matrix_;
};

Tensor2D ReadCSV(std::string);

Tensor2D Argmax(const Tensor2D& v, size_t axis = 1);

}  // namespace utils
