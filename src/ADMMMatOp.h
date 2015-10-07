#ifndef ADMMMATOP_H
#define ADMMMATOP_H

#include <Eigen/Core>

template <typename Scalar>
class MatOpSymLower
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Map<const Matrix> MapMat;
    typedef Eigen::Map< Eigen::Matrix<Scalar, Eigen::Dynamic, 1> > MapVec;
    typedef const Eigen::Ref<const Matrix> ConstGenericMatrix;

    const MapMat mat;
    const int n;

public:
    MatOpSymLower(ConstGenericMatrix &mat_) :
        mat(mat_.data(), mat_.rows(), mat_.cols()),
        n(mat.rows())
    {}

    int rows() { return n; }
    int cols() { return n; }

    // y_out = A * x_in
    void perform_op(Scalar *x_in, Scalar *y_out)
    {
        MapVec x(x_in, n);
        MapVec y(y_out, n);
        y.noalias() = mat.template selfadjointView<Eigen::Lower>() * x;
    }
};


template <typename Scalar>
class MatOpXX
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Map<const Matrix> MapMat;
    typedef Eigen::Map< Eigen::Matrix<Scalar, Eigen::Dynamic, 1> > MapVec;
    typedef const Eigen::Ref<const Matrix> ConstGenericMatrix;

    const MapMat mat;
    const bool is_wide;
    const int dim;

public:
    MatOpXX(ConstGenericMatrix &mat_) :
        mat(mat_.data(), mat_.rows(), mat_.cols()),
        is_wide(mat.cols() > mat.rows()),
        dim(std::min(mat.rows(), mat.cols()))
    {}

    int rows() { return dim; }
    int cols() { return dim; }

    // y_out = A * x_in
    void perform_op(Scalar *x_in, Scalar *y_out)
    {
        MapVec x(x_in, dim);
        MapVec y(y_out, dim);

        if(is_wide)
            y.noalias() = mat * (mat.transpose() * x);
        else
            y.noalias() = mat.transpose() * (mat * x);
    }
};


#endif // ADMMMATOP_H
