#ifndef MATOPDENSE_H
#define MATOPDENSE_H

#include <Eigen/Dense>

template <typename Scalar>
class MatOpSymLower: public MatOp<Scalar>
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Map<const Matrix> MapMat;
    typedef Eigen::Map< Eigen::Matrix<Scalar, Eigen::Dynamic, 1> > MapVec;

    const MapMat mat;
    MapVec vec_x;
    MapVec vec_y;

public:
    MatOpSymLower(const Matrix &mat_) :
        MatOp<Scalar>(mat_.rows(), mat_.cols()),
        mat(mat_.data(), mat_.rows(), mat_.cols()),
        vec_x(NULL, 1),
        vec_y(NULL, 1)
    {}

    virtual ~MatOpSymLower() {}

    // y_out = A * x_in
    void prod(Scalar *x_in, Scalar *y_out)
    {
        new (&vec_x) MapVec(x_in, mat.cols());
        new (&vec_y) MapVec(y_out, mat.rows());

        vec_y.noalias() = mat.template selfadjointView<Eigen::Lower>() * vec_x;
    }
};


template <typename Scalar>
class MatOpXX: public MatOp<Scalar>
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Map<const Matrix> MapMat;
    typedef Eigen::Map< Eigen::Matrix<Scalar, Eigen::Dynamic, 1> > MapVec;

    const MapMat mat;
    const bool is_wide;
    const int dim;
    MapVec vec_x;
    MapVec vec_y;

public:
    MatOpXX(const Matrix &mat_) :
        MatOp<Scalar>(std::min(mat_.rows(), mat_.cols()), std::min(mat_.rows(), mat_.cols())),
        mat(mat_.data(), mat_.rows(), mat_.cols()),
        is_wide(mat_.cols() > mat_.rows()),
        dim(std::min(mat_.rows(), mat_.cols())),
        vec_x(NULL, 1),
        vec_y(NULL, 1)
    {}

    virtual ~MatOpXX() {}

    // y_out = A * x_in
    void prod(Scalar *x_in, Scalar *y_out)
    {
        new (&vec_x) MapVec(x_in, dim);
        new (&vec_y) MapVec(y_out, dim);

        if(is_wide)
        {
            Vector tmp = mat.transpose() * vec_x;
            vec_y = mat * tmp;
        } else {
            Vector tmp = mat * vec_x;
            vec_y = mat.transpose() * tmp;
        }

    }
};


#endif // MATOPDENSE_H
