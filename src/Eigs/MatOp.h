#ifndef MATOP_H
#define MATOP_H

template <typename Scalar>
class MatOp
{
private:
    // Dimension of the matrix
    // m rows and n columns
    // In eigenvalue problems, they are assumed to be equal.
    const int m;
    const int n;

public:
    // Constructor
    MatOp(int m_, int n_) :
        m(m_), n(n_)
    {}
    // Destructor
    virtual ~MatOp() {}

    // y_out = A * x_in
    virtual void prod(Scalar *x_in, Scalar *y_out) = 0;

    int rows() { return m; }
    int cols() { return n; }
};

template <typename Scalar>
class MatOpWithTransProd: public virtual MatOp<Scalar>
{
public:
    // Constructor
    MatOpWithTransProd(int m_, int n_) :
        MatOp<Scalar>(m_, n_)
    {}
    // Destructor
    virtual ~MatOpWithTransProd() {}

    // y_out = A' * x_in
    virtual void trans_prod(Scalar *x_in, Scalar *y_out) = 0;
};

template <typename Scalar>
class MatOpWithRealShiftSolve: public virtual MatOp<Scalar>
{
public:
    // Constructor
    MatOpWithRealShiftSolve(int m_, int n_) :
        MatOp<Scalar>(m_, n_)
    {}
    // Destructor
    virtual ~MatOpWithRealShiftSolve() {}

    // setting sigma
    virtual void set_shift(Scalar sigma) = 0;
    // y_out = inv(A - sigma * I) * x_in
    virtual void shift_solve(Scalar *x_in, Scalar *y_out) = 0;
};

template <typename Scalar>
class MatOpWithComplexShiftSolve: public MatOpWithRealShiftSolve<Scalar>
{
public:
    // Constructor
    MatOpWithComplexShiftSolve(int m_, int n_) :
        MatOp<Scalar>(m_, n_),
        MatOpWithRealShiftSolve<Scalar>(m_, n_)
    {}
    // Destructor
    virtual ~MatOpWithComplexShiftSolve() {}

    // setting real shift
    virtual void set_shift(Scalar sigma)
    {
        this->set_shift(sigma, Scalar(0));
    }
    // setting complex shift
    virtual void set_shift(Scalar sigmar, Scalar sigmai) = 0;
};


#endif // MATOP_H
