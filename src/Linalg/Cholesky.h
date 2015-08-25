#ifndef SYMMETRICLDL_H
#define SYMMETRICLDL_H

#include <Eigen/Core>
#include "LapackWrapper.h"

namespace Linalg {



// Cholesky decomposition of a symmetric positive definite matrix
class Cholesky
{
private:
    typedef Eigen::MatrixXd Matrix;
    typedef Eigen::VectorXd Vector;

protected:
    int dim_n;          // size of the matrix
    Matrix mat_fac;     // storing factorization structures
    char mat_uplo;      // using upper triangle or lower triangle
    bool computed;      // whether factorization has been computed
public:
    Cholesky() :
        dim_n(0), computed(false)
    {}

    Cholesky(const Matrix &mat, const char uplo = 'L') :
        dim_n(mat.rows()),
        mat_uplo(uplo),
        computed(false)
    {
        compute(mat, uplo);
    }

    void compute(const Matrix &mat, const char uplo = 'L')
    {
        dim_n = mat.rows();
        mat_fac = mat;
        mat_uplo = uplo;

        int info;
        Linalg::dpotrf_(&uplo, &dim_n, mat_fac.data(), &dim_n, &info);

        if(info == 0)
            computed = true;
    }

    // Solve Ax = b and overwrite b by x
    void solve_inplace(Vector &b)
    {
        if(!computed)
            return;

        const int nrhs = 1;
        int info;
        Linalg::dpotrs_(&mat_uplo, &dim_n, &nrhs,
                        mat_fac.data(), &dim_n, b.data(), &dim_n, &info);
    }
};



} // namespace Linalg

#endif // SYMMETRICLDL_H
