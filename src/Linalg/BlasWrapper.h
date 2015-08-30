#ifndef BLASWRAPPER_H
#define BLASWRAPPER_H

#include <Eigen/Core>

namespace Linalg {



extern "C"
{
    void dgemv_(const char* transA, const int* m, const int* n,
                const double* alpha, const double* A, const int* ldA,
                const double* x, const int* incx,
                const double* beta, double* y, const int* incy);
    void dgemm_(const char* transA, const char* transB, const int* m, const int* n, const int* k,
                const double* alpha, const double* A, const int* ldA, const double* B, const int* ldB,
                const double* beta, double* C, const int* ldC);
    void dsyrk_(const char* uplo, const char* transA, const int* n, const int* k,
                const double* alpha, const double* A, const int* ldA,
                const double* beta, double* C, const int* ldC);
}

inline void mat_vec_prod(Eigen::VectorXd &res, const Eigen::MatrixXd &X, const Eigen::VectorXd &v)
{
    const double one = 1.0;
    const double zero = 0.0;
    const int n = X.rows();
    const int p = X.cols();
    const int inc = 1;

    res.resize(n);
    dgemv_("N", &n, &p, &one, X.data(), &n, v.data(), &inc, &zero, res.data(), &inc);
}

inline void mat_vec_prod(double *res, const double *X, const double *v, int n, int p)
{
    const double one = 1.0;
    const double zero = 0.0;
    const int inc = 1;

    dgemv_("N", &n, &p, &one, X, &n, v, &inc, &zero, res, &inc);
}

inline void mat_vec_tprod(Eigen::VectorXd &res, const Eigen::MatrixXd &X, const Eigen::VectorXd &v)
{
    const double one = 1.0;
    const double zero = 0.0;
    const int n = X.rows();
    const int p = X.cols();
    const int inc = 1;

    res.resize(p);
    dgemv_("T", &n, &p, &one, X.data(), &n, v.data(), &inc, &zero, res.data(), &inc);
}

inline void mat_vec_tprod(double *res, const double *X, const double *v, int n, int p)
{
    const double one = 1.0;
    const double zero = 0.0;
    const int inc = 1;

    dgemv_("T", &n, &p, &one, X, &n, v, &inc, &zero, res, &inc);
}

// Calculating X'X
inline void cross_prod(Eigen::MatrixXd &res, const Eigen::Map<const Eigen::MatrixXd> &X)
{
    const double one = 1.0;
    const double zero = 0.0;
    const char trans = 'T';
    const char no_trans = 'N';

    const int n = X.rows();
    const int p = X.cols();
    const double *x_ptr = X.data();

    res.resize(p, p);
    double *res_ptr = res.data();

    dgemm_(&trans, &no_trans, &p, &p, &n,
           &one, x_ptr, &n, x_ptr, &n,
           &zero, res_ptr, &p);
}

inline void cross_prod_lower(Eigen::MatrixXd &res, const Eigen::Map<const Eigen::MatrixXd> &X)
{
    const double one = 1.0;
    const double zero = 0.0;

    const int n = X.rows();
    const int p = X.cols();
    const double *x_ptr = X.data();

    res.resize(p, p);
    double *res_ptr = res.data();

    dsyrk_("L", "T", &p, &n,
           &one, x_ptr, &n,
           &zero, res_ptr, &p);
}

inline void tcross_prod_lower(Eigen::MatrixXd &res, const Eigen::Map<const Eigen::MatrixXd> &X)
{
    const double one = 1.0;
    const double zero = 0.0;

    const int n = X.rows();
    const int p = X.cols();
    const double *x_ptr = X.data();

    res.resize(n, n);
    double *res_ptr = res.data();

    dsyrk_("L", "N", &n, &p,
           &one, x_ptr, &n,
           &zero, res_ptr, &n);
}



} // namespace Linalg

#endif // BLASWRAPPER_H
