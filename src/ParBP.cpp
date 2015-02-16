#include "PADMMBP.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::ArrayXd;

using Rcpp::wrap;
using Rcpp::as;
using Rcpp::List;
using Rcpp::IntegerVector;
using Rcpp::NumericVector;
using Rcpp::NumericMatrix;
using Rcpp::Named;

typedef Eigen::Map<MatrixXd> MapMat;
typedef Eigen::Map<VectorXd> MapVec;
typedef Eigen::Map<ArrayXd>  MapArray;
typedef Eigen::SparseVector<double> SpVec;
typedef Eigen::SparseMatrix<double> SpMat;

RcppExport SEXP admm_parbp(SEXP x_, SEXP y_, SEXP nthread_, SEXP opts_)
{
BEGIN_RCPP

#if ADMM_PROFILE > 0
    clock_t t1, t2;
    t1 = clock();
#endif

    MatrixXd datX(as<MatrixXd>(x_));
    VectorXd datY(as<VectorXd>(y_));

    List opts(opts_);
    int maxit = as<int>(opts["maxit"]);
    double eps_abs = as<double>(opts["eps_abs"]);
    double eps_rel = as<double>(opts["eps_rel"]);
    double rho_ratio = as<double>(opts["rho_ratio"]);

#if ADMM_PROFILE > 0
    t2 = clock();
    Rcpp::Rcout << "part1: " << double(t2 - t1) / CLOCKS_PER_SEC << " secs.\n";
#endif

    int nthread = as<int>(nthread_);
#ifdef _OPENMP
    omp_set_num_threads(nthread);
#endif

    PADMMBP_Master solver(datX, datY, nthread, eps_abs, eps_rel);
    solver.init(rho_ratio);

#if ADMM_PROFILE > 0
    t1 = clock();
    Rcpp::Rcout << "part2: " << double(t1 - t2) / CLOCKS_PER_SEC << " secs.\n";
#endif

    int niter = solver.solve(maxit);
    SpMat beta(datX.cols(), 1);
    beta.col(0) = solver.get_x();
    beta.makeCompressed();

#if ADMM_PROFILE > 0
    t2 = clock();
    Rcpp::Rcout << "part3: " << double(t2 - t1) / CLOCKS_PER_SEC << " secs.\n";
#endif

    return List::create(Named("beta") = beta,
                        Named("niter") = niter);

END_RCPP
}
