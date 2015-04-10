#include "ADMMBP.h"

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

typedef Eigen::Map<const MatrixXd> MapMat;
typedef Eigen::Map<const VectorXd> MapVec;
typedef Eigen::SparseVector<double> SpVec;
typedef Eigen::SparseMatrix<double> SpMat;

RcppExport SEXP admm_bp(SEXP x_, SEXP y_, SEXP opts_)
{
BEGIN_RCPP

#if ADMM_PROFILE > 0
    clock_t t1, t2;
    t1 = clock();
#endif

    NumericMatrix x(x_);
    NumericVector y(y_);
    const MapMat datX(x.begin(), x.nrow(), x.ncol());
    const MapVec datY(y.begin(), y.length());

    List opts(opts_);
    int maxit = as<int>(opts["maxit"]);
    double eps_abs = as<double>(opts["eps_abs"]);
    double eps_rel = as<double>(opts["eps_rel"]);
    double rho = as<double>(opts["rho"]);

#if ADMM_PROFILE > 0
    t2 = clock();
    Rcpp::Rcout << "part1: " << double(t2 - t1) / CLOCKS_PER_SEC << " secs.\n";
#endif

    ADMMBP solver(datX, datY, rho, eps_abs, eps_rel);

#if ADMM_PROFILE > 0
    t1 = clock();
    Rcpp::Rcout << "part2: " << double(t1 - t2) / CLOCKS_PER_SEC << " secs.\n";
#endif

    int niter = solver.solve(maxit);
    SpMat beta(datX.cols(), 1);
    beta.col(0) = solver.get_z();
    beta.makeCompressed();

#if ADMM_PROFILE > 0
    t2 = clock();
    Rcpp::Rcout << "part3: " << double(t2 - t1) / CLOCKS_PER_SEC << " secs.\n";
#endif

    return List::create(Named("beta") = beta,
                        Named("niter") = niter);

END_RCPP
}
