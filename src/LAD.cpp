#include "ADMMLAD.h"
#include "DataStd.h"

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

RcppExport SEXP admm_lad(SEXP x_, SEXP y_, SEXP intercept_, SEXP opts_)
{
BEGIN_RCPP

    MatrixXd datX(as<MatrixXd>(x_));
    VectorXd datY(as<VectorXd>(y_));

    int n = datX.rows();
    int p = datX.cols();

    List opts(opts_);
    int maxit = as<int>(opts["maxit"]);
    double eps_abs = as<double>(opts["eps_abs"]);
    double eps_rel = as<double>(opts["eps_rel"]);
    double rho = as<double>(opts["rho"]);

    bool intercept = as<bool>(intercept_);

    DataStd<double> datstd(n, p, true, intercept);
    datstd.standardize(datX, datY);

    ADMMLAD solver(datX, datY, rho, eps_abs, eps_rel);

    int niter = solver.solve(maxit);
    ArrayXd beta(p + 1);
    beta.tail(p) = solver.get_x();
    datstd.recover(beta[0], beta.tail(p));

    return List::create(Named("beta") = beta,
                        Named("niter") = niter);

END_RCPP
}
