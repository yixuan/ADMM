#include "PADMMLasso.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::ArrayXd;
using Eigen::ArrayXXd;
using Rcpp::as;
using Rcpp::List;
using Rcpp::IntegerVector;
using Rcpp::NumericVector;
using Rcpp::Named;

typedef Eigen::Map<MatrixXd> MapMat;
typedef Eigen::Map<VectorXd> MapVec;
typedef Eigen::Map<ArrayXd>  MapArray;

RcppExport SEXP admm_parlasso(SEXP x_, SEXP y_, SEXP n_, SEXP p_, SEXP lambda_,
                              SEXP nlambda_, SEXP lmin_ratio_,
                              SEXP standardize_, SEXP intercept_,
                              SEXP opts_)
{
BEGIN_RCPP

    List datX(x_);
    List datY(y_);
    
    int n = as<int>(n_);
    int p = as<int>(p_);
    ArrayXd lambda(as<ArrayXd>(lambda_));
    int nlambda = lambda.size();
    
    List opts(opts_);
    int maxit = as<int>(opts["maxit"]);
    double eps_abs = as<double>(opts["eps_abs"]);
    double eps_rel = as<double>(opts["eps_rel"]);
    double rho_ratio = as<double>(opts["rho_ratio"]);

    bool standardize = as<bool>(standardize_);
    bool intercept = as<bool>(intercept_);
    
    PADMMLasso_Master solver(datX, datY, p, eps_abs, eps_rel);
    if(nlambda < 1)
    {
        double lmax = solver.lambda_max() / n;
        double lmin = as<double>(lmin_ratio_) * lmax;
        lambda.setLinSpaced(as<int>(nlambda_), log(lmax), log(lmin));
        lambda = lambda.exp();
        nlambda = lambda.size();
    }

    ArrayXXd beta(p + 1, nlambda);
    IntegerVector niter(nlambda);
    double ilambda = 0.0;

    for(int i = 0; i < nlambda; i++)
    {
        ilambda = lambda[i] * n;
        if(i == 0)
            solver.init(ilambda, rho_ratio * ilambda);
        else
            solver.init_warm(ilambda);

        niter[i] = solver.solve(maxit);
        beta(0, i) = 0;
        beta.col(i).segment(1, p) = solver.get_z();
    }

    return List::create(Named("lambda") = lambda,
                        Named("beta") = beta,
                        Named("niter") = niter);

END_RCPP
}
