#include "PADMMLasso.h"
#include "DataStd.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::ArrayXf;
using Eigen::ArrayXd;
using Eigen::ArrayXXf;

using Rcpp::wrap;
using Rcpp::as;
using Rcpp::List;
using Rcpp::Named;
using Rcpp::IntegerVector;

typedef Eigen::SparseVector<float> SpVec;
typedef Eigen::SparseMatrix<float> SpMat;

inline void write_beta_matrix(SpMat &betas, int col, float beta0, SpVec &coef)
{
    betas.insert(0, col) = beta0;

    for(SpVec::InnerIterator iter(coef); iter; ++iter)
    {
        betas.insert(iter.index() + 1, col) = iter.value();
    }
}

RcppExport SEXP admm_parlasso(SEXP x_, SEXP y_, SEXP lambda_,
                              SEXP nlambda_, SEXP lmin_ratio_,
                              SEXP standardize_, SEXP intercept_,
                              SEXP nthread_, SEXP opts_)
{
BEGIN_RCPP

    Rcpp::NumericMatrix xx(x_);
    Rcpp::NumericVector yy(y_);

    const int n = xx.rows();
    const int p = xx.cols();

    MatrixXf datX(n, p);
    VectorXf datY(n);

    // Copy data and convert type from double to float
    std::copy(xx.begin(), xx.end(), datX.data());
    std::copy(yy.begin(), yy.end(), datY.data());

    // In glmnet, we minimize
    //   1/(2n) * ||y - X * beta||^2 + lambda * ||beta||_1
    // which is equivalent to minimizing
    //   1/2 * ||y - X * beta||^2 + n * lambda * ||beta||_1
    ArrayXd lambda(as<ArrayXd>(lambda_));
    int nlambda = lambda.size();

    List opts(opts_);
    const int maxit        = as<int>(opts["maxit"]);
    const double eps_abs   = as<double>(opts["eps_abs"]);
    const double eps_rel   = as<double>(opts["eps_rel"]);
    const double rho       = as<double>(opts["rho"]);
    const bool standardize = as<bool>(standardize_);
    const bool intercept   = as<bool>(intercept_);

    DataStd<float> datstd(n, p, standardize, intercept);
    datstd.standardize(datX, datY);

    const int nthread = as<int>(nthread_);
    PADMMLasso_Master solver(datX, datY, nthread, eps_abs, eps_rel);

    if(nlambda < 1)
    {
        double lmax = solver.get_lambda_zero() / n * datstd.get_scaleY();
        double lmin = as<double>(lmin_ratio_) * lmax;
        lambda.setLinSpaced(as<int>(nlambda_), std::log(lmax), std::log(lmin));
        lambda = lambda.exp();
        nlambda = lambda.size();
    }

    SpMat beta(p + 1, nlambda);
    beta.reserve(Eigen::VectorXi::Constant(nlambda, std::min(n, p)));

    IntegerVector niter(nlambda);
    double ilambda = 0.0;

    for(int i = 0; i < nlambda; i++)
    {
        ilambda = lambda[i] * n / datstd.get_scaleY();
        if(i == 0)
            solver.init(ilambda, rho);
        else
            solver.init_warm(ilambda);

        niter[i] = solver.solve(maxit);
        SpVec res = solver.get_z();
        float beta0 = 0.0;
        datstd.recover(beta0, res);
        write_beta_matrix(beta, i, beta0, res);
    }

    beta.makeCompressed();

    return List::create(Named("lambda") = lambda,
                        Named("beta") = beta,
                        Named("niter") = niter);

END_RCPP
}
