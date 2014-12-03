#include "ADMMLasso.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Rcpp::as;
using Rcpp::List;
using Rcpp::Named;

RcppExport SEXP admm_lasso(SEXP x_, SEXP y_, SEXP lambda_,
                           SEXP maxit_,
                           SEXP eps_abs_, SEXP eps_rel_, SEXP rho_)
{
BEGIN_RCPP

    MatrixXd datX(as<MatrixXd>(x_));
    VectorXd datY(as<VectorXd>(y_));
    // In glmnet, we minimize
    //   1/(2n) * ||y - X * beta||^2 + lambda * ||beta||_1
    // which is equivalent to minimizing
    //   1/2 * ||y - X * beta||^2 + n * lambda * ||beta||_1
    double lambda = as<double>(lambda_) * datX.rows();
    
    int maxit = as<int>(maxit_);
    double eps_abs = as<double>(eps_abs_);
    double eps_rel = as<double>(eps_rel_);
    double rho = as<double>(rho_);
    
    ADMMLasso solver(datX, datY, lambda, eps_abs, eps_rel, rho);
    int i;
    for(i = 0; i < maxit; i++)
    {
        solver.update_x();
        solver.update_z();
        solver.update_y();
        
        // solver.debuginfo();
        
        if(solver.converged())
            break;
    }

    return List::create(Named("x") = solver.get_x(),
                        Named("z") = solver.get_z(),
                        Named("y") = solver.get_y(),
                        Named("niter") = i);

END_RCPP
}
