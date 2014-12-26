#include "ADMMLasso.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::ArrayXd;
using Rcpp::as;
using Rcpp::List;
using Rcpp::Named;

RcppExport SEXP admm_lasso(SEXP x_, SEXP y_, SEXP lambda_,
                           SEXP opts_)
{
BEGIN_RCPP

    MatrixXd datX(as<MatrixXd>(x_));
    VectorXd datY(as<VectorXd>(y_));
    // In glmnet, we minimize
    //   1/(2n) * ||y - X * beta||^2 + lambda * ||beta||_1
    // which is equivalent to minimizing
    //   1/2 * ||y - X * beta||^2 + n * lambda * ||beta||_1
    double lambda = as<double>(lambda_) * datX.rows();
    
    List opts(opts_);
    int maxit = as<int>(opts["maxit"]);
    double eps_abs = as<double>(opts["eps_abs"]);
    double eps_rel = as<double>(opts["eps_rel"]);
    double rho = as<double>(opts["rho"]);

    /*
    // Standardize datY
    double meanY = datY.mean();
    datY.array() -= meanY;
    double scaleY = datY.norm() / sqrt(double(datY.size()));
    datY.array() /= scaleY;
    
    // Standardize datX
    ArrayXd meanX = datX.colwise().mean();
    for(int i = 0; i < datX.cols(); i++)
    {
        datX.col(i).array() -= meanX[i];
    }
    ArrayXd scaleX = datX.colwise().norm() / sqrt(double(datX.rows()));
    for(int i = 0; i < datX.cols(); i++)
    {
        datX.col(i).array() /= scaleX[i];
    }
    */
    
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
    
    /*
    ArrayXd beta(datX.cols() + 1);
    beta.segment(1, datX.cols()) = solver.get_z().array() / scaleX * scaleY;
    */
    
    return List::create(Named("coef") = solver.get_z(),
                        Named("niter") = i);

END_RCPP
}
