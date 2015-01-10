#include "ADMMLasso.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::ArrayXd;
using Eigen::DenseBase;
using Rcpp::as;
using Rcpp::List;
using Rcpp::Named;

typedef Eigen::Map<MatrixXd> MapMat;
typedef Eigen::Map<VectorXd> MapVec;
typedef Eigen::Map<ArrayXd>  MapArray;

template <typename Derived>
inline double sd_n(const DenseBase<Derived> &v)
{
    double mean = v.mean();
    double s = 0.0, tmp = 0.0;
    int n = v.size();
    for(int i = 0; i < n; i++)
    {
        tmp = v[i] - mean;
        s += tmp * tmp;
    }
    s /= n;
    return sqrt(s);
}

RcppExport SEXP admm_lasso(SEXP x_, SEXP y_, SEXP lambda_,
                           SEXP standardize_, SEXP intercept_,
                           SEXP opts_)
{
BEGIN_RCPP

    MatrixXd datX(as<MatrixXd>(x_));
    VectorXd datY(as<VectorXd>(y_));
    
    // In glmnet, we minimize
    //   1/(2n) * ||y - X * beta||^2 + lambda * ||beta||_1
    // which is equivalent to minimizing
    //   1/2 * ||y - X * beta||^2 + n * lambda * ||beta||_1
    int n = datX.rows();
    int p = datX.cols();
    double lambda = as<double>(lambda_) * n;
    
    bool standardize = as<bool>(standardize_);
    bool intercept = as<bool>(intercept_);
    // flag - 0: standardize = FALSE, intercept = FALSE
    //             directly fit model
    // flag - 1: standardize = TRUE, intercept = FALSE
    //             scale x and y by their standard deviation
    // flag - 2: standardize = FALSE, intercept = TRUE
    //             center x, standardize y
    // flag - 3: standardize = TRUE, intercept = TRUE
    //             standardize x and y
    int flag = int(standardize) + 2 * int(intercept);
    
    List opts(opts_);
    int maxit = as<int>(opts["maxit"]);
    double eps_abs = as<double>(opts["eps_abs"]);
    double eps_rel = as<double>(opts["eps_rel"]);
    double rho = as<double>(opts["rho"]);

    // Standardize datY
    double meanY = 0.0, scaleY = 1.0;
    switch(flag)
    {
        case 1:
            scaleY = sd_n(datY);
            datY.array() /= scaleY;
            break;
        case 2:
        case 3:
            meanY = datY.mean();
            datY.array() -= meanY;
            scaleY = datY.norm() / sqrt(double(n));
            datY.array() /= scaleY;
            break;
        default:
            break;
    }
    
    // Standardize datX
    ArrayXd meanX(p);
    ArrayXd scaleX(p);
    switch(flag)
    {
        case 1:
            for(int i = 0; i < p; i++)
            {
                scaleX[i] = sd_n(datX.col(i));
                datX.col(i).array() /= scaleX[i];
            }
            break;
        case 2:
            meanX = datX.colwise().mean();
            for(int i = 0; i < p; i++)
            {
                datX.col(i).array() -= meanX[i];
            }
            break;
        case 3:
            meanX = datX.colwise().mean();
            for(int i = 0; i < p; i++)
            {
                datX.col(i).array() -= meanX[i];
            }
            scaleX = datX.colwise().norm() / sqrt(double(n));
            for(int i = 0; i < p; i++)
            {
                datX.col(i).array() /= scaleX[i];
            }
            break;
        default:
            break;
    }
    
    ADMMLasso solver(datX, datY, lambda / scaleY, eps_abs, eps_rel, rho);
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
    
    ArrayXd beta(p + 1);
    MapArray coef(&beta[1], p);
    coef = solver.get_z();
    switch(flag)
    {
        case 0:
            beta[0] = 0;
            break;
        case 1:
            beta[0] = 0;
            coef /= scaleX;
            coef *= scaleY;
            break;
        case 2:
            coef *= scaleY;
            beta[0] = meanY - (coef * meanX).sum();
            break;
        case 3:
            coef /= scaleX;
            coef *= scaleY;
            beta[0] = meanY - (coef * meanX).sum();
            break;
        default:
            break;
    }
    
    return List::create(Named("coef") = beta,
                        Named("niter") = i);

END_RCPP
}
