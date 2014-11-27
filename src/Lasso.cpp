#include "ADMMBase.h"

// minimize  1/2 * ||y - X * beta||^2 + lambda * ||beta||_1
//
// In ADMM form,
//   minimize f(x) + g(z)
//   s.t. x - z = 0
//
// f(beta) = 1/2 * ||y - X * beta||^2
// g(z) = lambda * ||z||_1
class ADMMLasso: public ADMMBase
{
private:
    typedef Eigen::MatrixXd MatrixXd;
    typedef Eigen::ArrayXd ArrayXd;
    typedef Eigen::LLT<MatrixXd> LLT;

    const MatrixXd *datX;         // data matrix
    const double lambda;          // L1 penalty
    const bool thinX;             // whether nrow(X) > ncol(X)
    const VectorXd cache_matvec;  // cache X'y
    MatrixXd cache_matprod;       // cache X'X if thinX = true,
                                  // or XX' if thinX = false
    ArrayXd cache_diag;           // diagonal elments of cache_matprod
    LLT solver;                   // matrix factorization
    
    virtual void A_mult(VectorXd &x) {}  // x -> x
    virtual void At_mult(VectorXd &x) {} // x -> x
    virtual void B_mult(VectorXd &x) {}  // x -> x
    virtual double c_norm() { return 0.0; }  // ||c||_2 = 0
    virtual void next_residual(VectorXd &res, const VectorXd &x, const VectorXd &z)
    {
        res.noalias() = x - z;
    }
    
    virtual void next_x(VectorXd &res)
    {
        // For a thin X,
        //   rhs = X'y + rho * aux_z - dual_y
        //   newx = inv(X'X + rho * I) * rhs
        //
        // For a wide X,
        //   inv(X'X + rho * I) = 1/rho * I -
        //       1/rho * X' * inv(XX' + rho * I) * X
        // so
        //   newx = 1/rho * rhs - 1/rho * X' * inv(XX' + rho * I) * X * rhs
        
        VectorXd rhs = cache_matvec + rho * aux_z - dual_y;
        if(thinX)
        {
            res.noalias() = solver.solve(rhs);
        } else {
            res = rhs;
            res.noalias() -= (*datX).transpose() * solver.solve((*datX) * rhs);
            res /= rho;
        }
    }
    virtual void next_z(VectorXd &res)
    {
        res.noalias() = main_x + dual_y / rho;
        soft_threshold(res, lambda / rho);
    }
    virtual void rho_changed_action()
    {
        cache_matprod.diagonal() = cache_diag + rho;
        solver.compute(cache_matprod);
    }
    
public:
    ADMMLasso(const MatrixXd &datX_, const VectorXd &datY_,
              double lambda_,
              double eps_abs_ = 1e-8,
              double eps_rel_ = 1e-8,
              double rho_ = 1e-3) :
        ADMMBase(datX_.cols(), datX_.cols(), datX_.cols(),
                 eps_abs_, eps_rel_, rho_),
        datX(&datX_), lambda(lambda_),
        thinX(datX_.rows() > datX_.cols()),
        cache_matvec(datX_.transpose() * datY_)
    {
        if(thinX)
            cache_matprod = datX_.transpose() * datX_;
        else
            cache_matprod = datX_ * datX_.transpose();
        
        cache_diag = cache_matprod.diagonal();
        rho_changed_action();
    }

    static void soft_threshold(VectorXd &vec, const double &penalty)
    {
        double *ptr = vec.data();
        for(int i = 0; i < vec.size(); i++)
        {
            if(ptr[i] > penalty)
                ptr[i] -= penalty;
            else if(ptr[i] < -penalty)
                ptr[i] += penalty;
            else
                ptr[i] = 0;
        }
    }
};



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