#include "ADMMBase.h"

class ADMMLasso: public ADMMBase
{
private:
    typedef Eigen::MatrixXd MatrixXd;

    const MatrixXd *datX;
    const VectorXd *datY;
    double lambda;
    
    virtual void A_mult(VectorXd &x) {}
    virtual void At_mult(VectorXd &x) {}
    virtual void B_mult(VectorXd &x) {}
    virtual double c_norm() { return 0.0; }
    virtual void residual(VectorXd &res, const VectorXd &x, const VectorXd &z)
    {
        res = x - z;
    }
    
    virtual VectorXd next_x()
    {
        MatrixXd rhs1 = (*datX).transpose() * (*datX) + rho * MatrixXd::Identity(dim_n, dim_n);
        VectorXd rhs2 = (*datX).transpose() * (*datY) + rho * aux_z - dual_y;
        VectorXd newx = rhs1.ldlt().solve(rhs2);

        return newx;
    }
    virtual VectorXd next_z()
    {
        VectorXd newz = main_x + dual_y / rho;
        soft_threshold(newz, lambda / rho);
        
        return newz;
    }
public:
    ADMMLasso(MatrixXd &datX_, VectorXd &datY_, double lambda_,
              double eps_abs_ = 1e-8,
              double eps_rel_ = 1e-8,
              double rho_ = 1e-4) :
        ADMMBase(datX_.cols(), datX_.cols(), datX_.cols(),
                 eps_abs_, eps_rel_, rho_),
        datX(&datX_), datY(&datY_), lambda(lambda_)
    {}

    static void soft_threshold(VectorXd &vec, const double &penalty)
    {
        for(int i = 0; i < vec.size(); i++)
        {
            if(vec[i] > penalty)
                vec[i] -= penalty;
            else if(vec[i] < -penalty)
                vec[i] += penalty;
            else
                vec[i] = 0;
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
    double lambda = as<double>(lambda_);
    
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