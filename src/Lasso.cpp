#include "ADMMBase.h"

class ADMMLasso: public ADMMBase
{
private:
    typedef Eigen::MatrixXd MatrixXd;
    typedef Eigen::LLT<MatrixXd> LLT;

    const MatrixXd *datX;  // data matrix
    const VectorXd *datY;  // response vector
    double lambda;         // L1 penalty
    
    LLT solver;  // matrix factorization
    
    virtual void A_mult(VectorXd &x) {}  // x -> x
    virtual void At_mult(VectorXd &x) {} // x -> x
    virtual void B_mult(VectorXd &x) {}  // x -> x
    virtual double c_norm() { return 0.0; }  // ||c||_2 = 0
    virtual void next_residual(VectorXd &res, const VectorXd &x, const VectorXd &z)
    {
        res = x - z;
    }
    
    virtual void next_x(VectorXd &res)
    {
        VectorXd rhs = (*datX).transpose() * (*datY) + rho * aux_z - dual_y;
        res = solver.solve(rhs);
    }
    virtual void next_z(VectorXd &res)
    {
        res = main_x + dual_y / rho;
        soft_threshold(res, lambda / rho);
    }
    virtual void rho_changed_action()
    {
        MatrixXd mat = (*datX).transpose() * (*datX);
        mat.diagonal().array() = mat.diagonal().array() + rho;
        solver.compute(mat);
    }
    
public:
    ADMMLasso(MatrixXd &datX_, VectorXd &datY_, double lambda_,
              double eps_abs_ = 1e-8,
              double eps_rel_ = 1e-8,
              double rho_ = 1e-4) :
        ADMMBase(datX_.cols(), datX_.cols(), datX_.cols(),
                 eps_abs_, eps_rel_, rho_),
        datX(&datX_), datY(&datY_), lambda(lambda_ * datX_.rows())
    {
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