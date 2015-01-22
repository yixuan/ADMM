#ifndef ADMMLASSO_H
#define ADMMLASSO_H

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
    typedef Eigen::VectorXd VectorXd;
    typedef Eigen::ArrayXd ArrayXd;
    typedef Eigen::HouseholderQR<MatrixXd> QRdecomp;

    double lambda;                // L1 penalty
    const VectorXd XY;            // X'y
    MatrixXd XQR;                 // QR decomposition of X
    
    virtual void A_mult(VectorXd &x) {}                     // x -> x
    virtual void At_mult(VectorXd &x) {}                    // x -> x
    virtual void B_mult(VectorXd &x) { x.noalias() = -x; }  // x -> x
    virtual double c_norm() { return 0.0; }                 // ||c||_2 = 0
    virtual void next_residual(VectorXd &res)
    {
        res.noalias() = main_x - aux_z;
    }
    
    virtual void next_x(VectorXd &res)
    {
        VectorXd b = XY + rho * aux_z - dual_y;
        VectorXd tmp = XQR.triangularView<Eigen::Upper>() * main_x;
        VectorXd r = b - XQR.triangularView<Eigen::Upper>().transpose() * tmp - rho * main_x;
        double rsq = r.squaredNorm();
        tmp.noalias() = XQR.triangularView<Eigen::Upper>() * r;
        double alpha = rsq / (rho * rsq + tmp.squaredNorm());
        res = main_x + alpha * r;
    }
    virtual void next_z(VectorXd &res)
    {
        res.noalias() = main_x + dual_y / rho;
        soft_threshold(res, lambda / rho);
    }
    virtual void rho_changed_action() {}
    
public:
    ADMMLasso(const MatrixXd &datX_, const VectorXd &datY_,
              double eps_abs_ = 1e-6,
              double eps_rel_ = 1e-6) :
        ADMMBase(datX_.cols(), datX_.cols(), datX_.cols(),
                 eps_abs_, eps_rel_),
        XY(datX_.transpose() * datY_)
    {
        QRdecomp decomp(datX_);
        XQR = decomp.matrixQR().topRows(std::min(datX_.cols(), datX_.rows()));
    }

    virtual double lambda_max() { return XY.array().abs().maxCoeff(); }

    // init() is a cold start for the first lambda
    virtual void init(double lambda_, double rho_)
    {
        main_x.setZero();
        aux_z.setZero();
        dual_y.setZero();
        lambda = lambda_;
        rho = rho_;
        eps_primal = 0.0;
        eps_dual = 0.0;
        resid_primal = 9999;
        resid_dual = 9999;

        rho_changed_action();
    }
    // when computing for the next lambda, we can use the
    // current main_x, aux_z, dual_y and rho as initial values
    virtual void init_warm(double lambda_)
    {
        lambda = lambda_;
        update_z();
        update_y();
        eps_primal = 0.0;
        eps_dual = 0.0;
        resid_primal = 9999;
        resid_dual = 9999;
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



#endif // ADMMLASSO_H
