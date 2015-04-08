#ifndef ADMMLASSO_H
#define ADMMLASSO_H

#include "ADMMBase.h"
#include "Eigs/SymEigsSolver.h"
#include "Eigs/MatOpDense.h"

// minimize  1/2 * ||y - X * beta||^2 + lambda * ||beta||_1
//
// In ADMM form,
//   minimize f(x) + g(z)
//   s.t. Ax + z = c
//
// x => beta
// z => -X * beta
// A => X
// b => y
// c => 0
// f(x) => lambda * ||x||_1
// g(z) => 1/2 * ||z + b||^2
class ADMMLasso: public ADMMBase< Eigen::VectorXd, Eigen::SparseVector<double> >
{
protected:
    typedef Eigen::MatrixXd MatrixXd;
    typedef Eigen::VectorXd VectorXd;
    typedef Eigen::ArrayXd ArrayXd;
    typedef Eigen::Map<const MatrixXd> MapMat;
    typedef Eigen::Map<const VectorXd> MapVec;
    typedef Eigen::SparseVector<double> SparseVector;

    const MapMat datX;            // pointer to data matrix
    const MapVec datY;            // pointer response vector
    const VectorXd XY;            // X'Y
    double lambda;                // L1 penalty
    double lambda0;               // minimum lambda to make coefficients all zero

    void A_mult (VectorXd &res, VectorXd &x) // x -> Ax
    {
        res.swap(x);
    }
    void At_mult(VectorXd &res, VectorXd &y) // y -> A'y
    {
        res.swap(y);
    }
    void B_mult (VectorXd &res, SparseVector &z) // z -> Bz
    {
        res = -z;
    }
    double c_norm() { return 0.0; } // ||c||_2
    void next_residual(VectorXd &res)
    {
        res = main_x;
        res -= aux_z;
    }

    static void soft_threshold(SparseVector &res, VectorXd &vec, const double &penalty)
    {
        res.setZero();
        res.reserve(vec.size() / 2);

        double *ptr = vec.data();
        for(int i = 0; i < vec.size(); i++)
        {
            if(ptr[i] > penalty)
                res.insertBack(i) = ptr[i] - penalty;
            else if(ptr[i] < -penalty)
                res.insertBack(i) = ptr[i] + penalty;
        }
    }

    virtual void regular_update(VectorXd &res)
    {
        VectorXd rhs = XY - dual_y;
        rhs += rho * aux_z;
        res = (datX.transpose() * datX + rho * MatrixXd::Identity(dim_main, dim_main)).llt().solve(rhs);
    }

    void next_x(VectorXd &res)
    {
        regular_update(res);
    }
    void next_z(SparseVector &res)
    {
        VectorXd vec = main_x + dual_y / rho;
        soft_threshold(res, vec, lambda / rho);
    }
    void rho_changed_action() {}

public:
    ADMMLasso(const MatrixXd &datX_, const VectorXd &datY_,
              double eps_abs_ = 1e-6,
              double eps_rel_ = 1e-6) :
        ADMMBase(datX_.cols(), datX_.cols(), datX_.cols(),
                 eps_abs_, eps_rel_),
        datX(datX_.data(), datX_.rows(), datX_.cols()),
        datY(datY_.data(), datY_.size()),
        XY(datX.transpose() * datY)
    {
        lambda0 = XY.array().abs().maxCoeff();
    }

    ADMMLasso(const double *datX_, const double *datY_,
              int n_, int p_,
              double eps_abs_ = 1e-6,
              double eps_rel_ = 1e-6) :
        ADMMBase(p_, p_, p_, eps_abs_, eps_rel_),
        datX(datX_, n_, p_),
        datY(datY_, n_),
        XY(datX.transpose() * datY)
    {
        lambda0 = XY.array().abs().maxCoeff();
    }

    double get_lambda_zero() { return lambda0; }

    // init() is a cold start for the first lambda
    void init(double lambda_, double rho_ratio_)
    {
        main_x.setZero();
        aux_z.setZero();
        dual_y.setZero();
        lambda = lambda_;
        rho = lambda_ / rho_ratio_;
        eps_primal = 0.0;
        eps_dual = 0.0;
        resid_primal = 9999;
        resid_dual = 9999;

        rho_changed_action();
    }
    // when computing for the next lambda, we can use the
    // current main_x, aux_z, dual_y and rho as initial values
    void init_warm(double lambda_)
    {
        lambda = lambda_;
        eps_primal = 0.0;
        eps_dual = 0.0;
        resid_primal = 9999;
        resid_dual = 9999;
    }
};



#endif // ADMMLASSO_H
