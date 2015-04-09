#ifndef ADMMLASSO_H
#define ADMMLASSO_H

#include "ADMMBase.h"
#include "Eigs/SymEigsSolver.h"
#include "Eigs/MatOpDense.h"

// minimize  1/2 * ||y - X * beta||^2 + lambda * ||beta||_1
//
// In ADMM form,
//   minimize f(x) + g(z)
//   s.t. x - z = 0
//
// x => beta
// z => -X * beta
// A => X
// b => y
// f(x) => 1/2 * ||Ax - b||^2
// g(z) => lambda * ||z||_1
class ADMMLasso: public ADMMBase< Eigen::VectorXd, Eigen::SparseVector<double> >
{
protected:
    typedef Eigen::MatrixXd MatrixXd;
    typedef Eigen::VectorXd VectorXd;
    typedef Eigen::ArrayXd ArrayXd;
    typedef Eigen::Map<const MatrixXd> MapMat;
    typedef Eigen::Map<const VectorXd> MapVec;
    typedef Eigen::SparseVector<double> SparseVector;
    typedef Eigen::LLT<MatrixXd> LLT;

    const MapMat datX;            // pointer to data matrix
    const MapVec datY;            // pointer response vector
    const MatrixXd XX;            // X'X
    const VectorXd XY;            // X'Y
    LLT solver;                   // matrix factorization

    double lambda;                // L1 penalty
    double lambda0;               // minimum lambda to make coefficients all zero



    // x -> Ax
    void A_mult (VectorXd &res, VectorXd &x)  { res.swap(x); }
    // y -> A'y
    void At_mult(VectorXd &res, VectorXd &y)  { res.swap(y); }
    // z -> Bz
    void B_mult (VectorXd &res, SparseVector &z) { res = -z; }
    // ||c||_2
    double c_norm() { return 0.0; }



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
    void next_x(VectorXd &res)
    {
        VectorXd rhs = XY - adj_y;
        rhs += rho * adj_z;
        res = solver.solve(rhs);
    }
    void next_z(SparseVector &res)
    {
        VectorXd vec = main_x + adj_y / rho;
        soft_threshold(res, vec, lambda / rho);
    }
    void next_residual(VectorXd &res)
    {
        res = main_x;
        res -= aux_z;
    }
    void rho_changed_action()
    {
        solver.compute(XX + rho * MatrixXd::Identity(dim_main, dim_main));
    }



    // Faster computation of epsilons and residuals
    double compute_eps_primal()
    {
        double r = std::max(main_x.norm(), aux_z.norm());
        return r * eps_rel + std::sqrt(double(dim_dual)) * eps_abs;
    }
    double compute_eps_dual()
    {
        return dual_y.norm() * eps_rel + sqrt(double(dim_main)) * eps_abs;
    }
    double compute_resid_dual(SparseVector &zdiff)
    {
        return rho * zdiff.norm();
    }
    double compute_resid_combined()
    {
        SparseVector tmp = aux_z - adj_z;
        return rho * resid_primal * resid_primal + rho * tmp.squaredNorm();
    }

public:
    ADMMLasso(const MatrixXd &datX_, const VectorXd &datY_,
              double eps_abs_ = 1e-6,
              double eps_rel_ = 1e-6) :
        ADMMBase(datX_.cols(), datX_.cols(), datX_.cols(),
                 eps_abs_, eps_rel_),
        datX(datX_.data(), datX_.rows(), datX_.cols()),
        datY(datY_.data(), datY_.size()),
        XX(datX.transpose() * datX),
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
        XX(datX.transpose() * datX),
        XY(datX.transpose() * datY)
    {
        lambda0 = XY.array().abs().maxCoeff();
    }

    double get_lambda_zero() { return lambda0; }

    // init() is a cold start for the first lambda
    void init(double lambda_, double rho_rel_)
    {
        main_x.setZero();
        aux_z.setZero();
        dual_y.setZero();

        adj_z.setZero();
        adj_y.setZero();

        lambda = lambda_;
        rho = lambda_ * rho_rel_;

        eps_primal = 0.0;
        eps_dual = 0.0;
        resid_primal = 9999;
        resid_dual = 9999;

        adj_a = 1.0;
        adj_c = 9999;

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

        // adj_a = 1.0;
        // adj_c = 9999;
    }
};



#endif // ADMMLASSO_H
