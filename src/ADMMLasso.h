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
class ADMMLasso: public ADMMBase<Eigen::SparseVector<double>, Eigen::VectorXd>
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
    double sprad;                 // spectral radius of X'X
    double lambda;                // L1 penalty
    double lambda0;               // minimum lambda to make coefficients all zero

    int iter_counter;             // which iteration are we in?

    VectorXd cache_Ax;            // cache Ax

    void A_mult(VectorXd &res, SparseVector &x) // x -> Ax
    {
        res.noalias() = datX * x;
    }
    void At_mult(VectorXd &res, VectorXd &y) // y -> A'y
    {
        // The correct operation should be the line below
        //     res.noalias() = (*datX).transpose() * y;
        // However, it is too expensive to calculate
        // A'y (in function compute_eps_dual())
        // and A'(newz - oldz) (in function update_z())
        // in every iteration.
        // Instead, we simply use ||newz - oldz||_2
        // and ||y||_2 to calculate dual residual and
        // eps_dual.
        // In this case, At_mult will be an identity transformation.
        res.swap(y);
    }
    void B_mult (VectorXd &res, VectorXd &z) // z -> Bz
    {
        res.swap(z);
    }
    double c_norm() { return 0.0; } // ||c||_2
    void next_residual(VectorXd &res)
    {
        res.noalias() = cache_Ax + aux_z;
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

    virtual void active_set_update(SparseVector &res)
    {
        double gamma = 2 * rho + sprad;
        double penalty = lambda / (rho * gamma);
        VectorXd vec = (cache_Ax + aux_z + dual_y / rho) / gamma;
        res = main_x;

        for(SparseVector::InnerIterator iter(res); iter; ++iter)
        {
            double val = iter.value() - vec.dot(datX.col(iter.index()));

            if(val > penalty)
                iter.valueRef() = val - penalty;
            else if(val < -penalty)
                iter.valueRef() = val + penalty;
            else
                iter.valueRef() = 0.0;
        }

        res.prune(0.0);
    }

    virtual void regular_update(SparseVector &res)
    {
        double gamma = 2 * rho + sprad;
        VectorXd tmp = cache_Ax + aux_z + dual_y / rho;
        VectorXd vec = datX.transpose() * tmp;
        vec /= -gamma;
        vec += main_x;
        soft_threshold(res, vec, lambda / (rho * gamma));
    }

    void next_x(SparseVector &res)
    {
        if(iter_counter % 10 == 0 && lambda < lambda0)
        {
            regular_update(res);
        } else {
            active_set_update(res);
        }
        iter_counter++;
    }
    void next_z(VectorXd &res)
    {
        cache_Ax = datX * main_x;
        res.noalias() = (datY + dual_y + rho * cache_Ax) / (-1 - rho);
    }
    void rho_changed_action() {}
    // a faster version compared to the base implementation
    double compute_eps_primal()
    {
        double r = std::max(cache_Ax.norm(), aux_z.norm());
        return r * eps_rel + sqrt(double(dim_dual)) * eps_abs;
    }
    // a faster version compared to the base implementation
    double compute_eps_dual()
    {
        return dual_y.norm() * eps_rel + sqrt(double(dim_dual)) * eps_abs;
    }

    // calculating the spectral radius of X'X
    // in this case it is the largest eigenvalue of X'X
    static double spectral_radius(const MatrixXd &X)
    {
        MatOpXX<double> op(X);
        SymEigsSolver<double, LARGEST_ALGE> eigs(&op, 1, 5);
        srand(0);
        eigs.init();
        eigs.compute(100, 0.1);
        VectorXd eval = eigs.eigenvalues();

        return eval[0];
    }

public:
    ADMMLasso(const MatrixXd &datX_, const VectorXd &datY_,
              double eps_abs_ = 1e-6,
              double eps_rel_ = 1e-6) :
        ADMMBase(datX_.cols(), datX_.rows(), datX_.rows(),
                 eps_abs_, eps_rel_),
        datX(datX_.data(), datX_.rows(), datX_.cols()),
        datY(datY_.data(), datY_.size()),
        cache_Ax(dim_dual)
    {
        lambda0 = (datX.transpose() * datY).array().abs().maxCoeff();
        sprad = spectral_radius(datX_);
    }

    ADMMLasso(const double *datX_, const double *datY_,
              int n_, int p_,
              double eps_abs_ = 1e-6,
              double eps_rel_ = 1e-6) :
        ADMMBase(p_, n_, n_, eps_abs_, eps_rel_),
        datX(datX_, n_, p_),
        datY(datY_, n_),
        cache_Ax(dim_dual)
    {
        lambda0 = (datX.transpose() * datY).array().abs().maxCoeff();
        MatrixXd X = datX;
        sprad = spectral_radius(X);
    }

    double get_lambda_zero() { return lambda0; }

    // init() is a cold start for the first lambda
    void init(double lambda_, double rho_ratio_)
    {
        main_x.setZero();
        cache_Ax.setZero();
        aux_z.setZero();
        dual_y.setZero();
        lambda = lambda_;
        rho = lambda_ / (rho_ratio_ * sprad);
        eps_primal = 0.0;
        eps_dual = 0.0;
        resid_primal = 9999;
        resid_dual = 9999;
        iter_counter = 0;

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
        iter_counter = 0;
    }
};



#endif // ADMMLASSO_H
