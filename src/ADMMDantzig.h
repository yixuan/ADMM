#ifndef ADMMDANTZIG_H
#define ADMMDANTZIG_H

#include "ADMMBase.h"

// minimize ||beta||_1
// s.t. ||X'(X * beta - y)||_inf <= lambda
//
// In ADMM form,
//   minimize f(x) + g(z)
//   s.t. Ax + z = c
//
// x => beta
// z => -X'(X * beta - y)
// A => X'X
// c => X'y
// f(x) => ||x||_1
// g(z) => 0, dom(z) = {z: ||z||_inf <= lambda}
class ADMMDantzig: public ADMMBase<Eigen::SparseVector<double>, Eigen::VectorXd>
{
protected:
    typedef Eigen::MatrixXd MatrixXd;
    typedef Eigen::VectorXd VectorXd;
    typedef Eigen::ArrayXd ArrayXd;
    typedef Eigen::SparseVector<double> SparseVector;
    typedef Eigen::HouseholderQR<MatrixXd> QRdecomp;
    typedef Eigen::HouseholderSequence<MatrixXd, VectorXd> QRQ;

    const MatrixXd *datX;         // pointer to data matrix
    const VectorXd *datY;         // pointer response vector
    bool use_XX;                  // whether to cache X'X
    MatrixXd XX;                  // X'X
    VectorXd XY;                  // X'y
    double XY_norm;               // L2 norm of X'y

    double sprad;                 // spectral radius of X'XX'X
    double lambda;                // penalty parameter
    double lambda0;               // minimum lambda to make coefficients all zero

    int iter_counter;             // which iteration are we in?

    VectorXd cache_Ax;            // cache Ax

    virtual void A_mult(VectorXd &res, SparseVector &x) // x -> Ax
    {
        if(use_XX)
        {
            res.noalias() = XX * x;
        } else {
            VectorXd tmp = (*datX) * x;
            res.noalias() = (*datX).transpose() * tmp;
        }
    }
    virtual void At_mult(VectorXd &res, VectorXd &y) // y -> A'y
    {
        // The correct operation should be res = A'y.
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
    virtual void B_mult(VectorXd &res, VectorXd &z) // z -> Bz
    {
        res.swap(z);
    }
    virtual double c_norm() { return XY_norm; } // ||c||_2
    virtual void next_residual(VectorXd &res)
    {
        res.noalias() = cache_Ax + aux_z - XY;
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
        double penalty = 1.0 / (rho * sprad);
        VectorXd vec = (*datX) * (cache_Ax + aux_z + dual_y / rho - XY) / sprad;
        res = main_x;

        for(SparseVector::InnerIterator iter(res); iter; ++iter)
        {
            double val = iter.value() - vec.dot((*datX).col(iter.index()));

            if(val > penalty)
                iter.valueRef() = val - penalty;
            else if(val < -penalty)
                iter.valueRef() = val + penalty;
            else
                iter.valueRef() = 0.0;
        }

        res.prune(0.0);
    }

    virtual void next_x(SparseVector &res)
    {
        if(iter_counter % 5 == 0 && lambda < lambda0)
        {
            VectorXd vec = (cache_Ax + aux_z + dual_y / rho - XY) / (-sprad);
            vec = (*datX).transpose() * ((*datX) * vec);
            vec += main_x;
            soft_threshold(res, vec, 1.0 / (rho * sprad));
        } else {
            active_set_update(res);
        }
        iter_counter++;
    }
    virtual void next_z(VectorXd &res)
    {
        if(use_XX)
        {
            cache_Ax.noalias() = XX * main_x;
        } else {
            VectorXd tmp = (*datX) * main_x;
            cache_Ax.noalias() = (*datX).transpose() * tmp;
        }

        VectorXd z = cache_Ax + dual_y / rho - XY;
        for(int i = 0; i < res.size(); i++)
        {
            if(z[i] > 0)
                res[i] = -std::min(z[i], lambda);
            else
                res[i] = std::min(-z[i], lambda);
        }
    }
    virtual void rho_changed_action() {}
    // a faster version compared to the base implementation
    virtual double compute_eps_primal()
    {
        double r = std::max(cache_Ax.norm(), aux_z.norm());
        r = std::max(r, XY_norm);
        return r * eps_rel + sqrt(double(dim_dual)) * eps_abs;
    }
    // a faster version compared to the base implementation
    virtual double compute_eps_dual()
    {
        return dual_y.norm() * eps_rel + sqrt(double(dim_dual)) * eps_abs;
    }

    // calculating the spectral radius of X'X
    // in this case it is the largest eigenvalue of X'X
    static double spectral_radius_xx(const MatrixXd &X)
    {
        Rcpp::NumericMatrix mat = Rcpp::wrap(X);

        Rcpp::Environment ADMM = Rcpp::Environment::namespace_env("ADMM");
        Rcpp::Function spectral_radius = ADMM[".spectral_radius_xx"];
        return Rcpp::as<double>(spectral_radius(mat));
    }

    // calculating the spectral radius of X, if X is positive definite
    // in this case it is the largest eigenvalue of X
    static double spectral_radius_x(const MatrixXd &X)
    {
        Rcpp::NumericMatrix mat = Rcpp::wrap(X);

        Rcpp::Environment ADMM = Rcpp::Environment::namespace_env("ADMM");
        Rcpp::Function spectral_radius = ADMM[".spectral_radius_x"];
        return Rcpp::as<double>(spectral_radius(mat));
    }

public:
    ADMMDantzig(const MatrixXd &datX_, const VectorXd &datY_,
                double eps_abs_ = 1e-6,
                double eps_rel_ = 1e-6) :
        ADMMBase(datX_.cols(), datX_.cols(), datX_.cols(),
                 eps_abs_, eps_rel_),
        datX(&datX_), datY(&datY_),
        use_XX(datX_.rows() > datX_.cols() && datX_.cols() <= 1000),
        XY(datX_.transpose() * datY_),
        XY_norm(XY.norm()),
        lambda0(XY.array().abs().maxCoeff()),
        cache_Ax(dim_dual)
    {
        if(use_XX)
        {
            XX.noalias() = datX_.transpose() * datX_;
            sprad = spectral_radius_x(XX);
        } else {
            sprad = spectral_radius_xx(datX_);
        }

        sprad *= sprad;
    }

    virtual double get_lambda_zero() { return lambda0; }

    // init() is a cold start for the first lambda
    virtual void init(double lambda_, double rho_ratio_)
    {
        main_x.setZero();
        cache_Ax.setZero();
        aux_z.setZero();
        dual_y.setZero();
        lambda = lambda_;
        rho = 1.0 / (rho_ratio_ * sprad);
        eps_primal = 0.0;
        eps_dual = 0.0;
        resid_primal = 9999;
        resid_dual = 9999;
        iter_counter = 0;

        rho_changed_action();
    }
    // when computing for the next lambda, we can use the
    // current main_x, aux_z, dual_y and rho as initial values
    virtual void init_warm(double lambda_)
    {
        lambda = lambda_;
        eps_primal = 0.0;
        eps_dual = 0.0;
        resid_primal = 9999;
        resid_dual = 9999;
        iter_counter = 0;
    }
};



#endif // ADMMDANTZIG_H
