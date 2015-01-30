#ifndef ADMMLASSO_H
#define ADMMLASSO_H

#include "ADMMBase.h"

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
private:
    typedef Eigen::MatrixXd MatrixXd;
    typedef Eigen::VectorXd VectorXd;
    typedef Eigen::ArrayXd ArrayXd;
    typedef Eigen::SparseVector<double> SparseVector;

    const MatrixXd *datX;         // pointer to data matrix
    const VectorXd *datY;         // pointer response vector
    double sprad;                 // spectral radius of X'X
    double lambda;                // L1 penalty
    double lambda0;               // minimum lambda to make coefficients all zero

    int active_set_niter;         // counter for fast active set update
    SparseVector active_set_save; // save the active set in previous iteration

    VectorXd cache_Ax;            // cache Ax
    
    virtual void A_mult(VectorXd &res, SparseVector &x) // x -> Ax
    {
        res.noalias() = (*datX) * x;
    }
    virtual void At_mult(VectorXd &res, VectorXd &y) // y -> A'y
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
    virtual void B_mult (VectorXd &res, VectorXd &z) // z -> Bz
    {
        res.swap(z);
    }  
    virtual double c_norm() { return 0.0; } // ||c||_2
    virtual void next_residual(VectorXd &res)
    {
        res.noalias() = cache_Ax + aux_z;
    }

    // whether the new active set (indices of non-zero coefficients)
    // is contained in the previous one
    static bool active_set_smaller(SparseVector &oldset, SparseVector &newset)
    {
        int n_new = newset.nonZeros();
        int n_old = oldset.nonZeros();

        if(n_new > n_old)
            return false;

        if(n_new == 0)
            return true;

        std::vector<int> v(n_new, -1);
        std::set_difference(newset.innerIndexPtr(), newset.innerIndexPtr() + n_new,
                            oldset.innerIndexPtr(), oldset.innerIndexPtr() + n_old,
                            v.begin());

        return v[0] == -1;
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
        if(active_set_niter >= 5)
        {
            #if ADMM_PROFILE > 1
            clock_t t1, t2;
            t1 = clock();
            #endif

            active_set_update(res);

            #if ADMM_PROFILE > 1
            t2 = clock();
            Rcpp::Rcout << "active set update: " << double(t2 - t1) / CLOCKS_PER_SEC << " secs\n";
            #endif

            if(active_set_niter >= 50)
                active_set_niter = 0;
        }
        else
        {
            double gamma = 2 * rho + sprad;
            VectorXd vec = cache_Ax + aux_z + dual_y / rho;

            #if ADMM_PROFILE > 1
            clock_t t1, t2;
            t1 = clock();
            #endif

            vec = -(*datX).transpose() * vec / gamma;

            #if ADMM_PROFILE > 1
            t2 = clock();
            Rcpp::Rcout << "matrix product in x update: " << double(t2 - t1) / CLOCKS_PER_SEC << " secs\n";
            #endif

            vec += main_x;
            soft_threshold(res, vec, lambda / (rho * gamma));
        }

        #if ADMM_PROFILE > 1
        Rcpp::Rcout << "# non-zero coefs: " << res.nonZeros() << std::endl;
        #endif
        
        if(active_set_smaller(active_set_save, res))
            active_set_niter++;
        else
            active_set_niter = 0;

        active_set_save = res;
    }
    virtual void next_z(VectorXd &res)
    {
        cache_Ax = (*datX) * main_x;
        res.noalias() = ((*datY) + dual_y + rho * cache_Ax) / (-1 - rho);
    }
    virtual void rho_changed_action() {}
    // a faster version compared to the base implementation
    virtual double compute_eps_primal()
    {
        double r = std::max(cache_Ax.norm(), aux_z.norm());
        return r * eps_rel + sqrt(double(dim_dual)) * eps_abs;
    }

    // calculating the spectral radius of X'X
    // in this case it is the largest eigenvalue of X'X
    static double spectral_radius(const MatrixXd &X)
    {
        Rcpp::NumericMatrix mat = Rcpp::wrap(X);
    
        Rcpp::Environment ADMM = Rcpp::Environment::namespace_env("ADMM");
        Rcpp::Function spectral_radius = ADMM[".spectral_radius"];
    
        return Rcpp::as<double>(spectral_radius(mat));
    }
    
public:
    ADMMLasso(const MatrixXd &datX_, const VectorXd &datY_,
              double eps_abs_ = 1e-6,
              double eps_rel_ = 1e-6) :
        ADMMBase(datX_.cols(), datX_.rows(), datX_.rows(),
                 eps_abs_, eps_rel_),
        datX(&datX_), datY(&datY_),
        active_set_save(1),
        cache_Ax(dim_dual)
    {
        lambda0 = ((*datX).transpose() * (*datY)).array().abs().maxCoeff();
        sprad = spectral_radius(datX_);
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
        rho = lambda_ / (rho_ratio_ * sprad);
        eps_primal = 0.0;
        eps_dual = 0.0;
        resid_primal = 9999;
        resid_dual = 9999;
        active_set_niter = 0;
        active_set_save.setZero();

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
        active_set_niter = 0;
        active_set_save.setZero();
    }
};



#endif // ADMMLASSO_H