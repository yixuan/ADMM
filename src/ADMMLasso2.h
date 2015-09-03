#ifndef ADMMLASSO_H
#define ADMMLASSO_H

#include "ADMMBase.h"
#include "Linalg/BlasWrapper.h"
#include "Eigs/SymEigsSolver.h"
#include "Eigs/MatOpDense.h"

#ifdef __AVX__
#include "AVX.h"
#endif

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
    typedef const Eigen::Ref<const MatrixXd> ConstGenericMatrix;
    typedef const Eigen::Ref<const VectorXd> ConstGenericVector;
    typedef Eigen::SparseVector<double> SparseVector;

    const MapMat datX;            // pointer to data matrix
    const MapVec datY;            // pointer response vector
    double sprad;                 // spectral radius of X'X
    double lambda;                // L1 penalty
    double lambda0;               // minimum lambda to make coefficients all zero

    int iter_counter;             // which iteration are we in?

    VectorXd cache_Ax;            // cache Ax
    VectorXd tmp;
#ifdef __AVX__
    __m256d *loaded_X;
    int nrowx;
    int ncolx;
#endif

    // x -> Ax
    void A_mult(VectorXd &res, SparseVector &x)
    {
        res.noalias() = datX * x;
    }
    // y -> A'y
    void At_mult(VectorXd &res, VectorXd &y)
    {
        res.noalias() = datX.transpose() * y;
    }
    // z -> Bz
    void B_mult(VectorXd &res, VectorXd &z)
    {
        res.swap(z);
    }
    // ||c||_2
    double c_norm() { return 0.0; }

    static void soft_threshold(SparseVector &res, const VectorXd &vec, const double &penalty)
    {
        int v_size = vec.size();
        res.setZero();
        res.reserve(v_size);

        const double *ptr = vec.data();
        for(int i = 0; i < v_size; i++)
        {
            if(ptr[i] > penalty)
                res.insertBack(i) = ptr[i] - penalty;
            else if(ptr[i] < -penalty)
                res.insertBack(i) = ptr[i] + penalty;
        }
    }

    void active_set_update(SparseVector &res)
    {
        const double gamma = sprad;
        double penalty = lambda / (rho * gamma);
        tmp.noalias() = (cache_Ax + aux_z + dual_y / rho) / gamma;
        res = main_x;

        double *val_ptr = res.valuePtr();
        const int *ind_ptr = res.innerIndexPtr();
        const int nnz = res.nonZeros();

#ifdef __AVX__
        int lenx;
        __m256d *loaded_tmp = load_vec_avx(tmp.data(), dim_dual, lenx);
#endif
        #pragma omp parallel for
        for(int i = 0; i < nnz; i++)
        {
#ifdef __AVX__
            const double val = val_ptr[i] - loaded_inner_product_avx(loaded_tmp, loaded_X + ind_ptr[i] * nrowx, lenx);
#else
            const double val = val_ptr[i] - tmp.dot(datX.col(ind_ptr[i]));
#endif

            if(val > penalty)
                val_ptr[i] = val - penalty;
            else if(val < -penalty)
                val_ptr[i] = val + penalty;
            else
                val_ptr[i] = 0.0;
        }

#ifdef __AVX__
        free(loaded_tmp);
#endif

        res.prune(0.0);
    }

    void next_x(SparseVector &res)
    {
        if(iter_counter % 10 == 0 && lambda < lambda0)
        {
            const double gamma = sprad;
            tmp.noalias() = cache_Ax + aux_z + dual_y / rho;
            VectorXd vec(dim_main);
#ifdef __AVX__
            mat_vec_tprod_avx(vec, datX, tmp);
            vec *= (-1.0 / gamma);
#else
            vec.noalias() = -datX.transpose() * tmp / gamma;
#endif
            vec += main_x;
            soft_threshold(res, vec, lambda / (rho * gamma));
        } else {
            active_set_update(res);
        }
        iter_counter++;
    }
    virtual void next_z(VectorXd &res)
    {
#ifdef __AVX__
        loaded_mat_spvec_prod_avx(cache_Ax.data(), dim_dual, loaded_X, nrowx, ncolx, main_x);
#else
        cache_Ax.noalias() = datX * main_x;
#endif

        res.noalias() = (datY + dual_y + rho * cache_Ax) / (-1 - rho);
    }
    void next_residual(VectorXd &res)
    {
        // res.noalias() = cache_Ax + aux_z;
        std::transform(cache_Ax.data(), cache_Ax.data() + dim_dual, aux_z.data(), res.data(), std::plus<double>());
    }
    void rho_changed_action() {}

    // Faster computation of epsilons and residuals
    double compute_eps_primal()
    {
        double r = std::max(cache_Ax.norm(), aux_z.norm());
        return r * eps_rel + std::sqrt(double(dim_dual)) * eps_abs;
    }
    double compute_eps_dual()
    {
        return std::sqrt(sprad) * dual_y.norm() * eps_rel + std::sqrt(double(dim_main)) * eps_abs;
    }
    double compute_resid_dual(const VectorXd &new_z)
    {
        return rho * std::sqrt(sprad) * (new_z - aux_z).norm();
    }

public:
    ADMMLasso(ConstGenericMatrix &datX_, ConstGenericVector &datY_,
              double eps_abs_ = 1e-6,
              double eps_rel_ = 1e-6) :
        ADMMBase(datX_.cols(), datX_.rows(), datX_.rows(),
                 eps_abs_, eps_rel_),
        datX(datX_.data(), datX_.rows(), datX_.cols()),
        datY(datY_.data(), datY_.size()),
        cache_Ax(dim_dual), tmp(dim_dual)
    {
        lambda0 = (datX.transpose() * datY).array().abs().maxCoeff();

        MatrixXd XX;
        Linalg::tcross_prod_lower(XX, datX);
        MatOpSymLower<double> op(XX);
        SymEigsSolver<double, LARGEST_ALGE> eigs(&op, 1, 3);
        srand(0);
        eigs.init();
        eigs.compute(10, 0.1);
        VectorXd evals = eigs.ritzvalues();
        sprad = evals[0];

#ifdef __AVX__
        ncolx = datX.cols();
        loaded_X = load_mat_avx(datX.data(), datX.rows(), datX.cols(), nrowx);
#endif
    }

    virtual ~ADMMLasso()
    {
#ifdef __AVX__
        free(loaded_X);
#endif
    }

    double get_lambda_zero() { return lambda0; }

    // init() is a cold start for the first lambda
    void init(double lambda_, double rho_)
    {
        main_x.setZero();
        cache_Ax.setZero();
        aux_z.setZero();
        dual_y.setZero();

        lambda = lambda_;
        rho = std::pow(lambda / sprad, 1.0 / 3);

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
