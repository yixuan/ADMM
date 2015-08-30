#ifndef ADMMLASSO_H
#define ADMMLASSO_H

#include "ADMMBase.h"
#include "Linalg/BlasWrapper.h"
#include "Eigs/SymEigsSolver.h"
#include "Eigs/MatOpDense.h"

#ifdef __AVX__
#include <immintrin.h>
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
    typedef Eigen::SparseVector<double> SparseVector;

    const MapMat datX;            // pointer to data matrix
    const MapVec datY;            // pointer response vector
    double sprad;                 // spectral radius of X'X
    double lambda;                // L1 penalty
    double lambda0;               // minimum lambda to make coefficients all zero

    int iter_counter;             // which iteration are we in?

    VectorXd cache_Ax;            // cache Ax

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

#ifdef __AVX__
    static double inner_product_avx(const double *x, const double *y, int len)
    {
        __m256d xx;
        __m256d yy;
        __m256d res = _mm256_setzero_pd();

        double r = 0.0;

        const char rem = (unsigned long)x % 32;
        const char head = rem ? (32 - rem) / sizeof(double) : 0;

        for(int i = 0; i < head; i++, x++, y++)
            r += (*x) * (*y);

        const int npack = (len - head) / 8;

        for(int i = 0; i < npack; i++, x += 8, y += 8)
        {
            xx = _mm256_load_pd(x);
            yy = _mm256_loadu_pd(y);
            res = _mm256_add_pd(res, _mm256_mul_pd(xx, yy));

            xx = _mm256_load_pd(x + 4);
            yy = _mm256_loadu_pd(y + 4);
            res = _mm256_add_pd(res, _mm256_mul_pd(xx, yy));
        }
        double *resp = (double*) &res;
        r += resp[0] + resp[1] + resp[2] + resp[3];

        for(int i = head + 8 * npack; i < len; i++, x++, y++)
            r += (*x) * (*y);

        return r;
    }
#endif

    void active_set_update(SparseVector &res)
    {
        const double gamma = sprad;
        double penalty = lambda / (rho * gamma);
        VectorXd vec = (cache_Ax + adj_z + adj_y / rho) / gamma;
        res = main_x;

        double *val_ptr = res.valuePtr();
        const int *ind_ptr = res.innerIndexPtr();
        const int nnz = res.nonZeros();

        #pragma omp parallel for
        for(int i = 0; i < nnz; i++)
        {
#ifdef __AVX__
            const double val = val_ptr[i] - inner_product_avx(vec.data(), datX.data() + ind_ptr[i] * dim_dual, dim_dual);
#else
            const double val = val_ptr[i] - vec.dot(datX.col(ind_ptr[i]));
#endif

            if(val > penalty)
                val_ptr[i] = val - penalty;
            else if(val < -penalty)
                val_ptr[i] = val + penalty;
            else
                val_ptr[i] = 0.0;
        }

        res.prune(0.0);
    }

    void next_x(SparseVector &res)
    {
        if(iter_counter % 10 == 0 && lambda < lambda0)
        {
            const double gamma = sprad;
            VectorXd vec = cache_Ax + adj_z + adj_y / rho;
            vec = -datX.transpose() * vec / gamma;
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

       double *Ax = cache_Ax.data();
       std::fill(Ax, Ax + dim_dual, double(0));

       const char rem = (unsigned long)Ax % 32;
       const char head = rem ? (32 - rem) / sizeof(double) : 0;
       const int npack = (dim_dual - head) / 8;
       __m256d mvec;
       __m256d vvec;
       __m256d cvec;

       const double *X0 = datX.data();
       const double *colptr;
       double *vptr;

       for(SparseVector::InnerIterator iter(main_x); iter; ++iter)
       {
           colptr = X0 + dim_dual * iter.index();
           vptr = Ax;

           const double val = iter.value();
           cvec = _mm256_set1_pd(val);

           for(int i = 0; i < head; i++, colptr++, vptr++)
               *vptr += *colptr * val;

           for(int i = 0; i < npack; i++, colptr += 8, vptr += 8)
           {
               mvec = _mm256_loadu_pd(colptr);
               mvec = _mm256_mul_pd(mvec, cvec);
               vvec = _mm256_load_pd(vptr);
               vvec = _mm256_add_pd(vvec, mvec);
               _mm256_store_pd(vptr, vvec);

               mvec = _mm256_loadu_pd(colptr + 4);
               mvec = _mm256_mul_pd(mvec, cvec);
               vvec = _mm256_load_pd(vptr + 4);
               vvec = _mm256_add_pd(vvec, mvec);
               _mm256_store_pd(vptr + 4, vvec);
           }
           for(int i = head + 8 * npack; i < dim_dual; i++, colptr++, vptr++)
               *vptr += *colptr * val;
        }

#else
        cache_Ax.noalias() = datX * main_x;
#endif

        res.noalias() = (datY + adj_y + rho * cache_Ax) / (-1 - rho);
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
    double compute_resid_dual()
    {
        return rho * std::sqrt(sprad) * (aux_z - old_z).norm();
    }
    double compute_resid_combined()
    {
        return rho * resid_primal * resid_primal + rho * (aux_z - adj_z).squaredNorm();
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

        MatrixXd XX;
        Linalg::tcross_prod_lower(XX, datX);
        MatOpSymLower<double> op(XX);
        SymEigsSolver<double, LARGEST_ALGE> eigs(&op, 1, 3);
        srand(0);
        eigs.init();
        eigs.compute(10, 0.1);
        VectorXd evals = eigs.ritzvalues();
        sprad = evals[0];
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

        MatrixXd XX;
        Linalg::tcross_prod_lower(XX, datX);
        MatOpSymLower<double> op(XX);
        SymEigsSolver<double, LARGEST_ALGE> eigs(&op, 1, 3);
        srand(0);
        eigs.init();
        eigs.compute(10, 0.1);
        VectorXd evals = eigs.ritzvalues();
        sprad = evals[0];
    }

    double get_lambda_zero() { return lambda0; }

    // init() is a cold start for the first lambda
    void init(double lambda_, double rho_)
    {
        main_x.setZero();
        cache_Ax.setZero();
        aux_z.setZero();
        dual_y.setZero();

        adj_z.setZero();
        adj_y.setZero();

        lambda = lambda_;
        rho = std::pow(lambda / sprad, 1.0 / 3);

        eps_primal = 0.0;
        eps_dual = 0.0;
        resid_primal = 9999;
        resid_dual = 9999;

        iter_counter = 0;

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

        iter_counter = 0;
    }
};



#endif // ADMMLASSO_H
