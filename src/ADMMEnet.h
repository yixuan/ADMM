#ifndef ADMMENET_H
#define ADMMENET_H

#include "ADMMLassoTall.h"
#include "ADMMLassoWide.h"

// minimize  1/2 * ||y - X * beta||^2 + lambda * enet(beta)
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
// g(z) => lambda * enet(z)
class ADMMEnetTall: public ADMMLassoTall
{
private:
    Scalar alpha;

    void enet(SparseVector &res, const Vector &vec, const double &penalty)
    {
        int v_size = vec.size();
        res.setZero();
        res.reserve(v_size);

        const Scalar thresh = alpha * penalty;
        const Scalar denom = 1.0 + penalty * (1.0 - alpha);
        const Scalar *ptr = vec.data();
        for(int i = 0; i < v_size; i++)
        {
            if(ptr[i] > thresh)
                res.insertBack(i) = (ptr[i] - thresh) / denom;
            else if(ptr[i] < -thresh)
                res.insertBack(i) = (ptr[i] + thresh) / denom;
        }
    }
    void next_z(SparseVector &res)
    {
        Vector vec = main_x + adj_y / rho;
        enet(res, vec, lambda / rho);
    }

public:
    ADMMEnetTall(ConstGenericMatrix &datX_,
                 ConstGenericVector &datY_,
                 double alpha_ = 1.0,
                 double eps_abs_ = 1e-6,
                 double eps_rel_ = 1e-6) :
        ADMMLassoTall(datX_, datY_, eps_abs_, eps_rel_),
        alpha(alpha_)
    {
        this->lambda0 /= (alpha + 0.0001);
    }
};



class ADMMEnetWide: public ADMMLassoWide
{
private:
    Scalar alpha;

    void enet(SparseVector &res, const Vector &vec, const double &penalty)
    {
        int v_size = vec.size();
        res.setZero();
        res.reserve(v_size);

        const Scalar thresh = alpha * penalty;
        const Scalar denom = 1.0 + penalty * (1.0 - alpha);
        const Scalar *ptr = vec.data();
        for(int i = 0; i < v_size; i++)
        {
            if(ptr[i] > thresh)
                res.insertBack(i) = (ptr[i] - thresh) / denom;
            else if(ptr[i] < -thresh)
                res.insertBack(i) = (ptr[i] + thresh) / denom;
        }
    }

    void active_set_update(SparseVector &res)
    {
        const Scalar gamma = sprad;
        const Scalar penalty = lambda / (rho * gamma);
        const Scalar thresh = alpha * penalty;
        const Scalar denom = 1.0 + penalty * (1.0 - alpha);
        tmp.noalias() = (cache_Ax + aux_z + dual_y / Scalar(rho)) / gamma;
        res = main_x;

        Scalar *val_ptr = res.valuePtr();
        const int *ind_ptr = res.innerIndexPtr();
        const int nnz = res.nonZeros();

#ifdef __AVX__
        vtrX.read_vec(tmp.data());
#endif
        #pragma omp parallel for
        for(int i = 0; i < nnz; i++)
        {
#ifdef __AVX__
            const Scalar val = val_ptr[i] - vtrX.ith_inner_product(ind_ptr[i]);
#else
            const Scalar val = val_ptr[i] - tmp.dot(datX.col(ind_ptr[i]));
#endif

            if(val > thresh)
                val_ptr[i] = (val - thresh) / denom;
            else if(val < -thresh)
                val_ptr[i] = (val + thresh) / denom;
            else
                val_ptr[i] = 0.0;
        }

        res.prune(0.0);
    }

    void next_x(SparseVector &res)
    {
        // iter_counter = 0, 3, 15, 63, .... (4^k - 1)
        if(is_regular_update(iter_counter) && lambda < lambda0)
        {
            const Scalar gamma = sprad;
            tmp.noalias() = cache_Ax + aux_z + dual_y / Scalar(rho);
            Vector vec(dim_main);
#ifdef __AVX__
            vtrX.trans_mult_vec(tmp, vec.data());
            vec *= (-1.0 / gamma);
#else
            vec.noalias() = -datX.transpose() * tmp / gamma;
#endif
            vec += main_x;
            enet(res, vec, lambda / (rho * gamma));
        } else {
            active_set_update(res);
        }
        iter_counter++;
    }

public:
    ADMMEnetWide(ADMMLassoWide::ConstGenericMatrix &datX_,
                 ADMMLassoWide::ConstGenericVector &datY_,
                 double alpha_ = 1.0,
                 double eps_abs_ = 1e-6,
                 double eps_rel_ = 1e-6) :
        ADMMLassoWide(datX_, datY_, eps_abs_, eps_rel_),
        alpha(alpha_)
    {
        this->lambda0 /= (alpha + 0.0001);
    }
};

#endif // ADMMENET_H
