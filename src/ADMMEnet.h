#ifndef ADMMENET_H
#define ADMMENET_H

#include "ADMMLassoTall.h"

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
    ADMMLassoTall::Scalar alpha;

    void enet(ADMMLassoTall::SparseVector &res, const ADMMLassoTall::Vector &vec, const double &penalty)
    {
        int v_size = vec.size();
        res.setZero();
        res.reserve(v_size);

        const ADMMLassoTall::Scalar thresh = alpha * penalty;
        const ADMMLassoTall::Scalar denom = 1.0 + penalty * (1.0 - alpha);
        const ADMMLassoTall::Scalar *ptr = vec.data();
        for(int i = 0; i < v_size; i++)
        {
            if(ptr[i] > penalty)
                res.insertBack(i) = (ptr[i] - thresh) / denom;
            else if(ptr[i] < -penalty)
                res.insertBack(i) = (ptr[i] + thresh) / denom;
        }
    }
    void next_z(ADMMLassoTall::SparseVector &res)
    {
        ADMMLassoTall::Vector vec = main_x + adj_y / rho;
        enet(res, vec, lambda / rho);
    }

public:
    ADMMLassoTall(ADMMLassoTall::ConstGenericMatrix &datX_,
                  ADMMLassoTall::ConstGenericVector &datY_,
                  double alpha_ = 1.0,
                  double eps_abs_ = 1e-6,
                  double eps_rel_ = 1e-6) :
        ADMMLassoTall(datX_, datY_, eps_abs_, eps_rel_),
        alpha(alpha_)
    {
        this->lambda0 /= (alpha + 0.0001);
    }
};



#endif // ADMMENET_H
