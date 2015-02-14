#ifndef ADMMENET_H
#define ADMMENET_H

#include "ADMMLasso.h"

// minimize  1/2 * ||y - X * beta||^2 + lambda * enet(beta)
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
// f(x) => lambda * enet(x)
// g(z) => 1/2 * ||z + b||^2
class ADMMEnet: public ADMMLasso
{
private:
    typedef Eigen::MatrixXd MatrixXd;
    typedef Eigen::VectorXd VectorXd;
    typedef Eigen::ArrayXd ArrayXd;
    typedef Eigen::SparseVector<double> SparseVector;

    double alpha;

    virtual void enet(SparseVector &res, VectorXd &vec, const double &penalty)
    {
        res.setZero();
        res.reserve(vec.size() / 2);

        double thresh = alpha * penalty;
        double denom = 1.0 + penalty * (1.0 - alpha);
        double *ptr = vec.data();
        for(int i = 0; i < vec.size(); i++)
        {
            if(ptr[i] > thresh)
                res.insertBack(i) = (ptr[i] - thresh) / denom;
            else if(ptr[i] < -thresh)
                res.insertBack(i) = (ptr[i] + thresh) / denom;
        }
    }

    virtual void active_set_update(SparseVector &res)
    {
        double gamma = 2 * rho + sprad;
        double penalty = lambda / (rho * gamma);
        double thresh = alpha * penalty;
        double denom = 1.0 + penalty * (1.0 - alpha);
        VectorXd vec = (cache_Ax + aux_z + dual_y / rho) / gamma;
        res = main_x;

        for(SparseVector::InnerIterator iter(res); iter; ++iter)
        {
            double val = iter.value() - vec.dot(datX.col(iter.index()));

            if(val > thresh)
                iter.valueRef() = (val - thresh) / denom;
            else if(val < -thresh)
                iter.valueRef() = (val + thresh) / denom;
            else
                iter.valueRef() = 0.0;
        }

        res.prune(0.0);
    }

    virtual void next_x(SparseVector &res)
    {
        if(iter_counter % 10 == 0 && lambda < lambda0)
        {
            double gamma = 2 * rho + sprad;
            VectorXd vec = cache_Ax + aux_z + dual_y / rho;
            vec = -datX.transpose() * vec / gamma;
            vec += main_x;
            enet(res, vec, lambda / (rho * gamma));
        } else {
            active_set_update(res);
        }
        iter_counter++;
    }

public:
    ADMMEnet(const MatrixXd &datX_, const VectorXd &datY_,
             double alpha_ = 1,
             double eps_abs_ = 1e-6,
             double eps_rel_ = 1e-6) :
        ADMMLasso(datX_, datY_, eps_abs_, eps_rel_),
        alpha(alpha_)
    {
        lambda0 /= alpha;
    }
};



#endif // ADMMENET_H
