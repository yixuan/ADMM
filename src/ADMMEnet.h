#ifndef ADMMENET_H
#define ADMMENET_H

#include "ADMMLasso.h"

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
class ADMMEnet: public ADMMLasso
{
private:
    typedef Eigen::MatrixXd MatrixXd;
    typedef Eigen::VectorXd VectorXd;
    typedef Eigen::ArrayXd ArrayXd;
    typedef Eigen::SparseVector<double> SparseVector;

    double alpha;

    void enet(SparseVector &res, VectorXd &vec, const double &penalty)
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
    void next_z(SparseVector &res)
    {
        VectorXd vec = main_x + adj_y / rho;
        enet(res, vec, lambda / rho);
    }

public:
    ADMMEnet(const MatrixXd &datX_, const VectorXd &datY_,
             double alpha_ = 1.0,
             double eps_abs_ = 1e-6,
             double eps_rel_ = 1e-6) :
        ADMMLasso(datX_, datY_, eps_abs_, eps_rel_),
        alpha(alpha_)
    {
        lambda0 /= alpha;
    }
};



#endif // ADMMENET_H
