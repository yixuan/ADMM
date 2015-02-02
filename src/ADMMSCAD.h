#ifndef ADMMSCAD_H
#define ADMMSCAD_H

#include "ADMMLasso.h"

// minimize  1/2 * ||y - X * beta||^2 + lambda * SCAD(beta)
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
// f(x) => lambda * SCAD(x)
// g(z) => 1/2 * ||z + b||^2
class ADMMSCAD: public ADMMLasso
{
private:
    typedef Eigen::MatrixXd MatrixXd;
    typedef Eigen::VectorXd VectorXd;
    typedef Eigen::ArrayXd ArrayXd;
    typedef Eigen::SparseVector<double> SparseVector;

    double pen_a;

    virtual void scad(SparseVector &res, VectorXd &vec, const double &penalty)
    {
        res.setZero();
        res.reserve(vec.size() / 2);

        for(int i = 0; i < vec.size(); i++)
        {
            double z = vec[i];
            if(z >= pen_a * penalty || z <= -pen_a * penalty)
                res.insertBack(i) = z;
            else if(z > penalty && z <= 2 * penalty)
                res.insertBack(i) = z - penalty;
            else if(z < -penalty && z >= -2 * penalty)
                res.insertBack(i) = z + penalty;
            else if(z > 2 * penalty && z < pen_a * penalty)
                res.insertBack(i) = ((pen_a - 1) * z - pen_a * penalty) / (pen_a - 2);
            else if(z < -2 * penalty && z > -pen_a * penalty)
                res.insertBack(i) = ((pen_a - 1) * z + pen_a * penalty) / (pen_a - 2);
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
            double z = iter.value() - vec.dot((*datX).col(iter.index()));

            if(z >= pen_a * penalty || z <= -pen_a * penalty)
                iter.valueRef() = z;
            else if(z > penalty && z <= 2 * penalty)
                iter.valueRef() = z - penalty;
            else if(z < -penalty && z >= -2 * penalty)
                iter.valueRef() = z + penalty;
            else if(z > 2 * penalty && z < pen_a * penalty)
                iter.valueRef() = ((pen_a - 1) * z - pen_a * penalty) / (pen_a - 2);
            else if(z < -2 * penalty && z > -pen_a * penalty)
                iter.valueRef() = ((pen_a - 1) * z + pen_a * penalty) / (pen_a - 2);
            else
                iter.valueRef() = 0.0;
        }

        res.prune(0.0);
    }
    
    virtual void next_x(SparseVector &res)
    {
        if(iter_counter % 10 == 0)
        {
            double gamma = 2 * rho + sprad;
            VectorXd vec = cache_Ax + aux_z + dual_y / rho;
            vec = -(*datX).transpose() * vec / gamma;
            vec += main_x;
            scad(res, vec, lambda / (rho * gamma));
        } else {
            active_set_update(res);
        }
        iter_counter++;        
    }
    
public:
    ADMMSCAD(const MatrixXd &datX_, const VectorXd &datY_,
             double pen_a_ = 3.7,
             double eps_abs_ = 1e-6,
             double eps_rel_ = 1e-6) :
        ADMMLasso(datX_, datY_, eps_abs_, eps_rel_),
        pen_a(pen_a_)
    {}
};



#endif // ADMMSCAD_H