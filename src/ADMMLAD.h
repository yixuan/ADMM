#ifndef ADMMLAD_H
#define ADMMLAD_H

#include "ADMMBase.h"

// minimize  ||y - X * beta||_1
//
// In ADMM form,
//   minimize f(x) + g(z)
//   s.t. Ax - z = c
//
// x => beta
// z => X * beta - y
// A => X
// c => y
// f(x) => 0
// g(z) => ||z||_1
class ADMMLAD: public ADMMBase< Eigen::VectorXd, Eigen::SparseVector<double> >
{
private:
    typedef Eigen::MatrixXd MatrixXd;
    typedef Eigen::VectorXd VectorXd;
    typedef Eigen::ArrayXd ArrayXd;
    typedef Eigen::SparseVector<double> SparseVector;
    typedef Eigen::HouseholderQR<MatrixXd> QRdecomp;
    typedef Eigen::HouseholderSequence<MatrixXd, VectorXd> QRQ;

    const MatrixXd *datX;         // pointer to data matrix
    const VectorXd *datY;         // pointer response vector
    double ynorm;                 // L2 norm of datY

    QRdecomp decomp;              // QR decomposition of datX
    QRQ decomp_Q;                 // Q operator in the QR decomposition
    VectorXd cache_Ax;            // cache Ax
    
    virtual void A_mult(VectorXd &res, VectorXd &x) // x -> Ax
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
    virtual void B_mult (VectorXd &res, SparseVector &z) // z -> Bz
    {
        res = -z;
    }  
    virtual double c_norm() { return ynorm; } // ||c||_2
    virtual void next_residual(VectorXd &res)
    {
        res.noalias() = cache_Ax - (*datY);
        res -= aux_z;
    }
    
    virtual void next_x(VectorXd &res)
    {
        // We actually do not need to calculate x. Rather,
        // only Ax is needed to update z and y.
        // Inside this function we only update Ax, and x
        // is calculated in get_x().
        // We also need to override compute_eps_primal() to obtain
        // the correct value of eps_primal.

        // Ax = Q_1 * Q_1' * (datY + aux_z - dual_y / rho)
        cache_Ax.noalias() = (*datY) - dual_y / rho;
        cache_Ax += aux_z;
        cache_Ax.applyOnTheLeft(decomp_Q.transpose());
        cache_Ax.tail(dim_dual - dim_main).setZero();
        cache_Ax.applyOnTheLeft(decomp_Q);
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

    virtual void next_z(SparseVector &res)
    {
        VectorXd vec = cache_Ax - (*datY) + dual_y / rho;
        soft_threshold(res, vec, 1.0 / rho);
    }

    virtual void rho_changed_action() {}

    // a faster version compared to the base implementation
    virtual double compute_eps_primal()
    {
        double r = std::max(cache_Ax.norm(), aux_z.norm());
        r = std::max(r, ynorm);
        return r * eps_rel + sqrt(double(dim_dual)) * eps_abs;
    }

    // a faster version compared to the base implementation
    virtual double compute_eps_dual()
    {
        return dual_y.norm() * eps_rel + sqrt(double(dim_dual)) * eps_abs;
    }

public:
    ADMMLAD(const MatrixXd &datX_, const VectorXd &datY_,
            double rho_ratio_ = 0.1,
            double eps_abs_ = 1e-6,
            double eps_rel_ = 1e-6) :
        ADMMBase(datX_.cols(), datX_.rows(), datX_.rows(),
                 eps_abs_, eps_rel_),
        datX(&datX_), datY(&datY_),
        ynorm(datY_.norm()),
        decomp(datX_),
        decomp_Q(decomp.householderQ()),
        cache_Ax(dim_dual)
    {
        main_x.setZero();
        cache_Ax.setZero();
        aux_z.setZero();
        dual_y.setZero();
        rho = 1.0 / rho_ratio_;
        eps_primal = 0.0;
        eps_dual = 0.0;
        resid_primal = 9999;
        resid_dual = 9999;
    }
    
    virtual VectorXd get_x()
    {
        VectorXd vec = (*datY) - dual_y / rho;
        vec += aux_z;
        main_x.noalias() = decomp.solve(vec);
        
        return main_x;
    }
};



#endif // ADMMLAD_H