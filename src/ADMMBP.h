#ifndef ADMMBP_H
#define ADMMBP_H

#include "ADMMBase.h"

// Basis Pursuit
//
// minimize  ||x||_1
// s.t.      Ax = b
//
// In ADMM form,
//   minimize f(x) + g(z)
//   s.t. x - z = 0
//
// f(x) => indicator function of Ax = b
// g(z) => ||z||_1
class ADMMBP: public ADMMBase< Eigen::VectorXd, Eigen::SparseVector<double> >
{
private:
    typedef Eigen::MatrixXd MatrixXd;
    typedef Eigen::VectorXd VectorXd;
    typedef Eigen::ArrayXd ArrayXd;
    typedef Eigen::Map<MatrixXd> MapMat;
    typedef Eigen::Map<VectorXd> MapVec;
    typedef Eigen::SparseVector<double> SparseVector;
    typedef Eigen::HouseholderQR<MatrixXd> QRdecomp;
    typedef Eigen::HouseholderSequence<MatrixXd, VectorXd> QRQ;

    const MapMat *datX;         // pointer to A
    VectorXd cache_AAAb;          // cache A'(AA')^(-1)b

    QRdecomp decomp;              // QR decomposition of A'
    QRQ decomp_Q;                 // Q operator in the QR decomposition

    virtual void A_mult(VectorXd &res, VectorXd &x) // x -> Ax
    {
        res.swap(x);
    }
    virtual void At_mult(VectorXd &res, VectorXd &y) // y -> A'y
    {
        res.swap(y);
    }
    virtual void B_mult (VectorXd &res, SparseVector &z) // z -> Bz
    {
        res = -z;
    }
    virtual double c_norm() { return 0.0; } // ||c||_2
    virtual void next_residual(VectorXd &res)
    {
        res = main_x;
        res -= aux_z;
    }

    virtual void next_x(VectorXd &res)
    {
        VectorXd vec = -dual_y / rho;
        vec += aux_z;
        res.noalias() = vec + cache_AAAb;
        vec.applyOnTheLeft(decomp_Q.transpose());
        vec.tail(dim_main - (*datX).rows()).setZero();
        vec.applyOnTheLeft(decomp_Q);
        res -= vec;
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
        VectorXd vec = main_x + dual_y / rho;
        soft_threshold(res, vec, 1.0 / rho);
    }

    virtual void rho_changed_action() {}

    // a faster version compared to the base implementation
    virtual double compute_eps_primal()
    {
        double r = std::max(main_x.norm(), aux_z.norm());
        return r * eps_rel + sqrt(double(dim_dual)) * eps_abs;
    }

    // a faster version compared to the base implementation
    virtual double compute_eps_dual()
    {
        return dual_y.norm() * eps_rel + sqrt(double(dim_main)) * eps_abs;
    }

public:
    ADMMBP(const MapMat &datX_, const MapVec &datY_,
           double rho_ratio_ = 0.1,
           double eps_abs_ = 1e-6,
           double eps_rel_ = 1e-6) :
        ADMMBase(datX_.cols(), datX_.cols(), datX_.cols(),
                 eps_abs_, eps_rel_),
        datX(&datX_),
        cache_AAAb(dim_main),
        decomp(datX_.transpose()),
        decomp_Q(decomp.householderQ())
    {
        MatrixXd decomp_R(datX_.rows(), datX_.rows());
        decomp_R.triangularView<Eigen::Upper>() = decomp.matrixQR().triangularView<Eigen::Upper>();
        cache_AAAb.setZero();
        cache_AAAb.head(datX_.rows()) = decomp_R.triangularView<Eigen::Upper>().transpose().solve(datY_);
        cache_AAAb.applyOnTheLeft(decomp_Q);

        main_x.setZero();
        aux_z.setZero();
        dual_y.setZero();
        rho = 1.0 / rho_ratio_;
        eps_primal = 0.0;
        eps_dual = 0.0;
        resid_primal = 9999;
        resid_dual = 9999;
    }
};



#endif // ADMMLAD_H
