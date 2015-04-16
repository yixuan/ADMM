#define EIGEN_DONT_PARALLELIZE

#ifndef ADMMBP_H
#define ADMMBP_H

#include "ADMMBase.h"
#include "Linalg/BlasWrapper.h"

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
    typedef Eigen::Map<const MatrixXd> MapMat;
    typedef Eigen::Map<const VectorXd> MapVec;
    typedef Eigen::SparseVector<double> SparseVector;
    typedef Eigen::LLT<MatrixXd> LLT;

    const MapMat *datX;           // pointer to A
    LLT solver;                   // Cholesky factorization of AA'
    VectorXd cache_AAAb;          // cache A'(AA')^(-1)b



    // x -> Ax
    void A_mult (VectorXd &res, VectorXd &x)  { res.swap(x); }
    // y -> A'y
    void At_mult(VectorXd &res, VectorXd &y)  { res.swap(y); }
    // z -> Bz
    void B_mult (VectorXd &res, SparseVector &z) { res = -z; }
    // ||c||_2
    double c_norm() { return 0.0; }



    void next_x(VectorXd &res)
    {
        VectorXd vec = -adj_y / rho;
        vec += adj_z;
        res.noalias() = vec + cache_AAAb;

        VectorXd tmp = (*datX) * vec;
        vec.noalias() = (*datX).transpose() * solver.solve(tmp);

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
    void next_z(SparseVector &res)
    {
        VectorXd vec = main_x + adj_y / rho;
        soft_threshold(res, vec, 1.0 / rho);
    }
    void next_residual(VectorXd &res)
    {
        res = main_x;
        res -= aux_z;
    }
    void rho_changed_action() {}



    // Faster computation of epsilons and residuals
    double compute_eps_primal()
    {
        double r = std::max(main_x.norm(), aux_z.norm());
        return r * eps_rel + std::sqrt(double(dim_dual)) * eps_abs;
    }
    double compute_eps_dual()
    {
        return dual_y.norm() * eps_rel + std::sqrt(double(dim_main)) * eps_abs;
    }
    double compute_resid_dual(SparseVector &zdiff)
    {
        return rho * zdiff.norm();
    }
    double compute_resid_combined()
    {
        SparseVector tmp = aux_z - adj_z;
        return rho * resid_primal * resid_primal + rho * tmp.squaredNorm();
    }

public:
    ADMMBP(const MapMat &datX_, const MapVec &datY_,
           double rho_ = 1.0,
           double eps_abs_ = 1e-6,
           double eps_rel_ = 1e-6) :
        ADMMBase(datX_.cols(), datX_.cols(), datX_.cols(),
                 eps_abs_, eps_rel_),
        datX(&datX_)
    {
        MatrixXd AA;
        Linalg::tcross_prod_lower(AA, datX_);
        solver.compute(AA.selfadjointView<Eigen::Lower>());
        cache_AAAb = datX_.transpose() * solver.solve(datY_);

        main_x.setZero();
        aux_z.setZero();
        dual_y.setZero();

        adj_z.setZero();
        adj_y.setZero();

        rho = rho_;

        eps_primal = 0.0;
        eps_dual = 0.0;
        resid_primal = 9999;
        resid_dual = 9999;
    }
};



#endif // ADMMLAD_H
