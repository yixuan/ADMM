#ifndef ADMMLAD_H
#define ADMMLAD_H

#include "ADMMBase.h"
#include "Linalg/BlasWrapper.h"

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
//
// We define xx := Ax to be the new x variable, which will simplify the problem
// In ADMM form,
//   minimize f(xx) + g(z)
//   s.t. xx - z = c
//
// xx => X * beta, xx belongs to Range(X)
// z  => X * beta - y
// c  => y
// f(x) => 0
// g(z) => ||z||_1
class ADMMLAD: public ADMMBase< Eigen::VectorXd, Eigen::SparseVector<double> >
{
private:
    typedef Eigen::MatrixXd MatrixXd;
    typedef Eigen::VectorXd VectorXd;
    typedef Eigen::ArrayXd ArrayXd;
    typedef Eigen::Map<const MatrixXd> MapMat;
    typedef Eigen::Map<const VectorXd> MapVec;
    typedef Eigen::SparseVector<double> SparseVector;
    typedef Eigen::LLT<MatrixXd> LLT;

    const MatrixXd *datX;         // pointer to data matrix
    const VectorXd *datY;         // pointer response vector
    double ynorm;                 // L2 norm of datY

    LLT solver;                   // Cholesky factorization of A'A



    // x -> Ax
    void A_mult (VectorXd &res, VectorXd &x)  { res.swap(x); }
    // y -> A'y
    void At_mult(VectorXd &res, VectorXd &y)  { res.swap(y); }
    // z -> Bz
    void B_mult (VectorXd &res, SparseVector &z) { res = -z; }
    // ||c||_2
    double c_norm() { return ynorm; }



    void next_x(VectorXd &res)
    {
        VectorXd vec = (*datY) - adj_y / rho;
        vec += adj_z;

        VectorXd tmp = (*datX).transpose() * vec;
        res.noalias() = (*datX) * solver.solve(tmp);
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
        VectorXd vec = main_x - (*datY) + adj_y / rho;
        soft_threshold(res, vec, 1.0 / rho);
    }
    void next_residual(VectorXd &res)
    {
        res.noalias() = main_x - (*datY);
        res -= aux_z;
    }
    void rho_changed_action() {}



    // Faster computation of epsilons and residuals
    double compute_eps_primal()
    {
        double r = std::max(main_x.norm(), aux_z.norm());
        r = std::max(r, ynorm);
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
    ADMMLAD(const MatrixXd &datX_, const VectorXd &datY_,
            double rho_ = 1.0,
            double eps_abs_ = 1e-6,
            double eps_rel_ = 1e-6) :
        ADMMBase(datX_.cols(), datX_.rows(), datX_.rows(),
                 eps_abs_, eps_rel_),
        datX(&datX_), datY(&datY_),
        ynorm(datY_.norm())
    {
        const MapMat mapX(datX_.data(), datX_.rows(), datX_.cols());
        MatrixXd AA;
        Linalg::cross_prod_lower(AA, mapX);
        solver.compute(AA.triangularView<Eigen::Lower>());

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

    VectorXd get_x()
    {
        VectorXd vec = (*datY) - adj_y / rho;
        vec += adj_z;
        return solver.solve((*datX).transpose() * vec);
    }
};



#endif // ADMMLAD_H
