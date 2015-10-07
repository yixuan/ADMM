#ifndef ADMMBP_H
#define ADMMBP_H

#include "FADMMBase.h"
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
class ADMMBP: public FADMMBase< Eigen::VectorXd, Eigen::SparseVector<double>, Eigen::VectorXd >
{
private:
    typedef Eigen::MatrixXd MatrixXd;
    typedef Eigen::VectorXd VectorXd;
    typedef Eigen::ArrayXd ArrayXd;
    typedef Eigen::Map<const MatrixXd> MapMat;
    typedef Eigen::Map<const VectorXd> MapVec;
    typedef Eigen::SparseVector<double> SparseVector;
    typedef Eigen::LLT<MatrixXd> LLT;

    const MapMat datX;            // pointer to A
    LLT solver;                   // Cholesky factorization of AA'
    VectorXd cache_AAAb;          // cache A'(AA')^(-1)b
    MatrixXd cache_LinvA;         // cache L^(-1)A, AA'=LL'
    VectorXd workspace;



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
        // vec += adj_z;
        // res.noalias() = vec + cache_AAAb;

        for(SparseVector::InnerIterator iter(adj_z); iter; ++iter)
            vec[iter.index()] += iter.value();
        std::transform(vec.data(), vec.data() + dim_dual, cache_AAAb.data(), res.data(), std::plus<double>());

// Version 1
        // VectorXd tmp = datX * vec;
        // vec.noalias() = datX.transpose() * solver.solve(tmp);
        // res -= vec;
// Version 2
        // res.noalias() -= cache_LinvA.transpose() * (cache_LinvA * vec);
// Version 3
        Linalg::mat_vec_prod(workspace, cache_LinvA, vec);
        Linalg::mat_vec_tprod(res, cache_LinvA, workspace, -1.0, 1.0);
    }

    static void soft_threshold(SparseVector &res, VectorXd &vec, const double &penalty)
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



    // Calculate ||v1 - v2||^2 when v1 and v2 are sparse
    static double diff_squared_norm(const SparseVector &v1, const SparseVector &v2)
    {
        const int n1 = v1.nonZeros(), n2 = v2.nonZeros();
        const double *v1_val = v1.valuePtr(), *v2_val = v2.valuePtr();
        const int *v1_ind = v1.innerIndexPtr(), *v2_ind = v2.innerIndexPtr();

        double r = 0.0;
        int i1 = 0, i2 = 0;
        while(i1 < n1 && i2 < n2)
        {
            if(v1_ind[i1] == v2_ind[i2])
            {
                double val = v1_val[i1] - v2_val[i2];
                r += val * val;
                i1++;
                i2++;
            } else if(v1_ind[i1] < v2_ind[i2]) {
                r += v1_val[i1] * v1_val[i1];
                i1++;
            } else {
                r += v2_val[i2] * v2_val[i2];
                i2++;
            }
        }
        while(i1 < n1)
        {
            r += v1_val[i1] * v1_val[i1];
            i1++;
        }
        while(i2 < n2)
        {
            r += v2_val[i2] * v2_val[i2];
            i2++;
        }

        return r;
    }

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
    double compute_resid_dual()
    {
        return rho * std::sqrt(diff_squared_norm(aux_z, old_z));
    }
    double compute_resid_combined()
    {
        return rho * resid_primal * resid_primal + rho * diff_squared_norm(aux_z, adj_z);
    }

public:
    ADMMBP(const MapMat &datX_, const MapVec &datY_,
           double rho_ = 1.0,
           double eps_abs_ = 1e-6,
           double eps_rel_ = 1e-6) :
        FADMMBase(datX_.cols(), datX_.cols(), datX_.cols(),
                  eps_abs_, eps_rel_),
        datX(datX_.data(), datX_.rows(), datX_.cols()),
        workspace(datX_.cols())
    {
        MatrixXd XX;
        Linalg::tcross_prod_lower(XX, datX_);
        solver.compute(XX.selfadjointView<Eigen::Lower>());
        cache_AAAb = datX_.transpose() * solver.solve(datY_);

        // Calculating T = inv(L) * X, solving LT=X
        const MatrixXd &L = solver.matrixL();
        const int nrow = datX_.rows();
        const int ncol = datX_.cols();
        const int nelem = nrow * ncol;
        cache_LinvA.resize(nrow, ncol);
        std::copy(datX_.data(), datX_.data() + nelem, cache_LinvA.data());

        const double alpha = 1.0;
        Linalg::dtrsm_("L", "L", "N", "N", &nrow, &ncol,
                       &alpha, L.data(), &nrow, cache_LinvA.data(), &nrow);

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
