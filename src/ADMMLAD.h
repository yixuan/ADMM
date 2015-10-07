#ifndef ADMMLAD_H
#define ADMMLAD_H

#include "FADMMBase.h"
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
class ADMMLAD: public FADMMBase< Eigen::VectorXd, Eigen::SparseVector<double>, Eigen::VectorXd >
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
    MatrixXd H;                   // cache hat matrix X * inv(X'X) * X'
                                  // if nrow(X) <= 2000
    const double ynorm;           // L2 norm of datY

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

        if(dim_dual <= 2000)
        {
            const double alpha = 1.0;
            const double beta = 0.0;
            const int one = 1;
            Linalg::dsymv_("L", &dim_dual, &alpha, H.data(), &dim_dual,
                           vec.data(), &one, &beta, res.data(), &one);
        } else {
            VectorXd tmp = (*datX).transpose() * vec;
            res.noalias() = (*datX) * solver.solve(tmp);
        }
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
        VectorXd vec = main_x - (*datY) + adj_y / rho;
        soft_threshold(res, vec, 1.0 / rho);
    }
    void next_residual(VectorXd &res)
    {
        // res.noalias() = main_x - (*datY);
        // res -= aux_z;

        std::transform(main_x.data(), main_x.data() + dim_dual, datY->data(), res.data(), std::minus<double>());
        for(SparseVector::InnerIterator iter(aux_z); iter; ++iter)
            res[iter.index()] -= iter.value();
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
        r = std::max(r, ynorm);
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
    ADMMLAD(const MatrixXd &datX_, const VectorXd &datY_,
            double rho_ = 1.0,
            double eps_abs_ = 1e-6,
            double eps_rel_ = 1e-6) :
        FADMMBase(datX_.rows(), datX_.rows(), datX_.rows(),
                  eps_abs_, eps_rel_),
        datX(&datX_), datY(&datY_),
        ynorm(datY_.norm())
    {
        const int nrow = datX_.rows();
        const int ncol = datX_.cols();
        const int nelem = nrow * ncol;
        // Calculating X'X
        MatrixXd XX;
        Linalg::cross_prod_lower(XX, datX_);
        // Cholesky decomposition X'X = LL'
        solver.compute(XX.selfadjointView<Eigen::Lower>());

        if(nrow <= 2000)
        {
            const MatrixXd &L = solver.matrixLLT();
            // Calculating T = X * inv(L'), solving TL'=X
            double *T = new double[nelem];
            std::copy(datX_.data(), datX_.data() + nelem, T);
            const double alpha = 1.0;
            Linalg::dtrsm_("R", "L", "T", "N", &nrow, &ncol,
                           &alpha, L.data(), &ncol, T, &nrow);
            // H = X * inv(X'X) * X' = TT'
            Linalg::tcross_prod_lower(H, MapMat(T, nrow, ncol));
            delete [] T;
        }

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
