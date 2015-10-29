#ifndef ADMMDANTZIG_H
#define ADMMDANTZIG_H

#include "ADMMBase.h"
#include "ADMMLasso.h"
#include "Linalg/BlasWrapper.h"
#include "Eigs/SymEigsSolver.h"
#include "Eigs/MatOpDense.h"

// minimize ||beta||_1
// s.t. ||X'(X * beta - y)||_inf <= lambda
//
// In ADMM form,
//   minimize f(x) + g(z)
//   s.t. Ax + z = c
//
// x => beta
// z => -X'(X * beta - y)
// A => X'X
// c => X'y
// f(x) => ||x||_1
// g(z) => 0, dom(z) = {z: ||z||_inf <= lambda}
class ADMMDantzig: public ADMMBase<Eigen::SparseVector<double>, Eigen::VectorXd>
{
protected:
    typedef Eigen::MatrixXd MatrixXd;
    typedef Eigen::VectorXd VectorXd;
    typedef Eigen::ArrayXd ArrayXd;
    typedef Eigen::Map<const MatrixXd> MapMat;
    typedef Eigen::Map<const VectorXd> MapVec;
    typedef Eigen::SparseVector<double> SparseVector;

    const MatrixXd *datX;         // pointer to data matrix
    const VectorXd *datY;         // pointer response vector
    bool use_XX;                  // whether to cache X'X
    MatrixXd XX;                  // X'X
    VectorXd XY;                  // X'y
    double XY_norm;               // L2 norm of X'y

    double sprad;                 // spectral radius of X'XX'X
    double lambda;                // penalty parameter
    double lambda0;               // minimum lambda to make coefficients all zero

    int iter_counter;             // which iteration are we in?

    VectorXd cache_Ax;            // cache Ax



    void A_mult(VectorXd &res, SparseVector &x) // x -> Ax
    {
        if(use_XX)
        {
            res.noalias() = XX * x;
        } else {
            VectorXd tmp = (*datX) * x;
            res.noalias() = (*datX).transpose() * tmp;
        }
    }
    void At_mult(VectorXd &res, VectorXd &y) // y -> A'y
    {
        // The correct operation should be res = A'y.
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
    void B_mult(VectorXd &res, VectorXd &z) // z -> Bz
    {
        res.swap(z);
    }
    double c_norm() { return XY_norm; } // ||c||_2



    /*void active_set_update(SparseVector &res)
    {
        double penalty = 1.0 / (rho * sprad);
        VectorXd vec = (*datX) * (cache_Ax + aux_z + dual_y / rho - XY) / sprad;
        res = main_x;

        for(SparseVector::InnerIterator iter(res); iter; ++iter)
        {
            double val = iter.value() - vec.dot((*datX).col(iter.index()));

            if(val > penalty)
                iter.valueRef() = val - penalty;
            else if(val < -penalty)
                iter.valueRef() = val + penalty;
            else
                iter.valueRef() = 0.0;
        }

        res.prune(0.0);
    }*/
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
    void next_x(SparseVector &res)
    {
        if(lambda > lambda0 - 1e-5)
        {
            res.setZero();
            return;
        }

        VectorXd y = XY - adj_z - adj_y / rho;
        ADMMLasso solver(XX, y, 1e-5, 1e-5);
        solver.init(1.0 / rho, -1.0);
        solver.solve(100);
        res = solver.get_z();

        /*VectorXd rhs = (cache_Ax + adj_z + adj_y / rho - XY) / (-sprad);
        VectorXd vec;
        if(use_XX)
        {
            vec.noalias() = XX * rhs;
        } else {
            VectorXd tmp = (*datX) * rhs;
            vec.noalias() = (*datX).transpose() * tmp;
        }
        vec += main_x;
        soft_threshold(res, vec, 1.0 / (rho * sprad));*/
    }
    void next_z(VectorXd &res)
    {
        if(use_XX)
        {
            cache_Ax.noalias() = XX * main_x;
        } else {
            VectorXd tmp = (*datX) * main_x;
            cache_Ax.noalias() = (*datX).transpose() * tmp;
        }

        VectorXd z = cache_Ax + adj_y / rho - XY;
        for(int i = 0; i < res.size(); i++)
        {
            if(z[i] > 0)
                res[i] = -std::min(z[i], lambda);
            else
                res[i] = std::min(-z[i], lambda);
        }
    }
    void next_residual(VectorXd &res)
    {
        res.noalias() = cache_Ax + aux_z - XY;
    }
    void rho_changed_action() {}



    // Faster computation of epsilons and residuals
    double compute_eps_primal()
    {
        double r = std::max(cache_Ax.norm(), aux_z.norm());
        r = std::max(r, XY_norm);
        return r * eps_rel + std::sqrt(double(dim_dual)) * eps_abs;
    }
    double compute_eps_dual()
    {
        return dual_y.norm() * eps_rel + std::sqrt(double(dim_dual)) * eps_abs;
    }
    double compute_resid_dual(VectorXd &zdiff)
    {
        return rho * zdiff.norm();
    }
    double compute_resid_combined()
    {
        VectorXd tmp = aux_z - adj_z;
        return rho * resid_primal * resid_primal + rho * tmp.squaredNorm();
    }

public:
    ADMMDantzig(const MatrixXd &datX_, const VectorXd &datY_,
                double eps_abs_ = 1e-6,
                double eps_rel_ = 1e-6) :
        ADMMBase(datX_.cols(), datX_.cols(), datX_.cols(),
                 eps_abs_, eps_rel_),
        datX(&datX_), datY(&datY_),
        use_XX(datX_.rows() > datX_.cols() && datX_.cols() <= 1000),
        XY(datX_.transpose() * datY_),
        XY_norm(XY.norm()),
        lambda0(XY.array().abs().maxCoeff()),
        cache_Ax(dim_dual)
    {
        if(use_XX)
        {
            const MapMat mapX(datX_.data(), datX_.rows(), datX_.cols());
            Linalg::cross_prod_lower(XX, mapX);
            XX.triangularView<Eigen::Upper>() = XX.transpose();

            MatOpSymLower<double> op(XX);
            SymEigsSolver<double, LARGEST_ALGE> eigs(&op, 1, 3);
            srand(0);
            eigs.init();
            eigs.compute(10, 0.1);
            VectorXd evals = eigs.ritzvalues();
            sprad = evals[0];
            //sprad *= sprad;
        } else {
            MatOpXX<double> op(datX_);
            SymEigsSolver<double, LARGEST_ALGE> eigs(&op, 1, 3);
            srand(0);
            eigs.init();
            eigs.compute(10, 0.1);
            VectorXd evals = eigs.ritzvalues();
            sprad = evals[0];
            //sprad *= sprad;
        }

        Rcpp::Rcout << "sprad = " << sprad << std::endl;
    }

    double get_lambda_zero() { return lambda0; }

    // init() is a cold start for the first lambda
    void init(double lambda_, double rho_)
    {
        main_x.setZero();
        aux_z.setZero();
        dual_y.setZero();

        adj_z.setZero();
        adj_y.setZero();

        cache_Ax.setZero();

        lambda = lambda_;
        rho = rho_ / sprad;

        rho = std::pow(sprad, -2.0 / 3.0);
        Rcpp::Rcout << "rho = " << rho << std::endl;

        eps_primal = 0.0;
        eps_dual = 0.0;
        resid_primal = 9999;
        resid_dual = 9999;
        iter_counter = 0;

        rho_changed_action();
    }
    // when computing for the next lambda, we can use the
    // current main_x, aux_z, dual_y and rho as initial values
    void init_warm(double lambda_)
    {
        lambda = lambda_;

        eps_primal = 0.0;
        eps_dual = 0.0;
        resid_primal = 9999;
        resid_dual = 9999;
        iter_counter = 0;
    }
};



#endif // ADMMDANTZIG_H
