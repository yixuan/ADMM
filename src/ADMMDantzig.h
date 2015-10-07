#ifndef ADMMDANTZIG_H
#define ADMMDANTZIG_H

#include "ADMMBase.h"
#include "Linalg/BlasWrapper.h"
#include "Spectra/SymEigsSolver.h"
#include "ADMMMatOp.h"

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
class ADMMDantzig: public ADMMBase<Eigen::SparseVector<double>, Eigen::VectorXd, Eigen::VectorXd>
{
protected:
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Map<const Matrix> MapMat;
    typedef Eigen::Map<const Vector> MapVec;
    typedef const Eigen::Ref<const Matrix> ConstGenericMatrix;
    typedef const Eigen::Ref<const Vector> ConstGenericVector;
    typedef Eigen::SparseVector<Scalar> SparseVector;

    MapMat datX;                  // pointer to data matrix
    MapVec datY;                  // pointer response vector
    bool use_XX;                  // whether to cache X'X
    Matrix XX;                    // X'X
    const Vector XY;              // X'y
    const Scalar XY_norm;         // L2 norm of X'y

    Scalar sprad;                 // spectral radius of X'XX'X
    Scalar lambda;                // penalty parameter
    const Scalar lambda0;         // minimum lambda to make coefficients all zero

    int iter_counter;             // which iteration are we in?

    Vector cache_Ax;              // cache Ax



    // x -> Ax
    void A_mult(Vector &res, SparseVector &x)
    {
        if(use_XX)
        {
            res.noalias() = XX * x;
        } else {
            res.noalias() = datX.transpose() * (datX * x);
        }
    }
    // y -> A'y
    void At_mult(Vector &res, Vector &y)
    {
        if(use_XX)
        {
            res.noalias() = XX * y;
        } else {
            res.noalias() = datX.transpose() * (datX * y);
        }
    }
    // z -> Bz
    void B_mult(Vector &res, Vector &z)
    {
        res.swap(z);
    }
    // ||c||_2
    double c_norm() { return XY_norm; }



    static void soft_threshold(SparseVector &res, const Vector &vec, const double &penalty)
    {
        int v_size = vec.size();
        res.setZero();
        res.reserve(v_size);

        const Scalar *ptr = vec.data();
        for(int i = 0; i < v_size; i++)
        {
            if(ptr[i] > penalty)
                res.insertBack(i) = ptr[i] - penalty;
            else if(ptr[i] < -penalty)
                res.insertBack(i) = ptr[i] + penalty;
        }
    }

    // 4^k - 1, k = 0, 1, 2, ...
    static bool is_regular_update(unsigned int x)
    {
        if(x == 0 || x == 3 || x == 15 || x == 63)  return true;
        x++;
        if( x & (x - 1) )  return false;
        return x & 0x55555555;
    }

    void next_x(SparseVector &res)
    {
        if(lambda > lambda0 - 1e-5)
        {
            res.setZero();
            return;
        }

        // iter_counter = 0, 3, 15, 63, .... (4^k - 1)
        if(is_regular_update(iter_counter))
        {
            regular_update(res);
        } else {
            // active_set_update(res);
            regular_update(res);
        }

        iter_counter++;
    }
    void regular_update(SparseVector &res)
    {
        Vector rhs = (cache_Ax + aux_z + dual_y / rho - XY) / (-sprad);
        Vector vec;
        if(use_XX)
        {
            vec.noalias() = XX * rhs;
        } else {
            vec.noalias() = datX.transpose() * (datX * rhs);
        }
        vec += main_x;
        soft_threshold(res, vec, 1.0 / (rho * sprad));
    }
    void active_set_update(SparseVector &res)
    {
        const Scalar gamma = sprad;
        const Scalar penalty = 1.0 / (rho * gamma);
        Vector vec = (cache_Ax + aux_z + dual_y / rho - XY) / gamma;
        res = main_x;

        Scalar *val_ptr = res.valuePtr();
        const int *ind_ptr = res.innerIndexPtr();
        const int nnz = res.nonZeros();

        #pragma omp parallel for
        for(int i = 0; i < nnz; i++)
        {
            const Scalar val = val_ptr[i] - vec.dot(XX.col(ind_ptr[i]));

            if(val > penalty)
                val_ptr[i] = val - penalty;
            else if(val < -penalty)
                val_ptr[i] = val + penalty;
            else
                val_ptr[i] = 0.0;
        }

        res.prune(0.0);
    }
    void next_z(Vector &res)
    {
        if(use_XX)
        {
            cache_Ax.noalias() = XX * main_x;
        } else {
            cache_Ax.noalias() = datX.transpose() * (datX * main_x);
        }

        Vector z = cache_Ax + dual_y / rho - XY;
        for(int i = 0; i < dim_aux; i++)
        {
            if(z[i] > 0)
                res[i] = -std::min(z[i], lambda);
            else
                res[i] = std::min(-z[i], lambda);
        }
    }
    void next_residual(Vector &res)
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
        return std::sqrt(sprad) * dual_y.norm() * eps_rel + std::sqrt(double(dim_main)) * eps_abs;
    }
    double compute_resid_dual(const Vector &new_z)
    {
        return rho * std::sqrt(sprad) * (new_z - aux_z).norm();
    }

public:
    ADMMDantzig(ConstGenericMatrix &datX_, ConstGenericVector &datY_,
                double eps_abs_ = 1e-6,
                double eps_rel_ = 1e-6) :
        ADMMBase(datX_.cols(), datX_.cols(), datX_.cols(),
                 eps_abs_, eps_rel_),
        datX(datX_.data(), datX_.rows(), datX_.cols()),
        datY(datY_.data(), datY_.size()),
        use_XX(datX.rows() > datX.cols() && datX.cols() <= 1000),
        XY(datX.transpose() * datY),
        XY_norm(XY.norm()),
        lambda0(XY.cwiseAbs().maxCoeff()),
        cache_Ax(dim_dual)
    {
        if(use_XX)
        {
            Linalg::cross_prod_lower(XX, datX);
            XX.triangularView<Eigen::Upper>() = XX.transpose();

            MatOpSymLower<double> op(XX);
            Spectra::SymEigsSolver< double, Spectra::LARGEST_ALGE, MatOpSymLower<double> > eigs(&op, 1, 3);
            srand(0);
            eigs.init();
            eigs.compute(10, 0.1);
            Vector evals = eigs.eigenvalues();
            sprad = evals[0];
            sprad *= sprad;
        } else {
            MatOpXX<double> op(datX);
            Spectra::SymEigsSolver< double, Spectra::LARGEST_ALGE, MatOpXX<double> > eigs(&op, 1, 3);
            srand(0);
            eigs.init();
            eigs.compute(10, 0.1);
            Vector evals = eigs.eigenvalues();
            sprad = evals[0];
            sprad *= sprad;
        }
    }

    double get_lambda_zero() const { return lambda0; }

    // init() is a cold start for the first lambda
    void init(double lambda_, double rho_)
    {
        main_x.setZero();
        aux_z.setZero();
        dual_y.setZero();
        cache_Ax.setZero();

        lambda = lambda_;
        rho = rho_;
        if(rho <= 0)
        {
            rho = 1.0 / std::sqrt(sprad);
        }

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
