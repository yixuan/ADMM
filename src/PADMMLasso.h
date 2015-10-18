#ifndef PADMMLASSO_H
#define PADMMLASSO_H

#include "PADMMBase.h"
#include "Linalg/BlasWrapper.h"

class PADMMLasso_Worker: public PADMMBase_Worker< float, Eigen::VectorXf, Eigen::SparseVector<float> >
{
private:
    typedef Eigen::SparseVector<float> SparseVector;
    typedef Eigen::LLT<Matrix> LLT;

    Vector Ab;
    LLT solver;

    // res = x in next iteration
    void next_x(Vector &res)
    {
        Vector rhs = Ab - dual_y;
        for(SparseVector::InnerIterator iter(aux_z); iter; ++iter)
            rhs[iter.index()] += rho * iter.value();

        if(subA.rows() >= subA.cols())
        {
            res.noalias() = solver.solve(rhs);
        } else {
            res.noalias() = rhs - subA.transpose() * solver.solve(subA * rhs);
            res /= float(rho);
        }

    }
    // res = primal residual in next iteration
    void next_residual(Vector &res)
    {
        res = main_x;
        res -= aux_z;
    }

public:
    PADMMLasso_Worker(ConstGenericMatrix &subA_, ConstGenericVector &subb_, SparseVector &aux_z_) :
        PADMMBase_Worker(subA_, subb_, aux_z_),
        Ab(subA.transpose() * subb)
    {}

    ~PADMMLasso_Worker() {}

    // init() is a cold start for the first lambda
    void init(double rho_)
    {
        main_x.setZero();
        dual_y.setZero();
        rho = rho_;

        Matrix AA;
        if(subA.rows() >= subA.cols())
            Linalg::cross_prod_lower(AA, subA);
        else
            Linalg::tcross_prod_lower(AA, subA);
        AA.diagonal().array() += rho;
        solver.compute(AA.selfadjointView<Eigen::Lower>());

        rho_changed_action();
    }

    void add_xu_to(Vector &res)
    {
        res.noalias() += main_x + dual_y / rho;
    }
};

class PADMMLasso_Master: public PADMMBase_Master< float, Eigen::VectorXf, Eigen::SparseVector<float> >
{
private:
    typedef Eigen::Map<const Matrix> MapMat;
    typedef Eigen::Map<const Vector> MapVec;
    typedef const Eigen::Ref<const Matrix> ConstGenericMatrix;
    typedef const Eigen::Ref<const Vector> ConstGenericVector;
    typedef Eigen::SparseVector<float> SparseVector;

    double lambda;         // L1 penalty parameter
    const double lambda0;  // with this lambda, all coefficients will be zero

    static void soft_threshold(SparseVector &res, const Vector &vec, const double &penalty)
    {
        int v_size = vec.size();
        res.setZero();
        res.reserve(v_size);

        const float *ptr = vec.data();
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
        Vector vec = Vector::Zero(dim_aux);
        for(int i = 0; i < n_comp; i++)
        {
            static_cast<PADMMLasso_Worker *>(worker[i])->add_xu_to(vec);
        }
        vec /= n_comp;
        soft_threshold(res, vec, lambda / (rho * n_comp));
    }

    // Calculate ||v1 - v2||^2 when v1 and v2 are sparse
    static double diff_squared_norm(const SparseVector &v1, const SparseVector &v2)
    {
        const int n1 = v1.nonZeros(), n2 = v2.nonZeros();
        const float *v1_val = v1.valuePtr(), *v2_val = v2.valuePtr();
        const int *v1_ind = v1.innerIndexPtr(), *v2_ind = v2.innerIndexPtr();

        float r = 0.0;
        int i1 = 0, i2 = 0;
        while(i1 < n1 && i2 < n2)
        {
            if(v1_ind[i1] == v2_ind[i2])
            {
                float val = v1_val[i1] - v2_val[i2];
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

    double compute_resid_dual(const SparseVector &new_z)
    {
        return rho * std::sqrt(n_comp * diff_squared_norm(new_z, aux_z));
    }

public:
    PADMMLasso_Master(ConstGenericMatrix &datX_, ConstGenericVector &datY_,
                      int nthread_,
                      double eps_abs_ = 1e-6,
                      double eps_rel_ = 1e-6) :
        PADMMBase_Master(datX_.cols(), nthread_, eps_abs_, eps_rel_),
        lambda0((datX_.transpose() * datY_).cwiseAbs().maxCoeff())
    {
        const int chunk_size = datX_.rows() / n_comp;
        const int last_size = chunk_size + datX_.rows() % n_comp;

#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(int i = 0; i < n_comp; i++)
        {
            if(i < n_comp - 1)
                worker[i] = new PADMMLasso_Worker(
                    datX_.block(i * chunk_size, 0, chunk_size, dim_main),
                    datY_.segment(i * chunk_size, chunk_size), aux_z);
            else
                worker[i] = new PADMMLasso_Worker(
                    datX_.block(i * chunk_size, 0, last_size, dim_main),
                    datY_.segment(i * chunk_size, last_size), aux_z);
        }
    }

    ~PADMMLasso_Master()
    {
        for(int i = 0; i < n_comp; i++)
        {
            delete worker[i];
        }
    }

    double get_lambda_zero() { return lambda0; }

    // init() is a cold start for the first lambda
    void init(double lambda_, double rho_)
    {
        aux_z.setZero();
        lambda = lambda_;
        rho = rho_;

        if(rho <= 0)
            rho = lambda / n_comp;

        for(int i = 0; i < n_comp; i++)
        {
            static_cast<PADMMLasso_Worker *>(worker[i])->init(rho);
        }
        eps_primal = 0.0;
        eps_dual = 0.0;
        resid_primal = 9999;
        resid_dual = 9999;

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
    }
};

#endif // PADMMLASSO_H
