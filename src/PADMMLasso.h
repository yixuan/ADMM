#ifndef PADMMLASSO_H
#define PADMMLASSO_H

#include "PADMMBase.h"

class PADMMLasso_Worker: public PADMMBase_Worker< Eigen::SparseVector<double> >
{
private:
    typedef Eigen::MatrixXd MatrixXd;
    typedef Eigen::VectorXd VectorXd;
    typedef Eigen::ArrayXd ArrayXd;
    typedef Eigen::SparseVector<double> SparseVector;
    typedef Eigen::Ref<const MatrixXd> RefMat;
    typedef Eigen::Map<MatrixXd> MapMat;
    
    double lambda; // L1 penalty parameter
    double sprad;  // spectral radius of A_i'A_i

    // res = x in next iteration
    virtual void next_x(SparseVector &res)
    {
        double gamma = 2 * rho + sprad;
        VectorXd vec = (*dual_y) / rho + (*resid_primal_vec);
        VectorXd vec2(dim_main);
        if(use_BLAS)
        {
            BLAStprod(vec2, -1.0/gamma, datX_ptr, dim_dual, dim_main, vec);
        }
        else
        {
            vec2.noalias() = -datX.transpose() * vec / gamma;
        }
        vec2 += main_x;
        soft_threshold(res, vec2, lambda / (rho * gamma));
    }
public:
    PADMMLasso_Worker(const RefMat &datX_,
                      const double *datX_ptr_,
                      const VectorXd &dual_y_,
                      const VectorXd &resid_primal_vec_,
                      bool use_BLAS_) :
        PADMMBase_Worker(datX_, datX_ptr_, dual_y_, resid_primal_vec_, use_BLAS_)
    {
        sprad = spectral_radius(datX_);
    }
     
    virtual ~PADMMLasso_Worker() {}

    virtual double get_spectral_radius() { return sprad; }

    // init() is a cold start for the first lambda
    virtual void init(double lambda_, double rho_)
    {
        main_x.setZero();
        Ax.setZero();
        aux_z.setZero();
        lambda = lambda_;
        rho = rho_;

        rho_changed_action();
    }
    // when computing for the next lambda, we can use the
    // current main_x, aux_z, dual_y and rho as initial values
    virtual void init_warm(double lambda_) { lambda = lambda_; }

    // calculating the spectral radius of X'X, i.e., the largest eigenvalue
    virtual double spectral_radius(const RefMat &X)
    {
        double sprad = 0.0;
    
        int n = X.rows();
        int p = X.cols();
        bool thinX = n > p;
        int dim = std::min(n, p);

        VectorXd evec(dim);
        VectorXd b(dim);
        VectorXd tmp(std::max(n, p));
        if(thinX)
        {
            tmp.noalias() = X * X.row(0);
            evec.noalias() = X.transpose() * tmp;
        }
        else
        {
            tmp.noalias() = X.transpose() * X.col(0);
            evec.noalias() = X * tmp;
        }

        for(int i = 0; i < 100; i++)
        {
            b = evec.normalized();
            if(thinX)
            {
                if(use_BLAS)
                {
                    BLASprod(tmp, 1.0, datX_ptr, dim_dual, dim_main, b);
                    BLAStprod(evec, 1.0, datX_ptr, dim_dual, dim_main, tmp);
                }
                else
                {
                    tmp.noalias() = X * b;
                    evec.noalias() = X.transpose() * tmp;
                }
                
            }
            else
            {
                if(use_BLAS)
                {
                    BLAStprod(tmp, 1.0, datX_ptr, dim_dual, dim_main, b);
                    BLASprod(evec, 1.0, datX_ptr, dim_dual, dim_main, tmp);
                }
                else
                {
                    tmp.noalias() = X.transpose() * b;
                    evec.noalias() = X * tmp;
                }
            }
            sprad = b.dot(evec);
            if((evec - sprad * b).norm() < 0.001 * sprad)
                break;
        }

        return sprad;
    }

    static void soft_threshold(SparseVector &res, VectorXd &vec, const double &penalty)
    {
        res.setZero();
        res.reserve(vec.size() / 2);

        double *ptr = vec.data();
        for(int i = 0; i < vec.size(); i++)
        {
            if(ptr[i] > penalty)
            {
                res.insertBack(i) = ptr[i] - penalty;
            }
            else if(ptr[i] < -penalty)
            {
                res.insertBack(i) = ptr[i] + penalty;
            }
        }
    }
};

class PADMMLasso_Master: public PADMMBase_Master< Eigen::SparseVector<double> >
{
private:
    typedef Eigen::MatrixXd MatrixXd;
    typedef Eigen::VectorXd VectorXd;
    typedef Eigen::ArrayXd ArrayXd;
    typedef Eigen::SparseVector<double> SparseVector;

    const VectorXd *datY;  // response vector
    double avg_sprad;      // average spectral radius of A_i' * A_i
    double lmax;           // with this lambda, all coefficients will be zero

    virtual void next_z_bar(VectorXd &res, VectorXd &Ax_bar)
    {
        res.noalias() = ((*datY) + rho * Ax_bar + dual_y) / (n_comp + rho);
    }
    virtual void rho_changed_action() {}
    
public:
    PADMMLasso_Master(const MatrixXd &datX_, const VectorXd &datY_,
                      int nthread_, bool use_BLAS_,
                      double eps_abs_ = 1e-6,
                      double eps_rel_ = 1e-6) :
        PADMMBase_Master(datX_, nthread_, eps_abs_, eps_rel_),
        datY(&datY_)
    {
        int chunk_size = dim_main / n_comp;
        int last_size = chunk_size + dim_main % n_comp;
        double avg_sprad_collector = 0.0;

#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic) reduction(+:avg_sprad_collector)
#endif
        for(int i = 0; i < n_comp; i++)
        {
            if(i < n_comp - 1)
                worker[i] = new PADMMLasso_Worker(
                    datX_.block(0, i * chunk_size, dim_dual, chunk_size),
                    &datX_(0, i * chunk_size),
                    dual_y, resid_primal_vec,
                    use_BLAS_);
            else
                worker[i] = new PADMMLasso_Worker(
                    datX_.rightCols(last_size),
                    &datX_(0, i * chunk_size),
                    dual_y, resid_primal_vec,
                    use_BLAS_);

            avg_sprad_collector += static_cast<PADMMLasso_Worker *>(worker[i])->get_spectral_radius();
        }

        avg_sprad = avg_sprad_collector / n_comp;
        lmax = (datX_.transpose() * datY_).array().abs().maxCoeff();
    }
        
    virtual ~PADMMLasso_Master()
    {
        for(int i = 0; i < n_comp; i++)
        {
            delete worker[i];
        }
    }

    virtual double lambda_max() { return lmax; }

    // init() is a cold start for the first lambda
    virtual void init(double lambda_, double rho_ratio_)
    {
        dual_y.setZero();
        resid_primal_vec.setZero();
        rho = lambda_ / (rho_ratio_ * avg_sprad);
        for(int i = 0; i < n_comp; i++)
        {
            static_cast<PADMMLasso_Worker *>(worker[i])->init(lambda_, rho);
        }
        eps_primal = 0.0;
        eps_dual = 0.0;
        resid_primal = 9999;
        resid_dual = 9999;

        rho_changed_action();
    }
    // when computing for the next lambda, we can use the
    // current main_x, aux_z, dual_y and rho as initial values
    virtual void init_warm(double lambda_)
    {
        for(int i = 0; i < n_comp; i++)
        {
            static_cast<PADMMLasso_Worker *>(worker[i])->init_warm(lambda_);
        }
        eps_primal = 0.0;
        eps_dual = 0.0;
        resid_primal = 9999;
        resid_dual = 9999;
    }

    virtual SparseVector get_x()
    {
        SparseVector res(dim_main);
        res.reserve(dim_main / 2);
        
        int offset = 0;
        for(int i = 0; i < n_comp; i++)
        {
            SparseVector comp = worker[i]->get_x();
            for(SparseVector::InnerIterator iter(comp); iter; ++iter)
            {
                res.insertBack(iter.index() + offset) = iter.value();
            }
            offset += comp.size();
        }
        return res;
    }
};

#endif // PADMMLASSO_H
