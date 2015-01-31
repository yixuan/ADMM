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
    typedef Eigen::Map<const MatrixXd> MapMat;
    
    double lambda; // L1 penalty parameter
    double sprad;  // spectral radius of A_i'A_i

    int iter_counter;

    virtual void active_set_update(SparseVector &res)
    {
        double gamma = 2 * rho + sprad;
        double penalty = lambda / (rho * gamma);
        VectorXd vec = ((*dual_y) / rho + (*resid_primal_vec)) / gamma;
        res = main_x;

        for(SparseVector::InnerIterator iter(res); iter; ++iter)
        {
            double val = iter.value() - vec.dot(datX.col(iter.index()));

            if(val > penalty)
            {
                iter.valueRef() = val - penalty;
            }
            else if(val < -penalty)
            {
                iter.valueRef() = val + penalty;
            }
            else
            {
                iter.valueRef() = 0.0;
            }
        }

        res.prune(0.0);
    }

    // res = x in next iteration
    virtual void next_x(SparseVector &res)
    {
        if(iter_counter % 10 == 0)
        {
            double gamma = 2 * rho + sprad;
            VectorXd tmp = (*dual_y) / rho + (*resid_primal_vec);
            VectorXd vec(dim_main);
            if(use_BLAS)
            {
                BLAStprod(vec, -1.0/gamma, datX.data(), dim_dual, dim_main, tmp);
            }
            else
            {
                vec.noalias() = -datX.transpose() * tmp / gamma;
            }
            vec += main_x;
            soft_threshold(res, vec, lambda / (rho * gamma));
        } else {
            active_set_update(res);
        }
        iter_counter++;        
    }

    // calculating the spectral radius of X'X
    // in this case it is the largest eigenvalue of X'X
    static double spectral_radius(const MapMat &X)
    {
        Rcpp::NumericMatrix mat = Rcpp::wrap(X);
    
        Rcpp::Environment ADMM = Rcpp::Environment::namespace_env("ADMM");
        Rcpp::Function spectral_radius = ADMM[".spectral_radius"];
    
        return Rcpp::as<double>(spectral_radius(mat));
    }

public:
    PADMMLasso_Worker(const double *datX_ptr_,
                      int X_rows_, int X_cols_,
                      const VectorXd &dual_y_,
                      const VectorXd &resid_primal_vec_,
                      bool use_BLAS_) :
        PADMMBase_Worker(datX_ptr_, X_rows_, X_cols_, dual_y_, resid_primal_vec_, use_BLAS_)
    {
        sprad = spectral_radius(datX);
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
        iter_counter = 0;

        rho_changed_action();
    }
    // when computing for the next lambda, we can use the
    // current main_x, aux_z, dual_y and rho as initial values
    virtual void init_warm(double lambda_)
    {
        lambda = lambda_;
        iter_counter = 0;
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
    double lambda0;        // with this lambda, all coefficients will be zero

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
        PADMMBase_Master(datX_, 2 * nthread_, eps_abs_, eps_rel_),
        datY(&datY_)
    {
        int chunk_size = dim_main / n_comp;
        int last_size = chunk_size + dim_main % n_comp;
        double avg_sprad_collector = 0.0;

// we can no longer parallelize this since the constructor will call an R function
/*
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic) reduction(+:avg_sprad_collector)
#endif
*/
        for(int i = 0; i < n_comp; i++)
        {
            if(i < n_comp - 1)
                worker[i] = new PADMMLasso_Worker(
                    &datX_(0, i * chunk_size), dim_dual, chunk_size,
                    dual_y, resid_primal_vec,
                    use_BLAS_);
            else
                worker[i] = new PADMMLasso_Worker(
                    &datX_(0, i * chunk_size), dim_dual, last_size,
                    dual_y, resid_primal_vec,
                    use_BLAS_);

            avg_sprad_collector += static_cast<PADMMLasso_Worker *>(worker[i])->get_spectral_radius();
        }

        avg_sprad = avg_sprad_collector / n_comp;
        lambda0 = (datX_.transpose() * datY_).array().abs().maxCoeff();
    }
        
    virtual ~PADMMLasso_Master()
    {
        for(int i = 0; i < n_comp; i++)
        {
            delete worker[i];
        }
    }

    virtual double get_lambda_zero() { return lambda0; }

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
