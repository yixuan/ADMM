#ifndef PADMMBP_H
#define PADMMBP_H

#include "PADMMBase.h"

class PADMMBP_Worker: public PADMMBase_Worker< Eigen::SparseVector<double> >
{
private:
    typedef Eigen::MatrixXd MatrixXd;
    typedef Eigen::VectorXd VectorXd;
    typedef Eigen::ArrayXd ArrayXd;
    typedef Eigen::SparseVector<double> SparseVector;
    typedef Eigen::Map<const MatrixXd> MapMat;

    double sprad;   // spectral radius of A_i'A_i

    int iter_counter;

    virtual void active_set_update(SparseVector &res)
    {
        double gamma = 2 * rho + sprad;
        double penalty = 1.0 / (rho * gamma);
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
            VectorXd vec = -datX.transpose() * tmp / gamma;
            vec += main_x;
            soft_threshold(res, vec, 1.0 / (rho * gamma));
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
        Rcpp::Function spectral_radius = ADMM[".spectral_radius_xx"];

        return Rcpp::as<double>(spectral_radius(mat));
    }

public:
    PADMMBP_Worker(const double *datX_ptr_,
                   int X_rows_, int X_cols_,
                   const VectorXd &dual_y_,
                   const VectorXd &resid_primal_vec_) :
        PADMMBase_Worker(datX_ptr_, X_rows_, X_cols_, dual_y_, resid_primal_vec_, false),
        iter_counter(0)
    {
        sprad = spectral_radius(datX);
        main_x.setZero();
        Ax.setZero();
        aux_z.setZero();
    }

    virtual ~PADMMBP_Worker() {}

    virtual double get_spectral_radius() { return sprad; }

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

class PADMMBP_Master: public PADMMBase_Master< Eigen::SparseVector<double> >
{
private:
    typedef Eigen::MatrixXd MatrixXd;
    typedef Eigen::VectorXd VectorXd;
    typedef Eigen::ArrayXd ArrayXd;
    typedef Eigen::SparseVector<double> SparseVector;

    const VectorXd *datY;  // response vector
    double avg_sprad;      // average spectral radius of A_i' * A_i

    virtual void next_z_bar(VectorXd &res, VectorXd &Ax_bar)
    {
        res.noalias() = (*datY) / n_comp;
    }
    virtual void rho_changed_action() {}

public:
    PADMMBP_Master(const MatrixXd &datX_, const VectorXd &datY_,
                   int nthread_,
                   double eps_abs_ = 1e-6,
                   double eps_rel_ = 1e-6) :
        PADMMBase_Master(datX_, nthread_, eps_abs_, eps_rel_),
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
                worker[i] = new PADMMBP_Worker(
                    &datX_(0, i * chunk_size), dim_dual, chunk_size,
                    dual_y, resid_primal_vec);
            else
                worker[i] = new PADMMBP_Worker(
                    &datX_(0, i * chunk_size), dim_dual, last_size,
                    dual_y, resid_primal_vec);

            avg_sprad_collector += static_cast<PADMMBP_Worker *>(worker[i])->get_spectral_radius();
        }

        avg_sprad = avg_sprad_collector / n_comp;
    }

    virtual ~PADMMBP_Master()
    {
        for(int i = 0; i < n_comp; i++)
        {
            delete worker[i];
        }
    }

    // init() is a cold start for the first lambda
    virtual void init(double rho_ratio_)
    {
        dual_y.setZero();
        resid_primal_vec.setZero();
        rho = 1.0 / (rho_ratio_ * avg_sprad);
        for(int i = 0; i < n_comp; i++)
        {
            static_cast<PADMMBP_Worker *>(worker[i])->update_rho(rho);
        }
        eps_primal = 0.0;
        eps_dual = 0.0;
        resid_primal = 9999;
        resid_dual = 9999;

        rho_changed_action();
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

#endif // PADMMBP_H
