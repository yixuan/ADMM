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
        vec = -datX.transpose() * vec / gamma;
        vec += main_x;
        soft_threshold(res, vec, lambda / (rho * gamma));
    }
public:
    PADMMLasso_Worker(const RefMat &datX_,
                      const VectorXd &dual_y_,
                      const VectorXd &resid_primal_vec_) :
        PADMMBase_Worker(datX_, dual_y_, resid_primal_vec_)
    {
        // sprad is the largest eigenvalue of A_i'A_i
        MatrixXd XX;
        if(dim_main > dim_dual)
            XX = datX_ * datX_.transpose();
        else
            XX = datX_.transpose() * datX_;

        int n = XX.cols();
        VectorXd evec = XX * XX.col(0);
        VectorXd b(n);
        int niter = 0;
        do {
            b = evec.normalized();
            evec.noalias() = XX * b;
            sprad = b.dot(evec);
            niter++;
        } while(niter < 100 && (evec - sprad * b).norm() > 0.001 * sprad);
    }
    
    virtual ~PADMMLasso_Worker() {}
    
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
    double lmax;           // with this lambda, all coefficients will be zero

    virtual void next_z_bar(VectorXd &res, VectorXd &Ax_bar)
    {
        res.noalias() = ((*datY) + rho * Ax_bar + dual_y) / (n_comp + rho);
    }
    virtual void rho_changed_action() {}
    
public:
    PADMMLasso_Master(const MatrixXd &datX_, const VectorXd &datY_,
                      int nthread_,
                      double eps_abs_ = 1e-6,
                      double eps_rel_ = 1e-6) :
        PADMMBase_Master(datX_, 2 * nthread_, eps_abs_, eps_rel_),
        datY(&datY_)
    {
        int chunk_size = dim_main / n_comp;
        int last_size = chunk_size + dim_main % n_comp;

        for(int i = 0; i < n_comp - 1; i++)
        {
            worker[i] = new PADMMLasso_Worker(
                datX_.block(0, i * chunk_size, dim_dual, chunk_size),
                dual_y, resid_primal_vec);
        }
        worker[n_comp - 1] = new PADMMLasso_Worker(
            datX_.rightCols(last_size),
            dual_y, resid_primal_vec);

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
    virtual void init(double lambda_, double rho_)
    {
        dual_y.setZero();
        resid_primal_vec.setZero();
        rho = rho_;
        for(int i = 0; i < n_comp; i++)
        {
            static_cast<PADMMLasso_Worker *>(worker[i])->init(lambda_, rho_);
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
