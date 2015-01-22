#ifndef PADMMBASE_H
#define PADMMBASE_H

#include <RcppEigen.h>
// #include <ctime>

// Parallel ADMM for seperable objective function with regularization
//   minimize \sum f_i(x) + g(x)
// Equivalent to
//   minimize \sum f_i(x_i) + g(z)
//   s.t. x_i - z = 0
//
// x_i(p, 1), z(p, 1)
//

class PADMMBase_Worker
{
protected:
    typedef Eigen::VectorXd VectorXd;

    int dim_par;            // length of x_i and z

    VectorXd main_x;        // parameters to be optimized
    const VectorXd *aux_z;  // auxiliary parameters from master
    VectorXd dual_y;        // Lagrangian multiplier
    VectorXd cache_resid_primal;  // store main_x - aux_z

    double rho;             // augmented Lagrangian parameter

    // res = x in next iteration
    virtual void next_x(VectorXd &res) = 0;
    // action when rho is changed, e.g. re-factorize matrices
    virtual void rho_changed_action() {}

public:
    PADMMBase_Worker(int dim_par_, VectorXd &aux_z_) :
        dim_par(dim_par_),
        main_x(dim_par_),
        aux_z(&aux_z_),
        dual_y(dim_par_)
    {}
    
    virtual ~PADMMBase_Worker() {}
    
    virtual void update_rho(double rho_)
    {
        rho = rho_;
        rho_changed_action();
    }

    virtual void update_x()
    {
        VectorXd newx;
        next_x(newx);
        main_x.swap(newx);
    }

    virtual void update_y()
    {
        cache_resid_primal = main_x - (*aux_z);
        dual_y.noalias() += rho * cache_resid_primal;
    }
    
    virtual double squared_resid_primal()
    {
        return cache_resid_primal.squaredNorm();
    }
    
    virtual double squared_xnorm() { return main_x.squaredNorm(); }
    virtual double squared_ynorm() { return dual_y.squaredNorm(); }
    virtual void add_x_to(VectorXd &res) { res += main_x; }
    virtual void add_y_to(VectorXd &res) { res += dual_y; }
};


class PADMMBase_Master
{
protected:
    typedef Eigen::VectorXd VectorXd;

    int dim_par;               // length of x_i and z
    int n_comp;                // number of components in the objective function
    std::vector<PADMMBase_Worker *> worker;  // each worker handles a component

    VectorXd aux_z;            // master maintains the update of z

    double rho;                // augmented Lagrangian parameter
    double eps_abs;            // absolute tolerance
    double eps_rel;            // relative tolerance

    double eps_primal;         // tolerance for primal residual
    double eps_dual;           // tolerance for dual residual

    double resid_primal;       // primal residual
    double resid_dual;         // dual residual

    // res = z in next iteration
    virtual void next_z(VectorXd &res) = 0;
    // action when rho is changed, e.g. re-factorize matrices
    virtual void rho_changed_action() {}

    // calculating eps_primal
    virtual double compute_eps_primal()
    {
        double *tmp = new double[n_comp];

#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for(int i = 0; i < n_comp; i++)
        {
            tmp[i] = worker[i]->squared_xnorm();
        }
        double xnorm = sqrt(std::accumulate(tmp, tmp + n_comp, 0.0));
        delete [] tmp;
        
        double znorm = sqrt(double(dim_par)) * aux_z.norm();
        
        return std::max(xnorm, znorm) * eps_rel + sqrt(double(dim_par)) * eps_abs;
    }
    // calculating eps_dual
    virtual double compute_eps_dual()
    {
        double *tmp = new double[n_comp];

#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for(int i = 0; i < n_comp; i++)
        {
            tmp[i] = worker[i]->squared_ynorm();
        }
        double ynorm = sqrt(std::accumulate(tmp, tmp + n_comp, 0.0));
        delete [] tmp;
        
        return ynorm * eps_rel + sqrt(double(dim_par)) * eps_abs;
    }
    // increase or decrease rho in iterations
    virtual void update_rho()
    {
        if(resid_primal > resid_dual)
        {
            rho *= 2;
            rho_changed_action();
        }
        else if(resid_dual > resid_primal)
        {
            rho *= 0.5;
            rho_changed_action();
        }
    }

public:
    PADMMBase_Master(int dim_par_, int n_comp_,
        double eps_abs_ = 1e-6, double eps_rel_ = 1e-6) :
        dim_par(dim_par_), n_comp(n_comp_),
        worker(n_comp_),
        aux_z(dim_par_),
        eps_abs(eps_abs_), eps_rel(eps_rel_)
    {}
    
    virtual ~PADMMBase_Master() {}

    virtual void update_x()
    {
        eps_primal = compute_eps_primal();
        eps_dual = compute_eps_dual();
        update_rho();

#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for(int i = 0; i < n_comp; i++)
        {
            worker[i]->update_rho(rho);
            worker[i]->update_x();
        }
    }
    virtual void update_z()
    {
        VectorXd newz(dim_par);
        next_z(newz);

        VectorXd dual = newz - aux_z;
        resid_dual = rho * sqrt(double(n_comp)) * dual.norm();

        aux_z.swap(newz);
    }
    virtual void update_y()
    {
        resid_primal = 0;
        double *tmp = new double[n_comp];

#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for(int i = 0; i < n_comp; i++)
        {
            worker[i]->update_y();
            tmp[i] = worker[i]->squared_resid_primal();
        }
        resid_primal = sqrt(std::accumulate(tmp, tmp + n_comp, 0.0));
        delete [] tmp;
    }

    virtual void debuginfo()
    {
        Rcpp::Rcout << "eps_primal = " << eps_primal << std::endl;
        Rcpp::Rcout << "resid_primal = " << resid_primal << std::endl;
        Rcpp::Rcout << "eps_dual = " << eps_dual << std::endl;
        Rcpp::Rcout << "resid_dual = " << resid_dual << std::endl;
        Rcpp::Rcout << "rho = " << rho << std::endl;
    }

    virtual bool converged()
    {
        return (resid_primal < eps_primal) &&
               (resid_dual < eps_dual);
    }

    virtual int solve(int maxit)
    {
        int i;
        // double tx = 0, tz = 0, ty = 0;
        // clock_t t1, t2;
        for(i = 0; i < maxit; i++)
        {
            // t1 = clock();
            update_x();
            // t2 = clock();
            // tx += double(t2 - t1) / CLOCKS_PER_SEC;
            update_z();
            // t1 = clock();
            // tz += double(t1 - t2) / CLOCKS_PER_SEC;
            update_y();
            // t2 = clock();
            // ty += double(t2 - t1) / CLOCKS_PER_SEC;

            // debuginfo();
            if(converged())
                break;
        }
        // Rcpp::Rcout << "time - x: " << tx << " secs\n";
        // Rcpp::Rcout << "time - z: " << tz << " secs\n";
        // Rcpp::Rcout << "time - y: " << ty << " secs\n";
        // Rcpp::Rcout << "time - x + y + z: " << tx + ty + tz << " secs\n";
        return i + 1;
    }

    virtual VectorXd get_z() { return aux_z; }
};


#endif // PADMMBASE_H
