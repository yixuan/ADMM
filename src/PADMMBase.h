#ifndef PADMMBASE_H
#define PADMMBASE_H

#include <RcppEigen.h>
// #include <ctime>

// Parallel ADMM by splitting variables
//   minimize loss(\sum A_i * x_i - b) + \sum r_i(x_i)
// Equivalent to
//   minimize loss(\sum z_i - b) + \sum r_i(x_i)
//   s.t. A_i * x_i - z = 0
//
// x_i(p_i, 1), z(n, 1)
//
template<typename VecTypeX>
class PADMMBase_Worker
{
protected:
    typedef Eigen::VectorXd VectorXd;
    typedef Eigen::MatrixXd MatrixXd;
    typedef Eigen::Ref<const MatrixXd> RefMat;

    const RefMat datX;       // (sub)data matrix sent to this worker 
    int dim_main;            // length of x_i
    int dim_dual;            // length of A_i * x_i and z

    VecTypeX main_x;         // parameters to be optimized
    VectorXd Ax;             // A_i * x_i
    VectorXd aux_z;          // A_i * x_i - Ax_bar + z_bar

    const VectorXd *dual_y;  // y from master
    const VectorXd *resid_primal_vec; // Ax_bar - z_bar from master

    double comp_squared_resid_dual; // squared norm of dual residual
    
    double rho;              // augmented Lagrangian parameter

    // res = x in next iteration
    virtual void next_x(VecTypeX &res) = 0;
    // res = z in next iteration
    virtual void next_z(VectorXd &res)
    {
        res.noalias() = Ax - (*resid_primal_vec);
    }
    // action when rho is changed, e.g. re-factorize matrices
    virtual void rho_changed_action() {}

public:
    PADMMBase_Worker(const RefMat &datX_,
                     const VectorXd &dual_y_,
                     const VectorXd &resid_primal_vec_) :
        datX(datX_),
        dim_main(datX_.cols()), dim_dual(datX_.rows()),
        main_x(dim_main), Ax(dim_dual), aux_z(dim_dual),
        dual_y(&dual_y_), resid_primal_vec(&resid_primal_vec_)
    {}
    
    virtual ~PADMMBase_Worker() {}
    
    virtual void update_rho(double rho_)
    {
        rho = rho_;
        rho_changed_action();
    }

    virtual void update_x()
    {
        VecTypeX newx(dim_main);
        newx.setZero();
        next_x(newx);
        main_x.swap(newx);

        Ax = datX * main_x;
        // Rcpp::Rcout << "Ax - worker: " << Ax[0] << std::endl;
    }

    virtual void update_z()
    {
        VectorXd newz(dim_dual);
        next_z(newz);

        VectorXd dual = newz - aux_z;
        comp_squared_resid_dual = (datX.transpose() * dual).squaredNorm();

        aux_z.swap(newz);
    }
    
    virtual double squared_resid_dual() { return comp_squared_resid_dual; }

    virtual void add_Ax_to(VectorXd &res) { res += Ax; }

    virtual VecTypeX get_x() { return main_x; };
};


template<typename VecTypeX>
class PADMMBase_Master
{
protected:
    typedef Eigen::VectorXd VectorXd;
    typedef Eigen::MatrixXd MatrixXd;

    const MatrixXd *datX;      // data matrix
    int dim_main;              // number of all predictors
    int dim_dual;              // length of A_i * x_i and z
    int n_comp;                // number of components in the objective function
    std::vector<PADMMBase_Worker<VecTypeX> *> worker;  // each worker handles a component

    VectorXd dual_y;           // master maintains the update of y
    VectorXd resid_primal_vec; // stores Ax_bar - z_bar
    double Ax_bar_norm;        // norm of Ax_bar
    double z_bar_norm;         // norm of z_bar

    double rho;                // augmented Lagrangian parameter
    double eps_abs;            // absolute tolerance
    double eps_rel;            // relative tolerance

    double eps_primal;         // tolerance for primal residual
    double eps_dual;           // tolerance for dual residual

    double resid_primal;       // primal residual
    double resid_dual;         // dual residual

    // res = z_bar
    virtual void next_z_bar(VectorXd &res, VectorXd &Ax_bar) = 0;
    // action when rho is changed, e.g. re-factorize matrices
    virtual void rho_changed_action() {}

    // calculating eps_primal
    virtual double compute_eps_primal()
    {
        double r = std::max(Ax_bar_norm, z_bar_norm);
        return r * sqrt(double(n_comp)) * eps_rel + sqrt(double(dim_dual)) * eps_abs;
    }
    // calculating eps_dual
    virtual double compute_eps_dual()
    {
        VectorXd Aty = (*datX).transpose() * dual_y;
        
        return Aty.norm() * eps_rel + sqrt(double(dim_main)) * eps_abs;
    }
    // increase or decrease rho in iterations
    virtual void update_rho()
    {
        if(resid_primal > 10 * resid_dual)
        {
            rho *= 2;
            rho_changed_action();
        }
        else if(resid_dual > 10 * resid_primal)
        {
            rho *= 0.5;
            rho_changed_action();
        }
    }

public:
    PADMMBase_Master(const MatrixXd &datX_, int n_comp_,
                     double eps_abs_ = 1e-6, double eps_rel_ = 1e-6) :
        datX(&datX_), dim_main(datX_.cols()), dim_dual(datX_.rows()),
        n_comp(n_comp_),
        worker(n_comp_),
        dual_y(dim_dual), resid_primal_vec(dim_dual),
        Ax_bar_norm(0.0), z_bar_norm(0.0),
        eps_abs(eps_abs_), eps_rel(eps_rel_)
    {}
    
    virtual ~PADMMBase_Master() {}

    virtual void update_x()
    {
        eps_primal = compute_eps_primal();
        eps_dual = compute_eps_dual();
        update_rho();

        for(int i = 0; i < n_comp; i++)
        {
            worker[i]->update_rho(rho);
            worker[i]->update_x();
        }
    }
    virtual void update_y()
    {
        // calculate Ax_bar
        VectorXd Ax_bar(dim_dual);
        Ax_bar.setZero();
        for(int i = 0; i < n_comp; i++)
        {
            worker[i]->add_Ax_to(Ax_bar);
        }
        Ax_bar /= n_comp;
        Ax_bar_norm = Ax_bar.norm();

        // calculate z_bar from Ax_bar
        VectorXd z_bar(dim_dual);
        next_z_bar(z_bar, Ax_bar);
        z_bar_norm = z_bar.norm();

        // calculate primal residual
        resid_primal_vec = Ax_bar - z_bar;
        resid_primal = sqrt(double(n_comp)) * resid_primal_vec.norm();

        // update dual variable
        dual_y.noalias() += rho * resid_primal_vec;
        
        // dual residual vector = A_i * x_i - Ax_bar + z_bar
        resid_dual = 0.0;
        for(int i = 0; i < n_comp; i++)
        {
            worker[i]->update_z();
            resid_dual += worker[i]->squared_resid_dual();
        }
        resid_dual = sqrt(resid_dual);
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
            update_y();
            // t1 = clock();
            // ty += double(t1 - t2) / CLOCKS_PER_SEC;

            // debuginfo();
            if(converged())
                break;
        }
        // Rcpp::Rcout << "time - x: " << tx << " secs\n";
        // Rcpp::Rcout << "time - y: " << ty << " secs\n";
        // Rcpp::Rcout << "time - x + y: " << tx + ty << " secs\n";
        return i + 1;
    }

    virtual VecTypeX get_x() = 0;
};


#endif // PADMMBASE_H
