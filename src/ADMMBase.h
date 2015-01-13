#ifndef ADMMBASE_H
#define ADMMBASE_H

#include <RcppEigen.h>

// General problem setting
//   minimize f(x) + g(z)
//   s.t. Ax + Bz = c
//
// x(n, 1), z(m, 1), A(p, n), B(p, m), c(p, 1)
//
class ADMMBase
{
protected:
    typedef Eigen::VectorXd VectorXd;
    typedef Eigen::Ref<const VectorXd> Ref;

    int dim_main;  // dimension of x
    int dim_aux;   // dimension of z
    int dim_dual;  // dimension of Ax + Bz - c

    VectorXd main_x;  // parameters to be optimized
    VectorXd aux_z;   // auxiliary parameters
    VectorXd dual_y;  // Lagrangian multiplier

    double rho;      // augmented Lagrangian parameter
    double eps_abs;  // absolute tolerance
    double eps_rel;  // relative tolerance

    double eps_primal;  // tolerance for primal residual
    double eps_dual;    // tolerance for dual residual

    double resid_primal;  // primal residual
    double resid_dual;    // dual residual

    virtual void A_mult(VectorXd &x) = 0;   // (inplace) operation x -> Ax
    virtual void At_mult(VectorXd &x) = 0;  // (inplace) operation x -> A'x
    virtual void B_mult(VectorXd &x) = 0;   // (inplace) operation x -> Bx
    virtual double c_norm() = 0;            // L2 norm of c

    // res = Ax + Bz - c
    virtual void next_residual(VectorXd &res, const VectorXd &x, const VectorXd &z) = 0;
    // res = x in next iteration
    virtual void next_x(VectorXd &res) = 0;
    // res = z in next iteration
    virtual void next_z(VectorXd &res) = 0;
    // action when rho is changed, e.g. re-factorize matrices
    virtual void rho_changed_action() {}

    // calculating eps_primal
    virtual double compute_eps_primal()
    {
        VectorXd xcopy = main_x;
        VectorXd zcopy = aux_z;
        A_mult(xcopy);
        B_mult(zcopy);
        double r = std::max(xcopy.norm(), zcopy.norm());
        r = std::max(r, c_norm());
        return r * eps_rel + sqrt(double(dim_dual)) * eps_abs;
    }
    // calculating eps_dual
    virtual double compute_eps_dual()
    {
        VectorXd ycopy = dual_y;
        At_mult(ycopy);
        return ycopy.norm() * eps_rel + sqrt(double(dim_main)) * eps_abs;
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
    ADMMBase(int n_, int m_, int p_,
             double eps_abs_ = 1e-6, double eps_rel_ = 1e-6) :
        dim_main(n_), dim_aux(m_), dim_dual(p_),
        main_x(n_), aux_z(m_), dual_y(p_),  // allocate space but do not set values
        eps_abs(eps_abs_), eps_rel(eps_rel_)
    {}
    
    // init() needs to be called every time we want to solve
    // for a new lambda
    virtual void init()
    {
        main_x.setZero();
        aux_z.setZero();
        dual_y.setZero();
        rho = 1e-3;
        eps_primal = 0.0;
        eps_dual = 0.0;
        resid_primal = 9999;
        resid_dual = 9999;
    }
    // provide rho
    virtual void init(double rho_)
    {
        main_x.setZero();
        aux_z.setZero();
        dual_y.setZero();
        rho = rho_;
        eps_primal = 0.0;
        eps_dual = 0.0;
        resid_primal = 9999;
        resid_dual = 9999;
    }
    // provide initial values
    virtual void init(const Ref &init_x, double rho_)
    {
        main_x = init_x;
        aux_z = init_x;
        dual_y.setZero();
        rho = rho_;
        eps_primal = 0.0;
        eps_dual = 0.0;
        resid_primal = 9999;
        resid_dual = 9999;
    }

    virtual void update_x()
    {
        eps_primal = compute_eps_primal();
        eps_dual = compute_eps_dual();
        update_rho();

        next_x(main_x);
    }
    virtual void update_z()
    {
        VectorXd newz(dim_aux);
        next_z(newz);

        VectorXd dual = newz - aux_z;
        B_mult(dual);
        At_mult(dual);
        resid_dual = rho * dual.norm();

        aux_z.swap(newz);
    }
    virtual void update_y()
    {
        VectorXd newr(dim_dual);
        next_residual(newr, main_x, aux_z);

        resid_primal = newr.norm();

        dual_y.noalias() += rho * newr;
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
        for(i = 0; i < maxit; i++)
        {
            update_x();
            update_z();
            update_y();
            // debuginfo();
            if(converged())
                break;
        }
        return i;
    }

    virtual VectorXd get_x() { return main_x; }
    virtual VectorXd get_z() { return aux_z; }
    virtual VectorXd get_y() { return dual_y; }
};



#endif // ADMMBASE_H
