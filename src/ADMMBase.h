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

    int dim_main;
    int dim_aux;
    int dim_dual;
    
    VectorXd main_x;
    VectorXd aux_z;
    VectorXd dual_y;

    double rho;
    double eps_abs;
    double eps_rel;

    double eps_primal;
    double eps_dual;

    double resid_primal;
    double resid_dual;

    virtual void A_mult(VectorXd &x) = 0;
    virtual void At_mult(VectorXd &x) = 0;
    virtual void B_mult(VectorXd &x) = 0;
    virtual double c_norm() = 0;
    
    virtual void next_residual(VectorXd &res, const VectorXd &x, const VectorXd &z) = 0;
    virtual void next_x(VectorXd &res) = 0;
    virtual void next_z(VectorXd &res) = 0;
    virtual void rho_changed_action() {}

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
    virtual double compute_eps_dual()
    {
        VectorXd ycopy = dual_y;
        At_mult(ycopy);
        return ycopy.norm() * eps_rel + sqrt(double(dim_main)) * eps_abs;
    }
    
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
             double eps_abs_ = 1e-8, double eps_rel_ = 1e-8,
             double rho_ = 1e-4) :
        dim_main(n_), dim_aux(m_), dim_dual(p_),
        main_x(VectorXd::Zero(n_)),
        aux_z(VectorXd::Zero(m_)),
        dual_y(VectorXd::Zero(p_)),
        rho(rho_), eps_abs(eps_abs_), eps_rel(eps_rel_),
        eps_primal(999), eps_dual(999), resid_primal(9999), resid_dual(9999)
    {}
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
        dual_y = dual_y + rho * newr;
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

    virtual VectorXd get_x() { return main_x; }
    virtual VectorXd get_z() { return aux_z; }
    virtual VectorXd get_y() { return dual_y; }
};



#endif // ADMMBASE_H