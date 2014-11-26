#ifndef ADMMBASE_H
#define ADMMBASE_H

#include <RcppEigen.h>

// General problem setting
//   minimize f(x) + g(z)
//   s.t. Ax + Bz = c
//
// x(n, 1), z(m, 1), A(p, n), B(p, m), c(p, 1)
//
// In this implementation, we assume that x, z and c are of type
// Eigen::VectorXd, and A, B are generic, since they could take
// special structures, e.g. scalar constants.
class ADMMBase
{
protected:
    typedef Eigen::VectorXd VectorXd;

    int constr_p;
    int dim_n;
    int dim_m;
    
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
    virtual void residual(VectorXd &res, const VectorXd &x, const VectorXd &z) = 0;

    virtual double compute_eps_primal()
    {
        VectorXd xcopy = main_x;
        VectorXd zcopy = aux_z;
        A_mult(xcopy);
        B_mult(zcopy);
        double r = std::max(xcopy.norm(), zcopy.norm());
        r = std::max(r, c_norm());
        return r * eps_rel + sqrt(double(constr_p)) * eps_abs;
    }
    virtual double compute_eps_dual()
    {
        VectorXd ycopy = dual_y;
        At_mult(ycopy);
        return ycopy.norm() * eps_rel + sqrt(double(dim_n)) * eps_abs;
    }
    virtual void update_rho()
    {
        if(resid_primal > 10 * resid_dual)
            rho *= 2;
        else if(resid_dual > 10 * resid_primal)
            rho *= 0.5;
    }

    virtual VectorXd next_x() = 0;
    virtual VectorXd next_z() = 0;

public:
    ADMMBase(int p_, int n_, int m_,
             double eps_abs_ = 1e-8, double eps_rel_ = 1e-8,
             double rho_ = 1e-4) :
        constr_p(p_), dim_n(n_), dim_m(m_),
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

        VectorXd newx = next_x();
        main_x.swap(newx);
    }
    virtual void update_z()
    {
        VectorXd newz = next_z();
        VectorXd dual = newz - aux_z;
        B_mult(dual);
        At_mult(dual);
        resid_dual = rho * dual.norm();
        aux_z.swap(newz);
    }
    virtual void update_y()
    {
        VectorXd newr(constr_p);
        residual(newr, main_x, aux_z);
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