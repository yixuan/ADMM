#ifndef ADMMBASE_H
#define ADMMBASE_H

#include <RcppEigen.h>
#include "Linalg/BlasWrapper.h"

// General problem setting
//   minimize f(x) + g(z)
//   s.t. Ax + Bz = c
//
// x(n, 1), z(m, 1), A(p, n), B(p, m), c(p, 1)
//
template<typename VecTypeX, typename VecTypeZ, typename VecTypeY>
class ADMMBase
{
protected:
    const int dim_main;   // dimension of x
    const int dim_aux;    // dimension of z
    const int dim_dual;   // dimension of Ax + Bz - c

    VecTypeX main_x;      // parameters to be optimized
    VecTypeZ aux_z;       // auxiliary parameters
    VecTypeY dual_y;      // Lagrangian multiplier

    double rho;           // augmented Lagrangian parameter
    const double eps_abs; // absolute tolerance
    const double eps_rel; // relative tolerance

    double eps_primal;    // tolerance for primal residual
    double eps_dual;      // tolerance for dual residual

    double resid_primal;  // primal residual
    double resid_dual;    // dual residual

    virtual void A_mult (VecTypeY &res, VecTypeX &x) = 0;   // operation res -> Ax, x can be overwritten
    virtual void At_mult(VecTypeY &res, VecTypeY &y) = 0;   // operation res -> A'y, y can be overwritten
    virtual void B_mult (VecTypeY &res, VecTypeZ &z) = 0;   // operation res -> Bz, z can be overwritten
    virtual double c_norm() = 0;                            // L2 norm of c

    // res = Ax + Bz - c
    virtual void next_residual(VecTypeY &res) = 0;
    // res = x in next iteration
    virtual void next_x(VecTypeX &res) = 0;
    // res = z in next iteration
    virtual void next_z(VecTypeZ &res) = 0;
    // action when rho is changed, e.g. re-factorize matrices
    virtual void rho_changed_action() {}

    // calculating eps_primal
    // eps_primal = sqrt(p) * eps_abs + eps_rel * max(||Ax||, ||Bz||, ||c||)
    virtual double compute_eps_primal()
    {
        VecTypeY xres, zres;
        VecTypeX xcopy = main_x;
        VecTypeZ zcopy = aux_z;
        A_mult(xres, xcopy);
        B_mult(zres, zcopy);
        double r = std::max(xres.norm(), zres.norm());
        r = std::max(r, c_norm());
        return r * eps_rel + std::sqrt(double(dim_dual)) * eps_abs;
    }
    // calculating eps_dual
    // eps_dual = sqrt(n) * eps_abs + eps_rel * ||A'y||
    virtual double compute_eps_dual()
    {
        VecTypeY yres, ycopy = dual_y;
        At_mult(yres, ycopy);

        return yres.norm() * eps_rel + std::sqrt(double(dim_main)) * eps_abs;
    }
    // calculating dual residual
    // resid_dual = rho * A'B(auxz - oldz)
    virtual double compute_resid_dual(const VecTypeZ &new_z)
    {
        VecTypeZ zdiff = new_z - aux_z;
        VecTypeY tmp;
        B_mult(tmp, zdiff);

        VecTypeY dual;
        At_mult(dual, tmp);

        return rho * dual.norm();
    }
    // increase or decrease rho in iterations
    virtual void update_rho()
    {
        if(resid_primal / eps_primal > 10 * resid_dual / eps_dual)
        {
            rho *= 2;
            rho_changed_action();
        }
        else if(resid_dual / eps_dual > 10 * resid_primal / eps_primal)
        {
            rho /= 2;
            rho_changed_action();
        }

        if(resid_primal < eps_primal)
        {
            rho /= 1.2;
            rho_changed_action();
        }
        
        if(resid_dual < eps_dual)
        {
            rho *= 1.2;
            rho_changed_action();
        }
    }
    // Debugging residual information
    void print_header(std::string title)
    {
        const int width = 80;
        const char sep = ' ';

        Rcpp::Rcout << std::endl << std::string(width, '=') << std::endl;
        Rcpp::Rcout << std::string((width - title.length()) / 2, ' ') << title << std::endl;
        Rcpp::Rcout << std::string(width, '-') << std::endl;

        Rcpp::Rcout << std::left << std::setw(7)  << std::setfill(sep) << "iter";
        Rcpp::Rcout << std::left << std::setw(13) << std::setfill(sep) << "eps_primal";
        Rcpp::Rcout << std::left << std::setw(13) << std::setfill(sep) << "resid_primal";
        Rcpp::Rcout << std::left << std::setw(13) << std::setfill(sep) << "eps_dual";
        Rcpp::Rcout << std::left << std::setw(13) << std::setfill(sep) << "resid_dual";
        Rcpp::Rcout << std::left << std::setw(13) << std::setfill(sep) << "rho";
        Rcpp::Rcout << std::endl;

        Rcpp::Rcout << std::string(width, '-') << std::endl;
    }
    void print_row(int iter)
    {
        const char sep = ' ';

        Rcpp::Rcout << std::left << std::setw(7)  << std::setfill(sep) << iter;
        Rcpp::Rcout << std::left << std::setw(13) << std::setfill(sep) << eps_primal;
        Rcpp::Rcout << std::left << std::setw(13) << std::setfill(sep) << resid_primal;
        Rcpp::Rcout << std::left << std::setw(13) << std::setfill(sep) << eps_dual;
        Rcpp::Rcout << std::left << std::setw(13) << std::setfill(sep) << resid_dual;
        Rcpp::Rcout << std::left << std::setw(13) << std::setfill(sep) << rho;
        Rcpp::Rcout << std::endl;
    }
    void print_footer()
    {
        const int width = 80;
        Rcpp::Rcout << std::string(width, '=') << std::endl << std::endl;
    }

public:
    ADMMBase(int n_, int m_, int p_,
             double eps_abs_ = 1e-6, double eps_rel_ = 1e-6) :
        dim_main(n_), dim_aux(m_), dim_dual(p_),
        main_x(n_), aux_z(m_), dual_y(p_),  // allocate space but do not set values
        eps_abs(eps_abs_), eps_rel(eps_rel_)
    {}

    virtual ~ADMMBase() {}

    void update_x()
    {
        eps_primal = compute_eps_primal();
        eps_dual = compute_eps_dual();

        VecTypeX newx(dim_main);
        next_x(newx);
        main_x.swap(newx);
    }
    void update_z()
    {
        VecTypeZ newz(dim_aux);
        next_z(newz);

        resid_dual = compute_resid_dual(newz);

        aux_z.swap(newz);
    }
    void update_y()
    {
        VecTypeY newr(dim_dual);
        next_residual(newr);

        resid_primal = newr.norm();

        // dual_y.noalias() += rho * newr;
        Linalg::vec_add(dual_y.data(), typename VecTypeY::RealScalar(rho), newr.data(), dim_dual);
    }

    bool converged()
    {
        return (resid_primal < eps_primal) &&
               (resid_dual < eps_dual);
    }

    int solve(int maxit)
    {
        int i;

        // print_header("ADMM iterations");

        for(i = 0; i < maxit; i++)
        {
            update_x();
            update_z();
            update_y();

            // print_row(i);

            if(converged())
                break;

            if(i > 3)
                update_rho();
        }

        // print_footer();

        return i + 1;
    }

    virtual VecTypeX get_x() { return main_x; }
    virtual VecTypeZ get_z() { return aux_z; }
    virtual VecTypeY get_y() { return dual_y; }
};



#endif // ADMMBASE_H
