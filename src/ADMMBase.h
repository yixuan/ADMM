#ifndef ADMMBASE_H
#define ADMMBASE_H

#include <RcppEigen.h>
// #include <ctime>

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
    virtual void next_residual(VectorXd &res) = 0;
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
    ADMMBase(int n_, int m_, int p_,
             double eps_abs_ = 1e-6, double eps_rel_ = 1e-6) :
        dim_main(n_), dim_aux(m_), dim_dual(p_),
        main_x(n_), aux_z(m_), dual_y(p_),  // allocate space but do not set values
        eps_abs(eps_abs_), eps_rel(eps_rel_)
    {}

    virtual void update_x()
    {
        eps_primal = compute_eps_primal();
        eps_dual = compute_eps_dual();
        update_rho();

        VectorXd newx(dim_main);
        next_x(newx);
        main_x.swap(newx);
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
        next_residual(newr);

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
        //Rcpp::Rcout << "time - x: " << tx << " secs\n";
        //Rcpp::Rcout << "time - z: " << tz << " secs\n";
        //Rcpp::Rcout << "time - y: " << ty << " secs\n";
        //Rcpp::Rcout << "time - x + y + z: " << tx + ty + tz << " secs\n";
        return i + 1;
    }

    virtual VectorXd get_x() { return main_x; }
    virtual VectorXd get_z() { return aux_z; }
    virtual VectorXd get_y() { return dual_y; }
};



#endif // ADMMBASE_H
