#ifndef FADMMBASE_H
#define FADMMBASE_H

#include <RcppEigen.h>

// General problem setting
//   minimize f(x) + g(z)
//   s.t. Ax + Bz = c
//
// x(n, 1), z(m, 1), A(p, n), B(p, m), c(p, 1)
//
template<typename VecTypeX, typename VecTypeZ>
class FADMMBase
{
protected:
    typedef Eigen::VectorXd VectorXd;

    const int dim_main;   // dimension of x
    const int dim_aux;    // dimension of z
    const int dim_dual;   // dimension of Ax + Bz - c

    VecTypeX main_x;      // parameters to be optimized
    VecTypeZ aux_z;       // auxiliary parameters
    VectorXd dual_y;      // Lagrangian multiplier

    VecTypeZ adj_z;       // adjusted z vector, used for acceleration
    VectorXd adj_y;       // adjusted y vector, used for acceleration
    VecTypeZ old_z;       // z vector in the previous iteration, used for acceleration
    VectorXd old_y;       // y vector in the previous iteration, used for acceleration
    double adj_a;         // coefficient used for acceleration
    double adj_c;         // coefficient used for acceleration

    double rho;           // augmented Lagrangian parameter
    const double eps_abs; // absolute tolerance
    const double eps_rel; // relative tolerance

    double eps_primal;    // tolerance for primal residual
    double eps_dual;      // tolerance for dual residual

    double resid_primal;  // primal residual
    double resid_dual;    // dual residual

    virtual void A_mult (VectorXd &res, VecTypeX &x) = 0;   // operation res -> Ax, x can be overwritten
    virtual void At_mult(VectorXd &res, VectorXd &y) = 0;   // operation res -> A'y, y can be overwritten
    virtual void B_mult (VectorXd &res, VecTypeZ &z) = 0;   // operation res -> Bz, z can be overwritten
    virtual double c_norm() = 0;                            // L2 norm of c

    // res = Ax + Bz - c
    virtual void next_residual(VectorXd &res) = 0;
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
        VectorXd xres, zres;
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
        VectorXd yres, ycopy = dual_y;

        At_mult(yres, ycopy);

        return yres.norm() * eps_rel + std::sqrt(double(dim_main)) * eps_abs;
    }
    // calculating dual residual
    // resid_dual = rho * A'B(auxz - oldz)
    virtual double compute_resid_dual()
    {
        VecTypeZ zdiff = aux_z - old_z;
        VectorXd tmp;
        B_mult(tmp, zdiff);

        VectorXd dual;
        At_mult(dual, tmp);

        return rho * dual.norm();
    }
    // calculating combined residual
    // resid_combined = rho * ||resid_primal||^2 + rho * ||auxz - adjz||^2
    virtual double compute_resid_combined()
    {
        VecTypeZ tmp = aux_z - adj_z;
        VectorXd tmp2;
        B_mult(tmp2, tmp);

        return rho * resid_primal * resid_primal + rho * tmp2.squaredNorm();
    }
    // increase or decrease rho in iterations
    virtual void update_rho()
    {
        /* if(resid_primal > 10 * resid_dual)
        {
            rho *= 2;
            rho_changed_action();
        }
        else if(resid_dual > 10 * resid_primal)
        {
            rho *= 0.5;
            rho_changed_action();
        } */
    }

public:
    FADMMBase(int n_, int m_, int p_,
              double eps_abs_ = 1e-6, double eps_rel_ = 1e-6) :
        dim_main(n_), dim_aux(m_), dim_dual(p_),
        main_x(n_), aux_z(m_), dual_y(p_),  // allocate space but do not set values
        adj_z(m_), adj_y(p_), adj_a(1.0), adj_c(9999),
        eps_abs(eps_abs_), eps_rel(eps_rel_)
    {}

    void update_x()
    {
        eps_primal = compute_eps_primal();
        eps_dual = compute_eps_dual();
        update_rho();

        VecTypeX newx(dim_main);
        next_x(newx);

        main_x.swap(newx);
    }
    void update_z()
    {
        VecTypeZ newz(dim_aux);
        next_z(newz);
        aux_z.swap(newz);

        resid_dual = compute_resid_dual();
    }
    void update_y()
    {
        VectorXd newr(dim_dual);
        next_residual(newr);

        resid_primal = newr.norm();

        // dual_y.noalias() = adj_y + rho * newr;
        std::transform(newr.data(), newr.data() + dim_dual, newr.data(), std::bind2nd(std::multiplies<double>(), rho));
        std::transform(adj_y.data(), adj_y.data() + dim_dual, newr.data(), dual_y.data(), std::plus<double>());
    }

    bool converged()
    {
        return (resid_primal < eps_primal) &&
               (resid_dual < eps_dual);
    }

    int solve(int maxit)
    {
        int i;

        for(i = 0; i < maxit; i++)
        {
            old_z = aux_z;
            old_y = dual_y;

            update_x();
            update_z();
            update_y();

            if(converged())
                break;

            double old_c = adj_c;
            adj_c = compute_resid_combined();

            // debug_info();

            if(adj_c < 0.999 * old_c)
            {
                double old_a = adj_a;
                adj_a = 0.5 + 0.5 * std::sqrt(1 + 4.0 * old_a * old_a);
                double ratio = (old_a - 1.0) / adj_a;
                adj_z = (1 + ratio) * aux_z - ratio * old_z;
                adj_y = (1 + ratio) * dual_y - ratio * old_y;
            } else {
                adj_a = 1.0;
                adj_z = old_z;
                adj_y = old_y;
                adj_c = old_c / 0.999;
            }
        }

        return i + 1;
    }

    virtual VecTypeX get_x() { return main_x; }
    virtual VecTypeZ get_z() { return aux_z; }
    virtual VectorXd get_y() { return dual_y; }
};



#endif // FADMMBASE_H
