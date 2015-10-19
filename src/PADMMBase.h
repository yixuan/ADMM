#ifndef PADMMBASE_H
#define PADMMBASE_H

#include <RcppEigen.h>
#include "Linalg/BlasWrapper.h"

// Parallel ADMM by splitting observations
//   minimize \sum loss(A_i * x - b_i) + r(x)
// Equivalent to
//   minimize \sum loss(A_i * x_i - b_i) + r(z)
//   s.t. x_i - z = 0
//
// x_i(p, 1), z(p, 1)
//
// main_x and dual_y are updated by workers, aux_z is updated by master
//
template<typename Scalar, typename VecTypeX, typename VecTypeZ>
class PADMMBase_Worker
{
protected:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Map<const Matrix> MapMat;
    typedef Eigen::Map<const Vector> MapVec;
    typedef const Eigen::Ref<const Matrix> ConstGenericMatrix;
    typedef const Eigen::Ref<const Vector> ConstGenericVector;

    Matrix subA;             // (sub) data matrix sent to this worker
    Vector subb;             // (sub) response vector sent to this worker
    const int dim_main;      // length of x_i

    VecTypeX main_x;         // parameters to be optimized
    const VecTypeZ &aux_z;   // z from master, read-only for workers
    Vector dual_y;           // Lagrangian multiplier
    double comp_squared_resid_primal;  // squared norm of primal residual on this worker
    double rho;              // augmented Lagrangian parameter

    // res = x in next iteration
    virtual void next_x(VecTypeX &res) = 0;
    // res = primal residual in next iteration
    virtual void next_residual(Vector &res) = 0;
    // action when rho is changed, e.g. re-factorize matrices
    virtual void rho_changed_action() {}

public:
    PADMMBase_Worker(ConstGenericMatrix &subA_, ConstGenericVector &subb_, VecTypeZ &aux_z_) :
        subA(subA_),
        subb(subb_),
        dim_main(subA.cols()),
        main_x(dim_main),
        aux_z(aux_z_),
        dual_y(dim_main)
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
        next_x(newx);
        main_x.swap(newx);
    }

    virtual void update_y()
    {
        Vector newr(dim_main);
        next_residual(newr);

        comp_squared_resid_primal = newr.squaredNorm();

        // dual_y.noalias() += rho * newr;
        Linalg::vec_add(dual_y.data(), Scalar(rho), newr.data(), dim_main);
    }

    virtual double squared_x_norm() { return main_x.squaredNorm(); }
    virtual double squared_y_norm() { return dual_y.squaredNorm(); }
    virtual double squared_resid_primal() { return comp_squared_resid_primal; }
};


template<typename Scalar, typename VecTypeX, typename VecTypeZ>
class PADMMBase_Master
{
protected:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

    const int dim_main;        // dimension of main_x
    const int dim_aux;         // dimension of aux_z
    const int dim_dual;        // dimension of dual_y
    const int n_comp;          // number of components in the objective function
    std::vector<PADMMBase_Worker<Scalar, VecTypeX, VecTypeZ> *> worker;  // each worker handles a component

    VecTypeZ aux_z;            // master maintains the update of z

    double rho;                // augmented Lagrangian parameter
    const double eps_abs;      // absolute tolerance
    const double eps_rel;      // relative tolerance

    double eps_primal;         // tolerance for primal residual
    double eps_dual;           // tolerance for dual residual

    double resid_primal;       // primal residual
    double resid_dual;         // dual residual

    // res = z_bar
    virtual void next_z(VecTypeZ &res) = 0;
    // action when rho is changed, e.g. re-factorize matrices
    virtual void rho_changed_action() {}

    // calculating eps_primal
    virtual double compute_eps_primal()
    {
        double x_norm_collector = 0.0;
        for(int i = 0; i < n_comp; i++)
        {
            x_norm_collector += worker[i]->squared_x_norm();
        }

        const double r = std::max(std::sqrt(x_norm_collector), aux_z.norm() * std::sqrt(n_comp));
        return r * eps_rel + std::sqrt(double(dim_dual * n_comp)) * eps_abs;
    }
    // calculating eps_dual
    virtual double compute_eps_dual()
    {
        double y_norm_collector = 0.0;
        for(int i = 0; i < n_comp; i++)
        {
            y_norm_collector += worker[i]->squared_y_norm();
        }

        return std::sqrt(y_norm_collector) * eps_rel + std::sqrt(double(dim_main * n_comp)) * eps_abs;
    }
    // calculating dual residual
    virtual double compute_resid_dual(const VecTypeZ &new_z)
    {
        VecTypeZ zdiff = new_z - aux_z;

        return rho * std::sqrt(n_comp) * zdiff.norm();
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
    PADMMBase_Master(int p_, int n_comp_, double eps_abs_ = 1e-6, double eps_rel_ = 1e-6) :
        dim_main(p_),
        dim_aux(p_),
        dim_dual(p_),
        n_comp(n_comp_),
        worker(n_comp_),
        aux_z(dim_aux),
        eps_abs(eps_abs_), eps_rel(eps_rel_)
    {}

    virtual ~PADMMBase_Master() {}

    void update_x()
    {
        eps_primal = compute_eps_primal();
        eps_dual = compute_eps_dual();
        update_rho();

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for(int i = 0; i < n_comp; i++)
        {
            worker[i]->update_rho(rho);
            worker[i]->update_x();
        }
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
        double resid_primal_collector = 0.0;

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static) reduction(+:resid_primal_collector)
        #endif
        for(int i = 0; i < n_comp; i++)
        {
            worker[i]->update_y();
            resid_primal_collector += worker[i]->squared_resid_primal();
        }

        resid_primal = std::sqrt(resid_primal_collector);
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
            update_x();
            update_z();
            update_y();

            if(converged())
                break;
        }

        return i + 1;
    }

    virtual VecTypeZ get_z() { return aux_z; }
};


#endif // PADMMBASE_H
