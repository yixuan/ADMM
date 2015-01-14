#ifndef ADMMLASSO_H
#define ADMMLASSO_H

#include "ADMMBase.h"

// minimize  1/2 * ||y - X * beta||^2 + lambda * ||beta||_1
//
// In ADMM form,
//   minimize f(x) + g(z)
//   s.t. x - z = 0
//
// f(beta) = 1/2 * ||y - X * beta||^2
// g(z) = lambda * ||z||_1
class ADMMLasso: public ADMMBase
{
private:
    typedef Eigen::MatrixXd MatrixXd;
    typedef Eigen::ArrayXd ArrayXd;
    typedef Eigen::LLT<MatrixXd> LLT;

    const MatrixXd *datX;         // data matrix
    double lambda;                // L1 penalty
    const bool thinX;             // whether nrow(X) > ncol(X)

    const VectorXd cache_XY;      // cache X'y
    MatrixXd cache_XX;            // cache X'X if thinX = true,
                                  // or XX' if thinX = false
    ArrayXd cache_XXdiag;         // diagonal elments of cache_XX
    LLT solver;                   // matrix factorization
    
    virtual void A_mult(VectorXd &x) {}  // x -> x
    virtual void At_mult(VectorXd &x) {} // x -> x
    virtual void B_mult(VectorXd &x) {}  // x -> x
    virtual double c_norm() { return 0.0; }  // ||c||_2 = 0
    virtual void next_residual(VectorXd &res, const VectorXd &x, const VectorXd &z)
    {
        res.noalias() = x - z;
    }
    
    virtual void next_x(VectorXd &res)
    {
        // For a thin X,
        //   rhs = X'y + rho * aux_z - dual_y
        //   newx = inv(X'X + rho * I) * rhs
        //
        // For a wide X,
        //   inv(X'X + rho * I) = 1/rho * I -
        //       1/rho * X' * inv(XX' + rho * I) * X
        // so
        //   newx = 1/rho * rhs - 1/rho * X' * inv(XX' + rho * I) * X * rhs
        
        VectorXd rhs = cache_XY + rho * aux_z - dual_y;
        if(thinX)
        {
            res.noalias() = solver.solve(rhs);
        } else {
            res = rhs;
            res.noalias() -= (*datX).transpose() * solver.solve((*datX) * rhs);
            res /= rho;
        }
    }
    virtual void next_z(VectorXd &res)
    {
        res.noalias() = main_x + dual_y / rho;
        soft_threshold(res, lambda / rho);
    }
    virtual void rho_changed_action()
    {
        cache_XX.diagonal() = cache_XXdiag + rho;
        solver.compute(cache_XX);
    }
    
public:
    ADMMLasso(const MatrixXd &datX_, const VectorXd &datY_,
              double eps_abs_ = 1e-6,
              double eps_rel_ = 1e-6) :
        ADMMBase(datX_.cols(), datX_.cols(), datX_.cols(),
                 eps_abs_, eps_rel_),
        datX(&datX_), thinX(datX_.rows() > datX_.cols()),
        cache_XY(datX_.transpose() * datY_)
    {
        if(thinX)
            cache_XX = datX_.transpose() * datX_;
        else
            cache_XX = datX_ * datX_.transpose();
        
        cache_XXdiag = cache_XX.diagonal();
        rho_changed_action();
    }

    // init() needs to be called every time we want to solve
    // for a new lambda
    virtual void init(double lambda_, double rho_)
    {
        main_x.setZero();
        aux_z.setZero();
        dual_y.setZero();
        lambda = lambda_;
        rho = rho_;
        eps_primal = 0.0;
        eps_dual = 0.0;
        resid_primal = 9999;
        resid_dual = 9999;
    }
    // provide initial values
    virtual void init(double lambda_, double rho_, const Ref &init_x)
    {
        main_x = init_x;
        aux_z = init_x;
        dual_y.setZero();
        lambda = lambda_;
        rho = rho_;
        eps_primal = 0.0;
        eps_dual = 0.0;
        resid_primal = 9999;
        resid_dual = 9999;
    }

    static void soft_threshold(VectorXd &vec, const double &penalty)
    {
        double *ptr = vec.data();
        for(int i = 0; i < vec.size(); i++)
        {
            if(ptr[i] > penalty)
                ptr[i] -= penalty;
            else if(ptr[i] < -penalty)
                ptr[i] += penalty;
            else
                ptr[i] = 0;
        }
    }
};



inline double sd_n(const Eigen::Ref<Eigen::VectorXd> &v)
{
    double mean = v.mean();
    double s = 0.0, tmp = 0.0;
    int n = v.size();
    for(int i = 0; i < n; i++)
    {
        tmp = v[i] - mean;
        s += tmp * tmp;
    }
    s /= n;
    return sqrt(s);
}

class DataStd
{
private:
    typedef Eigen::MatrixXd MatrixXd;
    typedef Eigen::VectorXd VectorXd;
    typedef Eigen::ArrayXd ArrayXd;
    typedef Eigen::Ref<ArrayXd> Ref;

    // flag - 0: standardize = FALSE, intercept = FALSE
    //             directly fit model
    // flag - 1: standardize = TRUE, intercept = FALSE
    //             scale x and y by their standard deviation
    // flag - 2: standardize = FALSE, intercept = TRUE
    //             center x, standardize y
    // flag - 3: standardize = TRUE, intercept = TRUE
    //             standardize x and y
    int flag;

    int n;
    int p;

    double meanY;
    double scaleY;
    ArrayXd meanX;
    ArrayXd scaleX;
public:
    DataStd(int n, int p, bool standardize, bool intercept)
    {
        this->n = n;
        this->p = p;
        this->flag = int(standardize) + 2 * int(intercept);
        this->meanY = 0.0;
        this->scaleY = 1.0;
        
        switch(flag)
        {
            case 1:
                scaleX.resize(p);
                break;
            case 2:
                meanX.resize(p);
                break;
            case 3:
                meanX.resize(p);
                scaleX.resize(p);
                break;
            default:
                break;
        }
    }

    void standardize(MatrixXd &X, VectorXd &Y)
    {
        // standardize Y
        switch(flag)
        {
            case 1:
                scaleY = sd_n(Y);
                Y.array() /= scaleY;
                break;
            case 2:
            case 3:
                meanY = Y.mean();
                Y.array() -= meanY;
                scaleY = Y.norm() / sqrt(double(n));
                Y.array() /= scaleY;
                break;
            default:
                break;
        }

        // standardize X
        switch(flag)
        {
            case 1:
                for(int i = 0; i < p; i++)
                {
                    scaleX[i] = sd_n(X.col(i));
                    X.col(i).array() /= scaleX[i];
                }
                break;
            case 2:
                meanX = X.colwise().mean();
                for(int i = 0; i < p; i++)
                {
                    X.col(i).array() -= meanX[i];
                }
                break;
            case 3:
                meanX = X.colwise().mean();
                for(int i = 0; i < p; i++)
                {
                    X.col(i).array() -= meanX[i];
                }
                scaleX = X.colwise().norm() / sqrt(double(n));
                for(int i = 0; i < p; i++)
                {
                    X.col(i).array() /= scaleX[i];
                }
                break;
            default:
                break;
        }
    }

    void recover(double &beta0, Ref coef)
    {
        switch(flag)
        {
            case 0:
                beta0 = 0;
                break;
            case 1:
                beta0 = 0;
                coef /= scaleX;
                coef *= scaleY;
                break;
            case 2:
                coef *= scaleY;
                beta0 = meanY - (coef * meanX).sum();
                break;
            case 3:
                coef /= scaleX;
                coef *= scaleY;
                beta0 = meanY - (coef * meanX).sum();
                break;
            default:
                break;
        }
    }

    double get_scaleY() { return scaleY; }
};

#endif // ADMMLASSO_H
