#ifndef DATASTD_H
#define DATASTD_H

#include <Eigen/Core>

template <typename Scalar = double>
class DataStd
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Array <Scalar, Eigen::Dynamic, 1> Array;
    typedef const Eigen::Ref<const Vector> ConstGenericVector;
    typedef Eigen::Ref<Array> ArrayRef;
    typedef Eigen::SparseVector<Scalar> SparseVector;

    // flag - 0: standardize = FALSE, intercept = FALSE
    //             directly fit model
    // flag - 1: standardize = TRUE, intercept = FALSE
    //             scale x and y by their standard deviation
    // flag - 2: standardize = FALSE, intercept = TRUE
    //             center x, standardize y
    // flag - 3: standardize = TRUE, intercept = TRUE
    //             standardize x and y
    const int flag;

    const int n;
    const int p;

    Scalar meanY;
    Scalar scaleY;
    Array  meanX;
    Array  scaleX;

    // Standard deviation using n as the denominator (population standard deviation)
    static Scalar sd_n(ConstGenericVector& v)
    {
        const Scalar mean = v.mean();
        const Scalar norm = (v.array() - mean).matrix().norm();
        return norm / std::sqrt(Scalar(v.size()));
    }

    // Standard deviation with weights. The sum of w is assumed to be 1
    // mean = sum w_i * x_i, variance = sum w_i * (x_i - mean)^2
    static Scalar sd_n(ConstGenericVector& v, const Array& w)
    {
        const Scalar mean = v.dot(w.matrix());
        const Scalar variance = ((v.array() - mean).square() * w).sum();
        return std::sqrt(variance);
    }

    // spvec -> spvec / arr, elementwise
    static void elementwise_quot(SparseVector& spvec, const Array& arr)
    {
        for(typename SparseVector::InnerIterator iter(spvec); iter; ++iter)
        {
            iter.valueRef() /= arr[iter.index()];
        }
    }

    // inner product of spvec and arr
    static Scalar sparse_inner_product(SparseVector& spvec, const Array& arr)
    {
        Scalar res = 0.0;
        for(typename SparseVector::InnerIterator iter(spvec); iter; ++iter)
        {
            res += iter.value() * arr[iter.index()];
        }
        return res;
    }

public:
    DataStd(int n_, int p_, bool standardize, bool intercept) :
        flag(int(standardize) + 2 * int(intercept)),
        n(n_),
        p(p_),
        meanY(0.0),
        scaleY(1.0)
    {
        if(flag == 3 || flag == 2)
            meanX.resize(p);
        if(flag == 3 || flag == 1)
            scaleX.resize(p);
    }

    // flag - 0: standardize = FALSE, intercept = FALSE
    //             directly fit model
    // flag - 1: standardize = TRUE, intercept = FALSE
    //             scale x and y by their standard deviation
    // flag - 2: standardize = FALSE, intercept = TRUE
    //             center x, standardize y
    // flag - 3: standardize = TRUE, intercept = TRUE
    //             standardize x and y
    void standardize(Matrix& X, Vector& Y)
    {
        const Scalar n_invsqrt = Scalar(1.0) / std::sqrt(Scalar(n));

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
                scaleY = Y.norm() * n_invsqrt;
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
                    X.col(i).array() *= (Scalar(1.0) / scaleX[i]);
                }
                break;
            case 2:
                for(int i = 0; i < p; i++)
                {
                    meanX[i] = X.col(i).mean();
                    X.col(i).array() -= meanX[i];
                }
                break;
            case 3:
                for(int i = 0; i < p; i++)
                {
                    meanX[i] = X.col(i).mean();
                    X.col(i).array() -= meanX[i];
                    scaleX[i] = X.col(i).norm() * n_invsqrt;
                    X.col(i).array() *= (Scalar(1.0) / scaleX[i]);
                }
                break;
            default:
                break;
        }
    }

    // Standardize with weights
    void standardize(Matrix& X, Vector& Y, Array& W)
    {
        // Make sure the weights sum up to 1
        W /= W.sum();
        Array sqrt_W = W.sqrt() * std::sqrt(W.size());
        // standardize Y
        switch(flag)
        {
            case 1:
                scaleY = sd_n(Y, W);
                Y.array() /= scaleY;
                Y.array() *= sqrt_W.array();
                break;
            case 2:
            case 3:
                meanY = Y.dot(W.matrix());
                Y.array() -= meanY;
                scaleY = std::sqrt((Y.array().square() * W).sum());
                Y.array() /= scaleY;
                Y.array() *= sqrt_W.array();
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
                    scaleX[i] = sd_n(X.col(i), W);
                    X.col(i).array() *= (1.0 / scaleX[i]);
                    X.col(i).array() *=  sqrt_W.array();
                }
                break;
            case 2:
                for(int i = 0; i < p; i++)
                {
                    meanX[i] = (X.col(i).array() * W).sum();
                    X.col(i).array() -= meanX[i];
                    X.col(i).array() *= sqrt_W.array();
                }
                break;
            case 3:
                for(int i = 0; i < p; i++)
                {
                    meanX[i] = (X.col(i).array() * W).sum();
                    X.col(i).array() -= meanX[i];
                    scaleX[i] = std::sqrt((X.col(i).array().square() * W).sum());
                    X.col(i).array() *= (1.0 / scaleX[i]);
                    X.col(i).array() *=  sqrt_W.array();
                }
                break;
            default:
                break;
        }
    }

    void recover(Scalar& beta0, ArrayRef coef)
    {
        switch(flag)
        {
            case 0:
                beta0 = Scalar(0);
                break;
            case 1:
                beta0 = Scalar(0);
                coef /= scaleX;
                coef *= scaleY;
                break;
            case 2:
                // problem for scaleY? case 2
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

    void recover(Scalar& beta0, SparseVector& coef)
    {
        switch(flag)
        {
            case 0:
                beta0 = 0;
                break;
            case 1:
                beta0 = 0;
                elementwise_quot(coef, scaleX);
                coef *= scaleY;
                break;
            case 2:
                coef *= scaleY;
                beta0 = meanY - sparse_inner_product(coef, meanX);
                break;
            case 3:
                elementwise_quot(coef, scaleX);
                coef *= scaleY;
                beta0 = meanY - sparse_inner_product(coef, meanX);
                break;
            default:
                break;
        }
    }

    Scalar get_scaleY() { return scaleY; }
};



#endif // DATASTD_H
