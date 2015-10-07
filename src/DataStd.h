#ifndef DATASTD_H
#define DATASTD_H

#include <Eigen/Core>

#ifdef __AVX__
#include "Linalg/AVX.h"
#endif

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

    static Scalar sd_n(ConstGenericVector &v)
    {
#ifdef __AVX__
        Scalar s, ss;
        const int vsize = v.size();
        get_ss_avx<Scalar>(v.data(), v.size(), s, ss);
        s /= vsize;
        return std::sqrt(ss / vsize - s * s);
#else
        Scalar mean = v.mean();
        Vector v_centered = v.array() - mean;

        return v_centered.norm() / std::sqrt(Scalar(v.size()));
#endif
    }

    // spvec -> spvec / arr, elementwise
    static void elementwise_quot(SparseVector &spvec, Array &arr)
    {
        for(typename SparseVector::InnerIterator iter(spvec); iter; ++iter)
        {
            iter.valueRef() /= arr[iter.index()];
        }
    }

    // inner product of spvec and arr
    static Scalar sparse_inner_product(SparseVector &spvec, Array &arr)
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

    void standardize(Matrix &X, Vector &Y)
    {
        Scalar n_invsqrt = 1.0 / std::sqrt(Scalar(n));

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
                    X.col(i).array() *= (1.0 / scaleX[i]);
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
                    /*meanX[i] = X.col(i).mean();
                    X.col(i).array() -= meanX[i];
                    scaleX[i] = X.col(i).norm() * n_invsqrt;
                    X.col(i).array() /= scaleX[i];*/
    #ifdef __AVX__
                    Scalar *begin = &X(0, i);
                    Scalar s, ss;
                    get_ss_avx<Scalar>(begin, n, s, ss);
                    meanX[i] = s / n;
                    scaleX[i] = std::sqrt(ss - s * s / n) * n_invsqrt;
                    standardize_vec_avx<Scalar>(begin, n, meanX[i], 1.0 / scaleX[i]);
    #else
                    Scalar *begin = &X(0, i);
                    Scalar *end = begin + n;
                    meanX[i] = X.col(i).mean();
                    std::transform(begin, end, begin, std::bind2nd(std::minus<Scalar>(), meanX[i]));
                    scaleX[i] = X.col(i).norm() * n_invsqrt;
                    std::transform(begin, end, begin, std::bind2nd(std::multiplies<Scalar>(), 1.0 / scaleX[i]));
    #endif
                }
                break;
            default:
                break;
        }
    }

    void recover(Scalar &beta0, ArrayRef coef)
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

    void recover(Scalar &beta0, SparseVector &coef)
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
