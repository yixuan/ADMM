#ifndef AVX_H
#define AVX_H

#include <Eigen/Core>
#include <Eigen/Sparse>

#ifdef __AVX__
#include <immintrin.h>

inline double inner_product_avx(const double *x, const double *y, int len)
{
    __m256d xx;
    __m256d yy;
    __m256d res = _mm256_setzero_pd();

    double r = 0.0;

    const char rem = (unsigned long)x % 32;
    const char head = rem ? (32 - rem) / sizeof(double) : 0;

    for(int i = 0; i < head; i++, x++, y++)
        r += (*x) * (*y);

    const int npack = (len - head) / 8;

    for(int i = 0; i < npack; i++, x += 8, y += 8)
    {
        xx = _mm256_load_pd(x);
        yy = _mm256_loadu_pd(y);
        res = _mm256_add_pd(res, _mm256_mul_pd(xx, yy));

        xx = _mm256_load_pd(x + 4);
        yy = _mm256_loadu_pd(y + 4);
        res = _mm256_add_pd(res, _mm256_mul_pd(xx, yy));
    }
    double *resp = (double*) &res;
    r += resp[0] + resp[1] + resp[2] + resp[3];

    for(int i = head + 8 * npack; i < len; i++, x++, y++)
        r += (*x) * (*y);

    return r;
}

void mat_spvec_product_avx(Eigen::VectorXd &res, const Eigen::Map<const Eigen::MatrixXd> &mat, const Eigen::SparseVector<double> &spvec)
{
    const int nrow = mat.rows();
    double *res_ptr = res.data();
    std::fill(res_ptr, res_ptr + nrow, double(0));

    const char rem = (unsigned long)res_ptr % 32;
    const char head = rem ? (32 - rem) / sizeof(double) : 0;
    const int npack = (nrow - head) / 8;
    __m256d mvec;
    __m256d vvec;
    __m256d cvec;

    const double *X0 = mat.data();
    const double *colptr;
    double *vptr;

    for(Eigen::SparseVector<double>::InnerIterator iter(spvec); iter; ++iter)
    {
        colptr = X0 + nrow * iter.index();
        vptr = res_ptr;

        const double val = iter.value();
        cvec = _mm256_set1_pd(val);

        for(int i = 0; i < head; i++, colptr++, vptr++)
            *vptr += *colptr * val;

        for(int i = 0; i < npack; i++, colptr += 8, vptr += 8)
        {
            mvec = _mm256_loadu_pd(colptr);
            mvec = _mm256_mul_pd(mvec, cvec);
            vvec = _mm256_load_pd(vptr);
            vvec = _mm256_add_pd(vvec, mvec);
            _mm256_store_pd(vptr, vvec);

            mvec = _mm256_loadu_pd(colptr + 4);
            mvec = _mm256_mul_pd(mvec, cvec);
            vvec = _mm256_load_pd(vptr + 4);
            vvec = _mm256_add_pd(vvec, mvec);
            _mm256_store_pd(vptr + 4, vvec);
        }
        for(int i = head + 8 * npack; i < nrow; i++, colptr++, vptr++)
            *vptr += *colptr * val;
     }
}
#endif // __AVX__



#endif // AVX_H
