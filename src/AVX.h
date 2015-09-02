#ifndef AVX_H
#define AVX_H

#include <Eigen/Core>
#include <Eigen/Sparse>

#ifdef __AVX__
#include <immintrin.h>

__m256d *load_mat_avx(const double *mat, const int nrow, const int ncol, int &nrowx)
{
    const int npack = nrow / 4;
    const int tail = nrow - npack * 4;
    nrowx = npack + int(tail != 0);
    double rem[4] = {0, 0, 0, 0};

    __m256d *loaded, *ret;
    posix_memalign((void **)&loaded, 32, nrowx * ncol * sizeof(__m256d));
    ret = loaded;

    const double *colptr;
    for(int i = 0; i < ncol; i++)
    {
        colptr = mat + nrow * i;
        for(int j = 0; j < npack; j++, colptr += 4, loaded++)
            *loaded = _mm256_loadu_pd(colptr);
        if(tail != 0)
        {
            for(int j = 0; j < tail; j++, colptr++)
                rem[j] = *colptr;

             *loaded = _mm256_loadu_pd(rem);
             loaded++;
        }
    }

    return ret;
}

__m256d *load_vec_avx(const double *vec, const int len, int &lenx)
{
    const int npack = len / 4;
    const int tail = len - npack * 4;
    lenx = npack + int(tail != 0);
    double rem[4] = {0, 0, 0, 0};

    __m256d *loaded, *ret;
    posix_memalign((void **)&loaded, 32, lenx * sizeof(__m256d));
    ret = loaded;

    const double *ptr = vec;
    for(int j = 0; j < npack; j++, ptr += 4, loaded++)
        *loaded = _mm256_loadu_pd(ptr);
    if(tail != 0)
    {
        for(int j = 0; j < tail; j++, ptr++)
            rem[j] = *ptr;

         *loaded = _mm256_loadu_pd(rem);
    }

    return ret;
}

inline void get_ss_avx(const double *x, const int len, double &sum, double &sum_of_square)
{
    __m256d xx;
    __m256d s = _mm256_setzero_pd();
    __m256d ss = _mm256_setzero_pd();

    sum = 0.0;
    sum_of_square = 0.0;

    const char rem = (unsigned long)x % 32;
    const char head = rem ? (32 - rem) / sizeof(double) : 0;
    for(int i = 0; i < head; i++, x++)
    {
        sum += *x;
        sum_of_square += (*x) * (*x);
    }

    const int npack = (len - head) / 8;
    for(int i = 0; i < npack; i++, x += 8)
    {
        xx = _mm256_load_pd(x);
        s = _mm256_add_pd(s, xx);
        ss = _mm256_add_pd(ss, _mm256_mul_pd(xx, xx));

        xx = _mm256_load_pd(x + 4);
        s = _mm256_add_pd(s, xx);
        ss = _mm256_add_pd(ss, _mm256_mul_pd(xx, xx));
    }
    double *resp = (double *) &s;
    sum += resp[0] + resp[1] + resp[2] + resp[3];
    resp = (double *) &ss;
    sum_of_square += resp[0] + resp[1] + resp[2] + resp[3];

    for(int i = head + 8 * npack; i < len; i++, x++)
    {
        sum += *x;
        sum_of_square += (*x) * (*x);
    }
}

// (x - center) * scale
inline void standardize_vec_avx(double *x, const int len, const double center, const double scale)
{
    __m256d xx;
    __m256d cc = _mm256_set1_pd(center);
    __m256d ss = _mm256_set1_pd(scale);

    const char rem = (unsigned long)x % 32;
    const char head = rem ? (32 - rem) / sizeof(double) : 0;
    for(int i = 0; i < head; i++, x++)
        *x = (*x - center) * scale;

    const int npack = (len - head) / 8;
    for(int i = 0; i < npack; i++, x += 8)
    {
        xx = _mm256_load_pd(x);
        xx = _mm256_mul_pd(_mm256_sub_pd(xx, cc), ss);
        _mm256_store_pd(x, xx);

        xx = _mm256_load_pd(x + 4);
        xx = _mm256_mul_pd(_mm256_sub_pd(xx, cc), ss);
        _mm256_store_pd(x + 4, xx);
    }

    for(int i = head + 8 * npack; i < len; i++, x++)
        *x = (*x - center) * scale;
}

inline double loaded_inner_product_avx(const __m256d *x, const __m256d *y, const int lenx)
{
    __m256d res = _mm256_setzero_pd();

    const int npack = lenx / 8;
    const int mainlen = npack * 8;

    for(int i = 0; i < npack; i++, x += 8, y += 8)
    {
        res = _mm256_add_pd(res, _mm256_mul_pd(x[0], y[0]));
        res = _mm256_add_pd(res, _mm256_mul_pd(x[1], y[1]));
        res = _mm256_add_pd(res, _mm256_mul_pd(x[2], y[2]));
        res = _mm256_add_pd(res, _mm256_mul_pd(x[3], y[3]));
        res = _mm256_add_pd(res, _mm256_mul_pd(x[4], y[4]));
        res = _mm256_add_pd(res, _mm256_mul_pd(x[5], y[5]));
        res = _mm256_add_pd(res, _mm256_mul_pd(x[6], y[6]));
        res = _mm256_add_pd(res, _mm256_mul_pd(x[7], y[7]));
    }

    for(int i = mainlen; i < lenx; i++, x++, y++)
        res = _mm256_add_pd(res, _mm256_mul_pd(*x, *y));

    res = _mm256_hadd_pd(res, res);
    double *resp = (double*) &res;
    return resp[0] + resp[2];
}

inline double inner_product_avx(const double *x, const double *y, const int len)
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

void loaded_mat_spvec_prod_avx(double *res, const int len,
                               const __m256d *mat, const int nrowx, const int ncolx,
                               const Eigen::SparseVector<double> &spvec)
{
    // Create an array of __m256d to store the result
    __m256d *r;
    posix_memalign((void **)&r, 32, nrowx * sizeof(__m256d));
    for(int i = 0; i < nrowx; i++)
        r[i] = _mm256_setzero_pd();

    __m256d c;  // Constant 4-vector
    const __m256d *colptr;  // Pointing to columns of mat
    __m256d *vptr;  // Pointing to elements of r

    int npack = nrowx / 8;
    int mainlen = npack * 8;

    for(Eigen::SparseVector<double>::InnerIterator iter(spvec); iter; ++iter)
    {
        colptr = mat + nrowx * iter.index();
        vptr = r;
        c = _mm256_set1_pd(iter.value());

        for(int i = 0; i < npack; i++, colptr += 8, vptr += 8)
        {
            vptr[0] = _mm256_add_pd(vptr[0], _mm256_mul_pd(colptr[0], c));
            vptr[1] = _mm256_add_pd(vptr[1], _mm256_mul_pd(colptr[1], c));
            vptr[2] = _mm256_add_pd(vptr[2], _mm256_mul_pd(colptr[2], c));
            vptr[3] = _mm256_add_pd(vptr[3], _mm256_mul_pd(colptr[3], c));
            vptr[4] = _mm256_add_pd(vptr[4], _mm256_mul_pd(colptr[4], c));
            vptr[5] = _mm256_add_pd(vptr[5], _mm256_mul_pd(colptr[5], c));
            vptr[6] = _mm256_add_pd(vptr[6], _mm256_mul_pd(colptr[6], c));
            vptr[7] = _mm256_add_pd(vptr[7], _mm256_mul_pd(colptr[7], c));
        }
        for(int i = mainlen; i < nrowx; i++, colptr++, vptr++)
            *vptr = _mm256_add_pd(*vptr, _mm256_mul_pd(*colptr, c));
     }

    npack = len / 4;
    mainlen = npack * 4;

    for(int i = 0; i < npack; i++, res += 4)
        _mm256_storeu_pd(res, r[i]);
    if(nrowx > npack)
    {
        double rem[4];
        _mm256_storeu_pd(rem, r[nrowx - 1]);
        for(int i = 0; i < len - mainlen; i++, res++)
            *res = rem[i];
    }

    free(r);
}

void mat_spvec_prod_avx(Eigen::VectorXd &res, const Eigen::Map<const Eigen::MatrixXd> &mat, const Eigen::SparseVector<double> &spvec)
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

void mat_vec_tprod_avx(Eigen::VectorXd &res_, const Eigen::Ref<const Eigen::MatrixXd> &mat_, const Eigen::Ref<const Eigen::VectorXd> &vec_)
{
    const double *mat = mat_.data();
    const double *vec = vec_.data();
    double *res = res_.data();
    const int nrow = mat_.rows();
    const int ncol = mat_.cols();
    const int npack = nrow / 4;

    // Reading vec into an array of __m256d
    __m256d *vecd;
    posix_memalign((void **)&vecd, 32, npack * sizeof(__m256d));
    const double *vecptr = vec;
    for(int i = 0; i < npack; i++, vecptr += 4)
        vecd[i] = _mm256_loadu_pd(vecptr);

    // Loop over columns
    #pragma omp parallel for num_threads(2)
    for(int i = 0; i < ncol; i++)
    {
        __m256d matd;
        __m256d r = _mm256_setzero_pd();
        const double *colptr = mat + nrow * i;
        for(int j = 0; j < npack; j++, colptr += 4)
        {
            matd = _mm256_loadu_pd(colptr);
            r = _mm256_add_pd(r, _mm256_mul_pd(vecd[j], matd));
            // r = _mm256_fmadd_pd(vecd[j], matd, r);
        }
        r = _mm256_hadd_pd(r, r);
        double *p = (double *)&r;
        double resi = p[0] + p[2];
        for(int j = 4 * npack; j < nrow; j++, colptr++)
            resi += (*colptr) * vec[j];

        res[i] = resi;
    }

   free(vecd);
}
#endif // __AVX__



#endif // AVX_H
