#ifndef AVX_H
#define AVX_H

#include <Eigen/Core>
#include <Eigen/Sparse>

#ifdef __AVX__
#include <immintrin.h>

class vtrMatrixf
{
private:
    typedef const Eigen::Ref<const Eigen::MatrixXf> ConstGenericMatrix;

    __m256 *data;
    __m256 *vdata;
    __m256 *mdata;

    int nrow;
    int ncol;
    int nrowx;

public:
    vtrMatrixf() {}

    void read_mat(ConstGenericMatrix &mat)
    {
        nrow = mat.rows();
        ncol = mat.cols();
        const int npack = nrow / 8;
        const int tail = nrow - npack * 8;
        nrowx = npack + int(tail != 0);

        float *rem;
        posix_memalign((void **)&rem, 32, 8 * sizeof(float));
        std::fill(rem, rem + 8, float(0));

        __m256 *data_ptr;
        posix_memalign((void **)&data, 32, nrowx * (ncol + 1) * sizeof(__m256));
        vdata = data;
        mdata = data + nrowx;
        data_ptr = mdata;

        const float *mat0 = mat.data();
        const float *col_ptr;
        const __m256 *end_ptr;

        if(tail == 0)
        {
            for(int i = 0; i < ncol; i++)
            {
                col_ptr = mat0 + nrow * i;
                end_ptr = data_ptr + npack;
                for( ; data_ptr < end_ptr; col_ptr += 8, data_ptr++)
                    *data_ptr = _mm256_loadu_ps(col_ptr);
            }
        } else {
            for(int i = 0; i < ncol; i++)
            {
                col_ptr = mat0 + nrow * i;
                end_ptr = data_ptr + npack;
                for( ; data_ptr < end_ptr; col_ptr += 8, data_ptr++)
                    *data_ptr = _mm256_loadu_ps(col_ptr);
                for(int j = 0; j < tail; j++, col_ptr++)
                    rem[j] = float(*col_ptr);

                *data_ptr = _mm256_load_ps(rem);
                data_ptr++;
            }
        }

        free(rem);

    }

    ~vtrMatrixf()
    {
        free(data);
    }

    void read_vec(const float *vec)
    {
        const int npack = nrow / 8;
        const int tail = nrow - npack * 8;

        float *rem;
        posix_memalign((void **)&rem, 32, 8 * sizeof(float));
        std::fill(rem, rem + 8, float(0));

        __m256 *data_ptr = vdata;
        __m256 *end_ptr = data_ptr + npack;
        for( ; data_ptr < end_ptr; vec += 8, data_ptr++)
            *data_ptr = _mm256_loadu_ps(vec);

        if(tail != 0)
        {
            for(int j = 0; j < tail; j++, vec++)
                rem[j] = float(*vec);

            *data_ptr = _mm256_load_ps(rem);
        }

        free(rem);
    }

    void mult_spvec(const Eigen::SparseVector<float> &spvec, float *res)
    {
        __m256 *v_ptr = vdata;
        for( ; v_ptr < mdata; v_ptr++)
            *v_ptr = _mm256_setzero_ps();

        __m256 c;  // Constant 8-vector
        const __m256 *col_ptr;  // Pointing to columns of mat
        const __m256 *end_ptr;

        int npack = nrowx / 8;
        int mainlen = npack * 8;

        for(Eigen::SparseVector<float>::InnerIterator iter(spvec); iter; ++iter)
        {
            col_ptr = mdata + nrowx * iter.index();
            v_ptr = vdata;
            end_ptr = v_ptr + mainlen;
            c = _mm256_set1_ps(iter.value());
            for( ; v_ptr < end_ptr; col_ptr += 8, v_ptr += 8)
            {
                v_ptr[0] = _mm256_add_ps(v_ptr[0], _mm256_mul_ps(col_ptr[0], c));
                v_ptr[1] = _mm256_add_ps(v_ptr[1], _mm256_mul_ps(col_ptr[1], c));
                v_ptr[2] = _mm256_add_ps(v_ptr[2], _mm256_mul_ps(col_ptr[2], c));
                v_ptr[3] = _mm256_add_ps(v_ptr[3], _mm256_mul_ps(col_ptr[3], c));
                v_ptr[4] = _mm256_add_ps(v_ptr[4], _mm256_mul_ps(col_ptr[4], c));
                v_ptr[5] = _mm256_add_ps(v_ptr[5], _mm256_mul_ps(col_ptr[5], c));
                v_ptr[6] = _mm256_add_ps(v_ptr[6], _mm256_mul_ps(col_ptr[6], c));
                v_ptr[7] = _mm256_add_ps(v_ptr[7], _mm256_mul_ps(col_ptr[7], c));
            }

            for( ; v_ptr < mdata; col_ptr++, v_ptr++)
                *v_ptr = _mm256_add_ps(*v_ptr, _mm256_mul_ps(*col_ptr, c));
         }

        npack = nrow / 8;
        mainlen = npack * 8;

        for(int i = 0; i < npack; i++, res += 8)
            _mm256_storeu_ps(res, vdata[i]);
        if(nrowx > npack)
        {
            float rem[8];
            _mm256_storeu_ps(rem, vdata[nrowx - 1]);
            for(int i = 0; i < nrow - mainlen; i++, res++)
                *res = rem[i];
        }
    }

    void mult_vec(const Eigen::VectorXf &vec, float *res)
    {
        __m256 *v_ptr = vdata;
        for( ; v_ptr < mdata; v_ptr++)
            *v_ptr = _mm256_setzero_ps();

        __m256 c;  // Constant 8-vector
        const __m256 *col_ptr;  // Pointing to columns of mat
        const __m256 *end_ptr;

        int npack = nrowx / 8;
        int mainlen = npack * 8;

        const float *vec_ptr = vec.data();
        for(int i = 0; i < ncol; i++)
        {
            col_ptr = mdata + nrowx * i;
            v_ptr = vdata;
            end_ptr = v_ptr + mainlen;
            c = _mm256_set1_ps(vec_ptr[i]);
            for( ; v_ptr < end_ptr; col_ptr += 8, v_ptr += 8)
            {
                v_ptr[0] = _mm256_add_ps(v_ptr[0], _mm256_mul_ps(col_ptr[0], c));
                v_ptr[1] = _mm256_add_ps(v_ptr[1], _mm256_mul_ps(col_ptr[1], c));
                v_ptr[2] = _mm256_add_ps(v_ptr[2], _mm256_mul_ps(col_ptr[2], c));
                v_ptr[3] = _mm256_add_ps(v_ptr[3], _mm256_mul_ps(col_ptr[3], c));
                v_ptr[4] = _mm256_add_ps(v_ptr[4], _mm256_mul_ps(col_ptr[4], c));
                v_ptr[5] = _mm256_add_ps(v_ptr[5], _mm256_mul_ps(col_ptr[5], c));
                v_ptr[6] = _mm256_add_ps(v_ptr[6], _mm256_mul_ps(col_ptr[6], c));
                v_ptr[7] = _mm256_add_ps(v_ptr[7], _mm256_mul_ps(col_ptr[7], c));
            }

            for( ; v_ptr < mdata; col_ptr++, v_ptr++)
                *v_ptr = _mm256_add_ps(*v_ptr, _mm256_mul_ps(*col_ptr, c));
         }

        npack = nrow / 8;
        mainlen = npack * 8;

        for(int i = 0; i < npack; i++, res += 8)
            _mm256_storeu_ps(res, vdata[i]);
        if(nrowx > npack)
        {
            float rem[8];
            _mm256_storeu_ps(rem, vdata[nrowx - 1]);
            for(int i = 0; i < nrow - mainlen; i++, res++)
                *res = rem[i];
        }
    }

    // http://stackoverflow.com/questions/13219146/how-to-sum-m256-horizontally
    static inline float _mm256_reduce_add_ps(__m256 x)
    {
        /* ( x3+x7, x2+x6, x1+x5, x0+x4 ) */
        const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
        /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
        const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
        /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
        const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
        /* Conversion to float is a no-op on x86-64 */
        return _mm_cvtss_f32(x32);
    }

    void trans_mult_vec(const Eigen::VectorXf &vec, float *res)
    {
        read_vec(vec.data());

        const int npack = nrowx / 8;
        const int mainlen = npack * 8;

        // Loop over columns
        #pragma omp parallel for num_threads(2)
        for(int i = 0; i < ncol; i++)
        {
            __m256 *col_ptr = mdata + nrowx * i;
            __m256 *v_ptr = vdata;
            __m256 *end_ptr = v_ptr + mainlen;
            __m256 r = _mm256_setzero_ps();
            for( ; v_ptr < end_ptr; col_ptr += 8, v_ptr += 8)
            {
                r = _mm256_add_ps(r, _mm256_mul_ps(v_ptr[0], col_ptr[0]));
                r = _mm256_add_ps(r, _mm256_mul_ps(v_ptr[1], col_ptr[1]));
                r = _mm256_add_ps(r, _mm256_mul_ps(v_ptr[2], col_ptr[2]));
                r = _mm256_add_ps(r, _mm256_mul_ps(v_ptr[3], col_ptr[3]));
                r = _mm256_add_ps(r, _mm256_mul_ps(v_ptr[4], col_ptr[4]));
                r = _mm256_add_ps(r, _mm256_mul_ps(v_ptr[5], col_ptr[5]));
                r = _mm256_add_ps(r, _mm256_mul_ps(v_ptr[6], col_ptr[6]));
                r = _mm256_add_ps(r, _mm256_mul_ps(v_ptr[7], col_ptr[7]));
            }
            for( ; v_ptr < mdata; col_ptr++, v_ptr++)
                r = _mm256_add_ps(r, _mm256_mul_ps(*v_ptr, *col_ptr));

            res[i] = _mm256_reduce_add_ps(r);
            // float *p = (float *)&r;
            // res[i] = p[0] + p[1] + p[2] + p[3] + p[4] + p[5] + p[6] + p[7];
        }
    }

    float ith_inner_product(const int i)
    {
        const int npack = nrowx / 8;
        const int mainlen = npack * 8;

        __m256 *col_ptr = mdata + nrowx * i;
        __m256 *v_ptr = vdata;
        __m256 *end_ptr = v_ptr + mainlen;
        __m256 r = _mm256_setzero_ps();
        for( ; v_ptr < end_ptr; col_ptr += 8, v_ptr += 8)
        {
            r = _mm256_add_ps(r, _mm256_mul_ps(v_ptr[0], col_ptr[0]));
            r = _mm256_add_ps(r, _mm256_mul_ps(v_ptr[1], col_ptr[1]));
            r = _mm256_add_ps(r, _mm256_mul_ps(v_ptr[2], col_ptr[2]));
            r = _mm256_add_ps(r, _mm256_mul_ps(v_ptr[3], col_ptr[3]));
            r = _mm256_add_ps(r, _mm256_mul_ps(v_ptr[4], col_ptr[4]));
            r = _mm256_add_ps(r, _mm256_mul_ps(v_ptr[5], col_ptr[5]));
            r = _mm256_add_ps(r, _mm256_mul_ps(v_ptr[6], col_ptr[6]));
            r = _mm256_add_ps(r, _mm256_mul_ps(v_ptr[7], col_ptr[7]));
        }
        for( ; v_ptr < mdata; col_ptr++, v_ptr++)
            r = _mm256_add_ps(r, _mm256_mul_ps(*v_ptr, *col_ptr));

        return _mm256_reduce_add_ps(r);
    }
};

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

inline void mat_spvec_prod_avx(Eigen::VectorXd &res, const Eigen::Map<const Eigen::MatrixXd> &mat, const Eigen::SparseVector<double> &spvec)
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

inline void mat_vec_tprod_avx(Eigen::VectorXd &res_, const Eigen::Ref<const Eigen::MatrixXd> &mat_, const Eigen::Ref<const Eigen::VectorXd> &vec_)
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
