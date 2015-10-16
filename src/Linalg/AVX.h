#ifndef AVX_H
#define AVX_H

#include <Eigen/Core>
#include <Eigen/Sparse>

#ifdef __AVX__
#include <immintrin.h>


class MemAligned
{
private:
    // http://stackoverflow.com/questions/6563120/what-does-posix-memalign-memalign-do
    static inline void *malloc_aligned(size_t align, size_t len)
    {
        // align == 0, or not a power of 2
        if(align == 0 || (align & (align - 1)))
            return (void *)0;

        // align is not a multiple of sizeof(void *)
        if(align % sizeof(void *))
            return (void *)0;

        // len + align - 1 to guarantee the length with alignment,
        // sizeof(size_t) to record the start position
        const size_t total = len + align - 1 + sizeof(size_t);
        char *data = (char *)malloc(total);

        if(data)
        {
            // the start location of "data"", used to free the memory
            const void * const start = (void *)data;
            // reserve space to store "start"
            data += sizeof(size_t);
            // find an integer greater than or equal to "data",
            // and is a multiple of "align"
            // the padding will be align - data % align
            size_t padding = align - (((size_t)data) % align);
            // move data to the aligned location
            data += padding;
            // location to write "start"
            size_t *recorder = (size_t *)(data - sizeof(size_t));
            // write "start" to recorder
            *recorder = (size_t)start;
        }

        return (void *)data;
    }

    static inline void free_aligned(void *ptr)
    {
        if(ptr)
        {
            char *data = (char *)ptr;
            size_t *recorder = (size_t *)(data - sizeof(size_t));
            data = (char *)(*recorder);
            free(data);
        }
    }
public:
    static inline void *allocate(size_t align, size_t len)
    {
        void *ptr;
    #ifdef _WIN32
        ptr = _aligned_malloc(len, align);
    #elif defined(posix_memalign)
        posix_memalign(&ptr, align, len);
    #else
        ptr = malloc_aligned(align, len);
    #endif

        return (void *) ptr;
    }

    static inline void destroy(void *ptr)
    {
    #ifdef _WIN32
        _aligned_free(ptr);
    #elif defined(posix_memalign)
        free(ptr);
    #else
        free_aligned(ptr);
    #endif
    }
};



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

        float *rem = (float *) MemAligned::allocate(32, 8 * sizeof(float));
        std::fill(rem, rem + 8, float(0));

        __m256 *data_ptr;
        data = (__m256 *) MemAligned::allocate(32, nrowx * (ncol + 1) * sizeof(__m256));
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

        MemAligned::destroy(rem);
    }

    ~vtrMatrixf()
    {
        MemAligned::destroy(data);
    }

    void read_vec(const float *vec)
    {
        const int npack = nrow / 8;
        const int tail = nrow - npack * 8;

        float *rem = (float *) MemAligned::allocate(32, 8 * sizeof(float));
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

        MemAligned::destroy(rem);
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



// Calculating \sum x_i and \sum x_i^2
template <typename T>
inline void get_ss_avx(const T *x, const int len, T &sum, T&sum_of_square) {}

template <>
inline void get_ss_avx<double>(const double *x, const int len, double &sum, double &sum_of_square)
{
    __m256d xx;
    __m256d s = _mm256_setzero_pd();
    __m256d ss = _mm256_setzero_pd();

    sum = 0.0;
    sum_of_square = 0.0;

    const double *x_end = x + len;

    const char rem = (unsigned long)x % 32;
    const char head = rem ? (32 - rem) / sizeof(double) : 0;

    const double *xfor = x + head;
    for( ; x < xfor; x++)
    {
        sum += *x;
        sum_of_square += (*x) * (*x);
    }

    const int npack = (len - head) / 8;
    xfor = x + 8 * npack;
    for( ; x < xfor; x += 8)
    {
        xx = _mm256_load_pd(x);
        s = _mm256_add_pd(s, xx);
        ss = _mm256_add_pd(ss, _mm256_mul_pd(xx, xx));

        xx = _mm256_load_pd(x + 4);
        s = _mm256_add_pd(s, xx);
        ss = _mm256_add_pd(ss, _mm256_mul_pd(xx, xx));
    }
    s = _mm256_hadd_pd(s, s);
    ss = _mm256_hadd_pd(ss, ss);

    double *resp = (double *) &s;
    sum += resp[0] + resp[2];
    resp = (double *) &ss;
    sum_of_square += resp[0] + resp[2];

    for( ; x < x_end; x++)
    {
        sum += *x;
        sum_of_square += (*x) * (*x);
    }
}

template <>
inline void get_ss_avx<float>(const float *x, const int len, float &sum, float &sum_of_square)
{
    __m256 xx;
    __m256 s = _mm256_setzero_ps();
    __m256 ss = _mm256_setzero_ps();

    sum = 0.0;
    sum_of_square = 0.0;

    const float *x_end = x + len;

    const char rem = (unsigned long)x % 32;
    const char head = rem ? (32 - rem) / sizeof(float) : 0;

    const float *xfor = x + head;
    for( ; x < xfor; x++)
    {
        sum += *x;
        sum_of_square += (*x) * (*x);
    }

    const int npack = (len - head) / 16;
    xfor = x + 16 * npack;
    for( ; x < xfor; x += 16)
    {
        xx = _mm256_load_ps(x);
        s = _mm256_add_ps(s, xx);
        ss = _mm256_add_ps(ss, _mm256_mul_ps(xx, xx));

        xx = _mm256_load_ps(x + 8);
        s = _mm256_add_ps(s, xx);
        ss = _mm256_add_ps(ss, _mm256_mul_ps(xx, xx));
    }
    sum += vtrMatrixf::_mm256_reduce_add_ps(s);
    sum_of_square += vtrMatrixf::_mm256_reduce_add_ps(ss);

    for( ; x < x_end; x++)
    {
        sum += *x;
        sum_of_square += (*x) * (*x);
    }
}



// x = > (x - center) * scale
template <typename T>
inline void standardize_vec_avx(T *x, const int len, const T center, const T scale) {}

template <>
inline void standardize_vec_avx<double>(double *x, const int len, const double center, const double scale)
{
    __m256d xx;
    __m256d cc = _mm256_set1_pd(center);
    __m256d ss = _mm256_set1_pd(scale);

    const double *x_end = x + len;

    const char rem = (unsigned long)x % 32;
    const char head = rem ? (32 - rem) / sizeof(double) : 0;

    const double *xfor = x + head;
    for( ; x < xfor; x++)
        *x = (*x - center) * scale;

    const int npack = (len - head) / 8;
    xfor = x + 8 * npack;
    for( ; x < xfor; x += 8)
    {
        xx = _mm256_load_pd(x);
        xx = _mm256_mul_pd(_mm256_sub_pd(xx, cc), ss);
        _mm256_store_pd(x, xx);

        xx = _mm256_load_pd(x + 4);
        xx = _mm256_mul_pd(_mm256_sub_pd(xx, cc), ss);
        _mm256_store_pd(x + 4, xx);
    }

    for( ; x < x_end; x++)
        *x = (*x - center) * scale;
}

template <>
inline void standardize_vec_avx<float>(float *x, const int len, const float center, const float scale)
{
    __m256 xx;
    __m256 cc = _mm256_set1_ps(center);
    __m256 ss = _mm256_set1_ps(scale);

    const float *x_end = x + len;

    const char rem = (unsigned long)x % 32;
    const char head = rem ? (32 - rem) / sizeof(float) : 0;

    const float *xfor = x + head;
    for( ; x < xfor; x++)
        *x = (*x - center) * scale;

    const int npack = (len - head) / 16;
    xfor = x + 16 * npack;
    for( ; x < xfor; x += 16)
    {
        xx = _mm256_load_ps(x);
        xx = _mm256_mul_ps(_mm256_sub_ps(xx, cc), ss);
        _mm256_store_ps(x, xx);

        xx = _mm256_load_ps(x + 8);
        xx = _mm256_mul_ps(_mm256_sub_ps(xx, cc), ss);
        _mm256_store_ps(x + 8, xx);
    }

    for( ; x < x_end; x++)
        *x = (*x - center) * scale;
}

#endif // __AVX__



#endif // AVX_H
