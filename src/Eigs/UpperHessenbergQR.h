#ifndef UpperHessenbergQR_H
#define UpperHessenbergQR_H

#include <Eigen/Dense>

// QR decomposition of an upper Hessenberg matrix
template <typename Scalar = double>
class UpperHessenbergQR
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, 2, 2> Matrix22;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Matrix<Scalar, 1, Eigen::Dynamic> RowVector;
    typedef Eigen::Array<Scalar, Eigen::Dynamic, 1> Array;

protected:
    int n;
    Matrix mat_T;
    // Gi = [ cos[i]  sin[i]]
    //      [-sin[i]  cos[i]]
    // Q = G1 * G2 * ... * G_{n-1}
    Array rot_cos;
    Array rot_sin;
    bool computed;
public:
    UpperHessenbergQR() :
        n(0), computed(false)
    {}

    UpperHessenbergQR(int n_) :
        n(n_),
        mat_T(n, n),
        rot_cos(n - 1),
        rot_sin(n - 1),
        computed(false)
    {}

    UpperHessenbergQR(const Matrix &mat) :
        n(mat.rows()),
        mat_T(n, n),
        rot_cos(n - 1),
        rot_sin(n - 1),
        computed(false)
    {
        compute(mat);
    }

    virtual void compute(const Matrix &mat)
    {
        n = mat.rows();
        mat_T.resize(n, n);
        rot_cos.resize(n - 1);
        rot_sin.resize(n - 1);

        mat_T = mat;

        Scalar xi, xj, r, c, s;
        Matrix22 Gt;
        for(int i = 0; i < n - 2; i++)
        {
            xi = mat_T(i, i);
            xj = mat_T(i + 1, i);
            r = std::sqrt(xi * xi + xj * xj);
            rot_cos[i] = c = xi / r;
            rot_sin[i] = s = -xj / r;
            // For a complete QR decomposition,
            // we first obtain the rotation matrix
            // G = [ cos  sin]
            //     [-sin  cos]
            // and then do T[i:(i + 1), i:(n - 1)] = G' * T[i:(i + 1), i:(n - 1)]

            Gt << c, -s, s, c;
            mat_T.block(i, i, 2, n - i) = Gt * mat_T.block(i, i, 2, n - i);

            // If we do not need to calculate the R matrix, then
            // only the cos and sin sequences are required.
            // In such case we only update T[i + 1, (i + 1):(n - 1)]
            // mat_T.block(i + 1, i + 1, 1, n - i - 1) *= c;
            // mat_T.block(i + 1, i + 1, 1, n - i - 1) += s * mat_T.block(i, i + 1, 1, n - i - 1);
        }
        // For i = n - 2
        xi = mat_T(n - 2, n - 2);
        xj = mat_T(n - 1, n - 2);
        r = std::sqrt(xi * xi + xj * xj);
        rot_cos[n - 2] = c = xi / r;
        rot_sin[n - 2] = s = -xj / r;
        Gt << c, -s, s, c;
        mat_T.template block<2, 2>(n - 2, n - 2) = Gt * mat_T.template block<2, 2>(n - 2, n - 2);

        computed = true;
    }

    Matrix matrix_R()
    {
        if(!computed)
            return Matrix();

        return mat_T;
    }

    // Y -> QY = G1 * G2 * ... * Y
    void apply_QY(Vector &Y)
    {
        if(!computed)
            return;

        Scalar *c = rot_cos.data() + n - 2,
               *s = rot_sin.data() + n - 2,
               *Yi = Y.data() + n - 2,
               tmp;
        for(int i = n - 2; i >= 0; i--)
        {
            // Y[i:(i + 1)] = Gi * Y[i:(i + 1)]
            // Gi = [ cos[i]  sin[i]]
            //      [-sin[i]  cos[i]]
            tmp = *Yi;
            // Yi[0] == Y[i], Yi[1] == Y[i + 1]
            Yi[0] =  (*c) * tmp + (*s) * Yi[1];
            Yi[1] = -(*s) * tmp + (*c) * Yi[1];

            Yi--;
            c--;
            s--;
        }
    }

    // Y -> Q'Y = G_{n-1}' * ... * G2' * G1' * Y
    void apply_QtY(Vector &Y)
    {
        if(!computed)
            return;

        Scalar *c = rot_cos.data(),
               *s = rot_sin.data(),
               *Yi = Y.data(),
               tmp;
        for(int i = 0; i < n - 1; i++)
        {
            // Y[i:(i + 1)] = Gi' * Y[i:(i + 1)]
            // Gi = [ cos[i]  sin[i]]
            //      [-sin[i]  cos[i]]
            tmp = *Yi;
            // Yi[0] == Y[i], Yi[1] == Y[i + 1]
            Yi[0] = (*c) * tmp - (*s) * Yi[1];
            Yi[1] = (*s) * tmp + (*c) * Yi[1];

            Yi++;
            c++;
            s++;
        }
    }

    // Y -> QY = G1 * G2 * ... * Y
    void apply_QY(Matrix &Y)
    {
        if(!computed)
            return;

        Scalar *c = rot_cos.data() + n - 2,
               *s = rot_sin.data() + n - 2;
        RowVector Yi(Y.cols());
        for(int i = n - 2; i >= 0; i--)
        {
            // Y[i:(i + 1), ] = Gi * Y[i:(i + 1), ]
            // Gi = [ cos[i]  sin[i]]
            //      [-sin[i]  cos[i]]
            Yi = Y.row(i);
            Y.row(i)     =  (*c) * Yi + (*s) * Y.row(i + 1);
            Y.row(i + 1) = -(*s) * Yi + (*c) * Y.row(i + 1);
            c--;
            s--;
        }
    }

    // Y -> Q'Y = G_{n-1}' * ... * G2' * G1' * Y
    void apply_QtY(Matrix &Y)
    {
        if(!computed)
            return;

        Scalar *c = rot_cos.data(),
               *s = rot_sin.data();
        RowVector Yi(Y.cols());
        for(int i = 0; i < n - 1; i++)
        {
            // Y[i:(i + 1), ] = Gi' * Y[i:(i + 1), ]
            // Gi = [ cos[i]  sin[i]]
            //      [-sin[i]  cos[i]]
            Yi = Y.row(i);
            Y.row(i)     = (*c) * Yi - (*s) * Y.row(i + 1);
            Y.row(i + 1) = (*s) * Yi + (*c) * Y.row(i + 1);
            c++;
            s++;
        }
    }

    // Y -> YQ = Y * G1 * G2 * ...
    void apply_YQ(Matrix &Y)
    {
        if(!computed)
            return;

        Scalar *c = rot_cos.data(),
               *s = rot_sin.data();
        Vector Yi(Y.rows());
        for(int i = 0; i < n - 1; i++)
        {
            // Y[, i:(i + 1)] = Y[, i:(i + 1)] * Gi
            // Gi = [ cos[i]  sin[i]]
            //      [-sin[i]  cos[i]]
            Yi = Y.col(i);
            Y.col(i)     = (*c) * Yi - (*s) * Y.col(i + 1);
            Y.col(i + 1) = (*s) * Yi + (*c) * Y.col(i + 1);
            c++;
            s++;
        }
    }

    // Y -> YQ' = Y * G_{n-1}' * ... * G2' * G1'
    void apply_YQt(Matrix &Y)
    {
        if(!computed)
            return;

        Scalar *c = rot_cos.data() + n - 2,
               *s = rot_sin.data() + n - 2;
        Vector Yi(Y.rows());
        for(int i = n - 2; i >= 0; i--)
        {
            // Y[, i:(i + 1)] = Y[, i:(i + 1)] * Gi'
            // Gi = [ cos[i]  sin[i]]
            //      [-sin[i]  cos[i]]
            Yi = Y.col(i);
            Y.col(i)     =  (*c) * Yi + (*s) * Y.col(i + 1);
            Y.col(i + 1) = -(*s) * Yi + (*c) * Y.col(i + 1);
            c--;
            s--;
        }
    }
};



// QR decomposition of a tridiagonal matrix as a special case of
// upper Hessenberg matrix
template <typename Scalar = double>
class TridiagQR: public UpperHessenbergQR<Scalar>
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;

public:
    TridiagQR() :
        UpperHessenbergQR<Scalar>()
    {}

    TridiagQR(int n_) :
        UpperHessenbergQR<Scalar>(n_)
    {}

    TridiagQR(const Matrix &mat) :
        UpperHessenbergQR<Scalar>(mat.rows())
    {
        this->compute(mat);
    }

    virtual void compute(const Matrix &mat)
    {
        this->n = mat.rows();
        this->mat_T.resize(this->n, this->n);
        this->rot_cos.resize(this->n - 1);
        this->rot_sin.resize(this->n - 1);

        this->mat_T.setZero();
        this->mat_T.diagonal() = mat.diagonal();
        this->mat_T.diagonal(1) = mat.diagonal(-1);
        this->mat_T.diagonal(-1) = mat.diagonal(-1);

        // A number of pointers to avoid repeated address calculation
        Scalar *Tii = this->mat_T.data(),  // pointer to T[i, i]
               *ptr,                       // some location relative to Tii
               *c = this->rot_cos.data(),  // pointer to the cosine vector
               *s = this->rot_sin.data(),  // pointer to the sine vector
               r, tmp;
        for(int i = 0; i < this->n - 2; i++)
        {
            // Tii[0] == T[i, i]
            // Tii[1] == T[i + 1, i]
            r = std::sqrt(Tii[0] * Tii[0] + Tii[1] * Tii[1]);
            *c =  Tii[0] / r;
            *s = -Tii[1] / r;

            // For a complete QR decomposition,
            // we first obtain the rotation matrix
            // G = [ cos  sin]
            //     [-sin  cos]
            // and then do T[i:(i + 1), i:(i + 2)] = G' * T[i:(i + 1), i:(i + 2)]

            // Update T[i, i] and T[i + 1, i]
            // The updated value of T[i, i] is known to be r
            // The updated value of T[i + 1, i] is known to be 0
            Tii[0] = r;
            Tii[1] = 0;
            // Update T[i, i + 1] and T[i + 1, i + 1]
            // ptr[0] == T[i, i + 1]
            // ptr[1] == T[i + 1, i + 1]
            ptr = Tii + this->n;
            tmp = *ptr;
            ptr[0] = (*c) * tmp - (*s) * ptr[1];
            ptr[1] = (*s) * tmp + (*c) * ptr[1];
            // Update T[i, i + 2] and T[i + 1, i + 2]
            // ptr[0] == T[i, i + 2] == 0
            // ptr[1] == T[i + 1, i + 2]
            ptr += this->n;
            ptr[0] = -(*s) * ptr[1];
            ptr[1] *= (*c);

            // Move from T[i, i] to T[i + 1, i + 1]
            Tii += this->n + 1;
            // Increase c and s by 1
            c++;
            s++;


            // If we do not need to calculate the R matrix, then
            // only the cos and sin sequences are required.
            // In such case we only update T[i + 1, (i + 1):(i + 2)]
            // this->mat_T(i + 1, i + 1) = (*c) * this->mat_T(i + 1, i + 1) + (*s) * this->mat_T(i, i + 1);
            // this->mat_T(i + 1, i + 2) *= (*c);
        }
        // For i = n - 2
        r = std::sqrt(Tii[0] * Tii[0] + Tii[1] * Tii[1]);
        *c =  Tii[0] / r;
        *s = -Tii[1] / r;
        Tii[0] = r;
        Tii[1] = 0;
        ptr = Tii + this->n;  // points to T[i - 2, i - 1]
        tmp = *ptr;
        ptr[0] = (*c) * tmp - (*s) * ptr[1];
        ptr[1] = (*s) * tmp + (*c) * ptr[1];

        this->computed = true;
    }

    // Calculate RQ, which will also be a tridiagonal matrix
    Matrix matrix_RQ()
    {
        if(!this->computed)
            return Matrix();

        // Make a copy of the R matrix
        Matrix RQ(this->n, this->n);
        RQ.setZero();
        RQ.diagonal() = this->mat_T.diagonal();
        RQ.diagonal(1) = this->mat_T.diagonal(1);

        // [m11  m12] will point to RQ[i:(i+1), i:(i+1)]
        // [m21  m22]
        Scalar *m11 = RQ.data(), *m12, *m21, *m22,
               *c = this->rot_cos.data(),
               *s = this->rot_sin.data(),
               tmp;
        for(int i = 0; i < this->n - 1; i++)
        {
            m21 = m11 + 1;
            m12 = m11 + this->n;
            m22 = m12 + 1;
            tmp = *m21;

            // Update diagonal and the below-subdiagonal
            *m11 = (*c) * (*m11) - (*s) * (*m12);
            *m21 = (*c) * tmp - (*s) * (*m22);
            *m22 = (*s) * tmp + (*c) * (*m22);

            // Move m11 to RQ[i+1, i+1]
            m11  = m22;
            c++;
            s++;
        }

        // Copy the below-subdiagonal to above-subdiagonal
        RQ.diagonal(1) = RQ.diagonal(-1);

        return RQ;
    }
};



#endif // UpperHessenbergQR_H
