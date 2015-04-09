#ifndef LAPACKWRAPPER_H
#define LAPACKWRAPPER_H

namespace Linalg {



extern "C"
{
    void dpotrf_(const char* uplo, const int* n, double* A, const int* lda, int* info);
    void dpotrs_(const char* uplo, const int* n, const int* nrhs,
                 const double* A, const int* lda, double* B, const int* ldb, int* info);
}



} // namespace Linalg

#endif // LAPACKWRAPPER_H
