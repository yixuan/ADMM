#ifndef DATASTD_H
#define DATASTD_H

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



#endif // DATASTD_H
