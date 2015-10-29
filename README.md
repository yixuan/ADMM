


## Introduction

`ADMM` is an R package that utilizes the Alternating Direction Method of Multipliers
(ADMM) algorithm to solve a broad range of statistical optimization problems.
Presently the models that `ADMM` has implemented include Lasso, Elastic Net,
Least Absolute Deviation and Basis Pursuit.

## Models

### Lasso

```r
library(glmnet)
library(ADMM)
set.seed(123)
n <- 100
p <- 20
m <- 5
b <- matrix(c(runif(m), rep(0, p - m)))
x <- matrix(rnorm(n * p, mean = 1.2, sd = 2), n, p)
y <- 5 + x %*% b + rnorm(n)

fit <- glmnet(x, y)
out_glmnet <- coef(fit, s = exp(-2), exact = TRUE)
out_admm <- admm_lasso(x, y)$penalty(exp(-2))$fit()
out_paradmm <- admm_lasso(x, y)$penalty(exp(-2))$parallel()$fit()

data.frame(glmnet = as.numeric(out_glmnet),
           admm = as.numeric(out_admm$beta),
           paradmm = as.numeric(out_paradmm$beta))
```

```
##          glmnet         admm      paradmm
## 1   5.357410774  5.357455254  5.357429504
## 2   0.178916019  0.178915471  0.178917870
## 3   0.683606818  0.683609307  0.683610320
## 4   0.310518550  0.310507625  0.310525119
## 5   0.861034415  0.861029863  0.861012816
## 6   0.879797912  0.879794598  0.879801810
## 7   0.007854581  0.007850002  0.007853498
## 8   0.000000000  0.000000000  0.000000000
## 9   0.000000000  0.000000000  0.000000000
## 10  0.023462980  0.023467677  0.023452930
## 11  0.010952896  0.010957017  0.010950469
## 12  0.000000000  0.000000000  0.000000000
## 13 -0.003800159 -0.003811116 -0.003801103
## 14  0.000000000  0.000000000  0.000000000
## 15  0.094591923  0.094586611  0.094600648
## 16  0.000000000  0.000000000  0.000000000
## 17  0.000000000  0.000000000  0.000000000
## 18  0.000000000  0.000000000  0.000000000
## 19  0.000000000  0.000000000  0.000000000
## 20 -0.002916255 -0.002929136 -0.002919935
## 21  0.000000000  0.000000000  0.000000000
```

### Elastic Net

```r
fit <- glmnet(x, y, alpha = 0.5)
out_glmnet <- coef(fit, s = exp(-2), exact = TRUE)
out_admm <- admm_enet(x, y)$penalty(exp(-2), alpha = 0.5)$fit()
data.frame(glmnet = as.numeric(out_glmnet),
           admm = as.numeric(out_admm$beta))
```

```
##          glmnet         admm
## 1   5.150556538  5.150497437
## 2   0.204543779  0.204526767
## 3   0.705652674  0.705665767
## 4   0.330650192  0.330640256
## 5   0.872594728  0.872611761
## 6   0.884433876  0.884422064
## 7   0.048044107  0.048055928
## 8   0.025072878  0.025097074
## 9   0.000000000  0.000000000
## 10  0.057804317  0.057830613
## 11  0.041853068  0.041876025
## 12 -0.004476248 -0.004499977
## 13 -0.035255637 -0.035279647
## 14  0.000000000  0.000000000
## 15  0.110919341  0.110915266
## 16  0.000000000  0.000000000
## 17  0.000000000  0.000000000
## 18  0.000000000  0.000000000
## 19  0.000000000  0.000000000
## 20 -0.021003756 -0.020984368
## 21  0.000000000  0.000000000
```

### Least Absolute Deviation
Least Absolute Deviation (LAD) minimizes `||y - Xb||_1` instead of
`||y - Xb||_2^2` (OLS), and is equivalent to median regression.


```r
library(quantreg)
out_rq <- rq.fit(x, y)
out_admm <- admm_lad(x, y, intercept = FALSE)$fit()

data.frame(rq_br = out_rq$coefficients,
           admm = out_admm$beta[-1])
```

```
##           rq_br          admm
## 1   0.463871497  0.4630289961
## 2   0.829243353  0.8324149339
## 3   0.151432833  0.1493799430
## 4   1.074107564  1.0707590072
## 5   0.958979798  0.9569585188
## 6   0.502539859  0.5028832829
## 7   0.337640338  0.3360263689
## 8   0.209127703  0.2120946512
## 9   0.361765382  0.3630356485
## 10  0.323168985  0.3217875563
## 11 -0.002009264  0.0007319653
## 12 -0.036099511 -0.0370447075
## 13  0.328007777  0.3290499302
## 14  0.296038071  0.2992857234
## 15  0.310187867  0.3117528782
## 16  0.071713681  0.0711670377
## 17  0.166827429  0.1622600454
## 18  0.260366502  0.2580854533
## 19  0.324487629  0.3251952295
## 20  0.209758565  0.2131039214
```

### Basis Pursuit

```r
set.seed(123)
n <- 50
p <- 100
nsig <- 15
beta_true <- c(runif(nsig), rep(0, p - nsig))
beta_true <- sample(beta_true)

x <- matrix(rnorm(n * p), n, p)
y <- drop(x %*% beta_true)
out_admm <- admm_bp(x, y)$fit()

range(beta_true - out_admm$beta)
```

```
## [1] -0.0006052779  0.0004780069
```


## Performance

### Lasso and Elastic Net


```r
library(microbenchmark)
library(ADMM)
library(glmnet)
# compute the full solution path, n > p
set.seed(123)
n <- 10000
p <- 1000
m <- 100
b <- matrix(c(runif(m), rep(0, p - m)))
x <- matrix(rnorm(n * p, sd = 2), n, p)
y <- x %*% b + rnorm(n)

lambdas1 = glmnet(x, y)$lambda
lambdas2 = glmnet(x, y, alpha = 0.6)$lambda

microbenchmark(
    "glmnet[lasso]" = {res1 <- glmnet(x, y)},
    "admm[lasso]"   = {res2 <- admm_lasso(x, y)$penalty(lambdas1)$fit()},
    "padmm[lasso]"  = {res3 <- admm_lasso(x, y)$penalty(lambdas1)$parallel()$fit()},
    "glmnet[enet]"  = {res4 <- glmnet(x, y, alpha = 0.6)},
    "admm[enet]"    = {res5 <- admm_enet(x, y)$penalty(lambdas2, alpha = 0.6)$fit()},
    times = 5
)
```

```
## Unit: milliseconds
##           expr      min        lq      mean    median        uq       max neval
##  glmnet[lasso] 939.0194  943.7227 1005.1232 1043.2826 1048.1939 1051.3973     5
##    admm[lasso] 320.1375  320.7603  321.6413  321.0283  321.0635  325.2172     5
##   padmm[lasso] 486.7478  510.7104  529.9721  512.5421  567.0847  572.7757     5
##   glmnet[enet] 950.9513 1048.1872 1031.4437 1049.9402 1051.2483 1056.8917     5
##     admm[enet] 286.9177  288.8735  289.0231  288.9758  289.1651  291.1834     5
```

```r
# difference of results
diffs = matrix(0, 3, 2)
rownames(diffs) = c("glmnet-admm [lasso]", "glmnet-padmm[lasso]", "glmnet-admm [enet]")
colnames(diffs) = c("min", "max")
diffs[1, ] = range(coef(res1) - res2$beta)
diffs[2, ] = range(coef(res1) - res3$beta)
diffs[3, ] = range(coef(res4) - res5$beta)
diffs
```

```
##                               min          max
## glmnet-admm [lasso] -0.0002873333 7.259293e-05
## glmnet-padmm[lasso] -0.0005554722 7.382258e-05
## glmnet-admm [enet]  -0.0002195360 8.176991e-05
```

```r
# p > n
set.seed(123)
n <- 1000
p <- 2000
m <- 100
b <- matrix(c(runif(m), rep(0, p - m)))
x <- matrix(rnorm(n * p, sd = 2), n, p)
y <- x %*% b + rnorm(n)

lambdas1 = glmnet(x, y)$lambda
lambdas2 = glmnet(x, y, alpha = 0.6)$lambda

microbenchmark(
    "glmnet[lasso]" = {res1 <- glmnet(x, y)},
    "admm[lasso]"   = {res2 <- admm_lasso(x, y)$penalty(lambdas1)$fit()},
    "padmm[lasso]"  = {res3 <- admm_lasso(x, y)$penalty(lambdas1)$parallel()$fit()},
    "glmnet[enet]"  = {res4 <- glmnet(x, y, alpha = 0.6)},
    "admm[enet]"    = {res5 <- admm_enet(x, y)$penalty(lambdas2, alpha = 0.6)$fit()},
    times = 5
)
```

```
## Unit: milliseconds
##           expr       min        lq      mean    median        uq       max neval
##  glmnet[lasso]  197.9279  198.2767  199.1576  199.3774  200.0559  200.1499     5
##    admm[lasso]  230.6332  237.1298  245.8944  247.4240  250.6257  263.6596     5
##   padmm[lasso] 5159.2198 5170.7308 5513.3797 5345.6322 5426.8846 6464.4313     5
##   glmnet[enet]  195.7102  196.7432  197.3977  197.7577  198.3421  198.4355     5
##     admm[enet]  225.8397  239.6536  247.2756  249.9269  252.2080  268.7499     5
```

```r
# difference of results
diffs[1, ] = range(coef(res1) - res2$beta)
diffs[2, ] = range(coef(res1) - res3$beta)
diffs[3, ] = range(coef(res4) - res5$beta)
diffs
```

```
##                              min         max
## glmnet-admm [lasso] -0.001518947 0.002055109
## glmnet-padmm[lasso] -0.001898237 0.002052009
## glmnet-admm [enet]  -0.001615556 0.001948477
```

### LAD


```r
library(ADMM)
library(quantreg)

set.seed(123)
n <- 1000
p <- 500
b <- runif(p)
x <- matrix(rnorm(n * p, sd = 2), n, p)
y <- x %*% b + rnorm(n)

microbenchmark(
    "quantreg[br]" = {res1 <- rq.fit(x, y)},
    "quantreg[fn]" = {res2 <- rq.fit(x, y, method = "fn")},
    "admm"         = {res3 <- admm_lad(x, y, intercept = FALSE)$fit()},
    times = 5
)
```

```
## Unit: milliseconds
##          expr        min         lq       mean     median         uq
##  quantreg[br] 2420.34862 2422.79215 2493.76695 2426.54150 2429.49592
##  quantreg[fn]  451.87866  452.94572  454.71406  453.56378  455.81717
##          admm   50.55354   51.17922   51.69386   51.59099   52.32357
##         max neval
##  2769.65653     5
##   459.36498     5
##    52.82196     5
```

```r
# difference of results
range(res1$coefficients - res3$beta[-1])
```

```
## [1] -0.006989109  0.006061505
```

```r
set.seed(123)
n <- 5000
p <- 1000
b <- runif(p)
x <- matrix(rnorm(n * p, sd = 2), n, p)
y <- x %*% b + rnorm(n)

microbenchmark(
    "quantreg[fn]" = {res1 <- rq.fit(x, y, method = "fn")},
    "admm"         = {res2 <- admm_lad(x, y, intercept = FALSE)$fit()},
    times = 5
)
```

```
## Unit: seconds
##          expr      min       lq     mean   median       uq      max neval
##  quantreg[fn] 6.156911 6.231430 7.686811 6.280464 9.861437 9.903813     5
##          admm 2.184311 2.187431 2.193200 2.189173 2.202042 2.203044     5
```

```r
# difference of results
range(res1$coefficients - res2$beta[-1])
```

```
## [1] -0.003577610  0.004135838
```

### Basis Pursuit


```r
set.seed(123)
n <- 1000
p <- 2000
nsig <- 100
beta_true <- c(runif(nsig), rep(0, p - nsig))
beta_true <- sample(beta_true)
x <- matrix(rnorm(n * p), n, p)
y <- drop(x %*% beta_true)

system.time(out_admm <- admm_bp(x, y)$fit())
```

```
##    user  system elapsed
##   0.996   0.169   0.292
```

```r
range(beta_true - out_admm$beta)
```

```
## [1] -0.001267782  0.002108828
```

```r
set.seed(123)
n <- 1000
p <- 10000
nsig <- 200
beta_true <- c(runif(nsig), rep(0, p - nsig))
beta_true <- sample(beta_true)
x <- matrix(rnorm(n * p), n, p)
y <- drop(x %*% beta_true)

system.time(out_admm <- admm_bp(x, y)$fit())
```

```
##    user  system elapsed
##  19.315   0.573   4.969
```

```r
range(beta_true - out_admm$beta)
```

```
## [1] -0.1575968  0.3361001
```
