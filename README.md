
## Introduction

`ADMM` is an R package that utilizes the Alternating Direction Method of Multipliers
(ADMM) algorithm to solve a broad range of statistical optimization problems.
Presently the models that `ADMM` has implemented include Lasso, Elastic Net,
Dantzig Selector, Least Absolute Deviation and Basis Pursuit.

## Models

### Lasso

```r
library(glmnet)
```

```
## Loading required package: Matrix
## Loaded glmnet 1.9-8
```

```r
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
## 1   5.357408680  5.357429137  5.357389792
## 2   0.178916647  0.178929108  0.178906561
## 3   0.683607030  0.683613545  0.683607644
## 4   0.310519000  0.310532229  0.310512272
## 5   0.861035473  0.861037854  0.861033057
## 6   0.879797060  0.879796568  0.879798921
## 7   0.007855064  0.007845977  0.007854352
## 8   0.000000000  0.000000000  0.000000000
## 9   0.000000000  0.000000000  0.000000000
## 10  0.023462451  0.023464519  0.023457740
## 11  0.010952924  0.010936241  0.010969479
## 12  0.000000000  0.000000000  0.000000000
## 13 -0.003799738 -0.003796027 -0.003804590
## 14  0.000000000  0.000000000  0.000000000
## 15  0.094591903  0.094574901  0.094606845
## 16  0.000000000  0.000000000  0.000000000
## 17  0.000000000  0.000000000  0.000000000
## 18  0.000000000  0.000000000  0.000000000
## 19  0.000000000  0.000000000  0.000000000
## 20 -0.002916296 -0.002931312 -0.002903598
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
## 1   5.150542835  5.150446636
## 2   0.204547201  0.204528822
## 3   0.705654049  0.705664749
## 4   0.330651551  0.330625221
## 5   0.872600768  0.872624787
## 6   0.884429725  0.884414979
## 7   0.048045833  0.048074995
## 8   0.025073267  0.025106514
## 9   0.000000000  0.000000000
## 10  0.057804709  0.057837107
## 11  0.041853709  0.041855231
## 12 -0.004476365 -0.004500434
## 13 -0.035254816 -0.035258401
## 14  0.000000000  0.000000000
## 15  0.110918735  0.110928256
## 16  0.000000000  0.000000000
## 17  0.000000000  0.000000000
## 18  0.000000000  0.000000000
## 19  0.000000000  0.000000000
## 20 -0.021003676 -0.020986037
## 21  0.000000000  0.000000000
```

### Dantzig Selector

```r
library(flare)
```

```
## Loading required package: lattice
## Loading required package: MASS
## Loading required package: igraph
```

```r
X <- scale(x)
Y <- y - mean(y)

out_flare <- slim(X, Y, nlambda = 20, lambda.min.ratio = 0.01, method = "dantzig")
```

```
## Sparse Linear Regression with L1 Regularization.
## Dantzig selector with screening.
## 
## slim options summary: 
## 20 lambdas used:
##  [1] 2.2600 1.7700 1.3900 1.0900 0.8570 0.6720 0.5280 0.4140 0.3250 0.2550
## [11] 0.2000 0.1570 0.1230 0.0967 0.0759 0.0596 0.0467 0.0367 0.0288 0.0226
## Method = dantzig 
## Degree of freedom: 0 -----> 18 
## Runtime: 0.0233233 secs
```

```r
out_admm <- admm_dantzig(X, Y)$penalty(nlambda = 20, lambda_min_ratio = 0.01)$fit()

range(out_flare$beta - out_admm$beta[-1, ])
```

```
## [1] -0.0002345712  0.0002493307
```

### Least Absolute Deviation
Least Absolute Deviation (LAD) minimizes `||y - Xb||_1` instead of
`||y - Xb||_2^2` (OLS), and is equivalent to median regression.


```r
library(quantreg)
```

```
## Loading required package: SparseM
## 
## Attaching package: 'SparseM'
## 
## The following object is masked from 'package:base':
## 
##     backsolve
```

```r
out_rq1 <- rq.fit(x, y)
out_rq2 <- rq.fit(x, y, method = "fn")
out_admm <- admm_lad(x, y, intercept = FALSE)$fit()

data.frame(rq_br = out_rq1$coefficients,
           rq_fn = out_rq2$coefficients,
           admm = out_admm$beta[-1])
```

```
##           rq_br        rq_fn         admm
## 1   0.463871497  0.463871497  0.463548802
## 2   0.829243353  0.829243353  0.831390739
## 3   0.151432833  0.151432833  0.151056746
## 4   1.074107564  1.074107564  1.071884940
## 5   0.958979798  0.958979797  0.957077697
## 6   0.502539859  0.502539859  0.503264926
## 7   0.337640338  0.337640338  0.336810662
## 8   0.209127703  0.209127703  0.210975682
## 9   0.361765382  0.361765382  0.361512519
## 10  0.323168985  0.323168985  0.322718103
## 11 -0.002009264 -0.002009264  0.000333214
## 12 -0.036099511 -0.036099511 -0.037343859
## 13  0.328007777  0.328007777  0.327904096
## 14  0.296038071  0.296038071  0.299182122
## 15  0.310187867  0.310187867  0.310677887
## 16  0.071713681  0.071713681  0.071117060
## 17  0.166827429  0.166827428  0.163873300
## 18  0.260366502  0.260366502  0.258644935
## 19  0.324487629  0.324487629  0.325495169
## 20  0.209758565  0.209758565  0.211760906
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
## [1] -0.0021346773  0.0009251025
```


## Performance

### Lasso and Elastic Net


```r
library(ADMM)
library(glmnet)
# compute the full solution path, n > p
set.seed(123)
n <- 20000
p <- 1000
m <- 100
b <- matrix(c(runif(m), rep(0, p - m)))
x <- matrix(rnorm(n * p, sd = 2), n, p)
y <- x %*% b + rnorm(n)

system.time(res1 <- glmnet(x, y, nlambda = 20))
```

```
##    user  system elapsed 
##   0.972   0.040   1.011
```

```r
system.time(res2 <- admm_lasso(x, y)$penalty(res1$lambda)$fit())
```

```
##    user  system elapsed 
##   3.182   0.077   3.256
```

```r
system.time(res3 <- admm_lasso(x, y)$penalty(res1$lambda)$parallel()$fit())
```

```
##    user  system elapsed 
##   5.115   0.130   3.072
```

```r
system.time(res4 <- glmnet(x, y, nlambda = 20, alpha = 0.6))
```

```
##    user  system elapsed 
##   0.976   0.031   1.005
```

```r
system.time(res5 <- admm_enet(x, y)$penalty(res4$lambda, alpha = 0.6)$fit())
```

```
##    user  system elapsed 
##   4.514   0.068   4.579
```

```r
res2$niter
```

```
##  [1] 15 18 19 21 18 17 16 16 15 14 13 13 15 15 14 12 12
```

```r
range(coef(res1) - res2$beta)
```

```
## [1] -0.0001709674  0.0001663994
```

```r
res3$niter
```

```
##  [1] 23 21 21 27 20 19 18 17 16 15 14 14 14 13 13 12 11
```

```r
range(coef(res1) - res3$beta)
```

```
## [1] -0.0005266707  0.0002907920
```

```r
res5$niter
```

```
##  [1] 12 28 30 30 29 28 27 26 24 23 21 20 21 20 20 18 17
```

```r
range(coef(res4) - res5$beta)
```

```
## [1] -0.0001677783  0.0001671976
```

```r
# p > n
set.seed(123)
n <- 2000
p <- 10000
m <- 100
b <- matrix(c(runif(m), rep(0, p - m)))
x <- matrix(rnorm(n * p, sd = 2), n, p)
y <- x %*% b + rnorm(n)

system.time(res1 <- glmnet(x, y, nlambda = 20))
```

```
##    user  system elapsed 
##   0.699   0.037   0.735
```

```r
system.time(res2 <- admm_lasso(x, y)$penalty(res1$lambda)$fit())
```

```
##    user  system elapsed 
##   2.299   0.070   2.366
```

```r
system.time(res3 <- admm_lasso(x, y)$penalty(res1$lambda)$parallel()$fit())
```

```
##    user  system elapsed 
##   3.898   0.117   2.240
```

```r
system.time(res4 <- glmnet(x, y, nlambda = 20, alpha = 0.6))
```

```
##    user  system elapsed 
##   0.709   0.031   0.739
```

```r
system.time(res5 <- admm_enet(x, y)$penalty(res4$lambda, alpha = 0.6)$fit())
```

```
##    user  system elapsed 
##   2.402   0.064   2.464
```

```r
res2$niter
```

```
##  [1] 36 39 41 42 42 40 40 39 38 36 35 34 34 32 30 30 33 42 52 62
```

```r
range(coef(res1) - res2$beta)
```

```
## [1] -0.0009174825  0.0009320037
```

```r
res3$niter
```

```
##  [1] 42 43 51 51 51 51 49 49 47 46 44 43 43 41 41 38 37 47 60 74
```

```r
range(coef(res1) - res3$beta)
```

```
## [1] -0.000989717  0.001007029
```

```r
res5$niter
```

```
##  [1] 41 38 45 45 45 45 44 44 42 41 40 39 37 35 35 33 32 39 48 59
```

```r
range(coef(res4) - res5$beta)
```

```
## [1] -0.001009431  0.001127142
```

### Dantzig Selector


```r
library(ADMM)
library(flare)
# compute the full solution path, n > p
set.seed(123)
n <- 1000
p <- 200
m <- 10
b <- matrix(c(runif(m), rep(0, p - m)))
x <- matrix(rnorm(n * p, sd = 2), n, p)
y <- x %*% b + rnorm(n)

X <- scale(x)
Y <- y - mean(y)

system.time(res1 <- slim(X, Y, nlambda = 20, lambda.min.ratio = 0.01,
                         method = "dantzig"))
```

```
## Sparse Linear Regression with L1 Regularization.
## Dantzig selector with screening.
## 
## slim options summary: 
## 20 lambdas used:
##  [1] 1.9900 1.5600 1.2200 0.9610 0.7540 0.5920 0.4650 0.3650 0.2860 0.2250
## [11] 0.1760 0.1380 0.1090 0.0851 0.0668 0.0524 0.0412 0.0323 0.0253 0.0199
## Method = dantzig 
## Degree of freedom: 0 -----> 101 
## Runtime: 3.890011 secs
```

```
##    user  system elapsed 
##   4.008   0.001   4.006
```

```r
system.time(res2 <- admm_dantzig(X, Y)$penalty(nlambda = 20, lambda_min_ratio = 0.01)$
                                      fit())
```

```
##    user  system elapsed 
##   1.067   0.000   1.067
```

```r
range(res1$beta - res2$beta[-1, ])
```

```
## [1] -0.005694931  0.000530968
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

system.time(res1 <- rq.fit(x, y))
```

```
##    user  system elapsed 
##   2.593   0.002   2.594
```

```r
system.time(res2 <- rq.fit(x, y, method = "fn"))
```

```
##    user  system elapsed 
##   0.831   0.000   0.830
```

```r
system.time(res3 <- admm_lad(x, y, intercept = FALSE)$fit())
```

```
##    user  system elapsed 
##   0.345   0.000   0.345
```

```r
range(res1$coefficients - res2$coefficients)
```

```
## [1] -1.424183e-09  1.000354e-09
```

```r
range(res1$coefficients - res3$beta[-1])
```

```
## [1] -0.002771277  0.003095859
```

```r
set.seed(123)
n <- 5000
p <- 1000
b <- runif(p)
x <- matrix(rnorm(n * p, sd = 2), n, p)
y <- x %*% b + rnorm(n)

system.time(res1 <- rq.fit(x, y, method = "fn"))
```

```
##    user  system elapsed 
##  21.096   0.015  21.094
```

```r
system.time(res2 <- admm_lad(x, y, intercept = FALSE)$fit())
```

```
##    user  system elapsed 
##   5.378   0.016   5.390
```

```r
range(res1$coefficients - res2$beta[-1])
```

```
## [1] -0.001757139  0.001472339
```
