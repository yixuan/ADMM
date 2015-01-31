
### ADMM

`ADMM` is an R package to solve Lasso-like problems using the ADMM algorithm.

### Comparison

Comparing `ADMM` with `glmnet`


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

## standardize = TRUE, intercept = TRUE
fit <- glmnet(x, y)
out_glmnet <- coef(fit, s = exp(-2), exact = TRUE)
out_admm <- admm_lasso(x, y, exp(-2))
out_paradmm <- admm_parlasso(x, y, exp(-2))
data.frame(glmnet = as.numeric(out_glmnet),
           admm = as.numeric(out_admm$beta),
           paradmm = as.numeric(out_paradmm$beta))
```

```
##          glmnet         admm      paradmm
## 1   5.357410774  5.357429137  5.357389792
## 2   0.178916019  0.178929108  0.178906561
## 3   0.683606818  0.683613545  0.683607644
## 4   0.310518550  0.310532229  0.310512272
## 5   0.861034415  0.861037854  0.861033057
## 6   0.879797912  0.879796568  0.879798921
## 7   0.007854581  0.007845977  0.007854352
## 8   0.000000000  0.000000000  0.000000000
## 9   0.000000000  0.000000000  0.000000000
## 10  0.023462980  0.023464519  0.023457740
## 11  0.010952896  0.010936241  0.010969479
## 12  0.000000000  0.000000000  0.000000000
## 13 -0.003800159 -0.003796027 -0.003804590
## 14  0.000000000  0.000000000  0.000000000
## 15  0.094591923  0.094574901  0.094606845
## 16  0.000000000  0.000000000  0.000000000
## 17  0.000000000  0.000000000  0.000000000
## 18  0.000000000  0.000000000  0.000000000
## 19  0.000000000  0.000000000  0.000000000
## 20 -0.002916255 -0.002931312 -0.002903598
## 21  0.000000000  0.000000000  0.000000000
```

```r
## standardize = TRUE, intercept = FALSE
fit2 <- glmnet(x, y, intercept = FALSE)
out_glmnet2 <- coef(fit2, s = exp(-2), exact = TRUE)
out_admm2 <- admm_lasso(x, y, exp(-2), intercept = FALSE)
out_paradmm2 <- admm_parlasso(x, y, exp(-2), intercept = FALSE)
data.frame(glmnet = as.numeric(out_glmnet2),
           admm = as.numeric(out_admm2$beta),
           paradmm = as.numeric(out_paradmm2$beta))
```

```
##       glmnet      admm   paradmm
## 1  0.0000000 0.0000000 0.0000000
## 2  0.3989404 0.3988151 0.3988076
## 3  0.9323307 0.9322342 0.9322227
## 4  0.2960961 0.2960428 0.2960435
## 5  1.1284730 1.1283248 1.1282866
## 6  0.9610648 0.9609833 0.9609607
## 7  0.4482071 0.4481891 0.4482293
## 8  0.1993521 0.1993898 0.1994147
## 9  0.1646473 0.1647658 0.1647635
## 10 0.2888623 0.2889599 0.2889548
## 11 0.2068456 0.2070895 0.2070774
## 12 0.0000000 0.0000000 0.0000000
## 13 0.1438574 0.1440456 0.1440587
## 14 0.2160424 0.2160289 0.2160362
## 15 0.3571222 0.3571956 0.3572313
## 16 0.1929267 0.1929354 0.1929303
## 17 0.1041142 0.1040767 0.1040694
## 18 0.2681668 0.2679948 0.2680214
## 19 0.2944246 0.2945293 0.2945005
## 20 0.2076512 0.2076185 0.2076295
## 21 0.2031142 0.2031232 0.2031060
```

```r
## standardize = FALSE, intercept = TRUE
fit3 <- glmnet(x, y, standardize = FALSE)
out_glmnet3 <- coef(fit3, s = exp(-2), exact = TRUE)
out_admm3 <- admm_lasso(x, y, exp(-2), standardize = FALSE)
out_paradmm3 <- admm_parlasso(x, y, exp(-2), standardize = FALSE)
data.frame(glmnet = as.numeric(out_glmnet3),
           admm = as.numeric(out_admm3$beta),
           paradmm = as.numeric(out_paradmm3$beta))
```

```
##          glmnet         admm      paradmm
## 1   5.113191236  5.113051416  5.113067967
## 2   0.206020869  0.205995013  0.205992573
## 3   0.722296881  0.722301917  0.722301433
## 4   0.340908906  0.340875004  0.340878561
## 5   0.889244299  0.889269297  0.889276222
## 6   0.896901065  0.896890181  0.896882804
## 7   0.042276131  0.042324502  0.042311520
## 8   0.022898533  0.022926489  0.022927028
## 9   0.000000000  0.000000000  0.000000000
## 10  0.060925430  0.060955846  0.060959676
## 11  0.038114821  0.038140184  0.038141930
## 12 -0.009097427 -0.009121254 -0.009127955
## 13 -0.037922334 -0.037932226 -0.037931931
## 14  0.000000000  0.000000000  0.000000000
## 15  0.105289570  0.105309433  0.105306255
## 16  0.000000000  0.000000000  0.000000000
## 17  0.000000000  0.000000000  0.000000000
## 18  0.000000000  0.000000000  0.000000000
## 19  0.000000000  0.000000000  0.000000000
## 20 -0.022807787 -0.022773906 -0.022774570
## 21  0.000000000  0.000000000  0.000000000
```

```r
## standardize = FALSE, intercept = FALSE
fit4 <- glmnet(x, y, standardize = FALSE, intercept = FALSE)
out_glmnet4 <- coef(fit4, s = exp(-2), exact = TRUE)
out_admm4 <- admm_lasso(x, y, exp(-2), standardize = FALSE, intercept = FALSE)
out_paradmm4 <- admm_parlasso(x, y, exp(-2), standardize = FALSE, intercept = FALSE)
data.frame(glmnet = as.numeric(out_glmnet4),
           admm = as.numeric(out_admm4$beta),
           paradmm = as.numeric(out_paradmm4$beta))
```

```
##       glmnet      admm   paradmm
## 1  0.0000000 0.0000000 0.0000000
## 2  0.3968864 0.3967705 0.3967551
## 3  0.9360224 0.9359327 0.9359360
## 4  0.3103771 0.3104839 0.3104697
## 5  1.1392292 1.1389702 1.1390021
## 6  0.9728917 0.9728621 0.9728646
## 7  0.4432684 0.4433274 0.4433175
## 8  0.2018397 0.2019532 0.2019419
## 9  0.1775602 0.1779181 0.1779146
## 10 0.2942154 0.2944545 0.2944637
## 11 0.2068157 0.2071670 0.2071648
## 12 0.0000000 0.0000000 0.0000000
## 13 0.1560136 0.1561227 0.1561106
## 14 0.2079899 0.2080986 0.2081161
## 15 0.3593078 0.3593453 0.3593390
## 16 0.1934218 0.1933899 0.1933922
## 17 0.1078094 0.1077042 0.1077189
## 18 0.2658637 0.2658593 0.2658382
## 19 0.2955308 0.2952677 0.2952758
## 20 0.2148104 0.2147641 0.2147633
## 21 0.2236823 0.2236053 0.2236098
```

### rho setting


```r
lambda <- exp(-2)
rho_ratio <- exp(seq(log(0.1), log(10), length.out = 100))
niter <- sapply(rho_ratio,
    function(r) admm_lasso(x, y, lambda, opts = list(rho_ratio = r))$niter
)
plot(rho_ratio, niter)
```

### Performance


```r
# compute the full solution path, n > p
set.seed(123)
n <- 50000
p <- 1000
m <- 100
b <- matrix(c(runif(m), rep(0, p - m)))
x <- matrix(rnorm(n * p, sd = 2), n, p)
y <- x %*% b + rnorm(n)

system.time(res1 <- glmnet(x, y, nlambda = 20))
```

```
##    user  system elapsed 
##   1.784   0.085   1.868
```

```r
system.time(res2 <- admm_lasso(x, y, nlambda = 20))
```

```
##    user  system elapsed 
##  11.174   0.167  11.334
```

```r
system.time(res3 <- admm_parlasso(x, y, nlambda = 20))
```

```
##    user  system elapsed 
##  12.924   0.313   8.029
```

```r
nlam = length(res1$lambda)
res2$niter
```

```
##  [1] 14 21 22 22 22 21 21 20 19 18 17 16 17 16 17 16 14 13 12 11
```

```r
range(coef(res1) - res2$beta[, 1:nlam])
```

```
## [1] -8.189371e-05  9.953476e-05
```

```r
res3$niter
```

```
##  [1] 21 19 22 19 18 17 17 16 15 14 13 12 12 11 11 10 10  9  8  7
```

```r
range(coef(res1) - res3$beta[, 1:nlam])
```

```
## [1] -0.0001420500  0.0002010485
```

```r
# p > n, single lambda
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
##   0.542   0.027   0.568
```

```r
system.time(res2 <- admm_lasso(x, y, nlambda = 20))
```

```
##    user  system elapsed 
##   2.147   0.067   2.213
```

```r
system.time(res3 <- admm_parlasso(x, y, nlambda = 20))
```

```
##    user  system elapsed 
##   3.678   0.104   2.115
```

```r
nlam = length(res1$lambda)
res2$niter
```

```
##  [1] 36 39 41 42 42 40 40 39 38 36 35 34 34 32 30 30 33 42 52 62
```

```r
range(coef(res1) - res2$beta[, 1:nlam])
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
range(coef(res1) - res3$beta[, 1:nlam])
```

```
## [1] -0.000989717  0.001007029
```

### rho setting


```r
lambda <- exp(-2)
rho_ratio <- exp(seq(log(0.1), log(10), length.out = 100))
niter <- sapply(rho_ratio,
    function(r) admm_lasso(x, y, lambda, opts = list(rho_ratio = r))$niter
)
plot(rho_ratio, niter)
```
