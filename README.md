
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
p <- 10
m <- 5
b <- matrix(c(runif(m), rep(0, p - m)))
x <- matrix(rnorm(n * p, mean = 1.2, sd = 2), n, p)
y <- 5 + x %*% b + rnorm(n)

## standardize = TRUE, intercept = TRUE
fit <- glmnet(x, y)
out_glmnet <- coef(fit, s = exp(-2), exact = TRUE)
out_admm <- admm_lasso(x, y, exp(-2))
data.frame(glmnet = as.numeric(out_glmnet), admm = out_admm$coef)
```

```
##         glmnet        admm
## 1   5.21035670  5.21035784
## 2   0.16405429  0.16405367
## 3   0.73050904  0.73050890
## 4   0.36518088  0.36518087
## 5   0.90493998  0.90493933
## 6   0.79807268  0.79807369
## 7   0.00000000  0.00000000
## 8   0.00000000  0.00000000
## 9  -0.03850441 -0.03850412
## 10  0.07174592  0.07174538
## 11  0.00000000  0.00000000
```

```r
## standardize = TRUE, intercept = FALSE
fit2 <- glmnet(x, y, intercept = FALSE)
out_glmnet2 <- coef(fit2, s = exp(-2), exact = TRUE)
out_admm2 <- admm_lasso(x, y, exp(-2), intercept = FALSE)
data.frame(glmnet = as.numeric(out_glmnet2), admm = out_admm2$coef)
```

```
##       glmnet      admm
## 1  0.0000000 0.0000000
## 2  0.5596375 0.5595453
## 3  1.1629401 1.1629051
## 4  0.6366979 0.6366831
## 5  1.2273086 1.2273788
## 6  0.9265080 0.9265526
## 7  0.4219237 0.4219422
## 8  0.2683371 0.2683575
## 9  0.2719755 0.2719306
## 10 0.5139927 0.5140359
## 11 0.3942982 0.3942893
```

```r
## standardize = FALSE, intercept = TRUE
fit3 <- glmnet(x, y, standardize = FALSE)
out_glmnet3 <- coef(fit3, s = exp(-2), exact = TRUE)
out_admm3 <- admm_lasso(x, y, exp(-2), standardize = FALSE)
data.frame(glmnet = as.numeric(out_glmnet3), admm = out_admm3$coef)
```

```
##          glmnet         admm
## 1   5.009113552  5.009127660
## 2   0.189741535  0.189728078
## 3   0.771295124  0.771291735
## 4   0.394785211  0.394778798
## 5   0.935271018  0.935275300
## 6   0.815624176  0.815622502
## 7   0.000000000  0.000000000
## 8   0.027578565  0.027583274
## 9  -0.070159605 -0.070159891
## 10  0.103312653  0.103314581
## 11  0.002570955  0.002572606
```

```r
## standardize = FALSE, intercept = FALSE
fit4 <- glmnet(x, y, standardize = FALSE, intercept = FALSE)
out_glmnet4 <- coef(fit4, s = exp(-2), exact = TRUE)
out_admm4 <- admm_lasso(x, y, exp(-2), standardize = FALSE, intercept = FALSE)
data.frame(glmnet = as.numeric(out_glmnet4), admm = out_admm4$coef)
```

```
##       glmnet      admm
## 1  0.0000000 0.0000000
## 2  0.5641629 0.5640922
## 3  1.1730348 1.1730096
## 4  0.6513611 0.6513492
## 5  1.2367039 1.2367607
## 6  0.9364155 0.9364455
## 7  0.4228142 0.4228216
## 8  0.2736202 0.2736319
## 9  0.2853958 0.2853626
## 10 0.5199241 0.5199556
## 11 0.3986638 0.3986609
```

### rho setting


```r
rho <- 1:200
niter <- sapply(rho, function(i) admm_lasso(x, y, exp(-2),
                                            opts = list(maxit = 3000L,
                                                        rho = i))$niter)
plot(rho, niter)
```

### Performance


```r
# high dimension, small sample
set.seed(123)
n <- 100
p <- 3000
m <- 10
b <- matrix(c(runif(m), rep(0, p - m)))
x <- matrix(rnorm(n * p, sd = 2), n, p)
y <- x %*% b + rnorm(n)

system.time(
    res1 <- coef(glmnet(x, y), s = exp(-2), exact = TRUE)
)
```

```
##    user  system elapsed 
##   0.245   0.004   0.249
```

```r
system.time(res2 <- admm_lasso(x, y, exp(-2), opts = list(maxit = 1000)))
```

```
##    user  system elapsed 
##   0.194   0.000   0.194
```

```r
range(as.numeric(res1) - res2$coef)
```

```
## [1] -0.02480136  0.02229468
```

### rho setting


```r
rho <- 1:200
niter <- sapply(rho, function(i) admm_lasso(x, y, exp(-2),
                                            opts = list(maxit = 3000L,
                                                        rho = i))$niter)
plot(rho, niter)
```
