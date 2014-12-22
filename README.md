
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
x <- matrix(rnorm(n * p, sd = 2), n, p)
y <- x %*% b + rnorm(n)

## non-standardized
fit <- glmnet(x, y, standardize = FALSE, intercept = FALSE)
out_glmnet <- coef(fit, s = exp(-2), exact = TRUE)
out_admm <- admm_lasso(x, y, exp(-2), rho = 10)
data.frame(glmnet = out_glmnet[-1], admm = out_admm$coef)
```

```
##          glmnet         admm
## 1   0.191276333  0.191273320
## 2   0.768951956  0.768950559
## 3   0.400495704  0.400495040
## 4   0.937123951  0.937123468
## 5   0.827245364  0.827245585
## 6   0.000000000  0.000000000
## 7   0.028758698  0.028758960
## 8  -0.067186179 -0.067186075
## 9   0.095128164  0.095128163
## 10  0.006433896  0.006434543
```

```r
## standardized
# use the "standardize" parameter provided by glmnet
fit1 <- glmnet(x, y, standardize = TRUE, intercept = TRUE)
out_glmnet1 <- coef(fit1, s = exp(-2), exact = TRUE)
# standardize data by yourself
x0 <- scale(x) / sqrt((n - 1) / n)
y0 <- c(scale(y)) / sqrt((n - 1) / n)
# double check
colSums(x0^2)
```

```
##  [1] 100 100 100 100 100 100 100 100 100 100
```

```r
sum(y0^2)
```

```
## [1] 100
```

```r
# scaling factor
scalex <- apply(x, 2, function(x) sd(x) * sqrt((n - 1) / n))
scaley <- sd(y) * sqrt((n - 1) / n)

fit2 <- glmnet(x0, y0, standardize = FALSE, intercept = FALSE)
out_glmnet2 <- coef(fit2, s = exp(-2) / scaley, exact = TRUE)[-1] * scaley / scalex
out_admm1 <- admm_lasso(x0, y0, exp(-2) / scaley, rho = 10)$coef * scaley / scalex

data.frame(glmnet_std = out_glmnet1[-1],
           glmnet_mystd = out_glmnet2,
           admm_mystd = out_admm1)
```

```
##     glmnet_std glmnet_mystd  admm_mystd
## 1   0.16405429   0.16405429  0.16405192
## 2   0.73050904   0.73050904  0.73050839
## 3   0.36518088   0.36518088  0.36517993
## 4   0.90493998   0.90493998  0.90493958
## 5   0.79807268   0.79807268  0.79807217
## 6   0.00000000   0.00000000  0.00000000
## 7   0.00000000   0.00000000  0.00000000
## 8  -0.03850441  -0.03850441 -0.03850437
## 9   0.07174592   0.07174592  0.07174680
## 10  0.00000000   0.00000000  0.00000000
```

### rho setting


```r
rho <- 1:200
niter <- sapply(rho, function(i) admm_lasso(x, y, exp(-2), maxit = 3000L, rho = i)$niter)
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
    res1 <- coef(glmnet(x, y, standardize = FALSE, intercept = FALSE),
                 s = exp(-2), exact = TRUE)
)
```

```
##    user  system elapsed 
##   0.227   0.002   0.230
```

```r
system.time(res2 <- admm_lasso(x, y, exp(-2), maxit = 1000))
```

```
##    user  system elapsed 
##   0.193   0.000   0.192
```

```r
range(as.numeric(res1)[-1] - res2$coef)
```

```
## [1] -0.04689106  0.05545088
```

### rho setting


```r
rho <- 1:200
niter <- sapply(rho, function(i) admm_lasso(x, y, exp(-2), maxit = 3000L, rho = i)$niter)
plot(rho, niter)
```
