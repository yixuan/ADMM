
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
data.frame(glmnet = as.numeric(out_glmnet), admm = out_admm$beta)
```

```
##         glmnet        admm
## 1   5.21035670  5.21036905
## 2   0.16405429  0.16404245
## 3   0.73050904  0.73050566
## 4   0.36518088  0.36517203
## 5   0.90493998  0.90494124
## 6   0.79807268  0.79806520
## 7   0.00000000  0.00000000
## 8   0.00000000  0.00000000
## 9  -0.03850441 -0.03850049
## 10  0.07174592  0.07175680
## 11  0.00000000  0.00000000
```

```r
## standardize = TRUE, intercept = FALSE
fit2 <- glmnet(x, y, intercept = FALSE)
out_glmnet2 <- coef(fit2, s = exp(-2), exact = TRUE)
out_admm2 <- admm_lasso(x, y, exp(-2), intercept = FALSE)
data.frame(glmnet = as.numeric(out_glmnet2), admm = out_admm2$beta)
```

```
##       glmnet      admm
## 1  0.0000000 0.0000000
## 2  0.5596375 0.5565264
## 3  1.1629401 1.1590343
## 4  0.6366979 0.6319627
## 5  1.2273086 1.2264041
## 6  0.9265080 0.9262898
## 7  0.4219237 0.4239971
## 8  0.2683371 0.2721587
## 9  0.2719755 0.2709847
## 10 0.5139927 0.5178473
## 11 0.3942982 0.3944803
```

```r
## standardize = FALSE, intercept = TRUE
fit3 <- glmnet(x, y, standardize = FALSE)
out_glmnet3 <- coef(fit3, s = exp(-2), exact = TRUE)
out_admm3 <- admm_lasso(x, y, exp(-2), standardize = FALSE)
data.frame(glmnet = as.numeric(out_glmnet3), admm = out_admm3$beta)
```

```
##          glmnet         admm
## 1   5.009113552  5.009130538
## 2   0.189741535  0.189723274
## 3   0.771295124  0.771284464
## 4   0.394785211  0.394788387
## 5   0.935271018  0.935280112
## 6   0.815624176  0.815620371
## 7   0.000000000  0.000000000
## 8   0.027578565  0.027567914
## 9  -0.070159605 -0.070150332
## 10  0.103312653  0.103305638
## 11  0.002570955  0.002587733
```

```r
## standardize = FALSE, intercept = FALSE
fit4 <- glmnet(x, y, standardize = FALSE, intercept = FALSE)
out_glmnet4 <- coef(fit4, s = exp(-2), exact = TRUE)
out_admm4 <- admm_lasso(x, y, exp(-2), standardize = FALSE, intercept = FALSE)
data.frame(glmnet = as.numeric(out_glmnet4), admm = out_admm4$beta)
```

```
##       glmnet      admm
## 1  0.0000000 0.0000000
## 2  0.5641629 0.5623697
## 3  1.1730348 1.1722307
## 4  0.6513611 0.6492344
## 5  1.2367039 1.2367846
## 6  0.9364155 0.9360529
## 7  0.4228142 0.4254555
## 8  0.2736202 0.2774644
## 9  0.2853958 0.2843470
## 10 0.5199241 0.5227374
## 11 0.3986638 0.3990461
```

### rho setting


```r
rho <- 1:200
lambda <- exp(-2)
rho_ratio <- rho / n / lambda
niter <- sapply(rho,
    function(i) admm_lasso(x, y, lambda, opts = list(rho_ratio = i))$niter
)
plot(rho, niter)
```

### Performance


```r
# compute the full solution path, n > p
set.seed(123)
n <- 1000
p <- 300
m <- 10
b <- matrix(c(runif(m), rep(0, p - m)))
x <- matrix(rnorm(n * p, sd = 2), n, p)
y <- x %*% b + rnorm(n)

system.time(res1 <- glmnet(x, y))
```

```
##    user  system elapsed 
##    0.06    0.00    0.06
```

```r
system.time(res2 <- admm_lasso(x, y))
```

```
##    user  system elapsed 
##    0.33    0.00    0.33
```

```r
# p > n, single lambda
set.seed(123)
n <- 100
p <- 3000
m <- 10
b <- matrix(c(runif(m), rep(0, p - m)))
x <- matrix(rnorm(n * p, sd = 2), n, p)
y <- x %*% b + rnorm(n)

system.time(res1 <- coef(glmnet(x, y), s = exp(-2), exact = TRUE))
```

```
##    user  system elapsed 
##     0.2     0.0     0.2
```

```r
system.time(res2 <- admm_lasso(x, y, exp(-2)))
```

```
##    user  system elapsed 
##    0.47    0.00    0.47
```

```r
res2$niter
```

```
## [1] 1546
```

```r
range(as.numeric(res1) - res2$beta)
```

```
## [1] -0.006212990  0.006598132
```

### rho setting


```r
rho <- 1:200
lambda <- exp(-2)
rho_ratio <- rho / n / lambda
niter <- sapply(rho,
    function(i) admm_lasso(x, y, lambda, opts = list(rho_ratio = i))$niter
)
plot(rho, niter)
```
