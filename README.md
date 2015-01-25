
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
out_paradmm <- admm_parlasso(x, y, exp(-2))
data.frame(glmnet = as.numeric(out_glmnet),
           admm = as.numeric(out_admm$beta),
           paradmm = out_paradmm$beta)
```

```
##         glmnet        admm     paradmm
## 1   5.21035670  5.21035632  5.21035442
## 2   0.16405429  0.16405398  0.16404909
## 3   0.73050904  0.73050971  0.73051127
## 4   0.36518088  0.36518165  0.36518331
## 5   0.90493998  0.90493967  0.90494107
## 6   0.79807268  0.79807201  0.79806791
## 7   0.00000000  0.00000000  0.00000000
## 8   0.00000000  0.00000000  0.00000000
## 9  -0.03850441 -0.03850399 -0.03849909
## 10  0.07174592  0.07174560  0.07174586
## 11  0.00000000  0.00000000  0.00000000
```

```r
## standardize = TRUE, intercept = FALSE
fit2 <- glmnet(x, y, intercept = FALSE)
out_glmnet2 <- coef(fit2, s = exp(-2), exact = TRUE)
out_admm2 <- admm_lasso(x, y, exp(-2), intercept = FALSE)
out_paradmm2 <- admm_parlasso(x, y, exp(-2), intercept = FALSE)
data.frame(glmnet = as.numeric(out_glmnet2),
           admm = as.numeric(out_admm2$beta),
           paradmm = out_paradmm2$beta)
```

```
##       glmnet      admm   paradmm
## 1  0.0000000 0.0000000 0.0000000
## 2  0.5596375 0.5595634 0.5595830
## 3  1.1629401 1.1629167 1.1629218
## 4  0.6366979 0.6366834 0.6366829
## 5  1.2273086 1.2273693 1.2273641
## 6  0.9265080 0.9265326 0.9265490
## 7  0.4219237 0.4219594 0.4219465
## 8  0.2683371 0.2683839 0.2683643
## 9  0.2719755 0.2719091 0.2719310
## 10 0.5139927 0.5140392 0.5140184
## 11 0.3942982 0.3942590 0.3942739
```

```r
## standardize = FALSE, intercept = TRUE
fit3 <- glmnet(x, y, standardize = FALSE)
out_glmnet3 <- coef(fit3, s = exp(-2), exact = TRUE)
out_admm3 <- admm_lasso(x, y, exp(-2), standardize = FALSE)
out_paradmm3 <- admm_parlasso(x, y, exp(-2), standardize = FALSE)
data.frame(glmnet = as.numeric(out_glmnet3),
           admm = as.numeric(out_admm3$beta),
           paradmm = out_paradmm3$beta)
```

```
##          glmnet         admm      paradmm
## 1   5.009113552  5.009125185  5.009079758
## 2   0.189741535  0.189728954  0.189736277
## 3   0.771295124  0.771292560  0.771304644
## 4   0.394785211  0.394778051  0.394794990
## 5   0.935271018  0.935274528  0.935292090
## 6   0.815624176  0.815623386  0.815621582
## 7   0.000000000  0.000000000  0.000000000
## 8   0.027578565  0.027583341  0.027577828
## 9  -0.070159605 -0.070159501 -0.070162916
## 10  0.103312653  0.103315006  0.103319651
## 11  0.002570955  0.002572667  0.002563695
```

```r
## standardize = FALSE, intercept = FALSE
fit4 <- glmnet(x, y, standardize = FALSE, intercept = FALSE)
out_glmnet4 <- coef(fit4, s = exp(-2), exact = TRUE)
out_admm4 <- admm_lasso(x, y, exp(-2), standardize = FALSE, intercept = FALSE)
out_paradmm4 <- admm_parlasso(x, y, exp(-2), standardize = FALSE, intercept = FALSE)
data.frame(glmnet = as.numeric(out_glmnet4),
           admm = as.numeric(out_admm4$beta),
           paradmm = out_paradmm4$beta)
```

```
##       glmnet      admm   paradmm
## 1  0.0000000 0.0000000 0.0000000
## 2  0.5641629 0.5641288 0.5641011
## 3  1.1730348 1.1730210 1.1730160
## 4  0.6513611 0.6513423 0.6513497
## 5  1.2367039 1.2367448 1.2367549
## 6  0.9364155 0.9364364 0.9364465
## 7  0.4228142 0.4228507 0.4228251
## 8  0.2736202 0.2736407 0.2736438
## 9  0.2853958 0.2853461 0.2853645
## 10 0.5199241 0.5199428 0.5199506
## 11 0.3986638 0.3986382 0.3986572
```

### rho setting


```r
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
##   0.033   0.001   0.034
```

```r
system.time(res2 <- admm_lasso(x, y))
```

```
##    user  system elapsed 
##   0.644   0.000   0.643
```

```r
system.time(res3 <- admm_parlasso(x, y))
```

```
##    user  system elapsed 
##   6.536   0.013   3.273
```

```r
system.time(res4 <- admm_parlasso(x, y, nthread = 4))
```

```
##    user  system elapsed 
##  11.870   0.011   2.968
```

```r
# p > n, single lambda
set.seed(123)
n <- 1000
p <- 3000
m <- 10
b <- matrix(c(runif(m), rep(0, p - m)))
x <- matrix(rnorm(n * p, sd = 2), n, p)
y <- x %*% b + rnorm(n)

system.time(res1 <- coef(glmnet(x, y), s = exp(-2), exact = TRUE))
```

```
##    user  system elapsed 
##   0.781   0.006   0.786
```

```r
system.time(res2 <- admm_lasso(x, y, exp(-2)))
```

```
##    user  system elapsed 
##   1.031   0.008   1.039
```

```r
system.time(res3 <- admm_parlasso(x, y, exp(-2)))
```

```
##    user  system elapsed 
##  17.823   0.022   8.931
```

```r
system.time(res4 <- admm_parlasso(x, y, exp(-2), nthread = 4))
```

```
##    user  system elapsed 
##  38.027   0.032   9.523
```

```r
res2$niter
```

```
## [1] 83
```

```r
range(as.numeric(res1) - as.numeric(res2$beta))
```

```
## [1] -8.424337e-06  1.323267e-05
```

```r
res3$niter
```

```
## [1] 2343
```

```r
range(as.numeric(res1) - res3$beta)
```

```
## [1] -0.0013210392  0.0009414977
```

```r
res4$niter
```

```
## [1] 2423
```

```r
range(as.numeric(res1) - res4$beta)
```

```
## [1] -0.0017592078  0.0003463298
```

### rho setting


```r
rho_ratio <- exp(seq(log(0.1), log(10), length.out = 100))
niter <- sapply(rho_ratio,
    function(r) admm_lasso(x, y, lambda, opts = list(rho_ratio = r))$niter
)
plot(rho_ratio, niter)
```
