
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
n = 100
p = 10
x = matrix(rnorm(n * p), n, p)
y = rnorm(n)

x = scale(x) / sqrt((n - 1) / n)
y = y - mean(y)
y = y / sqrt(sum(y^2)) * sqrt(n)

fit = glmnet(x, y, standardize = FALSE, intercept = FALSE)
coef(fit, s = exp(-2), exact = TRUE)
```

```
## 11 x 1 sparse Matrix of class "dgCMatrix"
##                       1
## (Intercept)  .         
## V1           .         
## V2           .         
## V3          -0.02247571
## V4           0.05364369
## V5           .         
## V6           .         
## V7           .         
## V8           .         
## V9           .         
## V10          0.02347461
```

```r
admm_lasso(x, y, exp(-2))
```

```
## $coef
##  [1]  0.00000000  0.00000000 -0.02247579  0.05364286  0.00000000
##  [6]  0.00000000  0.00000000  0.00000000  0.00000000  0.02347472
## 
## $niter
## [1] 60
```

### Performance


```r
library(glmnet)
library(ADMM)
set.seed(123)
n = 500
p = 1000
x = matrix(rnorm(n * p), n, p)
y = rnorm(n)

x = scale(x) / sqrt((n - 1) / n)
y = y - mean(y)
y = y / sqrt(sum(y^2)) * sqrt(n)

system.time(
    res1 <- coef(glmnet(x, y, standardize = FALSE, intercept = FALSE),
                 s = exp(-2), exact = TRUE)
)
```

```
##    user  system elapsed 
##   0.387   0.001   0.387
```

```r
system.time(res2 <- admm_lasso(x, y, exp(-2)))
```

```
##    user  system elapsed 
##   0.329   0.000   0.329
```

```r
range(as.numeric(res1)[-1] - res2$coef)
```

```
## [1] -9.982021e-07  2.956360e-06
```
