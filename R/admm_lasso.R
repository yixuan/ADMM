## Class to describe a Lasso model
ADMM_Lasso = setRefClass("ADMM_Lasso",
    fields = list(x = "matrix",
                  y = "numeric",
                  intercept = "logical",
                  standardize = "logical",
                  lambda = "numeric",
                  nlambda = "integer",
                  lambda_min_ratio = "numeric",
                  nthread = "integer",
                  maxit = "integer",
                  eps_abs = "numeric",
                  eps_rel = "numeric",
                  rho_ratio = "numeric")
)

## Class to store fitting results of Lasso model
ADMM_Lasso_fit = setRefClass("ADMM_Lasso_fit",
    fields = list(lambda = "numeric",
                  beta = "dgCMatrix",
                  niter = "integer")
)





##### Member functions of ADMM_Lasso #####

## Initialize fields including default values
ADMM_Lasso$methods(
    initialize = function(x, y, intercept = TRUE, standardize = TRUE, ...)
    {
        if(nrow(x) != length(y))
            stop("nrow(x) should be equal to length(y)")

        .self$x = as.matrix(x)
        .self$y = as.numeric(y)
        .self$intercept = as.logical(intercept)
        .self$standardize = as.logical(standardize)
        .self$lambda = numeric(0)
        .self$nlambda = 100L
        .self$lambda_min_ratio = ifelse(nrow(x) < ncol(x), 0.01, 0.0001)
        .self$nthread = 1L
        .self$maxit = 10000L
        .self$eps_abs = 1e-5
        .self$eps_rel = 1e-5
        .self$rho_ratio = 0.1
    }
)

## Print off ADMM_Lasso object
ADMM_Lasso$methods(
    show = function()
    {
        cat("ADMM Lasso model\n\n")
        cat(sprintf("$x: <%d x %d> matrix\n", nrow(.self$x), ncol(.self$x)))
        cat(sprintf("$y: <%d x 1> vector\n", length(.self$y)))

        fields = setdiff(names(ADMM_Lasso$fields()), c("x", "y"))
        for(field in fields)
            cat("$", field, ": ", .self$field(field), "\n", sep = "")
    }
)

## Set up penalty parameters
ADMM_Lasso$methods(
    penalty = function(lambda = NULL, nlambda = 100, lambda_min_ratio, ...)
    {
        lambda_val = sort(as.numeric(lambda), decreasing = TRUE)
        if(any(lambda_val <= 0))
            stop("lambda must be positive")
        
        if(nlambda[1] <= 0)
            stop("nlambda must be a positive integer")
        
        if(missing(lambda_min_ratio))
            lmr_val = ifelse(nrow(.self$x) < ncol(.self$x), 0.01, 0.0001)
        else
            lmr_val = as.numeric(lambda_min_ratio)
        
        if(lmr_val >= 1 | lmr_val <= 0)
            stop("lambda_min_ratio must be within (0, 1)")
        
        .self$lambda = lambda_val
        .self$nlambda = as.integer(nlambda[1])
        .self$lambda_min_ratio = lmr_val
        
        invisible(.self)
    }
)

## Specify parallel computing
ADMM_Lasso$methods(
    parallel = function(nthread = 2, ...)
    {
        nthread_val = as.integer(nthread)
        if(nthread_val < 1)
            nthread_val = 1L
        if(nthread_val >= ncol(.self$x) / 5)
            stop("nthread cannot exceed ncol(x)/5")
        
        .self$nthread = nthread_val
        
        invisible(.self)
    }
)

## Specify additional parameters
ADMM_Lasso$methods(
    opts = function(maxit = 10000, eps_abs = 1e-5, eps_rel = 1e-5,
                    rho_ratio = 0.1, ...)
    {
        if(maxit <= 0)
            stop("maxit should be positive")
        if(eps_abs < 0 | eps_rel < 0)
            stop("eps_abs and eps_rel should be nonnegative")
        if(rho_ratio <= 0)
            stop("rho_ratio should be positive")
        
        .self$maxit = as.integer(maxit)
        .self$eps_abs = as.numeric(eps_abs)
        .self$eps_rel = as.numeric(eps_rel)
        .self$rho_ratio = as.numeric(rho_ratio)
        
        invisible(.self)
    }
)

## Fit model and conduct the computing
ADMM_Lasso$methods(
    fit = function(...)
    {
        res = if(.self$nthread <= 1)
            .Call("admm_lasso", .self$x, .self$y, .self$lambda,
                  .self$nlambda, .self$lambda_min_ratio,
                  .self$standardize, .self$intercept,
                  list(maxit = .self$maxit,
                       eps_abs = .self$eps_abs,
                       eps_rel = .self$eps_rel,
                       rho_ratio = .self$rho_ratio),
                  PACKAGE = "ADMM")
        else
            .Call("admm_parlasso", .self$x, .self$y, .self$lambda,
                  .self$nlambda, .self$lambda_min_ratio,
                  .self$standardize, .self$intercept,
                  .self$nthread, FALSE,
                  list(maxit = .self$maxit,
                       eps_abs = .self$eps_abs,
                       eps_rel = .self$eps_rel,
                       rho_ratio = .self$rho_ratio),
                  PACKAGE = "ADMM")
        do.call(ADMM_Lasso_fit, res)
    }
)





##### Member functions of ADMM_Lasso_fit #####

## Print off ADMM_Lasso_fit object
ADMM_Lasso_fit$methods(
    show = function()
    {
        cat("ADMM Lasso fitting result\n\n")
        cat("$lambda\n")
        print(.self$lambda)
        cat("\n")
        cat("$beta\n")
        cat(sprintf("<%d x %d> sparse matrix\n", nrow(.self$beta), ncol(.self$beta)))
        cat("\n")
        cat("$niter\n")
        print(.self$niter)
    }
)

## Print off ADMM_Lasso_fit object
ADMM_Lasso_fit$methods(
    plot = function()
    {
        nlambda = length(.self$lambda)
        # If we only have one lambda we cannot create a path plot
        if(nlambda < 2)
            stop("need to have at least two lambda values")
        
        loglambda = log(.self$lambda)
        # Exclude variables that have zero coefficients for all lambdas
        rows_inc = apply(.self$beta, 1, function(x) any(x != 0))
        # Exclude intercept
        rows_inc[1] = FALSE
        mat = t(as.matrix(.self$beta[rows_inc, ]))
        nvar = ncol(mat)
        dat = data.frame(loglambda = rep(loglambda, nvar),
                         varid = rep(1:nvar, each = nlambda),
                         coef = as.numeric(mat))
        ggplot(dat, aes(x = loglambda, y = coef, color = factor(varid))) +
            geom_line(aes(group = varid)) +
            xlab(expression(log(lambda))) +
            ylab("Coefficients") +
            ggtitle("Solution path") +
            guides(color = FALSE)
    }
)





## Calculate the spectral radius of x'x.
## In this case it is the largest eigenvalue of x'x,
## and also the square of the largest singular value of x.
## This function will be called inside C++.
.spectral_radius = function(x)
{
    svds(x, k = 1, nu = 0, nv = 0,
         opts = list(ncv = 5, tol = 1.0, maxitr = 100))$d^2
}

## Exported function to fit ADMM Lasso
admm_lasso = function(x, y, intercept = TRUE, standardize = TRUE, ...)
{
    ADMM_Lasso(x, y, intercept, standardize, ...)
}
