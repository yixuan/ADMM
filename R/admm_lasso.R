admm_lasso = function(x, y, lambda = NULL,
                      nlambda = 100,
                      lambda_min_ratio = ifelse(nrow(x) < ncol(x), 0.01, 0.0001),
                      standardize = TRUE, intercept = TRUE,
                      opts = list())
{
    # default parameters
    opts_admm = list(maxit = 500L,
                     eps_abs = 1e-6,
                     eps_rel = 1e-6,
                     rho_ratio = 10)
    # update from opts
    opts_admm[names(opts)] = opts
    
    .Call("admm_lasso", as.matrix(x), as.numeric(y), as.numeric(lambda),
          as.integer(nlambda), as.numeric(lambda_min_ratio),
          as.logical(standardize), as.logical(intercept),
          as.list(opts_admm), PACKAGE = "ADMM")
}