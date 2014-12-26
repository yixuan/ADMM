admm_lasso = function(x, y, lambda = 1.0,
                      standardize = TRUE, intercept = TRUE,
                      opts = list())
{
    # default parameters
    opts_admm = list(maxit = 500L,
                     eps_abs = 1e-6,
                     eps_rel = 1e-6,
                     rho = 10 * lambda * nrow(x))
    # update from opts
    opts_admm[names(opts)] = opts
    
    .Call("admm_lasso", as.matrix(x), as.numeric(y), as.numeric(lambda),
          as.logical(standardize), as.logical(intercept),
          as.list(opts_admm), PACKAGE = "ADMM")
}