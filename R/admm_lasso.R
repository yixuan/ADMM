admm_lasso = function(x, y, lambda = NULL,
                      nlambda = 100,
                      lambda_min_ratio = ifelse(nrow(x) < ncol(x), 0.01, 0.0001),
                      standardize = TRUE, intercept = TRUE,
                      opts = list())
{
    # default parameters
    opts_admm = list(maxit = 10000L,
                     eps_abs = 1e-5,
                     eps_rel = 1e-5,
                     rho_ratio = 10)
    # update from opts
    opts_admm[names(opts)] = opts
    
    .Call("admm_lasso", as.matrix(x), as.numeric(y), as.numeric(lambda),
          as.integer(nlambda), as.numeric(lambda_min_ratio),
          as.logical(standardize), as.logical(intercept),
          as.list(opts_admm), PACKAGE = "ADMM")
}


admm_parlasso = function(x, y, lambda = NULL,
                         nlambda = 100,
                         lambda_min_ratio = ifelse(nrow(x) < ncol(x), 0.01, 0.0001),
                         standardize = TRUE, intercept = TRUE,
                         opts = list())
{
    # default parameters
    opts_admm = list(maxit = 10000L,
                     eps_abs = 1e-5,
                     eps_rel = 1e-5,
                     rho_ratio = 10)
    # update from opts
    opts_admm[names(opts)] = opts
    
    .Call("admm_parlasso", as.list(x), as.list(y),
          as.integer(length(unlist(y))), as.integer(ncol(x[[1]])),
          as.numeric(lambda), as.integer(nlambda), as.numeric(lambda_min_ratio),
          as.logical(standardize), as.logical(intercept),
          as.list(opts_admm), PACKAGE = "ADMM")
}
