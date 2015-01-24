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
                     rho_ratio = 0.1)
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
                         standardize = TRUE, intercept = TRUE, nthread = 2,
                         opts = list())
{
    # default parameters
    opts_admm = list(maxit = 10000L,
                     eps_abs = 1e-5,
                     eps_rel = 1e-5,
                     rho_ratio = 10)
    # update from opts
    opts_admm[names(opts)] = opts
    
    if(nthread < 1)
        nthread = 1
    if(nthread > 10)
        nthread = 10
    
    .Call("admm_parlasso", as.matrix(x), as.numeric(y), as.numeric(lambda),
          as.integer(nlambda), as.numeric(lambda_min_ratio),
          as.logical(standardize), as.logical(intercept), as.integer(nthread),
          as.list(opts_admm), PACKAGE = "ADMM")
}
