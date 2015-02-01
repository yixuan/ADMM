admm_lad = function(x, y, intercept = TRUE, opts = list())
{
    # default parameters
    opts_admm = list(maxit = 10000L,
                     eps_abs = 1e-5,
                     eps_rel = 1e-5,
                     rho_ratio = 1)
    # update from opts
    opts_admm[names(opts)] = opts
    
    .Call("admm_lad", as.matrix(x), as.numeric(y), as.logical(intercept),
          as.list(opts_admm), PACKAGE = "ADMM")
}

