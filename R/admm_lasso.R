admm_lasso = function(x, y, lambda = 1.0, maxit = 500L, eps_abs = 1e-6,
                      eps_rel = 1e-6, rho = 1e-4)
{
    .Call("admm_lasso", as.matrix(x), as.numeric(y), as.numeric(lambda),
          as.integer(maxit), as.numeric(eps_abs), as.numeric(eps_rel),
          as.numeric(rho), PACKAGE = "ADMM")
}