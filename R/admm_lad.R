## Class to describe an LAD model
ADMM_LAD = setRefClass("ADMM_LAD",
                       fields = list(intercept = "logical"),
                       contains = "ADMM_BP")

## Class to store fitting results of LAD model
ADMM_LAD_fit = setRefClass("ADMM_LAD_fit", contains = "ADMM_BP_fit")





##### Member functions of ADMM_LAD #####

## Initialize fields including default values
ADMM_LAD$methods(
    initialize = function(x, y, intercept = TRUE, ...)
    {
        if(nrow(x) <= ncol(x))
            stop("nrow(x) must be greater than ncol(x)")
        if(nrow(x) != length(y))
            stop("nrow(x) should be equal to length(y)")
        
        .self$x = as.matrix(x)
        .self$y = as.numeric(y)
        .self$maxit = 10000L
        .self$eps_abs = 1e-4
        .self$eps_rel = 1e-4
        .self$rho_ratio = 1
        .self$intercept = TRUE
    }
)

## Print off ADMM_LAD object
ADMM_LAD$methods(
    show = function()
    {
        cat("ADMM Least Absolute Deviation model\n\n")
        show_common()
    }
)

## Specify additional parameters
ADMM_LAD$methods(
    opts = function(maxit = 10000, eps_abs = 1e-4, eps_rel = 1e-4,
                    rho_ratio = 1, ...)
    {
        callSuper(maxit, eps_abs, eps_rel, rho_ratio, ...)
        
        invisible(.self)
    }
)

## Fit model and conduct the computing
ADMM_LAD$methods(
    fit = function(...)
    {
        res = .Call("admm_lad", .self$x, .self$y, .self$intercept,
                    list(maxit = .self$maxit,
                         eps_abs = .self$eps_abs,
                         eps_rel = .self$eps_rel,
                         rho_ratio = .self$rho_ratio),
                    PACKAGE = "ADMM")
        
        ADMM_LAD_fit(x = .self$x, y = .self$y, beta = res$beta, niter = res$niter)
    }
)





##### Member functions of ADMM_LAD_fit #####

## Print off ADMM_LAD_fit object
ADMM_LAD_fit$methods(
    show = function()
    {
        cat("ADMM Least Absolute Deviation fitting result\n\n")
        show_common()
    }
)

## Plot ADMM_LAD_fit object
ADMM_LAD_fit$methods(
    plot = function(type = "fit", ...)
    {
        yfit = as.numeric(.self$x %*% .self$beta[-1]) + .self$beta[1]
        dat = data.frame(yfit = yfit, y = .self$y)
        ggplot(dat, aes(x = yfit, y = y)) +
            geom_point() +
            xlab("Fitted values") +
            ylab("Observed values")
    }
)





admm_lad = function(x, y, ...)
{
    ADMM_LAD(x, y, ...)
}
