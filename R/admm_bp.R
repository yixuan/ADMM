## Class to describe a Basis Pursuit model
ADMM_BP = setRefClass("ADMM_BP",
                      fields = list(x = "matrix",
                                    y = "numeric",
                                    maxit = "integer",
                                    eps_abs = "numeric",
                                    eps_rel = "numeric",
                                    rho_ratio = "numeric")
)

## Class to store fitting results of Basis Pursuit model
ADMM_BP_fit = setRefClass("ADMM_BP_fit",
                          fields = list(beta = "dgCMatrix",
                                        niter = "integer")
)





##### Member functions of ADMM_BP #####

## Initialize fields including default values
ADMM_BP$methods(
    initialize = function(x, y, ...)
    {
        if(nrow(x) >= ncol(x))
            stop("ncol(x) must be greater than nrow(x)")
        if(nrow(x) != length(y))
            stop("nrow(x) should be equal to length(y)")
        
        .self$x = as.matrix(x)
        .self$y = as.numeric(y)
        .self$maxit = 10000L
        .self$eps_abs = 1e-4
        .self$eps_rel = 1e-4
        .self$rho_ratio = 0.1
    }
)

## Print off ADMM_BP object
ADMM_BP$methods(
    show_common = function()
    {
        cat(sprintf("$x: <%d x %d> matrix\n", nrow(.self$x), ncol(.self$x)))
        cat(sprintf("$y: <%d x 1> vector\n", length(.self$y)))
        
        fields = setdiff(names(.refClassDef@fieldClasses), c("x", "y"))
        for(field in fields)
            cat("$", field, ": ", paste(.self$field(field), collapse = " "),
                "\n", sep = "")
    },
    show = function()
    {
        cat("ADMM Basis Pursuit model\n\n")
        show_common()
    }
)

## Specify additional parameters
ADMM_BP$methods(
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
ADMM_BP$methods(
    fit = function(...)
    {
        res = .Call("admm_bp", .self$x, .self$y,
                    list(maxit = .self$maxit,
                         eps_abs = .self$eps_abs,
                         eps_rel = .self$eps_rel,
                         rho_ratio = .self$rho_ratio),
                    PACKAGE = "ADMM")
        
        do.call(ADMM_BP_fit, res)
    }
)





##### Member functions of ADMM_BP_fit #####

## Print off ADMM_BP_fit object
ADMM_BP_fit$methods(
    show_common = function()
    {
        cat("$beta\n")
        cat(sprintf("<%d x %d> sparse matrix\n", nrow(.self$beta), ncol(.self$beta)))
        cat("\n")
        cat("$niter\n")
        print(.self$niter)
    }
    show = function()
    {
        cat("ADMM Basis Pursuit fitting result\n\n")
        show_common()
    }
)

## Plot ADMM_BP_fit object
ADMM_BP_fit$methods(
    plot = function()
    {
        coefs = as.numeric(.self$beta)
        dat = data.frame(Index = seq_along(coefs),
                         Coefficients = coefs)
        ggplot(dat, aes(x = Index, y = Coefficients)) +
            geom_segment(aes(xend = Index, yend = 0))
    }
)





admm_bp = function(x, y, ...)
{
    ADMM_BP(x, y, ...)
}

