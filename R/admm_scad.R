## Class to describe a SCAD model
ADMM_SCAD = setRefClass("ADMM_SCAD",
                        fields = list(penalty_a = "numeric"),
                        contains = "ADMM_Lasso"
)

## Class to store fitting results of SCAD model
ADMM_SCAD_fit = setRefClass("ADMM_SCAD_fit", contains = "ADMM_Lasso_fit")





##### Member functions of ADMM_SCAD #####

## Initialize fields including default values
ADMM_SCAD$methods(
    initialize = function(...)
    {
        .self$penalty_a = 3.7
        callSuper(...)
    }
)

## Set up penalty parameters
ADMM_SCAD$methods(
    penalty = function(lambda = NULL, nlambda = 100, lambda_min_ratio,
                       penalty_a = 3.7, ...)
    {
        if(penalty_a <= 2)
            stop("penalty_a must be greater than 2")
        
        .self$penalty_a = as.numeric(penalty_a)
        callSuper(lambda, nlambda, lambda_min_ratio, ...)
        
        invisible(.self)
    }
)

## Fit model and conduct the computing
ADMM_SCAD$methods(
    fit = function(...)
    {
        res = .Call("admm_scad", .self$x, .self$y, .self$lambda,
                    .self$nlambda, .self$lambda_min_ratio,
                    .self$standardize, .self$intercept,
                    .self$penalty_a,
                    list(maxit = .self$maxit,
                         eps_abs = .self$eps_abs,
                         eps_rel = .self$eps_rel,
                         rho_ratio = .self$rho_ratio),
                    PACKAGE = "ADMM")
        do.call(ADMM_SCAD_fit, res)
    }
)





admm_scad = function(x, y, lambda = NULL,
                     nlambda = 100,
                     lambda_min_ratio = ifelse(nrow(x) < ncol(x), 0.01, 0.0001),
                     standardize = TRUE, intercept = TRUE,
                     pen_a = 3.7,
                     opts = list())
{
    # default parameters
    opts_admm = list(maxit = 10000L,
                     eps_abs = 1e-5,
                     eps_rel = 1e-5,
                     rho_ratio = 0.1)
    # update from opts
    opts_admm[names(opts)] = opts
    
    .Call("admm_scad", as.matrix(x), as.numeric(y), as.numeric(lambda),
          as.integer(nlambda), as.numeric(lambda_min_ratio),
          as.logical(standardize), as.logical(intercept),
          as.numeric(pen_a),
          as.list(opts_admm), PACKAGE = "ADMM")
}

