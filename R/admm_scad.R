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





#' Fitting A Penalized Least Squares Model With SCAD Threshold
#' 
#' @description This function will not directly conduct the computation,
#' but rather returns an object of class "\code{ADMM_SCAD}" that contains
#' several memeber functions to actually constructs and fits the model.
#' 
#' Member functions that are callable from this object are listed below:
#' 
#' \tabular{ll}{
#'   \code{$penalty()}  \tab Specify the penalty parameter. See section
#'                           \strong{Setting Penalty Parameter} for details.\cr
#'   \code{$opts()}     \tab Setting additional options. See section
#'                           \strong{Additional Options} for details.\cr
#'   \code{$fit()}      \tab Fit the model and do the actual computation.
#'                           See section \strong{Model Fitting} for details.
#' }
#' 
#' @param x The data matrix
#' @param y The response vector
#' @param intercept Whether to fit an intercept in the model. Default is \code{TRUE}.
#' @param standardize Whether to standardize the explanatory variables before
#'                    fitting the model. Default is \code{TRUE}. Fitted coefficients
#'                    are always returned on the original scale.
#' 
#' @examples set.seed(123)
#' n = 100
#' p = 20
#' b = runif(p)
#' x = matrix(rnorm(n * p, mean = 1.2, sd = 2), n, p)
#' y = 5 + c(x %*% b) + rnorm(n)
#' 
#' ## Directly fit the model
#' admm_scad(x, y)$fit()
#' 
#' ## Or, if you want to have more customization:
#' model = admm_scad(x, y)
#' print(model)
#' 
#' ## Specify the lambda sequence and the "a"-parameter
#' model$penalty(nlambda = 20, lambda_min_ratio = 0.01, penalty_a = 4)
#' 
#' ## Lower down precision for faster computation
#' model$opts(maxit = 100, eps_rel = 0.001)
#' 
#' ## Inspect the updated model setting
#' print(model)
#' 
#' ## Fit the model and do the actual computation
#' res = model$fit()
#' res$beta
#' 
#' ## Create a solution path plot
#' res$plot()
#' 
#' @author Yixuan Qiu <\url{http://statr.me}>
#' @export
admm_scad = function(x, y, intercept = TRUE, standardize = TRUE, ...)
{
    ADMM_SCAD(x, y, intercept, standardize, ...)
}
