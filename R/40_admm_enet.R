## Class to describe an elastic net model
ADMM_Enet = setRefClass("ADMM_Enet",
                        fields = list(alpha = "numeric"),
                        contains = "ADMM_Lasso"
)

## Class to store fitting results of elastic net model
ADMM_Enet_fit = setRefClass("ADMM_Enet_fit", contains = "ADMM_Lasso_fit")





##### Member functions of ADMM_Enet #####

## Initialize fields including default values
ADMM_Enet$methods(
    initialize = function(...)
    {
        .self$alpha = 1
        callSuper(...)
    }
)

## Print off ADMM_Enet object
ADMM_Enet$methods(
    show = function()
    {
        cat("ADMM Elastic Net model\n\n")
        show_common()
    }
)

## Set up penalty parameters
ADMM_Enet$methods(
    penalty = function(lambda = NULL, nlambda = 100, lambda_min_ratio,
                       alpha = 1, ...)
    {
        if(alpha < 0 | alpha > 1)
            stop("alpha must be within [0,1]")
        
        .self$alpha = as.numeric(alpha)
        callSuper(lambda, nlambda, lambda_min_ratio, ...)
        
        invisible(.self)
    }
)

## Fit model and conduct the computing
ADMM_Enet$methods(
    fit = function(...)
    {
        res = .Call("admm_enet", .self$x, .self$y, .self$lambda,
                    .self$nlambda, .self$lambda_min_ratio,
                    .self$standardize, .self$intercept,
                    .self$alpha,
                    list(maxit = .self$maxit,
                         eps_abs = .self$eps_abs,
                         eps_rel = .self$eps_rel,
                         rho_ratio = .self$rho_ratio),
                    PACKAGE = "ADMM")
        do.call(ADMM_Enet_fit, res)
    }
)





##### Member functions of ADMM_Lasso_fit #####

## Print off ADMM_Lasso_fit object
ADMM_Enet_fit$methods(
    show = function()
    {
        cat("ADMM Elastic Net fitting result\n\n")
        show_common()
    }
)





#' Fitting An Elastic Net Model Using ADMM Algorithm
#' 
#' @description This function will not directly conduct the computation,
#' but rather returns an object of class "\code{ADMM_Enet}" that contains
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
#' admm_enet(x, y)$penalty(alpha = 0.5)$fit()
#' 
#' ## Or, if you want to have more customization:
#' model = admm_enet(x, y)
#' print(model)
#' 
#' ## Specify the lambda sequence and the alpha parameter
#' model$penalty(nlambda = 20, lambda_min_ratio = 0.01, alpha = 0.5)
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
admm_enet = function(x, y, intercept = TRUE, standardize = TRUE, ...)
{
    ADMM_Enet(x, y, intercept, standardize, ...)
}
