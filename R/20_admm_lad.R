## Class to describe an LAD model
ADMM_LAD = setRefClass("ADMM_LAD",
                       fields = list(intercept = "logical"),
                       contains = "ADMM_BP")

## Class to store fitting results of LAD model
ADMM_LAD_fit = setRefClass("ADMM_LAD_fit",
                           fields = list(x = "matrix", y = "numeric"),
                           contains = "ADMM_BP_fit")





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
        .self$rho_ratio = 0.1
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
        g = ggplot(dat, aes(x = yfit, y = y)) +
            geom_point() +
            geom_abline(intercept = 0, slope = 1, color = "red") +
            xlab("Fitted values") +
            ylab("Observed values")
        print(g)
        invisible(g)
    }
)





#' Fitting A Least Absolute Deviation Model Using ADMM Algorithm
#' 
#' @description Least Absolute Deviation (LAD) is similar to an OLS regression
#' model, but it minimizes the absolute deviation
#' \eqn{\Vert y-X\beta \Vert_1}{||y - X\beta||_1} instead of the sum of squares
#' \eqn{\Vert y-X\beta \Vert_2^2}{||y - X\beta||_2^2}. LAD is equivalent to the
#' median regression, a special case of the quantile regression models. LAD is
#' a robust regression technique in the sense that the estimated coefficients are
#' insensitive to outliers.
#' 
#' This function will not directly conduct the computation,
#' but rather returns an object of class "\code{ADMM_LAD}" that contains
#' several memeber functions to actually constructs and fits the model.
#' 
#' Member functions that are callable from this object are listed below:
#' 
#' \tabular{ll}{
#'   \code{$opts()}     \tab Setting additional options. See section
#'                           \strong{Additional Options} for details.\cr
#'   \code{$fit()}      \tab Fit the model and do the actual computation.
#'                           See section \strong{Model Fitting} for details.
#' }
#' 
#' @param x The data matrix.
#' @param y The response vector.
#' @param intercept Whether to include an intercept term. Default is \code{TRUE}.
#' 
#' @section Additional Options:
#' Additional options related to ADMM algorithm can be set through the
#' \code{$opts()} member function of an "\code{ADMM_LAD}" object. The usage of
#' this method is
#' 
#' \preformatted{    model$opts(maxit = 10000, eps_abs = 1e-4, eps_rel = 1e-4,
#'                rho_ratio = 0.1)
#' }
#' 
#' Here \code{model} is the object returned by \code{admm_lad()}.
#' Explanation of the arguments is given below:
#' 
#' \describe{
#' \item{\code{maxit}}{Maximum number of iterations.}
#' \item{\code{eps_abs}}{Absolute tolerance parameter.}
#' \item{\code{eps_rel}}{Relative tolerance parameter.}
#' \item{\code{rho_ratio}}{ADMM step size parameter.}
#' }
#' 
#' This member function will implicitly return the "\code{ADMM_LAD}" object itself.
#' 
#' @section Model Fitting:
#' Model will be fit after calling the \code{$fit()} member function. This is no
#' argument that needs to be set. The function will return an object of class
#' "\code{ADMM_LAD_fit}", which contains the following fields:
#' 
#' \describe{
#' \item{\code{x}}{The data matrix.}
#' \item{\code{y}}{The response vector.}
#' \item{\code{beta}}{The estimated regression coefficients, including the intercept.}
#' \item{\code{niter}}{Number of ADMM iterations.}
#' }
#' 
#' Class "\code{ADMM_LAD_fit}" also contains a \code{$plot()} member function,
#' which plots the fitted values with observed values. See the examples below.
#' 
#' @examples
#' ## Robust regression with LAD ##
#' 
#' ## Generate data with an outlier
#' set.seed(123)
#' x = sort(rnorm(100))
#' y = x + rnorm(100, sd = 0.3)
#' y[1] = y[1] + 5
#' 
#' ## Build an LAD model (median regression)
#' model = admm_lad(x, y)
#' 
#' ## Lower down the precision for faster computation
#' model$opts(eps_rel = 1e-3)
#' 
#' ## Fit the model
#' res = model$fit()
#' 
#' ## Plot for the fitted values and observed values
#' res$plot()
#' 
#' ## The steps above can be accomplished using a chainable call
#' admm_lad(x, y)$opts(eps_rel = 1e-3)$fit()$plot()
#' 
#' ## Compare LAD with OLS
#' library(ggplot2)
#' ols = lm(y ~ x)$coefficients
#' d = data.frame(intercept = c(ols[1], res$beta[1], 0),
#'                slope = c(ols[2], res$beta[2], 1),
#'                method = c("OLS", "LAD", "Truth"))
#' ggplot(data.frame(x = x, y = y), aes(x = x, y = y)) +
#'     geom_point() +
#'     geom_abline(aes(intercept = intercept, slope = slope, color = method),
#'                 data = d, show_guide = TRUE)
#' 
#' @author Yixuan Qiu <\url{http://statr.me}>
#' @export
admm_lad = function(x, y, intercept = TRUE, ...)
{
    ADMM_LAD(x = as.matrix(x), y = as.numeric(y),
        intercept = as.logical(intercept), ...)
}
