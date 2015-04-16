## Class to describe an Dantzig selector model
ADMM_Dantzig = setRefClass("ADMM_Dantzig", contains = "ADMM_Lasso")

## Class to store fitting results of Dantzig selector model
ADMM_Dantzig_fit = setRefClass("ADMM_Dantzig_fit", contains = "ADMM_Lasso_fit")





##### Member functions of ADMM_Dantzig #####

## Initialize fields including default values
ADMM_Dantzig$methods(
    initialize = function(...)
    {
        callSuper(...)
    }
)

## Print off ADMM_Dantzig object
ADMM_Dantzig$methods(
    show = function()
    {
        cat("ADMM Dantzig Selector model\n\n")
        show_common()
    }
)

## Fit model and conduct the computing
ADMM_Dantzig$methods(
    fit = function(...)
    {
        res = .Call("admm_dantzig", .self$x, .self$y, .self$lambda,
                    .self$nlambda, .self$lambda_min_ratio,
                    .self$standardize, .self$intercept,
                    list(maxit = .self$maxit,
                         eps_abs = .self$eps_abs,
                         eps_rel = .self$eps_rel,
                         rho = .self$rho),
                    PACKAGE = "ADMM")
        do.call(ADMM_Dantzig_fit, res)
    }
)





##### Member functions of ADMM_Lasso_fit #####

## Print off ADMM_Lasso_fit object
ADMM_Dantzig_fit$methods(
    show = function()
    {
        cat("ADMM Dantzig Selector fitting result\n\n")
        show_common()
    }
)





#' Fitting A Dantzig Selector Model Using ADMM Algorithm
#' 
#' @description Dantzig Selector is a variable selection technique that seeks a
#' coefficient vector \eqn{\beta} that minimizes
#' \eqn{\Vert\beta\Vert_1}{||\beta||_1} subject to
#' \eqn{\Vert X'(X\beta-y)\Vert_\infty \le \lambda}{||X'(X * \beta - y)||_inf <= \lambda}
#' 
#' Here \eqn{n} is the sample size, \eqn{\lambda} is the regularization
#' parameter, and \eqn{\Vert\cdot\Vert_1}{||.||_1},
#' \eqn{\Vert\cdot\Vert_\infty}{||.||_inf} stand for the L1 norm and maximum
#' norm respectively.
#' 
#' This function will not directly conduct the computation,
#' but rather returns an object of class "\code{ADMM_Dantzig}" that contains
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
#' @section Setting Penalty Parameter:
#' The penalty parameter \eqn{\lambda} can be set through
#' the member function \code{$penalty()}, with the usage and parameters given below:
#' 
#' \preformatted{    model$penalty(lambda = NULL, nlambda = 100, lambda_min_ratio, ...)
#' }
#' 
#' \describe{
#' \item{\code{lambda}}{A user provided sequence of \eqn{\lambda}. If set to
#'                      \code{NULL}, the program will calculate its own sequence
#'                      according to \code{nlambda} and \code{lambda_min_ratio},
#'                      which starts from \eqn{\lambda_0} (with this
#'                      \eqn{\lambda} all coefficients will be zero) and ends at
#'                      \code{lambda0 * lambda_min_ratio}, containing
#'                      \code{nlambda} values equally spaced in the log scale.
#'                      It is recommended to set this parameter to be \code{NULL}
#'                      (the default).}
#' \item{\code{nlambda}}{Number of values in the \eqn{\lambda} sequence. Only used
#'                       when the program calculates its own \eqn{\lambda}
#'                       (by setting \code{lambda = NULL}).}
#' \item{\code{lambda_min_ratio}}{Smallest value in the \eqn{\lambda} sequence
#'                                as a fraction of \eqn{\lambda_0}. See
#'                                the explanation of the \code{lambda}
#'                                argument. This parameter is only used when
#'                                the program calculates its own \eqn{\lambda}
#'                                (by setting \code{lambda = NULL}). The default
#'                                value is the same as \pkg{glmnet}: 0.0001 if
#'                                \code{nrow(x) >= ncol(x)} and 0.01 otherwise.}
#' }
#' 
#' This member function will implicitly return the "\code{ADMM_Dantzig}" object itself.
#' 
#' @section Additional Options:
#' Additional options related to ADMM algorithm can be set through the
#' \code{$opts()} member function of an "\code{ADMM_Dantzig}" object. The usage of
#' this method is
#' 
#' \preformatted{    model$opts(maxit = 10000, eps_abs = 1e-5, eps_rel = 1e-5,
#'                rho_ratio = 0.1)
#' }
#' 
#' Here \code{model} is the object returned by \code{admm_dantzig()}.
#' Explanation of the arguments is given below:
#' 
#' \describe{
#' \item{\code{maxit}}{Maximum number of iterations.}
#' \item{\code{eps_abs}}{Absolute tolerance parameter.}
#' \item{\code{eps_rel}}{Relative tolerance parameter.}
#' \item{\code{rho_ratio}}{ADMM step size parameter.}
#' }
#' 
#' This member function will implicitly return the "\code{ADMM_Dantzig}" object itself.
#' 
#' @section Model Fitting:
#' Model will be fit after calling the \code{$fit()} member function. This is no
#' argument that needs to be set. The function will return an object of class
#' "\code{ADMM_Dantzig_fit}", which contains the following fields:
#' 
#' \describe{
#' \item{\code{lambda}}{The sequence of \eqn{\lambda} to build the solution path.}
#' \item{\code{beta}}{A sparse matrix containing the estimated coefficient vectors,
#'                    each column for one \eqn{\lambda}. Intercepts are in the
#'                    first row.}
#' \item{\code{niter}}{Number of ADMM iterations.}
#' }
#' 
#' Class "\code{ADMM_Dantzig_fit}" also contains a \code{$plot()} member function,
#' which plots the coefficient paths with the sequence of \eqn{\lambda}.
#' See the examples below.
#' 
#' @examples set.seed(123)
#' n = 100
#' p = 20
#' b = runif(p)
#' x = matrix(rnorm(n * p, mean = 1.2, sd = 2), n, p)
#' y = 5 + c(x %*% b) + rnorm(n)
#' 
#' ## Directly fit the model
#' admm_dantzig(x, y)$fit()
#' 
#' ## Or, if you want to have more customization:
#' model = admm_dantzig(x, y)
#' print(model)
#' 
#' ## Specify the lambda sequence
#' model$penalty(nlambda = 20, lambda_min_ratio = 0.01)
#' 
#' ## Lower down precision for faster computation
#' model$opts(maxit = 500, eps_rel = 0.001)
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
admm_dantzig = function(x, y, intercept = TRUE, standardize = TRUE, ...)
{
    ADMM_Dantzig(x, y, intercept, standardize, ...)
}
