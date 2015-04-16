## Class to describe a Lasso model
ADMM_Lasso = setRefClass("ADMM_Lasso",
    fields = list(x = "matrix",
                  y = "numeric",
                  intercept = "logical",
                  standardize = "logical",
                  lambda = "numeric",
                  nlambda = "integer",
                  lambda_min_ratio = "numeric",
                  nthread = "integer",
                  maxit = "integer",
                  eps_abs = "numeric",
                  eps_rel = "numeric",
                  rho = "numeric")
)

## Class to store fitting results of Lasso model
ADMM_Lasso_fit = setRefClass("ADMM_Lasso_fit",
    fields = list(lambda = "numeric",
                  beta = "dgCMatrix",
                  niter = "integer")
)





##### Member functions of ADMM_Lasso #####

## Initialize fields including default values
ADMM_Lasso$methods(
    initialize = function(x, y, intercept = TRUE, standardize = TRUE, ...)
    {
        if(nrow(x) != length(y))
            stop("nrow(x) should be equal to length(y)")

        .self$x = as.matrix(x)
        .self$y = as.numeric(y)
        .self$intercept = as.logical(intercept)
        .self$standardize = as.logical(standardize)
        .self$lambda = numeric(0)
        .self$nlambda = 100L
        .self$lambda_min_ratio = ifelse(nrow(x) < ncol(x), 0.01, 0.0001)
        .self$nthread = 1L
        .self$maxit = 10000L
        .self$eps_abs = 1e-5
        .self$eps_rel = 1e-5
        .self$rho = -1.0
    }
)

## Print off ADMM_Lasso object
ADMM_Lasso$methods(
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
        cat("ADMM Lasso model\n\n")
        show_common()
    }
)

## Set up penalty parameters
ADMM_Lasso$methods(
    penalty = function(lambda = NULL, nlambda = 100, lambda_min_ratio, ...)
    {
        lambda_val = sort(as.numeric(lambda), decreasing = TRUE)
        if(any(lambda_val <= 0))
            stop("lambda must be positive")
        
        if(nlambda[1] <= 0)
            stop("nlambda must be a positive integer")
        
        if(missing(lambda_min_ratio))
            lmr_val = ifelse(nrow(.self$x) < ncol(.self$x), 0.01, 0.0001)
        else
            lmr_val = as.numeric(lambda_min_ratio)
        
        if(lmr_val >= 1 | lmr_val <= 0)
            stop("lambda_min_ratio must be within (0, 1)")
        
        .self$lambda = lambda_val
        .self$nlambda = as.integer(nlambda[1])
        .self$lambda_min_ratio = lmr_val
        
        invisible(.self)
    }
)

## Specify parallel computing
ADMM_Lasso$methods(
    parallel = function(nthread = 2, ...)
    {
        nthread_val = as.integer(nthread)
        if(nthread_val < 1)
            nthread_val = 1L
        if(nthread_val >= ncol(.self$x) / 5)
            stop("nthread cannot exceed ncol(x)/5")
        
        .self$nthread = nthread_val
        
        invisible(.self)
    }
)

## Specify additional parameters
ADMM_Lasso$methods(
    opts = function(maxit = 10000, eps_abs = 1e-5, eps_rel = 1e-5,
                    rho = NULL, ...)
    {
        if(maxit <= 0)
            stop("maxit should be positive")
        if(eps_abs < 0 | eps_rel < 0)
            stop("eps_abs and eps_rel should be nonnegative")
        if(isTRUE(rho <= 0))
            stop("rho should be positive")
        
        .self$maxit = as.integer(maxit)
        .self$eps_abs = as.numeric(eps_abs)
        .self$eps_rel = as.numeric(eps_rel)
        .self$rho = if(isNULL(rho))  -1.0  else  as.numeric(rho)
        
        invisible(.self)
    }
)

## Fit model and conduct the computing
ADMM_Lasso$methods(
    fit = function(...)
    {
        res = if(.self$nthread <= 1)
            .Call("admm_lasso", .self$x, .self$y, .self$lambda,
                  .self$nlambda, .self$lambda_min_ratio,
                  .self$standardize, .self$intercept,
                  list(maxit = .self$maxit,
                       eps_abs = .self$eps_abs,
                       eps_rel = .self$eps_rel,
                       rho = .self$rho),
                  PACKAGE = "ADMM")
        else
            .Call("admm_parlasso", .self$x, .self$y, .self$lambda,
                  .self$nlambda, .self$lambda_min_ratio,
                  .self$standardize, .self$intercept,
                  .self$nthread, FALSE,
                  list(maxit = .self$maxit,
                       eps_abs = .self$eps_abs,
                       eps_rel = .self$eps_rel,
                       rho_rel = .self$rho_rel),
                  PACKAGE = "ADMM")
        do.call(ADMM_Lasso_fit, res)
    }
)





##### Member functions of ADMM_Lasso_fit #####

## Print off ADMM_Lasso_fit object
ADMM_Lasso_fit$methods(
    show_common = function()
    {
        cat("$lambda\n")
        print(.self$lambda)
        cat("\n")
        cat("$beta\n")
        cat(sprintf("<%d x %d> sparse matrix\n", nrow(.self$beta), ncol(.self$beta)))
        cat("\n")
        cat("$niter\n")
        print(.self$niter)
    },
    show = function()
    {
        cat("ADMM Lasso fitting result\n\n")
        show_common()
    }
)

## Plot ADMM_Lasso_fit object
ADMM_Lasso_fit$methods(
    plot = function(...)
    {
        nlambda = length(.self$lambda)
        # If we only have one lambda we cannot create a path plot
        if(nlambda < 2)
            stop("need to have at least two lambda values")
        
        loglambda = log(.self$lambda)
        # Exclude variables that have zero coefficients for all lambdas
        rows_inc = apply(.self$beta, 1, function(x) any(x != 0))
        # Exclude intercept
        rows_inc[1] = FALSE
        mat = t(as.matrix(.self$beta[rows_inc, ]))
        nvar = ncol(mat)
        dat = data.frame(loglambda = rep(loglambda, nvar),
                         varid = rep(1:nvar, each = nlambda),
                         coef = as.numeric(mat))
        ggplot(dat, aes(x = loglambda, y = coef, color = factor(varid))) +
            geom_line(aes(group = varid)) +
            xlab(expression(log(lambda))) +
            ylab("Coefficients") +
            ggtitle("Solution path") +
            guides(color = FALSE)
    }
)



#' Fitting A Lasso Model Using ADMM Algorithm
#' 
#' @description Lasso is a popular variable selection technique in high
#' dimensional regression analysis, which tries to find the coefficient vector
#' \eqn{\beta} that minimizes
#' \deqn{\frac{1}{2n}\Vert y-X\beta\Vert_2^2+\lambda\Vert\beta\Vert_1}{
#' 1/(2n) * ||y - X * \beta||_2^2 + \lambda * ||\beta||_1}
#' 
#' Here \eqn{n} is the sample size and \eqn{\lambda} is a regularization
#' parameter that controls the sparseness of \eqn{\beta}.
#' 
#' This function will not directly conduct the computation,
#' but rather returns an object of class "\code{ADMM_Lasso}" that contains
#' several memeber functions to actually constructs and fits the model.
#' 
#' Member functions that are callable from this object are listed below:
#' 
#' \tabular{ll}{
#'   \code{$penalty()}  \tab Specify the penalty parameter. See section
#'                           \strong{Setting Penalty Parameter} for details.\cr
#'   \code{$parallel()} \tab Specify the number of threads for parallel computing.
#'                           See section \strong{Parallel Computing} for details.\cr
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
#' @section Setting Penalty Parameter:
#' The penalty parameter \eqn{\lambda} can be set through the member function
#' \code{$penalty()}, with the usage and parameters given below:
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
#' This member function will implicitly return the "\code{ADMM_Lasso}" object itself.
#' 
#' @section Parallel Computing:
#' The Lasso model can be fitted with parallel computing by setting the number
#' of threads in the \code{$parallel()} member function. The usage of this method
#' is
#' 
#' \preformatted{    model$parallel(nthread = 2, ...)
#' }
#' 
#' Here \code{model} is the object returned by \code{admm_lasso()}, and
#' \code{nthread} is the number of threads to be used. \code{nthread} must be
#' less than \code{ncol(x) / 5}.
#' 
#' \strong{NOTE:} Even in serial version of \code{admm_lasso()}, most matrix
#' operations are implicitly parallelized when proper compiler options are
#' turned on. Hence the parallel version of \code{admm_lasso()} is not
#' necessarily faster than the serial one.
#' 
#' This member function will implicitly return the "\code{ADMM_Lasso}" object itself.
#' 
#' @section Additional Options:
#' Additional options related to ADMM algorithm can be set through the
#' \code{$opts()} member function of an "\code{ADMM_Lasso}" object. The usage of
#' this method is
#' 
#' \preformatted{    model$opts(maxit = 10000, eps_abs = 1e-5, eps_rel = 1e-5,
#'                rho = NULL)
#' }
#' 
#' Here \code{model} is the object returned by \code{admm_lasso()}.
#' Explanation of the arguments is given below:
#' 
#' \describe{
#' \item{\code{maxit}}{Maximum number of iterations.}
#' \item{\code{eps_abs}}{Absolute tolerance parameter.}
#' \item{\code{eps_rel}}{Relative tolerance parameter.}
#' \item{\code{rho}}{ADMM step size parameter. If set to \code{NULL}, the program
#'                   will compute a default one.}
#' }
#' 
#' This member function will implicitly return the "\code{ADMM_Lasso}" object itself.
#' 
#' @section Model Fitting:
#' Model will be fit after calling the \code{$fit()} member function. This is no
#' argument that needs to be set. The function will return an object of class
#' "\code{ADMM_Lasso_fit}", which contains the following fields:
#' 
#' \describe{
#' \item{\code{lambda}}{The sequence of \eqn{\lambda} to build the solution path.}
#' \item{\code{beta}}{A sparse matrix containing the estimated coefficient vectors,
#'                    each column for one \eqn{\lambda}. Intercepts are in the
#'                    first row.}
#' \item{\code{niter}}{Number of ADMM iterations.}
#' }
#' 
#' Class "\code{ADMM_Lasso_fit}" also contains a \code{$plot()} member function,
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
#' admm_lasso(x, y)$fit()
#' 
#' ## Or, if you want to have more customization:
#' model = admm_lasso(x, y)
#' print(model)
#' 
#' ## Specify the lambda sequence
#' model$penalty(nlambda = 20, lambda_min_ratio = 0.01)
#' 
#' ## Lower down precision for faster computation
#' model$opts(maxit = 100, eps_rel = 0.001)
#' 
#' ## Use parallel computing (not necessary for this small dataset here)
#' # model$parallel(nthread = 2)
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
admm_lasso = function(x, y, intercept = TRUE, standardize = TRUE, ...)
{
    ADMM_Lasso(x, y, intercept, standardize, ...)
}
