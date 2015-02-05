## Class to describe a Basis Pursuit model
ADMM_BP = setRefClass("ADMM_BP",
                      fields = list(x = "matrix",
                                    y = "numeric",
                                    maxit = "integer",
                                    eps_abs = "numeric",
                                    eps_rel = "numeric",
                                    rho_ratio = "numeric")
)

setClassUnion("CoefType", c("dgCMatrix", "numeric"))

## Class to store fitting results of Basis Pursuit model
ADMM_BP_fit = setRefClass("ADMM_BP_fit",
                          fields = list(beta = "CoefType",
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
    opts = function(maxit = 10000, eps_abs = 1e-4, eps_rel = 1e-4,
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
        if(class(.self$beta) == "dgCMatrix")
        {
            cat(sprintf("<%d x %d> sparse matrix\n",
                        nrow(.self$beta), ncol(.self$beta)))
        } else {
            cat(sprintf("<%d x 1> vector\n", length(.self$beta)))
        }
        cat("\n")
        cat("$niter\n")
        print(.self$niter)
    },
    show = function()
    {
        cat("ADMM Basis Pursuit fitting result\n\n")
        show_common()
    }
)

## Plot ADMM_BP_fit object
ADMM_BP_fit$methods(
    plot = function(...)
    {
        coefs = as.numeric(.self$beta)
        dat = data.frame(Index = seq_along(coefs),
                         Coefficients = coefs)
        g = ggplot(dat, aes(x = Index, y = Coefficients)) +
            geom_segment(aes(xend = Index, yend = 0))
        print(g)
        invisible(g)
    }
)





#' Fitting A Basis Pursuit Model Using ADMM Algorithm
#' 
#' @description Basis Pursuit is an optimization problem that minimizes
#' \eqn{\Vert \beta \Vert_1}{||\beta||_1} subject to
#' \eqn{y=X\beta}{y = X * \beta}. Here \eqn{X} is an \eqn{n} by \eqn{p}
#' matrix with \eqn{p > n}. Basis Pursuit is broadly applied in Compressed
#' Sensing to recover a sparse vector \eqn{\beta} from the transformed
#' lower dimensional vector \eqn{y}.
#' 
#' This function will not directly conduct the computation,
#' but rather returns an object of class "\code{ADMM_BP}" that contains
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
#' @param x The transformation matrix
#' @param y The transformed vector to recover from
#' 
#' @section Additional Options:
#' Additional options related to ADMM algorithm can be set through the
#' \code{$opts()} member function of an "\code{ADMM_BP}" object. The usage of
#' this method is
#' 
#' \preformatted{    model$opts(maxit = 10000, eps_abs = 1e-4, eps_rel = 1e-4,
#'                rho_ratio = 0.1)
#' }
#' 
#' Here \code{model} is the object returned by \code{admm_bp()}.
#' Explanation of the arguments is given below:
#' 
#' \describe{
#' \item{\code{maxit}}{Maximum number of iterations.}
#' \item{\code{eps_abs}}{Absolute tolerance parameter.}
#' \item{\code{eps_rel}}{Relative tolerance parameter.}
#' \item{\code{rho_ratio}}{ADMM step size parameter.}
#' }
#' 
#' This member function will implicitly return the "\code{ADMM_BP}" object itself.
#' 
#' @section Model Fitting:
#' Model will be fit after calling the \code{$fit()} member function. This is no
#' argument that needs to be set. The function will return an object of class
#' "\code{ADMM_BP_fit}", which contains the following fields:
#' 
#' \describe{
#' \item{\code{beta}}{The recovered \eqn{\beta} vector in sparse form.}
#' \item{\code{niter}}{Number of ADMM iterations.}
#' }
#' 
#' Class "\code{ADMM_BP_fit}" also contains a \code{$plot()} member function,
#' which plots the coefficients against their indices. See the examples below.
#' 
#' @examples
#' ## An Compressed Sensing example ##
#' 
#' ## Create a sparse signal vector
#' set.seed(123)
#' n = 50
#' p = 100
#' nsig = 15
#' beta_true = c(runif(nsig), rep(0, p - nsig))
#' beta_true = sample(beta_true)
#' 
#' ## Generate the transformation matrix and the compressed vector
#' x = matrix(rnorm(n * p), n, p)
#' y = drop(x %*% beta_true)
#' 
#' ## Build the model
#' model = admm_bp(x, y)
#' 
#' ## Request a higher precision
#' model$opts(eps_rel = 1e-5)
#' 
#' ## Fit the model
#' res = model$fit()
#' res
#' 
#' ## Plot for the recovered vector
#' res$plot()
#' 
#' ## The steps above can be accomplished using a chainable call
#' admm_bp(x, y)$opts(eps_rel = 1e-5)$fit()$plot()
#' 
#' ## Compare the true beta and the recovered one
#' library(ggplot2)
#' g = res$plot()
#' d = data.frame(ind = seq_along(beta_true),
#'                coef = beta_true)
#' g + geom_segment(aes(x = ind + 0.5, xend = ind + 0.5,
#'                      y = coef, yend = 0), data = d, color = "red")
#' 
#' @author Yixuan Qiu <\url{http://statr.me}>
#' @export
admm_bp = function(x, y, ...)
{
    ADMM_BP(x, y, ...)
}

