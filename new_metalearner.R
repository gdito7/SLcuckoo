
# Constrained Least Squared
method_consLS=function() 
{
  out <- list(require = "CVXR",
              computeCoef = function(Z, Y,
                                     libraryNames,
                                     obsWeights, control, verbose, ...) {
                cvRisk <- apply(Z, 2, function(x) mean(obsWeights *(x - Y)^2))
                names(cvRisk) <- libraryNames
                source("cons_ls.R")
                fit_ls <- cons_ls(sqrt(obsWeights) * Z, sqrt(obsWeights) * Y)
                if (verbose) {
                  message(paste("Constrained least squares convergence:", 
                                fit_ls$status == "optimal"))
                }
                coef1 <- fit_ls$solution
                names(coef1) <- libraryNames
                out <- list(cvRisk = cvRisk, coef = coef1, optimizer = fit_ls)
                return(out)
              },
              computePred = function(predY, coef, control, ...) {
                out <- crossprod(t(predY), coef)
                return(out)
              })
  invisible(out)
}


#Non negative Ridge Regression
method_NNRidge=function() 
{
  out <- list(require = "glmnet",
              computeCoef = function(Z, Y,
                                     libraryNames,
                                     obsWeights, control, verbose, ...) {
                cvRisk <- apply(Z, 2, function(x) mean(obsWeights *(x - Y)^2))
                names(cvRisk) <- libraryNames
                
                fit_ls <- glmnet::cv.glmnet(sqrt(obsWeights) * Z,
                                            sqrt(obsWeights) * Y,
                                            family="binomial",lower.limits=0,
                                            upper.limits=1,alpha=0,
                                            nfolds = 10)
                #fit_ls <- cons_ridge(sqrt(obsWeights) * Z, sqrt(obsWeights) * Y)
                if (verbose) {
                  message(paste("Non Negative Ridge convergence:"))
                }
                #coef1 <- fit_ls$solution
                coef1 <- as.numeric(coef.cv.glmnet(fit_ls))[-1]
                names(coef1) <- libraryNames
                out <- list(cvRisk = cvRisk, coef = coef1, optimizer = fit_ls)
                return(out)
              },
              computePred = function(predY, coef, control, ...) {
                out <- crossprod(t(predY), coef)
                return(out)
              })
  invisible(out)
}