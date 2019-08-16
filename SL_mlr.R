SL.mlr=function (Y, X, newX, family, obsWeights,learner="classif.rpart",
                 par.vals=list(),...) 
{
  require(mlr)
  Y2=as.factor(Y)
  dta=data.frame(X,Y2)
  
  if (family$family == "gaussian") {
    if(!stringr::str_detect(learner,"regr")){
      stop("gaussian family must use regr")
    }
    task=makeClassifTask(data=dta,target = "Y2")
    lrn=makeLearner(learner,predict.type = "response",par.vals = par.vals)
    mod=train(lrn,task)
    pred = predict(mod,newdata = newX)
    pred = pred$data[,1]
    
  }
  if (family$family == "binomial") {
    if(!stringr::str_detect(learner,"classif")){
      stop("binomial family must use classif")
    }
    #browser()
    task=makeClassifTask(data=dta,target = "Y2",positive = "1")
    lrn=makeLearner(learner,predict.type = "prob",par.vals = par.vals)
    mod=train(lrn,task)
    pred = predict(mod,newdata = newX)
    pred = getPredictionProbabilities(pred,cl="1")
  }
  
  fit <- list(object=mod)
  class(fit) <- c("SL.mlr")
  out <- list(pred = pred, fit = mod)
  return(out)
}
