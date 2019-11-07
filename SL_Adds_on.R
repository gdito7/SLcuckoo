#==================== Super Learner Cross-validation=============================
SL_crossval=function(dta,target,folds=10,fixed=FALSE,index=NULL){
  if(fixed){
   if(is.null(index)){
     stop("index must be not NULL when fixed=TRUE")
   }else{
     index_crossval=index
   }
  }else{
  index_crossval=caret::createFolds(as.factor(dta%>%pull(target)),k=folds,returnTrain = F)
  }
  Xtrain=purrr::map(seq_along(index_crossval),
                    function(i){
                      index=index_crossval[[i]]
                      dta%>%slice(-index)%>%
                        dplyr::select(-target)
                    })
  Ytrain=purrr::map(seq_along(index_crossval),
                    function(i){
                      index=index_crossval[[i]]
                      dta%>%slice(-index)%>%pull(target)
                    })
  
  Xtest=purrr::map(seq_along(index_crossval),
                   function(i){
                     index=index_crossval[[i]]
                     dta%>%slice(index)%>%dplyr::select(-target)
                   })
  Ytest=purrr::map(seq_along(index_crossval),
                   function(i){
                     index=index_crossval[[i]]
                     dta%>%slice(index)%>%pull(target)
                   })
  data_list=list("Xtrain"=Xtrain,"Ytrain"=Ytrain,"Xtest"=Xtest,"Ytest"=Ytest)
  return(data_list)
}

#=======================Super Learner Prediction===========================
predict_SuperLearner=function (object, newdata, X = NULL, Y = NULL, onlySL = FALSE, 
                               ...) 
{
  if (missing(newdata)) {
    out <- list(pred = object$SL.predict, library.predict = object$library.predict)
    return(out)
  }
  if (!object$control$saveFitLibrary) {
    stop("This SuperLearner fit was created using control$saveFitLibrary = FALSE, so new predictions cannot be made.")
  }
  k <- length(object$libraryNames)
  predY <- matrix(NA, nrow = nrow(newdata), ncol = k)
  colnames(predY) <- object$libraryNames
  if (onlySL) {
    whichLibrary <- which(object$coef > 0)
    predY <- matrix(0, nrow = nrow(newdata), ncol = k)
    for (mm in whichLibrary) {
      newdataMM <- subset(newdata, select = object$whichScreen[object$SL.library$library[mm, 
                                                                                         2], ])
      family <- object$family
      XMM <- if (is.null(X)) {
        NULL
      }
      else {
        subset(X, select = object$whichScreen[object$SL.library$library[mm, 
                                                                        2], ])
      }
      predY1[, mm] <- do.call("predict", list(object = object$fitLibrary[[mm]], 
                                              newdata = newdataMM, family = family, X = XMM, 
                                              Y = Y, ...))
      predY1[, mm]=predY$data[,1]
    }
    getPred <- object$method$computePred(predY = predY, 
                                         coef = object$coef, control = object$control)
    out <- list(pred = getPred, library.predict = predY)
  }
  else {
    for (mm in seq(k)) {
      newdataMM <- subset(newdata, select = object$whichScreen[object$SL.library$library[mm, 
                                                                                         2], ])
      family <- object$family
      XMM <- if (is.null(X)) {
        NULL
      }
      else {
        subset(X, select = object$whichScreen[object$SL.library$library[mm, 
                                                                        2], ])
      }
      predY1 <- do.call("predict", list(object = object$fitLibrary[[mm]], 
                                        newdata = newdataMM, family = family, X = XMM, 
                                        Y = Y, ...))
      predY[, mm]=predY1$data[,1]
    }
    getPred <- object$method$computePred(predY = predY, 
                                         coef = object$coef, control = object$control)
    out <- list(pred = getPred, library.predict = predY)
  }
  return(out)
}


#====================Training Super Learner===================================

SL_train=function(SLmodel,dta,verbose=T,pkg="mlr",...){
  
  
  mod=purrr::map(seq_along(dta$Ytrain),
                 function(i){
                   
                   if(verbose){
                     cat("Folds",i,"\n")
                     tictoc::tic("Elapsed Time:")
                     res=SLmodel(dta$Ytrain[[i]],dta$Xtrain[[i]],...)
                     tictoc::toc()
                     return(res)
                   }else{
                     SLmodel(dta$Ytrain[[i]],dta$Xtrain[[i]],...)
                   }
                   
                 })
  # browser()
  if(pkg=="mlr"){
  coefSL=purrr::map(seq_along(mod),function(i) coef(mod[[i]]))
  pred=purrr::map(seq_along(mod),function(i){
    predict_SuperLearner(mod[[i]],dta$Xtest[[i]])
  } )
  }else{
    coefSL=purrr::map(seq_along(mod),function(i) coef(mod[[i]]))
    pred=purrr::map(seq_along(mod),function(i){
      predict.SuperLearner(mod[[i]],dta$Xtest[[i]])
    } )
    
  }
  return(list("prediction"=pred,"coef_SL"=coefSL,"truth"=dta$Ytest))
}



# =================Calculate Performance of Super Learner================

binary_enc=function(y,thres=0.5,code=c(0,1)){
  
  ifelse(y>thres,code[2],code[1])
}

SLperformance=function(train,measure,type="SL"){
  if(type=="SL"){
    #    browser()
    prediction=purrr::invoke(c,purrr::map(seq_along(train$prediction),function(i){
      train$prediction[[i]]$pred
    }))
    prediction=binary_enc(prediction,code = c(1,0))
    truth=purrr::invoke(c,purrr::map(seq_along(train$truth),function(i){
      train$truth[[i]]
    }))
    measure(truth,prediction)
  }else{
    #Super Learner Prediction
    prediction=purrr::invoke(c,purrr::map(seq_along(train$prediction),function(i){
      train$prediction[[i]]$pred
    }))
    prediction=binary_enc(prediction,code = c(1,0))
    truth=purrr::invoke(c,purrr::map(seq_along(train$truth),function(i){
      train$truth[[i]]
    }))
    SL_prediction=measure(truth,prediction)
    #Prediction each folds
    truth=train$truth
    prediction=purrr::map(seq_along(train$prediction),function(i){
      train$prediction[[i]]$pred
    })
    prediction=purrr::map(prediction,binary_enc,code = c(1,0))
    folds_prediction=purrr::map2_dbl(truth,prediction,measure)
    return(list("final_prediction"=SL_prediction,
                "folds_prediction"=folds_prediction))
  }
}

#==================== Cross-validation Conformal=============================
SL_holdout_conformal=function(dta,target,prob=0.2){
  y=dta%>%pull(target)
  index_test=caret::createDataPartition(y,p=prob,list=F)
  index_train=setdiff(seq_along(y),index_test)
  index_cal=caret::createDataPartition(y[index_train],
                                       p=prob,list=F)
  index_train=setdiff(seq_along(y[index_train]),index_cal)
  Xtrain=dta%>%slice(index_train)%>%select(-target)
  Xtrain=list(Xtrain)
  Ytrain=dta%>%slice(index_train)%>%pull(target)
  Ytrain=list(Ytrain)
  Xtest=dta%>%slice(index_test)%>%select(-target)
  Xtest=list(Xtest)
  Ytest=dta%>%slice(index_test)%>%pull(target)
  Ytest=list(Ytest)
  Xcal=dta%>%slice(index_cal)%>%select(-target)
  Xcal=list(Xcal)
  Ycal=dta%>%slice(index_cal)%>%pull(target)
  Ycal=list(Ycal)
  data_list=list("Xtrain"=Xtrain,"Ytrain"=Ytrain,"Xtest"=Xtest,"Ytest"=Ytest,
                 "Xcal"=Xcal,"Ycal"=Ycal)
  return(data_list)
  
}
SL_crossval_conformal=function(dta,target,folds=10){
  index_crossval=caret::createFolds(dta%>%pull(target),k=folds,returnTrain = F)
  index_folds_cal=sample(seq_along(index_crossval))
  while(!all(index_folds_cal!=seq_along(index_crossval))){
    index_folds_cal=sample(seq_along(index_crossval))
  }
  index_cal=index_crossval[index_folds_cal]
  Xtrain=purrr::map(seq_along(index_crossval),
                    function(i){
                      dta%>%slice(
                        -union(index_crossval[[i]],index_cal[[i]])
                                  )%>%
                        dplyr::select(-target)
                    })
  Ytrain=purrr::map(seq_along(index_crossval),
                    function(i){
                      dta%>%slice(
                        -union(index_crossval[[i]],index_cal[[i]])
                        )%>%pull(target)
                    })
  
  Xtest=purrr::map(seq_along(index_crossval),
                   function(i){
                     dta%>%slice(index_crossval[[i]])%>%dplyr::select(-target)
                   })
  Ytest=purrr::map(seq_along(index_crossval),
                   function(i){
                     dta%>%slice(index_crossval[[i]])%>%pull(target)
                   })
  
  Xcal=purrr::map(seq_along(index_cal),
                  function(i){
                    dta%>%slice(index_cal[[i]])%>%
                      dplyr::select(-target)
                  })
  
  Ycal=purrr::map(seq_along(index_cal),
                  function(i){
                    dta%>%slice(index_cal[[i]])%>%pull(target)
                  })
  
  data_list=list("Xtrain"=Xtrain,"Ytrain"=Ytrain,"Xtest"=Xtest,"Ytest"=Ytest,
                 "Xcal"=Xcal,"Ycal"=Ycal)
  return(data_list)
}

SL_train_conformal=function(SLmodel,dta,verbose=T,...){
  
  
  mod=purrr::map(seq_along(dta$Ytrain),
                 function(i){
                   
                   if(verbose){
                     cat("Folds",i,"\n")
                     tictoc::tic("Elapsed Time:")
                     res=SLmodel(dta$Ytrain[[i]],dta$Xtrain[[i]],...)
                     tictoc::toc()
                     return(res)
                   }else{
                     SLmodel(dta$Ytrain[[i]],dta$Xtrain[[i]],...)
                   }
                   
                 })
  # browser()
  coefSL=purrr::map(seq_along(mod),function(i) coef(mod[[i]]))
  pred=purrr::map(seq_along(mod),function(i){
    predict_SuperLearner(mod[[i]],dta$Xtest[[i]])
  } )
  cal_pred=purrr::map(seq_along(mod),function(i){
    predict_SuperLearner(mod[[i]],dta$Xcal[[i]])
  })
  return(list("prediction"=pred,"coef_SL"=coefSL,"truth"=dta$Ytest,
              "cal_prediction"=cal_pred,"cal_truth"=dta$Ycal))
}



SLperformance_conformal=function(train,measure,type="SL",conf_inv=0.95){
  if(type=="SL"){
    
    prediction=purrr::invoke(c,purrr::map(seq_along(train$prediction),function(i){
      train$prediction[[i]]$pred
    }))
    
    cal_pred=purrr::invoke(c,purrr::map(seq_along(train$cal_prediction),function(i){
      train$cal_prediction[[i]]$pred
    }))
#    browser()
    lab_prediction=binary_enc(prediction,code = c(1,0))
    lab_calibation=binary_enc(cal_pred,code = c(1,0))
    truth=purrr::invoke(c,purrr::map(seq_along(train$truth),function(i){
      train$truth[[i]]
    }))
    
    conf_pred=conformal_pred(prob_val = cal_pred,prob_test = prediction,
                   lab_val = lab_calibation,lab_test = lab_prediction,
                   truth = truth,conf_inv = conf_inv)
    
    #truth1=factor(truth,levels = c("0","01","1","null"))
  
    pred_set1=factor(conf_pred$result$pred_set,levels = c("0","01","1","null"))
    pred_set_hi=case_when(pred_set1=="01"&truth==1~1,
              pred_set1=="01"&truth==0~0,
              pred_set1=="null"&truth==1~0,
              pred_set1=="null"&truth==0~1,
              pred_set1=="0" ~ 0,
              pred_set1=="1" ~ 1
              )
    pred_set_lo=case_when(pred_set1=="01"&truth==1~0,
                          pred_set1=="01"&truth==0~1,
                          pred_set1=="null"&truth==1~0,
                          pred_set1=="null"&truth==0~1,
                          pred_set1=="0" ~ 0,
                          pred_set1=="1" ~ 1
    )
    
    metric=as.data.frame(
      cbind("low"=measure(truth,pred_set_lo),"high"=measure(truth,pred_set_hi))
    )
    res=list("conformal_result"=conf_pred,"metric"=metric)
    return(res)
  }else{
    #Super Learner Prediction
    prediction=purrr::invoke(c,purrr::map(seq_along(train$prediction),function(i){
      train$prediction[[i]]$pred
    }))
    prediction=binary_enc(prediction,code = c(1,0))
    truth=purrr::invoke(c,purrr::map(seq_along(train$truth),function(i){
      train$truth[[i]]
    }))
    SL_prediction=measure(truth,prediction)
    #Prediction each folds
    truth=train$truth
    prediction=purrr::map(seq_along(train$prediction),function(i){
      train$prediction[[i]]$pred
    })
    prediction=purrr::map(prediction,binary_enc,code = c(1,0))
    folds_prediction=purrr::map2_dbl(truth,prediction,measure)
    return(list("final_prediction"=SL_prediction,
                "folds_prediction"=folds_prediction))
  }
}
