library(OpenML)
require(farff)
library(mlr)
library(tidyverse)
library(SuperLearner)

source("SL_mlr.R")
source("SL_Adds_on.R")
list_files=c(37,1487,40994,29,31,1480, 1464,
             1494,1063,1068)
list_datasets=map(list_files,function(files)
  getOMLDataSet(data.id=files)
)
names_dataset=c("Pima Indians Diabetes",
                "Ozone Level",
                "Climate Model Simulation",
                "Credict Approval",
                "Credit German",
                "Indian Liver Patient",
                "Blood Transfusion",
                "QSAR Biodegeneration",
                "NASA KC2 Software defect",
                "Nasa PC1 Software defect"
)
target_data=map_chr(list_datasets,function(list_dataset) list_dataset$target.features)
dta=map(list_datasets,function(list_dataset) list_dataset$data%>%na.omit)

dta_mlr=map(seq_along(dta),function(i){
  a=model.matrix(~.-1,dta[[i]]%>%select(-one_of(target_data[i]))
  )
  a1=as.data.frame(a)
  #a1=as.data.frame(predict(a,dta[[i]]))
  colnames(a1)=make.names(colnames(a1))
  a2=cbind(dta[[i]]%>%select(target_data[i]),a1)
  return(a2)
})



dta_task2=purrr::map(seq_along(target_data),function(i){

  target=target_data[i]
  set.seed(1710)
  holdout_index=caret::createDataPartition(dta_mlr[[i]]%>%pull(target),
                                           p = 0.2,list = F
                                           )
  
  task0 = makeClassifTask(data = dta_mlr[[i]][-holdout_index,], target = target,
                          positive=levels(dta_mlr[[i]]%>%pull(target))[2]
  )
  task00 = makeClassifTask(data = dta_mlr[[i]][holdout_index,], target = target,
                          positive=levels(dta_mlr[[i]]%>%pull(target))[2]
  )
  
  cat(names_dataset[i],"\n")
  set.seed(1710)
  rdesc = makeResampleDesc("CV", iters=10,stratify = T)
  rinst = makeResampleInstance(rdesc,task = task0)
  return(list("task0"=task0,"rinst"=rinst,"task00"=
                task00,"outTest"=holdout_index))
}
)
dtaSL2=map(seq_along(dta),function(i){
  dta=dta[[i]][-dta_task2[[i]]$outTest,]
  a=createDummyFeatures(dta%>%select(-one_of(target_data[i])),method="reference")
  #a=model.matrix(~.-1,dta[[i]])
  y=createDummyFeatures(dta%>%select(target_data[i]),method="reference")
  colnames(y)=target_data[i]
  a2=cbind(y,a)
  #a1=as.data.frame(a)
  colnames(a2)=make.names(colnames(a2))
  #  a2=a1
  #colnames(a2)[str_detect(colnames(a2),target_data[i])]=target_data[i]
  return(a2)
})
list_lrn1 = c("classif.ctree","classif.C50",
              "classif.J48","classif.rpart","classif.naiveBayes",
              "classif.nnet","classif.logreg","classif.OneR",
              "classif.IBk")

SLmod1=purrr::map(seq_along(dtaSL2[-c(2,8)]),function(j){
configureMlr(show.learner.output = FALSE, show.info = FALSE)
target=target_data[c(-2,-8)][j]
YY=dtaSL2[c(-2,-8)][[j]]%>%pull(target)
XX=dtaSL2[c(-2,-8)][[j]]%>%select(-target)
cat(names_dataset[c(-2,-8)][j],"\n")

SL_base1save=list()
for(i in seq_along(list_lrn1)){
  SL_base1=  create.Learner("SL.mlr",
                            params = list(learner=list_lrn1[i]
                                          #par.vals= base_param[i]),
                            ),
                            name_prefix = list_lrn1[i])
  SL_base1save[[i]]=SL_base1
}
base_lrn1=stringr::str_c(list_lrn1,"_1")
tictoc::tic("CV")
set.seed(1710)
mod=CV.SuperLearner(Y=YY,X=XX,family = binomial(),
                      SL.library = base_lrn1,method="method.AUC",
                      verbose = T,
                      cvControl = list(validRows=dta_task2[-c(2,8)][[j]]$rinst$test.inds),
                      innerCvControl = list(list(V = 10L,
                                       stratifyCV = TRUE,
                                       shuffle = TRUE))
                    )

tictoc::toc()
tictoc::tic("Prediction")
set.seed(1710)
mod1=SuperLearner(Y=YY,X=XX,family = binomial(),
                    SL.library = base_lrn1,method="method.AUC",
                    verbose = F,
                    cvControl = list(list(V = 10L,
                                               stratifyCV = TRUE,
                                               shuffle = TRUE))
)
tictoc::toc()
return(list("CV"=mod,"Pred"=mod1))
})
saveRDS(SLmod1,"SuperLearner_fix.rds")
evalSL=function(SLmod1){
predSL=map(seq_along(SLmod1),function(i){
  target=target_data[c(-2,-8)][i]
  index_test=dta_task2[c(-2,-8)][[i]]$outTest
  predict_SuperLearner(SLmod1[[i]]$Pred,newdata = dtaSL[c(-2,-8)][[i]]%>%
                       select(-target)%>%slice(index_test))$pred
  
}
)
truth=map(seq_along(SLmod1),function(i){
  target=target_data[c(-2,-8)][i]
  index_test=dta_task2[c(-2,-8)][[i]]$outTest
  dtaSL[c(-2,-8)][[i]]%>%slice(index_test)%>%
  pull(target)
}
)

predLib=map(seq_along(SLmod1),function(i){
  target=target_data[c(-2,-8)][i]
  index_test=dta_task2[c(-2,-8)][[i]]$outTest
  predict_SuperLearner(SLmod1[[i]]$Pred,newdata = dtaSL[c(-2,-8)][[i]]%>%
                               select(-target)%>%
                         slice(index_test))$library.predict
})
predSL1=map(predSL,function(predSL) ifelse(predSL>0.5,0,1))
predLib1=map(predLib,function(predLib) ifelse(predLib>0.5,0,1))

auc1=function(probabilities,truth){
  measureAUC(probabilities,truth,positive = 0,negative=1)
}
GMEAN1=function(truth,response){
    measureGMEAN(truth,response,positive = 1,negative = 0)
  }
recall1=function(truth,response){
    measureTPR(truth,response,positive=1)
  }
spec1=function(truth,response){
    measureTNR(truth,response,negative=0)
}

metric_SL=list("auc"=
map_dfc(seq_along(SLmod1),function(i){
auc1(predSL[[i]],truth[[i]])  
})%>% magrittr::set_colnames(names_dataset[-c(2,8)]),"gmean"=
map_dfc(seq_along(SLmod1),function(i){
GMEAN1(truth[[i]],predSL1[[i]])
})%>%magrittr::set_colnames(names_dataset[-c(2,8)]),"recall"=
map_dfc(seq_along(SLmod1),function(i){
recall1(truth[[i]],predSL1[[i]])
})%>%magrittr::set_colnames(names_dataset[-c(2,8)]),"spec"=
map_dfc(seq_along(SLmod1),function(i){
spec1(truth[[i]],predSL1[[i]])
})%>%magrittr::set_colnames(names_dataset[-c(2,8)])
)
metric_Lib=list("auc"=
do.call(cbind,map(seq_along(SLmod1),function(k)
  apply(predLib[[k]],2,function(i) auc1(i,truth[[k]]))
))%>%as.data.frame%>%
  magrittr::set_colnames(names_dataset[-c(2,8)])%>%
  tibble::rownames_to_column(var="name"),
"gmean"=
  do.call(cbind,map(seq_along(SLmod1),function(k)
    apply(predLib[[k]],2,function(i) auc1(i,truth[[k]]))
  ))%>%as.data.frame%>%
  magrittr::set_colnames(names_dataset[-c(2,8)])%>%
  tibble::rownames_to_column(var="name"),
"recall"=
  do.call(cbind,map(seq_along(SLmod1),function(k)
    apply(predLib[[k]],2,function(i) auc1(i,truth[[k]]))
  ))%>%as.data.frame%>%
  magrittr::set_colnames(names_dataset[-c(2,8)])%>%
  tibble::rownames_to_column(var="name"),
  "spec"=do.call(cbind,map(seq_along(SLmod1),function(k)
  apply(predLib[[k]],2,function(i) auc1(i,truth[[k]]))
))%>%as.data.frame%>%
  magrittr::set_colnames(names_dataset[-c(2,8)])%>%
  tibble::rownames_to_column(var="name")
)
return(list("SL"=metric_SL,"base"=metric_Lib))
}


fix_finSL=evalSL(SLmod1)
rbind(cbind("name"="SL",fix_finSL$SL$auc),fix_finSL$base$auc)
summary(SLmod1[[4]]$CV)

#======================famous Learner============================================



famous_lrn1 = c("classif.ranger","classif.xgboost",
                "classif.ada","classif.cforest","classif.gausspr",
                "classif.glmboost","classif.ksvm",
                "classif.extraTrees","classif.evtree")

famous_res=purrr::map(seq_along(target_data[-c(2,8)]),function(i){      
  configureMlr(show.learner.output = FALSE, show.info = FALSE)
  lrns0 = lapply(famous_lrn1, makeLearner)
  lrns0 = lapply(lrns0, setPredictType, "prob")
  
  cat(names_dataset[-c(2,8)][i],"\n")
  rinst = dta_task2[-c(2,8)][[i]]$rinst
  base_train0=lapply(lrns0,function(lrns){
    tictoc::tic(lrns$id)
    res=resample(lrns,task=dta_task2[-c(2,8)][[i]]$task0,resampling=rinst,
                 measures = list(auc,gmean,tpr,tnr))
    tictoc::toc()
    return(res)
  })
  base_acc=map(base_train0,function(base_train)base_train$aggr)
  base_fold=map(base_train0,function(base_train)base_train$measures.test)
  #  names(base_acc)=NULL
  res=list("base_acc"=base_acc,"base_fold"=base_fold)
  return(res)
})
saveRDS(famous_res,"famous_fix.rds")

famous_pred=purrr::map(seq_along(target_data[-c(2,8)]),function(i){      
  configureMlr(show.learner.output = FALSE, show.info = FALSE)
  lrns0 = lapply(famous_lrn1, makeLearner)
  lrns0 = lapply(lrns0, setPredictType, "prob")
  
  cat(names_dataset[-c(2,8)][i],"\n")
  rinst = dta_task2[-c(2,8)][[i]]$rinst

  mod_base=lapply(lrns0,function(lrns){
    tictoc::tic(lrns$id)
    res=train(lrns,task = dta_task2[-c(2,8)][[i]]$task0)
    tictoc::toc()
    return(res)
  })
  res=do.call(rbind,map(mod_base,function(mod_base){
  pred_base=predict(mod_base,task =  dta_task2[-c(2,8)][[i]]$task00)
  perf_base=performance(pred = pred_base,measures = list(auc,gmean,tpr,tnr) )
  return(perf_base)
  }))%>%as.data.frame%>%mutate(learner=famous_lrn1)%>%select(learner,1:4)
  res=list("pref_mod"=res)
  return(res)
})
saveRDS(famous_pred,"famous_pred_fix.rds")
fin_famous_pred=map(c("auc","gmean","tpr","tnr"),
                    function(measure){
                    map_dfc(famous_pred,function(famous_pred){
  famous_pred$pref_mod%>%select(measure)
})%>%magrittr::set_colnames(names_dataset[-c(2,8)])%>%
  mutate(learner=famous_lrn1)%>%
  select(learner,1:8)
})%>%magrittr::set_names(c("auc","gmean","tpr","tnr"))




SLmod2=purrr::map(seq_along(dtaSL2[-c(2,8)]),function(j){
  configureMlr(show.learner.output = FALSE, show.info = FALSE)
  target=target_data[c(-2,-8)][j]
  YY=dtaSL2[c(-2,-8)][[j]]%>%pull(target)
  XX=dtaSL2[c(-2,-8)][[j]]%>%select(-target)
  cat(names_dataset[c(-2,-8)][j],"\n")
  
  SL_base1save=list()
  for(i in seq_along(list_lrn1)){
    SL_base1=  create.Learner("SL.mlr",
                              params = list(learner=famous_lrn1[i]
                                            #par.vals= base_param[i]),
                              ),
                              name_prefix = famous_lrn1[i])
    SL_base1save[[i]]=SL_base1
  }
  base_lrn1=stringr::str_c(famous_lrn1,"_1")
  tictoc::tic("CV")
  set.seed(1710)
  mod=CV.SuperLearner(Y=YY,X=XX,family = binomial(),
                      SL.library = base_lrn1,method="method.AUC",
                      verbose = F,
                      cvControl = list(validRows=dta_task2[-c(2,8)][[j]]$rinst$test.inds),
                      innerCvControl = list(list(V = 10L,
                                                 stratifyCV = TRUE,
                                                 shuffle = TRUE))
  )
  
  tictoc::toc()
  tictoc::tic("Prediction")
  set.seed(1710)
  mod1=SuperLearner(Y=YY,X=XX,family = binomial(),
                    SL.library = base_lrn1,method="method.AUC",
                    verbose = F,
                    cvControl = list(list(V = 10L,
                                          stratifyCV = TRUE,
                                          shuffle = TRUE))
  )
  tictoc::toc()
  return(list("CV"=mod,"Pred"=mod1))
})
saveRDS(SLmod2,"FamousLearner_fix.rds")
