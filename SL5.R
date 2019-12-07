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



dta_task=purrr::map(seq_along(target_data),function(i){
  target=target_data[i]
  task0 = makeClassifTask(data = dta_mlr[[i]], target = target,
                          positive=levels(dta_mlr[[i]]%>%pull(target))[2]
  )
  cat(names_dataset[i],"\n")
  set.seed(1710)
  rdesc = makeResampleDesc("CV", iters=10,stratify = T)
  rinst = makeResampleInstance(rdesc,task = task0)
  return(list("task0"=task0,"rinst"=rinst))
}
)

dtaSL=map(seq_along(dta),function(i){
  a=createDummyFeatures(dta[[i]]%>%select(-one_of(target_data[i])),method="reference")
  #a=model.matrix(~.-1,dta[[i]])
  y=createDummyFeatures(dta[[i]]%>%select(target_data[i]),method="reference")
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
  YY=dtaSL[c(-2,-8)][[j]]%>%pull(target)
  XX=dtaSL[c(-2,-8)][[j]]%>%select(-target)
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
                      cvControl = list(validRows=dta_task[-c(2,8)][[j]]$rinst$test.inds),
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
saveRDS(SLmod1,"SuperLearner_fix_ver2.rds")

coef_base=map(1:8,function(i){
  readRDS("SuperLearner_fix_ver2.rds")[[i]]$CV$coef
})%>%magrittr::set_names(names_dataset[-c(2,8)])
openxlsx::write.xlsx(coef_base,"coef_SuperLearner_fix_ver2.xlsx")

CV_base_import=map(1:8,function(i){
  summary(readRDS("SuperLearner_fix_ver2.rds")[[i]]$CV)
}
)
gc()
CV_base=map_dfc(CV_base_import,function(SL) SL$Table[,2])%>%
  magrittr::set_colnames(names_dataset[-c(2,8)])%>%
  mutate(Model=c("SL_1","DSL_1",str_remove(list_lrn1,"classif.")))%>%
  select(Model,1:8)

openxlsx::write.xlsx(CV_base,"CV_base_ver2.xlsx")

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
  rinst = dta_task[-c(2,8)][[i]]$rinst
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





SLmod2=function(j){
  configureMlr(show.learner.output = FALSE, show.info = FALSE)
  target=target_data[c(-2,-8)][j]
  YY=dtaSL[c(-2,-8)][[j]]%>%pull(target)
  XX=dtaSL[c(-2,-8)][[j]]%>%select(-target)
  cat(names_dataset[c(-2,-8)][j],"\n")
  
  SL_base1save=list()
  for(i in seq_along(famous_lrn1)){
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
  mod=suppressWarnings(CV.SuperLearner(Y=YY,X=XX,family = binomial(),
                                       SL.library = base_lrn1,method="method.AUC",
                                       verbose = T,
                                       cvControl = list(validRows=dta_task[-c(2,8)][[j]]$rinst$test.inds),
                                       innerCvControl = list(list(V = 10L,
                                                                  stratifyCV = TRUE,
                                                                  shuffle = TRUE))
  )
  )
  
  tictoc::toc()
  tictoc::tic("Prediction")
  set.seed(1710)
  mod1=suppressWarnings(SuperLearner(Y=YY,X=XX,family = binomial(),
                                     SL.library = base_lrn1,method="method.AUC",
                                     verbose = F,
                                     cvControl = list(list(V = 10L,
                                                           stratifyCV = TRUE,
                                                           shuffle = TRUE))
  ))
  tictoc::toc()
  return(list("CV"=mod,"Pred"=mod1))
}


index=8
famous_mod=SLmod2(index)
saveRDS(famous_mod,paste0("/cloud/project/SuperLearner/",
                          "FamousLearner_ver2_",names_dataset[-c(2,8)][index], ".rds")
)



CV_famous_import=map(1:8,function(i){
  summary(readRDS(paste0("/cloud/project/SuperLearner/",
                         "FamousLearner_ver2_",names_dataset[-c(2,8)][i], ".rds")
  )$CV)
}
)

coef_famous=map(1:8,function(i){
  readRDS(paste0("/cloud/project/SuperLearner/",
                 "FamousLearner_ver2_",names_dataset[-c(2,8)][i], ".rds"))$CV$coef
})%>%magrittr::set_names(names_dataset[-c(2,8)])
openxlsx::write.xlsx(coef_famous,"coef_famous_ver2.xlsx")

gc()
CV_famous=map_dfc(CV_famous_import,function(SL) SL$Table[,2])%>%
  magrittr::set_colnames(names_dataset[-c(2,8)])%>%
  mutate(Model=c("SL_2","DSL_2",str_remove(famous_lrn1,"classif.")))%>%
  select(Model,1:8)

openxlsx::write.xlsx(CV_famous,"CV_famous_ver2.xlsx")
