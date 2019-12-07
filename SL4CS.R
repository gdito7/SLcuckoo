library(OpenML)
require(farff)
library(mlr)
library(tidyverse)
library(SuperLearner)

source("method_CSAUC.R")
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

SLCS1=purrr::map(seq_along(dtaSL2[-c(2,8)])[2],function(j){
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
                      SL.library = base_lrn1,method="method_CSAUC",
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
                    SL.library = base_lrn1,method="method_CSAUC",
                    verbose = F,
                    cvControl = list(list(V = 10L,
                                          stratifyCV = TRUE,
                                          shuffle = TRUE))
  )
  tictoc::toc()
  return(list("CV"=mod,"Pred"=mod1))
})


summaryCVSL((SLCS1[[1]]$CV))
a=readRDS("SuperLearner_fix.rds")[[2]]$CV
summary(a)
a$coef
