library(OpenML)
require(farff)
library(mlr)
library(tidyverse)
library(SuperLearner)


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




famous_lrn1 = c("classif.ranger","classif.xgboost",
              "classif.ada","classif.cforest","classif.gausspr",
              "classif.glmboost","classif.ksvm","classif.lssvm")

configureMlr(show.learner.output = FALSE, show.info = FALSE)

famous_res=purrr::map_dfc(seq_along(target_data),function(i){                   
  task0 = makeClassifTask(data = dta_mlr[[i]], target = target_data[i])
  lrns0 = lapply(famous_lrn1, makeLearner)
#  lrns0 = lapply(lrns0, setPredictType, "prob")
  
  cat(names_dataset[i],"\n")
  tictoc::tic("Base Learner")
  set.seed(123)
  rdesc = makeResampleDesc("CV", iters=5,stratify = T)
  rinst = makeResampleInstance(rdesc,task = task0)
  base_train0=lapply(lrns0,function(lrns){
  tictoc::tic(lrns$id)
    res=resample(lrns,task=task0,resampling=rinst,
                  measures = gmean)
    tictoc::toc()
    return(res)
  })
  tictoc::toc()
  base_acc=sapply(base_train0,function(base_train)base_train$aggr)
  names(base_acc)=NULL
  base_acc
})

famous_res=as.data.frame(t(famous_res))
colnames(famous_res)=stringr::str_remove(famous_lrn1,"classif.")
rownames(famous_res)=NULL
famous_res=cbind("name"=names_dataset,famous_res)











list_lrn1 = c("classif.ctree","classif.C50",
              "classif.J48","classif.rpart","classif.naiveBayes",
              "classif.nnet","classif.logreg","classif.OneR",
              "classif.IBk")

configureMlr(show.learner.output = FALSE, show.info = FALSE)

base_res=purrr::map_dfc(seq_along(target_data),function(i){                   
  task0 = makeClassifTask(data = dta_mlr[[i]], target = target_data[i])
  lrns0 = lapply(list_lrn1, makeLearner)
  #  lrns0 = lapply(lrns0, setPredictType, "prob")
  
  cat(names_dataset[i],"\n")
  tictoc::tic("Base Learner")
  set.seed(123)
  rdesc = makeResampleDesc("CV", iters=5,stratify = T)
  rinst = makeResampleInstance(rdesc,task = task0)
  base_train0=lapply(lrns0,function(lrns){
    tictoc::tic(lrns$id)
    res=resample(lrns,task=task0,resampling=rinst,
                 measures = gmean)
    tictoc::toc()
    return(res)
  })
  tictoc::toc()
  base_acc=sapply(base_train0,function(base_train)base_train$aggr)
  names(base_acc)=NULL
  base_acc
})


base_res=as.data.frame(t(base_res))
colnames(base_res)=stringr::str_remove(list_lrn1,"classif.")
rownames(base_res)=NULL
base_res=cbind("name"=names_dataset,base_res)



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






SL_base1save=list()
for(i in seq_along(base1)){
  SL_base1=  create.Learner("SL.mlr",
                            params = list(learner=list_lrn1[i]
                                          #par.vals= base_param[i]),
                            ),
                            name_prefix = base1[i])
  SL_base1save[[i]]=SL_base1
}

configureMlr(show.learner.output = FALSE, show.info = FALSE)

base_lrn1=stringr::str_c(list_lrn1,"_1")



modSL1=function(Ytrain,Xtrain){
  SuperLearner(Y=Ytrain,X=Xtrain,family = binomial(),
               SL.library = base_lrn1,method="method_consLS",
               verbose = F,
               cvControl = list(V = 5L,
                                stratifyCV = TRUE,
                                shuffle = TRUE))
}


result_SL1=purrr::map(seq_along(dtaSL), function(i){
  dta_SL=SL_crossval(dtaSL[[i]],target_data[i],folds=5)
  
  tictoc::tic("Super Learner")
  #tictoc::tic(stringr::str_c("Super Learner without ",base1[c(6,8)]))
  trainSL1=SL_train(modSL1,dta_SL)
  tictoc::toc()
  measureGMEAN1=function(truth,response){
    measureGMEAN(truth,response,positive = 1,negative = 0)
  }
  SLperformance(trainSL1,measureGMEAN1,"all")
})

fin_resSL1=data.frame(name=names_dataset,gmean=
                        purrr::map_dbl(result_SL1,function(i) i$final_prediction))


fin_resSL2=data.frame(name=names_data,gmean=
                        purrr::map_dbl(result_SL2,function(i) i$final_prediction))





dta_SL_conf=SL_crossval_conformal(dta_fin[[1]],targets[1],folds = 10)
tictoc::tic("Super Learner")
#tictoc::tic(stringr::str_c("Super Learner without ",base1[c(6,8)]))
trainSL1_conf=SL_train_conformal(modSL1,dta_SL_conf)
tictoc::toc()
measureGMEAN1=function(truth,response){
  measureGMEAN(truth,response,positive = 1,negative = 0)
}
SLperformance_conformal(trainSL1_conf,measureGMEAN1,conf_inv = 0.95)
