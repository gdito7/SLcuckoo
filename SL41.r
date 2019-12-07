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
famous_lrn1 = c("classif.ranger","classif.xgboost",
                "classif.ada","classif.cforest","classif.gausspr",
                "classif.glmboost","classif.ksvm",
                "classif.extraTrees","classif.evtree")



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
  pred_base=map(mod_base,function(mod_base){
    pred_base=predict(mod_base,task =  dta_task2[-c(2,8)][[i]]$task00)
    return(pred_base)
  })
  res=do.call(rbind,map(pred_base,function(pred_base){
    perf_base=performance(pred = pred_base,measures = list(auc,gmean,tpr,tnr) )
    return(perf_base)
  }))%>%as.data.frame%>%mutate(learner=famous_lrn1)%>%select(learner,1:4)
  res=list("pref_mod"=res,"pred_mod"=pred_base)
  return(res)
})
saveRDS(famous_pred,"famous_pred_fix.rds")

#famous_pred=readRDS("famous_pred_fix.rds")
fin_famous_pred=map(c("auc","gmean","tpr","tnr"),
                    function(measure){
                      map_dfc(famous_pred,function(famous_pred){
                        famous_pred$pref_mod%>%select(measure)
                      })%>%magrittr::set_colnames(names_dataset[-c(2,8)])%>%
                        mutate(learner=famous_lrn1)%>%
                        select(learner,1:8)
                    })%>%magrittr::set_names(c("auc","gmean","tpr","tnr"))


pred_famous_import=map(1:8,function(i){
  target=target_data[c(-2,-8)][i]
  index_test=dta_task2[c(-2,-8)][[i]]$outTest
  p1=coef(readRDS(paste0("/cloud/project/SuperLearner/",
                                         "FamousLearner_",names_dataset[-c(2,8)][i], ".rds")
  )$Pred)
  return(list("coefSL"=p1))
}
)

pred_famous_import1=map_dfc(pred_famous_import,function(i){
  i$coefSL
}
)%>%magrittr::set_colnames(names_dataset[-c(2,8)])
openxlsx::write.xlsx(pred_famous_import1,"coef_famous_pred.xlsx")

pred_Lib_famous=map(1:8,function(j)
  map_dfc(seq_along(famous_lrn1),function(i){
  famous_pred[[j]]$pred_mod[[i]]$data[,3]
})
)

pred_SL=function(library_pred,coef_SL){
predSL=as.matrix(library_pred)%*%coef_SL$coefSL
predLib=library_pred
return(list("predSL"=predSL,"predLib"=predLib))
}

predSLres=map(1:8,function(i) {
  pred_SL(pred_Lib_famous[[i]],pred_famous_import[[i]])
  })

pred_famous=map(1:8,function(k){
  target=target_data[c(-2,-8)][k]
index_test=dta_task2[c(-2,-8)][[k]]$outTest
truth=dtaSL[c(-2,-8)][[k]]%>%slice(index_test)%>%
  pull(target)

predSL1=predSLres[[k]]$predSL
predSL2=ifelse(predSLres[[k]]$predSL>0.5,0,1)
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
res_auc=auc1(predSL1,truth)
res_gmean=GMEAN1(truth,predSL2)
res_recall=recall1(truth,predSL2)
res_spec=spec1(truth,predSL2)

res=list("auc"=res_auc,"gmean"=res_gmean,"recall"=res_recall,"rspec"=res_spec)
return(res)
})

pred_famous1=map(1:4,function(j) cbind(learner="SuperLearner2",
                   map_dfc(pred_famous,function(i) i[[j]])%>%
                   magrittr::set_colnames(names_dataset[-c(2,8)]))
)%>%magrittr::set_names(c("auc","gmean","recall","spec"))
pred_famous2=map(1:4,function(i){
  rbind(pred_famous1[[i]],fin_famous_pred[[i]])
})%>%magrittr::set_names(c("auc","gmean","recall","spec"))

openxlsx::write.xlsx(pred_famous2,"pred_famous.xlsx")
