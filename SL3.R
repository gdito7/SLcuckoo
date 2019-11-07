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


famous_lrn1 = c("classif.ranger","classif.xgboost",
              "classif.ada","classif.cforest","classif.gausspr",
              "classif.glmboost","classif.ksvm",
              "classif.extraTrees","classif.evtree")



famous_res=purrr::map(seq_along(target_data[-c(2,8)]),function(i){      
  configureMlr(show.learner.output = FALSE, show.info = FALSE)
  lrns0 = lapply(famous_lrn1, makeLearner)
#  lrns0 = lapply(lrns0, setPredictType, "prob")
  
  cat(names_dataset[-c(2,8)][i],"\n")
  rinst = dta_task[-c(2,8)][[i]]$rinst
  base_train0=lapply(lrns0,function(lrns){
  tictoc::tic(lrns$id)
    res=resample(lrns,task=dta_task[-c(2,8)][[i]]$task0,resampling=rinst,
                  measures = list(gmean,tpr,tnr))
    tictoc::toc()
    return(res)
  })
  base_acc=map(base_train0,function(base_train)base_train$aggr)
  base_fold=map(base_train0,function(base_train)base_train$measures.test)
  #  names(base_acc)=NULL
  res=list("base_acc"=base_acc,"base_fold"=base_fold)
  return(res)
})
saveRDS(famous_res,file="famous_result.rds")
famous_res1=map(1:3,function(i)
  map_dfc(famous_res,function(famous_res) do.call(rbind,famous_res$base_acc)[,i])%>%
    t%>%
    as.data.frame%>%magrittr::set_rownames(NULL)%>%
    magrittr::set_colnames(stringr::str_remove(famous_lrn1,"classif."))%>%
    mutate(name=names_dataset[-c(2,8)])%>%select(name,1:10)
)%>%magrittr::set_names(c("G_mean","sens","spec"))

famous_res1=as.data.frame(t(famous_res))
colnames(famous_res1)=stringr::str_remove(famous_lrn1,"classif.")
rownames(famous_res1)=NULL
famous_res1=cbind("name"=names_dataset[-2],famous_res1)
openxlsx::write.xlsx(famous_res1,"famous.xlsx")
famous_res11=reshape2::melt(famous_res1,id.vars="name")
openxlsx::write.xlsx(famous_res11,"famous1.xlsx")






list_lrn1 = c("classif.ctree","classif.C50",
              "classif.J48","classif.rpart","classif.naiveBayes",
              "classif.nnet","classif.logreg","classif.OneR",
              "classif.IBk")



base_res=purrr::map(seq_along(target_data[c(-2,-8)]),function(i){  
  configureMlr(show.learner.output = FALSE, show.info = FALSE)
  lrns0 = lapply(list_lrn1, makeLearner)
  #  lrns0 = lapply(lrns0, setPredictType, "prob")
  
  cat(names_dataset[c(-2,-8)][i],"\n")
  rinst = dta_task[c(-2,-8)][[i]]$rinst
  base_train0=lapply(lrns0,function(lrns){
    tictoc::tic(lrns$id)
    set.seed(1710)
    res=resample(lrns,task=dta_task[c(-2,-8)][[i]]$task0,resampling=rinst,
                 measures = list(gmean,tpr,tnr))
    tictoc::toc()
    return(res)
  })
  tictoc::toc()
  base_acc=map(base_train0,function(base_train)base_train$aggr)
  base_fold=map(base_train0,function(base_train)base_train$measures.test)
#  names(base_acc)=NULL
  res=list("base_acc"=base_acc,"base_fold"=base_fold)
  return(res)
})

saveRDS(base_res,file="base_result.rds")
base_res1=map(1:3,function(i)
map_dfc(base_res,function(base_res) do.call(rbind,base_res$base_acc)[,i])%>%t%>%
  as.data.frame%>%magrittr::set_rownames(NULL)%>%
  magrittr::set_colnames(stringr::str_remove(list_lrn1,"classif."))%>%
  mutate(name=names_dataset[-c(2,8)])%>%select(name,1:10)
)%>%magrittr::set_names(c("G_mean","sens","spec"))

openxlsx::write.xlsx(base_res1,"base.xlsx")
base_res11=reshape2::melt(base_res1,id.vars="name")
openxlsx::write.xlsx(base_res11,"base1.xlsx")


#====================================SuperLearner=====================================================
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

result_SL1=purrr::map(seq_along(dtaSL[-c(2,8)])[1], function(j){
  SL_base1save=list()
  for(i in seq_along(list_lrn1)){
    SL_base1=  create.Learner("SL.mlr",
                              params = list(learner=list_lrn1[i]
                                            #par.vals= base_param[i]),
                              ),
                              name_prefix = list_lrn1[i])
    SL_base1save[[i]]=SL_base1
  }
  
  configureMlr(show.learner.output = FALSE, show.info = FALSE)
  
  base_lrn1=stringr::str_c(list_lrn1,"_1")
  
  modSL1=function(Ytrain,Xtrain){
    SuperLearner(Y=Ytrain,X=Xtrain,family = binomial(),
                 SL.library = base_lrn1,method="method.AUC",
                 verbose = F,
                 cvControl = list(V = 10L,
                                  stratifyCV = TRUE,
                                  shuffle = TRUE))
  }
  
  dta_SL=SL_crossval(dtaSL[-c(2,8)][[j]],target_data[-c(2,8)][j],fixed = TRUE,
                     index=dta_task[-c(2,8)][[j]]$rinst$test.inds)
  
  cat(names_dataset[-c(2,8)][j],"\n")
  tictoc::tic(names_dataset[-c(2,8)][j])
  
  set.seed(1710)
  trainSL1=SL_train(modSL1,dta_SL)
  tictoc::toc()
  measureGMEAN1=function(truth,response){
    measureGMEAN(truth,response,positive = 1,negative = 0)
  }
  recall1=function(truth,response){
  measureTPR(truth,response,positive=1)
  }
  spec1=function(truth,response){
  measureTNR(truth,response,negative=0)
  }
  auc1=function(probabilities,truth){
    cvAUC::cvAUC(predictions=probabilities,labels=truth)$cvAUC
  }
  
  res=list("mod"=trainSL1,
           "G_mean"=SLperformance(trainSL1,measureGMEAN1,"all"),
           "Sens"=SLperformance(trainSL1,recall1,"all"),
           "Spec"=SLperformance(trainSL1,spec1,"all"),
           "auc"=SLperformance(trainSL1,auc1,"all")
           )
  return(res)
})
saveRDS(result_SL1,"Superlearner.rds")

eval_base=function(result_SL1){
pred_base=map(result_SL1,function(k) {
  do.call(rbind,map(k$mod$prediction,function(j){
                  j$library.predict%>%
  as.data.frame%>%mutate_all(function(i) ifelse(i>0.5,0,1))
})
)
})

truth_base=map(result_SL1,function(k) {
  do.call(c,k$mod$truth)
})
measureGMEAN1=function(truth,response){
  measureGMEAN(truth,response,positive = 1,negative = 0)
}
recall1=function(truth,response){
  measureTPR(truth,response,positive=1)
}
spec1=function(truth,response){
  measureTNR(truth,response,negative=0)
}

res_gmean=do.call(rbind,map(seq_along(pred_base),function(k){
       ifelse(is.nan(
         apply(pred_base[[k]],2,function(i) measureGMEAN1(i,truth_base[[k]]))
       ),0,
       apply(pred_base[[k]],2,function(i) measureGMEAN1(i,truth_base[[k]]))
       )
}))%>%as.data.frame%>%
  magrittr::set_colnames(stringr::str_remove(list_lrn1,"classif."))%>%
  mutate(name=names_dataset[-c(2,8)])%>%select(name,1:10)

res_sens=do.call(rbind,map(seq_along(pred_base),function(k){
  ifelse(is.nan(
    apply(pred_base[[k]],2,function(i) recall1(i,truth_base[[k]]))
  ),0,
  apply(pred_base[[k]],2,function(i) recall1(i,truth_base[[k]]))
  )
}))%>%as.data.frame%>%
  magrittr::set_colnames(stringr::str_remove(list_lrn1,"classif."))%>%
  mutate(name=names_dataset[-c(2,8)])%>%select(name,1:10)

res_spec=do.call(rbind,map(seq_along(pred_base),function(k){
  ifelse(is.nan(
    apply(pred_base[[k]],2,function(i) spec1(i,truth_base[[k]]))
  ),0,
  apply(pred_base[[k]],2,function(i) spec1(i,truth_base[[k]]))
  )
}))%>%as.data.frame%>%
  magrittr::set_colnames(stringr::str_remove(list_lrn1,"classif."))%>%
  mutate(name=names_dataset[-c(2,8)])%>%select(name,1:10)

res=list("G_mean"=res_gmean,
         "Sens"=res_sens,
         "Spec"=res_spec
)
return(res)
}
base_resSL1=eval_base(result_SL1)
do.call(rbind,result_SL1[[1]]$mod$coef_SL)%>%as.data.frame%>%
  apply(MARGIN = 2,mean)
apply(do.call(rbind,result_SL1[[1]]$mod$coef_SL)%>%as.data.frame,
  2,mean)

fin_resSL1=data.frame(name=names_dataset[-c(2,8)],gmean=
                        purrr::map_dbl(result_SL1,function(i) i$G_mean$final_prediction),
                      sens=purrr::map_dbl(result_SL1,function(i) i$Sens$final_prediction),
                      spec=purrr::map_dbl(result_SL1,function(i) i$Spec$final_prediction),
                      auc=purrr::map_dbl(result_SL1,function(i) i$auc$final_prediction)
)

openxlsx::write.xlsx(fin_resSL1,"SuperLearner.xlsx")

#=====================Cuckoo Search Super Learner==============================================
library(googledrive)
drive_deauth()
#pth="https://github.com/gdito7/SLcuckoo/raw/master/shining-courage-250908-48d39d76c507.json"
pth="https://www.dropbox.com/s/rp6h3k27kgn7bch/shining-courage-250908-48d39d76c507.json?dl=1"
drive_auth(path =pth,
           use_oob = TRUE)
httr::set_config(httr::config(http_version = 1.1))
drive_find(n_max=30)

names_dwd_CSSL=paste0(names_dataset[-c(2,8)],"_final_CSSL_try3.rds")
load_CSSL=map(names_dwd_CSSL,function(names){
  drive_download(names,paste0("/cloud/project/CSSL/",names),overwrite=T)
  })

load_CSSL=map(names_dwd_CSSL,function(names){
  readRDS(paste0(
    "/cloud/project/CSSL/",
    names
  ))
})


best_param=purrr::map(seq_along(load_CSSL),function(i) {
  load_CSSL[[i]]$Best_Result$best_solution
}
)


result_SL2=purrr::map(seq_along(dtaSL[-c(2,8)]), function(j){
  dta_SL=SL_crossval(dtaSL[-c(2,8)][[j]],target_data[-c(2,8)][j],fixed = TRUE,
                     index=dta_task[-c(2,8)][[j]]$rinst$test.inds)
  base_param=list(
    ctree=list(testtype="Bonferroni",
               mincriterion=best_param[[j]]$par11,
               minsplit=best_param[[j]]$par12,
               minbucket=round(best_param[[j]]$par12/3),
               maxdepth=best_param[[j]]$par13)
    ,
    C50=list(rules=F,trials=best_param[[j]]$par21,winnow=
               ifelse(best_param[[j]]$par22==0,FALSE,TRUE),
             fuzzyThreshold=TRUE,
             earlyStopping=T)
    ,
    J48=list(C=best_param[[j]]$par31,M=best_param[[j]]$par32)
    ,
    rpart=list(minsplit = best_param[[j]]$par41,
               minbucket = round(best_param[[j]]$par41/3),
               maxdepth = best_param[[j]]$par42)
    ,
    naiveBayes=list(laplace=best_param[[j]]$par51)
    ,
    nnet=list(size=best_param[[j]]$par61,
              decay=best_param[[j]]$par62),
    logreg=list(model=T),
    OneR=list(B=best_param[[j]]$par81),
    IBk=list(K=best_param[[j]]$par91)
  )
  
  SL_base1save=list()
  
  for(k in seq_along(list_lrn1)){
    SL_base1=  create.Learner("SL.mlr",
                              params = list(learner=list_lrn1[k],
                                            par.vals= base_param[k])
                              ,
                              name_prefix = list_lrn1[k])
    SL_base1save[[k]]=SL_base1
  }
  
  configureMlr(show.learner.output = FALSE, show.info = FALSE)
  
  base_lrn1=stringr::str_c(list_lrn1,"_1")
  
  
  
  modSL1=function(Ytrain,Xtrain){
    SuperLearner(Y=Ytrain,X=Xtrain,family = binomial(),
                 SL.library = base_lrn1,method="method.NNLS",
                 verbose = F,
                 cvControl = list(V = 10L,
                                  stratifyCV = TRUE,
                                  shuffle = TRUE))
  }
  
  cat(names_dataset[-c(2,8)][j],"\n")
  tictoc::tic(names_dataset[-c(2,8)][j])
  set.seed(1710)
  trainSL1=SL_train(modSL1,dta_SL)
  tictoc::toc()
  measureGMEAN1=function(truth,response){
    measureGMEAN(truth,response,positive = 1,negative = 0)
  }
  recall1=function(truth,response){
    measureTPR(truth,response,positive=1)
  }
  spec1=function(truth,response){
    measureTNR(truth,response,negative=0)
  }
  
  
  res=list("mod"=trainSL1$coef,
           "G_mean"=SLperformance(trainSL1,measureGMEAN1,"all"),
           "Sens"=SLperformance(trainSL1,recall1,"all"),
           "Spec"=SLperformance(trainSL1,spec1,"all")
  )
  return(res)
})

fin_resSL2=data.frame(name=names_dataset[-c(2,8)],gmean=
                        purrr::map_dbl(result_SL2,function(i) i$G_mean$final_prediction),
                      sens=purrr::map_dbl(result_SL2,function(i) i$Sens$final_prediction),
                      spec=purrr::map_dbl(result_SL2,function(i) i$Spec$final_prediction))

openxlsx::write.xlsx(fin_resSL2,"CSSL.xlsx")


#=====================Super Tree==============================================



load_STpara=map(seq_along(names_dataset)[c(-2,-8)],function(j) readRDS(paste0(
  "/cloud/project/Super Tree Result/ ",
  names_dataset[j],
  "_dcs_res_SF.rds")
)
)

par_tree1=map(seq_along(names_dataset)[c(-9,-10)],
              function(i){
              load_STpara[[i]]$Best_Result$best_solution%>%as.numeric
}
)

result_ST=purrr::map(seq_along(par_tree1), function(j){
  dta_SL=SL_crossval(dtaSL[c(-2,-8)][[j]],target_data[c(-2,-8)][j],fixed = TRUE,
                     index=dta_task[c(-2,-8)][[j]]$rinst$test.inds)

  #par_tree2=seq(1,30,length.out = 15)
  par_trees=par_tree1[[j]]
  SL_tree1save=list()
  
  for(k in seq_along(par_trees)){
    SL_tree1=  create.Learner("SL.mlr",
                              params = list(learner="classif.rpart",
                                            par.vals= list(
                                              list(minsplit = par_trees[k])
                                            )
                              ),
                              name_prefix = paste0("tree","_",k))
    SL_tree1save[[k]]=SL_tree1
  }

  forest1=stringr::str_c("tree","_",seq_along(par_trees),"_1")
  modSF=function(Ytrain,Xtrain){
    SuperLearner(Y=Ytrain,X=Xtrain,family = binomial(),
                 SL.library = forest1 ,method="method.AUC",
                 verbose = F,
                 cvControl = list(V = 5L,
                                  stratifyCV = TRUE,
                                  shuffle = TRUE
                                  ))
  }
  cat(names_dataset[-c(2,8)][j],"\n")
  tictoc::tic(names_dataset[-2][j])
  #tictoc::tic(stringr::str_c("Super Learner without ",base1[c(6,8)]))
  trainSL1=SL_train(modSF,dta_SL,pkg = "mlr")
  tictoc::toc()
  
  measureGMEAN1=function(truth,response){
    measureGMEAN(truth,response,positive = 1,negative = 0)
  }
  recall1=function(truth,response){
    measureTPR(truth,response,positive=1)
  }
  spec1=function(truth,response){
    measureTNR(truth,response,negative=0)
  }
  res=list("mod"=trainSL1$coef,
           "G_mean"=SLperformance(trainSL1,measureGMEAN1,"all"),
           "Sens"=SLperformance(trainSL1,recall1,"all"),
           "Spec"=SLperformance(trainSL1,spec1,"all")
  )
  return(res)
  
})

fin_resST=data.frame(name=names_dataset[-c(2,8)],gmean=
                        purrr::map_dbl(result_ST,function(i) i$G_mean$final_prediction),
                      sens=purrr::map_dbl(result_ST,function(i) i$Sens$final_prediction),
                      spec=purrr::map_dbl(result_ST,function(i) i$Spec$final_prediction))


openxlsx::write.xlsx(fin_resST,"SuperTree.xlsx")


ST_folds_gmean=map(load_STpara, function(i)
  i$All_Result[[100]]$all_fit$folds_prediction
)
CSSL_folds_gmean=
  map(load_CSSL, function(i)
    i$All_Result[[100]]$all_fit$folds_prediction
  )
method.AUC()