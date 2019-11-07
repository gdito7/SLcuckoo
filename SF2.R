library(OpenML)
require(farff)
library(mlr)
library(tidyverse)
library(SuperLearner)

source("SL_mlr.R")
source("SL_Adds_on.R")
source("r_dcs.R")
source("resume_dcs.R")

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

result_CSF=function(j,status="first"){
  band_tree1=lapply(seq(10),function(i) c(5, floor(nrow(dtaSL2[[j]])*(4/5)) ))
  names(band_tree1)=paste0("par_tree",seq(10))
  
  cuckoo_SF=function(...){
    configureMlr(show.learner.output = FALSE, show.info = FALSE)
    par_trees=c(...)
    SL_tree1save=list()
    for(i in seq_along(par_trees)){
      SL_tree1=  create.Learner("SL.mlr",
                                params = list(learner="classif.rpart",
                                              par.vals= list(
                                                list(minsplit = par_trees[i])
                                              )
                                ),
                                name_prefix = paste0("tree","_",i))
      SL_tree1save[[i]]=SL_tree1
    }
    forest1=stringr::str_c("tree","_",seq_along(par_trees),"_1")
    
    target=target_data[c(-2,-8)][j]
    YY=dtaSL2[c(-2,-8)][[j]]%>%pull(target)
    XX=dtaSL2[c(-2,-8)][[j]]%>%select(-target)
    
    modSF=CV.SuperLearner(Y=YY,X=XX,family = binomial(),
                        SL.library = forest1,method="method.AUC",
                        verbose = F,
                        cvControl =  list(V = 5L,
                                               stratifyCV = TRUE,
                                               shuffle = TRUE),                       
                        innerCvControl = list(list(V = 5L,
                                                   stratifyCV = TRUE,
                                                   shuffle = TRUE))
    )
    res=summary(modSF)
    result=list("perf"=res$Table$Ave[1],"folds_perf"=res$Risk.SL)
    return(result)
  }
  if(status=="first"){
    
  opt_cuckoo_SF=r_dcs(cuckoo_SF,bnd = band_tree1,n=15,alpha=0.01,iter_max = 50,
                      primary_out = 1,save=T,pa=0.25,save_files=paste0(
                        "/cloud/project/Super Tree Result/ ",
                        names_dataset[-c(2,8)][j],
                        "_dcs_res_SF2.rds"),
                      online=F)
  }else{
    rds_files=readRDS(paste0("/cloud/project/Super Tree Result/ ",
                             names_dataset[-c(2,8)][j],"_dcs_res_SF2.rds"))
    opt_cuckoo_SF=resume_dcs(rds_files,cuckoo_SF,
                             bnd = band_tree1,n=15,alpha=0.01,iter_max = 50,
                             pa=0.25,
                             primary_out = 1,save=T,online=F,
                             save_files=paste0("/cloud/project/Super Tree Result/ ",
                                               names_dataset[-c(2,8)][j],
                                               "_dcs_res_SF2.rds"))
  }
  return(opt_cuckoo_SF)
}

set.seed(1710)
fin_CSF1=result_CSF(1,status = "resume")
