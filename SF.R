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


par_tree1=round(seq(5,1000,length.out = 10))
#par_tree2=seq(1,30,length.out = 15)
par_trees=par_tree1
SL_tree1save=list()
for(i in seq_along(par_tree1)){
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

modSF=function(Ytrain,Xtrain){
  SuperLearner(Y=Ytrain,X=Xtrain,family = binomial(),
               SL.library = forest1 ,method="method.NNLS",
               verbose = F,
               cvControl = list(V = 5L,
                                stratifyCV = TRUE,
                                shuffle = TRUE))
}


result_SF=purrr::map(seq_along(dtaSL)[1], function(i){
  set.seed(123)
  dta_SL=SL_crossval(dtaSL[[i]],target_data[i],folds=5)
  
  tictoc::tic("Super Learner")
  #tictoc::tic(stringr::str_c("Super Learner without ",base1[c(6,8)]))
  trainSL1=SL_train(modSF,dta_SL,pkg = "mlr")
  tictoc::toc()

  measureGMEAN1=function(truth,response){
    measureGMEAN(truth,response,positive = 1,negative = 0)
  }
  browser()
  SLperformance(trainSL1,measureGMEAN1,"all")
})


fin_resSF=data.frame(name=names_dataset,gmean=
                        purrr::map_dbl(result_SF,function(i) i$final_prediction))

result_CSF=function(j){
band_tree1=lapply(seq(10),function(i) c(5, floor(nrow(dtaSL[[j]])*(4/5)) ))
names(band_tree1)=paste0("par_tree",seq(10))

cuckoo_SF=function(...){
  #par_tree2=seq(1,30,length.out = 15)
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
  #  SL_tree1=  create.Learner("SL.rpart",
#                            tune = list(minsplit = par_tree1
                                        #,
                                        #                maxdepth = par_tree2
#                            )
#                            ,  name_prefix = "rpart")
  
  
  modSF=function(Ytrain,Xtrain){
    SuperLearner(Y=Ytrain,X=Xtrain,family = binomial(),
                 SL.library = SL_tree1$names,method="method.NNLS",
                 verbose = F,
                 cvControl = list(V = 5L,
                                  stratifyCV = TRUE,
                                  shuffle = TRUE))
  }
  
  

    dta_SL=SL_crossval(dtaSL[[j]],target_data[j],folds=5)
    
#    tictoc::tic("Super Learner")
    #tictoc::tic(stringr::str_c("Super Learner without ",base1[c(6,8)]))
    trainSL1=SL_train(modSF,dta_SL,pkg = "mlr",verbose = F)
#    tictoc::toc()
    
    measureGMEAN1=function(truth,response){
      measureGMEAN(truth,response,positive = 1,negative = 0)
    }
    SLperformance(trainSL1,measureGMEAN1,"all")
}

opt_cuckoo_SF=r_dcs(cuckoo_SF,bnd = band_tree1,n=20,alpha=0.01,iter_max = 100,
                    primary_out = 1,save=T,pa=0.05,save_files=paste0(
                      "/cloud/project/Super Tree Result/ ",
                      names_dataset[j],
                                                             "_dcs_res_SF.rds"),
                    online=F)
}

set.seed(7)
fin_CSFcb1=result_CSF(1)

set.seed(7)
fin_CSF1=result_CSF(1)
fin_CSF1$Best_Result

set.seed(7)
fin_CSF2=result_CSF(1)

set.seed(7)
fin_CSF3=result_CSF(3)

set.seed(7)
fin_CSF4=result_CSF(4)
set.seed(7)
fin_CSF5=result_CSF(5)
set.seed(7)
fin_CSF6=result_CSF(6)
set.seed(7)
fin_CSF7=result_CSF(7)
set.seed(7)
fin_CSF8=result_CSF(8)
set.seed(7)
fin_CSF9=result_CSF(9)
set.seed(1710)
fin_CSF10=result_CSF(10)

for(j in seq(1,10)){
saveRDS(get(paste0("fin_CSF",j)),
paste0(
  "/cloud/project/Super Tree Result/ ",
  names_dataset[j],
  "_dcs_res_SF.rds")
)
}
