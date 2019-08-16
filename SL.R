library(dplyr)
library(mlr)

list_files=list.files("dataset/",all.files=F)
dta=purrr::map(list_files,function(lf){
  readr::read_csv(stringr::str_c("dataset/",lf))
}
)
names_data=stringr::str_remove(list_files,".csv")
targets=c("Class","class","Num",
          "class","live","class","class","Outcome")
targets_class=purrr::map(seq_along(targets),function(i){
  dta[[i]]%>%count(.dots=targets[i])
})
names(targets_class)=names_data
targets_class

dta_fix=dta[-c(2,6,7)]
targets_fix=targets[-c(2,6,7)]
names_data_fix=names_data[-c(2,6,7)]

postive_class=c("recurrence-events","sick",
                "present","die",1)
dta_fin=purrr::map(seq_along(dta_fix), 
                   function(i){
                     res=dta_fix[[i]]%>%mutate(!!targets_fix[i]:=
                                                 ifelse(get(targets_fix[i])==
                                                          postive_class[i],1,0))%>%
                       mutate_if(is.character,factor)%>%na.omit
                     
                     res=res%>%mutate_if(is.factor,function(i)forcats::fct_lump(i,prop=0.2))
                     class(res)="data.frame"
                     colnames(res)=make.names(colnames(res))
                     return(res)
                   }
)

#base learner accuracy
dta_base=purrr::map(seq_along(dta_fix), 
                    function(i){
                      res=dta_fix[[i]]%>%
                        mutate_if(is.character,factor)%>%na.omit
                      colnames(res)=make.names(colnames(res))
                      res=res%>%mutate_if(is.factor,function(i)forcats::fct_lump(i,prop=0.2))
                      return(res)
                    })

base_res=purrr::map_dfc(seq_along(targets_fix),function(i){                   
  task0 = makeClassifTask(data = dta_base[[1]], target = targets_fix[1])
  lrns0 = lapply(base1, makeLearner)
  lrns0 = lapply(lrns0, setPredictType, "prob")
  tictoc::tic("Base Learner")
  base_train0=lapply(lrns0,function(lrns){
    tictoc::tic(lrns$id)
    res=crossval(lrns, task = task0,measures = acc, iters=10,stratify=T)
    tictoc::toc()
    return(res)
  })
  tictoc::toc()
  
  base_acc=sapply(base_train0,function(base_train)base_train$aggr)
  names(base_acc)=NULL
  base_acc
})

base_res=as.data.frame(t(base_res))
colnames(base_res)=stringr::str_remove(base1,"classif.")
rownames(base_res)=NULL
base_res=cbind("name"=names_data_fix,base_res)
base_res

#Super Learner
#==========================================================



library(SuperLearner)
source("SL_mlr.R")
source("new_metalearner.R")
source("SL_Adds_on.R")


#============================== Base Learner =====================================
source("base_learner.R")

base1 = c("classif.ctree",
          "classif.C50","classif.J48","classif.rpart","classif.naiveBayes",
          "classif.nnet","classif.logreg","classif.OneR","classif.IBk")

base_param=list(
  ctree=list(testtype="Bonferroni",
             mincriterion=par11,
             minsplit=par12,
             minbucket=round(par12/3),
             maxdepth=par13)
  ,
  C50=list(rules=F,trials=par21,winnow=par22,
           fuzzyThreshold=TRUE,
           earlyStopping=T)
  ,
  J48=list(C=par31,M=par32)
  ,
  rpart=list(minsplit = par41,
             minbucket = round(par41/3),
             maxdepth = par42)
  ,
  naiveBayes=list(laplace=par51)
  ,
  nnet=list(size=par61,
            decay=par62),
  logreg=list(model=T),
  OneR=list(B=par81),
  IBk=list(K=par91)
)

SL_base1save=list()
for(i in seq_along(base1)){
  SL_base1=  create.Learner("SL.mlr",
                            params = list(learner=base1[i],
                                          par.vals= base_param[i]),
                            name_prefix = base1[i])
  SL_base1save[[i]]=SL_base1
}

getMlrOptions()
configureMlr(show.learner.output = FALSE, show.info = FALSE)



base_lrn1=stringr::str_c(base1,"_1")

modSL1=function(Ytrain,Xtrain){
  SuperLearner(Y=Ytrain,X=Xtrain,family = binomial(),
               SL.library = base_lrn1,method="method_consLS",
               verbose = F,
               cvControl = list(V = 10L,
                                stratifyCV = FALSE,
                                shuffle = TRUE))
}


result_SL1=purrr::map(seq_along(dta_fin), function(i){
  dta_SL=SL_crossval(dta_fin[[i]],targets_fix[i])
  
  tictoc::tic("Super Learner")
  #tictoc::tic(stringr::str_c("Super Learner without ",base1[c(6,8)]))
  trainSL1=SL_train(modSL1,dta_SL)
  tictoc::toc()
  SLperformance(trainSL1,measureACC,"all")
})

fin_resSL1=data.frame(name=names_data_fix,accuracy=
                        purrr::map_dbl(result_SL1,function(i) i$final_prediction))

