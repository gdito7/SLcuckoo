library(dplyr)
library(mlr)





files_github=function(site,name_file){
  paste0(site,name_file)
}
site_files="https://raw.githubusercontent.com/gdito7/SLcuckoo/master/dataset/"
name_files=c("Breast%20Cancer.csv","Dermatology.csv","Heart%20Cleaveland.csv",
             "heart-statlog.csv","Hepatitis.csv","Lung%20Cancer.csv",
             "lymphography.csv","Pima%20Indian%20diabetes.csv")
list_files=files_github(site_files,name_files)

dta=purrr::map(list_files[-c(2,6,7)],function(lf){
  readr::read_csv(lf)
}
)

targets=c("Class","class","Num","class"
          ,"y","class","class","Outcome")
targets=targets[-c(2,6,7)]
postive_class=c("malignant","sick",
                "present","die",1)
names_data=c("Breast Cancer","Dermatology","Heart Cleaveland",
             "heart-statlog","Hepatitis","Lung Cancer",
             "lymphography","Pima Indian diabetes")
names_data=names_data[-c(2,6,7)]


dta_fin=purrr::map(seq_along(dta), 
                   function(i){
                     res=dta[[i]]%>%mutate(!!targets[i]:=
                                             ifelse(get(targets[i])==
                                                      postive_class[i],1,0))%>%
                       mutate_if(is.character,factor)%>%na.omit
                     
                     #res=res%>%mutate_if(is.factor,function(k){
                    #   prp=k%>%table%>%prop.table%>%as.vector%>%sort
                    #   if(any(prp<0.1)){
                    #     prp_crit=which(prp<0.1)
                    #     chs=length(prp_crit)
                         
                    #     if(sum(prp[prp_crit])<0.1){
                    #       prp_crit=which(prp>0.1)
                     #      chs=1
                    #       prp_chs=prp[prp_crit[chs]]
                    #       forcats::fct_lump(k,prop=prp_chs)
                    #     }else{
                    #       prp_chs=prp[prp_crit[chs]]
                    #       forcats::fct_lump(k,prop=prp_chs)
                    #     }
                    #   }else{
                    #     k
                    #   }
                    # }
                    # )
                     class(res)="data.frame"
                     colnames(res)=make.names(colnames(res))
                     return(res)
                   }
)



#base learner accuracy
dta_base=purrr::map(seq_along(dta), 
                    function(i){
                      res=dta[[i]]%>%mutate(!!targets[i]:=
                                              as.factor(ifelse(get(targets[i])==
                                                       postive_class[i],1,0)))%>%
                        mutate_if(is.character,factor)%>%na.omit
                      
                      res=res%>%mutate_if(is.factor,function(k){
                        prp=k%>%table%>%prop.table%>%as.vector%>%sort
                        if(any(prp<0.1)){
                          prp_crit=which(prp<0.1)
                          chs=length(prp_crit)
                          
                          if(sum(prp[prp_crit])<0.1){
                            prp_crit=which(prp>0.1)
                            chs=1
                            prp_chs=prp[prp_crit[chs]]
                            forcats::fct_lump(k,prop=prp_chs)
                          }else{
                            prp_chs=prp[prp_crit[chs]]
                            forcats::fct_lump(k,prop=prp_chs)
                          }
                        }else{
                          k
                        }
                      }
                      )
                      class(res)="data.frame"
                      colnames(res)=make.names(colnames(res))
                      return(res)
                    }
)

base_res=purrr::map_dfc(seq_along(targets_fix),function(i){                   
  task0 = makeClassifTask(data = dta_base[[i]], target = targets[i])
  lrns0 = lapply(base1, makeLearner)
  lrns0 = lapply(lrns0, setPredictType, "prob")
  tictoc::tic("Base Learner")
  base_train0=lapply(lrns0,function(lrns){
    tictoc::tic(lrns$id)
    res=crossval(lrns, task = task0,measures = gmean, iters=5,stratify=T)
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
          "classif.nnet","classif.logreg","classif.OneR",
          "classif.IBk")

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
                            params = list(learner=base1[i]
                                         #par.vals= base_param[i]),
                            ),
                            name_prefix = base1[i])
  SL_base1save[[i]]=SL_base1
}

configureMlr(show.learner.output = FALSE, show.info = FALSE)

base_lrn1=stringr::str_c(base1,"_1")



modSL1=function(Ytrain,Xtrain){
  SuperLearner(Y=Ytrain,X=Xtrain,family = binomial(),
               SL.library = base_lrn1,method="method_consLS",
               verbose = F,
               cvControl = list(V = 5L,
                                stratifyCV = FALSE,
                                shuffle = TRUE))
}


result_SL1=purrr::map(seq_along(dta_fin), function(i){
  dta_SL=SL_crossval(dta_fin[[i]],targets[i],folds=5)
  
  tictoc::tic("Super Learner")
  #tictoc::tic(stringr::str_c("Super Learner without ",base1[c(6,8)]))
  trainSL1=SL_train(modSL1,dta_SL)
  tictoc::toc()
  measureGMEAN1=function(truth,response){
    measureGMEAN(truth,response,positive = 1,negative = 0)
  }
  SLperformance(trainSL1,measureGMEAN1,"all")
})

fin_resSL1=data.frame(name=names_data,gmean=
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



library(googledrive)
drive_deauth()
#pth="https://github.com/gdito7/SLcuckoo/raw/master/shining-courage-250908-48d39d76c507.json"
pth="https://www.dropbox.com/s/rp6h3k27kgn7bch/shining-courage-250908-48d39d76c507.json?dl=1"
drive_auth(path =pth,
           use_oob = TRUE)
httr::set_config(httr::config(http_version = 1.1))
drive_find(n_max=30)


best_param=purrr::map(1:5,function(i) {
  res_fin_CSSL[[i]]$Best_Result$best_solution
}
)


base1 = c("classif.ctree",
          "classif.C50","classif.J48","classif.rpart","classif.naiveBayes",
          "classif.nnet","classif.logreg","classif.OneR",
          "classif.IBk")


SL_base1save=list()
for(i in seq_along(base1)){
  SL_base1=  create.Learner("SL.mlr",
                            params = list(learner=base1[i]
                                          #par.vals= base_param[i]),
                            ),
                            name_prefix = base1[i])
  SL_base1save[[i]]=SL_base1
}



base_lrn1=stringr::str_c(base1,"_1")



modSL1=function(Ytrain,Xtrain){
  SuperLearner(Y=Ytrain,X=Xtrain,family = binomial(),
               SL.library = base_lrn1,method="method_consLS",
               verbose = F,
               cvControl = list(V = 5L,
                                stratifyCV = FALSE,
                                shuffle = TRUE))
}

configureMlr(show.learner.output = FALSE, show.info = FALSE)

conf_SL1=purrr::map(seq_along(dta_fin), function(i){
  base_param=list(
    ctree=list(testtype="Bonferroni",
               mincriterion=best_param[[i]]$par11,
               minsplit=best_param[[i]]$par12,
               minbucket=round(best_param[[i]]$par12/3),
               maxdepth=best_param[[i]]$par13)
    ,
    C50=list(rules=F,trials=best_param[[i]]$par21,winnow=
               best_param[[i]]$par22,
             fuzzyThreshold=TRUE,
             earlyStopping=T)
    ,
    J48=list(C=best_param[[i]]$par31,M=best_param[[i]]$par32)
    ,
    rpart=list(minsplit = best_param[[i]]$par41,
               minbucket = round(best_param[[i]]$par41/3),
               maxdepth = best_param[[i]]$par42)
    ,
    naiveBayes=list(laplace=best_param[[i]]$par51)
    ,
    nnet=list(size=best_param[[i]]$par61,
              decay=best_param[[i]]$par62),
    logreg=list(model=T),
    OneR=list(B=best_param[[i]]$par81),
    IBk=list(K=best_param[[i]]$par91)
  )
  
  
  #dta_SL= SL_crossval_conformal(dta_fin[[i]],targets[i],folds=5)
  dta_SL=SL_holdout_conformal(dta_fin[[i]],targets[i])
  SL_base1save=list()
  for(k in seq_along(base1)){
    SL_base1=  create.Learner("SL.mlr",
                              params = list(learner=base1[k],
                                            par.vals= base_param[k])
                              ,
                              name_prefix = base1[i])
    SL_base1save[[k]]=SL_base1
  }
  set.seed(7)
  tictoc::tic("Super Learner")
  #tictoc::tic(stringr::str_c("Super Learner without ",base1[c(6,8)]))
  trainSL1=SL_train_conformal(modSL1,dta_SL)
  tictoc::toc()
  measureGMEAN1=function(truth,response){
    measureGMEAN(truth,response,positive = 1,negative = 0)
  }
  
  list("conformal"=SLperformance_conformal(trainSL1,measureGMEAN1,conf_inv = 0.95),
       "ordinary"=SLperformance(trainSL1,measureGMEAN1)
       )
})
conf_SL1_ver=purrr::map_dfr(1:5,function(i) conf_SL1[[i]]$conformal$metric)
