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





library(SuperLearner)
source("SL_mlr.R")
source("new_metalearner.R")
source("SL_Adds_on.R")


#============================== Base Learner =====================================
source("base_learner.R")

getMlrOptions()
configureMlr(show.learner.output = FALSE, show.info = FALSE)


base1 = c("classif.ctree",
          "classif.C50","classif.J48","classif.rpart","classif.naiveBayes",
          "classif.nnet","classif.logreg","classif.OneR","classif.IBk")

base_lrn1=stringr::str_c(base1,"_1")




result_SL2=function(j){
band_cont=list(par11=c(0.05,0.99),par31=c(0.01,0.5),par51=c(0,1000),
               par62=c(0.0001,0.5))
band_disc=list(par12=c(1,100),par13=c(1,100),par21=c(1,100),
               par32=c(1,100),
               par41=c(5,150),par42=c(1,30),par61=c(1,30),
               par81=c(1,100),par91=c(2,100))

name_bin_band=c("par22","meta1","meta2",stringr::str_c("base_learner",1:9))


cuckoo_SL=function(par11,par12,par13,par21,par22,par31,par32,
         par41,par42,par51,par61,par62,par81,par91,meta1,meta2,...)
{

base_param=list(
  ctree=list(testtype="Bonferroni",
             mincriterion=par11,
             minsplit=par12,
             minbucket=round(par12/3),
             maxdepth=par13)
  ,
  C50=list(rules=F,trials=par21,winnow=as.logical(par22),
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


nbase=c(...)
if(all(nbase==0)){
  nbase[sample(seq_along(nbase),1)]=1
}

lbl=nbase*seq_along(base_lrn1)
base_lrn2=base_lrn1[lbl[lbl!=0]]

metas=c(meta1,meta2)
if(any(duplicated(metas))){
  metas[sample(seq_along(metas),1)]=ifelse(
    metas[sample(seq_along(metas),1)]==1,0,1)
}
   

meta_lrn=c("method_consLS","method_NNRidge")
meta_lrn2=meta_lrn[metas[metas!=0]]
modSL2=function(Ytrain,Xtrain){

  SuperLearner(Y=Ytrain,X=Xtrain,family = binomial(),
               SL.library = base_lrn2,method=meta_lrn2,
               verbose = F,
               cvControl = list(V = 10L,
                                stratifyCV = FALSE,
                                shuffle = TRUE))

}



  dta_SL=SL_crossval(dta_fin[[j]],targets_fix[j])
  
#  tictoc::tic("Super Learner")
  #tictoc::tic(stringr::str_c("Super Learner without ",base1[c(6,8)]))
  trainSL2=SL_train(modSL2,dta_SL)
#  tictoc::toc()
  SLperformance(trainSL2,measureACC,"all")

}

opt_cuckoo_SL=r_mcs(cuckoo_SL,cont_bnd = band_cont,disc_bnd = band_disc,
      name_bin_bnd = name_bin_band,n=20,alpha=0.1,iter_max = 2,
      primary_out = 1)

}


fin_resSL1=data.frame(name=names_data_fix,accuracy=
                        purrr::map_dbl(result_SL1,function(i) i$final_prediction))

