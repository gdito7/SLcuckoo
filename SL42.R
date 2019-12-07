library(googledrive)
list_lrn1 = c("classif.ctree","classif.C50",
              "classif.J48","classif.rpart","classif.naiveBayes",
              "classif.nnet","classif.logreg","classif.OneR",
              "classif.IBk")
famous_lrn1 = c("classif.ranger","classif.xgboost",
                "classif.ada","classif.cforest","classif.gausspr",
                "classif.glmboost","classif.ksvm",
                "classif.extraTrees","classif.evtree")

drive_deauth()
#pth="https://github.com/gdito7/SLcuckoo/raw/master/shining-courage-250908-48d39d76c507.json"
pth="https://www.dropbox.com/s/rp6h3k27kgn7bch/shining-courage-250908-48d39d76c507.json?dl=1"
drive_auth(path =pth,
           use_oob = TRUE)
httr::set_config(httr::config(http_version = 1.1))

names_dwd_SL_All=paste0("AllLearner_",names_dataset[-c(2,8)], ".rds")
load_SL_All=map(names_dwd_SL_All,function(names){
  drive_download(names,paste0("/cloud/project/SL All/",names),overwrite=T)
})


CV_all_import=map(names_dwd_SL_All,function(names){
  summary(readRDS(paste0("/cloud/project/SL All/",names))$CV)
})

coef_All=map(names_dwd_SL_All,function(names){
  readRDS(paste0("/cloud/project/SL All/",names))$CV$coef
})%>%magrittr::set_names(names_dataset[-c(2,8)])
openxlsx::write.xlsx(coef_All,"coef_ALL.xlsx")

gc()
CV_all=map_dfc(CV_all_import,function(SL) SL$Table[,2])%>%
  magrittr::set_colnames(names_dataset[-c(2,8)])%>%
  mutate(Model=c("SL_3","DSL_3",str_remove(c(list_lrn1,famous_lrn1),"classif.")))%>%
  select(Model,1:8)
openxlsx::write.xlsx(CV_all,"CV_all.xlsx")

base_pred=purrr::map(seq_along(target_data[-c(2,8)]),function(i){      
  configureMlr(show.learner.output = FALSE, show.info = FALSE)
  lrns0 = lapply(list_lrn1, makeLearner)
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
saveRDS(base_pred,"base_pred_fix.rds")
gc()

pred_all_import=map(1:8,function(i){
  target=target_data[c(-2,-8)][i]
  index_test=dta_task2[c(-2,-8)][[i]]$outTest
  p1=coef(readRDS(paste0("/cloud/project/SL All/",names_dwd_SL_All[i])
  )$Pred)
  return(list("coefSL"=p1))
}
)
pred_all_import1=map_dfc(pred_all_import,function(i){
  i$coefSL
}
)%>%magrittr::set_colnames(names_dataset[-c(2,8)])
openxlsx::write.xlsx(pred_all_import1,"coef_all_pred.xlsx")

pred_Lib_allf=map(1:8,function(j)
  map_dfc(seq_along(famous_lrn1),function(i){
    famous_pred[[j]]$pred_mod[[i]]$data[,3]
  })
)
pred_Lib_allb=map(1:8,function(j)
  map_dfc(seq_along(list_lrn1),function(i){
    base_pred[[j]]$pred_mod[[i]]$data[,3]
  })
)
pred_lib_all=map(1:8,function(j){
  cbind(pred_Lib_allb[[j]],pred_Lib_allf[[j]])
})


pred_SL=function(library_pred,coef_SL){
  predSL=as.matrix(library_pred)%*%coef_SL$coefSL
  predLib=library_pred
  return(list("predSL"=predSL,"predLib"=predLib))
}

predSLres_all=map(1:8,function(i) {
  pred_SL(pred_lib_all[[i]],pred_all_import[[i]])
})

pred_all=map(1:8,function(k){
  target=target_data[c(-2,-8)][k]
  index_test=dta_task2[c(-2,-8)][[k]]$outTest
  truth=dtaSL[c(-2,-8)][[k]]%>%slice(index_test)%>%
    pull(target)
  
  predSL1=predSLres_all[[k]]$predSL
  predSL2=ifelse(predSLres_all[[k]]$predSL>0.5,0,1)
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

pred_all1=map(1:4,function(j) cbind(learner="SuperLearner3",
                                       map_dfc(pred_all,function(i) i[[j]])%>%
                                         magrittr::set_colnames(names_dataset[-c(2,8)]))
)%>%magrittr::set_names(c("auc","gmean","recall","spec"))
pred_all2=map(1:4,function(i){
  rbind(pred_all1[[i]],fix_finSL1[[i]]%>%rename("learner"="name"),fin_famous_pred[[i]])
})%>%magrittr::set_names(c("auc","gmean","recall","spec"))

openxlsx::write.xlsx(pred_all2,"pred_all.xlsx")

