library(googledrive)
drive_deauth()
#pth="https://github.com/gdito7/SLcuckoo/raw/master/shining-courage-250908-48d39d76c507.json"
pth="https://www.dropbox.com/s/rp6h3k27kgn7bch/shining-courage-250908-48d39d76c507.json?dl=1"
drive_auth(path =pth,
           use_oob = TRUE)
httr::set_config(httr::config(http_version = 1.1))
a=drive_find()

#drive_download(paste0(names_dataset[-c(2,8)][1],"_mcs_res_try4.rds"),
#               paste0(names_dataset[-c(2,8)][1],"_mcs_res_try4.rds"),
#               overwrite = T,verbose = F)
#a=readRDS(paste0(names_dataset[-c(2,8)][1],"_mcs_res_try4.rds"))
#a1=list("Best_Result"=a[[50]]$temp_rest$best,"All_Result"=a)
#saveRDS(a1,paste0(names_dataset[-c(2,8)][1],"_final_CSSL_try4.rds"))
#drive_upload(paste0(names_dataset[-c(2,8)][1],"_final_CSSL_try4.rds"),
#             overwrite = T,verbose=F)

names_dwd_CSSL=paste0(names_dataset[-c(2,8)],"_final_CSSL_try4.rds")
load_CSSL=map(names_dwd_CSSL,function(names){
  drive_download(names,paste0("/cloud/project/CSSL/",names),overwrite=T)
  readRDS(paste0(
    "/cloud/project/CSSL/",
    names
  ))
  
})


best_param=purrr::map(seq_along(load_CSSL),function(i) {
  load_CSSL[[i]]$Best_Result$best_solution
}
)
openxlsx::write.xlsx(best_param,"cuckoo_param.xlsx")

result_CSSL1=map(seq_along(dtaSL2[-c(2,8)]),function(j){
  
      configureMlr(show.learner.output = FALSE, show.info = FALSE)
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
                decay=best_param[[j]]$par62,MaxNWts=10000),
      logreg=list(model=T),
      OneR=list(B=best_param[[j]]$par81),
      IBk=list(K=best_param[[j]]$par91)
    )
    SL_base1save=list()
    for(i in seq_along(list_lrn1)){
      SL_base1=  create.Learner("SL.mlr",
                                params = list(learner=list_lrn1[i],
                                              par.vals= base_param[i]),
                                name_prefix = list_lrn1[i])
      SL_base1save[[i]]=SL_base1
    }
    
    base_lrn=stringr::str_c(list_lrn1,"_1")
    
    target=target_data[c(-2,-8)][j]
    YY=dtaSL2[c(-2,-8)][[j]]%>%pull(target)
    XX=dtaSL2[c(-2,-8)][[j]]%>%select(-target)
    cat(names_dataset[-c(2,8)][j],"\n")
    tictoc::tic("CV")
    set.seed(1710)
    mod=suppressWarnings(CV.SuperLearner(Y=YY,X=XX,family = binomial(),
                                         SL.library = base_lrn,method="method.AUC",
                                         verbose = T,
                                         cvControl = list(validRows=dta_task2[-c(2,8)][[j]]$rinst$test.inds),
                                         innerCvControl = list(list(V = 10L,
                                                                    stratifyCV = TRUE,
                                                                    shuffle = TRUE))
    )
    )
    
    tictoc::toc()
    tictoc::tic("Prediction")
    set.seed(1710)
    mod1=suppressWarnings(SuperLearner(Y=YY,X=XX,family = binomial(),
                                       SL.library = base_lrn,method="method.AUC",
                                       verbose = F,
                                       cvControl = list(list(V = 10L,
                                                             stratifyCV = TRUE,
                                                             shuffle = TRUE))
    ))
    tictoc::toc()
    return(list("CV"=mod,"Pred"=mod1))
  }

)
saveRDS(result_CSSL1,"CSSL_fix1.rds")

coef_CSSL=map(1:8,function(i){
readRDS("CSSL_fix1.rds")[[i]]$CV$coef
})%>%magrittr::set_names(names_dataset[-c(2,8)])
openxlsx::write.xlsx(coef_CSSL,"coef_CSSL.xlsx")

CV_CSSL_import=map(1:8,function(i){
  summary(readRDS("CSSL_fix1.rds")[[i]]$CV)
}
)
gc()
CV_CSSL=map_dfc(CV_CSSL_import,function(SL) SL$Table[,2])%>%
  magrittr::set_colnames(names_dataset[-c(2,8)])%>%
  mutate(Model=c("CSSL_1","CSDSL_1",str_remove(list_lrn1,"classif.")))%>%
  select(Model,1:8)
openxlsx::write.xlsx(CV_CSSL,"CV_CSSL.xlsx")


#result_CSSL1=readRDS("CSSL_fix1.rds")

pred_CSSL_import1=map_dfc(result_CSSL1,function(i){
  coef(i$Pred)
}
)%>%magrittr::set_colnames(names_dataset[-c(2,8)])
openxlsx::write.xlsx(pred_CSSL_import1,"coef_CSSL_pred.xlsx")

evalSL=function(SLmod1){
  predSL=map(seq_along(SLmod1),function(i){
    target=target_data[c(-2,-8)][i]
    index_test=dta_task2[c(-2,-8)][[i]]$outTest
    predict_SuperLearner(SLmod1[[i]]$Pred,newdata = dtaSL[c(-2,-8)][[i]]%>%
                           select(-target)%>%slice(index_test))$pred
    
  }
  )
  truth=map(seq_along(SLmod1),function(i){
    target=target_data[c(-2,-8)][i]
    index_test=dta_task2[c(-2,-8)][[i]]$outTest
    dtaSL[c(-2,-8)][[i]]%>%slice(index_test)%>%
      pull(target)
  }
  )
  
  predLib=map(seq_along(SLmod1),function(i){
    target=target_data[c(-2,-8)][i]
    index_test=dta_task2[c(-2,-8)][[i]]$outTest
    predict_SuperLearner(SLmod1[[i]]$Pred,newdata = dtaSL[c(-2,-8)][[i]]%>%
                           select(-target)%>%
                           slice(index_test))$library.predict
  })
  predSL1=map(predSL,function(predSL) ifelse(predSL>0.5,0,1))
  predLib1=map(predLib,function(predLib) ifelse(predLib>0.5,0,1))
  
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
  
  metric_SL=list("auc"=
                   map_dfc(seq_along(SLmod1),function(i){
                     auc1(predSL[[i]],truth[[i]])  
                   })%>% magrittr::set_colnames(names_dataset[-c(2,8)]),"gmean"=
                   map_dfc(seq_along(SLmod1),function(i){
                     GMEAN1(truth[[i]],predSL1[[i]])
                   })%>%magrittr::set_colnames(names_dataset[-c(2,8)]),"recall"=
                   map_dfc(seq_along(SLmod1),function(i){
                     recall1(truth[[i]],predSL1[[i]])
                   })%>%magrittr::set_colnames(names_dataset[-c(2,8)]),"spec"=
                   map_dfc(seq_along(SLmod1),function(i){
                     spec1(truth[[i]],predSL1[[i]])
                   })%>%magrittr::set_colnames(names_dataset[-c(2,8)])
  )
  metric_Lib=list("auc"=
                    do.call(cbind,map(seq_along(SLmod1),function(k)
                      apply(predLib[[k]],2,function(i) auc1(i,truth[[k]]))
                    ))%>%as.data.frame%>%
                    magrittr::set_colnames(names_dataset[-c(2,8)])%>%
                    tibble::rownames_to_column(var="name"),
                  "gmean"=
                    do.call(cbind,map(seq_along(SLmod1),function(k)
                      apply(predLib[[k]],2,function(i) auc1(i,truth[[k]]))
                    ))%>%as.data.frame%>%
                    magrittr::set_colnames(names_dataset[-c(2,8)])%>%
                    tibble::rownames_to_column(var="name"),
                  "recall"=
                    do.call(cbind,map(seq_along(SLmod1),function(k)
                      apply(predLib[[k]],2,function(i) auc1(i,truth[[k]]))
                    ))%>%as.data.frame%>%
                    magrittr::set_colnames(names_dataset[-c(2,8)])%>%
                    tibble::rownames_to_column(var="name"),
                  "spec"=do.call(cbind,map(seq_along(SLmod1),function(k)
                    apply(predLib[[k]],2,function(i) auc1(i,truth[[k]]))
                  ))%>%as.data.frame%>%
                    magrittr::set_colnames(names_dataset[-c(2,8)])%>%
                    tibble::rownames_to_column(var="name")
  )
  return(list("SL"=metric_SL,"base"=metric_Lib))
}


fix_finCSSL=evalSL(result_CSSL1)
fix_finCSSL1=map(1:4,function(i){
  rbind(cbind("name"="CSSL",fix_finCSSL$SL[[i]]),fix_finCSSL$base[[i]])
})%>%magrittr::set_names(c("auc","gmean","recall","spec"))
openxlsx::write.xlsx(fix_finCSSL1,"pred_CSSL.xlsx")


a=readRDS("SuperLearner_fix.rds")[[2]]$CV
map(1:10,function(i) a$AllSL[[1]]$libraryNames[which.min(a$AllSL[[i]]$cvRisk)])



all_res1=cbind("Nama data"=names_dataset[-c(2,8)],map_dfr(seq_along(load_CSSL),function(i) {
  data.frame("Nilai_AUC"=load_CSSL[[i]]$Best_Result$fitness)
}
))
openxlsx::write.xlsx(all_res1,"Cuckoo Result.xlsx")


