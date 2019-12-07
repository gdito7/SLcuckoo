library(googledrive)
drive_deauth()
#pth="https://github.com/gdito7/SLcuckoo/raw/master/shining-courage-250908-48d39d76c507.json"
pth="https://www.dropbox.com/s/rp6h3k27kgn7bch/shining-courage-250908-48d39d76c507.json?dl=1"
drive_auth(path =pth,
           use_oob = TRUE)
httr::set_config(httr::config(http_version = 1.1))


names_dwd_SF=paste0(names_dataset[-c(2,8)],"_Fin_Result_SF2.rds")

load_ST=map(names_dwd_SF,function(names) {
  drive_download(names,paste0("/cloud/project/Super Tree Result/ ",names),
                 overwrite=T)
  readRDS(paste0("/cloud/project/Super Tree Result/ ",names))
}
)

par_tree1=map(seq_along(names_dataset[-c(2,8)]),
              function(i){
                load_ST[[i]]$Best_Result$best_solution%>%as.numeric
              }
)


result_CST=map(seq_along(dtaSL2[-c(2,8)]),function(j){
  configureMlr(show.learner.output = FALSE, show.info = FALSE)
  par_trees=par_tree1[[j]]
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
  tictoc::tic("CV")
  set.seed(1710)
  mod=suppressWarnings(CV.SuperLearner(Y=YY,X=XX,family = binomial(),
                                       SL.library = forest1,method="method.AUC",
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
                                     SL.library = forest1,method="method.AUC",
                                     verbose = F,
                                     cvControl = list(list(V = 10L,
                                                           stratifyCV = TRUE,
                                                           shuffle = TRUE))
  ))
  tictoc::toc()
  return(list("CV"=mod,"Pred"=mod1))
  
})
saveRDS(result_CST,"CS_SuperTree.rds")

CV_ST_import=map(1:8,function(i){
  summary(readRDS("CS_SuperTree.rds")[[i]]$CV)
}
)
gc()
CV_ST=map_dfc(CV_ST_import,function(SL) SL$Table[,2])%>%
  magrittr::set_colnames(names_dataset[-c(2,8)])%>%
  mutate(Model=c("ST_1","DST_1",paste0("Tree",1:10)))%>%
  select(Model,1:8)
openxlsx::write.xlsx(CV_ST,"CV_ST.xlsx")

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


fix_finCST=evalSL(result_CST)
fix_finCST=map(1:4,function(i){
  rbind(cbind("name"="CSSL",fix_finCST$SL[[i]]),fix_finCST$base[[i]])
})%>%magrittr::set_names(c("auc","gmean","recall","spec"))
openxlsx::write.xlsx(fix_finCST,"pred_CST.xlsx")
