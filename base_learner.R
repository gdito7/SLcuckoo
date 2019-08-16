#=============================Conditional Tree=====================================
SL.classif.ctree=function(...)SL.mlr(...,learner = "classif.ctree"
                                     #,par.vals = list_par$ctree
)
#=========================== C50 Tree ============================================
SL.classif.C50=function(...)SL.mlr(...,learner = "classif.C50"
                                   #,par.vals = list_par$C50
)
#=========================== J48 or C45 Tree ===================================
SL.classif.J48=function(...)SL.mlr(...,learner = "classif.J48"
                                   #,par.vals = list_par$C50
)
#=========================== Classification and Regression  Tree ===================================
SL.classif.rpart=function(...)SL.mlr(...,learner = "classif.rpart"
                                     #,par.vals = list_par$C50
)
#=========================== Naive Bayes =========================================
SL.classif.naiveBayes=function(...)SL.mlr(...,learner = "classif.naiveBayes"
                                          #,par.vals = list_par$naiveBayes
)
#============================= Neural Network ====================================
SL.classif.nnet=function(...)SL.mlr(...,learner = "classif.nnet"
                                       #,par.vals = list_par$nnet
)
#=============================== Logistic Regression =============================
SL.classif.logreg=function(...)SL.mlr(...,learner = "classif.logreg"
                                      #,par.vals = list_par$logreg
)
#============================ Quadratic Discriminant Analysis ====================
SL.classif.OneR=function(...)SL.mlr(...,learner = "classif.OneR"
                                    #  ,par.vals = list(crossval=F,
                                    #                  estimate.error=F,
                                    #                   gamma=0,lambda=0)
)
#================= Instanced Based Learner or K-Nearest Neighbor ===================================
SL.classif.IBk=function(...)SL.mlr(...,learner = "classif.IBk"
                                   #,par.vals = list_par$gausspr
)
