source("r_cs.R")
source("r_bcs.R")
library(mlr)
library(dplyr)
# Easom's Test function
bound1=list(x=c(-100,100),y=c(-100,100))
fun1=function(x,y){-1*( -cos(x)*cos(y)*exp(-( (x-pi)^2 )-(( y-pi )^2) )) }

opt1_1=r_cs(fun1,bound1,iter_max = 10,save = T)
zz2=readRDS("cs_res.rds")
opt1_11=resume_rcs(zz2,fun1,bound1,iter_max = 50,save = T)



# bivariate Michalewicz function
bound2=list(x=c(0,5),y=c(0,5))
fun2=function(x,y,m=10){-1*( (-sin(x)*(sin((x^2)/pi))^(2*m))-
                               (sin(y)*(sin((2*(y)^2)/pi))^(2*m)) ) }

opt2=r_cs(fun2,bound2,n=30,verbose = T,iter_max = 70)
opt2









#SVM
dta=read.csv("dataset//Pima Indian diabetes.csv")

dta=dta%>%mutate(Outcome=factor(Outcome))

fun31=function(){
  task1=makeClassifTask(data=dta,target = "Outcome")
  lrn1= makeLearner("classif.ksvm")
  mod1=crossval(lrn1,task=task1,iters=10,stratify = T,measures = acc)
  list(mod1$aggr,mod1$measures.test)
}

configureMlr(show.learner.output = FALSE, show.info = FALSE)

bound3=list(C=c(0.1,1000))
fun32=function(C){
  task1=makeClassifTask(data=dta,target = "Outcome")
  lrn1= makeLearner("classif.ksvm",par.vals = list(C=C))
  mod1=crossval(lrn1,task=task1,iters=10,stratify = T,measures = acc)
  list(mod1$aggr,mod1$measures.test)
}

tictoc::tic()
opt3=r_cs(fun32,bound3,iter_max = 2,n=15,primary_out = 1,
          save = T)
tictoc::toc()

sv=readRDS("cs_res.rds")

tictoc::tic()
opt3_rsm=resume_rcs(sv,fun32,bound3,
                    iter_max = 5,n=15,
                    primary_out = 1, save=T)
tictoc::toc()


#=================================================================================
#Random Forest

fun41=function(){
  task1=makeClassifTask(data=dta,target = "Outcome")
  lrn1= makeLearner("classif.ranger")
  mod1=crossval(lrn1,task=task1,iters=10,stratify = T,measures = acc)
  list(mod1$aggr,mod1$measures.test)
}

configureMlr(show.learner.output = FALSE, show.info = FALSE)

bound4=list(mtry=c(1,(ncol(dta)-1)))
fun42=function(mtry){
  task1=makeClassifTask(data=dta,target = "Outcome")
  lrn1= makeLearner("classif.ranger",par.vals = list(mtry=mtry))
  mod1=crossval(lrn1,task=task1,iters=10,stratify = T,measures = acc)
  list(mod1$aggr,mod1$measures.test)
}

tictoc::tic()
set.seed(123)
opt41=r_dcs(fun42,bound4,iter_max = 2,n=15,primary_out = 1,
          save = F)
tictoc::toc()




#Variable Selection
set.seed(211)
X=matrix(rnorm(100*200),200,100)
betas=matrix(c(rep(1,100)),100,1)

ys=X%*%betas+rnorm(200)


var_sel=function(...){
  nvar=c(...)
  lbl=nvar*seq(ncol(X))
  X1=X[,lbl[lbl!=0]]
  mod2=lm(y~.-1,data =data.frame(y=ys,X1))
  ssm=(summary(mod2))
  #mlr::measureKendallTau(ys,mod2$fitted.values)[,1]
  ssm$adj.r.squared
}

var_sel(c(rep(1,5),rep(1,95)))
set.seed(123)

opt_var_sel=r_bcs(var_sel,d=100,iter_max = 50,alpha=1)

#0.1
0.692569
#0.01
0.6399175
#1
0.730386 
set.seed(211)
X=matrix(rnorm(20*200),200,20)
betas=matrix(c(rep(1,20)),20,1)

ys=X%*%betas+rnorm(200)

mini_var_sel=function(...){
  nvar=c(...)
  lbl=nvar*seq(ncol(X))
  X1=X[,lbl[lbl!=0]]
  mod2=lm(y~.-1,data =data.frame(y=ys,X1))
  ssm=(summary(mod2))
  #mlr::measureKendallTau(ys,mod2$fitted.values)[,1]
  ssm$adj.r.squared
}

mini_var_sel(c(rep(1,5),rep(1,15)))
set.seed(123)
opt_mini_var_sel=r_bcs(mini_var_sel,d=20,iter_max = 50,parallel = T,num_cores=10)
set.seed(123)
opta_mini_var_sel=r_abcs(mini_var_sel,d=20,iter_max = 50,parallel = F)
