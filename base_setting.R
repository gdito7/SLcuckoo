
#classif.ctree
par11=0.95 #minciterion #numeric
par12=20   #minsplit #integer
par13=0    #maxdepth #integer


#classif.C50
par21=1 #trials def 1 range 1 to Inf #integer
par22=T #winnow #logical


# classif.J48
par31=0.25 #C (Prining confidence) def 0.25 range 2.22e^-16 to 1 numeric
par32=2 #M (minimum number of instance) def 2 range 1 to Inf integer

#classif.rpart
par41=20 #minsplit def 20  range 1 to Inf #integer
par42=30 #maxdepth def 30 range 1 to 30   #integer

#classif.naivebayes
par51=0 #laplace def 0 range 0 to Inf #numeric

#classif.nnet
par61=3 # def 3 size def 3 range 0 to Inf #integer
par62=0 # def 0 deacy def 0 -Inf to Inf #numeric


#OneR
par81=6 #B(minimum bucket size) def 6 range 1 to Inf #integer


#IBk
par91=1 #K(Number of neighbors) def 1 range 1 to Inf # integer

rm(par11,par12,par13,par21,par22,par23,par31,par32,par41,par51,par52,
   par71,par72,par81,par82,par91)

ff=function(a,b,c,d,...){
  median(c(abs(a),b^2,c+0.5,log(d))
}

datacb=data.frame(b=rnorm(5),c=rbinom(5,size = 10,p=0.75),
                  d=rf(5,1,1),a=rt(5,1))
?rowwise()
f_pmap_aslist(datacb)

for(i in 1:5){
  set.seed(-0.5)
  cat("print 1 \n")
  print(runif(4))
  cat("print 2 \n")
  print(runif(2))
}


table(forcats::fct_lump(cb,n=2))