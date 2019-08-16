cons_ls=function(X,Y,lower=0,upper=Inf,total=1,solver="ECOS"){
  require(CVXR)
  betavar=Variable(ncol(X))
  N=nrow(X)
  obj=(1/(2*N))*((norm2(Y - cbind(X) %*% betavar))^2)
  if(!is.null(total)){
    cons1=sum(betavar)==total
  }else{cons1=NULL}
  
  if(!is.infinite(lower)){
    cons2=betavar>lower
  }else{cons2=NULL}
  if(!is.infinite(upper)){
    cons3=betavar<upper
  }else{cons3=NULL}
  criteria=c(is.null(cons1),is.null(cons2),is.null(cons3))
  cons_all=list(cons1,cons2,cons3)
  cons_all=cons_all[!criteria]
  problem = Problem(Minimize(obj),constraints = cons_all )
  result = solve(problem,solver=solver)
  return(list("solution"=result$getValue(betavar)[,1] %>% round(3),
              "status"=result$status,"solver"=result$solver))
}
