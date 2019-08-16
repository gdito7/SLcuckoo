r_bcs=function(fun,d,name_var="x",n=25,pa=0.25,alpha=1,Beta=1.5,iter_max=250,
               verbose=T,parallel=F,num_cores=NULL,binary="sigmoid",
               primary_out=NULL,save=F,save_files="bcs_res.rds")
  {
  # fun is objective function
  # d is number of dimenision for solution (egg)
  # n is number of host nests
  # pa is probability of cuckoo egg being discovered
  # alpha is scaling parameter
  # Beta is levy flight parameter
  # iter_max is maximum iteration
  # verbose will show result each iteration when set to TRUE
  
  #====================================Set Parallel==============================
  if(parallel){
    
    if(is.null(num_cores)){
      cl = (future::availableCores()-1)
    }else{
      cl=num_cores
    }
    future::plan(future::multiprocess,workers=cl,gc=T)
  }
  #====================================Evaluate Fitness==============================
  df2list <- function(df) {
    purrr::pmap(as.list(df), list)
  }
  fitness=function(x,fun){
    
    df2list(x)%>%furrr::future_map_dbl(function(x)purrr::invoke(fun,x))
  }
  
  
  #====================================Transformation==============================
  sigmoid=function(x){1/(1+exp(-x))}
  
  tanh_ed=function(x){
    y=tanh(x)
    y2=(min(y)-y)/(min(y)-max(y))
    return(y2)
  }
  if(binary=="tanh"){
    bin_func=tanh_ed
  }else{
    bin_func=sigmoid
  }
  rnd_seed=function(n,seed){ 
    set.seed(seed)
    x=runif(n)
    set.seed(NULL)
    return(x)
    }
  binary_trans=function(x,seed){
    u=purrr::map_dfc(seq(ncol(x)),function(i) rnd_seed(nrow(x),seed=seed))
    x1=bin_func(x)
    crit=x1>=u
    crit0=x1<u
    x1[crit]=1
    x1[crit0]=0
    return(as_tibble(x1))
  }
  
  #====================================Levy Flight============================== 
  # Based on Mantegna's algorithm in book Nature-Inspired Optimization Algorithms
  # n is sample taken from levy flight
  # Beta is scaling parameter (1<Beta<2)
  
  levy_flight = function(n,d,Beta) {
    sigma_u = ((gamma(1 + Beta) * sin(pi * Beta / 2)) /
                 (gamma((1 + Beta) / 2) * Beta * (2 ^ ((Beta - 1) / 2)))) ^ (1 / Beta)
    sigma_v = 1
    
    u = rnorm(n, 0, sigma_u)
    v = rnorm(n, 0, sigma_v)
    
    s= u / (abs(v) ^ (1 / Beta))
    s1=purrr::map_dfc(seq(d),function(i){s})
    return(s1)
  }
  
  #====================================Set limit boundary of solution==============================
  
  limit_bound=function(xx,bnd){
    
    lower=bnd%>%purrr::transpose()%>%
      magrittr::extract2(1)%>%
      as.data.frame%>%
      slice(rep(1:n(),each=nrow(xx)))
    upper=bnd%>%purrr::transpose()%>%magrittr::extract2(2)%>%
      as.data.frame%>%
      slice(rep(1:n(),each=nrow(xx)))
    xx[xx<lower]=lower[xx<lower]
    xx[xx>upper]=upper[xx>upper]
    return(xx)
  }
  #====================================Get Host egg==============================
  
  egg=function(n){
    bnd%>%purrr::transpose()%>%purrr::pmap_dfc(
      function(xmin,xmax){runif(n,min = xmin,max = xmax)}
    )
    
  }
  
  #====================================Get Cuckoo egg==============================
  
  cuckoo_egg=function(x0,seed){
    x0_bin=binary_trans(x0,seed = seed)
    cat("evaluating host fitnees value... ")
    x0_fit=fitness(x0_bin,fun)
    cat("done \n ")
    best_egg=x0%>%slice(which.max(x0_fit))%>%
      slice(rep(1:n(),each=nrow(x0)))
    
    x_cuckoo=x0+alpha*levy_flight(n,d,Beta = Beta)*(x0-best_egg)
    x_cuckoo=limit_bound(x_cuckoo,bnd)
    output=list("cuckoo"=x_cuckoo,"host_fit"=x0_fit)
    return(output)
  }
  #====================================New Generation==============================
  
  new_gen=function(x_cuckoo,x_host,host_fit,seed){
    x_cuckoo_bin=binary_trans(x_cuckoo,seed=seed)
    cat("evaluating cuckoo fitnees value... ")
    cuckoo_fit=fitness(x_cuckoo_bin,fun)
    cat("done \n")
    x_new=x_host
    replace_egg=cuckoo_fit>host_fit
    x_new[replace_egg,]=x_cuckoo[replace_egg,]
    return(x_new)
    
  }
  #====================================Empty Nest==============================
  
  empty_nest=function(x_new,pa){
    
    epsilon=purrr::map_dfc(seq(ncol(x_new)),function(x) runif(nrow(x_new)))
    crit=epsilon>pa
    H=fBasics::Heaviside(pa-epsilon)
    random1=sample.int(length(x_new))
    random2=sample.int(length(x_new))
    x_newest=x_new+alpha*levy_flight(n,d,Beta = Beta)*H*
      (x_new[random1]-x_new[random2])*crit
    x_newest=limit_bound(x_newest,bnd)
    
    return(x_newest)
  }
  
  
  #===================================Make computational Time Information==============================
  pb = progress::progress_bar$new(
    format = " Elapsed time: :elapsedfull time completion: :eta",
    clear = FALSE, width= 180,total = iter_max)
  

  #====================================Generate initial n host nest==============================
  bnd=rep(list(c(-10,10)),d)%>%
    magrittr::set_names(stringr::str_c(name_var,seq(d)))
  x_host=egg(n)
  
  #====================================Output Options==============================
  
  if(is.null(primary_out)){
    x0_host=binary_trans(x_host,seed = 1)
    fit_test=purrr::invoke(fun,x0_host%>%df2list%>%magrittr::extract2(1))
    if(length(fit_test)>1){
      stop("function output has more than one output. Argument primary_out must not be empty")
    }
  }else{
    fun1=fun
    if(!is.numeric(primary_out)){
      stop("primary_out must be numeric value")
    }
    fun=function(...){
      return(fun1(...)%>%magrittr::extract2(primary_out))
    }
    fitness1=function(x,fun){
      
      df2list(x)%>%furrr::future_map(function(x) purrr::invoke(fun,x))
    }
    extract_fitness=function(newest_fit,primary_out){
      map_dbl(newest_fit,function(x)x%>%magrittr::extract2(primary_out))
    }
    
  }
  all_res=list()
  #====================================Main Program==============================
  cat("=================Starting iteration=============== \n")
  for(i in seq(iter_max)){
    seed=runif(1)
    #Get cuckoo egg
    x_cuckoo=cuckoo_egg(x_host,seed = seed)
    #Replace bad host egg with cuckoo egg
    x_new=new_gen(x_cuckoo$cuckoo,x_host,x_cuckoo$host_fit,seed = seed)
    #Discovery by host bird: abandon nest and build new one
    x_newest=empty_nest(x_new,pa)
    x_newest_bin=binary_trans(x_newest,seed = seed)
    
    
    #find current best
    
    if(!is.null(primary_out)){
      cat("evaluating current fitnees value... ")
      newest_fit1=fitness1(x_newest_bin,fun1)
      cat("done \n\n")
      newest_fit=extract_fitness(newest_fit1,primary_out)
      newest_fit_all=newest_fit1%>%
        magrittr::extract2(which.max(newest_fit))
      
      best=list("fitness"=newest_fit%>%
                  magrittr::extract(which.max(newest_fit)),
                "best_solution"=x_newest_bin%>%slice(which.max(newest_fit)))
      
      temp_res=list("x_host"=x_host,"best"=best,"iteration"=i)
      
      all_res[[i]]=list("temp_rest"=temp_res,"all_fit"=newest_fit_all)
    }else{
      cat("evaluating current fitnees value... ")
      newest_fit=fitness(x_newest_bin,fun)
      cat("done \n\n")
      best=list("fitness"=newest_fit%>%
                  magrittr::extract(which.max(newest_fit)),
                "best_solution"=x_newest_bin%>%slice(which.max(newest_fit)))
      
      temp_res=list("x_host"=x_host,"best"=best,"iteration"=i)
      
      all_res[[i]]=temp_res
    }
    
    if(save){saveRDS(temp_res,save_files)}
    
    # keep best solution
    x_host=x_newest
    
    if(verbose){
      cat("iteration: ",i,"fitness: ",best$fitness,"\n")
      pb$tick()
      cat("\n")
      
    }
    
  }
  cat("====================Completed=================== \n")
#====================================Stop Paralel==============================  
  if(parallel){
    future::plan(future::sequential())
  }
  
  #====================================Output==============================
  if(!is.null(primary_out)){
  fin_result=list("Best_Result"=best,"All_Result"=all_res)
  }else{
    fin_result=best
  }
  return(fin_result)
}