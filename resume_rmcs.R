resume_rmcs=function(rmcs_files,fun,cont_bnd,disc_bnd,name_bin_bnd,n=25,pa=0.25,alpha=1,
                     Beta=1.5,iter_max=250,verbose=T,parallel=F,num_cores=NULL,
                     primary_out=NULL,save=F,save_files="mcs_res.rds"){
  
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
  
  fitness=function(x_cont,x_disc,x_bin,fun){
    x=cbind(x_cont,x_disc,x_bin)
    df2list(x)%>%furrr::future_map_dbl(function(x)purrr::invoke(fun,x))
  }
  
  #====================================Transformation==============================
  #Discrete
  extract_dec=function(x){
    x-floor(x)
  }
  rnd_seed=function(n,seed){ 
    set.seed(seed)
    x=runif(n)
    set.seed(NULL)
    return(x)
  }
  
  disc_trans=function(x,seed){
    
    u=purrr::map_dfc(seq(ncol(x)),function(i) rnd_seed(nrow(x),seed=seed))
    pp=extract_dec(x)
    crit=pp>=u
    crit0=pp<u
    x1=as.data.frame(x)
    x1[crit]=ceiling(x1[crit])
    x1[crit0]=floor(x1[crit0])
    return(as_tibble(x1))
  }
  
  #Binary
  sigmoid=function(x){1/(1+exp(-x))}
  
  binary_trans=function(x,seed){
    u=purrr::map_dfc(seq(ncol(x)),function(i) rnd_seed(nrow(x),seed=seed))
    x1=sigmoid(x)
    crit=x1>=u
    crit0=x1<u
    x1[crit]=1
    x1[crit0]=0
    return(as_tibble(x1))
  }
  
  
  #====================================Levy Flight============================== 
  #Based on Mantegna's algorithm in book Nature-Inspired Optimization Algorithms
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
  
  egg=function(n,bnd){
    bnd%>%purrr::transpose()%>%purrr::pmap_dfc(
      function(xmin,xmax){runif(n,min = xmin,max = xmax)}
    )
  }
  
  #====================================Get Cuckoo egg==============================
  
  cuckoo_egg=function(x_cont,x_disc,x_bin,
                      bnd_cont,bnd_disc,bnd_bin,seed){
    
    x0_disc=disc_trans(x_disc,seed)
    x0_bin=binary_trans(x_bin,seed)
    x0_fit=fitness(x_cont,x0_disc,x0_bin,fun)
    
    #Continous Cuckoo
    best_egg_cont=x_cont%>%
      slice(which.max(x0_fit))%>%
      slice(rep(1:n(),each=nrow(x_cont)))
    #Discrete Cuckoo
    best_egg_disc=x_disc%>%
      slice(which.max(x0_fit))%>%
      slice(rep(1:n(),each=nrow(x_disc)))
    #Binary Cuckoo
    best_egg_bin=x_bin%>%
      slice(which.max(x0_fit))%>%
      slice(rep(1:n(),each=nrow(x_bin)))
    
    #Continous Cuckoo
    x_cuckoo_cont=x_cont+alpha*levy_flight(n,cont_d,Beta = Beta)*
      (x_cont-best_egg_cont)
    x_cuckoo_cont=limit_bound(x_cuckoo_cont,bnd_cont)
    
    #Discrete Cuckoo
    x_cuckoo_disc=x_disc+alpha*levy_flight(n,disc_d,Beta = Beta)*
      (x_disc-best_egg_disc)
    x_cuckoo_disc=limit_bound(x_cuckoo_disc,bnd_disc)
    
    
    #Binary Cuckoo
    x_cuckoo_bin=x_bin+alpha*levy_flight(n,bin_d,Beta = Beta)*
      (x_bin-best_egg_bin)
    x_cuckoo_bin=limit_bound(x_cuckoo_bin,bnd_bin)
    
    output=list("cont"=x_cuckoo_cont,
                "disc"=x_cuckoo_disc,"bin"=x_cuckoo_bin,"host_fit"=x0_fit)
    return(output)
  }
  
  
  #====================================New Generation==============================
  
  new_gen=function(cont_x_cuckoo,disc_x_cuckoo,bin_x_cuckoo,
                   cont_x_host,disc_x_host,bin_x_host,host_fit,seed){
    #Continuous Generation
    x_cuckoo=cont_x_cuckoo
    
    
    #Discrete Generation
    x_cuckoo_disc=disc_trans(disc_x_cuckoo,seed)
    
    #Binary Generation
    x_cuckoo_bin=binary_trans(bin_x_cuckoo,seed)
    
    #Evaluate fitness
    cuckoo_fit=fitness(x_cuckoo,x_cuckoo_disc,x_cuckoo_bin,fun)
    
    
    cont_x_new=cont_x_host
    bin_x_new=bin_x_host
    disc_x_new=disc_x_host
    
    replace_egg=cuckoo_fit>host_fit
    cont_x_new[replace_egg,]=cont_x_cuckoo[replace_egg,]
    disc_x_new[replace_egg,]=disc_x_cuckoo[replace_egg,]
    bin_x_new[replace_egg,]=bin_x_cuckoo[replace_egg,]
    
    x_new_all=list("cont"=cont_x_new,"disc"=disc_x_new,"bin"=bin_x_new)
    return(x_new_all)
    
  }
  
  #====================================Empty Nest==============================
  empty_nest=function(x_cont,x_disc,x_bin,pa,
                      bnd_cont,bnd_disc,bnd_bin){
    #x_new=cbind(x_cont,x_disc,x_bin)
    #epsilon=purrr::map_dfc(seq(ncol(x_new)),~runif(nrow(x_new)))
    #crit=epsilon>pa
    
    #H=fBasics::Heaviside(pa-epsilon)
    #Continous New
    epsilon_cont=purrr::map_dfc(seq(ncol(x_cont)),
                                ~runif(nrow(x_cont)))
    crit_cont=epsilon_cont>pa
    H_cont=fBasics::Heaviside(pa-epsilon_cont)
    
    x_newest_cont=x_cont+alpha*levy_flight(n,cont_d,Beta = Beta)*H_cont*
      (x_cont[sample.int(length(x_cont))]-
         x_cont[sample.int(length(x_cont))])*crit_cont
    
    x_newest_cont=limit_bound(x_newest_cont,bnd_cont)
    
    #Discrete new
    epsilon_disc=purrr::map_dfc(seq(ncol(x_disc)),
                                ~runif(nrow(x_disc)))
    crit_disc=epsilon_disc>pa
    H_disc=fBasics::Heaviside(pa-epsilon_disc)
    
    x_newest_disc=x_disc+alpha*levy_flight(n,disc_d,Beta = Beta)*H_disc*
      (x_disc[sample.int(length(x_disc))]-
         x_disc[sample.int(length(x_disc))])*crit_disc
    
    x_newest_disc=limit_bound(x_newest_disc,bnd_disc)
    
    #Binary new
    epsilon_bin=purrr::map_dfc(seq(ncol(x_bin)),
                               ~runif(nrow(x_bin)))
    crit_bin=epsilon_bin>pa
    H_bin=fBasics::Heaviside(pa-epsilon_bin)
    
    x_newest_bin=x_bin+alpha*levy_flight(n,bin_d,Beta = Beta)*H_bin*
      (x_bin[sample.int(length(x_bin))]-
         x_bin[sample.int(length(x_bin))])*crit_bin
    
    x_newest_bin=limit_bound(x_newest_bin,bnd_bin)
    
    
    
    
    #    x_newest=x_new+alpha*levy_flight(n,d,Beta = Beta)*H*
    #      (x_new[sample.int(length(x_new))]-x_new[sample.int(length(x_new))])*crit
    
    #    ncol_cont=seq(ncol(x_cont))
    #    ncol_disc=seq((ncol(x_cont)+1),(ncol(x_cont)+ncol(x_disc)))
    #    ncol_bin=setdiff(ncol(x_newest),c(ncol_cont,ncol_disc))
    
    #    x_newest_cont=limit_bound(x_newest[,ncol_cont],bnd_cont)
    #    x_newest_disc=limit_bound(x_newest[,ncol_disc],bnd_disc)
    #    x_newest_bin=limit_bound(x_newest[,ncol_bin],bnd_bin)
    
    x_newest=list("cont"=x_newest_cont,"disc"=x_newest_disc,"bin"=x_newest_bin)
    return(x_newest)
  }
  
  #===================================Make computational Time Information==============================
  
  pb = progress::progress_bar$new(
    format = " Elapsed time: :elapsedfull time completion: :eta",
    clear = FALSE, width= 180,total = iter_max)
  
  #====================================Get boundary of solution==============================
  
  cont_d= length(cont_bnd)
  disc_d= length(disc_bnd)
  bin_d = length(name_bin_bnd)
  bin_bnd=rep(list(c(-10,10)),bin_d)%>%
    magrittr::set_names(name_bin_bnd)
  
  #====================================Output Options==============================
  if(is.null(primary_out)){
    x0_host_disc=disc_trans(x_disc,seed)
    x0_host_bin=binary_trans(x_bin,seed)
    x00=cbind(x_host,x0_host_disc,x0_host_bin)
    fit_test=purrr::invoke(fun,x00%>%df2list%>%magrittr::extract2(1))
    if(length(fit_test)>1){
      stop("function output has more than one output. Argument primary_out must not be empty")
    }
  }else{
    fun1=fun
    if(!is.numeric(primary_out)){
      stop("primary_out must be numeric value")
    }
    fun=function(...){
      return(fun1(...)[[primary_out]])
    }
    fitness1=function(x_cont,x_disc,x_bin,fun){
      x=cbind(x_cont,x_disc,x_bin)
      df2list(x)%>%furrr::future_map(function(x)purrr::invoke(fun,x))  }
    
    extract_fitness=function(newest_fit,primary_out){
      purrr::map_dbl(newest_fit,function(x)x%>%magrittr::extract2(primary_out))
    }
    
    
  }
  
  #====================================Generate initial n host nest==============================
  cont_x_host=rmcs_files$x_host_cont
  disc_x_host=rmcs_files$x_host_disc
  bin_x_host=rmcs_files$x_host_bin
  
  
  #====================================Main Program==============================
  all_res=list() 
  cat("=================Starting iteration=============== \n")
  
  for(i in seq(rmcs_files$iteration+1,iter_max)){
    #Get cuckoo egg
    seed=rnorm(1)
    x_cuckoo=cuckoo_egg(cont_x_host,disc_x_host,bin_x_host,
                        cont_bnd,disc_bnd,bin_bnd,seed = seed)
    
    #Replace bad host egg with cuckoo egg
    x_new=new_gen(x_cuckoo$cont,x_cuckoo$disc,x_cuckoo$bin,
                  cont_x_host,disc_x_host,bin_x_host,
                  x_cuckoo$host_fit,seed = seed)
    
    
    #Discovery by host bird: abandon nest and build new one
    x_newest=empty_nest(x_new$cont,x_new$disc,x_new$bin,pa,
                        cont_bnd,disc_bnd,bin_bnd
    )
    #x_newest_all=cbind(x_newest$cont,x_newest$disc,x_newest$bin)
    #newest_fit=fitness(x_newest$cont,x_newest$disc,x_newest$bin,fun)
    
    
    #find current best
    
    if(!is.null(primary_out)){
      
      cat("evaluating current fitnees value... ")
      x_newest_disc=disc_trans(x_newest$disc,seed)
      x_newest_bin=binary_trans(x_newest$bin,seed)
      newest_fit1=fitness1(x_newest$cont,x_newest_disc,x_newest_bin,fun1)
      cat("done \n\n")
      newest_fit=extract_fitness(newest_fit1,primary_out)
      newest_fit_all=newest_fit1%>%
        magrittr::extract2(which.max(newest_fit))
      x_newest_fin=cbind(x_newest$cont,x_newest_disc,x_newest_bin)
      best=list("fitness"=newest_fit%>%
                  magrittr::extract(which.max(newest_fit)),
                "best_solution"=x_newest_fin%>%slice(which.max(newest_fit)))
      
      temp_res=list("x_host_cont"=x_newest$cont,
                    "x_host_disc"=x_newest$disc,
                    "x_host_bin"=x_newest$bin,
                    "best"=best,"iteration"=i)
      
      all_res[[i]]=list("temp_rest"=temp_res,"all_fit"=newest_fit_all)
    }else{
      cat("evaluating current fitnees value... ")
      x_newest_disc=disc_trans(x_newest$disc,seed)
      x_newest_bin=binary_trans(x_newest$bin,seed)
      newest_fit=fitness(x_newest$cont,x_newest_disc,x_newest_bin,fun)
      cat("done \n\n")
      x_newest_fin=cbind(x_newest$cont,x_newest_disc,x_newest_bin)
      best=list("fitness"=newest_fit%>%
                  magrittr::extract(which.max(newest_fit)),
                "best_solution"=x_newest_fin%>%slice(which.max(newest_fit)))
      
      temp_res=list("x_host_cont"=x_newest$cont,
                    "x_host_disc"=x_newest$disc,
                    "x_host_bin"=x_newest$bin,
                    "best"=best,"iteration"=i)      
      all_res[[i]]=temp_res
    }
    
    # keep best solution
    cont_x_host=x_newest$cont
    disc_x_host=x_newest$disc
    bin_x_host=x_newest$bin
    
    
    if(save){
      
      saveRDS(temp_res,save_files)
    }
    
    
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
  #====================================SOutput==============================   
  if(!is.null(primary_out)){
    fin_result=list("Best_Result"=best,"All_Result"=all_res)
  }else{
    fin_result=best
  }
  return(fin_result)
}