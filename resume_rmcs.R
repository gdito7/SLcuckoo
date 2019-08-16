resume_rmcs=function(rmcs_files,fun,cont_bnd,disc_bnd,bin_d,name_d=NULL,
                     n=25,pa=0.25,alpha=1,
                     Beta=1.5,iter_max=250,
                    verbose=T,parallel=F,num_cores=NULL,save=F,
                    save_files="cs_res.rds"){
  
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
  fitness=function(x_cont,x_disc,x_bin,fun){
    x=cbind(x_cont,x_disc,x_bin)
    furrr::future_map_dbl(seq(nrow(x)),
                          function(i) purrr::invoke(fun,x%>%
                                                      slice(i)%>%
                                                      unlist(.,use.names = F)%>%
                                                      as.list))
  }
  
  #====================================Transformation==============================
  #Discrete
  extract_dec=function(x){
    x-floor(x)
  }
  
  disc_trans=function(x){
    
    u=purrr::map_dfc(seq(ncol(x)),~runif(nrow(x)))
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
  
  binary_trans=function(x){
    u=purrr::map_dfc(seq(ncol(x)),~runif(nrow(x)))
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
    lower=bnd$xmin
    upper=bnd$xmax
    purrr::map_dfc(seq(ncol(xx)),
                   function(i){
                     case_when(xx[[i]]<lower[i] ~ lower[i],
                               xx[[i]]>upper[i] ~ upper[i],
                               TRUE ~ xx[[i]]
                     )
                     
                   }
    )
    
  }
  
  #====================================Get Host egg==============================
  
  egg=function(n,bnd){
    purrr::pmap_dfc(bnd,function(xmin,xmax){
      runif(n,min = xmin,max = xmax)
    })
  }
  
  #====================================Get Cuckoo egg==============================
  
  cuckoo_egg=function(x_cont,x_disc,x_bin,bnd_cont,bnd_disc,bnd_bin){
    
    x0_disc=disc_trans(x_disc)
    x0_disc=bin_trans(x_bin)
    x0_fit=fitness(x_cont,x0_disc,x0_bin,fun)
    
    #Continous Cuckoo
    best_egg_cont=x_cont[which.max(x0_fit),]%>%slice(rep(1:n(),each=nrow(x_cont)))
    #Discrete Cuckoo
    best_egg_disc=x_disc[which.max(x0_fit),]%>%slice(rep(1:n(),each=nrow(x_disc)))
    #Binary Cuckoo
    best_egg_bin=x_bin[which.max(x0_fit),]%>%slice(rep(1:n(),each=nrow(x_bin)))
    
    #Continous Cuckoo
    x_cuckoo_cont=x_cont+alpha*levy_flight(n,d,Beta = Beta)*(x_cont-best_egg_cont)
    x_cuckoo_cont=limit_bound(x_cuckoo_cont,bnd_cont)
    
    #Discrete Cuckoo
    x_cuckoo_disc=x_disc+alpha*levy_flight(n,d,Beta = Beta)*(x_disc-best_egg_disc)
    x_cuckoo_disc=limit_bound(x_cuckoo_disc,bnd_disc)
    
    
    #Discrete Cuckoo
    x_cuckoo_bin=x_bin+alpha*levy_flight(n,d,Beta = Beta)*(x_bin-best_egg_bin)
    x_cuckoo_bin=limit_bound(x_cuckoo_bin,bnd_bin)
    
    output=list("cont"=x_cuckoo_cont,
                "disc"=x_cuckoo_disc,"bin"=x_cuckoo_bin,"host_fit"=x0_fit)
    return(output)
  }
  
  
  #====================================New Generation==============================
  
  new_gen=function(cont_x_cuckoo,disc_x_cuckoo,bin_x_cuckoo,
                   cont_x_host,disc_x_host,bin_x_host,host_fit){
    #Continuous Generation
    x_cuckoo=cont_x_cuckoo
    
    
    #Discrete Generation
    x_cuckoo_disc=disc_trans(disc_x_cuckoo)
    
    #Binary Generation
    x_cuckoo_bin=binary_trans(bin_x_cuckoo)
    
    #Evaluate fitness
    cuckoo_fit=fitness(x_cuckoo,x_cuckoo_disc,x_cuckoo_bin,fun)
    
    
    cont_x_new=cont_x_host
    bin_x_new=bin_x_host
    disc_x_new=disc_x_host
    
    replace_egg=cuckoo_fit>host_fit
    cont_x_new[replace_egg,]=cont_x_cuckoo[replace_egg,]
    disc_x_new[replace_egg,]=disc_x_cuckoo[replace_egg,]
    bin_x_new[replace_egg,]=bin_x_cuckoo[replace_egg,]
    
    x_new_all=list(cont_x_new,disc_x_new,bin_x_new)
    return(x_new_all)
    
  }
  
  #====================================Empty Nest==============================
  empty_nest=function(x_cont,x_disc,x_bin,pa,bnd_cont,bnd_disc,bnd_bin){
    x_new=cbind(x_cont,x_disc,x_bin)
    epsilon=purrr::map_dfc(seq(ncol(x_new)),~runif(nrow(x_new)))
    crit=epsilon>pa
    
    H=fBasics::Heaviside(pa-epsilon)
    x_newest=x_new+alpha*levy_flight(n,d,Beta = Beta)*H*
      (x_new[sample.int(length(x_new))]-x_new[sample.int(length(x_new))])*crit
    
    ncol_cont=seq(ncol(x_cont))
    ncol_disc=seq((ncol(x_cont)+1),(ncol(x_cont)+ncol(x_disc)))
    ncol_bin=setdiff(ncol(x_newest),c(ncol_cont,ncol_disc))
    
    x_newest_cont=limit_bound(x_newest[,ncol_cont],bnd_cont)
    x_newest_disc=limit_bound(x_newest[,ncol_disc],bnd_disc)
    x_newest_bin=limit_bound(x_newest[,ncol_bin],bnd_bin)
    
    x_newest=list("cont"=x_newest_cont,"disc"=x_newest_disc,"bin"=x_newes_bin)
    return(x_newest)
  }
  
  #===================================Make computational Time Information==============================
  
  pb = progress::progress_bar$new(
    format = " Elapsed time: :elapsedfull time completion: :eta",
    clear = FALSE, width= 180,total = iter_max)
  
  #====================================Get boundary of solution==============================
  
  cont_d=length(cont_bnd[[1]])
  disc_d=length(disc_bnd[[1]])
  bin_bnd=list(xmin=rep(-10,bin_d),xmax=rep(10,bin_d))
  
  #====================================Output Options==============================
  if(is.null(primary_out)){
    x0_host_disc=disc_trans(x_disc)
    x0_host_bin=bin_trans(x_bin)
    x00=cbind(x_host,x0_host_disc,x0_host_bin)
    fit_test=purrr::invoke(fun,x00%>%
                             slice(1)%>%
                             unlist(.,use.names = F)%>%
                             as.list)
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
      furrr::future_map_dbl(seq(nrow(x)),
                            function(i) purrr::invoke(fun,x%>%
                                                        slice(i)%>%
                                                        unlist(.,use.names = F)%>%
                                                        as.list))
    }
    
    
  }
  
  
  #====================================Generate initial n host nest==============================
  cont_x_host=rmcs_files$cont_host
  disc_x_host=rmcs_files$disc_host
  bin_x_host=rmcs_files$bin_host
  
  
  #====================================Main Program==============================
  
  for(i in seq(iter_max)){
    #Get cuckoo egg
    x_cuckoo=cuckoo_egg(cont_x_host,disc_x_host,bin_x_host,
                        cont_bnd,disc_bnd,bin_bnd)
    
    #Replace bad host egg with cuckoo egg
    x_new=new_gen(x_cuckoo$cont,x_cuckoo$disc,x_cuckoo$bin,
                  cont_x_host,disc_x_host,bin_x_host,x_cuckoo$host_fit)
    
    
    #Discovery by host bird: abandon nest and build new one
    x_newest=empty_nest(x_new$cont,x_new$disc,x_new$bin,pa,
                        cont_bnd,disc_bnd,bin_bnd)
    x_newest_all=cbind(x_newest$cont,x_newest$disc,x_newest$bin)
    newest_fit=fitness(x_newest$cont,x_newest$disc,x_newest$bin,fun)
    
    
    # keep best solution
    cont_x_host=x_newest$cont
    disc_x_host=x_newest$disc
    bin_x_host=x_newest$bin
    
    #find current best
    best=list("fitness"=newest_fit[which.max(newest_fit)],
              "best_solution"=x_newest_all[which.max(newest_fit),])
    
    temp_res=list("cont_host"=cont_x_host,
                  "disc_host"=disc_x_host,
                  "bin_host"=bin_x_host,
                  "best"=best,"iteration"=i)
    
    if(!is.null(primary_out)){
      newest_fit_all=fitness1(x_newest,fun1)[[which.max(newest_fit)]]
      all_res[[i]]=list("temp_rest"=temp_res,"all_fit"=newest_fit_all)
    }else{
      all_res[[i]]=temp_res
    }
    
    
    if(save){
      
      saveRDS(temp_res,save_files)
    }
    
    
    if(verbose){
      cat("iteration: ",i,"fitness: ",best$fitness,"\n")
      pb$tick()
      cat("\n")
      
    }
    
    
  }
  #====================================Stop Paralel==============================
  
  if(parallel){
    future::plan(future::sequential())
  }
  #====================================Give Solution Name==============================
  
  if( !is.null( names(cont_bnd[[1]]) )&
      is.null( names(disc_bnd[[1]]) )&
      is.null(name_d))
  {
    nm=c(names(cont_bnd[[1]]),names(disc_bnd[[1]]),name_d)
    colnames(best$best_solution)=nm
  }
  
  if(!is.null(primary_out)){
    fin_result=list("Best_Result"=best,"All_Result"=all_res)
  }else{
    fin_result=best
  }
  return(fin_result)
}