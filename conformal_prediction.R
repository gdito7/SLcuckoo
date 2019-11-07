conformal_pred=function(prob_val,prob_test,lab_val,lab_test,truth,conf_inv=0.95){
  if(conf_inv>1||conf_inv<=0){
    stop("Confidence Interval should be between 0 and 1")
  }
  #browser()
  alpha = 1 - conf_inv
  if(is.null(levels(lab_val))){
    lab_val=as.factor(lab_val)
  }
  unique_lab = levels(lab_val)
  check_lab = rep(unique_lab, each = 2)
  check_lab = split(check_lab, check_lab)
  check_lab = purrr::map_chr(check_lab,
                             function(i) {
                               stringr::str_flatten(i)
                             })
  
  
  crit_label_sc = purrr::map(seq_along(lab_test),
                             function(i) {
                               purrr::map(seq_along(unique_lab),
                                          function(j) {
                                            stringr::str_c(unique_lab[j],
                                                           as.character(lab_val))
                                          })
                             })
  
  
  ncf_score = 1 - prob_val
  ncf_score_test = 1 - prob_test
  
  p_value_func=function(i) {
    crit_ncf = purrr::map(seq_along(unique_lab), function(j) {
      ncf_score[crit_label_sc[[i]][[j]] == check_lab[j]] >=
        ncf_score_test[i]
    })
    p_value = purrr::map_dfc(crit_ncf, function(i) {
      mean(i)
    })
    names(p_value) = stringr::str_c("p_value_", unique_lab)
    return(p_value)
  }
  p_value = purrr::map(seq(lab_test),p_value_func)
  
  
  p_value = purrr::invoke(rbind, p_value)
  
  rst_fin = p_value %>% mutate_all(list(crit = ~ . > alpha))
  
  class_belong = apply(rst_fin %>% select(contains("crit")), 1,
                       function(i) {
                         if (all(!i)) {
                           "null"
                         } else{
                           which(i == TRUE)
                         }
                       })
  pred_set = purrr::map(class_belong,
                        function(i) {
                          if (all(i == "null")) {
                            "null"
                          } else{
                            unique_lab[i]
                          }
                        })
  pred_set_fin = purrr::map_chr(seq_along(pred_set), function(i) {
    stringr::str_flatten(pred_set[[i]])
  })
  pred_set_fin = factor(pred_set_fin)
  rst_fin = rst_fin %>% select(-contains("crit"))
  
  max_pval = apply(rst_fin, 1, max)
  min_pval = apply(rst_fin, 1, min)
  lab_max = unique_lab[apply(rst_fin, 1, which.max)]
  lab_min = unique_lab[apply(rst_fin, 1, which.min)]
  
  diagnostic = tibble(
    best_conf = floor((1 - min_pval) * 100),
    best_class = lab_max,
    credibility = floor((max_pval) * 100)
  )
  rst_fin = rst_fin %>% mutate(pred_set = pred_set_fin, truth=truth)
  return(list("result"=rst_fin,"diagnostic"=diagnostic))
}
