CB_ENV <- R6Class(
  "lgb.cb_env",
  cloneable=FALSE,
  public = list(
  	model=NULL,
  	cvfolds=NULL,
  	iteration=NULL,
  	begin_iteration=NULL,
  	end_iteration=NULL,
  	eval_result_list=list()
  )
)

# Format the evaluation metric string
format.eval.string <- function(eval_res, eval_err=NULL) {
  if (is.null(eval_res))
    stop('no evaluation results')
  if (length(eval_res) == 0)
    stop('no evaluation results')
  if (!is.null(eval_err)) {
    res <- sprintf('%s\'s %s:%g+%g', eval_res$data_name, eval_res$name, eval_res$value, eval_err)
  } else {
    res <- sprintf('%s\'s %s:%g', eval_res$data_name, eval_res$name, eval_res$value)
  }
  return(res)
}


cb.print.evaluation <- function(period=1){
	callback <- function(env){
    if(period > 0){
      i <- env$iteration
      if( (i - 1) %% period == 0
         | i == env$begin_iteration
         | i == env$end_iteration ){
        msg <- list(sprintf('[%d]:',i))
        for( j in 1:length(env$eval_result_list)) {
          msg <- c(msg, format.eval.string(env$eval_result_list[[j]]))
        }
        print(paste0(msg, collapse='\t'))
      }
    }
	}
  attr(callback, 'call') <- match.call()
  attr(callback, 'name') <- 'cb.print.evaluation'
  return(callback)
}
