Predictor <- R6Class(
  "lgb.Predictor",
  public = list(
    handle = NULL,
    finalize = function() {
      if(self$need_free_handle & !is.null(self$handle) & !lgb.is.null.handle(self$handle)){
        lgb.call("LGBM_BoosterFree_R", ret=NULL, self$handle)
      }
    }, 
    initialize = function(modelfile, need_free_handle=FALSE) {
      handle <- lgb.new.handle()
      if(typeof(modelfile) == "character") {
        handle <- lgb.call("LGBM_BoosterCreateFromModelfile_R", ret=handle, modelfile)
        private$need_free_handle = TRUE
      } else if (typeof(modelfile) == "lgb.Booster.handle") {
        handle <- modelfile
        private$need_free_handle = need_free_handle
      } else {
        stop("lgb.Predictor: modelfile must be either character filename, or lgb.Booster.handle")
      }
      class(handle) <- "lgb.Booster.handle"
      self$handle <- handle
    },

    predict = function(data, 
      num_iteration = NULL, rawscore = FALSE, predleaf = FALSE, header = FALSE, 
      reshape = FALSE) {

      if (is.null(num_iteration)) {
        num_iteration <- -1
        
      }

      num_row <- 0
      if (typeof(data) == "character") {
        tmp_filename <- tempfile(pattern = "lightgbm_")
        lgb.call("LGBM_BoosterPredictForFile_R", ret=NULL, self$handle, data, as.integer(header),
          as.integer(rawscore),
          as.integer(predleaf),
          as.integer(num_iteration),
          as.character(tmp_filename))
        preds <- read.delim(tmp_filename, header=FALSE, seq="\t")
        num_row <- nrow(preds)
        preds <- as.vector(t(preds))
        # delete temp file
        if(file.exists(tmp_filename)) { file.remove(tmp_filename) }
      } else {
        num_row <- nrow(data)
        npred <- 0
        npred <- lgb.call("LGBM_BoosterCalcNumPredict_R", ret=npred,
          self$handle,
          as.integer(num_row)
          as.integer(rawscore),
          as.integer(predleaf),
          as.integer(num_iteration))
		# allocte space for prediction
        preds <- rep(0.0, npred)
        if (is.matrix(data)) {
          preds <- lgb.call("LGBM_BoosterPredictForMat_R", ret=preds, 
            self$handle, 
            data,
            as.integer(nrow(data)),
            as.integer(ncol(data)),
            as.integer(rawscore),
            as.integer(predleaf),
            as.integer(num_iteration))
        } else if (class(data) == "dgCMatrix") {
          preds <- lgb.call("LGBM_BoosterPredictForCSC_R", ret=preds,
            self$handle, 
            data@p, 
            data@i, 
            data@x,
            length(data@p),
            length(data@x),
            nrow(data),
            as.integer(rawscore),
            as.integer(predleaf),
            as.integer(num_iteration))
        } else {
          stop(paste("predict: does not support to predict from ",
                   typeof(data)))
          }
      }

      if (length(preds) %% num_row != 0) {
        stop("predict: prediction length ", length(preds)," is not multiple of nrows(data) ", num_row)
      }
      npred_per_case <- length(preds) / num_row
      if (reshape && npred_per_case > 1) {
        preds <- matrix(preds, ncol = npred_per_case)
      }
      return(preds)
    }
  ), 
  private = list(
    need_free_handle = FALSE
  )
)
