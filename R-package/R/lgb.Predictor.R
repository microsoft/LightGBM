#' @importFrom methods is
#' @importFrom R6 R6Class
Predictor <- R6::R6Class(

  classname = "lgb.Predictor",
  cloneable = FALSE,
  public = list(

    # Finalize will free up the handles
    finalize = function() {

      # Check the need for freeing handle
      if (private$need_free_handle && !lgb.is.null.handle(private$handle)) {

        # Freeing up handle
        lgb.call("LGBM_BoosterFree_R", ret = NULL, private$handle)
        private$handle <- NULL

      }

    },

    # Initialize will create a starter model
    initialize = function(modelfile, ...) {
      params <- list(...)
      private$params <- lgb.params2str(params)
      # Create new lgb handle
      handle <- 0.0

      # Check if handle is a character
      if (is.character(modelfile)) {

        # Create handle on it
        handle <- lgb.call("LGBM_BoosterCreateFromModelfile_R", ret = handle, lgb.c_str(modelfile))
        private$need_free_handle <- TRUE

      } else if (methods::is(modelfile, "lgb.Booster.handle")) {

        # Check if model file is a booster handle already
        handle <- modelfile
        private$need_free_handle <- FALSE

      } else {

        # Model file is unknown
        stop("lgb.Predictor: modelfile must be either a character filename or an lgb.Booster.handle")

      }

      # Override class and store it
      class(handle) <- "lgb.Booster.handle"
      private$handle <- handle

    },

    # Get current iteration
    current_iter = function() {

      cur_iter <- 0L
      lgb.call("LGBM_BoosterGetCurrentIteration_R",  ret = cur_iter, private$handle)

    },

    # Predict from data
    predict = function(data,
                       num_iteration = NULL,
                       rawscore = FALSE,
                       predleaf = FALSE,
                       predcontrib = FALSE,
                       header = FALSE,
                       reshape = FALSE) {

      # Check if number of iterations is existing - if not, then set it to -1 (use all)
      if (is.null(num_iteration)) {
        num_iteration <- -1
      }

      # Set temporary variable
      num_row <- 0L

      # Check if data is a file name and not a matrix
      if (identical(class(data), "character") && length(data) == 1) {

        # Data is a filename, create a temporary file with a "lightgbm_" pattern in it
        tmp_filename <- tempfile(pattern = "lightgbm_")
        on.exit(unlink(tmp_filename), add = TRUE)

        # Predict from temporary file
        lgb.call("LGBM_BoosterPredictForFile_R", ret = NULL, private$handle, data,
          as.integer(header),
          as.integer(rawscore),
          as.integer(predleaf),
          as.integer(predcontrib),
          as.integer(num_iteration),
          private$params,
          lgb.c_str(tmp_filename))

        # Get predictions from file
        preds <- read.delim(tmp_filename, header = FALSE, sep = "\t")
        num_row <- nrow(preds)
        preds <- as.vector(t(preds))

      } else {

        # Not a file, we need to predict from R object
        num_row <- nrow(data)

        npred <- 0L

        # Check number of predictions to do
        npred <- lgb.call("LGBM_BoosterCalcNumPredict_R",
                          ret = npred,
                          private$handle,
                          as.integer(num_row),
                          as.integer(rawscore),
                          as.integer(predleaf),
                          as.integer(predcontrib),
                          as.integer(num_iteration))

        # Pre-allocate empty vector
        preds <- numeric(npred)

        # Check if data is a matrix
        if (is.matrix(data)) {
          preds <- lgb.call("LGBM_BoosterPredictForMat_R",
                            ret = preds,
                            private$handle,
                            data,
                            as.integer(nrow(data)),
                            as.integer(ncol(data)),
                            as.integer(rawscore),
                            as.integer(predleaf),
                            as.integer(predcontrib),
                            as.integer(num_iteration),
                            private$params)

        } else if (methods::is(data, "dgCMatrix")) {
          if (length(data@p) > 2147483647) {
            stop("Cannot support large CSC matrix")
          }
          # Check if data is a dgCMatrix (sparse matrix, column compressed format)
          preds <- lgb.call("LGBM_BoosterPredictForCSC_R",
                            ret = preds,
                            private$handle,
                            data@p,
                            data@i,
                            data@x,
                            length(data@p),
                            length(data@x),
                            nrow(data),
                            as.integer(rawscore),
                            as.integer(predleaf),
                            as.integer(predcontrib),
                            as.integer(num_iteration),
                            private$params)

        } else {

          # Cannot predict on unknown class
          # to-do: predict from lgb.Dataset
          stop("predict: cannot predict on data of class ", sQuote(class(data)))

        }
      }

      # Check if number of rows is strange (not a multiple of the dataset rows)
      if (length(preds) %% num_row != 0) {
        stop("predict: prediction length ", sQuote(length(preds))," is not a multiple of nrows(data): ", sQuote(num_row))
      }

      # Get number of cases per row
      npred_per_case <- length(preds) / num_row


      # Data reshaping

      if (predleaf | predcontrib) {

        # Predict leaves only, reshaping is mandatory
        preds <- matrix(preds, ncol = npred_per_case, byrow = TRUE)

      } else if (reshape && npred_per_case > 1) {

        # Predict with data reshaping
        preds <- matrix(preds, ncol = npred_per_case, byrow = TRUE)

      }

      # Return predictions
      return(preds)

    }

  ),
  private = list(handle = NULL,
                 need_free_handle = FALSE,
                 params = "")
)
