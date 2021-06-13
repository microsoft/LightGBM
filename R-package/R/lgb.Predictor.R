#' @importFrom methods is
#' @importFrom R6 R6Class
#' @importFrom utils read.delim
Predictor <- R6::R6Class(

  classname = "lgb.Predictor",
  cloneable = FALSE,
  public = list(

    # Finalize will free up the handles
    finalize = function() {

      # Check the need for freeing handle
      if (private$need_free_handle) {

        .Call(
          LGBM_BoosterFree_R
          , private$handle
        )
        private$handle <- NULL

      }

      return(invisible(NULL))

    },

    # Initialize will create a starter model
    initialize = function(modelfile, params = list()) {
      private$params <- lgb.params2str(params = params)
      handle <- NULL

      # Check if handle is a character
      if (is.character(modelfile)) {

        # Create handle on it
        handle <- .Call(
          LGBM_BoosterCreateFromModelfile_R
          , modelfile
        )
        private$need_free_handle <- TRUE

      } else if (methods::is(modelfile, "lgb.Booster.handle")) {

        # Check if model file is a booster handle already
        handle <- modelfile
        private$need_free_handle <- FALSE

      } else {

        stop("lgb.Predictor: modelfile must be either a character filename or an lgb.Booster.handle")

      }

      # Override class and store it
      class(handle) <- "lgb.Booster.handle"
      private$handle <- handle

      return(invisible(NULL))

    },

    # Get current iteration
    current_iter = function() {

      cur_iter <- 0L
      .Call(
        LGBM_BoosterGetCurrentIteration_R
        , private$handle
        , cur_iter
      )
      return(cur_iter)

    },

    # Predict from data
    predict = function(data,
                       start_iteration = NULL,
                       num_iteration = NULL,
                       rawscore = FALSE,
                       predleaf = FALSE,
                       predcontrib = FALSE,
                       header = FALSE,
                       reshape = FALSE) {

      # Check if number of iterations is existing - if not, then set it to -1 (use all)
      if (is.null(num_iteration)) {
        num_iteration <- -1L
      }
      # Check if start iterations is existing - if not, then set it to 0 (start from the first iteration)
      if (is.null(start_iteration)) {
        start_iteration <- 0L
      }

      num_row <- 0L

      # Check if data is a file name and not a matrix
      if (identical(class(data), "character") && length(data) == 1L) {

        # Data is a filename, create a temporary file with a "lightgbm_" pattern in it
        tmp_filename <- tempfile(pattern = "lightgbm_")
        on.exit(unlink(tmp_filename), add = TRUE)

        # Predict from temporary file
        .Call(
          LGBM_BoosterPredictForFile_R
          , private$handle
          , data
          , as.integer(header)
          , as.integer(rawscore)
          , as.integer(predleaf)
          , as.integer(predcontrib)
          , as.integer(start_iteration)
          , as.integer(num_iteration)
          , private$params
          , tmp_filename
        )

        # Get predictions from file
        preds <- utils::read.delim(tmp_filename, header = FALSE, sep = "\t")
        num_row <- nrow(preds)
        preds <- as.vector(t(preds))

      } else {

        # Not a file, we need to predict from R object
        num_row <- nrow(data)

        npred <- 0L

        # Check number of predictions to do
        .Call(
          LGBM_BoosterCalcNumPredict_R
          , private$handle
          , as.integer(num_row)
          , as.integer(rawscore)
          , as.integer(predleaf)
          , as.integer(predcontrib)
          , as.integer(start_iteration)
          , as.integer(num_iteration)
          , npred
        )

        # Pre-allocate empty vector
        preds <- numeric(npred)

        # Check if data is a matrix
        if (is.matrix(data)) {
          # this if() prevents the memory and computational costs
          # of converting something that is already "double" to "double"
          if (storage.mode(data) != "double") {
            storage.mode(data) <- "double"
          }
          .Call(
            LGBM_BoosterPredictForMat_R
            , private$handle
            , data
            , as.integer(nrow(data))
            , as.integer(ncol(data))
            , as.integer(rawscore)
            , as.integer(predleaf)
            , as.integer(predcontrib)
            , as.integer(start_iteration)
            , as.integer(num_iteration)
            , private$params
            , preds
          )

        } else if (methods::is(data, "dgCMatrix")) {
          if (length(data@p) > 2147483647L) {
            stop("Cannot support large CSC matrix")
          }
          # Check if data is a dgCMatrix (sparse matrix, column compressed format)
          .Call(
            LGBM_BoosterPredictForCSC_R
            , private$handle
            , data@p
            , data@i
            , data@x
            , length(data@p)
            , length(data@x)
            , nrow(data)
            , as.integer(rawscore)
            , as.integer(predleaf)
            , as.integer(predcontrib)
            , as.integer(start_iteration)
            , as.integer(num_iteration)
            , private$params
            , preds
          )

        } else {

          stop("predict: cannot predict on data of class ", sQuote(class(data)))

        }
      }

      # Check if number of rows is strange (not a multiple of the dataset rows)
      if (length(preds) %% num_row != 0L) {
        stop(
          "predict: prediction length "
          , sQuote(length(preds))
          , " is not a multiple of nrows(data): "
          , sQuote(num_row)
        )
      }

      # Get number of cases per row
      npred_per_case <- length(preds) / num_row


      # Data reshaping

      if (predleaf | predcontrib) {

        # Predict leaves only, reshaping is mandatory
        preds <- matrix(preds, ncol = npred_per_case, byrow = TRUE)

      } else if (reshape && npred_per_case > 1L) {

        # Predict with data reshaping
        preds <- matrix(preds, ncol = npred_per_case, byrow = TRUE)

      }

      return(preds)

    }

  ),
  private = list(
    handle = NULL
    , need_free_handle = FALSE
    , params = ""
  )
)
