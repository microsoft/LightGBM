Dataset <- R6Class(
  "lgb.Dataset",
  cloneable = FALSE,
  public = list(
    finalize = function() {
      if (!lgb.is.null.handle(private$handle)) {
        lgb.call("LGBM_DatasetFree_R", ret = NULL, private$handle)
        private$handle <- NULL
      }
    },
    initialize = function(data,
                          params              = list(),
                          reference           = NULL,
                          colnames            = NULL,
                          categorical_feature = NULL,
                          predictor           = NULL,
                          free_raw_data       = TRUE,
                          used_indices        = NULL,
                          info                = list(),
                          ...) {
      additional_params <- list(...)
      INFO_KEYS <- c('label', 'weight', 'init_score', 'group')
      for (key in names(additional_params)) {
        if (key %in% INFO_KEYS) {
          info[[key]] <- additional_params[[key]]
        } else {
          params[[key]] <- additional_params[[key]]
        }
      }
      if (!is.null(reference)) {
        if (!lgb.check.r6.class(reference, "lgb.Dataset")) {
          stop("lgb.Dataset: Can only use ", sQuote("lgb.Dataset"), " as reference")
        }
      }
      if (!is.null(predictor)) {
        if (!lgb.check.r6.class(predictor, "lgb.Predictor")) {
          stop("lgb.Dataset: Only can use ", sQuote("lgb.Predictor"), " as predictor")
        }
      }
      private$raw_data  <- data
      private$params    <- params
      private$reference <- reference
      private$colnames  <- colnames

      private$categorical_feature <- categorical_feature
      private$predictor           <- predictor
      private$free_raw_data       <- free_raw_data
      private$used_indices        <- used_indices
      private$info                <- info
    },
    create_valid = function(data, info = list(),  ...) {
      ret <- Dataset$new(
        data,
        private$params,
        self,
        private$colnames,
        private$categorical_feature,
        private$predictor,
        private$free_raw_data,
        NULL,
        info,
        ...
      )
      ret
    },
    construct = function() {
      if (!lgb.is.null.handle(private$handle)) {
        return(self)
      }
      # Get feature names
      cnames <- NULL
      if (is.matrix(private$raw_data) || is(private$raw_data, "dgCMatrix")) {
        cnames <- colnames(private$raw_data)
      }
      # set feature names if not exist
      if (is.null(private$colnames) && !is.null(cnames)) {
        private$colnames <- as.character(cnames)
      }
      # Get categorical feature index
      if (!is.null(private$categorical_feature)) {
        if (typeof(private$categorical_feature) == "character") {
            cate_indices <- as.list(match(private$categorical_feature, private$colnames) - 1)
            if (sum(is.na(cate_indices)) > 0) {
              stop("lgb.self.get.handle: supplied an unknown feature in categorical_feature: ", sQuote(private$categorical_feature[is.na(cate_indices)]))
            }
          } else {
            if (max(private$categorical_feature) > length(private$colnames)) {
              stop("lgb.self.get.handle: supplied a too large value in categorical_feature: ", max(private$categorical_feature), " but only ", length(private$colnames), " features")
            }
            cate_indices <- as.list(private$categorical_feature - 1)
          }
        private$params$categorical_feature <- cate_indices
      }
      # Check has header or not
      has_header <- FALSE
      if (!is.null(private$params$has_header) ||
          !is.null(private$params$header)) {
        if (tolower(as.character(private$params$has_header)) == "true"
            ||
            tolower(as.character(private$params$header)) == "true") {
          has_header <- TRUE
        }
      }
      # Generate parameter str
      params_str <- lgb.params2str(private$params)
      # get handle of reference dataset
      ref_handle <- NULL
      if (!is.null(private$reference)) {
        ref_handle <- private$reference$.__enclos_env__$private$get_handle()
      }
      handle <- lgb.new.handle()
      # not subset
      if (is.null(private$used_indices)) {
        if (is.character(private$raw_data)) {
          handle <- lgb.call(
              "LGBM_DatasetCreateFromFile_R",
              ret = handle,
              lgb.c_str(private$raw_data),
              params_str,
              ref_handle
            )
        } else if (is.matrix(private$raw_data)) {
          handle <- lgb.call(
              "LGBM_DatasetCreateFromMat_R",
              ret = handle,
              private$raw_data,
              nrow(private$raw_data),
              ncol(private$raw_data),
              params_str,
              ref_handle
            )
        } else if (is(private$raw_data, "dgCMatrix")) {
          handle <- lgb.call(
            "LGBM_DatasetCreateFromCSC_R",
            ret = handle,
            private$raw_data@p,
            private$raw_data@i,
            private$raw_data@x,
            length(private$raw_data@p),
            length(private$raw_data@x),
            nrow(private$raw_data),
            params_str,
            ref_handle
          )
        } else {
          stop(
            "lgb.Dataset.construct: does not support constructing from ", sQuote(class(private$raw_data))
          )
        }
      } else {
        # construct subset
        if (is.null(private$reference)) {
          stop("lgb.Dataset.construct: reference cannot be NULL for constructing data subset")
        }
        handle <- lgb.call(
            "LGBM_DatasetGetSubset_R",
            ret = handle,
            ref_handle,
            private$used_indices,
            length(private$used_indices),
            params_str
          )
      }
      class(handle) <- "lgb.Dataset.handle"
      private$handle <- handle
      # set feature names
      if (!is.null(private$colnames)) { self$set_colnames(private$colnames) }

      # load init score
      if (!is.null(private$predictor) &&
          is.null(private$used_indices)) {
        init_score <- private$predictor$predict(private$raw_data, rawscore = TRUE, reshape = TRUE)
        # do not need to transpose, for is col_marjor
        init_score <- as.vector(init_score)
        private$info$init_score <- init_score
      }
      if (isTRUE(private$free_raw_data)) { private$raw_data <- NULL }
      if (length(private$info) > 0) {
        # set infos
        for (i in seq_along(private$info)) {
          p <- private$info[i]
          self$setinfo(names(p), p[[1]])
        }
      }
      if (is.null(self$getinfo("label"))) {
        stop("lgb.Dataset.construct: label should be set")
      }
      self
    },
    dim = function() {
      if (!lgb.is.null.handle(private$handle)) {
        num_row <- 0L
        num_col <- 0L

        c(
          lgb.call("LGBM_DatasetGetNumData_R",    ret = num_row, private$handle),
          lgb.call("LGBM_DatasetGetNumFeature_R", ret = num_col, private$handle)
        )
      } else if (is.matrix(private$raw_data) || is(private$raw_data, "dgCMatrix")) {
        dim(private$raw_data)
      } else {
        stop(
          "dim: cannot get dimensions before dataset has been constructed, please call lgb.Dataset.construct explicitly"
        )
      }
    },
    get_colnames = function() {
      if (!lgb.is.null.handle(private$handle)) {
        cnames <- lgb.call.return.str("LGBM_DatasetGetFeatureNames_R", private$handle)
        private$colnames <- as.character(base::strsplit(cnames, "\t")[[1]])
        private$colnames
      } else if (is.matrix(private$raw_data) || is(private$raw_data, "dgCMatrix")) {
        colnames(private$raw_data)
      } else {
        stop(
          "dim: cannot get dimensions before dataset has been constructed, please call lgb.Dataset.construct explicitly"
        )
      }
    },
    set_colnames = function(colnames) {
      if (is.null(colnames)) { return(self) }
      colnames <- as.character(colnames)
      if (length(colnames) == 0) { return(self) }
      private$colnames <- colnames
      if (!lgb.is.null.handle(private$handle)) {
        merged_name <- paste0(as.list(private$colnames), collapse = "\t")
        lgb.call("LGBM_DatasetSetFeatureNames_R",
                 ret = NULL,
                 private$handle,
                 lgb.c_str(merged_name))
      }
      self
    },
    getinfo = function(name) {
      INFONAMES <- c("label", "weight", "init_score", "group")
      if (!is.character(name) ||
          length(name) != 1   ||
          !name %in% INFONAMES) {
        stop(
          "getinfo: name must one of the following: ", paste0(sQuote(INFONAMES), collapse = ", ")
        )
      }
      if (is.null(private$info[[name]]) && !lgb.is.null.handle(private$handle)) {
        info_len <- 0L
        info_len <- lgb.call("LGBM_DatasetGetFieldSize_R",
                             ret = info_len,
                             private$handle,
                             lgb.c_str(name))
        if (info_len > 0) {
          ret <- NULL
          ret <- if (name == "group") { integer(info_len) } else { rep(0.0, info_len) }
          ret <- lgb.call("LGBM_DatasetGetField_R",
                          ret = ret,
                          private$handle,
                          lgb.c_str(name))
          private$info[[name]] <- ret
        }
      }
      private$info[[name]]
    },
    setinfo = function(name, info) {
      INFONAMES <- c("label", "weight", "init_score", "group")
      if (!is.character(name) ||
          length(name) != 1   ||
          !name %in% INFONAMES) {
        stop(
          "setinfo: name must one of the following: ", paste0(sQuote(INFONAMES), collapse = ", ")
        )
      }
      info <- if (name == "group") { as.integer(info) } else { as.numeric(info) }
      private$info[[name]] <- info
      if (!lgb.is.null.handle(private$handle) && !is.null(info)) {
        if (length(info) > 0) {
          lgb.call(
            "LGBM_DatasetSetField_R",
            ret = NULL,
            private$handle,
            lgb.c_str(name),
            info,
            length(info)
          )
        }
      }
      self
    },
    slice = function(idxset, ...) {
      Dataset$new(
        NULL,
        private$params,
        self,
        private$colnames,
        private$categorical_feature,
        private$predictor,
        private$free_raw_data,
        idxset,
        NULL,
        ...
      )
    },
    update_params = function(params) {
      private$params <- modifyList(private$params, params)
      self
    },
    set_categorical_feature = function(categorical_feature) {
      if (identical(private$categorical_feature, categorical_feature)) { return(self) }
      if (is.null(private$raw_data)) {
        stop(
          "set_categorical_feature: cannot set categorical feature after freeing raw data,
          please set ", sQuote("free_raw_data = FALSE"), " when you construct lgb.Dataset"
        )
      }
      private$categorical_feature <- categorical_feature
      self$finalize()
      self
    },
    set_reference = function(reference) {
      self$set_categorical_feature(reference$.__enclos_env__$private$categorical_feature)
      self$set_colnames(reference$get_colnames())
      private$set_predictor(reference$.__enclos_env__$private$predictor)
      if (identical(private$reference, reference)) { return(self) }
      if (is.null(private$raw_data)) {
        stop(
          "set_reference: cannot set reference after freeing raw data,
          please set ", sQuote("free_raw_data = FALSE"), " when you construct lgb.Dataset"
        )
      }
      if (!is.null(reference)) {
        if (!lgb.check.r6.class(reference, "lgb.Dataset")) {
          stop("set_reference: Can only use lgb.Dataset as a reference")
        }
      }
      private$reference <- reference
      self$finalize()
      self
    },
    save_binary = function(fname) {
      self$construct()
      lgb.call("LGBM_DatasetSaveBinary_R",
               ret = NULL,
               private$handle,
               lgb.c_str(fname))
      self
    }
  ),
  private = list(
    handle              = NULL,
    raw_data            = NULL,
    params              = list(),
    reference           = NULL,
    colnames            = NULL,
    categorical_feature = NULL,
    predictor           = NULL,
    free_raw_data       = TRUE,
    used_indices        = NULL,
    info                = NULL,
    get_handle          = function() {
      if (lgb.is.null.handle(private$handle)) { self$construct() }
      private$handle
    },
    set_predictor = function(predictor) {
      if (identical(private$predictor, predictor)) { return(self) }
      if (is.null(private$raw_data)) {
        stop(
          "set_predictor: cannot set predictor after free raw data,
          please set ", sQuote("free_raw_data = FALSE"), " when you construct lgb.Dataset"
        )
      }
      if (!is.null(predictor)) {
        if (!lgb.check.r6.class(predictor, "lgb.Predictor")) {
          stop("set_predictor: Can only use lgb.Predictor as predictor")
        }
      }
      private$predictor <- predictor
      self$finalize()
      self
    }
  )
)

#' Contruct lgb.Dataset object
#'
#' Contruct lgb.Dataset object from dense matrix, sparse matrix
#' or local file (that was created previously by saving an \code{lgb.Dataset}).
#'
#' @param data a \code{matrix} object, a \code{dgCMatrix} object or a character representing a filename
#' @param params a list of parameters
#' @param reference reference dataset
#' @param colnames names of columns
#' @param categorical_feature categorical features
#' @param free_raw_data TRUE for need to free raw data after construct
#' @param info a list of information of the lgb.Dataset object
#' @param ... other information to pass to \code{info} or parameters pass to \code{params}
#' @return constructed dataset
#' @examples
#' \dontrun{
#'   data(agaricus.train, package='lightgbm')
#'   train <- agaricus.train
#'   dtrain <- lgb.Dataset(train$data, label=train$label)
#'   lgb.Dataset.save(dtrain, 'lgb.Dataset.data')
#'   dtrain <- lgb.Dataset('lgb.Dataset.data')
#'   lgb.Dataset.construct(dtrain)
#' }
#' @export
lgb.Dataset <- function(data,
                        params              = list(),
                        reference           = NULL,
                        colnames            = NULL,
                        categorical_feature = NULL,
                        free_raw_data       = TRUE,
                        info                = list(),
                        ...) {
  Dataset$new(
    data,
    params,
    reference,
    colnames,
    categorical_feature,
    NULL,
    free_raw_data,
    NULL,
    info,
    ...
  )
}


#' Contruct validation data
#'
#' Contruct validation data according to training data
#'
#' @param dataset \code{lgb.Dataset} object, training data
#' @param data a \code{matrix} object, a \code{dgCMatrix} object or a character representing a filename
#' @param info a list of information of the lgb.Dataset object
#' @param ... other information to pass to \code{info}.
#' @return constructed dataset
#' @examples
#' \dontrun{
#'   data(agaricus.train, package='lightgbm')
#'   train <- agaricus.train
#'   dtrain <- lgb.Dataset(train$data, label=train$label)
#'   data(agaricus.test, package='lightgbm')
#'   test <- agaricus.test
#'   dtest <- lgb.Dataset.create.valid(dtrain, test$data, label=test$label)
#' }
#' @export
lgb.Dataset.create.valid <- function(dataset, data, info = list(),  ...) {
  if (!lgb.is.Dataset(dataset)) {
    stop("lgb.Dataset.create.valid: input data should be an lgb.Dataset object")
  }
  dataset$create_valid(data, info, ...)
}

#' Construct Dataset explicitly
#'
#' @param dataset Object of class \code{lgb.Dataset}
#' @examples
#' \dontrun{
#'   data(agaricus.train, package='lightgbm')
#'   train <- agaricus.train
#'   dtrain <- lgb.Dataset(train$data, label=train$label)
#'   lgb.Dataset.construct(dtrain)
#' }
#' @export
lgb.Dataset.construct <- function(dataset) {
  if (!lgb.is.Dataset(dataset)) {
    stop("lgb.Dataset.construct: input data should be an lgb.Dataset object")
  }
  dataset$construct()
}

#' Dimensions of an lgb.Dataset
#'
#' Returns a vector of numbers of rows and of columns in an \code{lgb.Dataset}.
#' @param x Object of class \code{lgb.Dataset}
#' @param ... other parameters
#' @return a vector of numbers of rows and of columns
#'
#' @details
#' Note: since \code{nrow} and \code{ncol} internally use \code{dim}, they can also
#' be directly used with an \code{lgb.Dataset} object.
#'
#' @examples
#' dontrun{
#'   data(agaricus.train, package='lightgbm')
#'   train <- agaricus.train
#'   dtrain <- lgb.Dataset(train$data, label=train$label)
#'
#'   stopifnot(nrow(dtrain) == nrow(train$data))
#'   stopifnot(ncol(dtrain) == ncol(train$data))
#'   stopifnot(all(dim(dtrain) == dim(train$data)))
#' }
#' @rdname dim
#' @export
dim.lgb.Dataset <- function(x, ...) {
  if (!lgb.is.Dataset(x)) {
    stop("dim.lgb.Dataset: input data should be an lgb.Dataset object")
  }
  x$dim()
}

#' Handling of column names of \code{lgb.Dataset}
#'
#' Only column names are supported for \code{lgb.Dataset}, thus setting of
#' row names would have no effect and returned row names would be NULL.
#'
#' @param x object of class \code{lgb.Dataset}
#' @param value a list of two elements: the first one is ignored
#'        and the second one is column names
#'
#' @details
#' Generic \code{dimnames} methods are used by \code{colnames}.
#' Since row names are irrelevant, it is recommended to use \code{colnames} directly.
#'
#' @examples
#' dontrun{
#'   data(agaricus.train, package='lightgbm')
#'   train <- agaricus.train
#'   dtrain <- lgb.Dataset(train$data, label=train$label)
#'   lgb.Dataset.construct(dtrain)
#'   dimnames(dtrain)
#'   colnames(dtrain)
#'   colnames(dtrain) <- make.names(1:ncol(train$data))
#'   print(dtrain, verbose=TRUE)
#' }
#' @rdname dimnames.lgb.Dataset
#' @export
dimnames.lgb.Dataset <- function(x) {
  if (!lgb.is.Dataset(x)) {
    stop("dimnames.lgb.Dataset: input data should be an lgb.Dataset object")
  }
  list(NULL, x$get_colnames())
}

#' @rdname dimnames.lgb.Dataset
#' @export
`dimnames<-.lgb.Dataset` <- function(x, value) {
  if (!is.list(value) || length(value) != 2L)
    stop("invalid ", sQuote("value"), " given: must be a list of two elements")
  if (!is.null(value[[1L]])) { stop("lgb.Dataset does not have rownames") }
  if (is.null(value[[2]])) {
    x$set_colnames(NULL)
    return(x)
  }
  if (ncol(x) != length(value[[2]]))
    stop("can't assign ",
         sQuote(length(value[[2]])),
         " colnames to an lgb.Dataset with ",
         sQuote(ncol(x)), " columns")
  x$set_colnames(value[[2]])
  x
}

#' Slice a dataset
#'
#' Get a new \code{lgb.Dataset} containing the specified rows of
#' orginal lgb.Dataset object
#'
#' @param dataset Object of class "lgb.Dataset"
#' @param idxset a integer vector of indices of rows needed
#' @param ... other parameters (currently not used)
#' @return constructed sub dataset
#'
#' @examples
#' \dontrun{
#'   data(agaricus.train, package='lightgbm')
#'   train <- agaricus.train
#'   dtrain <- lgb.Dataset(train$data, label=train$label)
#'
#'   dsub <- slice(dtrain, 1:42)
#'   labels1 <- getinfo(dsub, 'label')
#' }
#' @export
slice <- function(dataset, ...) { UseMethod("slice") }

#' @rdname slice
#' @export
slice.lgb.Dataset <- function(dataset, idxset, ...) {
  if (!lgb.is.Dataset(dataset)) {
    stop("slice.lgb.Dataset: input dataset should be an lgb.Dataset object")
  }
  dataset$slice(idxset, ...)
}


#' Get information of an lgb.Dataset object
#'
#' @param dataset Object of class \code{lgb.Dataset}
#' @param name the name of the information field to get (see details)
#' @param ... other parameters
#' @return info data
#'
#' @details
#' The \code{name} field can be one of the following:
#'
#' \itemize{
#'     \item \code{label}: label lightgbm learn from ;
#'     \item \code{weight}: to do a weight rescale ;
#'     \item \code{group}: group size
#'     \item \code{init_score}: initial score is the base prediction lightgbm will boost from ;
#' }
#'
#' @examples
#' \dontrun{
#'   data(agaricus.train, package='lightgbm')
#'   train <- agaricus.train
#'   dtrain <- lgb.Dataset(train$data, label=train$label)
#'   lgb.Dataset.construct(dtrain)
#'   labels <- getinfo(dtrain, 'label')
#'   setinfo(dtrain, 'label', 1-labels)
#'
#'   labels2 <- getinfo(dtrain, 'label')
#'   stopifnot(all(labels2 == 1-labels))
#' }
#' @export
getinfo <- function(dataset, ...) { UseMethod("getinfo") }

#' @rdname getinfo
#' @export
getinfo.lgb.Dataset <- function(dataset, name, ...) {
  if (!lgb.is.Dataset(dataset)) {
    stop("getinfo.lgb.Dataset: input dataset should be an lgb.Dataset object")
  }
  dataset$getinfo(name)
}

#' Set information of an lgb.Dataset object
#'
#' @param dataset Object of class "lgb.Dataset"
#' @param name the name of the field to get
#' @param info the specific field of information to set
#' @param ... other parameters
#' @return passed object
#'
#' @details
#' The \code{name} field can be one of the following:
#'
#' \itemize{
#'     \item \code{label}: label lightgbm learn from ;
#'     \item \code{weight}: to do a weight rescale ;
#'     \item \code{init_score}: initial score is the base prediction lightgbm will boost from ;
#'     \item \code{group}.
#' }
#'
#' @examples
#' \dontrun{
#'   data(agaricus.train, package='lightgbm')
#'   train <- agaricus.train
#'   dtrain <- lgb.Dataset(train$data, label=train$label)
#'   lgb.Dataset.construct(dtrain)
#'   labels <- getinfo(dtrain, 'label')
#'   setinfo(dtrain, 'label', 1-labels)
#'   labels2 <- getinfo(dtrain, 'label')
#'   stopifnot(all.equal(labels2, 1-labels))
#' }
#' @export
setinfo <- function(dataset, ...) { UseMethod("setinfo") }

#' @rdname setinfo
#' @export
setinfo.lgb.Dataset <- function(dataset, name, info, ...) {
  if (!lgb.is.Dataset(dataset)) {
    stop("setinfo.lgb.Dataset: input dataset should be an lgb.Dataset object")
  }
  dataset$setinfo(name, info)
}

#' Set categorical feature of \code{lgb.Dataset}
#'
#' @param dataset object of class \code{lgb.Dataset}
#' @param categorical_feature categorical features
#' @return passed dataset
#' @examples
#' \dontrun{
#'   data(agaricus.train, package='lightgbm')
#'   train <- agaricus.train
#'   dtrain <- lgb.Dataset(train$data, label=train$label)
#'   lgb.Dataset.save(dtrain, 'lgb.Dataset.data')
#'   dtrain <- lgb.Dataset('lgb.Dataset.data')
#'   lgb.Dataset.set.categorical(dtrain, 1:2)
#' }
#' @rdname lgb.Dataset.set.categorical
#' @export
lgb.Dataset.set.categorical <- function(dataset, categorical_feature) {
  if (!lgb.is.Dataset(dataset)) {
    stop("lgb.Dataset.set.categorical: input dataset should be an lgb.Dataset object")
  }
  dataset$set_categorical_feature(categorical_feature)
}

#' Set reference of \code{lgb.Dataset}
#'
#' If you want to use validation data, you should set reference to training data
#'
#' @param dataset object of class \code{lgb.Dataset}
#' @param reference object of class \code{lgb.Dataset}
#' @return passed dataset
#' @examples
#' \dontrun{
#'   data(agaricus.train, package='lightgbm')
#'   train <- agaricus.train
#'   dtrain <- lgb.Dataset(train$data, label=train$label)
#'   data(agaricus.test, package='lightgbm')
#'   test <- agaricus.test
#'   dtest <- lgb.Dataset(test$data, test=train$label)
#'   lgb.Dataset.set.reference(dtest, dtrain)
#' }
#' @rdname lgb.Dataset.set.reference
#' @export
lgb.Dataset.set.reference <- function(dataset, reference) {
  if (!lgb.is.Dataset(dataset)) {
    stop("lgb.Dataset.set.reference: input dataset should be an lgb.Dataset object")
  }
  dataset$set_reference(reference)
}

#' Save \code{lgb.Dataset} to a binary file
#'
#' @param dataset object of class \code{lgb.Dataset}
#' @param fname object filename of output file
#' @return passed dataset
#' @examples
#' \dontrun{
#'   data(agaricus.train, package='lightgbm')
#'   train <- agaricus.train
#'   dtrain <- lgb.Dataset(train$data, label=train$label)
#'   lgb.Dataset.save(dtrain, "data.bin")
#' }
#' @rdname lgb.Dataset.save
#' @export
lgb.Dataset.save <- function(dataset, fname) {
  if (!lgb.is.Dataset(dataset)) {
    stop("lgb.Dataset.set: input dataset should be an lgb.Dataset object")
  }
  if (!is.character(fname)) {
    stop("lgb.Dataset.set: fname should be a character or a file connection")
  }
  dataset$save_binary(fname)
}
