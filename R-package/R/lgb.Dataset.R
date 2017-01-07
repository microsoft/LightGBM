Dataset <- R6Class(
  "lgb.Dataset",
  cloneable=FALSE,
  public = list(
    finalize = function() {
      if (!lgb.is.null.handle(private$handle)) {
        print("free dataset handle")
        lgb.call("LGBM_DatasetFree_R", ret = NULL, private$handle)
        private$handle <- NULL
      }
    },
    initialize = function(data,
                          params = list(),
                          reference = NULL,
                          colnames = NULL,
                          categorical_feature = NULL,
                          predictor = NULL,
                          free_raw_data = TRUE,
                          used_indices = NULL,
                          info = list(),
                          ...) {
      addiction_params <- list(...)
      for (key in names(addiction_params)) {
        if (key %in% c('label', 'weight', 'init_score', 'group')) {
          info[[key]] <- addiction_params[[key]]
        } else {
          params[[key]] <- addiction_params[[key]]
        }
      }
      if (!is.null(reference)) {
        if (!lgb.check.r6.class(reference, "lgb.Dataset")) {
          stop("lgb.Dataset: Only can use lgb.Dataset as reference")
        }
      }
      if (!is.null(predictor)) {
        if (!lgb.check.r6.class(predictor, "lgb.Predictor")) {
          stop("lgb.Dataset: Only can use lgb.Predictor as predictor")
        }
      }
      private$raw_data <- data
      private$params <- params
      private$reference <- reference
      private$colnames <- colnames
      
      private$categorical_feature <- categorical_feature
      private$predictor <- predictor
      private$free_raw_data <- free_raw_data
      private$used_indices <- used_indices
      private$info <- info
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
      return(ret)
    },
    construct = function() {
      if (!lgb.is.null.handle(private$handle)) {
        return(self)
      }
      # Get feature names
      cnames <- NULL
      if (is.matrix(private$raw_data) |
          class(private$raw_data) == "dgCMatrix") {
        cnames <- colnames(private$raw_data)
      }
      # set feature names if not exist
      if (is.null(private$colnames) & !is.null(cnames)) {
        private$colnames <- as.character(cnames)
      }
      # Get categorical feature index
      if (!is.null(private$categorical_feature)) {
        fname_dict <- list()
        if (!is.null(private$colnames)) {
          fname_dict <-
            as.list(setNames(0:(length(
              private$colnames
            ) - 1), private$colnames))
        }
        cate_indices <- list()
        for (key in private$categorical_feature) {
          if (is.character(key)) {
            idx <- fname_dict[[key]]
            if (is.null(idx)) {
              stop(paste("lgb.self.get.handle: cannot find feature name ", key))
            }
            cate_indices <- c(cate_indices, idx)
          } else {
            # one-based indices to zero-based
            idx <- as.integer(key - 1)
            cate_indices <- c(cate_indices, idx)
          }
        }
        private$params$categorical_feature <- cate_indices
      }
      # Check has header or not
      has_header <- FALSE
      if (!is.null(private$params$has_header) |
          !is.null(private$params$header)) {
        if (tolower(as.character(private$params$has_header)) == "true"
            |
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
        if (typeof(private$raw_data) == "character") {
          handle <-
            lgb.call(
              "LGBM_DatasetCreateFromFile_R",
              ret = handle,
              lgb.c_str(private$raw_data),
              params_str,
              ref_handle
            )
        } else if (is.matrix(private$raw_data)) {
          handle <-
            lgb.call(
              "LGBM_DatasetCreateFromMat_R",
              ret = handle,
              private$raw_data,
              nrow(private$raw_data),
              ncol(private$raw_data),
              params_str,
              ref_handle
            )
        } else if (class(private$raw_data) == "dgCMatrix") {
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
          stop(paste(
            "lgb.Dataset.construct: does not support to construct from ",
            typeof(private$raw_data)
          ))
        }
      } else {
        # construct subset
        if (is.null(private$reference)) {
          stop("lgb.Dataset.construct: reference cannot be NULL if construct subset")
        }
        handle <-
          lgb.call(
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
      if (!is.null(private$colnames)) {
        self$set_colnames(private$colnames)
      }
      
      # load init score
      if (!is.null(private$predictor) &
          is.null(private$used_indices)) {
        init_score <-
          private$predictor$predict(private$raw_data,
                                    rawscore = TRUE,
                                    reshape = TRUE)
        # not need to transpose, for is col_marjor
        init_score <- as.vector(init_score)
        private$info$init_score <- init_score
      }
      if (private$free_raw_data & !is.character(private$raw_data)) {
        private$raw_data <- NULL
      }
      if (length(private$info) > 0) {
        # set infos
        for (i in 1:length(private$info)) {
          p <- private$info[i]
          self$setinfo(names(p), p[[1]])
        }
      }
      if (is.null(self$getinfo("label"))) {
        stop("lgb.Dataset.construct: label should be set")
      }
      return(self)
    },
    dim = function() {
      if (!lgb.is.null.handle(private$handle)) {
        num_row <- as.integer(0)
        num_col <- as.integer(0)
        
        return(c(
          lgb.call("LGBM_DatasetGetNumData_R", ret = num_row, private$handle),
          lgb.call("LGBM_DatasetGetNumFeature_R", ret = num_col, private$handle)
        ))
      } else if (is.matrix(private$raw_data) |
                 class(private$raw_data) == "dgCMatrix") {
        return(dim(private$raw_data))
      } else {
        stop(
          "dim: cannot get Dimensions before dataset constructed, please call lgb.Dataset.construct explicit"
        )
      }
    },
    get_colnames = function() {
      if (!lgb.is.null.handle(private$handle)) {
        cnames <- lgb.call.return.str("LGBM_DatasetGetFeatureNames_R",
                                      private$handle)
        private$colnames <- as.character(strsplit(cnames, "\t")[[1]])
        return(private$colnames)
      } else if (is.matrix(private$raw_data) |
                 class(private$raw_data) == "dgCMatrix") {
        return(colnames(private$raw_data))
      } else {
        stop(
          "colnames: cannot get colnames before dataset constructed, please call lgb.Dataset.construct explicit"
        )
      }
    },
    set_colnames = function(colnames) {
      if(is.null(colnames)) return(self)
      colnames <- as.character(colnames)
      if(length(colnames) == 0) return(self)
      private$colnames <- colnames
      if (!lgb.is.null.handle(private$handle)) {
        merged_name <- paste0(as.list(private$colnames), collapse = "\t")
        lgb.call("LGBM_DatasetSetFeatureNames_R",
                 ret = NULL,
                 private$handle,
                 lgb.c_str(merged_name))
      }
      return(self)
    },
    getinfo = function(name) {
      if (typeof(name) != "character" ||
          length(name) != 1 ||
          !name %in% c('label', 'weight', 'init_score', 'group')) {
        stop(
          "getinfo: name must one of the following\n",
          "    'label', 'weight', 'init_score', 'group'"
        )
      }
      if (is.null(private$info[[name]]) &
          !lgb.is.null.handle(private$handle)) {
        info_len <- as.integer(0)
        info_len <-
          lgb.call("LGBM_DatasetGetFieldSize_R",
                   ret = info_len,
                   private$handle,
                   lgb.c_str(name))
        if (info_len > 0) {
          ret <- NULL
          if (name == "group") {
            ret <- integer(info_len)
          } else {
            ret <- rep(0.0, info_len)
          }
          ret <-
            lgb.call("LGBM_DatasetGetField_R",
                     ret = ret,
                     private$handle,
                     lgb.c_str(name))
          private$info[[name]] <- ret
        }
      }
      return(private$info[[name]])
    },
    setinfo = function(name, info) {
      if (typeof(name) != "character" ||
          length(name) != 1 ||
          !name %in% c('label', 'weight', 'init_score', 'group')) {
        stop(
          "setinfo: name must one of the following\n",
          "    'label', 'weight', 'init_score', 'group'"
        )
      }
      if (name == "group") {
        info <- as.integer(info)
      } else {
        info <- as.numeric(info)
      }
      private$info[[name]] <- info
      if (!lgb.is.null.handle(private$handle) & !is.null(info)) {
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
      return(self)
    },
    slice = function(idxset, ...) {
      ret <- Dataset$new(
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
      return(ret)
    },
    update_params = function(params){
      private$params <- modifyList(private$params, params)
    },
    set_categorical_feature = function(categorical_feature) {
      if (identical(private$categorical_feature, categorical_feature)) {
        return(self)
      }
      if (is.null(private$raw_data)) {
        stop(
          "set_categorical_feature: cannot set categorical feature after free raw data,
          please set free_raw_data=FALSE when construct lgb.Dataset"
        )
      }
      private$categorical_feature <- categorical_feature
      self$finalize()
      return(self)
    },
    set_reference = function(reference) {
      self$set_categorical_feature(reference$.__enclos_env__$private$categorical_feature)
      self$set_colnames(reference$get_colnames())
      private$set_predictor(reference$.__enclos_env__$private$predictor)
      if (identical(private$reference, reference)) {
        return(self)
      }
      if (is.null(private$raw_data)) {
        stop(
          "set_reference: cannot set reference after free raw data,
          please set free_raw_data=FALSE when construct lgb.Dataset"
        )
      }
      if (!is.null(reference)) {
        if (!lgb.check.r6.class(reference, "lgb.Dataset")) {
          stop("set_reference: Only can use lgb.Dataset as reference")
        }
      }
      private$reference <- reference
      self$finalize()
      return(self)
    },
    save_binary = function(fname) {
      self$construct()
      lgb.call("LGBM_DatasetSaveBinary_R",
               ret = NULL,
               private$handle,
               lgb.c_str(fname))
      return(self)
    }
  ),
  private = list(
    handle = NULL,
    raw_data = NULL,
    params = list(),
    reference = NULL,
    colnames = NULL,
    categorical_feature = NULL,
    predictor = NULL,
    free_raw_data = TRUE,
    used_indices = NULL,
    info = NULL,
    get_handle = function() {
      if (lgb.is.null.handle(private$handle)) {
        self$construct()
      }
      return(private$handle)
    },
    set_predictor = function(predictor) {
      if (identical(private$predictor, predictor)) {
        return(self)
      }
      if (is.null(private$raw_data)) {
        stop(
          "set_predictor: cannot set predictor after free raw data,
          please set free_raw_data=FALSE when construct lgb.Dataset"
        )
      }
      if (!is.null(predictor)) {
        if (!lgb.check.r6.class(predictor, "lgb.Predictor")) {
          stop("set_predictor: Only can use lgb.Predictor as predictor")
        }
      }
      private$predictor <- predictor
      self$finalize()
      return(self)
    }
  )
)

#' Contruct lgb.Dataset object
#'
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
#' data(agaricus.train, package='lightgbm')
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label=train$label)
#' lgb.Dataset.save(dtrain, 'lgb.Dataset.data')
#' dtrain <- lgb.Dataset('lgb.Dataset.data')
#' lgb.Dataset.construct(dtrain)
#' @export
lgb.Dataset <- function(data,
                        params = list(),
                        reference = NULL,
                        colnames = NULL,
                        categorical_feature = NULL,
                        free_raw_data = TRUE,
                        info = list(),
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

# internal helper method
lgb.is.Dataset <- function(x){
  if(lgb.check.r6.class(x, "lgb.Dataset")){
    return(TRUE)
  } else{
    return(FALSE)
  }
}

#' Contruct a validation data
#'
#' Contruct a validation data according to training data
#'
#' @param dataset \code{lgb.Dataset} object, training data
#' @param data a \code{matrix} object, a \code{dgCMatrix} object or a character representing a filename
#' @param info a list of information of the lgb.Dataset object
#' @param ... other information to pass to \code{info}.
#' @return constructed dataset
#' @examples
#' data(agaricus.train, package='lightgbm')
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label=train$label)
#' data(agaricus.test, package='lightgbm')
#' test <- agaricus.test
#' dtest <- lgb.Dataset.create.valid(dtrain, test$data, label=test$label)
#' @export
lgb.Dataset.create.valid <-
  function(dataset, data, info = list(),  ...) {
    if(!lgb.is.Dataset(dataset)) {
      stop("lgb.Dataset.create.valid: input data should be lgb.Dataset object")
    }
    return(dataset$create_valid(data, info, ...))
  }

#' Construct Dataset explicit
#'
#' Construct Dataset explicit
#'
#' @param dataset Object of class \code{lgb.Dataset}
#' @examples
#' data(agaricus.train, package='lightgbm')
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label=train$label)
#' lgb.Dataset.construct(dtrain)
#' @export
lgb.Dataset.construct <- function(dataset) {
  if(!lgb.is.Dataset(dataset)) {
    stop("lgb.Dataset.construct: input data should be lgb.Dataset object")
  }
  return(dataset$construct())
}

#' Dimensions of lgb.Dataset
#'
#' Dimensions of lgb.Dataset
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
#' data(agaricus.train, package='lightgbm')
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label=train$label)
#'
#' stopifnot(nrow(dtrain) == nrow(train$data))
#' stopifnot(ncol(dtrain) == ncol(train$data))
#' stopifnot(all(dim(dtrain) == dim(train$data)))
#'
#' @rdname dim
#' @export
dim.lgb.Dataset <- function(x, ...) {
  if(!lgb.is.Dataset(x)) {
    stop("dim.lgb.Dataset: input data should be lgb.Dataset object")
  }
  return(x$dim())
}

#' Handling of column names of \code{lgb.Dataset}
#'
#' Handling of column names of \code{lgb.Dataset}
#'
#' Only column names are supported for \code{lgb.Dataset}, thus setting of
#' row names would have no effect and returnten row names would be NULL.
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
#' data(agaricus.train, package='lightgbm')
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label=train$label)
#' lgb.Dataset.construct(dtrain)
#' dimnames(dtrain)
#' colnames(dtrain)
#' colnames(dtrain) <- make.names(1:ncol(train$data))
#' print(dtrain, verbose=TRUE)
#'
#' @rdname dimnames.lgb.Dataset
#' @export
dimnames.lgb.Dataset <- function(x) {
  if(!lgb.is.Dataset(x)) {
    stop("dimnames.lgb.Dataset: input data should be lgb.Dataset object")
  }
  return(list(NULL, x$get_colnames()))
}

#' @rdname dimnames.lgb.Dataset
#' @export
`dimnames<-.lgb.Dataset` <- function(x, value) {
  if (!is.list(value) || length(value) != 2L)
    stop("invalid 'dimnames' given: must be a list of two elements")
  if (!is.null(value[[1L]]))
    stop("lgb.Dataset does not have rownames")
  if (is.null(value[[2]])) {
    x$set_colnames(NULL)
    return(x)
  }
  if (ncol(x) != length(value[[2]]))
    stop("can't assign ",
         length(value[[2]]),
         " colnames to a ",
         ncol(x),
         " column lgb.Dataset")
  x$set_colnames(value[[2]])
  return(x)
}

#' Slice an dataset
#'
#' Get a new Dataset containing the specified rows of
#' orginal lgb.Dataset object
#'
#' @param dataset Object of class "lgb.Dataset"
#' @param idxset a integer vector of indices of rows needed
#' @param ... other parameters (currently not used)
#' @return constructed sub dataset
#'
#' @examples
#' data(agaricus.train, package='lightgbm')
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label=train$label)
#'
#' dsub <- slice(dtrain, 1:42)
#' labels1 <- getinfo(dsub, 'label')
#'
#' @export
slice <- function(dataset, ...)
  UseMethod("slice")

#' @rdname slice
#' @export
slice.lgb.Dataset <- function(dataset, idxset, ...) {
  if(!lgb.is.Dataset(dataset)) {
    stop("slice.lgb.Dataset: input data should be lgb.Dataset object")
  }
  return(dataset$slice(idxset, ...))
}


#' Get information of an lgb.Dataset object
#'
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
#' data(agaricus.train, package='lightgbm')
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label=train$label)
#' lgb.Dataset.construct(dtrain)
#' labels <- getinfo(dtrain, 'label')
#' setinfo(dtrain, 'label', 1-labels)
#'
#' labels2 <- getinfo(dtrain, 'label')
#' stopifnot(all(labels2 == 1-labels))
#' @export
getinfo <- function(dataset, ...)
  UseMethod("getinfo")

#' @rdname getinfo
#' @export
getinfo.lgb.Dataset <- function(dataset, name, ...) {
  if(!lgb.is.Dataset(dataset)) {
    stop("getinfo.lgb.Dataset: input data should be lgb.Dataset object")
  }
  return(dataset$getinfo(name))
}

#' Set information of an lgb.Dataset object
#'
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
#' data(agaricus.train, package='lightgbm')
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label=train$label)
#' lgb.Dataset.construct(dtrain)
#' labels <- getinfo(dtrain, 'label')
#' setinfo(dtrain, 'label', 1-labels)
#' labels2 <- getinfo(dtrain, 'label')
#' stopifnot(all.equal(labels2, 1-labels))
#' @export
setinfo <- function(dataset, ...)
  UseMethod("setinfo")

#' @rdname setinfo
#' @export
setinfo.lgb.Dataset <- function(dataset, name, info, ...) {
  if(!lgb.is.Dataset(dataset)) {
    stop("setinfo.lgb.Dataset: input data should be lgb.Dataset object")
  }
  return(dataset$setinfo(name, info))
}

#' set categorical feature of \code{lgb.Dataset}
#'
#' set categorical feature of \code{lgb.Dataset}
#'
#' @param dataset object of class \code{lgb.Dataset}
#' @param categorical_feature categorical features
#' @return passed dataset
#' @examples
#' data(agaricus.train, package='lightgbm')
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label=train$label)
#' lgb.Dataset.save(dtrain, 'lgb.Dataset.data')
#' dtrain <- lgb.Dataset('lgb.Dataset.data')
#' lgb.Dataset.set.categorical(dtrain, 1:2)
#' @rdname lgb.Dataset.set.categorical
#' @export
lgb.Dataset.set.categorical <-
  function(dataset, categorical_feature) {
    if(!lgb.is.Dataset(dataset)) {
      stop("lgb.Dataset.set.categorical: input data should be lgb.Dataset object")
    }
    return(dataset$set_categorical_feature(categorical_feature))
  }

#' set reference of \code{lgb.Dataset}
#'
#' set reference of \code{lgb.Dataset}. 
#' If you want to use validation data, you should set its reference to training data
#'
#' @param dataset object of class \code{lgb.Dataset}
#' @param reference object of class \code{lgb.Dataset}
#' @return passed dataset
#' @examples
#' data(agaricus.train, package='lightgbm')
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label=train$label)
#' data(agaricus.test, package='lightgbm')
#' test <- agaricus.test
#' dtest <- lgb.Dataset(test$data, test=train$label)
#' lgb.Dataset.set.reference(dtest, dtrain)
#' @rdname lgb.Dataset.set.reference
#' @export
lgb.Dataset.set.reference <- function(dataset, reference) {
  if(!lgb.is.Dataset(dataset)) {
    stop("lgb.Dataset.set.reference: input data should be lgb.Dataset object")
  }
  return(dataset$set_reference(reference))
}

#' save \code{lgb.Dataset} to binary file
#' 
#' save \code{lgb.Dataset} to binary file
#' 
#' @param dataset object of class \code{lgb.Dataset}
#' @param fname object filename of output file
#' @return passed dataset
#' @examples
#' data(agaricus.train, package='lightgbm')
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label=train$label)
#' lgb.Dataset.save(dtrain, "data.bin")
#' @rdname lgb.Dataset.save
#' @export
lgb.Dataset.save <- function(dataset, fname) {
  if(!lgb.is.Dataset(dataset)) {
    stop("lgb.Dataset.set: input data should be lgb.Dataset object")
  }
  if(!is.character(fname)) {
    stop("lgb.Dataset.set: filename should be character type")
  }
  return(dataset$save_binary(fname))
}
