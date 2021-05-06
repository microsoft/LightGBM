/*!
 * Copyright (c) 2017 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_R_H_
#define LIGHTGBM_R_H_

#include <LightGBM/c_api.h>

#define R_NO_REMAP
#define R_USE_C99_IN_CXX
#include <Rinternals.h>

#include "R_object_helper.h"

/*!
* \brief get string message of the last error
* \return err_msg string with error information
*/
LIGHTGBM_C_EXPORT SEXP LGBM_GetLastError_R();

// --- start Dataset interface

/*!
* \brief load Dataset from file like the command_line LightGBM does
* \param filename the name of the file
* \param parameters additional parameters
* \param reference used to align bin mapper with other Dataset, nullptr means not used
* \param out created Dataset
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetCreateFromFile_R(
  SEXP filename,
  SEXP parameters,
  LGBM_SE reference,
  LGBM_SE out
);

/*!
* \brief create a Dataset from Compressed Sparse Column (CSC) format
* \param indptr pointer to row headers
* \param indices findex
* \param data fvalue
* \param num_indptr number of cols in the matrix + 1
* \param nelem number of nonzero elements in the matrix
* \param num_row number of rows
* \param parameters additional parameters
* \param reference used to align bin mapper with other Dataset, nullptr means not used
* \param out created Dataset
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetCreateFromCSC_R(
  SEXP indptr,
  SEXP indices,
  SEXP data,
  SEXP num_indptr,
  SEXP nelem,
  SEXP num_row,
  SEXP parameters,
  LGBM_SE reference,
  LGBM_SE out
);

/*!
* \brief create Dataset from dense matrix
* \param data matrix data
* \param num_row number of rows
* \param num_col number columns
* \param parameters additional parameters
* \param reference used to align bin mapper with other Dataset, nullptr means not used
* \param out created Dataset
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetCreateFromMat_R(
  SEXP data,
  SEXP num_row,
  SEXP num_col,
  SEXP parameters,
  LGBM_SE reference,
  LGBM_SE out
);

/*!
* \brief Create subset of a Dataset
* \param handle handle of full Dataset
* \param used_row_indices Indices used in subset
* \param len_used_row_indices length of Indices used in subset
* \param parameters additional parameters
* \param out created Dataset
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetGetSubset_R(
  LGBM_SE handle,
  SEXP used_row_indices,
  SEXP len_used_row_indices,
  SEXP parameters,
  LGBM_SE out
);

/*!
* \brief save feature names to Dataset
* \param handle handle
* \param feature_names feature names
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetSetFeatureNames_R(
  LGBM_SE handle,
  SEXP feature_names
);

/*!
* \brief save feature names to Dataset
* \param handle handle
* \param feature_names feature names
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetGetFeatureNames_R(
  LGBM_SE handle,
  SEXP buf_len,
  SEXP actual_len,
  LGBM_SE feature_names
);

/*!
* \brief save Dataset to binary file
* \param handle an instance of Dataset
* \param filename file name
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetSaveBinary_R(
  LGBM_SE handle,
  SEXP filename
);

/*!
* \brief free Dataset
* \param handle an instance of Dataset
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetFree_R(
  LGBM_SE handle
);

/*!
* \brief set vector to a content in info
*        Note: group and group_id only work for C_API_DTYPE_INT32
*              label and weight only work for C_API_DTYPE_FLOAT32
* \param handle an instance of Dataset
* \param field_name field name, can be label, weight, group, group_id
* \param field_data pointer to vector
* \param num_element number of element in field_data
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetSetField_R(
  LGBM_SE handle,
  SEXP field_name,
  SEXP field_data,
  SEXP num_element
);

/*!
* \brief get size of info vector from Dataset
* \param handle an instance of Dataset
* \param field_name field name
* \param out size of info vector from Dataset
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetGetFieldSize_R(
  LGBM_SE handle,
  SEXP field_name,
  SEXP out
);

/*!
* \brief get info vector from Dataset
* \param handle an instance of Dataset
* \param field_name field name
* \param field_data pointer to vector
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetGetField_R(
  LGBM_SE handle,
  SEXP field_name,
  SEXP field_data
);

/*!
 * \brief Raise errors for attempts to update Dataset parameters.
 *        Some parameters cannot be updated after construction.
 * \param old_params Current Dataset parameters
 * \param new_params New Dataset parameters
 * \return R NULL value
 */
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetUpdateParamChecking_R(
  SEXP old_params,
  SEXP new_params
);

/*!
* \brief get number of data.
* \param handle the handle to the Dataset
* \param out The address to hold number of data
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetGetNumData_R(
  LGBM_SE handle,
  SEXP out
);

/*!
* \brief get number of features
* \param handle the handle to the Dataset
* \param out The output of number of features
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetGetNumFeature_R(
  LGBM_SE handle,
  SEXP out
);

// --- start Booster interfaces

/*!
* \brief create a new boosting learner
* \param train_data training Dataset
* \param parameters format: 'key1=value1 key2=value2'
* \param out handle of created Booster
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterCreate_R(
  LGBM_SE train_data,
  SEXP parameters,
  LGBM_SE out
);

/*!
* \brief free Booster
* \param handle handle to be freed
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterFree_R(
  LGBM_SE handle
);

/*!
* \brief load an existing Booster from model file
* \param filename filename of model
* \param out handle of created Booster
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterCreateFromModelfile_R(
  SEXP filename,
  LGBM_SE out
);

/*!
* \brief load an existing Booster from a string
* \param model_str string containing the model
* \param out handle of created Booster
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterLoadModelFromString_R(
  SEXP model_str,
  LGBM_SE out
);

/*!
* \brief Merge model in two Boosters to first handle
* \param handle handle primary Booster handle, will merge other handle to this
* \param other_handle secondary Booster handle
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterMerge_R(
  LGBM_SE handle,
  LGBM_SE other_handle
);

/*!
* \brief Add new validation to Booster
* \param handle Booster handle
* \param valid_data validation Dataset
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterAddValidData_R(
  LGBM_SE handle,
  LGBM_SE valid_data
);

/*!
* \brief Reset training data for Booster
* \param handle Booster handle
* \param train_data training Dataset
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterResetTrainingData_R(
  LGBM_SE handle,
  LGBM_SE train_data
);

/*!
* \brief Reset config for current Booster
* \param handle Booster handle
* \param parameters format: 'key1=value1 key2=value2'
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterResetParameter_R(
  LGBM_SE handle,
  SEXP parameters
);

/*!
* \brief Get number of classes
* \param handle Booster handle
* \param out number of classes
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterGetNumClasses_R(
  LGBM_SE handle,
  SEXP out
);

/*!
* \brief update the model in one round
* \param handle Booster handle
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterUpdateOneIter_R(
  LGBM_SE handle
);

/*!
* \brief update the model, by directly specifying gradient and second order gradient,
*       this can be used to support customized loss function
* \param handle Booster handle
* \param grad gradient statistics
* \param hess second order gradient statistics
* \param len length of grad/hess
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterUpdateOneIterCustom_R(
  LGBM_SE handle,
  SEXP grad,
  SEXP hess,
  SEXP len
);

/*!
* \brief Rollback one iteration
* \param handle Booster handle
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterRollbackOneIter_R(
  LGBM_SE handle
);

/*!
* \brief Get iteration of current boosting rounds
* \param handle Booster handle
* \param out iteration of boosting rounds
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterGetCurrentIteration_R(
  LGBM_SE handle,
  SEXP out
);

/*!
* \brief Get model upper bound value.
* \param handle Handle of Booster
* \param[out] out_results Result pointing to max value
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterGetUpperBoundValue_R(
    LGBM_SE handle,
    SEXP out_result
);

/*!
* \brief Get model lower bound value.
* \param handle Handle of Booster
* \param[out] out_results Result pointing to min value
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterGetLowerBoundValue_R(
    LGBM_SE handle,
    SEXP out_result
);

/*!
* \brief Get Name of eval
* \param eval_names eval names
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterGetEvalNames_R(
  LGBM_SE handle,
  SEXP buf_len,
  SEXP actual_len,
  LGBM_SE eval_names
);

/*!
* \brief get evaluation for training data and validation data
* \param handle Booster handle
* \param data_idx 0:training data, 1: 1st valid data, 2:2nd valid data ...
* \param out_result float array containing result
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterGetEval_R(
  LGBM_SE handle,
  SEXP data_idx,
  SEXP out_result
);

/*!
* \brief Get number of prediction for training data and validation data
* \param handle Booster handle
* \param data_idx 0:training data, 1: 1st valid data, 2:2nd valid data ...
* \param out size of predict
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterGetNumPredict_R(
  LGBM_SE handle,
  SEXP data_idx,
  SEXP out
);

/*!
* \brief Get prediction for training data and validation data.
*        This can be used to support customized eval function
* \param handle Booster handle
* \param data_idx 0:training data, 1: 1st valid data, 2:2nd valid data ...
* \param out_result, used to store predict result, should pre-allocate memory
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterGetPredict_R(
  LGBM_SE handle,
  SEXP data_idx,
  SEXP out_result
);

/*!
* \brief make prediction for file
* \param handle Booster handle
* \param data_filename filename of data file
* \param data_has_header data file has header or not
* \param is_rawscore 1 to get raw predictions, before transformations like
*                    converting to probabilities, 0 otherwise
* \param is_leafidx 1 to get record of which leaf in each tree
*                   observations fell into, 0 otherwise
* \param is_predcontrib 1 to get feature contributions, 0 otherwise
* \param start_iteration Start index of the iteration to predict
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \param parameter additional parameters
* \param result_filename filename of file to write predictions to
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterPredictForFile_R(
  LGBM_SE handle,
  SEXP data_filename,
  SEXP data_has_header,
  SEXP is_rawscore,
  SEXP is_leafidx,
  SEXP is_predcontrib,
  SEXP start_iteration,
  SEXP num_iteration,
  SEXP parameter,
  SEXP result_filename
);

/*!
* \brief Get number of prediction
* \param handle Booster handle
* \param num_row number of rows in input
* \param is_rawscore 1 to get raw predictions, before transformations like
*                    converting to probabilities, 0 otherwise
* \param is_leafidx 1 to get record of which leaf in each tree
*                   observations fell into, 0 otherwise
* \param is_predcontrib 1 to get feature contributions, 0 otherwise
* \param start_iteration Start index of the iteration to predict
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \param out_len length of prediction
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterCalcNumPredict_R(
  LGBM_SE handle,
  SEXP num_row,
  SEXP is_rawscore,
  SEXP is_leafidx,
  SEXP is_predcontrib,
  SEXP start_iteration,
  SEXP num_iteration,
  SEXP out_len
);

/*!
* \brief make prediction for a new Dataset
*        Note:  should pre-allocate memory for out_result,
*               for normal and raw score: its length is equal to num_class * num_data
*               for leaf index, its length is equal to num_class * num_data * num_iteration
* \param handle Booster handle
* \param indptr pointer to row headers
* \param indices findex
* \param data fvalue
* \param num_indptr number of cols in the matrix + 1
* \param nelem number of non-zero elements in the matrix
* \param num_row number of rows
* \param is_rawscore 1 to get raw predictions, before transformations like
*                    converting to probabilities, 0 otherwise
* \param is_leafidx 1 to get record of which leaf in each tree
*                   observations fell into, 0 otherwise
* \param is_predcontrib 1 to get feature contributions, 0 otherwise
* \param start_iteration Start index of the iteration to predict
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \param parameter additional parameters
* \param out_result prediction result
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterPredictForCSC_R(
  LGBM_SE handle,
  SEXP indptr,
  SEXP indices,
  SEXP data,
  SEXP num_indptr,
  SEXP nelem,
  SEXP num_row,
  SEXP is_rawscore,
  SEXP is_leafidx,
  SEXP is_predcontrib,
  SEXP start_iteration,
  SEXP num_iteration,
  SEXP parameter,
  SEXP out_result
);

/*!
* \brief make prediction for a new Dataset
*        Note:  should pre-allocate memory for out_result,
*               for normal and raw score: its length is equal to num_class * num_data
*               for leaf index, its length is equal to num_class * num_data * num_iteration
* \param handle Booster handle
* \param data pointer to the data space
* \param num_row number of rows
* \param num_col number columns
* \param is_rawscore 1 to get raw predictions, before transformations like
*                    converting to probabilities, 0 otherwise
* \param is_leafidx 1 to get record of which leaf in each tree
*                   observations fell into, 0 otherwise
* \param is_predcontrib 1 to get feature contributions, 0 otherwise
* \param start_iteration Start index of the iteration to predict
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \param parameter additional parameters
* \param out_result prediction result
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterPredictForMat_R(
  LGBM_SE handle,
  SEXP data,
  SEXP num_row,
  SEXP num_col,
  SEXP is_rawscore,
  SEXP is_leafidx,
  SEXP is_predcontrib,
  SEXP start_iteration,
  SEXP num_iteration,
  SEXP parameter,
  SEXP out_result
);

/*!
* \brief save model into file
* \param handle Booster handle
* \param num_iteration, <= 0 means save all
* \param feature_importance_type type of feature importance, 0: split, 1: gain
* \param filename file name
* \return R NULL value
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterSaveModel_R(
  LGBM_SE handle,
  SEXP num_iteration,
  SEXP feature_importance_type,
  SEXP filename
);

/*!
* \brief create string containing model
* \param handle handle
* \param num_iteration, <= 0 means save all
* \param out_str string of model
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterSaveModelToString_R(
  LGBM_SE handle,
  SEXP num_iteration,
  SEXP feature_importance_type,
  SEXP buffer_len,
  SEXP actual_len,
  LGBM_SE out_str
);

/*!
* \brief dump model to json
* \param handle handle
* \param num_iteration, <= 0 means save all
* \param out_str json format string of model
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterDumpModel_R(
  LGBM_SE handle,
  SEXP num_iteration,
  SEXP feature_importance_type,
  SEXP buffer_len,
  SEXP actual_len,
  LGBM_SE out_str
);

#endif  // LIGHTGBM_R_H_
