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
*  all functions in this file will return 0 on success
*  and -1 when an error occurred
* \return err_msg error information
* \return error information
*/
LIGHTGBM_C_EXPORT SEXP LGBM_GetLastError_R();

// --- start Dataset interface

/*!
* \brief load data set from file like the command_line LightGBM does
* \param filename the name of the file
* \param parameters additional parameters
* \param reference used to align bin mapper with other dataset, nullptr means not used
* \param out created dataset
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetCreateFromFile_R(
  LGBM_SE filename,
  LGBM_SE parameters,
  LGBM_SE reference,
  LGBM_SE out
);

/*!
* \brief create a dataset from CSC format
* \param indptr pointer to row headers
* \param indices findex
* \param data fvalue
* \param nindptr number of cols in the matrix + 1
* \param nelem number of nonzero elements in the matrix
* \param num_row number of rows
* \param parameters additional parameters
* \param reference used to align bin mapper with other dataset, nullptr means not used
* \param out created dataset
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetCreateFromCSC_R(
  LGBM_SE indptr,
  LGBM_SE indices,
  LGBM_SE data,
  LGBM_SE nindptr,
  LGBM_SE nelem,
  LGBM_SE num_row,
  LGBM_SE parameters,
  LGBM_SE reference,
  LGBM_SE out
);

/*!
* \brief create dataset from dense matrix
* \param data matric data
* \param nrow number of rows
* \param ncol number columns
* \param parameters additional parameters
* \param reference used to align bin mapper with other dataset, nullptr means not used
* \param out created dataset
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetCreateFromMat_R(
  LGBM_SE data,
  LGBM_SE nrow,
  LGBM_SE ncol,
  LGBM_SE parameters,
  LGBM_SE reference,
  LGBM_SE out
);

/*!
* \brief Create subset of a data
* \param handle handle of full dataset
* \param used_row_indices Indices used in subset
* \param len_used_row_indices length of Indices used in subset
* \param parameters additional parameters
* \param out created dataset
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetGetSubset_R(
  LGBM_SE handle,
  LGBM_SE used_row_indices,
  LGBM_SE len_used_row_indices,
  LGBM_SE parameters,
  LGBM_SE out
);

/*!
* \brief save feature names to Dataset
* \param handle handle
* \param feature_names feature names
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetSetFeatureNames_R(
  LGBM_SE handle,
  LGBM_SE feature_names
);

/*!
* \brief save feature names to Dataset
* \param handle handle
* \param feature_names feature names
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetGetFeatureNames_R(
  LGBM_SE handle,
  LGBM_SE buf_len,
  LGBM_SE actual_len,
  LGBM_SE feature_names
);

/*!
* \brief save dataset to binary file
* \param handle an instance of dataset
* \param filename file name
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetSaveBinary_R(
  LGBM_SE handle,
  LGBM_SE filename
);

/*!
* \brief free dataset
* \param handle an instance of dataset
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetFree_R(
  LGBM_SE handle
);

/*!
* \brief set vector to a content in info
*        Note: group and group_id only work for C_API_DTYPE_INT32
*              label and weight only work for C_API_DTYPE_FLOAT32
* \param handle an instance of dataset
* \param field_name field name, can be label, weight, group, group_id
* \param field_data pointer to vector
* \param num_element number of element in field_data
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetSetField_R(
  LGBM_SE handle,
  LGBM_SE field_name,
  LGBM_SE field_data,
  LGBM_SE num_element
);

/*!
* \brief get size of info vector from dataset
* \param handle an instance of dataset
* \param field_name field name
* \param out size of info vector from dataset
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetGetFieldSize_R(
  LGBM_SE handle,
  LGBM_SE field_name,
  LGBM_SE out
);

/*!
* \brief get info vector from dataset
* \param handle an instance of dataset
* \param field_name field name
* \param field_data pointer to vector
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetGetField_R(
  LGBM_SE handle,
  LGBM_SE field_name,
  LGBM_SE field_data
);

/*!
 * \brief Raise errors for attempts to update dataset parameters
 * \param old_params Current dataset parameters
 * \param new_params New dataset parameters
 * \return 0 when succeed, -1 when failure happens
 */
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetUpdateParamChecking_R(
  LGBM_SE old_params,
  LGBM_SE new_params
);

/*!
* \brief get number of data.
* \param handle the handle to the dataset
* \param out The address to hold number of data
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetGetNumData_R(
  LGBM_SE handle,
  LGBM_SE out
);

/*!
* \brief get number of features
* \param handle the handle to the dataset
* \param out The output of number of features
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_DatasetGetNumFeature_R(
  LGBM_SE handle,
  LGBM_SE out
);

// --- start Booster interfaces

/*!
* \brief create a new boosting learner
* \param train_data training data set
* \param parameters format: 'key1=value1 key2=value2'
* \param out handle of created Booster
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterCreate_R(
  LGBM_SE train_data,
  LGBM_SE parameters,
  LGBM_SE out
);

/*!
* \brief free obj in handle
* \param handle handle to be freed
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterFree_R(
  LGBM_SE handle
);

/*!
* \brief load an existing boosting from model file
* \param filename filename of model
* \param out handle of created Booster
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterCreateFromModelfile_R(
  LGBM_SE filename,
  LGBM_SE out
);

/*!
* \brief load an existing boosting from model_str
* \param model_str string containing the model
* \param out handle of created Booster
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterLoadModelFromString_R(
  LGBM_SE model_str,
  LGBM_SE out
);

/*!
* \brief Merge model in two boosters to first handle
* \param handle handle, will merge other handle to this
* \param other_handle
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterMerge_R(
  LGBM_SE handle,
  LGBM_SE other_handle
);

/*!
* \brief Add new validation to booster
* \param handle handle
* \param valid_data validation data set
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterAddValidData_R(
  LGBM_SE handle,
  LGBM_SE valid_data
);

/*!
* \brief Reset training data for booster
* \param handle handle
* \param train_data training data set
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterResetTrainingData_R(
  LGBM_SE handle,
  LGBM_SE train_data
);

/*!
* \brief Reset config for current booster
* \param handle handle
* \param parameters format: 'key1=value1 key2=value2'
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterResetParameter_R(
  LGBM_SE handle,
  LGBM_SE parameters
);

/*!
* \brief Get number of classes
* \param handle handle
* \param out number of classes
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterGetNumClasses_R(
  LGBM_SE handle,
  LGBM_SE out
);

/*!
* \brief update the model in one round
* \param handle handle
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterUpdateOneIter_R(
  LGBM_SE handle
);

/*!
* \brief update the model, by directly specify gradient and second order gradient,
*       this can be used to support customized loss function
* \param handle handle
* \param grad gradient statistics
* \param hess second order gradient statistics
* \param len length of grad/hess
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterUpdateOneIterCustom_R(
  LGBM_SE handle,
  LGBM_SE grad,
  LGBM_SE hess,
  LGBM_SE len
);

/*!
* \brief Rollback one iteration
* \param handle handle
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterRollbackOneIter_R(
  LGBM_SE handle
);

/*!
* \brief Get iteration of current boosting rounds
* \param out iteration of boosting rounds
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterGetCurrentIteration_R(
  LGBM_SE handle,
  LGBM_SE out
);

/*!
* \brief Get model upper bound value.
* \param handle Handle of booster
* \param[out] out_results Result pointing to max value
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterGetUpperBoundValue_R(
    LGBM_SE handle,
    LGBM_SE out_result
);

/*!
* \brief Get model lower bound value.
* \param handle Handle of booster
* \param[out] out_results Result pointing to min value
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterGetLowerBoundValue_R(
    LGBM_SE handle,
    LGBM_SE out_result
);

/*!
* \brief Get Name of eval
* \param eval_names eval names
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterGetEvalNames_R(
  LGBM_SE handle,
  LGBM_SE buf_len,
  LGBM_SE actual_len,
  LGBM_SE eval_names
);

/*!
* \brief get evaluation for training data and validation data
* \param handle handle
* \param data_idx 0:training data, 1: 1st valid data, 2:2nd valid data ...
* \param out_result float array contains result
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterGetEval_R(
  LGBM_SE handle,
  LGBM_SE data_idx,
  LGBM_SE out_result
);

/*!
* \brief Get number of prediction for training data and validation data
* \param handle handle
* \param data_idx 0:training data, 1: 1st valid data, 2:2nd valid data ...
* \param out size of predict
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterGetNumPredict_R(
  LGBM_SE handle,
  LGBM_SE data_idx,
  LGBM_SE out
);

/*!
* \brief Get prediction for training data and validation data.
*        This can be used to support customized eval function
* \param handle handle
* \param data_idx 0:training data, 1: 1st valid data, 2:2nd valid data ...
* \param out_result, used to store predict result, should pre-allocate memory
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterGetPredict_R(
  LGBM_SE handle,
  LGBM_SE data_idx,
  LGBM_SE out_result
);

/*!
* \brief make prediction for file
* \param handle handle
* \param data_filename filename of data file
* \param data_has_header data file has header or not
* \param is_rawscore
* \param is_leafidx
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \return 0 when succeed, -1 when failure happens
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterPredictForFile_R(
  LGBM_SE handle,
  LGBM_SE data_filename,
  LGBM_SE data_has_header,
  LGBM_SE is_rawscore,
  LGBM_SE is_leafidx,
  LGBM_SE is_predcontrib,
  LGBM_SE start_iteration,
  LGBM_SE num_iteration,
  LGBM_SE parameter,
  LGBM_SE result_filename
);

/*!
* \brief Get number of prediction
* \param handle handle
* \param num_row
* \param is_rawscore
* \param is_leafidx
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \param out_len length of prediction
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterCalcNumPredict_R(
  LGBM_SE handle,
  LGBM_SE num_row,
  LGBM_SE is_rawscore,
  LGBM_SE is_leafidx,
  LGBM_SE is_predcontrib,
  LGBM_SE start_iteration,
  LGBM_SE num_iteration,
  LGBM_SE out_len
);

/*!
* \brief make prediction for a new data set
*        Note:  should pre-allocate memory for out_result,
*               for normal and raw score: its length is equal to num_class * num_data
*               for leaf index, its length is equal to num_class * num_data * num_iteration
* \param handle handle
* \param indptr pointer to row headers
* \param indices findex
* \param data fvalue
* \param nindptr number of cols in the matrix + 1
* \param nelem number of non-zero elements in the matrix
* \param num_row number of rows
* \param is_rawscore
* \param is_leafidx
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \param out prediction result
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterPredictForCSC_R(
  LGBM_SE handle,
  LGBM_SE indptr,
  LGBM_SE indices,
  LGBM_SE data,
  LGBM_SE nindptr,
  LGBM_SE nelem,
  LGBM_SE num_row,
  LGBM_SE is_rawscore,
  LGBM_SE is_leafidx,
  LGBM_SE is_predcontrib,
  LGBM_SE start_iteration,
  LGBM_SE num_iteration,
  LGBM_SE parameter,
  LGBM_SE out_result
);

/*!
* \brief make prediction for a new data set
*        Note:  should pre-allocate memory for out_result,
*               for normal and raw score: its length is equal to num_class * num_data
*               for leaf index, its length is equal to num_class * num_data * num_iteration
* \param handle handle
* \param data pointer to the data space
* \param nrow number of rows
* \param ncol number columns
* \param is_rawscore
* \param is_leafidx
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \param out prediction result
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterPredictForMat_R(
  LGBM_SE handle,
  LGBM_SE data,
  LGBM_SE nrow,
  LGBM_SE ncol,
  LGBM_SE is_rawscore,
  LGBM_SE is_leafidx,
  LGBM_SE is_predcontrib,
  LGBM_SE start_iteration,
  LGBM_SE num_iteration,
  LGBM_SE parameter,
  LGBM_SE out_result
);

/*!
* \brief save model into file
* \param handle handle
* \param num_iteration, <= 0 means save all
* \param filename file name
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT SEXP LGBM_BoosterSaveModel_R(
  LGBM_SE handle,
  LGBM_SE num_iteration,
  LGBM_SE feature_importance_type,
  LGBM_SE filename
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
  LGBM_SE num_iteration,
  LGBM_SE feature_importance_type,
  LGBM_SE buffer_len,
  LGBM_SE actual_len,
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
  LGBM_SE num_iteration,
  LGBM_SE feature_importance_type,
  LGBM_SE buffer_len,
  LGBM_SE actual_len,
  LGBM_SE out_str
);

#endif  // LIGHTGBM_R_H_
