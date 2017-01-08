#ifndef LIGHTGBM_R_H_
#define LIGHTGBM_R_H_

#include <LightGBM/utils/log.h>
#include <cstdint>
#include <LightGBM/c_api.h>

#include "R_object_helper.h"



/*!
* \brief get string message of the last error
*  all function in this file will return 0 when succeed
*  and -1 when an error occured,
* \return err_msg error inforomation
* \return error inforomation
*/
DllExport SEXP LGBM_GetLastError_R(SEXP buf_len, SEXP actual_len, SEXP err_msg);

// --- start Dataset interface

/*!
* \brief load data set from file like the command_line LightGBM do
* \param filename the name of the file
* \param parameters additional parameters
* \param reference used to align bin mapper with other dataset, nullptr means don't used
* \param out created dataset
* \return 0 when succeed, -1 when failure happens
*/
DllExport SEXP LGBM_DatasetCreateFromFile_R(SEXP filename,
  SEXP parameters,
  SEXP reference,
  SEXP out,
  SEXP call_state);

/*!
* \brief create a dataset from CSC format
* \param indptr pointer to row headers
* \param indices findex
* \param data fvalue
* \param nindptr number of cols in the matrix + 1
* \param nelem number of nonzero elements in the matrix
* \param num_row number of rows
* \param parameters additional parameters
* \param reference used to align bin mapper with other dataset, nullptr means don't used
* \param out created dataset
* \return 0 when succeed, -1 when failure happens
*/
DllExport SEXP LGBM_DatasetCreateFromCSC_R(SEXP indptr,
  SEXP indices,
  SEXP data,
  SEXP nindptr,
  SEXP nelem,
  SEXP num_row,
  SEXP parameters,
  SEXP reference,
  SEXP out,
  SEXP call_state);


/*!
* \brief create dataset from dense matrix
* \param data matric data
* \param nrow number of rows
* \param ncol number columns
* \param parameters additional parameters
* \param reference used to align bin mapper with other dataset, nullptr means don't used
* \param out created dataset
* \return 0 when succeed, -1 when failure happens
*/
DllExport SEXP LGBM_DatasetCreateFromMat_R(SEXP data,
  SEXP nrow,
  SEXP ncol,
  SEXP parameters,
  SEXP reference,
  SEXP out,
  SEXP call_state);

/*!
* \brief Create subset of a data
* \param handle handle of full dataset
* \param used_row_indices Indices used in subset
* \param len_used_row_indices length of Indices used in subset
* \param parameters additional parameters
* \param out created dataset
* \return 0 when succeed, -1 when failure happens
*/
DllExport SEXP LGBM_DatasetGetSubset_R(SEXP handle,
  SEXP used_row_indices,
  SEXP len_used_row_indices,
  SEXP parameters,
  SEXP out,
  SEXP call_state);

/*!
* \brief save feature names to Dataset
* \param handle handle
* \param feature_names feature names
* \return 0 when succeed, -1 when failure happens
*/
DllExport SEXP LGBM_DatasetSetFeatureNames_R(SEXP handle,
  SEXP feature_names,
  SEXP call_state);

/*!
* \brief save feature names to Dataset
* \param handle handle
* \param feature_names feature names
* \return 0 when succeed, -1 when failure happens
*/
DllExport SEXP LGBM_DatasetGetFeatureNames_R(SEXP handle,
  SEXP buf_len,
  SEXP actual_len,
  SEXP feature_names,
  SEXP call_state);

/*!
* \brief save dateset to binary file
* \param handle a instance of dataset
* \param filename file name
* \return 0 when succeed, -1 when failure happens
*/
DllExport SEXP LGBM_DatasetSaveBinary_R(SEXP handle,
  SEXP filename,
  SEXP call_state);

/*!
* \brief free dataset
* \param handle a instance of dataset
* \return 0 when succeed, -1 when failure happens
*/
DllExport SEXP LGBM_DatasetFree_R(SEXP handle,
  SEXP call_state);

/*!
* \brief set vector to a content in info
*        Note: group and group only work for C_API_DTYPE_INT32
*              label and weight only work for C_API_DTYPE_FLOAT32
* \param handle a instance of dataset
* \param field_name field name, can be label, weight, group, group_id
* \param field_data pointer to vector
* \param num_element number of element in field_data
* \return 0 when succeed, -1 when failure happens
*/
DllExport SEXP LGBM_DatasetSetField_R(SEXP handle,
  SEXP field_name,
  SEXP field_data,
  SEXP num_element,
  SEXP call_state);

/*!
* \brief get size of info vector from dataset
* \param handle a instance of dataset
* \param field_name field name
* \param out size of info vector from dataset
* \return 0 when succeed, -1 when failure happens
*/
DllExport SEXP LGBM_DatasetGetFieldSize_R(SEXP handle,
  SEXP field_name,
  SEXP out,
  SEXP call_state);

/*!
* \brief get info vector from dataset
* \param handle a instance of dataset
* \param field_name field name
* \param field_data pointer to vector
* \return 0 when succeed, -1 when failure happens
*/
DllExport SEXP LGBM_DatasetGetField_R(SEXP handle,
  SEXP field_name,
  SEXP field_data,
  SEXP call_state);

/*!
* \brief get number of data.
* \param handle the handle to the dataset
* \param out The address to hold number of data
* \return 0 when succeed, -1 when failure happens
*/
DllExport SEXP LGBM_DatasetGetNumData_R(SEXP handle,
  SEXP out,
  SEXP call_state);

/*!
* \brief get number of features
* \param handle the handle to the dataset
* \param out The output of number of features
* \return 0 when succeed, -1 when failure happens
*/
DllExport SEXP LGBM_DatasetGetNumFeature_R(SEXP handle,
  SEXP out,
  SEXP call_state);

// --- start Booster interfaces

/*!
* \brief create an new boosting learner
* \param train_data training data set
* \param parameters format: 'key1=value1 key2=value2'
* \prama out handle of created Booster
* \return 0 when succeed, -1 when failure happens
*/
DllExport SEXP LGBM_BoosterCreate_R(SEXP train_data,
  SEXP parameters,
  SEXP out,
  SEXP call_state);

/*!
* \brief free obj in handle
* \param handle handle to be freed
* \return 0 when succeed, -1 when failure happens
*/
DllExport SEXP LGBM_BoosterFree_R(SEXP handle,
  SEXP call_state);

/*!
* \brief load an existing boosting from model file
* \param filename filename of model
* \prama out handle of created Booster
* \return 0 when succeed, -1 when failure happens
*/
DllExport SEXP LGBM_BoosterCreateFromModelfile_R(SEXP filename,
  SEXP out,
  SEXP call_state);

/*!
* \brief Merge model in two booster to first handle
* \param handle handle, will merge other handle to this
* \param other_handle
* \return 0 when succeed, -1 when failure happens
*/
DllExport SEXP LGBM_BoosterMerge_R(SEXP handle,
  SEXP other_handle,
  SEXP call_state);

/*!
* \brief Add new validation to booster
* \param handle handle
* \param valid_data validation data set
* \return 0 when succeed, -1 when failure happens
*/
DllExport SEXP LGBM_BoosterAddValidData_R(SEXP handle,
  SEXP valid_data,
  SEXP call_state);

/*!
* \brief Reset training data for booster
* \param handle handle
* \param train_data training data set
* \return 0 when succeed, -1 when failure happens
*/
DllExport SEXP LGBM_BoosterResetTrainingData_R(SEXP handle,
  SEXP train_data,
  SEXP call_state);

/*!
* \brief Reset config for current booster
* \param handle handle
* \param parameters format: 'key1=value1 key2=value2'
* \return 0 when succeed, -1 when failure happens
*/
DllExport SEXP LGBM_BoosterResetParameter_R(SEXP handle,
  SEXP parameters,
  SEXP call_state);

/*!
* \brief Get number of class
* \param handle handle
* \param out number of classes
* \return 0 when succeed, -1 when failure happens
*/
DllExport SEXP LGBM_BoosterGetNumClasses_R(SEXP handle, 
  SEXP out,
  SEXP call_state);

/*!
* \brief update the model in one round
* \param handle handle
* \return 0 when succeed, -1 when failure happens
*/
DllExport SEXP LGBM_BoosterUpdateOneIter_R(SEXP handle,
  SEXP call_state);

/*!
* \brief update the model, by directly specify gradient and second order gradient,
*       this can be used to support customized loss function
* \param handle handle
* \param grad gradient statistics
* \param hess second order gradient statistics
* \param len length of grad/hess
* \return 0 when succeed, -1 when failure happens
*/
DllExport SEXP LGBM_BoosterUpdateOneIterCustom_R(SEXP handle,
  SEXP grad,
  SEXP hess,
  SEXP len,
  SEXP call_state);

/*!
* \brief Rollback one iteration
* \param handle handle
* \return 0 when succeed, -1 when failure happens
*/
DllExport SEXP LGBM_BoosterRollbackOneIter_R(SEXP handle,
  SEXP call_state);

/*!
* \brief Get iteration of current boosting rounds
* \param out iteration of boosting rounds
* \return 0 when succeed, -1 when failure happens
*/
DllExport SEXP LGBM_BoosterGetCurrentIteration_R(SEXP handle, 
  SEXP out,
  SEXP call_state);

/*!
* \brief Get Name of eval
* \param eval_names eval names
* \return 0 when succeed, -1 when failure happens
*/
DllExport SEXP LGBM_BoosterGetEvalNames_R(SEXP handle, 
  SEXP buf_len,
  SEXP actual_len,
  SEXP eval_names,
  SEXP call_state);

/*!
* \brief get evaluation for training data and validation data
* \param handle handle
* \param data_idx 0:training data, 1: 1st valid data, 2:2nd valid data ...
* \param out_result float arrary contains result
* \return 0 when succeed, -1 when failure happens
*/
DllExport SEXP LGBM_BoosterGetEval_R(SEXP handle,
  SEXP data_idx,
  SEXP out_result,
  SEXP call_state);

/*!
* \brief Get number of prediction for training data and validation data
* \param handle handle
* \param data_idx 0:training data, 1: 1st valid data, 2:2nd valid data ...
* \param out size of predict
* \return 0 when succeed, -1 when failure happens
*/
DllExport SEXP LGBM_BoosterGetNumPredict_R(SEXP handle,
  SEXP data_idx,
  SEXP out,
  SEXP call_state);

/*!
* \brief Get prediction for training data and validation data
this can be used to support customized eval function
* \param handle handle
* \param data_idx 0:training data, 1: 1st valid data, 2:2nd valid data ...
* \param out_result, used to store predict result, should pre-allocate memory
* \return 0 when succeed, -1 when failure happens
*/
DllExport SEXP LGBM_BoosterGetPredict_R(SEXP handle,
  SEXP data_idx,
  SEXP out_result,
  SEXP call_state);

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
DllExport SEXP LGBM_BoosterPredictForFile_R(SEXP handle,
  SEXP data_filename,
  SEXP data_has_header,
  SEXP is_rawscore,
  SEXP is_leafidx,
  SEXP num_iteration,
  SEXP result_filename,
  SEXP call_state);

/*!
* \brief Get number of prediction
* \param handle handle
* \param num_row
* \param is_rawscore
* \param is_leafidx
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \param out_len lenght of prediction
* \return 0 when succeed, -1 when failure happens
*/
DllExport SEXP LGBM_BoosterCalcNumPredict_R(SEXP handle,
  SEXP num_row,
  SEXP is_rawscore,
  SEXP is_leafidx,
  SEXP num_iteration,
  SEXP out_len,
  SEXP call_state);

/*!
* \brief make prediction for an new data set
*        Note:  should pre-allocate memory for out_result,
*               for noraml and raw score: its length is equal to num_class * num_data
*               for leaf index, its length is equal to num_class * num_data * num_iteration
* \param handle handle
* \param indptr pointer to row headers
* \param indices findex
* \param data fvalue
* \param nindptr number of cols in the matrix + 1
* \param nelem number of nonzero elements in the matrix
* \param num_row number of rows
* \param is_rawscore
* \param is_leafidx
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \param out prediction result
* \return 0 when succeed, -1 when failure happens
*/
DllExport SEXP LGBM_BoosterPredictForCSC_R(SEXP handle,
  SEXP indptr,
  SEXP indices,
  SEXP data,
  SEXP nindptr,
  SEXP nelem,
  SEXP num_row,
  SEXP is_rawscore,
  SEXP is_leafidx,
  SEXP num_iteration,
  SEXP out_result,
  SEXP call_state);

/*!
* \brief make prediction for an new data set
*        Note:  should pre-allocate memory for out_result,
*               for noraml and raw score: its length is equal to num_class * num_data
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
DllExport SEXP LGBM_BoosterPredictForMat_R(SEXP handle,
  SEXP data,
  SEXP nrow,
  SEXP ncol,
  SEXP is_rawscore,
  SEXP is_leafidx,
  SEXP num_iteration,
  SEXP out_result,
  SEXP call_state);

/*!
* \brief save model into file
* \param handle handle
* \param num_iteration, <= 0 means save all
* \param filename file name
* \return 0 when succeed, -1 when failure happens
*/
DllExport SEXP LGBM_BoosterSaveModel_R(SEXP handle,
  SEXP num_iteration,
  SEXP filename,
  SEXP call_state);

/*!
* \brief dump model to json
* \param handle handle
* \param num_iteration, <= 0 means save all
* \param out_str json format string of model
* \return 0 when succeed, -1 when failure happens
*/
DllExport SEXP LGBM_BoosterDumpModel_R(SEXP handle,
  SEXP num_iteration,
  SEXP buffer_len,
  SEXP actual_len,
  SEXP out_str,
  SEXP call_state);

#endif // LIGHTGBM_R_H_