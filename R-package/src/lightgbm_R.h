#ifndef LIGHTGBM_R_H_
#define LIGHTGBM_R_H_

#include <Rinternals.h>
#include <R_ext/Random.h>
#include <Rmath.h>

#include <LightGBM/c_api.h>

/*!
* \brief check whether a handle is NULL
* \param handle
* \return whether it is null ptr
*/
DllExport SEXP LGBMCheckNullPtr_R(SEXP handle);


// --- start Dataset interface

/*!
* \brief load data set from file like the command_line LightGBM do
* \param filename the name of the file
* \param parameters additional parameters
* \param reference used to align bin mapper with other dataset, nullptr means don't used
* \return loaded dataset
*/
DllExport SEXP LGBM_DatasetCreateFromFile_R(SEXP filename,
  SEXP parameters,
  SEXP reference);

/*!
* \brief create a dataset from CSR format
* \param indptr pointer to row headers
* \param indices findex
* \param data fvalue
* \param num_col number of columns
* \param parameters additional parameters
* \param reference used to align bin mapper with other dataset, nullptr means don't used
* \return created dataset
*/
DllExport SEXP LGBM_DatasetCreateFromCSR_R(SEXP indptr,
  SEXP indices,
  SEXP data,
  SEXP num_col,
  SEXP parameters,
  SEXP reference);


/*!
* \brief create dataset from dense matrix
* \param data matric data
* \param parameters additional parameters
* \param reference used to align bin mapper with other dataset, nullptr means don't used
* \return created dataset
*/
DllExport SEXP LGBM_DatasetCreateFromMat_R(SEXP data,
  SEXP parameters,
  SEXP reference);

/*!
* \brief Create subset of a data
* \param handle handle of full dataset
* \param used_row_indices Indices used in subset
* \param parameters additional parameters
* \return subset of data
*/
DllExport SEXP LGBM_DatasetGetSubset_R(SEXP handle,
  SEXP used_row_indices,
  SEXP parameters);

/*!
* \brief save feature names to Dataset
* \param handle handle
* \param feature_names feature names
* \return R_NilValue
*/
DllExport SEXP LGBM_DatasetSetFeatureNames_R(SEXP handle,
  SEXP feature_names);


/*!
* \brief save dateset to binary file
* \param handle a instance of dataset
* \param filename file name
* \return R_NilValue
*/
DllExport SEXP LGBM_DatasetSaveBinary_R(SEXP handle,
  SEXP filename);

/*!
* \brief set vector to a content in info
*        Note: group and group only work for C_API_DTYPE_INT32
*              label and weight only work for C_API_DTYPE_FLOAT32
* \param handle a instance of dataset
* \param field_name field name, can be label, weight, group, group_id
* \param field_data pointer to vector
* \return R_NilValue
*/
DllExport SEXP LGBM_DatasetSetField_R(SEXP handle,
  SEXP field_name,
  SEXP field_data);

/*!
* \brief get info vector from dataset
* \param handle a instance of data matrix
* \param field_name field name
* \return the result
*/
DllExport SEXP LGBM_DatasetGetField_R(SEXP handle,
  SEXP field_name);

/*!
* \brief get number of data.
* \param handle the handle to the dataset
* \return number of data
*/
DllExport SEXP LGBM_DatasetGetNumData_R(SEXP handle);

/*!
* \brief get number of features
* \param handle the handle to the dataset
* \return number of features
*/
DllExport SEXP LGBM_DatasetGetNumFeature_R(SEXP handle);

// --- start Booster interfaces

/*!
* \brief create an new boosting learner
* \param train_data training data set
* \param parameters format: 'key1=value1 key2=value2'
* \return out created Booster
*/
DllExport SEXP LGBM_BoosterCreate_R(SEXP train_data,
  SEXP parameters);

/*!
* \brief load an existing boosting from model file
* \param filename filename of model
* \return handle of created Booster
*/
DllExport SEXP LGBM_BoosterCreateFromModelfile_R(SEXP filename);

/*!
* \brief Merge model in two booster to first handle
* \param handle handle, will merge other handle to this
* \param other_handle
* \return R_NilValue
*/
DllExport SEXP LGBM_BoosterMerge_R(SEXP handle,
  SEXP other_handle);

/*!
* \brief Add new validation to booster
* \param handle handle
* \param valid_data validation data set
* \return R_NilValue
*/
DllExport SEXP LGBM_BoosterAddValidData_R(SEXP handle,
  SEXP valid_data);

/*!
* \brief Reset training data for booster
* \param handle handle
* \param train_data training data set
* \return R_NilValue
*/
DllExport SEXP LGBM_BoosterResetTrainingData_R(SEXP handle,
  SEXP train_data);

/*!
* \brief Reset config for current booster
* \param handle handle
* \param parameters format: 'key1=value1 key2=value2'
* \return R_NilValue
*/
DllExport SEXP LGBM_BoosterResetParameter_R(SEXP handle, SEXP parameters);

/*!
* \brief Get number of class
* \param handle handle
* \return number of classes
*/
DllExport SEXP LGBM_BoosterGetNumClasses_R(SEXP handle);

/*!
* \brief update the model in one round
* \param handle handle
* \return bool, true means finished
*/
DllExport SEXP LGBM_BoosterUpdateOneIter_R(SEXP handle);

/*!
* \brief update the model, by directly specify gradient and second order gradient,
*       this can be used to support customized loss function
* \param handle handle
* \param grad gradient statistics
* \param hess second order gradient statistics
* \return bool, true means finished
*/
DllExport SEXP LGBM_BoosterUpdateOneIterCustom_R(SEXP handle,
  SEXP grad,
  SEXP hess);

/*!
* \brief Rollback one iteration
* \param handle handle
* \return R_NilValue
*/
DllExport SEXP LGBM_BoosterRollbackOneIter_R(SEXP handle);

/*!
* \brief Get iteration of current boosting rounds
* \return iteration of boosting rounds
*/
DllExport SEXP LGBM_BoosterGetCurrentIteration_R(SEXP handle);

/*!
* \brief Get number of eval
* \return total number of eval results
*/
DllExport SEXP LGBM_BoosterGetEvalCounts_R(SEXP handle);

/*!
* \brief Get Name of eval
* \return out_strs names of eval result
*/
DllExport SEXP LGBM_BoosterGetEvalNames_R(SEXP handle);

/*!
* \brief get evaluation for training data and validation data
* \param handle handle
* \param data_idx 0:training data, 1: 1st valid data, 2:2nd valid data ...
* \return float arrary contains result
*/
DllExport SEXP LGBM_BoosterGetEval_R(SEXP handle,
  SEXP data_idx);

/*!
* \brief Get prediction for training data and validation data
this can be used to support customized eval function
* \param handle handle
* \param data_idx 0:training data, 1: 1st valid data, 2:2nd valid data ...
* \return prediction result
*/
DllExport SEXP LGBM_BoosterGetPredict_R(SEXP handle,
  SEXP data_idx);

/*!
* \brief make prediction for file
* \param handle handle
* \param data_filename filename of data file
* \param data_has_header data file has header or not
* \param predict_type
*          C_API_PREDICT_NORMAL: normal prediction, with transform _R(if needed)
*          C_API_PREDICT_RAW_SCORE: raw score
*          C_API_PREDICT_LEAF_INDEX: leaf index
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \param result_filename filename of result file
* \return R_NilValue
*/
DllExport SEXP LGBM_BoosterPredictForFile_R(SEXP handle,
  SEXP data_filename,
  SEXP data_has_header,
  SEXP predict_type,
  SEXP num_iteration,
  SEXP result_filename);

/*!
* \brief make prediction for an new data set
*        Note:  should pre-allocate memory for out_result,
*               for noraml and raw score: its length is equal to num_class * num_data
*               for leaf index, its length is equal to num_class * num_data * num_iteration
* \param handle handle
* \param indptr pointer to row headers
* \param indices findex
* \param data fvalue
* \param num_col number of columns
* \param predict_type
*          C_API_PREDICT_NORMAL: normal prediction, with transform _R(if needed)
*          C_API_PREDICT_RAW_SCORE: raw score
*          C_API_PREDICT_LEAF_INDEX: leaf index
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \return prediction result
*/
DllExport SEXP LGBM_BoosterPredictForCSR_R(SEXP handle,
  SEXP indptr,
  SEXP indices,
  SEXP data,
  SEXP num_col,
  SEXP predict_type,
  SEXP num_iteration);

/*!
* \brief make prediction for an new data set
*        Note:  should pre-allocate memory for out_result,
*               for noraml and raw score: its length is equal to num_class * num_data
*               for leaf index, its length is equal to num_class * num_data * num_iteration
* \param handle handle
* \param data pointer to the data space
* \param predict_type
*          C_API_PREDICT_NORMAL: normal prediction, with transform _R(if needed)
*          C_API_PREDICT_RAW_SCORE: raw score
*          C_API_PREDICT_LEAF_INDEX: leaf index
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \return prediction result
*/
DllExport SEXP LGBM_BoosterPredictForMat_R(SEXP handle,
  SEXP data,
  SEXP predict_type,
  SEXP num_iteration);

/*!
* \brief save model into file
* \param handle handle
* \param num_iteration, <= 0 means save all
* \param filename file name
* \return R_NilValue
*/
DllExport SEXP LGBM_BoosterSaveModel_R(SEXP handle,
  SEXP num_iteration,
  SEXP filename);

/*!
* \brief dump model to json
* \param handle handle
* \return json format string of model
*/
DllExport SEXP LGBM_BoosterDumpModel_R(SEXP handle);

#endif // LIGHTGBM_R_H_