/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_C_API_H_
#define LIGHTGBM_C_API_H_

#include <LightGBM/export.h>

#include <cstdint>
#include <cstring>

/*!
* To avoid type conversion on large data, most of our expose interface support both for float_32 and float_64.
* Except following:
* 1. gradients and hessians.
* 2. Get current score for training data and validation
* The reason is because they are called frequently, the type-conversion on them maybe time cost.
*/

typedef void* DatasetHandle;
typedef void* BoosterHandle;

#define C_API_DTYPE_FLOAT32 (0)
#define C_API_DTYPE_FLOAT64 (1)
#define C_API_DTYPE_INT32   (2)
#define C_API_DTYPE_INT64   (3)
#define C_API_DTYPE_INT8    (4)

#define C_API_PREDICT_NORMAL     (0)
#define C_API_PREDICT_RAW_SCORE  (1)
#define C_API_PREDICT_LEAF_INDEX (2)
#define C_API_PREDICT_CONTRIB    (3)

/*!
 * \fn LGBM_GetLastError
 * \headerfile <LightGBM/export.h>
 * \brief Get string message of the last error.
 * \return error information
*/
LIGHTGBM_C_EXPORT const char* LGBM_GetLastError();

// --- start Dataset interface

/*!
 * \fn LGBM_DatasetCreateFromFile
 * \brief Load dataset from file (like LightGBM CLI version does).
 * \param filename The name of the file
 * \param parameters Additional parameters
 * \param reference Used to align bin mapper with other dataset, nullptr means isn't used
 * \param[out] out A loaded dataset
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetCreateFromFile(const char* filename,
                                                 const char* parameters,
                                                 const DatasetHandle reference,
                                                 DatasetHandle* out);

/*!
 * \fn LGBM_DatasetCreateFromSampledColumn
 * \brief Create an empty dataset by sampling data.
 * \param sample_indices Indices of sampled data
 * \param ncol Number of columns
 * \param num_per_col Size of each sampling column
 * \param num_sample_row Number of sampled rows
 * \param num_total_row Number of total rows
 * \param parameters Additional parameters
 * \param[out] out Created dataset
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetCreateFromSampledColumn(double** sample_data,
                                                          int** sample_indices,
                                                          int32_t ncol,
                                                          const int* num_per_col,
                                                          int32_t num_sample_row,
                                                          int32_t num_total_row,
                                                          const char* parameters,
                                                          DatasetHandle* out);

/*!
 * \fn LGBM_DatasetCreateByReference
 * \brief Create an empty dataset by reference Dataset.
 * \param reference Used to align bin mapper with other dataset
 * \param num_total_row Number of total rows
 * \param[out] out Created dataset
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetCreateByReference(const DatasetHandle reference,
                                                    int64_t num_total_row,
                                                    DatasetHandle* out);

/*!
 * \fn LGBM_DatasetPushRows
 * \brief Push data to existing dataset, if nrow + start_row == num_total_row, will call dataset->FinishLoad.
 * \param dataset Handle of dataset
 * \param data Pointer to the data space
 * \param data_type Type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64)
 * \param nrow Number of rows
 * \param ncol Number of columns
 * \param start_row Row start index
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetPushRows(DatasetHandle dataset,
                                           const void* data,
                                           int data_type,
                                           int32_t nrow,
                                           int32_t ncol,
                                           int32_t start_row);

/*!
 * \fn LGBM_DatasetPushRowsByCSR
 * \brief Push data to existing dataset, if nrow + start_row == num_total_row, will call dataset->FinishLoad.
 * \param dataset Handle of dataset
 * \param indptr Pointer to row headers
 * \param indptr_type Type of indptr, can be C_API_DTYPE_INT32 or C_API_DTYPE_INT64
 * \param indices Pointer to column indices
 * \param data Pointer to the data space
 * \param data_type Type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64
 * \param nindptr Number of rows in the matrix + 1
 * \param nelem Number of nonzero elements in the matrix
 * \param num_col Number of columns
 * \param start_row Row start index
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetPushRowsByCSR(DatasetHandle dataset,
                                                const void* indptr,
                                                int indptr_type,
                                                const int32_t* indices,
                                                const void* data,
                                                int data_type,
                                                int64_t nindptr,
                                                int64_t nelem,
                                                int64_t num_col,
                                                int64_t start_row);

/*!
 * \fn LGBM_DatasetCreateFromCSR
 * \brief Create a dataset from CSR format.
 * \param indptr Pointer to row headers
 * \param indptr_type Type of indptr, can be C_API_DTYPE_INT32 or C_API_DTYPE_INT64
 * \param indices Pointer to column indices
 * \param data Pointer to the data space
 * \param data_type Type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64
 * \param nindptr Number of rows in the matrix + 1
 * \param nelem Number of nonzero elements in the matrix
 * \param num_col Number of columns
 * \param parameters Additional parameters
 * \param reference Used to align bin mapper with other dataset, nullptr means isn't used
 * \param[out] out Created dataset
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetCreateFromCSR(const void* indptr,
                                                int indptr_type,
                                                const int32_t* indices,
                                                const void* data,
                                                int data_type,
                                                int64_t nindptr,
                                                int64_t nelem,
                                                int64_t num_col,
                                                const char* parameters,
                                                const DatasetHandle reference,
                                                DatasetHandle* out);

/*!
 * \fn LGBM_DatasetCreateFromCSRFunc
 * \brief Create a dataset from CSR format through callbacks.
 * \param get_row_funptr Pointer to std::function<void(int idx, std::vector<std::pair<int, double>>& ret)
 *          (called for every row and expected to clear and fill ret)
 * \param num_rows Number of rows
 * \param num_col Number of columns
 * \param parameters Additional parameters
 * \param reference Used to align bin mapper with other dataset, nullptr means isn't used
 * \param[out] out Created dataset
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetCreateFromCSRFunc(void* get_row_funptr,
                                                    int num_rows,
                                                    int64_t num_col,
                                                    const char* parameters,
                                                    const DatasetHandle reference,
                                                    DatasetHandle* out);



/*!
 * \fn LGBM_DatasetCreateFromCSC
 * \brief Create a dataset from CSC format.
 * \param col_ptr Pointer to column headers
 * \param col_ptr_type Type of col_ptr, can be C_API_DTYPE_INT32 or C_API_DTYPE_INT64
 * \param indices Pointer to row indices
 * \param data Pointer to the data space
 * \param data_type Type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64
 * \param ncol_ptr Number of columns in the matrix + 1
 * \param nelem Number of nonzero elements in the matrix
 * \param num_row Number of rows
 * \param parameters Additional parameters
 * \param reference Used to align bin mapper with other dataset, nullptr means isn't used
 * \param[out] out Created dataset
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetCreateFromCSC(const void* col_ptr,
                                                int col_ptr_type,
                                                const int32_t* indices,
                                                const void* data,
                                                int data_type,
                                                int64_t ncol_ptr,
                                                int64_t nelem,
                                                int64_t num_row,
                                                const char* parameters,
                                                const DatasetHandle reference,
                                                DatasetHandle* out);

/*!
 * \fn LGBM_DatasetCreateFromMat
 * \brief Create dataset from dense matrix.
 * \param data Pointer to the data space
 * \param data_type Type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64
 * \param nrow Number of rows
 * \param ncol Number of columns
 * \param is_row_major 1 for row-major, 0 for column-major
 * \param parameters Additional parameters
 * \param reference Used to align bin mapper with other dataset, nullptr means isn't used
 * \param[out] out Created dataset
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetCreateFromMat(const void* data,
                                                int data_type,
                                                int32_t nrow,
                                                int32_t ncol,
                                                int is_row_major,
                                                const char* parameters,
                                                const DatasetHandle reference,
                                                DatasetHandle* out);

/*!
 * \fn LGBM_DatasetCreateFromMats
 * \brief Create dataset from array of dense matrices.
 * \param nmat Number of dense matrices
 * \param data Pointer to the data space
 * \param data_type Type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64
 * \param nrow Number of rows
 * \param ncol Number of columns
 * \param is_row_major 1 for row-major, 0 for column-major
 * \param parameters Additional parameters
 * \param reference Used to align bin mapper with other dataset, nullptr means isn't used
 * \param[out] out Created dataset
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetCreateFromMats(int32_t nmat,
                                                 const void** data,
                                                 int data_type,
                                                 int32_t* nrow,
                                                 int32_t ncol,
                                                 int is_row_major,
                                                 const char* parameters,
                                                 const DatasetHandle reference,
                                                 DatasetHandle* out);

/*!
 * \fn LGBM_DatasetGetSubset
 * \brief Create subset of a data.
 * \param handle Handle of full dataset
 * \param used_row_indices Indices used in subset
 * \param num_used_row_indices Len of used_row_indices
 * \param parameters Additional parameters
 * \param[out] out Subset of data
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetGetSubset(const DatasetHandle handle,
                                            const int32_t* used_row_indices,
                                            int32_t num_used_row_indices,
                                            const char* parameters,
                                            DatasetHandle* out);

/*!
 * \fn LGBM_DatasetSetFeatureNames
 * \brief Save feature names to dataset.
 * \param handle Handle of dataset
 * \param feature_names Feature names
 * \param num_feature_names Number of feature names
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetSetFeatureNames(DatasetHandle handle,
                                                  const char** feature_names,
                                                  int num_feature_names);


/*!
 * \fn LGBM_DatasetGetFeatureNames
 * \brief Get feature names of dataset.
 * \param handle Handle of dataset
 * \param[out] feature_names Feature names, should pre-allocate memory
 * \param[out] num_feature_names Number of feature names
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetGetFeatureNames(DatasetHandle handle,
                                                  char** feature_names,
                                                  int* num_feature_names);


/*!
 * \fn LGBM_DatasetFree
 * \brief Free space for dataset.
 * \param handle Handle of dataset to be freed
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetFree(DatasetHandle handle);

/*!
 * \fn LGBM_DatasetSaveBinary
 * \brief Save dataset to binary file.
 * \param handle Handle of dataset
 * \param filename File name
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetSaveBinary(DatasetHandle handle,
                                             const char* filename);

/*!
 * \fn LGBM_DatasetDumpText
 * \brief Save dataset to text file, intended for debugging use only.
 * \param handle Handle of dataset
 * \param filename File name
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetDumpText(DatasetHandle handle,
                                           const char* filename);

/*!
 * \fn LGBM_DatasetSetField
 * \brief Set vector to a content in info.
 *        Note: monotone_constraints only works for C_API_DTYPE_INT8,
 *              group only works for C_API_DTYPE_INT32,
 *              label and weight only work for C_API_DTYPE_FLOAT32,
 *              init_score and feature_penalty only work for C_API_DTYPE_FLOAT64.
 * \param handle Handle of dataset
 * \param field_name Field name, can be label, weight, init_score, group, feature_penalty, monotone_constraints
 * \param field_data Pointer to data vector
 * \param num_element Number of elements in field_data
 * \param type Type of data pointer, can be C_API_DTYPE_INT8, C_API_DTYPE_INT32, C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetSetField(DatasetHandle handle,
                                           const char* field_name,
                                           const void* field_data,
                                           int num_element,
                                           int type);

/*!
 * \fn LGBM_DatasetGetField
 * \brief Get info vector from dataset.
 * \param handle Handle of dataset
 * \param field_name Field name
 * \param[out] out_len Used to set result length
 * \param[out] out_ptr Pointer to the result
 * \param[out] out_type Type of result pointer, can be C_API_DTYPE_INT8, C_API_DTYPE_INT32, C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetGetField(DatasetHandle handle,
                                           const char* field_name,
                                           int* out_len,
                                           const void** out_ptr,
                                           int* out_type);

/*!
 * \fn LGBM_DatasetUpdateParam
 * \brief Update parameters for a dataset.
 * \param handle Handle of dataset
 * \param parameters Parameters
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetUpdateParam(DatasetHandle handle,
                                              const char* parameters);

/*!
 * \fn LGBM_DatasetGetNumData
 * \brief Get number of data points.
 * \param handle Handle of dataset
 * \param[out] out The address to hold number of data points
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetGetNumData(DatasetHandle handle,
                                             int* out);

/*!
 * \fn LGBM_DatasetGetNumFeature
 * \brief Get number of features.
 * \param handle Handle of dataset
 * \param[out] out The address to hold number of features
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetGetNumFeature(DatasetHandle handle,
                                                int* out);

/*!
 * \fn LGBM_DatasetAddFeaturesFrom
 * \brief Add features from source to target.
 * \param target The handle of the dataset to add features to
 * \param source The handle of the dataset to take features from
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetAddFeaturesFrom(DatasetHandle target,
                                                  DatasetHandle source);

// --- start Booster interfaces

/*!
 * \fn LGBM_BoosterCreate
 * \brief Create a new boosting learner.
 * \param train_data Training dataset
 * \param parameters Parameters in format: 'key1=value1 key2=value2'
 * \param[out] out Handle of created booster
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterCreate(const DatasetHandle train_data,
                                         const char* parameters,
                                         BoosterHandle* out);

/*!
 * \fn LGBM_BoosterCreateFromModelfile
 * \brief Load an existing booster from model file.
 * \param filename Filename of model
 * \param[out] out_num_iterations Number of iterations of this booster
 * \param[out] out Handle of created booster
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterCreateFromModelfile(const char* filename,
                                                      int* out_num_iterations,
                                                      BoosterHandle* out);

/*!
 * \fn LGBM_BoosterLoadModelFromString
 * \brief Load an existing booster from string.
 * \param model_str Model string
 * \param[out] out_num_iterations Number of iterations of this booster
 * \param[out] out Handle of created booster
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterLoadModelFromString(const char* model_str,
                                                      int* out_num_iterations,
                                                      BoosterHandle* out);

/*!
 * \fn LGBM_BoosterFree
 * \brief Free space for booster.
 * \param handle Handle of booster to be freed
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterFree(BoosterHandle handle);

/*!
 * \fn LGBM_BoosterShuffleModels
 * \brief Shuffle models.
 * \param handle Handle of booster
 * \param start_iter The first iteration that will be shuffled
 * \param end_iter The last iteration that will be shuffled
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterShuffleModels(BoosterHandle handle,
                                                int start_iter,
                                                int end_iter);

/*!
 * \fn LGBM_BoosterMerge
 * \brief Merge model from other_handle into handle.
 * \param handle Handle of booster, will merge another booster into this one
 * \param other_handle Other handle of booster
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterMerge(BoosterHandle handle,
                                        BoosterHandle other_handle);

/*!
 * \fn LGBM_BoosterAddValidData
 * \brief Add new validation data to booster.
 * \param handle Handle of booster
 * \param valid_data Validation dataset
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterAddValidData(BoosterHandle handle,
                                               const DatasetHandle valid_data);

/*!
 * \fn LGBM_BoosterResetTrainingData
 * \brief Reset training data for booster.
 * \param handle Handle of booster
 * \param train_data Training dataset
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterResetTrainingData(BoosterHandle handle,
                                                    const DatasetHandle train_data);

/*!
 * \fn LGBM_BoosterResetParameter
 * \brief Reset config for booster.
 * \param handle Handle of booster
 * \param parameters Parameters in format: 'key1=value1 key2=value2'
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterResetParameter(BoosterHandle handle,
                                                 const char* parameters);

/*!
 * \fn LGBM_BoosterGetNumClasses
 * \brief Get number of classes.
 * \param handle Handle of booster
 * \param[out] out_len Number of classes
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterGetNumClasses(BoosterHandle handle,
                                                int* out_len);

/*!
 * \fn LGBM_BoosterUpdateOneIter
 * \brief Update the model for one iteration.
 * \param handle Handle of booster
 * \param param[out] is_finished 1 means the update was successfully finished (cannot split any more), 0 indicates failure
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterUpdateOneIter(BoosterHandle handle,
                                                int* is_finished);

/*!
 * \fn LGBM_BoosterRefit
 * \brief Refit the tree model using the new data (online learning).
 * \param handle Handle of booster
 * \param leaf_preds Pointer to predicted leaf indices
 * \param nrow Number of rows of leaf_preds
 * \param ncol Number of columns of leaf_preds
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterRefit(BoosterHandle handle,
                                        const int32_t* leaf_preds,
                                        int32_t nrow,
                                        int32_t ncol);

/*!
 * \fn LGBM_BoosterUpdateOneIterCustom
 * \brief Update the model by specifying gradient and Hessian directly
 *        (this can be used to support customized loss functions).
 * \param handle Handle of booster
 * \param grad The first order derivative (gradient) statistics
 * \param hess The second order derivative (Hessian) statistics
 * \param param[out] is_finished 1 means the update was successfully finished (cannot split any more), 0 indicates failure
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterUpdateOneIterCustom(BoosterHandle handle,
                                                      const float* grad,
                                                      const float* hess,
                                                      int* is_finished);

/*!
 * \fn LGBM_BoosterRollbackOneIter
 * \brief Rollback one iteration.
 * \param handle Handle of booster
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterRollbackOneIter(BoosterHandle handle);

/*!
 * \fn LGBM_BoosterGetCurrentIteration
 * \brief Get index of the current boosting iteration.
 * \param handle Handle of booster
 * \param[out] out_iteration Index of the current boosting iteration
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterGetCurrentIteration(BoosterHandle handle,
                                                      int* out_iteration);

/*!
 * \fn LGBM_BoosterNumModelPerIteration
 * \brief Get number of trees per iteration.
 * \param handle Handle of booster
 * \param[out] out_tree_per_iteration Number of trees per iteration
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterNumModelPerIteration(BoosterHandle handle,
                                                       int* out_tree_per_iteration);

/*!
 * \fn LGBM_BoosterNumberOfTotalModel
 * \brief Get number of weak sub-models.
 * \param handle Handle of booster
 * \param[out] out_models Number of weak sub-models
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterNumberOfTotalModel(BoosterHandle handle,
                                                     int* out_models);

/*!
 * \fn LGBM_BoosterGetEvalCounts
 * \brief Get number of evaluation datasets.
 * \param handle Handle of booster
 * \param[out] out_len Total number of evaluation datasets
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterGetEvalCounts(BoosterHandle handle,
                                                int* out_len);

/*!
 * \fn LGBM_BoosterGetEvalNames
 * \brief Get names of evaluation datasets.
 * \param handle Handle of booster
 * \param[out] out_len Total number of evaluation datasets
 * \param[out] out_strs Names of evaluation datasets, should pre-allocate memory
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterGetEvalNames(BoosterHandle handle,
                                               int* out_len,
                                               char** out_strs);

/*!
 * \fn LGBM_BoosterGetFeatureNames
 * \brief Get names of features.
 * \param handle Handle of booster
 * \param[out] out_len Total number of features
 * \param[out] out_strs Names of features, should pre-allocate memory
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterGetFeatureNames(BoosterHandle handle,
                                                  int* out_len,
                                                  char** out_strs);

/*!
 * \fn LGBM_BoosterGetNumFeature
 * \brief Get number of features.
 * \param handle Handle of booster
 * \param[out] out_len Total number of features
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterGetNumFeature(BoosterHandle handle,
                                                int* out_len);

/*!
 * \fn LGBM_BoosterGetEval
 * \brief Get evaluation for training data and validation data.
 *        Note: 1. You should call LGBM_BoosterGetEvalNames first to get the names of evaluation datasets.
 *              2. You should pre-allocate memory for out_results, you can get its length by LGBM_BoosterGetEvalCounts.
 * \param handle Handle of booster
 * \param data_idx Index of data, 0: training data, 1: 1st validation data, 2: 2nd validation data and so on
 * \param[out] out_len Length of output result
 * \param[out] out_result Array with evaluation results
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterGetEval(BoosterHandle handle,
                                          int data_idx,
                                          int* out_len,
                                          double* out_results);

/*!
 * \fn LGBM_BoosterGetNumPredict
 * \brief Get number of predictions for training data and validation data.
 *        This can be used to support customized evaluation functions.
 * \param handle Handle of booster
 * \param data_idx Index of data, 0: training data, 1: 1st validation data, 2: 2nd validation data and so on
 * \param[out] out_len Number of predictions
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterGetNumPredict(BoosterHandle handle,
                                                int data_idx,
                                                int64_t* out_len);

/*!
 * \fn LGBM_BoosterGetPredict
 * \brief Get prediction for training data and validation data.
 *        Note: You should pre-allocate memory for out_result, its length is equal to num_class * num_data.
 * \param handle Handle of booster
 * \param data_idx Index of data, 0: training data, 1: 1st validation data, 2: 2nd validation data and so on
 * \param[out] out_len Length of output result
 * \param[out] out_result Pointer to array with predictions
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterGetPredict(BoosterHandle handle,
                                             int data_idx,
                                             int64_t* out_len,
                                             double* out_result);

/*!
 * \fn LGBM_BoosterPredictForFile
 * \brief Make prediction for file.
 * \param handle Handle of booster
 * \param data_filename Filename of file with data
 * \param data_has_header Whether file has header or not
 * \param predict_type What should be predicted
 *          C_API_PREDICT_NORMAL: normal prediction, with transform (if needed)
 *          C_API_PREDICT_RAW_SCORE: raw score
 *          C_API_PREDICT_LEAF_INDEX: leaf index
 *          C_API_PREDICT_CONTRIB: feature contributions (SHAP values)
 * \param num_iteration Number of iterations for prediction, <= 0 means no limit
 * \param parameter Other parameters for prediction, e.g. early stopping for prediction.
 * \param result_filename Filename of result file in which predictions will be written
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterPredictForFile(BoosterHandle handle,
                                                 const char* data_filename,
                                                 int data_has_header,
                                                 int predict_type,
                                                 int num_iteration,
                                                 const char* parameter,
                                                 const char* result_filename);

/*!
 * \fn LGBM_BoosterCalcNumPredict
 * \brief Get number of predictions.
 * \param handle Handle of booster
 * \param num_row Number of rows
 * \param predict_type What should be predicted
 *          C_API_PREDICT_NORMAL: normal prediction, with transform (if needed)
 *          C_API_PREDICT_RAW_SCORE: raw score
 *          C_API_PREDICT_LEAF_INDEX: leaf index
 *          C_API_PREDICT_CONTRIB: feature contributions (SHAP values)
 * \param num_iteration Number of iterations for prediction, <= 0 means no limit
 * \param[out] out_len Length of prediction
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterCalcNumPredict(BoosterHandle handle,
                                                 int num_row,
                                                 int predict_type,
                                                 int num_iteration,
                                                 int64_t* out_len);

/*!
 * \fn LGBM_BoosterPredictForCSR
 * \brief Make prediction for a new dataset in CSR format.
 *        Note: You should pre-allocate memory for out_result:
 *              for normal and raw score, its length is equal to num_class * num_data;
 *              for leaf index, its length is equal to num_class * num_data * num_iteration;
 *              for feature contributions, its length is equal to num_class * num_data * (num_feature + 1).
 * \param handle Handle of booster
 * \param indptr Pointer to row headers
 * \param indptr_type Type of indptr, can be C_API_DTYPE_INT32 or C_API_DTYPE_INT64
 * \param indices Pointer to column indices
 * \param data Pointer to the data space
 * \param data_type Type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64
 * \param nindptr Number of rows in the matrix + 1
 * \param nelem Number of nonzero elements in the matrix
 * \param num_col Number of columns; when it's set to 0, then guess from data
 * \param predict_type What should be predicted
 *          C_API_PREDICT_NORMAL: normal prediction, with transform (if needed)
 *          C_API_PREDICT_RAW_SCORE: raw score
 *          C_API_PREDICT_LEAF_INDEX: leaf index
 *          C_API_PREDICT_CONTRIB: feature contributions (SHAP values)
 * \param num_iteration Number of iterations for prediction, <= 0 means no limit
 * \param parameter Other parameters for prediction, e.g. early stopping for prediction
 * \param[out] out_len Length of output result
 * \param[out] out_result Pointer to array with predictions
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterPredictForCSR(BoosterHandle handle,
                                                const void* indptr,
                                                int indptr_type,
                                                const int32_t* indices,
                                                const void* data,
                                                int data_type,
                                                int64_t nindptr,
                                                int64_t nelem,
                                                int64_t num_col,
                                                int predict_type,
                                                int num_iteration,
                                                const char* parameter,
                                                int64_t* out_len,
                                                double* out_result);

/*!
 * \fn LGBM_BoosterPredictForCSRSingleRow
 * \brief Make prediction for a new dataset in CSR format. This method re-uses the internal predictor structure 
 *        from previous calls and is optimized for single row invocation.
 *        Note: You should pre-allocate memory for out_result:
 *              for normal and raw score, its length is equal to num_class * num_data;
 *              for leaf index, its length is equal to num_class * num_data * num_iteration;
 *              for feature contributions, its length is equal to num_class * num_data * (num_feature + 1).
 * \param handle Handle of booster
 * \param indptr Pointer to row headers
 * \param indptr_type Type of indptr, can be C_API_DTYPE_INT32 or C_API_DTYPE_INT64
 * \param indices Pointer to column indices
 * \param data Pointer to the data space
 * \param data_type Type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64
 * \param nindptr Number of rows in the matrix + 1
 * \param nelem Number of nonzero elements in the matrix
 * \param num_col Number of columns; when it's set to 0, then guess from data
 * \param predict_type What should be predicted
 *          C_API_PREDICT_NORMAL: normal prediction, with transform (if needed)
 *          C_API_PREDICT_RAW_SCORE: raw score
 *          C_API_PREDICT_LEAF_INDEX: leaf index
 *          C_API_PREDICT_CONTRIB: feature contributions (SHAP values)
 * \param num_iteration Number of iterations for prediction, <= 0 means no limit
 * \param parameter Other parameters for prediction, e.g. early stopping for prediction.
 * \param[out] out_len Length of output result
 * \param[out] out_result Pointer to array with predictions
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterPredictForCSRSingleRow(BoosterHandle handle,
                                                         const void* indptr,
                                                         int indptr_type,
                                                         const int32_t* indices,
                                                         const void* data,
                                                         int data_type,
                                                         int64_t nindptr,
                                                         int64_t nelem,
                                                         int64_t num_col,
                                                         int predict_type,
                                                         int num_iteration,
                                                         const char* parameter,
                                                         int64_t* out_len,
                                                         double* out_result);


/*!
 * \fn LGBM_BoosterPredictForCSC
 * \brief Make prediction for a new dataset in CSC format.
 *        Note: You should pre-allocate memory for out_result:
 *              for normal and raw score, its length is equal to num_class * num_data;
 *              for leaf index, its length is equal to num_class * num_data * num_iteration;
 *              for feature contributions, its length is equal to num_class * num_data * (num_feature + 1).
 * \param handle Handle of booster
 * \param col_ptr Pointer to column headers
 * \param col_ptr_type Type of col_ptr, can be C_API_DTYPE_INT32 or C_API_DTYPE_INT64
 * \param indices Pointer to row indices
 * \param data Pointer to the data space
 * \param data_type Type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64
 * \param ncol_ptr Number of columns in the matrix + 1
 * \param nelem Number of nonzero elements in the matrix
 * \param num_row Number of rows
 * \param predict_type What should be predicted
 *          C_API_PREDICT_NORMAL: normal prediction, with transform (if needed)
 *          C_API_PREDICT_RAW_SCORE: raw score
 *          C_API_PREDICT_LEAF_INDEX: leaf index
 *          C_API_PREDICT_CONTRIB: feature contributions (SHAP values)
 * \param num_iteration Number of iteration for prediction, <= 0 means no limit
 * \param parameter Other parameters for prediction, e.g. early stopping for prediction.
 * \param[out] out_len Length of output result
 * \param[out] out_result Pointer to array with predictions
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterPredictForCSC(BoosterHandle handle,
                                                const void* col_ptr,
                                                int col_ptr_type,
                                                const int32_t* indices,
                                                const void* data,
                                                int data_type,
                                                int64_t ncol_ptr,
                                                int64_t nelem,
                                                int64_t num_row,
                                                int predict_type,
                                                int num_iteration,
                                                const char* parameter,
                                                int64_t* out_len,
                                                double* out_result);

/*!
 * \fn LGBM_BoosterPredictForMat
 * \brief Make prediction for a new dataset.
 *        Note: You should pre-allocate memory for out_result:
 *              for normal and raw score, its length is equal to num_class * num_data;
 *              for leaf index, its length is equal to num_class * num_data * num_iteration;
 *              for feature contributions, its length is equal to num_class * num_data * (num_feature + 1).
 * \param handle Handle of booster
 * \param data Pointer to the data space
 * \param data_type Type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64
 * \param nrow Number of rows
 * \param ncol Number of columns
 * \param is_row_major 1 for row-major, 0 for column-major
 * \param predict_type What should be predicted
 *          C_API_PREDICT_NORMAL: normal prediction, with transform (if needed)
 *          C_API_PREDICT_RAW_SCORE: raw score
 *          C_API_PREDICT_LEAF_INDEX: leaf index
 *          C_API_PREDICT_CONTRIB: feature contributions (SHAP values)
 * \param num_iteration Number of iteration for prediction, <= 0 means no limit
 * \param parameter Other parameters for prediction, e.g. early stopping for prediction.
 * \param[out] out_len Length of output result
 * \param[out] out_result Pointer to array with predictions
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterPredictForMat(BoosterHandle handle,
                                                const void* data,
                                                int data_type,
                                                int32_t nrow,
                                                int32_t ncol,
                                                int is_row_major,
                                                int predict_type,
                                                int num_iteration,
                                                const char* parameter,
                                                int64_t* out_len,
                                                double* out_result);

/*!
 * \fn LGBM_BoosterPredictForMatSingleRow
 * \brief Make prediction for an new dataset. This method re-uses the internal predictor structure 
 *        from previous calls and is optimized for single row invocation.
 *        Note: You should pre-allocate memory for out_result:
 *              for normal and raw score, its length is equal to num_class * num_data;
 *              for leaf index, its length is equal to num_class * num_data * num_iteration;
 *              for feature contributions, its length is equal to num_class * num_data * (num_feature + 1).
 * \param handle Handle of booster
 * \param data Pointer to the data space
 * \param data_type Type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64
 * \param ncol Number columns
 * \param is_row_major 1 for row major, 0 for column major
 * \param predict_type What should be predicted
 *          C_API_PREDICT_NORMAL: normal prediction, with transform (if needed)
 *          C_API_PREDICT_RAW_SCORE: raw score
 *          C_API_PREDICT_LEAF_INDEX: leaf index
 *          C_API_PREDICT_CONTRIB: feature contributions (SHAP values)
 * \param num_iteration Number of iteration for prediction, <= 0 means no limit
 * \param parameter Other parameters for prediction, e.g. early stopping for prediction.
 * \param[out] out_len Length of output result
 * \param[out] out_result Pointer to array with predictions
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterPredictForMatSingleRow(BoosterHandle handle,
                                                const void* data,
                                                int data_type,
                                                int ncol,
                                                int is_row_major,
                                                int predict_type,
                                                int num_iteration,
                                                const char* parameter,
                                                int64_t* out_len,
                                                double* out_result);

/*!
 * \fn LGBM_BoosterPredictForMats
 * \brief Make prediction for a new dataset presented in a form of array of pointers to rows.
 *        Note: You should pre-allocate memory for out_result:
 *              for normal and raw score, its length is equal to num_class * num_data;
 *              for leaf index, its length is equal to num_class * num_data * num_iteration;
 *              for feature contributions, its length is equal to num_class * num_data * (num_feature + 1).
 * \param handle Handle of booster
 * \param data Pointer to the data space
 * \param data_type Type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64
 * \param nrow Number of rows
 * \param ncol Number columns
 * \param predict_type What should be predicted
 *          C_API_PREDICT_NORMAL: normal prediction, with transform (if needed)
 *          C_API_PREDICT_RAW_SCORE: raw score
 *          C_API_PREDICT_LEAF_INDEX: leaf index
 *          C_API_PREDICT_CONTRIB: feature contributions (SHAP values)
 * \param num_iteration Number of iteration for prediction, <= 0 means no limit
 * \param parameter Other parameters for prediction, e.g. early stopping for prediction.
 * \param[out] out_len Length of output result
 * \param[out] out_result Pointer to array with predictions
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterPredictForMats(BoosterHandle handle,
                                                 const void** data,
                                                 int data_type,
                                                 int32_t nrow,
                                                 int32_t ncol,
                                                 int predict_type,
                                                 int num_iteration,
                                                 const char* parameter,
                                                 int64_t* out_len,
                                                 double* out_result);

/*!
 * \fn LGBM_BoosterSaveModel
 * \brief Save model into file.
 * \param handle Handle of booster
 * \param start_iteration Start index of the iteration that should be saved
 * \param num_iteration Index of the iteration that should be saved, <= 0 means save all
 * \param filename File name
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterSaveModel(BoosterHandle handle,
                                            int start_iteration,
                                            int num_iteration,
                                            const char* filename);

/*!
 * \fn LGBM_BoosterSaveModelToString
 * \brief Save model to string.
 * \param handle Handle of booster
 * \param start_iteration Start index of the iteration that should be saved
 * \param num_iteration Index of the iteration that should be saved, <= 0 means save all
 * \param buffer_len String buffer length, if buffer_len < out_len, you should re-allocate buffer
 * \param[out] out_len Actual output length
 * \param[out] out_str String of model, should pre-allocate memory
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterSaveModelToString(BoosterHandle handle,
                                                    int start_iteration,
                                                    int num_iteration,
                                                    int64_t buffer_len,
                                                    int64_t* out_len,
                                                    char* out_str);

/*!
 * \fn LGBM_BoosterDumpModel
 * \brief Dump model to JSON.
 * \param handle Handle of booster
 * \param start_iteration Start index of the iteration that should be dumped
 * \param num_iteration Index of the iteration that should be dumped, <= 0 means save all
 * \param buffer_len String buffer length, if buffer_len < out_len, you should re-allocate buffer
 * \param[out] out_len Actual output length
 * \param[out] out_str JSON format string of model, should pre-allocate memory
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterDumpModel(BoosterHandle handle,
                                            int start_iteration,
                                            int num_iteration,
                                            int64_t buffer_len,
                                            int64_t* out_len,
                                            char* out_str);

/*!
 * \fn LGBM_BoosterGetLeafValue
 * \brief Get leaf value.
 * \param handle Handle of booster
 * \param tree_idx Index of tree
 * \param leaf_idx Index of leaf
 * \param[out] out_val Output result from the specified leaf
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterGetLeafValue(BoosterHandle handle,
                                               int tree_idx,
                                               int leaf_idx,
                                               double* out_val);

/*!
 * \fn LGBM_BoosterSetLeafValue
 * \brief Set leaf value.
 * \param handle Handle of booster
 * \param tree_idx Index of tree
 * \param leaf_idx Index of leaf
 * \param val Leaf value
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterSetLeafValue(BoosterHandle handle,
                                               int tree_idx,
                                               int leaf_idx,
                                               double val);

/*!
 * \fn LGBM_BoosterFeatureImportance
 * \brief Get model feature importance.
 * \param handle Handle of booster
 * \param num_iteration Number of iterations for which feature importance is calculated, <= 0 means use all
 * \param importance_type Method of importance calculation:
 *          0 for split, result contains numbers of times the feature is used in a model
 *          1 for gain, result contains total gains of splits which use the feature
 * \param[out] out_results Result array with feature importance
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterFeatureImportance(BoosterHandle handle,
                                                    int num_iteration,
                                                    int importance_type,
                                                    double* out_results);

/*!
 * \fn LGBM_NetworkInit
 * \brief Initialize the network.
 * \param machines List of machines in format 'ip1:port1,ip2:port2'
 * \param local_listen_port TCP listen port for local machines
 * \param listen_time_out Socket time-out in minutes
 * \param num_machines Total number of machines
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_NetworkInit(const char* machines,
                                       int local_listen_port,
                                       int listen_time_out,
                                       int num_machines);

/*!
 * \fn LGBM_NetworkFree
 * \brief Finalize the network.
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_NetworkFree();


/*!
 * \fn LGBM_NetworkInitWithFunctions
 * \brief Initialize the network with external collective functions.
 * \param num_machines Total number of machines
 * \param rank Rank of local machine
 * \param reduce_scatter_ext_fun The external reduce-scatter function
 * \param allgather_ext_fun The external allgather function
 * \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_NetworkInitWithFunctions(int num_machines, int rank,
                                                    void* reduce_scatter_ext_fun,
                                                    void* allgather_ext_fun);


#if defined(_MSC_VER)
#define THREAD_LOCAL __declspec(thread)
#else
#define THREAD_LOCAL thread_local
#endif
// exception handle and error msg
static char* LastErrorMsg() { static THREAD_LOCAL char err_msg[512] = "Everything is fine"; return err_msg; }

#pragma warning(disable : 4996)
inline void LGBM_SetLastError(const char* msg) {
  std::strcpy(LastErrorMsg(), msg);
}

#endif  // LIGHTGBM_C_API_H_
