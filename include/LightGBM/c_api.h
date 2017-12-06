#ifndef LIGHTGBM_C_API_H_
#define LIGHTGBM_C_API_H_

#include <LightGBM/meta.h>

#include <cstdint>
#include <exception>
#include <stdexcept>
#include <cstring>
#include <string>

/*!
* To avoid type conversion on large data, most of our expose interface support both for float_32 and float_64.
* Except following:
* 1. gradients and hessians.
* 2. Get current score for training data and validation
* The reason is because they are called frequently, the type-conversion on them maybe time cost.
*/

#include <LightGBM/export.h>

typedef void* DatasetHandle;
typedef void* BoosterHandle;

#define C_API_DTYPE_FLOAT32 (0)
#define C_API_DTYPE_FLOAT64 (1)
#define C_API_DTYPE_INT32   (2)
#define C_API_DTYPE_INT64   (3)

#define C_API_PREDICT_NORMAL     (0)
#define C_API_PREDICT_RAW_SCORE  (1)
#define C_API_PREDICT_LEAF_INDEX (2)
#define C_API_PREDICT_CONTRIB    (3)

/*!
* \brief get string message of the last error
*  all function in this file will return 0 when succeed
*  and -1 when an error occured,
* \return const char* error inforomation
*/
LIGHTGBM_C_EXPORT const char* LGBM_GetLastError();

// --- start Dataset interface

/*!
* \brief load data set from file like the command_line LightGBM do
* \param filename the name of the file
* \param parameters additional parameters
* \param reference used to align bin mapper with other dataset, nullptr means don't used
* \param out a loaded dataset
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetCreateFromFile(const char* filename,
                                                 const char* parameters,
                                                 const DatasetHandle reference,
                                                 DatasetHandle* out);

/*!
* \brief create a empty dataset by sampling data.
* \param sample_data sampled data, grouped by the column.
* \param sample_indices indices of sampled data.
* \param ncol number columns
* \param num_per_col Size of each sampling column
* \param num_sample_row Number of sampled rows
* \param num_total_row number of total rows
* \param parameters additional parameters
* \param out created dataset
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
* \brief create a empty dataset by reference Dataset
* \param reference used to align bin mapper
* \param num_total_row number of total rows
* \param out created dataset
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetCreateByReference(const DatasetHandle reference,
                                                    int64_t num_total_row,
                                                    DatasetHandle* out);

/*!
* \brief push data to existing dataset, if nrow + start_row == num_total_row, will call dataset->FinishLoad
* \param dataset handle of dataset
* \param data pointer to the data space
* \param data_type type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64
* \param nrow number of rows
* \param ncol number columns
* \param start_row row start index
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetPushRows(DatasetHandle dataset,
                                           const void* data,
                                           int data_type,
                                           int32_t nrow,
                                           int32_t ncol,
                                           int32_t start_row);

/*!
* \brief push data to existing dataset, if nrow + start_row == num_total_row, will call dataset->FinishLoad
* \param dataset handle of dataset
* \param indptr pointer to row headers
* \param indptr_type type of indptr, can be C_API_DTYPE_INT32 or C_API_DTYPE_INT64
* \param indices findex
* \param data fvalue
* \param data_type type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64
* \param nindptr number of rows in the matrix + 1
* \param nelem number of nonzero elements in the matrix
* \param num_col number of columns
* \param start_row row start index
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
* \brief create a dataset from CSR format
* \param indptr pointer to row headers
* \param indptr_type type of indptr, can be C_API_DTYPE_INT32 or C_API_DTYPE_INT64
* \param indices findex
* \param data fvalue
* \param data_type type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64
* \param nindptr number of rows in the matrix + 1
* \param nelem number of nonzero elements in the matrix
* \param num_col number of columns
* \param parameters additional parameters
* \param reference used to align bin mapper with other dataset, nullptr means don't used
* \param out created dataset
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
* \brief create a dataset from CSC format
* \param col_ptr pointer to col headers
* \param col_ptr_type type of col_ptr, can be C_API_DTYPE_INT32 or C_API_DTYPE_INT64
* \param indices findex
* \param data fvalue
* \param data_type type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64
* \param ncol_ptr number of cols in the matrix + 1
* \param nelem number of nonzero elements in the matrix
* \param num_row number of rows
* \param parameters additional parameters
* \param reference used to align bin mapper with other dataset, nullptr means don't used
* \param out created dataset
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
* \brief create dataset from dense matrix
* \param data pointer to the data space
* \param data_type type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64
* \param nrow number of rows
* \param ncol number columns
* \param is_row_major 1 for row major, 0 for column major
* \param parameters additional parameters
* \param reference used to align bin mapper with other dataset, nullptr means don't used
* \param out created dataset
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
* \brief Create subset of a data
* \param handle handle of full dataset
* \param used_row_indices Indices used in subset
* \param num_used_row_indices len of used_row_indices
* \param parameters additional parameters
* \param out subset of data
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetGetSubset(
  const DatasetHandle handle,
  const int32_t* used_row_indices,
  int32_t num_used_row_indices,
  const char* parameters,
  DatasetHandle* out);

/*!
* \brief save feature names to Dataset
* \param handle handle
* \param feature_names feature names
* \param num_feature_names number of feature names
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetSetFeatureNames(
  DatasetHandle handle,
  const char** feature_names,
  int num_feature_names);


/*!
* \brief get feature names of Dataset
* \param handle handle
* \param feature_names feature names, should pre-allocate memory
* \param num_feature_names number of feature names
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetGetFeatureNames(
  DatasetHandle handle,
  char** feature_names,
  int* num_feature_names);


/*!
* \brief free space for dataset
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetFree(DatasetHandle handle);

/*!
* \brief save dateset to binary file
* \param handle a instance of dataset
* \param filename file name
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetSaveBinary(DatasetHandle handle,
                                             const char* filename);

/*!
* \brief set vector to a content in info
*        Note: group and group only work for C_API_DTYPE_INT32
*              label and weight only work for C_API_DTYPE_FLOAT32
* \param handle a instance of dataset
* \param field_name field name, can be label, weight, group, group_id
* \param field_data pointer to vector
* \param num_element number of element in field_data
* \param type C_API_DTYPE_FLOAT32 or C_API_DTYPE_INT32
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetSetField(DatasetHandle handle,
                                           const char* field_name,
                                           const void* field_data,
                                           int num_element,
                                           int type);

/*!
* \brief get info vector from dataset
* \param handle a instance of data matrix
* \param field_name field name
* \param out_len used to set result length
* \param out_ptr pointer to the result
* \param out_type  C_API_DTYPE_FLOAT32 or C_API_DTYPE_INT32
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetGetField(DatasetHandle handle,
                                           const char* field_name,
                                           int* out_len,
                                           const void** out_ptr,
                                           int* out_type);

/*!
* \brief get number of data.
* \param handle the handle to the dataset
* \param out The address to hold number of data
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetGetNumData(DatasetHandle handle,
                                             int* out);

/*!
* \brief get number of features
* \param handle the handle to the dataset
* \param out The output of number of features
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetGetNumFeature(DatasetHandle handle,
                                                int* out);

// --- start Booster interfaces

/*!
* \brief create an new boosting learner
* \param train_data training data set
* \param parameters format: 'key1=value1 key2=value2'
* \prama out handle of created Booster
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterCreate(const DatasetHandle train_data,
                                         const char* parameters,
                                         BoosterHandle* out);

/*!
* \brief load an existing boosting from model file
* \param filename filename of model
* \param out_num_iterations number of iterations of this booster
* \param out handle of created Booster
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterCreateFromModelfile(
  const char* filename,
  int* out_num_iterations,
  BoosterHandle* out);

/*!
* \brief load an existing boosting from string
* \param model_str model string
* \param out_num_iterations number of iterations of this booster
* \param out handle of created Booster
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterLoadModelFromString(
  const char* model_str,
  int* out_num_iterations,
  BoosterHandle* out);

/*!
* \brief free obj in handle
* \param handle handle to be freed
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterFree(BoosterHandle handle);

/*!
* \brief Merge model in two booster to first handle
* \param handle handle, will merge other handle to this
* \param other_handle
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterMerge(BoosterHandle handle,
                                        BoosterHandle other_handle);

/*!
* \brief Add new validation to booster
* \param handle handle
* \param valid_data validation data set
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterAddValidData(BoosterHandle handle,
                                               const DatasetHandle valid_data);

/*!
* \brief Reset training data for booster
* \param handle handle
* \param train_data training data set
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterResetTrainingData(BoosterHandle handle,
                                                    const DatasetHandle train_data);

/*!
* \brief Reset config for current booster
* \param handle handle
* \param parameters format: 'key1=value1 key2=value2'
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterResetParameter(BoosterHandle handle, const char* parameters);

/*!
* \brief Get number of class
* \param handle handle
* \param out_len number of class
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterGetNumClasses(BoosterHandle handle, int* out_len);

/*!
* \brief update the model in one round
* \param handle handle
* \param is_finished 1 means finised(cannot split any more)
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterUpdateOneIter(BoosterHandle handle, int* is_finished);

/*!
* \brief update the model, by directly specify gradient and second order gradient,
*       this can be used to support customized loss function
* \param handle handle
* \param grad gradient statistics
* \param hess second order gradient statistics
* \param is_finished 1 means finised(cannot split any more)
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterUpdateOneIterCustom(BoosterHandle handle,
                                                      const float* grad,
                                                      const float* hess,
                                                      int* is_finished);

/*!
* \brief Rollback one iteration
* \param handle handle
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterRollbackOneIter(BoosterHandle handle);

/*!
* \brief Get iteration of current boosting rounds
* \param out_iteration iteration of boosting rounds
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterGetCurrentIteration(BoosterHandle handle, int* out_iteration);

/*!
* \brief Get number of eval
* \param out_len total number of eval results
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterGetEvalCounts(BoosterHandle handle, int* out_len);

/*!
* \brief Get name of eval
* \param out_len total number of eval results
* \param out_strs names of eval result, need to pre-allocate memory before call this
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterGetEvalNames(BoosterHandle handle, int* out_len, char** out_strs);

/*!
* \brief Get name of features
* \param out_len total number of features
* \param out_strs names of features, need to pre-allocate memory before call this
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterGetFeatureNames(BoosterHandle handle, int* out_len, char** out_strs);

/*!
* \brief Get number of features
* \param out_len total number of features
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterGetNumFeature(BoosterHandle handle, int* out_len);

/*!
* \brief get evaluation for training data and validation data
Note: 1. you should call LGBM_BoosterGetEvalNames first to get the name of evaluation results
2. should pre-allocate memory for out_results, you can get its length by LGBM_BoosterGetEvalCounts
* \param handle handle
* \param data_idx 0:training data, 1: 1st valid data, 2:2nd valid data ...
* \param out_len len of output result
* \param out_result float arrary contains result
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterGetEval(BoosterHandle handle,
                                          int data_idx,
                                          int* out_len,
                                          double* out_results);

/*!
* \brief Get number of predict for inner dataset
this can be used to support customized eval function
Note:  should pre-allocate memory for out_result, its length is equal to num_class * num_data
* \param handle handle
* \param data_idx 0:training data, 1: 1st valid data, 2:2nd valid data ...
* \param out_len len of output result
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterGetNumPredict(BoosterHandle handle,
                                                int data_idx,
                                                int64_t* out_len);

/*!
* \brief Get prediction for training data and validation data
this can be used to support customized eval function
Note:  should pre-allocate memory for out_result, its length is equal to num_class * num_data
* \param handle handle
* \param data_idx 0:training data, 1: 1st valid data, 2:2nd valid data ...
* \param out_len len of output result
* \param out_result used to set a pointer to array, should allocate memory before call this function
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterGetPredict(BoosterHandle handle,
                                             int data_idx,
                                             int64_t* out_len,
                                             double* out_result);

/*!
* \brief make prediction for file
* \param handle handle
* \param data_filename filename of data file
* \param data_has_header data file has header or not
* \param predict_type
*          C_API_PREDICT_NORMAL: normal prediction, with transform (if needed)
*          C_API_PREDICT_RAW_SCORE: raw score
*          C_API_PREDICT_LEAF_INDEX: leaf index
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \param parameter Other parameters for the parameters, e.g. early stopping for prediction.
* \param result_filename filename of result file
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
* \brief Get number of prediction
* \param handle handle
* \param num_row
* \param predict_type
*          C_API_PREDICT_NORMAL: normal prediction, with transform (if needed)
*          C_API_PREDICT_RAW_SCORE: raw score
*          C_API_PREDICT_LEAF_INDEX: leaf index
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \param out_len length of prediction
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterCalcNumPredict(BoosterHandle handle,
                                                 int num_row,
                                                 int predict_type,
                                                 int num_iteration,
                                                 int64_t* out_len);

/*!
* \brief make prediction for an new data set
*        Note:  should pre-allocate memory for out_result,
*               for noraml and raw score: its length is equal to num_class * num_data
*               for leaf index, its length is equal to num_class * num_data * num_iteration
* \param handle handle
* \param indptr pointer to row headers
* \param indptr_type type of indptr, can be C_API_DTYPE_INT32 or C_API_DTYPE_INT64
* \param indices findex
* \param data fvalue
* \param data_type type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64
* \param nindptr number of rows in the matrix + 1
* \param nelem number of nonzero elements in the matrix
* \param num_col number of columns; when it's set to 0, then guess from data
* \param predict_type
*          C_API_PREDICT_NORMAL: normal prediction, with transform (if needed)
*          C_API_PREDICT_RAW_SCORE: raw score
*          C_API_PREDICT_LEAF_INDEX: leaf index
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \param parameter Other parameters for the parameters, e.g. early stopping for prediction.
* \param out_len len of output result
* \param out_result used to set a pointer to array, should allocate memory before call this function
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
* \brief make prediction for an new data set
*        Note:  should pre-allocate memory for out_result,
*               for noraml and raw score: its length is equal to num_class * num_data
*               for leaf index, its length is equal to num_class * num_data * num_iteration
* \param handle handle
* \param col_ptr pointer to col headers
* \param col_ptr_type type of col_ptr, can be C_API_DTYPE_INT32 or C_API_DTYPE_INT64
* \param indices findex
* \param data fvalue
* \param data_type type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64
* \param ncol_ptr number of cols in the matrix + 1
* \param nelem number of nonzero elements in the matrix
* \param num_row number of rows
* \param predict_type
*          C_API_PREDICT_NORMAL: normal prediction, with transform (if needed)
*          C_API_PREDICT_RAW_SCORE: raw score
*          C_API_PREDICT_LEAF_INDEX: leaf index
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \param parameter Other parameters for the parameters, e.g. early stopping for prediction.
* \param out_len len of output result
* \param out_result used to set a pointer to array, should allocate memory before call this function
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
* \brief make prediction for an new data set
*        Note:  should pre-allocate memory for out_result,
*               for noraml and raw score: its length is equal to num_class * num_data
*               for leaf index, its length is equal to num_class * num_data * num_iteration
* \param handle handle
* \param data pointer to the data space
* \param data_type type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64
* \param nrow number of rows
* \param ncol number columns
* \param is_row_major 1 for row major, 0 for column major
* \param predict_type
*          C_API_PREDICT_NORMAL: normal prediction, with transform (if needed)
*          C_API_PREDICT_RAW_SCORE: raw score
*          C_API_PREDICT_LEAF_INDEX: leaf index
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \param parameter Other parameters for the parameters, e.g. early stopping for prediction.
* \param out_len len of output result
* \param out_result used to set a pointer to array, should allocate memory before call this function
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
* \brief save model into file
* \param handle handle
* \param num_iteration, <= 0 means save all
* \param filename file name
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterSaveModel(BoosterHandle handle,
                                            int num_iteration,
                                            const char* filename);

/*!
* \brief save model to string
* \param handle handle
* \param num_iteration, <= 0 means save all
* \param buffer_len string buffer length, if buffer_len < out_len, re-allocate buffer
* \param out_len actual output length
* \param out_str string of model, need to pre-allocate memory before call this
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterSaveModelToString(BoosterHandle handle,
                                                    int num_iteration,
                                                    int64_t buffer_len,
                                                    int64_t* out_len,
                                                    char* out_str);

/*!
* \brief dump model to json
* \param handle handle
* \param num_iteration, <= 0 means save all
* \param buffer_len string buffer length, if buffer_len < out_len, re-allocate buffer
* \param out_len actual output length
* \param out_str json format string of model, need to pre-allocate memory before call this
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterDumpModel(BoosterHandle handle,
                                            int num_iteration,
                                            int64_t buffer_len,
                                            int64_t* out_len,
                                            char* out_str);

/*!
* \brief Get leaf value
* \param handle handle
* \param tree_idx index of tree
* \param leaf_idx index of leaf
* \param out_val out result
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterGetLeafValue(BoosterHandle handle,
                                               int tree_idx,
                                               int leaf_idx,
                                               double* out_val);

/*!
* \brief Set leaf value
* \param handle handle
* \param tree_idx index of tree
* \param leaf_idx index of leaf
* \param val leaf value
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterSetLeafValue(BoosterHandle handle,
                                               int tree_idx,
                                               int leaf_idx,
                                               double val);

/*!
* \brief get model feature importance
* \param handle handle
* \param num_iteration, <= 0 means use all
* \param importance_type: 0 for split, 1 for gain
* \param out_results output value array
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterFeatureImportance(BoosterHandle handle,
                                                    int num_iteration,
                                                    int importance_type,
                                                    double* out_results);

/*!
* \brief Initilize the network
* \param machines represent the nodes, format: ip1:port1,ip2:port2
* \param local_listen_port
* \param listen_time_out
* \param num_machines
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_NetworkInit(const char* machines,
                                       int local_listen_port,
                                       int listen_time_out,
                                       int num_machines);

/*!
* \brief Finalize the network
* \return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_NetworkFree();

LIGHTGBM_C_EXPORT int LGBM_GetFuncions(void* AllreduceFuncPtr,
                                       void* ReduceScatterFuncPtr, 
                                       void* AllgatherFuncPtr, 
                                       int num_machines, 
                                       int rank);

// exception handle and error msg
static char* LastErrorMsg() { static THREAD_LOCAL char err_msg[512] = "Everything is fine"; return err_msg; }

#pragma warning(disable : 4996)
inline void LGBM_SetLastError(const char* msg) {
  std::strcpy(LastErrorMsg(), msg);
}

inline int LGBM_APIHandleException(const std::exception& ex) {
  LGBM_SetLastError(ex.what());
  return -1;
}
inline int LGBM_APIHandleException(const std::string& ex) {
  LGBM_SetLastError(ex.c_str());
  return -1;
}

#define API_BEGIN() try {

#define API_END() } \
catch(std::exception& ex) { return LGBM_APIHandleException(ex); } \
catch(std::string& ex) { return LGBM_APIHandleException(ex); } \
catch(...) { return LGBM_APIHandleException("unknown exception"); } \
return 0;

#endif // LIGHTGBM_C_API_H_
