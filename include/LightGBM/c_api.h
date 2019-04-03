#ifndef LIGHTGBM_C_API_H_
#define LIGHTGBM_C_API_H_

#include <cstdint>
#include <cstring>

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
#define C_API_DTYPE_INT8    (4)

#define C_API_PREDICT_NORMAL     (0)
#define C_API_PREDICT_RAW_SCORE  (1)
#define C_API_PREDICT_LEAF_INDEX (2)
#define C_API_PREDICT_CONTRIB    (3)

/*!
 * @fn LGBM_GetLastError
 * @headerfile <LightGBM/export.h>
 * @brief Get string message of the last error.
 * @signature LIGHTGBM_C_EXPORT const char* LGBM_GetLastError();
 * @return const char* error inforomation
*/
LIGHTGBM_C_EXPORT const char* LGBM_GetLastError();

// --- start Dataset interface

/*!
 * @fn LGBM_DatasetCreateFromFile
 * @brief Load data set from file (like the command_line LightGBM do).
 * @signature LIGHTGBM_C_EXPORT int LGBM_DatasetCreateFromFile(const char* filename,
                                                               const char* parameters,
                                                               const DatasetHandle reference,
                                                               DatasetHandle* out);
 * @param filename The name of the file
 * @param parameters Additional parameters
 * @param reference Used to align bin mapper with other dataset, nullptr means don't used
 * @param[out] out A loaded dataset
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetCreateFromFile(const char* filename,
                                                 const char* parameters,
                                                 const DatasetHandle reference,
                                                 DatasetHandle* out);

/*!
 * @fn LGBM_DatasetCreateFromSampledColumn
 * @brief Create an empty dataset by sampling data.
 * @signature LIGHTGBM_C_EXPORT int LGBM_DatasetCreateFromSampledColumn(double** sample_data,
                                                                        int** sample_indices,
                                                                        int32_t ncol,
                                                                        const int* num_per_col,
                                                                        int32_t num_sample_row,
                                                                        int32_t num_total_row,
                                                                        const char* parameters,
                                                                        DatasetHandle* out);
 * @param sample_indices Indices of sampled data
 * @param ncol Number of columns
 * @param num_per_col Size of each sampling column
 * @param num_sample_row Number of sampled rows
 * @param num_total_row Number of total rows
 * @param parameters Additional parameters
 * @param[out] out Created dataset
 * @return 0 when succeed, -1 when failure happens
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
 * @fn LGBM_DatasetCreateByReference
 * @brief Create an empty dataset by reference Dataset.
 * @signature LIGHTGBM_C_EXPORT int LGBM_DatasetCreateByReference(const DatasetHandle reference,
                                                                  int64_t num_total_row,
                                                                  DatasetHandle* out);
 * @param reference Used to align bin mapper
 * @param num_total_row Number of total rows
 * @param[out] out Created dataset
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetCreateByReference(const DatasetHandle reference,
                                                    int64_t num_total_row,
                                                    DatasetHandle* out);

/*!
 * @fn LGBM_DatasetPushRows
 * @brief Push data to existing dataset, if nrow + start_row == num_total_row, will call dataset->FinishLoad.
 * @signature LIGHTGBM_C_EXPORT int LGBM_DatasetPushRows(DatasetHandle dataset,
                                                         const void* data,
                                                         int data_type,
                                                         int32_t nrow,
                                                         int32_t ncol,
                                                         int32_t start_row);
 * @param dataset Handle of dataset
 * @param data Pointer to the data space
 * @param data_type Type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64)
 * @param nrow Number of rows
 * @param ncol Number columns
 * @param start_row Row start index
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetPushRows(DatasetHandle dataset,
                                           const void* data,
                                           int data_type,
                                           int32_t nrow,
                                           int32_t ncol,
                                           int32_t start_row);

/*!
 * @fn LGBM_DatasetPushRowsByCSR
 * @brief Push data to existing dataset, if nrow + start_row == num_total_row, will call dataset->FinishLoad.
 * @signature LIGHTGBM_C_EXPORT int LGBM_DatasetPushRowsByCSR(DatasetHandle dataset,
                                                              const void* indptr,
                                                              int indptr_type,
                                                              const int32_t* indices,
                                                              const void* data,
                                                              int data_type,
                                                              int64_t nindptr,
                                                              int64_t nelem,
                                                              int64_t num_col,
                                                              int64_t start_row);
 * @param dataset Handle of dataset
 * @param indptr Pointer to row headers
 * @param indptr_type Type of indptr, can be C_API_DTYPE_INT32 or C_API_DTYPE_INT64
 * @param indices Findex
 * @param data Fvalue
 * @param data_type Type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64
 * @param nindptr Number of rows in the matrix + 1
 * @param nelem Number of nonzero elements in the matrix
 * @param num_col Number of columns
 * @param start_row Row start index
 * @return 0 when succeed, -1 when failure happens
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
 * @fn LGBM_DatasetCreateFromCSR
 * @brief Create a dataset from CSR format.
 * @signature LIGHTGBM_C_EXPORT int LGBM_DatasetCreateFromCSR(const void* indptr,
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
 * @param indptr Pointer to row headers
 * @param indptr_type Type of indptr, can be C_API_DTYPE_INT32 or C_API_DTYPE_INT64
 * @param indices Findex
 * @param data Fvalue
 * @param data_type Type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64
 * @param nindptr Number of rows in the matrix + 1
 * @param nelem Number of nonzero elements in the matrix
 * @param num_col Number of columns
 * @param parameters Additional parameters
 * @param reference Used to align bin mapper with other dataset, nullptr means don't used
 * @param[out] out Created dataset
 * @return 0 when succeed, -1 when failure happens
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
 * @fn LGBM_DatasetCreateFromCSRFunc
 * @brief create a dataset from CSR format through callbacks.
 * @signature LIGHTGBM_C_EXPORT int LGBM_DatasetCreateFromCSRFunc(void* get_row_funptr,
                                                                  int num_rows,
                                                                  int64_t num_col,
                                                                  const char* parameters,
                                                                  const DatasetHandle reference,
                                                                  DatasetHandle* out);
 * @param get_row_funptr Pointer to std::function<void(int idx, std::vector<std::pair<int, double>>& ret)
			Called for every row and expected to clear and fill ret
 * @param num_rows Number of rows
 * @param num_col Number of columns
 * @param parameters Additional parameters
 * @param reference Used to align bin mapper with other dataset, nullptr means don't used
 * @param[out] out Created dataset
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetCreateFromCSRFunc(void* get_row_funptr,
                                                int num_rows,
                                                int64_t num_col,
                                                const char* parameters,
                                                const DatasetHandle reference,
                                                DatasetHandle* out);


/*!
 * @fn LGBM_DatasetCreateFromCSC
 * @brief Create a dataset from CSC format.
 * @signature LIGHTGBM_C_EXPORT int LGBM_DatasetCreateFromCSC(const void* col_ptr,
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
 * @param col_ptr Pointer to col headers
 * @param col_ptr_type Type of col_ptr, can be C_API_DTYPE_INT32 or C_API_DTYPE_INT64
 * @param indices Findex
 * @param data Fvalue
 * @param data_type Type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64
 * @param ncol_ptr Number of cols in the matrix + 1
 * @param nelem Number of nonzero elements in the matrix
 * @param num_row Number of rows
 * @param parameters Additional parameters
 * @param reference Used to align bin mapper with other dataset, nullptr means don't used
 * @param[out] out Created dataset
 * @return 0 when succeed, -1 when failure happens
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
 * @fn LGBM_DatasetCreateFromMat
 * @brief Create dataset from dense matrix.
 * @signature LIGHTGBM_C_EXPORT int LGBM_DatasetCreateFromMat(const void* data,
                                                              int data_type,
                                                              int32_t nrow,
                                                              int32_t ncol,
                                                              int is_row_major,
                                                              const char* parameters,
                                                              const DatasetHandle reference,
                                                              DatasetHandle* out);
 * @param data Pointer to the data space
 * @param data_type Type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64
 * @param nrow Number of rows
 * @param ncol Number columns
 * @param is_row_major 1 for row major, 0 for column major
 * @param parameters Additional parameters
 * @param reference Used to align bin mapper with other dataset, nullptr means don't used
 * @param[out] out Created dataset
 * @return 0 when succeed, -1 when failure happens
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
 * @fn LGBM_DatasetCreateFromMats
 * @brief Create dataset from array of dense matrices.
 * @signature LIGHTGBM_C_EXPORT int LGBM_DatasetCreateFromMats(int32_t nmat,
                                                               const void** data,
                                                               int data_type,
                                                               int32_t* nrow,
                                                               int32_t ncol,
                                                               int is_row_major,
                                                               const char* parameters,
                                                               const DatasetHandle reference,
                                                               DatasetHandle* out);
 * @param data Pointer to the data space
 * @param data_type Type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64
 * @param nrow Number of rows
 * @param ncol Number columns
 * @param parameters Additional parameters
 * @param reference Used to align bin mapper with other dataset, nullptr means don't used
 * @param[out] out Created dataset
 * @return 0 when succeed, -1 when failure happens
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
 * @fn LGBM_DatasetGetSubset
 * @brief Create subset of a data.
 * @signature LIGHTGBM_C_EXPORT int LGBM_DatasetGetSubset(const DatasetHandle handle,
                                                          const int32_t* used_row_indices,
                                                          int32_t num_used_row_indices,
                                                          const char* parameters,
                                                          DatasetHandle* out);
 * @param handle Handle of full dataset
 * @param used_row_indices Indices used in subset
 * @param num_used_row_indices Len of used_row_indices
 * @param parameters Additional parameters
 * @param[out] out Subset of data
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetGetSubset(
  const DatasetHandle handle,
  const int32_t* used_row_indices,
  int32_t num_used_row_indices,
  const char* parameters,
  DatasetHandle* out);

/*!
 * @fn LGBM_DatasetSetFeatureNames
 * @brief Save feature names to Dataset.
 * @signature LIGHTGBM_C_EXPORT int LGBM_DatasetSetFeatureNames(DatasetHandle handle,
                                                                const char** feature_names,
                                                                int num_feature_names);
 * @param handle Handle
 * @param feature_names Feature names
 * @param num_feature_names Number of feature names
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetSetFeatureNames(
  DatasetHandle handle,
  const char** feature_names,
  int num_feature_names);


/*!
 * @fn LGBM_DatasetGetFeatureNames
 * @brief Get feature names of Dataset.
 * @signature LIGHTGBM_C_EXPORT int LGBM_DatasetGetFeatureNames(DatasetHandle handle,
                                                                char** feature_names,
                                                               int* num_feature_names);
 * @param handle Handle
 * @param feature_names Feature names, should pre-allocate memory
 * @param num_feature_names Number of feature names
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetGetFeatureNames(
  DatasetHandle handle,
  char** feature_names,
  int* num_feature_names);


/*!
 * @fn LGBM_DatasetFree
 * @brief Free space for dataset.
 * signature LIGHTGBM_C_EXPORT int LGBM_DatasetFree(DatasetHandle handle);
 * @param handle Handle
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetFree(DatasetHandle handle);

/*!
 * @fn LGBM_DatasetSaveBinary
 * @brief Save dataset to binary file.
 * @signature LIGHTGBM_C_EXPORT int LGBM_DatasetSaveBinary(DatasetHandle handle,
                                                           const char* filename);
 * @param handle An instance of dataset
 * @param filename File name
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetSaveBinary(DatasetHandle handle,
                                             const char* filename);

/*!
 * @fn LGBM_DatasetDumpText
 * @brief Save dataset to text file, intended for debugging use only.
 * @signature LIGHTGBM_C_EXPORT int LGBM_DatasetDumpText(DatasetHandle handle,
                                                         const char* filename);
 * @param handle An instance of dataset
 * @param filename File name
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetDumpText(DatasetHandle handle,
                                           const char* filename);

/*!
 * @fn LGBM_DatasetSetField
 * @brief Set vector to a content in info.
 *        Note: group and group only work for C_API_DTYPE_INT32,
 *              label and weight only work for C_API_DTYPE_FLOAT32.
 * @signature LIGHTGBM_C_EXPORT int LGBM_DatasetSetField(DatasetHandle handle,
                                                         const char* field_name,
                                                        const void* field_data,
                                                        int num_element,
                                                        int type);
 * @param handle An instance of dataset
 * @param field_name Field name, can be label, weight, group, group_id
 * @param field_data Pointer to vector
 * @param num_element Number of element in field_data
 * @param type C_API_DTYPE_FLOAT32 or C_API_DTYPE_INT32
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetSetField(DatasetHandle handle,
                                           const char* field_name,
                                           const void* field_data,
                                           int num_element,
                                           int type);

/*!
 * @fn LGBM_DatasetGetField
 * @brief Get info vector from dataset.
 * @signature LIGHTGBM_C_EXPORT int LGBM_DatasetGetField(DatasetHandle handle,
                                                         const char* field_name,
                                                         int* out_len,
                                                         const void** out_ptr,
                                                         int* out_type);
 * @param handle An instance of data matrix
 * @param field_name Field name
 * @param out_len Used to set result length
 * @param out_ptr Pointer to the result
 * @param out_type C_API_DTYPE_FLOAT32 or C_API_DTYPE_INT32
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetGetField(DatasetHandle handle,
                                           const char* field_name,
                                           int* out_len,
                                           const void** out_ptr,
                                           int* out_type);


/*!
 * @fn LGBM_DatasetUpdateParam
 * @brief Update parameters for a Dataset.
 * @signature LIGHTGBM_C_EXPORT int LGBM_DatasetUpdateParam(DatasetHandle handle, 
                                                            const char* parameters);
 * @param handle An instance of data matrix
 * @param parameters Parameters
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetUpdateParam(DatasetHandle handle, const char* parameters);

/*!
 * @fn LGBM_DatasetGetNumData
 * @brief Get number of data.
 * @signature LIGHTGBM_C_EXPORT int LGBM_DatasetGetNumData(DatasetHandle handle,
                                                           int* out);
 * @param handle The handle to the dataset
 * @param[out] out The address to hold number of data
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetGetNumData(DatasetHandle handle,
                                             int* out);

/*!
 * @fn LGBM_DatasetGetNumFeature
 * @brief Get number of features.
 * @signature LIGHTGBM_C_EXPORT int LGBM_DatasetGetNumFeature(DatasetHandle handle,
                                                              int* out);
 * @param handle The handle to the dataset
 * @param out The output of number of features
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetGetNumFeature(DatasetHandle handle,
                                                int* out);

/*!
 * @fn LGBM_DatasetAddFeaturesFrom
 * @brief Add features from source to target, then free source.
 * @signature LIGHTGBM_C_EXPORT int LGBM_DatasetAddFeaturesFrom(DatasetHandle target,
                                                                DatasetHandle source);
 * @param target The handle of the dataset to add features to
 * @param source The handle of the dataset to take features from
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_DatasetAddFeaturesFrom(DatasetHandle target,
                                                  DatasetHandle source);

// --- start Booster interfaces

/*!
 * @fn LGBM_BoosterCreate
 * @brief Create an new boosting learner.
 * @signature LIGHTGBM_C_EXPORT int LGBM_BoosterCreate(const DatasetHandle train_data,
                                                       const char* parameters,
                                                       BoosterHandle* out);
 * @param train_data Training data set
 * @param parameters Format: 'key1=value1 key2=value2'
 * @param[out] out Handle of created Booster
* @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterCreate(const DatasetHandle train_data,
                                         const char* parameters,
                                         BoosterHandle* out);

/*!
 * @fn LGBM_BoosterCreateFromModelfile
 * @brief Load an existing boosting from model file.
 * @signature LIGHTGBM_C_EXPORT int LGBM_BoosterCreateFromModelfile(const char* filename,
                                                                    int* out_num_iterations,
                                                                    BoosterHandle* out);
 * @param filename Filename of model
 * @param out_num_iterations Number of iterations of this booster
 * @param[out] out Handle of created Booster
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterCreateFromModelfile(
  const char* filename,
  int* out_num_iterations,
  BoosterHandle* out);

/*!
 * @fn LGBM_BoosterLoadModelFromString
 * @brief Load an existing boosting from string.
 * @signature LIGHTGBM_C_EXPORT int LGBM_BoosterLoadModelFromString(const char* model_str,
                                                                    int* out_num_iterations,
                                                                    BoosterHandle* out);
 * @param model_str Model string
 * @param out_num_iterations Number of iterations of this booster
 * @param[out] out Handle of created Booster
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterLoadModelFromString(
  const char* model_str,
  int* out_num_iterations,
  BoosterHandle* out);

/*!
 * @fn LGBM_BoosterFree
 * @brief Free obj in handle.
 * @signature LIGHTGBM_C_EXPORT int LGBM_BoosterFree(BoosterHandle handle);
 * @param handle Handle to be freed
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterFree(BoosterHandle handle);

/*!
 * @fn LGBM_BoosterShuffleModels
 * @brief Shuffle Models.
 * @signature  LIGHTGBM_C_EXPORT int LGBM_BoosterShuffleModels(BoosterHandle handle, 
                                                               int start_iter, int end_iter);
 * @param handle
 * @param start_iter The first iteration that will be shuffled
 * @param end_iter The last iteration that will be shuffled
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterShuffleModels(BoosterHandle handle, int start_iter, int end_iter);

/*!
 * @fn LGBM_BoosterMerge
 * @brief Merge model in two booster to first handle.
 * @signature LIGHTGBM_C_EXPORT int LGBM_BoosterMerge(BoosterHandle handle,
                                                      BoosterHandle other_handle);
 * @param handle Handle, will merge other handle to this
 * @param other_handle
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterMerge(BoosterHandle handle,
                                        BoosterHandle other_handle);

/*!
 * @fn LGBM_BoosterAddValidData
 * @brief Add new validation to booster.
 * @signature LIGHTGBM_C_EXPORT int LGBM_BoosterAddValidData(BoosterHandle handle,
                                                             const DatasetHandle valid_data);
 * @param handle handle
 * @param valid_data validation data set
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterAddValidData(BoosterHandle handle,
                                               const DatasetHandle valid_data);

/*!
 * @fn LGBM_BoosterResetTrainingData
 * @brief Reset training data for booster.
 * @signature LIGHTGBM_C_EXPORT int LGBM_BoosterResetTrainingData(BoosterHandle handle,
                                                                  const DatasetHandle train_data);
 * @param handle Handle
 * @param train_data Training data set
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterResetTrainingData(BoosterHandle handle,
                                                    const DatasetHandle train_data);

/*!
 * @fn LGBM_BoosterResetParameter
 * @brief Reset config for current booster.
 * @signature LIGHTGBM_C_EXPORT int LGBM_BoosterResetParameter(BoosterHandle handle, 
                                                               const char* parameters);
 * @param handle Handle
 * @param parameters Format: 'key1=value1 key2=value2'
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterResetParameter(BoosterHandle handle, const char* parameters);

/*!
 * @fn LGBM_BoosterGetNumClasses
 * @brief Get number of class.
 * @signature LIGHTGBM_C_EXPORT int LGBM_BoosterGetNumClasses(BoosterHandle handle, 
                                                              int* out_len);
 * @param handle Handle
 * @param out_len Number of class
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterGetNumClasses(BoosterHandle handle, int* out_len);

/*!
 * @fn LGBM_BoosterUpdateOneIter
 * @brief Update the model in one round.
 * @signature LIGHTGBM_C_EXPORT int LGBM_BoosterUpdateOneIter(BoosterHandle handle, 
                                                              int* is_finished);
 * @param handle Handle
 * @param is_finished 1 means finised(cannot split any more)
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterUpdateOneIter(BoosterHandle handle, int* is_finished);

/*!
 * @fn LGBM_BoosterRefit
 * @brief Refit the tree model using the new data (online learning).
 * @signature LIGHTGBM_C_EXPORT int LGBM_BoosterRefit(BoosterHandle handle, 
                                                      const int32_t* leaf_preds, 
                                                      int32_t nrow, 
                                                      int32_t ncol);
 * @param handle Handle
 * @param leaf_preds 
 * @param nrow Number of rows of leaf_preds
 * @param ncol Number of columns of leaf_preds
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterRefit(BoosterHandle handle, const int32_t* leaf_preds, int32_t nrow, int32_t ncol);

/*!
 * @fn LGBM_BoosterUpdateOneIterCustom
 * @brief Update the model, by directly specify gradient and second order gradient,
 *        this can be used to support customized loss function.
 * @signature LIGHTGBM_C_EXPORT int LGBM_BoosterUpdateOneIterCustom(BoosterHandle handle,
                                                                    const float* grad,
                                                                    const float* hess,
                                                                    int* is_finished);
 * @param handle Handle
 * @param grad Gradient statistics
 * @param hess Second order gradient statistics
 * @param is_finished 1 means finised(cannot split any more)
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterUpdateOneIterCustom(BoosterHandle handle,
                                                      const float* grad,
                                                      const float* hess,
                                                      int* is_finished);

/*!
 * @fn LGBM_BoosterRollbackOneIter
 * @brief Rollback one iteration.
 * @signature LIGHTGBM_C_EXPORT int LGBM_BoosterRollbackOneIter(BoosterHandle handle);
 * @param handle Handle
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterRollbackOneIter(BoosterHandle handle);

/*!
 * @fn LGBM_BoosterGetCurrentIteration
 * @brief Get iteration of current boosting rounds.
 * @signature LIGHTGBM_C_EXPORT int LGBM_BoosterGetCurrentIteration(BoosterHandle handle,
                                                                    int* out_iteration);
 * @param handle Handle
 * @param out_iteration Iteration of boosting rounds
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterGetCurrentIteration(BoosterHandle handle, int* out_iteration);

/*!
 * @fn LGBM_BoosterNumModelPerIteration
 * @brief Get number of tree per iteration.
 * @signature LIGHTGBM_C_EXPORT int LGBM_BoosterNumModelPerIteration(BoosterHandle handle, 
                                                                     int* out_tree_per_iteration);
 * @param handle Handle
 * @param out_tree_per_iteration Number of tree per iteration
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterNumModelPerIteration(BoosterHandle handle, int* out_tree_per_iteration);

/*!
 * @fn LGBM_BoosterNumberOfTotalModel
 * @brief Get number of weak sub-models.
 * @signature LIGHTGBM_C_EXPORT int LGBM_BoosterNumberOfTotalModel(BoosterHandle handle, 
                                                                   int* out_models);
 * @param handle Handle
 * @param out_models Number of weak sub-models
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterNumberOfTotalModel(BoosterHandle handle, int* out_models);

/*!
 * @fn LGBM_BoosterGetEvalCounts
 * @brief Get number of eval.
 * @signature LIGHTGBM_C_EXPORT int LGBM_BoosterGetEvalCounts(BoosterHandle handle, 
                                                              int* out_len);
 * @param handle Handle
 * @param out_len Total number of eval results
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterGetEvalCounts(BoosterHandle handle, int* out_len);

/*!
 * @fn LGBM_BoosterGetEvalNames
 * @brief Get name of eval.
 * @signature LIGHTGBM_C_EXPORT int LGBM_BoosterGetEvalNames(BoosterHandle handle, 
                                                             int* out_len, 
                                                             char** out_strs);
 * @param handle Handle
 * @param out_len Total number of eval results
 * @param out_strs Names of eval result, need to pre-allocate memory before call this
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterGetEvalNames(BoosterHandle handle, int* out_len, char** out_strs);

/*!
 * @fn LGBM_BoosterGetFeatureNames
 * @brief Get name of features.
 * @signature LIGHTGBM_C_EXPORT int LGBM_BoosterGetFeatureNames(BoosterHandle handle, 
                                                                int* out_len, 
                                                                char** out_strs);
 * @param handle Handle
 * @param out_len Total number of features
 * @param out_strs Names of features, need to pre-allocate memory before call this
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterGetFeatureNames(BoosterHandle handle, int* out_len, char** out_strs);

/*!
 * @fn LGBM_BoosterGetNumFeature
 * @brief Get number of features.
 * @signature LIGHTGBM_C_EXPORT int LGBM_BoosterGetNumFeature(BoosterHandle handle, 
                                                              int* out_len);
 * @param handle Handle
 * @param out_len Total number of features
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterGetNumFeature(BoosterHandle handle, int* out_len);

/*!
 * @fn LGBM_BoosterGetEval
 * @brief Get evaluation for training data and validation data.
          Note: 1. You should call LGBM_BoosterGetEvalNames first to get the name of evaluation results.
                2. You should pre-allocate memory for out_results, you can get its length by LGBM_BoosterGetEvalCounts.
 * @signature LIGHTGBM_C_EXPORT int LGBM_BoosterGetEval(BoosterHandle handle,
                                                        int data_idx,
                                                        int* out_len,
                                                        double* out_results);
 * @param handle Handle
 * @param data_idx 0:training data, 1: 1st valid data, 2:2nd valid data ...
 * @param out_len Len of output result
 * @param out_result Float arrary contains result
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterGetEval(BoosterHandle handle,
                                          int data_idx,
                                          int* out_len,
                                          double* out_results);

/*!
 * @fn LGBM_BoosterGetNumPredict
 * @brief Get number of predict for inner dataset.
          This can be used to support customized eval function.
          Note: Should pre-allocate memory for out_result, its length is equal to num_class * num_data.
 * @signature LIGHTGBM_C_EXPORT int LGBM_BoosterGetNumPredict(BoosterHandle handle,
                                                              int data_idx,
                                                              int64_t* out_len);
 * @param handle Handle
 * @param data_idx 0:training data, 1: 1st valid data, 2:2nd valid data, ...
 * @param out_len Len of output result
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterGetNumPredict(BoosterHandle handle,
                                                int data_idx,
                                                int64_t* out_len);

/*!
 * @fn LGBM_BoosterGetPredict
 * @brief Get prediction for training data and validation data.
          This can be used to support customized eval function.
          Note: Should pre-allocate memory for out_result, its length is equal to num_class * num_data.
 * @signature LIGHTGBM_C_EXPORT int LGBM_BoosterGetPredict(BoosterHandle handle,
                                                           int data_idx,
                                                           int64_t* out_len,
                                                           double* out_result);
 * @param handle Handle
 * @param data_idx 0:training data, 1: 1st valid data, 2:2nd valid data, ...
 * @param out_len Len of output result
 * @param out_result Used to set a pointer to array, should allocate memory before call this function
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterGetPredict(BoosterHandle handle,
                                             int data_idx,
                                             int64_t* out_len,
                                             double* out_result);

/*!
 * @fn LGBM_BoosterPredictForFile
 * @brief Make prediction for file.
 * @signature LIGHTGBM_C_EXPORT int LGBM_BoosterPredictForFile(BoosterHandle handle,
                                                               const char* data_filename,
                                                               int data_has_header,
                                                               int predict_type,
                                                               int num_iteration,
                                                               const char* parameter,
                                                               const char* result_filename);
 * @param handle Handle
 * @param data_filename Filename of data file
 * @param data_has_header Data file has header or not
 * @param predict_type
            C_API_PREDICT_NORMAL: normal prediction, with transform (if needed)
            C_API_PREDICT_RAW_SCORE: raw score
            C_API_PREDICT_LEAF_INDEX: leaf index
 * @param num_iteration Number of iteration for prediction, <= 0 means no limit
 * @param parameter Other parameters for the parameters, e.g. early stopping for prediction.
 * @param result_filename Filename of result file
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterPredictForFile(BoosterHandle handle,
                                                 const char* data_filename,
                                                 int data_has_header,
                                                 int predict_type,
                                                 int num_iteration,
                                                 const char* parameter,
                                                 const char* result_filename);

/*!
 * @fn LGBM_BoosterCalcNumPredict
 * @brief Get number of prediction.
 * @signature LIGHTGBM_C_EXPORT int LGBM_BoosterCalcNumPredict(BoosterHandle handle,
                                                               int num_row,
                                                               int predict_type,
                                                               int num_iteration,
                                                               int64_t* out_len);
 * @param handle Handle
 * @param num_row Number of rows
 * @param predict_type
           C_API_PREDICT_NORMAL: normal prediction, with transform (if needed)
           C_API_PREDICT_RAW_SCORE: raw score
           C_API_PREDICT_LEAF_INDEX: leaf index
 * @param num_iteration Number of iteration for prediction, <= 0 means no limit
 * @param out_len Length of prediction
* @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterCalcNumPredict(BoosterHandle handle,
                                                 int num_row,
                                                 int predict_type,
                                                 int num_iteration,
                                                 int64_t* out_len);

/*!
 * @fn LGBM_BoosterPredictForCSR
 * @brief Make prediction for an new data set.
 *        Note: Should pre-allocate memory for out_result:
 *              For normal and raw score: its length is equal to num_class * num_data;
 *              For leaf index, its length is equal to num_class * num_data * num_iteration.
 * @signature LIGHTGBM_C_EXPORT int LGBM_BoosterPredictForCSR(BoosterHandle handle,
                                                              void* indptr,
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
 * @param handle Handle
 * @param indptr Pointer to row headers
 * @param indptr_type Type of indptr, can be C_API_DTYPE_INT32 or C_API_DTYPE_INT64
 * @param indices Findex
 * @param data Fvalue
 * @param data_type Type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64
 * @param nindptr Number of rows in the matrix + 1
 * @param nelem Number of nonzero elements in the matrix
 * @param num_col Number of columns; when it's set to 0, then guess from data
 * @param predict_type
 *          C_API_PREDICT_NORMAL: normal prediction, with transform (if needed)
 *          C_API_PREDICT_RAW_SCORE: raw score
 *          C_API_PREDICT_LEAF_INDEX: leaf index
 * @param num_iteration Number of iteration for prediction, <= 0 means no limit
 * @param parameter Other parameters for the parameters, e.g. early stopping for prediction.
 * @param out_len Len of output result
 * @param out_result Used to set a pointer to array, should allocate memory before call this function
 * @return 0 when succeed, -1 when failure happens
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
 * @fn LGBM_BoosterPredictForCSRSingleRow
 * @brief Make prediction for an new data set. This method re-uses the internal predictor structure 
          from previous calls and is optimized for single row invocation.
          Note: Should pre-allocate memory for out_result:
                For normal and raw score: its length is equal to num_class * num_data;
                For leaf index, its length is equal to num_class * num_data * num_iteration.
 * @signature LIGHTGBM_C_EXPORT int LGBM_BoosterPredictForCSRSingleRow(BoosterHandle handle,
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
 * @param handle Handle
 * @param indptr Pointer to row headers
 * @param indptr_type Type of indptr, can be C_API_DTYPE_INT32 or C_API_DTYPE_INT64
 * @param indices Findex
 * @param data Fvalue
 * @param data_type Type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64
 * @param nindptr Number of rows in the matrix + 1
 * @param nelem Number of nonzero elements in the matrix
 * @param num_col Number of columns; when it's set to 0, then guess from data
 * @param predict_type
            C_API_PREDICT_NORMAL: normal prediction, with transform (if needed)
            C_API_PREDICT_RAW_SCORE: raw score
            C_API_PREDICT_LEAF_INDEX: leaf index
 * @param num_iteration Number of iteration for prediction, <= 0 means no limit
 * @param parameter Other parameters for the parameters, e.g. early stopping for prediction.
 * @param out_len Len of output result
 * @param out_result Used to set a pointer to array, should allocate memory before call this function
 * @return 0 when succeed, -1 when failure happens
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
 * @fn LGBM_BoosterPredictForCSC
 * @brief Make prediction for an new data set.
          Note: Should pre-allocate memory for out_result:
                For normal and raw score: its length is equal to num_class * num_data;
                For leaf index, its length is equal to num_class * num_data * num_iteration.
 * @signature LIGHTGBM_C_EXPORT int LGBM_BoosterPredictForCSC(BoosterHandle handle,
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
 * @param handle Handle
 * @param col_ptr Pointer to col headers
 * @param col_ptr_type Type of col_ptr, can be C_API_DTYPE_INT32 or C_API_DTYPE_INT64
 * @param indices Findex
 * @param data Fvalue
 * @param data_type Type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64
 * @param ncol_ptr Number of cols in the matrix + 1
 * @param nelem Number of nonzero elements in the matrix
 * @param num_row Number of rows
 * @param predict_type
           C_API_PREDICT_NORMAL: normal prediction, with transform (if needed)
           C_API_PREDICT_RAW_SCORE: raw score
           C_API_PREDICT_LEAF_INDEX: leaf index
 * @param num_iteration Number of iteration for prediction, <= 0 means no limit
 * @param parameter Other parameters for the parameters, e.g. early stopping for prediction.
 * @param out_len Len of output result
 * @param out_result Used to set a pointer to array, should allocate memory before call this function
 * @return 0 when succeed, -1 when failure happens
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
 * @fn LGBM_BoosterPredictForMat
 * @brief Make prediction for an new data set.
          Note: Should pre-allocate memory for out_result:
                For normal and raw score: its length is equal to num_class * num_data;
                For leaf index, its length is equal to num_class * num_data * num_iteration.
 * @signature LIGHTGBM_C_EXPORT int LGBM_BoosterPredictForMat(BoosterHandle handle,
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
 * @param handle Handle
 * @param data Pointer to the data space
 * @param data_type Type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64
 * @param nrow Number of rows
 * @param ncol Number columns
 * @param is_row_major 1 for row major, 0 for column major
 * @param predict_type
           C_API_PREDICT_NORMAL: normal prediction, with transform (if needed)
           C_API_PREDICT_RAW_SCORE: raw score
           C_API_PREDICT_LEAF_INDEX: leaf index
 * @param num_iteration Number of iteration for prediction, <= 0 means no limit
 * @param parameter Other parameters for the parameters, e.g. early stopping for prediction.
 * @param out_len Len of output result
 * @param out_result Used to set a pointer to array, should allocate memory before call this function
 * @return 0 when succeed, -1 when failure happens
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
 * @fn LGBM_BoosterPredictForMatSingleRow
 * @brief Make prediction for an new data set. This method re-uses the internal predictor structure 
          from previous calls and is optimized for single row invocation.
          Note: Should pre-allocate memory for out_result:
                For normal and raw score: its length is equal to num_class * num_data;
                For leaf index, its length is equal to num_class * num_data * num_iteration.
 * @signature LIGHTGBM_C_EXPORT int LGBM_BoosterPredictForMatSingleRow(BoosterHandle handle,
                                                                       const void* data,
                                                                       int data_type,
                                                                       int ncol,
                                                                       int is_row_major,
                                                                       int predict_type,
                                                                       int num_iteration,
                                                                       const char* parameter,
                                                                       int64_t* out_len,
                                                                       double* out_result);
 * @param handle Handle
 * @param data Pointer to the data space
 * @param data_type Type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64
 * @param ncol Number columns
 * @param is_row_major 1 for row major, 0 for column major
 * @param predict_type
           C_API_PREDICT_NORMAL: normal prediction, with transform (if needed)
           C_API_PREDICT_RAW_SCORE: raw score
           C_API_PREDICT_LEAF_INDEX: leaf index
 * @param num_iteration Number of iteration for prediction, <= 0 means no limit
 * @param parameter Other parameters for the parameters, e.g. early stopping for prediction.
 * @param out_len Len of output result
 * @param out_result Used to set a pointer to array, should allocate memory before call this function
 * @return 0 when succeed, -1 when failure happens
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
 * @fn LGBM_BoosterPredictForMats
 * @brief Make prediction for an new data set.
          Note: Should pre-allocate memory for out_result;
                For noraml and raw score: its length is equal to num_class * num_data;
                For leaf index, its length is equal to num_class * num_data * num_iteration. 
 * @signature LIGHTGBM_C_EXPORT int LGBM_BoosterPredictForMats(BoosterHandle handle,
                                                               const void** data,
                                                               int data_type,
                                                                int32_t nrow,
                                                               int32_t ncol,
                                                               int predict_type,
                                                               int num_iteration,
                                                                const char* parameter,
                                                               int64_t* out_len,
                                                               double* out_result);
 * @param handle Handle
 * @param data Pointer to the data space
 * @param data_type Type of data pointer, can be C_API_DTYPE_FLOAT32 or C_API_DTYPE_FLOAT64
 * @param nrow Number of rows
 * @param ncol Number columns
 * @param predict_type
           C_API_PREDICT_NORMAL: normal prediction, with transform (if needed)
           C_API_PREDICT_RAW_SCORE: raw score
           C_API_PREDICT_LEAF_INDEX: leaf index
 * @param num_iteration Number of iteration for prediction, <= 0 means no limit
 * @param parameter Other parameters for the parameters, e.g. early stopping for prediction.
 * @param out_len Len of output result
 * @param out_result Used to set a pointer to array, should allocate memory before call this function
 * @return 0 when succeed, -1 when failure happens
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
 * @fn LGBM_BoosterSaveModel
 * @brief Save model into file.
 * @signature LIGHTGBM_C_EXPORT int LGBM_BoosterSaveModel(BoosterHandle handle,
                                                          int start_iteration,
                                                          int num_iteration,
                                                          const char* filename);
 * @param handle Handle
 * @param start_iteration
 * @param num_iteration <= 0 means save all
 * @param filename File name
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterSaveModel(BoosterHandle handle,
                                            int start_iteration,
                                            int num_iteration,
                                            const char* filename);

/*!
 * @fn LGBM_BoosterSaveModelToString
 * @brief Save model to string.
 * @signature LIGHTGBM_C_EXPORT int LGBM_BoosterSaveModelToString(BoosterHandle handle,
                                                                  int start_iteration,
                                                                  int num_iteration,
                                                                  int64_t buffer_len,
                                                                  int64_t* out_len,
                                                                  char* out_str);
 * @param handle Handle
 * @param start_iteration
 * @param num_iteration <= 0 means save all
 * @param buffer_len String buffer length, if buffer_len < out_len, re-allocate buffer
 * @param out_len Actual output length
 * @param out_str String of model, need to pre-allocate memory before call this
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterSaveModelToString(BoosterHandle handle,
                                                    int start_iteration,
                                                    int num_iteration,
                                                    int64_t buffer_len,
                                                    int64_t* out_len,
                                                    char* out_str);

/*!
 * @fn LGBM_BoosterDumpModel
 * @brief Dump model to json.
 * @signature LIGHTGBM_C_EXPORT int LGBM_BoosterDumpModel(BoosterHandle handle,
                                                          int start_iteration,
                                                          int num_iteration,
                                                          int64_t buffer_len,
                                                          int64_t* out_len,
                                                          char* out_str);
 * @param handle Handle
 * @param start_iteration
 * @param num_iteration <= 0 means save all
 * @param buffer_len String buffer length, if buffer_len < out_len, re-allocate buffer
 * @param out_len Actual output length
 * @param out_str Json format string of model, need to pre-allocate memory before call this
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterDumpModel(BoosterHandle handle,
                                            int start_iteration,
                                            int num_iteration,
                                            int64_t buffer_len,
                                            int64_t* out_len,
                                            char* out_str);

/*!
 * @fn LGBM_BoosterGetLeafValue
 * @brief Get leaf value.
 * @signature LIGHTGBM_C_EXPORT int LGBM_BoosterGetLeafValue(BoosterHandle handle,
                                                             int tree_idx,
                                                             int leaf_idx,
                                                             double* out_val);
 * @param handle Handle
 * @param tree_idx Index of tree
 * @param leaf_idx Index of leaf
 * @param out_val Out result
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterGetLeafValue(BoosterHandle handle,
                                               int tree_idx,
                                               int leaf_idx,
                                               double* out_val);

/*!
 * @fn LGBM_BoosterSetLeafValue
 * @brief Set leaf value.
 * @signature LIGHTGBM_C_EXPORT int LGBM_BoosterSetLeafValue(BoosterHandle handle,
                                                             int tree_idx,
                                                             int leaf_idx,
                                                             double val);
 * @param handle Handle
 * @param tree_idx Index of tree
 * @param leaf_idx Index of leaf
 * @param val Leaf value
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterSetLeafValue(BoosterHandle handle,
                                               int tree_idx,
                                               int leaf_idx,
                                               double val);

/*!
 * @fn LGBM_BoosterFeatureImportance
 * @brief Get model feature importance.
 * @signature LIGHTGBM_C_EXPORT int LGBM_BoosterFeatureImportance(BoosterHandle handle,
                                                                  int num_iteration,
                                                                  int importance_type,
                                                                  double* out_results);
 * @param handle Handle
 * @param num_iteration <= 0 means use all
 * @param importance_type 0 for split, 1 for gain
 * @param out_results Output value array
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_BoosterFeatureImportance(BoosterHandle handle,
                                                    int num_iteration,
                                                    int importance_type,
                                                    double* out_results);

/*!
 * @fn LGBM_NetworkInit
 * @brief Initilize the network.
 * @signature LIGHTGBM_C_EXPORT int LGBM_NetworkInit(const char* machines,
                                                     int local_listen_port,
                                                     int listen_time_out,
                                                     int num_machines);
 * @param machines Represent the nodes, format: ip1:port1,ip2:port2
 * @param local_listen_port
 * @param listen_time_out
 * @param num_machines
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_NetworkInit(const char* machines,
                                       int local_listen_port,
                                       int listen_time_out,
                                       int num_machines);

/*!
 * @fn LGBM_NetworkFree
 * @brief Finalize the network.
 * @signature LIGHTGBM_C_EXPORT int LGBM_NetworkFree();
 * @return 0 when succeed, -1 when failure happens
*/
LIGHTGBM_C_EXPORT int LGBM_NetworkFree();

/*!
 * @fn LGBM_NetworkInitWithFunctions
 * @brief 
 * @signature LIGHTGBM_C_EXPORT int LGBM_NetworkInitWithFunctions(int num_machines, int rank,
                                                                  void* reduce_scatter_ext_fun,
                                                                  void* allgather_ext_fun);
 * @param num_machines
 * @param rank
 * @param reduce_scatter_ext_fun
 * @param allgather_ext_fun
 * @return 0 when succeed, -1 when failure happens
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

#endif // LIGHTGBM_C_API_H_