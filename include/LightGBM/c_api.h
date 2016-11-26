#ifndef LIGHTGBM_C_API_H_
#define LIGHTGBM_C_API_H_
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>
/*!
* To avoid type conversion on large data, most of our expose interface support both for float_32 and float_64.
* Except following:
* 1. gradients and hessians.
* 2. Get current score for training data and validation
* The reason is because they are called frequently, the type-conversion on them maybe time cost.
*/

#ifdef __cplusplus
#define DLL_EXTERN_C extern "C"
#else
#define DLL_EXTERN_C
#endif

#ifdef _MSC_VER
#define DllExport DLL_EXTERN_C __declspec(dllexport)
#else
#define DllExport DLL_EXTERN_C
#endif

typedef void* DatesetHandle;
typedef void* BoosterHandle;

#define C_API_DTYPE_FLOAT32 (0)
#define C_API_DTYPE_FLOAT64 (1)
#define C_API_DTYPE_INT32   (2)
#define C_API_DTYPE_INT64   (3)

#define C_API_PREDICT_NORMAL     (0)
#define C_API_PREDICT_RAW_SCORE  (1)
#define C_API_PREDICT_LEAF_INDEX (2)

/*!
* \brief get string message of the last error
*  all function in this file will return 0 when succeed
*  and -1 when an error occured,
* \return const char* error inforomation
*/
DllExport const char* LGBM_GetLastError();


// --- start Dataset interface

/*!
* \brief load data set from file like the command_line LightGBM do
* \param filename the name of the file
* \param parameters additional parameters
* \param reference used to align bin mapper with other dataset, nullptr means don't used
* \param out a loaded dataset
* \return 0 when succeed, -1 when failure happens
*/
DllExport int LGBM_CreateDatasetFromFile(const char* filename,
  const char* parameters,
  const DatesetHandle* reference,
  DatesetHandle* out);

/*!
* \brief load data set from binary file like the command_line LightGBM do
* \param filename the name of the file
* \param out a loaded dataset
* \return 0 when succeed, -1 when failure happens
*/
DllExport int LGBM_CreateDatasetFromBinaryFile(const char* filename,
  DatesetHandle* out);

/*!
* \brief create a dataset from CSR format
* \param indptr pointer to row headers
* \param indptr_type
* \param indices findex
* \param data fvalue
* \param data_type
* \param nindptr number of rows in the matrix + 1
* \param nelem number of nonzero elements in the matrix
* \param num_col number of columns; when it's set to 0, then guess from data
* \param parameters additional parameters
* \param reference used to align bin mapper with other dataset, nullptr means don't used
* \param out created dataset
* \return 0 when succeed, -1 when failure happens
*/
DllExport int LGBM_CreateDatasetFromCSR(const void* indptr,
  int indptr_type,
  const int32_t* indices,
  const void* data,
  int data_type,
  int64_t nindptr,
  int64_t nelem,
  int64_t num_col,
  const char* parameters,
  const DatesetHandle* reference,
  DatesetHandle* out);

/*!
* \brief create a dataset from CSC format
* \param col_ptr pointer to col headers
* \param col_ptr_type
* \param indices findex
* \param data fvalue
* \param data_type
* \param ncol_ptr number of rows in the matrix + 1
* \param nelem number of nonzero elements in the matrix
* \param num_row number of rows; when it's set to 0, then guess from data
* \param parameters additional parameters
* \param reference used to align bin mapper with other dataset, nullptr means don't used
* \param out created dataset
* \return 0 when succeed, -1 when failure happens
*/
DllExport int LGBM_CreateDatasetFromCSC(const void* col_ptr,
  int col_ptr_type,
  const int32_t* indices,
  const void* data,
  int data_type,
  int64_t ncol_ptr,
  int64_t nelem,
  int64_t num_row,
  const char* parameters,
  const DatesetHandle* reference,
  DatesetHandle* out);

/*!
* \brief create dataset from dense matrix
* \param data pointer to the data space
* \param data_type 0
* \param nrow number of rows
* \param ncol number columns
* \param is_row_major 1 for row major, 0 for column major
* \param parameters additional parameters
* \param reference used to align bin mapper with other dataset, nullptr means don't used
* \param out created dataset
* \return 0 when succeed, -1 when failure happens
*/
DllExport int LGBM_CreateDatasetFromMat(const void* data,
  int data_type,
  int32_t nrow,
  int32_t ncol,
  int is_row_major,
  const char* parameters,
  const DatesetHandle* reference,
  DatesetHandle* out);

/*!
* \brief free space for dataset
* \return 0 when succeed, -1 when failure happens
*/
DllExport int LGBM_DatasetFree(DatesetHandle handle);

/*!
* \brief save dateset to binary file
* \param handle a instance of dataset
* \param filename file name
* \return 0 when succeed, -1 when failure happens
*/
DllExport int LGBM_DatasetSaveBinary(DatesetHandle handle,
  const char* filename);

/*!
* \brief set vector to a content in info
* \param handle a instance of dataset
* \param field_name field name, can be label, weight, group
* \param field_data pointer to vector
* \param num_element number of element in field_data
* \param type float32 or int32
* \return 0 when succeed, -1 when failure happens
*/
DllExport int LGBM_DatasetSetField(DatesetHandle handle,
  const char* field_name,
  const void* field_data,
  int64_t num_element,
  int type);

/*!
* \brief get info vector from dataset
* \param handle a instance of data matrix
* \param field_name field name
* \param out_len used to set result length
* \param out_ptr pointer to the result
* \param out_type  float32 or int32
* \return 0 when succeed, -1 when failure happens
*/
DllExport int LGBM_DatasetGetField(DatesetHandle handle,
  const char* field_name,
  int64_t* out_len,
  const void** out_ptr,
  int* out_type);

/*!
* \brief get number of data.
* \param handle the handle to the dataset
* \param out The address to hold number of data
* \return 0 when succeed, -1 when failure happens
*/
DllExport int LGBM_DatasetGetNumData(DatesetHandle handle,
  int64_t* out);

/*!
* \brief get number of features
* \param handle the handle to the dataset
* \param out The output of number of features
* \return 0 when succeed, -1 when failure happens
*/
DllExport int LGBM_DatasetGetNumFeature(DatesetHandle handle,
  int64_t* out);

// --- start Booster interfaces

/*!
* \brief create an new boosting learner
* \param train_data training data set
* \param parameters format: 'key1=value1 key2=value2'
* \prama out handle of created Booster
* \return 0 when succeed, -1 when failure happens
*/
DllExport int LGBM_BoosterCreate(const DatesetHandle train_data,
  const char* parameters,
  BoosterHandle* out);

/*!
* \brief load an existing boosting from model file
* \param filename filename of model
* \param out_num_total_model number of total models
* \param out handle of created Booster
* \return 0 when succeed, -1 when failure happens
*/
DllExport int LGBM_BoosterCreateFromModelfile(
  const char* filename,
  int64_t* out_num_total_model,
  BoosterHandle* out);


/*!
* \brief free obj in handle
* \param handle handle to be freed
* \return 0 when succeed, -1 when failure happens
*/
DllExport int LGBM_BoosterFree(BoosterHandle handle);

/*!
* \brief Merge model in two booster to first handle
* \param handle handle, will merge other handle to this
* \param other_handle
* \return 0 when succeed, -1 when failure happens
*/
DllExport int LGBM_BoosterMerge(BoosterHandle handle,
  BoosterHandle other_handle);

/*!
* \brief Add new validation to booster
* \param handle handle
* \param valid_data validation data set
* \return 0 when succeed, -1 when failure happens
*/
DllExport int LGBM_BoosterAddValidData(BoosterHandle handle,
  const DatesetHandle valid_data);

/*!
* \brief Reset training data for booster
* \param handle handle
* \param train_data training data set
* \return 0 when succeed, -1 when failure happens
*/
DllExport int LGBM_BoosterResetTrainingData(BoosterHandle handle,
  const DatesetHandle train_data);

/*!
* \brief Reset config for current booster
* \param handle handle
* \param parameters format: 'key1=value1 key2=value2'
* \return 0 when succeed, -1 when failure happens
*/
DllExport int LGBM_BoosterResetParameter(BoosterHandle handle, const char* parameters);

/*!
* \brief Get number of class 
* \param handle handle
* \return number of class
*/
DllExport int LGBM_BoosterGetNumClasses(BoosterHandle handle, int64_t* out_len);

/*!
* \brief update the model in one round
* \param handle handle
* \param is_finished 1 means finised(cannot split any more)
* \return 0 when succeed, -1 when failure happens
*/
DllExport int LGBM_BoosterUpdateOneIter(BoosterHandle handle, int* is_finished);

/*!
* \brief update the model, by directly specify gradient and second order gradient,
*       this can be used to support customized loss function
* \param handle handle
* \param grad gradient statistics
* \param hess second order gradient statistics
* \param is_finished 1 means finised(cannot split any more)
* \return 0 when succeed, -1 when failure happens
*/
DllExport int LGBM_BoosterUpdateOneIterCustom(BoosterHandle handle,
  const float* grad,
  const float* hess,
  int* is_finished);

/*!
* \brief Rollback one iteration
* \param handle handle
* \return 0 when succeed, -1 when failure happens
*/
DllExport int LGBM_BoosterRollbackOneIter(BoosterHandle handle);

/*!
* \brief Get iteration of current boosting rounds
* \return iteration of boosting rounds
*/
DllExport int LGBM_BoosterGetCurrentIteration(BoosterHandle handle, int64_t* out_iteration);

/*!
* \brief Get number of eval 
* \return total number of eval result
*/
DllExport int LGBM_BoosterGetEvalCounts(BoosterHandle handle, int64_t* out_len);

/*!
* \brief Get number of eval
* \return total number of eval result
*/
DllExport int LGBM_BoosterGetEvalNames(BoosterHandle handle, int64_t* out_len, char** out_strs);

/*!
* \brief get evaluation for training data and validation data
* \param handle handle
* \param data_idx 0:training data, 1: 1st valid data, 2:2nd valid data ...
* \param out_len len of output result
* \param out_result the string containing evaluation statistics, should allocate memory before call this function
* \return 0 when succeed, -1 when failure happens
*/
DllExport int LGBM_BoosterGetEval(BoosterHandle handle,
  int data_idx,
  int64_t* out_len,
  float* out_results);

/*!
* \brief Get prediction for training data and validation data
this can be used to support customized eval function
* \param handle handle
* \param data_idx 0:training data, 1: 1st valid data, 2:2nd valid data ...
* \param out_len len of output result
* \param out_result used to set a pointer to array, should allocate memory before call this function
* \return 0 when succeed, -1 when failure happens
*/
DllExport int LGBM_BoosterGetPredict(BoosterHandle handle,
  int data_idx,
  int64_t* out_len,
  float* out_result);

/*!
* \brief make prediction for file
* \param handle handle
* \param data_filename filename of data file
* \param data_has_header data file has header or not
* \param predict_type
*          0:normal, with transform (if needed)
*          1:raw score
*          2:leaf index
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \param result_filename filename of result file
* \return 0 when succeed, -1 when failure happens
*/
DllExport int LGBM_BoosterPredictForFile(BoosterHandle handle,
  const char* data_filename,
  int data_has_header,
  int predict_type,
  int64_t num_iteration,
  const char* result_filename);

/*!
* \brief make prediction for an new data set
* \param handle handle
* \param indptr pointer to row headers
* \param indptr_type
* \param indices findex
* \param data fvalue
* \param data_type
* \param nindptr number of rows in the matrix + 1
* \param nelem number of nonzero elements in the matrix
* \param num_col number of columns; when it's set to 0, then guess from data
* \param predict_type
*          0:normal, with transform (if needed)
*          1:raw score
*          2:leaf index
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \param out_len len of output result
* \param out_result used to set a pointer to array, should allocate memory before call this function
* \return 0 when succeed, -1 when failure happens
*/
DllExport int LGBM_BoosterPredictForCSR(BoosterHandle handle,
  const void* indptr,
  int indptr_type,
  const int32_t* indices,
  const void* data,
  int data_type,
  int64_t nindptr,
  int64_t nelem,
  int64_t num_col,
  int predict_type,
  int64_t num_iteration,
  int64_t* out_len,
  float* out_result);

/*!
* \brief make prediction for an new data set
* \param handle handle
* \param data pointer to the data space
* \param data_type
* \param nrow number of rows
* \param ncol number columns
* \param is_row_major 1 for row major, 0 for column major
* \param predict_type
*          0:normal, with transform (if needed)
*          1:raw score
*          2:leaf index
* \param num_iteration number of iteration for prediction, <= 0 means no limit
* \param out_len len of output result
* \param out_result used to set a pointer to array, should allocate memory before call this function
* \return 0 when succeed, -1 when failure happens
*/
DllExport int LGBM_BoosterPredictForMat(BoosterHandle handle,
  const void* data,
  int data_type,
  int32_t nrow,
  int32_t ncol,
  int is_row_major,
  int predict_type,
  int64_t num_iteration,
  int64_t* out_len,
  float* out_result);

/*!
* \brief save model into file
* \param handle handle
* \param num_iteration, <= 0 means save all
* \param filename file name
* \return 0 when succeed, -1 when failure happens
*/
DllExport int LGBM_BoosterSaveModel(BoosterHandle handle,
  int num_iteration,
  const char* filename);



// some help functions used to convert data

std::function<std::vector<double>(int row_idx)>
RowFunctionFromDenseMatric(const void* data, int num_row, int num_col, int data_type, int is_row_major);

std::function<std::vector<std::pair<int, double>>(int row_idx)>
RowPairFunctionFromDenseMatric(const void* data, int num_row, int num_col, int data_type, int is_row_major);

std::function<std::vector<std::pair<int, double>>(int idx)>
RowFunctionFromCSR(const void* indptr, int indptr_type, const int32_t* indices,
  const void* data, int data_type, int64_t nindptr, int64_t nelem);

std::function<std::vector<std::pair<int, double>>(int idx)>
ColumnFunctionFromCSC(const void* col_ptr, int col_ptr_type, const int32_t* indices,
  const void* data, int data_type, int64_t ncol_ptr, int64_t nelem);

std::vector<double>
SampleFromOneColumn(const std::vector<std::pair<int, double>>& data, const std::vector<int>& indices);


// exception handle and error msg

static std::string& LastErrorMsg() { static thread_local std::string err_msg("Everything is fine"); return err_msg; }

inline void LGBM_SetLastError(const char* msg) {
  LastErrorMsg() = msg;
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
