#ifndef LIGHTGBM_C_API_H_
#define LIGHTGBM_C_API_H_
#include<cstdint>


/*!
* To avoid type conversion on large data, most of our expose interface support both for float_32 and float_64.
* Except following:
* 1. gradients and hessians. 
* 2. Get current score for training data and validation
* The reason is becaused they are called frequently, the type-conversion on them maybe time cost. 
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

/*!
* \brief get string message of the last error
*  all function in this file will return 0 when success
*  and -1 when an error occured,
* \return const char* error inforomation
*/
DllExport const char* LGBM_GetLastError();


// --- start Dataset inferfaces

/*!
* \brief load data set from file like the command_line LightGBM do
* \param parameters additional parameters
* \param filename the name of the file
* \param reference used to align bin mapper with other dataset, nullptr means don't used
* \param out a loaded dataset
* \return 0 when success, -1 when failure happens
*/
DllExport int LGBM_CreateDatasetFromFile(const char* filename,
  const char* parameters,
  const DatesetHandle* reference,
  DatesetHandle* out);

/*!
* \brief load data set from binary file like the command_line LightGBM do
* \param filename the name of the file
* \param out a loaded dataset
* \return 0 when success, -1 when failure happens
*/
DllExport int LGBM_CreateDatasetFromBinaryFile(const char* filename,
  DatesetHandle* out);

/*!
* \brief create a dataset from CSR format
* \param indptr pointer to row headers
* \param indices findex
* \param data fvalue
* \param float_type 0 for float_32 1 for float_64
* \param nindptr number of rows in the matix + 1
* \param nelem number of nonzero elements in the matrix
* \param num_col number of columns; when it's set to 0, then guess from data
* \param parameters additional parameters
* \param reference used to align bin mapper with other dataset, nullptr means don't used
* \param out created dataset
* \return 0 when success, -1 when failure happens
*/
DllExport int LGBM_CreateDatasetFromCSR(const int32_t* indptr,
  const int32_t* indices,
  const void* data,
  int float_type,
  uint64_t nindptr,
  uint64_t nelem,
  uint64_t num_col,
  const char* parameters,
  const DatesetHandle* reference,
  DatesetHandle* out);

/*!
* \brief create a dataset from CSC format
* \param col_ptr pointer to col headers
* \param indices findex
* \param data fvalue
* \param float_type 0 for float_32 1 for float_64
* \param nindptr number of rows in the matix + 1
* \param nelem number of nonzero elements in the matrix
* \param num_row number of rows; when it's set to 0, then guess from data
* \param parameters additional parameters
* \param reference used to align bin mapper with other dataset, nullptr means don't used
* \param out created dataset
* \return 0 when success, -1 when failure happens
*/
DllExport int LGBM_CreateDatasetFromCSC(const int32_t* col_ptr,
  const int32_t* indices,
  const void* data,
  int float_type,
  uint64_t nindptr,
  uint64_t nelem,
  uint64_t num_row,
  const char* parameters,
  const DatesetHandle* reference,
  DatesetHandle* out);

/*!
* \brief create dataset from dense matrix
* \param data pointer to the data space
* \param float_type 0 for float_32 1 for float_64
* \param nrow number of rows
* \param ncol number columns
* \param is_row_major 1 for row major, 0 for column major
* \param missing which value to represent missing value
* \param parameters additional parameters
* \param reference used to align bin mapper with other dataset, nullptr means don't used
* \param out created dataset
* \return 0 when success, -1 when failure happens
*/
DllExport int LGBM_CreateDatasetFromMat(const void* data,
  int float_type,
  int32_t nrow,
  int32_t ncol,
  int is_row_major,
  const char* parameters,
  const DatesetHandle* reference,
  DatesetHandle* out);

/*!
* \brief free space for dataset
* \return 0 when success, -1 when failure happens
*/
DllExport int LGBM_DatasetFree(DatesetHandle* handle);

/*!
* \brief save dateset to binary file
* \param handle a instance of dataset
* \param filename file name
* \return 0 when success, -1 when failure happens
*/
DllExport int LGBM_DatasetSaveBinary(DatesetHandle handle,
  const char* filename);

/*!
* \brief set vector to a content in info
* \param handle a instance of dataset
* \param field_name field name, can be label, weight, group
* \param field_data pointer to vector
* \param field_len number of element in field_data
* \param type float_32:0, uint32_t:1
* \return 0 when success, -1 when failure happens
*/
DllExport int LGBM_DatasetSetField(DatesetHandle handle,
  const char* field_name,
  const void* field_data,
  uint64_t num_element,
  int type);

/*!
* \brief get info vector from dataset
* \param handle a instance of data matrix
* \param field_name field name
* \param out_len used to set result length
* \param out_ptr pointer to the result
* \param out_type  float_32:0, uint32_t:1
* \return 0 when success, -1 when failure happens
*/
DllExport int LGBM_DatasetGetField(DatesetHandle handle,
  const char* field_name,
  uint64_t* out_len,
  const void** out_ptr,
  int* out_type);

/*!
* \brief get number of data.
* \param handle the handle to the dataset
* \param out The address to hold number of data
* \return 0 when success, -1 when failure happens
*/
DllExport int LGBM_DatasetGetNumData(DatesetHandle handle,
  uint64_t* out);

/*!
* \brief get number of features
* \param handle the handle to the dataset
* \param out The output of number of features
* \return 0 when success, -1 when failure happens
*/
DllExport int LGBM_DatasetGetNumFeature(DatesetHandle handle,
  uint64_t* out);

// --- start Booster interfaces

/*!
* \brief create an new boosting learner
* \param train_data traning data set
* \param valid_datas validation data sets
* \param valid_names names of validation data sets
* \param n_valid_datas number of validation set
* \param parameters format: 'key1=value1 key2=value2'
* \prama out handle of created Booster
* \return 0 when success, -1 when failure happens
*/
DllExport int LGBM_BoosterCreate(DatesetHandle train_data,
  DatesetHandle valid_datas[],
  const char* valid_names[],
  int n_valid_datas,
  const char* parameters,
  BoosterHandle* out);

/*!
* \brief load an exsiting boosting from model file
* \param filename filename of model
* \return 0 when success, -1 when failure happens
*/
DllExport int LGBM_BoosterLoadFromModelfile(
  const char* filename,
  BoosterHandle* out);

/*!
* \brief free obj in handle
* \param handle handle to be freed
* \return 0 when success, -1 when failure happens
*/
DllExport int LGBM_BoosterFree(BoosterHandle handle);

/*!
* \brief update the model in one round
* \param handle handle
* \param is_finished 1 means finised
* \return 0 when success, -1 when failure happens
*/
DllExport int LGBM_BoosterUpdateOneIter(BoosterHandle handle, int* is_finished);

/*!
* \brief update the model, by directly specify gradient and second order gradient,
*       this can be used to support customized loss function
* \param handle handle
* \param grad gradient statistics
* \param hess second order gradient statistics
* \param float_type 0 for float_32, 1 for float_64
* \param is_finished 1 means finised
* \return 0 when success, -1 when failure happens
*/
DllExport int LGBM_BoosterUpdateOneIterCustom(BoosterHandle handle,
  float* grad,
  float* hess,
  int* is_finished);

/*!
* \brief get evaluation for training data and validation datas
* \param handle handle
* \param data 0:training data, 1: 1st valid data, 2:2nd valid data ...
* \param out_result the string containing evaluation statistics
* \return 0 when success, -1 when failure happens
*/
DllExport int LGBM_BoosterEval(BoosterHandle handle,
  int data,
  const char** out_result);

/*!
* \brief make prediction for training data and validation datas
this can be used to support customized eval function
* \param handle handle
* \param data 0:training data, 1: 1st valid data, 2:2nd valid data ...
* \param predict_type
*          0:raw score
*          1:with sigmoid/softmax transform(if needed)
*          2:leaf index
* \param out_result used to set a pointer to array
* \return 0 when success, -1 when failure happens
*/
DllExport int LGBM_BoosterPredict(BoosterHandle handle,
  int data,
  int predict_type,
  const float** out_result);

/*!
* \brief make prediction for an new data set
* \param handle handle
* \param indptr pointer to row headers
* \param indices findex
* \param data fvalue
* \param nindptr number of rows in the matix + 1
* \param nelem number of nonzero elements in the matrix
* \param num_col number of columns; when it's set to 0, then guess from data
* \param predict_type
*          0:raw score
*          1:with sigmoid transform(if needed)
*          2:leaf index
* \param n_used_trees number of used tree
* \param out_result used to set a pointer to array
* \return 0 when success, -1 when failure happens
*/
DllExport int LGBM_BoosterPredictForCSR(BoosterHandle handle,
  const int32_t* indptr,
  const int32_t* indices,
  const void* data,
  int float_type,
  uint64_t nindptr,
  uint64_t nelem,
  uint64_t num_col,
  int predict_type,
  uint64_t n_used_trees,
  const double** out_result);

/*!
* \brief make prediction for an new data set
* \param handle handle
* \param col_ptr pointer to col headers
* \param indices findex
* \param data fvalue
* \param nindptr number of rows in the matix + 1
* \param nelem number of nonzero elements in the matrix
* \param num_row number of rows; when it's set to 0, then guess from data
* \param predict_type
*          0:raw score
*          1:with sigmoid transform(if needed)
*          2:leaf index
* \param n_used_trees number of used tree
* \param out_result used to set a pointer to array
* \return 0 when success, -1 when failure happens
*/
DllExport int LGBM_BoosterPredictForCSC(BoosterHandle handle,
  const int32_t* col_ptr,
  const int32_t* indices,
  const void* data,
  int float_type,
  uint64_t nindptr,
  uint64_t nelem,
  uint64_t num_row,
  int predict_type,
  uint64_t n_used_trees,
  const double** out_result);

/*!
* \brief make prediction for an new data set
* \param handle handle
* \param data pointer to the data space
* \param nrow number of rows
* \param ncol number columns
* \param missing which value to represent missing value
* \param predict_type
*          0:raw score
*          1:with sigmoid transform(if needed)
*          2:leaf index
* \param n_used_trees number of used tree
* \param out_result used to set a pointer to array
* \return 0 when success, -1 when failure happens
*/
DllExport int LGBM_BoosterPredictForMat(BoosterHandle handle,
  const void* data,
  int float_type,
  int32_t nrow,
  int32_t ncol,
  int predict_type,
  uint64_t n_used_trees,
  const double** out_result);

/*!
* \brief save model into file
* \param handle handle
* \param is_finished 1 means finised
* \param filename file name
* \return 0 when success, -1 when failure happens
*/
DllExport int LGBM_BoosterSaveModel(BoosterHandle handle,
  int is_finished,
  const char* filename);

#endif // LIGHTGBM_C_API_H_
