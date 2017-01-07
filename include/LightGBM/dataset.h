#ifndef LIGHTGBM_DATASET_H_
#define LIGHTGBM_DATASET_H_

#include <LightGBM/utils/random.h>
#include <LightGBM/utils/text_reader.h>

#include <LightGBM/meta.h>
#include <LightGBM/config.h>
#include <LightGBM/feature.h>

#include <vector>
#include <utility>
#include <functional>
#include <string>
#include <unordered_set>
#include <mutex>

namespace LightGBM {

/*! \brief forward declaration */
class DatasetLoader;

/*!
* \brief This class is used to store some meta(non-feature) data for training data,
*        e.g. labels, weights, initial scores, qurey level informations.
*
*        Some details:
*        1. Label, used for traning.
*        2. Weights, weighs of records, optional
*        3. Query Boundaries, necessary for lambdarank.
*           The documents of i-th query is in [ query_boundarise[i], query_boundarise[i+1] )
*        4. Query Weights, auto calculate by weights and query_boundarise(if both of them are existed)
*           the weight for i-th query is sum(query_boundarise[i] , .., query_boundarise[i+1]) / (query_boundarise[i + 1] -  query_boundarise[i+1])
*        5. Initial score. optional. if exsitng, the model will boost from this score, otherwise will start from 0.
*/
class Metadata {
public:
 /*!
  * \brief Null costructor
  */
  Metadata();
  /*!
  * \brief Initialization will load qurey level informations, since it is need for sampling data
  * \param data_filename Filename of data
  * \param init_score_filename Filename of initial score
  */
  void Init(const char* data_filename);
  /*!
  * \brief init as subset
  * \param metadata Filename of data
  * \param used_indices 
  * \param num_used_indices
  */
  void Init(const Metadata& metadata, const data_size_t* used_indices, data_size_t num_used_indices);
  /*!
  * \brief Initial with binary memory
  * \param memory Pointer to memory
  */
  void LoadFromMemory(const void* memory);
  /*! \brief Destructor */
  ~Metadata();

  /*!
  * \brief Initial work, will allocate space for label, weight(if exists) and query(if exists)
  * \param num_data Number of training data
  * \param weight_idx Index of weight column, < 0 means doesn't exists
  * \param query_idx Index of query id column, < 0 means doesn't exists
  */
  void Init(data_size_t num_data, int weight_idx, int query_idx);

  /*!
  * \brief Partition label by used indices
  * \param used_indices Indice of local used
  */
  void PartitionLabel(const std::vector<data_size_t>& used_indices);

  /*!
  * \brief Partition meta data according to local used indices if need
  * \param num_all_data Number of total training data, including other machines' data on parallel learning
  * \param used_data_indices Indices of local used training data
  */
  void CheckOrPartition(data_size_t num_all_data,
    const std::vector<data_size_t>& used_data_indices);

  void SetLabel(const float* label, data_size_t len);

  void SetWeights(const float* weights, data_size_t len);

  void SetQuery(const data_size_t* query, data_size_t len);

  void SetQueryId(const data_size_t* query_id, data_size_t len);

  /*!
  * \brief Set initial scores
  * \param init_score Initial scores, this class will manage memory for init_score.
  */
  void SetInitScore(const float* init_score, data_size_t len);


  /*!
  * \brief Save binary data to file
  * \param file File want to write
  */
  void SaveBinaryToFile(FILE* file) const;

  /*!
  * \brief Get sizes in byte of this object
  */
  size_t SizesInByte() const;

  /*!
  * \brief Get pointer of label
  * \return Pointer of label
  */
  inline const float* label() const { return label_.data(); }

  /*!
  * \brief Set label for one record
  * \param idx Index of this record
  * \param value Label value of this record
  */
  inline void SetLabelAt(data_size_t idx, float value)
  {
    label_[idx] = value;
  }

  /*!
  * \brief Set Weight for one record
  * \param idx Index of this record
  * \param value Weight value of this record
  */
  inline void SetWeightAt(data_size_t idx, float value)
  {
    weights_[idx] = value;
  }

  /*!
  * \brief Set Query Id for one record
  * \param idx Index of this record
  * \param value Query Id value of this record
  */
  inline void SetQueryAt(data_size_t idx, data_size_t value)
  {
    queries_[idx] = static_cast<data_size_t>(value);
  }

  /*!
  * \brief Get weights, if not exists, will return nullptr
  * \return Pointer of weights
  */
  inline const float* weights() const {
    if (!weights_.empty()) {
      return weights_.data();
    } else {
      return nullptr;
    }
  }

  /*!
  * \brief Get data boundaries on queries, if not exists, will return nullptr
  *        we assume data will order by query, 
  *        the interval of [query_boundaris[i], query_boundaris[i+1])
  *        is the data indices for query i.
  * \return Pointer of data boundaries on queries
  */
  inline const data_size_t* query_boundaries() const { 
    if (!query_boundaries_.empty()) {
      return query_boundaries_.data();
    } else {
      return nullptr;
    }
  }

  /*!
  * \brief Get Number of queries
  * \return Number of queries
  */
  inline const data_size_t num_queries() const { return num_queries_; }

  /*!
  * \brief Get weights for queries, if not exists, will return nullptr
  * \return Pointer of weights for queries
  */
  inline const float* query_weights() const { 
    if (!query_weights_.empty()) {
      return query_weights_.data();
    } else {
      return nullptr;
    }
  }

  /*!
  * \brief Get initial scores, if not exists, will return nullptr
  * \return Pointer of initial scores
  */
  inline const float* init_score() const { 
    if (!init_score_.empty()) {
      return init_score_.data();
    } else {
      return nullptr;
    }
  }

  /*!
  * \brief Get size of initial scores
  */
  inline data_size_t num_init_score() const { return num_init_score_; }

  /*! \brief Disable copy */
  Metadata& operator=(const Metadata&) = delete;
  /*! \brief Disable copy */
  Metadata(const Metadata&) = delete;

private:
  /*! \brief Load initial scores from file */
  void LoadInitialScore();
  /*! \brief Load wights from file */
  void LoadWeights();
  /*! \brief Load query boundaries from file */
  void LoadQueryBoundaries();
  /*! \brief Load query wights */
  void LoadQueryWeights();
  /*! \brief Filename of current data */
  const char* data_filename_;
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Number of weights, used to check correct weight file */
  data_size_t num_weights_;
  /*! \brief Label data */
  std::vector<float> label_;
  /*! \brief Weights data */
  std::vector<float> weights_;
  /*! \brief Query boundaries */
  std::vector<data_size_t> query_boundaries_;
  /*! \brief Query weights */
  std::vector<float> query_weights_;
  /*! \brief Number of querys */
  data_size_t num_queries_;
  /*! \brief Number of Initial score, used to check correct weight file */
  data_size_t num_init_score_;
  /*! \brief Initial score */
  std::vector<float> init_score_;
  /*! \brief Queries data */
  std::vector<data_size_t> queries_;
  /*! \brief mutex for threading safe call */
  std::mutex mutex_;
};


/*! \brief Interface for Parser */
class Parser {
public:

  /*! \brief virtual destructor */
  virtual ~Parser() {}

  /*!
  * \brief Parse one line with label
  * \param str One line record, string format, should end with '\0'
  * \param out_features Output columns, store in (column_idx, values)
  * \param out_label Label will store to this if exists
  */
  virtual void ParseOneLine(const char* str,
    std::vector<std::pair<int, double>>* out_features, double* out_label) const = 0;

  /*!
  * \brief Create a object of parser, will auto choose the format depend on file
  * \param filename One Filename of data
  * \param num_features Pass num_features of this data file if you know, <=0 means don't know
  * \param label_idx index of label column
  * \return Object of parser
  */
  static Parser* CreateParser(const char* filename, bool has_header, int num_features, int label_idx);
};

/*! \brief The main class of data set,
*          which are used to traning or validation
*/
class Dataset {
public:
  friend DatasetLoader;

  Dataset();

  Dataset(data_size_t num_data);

  /*! \brief Destructor */
  ~Dataset();

  bool CheckAlign(const Dataset& other) const {
    if (num_features_ != other.num_features_) {
      return false;
    }
    if (num_total_features_ != other.num_total_features_) {
      return false;
    }
    if (label_idx_ != other.label_idx_) {
      return false;
    }
    for (int i = 0; i < num_features_; ++i) {
      if (!features_[i]->CheckAlign(*(other.features_[i].get()))) {
        return false;
      }
    }
    return true;
  }

  inline void PushOneRow(int tid, data_size_t row_idx, const std::vector<double>& feature_values) {
    for (size_t i = 0; i < feature_values.size() && i < static_cast<size_t>(num_total_features_); ++i) {
      int feature_idx = used_feature_map_[i];
      if (feature_idx >= 0) {
        features_[feature_idx]->PushData(tid, row_idx, feature_values[i]);
      }
    }
  }

  inline void PushOneRow(int tid, data_size_t row_idx, const std::vector<std::pair<int, double>>& feature_values) {
    for (auto& inner_data : feature_values) {
      if (inner_data.first >= num_total_features_) { continue; }
      int feature_idx = used_feature_map_[inner_data.first];
      if (feature_idx >= 0) {
        features_[feature_idx]->PushData(tid, row_idx, inner_data.second);
      }
    }
  }

  inline int GetInnerFeatureIndex(int col_idx) const {
    return used_feature_map_[col_idx];
  }

  Dataset* Subset(const data_size_t* used_indices, data_size_t num_used_indices, bool is_enable_sparse) const;

  void FinishLoad();

  bool SetFloatField(const char* field_name, const float* field_data, data_size_t num_element);

  bool SetIntField(const char* field_name, const int* field_data, data_size_t num_element);

  bool GetFloatField(const char* field_name, data_size_t* out_len, const float** out_ptr);

  bool GetIntField(const char* field_name, data_size_t* out_len, const int** out_ptr);

  /*!
  * \brief Save current dataset into binary file, will save to "filename.bin"
  */
  void SaveBinaryFile(const char* bin_filename);

  void CopyFeatureMapperFrom(const Dataset* dataset, bool is_enable_sparse);

  /*!
  * \brief Get a feature pointer for specific index
  * \param i Index for feature
  * \return Pointer of feature
  */
  inline Feature* FeatureAt(int i) const { return features_[i].get(); }

  /*!
  * \brief Get meta data pointer
  * \return Pointer of meta data
  */
  inline const Metadata& metadata() const { return metadata_; }

  /*! \brief Get Number of used features */
  inline int num_features() const { return num_features_; }

  /*! \brief Get Number of total features */
  inline int num_total_features() const { return num_total_features_; }

  /*! \brief Get the index of label column */
  inline int label_idx() const { return label_idx_; }

  /*! \brief Get names of current data set */
  inline const std::vector<std::string>& feature_names() const { return feature_names_; }

  inline void set_feature_names(const std::vector<std::string>& feature_names) {
    if (feature_names.size() != static_cast<size_t>(num_total_features_)) {
      Log::Warning("size of feature_names error, should equal with total number of features");
      return;
    }
    feature_names_ = std::vector<std::string>(feature_names);
  }

  /*! \brief Get Number of data */
  inline data_size_t num_data() const { return num_data_; }

  /*! \brief Disable copy */
  Dataset& operator=(const Dataset&) = delete;
  /*! \brief Disable copy */
  Dataset(const Dataset&) = delete;

private:
  const char* data_filename_;
  /*! \brief Store used features */
  std::vector<std::unique_ptr<Feature>> features_;
  /*! \brief Mapper from real feature index to used index*/
  std::vector<int> used_feature_map_;
  /*! \brief Number of used features*/
  int num_features_;
  /*! \brief Number of total features*/
  int num_total_features_;
  /*! \brief Number of total data*/
  data_size_t num_data_;
  /*! \brief Store some label level data*/
  Metadata metadata_;
  /*! \brief index of label column */
  int label_idx_ = 0;
  /*! \brief store feature names */
  std::vector<std::string> feature_names_;
  /*! \brief store feature names */
  static const char* binary_file_token;
};

}  // namespace LightGBM

#endif   // LightGBM_DATA_H_
