#ifndef LIGHTGBM_DATA_H_
#define LIGHTGBM_DATA_H_

#include <LightGBM/utils/random.h>
#include <LightGBM/utils/text_reader.h>

#include <LightGBM/meta.h>

#include <vector>
#include <utility>
#include <functional>
#include <string>

namespace LightGBM {

/*! \brief forward declaration */
class Feature;

/*!
* \brief This class is used to store some meta(non-feature) data for tranining data,
*        e.g. labels, weights, initial scores, qurey level informations.
*
* Some details:
* 1. Label, used for traning.
* 2. Weights, weighs of record, optional
* 3. Query Boundaries, necessary for lambdarank.
*    The documents of i-th query is in [ query_boundarise[i], query_boundarise[i+1] )
* 4. Query Weights, auto calculate by weights and query_boundarise(if both of them are existed)
*    the weight for i-th query is sum(query_boundarise[i] , .., query_boundarise[i+1]) / (query_boundarise[i + 1] -  query_boundarise[i+1])
* 5. Initial score. optional. if exsitng, the model will boost from this score, otherwise will start from 0.
*/
class Metadata {
public:
 /*!
  * \brief Null costructor
  */
  Metadata();
  /*!
  * \brief Initialize, will load qurey level informations, since it is need for sampling data
  * \param data_filename Filename of data
  * \param init_score_filename Filename of initial score
  * \param is_int_label True if label is int type
  */
  void Init(const char* data_filename, const char* init_score_filename);
  /*!
  * \brief Initialize, only load initial score
  * \param init_score_filename Filename of initial score
  */
  void Init(const char* init_score_filename);
  /*!
  * \brief Initial with binary memory
  * \param memory Pointer to memory
  */
  void LoadFromMemory(const void* memory);
  /*! \brief Destructor */
  ~Metadata();

  /*!
  * \brief Initial work, will auto load weight, inital scores
  * \param num_data Number of training data
  */
  void InitLabel(data_size_t num_data);

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

  /*!
  * \brief Set initial scores
  * \param init_score Initial scores, this class will manage memory for init_score.
  */
  void SetInitScore(score_t* init_score);


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
  inline const float* label() const { return label_; }

  /*!
  * \brief Set label for one record
  * \param idx Index of this record
  * \param value Label value of this record
  */
  inline void SetLabelAt(data_size_t idx, double value)
  {
    label_[idx] = static_cast<float>(value);
  }

  /*!
  * \brief Get weights, if not exists, will return nullput
  * \return Pointer of weights
  */
  inline const float* weights()
            const { return weights_; }

  /*!
  * \brief Get data boundaries on queries, if not exists, will return nullput
  *        we assume data will order by query, 
  *        the interval of [query_boundaris[i], query_boundaris[i+1])
  *        is the data indices for query i.
  * \return Pointer of data boundaries on queries
  */
  inline const data_size_t* query_boundaries()
           const { return query_boundaries_; }

  /*!
  * \brief Get Number of queries
  * \return Number of queries
  */
  inline const data_size_t num_queries() const { return num_queries_; }

  /*!
  * \brief Get weights for queries, if not exists, will return nullput
  * \return Pointer of weights for queries
  */
  inline const float* query_weights() const { return query_weights_; }

  /*!
  * \brief Get initial scores, if not exists, will return nullput
  * \return Pointer of initial scores
  */
  inline const score_t* init_score() const { return init_score_; }

  /*! \brief Load initial scores from file */
  void LoadInitialScore();

private:
  /*! \brief Load wights from file */
  void LoadWeights();
  /*! \brief Load query boundaries from file */
  void LoadQueryBoundaries();
  /*! \brief Load query wights */
  void LoadQueryWeights();
  /*! \brief Filename of current data */
  const char* data_filename_;
  /*! \brief Filename of initial scores */
  const char* init_score_filename_;
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Number of weights, used to check correct weight file */
  data_size_t num_weights_;
  /*! \brief Label data */
  float* label_;
  /*! \brief Label data, int type */
  int16_t* label_int_;
  /*! \brief Weights data */
  float* weights_;
  /*! \brief Query boundaries */
  data_size_t* query_boundaries_;
  /*! \brief Query weights */
  float* query_weights_;
  /*! \brief Number of querys */
  data_size_t num_queries_;
  /*! \brief Number of Initial score, used to check correct weight file */
  data_size_t num_init_score_;
  /*! \brief Initial score */
  score_t* init_score_;
};


/*! \brief Interface for Parser */
class Parser {
public:
  /*! \brief virtual destructor */
  virtual ~Parser() {}
  /*!
  * \brief Parse one line with label
  * \param str One line record, string format, should end with '\0'
  * \param out_features Output features, store in (feature_idx, feature_value)
  * \param out_label Output label
  */
  virtual void ParseOneLine(const char* str,
    std::vector<std::pair<int, double>>* out_features,
    double* out_label) const = 0;

  /*!
  * \brief Parse one line with label
  * \param str One line record, string format, should end with '\0'
  * \param out_features Output features, store in (feature_idx, feature_value)
  * \param out_label Output label
  */
  virtual void ParseOneLine(const char* str,
    std::vector<std::pair<int, double>>* out_features) const = 0;

  /*!
  * \brief Create a object of parser, will auto choose the format depend on file
  * \param filename One Filename of data
  * \param num_features Pass num_features of this data file if you know, <=0 means don't know
  * \param has_label output, if num_features > 0, will output this data has label or not
  * \return Object of parser
  */
  static Parser* CreateParser(const char* filename, int num_features, bool* has_label);
};

using PredictFunction =
  std::function<double(const std::vector<std::pair<int, double>>&)>;

/*! \brief The main class of data set,
*          which are used to traning or validation
*/
class Dataset {
public:
  /*!
  * \brief Constructor
  * \param data_filename Filename of dataset
  * \param init_score_filename Filename of initial score
  * \param is_int_label True if label is int type
  * \param max_bin The maximal number of bin that feature values will bucket in
  * \param random_seed The seed for random generator
  * \param is_enable_sparse True for sparse feature
  * \param predict_fun Used for initial model, will give a prediction score based on this function, thenn set as initial score
  */
  Dataset(const char* data_filename, const char* init_score_filename,
    int max_bin, int random_seed, bool is_enable_sparse, const PredictFunction& predict_fun);

  /*!
  * \brief Constructor
  * \param data_filename Filename of dataset
  * \param is_int_label True if label is int type
  * \param max_bin The maximal number of bin that feature values will bucket in
  * \param random_seed The seed for random generator
  * \param is_enable_sparse True for sparse feature
  * \param predict_fun Used for initial model, will give a prediction score based on this function, thenn set as initial score
  */
  Dataset(const char* data_filename,
    int max_bin, int random_seed, bool is_enable_sparse,
                     const PredictFunction& predict_fun)
    : Dataset(data_filename, "", max_bin, random_seed,
                                    is_enable_sparse, predict_fun) {
  }

  /*! \brief Destructor */
  ~Dataset();

  /*!
  * \brief Load training data on parallel training
  * \param rank Rank of local machine
  * \param num_machines Total number of all machines
  * \param is_pre_partition True if data file is pre-partitioned
  * \param use_two_round_loading True if need to use two round loading
  */
  void LoadTrainData(int rank, int num_machines, bool is_pre_partition,
                                           bool use_two_round_loading);

  /*!
  * \brief Load training data on single machine training
  * \param use_two_round_loading True if need to use two round loading
  */
  inline void LoadTrainData(bool use_two_round_loading) {
    LoadTrainData(0, 1, false, use_two_round_loading);
  }

  /*!
  * \brief Load data and use bin mapper from other data set, general this function is used to extract feature for validation data
  * \param train_set Other loaded data set
  * \param use_two_round_loading True if need to use two round loading
  */
  void LoadValidationData(const Dataset* train_set, bool use_two_round_loading);

  /*!
  * \brief Save current dataset into binary file, will save to "filename.bin"
  */
  void SaveBinaryFile();

  /*!
  * \brief Get a feature pointer for specific index
  * \param i Index for feature
  * \return Pointer of feature
  */
  inline const Feature* FeatureAt(int i) const { return features_[i]; }

  /*!
  * \brief Get meta data pointer
  * \return Pointer of meta data
  */
  inline const Metadata& metadata() const { return metadata_; }

  /*! \brief Get Number of used features */
  inline int num_features() const { return num_features_; }

  /*! \brief Get Number of total features */
  inline int num_total_features() const { return num_total_features_; }

  /*! \brief Get Number of data */
  inline data_size_t num_data() const { return num_data_; }

  /*! \brief Disable copy */
  Dataset& operator=(const Dataset&) = delete;
  /*! \brief Disable copy */
  Dataset(const Dataset&) = delete;

private:
  /*!
  * \brief Load data content on memory. if num_machines > 1 and !is_pre_partition, will partition data
  * \param rank Rank of local machine
  * \param num_machines Total number of all machines
  * \param is_pre_partition True if data file is pre-partitioned
  */
  void LoadDataToMemory(int rank, int num_machines, bool is_pre_partition);

  /*!
  * \brief Sample data from memory, need load data to memory first
  * \param out_data Store the sampled data
  */
  void SampleDataFromMemory(std::vector<std::string>* out_data);

  /*!
  * \brief Sample data from file
  * \param rank Rank of local machine
  * \param num_machines Total number of all machines
  * \param is_pre_partition True if data file is pre-partitioned
  * \param out_data Store the sampled data
  */
  void SampleDataFromFile(int rank, int num_machines,
    bool is_pre_partition, std::vector<std::string>* out_data);

  /*!
  * \brief Get feature bin mapper from sampled data.
  * if num_machines > 1, differnt machines will construct bin mapper for different features, then have a global sync up
  * \param rank Rank of local machine
  * \param num_machines Total number of all machines
  */
  void ConstructBinMappers(int rank, int num_machines,
         const std::vector<std::string>& sample_data);

  /*! \brief Extract local features from memory */
  void ExtractFeaturesFromMemory();

  /*! \brief Extract local features from file */
  void ExtractFeaturesFromFile();

  /*! \brief Check can load from binary file */
  void CheckCanLoadFromBin();

  /*!
  * \brief Load data set from binary file
  * \param rank Rank of local machine
  * \param num_machines Total number of all machines
  * \param is_pre_partition True if data file is pre-partitioned
  */
  void LoadDataFromBinFile(int rank, int num_machines, bool is_pre_partition);

  /*! \brief Check this data set is null or not */
  void CheckDataset();

  /*! \brief Filename of data */
  const char* data_filename_;
  /*! \brief A reader class that can read text data */
  TextReader<data_size_t>* text_reader_;
  /*! \brief A parser class that can parse data */
  Parser* parser_;
  /*! \brief Store used features */
  std::vector<Feature*> features_;
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
  /*! \brief Random generator*/
  Random random_;
  /*! \brief The maximal number of bin that feature values will bucket in */
  int max_bin_;
  /*! \brief True if enable sparse */
  bool is_enable_sparse_;
  /*! \brief True if dataset is loaded from binary file */
  bool is_loading_from_binfile_;
  /*! \brief Number of global data, used for distributed learning */
  size_t global_num_data_ = 0;
  // used to local used data indices
  std::vector<data_size_t> used_data_indices_;
  // prediction function for initial model
  const PredictFunction& predict_fun_;
};

}  // namespace LightGBM

#endif   // LightGBM_DATA_H_
