#include <LightGBM/dataset.h>

#include <LightGBM/feature.h>
#include <LightGBM/network.h>

#include <omp.h>

#include <cstdio>
#include <unordered_map>
#include <limits>
#include <vector>
#include <utility>
#include <string>

namespace LightGBM {

Dataset::Dataset(const char* data_filename, const char* init_score_filename,
                 int max_bin, int random_seed, bool is_enable_sparse, const PredictFunction& predict_fun)
  :data_filename_(data_filename), random_(random_seed),
  max_bin_(max_bin), is_enable_sparse_(is_enable_sparse), predict_fun_(predict_fun) {

  CheckCanLoadFromBin();
  if (is_loading_from_binfile_ && predict_fun != nullptr) {
    Log::Stdout("cannot perform initial prediction for binary file, will use text file instead");
    is_loading_from_binfile_ = false;
  }

  if (!is_loading_from_binfile_) {
    // load weight, query information and initilize score
    metadata_.Init(data_filename, init_score_filename);
    // create text parser
    parser_ = Parser::CreateParser(data_filename_, 0, nullptr);
    if (parser_ == nullptr) {
      Log::Stderr("cannot recognise input data format, filename: %s", data_filename_);
    }
    // create text reader
    text_reader_ = new TextReader<data_size_t>(data_filename);
  } else {
    // only need to load initilize score, other meta data will load from bin flie
    metadata_.Init(init_score_filename);
    Log::Stdout("will load data set from binary file");
    parser_ = nullptr;
    text_reader_ = nullptr;
  }

}

Dataset::~Dataset() {
  if (parser_ != nullptr) { delete parser_; }
  if (text_reader_ != nullptr) { delete text_reader_; }
  for (auto& feature : features_) {
    delete feature;
  }
  features_.clear();
}

void Dataset::LoadDataToMemory(int rank, int num_machines, bool is_pre_partition) {
  used_data_indices_.clear();
  if (num_machines == 1 || is_pre_partition) {
    // read all lines
    num_data_ = text_reader_->ReadAllLines();
    global_num_data_ = num_data_;
  } else {  // need partition data
    // get query data
    const data_size_t* query_boundaries = metadata_.query_boundaries();

    if (query_boundaries == nullptr) {
      // if not contain query data, minimal sample unit is one record
      global_num_data_ = text_reader_->ReadAndFilterLines([this, rank, num_machines](data_size_t) {
        if (random_.NextInt(0, num_machines) == rank) {
          return true;
        } else {
          return false;
        }
      }, &used_data_indices_);
    } else {
      // if contain query data, minimal sample unit is one query
      data_size_t num_queries = metadata_.num_queries();
      data_size_t qid = -1;
      bool is_query_used = false;
      global_num_data_ = text_reader_->ReadAndFilterLines(
        [this, rank, num_machines, &qid, &query_boundaries, &is_query_used, num_queries]
      (data_size_t line_idx) {
        if (qid >= num_queries) {
          Log::Stderr("current query is exceed the range of query file, please ensure your query file is correct");
        }
        if (line_idx >= query_boundaries[qid + 1]) {
          // if is new query
          is_query_used = false;
          if (random_.NextInt(0, num_machines) == rank) {
            is_query_used = true;
          }
          ++qid;
        }
        return is_query_used;
      }, &used_data_indices_);
    }
    // set number of data
    num_data_ = static_cast<data_size_t>(used_data_indices_.size());
  }
}

void Dataset::SampleDataFromMemory(std::vector<std::string>* out_data) {
  const size_t sample_cnt = static_cast<size_t>(num_data_ < 50000 ? num_data_ : 50000);
  std::vector<size_t> sample_indices = random_.Sample(num_data_, sample_cnt);
  out_data->clear();
  for (size_t i = 0; i < sample_indices.size(); ++i) {
    const size_t idx = sample_indices[i];
    out_data->push_back(text_reader_->Lines()[idx]);
  }
}

void Dataset::SampleDataFromFile(int rank, int num_machines, bool is_pre_partition,
                                             std::vector<std::string>* out_data) {
  used_data_indices_.clear();
  const size_t sample_cnt = 50000;
  if (num_machines == 1 || is_pre_partition) {
    num_data_ = static_cast<data_size_t>(text_reader_->SampleFromFile(random_, sample_cnt, out_data));
    global_num_data_ = num_data_;
  } else {  // need partition data
    // get query data
    const data_size_t* query_boundaries = metadata_.query_boundaries();
    if (query_boundaries == nullptr) {
      // if not contain query file, minimal sample unit is one record
      global_num_data_ = text_reader_->SampleAndFilterFromFile([this, rank, num_machines]
      (data_size_t) {
        if (random_.NextInt(0, num_machines) == rank) {
          return true;
        } else {
          return false;
        }
      }, &used_data_indices_, random_, sample_cnt, out_data);
    } else {
      // if contain query file, minimal sample unit is one query
      data_size_t num_queries = metadata_.num_queries();
      data_size_t qid = -1;
      bool is_query_used = false;
      global_num_data_ = text_reader_->SampleAndFilterFromFile(
        [this, rank, num_machines, &qid, &query_boundaries, &is_query_used, num_queries]
      (data_size_t line_idx) {
        if (qid >= num_queries) {
          Log::Stderr("current query is exceed the range of query file, \
                             please ensure your query file is correct");
        }
        if (line_idx >= query_boundaries[qid + 1]) {
          // if is new query
          is_query_used = false;
          if (random_.NextInt(0, num_machines) == rank) {
            is_query_used = true;
          }
          ++qid;
        }
        return is_query_used;
      }, &used_data_indices_, random_, sample_cnt, out_data);
    }
    num_data_ = static_cast<data_size_t>(used_data_indices_.size());
  }
}

void Dataset::ConstructBinMappers(int rank, int num_machines, const std::vector<std::string>& sample_data) {
  // sample_values[i][j], means the value of j-th sample on i-th feature
  std::vector<std::vector<double>> sample_values;
  // temp buffer for one line features and label
  std::vector<std::pair<int, double>> oneline_features;
  double label;
  for (size_t i = 0; i < sample_data.size(); ++i) {
    oneline_features.clear();
    // parse features
    parser_->ParseOneLine(sample_data[i].c_str(), &oneline_features, &label);
    // push 0 first, then edit the value according existing feature values
    for (auto& feature_values : sample_values) {
      feature_values.push_back(0.0);
    }
    for (std::pair<int, double>& inner_data : oneline_features) {
      if (static_cast<size_t>(inner_data.first) >= sample_values.size()) {
        // if need expand feature set
        size_t need_size = inner_data.first - sample_values.size() + 1;
        for (size_t j = 0; j < need_size; ++j) {
          // push i+1 0
          sample_values.emplace_back(i + 1, 0.0);
        }
      }
      // edit the feature value
      sample_values[inner_data.first][i] = inner_data.second;
    }
  }

  features_.clear();

  // -1 means doesn't use this feature
  used_feature_map_ = std::vector<int>(sample_values.size(), -1);
  num_total_features_ = static_cast<int>(sample_values.size());
  // start find bins
  if (num_machines == 1) {
    std::vector<BinMapper*> bin_mappers(sample_values.size());
    // if only 1 machines, find bin locally
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < static_cast<int>(sample_values.size()); ++i) {
      bin_mappers[i] = new BinMapper();
      bin_mappers[i]->FindBin(&sample_values[i], max_bin_);
    }

    for (size_t i = 0; i < sample_values.size(); ++i) {
      if (!bin_mappers[i]->is_trival()) {
        // map real feature index to used feature index
        used_feature_map_[i] = static_cast<int>(features_.size());
        // push new feature
        features_.push_back(new Feature(static_cast<int>(i), bin_mappers[i],
                                             num_data_, is_enable_sparse_));
      } else {
        // if feature is trival(only 1 bin), free spaces
        Log::Stdout("Warning: feature %d only contains one value, will ignore it", i);
        delete bin_mappers[i];
      }
    }
  } else {
    // if have multi-machines, need find bin distributed
    // different machines will find bin for different features

    // start and len will store the process feature indices for different machines
    // machine i will find bins for features in [ strat[i], start[i] + len[i] )
    int* start = new int[num_machines];
    int* len = new int[num_machines];
    int total_num_feature = static_cast<int>(sample_values.size());
    int step = (total_num_feature + num_machines - 1) / num_machines;
    if (step < 1) { step = 1; }

    start[0] = 0;
    for (int i = 0; i < num_machines - 1; ++i) {
      len[i] = Common::Min<int>(step, total_num_feature - start[i]);
      start[i + 1] = start[i] + len[i];
    }
    len[num_machines - 1] = total_num_feature - start[num_machines - 1];
    // get size of bin mapper with max_bin_ size
    int type_size = BinMapper::SizeForSpecificBin(max_bin_);
    // since sizes of different feature may not be same, we expand all bin mapper to type_size 
    int buffer_size = type_size * total_num_feature;
    char* input_buffer = new char[buffer_size];
    char* output_buffer = new char[buffer_size];

    // find local feature bins and copy to buffer
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < len[rank]; ++i) {
      BinMapper* bin_mapper = new BinMapper();
      bin_mapper->FindBin(&sample_values[start[rank] + i], max_bin_);
      bin_mapper->CopyTo(input_buffer + i * type_size);
      // don't need this any more
      delete bin_mapper;
    }
    // convert to binary size
    for (int i = 0; i < num_machines; ++i) {
      start[i] *= type_size;
      len[i] *= type_size;
    }
    // gather global feature bin mappers
    Network::Allgather(input_buffer, buffer_size, start, len, output_buffer);
    // restore features bins from buffer
    for (int i = 0; i < total_num_feature; ++i) {
      BinMapper* bin_mapper = new BinMapper();
      bin_mapper->CopyFrom(output_buffer + i * type_size);
      if (!bin_mapper->is_trival()) {
        used_feature_map_[i] = static_cast<int>(features_.size());
        features_.push_back(new Feature(static_cast<int>(i), bin_mapper, num_data_, is_enable_sparse_));
      } else {
        delete bin_mapper;
      }
    }
    // free buffer
    delete[] start;
    delete[] len;
    delete[] input_buffer;
    delete[] output_buffer;
  }
  num_features_ = static_cast<int>(features_.size());
}


void Dataset::LoadTrainData(int rank, int num_machines, bool is_pre_partition, bool use_two_round_loading) {
  used_data_indices_.clear();
  if (!is_loading_from_binfile_ ) {
    if (!use_two_round_loading) {
      // read data to memory
      LoadDataToMemory(rank, num_machines, is_pre_partition);
      std::vector<std::string> sample_data;
      // sample data
      SampleDataFromMemory(&sample_data);
      // construct feature bin mappers
      ConstructBinMappers(rank, num_machines, sample_data);
      // initialize label
      metadata_.InitLabel(num_data_);
      // extract features
      ExtractFeaturesFromMemory();
    } else {
      std::vector<std::string> sample_data;
      // sample data from file
      SampleDataFromFile(rank, num_machines, is_pre_partition, &sample_data);
      // construct feature bin mappers
      ConstructBinMappers(rank, num_machines, sample_data);
      // initialize label
      metadata_.InitLabel(num_data_);

      // extract features
      ExtractFeaturesFromFile();
    }
  } else {
    // load data from binary file
    LoadDataFromBinFile(rank, num_machines, is_pre_partition);
  }
  // check meta data
  metadata_.CheckOrPartition(static_cast<data_size_t>(global_num_data_), used_data_indices_);
  // free memory
  used_data_indices_.clear();
  used_data_indices_.shrink_to_fit();
  // need to check training data
  CheckDataset();
}

void Dataset::LoadValidationData(const Dataset* train_set, bool use_two_round_loading) {
  used_data_indices_.clear();
  if (!is_loading_from_binfile_ ) {
    if (!use_two_round_loading) {
      // read data in memory
      LoadDataToMemory(0, 1, false);
      // initialize label
      metadata_.InitLabel(num_data_);
      features_.clear();
      // copy feature bin mapper data
      for (Feature* feature : train_set->features_) {
        features_.push_back(new Feature(feature->feature_index(), new BinMapper(*feature->bin_mapper()), num_data_, is_enable_sparse_));
      }
      used_feature_map_ = train_set->used_feature_map_;
      num_features_ = static_cast<int>(features_.size());
      // extract features
      ExtractFeaturesFromMemory();
    } else {
      // Get number of lines of data file
      num_data_ = static_cast<data_size_t>(text_reader_->CountLine());
      // initialize label
      metadata_.InitLabel(num_data_);
      features_.clear();
      // copy feature bin mapper data
      for (Feature* feature : train_set->features_) {
        features_.push_back(new Feature(feature->feature_index(), new BinMapper(*feature->bin_mapper()), num_data_, is_enable_sparse_));
      }
      used_feature_map_ = train_set->used_feature_map_;
      num_features_ = static_cast<int>(features_.size());
      // extract features
      ExtractFeaturesFromFile();
    }
  } else {
    // load from binary file
    LoadDataFromBinFile(0, 1, false);
  }
  // not need to check validation data
  // check meta data
  metadata_.CheckOrPartition(static_cast<data_size_t>(global_num_data_), used_data_indices_);
  // CheckDataset();
}

void Dataset::ExtractFeaturesFromMemory() {
  std::vector<std::pair<int, double>> oneline_features;
  double tmp_label = 0.0;
  if (predict_fun_ == nullptr) {
    // if doesn't need to prediction with initial model
    #pragma omp parallel for schedule(guided) private(oneline_features) firstprivate(tmp_label)
    for (data_size_t i = 0; i < num_data_; ++i) {
      const int tid = omp_get_thread_num();
      oneline_features.clear();
      // parser
      parser_->ParseOneLine(text_reader_->Lines()[i].c_str(), &oneline_features, &tmp_label);
      // set label
      metadata_.SetLabelAt(i, tmp_label);
      // free processed line:
      text_reader_->Lines()[i].clear();
      // shrink_to_fit will be very slow in linux, and seems not free memory, disable for now
      // text_reader_->Lines()[i].shrink_to_fit();
      // push data
      for (auto& inner_data : oneline_features) {
        int feature_idx = used_feature_map_[inner_data.first];
        if (feature_idx >= 0) {
          // if is used feature
          features_[feature_idx]->PushData(tid, i, inner_data.second);
        }
      }
    }
  } else {
    // if need to prediction with initial model
    score_t* init_score = new score_t[num_data_];
    #pragma omp parallel for schedule(guided) private(oneline_features) firstprivate(tmp_label)
    for (data_size_t i = 0; i < num_data_; ++i) {
      const int tid = omp_get_thread_num();
      oneline_features.clear();
      // parser
      parser_->ParseOneLine(text_reader_->Lines()[i].c_str(), &oneline_features, &tmp_label);
      // set initial score
      init_score[i] = static_cast<score_t>(predict_fun_(oneline_features));
      // set label
      metadata_.SetLabelAt(i, tmp_label);
      // free processed line:
      text_reader_->Lines()[i].clear();
      // shrink_to_fit will be very slow in linux, and seems not free memory, disable for now
      // text_reader_->Lines()[i].shrink_to_fit();
      // push data
      for (auto& inner_data : oneline_features) {
        int feature_idx = used_feature_map_[inner_data.first];
        if (feature_idx >= 0) {
          // if is used feature
          features_[feature_idx]->PushData(tid, i, inner_data.second);
        }
      }
    }
    // metadata_ will manage space of init_score
    metadata_.SetInitScore(init_score);
  }

  #pragma omp parallel for schedule(guided)
  for (int i = 0; i < num_features_; i++) {
    features_[i]->FinishLoad();
  }
  // text data can be free after loaded feature values
  text_reader_->Clear();
}


void Dataset::ExtractFeaturesFromFile() {
  score_t* init_score = nullptr;
  if (predict_fun_ != nullptr) {
    init_score = new score_t[num_data_];
  }
  std::function<void(data_size_t, const std::vector<std::string>&)> process_fun =
    [this, &init_score]
  (data_size_t start_idx, const std::vector<std::string>& lines) {
    std::vector<std::pair<int, double>> oneline_features;
    double tmp_label = 0.0;
    #pragma omp parallel for schedule(static) private(oneline_features) firstprivate(tmp_label)
    for (data_size_t i = 0; i < static_cast<data_size_t>(lines.size()); i++) {
      const int tid = omp_get_thread_num();
      oneline_features.clear();
      // parser
      parser_->ParseOneLine(lines[i].c_str(), &oneline_features, &tmp_label);
      // set initial score
      if (init_score != nullptr) {
        init_score[start_idx + i] = static_cast<score_t>(predict_fun_(oneline_features));
      }
      // set label
      metadata_.SetLabelAt(start_idx + i, tmp_label);
      // push data
      for (auto& inner_data : oneline_features) {
        int feature_idx = used_feature_map_[inner_data.first];
        if (feature_idx >= 0) {
          // if is used feature
          features_[feature_idx]->PushData(tid, start_idx + i, inner_data.second);
        }
      }
    }
  };

  if (used_data_indices_.size() > 0) {
    // only need part of data
    text_reader_->ReadPartAndProcessParallel(used_data_indices_, process_fun);
  } else {
    // need full data
    text_reader_->ReadAllAndProcessParallel(process_fun);
  }

  // metadata_ will manage space of init_score
  if (init_score != nullptr) {
    metadata_.SetInitScore(init_score);
  }

  #pragma omp parallel for schedule(guided)
  for (int i = 0; i < num_features_; i++) {
    features_[i]->FinishLoad();
  }
}

void Dataset::SaveBinaryFile() {
  // if is loaded from binary file, not need to save 
  if (!is_loading_from_binfile_) {
    std::string bin_filename(data_filename_);
    bin_filename.append(".bin");
    FILE* file;
    #ifdef _MSC_VER
    fopen_s(&file, bin_filename.c_str(), "wb");
    #else
    file = fopen(bin_filename.c_str(), "wb");
    #endif
    if (file == NULL) {
      Log::Stderr("cannot write binary data to %s ", bin_filename.c_str());
    }

    Log::Stdout("start save binary file for data %s", data_filename_);

    // get size of header
    size_t size_of_header = sizeof(global_num_data_) + sizeof(is_enable_sparse_)
      + sizeof(max_bin_) + sizeof(num_data_) + sizeof(num_features_) + sizeof(size_t) + sizeof(int) * used_feature_map_.size();
    fwrite(&size_of_header, sizeof(size_of_header), 1, file);
    // write header
    fwrite(&global_num_data_, sizeof(global_num_data_), 1, file);
    fwrite(&is_enable_sparse_, sizeof(is_enable_sparse_), 1, file);
    fwrite(&max_bin_, sizeof(max_bin_), 1, file);
    fwrite(&num_data_, sizeof(num_data_), 1, file);
    fwrite(&num_features_, sizeof(num_features_), 1, file);
    size_t num_used_feature_map = used_feature_map_.size();
    fwrite(&num_used_feature_map, sizeof(num_used_feature_map), 1, file);
    fwrite(used_feature_map_.data(), sizeof(int), num_used_feature_map, file);

    // get size of meta data
    size_t size_of_metadata = metadata_.SizesInByte();
    fwrite(&size_of_metadata, sizeof(size_of_metadata), 1, file);
    // write meta data
    metadata_.SaveBinaryToFile(file);

    // write feature data
    for (int i = 0; i < num_features_; ++i) {
      // get size of feature
      size_t size_of_feature = features_[i]->SizesInByte();
      fwrite(&size_of_feature, sizeof(size_of_feature), 1, file);
      // write feature
      features_[i]->SaveBinaryToFile(file);
    }
    fclose(file);
  }
}

void Dataset::CheckCanLoadFromBin() {
  std::string bin_filename(data_filename_);
  bin_filename.append(".bin");

  FILE* file;

  #ifdef _MSC_VER
  fopen_s(&file, bin_filename.c_str(), "rb");
  #else
  file = fopen(bin_filename.c_str(), "rb");
  #endif

  if (file == NULL) {
    is_loading_from_binfile_ = false;
  } else {
    is_loading_from_binfile_ = true;
    fclose(file);
  }
}

void Dataset::LoadDataFromBinFile(int rank, int num_machines, bool is_pre_partition) {
  std::string bin_filename(data_filename_);
  bin_filename.append(".bin");

  FILE* file;

  #ifdef _MSC_VER
  fopen_s(&file, bin_filename.c_str(), "rb");
  #else
  file = fopen(bin_filename.c_str(), "rb");
  #endif

  if (file == NULL) {
    Log::Stderr("cannot read binary data from %s", bin_filename.c_str());
  }

  // buffer to read binary file
  size_t buffer_size = 16 * 1024 * 1024;
  char* buffer = new char[buffer_size];

  // read size of header
  size_t read_cnt = fread(buffer, sizeof(size_t), 1, file);

  if (read_cnt != 1) {
    Log::Stderr("binary file format error at header size");
  }

  size_t size_of_head = *(reinterpret_cast<size_t*>(buffer));

  // re-allocmate space if not enough
  if (size_of_head > buffer_size) {
    delete[] buffer;
    buffer_size = size_of_head;
    buffer = new char[buffer_size];
  }
  // read header
  read_cnt = fread(buffer, 1, size_of_head, file);

  if (read_cnt != size_of_head) {
    Log::Stderr("binary file format error at header");
  }
  // get header 
  const char* mem_ptr = buffer;
  global_num_data_ = *(reinterpret_cast<const size_t*>(mem_ptr));
  mem_ptr += sizeof(global_num_data_);
  is_enable_sparse_ = *(reinterpret_cast<const bool*>(mem_ptr));
  mem_ptr += sizeof(is_enable_sparse_);
  max_bin_ = *(reinterpret_cast<const int*>(mem_ptr));
  mem_ptr += sizeof(max_bin_);
  num_data_ = *(reinterpret_cast<const data_size_t*>(mem_ptr));
  mem_ptr += sizeof(num_data_);
  num_features_ = *(reinterpret_cast<const int*>(mem_ptr));
  mem_ptr += sizeof(num_features_);
  size_t num_used_feature_map = *(reinterpret_cast<const size_t*>(mem_ptr));
  mem_ptr += sizeof(num_used_feature_map);
  const int* tmp_feature_map = reinterpret_cast<const int*>(mem_ptr);
  used_feature_map_.clear();
  for (size_t i = 0; i < num_used_feature_map; ++i) {
    used_feature_map_.push_back(tmp_feature_map[i]);
  }

  // read size of meta data
  read_cnt = fread(buffer, sizeof(size_t), 1, file);

  if (read_cnt != 1) {
    Log::Stderr("binary file format error at size of meta data");
  }

  size_t size_of_metadata = *(reinterpret_cast<size_t*>(buffer));

  // re-allocmate space if not enough
  if (size_of_metadata > buffer_size) {
    delete[] buffer;
    buffer_size = size_of_metadata;
    buffer = new char[buffer_size];
  }
  //  read meta data
  read_cnt = fread(buffer, 1, size_of_metadata, file);

  if (read_cnt != size_of_metadata) {
    Log::Stderr("binary file format error at meta data");
  }
  // load meta data
  metadata_.LoadFromMemory(buffer);

  used_data_indices_.clear();
  global_num_data_ = num_data_;
  // sample local used data if need to partition
  if (num_machines > 1 && !is_pre_partition) {
    const data_size_t* query_boundaries = metadata_.query_boundaries();
    if (query_boundaries == nullptr) {
      // if not contain query file, minimal sample unit is one record
      for (data_size_t i = 0; i < num_data_; i++) {
        if (random_.NextInt(0, num_machines) == rank) {
          used_data_indices_.push_back(i);
        } 
      }
    } else {
      // if contain query file, minimal sample unit is one query
      data_size_t num_queries = metadata_.num_queries();
      data_size_t qid = -1;
      bool is_query_used = false;
      for (data_size_t i = 0; i < num_data_; i++) {
        if (qid >= num_queries) {
          Log::Stderr("current query is exceed the range of query file, please ensure your query file is correct");
        }
        if (i >= query_boundaries[qid + 1]) {
          // if is new query
          is_query_used = false;
          if (random_.NextInt(0, num_machines) == rank) {
            is_query_used = true;
          }
          ++qid;
        }
        if (is_query_used) {
          used_data_indices_.push_back(i);
        }
      }
    }
    num_data_ = static_cast<data_size_t>(used_data_indices_.size());
  }
  metadata_.PartitionLabel(used_data_indices_);
  // read feature data
  for (int i = 0; i < num_features_; ++i) {
    // read feature size
    read_cnt = fread(buffer, sizeof(size_t), 1, file);
    if (read_cnt != 1) {
      Log::Stderr("binary file format error at feature %d's size", i);
    }
    size_t size_of_feature = *(reinterpret_cast<size_t*>(buffer));
    // re-allocmate space if not enough
    if (size_of_feature > buffer_size) {
      delete[] buffer;
      buffer_size = size_of_feature;
      buffer = new char[buffer_size];
    }

    read_cnt = fread(buffer, 1, size_of_feature, file);

    if (read_cnt != size_of_feature) {
      Log::Stderr("binary file format error at feature %d loading , read count %d", i, read_cnt);
    }
    features_.push_back(new Feature(buffer, static_cast<data_size_t>(global_num_data_), used_data_indices_));
  }
  delete[] buffer;
  fclose(file);
}

void Dataset::CheckDataset() {
  if (num_data_ <= 0) {
    Log::Stderr("data size of %s is zero", data_filename_);
  }
  if (features_.size() <= 0) {
    Log::Stderr("not useful feature of data %s", data_filename_);
  }
}

}  // namespace LightGBM
