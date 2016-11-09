#include <omp.h>

#include <LightGBM/utils/log.h>
#include <LightGBM/dataset_loader.h>
#include <LightGBM/feature.h>
#include <LightGBM/network.h>


namespace LightGBM {

DatasetLoader::DatasetLoader(const IOConfig& io_config, const PredictFunction& predict_fun)
  :io_config_(io_config), predict_fun_(predict_fun){

}

DatasetLoader::~DatasetLoader() {

}

void DatasetLoader::SetHeader(const char* filename) {
  TextReader<data_size_t> text_reader(filename, io_config_.has_header);
  std::unordered_map<std::string, int> name2idx;

  // get column names
  if (io_config_.has_header) {
    std::string first_line = text_reader.first_line();
    feature_names_ = Common::Split(first_line.c_str(), "\t ,");
    for (size_t i = 0; i < feature_names_.size(); ++i) {
      name2idx[feature_names_[i]] = static_cast<int>(i);
    }
  }
  std::string name_prefix("name:");

  // load label idx
  if (io_config_.label_column.size() > 0) {
    if (Common::StartsWith(io_config_.label_column, name_prefix)) {
      std::string name = io_config_.label_column.substr(name_prefix.size());
      if (name2idx.count(name) > 0) {
        label_idx_ = name2idx[name];
        Log::Info("Using column %s as label", name.c_str());
      } else {
        Log::Fatal("Could not find label column %s in data file", name.c_str());
      }
    } else {
      if (!Common::AtoiAndCheck(io_config_.label_column.c_str(), &label_idx_)) {
        Log::Fatal("label_column is not a number, \
                      if you want to use a column name, \
                      please add the prefix \"name:\" to the column name");
      }
      Log::Info("Using column number %d as label", label_idx_);
    }
  }
  if (feature_names_.size() > 0) {
    // erase label column name
    feature_names_.erase(feature_names_.begin() + label_idx_);
  }
  // load ignore columns
  if (io_config_.ignore_column.size() > 0) {
    if (Common::StartsWith(io_config_.ignore_column, name_prefix)) {
      std::string names = io_config_.ignore_column.substr(name_prefix.size());
      for (auto name : Common::Split(names.c_str(), ',')) {
        if (name2idx.count(name) > 0) {
          int tmp = name2idx[name];
          // skip for label column
          if (tmp > label_idx_) { tmp -= 1; }
          ignore_features_.emplace(tmp);
        } else {
          Log::Fatal("Could not find ignore column %s in data file", name.c_str());
        }
      }
    } else {
      for (auto token : Common::Split(io_config_.ignore_column.c_str(), ',')) {
        int tmp = 0;
        if (!Common::AtoiAndCheck(token.c_str(), &tmp)) {
          Log::Fatal("ignore_column is not a number, \
                        if you want to use a column name, \
                        please add the prefix \"name:\" to the column name");
        }
        // skip for label column
        if (tmp > label_idx_) { tmp -= 1; }
        ignore_features_.emplace(tmp);
      }
    }

  }

  // load weight idx
  if (io_config_.weight_column.size() > 0) {
    if (Common::StartsWith(io_config_.weight_column, name_prefix)) {
      std::string name = io_config_.weight_column.substr(name_prefix.size());
      if (name2idx.count(name) > 0) {
        weight_idx_ = name2idx[name];
        Log::Info("Using column %s as weight", name.c_str());
      } else {
        Log::Fatal("Could not find weight column %s in data file", name.c_str());
      }
    } else {
      if (!Common::AtoiAndCheck(io_config_.weight_column.c_str(), &weight_idx_)) {
        Log::Fatal("weight_column is not a number, \
                      if you want to use a column name, \
                      please add the prefix \"name:\" to the column name");
      }
      Log::Info("Using column number %d as weight", weight_idx_);
    }
    // skip for label column
    if (weight_idx_ > label_idx_) {
      weight_idx_ -= 1;
    }
    ignore_features_.emplace(weight_idx_);
  }

  if (io_config_.group_column.size() > 0) {
    if (Common::StartsWith(io_config_.group_column, name_prefix)) {
      std::string name = io_config_.group_column.substr(name_prefix.size());
      if (name2idx.count(name) > 0) {
        group_idx_ = name2idx[name];
        Log::Info("Using column %s as group/query id", name.c_str());
      } else {
        Log::Fatal("Could not find group/query column %s in data file", name.c_str());
      }
    } else {
      if (!Common::AtoiAndCheck(io_config_.group_column.c_str(), &group_idx_)) {
        Log::Fatal("group_column is not a number, \
                      if you want to use a column name, \
                      please add the prefix \"name:\" to the column name");
      }
      Log::Info("Using column number %d as group/query id", group_idx_);
    }
    // skip for label column
    if (group_idx_ > label_idx_) {
      group_idx_ -= 1;
    }
    ignore_features_.emplace(group_idx_);
  }
}

Dataset* DatasetLoader::LoadFromFile(const char* filename, int rank, int num_machines) {
  // don't support query id in data file when training in parallel
  if (num_machines > 1 && !io_config_.is_pre_partition) {
    if (group_idx_ > 0) {
      Log::Fatal("Using a query id without pre-partitioning the data file is not supported for parallel training. \
                  Please use an additional query file or pre-partition the data");
    }
  }
  auto parser = Parser::CreateParser(filename, io_config_.has_header, 0, label_idx_);
  if (parser == nullptr) {
    Log::Fatal("Could not recognize data format of %s", filename);
  }
  data_size_t num_global_data = 0;
  std::vector<data_size_t> used_data_indices;
  Dataset* dataset = new Dataset();
  dataset->data_filename_ = filename;
  dataset->num_class_ = io_config_.num_class;
  dataset->metadata_.Init(filename, dataset->num_class_);
  bool is_loading_from_binfile = CheckCanLoadFromBin(filename);
  if (!is_loading_from_binfile) {
    if (!io_config_.use_two_round_loading) {
      // read data to memory
      auto text_data = LoadTextDataToMemory(filename, dataset->metadata_, rank, num_machines,&num_global_data, &used_data_indices);
      dataset->num_data_ = static_cast<data_size_t>(text_data.size());
      // sample data
      auto sample_data = SampleTextDataFromMemory(text_data);
      // construct feature bin mappers
      ConstructBinMappersFromTextData(rank, num_machines, sample_data, parser, dataset);
      // initialize label
      dataset->metadata_.Init(dataset->num_data_, io_config_.num_class, weight_idx_, group_idx_);
      // extract features
      ExtractFeaturesFromMemory(text_data, parser, dataset);
      text_data.clear();
    } else {
      // sample data from file
      auto sample_data = SampleTextDataFromFile(filename, dataset->metadata_, rank, num_machines, &num_global_data, &used_data_indices);
      if (used_data_indices.size() > 0) {
        dataset->num_data_ = static_cast<data_size_t>(used_data_indices.size());
      } else {
        dataset->num_data_ = num_global_data;
      }
      // construct feature bin mappers
      ConstructBinMappersFromTextData(rank, num_machines, sample_data, parser, dataset);
      // initialize label
      dataset->metadata_.Init(dataset->num_data_, dataset->num_class_, weight_idx_, group_idx_);

      // extract features
      ExtractFeaturesFromFile(filename, parser, used_data_indices, dataset);
    }
  } else {
    // load data from binary file
    delete dataset;
    std::string bin_filename(filename);
    bin_filename.append(".bin");
    dataset = LoadFromBinFile(bin_filename.c_str(), rank, num_machines);
  }
  // check meta data
  dataset->metadata_.CheckOrPartition(num_global_data, used_data_indices);
  // need to check training data
  CheckDataset(dataset);
  delete parser;
  return dataset;
}



Dataset* DatasetLoader::LoadFromFileAlignWithOtherDataset(const char* filename, const Dataset* train_data) {
  auto parser = Parser::CreateParser(filename, io_config_.has_header, 0, label_idx_);
  if (parser == nullptr) {
    Log::Fatal("Could not recognize data format of %s", filename);
  }
  data_size_t num_global_data = 0;
  std::vector<data_size_t> used_data_indices;
  Dataset* dataset = new Dataset();
  dataset->data_filename_ = filename;
  dataset->num_class_ = io_config_.num_class;
  dataset->metadata_.Init(filename, dataset->num_class_);
  bool is_loading_from_binfile = CheckCanLoadFromBin(filename);
  if (!is_loading_from_binfile) {
    if (!io_config_.use_two_round_loading) {
      // read data in memory
      auto text_data = LoadTextDataToMemory(filename, dataset->metadata_, 0, 1, &num_global_data, &used_data_indices);
      dataset->num_data_ = static_cast<data_size_t>(text_data.size());
      // initialize label
      dataset->metadata_.Init(dataset->num_data_, dataset->num_class_, weight_idx_, group_idx_);
      dataset->CopyFeatureMapperFrom(train_data, io_config_.is_enable_sparse);
      // extract features
      ExtractFeaturesFromMemory(text_data, parser, dataset);
      text_data.clear();
    } else {
      TextReader<data_size_t> text_reader(filename, io_config_.has_header);
      // Get number of lines of data file
      dataset->num_data_ = static_cast<data_size_t>(text_reader.CountLine());
      num_global_data = dataset->num_data_;
      // initialize label
      dataset->metadata_.Init(dataset->num_data_, dataset->num_class_, weight_idx_, group_idx_);
      dataset->CopyFeatureMapperFrom(train_data, io_config_.is_enable_sparse);
      // extract features
      ExtractFeaturesFromFile(filename, parser, used_data_indices, dataset);
    }
  } else {
    // load data from binary file
    delete dataset;
    std::string bin_filename(filename);
    bin_filename.append(".bin");
    dataset = LoadFromBinFile(bin_filename.c_str(), 0, 1);
  }
  // not need to check validation data
  // check meta data
  dataset->metadata_.CheckOrPartition(num_global_data, used_data_indices);
  delete parser;
  return dataset;
}

Dataset* DatasetLoader::LoadFromBinFile(const char* bin_filename, int rank, int num_machines) {
  Dataset* dataset = new Dataset();
  FILE* file;
#ifdef _MSC_VER
  fopen_s(&file, bin_filename, "rb");
#else
  file = fopen(bin_filename, "rb");
#endif

  if (file == NULL) {
    Log::Fatal("Could not read binary data from %s", bin_filename);
  }

  // buffer to read binary file
  size_t buffer_size = 16 * 1024 * 1024;
  char* buffer = new char[buffer_size];

  // read size of header
  size_t read_cnt = fread(buffer, sizeof(size_t), 1, file);

  if (read_cnt != 1) {
    Log::Fatal("Binary file error: header has the wrong size");
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
    Log::Fatal("Binary file error: header is incorrect");
  }
  // get header
  const char* mem_ptr = buffer;
  dataset->num_data_ = *(reinterpret_cast<const data_size_t*>(mem_ptr));
  mem_ptr += sizeof(dataset->num_data_);
  dataset->num_class_ = *(reinterpret_cast<const int*>(mem_ptr));
  mem_ptr += sizeof(dataset->num_class_);
  dataset->num_features_ = *(reinterpret_cast<const int*>(mem_ptr));
  mem_ptr += sizeof(dataset->num_features_);
  dataset->num_total_features_ = *(reinterpret_cast<const int*>(mem_ptr));
  mem_ptr += sizeof(dataset->num_total_features_);
  size_t num_used_feature_map = *(reinterpret_cast<const size_t*>(mem_ptr));
  mem_ptr += sizeof(num_used_feature_map);
  const int* tmp_feature_map = reinterpret_cast<const int*>(mem_ptr);
  dataset->used_feature_map_.clear();
  for (size_t i = 0; i < num_used_feature_map; ++i) {
    dataset->used_feature_map_.push_back(tmp_feature_map[i]);
  }
  mem_ptr += sizeof(int) * num_used_feature_map;
  // get feature names
  dataset->feature_names_.clear();
  // write feature names
  for (int i = 0; i < dataset->num_total_features_; ++i) {
    int str_len = *(reinterpret_cast<const int*>(mem_ptr));
    mem_ptr += sizeof(int);
    std::stringstream str_buf;
    for (int j = 0; j < str_len; ++j) {
      char tmp_char = *(reinterpret_cast<const char*>(mem_ptr));
      mem_ptr += sizeof(char);
      str_buf << tmp_char;
    }
    dataset->feature_names_.emplace_back(str_buf.str());
  }

  // read size of meta data
  read_cnt = fread(buffer, sizeof(size_t), 1, file);

  if (read_cnt != 1) {
    Log::Fatal("Binary file error: meta data has the wrong size");
  }

  size_t size_of_metadata = *(reinterpret_cast<size_t*>(buffer));

  // re-allocate space if not enough
  if (size_of_metadata > buffer_size) {
    delete[] buffer;
    buffer_size = size_of_metadata;
    buffer = new char[buffer_size];
  }
  //  read meta data
  read_cnt = fread(buffer, 1, size_of_metadata, file);

  if (read_cnt != size_of_metadata) {
    Log::Fatal("Binary file error: meta data is incorrect");
  }
  // load meta data
  dataset->metadata_.LoadFromMemory(buffer);

  std::vector<data_size_t> used_data_indices;
  data_size_t num_global_data = dataset->num_data_;
  // sample local used data if need to partition
  if (num_machines > 1 && !io_config_.is_pre_partition) {
    const data_size_t* query_boundaries = dataset->metadata_.query_boundaries();
    if (query_boundaries == nullptr) {
      // if not contain query file, minimal sample unit is one record
      for (data_size_t i = 0; i < dataset->num_data_; ++i) {
        if (random_.NextInt(0, num_machines) == rank) {
          used_data_indices.push_back(i);
        }
      }
    } else {
      // if contain query file, minimal sample unit is one query
      data_size_t num_queries = dataset->metadata_.num_queries();
      data_size_t qid = -1;
      bool is_query_used = false;
      for (data_size_t i = 0; i < dataset->num_data_; ++i) {
        if (qid >= num_queries) {
          Log::Fatal("Current query exceeds the range of the query file, please ensure the query file is correct");
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
          used_data_indices.push_back(i);
        }
      }
    }
    dataset->num_data_ = static_cast<data_size_t>(used_data_indices.size());
  }
  dataset->metadata_.PartitionLabel(used_data_indices);
  // read feature data
  for (int i = 0; i < dataset->num_features_; ++i) {
    // read feature size
    read_cnt = fread(buffer, sizeof(size_t), 1, file);
    if (read_cnt != 1) {
      Log::Fatal("Binary file error: feature %d has the wrong size", i);
    }
    size_t size_of_feature = *(reinterpret_cast<size_t*>(buffer));
    // re-allocate space if not enough
    if (size_of_feature > buffer_size) {
      delete[] buffer;
      buffer_size = size_of_feature;
      buffer = new char[buffer_size];
    }

    read_cnt = fread(buffer, 1, size_of_feature, file);

    if (read_cnt != size_of_feature) {
      Log::Fatal("Binary file error: feature %d is incorrect, read count: %d", i, read_cnt);
    }
    dataset->features_.push_back(new Feature(buffer, num_global_data, used_data_indices));
  }
  delete[] buffer;
  fclose(file);
  dataset->is_loading_from_binfile_ = true;
  return dataset;
}

Dataset* DatasetLoader::CostructFromSampleData(std::vector<std::vector<double>>& sample_values, size_t total_sample_size, data_size_t num_data) {
  std::vector<BinMapper*> bin_mappers(sample_values.size());
#pragma omp parallel for schedule(guided)
  for (int i = 0; i < static_cast<int>(sample_values.size()); ++i) {
    bin_mappers[i] = new BinMapper();
    bin_mappers[i]->FindBin(&sample_values[i], total_sample_size, io_config_.max_bin);
  }

  Dataset* dataset = new Dataset();
  dataset->num_class_ = io_config_.num_class;
  dataset->features_.clear();
  dataset->num_data_ = num_data;
  // -1 means doesn't use this feature
  dataset->used_feature_map_ = std::vector<int>(bin_mappers.size(), -1);
  dataset->num_total_features_ = static_cast<int>(bin_mappers.size());

  for (size_t i = 0; i < bin_mappers.size(); ++i) {
    if (!bin_mappers[i]->is_trival()) {
      // map real feature index to used feature index
      dataset->used_feature_map_[i] = static_cast<int>(dataset->features_.size());
      // push new feature
      dataset->features_.push_back(new Feature(static_cast<int>(i), bin_mappers[i],
        dataset->num_data_, io_config_.is_enable_sparse));
    } else {
      // if feature is trival(only 1 bin), free spaces
      Log::Warning("Ignoring Column_%d , only has one value", i);
      delete bin_mappers[i];
    }
  }
  // fill feature_names_ if not header
  if (feature_names_.size() <= 0) {
    for (int i = 0; i < dataset->num_total_features_; ++i) {
      std::stringstream str_buf;
      str_buf << "Column_" << i;
      feature_names_.push_back(str_buf.str());
    }
  }
  dataset->feature_names_ = feature_names_;
  dataset->num_features_ = static_cast<int>(dataset->features_.size());
  dataset->metadata_.Init(dataset->num_data_, dataset->num_class_, NO_SPECIFIC, NO_SPECIFIC);
  return dataset;
}


// ---- private functions ----

void DatasetLoader::CheckDataset(const Dataset* dataset) {
  if (dataset->num_data_ <= 0) {
    Log::Fatal("Data file %s is empty", dataset->data_filename_);
  }
  if (dataset->features_.size() <= 0) {
    Log::Fatal("No usable features in data file %s", dataset->data_filename_);
  }
}

std::vector<std::string> DatasetLoader::LoadTextDataToMemory(const char* filename, const Metadata& metadata,
  int rank, int num_machines, int* num_global_data, 
  std::vector<data_size_t>* used_data_indices) {
  TextReader<data_size_t> text_reader(filename, io_config_.has_header);
  used_data_indices->clear();
  if (num_machines == 1 || io_config_.is_pre_partition) {
    // read all lines
    *num_global_data = text_reader.ReadAllLines();
  } else {  // need partition data
            // get query data
    const data_size_t* query_boundaries = metadata.query_boundaries();

    if (query_boundaries == nullptr) {
      // if not contain query data, minimal sample unit is one record
      *num_global_data = text_reader.ReadAndFilterLines([this, rank, num_machines](data_size_t) {
        if (random_.NextInt(0, num_machines) == rank) {
          return true;
        } else {
          return false;
        }
      }, used_data_indices);
    } else {
      // if contain query data, minimal sample unit is one query
      data_size_t num_queries = metadata.num_queries();
      data_size_t qid = -1;
      bool is_query_used = false;
      *num_global_data = text_reader.ReadAndFilterLines(
        [this, rank, num_machines, &qid, &query_boundaries, &is_query_used, num_queries]
      (data_size_t line_idx) {
        if (qid >= num_queries) {
          Log::Fatal("Current query exceeds the range of the query file, please ensure the query file is correct");
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
      }, used_data_indices);
    }
  }
  return std::move(text_reader.Lines());
}

std::vector<std::string> DatasetLoader::SampleTextDataFromMemory(const std::vector<std::string>& data) {
  size_t sample_cnt = static_cast<size_t>(io_config_.bin_construct_sample_cnt);
  if (sample_cnt > data.size()) {
    sample_cnt = data.size();
  }
  std::vector<size_t> sample_indices = random_.Sample(data.size(), sample_cnt);
  std::vector<std::string> out;
  for (size_t i = 0; i < sample_indices.size(); ++i) {
    const size_t idx = sample_indices[i];
    out.push_back(data[idx]);
  }
  return out;
}

std::vector<std::string> DatasetLoader::SampleTextDataFromFile(const char* filename, const Metadata& metadata, int rank, int num_machines, int* num_global_data, std::vector<data_size_t>* used_data_indices) {
  const data_size_t sample_cnt = static_cast<data_size_t>(io_config_.bin_construct_sample_cnt);
  TextReader<data_size_t> text_reader(filename, io_config_.has_header);
  std::vector<std::string> out_data;
  if (num_machines == 1 || io_config_.is_pre_partition) {
    *num_global_data = static_cast<data_size_t>(text_reader.SampleFromFile(random_, sample_cnt, &out_data));
  } else {  // need partition data
            // get query data
    const data_size_t* query_boundaries = metadata.query_boundaries();
    if (query_boundaries == nullptr) {
      // if not contain query file, minimal sample unit is one record
      *num_global_data = text_reader.SampleAndFilterFromFile([this, rank, num_machines]
      (data_size_t) {
        if (random_.NextInt(0, num_machines) == rank) {
          return true;
        } else {
          return false;
        }
      }, used_data_indices, random_, sample_cnt, &out_data);
    } else {
      // if contain query file, minimal sample unit is one query
      data_size_t num_queries = metadata.num_queries();
      data_size_t qid = -1;
      bool is_query_used = false;
      *num_global_data = text_reader.SampleAndFilterFromFile(
        [this, rank, num_machines, &qid, &query_boundaries, &is_query_used, num_queries]
      (data_size_t line_idx) {
        if (qid >= num_queries) {
          Log::Fatal("Query id exceeds the range of the query file, \
                      please ensure the query file is correct");
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
      }, used_data_indices, random_, sample_cnt, &out_data);
    }
  }
  return out_data;
}

void DatasetLoader::ConstructBinMappersFromTextData(int rank, int num_machines, const std::vector<std::string>& sample_data, const Parser* parser, Dataset* dataset) {
  // sample_values[i][j], means the value of j-th sample on i-th feature
  std::vector<std::vector<double>> sample_values;
  // temp buffer for one line features and label
  std::vector<std::pair<int, double>> oneline_features;
  double label;
  for (size_t i = 0; i < sample_data.size(); ++i) {
    oneline_features.clear();
    // parse features
    parser->ParseOneLine(sample_data[i].c_str(), &oneline_features, &label);
    for (std::pair<int, double>& inner_data : oneline_features) {
      if (std::fabs(inner_data.second) > 1e-15) {
        if (static_cast<size_t>(inner_data.first) >= sample_values.size()) {
          // if need expand feature set
          size_t need_size = inner_data.first - sample_values.size() + 1;
          for (size_t j = 0; j < need_size; ++j) {
            sample_values.emplace_back();
          }
        }
        sample_values[inner_data.first].push_back(inner_data.second);
      }
    }
  }

  dataset->features_.clear();

  // -1 means doesn't use this feature
  dataset->used_feature_map_ = std::vector<int>(sample_values.size(), -1);
  dataset->num_total_features_ = static_cast<int>(sample_values.size());

  // check the range of label_idx, weight_idx and group_idx
  CHECK(label_idx_ >= 0 && label_idx_ <= dataset->num_total_features_);
  CHECK(weight_idx_ < 0 || weight_idx_ < dataset->num_total_features_);
  CHECK(group_idx_ < 0 || group_idx_ < dataset->num_total_features_);

  // fill feature_names_ if not header
  if (feature_names_.size() <= 0) {
    for (int i = 0; i < dataset->num_total_features_; ++i) {
      std::stringstream str_buf;
      str_buf << "Column_" << i;
      feature_names_.push_back(str_buf.str());
    }
  }
  dataset->feature_names_ = feature_names_;
  // start find bins
  if (num_machines == 1) {
    std::vector<BinMapper*> bin_mappers(sample_values.size());
    // if only one machine, find bin locally
#pragma omp parallel for schedule(guided)
    for (int i = 0; i < static_cast<int>(sample_values.size()); ++i) {
      if (ignore_features_.count(i) > 0) {
        bin_mappers[i] = nullptr;
        continue;
      }
      bin_mappers[i] = new BinMapper();
      bin_mappers[i]->FindBin(&sample_values[i], sample_data.size(), io_config_.max_bin);
    }

    for (size_t i = 0; i < sample_values.size(); ++i) {
      if (bin_mappers[i] == nullptr) {
        Log::Warning("Ignoring feature %s", feature_names_[i].c_str());
      } else if (!bin_mappers[i]->is_trival()) {
        // map real feature index to used feature index
        dataset->used_feature_map_[i] = static_cast<int>(dataset->features_.size());
        // push new feature
        dataset->features_.push_back(new Feature(static_cast<int>(i), bin_mappers[i],
          dataset->num_data_, io_config_.is_enable_sparse));
      } else {
        // if feature is trival(only 1 bin), free spaces
        Log::Warning("Ignoring feature %s, only has one value", feature_names_[i].c_str());
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
    int type_size = BinMapper::SizeForSpecificBin(io_config_.max_bin);
    // since sizes of different feature may not be same, we expand all bin mapper to type_size
    int buffer_size = type_size * total_num_feature;
    char* input_buffer = new char[buffer_size];
    char* output_buffer = new char[buffer_size];

    // find local feature bins and copy to buffer
#pragma omp parallel for schedule(guided)
    for (int i = 0; i < len[rank]; ++i) {
      BinMapper* bin_mapper = new BinMapper();
      bin_mapper->FindBin(&sample_values[start[rank] + i], sample_data.size(), io_config_.max_bin);
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
      if (ignore_features_.count(i) > 0) {
        Log::Warning("Ignoring feature %s", feature_names_[i].c_str());
        continue;
      }
      BinMapper* bin_mapper = new BinMapper();
      bin_mapper->CopyFrom(output_buffer + i * type_size);
      if (!bin_mapper->is_trival()) {
        dataset->used_feature_map_[i] = static_cast<int>(dataset->features_.size());
        dataset->features_.push_back(new Feature(static_cast<int>(i), bin_mapper, dataset->num_data_, io_config_.is_enable_sparse));
      } else {
        Log::Warning("Ignoring feature %s, only has one value", feature_names_[i].c_str());
        delete bin_mapper;
      }
    }
    // free buffer
    delete[] start;
    delete[] len;
    delete[] input_buffer;
    delete[] output_buffer;
  }
  dataset->num_features_ = static_cast<int>(dataset->features_.size());
}

/*! \brief Extract local features from memory */
void DatasetLoader::ExtractFeaturesFromMemory(std::vector<std::string>& text_data, const Parser* parser, Dataset* dataset) {
  std::vector<std::pair<int, double>> oneline_features;
  double tmp_label = 0.0f;
  if (predict_fun_ == nullptr) {
    // if doesn't need to prediction with initial model
#pragma omp parallel for schedule(guided) private(oneline_features) firstprivate(tmp_label)
    for (data_size_t i = 0; i < dataset->num_data_; ++i) {
      const int tid = omp_get_thread_num();
      oneline_features.clear();
      // parser
      parser->ParseOneLine(text_data[i].c_str(), &oneline_features, &tmp_label);
      // set label
      dataset->metadata_.SetLabelAt(i, static_cast<float>(tmp_label));
      // free processed line:
      text_data[i].clear();
      // shrink_to_fit will be very slow in linux, and seems not free memory, disable for now
      // text_reader_->Lines()[i].shrink_to_fit();
      // push data
      for (auto& inner_data : oneline_features) {
        int feature_idx = dataset->used_feature_map_[inner_data.first];
        if (feature_idx >= 0) {
          // if is used feature
          dataset->features_[feature_idx]->PushData(tid, i, inner_data.second);
        } else {
          if (inner_data.first == weight_idx_) {
            dataset->metadata_.SetWeightAt(i, static_cast<float>(inner_data.second));
          } else if (inner_data.first == group_idx_) {
            dataset->metadata_.SetQueryAt(i, static_cast<data_size_t>(inner_data.second));
          }
        }
      }
    }
  } else {
    // if need to prediction with initial model
    float* init_score = new float[dataset->num_data_ * dataset->num_class_];
#pragma omp parallel for schedule(guided) private(oneline_features) firstprivate(tmp_label)
    for (data_size_t i = 0; i < dataset->num_data_; ++i) {
      const int tid = omp_get_thread_num();
      oneline_features.clear();
      // parser
      parser->ParseOneLine(text_data[i].c_str(), &oneline_features, &tmp_label);
      // set initial score
      std::vector<double> oneline_init_score = predict_fun_(oneline_features);
      for (int k = 0; k < dataset->num_class_; ++k) {
        init_score[k * dataset->num_data_ + i] = static_cast<float>(oneline_init_score[k]);
      }
      // set label
      dataset->metadata_.SetLabelAt(i, static_cast<float>(tmp_label));
      // free processed line:
      text_data[i].clear();
      // shrink_to_fit will be very slow in linux, and seems not free memory, disable for now
      // text_reader_->Lines()[i].shrink_to_fit();
      // push data
      for (auto& inner_data : oneline_features) {
        int feature_idx = dataset->used_feature_map_[inner_data.first];
        if (feature_idx >= 0) {
          // if is used feature
          dataset->features_[feature_idx]->PushData(tid, i, inner_data.second);
        } else {
          if (inner_data.first == weight_idx_) {
            dataset->metadata_.SetWeightAt(i, static_cast<float>(inner_data.second));
          } else if (inner_data.first == group_idx_) {
            dataset->metadata_.SetQueryAt(i, static_cast<data_size_t>(inner_data.second));
          }
        }
      }
    }
    // metadata_ will manage space of init_score
    dataset->metadata_.SetInitScore(init_score, dataset->num_data_ * dataset->num_class_);
    delete[] init_score;
  }
  dataset->FinishLoad();
  // text data can be free after loaded feature values
  text_data.clear();
}

/*! \brief Extract local features from file */
void DatasetLoader::ExtractFeaturesFromFile(const char* filename, const Parser* parser, const std::vector<data_size_t>& used_data_indices, Dataset* dataset) {
  float* init_score = nullptr;
  if (predict_fun_ != nullptr) {
    init_score = new float[dataset->num_data_ * dataset->num_class_];
  }
  std::function<void(data_size_t, const std::vector<std::string>&)> process_fun =
    [this, &init_score, &parser, &dataset]
  (data_size_t start_idx, const std::vector<std::string>& lines) {
    std::vector<std::pair<int, double>> oneline_features;
    double tmp_label = 0.0f;
#pragma omp parallel for schedule(static) private(oneline_features) firstprivate(tmp_label)
    for (data_size_t i = 0; i < static_cast<data_size_t>(lines.size()); ++i) {
      const int tid = omp_get_thread_num();
      oneline_features.clear();
      // parser
      parser->ParseOneLine(lines[i].c_str(), &oneline_features, &tmp_label);
      // set initial score
      if (init_score != nullptr) {
        std::vector<double> oneline_init_score = predict_fun_(oneline_features);
        for (int k = 0; k < dataset->num_class_; ++k) {
          init_score[k * dataset->num_data_ + start_idx + i] = static_cast<float>(oneline_init_score[k]);
        }
      }
      // set label
      dataset->metadata_.SetLabelAt(start_idx + i, static_cast<float>(tmp_label));
      // push data
      for (auto& inner_data : oneline_features) {
        int feature_idx = dataset->used_feature_map_[inner_data.first];
        if (feature_idx >= 0) {
          // if is used feature
          dataset->features_[feature_idx]->PushData(tid, start_idx + i, inner_data.second);
        } else {
          if (inner_data.first == weight_idx_) {
            dataset->metadata_.SetWeightAt(start_idx + i, static_cast<float>(inner_data.second));
          } else if (inner_data.first == group_idx_) {
            dataset->metadata_.SetQueryAt(start_idx + i, static_cast<data_size_t>(inner_data.second));
          }
        }
      }
    }
  };
  TextReader<data_size_t> text_reader(filename, io_config_.has_header);
  if (used_data_indices.size() > 0) {
    // only need part of data
    text_reader.ReadPartAndProcessParallel(used_data_indices, process_fun);
  } else {
    // need full data
    text_reader.ReadAllAndProcessParallel(process_fun);
  }

  // metadata_ will manage space of init_score
  if (init_score != nullptr) {
    dataset->metadata_.SetInitScore(init_score, dataset->num_data_ * dataset->num_class_);
    delete[] init_score;
  }
  dataset->FinishLoad();
}

/*! \brief Check can load from binary file */
bool DatasetLoader::CheckCanLoadFromBin(const char* filename) {
  std::string bin_filename(filename);
  bin_filename.append(".bin");

  FILE* file;

#ifdef _MSC_VER
  fopen_s(&file, bin_filename.c_str(), "rb");
#else
  file = fopen(bin_filename.c_str(), "rb");
#endif
  if (file == NULL) {
    return false;
  } else {
    fclose(file);
    return true;
  }
}

}