#include <LightGBM/dataset.h>

#include <LightGBM/feature.h>

#include <omp.h>

#include <cstdio>
#include <unordered_map>
#include <limits>
#include <vector>
#include <utility>
#include <string>
#include <sstream>

namespace LightGBM {

const char* Dataset::binary_file_token = "______LightGBM_Binary_File_Token______\n";

Dataset::Dataset() {
  data_filename_ = "noname";
  num_data_ = 0;
}

Dataset::Dataset(data_size_t num_data) {
  num_data_ = num_data;
  metadata_.Init(num_data_, -1, -1);
}

Dataset::~Dataset() {

}

void Dataset::FinishLoad() {
#pragma omp parallel for schedule(guided)
  for (int i = 0; i < num_features_; ++i) {
    features_[i]->FinishLoad();
  }
}

void Dataset::CopyFeatureMapperFrom(const Dataset* dataset, bool is_enable_sparse) {
  features_.clear();
  // copy feature bin mapper data
  for (const auto& feature : dataset->features_) {
    features_.emplace_back(std::unique_ptr<Feature>(
      new Feature(feature->feature_index(), 
        new BinMapper(*feature->bin_mapper()), 
        num_data_, 
        is_enable_sparse)
      ));
  }
  features_.shrink_to_fit();
  used_feature_map_ = dataset->used_feature_map_;
  num_features_ = static_cast<int>(features_.size());
  num_total_features_ = dataset->num_total_features_;
  feature_names_ = dataset->feature_names_;
  label_idx_ = dataset->label_idx_;
}

Dataset* Dataset::Subset(const data_size_t* used_indices, data_size_t num_used_indices, bool is_enable_sparse) const {
  auto ret = std::unique_ptr<Dataset>(new Dataset(num_used_indices));
  ret->CopyFeatureMapperFrom(this, is_enable_sparse);
#pragma omp parallel for schedule(guided)
  for (int fidx = 0; fidx < num_features_; ++fidx) {
    auto iterator = features_[fidx]->bin_data()->GetIterator(0);
    for (data_size_t i = 0; i < num_used_indices; ++i) {
      ret->features_[fidx]->PushBin(0, i, iterator->Get(used_indices[i]));
    }
  }
  ret->metadata_.Init(metadata_, used_indices, num_used_indices);
  return ret.release();
}

bool Dataset::SetFloatField(const char* field_name, const float* field_data, data_size_t num_element) {
  std::string name(field_name);
  name = Common::Trim(name);
  if (name == std::string("label") || name == std::string("target")) {
    metadata_.SetLabel(field_data, num_element);
  } else if (name == std::string("weight") || name == std::string("weights")) {
    metadata_.SetWeights(field_data, num_element);
  } else if (name == std::string("init_score")) {
    metadata_.SetInitScore(field_data, num_element);
  } else {
    return false;
  }
  return true;
}

bool Dataset::SetIntField(const char* field_name, const int* field_data, data_size_t num_element) {
  std::string name(field_name);
  name = Common::Trim(name);
  if (name == std::string("query") || name == std::string("group")) {
    metadata_.SetQuery(field_data, num_element);
  } else if (name == std::string("query_id") || name == std::string("group_id")) {
    metadata_.SetQueryId(field_data, num_element);
  } else {
    return false;
  }
  return true;
}

bool Dataset::GetFloatField(const char* field_name, data_size_t* out_len, const float** out_ptr) {
  std::string name(field_name);
  name = Common::Trim(name);
  if (name == std::string("label") || name == std::string("target")) {
    *out_ptr = metadata_.label();
    *out_len = num_data_;
  } else if (name == std::string("weight") || name == std::string("weights")) {
    *out_ptr = metadata_.weights();
    *out_len = num_data_;
  } else if (name == std::string("init_score")) {
    *out_ptr = metadata_.init_score();
    *out_len = num_data_;
  } else {
    return false;
  }
  return true;
}

bool Dataset::GetIntField(const char* field_name, data_size_t* out_len, const int** out_ptr) {
  std::string name(field_name);
  name = Common::Trim(name);
  if (name == std::string("query") || name == std::string("group")) {
    *out_ptr = metadata_.query_boundaries();
    *out_len = metadata_.num_queries() + 1;
  } else {
    return false;
  }
  return true;
}

void Dataset::SaveBinaryFile(const char* bin_filename) {
  if (bin_filename != nullptr 
      && std::string(bin_filename) == std::string(data_filename_)) {
    Log::Warning("Bianry file %s already existed", bin_filename);
    return;
  }
  // if not pass a filename, just append ".bin" of original file
  std::string bin_filename_str(data_filename_);
  if (bin_filename == nullptr || bin_filename[0] == '\0') {
    bin_filename_str.append(".bin");
    bin_filename = bin_filename_str.c_str();
  }
  bool is_file_existed = false;
  FILE* file;
#ifdef _MSC_VER
  fopen_s(&file, bin_filename, "rb");
#else
  file = fopen(bin_filename, "rb");
#endif

  if (file != NULL) {
    is_file_existed = true;
    Log::Warning("File %s existed, cannot save binary to it", bin_filename);
    fclose(file);
  }

  if (!is_file_existed) {
#ifdef _MSC_VER
    fopen_s(&file, bin_filename, "wb");
#else
    file = fopen(bin_filename, "wb");
#endif
    if (file == NULL) {
      Log::Fatal("Cannot write binary data to %s ", bin_filename);
    }
    Log::Info("Saving data to binary file %s", bin_filename);
    size_t size_of_token = std::strlen(binary_file_token);
    fwrite(binary_file_token, sizeof(char), size_of_token, file);
    // get size of header
    size_t size_of_header = sizeof(num_data_) + sizeof(num_features_) + sizeof(num_total_features_) 
      + sizeof(size_t) + sizeof(int) * used_feature_map_.size();
    // size of feature names
    for (int i = 0; i < num_total_features_; ++i) {
      size_of_header += feature_names_[i].size() + sizeof(int);
    }
    fwrite(&size_of_header, sizeof(size_of_header), 1, file);
    // write header
    fwrite(&num_data_, sizeof(num_data_), 1, file);
    fwrite(&num_features_, sizeof(num_features_), 1, file);
    fwrite(&num_total_features_, sizeof(num_features_), 1, file);
    size_t num_used_feature_map = used_feature_map_.size();
    fwrite(&num_used_feature_map, sizeof(num_used_feature_map), 1, file);
    fwrite(used_feature_map_.data(), sizeof(int), num_used_feature_map, file);

    // write feature names
    for (int i = 0; i < num_total_features_; ++i) {
      int str_len = static_cast<int>(feature_names_[i].size());
      fwrite(&str_len, sizeof(int), 1, file);
      const char* c_str = feature_names_[i].c_str();
      fwrite(c_str, sizeof(char), str_len, file);
    }

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

}  // namespace LightGBM
