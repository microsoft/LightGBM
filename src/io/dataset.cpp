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


Dataset::Dataset() {
  num_class_ = 1;
  num_data_ = 0;
  is_loading_from_binfile_ = false;
}

Dataset::~Dataset() {
  for (auto& feature : features_) {
    delete feature;
  }
  features_.clear();
}

void Dataset::FinishLoad() {
#pragma omp parallel for schedule(guided)
  for (int i = 0; i < num_features_; ++i) {
    features_[i]->FinishLoad();
  }
}

void Dataset::CopyFeatureMetadataTo(Dataset *dataset, bool is_enable_sparse) const {
  dataset->features_.clear();
  // copy feature bin mapper data
  for (Feature* feature : features_) {
    dataset->features_.push_back(new Feature(feature->feature_index(),
      new BinMapper(*feature->bin_mapper()), dataset->num_data_, is_enable_sparse));
  }
  dataset->used_feature_map_ = used_feature_map_;
  dataset->num_features_ = static_cast<int>(dataset->features_.size());
  dataset->num_total_features_ = num_total_features_;
  dataset->feature_names_ = feature_names_;
}

std::vector<const BinMapper*> Dataset::GetBinMappers() const {
  std::vector<const BinMapper*> ret(num_total_features_, nullptr);
  for (const auto feature : features_) {
    ret[feature->feature_index()] = feature->bin_mapper();
  }
  return ret;
}

void Dataset::SetField(const char* field_name, const void* field_data, data_size_t num_element, int type) {
  std::string name(field_name);
  name = Common::Trim(name);
  if (name == std::string("label") || name == std::string("target")) {
    if (type != 0) {
      Log::Fatal("type of label should be float");
    }
    metadata_.SetLabel(static_cast<const float*>(field_data), num_element);
  } else if (name == std::string("weight") || name == std::string("weights")) {
    if (type != 0) {
      Log::Fatal("type of weights should be float");
    }
    metadata_.SetWeights(static_cast<const float*>(field_data), num_element);
  } else if (name == std::string("init_score")) {
    if (type != 0) {
      Log::Fatal("type of init_score should be float");
    }
    metadata_.SetInitScore(static_cast<const float*>(field_data), num_element);
  } else if (name == std::string("query") || name == std::string("group")) {
    if (type != 1) {
      Log::Fatal("type of init_score should be int");
    }
    metadata_.SetQueryBoundaries(static_cast<const data_size_t*>(field_data), num_element);
  } else {
    Log::Fatal("unknow field name: %s", field_name);
  }
}

void Dataset::SaveBinaryFile(const char* bin_filename) {

  if (!is_loading_from_binfile_) {
    // if not pass a filename, just append ".bin" of original file
    if (bin_filename == nullptr || bin_filename[0] == '\0') {
      std::string bin_filename_str(data_filename_);
      bin_filename_str.append(".bin");
      bin_filename = bin_filename_str.c_str();
    }
    FILE* file;
#ifdef _MSC_VER
    fopen_s(&file, bin_filename, "wb");
#else
    file = fopen(bin_filename, "wb");
#endif
    if (file == NULL) {
      Log::Fatal("Cannot write binary data to %s ", bin_filename);
    }

    Log::Info("Saving data to binary file %s", data_filename_);

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
