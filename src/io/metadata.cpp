#include <LightGBM/dataset.h>

#include <LightGBM/utils/common.h>

#include <vector>
#include <string>

namespace LightGBM {

Metadata::Metadata()
  :label_(nullptr), label_int_(nullptr), weights_(nullptr), 
  query_boundaries_(nullptr),
  query_weights_(nullptr), init_score_(nullptr), queries_(nullptr){

}

void Metadata::Init(const char * data_filename, const char* init_score_filename, const int num_class) {
  data_filename_ = data_filename;
  init_score_filename_ = init_score_filename;
  num_class_ = num_class;
  // for lambdarank, it needs query data for partition data in parallel learning
  LoadQueryBoundaries();
  LoadWeights();
  LoadQueryWeights();
  LoadInitialScore();
}

void Metadata::Init(const char* init_score_filename, const int num_class) {
  init_score_filename_ = init_score_filename;
  num_class_ = num_class;
  LoadInitialScore();
}


Metadata::~Metadata() {
  if (label_ != nullptr) { delete[] label_; }
  if (weights_ != nullptr) { delete[] weights_; }
  if (query_boundaries_ != nullptr) { delete[] query_boundaries_; }
  if (query_weights_ != nullptr) { delete[] query_weights_; }
  if (init_score_ != nullptr) { delete[] init_score_; }
  if (queries_ != nullptr) { delete[] queries_; }
}


void Metadata::Init(data_size_t num_data, int num_class, int weight_idx, int query_idx) {
  num_data_ = num_data;
  num_class_ = num_class;
  label_ = new float[num_data_];
  if (weight_idx >= 0) {
    if (weights_ != nullptr) {
      Log::Info("using weight in data file, and ignore additional weight file");
      delete[] weights_; 
    }
    weights_ = new float[num_data_];
    num_weights_ = num_data_;
    memset(weights_, 0, sizeof(float) * num_data_);
  }
  if (query_idx >= 0) {
    if (query_boundaries_ != nullptr) {
      Log::Info("using query id in data file, and ignore additional query file");
      delete[] query_boundaries_;
    }
    if (query_weights_ != nullptr) { delete[] query_weights_; }
    queries_ = new data_size_t[num_data_];
    memset(queries_, 0, sizeof(data_size_t) * num_data_);
  }
}

void Metadata::PartitionLabel(const std::vector<data_size_t>& used_indices) {
  if (used_indices.size() <= 0) {
    return;
  }
  float* old_label = label_;
  num_data_ = static_cast<data_size_t>(used_indices.size());
  label_ = new float[num_data_];
  for (data_size_t i = 0; i < num_data_; ++i) {
    label_[i] = old_label[used_indices[i]];
  }
  delete[] old_label;
}

void Metadata::CheckOrPartition(data_size_t num_all_data, const std::vector<data_size_t>& used_data_indices) {
  if (used_data_indices.size() == 0) {
    if (queries_ != nullptr) {
      // need convert query_id to boundaries
      std::vector<data_size_t> tmp_buffer;
      data_size_t last_qid = -1;
      data_size_t cur_cnt = 0;
      for (data_size_t i = 0; i < num_data_; ++i) {
        if (last_qid != queries_[i]) {
          if (cur_cnt > 0) {
            tmp_buffer.push_back(cur_cnt);
          }
          cur_cnt = 0;
          last_qid = queries_[i];
        }
        ++cur_cnt;
      }
      tmp_buffer.push_back(cur_cnt);
      query_boundaries_ = new data_size_t[tmp_buffer.size() + 1];
      num_queries_ = static_cast<data_size_t>(tmp_buffer.size());
      query_boundaries_[0] = 0;
      for (size_t i = 0; i < tmp_buffer.size(); ++i) {
        query_boundaries_[i + 1] = query_boundaries_[i] + tmp_buffer[i];
      }
      LoadQueryWeights();
      delete[] queries_;
      queries_ = nullptr;
    }
    // check weights
    if (weights_ != nullptr && num_weights_ != num_data_) {
      Log::Fatal("Initial weight size doesn't equal to data");
      delete[] weights_;
      num_weights_ = 0;
      weights_ = nullptr;
    }

    // check query boundries
    if (query_boundaries_ != nullptr && query_boundaries_[num_queries_] != num_data_) {
      Log::Fatal("Initial query size doesn't equal to data");
      delete[] query_boundaries_;
      num_queries_ = 0;
      query_boundaries_ = nullptr;
    }

    // contain initial score file
    if (init_score_ != nullptr && num_init_score_ != num_data_) {
      delete[] init_score_;
      Log::Fatal("Initial score size doesn't equal to data");
      init_score_ = nullptr;
      num_init_score_ = 0;
    }
  } else {
    data_size_t num_used_data = static_cast<data_size_t>(used_data_indices.size());
    // check weights
    if (weights_ != nullptr && num_weights_ != num_all_data) {
      Log::Fatal("Initial weights size doesn't equal to data");
      delete[] weights_;
      num_weights_ = 0;
      weights_ = nullptr;
    }
    // check query boundries
    if (query_boundaries_ != nullptr && query_boundaries_[num_queries_] != num_all_data) {
      Log::Fatal("Initial query size doesn't equal to data");
      delete[] query_boundaries_;
      num_queries_ = 0;
      query_boundaries_ = nullptr;
    }

    // contain initial score file
    if (init_score_ != nullptr && num_init_score_ != num_all_data) {
      Log::Fatal("Initial score size doesn't equal to data");
      delete[] init_score_;
      num_init_score_ = 0;
      init_score_ = nullptr;
    }

    // get local weights
    if (weights_ != nullptr) {
      float* old_weights = weights_;
      num_weights_ = num_data_;
      weights_ = new float[num_data_];
      for (size_t i = 0; i < used_data_indices.size(); ++i) {
        weights_[i] = old_weights[used_data_indices[i]];
      }
      delete[] old_weights;
    }

    // get local query boundaries
    if (query_boundaries_ != nullptr) {
      std::vector<data_size_t> used_query;
      data_size_t data_idx = 0;
      for (data_size_t qid = 0; qid < num_queries_ && data_idx < num_used_data; ++qid) {
        data_size_t start = query_boundaries_[qid];
        data_size_t end = query_boundaries_[qid + 1];
        data_size_t len = end - start;
        if (used_data_indices[data_idx] > start) {
          continue;
        } else if (used_data_indices[data_idx] == start) {
          if (num_used_data >= data_idx + len && used_data_indices[data_idx + len - 1] == end - 1) {
            used_query.push_back(qid);
            data_idx += len;
          } else {
            Log::Fatal("Data partition error, data didn't match queries");
          }
        } else {
          Log::Fatal("Data partition error, data didn't match queries");
        }
      }
      data_size_t * old_query_boundaries = query_boundaries_;
      query_boundaries_ = new data_size_t[used_query.size() + 1];
      num_queries_ = static_cast<data_size_t>(used_query.size());
      query_boundaries_[0] = 0;
      for (data_size_t i = 0; i < num_queries_; ++i) {
        data_size_t qid = used_query[i];
        data_size_t len = old_query_boundaries[qid + 1] - old_query_boundaries[qid];
        query_boundaries_[i + 1] = query_boundaries_[i] + len;
      }
      delete[] old_query_boundaries;
    }

    // get local initial scores
    if (init_score_ != nullptr) {
      float* old_scores = init_score_;
      num_init_score_ = num_data_;
      init_score_ = new float[num_init_score_ * num_class_];
      for (int k = 0; k < num_class_; ++k){
        for (size_t i = 0; i < used_data_indices.size(); ++i) {
          init_score_[k * num_data_ + i] = old_scores[k * num_all_data + used_data_indices[i]];
        }    
      }
      delete[] old_scores;
    }

    // re-load query weight
    LoadQueryWeights();
  }
}


void Metadata::SetInitScore(const float* init_score, data_size_t len) {
  if (len != num_data_ * num_class_) {
    Log::Fatal("Length of initial score is not same with number of data");
  }
  if (init_score_ != nullptr) { delete[] init_score_; }
  num_init_score_ = num_data_;
  init_score_ = new float[len];
  for (data_size_t i = 0; i < len; ++i) {
    init_score_[i] = init_score[i];
  }
}

void Metadata::LoadWeights() {
  num_weights_ = 0;
  std::string weight_filename(data_filename_);
  // default weight file name
  weight_filename.append(".weight");
  TextReader<size_t> reader(weight_filename.c_str(), false);
  reader.ReadAllLines();
  if (reader.Lines().size() <= 0) {
    return;
  }
  Log::Info("Start loading weights");
  num_weights_ = static_cast<data_size_t>(reader.Lines().size());
  weights_ = new float[num_weights_];
  for (data_size_t i = 0; i < num_weights_; ++i) {
    double tmp_weight = 0.0f;
    Common::Atof(reader.Lines()[i].c_str(), &tmp_weight);
    weights_[i] = static_cast<float>(tmp_weight);
  }
}

void Metadata::LoadInitialScore() {
  num_init_score_ = 0;
  if (init_score_filename_[0] == '\0') { return; }
  TextReader<size_t> reader(init_score_filename_, false);
  reader.ReadAllLines();

  Log::Info("Start loading initial scores");
  num_init_score_ = static_cast<data_size_t>(reader.Lines().size());

  if (num_init_score_ <= 0){
      return ;
  }
  
  std::vector<std::string> oneline_init_score = Common::Split(reader.Lines()[0].c_str(), '\t');
  if (num_class_ != static_cast<int>(oneline_init_score.size())) {
    Log::Fatal("Number of classes in configuration and initial score file do not match.");    
  }
  
  init_score_ = new score_t[num_init_score_ * num_class_];
  double tmp = 0.0f;
  
  if (num_class_ == 1){
      for (data_size_t i = 0; i < num_init_score_; ++i) {
        Common::Atof(reader.Lines()[i].c_str(), &tmp);
        init_score_[i] = static_cast<score_t>(tmp);
      }
  } else {
      // first line
      for (int k = 0; k < num_class_; ++k) {
          Common::Atof(oneline_init_score[k].c_str(), &tmp);
          init_score_[k * num_data_] = static_cast<score_t>(tmp);
      }
      // remaining lines
      for (data_size_t i = 1; i < num_init_score_; ++i) {
        oneline_init_score = Common::Split(reader.Lines()[i].c_str(), '\t');  
        if (static_cast<int>(oneline_init_score.size()) != num_class_){
            Log::Fatal("Invalid initial score file. Redundant or insufficient columns.");
        }
        for (int k = 0; k < num_class_; ++k) {
          Common::Atof(oneline_init_score[k].c_str(), &tmp);
          init_score_[k * num_data_ + i] = static_cast<score_t>(tmp);
        }
      }
  }
}

void Metadata::LoadQueryBoundaries() {
  num_queries_ = 0;
  std::string query_filename(data_filename_);
  // default query file name
  query_filename.append(".query");
  TextReader<size_t> reader(query_filename.c_str(), false);
  reader.ReadAllLines();
  if (reader.Lines().size() <= 0) {
    return;
  }
  Log::Info("Start loading query boundries");
  query_boundaries_ = new data_size_t[reader.Lines().size() + 1];
  num_queries_ = static_cast<data_size_t>(reader.Lines().size());
  query_boundaries_[0] = 0;
  for (size_t i = 0; i < reader.Lines().size(); ++i) {
    int tmp_cnt;
    Common::Atoi(reader.Lines()[i].c_str(), &tmp_cnt);
    query_boundaries_[i + 1] = query_boundaries_[i] + static_cast<data_size_t>(tmp_cnt);
  }
}

void Metadata::LoadQueryWeights() {
  if (weights_ == nullptr || query_boundaries_ == nullptr) {
    return;
  }
  Log::Info("Start loading query weights");
  query_weights_ = new float[num_queries_];
  for (data_size_t i = 0; i < num_queries_; ++i) {
    query_weights_[i] = 0.0f;
    for (data_size_t j = query_boundaries_[i]; j < query_boundaries_[i + 1]; ++j) {
      query_weights_[i] += weights_[j];
    }
    query_weights_[i] /= (query_boundaries_[i + 1] - query_boundaries_[i]);
  }
}

void Metadata::LoadFromMemory(const void* memory) {
  const char* mem_ptr = reinterpret_cast<const char*>(memory);

  num_data_ = *(reinterpret_cast<const data_size_t*>(mem_ptr));
  mem_ptr += sizeof(num_data_);
  num_weights_ = *(reinterpret_cast<const data_size_t*>(mem_ptr));
  mem_ptr += sizeof(num_weights_);
  num_queries_ = *(reinterpret_cast<const data_size_t*>(mem_ptr));
  mem_ptr += sizeof(num_queries_);

  if (label_ != nullptr) { delete[] label_; }
  label_ = new float[num_data_];
  std::memcpy(label_, mem_ptr, sizeof(float)*num_data_);
  mem_ptr += sizeof(float)*num_data_;

  if (num_weights_ > 0) {
    if (weights_ != nullptr) { delete[] weights_; }
    weights_ = new float[num_weights_];
    std::memcpy(weights_, mem_ptr, sizeof(float)*num_weights_);
    mem_ptr += sizeof(float)*num_weights_;
  }
  if (num_queries_ > 0) {
    if (query_boundaries_ != nullptr) { delete[] query_boundaries_; }
    query_boundaries_ = new data_size_t[num_queries_ + 1];
    std::memcpy(query_boundaries_, mem_ptr, sizeof(data_size_t)*(num_queries_ + 1));
    mem_ptr += sizeof(data_size_t)*(num_queries_ + 1);
  }
  if (num_weights_ > 0 && num_queries_ > 0) {
    if (query_weights_ != nullptr) { delete[] query_weights_; }
    query_weights_ = new float[num_queries_];
    std::memcpy(query_weights_, mem_ptr, sizeof(float)*num_queries_);
    mem_ptr += sizeof(float)*num_queries_;
  }
}

void Metadata::SaveBinaryToFile(FILE* file) const {
  fwrite(&num_data_, sizeof(num_data_), 1, file);
  fwrite(&num_weights_, sizeof(num_weights_), 1, file);
  fwrite(&num_queries_, sizeof(num_queries_), 1, file);
  fwrite(label_, sizeof(float), num_data_, file);
  if (weights_ != nullptr) {
    fwrite(weights_, sizeof(float), num_weights_, file);
  }
  if (query_boundaries_ != nullptr) {
    fwrite(query_boundaries_, sizeof(data_size_t), num_queries_ + 1, file);
  }
  if (query_weights_ != nullptr) {
    fwrite(query_weights_, sizeof(float), num_queries_, file);
  }

}

size_t Metadata::SizesInByte() const  {
  size_t size = sizeof(num_data_) + sizeof(num_weights_)
    + sizeof(num_queries_);
  size += sizeof(float) * num_data_;
  if (weights_ != nullptr) {
    size += sizeof(float) * num_weights_;
  }
  if (query_boundaries_ != nullptr) {
    size += sizeof(data_size_t) * (num_queries_ + 1);
  }
  if (query_weights_ != nullptr) {
    size += sizeof(float) * num_queries_;
  }
  return size;
}


}  // namespace LightGBM
