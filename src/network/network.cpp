#include <LightGBM/network.h>

#include <LightGBM/utils/common.h>

#include "linkers.h"

#include <cstring>
#include <cstdlib>

namespace LightGBM {

// static member definition
THREAD_LOCAL int Network::num_machines_ = 1;
THREAD_LOCAL int Network::rank_ = 0;
THREAD_LOCAL std::unique_ptr<Linkers> Network::linkers_;
THREAD_LOCAL BruckMap Network::bruck_map_;
THREAD_LOCAL RecursiveHalvingMap Network::recursive_halving_map_;
THREAD_LOCAL std::vector<int> Network::block_start_;
THREAD_LOCAL std::vector<int>  Network::block_len_;
THREAD_LOCAL int Network::buffer_size_;
THREAD_LOCAL std::vector<char> Network::buffer_;
THREAD_LOCAL AllreduceFunction Network::AllreduceFuncPtr_ = NULL;
THREAD_LOCAL ReduceScatterFunction Network::ReduceScatterFuncPtr_ = NULL;
THREAD_LOCAL AllgatherFunction Network::AllgatherFuncPtr_ = NULL;


void Network::Init(NetworkConfig config) {
  if (config.num_machines > 1) {
    linkers_.reset(new Linkers(config));
    rank_ = linkers_->rank();
    num_machines_ = linkers_->num_machines();
    bruck_map_ = linkers_->bruck_map();
    recursive_halving_map_ = linkers_->recursive_halving_map();
    block_start_ = std::vector<int>(num_machines_);
    block_len_ = std::vector<int>(num_machines_);
    buffer_size_ = 1024 * 1024;
    buffer_.resize(buffer_size_);
    Log::Info("Local rank: %d, total number of machines: %d", rank_, num_machines_);
  }
}

void Network::Dispose() {
  num_machines_ = 1;
  rank_ = 0;
  linkers_.reset(new Linkers());
}

void Network::Allreduce(char* input, int input_size, int type_size, char* output, const ReduceFunction& reducer) {
  if (num_machines_ <= 1) {
    Log::Fatal("Please initilize the network interface first");
  }
  if (AllreduceFuncPtr_ != NULL) {
    return AllreduceFuncPtr_(input, input_size, type_size, output, reducer);
  }
  int count = input_size / type_size;
  // if small package or small count , do it by all gather.(reduce the communication times.)
  if (count < num_machines_ || input_size < 4096) {
    AllreduceByAllGather(input, input_size, output, reducer);
    return;
  }
  // assign the blocks to every rank.
  int step = (count + num_machines_ - 1) / num_machines_;
  if (step < 1) {
    step = 1;
  }
  block_start_[0] = 0;
  for (int i = 0; i < num_machines_ - 1; ++i) {
    block_len_[i] = std::min(step * type_size, input_size - block_start_[i]);
    block_start_[i + 1] = block_start_[i] + block_len_[i];
  }
  block_len_[num_machines_ - 1] = input_size - block_start_[num_machines_ - 1];
  // do reduce scatter
  ReduceScatter(input, input_size, block_start_.data(), block_len_.data(), output, reducer);
  // do all gather
  Allgather(output, input_size, block_start_.data(), block_len_.data(), output);
}

void Network::AllreduceByAllGather(char* input, int input_size, char* output, const ReduceFunction& reducer) {
  if (num_machines_ <= 1) {
    Log::Fatal("Please initilize the network interface first");
  }
  // assign blocks
  int all_size = input_size * num_machines_;
  block_start_[0] = 0;
  block_len_[0] = input_size;
  for (int i = 1; i < num_machines_; ++i) {
    block_start_[i] = block_start_[i - 1] + block_len_[i - 1];
    block_len_[i] = input_size;
  }
  // need use buffer here, since size of "output" is smaller than size after all gather
  if (input_size*num_machines_ > buffer_size_) {
    buffer_size_ = input_size*num_machines_;
    buffer_.resize(buffer_size_);
  }

  Allgather(input, all_size, block_start_.data(), block_len_.data(), buffer_.data());
  for (int i = 1; i < num_machines_; ++i) {
    reducer(buffer_.data() + block_start_[i], buffer_.data() + block_start_[0], input_size);
  }
  // copy back
  std::memcpy(output, buffer_.data(), input_size);
}

void Network::Allgather(char* input, int send_size, char* output) {
  if (num_machines_ <= 1) {
    Log::Fatal("Please initilize the network interface first");
  }
  if (num_machines_ <= 1) { return; }
  if (AllgatherFuncPtr_ != NULL) {
    return AllgatherFuncPtr_(input, send_size, output);
  }
  // assign blocks
  block_start_[0] = 0;
  block_len_[0] = send_size;
  for (int i = 1; i < num_machines_; ++i) {
    block_start_[i] = block_start_[i - 1] + block_len_[i - 1];
    block_len_[i] = send_size;
  }
  // start all gather
  Allgather(input, send_size * num_machines_, block_start_.data(), block_len_.data(), output);
}

void Network::Allgather(char* input, int all_size, const int* block_start, const int* block_len, char* output) {
  if (num_machines_ <= 1) {
    Log::Fatal("Please initilize the network interface first");
  }
  int write_pos = 0;
  // use output as receive buffer
  std::memcpy(output, input, block_len[rank_]);
  write_pos += block_len[rank_];
  int accumulated_block = 1;
  for (int i = 0; i < bruck_map_.k; ++i) {
    // get current local block size
    int cur_block_size = std::min(1 << i, num_machines_ - accumulated_block);
    // get out rank
    int out_rank = bruck_map_.out_ranks[i];
    // get in rank
    int in_rank = bruck_map_.in_ranks[i];
    // get send information
    int need_send_len = 0;
    // get recv information
    int need_recv_len = 0;
    for (int j = 0; j < cur_block_size; ++j) {
      need_send_len += block_len[(rank_ + j) % num_machines_];
      need_recv_len += block_len[(rank_ + accumulated_block + j) % num_machines_];
    }
    // send and recv at same time
    linkers_->SendRecv(out_rank, output, need_send_len, in_rank, output + write_pos, need_recv_len);
    write_pos += need_recv_len;
    accumulated_block += cur_block_size;
  }
  // rotate in-place
  std::reverse<char*>(output, output + all_size);
  std::reverse<char*>(output, output + block_start[rank_]);
  std::reverse<char*>(output + block_start[rank_], output + all_size);
}

void Network::ReduceScatter(char* input, int input_size, const int* block_start, const int* block_len, char* output, const ReduceFunction& reducer) {
  if (num_machines_ <= 1) {
    Log::Fatal("Please initilize the network interface first");
  }
  if (ReduceScatterFuncPtr_ != NULL) {
    return ReduceScatterFuncPtr_(input, input_size, block_start, block_len, output, reducer);
  }
  if (recursive_halving_map_.need_pairwise) {
    for (int i = 1; i < num_machines_; ++i) {
      int out_rank = (rank_ + i) % num_machines_;
      int in_rank = (rank_ - i + num_machines_) % num_machines_;
      linkers_->SendRecv(out_rank, input + block_start[out_rank], block_len[out_rank], in_rank, output, block_len[rank_]);
      reducer(output, input + block_start[rank_], block_len[rank_]);
    }
  } else {
    for (int i = 0; i < recursive_halving_map_.k; ++i) {
      // get target
      int target = recursive_halving_map_.ranks[i];
      int send_block_start = recursive_halving_map_.send_block_start[i];
      int recv_block_start = recursive_halving_map_.recv_block_start[i];
      // get send information
      int send_size = 0;
      for (int j = 0; j < recursive_halving_map_.send_block_len[i]; ++j) {
        send_size += block_len[send_block_start + j];
      }
      // get recv information
      int need_recv_cnt = 0;
      for (int j = 0; j < recursive_halving_map_.recv_block_len[i]; ++j) {
        need_recv_cnt += block_len[recv_block_start + j];
      }
      // send and recv at same time
      linkers_->SendRecv(target, input + block_start[send_block_start], send_size, target, output, need_recv_cnt);
      // reduce
      reducer(output, input + block_start[recv_block_start], need_recv_cnt);
    }
  }
  // copy result
  std::memcpy(output, input + block_start[rank_], block_len[rank_]);
}

}  // namespace LightGBM
