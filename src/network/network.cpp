#include <LightGBM/network.h>

#include <LightGBM/utils/common.h>

#include "linkers.h"

#include <cstring>
#include <cstdlib>

namespace LightGBM {

// static member definition
int Network::num_machines_;
int Network::rank_;
std::unique_ptr<Linkers> Network::linkers_;
BruckMap Network::bruck_map_;
RecursiveHalvingMap Network::recursive_halving_map_;
std::vector<int> Network::block_start_;
std::vector<int>  Network::block_len_;
int Network::buffer_size_;
std::vector<char> Network::buffer_;

void Network::Init(NetworkConfig config) {
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

void Network::Dispose() {

}

void Network::Allreduce(char* input, int input_size, int type_size, char* output, const ReduceFunction& reducer) {
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
    // get send information
    int send_len = 0;
    for (int j = 0; j < cur_block_size; ++j) {
      send_len += block_len[(rank_ + j) % num_machines_];
    }
    // get in rank
    int in_rank = bruck_map_.in_ranks[i];
    // get recv information
    int need_recv_cnt = 0;
    for (int j = 0; j < cur_block_size; ++j) {
      need_recv_cnt += block_len[(rank_ + accumulated_block + j) % num_machines_];
    }
    // send and recv at same time
    linkers_->SendRecv(out_rank, output, send_len, in_rank, output + write_pos, need_recv_cnt);
    write_pos += need_recv_cnt;
    accumulated_block += cur_block_size;
  }
  // rotate in-place
  std::reverse<char*>(output, output + all_size);
  std::reverse<char*>(output, output + block_start[rank_]);
  std::reverse<char*>(output + block_start[rank_], output + all_size);
}

void Network::ReduceScatter(char* input, int input_size, const int* block_start, const int* block_len, char* output, const ReduceFunction& reducer) {
  bool is_powerof_2 = (num_machines_ & (num_machines_ - 1)) == 0;
  if (!is_powerof_2) {
    if (recursive_halving_map_.type == RecursiveHalvingNodeType::Other) {
      // send local data to neighbor first
      linkers_->Send(recursive_halving_map_.neighbor, input, input_size);
    } else if (recursive_halving_map_.type == RecursiveHalvingNodeType::GroupLeader) {
      // receive neighbor data first
      int need_recv_cnt = input_size;
      linkers_->Recv(recursive_halving_map_.neighbor, output, need_recv_cnt);
      // reduce
      reducer(output, input, input_size);
    }
  }
  // start recursive halfing
  if (recursive_halving_map_.type != RecursiveHalvingNodeType::Other) {
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
  if (!is_powerof_2) {
    if (recursive_halving_map_.type == RecursiveHalvingNodeType::GroupLeader) {
      // send result to neighbor
      linkers_->Send(recursive_halving_map_.neighbor,
                     input + block_start[recursive_halving_map_.neighbor],
                     block_len[recursive_halving_map_.neighbor]);
    } else if (recursive_halving_map_.type == RecursiveHalvingNodeType::Other) {
      // receive result from neighbor
      int need_recv_cnt = block_len[rank_];
      linkers_->Recv(recursive_halving_map_.neighbor, output, need_recv_cnt);
      return;
    }
  }
  // copy result
  std::memcpy(output, input + block_start[rank_], block_len[rank_]);
}

}  // namespace LightGBM
