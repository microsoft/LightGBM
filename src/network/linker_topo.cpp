#include <LightGBM/network.h>

#include <LightGBM/utils/common.h>
#include <LightGBM/utils/log.h>

#include <string>
#include <vector>
#include <unordered_map>

namespace LightGBM {


BruckMap::BruckMap() {
  k = 0;
}

BruckMap::BruckMap(int n) {
  k = n;
  // default set to -1
  for (int i = 0; i < n; ++i) {
    in_ranks.push_back(-1);
    out_ranks.push_back(-1);
  }
}

BruckMap BruckMap::Construct(int rank, int num_machines) {
  // distance at k-th communication, distance[k] = 2^k
  std::vector<int> distance;
  int k = 0;
  for (k = 0; (1 << k) < num_machines; ++k) {
    distance.push_back(1 << k);
  }
  BruckMap bruckMap(k);
  for (int j = 0; j < k; ++j) {
    // set incoming rank at k-th commuication
    const int in_rank = (rank + distance[j]) % num_machines;
    bruckMap.in_ranks[j] = in_rank;
    // set outgoing rank at k-th commuication
    const int out_rank = (rank - distance[j] + num_machines) % num_machines;
    bruckMap.out_ranks[j] = out_rank;
  }
  return bruckMap;
}

RecursiveHalvingMap::RecursiveHalvingMap() {
  k = 0;
  is_prof2 = true;
  num_remain = 0;
}

RecursiveHalvingMap::RecursiveHalvingMap(int in_k, int in_remain, int in_rank, bool is_power_of2) {
  k = in_k;
  is_prof2 = is_power_of2;
  num_remain = in_remain;
  virtual_rank = in_rank;
  for (int i = 0; i < k; ++i) {
    // defalut set as -1
    ranks.push_back(-1);
    send_block_start.push_back(-1);
    send_block_len.push_back(-1);
    recv_block_start.push_back(-1);
    recv_block_len.push_back(-1);
  }
}

RecursiveHalvingMap RecursiveHalvingMap::Construct(int rank, int num_machines) {
  // construct all recursive halving map for all machines
  int k = 0;
  while ((1 << k) <= num_machines) { ++k; }
  // let 1 << k <= num_machines
  --k;
  // distance of each communication
  std::vector<int> distance;
  for (int i = 0; i < k; ++i) {
    distance.push_back(1 << (k - 1 - i));
  }

  int remain = num_machines - (1 << k);
  int virtual_rank = rank;
  // if virtual_rank not -1 will not excute recursize halving algorithm
  if (rank < 2 * remain) {
    if (rank % 2 == 0) {
      virtual_rank = -1;
    } else {
      virtual_rank = rank / 2;
    }
  } else {
    virtual_rank = rank - remain;
  }

  bool is_power_of2 = false;
  if ((1 << k) == num_machines) { is_power_of2 = true; }
  RecursiveHalvingMap rec_map(k, remain, virtual_rank, is_power_of2);
  if (virtual_rank != -1) {
    for (int i = 0; i < k; ++i) {
      // communication direction, %2 == 0 is positive
      const int dir = ((virtual_rank / distance[i]) % 2 == 0) ? 1 : -1;
      // neighbor at k-th communication
      const int next_node_idx = virtual_rank + dir * distance[i];
      rec_map.ranks[i] = next_node_idx;
      // receive data block at k-th communication
      const int recv_block_start = virtual_rank / distance[i];
      rec_map.recv_block_start[i] = recv_block_start * distance[i];
      rec_map.recv_block_len[i] = distance[i];
      // send data block at k-th communication
      const int send_block_start = next_node_idx / distance[i];
      rec_map.send_block_start[i] = send_block_start * distance[i];
      rec_map.send_block_len[i] = distance[i];
    }
  }
  return rec_map;
}

}  // namespace LightGBM

