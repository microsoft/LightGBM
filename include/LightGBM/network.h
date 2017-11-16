#ifndef LIGHTGBM_NETWORK_H_
#define LIGHTGBM_NETWORK_H_

#include <LightGBM/utils/log.h>

#include <LightGBM/meta.h>
#include <LightGBM/config.h>

#include <functional>
#include <vector>
#include <memory>

namespace LightGBM {

/*! \brief forward declaration */
class Linkers;

/*! \brief The network structure for all_gather */
class BruckMap {
public:
  /*! \brief The communication times for one all gather operation */
  int k;
  /*! \brief in_ranks[i] means the incomming rank on i-th communication */
  std::vector<int> in_ranks;
  /*! \brief out_ranks[i] means the out rank on i-th communication */
  std::vector<int> out_ranks;
  BruckMap();
  explicit BruckMap(int n);
  /*!
  * \brief Create the object of bruck map
  * \param rank Rank of this machine
  * \param num_machines The total number of machines
  * \return The object of bruck map
  */
  static BruckMap Construct(int rank, int num_machines);
};

/*! \brief Network structure for recursive halving algorithm */
class RecursiveHalvingMap {
public:
  bool need_pairwise;
  /*! \brief Communication times for one recursize halving algorithm  */
  int k;
  /*! \brief ranks[i] means the machines that will communicate with on i-th communication*/
  std::vector<int> ranks;
  /*! \brief  send_block_start[i] means send block start index at i-th communication*/
  std::vector<int> send_block_start;
  /*! \brief  send_block_start[i] means send block size at i-th communication*/
  std::vector<int> send_block_len;
  /*! \brief  send_block_start[i] means recv block start index at i-th communication*/
  std::vector<int> recv_block_start;
  /*! \brief  send_block_start[i] means recv block size  at i-th communication*/
  std::vector<int> recv_block_len;

  RecursiveHalvingMap();

  RecursiveHalvingMap(int k, bool in_need_pairwise);

  /*!
  * \brief Create the object of recursive halving map
  * \param rank Rank of this machine
  * \param num_machines The total number of machines
  * \return The object of recursive halving map
  */
  static RecursiveHalvingMap Construct(int rank, int num_machines);
};

/*! \brief A static class that contains some collective communication algorithm */
class Network {
public:
  /*!
  * \brief Initialize
  * \param config Config of network setting
  */
  static void Init(NetworkConfig config);
  /*! \brief Free this static class */
  static void Dispose();
  /*! \brief Get rank of this machine */
  static inline int rank();
  /*! \brief Get total number of machines */
  static inline int num_machines();

  /*!
  * \brief Perform all_reduce. if data size is small,
           will perform AllreduceByAllGather, else with call ReduceScatter followed allgather
  * \param input Input data
  * \param input_size The size of input data
  * \param type_size The size of one object in the reduce function
  * \param output Output result
  * \param reducer Reduce function
  */
  static void Allreduce(char* input, int input_size, int type_size,
    char* output, const ReduceFunction& reducer);

  /*!
  * \brief Perform all_reduce by using all_gather. it can be use to reduce communication time when data is small
  * \param input Input data
  * \param input_size The size of input data
  * \param output Output result
  * \param reducer Reduce function
  */
  static void AllreduceByAllGather(char* input, int input_size, char* output,
    const ReduceFunction& reducer);

  /*!
  * \brief Performing all_gather by using bruck algorithm. 
           Communication times is O(log(n)), and communication cost is O(send_size * number_machine)
  *        It can be used when all nodes have same input size.
  * \param input Input data
  * \param send_size The size of input data
  * \param output Output result
  */
  static void Allgather(char* input, int send_size, char* output);

  /*!
  * \brief Performing all_gather by using bruck algorithm. 
           Communication times is O(log(n)), and communication cost is O(all_size)
  *        It can be used when nodes have different input size.
  * \param input Input data
  * \param all_size The size of input data
  * \param block_start The block start for different machines
  * \param block_len The block size for different machines
  * \param output Output result
  */
  static void Allgather(char* input, int all_size, const int* block_start,
    const int* block_len, char* output);

  /*!
  * \brief Perform reduce scatter by using recursive halving algorithm. 
           Communication times is O(log(n)), and communication cost is O(input_size)
  * \param input Input data
  * \param input_size The size of input data
  * \param block_start The block start for different machines
  * \param block_len The block size for different machines
  * \param output Output result
  * \param reducer Reduce function
  */
  static void ReduceScatter(char* input, int input_size,
    const int* block_start, const int* block_len, char* output,
    const ReduceFunction& reducer);

  template<class T>
  static T GlobalSyncUpByMin(T& local) {
    T global = local;
    Allreduce(reinterpret_cast<char*>(&local),
              sizeof(local), sizeof(local),
              reinterpret_cast<char*>(&global),
              [] (const char* src, char* dst, int len) {
      int used_size = 0;
      const int type_size = sizeof(T);
      const T *p1;
      T *p2;
      while (used_size < len) {
        p1 = reinterpret_cast<const T *>(src);
        p2 = reinterpret_cast<T *>(dst);
        if (*p1 < *p2) {
          std::memcpy(dst, src, type_size);
        }
        src += type_size;
        dst += type_size;
        used_size += type_size;
      }
    });
    return global;
  }

  template<class T>
  static T GlobalSyncUpByMax(T& local) {
    T global = local;
    Allreduce(reinterpret_cast<char*>(&local),
              sizeof(local), sizeof(local),
              reinterpret_cast<char*>(&global),
              [] (const char* src, char* dst, int len) {
      int used_size = 0;
      const int type_size = sizeof(T);
      const T *p1;
      T *p2;
      while (used_size < len) {
        p1 = reinterpret_cast<const T *>(src);
        p2 = reinterpret_cast<T *>(dst);
        if (*p1 > *p2) {
          std::memcpy(dst, src, type_size);
        }
        src += type_size;
        dst += type_size;
        used_size += type_size;
      }
    });
    return global;
  }
  /*! \brief set variables and function ptrs */
  static void SetRank(int rank) { rank_ = rank;}
  static void SetNumMachines(int num_machines) { num_machines_ = num_machines; }
  static void SetAllReduce(AllreduceFunction AllreduceFuncPtr) { AllreduceFuncPtr_ = AllreduceFuncPtr;}
  static void SetReduceScatter(ReduceScatterFunction ReduceScatterFuncPtr) { ReduceScatterFuncPtr_ = ReduceScatterFuncPtr; }
  static void SetAllgather(AllgatherFunction AllgatherFuncPtr) { AllgatherFuncPtr_ = AllgatherFuncPtr; }

private:
  /*! \brief Number of all machines */
  static THREAD_LOCAL int num_machines_;
  /*! \brief Rank of local machine */
  static THREAD_LOCAL int rank_;
  /*! \brief The network interface, provide send/recv functions  */
  static THREAD_LOCAL std::unique_ptr<Linkers> linkers_;
  /*! \brief Bruck map for all gather algorithm*/
  static THREAD_LOCAL BruckMap bruck_map_;
  /*! \brief Recursive halving map for reduce scatter */
  static THREAD_LOCAL RecursiveHalvingMap recursive_halving_map_;
  /*! \brief Buffer to store block start index */
  static THREAD_LOCAL std::vector<int> block_start_;
  /*! \brief Buffer to store block size */
  static THREAD_LOCAL std::vector<int> block_len_;
  /*! \brief Buffer  */
  static THREAD_LOCAL std::vector<char> buffer_;
  /*! \brief Size of buffer_ */
  static THREAD_LOCAL int buffer_size_;
  /*! \brief Funcs*/
  static THREAD_LOCAL AllreduceFunction AllreduceFuncPtr_;
  static THREAD_LOCAL ReduceScatterFunction ReduceScatterFuncPtr_;
  static THREAD_LOCAL AllgatherFunction AllgatherFuncPtr_;
};

inline int Network::rank() {
  return rank_;
}

inline int Network::num_machines() {
  return num_machines_;
}

}  // namespace LightGBM

#endif   // LightGBM_NETWORK_H_
