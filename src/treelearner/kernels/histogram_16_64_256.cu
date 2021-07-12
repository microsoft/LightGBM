/*!
 * Copyright (c) 2020 IBM Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#include <LightGBM/meta.h>

#include <cstdint>
#include <cstdio>

#include "histogram_16_64_256.hu"

namespace LightGBM {

// atomic add for float number in local memory
inline __device__ void atomic_local_add_f(acc_type *addr, const acc_type val) {
    atomicAdd(addr, static_cast<acc_type>(val));
}

// histogram16 stuff
#ifdef ENABLE_ALL_FEATURES
#ifdef IGNORE_INDICES
#define KERNEL_NAME histogram16_fulldata
#else  // IGNORE_INDICES
#define KERNEL_NAME histogram16
#endif  // IGNORE_INDICES
#else  // ENABLE_ALL_FEATURES
#error "ENABLE_ALL_FEATURES should always be 1"
#define KERNEL_NAME histogram16
#endif  // ENABLE_ALL_FEATURES
#define NUM_BINS 16
#define LOCAL_MEM_SIZE ((sizeof(unsigned int) + 2 * sizeof(acc_type)) * NUM_BINS)

// this function will be called by histogram16
// we have one sub-histogram of one feature in local memory, and need to read others
inline void __device__ within_kernel_reduction16x4(const acc_type* __restrict__ feature_sub_hist,
                           const unsigned int skip_id,
                           const unsigned int old_val_cont_bin0,
                           const uint16_t num_sub_hist,
                           acc_type* __restrict__ output_buf,
                           acc_type* __restrict__ local_hist,
                           const size_t power_feature_workgroups) {
    const uint16_t ltid = threadIdx.x;
    acc_type grad_bin = local_hist[ltid * 2];
    acc_type hess_bin = local_hist[ltid * 2 + 1];
    unsigned int* __restrict__ local_cnt = reinterpret_cast<unsigned int *>(local_hist + 2 * NUM_BINS);

    unsigned int cont_bin;
    if (power_feature_workgroups != 0) {
      cont_bin = ltid ? local_cnt[ltid] : old_val_cont_bin0;
    } else {
      cont_bin = local_cnt[ltid];
    }
    uint16_t i;

    if (power_feature_workgroups != 0) {
        // add all sub-histograms for feature
        const acc_type* __restrict__ p = feature_sub_hist + ltid;
        for (i = 0; i < skip_id; ++i) {
            grad_bin += *p;          p += NUM_BINS;
            hess_bin += *p;          p += NUM_BINS;
            cont_bin += as_acc_int_type(*p); p += NUM_BINS;
        }

        // skip the counters we already have
        p += 3 * NUM_BINS;

        for (i = i + 1; i < num_sub_hist; ++i) {
            grad_bin += *p;          p += NUM_BINS;
            hess_bin += *p;          p += NUM_BINS;
            cont_bin += as_acc_int_type(*p); p += NUM_BINS;
        }
    }
    __syncthreads();

    output_buf[ltid * 2 + 0] = grad_bin;
    output_buf[ltid * 2 + 1] = hess_bin;
}

#if USE_CONSTANT_BUF == 1
__kernel void KERNEL_NAME(__global const uchar* restrict feature_data_base,
                      __constant const uchar* restrict feature_masks __attribute__((max_constant_size(65536))),
                      const data_size_t feature_size,
                      __constant const data_size_t* restrict data_indices __attribute__((max_constant_size(65536))),
                      const data_size_t num_data,
                      __constant const score_t* restrict ordered_gradients __attribute__((max_constant_size(65536))),
#if CONST_HESSIAN == 0
                      __constant const score_t* restrict ordered_hessians __attribute__((max_constant_size(65536))),
#else
                      const score_t const_hessian,
#endif
                      char* __restrict__ output_buf,
                      volatile int * sync_counters,
                      acc_type* __restrict__ hist_buf_base,
                      const size_t power_feature_workgroups) {
#else
__global__ void KERNEL_NAME(const uchar* feature_data_base,
                      const uchar* __restrict__ feature_masks,
                      const data_size_t feature_size,
                      const data_size_t* data_indices,
                      const data_size_t num_data,
                      const score_t*  ordered_gradients,
#if CONST_HESSIAN == 0
                      const score_t*  ordered_hessians,
#else
                      const score_t const_hessian,
#endif
                      char* __restrict__ output_buf,
                      volatile int * sync_counters,
                      acc_type* __restrict__ hist_buf_base,
                      const size_t power_feature_workgroups) {
#endif
    // allocate the local memory array aligned with float2, to guarantee correct alignment on NVIDIA platforms
    // otherwise a "Misaligned Address" exception may occur
    __shared__ float2 shared_array[LOCAL_MEM_SIZE/sizeof(float2)];
    const unsigned int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint16_t ltid = threadIdx.x;
    const uint16_t lsize = NUM_BINS;  // get_local_size(0);
    const uint16_t group_id = blockIdx.x;

    // local memory per workgroup is 3 KB
    // clear local memory
    unsigned int *ptr = reinterpret_cast<unsigned int *>(shared_array);
    for (int i = ltid; i < LOCAL_MEM_SIZE/sizeof(unsigned int); i += lsize) {
        ptr[i] = 0;
    }
    __syncthreads();
    // gradient/hessian histograms
    // assume this starts at 32 * 4 = 128-byte boundary // What does it mean? boundary??
    // total size: 2 * 256 * size_of(float) = 2 KB
    // organization: each feature/grad/hessian is at a different bank,
    //               as independent of the feature value as possible
    acc_type *gh_hist = reinterpret_cast<acc_type *>(shared_array);

    // counter histogram
    // total size: 256 * size_of(unsigned int) = 1 KB
    unsigned int *cnt_hist = reinterpret_cast<unsigned int *>(gh_hist + 2 * NUM_BINS);

    // odd threads (1, 3, ...) compute histograms for hessians first
    // even thread (0, 2, ...) compute histograms for gradients first
    // etc.
    uchar is_hessian_first = ltid & 1;

    uint16_t feature_id = group_id >> power_feature_workgroups;

    // each 2^POWER_FEATURE_WORKGROUPS workgroups process on one feature (compile-time constant)
    // feature_size is the number of examples per feature
    const uchar *feature_data = feature_data_base + feature_id * feature_size;

    // size of threads that process this feature4
    const unsigned int subglobal_size = lsize * (1 << power_feature_workgroups);

    // equavalent thread ID in this subgroup for this feature4
    const unsigned int subglobal_tid  = gtid - feature_id * subglobal_size;


    data_size_t ind;
    data_size_t ind_next;
    #ifdef IGNORE_INDICES
    ind = subglobal_tid;
    #else
    ind = data_indices[subglobal_tid];
    #endif

    // extract feature mask, when a byte is set to 0, that feature is disabled
    uchar feature_mask = feature_masks[feature_id];
    // exit if the feature is masked
    if (!feature_mask) {
        return;
    } else {
        feature_mask = feature_mask - 1;  // feature_mask is used for get feature (1: 4bit feature, 0: 8bit feature)
    }

    // STAGE 1: read feature data, and gradient and hessian
    // first half of the threads read feature data from global memory
    // We will prefetch data into the "next" variable at the beginning of each iteration
    uchar feature;
    uchar feature_next;
    uint16_t bin;

    feature = feature_data[ind >> feature_mask];
    if (feature_mask) {
       feature = (feature >> ((ind & 1) << 2)) & 0xf;
    }
    bin = feature;
    acc_type grad_bin = 0.0f, hess_bin = 0.0f;
    acc_type *addr_bin;

    // store gradient and hessian
    score_t grad, hess;
    score_t grad_next, hess_next;
    grad = ordered_gradients[ind];
    #if CONST_HESSIAN == 0
    hess = ordered_hessians[ind];
    #endif

    // there are 2^POWER_FEATURE_WORKGROUPS workgroups processing each feature4
    for (unsigned int i = subglobal_tid; i < num_data; i += subglobal_size) {
        // prefetch the next iteration variables
        // we don't need boundary check because we have made the buffer large
        int i_next = i + subglobal_size;
        #ifdef IGNORE_INDICES
        // we need to check to bounds here
        ind_next = i_next < num_data ? i_next : i;
        #else
        ind_next = data_indices[i_next];
        #endif

        grad_next = ordered_gradients[ind_next];
        #if CONST_HESSIAN == 0
        hess_next = ordered_hessians[ind_next];
        #endif

        // STAGE 2: accumulate gradient and hessian
        if (bin != feature) {
            addr_bin = gh_hist + bin * 2 + is_hessian_first;
            #if CONST_HESSIAN == 0
            acc_type acc_bin = is_hessian_first ? hess_bin : grad_bin;
            atomic_local_add_f(addr_bin, acc_bin);

            addr_bin = addr_bin + 1 - 2 * is_hessian_first;
            acc_bin = is_hessian_first ? grad_bin : hess_bin;
            atomic_local_add_f(addr_bin, acc_bin);

            #elif CONST_HESSIAN == 1
            atomic_local_add_f(addr_bin, grad_bin);
            #endif

            bin = feature;
            grad_bin = grad;
            hess_bin = hess;
        } else {
            grad_bin += grad;
            hess_bin += hess;
        }

        // prefetch the next iteration variables
        feature_next = feature_data[ind_next >> feature_mask];

        // STAGE 3: accumulate counter
        atomicAdd(cnt_hist + feature, 1);

        // STAGE 4: update next stat
        grad = grad_next;
        hess = hess_next;
        if (!feature_mask) {
            feature = feature_next;
        } else {
            feature = (feature_next >> ((ind_next & 1) << 2)) & 0xf;
        }
    }


    addr_bin = gh_hist + bin * 2 + is_hessian_first;
    #if CONST_HESSIAN == 0
    acc_type acc_bin = is_hessian_first ? hess_bin : grad_bin;
    atomic_local_add_f(addr_bin, acc_bin);

    addr_bin = addr_bin + 1 - 2 * is_hessian_first;
    acc_bin = is_hessian_first ? grad_bin : hess_bin;
    atomic_local_add_f(addr_bin, acc_bin);

    #elif CONST_HESSIAN == 1
    atomic_local_add_f(addr_bin, grad_bin);
    #endif
    __syncthreads();

    #if CONST_HESSIAN == 1
    // make a final reduction
    gh_hist[ltid * 2] += gh_hist[ltid * 2 + 1];
    gh_hist[ltid * 2 + 1] = const_hessian * cnt_hist[ltid];  // counter move to this position
    __syncthreads();
    #endif

#if POWER_FEATURE_WORKGROUPS != 0
    acc_type *__restrict__ output = reinterpret_cast<acc_type *>(output_buf) + group_id * 3 * NUM_BINS;
    // write gradients and Hessians
    acc_type *__restrict__ ptr_f = output;
    for (uint16_t i = ltid; i < 2 * NUM_BINS; i += lsize) {
        // even threads read gradients, odd threads read Hessians
        acc_type value = gh_hist[i];
        ptr_f[(i & 1) * NUM_BINS + (i >> 1)] = value;
    }
    // write counts
    acc_int_type *__restrict__ ptr_i = reinterpret_cast<acc_int_type *>(output + 2 * NUM_BINS);
    for (uint16_t i = ltid; i < NUM_BINS; i += lsize) {
        unsigned int value = cnt_hist[i];
        ptr_i[i] = value;
    }
    __syncthreads();
    __threadfence();
    unsigned int * counter_val = cnt_hist;
    // backup the old value
    unsigned int old_val = *counter_val;
    if (ltid == 0) {
        // all workgroups processing the same feature add this counter
        *counter_val = atomicAdd(const_cast<int*>(sync_counters + feature_id), 1);
    }
    // make sure everyone in this workgroup is here
    __syncthreads();
    // everyone in this workgroup: if we are the last workgroup, then do reduction!
    if (*counter_val == (1 << power_feature_workgroups) - 1) {
        if (ltid == 0) {
            sync_counters[feature_id] = 0;
        }
#else
    }
    // only 1 work group, no need to increase counter
    // the reduction will become a simple copy
    {
        unsigned int old_val;  // dummy
#endif
        // locate our feature's block in output memory
        unsigned int output_offset = (feature_id << power_feature_workgroups);
        acc_type const * __restrict__ feature_subhists =
                 reinterpret_cast<acc_type *>(output_buf) + output_offset * 3 * NUM_BINS;
        // skip reading the data already in local memory
        unsigned int skip_id = group_id - output_offset;
        // locate output histogram location for this feature4
        acc_type *__restrict__ hist_buf = hist_buf_base + feature_id * 2 * NUM_BINS;

        within_kernel_reduction16x4(feature_subhists, skip_id, old_val, 1 << power_feature_workgroups, hist_buf, reinterpret_cast<acc_type *>(shared_array), power_feature_workgroups);
    }
}

// end of histogram16 stuff

// histogram64 stuff
#undef KERNEL_NAME
#undef NUM_BINS
#undef LOCAL_MEM_SIZE
#ifdef ENABLE_ALL_FEATURES
#ifdef IGNORE_INDICES
#define KERNEL_NAME histogram64_fulldata
#else  // IGNORE_INDICES
#define KERNEL_NAME histogram64  // seems like ENABLE_ALL_FEATURES is set to 1 in the header if its disabled
// #define KERNEL_NAME histogram64_allfeats
#endif  // IGNORE_INDICES
#else  // ENABLE_ALL_FEATURES
#error "ENABLE_ALL_FEATURES should always be 1"
#define KERNEL_NAME histogram64
#endif  // ENABLE_ALL_FEATURES
#define NUM_BINS 64
#define LOCAL_MEM_SIZE ((sizeof(unsigned int) + 2 * sizeof(acc_type)) * NUM_BINS)

// this function will be called by histogram64
// we have one sub-histogram of one feature in local memory, and need to read others
inline void __device__ within_kernel_reduction64x4(const acc_type* __restrict__ feature_sub_hist,
                           const unsigned int skip_id,
                           const unsigned int old_val_cont_bin0,
                           const uint16_t num_sub_hist,
                           acc_type* __restrict__ output_buf,
                           acc_type* __restrict__ local_hist,
                           const size_t power_feature_workgroups) {
    const uint16_t ltid = threadIdx.x;
    acc_type grad_bin = local_hist[ltid * 2];
    acc_type hess_bin = local_hist[ltid * 2 + 1];
    unsigned int* __restrict__ local_cnt = reinterpret_cast<unsigned int *>(local_hist + 2 * NUM_BINS);

    unsigned int cont_bin;
    if (power_feature_workgroups != 0) {
      cont_bin = ltid ? local_cnt[ltid] : old_val_cont_bin0;
    } else {
      cont_bin = local_cnt[ltid];
    }
    uint16_t i;

    if (power_feature_workgroups != 0) {
        // add all sub-histograms for feature
        const acc_type* __restrict__ p = feature_sub_hist + ltid;
        for (i = 0; i < skip_id; ++i) {
            grad_bin += *p;          p += NUM_BINS;
            hess_bin += *p;          p += NUM_BINS;
            cont_bin += as_acc_int_type(*p); p += NUM_BINS;
        }

        // skip the counters we already have
        p += 3 * NUM_BINS;

        for (i = i + 1; i < num_sub_hist; ++i) {
            grad_bin += *p;          p += NUM_BINS;
            hess_bin += *p;          p += NUM_BINS;
            cont_bin += as_acc_int_type(*p); p += NUM_BINS;
        }
    }
    __syncthreads();

    output_buf[ltid * 2 + 0] = grad_bin;
    output_buf[ltid * 2 + 1] = hess_bin;
}

#if USE_CONSTANT_BUF == 1
__kernel void KERNEL_NAME(__global const uchar* restrict feature_data_base,
                      __constant const uchar* restrict feature_masks __attribute__((max_constant_size(65536))),
                      const data_size_t feature_size,
                      __constant const data_size_t* restrict data_indices __attribute__((max_constant_size(65536))),
                      const data_size_t num_data,
                      __constant const score_t* restrict ordered_gradients __attribute__((max_constant_size(65536))),
#if CONST_HESSIAN == 0
                      __constant const score_t* restrict ordered_hessians __attribute__((max_constant_size(65536))),
#else
                      const score_t const_hessian,
#endif
                      char* __restrict__ output_buf,
                      volatile int * sync_counters,
                      acc_type* __restrict__ hist_buf_base,
                      const size_t power_feature_workgroups) {
#else
__global__ void KERNEL_NAME(const uchar* feature_data_base,
                      const uchar* __restrict__ feature_masks,
                      const data_size_t feature_size,
                      const data_size_t* data_indices,
                      const data_size_t num_data,
                      const score_t*  ordered_gradients,
#if CONST_HESSIAN == 0
                      const score_t*  ordered_hessians,
#else
                      const score_t const_hessian,
#endif
                      char* __restrict__ output_buf,
                      volatile int * sync_counters,
                      acc_type* __restrict__ hist_buf_base,
                      const size_t power_feature_workgroups) {
#endif
    // allocate the local memory array aligned with float2, to guarantee correct alignment on NVIDIA platforms
    // otherwise a "Misaligned Address" exception may occur
    __shared__ float2 shared_array[LOCAL_MEM_SIZE/sizeof(float2)];
    const unsigned int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint16_t ltid = threadIdx.x;
    const uint16_t lsize = NUM_BINS;  // get_local_size(0);
    const uint16_t group_id = blockIdx.x;

    // local memory per workgroup is 3 KB
    // clear local memory
    unsigned int *ptr = reinterpret_cast<unsigned int *>(shared_array);
    for (int i = ltid; i < LOCAL_MEM_SIZE/sizeof(unsigned int); i += lsize) {
        ptr[i] = 0;
    }
    __syncthreads();
    // gradient/hessian histograms
    // assume this starts at 32 * 4 = 128-byte boundary // What does it mean? boundary??
    // total size: 2 * 256 * size_of(float) = 2 KB
    // organization: each feature/grad/hessian is at a different bank,
    //               as independent of the feature value as possible
    acc_type *gh_hist = reinterpret_cast<acc_type *>(shared_array);

    // counter histogram
    // total size: 256 * size_of(unsigned int) = 1 KB
    unsigned int *cnt_hist = reinterpret_cast<unsigned int *>(gh_hist + 2 * NUM_BINS);

    // odd threads (1, 3, ...) compute histograms for Hessians first
    // even thread (0, 2, ...) compute histograms for gradients first
    // etc.
    uchar is_hessian_first = ltid & 1;

    uint16_t feature_id = group_id >> power_feature_workgroups;

    // each 2^POWER_FEATURE_WORKGROUPS workgroups process on one feature (compile-time constant)
    // feature_size is the number of examples per feature
    const uchar *feature_data = feature_data_base + feature_id * feature_size;

    // size of threads that process this feature4
    const unsigned int subglobal_size = lsize * (1 << power_feature_workgroups);

    // equivalent thread ID in this subgroup for this feature4
    const unsigned int subglobal_tid  = gtid - feature_id * subglobal_size;

    data_size_t ind;
    data_size_t ind_next;
    #ifdef IGNORE_INDICES
    ind = subglobal_tid;
    #else
    ind = data_indices[subglobal_tid];
    #endif

    // extract feature mask, when a byte is set to 0, that feature is disabled
    uchar feature_mask = feature_masks[feature_id];
    // exit if the feature is masked
    if (!feature_mask) {
        return;
    } else {
        feature_mask = feature_mask - 1;  // feature_mask is used for get feature (1: 4bit feature, 0: 8bit feature)
    }

    // STAGE 1: read feature data, and gradient and hessian
    // first half of the threads read feature data from global memory
    // We will prefetch data into the "next" variable at the beginning of each iteration
    uchar feature;
    uchar feature_next;
    uint16_t bin;

    feature = feature_data[ind >> feature_mask];
    if (feature_mask) {
       feature = (feature >> ((ind & 1) << 2)) & 0xf;
    }
    bin = feature;
    acc_type grad_bin = 0.0f, hess_bin = 0.0f;
    acc_type *addr_bin;

    // store gradient and hessian
    score_t grad, hess;
    score_t grad_next, hess_next;
    grad = ordered_gradients[ind];
    #if CONST_HESSIAN == 0
    hess = ordered_hessians[ind];
    #endif

    // there are 2^POWER_FEATURE_WORKGROUPS workgroups processing each feature4
    for (unsigned int i = subglobal_tid; i < num_data; i += subglobal_size) {
        // prefetch the next iteration variables
        // we don't need bondary check because we have made the buffer large
        int i_next = i + subglobal_size;
        #ifdef IGNORE_INDICES
        // we need to check to bounds here
        ind_next = i_next < num_data ? i_next : i;
        #else
        ind_next = data_indices[i_next];
        #endif

        grad_next = ordered_gradients[ind_next];
        #if CONST_HESSIAN == 0
        hess_next = ordered_hessians[ind_next];
        #endif

        // STAGE 2: accumulate gradient and hessian
        if (bin != feature) {
            addr_bin = gh_hist + bin * 2 + is_hessian_first;
            #if CONST_HESSIAN == 0
            acc_type acc_bin = is_hessian_first ? hess_bin : grad_bin;
            atomic_local_add_f(addr_bin, acc_bin);

            addr_bin = addr_bin + 1 - 2 * is_hessian_first;
            acc_bin = is_hessian_first ? grad_bin : hess_bin;
            atomic_local_add_f(addr_bin, acc_bin);

            #elif CONST_HESSIAN == 1
            atomic_local_add_f(addr_bin, grad_bin);
            #endif

            bin = feature;
            grad_bin = grad;
            hess_bin = hess;
        } else {
            grad_bin += grad;
            hess_bin += hess;
        }

        // prefetch the next iteration variables
        feature_next = feature_data[ind_next >> feature_mask];

        // STAGE 3: accumulate counter
        atomicAdd(cnt_hist + feature, 1);

        // STAGE 4: update next stat
        grad = grad_next;
        hess = hess_next;
        if (!feature_mask) {
            feature = feature_next;
        } else {
            feature = (feature_next >> ((ind_next & 1) << 2)) & 0xf;
        }
    }

    addr_bin = gh_hist + bin * 2 + is_hessian_first;
    #if CONST_HESSIAN == 0
    acc_type acc_bin = is_hessian_first ? hess_bin : grad_bin;
    atomic_local_add_f(addr_bin, acc_bin);

    addr_bin = addr_bin + 1 - 2 * is_hessian_first;
    acc_bin = is_hessian_first ? grad_bin : hess_bin;
    atomic_local_add_f(addr_bin, acc_bin);

    #elif CONST_HESSIAN == 1
    atomic_local_add_f(addr_bin, grad_bin);
    #endif
    __syncthreads();

    #if CONST_HESSIAN == 1
    // make a final reduction
    gh_hist[ltid * 2] += gh_hist[ltid * 2 + 1];
    gh_hist[ltid * 2 + 1] = const_hessian * cnt_hist[ltid];  // counter move to this position
    __syncthreads();
    #endif

#if POWER_FEATURE_WORKGROUPS != 0
    acc_type *__restrict__ output = reinterpret_cast<acc_type *>(output_buf) + group_id * 3 * NUM_BINS;
    // write gradients and Hessians
    acc_type *__restrict__ ptr_f = output;
    for (uint16_t i = ltid; i < 2 * NUM_BINS; i += lsize) {
        // even threads read gradients, odd threads read Hessians
        acc_type value = gh_hist[i];
        ptr_f[(i & 1) * NUM_BINS + (i >> 1)] = value;
    }
    // write counts
    acc_int_type *__restrict__ ptr_i = reinterpret_cast<acc_int_type *>(output + 2 * NUM_BINS);
    for (uint16_t i = ltid; i < NUM_BINS; i += lsize) {
        unsigned int value = cnt_hist[i];
        ptr_i[i] = value;
    }
    __syncthreads();
    __threadfence();
    unsigned int * counter_val = cnt_hist;
    // backup the old value
    unsigned int old_val = *counter_val;
    if (ltid == 0) {
        // all workgroups processing the same feature add this counter
        *counter_val = atomicAdd(const_cast<int*>(sync_counters + feature_id), 1);
    }
    // make sure everyone in this workgroup is here
    __syncthreads();
    // everyone in this workgroup: if we are the last workgroup, then do reduction!
    if (*counter_val == (1 << power_feature_workgroups) - 1) {
        if (ltid == 0) {
            sync_counters[feature_id] = 0;
        }
#else
    }
    // only 1 work group, no need to increase counter
    // the reduction will become a simple copy
    {
        unsigned int old_val;  // dummy
#endif
        // locate our feature's block in output memory
        unsigned int output_offset = (feature_id << power_feature_workgroups);
        acc_type const * __restrict__ feature_subhists =
                 reinterpret_cast<acc_type *>(output_buf) + output_offset * 3 * NUM_BINS;
        // skip reading the data already in local memory
        unsigned int skip_id = group_id - output_offset;
        // locate output histogram location for this feature4
        acc_type *__restrict__ hist_buf = hist_buf_base + feature_id * 2 * NUM_BINS;

        within_kernel_reduction64x4(feature_subhists, skip_id, old_val, 1 << power_feature_workgroups, hist_buf, reinterpret_cast<acc_type *>(shared_array), power_feature_workgroups);
    }
}

// end of histogram64 stuff

// histogram256 stuff
#undef KERNEL_NAME
#undef NUM_BINS
#undef LOCAL_MEM_SIZE
#ifdef ENABLE_ALL_FEATURES
#ifdef IGNORE_INDICES
#define KERNEL_NAME histogram256_fulldata
#else  // IGNORE_INDICES
#define KERNEL_NAME histogram256  // seems like ENABLE_ALL_FEATURES is set to 1 in the header if its disabled
// #define KERNEL_NAME histogram256_allfeats
#endif  // IGNORE_INDICES
#else  // ENABLE_ALL_FEATURES
#error "ENABLE_ALL_FEATURES should always be 1"
#define KERNEL_NAME histogram256
#endif  // ENABLE_ALL_FEATURES
#define NUM_BINS 256
#define LOCAL_MEM_SIZE ((sizeof(unsigned int) + 2 * sizeof(acc_type)) * NUM_BINS)

// this function will be called by histogram256
// we have one sub-histogram of one feature in local memory, and need to read others
inline void __device__ within_kernel_reduction256x4(const acc_type* __restrict__ feature_sub_hist,
                           const unsigned int skip_id,
                           const unsigned int old_val_cont_bin0,
                           const uint16_t num_sub_hist,
                           acc_type* __restrict__ output_buf,
                           acc_type* __restrict__ local_hist,
                           const size_t power_feature_workgroups) {
    const uint16_t ltid = threadIdx.x;
    acc_type grad_bin = local_hist[ltid * 2];
    acc_type hess_bin = local_hist[ltid * 2 + 1];
    unsigned int* __restrict__ local_cnt = reinterpret_cast<unsigned int *>(local_hist + 2 * NUM_BINS);

    unsigned int cont_bin;
    if (power_feature_workgroups != 0) {
      cont_bin = ltid ? local_cnt[ltid] : old_val_cont_bin0;
    } else {
      cont_bin = local_cnt[ltid];
    }
    uint16_t i;

    if (power_feature_workgroups != 0) {
        // add all sub-histograms for feature
        const acc_type* __restrict__ p = feature_sub_hist + ltid;
        for (i = 0; i < skip_id; ++i) {
            grad_bin += *p;          p += NUM_BINS;
            hess_bin += *p;          p += NUM_BINS;
            cont_bin += as_acc_int_type(*p); p += NUM_BINS;
        }

        // skip the counters we already have
        p += 3 * NUM_BINS;

        for (i = i + 1; i < num_sub_hist; ++i) {
            grad_bin += *p;          p += NUM_BINS;
            hess_bin += *p;          p += NUM_BINS;
            cont_bin += as_acc_int_type(*p); p += NUM_BINS;
        }
    }

    __syncthreads();

    output_buf[ltid * 2 + 0] = grad_bin;
    output_buf[ltid * 2 + 1] = hess_bin;
}

#if USE_CONSTANT_BUF == 1
__kernel void KERNEL_NAME(__global const uchar* restrict feature_data_base,
                      __constant const uchar* restrict feature_masks __attribute__((max_constant_size(65536))),
                      const data_size_t feature_size,
                      __constant const data_size_t* restrict data_indices __attribute__((max_constant_size(65536))),
                      const data_size_t num_data,
                      __constant const score_t* restrict ordered_gradients __attribute__((max_constant_size(65536))),
#if CONST_HESSIAN == 0
                      __constant const score_t* restrict ordered_hessians __attribute__((max_constant_size(65536))),
#else
                      const score_t const_hessian,
#endif
                      char* __restrict__ output_buf,
                      volatile int * sync_counters,
                      acc_type* __restrict__ hist_buf_base,
                      const size_t power_feature_workgroups) {
#else
__global__ void KERNEL_NAME(const uchar* feature_data_base,
                      const uchar* __restrict__ feature_masks,
                      const data_size_t feature_size,
                      const data_size_t* data_indices,
                      const data_size_t num_data,
                      const score_t*  ordered_gradients,
#if CONST_HESSIAN == 0
                      const score_t*  ordered_hessians,
#else
                      const score_t const_hessian,
#endif
                      char* __restrict__ output_buf,
                      volatile int * sync_counters,
                      acc_type* __restrict__ hist_buf_base,
                      const size_t power_feature_workgroups) {
#endif
    // allocate the local memory array aligned with float2, to guarantee correct alignment on NVIDIA platforms
    // otherwise a "Misaligned Address" exception may occur
    __shared__ float2 shared_array[LOCAL_MEM_SIZE/sizeof(float2)];
    const unsigned int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint16_t ltid = threadIdx.x;
    const uint16_t lsize = NUM_BINS;  // get_local_size(0);
    const uint16_t group_id = blockIdx.x;

    // local memory per workgroup is 3 KB
    // clear local memory
    unsigned int *ptr = reinterpret_cast<unsigned int *>(shared_array);
    for (int i = ltid; i < LOCAL_MEM_SIZE/sizeof(unsigned int); i += lsize) {
        ptr[i] = 0;
    }
    __syncthreads();
    // gradient/hessian histograms
    // assume this starts at 32 * 4 = 128-byte boundary // What does it mean? boundary??
    // total size: 2 * 256 * size_of(float) = 2 KB
    // organization: each feature/grad/hessian is at a different bank,
    //               as indepedent of the feature value as possible
    acc_type *gh_hist = reinterpret_cast<acc_type *>(shared_array);

    // counter histogram
    // total size: 256 * size_of(unsigned int) = 1 KB
    unsigned int *cnt_hist = reinterpret_cast<unsigned int *>(gh_hist + 2 * NUM_BINS);

    // odd threads (1, 3, ...) compute histograms for hessians first
    // even thread (0, 2, ...) compute histograms for gradients first
    // etc.
    uchar is_hessian_first = ltid & 1;

    uint16_t feature_id = group_id >> power_feature_workgroups;

    // each 2^POWER_FEATURE_WORKGROUPS workgroups process on one feature (compile-time constant)
    // feature_size is the number of examples per feature
    const uchar *feature_data = feature_data_base + feature_id * feature_size;

    // size of threads that process this feature4
    const unsigned int subglobal_size = lsize * (1 << power_feature_workgroups);

    // equivalent thread ID in this subgroup for this feature4
    const unsigned int subglobal_tid  = gtid - feature_id * subglobal_size;

    data_size_t ind;
    data_size_t ind_next;
    #ifdef IGNORE_INDICES
    ind = subglobal_tid;
    #else
    ind = data_indices[subglobal_tid];
    #endif

    // extract feature mask, when a byte is set to 0, that feature is disabled
    uchar feature_mask = feature_masks[feature_id];
    // exit if the feature is masked
    if (!feature_mask) {
        return;
    } else {
        feature_mask = feature_mask - 1;  // feature_mask is used for get feature (1: 4bit feature, 0: 8bit feature)
    }

    // STAGE 1: read feature data, and gradient and hessian
    // first half of the threads read feature data from global memory
    // We will prefetch data into the "next" variable at the beginning of each iteration
    uchar feature;
    uchar feature_next;
    uint16_t bin;

    feature = feature_data[ind >> feature_mask];
    if (feature_mask) {
       feature = (feature >> ((ind & 1) << 2)) & 0xf;
    }
    bin = feature;
    acc_type grad_bin = 0.0f, hess_bin = 0.0f;
    acc_type *addr_bin;

    // store gradient and hessian
    score_t grad, hess;
    score_t grad_next, hess_next;
    grad = ordered_gradients[ind];
    #if CONST_HESSIAN == 0
    hess = ordered_hessians[ind];
    #endif

    // there are 2^POWER_FEATURE_WORKGROUPS workgroups processing each feature4
    for (unsigned int i = subglobal_tid; i < num_data; i += subglobal_size) {
        // prefetch the next iteration variables
        // we don't need boundary check because we have made the buffer large
        int i_next = i + subglobal_size;
        #ifdef IGNORE_INDICES
        // we need to check to bounds here
        ind_next = i_next < num_data ? i_next : i;
        #else
        ind_next = data_indices[i_next];
        #endif

        grad_next = ordered_gradients[ind_next];
        #if CONST_HESSIAN == 0
        hess_next = ordered_hessians[ind_next];
        #endif
        // STAGE 2: accumulate gradient and hessian
        if (bin != feature) {
            addr_bin = gh_hist + bin * 2 + is_hessian_first;
            #if CONST_HESSIAN == 0
            acc_type acc_bin = is_hessian_first ? hess_bin : grad_bin;
            atomic_local_add_f(addr_bin, acc_bin);

            addr_bin = addr_bin + 1 - 2 * is_hessian_first;
            acc_bin = is_hessian_first ? grad_bin : hess_bin;
            atomic_local_add_f(addr_bin, acc_bin);

            #elif CONST_HESSIAN == 1
            atomic_local_add_f(addr_bin, grad_bin);
            #endif

            bin = feature;
            grad_bin = grad;
            hess_bin = hess;
        } else {
            grad_bin += grad;
            hess_bin += hess;
        }

        // prefetch the next iteration variables
        feature_next = feature_data[ind_next >> feature_mask];

        // STAGE 3: accumulate counter
        atomicAdd(cnt_hist + feature, 1);

        // STAGE 4: update next stat
        grad = grad_next;
        hess = hess_next;
        if (!feature_mask) {
            feature = feature_next;
        } else {
            feature = (feature_next >> ((ind_next & 1) << 2)) & 0xf;
        }
    }

    addr_bin = gh_hist + bin * 2 + is_hessian_first;
    #if CONST_HESSIAN == 0
    acc_type acc_bin = is_hessian_first ? hess_bin : grad_bin;
    atomic_local_add_f(addr_bin, acc_bin);

    addr_bin = addr_bin + 1 - 2 * is_hessian_first;
    acc_bin = is_hessian_first ? grad_bin : hess_bin;

    atomic_local_add_f(addr_bin, acc_bin);

    #elif CONST_HESSIAN == 1
    atomic_local_add_f(addr_bin, grad_bin);
    #endif
     __syncthreads();

    #if CONST_HESSIAN == 1
    // make a final reduction
    gh_hist[ltid * 2] += gh_hist[ltid * 2 + 1];
    gh_hist[ltid * 2 + 1] = const_hessian * cnt_hist[ltid];  // counter move to this position
    __syncthreads();
    #endif

#if POWER_FEATURE_WORKGROUPS != 0
    acc_type *__restrict__ output = reinterpret_cast<acc_type *>(output_buf) + group_id * 3 * NUM_BINS;
    // write gradients and Hessians
    acc_type *__restrict__ ptr_f = output;
    for (uint16_t i = ltid; i < 2 * NUM_BINS; i += lsize) {
        // even threads read gradients, odd threads read Hessians
        acc_type value = gh_hist[i];
        ptr_f[(i & 1) * NUM_BINS + (i >> 1)] = value;
    }
    // write counts
    acc_int_type *__restrict__ ptr_i = reinterpret_cast<acc_int_type *>(output + 2 * NUM_BINS);
    for (uint16_t i = ltid; i < NUM_BINS; i += lsize) {
        unsigned int value = cnt_hist[i];
        ptr_i[i] = value;
    }
    __syncthreads();
    __threadfence();
    unsigned int * counter_val = cnt_hist;
    // backup the old value
    unsigned int old_val = *counter_val;
    if (ltid == 0) {
        // all workgroups processing the same feature add this counter
        *counter_val = atomicAdd(const_cast<int*>(sync_counters + feature_id), 1);
    }
    // make sure everyone in this workgroup is here
    __syncthreads();
    // everyone in this workgroup: if we are the last workgroup, then do reduction!
    if (*counter_val == (1 << power_feature_workgroups) - 1) {
        if (ltid == 0) {
            sync_counters[feature_id] = 0;
        }
#else
    }
    // only 1 work group, no need to increase counter
    // the reduction will become a simple copy
    {
        unsigned int old_val;  // dummy
#endif
        // locate our feature's block in output memory
        unsigned int output_offset = (feature_id << power_feature_workgroups);
        acc_type const * __restrict__ feature_subhists =
                 reinterpret_cast<acc_type *>(output_buf) + output_offset * 3 * NUM_BINS;
        // skip reading the data already in local memory
        unsigned int skip_id = group_id - output_offset;
        // locate output histogram location for this feature4
        acc_type *__restrict__ hist_buf = hist_buf_base + feature_id * 2 * NUM_BINS;

        within_kernel_reduction256x4(feature_subhists, skip_id, old_val, 1 << power_feature_workgroups, hist_buf, reinterpret_cast<acc_type *>(shared_array), power_feature_workgroups);
    }
}

// end of histogram256 stuff

}  // namespace LightGBM
