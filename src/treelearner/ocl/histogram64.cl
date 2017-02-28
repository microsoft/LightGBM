// this file can either be read and passed to an OpenCL compiler directly,
// or included in a C++11 source file as a string literal
#ifndef __OPENCL_VERSION__
// If we are including this file in C++,
// the entire source file following (except the last #endif) will become
// a raw string literal. The extra ")" is just for mathcing parentheses
// to make the editor happy. The extra ")" and extra endif will be skipped.
// DO NOT add anything between here and the next #ifdef, otherwise you need
// to modify the skip count at the end of this file.
R""()
#endif

#ifndef _HISTOGRAM_64_KERNEL_
#define _HISTOGRAM_64_KERNEL_

#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable

#define LOCAL_SIZE_0 256
#define NUM_BINS 64
// NUM_BANKS should be a power of 2
#define NUM_BANKS 4
// how many bits in thread ID represent the bank = log2(NUM_BANKS)
#define BANK_BITS 2
// mask for getting the bank ID
#define BANK_MASK (NUM_BANKS - 1)
#define HG_BIN_MULT (NUM_BANKS * 4 * 2)
#define CNT_BIN_MULT (NUM_BANKS * 4)
#define LOCAL_MEM_SIZE (4 * 3 * NUM_BINS * NUM_BANKS)

// unroll the atomic operation for a few times. Takes more code space, 
// but compiler can generate better code for faster atomics.
#define UNROLL_ATOMIC 1

// Options passed by compiler at run time:
// IGNORE_INDICES will be set when the kernel does not 
// #define IGNORE_INDICES
// #define FEATURE_SIZE (32768 * 1024)
// #define POWER_FEATURE_WORKGROUPS 10

// detect Nvidia platforms
#ifdef cl_nv_pragma_unroll
#define NVIDIA 1
#endif

typedef uint data_size_t;
typedef float score_t;

struct HistogramBinEntry {
    score_t sum_gradients;
    score_t sum_hessians;
    uint  cnt;
};


#define ATOMIC_FADD_SUB1 { \
    expected.f32 = current.f32; \
    next.f32 = expected.f32 + val; \
    current.u32 = atom_cmpxchg((volatile __local uint *)addr, expected.u32, next.u32); \
    if (current.u32 == expected.u32) \
        goto end; \
    }
#define ATOMIC_FADD_SUB2  ATOMIC_FADD_SUB1 \
                          ATOMIC_FADD_SUB1
#define ATOMIC_FADD_SUB4  ATOMIC_FADD_SUB2 \
                          ATOMIC_FADD_SUB2
#define ATOMIC_FADD_SUB8  ATOMIC_FADD_SUB4 \
                          ATOMIC_FADD_SUB4
#define ATOMIC_FADD_SUB16 ATOMIC_FADD_SUB8 \
                          ATOMIC_FADD_SUB8
#define ATOMIC_FADD_SUB32 ATOMIC_FADD_SUB16\
                          ATOMIC_FADD_SUB16
#define ATOMIC_FADD_SUB64 ATOMIC_FADD_SUB32\
                          ATOMIC_FADD_SUB32


// atomic add for float number in local memory
inline void atomic_local_add_f(__local float *addr, const float val)
{
#if NVIDIA == 1
    float res = 0;
    asm volatile ("atom.shared.add.f32 %0, [%1], %2;" : "=f"(res) : "l"(addr), "f"(val));
#elif UNROLL_ATOMIC == 1
    union{
        uint u32;
        float f32;
    } next, expected, current;
    current.f32 = *addr;
    // provide a fast path
    // then do the complete loop
    // this should work on all devices
    ATOMIC_FADD_SUB16
    do {
        expected.f32 = current.f32;
        next.f32 = expected.f32 + val;
        current.u32 = atom_cmpxchg((volatile __local uint *)addr, expected.u32, next.u32);
    } while (current.u32 != expected.u32);
    end:
        ;
#else
    // slow version
    union{
        uint u32;
        float f32;
    } next, expected, current;
    current.f32 = *addr;
    do{
        expected.f32 = current.f32;
        next.f32 = expected.f32 + val;
        current.u32 = atom_cmpxchg((volatile __local uint *)addr, expected.u32, next.u32);
    }while(current.u32 != expected.u32);
#endif
}

// this function will be called by histogram64
// we have one sub-histogram of one feature in registers, and need to read others
void within_kernel_reduction64x4(__global const float* restrict feature4_sub_hist, 
                           const uint skip_id,
                           float g_val, float h_val, uint cnt_val,
                           const ushort num_sub_hist,
                           __global float* restrict output_buf,
                           __local float * restrict local_hist) {
    const ushort ltid = get_local_id(0); // range 0 - 255
    const ushort lsize = LOCAL_SIZE_0;
    ushort feature_id = ltid & 3; // range 0 - 4
    const ushort bin_id = ltid >> 2; // range 0 - 63W
    #if POWER_FEATURE_WORKGROUPS != 0 
    // if there is only 1 work group, no need to do the reduction
    // printf("%d-pre(skip %d): %f %f %f %f %f %f %f %f %d %d %d %d", ltid, skip_id, f0_grad_bin, f1_grad_bin, f2_grad_bin, f3_grad_bin, f0_hess_bin, f1_hess_bin, f2_hess_bin, f3_hess_bin, f0_cont_bin, f1_cont_bin, f2_cont_bin, f3_cont_bin);
    // add all sub-histograms for 4 features
    __global const float* restrict p = feature4_sub_hist + ltid;
    ushort i;
    for (i = 0; i < skip_id; ++i) {
            g_val += *p;            p += NUM_BINS * 4; // 256 threads working on 4 features' 64 bins
            h_val += *p;            p += NUM_BINS * 4;
            cnt_val += as_uint(*p); p += NUM_BINS * 4;
    }
    // skip the counters we already have
    p += 3 * 4 * NUM_BINS;
    for (i = i + 1; i < num_sub_hist; ++i) {
            g_val += *p;            p += NUM_BINS * 4;
            h_val += *p;            p += NUM_BINS * 4;
            cnt_val += as_uint(*p); p += NUM_BINS * 4;
    }
    #endif
    // printf("%d-aft: %f %f %f %f %f %f %f %f %d %d %d %d", ltid, f0_grad_bin, f1_grad_bin, f2_grad_bin, f3_grad_bin, f0_hess_bin, f1_hess_bin, f2_hess_bin, f3_hess_bin, f0_cont_bin, f1_cont_bin, f2_cont_bin, f3_cont_bin);
    // now overwrite the local_hist for final reduction and output
    // reverse the f3...f0 order to match the real order
    feature_id = 3 - feature_id;
    local_hist[feature_id * 3 * NUM_BINS + bin_id * 3 + 0] = g_val;
    local_hist[feature_id * 3 * NUM_BINS + bin_id * 3 + 1] = h_val;
    local_hist[feature_id * 3 * NUM_BINS + bin_id * 3 + 2] = as_float(cnt_val);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (ushort i = ltid; i < 4 * 3 * NUM_BINS; i += lsize) {
        output_buf[i] = local_hist[i];
    }
}

__attribute__((reqd_work_group_size(LOCAL_SIZE_0, 1, 1)))
#if USE_CONSTANT_BUF == 1
__kernel void histogram64(__global const uchar4* restrict feature_data_base, 
                      __constant const data_size_t* restrict data_indices __attribute__((max_constant_size(65536))), 
                      const data_size_t num_data, 
                      __constant const score_t* restrict ordered_gradients __attribute__((max_constant_size(65536))), 
                      __constant const score_t* restrict ordered_hessians __attribute__((max_constant_size(65536))),
                      __global char* restrict output_buf,
                      __global volatile int * sync_counters,
                      __global float* restrict hist_buf_base) {
#else
__kernel void histogram64(__global const uchar4* feature_data_base, 
                      __global const data_size_t* data_indices, 
                      const data_size_t num_data, 
                      __global const score_t*  ordered_gradients, 
                      __global const score_t*  ordered_hessians,
                      __global char* restrict output_buf, 
                      __global volatile int * sync_counters,
                      __global float* restrict hist_buf_base) {
#endif
    __local float shared_array[LOCAL_MEM_SIZE];
    const uint gtid = get_global_id(0);
    const uint gsize = get_global_size(0);
    const ushort ltid = get_local_id(0);
    const ushort lsize = LOCAL_SIZE_0; // get_local_size(0);
    const ushort group_id = get_group_id(0);

    // local memory per workgroup is 12 KB
    // clear local memory
    __local uint * ptr = (__local uint *) shared_array;
    for (int i = ltid; i < LOCAL_MEM_SIZE; i += lsize) {
        ptr[i] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // gradient/hessian histograms
    // assume this starts at 32 * 4 = 128-byte boundary
    // each bank: 2 * 4 * 64 * size_of(float) = 2 KB
    // there are 4 banks (sub-histograms) used by 256 threads total 8 KB
    /* memory layout of gh_hist:
       -----------------------------------------------------------------------------------------------
       bk0_g_f0_bin0   bk0_g_f1_bin0   bk0_g_f2_bin0   bk0_g_f3_bin0   bk0_h_f0_bin0   bk0_h_f1_bin0   bk0_h_f2_bin0   bk0_h_f3_bin0
       bk1_g_f0_bin0   bk1_g_f1_bin0   bk1_g_f2_bin0   bk1_g_f3_bin0   bk1_h_f0_bin0   bk1_h_f1_bin0   bk1_h_f2_bin0   bk1_h_f3_bin0
       bk2_g_f0_bin0   bk2_g_f1_bin0   bk2_g_f2_bin0   bk2_g_f3_bin0   bk2_h_f0_bin0   bk2_h_f1_bin0   bk2_h_f2_bin0   bk2_h_f3_bin0
       bk3_g_f0_bin0   bk3_g_f1_bin0   bk3_g_f2_bin0   bk3_g_f3_bin0   bk3_h_f0_bin0   bk3_h_f1_bin0   bk3_h_f2_bin0   bk3_h_f3_bin0
       bk0_g_f0_bin1   bk0_g_f1_bin1   bk0_g_f2_bin1   bk0_g_f3_bin1   bk0_h_f0_bin1   bk0_h_f1_bin1   bk0_h_f2_bin1   bk0_h_f3_bin1
       bk1_g_f0_bin1   bk1_g_f1_bin1   bk1_g_f2_bin1   bk1_g_f3_bin1   bk1_h_f0_bin1   bk1_h_f1_bin1   bk1_h_f2_bin1   bk1_h_f3_bin1
       bk2_g_f0_bin1   bk2_g_f1_bin1   bk2_g_f2_bin1   bk2_g_f3_bin1   bk2_h_f0_bin1   bk2_h_f1_bin1   bk2_h_f2_bin1   bk2_h_f3_bin1
       bk3_g_f0_bin1   bk3_g_f1_bin1   bk3_g_f2_bin1   bk3_g_f3_bin1   bk3_h_f0_bin1   bk3_h_f1_bin1   bk3_h_f2_bin1   bk3_h_f3_bin1
       ...
       bk0_g_f0_bin255 bk0_g_f1_bin255 bk0_g_f2_bin255 bk0_g_f3_bin255 bk0_h_f0_bin255 bk0_h_f1_bin255 bk0_h_f2_bin255 bk0_h_f3_bin255
       bk1_g_f0_bin255 bk1_g_f1_bin255 bk1_g_f2_bin255 bk1_g_f3_bin255 bk1_h_f0_bin255 bk1_h_f1_bin255 bk1_h_f2_bin255 bk1_h_f3_bin255
       bk2_g_f0_bin255 bk2_g_f1_bin255 bk2_g_f2_bin255 bk2_g_f3_bin255 bk2_h_f0_bin255 bk2_h_f1_bin255 bk2_h_f2_bin255 bk2_h_f3_bin255
       bk3_g_f0_bin255 bk3_g_f1_bin255 bk3_g_f2_bin255 bk3_g_f3_bin255 bk3_h_f0_bin255 bk3_h_f1_bin255 bk3_h_f2_bin255 bk3_h_f3_bin255
       -----------------------------------------------------------------------------------------------
    */
    // with this organization, the LDS/shared memory bank is independent of the bin value
    // all threads within a quarter-wavefront (half-warp) will not have any bank conflict

    __local float * gh_hist = (__local float *)shared_array;
    // counter histogram
    // each bank: 4 * 64 * size_of(uint) = 1 KB
    // there are 4 banks used by 256 threads total 4 KB
    /* memory layout in cnt_hist:
       -----------------------------------------------
       bk0_c_f0_bin0   bk0_c_f1_bin0   bk0_c_f2_bin0   bk0_c_f3_bin0
       bk1_c_f0_bin0   bk1_c_f1_bin0   bk1_c_f2_bin0   bk1_c_f3_bin0
       bk2_c_f0_bin0   bk2_c_f1_bin0   bk2_c_f2_bin0   bk2_c_f3_bin0
       bk3_c_f0_bin0   bk3_c_f1_bin0   bk3_c_f2_bin0   bk3_c_f3_bin0
       bk0_c_f0_bin1   bk0_c_f1_bin1   bk0_c_f2_bin1   bk0_c_f3_bin1
       bk1_c_f0_bin1   bk1_c_f1_bin1   bk1_c_f2_bin1   bk1_c_f3_bin1
       bk2_c_f0_bin1   bk2_c_f1_bin1   bk2_c_f2_bin1   bk2_c_f3_bin1
       bk3_c_f0_bin1   bk3_c_f1_bin1   bk3_c_f2_bin1   bk3_c_f3_bin1
       ...
       bk0_c_f0_bin255 bk0_c_f1_bin255 bk0_c_f2_bin255 bk0_c_f3_bin255
       bk1_c_f0_bin255 bk1_c_f1_bin255 bk1_c_f2_bin255 bk1_c_f3_bin255
       bk2_c_f0_bin255 bk2_c_f1_bin255 bk2_c_f2_bin255 bk2_c_f3_bin255
       bk3_c_f0_bin255 bk3_c_f1_bin255 bk3_c_f2_bin255 bk3_c_f3_bin255
       -----------------------------------------------
    */
    __local uint * cnt_hist = (__local uint *)(gh_hist + 2 * 4 * NUM_BINS * NUM_BANKS);

    // thread 0, 1, 2, 3 compute histograms for gradients first
    // thread 4, 5, 6, 7 compute histograms for hessians  first
    // etc.
    uchar is_hessian_first = (ltid >> 2) & 1;
    // thread 0-7 write result to bank0, 8-15 to bank1, 16-23 to bank2, 24-31 to bank3
    ushort bank = (ltid >> 3) & BANK_MASK;
    
    // each 2^POWER_FEATURE_WORKGROUPS workgroups process on one feature
    // FEATURE_SIZE is the number of examples per feature
    // they are compile-time constants
    __global const uchar4* feature_data = feature_data_base + (group_id >> POWER_FEATURE_WORKGROUPS) * FEATURE_SIZE;
    // size of threads that process this feature4
    const uint subglobal_size = lsize * (1 << POWER_FEATURE_WORKGROUPS);
    // equavalent thread ID in this subgroup for this feature4
    const uint subglobal_tid  = gtid - (group_id >> POWER_FEATURE_WORKGROUPS) * subglobal_size;

    // STAGE 1: read feature data, and gradient and hessian
    // first half of the threads read feature data from global memory
    // 4 features stored in a tuple MSB...(0, 1, 2, 3)...LSB
    // We will prefetch data into the "next" variable at the beginning of each iteration
    uchar4 feature4;
    uchar4 feature4_next;
    // store gradient and hessian
    float stat1, stat2;
    float stat1_next, stat2_next;
    data_size_t ind;
    data_size_t ind_next;
    stat1 = ordered_gradients[subglobal_tid];
    stat2 = ordered_hessians[subglobal_tid];
    #ifdef IGNORE_INDICES
    ind = subglobal_tid;
    #else
    ind = data_indices[subglobal_tid];
    #endif
    feature4 = feature_data[ind];

    // there are 2^POWER_FEATURE_WORKGROUPS workgroups processing each feature4
    for (uint i = subglobal_tid; i < num_data; i += subglobal_size) {
        // prefetch the next iteration variables
        // we don't need bondary check because we have made the buffer larger
        stat1_next = ordered_gradients[i + subglobal_size];
        stat2_next = ordered_hessians[i + subglobal_size];
        #ifdef IGNORE_INDICES
        // we need to check to bounds here
        ind_next = i + subglobal_size < num_data ? i + subglobal_size : i;
        // start load next feature as early as possible
        feature4_next = feature_data[ind_next];
        #else
        ind_next = data_indices[i + subglobal_size];
        #endif
        // offset used to rotate feature4 vector
        ushort offset = (ltid & 0x3);
        // swap gradient and hessian for threads 4, 5, 6, 7
        float tmp = stat1;
        stat1 = is_hessian_first ? stat2 : stat1;
        stat2 = is_hessian_first ? tmp   : stat2;
        // stat1 = select(stat1, stat2, is_hessian_first);
        // stat2 = select(stat2, tmp, is_hessian_first);

        // STAGE 2: accumulate gradient and hessian
        ushort bin, addr;
        feature4 = as_uchar4(rotate(as_uint(feature4), (uint)offset*8));
        // thread 0, 1, 2, 3 now process feature 0, 1, 2, 3's gradients for example 0, 1, 2, 3
        // thread 4, 5, 6, 7 now process feature 0, 1, 2, 3's hessians  for example 4, 5, 6, 7
        bin = feature4.s3 & 0x3f;
        addr = bin * HG_BIN_MULT + bank * 8 + is_hessian_first * 4 + offset;
        atomic_local_add_f(gh_hist + addr, stat1);
        // thread 0, 1, 2, 3 now process feature 0, 1, 2, 3's hessians  for example 0, 1, 2, 3
        // thread 4, 5, 6, 7 now process feature 0, 1, 2, 3's gradients for example 4, 5, 6, 7
        addr += 4 - 8 * is_hessian_first;
        atomic_local_add_f(gh_hist + addr, stat2);
        // thread 0, 1, 2, 3 now process feature 1, 2, 3, 0's gradients for example 0, 1, 2, 3
        // thread 4, 5, 6, 7 now process feature 1, 2, 3, 0's hessians  for example 4, 5, 6, 7
        bin = feature4.s2 & 0x3f;
        offset = (offset + 1) & 0x3;
        addr = bin * HG_BIN_MULT + bank * 8 + is_hessian_first * 4 + offset;
        atomic_local_add_f(gh_hist + addr, stat1);
        // thread 0, 1, 2, 3 now process feature 1, 2, 3, 0's hessians  for example 0, 1, 2, 3
        // thread 4, 5, 6, 7 now process feature 1, 2, 3, 0's gradients for example 4, 5, 6, 7
        addr += 4 - 8 * is_hessian_first;
        atomic_local_add_f(gh_hist + addr, stat2);
        // thread 0, 1, 2, 3 now process feature 2, 3, 0, 1's gradients for example 0, 1, 2, 3
        // thread 4, 5, 6, 7 now process feature 2, 3, 0, 1's hessians  for example 4, 5, 6, 7
        bin = feature4.s1 & 0x3f;
        offset = (offset + 1) & 0x3;
        addr = bin * HG_BIN_MULT + bank * 8 + is_hessian_first * 4 + offset;
        atomic_local_add_f(gh_hist + addr, stat1);
        // thread 0, 1, 2, 3 now process feature 2, 3, 0, 1's hessians  for example 0, 1, 2, 3
        // thread 4, 5, 6, 7 now process feature 2, 3, 0, 1's gradients for example 4, 5, 6, 7
        addr += 4 - 8 * is_hessian_first;
        atomic_local_add_f(gh_hist + addr, stat2);

        // prefetch the next iteration variables
        // we don't need bondary check because if it is out of boundary, ind_next = 0
        /*
        if (i + subglobal_size >= num_data) {
            if (ind_next)
                printf("%d:%d outof bound index: %d\n", gtid, group_id, ind_next);
        }
        else 
        */
        #ifndef IGNORE_INDICES
        feature4_next = feature_data[ind_next];
        #endif

        // thread 0, 1, 2, 3 now process feature 3, 0, 1, 2's gradients for example 0, 1, 2, 3
        // thread 4, 5, 6, 7 now process feature 3, 0, 1, 2's hessians  for example 4, 5, 6, 7
        bin = feature4.s0 & 0x3f;
        offset = (offset + 1) & 0x3;
        addr = bin * HG_BIN_MULT + bank * 8 + is_hessian_first * 4 + offset;
        atomic_local_add_f(gh_hist + addr, stat1);
        // thread 0, 1, 2, 3 now process feature 3, 0, 1, 2's hessians  for example 0, 1, 2, 3
        // thread 4, 5, 6, 7 now process feature 3, 0, 1, 2's gradients for example 4, 5, 6, 7
        addr += 4 - 8 * is_hessian_first;
        atomic_local_add_f(gh_hist + addr, stat2);
        // STAGE 3: accumulate counter
        // there are 4 counters for 4 features
        // thread 0, 1, 2, 3 now process feature 0, 1, 2, 3's counts for example 0, 1, 2, 3
        bin = feature4.s3 & 0x3f;
        offset = (ltid & 0x3);
        addr = bin * CNT_BIN_MULT + bank * 4 + offset;
        atom_inc(cnt_hist + addr);
        // thread 0, 1, 2, 3 now process feature 1, 2, 3, 0's counts for example 0, 1, 2, 3
        bin = feature4.s2 & 0x3f;
        offset = (offset + 1) & 0x3;
        addr = bin * CNT_BIN_MULT + bank * 4 + offset;
        atom_inc(cnt_hist + addr);
        // thread 0, 1, 2, 3 now process feature 2, 3, 0, 1's counts for example 0, 1, 2, 3
        bin = feature4.s1 & 0x3f;
        offset = (offset + 1) & 0x3;
        addr = bin * CNT_BIN_MULT + bank * 4 + offset;
        atom_inc(cnt_hist + addr);
        // thread 0, 1, 2, 3 now process feature 3, 0, 1, 2's counts for example 0, 1, 2, 3
        bin = feature4.s0 & 0x3f;
        offset = (offset + 1) & 0x3;
        addr = bin * CNT_BIN_MULT + bank * 4 + offset;
        atom_inc(cnt_hist + addr);
        stat1 = stat1_next;
        stat2 = stat2_next;
        feature4 = feature4_next;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // now reduce the 4 banks of subhistograms into 1
    /* memory layout of gh_hist:
       -----------------------------------------------------------------------------------------------
       bk0_g_f0_bin0   bk0_g_f1_bin0   bk0_g_f2_bin0   bk0_g_f3_bin0   bk0_h_f0_bin0   bk0_h_f1_bin0   bk0_h_f2_bin0   bk0_h_f3_bin0
       bk1_g_f0_bin0   bk1_g_f1_bin0   bk1_g_f2_bin0   bk1_g_f3_bin0   bk1_h_f0_bin0   bk1_h_f1_bin0   bk1_h_f2_bin0   bk1_h_f3_bin0
       bk2_g_f0_bin0   bk2_g_f1_bin0   bk2_g_f2_bin0   bk2_g_f3_bin0   bk2_h_f0_bin0   bk2_h_f1_bin0   bk2_h_f2_bin0   bk2_h_f3_bin0
       bk3_g_f0_bin0   bk3_g_f1_bin0   bk3_g_f2_bin0   bk3_g_f3_bin0   bk3_h_f0_bin0   bk3_h_f1_bin0   bk3_h_f2_bin0   bk3_h_f3_bin0
       bk0_g_f0_bin1   bk0_g_f1_bin1   bk0_g_f2_bin1   bk0_g_f3_bin1   bk0_h_f0_bin1   bk0_h_f1_bin1   bk0_h_f2_bin1   bk0_h_f3_bin1
       bk1_g_f0_bin1   bk1_g_f1_bin1   bk1_g_f2_bin1   bk1_g_f3_bin1   bk1_h_f0_bin1   bk1_h_f1_bin1   bk1_h_f2_bin1   bk1_h_f3_bin1
       bk2_g_f0_bin1   bk2_g_f1_bin1   bk2_g_f2_bin1   bk2_g_f3_bin1   bk2_h_f0_bin1   bk2_h_f1_bin1   bk2_h_f2_bin1   bk2_h_f3_bin1
       bk3_g_f0_bin1   bk3_g_f1_bin1   bk3_g_f2_bin1   bk3_g_f3_bin1   bk3_h_f0_bin1   bk3_h_f1_bin1   bk3_h_f2_bin1   bk3_h_f3_bin1
       ...
       bk0_g_f0_bin255 bk0_g_f1_bin255 bk0_g_f2_bin255 bk0_g_f3_bin255 bk0_h_f0_bin255 bk0_h_f1_bin255 bk0_h_f2_bin255 bk0_h_f3_bin255
       bk1_g_f0_bin255 bk1_g_f1_bin255 bk1_g_f2_bin255 bk1_g_f3_bin255 bk1_h_f0_bin255 bk1_h_f1_bin255 bk1_h_f2_bin255 bk1_h_f3_bin255
       bk2_g_f0_bin255 bk2_g_f1_bin255 bk2_g_f2_bin255 bk2_g_f3_bin255 bk2_h_f0_bin255 bk2_h_f1_bin255 bk2_h_f2_bin255 bk2_h_f3_bin255
       bk3_g_f0_bin255 bk3_g_f1_bin255 bk3_g_f2_bin255 bk3_g_f3_bin255 bk3_h_f0_bin255 bk3_h_f1_bin255 bk3_h_f2_bin255 bk3_h_f3_bin255
       -----------------------------------------------------------------------------------------------
    */
    /* memory layout in cnt_hist:
       -----------------------------------------------
       bk0_c_f0_bin0   bk0_c_f1_bin0   bk0_c_f2_bin0   bk0_c_f3_bin0
       bk1_c_f0_bin0   bk1_c_f1_bin0   bk1_c_f2_bin0   bk1_c_f3_bin0
       bk2_c_f0_bin0   bk2_c_f1_bin0   bk2_c_f2_bin0   bk2_c_f3_bin0
       bk3_c_f0_bin0   bk3_c_f1_bin0   bk3_c_f2_bin0   bk3_c_f3_bin0
       bk0_c_f0_bin1   bk0_c_f1_bin1   bk0_c_f2_bin1   bk0_c_f3_bin1
       bk1_c_f0_bin1   bk1_c_f1_bin1   bk1_c_f2_bin1   bk1_c_f3_bin1
       bk2_c_f0_bin1   bk2_c_f1_bin1   bk2_c_f2_bin1   bk2_c_f3_bin1
       bk3_c_f0_bin1   bk3_c_f1_bin1   bk3_c_f2_bin1   bk3_c_f3_bin1
       ...
       bk0_c_f0_bin255 bk0_c_f1_bin255 bk0_c_f2_bin255 bk0_c_f3_bin255
       bk1_c_f0_bin255 bk1_c_f1_bin255 bk1_c_f2_bin255 bk1_c_f3_bin255
       bk2_c_f0_bin255 bk2_c_f1_bin255 bk2_c_f2_bin255 bk2_c_f3_bin255
       bk3_c_f0_bin255 bk3_c_f1_bin255 bk3_c_f2_bin255 bk3_c_f3_bin255
       -----------------------------------------------
    */
    float g_val = 0.0f;
    float h_val = 0.0f;
    uint cnt_val = 0;
    // 256 threads, working on 4 features and 64 bins,
    // so each thread has an independent feature/bin to work on.
    const ushort feature_id = ltid & 3; // range 0 - 4
    const ushort bin_id = ltid >> 2; // range 0 - 63
    const ushort offset = (ltid >> 2) & BANK_MASK; // helps avoid LDS bank conflicts
    for (int i = 0; i < NUM_BANKS; ++i) {
        ushort bank_id = (i + offset) & BANK_MASK;
        g_val += gh_hist[bin_id * HG_BIN_MULT + bank_id * 8 + feature_id];
        h_val += gh_hist[bin_id * HG_BIN_MULT + bank_id * 8 + feature_id + 4];
        cnt_val += cnt_hist[bin_id * CNT_BIN_MULT + bank_id * 4 + feature_id];
    }
    // now thread 0 - 3 holds feature 0, 1, 2, 3's gradient, hessian and count bin 0
    // now thread 4 - 7 holds feature 0, 1, 2, 3's gradient, hessian and count bin 1
    // etc,

    // write to output
    // write gradients and hessians histogram for all 4 features
    // output data in linear order for further reduction
    // output size = 4 (features) * 3 (counters) * 64 (bins) * sizeof(float)
    /* memory layout of output:
       g_f0_bin0   g_f1_bin0   g_f2_bin0   g_f3_bin0
       g_f0_bin1   g_f1_bin1   g_f2_bin1   g_f3_bin1
       ...
       g_f0_bin63  g_f1_bin63  g_f2_bin63  g_f3_bin63
       h_f0_bin0   h_f1_bin0   h_f2_bin0   h_f3_bin0
       h_f0_bin1   h_f1_bin1   h_f2_bin1   h_f3_bin1
       ...
       h_f0_bin63  h_f1_bin63  h_f2_bin63  h_f3_bin63
       c_f0_bin0   c_f1_bin0   c_f2_bin0   c_f3_bin0
       c_f0_bin1   c_f1_bin1   c_f2_bin1   c_f3_bin1
       ...
       c_f0_bin63  c_f1_bin63  c_f2_bin63  c_f3_bin63
    */
    // if there is only one workgroup processing this feature4, don't even need to write
    uint feature4_id = (group_id >> POWER_FEATURE_WORKGROUPS);
    #if POWER_FEATURE_WORKGROUPS != 0
    __global float * restrict output = (__global float * restrict)output_buf + group_id * 4 * 3 * NUM_BINS;
    // write gradients for 4 features
    output[0 * 4 * NUM_BINS + ltid] = g_val;
    // write hessians for 4 features
    output[1 * 4 * NUM_BINS + ltid] = h_val;
    // write counts for 4 features
    output[2 * 4 * NUM_BINS + ltid] = as_float(cnt_val);
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    mem_fence(CLK_GLOBAL_MEM_FENCE);
    // To avoid the cost of an extra reducting kernel, we have to deal with some 
    // gray area in OpenCL. We want the last work group that process this feature to
    // make the final reduction, and other threads will just quit.
    // This requires that the results written by other workgroups available to the
    // last workgroup (memory consistency)
    #if NVIDIA == 1
    // this is equavalent to CUDA __threadfence();
    // ensure the writes above goes to main memory and other workgroups can see it
    asm volatile("{\n\tmembar.gl;\n\t}\n\t" :::"memory");
    #else
    // FIXME: how to do the above on AMD GPUs??
    // GCN ISA says that the all writes will bypass L1 cache (write through),
    // however when the last thread is reading sub-histogram data we have to
    // make sure that no part of data is modified in local L1 cache of other workgroups.
    // Otherwise reading can be a problem (atomic operations to get consistency).
    // But in our case, the sub-histogram of this workgroup cannot be in the cache
    // of another workgroup, so the following trick will work just fine.
    #endif
    // Now, we want one workgroup to do the final reduction.
    // Other workgroups processing the same feature quit.
    // The is done by using an global atomic counter.
    // On AMD GPUs ideally this should be done in GDS,
    // but currently there is no easy way to access it via OpenCL.
    __local uint * counter_val = cnt_hist;
    if (ltid == 0) {
        // all workgroups processing the same feature add this counter
        *counter_val = atom_inc(sync_counters + feature4_id);
    }
    // make sure everyone in this workgroup is here
    barrier(CLK_LOCAL_MEM_FENCE);
    // everyone in this wrokgroup: if we are the last workgroup, then do reduction!
    if (*counter_val == (1 << POWER_FEATURE_WORKGROUPS) - 1) {
        if (ltid == 0) {
            // printf("workgroup %d start reduction!\n", group_id);
            // printf("feature_data[0] = %d %d %d %d", feature_data[0].s0, feature_data[0].s1, feature_data[0].s2, feature_data[0].s3);
            // clear the sync counter for using it next time
            sync_counters[feature4_id] = 0;
        }
    #else
    // only 1 work group, no need to increase counter
    // the reduction will become a simple copy
    if (1) {
        barrier(CLK_LOCAL_MEM_FENCE);
    #endif
        // locate our feature4's block in output memory
        uint output_offset = (feature4_id << POWER_FEATURE_WORKGROUPS);
        __global float const * restrict feature4_subhists = 
                 (__global float *)output_buf + output_offset * 4 * 3 * NUM_BINS;
        // skip reading the data already in local memory
        uint skip_id = group_id ^ output_offset;
        // locate output histogram location for this feature4
        __global float* restrict hist_buf = hist_buf_base + feature4_id * 4 * 3 * NUM_BINS;
        within_kernel_reduction64x4(feature4_subhists, skip_id, g_val, h_val, cnt_val, 1 << POWER_FEATURE_WORKGROUPS, hist_buf, shared_array);
    }
}

// The following line ends the string literal, adds an extra #endif at the end
// the +9 skips extra characters ")", newline, "#endif" and newline at the beginning
// )"" "\n#endif" + 9
#endif

