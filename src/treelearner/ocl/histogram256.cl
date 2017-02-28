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

#ifndef _HISTOGRAM_256_KERNEL_
#define _HISTOGRAM_256_KERNEL_

#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable


#define LOCAL_SIZE_0 256
#define NUM_BINS 256
#define LOCAL_MEM_SIZE (4 * 3 * NUM_BINS)

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

// 4n worker groups for 4n features, each one has 256 threads, to reduce the 256 bins for 3 values
/* memory layout of sub_hist_buf: 

   ------ sub-histogram block 0 ---------
   sub-histogram 0 for feature 0, 1, 2, 3            \
   sub-histogram 1 for feature 0, 1, 2, 3             |--- by work group 0, 1, 2, 3
   ...                                                |
   sub-histogram num_sub_hist for feature 0, 1, 2, 3 /                              
   ------ sub-histogram block 1 ---------
   sub-histogram 0 for feature 4, 5, 6, 7            \
   sub-histogram 1 for feature 4, 5, 6, 7             |--- by work group 4, 5, 6, 7
   ...                                                |
   sub-histogram num_sub_hist for feature 4, 5, 6, 7 /                              
   ------ sub-histogram block 2 ---------
   ...
   --------------------------------------
*/
__attribute__((reqd_work_group_size(NUM_BINS, 1, 1)))
__kernel void reduction256(__global const float* restrict sub_hist_buf, 
                           __global float* restrict output_buf,
                           const ushort num_sub_hist) {
    __local float buf[3 * NUM_BINS];
    const uint gtid = get_global_id(0);
    const ushort ltid = get_local_id(0);
    const ushort lsize = NUM_BINS;
    const ushort group_id = get_group_id(0);

    // each thread read the 256 bins
    uint counter_bin = 0;
    float grad_bin = 0;
    float hess_bin = 0;
    // each group of sub-histogram has 4 features, each feature has 3 counters, each counter has NUM_BINS bins
    // the 4 features are assigned to 4 work groups
    __global const float * restrict base = sub_hist_buf + (group_id >> 2) * (num_sub_hist * 4 * 3 * NUM_BINS) + (group_id & 0x3) * (3 * NUM_BINS);
    __global const float * restrict ptr_g = base;
    __global const float * restrict ptr_h = base + NUM_BINS;
    __global uint  * restrict ptr_c = (__global uint * restrict)ptr_h + NUM_BINS;
    // add all sub-histograms
    for (ushort i = 0; i < num_sub_hist; ++i) {
            grad_bin += ptr_g[ltid];
            hess_bin += ptr_h[ltid];
            counter_bin += ptr_c[ltid];
            // move to the next group of sub-histogram
            ptr_g += 4 * 3 * NUM_BINS;
            ptr_h += 4 * 3 * NUM_BINS;
            ptr_c += 4 * 3 * NUM_BINS;
    }
    // reorganizing data for output and final reduction
    buf[ltid * 3] = grad_bin;
    buf[ltid * 3 + 1] = hess_bin;
    buf[ltid * 3 + 2] = as_float(counter_bin);
    __global float * restrict output_base = output_buf + group_id * 3 * NUM_BINS;
    /* output layout:
       HistogramBinEntry f0[255]; -- worker group 0
       HistogramBinEntry f1[255]; -- worker group 1
       HistogramBinEntry f2[255]; -- worker group 2
       HistogramBinEntry f3[255]; -- worker group 3
       HistogramBinEntry f4[255]; -- worker group 4
       ...
    */
    barrier(CLK_LOCAL_MEM_FENCE);
    for (ushort i = ltid; i < 3 * NUM_BINS; i += lsize) {
        output_base[i] = buf[i];
    }
    // calculate the gain
    // TODO find the best split here
}

// this function will be called by histogram256
// we have one sub-histogram of one feature in local memory, and need to read others
void within_kernel_reduction256x4(__global const float* restrict feature4_sub_hist, 
                           const uint skip_id,
                           const uint old_val_f0_cont_bin0,
                           const ushort num_sub_hist,
                           __global float* restrict output_buf,
                           __local float * restrict local_hist) {
    const ushort ltid = get_local_id(0);
    const ushort lsize = LOCAL_SIZE_0;
    // initialize register counters from our local memory
    // TODO: try to avoid bank conflict here
    float f0_grad_bin = local_hist[ltid * 8];
    float f1_grad_bin = local_hist[ltid * 8 + 1];
    float f2_grad_bin = local_hist[ltid * 8 + 2];
    float f3_grad_bin = local_hist[ltid * 8 + 3];
    float f0_hess_bin = local_hist[ltid * 8 + 4];
    float f1_hess_bin = local_hist[ltid * 8 + 5];
    float f2_hess_bin = local_hist[ltid * 8 + 6];
    float f3_hess_bin = local_hist[ltid * 8 + 7];
    #if POWER_FEATURE_WORKGROUPS != 0
    uint  f0_cont_bin = ltid ? as_uint(local_hist[4 * 2 * NUM_BINS + ltid * 4]) : old_val_f0_cont_bin0;
    #else
    uint  f0_cont_bin = as_uint(local_hist[4 * 2 * NUM_BINS + ltid * 4]);
    #endif
    uint  f1_cont_bin = as_uint(local_hist[4 * 2 * NUM_BINS + ltid * 4 + 1]);
    uint  f2_cont_bin = as_uint(local_hist[4 * 2 * NUM_BINS + ltid * 4 + 2]);
    uint  f3_cont_bin = as_uint(local_hist[4 * 2 * NUM_BINS + ltid * 4 + 3]);
    // printf("%d-pre(skip %d): %f %f %f %f %f %f %f %f %d %d %d %d", ltid, skip_id, f0_grad_bin, f1_grad_bin, f2_grad_bin, f3_grad_bin, f0_hess_bin, f1_hess_bin, f2_hess_bin, f3_hess_bin, f0_cont_bin, f1_cont_bin, f2_cont_bin, f3_cont_bin);
#if POWER_FEATURE_WORKGROUPS != 0
    // add all sub-histograms for 4 features
    __global const float* restrict p = feature4_sub_hist + ltid;
    ushort i;
    for (i = 0; i < skip_id; ++i) {
            f0_grad_bin += *p;          p += NUM_BINS;
            f0_hess_bin += *p;          p += NUM_BINS;
            f0_cont_bin += as_uint(*p); p += NUM_BINS;
            f1_grad_bin += *p;          p += NUM_BINS;
            f1_hess_bin += *p;          p += NUM_BINS;
            f1_cont_bin += as_uint(*p); p += NUM_BINS;
            f2_grad_bin += *p;          p += NUM_BINS;
            f2_hess_bin += *p;          p += NUM_BINS;
            f2_cont_bin += as_uint(*p); p += NUM_BINS;
            f3_grad_bin += *p;          p += NUM_BINS;
            f3_hess_bin += *p;          p += NUM_BINS;
            f3_cont_bin += as_uint(*p); p += NUM_BINS;
    }
    // skip the counters we already have
    p += 3 * 4 * NUM_BINS;
    for (i = i + 1; i < num_sub_hist; ++i) {
            f0_grad_bin += *p;          p += NUM_BINS;
            f0_hess_bin += *p;          p += NUM_BINS;
            f0_cont_bin += as_uint(*p); p += NUM_BINS;
            f1_grad_bin += *p;          p += NUM_BINS;
            f1_hess_bin += *p;          p += NUM_BINS;
            f1_cont_bin += as_uint(*p); p += NUM_BINS;
            f2_grad_bin += *p;          p += NUM_BINS;
            f2_hess_bin += *p;          p += NUM_BINS;
            f2_cont_bin += as_uint(*p); p += NUM_BINS;
            f3_grad_bin += *p;          p += NUM_BINS;
            f3_hess_bin += *p;          p += NUM_BINS;
            f3_cont_bin += as_uint(*p); p += NUM_BINS;
    }
    // printf("%d-aft: %f %f %f %f %f %f %f %f %d %d %d %d", ltid, f0_grad_bin, f1_grad_bin, f2_grad_bin, f3_grad_bin, f0_hess_bin, f1_hess_bin, f2_hess_bin, f3_hess_bin, f0_cont_bin, f1_cont_bin, f2_cont_bin, f3_cont_bin);
    #endif
    // now overwrite the local_hist for final reduction and output
    barrier(CLK_LOCAL_MEM_FENCE);
    // reverse the f3...f0 order to match the real order
    local_hist[0 * 3 * NUM_BINS + ltid * 3 + 0] = f3_grad_bin;
    local_hist[0 * 3 * NUM_BINS + ltid * 3 + 1] = f3_hess_bin;
    local_hist[0 * 3 * NUM_BINS + ltid * 3 + 2] = as_float(f3_cont_bin);
    local_hist[1 * 3 * NUM_BINS + ltid * 3 + 0] = f2_grad_bin;
    local_hist[1 * 3 * NUM_BINS + ltid * 3 + 1] = f2_hess_bin;
    local_hist[1 * 3 * NUM_BINS + ltid * 3 + 2] = as_float(f2_cont_bin);
    local_hist[2 * 3 * NUM_BINS + ltid * 3 + 0] = f1_grad_bin;
    local_hist[2 * 3 * NUM_BINS + ltid * 3 + 1] = f1_hess_bin;
    local_hist[2 * 3 * NUM_BINS + ltid * 3 + 2] = as_float(f1_cont_bin);
    local_hist[3 * 3 * NUM_BINS + ltid * 3 + 0] = f0_grad_bin;
    local_hist[3 * 3 * NUM_BINS + ltid * 3 + 1] = f0_hess_bin;
    local_hist[3 * 3 * NUM_BINS + ltid * 3 + 2] = as_float(f0_cont_bin);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (ushort i = ltid; i < 4 * 3 * NUM_BINS; i += lsize) {
        output_buf[i] = as_float(local_hist[i]);
    }
}

__attribute__((reqd_work_group_size(LOCAL_SIZE_0, 1, 1)))
#if USE_CONSTANT_BUF == 1
__kernel void histogram256(__global const uchar4* restrict feature_data_base, 
                      __constant const data_size_t* restrict data_indices __attribute__((max_constant_size(65536))), 
                      const data_size_t num_data, 
                      __constant const score_t* restrict ordered_gradients __attribute__((max_constant_size(65536))), 
                      __constant const score_t* restrict ordered_hessians __attribute__((max_constant_size(65536))),
                      __global char* restrict output_buf,
                      __global volatile int * sync_counters,
                      __global float* restrict hist_buf_base) {
#else
__kernel void histogram256(__global const uchar4* feature_data_base, 
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
    // total size: 2 * 4 * 256 * size_of(float) = 8 KB
    // organization: each feature/grad/hessian is at a different bank, 
    //               as indepedent of the feature value as possible
    __local float * gh_hist = (__local float *)shared_array;
    // counter histogram
    // total size: 4 * 256 * size_of(uint) = 4 KB
    __local uint * cnt_hist = (__local uint *)(gh_hist + 2 * 4 * NUM_BINS);

    // thread 0, 1, 2, 3 compute histograms for gradients first
    // thread 4, 5, 6, 7 compute histograms for hessians  first
    // etc.
    uchar is_hessian_first = (ltid >> 2) & 1;
    
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
        bin = feature4.s3;
        addr = bin * 8 + is_hessian_first * 4 + offset;
        atomic_local_add_f(gh_hist + addr, stat1);
        // thread 0, 1, 2, 3 now process feature 0, 1, 2, 3's hessians  for example 0, 1, 2, 3
        // thread 4, 5, 6, 7 now process feature 0, 1, 2, 3's gradients for example 4, 5, 6, 7
        addr += 4 - 8 * is_hessian_first;
        atomic_local_add_f(gh_hist + addr, stat2);
        // thread 0, 1, 2, 3 now process feature 1, 2, 3, 0's gradients for example 0, 1, 2, 3
        // thread 4, 5, 6, 7 now process feature 1, 2, 3, 0's hessians  for example 4, 5, 6, 7
        bin = feature4.s2;
        offset = (offset + 1) & 0x3;
        addr = bin * 8 + is_hessian_first * 4 + offset;
        atomic_local_add_f(gh_hist + addr, stat1);
        // thread 0, 1, 2, 3 now process feature 1, 2, 3, 0's hessians  for example 0, 1, 2, 3
        // thread 4, 5, 6, 7 now process feature 1, 2, 3, 0's gradients for example 4, 5, 6, 7
        addr += 4 - 8 * is_hessian_first;
        atomic_local_add_f(gh_hist + addr, stat2);
        // thread 0, 1, 2, 3 now process feature 2, 3, 0, 1's gradients for example 0, 1, 2, 3
        // thread 4, 5, 6, 7 now process feature 2, 3, 0, 1's hessians  for example 4, 5, 6, 7
        bin = feature4.s1;
        offset = (offset + 1) & 0x3;
        addr = bin * 8 + is_hessian_first * 4 + offset;
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
        bin = feature4.s0;
        offset = (offset + 1) & 0x3;
        addr = bin * 8 + is_hessian_first * 4 + offset;
        atomic_local_add_f(gh_hist + addr, stat1);
        // thread 0, 1, 2, 3 now process feature 3, 0, 1, 2's hessians  for example 0, 1, 2, 3
        // thread 4, 5, 6, 7 now process feature 3, 0, 1, 2's gradients for example 4, 5, 6, 7
        addr += 4 - 8 * is_hessian_first;
        atomic_local_add_f(gh_hist + addr, stat2);
        // STAGE 3: accumulate counter
        // there are 4 counters for 4 features
        // thread 0, 1, 2, 3 now process feature 0, 1, 2, 3's counts for example 0, 1, 2, 3
        bin = feature4.s3;
        offset = (ltid & 0x3);
        addr = bin * 4 + offset;
        atom_inc(cnt_hist + addr);
        // thread 0, 1, 2, 3 now process feature 1, 2, 3, 0's counts for example 0, 1, 2, 3
        bin = feature4.s2;
        offset = (offset + 1) & 0x3;
        addr = bin * 4 + offset;
        atom_inc(cnt_hist + addr);
        // thread 0, 1, 2, 3 now process feature 2, 3, 0, 1's counts for example 0, 1, 2, 3
        bin = feature4.s1;
        offset = (offset + 1) & 0x3;
        addr = bin * 4 + offset;
        atom_inc(cnt_hist + addr);
        // thread 0, 1, 2, 3 now process feature 3, 0, 1, 2's counts for example 0, 1, 2, 3
        bin = feature4.s0;
        offset = (offset + 1) & 0x3;
        addr = bin * 4 + offset;
        atom_inc(cnt_hist + addr);
        stat1 = stat1_next;
        stat2 = stat2_next;
        feature4 = feature4_next;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // write to output
    // write gradients and hessians histogram for all 4 features
    /* memory layout in gh_hist (total 2 * 4 * 256 * sizeof(float) = 8 KB):
       -----------------------------------------------------------------------------------------------
       g_f0_bin0   g_f1_bin0   g_f2_bin0   g_f3_bin0   h_f0_bin0   h_f1_bin0   h_f2_bin0   h_f3_bin0
       g_f0_bin1   g_f1_bin1   g_f2_bin1   g_f3_bin1   h_f0_bin1   h_f1_bin1   h_f2_bin1   h_f3_bin1
       ...
       g_f0_bin255 g_f1_bin255 g_f2_bin255 g_f3_bin255 h_f0_bin255 h_f1_bin255 h_f2_bin255 h_f3_bin255
       -----------------------------------------------------------------------------------------------
    */
    /* memory layout in cnt_hist (total 4 * 256 * sizeof(uint) = 4 KB):
       -----------------------------------------------
       c_f0_bin0   c_f1_bin0   c_f2_bin0   c_f3_bin0
       c_f0_bin1   c_f1_bin1   c_f2_bin1   c_f3_bin1
       ...
       c_f0_bin255 c_f1_bin255 c_f2_bin255 c_f3_bin255
       -----------------------------------------------
    */
    // output data in linear order for further reduction
    // output size = 4 (features) * 3 (counters) * 256 (bins) * sizeof(float)
    /* memory layout of output:
       --------------------------------------------
       g_f0_bin0   g_f0_bin1   ...   g_f0_bin255   \
       h_f0_bin0   h_f0_bin1   ...   h_f0_bin255    |
       c_f0_bin0   c_f0_bin1   ...   c_f0_bin255    |
       g_f1_bin0   g_f1_bin1   ...   g_f1_bin255    |
       h_f1_bin0   h_f1_bin1   ...   h_f1_bin255    |
       c_f1_bin0   c_f1_bin1   ...   c_f1_bin255    |--- 1 sub-histogram block
       g_f2_bin0   g_f2_bin1   ...   g_f2_bin255    |
       h_f2_bin0   h_f2_bin1   ...   h_f2_bin255    |
       c_f2_bin0   c_f2_bin1   ...   c_f2_bin255    |
       g_f3_bin0   g_f3_bin1   ...   g_f3_bin255    |
       h_f3_bin0   h_f3_bin1   ...   h_f3_bin255    |
       c_f3_bin0   c_f3_bin1   ...   c_f3_bin255   /
       --------------------------------------------
    */
    uint feature4_id = (group_id >> POWER_FEATURE_WORKGROUPS);
    // if there is only one workgroup processing this feature4, don't even need to write
    #if POWER_FEATURE_WORKGROUPS != 0
    __global float * restrict output = (__global float * restrict)output_buf + group_id * 4 * 3 * NUM_BINS;
    // write gradients and hessians
    __global float * restrict ptr_f = output;
    for (ushort j = 0; j < 4; ++j) {
        for (ushort i = ltid; i < 2 * NUM_BINS; i += lsize) {
            // even threads read gradients, odd threads read hessians
            // FIXME: 2-way bank conflict
            float value = gh_hist[i * 4 + j];
            ptr_f[(i & 1) * NUM_BINS + (i >> 1)] = value;
        }
        ptr_f += 3 * NUM_BINS;
    }
    // write counts
    __global uint * restrict ptr_i = (__global uint * restrict)output + 2 * NUM_BINS;
    for (ushort j = 0; j < 4; ++j) {
        for (ushort i = ltid; i < NUM_BINS; i += lsize) {
            // FIXME: 2-way bank conflict
            uint value = cnt_hist[i * 4 + j];
            ptr_i[i] = value;
        }
        ptr_i += 3 * NUM_BINS;
    }
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
    // backup the old value
    uint old_val = *counter_val;
    if (ltid == 0) {
        // all workgroups processing the same feature add this counter
        *counter_val = atom_inc(sync_counters + feature4_id);
    }
    // make sure everyone in this workgroup is here
    barrier(CLK_LOCAL_MEM_FENCE);
    // everyone in this wrokgroup: if we are the last workgroup, then do reduction!
    if (*counter_val == (1 << POWER_FEATURE_WORKGROUPS) - 1) {
        if (ltid == 0) {
            // printf("workgroup %d: %g %g %g %g %g %g %g %g\n", group_id, gh_hist[0], gh_hist[1], gh_hist[2], gh_hist[3], gh_hist[4], gh_hist[5], gh_hist[6], gh_hist[7]);
            // printf("feature_data[0] = %d %d %d %d", feature_data[0].s0, feature_data[0].s1, feature_data[0].s2, feature_data[0].s3);
            // clear the sync counter for using it next time
            sync_counters[feature4_id] = 0;
        }
    #else
    // only 1 work group, no need to increase counter
    // the reduction will become a simple copy
    if (1) {
        uint old_val; // dummy
    #endif
        // locate our feature4's block in output memory
        uint output_offset = (feature4_id << POWER_FEATURE_WORKGROUPS);
        __global float const * restrict feature4_subhists = 
                 (__global float *)output_buf + output_offset * 4 * 3 * NUM_BINS;
        // skip reading the data already in local memory
        uint skip_id = group_id ^ output_offset;
        // locate output histogram location for this feature4
        __global float* restrict hist_buf = hist_buf_base + feature4_id * 4 * 3 * NUM_BINS;
        within_kernel_reduction256x4(feature4_subhists, skip_id, old_val, 1 << POWER_FEATURE_WORKGROUPS, hist_buf, shared_array);
        // if (ltid == 0) 
        //    printf("workgroup %d reduction done, %g %g %g %g %g %g %g %g\n", group_id, hist_buf[0], hist_buf[3*NUM_BINS], hist_buf[2*3*NUM_BINS], hist_buf[3*3*NUM_BINS], hist_buf[1], hist_buf[3*NUM_BINS+1], hist_buf[2*3*NUM_BINS+1], hist_buf[3*3*NUM_BINS+1]);
    }
}

// The following line ends the string literal, adds an extra #endif at the end
// the +9 skips extra characters ")", newline, "#endif" and newline at the beginning
// )"" "\n#endif" + 9
#endif

