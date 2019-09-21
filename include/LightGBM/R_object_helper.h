/*!
 * Copyright (c) 2017 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * \brief A simple wrapper for accessing data in R object.
 *
 * \note
 * Due to license issue, we cannot include R's header file, so use this simple wrapper instead.
 * However, if R changes the way it defines objects, this file will need to be updated as well.
 */
#ifndef R_OBJECT_HELPER_H_
#define R_OBJECT_HELPER_H_

#include <cstdint>
#include <cstdio>

#define TYPE_BITS 5
// use .Internal(internalsID()) to uuid
#define R_INTERNALS_UUID "2fdf6c18-697a-4ba7-b8ef-11c0d92f1327"


#ifdef R_VER_ABOVE_35
#define NAMED_BITS 16
struct lgbm_sxpinfo {
  unsigned int type : 5;
  unsigned int scalar : 1;
  unsigned int obj : 1;
  unsigned int alt : 1;
  unsigned int gp : 16;
  unsigned int mark : 1;
  unsigned int debug : 1;
  unsigned int trace : 1;
  unsigned int spare : 1;
  unsigned int gcgen : 1;
  unsigned int gccls : 3;
  unsigned int named : NAMED_BITS;
  unsigned int extra : 32 - NAMED_BITS;
};

// 64bit pointer
#if INTPTR_MAX == INT64_MAX
typedef int64_t R_xlen_t;
#else
typedef int R_xlen_t;
#endif

#else
struct lgbm_sxpinfo {
  unsigned int type : 5;
  unsigned int obj : 1;
  unsigned int named : 2;
  unsigned int gp : 16;
  unsigned int mark : 1;
  unsigned int debug : 1;
  unsigned int trace : 1;
  unsigned int spare : 1;
  unsigned int gcgen : 1;
  unsigned int gccls : 3;
};

typedef int R_xlen_t;
#endif

struct lgbm_primsxp {
  int offset;
};

struct lgbm_symsxp {
  struct LGBM_SER *pname;
  struct LGBM_SER *value;
  struct LGBM_SER *internal;
};

struct lgbm_listsxp {
  struct LGBM_SER *carval;
  struct LGBM_SER *cdrval;
  struct LGBM_SER *tagval;
};

struct lgbm_envsxp {
  struct LGBM_SER *frame;
  struct LGBM_SER *enclos;
  struct LGBM_SER *hashtab;
};

struct lgbm_closxp {
  struct LGBM_SER *formals;
  struct LGBM_SER *body;
  struct LGBM_SER *env;
};

struct lgbm_promsxp {
  struct LGBM_SER *value;
  struct LGBM_SER *expr;
  struct LGBM_SER *env;
};

typedef struct LGBM_SER {
  struct lgbm_sxpinfo sxpinfo;
  struct LGBM_SER* attrib;
  struct LGBM_SER* gengc_next_node, *gengc_prev_node;
  union {
    struct lgbm_primsxp primsxp;
    struct lgbm_symsxp symsxp;
    struct lgbm_listsxp listsxp;
    struct lgbm_envsxp envsxp;
    struct lgbm_closxp closxp;
    struct lgbm_promsxp promsxp;
  } u;
} LGBM_SER, *LGBM_SE;

struct lgbm_vecsxp {
  R_xlen_t length;
  R_xlen_t truelength;
};

typedef struct VECTOR_SER {
  struct lgbm_sxpinfo sxpinfo;
  struct LGBM_SER* attrib;
  struct LGBM_SER* gengc_next_node, *gengc_prev_node;
  struct lgbm_vecsxp vecsxp;
} VECTOR_SER, *VECSE;

typedef union { VECTOR_SER s; double align; } SEXPREC_ALIGN;

#define DATAPTR(x)  ((reinterpret_cast<SEXPREC_ALIGN*>(x)) + 1)

#define R_CHAR_PTR(x)     (reinterpret_cast<char*>DATAPTR(x))

#define R_INT_PTR(x)  (reinterpret_cast<int*> DATAPTR(x))

#define R_INT64_PTR(x)  (reinterpret_cast<int64_t*> DATAPTR(x))

#define R_REAL_PTR(x)     (reinterpret_cast<double*> DATAPTR(x))

#define R_AS_INT(x) (*(reinterpret_cast<int*> DATAPTR(x)))

#define R_AS_INT64(x) (*(reinterpret_cast<int64_t*> DATAPTR(x)))

#define R_IS_NULL(x) ((*reinterpret_cast<LGBM_SE>(x)).sxpinfo.type == 0)

// 64bit pointer
#if INTPTR_MAX == INT64_MAX

#define R_ADDR(x)  (reinterpret_cast<int64_t*> DATAPTR(x))

inline void R_SET_PTR(LGBM_SE x, void* ptr) {
  if (ptr == nullptr) {
    R_ADDR(x)[0] = (int64_t)(NULL);
  } else {
    R_ADDR(x)[0] = (int64_t)(ptr);
  }
}

inline void* R_GET_PTR(LGBM_SE x) {
  if (R_IS_NULL(x)) {
    return nullptr;
  } else {
    auto ret = reinterpret_cast<void*>(R_ADDR(x)[0]);
    if (ret == NULL) {
      ret = nullptr;
    }
    return ret;
  }
}

#else

#define R_ADDR(x)  (reinterpret_cast<int32_t*> DATAPTR(x))

inline void R_SET_PTR(LGBM_SE x, void* ptr) {
  if (ptr == nullptr) {
    R_ADDR(x)[0] = (int32_t)(NULL);
  } else {
    R_ADDR(x)[0] = (int32_t)(ptr);
  }
}

inline void* R_GET_PTR(LGBM_SE x) {
  if (R_IS_NULL(x)) {
    return nullptr;
  } else {
    auto ret = reinterpret_cast<void*>(R_ADDR(x)[0]);
    if (ret == NULL) {
      ret = nullptr;
    }
    return ret;
  }
}

#endif

#endif  // R_OBJECT_HELPER_H_
