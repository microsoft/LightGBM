/*
* A simple wrapper for access data in R object.
* Due to license issue(GPLv2), we cannot include R's header file, so use this simple wrapper instead.
* However, If R change its define of object, this file need to be updated as well.
*/
#ifndef R_OBJECT_HELPER_H_
#define R_OBJECT_HELPER_H_

#include <cstdint>

#define TYPE_BITS 5
struct sxpinfo_struct {
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

struct primsxp_struct {
  int offset;
};

struct symsxp_struct {
  struct SEXPREC *pname;
  struct SEXPREC *value;
  struct SEXPREC *internal;
};

struct listsxp_struct {
  struct SEXPREC *carval;
  struct SEXPREC *cdrval;
  struct SEXPREC *tagval;
};

struct envsxp_struct {
  struct SEXPREC *frame;
  struct SEXPREC *enclos;
  struct SEXPREC *hashtab;
};

struct closxp_struct {
  struct SEXPREC *formals;
  struct SEXPREC *body;
  struct SEXPREC *env;
};

struct promsxp_struct {
  struct SEXPREC *value;
  struct SEXPREC *expr;
  struct SEXPREC *env;
};

typedef struct SEXPREC {
  struct sxpinfo_struct sxpinfo;
  struct SEXPREC* attrib;
  struct SEXPREC* gengc_next_node, *gengc_prev_node;
  union {
    struct primsxp_struct primsxp;
    struct symsxp_struct symsxp;
    struct listsxp_struct listsxp;
    struct envsxp_struct envsxp;
    struct closxp_struct closxp;
    struct promsxp_struct promsxp;
  } u;
} SEXPREC, *SEXP;

struct vecsxp_struct {
  int length;
  int truelength;
};

typedef struct VECTOR_SEXPREC {
  struct sxpinfo_struct sxpinfo;
  struct SEXPREC* attrib;
  struct SEXPREC* gengc_next_node, *gengc_prev_node;
  struct vecsxp_struct vecsxp;
} VECTOR_SEXPREC, *VECSEXP;

typedef union { VECTOR_SEXPREC s; double align; } SEXPREC_ALIGN;

#define DATAPTR(x)  (((SEXPREC_ALIGN *) (x)) + 1)

#define R_CHAR_PTR(x)     ((char *) DATAPTR(x))

#define R_INT_PTR(x)  ((int *) DATAPTR(x))

#define R_REAL_PTR(x)     ((double *) DATAPTR(x))

#define R_AS_INT(x) (*((int *) DATAPTR(x)))

#define R_IS_NULL(x) ((*(SEXP)(x)).sxpinfo.type == 0)


// 64bit pointer
#if INTPTR_MAX == INT64_MAX

#define R_ADDR(x)  ((int64_t *) DATAPTR(x))

inline void R_SET_PTR(SEXP x, void* ptr) {
  if (ptr == nullptr) {
    R_ADDR(x)[0] = (int64_t)(NULL);
  } else {
    R_ADDR(x)[0] = (int64_t)(ptr);
  }
}

inline void* R_GET_PTR(SEXP x) {
  if (R_IS_NULL(x)) {
    return nullptr;
  } else {
    auto ret = (void *)(R_ADDR(x)[0]);
    if (ret == NULL) {
      ret = nullptr;
    }
    return ret;
  }
}

#else

#define R_ADDR(x)  ((int32_t *) DATAPTR(x))

inline void R_SET_PTR(SEXP x, void* ptr) {
  if (ptr == nullptr) {
    R_ADDR(x)[0] = (int32_t)(NULL);
  } else {
    R_ADDR(x)[0] = (int32_t)(ptr);
  }
}

inline void* R_GET_PTR(SEXP x) {
  if (R_IS_NULL(x)) {
    return nullptr;
  } else {
    auto ret = (void *)(R_ADDR(x)[0]);
    if (ret == NULL) {
      ret = nullptr;
    }
    return ret;
  }
}

#endif

#endif // R_OBJECT_HELPER_H_
