#ifndef NUMPY_MACROS__H
#define NUMPY_MACROS__H

/*
This is from the book 'Python Scripting for Computational Science'
by Hans Petter Langtangen, with some modifications.
*/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

/* define some macros for array dimension/type check
   and subscripting */
#define QUOTE(s) # s   /* turn s into string "s" */
#define NDIM_CHECK(a, expected_ndim) \
  if (PyArray_NDIM(a) != expected_ndim) { \
    PyErr_Format(PyExc_ValueError, \
    "%s array is %d-dimensional, but expected to be %d-dimensional",\
		 QUOTE(a), PyArray_NDIM(a), expected_ndim); \
    return NULL; \
  }
#define DIM_CHECK(a, dim, expected_length) \
  if (dim > PyArray_NDIM(a)) { \
    PyErr_Format(PyExc_ValueError, \
    "%s array has no %d dimension (max dim. is %d)", \
		 QUOTE(a), dim, PyArray_NDIM(a)); \
    return NULL; \
  } \
  if (PyArray_DIM(a, dim) != expected_length) { \
    PyErr_Format(PyExc_ValueError, \
    "%s array has wrong %d-dimension=%ld (expected %d)", \
		 QUOTE(a), dim, PyArray_DIM(a, dim), expected_length); \
    return NULL; \
  }
#define TYPE_CHECK(a, tp) \
  if (PyArray_TYPE(a) != tp) { \
    PyErr_Format(PyExc_TypeError, \
    "%s array is not of correct type (%d)", QUOTE(a), tp); \
    return NULL; \
  }
#define CALLABLE_CHECK(func) \
  if (!PyCallable_Check(func)) { \
    PyErr_Format(PyExc_TypeError, \
    "%s is not a callable function", QUOTE(func)); \
    return NULL; \
  }
#define CARRAY_CHECK(a) \
  if (!(PyArray_ISCONTIGUOUS(a) && PyArray_ISCARRAY(a))) { \
    PyErr_Format(PyExc_TypeError, \
    "%s is not a contiguous c-array", QUOTE(a)); \
    return NULL; \
  }
#define FARRAY_CHECK(a) \
  if (!(PyArray_ISFARRAY(a))) { \
    PyErr_Format(PyExc_TypeError, \
    "%s is not a contiguous f-array", QUOTE(a)); \
    return NULL; \
  }
#define CHECK(assertion, message) \
  if (!(assertion)) { \
    PyErr_Format(PyExc_ValueError, message); \
    return NULL; \
  }

#define DIND1(a, i) *((npy_float64 *) PyArray_GETPTR1(a, i))
#define DIND2(a, i, j) \
 *((npy_float64 *) PyArray_GETPTR2(a, i, j))
#define DIND3(a, i, j, k) \
 *((npy_float64 *) Py_Array_GETPTR3(a, i, j, k))

#define FIND1(a, i) *((npy_float32 *) PyArray_GETPTR1(a, i))
#define FIND2(a, i, j) \
 *((npy_float32 *) PyArray_GETPTR2(a, i, j))
#define FIND3(a, i, j, k) \
 *((npy_float32 *) Py_Array_GETPTR3(a, i, j, k))

#define IIND1(a, i) *((npy_int *) PyArray_GETPTR1(a, i))
#define IIND2(a, i, j) \
 *((npy_int *) PyArray_GETPTR2(a, i, j))
#define IIND3(a, i, j, k) \
 *((npy_int *) Py_Array_GETPTR3(a, i, j, k))

#define U8IND2(a, i, j) \
 *((npy_uint8 *) PyArray_GETPTR2(a, i, j))

#define U8IND3(a, i, j, k) \
 *((npy_uint8 *) PyArray_GETPTR3(a, i, j, k))

#endif
