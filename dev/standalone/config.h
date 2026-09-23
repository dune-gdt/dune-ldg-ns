/* Hand-written config.h for the standalone development build (see CMakeLists.txt), NOT used by the DUNE module build. */
#ifndef DUNE_LDG_NS_STANDALONE_CONFIG_H
#define DUNE_LDG_NS_STANDALONE_CONFIG_H
#define HAVE_DUNE_COMMON 1
#define HAVE_DUNE_GEOMETRY 1
#define HAVE_DUNE_GRID 1
#define HAVE_DUNE_ISTL 1
#define HAVE_DUNE_LOCALFUNCTIONS 1
#define HAVE_DUNE_ALUGRID 1
#define HAVE_DUNE_UGGRID 0
#define HAVE_DUNE_SPGRID 0
#define HAVE_DUNE_GRID_GLUE 0
#define HAVE_DUNE_TESTTOOLS 0
#define DUNE_MINIMAL_DEBUG_LEVEL 4
#define DUNE_HAVE_CXX_EXPERIMENTAL_IS_DETECTED 1
#define DUNE_HAVE_CXX_UNEVALUATED_CONTEXT_LAMBDA 1
#define DUNE_HAVE_CXX_STD_IDENTITY 1
#define HAVE_MKSTEMP 1
#include <dune-common-config.hh>
#include <dune-geometry-config.hh>
#include <dune-grid-config.hh>
#include <dune-istl-config.hh>
#include <dune-localfunctions-config.hh>
#include <dune-alugrid-config.hh>
#define DUNE_GDT_VERSION 0.0.0
#define HAVE_LAPACKE 0
#define HAVE_MKL 0
#define HAVE_LIKWID 0
#define ENABLE_PERFMON 0
#define HAVE_MAP_EMPLACE 1
#define HAS_WORKING_UNUSED_ATTRIBUTE 1
#define DS_MAX_MIC_THREADS 0
#define HAVE_SPE10_DATA 0
#define DXT_DISABLE_LARGE_TESTS 0
#define DUNE_XT_DO_TIMING 0
#define DUNE_XT_WITH_PYTHON_BINDINGS 0
#define ENABLE_ALBERTA 0
#define HAVE_ALBERTA 0
#define ALBERTA_DIM 2
#define HAVE_FASP 0
#define ENABLE_SUPERLU 0
#define HAVE_SUPERLU 0
#define ENABLE_UMFPACK 0
#ifndef HAVE_UMFPACK
#define HAVE_UMFPACK 0
#endif
#define HAVE_SUITESPARSE_UMFPACK 0
#define ENABLE_BOOST 0
#define ENABLE_MPI 0
#define HAVE_MPI 0
#define HAVE_TBB 1
#define HAVE_EIGEN 1
#define TBB_PREVIEW_GLOBAL_CONTROL 1
#include <dune/xt/common/fix-ambiguous-std-math-overloads.hh>
#endif
