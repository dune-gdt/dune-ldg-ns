# This file is part of the dune-ldg-ns project:
#   https://github.com/dune-gdt/dune-ldg-ns
# License: Dual licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#      or  GPL-2.0+ (http://opensource.org/licenses/gpl-license)
#          with "runtime exception" (http://www.dune-project.org/license.html)
# Authors:
#   René Fritze (2026)

# Standalone development build that bypasses dune-gdt's CMake/vcpkg machinery.
#
# This is how the module was developed and validated in an environment without access to vcpkg's download mirrors:
#   - DUNE core modules + dune-alugrid (2.10, commits pinned by the dune-gdt vcpkg registry) installed to DUNE_PREFIX,
#   - oneTBB installed to DUNE_PREFIX,
#   - Boost headers (merged libs/*/include of the boost superproject) in BOOST_INCLUDE_DIR and the boost sources in
#     BOOST_SOURCE_DIR (filesystem, timer, chrono and atomic are compiled from source here),
#   - Eigen 3.4 in EIGEN3_INCLUDE_DIR, googletest sources in GTEST_SOURCE_DIR,
#   - a dune-gdt checkout in DUNE_GDT_SOURCE_DIR, whose dune/xt sources are compiled into a static "dunext".
# See README.md ("Standalone development build") for the exact commits. Use the DUNE module build for everything else.
cmake_minimum_required(VERSION 3.20)
project(dune-ldg-ns-standalone CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(DUNE_GDT_SOURCE_DIR "" CACHE PATH "dune-gdt checkout")
set(DUNE_PREFIX "" CACHE PATH "install prefix of dune-{common,geometry,grid,istl,localfunctions,alugrid} and oneTBB")
set(BOOST_INCLUDE_DIR "" CACHE PATH "merged boost headers")
set(BOOST_SOURCE_DIR "" CACHE PATH "boost superproject checkout (libs/filesystem/src, ...)")
set(EIGEN3_INCLUDE_DIR "" CACHE PATH "Eigen 3.4")
set(GTEST_SOURCE_DIR "" CACHE PATH "googletest checkout")
foreach(_var DUNE_GDT_SOURCE_DIR DUNE_PREFIX BOOST_INCLUDE_DIR BOOST_SOURCE_DIR EIGEN3_INCLUDE_DIR GTEST_SOURCE_DIR)
  if(NOT ${_var})
    message(FATAL_ERROR "${_var} is required")
  endif()
endforeach()
get_filename_component(LDG_NS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(BOOST_LIBS ${BOOST_SOURCE_DIR}/libs)

file(GLOB XT_SOURCES ${DUNE_GDT_SOURCE_DIR}/dune/xt/common/*.cc ${DUNE_GDT_SOURCE_DIR}/dune/xt/common/parallel/*.cc
     ${DUNE_GDT_SOURCE_DIR}/dune/xt/grid/boundaryinfo/*.cc ${DUNE_GDT_SOURCE_DIR}/dune/xt/la/container/*.cc
     ${DUNE_GDT_SOURCE_DIR}/dune/xt/la/container/eigen/*.cc ${DUNE_GDT_SOURCE_DIR}/dune/xt/la/solver/*.cc)
list(FILTER XT_SOURCES EXCLUDE REGEX "(python|interpreter|mkl|lapacke)\\.cc$")
file(GLOB BOOST_SOURCES ${BOOST_LIBS}/filesystem/src/*.cpp ${BOOST_LIBS}/timer/src/*.cpp ${BOOST_LIBS}/chrono/src/*.cpp
     ${BOOST_LIBS}/atomic/src/lock_pool.cpp)

add_library(ldg_ns_common INTERFACE)
target_include_directories(ldg_ns_common SYSTEM INTERFACE ${BOOST_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR} ${DUNE_PREFIX}/include
                                                          ${GTEST_SOURCE_DIR}/googletest/include)
target_include_directories(ldg_ns_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR} ${DUNE_GDT_SOURCE_DIR} ${LDG_NS_SOURCE_DIR})
target_compile_definitions(ldg_ns_common INTERFACE HAVE_CONFIG_H=1 ENABLE_TBB=1 BOOST_FILESYSTEM_NO_DEPRECATED
                                                   BOOST_ALL_NO_LIB)
target_compile_options(ldg_ns_common INTERFACE -include ${CMAKE_CURRENT_SOURCE_DIR}/config.h -Wno-deprecated-declarations)

add_library(ldg_ns_boost STATIC ${BOOST_SOURCES})
target_include_directories(ldg_ns_boost SYSTEM PRIVATE ${BOOST_INCLUDE_DIR} ${BOOST_LIBS}/filesystem/src ${BOOST_LIBS}/atomic/src)
target_compile_definitions(ldg_ns_boost PRIVATE BOOST_FILESYSTEM_NO_CXX20_ATOMIC_REF BOOST_ALL_NO_LIB
                                                BOOST_FILESYSTEM_STATIC_LINK=1 BOOST_TIMER_STATIC_LINK=1 BOOST_CHRONO_STATIC_LINK=1)
add_library(ldg_ns_dunext STATIC ${XT_SOURCES})
target_link_libraries(ldg_ns_dunext PUBLIC ldg_ns_common)
add_library(ldg_ns_gtest STATIC ${GTEST_SOURCE_DIR}/googletest/src/gtest-all.cc)
target_include_directories(ldg_ns_gtest PUBLIC ${GTEST_SOURCE_DIR}/googletest/include PRIVATE ${GTEST_SOURCE_DIR}/googletest)

set(LDG_NS_LIBS ldg_ns_dunext ldg_ns_boost ${DUNE_PREFIX}/lib/libdunealugrid.a ${DUNE_PREFIX}/lib/libdunegrid.a
    ${DUNE_PREFIX}/lib/libdunegeometry.a ${DUNE_PREFIX}/lib/libdunecommon.a ${DUNE_PREFIX}/lib/libtbb.so pthread z)

enable_testing()
foreach(_test fstheta_coefficients kovasznay taylor_green)
  add_executable(test_${_test} ${LDG_NS_SOURCE_DIR}/test/${_test}.cc)
  target_link_libraries(test_${_test} PRIVATE ${LDG_NS_LIBS} ldg_ns_gtest)
  set_target_properties(test_${_test} PROPERTIES BUILD_RPATH ${DUNE_PREFIX}/lib)
  add_test(NAME ${_test} COMMAND test_${_test})
endforeach()
add_executable(ldg_ns_dfg_cylinder ${LDG_NS_SOURCE_DIR}/src/dfg_cylinder.cc)
target_link_libraries(ldg_ns_dfg_cylinder PRIVATE ${LDG_NS_LIBS})
set_target_properties(ldg_ns_dfg_cylinder PROPERTIES BUILD_RPATH ${DUNE_PREFIX}/lib)
