# This file is part of the dune-ldg-ns project:
#   https://github.com/dune-gdt/dune-ldg-ns
# Copyright 2026 dune-ldg-ns developers and contributors. All rights reserved.
# License: Dual licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#      or  GPL-2.0+ (http://opensource.org/licenses/gpl-license)
#          with "runtime exception" (http://www.dune-project.org/license.html)
# Authors:
#   René Fritze (2026)

# A regular DUNE module on top of a *configured and built* dune-gdt tree (dune-gdt main, CMake presets + vcpkg).
# The third-party dependencies are taken from dune-gdt's vcpkg installation instead of being built a second time, see
# CMakePresets.json and README.md.
cmake_minimum_required(VERSION 3.29...4.0)
project(
  dune-ldg-ns
  VERSION 0.1.0
  DESCRIPTION "LDG discretization and fractional-step theta time stepping for the incompressible Navier-Stokes equations"
  LANGUAGES CXX)

find_package(dune-common REQUIRED)
list(APPEND CMAKE_MODULE_PATH "${dune-common_MODULE_PATH}" "${PROJECT_SOURCE_DIR}/cmake/modules")
include(DuneMacros)

# dune_project() resolves "Depends: dune-gdt" from dune.module and includes dune-gdt's DuneGdtMacros.cmake, which
# finds Boost, Eigen, LAPACKE and TBB.
dune_project()
dune_enable_all_packages()

find_package(GTest CONFIG REQUIRED)

add_subdirectory(cmake/modules)
add_subdirectory(dune)
add_subdirectory(src)
add_subdirectory(test)

finalize_dune_project()
