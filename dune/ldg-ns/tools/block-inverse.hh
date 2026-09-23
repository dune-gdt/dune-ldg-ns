// This file is part of the dune-ldg-ns project:
//   https://github.com/dune-gdt/dune-ldg-ns
// Copyright 2026 dune-ldg-ns developers and contributors. All rights reserved.
// License: Dual licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
//      or  GPL-2.0+ (http://opensource.org/licenses/gpl-license)
//          with "runtime exception" (http://www.dune-project.org/license.html)
// Authors:
//   René Fritze (2026)

#ifndef DUNE_LDG_NS_TOOLS_BLOCK_INVERSE_HH
#define DUNE_LDG_NS_TOOLS_BLOCK_INVERSE_HH

#include <vector>

#include <dune/common/dynmatrix.hh>
#include <dune/common/dynvector.hh>

#include <dune/xt/common/disable_warnings.hh>
#include <Eigen/SparseCore>
#include <dune/xt/common/reenable_warnings.hh>

#include <dune/gdt/spaces/interface.hh>

namespace Dune {
namespace GDT {
namespace NavierStokes {


/**
 * \brief Exact inverse of an element-block-diagonal matrix (e.g. the mass matrix of a DG space).
 *
 * The blocks are given by the local DoFs of each element of the space; entries of the input outside these blocks are
 * ignored (and must be zero for the result to be the inverse).
 */
template <class GV, size_t r, size_t rC, class R, class SparseType>
SparseType element_block_inverse(const SpaceInterface<GV, r, rC, R>& space, const SparseType& block_diagonal_matrix)
{
  using TripletType = ::Eigen::Triplet<double>;
  std::vector<TripletType> triplets;
  triplets.reserve(space.mapper().size() * space.mapper().max_local_size());
  DynamicVector<size_t> indices(space.mapper().max_local_size());
  for (auto&& element : elements(space.grid_view())) {
    const size_t n = space.mapper().local_size(element);
    space.mapper().global_indices(element, indices);
    DynamicMatrix<double> block(n, n, 0.);
    for (size_t ii = 0; ii < n; ++ii)
      for (size_t jj = 0; jj < n; ++jj)
        block[ii][jj] = block_diagonal_matrix.coeff(indices[ii], indices[jj]);
    block.invert();
    for (size_t ii = 0; ii < n; ++ii)
      for (size_t jj = 0; jj < n; ++jj)
        triplets.emplace_back(indices[ii], indices[jj], block[ii][jj]);
  }
  SparseType result(block_diagonal_matrix.rows(), block_diagonal_matrix.cols());
  result.setFromTriplets(triplets.begin(), triplets.end());
  result.makeCompressed();
  return result;
} // ... element_block_inverse(...)


} // namespace NavierStokes
} // namespace GDT
} // namespace Dune

#endif // DUNE_LDG_NS_TOOLS_BLOCK_INVERSE_HH
