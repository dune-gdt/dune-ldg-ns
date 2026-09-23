// This file is part of the dune-ldg-ns project:
//   https://github.com/dune-gdt/dune-ldg-ns
// Copyright 2026 dune-ldg-ns developers and contributors. All rights reserved.
// License: Dual licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
//      or  GPL-2.0+ (http://opensource.org/licenses/gpl-license)
//          with "runtime exception" (http://www.dune-project.org/license.html)
// Authors:
//   René Fritze (2026)

/**
 * \file  saddle-point.hh
 * \brief Monolithic (Oseen-type) saddle point systems of the LDG discretization and their direct solution.
 *
 * Solves
 *   [ a M + c (A_visc + T)   (block diagonal, d copies)     tau B^T      (q) ] [ u      ]   [ r_u          ]
 *   [ tau B                                                -tau C        (0) ] [ lambda ] = [ tau r_p      ]
 *   [                                                        tau q^T      0 ] [ mu     ]   [ 0            ]
 * where the last row/column (zero-mean pressure) is only present if requested.
 *
 * The pattern of the monolithic matrix does not change between calls (all blocks carry the full DG stencil), so the
 * symbolic factorization is computed only once.
 *
 * \todo Block-preconditioned FGMRES (dune-istl) for 3D, see doc/design.md, Sec. 6.
 */
#ifndef DUNE_LDG_NS_SOLVERS_SADDLE_POINT_HH
#define DUNE_LDG_NS_SOLVERS_SADDLE_POINT_HH

#include <memory>
#include <string>
#include <vector>

#include <dune/xt/common/disable_warnings.hh>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#if HAVE_SUITESPARSE_UMFPACK
#  include <Eigen/UmfPackSupport>
#endif
#include <dune/xt/common/reenable_warnings.hh>

#include <dune/common/exceptions.hh>

namespace Dune {
namespace GDT {
namespace NavierStokes {


struct SaddlePointSolverOptions
{
  std::string type = "sparselu"; //!< "sparselu" (Eigen, always available) or "umfpack" (if SuiteSparse was found)
  bool mean_pressure_constraint = false; //!< append int_Omega p = 0 (pure Dirichlet problems)
};


template <size_t d>
class SaddlePointSolver
{
public:
  using RowMajorSparseType = ::Eigen::SparseMatrix<double, ::Eigen::RowMajor>;
  using SparseType = ::Eigen::SparseMatrix<double, ::Eigen::ColMajor>;
  using DenseVectorType = ::Eigen::VectorXd;

  SaddlePointSolver(const std::array<RowMajorSparseType, d>& divergence,
                    const RowMajorSparseType& pressure_stabilization,
                    const DenseVectorType& pressure_basis_integrals,
                    const SaddlePointSolverOptions& options = {})
    : divergence_(divergence)
    , pressure_stabilization_(pressure_stabilization)
    , pressure_basis_integrals_(pressure_basis_integrals)
    , options_(options)
    , N_(divergence[0].cols())
    , P_(divergence[0].rows())
  {
#if !HAVE_SUITESPARSE_UMFPACK
    DUNE_THROW_IF(options_.type == "umfpack", Dune::NotImplemented, "UMFPACK support is not available!");
#endif
  }

  size_t size() const
  {
    return d * N_ + P_ + (options_.mean_pressure_constraint ? 1 : 0);
  }

  /**
   * \param velocity_block the scalar N x N block a M + c (A_visc + T), used for each velocity component
   * \param tau            scaling of the pressure coupling (substep length, 1 for steady problems)
   * \param rhs_u          d*N right hand side of the momentum equations
   * \param rhs_p          P right hand side of the continuity equation (g_B, unscaled)
   * \param u              d*N solution (velocity)
   * \param lambda         P solution (pressure multiplier)
   */
  void solve(const RowMajorSparseType& velocity_block,
             const double tau,
             const DenseVectorType& rhs_u,
             const DenseVectorType& rhs_p,
             DenseVectorType& u,
             DenseVectorType& lambda)
  {
    assemble(velocity_block, tau);
    DenseVectorType rhs = DenseVectorType::Zero(size());
    rhs.segment(0, d * N_) = rhs_u;
    rhs.segment(d * N_, P_) = tau * rhs_p;
    DenseVectorType x;
    if (options_.type == "sparselu") {
      if (!sparselu_) {
        sparselu_ = std::make_unique<::Eigen::SparseLU<SparseType, ::Eigen::COLAMDOrdering<int>>>();
        sparselu_->analyzePattern(matrix_);
      }
      sparselu_->factorize(matrix_);
      DUNE_THROW_IF(sparselu_->info() != ::Eigen::Success,
                    Dune::MathError,
                    "SparseLU factorization failed: " << sparselu_->lastErrorMessage());
      x = sparselu_->solve(rhs);
    }
#if HAVE_SUITESPARSE_UMFPACK
    else if (options_.type == "umfpack") {
      if (!umfpack_) {
        umfpack_ = std::make_unique<::Eigen::UmfPackLU<SparseType>>();
        umfpack_->analyzePattern(matrix_);
      }
      umfpack_->factorize(matrix_);
      DUNE_THROW_IF(umfpack_->info() != ::Eigen::Success, Dune::MathError, "UMFPACK factorization failed!");
      x = umfpack_->solve(rhs);
    }
#endif
    else
      DUNE_THROW(Dune::NotImplemented, "Unknown solver type '" << options_.type << "'!");
    u = x.segment(0, d * N_);
    lambda = x.segment(d * N_, P_);
  } // ... solve(...)

  /// \brief The last assembled monolithic matrix (for inspection and tests).
  const SparseType& matrix() const
  {
    return matrix_;
  }

private:
  void assemble(const RowMajorSparseType& velocity_block, const double tau)
  {
    using TripletType = ::Eigen::Triplet<double>;
    std::vector<TripletType> triplets;
    triplets.reserve(d * velocity_block.nonZeros() + 2 * d * divergence_[0].nonZeros()
                     + pressure_stabilization_.nonZeros() + 2 * P_);
    for (size_t cc = 0; cc < d; ++cc)
      for (int row = 0; row < velocity_block.outerSize(); ++row)
        for (RowMajorSparseType::InnerIterator it(velocity_block, row); it; ++it)
          triplets.emplace_back(cc * N_ + it.row(), cc * N_ + it.col(), it.value());
    for (size_t jj = 0; jj < d; ++jj)
      for (int row = 0; row < divergence_[jj].outerSize(); ++row)
        for (RowMajorSparseType::InnerIterator it(divergence_[jj], row); it; ++it) {
          triplets.emplace_back(d * N_ + it.row(), jj * N_ + it.col(), tau * it.value());
          triplets.emplace_back(jj * N_ + it.col(), d * N_ + it.row(), tau * it.value());
        }
    for (int row = 0; row < pressure_stabilization_.outerSize(); ++row)
      for (RowMajorSparseType::InnerIterator it(pressure_stabilization_, row); it; ++it)
        triplets.emplace_back(d * N_ + it.row(), d * N_ + it.col(), -tau * it.value());
    if (options_.mean_pressure_constraint) {
      const size_t last = d * N_ + P_;
      for (size_t kk = 0; kk < P_; ++kk) {
        triplets.emplace_back(d * N_ + kk, last, tau * pressure_basis_integrals_[kk]);
        triplets.emplace_back(last, d * N_ + kk, tau * pressure_basis_integrals_[kk]);
      }
    }
    matrix_.resize(size(), size());
    matrix_.setFromTriplets(triplets.begin(), triplets.end());
    matrix_.makeCompressed();
  } // ... assemble(...)

  const std::array<RowMajorSparseType, d>& divergence_;
  const RowMajorSparseType& pressure_stabilization_;
  const DenseVectorType& pressure_basis_integrals_;
  const SaddlePointSolverOptions options_;
  const size_t N_;
  const size_t P_;
  SparseType matrix_;
  std::unique_ptr<::Eigen::SparseLU<SparseType, ::Eigen::COLAMDOrdering<int>>> sparselu_;
#if HAVE_SUITESPARSE_UMFPACK
  std::unique_ptr<::Eigen::UmfPackLU<SparseType>> umfpack_;
#endif
}; // class SaddlePointSolver


} // namespace NavierStokes
} // namespace GDT
} // namespace Dune

#endif // DUNE_LDG_NS_SOLVERS_SADDLE_POINT_HH
