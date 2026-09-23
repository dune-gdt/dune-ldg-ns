// This file is part of the dune-ldg-ns project:
//   https://github.com/dune-gdt/dune-ldg-ns
// Copyright 2026 dune-ldg-ns developers and contributors. All rights reserved.
// License: Dual licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
//      or  GPL-2.0+ (http://opensource.org/licenses/gpl-license)
//          with "runtime exception" (http://www.dune-project.org/license.html)
// Authors:
//   René Fritze (2026)

/**
 * \file  ldg-navier-stokes.hh
 * \brief Spatial LDG discretization of the incompressible Navier--Stokes equations (doc/design.md, Sec. 3).
 *
 * Unknowns (component-blocked, "dimwise" DoF numbering):
 *   u = (u_0, ..., u_{d-1}), u_i in S_h = DG Q_k/P_k (scalar),   p in Q_h = DG Q_{k-1}/P_{k-1} (or equal order + jump
 *   stabilization), sigma_ij = (d u_i / d x_j)_h in S_h (eliminated, block-diagonal mass matrix).
 *
 * Scalar building blocks (all N x N resp. P x N, N = dim S_h, P = dim Q_h):
 *   M            DG mass matrix (element-block diagonal), Minv its exact inverse,
 *   G_j          LDG gradient in direction j:   M sigma_ij = G_j u_i + b_ij(g_D),
 *   J            LDG C11 penalty (nu eta / h_F) on inner and Dirichlet faces,
 *   A_visc       = nu sum_j G_j^T Minv G_j + J    (identical for every velocity component),
 *   B_j          DG divergence, B_j = -G_j(beta = 0) with pressure test functions,
 *   T(w)         Oseen-linearized DG convection (identical for every velocity component),
 *   C            optional pressure jump stabilization (gamma h_F [[p]][[q]]).
 *
 * The semi-discrete system reads
 *   M du_i/dt + (A_visc + T(u)) u_i + B_i^T p = F_i(t; u),     sum_j B_j u_j - C p = g_B(t),
 * with the data vectors F_i (force, viscous and convective Dirichlet contributions) and g_B.
 */
#ifndef DUNE_LDG_NS_OPERATORS_LDG_NAVIER_STOKES_HH
#define DUNE_LDG_NS_OPERATORS_LDG_NAVIER_STOKES_HH

#include <array>
#include <memory>
#include <vector>

#include <dune/xt/common/disable_warnings.hh>
#include <Eigen/SparseCore>
#include <dune/xt/common/reenable_warnings.hh>

#include <dune/xt/common/parameter.hh>
#include <dune/xt/common/timedlogging.hh>
#include <dune/xt/functions/constant.hh>
#include <dune/xt/functions/grid-function.hh>
#include <dune/xt/grid/boundaryinfo/interfaces.hh>
#include <dune/xt/grid/boundaryinfo/types.hh>
#include <dune/xt/grid/filters/intersection.hh>
#include <dune/xt/grid/type_traits.hh>
#include <dune/xt/la/container/eigen.hh>

#include <dune/gdt/discretefunction/default.hh>
#include <dune/gdt/functionals/vector-based.hh>
#include <dune/gdt/local/bilinear-forms/integrals.hh>
#include <dune/gdt/local/functionals/integrals.hh>
#include <dune/gdt/local/integrands/product.hh>
#include <dune/gdt/operators/bilinear-form.hh>
#include <dune/gdt/operators/matrix.hh>
#include <dune/gdt/spaces/l2/discontinuous-lagrange.hh>

#include <dune/ldg-ns/local/integrands/convection.hh>
#include <dune/ldg-ns/local/integrands/ldg.hh>
#include <dune/ldg-ns/tools/block-inverse.hh>

namespace Dune {
namespace GDT {
namespace NavierStokes {


/// \brief Discretization parameters, see doc/design.md, Sec. 3.
template <size_t d>
struct LdgOptions
{
  int velocity_order = 2; //!< k
  int pressure_order = 1; //!< k - 1 (inf-sup stable) or k (requires pressure_penalty > 0)
  FieldVector<double, d> beta = FieldVector<double, d>(0.); //!< LDG switch, beta_F = beta . n
  double eta = 0.; //!< LDG penalty C11 = nu * eta / h_F (any eta > 0 is stable for LDG); eta <= 0 means 4 k^2
  double upwind = 1.; //!< 0: central convective flux, 1: full upwinding
  double pressure_penalty = 0.; //!< gamma of the pressure jump stabilization gamma h_F [[p]][[q]]
  bool use_tbb = false; //!< thread-parallel grid walks
};


template <class GV>
class LdgNavierStokesOperator
{
  static_assert(XT::Grid::is_view<GV>::value);
  using ThisType = LdgNavierStokesOperator;

public:
  static constexpr size_t d = GV::dimension;
  using GridViewType = GV;
  using E = XT::Grid::extract_entity_t<GV>;
  using I = XT::Grid::extract_intersection_t<GV>;
  using MatrixType = XT::LA::EigenRowMajorSparseMatrix<double>;
  using VectorType = XT::LA::EigenDenseVector<double>;
  using SparseType = ::Eigen::SparseMatrix<double, ::Eigen::RowMajor>;
  using DenseVectorType = ::Eigen::VectorXd;
  using ScalarSpaceType = DiscontinuousLagrangeSpace<GV, 1>;
  using VelocitySpaceType = DiscontinuousLagrangeSpace<GV, d>;
  using PressureSpaceType = DiscontinuousLagrangeSpace<GV, 1>;
  using BoundaryInfoType = XT::Grid::BoundaryInfo<I>;
  using VectorGridFunctionType = XT::Functions::GridFunction<E, d>;
  using OptionsType = LdgOptions<d>;

  /// \note boundary_info is stored by reference and has to outlive this operator.
  LdgNavierStokesOperator(const GV& grid_view,
                          const BoundaryInfoType& boundary_info,
                          const double viscosity,
                          const OptionsType& options = {})
    : grid_view_(grid_view)
    , boundary_info_(boundary_info)
    , nu_(viscosity)
    , options_(options)
    , scalar_space_(grid_view_, options_.velocity_order)
    , velocity_space_(grid_view_, options_.velocity_order, /*dimwise_global_mapping=*/true)
    , pressure_space_(grid_view_, options_.pressure_order)
    , logger_(XT::Common::TimedLogger().get("NavierStokes::LdgOperator"))
  {
    DUNE_THROW_IF(velocity_space_.mapper().size() != d * scalar_space_.mapper().size(),
                  Exceptions::operator_error,
                  "velocity and scalar spaces do not match!");
    DUNE_THROW_IF(options_.pressure_order == options_.velocity_order && !(options_.pressure_penalty > 0),
                  Exceptions::operator_error,
                  "equal-order pairs require pressure_penalty > 0!");
    assemble_linear_parts();
  }

  LdgNavierStokesOperator(const ThisType&) = delete;
  LdgNavierStokesOperator(ThisType&&) = delete;

  /// \name Spaces and sizes
  /// \{

  const GV& grid_view() const
  {
    return grid_view_;
  }

  const BoundaryInfoType& boundary_info() const
  {
    return boundary_info_;
  }

  double viscosity() const
  {
    return nu_;
  }

  /// \brief The LDG penalty parameter actually used (options().eta, or 4 k^2 if that is <= 0).
  double effective_eta() const
  {
    return options_.eta > 0 ? options_.eta : 4. * options_.velocity_order * options_.velocity_order;
  }

  const OptionsType& options() const
  {
    return options_;
  }

  const ScalarSpaceType& scalar_space() const
  {
    return scalar_space_;
  }

  const VelocitySpaceType& velocity_space() const
  {
    return velocity_space_;
  }

  const PressureSpaceType& pressure_space() const
  {
    return pressure_space_;
  }

  size_t num_scalar_dofs() const
  {
    return scalar_space_.mapper().size();
  }

  size_t num_velocity_dofs() const
  {
    return d * num_scalar_dofs();
  }

  size_t num_pressure_dofs() const
  {
    return pressure_space_.mapper().size();
  }

  /// \}
  /// \name Time-independent blocks (assembled once in the constructor)
  /// \{

  const SparseType& mass() const
  {
    return mass_;
  }

  const SparseType& inverse_mass() const
  {
    return inverse_mass_;
  }

  const SparseType& gradient(const size_t jj) const
  {
    return gradient_[jj];
  }

  const SparseType& viscous() const
  {
    return viscous_;
  }

  const SparseType& divergence(const size_t jj) const
  {
    return divergence_[jj];
  }

  const std::array<SparseType, d>& divergences() const
  {
    return divergence_;
  }

  const SparseType& pressure_stabilization() const
  {
    return pressure_stabilization_;
  }

  /// \brief q_k = int_Omega psi_k, used for the zero-mean pressure constraint.
  const DenseVectorType& pressure_basis_integrals() const
  {
    return pressure_basis_integrals_;
  }

  /// \}
  /// \name Data-dependent vectors
  /// \{

  /**
   * \brief Right hand side F_i(t) of the momentum equation without the convective Dirichlet part:
   *        (f_i, v) + (nu eta/h g_i, v)_{Gamma_D} - nu sum_j G_j^T Minv b_ij(g).
   */
  DenseVectorType momentum_rhs(const VectorGridFunctionType& force,
                               const VectorGridFunctionType& dirichlet,
                               const XT::Common::Parameter& param = {}) const
  {
    const size_t N = num_scalar_dofs();
    DenseVectorType result = DenseVectorType::Zero(d * N);
    for (size_t ii = 0; ii < d; ++ii) {
      auto functional = make_vector_functional<VectorType>(scalar_space_);
      functional.append(LocalElementIntegralFunctional<E>(LocalIntegrands::ComponentSource<E>(force, ii)), param);
      functional.append(LocalIntersectionIntegralFunctional<I>(
                            LocalIntegrands::BoundaryPenaltyData<I>(dirichlet, ii, penalty_coefficient())),
                        param,
                        dirichlet_filter());
      functional.assemble(options_.use_tbb);
      result.segment(ii * N, N) = functional.vector().backend();
      for (size_t jj = 0; jj < d; ++jj) {
        const DenseVectorType b_ij = sigma_dirichlet_data(dirichlet, ii, jj, param);
        result.segment(ii * N, N) -= nu_ * (gradient_[jj].transpose() * (inverse_mass_ * b_ij));
      }
    }
    return result;
  } // ... momentum_rhs(...)

  /// \brief Right hand side g_B(t)_k = int_{Gamma_D} psi_k g . n of the continuity equation.
  DenseVectorType continuity_rhs(const VectorGridFunctionType& dirichlet, const XT::Common::Parameter& param = {}) const
  {
    auto functional = make_vector_functional<VectorType>(pressure_space_);
    for (size_t jj = 0; jj < d; ++jj)
      functional.append(
          LocalIntersectionIntegralFunctional<I>(LocalIntegrands::BoundaryDataNormal<I>(dirichlet, jj, jj, 1.)),
          param,
          dirichlet_filter());
    functional.assemble(options_.use_tbb);
    return functional.vector().backend();
  }

  /// \brief The Oseen convection matrix T(w) (scalar, N x N) for the linearization velocity w (d*N DoFs).
  SparseType convection(const DenseVectorType& w_dofs) const
  {
    const auto w = make_velocity_function(w_dofs);
    const VectorGridFunctionType w_func(w);
    auto form = make_bilinear_form(grid_view_);
    form += LocalElementIntegralBilinearForm<E>(LocalIntegrands::ConvectionElement<E>(w_func));
    form += {LocalCouplingIntersectionIntegralBilinearForm<I>(
                 LocalIntegrands::ConvectionInnerFace<I>(w_func, options_.upwind)),
             XT::Grid::ApplyOn::InnerIntersectionsOnce<GV>()};
    form += {LocalIntersectionIntegralBilinearForm<I>(LocalIntegrands::ConvectionBoundary<I>(w_func)),
             XT::Grid::ApplyOn::BoundaryIntersections<GV>()};
    return assemble_scalar_matrix(scalar_space_, scalar_space_, form);
  }

  /// \brief Convective Dirichlet data -(w.n)^- g_i on Gamma_D (d*N vector, belongs to the right hand side).
  DenseVectorType convection_rhs(const DenseVectorType& w_dofs,
                                 const VectorGridFunctionType& dirichlet,
                                 const XT::Common::Parameter& param = {}) const
  {
    const size_t N = num_scalar_dofs();
    const auto w = make_velocity_function(w_dofs);
    const VectorGridFunctionType w_func(w);
    DenseVectorType result(d * N);
    for (size_t ii = 0; ii < d; ++ii) {
      auto functional = make_vector_functional<VectorType>(scalar_space_);
      functional.append(
          LocalIntersectionIntegralFunctional<I>(LocalIntegrands::ConvectionBoundaryData<I>(w_func, dirichlet, ii)),
          param,
          dirichlet_filter());
      functional.assemble(options_.use_tbb);
      result.segment(ii * N, N) = functional.vector().backend();
    }
    return result;
  }

  /**
   * \brief Momentum residual without time derivative and pressure (the "N(t, u)" of doc/design.md, Sec. 4):
   *        R_i = (A_visc + T(u)) u_i - F_i(t) - F^conv_i(t; u).
   */
  DenseVectorType spatial_residual(const DenseVectorType& u,
                                   const VectorGridFunctionType& force,
                                   const VectorGridFunctionType& dirichlet,
                                   const XT::Common::Parameter& param = {}) const
  {
    const size_t N = num_scalar_dofs();
    const SparseType T = convection(u);
    DenseVectorType result = -momentum_rhs(force, dirichlet, param) - convection_rhs(u, dirichlet, param);
    for (size_t ii = 0; ii < d; ++ii)
      result.segment(ii * N, N) += viscous_ * u.segment(ii * N, N) + T * u.segment(ii * N, N);
    return result;
  }

  /// \brief The LDG gradient sigma_ij = Minv (G_j u_i + b_ij), returned as d*d scalar DoF vectors (row-major in ij).
  std::vector<DenseVectorType> velocity_gradient(const DenseVectorType& u,
                                                 const VectorGridFunctionType& dirichlet,
                                                 const XT::Common::Parameter& param = {}) const
  {
    const size_t N = num_scalar_dofs();
    std::vector<DenseVectorType> sigma(d * d);
    for (size_t ii = 0; ii < d; ++ii)
      for (size_t jj = 0; jj < d; ++jj)
        sigma[ii * d + jj] =
            inverse_mass_ * (gradient_[jj] * u.segment(ii * N, N) + sigma_dirichlet_data(dirichlet, ii, jj, param));
    return sigma;
  }

  /// \brief L2 projection of a vector-valued function onto the velocity space (d*N DoFs).
  DenseVectorType l2_projection(const VectorGridFunctionType& function, const XT::Common::Parameter& param = {}) const
  {
    const size_t N = num_scalar_dofs();
    DenseVectorType result(d * N);
    for (size_t ii = 0; ii < d; ++ii) {
      auto functional = make_vector_functional<VectorType>(scalar_space_);
      functional.append(LocalElementIntegralFunctional<E>(LocalIntegrands::ComponentSource<E>(function, ii)), param);
      functional.assemble(options_.use_tbb);
      result.segment(ii * N, N) = inverse_mass_ * functional.vector().backend();
    }
    return result;
  }

  /// \}
  /// \name Discrete functions (views on DoF vectors)
  /// \{

  auto make_velocity_function(const DenseVectorType& dofs, const std::string& name = "velocity") const
  {
    return make_discrete_function(velocity_space_, VectorType(dofs), name);
  }

  auto make_pressure_function(const DenseVectorType& dofs, const std::string& name = "pressure") const
  {
    return make_discrete_function(pressure_space_, VectorType(dofs), name);
  }

  auto make_scalar_function(const DenseVectorType& dofs, const std::string& name = "scalar") const
  {
    return make_discrete_function(scalar_space_, VectorType(dofs), name);
  }

  /// \}

private:
  std::function<double(double)> penalty_coefficient() const
  {
    const double c = nu_ * effective_eta();
    return [c](const double h) { return c / h; };
  }

  XT::Grid::ApplyOn::CustomBoundaryIntersections<GV> dirichlet_filter() const
  {
    return XT::Grid::ApplyOn::CustomBoundaryIntersections<GV>(boundary_info_, new XT::Grid::DirichletBoundary());
  }

  /// \brief b_ij(g)_k = int_{Gamma_D} g_i n_j phi_k.
  DenseVectorType sigma_dirichlet_data(const VectorGridFunctionType& dirichlet,
                                       const size_t ii,
                                       const size_t jj,
                                       const XT::Common::Parameter& param) const
  {
    auto functional = make_vector_functional<VectorType>(scalar_space_);
    functional.append(
        LocalIntersectionIntegralFunctional<I>(LocalIntegrands::BoundaryDataNormal<I>(dirichlet, ii, jj, 1.)),
        param,
        dirichlet_filter());
    functional.assemble(options_.use_tbb);
    return functional.vector().backend();
  }

  template <class TestSpace, class AnsatzSpace, class FormType>
  SparseType assemble_scalar_matrix(const TestSpace& test_space,
                                    const AnsatzSpace& ansatz_space,
                                    const FormType& form,
                                    const XT::Common::Parameter& param = {}) const
  {
    auto op = make_matrix_operator<MatrixType>(grid_view_, ansatz_space, test_space, Stencil::element_and_intersection);
    op.append(form, param);
    op.assemble(options_.use_tbb);
    return op.matrix().backend();
  }

  void assemble_linear_parts()
  {
    logger_.info() << "assembling LDG blocks (N = " << num_scalar_dofs() << ", P = " << num_pressure_dofs() << ")"
                   << std::endl;
    // mass matrix and its element-block-diagonal inverse
    {
      auto form = make_bilinear_form(grid_view_);
      form += LocalElementIntegralBilinearForm<E>(LocalProductIntegrand<E>());
      mass_ = assemble_scalar_matrix(scalar_space_, scalar_space_, form);
      inverse_mass_ = element_block_inverse(scalar_space_, mass_);
    }
    // LDG gradients G_j and DG divergences B_j = -G_j(beta = 0)
    for (size_t jj = 0; jj < d; ++jj) {
      auto grad = make_bilinear_form(grid_view_);
      grad += LocalElementIntegralBilinearForm<E>(LocalIntegrands::DirectionalDerivative<E>(jj, 1.));
      grad += {LocalCouplingIntersectionIntegralBilinearForm<I>(
                   LocalIntegrands::WeightedAverageJumpNormal<I>(jj, -1., options_.beta)),
               XT::Grid::ApplyOn::InnerIntersectionsOnce<GV>()};
      grad +=
          {LocalIntersectionIntegralBilinearForm<I>(LocalIntegrands::BoundaryNormal<I>(jj, -1.)), dirichlet_filter()};
      gradient_[jj] = assemble_scalar_matrix(scalar_space_, scalar_space_, grad);
      auto div = make_bilinear_form(grid_view_);
      div += LocalElementIntegralBilinearForm<E>(LocalIntegrands::DirectionalDerivative<E>(jj, -1.));
      div += {LocalCouplingIntersectionIntegralBilinearForm<I>(LocalIntegrands::WeightedAverageJumpNormal<I>(jj, 1.)),
              XT::Grid::ApplyOn::InnerIntersectionsOnce<GV>()};
      div += {LocalIntersectionIntegralBilinearForm<I>(LocalIntegrands::BoundaryNormal<I>(jj, 1.)), dirichlet_filter()};
      divergence_[jj] = assemble_scalar_matrix(pressure_space_, scalar_space_, div);
    }
    // viscous block A_visc = nu sum_j G_j^T Minv G_j + J
    {
      auto penalty = make_bilinear_form(grid_view_);
      penalty += {
          LocalCouplingIntersectionIntegralBilinearForm<I>(LocalIntegrands::InnerJumpPenalty<I>(penalty_coefficient())),
          XT::Grid::ApplyOn::InnerIntersectionsOnce<GV>()};
      penalty += {LocalIntersectionIntegralBilinearForm<I>(LocalIntegrands::BoundaryPenalty<I>(penalty_coefficient())),
                  dirichlet_filter()};
      viscous_ = assemble_scalar_matrix(scalar_space_, scalar_space_, penalty);
      for (size_t jj = 0; jj < d; ++jj) {
        const SparseType Gt = gradient_[jj].transpose();
        const SparseType MinvG = inverse_mass_ * gradient_[jj];
        viscous_ += nu_ * SparseType(Gt * MinvG);
      }
      viscous_.makeCompressed();
    }
    // optional pressure stabilization
    pressure_stabilization_.resize(num_pressure_dofs(), num_pressure_dofs());
    if (options_.pressure_penalty > 0) {
      const double gamma = options_.pressure_penalty;
      auto stab = make_bilinear_form(grid_view_);
      stab += {LocalCouplingIntersectionIntegralBilinearForm<I>(
                   LocalIntegrands::InnerJumpPenalty<I>([gamma](const double h) { return gamma * h; })),
               XT::Grid::ApplyOn::InnerIntersectionsOnce<GV>()};
      pressure_stabilization_ = assemble_scalar_matrix(pressure_space_, pressure_space_, stab);
    }
    // pressure basis integrals
    {
      auto functional = make_vector_functional<VectorType>(pressure_space_);
      functional.append(LocalElementIntegralFunctional<E>(
          LocalProductIntegrand<E>().with_ansatz(XT::Functions::ConstantGridFunction<E>(1.))));
      functional.assemble(options_.use_tbb);
      pressure_basis_integrals_ = functional.vector().backend();
    }
  } // ... assemble_linear_parts(...)

  const GV grid_view_;
  const BoundaryInfoType& boundary_info_;
  const double nu_;
  const OptionsType options_;
  const ScalarSpaceType scalar_space_;
  const VelocitySpaceType velocity_space_;
  const PressureSpaceType pressure_space_;
  XT::Common::TimedLogManager logger_;
  SparseType mass_;
  SparseType inverse_mass_;
  std::array<SparseType, d> gradient_;
  std::array<SparseType, d> divergence_;
  SparseType viscous_;
  SparseType pressure_stabilization_;
  DenseVectorType pressure_basis_integrals_;
}; // class LdgNavierStokesOperator


} // namespace NavierStokes
} // namespace GDT
} // namespace Dune

#endif // DUNE_LDG_NS_OPERATORS_LDG_NAVIER_STOKES_HH
