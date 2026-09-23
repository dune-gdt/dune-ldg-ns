// This file is part of the dune-ldg-ns project:
//   https://github.com/dune-gdt/dune-ldg-ns
// Copyright 2026 dune-ldg-ns developers and contributors. All rights reserved.
// License: Dual licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
//      or  GPL-2.0+ (http://opensource.org/licenses/gpl-license)
//          with "runtime exception" (http://www.dune-project.org/license.html)
// Authors:
//   René Fritze (2026)

/**
 * \file  fractional-step-theta.hh
 * \brief Monolithic fractional-step theta scheme (and one-step theta schemes) for the LDG Navier--Stokes system.
 *
 * Every macro step t_n -> t_n + K consists of substeps (tau, c_imp, c_exp) with c_imp + c_exp = tau:
 *
 *   M u^b + c_imp R(t_b, u^b) + tau B^T lambda^b = M u^a - c_exp R(t_a, u^a),     B u^b - C lambda^b = g_B(t_b),
 *
 * where R(t, u) = (A_visc + T(u)) u - F(t; u) is the spatial momentum residual (LdgNavierStokesOperator::
 * spatial_residual). For FS-theta (theta = 1 - sqrt(2)/2, theta' = 1 - 2 theta, alpha = theta'/(1 - theta),
 * beta = 1 - alpha):
 *
 *   substep 1:  (theta K,   alpha theta K,  beta theta K)
 *   substep 2:  (theta' K,  beta theta' K,  alpha theta' K)
 *   substep 3:  (theta K,   alpha theta K,  beta theta K)
 *
 * so that c_imp = alpha theta K = beta theta' K in all substeps (identical implicit operator). The data F(t) are
 * weighted like the operator (c_imp F(t_b) + c_exp F(t_a)), which keeps second order for time-dependent forcing.
 *
 * The multiplier lambda is only a first-order approximation of p(t_b); a second-order pressure is available at any
 * time via recover_pressure() (doc/design.md, Sec. 4.3).
 */
#ifndef DUNE_LDG_NS_TIMESTEPPING_FRACTIONAL_STEP_THETA_HH
#define DUNE_LDG_NS_TIMESTEPPING_FRACTIONAL_STEP_THETA_HH

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <dune/xt/common/parameter.hh>
#include <dune/xt/common/timedlogging.hh>

#include <dune/ldg-ns/operators/ldg-navier-stokes.hh>
#include <dune/ldg-ns/solvers/saddle-point.hh>

namespace Dune {
namespace GDT {
namespace NavierStokes {


/// \brief One substep of a (fractional step) theta scheme: length tau, implicit and explicit weights.
struct ThetaSubstep
{
  double tau;
  double c_imp;
  double c_exp;
};


/// \brief The parameters theta, theta', alpha, beta of the FS-theta scheme.
struct FractionalStepThetaCoefficients
{
  static double theta()
  {
    return 1. - std::sqrt(2.) / 2.;
  }

  static double theta_prime()
  {
    return 1. - 2. * theta();
  }

  static double alpha()
  {
    return theta_prime() / (1. - theta());
  }

  static double beta()
  {
    return 1. - alpha();
  }
};


inline std::vector<ThetaSubstep> fractional_step_theta_substeps(const double K)
{
  using C = FractionalStepThetaCoefficients;
  const ThetaSubstep outer{C::theta() * K, C::alpha() * C::theta() * K, C::beta() * C::theta() * K};
  const ThetaSubstep inner{C::theta_prime() * K, C::beta() * C::theta_prime() * K, C::alpha() * C::theta_prime() * K};
  return {outer, inner, outer};
}


inline std::vector<ThetaSubstep> one_step_theta_substeps(const double K, const double theta)
{
  return {{K, theta * K, (1. - theta) * K}};
}


struct TimeStepperOptions
{
  std::string scheme = "fs-theta"; //!< "fs-theta", "crank-nicolson", "backward-euler" or "theta"
  double theta = 0.5; //!< only used for scheme == "theta"
  size_t max_picard_iterations = 30;
  double picard_tolerance = 1e-10; //!< relative change of the velocity in the M-norm
  SaddlePointSolverOptions linear_solver = {};
};


template <class GV>
class NavierStokesTimeStepper
{
public:
  using OperatorType = LdgNavierStokesOperator<GV>;
  static constexpr size_t d = OperatorType::d;
  using SparseType = typename OperatorType::SparseType;
  using DenseVectorType = typename OperatorType::DenseVectorType;
  using VectorGridFunctionType = typename OperatorType::VectorGridFunctionType;

private:
  using ColMajorSparseType = ::Eigen::SparseMatrix<double, ::Eigen::ColMajor>;
  using PressureLuType = ::Eigen::SparseLU<ColMajorSparseType, ::Eigen::COLAMDOrdering<int>>;

public:
  NavierStokesTimeStepper(const OperatorType& op,
                          VectorGridFunctionType force,
                          VectorGridFunctionType dirichlet,
                          const TimeStepperOptions& options = {})
    : op_(op)
    , force_(force)
    , dirichlet_(dirichlet)
    , options_(options)
    , solver_(op_.divergences(), op_.pressure_stabilization(), op_.pressure_basis_integrals(), options_.linear_solver)
    , logger_(XT::Common::TimedLogger().get("NavierStokes::TimeStepper"))
    , time_(0.)
    , velocity_(DenseVectorType::Zero(op_.num_velocity_dofs()))
    , pressure_multiplier_(DenseVectorType::Zero(op_.num_pressure_dofs()))
  {
  }

  /// \name State
  /// \{

  double time() const
  {
    return time_;
  }

  const DenseVectorType& velocity() const
  {
    return velocity_;
  }

  /// \brief The Lagrange multiplier of the last substep (first-order approximation of the pressure).
  const DenseVectorType& pressure_multiplier() const
  {
    return pressure_multiplier_;
  }

  size_t last_picard_iterations() const
  {
    return last_picard_iterations_;
  }

  void initialize(const DenseVectorType& velocity, const double t0)
  {
    velocity_ = velocity;
    time_ = t0;
  }

  /**
   * \brief Sets u(t0) to the discretely divergence-free L2 projection of the given velocity DoFs:
   *        [M B^T; B -C] [u; lambda] = [M u_given; g_B(t0)].
   */
  void project_and_initialize(const DenseVectorType& velocity, const double t0)
  {
    const size_t N = op_.num_scalar_dofs();
    DenseVectorType rhs_u(d * N);
    for (size_t cc = 0; cc < d; ++cc)
      rhs_u.segment(cc * N, N) = op_.mass() * velocity.segment(cc * N, N);
    const DenseVectorType rhs_p = op_.continuity_rhs(dirichlet_, time_param(t0));
    solver_.solve(op_.mass(), 1., rhs_u, rhs_p, velocity_, pressure_multiplier_);
    time_ = t0;
  }

  /// \}
  /// \name Time stepping
  /// \{

  std::vector<ThetaSubstep> substeps(const double K) const
  {
    if (options_.scheme == "fs-theta")
      return fractional_step_theta_substeps(K);
    if (options_.scheme == "crank-nicolson")
      return one_step_theta_substeps(K, 0.5);
    if (options_.scheme == "backward-euler")
      return one_step_theta_substeps(K, 1.);
    if (options_.scheme == "theta")
      return one_step_theta_substeps(K, options_.theta);
    DUNE_THROW(Dune::NotImplemented, "Unknown time stepping scheme '" << options_.scheme << "'!");
  }

  /// \brief Advances the solution by one macro time step K.
  void step(const double K)
  {
    const double t_end = time_ + K;
    for (const auto& substep : substeps(K))
      solve_substep(substep);
    time_ = t_end; // avoid round-off drift of the substep sum
  }

  /**
   * \brief Solves M u^b + c_imp R(t_b, u^b) + tau B^T lambda = M u^a - c_exp R(t_a, u^a) by Picard iteration.
   * \return the number of Picard iterations
   */
  size_t solve_substep(const ThetaSubstep& substep)
  {
    const size_t N = op_.num_scalar_dofs();
    const double t_a = time_;
    const double t_b = time_ + substep.tau;
    // explicit part
    DenseVectorType explicit_part(d * N);
    for (size_t cc = 0; cc < d; ++cc)
      explicit_part.segment(cc * N, N) = op_.mass() * velocity_.segment(cc * N, N);
    if (substep.c_exp != 0.)
      explicit_part -= substep.c_exp * op_.spatial_residual(velocity_, force_, dirichlet_, time_param(t_a));
    // implicit data
    const DenseVectorType F_b = op_.momentum_rhs(force_, dirichlet_, time_param(t_b));
    const DenseVectorType g_b = op_.continuity_rhs(dirichlet_, time_param(t_b));
    // Picard iteration
    DenseVectorType w = velocity_;
    DenseVectorType u_new(d * N);
    size_t iteration = 0;
    for (; iteration < options_.max_picard_iterations; ++iteration) {
      const SparseType T = op_.convection(w);
      const SparseType velocity_block = op_.mass() + substep.c_imp * (op_.viscous() + T);
      const DenseVectorType rhs_u =
          explicit_part + substep.c_imp * (F_b + op_.convection_rhs(w, dirichlet_, time_param(t_b)));
      solver_.solve(velocity_block, substep.tau, rhs_u, g_b, u_new, pressure_multiplier_);
      const double change = mass_norm(u_new - w);
      const double norm = std::max(mass_norm(u_new), 1e-14);
      w = u_new;
      logger_.debug() << "  t = " << t_b << ", Picard iteration " << iteration + 1 << ": relative change "
                      << change / norm << std::endl;
      if (change <= options_.picard_tolerance * norm) {
        ++iteration;
        break;
      }
    }
    if (iteration == options_.max_picard_iterations)
      logger_.warn() << "Picard iteration did not converge in substep to t = " << t_b << "!" << std::endl;
    velocity_ = u_new;
    time_ = t_b;
    last_picard_iterations_ = iteration;
    return iteration;
  } // ... solve_substep(...)

  /// \}
  /// \name Steady problems
  /// \{

  /// \brief Solves the steady problem R(t, u) + B^T p = 0, B u - C p = g_B(t) by Picard iteration.
  size_t solve_steady(const double t = 0.)
  {
    const auto param = time_param(t);
    const DenseVectorType F = op_.momentum_rhs(force_, dirichlet_, param);
    const DenseVectorType g = op_.continuity_rhs(dirichlet_, param);
    DenseVectorType w = velocity_;
    DenseVectorType u_new(op_.num_velocity_dofs());
    size_t iteration = 0;
    for (; iteration < options_.max_picard_iterations; ++iteration) {
      const SparseType velocity_block = op_.viscous() + op_.convection(w);
      const DenseVectorType rhs_u = F + op_.convection_rhs(w, dirichlet_, param);
      solver_.solve(velocity_block, 1., rhs_u, g, u_new, pressure_multiplier_);
      const double change = mass_norm(u_new - w);
      const double norm = std::max(mass_norm(u_new), 1e-14);
      w = u_new;
      logger_.info() << "steady Picard iteration " << iteration + 1 << ": relative change " << change / norm
                     << std::endl;
      if (change <= options_.picard_tolerance * norm) {
        ++iteration;
        break;
      }
    }
    velocity_ = u_new;
    time_ = t;
    last_picard_iterations_ = iteration;
    return iteration;
  } // ... solve_steady(...)

  /// \}
  /// \name Pressure recovery
  /// \{

  /**
   * \brief Second-order pressure at the current time from S p = -B M^{-1} R(t, u) - d/dt g_B(t), S = B M^{-1} B^T.
   *
   * d/dt g_B is approximated by central differences with step fd_step. Requires C = 0 (inf-sup stable pairs).
   */
  DenseVectorType recover_pressure(const double fd_step = 1e-6) const
  {
    DUNE_THROW_IF(op_.options().pressure_penalty > 0,
                  Dune::NotImplemented,
                  "Pressure recovery is only implemented without pressure stabilization!");
    const size_t N = op_.num_scalar_dofs();
    const size_t P = op_.num_pressure_dofs();
    const DenseVectorType R = op_.spatial_residual(velocity_, force_, dirichlet_, time_param(time_));
    const DenseVectorType dg = (op_.continuity_rhs(dirichlet_, time_param(time_ + fd_step))
                                - op_.continuity_rhs(dirichlet_, time_param(time_ - fd_step)))
                               / (2. * fd_step);
    DenseVectorType rhs = -dg;
    for (size_t jj = 0; jj < d; ++jj)
      rhs -= op_.divergence(jj) * (op_.inverse_mass() * R.segment(jj * N, N));
    const bool constrain = options_.linear_solver.mean_pressure_constraint;
    const size_t size = constrain ? P + 1 : P;
    if (!pressure_lu_) {
      // S = sum_j B_j Minv B_j^T, bordered with the zero-mean constraint (S has the constants in its kernel for pure
      // Dirichlet problems); S is time-independent, so the factorization is computed once.
      SparseType S(P, P);
      for (size_t jj = 0; jj < d; ++jj) {
        const SparseType MinvBt = op_.inverse_mass() * SparseType(op_.divergence(jj).transpose());
        S += SparseType(op_.divergence(jj) * MinvBt);
      }
      std::vector<::Eigen::Triplet<double>> triplets;
      for (int row = 0; row < S.outerSize(); ++row)
        for (typename SparseType::InnerIterator it(S, row); it; ++it)
          triplets.emplace_back(it.row(), it.col(), it.value());
      if (constrain)
        for (size_t kk = 0; kk < P; ++kk) {
          triplets.emplace_back(kk, P, op_.pressure_basis_integrals()[kk]);
          triplets.emplace_back(P, kk, op_.pressure_basis_integrals()[kk]);
        }
      ColMajorSparseType bordered(size, size);
      bordered.setFromTriplets(triplets.begin(), triplets.end());
      pressure_lu_ = std::make_unique<PressureLuType>();
      pressure_lu_->compute(bordered);
      DUNE_THROW_IF(pressure_lu_->info() != ::Eigen::Success, Dune::MathError, "pressure recovery failed!");
    }
    DenseVectorType bordered_rhs = DenseVectorType::Zero(size);
    bordered_rhs.segment(0, P) = rhs;
    const DenseVectorType x = pressure_lu_->solve(bordered_rhs);
    return x.segment(0, P);
  } // ... recover_pressure(...)

  /// \}

  double mass_norm(const DenseVectorType& u) const
  {
    const size_t N = op_.num_scalar_dofs();
    double result = 0.;
    for (size_t cc = 0; cc < d; ++cc)
      result += u.segment(cc * N, N).dot(op_.mass() * u.segment(cc * N, N));
    return std::sqrt(result);
  }

  static XT::Common::Parameter time_param(const double t)
  {
    return XT::Common::Parameter("t", t);
  }

private:
  const OperatorType& op_;
  const VectorGridFunctionType force_;
  const VectorGridFunctionType dirichlet_;
  const TimeStepperOptions options_;
  SaddlePointSolver<d> solver_;
  XT::Common::TimedLogManager logger_;
  double time_;
  DenseVectorType velocity_;
  DenseVectorType pressure_multiplier_;
  size_t last_picard_iterations_ = 0;
  mutable std::unique_ptr<PressureLuType> pressure_lu_;
}; // class NavierStokesTimeStepper


} // namespace NavierStokes
} // namespace GDT
} // namespace Dune

#endif // DUNE_LDG_NS_TIMESTEPPING_FRACTIONAL_STEP_THETA_HH
