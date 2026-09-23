// This file is part of the dune-ldg-ns project:
//   https://github.com/dune-gdt/dune-ldg-ns
// Copyright 2026 dune-ldg-ns developers and contributors. All rights reserved.
// License: Dual licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
//      or  GPL-2.0+ (http://opensource.org/licenses/gpl-license)
//          with "runtime exception" (http://www.dune-project.org/license.html)
// Authors:
//   René Fritze (2026)

/**
 * Temporal convergence of the monolithic FS-theta scheme for the decaying Taylor--Green vortex (2D) and a 3D smoke
 * test (z-invariant Taylor--Green vortex).
 *
 * The temporal error is measured against a reference solution with K_ref = K_min / 4 on the same mesh, so that the
 * spatial error cancels; the error against the exact solution is reported as well.
 */
#include "config.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <dune/common/parallel/mpihelper.hh>

#include <dune/xt/common/timedlogging.hh>
#include <dune/xt/grid/gridprovider/cube.hh>
#include <dune/xt/grid/grids.hh>

#include <dune/ldg-ns/operators/ldg-navier-stokes.hh>
#include <dune/ldg-ns/testcases/taylor-green.hh>
#include <dune/ldg-ns/timestepping/fractional-step-theta.hh>
#include <dune/ldg-ns/tools/errors.hh>

using namespace Dune;
using namespace Dune::GDT;
using namespace Dune::GDT::NavierStokes;


template <class G>
struct TaylorGreenRun
{
  using GV = typename G::LeafGridView;
  static constexpr size_t d = G::dimension;
  using OperatorType = LdgNavierStokesOperator<GV>;
  using DenseVectorType = typename OperatorType::DenseVectorType;

  struct Result
  {
    DenseVectorType velocity;
    DenseVectorType pressure_multiplier;
    DenseVectorType pressure_recovered;
    double velocity_error; //!< relative, against the exact solution
    double pressure_error_multiplier; //!< against the exact solution
    double pressure_error_recovered; //!< against the exact solution
    double divergence_residual; //!< || B u - g_B ||_2
    size_t max_picard_iterations;
  };

  TaylorGreenRun(const int k, const unsigned int n, const double viscosity)
    : problem_(viscosity)
    , grid_(XT::Grid::make_cube_grid<G>(
          TaylorGreenProblem<GV>::lower_left(), TaylorGreenProblem<GV>::upper_right(), filled_array(n)))
    , grid_view_(grid_.leaf_view())
    , op_(grid_view_, problem_.boundary_info(), problem_.viscosity(), make_options(k))
  {
  }

  static std::array<unsigned int, d> filled_array(const unsigned int n)
  {
    std::array<unsigned int, d> result;
    result.fill(n);
    return result;
  }

  static LdgOptions<d> make_options(const int k)
  {
    LdgOptions<d> options;
    options.velocity_order = k;
    options.pressure_order = k - 1;
    options.eta = 4. * k * k;
    return options;
  }

  Result run(const std::string& scheme, const double K, const double T)
  {
    TimeStepperOptions stepper_options;
    stepper_options.scheme = scheme;
    stepper_options.linear_solver.mean_pressure_constraint = true;
    stepper_options.picard_tolerance = 1e-11;
    NavierStokesTimeStepper<GV> stepper(op_, problem_.force_function(), problem_.dirichlet_function(), stepper_options);
    const auto u0 = TaylorGreenProblem<GV>::to_grid_function(problem_.initial_velocity(), "u0");
    stepper.project_and_initialize(op_.l2_projection(u0, XT::Common::Parameter("t", 0.)), 0.);
    const auto num_steps = static_cast<size_t>(std::llround(T / K));
    size_t max_picard = 0;
    for (size_t n = 0; n < num_steps; ++n) {
      stepper.step(K);
      max_picard = std::max(max_picard, stepper.last_picard_iterations());
    }
    Result result;
    result.velocity = stepper.velocity();
    result.pressure_multiplier = stepper.pressure_multiplier();
    result.pressure_recovered = stepper.recover_pressure();
    result.max_picard_iterations = max_picard;
    const double t = stepper.time();
    const auto exact_u = problem_.exact_velocity();
    const auto exact_p = problem_.exact_pressure();
    const int order = 2 * op_.options().velocity_order + 4;
    const auto e_u = l2_compare(
        grid_view_, op_.make_velocity_function(result.velocity), [&](const auto& x) { return exact_u(x, t); }, order);
    result.velocity_error = e_u.error / e_u.norm;
    const auto p_exact = [&](const auto& x) { return FieldVector<double, 1>(exact_p(x, t)); };
    result.pressure_error_multiplier =
        l2_compare(grid_view_, op_.make_pressure_function(result.pressure_multiplier), p_exact, order)
            .error_up_to_constant();
    result.pressure_error_recovered =
        l2_compare(grid_view_, op_.make_pressure_function(result.pressure_recovered), p_exact, order)
            .error_up_to_constant();
    DenseVectorType Bu = -op_.continuity_rhs(problem_.dirichlet_function(), XT::Common::Parameter("t", t));
    const size_t N = op_.num_scalar_dofs();
    for (size_t jj = 0; jj < d; ++jj)
      Bu += op_.divergence(jj) * result.velocity.segment(jj * N, N);
    result.divergence_residual = Bu.norm();
    return result;
  } // ... run(...)

  /// \brief L2 norm of the difference of two velocity (or pressure) DoF vectors.
  double velocity_difference(const DenseVectorType& a, const DenseVectorType& b) const
  {
    const DenseVectorType diff = a - b;
    return l2_compare(
               grid_view_,
               op_.make_velocity_function(diff),
               [](const auto& /*x*/) { return FieldVector<double, d>(0.); },
               2 * op_.options().velocity_order)
        .error;
  }

  double pressure_difference(const DenseVectorType& a, const DenseVectorType& b) const
  {
    const DenseVectorType diff = a - b;
    return l2_compare(
               grid_view_,
               op_.make_pressure_function(diff),
               [](const auto& /*x*/) { return FieldVector<double, 1>(0.); },
               2 * op_.options().velocity_order)
        .error_up_to_constant();
  }

  TaylorGreenProblem<GV> problem_;
  XT::Grid::GridProvider<G> grid_;
  GV grid_view_;
  OperatorType op_;
}; // struct TaylorGreenRun


std::vector<double> env_list(const char* name, std::vector<double> defaults)
{
  const char* value = std::getenv(name);
  if (!value)
    return defaults;
  std::vector<double> result;
  std::stringstream ss(value);
  for (std::string item; std::getline(ss, item, ',');)
    result.push_back(std::stod(item));
  return result;
}


std::vector<std::string> env_strings(const char* name, std::vector<std::string> defaults)
{
  const char* value = std::getenv(name);
  if (!value)
    return defaults;
  std::vector<std::string> result;
  std::stringstream ss(value);
  for (std::string item; std::getline(ss, item, ',');)
    result.push_back(item);
  return result;
}


double eoc(const double e_coarse, const double e_fine)
{
  return std::log(e_coarse / e_fine) / std::log(2.);
}


TEST(TaylorGreen2d, temporal_convergence)
{
  using G = YASP_2D_EQUIDISTANT_OFFSET;
  TaylorGreenRun<G> tg(/*k=*/2, /*n=*/8, /*viscosity=*/0.1);
  const double T = 1.;
  const std::vector<double> time_steps = env_list("DUNE_LDG_NS_TG_TIME_STEPS", {0.2, 0.1, 0.05});
  // minimal EOCs of (velocity, recovered pressure); Crank--Nicolson is only reported (not strongly A-stable, the
  // stiff error components of the initial projection are not damped and the asymptotic regime starts late)
  std::map<std::string, std::vector<double>> expected_min_eoc = {
      {"fs-theta", {1.8, 1.5}}, {"crank-nicolson", {-100., -100.}}, {"backward-euler", {0.8, 0.8}}};
  for (const std::string scheme :
       env_strings("DUNE_LDG_NS_TG_SCHEMES", {"fs-theta", "crank-nicolson", "backward-euler"})) {
    const auto reference = tg.run(scheme, time_steps.back() / 4., T);
    std::cout << "\nTaylor--Green 2D (nu = 0.1, Q2/Q1, 8x8), scheme " << scheme << ", T = " << T
              << " (reference: K = " << time_steps.back() / 4. << ")" << std::endl;
    std::cout << std::setw(8) << "K" << std::setw(12) << "|u-u_ref|" << std::setw(7) << "EOC" << std::setw(12)
              << "|p-p_ref|" << std::setw(7) << "EOC" << std::setw(12) << "|l-p_ref|" << std::setw(7) << "EOC"
              << std::setw(12) << "|u-u_ex|/|u|" << std::setw(12) << "|Bu-g|" << std::setw(7) << "Picard" << std::endl;
    std::vector<double> e_u, e_p, e_l;
    for (const double K : time_steps) {
      const auto result = tg.run(scheme, K, T);
      e_u.push_back(tg.velocity_difference(result.velocity, reference.velocity));
      e_p.push_back(tg.pressure_difference(result.pressure_recovered, reference.pressure_recovered));
      e_l.push_back(tg.pressure_difference(result.pressure_multiplier, reference.pressure_recovered));
      const size_t ii = e_u.size() - 1;
      std::cout << std::setw(8) << K << std::scientific << std::setprecision(3) << std::setw(12) << e_u[ii]
                << std::defaultfloat << std::setw(7) << std::setprecision(3)
                << (ii > 0 ? std::to_string(eoc(e_u[ii - 1], e_u[ii])).substr(0, 5) : "-") << std::scientific
                << std::setw(12) << e_p[ii] << std::defaultfloat << std::setw(7)
                << (ii > 0 ? std::to_string(eoc(e_p[ii - 1], e_p[ii])).substr(0, 5) : "-") << std::scientific
                << std::setw(12) << e_l[ii] << std::defaultfloat << std::setw(7)
                << (ii > 0 ? std::to_string(eoc(e_l[ii - 1], e_l[ii])).substr(0, 5) : "-") << std::scientific
                << std::setw(12) << result.velocity_error << std::setw(12) << result.divergence_residual
                << std::defaultfloat << std::setw(7) << result.max_picard_iterations << std::endl;
      EXPECT_LT(result.divergence_residual, 1e-9);
    }
    const size_t last = e_u.size() - 1;
    EXPECT_GT(eoc(e_u[last - 1], e_u[last]), expected_min_eoc[scheme][0]) << scheme;
    EXPECT_GT(eoc(e_p[last - 1], e_p[last]), expected_min_eoc[scheme][1]) << scheme;
  }
} // TEST(TaylorGreen2d, temporal_convergence)


TEST(TaylorGreen3d, smoke)
{
  using G = YASP_3D_EQUIDISTANT_OFFSET;
  TaylorGreenRun<G> tg(/*k=*/1, /*n=*/4, /*viscosity=*/0.1);
  const auto result = tg.run("fs-theta", 0.1, 0.3);
  std::cout << "\nTaylor--Green 3D (z-invariant, Q1/Q0, 4^3), FS-theta, K = 0.1, T = 0.3: |u-u_ex|/|u| = "
            << result.velocity_error << ", |p-p_ex| = " << result.pressure_error_recovered
            << ", |Bu-g| = " << result.divergence_residual << ", Picard <= " << result.max_picard_iterations
            << std::endl;
  EXPECT_LT(result.velocity_error, 0.3); // Q1 with h = 0.5: this is a smoke test, not a convergence test
  EXPECT_LT(result.divergence_residual, 1e-9);
  EXPECT_LT(result.max_picard_iterations, 30u);
}


int main(int argc, char** argv)
{
  Dune::MPIHelper::instance(argc, argv);
  XT::Common::TimedLogger().create(/*max_info_level=*/-1, /*max_debug_level=*/-1);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
