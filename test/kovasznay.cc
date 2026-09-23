// This file is part of the dune-ldg-ns project:
//   https://github.com/dune-gdt/dune-ldg-ns
// Copyright 2026 dune-ldg-ns developers and contributors. All rights reserved.
// License: Dual licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
//      or  GPL-2.0+ (http://opensource.org/licenses/gpl-license)
//          with "runtime exception" (http://www.dune-project.org/license.html)
// Authors:
//   René Fritze (2026)

/**
 * Spatial convergence of the steady LDG Navier--Stokes discretization for the Kovasznay flow (Re = 40) on
 * structured quadrilateral grids, k = 1, 2 (Q_k / Q_{k-1}).
 */
#include "config.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include <dune/common/parallel/mpihelper.hh>

#include <dune/xt/common/timedlogging.hh>
#include <dune/xt/grid/gridprovider/cube.hh>
#include <dune/xt/grid/grids.hh>

#include <dune/ldg-ns/operators/ldg-navier-stokes.hh>
#include <dune/ldg-ns/testcases/kovasznay.hh>
#include <dune/ldg-ns/timestepping/fractional-step-theta.hh>
#include <dune/ldg-ns/tools/errors.hh>

using namespace Dune;
using namespace Dune::GDT;
using namespace Dune::GDT::NavierStokes;

using G = YASP_2D_EQUIDISTANT_OFFSET;
using GV = typename G::LeafGridView;


struct KovasznayResult
{
  double h;
  double velocity_error;
  double pressure_error;
  size_t picard_iterations;
};


KovasznayResult run_kovasznay(const int k, const unsigned int num_elements)
{
  using ProblemType = KovasznayProblem<GV>;
  const ProblemType problem(40.);
  auto grid = XT::Grid::make_cube_grid<G>(ProblemType::lower_left(),
                                          ProblemType::upper_right(),
                                          {num_elements, static_cast<unsigned int>(4 * num_elements / 3)});
  const auto grid_view = grid.leaf_view();
  LdgOptions<2> options;
  options.velocity_order = k;
  options.pressure_order = k - 1;
  options.eta = 4. * k * k;
  LdgNavierStokesOperator<GV> op(grid_view, problem.boundary_info(), problem.viscosity(), options);
  TimeStepperOptions stepper_options;
  stepper_options.linear_solver.mean_pressure_constraint = true;
  stepper_options.picard_tolerance = 1e-11;
  stepper_options.max_picard_iterations = 50;
  NavierStokesTimeStepper<GV> stepper(op, problem.force_function(), problem.dirichlet_function(), stepper_options);
  const size_t iterations = stepper.solve_steady(0.);
  const auto u_h = op.make_velocity_function(stepper.velocity());
  const auto p_h = op.make_pressure_function(stepper.pressure_multiplier());
  const auto exact_u = problem.exact_velocity();
  const auto exact_p = problem.exact_pressure();
  const auto e_u = l2_compare(grid_view, u_h, [&](const auto& x) { return exact_u(x, 0.); }, 2 * k + 4);
  const auto e_p =
      l2_compare(grid_view, p_h, [&](const auto& x) { return FieldVector<double, 1>(exact_p(x, 0.)); }, 2 * k + 4);
  return {1.5 / num_elements, e_u.error / e_u.norm, e_p.error_up_to_constant(), iterations};
} // ... run_kovasznay(...)


class KovasznayTest : public ::testing::TestWithParam<int>
{};


TEST_P(KovasznayTest, spatial_convergence)
{
  const int k = GetParam();
  std::vector<KovasznayResult> results;
  for (const unsigned int n : {6u, 12u, 24u})
    results.push_back(run_kovasznay(k, n));
  std::cout << "Kovasznay (Re = 40), Q" << k << "/Q" << k - 1 << " LDG:" << std::endl;
  std::cout << std::setw(10) << "h" << std::setw(14) << "|u-u_h|/|u|" << std::setw(8) << "EOC" << std::setw(14)
            << "|p-p_h|" << std::setw(8) << "EOC" << std::setw(8) << "Picard" << std::endl;
  for (size_t ii = 0; ii < results.size(); ++ii) {
    std::cout << std::setw(10) << std::setprecision(4) << results[ii].h << std::setw(14) << std::scientific
              << std::setprecision(3) << results[ii].velocity_error << std::defaultfloat;
    if (ii > 0)
      std::cout << std::setw(8) << std::setprecision(3)
                << std::log(results[ii - 1].velocity_error / results[ii].velocity_error) / std::log(2.);
    else
      std::cout << std::setw(8) << "-";
    std::cout << std::setw(14) << std::scientific << std::setprecision(3) << results[ii].pressure_error
              << std::defaultfloat;
    if (ii > 0)
      std::cout << std::setw(8) << std::setprecision(3)
                << std::log(results[ii - 1].pressure_error / results[ii].pressure_error) / std::log(2.);
    else
      std::cout << std::setw(8) << "-";
    std::cout << std::setw(8) << results[ii].picard_iterations << std::endl;
  }
  const size_t last = results.size() - 1;
  const double eoc_u = std::log(results[last - 1].velocity_error / results[last].velocity_error) / std::log(2.);
  const double eoc_p = std::log(results[last - 1].pressure_error / results[last].pressure_error) / std::log(2.);
  // expected: k + 1 for the velocity, k for the pressure (L2)
  EXPECT_GT(eoc_u, k + 1 - 0.3);
  EXPECT_GT(eoc_p, k - 0.3);
}


INSTANTIATE_TEST_SUITE_P(LDG, KovasznayTest, ::testing::Values(1, 2));


int main(int argc, char** argv)
{
  Dune::MPIHelper::instance(argc, argv);
  XT::Common::TimedLogger().create(/*max_info_level=*/0, /*max_debug_level=*/-1);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
