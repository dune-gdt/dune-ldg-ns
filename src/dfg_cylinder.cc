// This file is part of the dune-ldg-ns project:
//   https://github.com/dune-gdt/dune-ldg-ns
// Copyright 2026 dune-ldg-ns developers and contributors. All rights reserved.
// License: Dual licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
//      or  GPL-2.0+ (http://opensource.org/licenses/gpl-license)
//          with "runtime exception" (http://www.dune-project.org/license.html)
// Authors:
//   René Fritze (2026)

/**
 * DFG benchmarks 2D-2 / 2D-3 (flow around a cylinder) with LDG in space and FS-theta in time.
 *
 * Usage: ldg_ns_dfg_cylinder <config.ini> [-key value ...]
 * Writes one CSV line per macro time step: t, c_D, c_L, Delta p (and Picard iterations).
 */
#include "config.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/parametertree.hh>
#include <dune/common/parametertreeparser.hh>
#include <dune/alugrid/grid.hh>
#include <dune/grid/io/file/gmshreader.hh>

#include <dune/xt/common/timedlogging.hh>

#include <dune/ldg-ns/functionals/benchmark-quantities.hh>
#include <dune/ldg-ns/operators/ldg-navier-stokes.hh>
#include <dune/ldg-ns/testcases/dfg-cylinder.hh>
#include <dune/ldg-ns/timestepping/fractional-step-theta.hh>

using namespace Dune;
using namespace Dune::GDT::NavierStokes;

using G = Dune::ALUGrid<2, 2, Dune::simplex, Dune::conforming>;
using GV = typename G::LeafGridView;


int main(int argc, char** argv)
{
  try {
    MPIHelper::instance(argc, argv);
    if (argc < 2) {
      std::cerr << "usage: " << argv[0] << " <config.ini> [-key value ...]" << std::endl;
      return EXIT_FAILURE;
    }
    ParameterTree config;
    ParameterTreeParser::readINITree(argv[1], config);
    ParameterTreeParser::readOptions(argc - 1, argv + 1, config);
    XT::Common::TimedLogger().create(config.get("logging.info", 0), config.get("logging.debug", -1));
    auto logger = XT::Common::TimedLogger().get("dfg_cylinder");

    // problem and grid
    using ProblemType = DfgCylinderProblem<GV>;
    const auto variant_name = config.get<std::string>("problem.variant", "2d-2");
    const ProblemType problem(variant_name == "2d-3" ? ProblemType::Variant::dfg_2d_3 : ProblemType::Variant::dfg_2d_2);
    GridFactory<G> factory;
    GmshReader<G>::read(factory, config.get<std::string>("grid.filename"), /*verbose=*/false, /*boundary=*/false);
    std::unique_ptr<G> grid(factory.createGrid());
    grid->globalRefine(config.get("grid.refinements", 0));
    const auto grid_view = grid->leafGridView();
    logger.info() << "grid: " << grid_view.size(0) << " triangles" << std::endl;

    // discretization
    LdgOptions<2> options;
    options.velocity_order = config.get("discretization.velocity_order", 2);
    options.pressure_order = config.get("discretization.pressure_order", options.velocity_order - 1);
    options.eta = config.get("discretization.eta", 4. * options.velocity_order * options.velocity_order);
    options.upwind = config.get("discretization.upwind", 1.);
    options.pressure_penalty = config.get("discretization.pressure_penalty", 0.);
    options.use_tbb = config.get("discretization.use_tbb", false);
    const LdgNavierStokesOperator<GV> op(grid_view, problem.boundary_info(), problem.viscosity(), options);
    logger.info() << "DoFs: velocity " << op.num_velocity_dofs() << ", pressure " << op.num_pressure_dofs()
                  << std::endl;

    // time stepping
    TimeStepperOptions stepper_options;
    stepper_options.scheme = config.get<std::string>("timestepping.scheme", "fs-theta");
    stepper_options.picard_tolerance = config.get("timestepping.picard_tolerance", 1e-8);
    stepper_options.max_picard_iterations = config.get("timestepping.max_picard_iterations", 30);
    stepper_options.linear_solver.type = config.get<std::string>("timestepping.linear_solver", "sparselu");
    NavierStokesTimeStepper<GV> stepper(op, problem.force_function(), problem.dirichlet_function(), stepper_options);
    const double K = config.get("timestepping.dt", 0.01);
    const double t_end = config.get("timestepping.t_end", problem.end_time());
    stepper.initialize(op.l2_projection(ProblemType::to_grid_function(problem.initial_velocity(), "u0"),
                                        XT::Common::Parameter("t", 0.)),
                       0.);
    const bool recover_pressure = config.get<std::string>("output.pressure", "recovered") == "recovered";

    // output
    std::ofstream csv(config.get<std::string>("output.csv", problem.name() + ".csv"));
    csv << "t,c_D,c_L,delta_p,picard" << std::endl;
    const auto dirichlet = problem.dirichlet_function();
    const double U_mean = 1.;
    const double coefficient = 2. / (U_mean * U_mean * ProblemType::D);
    const FieldVector<double, 2> a1{0.15, 0.2};
    const FieldVector<double, 2> a2{0.25, 0.2};
    const size_t vtk_every = config.get("output.vtk_every", 0);
    const auto num_steps = static_cast<size_t>(std::llround(t_end / K));
    for (size_t n = 1; n <= num_steps; ++n) {
      stepper.step(K);
      const double t = stepper.time();
      const auto p = recover_pressure ? stepper.recover_pressure() : stepper.pressure_multiplier();
      const auto force = body_force(
          op, stepper.velocity(), p, dirichlet, t, [](const auto& x) { return ProblemType::on_cylinder(x, 1e-3); });
      const auto p_h = op.make_pressure_function(p);
      const auto p1 = point_value(grid_view, p_h, a1);
      const auto p2 = point_value(grid_view, p_h, a2);
      const double delta_p = (p1 && p2) ? (*p1)[0] - (*p2)[0] : std::nan("");
      csv << std::setprecision(10) << t << "," << coefficient * force[0] << "," << coefficient * force[1] << ","
          << delta_p << "," << stepper.last_picard_iterations() << std::endl;
      logger.info() << "t = " << t << ": c_D = " << coefficient * force[0] << ", c_L = " << coefficient * force[1]
                    << ", dp = " << delta_p << " (" << stepper.last_picard_iterations() << " Picard its)" << std::endl;
      if (vtk_every > 0 && n % vtk_every == 0) {
        const auto u_h = op.make_velocity_function(stepper.velocity());
        visualize(u_h, problem.name() + "_velocity_" + std::to_string(n));
        visualize(p_h, problem.name() + "_pressure_" + std::to_string(n));
      }
    }
  } catch (Dune::Exception& e) {
    std::cerr << "\nDUNE reported error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  } catch (std::exception& e) {
    std::cerr << "\nstl reported error: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
} // ... main(...)
