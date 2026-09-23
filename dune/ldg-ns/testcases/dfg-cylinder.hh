// This file is part of the dune-ldg-ns project:
//   https://github.com/dune-gdt/dune-ldg-ns
// Copyright 2026 dune-ldg-ns developers and contributors. All rights reserved.
// License: Dual licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
//      or  GPL-2.0+ (http://opensource.org/licenses/gpl-license)
//          with "runtime exception" (http://www.dune-project.org/license.html)
// Authors:
//   René Fritze (2026)

#ifndef DUNE_LDG_NS_TESTCASES_DFG_CYLINDER_HH
#define DUNE_LDG_NS_TESTCASES_DFG_CYLINDER_HH

#include <cmath>

#include "interface.hh"

namespace Dune {
namespace GDT {
namespace NavierStokes {


/**
 * \brief DFG "flow around a cylinder" benchmarks 2D-2 and 2D-3 (M. Schaefer, S. Turek, Notes Numer. Fluid Mech. 52,
 *        Vieweg 1996; reference values for 2D-3: V. John, Int. J. Numer. Meth. Fluids 44 (2004) 777--788).
 *
 * Omega = [0, 2.2] x [0, 0.41] minus the disc of radius 0.05 around (0.2, 0.2), nu = 1e-3, rho = 1, f = 0,
 * inflow u(0, y, t) = (4 U(t) y (H - y) / H^2, 0), H = 0.41, no-slip on walls and cylinder, do-nothing outflow at x
 * = 2.2. 2D-2: U(t) = 1.5 (Re = 100, periodic vortex shedding), 2D-3: U(t) = 1.5 sin(pi t / 8), t in [0, 8]. Drag/lift
 * coefficients c = 2 F / (U_mean^2 D) with U_mean = 1, D = 0.1 (see functionals/benchmark-quantities.hh).
 *
 * The mesh is not generated here, see grids/dfg-cylinder-2d.geo.
 */
template <class GV>
class DfgCylinderProblem : public ProblemInterface<GV>
{
  using BaseType = ProblemInterface<GV>;

public:
  using BaseType::d;
  using typename BaseType::BoundaryInfoType;
  using typename BaseType::DomainType;
  using typename BaseType::VectorLambdaType;
  using typename BaseType::VelocityValueType;

  static_assert(d == 2, "Only the 2D DFG benchmarks are implemented!");

  static constexpr double H = 0.41;
  static constexpr double L = 2.2;
  static constexpr double D = 0.1;
  static constexpr double cx = 0.2;
  static constexpr double cy = 0.2;

  enum class Variant
  {
    dfg_2d_2,
    dfg_2d_3
  };

  explicit DfgCylinderProblem(const Variant variant = Variant::dfg_2d_2, const double geometry_tolerance = 1e-8)
    : variant_(variant)
    , tol_(geometry_tolerance)
    , boundary_info_([tol = geometry_tolerance](const auto& x) {
      return (x[0] > L - tol) ? BoundaryKind::outflow : BoundaryKind::dirichlet;
    })
  {
  }

  std::string name() const final
  {
    return variant_ == Variant::dfg_2d_2 ? "dfg_2d_2" : "dfg_2d_3";
  }

  double viscosity() const final
  {
    return 1e-3;
  }

  const BoundaryInfoType& boundary_info() const final
  {
    return boundary_info_;
  }

  bool pure_dirichlet() const final
  {
    return false;
  }

  double end_time() const
  {
    // 2D-2: integrate until the shedding is fully periodic and evaluate max/min over the last periods
    return variant_ == Variant::dfg_2d_2 ? 30. : 8.;
  }

  double max_inflow_velocity(const double t) const
  {
    return variant_ == Variant::dfg_2d_2 ? 1.5 : 1.5 * std::sin(M_PI * t / 8.);
  }

  /// \brief True iff x lies on the cylinder boundary (up to the mesh resolution tolerance).
  static bool on_cylinder(const DomainType& x, const double tol = 1e-3)
  {
    return std::abs(std::hypot(x[0] - cx, x[1] - cy) - 0.5 * D) < tol;
  }

  VectorLambdaType force() const final
  {
    return [](const DomainType& /*x*/, const double /*t*/) { return VelocityValueType(0.); };
  }

  VectorLambdaType dirichlet() const final
  {
    const auto variant = variant_;
    const double tol = tol_;
    return [variant, tol](const DomainType& x, const double t) {
      VelocityValueType g(0.);
      if (x[0] < tol) {
        const double U = variant == Variant::dfg_2d_2 ? 1.5 : 1.5 * std::sin(M_PI * t / 8.);
        g[0] = 4. * U * x[1] * (H - x[1]) / (H * H);
      }
      return g;
    };
  }

  VectorLambdaType initial_velocity() const final
  {
    return [](const DomainType& /*x*/, const double /*t*/) { return VelocityValueType(0.); };
  }

private:
  const Variant variant_;
  const double tol_;
  const BoundaryInfoType boundary_info_;
}; // class DfgCylinderProblem


} // namespace NavierStokes
} // namespace GDT
} // namespace Dune

#endif // DUNE_LDG_NS_TESTCASES_DFG_CYLINDER_HH
