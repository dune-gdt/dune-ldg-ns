// This file is part of the dune-ldg-ns project:
//   https://github.com/dune-gdt/dune-ldg-ns
// Copyright 2026 dune-ldg-ns developers and contributors. All rights reserved.
// License: Dual licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
//      or  GPL-2.0+ (http://opensource.org/licenses/gpl-license)
//          with "runtime exception" (http://www.dune-project.org/license.html)
// Authors:
//   René Fritze (2026)

#ifndef DUNE_LDG_NS_TESTCASES_TAYLOR_GREEN_HH
#define DUNE_LDG_NS_TESTCASES_TAYLOR_GREEN_HH

#include <cmath>

#include "interface.hh"

namespace Dune {
namespace GDT {
namespace NavierStokes {


/**
 * \brief Decaying Taylor--Green vortex on Omega = [-1, 1]^d, exact solution of the Navier--Stokes equations (f = 0):
 *
 *   u = (-cos(pi x) sin(pi y), sin(pi x) cos(pi y) [, 0]) F(t),   p = -1/4 (cos(2 pi x) + cos(2 pi y)) F(t)^2,
 *   F(t) = exp(-2 pi^2 nu t).
 *
 * In 3D this is the z-invariant extrusion (u_z = 0), which is an exact solution as well; it exercises all 3D code
 * paths with a known solution. (The classical 3D Taylor--Green transition benchmark at Re = 1600 has no closed-form
 * solution and is not covered here, see doc/design.md, Sec. 5.1.) The exact velocity is imposed weakly on the whole
 * boundary.
 */
template <class GV>
class TaylorGreenProblem : public ProblemInterface<GV>
{
  using BaseType = ProblemInterface<GV>;

public:
  using BaseType::d;
  using typename BaseType::BoundaryInfoType;
  using typename BaseType::DomainType;
  using typename BaseType::ScalarLambdaType;
  using typename BaseType::VectorLambdaType;
  using typename BaseType::VelocityValueType;

  static_assert(d == 2 || d == 3, "Taylor--Green is only available in 2D and 3D!");

  explicit TaylorGreenProblem(const double viscosity = 0.01)
    : nu_(viscosity)
    , boundary_info_([](const auto& /*x*/) { return BoundaryKind::dirichlet; })
  {
  }

  std::string name() const final
  {
    return "taylor_green_" + std::to_string(d) + "d";
  }

  double viscosity() const final
  {
    return nu_;
  }

  const BoundaryInfoType& boundary_info() const final
  {
    return boundary_info_;
  }

  bool pure_dirichlet() const final
  {
    return true;
  }

  static DomainType lower_left()
  {
    return DomainType(-1.);
  }

  static DomainType upper_right()
  {
    return DomainType(1.);
  }

  VectorLambdaType force() const final
  {
    return [](const DomainType& /*x*/, const double /*t*/) { return VelocityValueType(0.); };
  }

  VectorLambdaType dirichlet() const final
  {
    return exact_velocity();
  }

  VectorLambdaType initial_velocity() const final
  {
    return exact_velocity();
  }

  bool has_exact_solution() const final
  {
    return true;
  }

  VectorLambdaType exact_velocity() const final
  {
    const double nu = nu_;
    return [nu](const DomainType& x, const double t) {
      const double F = std::exp(-2. * M_PI * M_PI * nu * t);
      VelocityValueType u(0.);
      u[0] = -std::cos(M_PI * x[0]) * std::sin(M_PI * x[1]) * F;
      u[1] = std::sin(M_PI * x[0]) * std::cos(M_PI * x[1]) * F;
      return u;
    };
  }

  ScalarLambdaType exact_pressure() const final
  {
    const double nu = nu_;
    return [nu](const DomainType& x, const double t) {
      const double F = std::exp(-2. * M_PI * M_PI * nu * t);
      return -0.25 * (std::cos(2. * M_PI * x[0]) + std::cos(2. * M_PI * x[1])) * F * F;
    };
  }

private:
  const double nu_;
  const BoundaryInfoType boundary_info_;
}; // class TaylorGreenProblem


} // namespace NavierStokes
} // namespace GDT
} // namespace Dune

#endif // DUNE_LDG_NS_TESTCASES_TAYLOR_GREEN_HH
