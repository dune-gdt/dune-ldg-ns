// This file is part of the dune-ldg-ns project:
//   https://github.com/dune-gdt/dune-ldg-ns
// Copyright 2026 dune-ldg-ns developers and contributors. All rights reserved.
// License: Dual licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
//      or  GPL-2.0+ (http://opensource.org/licenses/gpl-license)
//          with "runtime exception" (http://www.dune-project.org/license.html)
// Authors:
//   René Fritze (2026)

#ifndef DUNE_LDG_NS_TESTCASES_KOVASZNAY_HH
#define DUNE_LDG_NS_TESTCASES_KOVASZNAY_HH

#include <cmath>

#include "interface.hh"

namespace Dune {
namespace GDT {
namespace NavierStokes {


/**
 * \brief Kovasznay flow (L. I. G. Kovasznay, Proc. Camb. Phil. Soc. 44, 1948), steady exact solution, f = 0:
 *
 *   u = 1 - exp(lambda x) cos(2 pi y),   v = lambda / (2 pi) exp(lambda x) sin(2 pi y),   p = -1/2 exp(2 lambda x) + c,
 *   lambda = Re/2 - sqrt(Re^2/4 + 4 pi^2),   nu = 1/Re,
 *
 * on Omega = [-0.5, 1] x [-0.5, 1.5] (Re = 40 by default) with the exact velocity imposed weakly on the boundary.
 */
template <class GV>
class KovasznayProblem : public ProblemInterface<GV>
{
  using BaseType = ProblemInterface<GV>;

public:
  using BaseType::d;
  using typename BaseType::BoundaryInfoType;
  using typename BaseType::DomainType;
  using typename BaseType::ScalarLambdaType;
  using typename BaseType::VectorLambdaType;
  using typename BaseType::VelocityValueType;

  static_assert(d == 2, "Kovasznay flow is a 2D problem!");

  explicit KovasznayProblem(const double reynolds = 40.)
    : re_(reynolds)
    , lambda_(0.5 * re_ - std::sqrt(0.25 * re_ * re_ + 4. * M_PI * M_PI))
    , boundary_info_([](const auto& /*x*/) { return BoundaryKind::dirichlet; })
  {
  }

  std::string name() const final
  {
    return "kovasznay";
  }

  double viscosity() const final
  {
    return 1. / re_;
  }

  double lambda() const
  {
    return lambda_;
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
    return {-0.5, -0.5};
  }

  static DomainType upper_right()
  {
    return {1., 1.5};
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
    return [](const DomainType& /*x*/, const double /*t*/) { return VelocityValueType(0.); };
  }

  bool has_exact_solution() const final
  {
    return true;
  }

  VectorLambdaType exact_velocity() const final
  {
    const double lambda = lambda_;
    return [lambda](const DomainType& x, const double /*t*/) {
      const double e = std::exp(lambda * x[0]);
      return VelocityValueType{1. - e * std::cos(2. * M_PI * x[1]),
                               lambda / (2. * M_PI) * e * std::sin(2. * M_PI * x[1])};
    };
  }

  ScalarLambdaType exact_pressure() const final
  {
    const double lambda = lambda_;
    return [lambda](const DomainType& x, const double /*t*/) { return -0.5 * std::exp(2. * lambda * x[0]); };
  }

private:
  const double re_;
  const double lambda_;
  const BoundaryInfoType boundary_info_;
}; // class KovasznayProblem


} // namespace NavierStokes
} // namespace GDT
} // namespace Dune

#endif // DUNE_LDG_NS_TESTCASES_KOVASZNAY_HH
