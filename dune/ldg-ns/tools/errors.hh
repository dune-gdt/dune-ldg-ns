// This file is part of the dune-ldg-ns project:
//   https://github.com/dune-gdt/dune-ldg-ns
// Copyright 2026 dune-ldg-ns developers and contributors. All rights reserved.
// License: Dual licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
//      or  GPL-2.0+ (http://opensource.org/licenses/gpl-license)
//          with "runtime exception" (http://www.dune-project.org/license.html)
// Authors:
//   René Fritze (2026)

#ifndef DUNE_LDG_NS_TOOLS_ERRORS_HH
#define DUNE_LDG_NS_TOOLS_ERRORS_HH

#include <cmath>
#include <functional>

#include <dune/common/fvector.hh>
#include <dune/geometry/quadraturerules.hh>

#include <dune/xt/functions/interfaces/grid-function.hh>
#include <dune/xt/grid/type_traits.hh>

namespace Dune {
namespace GDT {
namespace NavierStokes {


/// \brief Result of an L2 comparison: ||u_h - u||, ||u||, and the domain means of u_h and u (componentwise sum).
struct L2Comparison
{
  double error = 0.;
  double norm = 0.;
  double volume = 0.;
  double mean_discrete = 0.;
  double mean_exact = 0.;

  /// \brief || (u_h - mean(u_h)) - (u - mean(u)) || (for pressures determined up to a constant).
  double error_up_to_constant() const
  {
    const double diff = mean_discrete - mean_exact;
    return std::sqrt(std::max(error * error - volume * diff * diff, 0.));
  }
};


/**
 * \brief L2 comparison of a (discrete) grid function with an exact function given in global coordinates.
 * \param exact callable x -> FieldVector<double, r>
 */
template <class GV, size_t r, class Exact>
L2Comparison l2_compare(const GV& grid_view,
                        const XT::Functions::GridFunctionInterface<XT::Grid::extract_entity_t<GV>, r>& discrete,
                        const Exact& exact,
                        const int quadrature_order)
{
  L2Comparison result;
  auto local_discrete = discrete.local_function();
  for (auto&& element : elements(grid_view)) {
    local_discrete->bind(element);
    const auto geometry = element.geometry();
    for (auto&& qp : QuadratureRules<double, GV::dimension>::rule(element.type(), quadrature_order)) {
      const auto x_local = qp.position();
      const double weight = qp.weight() * geometry.integrationElement(x_local);
      const auto value_h = local_discrete->evaluate(x_local);
      const FieldVector<double, r> value = exact(geometry.global(x_local));
      for (size_t ii = 0; ii < r; ++ii) {
        const double diff = value_h[ii] - value[ii];
        result.error += weight * diff * diff;
        result.norm += weight * value[ii] * value[ii];
        result.mean_discrete += weight * value_h[ii];
        result.mean_exact += weight * value[ii];
      }
      result.volume += weight;
    }
  }
  result.error = std::sqrt(result.error);
  result.norm = std::sqrt(result.norm);
  result.mean_discrete /= result.volume;
  result.mean_exact /= result.volume;
  return result;
} // ... l2_compare(...)


} // namespace NavierStokes
} // namespace GDT
} // namespace Dune

#endif // DUNE_LDG_NS_TOOLS_ERRORS_HH
