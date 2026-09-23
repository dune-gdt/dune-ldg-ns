// This file is part of the dune-ldg-ns project:
//   https://github.com/dune-gdt/dune-ldg-ns
// Copyright 2026 dune-ldg-ns developers and contributors. All rights reserved.
// License: Dual licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
//      or  GPL-2.0+ (http://opensource.org/licenses/gpl-license)
//          with "runtime exception" (http://www.dune-project.org/license.html)
// Authors:
//   René Fritze (2026)

/**
 * \file  benchmark-quantities.hh
 * \brief Drag/lift forces from the LDG numerical fluxes and point values of DG functions (DFG benchmarks).
 */
#ifndef DUNE_LDG_NS_FUNCTIONALS_BENCHMARK_QUANTITIES_HH
#define DUNE_LDG_NS_FUNCTIONALS_BENCHMARK_QUANTITIES_HH

#include <functional>
#include <optional>

#include <dune/common/fvector.hh>
#include <dune/geometry/quadraturerules.hh>
#include <dune/geometry/referenceelements.hh>

#include <dune/ldg-ns/local/integrands/ldg.hh>
#include <dune/ldg-ns/operators/ldg-navier-stokes.hh>

namespace Dune {
namespace GDT {
namespace NavierStokes {


/**
 * \brief Force exerted by the fluid on the part S of the boundary selected by on_body (a predicate on face centers):
 *
 *   F = int_S (p n - nu sigma_hat n) ds,   nu sigma_hat n = nu sigma n - nu eta / h_F (u - g),
 *
 * where n is the outer normal of the fluid domain and sigma_hat the LDG flux on Dirichlet faces, i.e. the viscous and
 * pressure traction of the discrete momentum balance. The convective boundary flux -(u.n)^- (u - g) is not included; it
 * is quadratic in the (weakly imposed) boundary error.
 */
template <class GV>
FieldVector<double, GV::dimension>
body_force(const LdgNavierStokesOperator<GV>& op,
           const typename LdgNavierStokesOperator<GV>::DenseVectorType& velocity,
           const typename LdgNavierStokesOperator<GV>::DenseVectorType& pressure,
           const typename LdgNavierStokesOperator<GV>::VectorGridFunctionType& dirichlet,
           const double t,
           const std::function<bool(const FieldVector<double, GV::dimension>&)>& on_body)
{
  static constexpr size_t d = GV::dimension;
  const XT::Common::Parameter param("t", t);
  const auto sigma_dofs = op.velocity_gradient(velocity, dirichlet, param);
  std::vector<decltype(op.make_scalar_function(sigma_dofs[0]))> sigma;
  for (size_t ij = 0; ij < d * d; ++ij)
    sigma.emplace_back(op.make_scalar_function(sigma_dofs[ij]));
  const auto u_h = op.make_velocity_function(velocity);
  const auto p_h = op.make_pressure_function(pressure);
  auto local_u = u_h.local_function();
  auto local_p = p_h.local_function();
  auto local_g = dirichlet.local_function();
  std::vector<decltype(sigma[0].local_function())> local_sigma;
  for (size_t ij = 0; ij < d * d; ++ij)
    local_sigma.emplace_back(sigma[ij].local_function());
  const auto face_diameter = LocalIntegrands::default_face_diameter<typename LdgNavierStokesOperator<GV>::I>();
  const double nu = op.viscosity();
  const double eta = op.effective_eta();
  const int order = 2 * op.options().velocity_order + 2;
  FieldVector<double, d> force(0.);
  for (auto&& element : elements(op.grid_view())) {
    if (!element.hasBoundaryIntersections())
      continue;
    for (auto&& intersection : intersections(op.grid_view(), element)) {
      if (!intersection.boundary() || intersection.neighbor() || !on_body(intersection.geometry().center()))
        continue;
      local_u->bind(element);
      local_p->bind(element);
      local_g->bind(element);
      for (auto& ls : local_sigma)
        ls->bind(element);
      const double penalty = nu * eta / face_diameter(intersection);
      const auto face_geometry = intersection.geometry();
      for (auto&& qp : QuadratureRules<double, d - 1>::rule(intersection.type(), order)) {
        const auto x_face = qp.position();
        const double weight = qp.weight() * face_geometry.integrationElement(x_face);
        const auto x = intersection.geometryInInside().global(x_face);
        const auto n = intersection.unitOuterNormal(x_face);
        const auto u = local_u->evaluate(x);
        const auto g = local_g->evaluate(x, param);
        const double p = local_p->evaluate(x)[0];
        for (size_t ii = 0; ii < d; ++ii) {
          double sigma_n = 0.;
          for (size_t jj = 0; jj < d; ++jj)
            sigma_n += local_sigma[ii * d + jj]->evaluate(x)[0] * n[jj];
          force[ii] += weight * (p * n[ii] - nu * sigma_n + penalty * (u[ii] - g[ii]));
        }
      }
    }
  }
  return force;
} // ... body_force(...)


/**
 * \brief Point value of a discrete function (brute-force element search, meant for a few output points).
 * \note Returns the value of the first element containing x; for x on a face, the (discontinuous) value is taken from
 *       an arbitrary adjacent element.
 */
template <class GV, size_t r, class R>
std::optional<FieldVector<R, r>>
point_value(const GV& grid_view,
            const XT::Functions::GridFunctionInterface<XT::Grid::extract_entity_t<GV>, r, 1, R>& function,
            const FieldVector<double, GV::dimension>& x)
{
  auto local_function = function.local_function();
  for (auto&& element : elements(grid_view)) {
    const auto geometry = element.geometry();
    const auto x_local = geometry.local(x);
    if (referenceElement(geometry).checkInside(x_local)) {
      local_function->bind(element);
      FieldVector<R, r> value = local_function->evaluate(x_local);
      return value;
    }
  }
  return std::nullopt;
} // ... point_value(...)


} // namespace NavierStokes
} // namespace GDT
} // namespace Dune

#endif // DUNE_LDG_NS_FUNCTIONALS_BENCHMARK_QUANTITIES_HH
