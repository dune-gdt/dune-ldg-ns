// This file is part of the dune-ldg-ns project:
//   https://github.com/dune-gdt/dune-ldg-ns
// Copyright 2026 dune-ldg-ns developers and contributors. All rights reserved.
// License: Dual licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
//      or  GPL-2.0+ (http://opensource.org/licenses/gpl-license)
//          with "runtime exception" (http://www.dune-project.org/license.html)
// Authors:
//   René Fritze (2026)

#ifndef DUNE_LDG_NS_TOOLS_BOUNDARY_INFO_HH
#define DUNE_LDG_NS_TOOLS_BOUNDARY_INFO_HH

#include <functional>

#include <dune/geometry/referenceelements.hh>

#include <dune/xt/grid/boundaryinfo/interfaces.hh>
#include <dune/xt/grid/boundaryinfo/types.hh>

namespace Dune {
namespace GDT {
namespace NavierStokes {


/// \brief Boundary kinds of the Navier--Stokes problems: weak Dirichlet data or do-nothing outflow.
enum class BoundaryKind
{
  dirichlet,
  outflow
};


/**
 * \brief Stateless (hence thread safe) boundary info classifying boundary intersections by their center.
 *
 * Outflow faces are reported as XT::Grid::NeumannBoundary.
 */
template <class I>
class CenterBasedBoundaryInfo : public XT::Grid::BoundaryInfo<I>
{
  using BaseType = XT::Grid::BoundaryInfo<I>;

public:
  using typename BaseType::IntersectionType;
  using typename BaseType::WorldType;
  using ClassifierType = std::function<BoundaryKind(const WorldType& /*face_center*/)>;

  explicit CenterBasedBoundaryInfo(ClassifierType classifier)
    : BaseType("NavierStokes::CenterBasedBoundaryInfo")
    , classifier_(classifier)
  {
  }

  const XT::Grid::BoundaryType& type(const IntersectionType& intersection) const final
  {
    if (!intersection.boundary() || intersection.neighbor())
      return no_boundary_;
    const WorldType center = intersection.geometry().center();
    return classifier_(center) == BoundaryKind::dirichlet ? static_cast<const XT::Grid::BoundaryType&>(dirichlet_)
                                                          : static_cast<const XT::Grid::BoundaryType&>(outflow_);
  }

private:
  const ClassifierType classifier_;
  const XT::Grid::NoBoundary no_boundary_{};
  const XT::Grid::DirichletBoundary dirichlet_{};
  const XT::Grid::NeumannBoundary outflow_{};
}; // class CenterBasedBoundaryInfo


} // namespace NavierStokes
} // namespace GDT
} // namespace Dune

#endif // DUNE_LDG_NS_TOOLS_BOUNDARY_INFO_HH
