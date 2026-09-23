// This file is part of the dune-ldg-ns project:
//   https://github.com/dune-gdt/dune-ldg-ns
// Copyright 2026 dune-ldg-ns developers and contributors. All rights reserved.
// License: Dual licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
//      or  GPL-2.0+ (http://opensource.org/licenses/gpl-license)
//          with "runtime exception" (http://www.dune-project.org/license.html)
// Authors:
//   René Fritze (2026)

#ifndef DUNE_LDG_NS_TESTCASES_INTERFACE_HH
#define DUNE_LDG_NS_TESTCASES_INTERFACE_HH

#include <functional>
#include <string>

#include <dune/common/fvector.hh>

#include <dune/xt/common/parameter.hh>
#include <dune/xt/functions/generic/function.hh>
#include <dune/xt/functions/grid-function.hh>
#include <dune/xt/grid/type_traits.hh>

#include <dune/gdt/exceptions.hh>
#include <dune/ldg-ns/tools/boundary-info.hh>

namespace Dune {
namespace GDT {
namespace NavierStokes {


/**
 * \brief Problem definition: data are functions of (x, t), t is passed as the parameter "t".
 *
 * The velocity-valued data are exposed as XT::Functions::GridFunction<E, d>, so that they can be handed to the local
 * integrands directly.
 */
template <class GV>
class ProblemInterface
{
public:
  static constexpr size_t d = GV::dimension;
  using E = XT::Grid::extract_entity_t<GV>;
  using I = XT::Grid::extract_intersection_t<GV>;
  using DomainType = FieldVector<double, d>;
  using VelocityValueType = FieldVector<double, d>;
  using VectorGridFunctionType = XT::Functions::GridFunction<E, d>;
  using VectorLambdaType = std::function<VelocityValueType(const DomainType&, double)>;
  using ScalarLambdaType = std::function<double(const DomainType&, double)>;
  using BoundaryInfoType = CenterBasedBoundaryInfo<I>;

  virtual ~ProblemInterface() = default;

  virtual std::string name() const = 0;

  virtual double viscosity() const = 0;

  virtual const BoundaryInfoType& boundary_info() const = 0;

  /// \brief True iff there is no outflow boundary, i.e. the pressure is only determined up to a constant.
  virtual bool pure_dirichlet() const = 0;

  virtual VectorLambdaType force() const = 0;

  virtual VectorLambdaType dirichlet() const = 0;

  virtual VectorLambdaType initial_velocity() const = 0;

  virtual bool has_exact_solution() const
  {
    return false;
  }

  virtual VectorLambdaType exact_velocity() const
  {
    DUNE_THROW(Exceptions::operator_error, "problem " << name() << " has no exact solution!");
  }

  virtual ScalarLambdaType exact_pressure() const
  {
    DUNE_THROW(Exceptions::operator_error, "problem " << name() << " has no exact solution!");
  }

  /// \name Conversion of the lambdas to parametric grid functions
  /// \{

  static VectorGridFunctionType to_grid_function(VectorLambdaType func, const std::string& nm, const int order = 6)
  {
    using FunctionType = XT::Functions::GenericFunction<d, d>;
    return VectorGridFunctionType(FunctionType(
        order,
        [func](const auto& x, const auto& param) {
          typename FunctionType::RangeReturnType result = func(x, param.get("t").at(0));
          return result;
        },
        nm,
        XT::Common::ParameterType("t", 1)));
  }

  VectorGridFunctionType force_function() const
  {
    return to_grid_function(force(), name() + "_force");
  }

  VectorGridFunctionType dirichlet_function() const
  {
    return to_grid_function(dirichlet(), name() + "_dirichlet");
  }

  /// \}
}; // class ProblemInterface


} // namespace NavierStokes
} // namespace GDT
} // namespace Dune

#endif // DUNE_LDG_NS_TESTCASES_INTERFACE_HH
