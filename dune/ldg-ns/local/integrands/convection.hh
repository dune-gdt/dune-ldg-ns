// This file is part of the dune-ldg-ns project:
//   https://github.com/dune-gdt/dune-ldg-ns
// Copyright 2026 dune-ldg-ns developers and contributors. All rights reserved.
// License: Dual licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
//      or  GPL-2.0+ (http://opensource.org/licenses/gpl-license)
//          with "runtime exception" (http://www.dune-project.org/license.html)
// Authors:
//   René Fritze (2026)

/**
 * \file  convection.hh
 * \brief Local integrands of the Oseen-linearized DG convection form t_h(w; u, v) (doc/design.md, Sec. 3.4).
 *
 * t_h(w; u, v) =   sum_K int_K (w . grad u) v + 1/2 (div w) u v                      (Temam's skew-symmetrization)
 *                - sum_F int_F ({w} . n) [[u]] {v} + 1/2 ([[w]] . n) {u v}           (Di Pietro--Ern)
 *                + sum_F int_F upwind/2 |{w} . n| [[u]] [[v]]                          (upwinding)
 *                - int_{Gamma_D u Gamma_N} (w . n)^- u v   (+ int_{Gamma_D} (w . n)^- g v on the right hand side)
 *
 * The form is scalar and acts identically on every velocity component (Picard/Oseen linearization), and
 * t_h(w; v, v) = sum_F upwind/2 |{w}.n| [[v]]^2 + 1/2 int_{boundary} |w . n| v^2 >= 0 for any w (no discrete
 * divergence constraint on w is needed).
 */
#ifndef DUNE_LDG_NS_LOCAL_INTEGRANDS_CONVECTION_HH
#define DUNE_LDG_NS_LOCAL_INTEGRANDS_CONVECTION_HH

#include <cmath>

#include <dune/xt/functions/grid-function.hh>

#include <dune/gdt/exceptions.hh>
#include <dune/gdt/local/integrands/interfaces.hh>

namespace Dune {
namespace GDT {
namespace NavierStokes {
namespace LocalIntegrands {


/// \brief Element part: (w . grad u) v + 1/2 (div w) u v.
template <class E>
class ConvectionElement : public LocalBinaryElementIntegrandInterface<E>
{
  using ThisType = ConvectionElement;
  using BaseType = LocalBinaryElementIntegrandInterface<E>;

public:
  using BaseType::d;
  using typename BaseType::DomainType;
  using typename BaseType::F;
  using typename BaseType::LocalAnsatzBasisType;
  using typename BaseType::LocalTestBasisType;
  using VelocityType = XT::Functions::GridFunctionInterface<E, d>;

  explicit ConvectionElement(XT::Functions::GridFunction<E, d> w)
    : BaseType(w.parameter_type(), "NavierStokes::LocalIntegrands::ConvectionElement")
    , w_(w.copy_as_grid_function())
    , local_w_(w_->local_function())
  {
  }

  ConvectionElement(const ThisType& other)
    : BaseType(other)
    , w_(other.w_->copy_as_grid_function())
    , local_w_(w_->local_function())
  {
  }

  ConvectionElement(ThisType&& source) noexcept = default;

  std::unique_ptr<BaseType> copy_as_binary_element_integrand() const final
  {
    return std::make_unique<ThisType>(*this);
  }

protected:
  void post_bind(const E& element) final
  {
    local_w_->bind(element);
  }

public:
  int order(const LocalTestBasisType& test_basis,
            const LocalAnsatzBasisType& ansatz_basis,
            const XT::Common::Parameter& param = {}) const final
  {
    return local_w_->order(param) + test_basis.order(param) + ansatz_basis.order(param);
  }

  using BaseType::evaluate;

  void evaluate(const LocalTestBasisType& test_basis,
                const LocalAnsatzBasisType& ansatz_basis,
                const DomainType& x,
                DynamicMatrix<F>& result,
                const XT::Common::Parameter& param = {}) const final
  {
    this->ensure_size_and_clear_results(test_basis, ansatz_basis, result, param);
    test_basis.evaluate(x, test_values_, param);
    ansatz_basis.evaluate(x, ansatz_values_, param);
    ansatz_basis.jacobians(x, ansatz_grads_, param);
    const auto w = local_w_->evaluate(x, param);
    const auto grad_w = local_w_->jacobian(x, param);
    double div_w = 0.;
    for (size_t ii = 0; ii < d; ++ii)
      div_w += grad_w[ii][ii];
    const size_t rows = test_basis.size(param);
    const size_t cols = ansatz_basis.size(param);
    for (size_t ii = 0; ii < rows; ++ii)
      for (size_t jj = 0; jj < cols; ++jj)
        result[ii][jj] = ((w * ansatz_grads_[jj][0]) + 0.5 * div_w * ansatz_values_[jj][0]) * test_values_[ii][0];
  }

private:
  std::unique_ptr<VelocityType> w_;
  std::unique_ptr<typename VelocityType::LocalFunctionType> local_w_;
  mutable std::vector<typename LocalTestBasisType::RangeType> test_values_;
  mutable std::vector<typename LocalAnsatzBasisType::RangeType> ansatz_values_;
  mutable std::vector<typename LocalAnsatzBasisType::DerivativeRangeType> ansatz_grads_;
}; // class ConvectionElement


/// \brief Inner-face part: -({w}.n)[[u]]{v} - 1/2([[w]].n){uv} + upwind/2 |{w}.n| [[u]][[v]].
template <class I>
class ConvectionInnerFace : public LocalQuaternaryIntersectionIntegrandInterface<I>
{
  using ThisType = ConvectionInnerFace;
  using BaseType = LocalQuaternaryIntersectionIntegrandInterface<I>;

public:
  using BaseType::d;
  using typename BaseType::DomainType;
  using typename BaseType::E;
  using typename BaseType::F;
  using typename BaseType::IntersectionType;
  using typename BaseType::LocalAnsatzBasisType;
  using typename BaseType::LocalTestBasisType;
  using VelocityType = XT::Functions::GridFunctionInterface<E, d>;

  explicit ConvectionInnerFace(XT::Functions::GridFunction<E, d> w, const double upwind = 1.)
    : BaseType(w.parameter_type(), "NavierStokes::LocalIntegrands::ConvectionInnerFace")
    , w_(w.copy_as_grid_function())
    , local_w_in_(w_->local_function())
    , local_w_out_(w_->local_function())
    , upwind_(upwind)
  {
  }

  ConvectionInnerFace(const ThisType& other)
    : BaseType(other)
    , w_(other.w_->copy_as_grid_function())
    , local_w_in_(w_->local_function())
    , local_w_out_(w_->local_function())
    , upwind_(other.upwind_)
  {
  }

  ConvectionInnerFace(ThisType&& source) noexcept = default;

  std::unique_ptr<BaseType> copy_as_quaternary_intersection_integrand() const final
  {
    return std::make_unique<ThisType>(*this);
  }

protected:
  void post_bind(const IntersectionType& intrsctn) final
  {
    DUNE_THROW_IF(!intrsctn.neighbor(), Exceptions::integrand_error, "only for inner intersections!");
    local_w_in_->bind(intrsctn.inside());
    local_w_out_->bind(intrsctn.outside());
  }

public:
  int order(const LocalTestBasisType& test_basis_inside,
            const LocalAnsatzBasisType& ansatz_basis_inside,
            const LocalTestBasisType& test_basis_outside,
            const LocalAnsatzBasisType& ansatz_basis_outside,
            const XT::Common::Parameter& param = {}) const final
  {
    return std::max(local_w_in_->order(param), local_w_out_->order(param))
           + std::max(test_basis_inside.order(param), test_basis_outside.order(param))
           + std::max(ansatz_basis_inside.order(param), ansatz_basis_outside.order(param));
  }

  using BaseType::evaluate;

  void evaluate(const LocalTestBasisType& test_basis_inside,
                const LocalAnsatzBasisType& ansatz_basis_inside,
                const LocalTestBasisType& test_basis_outside,
                const LocalAnsatzBasisType& ansatz_basis_outside,
                const DomainType& point_in_reference_intersection,
                DynamicMatrix<F>& result_in_in,
                DynamicMatrix<F>& result_in_out,
                DynamicMatrix<F>& result_out_in,
                DynamicMatrix<F>& result_out_out,
                const XT::Common::Parameter& param = {}) const final
  {
    this->ensure_size_and_clear_results(test_basis_inside,
                                        ansatz_basis_inside,
                                        test_basis_outside,
                                        ansatz_basis_outside,
                                        result_in_in,
                                        result_in_out,
                                        result_out_in,
                                        result_out_out,
                                        param);
    const auto& intersection = this->intersection();
    const auto x_in = intersection.geometryInInside().global(point_in_reference_intersection);
    const auto x_out = intersection.geometryInOutside().global(point_in_reference_intersection);
    const auto normal = intersection.unitOuterNormal(point_in_reference_intersection);
    test_basis_inside.evaluate(x_in, test_in_, param);
    test_basis_outside.evaluate(x_out, test_out_, param);
    ansatz_basis_inside.evaluate(x_in, ansatz_in_, param);
    ansatz_basis_outside.evaluate(x_out, ansatz_out_, param);
    const auto w_in = local_w_in_->evaluate(x_in, param);
    const auto w_out = local_w_out_->evaluate(x_out, param);
    const double a = 0.5 * ((w_in + w_out) * normal); // {w} . n
    const double b = (w_in - w_out) * normal; // [[w]] . n
    const double c = 0.5 * upwind_ * std::abs(a);
    // coefficients of u^{in/out} v^{in/out}, see doc/design.md, Sec. 3.4
    const double c_in_in = -0.5 * a - 0.25 * b + c;
    const double c_in_out = 0.5 * a - c;
    const double c_out_in = -0.5 * a - c;
    const double c_out_out = 0.5 * a - 0.25 * b + c;
    const size_t rows_in = test_basis_inside.size(param);
    const size_t rows_out = test_basis_outside.size(param);
    const size_t cols_in = ansatz_basis_inside.size(param);
    const size_t cols_out = ansatz_basis_outside.size(param);
    for (size_t ii = 0; ii < rows_in; ++ii) {
      for (size_t jj = 0; jj < cols_in; ++jj)
        result_in_in[ii][jj] = c_in_in * ansatz_in_[jj][0] * test_in_[ii][0];
      for (size_t jj = 0; jj < cols_out; ++jj)
        result_in_out[ii][jj] = c_in_out * ansatz_out_[jj][0] * test_in_[ii][0];
    }
    for (size_t ii = 0; ii < rows_out; ++ii) {
      for (size_t jj = 0; jj < cols_in; ++jj)
        result_out_in[ii][jj] = c_out_in * ansatz_in_[jj][0] * test_out_[ii][0];
      for (size_t jj = 0; jj < cols_out; ++jj)
        result_out_out[ii][jj] = c_out_out * ansatz_out_[jj][0] * test_out_[ii][0];
    }
  } // ... evaluate(...)

private:
  std::unique_ptr<VelocityType> w_;
  std::unique_ptr<typename VelocityType::LocalFunctionType> local_w_in_;
  std::unique_ptr<typename VelocityType::LocalFunctionType> local_w_out_;
  const double upwind_;
  mutable std::vector<typename LocalTestBasisType::RangeType> test_in_;
  mutable std::vector<typename LocalTestBasisType::RangeType> test_out_;
  mutable std::vector<typename LocalAnsatzBasisType::RangeType> ansatz_in_;
  mutable std::vector<typename LocalAnsatzBasisType::RangeType> ansatz_out_;
}; // class ConvectionInnerFace


/**
 * \brief Boundary part -(w.n)^- u v, with (w.n)^- = min(w.n, 0).
 *
 * Used on Dirichlet faces (weak inflow condition) and on outflow faces (directional do-nothing: backflow is damped).
 */
template <class I>
class ConvectionBoundary : public LocalBinaryIntersectionIntegrandInterface<I>
{
  using ThisType = ConvectionBoundary;
  using BaseType = LocalBinaryIntersectionIntegrandInterface<I>;

public:
  using BaseType::d;
  using typename BaseType::DomainType;
  using typename BaseType::E;
  using typename BaseType::F;
  using typename BaseType::IntersectionType;
  using typename BaseType::LocalAnsatzBasisType;
  using typename BaseType::LocalTestBasisType;
  using VelocityType = XT::Functions::GridFunctionInterface<E, d>;

  explicit ConvectionBoundary(XT::Functions::GridFunction<E, d> w)
    : BaseType(w.parameter_type(), "NavierStokes::LocalIntegrands::ConvectionBoundary")
    , w_(w.copy_as_grid_function())
    , local_w_(w_->local_function())
  {
  }

  ConvectionBoundary(const ThisType& other)
    : BaseType(other)
    , w_(other.w_->copy_as_grid_function())
    , local_w_(w_->local_function())
  {
  }

  ConvectionBoundary(ThisType&& source) noexcept = default;

  std::unique_ptr<BaseType> copy_as_binary_intersection_integrand() const final
  {
    return std::make_unique<ThisType>(*this);
  }

protected:
  void post_bind(const IntersectionType& intrsctn) final
  {
    local_w_->bind(intrsctn.inside());
  }

public:
  int order(const LocalTestBasisType& test_basis,
            const LocalAnsatzBasisType& ansatz_basis,
            const XT::Common::Parameter& param = {}) const final
  {
    return local_w_->order(param) + test_basis.order(param) + ansatz_basis.order(param);
  }

  using BaseType::evaluate;

  void evaluate(const LocalTestBasisType& test_basis,
                const LocalAnsatzBasisType& ansatz_basis,
                const DomainType& point_in_reference_intersection,
                DynamicMatrix<F>& result,
                const XT::Common::Parameter& param = {}) const final
  {
    this->ensure_size_and_clear_results(test_basis, ansatz_basis, result, param);
    const auto& intersection = this->intersection();
    const auto x_in = intersection.geometryInInside().global(point_in_reference_intersection);
    const auto normal = intersection.unitOuterNormal(point_in_reference_intersection);
    test_basis.evaluate(x_in, test_values_, param);
    ansatz_basis.evaluate(x_in, ansatz_values_, param);
    const double wn_minus = std::min(local_w_->evaluate(x_in, param) * normal, 0.);
    const size_t rows = test_basis.size(param);
    const size_t cols = ansatz_basis.size(param);
    for (size_t ii = 0; ii < rows; ++ii)
      for (size_t jj = 0; jj < cols; ++jj)
        result[ii][jj] = -wn_minus * ansatz_values_[jj][0] * test_values_[ii][0];
  }

private:
  std::unique_ptr<VelocityType> w_;
  std::unique_ptr<typename VelocityType::LocalFunctionType> local_w_;
  mutable std::vector<typename LocalTestBasisType::RangeType> test_values_;
  mutable std::vector<typename LocalAnsatzBasisType::RangeType> ansatz_values_;
}; // class ConvectionBoundary


/// \brief Right hand side of ConvectionBoundary on Dirichlet faces: -(w.n)^- g_c v.
template <class I>
class ConvectionBoundaryData : public LocalUnaryIntersectionIntegrandInterface<I>
{
  using ThisType = ConvectionBoundaryData;
  using BaseType = LocalUnaryIntersectionIntegrandInterface<I>;

public:
  using BaseType::d;
  using typename BaseType::DomainType;
  using typename BaseType::E;
  using typename BaseType::F;
  using typename BaseType::IntersectionType;
  using typename BaseType::LocalTestBasisType;
  using VelocityType = XT::Functions::GridFunctionInterface<E, d>;

  ConvectionBoundaryData(XT::Functions::GridFunction<E, d> w,
                         XT::Functions::GridFunction<E, d> dirichlet_data,
                         const size_t component)
    : BaseType(w.parameter_type() + dirichlet_data.parameter_type(),
               "NavierStokes::LocalIntegrands::ConvectionBoundaryData")
    , w_(w.copy_as_grid_function())
    , g_(dirichlet_data.copy_as_grid_function())
    , local_w_(w_->local_function())
    , local_g_(g_->local_function())
    , component_(component)
  {
  }

  ConvectionBoundaryData(const ThisType& other)
    : BaseType(other)
    , w_(other.w_->copy_as_grid_function())
    , g_(other.g_->copy_as_grid_function())
    , local_w_(w_->local_function())
    , local_g_(g_->local_function())
    , component_(other.component_)
  {
  }

  ConvectionBoundaryData(ThisType&& source) noexcept = default;

  std::unique_ptr<BaseType> copy_as_unary_intersection_integrand() const final
  {
    return std::make_unique<ThisType>(*this);
  }

protected:
  void post_bind(const IntersectionType& intrsctn) final
  {
    local_w_->bind(intrsctn.inside());
    local_g_->bind(intrsctn.inside());
  }

public:
  int order(const LocalTestBasisType& test_basis, const XT::Common::Parameter& param = {}) const final
  {
    return local_w_->order(param) + local_g_->order(param) + test_basis.order(param);
  }

  using BaseType::evaluate;

  void evaluate(const LocalTestBasisType& test_basis,
                const DomainType& point_in_reference_intersection,
                DynamicVector<F>& result,
                const XT::Common::Parameter& param = {}) const final
  {
    this->ensure_size_and_clear_results(test_basis, result, param);
    const auto& intersection = this->intersection();
    const auto x_in = intersection.geometryInInside().global(point_in_reference_intersection);
    const auto normal = intersection.unitOuterNormal(point_in_reference_intersection);
    test_basis.evaluate(x_in, test_values_, param);
    const double wn_minus = std::min(local_w_->evaluate(x_in, param) * normal, 0.);
    const double c = -wn_minus * local_g_->evaluate(x_in, param)[component_];
    const size_t size = test_basis.size(param);
    for (size_t ii = 0; ii < size; ++ii)
      result[ii] = c * test_values_[ii][0];
  }

private:
  std::unique_ptr<VelocityType> w_;
  std::unique_ptr<VelocityType> g_;
  std::unique_ptr<typename VelocityType::LocalFunctionType> local_w_;
  std::unique_ptr<typename VelocityType::LocalFunctionType> local_g_;
  const size_t component_;
  mutable std::vector<typename LocalTestBasisType::RangeType> test_values_;
}; // class ConvectionBoundaryData


} // namespace LocalIntegrands
} // namespace NavierStokes
} // namespace GDT
} // namespace Dune

#endif // DUNE_LDG_NS_LOCAL_INTEGRANDS_CONVECTION_HH
