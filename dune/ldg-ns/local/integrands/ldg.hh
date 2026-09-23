// This file is part of the dune-ldg-ns project:
//   https://github.com/dune-gdt/dune-ldg-ns
// Copyright 2026 dune-ldg-ns developers and contributors. All rights reserved.
// License: Dual licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
//      or  GPL-2.0+ (http://opensource.org/licenses/gpl-license)
//          with "runtime exception" (http://www.dune-project.org/license.html)
// Authors:
//   René Fritze (2026)

/**
 * \file  ldg.hh
 * \brief Scalar local integrands from which all LDG blocks (gradient, divergence, penalties) are assembled.
 *
 * Conventions (see doc/design.md, Sec. 3):
 *   - on an inner intersection F, n is the unit outer normal of the inside element K^-,
 *     [[w]] = w^- - w^+ (scalar jump), {w} = (w^- + w^+)/2,
 *   - beta_F = beta . n is the LDG switch, the "weighted average" is {tau}_beta = (1/2 - beta_F) tau^- + (1/2 + beta_F)
 * tau^+, i.e. {tau}_beta = {tau} - beta_F [[tau]]. The product [[w]] n_j {tau}_beta is independent of the orientation
 * of F.
 *
 * All integrands are scalar (range dimension 1) and are applied per velocity component and/or per spatial
 * direction j, see NavierStokes::LdgGradientOperator.
 */
#ifndef DUNE_LDG_NS_LOCAL_INTEGRANDS_LDG_HH
#define DUNE_LDG_NS_LOCAL_INTEGRANDS_LDG_HH

#include <functional>

#include <dune/xt/common/fvector.hh>
#include <dune/xt/functions/grid-function.hh>
#include <dune/xt/grid/entity.hh>
#include <dune/xt/grid/intersection.hh>

#include <dune/gdt/exceptions.hh>
#include <dune/gdt/local/integrands/interfaces.hh>

namespace Dune {
namespace GDT {
namespace NavierStokes {
namespace LocalIntegrands {


/// \brief Default face size h_F: diameter of the intersection (d > 1), average element diameter (d == 1).
template <class I>
std::function<double(const I&)> default_face_diameter()
{
  return [](const I& intersection) {
    if constexpr (I::Entity::dimension == 1) {
      if (intersection.neighbor())
        return 0.5 * (XT::Grid::diameter(intersection.inside()) + XT::Grid::diameter(intersection.outside()));
      return XT::Grid::diameter(intersection.inside());
    } else
      return XT::Grid::diameter(intersection);
  };
}


/**
 * \brief Element integrand factor * (d/dx_j ansatz) * test.
 *
 * Volume part of the LDG gradient G_j (factor = 1) and of the DG divergence B_j (factor = -1).
 */
template <class E>
class DirectionalDerivative : public LocalBinaryElementIntegrandInterface<E>
{
  using ThisType = DirectionalDerivative;
  using BaseType = LocalBinaryElementIntegrandInterface<E>;

public:
  using typename BaseType::DomainType;
  using typename BaseType::F;
  using typename BaseType::LocalAnsatzBasisType;
  using typename BaseType::LocalTestBasisType;

  DirectionalDerivative(const size_t direction, const double factor = 1.)
    : BaseType({}, "NavierStokes::LocalIntegrands::DirectionalDerivative")
    , direction_(direction)
    , factor_(factor)
  {
    DUNE_THROW_IF(direction_ >= E::dimension, Exceptions::integrand_error, "direction = " << direction_);
  }

  DirectionalDerivative(const ThisType& other) = default;
  DirectionalDerivative(ThisType&& source) noexcept = default;

  std::unique_ptr<BaseType> copy_as_binary_element_integrand() const final
  {
    return std::make_unique<ThisType>(*this);
  }

  int order(const LocalTestBasisType& test_basis,
            const LocalAnsatzBasisType& ansatz_basis,
            const XT::Common::Parameter& param = {}) const final
  {
    // not order - 1: for Q_k the derivative keeps degree k in the transverse directions
    return test_basis.order(param) + ansatz_basis.order(param);
  }

  using BaseType::evaluate;

  void evaluate(const LocalTestBasisType& test_basis,
                const LocalAnsatzBasisType& ansatz_basis,
                const DomainType& point_in_reference_element,
                DynamicMatrix<F>& result,
                const XT::Common::Parameter& param = {}) const final
  {
    this->ensure_size_and_clear_results(test_basis, ansatz_basis, result, param);
    test_basis.evaluate(point_in_reference_element, test_values_, param);
    ansatz_basis.jacobians(point_in_reference_element, ansatz_grads_, param);
    const size_t rows = test_basis.size(param);
    const size_t cols = ansatz_basis.size(param);
    for (size_t ii = 0; ii < rows; ++ii)
      for (size_t jj = 0; jj < cols; ++jj)
        result[ii][jj] = factor_ * ansatz_grads_[jj][0][direction_] * test_values_[ii][0];
  }

private:
  const size_t direction_;
  const double factor_;
  mutable std::vector<typename LocalTestBasisType::RangeType> test_values_;
  mutable std::vector<typename LocalAnsatzBasisType::DerivativeRangeType> ansatz_grads_;
}; // class DirectionalDerivative


/**
 * \brief Inner-face integrand factor * [[w]] n_j {tau}_beta (w: ansatz, tau: test).
 *
 * With factor = -1 this is the inner-face part of the LDG gradient G_j, with factor = +1 and beta = 0 the
 * inner-face part of the DG divergence B_j (test = pressure).
 */
template <class I>
class WeightedAverageJumpNormal : public LocalQuaternaryIntersectionIntegrandInterface<I>
{
  using ThisType = WeightedAverageJumpNormal;
  using BaseType = LocalQuaternaryIntersectionIntegrandInterface<I>;

public:
  using BaseType::d;
  using typename BaseType::DomainType;
  using typename BaseType::F;
  using typename BaseType::IntersectionType;
  using typename BaseType::LocalAnsatzBasisType;
  using typename BaseType::LocalTestBasisType;
  using BetaType = FieldVector<double, d>;

  WeightedAverageJumpNormal(const size_t direction, const double factor, const BetaType& beta = BetaType(0.))
    : BaseType({}, "NavierStokes::LocalIntegrands::WeightedAverageJumpNormal")
    , direction_(direction)
    , factor_(factor)
    , beta_(beta)
  {
    DUNE_THROW_IF(direction_ >= d, Exceptions::integrand_error, "direction = " << direction_);
  }

  WeightedAverageJumpNormal(const ThisType& other) = default;
  WeightedAverageJumpNormal(ThisType&& source) noexcept = default;

  std::unique_ptr<BaseType> copy_as_quaternary_intersection_integrand() const final
  {
    return std::make_unique<ThisType>(*this);
  }

protected:
  void post_bind(const IntersectionType& intrsctn) final
  {
    DUNE_THROW_IF(!intrsctn.neighbor(), Exceptions::integrand_error, "only for inner intersections!");
  }

public:
  int order(const LocalTestBasisType& test_basis_inside,
            const LocalAnsatzBasisType& ansatz_basis_inside,
            const LocalTestBasisType& test_basis_outside,
            const LocalAnsatzBasisType& ansatz_basis_outside,
            const XT::Common::Parameter& param = {}) const final
  {
    return std::max(test_basis_inside.order(param), test_basis_outside.order(param))
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
    const double beta_F = beta_ * normal;
    const double c_in = factor_ * normal[direction_] * (0.5 - beta_F);
    const double c_out = factor_ * normal[direction_] * (0.5 + beta_F);
    const size_t rows_in = test_basis_inside.size(param);
    const size_t rows_out = test_basis_outside.size(param);
    const size_t cols_in = ansatz_basis_inside.size(param);
    const size_t cols_out = ansatz_basis_outside.size(param);
    // [[w]] = w^- - w^+
    for (size_t ii = 0; ii < rows_in; ++ii) {
      for (size_t jj = 0; jj < cols_in; ++jj)
        result_in_in[ii][jj] = c_in * ansatz_in_[jj][0] * test_in_[ii][0];
      for (size_t jj = 0; jj < cols_out; ++jj)
        result_in_out[ii][jj] = -c_in * ansatz_out_[jj][0] * test_in_[ii][0];
    }
    for (size_t ii = 0; ii < rows_out; ++ii) {
      for (size_t jj = 0; jj < cols_in; ++jj)
        result_out_in[ii][jj] = c_out * ansatz_in_[jj][0] * test_out_[ii][0];
      for (size_t jj = 0; jj < cols_out; ++jj)
        result_out_out[ii][jj] = -c_out * ansatz_out_[jj][0] * test_out_[ii][0];
    }
  } // ... evaluate(...)

private:
  const size_t direction_;
  const double factor_;
  const BetaType beta_;
  mutable std::vector<typename LocalTestBasisType::RangeType> test_in_;
  mutable std::vector<typename LocalTestBasisType::RangeType> test_out_;
  mutable std::vector<typename LocalAnsatzBasisType::RangeType> ansatz_in_;
  mutable std::vector<typename LocalAnsatzBasisType::RangeType> ansatz_out_;
}; // class WeightedAverageJumpNormal


/**
 * \brief Boundary-face integrand factor * w n_j tau (w: ansatz, tau: test, both traces from the inside).
 *
 * factor = -1: Dirichlet part of G_j; factor = +1: Dirichlet part of B_j.
 */
template <class I>
class BoundaryNormal : public LocalBinaryIntersectionIntegrandInterface<I>
{
  using ThisType = BoundaryNormal;
  using BaseType = LocalBinaryIntersectionIntegrandInterface<I>;

public:
  using BaseType::d;
  using typename BaseType::DomainType;
  using typename BaseType::F;
  using typename BaseType::LocalAnsatzBasisType;
  using typename BaseType::LocalTestBasisType;

  BoundaryNormal(const size_t direction, const double factor)
    : BaseType({}, "NavierStokes::LocalIntegrands::BoundaryNormal")
    , direction_(direction)
    , factor_(factor)
  {
  }

  BoundaryNormal(const ThisType& other) = default;
  BoundaryNormal(ThisType&& source) noexcept = default;

  std::unique_ptr<BaseType> copy_as_binary_intersection_integrand() const final
  {
    return std::make_unique<ThisType>(*this);
  }

  int order(const LocalTestBasisType& test_basis,
            const LocalAnsatzBasisType& ansatz_basis,
            const XT::Common::Parameter& param = {}) const final
  {
    return test_basis.order(param) + ansatz_basis.order(param);
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
    const double c = factor_ * normal[direction_];
    const size_t rows = test_basis.size(param);
    const size_t cols = ansatz_basis.size(param);
    for (size_t ii = 0; ii < rows; ++ii)
      for (size_t jj = 0; jj < cols; ++jj)
        result[ii][jj] = c * ansatz_values_[jj][0] * test_values_[ii][0];
  }

private:
  const size_t direction_;
  const double factor_;
  mutable std::vector<typename LocalTestBasisType::RangeType> test_values_;
  mutable std::vector<typename LocalAnsatzBasisType::RangeType> ansatz_values_;
}; // class BoundaryNormal


/**
 * \brief Boundary-face data integrand factor * g_c n_j tau for vector-valued data g (component c, direction j).
 *
 * Right hand side counterpart of BoundaryNormal: (g_c, n_j) = (i, j) gives the Dirichlet data of sigma_ij,
 * summing (j, j) over j gives the Dirichlet data g . n of the continuity equation.
 */
template <class I>
class BoundaryDataNormal : public LocalUnaryIntersectionIntegrandInterface<I>
{
  using ThisType = BoundaryDataNormal;
  using BaseType = LocalUnaryIntersectionIntegrandInterface<I>;

public:
  using BaseType::d;
  using typename BaseType::DomainType;
  using typename BaseType::E;
  using typename BaseType::F;
  using typename BaseType::IntersectionType;
  using typename BaseType::LocalTestBasisType;
  using DataType = XT::Functions::GridFunctionInterface<E, d>;

  BoundaryDataNormal(XT::Functions::GridFunction<E, d> data,
                     const size_t component,
                     const size_t direction,
                     const double factor = 1.)
    : BaseType(data.parameter_type(), "NavierStokes::LocalIntegrands::BoundaryDataNormal")
    , data_(data.copy_as_grid_function())
    , local_data_(data_->local_function())
    , component_(component)
    , direction_(direction)
    , factor_(factor)
  {
  }

  BoundaryDataNormal(const ThisType& other)
    : BaseType(other)
    , data_(other.data_->copy_as_grid_function())
    , local_data_(data_->local_function())
    , component_(other.component_)
    , direction_(other.direction_)
    , factor_(other.factor_)
  {
  }

  BoundaryDataNormal(ThisType&& source) noexcept = default;

  std::unique_ptr<BaseType> copy_as_unary_intersection_integrand() const final
  {
    return std::make_unique<ThisType>(*this);
  }

protected:
  void post_bind(const IntersectionType& intrsctn) final
  {
    local_data_->bind(intrsctn.inside());
  }

public:
  int order(const LocalTestBasisType& test_basis, const XT::Common::Parameter& param = {}) const final
  {
    return local_data_->order(param) + test_basis.order(param);
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
    const auto g = local_data_->evaluate(x_in, param);
    const double c = factor_ * g[component_] * normal[direction_];
    const size_t size = test_basis.size(param);
    for (size_t ii = 0; ii < size; ++ii)
      result[ii] = c * test_values_[ii][0];
  }

private:
  std::unique_ptr<DataType> data_;
  std::unique_ptr<typename DataType::LocalFunctionType> local_data_;
  const size_t component_;
  const size_t direction_;
  const double factor_;
  mutable std::vector<typename LocalTestBasisType::RangeType> test_values_;
}; // class BoundaryDataNormal


/**
 * \brief Inner-face jump penalty coefficient(h_F) * [[w]] [[v]].
 *
 * Velocity: coefficient(h) = nu * eta / h (LDG C11 term); pressure (equal order only): coefficient(h) = gamma * h.
 */
template <class I>
class InnerJumpPenalty : public LocalQuaternaryIntersectionIntegrandInterface<I>
{
  using ThisType = InnerJumpPenalty;
  using BaseType = LocalQuaternaryIntersectionIntegrandInterface<I>;

public:
  using typename BaseType::DomainType;
  using typename BaseType::F;
  using typename BaseType::IntersectionType;
  using typename BaseType::LocalAnsatzBasisType;
  using typename BaseType::LocalTestBasisType;

  InnerJumpPenalty(std::function<double(double)> coefficient_of_h,
                   std::function<double(const I&)> face_diameter = default_face_diameter<I>())
    : BaseType({}, "NavierStokes::LocalIntegrands::InnerJumpPenalty")
    , coefficient_of_h_(coefficient_of_h)
    , face_diameter_(face_diameter)
  {
  }

  InnerJumpPenalty(const ThisType& other) = default;
  InnerJumpPenalty(ThisType&& source) noexcept = default;

  std::unique_ptr<BaseType> copy_as_quaternary_intersection_integrand() const final
  {
    return std::make_unique<ThisType>(*this);
  }

  int order(const LocalTestBasisType& test_basis_inside,
            const LocalAnsatzBasisType& ansatz_basis_inside,
            const LocalTestBasisType& test_basis_outside,
            const LocalAnsatzBasisType& ansatz_basis_outside,
            const XT::Common::Parameter& param = {}) const final
  {
    return std::max(test_basis_inside.order(param), test_basis_outside.order(param))
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
    test_basis_inside.evaluate(x_in, test_in_, param);
    test_basis_outside.evaluate(x_out, test_out_, param);
    ansatz_basis_inside.evaluate(x_in, ansatz_in_, param);
    ansatz_basis_outside.evaluate(x_out, ansatz_out_, param);
    const double c = coefficient_of_h_(face_diameter_(intersection));
    const size_t rows_in = test_basis_inside.size(param);
    const size_t rows_out = test_basis_outside.size(param);
    const size_t cols_in = ansatz_basis_inside.size(param);
    const size_t cols_out = ansatz_basis_outside.size(param);
    for (size_t ii = 0; ii < rows_in; ++ii) {
      for (size_t jj = 0; jj < cols_in; ++jj)
        result_in_in[ii][jj] = c * ansatz_in_[jj][0] * test_in_[ii][0];
      for (size_t jj = 0; jj < cols_out; ++jj)
        result_in_out[ii][jj] = -c * ansatz_out_[jj][0] * test_in_[ii][0];
    }
    for (size_t ii = 0; ii < rows_out; ++ii) {
      for (size_t jj = 0; jj < cols_in; ++jj)
        result_out_in[ii][jj] = -c * ansatz_in_[jj][0] * test_out_[ii][0];
      for (size_t jj = 0; jj < cols_out; ++jj)
        result_out_out[ii][jj] = c * ansatz_out_[jj][0] * test_out_[ii][0];
    }
  } // ... evaluate(...)

private:
  const std::function<double(double)> coefficient_of_h_;
  const std::function<double(const I&)> face_diameter_;
  mutable std::vector<typename LocalTestBasisType::RangeType> test_in_;
  mutable std::vector<typename LocalTestBasisType::RangeType> test_out_;
  mutable std::vector<typename LocalAnsatzBasisType::RangeType> ansatz_in_;
  mutable std::vector<typename LocalAnsatzBasisType::RangeType> ansatz_out_;
}; // class InnerJumpPenalty


/// \brief Boundary-face penalty coefficient(h_F) * w v (Dirichlet part of the LDG C11 term).
template <class I>
class BoundaryPenalty : public LocalBinaryIntersectionIntegrandInterface<I>
{
  using ThisType = BoundaryPenalty;
  using BaseType = LocalBinaryIntersectionIntegrandInterface<I>;

public:
  using typename BaseType::DomainType;
  using typename BaseType::F;
  using typename BaseType::LocalAnsatzBasisType;
  using typename BaseType::LocalTestBasisType;

  BoundaryPenalty(std::function<double(double)> coefficient_of_h,
                  std::function<double(const I&)> face_diameter = default_face_diameter<I>())
    : BaseType({}, "NavierStokes::LocalIntegrands::BoundaryPenalty")
    , coefficient_of_h_(coefficient_of_h)
    , face_diameter_(face_diameter)
  {
  }

  BoundaryPenalty(const ThisType& other) = default;
  BoundaryPenalty(ThisType&& source) noexcept = default;

  std::unique_ptr<BaseType> copy_as_binary_intersection_integrand() const final
  {
    return std::make_unique<ThisType>(*this);
  }

  int order(const LocalTestBasisType& test_basis,
            const LocalAnsatzBasisType& ansatz_basis,
            const XT::Common::Parameter& param = {}) const final
  {
    return test_basis.order(param) + ansatz_basis.order(param);
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
    test_basis.evaluate(x_in, test_values_, param);
    ansatz_basis.evaluate(x_in, ansatz_values_, param);
    const double c = coefficient_of_h_(face_diameter_(intersection));
    const size_t rows = test_basis.size(param);
    const size_t cols = ansatz_basis.size(param);
    for (size_t ii = 0; ii < rows; ++ii)
      for (size_t jj = 0; jj < cols; ++jj)
        result[ii][jj] = c * ansatz_values_[jj][0] * test_values_[ii][0];
  }

private:
  const std::function<double(double)> coefficient_of_h_;
  const std::function<double(const I&)> face_diameter_;
  mutable std::vector<typename LocalTestBasisType::RangeType> test_values_;
  mutable std::vector<typename LocalAnsatzBasisType::RangeType> ansatz_values_;
}; // class BoundaryPenalty


/// \brief Boundary-face data integrand coefficient(h_F) * g_c v (right hand side of BoundaryPenalty).
template <class I>
class BoundaryPenaltyData : public LocalUnaryIntersectionIntegrandInterface<I>
{
  using ThisType = BoundaryPenaltyData;
  using BaseType = LocalUnaryIntersectionIntegrandInterface<I>;

public:
  using BaseType::d;
  using typename BaseType::DomainType;
  using typename BaseType::E;
  using typename BaseType::F;
  using typename BaseType::IntersectionType;
  using typename BaseType::LocalTestBasisType;
  using DataType = XT::Functions::GridFunctionInterface<E, d>;

  BoundaryPenaltyData(XT::Functions::GridFunction<E, d> data,
                      const size_t component,
                      std::function<double(double)> coefficient_of_h,
                      std::function<double(const I&)> face_diameter = default_face_diameter<I>())
    : BaseType(data.parameter_type(), "NavierStokes::LocalIntegrands::BoundaryPenaltyData")
    , data_(data.copy_as_grid_function())
    , local_data_(data_->local_function())
    , component_(component)
    , coefficient_of_h_(coefficient_of_h)
    , face_diameter_(face_diameter)
  {
  }

  BoundaryPenaltyData(const ThisType& other)
    : BaseType(other)
    , data_(other.data_->copy_as_grid_function())
    , local_data_(data_->local_function())
    , component_(other.component_)
    , coefficient_of_h_(other.coefficient_of_h_)
    , face_diameter_(other.face_diameter_)
  {
  }

  BoundaryPenaltyData(ThisType&& source) noexcept = default;

  std::unique_ptr<BaseType> copy_as_unary_intersection_integrand() const final
  {
    return std::make_unique<ThisType>(*this);
  }

protected:
  void post_bind(const IntersectionType& intrsctn) final
  {
    local_data_->bind(intrsctn.inside());
  }

public:
  int order(const LocalTestBasisType& test_basis, const XT::Common::Parameter& param = {}) const final
  {
    return local_data_->order(param) + test_basis.order(param);
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
    test_basis.evaluate(x_in, test_values_, param);
    const auto g = local_data_->evaluate(x_in, param);
    const double c = coefficient_of_h_(face_diameter_(intersection)) * g[component_];
    const size_t size = test_basis.size(param);
    for (size_t ii = 0; ii < size; ++ii)
      result[ii] = c * test_values_[ii][0];
  }

private:
  std::unique_ptr<DataType> data_;
  std::unique_ptr<typename DataType::LocalFunctionType> local_data_;
  const size_t component_;
  const std::function<double(double)> coefficient_of_h_;
  const std::function<double(const I&)> face_diameter_;
  mutable std::vector<typename LocalTestBasisType::RangeType> test_values_;
}; // class BoundaryPenaltyData


/// \brief Element source integrand f_c v for vector-valued data f (component c).
template <class E>
class ComponentSource : public LocalUnaryElementIntegrandInterface<E>
{
  using ThisType = ComponentSource;
  using BaseType = LocalUnaryElementIntegrandInterface<E>;

public:
  using BaseType::d;
  using typename BaseType::DomainType;
  using typename BaseType::F;
  using typename BaseType::LocalTestBasisType;
  using DataType = XT::Functions::GridFunctionInterface<E, d>;

  ComponentSource(XT::Functions::GridFunction<E, d> data, const size_t component, const double factor = 1.)
    : BaseType(data.parameter_type(), "NavierStokes::LocalIntegrands::ComponentSource")
    , data_(data.copy_as_grid_function())
    , local_data_(data_->local_function())
    , component_(component)
    , factor_(factor)
  {
  }

  ComponentSource(const ThisType& other)
    : BaseType(other)
    , data_(other.data_->copy_as_grid_function())
    , local_data_(data_->local_function())
    , component_(other.component_)
    , factor_(other.factor_)
  {
  }

  ComponentSource(ThisType&& source) noexcept = default;

  std::unique_ptr<BaseType> copy_as_unary_element_integrand() const final
  {
    return std::make_unique<ThisType>(*this);
  }

protected:
  void post_bind(const E& element) final
  {
    local_data_->bind(element);
  }

public:
  int order(const LocalTestBasisType& test_basis, const XT::Common::Parameter& param = {}) const final
  {
    return local_data_->order(param) + test_basis.order(param);
  }

  using BaseType::evaluate;

  void evaluate(const LocalTestBasisType& test_basis,
                const DomainType& point_in_reference_element,
                DynamicVector<F>& result,
                const XT::Common::Parameter& param = {}) const final
  {
    const size_t size = test_basis.size(param);
    if (result.size() < size)
      result.resize(size);
    result *= 0;
    test_basis.evaluate(point_in_reference_element, test_values_, param);
    const double c = factor_ * local_data_->evaluate(point_in_reference_element, param)[component_];
    for (size_t ii = 0; ii < size; ++ii)
      result[ii] = c * test_values_[ii][0];
  }

private:
  std::unique_ptr<DataType> data_;
  std::unique_ptr<typename DataType::LocalFunctionType> local_data_;
  const size_t component_;
  const double factor_;
  mutable std::vector<typename LocalTestBasisType::RangeType> test_values_;
}; // class ComponentSource


} // namespace LocalIntegrands
} // namespace NavierStokes
} // namespace GDT
} // namespace Dune

#endif // DUNE_LDG_NS_LOCAL_INTEGRANDS_LDG_HH
