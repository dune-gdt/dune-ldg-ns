// This file is part of the dune-ldg-ns project:
//   https://github.com/dune-gdt/dune-ldg-ns
// Copyright 2026 dune-ldg-ns developers and contributors. All rights reserved.
// License: Dual licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
//      or  GPL-2.0+ (http://opensource.org/licenses/gpl-license)
//          with "runtime exception" (http://www.dune-project.org/license.html)
// Authors:
//   René Fritze (2026)

/**
 * Grid-free checks of the FS-theta substep coefficients: algebraic identities, second order for a stiff linear ODE
 * with time-dependent forcing (the forcing is weighted like the operator), strong A-stability (|R(-inf)| < 1).
 */
#include "config.h"

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include <dune/ldg-ns/timestepping/fractional-step-theta.hh>

using namespace Dune::GDT::NavierStokes;


TEST(FractionalStepTheta, coefficient_identities)
{
  using C = FractionalStepThetaCoefficients;
  EXPECT_NEAR(C::theta(), 1. - std::sqrt(2.) / 2., 1e-15);
  EXPECT_NEAR(2. * C::theta() + C::theta_prime(), 1., 1e-15);
  EXPECT_NEAR(C::alpha() + C::beta(), 1., 1e-15);
  // identical implicit operator in all substeps
  EXPECT_NEAR(C::alpha() * C::theta(), C::beta() * C::theta_prime(), 1e-15);
  const double K = 0.37;
  double total = 0.;
  for (const auto& s : fractional_step_theta_substeps(K)) {
    EXPECT_NEAR(s.c_imp + s.c_exp, s.tau, 1e-15);
    EXPECT_NEAR(s.c_imp, C::alpha() * C::theta() * K, 1e-15);
    total += s.tau;
  }
  EXPECT_NEAR(total, K, 1e-15);
}


namespace {

// y' = -lambda y + f(t), exact y = sin t + cos 2t, one macro step of a theta-type scheme
double integrate(std::vector<ThetaSubstep> (*substeps)(double), const double lambda, const size_t N)
{
  const auto y_exact = [](double t) { return std::sin(t) + std::cos(2. * t); };
  const auto f = [&](double t) { return std::cos(t) - 2. * std::sin(2. * t) + lambda * y_exact(t); };
  const double K = 1. / N;
  double y = y_exact(0.);
  double t = 0.;
  for (size_t n = 0; n < N; ++n)
    for (const auto& s : substeps(K)) {
      y = (y - s.c_exp * lambda * y + s.c_imp * f(t + s.tau) + s.c_exp * f(t)) / (1. + s.c_imp * lambda);
      t += s.tau;
    }
  return std::abs(y - y_exact(1.));
}

std::vector<ThetaSubstep> fs_theta(double K)
{
  return fractional_step_theta_substeps(K);
}

std::vector<ThetaSubstep> backward_euler(double K)
{
  return one_step_theta_substeps(K, 1.);
}

} // namespace


TEST(FractionalStepTheta, second_order_with_forcing)
{
  for (const double lambda : {3., 300.}) {
    const double e1 = integrate(&fs_theta, lambda, 40);
    const double e2 = integrate(&fs_theta, lambda, 80);
    EXPECT_GT(std::log2(e1 / e2), 1.9) << "lambda = " << lambda;
  }
  const double b1 = integrate(&backward_euler, 3., 40);
  const double b2 = integrate(&backward_euler, 3., 80);
  EXPECT_NEAR(std::log2(b1 / b2), 1., 0.1);
}


TEST(FractionalStepTheta, strongly_a_stable)
{
  // amplification factor of one macro step for y' = -lambda y, K lambda -> infinity
  const double z = 1e8;
  double R = 1.;
  for (const auto& s : fractional_step_theta_substeps(1.))
    R *= (1. - s.c_exp * z) / (1. + s.c_imp * z);
  // limit: (-beta/alpha) (-alpha/beta) (-beta/alpha) = -beta/alpha = -(sqrt(2) - 1) / (2 - sqrt(2)) = -sqrt(2)/2
  EXPECT_LT(std::abs(R), 0.72);
  EXPECT_NEAR(std::abs(R), FractionalStepThetaCoefficients::beta() / FractionalStepThetaCoefficients::alpha(), 1e-6);
}


int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
