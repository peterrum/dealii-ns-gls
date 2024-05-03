#pragma once

#include <deal.II/base/conditional_ostream.h>

#include "config.h"
#include "operator_base.h"
#include "timer.h"

using namespace dealii;

/**
 * Nonlinear solver.
 */
class NonLinearSolverBase
{
public:
  virtual ~NonLinearSolverBase() = default;

  virtual void
  solve(VectorType<Number> &solution) const = 0;

  std::function<void(const VectorType<Number> &src)> setup_jacobian;

  std::function<void(const VectorType<Number> &src)> setup_preconditioner;

  std::function<void(VectorType<Number> &dst, const VectorType<Number> &src)>
    evaluate_residual;

  std::function<void(VectorType<Number> &dst)> evaluate_rhs;

  std::function<void(VectorType<Number> &dst, const VectorType<Number> &src)>
    solve_with_jacobian;

  std::function<void(const VectorType<Number> &dst)> postprocess;
};



/**
 * One step of fixed point iteration.
 */
class NonLinearSolverLinearized : public NonLinearSolverBase
{
public:
  NonLinearSolverLinearized();

  void
  solve(VectorType<Number> &solution) const override;

private:
};



/**
 * Basic Newton solver.
 */
class NonLinearSolverNewton : public NonLinearSolverBase
{
public:
  NonLinearSolverNewton(const bool inexact_newton);

  void
  solve(VectorType<Number> &solution) const override;

private:
  const ConditionalOStream pcout;

  const double       newton_tolerance;
  const unsigned int newton_max_iteration;
  const bool         inexact_newton;

  mutable MyTimerOutput timer;
};



/**
 * Simple Picard fixed-point iteration solver.
 */
class NonLinearSolverPicard : public NonLinearSolverBase
{
public:
  NonLinearSolverPicard();

  void
  solve(VectorType<Number> &solution) const override;

private:
  const ConditionalOStream pcout;

  const double       picard_tolerance;
  const unsigned int picard_max_iteration;
};
