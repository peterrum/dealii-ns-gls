#pragma once

#include <deal.II/base/conditional_ostream.h>

#include "config.h"
#include "operator_base.h"

using namespace dealii;

/**
 * Nonlinear solver.
 */
class NonLinearSolverBase
{
public:
  virtual void
  solve(VectorType &solution) const = 0;

  std::function<void(const VectorType &src)> setup_jacobian;

  std::function<void(const VectorType &src)> setup_preconditioner;

  std::function<void(VectorType &dst, const VectorType &src)> evaluate_residual;

  std::function<void(VectorType &dst)> evaluate_rhs;

  std::function<void(VectorType &dst, const VectorType &src)>
    solve_with_jacobian;

  std::function<void(const VectorType &dst)> postprocess;
};



/**
 * One step of fixed point iteration.
 */
class NonLinearSolverLinearized : public NonLinearSolverBase
{
public:
  NonLinearSolverLinearized();

  void
  solve(VectorType &solution) const override;

private:
};



/**
 * Basic Newton solver.
 */
class NonLinearSolverNewton : public NonLinearSolverBase
{
public:
  NonLinearSolverNewton();

  void
  solve(VectorType &solution) const override;

private:
  const ConditionalOStream pcout;

  const double       newton_tolerance;
  const unsigned int newton_max_iteration;
};



/**
 * Simple Picard fixed-point iteration solver.
 */
class NonLinearSolverPicardSimple : public NonLinearSolverBase
{
public:
  NonLinearSolverPicardSimple();

  void
  solve(VectorType &solution) const override;

private:
  const ConditionalOStream pcout;

  const double       picard_tolerance;
  const unsigned int picard_max_iteration;
};



/**
 * Advanced Picard fixed-point iteration solver.
 *
 * TODO: Not working yet.
 */
class NonLinearSolverPicard : public NonLinearSolverBase
{
public:
  NonLinearSolverPicard(const double theta);

  void
  solve(VectorType &solution) const override;

private:
  const ConditionalOStream pcout;

  mutable bool need_to_recompute_matrix;

  const double       picard_tolerance;
  const unsigned int picard_max_iteration;
  const bool         linear_always_recompute_matrix;
  const double       picard_reduction_ratio_admissible;
  const double       picard_reduction_ratio_recompute;
  const double       theta;
};
