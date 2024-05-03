#pragma once

#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/trilinos_solver.h>

#include "config.h"
#include "operator_base.h"
#include "preconditioner.h"
#include "timer.h"

using namespace dealii;

/**
 * Linear solvers.
 */
class LinearSolverBase
{
public:
  virtual ~LinearSolverBase() = default;

  virtual void
  initialize() = 0;

  virtual void
  solve(VectorType<Number> &dst, const VectorType<Number> &src) const = 0;
};



/**
 * Wrapper class around dealii::TrilinosWrappers::SolverDirect.
 */
class LinearSolverDirect : public LinearSolverBase
{
public:
  LinearSolverDirect(const OperatorBase<Number> &op);

  void
  initialize() override;

  void
  solve(VectorType<Number> &dst, const VectorType<Number> &src) const override;

private:
  const OperatorBase<Number>            &op;
  mutable TrilinosWrappers::SolverDirect solver;

  const ConditionalOStream pcout;

  mutable MyTimerOutput timer;
};



/**
 * Wrapper class around dealii::SolverGMRES.
 */
class LinearSolverGMRES : public LinearSolverBase
{
public:
  LinearSolverGMRES(const OperatorBase<Number> &op,
                    PreconditionerBase         &preconditioner,
                    const unsigned int          n_max_iterations,
                    const double                absolute_tolerance,
                    const double                relative_tolerance);

  void
  initialize() override;

  void
  solve(VectorType<Number> &dst, const VectorType<Number> &src) const override;

private:
  const OperatorBase<Number> &op;
  PreconditionerBase         &preconditioner;

  const unsigned int n_max_iterations;
  const double       absolute_tolerance;
  const double       relative_tolerance;

  const ConditionalOStream pcout;

  mutable MyTimerOutput timer;
};
