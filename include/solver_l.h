#pragma once

#include <deal.II/base/conditional_ostream.h>

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
  virtual void
  initialize() = 0;

  virtual void
  solve(VectorType &dst, const VectorType &src) const = 0;
};



/**
 * Wrapper class around dealii::GMRES.
 */
class LinearSolverGMRES : public LinearSolverBase
{
public:
  LinearSolverGMRES(const OperatorBase &op,
                    PreconditionerBase &preconditioner,
                    const unsigned int  n_max_iterations,
                    const double        absolute_tolerance,
                    const double        relative_tolerance);

  void
  initialize() override;

  void
  solve(VectorType &dst, const VectorType &src) const override;

private:
  const OperatorBase &op;
  PreconditionerBase &preconditioner;

  const unsigned int n_max_iterations;
  const double       absolute_tolerance;
  const double       relative_tolerance;

  const ConditionalOStream pcout;

  mutable MyTimerOutput timer;
};