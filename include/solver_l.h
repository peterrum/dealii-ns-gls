#pragma once

#include "config.h"
#include "operator_base.h"

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
                    const double        relative_tolerance)
    : op(op)
    , preconditioner(preconditioner)
    , n_max_iterations(n_max_iterations)
    , absolute_tolerance(absolute_tolerance)
    , relative_tolerance(relative_tolerance)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {}

  void
  initialize() override
  {
    // nothing to do
  }

  void
  solve(VectorType &dst, const VectorType &src) const override
  {
    const double linear_solver_tolerance =
      std::max(relative_tolerance * src.l2_norm(), absolute_tolerance);

    SolverControl solver_control(n_max_iterations,
                                 linear_solver_tolerance,
                                 true,
                                 true);

    typename SolverGMRES<VectorType>::AdditionalData solver_parameters;

    solver_parameters.max_n_tmp_vectors     = 30; // TODO
    solver_parameters.right_preconditioning = true;

    SolverGMRES<VectorType> solver(solver_control, solver_parameters);

    dst = 0.0;

    solver.solve(op, dst, src, preconditioner);

    pcout << "    [L] solved in " << solver_control.last_step()
          << " iterations." << std::endl;
  }

private:
  const OperatorBase &op;
  PreconditionerBase &preconditioner;

  const unsigned int n_max_iterations;
  const double       absolute_tolerance;
  const double       relative_tolerance;

  const ConditionalOStream pcout;
};