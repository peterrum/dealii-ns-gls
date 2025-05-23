#include "solver_l.h"

#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_richardson.h>

LinearSolverDirect::LinearSolverDirect(const OperatorBase<Number> &op)
  : op(op)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
{}

void
LinearSolverDirect::initialize()
{
  MyScope scope(timer, "direct::initialize");
  solver.initialize(op.get_system_matrix());
}

void
LinearSolverDirect::solve(VectorType<Number>       &dst,
                          const VectorType<Number> &src) const
{
  MyScope scope(timer, "direct::solve");
  solver.solve(dst, src);
}

LinearSolverGMRES::LinearSolverGMRES(const OperatorBase<Number> &op,
                                     PreconditionerBase         &preconditioner,
                                     const unsigned int n_max_iterations,
                                     const double       absolute_tolerance,
                                     const double       relative_tolerance)
  : op(op)
  , preconditioner(preconditioner)
  , n_max_iterations(n_max_iterations)
  , absolute_tolerance(absolute_tolerance)
  , relative_tolerance(relative_tolerance)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
{}

void
LinearSolverGMRES::initialize()
{
  // nothing to do
}

void
LinearSolverGMRES::solve(VectorType<Number>       &dst,
                         const VectorType<Number> &src) const
{
  MyScope scope(timer, "gmres::solve");

  const double linear_solver_tolerance =
    std::max(relative_tolerance * src.l2_norm(), absolute_tolerance);

  SolverControl solver_control(n_max_iterations,
                               linear_solver_tolerance,
                               true,
                               true);

  typename SolverGMRES<VectorType<Number>>::AdditionalData solver_parameters;

  solver_parameters.max_n_tmp_vectors     = 30; // TODO
  solver_parameters.right_preconditioning = true;

  SolverGMRES<VectorType<Number>> solver(solver_control, solver_parameters);

  dst = 0.0;

  solver.solve(op, dst, src, preconditioner);

  pcout << "    [L] solved in " << solver_control.last_step() << " iterations."
        << std::endl;

  preconditioner.print_stats();
}

LinearSolverRichardson::LinearSolverRichardson(
  const OperatorBase<Number> &op,
  PreconditionerBase         &preconditioner,
  const unsigned int          n_max_iterations,
  const double                absolute_tolerance,
  const double                relative_tolerance)
  : op(op)
  , preconditioner(preconditioner)
  , n_max_iterations(n_max_iterations)
  , absolute_tolerance(absolute_tolerance)
  , relative_tolerance(relative_tolerance)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
{}

void
LinearSolverRichardson::initialize()
{
  // nothing to do
}

void
LinearSolverRichardson::solve(VectorType<Number>       &dst,
                              const VectorType<Number> &src) const
{
  MyScope scope(timer, "richardson::solve");

  const double linear_solver_tolerance =
    std::max(relative_tolerance * src.l2_norm(), absolute_tolerance);

  SolverControl solver_control(n_max_iterations,
                               linear_solver_tolerance,
                               true,
                               true);

  SolverRichardson<VectorType<Number>> solver(solver_control);

  dst = 0.0;

  solver.solve(op, dst, src, preconditioner);

  pcout << "    [L] solved in " << solver_control.last_step() << " iterations."
        << std::endl;

  preconditioner.print_stats();
}
