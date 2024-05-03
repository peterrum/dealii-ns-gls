#include "solver_nl.h"

#include <deal.II/lac/affine_constraints.h>



NonLinearSolverLinearized::NonLinearSolverLinearized()
{}

void
NonLinearSolverLinearized::solve(VectorType<Number> &solution) const
{
  // set linearization point
  this->setup_jacobian(solution);

  // compute right-hans-side vector
  VectorType<Number> rhs;
  rhs.reinit(solution);
  this->evaluate_rhs(rhs);

  // solve linear system
  this->setup_preconditioner(solution);
  this->solve_with_jacobian(solution, rhs);
}



NonLinearSolverNewton::NonLinearSolverNewton(const bool inexact_newton)
  : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  , newton_tolerance(1.0e-7) // TODO
  , newton_max_iteration(30) // TODO
  , inexact_newton(inexact_newton)
{}

void
NonLinearSolverNewton::solve(VectorType<Number> &solution) const
{
  MyScope scope(timer, "newton::solve");

  VectorType<Number> rhs, inc;
  rhs.reinit(solution);
  inc.reinit(solution);

  // set linearization point
  this->setup_jacobian(solution);

  // compute right-hans-side vector
  this->evaluate_residual(rhs, solution);

  double       l2_norm       = rhs.l2_norm();
  unsigned int num_iteration = 0;

  pcout << "    [N] step " << num_iteration << "; residual = " << l2_norm
        << std::endl;

  while (l2_norm > newton_tolerance)
    {
      inc = 0.0;

      if (num_iteration == 0 || (inexact_newton == false))
        this->setup_preconditioner(solution);
      this->solve_with_jacobian(inc, rhs);

      solution.add(1.0, inc);

      if (postprocess)
        this->postprocess(solution);

      // set linearization point
      this->setup_jacobian(solution);

      // compute right-hans-side vector
      this->evaluate_residual(rhs, solution);

      // check convergence
      l2_norm = rhs.l2_norm();
      num_iteration++;

      pcout << "    [N] step " << num_iteration << " ; residual = " << l2_norm
            << std::endl;

      AssertThrow(num_iteration <= newton_max_iteration,
                  dealii::ExcMessage(
                    "Newton iteration did not converge. Final residual_0 is " +
                    std::to_string(l2_norm) + "."));
    }

  pcout << "    [N] solved in " << num_iteration << " iterations." << std::endl;
}



NonLinearSolverPicard::NonLinearSolverPicard()
  : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  , picard_tolerance(1.0e-7) // TODO
  , picard_max_iteration(30) // TODO
{}

void
NonLinearSolverPicard::solve(VectorType<Number> &solution) const
{
  double       l2_norm       = 1e10;
  unsigned int num_iteration = 0;

  VectorType<Number> rhs, tmp;
  rhs.reinit(solution);
  tmp.reinit(solution);

  while (l2_norm > picard_tolerance)
    {
      tmp = solution;

      // set linearization point
      this->setup_jacobian(solution);

      // compute right-hans-side vector
      this->evaluate_rhs(rhs);

      // solve linear system
      this->setup_preconditioner(solution);

      this->solve_with_jacobian(solution, rhs);

      // check convergence
      tmp -= solution;
      l2_norm = tmp.l2_norm();
      num_iteration++;

      AssertThrow(num_iteration <= picard_max_iteration,
                  dealii::ExcMessage(
                    "Picard iteration did not converge. Final residual_0 is " +
                    std::to_string(l2_norm) + "."));
    }

  pcout << "    [P] solved in " << num_iteration << " iterations." << std::endl;
}
