#pragma once

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
  NonLinearSolverLinearized()
  {}

  void
  solve(VectorType &solution) const override
  {
    // set linearization point
    this->setup_jacobian(solution);

    // compute right-hans-side vector
    VectorType rhs;
    rhs.reinit(solution);
    this->evaluate_rhs(rhs);

    // solve linear system
    this->setup_preconditioner(solution);
    this->solve_with_jacobian(solution, rhs);
  }

private:
};



/**
 * Basic Newton solver.
 */
class NonLinearSolverNewton : public NonLinearSolverBase
{
public:
  NonLinearSolverNewton()
    : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    , newton_tolerance(1.0e-7) // TODO
    , newton_max_iteration(30) // TODO
  {}

  void
  solve(VectorType &solution) const override
  {
    VectorType rhs, inc;
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

        AssertThrow(
          num_iteration <= newton_max_iteration,
          dealii::ExcMessage(
            "Newton iteration did not converge. Final residual_0 is " +
            std::to_string(l2_norm) + "."));
      }

    pcout << "    [N] solved in " << num_iteration << " iterations."
          << std::endl;
  }

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
  NonLinearSolverPicardSimple()
    : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    , picard_tolerance(1.0e-7) // TODO
    , picard_max_iteration(30) // TODO
  {}

  void
  solve(VectorType &solution) const override
  {
    double       l2_norm       = 1e10;
    unsigned int num_iteration = 0;

    VectorType rhs, tmp;
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

        AssertThrow(
          num_iteration <= picard_max_iteration,
          dealii::ExcMessage(
            "Picard iteration did not converge. Final residual_0 is " +
            std::to_string(l2_norm) + "."));
      }

    pcout << "    [P] solved in " << num_iteration << " iterations."
          << std::endl;
  }

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
  NonLinearSolverPicard(const double theta)
    : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    , need_to_recompute_matrix(true)
    , picard_tolerance(1.0e-7)                  // TODO
    , picard_max_iteration(30)                  // TODO
    , linear_always_recompute_matrix(false)     // TODO
    , picard_reduction_ratio_admissible(1.0e-1) // TODO
    , picard_reduction_ratio_recompute(1.0e-2)  // TODO
    , theta(theta)                              // TODO
  {}

  void
  solve(VectorType &solution) const override
  {
    VectorType residual_0, tmp, update;
    residual_0.reinit(solution);
    tmp.reinit(solution);
    update.reinit(solution);

    // copy vector
    tmp = solution;

    // set linearization point
    this->setup_jacobian(tmp);

    // compute residual_0
    this->evaluate_residual(residual_0, tmp);

    double linfty_norm = residual_0.l2_norm();

    pcout << "    [P] initial; residual_0 (linfty) = " << linfty_norm
          << std::endl;

    for (unsigned int i = 1; linfty_norm > picard_tolerance; ++i)
      {
        if (i > picard_max_iteration)
          {
            const auto error_string =
              std::string(
                "Picard iteration did not converge. Final residual_0 is ") +
              std::to_string(linfty_norm);
            AssertThrow(false, dealii::ExcMessage(error_string));
          }

        if (need_to_recompute_matrix || linear_always_recompute_matrix)
          {
            pcout << "    [P] step " << i << " ; recompute matrix" << std::endl;
            this->setup_preconditioner(tmp);
            need_to_recompute_matrix = false;
          }

        this->solve_with_jacobian(update, residual_0);
        tmp += update;

        // compute residual_0
        this->setup_jacobian(tmp);
        this->evaluate_residual(residual_0, tmp);
        double new_linfty_norm = residual_0.l2_norm();

        if (new_linfty_norm < picard_tolerance)
          {
            // accept new step
            linfty_norm = new_linfty_norm;
            pcout << "    [P] step " << i
                  << " ; residual_0 (linfty) = " << linfty_norm << std::endl;
          }
        else if (new_linfty_norm >
                 picard_reduction_ratio_admissible * linfty_norm)
          {
            // revert to previous step
            tmp -= update;
            this->setup_jacobian(tmp);

            need_to_recompute_matrix = true;

            pcout << "    [P] step " << i
                  << " ; inadmissible residual_0 reduction (" << new_linfty_norm
                  << " > " << linfty_norm << " ), recompute matrix\n"
                  << std::endl;
          }
        else if (new_linfty_norm >
                 picard_reduction_ratio_recompute * linfty_norm)
          {
            // accept new step
            linfty_norm = new_linfty_norm;
            pcout << "    [P] step " << i
                  << " ; residual_0 (linfty) = " << linfty_norm
                  << ", recompute matrix" << std::endl;
            need_to_recompute_matrix = true;
          }
        else
          {
            // accept new step
            linfty_norm = new_linfty_norm;
            pcout << "    [P] step " << i
                  << " ; residual_0 (linfty) = " << linfty_norm << std::endl;
          }
      }

    tmp *= 1. / theta;
    tmp.add((theta - 1.) / theta, solution);

    solution = tmp;
  }

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
