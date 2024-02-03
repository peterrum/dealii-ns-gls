#include <deal.II/base/mpi.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

using namespace dealii;

using Number           = double;
using VectorType       = LinearAlgebra::distributed::Vector<Number>;
using SparseMatrixType = TrilinosWrappers::SparseMatrix;

/**
 * Linear/nonlinear operator.
 */
class OperatorBase
{
public:
  virtual void
  set_linearization_point(const VectorType &src) = 0;

  virtual void
  evaluate_rhs(const VectorType &src) const = 0;

  virtual void
  vmult(VectorType &dst, const VectorType &src) const = 0;

  virtual const SparseMatrixType &
  get_system_matrix() const = 0;
};



template <int dim>
class NavierStokesOperator : public OperatorBase
{
public:
  void
  set_linearization_point(const VectorType &src) override
  {
    (void)src;
  }

  void
  evaluate_rhs(const VectorType &src) const override
  {
    (void)src;
  }

  void
  vmult(VectorType &dst, const VectorType &src) const override
  {
    (void)dst;
    (void)src;
  }

  const SparseMatrixType &
  get_system_matrix() const override
  {
    AssertThrow(false, ExcNotImplemented());

    return system_matrix;
  }

private:
  SparseMatrixType system_matrix;
};



/**
 * Preconditioners.
 */
class PreconditionerBase
{
public:
  virtual void
  initialize() = 0;

  virtual void
  vmult(VectorType &dst, const VectorType &src) const = 0;
};



class PreconditionerILU : public PreconditionerBase
{
public:
  PreconditionerILU(const OperatorBase &op)
    : op(op)
  {}

  void
  initialize() override
  {
    const auto &matrix = op.get_system_matrix();
    precon.initialize(matrix);
  }

  void
  vmult(VectorType &dst, const VectorType &src) const override
  {
    precon.vmult(dst, src);
  }

private:
  const OperatorBase &op;

  TrilinosWrappers::PreconditionILU precon;
};



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



class LinearSolverGMRES : public LinearSolverBase
{
public:
  LinearSolverGMRES(const OperatorBase &op, PreconditionerBase &preconditioner)
    : op(op)
    , preconditioner(preconditioner)
  {}

  void
  initialize() override
  {
    preconditioner.initialize();
  }

  void
  solve(VectorType &dst, const VectorType &src) const override
  {
    ReductionControl solver_control;

    SolverGMRES<VectorType> solver(solver_control);

    solver.solve(op, dst, src, preconditioner);
  }

private:
  const OperatorBase &op;
  PreconditionerBase &preconditioner;
};



/**
 * Nonlinear solver.
 */
class NonLinearSolverBase
{
public:
  virtual void
  solve(VectorType &solution) const = 0;
};



class NonLinearSolverLinearized : public NonLinearSolverBase
{
public:
  NonLinearSolverLinearized(OperatorBase &op, LinearSolverBase &linear_solver)
    : op(op)
    , linear_solver(linear_solver)
  {}

  void
  solve(VectorType &solution) const override
  {
    // set linearization point
    op.set_linearization_point(solution);

    // compute right-hans-side vector
    VectorType rhs;
    rhs.reinit(solution);
    op.evaluate_rhs(rhs);

    // solve linear system
    linear_solver.initialize();
    linear_solver.solve(solution, solution);
  }

private:
  OperatorBase     &op;
  LinearSolverBase &linear_solver;
};



template <int dim>
class Driver
{
public:
  void
  run()
  {
    // set up system

    // set up Navier-Stokes operator
    NavierStokesOperator<dim> ns_operator;

    // set up preconditioner
    std::shared_ptr<PreconditionerBase> preconditioner;

    preconditioner = std::make_shared<PreconditionerILU>(ns_operator);

    // set up linear solver
    std::shared_ptr<LinearSolverBase> linear_solver;

    linear_solver =
      std::make_shared<LinearSolverGMRES>(ns_operator, *preconditioner);

    // set up nonlinear solver
    std::shared_ptr<NonLinearSolverBase> nonlinear_solver;

    nonlinear_solver =
      std::make_shared<NonLinearSolverLinearized>(ns_operator, *linear_solver);

    VectorType solution;

    // initialize solution

    // perform time loop
    for (;;)
      {
        nonlinear_solver->solve(solution);
      }
  }

private:
};



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const int dim = 2;

  Driver<dim> driver;
  driver.run();
}