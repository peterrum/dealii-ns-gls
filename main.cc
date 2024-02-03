#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/vector_tools.h>

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

  virtual void
  initialize_dof_vector(VectorType &src) const = 0;
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



/**
 * Navier-Stokes operator.
 */
template <int dim>
class NavierStokesOperator : public OperatorBase
{
public:
  void
  set_linearization_point(const VectorType &src) override
  {
    AssertThrow(false, ExcNotImplemented());

    (void)src;
  }

  void
  evaluate_rhs(const VectorType &src) const override
  {
    AssertThrow(false, ExcNotImplemented());

    (void)src;
  }

  void
  vmult(VectorType &dst, const VectorType &src) const override
  {
    AssertThrow(false, ExcNotImplemented());

    (void)dst;
    (void)src;
  }

  const SparseMatrixType &
  get_system_matrix() const override
  {
    AssertThrow(false, ExcNotImplemented());

    return system_matrix;
  }

  void
  initialize_dof_vector(VectorType &src) const override
  {
    AssertThrow(false, ExcNotImplemented());

    (void)src;
  }

private:
  SparseMatrixType system_matrix;
};



template <int dim>
class Driver
{
public:
  void
  run()
  {
    const unsigned int fe_degree      = 2;
    const unsigned int mapping_degree = 1;

    const MPI_Comm comm = MPI_COMM_WORLD;

    // set up system
    parallel::distributed::Triangulation<dim> tria(comm);

    // TODO: create mesh

    FESystem<dim> fe(FE_Q<dim>(fe_degree), 1 + dim);

    DoFHandler<dim> dof_handler(tria);
    dof_handler.distribute_dofs(fe);

    QGauss<dim> quadrature(fe_degree + 1);

    MappingQ<dim> mapping(mapping_degree);

    std::vector<unsigned int> all_homogeneous_dbcs;
    std::vector<std::pair<unsigned int, std::shared_ptr<Function<dim>>>>
      all_inhomogeneous_dbcs;

    // set up constraints
    ComponentMask mask_v(dim + 1, true);
    mask_v.set(dim, false);

    ComponentMask mask_p(dim + 1, false);
    mask_p.set(dim, true);

    AffineConstraints<Number> constraints;

    for (const auto bci : all_homogeneous_dbcs)
      DoFTools::make_zero_boundary_constraints(dof_handler,
                                               bci,
                                               constraints,
                                               mask_v);

    AffineConstraints<Number> constraints_homogeneous;
    constraints_homogeneous.copy_from(constraints);

    for (const auto &[bci, _] : all_inhomogeneous_dbcs)
      DoFTools::make_zero_boundary_constraints(dof_handler,
                                               bci,
                                               constraints_homogeneous,
                                               mask_v);

    constraints.close();
    constraints_homogeneous.close();

    AffineConstraints<Number> constraints_inhomogeneous;
    // note: filled during time loop

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

    // initialize solution
    VectorType solution;
    ns_operator.initialize_dof_vector(solution);

    // perform time loop
    for (;;)
      {
        // set time-dependent inhomogeneous DBCs
        constraints_inhomogeneous.clear();
        for (const auto &[bci, fu] : all_inhomogeneous_dbcs)
          VectorTools::interpolate_boundary_values(
            mapping, dof_handler, bci, *fu, constraints_inhomogeneous, mask_v);
        constraints_inhomogeneous.close();

        // solve nonlinear problem
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