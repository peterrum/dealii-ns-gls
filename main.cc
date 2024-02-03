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
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>

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
  evaluate_rhs(VectorType &dst, const VectorType &src) const = 0;

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
    op.evaluate_rhs(rhs, solution);

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
  using FECellIntegrator = FEEvaluation<dim, -1, 0, dim + 1, Number>;

  NavierStokesOperator(
    const Mapping<dim>              &mapping,
    const DoFHandler<dim>           &dof_handler,
    const AffineConstraints<Number> &constraints_homogeneous,
    const AffineConstraints<Number> &constraints,
    const AffineConstraints<Number> &constraints_inhomogeneous,
    const Quadrature<dim>           &quadrature)
    : constraints_inhomogeneous(constraints_inhomogeneous)
  {
    const std::vector<const DoFHandler<dim> *> mf_dof_handlers = {&dof_handler,
                                                                  &dof_handler};
    const std::vector<const AffineConstraints<Number> *> mf_constraints = {
      &constraints_homogeneous, &constraints};

    typename MatrixFree<dim, Number>::AdditionalData additional_data;

    additional_data.mapping_update_flags = update_values | update_gradients;

    matrix_free.reinit(
      mapping, mf_dof_handlers, mf_constraints, quadrature, additional_data);
  }

  void
  set_linearization_point(const VectorType &vec) override
  {
    this->linearization_point = vec;
  }

  void
  evaluate_rhs(VectorType &dst, const VectorType &src) const override
  {
    // compute right-hand side
    this->matrix_free.cell_loop(
      &NavierStokesOperator<dim>::do_rhs_range, this, dst, src, true);

    // apply inhomogeneous DBC
    VectorType dst_temp, src_temp;
    dst_temp.reinit(dst);
    src_temp.reinit(src);

    constraints_inhomogeneous.distribute(dst_temp);

    this->matrix_free.cell_loop(
      &NavierStokesOperator<dim>::do_vmult_range<false>,
      this,
      dst_temp,
      src_temp,
      true);

    constraints_inhomogeneous.set_zero(dst_temp);

    dst -= dst_temp;
  }

  void
  vmult(VectorType &dst, const VectorType &src) const override
  {
    this->matrix_free.cell_loop(
      &NavierStokesOperator<dim>::do_vmult_range<true>, this, dst, src, true);
  }

  const SparseMatrixType &
  get_system_matrix() const override
  {
    initialize_system_matrix();

    return system_matrix;
  }

  void
  initialize_dof_vector(VectorType &vec) const override
  {
    matrix_free.initialize_dof_vector(vec);
  }

private:
  const AffineConstraints<Number> &constraints_inhomogeneous;

  MatrixFree<dim, Number> matrix_free;

  VectorType               linearization_point;
  mutable SparseMatrixType system_matrix;

  template <bool homogeneous>
  void
  do_vmult_range(const MatrixFree<dim, Number>               &matrix_free,
                 VectorType                                  &dst,
                 const VectorType                            &src,
                 const std::pair<unsigned int, unsigned int> &range) const
  {
    FECellIntegrator phi(matrix_free, homogeneous ? 0 : 1);

    for (auto cell = range.first; cell < range.second; ++cell)
      {
        phi.reinit(cell);
        phi.read_dof_values(src);

        do_vmult_cell(phi);

        phi.distribute_local_to_global(dst);
      }
  }

  void
  do_vmult_cell(FECellIntegrator &phi) const
  {
    phi.evaluate(EvaluationFlags::EvaluationFlags::values |
                 EvaluationFlags::EvaluationFlags::gradients);

    // TODO

    phi.integrate(EvaluationFlags::EvaluationFlags::values |
                  EvaluationFlags::EvaluationFlags::gradients);
  }

  void
  do_rhs_range(const MatrixFree<dim, Number>               &matrix_free,
               VectorType                                  &dst,
               const VectorType                            &src,
               const std::pair<unsigned int, unsigned int> &range) const
  {
    FECellIntegrator phi(matrix_free);

    for (auto cell = range.first; cell < range.second; ++cell)
      {
        phi.reinit(cell);
        phi.read_dof_values(src);

        do_rhs_cell(phi);

        phi.distribute_local_to_global(dst);
      }
  }

  void
  do_rhs_cell(FECellIntegrator &phi) const
  {
    phi.evaluate(EvaluationFlags::EvaluationFlags::values |
                 EvaluationFlags::EvaluationFlags::gradients);

    // TODO

    phi.integrate(EvaluationFlags::EvaluationFlags::values |
                  EvaluationFlags::EvaluationFlags::gradients);
  }

  void
  initialize_system_matrix() const
  {
    const bool system_matrix_is_empty =
      system_matrix.m() == 0 || system_matrix.n() == 0;

    const auto &dof_handler = matrix_free.get_dof_handler();
    const auto &constraints = matrix_free.get_affine_constraints();

    if (system_matrix_is_empty)
      {
        system_matrix.clear();

        TrilinosWrappers::SparsityPattern dsp;

        dsp.reinit(dof_handler.locally_owned_dofs(),
                   dof_handler.get_communicator());
        DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
        dsp.compress();

        system_matrix.reinit(dsp);
      }

    {
      if (system_matrix_is_empty == false)
        system_matrix = 0.0; // clear existing content

      MatrixFreeTools::compute_matrix(matrix_free,
                                      constraints,
                                      system_matrix,
                                      &NavierStokesOperator<dim>::do_vmult_cell,
                                      this);
    }
  }
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
    NavierStokesOperator<dim> ns_operator(mapping,
                                          dof_handler,
                                          constraints_homogeneous,
                                          constraints,
                                          constraints_inhomogeneous,
                                          quadrature);

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