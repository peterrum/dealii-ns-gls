#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>

#include "include/config.h"
#include "include/grid_cylinder.h"
#include "include/grid_cylinder_old.h"
#include "include/time_integration.h"

using namespace dealii;



/**
 * Linear/nonlinear operator.
 */
class OperatorBase : public Subscriptor
{
public:
  using value_type = Number;
  using size_type  = types::global_dof_index;

  virtual types::global_dof_index
  m() const = 0;

  Number
  el(unsigned int, unsigned int) const
  {
    Assert(false, ExcNotImplemented());
    return 0;
  }

  virtual void
  compute_inverse_diagonal(VectorType &diagonal) const = 0;

  virtual void
  invalidate_system() = 0;

  virtual void
  set_previous_solution(const SolutionHistory &vec) = 0;

  virtual void
  set_previous_solution(const VectorType &vec) = 0;

  virtual void
  set_linearization_point(const VectorType &src) = 0;

  virtual void
  evaluate_rhs(VectorType &dst) const = 0;

  virtual void
  evaluate_residual(VectorType &dst, const VectorType &src) const = 0;

  virtual void
  vmult(VectorType &dst, const VectorType &src) const = 0;

  void
  Tvmult(VectorType &dst, const VectorType &src) const
  {
    vmult(dst, src);
  }

  virtual void
  vmult_interface_down(VectorType &dst, const VectorType &src) const
  {
    AssertThrow(false, ExcNotImplemented());

    (void)dst;
    (void)src;
  }

  virtual void
  vmult_interface_up(VectorType &dst, const VectorType &src) const
  {
    AssertThrow(false, ExcNotImplemented());

    (void)dst;
    (void)src;
  }

  virtual std::vector<std::vector<bool>>
  extract_constant_modes() const
  {
    AssertThrow(false, ExcNotImplemented());
    return {};
  }

  virtual const AffineConstraints<Number> &
  get_constraints() const = 0;

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

    const int    current_preconditioner_fill_level = 0;
    const double ilu_atol                          = 1e-12;
    const double ilu_rtol                          = 1.00;
    TrilinosWrappers::PreconditionILU::AdditionalData ad(
      current_preconditioner_fill_level, ilu_atol, ilu_rtol, 0);

    precon.initialize(matrix, ad);
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



class PreconditionerAMG : public PreconditionerBase
{
public:
  PreconditionerAMG(const OperatorBase                   &op,
                    const std::vector<std::vector<bool>> &constant_modes)
    : op(op)
    , constant_modes(constant_modes)
  {}

  void
  initialize() override
  {
    const auto &matrix = op.get_system_matrix();

    typename TrilinosWrappers::PreconditionAMG::AdditionalData ad;

    ad.elliptic              = false;          // TODO
    ad.higher_order_elements = false;          //
    ad.n_cycles              = 1;              //
    ad.aggregation_threshold = 1e-14;          //
    ad.constant_modes        = constant_modes; //
    ad.smoother_sweeps       = 2;              //
    ad.smoother_overlap      = 1;              //
    ad.output_details        = false;          //
    ad.smoother_type         = "ILU";          //
    ad.coarse_type           = "ILU";          //

    precon.initialize(matrix, ad);
  }

  void
  vmult(VectorType &dst, const VectorType &src) const override
  {
    precon.vmult(dst, src);
  }

private:
  const OperatorBase &op;

  const std::vector<std::vector<bool>> constant_modes;

  TrilinosWrappers::PreconditionAMG precon;
};


#include "include/multigrid.h"


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



/**
 * Matrix-free Navier-Stokes operator.
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
    const Quadrature<dim>           &quadrature,
    const Number                     nu,
    const Number                     c_1,
    const Number                     c_2,
    const TimeIntegratorData        &time_integrator_data,
    const bool                       consider_time_deriverative,
    const bool                       increment_form,
    const bool                       cell_wise_stabilization,
    const unsigned int               mg_level = numbers::invalid_unsigned_int)
    : constraints_inhomogeneous(constraints_inhomogeneous)
    , theta(time_integrator_data.get_theta())
    , nu(nu)
    , c_1(c_1)
    , c_2(c_2)
    , time_integrator_data(time_integrator_data)
    , consider_time_deriverative(consider_time_deriverative)
    , increment_form(increment_form)
    , cell_wise_stabilization(cell_wise_stabilization)
    , valid_system(false)
  {
    const std::vector<const DoFHandler<dim> *> mf_dof_handlers = {&dof_handler,
                                                                  &dof_handler};
    const std::vector<const AffineConstraints<Number> *> mf_constraints = {
      &constraints_homogeneous, &constraints};

    typename MatrixFree<dim, Number>::AdditionalData additional_data;

    additional_data.mapping_update_flags = update_values | update_gradients;
    additional_data.mg_level             = mg_level;

    matrix_free.reinit(
      mapping, mf_dof_handlers, mf_constraints, quadrature, additional_data);

    for (auto i : this->matrix_free.get_constrained_dofs())
      constrained_indices.push_back(i);

    if (consider_time_deriverative)
      {
        AssertThrow(theta[0] == 1.0, ExcInternalError());
      }

    if (this->matrix_free.get_mg_level() != numbers::invalid_unsigned_int)
      {
        std::vector<types::global_dof_index> interface_indices;
        IndexSet                             refinement_edge_indices;
        refinement_edge_indices = get_refinement_edges(this->matrix_free);
        refinement_edge_indices.fill_index_vector(interface_indices);

        edge_constrained_indices.clear();
        edge_constrained_indices.reserve(interface_indices.size());
        edge_constrained_values.resize(interface_indices.size());
        const IndexSet &locally_owned =
          this->matrix_free.get_dof_handler().locally_owned_mg_dofs(
            this->matrix_free.get_mg_level());
        for (unsigned int i = 0; i < interface_indices.size(); ++i)
          if (locally_owned.is_element(interface_indices[i]))
            edge_constrained_indices.push_back(
              locally_owned.index_within_set(interface_indices[i]));

        this->has_edge_constrained_indices =
          Utilities::MPI::max(edge_constrained_indices.size(),
                              dof_handler.get_communicator()) > 0;
      }
  }

  const AffineConstraints<Number> &
  get_constraints() const override
  {
    return matrix_free.get_affine_constraints();
  }

  virtual types::global_dof_index
  m() const override
  {
    if (this->matrix_free.get_mg_level() != numbers::invalid_unsigned_int)
      return this->matrix_free.get_dof_handler().n_dofs(
        this->matrix_free.get_mg_level());
    else
      return this->matrix_free.get_dof_handler().n_dofs();
  }

  std::vector<std::vector<bool>>
  extract_constant_modes() const override
  {
    std::vector<std::vector<bool>> constant_modes;

    if (this->matrix_free.get_mg_level() != numbers::invalid_unsigned_int)
      return constant_modes; // TODO

    ComponentMask components(dim + 1, true);
    DoFTools::extract_constant_modes(this->matrix_free.get_dof_handler(),
                                     components,
                                     constant_modes);

    return constant_modes;
  }

  virtual void
  compute_inverse_diagonal(VectorType &diagonal) const override
  {
    matrix_free.initialize_dof_vector(diagonal);
    MatrixFreeTools::compute_diagonal(
      matrix_free,
      diagonal,
      &NavierStokesOperator<dim>::do_vmult_cell<false>,
      this);

    for (unsigned int i = 0; i < edge_constrained_indices.size(); ++i)
      diagonal.local_element(edge_constrained_indices[i]) = 0.0;

    for (auto &i : diagonal)
      i = (std::abs(i) > 1.0e-10) ? (1.0 / i) : 1.0;
  }

  void
  invalidate_system() override
  {
    this->valid_system = false;
  }

  void
  set_previous_solution(const SolutionHistory &history) override
  {
    this->valid_system = false;

    const unsigned n_cells             = matrix_free.n_cell_batches();
    const unsigned n_quadrature_points = matrix_free.get_quadrature().size();

    u_time_derivative_old.reinit(n_cells, n_quadrature_points);

    FEEvaluation<dim, -1, 0, dim, Number> integrator(matrix_free);

    VectorType vec_old;
    vec_old.reinit(history.get_vectors()[1]);

    for (unsigned int i = 1; i <= time_integrator_data.get_order(); ++i)
      vec_old.add(time_integrator_data.get_weights()[i],
                  history.get_vectors()[i]);

    vec_old.update_ghost_values();

    for (unsigned int cell = 0; cell < n_cells; ++cell)
      {
        integrator.reinit(cell);

        integrator.read_dof_values_plain(vec_old);
        integrator.evaluate(EvaluationFlags::EvaluationFlags::values);

        for (const auto q : integrator.quadrature_point_indices())
          u_time_derivative_old[cell][q] = integrator.get_value(q);
      }

    this->set_previous_solution(history.get_vectors()[1]);
  }

  void
  set_previous_solution(const VectorType &vec) override
  {
    this->valid_system = false;

    const unsigned fe_degree =
      matrix_free.get_dof_handler().get_fe().tensor_degree();
    const unsigned n_cells             = matrix_free.n_cell_batches();
    const unsigned n_quadrature_points = matrix_free.get_quadrature().size();

    u_old_gradient.reinit(n_cells, n_quadrature_points);
    p_old_gradient.reinit(n_cells, n_quadrature_points);

    delta_1.resize(n_cells);
    delta_2.resize(n_cells);

    delta_1_q.reinit(n_cells, n_quadrature_points);
    delta_2_q.reinit(n_cells, n_quadrature_points);

    FEEvaluation<dim, -1, 0, dim, Number> integrator(matrix_free);
    FEEvaluation<dim, -1, 0, 1, Number>   integrator_scalar(matrix_free,
                                                          0,
                                                          0,
                                                          dim);

    const bool has_ghost_elements = vec.has_ghost_elements();

    AssertThrow(has_ghost_elements == false, ExcInternalError());

    if (has_ghost_elements == false)
      vec.update_ghost_values();

    const auto tau = this->time_integrator_data.get_current_dt();

    for (unsigned int cell = 0; cell < n_cells; ++cell)
      {
        integrator.reinit(cell);

        integrator.read_dof_values_plain(vec);
        integrator.evaluate(EvaluationFlags::EvaluationFlags::values |
                            EvaluationFlags::EvaluationFlags::gradients);

        integrator_scalar.reinit(cell);
        integrator_scalar.read_dof_values_plain(vec);
        integrator_scalar.evaluate(EvaluationFlags::EvaluationFlags::gradients);

        // precompute value/gradient of linearization point at quadrature points
        for (const auto q : integrator.quadrature_point_indices())
          {
            u_old_gradient[cell][q] = integrator.get_gradient(q);

            p_old_gradient[cell][q] = integrator_scalar.get_gradient(q);
          }

        // compute stabilization parameters (cell-wise)
        VectorizedArray<Number> u_max = 0.0;
        for (const auto q : integrator.quadrature_point_indices())
          u_max = std::max(integrator.get_value(q).norm(), u_max);

        for (unsigned int v = 0;
             v < matrix_free.n_active_entries_per_cell_batch(cell);
             ++v)
          {
            const auto cell_it = matrix_free.get_cell_iterator(cell, v);
            const auto h       = cell_it->minimum_vertex_distance();

            if (nu[0] < h)
              {
                delta_1[cell][v] =
                  c_1 /
                  std::sqrt(1. / (tau * tau) + u_max[v] * u_max[v] / (h * h));
                delta_2[cell][v] = c_2 * h;
              }
            else
              {
                delta_1[cell][v] = c_1 * h * h;
                delta_2[cell][v] = c_2 * h * h;
              }
          }

        // compute stabilization parameters (q-point-wise)
        // adopted from:
        // https://github.com/lethe-cfd/lethe/blob/d8e115f175e34628e96243fce6eec00d7fcaf3c1/source/solvers/mf_navier_stokes_operators.cc#L222-L246
        // https://github.com/lethe-cfd/lethe/blob/d8e115f175e34628e96243fce6eec00d7fcaf3c1/source/solvers/mf_navier_stokes_operators.cc#L657-L668
        VectorizedArray<Number> h;
        for (unsigned int v = 0;
             v < matrix_free.n_active_entries_per_cell_batch(cell);
             ++v)
          {
            const double h_k =
              matrix_free.get_cell_iterator(cell, v)->measure();

            if (dim == 2)
              h[v] = std::sqrt(4. * h_k / M_PI) / fe_degree;
            else if (dim == 3)
              h[v] = std::pow(6 * h_k / M_PI, 1. / 3.) / fe_degree;
          }

        for (const auto q : integrator.quadrature_point_indices())
          {
            VectorizedArray<Number> u_mag_squared = 1e-12;
            for (unsigned int k = 0; k < dim; ++k)
              u_mag_squared +=
                Utilities::fixed_power<2>(integrator.get_value(q)[k]);

            delta_1_q[cell][q] =
              1. / std::sqrt(Utilities::fixed_power<2>(1.0 / tau) +
                             4. * u_mag_squared / h / h +
                             9. * Utilities::fixed_power<2>(4. * nu / (h * h)));

            delta_2_q[cell][q] = std::sqrt(u_mag_squared) * h * 0.5;
          }
      }

    if (has_ghost_elements == false)
      vec.zero_out_ghost_values();
  }

  void
  set_linearization_point(const VectorType &vec) override
  {
    this->valid_system = false;

    const unsigned n_cells             = matrix_free.n_cell_batches();
    const unsigned n_quadrature_points = matrix_free.get_quadrature().size();

    u_star_value.reinit(n_cells, n_quadrature_points);
    u_star_gradient.reinit(n_cells, n_quadrature_points);
    p_star_gradient.reinit(n_cells, n_quadrature_points);

    FEEvaluation<dim, -1, 0, dim, Number> integrator(matrix_free);
    FEEvaluation<dim, -1, 0, 1, Number>   integrator_scalar(matrix_free,
                                                          0,
                                                          0,
                                                          dim);

    const bool has_ghost_elements = vec.has_ghost_elements();

    AssertThrow(has_ghost_elements == false, ExcInternalError());

    if (has_ghost_elements == false)
      vec.update_ghost_values();

    for (unsigned int cell = 0; cell < n_cells; ++cell)
      {
        integrator.reinit(cell);
        integrator.read_dof_values_plain(vec);
        integrator.evaluate(EvaluationFlags::EvaluationFlags::values |
                            EvaluationFlags::EvaluationFlags::gradients);

        integrator_scalar.reinit(cell);
        integrator_scalar.read_dof_values_plain(vec);
        integrator_scalar.evaluate(EvaluationFlags::EvaluationFlags::gradients);

        for (const auto q : integrator.quadrature_point_indices())
          {
            u_star_value[cell][q]    = integrator.get_value(q);
            u_star_gradient[cell][q] = integrator.get_gradient(q);
            p_star_gradient[cell][q] = integrator_scalar.get_gradient(q);
          }
      }

    if (has_ghost_elements == false)
      vec.zero_out_ghost_values();
  }

  void
  evaluate_rhs(VectorType &dst) const override
  {
    // apply inhomogeneous DBC
    VectorType src;
    src.reinit(dst);
    constraints_inhomogeneous.distribute(src);

    // perform vmult
    this->matrix_free.cell_loop(
      &NavierStokesOperator<dim>::do_vmult_range<true>, this, dst, src, true);

    // apply constraints
    matrix_free.get_affine_constraints(0).set_zero(dst);

    // move to rhs
    dst *= -1.0;
  }

  virtual void
  evaluate_residual(VectorType &dst, const VectorType &src) const override
  {
    // apply inhomogeneous DBC
    VectorType tmp = src;                      // TODO: needed?
    constraints_inhomogeneous.distribute(tmp); //

    this->matrix_free.cell_loop(
      &NavierStokesOperator<dim>::do_vmult_range<true>, this, dst, tmp, true);

    // apply constraints
    matrix_free.get_affine_constraints(0).set_zero(dst);

    // move to rhs
    dst *= -1.0;
  }

  void
  vmult(VectorType &dst, const VectorType &src) const override
  {
    // save values for edge constrained dofs and set them to 0 in src vector
    for (unsigned int i = 0; i < edge_constrained_indices.size(); ++i)
      {
        edge_constrained_values[i] = std::pair<Number, Number>(
          src.local_element(edge_constrained_indices[i]),
          dst.local_element(edge_constrained_indices[i]));

        const_cast<LinearAlgebra::distributed::Vector<Number> &>(src)
          .local_element(edge_constrained_indices[i]) = 0.;
      }

    this->matrix_free.cell_loop(
      &NavierStokesOperator<dim>::do_vmult_range<false>, this, dst, src, true);

    for (unsigned int i = 0; i < constrained_indices.size(); ++i)
      dst.local_element(constrained_indices[i]) =
        src.local_element(constrained_indices[i]);

    // restoring edge constrained dofs in src and dst
    for (unsigned int i = 0; i < edge_constrained_indices.size(); ++i)
      {
        const_cast<LinearAlgebra::distributed::Vector<Number> &>(src)
          .local_element(edge_constrained_indices[i]) =
          edge_constrained_values[i].first;
        dst.local_element(edge_constrained_indices[i]) =
          edge_constrained_values[i].first;
      }
  }

  void
  vmult_interface_down(VectorType &dst, const VectorType &src) const override
  {
    this->matrix_free.cell_loop(
      &NavierStokesOperator<dim>::do_vmult_range<false>, this, dst, src, true);

    // set constrained dofs as the sum of current dst value and src value
    for (unsigned int i = 0; i < constrained_indices.size(); ++i)
      dst.local_element(constrained_indices[i]) =
        src.local_element(constrained_indices[i]);
  }

  void
  vmult_interface_up(VectorType &dst, const VectorType &src) const override
  {
    if (has_edge_constrained_indices == false)
      {
        dst = Number(0.);
        return;
      }

    dst = 0.0;

    // make a copy of src vector and set everything to 0 except edge
    // constrained dofs
    VectorType src_cpy;
    src_cpy.reinit(src, /*omit_zeroing_entries=*/false);

    for (unsigned int i = 0; i < edge_constrained_indices.size(); ++i)
      src_cpy.local_element(edge_constrained_indices[i]) =
        src.local_element(edge_constrained_indices[i]);

    // do loop with copy of src
    this->matrix_free.cell_loop(
      &NavierStokesOperator<dim>::do_vmult_range<false>,
      this,
      dst,
      src_cpy,
      false);
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

  const VectorizedArray<Number> theta;
  const VectorizedArray<Number> nu;
  const Number                  c_1;
  const Number                  c_2;
  const TimeIntegratorData     &time_integrator_data;
  const bool                    consider_time_deriverative;
  const bool                    increment_form;
  const bool                    cell_wise_stabilization;

  mutable bool valid_system;

  AlignedVector<VectorizedArray<Number>> delta_1;
  AlignedVector<VectorizedArray<Number>> delta_2;

  Table<2, VectorizedArray<Number>> delta_1_q;
  Table<2, VectorizedArray<Number>> delta_2_q;

  Table<2, Tensor<1, dim, VectorizedArray<Number>>> u_star_value;
  Table<2, Tensor<2, dim, VectorizedArray<Number>>> u_star_gradient;
  Table<2, Tensor<1, dim, VectorizedArray<Number>>> p_star_gradient;

  Table<2, Tensor<1, dim, VectorizedArray<Number>>> u_time_derivative_old;
  Table<2, Tensor<2, dim, VectorizedArray<Number>>> u_old_gradient;
  Table<2, Tensor<1, dim, VectorizedArray<Number>>> p_old_gradient;

  std::vector<unsigned int> constrained_indices;

  template <bool evaluate_residual>
  void
  do_vmult_range(const MatrixFree<dim, Number>               &matrix_free,
                 VectorType                                  &dst,
                 const VectorType                            &src,
                 const std::pair<unsigned int, unsigned int> &range) const
  {
    FECellIntegrator phi(matrix_free, 0);

    for (auto cell = range.first; cell < range.second; ++cell)
      {
        phi.reinit(cell);

        if (evaluate_residual)
          phi.read_dof_values_plain(src);
        else
          phi.read_dof_values(src);

        do_vmult_cell<evaluate_residual>(phi);

        phi.distribute_local_to_global(dst);
      }
  }

  /**
   * Fixed-point system:
   *
   * (v, ∂t(u)) + (v, S⋅∇B) - (div(v), p) + (ε(v), νε(B))
   *            + δ_1 (S⋅∇v, ∂t(u) + ∇P + S⋅∇B) + δ_2 (div(v), div(B)) = 0
   *              +----------- SUPG ----------+   +------- GD -------+
   *
   * (q, div(B)) + δ_1 (∇q, ∂t(u) + ∇p + S⋅∇B) = 0
   *               +---------- PSPG ---------+
   *
   * with the following nomenclature:
   *  - S     := u^*
   *  - B     := θ u^{n+1} + (1-θ) u^{n}
   *  - P     := θ p^{n+1} + (1-θ) p^{n}
   *  - p     := p^{n+1}
   *  - ∂t(u) := time deriverative (one-step-theta method, BDF)
   *
   *
   * Linearized system (only BDF):
   *
   * (v, ∂t'(u) + U⋅∇u + u⋅∇U) - (div(v), p) + (ε(v), νε(u))
   *            + δ_1 (U⋅∇v, ∂t'(u) + U⋅∇u + u⋅∇U + ∇p) -> SUPG (1)
   *            + δ_1 (u⋅∇v, ∂t'(U) + U⋅∇U + ∇P)        -> SUPG (2)
   *            + δ_2 (div(v), div(u))                  -> GD
   *
   * (q, div(u)) + δ_1 (∇q, ∂t'(u) + U⋅∇u + u⋅∇U + ∇p)
   *               +-------------- PSPG -------------+
   *
   *                       ... with U/P being the linearization point
   */
  template <bool evaluate_residual>
  void
  do_vmult_cell(FECellIntegrator &integrator) const
  {
    if (evaluate_residual || !this->increment_form)
      {
        const unsigned int cell = integrator.get_current_cell_index();
        const auto weight = this->time_integrator_data.get_primary_weight();
        const auto theta  = this->theta;
        const auto nu     = this->nu;

        integrator.evaluate(EvaluationFlags::EvaluationFlags::values |
                            EvaluationFlags::EvaluationFlags::gradients);

        for (const auto q : integrator.quadrature_point_indices())
          {
            typename FECellIntegrator::value_type    value_result    = {};
            typename FECellIntegrator::gradient_type gradient_result = {};

            const auto value    = integrator.get_value(q);
            const auto gradient = integrator.get_gradient(q);

            const auto delta_1 = cell_wise_stabilization ?
                                   this->delta_1[cell] :
                                   this->delta_1_q[cell][q];
            const auto delta_2 = cell_wise_stabilization ?
                                   this->delta_2[cell] :
                                   this->delta_2_q[cell][q];

            const VectorizedArray<Number>                 p_value = value[dim];
            const Tensor<1, dim, VectorizedArray<Number>> p_gradient =
              gradient[dim];
            Tensor<1, dim, VectorizedArray<Number>> p_bar_gradient =
              theta * gradient[dim];

            const Tensor<1, dim, VectorizedArray<Number>> u_star_value =
              this->u_star_value[cell][q];
            Tensor<1, dim, VectorizedArray<Number>> u_time_derivative;
            Tensor<2, dim, VectorizedArray<Number>> u_bar_gradient;

            for (unsigned int d = 0; d < dim; ++d)
              {
                u_time_derivative[d] = value[d] * weight;
                u_bar_gradient[d]    = theta * gradient[d];
              }

            if (evaluate_residual)
              u_time_derivative += this->u_time_derivative_old[cell][q];

            if (evaluate_residual && (theta[0] != 1.0))
              {
                u_bar_gradient += (VectorizedArray<Number>(1.0) - theta) *
                                  this->u_old_gradient[cell][q];
                p_bar_gradient += (VectorizedArray<Number>(1.0) - theta) *
                                  this->p_old_gradient[cell][q];
              }

            // precompute: div(B)
            VectorizedArray<Number> div_bar = u_bar_gradient[0][0];
            for (unsigned int d = 1; d < dim; ++d)
              div_bar += u_bar_gradient[d][d];

            // precompute: S⋅∇B
            const auto s_grad_b = u_bar_gradient * u_star_value;

            // velocity block:
            //  a)  (v, ∂t(u))
            for (unsigned int d = 0; d < dim; ++d)
              value_result[d] = u_time_derivative[d];

            //  b)  (v, S⋅∇B)
            for (unsigned int d = 0; d < dim; ++d)
              value_result[d] += s_grad_b[d];

            //  c)  - (div(v), p)
            for (unsigned int d = 0; d < dim; ++d)
              gradient_result[d][d] -= p_value;

            //  d)  (ε(v), νε(B))
            for (unsigned int d = 0; d < dim; ++d)
              gradient_result[d][d] += u_bar_gradient[d][d] * nu;

            for (unsigned int e = 0, counter = dim; e < dim; ++e)
              for (unsigned int d = e + 1; d < dim; ++d, ++counter)
                {
                  const auto tmp =
                    (u_bar_gradient[d][e] + u_bar_gradient[e][d]) * (nu * 0.5);
                  gradient_result[d][e] += tmp;
                  gradient_result[e][d] += tmp;
                }

            //  e)  δ_1 (S⋅∇v, ∂t(u) + ∇P + S⋅∇B) -> SUPG stabilization
            const auto residual_0 =
              delta_1 * ((consider_time_deriverative ?
                            u_time_derivative :
                            Tensor<1, dim, VectorizedArray<Number>>()) +
                         p_bar_gradient + s_grad_b);
            for (unsigned int d0 = 0; d0 < dim; ++d0)
              for (unsigned int d1 = 0; d1 < dim; ++d1)
                gradient_result[d0][d1] += u_star_value[d1] * residual_0[d0];

            //  f) δ_2 (div(v), div(B)) -> GD stabilization
            for (unsigned int d = 0; d < dim; ++d)
              gradient_result[d][d] += delta_2 * div_bar;



            // pressure block:
            //  a)  (q, div(B))
            value_result[dim] = div_bar;

            //  b)  δ_1 (∇q, ∂t(u) + ∇p + S⋅∇B) -> PSPG stabilization
            gradient_result[dim] =
              delta_1 * ((consider_time_deriverative ?
                            u_time_derivative :
                            Tensor<1, dim, VectorizedArray<Number>>()) +
                         p_gradient + s_grad_b);


            integrator.submit_value(value_result, q);
            integrator.submit_gradient(gradient_result, q);
          }

        integrator.integrate(EvaluationFlags::EvaluationFlags::values |
                             EvaluationFlags::EvaluationFlags::gradients);
      }
    else
      {
        const unsigned int cell = integrator.get_current_cell_index();

        const auto weight = this->time_integrator_data.get_primary_weight();
        const auto nu     = this->nu;

        integrator.evaluate(EvaluationFlags::EvaluationFlags::values |
                            EvaluationFlags::EvaluationFlags::gradients);

        for (const auto q : integrator.quadrature_point_indices())
          {
            typename FECellIntegrator::value_type    value_result    = {};
            typename FECellIntegrator::gradient_type gradient_result = {};

            const auto value    = integrator.get_value(q);
            const auto gradient = integrator.get_gradient(q);

            const auto delta_1 = cell_wise_stabilization ?
                                   this->delta_1[cell] :
                                   this->delta_1_q[cell][q];
            const auto delta_2 = cell_wise_stabilization ?
                                   this->delta_2[cell] :
                                   this->delta_2_q[cell][q];

            const VectorizedArray<Number>                 p_value = value[dim];
            const Tensor<1, dim, VectorizedArray<Number>> p_gradient =
              gradient[dim];
            const Tensor<1, dim, VectorizedArray<Number>> p_star_gradient =
              this->p_star_gradient[cell][q];

            const Tensor<1, dim, VectorizedArray<Number>> u_star_value =
              this->u_star_value[cell][q];
            const Tensor<2, dim, VectorizedArray<Number>> u_star_gradient =
              this->u_star_gradient[cell][q];
            Tensor<1, dim, VectorizedArray<Number>> u_time_derivative;
            Tensor<1, dim, VectorizedArray<Number>> u_value;
            Tensor<2, dim, VectorizedArray<Number>> u_gradient;

            for (unsigned int d = 0; d < dim; ++d)
              {
                u_time_derivative[d] = value[d] * weight;
                u_value[d]           = value[d];
                u_gradient[d]        = gradient[d];
              }

            // precompute: div(u)
            VectorizedArray<Number> div_u = u_gradient[0][0];
            for (unsigned int d = 1; d < dim; ++d)
              div_u += u_gradient[d][d];

            // precompute: U⋅∇u
            const auto s_grad_u = u_gradient * u_star_value;

            // precompute: u⋅∇U
            const auto u_grad_s = u_star_gradient * u_value;

            // precompute: U⋅∇U
            const auto s_grad_s = u_star_gradient * u_star_value;

            // velocity block:
            //  a)  (v, ∂t'(u) + U⋅∇u + u⋅∇U)
            for (unsigned int d = 0; d < dim; ++d)
              value_result[d] =
                u_time_derivative[d] + s_grad_u[d] + u_grad_s[d];

            //  b)  - (div(v), p)
            for (unsigned int d = 0; d < dim; ++d)
              gradient_result[d][d] -= p_value;

            //  c)  (ε(v), νε(u))
            for (unsigned int d = 0; d < dim; ++d)
              gradient_result[d][d] += u_gradient[d][d] * nu;
            for (unsigned int e = 0, counter = dim; e < dim; ++e)
              for (unsigned int d = e + 1; d < dim; ++d, ++counter)
                {
                  const auto tmp =
                    (u_gradient[d][e] + u_gradient[e][d]) * (nu * 0.5);
                  gradient_result[d][e] += tmp;
                  gradient_result[e][d] += tmp;
                }

            //  d)  δ_1 (U⋅∇v, ∂t'(u) + ∇p + U⋅∇u + u⋅∇U) +
            //      δ_1 (u⋅∇v, ∂t'(U) + ∇P + U⋅∇U) -> SUPG stabilization
            const auto residual_0 =
              delta_1 * ((consider_time_deriverative ?
                            u_time_derivative :
                            Tensor<1, dim, VectorizedArray<Number>>()) +
                         p_gradient + s_grad_u + u_grad_s);
            const auto residual_1 =
              delta_1 * ((consider_time_deriverative ?
                            (u_star_value * weight +
                             this->u_time_derivative_old[cell][q]) :
                            Tensor<1, dim, VectorizedArray<Number>>()) +
                         p_star_gradient + s_grad_s);
            for (unsigned int d0 = 0; d0 < dim; ++d0)
              for (unsigned int d1 = 0; d1 < dim; ++d1)
                gradient_result[d0][d1] += u_star_value[d1] * residual_0[d0] +
                                           u_value[d1] * residual_1[d0];

            //  e)  δ_2 (div(v), div(u)) -> GD stabilization
            for (unsigned int d = 0; d < dim; ++d)
              gradient_result[d][d] += delta_2 * div_u;



            // pressure block:
            //  a)  (q, div(u))
            value_result[dim] = div_u;

            //  b)  δ_1 (∇q, ∂t'(u) + ∇p + U⋅∇u + u⋅∇U) -> PSPG stabilization
            gradient_result[dim] =
              delta_1 * ((consider_time_deriverative ?
                            u_time_derivative :
                            Tensor<1, dim, VectorizedArray<Number>>()) +
                         p_gradient + s_grad_u + u_grad_s);


            integrator.submit_value(value_result, q);
            integrator.submit_gradient(gradient_result, q);
          }

        integrator.integrate(EvaluationFlags::EvaluationFlags::values |
                             EvaluationFlags::EvaluationFlags::gradients);
      }
  }

  void
  initialize_system_matrix() const
  {
    const auto &dof_handler = matrix_free.get_dof_handler();
    const auto &constraints = matrix_free.get_affine_constraints();

    if (system_matrix.m() == 0 || system_matrix.n() == 0)
      {
        system_matrix.clear();

        TrilinosWrappers::SparsityPattern dsp;

        dsp.reinit(this->matrix_free.get_mg_level() !=
                       numbers::invalid_unsigned_int ?
                     dof_handler.locally_owned_mg_dofs(
                       this->matrix_free.get_mg_level()) :
                     dof_handler.locally_owned_dofs(),
                   dof_handler.get_communicator());

        if (this->matrix_free.get_mg_level() != numbers::invalid_unsigned_int)
          MGTools::make_sparsity_pattern(dof_handler,
                                         dsp,
                                         this->matrix_free.get_mg_level(),
                                         constraints);
        else
          DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);

        dsp.compress();

        system_matrix.reinit(dsp);
      }

    if (this->valid_system == false)
      {
        system_matrix = 0.0;

        MatrixFreeTools::compute_matrix(
          matrix_free,
          constraints,
          system_matrix,
          &NavierStokesOperator<dim>::do_vmult_cell<false>,
          this);

        this->valid_system = true;
      }
  }

  std::vector<unsigned int> edge_constrained_indices;

  bool has_edge_constrained_indices = false;

  mutable std::vector<std::pair<Number, Number>> edge_constrained_values;

  std::vector<bool> edge_constrained_cell;

  static IndexSet
  get_refinement_edges(const MatrixFree<dim, Number> &matrix_free)
  {
    const unsigned int level = matrix_free.get_mg_level();

    std::vector<IndexSet> refinement_edge_indices;
    refinement_edge_indices.clear();
    const unsigned int nlevels =
      matrix_free.get_dof_handler().get_triangulation().n_global_levels();
    refinement_edge_indices.resize(nlevels);
    for (unsigned int l = 0; l < nlevels; l++)
      refinement_edge_indices[l] =
        IndexSet(matrix_free.get_dof_handler().n_dofs(l));

    MGTools::extract_inner_interface_dofs(matrix_free.get_dof_handler(),
                                          refinement_edge_indices);
    return refinement_edge_indices[level];
  }
};



/**
 * Matrix-based Navier-Stokes operator.
 */
template <int dim>
class NavierStokesOperatorMatrixBased : public OperatorBase
{
public:
  NavierStokesOperatorMatrixBased(
    const Mapping<dim>              &mapping,
    const DoFHandler<dim>           &dof_handler,
    const AffineConstraints<Number> &constraints,
    const Quadrature<dim>           &quadrature,
    const Number                     nu,
    const Number                     c_1,
    const Number                     c_2,
    const TimeIntegratorData        &time_integrator_data)
    : mapping(mapping)
    , dof_handler(dof_handler)
    , constraints(constraints)
    , quadrature(quadrature)
    , theta(time_integrator_data.get_theta())
    , nu(nu)
    , c_1(c_1)
    , c_2(c_2)
    , time_integrator_data(time_integrator_data)
    , valid_system(false)
  {
    this->partitioner = std::make_shared<Utilities::MPI::Partitioner>(
      dof_handler.locally_owned_dofs(),
      DoFTools::extract_locally_active_dofs(dof_handler),
      dof_handler.get_communicator());

    // initialize system vector
    this->initialize_dof_vector(system_rhs);

    // initialize system matrix
    TrilinosWrappers::SparsityPattern dsp;

    dsp.reinit(dof_handler.locally_owned_dofs(),
               dof_handler.get_communicator());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
    dsp.compress();

    system_matrix.reinit(dsp);
  }

  const AffineConstraints<Number> &
  get_constraints() const override
  {
    return constraints;
  }

  types::global_dof_index
  m() const override
  {
    AssertThrow(false, ExcNotImplemented());

    return 0;
  }

  void
  compute_inverse_diagonal(VectorType &) const override
  {
    AssertThrow(false, ExcNotImplemented());
  }

  void
  invalidate_system() override
  {
    this->valid_system = false;
  }

  void
  set_previous_solution(const SolutionHistory &vec) override
  {
    this->set_previous_solution(vec.get_vectors()[1]);
  }

  void
  set_previous_solution(const VectorType &vec) override
  {
    this->previous_solution = vec;
    this->previous_solution.update_ghost_values();

    this->valid_system = false;
  }

  void
  set_linearization_point(const VectorType &src) override
  {
    this->linearization_point = src;
    this->linearization_point.update_ghost_values();

    this->valid_system = false;
  }

  void
  evaluate_rhs(VectorType &dst) const override
  {
    compute_system_matrix_and_vector();
    dst = system_rhs;
  }

  virtual void
  evaluate_residual(VectorType &dst, const VectorType &src) const override
  {
    (void)dst;
    (void)src;
  }

  void
  vmult(VectorType &dst, const VectorType &src) const override
  {
    get_system_matrix().vmult(dst, src);
  }

  const SparseMatrixType &
  get_system_matrix() const override
  {
    compute_system_matrix_and_vector();
    return system_matrix;
  }

  void
  initialize_dof_vector(VectorType &src) const override
  {
    src.reinit(partitioner);
  }

private:
  const Mapping<dim>              &mapping;
  const DoFHandler<dim>           &dof_handler;
  const AffineConstraints<Number> &constraints;
  const Quadrature<dim>           &quadrature;
  const Number                     theta;
  const Number                     nu;
  const Number                     c_1;
  const Number                     c_2;
  const TimeIntegratorData        &time_integrator_data;

  mutable bool valid_system;

  VectorType previous_solution;
  VectorType linearization_point;

  mutable SparseMatrixType system_matrix;
  mutable VectorType       system_rhs;

  std::shared_ptr<Utilities::MPI::Partitioner> partitioner;

  void
  compute_system_matrix_and_vector() const
  {
    if (valid_system)
      return;

    valid_system = true;

    system_matrix = 0.;
    system_rhs    = 0.;

    const auto &finite_element = this->dof_handler.get_fe();

    FEValues<dim>              fe_values(finite_element,
                            quadrature,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
    FEValuesViews::Vector<dim> velocities(fe_values, 0);
    FEValuesViews::Scalar<dim> pressure(fe_values, dim);

    const unsigned int dofs_per_cell = finite_element.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature.size();

    FullMatrix<double> cell_contribution(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs_contribution(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<dealii::Tensor<1, dim>> u_0(n_q_points);
    std::vector<dealii::Tensor<1, dim>> u_star(n_q_points);
    std::vector<dealii::Tensor<2, dim>> grad_u_0(n_q_points);
    std::vector<double>                 div_u_0(n_q_points);
    std::vector<double>                 p_0(n_q_points);
    std::vector<dealii::Tensor<1, dim>> grad_p_0(n_q_points);

    const auto &Vu_0    = this->previous_solution;
    const auto &Vu_star = this->linearization_point;

    const auto tau = time_integrator_data.get_current_dt();

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        cell_contribution     = 0.;
        cell_rhs_contribution = 0.;

        fe_values.reinit(cell);

        velocities.get_function_values(Vu_0, u_0);
        velocities.get_function_values(Vu_star, u_star);
        velocities.get_function_divergences(Vu_0, div_u_0);

        velocities.get_function_gradients(Vu_0, grad_u_0);
        pressure.get_function_gradients(Vu_0, grad_p_0);
        pressure.get_function_values(Vu_0, p_0);

        const double h = cell->minimum_vertex_distance();

        const double u_max   = std::accumulate(u_0.begin(),
                                             u_0.end(),
                                             0.,
                                             [](const auto m, const auto u) {
                                               return std::max(m, u.norm());
                                             });
        double       delta_1 = 0.0;
        double       delta_2 = 0.0;
        if (nu < h)
          {
            delta_1 =
              c_1 / std::sqrt(1. / (tau * tau) + u_max * u_max / (h * h));
            delta_2 = c_2 * h;
          }
        else
          {
            delta_1 = c_1 * h * h;
            delta_2 = c_2 * h * h;
          }

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            const auto JxW = fe_values.JxW(q);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                // test space
                const auto v_i      = velocities.value(i, q);
                const auto grad_v_i = velocities.gradient(i, q);
                const auto div_v_i  = velocities.divergence(i, q);
                const auto eps_v_i  = velocities.symmetric_gradient(i, q);
                const auto q_i      = pressure.value(i, q);
                const auto grad_q_i = pressure.gradient(i, q);

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    // ansatz space
                    const auto u_j      = velocities.value(j, q);
                    const auto grad_u_j = velocities.gradient(j, q);
                    const auto div_u_j  = velocities.divergence(j, q);
                    const auto eps_u_j  = velocities.symmetric_gradient(j, q);
                    const auto p_j      = pressure.value(j, q);
                    const auto grad_p_j = pressure.gradient(j, q);

                    Number cell_lhs = 0.0;

                    // clang-format off
                    // velocity
                    cell_lhs += u_j * v_i;                                                                          // a
                    cell_lhs += theta * tau * (grad_u_j * u_star[q]) * v_i;                                         // b
                    cell_lhs -= tau * p_j * div_v_i;                                                                // c
                    cell_lhs += theta * tau * nu * scalar_product(eps_u_j, eps_v_i);                                // d
                    cell_lhs += theta * tau * delta_1 * (grad_u_j * u_star[q] + grad_p_j) * (grad_v_i * u_star[q]); // e
                    cell_lhs += theta * tau * delta_2 * div_u_j * div_v_i;                                          // f

                    // pressure
                    cell_lhs += theta * div_u_j * q_i;                                          // a
                    cell_lhs += delta_1 * (grad_p_j + theta * grad_u_j * u_star[q]) * grad_q_i; // b
                    // clang-format on

                    cell_lhs *= JxW;
                    cell_contribution(i, j) += cell_lhs;
                  }

                Number cell_rhs = 0.0;

                // clang-format off
                // velocity
                cell_rhs += u_0[q] * v_i;                                                                                     // a
                cell_rhs -= (1.0 - theta) * tau * (grad_u_0[q] * u_star[q]) * v_i;                                            // b
                cell_rhs -= (1.0 - theta) * tau * nu * scalar_product(symmetrize(grad_u_0[q]), eps_v_i);                      // d
                cell_rhs -= (1.0 - theta) * tau * delta_1 * (grad_u_0[q] * u_star[q] + grad_p_0[q]) * (grad_v_i * u_star[q]); // e
                cell_rhs -= (1.0 - theta) * tau * delta_2 * div_u_0[q] * div_v_i;                                             // f
                
                // pressure
                cell_rhs -= (1.0 - theta) * div_u_0[q] * q_i;                               // a
                cell_rhs -= delta_1 * ((1.0 - theta) * grad_u_0[q] * u_star[q]) * grad_q_i; // b
                // clang-format on

                cell_rhs *= JxW;

                cell_rhs_contribution(i) += cell_rhs;
              }
          }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_contribution,
                                               cell_rhs_contribution,
                                               local_dof_indices,
                                               system_matrix,
                                               system_rhs);
      }

    system_rhs.compress(VectorOperation::add);
    system_matrix.compress(VectorOperation::add);
  }
};



/**
 * Collection of parameters.
 */
struct Parameters
{
  // system
  unsigned int dim                  = 2;
  unsigned int fe_degree            = 1;
  unsigned int mapping_degree       = 1;
  unsigned int n_global_refinements = 0;

  // simulation
  std::string simulation_name = "channel";

  // system
  double       dt             = 0.0;
  double       cfl            = 0.1;
  double       t_final        = 3.0;
  double       theta          = 0.5;
  unsigned int bdf_order      = 1;
  std::string  time_intration = "theta";

  // NSE-GLS parameters
  double nu                         = 0.1;
  double c_1                        = 4.0;
  double c_2                        = 2.0;
  bool   consider_time_deriverative = false;
  bool   cell_wise_stabilization    = true;

  // implmentation of operator evaluation
  bool use_matrix_free_ns_operator = true;

  // linear solver
  unsigned int lin_n_max_iterations   = 10000;
  double       lin_absolute_tolerance = 1e-12;
  double       lin_relative_tolerance = 1e-8;


  // preconditioner of linear solver
  std::string                     preconditioner = "ILU";
  PreconditionerGMGAdditionalData gmg;

  // nonlinear solver
  std::string nonlinear_solver = "linearized";

  // output
  std::string paraview_prefix    = "results";
  double      output_granularity = 0.0;

  // simulation-specific parameters (TODO)
  bool no_slip = true;

  void
  parse(const std::string file_name)
  {
    dealii::ParameterHandler prm;
    add_parameters(prm);

    prm.parse_input(file_name, "", true);
  }

private:
  void
  add_parameters(ParameterHandler &prm)
  {
    // system
    prm.add_parameter("dim", dim);
    prm.add_parameter("fe degree", fe_degree);
    prm.add_parameter("mapping degree", mapping_degree);
    prm.add_parameter("n global refinements", n_global_refinements);

    // simulation
    prm.add_parameter("simulation name", simulation_name);

    // time stepping
    prm.add_parameter("dt", dt);
    prm.add_parameter("cfl", cfl);
    prm.add_parameter("t final", t_final);
    prm.add_parameter("theta", theta);
    prm.add_parameter("bdf order", bdf_order);
    prm.add_parameter("time intration",
                      time_intration,
                      "",
                      Patterns::Selection("bdf|theta"));

    // NSE-GLS parameters
    prm.add_parameter("nu", nu);
    prm.add_parameter("c1", c_1);
    prm.add_parameter("c2", c_2);
    prm.add_parameter("consider time deriverative", consider_time_deriverative);
    prm.add_parameter("cell wise stabilization", cell_wise_stabilization);

    // implmentation of operator evaluation
    prm.add_parameter("use matrix free ns operator",
                      use_matrix_free_ns_operator);

    // linear solver
    prm.add_parameter("lin n max iterations", lin_n_max_iterations);
    prm.add_parameter("lin absolute tolerance", lin_absolute_tolerance);
    prm.add_parameter("lin relative tolerance", lin_relative_tolerance);

    // preconditioner of linear solver
    prm.add_parameter("preconditioner",
                      preconditioner,
                      "",
                      Patterns::Selection("AMG|GMG|ILU|GMG-LS"));
    gmg.add_parameters(prm);

    // nonlinear solver
    prm.add_parameter("nonlinear solver",
                      nonlinear_solver,
                      "",
                      Patterns::Selection(
                        "linearized|Picard simple|Picard|Newton"));

    // output
    prm.add_parameter("paraview prefix", paraview_prefix);
    prm.add_parameter("output granularity", output_granularity);

    // simulation-specific
    prm.add_parameter("no slip", no_slip);
  }
};



/**
 * Base class for simulations.
 */
template <int dim>
class SimulationBase
{
public:
  struct BoundaryDescriptor
  {
    std::vector<unsigned int> all_homogeneous_dbcs;
    std::vector<unsigned int> all_homogeneous_nbcs;
    std::vector<std::pair<unsigned int, std::shared_ptr<Function<dim, Number>>>>
      all_inhomogeneous_dbcs;

    std::vector<unsigned int> all_slip_bcs;
  };

  virtual void
  create_triangulation(Triangulation<dim> &tria,
                       const unsigned int  n_global_refinements) const = 0;

  virtual BoundaryDescriptor
  get_boundary_descriptor() const = 0;

  virtual void
  postprocess(const double           t,
              const Mapping<dim>    &mapping,
              const DoFHandler<dim> &dof_handler,
              const VectorType      &vector) const
  {
    // to be implemented in derived classes
    (void)t;
    (void)mapping;
    (void)dof_handler;
    (void)vector;
  }
};



/**
 * Channel simulation.
 */
template <int dim>
class SimulationChannel : public SimulationBase<dim>
{
public:
  using BoundaryDescriptor = typename SimulationBase<dim>::BoundaryDescriptor;

  SimulationChannel()
    : n_stretching(4)
  {}

  void
  create_triangulation(Triangulation<dim> &tria,
                       const unsigned int  n_global_refinements) const override
  {
    std::vector<unsigned int> n_subdivisions(dim, 1);
    n_subdivisions[0] *= n_stretching;

    Point<dim> p0;
    Point<dim> p1;

    for (unsigned int d = 0; d < dim; ++d)
      p1[d] = 1.0;
    p1[0] *= n_stretching;

    GridGenerator::subdivided_hyper_rectangle(
      tria, n_subdivisions, p0, p1, true);

    tria.refine_global(2);

    tria.refine_global(n_global_refinements);
  }

  virtual BoundaryDescriptor
  get_boundary_descriptor() const override
  {
    BoundaryDescriptor bcs;

    bcs.all_inhomogeneous_dbcs.emplace_back(0,
                                            std::make_shared<InflowVelocity>());

    bcs.all_homogeneous_nbcs.push_back(1);

    for (unsigned d = 1; d < dim; ++d)
      {
        bcs.all_homogeneous_dbcs.push_back(2 * d);
        bcs.all_homogeneous_dbcs.push_back(2 * d + 1);
      }

    return bcs;
  }

private:
  const unsigned int n_stretching;

  class InflowVelocity : public Function<dim, Number>
  {
  public:
    InflowVelocity()
      : Function<dim>(dim + 1)
    {}

    Number
    value(const Point<dim> &p, const unsigned int component) const override
    {
      (void)p;

      if (component == 0)
        return 1.0;
      else
        return 0.0;
    }

  private:
  };
};



/**
 * Flow-past cylinder simulation.
 */
template <int dim>
class SimulationCylinder : public SimulationBase<dim>
{
public:
  using BoundaryDescriptor = typename SimulationBase<dim>::BoundaryDescriptor;

  SimulationCylinder(const double nu, const bool use_no_slip_cylinder_bc)
    : use_no_slip_cylinder_bc(use_no_slip_cylinder_bc)
    , nu(nu)
  {
    drag_lift_pressure_file.open("drag_lift_pressure.m", std::ios::out);
  }

  ~SimulationCylinder()
  {
    drag_lift_pressure_file.close();
  }

  void
  create_triangulation(Triangulation<dim> &tria,
                       const unsigned int  n_global_refinements) const override
  {
    ExaDG::FlowPastCylinder::create_coarse_grid(tria);

    tria.refine_global(n_global_refinements);
  }

  virtual BoundaryDescriptor
  get_boundary_descriptor() const override
  {
    BoundaryDescriptor bcs;

    // inflow
    bcs.all_inhomogeneous_dbcs.emplace_back(
      0, std::make_shared<InflowBoundaryValues>());

    // outflow
    bcs.all_homogeneous_nbcs.push_back(1);

    // walls
    bcs.all_slip_bcs.push_back(2);

    // cylinder
    if (use_no_slip_cylinder_bc)
      bcs.all_homogeneous_dbcs.push_back(3);
    else
      bcs.all_slip_bcs.push_back(3);

    return bcs;
  }

  void
  postprocess(const double           t,
              const Mapping<dim>    &mapping,
              const DoFHandler<dim> &dof_handler,
              const VectorType      &solution) const override
  {
    const bool has_ghost_elements = solution.has_ghost_elements();

    AssertThrow(has_ghost_elements == false, ExcInternalError());

    if (has_ghost_elements == false)
      solution.update_ghost_values();

    double drag = 0, lift = 0, p_diff = 0;

    const MPI_Comm comm = dof_handler.get_communicator();

    QGauss<dim - 1> face_quadrature_formula(3);
    const int       n_q_points = face_quadrature_formula.size();

    FEFaceValues<dim> fe_face_values(dof_handler.get_fe(),
                                     face_quadrature_formula,
                                     update_values | update_quadrature_points |
                                       update_gradients | update_JxW_values |
                                       update_normal_vectors);

    FEValuesViews::Vector<dim> velocities(fe_face_values, 0);
    FEValuesViews::Scalar<dim> pressure(fe_face_values, dim);

    std::vector<dealii::SymmetricTensor<2, dim>> eps_u(n_q_points);
    std::vector<double>                          p(n_q_points);

    Tensor<2, dim> fluid_stress;
    Tensor<1, dim> forces;

    double drag_local = 0;
    double lift_local = 0;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        for (const auto face : cell->face_indices())
          if (cell->face(face)->at_boundary() &&
              (cell->face(face)->boundary_id() == 3))
            {
              fe_face_values.reinit(cell, face);
              std::vector<Point<dim>> q_points =
                fe_face_values.get_quadrature_points();

              velocities.get_function_symmetric_gradients(solution, eps_u);
              pressure.get_function_values(solution, p);

              for (int q = 0; q < n_q_points; ++q)
                {
                  const Tensor<1, dim> normal_vector =
                    -fe_face_values.normal_vector(q);

                  Tensor<2, dim> fluid_pressure;
                  fluid_pressure[0][0] = p[q];
                  fluid_pressure[1][1] = p[q];

                  const Tensor<2, dim> fluid_stress =
                    nu * eps_u[q] - fluid_pressure;

                  const Tensor<1, dim> forces =
                    fluid_stress * normal_vector * fe_face_values.JxW(q);

                  drag_local += forces[0];
                  lift_local += forces[1];
                }
            }
      }

    drag = Utilities::MPI::sum(drag_local, comm);
    lift = Utilities::MPI::sum(lift_local, comm);

    // calculate pressure drop

    // 1) set up evaluation routine (TODO: can be reused!)
    std::shared_ptr<Utilities::MPI::RemotePointEvaluation<dim>> rpe;

    if (rpe == nullptr)
      {
        Point<dim> p1, p2;
        p1[0] = ExaDG::FlowPastCylinder::X_C - ExaDG::FlowPastCylinder::D * 0.5;
        p2[0] = ExaDG::FlowPastCylinder::X_C + ExaDG::FlowPastCylinder::D * 0.5;

        std::vector<Point<dim>> points;

        if (Utilities::MPI::this_mpi_process(comm) == 0)
          {
            points.push_back(p1);
            points.push_back(p2);
          }

        auto rpe_temp =
          std::make_shared<Utilities::MPI::RemotePointEvaluation<dim>>();
        rpe_temp->reinit(points, dof_handler.get_triangulation(), mapping);

        rpe = rpe_temp;
      }

    const auto values = VectorTools::point_values<1>(
      *rpe, dof_handler, solution, VectorTools::EvaluationFlags::avg, dim);

    if (Utilities::MPI::this_mpi_process(comm) == 0)
      p_diff = values[0] - values[1];

    // write to file
    if (Utilities::MPI::this_mpi_process(comm) == 0)
      {
        drag_lift_pressure_file << t << "\t" << drag << "\t" << lift << "\t"
                                << p_diff << "\n";
        drag_lift_pressure_file.flush();
      }

    // clean up
    if (has_ghost_elements == false)
      solution.zero_out_ghost_values();
  }

private:
  const bool   use_no_slip_cylinder_bc;
  const double nu;

  std::shared_ptr<const Utilities::MPI::RemotePointEvaluation<dim>> rpe;

  mutable std::ofstream drag_lift_pressure_file;

  class InflowBoundaryValues : public Function<dim>
  {
  public:
    InflowBoundaryValues()
      : Function<dim>(dim + 1)
      , t_(0.0){};

    double
    value(const Point<dim> &p, const unsigned int component) const override
    {
      const double Um = 1.5;
      const double H  = 0.41;
      const double y  = p[1] - H / 2.0;

      (void)Um;
      (void)H;
      (void)y;

      /// FIXME here. Somehow the velocity is too small
      /// I don't know why.
      const double u_val = 1.0;
      // const double u_val = 2.0 * 4.0 * Um * (y + H / 2.0) * (H / 2.0 - y)
      //*
      //  std::sin((t_+1e-10) * numbers::PI / 8.0) / (H * H)
      ;
      const double v_val = 0.0;
      const double p_val = 0.0;

      if (component == 0)
        return u_val;
      else if (component == 1)
        return v_val;
      else if (component == 2)
        return p_val;

      return 0;
    }

  private:
    const double t_;
  };
};



/**
 * Flow-past cylinder simulation with alternative mesh.
 */
template <int dim>
class SimulationCylinderOld : public SimulationBase<dim>
{
public:
  using BoundaryDescriptor = typename SimulationBase<dim>::BoundaryDescriptor;

  SimulationCylinderOld(const double nu, const bool use_no_slip_cylinder_bc)
    : use_no_slip_cylinder_bc(use_no_slip_cylinder_bc)
    , nu(nu)
  {
    drag_lift_pressure_file.open("drag_lift_pressure.m", std::ios::out);
  }

  ~SimulationCylinderOld()
  {
    drag_lift_pressure_file.close();
  }

  void
  create_triangulation(Triangulation<dim> &tria,
                       const unsigned int  n_global_refinements) const override
  {
    if (false /* TODO */)
      cylinder(tria,
               ExaDG::FlowPastCylinder::L2 - ((dim == 2) ?
                                                ExaDG::FlowPastCylinder::L1 :
                                                ExaDG::FlowPastCylinder::X_0),
               ExaDG::FlowPastCylinder::H,
               ExaDG::FlowPastCylinder::X_C,
               ExaDG::FlowPastCylinder::D);
    else
      cylinder(tria, 4.0, 2.0, 0.6, 0.5);

    tria.refine_global(n_global_refinements);
  }

  virtual BoundaryDescriptor
  get_boundary_descriptor() const override
  {
    BoundaryDescriptor bcs;

    // inflow
    bcs.all_inhomogeneous_dbcs.emplace_back(
      0, std::make_shared<InflowBoundaryValues>());

    // outflow
    bcs.all_homogeneous_nbcs.push_back(1);

    // walls
    bcs.all_slip_bcs.push_back(2);

    // cylinder
    if (use_no_slip_cylinder_bc)
      bcs.all_homogeneous_dbcs.push_back(3);
    else
      bcs.all_slip_bcs.push_back(3);

    return bcs;
  }

  void
  postprocess(const double           t,
              const Mapping<dim>    &mapping,
              const DoFHandler<dim> &dof_handler,
              const VectorType      &solution) const override
  {
    const bool has_ghost_elements = solution.has_ghost_elements();

    AssertThrow(has_ghost_elements == false, ExcInternalError());

    if (has_ghost_elements == false)
      solution.update_ghost_values();

    double drag = 0, lift = 0, p_diff = 0;

    const MPI_Comm comm = dof_handler.get_communicator();

    QGauss<dim - 1> face_quadrature_formula(3);
    const int       n_q_points = face_quadrature_formula.size();

    FEFaceValues<dim> fe_face_values(dof_handler.get_fe(),
                                     face_quadrature_formula,
                                     update_values | update_quadrature_points |
                                       update_gradients | update_JxW_values |
                                       update_normal_vectors);

    FEValuesViews::Vector<dim> velocities(fe_face_values, 0);
    FEValuesViews::Scalar<dim> pressure(fe_face_values, dim);

    std::vector<dealii::SymmetricTensor<2, dim>> eps_u(n_q_points);
    std::vector<double>                          p(n_q_points);

    Tensor<2, dim> fluid_stress;
    Tensor<1, dim> forces;

    double drag_local = 0;
    double lift_local = 0;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        for (const auto face : cell->face_indices())
          if (cell->face(face)->at_boundary() &&
              (cell->face(face)->boundary_id() == 3))
            {
              fe_face_values.reinit(cell, face);
              std::vector<Point<dim>> q_points =
                fe_face_values.get_quadrature_points();

              velocities.get_function_symmetric_gradients(solution, eps_u);
              pressure.get_function_values(solution, p);

              for (int q = 0; q < n_q_points; ++q)
                {
                  const Tensor<1, dim> normal_vector =
                    -fe_face_values.normal_vector(q);

                  Tensor<2, dim> fluid_pressure;
                  fluid_pressure[0][0] = p[q];
                  fluid_pressure[1][1] = p[q];

                  const Tensor<2, dim> fluid_stress =
                    nu * eps_u[q] - fluid_pressure;

                  const Tensor<1, dim> forces =
                    fluid_stress * normal_vector * fe_face_values.JxW(q);

                  drag_local += forces[0];
                  lift_local += forces[1];
                }
            }
      }

    drag = Utilities::MPI::sum(drag_local, comm);
    lift = Utilities::MPI::sum(lift_local, comm);

    // calculate pressure drop

    // 1) set up evaluation routine (TODO: can be reused!)
    std::shared_ptr<Utilities::MPI::RemotePointEvaluation<dim>> rpe;

    if (rpe == nullptr)
      {
        Point<dim> p1, p2;
        p1[0] = ExaDG::FlowPastCylinder::X_C - ExaDG::FlowPastCylinder::D * 0.5;
        p2[0] = ExaDG::FlowPastCylinder::X_C + ExaDG::FlowPastCylinder::D * 0.5;
        p1[1] = p2[1] = ExaDG::FlowPastCylinder::H / 2.0;

        std::vector<Point<dim>> points;

        if (Utilities::MPI::this_mpi_process(comm) == 0)
          {
            points.push_back(p1);
            points.push_back(p2);
          }

        auto rpe_temp =
          std::make_shared<Utilities::MPI::RemotePointEvaluation<dim>>();
        rpe_temp->reinit(points, dof_handler.get_triangulation(), mapping);

        rpe = rpe_temp;
      }

    const auto values = VectorTools::point_values<1>(
      *rpe, dof_handler, solution, VectorTools::EvaluationFlags::avg, dim);

    if (Utilities::MPI::this_mpi_process(comm) == 0)
      p_diff = values[0] - values[1];

    // write to file
    if (Utilities::MPI::this_mpi_process(comm) == 0)
      {
        drag_lift_pressure_file << t << "\t" << drag << "\t" << lift << "\t"
                                << p_diff << "\n";
        drag_lift_pressure_file.flush();
      }

    // clean up
    if (has_ghost_elements == false)
      solution.zero_out_ghost_values();
  }

private:
  const bool   use_no_slip_cylinder_bc;
  const double nu;

  std::shared_ptr<const Utilities::MPI::RemotePointEvaluation<dim>> rpe;

  mutable std::ofstream drag_lift_pressure_file;

  class InflowBoundaryValues : public Function<dim>
  {
  public:
    InflowBoundaryValues()
      : Function<dim>(dim + 1)
      , t_(0.0){};

    double
    value(const Point<dim> &p, const unsigned int component) const override
    {
      const double Um = 1.5;
      const double H  = 0.41;
      const double y  = p[1];

      (void)Um;
      (void)H;
      (void)y;

      /// FIXME here. Somehow the velocity is too small
      /// I don't know why.
      const double u_val = 1.0;
      // const double u_val = 2.0 * 4.0 * Um * (y + H / 2.0) * (H / 2.0 - y)
      //*
      //  std::sin((t_+1e-10) * numbers::PI / 8.0) / (H * H)
      ;
      const double v_val = 0.0;
      const double p_val = 0.0;

      if (component == 0)
        return u_val;
      else if (component == 1)
        return v_val;
      else if (component == 2)
        return p_val;

      return 0;
    }

  private:
    const double t_;
  };
};



/**
 * Flow-past cylinder simulation with alternative mesh.
 */
template <int dim>
class SimulationCylinderLethe : public SimulationBase<dim>
{
public:
  using BoundaryDescriptor = typename SimulationBase<dim>::BoundaryDescriptor;

  SimulationCylinderLethe()
    : use_no_slip_cylinder_bc(true)
  {}

  void
  create_triangulation(Triangulation<dim> &tria,
                       const unsigned int  n_global_refinements) const override
  {
    GridIn<dim> grid_in(tria);
    grid_in.read("../mesh/cylinder.msh");

    Point<dim>                   circleCenter(8, 8);
    const SphericalManifold<dim> manifold_description(circleCenter);
    tria.set_manifold(0, manifold_description);

    tria.refine_global(n_global_refinements);
  }

  virtual BoundaryDescriptor
  get_boundary_descriptor() const override
  {
    BoundaryDescriptor bcs;

    // inflow
    bcs.all_inhomogeneous_dbcs.emplace_back(
      1, std::make_shared<InflowBoundaryValues>());

    // outflow
    // bcs.all_homogeneous_nbcs.push_back(4);

    // walls
    bcs.all_slip_bcs.push_back(2);

    // cylinder
    if (use_no_slip_cylinder_bc)
      bcs.all_homogeneous_dbcs.push_back(0);
    else
      bcs.all_slip_bcs.push_back(0);

    return bcs;
  }

  void
  postprocess(const double           t,
              const Mapping<dim>    &mapping,
              const DoFHandler<dim> &dof_handler,
              const VectorType      &solution) const override
  {
    // nothing to do
    (void)t;
    (void)mapping;
    (void)dof_handler;
    (void)solution;
  }

private:
  const bool use_no_slip_cylinder_bc;

  class InflowBoundaryValues : public Function<dim>
  {
  public:
    InflowBoundaryValues()
      : Function<dim>(dim + 1)
      , t_(0.0){};

    double
    value(const Point<dim> &p, const unsigned int component) const override
    {
      (void)p;

      const double u_val = 1.0;
      const double v_val = 0.0;
      const double p_val = 0.0;

      if (component == 0)
        return u_val;
      else if (component == 1)
        return v_val;
      else if (component == 2)
        return p_val;

      return 0;
    }

  private:
    const double t_;
  };
};



/**
 * Flow-past cylinder simulation with alternative mesh.
 */
template <int dim>
class SimulationCylinderLethe2 : public SimulationBase<dim>
{
public:
  using BoundaryDescriptor = typename SimulationBase<dim>::BoundaryDescriptor;

  SimulationCylinderLethe2()
    : use_no_slip_cylinder_bc(true)
  {}

  void
  create_triangulation(Triangulation<dim> &tria,
                       const unsigned int  n_global_refinements) const override
  {
    GridGenerator::channel_with_cylinder(tria, 0.03, 2, 2.0, true);

    tria.refine_global(n_global_refinements);

    if (true)
      {
        const auto bb =
          BoundingBox<dim>(Point<dim>(0.2, 0.2)).create_extended(0.12);

        for (const auto &cell : tria.active_cell_iterators())
          if (bb.point_inside(cell->center()))
            cell->set_refine_flag();
        tria.execute_coarsening_and_refinement();
      }
  }

  virtual BoundaryDescriptor
  get_boundary_descriptor() const override
  {
    BoundaryDescriptor bcs;

    // inflow
    bcs.all_inhomogeneous_dbcs.emplace_back(
      0, std::make_shared<InflowBoundaryValues>());

    // walls
    bcs.all_slip_bcs.push_back(3);

    // cylinder
    if (use_no_slip_cylinder_bc)
      bcs.all_homogeneous_dbcs.push_back(2);
    else
      bcs.all_slip_bcs.push_back(2);

    return bcs;
  }

  void
  postprocess(const double           t,
              const Mapping<dim>    &mapping,
              const DoFHandler<dim> &dof_handler,
              const VectorType      &solution) const override
  {
    // nothing to do
    (void)t;
    (void)mapping;
    (void)dof_handler;
    (void)solution;
  }

private:
  const bool use_no_slip_cylinder_bc;

  class InflowBoundaryValues : public Function<dim>
  {
  public:
    InflowBoundaryValues()
      : Function<dim>(dim + 1)
      , t_(0.0){};

    double
    value(const Point<dim> &p, const unsigned int component) const override
    {
      (void)p;

      const double u_val = 1.0;
      const double v_val = 0.0;
      const double p_val = 0.0;

      if (component == 0)
        return u_val;
      else if (component == 1)
        return v_val;
      else if (component == 2)
        return p_val;

      return 0;
    }

  private:
    const double t_;
  };
};



/**
 * Flow-past cylinder simulation with alternative mesh.
 */
template <int dim>
class SimulationRotation : public SimulationBase<dim>
{
public:
  using BoundaryDescriptor = typename SimulationBase<dim>::BoundaryDescriptor;

  SimulationRotation()
    : use_no_slip_cylinder_bc(true)
  {}

  void
  create_triangulation(Triangulation<dim> &tria,
                       const unsigned int  n_global_refinements) const override
  {
    GridGenerator::hyper_shell(tria, Point<dim>(), 0.25, 1, 4, true);

    tria.refine_global(n_global_refinements);

    if (true)
      {
        for (const auto &cell : tria.active_cell_iterators())
          if (cell->at_boundary())
            cell->set_refine_flag();
        tria.execute_coarsening_and_refinement();
      }
    else if (false)
      {
        for (const auto &cell : tria.active_cell_iterators())
          for (const auto &face : cell->face_iterators())
            if (face->at_boundary() && (face->boundary_id() == 0))
              cell->set_refine_flag();
        tria.execute_coarsening_and_refinement();
      }
  }

  virtual BoundaryDescriptor
  get_boundary_descriptor() const override
  {
    BoundaryDescriptor bcs;

    // inflow
    bcs.all_inhomogeneous_dbcs.emplace_back(
      0, std::make_shared<InflowBoundaryValues>());

    // walls
    bcs.all_homogeneous_dbcs.push_back(1);

    return bcs;
  }

  void
  postprocess(const double           t,
              const Mapping<dim>    &mapping,
              const DoFHandler<dim> &dof_handler,
              const VectorType      &solution) const override
  {
    // nothing to do
    (void)t;
    (void)mapping;
    (void)dof_handler;
    (void)solution;
  }

private:
  const bool use_no_slip_cylinder_bc;

  class InflowBoundaryValues : public Function<dim>
  {
  public:
    InflowBoundaryValues()
      : Function<dim>(dim + 1)
      , t_(0.0){};

    double
    value(const Point<dim> &p, const unsigned int component) const override
    {
      (void)p;

      if (component == 0)
        return -p[1];
      else if (component == 1)
        return p[0];
      else if (component == 2)
        return 0;

      return 0;
    }

  private:
    const double t_;
  };
};



/**
 * Driver class for executing the simulation.
 */
template <int dim>
class Driver
{
public:
  Driver(const Parameters &params)
    : params(params)
    , comm(MPI_COMM_WORLD)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0)
  {}

  void
  run()
  {
    // select simulation case
    std::shared_ptr<SimulationBase<dim>> simulation;

    if (params.simulation_name == "channel")
      simulation = std::make_shared<SimulationChannel<dim>>();
    else if (params.simulation_name == "cylinder")
      simulation =
        std::make_shared<SimulationCylinder<dim>>(params.nu, params.no_slip);
    else if (params.simulation_name == "cylinder old")
      simulation =
        std::make_shared<SimulationCylinderOld<dim>>(params.nu, params.no_slip);
    else if (params.simulation_name == "cylinder lethe")
      simulation = std::make_shared<SimulationCylinderLethe<dim>>();
    else if (params.simulation_name == "cylinder lethe 2")
      simulation = std::make_shared<SimulationCylinderLethe2<dim>>();
    else if (params.simulation_name == "rotation")
      simulation = std::make_shared<SimulationRotation<dim>>();
    else
      AssertThrow(false, ExcNotImplemented());

    // set up system
    parallel::distributed::Triangulation<dim> tria(
      comm,
      ::Triangulation<dim>::none,
      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy);

    simulation->create_triangulation(tria, params.n_global_refinements);

    tria.reset_all_manifolds(); // TODO: problem with ChartManifold

    const auto bcs = simulation->get_boundary_descriptor();

    FESystem<dim> fe(FE_Q<dim>(params.fe_degree), dim + 1);

    DoFHandler<dim> dof_handler(tria);
    dof_handler.distribute_dofs(fe);

    pcout << "    [I] Number of active cells:    "
          << tria.n_global_active_cells()
          << "\n    [I] Global degrees of freedom: " << dof_handler.n_dofs()
          << std::endl;

    QGauss<dim> quadrature(params.fe_degree + 1);

    MappingQ<dim> mapping(params.mapping_degree);

    // set up constraints
    ComponentMask mask_v(dim + 1, true);
    mask_v.set(dim, false);

    ComponentMask mask_p(dim + 1, false);
    mask_p.set(dim, true);

    const auto locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);

    AffineConstraints<Number> constraints;
    constraints.reinit(locally_relevant_dofs);

    AffineConstraints<Number> constraints_copy;

    for (const auto bci : bcs.all_homogeneous_dbcs)
      DoFTools::make_zero_boundary_constraints(dof_handler,
                                               bci,
                                               constraints,
                                               mask_v);

    for (const auto bci : bcs.all_homogeneous_nbcs)
      DoFTools::make_zero_boundary_constraints(dof_handler,
                                               bci,
                                               constraints,
                                               mask_p);

    for (const auto bci : bcs.all_slip_bcs)
      VectorTools::compute_no_normal_flux_constraints(
        dof_handler, 0, {bci}, constraints, mapping);

    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    constraints_copy.copy_from(constraints);

    AffineConstraints<Number> constraints_homogeneous;
    constraints_homogeneous.copy_from(constraints);

    for (const auto &[bci, _] : bcs.all_inhomogeneous_dbcs)
      DoFTools::make_zero_boundary_constraints(dof_handler,
                                               bci,
                                               constraints_homogeneous,
                                               mask_v);

    constraints.close();
    constraints_homogeneous.close();

    AffineConstraints<Number> constraints_inhomogeneous;
    // note: filled during time loop

    // set up time integration scheme
    std::shared_ptr<TimeIntegratorData> time_integrator_data;

    if (params.time_intration == "bdf")
      time_integrator_data =
        std::make_shared<TimeIntegratorDataBDF>(params.bdf_order);
    else if (params.time_intration == "theta")
      time_integrator_data =
        std::make_shared<TimeIntegratorDataTheta>(params.theta);
    else
      AssertThrow(false, ExcNotImplemented());

    // set up Navier-Stokes operator
    std::shared_ptr<OperatorBase> ns_operator;

    if (params.use_matrix_free_ns_operator)
      {
        const bool increment_form = params.nonlinear_solver == "Newton";

        ns_operator = std::make_shared<NavierStokesOperator<dim>>(
          mapping,
          dof_handler,
          constraints_homogeneous,
          constraints,
          constraints_inhomogeneous,
          quadrature,
          params.nu,
          params.c_1,
          params.c_2,
          *time_integrator_data,
          params.consider_time_deriverative,
          increment_form,
          params.cell_wise_stabilization);
      }
    else
      {
        AssertThrow(params.nonlinear_solver != "Newton", ExcInternalError());

        ns_operator = std::make_shared<NavierStokesOperatorMatrixBased<dim>>(
          mapping,
          dof_handler,
          constraints_inhomogeneous,
          quadrature,
          params.nu,
          params.c_1,
          params.c_2,
          *time_integrator_data);
      }

    // set up preconditioner
    std::vector<std::shared_ptr<const Triangulation<dim>>> mg_trias;

    MGLevelObject<DoFHandler<dim>>           mg_dof_handlers;
    MGLevelObject<AffineConstraints<Number>> mg_constraints;

    MGLevelObject<std::shared_ptr<OperatorBase>>       mg_ns_operators;
    MGLevelObject<MGTwoLevelTransfer<dim, VectorType>> mg_transfers;
    MGLevelObject<MGTwoLevelTransfer<dim, VectorType>>
      mg_transfers_no_constraints;

    std::shared_ptr<MGTransferGlobalCoarsening<dim, VectorType>>
      mg_transfer_no_constraints;

    std::shared_ptr<PreconditionerBase> preconditioner;

    if (params.preconditioner == "ILU")
      preconditioner = std::make_shared<PreconditionerILU>(*ns_operator);
    else if (params.preconditioner == "AMG")
      {
        std::vector<std::vector<bool>> constant_modes;

        ComponentMask components(dim + 1, true);
        DoFTools::extract_constant_modes(dof_handler,
                                         components,
                                         constant_modes);

        preconditioner =
          std::make_shared<PreconditionerAMG>(*ns_operator, constant_modes);
      }
    else if (params.preconditioner == "GMG")
      {
        mg_trias =
          MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(
            dof_handler.get_triangulation());

        if (true /*TODO*/)
          {
            for (unsigned int i = 0; i < mg_trias.size(); ++i)
              {
                DataOut<dim> data_out;
                data_out.attach_triangulation(*mg_trias[i]);
                data_out.build_patches();

                data_out.write_vtu_in_parallel("grid." + std::to_string(i) +
                                                 ".vtu",
                                               MPI_COMM_WORLD);
              }
          }

        unsigned int minlevel = 0;
        unsigned int maxlevel = mg_trias.size() - 1;

        mg_dof_handlers.resize(minlevel, maxlevel);
        mg_constraints.resize(minlevel, maxlevel);
        mg_ns_operators.resize(minlevel, maxlevel);
        mg_transfers.resize(minlevel, maxlevel);
        mg_transfers_no_constraints.resize(minlevel, maxlevel);

        for (unsigned int level = minlevel; level <= maxlevel; ++level)
          {
            auto &dof_handler = mg_dof_handlers[level];
            auto &constraints = mg_constraints[level];

            dof_handler.reinit(*mg_trias[level]);
            dof_handler.distribute_dofs(fe);

            const auto locally_relevant_dofs =
              DoFTools::extract_locally_relevant_dofs(dof_handler);

            constraints.reinit(locally_relevant_dofs);

            for (const auto bci : bcs.all_homogeneous_dbcs)
              DoFTools::make_zero_boundary_constraints(dof_handler,
                                                       bci,
                                                       constraints,
                                                       mask_v);

            for (const auto bci : bcs.all_homogeneous_nbcs)
              DoFTools::make_zero_boundary_constraints(dof_handler,
                                                       bci,
                                                       constraints,
                                                       mask_p);

            for (const auto bci : bcs.all_slip_bcs)
              VectorTools::compute_no_normal_flux_constraints(
                dof_handler, 0, {bci}, constraints, mapping);

            for (const auto &[bci, _] : bcs.all_inhomogeneous_dbcs)
              DoFTools::make_zero_boundary_constraints(dof_handler,
                                                       bci,
                                                       constraints,
                                                       mask_v);

            DoFTools::make_hanging_node_constraints(dof_handler, constraints);

            constraints.close();

            if (params.use_matrix_free_ns_operator)
              {
                const bool increment_form = params.nonlinear_solver == "Newton";

                mg_ns_operators[level] =
                  std::make_shared<NavierStokesOperator<dim>>(
                    mapping,
                    dof_handler,
                    constraints,
                    constraints,
                    constraints,
                    quadrature,
                    params.nu,
                    params.c_1,
                    params.c_2,
                    *time_integrator_data,
                    params.consider_time_deriverative,
                    increment_form,
                    params.cell_wise_stabilization);
              }
            else
              {
                AssertThrow(params.nonlinear_solver != "Newton",
                            ExcInternalError());

                mg_ns_operators[level] =
                  std::make_shared<NavierStokesOperatorMatrixBased<dim>>(
                    mapping,
                    dof_handler,
                    constraints,
                    quadrature,
                    params.nu,
                    params.c_1,
                    params.c_2,
                    *time_integrator_data);
              }
          }


        // create transfer operator for interpolation to the levels (without
        // constraints)
        for (unsigned int level = minlevel; level < maxlevel; ++level)
          mg_transfers_no_constraints[level + 1].reinit(
            mg_dof_handlers[level + 1], mg_dof_handlers[level]);

        mg_transfer_no_constraints =
          std::make_shared<MGTransferGlobalCoarsening<dim, VectorType>>(
            mg_transfers_no_constraints, [&](const auto l, auto &vec) {
              mg_ns_operators[l]->initialize_dof_vector(vec);
            });


        // create transfer operator for preconditioner (with constraints)
        for (unsigned int level = minlevel; level < maxlevel; ++level)
          mg_transfers[level + 1].reinit(mg_dof_handlers[level + 1],
                                         mg_dof_handlers[level],
                                         mg_constraints[level + 1],
                                         mg_constraints[level]);

        std::shared_ptr<MGTransferGlobalCoarsening<dim, VectorType>> transfer =
          std::make_shared<MGTransferGlobalCoarsening<dim, VectorType>>(
            mg_transfers, [&](const auto l, auto &vec) {
              mg_ns_operators[l]->initialize_dof_vector(vec);
            });

        // create preconditioner
        preconditioner = std::make_shared<PreconditionerGMG<dim>>(
          dof_handler, mg_ns_operators, transfer, false, params.gmg);
      }
    else if (params.preconditioner == "GMG-LS")
      {
        dof_handler.distribute_mg_dofs(); // TODO

        unsigned int minlevel = 0;
        unsigned int maxlevel = tria.n_global_levels() - 1;

        mg_constraints.resize(minlevel, maxlevel);
        mg_ns_operators.resize(minlevel, maxlevel);

        MGConstrainedDoFs mg_constrained_dofs;
        mg_constrained_dofs.initialize(dof_handler);

        for (const auto bci : bcs.all_homogeneous_dbcs)
          mg_constrained_dofs.make_zero_boundary_constraints(dof_handler,
                                                             {bci},
                                                             mask_v);

        for (const auto bci : bcs.all_homogeneous_nbcs)
          mg_constrained_dofs.make_zero_boundary_constraints(dof_handler,
                                                             {bci},
                                                             mask_p);

        for (const auto &[bci, _] : bcs.all_inhomogeneous_dbcs)
          mg_constrained_dofs.make_zero_boundary_constraints(dof_handler,
                                                             {bci},
                                                             mask_v);

        for (unsigned int level = minlevel; level <= maxlevel; ++level)
          {
            AffineConstraints<Number> constraints;

            const auto &refinement_edge_indices =
              mg_constrained_dofs.get_refinement_edge_indices(level);

            const auto locally_relevant_dofs =
              DoFTools::extract_locally_relevant_level_dofs(dof_handler, level);
            constraints.reinit(locally_relevant_dofs);

            for (const auto bci : bcs.all_slip_bcs)
              VectorTools::compute_no_normal_flux_constraints_on_level(
                dof_handler,
                0,
                {bci},
                constraints,
                mapping,
                refinement_edge_indices,
                level);

            constraints.close();

            mg_constrained_dofs.add_user_constraints(level, constraints);
          }

        for (unsigned int level = minlevel; level <= maxlevel; ++level)
          {
            auto &constraints = mg_constraints[level];

            const auto locally_relevant_dofs =
              DoFTools::extract_locally_relevant_level_dofs(dof_handler, level);
            constraints.reinit(locally_relevant_dofs);

            mg_constrained_dofs.merge_constraints(
              constraints, level, true, false, true, true);

            if (params.use_matrix_free_ns_operator)
              {
                const bool increment_form = params.nonlinear_solver == "Newton";
                mg_ns_operators[level] =
                  std::make_shared<NavierStokesOperator<dim>>(
                    mapping,
                    dof_handler,
                    constraints,
                    constraints,
                    constraints,
                    quadrature,
                    params.nu,
                    params.c_1,
                    params.c_2,
                    *time_integrator_data,
                    params.consider_time_deriverative,
                    increment_form,
                    params.cell_wise_stabilization,
                    level);
              }
            else
              {
                AssertThrow(false, ExcNotImplemented());
              }
          }


        // create transfer operator for interpolation to the levels (without
        // constraints)
        mg_transfer_no_constraints =
          std::make_shared<MGTransferGlobalCoarsening<dim, VectorType>>();
        mg_transfer_no_constraints->build(
          dof_handler, [&](const auto l, auto &vec) {
            mg_ns_operators[l]->initialize_dof_vector(vec);
          });

        // create transfer operator for preconditioner (with constraints)
        std::shared_ptr<MGTransferGlobalCoarsening<dim, VectorType>> transfer =
          std::make_shared<MGTransferGlobalCoarsening<dim, VectorType>>(
            mg_constrained_dofs);
        transfer->build(dof_handler, [&](const auto l, auto &vec) {
          mg_ns_operators[l]->initialize_dof_vector(vec);
        });

        // create preconditioner
        preconditioner = std::make_shared<PreconditionerGMG<dim>>(
          dof_handler, mg_ns_operators, transfer, true, params.gmg);
      }
    else
      AssertThrow(false, ExcNotImplemented());


    // set up linear solver
    std::shared_ptr<LinearSolverBase> linear_solver;

    if (true)
      linear_solver =
        std::make_shared<LinearSolverGMRES>(*ns_operator,
                                            *preconditioner,
                                            params.lin_n_max_iterations,
                                            params.lin_absolute_tolerance,
                                            params.lin_relative_tolerance);
    else
      AssertThrow(false, ExcNotImplemented());

    // set up nonlinear solver
    std::shared_ptr<NonLinearSolverBase> nonlinear_solver;

    if (params.nonlinear_solver == "linearized")
      nonlinear_solver = std::make_shared<NonLinearSolverLinearized>();
    else if (params.nonlinear_solver == "Picard simple")
      nonlinear_solver = std::make_shared<NonLinearSolverPicardSimple>();
    else if (params.nonlinear_solver == "Picard")
      nonlinear_solver = std::make_shared<NonLinearSolverPicard>(
        time_integrator_data->get_theta());
    else if (params.nonlinear_solver == "Newton")
      nonlinear_solver = std::make_shared<NonLinearSolverNewton>();
    else
      AssertThrow(false, ExcNotImplemented());

    const auto set_previous_solution = [&](const SolutionHistory &solution) {
      ns_operator->set_previous_solution(solution);

      if (params.preconditioner == "GMG" || params.preconditioner == "GMG-LS")
        {
          MGLevelObject<SolutionHistory> all_mg_solution(
            mg_ns_operators.min_level(),
            mg_ns_operators.max_level(),
            time_integrator_data->get_order() + 1);

          for (unsigned int i = 1; i <= time_integrator_data->get_order(); ++i)
            {
              MGLevelObject<VectorType> mg_solution(
                mg_ns_operators.min_level(), mg_ns_operators.max_level());

              mg_transfer_no_constraints->interpolate_to_mg(
                dof_handler, mg_solution, solution.get_vectors()[i]);

              for (unsigned int l = mg_ns_operators.min_level();
                   l <= mg_ns_operators.max_level();
                   ++l)
                all_mg_solution[l].get_vectors()[i] = mg_solution[l];
            }

          for (unsigned int l = mg_ns_operators.min_level();
               l <= mg_ns_operators.max_level();
               ++l)
            mg_ns_operators[l]->set_previous_solution(all_mg_solution[l]);
        }
    };

    nonlinear_solver->setup_jacobian = [&](const VectorType &src) {
      ns_operator->set_linearization_point(src);

      // note: for the level operators, this is done during setup of
      // preconditioner
    };

    nonlinear_solver->setup_preconditioner = [&](const VectorType &solution) {
      if (params.preconditioner == "GMG" || params.preconditioner == "GMG-LS")
        {
          MGLevelObject<VectorType> mg_solution(mg_ns_operators.min_level(),
                                                mg_ns_operators.max_level());

          mg_transfer_no_constraints->interpolate_to_mg(dof_handler,
                                                        mg_solution,
                                                        solution);

          for (unsigned int l = mg_ns_operators.min_level();
               l <= mg_ns_operators.max_level();
               ++l)
            mg_ns_operators[l]->set_linearization_point(mg_solution[l]);
        }

      if (preconditioner)
        preconditioner->initialize();

      linear_solver->initialize();
    };

    nonlinear_solver->evaluate_rhs = [&](VectorType &dst) {
      ns_operator->evaluate_rhs(dst);
    };

    nonlinear_solver->evaluate_residual = [&](VectorType       &dst,
                                              const VectorType &src) {
      ns_operator->evaluate_residual(dst, src);
    };

    nonlinear_solver->solve_with_jacobian = [&](VectorType       &dst,
                                                const VectorType &src) {
      constraints_homogeneous.set_zero(const_cast<VectorType &>(src));
      linear_solver->solve(dst, src);
      constraints_homogeneous.distribute(dst);
    };

    if (false)
      nonlinear_solver->postprocess = [&](const VectorType &dst) {
        output(0.0, mapping, dof_handler, dst, true);
      };

    // initialize solution
    SolutionHistory solution(time_integrator_data->get_order() + 1);

    for (auto &vec : solution.get_vectors())
      ns_operator->initialize_dof_vector(vec);

    {
      // set time-dependent inhomogeneous DBCs
      constraints_inhomogeneous.clear();
      constraints_inhomogeneous.copy_from(constraints_copy);
      for (const auto &[bci, fu] : bcs.all_inhomogeneous_dbcs)
        {
          fu->set_time(0.0); // TODO: correct?
          VectorTools::interpolate_boundary_values(
            mapping, dof_handler, bci, *fu, constraints_inhomogeneous, mask_v);
        }
      constraints_inhomogeneous.close();

      constraints_inhomogeneous.distribute(solution.get_current_solution());
    }

    const double dt =
      (params.dt != 0.0) ?
        params.dt :
        (GridTools::minimal_cell_diameter(tria, mapping) * params.cfl);

    double       t       = 0.0;
    unsigned int counter = 1;

    output(t, mapping, dof_handler, solution.get_current_solution());
    simulation->postprocess(t,
                            mapping,
                            dof_handler,
                            solution.get_current_solution());

    // perform time loop
    for (; t < params.t_final; ++counter)
      {
        pcout << "\ncycle\t" << counter << " at time t = " << t;
        pcout << " with delta_t = " << dt << std::endl;

        // set time-dependent inhomogeneous DBCs
        constraints_inhomogeneous.clear();
        constraints_inhomogeneous.copy_from(constraints_copy);
        for (const auto &[bci, fu] : bcs.all_inhomogeneous_dbcs)
          {
            fu->set_time(t); // TODO: correct?
            VectorTools::interpolate_boundary_values(mapping,
                                                     dof_handler,
                                                     bci,
                                                     *fu,
                                                     constraints_inhomogeneous,
                                                     mask_v);
          }
        constraints_inhomogeneous.close();

        // set time step size
        time_integrator_data->update_dt(dt);

        ns_operator->invalidate_system(); // TODO

        if (params.preconditioner == "GMG" || params.preconditioner == "GMG-LS")
          {
            for (unsigned int l = mg_ns_operators.min_level();
                 l <= mg_ns_operators.max_level();
                 ++l)
              mg_ns_operators[l]->invalidate_system(); // TODO
          }

        // set previous solution
        solution.commit_solution();

        set_previous_solution(solution);

        auto &current_solution = solution.get_current_solution();

        // solve nonlinear problem
        nonlinear_solver->solve(current_solution);

        // apply constraints
        constraints_inhomogeneous.distribute(current_solution);
        constraints.distribute(current_solution);

        pcout << "    [S] l2-norm of solution: " << current_solution.l2_norm()
              << std::endl;

        t += dt;

        // postprocessing
        output(t, mapping, dof_handler, current_solution);
        simulation->postprocess(t, mapping, dof_handler, current_solution);
      }
  }

private:
  const Parameters         params;
  const MPI_Comm           comm;
  const ConditionalOStream pcout;

  void
  output(const double           time,
         const Mapping<dim>    &mapping,
         const DoFHandler<dim> &dof_handler,
         const VectorType      &vector,
         const bool             force = false) const
  {
    static unsigned int counter = 0;

    if ((force == false) && ((time + std::numeric_limits<double>::epsilon()) <
                             counter * params.output_granularity))
      return;

    const std::string file_name =
      params.paraview_prefix + "." + std::to_string(counter) + ".vtu";

    pcout << "    [O] output VTU (" << file_name << ")" << std::endl;

    DataOutBase::VtkFlags flags;
    flags.time                     = time;
    flags.write_higher_order_cells = true;

    DataOut<dim> data_out;
    data_out.set_flags(flags);
    data_out.attach_dof_handler(dof_handler);

    std::vector<std::string> labels(dim + 1, "u");
    labels[dim] = "p";

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim + 1, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation[dim] =
      DataComponentInterpretation::component_is_scalar;

    data_out.add_data_vector(vector,
                             {"u", "u", "p"},
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    data_out.build_patches(mapping, params.fe_degree);

    data_out.write_vtu_in_parallel(file_name, dof_handler.get_communicator());

    counter++;
  }
};



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  Parameters params;

  if (argc > 1)
    params.parse(std::string(argv[1]));

  if (params.dim == 2)
    {
      Driver<2> driver(params);
      driver.run();
    }
  else if (params.dim == 3)
    {
      Driver<3> driver(params);
      driver.run();
    }
  else
    {
      AssertThrow(false, ExcNotImplemented());
    }
}
