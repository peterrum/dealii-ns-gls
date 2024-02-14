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

#include "include/grid_cylinder.h"
#include "include/grid_cylinder_old.h"

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
  set_time_step_size(const Number tau) = 0;

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



class PreconditionerAMG : public PreconditionerBase
{
public:
  PreconditionerAMG(const OperatorBase &op)
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

  TrilinosWrappers::PreconditionAMG precon;
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
    , n_max_iterations(100) // TODO
    , abs_tolerance(1e-12)  // TODO
    , rel_tolerance(1e-8)   // TODO
    , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {}

  void
  initialize() override
  {
    preconditioner.initialize();
  }

  void
  solve(VectorType &dst, const VectorType &src) const override
  {
    ReductionControl solver_control(n_max_iterations,
                                    abs_tolerance,
                                    rel_tolerance);

    SolverGMRES<VectorType> solver(solver_control);

    try
      {
        solver.solve(op, dst, src, preconditioner);
      }
    catch (const SolverControl::NoConvergence &)
      {}

    pcout << "    [L] solved in " << solver_control.last_step()
          << " iterations." << std::endl;
  }

private:
  const OperatorBase &op;
  PreconditionerBase &preconditioner;

  const unsigned int n_max_iterations;
  const double       abs_tolerance;
  const double       rel_tolerance;

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

    // linear_solver.initialize();

    // compute right-hans-side vector
    VectorType rhs;
    rhs.reinit(solution);
    op.evaluate_rhs(rhs);

    // solve linear system
    linear_solver.initialize();
    linear_solver.solve(solution, rhs);
  }

private:
  OperatorBase     &op;
  LinearSolverBase &linear_solver;
};



class NonLinearSolverPicardSimple : public NonLinearSolverBase
{
public:
  NonLinearSolverPicardSimple(OperatorBase &op, LinearSolverBase &linear_solver)
    : op(op)
    , linear_solver(linear_solver)
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
        op.set_linearization_point(solution);

        // compute right-hans-side vector
        op.evaluate_rhs(rhs);

        // solve linear system
        if (true || (num_iteration == 0))
          linear_solver.initialize();

        linear_solver.solve(solution, rhs);

        // check convergence
        tmp -= solution;
        l2_norm = tmp.l2_norm();
        num_iteration++;

        AssertThrow(num_iteration <= picard_max_iteration,
                    dealii::ExcMessage(
                      "Picard iteration did not converge. Final residual is " +
                      std::to_string(l2_norm) + "."));
      }
  }

private:
  OperatorBase     &op;
  LinearSolverBase &linear_solver;

  const double       picard_tolerance;
  const unsigned int picard_max_iteration;
};



class NonLinearSolverPicard : public NonLinearSolverBase
{
public:
  NonLinearSolverPicard(OperatorBase     &op,
                        LinearSolverBase &linear_solver,
                        const double      theta)
    : op(op)
    , linear_solver(linear_solver)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
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
    VectorType residual, tmp, update;
    residual.reinit(solution);
    tmp.reinit(solution);
    update.reinit(solution);

    // copy vector
    tmp = solution;

    // set linearization point
    op.set_linearization_point(tmp);

    // compute residual
    op.evaluate_residual(residual, tmp);

    double linfty_norm = residual.l2_norm();

    pcout << "    [P] initial; residual (linfty) = " << linfty_norm
          << std::endl;

    for (unsigned int i = 1; linfty_norm > picard_tolerance; ++i)
      {
        if (i > picard_max_iteration)
          {
            const auto error_string =
              std::string(
                "Picard iteration did not converge. Final residual is ") +
              std::to_string(linfty_norm);
            AssertThrow(false, dealii::ExcMessage(error_string));
          }

        if (need_to_recompute_matrix || linear_always_recompute_matrix)
          {
            pcout << "    [P] step " << i << " ; recompute matrix" << std::endl;
            linear_solver.initialize();
            need_to_recompute_matrix = false;
          }

        linear_solver.solve(update, residual);
        tmp += update;

        // compute residual
        op.set_linearization_point(tmp);
        op.evaluate_residual(residual, tmp);
        double new_linfty_norm = residual.l2_norm();

        if (new_linfty_norm < picard_tolerance)
          {
            // accept new step
            linfty_norm = new_linfty_norm;
            pcout << "    [P] step " << i
                  << " ; residual (linfty) = " << linfty_norm << std::endl;
          }
        else if (new_linfty_norm >
                 picard_reduction_ratio_admissible * linfty_norm)
          {
            // revert to previous step
            tmp -= update;
            op.set_linearization_point(tmp);

            need_to_recompute_matrix = true;

            pcout << "    [P] step " << i
                  << " ; inadmissible residual reduction (" << new_linfty_norm
                  << " > " << linfty_norm << " ), recompute matrix\n"
                  << std::endl;
          }
        else if (new_linfty_norm >
                 picard_reduction_ratio_recompute * linfty_norm)
          {
            // accept new step
            linfty_norm = new_linfty_norm;
            pcout << "    [P] step " << i
                  << " ; residual (linfty) = " << linfty_norm
                  << ", recompute matrix" << std::endl;
            need_to_recompute_matrix = true;
          }
        else
          {
            // accept new step
            linfty_norm = new_linfty_norm;
            pcout << "    [P] step " << i
                  << " ; residual (linfty) = " << linfty_norm << std::endl;
          }
      }

    tmp *= 1. / theta;
    tmp.add((theta - 1.) / theta, solution);

    solution = tmp;
  }

private:
  OperatorBase     &op;
  LinearSolverBase &linear_solver;

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
    const Quadrature<dim>           &quadrature,
    const Number                     theta,
    const Number                     nu,
    const Number                     c_1,
    const Number                     c_2)
    : constraints_inhomogeneous(constraints_inhomogeneous)
    , theta(theta)
    , nu(nu)
    , c_1(c_1)
    , c_2(c_2)
    , valid_system(false)
  {
    const std::vector<const DoFHandler<dim> *> mf_dof_handlers = {&dof_handler,
                                                                  &dof_handler};
    const std::vector<const AffineConstraints<Number> *> mf_constraints = {
      &constraints_homogeneous, &constraints};

    typename MatrixFree<dim, Number>::AdditionalData additional_data;

    additional_data.mapping_update_flags = update_values | update_gradients;

    matrix_free.reinit(
      mapping, mf_dof_handlers, mf_constraints, quadrature, additional_data);

    for (auto i : this->matrix_free.get_constrained_dofs())
      constrained_indices.push_back(i);
  }

  void
  set_time_step_size(const Number tau) override
  {
    this->valid_system = false;

    this->tau = tau;
  }

  void
  set_previous_solution(const VectorType &vec) override
  {
    this->valid_system = false;

    const unsigned n_cells             = matrix_free.n_cell_batches();
    const unsigned n_quadrature_points = matrix_free.get_quadrature().size();

    old_value.reinit(n_cells, n_quadrature_points);
    old_gradient.reinit(n_cells, n_quadrature_points);
    old_gradient_p.reinit(n_cells, n_quadrature_points);

    delta_1.resize(n_cells);
    delta_2.resize(n_cells);

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

        // precompute value/gradient of linearization point at quadrature points
        for (const auto q : integrator.quadrature_point_indices())
          {
            old_value[cell][q]    = integrator.get_value(q);
            old_gradient[cell][q] = integrator.get_gradient(q);

            old_gradient_p[cell][q] = integrator_scalar.get_gradient(q);
          }

        // compute stabilization parameters
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
                  c_1 / std::sqrt(1. / (tau[0] * tau[0]) +
                                  u_max[v] * u_max[v] / (h * h));
                delta_2[cell][v] = c_2 * h;
              }
            else
              {
                delta_1[cell][v] = c_1 * h * h;
                delta_2[cell][v] = c_2 * h * h;
              }
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

    star_value.reinit(n_cells, n_quadrature_points);

    FEEvaluation<dim, -1, 0, dim, Number> integrator(matrix_free);

    const bool has_ghost_elements = vec.has_ghost_elements();

    AssertThrow(has_ghost_elements == false, ExcInternalError());

    if (has_ghost_elements == false)
      vec.update_ghost_values();

    for (unsigned int cell = 0; cell < n_cells; ++cell)
      {
        integrator.reinit(cell);

        integrator.read_dof_values_plain(vec);

        integrator.evaluate(EvaluationFlags::EvaluationFlags::values);

        for (const auto q : integrator.quadrature_point_indices())
          star_value[cell][q] = integrator.get_value(q);
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
    this->matrix_free.cell_loop(
      &NavierStokesOperator<dim>::do_vmult_range<true>, this, dst, src, true);

    // apply constraints
    matrix_free.get_affine_constraints(0).set_zero(dst);

    // move to rhs
    dst *= -1.0;
  }

  void
  vmult(VectorType &dst, const VectorType &src) const override
  {
    this->matrix_free.cell_loop(
      &NavierStokesOperator<dim>::do_vmult_range<false>, this, dst, src, true);

    for (unsigned int i = 0; i < constrained_indices.size(); ++i)
      dst.local_element(constrained_indices[i]) =
        src.local_element(constrained_indices[i]);
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

  VectorizedArray<Number> tau;

  const VectorizedArray<Number> theta;
  const VectorizedArray<Number> nu;
  const Number                  c_1;
  const Number                  c_2;

  mutable bool valid_system;

  AlignedVector<VectorizedArray<Number>> delta_1;
  AlignedVector<VectorizedArray<Number>> delta_2;

  Table<2, Tensor<1, dim, VectorizedArray<Number>>> star_value;
  Table<2, Tensor<1, dim, VectorizedArray<Number>>> old_value;
  Table<2, Tensor<2, dim, VectorizedArray<Number>>> old_gradient;
  Table<2, Tensor<1, dim, VectorizedArray<Number>>> old_gradient_p;

  std::vector<unsigned int> constrained_indices;

  template <bool evaluate_residual>
  void
  do_vmult_range(const MatrixFree<dim, Number>               &matrix_free,
                 VectorType                                  &dst,
                 const VectorType                            &src,
                 const std::pair<unsigned int, unsigned int> &range) const
  {
    FECellIntegrator phi(matrix_free, evaluate_residual ? 1 : 0);

    for (auto cell = range.first; cell < range.second; ++cell)
      {
        phi.reinit(cell);
        phi.read_dof_values(src);

        do_vmult_cell<evaluate_residual>(phi);

        phi.distribute_local_to_global(dst);
      }
  }

  /**
   * (v, D) + τ (v, S⋅∇B) - τ (div(v), p) + τ (ε(v), νε(B))
   *        + δ_1 τ (S⋅∇v, ∇P + S⋅∇B) + δ_2 τ (div(v), div(B)) = 0
   *          +------- SUPG --------+   +-------- GD --------+
   *
   * (q, div(B)) + δ_1 (∇q, ∇p + S⋅∇B) = 0
   *               +------ PSPG -----+
   *
   * with the following nomenclature:
   *  - S := u^*
   *  - B := θ u^{n+1} + (1-θ) u^{n}
   *  - P := θ p^{n+1} + (1-θ) p^{n}
   *  - D := u^{n+1} - u^{n}
   *  - p := p^{n+1}.
   */
  template <bool evaluate_residual>
  void
  do_vmult_cell(FECellIntegrator &integrator) const
  {
    const unsigned int cell = integrator.get_current_cell_index();

    const auto delta_1 = this->delta_1[cell];
    const auto delta_2 = this->delta_2[cell];
    const auto tau     = this->tau;
    const auto theta   = this->theta;
    const auto nu      = this->nu;

    integrator.evaluate(EvaluationFlags::EvaluationFlags::values |
                        EvaluationFlags::EvaluationFlags::gradients);

    for (const auto q : integrator.quadrature_point_indices())
      {
        typename FECellIntegrator::value_type    value_result    = {};
        typename FECellIntegrator::gradient_type gradient_result = {};

        const auto value    = integrator.get_value(q);
        const auto gradient = integrator.get_gradient(q);

        const VectorizedArray<Number>                 p_value = value[dim];
        const Tensor<1, dim, VectorizedArray<Number>> p_gradient =
          gradient[dim];
        Tensor<1, dim, VectorizedArray<Number>> p_bar_gradient =
          theta * gradient[dim];

        const Tensor<1, dim, VectorizedArray<Number>> u_star_value =
          star_value[cell][q];
        Tensor<1, dim, VectorizedArray<Number>> u_delta_value;
        Tensor<2, dim, VectorizedArray<Number>> u_bar_gradient;

        for (unsigned int d = 0; d < dim; ++d)
          {
            u_delta_value[d]  = value[d];
            u_bar_gradient[d] = theta * gradient[d];
          }

        if (evaluate_residual)
          {
            u_delta_value -= old_value[cell][q];
            u_bar_gradient +=
              (VectorizedArray<Number>(1) - theta) * old_gradient[cell][q];

            p_bar_gradient +=
              (VectorizedArray<Number>(1) - theta) * old_gradient_p[cell][q];
          }

        // precompute: div(B)
        VectorizedArray<Number> div_bar = u_bar_gradient[0][0];
        for (unsigned int d = 1; d < dim; ++d)
          div_bar += u_bar_gradient[d][d];

        // precompute: S⋅∇B
        const auto s_grad_b = u_bar_gradient * u_star_value;

        // velocity block:
        //  a)  (v, D)
        for (unsigned int d = 0; d < dim; ++d)
          value_result[d] = u_delta_value[d];

        //  b)  τ (v, S⋅∇B)
        for (unsigned int d = 0; d < dim; ++d)
          value_result[d] += s_grad_b[d] * tau;

        //  c)  - τ (div(v), p)
        for (unsigned int d = 0; d < dim; ++d)
          gradient_result[d][d] -= p_value * tau;

        //  d)  τ (ε(v), νε(B))
        for (unsigned int d = 0; d < dim; ++d)
          gradient_result[d][d] += u_bar_gradient[d][d] * (nu * tau);

        for (unsigned int e = 0, counter = dim; e < dim; ++e)
          for (unsigned int d = e + 1; d < dim; ++d, ++counter)
            {
              const auto tmp = (u_bar_gradient[d][e] + u_bar_gradient[e][d]) *
                               (nu * tau * 0.5);
              gradient_result[d][e] += tmp;
              gradient_result[e][d] += tmp;
            }

        //  e)  δ_1 τ (S⋅∇v, ∇P + S⋅∇B) -> SUPG stabilization
        const auto residual = (delta_1 * tau) * (p_bar_gradient + s_grad_b);
        for (unsigned int d0 = 0; d0 < dim; ++d0)
          for (unsigned int d1 = 0; d1 < dim; ++d1)
            gradient_result[d0][d1] += u_star_value[d1] * residual[d0];

        //  f) δ_2 τ (div(v), div(B)) -> GD stabilization
        for (unsigned int d = 0; d < dim; ++d)
          gradient_result[d][d] += (delta_2 * tau) * div_bar;



        // pressure block:
        //  a)  (q, div(B))
        value_result[dim] = div_bar;

        //  b)  δ_1 (∇q, ∇p + S⋅∇B) -> PSPG stabilization
        gradient_result[dim] = delta_1 * (p_gradient + s_grad_b);


        integrator.submit_value(value_result, q);
        integrator.submit_gradient(gradient_result, q);
      }

    integrator.integrate(EvaluationFlags::EvaluationFlags::values |
                         EvaluationFlags::EvaluationFlags::gradients);
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

        dsp.reinit(dof_handler.locally_owned_dofs(),
                   dof_handler.get_communicator());
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
};



template <int dim>
class NavierStokesOperatorMatrixBased : public OperatorBase
{
public:
  NavierStokesOperatorMatrixBased(const Mapping<dim>              &mapping,
                                  const DoFHandler<dim>           &dof_handler,
                                  const AffineConstraints<Number> &constraints,
                                  const Quadrature<dim>           &quadrature,
                                  const Number                     theta,
                                  const Number                     nu,
                                  const Number                     c_1,
                                  const Number                     c_2)
    : mapping(mapping)
    , dof_handler(dof_handler)
    , constraints(constraints)
    , quadrature(quadrature)
    , theta(theta)
    , nu(nu)
    , c_1(c_1)
    , c_2(c_2)
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

  void
  set_time_step_size(const Number tau) override
  {
    this->tau = tau;

    this->valid_system = false;
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

  mutable bool valid_system;

  Number tau;

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
  double cfl     = 0.1;
  double t_final = 3.0;
  double theta   = 0.5;

  // NSE-GLS parameters
  double nu  = 0.1;
  double c_1 = 4.0;
  double c_2 = 2.0;

  // implmentation of operator evaluation
  bool use_matrix_free_ns_operator = true;

  // preconditioner of linear solver
  std::string preconditioner = "ILU";

  // nonlinear solver
  std::string nonlinear_solver = "linearized";

  // output
  std::string paraview_prefix = "results";

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
    prm.add_parameter("cfl", cfl);
    prm.add_parameter("t final", t_final);
    prm.add_parameter("theta", theta);

    // NSE-GLS parameters
    prm.add_parameter("nu", nu);
    prm.add_parameter("c1", c_1);
    prm.add_parameter("c2", c_2);

    // implmentation of operator evaluation
    prm.add_parameter("use matrix free ns operator",
                      use_matrix_free_ns_operator);

    // preconditioner of linear solver
    prm.add_parameter("preconditioner",
                      preconditioner,
                      "",
                      Patterns::Selection("AMG|ILU"));

    // nonlinear solver
    prm.add_parameter("nonlinear solver",
                      nonlinear_solver,
                      "",
                      Patterns::Selection("linearized|Picard simple|Picard"));

    // output
    prm.add_parameter("paraview prefix", paraview_prefix);
  }
};



template <int dim, typename Number>
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
  create_triangulation(Triangulation<dim> &tria) const = 0;

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



template <int dim>
class SimulationChannel : public SimulationBase<dim>
{
public:
  using BoundaryDescriptor = typename SimulationBase<dim>::BoundaryDescriptor;

  SimulationChannel()
    : n_stretching(4)
  {}

  void
  create_triangulation(Triangulation<dim> &tria) const override
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
  }

  virtual BoundaryDescriptor
  get_boundary_descriptor() const override
  {
    BoundaryDescriptor bcs;

    bcs.all_inhomogeneous_dbcs.emplace_back(
      0, std::make_shared<InflowVelocity<dim, Number>>());

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
};



template <int dim>
class SimulationCylinder : public SimulationBase<dim>
{
public:
  using BoundaryDescriptor = typename SimulationBase<dim>::BoundaryDescriptor;

  SimulationCylinder(const double nu)
    : use_no_slip_cylinder_bc(true)
    , nu(nu)
  {
    drag_lift_pressure_file.open("drag_lift_pressure.m", std::ios::out);
  }

  ~SimulationCylinder()
  {
    drag_lift_pressure_file.close();
  }

  void
  create_triangulation(Triangulation<dim> &tria) const override
  {
    ExaDG::FlowPastCylinder::create_coarse_grid(tria);
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



template <int dim>
class SimulationCylinderOld : public SimulationBase<dim>
{
public:
  using BoundaryDescriptor = typename SimulationBase<dim>::BoundaryDescriptor;

  SimulationCylinderOld(const double nu)
    : use_no_slip_cylinder_bc(true)
    , nu(nu)
  {
    drag_lift_pressure_file.open("drag_lift_pressure.m", std::ios::out);
  }

  ~SimulationCylinderOld()
  {
    drag_lift_pressure_file.close();
  }

  void
  create_triangulation(Triangulation<dim> &tria) const override
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



template <int dim>
class Driver
{
public:
  Driver(const Parameters &params)
    : params(params)
  {}

  void
  run()
  {
    const MPI_Comm comm = MPI_COMM_WORLD;

    ConditionalOStream pcout(std::cout,
                             Utilities::MPI::this_mpi_process(comm) == 0);

    // select simulation case
    std::shared_ptr<SimulationBase<dim>> simulation;

    if (params.simulation_name == "channel")
      simulation = std::make_shared<SimulationChannel<dim>>();
    else if (params.simulation_name == "cylinder")
      simulation = std::make_shared<SimulationCylinder<dim>>(params.nu);
    else if (params.simulation_name == "cylinder old")
      simulation = std::make_shared<SimulationCylinderOld<dim>>(params.nu);
    else
      AssertThrow(false, ExcNotImplemented());

    // set up system
    parallel::distributed::Triangulation<dim> tria(comm);

    simulation->create_triangulation(tria);
    tria.refine_global(params.n_global_refinements);

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

    // set up Navier-Stokes operator
    std::shared_ptr<OperatorBase> ns_operator;

    if (params.use_matrix_free_ns_operator)
      ns_operator =
        std::make_shared<NavierStokesOperator<dim>>(mapping,
                                                    dof_handler,
                                                    constraints_homogeneous,
                                                    constraints,
                                                    constraints_inhomogeneous,
                                                    quadrature,
                                                    params.theta,
                                                    params.nu,
                                                    params.c_1,
                                                    params.c_2);
    else
      ns_operator = std::make_shared<NavierStokesOperatorMatrixBased<dim>>(
        mapping,
        dof_handler,
        constraints_inhomogeneous,
        quadrature,
        params.theta,
        params.nu,
        params.c_1,
        params.c_2);

    // set up preconditioner
    std::shared_ptr<PreconditionerBase> preconditioner;

    if (params.preconditioner == "ILU")
      preconditioner = std::make_shared<PreconditionerILU>(*ns_operator);
    else if (params.preconditioner == "AMG")
      preconditioner = std::make_shared<PreconditionerAMG>(*ns_operator);
    else
      AssertThrow(false, ExcNotImplemented());


    // set up linear solver
    std::shared_ptr<LinearSolverBase> linear_solver;

    linear_solver =
      std::make_shared<LinearSolverGMRES>(*ns_operator, *preconditioner);

    // set up nonlinear solver
    std::shared_ptr<NonLinearSolverBase> nonlinear_solver;

    if (params.nonlinear_solver == "linearized")
      nonlinear_solver =
        std::make_shared<NonLinearSolverLinearized>(*ns_operator,
                                                    *linear_solver);
    else if (params.nonlinear_solver == "Picard simple")
      nonlinear_solver =
        std::make_shared<NonLinearSolverPicardSimple>(*ns_operator,
                                                      *linear_solver);
    else if (params.nonlinear_solver == "Picard")
      nonlinear_solver = std::make_shared<NonLinearSolverPicard>(*ns_operator,
                                                                 *linear_solver,
                                                                 params.theta);
    else
      AssertThrow(false, ExcNotImplemented());

    // initialize solution
    VectorType solution;
    ns_operator->initialize_dof_vector(solution);

    const double dt =
      GridTools::minimal_cell_diameter(tria, mapping) * params.cfl;

    double       t       = 0.0;
    unsigned int counter = 1;

    output(t, mapping, dof_handler, solution);

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
        ns_operator->set_time_step_size(dt);

        // set previous solution
        ns_operator->set_previous_solution(solution);

        // solve nonlinear problem
        nonlinear_solver->solve(solution);

        // apply constraints
        constraints_inhomogeneous.distribute(solution);
        constraints.distribute(solution);

        pcout << "    [S] l2-norm of solution: " << solution.l2_norm()
              << std::endl;

        t += dt;

        // postprocessing
        output(t, mapping, dof_handler, solution);
        simulation->postprocess(t, mapping, dof_handler, solution);
      }
  }

private:
  const Parameters params;

  void
  output(const double           time,
         const Mapping<dim>    &mapping,
         const DoFHandler<dim> &dof_handler,
         const VectorType      &vector) const
  {
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

    static unsigned int counter = 0;

    const std::string file_name =
      params.paraview_prefix + "." + std::to_string(counter) + ".vtu";

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