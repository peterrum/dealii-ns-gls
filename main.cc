#include <deal.II/base/mpi.h>
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
  set_time_step_size(const Number tau) override
  {
    this->tau     = tau;
    this->inv_tau = Number(1.0) / tau;
  }

  void
  set_previous_solution(const VectorType &vec) override
  {
    const unsigned n_cells             = matrix_free.n_cell_batches();
    const unsigned n_quadrature_points = matrix_free.get_quadrature().size();

    old_value.reinit(n_cells, n_quadrature_points);
    old_gradient.reinit(n_cells, n_quadrature_points);

    delta_1.resize(n_cells);
    delta_2.resize(n_cells);

    FEEvaluation<dim, -1, 0, dim, Number> integrator(matrix_free);

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

        // precompute value/gradient of linearization point at quadrature points
        for (const auto q : integrator.quadrature_point_indices())
          {
            old_value[cell][q]    = integrator.get_value(q);
            old_gradient[cell][q] = integrator.get_gradient(q);
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
      }

    if (has_ghost_elements == false)
      vec.zero_out_ghost_values();
  }

  void
  set_linearization_point(const VectorType &vec) override
  {
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
    constraints_inhomogeneous.set_zero(dst);

    // move to rhs
    dst *= -1.0;
  }

  void
  vmult(VectorType &dst, const VectorType &src) const override
  {
    this->matrix_free.cell_loop(
      &NavierStokesOperator<dim>::do_vmult_range<false>, this, dst, src, true);
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

  Number                  tau;
  VectorizedArray<Number> inv_tau;

  const VectorizedArray<Number> theta;
  const VectorizedArray<Number> nu;
  const Number                  c_1;
  const Number                  c_2;

  AlignedVector<VectorizedArray<Number>> delta_1;
  AlignedVector<VectorizedArray<Number>> delta_2;

  Table<2, Tensor<1, dim, VectorizedArray<Number>>> star_value;
  Table<2, Tensor<1, dim, VectorizedArray<Number>>> old_value;
  Table<2, Tensor<2, dim, VectorizedArray<Number>>> old_gradient;

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
   * (v, D/tau) + (v, S⋅∇B) - (∇⋅v, p) + (ε(v), νε(B))
   *            + δ_1 (S⋅∇v, ∇p + S⋅∇B) + δ_2 (∇⋅v, div(B)) = 0
   *              +------ SUPG -------+   +------ GD -----+
   *
   * (q, div(B)) + δ_1 (∇q, ∇p + S⋅∇B) = 0
   *               +------ PSPG -----+
   *
   * with the following nomenclature:
   *  - S := u^*
   *  - B := θ u^{n+1} + (1-θ) u^{n}
   *  - D := u^{n+1} - u^{n}
   */
  template <bool evaluate_residual>
  void
  do_vmult_cell(FECellIntegrator &integrator) const
  {
    const unsigned int cell = integrator.get_current_cell_index();

    const auto delta_1 = this->delta_1[cell];
    const auto delta_2 = this->delta_2[cell];
    const auto inv_tau = this->inv_tau;
    const auto theta   = this->theta;
    const auto nu      = this->nu;

    integrator.evaluate(EvaluationFlags::EvaluationFlags::values |
                        EvaluationFlags::EvaluationFlags::gradients);

    for (const auto q : integrator.quadrature_point_indices())
      {
        typename FECellIntegrator::value_type value = integrator.get_value(q);
        typename FECellIntegrator::gradient_type gradient =
          integrator.get_gradient(q);

        Tensor<1, dim, VectorizedArray<Number>> value_star =
          star_value[cell][q];
        Tensor<1, dim, VectorizedArray<Number>> value_delta;
        Tensor<2, dim, VectorizedArray<Number>> gradient_bar;

        for (unsigned int d = 0; d < dim; ++d)
          {
            value_delta[d]  = value[d];
            gradient_bar[d] = theta * gradient[d];
          }

        if (evaluate_residual)
          {
            value_delta -= old_value[cell][q];
            gradient_bar +=
              (VectorizedArray<Number>(1) - theta) * old_gradient[cell][q];
          }

        // precompute: div(B)
        VectorizedArray<Number> div_bar = gradient_bar[0][0];
        for (unsigned int d = 1; d < dim; ++d)
          div_bar += gradient_bar[d][d];

        // precompute scaled residual: residual := δ_1 (∇p + S⋅∇B)
        Tensor<1, dim, VectorizedArray<Number>> residual;
        residual = gradient[dim];
        for (unsigned int d = 0; d < dim; ++d)
          residual += value_star[d] * gradient_bar[d];
        residual *= delta_1;

        typename FECellIntegrator::value_type    value_result;
        typename FECellIntegrator::gradient_type gradient_result;

        // velocity block:
        //  a)  (v, D/tau)
        for (unsigned int d = 0; d < dim; ++d)
          value_result[d] = value_delta[d] * inv_tau;

        //  b)  (v, S⋅∇B)
        for (unsigned int d0 = 0; d0 < dim; ++d0)
          for (unsigned int d1 = 0; d1 < dim; ++d1)
            value_result[d0] += value_star[d1] * gradient_bar[d0][d1];

        //  c)  -(div(v), p)
        for (unsigned int d = 0; d < dim; ++d)
          gradient_result[d][d] -= value[dim];

        //  d)  (ε(v), νε(B))
        Tensor<2, dim, VectorizedArray<Number>> symm_gradient_bar;

        for (unsigned int d0 = 0; d0 < dim; ++d0)
          for (unsigned int d1 = 0; d1 < dim; ++d1)
            symm_gradient_bar[d0][d1] =
              (gradient_bar[d0][d1] + gradient_bar[d1][d0]) * 0.5;

        symm_gradient_bar *= nu;

        for (unsigned int d0 = 0; d0 < dim; ++d0)
          for (unsigned int d1 = 0; d1 < dim; ++d1)
            {
              gradient_result[d0][d1] += symm_gradient_bar[d0][d1] * 0.5;
              gradient_result[d1][d0] += symm_gradient_bar[d0][d1] * 0.5;
            }

        //  e)  δ_1 (S⋅∇v, residual) -> SUPG stabilization
        for (unsigned int d = 0; d < dim; ++d)
          gradient_result[d][d] += value_star[d] * residual[d];

        //  f) δ_2 (div(v), div(B)) -> GD stabilization
        for (unsigned int d = 0; d < dim; ++d)
          gradient_result[d][d] += delta_2 * div_bar;



        // pressure block:
        //  a)  (q, div(B))
        for (unsigned int d = 0; d < dim; ++d)
          value_result[dim] += div_bar;

        //  b)  δ_1 (∇q, residual) -> PSPG stabilization
        gradient_result[dim] = residual;


        integrator.submit_value(value_result, q);
        integrator.submit_gradient(gradient_result, q);
      }

    integrator.integrate(EvaluationFlags::EvaluationFlags::values |
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

      MatrixFreeTools::compute_matrix(
        matrix_free,
        constraints,
        system_matrix,
        &NavierStokesOperator<dim>::do_vmult_cell<false>,
        this);
    }
  }
};



struct Parameters
{
  unsigned int dim            = 2;
  unsigned int fe_degree      = 2;
  unsigned int mapping_degree = 1;
  double       cfl            = 1.0;
  double       t_final        = 1.0;
  double       theta          = 0.5;
  double       nu             = 1.0;
  double       c_1            = 4.0;
  double       c_2            = 2.0;
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

    // set up system
    parallel::distributed::Triangulation<dim> tria(comm);

    std::vector<unsigned int> all_homogeneous_dbcs;
    std::vector<unsigned int> all_homogeneous_nbcs;
    std::vector<std::pair<unsigned int, std::shared_ptr<Function<dim, Number>>>>
      all_inhomogeneous_dbcs;

    // TODO: modularize initialization
    {
      const unsigned int n = 4;

      std::vector<unsigned int> n_subdivisions(dim, n);
      n_subdivisions[0] *= n;

      Point<dim> p0;
      Point<dim> p1;

      for (unsigned int d = 0; d < dim; ++d)
        p1[d] = 1.0;
      p1[0] *= n;

      GridGenerator::subdivided_hyper_rectangle(
        tria, n_subdivisions, p0, p1, true);

      all_inhomogeneous_dbcs.emplace_back(
        0, std::make_shared<InflowVelocity<dim, Number>>());

      all_homogeneous_nbcs.push_back(1);

      for (unsigned d = 1; d < dim; ++d)
        {
          all_homogeneous_dbcs.push_back(2 * d);
          all_homogeneous_dbcs.push_back(2 * d + 1);
        }
    }

    FESystem<dim> fe(FE_Q<dim>(params.fe_degree), dim + 1);

    DoFHandler<dim> dof_handler(tria);
    dof_handler.distribute_dofs(fe);

    QGauss<dim> quadrature(params.fe_degree + 1);

    MappingQ<dim> mapping(params.mapping_degree);

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

    for (const auto bci : all_homogeneous_nbcs)
      DoFTools::make_zero_boundary_constraints(dof_handler,
                                               bci,
                                               constraints,
                                               mask_p);

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
                                          quadrature,
                                          params.theta,
                                          params.nu,
                                          params.c_1,
                                          params.c_2);

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

    const double dt =
      GridTools::minimal_cell_diameter(tria, mapping) * params.cfl;

    double       t       = 0.0;
    unsigned int counter = 0;

    output(mapping, dof_handler, solution);

    // perform time loop
    for (; t < params.t_final; ++counter)
      {
        // set time-dependent inhomogeneous DBCs
        constraints_inhomogeneous.clear();
        for (const auto &[bci, fu] : all_inhomogeneous_dbcs)
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
        ns_operator.set_time_step_size(dt);

        // set previous solution
        ns_operator.set_previous_solution(solution);

        // solve nonlinear problem
        nonlinear_solver->solve(solution);

        t += dt;

        output(mapping, dof_handler, solution);
      }
  }

private:
  const Parameters params;

  void
  output(const Mapping<dim>    &mapping,
         const DoFHandler<dim> &dof_handler,
         const VectorType      &vector) const
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    data_out.add_data_vector(vector, "result");

    data_out.build_patches(mapping, params.fe_degree);

    static unsigned int counter = 0;

    const std::string file_name = "results." + std::to_string(counter) + ".vtu";

    data_out.write_vtu_in_parallel(file_name, dof_handler.get_communicator());

    counter++;
  }
};



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  Parameters params; // TODO: read parameters

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