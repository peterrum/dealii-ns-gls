
#include "operator_ns.h"

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/matrix_free/tools.h>

#include <deal.II/multigrid/mg_tools.h>

#include "config.h"

using namespace dealii;

template <int dim, int spacedim>
Table<2, bool>
create_bool_dof_mask(const FiniteElement<dim, spacedim> &fe,
                     const Quadrature<dim>              &quadrature)
{
  const auto compute_scalar_bool_dof_mask = [&quadrature](const auto &fe) {
    Table<2, bool>           bool_dof_mask(fe.dofs_per_cell, fe.dofs_per_cell);
    MappingQ1<dim, spacedim> mapping;
    FEValues<dim>            fe_values(mapping, fe, quadrature, update_values);

    Triangulation<dim, spacedim> tria;
    GridGenerator::hyper_cube(tria);

    fe_values.reinit(tria.begin());
    for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
      for (unsigned int j = 0; j < fe.dofs_per_cell; ++j)
        {
          double sum = 0;
          for (unsigned int q = 0; q < quadrature.size(); ++q)
            sum += fe_values.shape_value(i, q) * fe_values.shape_value(j, q);
          if (sum != 0)
            bool_dof_mask(i, j) = true;
        }

    return bool_dof_mask;
  };

  Table<2, bool> bool_dof_mask(fe.dofs_per_cell, fe.dofs_per_cell);

  if (fe.n_components() == 1)
    {
      bool_dof_mask = compute_scalar_bool_dof_mask(fe);
    }
  else
    {
      const auto scalar_bool_dof_mask =
        compute_scalar_bool_dof_mask(fe.base_element(0));

      for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
        for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
          if (scalar_bool_dof_mask[fe.system_to_component_index(i).second]
                                  [fe.system_to_component_index(j).second])
            bool_dof_mask[i][j] = true;
    }


  return bool_dof_mask;
}

/**
 * Matrix-free Navier-Stokes operator.
 */
template <int dim, typename Number>
NavierStokesOperator<dim, Number>::NavierStokesOperator(
  const Mapping<dim>              &mapping,
  const DoFHandler<dim>           &dof_handler,
  const AffineConstraints<Number> &constraints_homogeneous,
  const AffineConstraints<Number> &constraints,
  const AffineConstraints<Number> &constraints_inhomogeneous,
  const Quadrature<dim>           &quadrature,
  const Number                     nu,
  const Number                     c_1,
  const Number                     c_2,
  const std::set<unsigned int>    &all_outflow_bcs_cut,
  const std::map<unsigned int, std::shared_ptr<Function<dim, double>>>
                           &all_outflow_bcs_nitsche,
  const TimeIntegratorData &time_integrator_data,
  const bool                consider_time_derivative,
  const bool                increment_form,
  const bool                cell_wise_stabilization,
  const unsigned int        mg_level)
  : constraints_inhomogeneous(constraints_inhomogeneous)
  , theta(time_integrator_data.get_theta())
  , nu(nu)
  , c_1(c_1)
  , c_2(c_2)
  , all_outflow_bcs_cut(all_outflow_bcs_cut)
  , all_outflow_bcs_nitsche(all_outflow_bcs_nitsche)
  , needs_face_integrals(
      !(all_outflow_bcs_cut.empty() && all_outflow_bcs_nitsche.empty()))
  , time_integrator_data(time_integrator_data)
  , consider_time_derivative(consider_time_derivative &&
                               (time_integrator_data.get_order() > 0))
  , increment_form(increment_form)
  , cell_wise_stabilization(cell_wise_stabilization)
  , compute_penalty_parameters_for_previous_solution(false)
  , bool_dof_mask(create_bool_dof_mask(dof_handler.get_fe(), quadrature))
  , valid_system(false)
{
  const std::vector<const DoFHandler<dim> *> mf_dof_handlers = {&dof_handler,
                                                                &dof_handler};
  const std::vector<const AffineConstraints<Number> *> mf_constraints = {
    &constraints_homogeneous, &constraints};

  typename MatrixFree<dim, Number>::AdditionalData additional_data;

  additional_data.mapping_update_flags = update_values | update_gradients;
  if (!all_outflow_bcs_cut.empty())
    additional_data.mapping_update_flags_boundary_faces = update_values;
  if (!all_outflow_bcs_nitsche.empty())
    additional_data.mapping_update_flags_boundary_faces =
      update_values | update_gradients | update_quadrature_points;
  additional_data.mg_level = mg_level;

  matrix_free.reinit(
    mapping, mf_dof_handlers, mf_constraints, quadrature, additional_data);

  for (auto i : this->matrix_free.get_constrained_dofs())
    constrained_indices.push_back(i);

  if (consider_time_derivative)
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

template <int dim, typename Number>
const AffineConstraints<Number> &
NavierStokesOperator<dim, Number>::get_constraints() const
{
  return matrix_free.get_affine_constraints();
}

template <int dim, typename Number>
types::global_dof_index
NavierStokesOperator<dim, Number>::m() const
{
  if (this->matrix_free.get_mg_level() != numbers::invalid_unsigned_int)
    return this->matrix_free.get_dof_handler().n_dofs(
      this->matrix_free.get_mg_level());
  else
    return this->matrix_free.get_dof_handler().n_dofs();
}

template <int dim, typename Number>
std::vector<std::vector<bool>>
NavierStokesOperator<dim, Number>::extract_constant_modes() const
{
  std::vector<std::vector<bool>> constant_modes;

  ComponentMask components(dim + 1, true);

  if (this->matrix_free.get_mg_level() == numbers::invalid_unsigned_int)
    DoFTools::extract_constant_modes(this->matrix_free.get_dof_handler(),
                                     components,
                                     constant_modes);
  else
    DoFTools::extract_level_constant_modes(this->matrix_free.get_mg_level(),
                                           this->matrix_free.get_dof_handler(),
                                           components,
                                           constant_modes);


  return constant_modes;
}

template <int dim, typename Number>
void
NavierStokesOperator<dim, Number>::compute_inverse_diagonal(
  VectorType<Number> &diagonal) const
{
  MyScope scope(timer, "ns::compute_inverse_diagonal");

  std::function<void(FEFaceIntegrator &)> boundary_function;

  if (needs_face_integrals)
    boundary_function = [&](auto &integrator) {
      this->template do_vmult_boundary<false>(integrator);
    };

  matrix_free.initialize_dof_vector(diagonal);
  MatrixFreeTools::
    compute_diagonal<dim, -1, 0, dim + 1, Number, VectorizedArray<Number>>(
      matrix_free,
      diagonal,
      [&](auto &integrator) {
        this->template do_vmult_cell<false>(integrator);
      },
      {},
      boundary_function);

  for (unsigned int i = 0; i < edge_constrained_indices.size(); ++i)
    diagonal.local_element(edge_constrained_indices[i]) = 0.0;

  for (auto &i : diagonal)
    i = (std::abs(i) > 1.0e-10) ? (1.0 / i) : 1.0;
}

template <int dim, typename Number>
void
NavierStokesOperator<dim, Number>::invalidate_system()
{
  this->valid_system = false;
}

template <int dim, typename Number>
void
NavierStokesOperator<dim, Number>::set_previous_solution(
  const SolutionHistory<Number> &history)
{
  MyScope scope(timer, "ns::set_previous_solution");

  this->valid_system = false;

  if (this->time_integrator_data.get_order() == 0)
    return;

  const unsigned n_cells             = matrix_free.n_cell_batches();
  const unsigned n_quadrature_points = matrix_free.get_quadrature().size();

  u_time_derivative_old.reinit(n_cells, n_quadrature_points);

  FEEvaluation<dim, -1, 0, dim, Number> integrator(matrix_free);

  VectorType<Number> vec_old;
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

  if (theta[0] != 1.0)
    {
      const auto &vec = history.get_vectors()[1];

      u_old_gradient.reinit(n_cells, n_quadrature_points);
      p_old_gradient.reinit(n_cells, n_quadrature_points);

      FEEvaluation<dim, -1, 0, 1, Number> integrator_scalar(matrix_free,
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
          integrator.evaluate(EvaluationFlags::EvaluationFlags::gradients);

          integrator_scalar.reinit(cell);
          integrator_scalar.read_dof_values_plain(vec);
          integrator_scalar.evaluate(
            EvaluationFlags::EvaluationFlags::gradients);

          // precompute value/gradient of linearization point at quadrature
          // points
          for (const auto q : integrator.quadrature_point_indices())
            {
              u_old_gradient[cell][q] = integrator.get_gradient(q);

              p_old_gradient[cell][q] = integrator_scalar.get_gradient(q);
            }
        }

      if (has_ghost_elements == false)
        vec.zero_out_ghost_values();
    }

  if (compute_penalty_parameters_for_previous_solution == true)
    this->compute_penalty_parameters(history.get_vectors()[1]);
}

template <int dim, typename Number>
void
NavierStokesOperator<dim, Number>::compute_penalty_parameters(
  const VectorType<Number> &vec)
{
  this->valid_system = false;

  const unsigned n_cells             = matrix_free.n_cell_batches();
  const unsigned n_inner_faces       = matrix_free.n_inner_face_batches();
  const unsigned n_boundary_faces    = matrix_free.n_boundary_face_batches();
  const unsigned n_quadrature_points = matrix_free.get_quadrature().size();
  const unsigned n_face_quadrature_points =
    matrix_free.get_face_quadrature().size();
  const unsigned fe_degree =
    matrix_free.get_dof_handler().get_fe().tensor_degree();

  FEEvaluation<dim, -1, 0, dim, Number>     integrator(matrix_free);
  FEFaceEvaluation<dim, -1, 0, dim, Number> face_integrator(matrix_free);

  const bool has_ghost_elements = vec.has_ghost_elements();

  AssertThrow(has_ghost_elements == false, ExcInternalError());

  if (has_ghost_elements == false)
    vec.update_ghost_values();

  const auto tau  = this->time_integrator_data.get_current_dt();
  const auto stau = (tau == 0.0) ? 0.0 : (1.0 / tau);

  delta_1.resize(n_cells);
  delta_2.resize(n_cells);

  delta_1_q.reinit(n_cells, n_quadrature_points);
  delta_2_q.reinit(n_cells, n_quadrature_points);

  for (unsigned int cell = 0; cell < n_cells; ++cell)
    {
      integrator.reinit(cell);

      integrator.read_dof_values_plain(vec);
      integrator.evaluate(EvaluationFlags::EvaluationFlags::values);

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
                c_1 / std::sqrt(Utilities::fixed_power<2>(stau) +
                                u_max[v] * u_max[v] / (h * h));
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
          const double h_k = matrix_free.get_cell_iterator(cell, v)->measure();

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
            1. / std::sqrt(Utilities::fixed_power<2>(stau) +
                           4. * u_mag_squared / h / h +
                           9. * Utilities::fixed_power<2>(4. * nu / (h * h)));

          delta_2_q[cell][q] = std::sqrt(u_mag_squared) * h * 0.5;
        }
    }

  if (!(all_outflow_bcs_cut.empty() && all_outflow_bcs_nitsche.empty()))
    {
      // adopted from
      // https://github.com/chaos-polymtl/lethe/blob/9e25b0bd06dfd9392baf798ed3758abaee6dbb9e/source/solvers/fluid_dynamics_matrix_free_operators.cc#L965
      
      const double beta = 1.0; // TODO

      const unsigned fe_degree =
        matrix_free.get_dof_handler().get_fe().tensor_degree();

      effective_beta_face.reinit(n_boundary_faces);

      for (unsigned int face = n_inner_faces;
           face < n_inner_faces + n_boundary_faces;
           ++face)
        {
          VectorizedArray<Number> cell_size = 1.0;

          for (unsigned int v = 0;
               v < matrix_free.n_active_entries_per_face_batch(face);
               ++v)
            {
              const auto [cell_it, _] = matrix_free.get_face_iterator(face, v);

              if (dim == 2)
                cell_size[v] =
                  std::sqrt(4. * cell_it->measure() / M_PI) / fe_degree;
              else if (dim == 3)
                cell_size[v] =
                  std::pow(6 * cell_it->measure() / M_PI, 1. / 3.) / fe_degree;
            }

          effective_beta_face[face - n_inner_faces] =
            beta / std::pow(cell_size, static_cast<Number>(fe_degree + 1));
        }
    }

  if (!all_outflow_bcs_cut.empty())
    {
      face_velocity.reinit(n_boundary_faces, n_face_quadrature_points);

      for (unsigned int face = n_inner_faces;
           face < n_inner_faces + n_boundary_faces;
           ++face)
        {
          face_integrator.reinit(face);

          face_integrator.read_dof_values_plain(vec);
          face_integrator.evaluate(EvaluationFlags::EvaluationFlags::values);

          for (const auto q : face_integrator.quadrature_point_indices())
            face_velocity[face - n_inner_faces][q] =
              face_integrator.get_value(q);
        }
    }
  else if (!all_outflow_bcs_nitsche.empty())
    {
      // adopted from
      // https://github.com/chaos-polymtl/lethe/blob/9e25b0bd06dfd9392baf798ed3758abaee6dbb9e/source/solvers/fluid_dynamics_matrix_free_operators.cc#L966

      face_target_velocity.reinit(n_boundary_faces, n_face_quadrature_points);

      for (unsigned int face = n_inner_faces;
           face < n_inner_faces + n_boundary_faces;
           ++face)
        {
          face_integrator.reinit(face);

          const auto fu_ptr =
            all_outflow_bcs_nitsche.find(face_integrator.boundary_id());

          if (fu_ptr == all_outflow_bcs_nitsche.end())
            continue;

          for (const auto q : face_integrator.quadrature_point_indices())
            {
              const auto point_batch = face_integrator.quadrature_point(q);

              Tensor<1, dim + 1, VectorizedArray<Number>> target_velocity_value;

              for (unsigned int v = 0;
                   v < matrix_free.n_active_entries_per_face_batch(face);
                   ++v)
                {
                  Point<dim> point;

                  for (unsigned int d = 0; d < dim; ++d)
                    point[d] = point_batch[d][v];

                  for (unsigned int d = 0; d < dim; ++d)
                    target_velocity_value[d][v] =
                      fu_ptr->second->value(point, d);
                }

              face_target_velocity[face - n_inner_faces][q] =
                target_velocity_value;
            }
        }
    }


  if (has_ghost_elements == false)
    vec.zero_out_ghost_values();
}



template <int dim, typename Number>
double
NavierStokesOperator<dim, Number>::get_max_u(
  const VectorType<Number> &vec) const
{
  FEEvaluation<dim, -1, 0, dim, Number> integrator(matrix_free);
  FEEvaluation<dim, -1, 0, 1, Number> integrator_scalar(matrix_free, 0, 0, dim);

  const bool has_ghost_elements = vec.has_ghost_elements();

  AssertThrow(has_ghost_elements == false, ExcInternalError());

  if (has_ghost_elements == false)
    vec.update_ghost_values();

  VectorizedArray<Number> u_max = 0.0;

  for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
    {
      integrator.reinit(cell);

      integrator.read_dof_values_plain(vec);
      integrator.evaluate(EvaluationFlags::EvaluationFlags::values |
                          EvaluationFlags::EvaluationFlags::gradients);

      for (const auto q : integrator.quadrature_point_indices())
        u_max = std::max(integrator.get_value(q).norm(), u_max);
    }

  Number u_max_scalar = 0.0;

  for (const auto v : u_max)
    u_max_scalar = std::max(u_max_scalar, v);

  if (has_ghost_elements == false)
    vec.zero_out_ghost_values();

  return Utilities::MPI::max(u_max_scalar, MPI_COMM_WORLD);
}

template <int dim, typename Number>
void
NavierStokesOperator<dim, Number>::set_linearization_point(
  const VectorType<Number> &vec)
{
  MyScope scope(timer, "ns::set_linearization_point");

  this->valid_system = false;

  const unsigned n_cells             = matrix_free.n_cell_batches();
  const unsigned n_quadrature_points = matrix_free.get_quadrature().size();

  u_star_value.reinit(n_cells, n_quadrature_points);
  u_star_gradient.reinit(n_cells, n_quadrature_points);
  p_star_gradient.reinit(n_cells, n_quadrature_points);

  FEEvaluation<dim, -1, 0, dim, Number> integrator(matrix_free);
  FEEvaluation<dim, -1, 0, 1, Number> integrator_scalar(matrix_free, 0, 0, dim);

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

  if (compute_penalty_parameters_for_previous_solution == false)
    this->compute_penalty_parameters(vec);
}

template <int dim, typename Number>
void
NavierStokesOperator<dim, Number>::evaluate_rhs(VectorType<Number> &dst) const
{
  MyScope scope(timer, "ns::evaluate_rhs");

  // apply inhomogeneous DBC
  VectorType<Number> src;
  src.reinit(dst);
  constraints_inhomogeneous.distribute(src);

  // perform vmult
  this->matrix_free.cell_loop(
    &NavierStokesOperator<dim, Number>::do_vmult_range<true>,
    this,
    dst,
    src,
    true);

  // apply constraints
  matrix_free.get_affine_constraints(0).set_zero(dst);

  // move to rhs
  dst *= -1.0;
}

template <int dim, typename Number>
void
NavierStokesOperator<dim, Number>::evaluate_residual(
  VectorType<Number>       &dst,
  const VectorType<Number> &src) const
{
  MyScope scope(timer, "ns::evaluate_residual");

  // apply inhomogeneous DBC
  VectorType<Number> tmp = src;              // TODO: needed?
  constraints_inhomogeneous.distribute(tmp); //

  if (!needs_face_integrals)
    this->matrix_free.cell_loop(
      &NavierStokesOperator<dim, Number>::do_vmult_range<true>,
      this,
      dst,
      tmp,
      true);
  else
    this->matrix_free.loop(
      &NavierStokesOperator<dim, Number>::do_vmult_range<true>,
      &NavierStokesOperator<dim, Number>::do_vmult_face_range<true>,
      &NavierStokesOperator<dim, Number>::do_vmult_boundary_range<true>,
      this,
      dst,
      tmp,
      true);

  // apply constraints
  matrix_free.get_affine_constraints(0).set_zero(dst);

  // move to rhs
  dst *= -1.0;
}

template <int dim, typename Number>
void
NavierStokesOperator<dim, Number>::vmult(VectorType<Number>       &dst,
                                         const VectorType<Number> &src) const
{
  MyScope scope(timer, "ns::vmult");

  // save values for edge constrained dofs and set them to 0 in src vector
  for (unsigned int i = 0; i < edge_constrained_indices.size(); ++i)
    {
      edge_constrained_values[i] = std::pair<Number, Number>(
        src.local_element(edge_constrained_indices[i]),
        dst.local_element(edge_constrained_indices[i]));

      const_cast<LinearAlgebra::distributed::Vector<Number> &>(src)
        .local_element(edge_constrained_indices[i]) = 0.;
    }

  if (!needs_face_integrals)
    this->matrix_free.cell_loop(
      &NavierStokesOperator<dim, Number>::do_vmult_range<false>,
      this,
      dst,
      src,
      true);
  else
    this->matrix_free.loop(
      &NavierStokesOperator<dim, Number>::do_vmult_range<false>,
      &NavierStokesOperator<dim, Number>::do_vmult_face_range<false>,
      &NavierStokesOperator<dim, Number>::do_vmult_boundary_range<false>,
      this,
      dst,
      src,
      true);

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

template <int dim, typename Number>
void
NavierStokesOperator<dim, Number>::vmult_interface_down(
  VectorType<Number>       &dst,
  const VectorType<Number> &src) const
{
  MyScope scope(timer, "ns::vmult_interface_down");

  this->matrix_free.cell_loop(
    &NavierStokesOperator<dim, Number>::do_vmult_range<false>,
    this,
    dst,
    src,
    true);

  // set constrained dofs as the sum of current dst value and src value
  for (unsigned int i = 0; i < constrained_indices.size(); ++i)
    dst.local_element(constrained_indices[i]) =
      src.local_element(constrained_indices[i]);
}

template <int dim, typename Number>
void
NavierStokesOperator<dim, Number>::vmult_interface_up(
  VectorType<Number>       &dst,
  const VectorType<Number> &src) const
{
  MyScope scope(timer, "ns::vmult_interface_up");

  if (has_edge_constrained_indices == false)
    {
      dst = Number(0.);
      return;
    }

  dst = 0.0;

  // make a copy of src vector and set everything to 0 except edge
  // constrained dofs
  VectorType<Number> src_cpy;
  src_cpy.reinit(src, /*omit_zeroing_entries=*/false);

  for (unsigned int i = 0; i < edge_constrained_indices.size(); ++i)
    src_cpy.local_element(edge_constrained_indices[i]) =
      src.local_element(edge_constrained_indices[i]);

  // do loop with copy of src
  this->matrix_free.cell_loop(
    &NavierStokesOperator<dim, Number>::do_vmult_range<false>,
    this,
    dst,
    src_cpy,
    false);
}

template <int dim, typename Number>
const SparseMatrixType &
NavierStokesOperator<dim, Number>::get_system_matrix() const
{
  initialize_system_matrix();

  return system_matrix;
}

template <int dim, typename Number>
void
NavierStokesOperator<dim, Number>::initialize_dof_vector(
  VectorType<Number> &vec) const
{
  matrix_free.initialize_dof_vector(vec);
}

template <int dim, typename Number>
template <bool evaluate_residual>
void
NavierStokesOperator<dim, Number>::do_vmult_range(
  const MatrixFree<dim, Number>               &matrix_free,
  VectorType<Number>                          &dst,
  const VectorType<Number>                    &src,
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

template <int dim, typename Number>
template <bool evaluate_residual>
void
NavierStokesOperator<dim, Number>::do_vmult_face_range(
  const MatrixFree<dim, Number>               &matrix_free,
  VectorType<Number>                          &dst,
  const VectorType<Number>                    &src,
  const std::pair<unsigned int, unsigned int> &range) const
{
  (void)matrix_free;
  (void)dst;
  (void)src;
  (void)range;

  // nothing to do
}

template <int dim, typename Number>
template <bool evaluate_residual>
void
NavierStokesOperator<dim, Number>::do_vmult_boundary_range(
  const MatrixFree<dim, Number>               &matrix_free,
  VectorType<Number>                          &dst,
  const VectorType<Number>                    &src,
  const std::pair<unsigned int, unsigned int> &range) const
{
  FEFaceIntegrator phi(matrix_free, true, 0);

  for (auto cell = range.first; cell < range.second; ++cell)
    {
      phi.reinit(cell);

      if ((all_outflow_bcs_cut.find(phi.boundary_id()) ==
           all_outflow_bcs_cut.end()) &&
          (all_outflow_bcs_nitsche.find(phi.boundary_id()) ==
           all_outflow_bcs_nitsche.end()))
        continue;

      if (evaluate_residual)
        phi.read_dof_values_plain(src);
      else
        phi.read_dof_values(src);

      do_vmult_boundary<evaluate_residual>(phi);

      phi.distribute_local_to_global(dst);
    }
}

namespace
{
  /**
   * compute: ε(v) : ε(u) <-> ∑_i∑_j v_{i,j}u_{i,j}
   *
   * example:
   *   ε(u) =
   *    |         u_{1,1}           (u_{1,2} + u_{1,2}) / 2 |
   *    | (u_{1,2} + u_{1,2}) / 2           u_{2,2}         |
   *
   *   ε(v) =
   *    |         v_{1,1}           (v_{1,2} + v_{1,2}) / 2 |
   *    | (v_{1,2} + v_{1,2}) / 2           v_{2,2}         |
   *
   *   ε(v) : ε(u) =
   *     v_{1,1} u_{1,1} + v_{2,2} u_{2,2} +
   *     (v_{1,2} + v_{2,1}) * (u_{1,2} + u_{2,1}) / 2
   */
  template <int dim, int dim_, typename Number>
  inline DEAL_II_ALWAYS_INLINE void
  symm_scalar_product_add(Tensor<1, dim_, Tensor<1, dim, Number>> &v_gradient,
                          const Tensor<2, dim, Number>            &u_gradient,
                          const Number                            &factor)
  {
    for (unsigned int d = 0; d < dim; ++d)
      v_gradient[d][d] += u_gradient[d][d] * factor;

    for (unsigned int e = 0; e < dim; ++e)
      for (unsigned int d = e + 1; d < dim; ++d)
        {
          const auto tmp =
            (u_gradient[d][e] + u_gradient[e][d]) * (factor * 0.5);
          v_gradient[d][e] += tmp;
          v_gradient[e][d] += tmp;
        }
  }
} // namespace

/**
 * Fixed-point system:
 *
 * (v, ∂t(u)) + (v, S⋅∇B) - (div(v), p) + (ε(v), 2νε(B))
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
 *  - ∂t(u) := time derivative (one-step-theta method, BDF)
 *
 *
 * Linearized system (only BDF):
 *
 * (v, ∂t'(u) + U⋅∇u + u⋅∇U) - (div(v), p) + (ε(v), 2νε(u))
 *            + δ_1 (U⋅∇v, ∂t'(u) + U⋅∇u + u⋅∇U + ∇p) -> SUPG (1)
 *            + δ_1 (u⋅∇v, ∂t'(U) + U⋅∇U + ∇P)        -> SUPG (2)
 *            + δ_2 (div(v), div(u))                  -> GD
 *
 * (q, div(u)) + δ_1 (∇q, ∂t'(u) + U⋅∇u + u⋅∇U + ∇p)
 *               +-------------- PSPG -------------+
 *
 *                       ... with U/P being the linearization point
 */
template <int dim, typename Number>
template <bool evaluate_residual>
void
NavierStokesOperator<dim, Number>::do_vmult_cell(
  FECellIntegrator &integrator) const
{
  if (evaluate_residual || !this->increment_form)
    {
      const unsigned int cell = integrator.get_current_cell_index();
      const auto weight       = this->time_integrator_data.get_primary_weight();
      const auto theta        = this->theta;
      const auto nu           = this->nu;

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

          if (evaluate_residual && (this->u_time_derivative_old.size(0) > 0))
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

          //  d)  (ε(v), 2νε(B))
          symm_scalar_product_add(gradient_result, u_bar_gradient, nu * 2.0);

          //  e)  δ_1 (S⋅∇v, ∂t(u) + ∇P + S⋅∇B) -> SUPG stabilization
          const auto residual_0 =
            delta_1 * ((consider_time_derivative ?
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
            delta_1 * ((consider_time_derivative ?
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
            value_result[d] = u_time_derivative[d] + s_grad_u[d] + u_grad_s[d];

          //  b)  - (div(v), p)
          for (unsigned int d = 0; d < dim; ++d)
            gradient_result[d][d] -= p_value;

          //  c)  (ε(v), 2νε(u))
          symm_scalar_product_add(gradient_result, u_gradient, nu * 2.0);

          //  d)  δ_1 (U⋅∇v, ∂t'(u) + ∇p + U⋅∇u + u⋅∇U) +
          //      δ_1 (u⋅∇v, ∂t'(U) + ∇P + U⋅∇U) -> SUPG stabilization
          const auto residual_0 =
            delta_1 * ((consider_time_derivative ?
                          u_time_derivative :
                          Tensor<1, dim, VectorizedArray<Number>>()) +
                       p_gradient + s_grad_u + u_grad_s);
          const auto residual_1 =
            delta_1 *
            ((consider_time_derivative ?
                (u_star_value * weight + this->u_time_derivative_old[cell][q]) :
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
            delta_1 * ((consider_time_derivative ?
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

template <int dim, typename Number>
template <bool evaluate_residual>
void
NavierStokesOperator<dim, Number>::do_vmult_face(
  FEFaceIntegrator &integrator) const
{
  (void)integrator;

  // nothing to do
}

template <int dim, typename Number>
template <bool evaluate_residual>
void
NavierStokesOperator<dim, Number>::do_vmult_boundary(
  FEFaceIntegrator &integrator) const
{
  if (all_outflow_bcs_cut.find(integrator.boundary_id()) !=
      all_outflow_bcs_cut.end())
    {
      const auto         face       = integrator.get_current_cell_index();
      const unsigned int face_index = face - matrix_free.n_inner_face_batches();

      const auto penalty_parameter = this->effective_beta_face[face_index];

      integrator.evaluate(EvaluationFlags::EvaluationFlags::values);

      for (const auto q : integrator.quadrature_point_indices())
        {
          typename FEFaceIntegrator::value_type value_result = {};

          const auto value = integrator.get_value(q);

          Tensor<1, dim, VectorizedArray<Number>> star_velocity;

          if (evaluate_residual)
            {
              for (unsigned int d = 0; d < dim; ++d)
                star_velocity[d] = value[d];
            }
          else
            {
              star_velocity = face_velocity[face_index][q];
            }

          const VectorizedArray<Number> zero = 0.0;

          VectorizedArray<Number> normal_outflux =
            star_velocity * integrator.get_normal_vector(q);
          normal_outflux = std::min(zero, normal_outflux);

          for (unsigned int d = 0; d < dim; ++d)
            value_result[d] = penalty_parameter * normal_outflux * value[d];

          integrator.submit_value(value_result, q);
        }

      integrator.integrate(EvaluationFlags::EvaluationFlags::values);
    }
  else if (all_outflow_bcs_nitsche.find(integrator.boundary_id()) !=
           all_outflow_bcs_nitsche.end())
    {
      // adopted from:
      // https://github.com/chaos-polymtl/lethe/blob/9e25b0bd06dfd9392baf798ed3758abaee6dbb9e/source/solvers/fluid_dynamics_matrix_free_operators.cc#L1241

      const auto         face       = integrator.get_current_cell_index();
      const unsigned int face_index = face - matrix_free.n_inner_face_batches();

      const auto penalty_parameter = this->effective_beta_face[face_index];

      integrator.evaluate(EvaluationFlags::EvaluationFlags::values |
                          EvaluationFlags::EvaluationFlags::gradients);

      for (const auto q : integrator.quadrature_point_indices())
        {
          typename FEFaceIntegrator::value_type    value_result    = {};
          typename FEFaceIntegrator::gradient_type gradient_result = {};

          const auto normal_vector = integrator.get_normal_vector(q);

          auto       value    = integrator.get_value(q);
          const auto gradient = integrator.get_gradient(q);

          // If we are assembling the residual, substract the target velocity
          // from the velocity value.
          if constexpr (evaluate_residual)
            value -= this->face_target_velocity[face_index][q];

          for (unsigned int d = 0; d < dim; ++d)
            {
              // Assemble (v,beta (u-u_target))
              value_result[d] += penalty_parameter * value[d];

              // Assemble ν(v,∇δu·n)
              for (unsigned int i = 0; i < dim; ++i)
                value_result[d] -= nu * gradient[d][i] * normal_vector[i];

              // Assemble ν(∇v·n,(u-u_target))
              for (unsigned int i = 0; i < dim; ++i)
                gradient_result[d][i] -= nu * value[d] * normal_vector[i];
            }

          integrator.submit_value(value_result, q);
          integrator.submit_gradient(gradient_result, q);
        }
      integrator.integrate(EvaluationFlags::EvaluationFlags::values |
                           EvaluationFlags::EvaluationFlags::gradients);
    }
  else
    {
      const VectorizedArray<Number> zero = 0.0;

      for (unsigned int i = 0; i < integrator.dofs_per_cell; ++i)
        integrator.begin_dof_values()[i] = zero;

      return;
    }
}

template <int dim, typename Number>
void
NavierStokesOperator<dim, Number>::initialize_system_matrix() const
{
  MyScope scope(timer, "ns::initialize_system_matrix");

  const bool keep_constrained_dofs = false;

  const auto &dof_handler = matrix_free.get_dof_handler();
  const auto &constraints = matrix_free.get_affine_constraints();

  if (system_matrix.m() == 0 || system_matrix.n() == 0)
    {
      system_matrix.clear();

      TrilinosWrappers::SparsityPattern dsp;

      const unsigned int mg_level = this->matrix_free.get_mg_level();

      dsp.reinit(mg_level != numbers::invalid_unsigned_int ?
                   dof_handler.locally_owned_mg_dofs(mg_level) :
                   dof_handler.locally_owned_dofs(),
                 dof_handler.get_communicator());

      if (mg_level != numbers::invalid_unsigned_int)
        {
          // the following code does the same as
          // MGTools::make_sparsity_pattern() but also
          // consideres bool_dof_mask for FE_Q_iso_Q1
          std::vector<types::global_dof_index> dofs_on_this_cell;

          for (const auto &cell : dof_handler.cell_iterators_on_level(mg_level))
            if (cell->is_locally_owned_on_level())
              {
                const unsigned int dofs_per_cell =
                  dof_handler.get_fe().n_dofs_per_cell();
                dofs_on_this_cell.resize(dofs_per_cell);
                cell->get_mg_dof_indices(dofs_on_this_cell);

                constraints.add_entries_local_to_global(dofs_on_this_cell,
                                                        dsp,
                                                        keep_constrained_dofs,
                                                        bool_dof_mask);
              }
        }
      else
        {
          // the following code does the same as
          // DoFTools::make_sparsity_pattern() but also
          // consideres bool_dof_mask for FE_Q_iso_Q1
          std::vector<types::global_dof_index> dofs_on_this_cell;

          for (const auto &cell : dof_handler.active_cell_iterators())
            if (cell->is_locally_owned())
              {
                const unsigned int dofs_per_cell =
                  cell->get_fe().n_dofs_per_cell();
                dofs_on_this_cell.resize(dofs_per_cell);
                cell->get_dof_indices(dofs_on_this_cell);

                constraints.add_entries_local_to_global(dofs_on_this_cell,
                                                        dsp,
                                                        keep_constrained_dofs,
                                                        bool_dof_mask);
              }
        }

      dsp.compress();

      system_matrix.reinit(dsp);
    }

  if (this->valid_system == false)
    {
      system_matrix = 0.0;

      const auto &lexicographic_numbering =
        matrix_free.get_shape_info().lexicographic_numbering;

      unsigned int cell   = numbers::invalid_unsigned_int;
      unsigned int face   = numbers::invalid_unsigned_int;
      unsigned int column = numbers::invalid_unsigned_int;

      std::function<void(FEFaceIntegrator &)> boundary_function;

      if (needs_face_integrals)
        boundary_function = [&](auto &integrator) {
          if (face != integrator.get_current_cell_index())
            {
              face   = integrator.get_current_cell_index();
              column = 0;
            }

          this->template do_vmult_boundary<false>(integrator);

          // remove spurious entries for FE_Q_iso_Q1
          for (unsigned int i = 0; i < lexicographic_numbering.size(); ++i)
            if (!bool_dof_mask[lexicographic_numbering[i]]
                              [lexicographic_numbering[column]])
              integrator.begin_dof_values()[i] = 0.0;

          column++;
        };

      MatrixFreeTools::
        compute_matrix<dim, -1, 0, dim + 1, Number, VectorizedArray<Number>>(
          matrix_free,
          constraints,
          system_matrix,
          [&](auto &integrator) {
            if (cell != integrator.get_current_cell_index())
              {
                cell   = integrator.get_current_cell_index();
                column = 0;
              }

            this->template do_vmult_cell<false>(integrator);

            // remove spurious entries for FE_Q_iso_Q1
            for (unsigned int i = 0; i < lexicographic_numbering.size(); ++i)
              if (!bool_dof_mask[lexicographic_numbering[i]]
                                [lexicographic_numbering[column]])
                integrator.begin_dof_values()[i] = 0.0;

            column++;
          },
          {},
          boundary_function);

      this->valid_system = true;
    }
}

template <int dim, typename Number>
IndexSet
NavierStokesOperator<dim, Number>::get_refinement_edges(
  const MatrixFree<dim, Number> &matrix_free)
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



/**
 * Matrix-based Navier-Stokes operator.
 */
template <int dim, typename Number>
NavierStokesOperatorMatrixBased<dim, Number>::NavierStokesOperatorMatrixBased(
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

  dsp.reinit(dof_handler.locally_owned_dofs(), dof_handler.get_communicator());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
  dsp.compress();

  system_matrix.reinit(dsp);
}

template <int dim, typename Number>
const AffineConstraints<Number> &
NavierStokesOperatorMatrixBased<dim, Number>::get_constraints() const
{
  return constraints;
}

template <int dim, typename Number>
types::global_dof_index
NavierStokesOperatorMatrixBased<dim, Number>::m() const
{
  AssertThrow(false, ExcNotImplemented());

  return 0;
}

template <int dim, typename Number>
void
NavierStokesOperatorMatrixBased<dim, Number>::compute_inverse_diagonal(
  VectorType<Number> &) const
{
  AssertThrow(false, ExcNotImplemented());
}

template <int dim, typename Number>
void
NavierStokesOperatorMatrixBased<dim, Number>::invalidate_system()
{
  this->valid_system = false;
}

template <int dim, typename Number>
void
NavierStokesOperatorMatrixBased<dim, Number>::set_previous_solution(
  const SolutionHistory<Number> &vectors)
{
  const auto &vec = vectors.get_vectors()[1];

  this->previous_solution = vec;
  this->previous_solution.update_ghost_values();

  this->valid_system = false;
}

template <int dim, typename Number>
void
NavierStokesOperatorMatrixBased<dim, Number>::set_linearization_point(
  const VectorType<Number> &src)
{
  this->linearization_point = src;
  this->linearization_point.update_ghost_values();

  this->valid_system = false;
}

template <int dim, typename Number>
void
NavierStokesOperatorMatrixBased<dim, Number>::evaluate_rhs(
  VectorType<Number> &dst) const
{
  compute_system_matrix_and_vector();
  dst = system_rhs;
}

template <int dim, typename Number>
void
NavierStokesOperatorMatrixBased<dim, Number>::evaluate_residual(
  VectorType<Number>       &dst,
  const VectorType<Number> &src) const
{
  (void)dst;
  (void)src;
}

template <int dim, typename Number>
void
NavierStokesOperatorMatrixBased<dim, Number>::vmult(
  VectorType<Number>       &dst,
  const VectorType<Number> &src) const
{
  get_system_matrix().vmult(dst, src);
}

template <int dim, typename Number>
const SparseMatrixType &
NavierStokesOperatorMatrixBased<dim, Number>::get_system_matrix() const
{
  compute_system_matrix_and_vector();
  return system_matrix;
}

template <int dim, typename Number>
void
NavierStokesOperatorMatrixBased<dim, Number>::initialize_dof_vector(
  VectorType<Number> &src) const
{
  src.reinit(partitioner);
}

template <int dim, typename Number>
void
NavierStokesOperatorMatrixBased<dim, Number>::compute_system_matrix_and_vector()
  const
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
          delta_1 = c_1 / std::sqrt(1. / (tau * tau) + u_max * u_max / (h * h));
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
                    cell_lhs += theta * tau * 2.0 * nu * scalar_product(eps_u_j, eps_v_i);                                // d
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
                cell_rhs -= (1.0 - theta) * tau * 2.0 * nu * scalar_product(symmetrize(grad_u_0[q]), eps_v_i);                      // d
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


template class NavierStokesOperator<2, double>;
template class NavierStokesOperator<3, double>;
template class NavierStokesOperator<2, float>;
template class NavierStokesOperator<3, float>;
template class NavierStokesOperatorMatrixBased<2, double>;
template class NavierStokesOperatorMatrixBased<3, double>;
