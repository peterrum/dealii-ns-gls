#pragma once

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/multigrid/mg_tools.h>

#include "config.h"
#include "operator_base.h"
#include "timer.h"

using namespace dealii;

/**
 * Matrix-free Navier-Stokes operator.
 */
template <int dim, typename Number>
class NavierStokesOperator : public OperatorBase<Number>
{
public:
  using FECellIntegrator = FEEvaluation<dim, -1, 0, dim + 1, Number>;
  using FEFaceIntegrator = FEFaceEvaluation<dim, -1, 0, dim + 1, Number>;

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
    const unsigned int               mg_level = numbers::invalid_unsigned_int);

  const AffineConstraints<Number> &
  get_constraints() const override;

  virtual types::global_dof_index
  m() const override;

  std::vector<std::vector<bool>>
  extract_constant_modes() const override;

  virtual void
  compute_inverse_diagonal(VectorType<Number> &diagonal) const override;

  void
  invalidate_system() override;

  void
  set_previous_solution(const SolutionHistory<Number> &history) override;

  void
  compute_penalty_parameters(const VectorType<Number> &vec);

  void
  set_linearization_point(const VectorType<Number> &vec) override;

  void
  evaluate_rhs(VectorType<Number> &dst) const override;

  virtual void
  evaluate_residual(VectorType<Number>       &dst,
                    const VectorType<Number> &src) const override;

  void
  vmult(VectorType<Number> &dst, const VectorType<Number> &src) const override;

  void
  vmult_interface_down(VectorType<Number>       &dst,
                       const VectorType<Number> &src) const override;

  void
  vmult_interface_up(VectorType<Number>       &dst,
                     const VectorType<Number> &src) const override;

  const SparseMatrixType &
  get_system_matrix() const override;

  void
  initialize_dof_vector(VectorType<Number> &vec) const override;

  double
  get_max_u(const VectorType<Number> &src) const override;

private:
  const AffineConstraints<Number> &constraints_inhomogeneous;

  MatrixFree<dim, Number> matrix_free;

  VectorType<Number>       linearization_point;
  mutable SparseMatrixType system_matrix;

  const VectorizedArray<Number> theta;
  const VectorizedArray<Number> nu;
  const Number                  c_1;
  const Number                  c_2;
  const TimeIntegratorData     &time_integrator_data;
  const bool                    consider_time_deriverative;
  const bool                    increment_form;
  const bool                    cell_wise_stabilization;
  const bool compute_penalty_parameters_for_previous_solution;

  const Table<2, bool> bool_dof_mask;

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

  Table<2, Tensor<1, dim, VectorizedArray<Number>>> face_velocity;

  std::vector<unsigned int> constrained_indices;

  template <bool evaluate_residual>
  void
  do_vmult_range(const MatrixFree<dim, Number>               &matrix_free,
                 VectorType<Number>                          &dst,
                 const VectorType<Number>                    &src,
                 const std::pair<unsigned int, unsigned int> &range) const;

  template <bool evaluate_residual>
  void
  do_vmult_cell(FECellIntegrator &integrator) const;

  template <bool evaluate_residual>
  void
  do_vmult_face_range(const MatrixFree<dim, Number>               &matrix_free,
                      VectorType<Number>                          &dst,
                      const VectorType<Number>                    &src,
                      const std::pair<unsigned int, unsigned int> &range) const;

  template <bool evaluate_residual>
  void
  do_vmult_face(FEFaceIntegrator &integrator) const;

  template <bool evaluate_residual>
  void
  do_vmult_boundary_range(
    const MatrixFree<dim, Number>               &matrix_free,
    VectorType<Number>                          &dst,
    const VectorType<Number>                    &src,
    const std::pair<unsigned int, unsigned int> &range) const;

  template <bool evaluate_residual>
  void
  do_vmult_boundary(FEFaceIntegrator &integrator) const;

  void
  initialize_system_matrix() const;

  std::vector<unsigned int> edge_constrained_indices;

  bool has_edge_constrained_indices = false;

  mutable std::vector<std::pair<Number, Number>> edge_constrained_values;

  std::vector<bool> edge_constrained_cell;

  IndexSet
  get_refinement_edges(const MatrixFree<dim, Number> &matrix_free);

  mutable MyTimerOutput timer;
};



/**
 * Matrix-based Navier-Stokes operator.
 */
template <int dim, typename Number>
class NavierStokesOperatorMatrixBased : public OperatorBase<Number>
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
    const TimeIntegratorData        &time_integrator_data);

  const AffineConstraints<Number> &
  get_constraints() const override;

  types::global_dof_index
  m() const override;

  void
  compute_inverse_diagonal(VectorType<Number> &) const override;

  void
  invalidate_system() override;

  void
  set_previous_solution(const SolutionHistory<Number> &vec) override;

  void
  set_linearization_point(const VectorType<Number> &src) override;

  void
  evaluate_rhs(VectorType<Number> &dst) const override;

  virtual void
  evaluate_residual(VectorType<Number>       &dst,
                    const VectorType<Number> &src) const override;

  void
  vmult(VectorType<Number> &dst, const VectorType<Number> &src) const override;

  const SparseMatrixType &
  get_system_matrix() const override;

  void
  initialize_dof_vector(VectorType<Number> &src) const override;

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

  VectorType<Number> previous_solution;
  VectorType<Number> linearization_point;

  mutable SparseMatrixType   system_matrix;
  mutable VectorType<Number> system_rhs;

  std::shared_ptr<Utilities::MPI::Partitioner> partitioner;

  void
  compute_system_matrix_and_vector() const;
};
