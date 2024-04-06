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
    const unsigned int               mg_level = numbers::invalid_unsigned_int);

  const AffineConstraints<Number> &
  get_constraints() const override;

  virtual types::global_dof_index
  m() const override;

  std::vector<std::vector<bool>>
  extract_constant_modes() const override;

  virtual void
  compute_inverse_diagonal(VectorType &diagonal) const override;

  void
  invalidate_system() override;

  void
  set_previous_solution(const SolutionHistory &history) override;

  void
  set_previous_solution(const VectorType &vec) override;

  void
  set_linearization_point(const VectorType &vec) override;

  void
  evaluate_rhs(VectorType &dst) const override;

  virtual void
  evaluate_residual(VectorType &dst, const VectorType &src) const override;

  void
  vmult(VectorType &dst, const VectorType &src) const override;

  void
  vmult_interface_down(VectorType &dst, const VectorType &src) const override;

  void
  vmult_interface_up(VectorType &dst, const VectorType &src) const override;

  const SparseMatrixType &
  get_system_matrix() const override;

  void
  initialize_dof_vector(VectorType &vec) const override;

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
                 const std::pair<unsigned int, unsigned int> &range) const;

  template <bool evaluate_residual>
  void
  do_vmult_cell(FECellIntegrator &integrator) const;

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
    const TimeIntegratorData        &time_integrator_data);

  const AffineConstraints<Number> &
  get_constraints() const override;

  types::global_dof_index
  m() const override;

  void
  compute_inverse_diagonal(VectorType &) const override;

  void
  invalidate_system() override;

  void
  set_previous_solution(const SolutionHistory &vec) override;

  void
  set_previous_solution(const VectorType &vec) override;

  void
  set_linearization_point(const VectorType &src) override;

  void
  evaluate_rhs(VectorType &dst) const override;

  virtual void
  evaluate_residual(VectorType &dst, const VectorType &src) const override;

  void
  vmult(VectorType &dst, const VectorType &src) const override;

  const SparseMatrixType &
  get_system_matrix() const override;

  void
  initialize_dof_vector(VectorType &src) const override;

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
  compute_system_matrix_and_vector() const;
};
