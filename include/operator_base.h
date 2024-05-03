#pragma once

#include <deal.II/lac/affine_constraints.h>

#include "config.h"
#include "time_integration.h"

using namespace dealii;

/**
 * Linear/nonlinear operator.
 */
template <typename Number = double>
class OperatorBase : public Subscriptor
{
public:
  using value_type = Number;
  using size_type  = types::global_dof_index;

  virtual types::global_dof_index
  m() const = 0;

  Number
  el(unsigned int, unsigned int) const;

  virtual void
  compute_inverse_diagonal(VectorType<Number> &diagonal) const = 0;

  virtual void
  invalidate_system() = 0;

  virtual void
  set_previous_solution(const SolutionHistory<Number> &vec) = 0;

  virtual void
  set_linearization_point(const VectorType<Number> &src) = 0;

  virtual void
  evaluate_rhs(VectorType<Number> &dst) const = 0;

  virtual void
  evaluate_residual(VectorType<Number>       &dst,
                    const VectorType<Number> &src) const = 0;

  virtual void
  vmult(VectorType<Number> &dst, const VectorType<Number> &src) const = 0;

  void
  Tvmult(VectorType<Number> &dst, const VectorType<Number> &src) const;

  virtual void
  vmult_interface_down(VectorType<Number>       &dst,
                       const VectorType<Number> &src) const;

  virtual void
  vmult_interface_up(VectorType<Number>       &dst,
                     const VectorType<Number> &src) const;

  virtual std::vector<std::vector<bool>>
  extract_constant_modes() const;

  virtual const AffineConstraints<Number> &
  get_constraints() const = 0;

  virtual const SparseMatrixType &
  get_system_matrix() const = 0;

  virtual void
  initialize_dof_vector(VectorType<Number> &src) const = 0;

  virtual double
  get_max_u(const VectorType<Number> &src) const;
};
