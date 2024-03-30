#pragma once

#include <deal.II/lac/affine_constraints.h>

#include "config.h"
#include "time_integration.h"

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
