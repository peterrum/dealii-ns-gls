#pragma once

#include <deal.II/lac/trilinos_precondition.h>

#include "config.h"
#include "operator_base.h"

using namespace dealii;

/**
 * Preconditioners.
 */
class PreconditionerBase
{
public:
  virtual ~PreconditionerBase() = default;

  virtual void
  initialize() = 0;

  virtual void
  vmult(VectorType<Number> &dst, const VectorType<Number> &src) const = 0;

  virtual void
  print_stats() const = 0;
};



class PreconditionerILU : public PreconditionerBase
{
public:
  PreconditionerILU(const OperatorBase<Number> &op);

  void
  initialize() override;

  void
  vmult(VectorType<Number> &dst, const VectorType<Number> &src) const override;

  void
  print_stats() const override;

private:
  const OperatorBase<Number> &op;

  TrilinosWrappers::PreconditionILU precon;
};



class PreconditionerAMG : public PreconditionerBase
{
public:
  PreconditionerAMG(const OperatorBase<Number>           &op,
                    const std::vector<std::vector<bool>> &constant_modes);

  void
  initialize() override;

  void
  vmult(VectorType<Number> &dst, const VectorType<Number> &src) const override;

  void
  print_stats() const override;

private:
  const OperatorBase<Number> &op;

  const std::vector<std::vector<bool>> constant_modes;

  TrilinosWrappers::PreconditionAMG precon;
};
