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
  virtual void
  initialize() = 0;

  virtual void
  vmult(VectorType &dst, const VectorType &src) const = 0;
};



class PreconditionerILU : public PreconditionerBase
{
public:
  PreconditionerILU(const OperatorBase &op);

  void
  initialize() override;

  void
  vmult(VectorType &dst, const VectorType &src) const override;

private:
  const OperatorBase &op;

  TrilinosWrappers::PreconditionILU precon;
};



class PreconditionerAMG : public PreconditionerBase
{
public:
  PreconditionerAMG(const OperatorBase                   &op,
                    const std::vector<std::vector<bool>> &constant_modes);

  void
  initialize() override;

  void
  vmult(VectorType &dst, const VectorType &src) const override;

private:
  const OperatorBase &op;

  const std::vector<std::vector<bool>> constant_modes;

  TrilinosWrappers::PreconditionAMG precon;
};
