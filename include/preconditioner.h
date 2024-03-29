#pragma once

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
