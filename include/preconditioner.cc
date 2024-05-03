#include "preconditioner.h"



PreconditionerILU::PreconditionerILU(const OperatorBase<Number> &op)
  : op(op)
{}

void
PreconditionerILU::initialize()
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
PreconditionerILU::vmult(VectorType<Number>       &dst,
                         const VectorType<Number> &src) const
{
  precon.vmult(dst, src);
}

void
PreconditionerILU::print_stats() const
{
  // nothing to do
}



PreconditionerAMG::PreconditionerAMG(
  const OperatorBase<Number>           &op,
  const std::vector<std::vector<bool>> &constant_modes)
  : op(op)
  , constant_modes(constant_modes)
{}

void
PreconditionerAMG::initialize()
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
PreconditionerAMG::vmult(VectorType<Number>       &dst,
                         const VectorType<Number> &src) const
{
  precon.vmult(dst, src);
}

void
PreconditionerAMG::print_stats() const
{
  // nothing to do
}
