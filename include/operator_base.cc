#include "operator_base.h"



Number
OperatorBase::el(unsigned int, unsigned int) const
{
  Assert(false, ExcNotImplemented());
  return 0;
}

void
OperatorBase::Tvmult(VectorType &dst, const VectorType &src) const
{
  vmult(dst, src);
}

void
OperatorBase::vmult_interface_down(VectorType &dst, const VectorType &src) const
{
  AssertThrow(false, ExcNotImplemented());

  (void)dst;
  (void)src;
}

void
OperatorBase::vmult_interface_up(VectorType &dst, const VectorType &src) const
{
  AssertThrow(false, ExcNotImplemented());

  (void)dst;
  (void)src;
}

std::vector<std::vector<bool>>
OperatorBase::extract_constant_modes() const
{
  AssertThrow(false, ExcNotImplemented());
  return {};
}

double
OperatorBase::get_max_u(const VectorType &src) const
{
  (void)src;
  return 1.0; // TODO
}
