#include "operator_base.h"


template <typename Number>
Number
OperatorBase<Number>::el(unsigned int, unsigned int) const
{
  Assert(false, ExcNotImplemented());
  return 0;
}

template <typename Number>
void
OperatorBase<Number>::Tvmult(VectorType<Number>       &dst,
                             const VectorType<Number> &src) const
{
  vmult(dst, src);
}

template <typename Number>
void
OperatorBase<Number>::vmult_interface_down(VectorType<Number>       &dst,
                                           const VectorType<Number> &src) const
{
  AssertThrow(false, ExcNotImplemented());

  (void)dst;
  (void)src;
}

template <typename Number>
void
OperatorBase<Number>::vmult_interface_up(VectorType<Number>       &dst,
                                         const VectorType<Number> &src) const
{
  AssertThrow(false, ExcNotImplemented());

  (void)dst;
  (void)src;
}

template <typename Number>
std::vector<std::vector<bool>>
OperatorBase<Number>::extract_constant_modes() const
{
  AssertThrow(false, ExcNotImplemented());
  return {};
}

template <typename Number>
double
OperatorBase<Number>::get_max_u(const VectorType<Number> &src) const
{
  (void)src;
  return 1.0; // TODO
}

template class OperatorBase<double>;
template class OperatorBase<float>;
