#include "time_integration.h"


TimeIntegratorDataBDF::TimeIntegratorDataBDF(const unsigned int order)
  : order(order)
  , dt(order)
  , weights(order + 1)
{}

void
TimeIntegratorDataBDF::update_dt(const Number dt_new)
{
  for (int i = get_order() - 2; i >= 0; i--)
    {
      dt[i + 1] = dt[i];
    }

  dt[0] = dt_new;

  update_weights();
}

Number
TimeIntegratorDataBDF::get_primary_weight() const
{
  return weights[0];
}

const std::vector<Number> &
TimeIntegratorDataBDF::get_weights() const
{
  return weights;
}

unsigned int
TimeIntegratorDataBDF::get_order() const
{
  return order;
}

Number
TimeIntegratorDataBDF::get_current_dt() const
{
  return dt[0];
}

Number
TimeIntegratorDataBDF::get_theta() const
{
  return 1.0;
}

unsigned int
TimeIntegratorDataBDF::effective_order() const
{
  return std::count_if(dt.begin(), dt.end(), [](const auto &v) {
    return v > 0;
  });
}

void
TimeIntegratorDataBDF::update_weights()
{
  std::fill(weights.begin(), weights.end(), 0);

  if (effective_order() == 3)
    {
      weights[1] = -(dt[0] + dt[1]) * (dt[0] + dt[1] + dt[2]) /
                   (dt[0] * dt[1] * (dt[1] + dt[2]));
      weights[2] =
        dt[0] * (dt[0] + dt[1] + dt[2]) / (dt[1] * dt[2] * (dt[0] + dt[1]));
      weights[3] = -dt[0] * (dt[0] + dt[1]) /
                   (dt[2] * (dt[1] + dt[2]) * (dt[0] + dt[1] + dt[2]));
      weights[0] = -(weights[1] + weights[2] + weights[3]);
    }
  else if (effective_order() == 2)
    {
      weights[0] = (2 * dt[0] + dt[1]) / (dt[0] * (dt[0] + dt[1]));
      weights[1] = -(dt[0] + dt[1]) / (dt[0] * dt[1]);
      weights[2] = dt[0] / (dt[1] * (dt[0] + dt[1]));
    }
  else if (effective_order() == 1)
    {
      weights[0] = 1.0 / dt[0];
      weights[1] = -1.0 / dt[0];
    }
  else
    {
      AssertThrow(effective_order() <= 3, ExcMessage("Not implemented"));
    }
}



TimeIntegratorDataTheta::TimeIntegratorDataTheta(const Number theta)
  : theta(theta)
  , weights(2)
{}

void
TimeIntegratorDataTheta::update_dt(const Number dt_new)
{
  this->dt = dt_new;

  weights[0] = +1.0 / this->dt;
  weights[1] = -1.0 / this->dt;
}

Number
TimeIntegratorDataTheta::get_primary_weight() const
{
  return weights[0];
}

const std::vector<Number> &
TimeIntegratorDataTheta::get_weights() const
{
  return weights;
}

unsigned int
TimeIntegratorDataTheta::get_order() const
{
  return 1;
}

Number
TimeIntegratorDataTheta::get_current_dt() const
{
  return dt;
}

Number
TimeIntegratorDataTheta::get_theta() const
{
  return theta;
}



TimeIntegratorDataNone::TimeIntegratorDataNone()
{}

void
TimeIntegratorDataNone::update_dt(const Number dt_new)
{
  (void)dt_new;
}

Number
TimeIntegratorDataNone::get_primary_weight() const
{
  return 0.0;
}

const std::vector<Number> &
TimeIntegratorDataNone::get_weights() const
{
  return weights;
}

unsigned int
TimeIntegratorDataNone::get_order() const
{
  return 0;
}

Number
TimeIntegratorDataNone::get_current_dt() const
{
  return 1.0;
}

Number
TimeIntegratorDataNone::get_theta() const
{
  return 1.0;
}



SolutionHistory::SolutionHistory(const unsigned int size)
  : solutions(size)
{}

VectorType<Number> &
SolutionHistory::get_current_solution()
{
  return solutions[0];
}

std::vector<VectorType<Number>> &
SolutionHistory::get_vectors()
{
  return solutions;
}

const std::vector<VectorType<Number>> &
SolutionHistory::get_vectors() const
{
  return solutions;
}

void
SolutionHistory::commit_solution()
{
  for (int i = solutions.size() - 2; i >= 0; --i)
    solutions[i + 1].copy_locally_owned_data_from(solutions[i]);
}
