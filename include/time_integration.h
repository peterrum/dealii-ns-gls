#pragma once

#include "config.h"

using namespace dealii;

/**
 * Base class for BDF and θ-method.
 */
class TimeIntegratorData
{
public:
  virtual void
  update_dt(const Number dt_new) = 0;

  virtual Number
  get_primary_weight() const = 0;

  virtual const std::vector<Number> &
  get_weights() const = 0;

  virtual unsigned int
  get_order() const = 0;

  virtual Number
  get_current_dt() const = 0;

  virtual Number
  get_theta() const = 0;
};



/**
 * BDF implementation.
 */
class TimeIntegratorDataBDF : public TimeIntegratorData
{
public:
  TimeIntegratorDataBDF(const unsigned int order)
    : order(order)
    , dt(order)
    , weights(order + 1)
  {}

  void
  update_dt(const Number dt_new) override
  {
    for (int i = get_order() - 2; i >= 0; i--)
      {
        dt[i + 1] = dt[i];
      }

    dt[0] = dt_new;

    update_weights();
  }

  Number
  get_primary_weight() const override
  {
    return weights[0];
  }

  const std::vector<Number> &
  get_weights() const override
  {
    return weights;
  }

  unsigned int
  get_order() const override
  {
    return order;
  }

  Number
  get_current_dt() const override
  {
    return dt[0];
  }

  Number
  get_theta() const override
  {
    return 1.0;
  }

private:
  unsigned int
  effective_order() const
  {
    return std::count_if(dt.begin(), dt.end(), [](const auto &v) {
      return v > 0;
    });
  }

  void
  update_weights()
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

  unsigned int        order;
  std::vector<Number> dt;
  std::vector<Number> weights;
};



/**
 * θ-method implementation.
 */
class TimeIntegratorDataTheta : public TimeIntegratorData
{
public:
  TimeIntegratorDataTheta(const Number theta)
    : theta(theta)
    , weights(2)
  {}

  void
  update_dt(const Number dt_new) override
  {
    this->dt = dt_new;

    weights[0] = +1.0 / this->dt;
    weights[1] = -1.0 / this->dt;
  }

  Number
  get_primary_weight() const override
  {
    return weights[0];
  }

  const std::vector<Number> &
  get_weights() const override
  {
    return weights;
  }

  unsigned int
  get_order() const override
  {
    return 1;
  }

  Number
  get_current_dt() const override
  {
    return dt;
  }

  Number
  get_theta() const override
  {
    return theta;
  }

private:
  Number              theta;
  Number              dt;
  std::vector<Number> weights;
};



/**
 * A container storing solution vectors of multiple time steps.
 */
class SolutionHistory
{
public:
  SolutionHistory(const unsigned int size)
    : solutions(size)
  {}

  VectorType &
  get_current_solution()
  {
    return solutions[0];
  }

  std::vector<VectorType> &
  get_vectors()
  {
    return solutions;
  }

  const std::vector<VectorType> &
  get_vectors() const
  {
    return solutions;
  }

  void
  commit_solution()
  {
    for (int i = solutions.size() - 2; i >= 0; --i)
      solutions[i + 1].copy_locally_owned_data_from(solutions[i]);
  }

  std::vector<VectorType> solutions;
};
