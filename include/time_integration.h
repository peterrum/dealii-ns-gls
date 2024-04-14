#pragma once

#include "config.h"

using namespace dealii;

/**
 * Base class for BDF and θ-method.
 */
class TimeIntegratorData
{
public:
  virtual ~TimeIntegratorData() = default;

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
  TimeIntegratorDataBDF(const unsigned int order);

  void
  update_dt(const Number dt_new) override;

  Number
  get_primary_weight() const override;

  const std::vector<Number> &
  get_weights() const override;

  unsigned int
  get_order() const override;

  Number
  get_current_dt() const override;

  Number
  get_theta() const override;

private:
  unsigned int
  effective_order() const;

  void
  update_weights();

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
  TimeIntegratorDataTheta(const Number theta);

  void
  update_dt(const Number dt_new) override;

  Number
  get_primary_weight() const override;

  const std::vector<Number> &
  get_weights() const override;

  unsigned int
  get_order() const override;

  Number
  get_current_dt() const override;

  Number
  get_theta() const override;

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
  SolutionHistory(const unsigned int size);

  VectorType &
  get_current_solution();

  std::vector<VectorType> &
  get_vectors();

  const std::vector<VectorType> &
  get_vectors() const;

  void
  commit_solution();

  std::vector<VectorType> solutions;
};
