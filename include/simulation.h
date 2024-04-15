#pragma once

#include <deal.II/base/function.h>

#include <deal.II/grid/tria.h>

#include <deal.II/numerics/vector_tools.h>

#include <fstream>

#include "config.h"

using namespace dealii;

/**
 * Base class for simulations.
 */
template <int dim>
class SimulationBase
{
public:
  struct BoundaryDescriptor
  {
    std::vector<unsigned int> all_homogeneous_dbcs;
    std::vector<unsigned int> all_homogeneous_nbcs;
    std::vector<std::pair<unsigned int, std::shared_ptr<Function<dim, Number>>>>
      all_inhomogeneous_dbcs;

    std::vector<unsigned int> all_slip_bcs;
  };

  virtual void
  create_triangulation(Triangulation<dim> &tria,
                       const unsigned int  n_global_refinements) const = 0;

  virtual BoundaryDescriptor
  get_boundary_descriptor() const = 0;

  virtual void
  postprocess(const double           t,
              const Mapping<dim>    &mapping,
              const DoFHandler<dim> &dof_handler,
              const VectorType      &vector) const;

  virtual void
  parse_parameters(const std::string &file_name);
};



/**
 * Channel simulation.
 */
template <int dim>
class SimulationChannel : public SimulationBase<dim>
{
public:
  using BoundaryDescriptor = typename SimulationBase<dim>::BoundaryDescriptor;

  SimulationChannel();

  void
  create_triangulation(Triangulation<dim> &tria,
                       const unsigned int  n_global_refinements) const override;

  virtual BoundaryDescriptor
  get_boundary_descriptor() const override;

private:
  const unsigned int n_stretching;

  class InflowVelocity : public Function<dim, Number>
  {
  public:
    InflowVelocity()
      : Function<dim>(dim + 1)
    {}

    Number
    value(const Point<dim> &p, const unsigned int component) const override
    {
      (void)p;

      if (component == 0)
        return 1.0;
      else
        return 0.0;
    }

  private:
  };
};



/**
 * Flow-past cylinder simulation with alternative mesh.
 */
template <int dim>
class SimulationCylinder : public SimulationBase<dim>
{
public:
  using BoundaryDescriptor = typename SimulationBase<dim>::BoundaryDescriptor;

  SimulationCylinder();

  ~SimulationCylinder();

  void
  create_triangulation(Triangulation<dim> &tria,
                       const unsigned int  n_global_refinements) const override;

  virtual BoundaryDescriptor
  get_boundary_descriptor() const override;

  void
  postprocess(const double           t,
              const Mapping<dim>    &mapping,
              const DoFHandler<dim> &dof_handler,
              const VectorType      &solution) const override;

  void
  parse_parameters(const std::string &file_name) override;

private:
  bool   use_no_slip_cylinder_bc;
  double nu;
  bool   symm;
  bool   rotate;
  double t_init;
  int    reset_manifold_level;

  std::shared_ptr<const Utilities::MPI::RemotePointEvaluation<dim>> rpe;

  mutable std::ofstream drag_lift_pressure_file;

  class InflowBoundaryValues : public Function<dim>
  {
  public:
    InflowBoundaryValues(const double t_init)
      : Function<dim>(dim + 1)
      , t_init(t_init)
    {}

    double
    value(const Point<dim> &p, const unsigned int component) const override
    {
      (void)p;

      const double u_val =
        1.0 * ((t_init == 0) ? 1.0 : std::min(this->get_time() / t_init, 1.0));

      const double v_val = 0.0;
      const double p_val = 0.0;

      if (component == 0)
        return u_val;
      else if (component == 1)
        return v_val;
      else if (component == 2)
        return p_val;

      return 0;
    }

  private:
    const double t_init;
  };
};



/**
 * Flow-past cylinder simulation with alternative mesh.
 */
template <int dim>
class SimulationRotation : public SimulationBase<dim>
{
public:
  using BoundaryDescriptor = typename SimulationBase<dim>::BoundaryDescriptor;

  SimulationRotation();

  void
  create_triangulation(Triangulation<dim> &tria,
                       const unsigned int  n_global_refinements) const override;

  virtual BoundaryDescriptor
  get_boundary_descriptor() const override;

  void
  postprocess(const double           t,
              const Mapping<dim>    &mapping,
              const DoFHandler<dim> &dof_handler,
              const VectorType      &solution) const override;

private:
  const bool use_no_slip_cylinder_bc;

  class InflowBoundaryValues : public Function<dim>
  {
  public:
    InflowBoundaryValues()
      : Function<dim>(dim + 1)
      , t_(0.0){};

    double
    value(const Point<dim> &p, const unsigned int component) const override
    {
      (void)p;

      if (component == 0)
        return -p[1];
      else if (component == 1)
        return p[0];
      else if (component == 2)
        return 0;

      return 0;
    }

  private:
    const double t_;
  };
};
