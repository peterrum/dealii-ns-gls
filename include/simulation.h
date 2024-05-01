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

  virtual ~SimulationBase() = default;

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

  virtual double
  get_u_max() const;
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

  double
  get_u_max() const override;

private:
  bool        use_no_slip_cylinder_bc;
  bool        use_no_slip_wall_bc;
  double      nu;
  bool        rotate;
  double      t_init;
  int         reset_manifold_level;
  double      u_max;
  std::string paraview_prefix;
  double      output_granularity;

  double length;
  double height;
  double cylinder_position;
  double diameter;
  double shift;

  mutable std::shared_ptr<const Utilities::MPI::RemotePointEvaluation<dim>> rpe;

  mutable std::ofstream drag_lift_pressure_file;
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
};
