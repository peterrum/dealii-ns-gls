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

    std::set<unsigned int> all_outflow_bcs_cut;
    std::map<unsigned int, std::shared_ptr<Function<dim, Number>>>
      all_outflow_bcs_nitsche;

    std::vector<unsigned int> all_slip_bcs;

    std::vector<std::tuple<unsigned int, unsigned int, unsigned int>>
      periodic_bcs;
  };

  virtual ~SimulationBase() = default;

  virtual void
  create_triangulation(Triangulation<dim> &tria,
                       const unsigned int  n_global_refinements) const = 0;

  virtual BoundaryDescriptor
  get_boundary_descriptor() const = 0;

  virtual void
  postprocess(const double              t,
              const Mapping<dim>       &mapping,
              const DoFHandler<dim>    &dof_handler,
              const VectorType<Number> &vector) const;

  virtual void
  parse_parameters(const std::string &file_name);

  virtual double
  get_u_max() const;

  virtual std::shared_ptr<Mapping<dim>>
  get_mapping(const Triangulation<dim> &tria,
              const unsigned int        mapping_degree) const;
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
  postprocess(const double              t,
              const Mapping<dim>       &mapping,
              const DoFHandler<dim>    &dof_handler,
              const VectorType<Number> &solution) const override;

  void
  parse_parameters(const std::string &file_name) override;

  double
  get_u_max() const override;

  virtual std::shared_ptr<Mapping<dim>>
  get_mapping(const Triangulation<dim> &tria,
              const unsigned int        mapping_degree) const override;

private:
  bool        use_no_slip_cylinder_bc;
  bool        use_no_slip_wall_bc;
  double      nu;
  bool        rotate;
  double      distortion;
  double      t_init;
  int         reset_manifold_level;
  double      u_max;
  std::string paraview_prefix;
  double      output_granularity;

  double geometry_channel_length;
  double geometry_channel_extra_length;
  double geometry_channel_height;
  double geometry_cylinder_position;
  double geometry_cylinder_diameter;
  double geometry_cylinder_shift;

  unsigned int fe_degree;
  unsigned int mapping_degree;

  bool use_exact_normal;
  bool use_symmetric_walls;
  bool use_outflow_bc_weak_cut;
  bool use_outflow_bc_weak_nitsche;
  bool use_outflow_bc_strong;

  mutable std::shared_ptr<const Utilities::MPI::RemotePointEvaluation<dim>> rpe;

  mutable std::ofstream drag_lift_pressure_file;

  template <int structdim>
  std::shared_ptr<Mapping<structdim, dim>>
  get_mapping_private(const Triangulation<structdim, dim> &tria,
                      const unsigned int mapping_degree) const;
};



/**
 * Taylor-Couette flow.
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
  postprocess(const double              t,
              const Mapping<dim>       &mapping,
              const DoFHandler<dim>    &dof_handler,
              const VectorType<Number> &solution) const override;

private:
};



/**
 * Flow-past sphere.
 */
template <int dim>
class SimulationSphere : public SimulationBase<dim>
{
public:
  using BoundaryDescriptor = typename SimulationBase<dim>::BoundaryDescriptor;

  SimulationSphere();

  void
  create_triangulation(Triangulation<dim> &tria,
                       const unsigned int  n_global_refinements) const override;

  virtual BoundaryDescriptor
  get_boundary_descriptor() const override;

  void
  postprocess(const double              t,
              const Mapping<dim>       &mapping,
              const DoFHandler<dim>    &dof_handler,
              const VectorType<Number> &solution) const override;

private:
};
