#include "simulation.h"

#include <deal.II/base/parameter_handler.h>

#include <deal.II/grid/grid_in.h>

#include <deal.II/physics/transformations.h>

#include "grid_cylinder.h"
#include "grid_cylinder_old.h"

using namespace dealii;

namespace InflowBoundaryValues
{
  template <int dim>
  class Channel : public Function<dim>
  {
  public:
    Channel(const double t_init,
            const double u_max,
            const bool   no_slip_bc = false,
            const double H          = 0.0,
            const double shift      = 0.0)
      : Function<dim>(dim + 1)
      , t_init(t_init)
      , u_max(u_max)
      , no_slip_bc(no_slip_bc)
      , H(H)
      , shift(shift)
    {}

    double
    value(const Point<dim> &p, const unsigned int component) const override
    {
      AssertDimension(dim, 2);

      const double alpha =
        (t_init == 0) ? 1.0 : std::min(this->get_time() / t_init, 1.0);

      const double y = p[1] - shift;

      const double u_val =
        u_max * (no_slip_bc ? (4 * y * (H - y) / H / H) : 1.0);

      if (component == 0)
        return u_val * alpha;
      else
        return 0;
    }

  private:
    const double t_init;
    const double u_max;
    const bool   no_slip_bc;
    const double H;
    const double shift;
  };

  template <int dim>
  class Rotation : public Function<dim>
  {
  public:
    Rotation()
      : Function<dim>(dim + 1){};

    double
    value(const Point<dim> &p, const unsigned int component) const override
    {
      if (component == 0)
        return -p[1];
      else if (component == 1)
        return p[0];
      else
        return 0;
    }

  private:
  };
} // namespace InflowBoundaryValues

template <int dim>
void
SimulationBase<dim>::postprocess(const double           t,
                                 const Mapping<dim>    &mapping,
                                 const DoFHandler<dim> &dof_handler,
                                 const VectorType      &vector) const
{
  // to be implemented in derived classes
  (void)t;
  (void)mapping;
  (void)dof_handler;
  (void)vector;
}



template <int dim>
void
SimulationBase<dim>::parse_parameters(const std::string &file_name)
{
  // to be implemented in derived classes
  (void)file_name;
}



template <int dim>
double
SimulationBase<dim>::get_u_max() const
{
  return 1.0;
}



template <int dim>
SimulationChannel<dim>::SimulationChannel()
  : n_stretching(4)
{}

template <int dim>
void
SimulationChannel<dim>::create_triangulation(
  Triangulation<dim> &tria,
  const unsigned int  n_global_refinements) const
{
  std::vector<unsigned int> n_subdivisions(dim, 1);
  n_subdivisions[0] *= n_stretching;

  Point<dim> p0;
  Point<dim> p1;

  for (unsigned int d = 0; d < dim; ++d)
    p1[d] = 1.0;
  p1[0] *= n_stretching;

  GridGenerator::subdivided_hyper_rectangle(tria, n_subdivisions, p0, p1, true);

  tria.refine_global(2);

  tria.refine_global(n_global_refinements);
}

template <int dim>
SimulationChannel<dim>::BoundaryDescriptor
SimulationChannel<dim>::get_boundary_descriptor() const
{
  BoundaryDescriptor bcs;

  bcs.all_inhomogeneous_dbcs.emplace_back(
    0, std::make_shared<InflowBoundaryValues::Channel<dim>>(0.0, 1.0));

  bcs.all_homogeneous_nbcs.push_back(1);

  for (unsigned d = 1; d < dim; ++d)
    {
      bcs.all_homogeneous_dbcs.push_back(2 * d);
      bcs.all_homogeneous_dbcs.push_back(2 * d + 1);
    }

  return bcs;
}



/**
 * Flow-past cylinder simulation with alternative mesh.
 */
template <int dim>
SimulationCylinder<dim>::SimulationCylinder()
  : use_no_slip_cylinder_bc(true)
  , use_no_slip_wall_bc(true)
  , nu(0.0)
  , symm(true)
  , rotate(false)
  , t_init(0.0)
  , reset_manifold_level(-1)
  , u_max(1.0)
{
  drag_lift_pressure_file.open("drag_lift_pressure.m", std::ios::out);
}

template <int dim>
SimulationCylinder<dim>::~SimulationCylinder()
{
  drag_lift_pressure_file.close();
}

template <int dim>
void
SimulationCylinder<dim>::parse_parameters(const std::string &file_name)
{
  if (file_name == "")
    return;

  dealii::ParameterHandler prm;

  prm.add_parameter("nu", nu);
  prm.add_parameter("simulation no slip cylinder", use_no_slip_cylinder_bc);
  prm.add_parameter("simulation no slip wall", use_no_slip_wall_bc);
  prm.add_parameter("simulation symmetric", symm);
  prm.add_parameter("simulation rotate", rotate);
  prm.add_parameter("simulation t init", t_init);
  prm.add_parameter("simulation reset manifold level", reset_manifold_level);
  prm.add_parameter("simulation u max", u_max);

  std::string paraview_prefix;
  prm.add_parameter("paraview prefix", paraview_prefix);

  prm.parse_input(file_name, "", true);

  if (paraview_prefix != "")
    {
      drag_lift_pressure_file.close();
      drag_lift_pressure_file.open(paraview_prefix + "_drag_lift_pressure.m",
                                   std::ios::out);
    }
}



template <int dim>
double
SimulationCylinder<dim>::get_u_max() const
{
  return u_max;
}

template <int dim>
void
SimulationCylinder<dim>::create_triangulation(
  Triangulation<dim> &tria,
  const unsigned int  n_global_refinements) const
{
  const double diameter = 0.1;

  cylinder(tria, 2.2, 0.41, 0.2, diameter, symm);

  if (reset_manifold_level == 0)
    {
      tria.reset_all_manifolds();
      tria.refine_global(n_global_refinements);
    }
  else if (static_cast<unsigned int>(reset_manifold_level) >
           n_global_refinements)
    {
      tria.refine_global(n_global_refinements);
    }
  else
    {
      tria.refine_global(reset_manifold_level);
      tria.reset_all_manifolds();
      tria.refine_global(n_global_refinements - reset_manifold_level);
    }

  if (rotate)
    {
      const double angle = 0.2;
      const double factor_i =
        (reset_manifold_level == -1) ?
          1.0 :
          std::cos(numbers::PI / 8.0 / (1 + reset_manifold_level));

      const auto matrix =
        Physics::Transformations::Rotations::rotation_matrix_2d(angle);

      const auto bb =
        BoundingBox<2>(Point<2>()).create_extended(diameter - 1e-6);

      std::vector<bool> vertex_state(tria.n_vertices(), false);

      for (auto cell : tria.cell_iterators())
        for (const auto v : cell->vertex_indices())
          if (vertex_state[cell->vertex_index(v)] == false)
            {
              auto &vertex = cell->vertex(v);

              Point<2> vertex_2D(vertex[0], vertex[1]);

              if (bb.point_inside(vertex_2D))
                {
                  double factor = diameter / std::max(std::abs(vertex_2D[0]),
                                                      std::abs(vertex_2D[1]));

                  factor =
                    (vertex_2D.norm() - (factor_i * diameter / 2)) /
                    (vertex_2D.norm() * factor - (factor_i * diameter / 2));

                  vertex_2D = (matrix * vertex_2D) * (1.0 - factor) +
                              vertex_2D * (factor);

                  vertex[0] = vertex_2D[0];
                  vertex[1] = vertex_2D[1];
                }

              vertex_state[cell->vertex_index(v)] = true;
            }
    }
}

template <int dim>
SimulationCylinder<dim>::BoundaryDescriptor
SimulationCylinder<dim>::get_boundary_descriptor() const
{
  BoundaryDescriptor bcs;

  // inflow
  bcs.all_inhomogeneous_dbcs.emplace_back(
    0,
    std::make_shared<InflowBoundaryValues::Channel<dim>>(
      t_init, u_max, use_no_slip_wall_bc, 0.41, symm ? -0.205 : -0.200));

  // outflow
  bcs.all_homogeneous_nbcs.push_back(1);

  // walls
  if (use_no_slip_wall_bc)
    bcs.all_homogeneous_dbcs.push_back(2);
  else
    bcs.all_slip_bcs.push_back(2);

  // cylinder
  if (use_no_slip_cylinder_bc)
    bcs.all_homogeneous_dbcs.push_back(3);
  else
    bcs.all_slip_bcs.push_back(3);

  return bcs;
}

template <int dim>
void
SimulationCylinder<dim>::postprocess(const double           t,
                                     const Mapping<dim>    &mapping,
                                     const DoFHandler<dim> &dof_handler,
                                     const VectorType      &solution) const
{
  const bool has_ghost_elements = solution.has_ghost_elements();

  AssertThrow(has_ghost_elements == false, ExcInternalError());

  if (has_ghost_elements == false)
    solution.update_ghost_values();

  double drag = 0, lift = 0, p_diff = 0;

  const MPI_Comm comm = dof_handler.get_communicator();

  QGauss<dim - 1> face_quadrature_formula(3);
  const int       n_q_points = face_quadrature_formula.size();

  FEFaceValues<dim> fe_face_values(dof_handler.get_fe(),
                                   face_quadrature_formula,
                                   update_values | update_quadrature_points |
                                     update_gradients | update_JxW_values |
                                     update_normal_vectors);

  FEValuesViews::Vector<dim> velocities(fe_face_values, 0);
  FEValuesViews::Scalar<dim> pressure(fe_face_values, dim);

  std::vector<dealii::SymmetricTensor<2, dim>> eps_u(n_q_points);
  std::vector<double>                          p(n_q_points);

  Tensor<2, dim> fluid_stress;
  Tensor<1, dim> forces;

  double drag_local = 0;
  double lift_local = 0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      for (const auto face : cell->face_indices())
        if (cell->face(face)->at_boundary() &&
            (cell->face(face)->boundary_id() == 3))
          {
            fe_face_values.reinit(cell, face);
            std::vector<Point<dim>> q_points =
              fe_face_values.get_quadrature_points();

            velocities.get_function_symmetric_gradients(solution, eps_u);
            pressure.get_function_values(solution, p);

            for (int q = 0; q < n_q_points; ++q)
              {
                const Tensor<1, dim> normal_vector =
                  -fe_face_values.normal_vector(q);

                Tensor<2, dim> fluid_pressure;
                fluid_pressure[0][0] = p[q];
                fluid_pressure[1][1] = p[q];

                const Tensor<2, dim> fluid_stress =
                  nu * eps_u[q] - fluid_pressure;

                const Tensor<1, dim> forces =
                  fluid_stress * normal_vector * fe_face_values.JxW(q);

                drag_local += forces[0];
                lift_local += forces[1];
              }
          }
    }

  drag = Utilities::MPI::sum(drag_local, comm);
  lift = Utilities::MPI::sum(lift_local, comm);

  // calculate pressure drop

  // 1) set up evaluation routine (TODO: can be reused!)
  std::shared_ptr<Utilities::MPI::RemotePointEvaluation<dim>> rpe;

  if (rpe == nullptr)
    {
      Point<dim> p1, p2;
      p1[0] = ExaDG::FlowPastCylinder::X_C - ExaDG::FlowPastCylinder::D * 0.5;
      p2[0] = ExaDG::FlowPastCylinder::X_C + ExaDG::FlowPastCylinder::D * 0.5;
      p1[1] = p2[1] = ExaDG::FlowPastCylinder::H / 2.0;

      std::vector<Point<dim>> points;

      if (Utilities::MPI::this_mpi_process(comm) == 0)
        {
          points.push_back(p1);
          points.push_back(p2);
        }

      auto rpe_temp =
        std::make_shared<Utilities::MPI::RemotePointEvaluation<dim>>();
      rpe_temp->reinit(points, dof_handler.get_triangulation(), mapping);

      rpe = rpe_temp;
    }

  const auto values = VectorTools::point_values<1>(
    *rpe, dof_handler, solution, VectorTools::EvaluationFlags::avg, dim);

  if (Utilities::MPI::this_mpi_process(comm) == 0)
    p_diff = values[0] - values[1];

  // write to file
  if (Utilities::MPI::this_mpi_process(comm) == 0)
    {
      drag_lift_pressure_file << t << "\t" << drag << "\t" << lift << "\t"
                              << p_diff << "\n";
      drag_lift_pressure_file.flush();
    }

  // clean up
  if (has_ghost_elements == false)
    solution.zero_out_ghost_values();
}



template <int dim>
SimulationRotation<dim>::SimulationRotation()
{}

template <int dim>
void
SimulationRotation<dim>::create_triangulation(
  Triangulation<dim> &tria,
  const unsigned int  n_global_refinements) const
{
  GridGenerator::hyper_shell(tria, Point<dim>(), 0.25, 1, 4, true);

  tria.refine_global(n_global_refinements);

  if (true)
    {
      for (const auto &cell : tria.active_cell_iterators())
        if (cell->at_boundary())
          cell->set_refine_flag();
      tria.execute_coarsening_and_refinement();
    }
  else if (false)
    {
      for (const auto &cell : tria.active_cell_iterators())
        for (const auto &face : cell->face_iterators())
          if (face->at_boundary() && (face->boundary_id() == 0))
            cell->set_refine_flag();
      tria.execute_coarsening_and_refinement();
    }
}

template <int dim>
SimulationRotation<dim>::BoundaryDescriptor
SimulationRotation<dim>::get_boundary_descriptor() const
{
  BoundaryDescriptor bcs;

  // inflow
  bcs.all_inhomogeneous_dbcs.emplace_back(
    0, std::make_shared<InflowBoundaryValues::Rotation<dim>>());

  // walls
  bcs.all_homogeneous_dbcs.push_back(1);

  return bcs;
}

template <int dim>
void
SimulationRotation<dim>::postprocess(const double           t,
                                     const Mapping<dim>    &mapping,
                                     const DoFHandler<dim> &dof_handler,
                                     const VectorType      &solution) const
{
  // nothing to do
  (void)t;
  (void)mapping;
  (void)dof_handler;
  (void)solution;
}


template class SimulationBase<2>;
template class SimulationBase<3>;
template class SimulationChannel<2>;
template class SimulationChannel<3>;
template class SimulationCylinder<2>;
template class SimulationCylinder<3>;
template class SimulationRotation<2>;
template class SimulationRotation<3>;