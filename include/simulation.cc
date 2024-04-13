
#include "simulation.h"

#include <deal.II/grid/grid_in.h>

#include "grid_cylinder.h"
#include "grid_cylinder_old.h"

using namespace dealii;

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

  bcs.all_inhomogeneous_dbcs.emplace_back(0,
                                          std::make_shared<InflowVelocity>());

  bcs.all_homogeneous_nbcs.push_back(1);

  for (unsigned d = 1; d < dim; ++d)
    {
      bcs.all_homogeneous_dbcs.push_back(2 * d);
      bcs.all_homogeneous_dbcs.push_back(2 * d + 1);
    }

  return bcs;
}



/**
 * Flow-past cylinder simulation.
 */
template <int dim>
SimulationCylinderExadg<dim>::SimulationCylinderExadg(
  const double nu,
  const bool   use_no_slip_cylinder_bc)
  : use_no_slip_cylinder_bc(use_no_slip_cylinder_bc)
  , nu(nu)
{
  drag_lift_pressure_file.open("drag_lift_pressure.m", std::ios::out);
}

template <int dim>
SimulationCylinderExadg<dim>::~SimulationCylinderExadg()
{
  drag_lift_pressure_file.close();
}

template <int dim>
void
SimulationCylinderExadg<dim>::create_triangulation(
  Triangulation<dim> &tria,
  const unsigned int  n_global_refinements) const
{
  ExaDG::FlowPastCylinder::create_coarse_grid(tria);

  tria.refine_global(n_global_refinements);

  tria.reset_all_manifolds(); // TODO: problem with ChartManifold
}

template <int dim>
SimulationCylinderExadg<dim>::BoundaryDescriptor
SimulationCylinderExadg<dim>::get_boundary_descriptor() const
{
  BoundaryDescriptor bcs;

  // inflow
  bcs.all_inhomogeneous_dbcs.emplace_back(
    0, std::make_shared<InflowBoundaryValues>());

  // outflow
  bcs.all_homogeneous_nbcs.push_back(1);

  // walls
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
SimulationCylinderExadg<dim>::postprocess(const double           t,
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



/**
 * Flow-past cylinder simulation with alternative mesh.
 */
template <int dim>
SimulationCylinderOld<dim>::SimulationCylinderOld(
  const double nu,
  const bool   use_no_slip_cylinder_bc)
  : use_no_slip_cylinder_bc(use_no_slip_cylinder_bc)
  , nu(nu)
{
  drag_lift_pressure_file.open("drag_lift_pressure.m", std::ios::out);
}

template <int dim>
SimulationCylinderOld<dim>::~SimulationCylinderOld()
{
  drag_lift_pressure_file.close();
}

template <int dim>
void
SimulationCylinderOld<dim>::create_triangulation(
  Triangulation<dim> &tria,
  const unsigned int  n_global_refinements) const
{
  if (false /* TODO */)
    cylinder(tria,
             ExaDG::FlowPastCylinder::L2 - ((dim == 2) ?
                                              ExaDG::FlowPastCylinder::L1 :
                                              ExaDG::FlowPastCylinder::X_0),
             ExaDG::FlowPastCylinder::H,
             ExaDG::FlowPastCylinder::X_C,
             ExaDG::FlowPastCylinder::D);
  else
    cylinder(tria, 4.0, 2.0, 0.6, 0.5);

  tria.refine_global(n_global_refinements);
}

template <int dim>
SimulationCylinderOld<dim>::BoundaryDescriptor
SimulationCylinderOld<dim>::get_boundary_descriptor() const
{
  BoundaryDescriptor bcs;

  // inflow
  bcs.all_inhomogeneous_dbcs.emplace_back(
    0, std::make_shared<InflowBoundaryValues>());

  // outflow
  bcs.all_homogeneous_nbcs.push_back(1);

  // walls
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
SimulationCylinderOld<dim>::postprocess(const double           t,
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



/**
 * Flow-past cylinder simulation with alternative mesh.
 */
template <int dim>
SimulationCylinderDealii<dim>::SimulationCylinderDealii(
  const bool use_no_slip_cylinder_bc)
  : use_no_slip_cylinder_bc(use_no_slip_cylinder_bc)
{}

template <int dim>
void
SimulationCylinderDealii<dim>::create_triangulation(
  Triangulation<dim> &tria,
  const unsigned int  n_global_refinements) const
{
  GridGenerator::channel_with_cylinder(tria, 0.03, 2, 2.0, true);

  tria.refine_global(n_global_refinements);

  if (false)
    {
      const auto bb =
        BoundingBox<dim>(Point<dim>(0.2, 0.2)).create_extended(0.12);

      for (const auto &cell : tria.active_cell_iterators())
        if (bb.point_inside(cell->center()))
          cell->set_refine_flag();
      tria.execute_coarsening_and_refinement();
    }
}

template <int dim>
SimulationCylinderDealii<dim>::BoundaryDescriptor
SimulationCylinderDealii<dim>::get_boundary_descriptor() const
{
  BoundaryDescriptor bcs;

  // inflow
  bcs.all_inhomogeneous_dbcs.emplace_back(
    0, std::make_shared<InflowBoundaryValues>());

  // walls
  bcs.all_slip_bcs.push_back(3);

  // cylinder
  if (use_no_slip_cylinder_bc)
    bcs.all_homogeneous_dbcs.push_back(2);
  else
    bcs.all_slip_bcs.push_back(2);

  return bcs;
}

template <int dim>
void
SimulationCylinderDealii<dim>::postprocess(const double           t,
                                           const Mapping<dim>    &mapping,
                                           const DoFHandler<dim> &dof_handler,
                                           const VectorType &solution) const
{
  // nothing to do
  (void)t;
  (void)mapping;
  (void)dof_handler;
  (void)solution;
}



template <int dim>
SimulationRotation<dim>::SimulationRotation()
  : use_no_slip_cylinder_bc(true)
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
    0, std::make_shared<InflowBoundaryValues>());

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
template class SimulationCylinderExadg<2>;
template class SimulationCylinderExadg<3>;
template class SimulationCylinderOld<2>;
template class SimulationCylinderOld<3>;
template class SimulationCylinderDealii<2>;
template class SimulationCylinderDealii<3>;
template class SimulationRotation<2>;
template class SimulationRotation<3>;
