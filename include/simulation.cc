
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


namespace dealii
{
  namespace GridGenerator
  {
    namespace internal
    {
      template <int dim, int spacedim>
      double
      minimal_vertex_distance(const Triangulation<dim, spacedim> &triangulation)
      {
        double length = std::numeric_limits<double>::max();
        for (const auto &cell : triangulation.active_cell_iterators())
          for (unsigned int n = 0; n < GeometryInfo<dim>::lines_per_cell; ++n)
            length = std::min(length, cell->line(n)->diameter());
        return length;
      }
    } // namespace internal

    void
    my_channel_with_cylinder(Triangulation<2>  &tria,
                             const double       shell_region_width,
                             const unsigned int n_shells,
                             const double       skewness,
                             const bool         colorize)
    {
      Assert(0.0 <= shell_region_width && shell_region_width < 0.05,
             ExcMessage("The width of the shell region must be less than 0.05 "
                        "(and preferably close to 0.03)"));
      const types::manifold_id polar_manifold_id = 0;
      const types::manifold_id tfi_manifold_id   = 1;

      // We begin by setting up a grid that is 4 by 22 cells. While not
      // squares, these have pretty good aspect ratios.
      Triangulation<2> bulk_tria;
      GridGenerator::subdivided_hyper_rectangle(bulk_tria,
                                                {22u, 4u},
                                                Point<2>(0.0, 0.0),
                                                Point<2>(2.2, 0.41));
      // bulk_tria now looks like this:
      //
      //   +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
      //   |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
      //   +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
      //   |  |XX|XX|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
      //   +--+--O--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
      //   |  |XX|XX|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
      //   +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
      //   |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
      //   +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
      //
      // Note that these cells are not quite squares: they are all 0.1 by
      // 0.1025.
      //
      // The next step is to remove the cells marked with XXs: we will place
      // the grid around the cylinder there later. The next loop does two
      // things:
      // 1. Determines which cells need to be removed from the Triangulation
      //    (i.e., find the cells marked with XX in the picture).
      // 2. Finds the location of the vertex marked with 'O' and uses that to
      //    calculate the shift vector for aligning cylinder_tria with
      //    tria_without_cylinder.
      std::set<Triangulation<2>::active_cell_iterator> cells_to_remove;
      Tensor<1, 2> cylinder_triangulation_offset;
      for (const auto &cell : bulk_tria.active_cell_iterators())
        {
          if ((cell->center() - Point<2>(0.2, 0.2)).norm() < 0.15)
            cells_to_remove.insert(cell);

          if (cylinder_triangulation_offset == Tensor<1, 2>())
            {
              for (const unsigned int vertex_n :
                   GeometryInfo<2>::vertex_indices())
                if (cell->vertex(vertex_n) == Point<2>())
                  {
                    // cylinder_tria is centered at zero, so we need to
                    // shift it up and to the right by two cells:
                    cylinder_triangulation_offset =
                      2.0 * (cell->vertex(3) - Point<2>());
                    break;
                  }
            }
        }
      Triangulation<2> tria_without_cylinder;
      GridGenerator::create_triangulation_with_removed_cells(
        bulk_tria, cells_to_remove, tria_without_cylinder);

      // set up the cylinder triangulation. Note that this function sets the
      // manifold ids of the interior boundary cells to 0
      // (polar_manifold_id).
      Triangulation<2> cylinder_tria;
      GridGenerator::hyper_cube_with_cylindrical_hole(cylinder_tria,
                                                      0.05 + shell_region_width,
                                                      0.41 / 4.0);
      // The bulk cells are not quite squares, so we need to move the left
      // and right sides of cylinder_tria inwards so that it fits in
      // bulk_tria:
      for (const auto &cell : cylinder_tria.active_cell_iterators())
        for (const unsigned int vertex_n : GeometryInfo<2>::vertex_indices())
          {
            if (std::abs(cell->vertex(vertex_n)[0] - -0.41 / 4.0) < 1e-10)
              cell->vertex(vertex_n)[0] = -0.1;
            else if (std::abs(cell->vertex(vertex_n)[0] - 0.41 / 4.0) < 1e-10)
              cell->vertex(vertex_n)[0] = 0.1;
          }

      // Assign interior manifold ids to be the TFI id.
      for (const auto &cell : cylinder_tria.active_cell_iterators())
        {
          cell->set_manifold_id(tfi_manifold_id);
          for (const unsigned int face_n : GeometryInfo<2>::face_indices())
            if (!cell->face(face_n)->at_boundary())
              cell->face(face_n)->set_manifold_id(tfi_manifold_id);
        }
      if (0.0 < shell_region_width)
        {
          Assert(0 < n_shells,
                 ExcMessage("If the shell region has positive width then "
                            "there must be at least one shell."));
          Triangulation<2> shell_tria;
          GridGenerator::concentric_hyper_shells(shell_tria,
                                                 Point<2>(),
                                                 0.05,
                                                 0.05 + shell_region_width,
                                                 n_shells,
                                                 skewness,
                                                 8);

          // Make the tolerance as large as possible since these cells can
          // be quite close together
          const double vertex_tolerance =
            std::min(internal::minimal_vertex_distance(shell_tria),
                     internal::minimal_vertex_distance(cylinder_tria)) *
            0.5;

          shell_tria.set_all_manifold_ids(polar_manifold_id);
          Triangulation<2> temp;
          GridGenerator::merge_triangulations(
            shell_tria, cylinder_tria, temp, vertex_tolerance, true);
          cylinder_tria = std::move(temp);
        }
      GridTools::shift(cylinder_triangulation_offset, cylinder_tria);

      // Compute the tolerance again, since the shells may be very close to
      // each-other:
      const double vertex_tolerance =
        std::min(internal::minimal_vertex_distance(tria_without_cylinder),
                 internal::minimal_vertex_distance(cylinder_tria)) /
        10;
      GridGenerator::merge_triangulations(
        tria_without_cylinder, cylinder_tria, tria, vertex_tolerance, true);

      // Move the vertices in the middle of the faces of cylinder_tria slightly
      // to give a better mesh quality. We have to balance the quality of these
      // cells with the quality of the outer cells (initially rectangles). For
      // constant radial distance, we would place them at the distance 0.1 *
      // sqrt(2.) from the center. In case the shell region width is more than
      // 0.1/6., we choose to place them at 0.1 * 4./3. from the center, which
      // ensures that the shortest edge of the outer cells is 2./3. of the
      // original length. If the shell region width is less, we make the edge
      // length of the inner part and outer part (in the shorter x direction)
      // the same.
      {
        const double shift =
          std::min(0.125 + shell_region_width * 0.5, 0.1 * 4. / 3.);
        for (const auto &cell : tria.active_cell_iterators())
          for (const unsigned int v : GeometryInfo<2>::vertex_indices())
            if (cell->vertex(v).distance(Point<2>(0.1, 0.205)) < 1e-10)
              cell->vertex(v) = Point<2>(0.2 - shift, 0.205);
            else if (cell->vertex(v).distance(Point<2>(0.3, 0.205)) < 1e-10)
              cell->vertex(v) = Point<2>(0.2 + shift, 0.205);
            else if (cell->vertex(v).distance(Point<2>(0.2, 0.1025)) < 1e-10)
              cell->vertex(v) = Point<2>(0.2, 0.2 - shift);
            else if (cell->vertex(v).distance(Point<2>(0.2, 0.3075)) < 1e-10)
              cell->vertex(v) = Point<2>(0.2, 0.2 + shift);
      }

      // Ensure that all manifold ids on a polar cell really are set to the
      // polar manifold id:
      for (const auto &cell : tria.active_cell_iterators())
        if (cell->manifold_id() == polar_manifold_id)
          cell->set_all_manifold_ids(polar_manifold_id);

      // Ensure that all other manifold ids (including the interior faces
      // opposite the cylinder) are set to the flat manifold id:
      for (const auto &cell : tria.active_cell_iterators())
        if (cell->manifold_id() != polar_manifold_id &&
            cell->manifold_id() != tfi_manifold_id)
          cell->set_all_manifold_ids(numbers::flat_manifold_id);

      // We need to calculate the current center so that we can move it later:
      // to start get a unique list of (points to) vertices on the cylinder
      std::vector<Point<2> *> cylinder_pointers;
      for (const auto &face : tria.active_face_iterators())
        if (face->manifold_id() == polar_manifold_id)
          {
            cylinder_pointers.push_back(&face->vertex(0));
            cylinder_pointers.push_back(&face->vertex(1));
          }
      // de-duplicate
      std::sort(cylinder_pointers.begin(), cylinder_pointers.end());
      cylinder_pointers.erase(std::unique(cylinder_pointers.begin(),
                                          cylinder_pointers.end()),
                              cylinder_pointers.end());

      // find the current center...
      Point<2> center;
      for (const Point<2> *const ptr : cylinder_pointers)
        center += *ptr / double(cylinder_pointers.size());

      // and recenter at (0.2, 0.2)
      for (Point<2> *const ptr : cylinder_pointers)
        *ptr += Point<2>(0.2, 0.2) - center;

      // attach manifolds
      PolarManifold<2> polar_manifold(Point<2>(0.2, 0.2));
      tria.set_manifold(polar_manifold_id, polar_manifold);

      tria.set_manifold(tfi_manifold_id, FlatManifold<2>());
      TransfiniteInterpolationManifold<2> inner_manifold;
      inner_manifold.initialize(tria);
      tria.set_manifold(tfi_manifold_id, inner_manifold);

      if (colorize)
        for (const auto &face : tria.active_face_iterators())
          if (face->at_boundary())
            {
              const Point<2> center = face->center();
              // left side
              if (std::abs(center[0] - 0.0) < 1e-10)
                face->set_boundary_id(0);
              // right side
              else if (std::abs(center[0] - 2.2) < 1e-10)
                face->set_boundary_id(1);
              // cylinder boundary
              else if (face->manifold_id() == polar_manifold_id)
                face->set_boundary_id(2);
              // sides of channel
              else
                {
                  Assert(std::abs(center[1] - 0.00) < 1.0e-10 ||
                           std::abs(center[1] - 0.41) < 1.0e-10,
                         ExcInternalError());
                  face->set_boundary_id(3);
                }
            }
    }

    void
    my_channel_with_cylinder(Triangulation<3>  &tria,
                             const double       shell_region_width,
                             const unsigned int n_shells,
                             const double       skewness,
                             const bool         colorize)
    {
      Triangulation<2> tria_2;
      my_channel_with_cylinder(
        tria_2, shell_region_width, n_shells, skewness, colorize);
      extrude_triangulation(tria_2, 5, 0.41, tria, true);

      // set up the new 3d manifolds
      const types::manifold_id      cylindrical_manifold_id = 0;
      const types::manifold_id      tfi_manifold_id         = 1;
      const PolarManifold<2> *const m_ptr =
        dynamic_cast<const PolarManifold<2> *>(
          &tria_2.get_manifold(cylindrical_manifold_id));
      Assert(m_ptr != nullptr, ExcInternalError());
      const Point<3>     axial_point(m_ptr->center[0], m_ptr->center[1], 0.0);
      const Tensor<1, 3> direction{{0.0, 0.0, 1.0}};

      tria.set_manifold(cylindrical_manifold_id, FlatManifold<3>());
      tria.set_manifold(tfi_manifold_id, FlatManifold<3>());
      const CylindricalManifold<3> cylindrical_manifold(direction, axial_point);
      TransfiniteInterpolationManifold<3> inner_manifold;
      inner_manifold.initialize(tria);
      tria.set_manifold(cylindrical_manifold_id, cylindrical_manifold);
      tria.set_manifold(tfi_manifold_id, inner_manifold);

      // From extrude_triangulation: since the maximum boundary id of tria_2 was
      // 3, the bottom boundary id is 4 and the top is 5: both are walls, so set
      // them to 3
      if (colorize)
        for (const auto &face : tria.active_face_iterators())
          if (face->boundary_id() == 4 || face->boundary_id() == 5)
            face->set_boundary_id(3);
    }
  } // namespace GridGenerator
} // namespace dealii



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
  GridGenerator::my_channel_with_cylinder(tria, 0.03, 2, 2.0, true);

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
