#pragma once

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

template <int spacedim>
void
cylinder(Triangulation<2, spacedim> &triangulation,
         const double                length,
         const double                height,
         const double                cylinder_position,
         const double                cylinder_diameter,
         const double                shift,
         const bool                  use_symmetric_walls,
         const bool                  for_3D = false)
{
  constexpr int dim = 2;

  using namespace dealii;

  dealii::Triangulation<dim, spacedim> tria1, tria2, tria3, tria4, tria5, tria6,
    tria7, tria8, tria9, tria_tmp;

  // center
  GridGenerator::hyper_cube_with_cylindrical_hole(
    tria1, cylinder_diameter / 2., cylinder_diameter, 0.05, 1, false);

  GridGenerator::subdivided_hyper_rectangle(
    tria2,
    {2, 1},
    Point<2>(-cylinder_diameter, -cylinder_diameter),
    Point<2>(cylinder_diameter, -height / 2. + shift));

  GridGenerator::subdivided_hyper_rectangle(
    tria3,
    {2, 1},
    Point<2>(-cylinder_diameter, cylinder_diameter),
    Point<2>(cylinder_diameter, height / 2. + shift));

  // right
  GridGenerator::subdivided_hyper_rectangle(
    tria4,
    {18, 2},
    Point<2>(cylinder_diameter, -cylinder_diameter),
    Point<2>(length - cylinder_position, cylinder_diameter));

  GridGenerator::subdivided_hyper_rectangle(
    tria5,
    {18, 1},
    Point<2>(cylinder_diameter, cylinder_diameter),
    Point<2>(length - cylinder_position, height / 2. + shift));

  GridGenerator::subdivided_hyper_rectangle(
    tria6,
    {18, 1},
    Point<2>(cylinder_diameter, -height / 2. + shift),
    Point<2>(length - cylinder_position, -cylinder_diameter));

  // left
  GridGenerator::subdivided_hyper_rectangle(
    tria7,
    {for_3D ? 4u : 1u, 2},
    Point<2>(-cylinder_position, -cylinder_diameter),
    Point<2>(-cylinder_diameter, cylinder_diameter));

  GridGenerator::subdivided_hyper_rectangle(
    tria8,
    {for_3D ? 4u : 1u, 1},
    Point<2>(-cylinder_position, cylinder_diameter),
    Point<2>(-cylinder_diameter, height / 2. + shift));

  GridGenerator::subdivided_hyper_rectangle(
    tria9,
    {for_3D ? 4u : 1u, 1},
    Point<2>(-cylinder_position, -height / 2. + shift),
    Point<2>(-cylinder_diameter, -cylinder_diameter));

  tria_tmp.set_mesh_smoothing(triangulation.get_mesh_smoothing());
  GridGenerator::merge_triangulations(
    {&tria1, &tria2, &tria3, &tria4, &tria5, &tria6, &tria7, &tria8, &tria9},
    tria_tmp,
    1.e-12,
    true);
  triangulation.copy_triangulation(tria_tmp);

  /* Restore polar manifold for disc: */

  triangulation.set_manifold(0,
                             PolarManifold<dim, spacedim>(Point<spacedim>()));

  /* Fix up position of left boundary: */

  // for (auto cell : triangulation.active_cell_iterators())
  //   for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
  //     {
  //       auto &vertex = cell->vertex(v);
  //       if (vertex[0] <= -cylinder_diameter + 1.e-6)
  //         vertex[0] = -cylinder_position;
  //     }

  /*
   * Set boundary ids:
   */

  for (auto cell : triangulation.active_cell_iterators())
    {
      for (auto f : GeometryInfo<dim>::face_indices())
        {
          const auto face = cell->face(f);

          if (!face->at_boundary())
            continue;

          /*
           * We want slip boundary conditions (i.e. indicator 1) at top and
           * bottom of the channel, as well as on the obstacle. On the left
           * side we set inflow conditions (indicator 2) and on the right
           * side we set indicator 0, i.e. do nothing.
           */

          const auto center = face->center();

          if (center[0] > length - cylinder_position - 1.e-6)
            // outflow
            face->set_boundary_id(1);
          else if (center[0] < -cylinder_position + 1.e-6)
            // inflow
            face->set_boundary_id(0);
          else if (std::abs(center[1] - (+height / 2. + shift)) < 1.e-6)
            // wall (top)
            face->set_boundary_id(4);
          else if (std::abs(center[1] - (-height / 2. + shift)) < 1.e-6)
            // wall (bottom)
            face->set_boundary_id(3);
          else
            face->set_boundary_id(2);
        }
    }

  if (use_symmetric_walls)
    {
      std::vector<GridTools::PeriodicFacePair<
        typename Triangulation<2, spacedim>::cell_iterator>>
        periodic_faces;

      GridTools::collect_periodic_faces(triangulation, 3, 4, 1, periodic_faces);

      triangulation.add_periodicity(periodic_faces);
    }
}

void
cylinder(Triangulation<3, 3> &triangulation,
         const double         length,
         const double         height,
         const double         cylinder_position,
         const double         cylinder_diameter,
         const double         shift,
         const bool           use_symmetric_walls)
{
  dealii::Triangulation<2, 2> tria1;

  cylinder(tria1,
           length,
           height,
           cylinder_position,
           cylinder_diameter,
           shift,
           use_symmetric_walls,
           true);

  dealii::Triangulation<3, 3> tria2;
  tria2.set_mesh_smoothing(triangulation.get_mesh_smoothing());

  GridGenerator::extrude_triangulation(tria1, 5, height, tria2, true);
  dealii::GridTools::transform(
    [height](auto point) {
      return point - dealii::Tensor<1, 3>{{0, 0, height / 2.}};
    },
    tria2);

  triangulation.copy_triangulation(tria2);

  /*
   * Reattach an appropriate manifold ID:
   */

  triangulation.set_manifold(0,
                             CylindricalManifold<3>(Tensor<1, 3>{{0., 0., 1.}},
                                                    Point<3>()));

  /*
   * Set boundary ids:
   */

  for (auto cell : triangulation.active_cell_iterators())
    {
      for (const auto f : cell->face_indices())
        {
          const auto face = cell->face(f);

          if (!face->at_boundary())
            continue;

          const auto center = face->center();

          if (center[0] > length - cylinder_position - 1.e-6)
            // outflow
            face->set_boundary_id(1);
          else if (center[0] < -cylinder_position + 1.e-6)
            // inflow
            face->set_boundary_id(0);
          else if (std::abs(center[1] - (+height / 2. + shift)) < 1.e-6)
            // wall (top)
            face->set_boundary_id(4);
          else if (std::abs(center[1] - (-height / 2. + shift)) < 1.e-6)
            // wall (bottom)
            face->set_boundary_id(3);
          else if (std::abs(center[2] - (+height / 2.)) < 1.e-6)
            // wall (top)
            face->set_boundary_id(6);
          else if (std::abs(center[2] - (-height / 2.)) < 1.e-6)
            // wall (bottom)
            face->set_boundary_id(5);
          else
            face->set_boundary_id(2);
        }
    }

  if (use_symmetric_walls)
    {
      std::vector<GridTools::PeriodicFacePair<
        typename Triangulation<3, 3>::cell_iterator>>
        periodic_faces;

      GridTools::collect_periodic_faces(triangulation, 3, 4, 1, periodic_faces);
      GridTools::collect_periodic_faces(triangulation, 5, 6, 2, periodic_faces);

      triangulation.add_periodicity(periodic_faces);
    }
}

void
cylinder_crossection(Triangulation<2, 3> &triangulation,
                     const double         length,
                     const double         height,
                     const double         cylinder_position,
                     const double         cylinder_diameter,
                     const double         shift,
                     const bool           for_3D)
{
  const unsigned int dim      = 2;
  const unsigned int spacedim = 3;

  (void)shift;

  dealii::Triangulation<dim, spacedim> tria1, tria2, tria3, tria4, tria5, tria6,
    tria7, tria8, tria9, tria_tmp;

  // center
  GridGenerator::subdivided_hyper_rectangle(
    tria1,
    {1, 4},
    Point<2>(-cylinder_diameter, -height / 2.),
    Point<2>(-0.5 * cylinder_diameter, height / 2.));
  GridGenerator::subdivided_hyper_rectangle(
    tria2,
    {1, 4},
    Point<2>(+0.5 * cylinder_diameter, -height / 2.),
    Point<2>(+cylinder_diameter, height / 2.));

  // right
  GridGenerator::subdivided_hyper_rectangle(
    tria3,
    {18, 4},
    Point<2>(cylinder_diameter, -height / 2.),
    Point<2>(length - cylinder_position, height / 2.));

  // left
  GridGenerator::subdivided_hyper_rectangle(
    tria4,
    {for_3D ? 4u : 1u, 4},
    Point<2>(-cylinder_position, -height / 2.),
    Point<2>(-cylinder_diameter, height / 2.));

  tria_tmp.set_mesh_smoothing(triangulation.get_mesh_smoothing());
  GridGenerator::merge_triangulations({&tria1, &tria2, &tria3, &tria4},
                                      tria_tmp,
                                      1.e-12,
                                      true);
  triangulation.copy_triangulation(tria_tmp);


  Tensor<1, 3, double> normal;
  normal[0] = 1.0;

  GridTools::rotate(normal, numbers::PI / 2., triangulation);
}
