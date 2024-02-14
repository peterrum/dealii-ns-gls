#pragma once


template <int dim, int spacedim, template <int, int> class Triangulation>
void
cylinder(Triangulation<dim, spacedim> &,
         const double /*length*/,
         const double /*height*/,
         const double /*cylinder_position*/,
         const double /*cylinder_height*/)
{
  AssertThrow(false, dealii::ExcNotImplemented());
}


template <template <int, int> class Triangulation>
void
cylinder(Triangulation<2, 2> &triangulation,
         const double         length,
         const double         height,
         const double         cylinder_position,
         const double         cylinder_diameter)
{
  constexpr int dim = 2;

  using namespace dealii;

  dealii::Triangulation<dim, dim> tria1, tria2, tria3, tria4, tria5, tria6,
    tria7;

  GridGenerator::hyper_cube_with_cylindrical_hole(
    tria1, cylinder_diameter / 2., cylinder_diameter, 0.05, 1, false);

  GridGenerator::subdivided_hyper_rectangle(
    tria2,
    {2, 1},
    Point<2>(-cylinder_diameter, -cylinder_diameter),
    Point<2>(cylinder_diameter, -height / 2. + 0.01));

  GridGenerator::subdivided_hyper_rectangle(
    tria3,
    {2, 1},
    Point<2>(-cylinder_diameter, cylinder_diameter),
    Point<2>(cylinder_diameter, height / 2. + 0.01));

  GridGenerator::subdivided_hyper_rectangle(
    tria4,
    {18, 2},
    Point<2>(cylinder_diameter, -cylinder_diameter),
    Point<2>(length - cylinder_position, cylinder_diameter));

  GridGenerator::subdivided_hyper_rectangle(
    tria5,
    {18, 1},
    Point<2>(cylinder_diameter, cylinder_diameter),
    Point<2>(length - cylinder_position, height / 2. + 0.01));

  GridGenerator::subdivided_hyper_rectangle(
    tria6,
    {18, 1},
    Point<2>(cylinder_diameter, -height / 2. + 0.01),
    Point<2>(length - cylinder_position, -cylinder_diameter));

  tria7.set_mesh_smoothing(triangulation.get_mesh_smoothing());
  GridGenerator::merge_triangulations(
    {&tria1, &tria2, &tria3, &tria4, &tria5, &tria6}, tria7, 1.e-12, true);
  triangulation.copy_triangulation(tria7);

  /* Restore polar manifold for disc: */

  triangulation.set_manifold(0, PolarManifold<2>(Point<2>()));

  /* Fix up position of left boundary: */

  for (auto cell : triangulation.active_cell_iterators())
    for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
      {
        auto &vertex = cell->vertex(v);
        if (vertex[0] <= -cylinder_diameter + 1.e-6)
          vertex[0] = -cylinder_position;
      }

  /*
   * Set boundary ids:
   */

  for (auto cell : triangulation.active_cell_iterators())
    {
      for (auto f : GeometryInfo<2>::face_indices())
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
          else if (std::abs(center[1] - (+height / 2. + 0.01)) < 1.e-6)
            // wall (top)
            face->set_boundary_id(2);
          else if (std::abs(center[1] - (-height / 2. + 0.01)) < 1.e-6)
            // wall (bottom)
            face->set_boundary_id(2);
          else
            face->set_boundary_id(3);
        }
    }
}


template <template <int, int> class Triangulation>
void
cylinder(Triangulation<3, 3> &triangulation,
         const double         length,
         const double         height,
         const double         cylinder_position,
         const double         cylinder_diameter)
{
  using namespace dealii;

  dealii::Triangulation<2, 2> tria1;

  cylinder(tria1, length, height, cylinder_position, cylinder_diameter);

  dealii::Triangulation<3, 3> tria2;
  tria2.set_mesh_smoothing(triangulation.get_mesh_smoothing());

  GridGenerator::extrude_triangulation(tria1, 4, height, tria2, true);
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
      for (auto f : GeometryInfo<3>::face_indices())
        {
          const auto face = cell->face(f);

          if (!face->at_boundary())
            continue;

          /*
           * We want slip boundary conditions (i.e. indicator 1) almost
           * everywhere except on the faces with normal in x-direction.
           * There, on the left side we set inflow conditions (indicator 2)
           * and on the right side we set indicator 0, i.e. do nothing.
           */

          const auto center = face->center();

          if (center[0] > length - cylinder_position - 1.e-6)
            // outflow
            face->set_boundary_id(1);
          else if (center[0] < -cylinder_position + 1.e-6)
            // inflow
            face->set_boundary_id(0);
          else if (std::abs(center[1]) > height / 2. - 1.e-6)
            // y-wall
            face->set_boundary_id(2);
          else if (std::abs(center[2]) > height / 2. - 1.e-6)
            // z-wall
            face->set_boundary_id(2);
          else
            face->set_boundary_id(3);
        }
    }
}
