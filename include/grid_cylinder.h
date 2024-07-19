#pragma once

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

namespace dealii
{
  namespace GridGenerator
  {
    template <int spacedim>
    void
    my_hyper_shell(Triangulation<2, spacedim> &tria,
                   const Point<spacedim>      &center,
                   const double                inner_radius,
                   const double                outer_radius,
                   const unsigned int          n_cells,
                   const bool                  colorize = false)
    {
      Assert((inner_radius > 0) && (inner_radius < outer_radius),
             ExcInvalidRadii());

      const double pi = numbers::PI;

      // determine the number of cells
      // for the grid. if not provided by
      // the user determine it such that
      // the length of each cell on the
      // median (in the middle between
      // the two circles) is equal to its
      // radial extent (which is the
      // difference between the two
      // radii)
      const unsigned int N =
        (n_cells == 0 ? static_cast<unsigned int>(std::ceil(
                          (2 * pi * (outer_radius + inner_radius) / 2) /
                          (outer_radius - inner_radius))) :
                        n_cells);

      // set up N vertices on the
      // outer and N vertices on
      // the inner circle. the
      // first N ones are on the
      // outer one, and all are
      // numbered counter-clockwise
      std::vector<Point<spacedim>> vertices(2 * N);
      for (unsigned int i = 0; i < N; ++i)
        {
          Point<spacedim> point;
          point[0] = std::cos(2 * pi * i / N);
          point[1] = std::sin(2 * pi * i / N);

          vertices[i]     = point * outer_radius;
          vertices[i + N] = vertices[i] * (inner_radius / outer_radius);

          vertices[i] += center;
          vertices[i + N] += center;
        }

      std::vector<CellData<2>> cells(N, CellData<2>());

      for (unsigned int i = 0; i < N; ++i)
        {
          cells[i].vertices[0] = i;
          cells[i].vertices[1] = (i + 1) % N;
          cells[i].vertices[2] = N + i;
          cells[i].vertices[3] = N + ((i + 1) % N);

          cells[i].material_id = 0;
        }

      tria.create_triangulation(vertices, cells, SubCellData());

      AssertThrow(!colorize, ExcNotImplemented());

      tria.set_all_manifold_ids(0);
      tria.set_manifold(0, SphericalManifold<2, spacedim>(center));
    }

    template <int spacedim>
    void
    my_hyper_cube_with_cylindrical_hole(
      Triangulation<2, spacedim> &triangulation,
      const double                inner_radius,
      const double                outer_radius,
      const double,       // width,
      const unsigned int, // width_repetition,
      const bool colorize)
    {
      const int dim = 2;

      Assert(inner_radius < outer_radius,
             ExcMessage("outer_radius has to be bigger than inner_radius."));

      const Point<spacedim> center;

      // We create a hyper_shell (i.e., an annulus) in two dimensions, and then
      // we modify it by pulling the vertices on the diagonals out to where the
      // corners of a square would be:
      my_hyper_shell(triangulation, center, inner_radius, outer_radius, 8);
      triangulation.set_all_manifold_ids(numbers::flat_manifold_id);
      std::vector<bool> treated_vertices(triangulation.n_vertices(), false);
      for (const auto &cell : triangulation.active_cell_iterators())
        {
          for (auto f : GeometryInfo<dim>::face_indices())
            if (cell->face(f)->at_boundary())
              for (const unsigned int v : cell->face(f)->vertex_indices())
                if (/* is the vertex on the outer ring? */
                    (std::fabs(cell->face(f)->vertex(v).norm() - outer_radius) <
                     1e-12 * outer_radius)
                    /* and */
                    &&
                    /* is the vertex on one of the two diagonals? */
                    (std::fabs(std::fabs(cell->face(f)->vertex(v)[0]) -
                               std::fabs(cell->face(f)->vertex(v)[1])) <
                     1e-12 * outer_radius))
                  cell->face(f)->vertex(v) *= std::sqrt(2.);
        }
      const double eps = 1e-3 * outer_radius;
      for (const auto &cell : triangulation.active_cell_iterators())
        {
          for (const unsigned int f : cell->face_indices())
            if (cell->face(f)->at_boundary())
              {
                const double dx = cell->face(f)->center()[0] - center[0];
                const double dy = cell->face(f)->center()[1] - center[1];
                if (colorize)
                  {
                    if (std::abs(dx + outer_radius) < eps)
                      cell->face(f)->set_boundary_id(0);
                    else if (std::abs(dx - outer_radius) < eps)
                      cell->face(f)->set_boundary_id(1);
                    else if (std::abs(dy + outer_radius) < eps)
                      cell->face(f)->set_boundary_id(2);
                    else if (std::abs(dy - outer_radius) < eps)
                      cell->face(f)->set_boundary_id(3);
                    else
                      {
                        cell->face(f)->set_boundary_id(4);
                        cell->face(f)->set_manifold_id(0);
                      }
                  }
                else
                  {
                    const double d = (cell->face(f)->center() - center).norm();
                    if (d - inner_radius < 0)
                      {
                        cell->face(f)->set_boundary_id(1);
                        cell->face(f)->set_manifold_id(0);
                      }
                    else
                      cell->face(f)->set_boundary_id(0);
                  }
              }
        }
      triangulation.set_manifold(0, PolarManifold<2, spacedim>(center));
    }
  } // namespace GridGenerator
} // namespace dealii



template <int spacedim>
void
cylinder(Triangulation<2, spacedim> &triangulation,
         const double                length,
         const double                height,
         const double                cylinder_position,
         const double                cylinder_diameter,
         const double                shift,
         const bool                  for_3D = false)
{
  constexpr int dim = 2;

  using namespace dealii;

  dealii::Triangulation<dim, spacedim> tria1, tria2, tria3, tria4, tria5, tria6,
    tria7, tria8, tria9, tria_tmp;

  // center
  GridGenerator::my_hyper_cube_with_cylindrical_hole(
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
            face->set_boundary_id(2);
          else if (std::abs(center[1] - (-height / 2. + shift)) < 1.e-6)
            // wall (bottom)
            face->set_boundary_id(2);
          else
            face->set_boundary_id(3);
        }
    }
}

void
cylinder(Triangulation<3, 3> &triangulation,
         const double         length,
         const double         height,
         const double         cylinder_position,
         const double         cylinder_diameter,
         const double         shift)
{
  dealii::Triangulation<2, 2> tria1;

  cylinder(
    tria1, length, height, cylinder_position, cylinder_diameter, shift, true);

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
          else if (std::abs(center[1] - (+height / 2. + shift)) < 1.e-6)
            // wall (top)
            face->set_boundary_id(2);
          else if (std::abs(center[1] - (-height / 2. + shift)) < 1.e-6)
            // wall (bottom)
            face->set_boundary_id(2);
          else if (std::abs(center[2] - (+height / 2.)) < 1.e-6)
            // wall (top)
            face->set_boundary_id(2);
          else if (std::abs(center[2] - (-height / 2.)) < 1.e-6)
            // wall (bottom)
            face->set_boundary_id(2);
          else
            face->set_boundary_id(3);
        }
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
