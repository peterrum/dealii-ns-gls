/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef APPLICATIONS_GRID_TOOLS_MESH_FLOW_PAST_CYLINDER_H_
#define APPLICATIONS_GRID_TOOLS_MESH_FLOW_PAST_CYLINDER_H_

// deal.II
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold.h>
#include <deal.II/grid/manifold_lib.h>

// boost
#include <boost/math/special_functions/sign.hpp>

namespace ExaDG
{
  /**
   * Class that provides a spherical manifold applied to one of the faces of a
   * quadrilateral element. On the face subject to the spherical manifold
   * intermediate points are inserted so that an equidistant distribution of
   * points in terms of arclength is obtained. When refining the mesh, all child
   * cells are subject to this "one-sided" spherical volume manifold. This
   * manifold description is available for the two-dimensional case, and for the
   * three-dimensional case with the restriction that the geometry has to be
   * extruded in x3/z-direction.
   */
  template <int dim>
  class OneSidedCylindricalManifold
    : public dealii::ChartManifold<dim, dim, dim>
  {
  public:
    OneSidedCylindricalManifold(
      const dealii::Triangulation<dim>                         &tria_in,
      const typename dealii::Triangulation<dim>::cell_iterator &cell_in,
      const unsigned int                                        face_in,
      const dealii::Point<dim>                                 &center_in)
      : alpha(1.0)
      , radius(1.0)
      , tria(tria_in)
      , cell(cell_in)
      , face(face_in)
      , center(center_in)
    {
      AssertThrow(tria.all_reference_cells_are_hyper_cube(),
                  dealii::ExcMessage(
                    "This class is only implemented for hypercube elements."));

      AssertThrow(
        face <= 3,
        dealii::ExcMessage(
          "One sided spherical manifold can only be applied to face f=0,1,2,3."));

      // get center coordinates in x1-x2 plane
      x_C[0] = center[0];
      x_C[1] = center[1];

      // determine x_1 and x_2 which denote the end points of the face that is
      // subject to the spherical manifold.
      dealii::Point<dim> x_1, x_2;
      x_1 = cell->vertex(get_vertex_id(0));
      x_2 = cell->vertex(get_vertex_id(1));

      dealii::Point<2> x_1_2d = dealii::Point<2>(x_1[0], x_1[1]);
      dealii::Point<2> x_2_2d = dealii::Point<2>(x_2[0], x_2[1]);

      initialize(x_1_2d, x_2_2d);
    }

    void
    initialize(const dealii::Point<2> &x_1, const dealii::Point<2> &x_2)
    {
      const double tol = 1.e-12;

      v_1 = x_1 - x_C;
      v_2 = x_2 - x_C,

      // calculate radius of spherical manifold
        radius = v_1.norm();

      // check correctness of geometry and parameters
      double radius_check = v_2.norm();
      AssertThrow(
        std::abs(radius - radius_check) < tol * radius,
        dealii::ExcMessage(
          "Invalid geometry parameters. To apply a spherical manifold both "
          "end points of the face must have the same distance from the center."));

      // normalize v_1 and v_2
      v_1 /= v_1.norm();
      v_2 /= v_2.norm();

      // calculate angle between v_1 and v_2
      alpha = std::acos(v_1 * v_2);

      // calculate vector that is perpendicular to v_1 in plane that is spanned
      // by v_1 and v_2
      normal = v_2 - (v_2 * v_1) * v_1;

      AssertThrow(normal.norm() > tol,
                  dealii::ExcMessage("Vector must not have length 0."));

      normal /= normal.norm();
    }

    /*
     *  push_forward operation that maps point xi in reference coordinates
     * [0,1]^d to point x in physical coordinates
     */
    dealii::Point<dim>
    push_forward(const dealii::Point<dim> &xi) const override
    {
      dealii::Point<dim> x;

      // standard mapping from reference space to physical space using d-linear
      // shape functions
      for (const unsigned int v : cell->vertex_indices())
        {
          double shape_function_value =
            cell->reference_cell().d_linear_shape_function(xi, v);
          x += shape_function_value * cell->vertex(v);
        }

      // Add contribution of spherical manifold.
      // Here, we only operate in the xi1-xi2 plane.

      // set xi_face, xi_other to xi[0],xi[1] depending on the face that is
      // subject to the manifold
      unsigned int index_face  = get_index_face();
      unsigned int index_other = get_index_other();
      const double xi_face     = xi[index_face];
      const double xi_other    = xi[index_other];

      // calculate deformation related to the spherical manifold
      double beta = xi_face * alpha;

      dealii::Tensor<1, 2> direction;
      direction = std::cos(beta) * v_1 + std::sin(beta) * normal;

      Assert(std::abs(direction.norm() - 1.0) < 1.e-12,
             dealii::ExcMessage("Vector must have length 1."));

      // calculate point x_S on spherical manifold
      dealii::Tensor<1, 2> x_S;
      x_S = x_C + radius * direction;

      // calculate displacement as compared to straight sided quadrilateral
      // element on the face that is subject to the manifold
      dealii::Tensor<1, 2> displ, x_lin;
      for (unsigned int v :
           dealii::ReferenceCells::template get_hypercube<1>().vertex_indices())
        {
          double shape_function_value =
            dealii::ReferenceCells::template get_hypercube<1>()
              .d_linear_shape_function(dealii::Point<1>(xi_face), v);

          unsigned int       vertex_id = get_vertex_id(v);
          dealii::Point<dim> vertex    = cell->vertex(vertex_id);

          x_lin[0] += shape_function_value * vertex[0];
          x_lin[1] += shape_function_value * vertex[1];
        }

      displ = x_S - x_lin;

      // deformation decreases linearly in the second (other) direction
      dealii::Point<1> xi_other_1d = dealii::Point<1>(xi_other);
      unsigned int     index_1d    = get_index_1d();
      double fading_value = dealii::ReferenceCells::template get_hypercube<1>()
                              .d_linear_shape_function(xi_other_1d, index_1d);
      x[0] += fading_value * displ[0];
      x[1] += fading_value * displ[1];

      Assert(dealii::numbers::is_finite(x.norm_square()),
             dealii::ExcMessage("Invalid point found"));

      return x;
    }

    /*
     *  Calculate vertex_id of 2d object (cell in 2d, face4 in 3d)
     *  given the vertex_id of the 1d object (vertex_id_1d = 0,1).
     */
    unsigned int
    get_vertex_id(unsigned int vertex_id_1d) const
    {
      unsigned int vertex_id = 0;

      if (face == 0)
        vertex_id = 2 * vertex_id_1d;
      else if (face == 1)
        vertex_id = 1 + 2 * vertex_id_1d;
      else if (face == 2)
        vertex_id = vertex_id_1d;
      else if (face == 3)
        vertex_id = 2 + vertex_id_1d;

      return vertex_id;
    }

    /*
     *  Calculate index of 1d linear shape function (0 or 1)
     *  that takes a value of 1 on the specified face.
     */
    unsigned int
    get_index_1d() const
    {
      unsigned int index_1d = 0;

      if (face == 0 or face == 2)
        index_1d = 0;
      else if (face == 1 or face == 3)
        index_1d = 1;
      else
        Assert(false, dealii::ExcMessage("Face ID is invalid."));

      return index_1d;
    }

    /*
     *  Calculate which xi-coordinate corresponds to the
     *  tangent direction of the respective face
     */
    unsigned int
    get_index_face() const
    {
      unsigned int index_face = 0;

      if (face == 0 or face == 1)
        index_face = 1;
      else if (face == 2 or face == 3)
        index_face = 0;
      else
        Assert(false, dealii::ExcMessage("Face ID is invalid."));

      return index_face;
    }

    /*
     *  Calculate which xi-coordinate corresponds to
     *  the normal direction of the respective face
     *  in xi1-xi2-plane.
     */
    unsigned int
    get_index_other() const
    {
      return 1 - get_index_face();
    }

    /*
     *  This function calculates the inverse Jacobi matrix dx/d(xi) =
     * d(phi(xi))/d(xi) of the push-forward operation phi: [0,1]^d -> R^d: xi ->
     * x = phi(xi) We assume that the gradient of the standard bilinear shape
     * functions is sufficient to find the solution.
     */
    dealii::Tensor<2, dim>
    get_inverse_jacobian(const dealii::Point<dim> &xi) const
    {
      dealii::Tensor<2, dim> jacobian;

      // standard mapping from reference space to physical space using d-linear
      // shape functions
      for (const unsigned int v : cell->vertex_indices())
        {
          dealii::Tensor<1, dim> shape_function_gradient =
            cell->reference_cell().d_linear_shape_function_gradient(xi, v);
          jacobian += outer_product(cell->vertex(v), shape_function_gradient);
        }

      return invert(jacobian);
    }

    /*
     *  pull_back operation that maps point x in physical coordinates
     *  to point xi in reference coordinates [0,1]^d using the
     *  push_forward operation and Newton's method
     */
    dealii::Point<dim>
    pull_back(const dealii::Point<dim> &x) const override
    {
      dealii::Point<dim>     xi;
      dealii::Tensor<1, dim> residual = push_forward(xi) - x;
      dealii::Tensor<1, dim> delta_xi;

      // Newton method to solve nonlinear pull_back operation
      unsigned int n_iter = 0, MAX_ITER = 100;
      const double TOL = 1.e-12;
      while (residual.norm() > TOL and n_iter < MAX_ITER)
        {
          // multiply by -1.0, i.e., shift residual to the rhs
          residual *= -1.0;

          // solve linear problem
          delta_xi = get_inverse_jacobian(xi) * residual;

          // add increment
          xi += delta_xi;

          // make sure that xi is in the valid range [0,1]^d
          if (xi[0] < 0.0)
            xi[0] = 0.0;
          else if (xi[0] > 1.0)
            xi[0] = 1.0;

          if (xi[1] < 0.0)
            xi[1] = 0.0;
          else if (xi[1] > 1.0)
            xi[1] = 1.0;

          // evaluate residual
          residual = push_forward(xi) - x;

          // increment counter
          ++n_iter;
        }

      Assert(n_iter < MAX_ITER,
             dealii::ExcMessage(
               "Newton solver did not converge to given tolerance. "
               "Maximum number of iterations exceeded."));

      Assert(xi[0] >= 0.0 and xi[0] <= 1.0,
             dealii::ExcMessage(
               "Pull back operation generated invalid xi[0] values."));

      Assert(xi[1] >= 0.0 and xi[1] <= 1.0,
             dealii::ExcMessage(
               "Pull back operation generated invalid xi[1] values."));

      return xi;
    }

    std::unique_ptr<dealii::Manifold<dim>>
    clone() const override
    {
      return std::make_unique<OneSidedCylindricalManifold<dim>>(tria,
                                                                cell,
                                                                face,
                                                                center);
    }

  private:
    dealii::Point<2>     x_C;
    dealii::Tensor<1, 2> v_1;
    dealii::Tensor<1, 2> v_2;
    dealii::Tensor<1, 2> normal;
    double               alpha;
    double               radius;

    const dealii::Triangulation<dim>                  &tria;
    typename dealii::Triangulation<dim>::cell_iterator cell;
    unsigned int                                       face;
    dealii::Point<dim>                                 center;
  };

  /**
   * Class that provides a conical manifold applied to one of the faces of a
   * hexahedral element. On the face subject to the conical manifold
   * intermediate points are inserted so that an equidistant distribution of
   * points in terms of arclength is obtained. When refining the mesh, all child
   * cells are subject to this "one-sided" conical volume manifold. This
   * manifold description is only available for the three-dimensional case where
   * the axis of the cone has to be along the x3/z-direction.
   */
  template <int dim>
  class OneSidedConicalManifold : public dealii::ChartManifold<dim, dim, dim>
  {
  public:
    OneSidedConicalManifold(
      const dealii::Triangulation<dim>                         &tria_in,
      const typename dealii::Triangulation<dim>::cell_iterator &cell_in,
      const unsigned int                                        face_in,
      const dealii::Point<dim>                                 &center_in,
      const double                                              r_0_in,
      const double                                              r_1_in)
      : alpha(1.0)
      , tria(tria_in)
      , cell(cell_in)
      , face(face_in)
      , center(center_in)
      , r_0(r_0_in)
      , r_1(r_1_in)
    {
      AssertThrow(tria.all_reference_cells_are_hyper_cube(),
                  dealii::ExcMessage(
                    "This class is only implemented for hypercube elements."));

      AssertThrow(
        dim == 3,
        dealii::ExcMessage(
          "OneSidedConicalManifold can only be used for 3D problems."));

      AssertThrow(
        face <= 3,
        dealii::ExcMessage(
          "One sided spherical manifold can only be applied to face f=0,1,2,3."));

      // get center coordinates in x1-x2 plane
      x_C[0] = center[0];
      x_C[1] = center[1];

      // determine x_1 and x_2 which denote the end points of the face that is
      // subject to the spherical manifold.
      dealii::Point<dim> x_1, x_2;
      x_1 = cell->vertex(get_vertex_id(0));
      x_2 = cell->vertex(get_vertex_id(1));

      dealii::Point<2> x_1_2d = dealii::Point<2>(x_1[0], x_1[1]);
      dealii::Point<2> x_2_2d = dealii::Point<2>(x_2[0], x_2[1]);

      initialize(x_1_2d, x_2_2d);
    }

    void
    initialize(const dealii::Point<2> &x_1, const dealii::Point<2> &x_2)
    {
      const double tol = 1.e-12;

      v_1 = x_1 - x_C;
      v_2 = x_2 - x_C,

      // calculate radius of spherical manifold
        r_0 = v_1.norm();

      // check correctness of geometry and parameters
      double radius_check = v_2.norm();

      AssertThrow(
        std::abs(r_0 - radius_check) < tol * r_0,
        dealii::ExcMessage(
          "Invalid geometry parameters. To apply a spherical manifold both "
          "end points of the face must have the same distance from the center."));

      // normalize v_1 and v_2
      v_1 /= v_1.norm();
      v_2 /= v_2.norm();

      // calculate angle between v_1 and v_2
      alpha = std::acos(v_1 * v_2);

      // calculate vector that is perpendicular to v_1 in plane that is spanned
      // by v_1 and v_2
      normal = v_2 - (v_2 * v_1) * v_1;

      AssertThrow(normal.norm() > tol,
                  dealii::ExcMessage("Vector must not have length 0."));

      normal /= normal.norm();
    }

    /*
     *  push_forward operation that maps point xi in reference coordinates
     * [0,1]^d to point x in physical coordinates
     */
    dealii::Point<dim>
    push_forward(const dealii::Point<dim> &xi) const override
    {
      dealii::Point<dim> x;

      // standard mapping from reference space to physical space using d-linear
      // shape functions
      for (const unsigned int v : cell->vertex_indices())
        {
          double shape_function_value =
            cell->reference_cell().d_linear_shape_function(xi, v);
          x += shape_function_value * cell->vertex(v);
        }

      // Add contribution of conical manifold.
      // Here, we only operate in the xi1-xi2 plane.

      // set xi_face, xi_other to xi[0],xi[1] depending on the face that is
      // subject to the manifold
      unsigned int index_face  = get_index_face();
      unsigned int index_other = get_index_other();
      const double xi_face     = xi[index_face];
      const double xi_other    = xi[index_other];

      // calculate deformation related to the conical manifold
      double beta = xi_face * alpha;

      dealii::Tensor<1, 2> direction;
      direction = std::cos(beta) * v_1 + std::sin(beta) * normal;

      Assert(std::abs(direction.norm() - 1.0) < 1.e-12,
             dealii::ExcMessage("Vector must have length 1."));

      // calculate point x_S on spherical manifold
      dealii::Tensor<1, 2> x_S;
      x_S = x_C + r_0 * direction;

      // calculate displacement as compared to straight sided quadrilateral
      // element on the face that is subject to the manifold
      dealii::Tensor<1, 2> displ, x_lin;
      for (const unsigned int v :
           dealii::ReferenceCells::template get_hypercube<1>().vertex_indices())
        {
          double shape_function_value =
            dealii::ReferenceCells::template get_hypercube<1>()
              .d_linear_shape_function(dealii::Point<1>(xi_face), v);

          unsigned int       vertex_id = get_vertex_id(v);
          dealii::Point<dim> vertex    = cell->vertex(vertex_id);

          x_lin[0] += shape_function_value * vertex[0];
          x_lin[1] += shape_function_value * vertex[1];
        }

      displ = x_S - x_lin;

      // conical manifold
      displ *= (1 - xi[2] * (r_0 - r_1) / r_0);

      // deformation decreases linearly in the second (other) direction
      dealii::Point<1> xi_other_1d = dealii::Point<1>(xi_other);
      unsigned int     index_1d    = get_index_1d();
      double fading_value = dealii::ReferenceCells::template get_hypercube<1>()
                              .d_linear_shape_function(xi_other_1d, index_1d);
      x[0] += fading_value * displ[0];
      x[1] += fading_value * displ[1];

      Assert(dealii::numbers::is_finite(x.norm_square()),
             dealii::ExcMessage("Invalid point found"));

      return x;
    }

    /*
     *  Calculate vertex_id of 2d object (cell in 2d, face4 in 3d)
     *  given the vertex_id of the 1d object (vertex_id_1d = 0,1).
     */
    unsigned int
    get_vertex_id(unsigned int vertex_id_1d) const
    {
      unsigned int vertex_id = 0;

      if (face == 0)
        vertex_id = 2 * vertex_id_1d;
      else if (face == 1)
        vertex_id = 1 + 2 * vertex_id_1d;
      else if (face == 2)
        vertex_id = vertex_id_1d;
      else if (face == 3)
        vertex_id = 2 + vertex_id_1d;

      return vertex_id;
    }

    /*
     *  Calculate index of 1d linear shape function (0 or 1)
     *  that takes a value of 1 on the specified face.
     */
    unsigned int
    get_index_1d() const
    {
      unsigned int index_1d = 0;

      if (face == 0 or face == 2)
        index_1d = 0;
      else if (face == 1 or face == 3)
        index_1d = 1;
      else
        Assert(false, dealii::ExcMessage("Face ID is invalid."));

      return index_1d;
    }

    /*
     *  Calculate which xi-coordinate corresponds to the
     *  tangent direction of the respective face
     */
    unsigned int
    get_index_face() const
    {
      unsigned int index_face = 0;

      if (face == 0 or face == 1)
        index_face = 1;
      else if (face == 2 or face == 3)
        index_face = 0;
      else
        Assert(false, dealii::ExcMessage("Face ID is invalid."));

      return index_face;
    }

    /*
     *  Calculate which xi-coordinate corresponds to
     *  the normal direction of the respective face
     *  in xi1-xi2-plane.
     */
    unsigned int
    get_index_other() const
    {
      return 1 - get_index_face();
    }

    /*
     *  This function calculates the inverse Jacobi matrix dx/d(xi) =
     * d(phi(xi))/d(xi) of the push-forward operation phi: [0,1]^d -> R^d: xi ->
     * x = phi(xi) We assume that the gradient of the standard bilinear shape
     * functions is sufficient to find the solution.
     */
    dealii::Tensor<2, dim>
    get_inverse_jacobian(const dealii::Point<dim> &xi) const
    {
      dealii::Tensor<2, dim> jacobian;

      // standard mapping from reference space to physical space using d-linear
      // shape functions
      for (const unsigned int v : cell->vertex_indices())
        {
          dealii::Tensor<1, dim> shape_function_gradient =
            cell->reference_cell().d_linear_shape_function_gradient(xi, v);
          jacobian += outer_product(cell->vertex(v), shape_function_gradient);
        }

      return invert(jacobian);
    }

    /*
     *  pull_back operation that maps point x in physical coordinates
     *  to point xi in reference coordinates [0,1]^d using the
     *  push_forward operation and Newton's method
     */
    dealii::Point<dim>
    pull_back(const dealii::Point<dim> &x) const override
    {
      dealii::Point<dim>     xi;
      dealii::Tensor<1, dim> residual = push_forward(xi) - x;
      dealii::Tensor<1, dim> delta_xi;

      // Newton method to solve nonlinear pull_back operation
      unsigned int n_iter = 0, MAX_ITER = 100;
      const double TOL = 1.e-12;
      while (residual.norm() > TOL and n_iter < MAX_ITER)
        {
          // multiply by -1.0, i.e., shift residual to the rhs
          residual *= -1.0;

          // solve linear problem
          delta_xi = get_inverse_jacobian(xi) * residual;

          // add increment
          xi += delta_xi;

          // make sure that xi is in the valid range [0,1]^d
          if (xi[0] < 0.0)
            xi[0] = 0.0;
          else if (xi[0] > 1.0)
            xi[0] = 1.0;

          if (xi[1] < 0.0)
            xi[1] = 0.0;
          else if (xi[1] > 1.0)
            xi[1] = 1.0;

          if (xi[2] < 0.0)
            xi[2] = 0.0;
          else if (xi[2] > 1.0)
            xi[2] = 1.0;

          // evaluate residual
          residual = push_forward(xi) - x;

          // increment counter
          ++n_iter;
        }

      Assert(n_iter < MAX_ITER,
             dealii::ExcMessage(
               "Newton solver did not converge to given tolerance. "
               "Maximum number of iterations exceeded."));

      Assert(xi[0] >= 0.0 and xi[0] <= 1.0,
             dealii::ExcMessage(
               "Pull back operation generated invalid xi[0] values."));

      Assert(xi[1] >= 0.0 and xi[1] <= 1.0,
             dealii::ExcMessage(
               "Pull back operation generated invalid xi[1] values."));

      Assert(xi[2] >= 0.0 and xi[2] <= 1.0,
             dealii::ExcMessage(
               "Pull back operation generated invalid xi[2] values."));

      return xi;
    }

    std::unique_ptr<dealii::Manifold<dim>>
    clone() const override
    {
      return std::make_unique<OneSidedConicalManifold<dim>>(
        tria, cell, face, center, r_0, r_1);
    }


  private:
    dealii::Point<2>     x_C;
    dealii::Tensor<1, 2> v_1;
    dealii::Tensor<1, 2> v_2;
    dealii::Tensor<1, 2> normal;
    double               alpha;

    const dealii::Triangulation<dim>                  &tria;
    typename dealii::Triangulation<dim>::cell_iterator cell;
    unsigned int                                       face;

    dealii::Point<dim> center;

    // radius of cone at xi_3 = 0 (-> r_0) and at xi_3 = 1 (-> r_1)
    double r_0, r_1;
  };


  /**
   * Own implementation of cylindrical manifold with an equidistant distribution
   * of nodes along the cylinder surface.
   */
  template <int dim, int spacedim = dim>
  class MyCylindricalManifold
    : public dealii::ChartManifold<dim, spacedim, spacedim>
  {
  public:
    MyCylindricalManifold(const dealii::Point<spacedim> center_in)
      : dealii::ChartManifold<dim, spacedim, spacedim>(
          MyCylindricalManifold<dim, spacedim>::get_periodicity())
      , center(center_in)
    {}

    dealii::Tensor<1, spacedim>
    get_periodicity()
    {
      dealii::Tensor<1, spacedim> periodicity;

      // angle theta is 2*pi periodic
      periodicity[1] = 2 * dealii::numbers::PI;
      return periodicity;
    }

    dealii::Point<spacedim>
    push_forward(const dealii::Point<spacedim> &ref_point) const override
    {
      const double radius = ref_point[0];
      const double theta  = ref_point[1];

      Assert(ref_point[0] >= 0.0,
             dealii::ExcMessage("Radius must be positive."));

      dealii::Point<spacedim> space_point;
      if (radius > 1e-10)
        {
          AssertThrow(spacedim == 2 or spacedim == 3,
                      dealii::ExcMessage(
                        "Only implemented for 2D and 3D case."));

          space_point[0] = radius * cos(theta);
          space_point[1] = radius * sin(theta);

          if (spacedim == 3)
            space_point[2] = ref_point[2];
        }

      return space_point + center;
    }

    dealii::Point<spacedim>
    pull_back(const dealii::Point<spacedim> &space_point) const override
    {
      dealii::Tensor<1, spacedim> vector;
      vector[0] = space_point[0] - center[0];
      vector[1] = space_point[1] - center[1];
      // for the 3D case: vector[2] will always be 0.

      const double radius = vector.norm();

      dealii::Point<spacedim> ref_point;
      ref_point[0] = radius;
      ref_point[1] = atan2(vector[1], vector[0]);
      if (ref_point[1] < 0)
        ref_point[1] += 2.0 * dealii::numbers::PI;
      if (spacedim == 3)
        ref_point[2] = space_point[2];

      return ref_point;
    }

    std::unique_ptr<dealii::Manifold<dim>>
    clone() const override
    {
      return std::make_unique<MyCylindricalManifold<dim, spacedim>>(center);
    }


  private:
    dealii::Point<dim> center;
  };
} // namespace ExaDG

namespace ExaDG
{
  namespace FlowPastCylinder
  {
    enum class CylinderType
    {
      Circular,
      Square
    };

    // physical dimensions (diameter D and center coordinate Y_C can be varied)
    const double X_0 = 0.0; // origin (x-coordinate)
    const double Y_0 = 0.0; // origin (y-coordinate)
    const double L1  = 0.3; // x-coordinate of inflow boundary (2d test cases)
    const double L2 =
      2.5; // x-coordinate of outflow boundary (=length for 3d test cases)
    const double H   = 0.41; // height of channel
    const double D   = 0.1;  // cylinder diameter
    const double X_C = 0.5;  // center of cylinder (x-coordinate)
    const double Y_C = 0.2;  // center of cylinder (y-coordinate)

    namespace CircularCylinder
    {
      const double X_1 =
        L1; // left x-coordinate of mesh block around the cylinder
      const double X_2 =
        0.7; // right x-coordinate of mesh block around the cylinder
      const double R = D / 2.0; // cylinder radius

      // MeshType
      // Type1: no refinement around cylinder surface (coarsest mesh has 34
      // elements in 2D) Type2: two layers of spherical cells around cylinder
      // (used in Fehn et al. (JCP, 2017, "On the stability of projection
      // methods ...")),
      //        (coarsest mesh has 50 elements in 2D)
      // Type3: coarse mesh has only one element in direction perpendicular to
      // flow direction,
      //        one layer of spherical cells around cylinder for coarsest mesh
      //        (coarsest mesh has 12 elements in 2D)
      // Type4: no refinement around cylinder, coarsest mesh consists of 4 cells
      // for the block that
      //        that surrounds the cylinder (coarsest mesh has 8 elements in 2D)
      enum class MeshType
      {
        Type1,
        Type2,
        Type3,
        Type4
      };
      const MeshType MESH_TYPE = MeshType::Type2;

      // needed for mesh type 2 with two layers of spherical cells around
      // cylinder
      const double R_1 = 1.2 * R;
      const double R_2 = 1.7 * R;

      // needed for mesh type 3 with one layers of spherical cells around
      // cylinder
      const double R_3 = 1.75 * R;

      // ManifoldType
      // Surface manifold: when refining the mesh only the cells close to the
      // manifold-surface are curved (should not be used!) Volume manifold: when
      // refining the mesh all child cells are curved since it is a volume
      // manifold
      enum class ManifoldType
      {
        SurfaceManifold,
        VolumeManifold
      };
      const ManifoldType MANIFOLD_TYPE = ManifoldType::VolumeManifold;

      // manifold ID of spherical manifold
      const unsigned int MANIFOLD_ID = 10;

      // vectors of manifold_ids and face_ids
      std::vector<unsigned int> manifold_ids;
      std::vector<unsigned int> face_ids;

      template <int dim>
      void
      set_boundary_ids(dealii::Triangulation<dim> &tria, bool compute_in_2d)
      {
        // Set the cylinder boundary to 2, outflow to 1, the rest to 0.
        for (auto cell : tria.cell_iterators())
          {
            for (const auto &f : cell->face_indices())
              {
                if (cell->face(f)->at_boundary())
                  {
                    dealii::Point<dim> point_on_centerline;
                    point_on_centerline[0] = X_C;
                    point_on_centerline[1] = Y_C;
                    if (dim == 3)
                      point_on_centerline[dim - 1] = cell->face(f)->center()[2];

                    if (std::abs(cell->face(f)->center()[0] -
                                 (compute_in_2d ? L1 : X_0)) < 1e-12)
                      cell->face(f)->set_all_boundary_ids(0); // PM: modified
                    else if (std::abs(cell->face(f)->center()[0] - L2) < 1e-12)
                      cell->face(f)->set_all_boundary_ids(1); // PM: modified
                    else if (point_on_centerline.distance(
                               cell->face(f)->center()) <= R)
                      cell->face(f)->set_all_boundary_ids(3); // PM: modified
                    else
                      cell->face(f)->set_all_boundary_ids(2); // PM: modified
                  }
              }
          }
      }

      void
      do_create_coarse_triangulation(dealii::Triangulation<2> &tria,
                                     const bool compute_in_2d = true)
      {
        AssertThrow(std::abs((X_2 - X_1) - 2.0 * (X_C - X_1)) < 1.0e-12,
                    dealii::ExcMessage(
                      "Geometry parameters X_1, X_2, X_C invalid!"));

        dealii::Point<2> center = dealii::Point<2>(X_C, Y_C);

        if (MESH_TYPE == MeshType::Type1)
          {
            dealii::Triangulation<2> left, middle, right, tmp, tmp2;

            // left part (only needed for 3D problem)
            std::vector<unsigned int> ref_1(2, 2);
            ref_1[1] = 2;
            dealii::GridGenerator::subdivided_hyper_rectangle(
              left,
              ref_1,
              dealii::Point<2>(X_0, Y_0),
              dealii::Point<2>(X_1, H),
              false);

            // right part (2D and 3D)
            std::vector<unsigned int> ref_2(2, 9);
            ref_2[1] = 2;
            dealii::GridGenerator::subdivided_hyper_rectangle(
              right,
              ref_2,
              dealii::Point<2>(X_2, Y_0),
              dealii::Point<2>(L2, H),
              false);

            // create middle part first as a hyper shell
            const double       outer_radius = (X_2 - X_1) / 2.0;
            const unsigned int n_cells      = 4;
            dealii::Point<2>   current_center =
              dealii::Point<2>((X_1 + X_2) / 2.0, outer_radius);
            dealii::GridGenerator::hyper_shell(
              middle, current_center, R, outer_radius, n_cells, true);
            MyCylindricalManifold<2> boundary(current_center);
            middle.set_manifold(0, boundary);
            middle.refine_global(1);

            // then move the vertices to the points where we want them to be to
            // create a slightly asymmetric cube with a hole
            for (const auto &cell : middle.cell_iterators())
              {
                for (const auto &v : cell->vertex_indices())
                  {
                    dealii::Point<2> &vertex = cell->vertex(v);
                    if (std::abs(vertex[0] - X_2) < 1e-10 and
                        std::abs(vertex[1] - current_center[1]) < 1e-10)
                      vertex = dealii::Point<2>(X_2, H / 2.0);
                    else if (std::abs(vertex[0] -
                                      (current_center[0] +
                                       outer_radius / std::sqrt(2.0))) <
                               1e-10 and
                             std::abs(vertex[1] -
                                      (current_center[1] +
                                       outer_radius / std::sqrt(2.0))) < 1e-10)
                      vertex = dealii::Point<2>(X_2, H);
                    else if (std::abs(vertex[0] -
                                      (current_center[0] +
                                       outer_radius / std::sqrt(2.0))) <
                               1e-10 and
                             std::abs(vertex[1] -
                                      (current_center[1] -
                                       outer_radius / std::sqrt(2.0))) < 1e-10)
                      vertex = dealii::Point<2>(X_2, Y_0);
                    else if (std::abs(vertex[0] - current_center[0]) < 1e-10 and
                             std::abs(vertex[1] - (X_2 - X_1)) < 1e-10)
                      vertex = dealii::Point<2>(current_center[0], H);
                    else if (std::abs(vertex[0] - current_center[0]) < 1e-10 and
                             std::abs(vertex[1] - X_0) < 1e-10)
                      vertex = dealii::Point<2>(current_center[0], X_0);
                    else if (std::abs(vertex[0] -
                                      (current_center[0] -
                                       outer_radius / std::sqrt(2.0))) <
                               1e-10 and
                             std::abs(vertex[1] -
                                      (current_center[1] +
                                       outer_radius / std::sqrt(2.0))) < 1e-10)
                      vertex = dealii::Point<2>(X_1, H);
                    else if (std::abs(vertex[0] -
                                      (current_center[0] -
                                       outer_radius / std::sqrt(2.0))) <
                               1e-10 and
                             std::abs(vertex[1] -
                                      (current_center[1] -
                                       outer_radius / std::sqrt(2.0))) < 1e-10)
                      vertex = dealii::Point<2>(X_1, Y_0);
                    else if (std::abs(vertex[0] - X_1) < 1e-10 and
                             std::abs(vertex[1] - current_center[1]) < 1e-10)
                      vertex = dealii::Point<2>(X_1, H / 2.0);
                  }
              }

            // the same for the inner circle
            for (const auto &cell : middle.cell_iterators())
              {
                for (const auto &v : cell->vertex_indices())
                  {
                    dealii::Point<2> &vertex = cell->vertex(v);

                    // allow to shift cylinder center
                    if (std::abs(vertex.distance(current_center) - R) <
                          1.e-10 or
                        std::abs(vertex.distance(current_center) -
                                 (R + (outer_radius - R) / 2.0)) < 1.e-10)
                      {
                        vertex[0] += center[0] - current_center[0];
                        vertex[1] += center[1] - current_center[1];
                      }
                  }
              }

            // we have to copy the triangulation because we cannot merge
            // triangulations with refinement ...
            dealii::GridGenerator::flatten_triangulation(middle, tmp2);

            if (compute_in_2d)
              {
                dealii::GridGenerator::merge_triangulations(tmp2, right, tria);
              }
            else
              {
                dealii::GridGenerator::merge_triangulations(left, tmp2, tmp);
                dealii::GridGenerator::merge_triangulations(tmp, right, tria);
              }

            if (compute_in_2d)
              {
                // set manifold ID's
                tria.set_all_manifold_ids(0);

                for (auto cell : tria.cell_iterators())
                  {
                    if (MANIFOLD_TYPE == ManifoldType::SurfaceManifold)
                      {
                        for (const auto &f : cell->face_indices())
                          {
                            if (cell->face(f)->at_boundary() and
                                center.distance(cell->face(f)->center()) <= R)
                              {
                                cell->face(f)->set_all_manifold_ids(
                                  MANIFOLD_ID);
                              }
                          }
                      }
                    else if (MANIFOLD_TYPE == ManifoldType::VolumeManifold)
                      {
                        for (const auto &f : cell->face_indices())
                          {
                            if (cell->face(f)->at_boundary())
                              {
                                bool face_at_sphere_boundary = true;
                                for (const auto &v :
                                     cell->face(f)->vertex_indices())
                                  {
                                    if (std::abs(center.distance(
                                                   cell->face(f)->vertex(v)) -
                                                 R) > 1e-12)
                                      {
                                        face_at_sphere_boundary = false;
                                        break;
                                      }
                                  }

                                if (face_at_sphere_boundary)
                                  {
                                    face_ids.push_back(f);
                                    unsigned int manifold_id =
                                      MANIFOLD_ID + manifold_ids.size() + 1;
                                    cell->set_all_manifold_ids(manifold_id);
                                    manifold_ids.push_back(manifold_id);
                                    break;
                                  }
                              }
                          }
                      }
                    else
                      {
                        AssertThrow(
                          MANIFOLD_TYPE == ManifoldType::SurfaceManifold or
                            MANIFOLD_TYPE == ManifoldType::VolumeManifold,
                          dealii::ExcMessage(
                            "Specified manifold type not implemented"));
                      }
                  }
              }
          }
        else if (MESH_TYPE == MeshType::Type2)
          {
            MyCylindricalManifold<2> cylinder_manifold(center);

            dealii::Triangulation<2> left, circle_1, circle_2, circle_tmp,
              middle, middle_tmp, middle_tmp2, right, tmp_3D;

            // left part (only needed for 3D problem)
            std::vector<unsigned int> ref_1(2, 2);
            ref_1[1] = 2;
            dealii::GridGenerator::subdivided_hyper_rectangle(
              left,
              ref_1,
              dealii::Point<2>(X_0, Y_0),
              dealii::Point<2>(X_1, H),
              false);

            // right part (2D and 3D)
            std::vector<unsigned int> ref_2(2, 9);
            ref_2[1] = 2;
            dealii::GridGenerator::subdivided_hyper_rectangle(
              right,
              ref_2,
              dealii::Point<2>(X_2, Y_0),
              dealii::Point<2>(L2, H),
              false);

            // create middle part first as a hyper shell
            const double       outer_radius = (X_2 - X_1) / 2.0;
            const unsigned int n_cells      = 4;
            dealii::GridGenerator::hyper_shell(
              middle, center, R_2, outer_radius, n_cells, true);
            middle.set_all_manifold_ids(MANIFOLD_ID);
            middle.set_manifold(MANIFOLD_ID, cylinder_manifold);
            middle.refine_global(1);

            // two inner circles in order to refine towards the cylinder surface
            const unsigned int n_cells_circle = 8;
            dealii::GridGenerator::hyper_shell(
              circle_1, center, R, R_1, n_cells_circle, true);
            dealii::GridGenerator::hyper_shell(
              circle_2, center, R_1, R_2, n_cells_circle, true);

            // then move the vertices to the points where we want them to be to
            // create a slightly asymmetric cube with a hole
            for (const auto &cell : middle.cell_iterators())
              {
                for (const auto &v : cell->vertex_indices())
                  {
                    dealii::Point<2> &vertex = cell->vertex(v);
                    if (std::abs(vertex[0] - X_2) < 1e-10 and
                        std::abs(vertex[1] - Y_C) < 1e-10)
                      {
                        vertex = dealii::Point<2>(X_2, H / 2.0);
                      }
                    else if (std::abs(vertex[0] - (X_C + (X_2 - X_1) / 2.0 /
                                                           std::sqrt(2))) <
                               1e-10 and
                             std::abs(vertex[1] - (Y_C + (X_2 - X_1) / 2.0 /
                                                           std::sqrt(2))) <
                               1e-10)
                      {
                        vertex = dealii::Point<2>(X_2, H);
                      }
                    else if (std::abs(vertex[0] - (X_C + (X_2 - X_1) / 2.0 /
                                                           std::sqrt(2))) <
                               1e-10 and
                             std::abs(vertex[1] - (Y_C - (X_2 - X_1) / 2.0 /
                                                           std::sqrt(2))) <
                               1e-10)
                      {
                        vertex = dealii::Point<2>(X_2, Y_0);
                      }
                    else if (std::abs(vertex[0] - X_C) < 1e-10 and
                             std::abs(vertex[1] - (Y_C + (X_2 - X_1) / 2.0)) <
                               1e-10)
                      {
                        vertex = dealii::Point<2>(X_C, H);
                      }
                    else if (std::abs(vertex[0] - X_C) < 1e-10 and
                             std::abs(vertex[1] - (Y_C - (X_2 - X_1) / 2.0)) <
                               1e-10)
                      {
                        vertex = dealii::Point<2>(X_C, Y_0);
                      }
                    else if (std::abs(vertex[0] - (X_C - (X_2 - X_1) / 2.0 /
                                                           std::sqrt(2))) <
                               1e-10 and
                             std::abs(vertex[1] - (Y_C + (X_2 - X_1) / 2.0 /
                                                           std::sqrt(2))) <
                               1e-10)
                      {
                        vertex = dealii::Point<2>(X_1, H);
                      }
                    else if (std::abs(vertex[0] - (X_C - (X_2 - X_1) / 2.0 /
                                                           std::sqrt(2))) <
                               1e-10 and
                             std::abs(vertex[1] - (Y_C - (X_2 - X_1) / 2.0 /
                                                           std::sqrt(2))) <
                               1e-10)
                      {
                        vertex = dealii::Point<2>(X_1, Y_0);
                      }
                    else if (std::abs(vertex[0] - X_1) < 1e-10 and
                             std::abs(vertex[1] - Y_C) < 1e-10)
                      {
                        vertex = dealii::Point<2>(X_1, H / 2.0);
                      }
                  }
              }

            // we have to copy the triangulation because we cannot merge
            // triangulations with refinement ...
            dealii::GridGenerator::flatten_triangulation(middle, middle_tmp);

            dealii::GridGenerator::merge_triangulations(circle_1,
                                                        circle_2,
                                                        circle_tmp);
            dealii::GridGenerator::merge_triangulations(middle_tmp,
                                                        circle_tmp,
                                                        middle_tmp2);

            if (compute_in_2d)
              {
                dealii::GridGenerator::merge_triangulations(middle_tmp2,
                                                            right,
                                                            tria);
              }
            else // 3D
              {
                dealii::GridGenerator::merge_triangulations(left,
                                                            middle_tmp2,
                                                            tmp_3D);
                dealii::GridGenerator::merge_triangulations(tmp_3D,
                                                            right,
                                                            tria);
              }

            if (compute_in_2d)
              {
                // set manifold ID's
                tria.set_all_manifold_ids(0);

                for (auto cell : tria.cell_iterators())
                  {
                    if (MANIFOLD_TYPE == ManifoldType::SurfaceManifold)
                      {
                        if (center.distance(cell->center()) <= R_2)
                          cell->set_all_manifold_ids(MANIFOLD_ID);
                      }
                    else if (MANIFOLD_TYPE == ManifoldType::VolumeManifold)
                      {
                        if (center.distance(cell->center()) <= R_2)
                          {
                            cell->set_all_manifold_ids(MANIFOLD_ID);
                          }
                        else
                          {
                            for (const auto &f : cell->face_indices())
                              {
                                if (cell->face(f)->at_boundary())
                                  {
                                    bool face_at_sphere_boundary = true;
                                    for (const auto &v :
                                         cell->face(f)->vertex_indices())
                                      {
                                        if (std::abs(
                                              center.distance(
                                                cell->face(f)->vertex(v)) -
                                              R_2) > 1e-12)
                                          {
                                            face_at_sphere_boundary = false;
                                            break;
                                          }
                                      }

                                    if (face_at_sphere_boundary)
                                      {
                                        face_ids.push_back(f);
                                        unsigned int manifold_id =
                                          MANIFOLD_ID + manifold_ids.size() + 1;
                                        cell->set_all_manifold_ids(manifold_id);
                                        manifold_ids.push_back(manifold_id);
                                        break;
                                      }
                                  }
                              }
                          }
                      }
                  }
              }
          }
        else if (MESH_TYPE == MeshType::Type3)
          {
            dealii::Triangulation<2> left, middle, circle, middle_tmp, right,
              tmp_3D;

            // left part (only needed for 3D problem)
            std::vector<unsigned int> ref_1(2, 1);
            dealii::GridGenerator::subdivided_hyper_rectangle(
              left,
              ref_1,
              dealii::Point<2>(X_0, Y_0),
              dealii::Point<2>(X_1, H),
              false);

            // right part (2D and 3D)
            std::vector<unsigned int> ref_2(2, 4);
            ref_2[1] = 1;
            dealii::GridGenerator::subdivided_hyper_rectangle(
              right,
              ref_2,
              dealii::Point<2>(X_2, Y_0),
              dealii::Point<2>(L2, H),
              false);

            // middle part
            const double       outer_radius = (X_2 - X_1) / 2.0;
            const unsigned int n_cells      = 4;
            dealii::Point<2>   origin;

            // inner circle around cylinder
            dealii::GridGenerator::hyper_shell(
              circle, origin, R, R_3, n_cells, true);
            dealii::GridTools::rotate(dealii::numbers::PI / 4, circle);
            dealii::GridTools::shift(dealii::Point<2>(outer_radius + X_1,
                                                      outer_radius),
                                     circle);

            // create middle part first as a hyper shell
            dealii::GridGenerator::hyper_shell(middle,
                                               origin,
                                               R_3,
                                               outer_radius * std::sqrt(2.0),
                                               n_cells,
                                               true);
            dealii::GridTools::rotate(dealii::numbers::PI / 4, middle);
            dealii::GridTools::shift(dealii::Point<2>(outer_radius + X_1,
                                                      outer_radius),
                                     middle);

            // then move the vertices to the points where we want them to be
            for (const auto &cell : middle.cell_iterators())
              {
                for (const auto &v : cell->vertex_indices())
                  {
                    dealii::Point<2> &vertex = cell->vertex(v);

                    // shift two points at the top to a height of H
                    if (std::abs(vertex[0] - X_1) < 1e-10 and
                        std::abs(vertex[1] - (X_2 - X_1)) < 1e-10)
                      {
                        vertex = dealii::Point<2>(X_1, H);
                      }
                    else if (std::abs(vertex[0] - X_2) < 1e-10 and
                             std::abs(vertex[1] - (X_2 - X_1)) < 1e-10)
                      {
                        vertex = dealii::Point<2>(X_2, H);
                      }

                    // allow to shift cylinder center
                    dealii::Point<2> current_center =
                      dealii::Point<2>((X_1 + X_2) / 2.0, (X_2 - X_1) / 2.0);
                    if (std::abs(vertex.distance(current_center) - R_3) <
                        1.e-10)
                      {
                        vertex[0] += center[0] - current_center[0];
                        vertex[1] += center[1] - current_center[1];
                      }
                  }
              }

            // the same for the inner circle
            for (const auto &cell : circle.cell_iterators())
              {
                for (const auto &v : cell->vertex_indices())
                  {
                    dealii::Point<2> &vertex = cell->vertex(v);

                    // allow to shift cylinder center
                    dealii::Point<2> current_center =
                      dealii::Point<2>((X_1 + X_2) / 2.0, (X_2 - X_1) / 2.0);
                    if (std::abs(vertex.distance(current_center) - R) <
                          1.e-10 or
                        std::abs(vertex.distance(current_center) - R_3) <
                          1.e-10)
                      {
                        vertex[0] += center[0] - current_center[0];
                        vertex[1] += center[1] - current_center[1];
                      }
                  }
              }

            dealii::GridGenerator::merge_triangulations(circle,
                                                        middle,
                                                        middle_tmp);

            if (compute_in_2d)
              {
                dealii::GridGenerator::merge_triangulations(middle_tmp,
                                                            right,
                                                            tria);
              }
            else // 3D
              {
                dealii::GridGenerator::merge_triangulations(left,
                                                            middle_tmp,
                                                            tmp_3D);
                dealii::GridGenerator::merge_triangulations(tmp_3D,
                                                            right,
                                                            tria);
              }

            if (compute_in_2d)
              {
                // set manifold ID's
                tria.set_all_manifold_ids(0);

                for (auto cell : tria.cell_iterators())
                  {
                    if (MANIFOLD_TYPE == ManifoldType::VolumeManifold)
                      {
                        if (center.distance(cell->center()) <= R_3)
                          {
                            cell->set_all_manifold_ids(MANIFOLD_ID);
                          }
                        else
                          {
                            for (const auto &f : cell->face_indices())
                              {
                                if (cell->face(f)->at_boundary())
                                  {
                                    bool face_at_sphere_boundary = true;
                                    for (const auto &v :
                                         cell->face(f)->vertex_indices())
                                      {
                                        if (std::abs(
                                              center.distance(
                                                cell->face(f)->vertex(v)) -
                                              R_3) > 1e-12)
                                          {
                                            face_at_sphere_boundary = false;
                                            break;
                                          }
                                      }

                                    if (face_at_sphere_boundary)
                                      {
                                        face_ids.push_back(f);
                                        unsigned int manifold_id =
                                          MANIFOLD_ID + manifold_ids.size() + 1;
                                        cell->set_all_manifold_ids(manifold_id);
                                        manifold_ids.push_back(manifold_id);
                                        break;
                                      }
                                  }
                              }
                          }
                      }
                    else
                      {
                        AssertThrow(
                          MANIFOLD_TYPE == ManifoldType::VolumeManifold,
                          dealii::ExcMessage(
                            "Specified manifold type not implemented."));
                      }
                  }
              }
          }
        else if (MESH_TYPE == MeshType::Type4)
          {
            dealii::Triangulation<2> left, middle, circle, right, tmp_3D;

            // left part (only needed for 3D problem)
            std::vector<unsigned int> ref_1(2, 1);
            dealii::GridGenerator::subdivided_hyper_rectangle(
              left,
              ref_1,
              dealii::Point<2>(X_0, Y_0),
              dealii::Point<2>(X_1, H),
              false);

            // right part (2D and 3D)
            std::vector<unsigned int> ref_2(2, 4);
            ref_2[1] = 1; // only one cell over channel height
            dealii::GridGenerator::subdivided_hyper_rectangle(
              right,
              ref_2,
              dealii::Point<2>(X_2, Y_0),
              dealii::Point<2>(L2, H),
              false);

            // middle part
            const double       outer_radius = (X_2 - X_1) / 2.0;
            const unsigned int n_cells      = 4;
            dealii::Point<2>   origin;

            // create middle part first as a hyper shell
            dealii::GridGenerator::hyper_shell(
              middle, origin, R, outer_radius * std::sqrt(2.0), n_cells, true);
            dealii::GridTools::rotate(dealii::numbers::PI / 4, middle);
            dealii::GridTools::shift(dealii::Point<2>(outer_radius + X_1,
                                                      outer_radius),
                                     middle);

            // then move the vertices to the points where we want them to be
            for (const auto &cell : middle.cell_iterators())
              {
                for (const auto &v : cell->vertex_indices())
                  {
                    dealii::Point<2> &vertex = cell->vertex(v);

                    // shift two points at the top to a height of H
                    if (std::abs(vertex[0] - X_1) < 1e-10 and
                        std::abs(vertex[1] - (X_2 - X_1)) < 1e-10)
                      {
                        vertex = dealii::Point<2>(X_1, H);
                      }
                    else if (std::abs(vertex[0] - X_2) < 1e-10 and
                             std::abs(vertex[1] - (X_2 - X_1)) < 1e-10)
                      {
                        vertex = dealii::Point<2>(X_2, H);
                      }

                    // allow to shift cylinder center
                    dealii::Point<2> current_center =
                      dealii::Point<2>((X_1 + X_2) / 2.0, (X_2 - X_1) / 2.0);
                    if (std::abs(vertex.distance(current_center) - R) < 1.e-10)
                      {
                        vertex[0] += center[0] - current_center[0];
                        vertex[1] += center[1] - current_center[1];
                      }
                  }
              }


            if (compute_in_2d)
              {
                dealii::GridGenerator::merge_triangulations(middle,
                                                            right,
                                                            tria);
              }
            else // 3D
              {
                dealii::GridGenerator::merge_triangulations(left,
                                                            middle,
                                                            tmp_3D);
                dealii::GridGenerator::merge_triangulations(tmp_3D,
                                                            right,
                                                            tria);
              }

            if (compute_in_2d)
              {
                // set manifold ID's
                tria.set_all_manifold_ids(0);

                for (auto cell : tria.cell_iterators())
                  {
                    if (MANIFOLD_TYPE == ManifoldType::VolumeManifold)
                      {
                        for (const auto &f : cell->face_indices())
                          {
                            if (cell->face(f)->at_boundary())
                              {
                                bool face_at_sphere_boundary = true;
                                for (const auto &v :
                                     cell->face(f)->vertex_indices())
                                  {
                                    if (std::abs(center.distance(
                                                   cell->face(f)->vertex(v)) -
                                                 R) > 1e-12)
                                      {
                                        face_at_sphere_boundary = false;
                                        break;
                                      }
                                  }

                                if (face_at_sphere_boundary)
                                  {
                                    face_ids.push_back(f);
                                    unsigned int manifold_id =
                                      MANIFOLD_ID + manifold_ids.size() + 1;
                                    cell->set_all_manifold_ids(manifold_id);
                                    manifold_ids.push_back(manifold_id);
                                    break;
                                  }
                              }
                          }
                      }
                    else
                      {
                        AssertThrow(
                          MANIFOLD_TYPE == ManifoldType::VolumeManifold,
                          dealii::ExcMessage(
                            "Specified manifold type not implemented."));
                      }
                  }
              }
          }
        else
          {
            AssertThrow(
              MESH_TYPE == MeshType::Type1 or MESH_TYPE == MeshType::Type2 or
                MESH_TYPE == MeshType::Type3 or MESH_TYPE == MeshType::Type4,
              dealii::ExcMessage("Specified mesh type not implemented"));
          }

        if (compute_in_2d == true)
          {
            // Set boundary ID's
            set_boundary_ids<2>(tria, compute_in_2d);
          }
      }


      void
      do_create_coarse_triangulation(dealii::Triangulation<3> &tria)
      {
        dealii::Triangulation<2> tria_2d;
        do_create_coarse_triangulation(tria_2d, false);

        if (MESH_TYPE == MeshType::Type1)
          {
            dealii::GridGenerator::extrude_triangulation(tria_2d, 3, H, tria);

            // set manifold ID's
            tria.set_all_manifold_ids(0);

            if (MANIFOLD_TYPE == ManifoldType::SurfaceManifold)
              {
                for (auto cell : tria.cell_iterators())
                  {
                    for (const auto &f : cell->face_indices())
                      {
                        if (cell->face(f)->at_boundary() and
                            dealii::Point<3>(X_C,
                                             Y_C,
                                             cell->face(f)->center()[2])
                                .distance(cell->face(f)->center()) <= R)
                          {
                            cell->face(f)->set_all_manifold_ids(MANIFOLD_ID);
                          }
                      }
                  }
              }
            else if (MANIFOLD_TYPE == ManifoldType::VolumeManifold)
              {
                for (auto cell : tria.cell_iterators())
                  {
                    for (const auto &f : cell->face_indices())
                      {
                        if (cell->face(f)->at_boundary())
                          {
                            bool face_at_sphere_boundary = true;
                            for (const auto &v :
                                 cell->face(f)->vertex_indices())
                              {
                                if (std::abs(
                                      dealii::Point<3>(
                                        X_C, Y_C, cell->face(f)->vertex(v)[2])
                                        .distance(cell->face(f)->vertex(v)) -
                                      R) > 1e-12)
                                  {
                                    face_at_sphere_boundary = false;
                                    break;
                                  }
                              }

                            if (face_at_sphere_boundary)
                              {
                                face_ids.push_back(f);
                                unsigned int manifold_id =
                                  MANIFOLD_ID + manifold_ids.size() + 1;
                                cell->set_all_manifold_ids(manifold_id);
                                manifold_ids.push_back(manifold_id);
                                break;
                              }
                          }
                      }
                  }
              }
            else
              {
                AssertThrow(MANIFOLD_TYPE == ManifoldType::SurfaceManifold or
                              MANIFOLD_TYPE == ManifoldType::VolumeManifold,
                            dealii::ExcMessage(
                              "Specified manifold type not implemented"));
              }
          }
        else if (MESH_TYPE == MeshType::Type2)
          {
            dealii::GridGenerator::extrude_triangulation(tria_2d, 3, H, tria);

            // set manifold ID's
            tria.set_all_manifold_ids(0);

            if (MANIFOLD_TYPE == ManifoldType::SurfaceManifold)
              {
                for (auto cell : tria.cell_iterators())
                  {
                    if (dealii::Point<3>(X_C, Y_C, cell->center()[2])
                          .distance(cell->center()) <= R_2)
                      cell->set_all_manifold_ids(MANIFOLD_ID);
                  }
              }
            else if (MANIFOLD_TYPE == ManifoldType::VolumeManifold)
              {
                for (auto cell : tria.cell_iterators())
                  {
                    if (dealii::Point<3>(X_C, Y_C, cell->center()[2])
                          .distance(cell->center()) <= R_2)
                      cell->set_all_manifold_ids(MANIFOLD_ID);
                    else
                      {
                        for (const auto &f : cell->face_indices())
                          {
                            if (cell->face(f)->at_boundary())
                              {
                                bool face_at_sphere_boundary = true;
                                for (const auto &v :
                                     cell->face(f)->vertex_indices())
                                  {
                                    if (std::abs(dealii::Point<3>(
                                                   X_C,
                                                   Y_C,
                                                   cell->face(f)->vertex(v)[2])
                                                   .distance(
                                                     cell->face(f)->vertex(v)) -
                                                 R_2) > 1e-12)
                                      {
                                        face_at_sphere_boundary = false;
                                        break;
                                      }
                                  }

                                if (face_at_sphere_boundary)
                                  {
                                    face_ids.push_back(f);
                                    unsigned int manifold_id =
                                      MANIFOLD_ID + manifold_ids.size() + 1;
                                    cell->set_all_manifold_ids(manifold_id);
                                    manifold_ids.push_back(manifold_id);
                                    break;
                                  }
                              }
                          }
                      }
                  }
              }
            else
              {
                AssertThrow(MANIFOLD_TYPE == ManifoldType::SurfaceManifold or
                              MANIFOLD_TYPE == ManifoldType::VolumeManifold,
                            dealii::ExcMessage(
                              "Specified manifold type not implemented"));
              }
          }
        else if (MESH_TYPE == MeshType::Type3)
          {
            dealii::GridGenerator::extrude_triangulation(tria_2d, 2, H, tria);

            // set manifold ID's
            tria.set_all_manifold_ids(0);

            if (MANIFOLD_TYPE == ManifoldType::VolumeManifold)
              {
                for (auto cell : tria.cell_iterators())
                  {
                    if (dealii::Point<3>(X_C, Y_C, cell->center()[2])
                          .distance(cell->center()) <= R_3)
                      {
                        cell->set_all_manifold_ids(MANIFOLD_ID);
                      }
                    else
                      {
                        for (const auto &f : cell->face_indices())
                          {
                            if (cell->face(f)->at_boundary())
                              {
                                bool face_at_sphere_boundary = true;
                                for (const auto &v :
                                     cell->face(f)->vertex_indices())
                                  {
                                    if (std::abs(dealii::Point<3>(
                                                   X_C,
                                                   Y_C,
                                                   cell->face(f)->vertex(v)[2])
                                                   .distance(
                                                     cell->face(f)->vertex(v)) -
                                                 R_3) > 1e-12)
                                      {
                                        face_at_sphere_boundary = false;
                                        break;
                                      }
                                  }

                                if (face_at_sphere_boundary)
                                  {
                                    face_ids.push_back(f);
                                    unsigned int manifold_id =
                                      MANIFOLD_ID + manifold_ids.size() + 1;
                                    cell->set_all_manifold_ids(manifold_id);
                                    manifold_ids.push_back(manifold_id);
                                    break;
                                  }
                              }
                          }
                      }
                  }
              }
            else
              {
                AssertThrow(MANIFOLD_TYPE == ManifoldType::VolumeManifold,
                            dealii::ExcMessage(
                              "Specified manifold type not implemented"));
              }
          }
        else if (MESH_TYPE == MeshType::Type4)
          {
            dealii::GridGenerator::extrude_triangulation(tria_2d, 2, H, tria);

            // set manifold ID's
            tria.set_all_manifold_ids(0);

            if (MANIFOLD_TYPE == ManifoldType::VolumeManifold)
              {
                for (auto cell : tria.cell_iterators())
                  {
                    for (const auto &f : cell->face_indices())
                      {
                        if (cell->face(f)->at_boundary())
                          {
                            bool face_at_sphere_boundary = true;
                            for (const auto &v :
                                 cell->face(f)->vertex_indices())
                              {
                                if (std::abs(
                                      dealii::Point<3>(
                                        X_C, Y_C, cell->face(f)->vertex(v)[2])
                                        .distance(cell->face(f)->vertex(v)) -
                                      R) > 1e-12)
                                  {
                                    face_at_sphere_boundary = false;
                                    break;
                                  }
                              }

                            if (face_at_sphere_boundary)
                              {
                                face_ids.push_back(f);
                                unsigned int manifold_id =
                                  MANIFOLD_ID + manifold_ids.size() + 1;
                                cell->set_all_manifold_ids(manifold_id);
                                manifold_ids.push_back(manifold_id);
                                break;
                              }
                          }
                      }
                  }
              }
            else
              {
                AssertThrow(MANIFOLD_TYPE == ManifoldType::VolumeManifold,
                            dealii::ExcMessage(
                              "Specified manifold type not implemented"));
              }
          }
        else
          {
            AssertThrow(
              MESH_TYPE == MeshType::Type1 or MESH_TYPE == MeshType::Type2 or
                MESH_TYPE == MeshType::Type3 or MESH_TYPE == MeshType::Type4,
              dealii::ExcMessage("Specified mesh type not implemented"));
          }

        // Set boundary ID's
        set_boundary_ids<3>(tria, false);
      }

      template <int dim>
      void
      create_coarse_triangulation(dealii::Triangulation<dim> &triangulation)
      {
        dealii::Point<dim> center;
        center[0] = X_C;
        center[1] = Y_C;

        dealii::Point<3> center_cyl_manifold;
        center_cyl_manifold[0] = center[0];
        center_cyl_manifold[1] = center[1];

        // apply this manifold for all mesh types
        dealii::Point<3> direction;
        direction[2] = 1.;

        static std::shared_ptr<dealii::Manifold<dim>> cylinder_manifold;

        if (MANIFOLD_TYPE == ManifoldType::SurfaceManifold)
          {
            cylinder_manifold = std::shared_ptr<dealii::Manifold<dim>>(
              dim == 2 ?
                static_cast<dealii::Manifold<dim> *>(
                  new dealii::SphericalManifold<dim>(center)) :
                reinterpret_cast<dealii::Manifold<dim> *>(
                  new dealii::CylindricalManifold<3>(direction,
                                                     center_cyl_manifold)));
          }
        else if (MANIFOLD_TYPE == ManifoldType::VolumeManifold)
          {
            cylinder_manifold = std::shared_ptr<dealii::Manifold<dim>>(
              static_cast<dealii::Manifold<dim> *>(
                new MyCylindricalManifold<dim>(center)));
          }
        else
          {
            AssertThrow(MANIFOLD_TYPE == ManifoldType::SurfaceManifold or
                          MANIFOLD_TYPE == ManifoldType::VolumeManifold,
                        dealii::ExcMessage(
                          "Specified manifold type not implemented"));
          }

        do_create_coarse_triangulation(triangulation);
        triangulation.set_manifold(MANIFOLD_ID, *cylinder_manifold);

        // generate vector of manifolds and apply manifold to all cells that
        // have been marked
        static std::vector<std::shared_ptr<dealii::Manifold<dim>>> manifold_vec;
        manifold_vec.resize(manifold_ids.size());

        for (unsigned int i = 0; i < manifold_ids.size(); ++i)
          {
            for (const auto &cell : triangulation.cell_iterators())
              {
                if (cell->manifold_id() == manifold_ids[i])
                  {
                    manifold_vec[i] = std::shared_ptr<dealii::Manifold<dim>>(
                      static_cast<dealii::Manifold<dim> *>(
                        new OneSidedCylindricalManifold<dim>(
                          triangulation, cell, face_ids[i], center)));
                    triangulation.set_manifold(manifold_ids[i],
                                               *(manifold_vec[i]));
                  }
              }
          }

        triangulation.set_manifold(0, dealii::FlatManifold<dim>()); // PM: new
      }

    } // namespace CircularCylinder

    namespace SquareCylinder
    {
      const double L = L2; // length of 3D cylinder
      const double I_x =
        X_C -
        0.5 * D; // position of cylinder in x-direction (bottom left corner)
      const double I_y =
        Y_C -
        0.5 * D; // position of cylinder in y-direction (bottom left corner)

      // left x-coordinate of mesh block around the cylinder
      const double X_1 = L1;

      // shift nodes in a trapez like manner to the square cylinder
      const bool adaptive_mesh_shift = false;

      // = 1 for adaptive_mesh_shift == true
      // > 1 extends the region of a finer mesh to the right of the cylinder
      // (allowed only in the case adaptive_mesh_shift == false)
      const unsigned int FAC_X_2 = 2;

      // right x-coordinate of mesh block around the cylinder
      const double X_2 = X_C + 0.5 * D + FAC_X_2 * I_y;

      const double Y_1 = Y_0 + I_y; // y level of bottom of square cylinder
      const double Y_2 = Y_1 + D;   // y level of top of square cylinder

      // number of elements in y direction
      const unsigned int nele_y_bottom = 3;
      const unsigned int nele_y_top    = 3;
      const unsigned int nele_y_middle = 2;

      // number of elements in x direction
      const unsigned int nele_x_left          = 2;
      const unsigned int nele_x_right         = 10;
      const unsigned int nele_x_middle_middle = 2;
      const unsigned int nele_x_middle_left   = 3;
      const unsigned int nele_x_middle_right  = FAC_X_2 * 3;

      // number of elements in z direction
      const unsigned int nele_z = 5;

      const double h_y_2 = (H - Y_2) / nele_y_top;
      const double h_y_1 = (Y_2 - Y_1) / nele_y_middle;
      const double h_y_0 = (Y_1 - Y_0) / nele_y_bottom;

      const double h_x_2 = (L - X_2) / nele_x_right;
      const double h_x_1 = D / nele_x_middle_middle;
      const double h_x_0 = (X_1 - X_0) / nele_x_left;

      void
      create_trapezoid(dealii::Triangulation<2> &tria,
                       std::vector<unsigned int> ref,
                       const dealii::Point<2>    x_0,
                       const double              length,
                       const double              height,
                       const double              max_shift,
                       const double              min_shift)
      {
        dealii::Triangulation<2> tmp;

        dealii::Point<2> x_1 = x_0 + dealii::Point<2>(length, height);

        dealii::GridGenerator::subdivided_hyper_rectangle(
          tmp, ref, x_0, x_1, false);

        for (dealii::Triangulation<2>::vertex_iterator v = tmp.begin_vertex();
             v != tmp.end_vertex();
             ++v)
          {
            dealii::Point<2> &vertex = v->vertex();

            if (0 < vertex[0] and vertex[0] < length)
              vertex = dealii::Point<2>(vertex[0] + 0.75 * length / ref[0] *
                                                      vertex[0] / length,
                                        vertex[1]);

            const double m = (max_shift - min_shift) / height;

            const double b = max_shift - m * x_1[1];

            vertex =
              dealii::Point<2>(vertex[0],
                               vertex[1] + (m * vertex[1] + b) *
                                             (length - vertex[0]) / length);
          }

        tria.copy_triangulation(tmp);
      }


      template <int dim>
      void
      set_boundary_ids(dealii::Triangulation<dim> &tria)
      {
        // Set the wall boundary and inflow to 0, cylinder boundary to 2,
        // outflow to 1
        for (auto cell : tria.cell_iterators())
          {
            for (const auto &f : cell->face_indices())
              {
                if (cell->face(f)->at_boundary())
                  {
                    dealii::Point<dim> point_on_centerline;
                    point_on_centerline[0] = X_C;
                    point_on_centerline[1] = Y_C;
                    if (dim == 3)
                      point_on_centerline[dim - 1] = cell->face(f)->center()[2];

                    if (dim == 3 ?
                          std::abs(cell->face(f)->center()[0] - X_0) < 1e-12 :
                          std::abs(cell->face(f)->center()[0] - X_1) <
                            1e-12) // inflow
                      {
                        cell->face(f)->set_all_boundary_ids(0);
                      }
                    else if (std::abs(cell->face(f)->center()[0] - (X_0 + L)) <
                             1e-12) // outflow
                      {
                        cell->face(f)->set_all_boundary_ids(1);
                      }
                    else if (std::abs(cell->face(f)->center()[0] -
                                      point_on_centerline[0]) <=
                               (0.5 * D + 1e-12) and
                             std::abs(cell->face(f)->center()[1] -
                                      point_on_centerline[1]) <=
                               (0.5 * D + 1e-12)) // square cylinder walls
                      {
                        cell->face(f)->set_all_boundary_ids(2);
                      }
                    else // domain walls
                      {
                        cell->face(f)->set_all_boundary_ids(0);
                      }
                  }
              }
          }
      }

      template <unsigned int dim>
      void
      do_create_coarse_triangulation(dealii::Triangulation<2> &triangulation,
                                     bool                      is_2d = true)
      {
        const dealii::Triangulation<2>::MeshSmoothing mesh_smoothing =
          triangulation.get_mesh_smoothing();

        dealii::Triangulation<2> left(mesh_smoothing),
          left_bottom(mesh_smoothing), left_middle(mesh_smoothing),
          left_top(mesh_smoothing), middle(mesh_smoothing),
          middle_top(mesh_smoothing), middle_bottom(mesh_smoothing),
          middle_left(mesh_smoothing), middle_right(mesh_smoothing),
          middle_left_top(mesh_smoothing), middle_left_bottom(mesh_smoothing),
          middle_right_top(mesh_smoothing), middle_right_bottom(mesh_smoothing),
          right(mesh_smoothing), right_bottom(mesh_smoothing),
          right_middle(mesh_smoothing), right_top(mesh_smoothing),
          tmp(mesh_smoothing);

        // left
        std::vector<unsigned int> ref_left_top    = {nele_x_left, nele_y_top};
        std::vector<unsigned int> ref_left_bottom = {nele_x_left,
                                                     nele_y_bottom};
        std::vector<unsigned int> ref_left_middle = {nele_x_left,
                                                     nele_y_middle};

        // right
        std::vector<unsigned int> ref_right_top    = {nele_x_right, nele_y_top};
        std::vector<unsigned int> ref_right_bottom = {nele_x_right,
                                                      nele_y_bottom};
        std::vector<unsigned int> ref_right_middle = {nele_x_right,
                                                      nele_y_middle};

        // middle
        std::vector<unsigned int> ref_middle_top       = {nele_x_middle_middle,
                                                          nele_y_top};
        std::vector<unsigned int> ref_middle_top_right = {nele_x_middle_right,
                                                          nele_y_top};
        std::vector<unsigned int> ref_middle_top_left  = {nele_x_middle_left,
                                                          nele_y_top};

        std::vector<unsigned int> ref_middle_bottom_right = {
          nele_x_middle_right, nele_y_bottom};
        std::vector<unsigned int> ref_middle_bottom_left = {nele_x_middle_left,
                                                            nele_y_bottom};
        std::vector<unsigned int> ref_middle_bottom = {nele_x_middle_middle,
                                                       nele_y_bottom};

        std::vector<unsigned int> ref_middle_right = {nele_x_middle_right,
                                                      nele_y_middle};
        std::vector<unsigned int> ref_middle_left  = {nele_x_middle_left,
                                                      nele_y_middle};

        // left part
        dealii::GridGenerator::subdivided_hyper_rectangle(
          left_bottom,
          ref_left_bottom,
          dealii::Point<2>(X_0, Y_0),
          dealii::Point<2>(X_1, Y_1),
          false);

        dealii::GridGenerator::subdivided_hyper_rectangle(
          left_middle,
          ref_left_middle,
          dealii::Point<2>(X_0, Y_1),
          dealii::Point<2>(X_1, Y_2),
          false);

        dealii::GridGenerator::subdivided_hyper_rectangle(
          left_top,
          ref_left_top,
          dealii::Point<2>(X_0, Y_2),
          dealii::Point<2>(X_1, H),
          false);

        // merge left triangulations
        dealii::GridGenerator::merge_triangulations(left_bottom,
                                                    left_middle,
                                                    tmp);
        dealii::GridGenerator::merge_triangulations(tmp, left_top, left);

        // right part
        dealii::GridGenerator::subdivided_hyper_rectangle(
          right_bottom,
          ref_right_bottom,
          dealii::Point<2>(X_2, Y_0),
          dealii::Point<2>(L, Y_1),
          false);

        dealii::GridGenerator::subdivided_hyper_rectangle(
          right_middle,
          ref_right_middle,
          dealii::Point<2>(X_2, Y_1),
          dealii::Point<2>(L, Y_2),
          false);

        dealii::GridGenerator::subdivided_hyper_rectangle(
          right_top,
          ref_right_top,
          dealii::Point<2>(X_2, Y_2),
          dealii::Point<2>(L, H),
          false);

        // merge right triangulations
        dealii::GridGenerator::merge_triangulations(right_bottom,
                                                    right_middle,
                                                    tmp);
        dealii::GridGenerator::merge_triangulations(tmp, right_top, right);

        // middle part
        if (not adaptive_mesh_shift)
          {
            // create middle bottom part
            dealii::GridGenerator::subdivided_hyper_rectangle(
              middle_bottom,
              ref_middle_bottom,
              dealii::Point<2>(X_0 + I_x, Y_0),
              dealii::Point<2>(X_0 + I_x + D, Y_1),
              false);

            // create middle top part
            dealii::GridGenerator::subdivided_hyper_rectangle(
              middle_top,
              ref_middle_top,
              dealii::Point<2>(X_0 + I_x, Y_2),
              dealii::Point<2>(X_0 + I_x + D, H),
              false);

            // create middle left part
            dealii::GridGenerator::subdivided_hyper_rectangle(
              middle_left,
              ref_middle_left,
              dealii::Point<2>(X_1, Y_1),
              dealii::Point<2>(X_0 + I_x, Y_2),
              false);

            dealii::GridGenerator::subdivided_hyper_rectangle(
              middle_left_top,
              ref_middle_top_left,
              dealii::Point<2>(X_1, Y_2),
              dealii::Point<2>(X_0 + I_x, H),
              false);

            dealii::GridGenerator::subdivided_hyper_rectangle(
              middle_left_bottom,
              ref_middle_bottom_left,
              dealii::Point<2>(X_1, Y_0),
              dealii::Point<2>(X_0 + I_x, Y_1),
              false);

            // create middle right part
            dealii::GridGenerator::subdivided_hyper_rectangle(
              middle_right,
              ref_middle_right,
              dealii::Point<2>(X_0 + I_x + D, Y_1),
              dealii::Point<2>(X_2, Y_2),
              false);

            dealii::GridGenerator::subdivided_hyper_rectangle(
              middle_right_top,
              ref_middle_top_right,
              dealii::Point<2>(X_0 + I_x + D, Y_2),
              dealii::Point<2>(X_2, H),
              false);

            dealii::GridGenerator::subdivided_hyper_rectangle(
              middle_right_bottom,
              ref_middle_bottom_right,
              dealii::Point<2>(X_0 + I_x + D, Y_0),
              dealii::Point<2>(X_2, Y_1),
              false);
          }
        else
          {
            dealii::Triangulation<2> middle_right_right;

            unsigned int nele_trapezoid = 4;

            std::vector<unsigned int> ref_trapezoid_left_bottom = {
              nele_trapezoid, 4};
            std::vector<unsigned int> ref_trapezoid_left_middle = {
              nele_trapezoid, 1};
            std::vector<unsigned int> ref_trapezoid_left_top = {nele_trapezoid,
                                                                3};

            std::vector<unsigned int> ref_trapezoid_right_bottom = {
              nele_trapezoid, 4};
            std::vector<unsigned int> ref_trapezoid_right_middle = {
              nele_trapezoid, 1};
            std::vector<unsigned int> ref_trapezoid_right_top = {nele_trapezoid,
                                                                 3};

            std::vector<unsigned int> ref_trapezoid_top = {nele_trapezoid, 6};
            std::vector<unsigned int> ref_trapezoid_bottom = {nele_trapezoid,
                                                              6};

            // create middle left trapez
            create_trapezoid(middle_left_top,
                             ref_trapezoid_left_top,
                             dealii::Point<2>(0, 0),
                             I_x - X_1,
                             3.0 * D / 8.0,
                             H - Y_2,
                             3.0 * D / 8.0);

            dealii::Tensor<1, 2> shift_middle_left_top;
            shift_middle_left_top[0] = X_1;
            shift_middle_left_top[1] = Y_1 + D / 2 + D / 8;
            dealii::GridTools::shift(shift_middle_left_top, middle_left_top);

            create_trapezoid(middle_left,
                             ref_trapezoid_left_middle,
                             dealii::Point<2>(0, 0),
                             I_x - X_1,
                             1.0 * D / 8.0,
                             3.0 * D / 8.0,
                             0);

            dealii::Tensor<1, 2> shift_middle_left;
            shift_middle_left[0] = X_1;
            shift_middle_left[1] = Y_1 + D / 2;
            dealii::GridTools::shift(shift_middle_left, middle_left);

            create_trapezoid(middle_left_bottom,
                             ref_trapezoid_left_bottom,
                             dealii::Point<2>(0, 0),
                             I_x - X_1,
                             D / 2.0,
                             0,
                             -Y_1);

            dealii::Tensor<1, 2> shift_middle_left_bottom;
            shift_middle_left_bottom[0] = X_1;
            shift_middle_left_bottom[1] = Y_1;
            dealii::GridTools::shift(shift_middle_left_bottom,
                                     middle_left_bottom);

            // middle right trapez
            create_trapezoid(middle_right_top,
                             ref_trapezoid_right_top,
                             dealii::Point<2>(0, 0),
                             I_y,
                             3.0 * D / 8.0,
                             -3.0 * D / 8.0,
                             -(H - Y_2));

            dealii::Tensor<1, 2> shift_middle_right_top;
            shift_middle_right_top[0] = I_x + D + I_y;
            shift_middle_right_top[1] = Y_2;
            dealii::GridTools::rotate(M_PI, middle_right_top);
            dealii::GridTools::shift(shift_middle_right_top, middle_right_top);

            create_trapezoid(middle_right,
                             ref_trapezoid_right_middle,
                             dealii::Point<2>(0, 0),
                             I_y,
                             1.0 * D / 8.0,
                             0,
                             -3.0 * D / 8.0);

            dealii::Tensor<1, 2> shift_middle_right;
            shift_middle_right[0] = I_x + D + I_y;
            shift_middle_right[1] = Y_1 + D / 2 + D / 8;
            dealii::GridTools::rotate(M_PI, middle_right);
            dealii::GridTools::shift(shift_middle_right, middle_right);

            create_trapezoid(middle_right_bottom,
                             ref_trapezoid_right_bottom,
                             dealii::Point<2>(0, 0),
                             I_y,
                             D / 2.0,
                             Y_1,
                             0);

            dealii::Tensor<1, 2> shift_middle_right_bottom;
            shift_middle_right_bottom[0] = I_x + D + I_y;
            shift_middle_right_bottom[1] = Y_1 + D / 2;
            dealii::GridTools::rotate(M_PI, middle_right_bottom);
            dealii::GridTools::shift(shift_middle_right_bottom,
                                     middle_right_bottom);

            // create top trapez
            create_trapezoid(middle_top,
                             ref_trapezoid_top,
                             dealii::Point<2>(0, 0),
                             H - Y_2,
                             D,
                             I_y,
                             -(I_x - X_1));

            dealii::Tensor<1, 2> shift_middle_top;
            shift_middle_top[0] = I_x;
            shift_middle_top[1] = H;
            dealii::GridTools::rotate(-M_PI / 2.0, middle_top);
            dealii::GridTools::shift(shift_middle_top, middle_top);

            // create bottom trapez
            create_trapezoid(middle_bottom,
                             ref_trapezoid_bottom,
                             dealii::Point<2>(0, 0),
                             Y_1,
                             D,
                             I_y,
                             -(I_x - X_1));

            dealii::Tensor<1, 2> shift_middle_bottom;
            shift_middle_bottom[0] = I_x + D;
            shift_middle_bottom[1] = 0;
            dealii::GridTools::rotate(M_PI / 2.0, middle_bottom);
            dealii::GridTools::shift(shift_middle_bottom, middle_bottom);
          }

        // merge middle left triangulations
        dealii::GridGenerator::merge_triangulations(middle_left,
                                                    middle_left_top,
                                                    tmp);
        dealii::GridGenerator::merge_triangulations(tmp,
                                                    middle_left_bottom,
                                                    middle_left);

        // merge middle right triangulations
        dealii::GridGenerator::merge_triangulations(middle_right,
                                                    middle_right_top,
                                                    tmp);
        dealii::GridGenerator::merge_triangulations(tmp,
                                                    middle_right_bottom,
                                                    middle_right);


        // merge middle triangulations
        dealii::GridGenerator::merge_triangulations(
          {&middle_bottom, &middle_left, &middle_top, &middle_right}, middle);

        // merge middle and right together
        dealii::GridGenerator::merge_triangulations(right, middle, tmp);

        // merge left to middle and right in 3D case
        if (is_2d)
          triangulation.copy_triangulation(tmp);
        else
          dealii::GridGenerator::merge_triangulations(tmp, left, triangulation);
      }

      template <unsigned int dim>
      void
      do_create_coarse_triangulation(dealii::Triangulation<3> &triangulation)
      {
        dealii::Triangulation<2> tria_2D;

        if (triangulation.get_mesh_smoothing() ==
            dealii::Triangulation<3>::none)
          {
            tria_2D.set_mesh_smoothing(dealii::Triangulation<2>::none);
          }
        else if (triangulation.get_mesh_smoothing() ==
                 dealii::Triangulation<3>::limit_level_difference_at_vertices)
          {
            tria_2D.set_mesh_smoothing(
              dealii::Triangulation<2>::limit_level_difference_at_vertices);
          }
        else
          AssertThrow(false,
                      dealii::ExcMessage("Invalid parameter mesh smoothing."));

        do_create_coarse_triangulation<2>(tria_2D, false);

        dealii::GridGenerator::extrude_triangulation(tria_2D,
                                                     nele_z,
                                                     H,
                                                     triangulation);
      }

      template <unsigned int dim>
      void
      create_coarse_triangulation(dealii::Triangulation<dim> &triangulation)
      {
        do_create_coarse_triangulation<dim>(triangulation);

        // set boundary ids
        set_boundary_ids<dim>(triangulation);
      }

    } // namespace SquareCylinder

    template <int dim>
    void
    create_coarse_grid(
      dealii::Triangulation<dim> &triangulation,
      const CylinderType          cylinder_type = CylinderType::Circular)
    {
      switch (cylinder_type)
        {
          case CylinderType::Circular:
            {
              CircularCylinder::create_coarse_triangulation<dim>(triangulation);
              break;
            }
          case CylinderType::Square:
            {
              SquareCylinder::create_coarse_triangulation<dim>(triangulation);
              break;
            }
          default:
            AssertThrow(false, dealii::ExcNotImplemented());
        }
    }

  } // namespace FlowPastCylinder
} // namespace ExaDG

#endif /* APPLICATIONS_GRID_TOOLS_MESH_FLOW_PAST_CYLINDER_H_ */