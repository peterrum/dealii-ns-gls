#pragma once

#include "grid_cylinder.h"
#include "grid_cylinder_old.h"

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
              const VectorType      &vector) const
  {
    // to be implemented in derived classes
    (void)t;
    (void)mapping;
    (void)dof_handler;
    (void)vector;
  }
};



/**
 * Channel simulation.
 */
template <int dim>
class SimulationChannel : public SimulationBase<dim>
{
public:
  using BoundaryDescriptor = typename SimulationBase<dim>::BoundaryDescriptor;

  SimulationChannel()
    : n_stretching(4)
  {}

  void
  create_triangulation(Triangulation<dim> &tria,
                       const unsigned int  n_global_refinements) const override
  {
    std::vector<unsigned int> n_subdivisions(dim, 1);
    n_subdivisions[0] *= n_stretching;

    Point<dim> p0;
    Point<dim> p1;

    for (unsigned int d = 0; d < dim; ++d)
      p1[d] = 1.0;
    p1[0] *= n_stretching;

    GridGenerator::subdivided_hyper_rectangle(
      tria, n_subdivisions, p0, p1, true);

    tria.refine_global(2);

    tria.refine_global(n_global_refinements);
  }

  virtual BoundaryDescriptor
  get_boundary_descriptor() const override
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
 * Flow-past cylinder simulation.
 */
template <int dim>
class SimulationCylinder : public SimulationBase<dim>
{
public:
  using BoundaryDescriptor = typename SimulationBase<dim>::BoundaryDescriptor;

  SimulationCylinder(const double nu, const bool use_no_slip_cylinder_bc)
    : use_no_slip_cylinder_bc(use_no_slip_cylinder_bc)
    , nu(nu)
  {
    drag_lift_pressure_file.open("drag_lift_pressure.m", std::ios::out);
  }

  ~SimulationCylinder()
  {
    drag_lift_pressure_file.close();
  }

  void
  create_triangulation(Triangulation<dim> &tria,
                       const unsigned int  n_global_refinements) const override
  {
    ExaDG::FlowPastCylinder::create_coarse_grid(tria);

    tria.refine_global(n_global_refinements);
  }

  virtual BoundaryDescriptor
  get_boundary_descriptor() const override
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

  void
  postprocess(const double           t,
              const Mapping<dim>    &mapping,
              const DoFHandler<dim> &dof_handler,
              const VectorType      &solution) const override
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

private:
  const bool   use_no_slip_cylinder_bc;
  const double nu;

  std::shared_ptr<const Utilities::MPI::RemotePointEvaluation<dim>> rpe;

  mutable std::ofstream drag_lift_pressure_file;

  class InflowBoundaryValues : public Function<dim>
  {
  public:
    InflowBoundaryValues()
      : Function<dim>(dim + 1)
      , t_(0.0){};

    double
    value(const Point<dim> &p, const unsigned int component) const override
    {
      const double Um = 1.5;
      const double H  = 0.41;
      const double y  = p[1] - H / 2.0;

      (void)Um;
      (void)H;
      (void)y;

      /// FIXME here. Somehow the velocity is too small
      /// I don't know why.
      const double u_val = 1.0;
      // const double u_val = 2.0 * 4.0 * Um * (y + H / 2.0) * (H / 2.0 - y)
      //*
      //  std::sin((t_+1e-10) * numbers::PI / 8.0) / (H * H)
      ;
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
    const double t_;
  };
};



/**
 * Flow-past cylinder simulation with alternative mesh.
 */
template <int dim>
class SimulationCylinderOld : public SimulationBase<dim>
{
public:
  using BoundaryDescriptor = typename SimulationBase<dim>::BoundaryDescriptor;

  SimulationCylinderOld(const double nu, const bool use_no_slip_cylinder_bc)
    : use_no_slip_cylinder_bc(use_no_slip_cylinder_bc)
    , nu(nu)
  {
    drag_lift_pressure_file.open("drag_lift_pressure.m", std::ios::out);
  }

  ~SimulationCylinderOld()
  {
    drag_lift_pressure_file.close();
  }

  void
  create_triangulation(Triangulation<dim> &tria,
                       const unsigned int  n_global_refinements) const override
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

  virtual BoundaryDescriptor
  get_boundary_descriptor() const override
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

  void
  postprocess(const double           t,
              const Mapping<dim>    &mapping,
              const DoFHandler<dim> &dof_handler,
              const VectorType      &solution) const override
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

private:
  const bool   use_no_slip_cylinder_bc;
  const double nu;

  std::shared_ptr<const Utilities::MPI::RemotePointEvaluation<dim>> rpe;

  mutable std::ofstream drag_lift_pressure_file;

  class InflowBoundaryValues : public Function<dim>
  {
  public:
    InflowBoundaryValues()
      : Function<dim>(dim + 1)
      , t_(0.0){};

    double
    value(const Point<dim> &p, const unsigned int component) const override
    {
      const double Um = 1.5;
      const double H  = 0.41;
      const double y  = p[1];

      (void)Um;
      (void)H;
      (void)y;

      /// FIXME here. Somehow the velocity is too small
      /// I don't know why.
      const double u_val = 1.0;
      // const double u_val = 2.0 * 4.0 * Um * (y + H / 2.0) * (H / 2.0 - y)
      //*
      //  std::sin((t_+1e-10) * numbers::PI / 8.0) / (H * H)
      ;
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
    const double t_;
  };
};



/**
 * Flow-past cylinder simulation with alternative mesh.
 */
template <int dim>
class SimulationCylinderLethe : public SimulationBase<dim>
{
public:
  using BoundaryDescriptor = typename SimulationBase<dim>::BoundaryDescriptor;

  SimulationCylinderLethe()
    : use_no_slip_cylinder_bc(true)
  {}

  void
  create_triangulation(Triangulation<dim> &tria,
                       const unsigned int  n_global_refinements) const override
  {
    GridIn<dim> grid_in(tria);
    grid_in.read("../mesh/cylinder.msh");

    Point<dim>                   circleCenter(8, 8);
    const SphericalManifold<dim> manifold_description(circleCenter);
    tria.set_manifold(0, manifold_description);

    tria.refine_global(n_global_refinements);
  }

  virtual BoundaryDescriptor
  get_boundary_descriptor() const override
  {
    BoundaryDescriptor bcs;

    // inflow
    bcs.all_inhomogeneous_dbcs.emplace_back(
      1, std::make_shared<InflowBoundaryValues>());

    // outflow
    // bcs.all_homogeneous_nbcs.push_back(4);

    // walls
    bcs.all_slip_bcs.push_back(2);

    // cylinder
    if (use_no_slip_cylinder_bc)
      bcs.all_homogeneous_dbcs.push_back(0);
    else
      bcs.all_slip_bcs.push_back(0);

    return bcs;
  }

  void
  postprocess(const double           t,
              const Mapping<dim>    &mapping,
              const DoFHandler<dim> &dof_handler,
              const VectorType      &solution) const override
  {
    // nothing to do
    (void)t;
    (void)mapping;
    (void)dof_handler;
    (void)solution;
  }

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

      const double u_val = 1.0;
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
    const double t_;
  };
};



/**
 * Flow-past cylinder simulation with alternative mesh.
 */
template <int dim>
class SimulationCylinderLethe2 : public SimulationBase<dim>
{
public:
  using BoundaryDescriptor = typename SimulationBase<dim>::BoundaryDescriptor;

  SimulationCylinderLethe2()
    : use_no_slip_cylinder_bc(true)
  {}

  void
  create_triangulation(Triangulation<dim> &tria,
                       const unsigned int  n_global_refinements) const override
  {
    GridGenerator::channel_with_cylinder(tria, 0.03, 2, 2.0, true);

    tria.refine_global(n_global_refinements);

    if (true)
      {
        const auto bb =
          BoundingBox<dim>(Point<dim>(0.2, 0.2)).create_extended(0.12);

        for (const auto &cell : tria.active_cell_iterators())
          if (bb.point_inside(cell->center()))
            cell->set_refine_flag();
        tria.execute_coarsening_and_refinement();
      }
  }

  virtual BoundaryDescriptor
  get_boundary_descriptor() const override
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

  void
  postprocess(const double           t,
              const Mapping<dim>    &mapping,
              const DoFHandler<dim> &dof_handler,
              const VectorType      &solution) const override
  {
    // nothing to do
    (void)t;
    (void)mapping;
    (void)dof_handler;
    (void)solution;
  }

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

      const double u_val = 1.0;
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
    const double t_;
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

  SimulationRotation()
    : use_no_slip_cylinder_bc(true)
  {}

  void
  create_triangulation(Triangulation<dim> &tria,
                       const unsigned int  n_global_refinements) const override
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

  virtual BoundaryDescriptor
  get_boundary_descriptor() const override
  {
    BoundaryDescriptor bcs;

    // inflow
    bcs.all_inhomogeneous_dbcs.emplace_back(
      0, std::make_shared<InflowBoundaryValues>());

    // walls
    bcs.all_homogeneous_dbcs.push_back(1);

    return bcs;
  }

  void
  postprocess(const double           t,
              const Mapping<dim>    &mapping,
              const DoFHandler<dim> &dof_handler,
              const VectorType      &solution) const override
  {
    // nothing to do
    (void)t;
    (void)mapping;
    (void)dof_handler;
    (void)solution;
  }

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
