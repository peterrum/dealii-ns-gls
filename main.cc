#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>

#include "include/config.h"
#include "include/multigrid.h"
#include "include/operator_base.h"
#include "include/operator_ns.h"
#include "include/preconditioner.h"
#include "include/simulation.h"
#include "include/solver_l.h"
#include "include/solver_nl.h"
#include "include/time_integration.h"

using namespace dealii;



/**
 * Collection of parameters.
 */
struct Parameters
{
  // system
  unsigned int fe_degree            = 1;
  unsigned int mapping_degree       = 1;
  unsigned int n_global_refinements = 0;
  bool         mg_use_fe_q_iso_q1   = false;

  // simulation
  std::string simulation_name = "channel";

  // system
  double       dt             = 0.0;
  double       cfl            = 0.1;
  double       t_final        = 3.0;
  double       theta          = 0.5;
  unsigned int bdf_order      = 1;
  std::string  time_intration = "theta";

  // NSE-GLS parameters
  double nu                         = 0.1;
  double c_1                        = 4.0;
  double c_2                        = 2.0;
  bool   consider_time_deriverative = false;
  bool   cell_wise_stabilization    = true;

  // implmentation of operator evaluation
  bool use_matrix_free_ns_operator = true;

  // linear solver
  std::string  linear_solver          = "GMRES";
  unsigned int lin_n_max_iterations   = 10000;
  double       lin_absolute_tolerance = 1e-12;
  double       lin_relative_tolerance = 1e-8;


  // preconditioner of linear solver
  std::string                     preconditioner = "ILU";
  PreconditionerGMGAdditionalData gmg;

  bool gmg_constraint_coarse_pressure_dof = false;

  // nonlinear solver
  std::string nonlinear_solver = "linearized";
  bool        newton_inexact   = false;

  // output
  std::string paraview_prefix    = "results";
  double      output_granularity = 0.0;

  void
  parse(const std::string file_name)
  {
    if (file_name == "")
      return;

    dealii::ParameterHandler prm;
    add_parameters(prm);

    prm.parse_input(file_name, "", true);
  }

private:
  void
  add_parameters(ParameterHandler &prm)
  {
    // system
    prm.add_parameter("fe degree", fe_degree);
    prm.add_parameter("mapping degree", mapping_degree);
    prm.add_parameter("n global refinements", n_global_refinements);
    prm.add_parameter("gmg coarse grid use fe q iso q1", mg_use_fe_q_iso_q1);

    // simulation
    prm.add_parameter("simulation name", simulation_name);

    // time stepping
    prm.add_parameter("dt", dt);
    prm.add_parameter("cfl", cfl);
    prm.add_parameter("t final", t_final);
    prm.add_parameter("theta", theta);
    prm.add_parameter("bdf order", bdf_order);
    prm.add_parameter("time intration",
                      time_intration,
                      "",
                      Patterns::Selection("bdf|theta|none"));

    // NSE-GLS parameters
    prm.add_parameter("nu", nu);
    prm.add_parameter("c1", c_1);
    prm.add_parameter("c2", c_2);
    prm.add_parameter("consider time deriverative", consider_time_deriverative);
    prm.add_parameter("cell wise stabilization", cell_wise_stabilization);

    // implmentation of operator evaluation
    prm.add_parameter("use matrix free ns operator",
                      use_matrix_free_ns_operator);

    // linear solver
    prm.add_parameter("linear solver",
                      linear_solver,
                      "",
                      Patterns::Selection("GMRES|direct|Richardson"));
    prm.add_parameter("lin n max iterations", lin_n_max_iterations);
    prm.add_parameter("lin absolute tolerance", lin_absolute_tolerance);
    prm.add_parameter("lin relative tolerance", lin_relative_tolerance);

    // preconditioner of linear solver
    prm.add_parameter("preconditioner",
                      preconditioner,
                      "",
                      Patterns::Selection("AMG|GMG|ILU|GMG-LS"));
    gmg.add_parameters(prm);
    prm.add_parameter("gmg constraint coarse pressure dof",
                      gmg_constraint_coarse_pressure_dof);

    // nonlinear solver
    prm.add_parameter("nonlinear solver",
                      nonlinear_solver,
                      "",
                      Patterns::Selection("linearized|Picard|Newton"));
    prm.add_parameter("newton inexact", newton_inexact);

    // output
    prm.add_parameter("paraview prefix", paraview_prefix);
    prm.add_parameter("output granularity", output_granularity);
  }
};



/**
 * Driver class for executing the simulation.
 */
template <int dim>
class Driver
{
public:
  Driver(const std::string &parameter_file_name)
    : parameter_file_name(parameter_file_name)
    , comm(MPI_COMM_WORLD)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0)
  {
    params.parse(parameter_file_name);
  }

  void
  run()
  {
    // select simulation case
    std::shared_ptr<SimulationBase<dim>> simulation;

    if (params.simulation_name == "channel")
      simulation = std::make_shared<SimulationChannel<dim>>();
    else if (params.simulation_name == "cylinder")
      simulation = std::make_shared<SimulationCylinder<dim>>();
    else if (params.simulation_name == "rotation")
      simulation = std::make_shared<SimulationRotation<dim>>();
    else if (params.simulation_name == "sphere")
      simulation = std::make_shared<SimulationSphere<dim>>();
    else
      AssertThrow(false, ExcNotImplemented());
    simulation->parse_parameters(parameter_file_name);

    // set up system
    parallel::distributed::Triangulation<dim> tria(
      comm,
      ::Triangulation<dim>::none,
      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy);

    simulation->create_triangulation(tria, params.n_global_refinements);

    const auto bcs = simulation->get_boundary_descriptor();

    FESystem<dim> fe(FE_Q<dim>(params.fe_degree), dim + 1);

    DoFHandler<dim> dof_handler(tria);
    dof_handler.distribute_dofs(fe);

    DoFHandler<dim> dof_handler_q_iso_q1;

    pcout << "    [I] Number of active cells:    "
          << tria.n_global_active_cells()
          << "\n    [I] Global degrees of freedom: " << dof_handler.n_dofs()
          << std::endl;

    Quadrature<dim> quadrature = QGauss<dim>(params.fe_degree + 1);

    const unsigned int mapping_degree =
      (params.mapping_degree == 0) ? params.fe_degree : params.mapping_degree;

    const auto mapping = simulation->get_mapping(tria, mapping_degree);

    // set up constraints
    ComponentMask mask_v(dim + 1, true);
    mask_v.set(dim, false);

    ComponentMask mask_p(dim + 1, false);
    mask_p.set(dim, true);

    const auto locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);

    AffineConstraints<Number> constraints;
    constraints.reinit(locally_relevant_dofs);

    AffineConstraints<Number> constraints_copy;

    for (const auto bci : bcs.all_homogeneous_dbcs)
      DoFTools::make_zero_boundary_constraints(dof_handler,
                                               bci,
                                               constraints,
                                               mask_v);

    for (const auto bci : bcs.all_homogeneous_nbcs)
      DoFTools::make_zero_boundary_constraints(dof_handler,
                                               bci,
                                               constraints,
                                               mask_p);

    for (const auto bci : bcs.all_slip_bcs)
      VectorTools::compute_no_normal_flux_constraints(
        dof_handler, 0, {bci}, constraints, *mapping, false);

    for (const auto &[face_0, face_1, direction] : bcs.periodic_bcs)
      DoFTools::make_periodicity_constraints(
        dof_handler, face_0, face_1, direction, constraints);

    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    constraints_copy.copy_from(constraints);

    AffineConstraints<Number> constraints_homogeneous;
    constraints_homogeneous.copy_from(constraints);

    for (const auto &[bci, _] : bcs.all_inhomogeneous_dbcs)
      DoFTools::make_zero_boundary_constraints(dof_handler,
                                               bci,
                                               constraints_homogeneous,
                                               mask_v);

    constraints.close();
    constraints_homogeneous.close();

    AffineConstraints<Number> constraints_inhomogeneous;
    // note: filled during time loop

    // set up time integration scheme
    std::shared_ptr<TimeIntegratorData> time_integrator_data;

    if (params.time_intration == "bdf")
      time_integrator_data =
        std::make_shared<TimeIntegratorDataBDF>(params.bdf_order);
    else if (params.time_intration == "theta")
      time_integrator_data =
        std::make_shared<TimeIntegratorDataTheta>(params.theta);
    else if (params.time_intration == "none")
      time_integrator_data = std::make_shared<TimeIntegratorDataNone>();
    else
      AssertThrow(false, ExcNotImplemented());

    // set up Navier-Stokes operator
    std::shared_ptr<OperatorBase<Number>> ns_operator;

    if (params.use_matrix_free_ns_operator)
      {
        const bool increment_form = params.nonlinear_solver == "Newton";

        ns_operator = std::make_shared<NavierStokesOperator<dim, Number>>(
          *mapping,
          dof_handler,
          constraints_homogeneous,
          constraints,
          constraints_inhomogeneous,
          quadrature,
          params.nu,
          params.c_1,
          params.c_2,
          bcs.all_outflow_bcs_cut,
          bcs.all_outflow_bcs_nitsche,
          *time_integrator_data,
          params.consider_time_deriverative,
          increment_form,
          params.cell_wise_stabilization);
      }
    else
      {
        AssertThrow(params.nonlinear_solver != "Newton", ExcInternalError());

        ns_operator =
          std::make_shared<NavierStokesOperatorMatrixBased<dim, Number>>(
            *mapping,
            dof_handler,
            constraints_inhomogeneous,
            quadrature,
            params.nu,
            params.c_1,
            params.c_2,
            *time_integrator_data);
      }

    // set up preconditioner
    std::vector<std::shared_ptr<const Triangulation<dim>>> mg_trias;

    MGLevelObject<DoFHandler<dim>>             mg_dof_handlers;
    MGLevelObject<AffineConstraints<MGNumber>> mg_constraints;

    MGLevelObject<std::shared_ptr<OperatorBase<MGNumber>>> mg_ns_operators;
    MGLevelObject<MGTwoLevelTransfer<dim, VectorType<MGNumber>>> mg_transfers;
    MGLevelObject<MGTwoLevelTransfer<dim, VectorType<MGNumber>>>
      mg_transfers_no_constraints;

    std::shared_ptr<MGTransferGlobalCoarsening<dim, VectorType<MGNumber>>>
      mg_transfer_no_constraints;

    std::shared_ptr<PreconditionerBase> preconditioner;

    if (params.preconditioner == "ILU")
      preconditioner = std::make_shared<PreconditionerILU>(*ns_operator);
    else if (params.preconditioner == "AMG")
      {
        std::vector<std::vector<bool>> constant_modes;

        ComponentMask components(dim + 1, true);
        DoFTools::extract_constant_modes(dof_handler,
                                         components,
                                         constant_modes);

        preconditioner =
          std::make_shared<PreconditionerAMG>(*ns_operator, constant_modes);
      }
    else if (params.preconditioner == "GMG")
      {
        mg_trias =
          MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(
            dof_handler.get_triangulation());

        unsigned int minlevel = 0;
        unsigned int maxlevel = mg_trias.size() - 1;

        mg_dof_handlers.resize(minlevel, maxlevel);
        mg_constraints.resize(minlevel, maxlevel);
        mg_ns_operators.resize(minlevel, maxlevel);
        mg_transfers.resize(minlevel, maxlevel);
        mg_transfers_no_constraints.resize(minlevel, maxlevel);

        for (unsigned int level = minlevel; level <= maxlevel; ++level)
          {
            const auto mapping =
              simulation->get_mapping(*mg_trias[level], mapping_degree);

            if (true)
              {
                DataOut<dim> data_out;
                data_out.attach_triangulation(*mg_trias[level]);
                data_out.build_patches(*mapping, params.fe_degree);

                data_out.write_vtu_in_parallel("grid." + std::to_string(level) +
                                                 ".vtu",
                                               MPI_COMM_WORLD);
              }

            auto &dof_handler = mg_dof_handlers[level];
            auto &constraints = mg_constraints[level];

            auto quadrature_mg = quadrature;

            const auto points =
              true ? QGaussLobatto<1>(params.fe_degree + 1).get_points() :
                     QIterated<1>(QGaussLobatto<1>(2), params.fe_degree)
                       .get_points();

            if (params.mg_use_fe_q_iso_q1 && (level == minlevel))
              quadrature_mg = QIterated<dim>(QGauss<1>(2), points);

            dof_handler.reinit(*mg_trias[level]);

            if (params.mg_use_fe_q_iso_q1 && (level == minlevel))
              dof_handler.distribute_dofs(
                FESystem<dim>(FE_Q_iso_Q1<dim>(points), dim + 1));
            else
              dof_handler.distribute_dofs(fe);

            const auto locally_relevant_dofs =
              DoFTools::extract_locally_relevant_dofs(dof_handler);

            constraints.reinit(locally_relevant_dofs);

            if (params.gmg_constraint_coarse_pressure_dof &&
                (level == minlevel))
              {
                auto min_index = numbers::invalid_dof_index;

                std::vector<types::global_dof_index> dof_indices;

                for (const auto &cell : dof_handler.active_cell_iterators())
                  if (cell->is_locally_owned())
                    {
                      const auto &fe = cell->get_fe();

                      dof_indices.resize(fe.n_dofs_per_cell());
                      cell->get_dof_indices(dof_indices);

                      for (unsigned int i = 0; i < dof_indices.size(); ++i)
                        if (fe.system_to_component_index(i).first == dim)
                          min_index = std::min(min_index, dof_indices[i]);
                    }

                min_index = Utilities::MPI::min(min_index, comm);

                if (locally_relevant_dofs.is_element(min_index))
                  constraints.add_line(min_index);
              }

            for (const auto bci : bcs.all_homogeneous_dbcs)
              DoFTools::make_zero_boundary_constraints(dof_handler,
                                                       bci,
                                                       constraints,
                                                       mask_v);

            for (const auto bci : bcs.all_homogeneous_nbcs)
              DoFTools::make_zero_boundary_constraints(dof_handler,
                                                       bci,
                                                       constraints,
                                                       mask_p);

            for (const auto bci : bcs.all_slip_bcs)
              VectorTools::compute_no_normal_flux_constraints(
                dof_handler, 0, {bci}, constraints, *mapping, false);

            for (const auto &[face_0, face_1, direction] : bcs.periodic_bcs)
              DoFTools::make_periodicity_constraints(
                dof_handler, face_0, face_1, direction, constraints);

            for (const auto &[bci, _] : bcs.all_inhomogeneous_dbcs)
              DoFTools::make_zero_boundary_constraints(dof_handler,
                                                       bci,
                                                       constraints,
                                                       mask_v);

            DoFTools::make_hanging_node_constraints(dof_handler, constraints);

            constraints.close();

            if (params.use_matrix_free_ns_operator)
              {
                const bool increment_form = params.nonlinear_solver == "Newton";

                mg_ns_operators[level] =
                  std::make_shared<NavierStokesOperator<dim, MGNumber>>(
                    *mapping,
                    dof_handler,
                    constraints,
                    constraints,
                    constraints,
                    quadrature_mg,
                    params.nu,
                    params.c_1,
                    params.c_2,
                    bcs.all_outflow_bcs_cut,
                    bcs.all_outflow_bcs_nitsche,
                    *time_integrator_data,
                    params.consider_time_deriverative,
                    increment_form,
                    params.cell_wise_stabilization);
              }
            else
              {
                AssertThrow(false, ExcNotImplemented());
              }
          }


        // create transfer operator for interpolation to the levels (without
        // constraints)
        for (unsigned int level = minlevel; level < maxlevel; ++level)
          mg_transfers_no_constraints[level + 1].reinit(
            mg_dof_handlers[level + 1], mg_dof_handlers[level]);

        mg_transfer_no_constraints = std::make_shared<
          MGTransferGlobalCoarsening<dim, VectorType<MGNumber>>>(
          mg_transfers_no_constraints, [&](const auto l, auto &vec) {
            mg_ns_operators[l]->initialize_dof_vector(vec);
          });


        // create transfer operator for preconditioner (with constraints)
        for (unsigned int level = minlevel; level < maxlevel; ++level)
          mg_transfers[level + 1].reinit(mg_dof_handlers[level + 1],
                                         mg_dof_handlers[level],
                                         mg_constraints[level + 1],
                                         mg_constraints[level]);

        std::shared_ptr<MGTransferGlobalCoarsening<dim, VectorType<MGNumber>>>
          transfer = std::make_shared<
            MGTransferGlobalCoarsening<dim, VectorType<MGNumber>>>(
            mg_transfers, [&](const auto l, auto &vec) {
              mg_ns_operators[l]->initialize_dof_vector(vec);
            });

        // create preconditioner
        preconditioner = std::make_shared<PreconditionerGMG<dim>>(
          dof_handler, mg_ns_operators, transfer, false, params.gmg);
      }
    else if (params.preconditioner == "GMG-LS")
      {
        dof_handler.distribute_mg_dofs(); // TODO

        const auto points =
          true ?
            QGaussLobatto<1>(params.fe_degree + 1).get_points() :
            QIterated<1>(QGaussLobatto<1>(2), params.fe_degree).get_points();

        if (params.mg_use_fe_q_iso_q1)
          {
            dof_handler_q_iso_q1.reinit(tria);
            dof_handler_q_iso_q1.distribute_dofs(
              FESystem<dim>(FE_Q_iso_Q1<dim>(points), dim + 1));
            dof_handler_q_iso_q1.distribute_mg_dofs();
          }

        unsigned int minlevel = 0;
        unsigned int maxlevel = tria.n_global_levels() - 1;

        mg_constraints.resize(minlevel, maxlevel);
        mg_ns_operators.resize(minlevel, maxlevel);

        MGConstrainedDoFs mg_constrained_dofs;
        mg_constrained_dofs.initialize(dof_handler);

        for (const auto bci : bcs.all_homogeneous_dbcs)
          mg_constrained_dofs.make_zero_boundary_constraints(dof_handler,
                                                             {bci},
                                                             mask_v);

        for (const auto bci : bcs.all_homogeneous_nbcs)
          mg_constrained_dofs.make_zero_boundary_constraints(dof_handler,
                                                             {bci},
                                                             mask_p);

        for (const auto &[bci, _] : bcs.all_inhomogeneous_dbcs)
          mg_constrained_dofs.make_zero_boundary_constraints(dof_handler,
                                                             {bci},
                                                             mask_v);

        for (unsigned int level = minlevel; level <= maxlevel; ++level)
          {
            AffineConstraints<Number> constraints;

            const auto &refinement_edge_indices =
              mg_constrained_dofs.get_refinement_edge_indices(level);

            const auto locally_relevant_dofs =
              DoFTools::extract_locally_relevant_level_dofs(dof_handler, level);
            constraints.reinit(locally_relevant_dofs);

            if (params.gmg_constraint_coarse_pressure_dof &&
                (level == minlevel))
              {
                auto min_index = numbers::invalid_dof_index;

                std::vector<types::global_dof_index> dof_indices;

                for (const auto &cell :
                     dof_handler.mg_cell_iterators_on_level(minlevel))
                  if (cell->is_locally_owned_on_level())
                    {
                      const auto &fe = cell->get_fe();

                      dof_indices.resize(fe.n_dofs_per_cell());
                      cell->get_mg_dof_indices(dof_indices);

                      for (unsigned int i = 0; i < dof_indices.size(); ++i)
                        if (fe.system_to_component_index(i).first == dim)
                          min_index = std::min(min_index, dof_indices[i]);
                    }

                min_index = Utilities::MPI::min(min_index, comm);

                if (locally_relevant_dofs.is_element(min_index))
                  constraints.add_line(min_index);
              }

            for (const auto bci : bcs.all_slip_bcs)
              VectorTools::compute_no_normal_flux_constraints_on_level(
                dof_handler,
                0,
                {bci},
                constraints,
                *mapping,
                refinement_edge_indices,
                level,
                false);

            constraints.close();

            mg_constrained_dofs.add_user_constraints(level, constraints);
          }

        for (unsigned int level = minlevel; level <= maxlevel; ++level)
          {
            auto &constraints = mg_constraints[level];

            const auto locally_relevant_dofs =
              DoFTools::extract_locally_relevant_level_dofs(dof_handler, level);
            constraints.reinit(locally_relevant_dofs);

            mg_constrained_dofs.merge_constraints(
              constraints, level, true, false, true, true);

            auto quadrature_mg = quadrature;

            if (params.mg_use_fe_q_iso_q1 && (level == minlevel))
              quadrature_mg = QIterated<dim>(QGauss<1>(2), points);

            if (params.use_matrix_free_ns_operator)
              {
                const bool increment_form = params.nonlinear_solver == "Newton";
                mg_ns_operators[level] =
                  std::make_shared<NavierStokesOperator<dim, MGNumber>>(
                    *mapping,
                    (params.mg_use_fe_q_iso_q1 && (level == minlevel)) ?
                      dof_handler_q_iso_q1 :
                      dof_handler,
                    constraints,
                    constraints,
                    constraints,
                    quadrature_mg,
                    params.nu,
                    params.c_1,
                    params.c_2,
                    bcs.all_outflow_bcs_cut,
                    bcs.all_outflow_bcs_nitsche,
                    *time_integrator_data,
                    params.consider_time_deriverative,
                    increment_form,
                    params.cell_wise_stabilization,
                    level);
              }
            else
              {
                AssertThrow(false, ExcNotImplemented());
              }
          }


        // create transfer operator for interpolation to the levels (without
        // constraints)
        mg_transfer_no_constraints = std::make_shared<
          MGTransferGlobalCoarsening<dim, VectorType<MGNumber>>>();
        mg_transfer_no_constraints->build(
          dof_handler, [&](const auto l, auto &vec) {
            mg_ns_operators[l]->initialize_dof_vector(vec);
          });

        // create transfer operator for preconditioner (with constraints)
        std::shared_ptr<MGTransferGlobalCoarsening<dim, VectorType<MGNumber>>>
          transfer = std::make_shared<
            MGTransferGlobalCoarsening<dim, VectorType<MGNumber>>>(
            mg_constrained_dofs);
        transfer->build(dof_handler, [&](const auto l, auto &vec) {
          mg_ns_operators[l]->initialize_dof_vector(vec);
        });

        // create preconditioner
        preconditioner = std::make_shared<PreconditionerGMG<dim>>(
          dof_handler, mg_ns_operators, transfer, true, params.gmg);
      }
    else
      AssertThrow(false, ExcNotImplemented());


    // set up linear solver
    std::shared_ptr<LinearSolverBase> linear_solver;

    if (params.linear_solver == "GMRES")
      linear_solver =
        std::make_shared<LinearSolverGMRES>(*ns_operator,
                                            *preconditioner,
                                            params.lin_n_max_iterations,
                                            params.lin_absolute_tolerance,
                                            params.lin_relative_tolerance);
    else if (params.linear_solver == "direct")
      linear_solver = std::make_shared<LinearSolverDirect>(*ns_operator);
    else if (params.linear_solver == "Richardson")
      linear_solver =
        std::make_shared<LinearSolverRichardson>(*ns_operator,
                                                 *preconditioner,
                                                 params.lin_n_max_iterations,
                                                 params.lin_absolute_tolerance,
                                                 params.lin_relative_tolerance);
    else
      AssertThrow(false, ExcNotImplemented());

    // set up nonlinear solver
    std::shared_ptr<NonLinearSolverBase> nonlinear_solver;

    if (params.nonlinear_solver == "linearized")
      nonlinear_solver = std::make_shared<NonLinearSolverLinearized>();
    else if (params.nonlinear_solver == "Picard")
      nonlinear_solver = std::make_shared<NonLinearSolverPicard>();
    else if (params.nonlinear_solver == "Newton")
      nonlinear_solver =
        std::make_shared<NonLinearSolverNewton>(params.newton_inexact);
    else
      AssertThrow(false, ExcNotImplemented());

    const auto set_previous_solution =
      [&](const SolutionHistory<Number> &solution) {
        ns_operator->set_previous_solution(solution);

        if (params.preconditioner == "GMG" || params.preconditioner == "GMG-LS")
          {
            MGLevelObject<SolutionHistory<MGNumber>> all_mg_solution(
              mg_ns_operators.min_level(),
              mg_ns_operators.max_level(),
              time_integrator_data->get_order() + 1);

            for (unsigned int i = 1; i <= time_integrator_data->get_order();
                 ++i)
              {
                MGLevelObject<VectorType<MGNumber>> mg_solution(
                  mg_ns_operators.min_level(), mg_ns_operators.max_level());

                mg_transfer_no_constraints->interpolate_to_mg(
                  dof_handler, mg_solution, solution.get_vectors()[i]);

                for (unsigned int l = mg_ns_operators.min_level();
                     l <= mg_ns_operators.max_level();
                     ++l)
                  all_mg_solution[l].get_vectors()[i] = mg_solution[l];
              }

            for (unsigned int l = mg_ns_operators.min_level();
                 l <= mg_ns_operators.max_level();
                 ++l)
              mg_ns_operators[l]->set_previous_solution(all_mg_solution[l]);
          }
      };

    nonlinear_solver->setup_jacobian = [&](const VectorType<Number> &src) {
      ScopedName sc("setup_jacobian");
      MyScope    scope(timer, sc);

      ns_operator->set_linearization_point(src);

      // note: for the level operators, this is done during setup of
      // preconditioner
    };

    nonlinear_solver->setup_preconditioner =
      [&](const VectorType<Number> &solution) {
        ScopedName sc("setup_preconditioner");
        MyScope    scope(timer, sc);

        if (params.preconditioner == "GMG" || params.preconditioner == "GMG-LS")
          {
            MGLevelObject<VectorType<MGNumber>> mg_solution(
              mg_ns_operators.min_level(), mg_ns_operators.max_level());

            mg_transfer_no_constraints->interpolate_to_mg(dof_handler,
                                                          mg_solution,
                                                          solution);

            for (unsigned int l = mg_ns_operators.min_level();
                 l <= mg_ns_operators.max_level();
                 ++l)
              mg_ns_operators[l]->set_linearization_point(mg_solution[l]);
          }

        if (preconditioner)
          preconditioner->initialize();

        linear_solver->initialize();
      };

    nonlinear_solver->evaluate_rhs = [&](VectorType<Number> &dst) {
      ScopedName sc("evaluate_rhs");
      MyScope    scope(timer, sc);

      ns_operator->evaluate_rhs(dst);
    };

    nonlinear_solver->evaluate_residual = [&](VectorType<Number>       &dst,
                                              const VectorType<Number> &src) {
      ScopedName sc("evaluate_residual");
      MyScope    scope(timer, sc);

      ns_operator->evaluate_residual(dst, src);
    };

    nonlinear_solver->solve_with_jacobian = [&](VectorType<Number>       &dst,
                                                const VectorType<Number> &src) {
      ScopedName sc("solve_with_jacobian");
      MyScope    scope(timer, sc);

      constraints_homogeneous.set_zero(const_cast<VectorType<Number> &>(src));
      linear_solver->solve(dst, src);
      constraints_homogeneous.distribute(dst);
    };

    if (false)
      nonlinear_solver->postprocess = [&](const VectorType<Number> &dst) {
        output(0.0, *mapping, dof_handler, dst, true);
      };

    // initialize solution
    SolutionHistory<Number> solution(time_integrator_data->get_order() + 1);

    for (auto &vec : solution.get_vectors())
      ns_operator->initialize_dof_vector(vec);

    {
      // set time-dependent inhomogeneous DBCs
      constraints_inhomogeneous.clear();
      constraints_inhomogeneous.copy_from(constraints_copy);
      for (const auto &[bci, fu] : bcs.all_inhomogeneous_dbcs)
        {
          fu->set_time(0.0); // TODO: correct?
          VectorTools::interpolate_boundary_values(
            *mapping, dof_handler, bci, *fu, constraints_inhomogeneous, mask_v);
        }
      for (const auto &[bci, fu] : bcs.all_outflow_bcs_nitsche)
        {
          fu->set_time(0.0); // TODO: correct?
        }
      constraints_inhomogeneous.close();

      constraints_inhomogeneous.distribute(solution.get_current_solution());
    }

    double       t       = 0.0;
    unsigned int counter = 1;

    output(t, *mapping, dof_handler, solution.get_current_solution());
    simulation->postprocess(t,
                            *mapping,
                            dof_handler,
                            solution.get_current_solution());

    const double min_dx = GridTools::minimal_cell_diameter(tria, *mapping);

    // perform time loop
    for (; t < params.t_final; ++counter)
      {
        ScopedName sc("loop");
        MyScope    scope(timer, sc);

        const double u_max =
          ns_operator->get_max_u(solution.get_current_solution());

        const double dt =
          (params.dt != 0.0) ?
            params.dt :
            (min_dx * params.cfl / std::max(u_max, simulation->get_u_max()));

        pcout << "\ncycle\t" << counter << " at time t = " << t;
        pcout << " with delta_t = " << dt << " and u_max = " << u_max
              << std::endl;

        // set time-dependent inhomogeneous DBCs
        constraints_inhomogeneous.clear();
        constraints_inhomogeneous.copy_from(constraints_copy);
        for (const auto &[bci, fu] : bcs.all_inhomogeneous_dbcs)
          {
            fu->set_time(t); // TODO: correct?
            VectorTools::interpolate_boundary_values(*mapping,
                                                     dof_handler,
                                                     bci,
                                                     *fu,
                                                     constraints_inhomogeneous,
                                                     mask_v);
          }
        for (const auto &[bci, fu] : bcs.all_outflow_bcs_nitsche)
          {
            fu->set_time(t); // TODO: correct?
          }
        constraints_inhomogeneous.close();

        // set time step size
        time_integrator_data->update_dt(dt);

        ns_operator->invalidate_system(); // TODO

        if (params.preconditioner == "GMG" || params.preconditioner == "GMG-LS")
          {
            for (unsigned int l = mg_ns_operators.min_level();
                 l <= mg_ns_operators.max_level();
                 ++l)
              mg_ns_operators[l]->invalidate_system(); // TODO
          }

        // set previous solution
        solution.commit_solution();

        set_previous_solution(solution);

        auto &current_solution = solution.get_current_solution();

        // solve nonlinear problem
        nonlinear_solver->solve(current_solution);

        // apply constraints
        constraints_inhomogeneous.distribute(current_solution);
        constraints.distribute(current_solution);

        pcout << "    [S] l2-norm of solution: " << current_solution.l2_norm()
              << std::endl;

        t += dt;

        // postprocessing
        if (time_integrator_data->get_order() > 0)
          {
            output(t, *mapping, dof_handler, current_solution);
            simulation->postprocess(t, *mapping, dof_handler, current_solution);
          }
        else
          {
            output(t, *mapping, dof_handler, current_solution, true);
            simulation->postprocess(t, *mapping, dof_handler, current_solution);
            break;
          }

        pcout << "\x1B[2J\x1B[H";
      }

    TimerCollection::print_all_wall_time_statistics(true);
  }

private:
  const std::string        parameter_file_name;
  Parameters               params;
  const MPI_Comm           comm;
  const ConditionalOStream pcout;

  mutable MyTimerOutput timer;

  void
  output(const double              time,
         const Mapping<dim>       &mapping,
         const DoFHandler<dim>    &dof_handler,
         const VectorType<Number> &vector,
         const bool                force = false) const
  {
    static unsigned int counter = 0;

    if ((force == false) && ((time + std::numeric_limits<double>::epsilon()) <
                             counter * params.output_granularity))
      return;

    ScopedName sc("postprocess");
    MyScope    scope(timer, sc);

    const std::string file_name =
      params.paraview_prefix + "." + std::to_string(counter) + ".vtu";

    pcout << "    [O] output VTU (" << file_name << ")" << std::endl;

    DataOutBase::VtkFlags flags;
    flags.time                     = time;
    flags.write_higher_order_cells = true;

    DataOut<dim> data_out;
    data_out.set_flags(flags);
    data_out.attach_dof_handler(dof_handler);

    std::vector<std::string> labels(dim + 1, "u");
    labels[dim] = "p";

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim + 1, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation[dim] =
      DataComponentInterpretation::component_is_scalar;

    data_out.add_data_vector(vector,
                             labels,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    data_out.build_patches(mapping, params.fe_degree);

    data_out.write_vtu_in_parallel(file_name, dof_handler.get_communicator());

    counter++;
  }
};



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  // parse input file to get dimension
  const std::string file_name = (argc == 1) ? "" : argv[1];
  unsigned int      dim       = 2;

  dealii::ParameterHandler prm;
  prm.add_parameter("dim", dim);

  if (file_name != "")
    prm.parse_input(file_name, "", true);

  if (dim == 2)
    {
      // 2D simulation
      Driver<2> driver(file_name);
      driver.run();
    }
  else if (dim == 3)
    {
      // 3D simulation
      Driver<3> driver(file_name);
      driver.run();
    }
  else
    {
      AssertThrow(false, ExcNotImplemented());
    }
}
