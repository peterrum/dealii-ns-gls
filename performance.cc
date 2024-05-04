#include <deal.II/base/mpi.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>

#include "include/operator_ns.h"

using namespace dealii;

template <int dim>
void
run(const unsigned int n_global_refinements, const unsigned int fe_degree)
{
  const unsigned int mapping_degree             = 1;
  double             nu                         = 0.1;
  double             c_1                        = 4.0;
  double             c_2                        = 2.0;
  bool               consider_time_deriverative = false;
  bool               cell_wise_stabilization    = true;
  const bool         increment_form             = true;
  const unsigned int bdf_order                  = 2;
  const unsigned int n_repetitions              = 10;

  const ConditionalOStream pcout(
    std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(tria);
  tria.refine_global(n_global_refinements);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(FESystem<dim>(FE_Q<dim>(fe_degree), dim + 1));

  pcout << "Number of DoFs: " << dof_handler.n_dofs() << std::endl;

  QGauss<dim>   quadrature(fe_degree + 1);
  MappingQ<dim> mapping(mapping_degree);

  AffineConstraints<Number> constraints;
  constraints.close();

  TimeIntegratorDataBDF time_integrator_data(bdf_order);

  time_integrator_data.update_dt(0.1);

  NavierStokesOperator<dim, Number> ns_operator(mapping,
                                                dof_handler,
                                                constraints,
                                                constraints,
                                                constraints,
                                                quadrature,
                                                nu,
                                                c_1,
                                                c_2,
                                                time_integrator_data,
                                                consider_time_deriverative,
                                                increment_form,
                                                cell_wise_stabilization);


  // Set solution history
  SolutionHistory<Number> prev_solution(time_integrator_data.get_order() + 1);
  for (auto &vec : prev_solution.get_vectors())
    ns_operator.initialize_dof_vector(vec);
  ns_operator.set_previous_solution(prev_solution);

  // set linearization point
  VectorType<Number> solution;
  ns_operator.initialize_dof_vector(solution);
  ns_operator.set_linearization_point(solution);

  // perform vmult
  VectorType<Number> src, dst;
  ns_operator.initialize_dof_vector(src);
  ns_operator.initialize_dof_vector(dst);

  MyTimerOutput timer;

  {
    MyScope scope(timer, "ns::vmult::mf");
    for (unsigned int i = 0; i < n_repetitions; ++i)
      ns_operator.vmult(dst, src);
  }

  {
    const auto &matrix = ns_operator.get_system_matrix();

    MyScope scope(timer, "ns::vmult::mb");
    for (unsigned int i = 0; i < n_repetitions; ++i)
      matrix.vmult(dst, src);
  }

  {
    typename MatrixFree<dim, Number>::AdditionalData additional_data;

    additional_data.mapping_update_flags = update_values | update_gradients;

    MatrixFree<dim, Number> matrix_free;
    matrix_free.reinit(
      mapping, dof_handler, constraints, quadrature, additional_data);

    VectorType<Number> src, dst;
    matrix_free.initialize_dof_vector(src);
    matrix_free.initialize_dof_vector(dst);

    MyScope scope(timer, "poisson::vmult::mf");
    for (unsigned int i = 0; i < n_repetitions; ++i)
      {
        matrix_free.template cell_loop<VectorType<Number>, VectorType<Number>>(
          [](const auto &data, auto &dst, const auto &src, const auto range) {
            FEEvaluation<dim, -1, 0, dim + 1, Number> phi(data);

            for (unsigned int cell = range.first; cell < range.second; ++cell)
              {
                phi.reinit(cell);

                phi.gather_evaluate(
                  src,
                  EvaluationFlags::EvaluationFlags::values |
                    EvaluationFlags::EvaluationFlags::gradients);

                for (const auto q : phi.quadrature_point_indices())
                  {
                    phi.submit_value(phi.get_value(q), q);
                    phi.submit_gradient(phi.get_gradient(q), q);
                  }

                phi.integrate_scatter(
                  EvaluationFlags::EvaluationFlags::values |
                    EvaluationFlags::EvaluationFlags::gradients,
                  dst);
              }
          },
          dst,
          src,
          true);
      }
  }

  TimerCollection::print_all_wall_time_statistics(true);
}


int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const ConditionalOStream pcout(
    std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  pcout << "Running in "
        <<
#ifdef DEBUG
    "DEBUG"
#else
    "RELEASE"
#endif
        << " mode" << std::endl;

  const unsigned int dim = (argc >= 2) ? std::atoi(argv[1]) : 2;
  const unsigned int n_global_refinements =
    (argc >= 3) ? std::atoi(argv[2]) : 5;
  const unsigned int fe_degree = (argc >= 4) ? std::atoi(argv[3]) : 1;

  if (dim == 2)
    {
      run<2>(n_global_refinements, fe_degree);
    }
  else if (dim == 3)
    {
      run<3>(n_global_refinements, fe_degree);
    }
  else
    {
      AssertThrow(false, ExcNotImplemented());
    }
}