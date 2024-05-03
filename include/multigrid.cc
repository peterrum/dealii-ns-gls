#include "multigrid.h"



// TODO: make part of deal.II
namespace dealii
{
  /**
   * Coarse grid solver using a preconditioner only. This is a little wrapper,
   * transforming a preconditioner into a coarse grid solver.
   */
  template <class VectorType, class PreconditionerType>
  class MGCoarseGridApplyPreconditioner : public MGCoarseGridBase<VectorType>
  {
  public:
    /**
     * Default constructor.
     */
    MGCoarseGridApplyPreconditioner();

    /**
     * Constructor. Store a pointer to the preconditioner for later use.
     */
    MGCoarseGridApplyPreconditioner(const PreconditionerType &precondition);

    /**
     * Clear the pointer.
     */
    void
    clear();

    /**
     * Initialize new data.
     */
    void
    initialize(const PreconditionerType &precondition);

    /**
     * Implementation of the abstract function.
     */
    virtual void
    operator()(const unsigned int level,
               VectorType        &dst,
               const VectorType  &src) const override;

  private:
    /**
     * Reference to the preconditioner.
     */
    SmartPointer<
      const PreconditionerType,
      MGCoarseGridApplyPreconditioner<VectorType, PreconditionerType>>
      preconditioner;
  };



  template <class VectorType, class PreconditionerType>
  MGCoarseGridApplyPreconditioner<VectorType, PreconditionerType>::
    MGCoarseGridApplyPreconditioner()
    : preconditioner(0, typeid(*this).name())
  {}



  template <class VectorType, class PreconditionerType>
  MGCoarseGridApplyPreconditioner<VectorType, PreconditionerType>::
    MGCoarseGridApplyPreconditioner(const PreconditionerType &preconditioner)
    : preconditioner(&preconditioner, typeid(*this).name())
  {}



  template <class VectorType, class PreconditionerType>
  void
  MGCoarseGridApplyPreconditioner<VectorType, PreconditionerType>::initialize(
    const PreconditionerType &preconditioner_)
  {
    preconditioner = &preconditioner_;
  }



  template <class VectorType, class PreconditionerType>
  void
  MGCoarseGridApplyPreconditioner<VectorType, PreconditionerType>::clear()
  {
    preconditioner = 0;
  }



  namespace internal
  {
    namespace MGCoarseGridApplyPreconditioner
    {
      template <class VectorType,
                class PreconditionerType,
                typename std::enable_if<
                  std::is_same<typename VectorType::value_type, double>::value,
                  VectorType>::type * = nullptr>
      void
      solve(const PreconditionerType preconditioner,
            VectorType              &dst,
            const VectorType        &src)
      {
        // to allow the case that the preconditioner was only set up on a
        // subset of processes
        if (preconditioner != nullptr)
          preconditioner->vmult(dst, src);
      }

      template <class VectorType,
                class PreconditionerType,
                typename std::enable_if<
                  !std::is_same<typename VectorType::value_type, double>::value,
                  VectorType>::type * = nullptr>
      void
      solve(const PreconditionerType preconditioner,
            VectorType              &dst,
            const VectorType        &src)
      {
        LinearAlgebra::distributed::Vector<double> src_;
        LinearAlgebra::distributed::Vector<double> dst_;

        src_ = src;
        dst_ = dst;

        // to allow the case that the preconditioner was only set up on a
        // subset of processes
        if (preconditioner != nullptr)
          preconditioner->vmult(dst_, src_);

        dst = dst_;
      }
    } // namespace MGCoarseGridApplyPreconditioner
  }   // namespace internal


  template <class VectorType, class PreconditionerType>
  void
  MGCoarseGridApplyPreconditioner<VectorType, PreconditionerType>::operator()(
    const unsigned int /*level*/,
    VectorType       &dst,
    const VectorType &src) const
  {
    internal::MGCoarseGridApplyPreconditioner::solve(preconditioner, dst, src);
  }
} // namespace dealii



void
PreconditionerGMGAdditionalData::add_parameters(ParameterHandler &prm)
{
  // smoother
  prm.add_parameter("gmg output details", output_details);

  // smoother
  prm.add_parameter("gmg smoothing n iterations", smoothing_n_iterations);

  // coarse-grid solver
  prm.add_parameter("gmg coarse grid solver",
                    coarse_grid_solver,
                    "",
                    Patterns::Selection("AMG|direct|identity"));
  prm.add_parameter("gmg coarse grid iterate", coarse_grid_iterate);

  // coarse-grid GMRES
  prm.add_parameter("gmg coarse grid gmres reltol", coarse_grid_gmres_reltol);

  // coarse-grid AMG
  prm.add_parameter("gmg coarse grid amg use default parameters",
                    coarse_grid_amg_use_default_parameters);
}



template <int dim>
PreconditionerGMG<dim>::PreconditionerGMG(
  const DoFHandler<dim>                                        &dof_handler,
  const MGLevelObject<std::shared_ptr<OperatorBase<MGNumber>>> &op,
  const std::shared_ptr<MGTransferGlobalCoarsening<dim, VectorType<MGNumber>>>
                                        &transfer,
  const bool                             consider_edge_constraints,
  const PreconditionerGMGAdditionalData &additional_data)
  : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  , pcout_cond(std::cout,
               (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) &&
                 additional_data.output_details)
  , dof_handler(dof_handler)
  , op(op)
  , transfer(transfer)
  , consider_edge_constraints(consider_edge_constraints)
  , additional_data(additional_data)
{}



template <int dim>
void
PreconditionerGMG<dim>::vmult(VectorType<Number>       &dst,
                              const VectorType<Number> &src) const
{
  MyScope scope(timer, "gmg::vmult");

  Assert(preconditioner, ExcInternalError());
  preconditioner->vmult(dst, src);

  if (additional_data.coarse_grid_iterate)
    {
      pcout_cond << "    [C] solved in "
                 << coarse_grid_solver_control->last_step() << " iterations."
                 << std::endl;

      n_coarse_iterations.emplace_back(coarse_grid_solver_control->last_step());
    }
}



template <int dim>
void
PreconditionerGMG<dim>::print_stats() const
{
  if (additional_data.coarse_grid_iterate == false)
    return;

  if (n_coarse_iterations.empty())
    {
      pcout << "    [C] solved in 0 iterations." << std::endl;
      return;
    }

  pcout << "    [C] solved in [" << n_coarse_iterations[0];
  for (unsigned int i = 1; i < n_coarse_iterations.size(); ++i)
    pcout << " + " << n_coarse_iterations[i];
  pcout << "] iterations." << std::endl;

  n_coarse_iterations.clear();
}



template <int dim>
void
PreconditionerGMG<dim>::initialize()
{
  MyScope scope(timer, "gmg::initialize");

  pcout_cond << "    [M] initialize" << std::endl;

  const unsigned int min_level = transfer->min_level();
  const unsigned int max_level = transfer->max_level();

  // wrap level operators
  if (consider_edge_constraints)
    {
      op_ls.resize(min_level, max_level);
      for (unsigned int level = min_level; level <= max_level; level++)
        op_ls[level].initialize(*op[level]);
      mg_matrix = std::make_unique<mg::Matrix<VectorType<MGNumber>>>(op_ls);
    }
  else
    {
      mg_matrix = std::make_unique<mg::Matrix<VectorType<MGNumber>>>(op);
    }

  // create interface matrices
  if (consider_edge_constraints)
    {
      mg_interface_matrices.resize(min_level, max_level);
      for (unsigned int level = min_level; level <= max_level; ++level)
        mg_interface_matrices[level].initialize(*op[level]);
      mg_interface = std::make_unique<mg::Matrix<VectorType<MGNumber>>>(
        mg_interface_matrices);
    }

  // setup smoothers on each level
  MGLevelObject<typename SmootherType::AdditionalData> smoother_data(min_level,
                                                                     max_level);

  {
    MyScope scope(timer, "gmg::initialize::smoother::init0");

    for (unsigned int level = min_level; level <= max_level; ++level)
      {
        smoother_data[level].preconditioner =
          std::make_shared<SmootherPreconditionerType>();
        op[level]->compute_inverse_diagonal(
          smoother_data[level].preconditioner->get_vector());
        smoother_data[level].relaxation      = 0.0;
        smoother_data[level].smoothing_range = additional_data.smoothing_range;
        smoother_data[level].n_iterations =
          additional_data.smoothing_n_iterations;
        smoother_data[level].eig_cg_n_iterations =
          additional_data.smoothing_eig_cg_n_iterations;
        smoother_data[level].eigenvalue_algorithm =
          SmootherType::AdditionalData::EigenvalueAlgorithm::power_iteration;
        smoother_data[level].constraints.copy_from(
          op[level]->get_constraints());
      }
  }

  if (false)
    for (unsigned int level = min_level; level <= min_level; ++level)
      {
        const auto &matrix = op[level]->get_system_matrix();

        LAPACKFullMatrix<double> lapack_full_matrix;
        lapack_full_matrix.copy_from(matrix);
        lapack_full_matrix.compute_eigenvalues();

        std::vector<double> eigenvalues;

        for (unsigned int i = 0; i < lapack_full_matrix.m(); ++i)
          eigenvalues.push_back(lapack_full_matrix.eigenvalue(i).real());

        std::sort(eigenvalues.begin(), eigenvalues.end());

        std::cout << level << " " << eigenvalues.size() << " " << eigenvalues[0]
                  << " " << eigenvalues.back() << std::endl;

        if (false)
          {
            for (const auto i : eigenvalues)
              std::cout << i << " ";
            std::cout << std::endl;
          }
      }

  mg_smoother =
    std::make_unique<MGSmootherPrecondition<LevelMatrixType,
                                            SmootherType,
                                            VectorType<MGNumber>>>();
  mg_smoother->initialize(op, smoother_data);

  {
    MyScope scope(timer, "gmg::initialize::smoother::init1");
    for (unsigned int level = min_level; level <= max_level; ++level)
      {
        VectorType<MGNumber> vec;
        op[level]->initialize_dof_vector(vec);
        const auto ev = mg_smoother->smoothers[level].estimate_eigenvalues(vec);

        pcout_cond << "    [M]  - level: " << level << ", omega: "
                   << mg_smoother->smoothers[level].get_relaxation()
                   << ", ev_min: " << ev.min_eigenvalue_estimate
                   << ", ev_max: " << ev.max_eigenvalue_estimate << std::endl;
      }
  }

  if (additional_data.coarse_grid_solver == "AMG")
    {
      MyScope scope(timer, "gmg::initialize::amg");

      if (!additional_data.coarse_grid_amg_use_default_parameters)
        {
          TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;

          amg_data.elliptic = additional_data.coarse_grid_amg_elliptic;
          amg_data.higher_order_elements =
            additional_data.coarse_grid_amg_higher_order_elements;
          amg_data.n_cycles = additional_data.coarse_grid_amg_n_cycles;
          amg_data.aggregation_threshold =
            additional_data.coarse_grid_amg_aggregation_threshold;
          amg_data.smoother_sweeps =
            additional_data.coarse_grid_amg_smoother_sweeps;
          amg_data.smoother_overlap =
            additional_data.coarse_grid_amg_smoother_overlap;
          amg_data.output_details = additional_data.output_details;
          amg_data.smoother_type =
            additional_data.coarse_grid_amg_smoother_type.c_str();
          amg_data.coarse_type =
            additional_data.coarse_grid_amg_coarse_type.c_str();
          amg_data.constant_modes = op[min_level]->extract_constant_modes();

          Teuchos::ParameterList              parameter_ml;
          std::unique_ptr<Epetra_MultiVector> distributed_constant_modes;
          amg_data.set_parameters(parameter_ml,
                                  distributed_constant_modes,
                                  op[min_level]->get_system_matrix());

          const double ilu_fill = 1.0;
          const double ilu_atol = 1e-6;
          const double ilu_rtol = 1.0;

          parameter_ml.set("smoother: ifpack level-of-fill", ilu_fill);
          parameter_ml.set("smoother: ifpack absolute threshold", ilu_atol);
          parameter_ml.set("smoother: ifpack relative threshold", ilu_rtol);

          parameter_ml.set("coarse: ifpack level-of-fill", ilu_fill);
          parameter_ml.set("coarse: ifpack absolute threshold", ilu_atol);
          parameter_ml.set("coarse: ifpack relative threshold", ilu_rtol);

          precondition_amg =
            std::make_unique<TrilinosWrappers::PreconditionAMG>();

          const auto &matrix = op[min_level]->get_system_matrix();

          precondition_amg->initialize(matrix, parameter_ml);
        }
      else
        {
          TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;

          precondition_amg =
            std::make_unique<TrilinosWrappers::PreconditionAMG>();

          const auto &matrix = op[min_level]->get_system_matrix();

          precondition_amg->initialize(matrix, amg_data);
        }
    }
  else if (additional_data.coarse_grid_solver == "direct")
    {
      MyScope scope(timer, "gmg::initialize::direct");

      precondition_direct = std::make_unique<TrilinosWrappers::SolverDirect>();

      precondition_direct->initialize(op[min_level]->get_system_matrix());
    }
  else if (additional_data.coarse_grid_solver == "identity")
    {
      precondition_identity = std::make_unique<PreconditionIdentity>();
    }
  else
    {
      AssertThrow(false, ExcInternalError());
    }

  if (!additional_data.coarse_grid_iterate)
    {
      if (additional_data.coarse_grid_solver == "AMG")
        mg_coarse = std::make_unique<
          MGCoarseGridApplyPreconditioner<VectorType<MGNumber>,
                                          TrilinosWrappers::PreconditionAMG>>(
          *precondition_amg);
      else if (additional_data.coarse_grid_solver == "direct")
        mg_coarse = std::make_unique<
          MGCoarseGridApplyPreconditioner<VectorType<MGNumber>,
                                          TrilinosWrappers::SolverDirect>>(
          *precondition_direct);
      else if (additional_data.coarse_grid_solver == "identity")
        mg_coarse = std::make_unique<
          MGCoarseGridApplyPreconditioner<VectorType<MGNumber>,
                                          PreconditionIdentity>>(
          *precondition_identity);
      else
        AssertThrow(false, ExcInternalError());
    }
  else
    {
      AssertThrow(min_level != max_level, ExcInternalError());

      coarse_grid_solver_control = std::make_unique<ReductionControl>(
        additional_data.coarse_grid_gmres_maxiter,
        additional_data.coarse_grid_gmres_abstol,
        additional_data.coarse_grid_gmres_reltol,
        false,
        false);

      coarse_grid_solver = std::make_unique<SolverGMRES<VectorType<Number>>>(
        *coarse_grid_solver_control);

      if (false)
        coarse_grid_solver->connect(
          [](const auto i, const auto v, const auto &) -> SolverControl::State {
            std::cout << i << ": " << v << std::endl;

            return SolverControl::State::success;
          });

      mg_coarse = std::make_unique<
        MGCoarseGridIterativeSolver<VectorType<MGNumber>,
                                    SolverGMRES<VectorType<Number>>,
                                    TrilinosWrappers::SparseMatrix,
                                    TrilinosWrappers::PreconditionAMG>>(
        *coarse_grid_solver,
        op[min_level]->get_system_matrix(),
        *precondition_amg);
    }

  mg = std::make_unique<Multigrid<VectorType<MGNumber>>>(*mg_matrix,
                                                         *mg_coarse,
                                                         *transfer,
                                                         *mg_smoother,
                                                         *mg_smoother,
                                                         min_level,
                                                         max_level);

  if (consider_edge_constraints &&
      dof_handler.get_triangulation().has_hanging_nodes())
    mg->set_edge_in_matrix(*mg_interface);

  preconditioner =
    std::make_unique<PreconditionMG<dim, VectorType<MGNumber>, MGTransferType>>(
      dof_handler, *mg, *transfer);

  const auto create_mg_timer_function = [&](const std::string &label) {
    return [label, this](const bool flag, const unsigned int level) {
      const std::string label_full =
        (label == "") ?
          ("gmg::vmult::level_" + std::to_string(level)) :
          ("gmg::vmult::level_" + std::to_string(level) + "::" + label);

      if (flag)
        this->timer.enter_subsection(label_full);
      else
        this->timer.leave_subsection(label_full);
    };
  };

  mg->connect_pre_smoother_step(
    create_mg_timer_function("0_pre_smoother_step"));
  mg->connect_residual_step(create_mg_timer_function("1_residual_step"));
  mg->connect_restriction(create_mg_timer_function("2_restriction"));
  mg->connect_coarse_solve(create_mg_timer_function(""));
  mg->connect_prolongation(create_mg_timer_function("3_prolongation"));
  mg->connect_edge_prolongation(
    create_mg_timer_function("4_edge_prolongation"));
  mg->connect_post_smoother_step(
    create_mg_timer_function("5_post_smoother_step"));


  const auto create_mg_precon_timer_function = [&](const std::string &label) {
    return [label, this](const bool flag) {
      const std::string label_full = "gmg::vmult::" + label;

      if (flag)
        this->timer.enter_subsection(label_full);
      else
        this->timer.leave_subsection(label_full);
    };
  };

  preconditioner->connect_transfer_to_mg(
    create_mg_precon_timer_function("transfer_to_mg"));
  preconditioner->connect_transfer_to_global(
    create_mg_precon_timer_function("transfer_to_global"));

  pcout_cond << std::endl;
}

template class PreconditionerGMG<2>;
template class PreconditionerGMG<3>;
