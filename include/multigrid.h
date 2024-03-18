#pragma once

#include <deal.II/lac/precondition.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/multigrid.h>


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



struct PreconditionerGMGAdditionalData
{
  // smoother (relaxation + point Jacobi)
  double       smoothing_range               = 20;
  unsigned int smoothing_n_iterations        = 5;
  unsigned int smoothing_eig_cg_n_iterations = 20;

  // coarse-grid solver type
  std::string coarse_grid_solver  = "AMG";
  bool        coarse_grid_iterate = true;

  // coarse-grid GMRES
  unsigned int coarse_grid_gmres_maxiter = 10000;
  double       coarse_grid_gmres_abstol  = 1e-20;
  double       coarse_grid_gmres_reltol  = 1e-4;

  // coarse-grid AMG
  bool         coarse_grid_amg_elliptic              = false;
  bool         coarse_grid_amg_higher_order_elements = false;
  unsigned int coarse_grid_amg_n_cycles              = 1;
  double       coarse_grid_amg_aggregation_threshold = 1e-14;
  unsigned int coarse_grid_amg_smoother_sweeps       = 2;
  unsigned int coarse_grid_amg_smoother_overlap      = 1;
  bool         coarse_grid_amg_output_details        = false;
  std::string  coarse_grid_amg_smoother_type         = "ILU";
  std::string  coarse_grid_amg_coarse_type           = "ILU";

  void
  add_parameters(ParameterHandler &prm)
  {
    // smoother
    prm.add_parameter("gmg smoothing n iterations", smoothing_n_iterations);

    // coarse-grid solver
    prm.add_parameter("gmg coarse grid solver",
                      coarse_grid_solver,
                      "",
                      Patterns::Selection("AMG"));
    prm.add_parameter("gmg coarse grid iterate", coarse_grid_iterate);

    // coarse-grid GMRES
    prm.add_parameter("gmg coarse grid gmres reltol", coarse_grid_gmres_reltol);

    // coarse-grid AMG
    // TODO
  }
};



template <int dim>
class PreconditionerGMG : public PreconditionerBase
{
public:
  using LevelMatrixType = OperatorBase;

  using SmootherPreconditionerType = DiagonalMatrix<VectorType>;
  using SmootherType =
    PreconditionRelaxation<LevelMatrixType, SmootherPreconditionerType>;

  using MGTransferType = MGTransferGlobalCoarsening<dim, VectorType>;

  PreconditionerGMG(
    const MGLevelObject<std::shared_ptr<OperatorBase>> &op,
    const std::shared_ptr<MGTransferGlobalCoarsening<dim, VectorType>>
                                          &transfer,
    const PreconditionerGMGAdditionalData &additional_data)
    : pcout(std::cout,
            (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) &&
              false /*TODO: introduce verbosity*/)
    , op(op)
    , transfer(transfer)
    , additional_data(additional_data)
  {}

  void
  vmult(VectorType &dst, const VectorType &src) const override
  {
    Assert(preconditioner, ExcInternalError());
    preconditioner->vmult(dst, src);
  }

  void
  initialize() override
  {
    pcout << "    [M] initialize" << std::endl;

    const unsigned int min_level = transfer->min_level();
    const unsigned int max_level = transfer->max_level();

    // wrap level operators
    mg_matrix = std::make_unique<mg::Matrix<VectorType>>(op);

    // setup smoothers on each level
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data(
      min_level, max_level);

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

    mg_smoother = std::make_unique<
      MGSmootherPrecondition<LevelMatrixType, SmootherType, VectorType>>();
    mg_smoother->initialize(op, smoother_data);

    for (unsigned int level = min_level; level <= max_level; ++level)
      {
        VectorType vec;
        op[level]->initialize_dof_vector(vec);
        const auto ev = mg_smoother->smoothers[level].estimate_eigenvalues(vec);

        pcout << "    [M]  - level: " << level
              << ", omega: " << mg_smoother->smoothers[level].get_relaxation()
              << ", ev_min: " << ev.min_eigenvalue_estimate
              << ", ev_max: " << ev.max_eigenvalue_estimate << std::endl;
      }

    if (additional_data.coarse_grid_solver == "AMG")
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
        amg_data.output_details =
          additional_data.coarse_grid_amg_output_details;
        amg_data.smoother_type =
          additional_data.coarse_grid_amg_smoother_type.c_str();
        amg_data.coarse_type =
          additional_data.coarse_grid_amg_coarse_type.c_str();
        amg_data.constant_modes = op[min_level]->extract_constant_modes();

        precondition_amg =
          std::make_unique<TrilinosWrappers::PreconditionAMG>();
        precondition_amg->initialize(op[min_level]->get_system_matrix(),
                                     amg_data);
      }
    else
      {
        AssertThrow(false, ExcInternalError());
      }

    if (!additional_data.coarse_grid_iterate)
      {
        mg_coarse = std::make_unique<
          MGCoarseGridApplyPreconditioner<VectorType,
                                          TrilinosWrappers::PreconditionAMG>>(
          *precondition_amg);
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

        coarse_grid_solver = std::make_unique<SolverGMRES<VectorType>>(
          *coarse_grid_solver_control);

        mg_coarse = std::make_unique<
          MGCoarseGridIterativeSolver<VectorType,
                                      SolverGMRES<VectorType>,
                                      OperatorBase,
                                      TrilinosWrappers::PreconditionAMG>>(
          *coarse_grid_solver, *op[min_level], *precondition_amg);
      }

    mg = std::make_unique<Multigrid<VectorType>>(*mg_matrix,
                                                 *mg_coarse,
                                                 *transfer,
                                                 *mg_smoother,
                                                 *mg_smoother,
                                                 min_level,
                                                 max_level);

    preconditioner =
      std::make_unique<PreconditionMG<dim, VectorType, MGTransferType>>(
        dof_handler_dummy, *mg, *transfer);

    pcout << std::endl;
  }

private:
  const ConditionalOStream pcout;

  const MGLevelObject<std::shared_ptr<OperatorBase>> &op;
  const std::shared_ptr<MGTransferType>               transfer;

  const PreconditionerGMGAdditionalData additional_data;

  DoFHandler<dim> dof_handler_dummy;

  mutable std::unique_ptr<mg::Matrix<VectorType>> mg_matrix;

  mutable std::unique_ptr<
    MGSmootherPrecondition<LevelMatrixType, SmootherType, VectorType>>
    mg_smoother;

  mutable std::unique_ptr<TrilinosWrappers::PreconditionAMG> precondition_amg;

  mutable std::unique_ptr<SolverControl> coarse_grid_solver_control;

  mutable std::unique_ptr<SolverGMRES<VectorType>> coarse_grid_solver;

  mutable std::unique_ptr<MGCoarseGridBase<VectorType>> mg_coarse;

  mutable std::unique_ptr<Multigrid<VectorType>> mg;

  mutable std::unique_ptr<PreconditionMG<dim, VectorType, MGTransferType>>
    preconditioner;
};