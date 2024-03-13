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



template <int dim>
class PreconditionerGMG : public PreconditionerBase
{
public:
  struct PreconditionerGMGAdditionalData
  {
    double       smoothing_range               = 20;
    unsigned int smoothing_degree              = 1;
    unsigned int smoothing_eig_cg_n_iterations = 20;

    unsigned int coarse_grid_smoother_sweeps = 1;
    unsigned int coarse_grid_n_cycles        = 1;
    std::string  coarse_grid_smoother_type   = "ILU";

    std::string coarse_grid_type = "AMG";
  };

  using LevelMatrixType = OperatorBase;

  using SmootherPreconditionerType = DiagonalMatrix<VectorType>;
  using SmootherType =
    PreconditionRelaxation<LevelMatrixType, SmootherPreconditionerType>;

  using MGTransferType = MGTransferGlobalCoarsening<dim, VectorType>;

  PreconditionerGMG(
    const MGLevelObject<std::shared_ptr<OperatorBase>> &op,
    const std::shared_ptr<MGTransferGlobalCoarsening<dim, VectorType>>
      &transfer)
    : op(op)
    , transfer(transfer)
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
    PreconditionerGMGAdditionalData additional_data; // TODO

    const unsigned int min_level = transfer->min_level();
    const unsigned int max_level = transfer->max_level();

    MGLevelObject<std::shared_ptr<OperatorBase>> op(min_level, max_level);

    for (unsigned int l = min_level; l <= max_level; ++l)
      op[l] = this->op[l];

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
        smoother_data[level].n_iterations    = additional_data.smoothing_degree;
        smoother_data[level].eig_cg_n_iterations =
          additional_data.smoothing_eig_cg_n_iterations;
        smoother_data[level].constraints.copy_from(
          op[level]->get_constraints());
      }

    mg_smoother = std::make_unique<
      MGSmootherPrecondition<LevelMatrixType, SmootherType, VectorType>>();
    mg_smoother->initialize(op, smoother_data);

    if (additional_data.coarse_grid_type == "AMG")
      {
        TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
        amg_data.smoother_sweeps = additional_data.coarse_grid_smoother_sweeps;
        amg_data.n_cycles        = additional_data.coarse_grid_n_cycles;
        amg_data.smoother_type =
          additional_data.coarse_grid_smoother_type.c_str();
        amg_data.output_details = false;

        precondition_amg =
          std::make_unique<TrilinosWrappers::PreconditionAMG>();
        precondition_amg->initialize(op[min_level]->get_system_matrix());
        mg_coarse = std::make_unique<
          MGCoarseGridApplyPreconditioner<VectorType,
                                          TrilinosWrappers::PreconditionAMG>>(
          *precondition_amg);
      }
    else
      {
        AssertThrow(false, ExcInternalError());
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
  }

private:
  const MGLevelObject<std::shared_ptr<OperatorBase>> &op;
  const std::shared_ptr<MGTransferType>               transfer;

  DoFHandler<dim> dof_handler_dummy;

  mutable std::unique_ptr<mg::Matrix<VectorType>> mg_matrix;

  mutable std::unique_ptr<
    MGSmootherPrecondition<LevelMatrixType, SmootherType, VectorType>>
    mg_smoother;

  mutable std::unique_ptr<TrilinosWrappers::PreconditionAMG> precondition_amg;

  mutable std::unique_ptr<MGCoarseGridBase<VectorType>> mg_coarse;

  mutable std::unique_ptr<Multigrid<VectorType>> mg;

  mutable std::unique_ptr<PreconditionMG<dim, VectorType, MGTransferType>>
    preconditioner;
};