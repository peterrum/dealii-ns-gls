#pragma once

#include <deal.II/lac/precondition.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/multigrid.h>

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

    unsigned int coarse_grid_maxiter = 1000;
    double       coarse_grid_abstol  = 1e-20;
    double       coarse_grid_reltol  = 1e-4;
    std::string  coarse_grid_type    = "cg_with_chebyshev";
  };

  using LevelMatrixType = OperatorBase;

  using SmootherPreconditionerType = DiagonalMatrix<VectorType>;
  using SmootherType               = PreconditionChebyshev<LevelMatrixType,
                                             VectorType,
                                             SmootherPreconditionerType>;

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
        smoother_data[level].smoothing_range = additional_data.smoothing_range;
        smoother_data[level].degree          = additional_data.smoothing_degree;
        smoother_data[level].eig_cg_n_iterations =
          additional_data.smoothing_eig_cg_n_iterations;
      }

    mg_smoother = std::make_unique<
      MGSmootherPrecondition<LevelMatrixType, SmootherType, VectorType>>();
    mg_smoother->initialize(op, smoother_data);

    for (unsigned int level = min_level; level <= max_level; ++level)
      {
        VectorType vec;
        op[level]->initialize_dof_vector(vec);
        mg_smoother->smoothers[level].estimate_eigenvalues(vec);
      }

    coarse_grid_solver_control =
      std::make_unique<ReductionControl>(additional_data.coarse_grid_maxiter,
                                         additional_data.coarse_grid_abstol,
                                         additional_data.coarse_grid_reltol,
                                         false,
                                         false);
    coarse_grid_solver =
      std::make_unique<SolverCG<VectorType>>(*coarse_grid_solver_control);

    if (additional_data.coarse_grid_type == "cg_with_chebyshev")
      {
        typename SmootherType::AdditionalData smoother_data;

        smoother_data.preconditioner =
          std::make_shared<DiagonalMatrix<VectorType>>();
        op[min_level]->compute_inverse_diagonal(
          smoother_data.preconditioner->get_vector());
        smoother_data.smoothing_range = additional_data.smoothing_range;
        smoother_data.degree          = additional_data.smoothing_degree;
        smoother_data.eig_cg_n_iterations =
          additional_data.smoothing_eig_cg_n_iterations;

        precondition_chebyshev =
          std::make_unique<PreconditionChebyshev<LevelMatrixType,
                                                 VectorType,
                                                 DiagonalMatrix<VectorType>>>();

        precondition_chebyshev->initialize(*op[min_level], smoother_data);

        mg_coarse = std::make_unique<MGCoarseGridIterativeSolver<
          VectorType,
          SolverCG<VectorType>,
          LevelMatrixType,
          PreconditionChebyshev<LevelMatrixType,
                                VectorType,
                                DiagonalMatrix<VectorType>>>>(
          *coarse_grid_solver, *op[min_level], *precondition_chebyshev);
      }
    else
      {
        AssertThrow(false, ExcNotImplemented());
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

  mutable std::unique_ptr<ReductionControl>     coarse_grid_solver_control;
  mutable std::unique_ptr<SolverCG<VectorType>> coarse_grid_solver;

  mutable std::unique_ptr<PreconditionChebyshev<LevelMatrixType,
                                                VectorType,
                                                DiagonalMatrix<VectorType>>>
    precondition_chebyshev;

  mutable std::unique_ptr<MGCoarseGridBase<VectorType>> mg_coarse;

  mutable std::unique_ptr<Multigrid<VectorType>> mg;

  mutable std::unique_ptr<PreconditionMG<dim, VectorType, MGTransferType>>
    preconditioner;
};