#pragma once

#include <deal.II/lac/precondition.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/multigrid.h>

// TODO: make part of deal.II
template <typename MatrixType, typename VectorType, typename PreconditionerType>
class MyPreconditionRelaxation : public Subscriptor
{
public:
  using size_type = types::global_dof_index;

  class AdditionalData
  {
  public:
    enum class EigenvalueAlgorithm
    {
      lanczos,
      power_iteration
    };

    AdditionalData(const unsigned int        degree              = 1,
                   const double              smoothing_range     = 0.,
                   const unsigned int        eig_cg_n_iterations = 8,
                   const double              eig_cg_residual     = 1e-2,
                   const double              max_eigenvalue      = 1,
                   const EigenvalueAlgorithm eigenvalue_algorithm =
                     EigenvalueAlgorithm::lanczos)
      : degree(degree)
      , smoothing_range(smoothing_range)
      , eig_cg_n_iterations(eig_cg_n_iterations)
      , eig_cg_residual(eig_cg_residual)
      , max_eigenvalue(max_eigenvalue)
      , eigenvalue_algorithm(eigenvalue_algorithm)
    {}

    unsigned int degree;

    double smoothing_range;

    unsigned int eig_cg_n_iterations;

    double eig_cg_residual;

    double max_eigenvalue;

    AffineConstraints<double> constraints;

    std::shared_ptr<PreconditionerType> preconditioner;

    EigenvalueAlgorithm eigenvalue_algorithm;
  };

  void
  initialize(const MatrixType     &A,
             const AdditionalData &parameters = AdditionalData())
  {
    // TODO: move eigenvalue estimation to first usage

    typename PreconditionChebyshev<MatrixType, VectorType, PreconditionerType>::
      AdditionalData parameters_chebyshev;

    PreconditionChebyshev<MatrixType, VectorType, PreconditionerType> chebyshev;

    chebyshev.initialize(A, parameters_chebyshev);

    VectorType vec;
    A.initialize_dof_vector(vec);

    const auto evs = chebyshev.estimate_eigenvalues(vec);

    const double alpha =
      (parameters.smoothing_range > 1. ?
         evs.max_eigenvalue_estimate / parameters.smoothing_range :
         std::min(0.9 * evs.max_eigenvalue_estimate,
                  evs.min_eigenvalue_estimate));

    const double omega = 2.0 / (alpha + evs.max_eigenvalue_estimate);

    typename PreconditionRelaxation<MatrixType,
                                    PreconditionerType>::AdditionalData
      parameters_relaxation;

    parameters_relaxation.relaxation   = omega;
    parameters_relaxation.n_iterations = parameters.degree;

    relaxation.initialize(A, parameters_relaxation);
  }

  void
  clear()
  {
    relaxation.clear();
  }

  size_type
  m() const
  {
    return relaxation.m();
  }

  size_type
  n() const
  {
    return relaxation.n();
  }

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    relaxation.vmult(dst, src);
  }

  void
  Tvmult(VectorType &dst, const VectorType &src) const
  {
    relaxation.Tvmult(dst, src);
  }

  void
  step(VectorType &dst, const VectorType &src) const
  {
    relaxation.step(dst, src);
  }

  void
  Tstep(VectorType &dst, const VectorType &src) const
  {
    relaxation.Tstep(dst, src);
  }

protected:
  PreconditionRelaxation<MatrixType, PreconditionerType> relaxation;
};


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
  using SmootherType               = MyPreconditionRelaxation<LevelMatrixType,
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

    if (additional_data.coarse_grid_type == "AMG")
      {
        TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
        amg_data.smoother_sweeps = additional_data.coarse_grid_smoother_sweeps;
        amg_data.n_cycles        = additional_data.coarse_grid_n_cycles;
        amg_data.smoother_type =
          additional_data.coarse_grid_smoother_type.c_str();
        amg_data.output_details = false;

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