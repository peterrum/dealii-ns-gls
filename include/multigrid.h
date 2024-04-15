#pragma once

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_solver.h>

#include <deal.II/matrix_free/operators.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/multigrid.h>

#include "config.h"
#include "preconditioner.h"
#include "timer.h"


struct PreconditionerGMGAdditionalData
{
  bool output_details = false;

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
  bool         coarse_grid_amg_use_default_parameters = true;
  bool         coarse_grid_amg_elliptic               = false;
  bool         coarse_grid_amg_higher_order_elements  = false;
  unsigned int coarse_grid_amg_n_cycles               = 1;
  double       coarse_grid_amg_aggregation_threshold  = 1e-14;
  unsigned int coarse_grid_amg_smoother_sweeps        = 2;
  unsigned int coarse_grid_amg_smoother_overlap       = 1;
  bool         coarse_grid_amg_output_details         = false;
  std::string  coarse_grid_amg_smoother_type          = "ILU";
  std::string  coarse_grid_amg_coarse_type            = "ILU";

  void
  add_parameters(ParameterHandler &prm);
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
    const DoFHandler<dim>                              &dof_handler,
    const MGLevelObject<std::shared_ptr<OperatorBase>> &op,
    const std::shared_ptr<MGTransferGlobalCoarsening<dim, VectorType>>
                                          &transfer,
    const bool                             consider_edge_constraints,
    const PreconditionerGMGAdditionalData &additional_data);

  void
  vmult(VectorType &dst, const VectorType &src) const override;

  void
  print_stats() const override;

  void
  initialize() override;

private:
  const ConditionalOStream pcout;
  const ConditionalOStream pcout_cond;

  const DoFHandler<dim>                              &dof_handler;
  const MGLevelObject<std::shared_ptr<OperatorBase>> &op;
  const std::shared_ptr<MGTransferType>               transfer;

  const bool consider_edge_constraints;

  const PreconditionerGMGAdditionalData additional_data;

  mutable MGLevelObject<
    MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType>>
    op_ls;

  mutable std::unique_ptr<mg::Matrix<VectorType>> mg_matrix;

  mutable MGLevelObject<
    MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType>>
    mg_interface_matrices;

  mutable std::unique_ptr<mg::Matrix<VectorType>> mg_interface;

  mutable std::unique_ptr<
    MGSmootherPrecondition<LevelMatrixType, SmootherType, VectorType>>
    mg_smoother;

  mutable std::unique_ptr<TrilinosWrappers::PreconditionAMG> precondition_amg;

  mutable std::unique_ptr<TrilinosWrappers::SolverDirect> precondition_direct;

  mutable std::unique_ptr<PreconditionIdentity> precondition_identity;

  mutable std::unique_ptr<SolverControl> coarse_grid_solver_control;

  mutable std::unique_ptr<SolverGMRES<VectorType>> coarse_grid_solver;

  mutable std::unique_ptr<MGCoarseGridBase<VectorType>> mg_coarse;

  mutable std::unique_ptr<Multigrid<VectorType>> mg;

  mutable std::unique_ptr<PreconditionMG<dim, VectorType, MGTransferType>>
    preconditioner;

  mutable std::vector<unsigned int> n_coarse_iterations;

  mutable MyTimerOutput timer;
};
