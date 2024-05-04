#include <deal.II/base/config.h>

#include <deal.II/grid/tria.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/vector_access_internal.h>


namespace dealii
{
  namespace MyMatrixFreeTools
  {
    template <int dim,
              int fe_degree,
              int n_q_points_1d,
              int n_components,
              typename Number,
              typename VectorizedArrayType,
              typename MatrixType>
    void
    compute_matrix(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      const AffineConstraints<Number>                    &constraints,
      MatrixType                                         &matrix,
      const std::function<void(FEEvaluation<dim,
                                            fe_degree,
                                            n_q_points_1d,
                                            n_components,
                                            Number,
                                            VectorizedArrayType> &)>
                        &local_vmult,
      const unsigned int dof_no                   = 0,
      const unsigned int quad_no                  = 0,
      const unsigned int first_selected_component = 0);

    template <typename CLASS,
              int dim,
              int fe_degree,
              int n_q_points_1d,
              int n_components,
              typename Number,
              typename VectorizedArrayType,
              typename MatrixType>
    void
    compute_matrix(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      const AffineConstraints<Number>                    &constraints,
      MatrixType                                         &matrix,
      void (CLASS::*cell_operation)(FEEvaluation<dim,
                                                 fe_degree,
                                                 n_q_points_1d,
                                                 n_components,
                                                 Number,
                                                 VectorizedArrayType> &) const,
      const CLASS       *owning_class,
      const unsigned int dof_no                   = 0,
      const unsigned int quad_no                  = 0,
      const unsigned int first_selected_component = 0);

    // implementations

#ifndef DOXYGEN

    namespace internal
    {
      /**
       * If value type of matrix and constrains equals, return a reference
       * to the given AffineConstraint instance.
       */
      template <typename MatrixType,
                typename Number,
                std::enable_if_t<std::is_same_v<
                  typename std::remove_const<typename std::remove_reference<
                    typename MatrixType::value_type>::type>::type,
                  typename std::remove_const<typename std::remove_reference<
                    Number>::type>::type>> * = nullptr>
      const AffineConstraints<typename MatrixType::value_type> &
      create_new_affine_constraints_if_needed(
        const MatrixType &,
        const AffineConstraints<Number> &constraints,
        std::unique_ptr<AffineConstraints<typename MatrixType::value_type>> &)
      {
        return constraints;
      }

      /**
       * If value type of matrix and constrains do not equal, a new
       * AffineConstraint instance with the value type of the matrix is
       * created and a reference to it is returned.
       */
      template <typename MatrixType,
                typename Number,
                std::enable_if_t<!std::is_same_v<
                  typename std::remove_const<typename std::remove_reference<
                    typename MatrixType::value_type>::type>::type,
                  typename std::remove_const<typename std::remove_reference<
                    Number>::type>::type>> * = nullptr>
      const AffineConstraints<typename MatrixType::value_type> &
      create_new_affine_constraints_if_needed(
        const MatrixType &,
        const AffineConstraints<Number> &constraints,
        std::unique_ptr<AffineConstraints<typename MatrixType::value_type>>
          &new_constraints)
      {
        new_constraints = std::make_unique<
          AffineConstraints<typename MatrixType::value_type>>();
        new_constraints->copy_from(constraints);

        return *new_constraints;
      }
    } // namespace internal

    template <int dim,
              int fe_degree,
              int n_q_points_1d,
              int n_components,
              typename Number,
              typename VectorizedArrayType,
              typename MatrixType>
    void
    compute_matrix(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      const AffineConstraints<Number>                    &constraints_in,
      MatrixType                                         &matrix,
      const std::function<void(FEEvaluation<dim,
                                            fe_degree,
                                            n_q_points_1d,
                                            n_components,
                                            Number,
                                            VectorizedArrayType> &)>
                        &local_vmult,
      const unsigned int dof_no,
      const unsigned int quad_no,
      const unsigned int first_selected_component)
    {
      const auto &quadrature = matrix_free.get_quadrature(quad_no);
      const auto &fe         = matrix_free.get_dof_handler(dof_no).get_fe();

      const auto compute_scalar_bool_dof_mask = [&quadrature](const auto &fe) {
        Table<2, bool> bool_dof_mask(fe.dofs_per_cell, fe.dofs_per_cell);
        MappingQ1<dim> mapping;
        FEValues<dim>  fe_values(mapping, fe, quadrature, update_values);

        Triangulation<dim> tria;
        GridGenerator::hyper_cube(tria);

        fe_values.reinit(tria.begin());
        for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
          for (unsigned int j = 0; j < fe.dofs_per_cell; ++j)
            {
              double sum = 0;
              for (unsigned int q = 0; q < quadrature.size(); ++q)
                sum +=
                  fe_values.shape_value(i, q) * fe_values.shape_value(j, q);
              if (sum != 0)
                bool_dof_mask(i, j) = true;
            }

        return bool_dof_mask;
      };

      Table<2, bool> bool_dof_mask(fe.dofs_per_cell, fe.dofs_per_cell);

      AssertDimension(n_components, fe.n_components());

      if (fe.n_components() == 1)
        {
          bool_dof_mask = compute_scalar_bool_dof_mask(fe);
        }
      else
        {
          const auto scalar_bool_dof_mask =
            compute_scalar_bool_dof_mask(fe.base_element(0));

          for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
            for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
              if (scalar_bool_dof_mask[fe.system_to_component_index(i).second]
                                      [fe.system_to_component_index(j).second])
                bool_dof_mask[i][j] = true;
        }

      std::unique_ptr<AffineConstraints<typename MatrixType::value_type>>
        constraints_for_matrix;
      const AffineConstraints<typename MatrixType::value_type> &constraints =
        internal::create_new_affine_constraints_if_needed(
          matrix, constraints_in, constraints_for_matrix);

      matrix_free.template cell_loop<MatrixType, MatrixType>(
        [&](const auto &, auto &dst, const auto &, const auto range) {
          FEEvaluation<dim,
                       fe_degree,
                       n_q_points_1d,
                       n_components,
                       Number,
                       VectorizedArrayType>
            integrator(
              matrix_free, range, dof_no, quad_no, first_selected_component);

          const unsigned int dofs_per_cell = integrator.dofs_per_cell;

          std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
          std::vector<types::global_dof_index> dof_indices_mf(dofs_per_cell);

          std::array<FullMatrix<typename MatrixType::value_type>,
                     VectorizedArrayType::size()>
            matrices;

          std::fill_n(matrices.begin(),
                      VectorizedArrayType::size(),
                      FullMatrix<typename MatrixType::value_type>(
                        dofs_per_cell, dofs_per_cell));

          const auto lexicographic_numbering =
            matrix_free
              .get_shape_info(dof_no,
                              quad_no,
                              first_selected_component,
                              integrator.get_active_fe_index(),
                              integrator.get_active_quadrature_index())
              .lexicographic_numbering;

          for (auto cell = range.first; cell < range.second; ++cell)
            {
              integrator.reinit(cell);

              const unsigned int n_filled_lanes =
                matrix_free.n_active_entries_per_cell_batch(cell);

              for (unsigned int v = 0; v < n_filled_lanes; ++v)
                matrices[v] = 0.0;

              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    integrator.begin_dof_values()[i] =
                      static_cast<Number>(i == j);

                  local_vmult(integrator);

                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    for (unsigned int v = 0; v < n_filled_lanes; ++v)
                      matrices[v](i, j) = integrator.begin_dof_values()[i][v];
                }

              for (unsigned int v = 0; v < n_filled_lanes; ++v)
                {
                  const auto cell_v =
                    matrix_free.get_cell_iterator(cell, v, dof_no);

                  if (matrix_free.get_mg_level() !=
                      numbers::invalid_unsigned_int)
                    cell_v->get_mg_dof_indices(dof_indices);
                  else
                    cell_v->get_dof_indices(dof_indices);

                  for (unsigned int j = 0; j < dof_indices.size(); ++j)
                    dof_indices_mf[j] = dof_indices[lexicographic_numbering[j]];

                  // new: remove small entries (TODO: only for FE_Q_iso_1)
                  for (unsigned int i = 0; i < matrices[v].m(); ++i)
                    for (unsigned int j = 0; j < matrices[v].n(); ++j)
                      if (!bool_dof_mask[lexicographic_numbering[i]]
                                        [lexicographic_numbering[j]])
                        matrices[v][i][j] = 0.0;

                  constraints.distribute_local_to_global(matrices[v],
                                                         dof_indices_mf,
                                                         dst);
                }
            }
        },
        matrix,
        matrix);

      matrix.compress(VectorOperation::add);
    }

    template <typename CLASS,
              int dim,
              int fe_degree,
              int n_q_points_1d,
              int n_components,
              typename Number,
              typename VectorizedArrayType,
              typename MatrixType>
    void
    compute_matrix(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      const AffineConstraints<Number>                    &constraints,
      MatrixType                                         &matrix,
      void (CLASS::*cell_operation)(FEEvaluation<dim,
                                                 fe_degree,
                                                 n_q_points_1d,
                                                 n_components,
                                                 Number,
                                                 VectorizedArrayType> &) const,
      const CLASS       *owning_class,
      const unsigned int dof_no,
      const unsigned int quad_no,
      const unsigned int first_selected_component)
    {
      compute_matrix<dim,
                     fe_degree,
                     n_q_points_1d,
                     n_components,
                     Number,
                     VectorizedArrayType,
                     MatrixType>(
        matrix_free,
        constraints,
        matrix,
        [&](auto &feeval) { (owning_class->*cell_operation)(feeval); },
        dof_no,
        quad_no,
        first_selected_component);
    }

#endif // DOXYGEN

  } // namespace MyMatrixFreeTools
} // namespace dealii
