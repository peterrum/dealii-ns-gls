#pragma once

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

using Number           = double;
using VectorType       = dealii::LinearAlgebra::distributed::Vector<Number>;
using SparseMatrixType = dealii::TrilinosWrappers::SparseMatrix;
