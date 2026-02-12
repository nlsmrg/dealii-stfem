#pragma once
#include <deal.II/lac/sparse_matrix_tools.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>

#include <tuple>

#include "types.h"

namespace dealii
{
  namespace SparseMatrixTools
  {
    namespace internal
    {
      template <typename SparseMatrixType>
      std::pair<IndexSet, IndexSet>
      get_cache(
        const SparseMatrixType                                  &system_matrix,
        const std::vector<std::vector<types::global_dof_index>> &row_indices)
      { // 0) determine which rows are locally owned and which ones are remote
        const auto local_size = internal::get_local_size(system_matrix);
        const auto prefix_sum = Utilities::MPI::partial_and_total_sum(
          local_size, internal::get_mpi_communicator(system_matrix));
        IndexSet locally_owned_dofs(std::get<1>(prefix_sum));
        locally_owned_dofs.add_range(std::get<0>(prefix_sum),
                                     std::get<0>(prefix_sum) + local_size);

        std::vector<dealii::types::global_dof_index> ghost_indices_vector;

        for (const auto &i : row_indices)
          ghost_indices_vector.insert(ghost_indices_vector.end(),
                                      i.begin(),
                                      i.end());

        std::sort(ghost_indices_vector.begin(), ghost_indices_vector.end());

        IndexSet locally_active_dofs(std::get<1>(prefix_sum));
        locally_active_dofs.add_indices(ghost_indices_vector.begin(),
                                        ghost_indices_vector.end());

        locally_active_dofs.subtract_set(locally_owned_dofs);


        return std::make_pair(locally_owned_dofs, locally_active_dofs);
      }
    } // namespace internal

    template <typename SparseMatrixType,
              typename SparsityPatternType,
              typename Number>
    void
    restrict_to_full_matrices_(
      const SparseMatrixType                                  &system_matrix,
      const SparsityPatternType                               &sparsity_pattern,
      const std::vector<std::vector<types::global_dof_index>> &row_indices,
      const std::vector<std::vector<types::global_dof_index>> &col_indices,
      std::vector<FullMatrix<Number>>                         &blocks,
      const VectorT<Number>       &scaling = VectorT<Number>(),
      const SparsityCache<Number> &cache   = std::nullopt)
    {
      auto const &[locally_owned_dofs, locally_active_dofs] =
        cache.has_value() ?
          cache.value() :
          internal::get_cache<SparseMatrixType>(system_matrix, row_indices);
      // 1) collect remote rows of sparse matrix
      const auto locally_relevant_matrix_entries =
        internal::extract_remote_rows<Number>(system_matrix,
                                              sparsity_pattern,
                                              locally_active_dofs,
                                              internal::get_mpi_communicator(
                                                system_matrix));

      // 2) loop over all cells and "revert" assembly
      blocks.clear();
      blocks.resize(row_indices.size());

      for (unsigned int c = 0; c < row_indices.size(); ++c)
        {
          if (row_indices[c].empty() || col_indices[c].empty())
            continue;

          const auto &local_dof_row_indices = row_indices[c];
          const auto &local_dof_col_indices = col_indices[c];
          auto       &cell_matrix           = blocks[c];

          // allocate memory
          const unsigned int n_local_rows = local_dof_row_indices.size();
          const unsigned int n_local_cols = local_dof_col_indices.size();

          cell_matrix = FullMatrix<Number>(n_local_rows, n_local_cols);

          // loop over all entries of the restricted element matrix and
          // do different things if rows are locally owned or not and
          // if column entries of that row exist or not
          for (unsigned int i = 0; i < n_local_rows; ++i)
            for (unsigned int j = 0; j < n_local_cols; ++j)
              {
                if (locally_owned_dofs.is_element(
                      local_dof_row_indices[i])) // row is local
                  {
                    cell_matrix(i, j) =
                      sparsity_pattern.exists(local_dof_row_indices[i],
                                              local_dof_col_indices[j]) ?
                        system_matrix(local_dof_row_indices[i],
                                      local_dof_col_indices[j]) :
                        0;
                  }
                else // row is ghost
                  {
                    Assert(locally_active_dofs.is_element(
                             local_dof_row_indices[i]),
                           ExcInternalError());

                    const auto &row_entries = locally_relevant_matrix_entries
                      [locally_active_dofs.index_within_set(
                        local_dof_row_indices[i])];

                    const auto ptr = std::lower_bound(
                      row_entries.begin(),
                      row_entries.end(),
                      std::pair<types::global_dof_index, Number>{
                        local_dof_col_indices[j], /*dummy*/ 0.0},
                      [](const auto a, const auto b) {
                        return a.first < b.first;
                      });

                    if (ptr != row_entries.end() &&
                        local_dof_col_indices[j] == ptr->first)
                      cell_matrix(i, j) = ptr->second;
                    else
                      cell_matrix(i, j) = 0.0;
                  }
                if (scaling.size() != 0)
                  cell_matrix(i, j) *= scaling[local_dof_row_indices[i]];
              }
        }
    }
    template <typename BlockSparseMatrixType,
              typename BlockSparsityPatternType,
              typename Number>
    Table<2, std::vector<FullMatrix<Number>>>
    restrict_to_full_block_matrices_(
      const BlockSparseMatrixType    &system_matrix,
      const BlockSparsityPatternType &sparsity_pattern,
      const std::vector<std::vector<std::vector<types::global_dof_index>>>
        &row_indices,
      const std::vector<std::vector<std::vector<types::global_dof_index>>>
                                           &col_indices,
      const BlockVectorT<Number>           &scaling = BlockVectorT<Number>(),
      const Table<2, bool>                 &mask    = Table<2, bool>(),
      const Table<2, SparsityCache<Number>> cache =
        Table<2, SparsityCache<Number>>())
    {
      Table<2, std::vector<FullMatrix<Number>>> blocks_(
        sparsity_pattern.n_block_rows(), sparsity_pattern.n_block_cols());
      for (unsigned int i = 0; i < sparsity_pattern.n_block_rows(); ++i)
        for (unsigned int j = 0; j < sparsity_pattern.n_block_cols(); ++j)
          if (mask.empty() || mask(i, j))
            restrict_to_full_matrices_(system_matrix.block(i, j),
                                       sparsity_pattern.block(i, j),
                                       row_indices[i],
                                       col_indices[j],
                                       blocks_(i, j),
                                       scaling.block(i),
                                       cache.empty() ? std::nullopt :
                                                       cache(i, j));

#ifdef DEBUG
      for (unsigned int i = 0; i < sparsity_pattern.n_block_rows(); ++i)
        for (unsigned int j = 0; j < sparsity_pattern.n_block_cols(); ++j)
          if (mask.empty() || mask(i, j))
            AssertDimension(row_indices[i].size(), blocks_(i, j).size());
#endif
      return blocks_;
    }

    template <typename BlockSparseMatrixType,
              typename BlockSparsityPatternType,
              typename Number>
    void
    restrict_to_full_matrices_(
      const BlockSparseMatrixType    &system_matrix,
      const BlockSparsityPatternType &sparsity_pattern,
      const std::vector<std::vector<std::vector<types::global_dof_index>>>
        &row_indices,
      const std::vector<std::vector<std::vector<types::global_dof_index>>>
                                            &col_indices,
      std::vector<FullMatrix<Number>>       &blocks,
      const BlockVectorT<Number>            &scaling = BlockVectorT<Number>(),
      const Table<2, bool>                  &mask    = Table<2, bool>(),
      const Table<2, SparsityCache<Number>> &cache =
        Table<2, SparsityCache<Number>>())
    {
      Table<2, std::vector<FullMatrix<Number>>> blocks_ =
        restrict_to_full_block_matrices_(system_matrix,
                                         sparsity_pattern,
                                         row_indices,
                                         col_indices,
                                         scaling,
                                         mask,
                                         cache);

      unsigned int n_blocks;
      for (unsigned int iv = 0; iv < mask.size(0); ++iv)
        for (unsigned int jv = 0; jv < mask.size(1); ++jv)
          if ((mask.empty() || mask(iv, jv)))
            n_blocks = blocks_(iv, jv).size();

      blocks.resize(n_blocks);
      for (unsigned int ii = 0; ii < blocks.size(); ++ii)
        {
          auto        &block  = blocks[ii];
          unsigned int n_rows = 0, n_cols = 0;
          for (unsigned int i = 0; i < blocks_.size(0); ++i)
            n_rows += row_indices[i][ii].size();
          for (unsigned int i = 0; i < blocks_.size(1); ++i)
            n_cols += col_indices[i][ii].size();

          block = FullMatrix<Number>(n_rows, n_cols);
          for (unsigned int i = 0; i < blocks_.size(0); ++i)
            for (unsigned int j = 0; j < blocks_.size(1); ++j)
              if (mask.empty() || mask(i, j))
                {
                  auto const &block_block = blocks_(i, j)[ii];
                  block.fill(block_block,
                             i == 0 ? 0 : blocks_(i - 1, j)[ii].m(),
                             j == 0 ? 0 : blocks_(i, j - 1)[ii].n(),
                             0,
                             0);
                }
        }
    }
  } // namespace SparseMatrixTools

  template <int dim, typename Number>
  void
  make_block_sparsity_pattern_block(
    const DoFHandler<dim>             &dof_row,
    const DoFHandler<dim>             &dof_col,
    TrilinosWrappers::SparsityPattern &sparsity,
    const AffineConstraints<Number>   &constraints_row,
    const AffineConstraints<Number>   &constraints_col,
    const bool                         keep_constrained_dofs,
    const types::subdomain_id          subdomain_id)
  {
    const types::global_dof_index n_dofs_row = dof_row.n_dofs();
    const types::global_dof_index n_dofs_col = dof_col.n_dofs();
    (void)n_dofs_row;
    (void)n_dofs_col;
    Assert(sparsity.n_rows() == n_dofs_row,
           ExcDimensionMismatch(sparsity.n_rows(), n_dofs_row));
    Assert(sparsity.n_cols() == n_dofs_col,
           ExcDimensionMismatch(sparsity.n_cols(), n_dofs_col));
    // See DoFTools::make_sparsity_pattern
    if (const auto *triangulation =
          dynamic_cast<const parallel::DistributedTriangulationBase<dim> *>(
            &dof_row.get_triangulation()))
      Assert((subdomain_id == numbers::invalid_subdomain_id) ||
               (subdomain_id == triangulation->locally_owned_subdomain()),
             ExcMessage(
               "For distributed Triangulation objects and associated "
               "DoFHandler objects, asking for any subdomain other than the "
               "locally owned one does not make sense."));

    std::vector<types::global_dof_index> dofs_row_on_this_cell;
    dofs_row_on_this_cell.reserve(
      dof_row.get_fe_collection().max_dofs_per_cell());
    std::vector<types::global_dof_index> dofs_col_on_this_cell;
    dofs_col_on_this_cell.reserve(
      dof_col.get_fe_collection().max_dofs_per_cell());

    // See DoFTools::make_sparsity_pattern
    for (const auto &cell_row : dof_row.active_cell_iterators())
      if (((subdomain_id == numbers::invalid_subdomain_id) ||
           (subdomain_id == cell_row->subdomain_id())) &&
          cell_row->is_locally_owned())
        {
          typename DoFHandler<dim>::active_cell_iterator cell_col =
            cell_row->as_dof_handler_iterator(dof_col);
          const unsigned int dofs_per_cell_row =
            cell_row->get_fe().n_dofs_per_cell();
          dofs_row_on_this_cell.resize(dofs_per_cell_row);
          cell_row->get_dof_indices(dofs_row_on_this_cell);
          const unsigned int dofs_per_cell_col =
            cell_col->get_fe().n_dofs_per_cell();
          dofs_col_on_this_cell.resize(dofs_per_cell_col);
          cell_col->get_dof_indices(dofs_col_on_this_cell);
          // See DoFTools::make_sparsity_pattern
          constraints_row.add_entries_local_to_global(dofs_row_on_this_cell,
                                                      constraints_col,
                                                      dofs_col_on_this_cell,
                                                      sparsity,
                                                      keep_constrained_dofs);
        }
  }

  template <int dim,
            int fe_degree,
            int n_q_points_1d,
            int n_components,
            typename Number,
            typename VectorizedArrayType,
            typename MatrixType>
  void
  compute_block_matrix_block(
    const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
    const AffineConstraints<Number>                    &constraints,
    MatrixType                                         &matrix,
    const std::function<void(FEEvaluation<dim,
                                          fe_degree,
                                          n_q_points_1d,
                                          n_components,
                                          Number,
                                          VectorizedArrayType> &)>
                      &cell_operation,
    const unsigned int dof_r_no                   = 0,
    const unsigned int dof_c_no                   = 0,
    const unsigned int quad_r_no                  = 0,
    const unsigned int quad_c_no                  = 0,
    const unsigned int first_selected_r_component = 0,
    const unsigned int first_selected_c_component = 0)
  {
    DEAL_II_NOT_IMPLEMENTED();
  }

} // namespace dealii
