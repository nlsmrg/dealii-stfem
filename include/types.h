// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by Nils Margenberg and Peter Munch

#pragma once
#include <deal.II/base/types.h>

#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

#include <functional>
#include <vector>

using Number = double;
template <typename Number>
using VectorT = dealii::LinearAlgebra::distributed::Vector<Number>;
template <typename Number>
using BlockVectorT = dealii::LinearAlgebra::distributed::BlockVector<Number>;
using SparseMatrixType         = dealii::TrilinosWrappers::SparseMatrix;
using SparsityPatternType      = dealii::TrilinosWrappers::SparsityPattern;
using BlockSparseMatrixType    = dealii::TrilinosWrappers::BlockSparseMatrix;
using BlockSparsityPatternType = dealii::TrilinosWrappers::BlockSparsityPattern;

template <typename Number>
using BlockVectorSliceT =
  std::vector<std::reference_wrapper<const VectorT<Number>>>;
template <typename Number>
using MutableBlockVectorSliceT =
  std::vector<std::reference_wrapper<VectorT<Number>>>;
template <typename Number>
using SparsityCache =
  std::optional<std::pair<dealii::IndexSet, dealii::IndexSet>>;
namespace dealii
{
  template <int dim, typename Number, typename Number2 = Number>
  Tensor<2, dim, Number2>
  initialize_tensor_from_vector(const std::vector<Number> &values)
  {
    Tensor<2, dim, Number2> tensor;
    if (values.size() == 1)
      for (unsigned int i = 0; i < dim; ++i)
        tensor[i][i] = values[0];
    else if (values.size() == dim)
      for (unsigned int i = 0; i < dim; ++i)
        tensor[i][i] = values[i];
    else if (values.size() == dim * dim)
      for (unsigned int i = 0, index = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
          tensor[i][j] = values[index++];
    else
      Assert(false, ExcInternalError());

    return tensor;
  }

  template <int dim, typename Number, typename Number2 = Number>
  constexpr dealii::Tensor<1, dim, Number>
  constant_tensor(Number2 const &number)
  {
    dealii::Tensor<1, dim, Number> ten{};
    for (size_t i = 0; i < dim; ++i)
      ten[i] = number;
    return ten;
  }

  template <int dim, typename Number, typename Number2 = Number>
  constexpr dealii::Tensor<2, dim, Number>
  diagonal_tensor(Number2 const &number)
  {
    dealii::Tensor<2, dim, Number> ten{};
    for (size_t i = 0; i < dim; ++i)
      ten[i][i] = number;
    return ten;
  }
} // namespace dealii
constexpr dealii::types::manifold_id cylindrical_manifold_id = 0;
constexpr dealii::types::manifold_id tfi_manifold_id         = 1;

enum class ProblemType : unsigned int
{
  heat    = 1,
  wave    = 2,
  stokes  = 3,
  maxwell = 4,
  cdr     = 5
};

enum class CDRProblem : unsigned int
{
  rotating_hill     = 0,
  rotating_hill_eps = 1,
  step_layer        = 2,
  hemker            = 3,
  fichera           = 4
};

enum class CoarseningType : int
{
  space_or_time  = 0,
  space_and_time = 1,
};

enum class SupportedSmoothers : unsigned int
{
  Identity   = 0,
  Relaxation = 1,
  Chebyshev  = 2
};

enum class NonlinearTreatment : unsigned int
{
  None     = 0,
  Implicit = 1,
  Explicit = 2,
};

enum class NonlinearExtrapolation : unsigned int
{
  Auto         = 0,
  Constant     = 1,
  Polynomial   = 2,
  LeastSquares = 3,
};

enum class PatchType
{
  element,
  vertex_star,
};

static std::unordered_map<std::string, ProblemType> const str_to_problem_type =
  {{"heat", ProblemType::heat},
   {"wave", ProblemType::wave},
   {"cdr", ProblemType::cdr},
   {"maxwell", ProblemType::maxwell},
   {"stokes", ProblemType::stokes}};

static std::unordered_map<std::string,
                          dealii::MGTransferGlobalCoarseningTools::
                            PolynomialCoarseningSequenceType> const
  str_to_polynomial_coarsening_type = {
    {"bisect",
     dealii::MGTransferGlobalCoarseningTools::PolynomialCoarseningSequenceType::
       bisect},
    {"decrease_by_one",
     dealii::MGTransferGlobalCoarseningTools::PolynomialCoarseningSequenceType::
       decrease_by_one},
    {"go_to_one",
     dealii::MGTransferGlobalCoarseningTools::PolynomialCoarseningSequenceType::
       go_to_one}};

static std::unordered_map<std::string, CoarseningType> const
  str_to_coarsening_type = {{"space_and_time", CoarseningType::space_and_time},
                            {"space_or_time", CoarseningType::space_or_time}};

static std::unordered_map<std::string, SupportedSmoothers> const
  str_to_smoother_type = {{"chebyshev", SupportedSmoothers::Chebyshev},
                          {"relaxation", SupportedSmoothers::Relaxation}};
static std::unordered_map<std::string, CDRProblem> const str_to_cdr_problem = {
  {"rotatingHill", CDRProblem::rotating_hill},
  {"rotatingHillEps", CDRProblem::rotating_hill_eps},
  {"stepLayer", CDRProblem::step_layer},
  {"hemker", CDRProblem::hemker},
  {"fichera", CDRProblem::fichera}};

static std::unordered_map<std::string, NonlinearTreatment> const
  str_to_nonlinear_treatment = {{"none", NonlinearTreatment::None},
                                {"explicit", NonlinearTreatment::Explicit},
                                {"implicit", NonlinearTreatment::Implicit}};

static std::unordered_map<std::string, NonlinearExtrapolation> const
  str_to_nonlinear_extrapolation = {
    {"auto", NonlinearExtrapolation::Auto},
    {"constant", NonlinearExtrapolation::Constant},
    {"polynomial", NonlinearExtrapolation::Polynomial},
    {"leastSquares", NonlinearExtrapolation::LeastSquares}};


static std::unordered_map<std::string, PatchType> const str_to_patch_type =
  {{"element", PatchType::element}, {"vertexStar", PatchType::vertex_star}};
