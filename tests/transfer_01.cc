// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by Nils Margenberg and Peter Munch

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/repartitioning_policy_tools.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_creator.h>

#include "exact_solution.h"
#include "fe_time.h"
#include "operators.h"
#include "stmg.h"
#include "time_integrators.h"

using namespace dealii;

template <typename BlockVectorType>
class MGTransferST : public MGTransferBase<BlockVectorType>
{
public:
  using Number = typename BlockVectorType::value_type;

  void
  prolongate(const unsigned int,
             BlockVectorType &,
             const BlockVectorType &) const final
  {
    AssertThrow(false, ExcNotImplemented());
  }

  void
  prolongate_and_add(const unsigned int     to_level,
                     BlockVectorType       &dst,
                     const BlockVectorType &src) const final
  {
    const auto &prolongation_matrix = prolongation_matrices[to_level - 1];

    Vector<Number> dst_local(dst.n_blocks());
    Vector<Number> src_local(src.n_blocks());

    for (const auto i : dst.block(0).locally_owned_elements())
      {
        for (unsigned int j = 0; j < src_local.size(); ++j)
          src_local[j] = src.block(j)[i];

        prolongation_matrix.vmult(dst_local, src_local);

        for (unsigned int j = 0; j < dst_local.size(); ++j)
          dst.block(j)[i] = dst_local[j];
      }
  }

  void
  restrict_and_add(const unsigned int     from_level,
                   BlockVectorType       &dst,
                   const BlockVectorType &src) const final
  {
    const auto &restriction_matrix = restriction_matrices[from_level - 1];

    Vector<Number> dst_local(dst.n_blocks());
    Vector<Number> src_local(src.n_blocks());

    for (const auto i : dst.block(0).locally_owned_elements())
      {
        for (unsigned int j = 0; j < src_local.size(); ++j)
          src_local[j] = src.block(j)[i];

        restriction_matrix.vmult(dst_local, src_local);

        for (unsigned int j = 0; j < dst_local.size(); ++j)
          dst.block(j)[i] = dst_local[j];
      }
  }

  template <int dim>
  void
  copy_to_mg(const DoFHandler<dim>          &dof_handler,
             MGLevelObject<BlockVectorType> &dst,
             const BlockVectorType          &src) const
  {
    (void)dof_handler;

    const unsigned int min_level = 0;
    const unsigned int max_level = prolongation_matrices.size();

    dst.resize(min_level, max_level);

    for (unsigned int l = min_level; l <= max_level; ++l)
      {
        const unsigned int n_blocks =
          (l == max_level) ? prolongation_matrices[max_level - 1].m() :
                             prolongation_matrices[l].n();
        dst[l].reinit(n_blocks);
        for (unsigned int b = 0; b < n_blocks; ++b)
          dst[l].block(b).reinit(partitioners[l]);
      }

    dst[max_level] = src;
  }

  template <int dim>
  void
  copy_from_mg(const DoFHandler<dim>                &dof_handler,
               BlockVectorType                      &dst,
               const MGLevelObject<BlockVectorType> &src) const
  {
    (void)dof_handler;

    const unsigned int max_level = prolongation_matrices.size();

    dst = src[max_level];
  }

  void
  build(TimeStepType const type,
        unsigned int       r,
        unsigned int       n_timesteps_at_once,
        const std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>>
                                  &partitioners_,
        const bool                 restrict_is_transpose_prolongate,
        const std::vector<MGType> &mg_type_level)
  {
    partitioners                = partitioners_;
    unsigned int const n_levels = partitioners.size();
    prolongation_matrices.clear();
    prolongation_matrices.resize(n_levels - 1);
    restriction_matrices.clear();
    restriction_matrices.resize(n_levels - 1);
    unsigned int n_k_levels = 0, n_tau_levels = 0;
    for (auto const &el : mg_type_level)
      if (el == MGType::k)
        ++n_k_levels;
      else if (el == MGType::tau)
        ++n_tau_levels;
    AssertDimension(n_levels - 1, mg_type_level.size());
    Assert((type == TimeStepType::DG ? r + 1 : r >= n_k_levels),
           ExcLowerRange(r, n_k_levels));
    auto         p_matrix = prolongation_matrices.rbegin();
    auto         r_matrix = restriction_matrices.rbegin();
    unsigned int i        = mg_type_level.size() - 1;
    for (auto mgt = mg_type_level.rbegin(); mgt != mg_type_level.rend();
         ++mgt, --i, ++p_matrix, ++r_matrix)
      {
        bool k_mg = mg_type_level[i] == MGType::k;
        *p_matrix =
          k_mg ?
            get_time_projection_matrix(type, r - 1, r, n_timesteps_at_once) :
            get_time_prolongation_matrix(type, r, n_timesteps_at_once);
        if (restrict_is_transpose_prolongate)
          {
            r_matrix->reinit(p_matrix->n(), p_matrix->m());
            r_matrix->copy_transposed(*p_matrix);
          }
        else
          *r_matrix =
            k_mg ?
              get_time_projection_matrix(type, r, r - 1, n_timesteps_at_once) :
              get_time_restriction_matrix(type, r, n_timesteps_at_once);
        if (k_mg)
          --r;
        else
          n_timesteps_at_once /= 2;
      }
  }

private:
  std::vector<FullMatrix<Number>> prolongation_matrices;
  std::vector<FullMatrix<Number>> restriction_matrices;
  std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>> partitioners;
};

template <int dim, typename LevelMatrixType>
class TimeGMG
{
  using Number                     = double;
  using BlockVectorType            = BlockVectorT<Number>;
  using VectorType                 = VectorT<Number>;
  using MGTransferType             = MGTransferST<BlockVectorType>;
  using SmootherPreconditionerType = PreconditionVanka<Number>;
  using SmootherType =
    PreconditionRelaxation<LevelMatrixType, SmootherPreconditionerType>;
  using MGSmootherType =
    MGSmootherPrecondition<LevelMatrixType, SmootherType, BlockVectorType>;

public:
  TimeGMG(
    TimerOutput               &timer,
    TimeStepType const         type,
    unsigned int const         r,
    unsigned int const         n_timesteps_at_once,
    const std::vector<MGType> &mg_type_level,
    const DoFHandler<dim>     &dof_handler,
    const MGLevelObject<std::shared_ptr<const DoFHandler<dim>>>
      &mg_dof_handlers,
    const MGLevelObject<std::shared_ptr<const AffineConstraints<Number>>>
                                                                &mg_constraints,
    const MGLevelObject<std::shared_ptr<const LevelMatrixType>> &mg_operators,
    const MGLevelObject<std::shared_ptr<SmootherPreconditionerType>>
      &mg_smoother_)
    : timer(timer)
    , dof_handler(dof_handler)
    , mg_dof_handlers(mg_dof_handlers)
    , mg_constraints(mg_constraints)
    , mg_operators(mg_operators)
    , precondition_vanka(mg_smoother_)
    , min_level(mg_dof_handlers.min_level())
    , max_level(mg_dof_handlers.max_level())
    , partitioners(max_level + 1 - min_level)
  {
    for (unsigned int l = min_level; l <= max_level; ++l)
      {
        VectorType vector;
        mg_operators[l]->initialize_dof_vector(vector);
        partitioners[l - min_level] = vector.get_partitioner();
      }

    transfer_block = std::make_unique<MGTransferST<BlockVectorType>>();
    transfer_block->build(
      type, r, n_timesteps_at_once, partitioners, true, mg_type_level);
  }

  void
  reinit() const
  {
    PreconditionerGMGAdditionalData additional_data;
    additional_data.estimate_relaxation = true;
    additional_data.smoothing_range     = 2;
    // wrap level operators
    mg_matrix = mg::Matrix<BlockVectorType>(mg_operators);
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data(
      min_level, max_level);
    // setup smoothers on each level
    for (unsigned int level = min_level; level <= max_level; ++level)
      {
        smoother_data[level].preconditioner = precondition_vanka[level];
        smoother_data[level].n_iterations   = 1;
        smoother_data[level].relaxation = additional_data.estimate_relaxation ?
                                            estimate_relaxation(level) :
                                            1.0;
      }
    mg_smoother = std::make_unique<MGSmootherType>(1, true, false, false);
    mg_smoother->initialize(mg_operators, smoother_data);
    mg_coarse = std::make_unique<MGCoarseGridApplySmoother<BlockVectorType>>(
      *mg_smoother);
    mg = std::make_unique<Multigrid<BlockVectorType>>(mg_matrix,
                                                      *mg_coarse,
                                                      *transfer_block,
                                                      *mg_smoother,
                                                      *mg_smoother,
                                                      min_level,
                                                      max_level);
    preconditioner =
      std::make_unique<PreconditionMG<dim, BlockVectorType, MGTransferType>>(
        dof_handler, *mg, *transfer_block);
  }

  double
  estimate_relaxation(unsigned level) const
  {
    if (level == 0)
      return 1;

    PreconditionerGMGAdditionalData additional_data;
    using ChebyshevPreconditionerType =
      PreconditionChebyshev<LevelMatrixType,
                            BlockVectorType,
                            SmootherPreconditionerType>;

    typename ChebyshevPreconditionerType::AdditionalData
      chebyshev_additional_data;
    chebyshev_additional_data.preconditioner  = precondition_vanka[level];
    chebyshev_additional_data.smoothing_range = additional_data.smoothing_range;
    chebyshev_additional_data.degree = additional_data.smoothing_degree;
    chebyshev_additional_data.eig_cg_n_iterations =
      additional_data.smoothing_eig_cg_n_iterations;
    chebyshev_additional_data.eigenvalue_algorithm =
      ChebyshevPreconditionerType::AdditionalData::EigenvalueAlgorithm::
        power_iteration;
    chebyshev_additional_data.polynomial_type =
      ChebyshevPreconditionerType::AdditionalData::PolynomialType::fourth_kind;
    auto chebyshev = std::make_shared<ChebyshevPreconditionerType>();
    chebyshev->initialize(*mg_operators[level], chebyshev_additional_data);

    BlockVectorType vec;
    mg_operators[level]->initialize_dof_vector(vec);

    const auto evs = chebyshev->estimate_eigenvalues(vec);

    const double alpha = (chebyshev_additional_data.smoothing_range > 1. ?
                            evs.max_eigenvalue_estimate /
                              chebyshev_additional_data.smoothing_range :
                            std::min(0.9 * evs.max_eigenvalue_estimate,
                                     evs.min_eigenvalue_estimate));

    double omega = 2.0 / (alpha + evs.max_eigenvalue_estimate);
    deallog << "\n-Eigenvalue estimation level " << level
            << ":\n"
               "    Relaxation parameter: "
            << omega
            << "\n"
               "    Minimum eigenvalue: "
            << evs.min_eigenvalue_estimate
            << "\n"
               "    Maximum eigenvalue: "
            << evs.max_eigenvalue_estimate << std::endl;

    return omega;
  }

  template <typename SolutionVectorType = BlockVectorType>
  void
  vmult(SolutionVectorType &dst, const SolutionVectorType &src) const
  {
    TimerOutput::Scope scope(timer, "gmg");
    preconditioner->vmult(dst, src);
  }

private:
  TimerOutput &timer;

  const DoFHandler<dim>                                      &dof_handler;
  const MGLevelObject<std::shared_ptr<const DoFHandler<dim>>> mg_dof_handlers;
  const MGLevelObject<std::shared_ptr<const AffineConstraints<Number>>>
                                                              mg_constraints;
  const MGLevelObject<std::shared_ptr<const LevelMatrixType>> mg_operators;
  const MGLevelObject<std::shared_ptr<SmootherPreconditionerType>>
    precondition_vanka;

  const unsigned int min_level;
  const unsigned int max_level;

  std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>> partitioners;
  std::unique_ptr<MGTransferType> transfer_block;

  mutable mg::Matrix<BlockVectorType> mg_matrix;


  mutable std::unique_ptr<MGSmootherType> mg_smoother;

  mutable std::unique_ptr<MGCoarseGridBase<BlockVectorType>> mg_coarse;
  mutable std::unique_ptr<SolverControl>                solver_control_coarse;
  mutable std::unique_ptr<SolverGMRES<BlockVectorType>> gmres_coarse;
  mutable std::unique_ptr<
    PreconditionRelaxation<LevelMatrixType, DiagonalMatrix<BlockVectorType>>>
    preconditioner_coarse;

  mutable std::unique_ptr<Multigrid<BlockVectorType>> mg;

  mutable std::unique_ptr<PreconditionMG<dim, BlockVectorType, MGTransferType>>
    preconditioner;
};


template <int dim>
void
test(dealii::ConditionalOStream &pcout,
     MPI_Comm const              comm_global,
     TimeStepType const          type,
     unsigned int                n_timesteps_at_once = 1,
     bool                        test_robustness     = true)
{
  using Number = double;
  ConvergenceTable table;
  MappingQ1<dim>   mapping;

  const bool print_timing = false;

  auto convergence_test = [&](int const          refinement,
                              unsigned int const fe_degree,
                              bool               do_output,
                              unsigned int       n_timesteps_at_once) {
    using VectorType      = VectorT<Number>;
    using BlockVectorType = BlockVectorT<Number>;
    const unsigned int n_blocks =
      (type == TimeStepType::DG ? fe_degree + 1 : fe_degree) *
      n_timesteps_at_once;
    auto const  basis = get_time_basis(type, fe_degree);
    FE_Q<dim>   fe(fe_degree + 1);
    QGauss<dim> quad(fe.tensor_degree() + 1);

    auto tria =
      std::make_shared<parallel::distributed::Triangulation<dim>>(comm_global);
    DoFHandler<dim> dof_handler(*tria);

    GridGenerator::hyper_cube(*tria);
    tria->refine_global(test_robustness ? 1 : refinement);

    dof_handler.distribute_dofs(fe);

    AffineConstraints<Number> constraints;
    IndexSet                  locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    DoFTools::make_zero_boundary_constraints(dof_handler, constraints);
    constraints.close();
    pcout << ":: Number of active cells: " << tria->n_active_cells() << "\n"
          << ":: Number of degrees of freedom: " << dof_handler.n_dofs() << "\n"
          << std::endl;

    // create sparsity pattern
    SparsityPatternType sparsity_pattern(dof_handler.locally_owned_dofs(),
                                         dof_handler.locally_owned_dofs(),
                                         dof_handler.get_communicator());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    sparsity_pattern,
                                    constraints,
                                    false);
    sparsity_pattern.compress();

    // create scalar siffness matrix
    SparseMatrixType K;
    K.reinit(sparsity_pattern);

    // create scalar mass matrix
    SparseMatrixType M;
    M.reinit(sparsity_pattern);

    double time           = 0.;
    double time_step_size = 1.0 * pow(2.0, -(refinement + 1));
    double end_time       = 1.;
    Number frequency      = 1.0;

    // matrix-free operators
    MatrixFreeOperatorScalar<dim, Number> K_mf(
      mapping, dof_handler, constraints, quad, 0.0, 1.0);
    MatrixFreeOperatorScalar<dim, Number> M_mf(
      mapping, dof_handler, constraints, quad, 1.0, 0.0);
    K_mf.compute_system_matrix(K);
    M_mf.compute_system_matrix(M);
    // We need the case n_timesteps_at_once=1 matrices always for the
    // integration of the source f
    auto [Alpha_1, Beta_1, Gamma_1, Zeta_1] =
      get_fe_time_weights<Number>(type, fe_degree, time_step_size, 1);
    auto [Alpha, Beta, Gamma, Zeta] = get_fe_time_weights<Number>(
      type, fe_degree, time_step_size, n_timesteps_at_once);
    std::vector<MGType> mg_type_level =
      get_time_mg_sequence(1,
                           fe_degree,
                           type == TimeStepType::DG ? 0 : 1,
                           n_timesteps_at_once,
                           1,
                           MGType::k);

    TimerOutput timer(pcout,
                      TimerOutput::never,
                      TimerOutput::cpu_and_wall_times);

    FullMatrix<Number> lhs_uK, lhs_uM, rhs_uK, rhs_uM, zero(Gamma.m(), 1);
    std::unique_ptr<SystemMatrix<Number, MatrixFreeOperatorScalar<dim, Number>>>
      rhs_matrix, matrix;
    lhs_uK = Alpha;
    lhs_uM = Beta;
    rhs_uK = (type == TimeStepType::CGP) ? Gamma : zero;
    rhs_uM = (type == TimeStepType::CGP) ? Zeta : Gamma;
    matrix = std::make_unique<
      SystemMatrix<Number, MatrixFreeOperatorScalar<dim, Number>>>(
      timer, K_mf, M_mf, lhs_uK, lhs_uM);
    rhs_matrix = std::make_unique<
      SystemMatrix<Number, MatrixFreeOperatorScalar<dim, Number>>>(
      timer, K_mf, M_mf, rhs_uK, rhs_uM);

    /// TimeGMG
    RepartitioningPolicyTools::DefaultPolicy<dim>          policy(true);
    std::vector<std::shared_ptr<const Triangulation<dim>>> mg_triangulations(
      mg_type_level.size() + 1, tria);
    const unsigned int min_level = 0;
    const unsigned int max_level = mg_triangulations.size() - 1;
    pcout << ":: Min Level " << min_level << "  Max Level " << max_level
          << std::endl;
    MGLevelObject<std::shared_ptr<const DoFHandler<dim>>> mg_dof_handlers(
      min_level, max_level);
    MGLevelObject<std::shared_ptr<const SparsityPatternType>>
      mg_sparsity_patterns(min_level, max_level);
    MGLevelObject<std::shared_ptr<const SparseMatrixType>> mg_M(min_level,
                                                                max_level);
    MGLevelObject<std::shared_ptr<const SparseMatrixType>> mg_K(min_level,
                                                                max_level);
    MGLevelObject<std::shared_ptr<
      const SystemMatrix<Number, MatrixFreeOperatorScalar<dim, Number>>>>
      mg_operators(min_level, max_level);
    MGLevelObject<std::shared_ptr<const MatrixFreeOperatorScalar<dim, Number>>>
      mg_M_mf(min_level, max_level);
    MGLevelObject<std::shared_ptr<const MatrixFreeOperatorScalar<dim, Number>>>
      mg_K_mf(min_level, max_level);
    MGLevelObject<std::shared_ptr<const AffineConstraints<double>>>
      mg_constraints(min_level, max_level);
    MGLevelObject<std::shared_ptr<PreconditionVanka<Number>>>
         precondition_vanka(min_level, max_level);
    auto fetw = get_fe_time_weights<double>(
      type, fe_degree, time_step_size, n_timesteps_at_once, mg_type_level);

    for (unsigned int l = min_level; l <= max_level; ++l)
      {
        auto dof_handler_ =
          std::make_shared<DoFHandler<dim>>(*mg_triangulations[l]);
        auto constraints_ = std::make_shared<AffineConstraints<Number>>();
        dof_handler_->distribute_dofs(fe);

        IndexSet locally_relevant_dofs;
        DoFTools::extract_locally_relevant_dofs(*dof_handler_,
                                                locally_relevant_dofs);

        constraints_->reinit(locally_relevant_dofs);
        DoFTools::make_zero_boundary_constraints(*dof_handler_,
                                                 0,
                                                 *constraints_);
        constraints_->close();

        // matrix-free operators
        auto K_mf_ = std::make_shared<MatrixFreeOperatorScalar<dim, Number>>(
          mapping, *dof_handler_, *constraints_, quad, 0.0, 1.0);
        auto M_mf_ = std::make_shared<MatrixFreeOperatorScalar<dim, Number>>(
          mapping, *dof_handler_, *constraints_, quad, 1.0, 0.0);


        mg_operators[l] = std::make_shared<
          SystemMatrix<Number, MatrixFreeOperatorScalar<dim, Number>>>(
          timer, *K_mf_, *M_mf_, fetw[l][0], fetw[l][1]);

        auto sparsity_pattern_ = std::make_shared<SparsityPatternType>(
          dof_handler_->locally_owned_dofs(),
          dof_handler_->locally_owned_dofs(),
          dof_handler_->get_communicator());
        DoFTools::make_sparsity_pattern(*dof_handler_,
                                        *sparsity_pattern_,
                                        *constraints_,
                                        false);
        sparsity_pattern_->compress();

        auto K_ = std::make_shared<SparseMatrixType>();
        K_->reinit(*sparsity_pattern_);
        auto M_ = std::make_shared<SparseMatrixType>();
        M_->reinit(*sparsity_pattern_);
        K_mf_->compute_system_matrix(*K_);
        M_mf_->compute_system_matrix(*M_);
        mg_sparsity_patterns[l] = sparsity_pattern_;
        mg_M_mf[l]              = M_mf_;
        mg_K_mf[l]              = K_mf_;
        mg_M[l]                 = M_;
        mg_K[l]                 = K_;
        mg_dof_handlers[l]      = dof_handler_;
        mg_constraints[l]       = constraints_;
        precondition_vanka[l] =
          std::make_shared<PreconditionVanka<Number>>(timer,
                                                      mg_K[l],
                                                      mg_M[l],
                                                      mg_sparsity_patterns[l],
                                                      fetw[l][0],
                                                      fetw[l][1],
                                                      mg_dof_handlers[l]);
      }

    std::unique_ptr<BlockVectorT<Number>> tmp1, tmp2;
    if (!std::is_same_v<Number, Number>)
      {
        tmp1 = std::make_unique<BlockVectorT<Number>>();
        tmp2 = std::make_unique<BlockVectorT<Number>>();
        matrix->initialize_dof_vector(*tmp1);
        matrix->initialize_dof_vector(*tmp2);
      }
    using Preconditioner =
      TimeGMG<dim, SystemMatrix<Number, MatrixFreeOperatorScalar<dim, Number>>>;
    auto preconditioner = std::make_unique<Preconditioner>(timer,
                                                           type,
                                                           fe_degree,
                                                           n_timesteps_at_once,
                                                           mg_type_level,
                                                           dof_handler,
                                                           mg_dof_handlers,
                                                           mg_constraints,
                                                           mg_operators,
                                                           precondition_vanka);
    preconditioner->reinit();
    //
    /// TimeGMG

    std::unique_ptr<Function<dim, Number>> rhs_function;
    std::unique_ptr<Function<dim, Number>> exact_solution =
      std::make_unique<ExactSolution<dim, Number>>(frequency);
    rhs_function = std::make_unique<RHSFunction<dim, Number>>(frequency);
    auto integrate_rhs_function =
      [&mapping, &dof_handler, &quad, &rhs_function, &constraints](
        const double time, VectorType &rhs) -> void {
      rhs_function->set_time(time);
      rhs = 0.0;
      VectorTools::create_right_hand_side(
        mapping, dof_handler, quad, *rhs_function, rhs, constraints);
    };
    [[maybe_unused]] auto evaluate_exact_solution =
      [&mapping, &dof_handler, &exact_solution](const double time,
                                                VectorType  &tmp) -> void {
      exact_solution->set_time(time);
      VectorTools::interpolate(mapping, dof_handler, *exact_solution, tmp);
    };

    auto evaluate_numerical_solution =
      [&constraints, &basis, &type](const double           time,
                                    VectorType            &tmp,
                                    BlockVectorType const &x,
                                    VectorType const      &prev_x,
                                    unsigned block_offset = 0) -> void {
      int i = 0;
      tmp   = 0.0;
      for (auto const &el : basis)
        {
          if (double v = el.value(time); v != 0.0)
            {
              if (type == TimeStepType::DG)
                tmp.add(v, x.block(block_offset + i));
              else
                tmp.add(v,
                        (block_offset + i == 0) ?
                          prev_x :
                          x.block(block_offset + i - 1));
            }
          ++i;
        }
      constraints.distribute(tmp);
    };

    BlockVectorType x(n_blocks);
    matrix->initialize_dof_vector(x);
    VectorType prev_x;
    matrix->initialize_dof_vector(prev_x);

    VectorType exact;
    matrix->initialize_dof_vector(exact);
    VectorType numeric;
    matrix->initialize_dof_vector(numeric);

    unsigned int                 timestep_number = 0;
    ErrorCalculator<dim, Number> error_calculator(type,
                                                  fe_degree,
                                                  fe_degree,
                                                  mapping,
                                                  dof_handler,
                                                  *exact_solution,
                                                  evaluate_numerical_solution);

    using TimeHeat = TimeIntegratorFO<
      dim,
      Number,
      Preconditioner,
      SystemMatrix<Number, MatrixFreeOperatorScalar<dim, Number>>>;
    std::vector<std::function<void(const double, VectorType &)>>
             integrate_rhs_function_{integrate_rhs_function};
    TimeHeat step(type,
                  fe_degree,
                  Alpha_1,
                  Gamma_1,
                  1.e-12,
                  *matrix,
                  *preconditioner,
                  *rhs_matrix,
                  integrate_rhs_function_,
                  n_timesteps_at_once,
                  true);

    // interpolate initial value
    auto nt_dofs = static_cast<unsigned int>(n_blocks / n_timesteps_at_once);
    evaluate_exact_solution(0, x.block(x.n_blocks() - 1));
    double       l2 = 0., l8 = -1., h1_semi = 0.;
    unsigned int total_gmres_iterations = 0;
    while (time < end_time)
      {
        TimerOutput::Scope scope(timer, "step");

        ++timestep_number;
        dealii::deallog << "Step " << timestep_number << " t = " << time
                        << std::endl;
        prev_x = x.block(x.n_blocks() - 1);
        step.solve(x, prev_x, timestep_number, time, time_step_size);
        total_gmres_iterations += step.last_step();
        for (unsigned int i = 0; i < n_blocks; ++i)
          constraints.distribute(x.block(i));
        auto error_on_In = error_calculator.evaluate_error(
          time, time_step_size, x, prev_x, n_timesteps_at_once);
        time += n_timesteps_at_once * time_step_size;

        l2 += error_on_In[VectorTools::L2_norm];
        l8 = std::max(error_on_In[VectorTools::Linfty_norm], l8);
        h1_semi += error_on_In[VectorTools::H1_seminorm];
        if (do_output)
          {
            numeric = 0.0;
            evaluate_numerical_solution(
              1.0, numeric, x, prev_x, (n_timesteps_at_once - 1) * nt_dofs);
            DataOut<dim> data_out;
            data_out.attach_dof_handler(dof_handler);
            data_out.add_data_vector(numeric, "solution");
            data_out.build_patches();
            std::ofstream output("solution." +
                                 Utilities::int_to_string(timestep_number, 4) +
                                 ".vtu");
            data_out.write_vtu(output);
          }
      }
    double average_gmres_iter = static_cast<double>(total_gmres_iterations) /
                                static_cast<double>(timestep_number);
    pcout << "\nAverage GMRES iterations: " << average_gmres_iter << " (over "
          << timestep_number << " timesteps)\n"
          << std::endl;
    if (print_timing)
      timer.print_wall_time_statistics(MPI_COMM_WORLD);

    unsigned int const n_active_cells = tria->n_active_cells();
    unsigned int const n_dofs         = dof_handler.n_dofs();
    table.add_value("cells", n_active_cells);
    table.add_value("s-dofs", n_dofs);
    table.add_value("t-dofs", n_blocks);
    table.add_value("st-dofs", timestep_number * n_dofs * n_blocks);
    table.add_value("L\u221E-L\u221E", l8);
    table.add_value("L2-L2", std::sqrt(l2));
    table.add_value("L2-H1_semi", std::sqrt(h1_semi));
  };
  for (int j = 1; j < 4; ++j)
    {
      for (int i = 2; i < 5; ++i)
        convergence_test(i,
                         type == TimeStepType::DG ? j : j + 1,
                         false,
                         n_timesteps_at_once);

      table.set_precision("L\u221E-L\u221E", 5);
      table.set_precision("L2-L2", 5);
      table.set_precision("L2-H1_semi", 5);
      table.set_scientific("L\u221E-L\u221E", true);
      table.set_scientific("L2-L2", true);
      table.set_scientific("L2-H1_semi", true);
      table.evaluate_convergence_rates("L\u221E-L\u221E",
                                       ConvergenceTable::reduction_rate_log2);
      table.evaluate_convergence_rates("L2-L2",
                                       ConvergenceTable::reduction_rate_log2);
      table.evaluate_convergence_rates("L2-H1_semi",
                                       ConvergenceTable::reduction_rate_log2);
      if (pcout.is_active())
        table.write_text(pcout.get_stream());
      pcout << std::endl;
      table.clear();
    }
}


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  dealii::ConditionalOStream pcout(
    std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  std::string filename =
    "proc" +
    std::to_string(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)) +
    ".log";
  std::ofstream pout(filename);
  dealii::deallog.attach(pout);
  dealii::deallog.depth_console(0);
  MPI_Comm  comm = MPI_COMM_WORLD;
  const int dim  = 2;
  dealii::deallog << "HEAT single step" << std::endl;
  test<dim>(pcout, comm, TimeStepType::DG);
  test<dim>(pcout, comm, TimeStepType::CGP);
  dealii::deallog << "HEAT 2 steps at once" << std::endl;
  test<dim>(pcout, comm, TimeStepType::DG, 2);
  test<dim>(pcout, comm, TimeStepType::CGP, 2);
  dealii::deallog << std::endl;

  pcout << std::endl;
  return 0;
}
