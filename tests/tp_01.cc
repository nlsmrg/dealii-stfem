// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by Nils Margenberg and Peter Munch

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/repartitioning_policy_tools.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_creator.h>

#include "include/exact_solution.h"
#include "include/fe_time.h"
#include "include/getopt++.h"
#include "include/gmg.h"
#include "include/operators.h"
#include "include/time_integrators.h"

using namespace dealii;
using dealii::numbers::PI;

template <typename Number_dst, typename Number_src>
FullMatrix<Number_dst>
convert_to(FullMatrix<Number_src> const &in)
{
  FullMatrix<Number_dst> out(in.m(), in.n());
  out.copy_from(in);
  return out;
}

template <typename Number, typename NumberPreconditioner = Number>
void
test(dealii::ConditionalOStream &pcout,
     MPI_Comm const              comm_global,
     std::string                 file_name,
     int                         dim)
{
  std::variant<Parameters<2>, Parameters<3>> parameters;
  if (dim == 2)
    parameters = Parameters<2>();
  else
    parameters = Parameters<3>();
  std::visit([&](auto &p) { p.parse(file_name); }, parameters);
  ConvergenceTable table;
  ConvergenceTable itable;

  auto convergence_test = [&]<int dim>(int const              refinement,
                                       int const              fe_degree,
                                       Parameters<dim> const &parameters) {
    const bool print_timing      = parameters.print_timing;
    const bool space_time_mg     = parameters.space_time_mg;
    const bool time_before_space = parameters.time_before_space;
    const bool is_cgp            = parameters.type == TimeStepType::CGP;
    Assert(parameters.fe_degree >= (is_cgp ? 1 : 0),
           ExcLowerRange(parameters.fe_degree, (is_cgp ? 1 : 0)));
    Assert(parameters.refinement >= 1, ExcLowerRange(parameters.refinement, 1));

    MappingQ1<dim>     mapping;
    bool const         do_output           = parameters.do_output;
    unsigned int const n_timesteps_at_once = parameters.n_timesteps_at_once;

    using VectorType            = VectorT<Number>;
    using BlockVectorType       = BlockVectorT<Number>;
    const unsigned int nt_dofs  = is_cgp ? fe_degree : fe_degree + 1;
    const unsigned int n_blocks = nt_dofs * n_timesteps_at_once;

    auto const  basis = get_time_basis(parameters.type, fe_degree);
    FE_Q<dim>   fe(fe_degree + 1);
    QGauss<dim> quad(fe.tensor_degree() + 1);

    parallel::distributed::Triangulation<dim> tria(comm_global);
    DoFHandler<dim>                           dof_handler(tria);

    GridGenerator::subdivided_hyper_rectangle(tria,
                                              parameters.subdivisions,
                                              parameters.hyperrect_lower_left,
                                              parameters.hyperrect_upper_right);
    double spc_step = GridTools::minimal_cell_diameter(tria) / std::sqrt(dim);
    tria.refine_global(refinement);
    if (parameters.distort_grid != 0.0)
      GridTools::distort_random(parameters.distort_grid, tria);

    dof_handler.distribute_dofs(fe);

    AffineConstraints<Number> constraints;
    IndexSet                  locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    DoFTools::make_zero_boundary_constraints(dof_handler, constraints);
    constraints.close();
    pcout << ":: Number of active cells: " << tria.n_global_active_cells()
          << "\n"
          << ":: Number of degrees of freedom: " << dof_handler.n_dofs()
          << "\n";

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

    double       time     = 0.;
    double       time_len = parameters.end_time - time;
    unsigned int n_steps  = static_cast<unsigned int>((time_len) / spc_step);
    double time_step_size = time_len * pow(2.0, -(refinement + 1)) / n_steps;
    Number frequency      = parameters.frequency;

    Coefficient<dim> coeff(parameters);
    // matrix-free operators
    MatrixFreeOperator<dim, Number> K_mf(
      mapping, dof_handler, constraints, quad, 0.0, 1.0);
    MatrixFreeOperator<dim, Number> M_mf(
      mapping, dof_handler, constraints, quad, 1.0, 0.0);
    if (!parameters.space_time_conv_test)
      K_mf.evaluate_coefficient(coeff);

    if (false)
      {
        MatrixCreator::create_laplace_matrix<dim, dim>(
          mapping, dof_handler, quad, K, nullptr, constraints);
        MatrixCreator::create_mass_matrix<dim, dim>(
          mapping, dof_handler, quad, M, nullptr, constraints);
      }
    else
      {
        K_mf.compute_system_matrix(K);
        M_mf.compute_system_matrix(M);
      }

    // We need the case n_timesteps_at_once=1 matrices always for the
    // integration of the source f
    auto [Alpha_1, Beta_1, Gamma_1, Zeta_1] = get_fe_time_weights<Number>(
      parameters.type, fe_degree, time_step_size, 1);
    auto [Alpha, Beta, Gamma, Zeta] = get_fe_time_weights<Number>(
      parameters.type, fe_degree, time_step_size, n_timesteps_at_once);

    TimerOutput timer(pcout,
                      TimerOutput::never,
                      TimerOutput::cpu_and_wall_times);

    FullMatrix<Number> lhs_uK, lhs_uM, rhs_uK, rhs_uM, rhs_vM,
      zero(Gamma.m(), Gamma.n());
    std::unique_ptr<SystemMatrix<Number, MatrixFreeOperator<dim, Number>>>
      rhs_matrix, rhs_matrix_v, matrix;
    if (parameters.problem == ProblemType::wave)
      {
        auto [Alpha_lhs, Beta_lhs, rhs_uK_, rhs_uM_, rhs_vM_] =
          get_fe_time_weights_wave(parameters.type,
                                   Alpha_1,
                                   Beta_1,
                                   Gamma_1,
                                   Zeta_1,
                                   n_timesteps_at_once);

        lhs_uK       = Alpha_lhs;
        lhs_uM       = Beta_lhs;
        rhs_uK       = rhs_uK_;
        rhs_uM       = rhs_uM_;
        rhs_vM       = rhs_vM_;
        rhs_matrix_v = std::make_unique<
          SystemMatrix<Number, MatrixFreeOperator<dim, Number>>>(
          timer, K_mf, M_mf, zero, rhs_vM);
      }
    else
      {
        lhs_uK = Alpha;
        lhs_uM = Beta;
        rhs_uK = is_cgp ? Gamma : zero;
        rhs_uM = is_cgp ? Zeta : Gamma;
      }
    matrix =
      std::make_unique<SystemMatrix<Number, MatrixFreeOperator<dim, Number>>>(
        timer, K_mf, M_mf, lhs_uK, lhs_uM);
    rhs_matrix =
      std::make_unique<SystemMatrix<Number, MatrixFreeOperator<dim, Number>>>(
        timer, K_mf, M_mf, rhs_uK, rhs_uM);

    /// GMG
    RepartitioningPolicyTools::DefaultPolicy<dim>          policy(true);
    std::vector<std::shared_ptr<const Triangulation<dim>>> mg_triangulations =
      MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(
        tria, policy);
    unsigned int fe_degree_min =
      space_time_mg ? parameters.fe_degree_min : fe_degree;
    unsigned int n_timesteps_min =
      space_time_mg ? std::max(parameters.n_timesteps_at_once_min, 1) :
                      n_timesteps_at_once;
    std::vector<TimeMGType> mg_type_level =
      get_time_mg_sequence(mg_triangulations.size(),
                           fe_degree,
                           fe_degree_min,
                           n_timesteps_at_once,
                           n_timesteps_min,
                           TimeMGType::k,
                           time_before_space);
    mg_triangulations =
      get_space_time_triangulation(mg_type_level, mg_triangulations);


    const unsigned int min_level = 0;
    const unsigned int max_level = mg_triangulations.size() - 1;
    pcout << ":: Min Level " << min_level << "  Max Level " << max_level
          << "\n";
    MGLevelObject<std::shared_ptr<const DoFHandler<dim>>> mg_dof_handlers(
      min_level, max_level);
    MGLevelObject<
      std::shared_ptr<const MatrixFreeOperator<dim, NumberPreconditioner>>>
      mg_M_mf(min_level, max_level);
    MGLevelObject<
      std::shared_ptr<const MatrixFreeOperator<dim, NumberPreconditioner>>>
      mg_K_mf(min_level, max_level);
    MGLevelObject<
      std::shared_ptr<const AffineConstraints<NumberPreconditioner>>>
      mg_constraints(min_level, max_level);
    MGLevelObject<std::shared_ptr<
      const SystemMatrix<NumberPreconditioner,
                         MatrixFreeOperator<dim, NumberPreconditioner>>>>
      mg_operators(min_level, max_level);
    MGLevelObject<std::shared_ptr<PreconditionVanka<NumberPreconditioner>>>
      precondition_vanka(min_level, max_level);
    std::vector<std::array<FullMatrix<NumberPreconditioner>, 4>> fetw;
    std::vector<std::array<FullMatrix<NumberPreconditioner>, 5>> fetw_w;
    if (parameters.problem == ProblemType::heat)
      fetw =
        get_fe_time_weights<Number, NumberPreconditioner>(parameters.type,
                                                          fe_degree,
                                                          time_step_size,
                                                          n_timesteps_at_once,
                                                          mg_type_level);
    else if (parameters.problem == ProblemType::wave)
      fetw_w = get_fe_time_weights_wave<Number, NumberPreconditioner>(
        parameters.type,
        fe_degree,
        time_step_size,
        n_timesteps_at_once,
        mg_type_level);

    for (unsigned int l = min_level; l <= max_level; ++l)
      {
        auto dof_handler_ =
          std::make_shared<DoFHandler<dim>>(*mg_triangulations[l]);
        auto constraints_ =
          std::make_shared<AffineConstraints<NumberPreconditioner>>();
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
        auto K_mf_ =
          std::make_shared<MatrixFreeOperator<dim, NumberPreconditioner>>(
            mapping, *dof_handler_, *constraints_, quad, 0.0, 1.0);
        auto M_mf_ =
          std::make_shared<MatrixFreeOperator<dim, NumberPreconditioner>>(
            mapping, *dof_handler_, *constraints_, quad, 1.0, 0.0);
        if (!parameters.space_time_conv_test)
          K_mf_->evaluate_coefficient(coeff);

        auto const &lhs_uK_p =
          parameters.problem == ProblemType::heat ? fetw[l][0] : fetw_w[l][0];
        auto const &lhs_uM_p =
          parameters.problem == ProblemType::heat ? fetw[l][1] : fetw_w[l][1];

        mg_operators[l] = std::make_shared<
          SystemMatrix<NumberPreconditioner,
                       MatrixFreeOperator<dim, NumberPreconditioner>>>(
          timer, *K_mf_, *M_mf_, lhs_uK_p, lhs_uM_p);

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

        // matrix->attach(*mg_operators[l]);
        mg_M_mf[l]         = M_mf_;
        mg_K_mf[l]         = K_mf_;
        mg_dof_handlers[l] = dof_handler_;
        mg_constraints[l]  = constraints_;
        precondition_vanka[l] =
          std::make_shared<PreconditionVanka<NumberPreconditioner>>(
            timer,
            K_,
            M_,
            sparsity_pattern_,
            lhs_uK_p,
            lhs_uM_p,
            mg_dof_handlers[l]);
      }



    // std::shared_ptr<MGSmootherBase<BlockVectorType>> smoother =
    //   std::make_shared<MGSmootherIdentity<BlockVectorType>>();
    std::unique_ptr<BlockVectorT<NumberPreconditioner>> tmp1, tmp2;
    if (!std::is_same_v<Number, NumberPreconditioner>)
      {
        tmp1 = std::make_unique<BlockVectorT<NumberPreconditioner>>();
        tmp2 = std::make_unique<BlockVectorT<NumberPreconditioner>>();
        matrix->initialize_dof_vector(*tmp1);
        matrix->initialize_dof_vector(*tmp2);
      }
    using Preconditioner =
      GMG<dim,
          NumberPreconditioner,
          SystemMatrix<NumberPreconditioner,
                       MatrixFreeOperator<dim, NumberPreconditioner>>>;
    auto preconditioner = std::make_unique<Preconditioner>(timer,
                                                           parameters,
                                                           fe_degree,
                                                           n_timesteps_at_once,
                                                           mg_type_level,
                                                           dof_handler,
                                                           mg_dof_handlers,
                                                           mg_constraints,
                                                           mg_operators,
                                                           precondition_vanka,
                                                           std::move(tmp1),
                                                           std::move(tmp2));
    preconditioner->reinit();
    //
    /// GMG

    std::unique_ptr<Function<dim, Number>> rhs_function;
    std::unique_ptr<Function<dim, Number>> exact_solution, exact_solution_v;
    if (parameters.space_time_conv_test)
      {
        exact_solution =
          std::make_unique<ExactSolution<dim, Number>>(frequency);
        if (parameters.problem == ProblemType::wave)
          {
            rhs_function =
              std::make_unique<wave::RHSFunction<dim, Number>>(frequency);
            exact_solution_v =
              std::make_unique<wave::ExactSolutionV<dim, Number>>(frequency);
          }
        else
          {
            rhs_function =
              std::make_unique<RHSFunction<dim, Number>>(frequency);
          }
      }
    else
      {
        exact_solution = std::make_unique<Functions::CutOffFunctionCinfty<dim>>(
          1.e-2, parameters.source, 1, numbers::invalid_unsigned_int, true);
        rhs_function = std::make_unique<Functions::ZeroFunction<dim, Number>>();
        exact_solution_v =
          std::make_unique<Functions::ZeroFunction<dim, Number>>();
      }
    auto integrate_rhs_function =
      [&mapping, &dof_handler, &quad, &rhs_function, &constraints, &parameters](
        const double time, VectorType &rhs) -> void {
      rhs_function->set_time(time);
      rhs = 0.0;
      if (parameters.space_time_conv_test)
        VectorTools::create_right_hand_side(
          mapping, dof_handler, quad, *rhs_function, rhs, constraints);
    };
    auto evaluate_exact_solution = [&mapping,
                                    &dof_handler,
                                    &exact_solution,
                                    &parameters](const double time,
                                                 VectorType  &tmp) -> void {
      exact_solution->set_time(time);
      VectorTools::interpolate(mapping, dof_handler, *exact_solution, tmp);
    };
    auto evaluate_exact_v_solution = [&mapping,
                                      &dof_handler,
                                      &exact_solution_v,
                                      &parameters](const double time,
                                                   VectorType  &tmp) -> void {
      exact_solution_v->set_time(time);
      VectorTools::interpolate(mapping, dof_handler, *exact_solution_v, tmp);
    };
    auto evaluate_numerical_solution =
      [&constraints, &basis, &is_cgp](const double           time,
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
              if (!is_cgp)
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

    BlockVectorType x(n_blocks), v(n_blocks);
    for (unsigned int i = 0; i < n_blocks; ++i)
      matrix->initialize_dof_vector(x.block(i));
    VectorType prev_x, prev_v;
    matrix->initialize_dof_vector(prev_x);
    if (parameters.problem == ProblemType::wave)
      {
        matrix->initialize_dof_vector(prev_v);
        for (unsigned int i = 0; i < n_blocks; ++i)
          matrix->initialize_dof_vector(v.block(i));
      }
    // Point eval
    auto real_points = dim == 2 ?
                         std::vector<Point<dim, Number>>{{0.75, 0}} :
                         std::vector<Point<dim, Number>>{{0.75, 0, 0},
                                                         {0, 0, 0.75},
                                                         {0.75, 0.1, 0.75}};

    Utilities::MPI::RemotePointEvaluation<dim, dim> rpe;
    rpe.reinit(real_points, tria, mapping);
    unsigned int i_eval_f          = 0;
    auto const   evaluate_function = [&](const ArrayView<Number> &values,
                                       const auto              &cell_data) {
      unsigned int              ii = i_eval_f;
      FEPointEvaluation<1, dim> fe_point(mapping, fe, update_values);
      std::vector<Number>       local_values;
      for (const auto cell : cell_data.cell_indices())
        {
          auto const cell_dofs =
            cell_data.get_active_cell_iterator(cell)->as_dof_handler_iterator(
              dof_handler);
          auto const unit_points = cell_data.get_unit_points(cell);
          auto const local_value = cell_data.get_data_view(cell, values);
          local_values.resize(cell_dofs->get_fe().n_dofs_per_cell());
          cell_dofs->get_dof_values(x.block(ii),
                                    local_values.begin(),
                                    local_values.end());

          fe_point.reinit(cell_dofs, unit_points);
          fe_point.evaluate(local_values, EvaluationFlags::values);

          for (unsigned int q = 0; q < unit_points.size(); ++q)
            local_value[q] = fe_point.get_value(q);
        }
    };


#ifdef DEBUG
    VectorType exact;
    matrix->initialize_dof_vector(exact);
#endif
    VectorType numeric;
    matrix->initialize_dof_vector(numeric);

    unsigned int                 timestep_number = 0;
    ErrorCalculator<dim, Number> error_calculator(parameters.type,
                                                  fe_degree,
                                                  fe_degree,
                                                  mapping,
                                                  dof_handler,
                                                  *exact_solution,
                                                  evaluate_numerical_solution);

    std::unique_ptr<TimeIntegrator<dim, Number, Preconditioner>> step;
    if (parameters.problem == ProblemType::heat)
      step = std::make_unique<TimeIntegratorHeat<dim, Number, Preconditioner>>(
        parameters.type,
        fe_degree,
        Alpha_1,
        Gamma_1,
        1.e-12,
        *matrix,
        *preconditioner,
        *rhs_matrix,
        integrate_rhs_function,
        n_timesteps_at_once,
        parameters.extrapolate);
    else
      step = std::make_unique<TimeIntegratorWave<dim, Number, Preconditioner>>(
        parameters.type,
        fe_degree,
        Alpha_1,
        Beta_1,
        Gamma_1,
        Zeta_1,
        1.e-12,
        *matrix,
        *preconditioner,
        *rhs_matrix,
        *rhs_matrix_v,
        integrate_rhs_function,
        n_timesteps_at_once,
        parameters.extrapolate);

    // interpolate initial value
    evaluate_exact_solution(0, x.block(x.n_blocks() - 1));
    if (parameters.problem == ProblemType::wave)
      evaluate_exact_v_solution(0, v.block(v.n_blocks() - 1));
    double           l2 = 0., l8 = -1., h1_semi = 0.;
    constexpr double qNaN           = std::numeric_limits<double>::quiet_NaN();
    bool const       st_convergence = parameters.space_time_conv_test;
    int              i = 0, total_gmres_iterations = 0;

    unsigned int samples_per_interval = (fe_degree + 1) * (fe_degree + 1);
    double       sample_step          = 1.0 / (samples_per_interval - 1);
    i_eval_f                          = x.n_blocks() - 1;
    x.block(i_eval_f).update_ghost_values();
    std::vector<Number> output_point_evaluation =
      rpe.template evaluate_and_process<Number>(evaluate_function);
    x.block(i_eval_f).zero_out_ghost_values();
    if (!rpe.is_map_unique())
      {
        auto const         &point_indices = rpe.get_point_ptrs();
        std::vector<Number> new_output;
        new_output.reserve(point_indices.size() - 1);
        for (auto el : point_indices)
          if (el < output_point_evaluation.size())
            new_output.push_back(output_point_evaluation[el]);

        output_point_evaluation.swap(new_output);
      }

    std::vector<Number> prev_output_pt_eval = output_point_evaluation;
    FullMatrix<Number>  output_pt_eval(fe_degree + 1, real_points.size());
    FullMatrix<Number>  time_evaluator =
      get_time_evaluation_matrix<Number>(basis, samples_per_interval);
    FullMatrix<Number> output_pt_eval_res(samples_per_interval,
                                          real_points.size());
    auto const         do_point_evaluation = [&]() {
      Assert(output_pt_eval.n() >= prev_output_pt_eval.size(),
             ExcLowerRange(output_pt_eval.n(), prev_output_pt_eval.size()));
      for (unsigned int it = 0; it < n_timesteps_at_once; ++it)
        {
          if (is_cgp)
            std::copy_n(prev_output_pt_eval.begin(),
                        prev_output_pt_eval.size(),
                        output_pt_eval.begin(0));
          for (unsigned int t_dof = 0; t_dof < nt_dofs; ++t_dof)
            {
              i_eval_f = it * nt_dofs + t_dof;
              x.block(i_eval_f).update_ghost_values();
              output_point_evaluation =
                rpe.template evaluate_and_process<Number>(evaluate_function);
              if (!rpe.is_map_unique())
                {
                  auto const         &point_indices = rpe.get_point_ptrs();
                  std::vector<Number> new_output;
                  new_output.reserve(point_indices.size() - 1);
                  for (auto el : point_indices)
                    if (el < output_point_evaluation.size())
                      new_output.push_back(output_point_evaluation[el]);

                  output_point_evaluation.swap(new_output);
                }
              Assert(output_pt_eval.m() > t_dof + is_cgp,
                     ExcLowerRange(output_pt_eval.m(), t_dof + is_cgp));
              std::copy_n(output_point_evaluation.begin(),
                          output_point_evaluation.size(),
                          output_pt_eval.begin(t_dof + is_cgp));
              x.block(i_eval_f).zero_out_ghost_values();
            }
          time_evaluator.mmult(output_pt_eval_res, output_pt_eval);
          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            {
              std::ofstream file(parameters.functional_file, std::ios::app);
              for (unsigned int row = 0; row < output_pt_eval_res.m(); ++row)
                {
                  double t_ = time + time_step_size * (it + row * sample_step);
                  file << std::setw(16) << std::scientific << t_;
                  for (unsigned int c = 0; c < output_pt_eval_res.n(); ++c)
                    file << std::setw(16) << std::scientific << " "
                         << output_pt_eval_res(row, c);
                  file << '\n';
                }
              file << '\n';
            }

          prev_output_pt_eval = output_point_evaluation;
        }
    };
    auto const data_output = [&](VectorType const &v, std::string const &name) {
      DataOut<dim> data_out;
      data_out.attach_dof_handler(dof_handler);
      data_out.add_data_vector(v, "u");
      data_out.build_patches();
      data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));
      data_out.write_vtu_with_pvtu_record(
        "./", name, timestep_number, tria.get_communicator(), 4);
    };

    while (time < parameters.end_time)
      {
        TimerOutput::Scope scope(timer, "step");

        ++timestep_number;
        dealii::deallog << "Step " << timestep_number << " t = " << time
                        << std::endl;
        prev_x = x.block(x.n_blocks() - 1);
        if (parameters.problem == ProblemType::heat)
          static_cast<TimeIntegratorHeat<dim, Number, Preconditioner> const *>(
            step.get())
            ->solve(x, prev_x, timestep_number, time, time_step_size);
        else
          {
            prev_v = v.block(v.n_blocks() - 1);
            static_cast<
              TimeIntegratorWave<dim, Number, Preconditioner> const *>(
              step.get())
              ->solve(
                x, v, prev_x, prev_v, timestep_number, time, time_step_size);
          }
        total_gmres_iterations += step->last_step();
        for (unsigned int i = 0; i < n_blocks; ++i)
          constraints.distribute(x.block(i));
        if (st_convergence)
          {
            auto error_on_In = error_calculator.evaluate_error(
              time, time_step_size, x, prev_x, n_timesteps_at_once);
            l2 += error_on_In[VectorTools::L2_norm];
            l8 = std::max(error_on_In[VectorTools::Linfty_norm], l8);
            h1_semi += error_on_In[VectorTools::H1_seminorm];
          }
        else
          do_point_evaluation();

        time += n_timesteps_at_once * time_step_size;
        ++i;

        if (do_output)
          {
            numeric = 0.0;
            evaluate_numerical_solution(
              1.0, numeric, x, prev_x, (n_timesteps_at_once - 1) * nt_dofs);
            data_output(numeric, "solution");
          }
#ifdef DEBUG
        if (do_output && st_convergence)
          {
            exact = 0.0;
            evaluate_exact_solution(time, exact);
            data_output(exact, "exact");
          }
#endif
      }
    double average_gmres_iter = static_cast<double>(total_gmres_iterations) /
                                static_cast<double>(timestep_number);
    pcout << "Average GMRES iterations " << average_gmres_iter << " ("
          << total_gmres_iterations << " gmres_iterations / " << timestep_number
          << " timesteps)\n"
          << std::endl;
    if (print_timing)
      timer.print_wall_time_statistics(MPI_COMM_WORLD);

    auto const   n_active_cells = tria.n_global_active_cells();
    size_t const n_dofs         = static_cast<size_t>(dof_handler.n_dofs());
    size_t const st_dofs        = i * n_dofs * n_blocks;
    size_t const work           = n_dofs * n_blocks * total_gmres_iterations;
    table.add_value("cells", n_active_cells);
    table.add_value("s-dofs", n_dofs);
    table.add_value("t-dofs", n_blocks);
    table.add_value("st-dofs", st_dofs);
    table.add_value("work", work);
    table.add_value("L\u221E-L\u221E", st_convergence ? l8 : qNaN);
    table.add_value("L2-L2", st_convergence ? std::sqrt(l2) : qNaN);
    table.add_value("L2-H1_semi", st_convergence ? std::sqrt(h1_semi) : qNaN);
    itable.add_value(std::to_string(refinement), average_gmres_iter);
  };
  auto const [k, d_cyc, r_cyc, r] = std::visit(
    [](auto const &p) {
      return std::make_tuple(p.fe_degree,
                             p.n_deg_cycles,
                             p.n_ref_cycles,
                             p.refinement);
    },
    parameters);

  for (unsigned int j = k; j < k + d_cyc; ++j)
    {
      itable.add_value("k \\ r", j);
      for (unsigned int i = r; i < r + r_cyc; ++i)
        if (dim == 2)
          convergence_test(i, j, std::get<Parameters<2>>(parameters));
        else
          convergence_test(i, j, std::get<Parameters<3>>(parameters));

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
      pcout << "Convergence table k=" << j << std::endl;
      if (pcout.is_active())
        table.write_text(pcout.get_stream());
      pcout << std::endl;
      table.clear();
    }
  pcout << "Iteration count table\n";
  if (pcout.is_active())
    itable.write_text(pcout.get_stream());
  pcout << std::endl;
}



int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  dealii::ConditionalOStream       pcout(
    std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  std::string filename =
    "proc" +
    std::to_string(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)) +
    ".log";
  std::ofstream pout(filename);
  dealii::deallog.attach(pout);
  dealii::deallog.depth_console(0);
  MPI_Comm    comm               = MPI_COMM_WORLD;
  std::string file               = "default";
  int         dim                = 2;
  bool        precondition_float = true;
  {
    namespace arg_t = util::arg_type;
    util::cl_options clo(argc, argv);
    clo.insert(file, "file", arg_t::required, 'f', "Path to parameterfile");
    clo.insert(dim, "dim", arg_t::required, 'd', "Spatial dimensions");
    clo.insert(precondition_float, "precondition_float", arg_t::none, 'p');
  }
  auto tst = [&](std::string file_name) {
    if (precondition_float)
      test<double, float>(pcout, comm, file_name, dim);
    else
      test<double, double>(pcout, comm, file_name, dim);
  };
  if (file == "default")
    {
#ifdef TESTDIRECTORY
      std::string test_dir(TESTDIRECTORY);
#else
      Assert(
        false,
        ExcMessage(
          "If TESTDIRECTORY is not defined parameter file have to be provided"));
      std::exit();
#endif
      std::vector<std::pair<std::string, std::string>> tests = {
        {"HEAT 2 steps at once DG\n", test_dir + "json/tf01.json"},
        {"", test_dir + "json/tf02.json"},
        {"HEAT single step\n", test_dir + "json/tf03.json"},
        {"", test_dir + "json/tf04.json"},
        {"WAVE 4 steps at once\n", test_dir + "json/tf05.json"},
        {"", test_dir + "json/tf06.json"},
        {"WAVE single step\n", test_dir + "json/tf07.json"},
        {"", test_dir + "json/tf08.json"}};
      for (const auto &[header, file_name] : tests)
        {
          dealii::deallog << header;
          tst(file_name);
        }
    }
  else
    tst(file);

  dealii::deallog << std::endl;
  pcout << std::endl;
}
