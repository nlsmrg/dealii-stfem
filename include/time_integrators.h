// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by Nils Margenberg and Peter Munch

#pragma once

#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>

#include "newton.h"
#include "operators.h"
#include "stmg.h"
#include "types.h"

namespace dealii
{
  struct EmptyNitsche
  {};

  /** Time stepping by DG and CGP variational time discretizations
   *
   * This time integrator is suited for linear problems. For nonlinear problems
   * we would need a few extensions in order to integrate nonlinear terms
   * accurately.
   */
  template <int dim,
            typename Number,
            typename Preconditioner,
            typename System,
            typename RHSSystem,
            typename NitscheIntegrator = EmptyNitsche>
  class TimeIntegrator
  {
  public:
    using VectorType      = VectorT<Number>;
    using BlockVectorType = BlockVectorT<Number>;
    using NonlinearSolver = Newton<dim, Number, BlockVectorType, System>;

    TimeIntegrator(
      TimeStepType              type_,
      unsigned int              time_degree_,
      FullMatrix<Number> const &Alpha_,
      FullMatrix<Number> const &Gamma_,
      double const              gmres_tolerance_,
      System const             &matrix_,
      Preconditioner const     &preconditioner_,
      RHSSystem const          &rhs_matrix_,
      std::vector<std::function<void(const double, VectorType &)>>
                               integrate_rhs_function,
      unsigned int             n_timesteps_at_once_,
      bool                     extrapolate_,
      const NitscheIntegrator &nitsche_             = NitscheIntegrator(),
      NonlinearTreatment       nonlinear_treatment_ = NonlinearTreatment::None,
      double                   abstol               = 1.e-12)
      : type(type_)
      , time_degree(time_degree_)
      , quad_time(get_time_quad(type, time_degree))
      , Alpha(Alpha_)
      , Gamma(Gamma_)
      , solver_control(nonlinear_treatment_ == NonlinearTreatment::Implicit ?
                         50 :
                         200,
                       abstol,
                       gmres_tolerance_,
                       false,
                       true)
      , solver(solver_control,
               typename SolverFGMRES<BlockVectorType>::AdditionalData::
                 AdditionalData(
                   nonlinear_treatment_ == NonlinearTreatment::Implicit ? 50 :
                                                                          100))
      , preconditioner(preconditioner_)
      , matrix(matrix_)
      , rhs_matrix(rhs_matrix_)
      , nitsche(nitsche_)
      , integrate_rhs_function(integrate_rhs_function)
      , n_timesteps_at_once(n_timesteps_at_once_)
      , idx(n_timesteps_at_once,
            integrate_rhs_function.size(),
            (type == TimeStepType::DG ? time_degree + 1 : time_degree))
      , dirichlet_color_to_fun(idx.n_variables())
      , do_extrapolate(extrapolate_)
      , nonlinear_treatment(nonlinear_treatment_)
    {}

    void
    assemble_force(BlockVectorType &rhs,
                   double const     time,
                   double const     time_step) const
    {
      AssertDimension(rhs.n_blocks(), idx.n_blocks());
      BlockVectorType tmp(integrate_rhs_function.size());
      for (unsigned int i = 0; i < integrate_rhs_function.size(); ++i)
        matrix.initialize_dof_vector(tmp.block(i), i);

      for (unsigned int it = 0; it < idx.n_timesteps_at_once(); ++it)
        for (unsigned int j = 0; j < quad_time.size(); ++j)
          {
            double time_ =
              time + time_step * it + time_step * quad_time.point(j)[0];
            for (unsigned int iv = 0; iv < idx.n_variables(); ++iv)
              {
                integrate_rhs_function[iv](time_, tmp.block(iv));
                /// Here we exploit that Alpha is a diagonal matrix
                if (type == TimeStepType::DG)
                  rhs.block(idx.index(it, iv, j))
                    .add(Alpha(idx.index(0, iv, j), idx.index(0, iv, j)),
                         tmp.block(iv));
                else
                  {
                    if (j == 0)
                      for (unsigned int i = 0; i < idx.n_timedofs(); ++i)
                        rhs.block(idx.index(it, iv, i))
                          .add(-Gamma(idx.index(0, iv, i), 0), tmp.block(iv));
                    else
                      rhs.block(idx.index(it, iv, j - 1))
                        .add(Alpha(idx.index(0, iv, j - 1),
                                   idx.index(0, iv, j - 1)),
                             tmp.block(iv));
                  }
              }
          }
    }

    void
    add_dirichlet_function(unsigned int           var,
                           types::boundary_id     b_id,
                           Function<dim, Number> *f) const
    {
      dirichlet_color_to_fun[var].emplace(b_id, f);
      dirichlet_function_added = true;
    }

    template <typename T = NitscheIntegrator>
    typename std::enable_if<std::is_same<T, EmptyNitsche>::value>::type
    assemble_nitsche(BlockVectorType &, double const, double const) const
    {}

    template <typename T = NitscheIntegrator>
    typename std::enable_if<!std::is_same<T, EmptyNitsche>::value>::type
    assemble_nitsche(BlockVectorType &rhs,
                     double const     time_,
                     double const     time_step) const
    {
      if (!dirichlet_function_added)
        return;
      AssertDimension(rhs.n_blocks(), idx.n_blocks());
      BlockVectorType tmp(idx.n_variables());
      nitsche.initialize_dof_vector(tmp);
      for (unsigned int it = 0; it < idx.n_timesteps_at_once(); ++it)
        for (unsigned int j = 0; j < quad_time.size(); ++j)
          {
            double time =
              time_ + time_step * it + time_step * quad_time.point(j)[0];
            for (unsigned int iv = 0; iv < idx.n_variables(); ++iv)
              {
                if (dirichlet_color_to_fun[iv].empty())
                  continue;
                for (auto &[bid, f] : dirichlet_color_to_fun[iv])
                  f->set_time(time);
                nitsche.set_dirichlet_functions(dirichlet_color_to_fun[iv]);

                nitsche.vmult(tmp);
                if (type == TimeStepType::DG)
                  for (unsigned int jv = 0; jv < idx.n_variables(); ++jv)
                    rhs.block(idx.index(it, jv, j))
                      .add(Alpha(idx.index(0, iv, j), idx.index(0, jv, j)),
                           tmp.block(jv));
                else
                  for (unsigned int jv = 0; jv < idx.n_variables(); ++jv)
                    {
                      if (j == 0)
                        for (unsigned int i = 0; i < idx.n_timedofs(); ++i)
                          rhs.block(idx.index(it, jv, i))
                            .add(-Gamma(idx.index(0, iv, i), 0), tmp.block(jv));
                      else
                        rhs.block(idx.index(it, jv, j - 1))
                          .add(Alpha(idx.index(0, iv, j - 1),
                                     idx.index(0, iv, j - 1)),
                               tmp.block(jv));
                    }
              }
          }
    }

    unsigned int
    last_step() const
    {
      if (nonlinear_treatment == NonlinearTreatment::Implicit)
        return total_lin_iter;
      else
        return solver_control.last_step();
    }

    unsigned int
    nonlinear_steps() const
    {
      if (nonlinear_treatment == NonlinearTreatment::Implicit)
        return total_nonlinear_iter;
      else
        return 1u;
    }

    void
    setup_nonlinear(NewtonData const &newton_data) const
    {
      Assert(nonlinear_treatment == NonlinearTreatment::None ||
               update_linearization_user,
             ExcInternalError());
      if (nonlinear_treatment == NonlinearTreatment::Implicit)
        nonlinear_solver = std::make_unique<NonlinearSolver>(
          newton_data,
          matrix,
          this->update_linearization_user,
          this->update_preconditioner_user,
          [this](BlockVectorType &x, BlockVectorType const &rhs) {
            try
              {
                this->solver.solve(this->matrix, x, rhs, this->preconditioner);
              }
            catch (const SolverControl::NoConvergence &e)
              {
                // Let Newton handle problems
              }
            return solver_control.last_step();
          },
          [this](double tol) { solver_control.set_reduction(tol); });
    }

    std::function<void(BlockVectorType const &)> update_linearization_user;
    std::function<void()>                        update_preconditioner_user;

  protected:
    void
    update_linearization(BlockVectorType const &x) const
    {
      Assert(nonlinear_treatment == NonlinearTreatment::None ||
               update_linearization_user,
             ExcInternalError());
      if (nonlinear_treatment != NonlinearTreatment::None)
        this->update_linearization_user(x);
    }
    void
    update_preconditioner() const
    {
      ++update_calls;
      Assert(nonlinear_treatment == NonlinearTreatment::None ||
               update_preconditioner_user,
             ExcInternalError());
      if ((update_calls % update_every_step == 0) &&
          nonlinear_treatment != NonlinearTreatment::None)
        this->update_preconditioner_user();
    }

    void
    extrapolate(BlockVectorType &x, BlockVectorType const &prev_x) const
    {
      for (unsigned int it = 0; it < idx.n_timesteps_at_once(); ++it)
        for (unsigned int i = 0; i < idx.n_variables(); ++i)
          for (unsigned int j = 0; j < idx.n_timedofs(); ++j)
            if (do_extrapolate)
              x.block(idx.index(it, i, j)) = prev_x.block(i);
            else
              x.block(idx.index(it, i, j)) = 0.0;
    }

    TimeStepType         type;
    unsigned int         time_degree;
    mutable unsigned int update_calls = 0;
    unsigned int  update_every_step = std::numeric_limits<unsigned int>::max();
    Quadrature<1> quad_time;
    FullMatrix<Number> const &Alpha;
    FullMatrix<Number> const &Gamma;

    mutable ReductionControl                 solver_control;
    mutable SolverFGMRES<BlockVectorType>    solver;
    Preconditioner const                    &preconditioner;
    System const                            &matrix;
    RHSSystem const                         &rhs_matrix;
    NitscheIntegrator const                 &nitsche;
    mutable std::unique_ptr<NonlinearSolver> nonlinear_solver;

    std::vector<std::function<void(const double, VectorType &)>>
                 integrate_rhs_function;
    unsigned int n_timesteps_at_once;
    BlockSlice   idx;

    mutable std::vector<
      std::map<types::boundary_id, dealii::Function<dim, Number> *>>
                 dirichlet_color_to_fun;
    mutable bool dirichlet_function_added = false;

    bool                 do_extrapolate;
    FullMatrix<Number>   extrapolation_matrix;
    NonlinearTreatment   nonlinear_treatment;
    mutable unsigned int total_lin_iter, total_nonlinear_iter;
  };


  template <int dim,
            typename Number,
            typename Preconditioner,
            typename System,
            typename RHSSystem         = System,
            typename NitscheIntegrator = EmptyNitsche>
  class TimeIntegratorFO : public TimeIntegrator<dim,
                                                 Number,
                                                 Preconditioner,
                                                 System,
                                                 RHSSystem,
                                                 NitscheIntegrator>
  {
  public:
    using VectorType = typename TimeIntegrator<dim,
                                               Number,
                                               Preconditioner,
                                               System,
                                               RHSSystem,
                                               NitscheIntegrator>::VectorType;
    using BlockVectorType =
      typename TimeIntegrator<dim,
                              Number,
                              Preconditioner,
                              System,
                              RHSSystem,
                              NitscheIntegrator>::BlockVectorType;
    using NonlinearSolver =
      typename TimeIntegrator<dim,
                              Number,
                              Preconditioner,
                              System,
                              RHSSystem,
                              NitscheIntegrator>::NonlinearSolver;

    TimeIntegratorFO(
      TimeStepType              type_,
      unsigned int              time_degree_,
      FullMatrix<Number> const &Alpha_,
      FullMatrix<Number> const &Gamma_,
      double const              gmres_tolerance_,
      System const             &matrix_,
      Preconditioner const     &preconditioner_,
      RHSSystem const          &rhs_matrix_,
      std::vector<std::function<void(const double, VectorType &)>>
                               integrate_rhs_function,
      unsigned int             n_timesteps_at_once_,
      bool                     extrapolate          = true,
      const NitscheIntegrator &nitsche              = NitscheIntegrator(),
      NonlinearTreatment       nonlinear_treatment_ = NonlinearTreatment::None,
      double                   abstol               = 1.e-12)
      : TimeIntegrator<dim,
                       Number,
                       Preconditioner,
                       System,
                       RHSSystem,
                       NitscheIntegrator>(type_,
                                          time_degree_,
                                          Alpha_,
                                          Gamma_,
                                          gmres_tolerance_,
                                          matrix_,
                                          preconditioner_,
                                          rhs_matrix_,
                                          integrate_rhs_function,
                                          n_timesteps_at_once_,
                                          extrapolate,
                                          nitsche,
                                          nonlinear_treatment_,
                                          abstol)
    {}

    void
    solve(BlockVectorType   &x,
          const VectorType  &prev_x,
          BlockVectorType   &rhs,
          const unsigned int timestep_number,
          const double       time,
          const double       time_step) const
    {
      AssertDimension(this->idx.n_variables(), 1);
      BlockVectorType prev_x_(1);
      prev_x_.block(0).swap(const_cast<VectorType &>(prev_x));
      this->solve(x, prev_x_, rhs, timestep_number, time, time_step);
      prev_x_.block(0).swap(const_cast<VectorType &>(prev_x));
    }

    void
    solve(BlockVectorType &x,
          BlockVectorType &prev_x,
          BlockVectorType &rhs,
          const unsigned int,
          const double time,
          const double time_step) const
    {
      if (this->type == TimeStepType::CGP &&
          this->nonlinear_treatment != NonlinearTreatment::None)
        this->rhs_matrix.set_data(prev_x);
      this->rhs_matrix.vmult_slice(rhs, prev_x);
      this->assemble_nitsche(rhs, time, time_step);
      this->assemble_force(rhs, time, time_step);
      this->matrix.set_rhs(rhs);
      this->extrapolate(x, prev_x);
      this->update_linearization(x);
      this->update_preconditioner();
      if (this->nonlinear_treatment == NonlinearTreatment::Implicit)
        {
          auto [nl_iter, l_iter]     = this->nonlinear_solver->solve(x, rhs);
          this->total_lin_iter       = l_iter;
          this->total_nonlinear_iter = nl_iter;
        }
      else
        try
          {
            this->solver.solve(this->matrix, x, rhs, this->preconditioner);
          }
        catch (const SolverControl::NoConvergence &e)
          {
            // AssertThrow(false, ExcMessage(e.what()));
          }
    }


    void
    solve(BlockVectorType   &x,
          BlockVectorType   &prev_x,
          const unsigned int timestep_number,
          const double       time,
          const double       time_step) const
    {
      BlockVectorType rhs(x.n_blocks());
      for (unsigned int j = 0; j < rhs.n_blocks(); ++j)
        rhs.block(j).reinit(x.block(j).get_partitioner());
      this->solve(x, prev_x, rhs, timestep_number, time, time_step);
    }
  };

  template <int dim,
            typename Number,
            typename Preconditioner,
            typename System,
            typename RHSSystem = System>
  class TimeIntegratorWave
    : public TimeIntegrator<dim, Number, Preconditioner, System, RHSSystem>
  {
  public:
    using VectorType =
      typename TimeIntegrator<dim, Number, Preconditioner, System, RHSSystem>::
        VectorType;
    using BlockVectorType =
      typename TimeIntegrator<dim, Number, Preconditioner, System, RHSSystem>::
        BlockVectorType;

    TimeIntegratorWave(
      TimeStepType              type_,
      unsigned int              time_degree_,
      FullMatrix<Number> const &Alpha_,
      FullMatrix<Number> const &Beta_,
      FullMatrix<Number> const &Gamma_,
      FullMatrix<Number> const &Zeta_,
      double const              gmres_tolerance_,
      System const             &matrix_,
      Preconditioner const     &preconditioner_,
      RHSSystem const          &rhs_matrix_,
      RHSSystem const          &rhs_matrix_v_,
      std::vector<std::function<void(const double, VectorType &)>>
                   integrate_rhs_function,
      unsigned int n_timesteps_at_once_,
      bool         extrapolate = true)
      : TimeIntegrator<dim, Number, Preconditioner, System, RHSSystem>(
          type_,
          time_degree_,
          Alpha_,
          Gamma_,
          gmres_tolerance_,
          matrix_,
          preconditioner_,
          rhs_matrix_,
          integrate_rhs_function,
          n_timesteps_at_once_,
          extrapolate)
      , rhs_matrix_v(rhs_matrix_v_)
      , Beta(Beta_)
      , Zeta(Zeta_)
      , Alpha_inv(this->Alpha)
      , AixB(this->Alpha.m(), this->Alpha.n())
      , AixG(this->Alpha.m(), this->Gamma.n())
      , AixZ(this->Alpha.m(), this->Zeta.n())
    {
      Alpha_inv.gauss_jordan();
      Alpha_inv.mmult(this->AixB, this->Beta);
      Alpha_inv.mmult(this->AixG, this->Gamma);
      Alpha_inv.mmult(this->AixZ, this->Zeta);
      if (this->type == TimeStepType::DG)
        AixG *= -1.0;
      else
        AixZ *= -1.0;
    }

    void
    solve(BlockVectorType  &u,
          BlockVectorType  &v,
          BlockVectorType  &rhs,
          VectorType const &prev_u,
          VectorType const &prev_v,
          const unsigned int,
          const double time,
          const double time_step) const
    {
      BlockVectorType prev_(1);
      prev_.block(0).swap(const_cast<VectorType &>(prev_u));
      this->rhs_matrix.vmult_slice(rhs, prev_);
      this->extrapolate(u, prev_);
      prev_.block(0).swap(const_cast<VectorType &>(prev_u));

      prev_.block(0).swap(const_cast<VectorType &>(prev_v));
      this->rhs_matrix_v.vmult_slice_add(rhs, prev_);
      prev_.block(0).swap(const_cast<VectorType &>(prev_v));

      this->assemble_force(rhs, time, time_step);
      this->matrix.set_rhs(rhs);
      try
        {
          this->solver.solve(this->matrix, u, rhs, this->preconditioner);
        }
      catch (const SolverControl::NoConvergence &e)
        {
          AssertThrow(false, ExcMessage(e.what()));
        }
      unsigned int nt_dofs = AixB.m();
      v                    = 0.0;
      for (unsigned int it = 0; it < this->n_timesteps_at_once; ++it)
        {
          VectorType const &prev_u_ =
            it == 0 ? prev_u : u.block(it * nt_dofs - 1);
          tensorproduct_add(v, AixB, u, it * nt_dofs);
          if (this->type == TimeStepType::DG)
            tensorproduct_add(v, AixG, prev_u_, it * nt_dofs);
          else
            {
              VectorType const &prev_v_ =
                it == 0 ? prev_v : v.block(it * nt_dofs - 1);
              tensorproduct_add(v, AixG, prev_v_, it * nt_dofs);
              tensorproduct_add(v, AixZ, prev_u_, it * nt_dofs);
            }
        }
    }

  private:
    RHSSystem const &rhs_matrix_v;

    FullMatrix<Number> const &Beta;
    FullMatrix<Number> const &Zeta;

    FullMatrix<Number> Alpha_inv;
    FullMatrix<Number> AixB;
    FullMatrix<Number> AixG;
    FullMatrix<Number> AixZ;
  };
} // namespace dealii
