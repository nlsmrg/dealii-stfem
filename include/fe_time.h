// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by Nils Margenberg and Peter Munch

#pragma once

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>

#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

#include <array>

#include "types.h"

enum class TimeStepType : unsigned int
{
  CGP = 1,
  DG  = 2,
  GCC = 3
};
static std::unordered_map<std::string, TimeStepType> const str_to_time_type = {
  {"CGP", TimeStepType::CGP},
  {"DG", TimeStepType::DG},
  {"GCC", TimeStepType::GCC}};

enum class MGType : char
{
  tau = 't',
  k   = 'k',
  h   = 'h',
  p   = 'p',
};

namespace dealii
{

  bool
  is_space_lvl(MGType mg);
  bool
  is_time_lvl(MGType mg);

  template <typename Number>
  std::array<FullMatrix<Number>, 3>
  get_dg_weights(unsigned int const r, double const delta0 = 0.0);

  template <typename Number>
  std::array<FullMatrix<Number>, 2>
  get_cg_weights(unsigned int const r, double const delta0 = 0.0);

  // version in deal.II only goes to 1
  unsigned int
  create_next_polynomial_coarsening_degree(
    const unsigned int previous_fe_degree,
    const MGTransferGlobalCoarseningTools::PolynomialCoarseningSequenceType
                &p_sequence,
    unsigned int k_min = 0u);

  std::vector<unsigned int>
  get_poly_mg_sequence(
    unsigned int const k_max,
    unsigned int const k_min,
    MGTransferGlobalCoarseningTools::PolynomialCoarseningSequenceType const
      p_seq);

  template <int dim, typename... fe_types>
  std::vector<std::vector<std::unique_ptr<FiniteElement<dim>>>>
  get_fe_pmg_sequence(
    const std::vector<unsigned int>                     &pmg_sequence,
    const std::array<unsigned int, sizeof...(fe_types)> &multiplicities,
    const fe_types &...finite_elements)
  {
    Assert(!pmg_sequence.empty(),
           dealii::ExcMessage("The pmg_sequence must not be empty."));
    std::vector<std::vector<std::unique_ptr<FiniteElement<dim>>>> fe_pmg(
      pmg_sequence.size());
    std::array<int, sizeof...(finite_elements)> fe_degrees = {
      {static_cast<int>(finite_elements.tensor_degree())...}};
    std::array<int, sizeof...(fe_types)> strides;
    std::transform(fe_degrees.begin(),
                   fe_degrees.end(),
                   strides.begin(),
                   [pmg_base = pmg_sequence.back()](int degree) {
                     return degree - pmg_base;
                   });
    for (int l = fe_pmg.size() - 1; l >= 0; --l)
      {
        fe_pmg[l].reserve(sizeof...(finite_elements));
        unsigned int i = 0;
        (..., [&] {
          if (multiplicities[i] == 1)
            fe_pmg[l].emplace_back(std::make_unique<std::decay_t<fe_types>>(
              pmg_sequence[l] + strides[i]));
          else
            fe_pmg[l].emplace_back(std::make_unique<FESystem<dim, dim>>(
              std::decay_t<fe_types>(pmg_sequence[l] + strides[i]),
              multiplicities[i]));
          ++i;
        }());
        Assert(fe_pmg[l].size() == sizeof...(finite_elements),
               ExcInternalError());
      }

    return fe_pmg;
  }

  std::vector<MGType>
  get_mg_sequence(
    unsigned int              n_sp_lvl,
    std::vector<unsigned int> k_seq,
    std::vector<unsigned int> p_seq,
    unsigned int const        n_timesteps_at_once,
    unsigned int const        n_timesteps_at_once_min = 1,
    MGType                    lower_lvl               = MGType::k,
    CoarseningType            coarsening_type = CoarseningType::space_and_time,
    bool                      time_before_space     = false,
    bool                      use_p_multigrid_space = false,
    bool                      zip_from_back         = true);

  std::vector<unsigned int>
  get_precondition_stmg_types(
    std::vector<MGType> const &mg_type_level,
    CoarseningType             coarsening_type,
    bool                       time_before_space,
    [[maybe_unused]] bool      zip_from_back,
    SupportedSmoothers         smoother = SupportedSmoothers::Relaxation);

  template <int dim>
  std::vector<std::shared_ptr<const Triangulation<dim>>>
  get_space_time_triangulation(
    std::vector<MGType> const &mg_type_level,
    std::vector<std::shared_ptr<const Triangulation<dim>>> const
      &mg_triangulations)
  {
    AssertDimension(mg_triangulations.size() - 1,
                    std::count(mg_type_level.begin(),
                               mg_type_level.end(),
                               MGType::h));
    std::vector<std::shared_ptr<const Triangulation<dim>>>
         new_mg_triangulations(1 + mg_type_level.size());
    auto mg_tria_it              = mg_triangulations.rbegin();
    new_mg_triangulations.back() = *mg_tria_it;

    unsigned int ii = mg_type_level.size() - 1;
    for (auto mgt = mg_type_level.rbegin(); mgt != mg_type_level.rend();
         ++mgt, --ii)
      {
        if (*mgt == MGType::h)
          ++mg_tria_it;
        new_mg_triangulations.at(ii) = *mg_tria_it;
      }
    return new_mg_triangulations;
  }

  template <typename Number>
  std::array<FullMatrix<Number>, 5>
  get_fe_time_weights_wave(TimeStepType              type,
                           FullMatrix<Number> const &Alpha,
                           FullMatrix<Number> const &Beta,
                           FullMatrix<Number> const &Gamma,
                           FullMatrix<Number> const &Zeta,
                           unsigned int              n_timesteps_at_once = 1)
  {
    FullMatrix<Number> Alpha_inv(Alpha.m(), Alpha.n());
    Alpha_inv.invert(Alpha);

    FullMatrix<Number> BxAixB(Alpha.m(), Alpha.n());
    // Note that the order of the arguments is not the order of the mmults:
    // Alpha_inv, Beta, Beta corresponds to Beta*Alpha_inv*Beta
    // Generally A,B,D -> B*A*D.
    BxAixB.triple_product(Alpha_inv, Beta, Beta);

    // u0 mass
    FullMatrix<Number> BxAixG(Gamma.m(), Gamma.n());
    BxAixG.triple_product(Alpha_inv, Beta, Gamma);
    // u00
    double gxai = Gamma(Gamma.m() - 1, 0) / Alpha(Gamma.m() - 1, Gamma.m() - 1);
    FullMatrix<Number> GxAixG = Gamma;
    GxAixG *= gxai;

    FullMatrix<Number> Beta_row(1, Beta.n());
    for (auto b = Beta.begin(Beta.m() - 1), br = Beta_row.begin(0);
         b != Beta.end(Beta.m() - 1);
         ++br, ++b)
      *br = *b;

    FullMatrix<Number> GxAixB(Gamma.m(), Beta.n());
    Gamma.mmult(GxAixB, Beta_row);
    GxAixB /= Alpha(Gamma.m() - 1, Gamma.m() - 1);



    unsigned int nt_dofs_intvl = Alpha.m();
    unsigned int nt_dofs_tot   = nt_dofs_intvl * n_timesteps_at_once;
    std::array<FullMatrix<Number>, 5> ret{
      {FullMatrix<Number>(nt_dofs_tot, nt_dofs_tot),
       FullMatrix<Number>(nt_dofs_tot, nt_dofs_tot),
       FullMatrix<Number>(nt_dofs_tot, 1),
       FullMatrix<Number>(nt_dofs_tot, 1),
       FullMatrix<Number>(nt_dofs_tot, 1)}};
    if (type == TimeStepType::CGP)
      {
        FullMatrix<Number> BxAixZ(Zeta.m(), Zeta.n());
        BxAixZ.triple_product(Alpha_inv, Beta, Zeta);

        FullMatrix<Number> ZmBxAixG = Zeta;
        ZmBxAixG.add(-1.0, BxAixG);
        FullMatrix<Number> ZmBxAixB(Gamma.m(), Beta.n());
        ZmBxAixG.mmult(ZmBxAixB, Beta_row);
        ZmBxAixB /= Alpha(Gamma.m() - 1, Gamma.m() - 1);
        double zxai = Zeta(Zeta.m() - 1, 0) / Alpha(Zeta.m() - 1, Zeta.m() - 1);

        for (unsigned int it = 0; it < n_timesteps_at_once; ++it)
          for (unsigned int jt = 0; jt <= it; ++jt)
            for (unsigned int i = 0; i < nt_dofs_intvl; ++i)
              {
                // rhs
                if (it == 0 && jt == 0)
                  {
                    ret[2](i, 0) = Gamma(i, 0);
                    ret[3](i, 0) = BxAixZ(i, 0);
                    ret[4](i, 0) = ZmBxAixG(i, 0);
                  }
                else if (jt == 0)
                  {
                    ret[3](i + it * nt_dofs_intvl, 0) =
                      -zxai * pow(gxai, it - 1) * ZmBxAixG(i, 0);
                    ret[4](i + it * nt_dofs_intvl, 0) =
                      pow(gxai, it) * ZmBxAixG(i, 0);
                  }

                if (it == jt + 1) // lower diagonal
                  {
                    ret[0](i + it * nt_dofs_intvl,
                           nt_dofs_intvl - 1 + jt * nt_dofs_intvl) =
                      -Gamma(i, 0);
                    ret[1](i + it * nt_dofs_intvl,
                           nt_dofs_intvl - 1 + jt * nt_dofs_intvl) =
                      -BxAixZ(i, 0);
                  }

                if (it == jt) // Main diagonal
                  for (unsigned int j = 0; j < nt_dofs_intvl; ++j)
                    {
                      ret[0](i + it * nt_dofs_intvl, j + it * nt_dofs_intvl) =
                        Alpha(i, j);
                      ret[1](i + it * nt_dofs_intvl, j + it * nt_dofs_intvl) =
                        BxAixB(i, j);
                    }
                else // lower triangle
                  for (unsigned int j = 0; j < nt_dofs_intvl; ++j)
                    ret[1](i + it * nt_dofs_intvl, j + jt * nt_dofs_intvl) +=
                      -pow(gxai, it - jt - 1) * ZmBxAixB(i, j) +
                      (it > 1 && it - 1 > jt && j == nt_dofs_intvl - 1 ?
                         pow(gxai, it - jt - 2) * zxai * ZmBxAixG(i, 0) :
                         0.0);
              }
      }
    else if (type == TimeStepType::DG)
      {
        for (unsigned int it = 0; it < n_timesteps_at_once; ++it)
          for (unsigned int i = 0; i < nt_dofs_intvl; ++i)
            {
              // 1st block rhs
              if (it == 0)
                {
                  ret[3](i, 0) = BxAixG(i, 0);
                  ret[4](i, 0) = Gamma(i, 0);
                }
              // 2nd block rhs
              if (it == 1)
                ret[3](nt_dofs_intvl + i, 0) = -GxAixG(i, 0);

              // 1st lower block diagonal
              if (it < n_timesteps_at_once - 1)
                for (unsigned int j = 0; j < nt_dofs_intvl; ++j)
                  {
                    ret[1](j + (it + 1) * nt_dofs_intvl,
                           i + it * nt_dofs_intvl) =
                      -GxAixB(j, i) -
                      (i == nt_dofs_intvl - 1 ? BxAixG(j, 0) : 0.0);
                  }

              // 2nd lower diagonal
              if (static_cast<int>(it) <
                    static_cast<int>(n_timesteps_at_once) - 2 &&
                  i == nt_dofs_intvl - 1)
                for (unsigned int j = 0; j < nt_dofs_intvl; ++j)
                  ret[1](j + (it + 2) * nt_dofs_intvl, i + it * nt_dofs_intvl) =
                    GxAixG(j, 0);

              // Main diagonal
              for (unsigned int j = 0; j < nt_dofs_intvl; ++j)
                {
                  ret[0](i + it * nt_dofs_intvl, j + it * nt_dofs_intvl) =
                    Alpha(i, j);
                  ret[1](i + it * nt_dofs_intvl, j + it * nt_dofs_intvl) =
                    BxAixB(i, j);
                }
            }
      }
    return ret;
  }

  template <typename Number>
  FullMatrix<Number>
  get_time_evaluation_matrix(
    std::vector<Polynomials::Polynomial<double>> const &basis,
    unsigned int                                        samples_per_interval)
  {
    double             sample_step = 1.0 / (samples_per_interval - 1);
    FullMatrix<Number> time_evaluator(samples_per_interval, basis.size());
    for (unsigned int s = 0; s < samples_per_interval; ++s)
      {
        double time_ = s * sample_step;
        auto   te    = time_evaluator.begin(s);
        for (auto const &el : basis)
          {
            *te = el.value(time_);
            ++te;
          }
      }
    return time_evaluator;
  }

  template <typename Number>
  FullMatrix<Number>
  get_time_evaluation_matrix(
    std::vector<Polynomials::Polynomial<double>> const &basis,
    std::vector<Point<1>>                               points)
  {
    FullMatrix<Number> time_evaluator(points.size(), basis.size());
    for (unsigned int s = 0; s < points.size(); ++s)
      {
        double time_ = points[s][0];
        auto   te    = time_evaluator.begin(s);
        for (auto const &el : basis)
          {
            *te = el.value(time_);
            ++te;
          }
      }
    return time_evaluator;
  }


  /** Generates the time integration weights for time continuous
   * Galerkin-Petrov discretizations or time discontinuous Galerkin
   * discretizations
   *
   * For a given order r the method returns 4 matrices. In the continouus
   * Galerkin-Petrov (CG) case these are:
   * 1. time integration weights, commonly denoted by Alpha (r x r)
   * 2. time integration weights for temporal derivatives ,
   *    commonly denoted by Beta (r x r)
   * 3. time integration weights for terms which can be put on the RHS,
   *    commonly denoded by Gamma (r x 1)
   * 4. time integration weights for temporal derivatives which can be put on
   *    the RHS, commonly denoded by Zeta (r x 1)
   *
   * In the discontinouus Galerkin (DG) case these are:
   * 1. time integration weights, commonly denoted by Alpha (r+1 x r+1)
   * 2. time integration weights for temporal derivatives ,
   *    commonly denoted by Beta (r+1 x r+1)
   * 3. time integration weights for terms which can be put on the RHS,
   *    commonly denoded by Gamma (r+1 x 1)
   * 4. the fourth matrix is zero in the DG case, as we only get jump terms in
   *    the DG case (r+1 x 1)
   */
  template <typename Number>
  std::array<FullMatrix<Number>, 4>
  get_fe_time_weights(TimeStepType       type,
                      unsigned int const r,
                      double             time_step_size,
                      unsigned int       n_timesteps_at_once = 1,
                      double             delta0              = 0.0)
  {
    std::array<FullMatrix<Number>, 4> tmp;
    if (type == TimeStepType::CGP)
      {
        tmp = split_lhs_rhs(get_cg_weights<Number>(r, delta0));
        tmp[2] *= time_step_size;
      }
    else if (type == TimeStepType::DG)
      {
        tmp    = split_lhs_rhs(get_dg_weights<Number>(r, delta0));
        tmp[3] = tmp[2];
        tmp[2] = 0.0;
      }
    tmp[0] *= time_step_size;

    unsigned int nt_dofs_intvl = tmp[0].m();
    unsigned int nt_dofs_tot   = nt_dofs_intvl * n_timesteps_at_once;
    std::array<FullMatrix<Number>, 4> ret{
      {FullMatrix<Number>(nt_dofs_tot, nt_dofs_tot),
       FullMatrix<Number>(nt_dofs_tot, nt_dofs_tot),
       FullMatrix<Number>(nt_dofs_tot, 1),
       FullMatrix<Number>(nt_dofs_tot, 1)}};

    for (unsigned int it = 0; it < n_timesteps_at_once; ++it)
      for (unsigned int i = 0; i < nt_dofs_intvl; ++i)
        {
          // First lower block diagonal
          if (it < n_timesteps_at_once - 1 && i == nt_dofs_intvl - 1)
            for (unsigned int j = 0; j < nt_dofs_intvl; ++j)
              {
                ret[0](j + (it + 1) * nt_dofs_intvl, i + it * nt_dofs_intvl) =
                  -tmp[2](j, 0);
                ret[1](j + (it + 1) * nt_dofs_intvl, i + it * nt_dofs_intvl) =
                  -tmp[3](j, 0);
              }

          // Main block diagonal
          for (unsigned int j = 0; j < nt_dofs_intvl; ++j)
            {
              ret[0](i + it * nt_dofs_intvl, j + it * nt_dofs_intvl) =
                tmp[0](i, j);
              ret[1](i + it * nt_dofs_intvl, j + it * nt_dofs_intvl) =
                tmp[1](i, j);
            }
        }
    for (unsigned int i = 0; i < nt_dofs_intvl; ++i)
      {
        ret[2](i, 0) = tmp[type == TimeStepType::CGP ? 2 : 3](i, 0);
        ret[3](i, 0) = tmp[type == TimeStepType::CGP ? 3 : 2](i, 0);
      }
    return ret;
  }

  template <
    typename Number,
    typename F = std::array<FullMatrix<Number>, 4> (
        &)(TimeStepType, unsigned int const, double, unsigned int, double)>
  std::vector<std::array<FullMatrix<Number>, 4>>
  get_fe_time_weights(TimeStepType                     type,
                      double                           time_step_size,
                      unsigned int                     n_timesteps_at_once,
                      double                           delta0,
                      std::vector<MGType> const       &mg_type_level,
                      std::vector<unsigned int> const &poly_time_sequence,
                      F const &get_fetw = get_fe_time_weights<Number>)
  {
    std::vector<std::array<FullMatrix<Number>, 4>> time_weights(
      mg_type_level.size() + 1);
    auto tw   = time_weights.rbegin();
    auto p_mg = poly_time_sequence.rbegin();
    *tw = get_fetw(type, *p_mg, time_step_size, n_timesteps_at_once, delta0);
    ++tw;
    for (auto mgt = mg_type_level.rbegin(); mgt != mg_type_level.rend();
         ++mgt, ++tw)
      {
        if (*mgt == MGType::k)
          ++p_mg;
        else if (*mgt == MGType::tau)
          n_timesteps_at_once /= 2, time_step_size *= 2;
        *tw =
          get_fetw(type, *p_mg, time_step_size, n_timesteps_at_once, delta0);
      }
    Assert(tw == time_weights.rend(), ExcInternalError());
    return time_weights;
  }

  template <typename Number>
  std::vector<std::array<FullMatrix<Number>, 5>>
  get_fe_time_weights_wave(TimeStepType                     type,
                           double                           time_step_size,
                           unsigned int                     n_timesteps_at_once,
                           double                           delta0,
                           std::vector<MGType> const       &mg_type_level,
                           std::vector<unsigned int> const &poly_time_sequence)
  {
    auto time_weights = get_fe_time_weights<Number>(type,
                                                    time_step_size,
                                                    n_timesteps_at_once,
                                                    delta0,
                                                    mg_type_level,
                                                    poly_time_sequence);
    std::vector<std::array<FullMatrix<Number>, 5>> time_weights_wave(
      mg_type_level.size() + 1);
    auto tw      = time_weights.rbegin();
    auto tw_wave = time_weights_wave.rbegin();
    *tw_wave =
      get_fe_time_weights_wave(type, (*tw)[0], (*tw)[1], (*tw)[2], (*tw)[3]);
    ++tw, ++tw_wave;
    for (auto mgt = mg_type_level.rbegin(); mgt != mg_type_level.rend();
         ++mgt, ++tw, ++tw_wave)
      {
        *tw_wave = get_fe_time_weights_wave(
          type, (*tw)[0], (*tw)[1], (*tw)[2], (*tw)[3]);
      }
    Assert(tw == time_weights.rend(), ExcInternalError());
    return time_weights_wave;
  }

  Quadrature<1>
  get_time_quad(TimeStepType type, unsigned int const r);

  std::vector<Polynomials::Polynomial<double>>
  get_time_basis(TimeStepType type, unsigned int const r);

  /** Utility function for splitting the weights into parts belonging to the RHS
   * and LHS.
   */
  template <typename Number>
  std::array<FullMatrix<Number>, 4>
  split_lhs_rhs(std::array<FullMatrix<Number>, 2> const &time_weights)
  {
    // Initialize return array of matrices
    std::array<FullMatrix<Number>, 4> ret{
      {FullMatrix<Number>(time_weights[0].m(), time_weights[0].n() - 1),
       FullMatrix<Number>(time_weights[0].m(), time_weights[0].n() - 1),
       FullMatrix<Number>(time_weights[0].m(), 1),
       FullMatrix<Number>(time_weights[0].m(), 1)}};
    // Scatter matrices
    ret[0].fill(time_weights[0], 0, 0, 0, 1);
    ret[1].fill(time_weights[1], 0, 0, 0, 1);
    ret[2].fill(time_weights[0], 0, 0, 0, 0);
    ret[3].fill(time_weights[1], 0, 0, 0, 0);
    ret[2] *= -1.0;
    ret[3] *= -1.0;
    return ret;
  }

  template <typename Number>
  std::array<FullMatrix<Number>, 4>
  split_lhs_rhs(std::array<FullMatrix<Number>, 3> time_weights)
  {
    // DG
    return {{time_weights[0],
             time_weights[1],
             time_weights[2],
             FullMatrix<Number>(time_weights[2].m(), 1)}};
  }

  template <typename Number>
  FullMatrix<Number>
  build_derivative_matrix(
    const std::vector<Polynomials::Polynomial<double>> &basis,
    const std::vector<Point<1>>                        &old_points)
  {
    const unsigned     n = basis.size();
    FullMatrix<Number> D(n, n);
    for (unsigned i = 0; i < n; ++i)
      for (unsigned j = 0; j < n; ++j)
        D(i, j) = basis[j].value(old_points[i][0]);
    return D;
  }

  template <typename Number>
  FullMatrix<Number>
  construct_extrapolation_matrix(TimeStepType type,
                                 unsigned int r,
                                 double       shift,
                                 double       gradient_penalty,
                                 double       filter_strength,
                                 bool         extrapolate_constant = false)
  {
    auto old_n_dofs = type == TimeStepType::DG ? r + 2 : r + 1;
    if (extrapolate_constant)
      {
        auto               new_n_dofs = type == TimeStepType::DG ? r + 1 : r;
        FullMatrix<double> M_extrapolate(new_n_dofs, old_n_dofs);
        for (auto i = 0u; i < new_n_dofs; ++i)
          M_extrapolate(i, old_n_dofs - 1) = 1.;
        return M_extrapolate;
      }

    auto new_basis  = get_time_basis(type, r);
    auto new_points = get_time_quad(type, r).get_points();

    std::vector<Point<1>> old_points;
    if (type == TimeStepType::DG)
      {
        old_points.push_back(Point<1>(0.0));
        auto points_ = get_time_quad(type, r).get_points();
        old_points.insert(old_points.end(), points_.begin(), points_.end());
      }
    else
      old_points = get_time_quad(type, r).get_points();
    auto old_basis = Polynomials::generate_complete_Lagrange_basis(old_points);
    auto transform = [shift](double x) { return x + shift; };
    FullMatrix<double> M_interpolate(r + 1, old_n_dofs);
    for (unsigned i = 0; i < r + 1; ++i)
      {
        double x_trans = transform(new_points[i][0]);
        for (unsigned j = 0; j < old_n_dofs; ++j)
          M_interpolate(i, j) = old_basis[j].value(x_trans);
      }

    FullMatrix<double> M_new_basis(r + 1, r + 1);
    for (unsigned i = 0; i < r + 1; ++i)
      {
        double x_new = new_points[i][0];
        for (unsigned j = 0; j < r + 1; ++j)
          M_new_basis(i, j) = new_basis[j].value(x_new);
      }
    FullMatrix<double> M_new_basis_inv = M_new_basis;
    M_new_basis_inv.invert(M_new_basis_inv);
    FullMatrix<double> M_extrapolate(r + 1, old_n_dofs);
    M_new_basis_inv.mmult(M_extrapolate, M_interpolate);


    std::vector<Polynomials::Polynomial<double>> poly_derivative(
      new_basis.size());
    for (unsigned int i = 0; i < new_basis.size(); ++i)
      poly_derivative[i] = new_basis[i].derivative();

    auto D = build_derivative_matrix<Number>(poly_derivative, old_points);
    FullMatrix<Number> D_transpose(r + 1, r + 1);
    D_transpose.copy_transposed(D);
    FullMatrix<Number> DTD(r + 1, r + 1);
    D_transpose.mmult(DTD, D);
    FullMatrix<Number> G = IdentityMatrix(r + 1);
    for (unsigned i = 0; i <= r; ++i)
      for (unsigned j = 0; j <= r; ++j)
        G(i, j) += gradient_penalty * DTD(i, j);
    FullMatrix<Number> F(r + 1, r + 1);
    F = 0.0;
    for (unsigned i = 0; i <= r; ++i)
      F(i, i) = 1.0 / (1.0 + filter_strength * i * i);

    {
      FullMatrix<Number> temp(r + 1, old_n_dofs);
      G.mmult(temp, M_extrapolate);
      F.mmult(M_extrapolate, temp);
    }

    if (type == TimeStepType::DG)
      return M_extrapolate;

    FullMatrix<Number> M_extrapolate_cg(M_extrapolate.m() - 1,
                                        M_extrapolate.n());
    M_extrapolate_cg.fill(M_extrapolate, 0, 0, 1, 0);
    return M_extrapolate_cg;
  }

  template <typename Number>
  FullMatrix<Number>
  get_extrapolation_matrix(TimeStepType           type,
                           NonlinearExtrapolation nonlinear_extra,
                           unsigned int           r,
                           double                 shift,
                           double                 gradient_penalty,
                           double                 filter_strength)
  {
    FullMatrix<Number> extrapolation;
    if (nonlinear_extra == NonlinearExtrapolation::Auto)
      {
        extrapolation = construct_extrapolation_matrix<Number>(
          type, r, shift, gradient_penalty, filter_strength, (r <= 1u));
      }
    else if (nonlinear_extra == NonlinearExtrapolation::Constant)
      extrapolation = construct_extrapolation_matrix<Number>(
        type, r, shift, gradient_penalty, filter_strength, true);
    else if (nonlinear_extra == NonlinearExtrapolation::Polynomial)
      extrapolation = construct_extrapolation_matrix<Number>(
        type, r, shift, gradient_penalty, filter_strength, false);

    return extrapolation;
  }

  template <typename Number>
  std::array<FullMatrix<Number>, 2>
  get_cg_weights(unsigned int const r, double const)
  {
    static std::unordered_map<unsigned int, std::array<FullMatrix<Number>, 2>>
      cache;
    if (auto it = cache.find(r); it != cache.end())
      return it->second;

    auto const trial_points = QGaussLobatto<1>(r + 1).get_points();
    std::vector<Point<1>> const test_points(trial_points.begin() + 1,
                                            trial_points.end());

    auto const poly_lobatto = get_time_basis(TimeStepType::CGP, r);
    auto const poly_test =
      Polynomials::generate_complete_Lagrange_basis(test_points);

    std::vector<Polynomials::Polynomial<double>> poly_lobatto_derivative(
      poly_lobatto.size());

    for (unsigned int i = 0; i < poly_lobatto.size(); ++i)
      poly_lobatto_derivative[i] = poly_lobatto[i].derivative();

    std::vector<Polynomials::Polynomial<double>> poly_test_derivative(
      poly_test.size());

    for (unsigned int i = 0; i < poly_test_derivative.size(); ++i)
      poly_test_derivative[i] = poly_test[i].derivative();


    QGauss<1> const    quad(r + 2);
    FullMatrix<Number> matrix(r, r + 1);
    FullMatrix<Number> matrix_der(r, r + 1);

    // Later multiply with tau_n
    for (unsigned int i = 0; i < r; ++i)
      for (unsigned int j = 0; j < r + 1; ++j)
        for (unsigned int q = 0; q < quad.size(); ++q)
          matrix(i, j) += quad.weight(q) *
                          poly_test[i].value(quad.point(q)[0]) *
                          poly_lobatto[j].value(quad.point(q)[0]);


    for (unsigned int i = 0; i < r; ++i)
      for (unsigned int j = 0; j < r + 1; ++j)
        {
          for (unsigned int q = 0; q < quad.size(); ++q)
            matrix_der(i, j) +=
              quad.weight(q) * poly_test[i].value(quad.point(q)[0]) *
              poly_lobatto_derivative[j].value(quad.point(q)[0]);
        }
    cache[r] = {{matrix, matrix_der}};
    return cache[r];
  }

  template <typename Number>
  std::array<FullMatrix<Number>, 3>
  get_dg_weights(unsigned int const r, double const)
  {
    static std::unordered_map<unsigned int, std::array<FullMatrix<Number>, 3>>
      cache;
    if (auto it = cache.find(r); it != cache.end())
      return it->second;

    // Radau quadrature
    auto const poly = get_time_basis(TimeStepType::DG, r);

    std::vector<Polynomials::Polynomial<double>> poly_derivative(poly.size());
    for (unsigned int i = 0; i < poly.size(); ++i)
      poly_derivative[i] = poly[i].derivative();

    QGauss<1> const quad(r + 2);

    FullMatrix<Number> lhs_matrix(r + 1, r + 1); // Later multiply with tau_n
    FullMatrix<Number> jump_matrix(r + 1, 1);
    FullMatrix<Number> lhs_matrix_der(r + 1, r + 1);

    for (unsigned int i = 0; i < r + 1; ++i)
      {
        jump_matrix(i, 0) = poly[i].value(0.0);
        for (unsigned int j = 0; j < r + 1; ++j)
          for (unsigned int q = 0; q < quad.size(); ++q)
            lhs_matrix(i, j) += quad.weight(q) *
                                poly[i].value(quad.point(q)[0]) *
                                poly[j].value(quad.point(q)[0]);
      }

    for (unsigned int i = 0; i < r + 1; ++i)
      for (unsigned int j = 0; j < r + 1; ++j)
        {
          // Jump
          lhs_matrix_der(i, j) += poly[i].value(0) * poly[j].value(0);
          // Integration
          for (unsigned int q = 0; q < quad.size(); ++q)
            lhs_matrix_der(i, j) += quad.weight(q) *
                                    poly[i].value(quad.point(q)[0]) *
                                    poly_derivative[j].value(quad.point(q)[0]);
        }

    cache[r] = {{lhs_matrix, lhs_matrix_der, jump_matrix}};
    return cache[r];
  }

  std::vector<size_t>
  get_fe_q_permutation(FE_Q<1> const &fe_time);

  template <typename Number = double>
  FullMatrix<Number>
  get_time_projection_matrix(TimeStepType       type,
                             unsigned int const r_src,
                             unsigned int const r_dst,
                             unsigned int const n_timesteps_at_once)
  {
    auto n_dofs_intvl_dst = (type == TimeStepType::DG) ? r_dst + 1 : r_dst;
    auto n_dofs_intvl_src = (type == TimeStepType::DG) ? r_src + 1 : r_src;

    auto n_dofs_dst = (type == TimeStepType::DG) ?
                        n_timesteps_at_once * (r_dst + 1) :
                        (n_timesteps_at_once * r_dst + 1);
    auto n_dofs_src = (type == TimeStepType::DG) ?
                        n_timesteps_at_once * (r_src + 1) :
                        (n_timesteps_at_once * r_src + 1);

    FullMatrix<Number> projection(r_dst + 1, r_src + 1),
      projection_n(n_dofs_dst, n_dofs_src);

    if (type == TimeStepType::DG)
      {
        auto quad_time_src =
          QGaussRadau<1>(r_src + 1, QGaussRadau<1>::EndPoint::right);
        FE_DGQArbitraryNodes<1> fe_time_src(quad_time_src.get_points());
        auto                    quad_time_dst =
          QGaussRadau<1>(r_dst + 1, QGaussRadau<1>::EndPoint::right);
        FE_DGQArbitraryNodes<1> fe_time_dst(quad_time_dst.get_points());
        FETools::get_projection_matrix(fe_time_src, fe_time_dst, projection);
      }
    else
      {
        auto    quad_time_src = QGaussLobatto<1>(r_src + 1);
        FE_Q<1> fe_time_src(quad_time_src.get_points());
        auto    quad_time_dst = QGaussLobatto<1>(r_dst + 1);
        FE_Q<1> fe_time_dst(quad_time_dst.get_points());

        FullMatrix<Number> projection_(fe_time_dst.n_dofs_per_cell(),
                                       fe_time_src.n_dofs_per_cell());
        FETools::get_projection_matrix(fe_time_src, fe_time_dst, projection_);
        auto perm_dst = get_fe_q_permutation(fe_time_dst);
        auto perm_src = get_fe_q_permutation(fe_time_src);
        projection.fill_permutation(projection_, perm_dst, perm_src);
      }

    for (unsigned int it = 0; it < n_timesteps_at_once; ++it)
      projection_n.fill(
        projection, it * n_dofs_intvl_dst, it * n_dofs_intvl_src, 0, 0);

    if (type == TimeStepType::CGP)
      {
        FullMatrix<Number> projection_n_(n_dofs_dst - 1, n_dofs_src - 1);
        projection_n_.fill(projection_n, 0, 0, 1, 1);
        return projection_n_;
      }
    return projection_n;
  }

  template <typename Number = double>
  FullMatrix<Number>
  get_time_prolongation_matrix(TimeStepType       time_type,
                               unsigned int const r,
                               unsigned int const n_timesteps_at_once = 2)
  {
    Assert((n_timesteps_at_once > 1 &&
            ((n_timesteps_at_once & (n_timesteps_at_once - 1)) == 0)),
           ExcMessage("Has to be a power of 2 and more than one timestep"));
    FullMatrix<Number> prolongation, prolongation_n;
    Quadrature<1>      quad_time;
    if (time_type == TimeStepType::DG)
      {
        quad_time = QGaussRadau<1>(r + 1, QGaussRadau<1>::EndPoint::right);
        FE_DGQArbitraryNodes<1> fe_time(quad_time.get_points());
        auto left_interval  = fe_time.get_prolongation_matrix(0),
             right_interval = fe_time.get_prolongation_matrix(1);
        prolongation.reinit(2 * (r + 1), r + 1);
        prolongation.fill(left_interval, 0, 0, 0, 0);
        prolongation.fill(right_interval, left_interval.m(), 0, 0, 0);
      }
    else if (time_type == TimeStepType::CGP)
      {
        quad_time = QGaussLobatto<1>(r + 1);
        FE_Q<1> fe_time(quad_time.get_points());
        auto    left_interval_ = fe_time.get_prolongation_matrix(0),
             right_interval_   = fe_time.get_prolongation_matrix(1);
        FullMatrix<Number> right_interval(r + 1, r + 1),
          left_interval(r + 1, r + 1);
        auto perm = get_fe_q_permutation(fe_time);
        right_interval.fill_permutation(right_interval_, perm, perm);
        left_interval.fill_permutation(left_interval_, perm, perm);
        prolongation.reinit(2 * r, r);
        prolongation.fill(left_interval, 0, 0, 1, 1);
        prolongation.fill(right_interval, r, 0, 1, 1);
      }
    auto n_dofs_intvl = (time_type == TimeStepType::DG) ? r + 1 : r;
    prolongation_n.reinit(n_dofs_intvl * n_timesteps_at_once,
                          n_dofs_intvl * n_timesteps_at_once / 2);
    for (unsigned int it = 0; it < n_timesteps_at_once / 2; ++it)
      prolongation_n.fill(
        prolongation, it * 2 * n_dofs_intvl, it * n_dofs_intvl, 0, 0);

    return prolongation_n;
  }

  template <typename Number = double>
  FullMatrix<Number>
  get_time_restriction_matrix(TimeStepType       time_type,
                              unsigned int const r,
                              unsigned int const n_timesteps_at_once = 2)
  {
    Assert((n_timesteps_at_once > 1 &&
            ((n_timesteps_at_once & (n_timesteps_at_once - 1)) == 0)),
           ExcMessage("Has to be a power of 2 and more than one timestep"));
    FullMatrix<Number> restriction, restriction_n;
    Quadrature<1>      quad_time;
    if (time_type == TimeStepType::DG)
      {
        quad_time = QGaussRadau<1>(r + 1, QGaussRadau<1>::EndPoint::right);
        FE_DGQArbitraryNodes<1> fe_time(quad_time.get_points());
        auto left_interval  = fe_time.get_restriction_matrix(0),
             right_interval = fe_time.get_restriction_matrix(1);
        restriction.reinit(r + 1, 2 * (r + 1));
        restriction.fill(left_interval, 0, 0, 0, 0);
        restriction.fill(right_interval, 0, left_interval.n(), 0, 0);
      }
    else if (time_type == TimeStepType::CGP)
      {
        quad_time = QGaussLobatto<1>(r + 1);
        FE_Q<1> fe_time(quad_time.get_points());

        auto left_interval_  = fe_time.get_restriction_matrix(0),
             right_interval_ = fe_time.get_restriction_matrix(1);
        FullMatrix<Number> right_interval(r + 1, r + 1),
          left_interval(r + 1, r + 1);
        auto perm = get_fe_q_permutation(fe_time);
        right_interval.fill_permutation(right_interval_, perm, perm);
        left_interval.fill_permutation(left_interval_, perm, perm);
        restriction.reinit(r, 2 * r);
        restriction.fill(left_interval, 0, 0, 1, 1);
        restriction.fill(right_interval, 0, r, 1, 1);
      }
    auto n_dofs_intvl = (time_type == TimeStepType::DG) ? r + 1 : r;
    restriction_n.reinit(n_dofs_intvl * n_timesteps_at_once / 2,
                         n_dofs_intvl * n_timesteps_at_once);
    for (unsigned int it = 0; it < n_timesteps_at_once / 2; ++it)
      restriction_n.fill(
        restriction, it * n_dofs_intvl, it * 2 * n_dofs_intvl, 0, 0);

    return restriction_n;
  }


  class block_indexing
  {
  public:
    block_indexing(block_indexing const &other)
      : n_timesteps_at_once_(other.n_timesteps_at_once())
      , n_variables_(other.n_variables())
      , n_timedofs_(other.n_timedofs())
      , cache(other.cache)
    {}

    block_indexing(block_indexing &&other) noexcept
      : n_timesteps_at_once_(other.n_timesteps_at_once())
      , n_variables_(other.n_variables())
      , n_timedofs_(other.n_timedofs())
      , cache(other.cache)
    {}

    block_indexing &
    operator=(block_indexing const &) = default;
    block_indexing &
    operator=(block_indexing &&) noexcept = default;

    block_indexing(unsigned int n_timesteps_at_once,
                   unsigned int n_variables,
                   unsigned int n_timedofs)
      : n_timesteps_at_once_(n_timesteps_at_once)
      , n_variables_(n_variables)
      , n_timedofs_(n_timedofs)
      , cache(n_blocks())
    {
      for (unsigned int i = 0; i < n_blocks(); ++i)
        cache[i] = decompose(i);
    }

    static inline void
    set_variable_major(bool is_variable_major_)
    { // allow setting once
      if (!is_variable_major_set)
        is_variable_major = is_variable_major_;
      is_variable_major_set = true;
    }

    static inline bool
    get_variable_major()
    {
      return is_variable_major;
    }

    std::array<unsigned int, 3>
    operator()(unsigned int index) const
    {
      Assert(n_blocks() > index, ExcLowerRange(n_blocks(), index));
      return cache[index];
    }

    unsigned int
    operator()(unsigned int timestep,
               unsigned int variable,
               unsigned int timedof) const
    {
      if (get_variable_major())
        return timestep * (n_variables_ * n_timedofs_) +
               variable * n_timedofs_ + timedof;
      else
        return timestep * (n_variables_ * n_timedofs_) +
               timedof * n_variables_ + variable;
    }

    unsigned int
    n_timesteps_at_once() const
    {
      return n_timesteps_at_once_;
    }
    unsigned int
    n_variables() const
    {
      return n_variables_;
    }
    unsigned int
    n_timedofs() const
    {
      return n_timedofs_;
    }

    unsigned int
    n_blocks() const
    {
      return n_timedofs() * n_timesteps_at_once() * n_variables();
    }

  private:
    std::array<unsigned int, 3>
    decompose(unsigned int index) const
    {
      Assert((n_blocks() - 1 >= index), ExcLowerRange(n_blocks() - 1, index));
      unsigned int timestep, variable, timedof;
      if (get_variable_major())
        {
          timestep = index / (n_variables_ * n_timedofs_);
          variable = (index % (n_variables_ * n_timedofs_)) / n_timedofs_;
          timedof  = (index % (n_variables_ * n_timedofs_)) % n_timedofs_;
        }
      else
        {
          timestep = index / (n_variables_ * n_timedofs_);
          timedof  = (index % (n_variables_ * n_timedofs_)) / n_variables_;
          variable = (index % (n_variables_ * n_timedofs_)) % n_variables_;
        }
      return {{timestep, variable, timedof}};
    }

    unsigned int n_timesteps_at_once_, n_variables_, n_timedofs_;
    std::vector<std::array<unsigned int, 3>> cache;

    static inline bool is_variable_major     = true;
    static inline bool is_variable_major_set = false;
  };

  class BlockSlice
  {
  public:
    BlockSlice()
      : idx(1, 1, 1)
    {}

    BlockSlice(BlockSlice const &other)
      : idx(other.idx)
    {}

    BlockSlice(BlockSlice &&other) noexcept
      : idx(std::move(other.idx))
    {}

    BlockSlice &
    operator=(const BlockSlice &) = default;
    BlockSlice &
    operator=(BlockSlice &&) noexcept = default;

    BlockSlice(unsigned int n_timesteps_at_once_,
               unsigned int n_variables_,
               unsigned int n_timedofs_)
      : idx(n_timesteps_at_once_, n_variables_, n_timedofs_)
    {}

    std::vector<unsigned int>
    get_variable(unsigned int timestep, unsigned int timedof) const
    {
      std::vector<unsigned int> indices;
      indices.reserve(n_variables());

      for (unsigned int variable = 0; variable < n_variables(); ++variable)
        indices.push_back(idx(timestep, variable, timedof));

      return indices;
    }

    std::vector<unsigned int>
    get_time(unsigned int variable) const
    {
      std::vector<unsigned int> indices;
      indices.reserve(n_timesteps_at_once() * n_timedofs());

      for (unsigned int tsp = 0; tsp < n_timesteps_at_once(); ++tsp)
        for (unsigned int timedof = 0; timedof < n_timedofs(); ++timedof)
          indices.push_back(idx(tsp, variable, timedof));
      return indices;
    }

    template <typename Number>
    BlockVectorSliceT<Number>
    get_slice(const BlockVectorT<Number>      &block_vector,
              std::vector<unsigned int> const &indices) const
    {
      BlockVectorSliceT<Number> result;
      result.reserve(indices.size());
      for (unsigned int i = 0; i < indices.size(); ++i)
        result.push_back(std::cref(block_vector.block(indices[i])));

      return result;
    }
    template <typename Number>
    MutableBlockVectorSliceT<Number>
    get_slice(BlockVectorT<Number>            &block_vector,
              std::vector<unsigned int> const &indices) const
    {
      MutableBlockVectorSliceT<Number> result;
      result.reserve(indices.size());
      for (unsigned int i = 0; i < indices.size(); ++i)
        result.push_back(std::ref(block_vector.block(indices[i])));

      return result;
    }

    template <typename Number>
    BlockVectorSliceT<Number>
    get_time(const BlockVectorT<Number> &block_vector,
             unsigned int                variable) const
    {
      std::vector<unsigned int> indices = get_time(variable);
      return get_slice(block_vector, indices);
    }

    template <typename Number>
    MutableBlockVectorSliceT<Number>
    get_time(BlockVectorT<Number> &block_vector, unsigned int variable) const
    {
      std::vector<unsigned int> indices = get_time(variable);
      return get_slice(block_vector, indices);
    }

    template <typename Number>
    BlockVectorSliceT<Number>
    get_time(const BlockVectorT<Number> &block_vector,
             const BlockVectorT<Number> &prev_vector,
             unsigned int                variable) const
    {
      AssertDimension(prev_vector.n_blocks(), this->n_variables());
      BlockVectorSliceT<Number> result;
      std::vector<unsigned int> indices = get_time(variable);
      result.reserve(indices.size() + 1);
      result.push_back(std::cref(prev_vector.block(variable)));
      auto result_ = get_slice(block_vector, indices);
      result.insert(result.end(), result_.begin(), result_.end());
      return result;
    }

    template <typename Number>
    MutableBlockVectorSliceT<Number>
    get_time(BlockVectorT<Number> &block_vector,
             BlockVectorT<Number> &prev_vector,
             unsigned int          variable) const
    {
      AssertDimension(prev_vector.n_blocks(), this->n_variables());
      MutableBlockVectorSliceT<Number> result;
      std::vector<unsigned int>        indices = get_time(variable);
      result.reserve(indices.size() + 1);
      result.push_back(std::ref(prev_vector.block(variable)));
      auto result_ = get_slice(block_vector, indices);
      result.insert(result.end(), result_.begin(), result_.end());
      return result;
    }


    template <typename Number>
    BlockVectorSliceT<Number>
    get_time(const BlockVectorT<Number> &block_vector,
             const VectorT<Number>      &prev_vector) const
    {
      AssertDimension(1, this->n_variables());
      BlockVectorSliceT<Number> result;
      std::vector<unsigned int> indices = get_time(0);
      result.reserve(indices.size() + 1);
      result.push_back(std::cref(prev_vector));
      auto result_ = get_slice(block_vector, indices);
      result.insert(result.end(), result_.begin(), result_.end());
      return result;
    }

    template <typename Number>
    MutableBlockVectorSliceT<Number>
    get_time(BlockVectorT<Number> &block_vector,
             VectorT<Number>      &prev_vector) const
    {
      AssertDimension(1, this->n_variables());
      MutableBlockVectorSliceT<Number> result;
      std::vector<unsigned int>        indices = get_time(0);
      result.reserve(indices.size() + 1);
      result.push_back(std::ref(prev_vector));
      auto result_ = get_slice(block_vector, indices);
      result.insert(result.end(), result_.begin(), result_.end());
      return result;
    }

    template <typename Number>
    BlockVectorSliceT<Number>
    get_variable(const BlockVectorT<Number> &block_vector,
                 unsigned int                timestep,
                 unsigned int                timedof) const
    {
      auto indices = get_variable(timestep, timedof);
      return get_slice(block_vector, indices);
    }

    template <typename Number>
    MutableBlockVectorSliceT<Number>
    get_variable(BlockVectorT<Number> &block_vector,
                 unsigned int          timestep,
                 unsigned int          timedof) const
    {
      auto indices = get_variable(timestep, timedof);
      return get_slice(block_vector, indices);
    }

    template <typename Number>
    BlockVectorSliceT<Number>
    get_variable(const BlockVectorT<Number> &block_vector,
                 unsigned int                time) const
    {
      unsigned int ts = time / n_timedofs();
      unsigned int td = time % n_timedofs();
      return get_variable(block_vector, ts, td);
    }

    template <typename Number>
    MutableBlockVectorSliceT<Number>
    get_variable(BlockVectorT<Number> &block_vector, unsigned int time) const
    {
      unsigned int ts = time / n_timedofs();
      unsigned int td = time % n_timedofs();
      return get_variable(block_vector, ts, td);
    }
    unsigned int
    n_timesteps_at_once() const
    {
      return idx.n_timesteps_at_once();
    }
    unsigned int
    n_variables() const
    {
      return idx.n_variables();
    }
    unsigned int
    n_timedofs() const
    {
      return idx.n_timedofs();
    }

    unsigned int
    n_blocks() const
    {
      return idx.n_blocks();
    }

    auto
    decompose(unsigned int i) const
    {
      return idx(i);
    }

    auto
    index(unsigned int timestep,
          unsigned int variable,
          unsigned int timedof) const
    {
      Assert((n_blocks() - 1 >= idx(timestep, variable, timedof)),
             ExcLowerRange(n_blocks() - 1, idx(timestep, variable, timedof)));
      return idx(timestep, variable, timedof);
    }

  private:
    block_indexing idx;
  };

  template <typename Number>
  void
  get_linear_combination(BlockVectorT<Number>       &mean,
                         BlockSlice const           &blk_src,
                         std::vector<double> const  &weights,
                         BlockVectorT<Number> const &x)
  {
    mean.reinit(blk_src.n_variables());
    for (unsigned int v = 0; v < blk_src.n_variables(); ++v)
      {
        BlockVectorSliceT<Number> src_v = blk_src.get_time(x, v);
        mean.block(v).reinit(src_v[0].get());

        for (unsigned int i = 0; i < src_v.size(); ++i)
          mean.block(v).add(static_cast<Number>(weights[i]), src_v[i].get());
      }
  }

  template <typename Number>
  void
  extrapolate_nonlinear(BlockVectorT<Number>       &x_extrapolated,
                        FullMatrix<Number> const   &matrix,
                        BlockSlice const           &blk_src,
                        BlockVectorT<Number> const &x,
                        BlockVectorT<Number> const &prev_x)
  {
    for (unsigned int v = 0; v < blk_src.n_variables(); ++v)
      {
        BlockVectorSliceT<Number>        src_v = blk_src.get_time(x, prev_x, v);
        MutableBlockVectorSliceT<Number> dst_v =
          blk_src.get_time(x_extrapolated, v);
        AssertDimension(matrix.m(), dst_v.size());
        AssertDimension(matrix.n(), src_v.size());
        tensorproduct(dst_v, matrix, src_v);
      }
  }

  template <typename Number>
  void
  extrapolate_nonlinear(BlockVectorT<Number>       &x_extrapolated,
                        FullMatrix<Number> const   &matrix,
                        BlockSlice const           &blk_src,
                        BlockVectorT<Number> const &x,
                        VectorT<Number> const      &prev_x)
  {
    for (unsigned int v = 0; v < blk_src.n_variables(); ++v)
      {
        BlockVectorSliceT<Number>        src_v = blk_src.get_time(x, prev_x);
        MutableBlockVectorSliceT<Number> dst_v =
          blk_src.get_time(x_extrapolated, v);
        AssertDimension(matrix.m(), dst_v.size());
        AssertDimension(matrix.n(), src_v.size());
        tensorproduct(dst_v, matrix, src_v);
      }
  }

  template <typename Number>
  std::array<FullMatrix<Number>, 4>
  get_fe_time_weights_stokes(TimeStepType       type,
                             unsigned int const r,
                             double             time_step_size,
                             unsigned int       n_timesteps_at_once = 1,
                             double             delta0              = 0.0)
  {
    auto tw = get_fe_time_weights<Number>(
      type, r, time_step_size, n_timesteps_at_once, delta0);
    std::array<FullMatrix<Number>, 4> ret;
    for (int i = 0; i < 2; ++i)
      ret[i].reinit(tw[i].m() * 2, tw[i].n() * 2);
    for (int i = 2; i < 4; ++i)
      ret[i].reinit(tw[i].m() * 2, tw[i].n());
    BlockSlice blk_slice(n_timesteps_at_once,
                         2,
                         (type == TimeStepType::DG) ? r + 1 : r);

    for (int iv = 0; iv < 2; ++iv)
      {
        auto variable_indices = blk_slice.get_time(iv);
        AssertDimension(variable_indices.size(), tw[0].m());
        // Stokes
        for (int jv = 0; jv < 2; ++jv)
          {
            auto jvariable_indices = blk_slice.get_time(jv);
            if (!(jv == 1 && iv == 1))
              tw[0].scatter_matrix_to(variable_indices,
                                      jvariable_indices,
                                      ret[0]);
          }
        // \partial_t v
        if (iv == 0)
          {
            tw[1].scatter_matrix_to(variable_indices, variable_indices, ret[1]);
            for (int i = 2; i < 4; ++i)
              tw[i].scatter_matrix_to(variable_indices, {0}, ret[i]);
          }
        if (iv == 1 && type == TimeStepType::CGP)
          tw[2].scatter_matrix_to(variable_indices, {0}, ret[2]);
      }
    return ret;
  }


  template <typename Number>
  std::array<FullMatrix<Number>, 4>
  get_fe_time_weights_2variable_evolutionary(
    TimeStepType       type,
    unsigned int const r,
    double             time_step_size,
    unsigned int       n_timesteps_at_once = 1,
    double             delta0              = 0.0)
  {
    auto tw = get_fe_time_weights<Number>(
      type, r, time_step_size, n_timesteps_at_once, delta0);
    std::array<FullMatrix<Number>, 4> ret;
    for (int i = 0; i < 2; ++i)
      ret[i].reinit(tw[i].m() * 2, tw[i].n() * 2);
    for (int i = 2; i < 4; ++i)
      ret[i].reinit(tw[i].m() * 2, tw[i].n());
    BlockSlice blk_slice(n_timesteps_at_once,
                         2,
                         (type == TimeStepType::DG) ? r + 1 : r);

    for (int iv = 0; iv < 2; ++iv)
      {
        auto variable_indices       = blk_slice.get_time(iv);
        auto other_variable_indices = blk_slice.get_time(iv == 0 ? 1 : 0);
        AssertDimension(variable_indices.size(), tw[0].m());

        // For spatial operator
        tw[0].scatter_matrix_to(variable_indices,
                                other_variable_indices,
                                ret[0]);
        // \partial_t
        tw[1].scatter_matrix_to(variable_indices, variable_indices, ret[1]);
        // rhs
        for (int i = 2; i < 4; ++i)
          tw[i].scatter_matrix_to(variable_indices, {0}, ret[i]);
      }
    return ret;
  }

  template <typename Number>
  BlockVectorSliceT<Number>
  copy_blocks(const BlockVectorT<Number> &bv)
  {
    BlockVectorSliceT<Number> refs;
    refs.reserve(bv.n_blocks());
    for (unsigned int i = 0; i < bv.n_blocks(); ++i)
      refs.emplace_back(std::cref(bv.block(i)));
    return refs;
  }

  template <typename Number>
  inline void
  swap(MutableBlockVectorSliceT<Number> &slice, BlockVectorT<Number> &vec)
  {
    AssertDimension(slice.size(), vec.n_blocks());
    for (unsigned int i = 0; i < slice.size(); ++i)
      slice[i].get().swap(vec.block(i));
  }

  template <typename Number>
  inline void
  swap(BlockVectorT<Number> &vec, MutableBlockVectorSliceT<Number> &slice)
  { // syntactic sugar for swapping back and forth
    swap(slice, vec);
  }

  template <typename Number>
  void
  equ(BlockVectorT<Number> &blk, BlockVectorSliceT<Number> const &slice)
  {
    AssertDimension(slice.size(), blk.n_blocks());
    for (unsigned int i = 0; i < slice.size(); ++i)
      blk.block(i).equ(1.0, slice[i].get());
  }
  template <typename Number>
  void
  equ(BlockVectorT<Number> &blk, MutableBlockVectorSliceT<Number> const &slice)
  {
    AssertDimension(slice.size(), blk.n_blocks());
    for (unsigned int i = 0; i < slice.size(); ++i)
      blk.block(i).equ(1.0, slice[i].get());
  }
} // namespace dealii
