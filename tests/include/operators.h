// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by Nils Margenberg and Peter Munch

#pragma once

#include <deal.II/base/subscriptor.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#include "types.h"

namespace dealii
{
  template <typename Number>
  void
  tensorproduct_add(BlockVectorT<Number>     &c,
                    FullMatrix<Number> const &A,
                    VectorT<Number> const    &b,
                    int                       block_offset = 0)
  {
    const unsigned int n_blocks = A.m();
    AssertDimension(A.n(), 1);
    for (unsigned int i = 0; i < n_blocks; ++i)
      if (A(i, 0) != 0.0)
        c.block(block_offset + i).add(A(i, 0), b);
  }

  template <typename Number>
  BlockVectorT<Number>
  operator*(const FullMatrix<Number> &A, VectorT<Number> const &b)
  {
    const unsigned int   n_blocks = A.m();
    BlockVectorT<Number> c(n_blocks);
    for (unsigned int i = 0; i < n_blocks; ++i)
      c.block(i).reinit(b);
    tensorproduct_add(c, A, b);
    return c;
  }

  template <typename Number>
  void
  tensorproduct_add(BlockVectorT<Number>       &c,
                    FullMatrix<Number> const   &A,
                    BlockVectorT<Number> const &b,
                    int                         block_offset = 0)
  {
    const unsigned int n_blocks = A.n();
    const unsigned int m_blocks = A.m();
    for (unsigned int i = 0; i < m_blocks; ++i)
      for (unsigned int j = 0; j < n_blocks; ++j)
        if (A(i, j) != 0.0)
          c.block(block_offset + i).add(A(i, j), b.block(block_offset + j));
  }

  template <typename Number>
  BlockVectorT<Number>
  operator*(const FullMatrix<Number> &A, BlockVectorT<Number> const &b)
  {
    BlockVectorT<Number> c(A.m());
    for (unsigned int i = 0; i < A.m(); ++i)
      c.block(i).reinit(b.block(i));
    tensorproduct_add(c, A, b);
    return c;
  }

  template <typename Number, typename SystemMatrixType>
  class SystemMatrix final : public Subscriptor
  {
  public:
    using BlockVectorType = BlockVectorT<Number>;
    using VectorType      = VectorT<Number>;

    SystemMatrix(TimerOutput              &timer,
                 const SystemMatrixType   &K,
                 const SystemMatrixType   &M,
                 const FullMatrix<Number> &Alpha_,
                 const FullMatrix<Number> &Beta_)
      : timer(timer)
      , K(K)
      , M(M)
      , Alpha(Alpha_)
      , Beta(Beta_)
      , alpha_is_zero(Alpha.all_zero())
      , beta_is_zero(Beta.all_zero())
    {
      AssertDimension(Alpha.m(), Beta.m());
      AssertDimension(Alpha.n(), Beta.n());
    }
    void
    vmult(BlockVectorType &dst, const BlockVectorType &src) const
    {
      TimerOutput::Scope scope(timer, "vmult");

      const unsigned int n_blocks = src.n_blocks();
      AssertDimension(Alpha.m(), n_blocks);
      dst = 0.0;
      VectorType tmp;
      K.initialize_dof_vector(tmp);
      for (unsigned int i = 0; i < n_blocks; ++i)
        {
          K.vmult(tmp, src.block(i));

          for (unsigned int j = 0; j < n_blocks; ++j)
            if (Alpha(j, i) != 0.0)
              dst.block(j).add(Alpha(j, i), tmp);
        }

      M.initialize_dof_vector(tmp);
      for (unsigned int i = 0; i < n_blocks; ++i)
        {
          M.vmult(tmp, src.block(i));

          for (unsigned int j = 0; j < n_blocks; ++j)
            if (Beta(j, i) != 0.0)
              dst.block(j).add(Beta(j, i), tmp);
        }
    }

    void
    Tvmult(BlockVectorType &dst, const BlockVectorType &src) const
    {
      TimerOutput::Scope scope(timer, "Tvmult");

      const unsigned int n_blocks = src.n_blocks();

      dst = 0.0;
      VectorType tmp;
      K.initialize_dof_vector(tmp);
      for (unsigned int i = 0; i < n_blocks; ++i)
        {
          K.vmult(tmp, src.block(i));

          for (unsigned int j = 0; j < n_blocks; ++j)
            if (Alpha(i, j) != 0.0)
              dst.block(j).add(Alpha(i, j), tmp);
        }

      M.initialize_dof_vector(tmp);
      for (unsigned int i = 0; i < n_blocks; ++i)
        {
          M.vmult(tmp, src.block(i));

          for (unsigned int j = 0; j < n_blocks; ++j)
            if (Beta(i, j) != 0.0)
              dst.block(j).add(Beta(i, j), tmp);
        }
    }

    // Specialization for a nx1 matrix. Useful for rhs assembly
    void
    vmult_add(BlockVectorType &dst, const VectorType &src) const
    {
      TimerOutput::Scope scope(timer, "vmult");

      const unsigned int n_blocks = dst.n_blocks();

      VectorType tmp;
      if (!alpha_is_zero)
        {
          K.initialize_dof_vector(tmp);
          K.vmult(tmp, src);
          for (unsigned int j = 0; j < n_blocks; ++j)
            if (Alpha(j, 0) != 0.0)
              dst.block(j).add(Alpha(j, 0), tmp);
        }

      if (!beta_is_zero)
        {
          M.initialize_dof_vector(tmp);
          M.vmult(tmp, src);
          for (unsigned int j = 0; j < n_blocks; ++j)
            if (Beta(j, 0) != 0.0)
              dst.block(j).add(Beta(j, 0), tmp);
        }
    }

    void
    vmult(BlockVectorType &dst, const VectorType &src) const
    {
      dst = 0.0;
      vmult_add(dst, src);
    }

    std::shared_ptr<DiagonalMatrix<BlockVectorType>>
    get_matrix_diagonal() const
    {
      BlockVectorType vec(Alpha.m());
      for (unsigned int i = 0; i < Alpha.m(); ++i)
        {
          vec.block(i) = K.get_matrix_diagonal()->get_vector();
          vec.block(i).sadd(Alpha(i, i),
                            Beta(i, i),
                            M.get_matrix_diagonal()->get_vector());
        }
      return std::make_shared<DiagonalMatrix<BlockVectorType>>(vec);
    }

    std::shared_ptr<DiagonalMatrix<BlockVectorType>>
    get_matrix_diagonal_inverse() const
    {
      BlockVectorType vec(Alpha.m());
      for (unsigned int i = 0; i < Alpha.m(); ++i)
        {
          vec.block(i) = K.get_matrix_diagonal_inverse()->get_vector();
          vec.block(i).sadd(1. / Alpha(i, i),
                            1. / Beta(i, i),
                            M.get_matrix_diagonal_inverse()->get_vector());
        }
      return std::make_shared<DiagonalMatrix<BlockVectorType>>(vec);
    }

    types::global_dof_index
    m() const
    {
      return Alpha.m() * M.m();
    }

    Number
    el(unsigned int, unsigned int) const
    {
      Assert(false, ExcNotImplemented());
      return 0.0;
    }

    template <typename Number2>
    void
    initialize_dof_vector(VectorT<Number2> &vec) const
    {
      K.initialize_dof_vector(vec);
    }

    template <typename Number2>
    void
    initialize_dof_vector(BlockVectorT<Number2> &vec) const
    {
      vec.reinit(Alpha.m());
      for (unsigned int i = 0; i < vec.n_blocks(); ++i)
        this->initialize_dof_vector(vec.block(i));
    }

  private:
    TimerOutput              &timer;
    const SystemMatrixType   &K;
    const SystemMatrixType   &M;
    const FullMatrix<Number> &Alpha;
    const FullMatrix<Number> &Beta;

    // Only used for nx1: small optimization to avoid unnecessary vmult
    bool alpha_is_zero;
    bool beta_is_zero;
  };

  template <int dim>
  class Coefficient final : public Function<dim>
  {
    double             c1, c2, c3;
    bool               distorted;
    Point<dim>         lower_left;
    Point<dim>         step_size;
    Table<dim, double> distortion;

    template <typename number>
    double
    get_coefficient(number px, number py) const
    {
      if (py >= 0.2)
        {
          if (px < 0.2)
            return c2;
          else
            return c3;
        }
      return c1;
    }


  public:
    Coefficient(Parameters<dim> const &params,
                double                 c1_ = 1.0,
                double                 c2_ = 9.0,
                double                 c3_ = 16.0)
      : c1(c1_)
      , c2(c2_)
      , c3(c3_)
      , distorted(params.distort_coeff != 0.0)
      , lower_left(params.hyperrect_lower_left)
    {
      if (distorted)
        {
          auto const &subdivisions = params.subdivisions;
          if constexpr (dim == 2)
            distortion = Table<2, double>(subdivisions[0], subdivisions[1]);
          else
            distortion = Table<3, double>(subdivisions[0],
                                          subdivisions[1],
                                          subdivisions[2]);
          std::vector<double>    tmp(distortion.n_elements());
          boost::random::mt19937 rng(boost::random::mt19937::default_seed);
          boost::random::uniform_real_distribution<> uniform_distribution(
            1 - params.distort_coeff, 1 + params.distort_coeff);
          std::generate(tmp.begin(), tmp.end(), [&]() {
            return uniform_distribution(rng);
          });
          distortion.fill(tmp.begin());
          auto const extent =
            params.hyperrect_upper_right - params.hyperrect_lower_left;
          for (int i = 0; i < dim; ++i)
            step_size[i] = extent[i] / subdivisions[i];
        }
    }

    virtual double
    value(const Point<dim> &p, const unsigned int /*component*/) const override
    {
      return get_coefficient(p[0], p[1]);
    }

    template <typename number>
    number
    value(const Point<dim, number> &p) const
    {
      number value;
      auto   v = value.begin();
      if constexpr (dim == 2)
        for (auto px = p[0].begin(), py = p[1].begin(); px != p[0].end();
             ++px, ++py, ++v)
          {
            *v = get_coefficient(*px, *py);
            if (distorted)
              *v *= distortion(
                static_cast<unsigned>((*px - lower_left[0]) / step_size[0]),
                static_cast<unsigned>((*py - lower_left[1]) / step_size[1]));
          }
      else
        for (auto px = p[0].begin(), py = p[1].begin(), pz = p[2].begin();
             px != p[0].end();
             ++px, ++py, ++pz, ++v)
          {
            *v = get_coefficient(*px, *py);
            if (distorted)
              *v *= distortion(
                static_cast<unsigned>((*px - lower_left[0]) / step_size[0]),
                static_cast<unsigned>((*py - lower_left[1]) / step_size[1]),
                static_cast<unsigned>((*pz - lower_left[2]) / step_size[2]));
          }
      return value;
    }
  };

  template <int dim, typename Number>
  class MatrixFreeOperator
  {
  public:
    using BlockVectorType = BlockVectorT<Number>;
    using VectorType      = VectorT<Number>;

    MatrixFreeOperator(const Mapping<dim>              &mapping,
                       const DoFHandler<dim>           &dof_handler,
                       const AffineConstraints<Number> &constraints,
                       const Quadrature<dim>           &quadrature,
                       const double                     mass_matrix_scaling,
                       const double                     laplace_matrix_scaling)
      : mass_matrix_scaling(mass_matrix_scaling)
      , laplace_matrix_scaling(laplace_matrix_scaling)
      , has_mass_coefficient(false)
      , has_laplace_coefficient(false)
    {
      mass_matrix_coefficient.clear();
      laplace_matrix_coefficient.clear();
      typename MatrixFree<dim, Number>::AdditionalData additional_data;
      additional_data.mapping_update_flags =
        update_values | update_gradients | update_quadrature_points;

      matrix_free.reinit(
        mapping, dof_handler, constraints, quadrature, additional_data);

      compute_diagonal();
    }

    template <typename Number2>
    void
    initialize_dof_vector(VectorT<Number2> &vec) const
    {
      matrix_free.initialize_dof_vector(vec);
    }

    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      matrix_free.cell_loop(
        &MatrixFreeOperator::do_cell_integral_range, this, dst, src, true);
    }

    void
    compute_system_matrix(SparseMatrixType &sparse_matrix) const
    {
      MatrixFreeTools::compute_matrix(
        matrix_free,
        matrix_free.get_affine_constraints(),
        sparse_matrix,
        &MatrixFreeOperator::do_cell_integral_local,
        this);
    }

    std::shared_ptr<DiagonalMatrix<VectorType>> const &
    get_matrix_diagonal() const
    {
      return diagonal;
    }

    std::shared_ptr<DiagonalMatrix<VectorType>> const &
    get_matrix_diagonal_inverse() const
    {
      return diagonal_inverse;
    }

    types::global_dof_index
    m() const
    {
      return matrix_free.get_dof_handler().n_dofs();
    }

    Number
    el(unsigned int, unsigned int) const
    {
      Assert(false, ExcNotImplemented());
      return 0.0;
    }

    void
    evaluate_coefficient(const Coefficient<dim> &coefficient_fun)
    {
      FECellIntegrator   integrator(matrix_free);
      const unsigned int n_cells = matrix_free.n_cell_batches();
      if (mass_matrix_scaling != 0.0)
        mass_matrix_coefficient.reinit(n_cells, integrator.n_q_points);
      if (laplace_matrix_scaling != 0.0)
        laplace_matrix_coefficient.reinit(n_cells, integrator.n_q_points);

      for (unsigned int cell = 0; cell < n_cells; ++cell)
        {
          integrator.reinit(cell);
          for (const unsigned int q : integrator.quadrature_point_indices())
            {
              if (mass_matrix_scaling != 0.0)
                mass_matrix_coefficient(cell, q) =
                  coefficient_fun.value(integrator.quadrature_point(q));
              if (laplace_matrix_scaling != 0.0)
                laplace_matrix_coefficient(cell, q) =
                  coefficient_fun.value(integrator.quadrature_point(q));
            }
        }
      if (!mass_matrix_coefficient.empty())
        has_mass_coefficient = true;
      if (!laplace_matrix_coefficient.empty())
        has_laplace_coefficient = true;
    }

  private:
    using FECellIntegrator = FEEvaluation<dim, -1, 0, 1, Number>;

    void
    compute_diagonal()
    {
      diagonal         = std::make_shared<DiagonalMatrix<VectorType>>();
      diagonal_inverse = std::make_shared<DiagonalMatrix<VectorType>>();
      VectorType &diagonal_inv_vector = diagonal_inverse->get_vector();
      VectorType &diagonal_vector     = diagonal->get_vector();
      initialize_dof_vector(diagonal_inv_vector);
      initialize_dof_vector(diagonal_vector);
      MatrixFreeTools::compute_diagonal(
        matrix_free,
        diagonal_vector,
        &MatrixFreeOperator::do_cell_integral_local,
        this);
      diagonal_inv_vector = diagonal_vector;
      auto constexpr tol  = std::sqrt(std::numeric_limits<Number>::epsilon());
      for (auto &i : diagonal_inv_vector)
        i = std::abs(i) > tol ? 1. / i : 1.;
    }

    void
    do_cell_integral_range(
      const MatrixFree<dim, Number>               &matrix_free,
      VectorType                                  &dst,
      const VectorType                            &src,
      const std::pair<unsigned int, unsigned int> &range) const
    {
      FECellIntegrator integrator(matrix_free);

      for (unsigned int cell = range.first; cell < range.second; ++cell)
        {
          integrator.reinit(cell);

          // gather
          integrator.read_dof_values(src);

          do_cell_integral_local(integrator);

          // scatter
          integrator.distribute_local_to_global(dst);
        }
    }

    void
    do_cell_integral_local(FECellIntegrator &integrator) const
    {
      unsigned int const cell = integrator.get_current_cell_index();
      // evaluate
      if (mass_matrix_scaling != 0.0 && laplace_matrix_scaling != 0.0)
        integrator.evaluate(EvaluationFlags::values |
                            EvaluationFlags::gradients);
      else if (mass_matrix_scaling != 0.0)
        integrator.evaluate(EvaluationFlags::values);
      else if (laplace_matrix_scaling != 0.0)
        integrator.evaluate(EvaluationFlags::gradients);

      // quadrature
      for (unsigned int q = 0; q < integrator.n_q_points; ++q)
        {
          if (mass_matrix_scaling != 0.0)
            integrator.submit_value((has_mass_coefficient ?
                                       mass_matrix_coefficient(cell, q) :
                                       mass_matrix_scaling) *
                                      integrator.get_value(q),
                                    q);
          if (laplace_matrix_scaling != 0.0)
            integrator.submit_gradient((has_laplace_coefficient ?
                                          laplace_matrix_coefficient(cell, q) :
                                          laplace_matrix_scaling) *
                                         integrator.get_gradient(q),
                                       q);
        }

      // integrate
      if (mass_matrix_scaling != 0.0 && laplace_matrix_scaling != 0.0)
        integrator.integrate(EvaluationFlags::values |
                             EvaluationFlags::gradients);
      else if (mass_matrix_scaling != 0.0)
        integrator.integrate(EvaluationFlags::values);
      else if (laplace_matrix_scaling != 0.0)
        integrator.integrate(EvaluationFlags::gradients);
    }

    std::shared_ptr<DiagonalMatrix<VectorType>> diagonal;
    std::shared_ptr<DiagonalMatrix<VectorType>> diagonal_inverse;

    MatrixFree<dim, Number> matrix_free;

    Number mass_matrix_scaling;
    Number laplace_matrix_scaling;

    bool                              has_mass_coefficient    = false;
    bool                              has_laplace_coefficient = false;
    Table<2, VectorizedArray<Number>> mass_matrix_coefficient;
    Table<2, VectorizedArray<Number>> laplace_matrix_coefficient;
  };
} // namespace dealii
