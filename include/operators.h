// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by Nils Margenberg and Peter Munch

#pragma once
#include <deal.II/base/subscriptor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/diagonal_matrix.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/numerics/vector_tools.h>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#include "compute_block_matrix.h"
#include "parameters.h"
#include "types.h"

namespace dealii
{
  template <typename Number, std::size_t width>
  inline DEAL_II_ALWAYS_INLINE VectorizedArray<Number, width>
  operator>=(const VectorizedArray<Number, width> &lhs,
             const VectorizedArray<Number, width> &rhs)
  {
    VectorizedArray<Number, width> tmp;
    for (unsigned int i = 0; i < VectorizedArray<Number, width>::size(); ++i)
      tmp[i] = static_cast<Number>(lhs[i] >= rhs[i]);

    return tmp;
  }

  template <typename Number, std::size_t width>
  inline DEAL_II_ALWAYS_INLINE VectorizedArray<Number, width>
  operator<(const VectorizedArray<Number, width> &lhs,
            const VectorizedArray<Number, width> &rhs)
  {
    VectorizedArray<Number, width> tmp;
    for (unsigned int i = 0; i < VectorizedArray<Number, width>::size(); ++i)
      tmp[i] = static_cast<Number>(lhs[i] < rhs[i]);

    return tmp;
  }

  namespace internal
  {
    template <int dim, typename Number>
    void
    set_task_parallel_scheme(
      MPI_Comm const                                   &comm,
      typename MatrixFree<dim, Number>::AdditionalData &additional_data)
    {
      if (Utilities::MPI::n_mpi_processes(comm) > 1)
        additional_data.tasks_parallel_scheme =
          MatrixFree<dim, Number>::AdditionalData::TasksParallelScheme::none;
      else
        additional_data.tasks_parallel_scheme = MatrixFree<dim, Number>::
          AdditionalData::TasksParallelScheme::partition_partition;
    }

    template <int dim, typename Number, typename T, typename F>
    void
    set_values(T &v, const Point<dim, Number> &p, F &&fill_value)
    {
      for (unsigned int i = 0; i < p[0].size(); ++i)
        {
          Point<dim> point;
          for (unsigned int d = 0; d < dim; ++d)
            point[d] = p[d][i];
          fill_value(v, i, point);
        }
    }

    template <int dim, typename Number>
    void
    set_2nd_rank(Tensor<2, dim, Number>                           &v,
                 const Point<dim, Number>                         &p,
                 const Function<dim, typename Number::value_type> &g)
    {
      set_values(v, p, [&g](auto &v, unsigned int i, const auto &point) {
        for (unsigned int r = 0; r < dim; ++r)
          for (unsigned int c = 0; c < dim; ++c)
            v[r][c][i] = g.value(point, c + dim * r);
      });
    }


    template <int dim, typename Number>
    void
    set_vector(Tensor<1, dim, Number>                           &v,
               const Point<dim, Number>                         &p,
               const Function<dim, typename Number::value_type> &g)
    {
      set_values(v, p, [&g](auto &v, unsigned int i, const auto &point) {
        for (unsigned int d = 0; d < dim; ++d)
          v[d][i] = g.value(point, d);
      });
    }

    template <int dim, typename Number>
    void
    set_scalar(Number                                           &v,
               const Point<dim, Number>                         &p,
               const Function<dim, typename Number::value_type> &g)
    {
      set_values(v, p, [&g](auto &v, unsigned int i, const auto &point) {
        v[i] = g.value(point);
      });
    }

    template <int dim, typename Number>
    void
    set_scalar_gradient(Tensor<1, dim, Number>                           &v,
                        const Point<dim, Number>                         &p,
                        const Function<dim, typename Number::value_type> &g)
    {
      set_values(v, p, [&g](auto &v, unsigned int i, const auto &point) {
        auto g_grad = g.gradient(point, 0);
        for (unsigned int d = 0; d < dim; ++d)
          v[d][i] = g_grad[d];
      });
    }

    template <int dim, typename Number>
    void
    set_vector_gradient(Tensor<2, dim, Number>                           &v,
                        const Point<dim, Number>                         &p,
                        const Function<dim, typename Number::value_type> &g)
    {
      set_values(v, p, [&g](auto &v, unsigned int i, const auto &point) {
        for (unsigned int r = 0; r < dim; ++r)
          {
            auto g_grad = g.gradient(point, r);
            for (unsigned int c = 0; c < dim; ++c)
              v[r][c][i] = g_grad[c];
          }
      });
    }

    template <typename Number>
    void
    scatter(BlockVectorT<Number>       &dst,
            const BlockVectorT<Number> &tmp,
            const FullMatrix<Number>   &matrix,
            const BlockSlice           &blk_slice,
            unsigned int                jt,
            unsigned int                jv,
            unsigned int                jd,
            unsigned int                it,
            unsigned int                iv,
            unsigned int                id)
    {
      unsigned int j = blk_slice.index(jt, jv, jd);
      unsigned int i = blk_slice.index(it, iv, id);
      if (std::abs(matrix(j, i)) > 10 * std::numeric_limits<Number>::epsilon())
        dst.block(j).add(matrix(j, i), tmp.block(jv));
    }

    template <typename Number>
    void
    scatter(BlockVectorT<Number>       &dst,
            const BlockVectorT<Number> &tmp,
            const FullMatrix<Number>   &matrix,
            const BlockSlice           &blk_slice,
            unsigned int                v,
            unsigned int                jt,
            unsigned int                jd,
            unsigned int                it,
            unsigned int                id)
    {
      scatter(dst, tmp, matrix, blk_slice, jt, v, jd, it, v, id);
    }

    template <typename T, typename Number, typename = std::void_t<>>
    struct has_set_data : std::false_type
    {};

    template <typename T, typename Number>
    struct has_set_data<T,
                        Number,
                        std::void_t<decltype(std::declval<T>().set_data(
                          std::declval<BlockVectorSliceT<Number>>()))>>
      : std::true_type
    {};

    template <typename T, typename Number, typename = std::void_t<>>
    struct has_form : std::false_type
    {};

    template <typename T, typename Number>
    struct has_form<T,
                    Number,
                    std::void_t<decltype(std::declval<T>().form(
                      std::declval<BlockVectorT<Number> &>(),
                      std::declval<BlockVectorT<Number> const &>()))>>
      : std::true_type
    {};

    template <typename T, typename Number, typename = std::void_t<>>
    struct has_form_vector : std::false_type
    {};

    template <typename T, typename Number>
    struct has_form_vector<T,
                           Number,
                           std::void_t<decltype(std::declval<T>().form(
                             std::declval<VectorT<Number> &>(),
                             std::declval<VectorT<Number> const &>()))>>
      : std::true_type
    {};
  } // namespace internal

  template <int dim, int n_components, typename Number>
  using DirichletValue =
    typename std::conditional<n_components == dim,
                              Tensor<1, dim, VectorizedArray<Number>>,
                              VectorizedArray<Number>>::type;

  template <int dim, int n_components, typename Number>
  using DirichletGradient =
    typename std::conditional<n_components == dim,
                              Tensor<2, dim, VectorizedArray<Number>>,
                              Tensor<1, dim, VectorizedArray<Number>>>::type;


  template <int dim, typename Number>
  void
  get_h_cell(AlignedVector<VectorizedArray<Number>> &array_h,
             MatrixFree<dim, Number> const          &matrix_free,
             unsigned int const                      dof_index)
  {
    unsigned int n_cells =
      matrix_free.n_cell_batches() + matrix_free.n_ghost_cell_batches();
    array_h.resize(n_cells);

    FEEvaluation<dim, -1, 0, dim, Number> fe_values(matrix_free);
    for (unsigned int cell = 0; cell < matrix_free.n_cell_batches() +
                                         matrix_free.n_ghost_cell_batches();
         ++cell)
      {
        fe_values.reinit(cell);
        Number volume = 0;
        for (unsigned int q = 0; q < fe_values.n_q_points; ++q)
          volume += fe_values.JxW(q);

        array_h[cell] = pow(volume, 1.0 / dim);
      }
  }


  template <int dim, typename Number, int n_components = 1>
  void
  get_h_face(AlignedVector<VectorizedArray<Number>> &array_h,
             MatrixFree<dim, Number> const          &matrix_free,
             unsigned int const                      dof_index = 0)
  {
    unsigned int n_face = matrix_free.n_inner_face_batches() +
                          matrix_free.n_boundary_face_batches() +
                          matrix_free.n_ghost_inner_face_batches();
    array_h.resize(n_face);
    // Create FEFaceEvaluation objects for inner and boundary faces
    FEFaceEvaluation<dim, -1, 0, n_components, Number> fe_face(matrix_free,
                                                               true,
                                                               dof_index);

    const unsigned int total_face_batches =
      matrix_free.n_inner_face_batches() +
      matrix_free.n_boundary_face_batches();
    for (unsigned int face = 0; face < total_face_batches; ++face)
      {
        fe_face.reinit(face);
        VectorizedArray<Number> area = 0;
        for (unsigned int q = 0; q < fe_face.n_q_points; ++q)
          area += fe_face.JxW(q);

        array_h[face] = std::pow(area, static_cast<Number>(1.0 / (dim - 1)));
      }
  }

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
  void
  tensorproduct_add(MutableBlockVectorSliceT<Number> &c,
                    FullMatrix<Number> const         &A,
                    BlockVectorSliceT<Number> const  &b,
                    int                               block_offset = 0)
  {
    const unsigned int n_blocks = A.n();
    const unsigned int m_blocks = A.m();
    for (unsigned int i = 0; i < m_blocks; ++i)
      for (unsigned int j = 0; j < n_blocks; ++j)
        if (A(i, j) != 0.0)
          c[block_offset + i].get().add(A(i, j), b[block_offset + j].get());
  }

  template <typename Number>
  void
  tensorproduct(MutableBlockVectorSliceT<Number> &c,
                FullMatrix<Number> const         &A,
                BlockVectorSliceT<Number> const  &b,
                int                               block_offset = 0)
  {
    const unsigned int n_blocks = A.n();
    const unsigned int m_blocks = A.m();
    for (unsigned int i = 0; i < m_blocks; ++i)
      {
        c[block_offset + i].get() = 0.0;
        for (unsigned int j = 0; j < n_blocks; ++j)
          if (A(i, j) != 0.0)
            c[block_offset + i].get().add(A(i, j), b[block_offset + j].get());
      }
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

  template <typename T, typename = void>
  struct HasLocallyOwnedDomainIndices : std::false_type
  {};

  template <typename T>
  struct HasLocallyOwnedDomainIndices<
    T,
    std::void_t<decltype(std::declval<T>().locally_owned_domain_indices())>>
    : std::true_type
  {};

  template <typename T, typename VectorType, typename = void>
  struct HasInitializeDofVector : std::false_type
  {};

  template <typename T, typename VectorType>
  struct HasInitializeDofVector<
    T,
    VectorType,
    std::void_t<decltype(std::declval<T>().initialize_dof_vector(
      std::declval<VectorType &>()))>> : std::true_type
  {};
  using internal::AffineConstraints::IsBlockMatrix;

  enum class OperatorMode
  {
    none,
    jacobian,
    form
  };

  template <int dim,
            typename Number,
            typename SystemMatrixTypeK,
            typename SystemMatrixTypeM = SystemMatrixTypeK>
  class SystemMatrixBase : public Subscriptor
  {
  public:
    using BlockVectorType = BlockVectorT<Number>;
    using VectorType      = VectorT<Number>;

    SystemMatrixBase(
      TimerOutput              &timer,
      const SystemMatrixTypeK  &K,
      const SystemMatrixTypeM  &M,
      const FullMatrix<Number> &Alpha_,
      const FullMatrix<Number> &Beta_,
      BlockSlice                blk_slice_           = {},
      NonlinearTreatment        nonlinear_treatment_ = NonlinearTreatment::None)
      : timer(timer)
      , K(K)
      , M(M)
      , Alpha(Alpha_)
      , Beta(Beta_)
      , blk_slice(blk_slice_)
      , nonlinear_treatment(nonlinear_treatment_)
      , nonlinear(nonlinear_treatment != NonlinearTreatment::None)
      , alpha_is_zero(Alpha.all_zero())
      , beta_is_zero(Beta.all_zero())
    {
      AssertDimension(Alpha.m(), Beta.m());
      AssertDimension(Alpha.n(), Beta.n());
    }

    virtual void
    vmult(BlockVectorType &dst, const BlockVectorType &src) const = 0;

    virtual void
    form(BlockVectorType &dst, const BlockVectorType &src) const
    {
      this->vmult(dst, src);
    }

    virtual void
    Tvmult(BlockVectorType &dst, const BlockVectorType &src) const = 0;

    // Specialization for a nx1 matrix. Useful for rhs assembly
    virtual void
    vmult_slice_add(BlockVectorType &dst, BlockVectorType const &src) const = 0;

    void
    vmult_slice(BlockVectorType &dst, BlockVectorType const &src) const
    {
      dst = 0.0;
      vmult_slice_add(dst, src);
    }

    void
    set_data(BlockVectorType const &solution_linearization_) const
    {
      solution_linearization = &solution_linearization_;
    }

    virtual types::global_dof_index
    m() const
    {
      return Alpha.m() * std::max(K.m(), M.m());
    }

    virtual types::global_dof_index
    n() const
    {
      AssertDimension(m(), Alpha.n() * std::max(K.m(), M.m()));
      return m();
    }

    virtual Number
    el(unsigned int, unsigned int) const
    {
      Assert(false, ExcNotImplemented());
      return 0.0;
    }

    virtual std::shared_ptr<DiagonalMatrix<BlockVectorType>>
    get_matrix_diagonal() const
    {
      DEAL_II_NOT_IMPLEMENTED();
    }
    virtual std::shared_ptr<DiagonalMatrix<BlockVectorType>>
    get_matrix_diagonal_inverse() const
    {
      DEAL_II_NOT_IMPLEMENTED();
    }

    template <typename Number2>
    void
    initialize_spatial_dof_vector(VectorT<Number2> &dst) const
    {
      if constexpr (HasLocallyOwnedDomainIndices<SystemMatrixTypeK>::value)
        dst.reinit(K.locally_owned_domain_indices());
      else if constexpr (HasInitializeDofVector<SystemMatrixTypeK,
                                                VectorType>::value)
        K.initialize_dof_vector(dst);
      else
        DEAL_II_NOT_IMPLEMENTED();
    }

    template <typename Number2>
    void
    initialize_spatial_dof_vector(VectorT<Number2> &dst, unsigned int v) const
    {
      Assert(v <= blk_slice.n_variables() - 1,
             ExcLowerRange(v, blk_slice.n_variables()));
      if constexpr (IsBlockMatrix<SystemMatrixTypeK>::value)
        dst.reinit(K.block(0, v).locally_owned_domain_indices(),
                   K.block(0, v).get_mpi_communicator());
      else if constexpr (HasInitializeDofVector<SystemMatrixTypeK,
                                                BlockVectorType>::value)
        K.initialize_dof_vector(dst, v);
      else
        DEAL_II_NOT_IMPLEMENTED();
    }

    template <typename Number2>
    void
    initialize_spatial_dof_vector(BlockVectorT<Number2> &dst) const
    {
      if constexpr (IsBlockMatrix<SystemMatrixTypeK>::value)
        for (unsigned int i = 0; i < K.n_block_cols(); ++i)
          initialize_spatial_dof_vector(dst.block(i), i);
      else if constexpr (HasInitializeDofVector<SystemMatrixTypeK,
                                                BlockVectorType>::value)
        K.initialize_dof_vector(dst);
      else
        DEAL_II_NOT_IMPLEMENTED();
    }

  protected:
    TimerOutput              &timer;
    const SystemMatrixTypeK  &K;
    const SystemMatrixTypeM  &M;
    const FullMatrix<Number> &Alpha;
    const FullMatrix<Number> &Beta;

    template <typename T>
    void
    set_linearization_data(T                     &member,
                           BlockVectorType const &vec,
                           unsigned int           it,
                           unsigned int           id) const
    {
      if (nonlinear)
        {
          if constexpr (internal::has_set_data<T, Number>::value)
            {
              AssertDimension(blk_slice.n_blocks(), vec.n_blocks());
              member.set_data(blk_slice.get_variable(vec, it, id));
            }
          else
            Assert(false, ExcInternalError());
        }
    }

    template <typename T>
    void
    set_linearization_data(T &member, VectorType const &vec) const
    {
      if (nonlinear)
        {
          if constexpr (internal::has_set_data<T, Number>::value)
            {
              BlockVectorSliceT<Number> vec_{std::cref(vec)};
              member.set_data(std::move(vec_));
            }
          else
            Assert(false, ExcInternalError());
        }
    }



    template <typename T>
    void
    set_linearization_data_slice(T &member, BlockVectorType const &vec) const
    {
      if (nonlinear)
        {
          if constexpr (internal::has_set_data<T, Number>::value)
            {
              BlockVectorSliceT<Number> v_slice;
              v_slice.reserve(vec.n_blocks());
              for (unsigned int v = 0; v < vec.n_blocks(); ++v)
                v_slice.push_back(vec.block(v));
              member.set_data(v_slice);
            }
          else
            Assert(false, ExcInternalError());
        }
    }

    mutable BlockVectorType const *solution_linearization;

    BlockSlice         blk_slice;
    NonlinearTreatment nonlinear_treatment;
    bool               nonlinear;
    // Only used for nx1: small optimization to avoid unnecessary vmult
    bool alpha_is_zero;
    bool beta_is_zero;
  };

  template <int dim,
            typename Number,
            typename SystemMatrixTypeK,
            typename SystemMatrixTypeM = SystemMatrixTypeK>
  class SystemMatrix final
    : public SystemMatrixBase<dim, Number, SystemMatrixTypeK, SystemMatrixTypeM>
  {
  public:
    using BlockVectorType = BlockVectorT<Number>;
    using VectorType      = VectorT<Number>;

    SystemMatrix(
      TimerOutput              &timer,
      const SystemMatrixTypeK  &K,
      const SystemMatrixTypeM  &M,
      const FullMatrix<Number> &Alpha_,
      const FullMatrix<Number> &Beta_,
      BlockSlice                blk_slice_           = {},
      NonlinearTreatment        nonlinear_treatment_ = NonlinearTreatment::None)
      : SystemMatrixBase<dim, Number, SystemMatrixTypeK, SystemMatrixTypeM>(
          timer,
          K,
          M,
          Alpha_,
          Beta_,
          blk_slice_,
          nonlinear_treatment_)
    {}

    virtual void
    vmult(BlockVectorType &dst, const BlockVectorType &src) const override
    {
      tensorproduct_eval(dst, src, true);
    }

    virtual void
    form(BlockVectorType &dst, const BlockVectorType &src) const override
    {
      tensorproduct_eval(dst, src, false);
    }

    void
    tensorproduct_eval(BlockVectorType       &dst,
                       const BlockVectorType &src,
                       bool                   is_vmult) const
    {
      TimerOutput::Scope scope(this->timer, "vmult");

      const unsigned int n_blocks = src.n_blocks();
      AssertDimension(this->Alpha.m(), n_blocks);
      dst = 0.0;
      VectorType tmp;
      this->initialize_spatial_dof_vector(tmp);
      for (unsigned int i = 0; i < n_blocks; ++i)
        {
          this->set_linearization_data(this->K,
                                       this->solution_linearization->block(i));

          if constexpr (internal::has_form_vector<SystemMatrixTypeK,
                                                  Number>::value)
            {
              if (is_vmult)
                this->K.form(tmp, src.block(i));
              else
                this->K.vmult(tmp, src.block(i));
            }
          else
            this->K.vmult(tmp, src.block(i));

          for (unsigned int j = 0; j < n_blocks; ++j)
            if (this->Alpha(j, i) != 0.0)
              dst.block(j).add(this->Alpha(j, i), tmp);


          if constexpr (internal::has_form_vector<SystemMatrixTypeM,
                                                  Number>::value)
            {
              if (is_vmult)
                this->M.form(tmp, src.block(i));
              else
                this->M.vmult(tmp, src.block(i));
            }
          else
            this->M.vmult(tmp, src.block(i));

          for (unsigned int j = 0; j < n_blocks; ++j)
            if (this->Beta(j, i) != 0.0)
              dst.block(j).add(this->Beta(j, i), tmp);
        }
    }

    virtual void
    Tvmult(BlockVectorType &dst, const BlockVectorType &src) const override
    {
      TimerOutput::Scope scope(this->timer, "Tvmult");

      const unsigned int n_blocks = src.n_blocks();

      dst = 0.0;
      VectorType tmp;
      this->initialize_spatial_dof_vector(tmp);
      for (unsigned int i = 0; i < n_blocks; ++i)
        {
          this->K.vmult(tmp, src.block(i));
          for (unsigned int j = 0; j < n_blocks; ++j)
            if (this->Alpha(i, j) != 0.0)
              dst.block(j).add(this->Alpha(i, j), tmp);

          this->M.vmult(tmp, src.block(i));
          for (unsigned int j = 0; j < n_blocks; ++j)
            if (this->Beta(i, j) != 0.0)
              dst.block(j).add(this->Beta(i, j), tmp);
        }
    }

    // Specialization for a nx1 matrix. Useful for rhs assembly
    virtual void
    vmult_slice_add(BlockVectorType       &dst,
                    BlockVectorType const &src) const override
    {
      TimerOutput::Scope scope(this->timer, "vmult");

      const unsigned int n_blocks = dst.n_blocks();

      VectorType tmp;
      this->initialize_spatial_dof_vector(tmp);
      if (!this->alpha_is_zero)
        {
          this->set_linearization_data_slice(this->K,
                                             *this->solution_linearization);
          if constexpr (internal::has_form_vector<SystemMatrixTypeK,
                                                  Number>::value)
            this->K.form(tmp, src.block(0));
          else
            this->K.vmult(tmp, src.block(0));
          for (unsigned int j = 0; j < n_blocks; ++j)
            if (this->Alpha(j, 0) != 0.0)
              dst.block(j).add(this->Alpha(j, 0), tmp);
        }

      if (!this->beta_is_zero)
        {
          this->M.vmult(tmp, src.block(0));
          for (unsigned int j = 0; j < n_blocks; ++j)
            if (this->Beta(j, 0) != 0.0)
              dst.block(j).add(this->Beta(j, 0), tmp);
        }
    }

    virtual std::shared_ptr<DiagonalMatrix<BlockVectorType>>
    get_matrix_diagonal() const override
    {
      BlockVectorType vec(this->Alpha.m());
      for (unsigned int i = 0; i < this->Alpha.m(); ++i)
        {
          vec.block(i) = this->K.get_matrix_diagonal()->get_vector();
          vec.block(i).sadd(this->Alpha(i, i),
                            this->Beta(i, i),
                            this->M.get_matrix_diagonal()->get_vector());
        }
      return std::make_shared<DiagonalMatrix<BlockVectorType>>(vec);
    }

    virtual std::shared_ptr<DiagonalMatrix<BlockVectorType>>
    get_matrix_diagonal_inverse() const override
    {
      BlockVectorType vec(this->Alpha.m());
      for (unsigned int i = 0; i < this->Alpha.m(); ++i)
        {
          vec.block(i) = this->K.get_matrix_diagonal_inverse()->get_vector();
          vec.block(i).sadd(
            1. / this->Alpha(i, i),
            1. / this->Beta(i, i),
            this->M.get_matrix_diagonal_inverse()->get_vector());
        }
      return std::make_shared<DiagonalMatrix<BlockVectorType>>(vec);
    }

    virtual types::global_dof_index
    m() const override
    {
      return this->Alpha.m() * this->M.m();
    }

    template <typename Number2>
    void
    initialize_dof_vector(VectorT<Number2> &vec, unsigned int = 1) const
    {
      this->initialize_spatial_dof_vector(vec);
    }

    template <typename Number2>
    void
    initialize_dof_vector(BlockVectorT<Number2> &vec) const
    {
      vec.reinit(this->Alpha.m());
      for (unsigned int i = 0; i < vec.n_blocks(); ++i)
        this->initialize_spatial_dof_vector(vec.block(i));
    }
  };


  template <int dim,
            typename Number,
            typename StokesMatrixType,
            typename MassMatrixType>
  class SystemMatrixStokes final
    : public SystemMatrixBase<dim, Number, StokesMatrixType, MassMatrixType>
  {
  public:
    using BlockVectorType = BlockVectorT<Number>;
    using VectorType      = VectorT<Number>;

    SystemMatrixStokes(
      TimerOutput              &timer,
      const StokesMatrixType   &K,
      const MassMatrixType     &M,
      const FullMatrix<Number> &Alpha_,
      const FullMatrix<Number> &Beta_,
      const BlockSlice         &blk_slice_,
      NonlinearTreatment        nonlinear_treatment_ = NonlinearTreatment::None)
      : SystemMatrixBase<dim, Number, StokesMatrixType, MassMatrixType>(
          timer,
          K,
          M,
          Alpha_,
          Beta_,
          blk_slice_,
          nonlinear_treatment_)
    {}


    virtual void
    vmult(BlockVectorType &dst, const BlockVectorType &src) const override
    {
      tensorproduct_eval(dst, src, true);
    }

    virtual void
    form(BlockVectorType &dst, const BlockVectorType &src) const override
    {
      tensorproduct_eval(dst, src, false);
    }

    virtual void
    Tvmult(BlockVectorType &dst, const BlockVectorType &src) const override
    {
      TimerOutput::Scope scope(this->timer, "vmult");
      auto const        &blk_slice = this->blk_slice;
      AssertDimension(this->Alpha.m(), src.n_blocks());
      dst = 0.0;
      BlockVectorType tmp(2);
      this->initialize_spatial_dof_vector(tmp);
      BlockVectorType tmp_src(tmp.n_blocks());

      auto const &n_timesteps_at_once = blk_slice.n_timesteps_at_once();
      for (unsigned int it = 0; it < n_timesteps_at_once; ++it)
        for (unsigned int id = 0; id < blk_slice.n_timedofs(); ++id)
          {
            auto slice =
              blk_slice.get_variable(const_cast<BlockVectorType &>(src),
                                     it,
                                     id);
            swap(tmp_src, slice);
            //  Stokes
            this->K.vmult(tmp, tmp_src);
            for (unsigned int jt = 0; jt < n_timesteps_at_once; ++jt)
              for (unsigned int jd = 0; jd < blk_slice.n_timedofs(); ++jd)
                for (unsigned int v = 0; v < blk_slice.n_variables(); ++v)
                  internal::scatter(
                    dst, tmp, this->Alpha, blk_slice, v, it, id, jt, jd);

            // \partial_t v
            this->M.vmult(tmp.block(0), tmp_src.block(0));
            for (unsigned int jt = 0; jt < n_timesteps_at_once; ++jt)
              for (unsigned int jd = 0; jd < blk_slice.n_timedofs(); ++jd)
                internal::scatter(
                  dst, tmp, this->Beta, blk_slice, 0, it, id, jt, jd);

            swap(slice, tmp_src);
          }
    }

    // Specialization for a nx1 matrix. Useful for rhs assembly
    virtual void
    vmult_slice_add(BlockVectorType       &dst,
                    BlockVectorType const &src) const override
    {
      TimerOutput::Scope scope(this->timer, "vmult");
      auto const        &blk_slice = this->blk_slice;
      AssertDimension(this->Alpha.n(), 1);
      AssertDimension(src.n_blocks(), blk_slice.n_variables());

      BlockVectorType tmp(2);
      this->initialize_spatial_dof_vector(tmp);
      if (!this->alpha_is_zero)
        this->set_linearization_data_slice(this->K,
                                           *this->solution_linearization);

      for (unsigned int it = 0; it < blk_slice.n_timesteps_at_once(); ++it)
        for (unsigned int id = 0; id < blk_slice.n_timedofs(); ++id)
          {
            // Stokes
            if (!this->alpha_is_zero)
              {
                this->K.form(tmp, src);
                for (unsigned int v = 0; v < blk_slice.n_variables(); ++v)
                  internal::scatter(
                    dst, tmp, this->Alpha, blk_slice, it, v, id, 0, 0, 0);
              }
            // \partial_t v
            if (!this->beta_is_zero)
              {
                this->M.vmult(tmp.block(0), src.block(0));
                internal::scatter(
                  dst, tmp, this->Beta, blk_slice, it, 0, id, 0, 0, 0);
              }
          }
    }

    virtual std::shared_ptr<DiagonalMatrix<BlockVectorType>>
    get_matrix_diagonal() const override
    {
      Assert(false, ExcNotImplemented());
      return std::make_shared<DiagonalMatrix<BlockVectorType>>();
    }

    virtual std::shared_ptr<DiagonalMatrix<BlockVectorType>>
    get_matrix_diagonal_inverse() const override
    {
      Assert(false, ExcNotImplemented());
      return std::make_shared<DiagonalMatrix<BlockVectorType>>();
    }

    template <typename Number2>
    void
    initialize_dof_vector(VectorT<Number2> &vec, unsigned int variable) const
    {
      this->initialize_spatial_dof_vector(vec, variable);
    }

    template <typename Number2>
    void
    initialize_dof_vector(BlockVectorT<Number2> &vec) const
    {
      auto const &blk_slice = this->blk_slice;
      vec.reinit(this->Alpha.m());
      for (unsigned int i = 0; i < this->Alpha.m(); ++i)
        {
          auto const &[tsp, v, td] = blk_slice.decompose(i);
          initialize_dof_vector(vec.block(i), v);
        }
    }

  private:
    void
    tensorproduct_eval(BlockVectorType       &dst,
                       BlockVectorType const &src,
                       bool                   is_vmult) const
    {
      TimerOutput::Scope scope(this->timer, "vmult");
      auto const        &blk_slice = this->blk_slice;
      AssertDimension(this->Alpha.m(), src.n_blocks());
      dst = 0.0;
      BlockVectorType tmp(2);
      this->initialize_spatial_dof_vector(tmp);
      BlockVectorType tmp_src(tmp.n_blocks());
      if (this->nonlinear)
        Assert(this->solution_linearization != nullptr,
               ExcMessage("No linearization set"));
      auto const &n_timesteps_at_once = blk_slice.n_timesteps_at_once();
      for (unsigned int it = 0; it < n_timesteps_at_once; ++it)
        for (unsigned int id = 0; id < blk_slice.n_timedofs(); ++id)
          {
            auto slice =
              blk_slice.get_variable(const_cast<BlockVectorType &>(src),
                                     it,
                                     id);
            swap(tmp_src, slice);
            this->set_linearization_data(this->K,
                                         *this->solution_linearization,
                                         it,
                                         id);
            // Stokes
            if (is_vmult)
              this->K.vmult(tmp, tmp_src);
            else
              this->K.form(tmp, tmp_src);
            for (unsigned int jt = 0; jt < n_timesteps_at_once; ++jt)
              for (unsigned int jd = 0; jd < blk_slice.n_timedofs(); ++jd)
                for (unsigned int jv = 0; jv < blk_slice.n_variables(); ++jv)
                  internal::scatter(
                    dst, tmp, this->Alpha, blk_slice, jt, jv, jd, it, 0, id);

            // \partial_t v - linear, so vmult <=> form
            this->M.vmult(tmp.block(0), tmp_src.block(0));
            for (unsigned int jt = 0; jt < n_timesteps_at_once; ++jt)
              for (unsigned int jd = 0; jd < blk_slice.n_timedofs(); ++jd)
                internal::scatter(
                  dst, tmp, this->Beta, blk_slice, 0, jt, jd, it, id);

            swap(slice, tmp_src);
          }
    }
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

  template <int dim, int n_components, typename Number>
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
                       const double                     laplace_matrix_scaling,
                       const Number                     delta0 = 0.0,
                       const Number                     delta1 = 0.0)
      : mass_matrix_scaling(mass_matrix_scaling)
      , laplace_matrix_scaling(laplace_matrix_scaling)
      , has_mass_coefficient(false)
      , has_laplace_coefficient(false)
    {
      mass_matrix_coefficient.clear();
      laplace_matrix_coefficient.clear();
      typename MatrixFree<dim, Number>::AdditionalData additional_data;
      internal::set_task_parallel_scheme<dim, Number>(
        dof_handler.get_communicator(), additional_data);
      additional_data.mapping_update_flags =
        update_values | update_gradients | update_quadrature_points;
      // Right now this is only to avoid errors when the operator is
      // stabilized!
      if (delta0 > 0.0 || delta1 > 0.0)
        additional_data.mapping_update_flags_inner_faces =
          update_values | update_gradients | update_normal_vectors |
          update_quadrature_points;

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
    compute_system_matrix(
      SparseMatrixType                &sparse_matrix,
      AffineConstraints<Number> const *constraints = nullptr) const
    {
      if (constraints == nullptr)
        constraints = &matrix_free.get_affine_constraints();
      MatrixFreeTools::compute_matrix(
        matrix_free,
        *constraints,
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
    using FECellIntegrator = FEEvaluation<dim, -1, 0, n_components, Number>;

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
  template <int dim, typename Number>
  using MatrixFreeOperatorScalar = MatrixFreeOperator<dim, 1, Number>;
  template <int dim, typename Number>
  using MatrixFreeOperatorVector = MatrixFreeOperator<dim, dim, Number>;

  template <int dim, typename Number>
  class StokesMatrixFreeOperator
  {
  public:
    using BlockVectorType = BlockVectorT<Number>;
    using VectorType      = VectorT<Number>;

    StokesMatrixFreeOperator(
      const Mapping<dim>                                   &mapping,
      const std::vector<const DoFHandler<dim> *>           &dof_handlers,
      const std::vector<const AffineConstraints<Number> *> &constraints,
      const std::vector<Quadrature<dim>>                   &quadrature,
      const Number                                          viscosity_,
      Number                                                p_order_,
      Number                                                p_reg_,
      std::set<types::boundary_id> const &weak_boundary_ids_    = {},
      std::set<types::boundary_id> const &outflow_boundary_ids_ = {},
      bool                                symmetric_tensor_     = false,
      bool                                symmetric_nitsche_    = true,
      const Number                        penalty1_             = 20,
      const Number                        penalty2_             = 10,
      const Number                        outflow_penalty_      = 0.0,
      const Number                        delta0_               = 0.0,
      const Number                        delta1_               = 0.0,
      NonlinearTreatment nonlinear_treatment_ = NonlinearTreatment::None)
      : weak_boundary_ids(weak_boundary_ids_)
      , outflow_boundary_ids(outflow_boundary_ids_)
      , symmetric_tensor(symmetric_tensor_)
      , symmetric_nitsche(symmetric_nitsche_)
      , viscosity(viscosity_)
      , p_order(p_order_)
      , p_reg(p_reg_)
      , gamma1(viscosity * penalty1_)
      , gamma2(penalty2_)
      , outflow_penalty(outflow_penalty_)
      , delta0(delta0_)
      , delta1(delta1_)
      , data_access_on_faces(
          (delta0 > 0.0 || delta1 > 0.0) ?
            MatrixFree<dim, Number>::DataAccessOnFaces::gradients :
            MatrixFree<dim, Number>::DataAccessOnFaces::none)
      , nonlinear_treatment(nonlinear_treatment_)
      , nonlinear(nonlinear_treatment != NonlinearTreatment::None)
      , loop_type(!weak_boundary_ids.empty() || delta0 > 0.0 ? LoopType::Full :
                                                               LoopType::Cell)
    {
      typename MatrixFree<dim, Number>::AdditionalData additional_data;
      internal::set_task_parallel_scheme<dim, Number>(
        dof_handlers[0]->get_communicator(), additional_data);
      additional_data.mapping_update_flags = update_values | update_gradients;
      additional_data.mapping_update_flags_boundary_faces =
        update_values | update_gradients | update_normal_vectors |
        update_quadrature_points;
      if (delta0 > 0.0 || delta1 > 0.0)
        additional_data.mapping_update_flags_inner_faces =
          update_values | update_gradients | update_normal_vectors |
          update_quadrature_points;
      matrix_free.reinit(
        mapping, dof_handlers, constraints, quadrature, additional_data);
      get_h_face<dim, Number, dim>(h, matrix_free, 0);
      if (nonlinear)
        {
          velocity_lin =
            std::make_unique<FECellIntegratorU>(this->matrix_free, 0);
          velocity_lin_face =
            std::make_unique<FEFaceIntegratorU>(this->matrix_free, true, 0, 0);
          velocity_lin_face_ex =
            std::make_unique<FEFaceIntegratorU>(this->matrix_free, false, 0, 0);
        }
    }

    template <typename Number2>
    void
    initialize_dof_vector(BlockVectorT<Number2> &vec) const
    {
      vec.reinit(2);
      initialize_dof_vector(vec.block(0), 0);
      initialize_dof_vector(vec.block(1), 1);
    }

    template <typename Number2>
    void
    initialize_dof_vector(
      const std::vector<std::reference_wrapper<VectorT<Number2>>> &vec) const
    {
      initialize_dof_vector(vec[0].get(), 0);
      initialize_dof_vector(vec[1].get(), 1);
    }

    template <typename Number2>
    void
    initialize_dof_vector(VectorT<Number2> &vec, unsigned int variable) const
    {
      matrix_free.initialize_dof_vector(vec, variable);
    }

    void
    form(BlockVectorType &dst, const BlockVectorType &src) const
    {
      if (!nonlinear)
        dispatch_loop<OperatorMode::none>(dst, src);
      else
        dispatch_loop<OperatorMode::form>(dst, src);
    }

    void
    vmult(BlockVectorType &dst, const BlockVectorType &src) const
    {
      if (nonlinear_treatment == NonlinearTreatment::None)
        dispatch_loop<OperatorMode::none>(dst, src);
      else if (nonlinear_treatment == NonlinearTreatment::Explicit)
        dispatch_loop<OperatorMode::form>(dst, src);
      else if (nonlinear_treatment == NonlinearTreatment::Implicit)
        dispatch_loop<OperatorMode::jacobian>(dst, src);
    }

    void
    compute_system_matrix(
      BlockSparseMatrixType                         &sparse_matrix,
      std::vector<const AffineConstraints<Number> *> constraints =
        std::vector<const AffineConstraints<Number> *>()) const
    {
      if (constraints.empty())
        constraints = std::vector<const AffineConstraints<Number> *>{
          &matrix_free.get_affine_constraints(0),
          &matrix_free.get_affine_constraints(1)};

      if (nonlinear_treatment == NonlinearTreatment::None)
        compute_matrix_helper<OperatorMode::none>(sparse_matrix, constraints);
      else if (nonlinear_treatment == NonlinearTreatment::Explicit)
        compute_matrix_helper<OperatorMode::form>(sparse_matrix, constraints);
      else if (nonlinear_treatment == NonlinearTreatment::Implicit)
        compute_matrix_helper<OperatorMode::jacobian>(sparse_matrix,
                                                      constraints);
    }

    types::global_dof_index
    m() const
    {
      return matrix_free.get_dof_handler(0).n_dofs() +
             matrix_free.get_dof_handler(1).n_dofs();
    }

    Number
    el(unsigned int, unsigned int) const
    {
      Assert(false, ExcNotImplemented());
      return 0.0;
    }

    void
    set_data(BlockVectorSliceT<Number> data_slice) const
    {
      Assert(nonlinear, dealii::ExcMessage("not allowed"));
      Assert(data_slice.size(), dealii::ExcInternalError());
      data_lin = data_slice;
      AssertDimension(data_lin[0].get().size(),
                      matrix_free.get_dof_handler(0).n_dofs());
      AssertDimension(data_lin[1].get().size(),
                      matrix_free.get_dof_handler(1).n_dofs());
    }

    Tensor<1, dim, Number>
    compute_drag_lift(BlockVectorSliceT<Number> const &src,
                      std::set<types::boundary_id>     obstacle_id,
                      Number                           scale) const
    {
      auto const &mpi_comm = matrix_free.get_dof_handler().get_communicator();
      auto const &v_v      = src[0].get();
      auto const &p_v      = src[1].get();
      FEFaceIntegratorU velocity(matrix_free, true, 0, 0);
      FEFaceIntegratorP pressure(matrix_free, true, 1, 0);

      Tensor<1, dim, Number> f;
      for (unsigned int face = matrix_free.n_inner_face_batches();
           face < (matrix_free.n_inner_face_batches() +
                   matrix_free.n_boundary_face_batches());
           ++face)
        {
          velocity.reinit(face);
          pressure.reinit(face);
          if (auto it = obstacle_id.find(velocity.boundary_id());
              it != obstacle_id.end())
            {
              velocity.read_dof_values(v_v);
              pressure.read_dof_values(p_v);
              velocity.evaluate(EvaluationFlags::gradients);
              pressure.evaluate(EvaluationFlags::values);
              if (p_order == 2.0)
                for (unsigned int q = 0; q < velocity.n_q_points; ++q)
                  {
                    auto p      = pressure.get_value(q);
                    auto n      = velocity.get_normal_vector(q);
                    auto grad_v = velocity.get_gradient(q);
                    auto tau =
                      p * n - viscosity * (grad_v + transpose(grad_v)) * n;
                    velocity.submit_value(tau, q);
                  }
              else
                for (unsigned int q = 0; q < velocity.n_q_points; ++q)
                  {
                    auto       sym_grad_u = velocity.get_symmetric_gradient(q);
                    auto const p          = pressure.get_value(q);
                    auto const E          = sym_grad_u.norm();
                    auto const mu         = std::sqrt(E * E + p_reg * p_reg);
                    auto const beta =
                      std::pow(mu, static_cast<Number>(p_order - 2.0));
                    auto const nu  = 2.0 * beta * viscosity;
                    auto       n   = velocity.get_normal_vector(q);
                    auto       tau = p * n - nu * sym_grad_u * n;
                    velocity.submit_value(tau, q);
                  }

              auto f_local = velocity.integrate_value();
              for (unsigned int d = 0; d < dim; ++d)
                for (unsigned int n = 0;
                     n < matrix_free.n_active_entries_per_face_batch(face);
                     ++n)
                  f[d] += scale * f_local[d][n];
            }
        }
      f = Utilities::MPI::sum(f, mpi_comm);
      return f;
    }

    double
    compute_divergence(
      VectorType const                       &src,
      LinearAlgebra::ReadWriteVector<Number> &cell_vector) const
    {
      VectorType dummy;
      std::function<void(const MatrixFree<dim, Number> &,
                         VectorType &,
                         const VectorType &,
                         const std::pair<unsigned int, unsigned int> &)>
        cell_op = [this, &cell_vector](
                    const MatrixFree<dim, Number>               &matrix_free,
                    VectorType                                  &dst,
                    const VectorType                            &src,
                    const std::pair<unsigned int, unsigned int> &range) {
          this->divergence_cell_loop(matrix_free, dst, src, range, cell_vector);
        };

      matrix_free.cell_loop(cell_op, dummy, src, false);
      auto divergence =
        std::accumulate(cell_vector.begin(), cell_vector.end(), 0.);
      auto const &mpi_comm = matrix_free.get_dof_handler().get_communicator();
      divergence           = Utilities::MPI::sum(divergence, mpi_comm);
      return std::sqrt(divergence);
    }

    void
    divergence_cell_loop(
      const MatrixFree<dim, Number> &matrix_free,
      VectorType &,
      const VectorType                            &src,
      const std::pair<unsigned int, unsigned int> &range,
      LinearAlgebra::ReadWriteVector<Number>      &cell_vector) const
    {
      FECellIntegratorU velocity(matrix_free, 0);
      for (unsigned int cell = range.first; cell < range.second; ++cell)
        {
          Number divergence = 0;
          velocity.reinit(cell);
          velocity.read_dof_values_plain(src);
          velocity.evaluate(EvaluationFlags::gradients);
          for (unsigned int q = 0; q < velocity.n_q_points; q++)
            {
              auto div_u = velocity.get_divergence(q);
              divergence += (div_u * div_u * velocity.JxW(q)).sum();
            }
          cell_vector[cell] = divergence;
        }
    }

  private:
    enum class LoopType
    {
      Cell,
      Full
    };

    template <OperatorMode mode>
    void
    dispatch_loop(BlockVectorType &dst, const BlockVectorType &src) const
    {
      if (loop_type == LoopType::Full)
        matrix_free.loop(
          &StokesMatrixFreeOperator::do_cell_integral_range<mode>,
          &StokesMatrixFreeOperator::do_face_integral_range,
          &StokesMatrixFreeOperator::do_boundary_integral_range<mode>,
          this,
          dst,
          src,
          true,
          data_access_on_faces,
          data_access_on_faces);
      else if (loop_type == LoopType::Cell)
        matrix_free.cell_loop(
          &StokesMatrixFreeOperator::do_cell_integral_range<mode>,
          this,
          dst,
          src,
          true);
    }

    template <OperatorMode mode>
    void
    compute_matrix_helper(
      BlockSparseMatrixType                                &sparse_matrix,
      const std::vector<const AffineConstraints<Number> *> &constraints) const
    {
      if (!weak_boundary_ids.empty() || delta0 > 0.0)
        MatrixFreeTools::compute_matrix(
          matrix_free,
          constraints,
          sparse_matrix,
          &StokesMatrixFreeOperator::do_cell_integral_local<mode>,
          &StokesMatrixFreeOperator::do_face_integral_local,
          &StokesMatrixFreeOperator::do_boundary_face_integral_local<mode>,
          this);
      else
        MatrixFreeTools::compute_matrix(
          matrix_free,
          constraints,
          sparse_matrix,
          &StokesMatrixFreeOperator::do_cell_integral_local<mode>,
          this);
    }

    using FECellIntegratorP = FEEvaluation<dim, -1, 0, 1, Number>;
    using FECellIntegratorU = FEEvaluation<dim, -1, 0, dim, Number>;
    using FEFaceIntegratorP = FEFaceEvaluation<dim, -1, 0, 1, Number>;
    using FEFaceIntegratorU = FEFaceEvaluation<dim, -1, 0, dim, Number>;

    template <OperatorMode op_mode = OperatorMode::none>
    void
    do_cell_integral_range(
      const MatrixFree<dim, Number>               &matrix_free,
      BlockVectorType                             &dst,
      const BlockVectorType                       &src,
      const std::pair<unsigned int, unsigned int> &range) const
    {
      FECellIntegratorU velocity(matrix_free, 0);
      FECellIntegratorP pressure(matrix_free, 1);
      for (unsigned int cell = range.first; cell < range.second; ++cell)
        {
          velocity.reinit(cell);
          velocity.read_dof_values(src.block(0));
          pressure.reinit(cell);
          pressure.read_dof_values(src.block(1));

          do_cell_integral_local<op_mode>(velocity, pressure);

          velocity.distribute_local_to_global(dst.block(0));
          pressure.distribute_local_to_global(dst.block(1));
        }
    }

    template <OperatorMode op_mode>
    void
    do_cell_integral_local(FECellIntegratorU &velocity,
                           FECellIntegratorP &pressure) const
    {
      if (p_order == 2)
        do_cell_integral_local_ns<op_mode>(velocity, pressure);
      else
        do_cell_integral_local_p_lap<op_mode>(velocity, pressure);

      velocity.integrate(EvaluationFlags::gradients);
      pressure.integrate(EvaluationFlags::values);
    }

    template <OperatorMode op_mode>
    void
    do_cell_integral_local_ns(FECellIntegratorU &velocity,
                              FECellIntegratorP &pressure) const
    {
      auto constexpr navier =
        op_mode == OperatorMode::form || op_mode == OperatorMode::jacobian;
      if (navier)
        {
          Assert(velocity_lin, ExcInternalError());
          velocity_lin->reinit(velocity.get_current_cell_index());
          velocity_lin->read_dof_values(data_lin[0].get());
          velocity_lin->evaluate(EvaluationFlags::values);
          velocity.evaluate(EvaluationFlags::values |
                            EvaluationFlags::gradients);
        }
      else
        velocity.evaluate(EvaluationFlags::gradients);
      pressure.evaluate(EvaluationFlags::values);
      for (unsigned int q = 0; q < velocity.n_q_points; ++q)
        {
          Tensor<2, dim, VectorizedArray<Number>> grad_u =
            symmetric_tensor ? velocity.get_symmetric_gradient(q) :
                               velocity.get_gradient(q);
          auto const p     = pressure.get_value(q);
          auto const div_u = trace(grad_u);
          grad_u *= viscosity;

          for (unsigned int i = 0; i < dim; ++i)
            grad_u[i][i] -= p;
          if (nonlinear)
            {
              if constexpr (op_mode == OperatorMode::jacobian)
                {
                  auto const delta_u    = velocity.get_value(q);
                  auto const u          = velocity_lin->get_value(q);
                  auto const convection = outer_product(u, delta_u);
                  grad_u -= convection;
                  grad_u -= transpose(convection);
                }
              else if constexpr (op_mode == OperatorMode::form)
                {
                  auto const delta_u = velocity.get_value(q);
                  grad_u -= outer_product(delta_u, delta_u);
                }
            }

          pressure.submit_value(div_u, q);
          velocity.submit_gradient(grad_u, q);
        }
    }

    template <OperatorMode op_mode>
    void
    do_cell_integral_local_p_lap(FECellIntegratorU &velocity,
                                 FECellIntegratorP &pressure) const
    {
      Assert(symmetric_tensor, ExcInternalError());
      Assert(velocity_lin, ExcInternalError());
      velocity_lin->reinit(velocity.get_current_cell_index());
      velocity_lin->read_dof_values(data_lin[0].get());
      velocity_lin->evaluate(EvaluationFlags::values |
                             EvaluationFlags::gradients);
      velocity.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

      pressure.evaluate(EvaluationFlags::values);
      for (unsigned int q = 0; q < velocity.n_q_points; ++q)
        {
          auto       sym_grad_u     = velocity.get_symmetric_gradient(q);
          auto const sym_grad_u_lin = velocity_lin->get_symmetric_gradient(q);
          auto const p              = pressure.get_value(q);
          auto const div_u          = trace(sym_grad_u);
          auto const E              = sym_grad_u_lin.norm();
          auto const mu             = std::sqrt(E * E + p_reg * p_reg);
          auto const beta = std::pow(mu, static_cast<Number>(p_order - 2.0));
          auto const nu   = beta * 2.0 * viscosity;

          if constexpr (op_mode == OperatorMode::jacobian)
            {
              auto const sym_grad_delta_u = sym_grad_u;
              sym_grad_u *= nu;
              auto const mup4 =
                std::pow(mu, static_cast<Number>(p_order - 4.0));
              auto const gamma = (p_order - 2.0) * mup4;
              sym_grad_u += 2.0 * viscosity * gamma *
                            scalar_product(sym_grad_u_lin, sym_grad_delta_u) *
                            sym_grad_u_lin;
            }
          else if constexpr (op_mode == OperatorMode::form)
            sym_grad_u *= nu;
          else
            Assert(!nonlinear, ExcMessage("not implemented"));

          for (unsigned int d = 0; d < dim; ++d)
            sym_grad_u[d][d] -= p;

          Tensor<2, dim, VectorizedArray<Number>> total_grad = sym_grad_u;
          auto const u       = velocity_lin->get_value(q);
          auto const delta_u = velocity.get_value(q);

          if constexpr (op_mode == OperatorMode::jacobian)
            {
              auto const convection = outer_product(u, delta_u);
              total_grad -= convection;
              total_grad -= transpose(convection);
            }
          else if constexpr (op_mode == OperatorMode::form)
            total_grad -= outer_product(delta_u, u);

          pressure.submit_value(div_u, q);
          velocity.submit_gradient(total_grad, q);
        }
    }

    void
    do_face_integral_range(
      const MatrixFree<dim, Number>               &matrix_free,
      BlockVectorType                             &dst,
      const BlockVectorType                       &src,
      const std::pair<unsigned int, unsigned int> &range) const
    {
      FEFaceIntegratorU u_in(matrix_free, true, 0, 0);
      FEFaceIntegratorU u_ex(matrix_free, false, 0, 0);
      FEFaceIntegratorP p_in(matrix_free, true, 1, 0);
      FEFaceIntegratorP p_ex(matrix_free, false, 1, 0);
      std::pair<FEFaceIntegratorU &, FEFaceIntegratorU &> u(u_in, u_ex);
      std::pair<FEFaceIntegratorP &, FEFaceIntegratorP &> p(p_in, p_ex);

      for (unsigned int face = range.first; face < range.second; ++face)
        {
          u_in.reinit(face);
          u_ex.reinit(face);
          u_in.read_dof_values(src.block(0));
          u_ex.read_dof_values(src.block(0));

          do_face_integral_local(u, p);

          u_in.distribute_local_to_global(dst.block(0));
          u_ex.distribute_local_to_global(dst.block(0));
        }
    }

    void
    do_face_integral_local(
      std::pair<FEFaceIntegratorU &, FEFaceIntegratorU &> const &u,
      std::pair<FEFaceIntegratorP &, FEFaceIntegratorP &> const &) const
    {
      FEFaceIntegratorU &u_in = u.first;
      FEFaceIntegratorU &u_ex = u.second;
      Assert(u_in.is_interior_face(), ExcInternalError());
      Assert(!u_ex.is_interior_face(), ExcInternalError());
      auto face_in = u_in.get_cell_or_face_batch_id();

      if (nonlinear)
        {
          Assert(velocity_lin_face, ExcInternalError());
          Assert(velocity_lin_face_ex, ExcInternalError());
          velocity_lin_face->reinit(face_in);
          velocity_lin_face->read_dof_values(data_lin[0].get());
          velocity_lin_face->evaluate(EvaluationFlags::values);
          velocity_lin_face_ex->reinit(face_in);
          velocity_lin_face_ex->read_dof_values(data_lin[0].get());
          velocity_lin_face_ex->evaluate(EvaluationFlags::values);
        }

      auto degree = std::pow(u_in.dofs_per_component, 1.0 / dim) - 1;
      auto pa     = degree * degree * degree * std::sqrt(degree);

      u_ex.evaluate(EvaluationFlags::gradients);
      u_in.evaluate(EvaluationFlags::gradients | EvaluationFlags::values);
      for (unsigned int q = 0; q < u_in.n_q_points; ++q)
        {
          auto const normal = u_in.normal_vector(q);
          auto const u_i    = velocity_lin_face->get_value(q);
          auto const u_e    = velocity_lin_face_ex->get_value(q);
          auto const b      = Number(0.5) * (u_i + u_e);
          auto const b_n_sq = (b * normal) * (b * normal);
          auto const w      = (h[face_in] * h[face_in] / pa);
          auto const delta_conv =
            delta0 * w * std::sqrt(b_n_sq + 1.e-2 * b.norm_square());
          auto const delta_full = delta1 * w * b.norm();

          auto const grad_u_i      = u_in.get_gradient(q);
          auto const grad_u_e      = u_ex.get_gradient(q);
          auto const jump_grad_u   = grad_u_i - grad_u_e;
          auto const jump_grad_u_n = jump_grad_u * normal;

          if (delta1 > 0.0)
            {
              Tensor<2, dim, VectorizedArray<Number>> G =
                delta_conv * outer_product(jump_grad_u_n, normal);
              G += delta_full * jump_grad_u;

              u_in.submit_gradient(G, q);
              u_ex.submit_gradient(-G, q);
            }
          else
            {
              u_ex.submit_normal_derivative(-delta_conv * jump_grad_u_n, q);
              u_in.submit_normal_derivative(delta_conv * jump_grad_u_n, q);
            }
        }
      u_in.integrate(EvaluationFlags::gradients);
      u_ex.integrate(EvaluationFlags::gradients);
    }

    template <OperatorMode op_mode>
    void
    do_boundary_integral_range(
      const MatrixFree<dim, Number>               &matrix_free,
      BlockVectorType                             &dst,
      const BlockVectorType                       &src,
      const std::pair<unsigned int, unsigned int> &range) const
    {
      FEFaceIntegratorU velocity(matrix_free, true, 0, 0);
      FEFaceIntegratorP pressure(matrix_free, true, 1, 0);
      for (unsigned int face = range.first; face < range.second; ++face)
        {
          velocity.reinit(face);
          velocity.read_dof_values(src.block(0));
          pressure.reinit(face);
          pressure.read_dof_values(src.block(1));

          do_boundary_face_integral_local<op_mode>(velocity, pressure);

          velocity.distribute_local_to_global(dst.block(0));
          pressure.distribute_local_to_global(dst.block(1));
        }
    }

    template <OperatorMode op_mode>
    void
    do_boundary_face_integral_local(FEFaceIntegratorU &velocity,
                                    FEFaceIntegratorP &pressure) const
    {
      auto const face      = velocity.get_cell_or_face_batch_id();
      auto const curr_b_id = velocity.boundary_id();
      auto const id        = weak_boundary_ids.find(curr_b_id);
      auto const o_id      = outflow_boundary_ids.find(curr_b_id);
      if (o_id != outflow_boundary_ids.end())
        {
          Assert(id == weak_boundary_ids.end(), ExcInternalError());
          if (nonlinear)
            {
              Assert(velocity_lin_face, ExcInternalError());
              velocity_lin_face->reinit(face);
              velocity_lin_face->read_dof_values(data_lin[0].get());
              if (p_order == 2.0)
                velocity_lin_face->evaluate(EvaluationFlags::values);
              else
                velocity_lin_face->evaluate(EvaluationFlags::values |
                                            EvaluationFlags::gradients);
            }
          velocity.evaluate(EvaluationFlags::values |
                            EvaluationFlags::gradients);
          pressure.evaluate(EvaluationFlags::values);
          for (unsigned int q = 0; q < velocity.n_q_points; ++q)
            {
              auto const delta_u      = velocity.get_value(q);
              auto const grad_delta_u = velocity.get_gradient(q);
              auto const p            = pressure.get_value(q);
              auto const normal       = velocity.normal_vector(q);
              Tensor<1, dim, VectorizedArray<Number>> const u =
                nonlinear ? velocity_lin_face->get_value(q) :
                            Tensor<1, dim, VectorizedArray<Number>>();
              auto const sym_grad_u =
                nonlinear ? velocity_lin_face->get_symmetric_gradient(q) :
                            SymmetricTensor<2, dim, VectorizedArray<Number>>();

              Tensor<1, dim, VectorizedArray<Number>> dn;
              VectorizedArray<Number>                 pc = 0.0;
              Tensor<2, dim, VectorizedArray<Number>> gdn;
              auto const                              un = u * normal;
              if constexpr (op_mode == OperatorMode::jacobian)
                {
                  dn += delta_u * un + un * delta_u;
                  // auto const hun = un < VectorizedArray<Number>(0.0);
                  // dn += -hun * (delta_u * un + un * delta_u);
                  dn += -std::min(un, VectorizedArray<Number>(0)) * delta_u;
                }
              else if constexpr (op_mode == OperatorMode::form)
                {
                  dn += delta_u * (normal * delta_u);
                  dn += -std::min(un, VectorizedArray<Number>(0)) * delta_u;
                }
              auto grad_u = grad_delta_u;
              if (p_order != 2.0)
                {
                  auto const E  = sym_grad_u.norm();
                  auto const mu = std::sqrt(E * E + p_reg * p_reg);
                  auto const beta =
                    std::pow(mu, static_cast<Number>(p_order - 2.0));
                  auto const nu = beta * viscosity;
                  grad_u *= nu;
                }
              else
                grad_u *= viscosity;
              for (unsigned int d = 0; d < dim; ++d)
                grad_u[d][d] -= p;
              dn += grad_u * normal;
              velocity.submit_value(dn, q);
              velocity.submit_gradient(gdn, q);
              pressure.submit_value(pc, q);
            }
        }
      else if (id != weak_boundary_ids.end())
        {
          Assert(o_id == outflow_boundary_ids.end(), ExcInternalError());
          if (p_order == 2.0)
            do_boundary_face_integral_local_ns<op_mode>(face,
                                                        velocity,
                                                        pressure);
          else
            do_boundary_face_integral_local_p_lap<op_mode>(face,
                                                           velocity,
                                                           pressure);
        }
      else
        {
          std::fill_n(velocity.begin_values(),
                      velocity.n_q_points * velocity.n_components,
                      Number(0.0));
          std::fill_n(velocity.begin_gradients(),
                      velocity.n_q_points * velocity.n_components *
                        velocity.n_components,
                      Number(0.0));
          std::fill_n(pressure.begin_values(),
                      pressure.n_q_points * pressure.n_components,
                      Number(0.0));
        }

      if (symmetric_nitsche)
        velocity.integrate(EvaluationFlags::values |
                           EvaluationFlags::gradients);
      else
        velocity.integrate(EvaluationFlags::values);
      pressure.integrate(EvaluationFlags::values);
    }


    template <OperatorMode op_mode>
    void
    do_boundary_face_integral_local_p_lap(unsigned int       face,
                                          FEFaceIntegratorU &velocity,
                                          FEFaceIntegratorP &pressure) const
    {
      if (nonlinear)
        {
          Assert(velocity_lin, ExcInternalError());
          velocity_lin_face->reinit(face);
          velocity_lin_face->read_dof_values(data_lin[0].get());
          if (p_order == 2.0)
            velocity_lin_face->evaluate(EvaluationFlags::values);
          else
            velocity_lin_face->evaluate(EvaluationFlags::values |
                                        EvaluationFlags::gradients);
        }
      auto degree = std::pow(velocity.dofs_per_component, 1.0 / dim) - 1;
      auto pa     = degree * degree;

      velocity.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
      pressure.evaluate(EvaluationFlags::values);
      for (unsigned int q = 0; q < velocity.n_q_points; ++q)
        {
          auto const normal = velocity.normal_vector(q);
          auto const u      = velocity.get_value(q);
          auto const p      = pressure.get_value(q);
          auto const sym_grad_u_lin =
            velocity_lin_face->get_symmetric_gradient(q);
          auto const sym_grad_delta_u = velocity.get_symmetric_gradient(q);
          auto const E                = sym_grad_u_lin.norm();
          auto const mu               = std::sqrt(E * E + p_reg * p_reg);
          auto const beta = std::pow(mu, static_cast<Number>(p_order - 2.0));
          auto const nu   = beta * 2.0 * viscosity;

          Tensor<1, dim, VectorizedArray<Number>> nitsche_1;

          if constexpr (op_mode == OperatorMode::jacobian)
            {
              auto const mup4 =
                std::pow(mu, static_cast<Number>(p_order - 4.0));
              auto const gamma = (p_order - 2.0) * mup4;
              auto const flux  = nu * sym_grad_delta_u;
              auto const b     = velocity_lin_face->get_value(q);
              auto const bn    = b * normal;
              nitsche_1        = -flux * normal -
                          2.0 * viscosity * gamma *
                            (sym_grad_u_lin * sym_grad_delta_u) * normal *
                            sym_grad_u_lin +
                          p * normal + (pa * gamma1 / h[face]) * u +
                          (pa * gamma2 / h[face]) * normal * (u * normal);
              nitsche_1 += bn * u + (u * normal) * b;
              // auto const hun = bn < VectorizedArray<Number>(0.0);
              // nitsche_1 += -hun * (bn * u + (u * normal) * b);
              nitsche_1 -= std::min(b * normal, VectorizedArray<Number>(0)) * u;
            }
          else if constexpr (op_mode == OperatorMode::form ||
                             op_mode == OperatorMode::none)
            {
              auto const flux = nu * sym_grad_delta_u;
              auto const b    = velocity_lin_face->get_value(q);

              nitsche_1 = -flux * normal + p * normal +
                          (pa * gamma1 / h[face]) * u +
                          (pa * gamma2 / h[face]) * normal * (u * normal);
              nitsche_1 += (b * normal) * u;
              nitsche_1 -= std::min(b * normal, VectorizedArray<Number>(0)) * u;
            }
          else
            Assert(!nonlinear, ExcMessage("not implemented"));
          auto const nitsche_2 = nu * u;

          velocity.submit_value(nitsche_1, q);
          if (symmetric_nitsche)
            velocity.submit_normal_derivative(-nitsche_2, q);
          pressure.submit_value(-u * normal, q);
        }
    }

    template <OperatorMode op_mode>
    void
    do_boundary_face_integral_local_ns(unsigned int       face,
                                       FEFaceIntegratorU &velocity,
                                       FEFaceIntegratorP &pressure) const
    {
      Assert(!symmetric_tensor, ExcInternalError());
      if (nonlinear)
        {
          Assert(velocity_lin, ExcInternalError());
          velocity_lin_face->reinit(face);
          velocity_lin_face->read_dof_values(data_lin[0].get());
          velocity_lin_face->evaluate(EvaluationFlags::values);
        }
      velocity.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
      pressure.evaluate(EvaluationFlags::values);
      auto degree = std::pow(velocity.dofs_per_component, 1.0 / dim) - 1;
      auto pa     = degree * degree;

      for (unsigned int q = 0; q < velocity.n_q_points; ++q)
        {
          auto const normal = velocity.normal_vector(q);
          auto const u      = velocity.get_value(q);
          auto const p      = pressure.get_value(q);
          Tensor<1, dim, VectorizedArray<Number>> const u_lin =
            nonlinear ? velocity_lin_face->get_value(q) :
                        Tensor<1, dim, VectorizedArray<Number>>();

          auto nitsche_1 = -viscosity * velocity.get_gradient(q) * normal +
                           p * normal + (pa * gamma1 / h[face]) * u +
                           (pa * gamma2 / h[face]) * normal * (u * normal);
          if constexpr (op_mode == OperatorMode::jacobian)
            {
              nitsche_1 += (u_lin * normal) * u + (u * normal) * u_lin;
              nitsche_1 -=
                std::min(u * normal, VectorizedArray<Number>(0)) * u_lin;
            }
          else if constexpr (op_mode == OperatorMode::form)
            {
              nitsche_1 += (u * normal) * u;
              nitsche_1 -=
                std::min(u * normal, VectorizedArray<Number>(0)) * u_lin;
            }

          velocity.submit_value(nitsche_1, q);
          if (symmetric_nitsche)
            velocity.submit_normal_derivative(-viscosity * u, q);
          pressure.submit_value(-u * normal, q);
        }
    }

    MatrixFree<dim, Number>                    matrix_free;
    mutable std::unique_ptr<FECellIntegratorU> velocity_lin;
    mutable std::unique_ptr<FEFaceIntegratorU> velocity_lin_face;
    mutable std::unique_ptr<FEFaceIntegratorU> velocity_lin_face_ex;
    mutable BlockVectorSliceT<Number>          data_lin;
    std::set<types::boundary_id>               weak_boundary_ids{};
    std::set<types::boundary_id>               outflow_boundary_ids{};
    bool                                       symmetric_tensor  = false;
    bool                                       symmetric_nitsche = true;
    Number                                     viscosity         = 1.0;
    Number                                     p_order;
    Number                                     p_reg;
    AlignedVector<VectorizedArray<Number>>     h;
    Number gamma1 = 20, gamma2 = 10, outflow_penalty = 0.0, delta0 = 0.0,
           delta1 = 0.0;
    typename MatrixFree<dim, Number>::DataAccessOnFaces data_access_on_faces;
    NonlinearTreatment                                  nonlinear_treatment;
    bool                                                nonlinear;
    LoopType                                            loop_type;
  };

  template <int dim, typename Number>
  class StokesNitscheMatrixFreeOperator
  {
  public:
    using BlockVectorType = BlockVectorT<Number>;
    using VectorType      = VectorT<Number>;

    StokesNitscheMatrixFreeOperator(
      const Mapping<dim>                                   &mapping,
      const std::vector<const DoFHandler<dim> *>           &dof_handlers,
      const std::vector<const AffineConstraints<Number> *> &constraints,
      const std::vector<Quadrature<dim>>                   &quadrature,
      const Number                                          viscosity_,
      Number                                                p_order_,
      Number                                                p_reg_,
      bool                                                  symmetric_tensor_,
      bool                                                  symmetric_nitsche_,
      const Number                                          penalty1_,
      const Number                                          penalty2_,
      const Number                                          outflow_penalty_,
      const bool                                            is_nonlinear)
      : viscosity(viscosity_)
      , p_order(p_order_)
      , p_reg(p_reg_)
      , symmetric_tensor(symmetric_tensor_)
      , symmetric_nitsche(symmetric_nitsche_)
      , gamma1(viscosity * penalty1_)
      , gamma2(penalty2_)
      , outflow_penalty(outflow_penalty_)
      , nonlinear(is_nonlinear)
    {
      typename MatrixFree<dim, Number>::AdditionalData additional_data;
      internal::set_task_parallel_scheme<dim, Number>(
        dof_handlers[0]->get_communicator(), additional_data);
      additional_data.mapping_update_flags             = update_default;
      additional_data.mapping_update_flags_inner_faces = update_default;
      additional_data.mapping_update_flags_boundary_faces =
        update_values | update_gradients | update_normal_vectors |
        update_quadrature_points;
      matrix_free.reinit(
        mapping, dof_handlers, constraints, quadrature, additional_data);
      get_h_face(h, matrix_free, 1);
    }

    void
    set_dirichlet_functions(
      const std::map<types::boundary_id, Function<dim, Number> *>
        &dirichlet_color_to_fun) const
    {
      this->dirichlet_color_to_fun = &dirichlet_color_to_fun;
    }

    template <typename Number2>
    void
    initialize_dof_vector(BlockVectorT<Number2> &vec) const
    {
      vec.reinit(2);
      initialize_dof_vector(vec.block(0), 0);
      initialize_dof_vector(vec.block(1), 1);
    }

    template <typename Number2>
    void
    initialize_dof_vector(
      const std::vector<std::reference_wrapper<VectorT<Number2>>> &vec) const
    {
      initialize_dof_vector(vec[0].get(), 0);
      initialize_dof_vector(vec[1].get(), 1);
    }

    template <typename Number2>
    void
    initialize_dof_vector(VectorT<Number2> &vec, unsigned int variable) const
    {
      matrix_free.initialize_dof_vector(vec, variable);
    }

    void
    vmult(BlockVectorType &dst) const
    {
      BlockVectorType dummy;
      if (dirichlet_color_to_fun)
        matrix_free.loop(
          &StokesNitscheMatrixFreeOperator::do_cell_integral_range,
          &StokesNitscheMatrixFreeOperator::do_face_integral_range,
          &StokesNitscheMatrixFreeOperator::do_boundary_integral_range,
          this,
          dst,
          dummy,
          true,
          MatrixFree<dim, Number>::DataAccessOnFaces::none,
          MatrixFree<dim, Number>::DataAccessOnFaces::none);
    }

    types::global_dof_index
    m() const
    {
      return matrix_free.get_dof_handler(0).n_dofs() +
             matrix_free.get_dof_handler(1).n_dofs();
    }

    Number
    el(unsigned int, unsigned int) const
    {
      Assert(false, ExcNotImplemented());
      return 0.0;
    }

  private:
    using FECellIntegratorP = FEEvaluation<dim, -1, 0, 1, Number>;
    using FECellIntegratorU = FEEvaluation<dim, -1, 0, dim, Number>;
    using FEFaceIntegratorP = FEFaceEvaluation<dim, -1, 0, 1, Number>;
    using FEFaceIntegratorU = FEFaceEvaluation<dim, -1, 0, dim, Number>;

    void
    do_cell_integral_range(const MatrixFree<dim, Number> &,
                           BlockVectorType &,
                           const BlockVectorType &,
                           const std::pair<unsigned int, unsigned int> &) const
    {}


    void
    do_cell_integral_local(FECellIntegratorU &, FECellIntegratorP &) const
    {}

    void
    do_face_integral_range(const MatrixFree<dim, Number> &,
                           BlockVectorType &,
                           const BlockVectorType &,
                           const std::pair<unsigned int, unsigned int> &) const
    {}

    void
    do_face_integral_local(
      std::pair<FEFaceIntegratorU &, FEFaceIntegratorU &> const &,
      std::pair<FEFaceIntegratorP &, FEFaceIntegratorP &> const &) const
    {}
    void
    do_boundary_integral_range(
      const MatrixFree<dim, Number> &matrix_free,
      BlockVectorType               &dst,
      const BlockVectorType &,
      const std::pair<unsigned int, unsigned int> &range) const
    {
      FEFaceIntegratorU velocity(matrix_free, true, 0, 0);
      FEFaceIntegratorP pressure(matrix_free, true, 1, 0);
      for (unsigned int face = range.first; face < range.second; ++face)
        {
          velocity.reinit(face);
          auto const curr_b_id = velocity.boundary_id();
          if (auto g = dirichlet_color_to_fun->find(curr_b_id);
              g != dirichlet_color_to_fun->end())
            {
              auto degree =
                std::pow(velocity.dofs_per_component, 1.0 / dim) - 1;
              auto pa = degree * degree;

              std::vector<DirichletValue<dim, dim, Number>>    dv;
              std::vector<DirichletGradient<dim, dim, Number>> dv_grad;
              dv.resize(velocity.n_q_points);
              dv_grad.resize(velocity.n_q_points);
              for (unsigned int q = 0; q < velocity.n_q_points; ++q)
                {
                  internal::set_vector(dv[q],
                                       velocity.quadrature_point(q),
                                       *(g->second));
                  internal::set_vector_gradient(dv_grad[q],
                                                velocity.quadrature_point(q),
                                                *(g->second));
                }

              pressure.reinit(face);
              if (p_order != 2.0)
                for (unsigned int q = 0; q < velocity.n_q_points; ++q)
                  {
                    auto        normal = velocity.normal_vector(q);
                    auto const &g      = dv[q];
                    SymmetricTensor<2, dim, VectorizedArray<Number>>
                               sym_g_grad = symmetrize(dv_grad[q]);
                    auto const E          = sym_g_grad.norm();
                    auto const mu         = std::sqrt(E * E + p_reg * p_reg);
                    auto const beta =
                      std::pow(mu, static_cast<Number>(p_order - 2.0));
                    auto const nu = beta * 2.0 * viscosity;

                    auto nitsche_1 =
                      (pa * gamma1 / h[face]) * g +
                      (pa * gamma2 / h[face]) * normal * (g * normal);
                    auto nitsche_2 = nu * g;
                    nitsche_1 -=
                      std::min(g * normal, VectorizedArray<Number>(0)) * g;
                    velocity.submit_value(nitsche_1, q);
                    if (symmetric_nitsche)
                      velocity.submit_normal_derivative(-nitsche_2, q);
                    pressure.submit_value(-g * normal, q);
                  }
              else
                for (unsigned int q = 0; q < velocity.n_q_points; ++q)
                  {
                    auto        normal = velocity.normal_vector(q);
                    auto const &g      = dv[q];
                    auto        nitsche_1 =
                      (pa * gamma1 / h[face]) * g +
                      (pa * gamma2 / h[face]) * normal * (g * normal);
                    // nitsche_1 -=
                    //   std::min(g * normal, VectorizedArray<Number>(0)) * g;
                    velocity.submit_value(nitsche_1, q);
                    if (symmetric_nitsche)
                      velocity.submit_normal_derivative(-viscosity * g, q);
                    pressure.submit_value(-g * normal, q);
                  }

              if (symmetric_nitsche)
                velocity.integrate(EvaluationFlags::values |
                                   EvaluationFlags::gradients);
              else
                velocity.integrate(EvaluationFlags::values);
              pressure.integrate(EvaluationFlags::values);
              velocity.distribute_local_to_global(dst.block(0));
              pressure.distribute_local_to_global(dst.block(1));
            }
        }
    }

    mutable std::map<types::boundary_id, Function<dim, Number> *> const
                           *dirichlet_color_to_fun = nullptr;
    MatrixFree<dim, Number> matrix_free;
    Number                  viscosity = 1.0;
    Number                  p_order;
    Number                  p_reg;
    bool                    symmetric_tensor  = false;
    bool                    symmetric_nitsche = true;
    Number                  gamma1 = 20, gamma2 = 10, outflow_penalty = 0.0;
    bool                    nonlinear;
    AlignedVector<VectorizedArray<Number>> h;
  };

  template <int dim,
            typename Number,
            typename PDEOperator,
            typename JacOperator = PDEOperator>
  class PDE
  {
    using BlockVectorType = BlockVectorT<Number>;
    using VectorType      = VectorT<Number>;

  public:
    void
    init(PDEOperator const &pde_operator, BlockVectorType const &rhs)
    {
      this->pde_operator = &pde_operator;
      this->jac_operator = &pde_operator;
      this->rhs          = &rhs;
    }

    void
    init(PDEOperator const     &pde_operator,
         JacOperator const     &jac_operator,
         BlockVectorType const &rhs)
    {
      this->pde_operator = &pde_operator;
      this->jac_operator = &jac_operator;
      this->rhs          = &rhs;
    }

    void
    set_rhs(BlockVectorType const &rhs) const
    {
      this->rhs = &rhs;
    }

    void
    set_data(BlockVectorType const &data) const
    {
      this->pde_operator->set_data(data);
      if (this->pde_operator != this->jac_operator)
        this->jac_operator->set_data(data);
    }

    /// Residual, i.e. RHS - form.
    void
    residual(BlockVectorType       &dst,
             BlockVectorType const &src,
             BlockVectorType const &rhs) const
    {
      this->rhs = &rhs;
      residual(dst, src);
    }

    /// Residual, i.e. RHS - form.
    void
    residual(BlockVectorType &dst, BlockVectorType const &src) const
    {
      this->form(dst, src);
      dst.sadd(-1.0, 1.0, *this->rhs);
    }

    /// Evaluation of weak form.
    void
    form(BlockVectorType &dst, BlockVectorType const &src) const
    {
      if constexpr (internal::has_form<PDEOperator, Number>::value)
        pde_operator->form(dst, src);
      else
        pde_operator->vmult(dst, src);
    }

    /// vmult with Jacobian.
    void
    vmult(BlockVectorType &dst, BlockVectorType const &src) const
    {
      jac_operator->vmult(dst, src);
    }

    template <typename Number2>
    void
    initialize_dof_vector(VectorT<Number2> &vec, unsigned int i = 0) const
    {
      pde_operator->initialize_dof_vector(vec, i);
    }

    template <typename Number2>
    void
    initialize_dof_vector(BlockVectorT<Number2> &vec) const
    {
      pde_operator->initialize_dof_vector(vec);
    }


  private:
    mutable BlockVectorType const *rhs;
    PDEOperator const             *pde_operator;
    JacOperator const             *jac_operator;
  };

  template <int dim, typename Number>
  using NavierStokesOperator =
    PDE<dim,
        Number,
        SystemMatrixStokes<dim,
                           Number,
                           StokesMatrixFreeOperator<dim, Number>,
                           MatrixFreeOperatorVector<dim, Number>>,
        SystemMatrixStokes<dim,
                           Number,
                           StokesMatrixFreeOperator<dim, Number>,
                           MatrixFreeOperatorVector<dim, Number>>>;

  template <int dim, typename Number>
  using StokesOperator =
    PDE<dim,
        Number,
        SystemMatrixStokes<dim,
                           Number,
                           StokesMatrixFreeOperator<dim, Number>,
                           MatrixFreeOperatorVector<dim, Number>>,
        SystemMatrixStokes<dim,
                           Number,
                           StokesMatrixFreeOperator<dim, Number>,
                           MatrixFreeOperatorVector<dim, Number>>>;

  template <int dim, typename Number>
  using AcousticWaveOperator =
    PDE<dim,
        Number,
        SystemMatrix<dim,
                     Number,
                     MatrixFreeOperatorScalar<dim, Number>,
                     MatrixFreeOperatorScalar<dim, Number>>,
        SystemMatrix<dim,
                     Number,
                     MatrixFreeOperatorScalar<dim, Number>,
                     MatrixFreeOperatorScalar<dim, Number>>>;

  template <int dim, typename Number>
  using HeatOperator = PDE<dim,
                           Number,
                           SystemMatrix<dim,
                                        Number,
                                        MatrixFreeOperatorScalar<dim, Number>,
                                        MatrixFreeOperatorScalar<dim, Number>>,
                           SystemMatrix<dim,
                                        Number,
                                        MatrixFreeOperatorScalar<dim, Number>,
                                        MatrixFreeOperatorScalar<dim, Number>>>;
  template <typename Number>
  void
  boundary_values_map_to_constraints(
    AffineConstraints<Number>                       &constraints,
    const std::map<types::global_dof_index, double> &boundary_values,
    double                                           scale = 1.0)
  {
    for (auto const &[dof, boundary_value] : boundary_values)
      if (constraints.can_store_line(dof) && !constraints.is_constrained(dof))
        {
          constraints.add_line(dof);
          if (boundary_value != 0 && scale != 0)
            constraints.set_inhomogeneity(dof, scale * boundary_value);
        }
  }

  template <typename Number>
  void
  set_inhomogeneity(
    VectorT<Number>                                 &v,
    const std::map<types::global_dof_index, Number> &boundary_values)
  {
    for (auto [id, val] : boundary_values)
      if (v.get_partitioner()->in_local_range(id))
        v[id] = val;

    v.update_ghost_values();
  }

  template <typename Number>
  void
  set_inhomogeneity_zero(
    VectorT<Number>                                 &v,
    const std::map<types::global_dof_index, Number> &boundary_values)
  {
    for (auto [id, val] : boundary_values)
      if (v.get_partitioner()->in_local_range(id))
        v[id] = Number(0);

    v.update_ghost_values();
  }

  template <typename Number>
  void
  set_inhomogeneity(BlockVectorT<Number> &v,
                    const std::vector<std::map<types::global_dof_index, Number>>
                      &boundary_values)
  {
    AssertDimension(v.n_blocks(), boundary_values.size());
    for (unsigned int i = 0; i < v.n_blocks(); ++i)
      set_inhomogeneity(v.block(i), boundary_values[i]);
  }

  template <typename Number>
  void
  set_inhomogeneity_zero(
    BlockVectorT<Number> &v,
    const std::vector<std::map<types::global_dof_index, Number>>
      &boundary_values)
  {
    AssertDimension(v.n_blocks(), boundary_values.size());
    for (unsigned int i = 0; i < v.n_blocks(); ++i)
      set_inhomogeneity_zero(v.block(i), boundary_values[i]);
  }

  template <int dim, typename Number>
  std::map<types::global_dof_index, Number>
  get_inhomogeneous_boundary(
    const Mapping<dim>    &mapping,
    const DoFHandler<dim> &dof,
    std::map<types::boundary_id, const Function<dim, Number> *> const
                        &dirichlet_color_to_fun,
    const ComponentMask &mask = {})
  {
    std::map<types::global_dof_index, Number> boundary_values;
    VectorTools::interpolate_boundary_values(
      mapping, dof, dirichlet_color_to_fun, boundary_values, mask);

    return boundary_values;
  }

  template <int dim, typename Number>
  void
  get_inhomogeneous_boundary(
    std::vector<std::map<types::global_dof_index, Number>> &boundary_values,
    double                                                  t,
    double                                                  dt,
    TimeStepType                                            type,
    unsigned int                                            var,
    const BlockSlice                                       &blk_slice,
    std::map<types::boundary_id, Function<dim, Number> *> const
                          &dirichlet_color_to_fun,
    const Mapping<dim>    &mapping,
    const DoFHandler<dim> &dof,
    const ComponentMask   &mask = {})
  {
    AssertDimension(boundary_values.size(), blk_slice.n_blocks());
    auto shift = type == TimeStepType::DG ? 0 : 1;
    auto qt    = get_time_quad(type,
                            blk_slice.n_timedofs() -
                              (type == TimeStepType::DG ? 1 : 0));
    std::map<types::boundary_id, const Function<dim, Number> *>
      dirichlet_color_to_fun_;
    for (const auto &pair : dirichlet_color_to_fun)
      dirichlet_color_to_fun_.insert(
        {pair.first, static_cast<const Function<dim, Number> *>(pair.second)});

    for (unsigned int it = 0; it < blk_slice.n_timesteps_at_once(); ++it)
      for (unsigned int id = 0; id < blk_slice.n_timedofs(); ++id)
        {
          double time = t + dt * it + dt * qt.point(shift + id)[0];
          for (auto [b_id, g] : dirichlet_color_to_fun)
            g->set_time(time);
          boundary_values[blk_slice.index(it, var, id)] =
            get_inhomogeneous_boundary(mapping,
                                       dof,
                                       dirichlet_color_to_fun_,
                                       mask);
        }
  }
} // namespace dealii
