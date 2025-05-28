#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/repartitioning_policy_tools.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_dgp.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_creator.h>

#include "exact_solution.h"
#include "fe_time.h"
#include "grids.h"
#include "operators.h"
#include "stmg.h"
#include "time_integrators.h"

using namespace dealii;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  dealii::ConditionalOStream       pcout(
    std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  MPI_Comm                                comm = MPI_COMM_WORLD;
  constexpr int                           dim  = 2;
  parallel::distributed::Triangulation<2> tria(comm);
  dealii::GridGenerator::hyper_cube(tria);
  tria.refine_global(3);
  FE_DGP<dim> fe(3);
  auto        poly_mg_sequence =
    get_poly_mg_sequence(3,
                         1,
                         MGTransferGlobalCoarseningTools::
                           PolynomialCoarseningSequenceType::decrease_by_one);
  std::vector<std::vector<std::unique_ptr<FiniteElement<dim>>>> fe_pmg =
    get_fe_pmg_sequence<dim>(poly_mg_sequence,
                             {{1}},
                             FE_DGP<dim>(poly_mg_sequence.back()));
  DoFHandler<dim> dof(tria);
  dof.distribute_dofs(fe);

  AffineConstraints<double>                        constraints;
  MappingQ<dim>                                    mapping(dof.get_fe().degree);
  MatrixFree<dim, double>                          mf_data;
  const QGauss<dim>                                quad(2);
  typename MatrixFree<dim, double>::AdditionalData data;
  MatrixFreeOperatorScalar<dim, double>            M_mf(
    mapping, dof, constraints, quad, 1.0, 0.0);
  auto M = std::make_shared<SparseMatrixType>();

  auto sparsity_pattern =
    std::make_shared<SparsityPatternType>(dof.locally_owned_dofs(),
                                          dof.locally_owned_dofs(),
                                          dof.get_communicator());
  DoFTools::make_sparsity_pattern(dof, *sparsity_pattern, constraints, false);
  sparsity_pattern->compress();
  M->reinit(*sparsity_pattern);
  M_mf.compute_system_matrix(*M);

  RepartitioningPolicyTools::DefaultPolicy<dim>          policy(true);
  std::vector<std::shared_ptr<const Triangulation<dim>>> mg_triangulations =
    MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(
      tria, policy);

  BlockSlice          blk_slice(1, 1, 1);
  std::vector<MGType> mg_type_level(mg_triangulations.size() - 1, MGType::h);
  std::vector<MGType> mg_p_level(fe_pmg.size() - 1, MGType::p);
  mg_type_level.insert(mg_type_level.end(),
                       mg_p_level.begin(),
                       mg_p_level.end());
  const unsigned int min_level = 0;
  const unsigned int max_level = mg_type_level.size();
  MGLevelObject<std::shared_ptr<const DoFHandler<dim>>> mg_dof_handlers(
    min_level, max_level);

  MGLevelObject<std::shared_ptr<const MatrixFreeOperatorScalar<dim, double>>>
    mg_M_mf(min_level, max_level);

  MGLevelObject<std::shared_ptr<const AffineConstraints<double>>>
    mg_constraints(min_level, max_level);
  for (auto mgt : mg_type_level)
    std::cout << static_cast<char>(mgt) << ' ';
  std::cout << std::endl;
  auto fe_   = fe_pmg.begin();
  auto tria_ = mg_triangulations.begin();
  for (unsigned int l = min_level, i = 0; l <= max_level; ++l, ++i)
    {
      std::cout << "On level: " << l << std::endl;

      auto dof_handler_ = std::make_shared<DoFHandler<dim>>(**tria_);
      auto constraints_ = std::make_shared<AffineConstraints<double>>();

      auto const &fe_dgp = *(*fe_)[0];
      dof_handler_->distribute_dofs(fe_dgp);
      if (i < mg_type_level.size() && mg_type_level[i] == MGType::p)
        ++fe_;
      if (i < mg_type_level.size() && mg_type_level[i] == MGType::h)
        ++tria_;
      std::cout << "Distributed dofs." << std::endl;
      IndexSet locally_relevant_dofs;
      DoFTools::extract_locally_relevant_dofs(*dof_handler_,
                                              locally_relevant_dofs);
      constraints_->reinit(locally_relevant_dofs);
      constraints_->close();
      auto M_mf_ = std::make_shared<MatrixFreeOperatorScalar<dim, double>>(
        mapping, *dof_handler_, *constraints_, quad, 1.0, 0.0);
      auto sparsity_pattern_ = std::make_shared<SparsityPatternType>(
        dof_handler_->locally_owned_dofs(),
        dof_handler_->locally_owned_dofs(),
        dof_handler_->get_communicator());
      DoFTools::make_sparsity_pattern(*dof_handler_,
                                      *sparsity_pattern_,
                                      *constraints_,
                                      false);
      sparsity_pattern_->compress();
      auto M_ = std::make_shared<SparseMatrixType>();
      M_->reinit(*sparsity_pattern_);
      M_mf_->compute_system_matrix(*M_);
      mg_M_mf[l]         = M_mf_;
      mg_dof_handlers[l] = dof_handler_;
      mg_constraints[l]  = constraints_;
      {
        std::cout << "Simple compute system matrix...\n";
        VectorT<double> src(2);
        VectorT<double> dst(2);
        VectorT<double> dst2(2);
        M_mf_->initialize_dof_vector(src);
        M_mf_->initialize_dof_vector(dst);
        M_mf_->initialize_dof_vector(dst2);
        for (unsigned int i = 0; i < src.size(); ++i)
          {
            src    = 0.0;
            src(i) = 1.0;
            M_mf_->vmult(dst, src);
            M_->vmult(dst2, src);
            dst.add(-1.0, dst2);
            if (dst.l2_norm() <= 1.e-16)
              std::cout << " 0";
          }
        std::cout << std::endl;
      }
    }
  Assert(fe_ == fe_pmg.end() - 1, ExcInternalError());
  Assert(tria_ == mg_triangulations.end() - 1, ExcInternalError());
  std::vector<BlockSlice> blk_indices(mg_dof_handlers.n_levels(), blk_slice);
  auto                    transfer_block = build_stmg_transfers<dim, double>(
    TimeStepType::DG,
    mg_dof_handlers,
    mg_constraints,
    [&](const unsigned int l, VectorT<double> &vec, const unsigned int) {
      mg_M_mf[l]->initialize_dof_vector(vec);
    },
    true,
    mg_type_level,
    blk_indices);
  {
    MGLevelObject<BlockVectorT<double>> dst(min_level, max_level);
    BlockVectorT<double>                src(1);
    M_mf.initialize_dof_vector(src.block(0));
    transfer_block->copy_to_mg(dof, dst, src);
  }
  for (unsigned int l = min_level; l < max_level; ++l)
    {
      BlockVectorT<double> dst(1);
      BlockVectorT<double> src(1);
      mg_M_mf[l]->initialize_dof_vector(src.block(0));
      mg_M_mf[l + 1]->initialize_dof_vector(dst.block(0));
      src = 1.0;
      transfer_block->prolongate_and_add(l + 1, dst, src);
      dst.print(std::cout, 2, false);
    }
  for (unsigned int l = min_level; l < max_level; ++l)
    {
      BlockVectorT<double> dst(1);
      BlockVectorT<double> src(1);
      mg_M_mf[l + 1]->initialize_dof_vector(src.block(0));
      mg_M_mf[l]->initialize_dof_vector(dst.block(0));
      src = 1.0;
      transfer_block->restrict_and_add(l + 1, dst, src);
      dst.print(std::cout, 2, false);
    }

  dealii::deallog << std::endl;
  pcout << std::endl;
}
