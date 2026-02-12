#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include "exact_solution.h"
#include "stmg.h"
#include "stokes.h"
#include "types.h"
namespace dealii
{
  template <typename Number, int dim>
  void
  setup_constraints_up(dealii::AffineConstraints<Number> &constraints_u,
                       dealii::AffineConstraints<Number> &constraints_p,
                       DoFHandler<dim> const             &dof_handler_u,
                       DoFHandler<dim> const             &dof_handler_p,
                       Parameters<dim> const             &parameters,
                       stokes::Parameters const          &stokes_parameters)
  {
    DoFTools::make_hanging_node_constraints(dof_handler_u, constraints_u);
    if (parameters.space_time_conv_test)
      {
        if (!parameters.nitsche_boundary)
          DoFTools::make_zero_boundary_constraints(dof_handler_u,
                                                   constraints_u);
      }
    else
      {
        if (stokes_parameters.dfg_benchmark == 0)
          {
            DoFTools::make_zero_boundary_constraints(dof_handler_u,
                                                     3,
                                                     constraints_u);
            DoFTools::make_zero_boundary_constraints(dof_handler_u,
                                                     2,
                                                     constraints_u);
            DoFTools::make_zero_boundary_constraints(dof_handler_u,
                                                     0,
                                                     constraints_u);
            if (dim == 3)
              {
                DoFTools::make_zero_boundary_constraints(dof_handler_u,
                                                         4,
                                                         constraints_u);
                DoFTools::make_zero_boundary_constraints(dof_handler_u,
                                                         5,
                                                         constraints_u);
              }
          }
      }
    DoFTools::make_hanging_node_constraints(dof_handler_p, constraints_p);
  }

  template <typename Number, int dim>
  void
  setup_constraints(dealii::AffineConstraints<Number> &constraints,
                    DoFHandler<dim> const             &dof_handler,
                    Parameters<dim> const             &parameters)
  {
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    if (!parameters.nitsche_boundary)
      DoFTools::make_zero_boundary_constraints(dof_handler, constraints);
  }


  template <typename Number, int dim>
  std::pair<std::vector<std::unique_ptr<Function<dim, Number>>>,
            std::map<types::boundary_id, dealii::Function<dim> *>>
  get_dirichlet_function(Parameters<dim> const    &parameters,
                         stokes::Parameters const &stokes_parameters = {})
  {
    std::pair<std::vector<std::unique_ptr<Function<dim, Number>>>,
              std::map<types::boundary_id, dealii::Function<dim> *>>
      ret;

    if (parameters.problem == ProblemType::stokes)
      {
        ret.first.reserve(1);
        if (!parameters.space_time_conv_test)
          {
            if (stokes_parameters.dfg_benchmark != 0)
              {
                ret.first.emplace_back(
                  std::make_unique<stokes::InflowDfg<dim, Number>>(
                    stokes_parameters));
                ret.second.emplace(0, ret.first.back().get());

                ret.first.emplace_back(
                  std::make_unique<Functions::ZeroFunction<dim, Number>>(dim));
                ret.second.emplace(3, ret.first.back().get());

                ret.first.emplace_back(
                  std::make_unique<Functions::ZeroFunction<dim, Number>>(dim));
                ret.second.emplace(2, ret.first.back().get());
                return ret;
              }
            else
              {
                if (stokes_parameters.lid_driven_bc == 1)
                  ret.first.emplace_back(
                    std::make_unique<stokes::LidDriven<dim, Number>>(
                      stokes_parameters));
                else if (stokes_parameters.lid_driven_bc == 2)
                  ret.first.emplace_back(
                    std::make_unique<
                      stokes::LidDrivenDiscontinuous<dim, Number>>(
                      stokes_parameters));
                else if (stokes_parameters.lid_driven_bc == 3)
                  ret.first.emplace_back(
                    std::make_unique<stokes::LidDrivenSawTooth<dim, Number>>(
                      stokes_parameters));

                ret.second.emplace(1, ret.first.back().get());
                return ret;
              }
          }
        else
          {
            ret.first.emplace_back(
              std::make_unique<Functions::ZeroFunction<dim, Number>>(dim));
            ret.second.emplace(0, ret.first.back().get());
            return ret;
          }
      }
    return ret;
  }


  template <int dim>
  std::set<types::boundary_id>
  get_outflow_boundaries(Parameters<dim> const    &parameters,
                         stokes::Parameters const &stokes_parameters = {})
  {
    std::set<types::boundary_id> boundary_ids{};
    if (parameters.problem == ProblemType::stokes)
      {
        if (!parameters.space_time_conv_test)
          if (stokes_parameters.dfg_benchmark != 0)
            boundary_ids.emplace(1);
      }

    return boundary_ids;
  }

  template <typename Number, int dim>
  std::pair<std::vector<std::unique_ptr<Function<dim, Number>>>,
            std::map<types::boundary_id, dealii::Function<dim> *>>
  get_dirichlet_function(Parameters<dim> const &parameters)
  {
    stokes::Parameters stokes_parameters;
    return get_dirichlet_function<Number>(parameters, stokes_parameters);
  }

  template <int dim>
  void
  setup_triangulation(Triangulation<dim>    &tria,
                      Parameters<dim> const &parameters)
  {
    // for wave and heat and stokes convergence
    if (parameters.grid_descriptor == "hyperRectangle")
      GridGenerator::subdivided_hyper_rectangle(
        tria,
        parameters.subdivisions,
        parameters.hyperrect_lower_left,
        parameters.hyperrect_upper_right,
        parameters.colorize_boundary);
    else if (parameters.grid_descriptor == "unstructuredHyperRectangle")
      {
        std::vector<Point<dim>>   vertices1, vertices2;
        Point<dim>                p_ll, p_ur;
        std::vector<unsigned int> repetitions;
        {
          Triangulation<dim> tmp2;
          if (dim == 2)
            {
              vertices1   = {{0.25, 1.0}, {0.25, 0.0}, {0.75, 1.0}};
              vertices2   = {{0.25, 0.0}, {0.75, 0.0}, {0.75, 1.0}};
              repetitions = {{1, 2}};
              p_ll        = {0.0, 0.0};
              p_ur        = {1.0, 1.0};
            }
          else
            {
              DEAL_II_NOT_IMPLEMENTED();
            }
          GridGenerator::simplex(tria, vertices1);
          GridGenerator::simplex(tmp2, vertices2);
          GridGenerator::merge_triangulations(tria, tmp2, tria, 1.e-3);
        }
        {
          Triangulation<dim> tmp2;
          GridGenerator::subdivided_hyper_rectangle(tmp2,
                                                    repetitions,
                                                    p_ll,
                                                    vertices1[0]);
          GridGenerator::merge_triangulations(tria, tmp2, tria, 1.e-3);
        }
        {
          Triangulation<dim> tmp2;
          GridGenerator::subdivided_hyper_rectangle(tmp2,
                                                    repetitions,
                                                    vertices2[1],
                                                    p_ur);
          GridGenerator::merge_triangulations(tria, tmp2, tria, 1.e-3);
        }
        return;
      }
    else if (parameters.grid_descriptor == "dfgBenchmark")
      {
        if constexpr (dim == 2)
          GridGenerator::channel_with_cylinder(tria, 0.025, 1, 1.0, true);
        else
          {
            Triangulation<3> tmp;
            GridGenerator::channel_with_cylinder(tmp, 0.025, 1, 1.0, true);
            tmp.reset_all_manifolds();
            Tensor<1, 3> shift;
            shift[0] = 0.3;
            GridTools::shift(shift, tmp);
            Triangulation<3> front_tria;
            GridGenerator::subdivided_hyper_rectangle(front_tria,
                                                      {3u, 4u, 4u},
                                                      Point<3>(0.0, 0.0, 0.0),
                                                      Point<3>(0.3,
                                                               0.41,
                                                               0.41));
            GridGenerator::merge_triangulations(
              front_tria, tmp, tria, 1.e-3, true, false);
            const Point<3>     axial_point(0.5, 0.2, 0.0);
            const Tensor<1, 3> direction{{0.0, 0.0, 1.0}};
            tria.set_manifold(cylindrical_manifold_id, FlatManifold<3>());
            tria.set_manifold(tfi_manifold_id, FlatManifold<3>());
            const CylindricalManifold<3>        cylindrical_manifold(direction,
                                                              axial_point);
            TransfiniteInterpolationManifold<3> inner_manifold;
            inner_manifold.initialize(tria);
            tria.set_manifold(cylindrical_manifold_id, cylindrical_manifold);
            tria.set_manifold(tfi_manifold_id, inner_manifold);

            for (const auto &face : tria.active_face_iterators())
              if (face->at_boundary())
                {
                  auto const &center = face->center();
                  if (std::abs(center[0] - 0.0) < 1e-8)
                    face->set_boundary_id(0);
                  else if (std::abs(center[0] - 2.5) < 1e-8)
                    face->set_boundary_id(1);
                  else if (face->manifold_id() == cylindrical_manifold_id)
                    face->set_boundary_id(2);
                  else
                    face->set_boundary_id(3);
                }
          }
      }
    else if (parameters.grid_descriptor == "dfgBenchmarkSquare")
      {
        if constexpr (dim == 2)
          {
            Triangulation<2> tmp;
            GridGenerator::subdivided_hyper_rectangle(
              tmp,
              {std::vector<double>{
                 0.15, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.35, 0.35},
               std::vector<double>{0.15, 0.1, 0.16}},
              dealii::Point<2>(0.0, 0.0),
              dealii::Point<2>(2.2, 0.41));
            std::set<typename Triangulation<dim>::active_cell_iterator>
              rm_cells;
            for (const auto &cell : tmp.active_cell_iterators())
              if (auto p = cell->center() - dealii::Point<2>(0.2, 0.2);
                  p.norm() < 0.05)
                rm_cells.insert(cell);

            GridGenerator::create_triangulation_with_removed_cells(tmp,
                                                                   rm_cells,
                                                                   tria);

            for (const auto &face : tria.active_face_iterators())
              if (face->at_boundary())
                {
                  auto const &center = face->center();
                  if (std::abs(center[0] - 0.0) < 1e-8)
                    face->set_boundary_id(0);
                  else if (std::abs(center[0] - 2.2) < 1e-8)
                    face->set_boundary_id(1);
                  else if (auto p = face->center() - dealii::Point<2>(0.2, 0.2);
                           p.norm() <= 0.1)
                    face->set_boundary_id(2);
                  else
                    face->set_boundary_id(3);
                }
          }
        else
          {
            Triangulation<3> tmp;
            dealii::GridGenerator::subdivided_hyper_rectangle(
              tmp,
              {std::vector<double>{
                 0.3, 0.15, 0.1, 0.15, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.3},
               std::vector<double>{0.15, 0.1, 0.16},
               std::vector<double>{0.13666666666666666,
                                   0.13666666666666666,
                                   0.13666666666666666}},
              dealii::Point<3>(0.0, 0.0, 0.0),
              dealii::Point<3>(2.5, 0.41, 0.41));
            std::set<typename Triangulation<dim>::active_cell_iterator>
              rm_cells;
            for (const auto &cell : tmp.active_cell_iterators())
              {
                auto p     = cell->center() - dealii::Point<3>(0.5, 0.2, 0.2);
                p[dim - 1] = 0;
                if (p.norm() <= 0.05)
                  rm_cells.insert(cell);
              }
            GridGenerator::create_triangulation_with_removed_cells(tmp,
                                                                   rm_cells,
                                                                   tria);

            for (const auto &face : tria.active_face_iterators())
              if (face->at_boundary())
                {
                  auto const &center = face->center();
                  auto        p      = center - dealii::Point<3>(0.5, 0.2, 0.2);
                  p[dim - 1]         = 0;
                  if (std::abs(center[0] - 0.0) < 1e-8)
                    face->set_boundary_id(0);
                  else if (std::abs(center[0] - 2.5) < 1e-8)
                    face->set_boundary_id(1);
                  else if (p.norm() <= 0.1)
                    face->set_boundary_id(2);
                  else
                    face->set_boundary_id(3);
                }
          }
      }
    else if (parameters.grid_descriptor == "bendingPipe")
      {
      }
  }
} // namespace dealii
