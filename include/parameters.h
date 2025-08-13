// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2024 by Nils Margenberg and Peter Munch

#pragma once
#include <deal.II/base/parameter_handler.h>

#include <fstream>

#include "fe_time.h"
namespace dealii
{
  struct PreconditionerGMGAdditionalData
  {
    double       smoothing_range               = 1;
    unsigned int smoothing_degree              = 5;
    unsigned int smoothing_eig_cg_n_iterations = 20;
    unsigned int smoothing_steps               = 1;

    double relaxation = 0.0;

    std::string coarse_grid_smoother_type = "Smoother";

    SupportedSmoothers smoother = SupportedSmoothers::Relaxation;

    unsigned int coarse_grid_maxiter = 10;
    double       coarse_grid_abstol  = 1e-20;
    double       coarse_grid_reltol  = 1e-4;

    bool restrict_is_transpose_prolongate = true;
    bool variable                         = true;
  };

  template <int dim>
  struct Parameters
  {
    bool                   do_output              = false;
    bool                   do_higher_order_output = false;
    bool                   print_timing           = false;
    bool                   space_time_mg          = true;
    bool                   time_before_space      = false;
    TimeStepType           type                   = TimeStepType::CGP;
    ProblemType            problem                = ProblemType::wave;
    PatchType              patch_type             = PatchType::element;
    NonlinearTreatment     nonlinear_treatment = NonlinearTreatment::Implicit;
    NonlinearExtrapolation nonlinear_extrapolation =
      NonlinearExtrapolation::Auto;
    bool is_nonlinear = false;

    CoarseningType coarsening_type        = CoarseningType::space_or_time;
    bool           space_time_level_first = true;
    bool           use_pmg                = false;
    MGTransferGlobalCoarseningTools::PolynomialCoarseningSequenceType
      poly_coarsening = MGTransferGlobalCoarseningTools::
        PolynomialCoarseningSequenceType::bisect;
    unsigned int n_timesteps_at_once     = 1;
    int          n_timesteps_at_once_min = -1;
    unsigned int fe_degree               = 1;
    int          fe_degree_min           = -1;
    int          fe_degree_min_space     = -1;
    unsigned int n_deg_cycles            = 1;
    unsigned int n_ref_cycles            = 1;
    double       frequency               = 1.0;
    double       rel_tol                 = 1.0e-12;
    int          refinement              = 2;
    int          time_refine_offset      = 1;
    bool         space_time_conv_test    = true;
    bool         extrapolate             = true;
    bool         colorize_boundary       = false;
    bool         nitsche_boundary        = false;
    std::string  functional_file         = "functionals.txt";
    std::string  grid_descriptor         = "hyperRectangle";
    std::string  additional_file         = "";
    Point<dim>   hyperrect_lower_left =
      dim == 2 ? Point<dim>(0., 0.) : Point<dim>(0., 0., 0.);
    Point<dim> hyperrect_upper_right =
      dim == 2 ? Point<dim>(1., 1.) : Point<dim>(1., 1., 1.);
    std::vector<unsigned int> subdivisions  = std::vector<unsigned int>(dim, 1);
    double                    distort_grid  = 0.0;
    double                    distort_coeff = 0.0;
    Point<dim> source = .5 * hyperrect_lower_left + .5 * hyperrect_upper_right;
    double     end_time   = 1.0;
    double     delta_time = 0.0;


    PreconditionerGMGAdditionalData mg_data;
    void
    parse(const std::string file_name)
    {
      std::string type_, problem_,
        p_mg_ = "bisect", c_type_ = "space_or_time", smoother = "relaxation",
        nonlinear_treatment_ = "none", nonlinear_extra_ = "auto",
        patch_type_ = "element";
      dealii::ParameterHandler prm;
      prm.add_parameter("doOutput", do_output);
      prm.add_parameter("doHigherOrderOutput", do_higher_order_output);
      prm.add_parameter("printTiming", print_timing);
      prm.add_parameter("spaceTimeMg", space_time_mg);
      prm.add_parameter("mgTimeBeforeSpace", time_before_space);
      prm.add_parameter("timeType", type_);
      prm.add_parameter("problemType", problem_);
      prm.add_parameter("patchType", patch_type_);
      prm.add_parameter("nonlinearTreatment", nonlinear_treatment_);
      prm.add_parameter("nonlinearExtrapolation", nonlinear_extra_);
      prm.add_parameter("pMgType", p_mg_);
      prm.add_parameter("coarseningType", c_type_);
      prm.add_parameter("spaceTimeLevelFirst", space_time_level_first);
      prm.add_parameter("usePMg", use_pmg);
      prm.add_parameter("nTimestepsAtOnce", n_timesteps_at_once);
      prm.add_parameter("nTimestepsAtOnceMin", n_timesteps_at_once_min);
      prm.add_parameter("feDegree", fe_degree);
      prm.add_parameter("feDegreeMin", fe_degree_min);
      prm.add_parameter("feDegreeMinSpace", fe_degree_min_space);
      prm.add_parameter("nDegCycles", n_deg_cycles);
      prm.add_parameter("nRefCycles", n_ref_cycles);
      prm.add_parameter("frequency", frequency);
      prm.add_parameter("relativeTolerance", rel_tol);
      prm.add_parameter("refinement", refinement);
      prm.add_parameter("timeRefineOffset", time_refine_offset);
      prm.add_parameter("spaceTimeConvergenceTest", space_time_conv_test);
      prm.add_parameter("extrapolate", extrapolate);
      prm.add_parameter("colorizeBoundary", colorize_boundary);
      prm.add_parameter("nitscheBoundary", nitsche_boundary);
      prm.add_parameter("functionalFile", functional_file);
      prm.add_parameter("gridDescriptor", grid_descriptor);
      prm.add_parameter("additionalFile", additional_file);
      prm.add_parameter("hyperRectLowerLeft", hyperrect_lower_left);
      prm.add_parameter("hyperRectUpperRight", hyperrect_upper_right);
      prm.add_parameter("subdivisions", subdivisions);
      prm.add_parameter("distortGrid", distort_grid);
      prm.add_parameter("distortCoeff", distort_coeff);
      prm.add_parameter("sourcePoint", source);
      prm.add_parameter("endTime", end_time);
      prm.add_parameter("deltaTime", delta_time);

      prm.add_parameter("smoother", smoother);
      prm.add_parameter("smoothingDegree", mg_data.smoothing_degree);
      prm.add_parameter("smoothingSteps", mg_data.smoothing_steps);
      prm.add_parameter("smoothingRange", mg_data.smoothing_range);
      prm.add_parameter("relaxation", mg_data.relaxation);
      prm.add_parameter("coarseGridSmootherType",
                        mg_data.coarse_grid_smoother_type);
      prm.add_parameter("coarseGridMaxiter", mg_data.coarse_grid_maxiter);
      prm.add_parameter("coarseGridAbstol", mg_data.coarse_grid_abstol);
      prm.add_parameter("coarseGridReltol", mg_data.coarse_grid_reltol);
      prm.add_parameter("restrictIsTransposeProlongate",
                        mg_data.restrict_is_transpose_prolongate);
      prm.add_parameter("variable", mg_data.variable);
      AssertIsFinite(frequency);
      AssertIsFinite(distort_grid);
      AssertIsFinite(distort_coeff);
      AssertIsFinite(end_time);

      std::ifstream file;
      file.open(file_name);
      prm.parse_input_from_json(file, true);
      type                = str_to_time_type.at(type_);
      problem             = str_to_problem_type.at(problem_);
      patch_type          = str_to_patch_type.at(patch_type_);
      nonlinear_treatment = str_to_nonlinear_treatment.at(nonlinear_treatment_);
      nonlinear_extrapolation =
        str_to_nonlinear_extrapolation.at(nonlinear_extra_);
      is_nonlinear     = nonlinear_treatment != NonlinearTreatment::None;
      poly_coarsening  = str_to_polynomial_coarsening_type.at(p_mg_);
      coarsening_type  = str_to_coarsening_type.at(c_type_);
      mg_data.smoother = str_to_smoother_type.at(smoother);
      if (n_timesteps_at_once_min == -1)
        n_timesteps_at_once_min = n_timesteps_at_once / 2;

      n_timesteps_at_once_min =
        std::clamp(n_timesteps_at_once_min,
                   1,
                   static_cast<int>(n_timesteps_at_once));
      const int lowest_degree = type == TimeStepType::DG ? 0 : 1;
      if (fe_degree_min == -1)
        fe_degree_min = fe_degree - 1;
      fe_degree_min =
        std::clamp(fe_degree_min, lowest_degree, static_cast<int>(fe_degree));
      if (fe_degree_min_space == -1)
        fe_degree_min_space = fe_degree_min;
    }
  };
} // namespace dealii
