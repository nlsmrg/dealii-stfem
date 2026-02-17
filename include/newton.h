#pragma once

#include <tuple>

#include "operators.h"
#include "types.h"

namespace dealii
{

  struct NewtonData
  {
    double       rel_tol              = 1e-8;  // Relative tolerance
    double       abs_tol              = 1e-10; // Absolute tolerance
    double       lambda0              = 1.0;   // damping factor
    double       rho                  = 1.e-3; // convergence criterion on rate
    unsigned int max_iter             = 20;    // Maximum Newton iterations
    unsigned int max_line_search_iter = 10;    // Max backtracks
    double       alpha                = 1e-4;  // Armijo parameter
    double       min_step             = 1e-12; // Minimum step size
    double       reduction_factor     = 0.7;   //
    bool update_once_converged = false; // Update precon after convergence
    bool do_update             = false; // Update precon
    unsigned int       update_every_newton_iter = 1; // Update frequency
    LineSearchStrategy line_search_strategy =
      LineSearchStrategy::linear_decrease;
    bool         nonmonotone      = true; // use max over window
    unsigned int nm_window        = 5;    // window size M (0 => monotone)
    double       eta_max          = 0.8;  // max linear residual ratio
    double       eta_min          = 1.0e-3;  // min linear residual ratio
    double       eta_0            = 0.4;  // initial
    double       eta_c            = 0.5;  // EW constant
    double       eta_theta        = 1.5;  // EW exponent
    unsigned int lin_critical_rel = 3;
    unsigned int lin_critical_abs = 30;

    double   keep_going_rate     = 0.2; // good contraction, keep going
    double   stall_rate          = 0.8; // stop if one tol met
    unsigned max_extra_after_tol = 2;   // cap extra steps once one tol is met

    void
    parse(const std::string file_name);
  };

  template <int dim, typename Number, typename VectorType, typename PDEOperator>
  class Newton
  {
  public:
    Newton(NewtonData const  &solver_data_,
           PDEOperator const &pde_operator_,
           std::function<void(BlockVectorT<Number> const &)> const
                                       &update_linearization_,
           std::function<void()> const &update_preconditioner_,
           std::function<unsigned int(BlockVectorT<Number> &,
                                      BlockVectorT<Number> const &)> const
                                             &linear_solve_,
           std::function<void(double)> const &set_tolerance_)
      : solver_data(solver_data_)
      , update_linearization(update_linearization_)
      , update_preconditioner(update_preconditioner_)
      , set_tolerance(set_tolerance_)
      , linear_solve(linear_solve_)
      , pde_operator(pde_operator_)
    {
      pde_operator.initialize_dof_vector(update);
      pde_operator.initialize_dof_vector(residual);
    }

    std::pair<unsigned int /* total nl  iter */,
              unsigned int /* total lin iter */>
    solve(VectorType &solution, VectorType const &rhs) const
    {
      deallog << "[Newton] Starting nonlinear solve | max_iter: "
              << solver_data.max_iter << ", abs_tol: " << solver_data.abs_tol
              << ", rel_tol: " << solver_data.rel_tol << "\n";

      bool has_been_updated   = false;
      bool should_update      = false;
      newton_iter             = 0;
      total_linear_iterations = 0;
      unsigned int lin_iters  = 0;
      first_one_tol_iter      = -1;
      pde_operator.residual(residual, solution, rhs);
      double       norm_r  = residual.l2_norm();
      double const norm_r0 = norm_r;
      deallog << "[Newton " << newton_iter << "] Residual norm: " << norm_r
              << "\n";
      bool converged = check_convergence(norm_r, norm_r0);
      if (converged)
        return {newton_iter, total_linear_iterations};

      while (newton_iter < solver_data.max_iter)
        {
          double prev_norm_r = norm_r;
          update             = 0.0;

          pde_operator.set_data(solution);
          update_linearization(solution);

          should_update = (last_rate > solver_data.rho);
          bool const periodic =
            (solver_data.update_every_newton_iter > 0) &&
            ((newton_iter + 1) % solver_data.update_every_newton_iter == 0);
          double const mean_prev =
            (newton_iter > 0) ?
              (total_linear_iterations / static_cast<double>(newton_iter)) :
              static_cast<double>(lin_iters);
          auto const outlier_on_linear_iters =
            (newton_iter > 0 &&
             lin_iters > solver_data.lin_critical_rel * mean_prev) ||
            lin_iters >= solver_data.lin_critical_abs;

          bool const update_precon = should_update ||
                                     (newton_iter > 0 && periodic) ||
                                     outlier_on_linear_iters;


          if (update_precon && !has_been_updated)
            {
              deallog << "[Newton " << newton_iter
                      << "] Updating preconditioner. " << should_update << " | "
                      << (newton_iter > 0 && periodic) << " | "
                      << outlier_on_linear_iters << "\n";
              update_preconditioner();
              has_been_updated = true;
            }

          if (newton_iter == 0)
            set_tolerance(solver_data.eta_0);
          else
            {
              double eta =
                std::clamp(solver_data.eta_c *
                             std::pow(last_rate, solver_data.eta_theta),
                           solver_data.eta_min,
                           solver_data.eta_max);
              this->set_tolerance(eta);
            }

          deallog << "[Newton " << newton_iter << "] Solving linear system... ";
          lin_iters = linear_solve(update, residual);
          deallog << lin_iters << " iterations.\n";
          total_linear_iterations += lin_iters;

          double       norm_r_before = norm_r;
          unsigned int ls_iter       = 0;
          double       norm_r_trial  = 0.0;

          double phi0 = 0.5 * norm_r_before * norm_r_before;

          // non-monotone window (track recent phi)
          std::deque<double> phi_hist;
          phi_hist.push_back(phi0);

          double phi_trial  = 0.0;
          double alpha_step = solver_data.lambda0;

          // directional derivative at 0 (exact Newton: g0 = -||R||^2)
          double g0     = -norm_r_before * norm_r_before;
          double alpha1 = 0.0, phi1 = phi0; // last rejected
          double alpha2 = 0.0, phi2 = phi0; // prev rejected
          bool   first_fail = true;

          for (; ls_iter < std::max(1u, solver_data.max_line_search_iter);
               ++ls_iter)
            {
              solution.add(alpha_step, update);
              pde_operator.residual(residual, solution, rhs);
              norm_r_trial = residual.l2_norm();
              phi_trial    = 0.5 * norm_r_trial * norm_r_trial;

              // Armijo (monotone or nonmonotone)
              double phi_max = phi0;
              if (solver_data.nonmonotone && solver_data.nm_window > 0)
                {
                  if (phi_hist.size() > solver_data.nm_window)
                    phi_hist.pop_front();
                  phi_max = *std::max_element(phi_hist.begin(), phi_hist.end());
                }


              if (double rhs_p = phi_max + solver_data.alpha * alpha_step * g0;
                  phi_trial <= rhs_p)
                {
                  // accept
                  deallog << "[Newton " << newton_iter
                          << "] Armijo ok at α=" << alpha_step
                          << "  φ=" << phi_trial << "\n";
                  phi_hist.push_back(phi_trial);
                  break;
                }

              // reject
              solution.add(-alpha_step, update);
              // fallback if interpolation fails
              double alpha_new = solver_data.reduction_factor * alpha_step;

              if (first_fail)
                {
                  // quadratic model using phi(0), phi(alpha), g0
                  double denom = 2.0 * (phi_trial - phi0 - g0 * alpha_step);
                  if (denom > 0.0)
                    alpha_new =
                      std::clamp(-g0 * alpha_step * alpha_step / denom,
                                 0.1 * alpha_step,
                                 0.5 * alpha_step);
                  first_fail = false;
                  alpha1     = alpha_step;
                  phi1       = phi_trial;
                }
              else
                {
                  // cubic ip using (alpha_1,phi_1), (alpha_2,phi_2), phi_0, g0
                  double a = alpha1, b = alpha2;
                  double fa = phi1, fb = phi2;
                  // coefficients for cubic minimizer (Powell)
                  double d1    = fa - phi0 - g0 * a;
                  double d2    = fb - phi0 - g0 * b;
                  double denom = (a - b) * (a - b) * b - (b - a) * (b - a) * a;
                  if (std::abs(denom) > 1e-32)
                    {
                      double A    = (d1 * b - d2 * a) / denom;
                      double B    = (a * a * d2 - b * b * d1) / denom;
                      double disc = B * B - 3.0 * A * g0;
                      if (A > 0.0 && disc > 0.0)
                        {
                          double alpha_cand =
                            (-B + std::sqrt(disc)) / (3.0 * A);
                          alpha_new = std::clamp(alpha_cand,
                                                 0.1 * alpha_step,
                                                 0.5 * alpha_step);
                        }
                    }
                  alpha2 = alpha1;
                  phi2   = phi1;
                  alpha1 = alpha_step;
                  phi1   = phi_trial;
                }

              alpha_step = alpha_new;

              deallog << "[Newton " << newton_iter
                      << "] Armijo fail, new α=" << alpha_step
                      << ", φ=" << phi_trial << "\n";

              if (alpha_step < solver_data.min_step)
                {
                  deallog << "[Newton " << newton_iter
                          << "] Line search hit α_min=" << alpha_step
                          << " - giving up.\n";
                  break;
                }
            }

          norm_r = norm_r_trial;
          update_rates(newton_iter + 1, norm_r, prev_norm_r, norm_r0);
          ++newton_iter;

          deallog << "[Newton " << newton_iter << "] norm_r=" << norm_r
                  << ", rel=" << norm_r / norm_r0 << "\n";

          converged = check_convergence(norm_r, norm_r0);
          if (converged)
            {
              deallog << "[Newton] Converged after " << newton_iter
                      << " iterations.\n";
              break;
            }
        }

      if (!converged && (newton_iter >= solver_data.max_iter))
        deallog << "[Newton] Did not converge after " << newton_iter
                << " iterations.\n";

      if (should_update && solver_data.update_once_converged)
        {
          deallog << "[Newton] Updating preconditioner on converge.\n";
          update_linearization(solution);
          update_preconditioner();
        }

      return {newton_iter, total_linear_iterations};
    }

  private:
    void
    update_rates(unsigned int iter,
                 double       residual_norm,
                 double       prev_norm,
                 double       first_norm) const
    {
      if (iter > 1)
        {
          double b  = residual_norm / first_norm;
          double p  = 1.0 / (iter - 1);
          rate      = std::pow(b, p);
          last_rate = residual_norm / prev_norm;
        }
      else
        {
          rate      = residual_norm / first_norm;
          last_rate = rate;
        }
      deallog << "[Newton rates] rate=" << rate << ", last_rate=" << last_rate
              << "\n";
    }

    bool
    check_convergence(double norm_r, double norm_r0) const
    {
      bool const abs_ok = norm_r <= solver_data.abs_tol;
      bool const rel_ok =
        (norm_r0 > 0.0) ? (norm_r <= solver_data.rel_tol * norm_r0) : abs_ok;

      return abs_ok || rel_ok;
    }

    NewtonData                                        solver_data;
    std::function<void(BlockVectorT<Number> const &)> update_linearization;
    std::function<void()>                             update_preconditioner;
    std::function<void(double)>                       set_tolerance;
    std::function<unsigned int(BlockVectorT<Number> &,
                               BlockVectorT<Number> const &)>
                         linear_solve;
    PDEOperator const   &pde_operator;
    mutable VectorType   update, residual;
    mutable double       rate = 1.0, last_rate = 0.0;
    mutable int          first_one_tol_iter = -1;
    mutable unsigned int newton_iter = 0, total_linear_iterations = 0;
  };

} // namespace dealii
