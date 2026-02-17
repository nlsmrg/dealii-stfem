#include "newton.h"

namespace dealii
{
  void
  NewtonData::parse(const std::string file_name)
  {
    std::string      line_search_strategy_ = "fixedFactor";
    ParameterHandler prm;
    prm.add_parameter("relTol", rel_tol);
    prm.add_parameter("absTol", abs_tol);
    prm.add_parameter("lambda0", lambda0);
    prm.add_parameter("rho", rho);
    prm.add_parameter("maxIter", max_iter);
    prm.add_parameter("maxLineSearchIter", max_line_search_iter);
    prm.add_parameter("alpha", alpha);
    prm.add_parameter("minStep", min_step);
    prm.add_parameter("reductionFactor", reduction_factor);
    prm.add_parameter("doUpdate", do_update);
    prm.add_parameter("updateEveryNewtonIter", update_every_newton_iter);
    prm.add_parameter("updateOnceConverged", update_once_converged);
    prm.add_parameter("lineSearchStrategy", line_search_strategy_);
    prm.add_parameter("nonMonotone", nonmonotone);
    prm.add_parameter("nmWindow", nm_window);
    prm.add_parameter("etaMax", eta_max);
    prm.add_parameter("etaMin", eta_min);
    prm.add_parameter("eta0", eta_0);
    prm.add_parameter("etaC", eta_c);
    prm.add_parameter("etaTheta", eta_theta);
    prm.add_parameter("linCriticalRel", lin_critical_rel);
    prm.add_parameter("linCriticalAbs", lin_critical_abs);
    std::ifstream file;
    file.open(file_name);
    prm.parse_input_from_json(file, true);
    line_search_strategy = str_to_ls_strategy.at(line_search_strategy_);
  }
} // namespace dealii
