#include "stokes.h"

#include <fstream>
namespace dealii::stokes
{
  void
  Parameters::parse(const std::string file_name)
  {
    ParameterHandler prm;
    prm.add_parameter("computeDragLift", compute_drag_lift);
    prm.add_parameter("rho", rho);
    prm.add_parameter("characteristicDiam", characteristic_diameter);
    prm.add_parameter("uMean", u_mean);
    prm.add_parameter("viscosity", viscosity);
    prm.add_parameter("pOrder", p_order);
    prm.add_parameter("pRegularization", p_regularization);
    prm.add_parameter("symmetricTensor", symmetric_tensor);
    prm.add_parameter("delta0", delta0);
    prm.add_parameter("delta1", delta1);
    prm.add_parameter("penalty1", penalty1);
    prm.add_parameter("penalty2", penalty2);
    prm.add_parameter("outflowPenalty", outflow_penalty);
    prm.add_parameter("meanPressure", mean_pressure);
    prm.add_parameter("dGPressure", dg_pressure);
    prm.add_parameter("dfgBenchmark", dfg_benchmark);
    prm.add_parameter("lidDrivenBc", lid_driven_bc);
    std::ifstream file;
    file.open(file_name);
    prm.parse_input_from_json(file, true);
  }
} // namespace dealii::stokes
