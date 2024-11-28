#include <chrono>
#include <ctime>
#include <spdlog/spdlog.h>
#include <spdlog/cfg/env.h>

#ifdef KVRUN
#include "kv_runner.h"
#elif defined(SCALABILITY)
#include "scalability_runner.h"
#else
#include "simulation_runner.h"
#endif



int main(int argc, char* argv[]) {
  spdlog::cfg::load_env_levels();
  auto start = std::chrono::system_clock::now();
  #ifdef KVRUN
  NDPSim::KVRunner runner = NDPSim::KVRunner(argc, argv);
  #elif defined(SCALABILITY)
  NDPSim::ScalabilityRunner runner = NDPSim::ScalabilityRunner(argc, argv);
  #else 
  NDPSim::SimulationRunner runner = NDPSim::SimulationRunner(argc, argv);
  #endif
  runner.run();
  auto end = std::chrono::system_clock::now();
  spdlog::info(
      "Simulation time: {} seconds",
      std::chrono::duration_cast<std::chrono::seconds>(end - start)
          .count());
  return 0;
}