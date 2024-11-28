#include "command_line_parser.h"
#include "loop_parser.h"
#include "ndp_unit.h"
#include "m2ndp_parser.h"
#include "m2ndp_config.h"
#include <spdlog/spdlog.h>
#include <spdlog/cfg/env.h>
#include <fstream>

namespace po = boost::program_options;
using namespace NDPSim;
typedef enum {
  SINGLE,
  LOOP
} TraceOption;

void functional_simulation(int argc, char** argv) {
  spdlog::cfg::load_env_levels();
  CommandLineParser cmd_parser = CommandLineParser();
  cmd_parser.add_command_line_option<std::string>("ndp_trace",
                                                  "path for ndp trace file");
  cmd_parser.add_command_line_option<std::string>(
      "memory_map", "path for initialized memory map");
  cmd_parser.add_command_line_option<std::string>("target_map",
                                                  "path for target memory map");
  cmd_parser.add_command_line_option<std::string>("launch_file",
                                                  "path for launch trace file");
  cmd_parser.add_command_line_option<std::string>("config",
                                                  "path for config file");
  cmd_parser.add_command_line_option<int>("option", "trace option");
  try {
    cmd_parser.parse(argc, argv);
  } catch (const CommandLineParser::ParsingError& e) {
    spdlog::error(
        "Command line argument parrsing error captured. Error message: {}",
        e.what());
    throw(e);
  }
  std::string ndp_file_path;
  std::string memory_map_path;
  std::string target_map_path;
  std::string config_path;
  std::string launch_file_path;
  TraceOption trace_option = SINGLE;

  cmd_parser.set_if_defined("ndp_trace", &ndp_file_path);
  cmd_parser.set_if_defined("memory_map", &memory_map_path);
  cmd_parser.set_if_defined("target_map", &target_map_path);
  cmd_parser.set_if_defined("launch_file", &launch_file_path);
  cmd_parser.set_if_defined("config", &config_path);
  cmd_parser.set_if_defined("option", (int*)&trace_option);

  HashMemoryMap memory_map(memory_map_path);

  if (trace_option == SINGLE) {
    // Initialize NDP fuctions, Memory map, NDP units
    spdlog::info("Running FuncSim in Single Kernel Mode..");

    NdpKernel ndp_kernel;
    std::ifstream launch_file(launch_file_path);
    std::string line;
    
    std::vector<NdpUnit*> ndp_units;
    M2NDPConfig *config = new M2NDPConfig(config_path, 1);
    int num_ndp = config->get_num_ndp_units();
    M2NDPParser::parse_ndp_kernel(num_ndp, ndp_file_path, &ndp_kernel, config);
    if (config->get_use_synthetic_memory())
        memory_map.set_synthetic_memory(config->get_synthetic_base_address(),
                                        config->get_synthetic_memory_size());
    ndp_units.resize(num_ndp);
    for (int id = 0; id < num_ndp; id++) {
      ndp_units[id] = new NdpUnit(config, &memory_map, id);
    }

    // // Run Functional simulation on NDP unit
    int iter = 0;
    while(std::getline(launch_file, line)) {
      for (int id = 0; id < num_ndp; id++) {
        spdlog::info("Iter {} NDP {} Run..", iter, id);
        ndp_units[id]->Run(id, &ndp_kernel, line);
      }
      iter++;
    }
    
  } else if (trace_option == LOOP) {
    spdlog::info("Running FuncSim in Loop Mode..");

    std::vector<NdpUnit*> ndp_units;
    M2NDPConfig *config = new M2NDPConfig(config_path, 1);
    int num_ndp = config->get_num_ndp_units();

    LoopParser loop_parser(config, ndp_file_path);
    std::vector<NdpKernel> *ndp_kernels = loop_parser.GetNDPKernels();
    int num_kernel = loop_parser.GetNumKernel();
    int num_loop = loop_parser.GetNumLoop();

    ndp_units.resize(num_ndp);
    for (int id = 0; id < num_ndp; id++) {
      ndp_units[id] = new NdpUnit(config, &memory_map, id);
    }

    for (int loop = 0; loop < num_loop; loop++) {
      spdlog::info("Loop {} Run..", loop);
      for (int kernel_id=0; kernel_id < num_kernel; kernel_id++) {
        for (int id = 0; id < num_ndp; id++) {
          spdlog::info("NDP {} Run Kernel {}", id, kernel_id);
          ndp_units[id]->Run(id, &(*ndp_kernels)[kernel_id], launch_file_path);
        }  
      }  
    }
  } 
  HashMemoryMap target_map(target_map_path);
  // Check memory value
  if (target_map.Match(memory_map)) {
    spdlog::info("Functional simulation success");
    printf(" _____                             \n");
    printf("/  ___|                             _ _ \n");
    printf("\\ `--. _   _  ___ ___ ___  ___ ___ | | | \n");
    printf(" `--. \\ | | |/ __/ __/ _ \\/ __/ __|| | |\n");
    printf("/\\__/ / |_| | (_| (_|  __/\\__ \\__ \\|_|_|\n");
    printf("\\____/ \\__,_|\\___\\___\\___||___/___/(_|_)\n");
  } else {
    spdlog::info("Value Missmatch: Fucntional simulation fail");
  }
}

int main(int argc, char** argv) {
  functional_simulation(argc, argv);
  return 0;
}