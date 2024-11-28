#ifndef SCALABILITY_RUNNER_H
#define SCALABILITY_RUNNER_H
#include "cxl_link.h"
#include "m2ndp_config.h"
#include "m2ndp_switch.h"
namespace NDPSim {
struct NdpCommand {
  std::string ndp_kernel_path;
  bool is_barrier;
  bool valid;
  bool finished;
  int waiting_memory_request;
  std::vector<std::queue<mem_fetch*>> memory_reqs;
};

class ScalabilityRunner {
 public:
  ScalabilityRunner(int argc, char* argv[]);
  void run();
  void match_memorymap();

 private:
  void parse_ndp_trace(int buffer_id);
  bool check_ndp_kenrel_active();
  bool check_all_memory_reqs_empty();
  bool check_all_simulaiton_finished();
  bool check_single_simulation_finished(int buffer_id);
  bool check_can_launch_ndp_kernel(int buffer_id);
  void launch_ndp_kernel(NdpCommand command, int buffer_id);
  void process_memory_access();
  void fill_memory_access(NdpCommand& command, std::string line);
  int m_num_hosts;
  int m_num_m2ndps;
  std::string m_config_file_path;
  std::string m_trace_dir_path;
  std::string m_output_filename;
  FILE* m_output_file;
  FILE* m_energy_file;
  M2NDPConfig* m_m2ndp_config;
  std::vector<M2NDP*> m_m2m2ndps;
  CxlLink* m_cxl_link;
  bool m_use_synthetic_memory;

  // MemoryMap* m_memory_map;
  // MemoryMap* m_target_map;
  std::vector<MemoryMap*> m_memory_map;
  std::vector<MemoryMap*> m_target_map;

  int remaing_memory_reqs;
  std::vector<std::queue<mem_fetch*>> m_memory_reqs;
  std::vector<std::queue<NdpCommand>> m_ndp_commands;
};
}
#endif