#ifdef TIMING_SIMULATION
#ifndef M2NDP_H
#define M2NDP_H
#include "interconnect_interface.hpp"
#include "cxl_link.h"
#include "ndp_ramulator.h"
#include "m2ndp_config.h"
#include "common.h"
namespace NDPSim {

class M2NDP {
 public:
  M2NDP(M2NDPConfig *config, MemoryMap *memory_map, int buffer_id);
  ~M2NDP();
  void cxl_link_cycle();
  void cycle();
  void set_cxl_link(CxlLink *link) { m_cxl_link = link; }
  void register_ndp_kernel(int host_id, string kernel_path);
  bool can_register_ndp_kernel(int host_id);
  bool is_active();
  bool is_launch_active(int launch_id);
  void display_stats(FILE *fp);
  void print_energy_stats(FILE *fp);
  void print_access_time(FILE *fp);

 private:
  int m_buffer_id;
  int m_contiguous_ch;
  unsigned long long tot_cycles;
  unsigned long long memory_cycles;
  M2NDPConfig *m_config;
  NdpRamulator *m_ramulator;
  CxlLink *m_cxl_link;
  InterconnectInterface *m_icnt;
  MemoryMap *m_memory_map;
  int m_num_m2ndp;
  int m_num_ndp_units;
  int m_num_banks;
  int m_inter_cxl_round_robin;
  std::vector<int> m_num_read_request;
  std::vector<int> m_num_read_response;
  std::vector<int> m_num_write_request;
  std::vector<int> m_num_write_response;
  std::vector<int> m_host_round_robin;
  std::vector<std::vector<NdpKernel*>> m_ndp_kernels;
  std::vector<KernelLaunchInfo> m_launch_infos;

  std::vector<NdpUnit*> m_ndp_units;
  std::vector<NdpStats> m_ndp_stats;
  NdpStats m_total_stats;

  std::vector<std::deque<mem_fetch*>> m_cxl_command_response;

  int *rr_index;
  std::vector<std::vector<int>> fair_index;
  std::vector<int> m_uthread_reqs;

  std::string get_kernel_name(int host_id, int kernel_id);
  void launch_ndp_kernel(int host_id, KernelLaunchInfo launch_info);
  void transfer_cxl_to_memory();
  void transfer_memory_to_cxl();
  void check_kernel_active();

  int get_ndp_id(int index);
  int get_output_port_id(mem_fetch* mf);
  void count_access_per_m2ndp(mem_fetch* mf);
};
}  // namespace NDPSim
#endif
#endif