#ifdef TIMING_SIMULATION
#ifndef PACKET_FILTER_H
#define PACKET_FILTER_H

#include <queue>
#include <vector>

#include "delayqueue.h"
#include "mem_fetch.h"
#include "m2ndp_config.h"
#include "common_defs.h"

namespace NDPSim {
class UThreadGenerator {
 public:
  UThreadGenerator(M2NDPConfig* config, int ndp_id,
                   fifo_pipeline<RequestInfo>* matched_requests);
  void launch(KernelLaunchInfo info);

  void register_kernel(NdpKernel* kernel);
  void unregister_kernel(int kernel_id);

  void finish_launch(int launch_id);
  bool is_launch_active(int launch_id);
  bool is_active();
  void print_status();
  void print_all(FILE* fp);
  void cycle();
  void increase_filter_count(int index);
  void increase_count(int launch_id);
  uint32_t get_allocated_spad_size();
  bool generate_uthreads(int threads);
 private:
  M2NDPConfig* m_config;
  int m_ndp_id;
  unsigned m_total_functions;
  bool m_request_processed;
  bool m_launched = false;
  
  std::set<int> m_registered_functions;
  std::map<int, int> m_num_kernel_bodies;
  std::set<int> m_active_launch_ids;
  std::map<int,KernelLaunchInfo> m_launch_infos;
  fifo_pipeline<RequestInfo>* m_uthread_request_queue;
  std::deque<int> m_launch_queue;
  std::map<int, std::deque<RequestInfo*>> m_generated_requests;
  std::map<int, int> m_total_requests;
  std::map<int, int> m_count_requests;

  bool check_addr_match(uint64_t addr);
  int get_next_launch_id();
  bool check_can_issue(RequestInfo* info);
  void check_kernel_launch();
};
}  // namespace NDPSim
#endif
#endif