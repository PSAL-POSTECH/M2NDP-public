#ifndef CONCURRENT_KV_RUNNER_H
#define CONCURRENT_KV_RUNNER_H
#include <map>
#include <queue>

#include "cxl_link.h"
#include "m2ndp_config.h"
#include "m2ndp.h"

namespace NDPSim {
struct NdpCommand {
  std::string ndp_kernel_path;
  bool is_barrier;
  bool valid;
  bool finished;
  int waiting_memory_request;
  std::queue<mem_fetch*> memory_reqs;
};

class InjectionQueue {
  public:
    InjectionQueue(double injection_rate, int max_mps, M2NDPConfig* m2ndp_config, bool serial);
    void initialize_commands(std::queue<NdpCommand> *commands);
    void cycle();
    mem_fetch* top();
    void pop();
    bool empty();
    void finish(mem_fetch* mf);
    void print_avg_latency();
    struct InjectionQueueEntry {
      mem_fetch* mf;
      uint64_t inject_time;
    };
    bool finished();
  private: 
    bool serial;
    double m_injection_rate;
    int m_max_mps;
    M2NDPConfig* m_m2ndp_config;
    std::queue<InjectionQueueEntry> m_queue;
    std::unordered_map<mem_fetch*, uint64_t> m_injection_times;
    std::queue<NdpCommand> *m_commands;
    std::vector<uint64_t> m_latencies;
    float m_counter = 0;    
    uint64_t m_total_latency = 0;
    uint64_t m_total_requests = 0;
};

class KVRunner {
 public:
  KVRunner(int argc, char* argv[]);
  void run();

 private:
  void register_kernelslist();
  void parse_ndp_trace();
  bool check_ndp_kenrel_active();
  bool check_all_memory_reqs_empty();
  bool check_all_simulaiton_finished();
  bool check_single_simulation_finished();
  bool check_can_launch_ndp_kernel();
  void launch_ndp_kernel(NdpCommand command);
  void process_memory_access();
  void fill_memory_access(NdpCommand& command, std::string line);
  int m_num_m2ndps;
  double m_injection_rate;
  int  m_max_mps;
  std::string m_config_file_path;
  std::string m_trace_dir_path;
  std::string m_output_filename;
  FILE* m_output_file;
  FILE* m_energy_file;
  M2NDPConfig* m_m2ndp_config;
  std::vector<M2NDP*> m_m2ndps;
  CxlLink* m_cxl_link;

  MemoryMap* m_memory_map;

  int remaing_memory_reqs;
  std::queue<mem_fetch*> m_memory_reqs;
  std::queue<NdpCommand> m_ndp_commands;
  InjectionQueue m_injection_queue;
  std::queue<std::pair<mem_fetch*, uint64_t>> m_bi_reqs;
};
}
#endif