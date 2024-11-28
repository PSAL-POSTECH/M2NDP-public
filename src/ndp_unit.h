#ifndef FUNCSIM_NDP_UNIT_H_
#define FUNCSIM_NDP_UNIT_H_

#include "common.h"
#include "memory_map.h"
#include "m2ndp_config.h"
#include "register_unit.h"
#include "sub_core.h"
#ifdef TIMING_SIMULATION
#include "delayqueue.h"
#include "execution_unit.h"
#include "ldst_unit.h"
#include "instruction_buffer.h"
#include "instruction_queue.h"
#include "mem_fetch.h"
#include "ndp_stats.h"
#include "tlb.h"
#include "uthread_generator.h"
#endif
namespace NDPSim {

class NdpUnit {
 public:
  NdpUnit() {}
  NdpUnit(M2NDPConfig* config, MemoryMap* memory_map, int id);
  void Run(int id, NdpKernel* ndp_kernel, std::string line);

#ifdef TIMING_SIMULATION
  NdpUnit(M2NDPConfig* config, MemoryMap* memory_map, NdpStats* stats, int id);
  void cycle();
  void push_from_mem(mem_fetch* req, int bank);
  mem_fetch* top_to_mem(int bank);
  bool to_mem_empty(int bank);
  void pop_to_mem(int bank);
  bool from_mem_full(int bank);

  bool is_active();
  bool can_register();
  bool is_launch_active(int launch_id);
  void register_ndp_kernel(NdpKernel* ndp_kernel);
  void launch_ndp_kernel(KernelLaunchInfo info);
  NdpStats get_stats();
  void print_ndp_stats();
  void print_uthread_stats();

  bool generate_uthreads() { return m_uthread_generator->generate_uthreads(1); }
#endif

 private:
  M2NDPConfig* m_config;
  bool m_sub_core_mode;
  int m_num_ndp;
  int m_num_sub_core;
  int m_id;
  int m_num_xreg;
  int m_num_freg;
  int m_num_vreg;
  NdpKernel* m_ndp_kernel;
  MemoryMap* m_memory_map;
  std::vector<SubCore*> m_sub_core_units;
#ifdef TIMING_SIMULATION
  UThreadGenerator* m_uthread_generator;
  InstructionBuffer* m_instruction_buffer;
  LDSTUnit* m_ldst_unit;
  int m_sub_core_rr = 0; // Sub-core round-robin
  Tlb* m_dtlb;
  Tlb* m_itlb;
  CacheConfig m_icache_config;
  Cache* m_icache;
  DelayQueue<mem_fetch*> m_icache_queue;
  // request queues
  fifo_pipeline<mem_fetch> m_inst_req;
  fifo_pipeline<mem_fetch> m_tlb_req;
  std::vector<fifo_pipeline<mem_fetch>> m_to_mem;
  std::vector<fifo_pipeline<mem_fetch>> m_from_mem;
  std::vector<fifo_pipeline<int64_t>> m_to_reg; //[sub_core_id] = reg_id
  fifo_pipeline<RequestInfo> m_matched_requests;
  //to connect sub-cores
  std::vector<fifo_pipeline<InstColumn>> m_inst_column_q;
  int m_last_sub_core_fetched;

  std::vector<fifo_pipeline<mem_fetch>> m_to_icache;
  std::vector<fifo_pipeline<mem_fetch>> m_from_icache;
  std::vector<fifo_pipeline<std::pair<NdpInstruction, Context>>> m_to_ldst_unit;
  std::vector<fifo_pipeline<std::pair<NdpInstruction, Context>>> m_to_spad_unit;
  std::vector<fifo_pipeline<std::pair<NdpInstruction, Context>>> m_to_v_ldst_unit;
  std::vector<fifo_pipeline<std::pair<NdpInstruction, Context>>> m_to_v_spad_unit;

  std::queue<Context> m_finished_contexts;
  std::vector<int> uthread_count;
  NdpStats* m_stats;
#endif
  // Stat Counters
  uint64_t m_ndp_cycles = 0;

#ifdef TIMING_SIMULATION
  void handle_finished_context();
  void rf_writeback();
  void from_mem_handle();
  void l1_inst_cache_cycle();
  void to_l1_inst_cache();
  void tlb_cycle();
  void connect_to_ldst_unit();
  void connect_instruction_buffer_to_sub_core();
  void request_instruction_lookup();
  
  bool check_finished_context() { return !m_finished_contexts.empty(); }
  Context pop_finished_context() {
    Context finished_context = m_finished_contexts.front();
    m_finished_contexts.pop();
    return finished_context;
  }
#endif
};
}  // namespace NDPSim
#endif  // FUNCSIM_NDP_UNIT_H_
