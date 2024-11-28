#ifdef TIMING_SIMULATION
#ifndef LDST_UNIT_H
#define LDST_UNIT_H
#include <queue>
#include <set>

#include "common.h"
#include "delay_queue.h"
#include "delayqueue.h"
#include "mem_fetch.h"
#include "ndp_instruction.h"
#include "m2ndp_config.h"
#include "register_unit.h"
#include "tlb.h"
#include "ndp_stats.h"
#include "cache.h"
namespace NDPSim {


class NdpStats;

using ExecutionDelayQueue = DelayQueue<std::pair<NdpInstruction, Context>>;

class LDSTUnit {
 public:
  LDSTUnit(M2NDPConfig *config, int ndp_id, 
                std::queue<Context> *finished_contexts,
                std::vector<fifo_pipeline<mem_fetch>> *to_mem,
                std::vector<fifo_pipeline<mem_fetch>> *from_mem, 
                std::vector<fifo_pipeline<int64_t>> *to_reg,
                Tlb *tlb, NdpStats *stats);
  void set_l1d_size(int smem_size);
  bool active();
  void cycle();
  Status issue_ldst_unit(NdpInstruction inst, Context context, bool spad_op);
  bool full();
  bool check_unit_finished();
  void handle_from_mem(int bank);
  void dump_current_state();
  CacheStats get_l1d_stats();

 private:
  int m_ndp_id;
  bool skip_l1d;
  int m_l1d_banks;
  CacheConfig m_l1d_config;
  DataCache *m_l1d_cache;
  std::vector<std::vector<mem_fetch*>> m_l1_latency_queue_;
  std::vector<DelayQueue<mem_fetch*>> m_l1_latency_queue;
  fifo_pipeline<mem_fetch> m_to_tlb_queue;
  uint64_t m_cycle;
  M2NDPConfig *m_config;
  Tlb *m_tlb;
  uint32_t m_tlb_waiting_queue; 
  NdpStats *m_stats;
  std::vector<fifo_pipeline<mem_fetch>> *m_to_mem;
  std::vector<fifo_pipeline<mem_fetch>> *m_from_mem;
  std::vector<fifo_pipeline<int64_t>> *m_to_reg; //m_to_reg[sub_core_id] = reg_id
  /* Scalar units */
  std::vector<ExecutionDelayQueue> m_ldst_units;
  std::vector<ExecutionDelayQueue> m_spad_units;

  /* Vector units */
  std::vector<ExecutionDelayQueue> m_v_ldst_units;
  std::vector<ExecutionDelayQueue> m_v_spad_units;

  ExecutionDelayQueue m_spad_delay_queue = ExecutionDelayQueue("spad_delay_queue");

  robin_hood::unordered_map<mem_fetch*, NdpInstruction> m_pending_ldst;
  robin_hood::unordered_map<mem_fetch*, Context> m_pending_ldst_context;
  uint32_t m_pending_write_count;
  std::queue<Context> *m_finished_contexts;

  std::set<uint64_t> m_lock_spad_addr;
  void process_ldst_inst(ExecutionDelayQueue &ldst_unit);
  void process_spad_inst(ExecutionDelayQueue &spad_unit);
  bool check_units_full(std::vector<ExecutionDelayQueue> &units);
  bool check_units_finished(std::vector<ExecutionDelayQueue> &units);
  std::vector<ExecutionDelayQueue> &get_ldst_unit(NdpInstruction inst, bool spad);
  int m_l1_hit_latency;
  int m_max_l1_latency_queue_size;
  std::vector<int> m_l1_latency_pipeline_max_index;
  bool check_l1_hit_pipeline_full(NdpInstruction &inst);
  void l1_latency_queue_cycle();
  void process_l1d_access();
  void handle_response(mem_fetch* mf);
};
}  // namespace NDPSim
#endif
#endif