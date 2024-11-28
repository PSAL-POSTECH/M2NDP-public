#ifdef TIMING_SIMULATION
#ifndef EXECUTION_UNIT_H
#define EXECUTION_UNIT_H
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

namespace NDPSim {


class NdpStats;

enum UnitType {
  SPAD_UNIT,
  LDST_UNIT,
  INT_UNIT,
  SF_UNIT,
  FP_UNIT
};

using ExecutionDelayQueue = DelayQueue<std::pair<NdpInstruction, Context>>;

class ExecutionUnit {
 public:
  ExecutionUnit(M2NDPConfig *config, int ndp_id, int sub_core_id, RegisterUnit *register_unit,
                std::queue<Context> *finished_contexts,
                fifo_pipeline<std::pair<NdpInstruction, Context>> *to_ldst_unit,
                fifo_pipeline<std::pair<NdpInstruction, Context>> *to_spad_unit,
                fifo_pipeline<std::pair<NdpInstruction, Context>> *to_v_ldst_unit,
                fifo_pipeline<std::pair<NdpInstruction, Context>> *to_v_spad_unit,
                NdpStats *stats);
  bool active();
  void cycle();
  Status issue(NdpInstruction &inst, Context context);
  bool full();
  bool check_unit_finished();

  void dump_current_state();

 private:
  int m_ndp_id;
  int m_sub_core_id;
  uint64_t m_cycle;
  M2NDPConfig *m_config;
  RegisterUnit *m_register_unit;
  NdpStats *m_stats;
  fifo_pipeline<std::pair<NdpInstruction, Context>> *m_to_ldst_unit;
  fifo_pipeline<std::pair<NdpInstruction, Context>> *m_to_spad_unit;
  fifo_pipeline<std::pair<NdpInstruction, Context>> *m_to_v_ldst_unit;
  fifo_pipeline<std::pair<NdpInstruction, Context>> *m_to_v_spad_unit;
  /* Scalar units */
  std::vector<ExecutionDelayQueue> m_i_units;
  std::vector<ExecutionDelayQueue> m_f_units;
  std::vector<ExecutionDelayQueue> m_sf_units;
  std::vector<ExecutionDelayQueue> m_address_units;

  /* Vector units */
  std::vector<ExecutionDelayQueue> m_v_i_units;
  std::vector<ExecutionDelayQueue> m_v_f_units;
  std::vector<ExecutionDelayQueue> m_v_sf_units;
  std::vector<ExecutionDelayQueue> m_v_address_units;

  std::queue<Context> *m_finished_contexts;
  std::queue<Context> m_branch_contexts;

  std::set<uint64_t> m_lock_spad_addr;
  void process_default_execution_units(std::vector<ExecutionDelayQueue>& units);
  void process_address_inst(ExecutionDelayQueue &address_unit);
  bool check_units_full(std::vector<ExecutionDelayQueue> &units);
  bool check_units_finished(std::vector<ExecutionDelayQueue> &units);
  Status check_dependency_fail(NdpInstruction& inst, Context& context);
  Status check_resource_fail(NdpInstruction& inst);
  std::vector<ExecutionDelayQueue> &get_unit(NdpInstruction& inst);
  fifo_pipeline<std::pair<NdpInstruction, Context>> *get_fifo_pipeline(NdpInstruction& inst, bool spad);
  UnitType get_unit_type(NdpInstruction& inst);
  void increase_issue_count(NdpInstruction& inst);
};
}  // namespace NDPSim
#endif
#endif