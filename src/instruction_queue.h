#ifdef TIMING_SIMULATION
#ifndef INSTRUCTION_QUEUE_H
#define INSTRUCTION_QUEUE_H
#include <deque>
#include <list>
#include <utility>

#include "cache.h"
#include "delay_queue.h"
#include "ndp_instruction.h"
#include "m2ndp_config.h"
#include "cache.h"

#include "ndp_stats.h"
namespace NDPSim {
typedef std::pair<int, int> DependencyKey;
typedef struct {
  RequestType current_type;
  int current_kernel_body_id;
  int count;
} DependencyInfo;

class InstructionQueue {
 public:
  InstructionQueue(M2NDPConfig *config, int id, int sub_core_id, Cache* icache);
  bool can_push(InstColumn* inst_column);
  void push(InstColumn* inst_column);
  NdpInstruction fetch_instruction();
  Context fetch_context();
  void fetch_success() {
    (*m_inst_columns_iter)->fetched_inst = (*m_inst_columns_iter)->current_inst;
    (*m_inst_columns_iter)->counter = 0;
     (*m_inst_columns_iter)->current_inst = NULL; }
  void cycle();
  bool can_fetch();
  Status inst_fetch_fail();
  bool all_empty();
  bool empty();
  void next();
  int get_num_columns();
  void finish_inst_column(int col_id);
  void set_ideal_icache();
 private:
  int m_id;
  int m_sub_core_id;
  int m_inst_col_id;
  bool m_ideal_icache = false;
  int m_l0_icache_hit_latency;
  M2NDPConfig *m_config;
  Cache *m_icache;
  DelayQueue<mem_fetch*> m_icache_queue;
  int32_t m_num_columns;
  std::deque<InstColumn*> m_inst_columns;
  std::deque<InstColumn*>::iterator m_inst_columns_iter;
  std::map<DependencyKey, DependencyInfo> m_dependency_info;
  std::deque<InstColumn*>::iterator find_inst_column(int inst_col_id);
  bool check_finished(std::deque<InstColumn*>::iterator iter);
  DependencyKey get_dependency_key(InstColumn* inst_column);
};
}
#endif  // INSTRUCTOIN_QUEUE_H
#endif  // TIMING_SIMULATION