#include "sub_core.h"

#include <math.h>

#include <fstream>
#include <sstream>

#include "ndp_instruction.h"
#include "m2ndp_parser.h"
namespace NDPSim {

SubCore::SubCore(M2NDPConfig* config, MemoryMap* memory_map, int id, int sub_core_id)
    : m_config(config), m_memory_map(memory_map), m_id(id), m_sub_core_id(sub_core_id) {
  m_num_ndp = config->get_num_ndp_units();
  m_register_unit = new RegisterUnit(m_config->get_num_x_registers(),
                                     m_config->get_num_f_registers(),
                                     m_config->get_num_v_registers());
}

#ifdef TIMING_SIMULATION
SubCore::SubCore(M2NDPConfig* config, MemoryMap* memory_map, NdpStats* stats, int id, int sub_core_id,
                fifo_pipeline<InstColumn>* inst_column_q, fifo_pipeline<mem_fetch>* to_icache, fifo_pipeline<mem_fetch>* from_icache,
                fifo_pipeline<std::pair<NdpInstruction, Context>> *to_ldst_unit,
                fifo_pipeline<std::pair<NdpInstruction, Context>> *to_spad_unit,
                fifo_pipeline<std::pair<NdpInstruction, Context>> *to_v_ldst_unit,
                fifo_pipeline<std::pair<NdpInstruction, Context>> *to_v_spad_unit,
                std::queue<Context> *finished_contexts)
    : m_config(config), m_memory_map(memory_map), m_stats(stats), m_id(id), m_sub_core_id(sub_core_id), 
    m_inst_column_q(inst_column_q), m_to_icache(to_icache), m_from_icache(from_icache), m_finished_contexts(finished_contexts),
    m_to_ldst_unit(to_ldst_unit), m_to_spad_unit(to_spad_unit), m_to_v_ldst_unit(to_v_ldst_unit), m_to_v_spad_unit(to_v_spad_unit) {
  m_num_ndp = config->get_num_ndp_units();
  m_register_unit = new RegisterUnit(m_config->get_num_x_registers(),
                                     m_config->get_num_f_registers(),
                                     m_config->get_num_v_registers());

  m_icache_config.init(m_config->get_l0icache_config(), m_config);
  m_l0_icache = new ReadOnlyCache("l0_icache", m_icache_config, m_id, 0, m_to_icache); //TODO: l0 icache config
  m_instruction_queue = new InstructionQueue(m_config, m_id, m_sub_core_id, m_l0_icache);
  m_execution_unit = new ExecutionUnit(m_config, m_id, m_sub_core_id, m_register_unit,
                                        m_finished_contexts,
                                        m_to_ldst_unit,
                                        m_to_spad_unit,
                                        m_to_v_ldst_unit,
                                        m_to_v_spad_unit,
                                        m_stats);

  if (m_config->get_ideal_icache()) 
    m_instruction_queue->set_ideal_icache();
  m_active_queque_count = 0;
  m_issue_fail_count = 0;
  m_deadlock_count = 1000000;
}
#endif

void SubCore::ExecuteInitializer(MemoryMap* spad_map, RequestInfo* info) {
  std::deque<NdpInstruction> renamed = m_register_unit->Convert(
      m_ndp_kernel->initializer_insts, info->id, info);
  ExecuteInsts_Array(spad_map, renamed, info, m_ndp_kernel->loop_map);
  m_register_unit->FreeRegs(info->id);
}

void SubCore::ExecuteKernelBody(MemoryMap* spad_map, RequestInfo* info,
                                int kernel_body_id) {
  std::deque<NdpInstruction> renamed = m_register_unit->Convert(
      m_ndp_kernel->kernel_body_insts[kernel_body_id], info->id, info);
  ExecuteInsts_Array(spad_map, renamed, info, m_ndp_kernel->loop_map);
  m_register_unit->FreeRegs(info->id);
}

void SubCore::ExecuteFinalizer(MemoryMap* spad_map, RequestInfo* info) {
  std::deque<NdpInstruction> finalizer_renamed =
      m_register_unit->Convert(m_ndp_kernel->finalizer_insts, info->id, info);
  ExecuteInsts_Array(spad_map, finalizer_renamed, info, m_ndp_kernel->loop_map);
  m_register_unit->FreeRegs(info->id);
}

void SubCore::ExecuteInsts(MemoryMap* spad_map,
                           std::deque<NdpInstruction> insts,
                           RequestInfo* info) {
  CSR csr;
  Context context;
  context.ndp_id = m_id;
  context.csr = &csr;
  context.memory_map = m_memory_map;
  context.scratchpad_map = spad_map;
  context.register_map = m_register_unit;
  context.request_info = info;
  for (NdpInstruction inst : insts) {
    inst.Execute(context);
  }
}

void SubCore::ExecuteInsts_Array(MemoryMap* spad_map,
                                 std::deque<NdpInstruction> insts,
                                 RequestInfo* info,
                                 const std::map<int, int>& loop_map) {
  CSR csr;
  Context context;
  context.ndp_id = m_id;
  context.csr = &csr;
  context.memory_map = m_memory_map;
  context.scratchpad_map = spad_map;
  context.register_map = m_register_unit;
  context.request_info = info;
  for (int i = 0; i < insts.size(); i++) {
    try {
      insts.at(i).Execute(context);
      int key = insts.at(i).IsBranch();
      if (key >= 0) {
        i = loop_map.find(key)->second - 1;
      }
    } catch (const std::runtime_error& error) {
      spdlog::error("=============================EXECUTION PANIC=============================");
      for (int j= i-5 > 0 ?: 0; j<i; j++)
        spdlog::error("[{}/{}] {}", j, insts.size(), insts.at(j).sInst);
      spdlog::error("[{}/{}] {} <-- triggered panic!!", i, insts.size(), insts.at(i).sInst);
      m_register_unit->DumpRegisterFile(info->offset / PACKET_SIZE);
      throw error;
    }
  }
}

#ifdef TIMING_SIMULATION
void SubCore::cycle() {
  execute_instruction();
  l0_inst_cache_cycle();
  instruction_queue_allocate();
  instruction_register_allocate();
  m_instruction_queue->cycle();
  m_execution_unit->cycle();
}

void SubCore::execute_instruction() {
  std::deque<Status> issue_fail_reasons;
  bool issued = false;
  int issue_count = 0;
  if (m_instruction_queue->get_num_columns() == 0) {
    issue_fail_reasons.push_back(EMPTY_QUEUE);
  }
  m_stats->inc_ndp_inst_queue_size(m_instruction_queue->get_num_columns());
  m_active_queque_count += m_instruction_queue->get_num_columns();
  for (int qid = 0; qid < m_instruction_queue->get_num_columns(); qid++) {
    if (m_execution_unit->full()) break;
    if (m_instruction_queue->can_fetch()) {
      NdpInstruction inst = m_instruction_queue->fetch_instruction();
      Context context = m_instruction_queue->fetch_context();
      context.memory_map = m_memory_map;
      context.scratchpad_map = context.request_info->scratchpad_map;
      context.register_map = m_register_unit;
      // try {
        Status status = m_execution_unit->issue(inst, context);
        if (status == ISSUE_SUCCESS) {
          issued = true;
          issue_count++;
          m_instruction_queue->fetch_success();
        } else {
          issue_fail_reasons.push_back(status);
        }
      // } catch (const std::runtime_error& error) {
        // spdlog::error("=============================EXECUTION PANIC=============================");
        // spdlog::error("{} <-- triggered panic!!", inst.sInst);
        // spdlog::error("{}", context.request_info->offset);
        // m_register_unit->DumpRegisterFile(-1);
        // throw error;
      // }
    } else {
      Status status = m_instruction_queue->inst_fetch_fail();
      if (status != SUCCESS)
        issue_fail_reasons.push_back(status);
    }
    m_instruction_queue->next();
    if(m_config->get_max_inst_issue() && issue_count >= m_config->get_max_inst_issue()) break;
  }
  if (issued) {
    if (m_sub_core_id == 0)
      m_stats->inc_issue_success(issue_count);
    m_stats->inc_inst_issue_count(issue_count);
    m_issue_fail_count = 0;
  } else {
    if (!m_instruction_queue->all_empty()) m_issue_fail_count++;
  }
  m_stats->add_status_list(issue_fail_reasons);
  if (m_issue_fail_count > 50000 &&
      !m_instruction_queue->all_empty()) {
    spdlog::error("NDP {} is deadlocked {} {}", m_id, m_issue_fail_count,
                  m_deadlock_count);
    for (int i = 0; i < m_instruction_queue->get_num_columns(); i++) {
      if (m_instruction_queue->can_fetch()) {
        spdlog::error("Blocked inst {}",
                      m_instruction_queue->fetch_instruction().sInst);
      } else {
        spdlog::error("Blocked inst");
      }
      m_instruction_queue->next();
    }
    spdlog::error("{}", issue_fail_reasons);
    m_execution_unit->dump_current_state();
    assert(false);
    throw std::runtime_error("Deadlock");
  }
}

void SubCore::l0_inst_cache_cycle() {
  //connect m_from_icache to m_l0_icache
  if (!m_from_icache->empty()) {
    mem_fetch* mf = m_from_icache->top();
    if (m_l0_icache->waiting_for_fill(mf)) {
      if (m_l0_icache->fill_port_free()) {
        m_l0_icache->fill(mf, m_config->get_ndp_cycle());
        m_from_icache->pop();
      }
    }
  }
  m_l0_icache->cycle();
}

void SubCore::instruction_queue_allocate() {
  if (m_register_unit->RenameEmpty()) return;
  InstColumn* inst = m_register_unit->RenameTop();
  if (!m_instruction_queue->can_push(inst)) {
    m_stats->inc_inst_queue_full();
    return;
  }
  m_register_unit->RenamePop();
  if (inst->insts.empty()) {
    //have to push to finished_contexts
    //Make temp context
    Context temp_context;
    temp_context.ndp_id = m_id;
    temp_context.sub_core_id = -1; //To indicate this is empty inst context
    temp_context.request_info = inst->req;

    m_finished_contexts->push(temp_context);
  } else {
    m_instruction_queue->push(inst);
  }
}

void SubCore::instruction_register_allocate() {
  if (m_inst_column_q->empty()) return;
  InstColumn* inst = m_inst_column_q->top();
  if (m_register_unit->RenameFull(inst)) {
    m_stats->inc_register_stall();
    return;
  }
  m_inst_column_q->pop();
  m_register_unit->RenamePush(inst, inst->req->id);
}

bool SubCore::is_active() {

  return m_execution_unit->active() ||
         !m_instruction_queue->all_empty() ||
         !m_inst_column_q->empty() ||
         !m_to_icache->empty() ||
         !m_from_icache->empty();
}

RegisterStats SubCore::get_register_stats() {
  return m_register_unit->get_register_stats();
}

CacheStats SubCore::get_l0_icache_stats() {
  return m_l0_icache->get_stats();
}

void SubCore::print_sub_core_stats() {
  float avg_active_queue =
      ((float)m_active_queque_count) / m_config->get_log_interval();
  if (m_id == 0 && m_sub_core_id == 0)
    spdlog::info("NDP {:2} SUB-CORE {:2} : average active queue count: {:.2f} / {} ({:.2f} %)",
                 m_id, m_sub_core_id, avg_active_queue, m_config->get_uthread_slots(),
                 avg_active_queue / m_config->get_uthread_slots() * 100);
  else
    spdlog::debug(
        "NDP {:2} SUB-CORE {:2}: average active queue count: {:.2f} / {} ({:.2f} %)", 
        m_id, m_sub_core_id, avg_active_queue, m_config->get_uthread_slots(),
        avg_active_queue / m_config->get_uthread_slots() * 100);
  m_active_queque_count = 0;
}

#endif
}  // namespace NDPSim