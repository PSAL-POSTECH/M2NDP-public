#include "ndp_unit.h"

#include <math.h>

#include <fstream>
#include <sstream>

#include "ndp_instruction.h"
#include "m2ndp_parser.h"
namespace NDPSim {

NdpUnit::NdpUnit(M2NDPConfig* config, MemoryMap* memory_map, int id)
    : m_config(config), m_memory_map(memory_map), m_id(id) {
  m_num_ndp = config->get_num_ndp_units();
  m_sub_core_mode = config->is_enable_sub_core();
  m_num_sub_core = config->get_num_sub_core();
  m_sub_core_units.resize(m_num_sub_core);
  for (int i = 0 ; i < m_num_sub_core; i++) {
    m_sub_core_units[i] = new SubCore(m_config, m_memory_map, m_id, i);
  }
}

#ifdef TIMING_SIMULATION
NdpUnit::NdpUnit(M2NDPConfig* config, MemoryMap* memory_map, NdpStats* stats,
                 int id)
    : m_config(config), m_memory_map(memory_map), m_stats(stats), m_id(id) {
  m_num_ndp = config->get_num_ndp_units();
  m_sub_core_mode = config->is_enable_sub_core();
  m_num_sub_core = config->get_num_sub_core();
  m_inst_req = fifo_pipeline<mem_fetch>("inst_req", 0,
                                        m_config->get_request_queue_size());
  m_tlb_req = fifo_pipeline<mem_fetch>("tlb_req", 0,
                                       m_config->get_request_queue_size());
  int num_banks = m_config->get_l2d_num_banks();
  m_from_mem.resize(
      num_banks, fifo_pipeline<mem_fetch>("from_mem", 0,
                                          m_config->get_request_queue_size()));
  m_to_mem.resize(num_banks,
                  fifo_pipeline<mem_fetch>("to_mem", 0,
                                           m_config->get_request_queue_size()));
  m_to_reg.resize(m_num_sub_core,
                  fifo_pipeline<int64_t>("to_reg", 0,
                                     m_config->get_request_queue_size()));
  m_matched_requests = fifo_pipeline<RequestInfo>(
      "matched_requests", 0, 2048);
  m_inst_column_q.resize(
      m_num_sub_core, fifo_pipeline<InstColumn>("inst_column", 0,
                                                m_config->get_request_queue_size()));
  m_to_icache.resize(
      m_num_sub_core, fifo_pipeline<mem_fetch>("to_icache", 0,
                                               m_config->get_request_queue_size()));
  m_from_icache.resize(
      m_num_sub_core, fifo_pipeline<mem_fetch>("from_icache", 0,
                                               m_config->get_request_queue_size()));
  m_to_ldst_unit.resize(
      m_num_sub_core, fifo_pipeline<std::pair<NdpInstruction, Context>>(
                         "to_ldst_unit", 0, m_config->get_request_queue_size()));
  m_to_spad_unit.resize(
      m_num_sub_core, fifo_pipeline<std::pair<NdpInstruction, Context>>(
                         "to_spad_unit", 0, m_config->get_request_queue_size()));
  m_to_v_ldst_unit.resize(
      m_num_sub_core, fifo_pipeline<std::pair<NdpInstruction, Context>>(
                         "to_v_ldst_unit", 0, m_config->get_request_queue_size()));
  m_to_v_spad_unit.resize(
      m_num_sub_core, fifo_pipeline<std::pair<NdpInstruction, Context>>(
                         "to_v_spad_unit", 0, m_config->get_request_queue_size()));
  m_dtlb = new Tlb(id, m_config, m_config->get_dtlb_config(), &m_tlb_req);
  m_itlb = new Tlb(id, m_config, m_config->get_itlb_config(), &m_tlb_req);
  if(m_config->get_ideal_tlb()) {
    m_dtlb->set_ideal_tlb();
    m_itlb->set_ideal_tlb();
  }
  m_uthread_generator = new UThreadGenerator(m_config, m_id, &m_matched_requests);
  m_instruction_buffer =
      new InstructionBuffer(m_config->get_inst_buffer_size(),
                            m_config->get_request_queue_size(), m_id);
  m_icache_config.init(m_config->get_l1icache_config(), m_config);
  m_icache = new ReadOnlyCache("l1icache", m_icache_config, m_id, 0, &m_inst_req);
  m_icache_queue = DelayQueue<mem_fetch *>("l1_icache_queue", true, m_config->get_request_queue_size());
                            
  m_sub_core_units.resize(m_num_sub_core);
  for (int sub_core_id = 0; sub_core_id < m_num_sub_core; sub_core_id++) {
    m_sub_core_units[sub_core_id] = new SubCore(m_config, m_memory_map, stats, m_id, sub_core_id,
                                      &(m_inst_column_q[sub_core_id]), &(m_to_icache[sub_core_id]), &(m_from_icache[sub_core_id]),
                                      &(m_to_ldst_unit[sub_core_id]), &(m_to_spad_unit[sub_core_id]),
                                      &(m_to_v_ldst_unit[sub_core_id]), &(m_to_v_spad_unit[sub_core_id]),
                                      &m_finished_contexts);
    
  }

  m_ldst_unit = new LDSTUnit(m_config, m_id, 
                            &m_finished_contexts,
                            &m_to_mem, &m_from_mem, 
                            &m_to_reg,
                            m_dtlb, m_stats);
  m_last_sub_core_fetched = 0;
  uthread_count.resize(m_sub_core_units.size(), 0);
}
#endif

void NdpUnit::Run(int id, NdpKernel* ndp_kernel, std::string line) {
  m_ndp_kernel = ndp_kernel;
  //set ndp_kernel to sub_core
  for (int i = 0; i < m_num_sub_core; i++) {
    m_sub_core_units[i]->set_ndp_kernel(ndp_kernel);
  }
  KernelLaunchInfo* kinfo = M2NDPParser::parse_kernel_launch(line, 0);
  kinfo->launch_id = 0;
  assert(kinfo->smem_size <= m_config->get_spad_size() * 1024); // in KB
  MemoryMap* scratchpad_map =
      new HashMemoryMap(SCRATCHPAD_BASE, kinfo->smem_size);
  bool first = true;
  uint64_t spad_addr = SCRATCHPAD_BASE;
  int num_args = kinfo->arg_size / DOUBLE_SIZE - kinfo->num_float_args;
  int num_args_per_packet = PACKET_SIZE / DOUBLE_SIZE;
  int outter = 0;
  for(outter = 0; outter < (num_args - 1) / num_args_per_packet + 1; outter++){
    VectorData data(64,1);
    data.SetType(INT64);
    for(int inner = 0; inner < num_args_per_packet; inner++) {
      int offset = outter * num_args_per_packet + inner;
      if(offset >= num_args) break;
      data.SetData((int64_t)kinfo->args[offset], inner);
      spdlog::debug("NDP {} : Scratchpad {:x} {:x}", m_id, spad_addr + offset * 8,
                  kinfo->args[offset]);
    }
    scratchpad_map->Store(spad_addr + outter * PACKET_SIZE, data);
  }
  if (kinfo->num_float_args > 0) {
    assert(kinfo->num_float_args <= MAX_NR_FLOAT_ARG);
    VectorData data(32,1);
    data.SetType(FLOAT32);
    for (int float_idx = 0; float_idx < kinfo->num_float_args; float_idx++) {
      data.SetData((float)kinfo->float_args[float_idx], float_idx);
      spdlog::debug("NDP {} : Scratchpad {:x} {}", m_id,
                    spad_addr + outter * PACKET_SIZE,
                    kinfo->float_args[float_idx]);
    }
    scratchpad_map->Store(spad_addr + outter * PACKET_SIZE, data);
  }
  int req_id = 0;
  try {
    for(int k_id = 0 ; k_id < ndp_kernel->num_kernel_bodies; k_id++) {
      uint32_t count = 0;
      while (count * PACKET_SIZE < kinfo->size) {
        uint64_t target_addr = kinfo->base_addr + count * PACKET_SIZE;
        count++;
        int addr_ndp_id = m_config->get_matched_unit_id(target_addr);
        if (addr_ndp_id != m_id) continue;
        if (first) {
          RequestInfo info;
          info.addr = kinfo->base_addr;
          info.offset = m_id;
          info.kernel_id = kinfo->kernel_id;
          info.launch_id = kinfo->launch_id;
          info.id = req_id++;
          //only use 1 sub-core in functional only mode
          m_sub_core_units[0]->ExecuteInitializer(scratchpad_map, &info);
          first = false;
        }
        uint64_t offset = target_addr - kinfo->base_addr;
        RequestInfo info;
        info.addr = target_addr;  // base_addr, in previous
        info.offset = offset;
        info.launch_id = kinfo->launch_id;
        info.kernel_body_id = k_id;
        info.id = req_id++;
        m_sub_core_units[0]->ExecuteKernelBody(scratchpad_map, &info, k_id);
      }
    }
    if (!first) {
      RequestInfo info;
      info.addr = kinfo->base_addr;
      info.offset = m_id;
      info.kernel_id = kinfo->kernel_id;
      info.launch_id = kinfo->launch_id;
      info.id = req_id++;
      m_sub_core_units[0]->ExecuteFinalizer(scratchpad_map, &info);
    }
    delete scratchpad_map;
  } catch (const std::runtime_error& error) {
    spdlog::error("=============================NDP Unit {} DUMP=============================", m_id);
    spdlog::error("Scratchpad Memory Dump:");
    scratchpad_map->DumpMemory();
    delete scratchpad_map;
    spdlog::error("Global Memory Dump:");
    m_memory_map->DumpMemory();
    throw error;
  }
}

#ifdef TIMING_SIMULATION
void NdpUnit::cycle() {
  handle_finished_context();
  rf_writeback();
  from_mem_handle();
  l1_inst_cache_cycle();
  to_l1_inst_cache();
  tlb_cycle();
  m_ldst_unit->cycle();
  connect_to_ldst_unit();
  // try {
    bool inst_queue_empty = true;
    for (int i = 0; i < m_num_sub_core; i++) {
      m_sub_core_units[i]->cycle();
      inst_queue_empty = inst_queue_empty && m_sub_core_units[i]->is_inst_queue_empty();
    }
    if(inst_queue_empty && m_config->is_coarse_grained()) {
      m_uthread_generator->generate_uthreads(m_config->get_uthread_slots() * m_num_sub_core);
    }
  // } catch (const std::runtime_error& error) {
    // spdlog::error("=============================NDP Unit {} DUMP=============================", m_id);
    // spdlog::error("Global Memory Dump:");
    // m_memory_map->DumpMemory();
    // throw error;
  // }
  //fetch one inst_column to sub_core per cycle
  connect_instruction_buffer_to_sub_core();
  request_instruction_lookup();
  m_uthread_generator->cycle();
  m_instruction_buffer->cycle();
  m_ndp_cycles++;
  m_stats->inc_cycle();
}

void NdpUnit::handle_finished_context() {
  while (check_finished_context()) {
    Context context = pop_finished_context();
    int sub_core_id = context.sub_core_id;
    if (context.request_info->type == FINALIZER) {
      m_uthread_generator->finish_launch(context.request_info->launch_id);
    }

    if (sub_core_id == -1) {
      m_uthread_generator->increase_count(context.request_info->launch_id);
    } else {
      m_uthread_generator->increase_count(context.request_info->launch_id);
      m_sub_core_units[sub_core_id]->free_rf(context.request_info->id);
      m_sub_core_units[sub_core_id]->free_inst_column(context.inst_col_id);
    }
  }
}

void NdpUnit::rf_writeback() {
  //connect LDST unit to register unit
  //WB
  //have to discuss - rf writeback all data from m_to_reg at one cycle?
  for (int sub_core_id = 0; sub_core_id < m_num_sub_core; sub_core_id++) {
    uint64_t num_vreg_wb_cnt = 0;
    uint64_t num_xreg_wb_cnt = 0;
    uint64_t num_freg_wb_cnt = 0;
    uint64_t num_total_wb_cnt = 0;
    while (!m_to_reg[sub_core_id].empty()) {
      int64_t *reg_id = m_to_reg[sub_core_id].top();
      m_sub_core_units[sub_core_id]->set_rf_ready(*reg_id);
      m_to_reg[sub_core_id].pop();

      //count wb
      if (m_sub_core_units[sub_core_id]->is_vreg(*reg_id)) {
        num_vreg_wb_cnt++;
      } else if (m_sub_core_units[sub_core_id]->is_xreg(*reg_id)) {
        num_xreg_wb_cnt++;
      } else if (m_sub_core_units[sub_core_id]->is_freg(*reg_id)) {
        num_freg_wb_cnt++;
      }
      
      delete reg_id;
    }
    //update max wb per cycle
    num_total_wb_cnt = num_vreg_wb_cnt + num_xreg_wb_cnt + num_freg_wb_cnt;
    if (m_stats->get_max_wb_per_cyc() < num_total_wb_cnt) {
      m_stats->set_max_wb_per_cyc(num_total_wb_cnt, num_vreg_wb_cnt, num_xreg_wb_cnt, num_freg_wb_cnt);
    }
  }
}

void NdpUnit::from_mem_handle() {
  for (int bank = 0; bank < m_config->get_l2d_num_banks(); bank++) {
    if (!m_from_mem[bank].empty()) {
      mem_fetch* mf = m_from_mem[bank].top();
      mf->current_state = "from mem handle";
      if (m_dtlb->waiting_for_fill(mf)) {
        if (m_dtlb->fill_port_free()) {

          m_dtlb->fill(mf);
          m_from_mem[bank].pop();
          
        }
      } else if (m_itlb->waiting_for_fill(mf)) {
        if (m_itlb->fill_port_free()) {
          m_itlb->fill(mf);
          m_from_mem[bank].pop();
        }
      } else if (m_icache->waiting_for_fill(mf)) {
        if (m_icache->fill_port_free()) {
          m_icache->fill(mf, m_config->get_sim_cycle());
          m_from_mem[bank].pop();
        }
      } else {
        m_ldst_unit->handle_from_mem(bank);
      }
    }
  }
}

void NdpUnit::l1_inst_cache_cycle() {
  m_icache->cycle();
  m_icache_queue.cycle();

  //connect l1 icache to tlb & mem
  std::deque<Status> queue_full_reasons;
  if (!m_inst_req.empty() && m_itlb->data_port_free()) {
    mem_fetch* mf = m_inst_req.top();
    m_itlb->access(mf);
    m_inst_req.pop();
  }
  if (m_itlb->data_ready()) {
    mem_fetch* mf = m_itlb->get_data();
    int bank = m_config->get_bank_index(mf->get_addr()) % m_config->get_l2d_num_banks();
    if (!m_to_mem[bank].full()) {
      m_to_mem[bank].push(mf);
      m_itlb->pop_data();
    } else
      queue_full_reasons.push_back(TO_MEM_FULL);
  }
  //access l1 icache & connect l1 icache to sub-core
  for (int i = 0; i < m_num_sub_core; i++) {
    if (m_icache_queue.empty()) break;
    mem_fetch* mf = m_icache_queue.top();
    int sub_core_id = mf->get_sub_core_id();

    std::deque<CacheEvent> events;
    CacheRequestStatus status = 
      m_icache->access(mf->get_addr(), m_config->get_ndp_cycle(), mf, events);
    if (status == HIT) {
      if (!m_from_icache[sub_core_id].full()) {
        m_from_icache[sub_core_id].push(mf);
        m_icache_queue.pop();
      } else {
        queue_full_reasons.push_back(FROM_ICACHE_FULL);
      }
    } else if (status != RESERVATION_FAIL) {
      m_icache_queue.pop();
    }
  }
  for (int i = 0; i < m_num_sub_core; i++) {
    if (m_icache->access_ready()) {
      mem_fetch* mf = m_icache->top_next_access();
      int sub_core_id = mf->get_sub_core_id();
      if (!m_from_icache[sub_core_id].full()) {
        mem_fetch* mf = m_icache->pop_next_access();
        m_from_icache[sub_core_id].push(mf);
      } else {
        queue_full_reasons.push_back(FROM_ICACHE_FULL);
      }
    }
  }
  m_stats->add_status_list(queue_full_reasons);
}

void NdpUnit::to_l1_inst_cache() {
  //connect m_to_icache to icache 
  //each sub-core is connected to l1 icache
  for (int sub_core_id = 0; sub_core_id < m_num_sub_core; sub_core_id++) {
    if (!m_to_icache[sub_core_id].empty()) {
      mem_fetch* mf = m_to_icache[sub_core_id].top();
      if (!m_icache_queue.full()) { //l1 icache latency model
        m_icache_queue.push(mf, m_config->get_l1icache_hit_latency());
        m_to_icache[sub_core_id].pop();
      }
    }
  }
}

void NdpUnit::tlb_cycle() {
  std::deque<Status> queue_full_reasons;
  for (int bank = 0; bank < m_config->get_l2d_num_banks(); bank++) {
    m_dtlb->bank_access_cycle();
    if (!m_tlb_req.empty()) {
      mem_fetch* mf = m_tlb_req.top();
      int target_bnak = m_config->get_bank_index(mf->get_addr()) %
                        m_config->get_l2d_num_banks();
      if (bank != target_bnak) continue;
      if (!m_to_mem[bank].full()) {
        m_to_mem[bank].push(mf);
        m_tlb_req.pop();
      } else
        queue_full_reasons.push_back(TO_MEM_FULL);
    }
  }
  m_stats->add_status_list(queue_full_reasons);
  m_dtlb->cycle();
  m_itlb->bank_access_cycle();
  m_itlb->cycle();
}

void NdpUnit::connect_to_ldst_unit() {
  //connect sub-core to ldst unit
  std::deque<Status> issue_fail_reasons;
  for (int iter = 0; iter < m_num_sub_core; iter++) {
    int sub_core_id = (m_sub_core_rr + iter) % m_num_sub_core;
    if (!m_to_ldst_unit[sub_core_id].empty()) {
      std::pair<NdpInstruction, Context> *inst_context =
          m_to_ldst_unit[sub_core_id].top();
      Status status = m_ldst_unit->issue_ldst_unit(inst_context->first, inst_context->second,
                                   /*spad_op*/false);
      if (status == ISSUE_SUCCESS) {
        m_to_ldst_unit[sub_core_id].pop();
        delete inst_context;
      } else {
        issue_fail_reasons.push_back(status);
      }
    }
    if (!m_to_spad_unit[sub_core_id].empty()) {
      std::pair<NdpInstruction, Context> *inst_context =
          m_to_spad_unit[sub_core_id].top();
      Status status = m_ldst_unit->issue_ldst_unit(inst_context->first, inst_context->second,
                                   /*spad_op*/true);
      if (status == ISSUE_SUCCESS) {
        m_to_spad_unit[sub_core_id].pop();
        delete inst_context;
      } else {
        issue_fail_reasons.push_back(status);
      }
    }
    if (!m_to_v_ldst_unit[sub_core_id].empty()) {
      std::pair<NdpInstruction, Context> *inst_context =
          m_to_v_ldst_unit[sub_core_id].top();
      Status status = m_ldst_unit->issue_ldst_unit(inst_context->first, inst_context->second,
                                   /*spad_op*/false);
      if (status == ISSUE_SUCCESS) {
        m_to_v_ldst_unit[sub_core_id].pop();
        delete inst_context;
      } else {
        issue_fail_reasons.push_back(status);
      }
    }
    if (!m_to_v_spad_unit[sub_core_id].empty()) {
      std::pair<NdpInstruction, Context> *inst_context =
          m_to_v_spad_unit[sub_core_id].top();
      Status status = m_ldst_unit->issue_ldst_unit(inst_context->first, inst_context->second,
                                   /*spad_op*/true);
      if (status == ISSUE_SUCCESS) {
        m_to_v_spad_unit[sub_core_id].pop();
        delete inst_context;
      } else {
        issue_fail_reasons.push_back(status);
      }
    }
  }
  m_stats->add_status_list(issue_fail_reasons);
  m_sub_core_rr = (m_sub_core_rr + 1) % m_num_sub_core;
}

void NdpUnit::connect_instruction_buffer_to_sub_core() {
  if (!m_instruction_buffer->can_pop()) return;
  for (unsigned i = 0; i < m_num_sub_core; i++) {
    unsigned sub_core_id = (m_last_sub_core_fetched + 1 + i) % m_num_sub_core;
    bool sc_issue_condition = false;
    if(m_sub_core_units[sub_core_id]->is_inst_queue_empty()) {
      uthread_count[sub_core_id] = m_config->get_uthread_slots();
    }
    if (m_inst_column_q[sub_core_id].full()) {
      m_stats->inc_inst_column_q_full();
    } else if(uthread_count[sub_core_id] > 0) {
      InstColumn* inst = m_instruction_buffer->top();
      m_inst_column_q[sub_core_id].push(inst);
      m_instruction_buffer->pop();
      m_last_sub_core_fetched = sub_core_id;
      if(m_config->is_subcore_coarse_grained()) {
        uthread_count[sub_core_id]--;
      }
      break;
    }
  } 
}

void NdpUnit::request_instruction_lookup() {
  if (m_matched_requests.empty()) return;
  if (!m_instruction_buffer->can_push_request()) return;
  RequestInfo* info = m_matched_requests.top();
  m_matched_requests.pop();
  m_instruction_buffer->push(info);
}

void NdpUnit::push_from_mem(mem_fetch* mf, int bank) {
  m_from_mem[bank].push(mf);
}

mem_fetch* NdpUnit::top_to_mem(int bank) { return m_to_mem[bank].top(); }

bool NdpUnit::to_mem_empty(int bank) { return m_to_mem[bank].empty(); }

void NdpUnit::pop_to_mem(int bank) { m_to_mem[bank].pop(); }

bool NdpUnit::from_mem_full(int bank) { return m_from_mem[bank].full(); }

bool NdpUnit::is_active() {
  bool to_mem_empty = true;
  bool from_mem_empty = true;
  for (int i = 0; i < m_config->get_l2d_num_banks(); i++) {
    to_mem_empty = to_mem_empty && m_to_mem[i].empty();
    from_mem_empty = from_mem_empty && m_from_mem[i].empty();
  }
  bool sub_core_active = false;
  for (int i = 0; i < m_num_sub_core; i++) {
    sub_core_active = sub_core_active || m_sub_core_units[i]->is_active();
  }
  return !m_finished_contexts.empty() || !to_mem_empty || !from_mem_empty ||
        sub_core_active ||
        m_uthread_generator->is_active() || 
        m_ldst_unit->active() ||
        !m_inst_req.empty() ||
        !m_tlb_req.empty() ||
        !m_matched_requests.empty();
}

bool NdpUnit::is_launch_active(int launch_id) {
  return m_uthread_generator->is_launch_active(launch_id);
}

bool NdpUnit::can_register() { return m_instruction_buffer->can_register(); }

void NdpUnit::register_ndp_kernel(NdpKernel* ndp_kernel) {
  assert(can_register());
  m_instruction_buffer->register_kernel(ndp_kernel);
  m_uthread_generator->register_kernel(ndp_kernel);
}

void NdpUnit::launch_ndp_kernel(KernelLaunchInfo info) {
  m_uthread_generator->launch(info);
  m_ldst_unit->set_l1d_size(m_uthread_generator->get_allocated_spad_size());
}

NdpStats NdpUnit::get_stats() {
  m_stats->set_itlb_stats(m_itlb->get_stats());
  m_stats->set_dtlb_stats(m_dtlb->get_stats());
  m_stats->set_icache_stats(m_icache->get_stats());
  m_stats->set_l1d_stats(m_ldst_unit->get_l1d_stats());
  RegisterStats reg_stats;
  for (int i = 0; i < m_num_sub_core; i++) {
    reg_stats += m_sub_core_units[i]->get_register_stats();
  }
  CacheStats l0_icache_stats;
  for (int i = 0; i < m_num_sub_core; i++) {
    l0_icache_stats += m_sub_core_units[i]->get_l0_icache_stats();
  }
  m_stats->set_regsiter_stats(reg_stats);
  m_stats->set_l0_icache_stats(l0_icache_stats);
  return *m_stats;
}

void NdpUnit::print_ndp_stats() {
  if (m_id == 0) {
    for (int i = 0; i < m_num_sub_core; i++) {
      m_sub_core_units[i]->print_sub_core_stats();
    }
  } else {
    for (int i = 0; i < m_num_sub_core; i++) {
      m_sub_core_units[i]->print_sub_core_stats();
    }
  }
}

void NdpUnit::print_uthread_stats() { m_uthread_generator->print_status(); }
#endif
}  // namespace NDPSim
