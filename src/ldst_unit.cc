#ifdef TIMING_SIMULATION
#include "ldst_unit.h"
#include <random>
#include "memory_map.h"
namespace NDPSim {

LDSTUnit::LDSTUnit(M2NDPConfig *config, int ndp_id, 
                std::queue<Context> *finished_contexts,
                std::vector<fifo_pipeline<mem_fetch>> *to_mem,
                std::vector<fifo_pipeline<mem_fetch>> *from_mem, 
                std::vector<fifo_pipeline<int64_t>> *to_reg,
                Tlb *tlb, NdpStats *stats) 
    : m_ndp_id(ndp_id),
      m_cycle(0),
      m_config(config),
      m_finished_contexts(finished_contexts),
      m_to_mem(to_mem),
      m_from_mem(from_mem),
      m_to_reg(to_reg),
      m_tlb(tlb),
      m_stats(stats),
      m_pending_write_count(0),
      m_tlb_waiting_queue(0) {

  for (int i = 0; i < config->get_num_ldst_units(); i++) {
    m_ldst_units.push_back(
        ExecutionDelayQueue("ldst_unit_" + std::to_string(i)));
  }
  for (int i = 0; i < config->get_num_spad_units(); i++) {
    m_spad_units.push_back(
        ExecutionDelayQueue("spad_unit_" + std::to_string(i)));
  }
  
  for (int i = 0; i < config->get_num_v_ldst_units(); i++) {
    m_v_ldst_units.push_back(
        ExecutionDelayQueue("v_ldst_unit_" + std::to_string(i)));
  }
  for (int i = 0; i < config->get_num_v_spad_units(); i++) {
    m_v_spad_units.push_back(
        ExecutionDelayQueue("v_spad_unit_" + std::to_string(i)));
  }

  m_l1_hit_latency = config->get_skip_l1d() ? 0 : config->get_l1d_hit_latency();
  m_max_l1_latency_queue_size =
      m_l1_hit_latency + PACKET_SIZE * MAX_VLMUL;  // Can fill multiple pipeline reqs at once
  m_l1_latency_queue.resize(config->get_l1d_num_banks(),
     DelayQueue<mem_fetch*>("l1_latency_queue", true, m_max_l1_latency_queue_size));
  m_l1_latency_pipeline_max_index.resize(config->get_l1d_num_banks());
  if(!config->get_skip_l1d()) {
    m_l1d_config.init(config->get_l1d_config(), config);
    m_to_tlb_queue = fifo_pipeline<mem_fetch>("to_tlb", 0, config->get_request_queue_size());
    m_l1d_cache = new DataCache(std::string("l1d_cache"), m_l1d_config,
                                m_ndp_id, 0, &m_to_tlb_queue, true);
  }
}

void LDSTUnit::set_l1d_size(int smem_size) {
  int max_sets = m_l1d_config.get_max_sets();
  if(m_config->get_adaptive_l1d()) {
    assert(m_config->get_spad_size() == m_l1d_config.get_origin_size());
    int smem_option_size = m_config->smem_option_size(smem_size);
    printf("smem_option_size %d %d\n", smem_size, smem_option_size);
    float origin_size = m_l1d_config.get_origin_size();
    float ratio = (origin_size- smem_option_size) / origin_size;
    int new_sets = (max_sets * ratio + 0.5);
    m_l1d_config.set_sets(new_sets);
    spdlog::info("NDP {}: L1d size: {} KB -> {} KB, sets: {} -> {}",
     m_ndp_id, origin_size / 1024, m_l1d_config.get_total_size_in_kb(), max_sets, new_sets);
  }
}

bool LDSTUnit::active() {
  return m_pending_write_count > 0 ||
         !check_unit_finished();
}

void LDSTUnit::cycle() {
  m_cycle++;
  std::deque<Status> status_list;

  for (auto &ldst_unit : m_ldst_units) {
    ldst_unit.cycle();
    process_ldst_inst(ldst_unit);
  }
  for (auto &spad_unit : m_spad_units) {
    spad_unit.cycle();
    process_spad_inst(spad_unit);
  }
  
  for (auto &v_ldst_unit : m_v_ldst_units) {
    v_ldst_unit.cycle();
    process_ldst_inst(v_ldst_unit);
  }
  for (auto &v_spad_unit : m_v_spad_units) {
    v_spad_unit.cycle();
    process_spad_inst(v_spad_unit);
  }
  
  m_spad_delay_queue.cycle();

  // Tlb handle
  bool queue_full = false;
  while (m_tlb->data_ready() && !queue_full) {
    for (int bank = 0; bank < m_config->get_l2d_num_banks(); bank++) {
      mem_fetch *mf = m_tlb->get_data();
      assert(mf);
      assert(mf->get_timestamp() <= m_config->get_ndp_cycle());
      int target_bank = m_config->get_bank_index(mf->get_addr()) % m_config->get_l2d_num_banks();
      if (bank != target_bank) continue;
      if (!(*m_to_mem)[bank].full()) {
        (*m_to_mem)[bank].push(mf);
        m_tlb->pop_data();
        m_stats->add_memory_status(TO_MEM_PUSH_MY_CHANNEL);
      } else {
        status_list.push_back(TO_MEM_FULL);
        queue_full = true;
      }
      break;
    }
  }
  m_stats->add_status_list(status_list);

  // L1d handle
  if(!m_config->get_skip_l1d()) {
    m_l1d_cache->cycle();
    l1_latency_queue_cycle();
    process_l1d_access();
  }
}

bool LDSTUnit::full() {
  bool full = true;
  if (!check_units_full(m_ldst_units)) return false;
  if (!check_units_full(m_spad_units)) return false;
  if (!check_units_full(m_v_ldst_units)) return false;
  if (!check_units_full(m_v_spad_units)) return false;
  return true;
}

bool LDSTUnit::check_units_full(std::vector<ExecutionDelayQueue> &units) {
  bool full = true;
  for (auto &unit : units) {
    full = full && unit.full();
    if (!full) return false;
  }
  return full;
}

Status LDSTUnit::issue_ldst_unit(NdpInstruction inst, Context context, bool spad_op) {
  bool address_unit_pop = false;
  int request_size = inst.addr_set.size() * context.csr->vtype_vlmul;
  AluOpType alu_op_type = LDST_OP;
  int inst_latency = m_config->get_ndp_op_latencies(alu_op_type);
  int inst_issue_interval =
      m_config->get_ndp_op_initialiation_interval(alu_op_type);
  for (auto &unit : get_ldst_unit(inst, spad_op)) {
    if (!unit.full()) {
      unit.push({inst, context}, inst_latency, inst_issue_interval);
      address_unit_pop = true;
      break;
    }
  }
  if (address_unit_pop) {
    return ISSUE_SUCCESS;
  } else {
    if (spad_op) return SPAD_DELAY_QUEUE_FULL;
    else
      return LDST_UNIT_FULL;
  }
}

void LDSTUnit::process_ldst_inst(ExecutionDelayQueue &ldst_unit) {
  if (ldst_unit.empty()) return;
  NdpInstruction inst = ldst_unit.top().first;
  Context context = ldst_unit.top().second;
  if(check_l1_hit_pipeline_full(inst)) return;

  if (inst.CheckLoadOp()) {
    // Load Global Memory
    ReadRequestInfo *info = NULL;
    for (auto iter = inst.addr_set.begin(); iter != inst.addr_set.end();
          ++iter) {
      uint64_t addr = (*iter);
      mem_fetch *mf =
          new mem_fetch(addr, GLOBAL_ACC_R, READ_REQUEST,
                        m_config->get_packet_size(), CXL_OVERHEAD, m_cycle);
      mf->set_from_ndp(true);
      mf->set_ndp_id(m_ndp_id);
      mf->set_channel(m_config->get_channel_index(addr));
      if(m_config->is_bi_enabled()) {
        if( rand() % 100 < (m_config->get_bi_rate()*100) 
          && !m_config->is_handled_bi_addr(addr) && !m_config->is_bi_inprogress(addr)) {
          mf->set_bi();
          m_config->insert_handled_bi_addr(addr);
          m_config->insert_inprogress_bi_addr(addr);
          m_config->m_inprogress_mfs[(uint64_t) mf->get_addr()/m_config->get_bi_unit_size()] =  mf;
          mf->set_data_size(m_config->get_bi_unit_size());
          mf->current_state = "BI_LOAD";
        }
        else if (!m_config->is_handled_bi_addr(addr) && !m_config->is_bi_inprogress(addr)) {
          m_config->insert_handled_bi_addr(addr);
          mf->current_state = "NOT BI";
        } else if (m_config->is_bi_inprogress(addr)) {
          mf->current_state = "BI_INPROGRESS";
          assert(m_config->m_inprogress_mfs.find((uint64_t) mf->get_addr()/m_config->get_bi_unit_size())->second->get_ndp_id() == m_ndp_id);
        }
      }
      if (info == NULL) {
        info = new ReadRequestInfo();
        info->key_mf = mf;
        info->count = 1;
      } else {
        info->count += 1;
      }
      mf->set_read_request(info);
      int bank_id = m_config->get_bank_index(addr) % m_config->get_l1d_num_banks(); 
      if (inst.CheckAmoOp())  // Global Atomic opeation performs on L2 cache
      {
        mf->set_atomic(true);
        m_l1_latency_queue[bank_id].push(mf, 0);
      }
      else {
        m_l1_latency_queue[bank_id].push(mf, m_l1_hit_latency);
      }
    }
    m_pending_ldst[info->key_mf] = inst;
    m_pending_ldst_context[info->key_mf] = context;
  } else {
    // Store Global Memory
    for (auto iter = inst.addr_set.begin(); iter != inst.addr_set.end();
          ++iter) {
      uint64_t addr = (*iter);
      mem_fetch *mf =
          new mem_fetch(addr, GLOBAL_ACC_W, WRITE_REQUEST,
                        m_config->get_packet_size(), CXL_OVERHEAD, m_cycle);
      mf->set_from_ndp(true);
      mf->set_ndp_id(m_ndp_id);
      mf->set_channel(m_config->get_channel_index(addr));
      if(m_config->is_bi_enabled()) {
        if(rand() % 100 < (m_config->get_bi_rate()*100) && !m_config->is_handled_bi_addr(addr)
        && !m_config->is_bi_inprogress(addr)) {
          mem_fetch *mf_bi =
          new mem_fetch(addr, GLOBAL_ACC_W, WRITE_REQUEST,
                        m_config->get_packet_size(), CXL_OVERHEAD, m_cycle);
          mf_bi->set_from_ndp(true);
          mf_bi->set_ndp_id(m_ndp_id);
          mf->set_channel(m_config->get_channel_index(addr));
          mf_bi->set_bi();
          m_config->insert_handled_bi_addr(addr);
          mf_bi->set_data_size(CXL_OVERHEAD); // write invalidation command
        }
      }
      int bank_id =
          m_config->get_bank_index(addr) % m_config->get_l1d_num_banks();
      m_l1_latency_queue[bank_id].push(mf, m_l1_hit_latency);
      m_pending_write_count++;
    }
  }
  ldst_unit.pop();
  if (context.last_inst && !inst.CheckLoadOp()) {
    m_finished_contexts->push(context);
  }
}

void LDSTUnit::process_spad_inst(ExecutionDelayQueue &spad_unit) {
  // handle from spad
  if (!m_spad_delay_queue.empty()) {
    NdpInstruction inst = m_spad_delay_queue.top().first;
    Context context = m_spad_delay_queue.top().second;
    if (inst.CheckLoadOp()) {
      int sub_core_id = context.sub_core_id;
      assert(!((*m_to_reg)[sub_core_id].full()));
      int64_t* dest = new int64_t();
      *dest = inst.dest;
      (*m_to_reg)[sub_core_id].push(dest);
    }
    m_spad_delay_queue.pop();
  }
  if (spad_unit.empty()) return;
  NdpInstruction inst = spad_unit.top().first;
  Context context = spad_unit.top().second;
  bool spad_pop_success = false;
  uint64_t first_addr = *inst.addr_set.begin();
  if (MemoryMap::CheckScratchpad(first_addr)) {
    // Scratchpad memory access
    if (!m_spad_delay_queue.full()) {
      uint32_t latency = m_config->get_spad_latency();
      m_spad_delay_queue.push({inst, context}, latency, 1);
      spad_pop_success = true;
      if (inst.CheckAmoOp()) {
        m_stats->add_memory_status(SCRATCHPAD_WRITE, inst.addr_set.size());
      } else if (inst.CheckLoadOp())
        m_stats->add_memory_status(SCRATCHPAD_READ, inst.addr_set.size());
      else if (inst.CheckStoreOp())
        m_stats->add_memory_status(SCRATCHPAD_WRITE, inst.addr_set.size());
      else
        assert(0);
    }
  } else {
    assert(0);
  }
  if (spad_pop_success) {
    spad_unit.pop();
    if (context.last_inst) {
      m_finished_contexts->push(context);
    }
  }
}

void LDSTUnit::handle_from_mem(int bank) {
  if (!(*m_from_mem)[bank].empty()) {
    mem_fetch *mf = (*m_from_mem)[bank].top();
    if(!m_config->get_skip_l1d() && !mf->is_atomic()) {
      if(m_l1d_cache->waiting_for_fill(mf)) {
        m_l1d_cache->fill(mf, m_config->get_ndp_cycle());
      }
      else if(mf->get_type() == WRITE_ACK ) {
        if(mf->get_access_type() == L1_CACHE_WB) {
          delete mf;
        }
        else {
          assert(mf->get_access_type() == GLOBAL_ACC_W);
          handle_response(mf);
        }
      } else {
        handle_response(mf);
      }
    }
    else {
      handle_response(mf);
    } 
    (*m_from_mem)[bank].pop();
  }
}


bool LDSTUnit::check_unit_finished() {
  bool finish = true;
  if (!check_units_finished(m_ldst_units)) return false;
  if (!check_units_finished(m_spad_units)) return false;
  if (!check_units_finished(m_v_ldst_units)) return false;
  if (!check_units_finished(m_v_spad_units)) return false;
  return finish;
}

bool LDSTUnit::check_units_finished(
    std::vector<ExecutionDelayQueue> &units) {
  bool finish = true;
  for (auto &unit : units) {
    if (!unit.queue_empty()) finish = false;
  }
  return finish;
}

std::vector<ExecutionDelayQueue> &LDSTUnit::get_ldst_unit(
    NdpInstruction inst, bool spad) {
  if (spad) {
    if (inst.CheckVectorOp()) return m_v_spad_units;
    return m_spad_units;
  } else {
    if (inst.CheckVectorOp()) return m_v_ldst_units;
    return m_ldst_units;
  }
}

void LDSTUnit::dump_current_state() {
  spdlog::error("NDP {}: cycle {} ", m_ndp_id, m_cycle);
  for (auto &[key, val] : m_pending_ldst) {
    spdlog::error("LDST: {} {} {} {:x}", (uint64_t)key, val.sInst, val.dest,
                  *val.addr_set.begin());
  }
}

CacheStats LDSTUnit::get_l1d_stats() {
  if(!m_config->get_skip_l1d())
    return m_l1d_cache->get_stats();
  else
    return CacheStats();
}

bool LDSTUnit::check_l1_hit_pipeline_full(NdpInstruction &inst) {
  std::vector<int> bank_count = std::vector<int>(m_config->get_l1d_num_banks(), 0);
  for(auto iter : inst.addr_set) {
    int bank_id = m_config->get_bank_index(iter) % m_config->get_l1d_num_banks();
    bank_count[bank_id]++;
  }
  for(int bank = 0; bank < m_config->get_l1d_num_banks(); bank++) {
    if(m_l1_latency_pipeline_max_index[bank] + bank_count[bank] > m_max_l1_latency_queue_size) {
      return true;
    }
  }
  return false;
}

void LDSTUnit::l1_latency_queue_cycle() {
  for(int bank = 0; bank < m_config->get_l1d_num_banks(); bank++) {
    int max_index = 0;
    m_l1_latency_queue[bank].cycle();
  }
}

void LDSTUnit::process_l1d_access() {
  for(int bank = 0; bank < m_config->get_l1d_num_banks(); bank++) {
    //Core to L1 cache  
    if(!m_l1_latency_queue[bank].empty()) {
      mem_fetch* mf = m_l1_latency_queue[bank].top();
      // if ((mf->is_atomic() || m_config->get_skip_l1d() ||m_config->is_bi_inprogress(mf->get_addr())) &&
      if ((mf->is_atomic() || m_config->get_skip_l1d()) &&
          !m_to_tlb_queue.full()) {
        mf->current_state = "to TLB queue";
        m_to_tlb_queue.push(mf);
        m_l1_latency_queue[bank].pop();
      }
      else {
        int bank_id = m_config->get_bank_index(mf->get_addr()) %
                      m_config->get_l1d_num_banks();
        assert(bank_id == bank);
        assert(mf->get_timestamp() <= m_config->get_ndp_cycle());
        assert(mf->get_access_type() == GLOBAL_ACC_R ||
               mf->get_access_type() == GLOBAL_ACC_W);
        std::deque<CacheEvent> events;
        CacheRequestStatus status =
            m_l1d_cache->access(mf->get_addr(), m_cycle, mf, events);
        bool write_sent = CacheEvent::was_write_sent(events);
        bool read_sent = CacheEvent::was_read_sent(events);
        if (status == HIT) {
          m_l1_latency_queue[bank].pop();
          mf->current_state = "L1 hit\n";
          if (!write_sent) {
            mf->set_reply();
            handle_response(mf);
          }
        } else if (status == RESERVATION_FAIL) {
          mf->current_state = "L1 reservation fail\n";
          assert(!read_sent);
          assert(!write_sent);
        } else {
          assert(status == MISS || status == HIT_RESERVED);
          mf->current_state = "L1 miss\n";
          m_l1_latency_queue[bank].pop();
          if (mf->is_write() &&
              m_l1d_config.get_write_policy() != WRITE_THROUGH &&
              (m_l1d_config.get_write_alloc_policy() == FETCH_ON_WRITE ||
               m_l1d_config.get_write_alloc_policy() == LAZY_FETCH_ON_READ) &&
              CacheEvent::was_write_allocate_sent(events)) {
            mf->set_reply();
            handle_response(mf);
          }
        }
      }
    }
    //L1 cache to Core
    if(m_l1d_cache->access_ready()) {
      mem_fetch* mf = m_l1d_cache->top_next_access();
      assert(mf);
      if (mf->is_request()) mf->set_reply();
      handle_response(mf);
      m_l1d_cache->pop_next_access();
    }
    //L1 cache to TLB
    if(!m_to_tlb_queue.empty()) {
      mem_fetch* mf = m_to_tlb_queue.top();
      assert(mf);
      assert(mf->get_timestamp() <= m_config->get_ndp_cycle());
      if(!m_tlb->full()) {
        m_tlb->access(mf);
        m_to_tlb_queue.pop();
      }
    }
  }
}

void LDSTUnit::handle_response(mem_fetch* mf) {
  // assert(mf->get_from_ndp() && mf->get_ndp_id() == m_ndp_id);
  assert(!mf->is_request());
  if (mf->get_type() == READ_REPLY ||
    (mf->get_type() == WRITE_ACK && mf->is_bi_writeback())) {
    ReadRequestInfo *info = mf->get_readreq_mf_info();
    info->count--;
    if (mf != info->key_mf) delete mf;
    if (info->count != 0) {
      return;
    } else {
      mf = info->key_mf;
      delete info;
    }
    assert(m_pending_ldst.find(mf) != m_pending_ldst.end());
    NdpInstruction inst = m_pending_ldst[mf];
    Context context = m_pending_ldst_context[mf];
    int sub_core_id = context.sub_core_id;
    assert(!((*m_to_reg)[sub_core_id].full()));
    int64_t *dest = new int64_t();
    *dest = inst.dest;
    (*m_to_reg)[sub_core_id].push(dest);
    m_pending_ldst.erase(mf);
    m_pending_ldst_context.erase(mf);
    if (context.last_inst) {
      m_finished_contexts->push(context);
    }
  } else {
    assert(mf->get_type() == WRITE_ACK);
    m_pending_write_count--;
  }
  delete mf;
}


}  // namespace NDPSim

#endif
