#ifdef TIMING_SIMULATION
#include "instruction_queue.h"
#define INITIALIZER_DEPENDENCY -1215752192
namespace NDPSim {

InstructionQueue::InstructionQueue(M2NDPConfig *config, int id, int sub_core_id, Cache *icache) {
  m_id = id;
  m_sub_core_id = sub_core_id;
  m_inst_col_id = 0;
  m_num_columns = 0;
  m_config = config;
  m_inst_columns_iter = m_inst_columns.begin();
  m_icache = icache;
  m_icache_queue = DelayQueue<mem_fetch *>("l0_icache_queue", true,
                                           m_config->get_request_queue_size());
  m_l0_icache_hit_latency = m_config->get_l0icache_hit_latency();
}

void InstructionQueue::set_ideal_icache() { 
  m_ideal_icache = true;
  m_l0_icache_hit_latency = 0;
}

int InstructionQueue::get_num_columns() { return m_num_columns; }

bool InstructionQueue::can_push(InstColumn *inst_column) {
  if( m_num_columns >= m_config->get_uthread_slots()) return false;
  DependencyKey key = get_dependency_key(inst_column);
  if(m_dependency_info.find(key) != m_dependency_info.end()) {
    if(m_dependency_info[key].current_type == inst_column->type &&
        m_dependency_info[key].current_kernel_body_id == inst_column->req->kernel_body_id) {
      return true;
    }
    else {
      return false;
    }
  }
  else {
    // If a dependency with the same key is not found, return true
    return true;
  }
}

void InstructionQueue::push(InstColumn *inst_column) {
  assert(can_push(inst_column));
  if (inst_column->insts.empty()) assert(0 && "This should not happen");
  inst_column->valid = true;
  inst_column->id = m_inst_col_id++;
  m_num_columns++;
  m_inst_columns.push_back(inst_column);
  m_inst_columns_iter = m_inst_columns.begin();
  DependencyKey key = get_dependency_key(inst_column);
  if (m_dependency_info.find(key) == m_dependency_info.end()) {
    m_dependency_info[key].current_type = inst_column->type;
    m_dependency_info[key].current_kernel_body_id = inst_column->req->kernel_body_id;
    m_dependency_info[key].count = 1;
  } else {
    assert(m_dependency_info[key].current_type == inst_column->type &&
           m_dependency_info[key].current_kernel_body_id == inst_column->req->kernel_body_id);
    m_dependency_info[key].count++;
  }
}

NdpInstruction InstructionQueue::fetch_instruction() {
  assert(can_fetch());
  return *(*m_inst_columns_iter)->current_inst;
}

Context InstructionQueue::fetch_context() {
  assert(can_fetch());
  Context context;
  context.ndp_id = m_id;
  context.sub_core_id = m_sub_core_id;
  context.inst_col_id = (*m_inst_columns_iter)->id;
  context.request_info = (*m_inst_columns_iter)->req;
  context.csr = &(*m_inst_columns_iter)->csr;
  context.loop_map = &(*m_inst_columns_iter)->loop_map;
  context.last_inst = ((*m_inst_columns_iter)->csr.pc ==
                       (*m_inst_columns_iter)->insts.size() - 1) &&
                      !(*m_inst_columns_iter)
                           ->insts[(*m_inst_columns_iter)->csr.pc]
                           .CheckBranchOp();
  context.max_pc = (*m_inst_columns_iter)->insts.size() - 1;
  context.exit = false;
  (*m_inst_columns_iter)->feteched_last = context.last_inst;
  return context;
}

bool InstructionQueue::can_fetch() {
  return inst_fetch_fail() == SUCCESS;
}

Status InstructionQueue::inst_fetch_fail() {
  if ((*m_inst_columns_iter)->csr.block)
    return INSTRUCTION_BRANCH_WAIT;  // Branch
  if ((*m_inst_columns_iter)->pending ||
      (*m_inst_columns_iter)->current_inst == NULL)  // icache
    return INSTRUCTION_CACHE_WAIT;
  return SUCCESS;
}

void InstructionQueue::cycle() {
  // Fetch instruction from instruction cache
  m_icache_queue.cycle();
  int num_inst_fetch =
      std::min(m_config->get_inst_fetch_per_cycle(), m_num_columns);
  for (int i = 0; i < num_inst_fetch; i++) {
    if (m_icache_queue.empty()) break;
    mem_fetch *mf = m_icache_queue.top();

    std::deque<CacheEvent> events;
    CacheRequestStatus status = MISS;
    if(!m_ideal_icache)
        status = m_icache->access(mf->get_addr(), m_config->get_ndp_cycle(), mf, events);
    if (status == HIT || m_ideal_icache) {
      InstColumn *col = mf->get_inst_column();
      col->current_inst = &col->insts.at(col->csr.pc);
      col->pending = false;
      m_icache_queue.pop();
    } else if (status != RESERVATION_FAIL) {
      m_icache_queue.pop();
    }
  }
  for (int i = 0; i < num_inst_fetch; i++) {
    if (m_icache->access_ready()) {
      mem_fetch *mf = m_icache->pop_next_access();
      if (mf->is_request()) mf->set_reply();
      InstColumn *col = mf->get_inst_column();
      assert(col->pending);
      col->pending = false;
      col->current_inst = &col->insts.at(col->csr.pc);
      delete mf;
    } else
      break;
  }
  auto iter = m_inst_columns_iter;
  for (int i = 0; i < num_inst_fetch; i++) {
    if ((*iter)->current_inst == NULL && !(*iter)->csr.block &&
        !(*iter)->pending && (*iter)->csr.pc < (*iter)->insts.size()) {
      uint64_t inst_addr = (*iter)->base_addr + (*iter)->csr.pc * WORD_SIZE;
      mem_fetch *mf =
          new mem_fetch(inst_addr, INST_ACC_R, READ_REQUEST, WORD_SIZE,
                        CXL_OVERHEAD, m_config->get_ndp_cycle());
      mf->set_src_id((*iter)->id);
      mf->set_from_ndp(true);
      mf->set_ndp_id(m_id);
      mf->set_sub_core_id(m_sub_core_id);
      mf->set_inst_column(*iter);
      mf->set_channel(m_config->get_channel_index(inst_addr));
      m_icache_queue.push(mf, m_l0_icache_hit_latency); //TODO: l0 icache hit latency
      (*iter)->pending = true;
    }
    iter++;
    if (iter == m_inst_columns.end()) {
      iter = m_inst_columns.begin();
    }
  }
}

void InstructionQueue::finish_inst_column(int col_id) {
  auto iter = find_inst_column(col_id);
  InstColumn *col = *iter;
  DependencyKey key = get_dependency_key(col);
  assert(m_dependency_info.find(key) != m_dependency_info.end());
  m_dependency_info[key].count--;
  if (m_dependency_info[key].count == 0) {
    m_dependency_info.erase(key);
  }
  m_num_columns--;
  m_inst_columns.erase(iter);
  m_inst_columns_iter = m_inst_columns.begin();
  delete col->req;
  delete col;
}

bool InstructionQueue::all_empty() { return m_inst_columns.empty(); }

bool InstructionQueue::empty() { return !(*m_inst_columns_iter)->valid; }

void InstructionQueue::next() {
  m_inst_columns_iter++;
  if (m_inst_columns_iter == m_inst_columns.end()) {
    m_inst_columns_iter = m_inst_columns.begin();
  }
}

std::deque<InstColumn *>::iterator InstructionQueue::find_inst_column(
    int inst_col_id) {
  for (std::deque<InstColumn *>::iterator itr = m_inst_columns.begin();
       itr != m_inst_columns.end(); itr++) {
    if ((*itr)->id == inst_col_id) {
      return itr;
    }
  }
  assert(0);
  return m_inst_columns.end();
}

bool InstructionQueue::check_finished(std::deque<InstColumn *>::iterator iter) {
  return (*iter)->insts.size() <= (*iter)->csr.pc;
}
DependencyKey InstructionQueue::get_dependency_key(InstColumn *inst_column) {
  return std::pair<int, int>(inst_column->kernel_id,
                             inst_column->req->launch_id);
}
}  // namespace NDPSim
#endif