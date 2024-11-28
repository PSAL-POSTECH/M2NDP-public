#ifdef TIMING_SIMULATION
#include "tlb.h"

#include "mem_fetch.h"
namespace NDPSim {

Tlb::Tlb(int id, M2NDPConfig* config, std::string tlb_config,
         fifo_pipeline<mem_fetch>* to_mem_queue)
    : m_id(id), m_config(config), m_to_mem_queue(to_mem_queue) {
  m_page_size = config->get_tlb_page_size();
  m_tlb_config.init(tlb_config, config);
  m_tlb = new ReadOnlyCache("tlb", m_tlb_config, id, 0, m_to_mem_queue);
  m_tlb_entry_size = m_config->get_tlb_entry_size();
  m_finished_mf = fifo_pipeline<mem_fetch>("tlb_finished_mf", 0,
                                           m_config->get_request_queue_size());
  m_tlb_request_queue = DelayQueue<mem_fetch*>(
      "tlb_req_queue", true, m_config->get_request_queue_size());
  m_dram_tlb_latency_queue = DelayQueue<mem_fetch*> (
    "dram_tlb_latency_queue", true, m_config->get_request_queue_size());
  m_tlb_hit_latency = m_config->get_tlb_hit_latency();
  m_accessed_tlb_addr = m_config->get_accessed_tlb_addr();
}

void Tlb::set_ideal_tlb() {
  m_ideal_tlb = true;
  m_tlb_hit_latency = 0;
}
bool Tlb::fill_port_free() { return m_tlb->fill_port_free(); }
bool Tlb::data_port_free() { return m_tlb->data_port_free(); }

bool Tlb::full() { return full(0); }

bool Tlb::full(uint64_t mf_size) {
  return m_tlb_request_queue.size() + m_dram_tlb_latency_queue.size() + mf_size >=
         m_config->get_request_queue_size();
}

void Tlb::fill(mem_fetch* mf) {
  mf->current_state = "TLB Fill";
  if (!m_config->is_dram_tlb_miss_handling_enabled()) {
    assert(mf->get_addr() >= DRAM_TLB_BASE);
    m_tlb->fill(mf, m_config->get_ndp_cycle());
    return;
  } else {
    // uint64_t tlb_addr = get_tlb_addr(mf->get_addr());
    uint64_t tlb_addr = mf->get_addr();
    if (m_accessed_tlb_addr->find(tlb_addr) == m_accessed_tlb_addr->end()) {
      // printf("DRAM-TLB miss: %lx\n", tlb_addr);
      if(m_config->get_use_dram_tlb()) 
        m_accessed_tlb_addr->insert(tlb_addr);
      m_dram_tlb_latency_queue.push(
          mf, m_config->get_dram_tlb_miss_handling_latency());
      return;
    }
    else {
      // printf("DRAM-TLB hit: %lx\n", tlb_addr);
      m_tlb->fill(mf, m_config->get_ndp_cycle());
      return;
    }
  }
}

bool Tlb::waiting_for_fill(mem_fetch* mf) {
  return m_tlb->waiting_for_fill(mf);
}

void Tlb::access(mem_fetch* mf) {
  assert(!full());
  m_tlb_request_queue.push(mf, m_tlb_hit_latency);
}

bool Tlb::data_ready() { return !m_finished_mf.empty(); }

mem_fetch* Tlb::get_data() { return m_finished_mf.top(); }

void Tlb::pop_data() { m_finished_mf.pop(); }

void Tlb::cycle() {
  // Tlb cache cycle & decrease latency cycle
  m_tlb->cycle();
  m_tlb_request_queue.cycle();
  m_dram_tlb_latency_queue.cycle();
}

void Tlb::bank_access_cycle() {
  // Handle dram tlb latency cycle
  if(!m_dram_tlb_latency_queue.empty()) {
    mem_fetch* mf = m_dram_tlb_latency_queue.top();
    m_tlb->fill(mf, m_config->get_ndp_cycle());
    m_dram_tlb_latency_queue.pop();
  }
  // Handle tlb bank access cycle
  if (m_tlb->access_ready() && !m_finished_mf.full()) {
    mem_fetch* mf = m_tlb->pop_next_access();
    if (mf->is_request()) mf->set_reply();
    m_finished_mf.push(mf->get_tlb_original_mf());
    delete mf;
  }
  // Handle tlb latency cycle
  if (!m_tlb_request_queue.empty() && data_port_free()) {
    mem_fetch* mf = m_tlb_request_queue.top();
    uint64_t addr = mf->get_addr();
    uint64_t tlb_addr = get_tlb_addr(addr);
    mem_fetch* tlb_mf =
        new mem_fetch(tlb_addr, TLB_ACC_R, READ_REQUEST, m_tlb_entry_size,
                      CXL_OVERHEAD, m_config->get_ndp_cycle());
    tlb_mf->set_from_ndp(true);
    tlb_mf->set_ndp_id(m_id);
    tlb_mf->set_tlb_original_mf(mf);
    tlb_mf->set_channel(m_config->get_channel_index(tlb_addr));
    std::deque<CacheEvent> events;
    CacheRequestStatus stat = MISS;
    if (!m_ideal_tlb)
      stat = m_tlb->access(tlb_addr, m_config->get_ndp_cycle(), tlb_mf, events);
    if ((stat == HIT || m_ideal_tlb) && !m_finished_mf.full()) { // alway hit if ideal tlb
      m_finished_mf.push(mf);
      delete tlb_mf;
      m_tlb_request_queue.pop();
    } else if (stat == HIT && m_finished_mf.full()) {
      delete tlb_mf;
    } else if (stat != RESERVATION_FAIL) {
      m_tlb_request_queue.pop();
    } else if (stat == RESERVATION_FAIL) {
      delete tlb_mf;
    }
  }
}

CacheStats Tlb::get_stats() { return m_tlb->get_stats(); }

uint64_t Tlb::get_tlb_addr(uint64_t addr) {
  return addr / m_page_size * m_tlb_entry_size + DRAM_TLB_BASE;
}
}  // namespace NDPSim
#endif