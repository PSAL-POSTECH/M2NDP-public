#ifdef TIMING_SIMULATION
#include "ndp_ramulator.h"

#include <string>
namespace NDPSim {

NdpRamulator::NdpRamulator(unsigned buffer_id, unsigned memory_id,
                           MemoryMap* memory_map, unsigned long long* cycles,
                           unsigned num_cores, std::string ramulator_config,
                           M2NDPConfig* m2ndp_config_, std::string out)
    : Ramulator(memory_id, num_cores, ramulator_config, out,
                m2ndp_config_->get_log_interval()),
      m_memory_map(memory_map),
      m_buffer_id(buffer_id),
      m_config(m2ndp_config_) {
  m_cache_config.init(m_config->get_l2d_config(), m_config);
  m_caches.resize(m_config->get_num_channels());
  m_stats.resize(m_config->get_num_channels());

  for (int i = 0; i < m_config->get_num_channels(); i++) {
    m_to_mem_queue.push_back(fifo_pipeline<mem_fetch>(
        "to_mem_queue", 0, m_config->get_request_queue_size()));
  }

  m_cache_latency_queue.resize(m_config->get_num_channels());
  m_to_crossbar_queue.resize(m_config->get_num_channels());
  m_from_crossbar_queue.resize(m_config->get_num_channels());
  m_to_crossbar_bi_queue.resize(m_config->get_num_channels());
  for (int i = 0; i < m_config->get_num_channels(); i++) {
    int ch_id = get_memory_channel(i);
    m_stats[i].set_id(ch_id);
    m_caches[i] = new DataCache(std::string("ndp_l2_cache"), m_cache_config,
                                ch_id, 0, &(m_to_mem_queue[i]));
    m_to_crossbar_bi_queue[i].resize(m_config->get_l2d_num_banks());
    for (int bank = 0; bank < m_config->get_l2d_num_banks(); bank++) {
      m_cache_latency_queue[i].push_back(DelayQueue<mem_fetch*>(
          "cache_latency_queue", true, m_config->get_request_queue_size()));
      m_to_crossbar_queue[i].push_back(fifo_pipeline<mem_fetch>(
          "to_crossbar_queue", 0, m_config->get_request_queue_size()));
      m_from_crossbar_queue[i].push_back(fifo_pipeline<mem_fetch>(
          "from_crossbar_queue", 0, m_config->get_request_queue_size()));
    }
  }
  count = 0;
  mem_count = 0;
}

void NdpRamulator::dram_cycle() {
  Ramulator::cycle();
  process_memory_requests();
}

void NdpRamulator::cache_cycle() {
  for (int i = 0; i < m_config->get_num_channels(); i++) {
    for (int bank = 0; bank < m_config->get_l2d_num_banks(); bank++) {

      m_cache_latency_queue[i][bank].cycle();
      // NDP to Cache

      if (!m_from_crossbar_queue[i][bank].empty() &&
          m_caches[i]->data_port_free()) {
        int memory_channel = get_memory_channel(i);
        mem_fetch* req = m_from_crossbar_queue[i][bank].top();
        req->current_state = "NDP ramulator top from crossbar";
        int channel = m_config->get_channel_index(req->get_addr());
        assert(channel == memory_channel);
        std::deque<CacheEvent> events;
        CacheRequestStatus status = m_caches[i]->access(
            req->get_addr(), m_config->get_ndp_cycle(), req, events);
        bool write_sent = CacheEvent::was_write_sent(events);
        bool read_sent = CacheEvent::was_read_sent(events);
        if (status == HIT) {
          if (!write_sent) {
            req->set_reply();
            req->current_state = "L2 hit";
            m_cache_latency_queue[i][bank].push(
                req, m_config->get_l2d_hit_latency());
          }
          m_from_crossbar_queue[i][bank].pop();
        } else if (status != RESERVATION_FAIL) {
          req->current_state = "L2MISS";
          if (req->is_write() &&
              (m_cache_config.get_write_alloc_policy() == FETCH_ON_WRITE ||
               m_cache_config.get_write_alloc_policy() == LAZY_FETCH_ON_READ)) {
            req->set_reply();
            req->current_state = "L2MISS write";
            m_cache_latency_queue[i][bank].push(
                req, m_config->get_l2d_hit_latency());
          }
          m_from_crossbar_queue[i][bank].pop();
        } else {
          // Status Reservation fail
          m_stats[i].add_status(CACHE_RESERVATION_FAIL);
          assert(!write_sent);
          assert(!read_sent);
        }
        m_stats[i].add_status(CACHE_DATA_ACCESS);
      }

      // Cache to NDP
      if (m_caches[i]->access_ready() &&
          !m_cache_latency_queue[i][bank].full()) {
        mem_fetch* req = m_caches[i]->top_next_access();
        req->current_state = "L2 top next access";
        if (req->is_request()) req->set_reply();
        m_cache_latency_queue[i][bank].push(req,
                                            m_config->get_l2d_hit_latency());
        m_caches[i]->pop_next_access();
      }
      if(!m_to_crossbar_bi_queue[i][bank].empty()) {
        if (!m_to_crossbar_queue[i][bank].full()) {
          mem_fetch* req = m_to_crossbar_bi_queue[i][bank].front();
          req->current_state = "bi req to crossbar";
          m_to_crossbar_queue[i][bank].push(req);
          m_to_crossbar_bi_queue[i][bank].pop();
        }
      }
      else if (!m_cache_latency_queue[i][bank].empty()) {
        mem_fetch* req = m_cache_latency_queue[i][bank].top();
        if(m_config->is_bi_enabled() && m_config->is_bi_inprogress(req->get_addr()) 
          && req->is_write() && req->get_data() == (void*)0x77){
          req->current_state = "bi response";
          m_to_crossbar_bi_queue[i][bank].push(req);
          m_cache_latency_queue[i][bank].pop();
          for(auto &bi_req : _bi_pending_lds[req->get_addr()/m_config->get_bi_unit_size()]){
            bi_req.mf->current_state = "bi pending reqs -> to_crossbar_bi_reqs";
            m_to_crossbar_bi_queue[bi_req.ch][bi_req.bank].push(bi_req.mf);
          }
          _bi_pending_lds.erase(req->get_addr()/m_config->get_bi_unit_size());
          m_config->remove_inprogress_bi_addr(req->get_addr());
          m_config->m_inprogress_mfs.erase(req->get_addr()/m_config->get_bi_unit_size());
        }
        else if(m_config->is_bi_enabled() && m_config->is_bi_inprogress(req->get_addr()) && !req->is_write()) {
          req->current_state = "bi pending reqs";
          _bi_pending_lds[req->get_addr()/m_config->get_bi_unit_size()].push_back({req, i, bank});
          m_cache_latency_queue[i][bank].pop();
        }
        else if (!m_to_crossbar_queue[i][bank].full()) {
          req->current_state = "to crossbar";
          m_to_crossbar_queue[i][bank].push(req);
          m_cache_latency_queue[i][bank].pop();
        }
      }
    }
    m_caches[i]->cycle();
  }
  //for (int i = 0; i < m_config->get_ndp_units_per_buffer(); i++) {
  for (int i = 0; i < m_config->get_num_channels(); i++) {
    if (m_config->get_ndp_cycle() % m_config->get_log_interval() == 0) {
      m_caches[i]->print_cache_stats();
    }
  }
}

bool NdpRamulator::full(int port_num, int bank) {
  return m_from_crossbar_queue[port_num][bank].full();
}

void NdpRamulator::push(class mem_fetch* mf, int port_num, int bank) {
  m_from_crossbar_queue[port_num][bank].push(mf);
}

class mem_fetch* NdpRamulator::pop(int port_num, int bank) {
  return m_to_crossbar_queue[port_num][bank].pop();
}

class mem_fetch* NdpRamulator::top(int port_num, int bank) {
  if (m_to_crossbar_queue[port_num][bank].empty()) {
    return NULL;
  }
  return m_to_crossbar_queue[port_num][bank].top();
}

bool NdpRamulator::is_active() {
  for (int i = 0; i < m_config->get_num_channels(); i++) {
    for (int j = 0; j < m_config->get_l2d_num_banks(); j++) {
      if (!m_from_crossbar_queue[i][j].empty() || !m_to_crossbar_queue[i][j].empty() ||
          !m_cache_latency_queue[i][j].queue_empty()) {
        return true;
      }
    }
  }
  return Ramulator::is_active(); 
}

void NdpRamulator::print_all(FILE* fp) {
  CacheStats stats;
  for (int i = 0; i < m_config->get_num_channels(); i++) {
    fprintf(fp, "=======Channel %d D-Cache=======\n", i);
    m_caches[i]->get_stats().print_stats(fp);
    stats += m_caches[i]->get_stats();
  }
  fprintf(fp, "=======Total D-Cache=======\n");
  stats.print_stats(fp, "Total D-Cache");
  Ramulator::print(fp);
}

void NdpRamulator::print_energy_stats(FILE* fp) {
  CacheStats stats;
  for (int i = 0; i < m_config->get_num_channels(); i++) {
    stats += m_caches[i]->get_stats();
  }
  stats.print_energy_stats(fp, "L2D");
  fprintf(fp, "MEM_RD: %d\n", get_num_reads());
  fprintf(fp, "MEM_WR: %d\n", get_num_writes());
  fprintf(fp, "MEM_PRE: %d\n", get_precharges());
  fprintf(fp, "MEM_ACT: %d\n", get_activations());
  fprintf(fp, "MEM_REF: %d\n", get_refreshes());
}

void NdpRamulator::finish() { Ramulator::finish(); }

void NdpRamulator::process_memory_requests() {
  for (int i = 0; i < ramulator_configs.get_channels(); i++) {
    int memory_channel = get_memory_channel(i);
    // From Cache to Ramulator
    if (!m_to_mem_queue[i].empty() && !from_gpgpusim_full(i)) {
      mem_fetch* mf = m_to_mem_queue[i].top();
      assert(memory_channel == m_config->get_channel_index(mf->get_addr()));
      Ramulator::push(mf, i);
      m_to_mem_queue[i].pop();
    }

    if (return_queue_top(i)) {
      // From memory response
      mem_fetch* req = return_queue_top(i);
      assert(memory_channel == m_config->get_channel_index(req->get_addr()));
      if (m_caches[i]->waiting_for_fill(req)) {
        if (m_caches[i]->fill_port_free()) {
          m_caches[i]->fill(req, m_config->get_ndp_cycle());
          return_queue_pop(i);
          m_stats[i].add_status(CACHE_DATA_FILL);
        } else
          m_stats[i].add_status(FILL_PORT_FULL);
      } else {
        if (req->get_access_type() == L2_CACHE_WB &&
            req->get_type() == WRITE_ACK) {
          return_queue_pop(i);
          delete req;
        }
      }
    }
  }
}

int NdpRamulator::get_memory_channel(int ch) {
  int contiguous_ch =
      m_config->get_channel_interleave_size() / m_config->get_packet_size();
  int memory_channel =
      // m_buffer_id * contiguous_ch +
      (ch / contiguous_ch) * contiguous_ch +
      (ch % contiguous_ch);
  return memory_channel;
}
}  // namespace NDPSim
#endif