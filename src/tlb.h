#ifdef TIMING_SIMULATION
#ifndef TLB_H
#define TLB_H
#include "cache.h"
#include "common.h"
#include "cache.h"
#include "delay_queue.h"
#include "m2ndp_config.h"
namespace NDPSim {
class Tlb {
 public:
  Tlb(int id, M2NDPConfig *config, std::string tlb_config,fifo_pipeline<mem_fetch> *to_mem_queue);
  void set_ideal_tlb();
  bool fill_port_free();
  bool data_port_free();
  bool full();
  bool full(uint64_t mf_size);
  void fill(mem_fetch *mf);
  bool waiting_for_fill(mem_fetch *mf);
  void access(mem_fetch* mf);
  bool data_ready();
  mem_fetch* get_data();
  void pop_data();
  void cycle();
  void bank_access_cycle();
  CacheStats get_stats();
 private:
  uint64_t get_tlb_addr(uint64_t addr);
 private:
  int m_id;
  int m_page_size;
  int m_tlb_entry_size;
  int m_tlb_hit_latency;
  bool m_ideal_tlb = false;
  M2NDPConfig *m_config;
  fifo_pipeline<mem_fetch> *m_to_mem_queue;
  fifo_pipeline<mem_fetch> m_finished_mf;
  DelayQueue<mem_fetch*> m_tlb_request_queue;
  DelayQueue<mem_fetch*> m_dram_tlb_latency_queue;
  std::set<uint64_t> *m_accessed_tlb_addr;
  CacheConfig m_tlb_config;
  Cache *m_tlb;
};
}  // namespace NDPSim
#endif
#endif