#ifdef TIMING_SIMULATION
#ifndef NDP_RAMULATOR
#define NDP_RAMULATOR

#include "Ramulator.h"
#include "ndp_unit.h"
#include "m2ndp_config.h"
#include "cache.h"
namespace NDPSim {

class NdpRamulator : public Ramulator {
 public:
  NdpRamulator(unsigned buffer_id, unsigned memory_id,
               MemoryMap *memory_map, unsigned long long* cycles, 
               unsigned num_cores, std::string ramulator_config, 
               M2NDPConfig* m2ndp_config_, std::string out);
  void dram_cycle();
  void cache_cycle();
  bool full(int port_num, int bank = 0);
  void push(mem_fetch* mf, int port_num, int bank = 0);
  mem_fetch* pop(int port_num, int bank = 0);
  mem_fetch* top(int port_num, int bank = 0);
  bool is_active() override;
  void print_all(FILE* fp);
  void print_energy_stats(FILE* fp);
  void finish();
  int get_memory_channel(int ch);

 private:
  struct BIPendingInfo {
    mem_fetch* mf;
    int ch;
    int bank;
  };
  unsigned m_buffer_id;
  unsigned long long count;
  unsigned long long mem_count;
  M2NDPConfig* m_config;
  MemoryMap *m_memory_map;
  CacheConfig m_cache_config;
  
  std::vector<Cache*> m_caches;
  std::vector<NdpStats> m_stats;
  std::vector<std::vector<fifo_pipeline<mem_fetch>>> m_from_crossbar_queue;
  std::vector<std::vector<fifo_pipeline<mem_fetch>>> m_to_crossbar_queue;
  std::vector<std::vector<std::queue<mem_fetch*>>> m_to_crossbar_bi_queue;

  robin_hood::unordered_map<uint64_t, std::vector<BIPendingInfo>> _bi_pending_lds;
  std::vector<fifo_pipeline<mem_fetch>> m_to_mem_queue;
  std::vector<std::vector<DelayQueue<mem_fetch*>>> m_cache_latency_queue;
  void process_memory_requests();
};
}  // namespace NDPSim
#endif
#endif