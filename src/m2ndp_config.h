#ifndef M2NDP_CONFIG_H
#define M2NDP_CONFIG_H
#include <deque>
#include <string>

#include "cache_defs.h"
#include "m2ndp_parser.h"

namespace NDPSim {
enum memory_decode {
  DEVICE,
  RAM,
  BANK_GROUP,
  BANK,
  MEM_ROW,
  COLUMN,
  CH,
  BURST,
  NUM_ADDR_DEC
};

#ifdef TIMING_SIMULATION
class cxl_link;
#define gs_min2(a, b) (((a) < (b)) ? (a) : (b))
#define min3(x, y, z) (((x) < (y) && (x) < (z)) ? (x) : (gs_min2((y), (z))))

class M2NDP;

#endif
class M2NDPConfig {
 public:
  M2NDPConfig(std::string config_path, int num_hosts);
  void set_use_synthetic_memory(bool use_synthetic_memory) {
    m_use_synthetic_memory = use_synthetic_memory;
  }
  const int get_num_x_registers() { return m_num_x_registers; }
  const int get_num_f_registers() { return m_num_f_registers; }
  const int get_num_v_registers() { return m_num_v_registers; }
  const int get_spad_size() { return m_spad_size; }
  const int get_num_ndp_units() { return m_num_ndp_units; }
  const bool get_use_synthetic_memory() { return m_use_synthetic_memory; }
  const uint64_t get_synthetic_base_address() {
    return m_synthetic_base_address;
  }
  const uint64_t get_synthetic_memory_size() { return m_synthetic_memory_size; }
  const uint64_t get_channel_index(uint64_t origin_addr);
  const uint64_t get_m2ndp_index(uint64_t origin_addr);
  int get_matched_unit_id(uint64_t origin_addr) {
    return origin_addr / m_stride_size % m_num_ndp_units;
  }
  const int get_num_channels() { return m_num_channels; }
  const bool is_enable_sub_core() { return m_enable_sub_core; }
  const int get_num_sub_core() { return m_num_sub_core; }

#ifdef TIMING_SIMULATION
  void set_output_file(FILE* fp) { m_output_file = fp; }
  // Getter funtiionc
  FILE* get_output_file() { return m_output_file; }
  const int get_num_hosts() { return m_num_hosts; }
  const int get_num_memories() { return m_num_memories; }
  const int get_num_memory_buffers() { return m_num_memory_buffers; }
  const int get_links_per_host() { return m_links_per_host; }
  const int get_links_per_memory_buffer() { return m_links_per_memory_buffer; } 
  const int get_cxl_link_buffer_size() { return m_cxl_link_buffer_size; }
  const int get_num_m2ndps() { return m_num_memory_buffers; }
  const int get_links_per_m2ndp() { return m_links_per_memory_buffer; }
  const std::string get_cxl_link_config_path() {
    return m_cxl_link_config_path;
  }
  const std::string get_ramulator_config_path() {
    return m_ramulator_config_path;
  }
  const std::string get_local_crossbar_config_path() {
    return m_local_crossbar_config_path;
  }
  const uint64_t get_bank_index(uint64_t origin_addr);
  const unsigned long long get_sim_cycle() { return m_core_cycle; }
  const unsigned long long get_ndp_cycle() { return m_ndp_cycle; }
  const unsigned addr_to_ramulator_addr(uint64_t addr);
  const void cycle() { m_clock_mask = next_clock_domain(); }
  const bool is_link_cycle();
  const bool is_buffer_dram_cycle();
  const bool is_buffer_ndp_cycle();
  const bool is_cache_cycle();
  const uint64_t get_local_addr(uint64_t addr);
  const bool is_in_ndp_addr_space(mem_fetch* mf);
  const void print_core_cycle() { printf("core cycle : %lld\n", m_core_cycle); }
  const bool is_functional_sim() { return m_functional_sim; }
  const bool is_always_crossbar() { return m_always_crossbar; }
  const int get_channel_interleave_size() { return m_channel_interleave_size; }
  const int get_m2ndp_interleave_size() { return m_m2ndp_interleave_size; }
  const int get_packet_size() { return m_packet_size; }
  const int get_simd_width() { return m_simd_width; }
  const int get_ndp_units_per_buffer() { return m_ndp_units_per_buffer; }
  const int get_request_queue_size() { return m_request_queue_size; }
  bool get_skip_l1d() { return m_skip_l1d; }
  const std::string get_l1d_config() { return m_l1d_config_str; }
  const int get_l1d_hit_latency() { return m_l1d_hit_latency; }
  const int get_l1d_num_banks() { return m_l1d_num_banks; }
  const std::string get_l2d_config() { return m_l2d_config_str; }
  const int get_l2d_hit_latency() { return m_l2d_hit_latency; }
  const int get_l2d_num_banks() { return m_l2d_num_banks; }
  const std::string get_l0icache_config() { return m_l0icache_config_str; }
  const std::string get_l1icache_config() { return m_l1icache_config_str; }
  const int get_l0icache_hit_latency() { return m_l0icache_hit_latency; }
  const int get_l1icache_hit_latency() { return m_l1icache_hit_latency; }
  const int get_spad_latency() { return m_spad_latency; }
  const bool get_use_dram_tlb() { return m_use_dram_tlb; }
  const bool get_ideal_tlb() { return m_ideal_tlb; }
  const bool get_ideal_icache() { return m_ideal_icache; }
  const int get_tlb_page_size() { return m_tlb_page_size; }
  const int get_tlb_entry_size() { return m_tlb_entry_size; }
  const std::string get_dtlb_config() { return m_dtlb_config; }
  const std::string get_itlb_config() { return m_itlb_config; }
  const int get_tlb_hit_latency() { return m_tlb_hit_latency; }
  const int get_max_kernel_register() { return m_max_kernel_register; }
  const int get_max_kernel_launch() { return m_max_kernel_launch; }
  const int get_max_dma_list_size() { return m_max_dma_list_size; }
  const int get_inst_buffer_size() { return m_inst_buffer_size; }
  const int get_uthread_slots() { return m_uthread_slots; }
  const int get_max_inst_issue() { return m_max_inst_issue; }
  const int get_inst_fetch_per_cycle() { return m_inst_fetch_per_cycle; }
  const int get_num_i_units() { return m_num_i_units; }
  const int get_num_f_units() { return m_num_f_units; }
  const int get_num_sf_units() { return m_num_sf_units; }
  const int get_num_ldst_units() { return m_num_ldst_units; }
  const int get_num_address_units() { return m_num_address_units; }
  const int get_num_spad_units() { return m_num_spad_units; }
  const int get_num_v_i_units() { return m_num_v_i_units; }
  const int get_num_v_f_units() { return m_num_v_f_units; }
  const int get_num_v_sf_units() { return m_num_v_sf_units; }
  const int get_num_v_ldst_units() { return m_num_v_ldst_units; }
  const int get_num_v_address_units() { return m_num_v_address_units; }
  const int get_num_v_spad_units() { return m_num_v_spad_units; }
  const int get_ndp_op_latencies(int op_type) {
    return m_ndp_op_latencies[op_type];
  }
  const int get_ndp_op_initialiation_interval(int op_type) {
    return m_ndp_op_initialiation_interval[op_type];
  }
  const bool get_use_unified_alu() { return m_use_unified_alu; }
  const int get_log_interval() { return m_log_interval; }
  void print_config(FILE* fp);
  bool get_debug_mode() { return false; }
  const double get_link_period() { return m_link_period; }
  const double get_buffer_dram_period() { return m_buffer_dram_period; }
  const double get_buffer_ndp_period() { return m_buffer_ndp_period; }
  const double get_cache_period() { return m_cache_period; }
  const double get_core_period() { return m_core_period; }
  void increase_core_time();
  void increase_core_cycle() { m_core_cycle++; }
  bool is_core_time_minimum();
  bool is_dram_tlb_miss_handling_enabled() {
    return m_enable_dram_tlb_miss_handling;
  }
  int get_dram_tlb_miss_handling_latency() {
    return m_dram_tlb_miss_handling_latency;
  }
  std::set<uint64_t>* get_accessed_tlb_addr() { return &m_accessed_tlb_addr; }
  bool is_bi_enabled() { return enable_random_bi; }
  float get_bi_rate() { return random_bi_rate; }
  bool is_handled_bi_addr(uint64_t addr) {
    return m_handled_bi_addr.find(addr/bi_addr_unit_size) != m_handled_bi_addr.end();
  }
  bool is_bi_inprogress(uint64_t addr ) {
    return m_inprogress_bi_addr.find(addr/bi_addr_unit_size) != m_inprogress_bi_addr.end();
  }
  void insert_handled_bi_addr(uint64_t addr) {
    m_handled_bi_addr.insert(addr/bi_addr_unit_size);
  }
  void insert_inprogress_bi_addr(uint64_t addr) {
    m_inprogress_bi_addr.insert(addr/bi_addr_unit_size);
  }
  void remove_inprogress_bi_addr(uint64_t addr) {
    m_inprogress_bi_addr.erase(addr/bi_addr_unit_size);
  }
  int get_bi_host_latency() { return bi_host_latency; }
  int get_bi_unit_size() { return bi_addr_unit_size; }
  void set_functional_sim(bool functional_sim) { m_functional_sim = functional_sim; }
  bool get_adaptive_l1d() { return m_adaptive_l1d; }
  int smem_option_size(int origin_size) {
    for (int i = 0; i < shmem_option.size(); i++) {
      if (shmem_option[i]*1024 >= origin_size) {
        return shmem_option[i]*1024;
      }
    }
    return shmem_option[shmem_option.size() - 1]*1024;
  }
  bool is_coarse_grained() { return m_coarse_grained; }
  bool is_sc_work() { return m_sc_work; }
  bool is_subcore_coarse_grained() { return m_sub_core_coarse_grained; }
  bool is_all_vector() { return m_all_vector; }
  bool is_sc_work_addr() {return m_sc_work_addr;}
  std::map<uint64_t,mem_fetch*> m_inprogress_mfs;
#endif

 private:
  std::string m_config_dir;
  std::string m_config_path;
  // NDP config
  int m_packet_size;
  int m_simd_width;
  int m_num_ndp_units;
  int m_num_hosts;

  // Register file configuration
  int m_num_x_registers;
  int m_num_f_registers;
  int m_num_v_registers;

  // Scratch pad memory configuration
  int m_spad_size;
  bool m_use_synthetic_memory = 0;
  uint64_t m_synthetic_base_address;
  uint64_t m_synthetic_memory_size;

  uint64_t m_addrdec_mask[memory_decode::NUM_ADDR_DEC];
  int m_channel_indexing_policy = 0;
  int m_num_channels;

  bool m_enable_sub_core = true;
  int m_num_sub_core = 4;
  int m_stride_size = 256;
#ifdef TIMING_SIMULATION
  bool m_functional_sim;
  bool m_always_crossbar = false;  // if true, mem_access always use crossbar
  int m_channel_interleave_size = 32;
  int m_m2ndp_interleave_size;
  FILE* m_output_file;
  unsigned long long m_core_cycle = 0;
  unsigned long long m_ndp_cycle = 0;
  unsigned m_clock_mask;
  int m_num_memories;

  int m_num_memory_buffers;
  int m_links_per_host;
  int m_links_per_memory_buffer;
  int m_ndp_units_per_buffer;
  int m_cxl_link_buffer_size;
  int m_request_queue_size;
  double m_core_period, m_dram_period;
  double m_link_period, m_buffer_dram_period, m_buffer_ndp_period,
      m_cache_period;
  double m_core_time, m_dram_time;
  double m_link_time, m_buffer_dram_time, m_buffer_ndp_time, m_cache_time;

  // Cache config path
  bool m_skip_l1d = false;
  bool m_adaptive_l1d = false;
  std::string m_l1d_config_str;
  int m_l1d_hit_latency;
  int m_l1d_num_banks = 1;
  std::vector<int> shmem_option = {0, 8, 16, 32, 64, 128};
  std::string m_l2d_config_str;
  int m_l2d_hit_latency;
  int m_l2d_num_banks = 1;
  
  std::string m_l0icache_config_str;
  int m_l0icache_hit_latency = 1;
  std::string m_l1icache_config_str;
  int m_l1icache_hit_latency = 1;
  int m_tlb_hit_latency = 1;
  bool m_ideal_tlb = false;
  bool m_ideal_icache = false;
  // Tlb configuration
  int m_tlb_page_size;
  int m_tlb_entry_size;
  std::string m_dtlb_config;
  std::string m_itlb_config;
  // Spad configuration
  int m_spad_latency = 1;

  int m_max_dma_list_size;

  int m_max_kernel_register;
  int m_max_kernel_launch;
  int m_inst_buffer_size;

  // Execution unit configuration
  int m_uthread_slots;
  int m_max_inst_issue = 0;
  int m_inst_fetch_per_cycle;
  int m_num_i_units;
  int m_num_f_units;
  int m_num_sf_units;
  int m_num_ldst_units;
  int m_num_address_units;
  int m_num_spad_units;
  int m_num_v_i_units;
  int m_num_v_f_units;
  int m_num_v_sf_units;
  int m_num_v_ldst_units;
  int m_num_v_address_units;
  int m_num_v_spad_units;
  int m_ndp_op_latencies[NUM_ALU_OP_TYPE];
  int m_ndp_op_initialiation_interval[NUM_ALU_OP_TYPE];
  bool m_use_unified_alu;
  int m_log_interval;
  int next_clock_domain();
  // Memory management configure
  std::string m_ramulator_config_path;

  // CXL link booksim config
  std::string m_cxl_link_config_path;
  std::string m_local_crossbar_config_path;

  // Dram tlb miss handling model
  bool m_use_dram_tlb = true;
  bool m_enable_dram_tlb_miss_handling = false;
  int m_dram_tlb_miss_handling_latency = 0;
  bool enable_random_bi = false;
  float random_bi_rate = 0.0;
  int bi_addr_unit_size = 0;
  int bi_host_latency = 0;
  std::set<uint64_t> m_accessed_tlb_addr;
  std::set<uint64_t> m_handled_bi_addr;
  std::set<uint64_t> m_inprogress_bi_addr;

  bool m_coarse_grained = false;
  bool m_sc_work = false;
  bool m_sc_work_addr = false;
  bool m_sub_core_coarse_grained = false;
  bool m_all_vector = false;
  friend M2NDP;
#endif
  uint64_t get_partial_bits(memory_decode mask, uint64_t addr);
  uint64_t get_extracted_address(memory_decode mask, uint64_t addr);
  uint64_t get_higher_bits(memory_decode mask, uint64_t addr);
  uint64_t insert_to_address(memory_decode mask, uint64_t addr, uint64_t bits);
  friend M2NDPParser;
};
}  // namespace NDPSim
#endif