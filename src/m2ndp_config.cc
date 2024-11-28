#include "m2ndp_config.h"

#include <bitset>

#include "m2ndp_parser.h"
#include "hashing.h"
#define CORE 0x01
#define CACHE 0x02
#define DRAM 0x04
#define ICNT 0x08
#define LINK 0x10
#define BUFFER_DRAM 0x20
#define BUFFER_NDP 0x40
#define MHz *1000000
namespace NDPSim {

M2NDPConfig::M2NDPConfig(std::string config_path, int num_hosts) {
  m_num_hosts = num_hosts;
  m_config_path = config_path;
  M2NDPParser::parse_config(config_path, this);
  #ifdef TIMING_SIMULATION
  m_link_time = 0;
  m_buffer_dram_time = 0;
  m_buffer_ndp_time = 0;
  m_core_time = 0;
  m_dram_time = 0;
  m_cache_time = 0;
  #endif
}

const uint64_t M2NDPConfig::get_channel_index(uint64_t origin_addr) {
  int ch_bits = get_partial_bits(CH, origin_addr);
  int higher_bits = get_higher_bits(CH, origin_addr);
  switch(m_channel_indexing_policy) {
    case 0:
      return ch_bits;
    case 1:
      return ipoly_hash_function(higher_bits, ch_bits, m_num_channels);
    case 2:
      return bitwise_hash_function(higher_bits, ch_bits, m_num_channels);
    default:
      return 0;
  }
}

const uint64_t M2NDPConfig::get_m2ndp_index(uint64_t origin_addr) {
  int dev_bits = get_partial_bits(DEVICE, origin_addr);
  return dev_bits;
}

uint64_t M2NDPConfig::get_partial_bits(memory_decode mask, uint64_t addr) {
  unsigned result = 0x0;
  int mask_bit_count = std::bitset<64>(m_addrdec_mask[mask]).count();
  uint64_t mask_bits = m_addrdec_mask[mask];
  uint64_t addr_bits = addr & mask_bits;

  int id = 0;
  for (int i = 0; i < 64; i++) {
    if (mask_bits & 0x1) {
      result |= ((0x1 & addr_bits) << id);
      id++;
      mask_bits = mask_bits >> 1;
      addr_bits = addr_bits >> 1;
    } else {
      mask_bits = mask_bits >> 1;
      addr_bits = addr_bits >> 1;
    }
  }

  return result;
}

uint64_t M2NDPConfig::get_higher_bits(memory_decode mask, uint64_t addr) {
  unsigned result = 0x0;
  int mask_bit_count = std::bitset<64>(m_addrdec_mask[mask]).count();
  uint64_t mask_bits = ~m_addrdec_mask[mask];
  int max_zero_bit = 0;
  for (int i = 0; i < 64; i++) {
    if((mask_bits & 0x1) == 0) {
      max_zero_bit = i;
    } 
    mask_bits = mask_bits >> 1;
  }
  return addr >> max_zero_bit;
}

#ifdef TIMING_SIMULATION
const bool M2NDPConfig::is_link_cycle() { return m_clock_mask & LINK; }

const bool M2NDPConfig::is_buffer_dram_cycle() {
  return m_clock_mask & BUFFER_DRAM;
}

const bool M2NDPConfig::is_buffer_ndp_cycle() {
  return m_clock_mask & BUFFER_NDP;
}

const bool M2NDPConfig::is_cache_cycle() { return m_clock_mask & CACHE; }

int M2NDPConfig::next_clock_domain() {
  double smallest = min3(m_link_time, m_buffer_dram_time, m_buffer_ndp_time);
  smallest = min3(smallest, m_core_time, m_dram_time);
  smallest = gs_min2(smallest, m_cache_time);
  int mask = 0x00;
  if (m_link_time <= smallest) {
    m_link_time += m_link_period;
    mask |= LINK;
  }
  if (m_buffer_dram_time <= smallest) {
    m_buffer_dram_time += m_buffer_dram_period;
    mask |= BUFFER_DRAM;
  }
  if (m_buffer_ndp_time <= smallest) {
    m_buffer_ndp_time += m_buffer_ndp_period;
    mask |= BUFFER_NDP;
    m_ndp_cycle++;
  }
  if (m_cache_time <= smallest) {
    m_cache_time += m_cache_period;
    mask |= CACHE;
  }
  if (m_core_time <= smallest) {
    m_core_time += m_core_period;
    mask |= CORE;
  }
  if(m_dram_time <= smallest) {
    m_dram_time += m_dram_period;
    mask |= DRAM;
  }
  return mask;
}



const uint64_t M2NDPConfig::get_bank_index(uint64_t origin_addr) {
  return get_partial_bits(BANK, origin_addr);
}


void M2NDPConfig::print_config(FILE *fp) {
  fprintf(fp, "=======M2NDP configuration====\n");
  fprintf(fp, "functional_simulation:\t %d\n", m_functional_sim);
  fprintf(fp, "ndp_configuration_path:\t %s\n", m_config_path.c_str());
  fprintf(fp, "ramulator_config_path:\t %s\n", m_ramulator_config_path.c_str());
  fprintf(fp, "cxl_link_config_path:\t %s\n", m_cxl_link_config_path.c_str());
  fprintf(fp, "local_crossbar_config_path:\t %s\n",
          m_local_crossbar_config_path.c_str());
  fprintf(fp, "enable_sub_core:\t %d\n", m_enable_sub_core);
  fprintf(fp, "num_sub_core:\t %d\n", m_num_sub_core);
  fprintf(fp, "packet_size:\t %d\n", m_packet_size);
  fprintf(fp, "chnnel_interleave_size:\t %d\n", m_channel_interleave_size);
  fprintf(fp, "m2ndp_interleave_size:\t %d\n", m_m2ndp_interleave_size);
  fprintf(fp, "num_hosts:\t %d\n", m_num_hosts);
  fprintf(fp, "m_num_memory_buffers:\t %d\n", m_num_memory_buffers);
  fprintf(fp, "Core period (s):\t %.20lf\n", m_core_period);
  fprintf(fp, "Link period (s):\t %.20lf\n", m_link_period);
  fprintf(fp, "DRAM period (s):\t %.20lf\n", m_dram_period);
  fprintf(fp, "Buffer DRAM period (s):\t %.20lf\n", m_buffer_dram_period);
  fprintf(fp, "Buffer NDP period (s):\t %.20lf\n", m_buffer_ndp_period);
  fprintf(fp, "Cache period (us):\t %f\n", m_cache_period);
  fprintf(fp, "\nNDP_UNIT CONFIGS\n");
  fprintf(fp, "ndp_units_per_buffer:\t %d\n", m_ndp_units_per_buffer);
  fprintf(fp, "num_ndp_units:\t %d\n", m_num_ndp_units);
  fprintf(fp, "simd_width:\t %d\n", m_simd_width);
  fprintf(fp, "num_i_units:\t %d\n", m_num_i_units);
  fprintf(fp, "num_f_units:\t %d\n", m_num_f_units);
  fprintf(fp, "num_sf_units:\t %d\n", m_num_sf_units);
  fprintf(fp, "num_ldst_units:\t %d\n", m_num_ldst_units);
  fprintf(fp, "num_address_units:\t %d\n", m_num_address_units);
  fprintf(fp, "num_spad_units:\t %d\n", m_num_spad_units);
  fprintf(fp, "num_v_i_units:\t %d\n", m_num_v_i_units);
  fprintf(fp, "num_v_f_units:\t %d\n", m_num_v_f_units);
  fprintf(fp, "num_v_sf_units:\t %d\n", m_num_v_sf_units);
  fprintf(fp, "num_v_ldst_units:\t %d\n", m_num_v_ldst_units);
  fprintf(fp, "num_v_address_units:\t %d\n", m_num_v_address_units);
  fprintf(fp, "num_v_spad_units:\t %d\n", m_num_v_spad_units);
  fprintf(fp, "num_x_registers:\t %d\n", m_num_x_registers);
  fprintf(fp, "num_f_registers:\t %d\n", m_num_f_registers);
  fprintf(fp, "num_v_registers:\t %d\n", m_num_v_registers);
  fprintf(fp, "spad_size:\t %d\n", m_spad_size);
  fprintf(fp, "spad_latency:\t %d\n", m_spad_latency);
  fprintf(fp, "skip_l1d:\t %d\n", m_skip_l1d);
  fprintf(fp, "l1d_config:\t %s\n", m_l1d_config_str.c_str());
  fprintf(fp, "l1d_hit_latency:\t %d\n", m_l1d_hit_latency);
  fprintf(fp, "l1d_num_banks:\t %d\n", m_l1d_num_banks);
  fprintf(fp, "l2d_config:\t %s\n", m_l2d_config_str.c_str());
  fprintf(fp, "l2d_hit_latency:\t %d\n", m_l2d_hit_latency);
  fprintf(fp, "l2d_num_banks:\t %d\n", m_l2d_num_banks);
  fprintf(fp, "l0icache_config:\t %s\n", m_l0icache_config_str.c_str());
  fprintf(fp, "l0icache_hit_latency:\t %d\n", m_l0icache_hit_latency);
  fprintf(fp, "l1icache_config:\t %s\n", m_l1icache_config_str.c_str());
  fprintf(fp, "l1icache_hit_latency:\t %d\n", m_l1icache_hit_latency);
  fprintf(fp, "ideal_tlb:\t %d\n", m_ideal_tlb);
  fprintf(fp, "tlb_hit_latency:\t %d\n", m_tlb_hit_latency);
  fprintf(fp, "tlb_page_size:\t %d\n", m_tlb_page_size);
  fprintf(fp, "tlb_entry_size:\t %d\n", m_tlb_entry_size);
  fprintf(fp, "itlb_config:\t %s\n", m_itlb_config.c_str());
  fprintf(fp, "dtlb_config:\t %s\n", m_dtlb_config.c_str());
  fprintf(fp, "request_queue_size:\t %d\n", m_request_queue_size);
  fprintf(fp, "max_dma_list_size:\t %d\n", m_max_dma_list_size);
  fprintf(fp, "uthread_slots:\t %d\n", m_uthread_slots);
  fprintf(fp, "inst_fetch_per_cycle:\t %d\n", m_inst_fetch_per_cycle);
  fprintf(fp, "ndp_op_latencies:\t\n");
  for (int i = 0; i < NUM_ALU_OP_TYPE; i++) {
    fprintf(fp, "\t%s:\t%d\n", AluOpTypeString[i], m_ndp_op_latencies[i]);
  }
  fprintf(fp, "ndp_op_initialiation_interval:\t\n");
  for (int i = 0; i < NUM_ALU_OP_TYPE; i++) {
    fprintf(fp, "\t%s:\t%d\n", AluOpTypeString[i],
            m_ndp_op_initialiation_interval[i]);
  }
  fprintf(fp, "log_interval:\t %d\n", m_log_interval);
  fprintf(fp, "=============================\n");
  spdlog::info("=======M2NDP configuration====");
}

void M2NDPConfig::increase_core_time() {
  m_core_time += m_core_period; 
}

bool M2NDPConfig::is_core_time_minimum() {
  //check current core_time is only smallest among other components' next time
  double next_link_time = m_link_time + m_link_period;
  double next_dram_time = m_dram_time + m_dram_period;
  double next_buffer_dram_time = m_buffer_dram_time + m_buffer_dram_period;
  double next_buffer_ndp_time = m_buffer_ndp_time + m_buffer_ndp_period;
  double next_cache_time = m_cache_time + m_cache_period;
  return ((m_core_time < next_link_time) 
          && (m_core_time < next_buffer_dram_time) 
          && (m_core_time < next_buffer_ndp_time) 
          && (m_core_time < next_cache_time) 
          && (m_core_time < next_dram_time));
}
#endif
}