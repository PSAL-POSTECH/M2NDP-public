#ifndef MEM_FETCH_H
#ifdef TIMING_SIMULATION
#define MEM_FETCH_H
#define CXL_OVERHEAD 8
#define WRITE_PACKET_SIZE 8
#include <bitset>
#include <cassert>
#include <deque>
#include <vector>
#include <functional>
#include "cache_defs.h"

namespace NDPSim {

typedef uint64_t new_addr_type;
enum mem_access_type {
  GLOBAL_ACC_R,
  GLOBAL_ACC_W,
  INST_ACC_R,
  TLB_ACC_R,
  L1_CACHE_WA, /* Data L1 cache write alloc */
  L2_CACHE_WA, /* Data L2 cache write alloc */
  L1_CACHE_WB, /* Data L1 cache write back */
  L2_CACHE_WB, /* Data L2 cache write back */
  DMA_ALLOC_W,
  HOST_ACC_R,
  HOST_ACC_W,
  NUM_MEM_ACCESS_TYPE
};
static const char* mem_access_type_str[] = {
    "GLOBAL_ACC_R", "GLOBAL_ACC_W",  "INST_ACC_R",    "TLB_ACC_R",
    "L1_CACHE_WA",  "L2_CACHE_WA", "L1_CACHE_WB", "L2_CACHE_WB",
    "DMA_ALLOC_W",  "HOST_ACC_R",    "HOST_ACC_W"};
enum mf_type { READ_REQUEST = 0, WRITE_REQUEST, READ_REPLY, WRITE_ACK };

typedef std::bitset<MAX_MEMORY_ACCESS_SIZE> MemAccessByteMask;

class mem_fetch;
struct InstColumn;
struct ReadRequestInfo {
  mem_fetch* key_mf;
  uint32_t count;
};
class mem_fetch {
 public:
  mem_fetch(new_addr_type addr, mem_access_type acc_type, mf_type type,
            unsigned data_size, unsigned ctrl_size,
            unsigned long long timestamp);
  mem_fetch(new_addr_type addr, mem_access_type acc_type, mf_type type,
            unsigned data_size, unsigned ctrl_size, MemAccessByteMask byte_mask,
            SectorMask sector_mask, unsigned long long timestamp);
  mem_fetch(std::deque<mem_fetch*> mfs);  // for wrapping multiple mfs into one
  new_addr_type get_addr() { return m_addr; }
  ~mem_fetch() { m_valid = false; }
  void set_reply();
  void convert_to_write_request(new_addr_type addr);
  void set_from_dma() { m_from_dma = true; }
  void set_from_dma(bool from_dma) { m_from_dma = from_dma; }
  bool get_from_dma() { return m_from_dma; }
  void set_filtering() { m_filtering = true; }
  void set_from_ndp(bool from_ndp) { m_from_ndp = from_ndp; }
  void set_ndp_id(int ndp_id) { m_ndp_id = ndp_id; }
  void set_sub_core_id(int sub_core_id) {m_sub_core_id = sub_core_id;}
  void set_src_id(unsigned src_id) { m_src_id = src_id; }
  unsigned get_src_id() { return m_src_id; }
  void set_channel(unsigned channel) { m_channel = channel; }
  void set_data(void* data) { m_data = data; }
  void set_data_size(unsigned size) { m_data_size = size; }
  void set_read_request(ReadRequestInfo* info) { 
    m_readreq_mf_info = info;
  }
  ReadRequestInfo *get_readreq_mf_info() {
    return m_readreq_mf_info;
  }
  unsigned get_size() {
    if (m_type == WRITE_REQUEST || m_type == READ_REPLY) {
      return m_data_size + m_ctrl_size;
    } else {
      return m_ctrl_size;
    }
  }
  void set_addr(new_addr_type addr) { m_addr = addr; }
  void set_tlb_original_mf(mem_fetch* mf) {
    m_tlb_request = true;
    m_tlb_original_mf = mf;
  }
  bool is_write();
  bool is_request();
  void set_atomic(bool atomic) { m_atomic = atomic; }
  bool is_atomic() { return m_atomic; }
  unsigned size() { return m_ctrl_size + m_data_size; }
  mem_access_type get_access_type() { return m_mem_access_type; }
  mf_type get_type() { return m_type; }
  bool get_from_ndp() { return m_from_ndp; }
  bool get_to_ndp() { return m_from_ndp; }
  bool get_filtering() { return m_filtering; }
  bool get_not_on_the_fly() { return !m_on_the_fly; }
  unsigned get_data_size() { return m_data_size; }
  unsigned get_ctrl_size() { return m_ctrl_size; }
  unsigned get_request_uid() { return m_request_id; }
  unsigned get_channel() { return m_channel; }
  int get_ndp_id() { return m_ndp_id; }
  int get_sub_core_id() { return m_sub_core_id; }
  bool get_from_m2ndp() { return get_from_ndp(); }
  unsigned get_ramulator_addr() { return (unsigned)m_addr; }
  void* get_data() { return m_data; }
  SectorMask get_access_sector_mask() { return m_sector_mask; }
  MemAccessByteMask get_access_byte_mask() { return m_byte_mask; }
  void set_dirty_mask(SectorMask dirty_mask) { m_dirty_mask = dirty_mask; }
  SectorMask get_dirty_mask() { return m_dirty_mask; }
  mem_fetch* get_original_mf() { return m_original_mf; }
  bool is_tlb_request() { return m_tlb_request; }
  mem_fetch* get_tlb_original_mf() {
    assert(m_tlb_request);
    return m_tlb_original_mf;
  }
  std::deque<mem_fetch*> get_wrapped_mf() { return m_wrapped_mf; }
  void set_host_id(unsigned host_id) { m_src_id = host_id; }
  unsigned get_host_id() { return m_src_id; }
  unsigned get_timestamp() const { return m_timestamp; }
  InstColumn* get_inst_column() { return m_inst_column; }
  void set_inst_column(InstColumn* inst_column) {
    m_inst_column = inst_column;
  }
  void set_bi() { m_bi_request = true; }
  void unset_bi() { m_bi_request = false;}
  bool is_bi() { return m_bi_request; }
  void set_bi_writeack() { m_bi_writeback = true; }
  bool is_bi_writeback() { 
    m_mem_access_type = GLOBAL_ACC_W;
    return m_bi_writeback; 
  }
  bool is_uthread_request() { return m_uthread_request; }
  void set_uthread_request() { m_uthread_request = true; }
  bool is_sc_addr() { return m_sc_addr; }
  void set_sc_addr() { m_sc_addr = true; }
  std::string current_state = "NONE";
  uint64_t request_cycle;
  uint64_t response_cycle;
 private:
  bool m_valid = false;
  uint64_t m_request_id;
  unsigned m_data_size;
  unsigned m_ctrl_size;
  new_addr_type m_addr;
  mem_access_type m_mem_access_type;
  mf_type m_type;
  unsigned m_channel;
  unsigned long long m_timestamp;
  SectorMask m_sector_mask;
  SectorMask m_dirty_mask;
  MemAccessByteMask m_byte_mask;
  bool m_atomic;
  unsigned m_ndp_id;
  unsigned m_src_id;
  unsigned m_sub_core_id;
  bool m_from_dma = false;
  bool m_from_ndp = false;
  bool m_filtering = true;
  bool m_on_the_fly = true;
  void* m_data;
  bool m_tlb_request = false;
  mem_fetch* m_tlb_original_mf = NULL;
  mem_fetch* m_original_mf = NULL;
  ReadRequestInfo *m_readreq_mf_info = NULL;
  std::deque<mem_fetch*> m_wrapped_mf;
  InstColumn* m_inst_column = NULL;
  bool m_bi_request = false;
  bool m_bi_writeback = false;
  bool m_uthread_request = false;
  bool m_sc_addr = false;

};
}
#endif
#endif