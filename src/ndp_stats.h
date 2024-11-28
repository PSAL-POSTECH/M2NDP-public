#ifdef TIMING_SIMULATION
#ifndef NDP_STATS_H
#define NDP_STATS_H
#include "cache_stats.h"
#include "register_unit.h"
#include <list>
#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>
#include <cstdarg>

namespace NDPSim {

// class ExecutionUnit;
class ExecutionUnit;

enum Status {
  EMPTY_QUEUE,
  // DEPENDENCY_WAIT
  SCALAR_DEPENDENCY_WAIT,
  AMO_DEPENDENCY_WAIT,
  VECTOR_DEPENDENCY_WAIT,
  VLE_DEPENDENCY_WAIT,
  VSE_DEPENDENCY_WAIT,
  // RESOURCE_WAIT
  SPAD_UNIT_FULL,
  LDST_UNIT_FULL,
  SCALAR_UNIT_FULL,
  SF_UNIT_FULL,
  FP_UNIT_FULL,
  SPAD_VECTOR_UNIT_FULL,
  LDST_VECTOR_UNIT_FULL,
  SCALAR_VECTOR_UNIT_FULL,
  SF_VECTOR_UNIT_FULL,
  FP_VECTOR_UNIT_FULL,
  ADDRESS_UNIT_FULL,
  ADDRESS_VECTOR_UNIT_FULL,
  // INSTRUCTION
  INSTRUCTION_BRANCH_WAIT,
  INSTRUCTION_CACHE_WAIT,
  INSTRUCTION_BLOCK_DEPENDENCY_WAIT,
  // QUEUE
  TO_MEM_FULL,
  TO_CROSSBAR_FULL,
  FROM_MEM_FULL,
  FROM_CROSSBAR_FULL,
  TLB_QUEUE_FULL,
  EXECUTION_UNIT_FULL,
  SPAD_DELAY_QUEUE_FULL,
  DATA_PORT_FULL,
  FILL_PORT_FULL,
  SPAD_QUEUE_FULL,
  LDST_QUEUE_FULL,
  FROM_ICACHE_FULL,

  // CACHE
  CACHE_RESERVATION_FAIL,
  CACHE_DATA_ACCESS,
  CACHE_DATA_FILL,
  
  // OTHER
  ISSUE_SUCCESS,
  SUCCESS,
  NUM_STATUS
};

enum MEM_STATUS {
  TO_MEM_PUSH_MY_CHANNEL,
  TO_MEM_PUSH_OTHER_CHANNEL,
  SCRATCHPAD_READ,
  SCRATCHPAD_WRITE,
  SCRATCHPAD_ACCESS_COUNT,
  NUM_MEM_STATUS
};

enum WaitRegEnum {
  // D 0 1 2 3 4 M
  DEST,
  SRC0,
  SRC1,
  SRC2,
  SRC3,
  SRC4,
  MASK,
  REG_COUNT
};

typedef struct _WaitInstruction{
  std::string s_inst;
  uint32_t wait_reg_mask;
  bool operator<(const _WaitInstruction& other) const {
    if (s_inst == other.s_inst)
      return wait_reg_mask < other.wait_reg_mask;
    return s_inst < other.s_inst;
  }
  _WaitInstruction(std::string inst) {
    s_inst = inst;
    wait_reg_mask = 0;
  }
  _WaitInstruction(std::string inst, int count, ...) {
    s_inst = inst;
    wait_reg_mask = 0x0;

    va_list args;
    va_start(args, count);
    for (int i = 0; i < count; i++) {
      WaitRegEnum reg = static_cast<WaitRegEnum>(va_arg(args, int));
      wait_reg_mask |= 0x1 << reg;
    }
    va_end(args);
  }
  void add_waiting_reg(WaitRegEnum reg) {
    wait_reg_mask |= 0x1 << reg;
  }
} WaitInstruction;

static const char *RegString[] = { "DEST", "SRC0", "SRC1", "SRC2",
                                   "SRC3", "SRC4", "MASK"};

static const char *StatusString[] = {"EMPTY_QUEUE", 
                                    // DEPENDENCY_WAIT
                                    "SCALAR_DEPENDENCY_WAIT", "AMO_DEPENDENCY_WAIT",
                                    "VECTOR_DEPENDENCY_WAIT", "VLE_DEPENDENCY_WAIT",
                                    "VSE_DEPENDENCY_WAIT",
                                    // RESOURCE_WAIT
                                    "SPAD_UNIT_FULL",
                                    "LDST_UNIT_FULL", "SCALAR_UNIT_FULL",
                                    "SF_UNIT_FULL", "FP_UNIT_FULL",
                                    "SPAD_VECTOR_UNIT_FULL",
                                    "LDST_VECTOR_UNIT_FULL", "SCALAR_VECTOR_UNIT_FULL",
                                    "SF_VECTOR_UNIT_FULL", "FP_VECTOR_UNIT_FULL",
                                    "ADDRESS_UNIT_FULL", "ADDRESS_VECTOR_UNIT_FULL",
                                    // INSTRUCTION
                                    "INSTRUCTION_BRANCH_WAIT", "INSTRUCTION_CACHE_WAIT",
                                    "INSTRUCTION_BLOCK_DEPENDENCY_WAIT",
                                    // QUEUE FULL
                                    "TO_MEM_FULL", "TO_CROSSBAR_FULL",
                                    "FROM_MEM_FULL", "FROM_CROSSBAR_FULL",
                                    "TLB_QUEUE_FULL", "EXECUTION_UNIT_FULL",
                                    "SPAD_DELAY_QUEUE_FULL",
                                    "DATA_PORT_FULL", "FILL_PORT_FULL", 
                                    "SPAD_QUEUE_FULL", "LDST_QUEUE_FULL",
                                    "FROM_ICACHE_FULL",
                                    // CACHE
                                    "CACHE_RESERVATION_FAIL",
                                    "CACHE_DATA_ACCESS", "CACHE_DATA_FILL",
                                    // OTHER
                                    "ISSUE_SUCCESS",
                                    };
static const char *MemStatusString[] = {"TO_MEM_PUSH_MY_CHANNEL",
                                        "TO_MEM_PUSH_OTHER_CHANNEL",
                                        "SCRATCHPAD_READ",
                                        "SCRATCHPAD_WRITE",
                                        "SCRATCHPAD_ACCESS_COUNT"};
static const char *RegStatusString[] = {"X_READ", "F_READ", "V_READ",
                                        "X_WRITE", "F_WRITE", "V_WRITE"};


class NdpStats {
  public:
    NdpStats();
    void set_num_sub_core(int num_sub_core) { m_num_sub_core = num_sub_core; }
    void set_id(int id) { m_id = id; }
    void clear();
    void inc_cycle();
    void inc_ndp_inst_queue_size(int size);
    void inc_issue_success(int issue_count);
    void inc_inst_issue_count(int issue_count) { m_inst_issue_count += issue_count; }
    void add_status_list(std::deque<Status> issue_fail_reasons);
    void add_status(Status issue_fail_reason);
    void add_memory_status(MEM_STATUS mem_status, int access_size = 1);
    void add_wait_instruction(const WaitInstruction &wait_inst);
    void inc_inst_queue_full();
    void inc_register_stall();
    void inc_inst_column_q_full();

    uint64_t get_memory_access_status(MEM_STATUS mem_status) {
      return m_memory_status[mem_status];
    }
    uint64_t get_register_status(RegisterStats::RegStatEnum stat) {
      return m_register_stats.register_stats[stat];
    }

    void set_itlb_stats(CacheStats itlb) { m_itlb_stasts = itlb;}
    void set_icache_stats(CacheStats icache) { m_icache_stats = icache;}
    void set_dtlb_stats(CacheStats dtlb) { m_dtlb_stats = dtlb;}
    void set_l1d_stats(CacheStats dcache) { m_l1d_stats = dcache;}
    void set_l2d_stats(CacheStats dcache) { m_l2d_stats = dcache; }
    void set_regsiter_stats(RegisterStats reg) {m_register_stats = reg;}
    void set_l0_icache_stats(CacheStats icache) { m_l0_icache_stats = icache; }

    //Issue count
    void inc_i_unit_issue_count() { m_i_unit_issue_count++; }
    void inc_f_unit_issue_count() { m_f_unit_issue_count++; }
    void inc_sf_unit_issue_count() { m_sf_unit_issue_count++; }
    void inc_addr_unit_issue_count() { m_addr_unit_issue_count++; }
    void inc_ldst_unit_issue_count() { m_ldst_unit_issue_count++; }
    void inc_spad_unit_issue_count() { m_spad_unit_issue_count++; }
    void inc_v_unit_issue_count() { m_v_unit_issue_count++; }
    void inc_v_sf_unit_issue_count() { m_v_sf_unit_issue_count++; }
    void inc_v_addr_unit_issue_count() { m_v_addr_unit_issue_count++; }
    void inc_v_ldst_unit_issue_count() { m_v_ldst_unit_issue_count++; }
    void inc_v_spad_unit_issue_count() { m_v_spad_unit_issue_count++; }

    //RF WB count
    void set_max_wb_per_cyc(uint64_t total_count, uint64_t vreg_count, uint64_t xreg_count, uint64_t freg_count);
    uint64_t get_max_wb_per_cyc() { return m_max_reg_wb_per_cyc_per_sub_core; }
    //Get count
    uint64_t get_i_unit_issue_count() { return m_i_unit_issue_count; }
    uint64_t get_f_unit_issue_count() { return m_f_unit_issue_count; }
    uint64_t get_sf_unit_issue_count() { return m_sf_unit_issue_count; }
    uint64_t get_addr_unit_issue_count() { return m_addr_unit_issue_count; }
    uint64_t get_ldst_unit_issue_count() { return m_ldst_unit_issue_count; }
    uint64_t get_spad_unit_issue_count() { return m_spad_unit_issue_count; }
    uint64_t get_v_unit_issue_count() { return m_v_unit_issue_count; }
    uint64_t get_v_sf_unit_issue_count() { return m_v_sf_unit_issue_count; }
    uint64_t get_v_addr_unit_issue_count() { return m_v_addr_unit_issue_count; }
    uint64_t get_v_ldst_unit_issue_count() { return m_v_ldst_unit_issue_count; }
    uint64_t get_v_spad_unit_issue_count() { return m_v_spad_unit_issue_count; }
    uint64_t inc_active_vlane_count(int active,int total) {
      m_v_active_lane_issue_count += active;
      m_v_total_laine_issue_count += total;
    }
  
    NdpStats operator+(const NdpStats &other);
    NdpStats &operator+=(const NdpStats &other);
    void print_stats_interval();
    void print_stats(FILE *out, const char *ndp_name="NdpStats") const;
    void print_energy_stats(FILE *out) const;
  private:
    uint32_t m_id;
    int m_num_sub_core = 4;
    uint64_t m_cycle = 0;
    uint64_t m_inst_issue_count = 0;
    uint64_t m_max_ndp_inst_queue_size = 0;
    uint64_t m_ndp_inst_queue_size = 0;
    uint64_t m_issue_success_cycle = 0;
    uint64_t m_issue_success_count = 0;

    uint64_t m_i_unit_issue_count = 0;
    uint64_t m_f_unit_issue_count = 0;
    uint64_t m_sf_unit_issue_count = 0;
    uint64_t m_addr_unit_issue_count = 0;
    uint64_t m_ldst_unit_issue_count = 0;
    uint64_t m_spad_unit_issue_count = 0;

    uint64_t m_v_unit_issue_count = 0;
    uint64_t m_v_sf_unit_issue_count = 0;
    uint64_t m_v_addr_unit_issue_count = 0;
    uint64_t m_v_ldst_unit_issue_count = 0;
    uint64_t m_v_spad_unit_issue_count = 0;
    uint64_t m_v_active_lane_issue_count = 0;
    uint64_t m_v_total_laine_issue_count = 0;
    uint64_t m_spad_read_count = 0;
    uint64_t m_spad_write_count = 0;

    std::vector<uint64_t> m_status;
    std::vector<uint64_t> m_memory_status;
    std::map<WaitInstruction, uint64_t> m_wait_insts;
    uint64_t m_total_status = 0;
    uint64_t m_total_queue_full_reasons = 0;
    uint64_t m_inst_queue_full_count = 0;
    uint64_t m_register_stall_count = 0;
    uint64_t m_inst_column_q_full_count = 0;
    
    CacheStats m_itlb_stasts;
    CacheStats m_icache_stats;
    CacheStats m_dtlb_stats;
    CacheStats m_l1d_stats;
    CacheStats m_l2d_stats;
    CacheStats m_l0_icache_stats;

    RegisterStats m_register_stats;

    uint64_t m_prev_issue_success_count;
    uint64_t m_prev_cache_access;
    uint64_t m_prev_cache_fill;

    uint64_t m_max_vreg_wb_per_cyc_per_sub_core = 0;
    uint64_t m_max_xreg_wb_per_cyc_per_sub_core = 0;
    uint64_t m_max_freg_wb_per_cyc_per_sub_core = 0;
    uint64_t m_max_reg_wb_per_cyc_per_sub_core = 0;
};
}  // namespace NDPSim
#endif
#endif