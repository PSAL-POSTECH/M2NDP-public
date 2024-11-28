#ifdef TIMING_SIMULATION
#include "ndp_stats.h"

#include "execution_unit.h"

namespace NDPSim {

NdpStats::NdpStats() {
  m_status.resize(NUM_STATUS, 0);
  m_memory_status.resize(NUM_MEM_STATUS, 0);
  m_prev_issue_success_count = 0;
  m_prev_cache_access = 0;
  m_prev_cache_fill = 0;
  clear();
}

void NdpStats::clear() {
  for (int i = 0; i < NUM_STATUS; i++) {
    m_status[i] = 0;
  }
  for (int i = 0; i < NUM_MEM_STATUS; i++) {
    m_memory_status[i] = 0;
  }
  m_cycle = 0;
  m_max_ndp_inst_queue_size = 0;
  m_ndp_inst_queue_size = 0;
  m_issue_success_cycle = 0;
  m_issue_success_count = 0;
  m_inst_queue_full_count = 0;
  m_register_stall_count = 0;
  m_inst_column_q_full_count = 0;
  m_itlb_stasts.clear();
  m_icache_stats.clear();
  m_dtlb_stats.clear();
  m_l1d_stats.clear();
  m_l2d_stats.clear();
}

void NdpStats::inc_cycle() { m_cycle++; }

void NdpStats::inc_ndp_inst_queue_size(int size) {
  m_ndp_inst_queue_size += size;
  if (size > m_max_ndp_inst_queue_size) {
    m_max_ndp_inst_queue_size = size;
  }
}

void NdpStats::inc_issue_success(int issue_count) {
  m_issue_success_cycle++;
  m_issue_success_count += issue_count;
}

void NdpStats::add_status_list(std::deque<Status> status_list) {
  for (auto it = status_list.begin(); it != status_list.end(); it++) {
    m_status[*it]++;
    m_total_status++;
  }
}

void NdpStats::add_status(Status status) {
  m_status[status]++;
  m_total_status++;
}

void NdpStats::add_memory_status(MEM_STATUS mem_status, int access_size) {
  m_memory_status[mem_status]++;
  if (mem_status == SCRATCHPAD_READ || mem_status == SCRATCHPAD_WRITE)
    m_memory_status[SCRATCHPAD_ACCESS_COUNT] += access_size;
}

void NdpStats::add_wait_instruction(const WaitInstruction &wait_inst) {
  if (m_wait_insts.find(wait_inst) != m_wait_insts.end()) {
    m_wait_insts.at(wait_inst) += 1;
  } else
    m_wait_insts[wait_inst] = 1;
}

void NdpStats::inc_inst_queue_full() { m_inst_queue_full_count++; }

void NdpStats::inc_register_stall() { m_register_stall_count++; }

void NdpStats::inc_inst_column_q_full() { m_inst_column_q_full_count++; }

void NdpStats::set_max_wb_per_cyc(uint64_t total_count, uint64_t vreg_count, uint64_t xreg_count, uint64_t freg_count) {
  m_max_reg_wb_per_cyc_per_sub_core = total_count;
  m_max_vreg_wb_per_cyc_per_sub_core = vreg_count;
  m_max_xreg_wb_per_cyc_per_sub_core = xreg_count;
  m_max_freg_wb_per_cyc_per_sub_core = freg_count;
}

NdpStats NdpStats::operator+(const NdpStats &other) {
  NdpStats sum;
  sum.m_l0_icache_stats = m_l0_icache_stats + other.m_l0_icache_stats;
  sum.m_itlb_stasts = m_itlb_stasts + other.m_itlb_stasts;
  sum.m_icache_stats = m_icache_stats + other.m_icache_stats;
  sum.m_dtlb_stats = m_dtlb_stats + other.m_dtlb_stats;
  sum.m_l1d_stats = m_l1d_stats + other.m_l1d_stats;
  sum.m_l2d_stats = m_l2d_stats + other.m_l2d_stats;
  sum.m_register_stats = m_register_stats + other.m_register_stats;
  for(auto it = other.m_wait_insts.begin(); it != other.m_wait_insts.end(); it++) {
    if(sum.m_wait_insts.find(it->first) != sum.m_wait_insts.end())
      sum.m_wait_insts[it->first] += it->second;
    else
      sum.m_wait_insts[it->first] = it->second;
  }

  for (int i = 0; i < m_status.size(); i++)
    sum.m_status[i] += m_status[i] + other.m_status[i];
  for (int i = 0; i < m_memory_status.size(); i++)
    sum.m_memory_status[i] += m_memory_status[i] + other.m_memory_status[i];
  for (const auto &iter : m_wait_insts)
    if (sum.m_wait_insts.find(iter.first) != sum.m_wait_insts.end())
      sum.m_wait_insts[iter.first] += iter.second;
    else
      sum.m_wait_insts[iter.first] = iter.second;
  sum.m_inst_issue_count = m_inst_issue_count + other.m_inst_issue_count;
  sum.m_i_unit_issue_count = m_i_unit_issue_count + other.m_i_unit_issue_count;
  sum.m_f_unit_issue_count = m_f_unit_issue_count + other.m_f_unit_issue_count;
  sum.m_sf_unit_issue_count =
      m_sf_unit_issue_count + other.m_sf_unit_issue_count;
  sum.m_addr_unit_issue_count =
      m_addr_unit_issue_count + other.m_addr_unit_issue_count;
  sum.m_ldst_unit_issue_count =
      m_ldst_unit_issue_count + other.m_ldst_unit_issue_count;
  sum.m_spad_unit_issue_count =
      m_spad_unit_issue_count + other.m_spad_unit_issue_count;
  sum.m_v_unit_issue_count = m_v_unit_issue_count + other.m_v_unit_issue_count;
  sum.m_v_sf_unit_issue_count =
      m_v_sf_unit_issue_count + other.m_v_sf_unit_issue_count;
  sum.m_v_addr_unit_issue_count =
      m_v_addr_unit_issue_count + other.m_v_addr_unit_issue_count;
  sum.m_v_ldst_unit_issue_count =
      m_v_ldst_unit_issue_count + other.m_v_ldst_unit_issue_count;
  sum.m_v_spad_unit_issue_count =
      m_v_spad_unit_issue_count + other.m_v_spad_unit_issue_count;
  sum.m_v_active_lane_issue_count
      = m_v_active_lane_issue_count + other.m_v_active_lane_issue_count;
  sum.m_v_total_laine_issue_count
      = m_v_total_laine_issue_count + other.m_v_total_laine_issue_count;
  sum.m_spad_read_count = m_spad_read_count + other.m_spad_read_count;
  sum.m_spad_write_count = m_spad_write_count + other.m_spad_write_count;

  return sum;
}

NdpStats &NdpStats::operator+=(const NdpStats &other) {
  m_l0_icache_stats += other.m_l0_icache_stats;
  m_itlb_stasts += other.m_itlb_stasts;
  m_icache_stats += other.m_icache_stats;
  m_dtlb_stats += other.m_dtlb_stats;
  m_l1d_stats += other.m_l1d_stats;
  m_l2d_stats += other.m_l2d_stats;
  m_register_stats += other.m_register_stats;

  for (int i = 0; i < m_status.size(); i++) m_status[i] += other.m_status[i];
  for (int i = 0; i < m_memory_status.size(); i++)
    m_memory_status[i] += other.m_memory_status[i];
  for (const auto &iter : other.m_wait_insts)
    if (m_wait_insts.find(iter.first) != m_wait_insts.end())
      m_wait_insts[iter.first] += iter.second;
    else
      m_wait_insts[iter.first] = iter.second;
  
  m_inst_issue_count += other.m_inst_issue_count;
  m_i_unit_issue_count += other.m_i_unit_issue_count;
  m_f_unit_issue_count += other.m_f_unit_issue_count;
  m_sf_unit_issue_count += other.m_sf_unit_issue_count;
  m_addr_unit_issue_count += other.m_addr_unit_issue_count;
  m_ldst_unit_issue_count += other.m_ldst_unit_issue_count;
  m_spad_unit_issue_count += other.m_spad_unit_issue_count;

  m_v_unit_issue_count += other.m_v_unit_issue_count;
  m_v_sf_unit_issue_count += other.m_v_sf_unit_issue_count;
  m_v_addr_unit_issue_count += other.m_v_addr_unit_issue_count;
  m_v_ldst_unit_issue_count += other.m_v_ldst_unit_issue_count;
  m_v_spad_unit_issue_count += other.m_v_spad_unit_issue_count;
  m_v_active_lane_issue_count += other.m_v_active_lane_issue_count;
  m_v_total_laine_issue_count += other.m_v_total_laine_issue_count;

  m_spad_read_count += other.m_spad_read_count;
  m_spad_write_count += other.m_spad_write_count;
  return *this;
}

void NdpStats::print_stats_interval() {
  if (m_id == 0) {
    spdlog::info("NDP {:2}: Cache Data Access {:5}, Data Fill {:5}", m_id,
                 m_status[CACHE_DATA_ACCESS] - m_prev_cache_access,
                 m_status[CACHE_DATA_FILL] - m_prev_cache_fill);
    spdlog::info("NDP {:2}: IPC {:5}", m_id,
                 (m_issue_success_count - m_prev_issue_success_count) / 1000.0);
  } else {
    spdlog::debug("NDP {:2}: Cache Data Access {:5}, Data Fill {:5}", m_id,
                  m_status[CACHE_DATA_ACCESS] - m_prev_cache_access,
                  m_status[CACHE_DATA_FILL] - m_prev_cache_fill);
    spdlog::debug(
        "NDP {:2}: IPC {:5}", m_id,
        (m_issue_success_count - m_prev_issue_success_count) / 1000.0);
  }
  m_prev_cache_access = m_status[CACHE_DATA_ACCESS];
  m_prev_cache_fill = m_status[CACHE_DATA_FILL];
  m_prev_issue_success_count = m_issue_success_count;
}

void NdpStats::print_stats(FILE *out, const char *ndp_name) const {
  float issue_rate = ((float)m_issue_success_cycle) / m_cycle * 100;
  float avg_issue_count =
      ((float)m_issue_success_count) / m_issue_success_cycle;
  uint64_t sum = 0;
  fprintf(out, "%s: \n", ndp_name);
  fprintf(out,
          "Cycle %llu\tIssue Success Cycle %llu\tIssue success rate %.2f "
          "%\tAVG isesue count %.2f\n",
          m_cycle, m_issue_success_cycle, issue_rate, avg_issue_count);
  fprintf(out, "Per Sub-core AVG NDP Inst Queue Size: %.2f\tMAX NDP Inst Queue Size: %lld\n",
          (((float)m_ndp_inst_queue_size) / m_num_sub_core)/ m_cycle, m_max_ndp_inst_queue_size);
  fprintf(out, "Status :\n");
  for (int i = 0; i < NUM_STATUS; i++) {
    if (i == ISSUE_SUCCESS || i == SUCCESS)
      continue;
    else if (i == EMPTY_QUEUE) {
      fprintf(out, "Instruction Issue Fail:\n");
      sum = 0;
      for (int i = EMPTY_QUEUE; i < INSTRUCTION_BRANCH_WAIT; i++)
        sum += m_status[i];
    } else if (i == INSTRUCTION_BRANCH_WAIT) {
      fprintf(out, "Instruction Wait:\n");
      sum = 0;
      for (int i = INSTRUCTION_BRANCH_WAIT; i < TO_MEM_FULL; i++)
        sum += m_status[i];
    } else if (i == TO_MEM_FULL) {
      fprintf(out, "Queue Full:\n");
      sum = 0;
      for (int i = TO_MEM_FULL; i < CACHE_RESERVATION_FAIL; i++)
        sum += m_status[i];
    } else if (i == CACHE_RESERVATION_FAIL) {
      fprintf(out, "Cache:\n");
      sum = 0;
      for (int i = CACHE_RESERVATION_FAIL; i < CACHE_RESERVATION_FAIL + 3; i++)
        sum += m_status[i];
    }
    fprintf(out, "\t%s: %llu (%.2f %)\n", StatusString[i], m_status[i],
            ((float)m_status[i]) / sum * 100);
  }
  fprintf(out, "Wait Instructions [wait register]:\n");
  for (auto iter = m_wait_insts.begin(); iter != m_wait_insts.end(); ++iter) {
    std::string s_reg = "";
    for (int i = 0; i < REG_COUNT; i++)
      if (iter->first.wait_reg_mask & 0x1 << i) {
        s_reg += RegString[i];
        s_reg += " ";
      }
    fprintf(out, "\t%s [%s] : %lld\n", iter->first.s_inst.c_str(),
            s_reg.c_str(), iter->second);
  }
  fprintf(out, "Execution Unit Status:\n");
  fprintf(out, "Total Inst Issue Count: %llu\n", m_inst_issue_count);
  fprintf(out, "Total NDP Cycle: %llu\n", m_cycle);
  fprintf(out, "IPC: %.2f \n", float(m_inst_issue_count) / m_cycle);
  fprintf(out, "I unit issue count: %llu\n", m_i_unit_issue_count);
  fprintf(out, "F unit issue count: %llu\n", m_f_unit_issue_count);
  fprintf(out, "SF unit issue count: %llu\n", m_sf_unit_issue_count);
  fprintf(out, "ADDR unit issue count: %llu\n", m_addr_unit_issue_count);
  fprintf(out, "LDST unit issue count: %llu\n", m_ldst_unit_issue_count);
  fprintf(out, "SPAD unit issue count: %llu\n", m_spad_unit_issue_count);
  fprintf(out, "V unit issue count: %llu\n", m_v_unit_issue_count);

  fprintf(out, "V total lane issue count: %llu\n", m_v_total_laine_issue_count);
  fprintf(out, "V active lane issue count: %llu\n", m_v_active_lane_issue_count);

  fprintf(out, "V_SF unit issue count: %llu\n", m_v_sf_unit_issue_count);
  fprintf(out, "V_ADDR unit issue count: %llu\n", m_v_addr_unit_issue_count);
  fprintf(out, "V_LDST unit issue count: %llu\n", m_v_ldst_unit_issue_count);
  fprintf(out, "V_SPAD unit issue count: %llu\n", m_v_spad_unit_issue_count);
  fprintf(out, "Memory Status:\n");
  for (int i = 0; i < NUM_MEM_STATUS; i++) {
    fprintf(out, "\t%s: %llu (%.2f per cycle)\n", MemStatusString[i],
            m_memory_status[i], ((float)m_memory_status[i]) / m_cycle);
  }
  fprintf(out, "Register Status:\n");
  for (int i = 0; i < RegisterStats::REG_STAT_NUM; i++) {
    fprintf(out, "\t%s: %u \n", RegStatusString[i],
            m_register_stats.register_stats[i]);
  }
  fprintf(out, "\tMax Register WB Per Cycle Per Sub-core: %llu\n", m_max_reg_wb_per_cyc_per_sub_core);
  fprintf(out, "\tMax V Register WB Per Cycle Per Sub-core: %llu\n", m_max_vreg_wb_per_cyc_per_sub_core);
  fprintf(out, "\tMax X Register WB Per Cycle Per Sub-core: %llu\n", m_max_xreg_wb_per_cyc_per_sub_core);
  fprintf(out, "\tMax F Register WB Per Cycle Per Sub-core: %llu\n", m_max_freg_wb_per_cyc_per_sub_core);

  fprintf(out, "Cache Avg Hit Miss:\n");

  fprintf(out, "Inst Queue Full Count: %llu\n", m_inst_queue_full_count);
  fprintf(out, "Register Stall Count: %llu\n", m_register_stall_count);
  fprintf(out, "Inst Column Q at each sub-core Full Count: %llu\n", m_inst_column_q_full_count);
  fprintf(out, "=========I-TLB========\n");
  m_itlb_stasts.print_stats(out, "I-TLB");
  fprintf(out, "=========I-Cache========\n");
  m_l0_icache_stats.print_stats(out, "L0I-Cache");
  m_icache_stats.print_stats(out, "L1I-Cache");
  fprintf(out, "=========D-TLB========\n");
  m_dtlb_stats.print_stats(out, "D-TLB");
  fprintf(out, "=========L1-D Cache========\n");
  m_l1d_stats.print_stats(out, "L1-D Cache");
}

void NdpStats::print_energy_stats(FILE *out) const {
  fprintf(out, "I_ISSUE_CNT: %llu\n", m_i_unit_issue_count);
  fprintf(out, "F_ISSUE_CNT: %llu\n", m_f_unit_issue_count);
  fprintf(out, "SF_ISSUE_CNT: %llu\n", m_sf_unit_issue_count);
  fprintf(out, "ADDR_ISSUE_CNT: %llu\n", m_addr_unit_issue_count);
  fprintf(out, "LDST_ISSUE_CNT: %llu\n", m_ldst_unit_issue_count);
  fprintf(out, "SPAD_ISSUE_CNT: %llu\n", m_spad_unit_issue_count);
  fprintf(out, "V_ISSUE_CNT: %llu\n", m_v_unit_issue_count);
  fprintf(out, "V_SF_ISSUE_CNT: %llu\n", m_v_sf_unit_issue_count);
  fprintf(out, "V_ADDR_ISSUE_CNT: %llu\n", m_v_addr_unit_issue_count);
  fprintf(out, "V_LDST_ISSUE_CNT: %llu\n", m_v_ldst_unit_issue_count);
  fprintf(out, "V_SPAD_ISSUE_CNT: %llu\n", m_v_spad_unit_issue_count);
  fprintf(out, "SPAD_ACCESS_CNT: %llu\n",
          m_memory_status[SCRATCHPAD_ACCESS_COUNT]);
  fprintf(out, "XREG_RD: %llu\n",
          m_register_stats.register_stats[RegisterStats::X_READ]);
  fprintf(out, "XREG_WR: %llu\n",
          m_register_stats.register_stats[RegisterStats::X_WRITE]);
  fprintf(out, "FREG_RD: %llu\n",
          m_register_stats.register_stats[RegisterStats::F_READ]);
  fprintf(out, "FREG_WR: %llu\n",
          m_register_stats.register_stats[RegisterStats::F_WRITE]);
  fprintf(out, "VREG_RD: %llu\n",
          m_register_stats.register_stats[RegisterStats::V_READ]);
  fprintf(out, "VREG_WR: %llu\n",
          m_register_stats.register_stats[RegisterStats::V_WRITE]);
  m_itlb_stasts.print_energy_stats(out, "ITLB");
  m_icache_stats.print_energy_stats(out, "L1I-CACHE");
  m_dtlb_stats.print_energy_stats(out, "DTLB");
  m_l1d_stats.print_energy_stats(out, "L1D");
  m_l0_icache_stats.print_energy_stats(out, "L0I-CACHE");
} 

}  // namespace NDPSim
#endif