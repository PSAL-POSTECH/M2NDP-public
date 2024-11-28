#ifdef TIMING_SIMULATION
#include "m2ndp.h"

#include <bitset>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "ndp_instruction.h"
namespace NDPSim {
static unsigned long long global_launch_id = 0;
M2NDP::M2NDP(M2NDPConfig *config, MemoryMap *memory_map,
                       int buffer_id) {
  m_config = config;
  m_memory_map = memory_map;
  m_buffer_id = buffer_id;
  memory_cycles = 0;
  m_inter_cxl_round_robin = 0;
  m_num_ndp_units = m_config->get_ndp_units_per_buffer();
  m_num_banks = m_config->get_l2d_num_banks();
  m_num_m2ndp = m_config->get_num_m2ndps();
  m_contiguous_ch =
      m_config->get_channel_interleave_size() / m_config->get_packet_size();

  m_icnt = InterconnectInterface::New(
      m_config->get_local_crossbar_config_path().c_str());
  m_icnt->CreateInterconnect(
      m_config->get_links_per_memory_buffer() * m_num_banks + m_num_ndp_units * m_num_banks,
      m_config->get_num_channels() * m_num_banks);
  m_icnt->Init();

  tot_cycles = 0;
  memory_cycles = 1;

  std::string name("CXL_");
  if (m_config->get_channel_interleave_size() == 256) {
    name = std::string("M2NDP_");
  }
  name += std::to_string(m_buffer_id);
  name += ("_ramulator");
  int ndp_id = buffer_id * m_config->get_num_memory_buffers();
  m_ramulator = new NdpRamulator(
      buffer_id, ndp_id, m_memory_map, &tot_cycles, m_config->m_num_hosts + 1,
      m_config->m_ramulator_config_path, m_config, name);

  m_ndp_units.resize(m_num_ndp_units);
  m_ndp_stats.resize(m_num_ndp_units);
  for (int i = 0; i < m_num_ndp_units; i++) {
    int id = m_buffer_id * m_num_ndp_units + i;
    id = get_ndp_id(i);
    m_ndp_stats[i].set_num_sub_core(m_config->get_num_sub_core());
    m_ndp_units[i] = new NdpUnit(m_config, m_memory_map, &m_ndp_stats[i], id);
  }
  m_total_stats.set_num_sub_core(m_config->get_num_sub_core());
  m_ndp_kernels.resize(m_config->m_num_hosts);
  m_host_round_robin.resize(m_config->m_num_hosts);
  m_cxl_command_response.resize(m_config->get_links_per_memory_buffer());
  m_num_read_request.resize(m_num_m2ndp);
  m_num_read_response.resize(m_num_m2ndp);
  m_num_write_request.resize(m_num_m2ndp);
  m_num_write_response.resize(m_num_m2ndp);
  for (int i = 0; i < m_config->m_num_hosts; i++) {
    m_host_round_robin[i] = 0;
  }
  m_uthread_reqs.resize(m_ndp_units.size(), 0);
}

M2NDP::~M2NDP() { delete m_ramulator; }

void M2NDP::cxl_link_cycle() {
  transfer_cxl_to_memory();
  transfer_memory_to_cxl();
}

void M2NDP::cycle() {
  int cxl_port_offset = m_config->get_links_per_memory_buffer() * m_num_banks;
  if (m_config->is_link_cycle()) {
    cxl_link_cycle();
  }

  if (m_config->is_cache_cycle()) {
    m_ramulator->cache_cycle();
  }

  if (m_config->is_buffer_ndp_cycle()) {
    // NDP to ICNT
    for (int i = 0; i < m_num_ndp_units; i++) {
      for (int bank = 0; bank < m_config->get_l2d_num_banks(); bank++) {
        if (!m_ndp_units[i]->to_mem_empty(bank)) {
          mem_fetch *mf = m_ndp_units[i]->top_to_mem(bank);
          mf->current_state = "top to mem";
          int input_id = cxl_port_offset + i * m_num_banks + bank;
          int output_id = get_output_port_id(mf);
          if(m_config->is_sc_work_addr()) {
            mf->set_sc_addr();
            if(m_cxl_link->has_buffer_from_host(0, 0, mf)) {
              m_cxl_link->push_from_host(0, 0, mf);
              m_ndp_units[i]->pop_to_mem(bank);
            }
          } else if (m_icnt->HasBuffer(input_id, mf->get_size(), mf->get_from_ndp())) {
            mf->current_state = "m_ndp -> push icnt";
            m_icnt->Push(input_id, output_id, mf, mf->get_size());
            m_ndp_units[i]->pop_to_mem(bank);
          }
        }
      }
    }
    // ICNT to NDP
    for (int i = 0; i < m_num_ndp_units; i++) {
      for (int bank = 0; bank < m_config->get_l2d_num_banks(); bank++) {
        int port_id = cxl_port_offset + i * m_num_banks + bank;
        if (!m_ndp_units[i]->from_mem_full(bank) &&
            m_icnt->Top(port_id) != NULL) {
          mem_fetch *mf = (mem_fetch *)m_icnt->Top(port_id);
          mf->current_state = "icnt -> to ndp";
          if (!mf->is_request() || mf->get_ndp_id() == get_ndp_id(i)) {
            mf->current_state = "icnt -> to ndp push";
            m_ndp_units[i]->push_from_mem(mf, bank);
            m_icnt->Pop(port_id);
          } else {
            mf->current_state = "other port";
            m_icnt->Push(port_id, 0, mf, mf->get_size());
          }
        }
      }
    }
    // Memory to ICNT
    for (int i = 0; i < m_config->get_num_channels(); i++) {
      for (int bank = 0; bank < m_config->get_l2d_num_banks(); bank++) {
        int input_id = cxl_port_offset +
                       m_num_ndp_units * m_num_banks;
        input_id += i * m_num_banks + bank;
        if (m_ramulator->top(i, bank) != NULL) {
          mem_fetch *mf = m_ramulator->top(i, bank);
          int output_id = get_output_port_id(mf);
          if (m_icnt->HasBuffer(input_id, mf->get_size(), mf->get_from_ndp())) {
            mf->current_state = "to crossbar -> icnt";
            m_icnt->Push(input_id, output_id, mf, mf->get_size());
            m_ramulator->pop(i, bank);
          }
        }
      }
    }
    // ICNT to Memory
    for (int i = 0; i < m_config->get_num_channels(); i++) {
      for (int bank = 0; bank < m_config->get_l2d_num_banks(); bank++) {
        int input_id = cxl_port_offset +
                       m_num_ndp_units * m_num_banks;
        input_id += i * m_num_banks + bank;
        if (!m_ramulator->full(i, bank) && m_icnt->Top(input_id) != NULL) {
          mem_fetch *mf = (mem_fetch *)m_icnt->Top(input_id);
          mf->current_state = "icnt -> to ramulator";
          assert(m_config->get_channel_index(mf->get_addr()) == i);
          m_ramulator->push(mf, i, bank);
          m_icnt->Pop(input_id);
        }
      }
    }
    m_icnt->Advance();
    if (m_config->get_ndp_cycle() % m_config->get_log_interval() == 0) {
      for(int i = 0; i < m_num_ndp_units; i++) {
        m_ndp_units[i]->print_ndp_stats();
      }
      for(int i = 0; i < m_num_ndp_units; i++) {
        m_ndp_units[i]->print_uthread_stats();
      }
    }
    for (int i = 0; i < m_num_ndp_units; i++) {
      m_ndp_units[i]->cycle();
      
    }
  }
  if (m_config->is_buffer_dram_cycle()) {
    m_ramulator->dram_cycle();
  }
  tot_cycles = m_config->get_sim_cycle();
  check_kernel_active();
}

void M2NDP::register_ndp_kernel(int host_id, string kernel_path) {
  NdpKernel *ndp_kernel = new NdpKernel();
  m_ndp_kernels[host_id].push_back(ndp_kernel);
  M2NDPParser::parse_ndp_kernel(m_config->get_num_ndp_units(), kernel_path,
                               ndp_kernel, m_config);
  spdlog::info(
      "Host {} Registered NDP kernel {} at  core cycle {} ndp cycle "
      "{} to CXL {}",
      host_id, ndp_kernel->kernel_name, m_config->get_sim_cycle(),
      m_config->get_ndp_cycle(), m_buffer_id);
  for (int j = 0; j < m_num_ndp_units; j++) {
    int ndp_id = get_ndp_id(j);
    m_ndp_units[j]->register_ndp_kernel(ndp_kernel);
  }
}

bool M2NDP::can_register_ndp_kernel(int host_id) {
  return m_ndp_kernels[host_id].size() < m_config->get_max_kernel_register();
}

bool M2NDP::is_active() {
  bool active = false;
  for (int i = 0; i < m_config->get_num_hosts(); i++) {
    active = active || is_launch_active(i) || m_cxl_link->is_active() ||
             m_ramulator->is_active();
  }
  return active;
}

bool M2NDP::is_launch_active(int host_id) {
  for (auto iter = m_launch_infos.begin(); iter != m_launch_infos.end();
       iter++) {
    if (iter->host_id == host_id) {
      for (int i = 0; i < m_num_ndp_units; i++) {
        if (m_ndp_units[i]->is_launch_active(iter->launch_id)) {
          return true;
        }
      }
    }
  }
  return false;
}

void M2NDP::display_stats(FILE *fp) {
  m_icnt->DisplayStats(fp);
  m_ramulator->finish();
  m_ramulator->print_all(fp);
  m_total_stats = m_ndp_stats[0];
  for (int i = 0; i < m_num_ndp_units; i++) {
    m_ndp_units[i]->get_stats();
    fprintf(fp, "======= NDP unit %d stats ======\n", i);
    m_ndp_stats[i].print_stats(fp);
    if(i != 0)
     m_total_stats += m_ndp_stats[i];
  }
  fprintf(fp, "======= Total NDP ======\n");
  m_total_stats.print_stats(fp);
}

void M2NDP::print_energy_stats(FILE *fp) {
  m_ramulator->print_energy_stats(fp);
  NdpStats total;
  total.set_num_sub_core(m_config->get_num_sub_core());
  for (int i = 0; i < m_num_ndp_units; i++) {
    total += m_ndp_units[i]->get_stats();
  }
  total.print_energy_stats(fp);
  m_icnt->print_energy_stats(fp, "XBAR");
  fprintf(fp, "STATIC: %lf\n", m_config->m_buffer_ndp_time * m_config->m_num_ndp_units);
  fprintf(fp, "CONST: %lf\n", m_config->m_buffer_ndp_time);
}

/*
 * Private function
 */
void M2NDP::launch_ndp_kernel(int host_id, KernelLaunchInfo info) {
  int launch_id = global_launch_id++;
  info.launch_id = launch_id;
  info.kernel_name = get_kernel_name(host_id, info.kernel_id);
  m_launch_infos.push_back(info);
  for (int i = 0; i < m_num_ndp_units; i++) {
    int ndp_id = get_ndp_id(i);
    m_ndp_units[i]->launch_ndp_kernel(info);
  }
  spdlog::info("Gate info: host {} launched NDP kernel {} launch id {} at core cycle {} ndp "
               "cycle {} to CXL {}",
               host_id, info.kernel_name, launch_id, m_config->get_sim_cycle(),
               m_config->get_ndp_cycle(), m_buffer_id);
}

void M2NDP::transfer_cxl_to_memory() {

  for(int i = 0; i < m_ndp_stats.size(); i++) {
    if(m_uthread_reqs[i] > 0) {
      if(m_ndp_units[i]->generate_uthreads()) {
        m_uthread_reqs[i]--;
      }
    }
  }

  for (int i = 0; i < m_config->get_links_per_memory_buffer(); i++) {
    for (int bank = 0; bank < m_num_banks; bank++) {
      int port_id = i * m_num_banks + bank;
      // Process memory fetch request that from host to CXL memory buffer
      mem_fetch *mf = m_cxl_link->top_from_memory_buffer(m_buffer_id, i);
      if (mf && mf->is_uthread_request()) {
        uint64_t addr = mf->get_addr();
        int ndp_id = m_config->get_matched_unit_id(addr);
        m_uthread_reqs[ndp_id]++;
        m_cxl_link->pop_from_memory_buffer(m_buffer_id, i);
        delete mf;
      }
      else if (mf && mf->get_addr() == KERNEL_LAUNCH_ADDR) {
        KernelLaunchInfo *info = (KernelLaunchInfo *)mf->get_data();
        m_cxl_link->pop_from_memory_buffer(m_buffer_id, i);
        if (info->sync) {
          info->cxl_command = mf;
        } else {
          mf->set_reply();
          m_cxl_command_response[i].push_back(mf);
        }
        launch_ndp_kernel(mf->get_host_id(), *info);
      } else if (mf && !mf->get_from_ndp()) {
        int output_id = get_output_port_id(mf);
        int size = mf->get_size();
        if (m_icnt->HasBuffer(port_id, size)) {
          m_icnt->Push(port_id, output_id, mf, size);
          m_cxl_link->pop_from_memory_buffer(m_buffer_id, i);
        }
      } else if (mf) {
        // Process from NDP request
        int channel_id;
        int output_id;
        int size = mf->get_size();
        if (mf->is_request()) {
          assert(m_config->get_m2ndp_index(mf->get_addr()) ==
                  m_buffer_id);
        }
        output_id = get_output_port_id(mf);
        if (m_icnt->HasBuffer(port_id, size, true)) {
          m_icnt->Push(port_id, output_id, mf, size);
          count_access_per_m2ndp(mf);
          m_cxl_link->pop_from_memory_buffer(m_buffer_id, i);
        }
      }
    }
  }
}

void M2NDP::transfer_memory_to_cxl() {
  // normal distribution
  for (int i = 0; i < m_config->m_links_per_memory_buffer; i++) {
    for (int bank = 0; bank < m_num_banks; bank++) {
      int id = i * m_num_banks + bank;
      if (!m_cxl_command_response[i].empty()) {
        mem_fetch *mf = (mem_fetch *)m_cxl_command_response[i].front();
        if (m_cxl_link->has_buffer_from_memory_buffer(m_buffer_id, i, mf)) {
          m_cxl_link->push_from_memory_buffer(m_buffer_id, i, mf);
          m_cxl_command_response[i].pop_front();
        }
      } else if (m_icnt->Top(id) != NULL) {
        mem_fetch *mf = (mem_fetch *)m_icnt->Top(id);
        if (m_cxl_link->has_buffer_from_memory_buffer(m_buffer_id, i, mf)) {
          m_cxl_link->push_from_memory_buffer(m_buffer_id, i, mf);
          m_icnt->Pop(id);
        }
      }
    }
  }
}

void M2NDP::check_kernel_active() {
  for (auto iter = m_launch_infos.begin(); iter != m_launch_infos.end();
       iter++) {
    KernelLaunchInfo launch_info = *iter;
    bool all_finished = true;
    for (int ndp = 0; ndp < m_num_ndp_units; ndp++) {
      int ndp_id = get_ndp_id(ndp);
      all_finished = all_finished &&
                     !m_ndp_units[ndp]->is_launch_active(launch_info.launch_id);
    }
    if (all_finished) {
      spdlog::info(
          "Gantt info: host {} finished NDP kernel {}  launch id {} at core cycle {}  "
          "ndp "
          "cycle {} at CXL {}",
          launch_info.host_id, launch_info.kernel_name, launch_info.launch_id,
          m_config->get_sim_cycle(), m_config->get_ndp_cycle(), m_buffer_id);
      if (iter->sync) {
        launch_info.cxl_command->set_reply();
        int get_output_id = get_output_port_id(launch_info.cxl_command);
        m_cxl_command_response[get_output_id].push_back(
            launch_info.cxl_command);
      }
      m_launch_infos.erase(iter);
      return;
    }
  }
}

int M2NDP::get_ndp_id(int index) {
  return m_buffer_id * m_num_ndp_units + index;
}

int M2NDP::get_output_port_id(mem_fetch *mf) {
  // Process from host request
  int ndp_port_offset = m_num_ndp_units * m_num_banks;
  int cxl_port_offset = m_config->get_links_per_m2ndp() * m_num_banks;
  int link_per_host = m_config->get_links_per_host() * m_num_banks;
  int bank_id = m_config->get_bank_index(mf->get_addr()) % m_num_banks;
  int channel_id = m_config->get_channel_index(mf->get_addr());
  int dev_id = m_config->get_m2ndp_index(mf->get_addr());
  if (!mf->is_request() && mf->get_from_ndp()) {
    channel_id = mf->get_ndp_id();
  }
  int channel_dest = channel_id;
  if ((m_buffer_id != dev_id) && mf->get_from_ndp() &&
      (mf->get_access_type() == GLOBAL_ACC_R || mf->get_access_type() == GLOBAL_ACC_W)) {
    if (!mf->is_request()) {
      int ndp_id = mf->get_ndp_id() % m_config->get_num_ndp_units();
      return cxl_port_offset + ndp_id * m_num_banks + bank_id;
    }
    return (channel_dest * m_num_banks + bank_id) % link_per_host;
  }
  if (!mf->is_request() && mf->get_from_ndp()) {
    channel_dest = mf->get_ndp_id() % m_num_ndp_units;
    if (m_buffer_id != int(mf->get_ndp_id() / m_num_ndp_units)) {
      return (channel_dest * m_num_banks + bank_id) % link_per_host;
    }
  }
  if (mf->is_request() && !mf->is_bi()) {
    return cxl_port_offset + ndp_port_offset + channel_dest * m_num_banks +
           bank_id;
  } else if (mf->get_from_ndp() && !mf->is_request()) {
    return cxl_port_offset + channel_dest * m_num_banks + bank_id;
  } else if ((!mf->get_from_ndp() && !mf->is_request()) 
    || (mf->is_bi() && mf->is_request())) { // route back-invaildate request to host
    int host_id = mf->get_host_id();
    return host_id * link_per_host +
           (channel_dest * m_num_banks + bank_id) % link_per_host;
  }
  return 0;
}

void M2NDP::count_access_per_m2ndp(mem_fetch* mf) {
  int mf_m2ndp = int(mf->get_ndp_id() / m_num_ndp_units);
  int reply_m2ndp = m_config->get_m2ndp_index(mf->get_addr());
  switch(mf->get_type()) {
    case READ_REQUEST:
      m_num_read_request[mf_m2ndp]++;
      break;
    case READ_REPLY:
      m_num_read_response[reply_m2ndp]++;
      break;
    case WRITE_REQUEST:
      m_num_write_request[mf_m2ndp]++;
      break;
    case WRITE_ACK:
      m_num_write_response[reply_m2ndp]++;
      break;
    default:
      break;
  }
}

void M2NDP::print_access_time(FILE* fp) {
  spdlog::info("[M2NDP:{}]", m_buffer_id);
  for (int i = 0; i < m_num_m2ndp; i++) {
    fprintf(fp, "M2NDP: %d read request: %d read response: %d write request: %d write response: %d\n",
    i, m_num_read_request[i], m_num_read_response[i], m_num_write_request[i], m_num_write_response[i]);
    spdlog::info("M2NDP: {} read request: {} read response: {} write request: {} write response: {}", 
    i, m_num_read_request[i], m_num_read_response[i], m_num_write_request[i], m_num_write_response[i]);
  }
}

std::string M2NDP::get_kernel_name(int host_id, int kernel_id) {
  for (auto kernel : m_ndp_kernels[host_id]) {
    if (kernel->kernel_id == kernel_id) {
      return kernel->kernel_name;
    }
  }
  return "";
}
}  // namespace NDPSimg¡
#endif