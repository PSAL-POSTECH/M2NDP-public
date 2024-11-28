#ifdef TIMING_SIMULATION
#include "cxl_link.h"

#include <fstream>
#include <mutex>
#include <random>

#include "m2ndp_config.h"
namespace NDPSim {

CxlLink::CxlLink(M2NDPConfig* m2ndp_config) {
  m_m2ndp_config = m2ndp_config;
  m_cycles = 0;
  m_interconnect_interface = InterconnectInterface::New(
      m_m2ndp_config->get_cxl_link_config_path().c_str());
  m_host_ports =
      m_m2ndp_config->get_num_hosts() * m_m2ndp_config->get_links_per_host();
  m_m2ndp_ports = m_m2ndp_config->get_num_m2ndps() *
                 m_m2ndp_config->get_links_per_m2ndp();
  m_num_ports = m_host_ports + m_m2ndp_ports;
  m_num_connections = 1; // connection must be 1
      // m_m2ndp_config->get_links_per_host() / m_m2ndp_config->get_num_m2ndps();
  m_interconnect_interface->CreateInterconnect(m_host_ports, m_m2ndp_ports);
  m_interconnect_interface->Init();
  m_m2ndp_offset =
      m2ndp_config->get_links_per_host() * m2ndp_config->get_num_hosts();
  m_from_host_buffers.resize(m_host_ports);
  m_to_host_buffers.resize(m_host_ports);
  for (int i = 0; i < m_host_ports; i++) {
    m_from_host_buffers[i].resize(m_m2ndp_ports);
  }
  m_from_m2ndp_buffers.resize(m_m2ndp_ports);
  m_to_m2ndp_buffers.resize(m_m2ndp_ports);
  for (int i = 0; i < m_m2ndp_ports; i++) {
    m_from_m2ndp_buffers[i].resize(m_host_ports + m_m2ndp_ports);
  }
}

bool CxlLink::has_buffer_from_host(int host_id, int node_no, mem_fetch* mf) {
  int id = host_id * m_m2ndp_config->get_links_per_host() + node_no;
  int switch_id = node_no % m_m2ndp_config->get_num_m2ndps();
  int switch_offset = node_no / m_m2ndp_config->get_num_m2ndps();
  int out_id = switch_id * m_m2ndp_config->get_links_per_m2ndp() +
               host_id * m_num_connections + switch_offset;
  id = host_id; // remove this for other topology
  out_id = node_no;
  return m_from_host_buffers[id][out_id].size() <
         m_m2ndp_config->get_cxl_link_buffer_size();
}

void CxlLink::push_from_host(int host_id, int node_no, mem_fetch* mf) {
  assert(mf != NULL);
  int id = host_id * m_m2ndp_config->get_links_per_host() + node_no;
  int ch_id = m_m2ndp_config->get_channel_index(mf->get_addr());
  int switch_id = node_no % m_m2ndp_config->get_num_m2ndps();
  int switch_offset = node_no / m_m2ndp_config->get_num_m2ndps();
  int out_id = switch_id * m_m2ndp_config->get_links_per_m2ndp() +
               host_id * m_num_connections + switch_offset;
  id = host_id; // remove this for other topology
  out_id = node_no;
  m_from_host_buffers[id][out_id].push_back(mf);
}

mem_fetch* CxlLink::pop_from_host(int host_id, int node_no) {
  int id = host_id * m_m2ndp_config->get_links_per_host() + node_no;
  id = host_id; // TODO: remove this for other topology
  if (m_to_host_buffers[id].empty()) {
    return NULL;
  }
  mem_fetch* mf = m_to_host_buffers[id].front();
  m_to_host_buffers[id].pop_front();
  return mf;
}

mem_fetch* CxlLink::top_from_host(int host_id, int node_no) {
  int id = host_id + m_m2ndp_config->get_num_hosts() * node_no;
  id = host_id; // TODO: remove this for other topology
  if (m_to_host_buffers[id].empty()) {
    return NULL;
  }
  return m_to_host_buffers[id].front();
}

bool CxlLink::has_buffer_from_memory_buffer(int buffer_id, int node_no,
                                            mem_fetch* mf) {
  int id = buffer_id + m_m2ndp_config->get_num_m2ndps() * node_no;
  int host_id = node_no / m_num_connections;
  int offset = node_no % m_num_connections;
  int out_id = host_id * m_m2ndp_config->get_links_per_host() +
               buffer_id * m_num_connections + offset;
  id = buffer_id; // TODO: remove this for other topology
  out_id = node_no; // TODO: remove this for other topology
  if (mf->get_addr() != KERNEL_LAUNCH_ADDR && mf->get_from_ndp())
    out_id = m_m2ndp_offset + m_m2ndp_config->get_m2ndp_index(mf->get_addr());
  return m_from_m2ndp_buffers[id][out_id].size() <
         m_m2ndp_config->get_cxl_link_buffer_size();
}

void CxlLink::push_from_memory_buffer(int buffer_id, int node_no,
                                      mem_fetch* mf) {
  int id = buffer_id + m_m2ndp_config->get_num_m2ndps() * node_no;
  int host_id = node_no / m_num_connections;
  int offset = node_no % m_num_connections;
  int out_id = host_id * m_m2ndp_config->get_links_per_host() +
               buffer_id * m_num_connections + offset;
  id = buffer_id; // TODO: remove this for other topology
  out_id = node_no; // TODO: remove this for other topology
  if (mf->get_addr() != KERNEL_LAUNCH_ADDR && mf->get_from_ndp() && !mf->is_bi()) {
    if (!mf->is_request()) {
      if (mf->get_addr() == 0x90000000000)
        spdlog::info("[{}] : from M2NDP{} to M2NDP{}", mem_access_type_str[mf->get_type()], buffer_id, (int)((mf->get_ndp_id() / m_m2ndp_config->get_num_m2ndps())));
      out_id = m_m2ndp_offset + (int)((mf->get_ndp_id() / m_m2ndp_config->get_num_ndp_units()));
    } else {
      out_id = m_m2ndp_offset + m_m2ndp_config->get_m2ndp_index(mf->get_addr());
    }
  }
  m_from_m2ndp_buffers[id][out_id].push_back(mf);
}

mem_fetch* CxlLink::pop_from_memory_buffer(int buffer_id, int node_no) {
  int id = buffer_id * m_m2ndp_config->get_links_per_m2ndp() + node_no;
  id = buffer_id; // TODO: remove this for other topology
  if (m_to_m2ndp_buffers[id].empty()) return NULL;
  mem_fetch* mf = m_to_m2ndp_buffers[id].front();
  m_to_m2ndp_buffers[id].pop_front();
  return mf;
}

mem_fetch* CxlLink::top_from_memory_buffer(int buffer_id, int node_no) {
  int id = buffer_id * m_m2ndp_config->get_links_per_m2ndp() + node_no;
  id = buffer_id; // TODO: remove this for other topology
  if (m_to_m2ndp_buffers[id].empty()) return NULL;
  return m_to_m2ndp_buffers[id].front();
}

void CxlLink::cycle() {
  if (m_m2ndp_config->is_link_cycle()) {
    handle_from_host();
    handle_to_host();
    handle_from_m2ndp();
    handle_to_m2ndp();
    m_interconnect_interface->Advance();
    m_cycles++;
    int freq = m_m2ndp_config->get_log_interval();
    if (m_cycles % freq == 0) {
      for (int host = 0; host < m_m2ndp_config->get_links_per_host(); host++) {
        for (int ndp = 0; ndp < m_m2ndp_config->get_links_per_m2ndp();
             ndp++) {
          unsigned m2ndp_to_host_flists =
              m_interconnect_interface->GetProcessedFlits(host);
          unsigned host_to_m2ndp_flits =
              m_interconnect_interface->GetProcessedFlits(
                  m_m2ndp_config->get_links_per_host() + ndp);
          spdlog::info(
              "CXL-Link Util:\tFrom host to CXL: {:.2f}% From CXL to host: "
              "{:.2f}%",
              (double)(host_to_m2ndp_flits - prior_host_to_m2ndp_flits) /
                  (double)freq * 100,
              (double)(m2ndp_to_host_flists - prior_m2ndp_to_host_flits) /
                  (double)freq * 100);
          prior_m2ndp_to_host_flits = m2ndp_to_host_flists;
          prior_host_to_m2ndp_flits = host_to_m2ndp_flits;
        }
      }
    }
  }
}

bool CxlLink::is_active() { return m_interconnect_interface->Busy(); }

void CxlLink::display_stats(FILE* fp) {
  m_interconnect_interface->DisplayStats(fp);
}

void CxlLink::print_energy_stats(FILE* fp, const char* name) {
  m_interconnect_interface->print_energy_stats(fp, name);
}

void CxlLink::handle_from_host() {
  for (int i = 0; i < m_host_ports; i++) {
    for (int j = 0; j < m_m2ndp_ports; j++) {
      int n = (to_m2ndp_rr + j) % m_m2ndp_ports;
      if (m_from_host_buffers[i][n].empty()) continue;
      mem_fetch* mf = m_from_host_buffers[i][n].front();
      if (!m_interconnect_interface->HasBuffer(i, mf->get_size())) continue;
      int size = mf->get_size();
      if (mf->is_sc_addr()) 
        size = CXL_OVERHEAD;

      m_interconnect_interface->Push(i, m_m2ndp_offset + n, mf, size);
      m_from_host_buffers[i][n].pop_front();
      break;
    }
  }
  to_m2ndp_rr = (to_m2ndp_rr + 1) % m_m2ndp_ports;
}

void CxlLink::handle_to_host() {
  for (int i = 0; i < m_host_ports; i++) {
    mem_fetch* mf = (mem_fetch*)m_interconnect_interface->Top(i);
    if (mf == NULL) continue;
    if (m_to_host_buffers[i].size() >=
        m_m2ndp_config->get_cxl_link_buffer_size())
      continue;
    if (mf->get_from_dma()) {
      assert(!mf->is_request());
      delete mf;
    } else {
      m_to_host_buffers[i].push_back(mf);
    }
    m_interconnect_interface->Pop(i);
  }
}

void CxlLink::handle_from_m2ndp() {
  for (int i = 0; i < m_m2ndp_ports; i++) {
    for (int j = 0; j < m_num_ports; j++) {
      int n = (to_host_rr + j) % m_num_ports;
      int size = 0;
      if (m_from_m2ndp_buffers[i][n].empty()) continue;
      mem_fetch* mf = m_from_m2ndp_buffers[i][n].front();
      if (!m_interconnect_interface->HasBuffer(m_m2ndp_offset + i,
                                               mf->get_size()))
        continue;
      int input_id = m_m2ndp_offset + i;
      int output_id = n;
      m_interconnect_interface->Push(input_id, output_id, mf, mf->get_size());
      m_from_m2ndp_buffers[i][n].pop_front();
      break;
    }
  }
  to_host_rr = (to_host_rr + 1) % m_host_ports;
}

void CxlLink::handle_to_m2ndp() {
  for (int i = 0; i < m_m2ndp_ports; i++) {
    mem_fetch* mf =
        (mem_fetch*)m_interconnect_interface->Top(m_m2ndp_offset + i);
    if (mf == NULL) continue;
    if (m_to_m2ndp_buffers[i].size() >=
        m_m2ndp_config->get_cxl_link_buffer_size())
      continue;
    m_to_m2ndp_buffers[i].push_back(mf);
    m_interconnect_interface->Pop(m_m2ndp_offset + i);
  }
}
}  // namespace NDPSim
#endif