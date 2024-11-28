#ifdef TIMING_SIMULATION
#include "mem_fetch.h"
namespace NDPSim {

static unsigned long long unique_uid = 0;

mem_fetch::mem_fetch(new_addr_type addr, mem_access_type acc_type, mf_type type,
                     unsigned data_size, unsigned ctrl_size,
                     unsigned long long timestamp)
    : m_addr(addr),
      m_mem_access_type(acc_type),
      m_type(type),
      m_data_size(data_size),
      m_ctrl_size(ctrl_size),
      m_timestamp(timestamp) {
  m_request_id = unique_uid++;
  m_sector_mask.set(addr % (MAX_MEMORY_ACCESS_SIZE) / MEM_ACCESS_SIZE);
  m_valid = true;
  m_src_id = 0;
  m_atomic = false;
}

mem_fetch::mem_fetch(new_addr_type addr, mem_access_type acc_type, mf_type type,
                     unsigned data_size, unsigned ctrl_size,
                     MemAccessByteMask byte_mask, SectorMask sector_mask,
                     unsigned long long timestamp)
    : m_addr(addr),
      m_mem_access_type(acc_type),
      m_type(type),
      m_data_size(data_size),
      m_ctrl_size(ctrl_size),
      m_byte_mask(byte_mask),
      m_sector_mask(sector_mask),
      m_timestamp(timestamp) {
  m_request_id = unique_uid++;
  m_sector_mask.set(addr % (MAX_MEMORY_ACCESS_SIZE) / MEM_ACCESS_SIZE);
  m_valid = true;
  m_src_id = 0;
  m_atomic = false;
}

mem_fetch::mem_fetch(std::deque<mem_fetch*> wrapped_mf) {
  mem_fetch* mf = wrapped_mf.front();
  m_wrapped_mf = wrapped_mf;
  m_type = mf->get_type();
  m_addr = mf->get_addr();
  m_src_id = mf->get_host_id();
  m_mem_access_type = mf->get_access_type();
}

void mem_fetch::set_reply() {
  if (m_type == mf_type::READ_REQUEST)
    m_type = mf_type::READ_REPLY;
  else if (m_type == mf_type::WRITE_REQUEST)
    m_type = mf_type::WRITE_ACK;
  else
    assert(0);
}

void mem_fetch::convert_to_write_request(new_addr_type addr) {
  assert(m_type == mf_type::READ_REPLY);
  m_type = mf_type::WRITE_REQUEST;
  m_addr = addr;
}

bool mem_fetch::is_write() {
  return m_type == mf_type::WRITE_REQUEST || m_type == mf_type::WRITE_ACK;
}

bool mem_fetch::is_request() {
  return m_type == mf_type::READ_REQUEST || m_type == mf_type::WRITE_REQUEST;
}
}

#endif
