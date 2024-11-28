#ifdef TIMING_SIMULATION
#include "uthread_generator.h"

#include "memory_map.h"
namespace NDPSim {

static unsigned long long global_req_id = 0;

UThreadGenerator::UThreadGenerator(M2NDPConfig* config, int ndp_id,
                                   fifo_pipeline<RequestInfo>* request_queue)
    : m_ndp_id(ndp_id),
      m_uthread_request_queue(request_queue) {
  m_config = config;
}

void UThreadGenerator::register_kernel(NdpKernel* kernel) {
  m_registered_functions.insert(kernel->kernel_id);
  m_num_kernel_bodies[kernel->kernel_id] = kernel->num_kernel_bodies;
  spdlog::info("NDP {} : Register kernel {}", m_ndp_id, kernel->kernel_id);
}

void UThreadGenerator::unregister_kernel(int kernel_id) {
  m_registered_functions.erase(kernel_id);
  m_num_kernel_bodies.erase(kernel_id);
}

bool UThreadGenerator::is_active() { return !m_launch_infos.empty(); }

bool UThreadGenerator::is_launch_active(int launch_id) {
  return m_launch_infos.find(launch_id) != m_launch_infos.end();
}

void UThreadGenerator::print_status() {
  for (auto info_map : m_launch_infos) {
    KernelLaunchInfo info = info_map.second;
    int count = m_count_requests[info.launch_id];
    int total = m_total_requests[info.launch_id];
    if (m_ndp_id == 0)
      spdlog::info("NDP {} Launch ID {} {}:  base {:x}, size {} count {}/{}",
                   m_ndp_id, info.launch_id, info.kernel_name, info.base_addr,
                   info.size, count, total);
    else
      spdlog::debug("NDP {} Launch ID {} {}:  base {:x}, size {} count {}/{}",
                    m_ndp_id, info.launch_id, info.kernel_name, info.base_addr,
                    info.size, count, total);
  }
}

void UThreadGenerator::launch(KernelLaunchInfo kinfo) {
  uint64_t base = kinfo.base_addr;
  uint64_t size = kinfo.size;
  uint64_t kernel_id = kinfo.kernel_id;
  int launch_id = kinfo.launch_id;
  m_launch_infos[kinfo.launch_id] = kinfo;
  assert(kinfo.smem_size <= m_config->get_spad_size()); // SMEM size should be less than SPAD size
  m_launch_infos[kinfo.launch_id].scratchpad_map = new HashMemoryMap(SCRATCHPAD_BASE, kinfo.smem_size);
  assert(size % PACKET_SIZE == 0);
  bool initializer = true;
  std::deque<RequestInfo*> requests;
  if (m_registered_functions.find(kernel_id) == m_registered_functions.end()) {
    spdlog::warn("NDP {} : Function {} is not registered", m_ndp_id, kernel_id);
    return;
  }
  spdlog::info("NDP {} Kernel launch for function {} base {:x} size {}",
               m_ndp_id, kernel_id, base, size);
  uint64_t ndp_offset = 0;
  uint64_t spad_addr = SCRATCHPAD_BASE; 
  int num_args = kinfo.arg_size / DOUBLE_SIZE - kinfo.num_float_args;
  int num_args_per_packet = PACKET_SIZE / DOUBLE_SIZE;
  int outter = 0;
  for (outter = 0; outter < (num_args - 1) / num_args_per_packet + 1; outter++) {
    VectorData data(64,1);
    data.SetType(INT64);
    for (int inner = 0; inner < num_args_per_packet; inner++) {
      int offset = outter * num_args_per_packet + inner;
      if (offset >= num_args) break;
      data.SetData((int64_t)kinfo.args[offset], inner);
      spdlog::debug("NDP {} : Scratchpad {:x} {:x}", m_ndp_id, spad_addr,
                    kinfo.args[offset]);
    }
    m_launch_infos[kinfo.launch_id].scratchpad_map->Store(
        spad_addr + outter * PACKET_SIZE, data);
  }
  if (kinfo.num_float_args > 0) {
    assert(kinfo.num_float_args <= MAX_NR_FLOAT_ARG);
    VectorData data(32,1);
    data.SetType(FLOAT32);
    for (int float_idx = 0; float_idx < kinfo.num_float_args; float_idx++) {
      data.SetData((float)kinfo.float_args[float_idx], float_idx);
      spdlog::debug("NDP {} : Scratchpad {:x} {:x}", m_ndp_id,
                    spad_addr + outter * PACKET_SIZE,
                    kinfo.float_args[float_idx]);
    }
    m_launch_infos[kinfo.launch_id].scratchpad_map->Store(
        spad_addr + outter * PACKET_SIZE, data);
  }
  
  for (uint64_t addr = base; addr < base + size; addr += PACKET_SIZE) {
    if (!check_addr_match(addr)) continue;
    if (initializer) {
      RequestInfo* info = new RequestInfo();
      info->kernel_id = kernel_id;
      info->launch_id = launch_id;
      info->id = global_req_id++;
      info->addr = base;
      info->offset = m_ndp_id;
      info->type = INITIALIZER;
      info->scratchpad_map = m_launch_infos[kinfo.launch_id].scratchpad_map;
      requests.push_back(info);
      initializer = false;
    }
    RequestInfo* info = new RequestInfo();
    info->kernel_id = kernel_id;
    info->id = global_req_id++;
    info->launch_id = launch_id;
    info->addr = addr;
    info->offset = addr - base;
    info->type = KERNEL_BODY;
    info->kernel_body_id = 0;
    info->scratchpad_map = m_launch_infos[kinfo.launch_id].scratchpad_map;
    requests.push_back(info);
  }
  for(int kid = 1; kid < m_num_kernel_bodies[kernel_id]; kid++) {
    for (uint64_t addr = base; addr < base + size; addr += PACKET_SIZE) {
      if (!check_addr_match(addr)) continue;
      RequestInfo* info = new RequestInfo();
      info->kernel_id = kernel_id;
      info->id = global_req_id++;
      info->launch_id = launch_id;
      info->addr = addr;
      info->offset = addr - base;
      info->type = KERNEL_BODY;
      info->kernel_body_id = kid;
      info->scratchpad_map = m_launch_infos[kinfo.launch_id].scratchpad_map;
      requests.push_back(info);
    }
  }
  if (requests.size() > 0) {
    RequestInfo* info = new RequestInfo();
    info->kernel_id = kernel_id;
    info->id = global_req_id++;
    info->launch_id = launch_id;
    info->addr = base;
    info->offset = m_ndp_id;
    info->type = FINALIZER;
    info->scratchpad_map = m_launch_infos[kinfo.launch_id].scratchpad_map;
    requests.push_back(info);
  }
  m_total_requests[launch_id] = requests.size();
  m_count_requests[launch_id] = 0;
  if (!requests.empty()) {
    m_launch_queue.push_back(launch_id);
    m_generated_requests[launch_id] = requests;
  } 
  else {
    finish_launch(launch_id);
  } 
  check_kernel_launch();
}

void UThreadGenerator::increase_count(int launch_id) {
  m_count_requests[launch_id] += 1;
}

void UThreadGenerator::finish_launch(int launch_id) {
  m_active_launch_ids.erase(launch_id);
  m_total_requests.erase(launch_id);
  m_count_requests.erase(launch_id);
  delete m_launch_infos[launch_id].scratchpad_map;
  m_launch_infos.erase(launch_id);
  spdlog::debug("NDP {} : Finish kernel launch {}", m_ndp_id, launch_id);
}

void UThreadGenerator::cycle() {
  check_kernel_launch();
  if(!m_config->is_coarse_grained() && !m_config->is_sc_work()) {
    generate_uthreads(1);
  }
  if (!m_active_launch_ids.empty() && !m_generated_requests.empty()) {
      int launch_id = get_next_launch_id();
      if(launch_id == -1) return; // no available launch id
      if (m_generated_requests[launch_id].empty()) {
        m_generated_requests.erase(launch_id);
        return;
      }
      RequestInfo* info = m_generated_requests[launch_id].front();
      if (info->type == FINALIZER) {
        //have to check KERNEL_BODY requests are done at all sub-core
        if (m_count_requests[launch_id] == (m_total_requests[launch_id] - 1)) {
          spdlog::debug("NDP {}: Push to uthread_requests_queue as all KERNEL_BODY is done", m_ndp_id);
          m_uthread_request_queue->push(info);
          m_generated_requests[launch_id].pop_front();
        }
      } else if (info->type == INITIALIZER){
        //INITIALIZER
        m_uthread_request_queue->push(info);
        m_generated_requests[launch_id].pop_front();
      }
    }
  m_launched = false;
}

bool UThreadGenerator::generate_uthreads(int threads) {
  bool can_issue = false;
  for (int i = 0; i < threads; i++) {
    if (!m_active_launch_ids.empty() && !m_generated_requests.empty()) {
      int launch_id = get_next_launch_id();
      if(launch_id == -1) { return false;} // no available launch id
      if (m_generated_requests[launch_id].empty()) {
        m_generated_requests.erase(launch_id);
        return false;
      }
      RequestInfo* info = m_generated_requests[launch_id].front();
      if (info->type == KERNEL_BODY) {
        //have to check INITIALIZER request is done at sub-core
        if (m_count_requests[launch_id] >= 1) {
          m_uthread_request_queue->push(info);
          m_generated_requests[launch_id].pop_front();
          can_issue = true;
        }
      }
    }
  }
  return can_issue;
}

void UThreadGenerator::check_kernel_launch() {
  if (!m_launch_queue.empty() && !m_launched) {
    int launch_id = m_launch_queue.front();
    if (m_launch_infos[launch_id].smem_size + get_allocated_spad_size() <= m_config->get_spad_size() - 8192
      && m_active_launch_ids.size() < m_config->get_max_kernel_launch()) {
      m_active_launch_ids.insert(launch_id);
      m_launch_queue.pop_front();
      m_launched = true;
      if(m_ndp_id == 0)
        spdlog::info("NDP {} : Launching kernel {}, total active kernels {}", m_ndp_id,
                   m_launch_infos[launch_id].kernel_name, m_active_launch_ids.size());
    }
  }
}

bool UThreadGenerator::check_addr_match(uint64_t addr) {
  return m_config->get_matched_unit_id(addr) == (m_ndp_id % m_config->get_num_ndp_units()); // 256B aligned
}

uint32_t UThreadGenerator::get_allocated_spad_size() {
  uint32_t size = 0;
  for (auto id : m_active_launch_ids) {
    KernelLaunchInfo info = m_launch_infos[id];
    size += info.smem_size;
  }
  return size;
}
int UThreadGenerator::get_next_launch_id() {
  for(auto active_launch_id : m_active_launch_ids) {
    if (m_generated_requests.find(active_launch_id) != m_generated_requests.end()) {
      if(m_generated_requests[active_launch_id].empty()) continue;
      RequestInfo *info = m_generated_requests[active_launch_id].front();
      if(check_can_issue(info)) return active_launch_id;
    }
  }
  return -1;
}

bool UThreadGenerator::check_can_issue(RequestInfo* info) {
  
  if(info->type == INITIALIZER) return true;
  else if(info->type == KERNEL_BODY) {
    if(m_count_requests[info->launch_id] == 0) return false;
    int reqs_per_body = (m_total_requests[info->launch_id] - 2) / m_num_kernel_bodies[info->kernel_id] ;
    int after_initializer = m_count_requests[info->launch_id] -1;
    if (info->kernel_body_id == after_initializer / reqs_per_body) {
      return true;
    }
      
    else return false;
  }
  else if(info->type == FINALIZER) {
    if(m_count_requests[info->launch_id] == (m_total_requests[info->launch_id] - 1)) return true;
    else return false;
  }
  else return false;
}
}  // namespace NDPSim

#endif