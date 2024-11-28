#ifdef ACCELSIM_BUILD
#include "m2ndp_module.h"

#include <string>
#include <spdlog/spdlog.h>
#include <spdlog/cfg/env.h>

#include "common.h"
#include "m2ndp_config.h"
#include "m2ndp.h"
using namespace NDPSim;
M2NDPModule::M2NDPModule(int num_gpus, int num_ndexes, const char* config_file) {
  spdlog::cfg::load_env_levels();
  printf("Creating M2NDP Module\n");
  std::string config_file_str(config_file);
  m_m2ndp_config = new NDPSim::M2NDPConfig(config_file_str, num_gpus);
  m_cxl_link = new NDPSim::CxlLink(m_m2ndp_config);
  m_m2ndps.resize(num_ndexes);
  for (int i = 0; i < num_ndexes; i++) {
    m_m2ndps[i] = new NDPSim::M2NDP(m_m2ndp_config, NULL, i);
    m_m2ndps[i]->set_cxl_link(m_cxl_link);
  }
}

void M2NDPModule::set_output_file(FILE* fp) {
  m_m2ndp_config->set_output_file(fp);
}

int M2NDPModule::get_num_ndp() { return m_m2ndp_config->get_num_ndp_units(); }

int M2NDPModule::get_links_per_gpu() {
  return m_m2ndp_config->get_links_per_gpu();
}

int M2NDPModule::get_num_ndp_per_switch() {
  return m_m2ndp_config->get_ndp_units_per_buffer();
}

int M2NDPModule::get_num_memory_buffers() {
  return m_m2ndp_config->get_num_m2ndps();
}

int M2NDPModule::get_m2ndp_interleave_size() {
  return m_m2ndp_config->get_m2ndp_interleave_size();
}

void M2NDPModule::cycle() {
  m_m2ndp_config->cycle();
  m_cxl_link->cycle();
  for (int i = 0; i < m_m2ndps.size(); i++) {
    m_m2ndps[i]->cycle();
  }
}

bool M2NDPModule::active() {
  bool active = false;
  for (int i = 0; i < m_m2ndps.size(); i++) {
    active = active || m_m2ndps[i]->is_active();
  }
  return active;
}

bool M2NDPModule::active_for_gpu(int gpu_id) {
  bool active = false;
  for (int i = 0; i < m_m2ndps.size(); i++) {
    active = active || m_m2ndps[i]->is_ndp_kernel_active(gpu_id);
  }
  return active;
}

bool M2NDPModule::can_launch_ndp_kernel(int gpu_id) {
  bool can_launch = true;
  for (int i = 0; i < m_m2ndps.size(); i++) {
    can_launch =
        can_launch && m_m2ndps[i]->can_launch_ndp_kernel(gpu_id);
  }
  return can_launch;
}

void M2NDPModule::launch_ndp_kernel(int gpu_id, int m2ndp_id,
                                   const char* kernel_path) {
  std::string kernel_path_str(kernel_path);
  m_m2ndps[m2ndp_id]->ndp_kernel_launch(gpu_id, kernel_path_str, false);
}

void M2NDPModule::send_dma_request(uint64_t base_addr, uint64_t size,
                                  uint64_t stride) {
  spdlog::info("Send DMA Request {:x} {:x} {:x}", base_addr, size, stride);
  int num_ndp_units = m_m2ndp_config->get_num_ndp_units();
  for (int i = 0; i < num_ndp_units; i++) {
    uint64_t dma_req_addr =
        DMA_BASE + i * m_m2ndp_config->get_channel_interleave_size();
    NDPSim::mem_fetch* mf = new NDPSim::mem_fetch(
        dma_req_addr, DMA_ALLOC_W, WRITE_REQUEST, PACKET_SIZE, CXL_OVERHEAD,
        m_m2ndp_config->get_sim_cycle());
    VectorData* data = new VectorData(PACKET_SIZE, 1);
    data->SetData((int64_t)base_addr, 0);
    data->SetData((int64_t)size, 1);
    data->SetData((int64_t)stride, 2);
    mf->set_data(data);
    mf->set_from_dma();
    mf->set_ndp_id(addr_to_ndp_id(dma_req_addr));
    spdlog::info("Send DMA Request {:x} {:x} {:x}", data->GetLongData(0),
                 data->GetLongData(1), data->GetLongData(2));
    m_cxl_link->push_from_gpu(0, i % m_m2ndp_config->get_links_per_gpu(), mf);
  }
}

void M2NDPModule::display_stats(FILE* fp) {
  fprintf(fp, "========== CXL CONFIGS ==========\n");
  m_m2ndp_config->print_config(fp);
  fprintf(fp, "========== CXL LINK STATS ==========\n");
  m_cxl_link->display_stats(fp);
  fprintf(fp, "========== CXL MEMORY BUFFER STATS  ==========\n");
  for (int i = 0; i < m_m2ndps.size(); i++) {
    m_m2ndps[i]->display_stats(fp);
  }

}

bool M2NDPModule::cxl_can_push_from_gpu(int gpu_id, int node_id, CxlAccess req) {
  return m_cxl_link->has_buffer_from_gpu(gpu_id, node_id, NULL);
}

void M2NDPModule::cxl_push_from_gpu(int gpu_id, int node_id, CxlAccess req) {
  NDPSim::mem_fetch* mf;
  if (req.from_ndp &&
      (req.type == CXL_READ_RESPONSE || req.type == CXL_WRITE_RESPONSE)) {
    mf = (NDPSim::mem_fetch*)req.data;
  } else {
    assert(req.from_ndp == false);
    mem_access_type access_type =
        req.type == CXL_READ_REQUEST ? HOST_ACC_R : HOST_ACC_W;
    mf_type type = req.type == CXL_READ_REQUEST ? READ_REQUEST : WRITE_REQUEST;
    mf = new NDPSim::mem_fetch(req.addr, access_type, type, req.size,
                               CXL_OVERHEAD, m_m2ndp_config->get_sim_cycle());
    mf->set_ndp_id(addr_to_ndp_id(req.addr));
    mf->set_data(req.data);
  }
  m_cxl_link->push_from_gpu(gpu_id, node_id, mf);
}

bool M2NDPModule::can_pop_from_gpu(int gpu_id, int node_id) {
  return m_cxl_link->top_from_gpu(gpu_id, node_id) != NULL;
}

CxlAccess M2NDPModule::cxl_top_from_gpu(int gpu_id, int node_id) {
  NDPSim::mem_fetch* mf = m_cxl_link->top_from_gpu(gpu_id, node_id);
  if(mf == NULL) {
    return CxlAccess{false, 0, 0, 0, 0, CXL_READ_REQUEST, NULL};
  }
  if (mf->get_from_ndp()) {
    assert(mf->is_request());
    CxlAccessType type = mf->is_write() ? CXL_WRITE_REQUEST : CXL_READ_REQUEST;
    return CxlAccess{true,
                     mf->get_gpu_id(),
                     mf->get_ndp_id(),
                     mf->get_addr(),
                     mf->get_data_size(),
                     type,
                     mf, 
                     mf->get_timestamp()};
  } else {
    assert(!mf->is_request());
    CxlAccessType type = mf->is_write() ? CXL_WRITE_RESPONSE : CXL_READ_RESPONSE;
    CxlAccess access =
        CxlAccess{false,          mf->get_gpu_id(),    mf->get_ndp_id(),
                  mf->get_addr(), mf->get_data_size(), type,
                  mf->get_data(), mf->get_timestamp()};
    return access;
  }
}

CxlAccess M2NDPModule::cxl_pop_from_gpu(int gpu_id, int node_id) {
  CxlAccess access = cxl_top_from_gpu(gpu_id, node_id);
  NDPSim::mem_fetch* mf = m_cxl_link->pop_from_gpu(gpu_id, node_id);
  if (!mf->get_from_ndp()) {
    delete mf;
  }
  return access;
}

int M2NDPModule::addr_to_ndp_id(uint64_t addr) {
  return m_m2ndp_config->get_channel_index(addr) %
         m_m2ndp_config->get_num_ndp_units();
}
#endif
