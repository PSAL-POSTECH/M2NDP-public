/*
  * Library interface for M2NDP module
*/
#ifndef M2NDP_MODULE_H
#define M2NDP_MODULE_H

#include <vector>
#include <stdio.h>
#include <cstdint>

namespace NDPSim {
class M2NDP;
class M2NDPConfig;
class CxlLink;
}  // namespace NDPSim

enum CxlAccessType {
  CXL_READ_REQUEST,
  CXL_READ_RESPONSE,
  CXL_WRITE_REQUEST,
  CXL_WRITE_RESPONSE
};

struct CxlAccess {
  bool from_ndp;
  int host_id;
  int ndp_id;
  uint64_t addr;
  uint64_t size;
  CxlAccessType type;
  void* data;
  uint64_t timestamp;
};

class M2NDPModule {
  public:
    M2NDPModule(int num_gpus, int num_m2ndps, const char* config_file);

    //Write Configuration
    void set_output_file(FILE* fp);

    //Read Cofiguration
    int get_num_ndp();
    int get_links_per_gpu();
    int get_num_ndp_per_switch();
    int get_num_memory_buffers();
    int get_m2ndp_interleave_size();
    
    //M2NDP Operations
    void cycle();
    bool active();
    bool active_for_gpu(int gpu_id);
    bool can_launch_ndp_kernel(int gpu_id);
    void launch_ndp_kernel(int gpu_id, int ndp_id, const char* kernel_path);
    void send_dma_request(uint64_t base_addr, uint64_t size, uint64_t stride);
    void display_stats(FILE* fp);
    //CXL Link Operations
    bool cxl_can_push_from_gpu(int gpu_id, int node_id, CxlAccess req);
    void cxl_push_from_gpu(int gpu_id, int node_id, CxlAccess req);
    bool can_pop_from_gpu(int gpu_id, int node_id);
    CxlAccess cxl_top_from_gpu(int gpu_id, int node_id);
    CxlAccess cxl_pop_from_gpu(int gpu_id, int node_id);

 private:
  NDPSim::M2NDPConfig *m_m2ndp_config;
  NDPSim::CxlLink *m_cxl_link;
  std::vector<NDPSim::M2NDP*> m_m2ndps;
  int addr_to_ndp_id(uint64_t addr);
};

#endif
