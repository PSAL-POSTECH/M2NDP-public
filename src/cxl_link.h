#ifdef TIMING_SIMULATION
#ifndef CXL_LINK
#define CXL_LINK

#include <mutex>
#include <string>
#include "interconnect_interface.hpp"
#include "m2ndp_config.h"
#include "mem_fetch.h"
#include <random>
namespace NDPSim {

class M2NDPConfig;
class CxlLink {
  public:
    CxlLink(){}
    CxlLink(M2NDPConfig *m2ndp_config);    
    bool has_buffer_from_host(int host_id, int node_no, mem_fetch *mf);
    bool has_buffer_from_memory_buffer(int host_id, int node_no, mem_fetch *mf);

    void push_from_host(int host_id, int node_no, mem_fetch* mf);
    mem_fetch* pop_from_host(int host_id, int node_no);
    mem_fetch* top_from_host(int host_id, int node_no);
    
    void push_from_memory_buffer(int buffer_id, int node_no, mem_fetch* mf);
    mem_fetch* top_from_memory_buffer(int buffer_id, int node_no);
    mem_fetch* pop_from_memory_buffer(int buffer_id, int node_no);
    
    void cycle();

    
    void display_stats(FILE *fp);
    void print_energy_stats(FILE *fp, const char* name);
    M2NDPConfig* get_m2ndp_config();
    int get_links_per_host();
    int get_num_hosts();
    int get_num_memory_buffers();
    int get_packet_size();
    int get_m2ndp_interleave_size();
    bool get_compression();
    bool is_active();

  private:
    InterconnectInterface *m_interconnect_interface;
    M2NDPConfig *m_m2ndp_config;
    int m_m2ndp_offset;
    uint64_t m_cycles = 0;
    int m_host_ports;
    int m_m2ndp_ports;
    int m_num_ports;
    int m_num_connections;
    std::vector<std::vector<std::deque<mem_fetch*>>> m_from_host_buffers;
    std::vector<std::deque<mem_fetch*>> m_to_host_buffers;
    std::vector<std::vector<std::deque<mem_fetch*>>> m_from_m2ndp_buffers;
    std::vector<std::deque<mem_fetch*>> m_to_m2ndp_buffers;
    uint32_t prior_m2ndp_to_host_flits = 0;
    uint32_t prior_host_to_m2ndp_flits = 0;
    int to_m2ndp_rr = 0;
    int to_host_rr = 0;
    void handle_from_host();
    void handle_to_host();
    void handle_from_m2ndp();
    void handle_to_m2ndp();
};
}

#endif
#endif