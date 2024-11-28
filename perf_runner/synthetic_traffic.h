#ifndef SYNTHETIC_TRAFFIC_RUNNER_H
#define SYNTHETIC_TRAFFIC_RUNNER_H

#include "cxl_link.h"
#include "m2ndp_config.h"
#include "m2ndp_switch.h"

namespace NDPSim {
class SyntheticTrafficRunner {
  public: 
    SyntheticTrafficRunner(int argc, char* argv[]);
    void run();
    void match_memorymap() {}
  private:
    std::string m_config_file_path;
    int m_num_reqs;
    M2NDPConfig* m_m2ndp_config;
    CxlLink* m_cxl_link;
    M2NDP *m_m2ndp;
    std::vector<std::queue<mem_fetch*>> m_memory_reqs;
};
} // namespace NDPSim
#endif