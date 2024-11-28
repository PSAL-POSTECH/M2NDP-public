#include "synthetic_traffic.h"

#include "command_line_parser.h"

namespace NDPSim {
SyntheticTrafficRunner::SyntheticTrafficRunner(int argc, char* argv[]) {
  CommandLineParser cmd_parser = CommandLineParser();
  cmd_parser.add_command_line_option<std::string>("config",
                                                  "path for m2ndp config file");
  cmd_parser.add_command_line_option<int>(
      "num_reqs", "number of memory request");
  try {
    cmd_parser.parse(argc, argv);
  } catch (const CommandLineParser::ParsingError& e) {
    spdlog::error(
        "Command line argument parrsing error captured. Error message: {}",
        e.what());
    throw(e);
  }
  cmd_parser.set_if_defined("config", &m_config_file_path);
  cmd_parser.set_if_defined("num_reqs", &m_num_reqs);

  m_m2ndp_config = new M2NDPConfig(m_config_file_path, 1);
  m_m2ndp_config->set_functional_sim(false);
  m_cxl_link = new CxlLink(m_m2ndp_config);
  m_m2ndp = new M2NDP(m_m2ndp_config, NULL, 0);
  m_m2ndp->set_cxl_link(m_cxl_link);
}

void SyntheticTrafficRunner::run() {
  extern Stats_NDPSim::StatList statlist;
  FILE *output_file = fopen("output.txt", "w");
  int finish_counter = 0;
  bool push_mem_req = true;
  uint64_t push_cycle = 0;
  while (finish_counter < m_num_reqs) {
    m_m2ndp_config->cycle();
    m_cxl_link->cycle();
    m_m2ndp->cycle();
    if(push_mem_req ) {
      uint64_t random_addr = rand() % 100000000;
      mem_fetch *mf = new mem_fetch(random_addr, GLOBAL_ACC_R, READ_REQUEST,
        PACKET_SIZE, CXL_OVERHEAD, 0);
      mf->set_channel(m_m2ndp_config->get_channel_index(random_addr));
      printf("Push to CXL link time: %d\n", m_m2ndp_config->get_ndp_cycle());
      m_cxl_link->push_from_host(0, 0, mf);
      push_cycle = m_m2ndp_config->get_ndp_cycle();
      push_mem_req = false;
    }
    if (m_cxl_link->top_from_host(0, 0) != NULL) {
      mem_fetch* mf = m_cxl_link->top_from_host(0, 0);
      m_cxl_link->pop_from_host(0, 0);
      delete mf;
      uint64_t req_cycle = m_m2ndp_config->get_ndp_cycle() - push_cycle;
      printf("Pop CXL link time: %d\n", m_m2ndp_config->get_ndp_cycle());
      spdlog::info("Memory request latency: {}", req_cycle);
      push_mem_req = true;
      finish_counter++;
    }
  }
  fprintf(output_file, "========== CXL LINK STATS ==========\n");
  m_cxl_link->display_stats(output_file);
  fprintf(output_file, "========== CXL MEMORY BUFFER STATS  ==========\n");
  m_m2ndp->display_stats(output_file);
  Stats_NDPSim::statlist.printall();
}
} // namespace NDPSim