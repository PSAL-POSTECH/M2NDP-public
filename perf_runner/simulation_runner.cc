#include "simulation_runner.h"

#include <fstream>
#include <iostream>
#include <string>

#include "command_line_parser.h"
namespace po = boost::program_options;
namespace NDPSim {
SimulationRunner::SimulationRunner(int argc, char* argv[])
    : remaing_memory_reqs(0), m_memory_map(NULL), m_target_map(NULL) {
  CommandLineParser cmd_parser = CommandLineParser();
  cmd_parser.add_command_line_option<std::string>("config",
                                                  "path for m2ndp config file");
  cmd_parser.add_command_line_option<std::string>(
      "trace", "path for ndp trace file directory");
  cmd_parser.add_command_line_option<int>("num_hosts",
                                          "number of hosts in the system");
  cmd_parser.add_command_line_option<int>(
      "num_m2ndps", "number of ndp switches in the system");
  cmd_parser.add_command_line_option<std::string>("output", "output filename");
  cmd_parser.add_command_line_option<bool>("synthetic_memory",
                                                "use synthetic memory");
  cmd_parser.add_command_line_option<bool>("serial_launch",
                                                "launch single kernel at a time");
  try {
    cmd_parser.parse(argc, argv);
  } catch (const CommandLineParser::ParsingError& e) {
    spdlog::error(
        "Command line argument parrsing error captured. Error message: {}",
        e.what());
    throw(e);
  }
  m_output_filename = "";
  m_serial_launch = false;
  m_use_synthetic_memory = false;
  cmd_parser.set_if_defined("config", &m_config_file_path);
  cmd_parser.set_if_defined("trace", &m_trace_dir_path);
  cmd_parser.set_if_defined("num_hosts", &m_num_hosts);
  cmd_parser.set_if_defined("num_m2ndps", &m_num_m2ndps);
  cmd_parser.set_if_defined("output", &m_output_filename);
  cmd_parser.set_if_defined("synthetic_memory", &m_use_synthetic_memory);
  cmd_parser.set_if_defined("serial_launch", &m_serial_launch);
  m_memory_reqs.resize(m_num_hosts);
  m_m2ndp_config = new M2NDPConfig(m_config_file_path, m_num_hosts);
  m_m2ndps.resize(m_num_m2ndps);
  m_cxl_link = new CxlLink(m_m2ndp_config);
  m_m2ndp_config->set_use_synthetic_memory(m_use_synthetic_memory);
  m_output_file = fopen(
      m_output_filename.empty() ? "output.out" : m_output_filename.c_str(),
      "w");
  m_energy_file =
      fopen(m_output_filename.empty() ? "energy.out"
                                      : ("energy_" + m_output_filename).c_str(),
            "w");
  m_m2ndp_config->set_output_file(m_output_file);
  m_m2ndp_config->print_config(m_output_file);
  parse_ndp_trace();
  for (int i = 0; i < m_num_m2ndps; i++) {
    m_m2ndps[i] = new M2NDP(m_m2ndp_config, m_memory_map, i);
    m_m2ndps[i]->set_cxl_link(m_cxl_link);
  }
}

void SimulationRunner::run() {
  extern Stats_NDPSim::StatList statlist;

  NdpCommand running_command;
  remaing_memory_reqs = 0;
  while (!check_all_simulaiton_finished()) {
    // printf("remaing_memory_reqs = %d\n", remaing_memory_reqs);

    m_m2ndp_config->cycle();
    m_cxl_link->cycle();
    for (int i = 0; i < m_num_m2ndps; i++) {
      m_m2ndps[i]->cycle();
    }
    if(!m_bi_reqs.empty() &&m_bi_reqs.front().second >= m_m2ndp_config->get_ndp_cycle()) {
      mem_fetch* mf = m_bi_reqs.front().first;
      m_bi_reqs.pop();
      m_memory_reqs[0].push(mf);
    }
    if (check_single_simulation_finished()) {
      if (!m_ndp_commands.empty()) {
        NdpCommand command = m_ndp_commands.front();
        if (command.is_barrier) {
          if (!check_ndp_kenrel_active() && remaing_memory_reqs == 0 &&
              check_all_memory_reqs_empty()) {
            printf("NDP barrier popped\n");
            m_ndp_commands.pop();
          }
        }
        if (check_can_launch_ndp_kernel()) {
          launch_ndp_kernel(command);
          m_ndp_commands.pop();
        }
      }
    }
    process_memory_access();
  }
  fprintf(m_output_file, "========== CXL LINK STATS ==========\n");
  m_cxl_link->display_stats(m_output_file);
  m_cxl_link->print_energy_stats(m_energy_file, "LINK");
  fprintf(m_output_file, "========== CXL MEMORY BUFFER STATS  ==========\n");
  for (auto m2ndp : m_m2ndps) {
    m2ndp->display_stats(m_output_file);
    m2ndp->print_energy_stats(m_energy_file);
  }
  Stats_NDPSim::statlist.printall();
  spdlog::info("CXL BUFFER FINISHED");
  spdlog::info("EXPR FINISHED {}", m_m2ndp_config->get_sim_cycle());

  if (m_m2ndp_config->is_functional_sim()) match_memorymap();
  fclose(m_output_file);
}

bool SimulationRunner::check_all_simulaiton_finished() {
  bool running = remaing_memory_reqs > 0 || !check_all_memory_reqs_empty() ||
                 !m_ndp_commands.empty();

  running = running || m_cxl_link->is_active();
  for (int i = 0; i < m_num_m2ndps; i++) {
    running = running || m_m2ndps[i]->is_active();
  }
  return !running;
}

bool SimulationRunner::check_single_simulation_finished() {
  bool running = remaing_memory_reqs > 0 || !check_all_memory_reqs_empty();

  running = running || m_cxl_link->is_active();
  for (int i = 0; i < m_num_m2ndps; i++) {
    running = running || m_m2ndps[i]->is_active();
  }
  return !running;
}

bool SimulationRunner::check_can_launch_ndp_kernel() {
  bool can_launch = true;
  for (auto m2ndp : m_m2ndps) {
    can_launch = can_launch && m2ndp->can_register_ndp_kernel(0);
  }
  return can_launch;
}

void SimulationRunner::launch_ndp_kernel(NdpCommand command) {
  spdlog::info("Launching NDP kernel: {} at cycle {}", command.ndp_kernel_path,
               m_m2ndp_config->get_sim_cycle());
  for (auto m2ndp : m_m2ndps) {
    m2ndp->register_ndp_kernel(0, command.ndp_kernel_path);
  }
  m_memory_reqs = command.memory_reqs;
  assert(m_memory_reqs[0].size() > 0);
}

bool SimulationRunner::check_ndp_kenrel_active() {
  bool active = false;
  for (auto m2ndp : m_m2ndps) {
    active = active || m2ndp->is_launch_active(0);
  }
  return active;
}

bool SimulationRunner::check_all_memory_reqs_empty() {
  bool empty = true;
  for (auto req : m_memory_reqs) {
    empty = empty && req.empty();
  }
  return empty;
}

void SimulationRunner::process_memory_access() {
  assert(m_m2ndp_config->get_links_per_host() > 0);
  for (int host = 0; host < m_num_hosts; host++) {
    for (int node_id = 0; node_id < m_m2ndp_config->get_links_per_host();
         node_id++) {
      bool is_kernel_launch = false;
      if(!m_memory_reqs[0].empty()) {
        is_kernel_launch = m_memory_reqs[0].front()->get_addr() == KERNEL_LAUNCH_ADDR;
      }
      bool serial_execution_check 
        = (!m_serial_launch || !is_kernel_launch || !(check_ndp_kenrel_active() || remaing_memory_reqs > 0));
      if (!m_memory_reqs[host].empty() && serial_execution_check) {
        mem_fetch* mf = m_memory_reqs[host].front();
        if (m_cxl_link->has_buffer_from_host(host, node_id, mf)) {
          m_cxl_link->push_from_host(host, node_id, mf);
          m_memory_reqs[host].pop();
          if(!mf->is_bi() && !mf->is_bi_writeback() && !mf->is_uthread_request())
            remaing_memory_reqs++;
        }
      }
      if (m_cxl_link->top_from_host(host, node_id) != NULL) {
        mem_fetch* mf = m_cxl_link->top_from_host(host, node_id);
        if(!mf->is_request()) {
          assert(!mf->get_from_ndp() && !mf->get_from_dma());
          m_cxl_link->pop_from_host(host, node_id);
          remaing_memory_reqs--;
          if (mf->is_write()) {
            KernelLaunchInfo* data = (KernelLaunchInfo*)mf->get_data();
            delete data;
          }
          delete mf;
        }
        else {
          mf->current_state = "BI handled";
          mf->set_reply();
          mf->set_data((void*) 0x77);
          if(!mf->is_write()) {
            mf->convert_to_write_request(mf->get_addr());
            mf->unset_bi();
            mf->set_bi_writeack();
          }
          m_bi_reqs.push(std::make_pair(mf, m_m2ndp_config->get_ndp_cycle() + m_m2ndp_config->get_bi_host_latency()));
          m_cxl_link->pop_from_host(host, node_id);
        }
        
      }
    }
  }
}

void SimulationRunner::parse_ndp_trace() {
  std::ifstream ifs(m_trace_dir_path + "/kernelslist.g");
  std::vector<std::string> kernel_names;
  std::string line;
  while (ifs >> line) {
    kernel_names.push_back(line);
  }
  ifs.close();
  int cnt = 0;
  int num_kernels = kernel_names.size();
  spdlog::info("Launching {} NDP kernels", kernel_names.size());
  for (auto kernel_name : kernel_names) {
    std::string file_name;
    file_name = kernel_name;
    spdlog::debug("Parsing trace...{}", cnt++);
    if (m_m2ndp_config->is_functional_sim() && cnt == 1 ) { 
      //make input memory map at first kernel
      std::string input_memory_path =
          m_trace_dir_path + "/" + file_name + "_input.data";
      m_memory_map = new HashMemoryMap(input_memory_path);
      if (m_m2ndp_config->get_use_synthetic_memory())
        m_memory_map->set_synthetic_memory(
            m_m2ndp_config->get_synthetic_base_address(),
            m_m2ndp_config->get_synthetic_memory_size());
    }
    if (m_m2ndp_config->is_functional_sim() && cnt == num_kernels) {
      //make target memory map at last kernel
      std::string target_memory_path =
          m_trace_dir_path + "/" + file_name + "_output.data";
      m_target_map = new HashMemoryMap(target_memory_path);
    }
    NdpCommand command = NdpCommand{
        .ndp_kernel_path = m_trace_dir_path + "/" + kernel_name + ".traceg",
        .is_barrier = false};
    std::ifstream ifs_data(m_trace_dir_path + "/" + kernel_name +
                           "_launch.txt");
    assert(ifs_data.is_open());
    std::string line;
    spdlog::info("Parsing NDP kernel: {}", kernel_name);

    while (std::getline(ifs_data, line)) {
      if (line.find("#") != std::string::npos) continue;
      fill_memory_access(command, line);
      spdlog::debug("Memory access filled");
    }
    m_ndp_commands.push(command);
  }
}

void SimulationRunner::fill_memory_access(NdpCommand& command,
                                          std::string line) {
  command.memory_reqs.resize(m_num_hosts);
  int host_id = 0;
  KernelLaunchInfo* info = M2NDPParser::parse_kernel_launch(line, host_id);
  for (int i = 0; i < m_m2ndp_config->get_num_m2ndps(); i++) {
    uint64_t addr = KERNEL_LAUNCH_ADDR;
    mem_fetch* mf = new mem_fetch(addr, DMA_ALLOC_W, WRITE_REQUEST, PACKET_SIZE,
                                  CXL_OVERHEAD, 0);
    mf->set_filtering();
    mf->set_from_ndp(false);
    mf->set_data(info);
    mf->set_channel(m_m2ndp_config->get_channel_index(addr));
    command.memory_reqs[host_id].push(mf);

    if(m_m2ndp_config->is_sc_work()) {
      generate_uthresds(command, info->base_addr, info->size);
    }
  }
}

void SimulationRunner::match_memorymap() {
  if (m_target_map->Match(*m_memory_map)) {
    spdlog::info("MEMROY MATCH SUCCESS\n");
  } else {
    spdlog::info("MEMROY MATCH FAIL\n");
  }
}

void SimulationRunner::generate_uthresds(NdpCommand& command, uint64_t base_addr, uint64_t size) {
  int iter = 1;
  if (command.ndp_kernel_path.find("fc") != std::string::npos) {
    iter = 2;
  }
  if (command.ndp_kernel_path.find("proj") != std::string::npos) {
    iter = 2;
  }
  if (command.ndp_kernel_path.find("attention1") != std::string::npos) {
    iter = 3;
  }
  for (int i = 0; i < iter ; i++) {
    for (uint64_t offset = 0; offset < size; offset+=PACKET_SIZE) {
      mem_fetch* mf = new mem_fetch(base_addr + offset, DMA_ALLOC_W, WRITE_REQUEST, PACKET_SIZE,
                                    CXL_OVERHEAD, 0);
      mf->set_filtering();
      mf->set_from_ndp(false);
      mf->set_uthread_request();
      mf->set_channel(m_m2ndp_config->get_channel_index(base_addr + offset));
      command.memory_reqs[0].push(mf);
    }
  }
  printf("command mem size %d\n", command.memory_reqs[0].size());
}
}  // namespace NDPSim