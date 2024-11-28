#include "scalability_runner.h"

#include <fstream>
#include <iostream>
#include <string>

#include "command_line_parser.h"
namespace po = boost::program_options;
namespace NDPSim {
ScalabilityRunner::ScalabilityRunner(int argc, char* argv[])
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
  try {
    cmd_parser.parse(argc, argv);
  } catch (const CommandLineParser::ParsingError& e) {
    spdlog::error(
        "Command line argument parrsing error captured. Error message: {}",
        e.what());
    throw(e);
  }
  m_output_filename = "";
  m_use_synthetic_memory = false;
  cmd_parser.set_if_defined("config", &m_config_file_path);
  cmd_parser.set_if_defined("trace", &m_trace_dir_path);
  cmd_parser.set_if_defined("num_hosts", &m_num_hosts);
  cmd_parser.set_if_defined("num_m2ndps", &m_num_m2ndps);
  cmd_parser.set_if_defined("output", &m_output_filename);
  cmd_parser.set_if_defined("synthetic_memory", &m_use_synthetic_memory);
  m_memory_reqs.resize(m_num_hosts);
  m_m2ndp_config = new M2NDPConfig(m_config_file_path, m_num_hosts);
  m_m2ndps.resize(m_num_m2ndps);
  m_cxl_link = new CxlLink(m_m2ndp_config);
  m_memory_map.resize(m_num_m2ndps);
  m_target_map.resize(m_num_m2ndps);
  m_ndp_commands.resize(m_num_m2ndps);

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
  for (int i = 0; i < m_num_m2ndps; i++) {
    spdlog::info("Parsing NDP trace...{}", i);
    parse_ndp_trace(i);
    m_m2ndps[i] = new M2NDP(m_m2ndp_config, m_memory_map.at(i), i);
    m_m2ndps[i]->set_cxl_link(m_cxl_link);
  }
}

void ScalabilityRunner::run() {
  extern Stats_NDPSim::StatList statlist;

  NdpCommand running_command;
  remaing_memory_reqs = 0;
  while (!check_all_simulaiton_finished()) {
    // printf("remaing_memory_reqs = %d\n", remaing_memory_reqs);

    m_m2ndp_config->cycle();
    m_cxl_link->cycle();
    for (int i = 0; i < m_num_m2ndps; i++) {
      m_m2ndps[i]->cycle();
      if (!m_m2ndps[i]->is_active()) {
        if (!m_ndp_commands[i].empty()) {
          NdpCommand command = m_ndp_commands[i].front();
          if (command.is_barrier) {
            if (!check_ndp_kenrel_active() && remaing_memory_reqs == 0 &&
                check_all_memory_reqs_empty()) {
              printf("NDP barrier popped\n");
              m_ndp_commands[i].pop();
            }
          }
          if (check_can_launch_ndp_kernel(i)) {
            launch_ndp_kernel(command, i);
            m_ndp_commands[i].pop();
          }
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
  fprintf(m_output_file, "========== M2NDP ACCESS TIME ==========\n");
  for (int i = 0; i < m_num_m2ndps; i++) {
    M2NDPSwitch* m2ndp = m_m2ndps[i];
    fprintf(m_output_file, "[M2NDP %d]\n", i);
    m2ndp->print_access_time(m_output_file);
  }
  Stats_NDPSim::statlist.printall();
  spdlog::info("CXL BUFFER FINISHED");
  spdlog::info("EXPR FINISHED {} NDP CYCLE: {}", m_m2ndp_config->get_sim_cycle(), m_m2ndp_config->get_ndp_cycle());

  if (m_m2ndp_config->is_functional_sim()) match_memorymap();
  fclose(m_output_file);
}

bool ScalabilityRunner::check_all_simulaiton_finished() {
  bool running = remaing_memory_reqs > 0 || !check_all_memory_reqs_empty();
  for (int i = 0; i < m_num_m2ndps; i++) {
    running = running || !m_ndp_commands[i].empty();
  }

  running = running || m_cxl_link->is_active();
  for (int i = 0; i < m_num_m2ndps; i++) {
    running = running || m_m2ndps[i]->is_active();
  }
  return !running;
}

bool ScalabilityRunner::check_single_simulation_finished(int buffer_id) {
  bool running = remaing_memory_reqs > 0 || !check_all_memory_reqs_empty();

  running = running || m_cxl_link->is_active();
  running = running || m_m2ndps[buffer_id]->is_active();
  return !running;
}

bool ScalabilityRunner::check_can_launch_ndp_kernel(int buffer_id) {
  // bool can_launch = true;
  // for (auto m2ndp : m_m2ndps) {
  //   can_launch = can_launch && m2ndp->can_register_ndp_kernel(0);
  // }
  // return can_launch;
  return m_m2ndps[buffer_id]->can_register_ndp_kernel(0);
}

void ScalabilityRunner::launch_ndp_kernel(NdpCommand command, int buffer_id) {
  spdlog::info("Launching NDP kernel: {} at cycle {}", command.ndp_kernel_path,
               m_m2ndp_config->get_sim_cycle());
  // for (auto m2ndp : m_m2ndps) {
  m_m2ndps[buffer_id]->register_ndp_kernel(0, command.ndp_kernel_path);
  // }
  // m_memory_reqs = command.memory_reqs;
  while (!command.memory_reqs[0].empty()) { // 0 is host id
    m_memory_reqs[0].push(command.memory_reqs[0].front());
    command.memory_reqs[0].pop();
  }
  assert(m_memory_reqs[0].size() > 0);
}

bool ScalabilityRunner::check_ndp_kenrel_active() {
  bool active = false;
  for (auto m2ndp : m_m2ndps) {
    active = active || m2ndp->is_launch_active(0);
  }
  return active;
}

bool ScalabilityRunner::check_all_memory_reqs_empty() {
  bool empty = true;
  for (auto req : m_memory_reqs) {
    empty = empty && req.empty();
  }
  return empty;
}

void ScalabilityRunner::process_memory_access() {
  assert(m_m2ndp_config->get_links_per_host() > 0);
  for (int host = 0; host < m_num_hosts; host++) {
    for (int node_id = 0; node_id < ceil(m_num_m2ndps/m_num_hosts); node_id++) {
    // for (int node_id = 0; node_id < m_m2ndp_config->get_links_per_host(); //TODO: other topology
    //      node_id++) {
      if (!m_memory_reqs[host].empty()) {
        mem_fetch* mf = m_memory_reqs[host].front();
        if (m_cxl_link->has_buffer_from_host(host, node_id, mf)) {
          m_cxl_link->push_from_host(host, node_id, mf);
          m_memory_reqs[host].pop();
          remaing_memory_reqs++;
        }
      }
      if (m_cxl_link->top_from_host(host, node_id) != NULL) {
        mem_fetch* mf = m_cxl_link->top_from_host(host, node_id);
        assert(!mf->is_request());
        assert(!mf->get_from_ndp() && !mf->get_from_dma());
        m_cxl_link->pop_from_host(host, node_id);
        remaing_memory_reqs--;
        if (mf->is_write()) {
          KernelLaunchInfo* data = (KernelLaunchInfo*)mf->get_data();
          delete data;
        }
        delete mf;
      }
    }
  }
}

void ScalabilityRunner::parse_ndp_trace(int buffer_id) {
  std::string trace_dir_path = m_trace_dir_path + "/" + to_string(buffer_id);
  std::ifstream ifs(trace_dir_path + "/kernelslist.g");
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
          trace_dir_path + "/" + file_name + "_input.data";
      m_memory_map[buffer_id] = new HashMemoryMap(input_memory_path);
      if (m_m2ndp_config->get_use_synthetic_memory())
        m_memory_map[buffer_id]->set_synthetic_memory(
            m_m2ndp_config->get_synthetic_base_address(),
            m_m2ndp_config->get_synthetic_memory_size());
    }
    if (m_m2ndp_config->is_functional_sim() && cnt == num_kernels) {
      //make target memory map at last kernel
      std::string target_memory_path =
          trace_dir_path + "/" + file_name + "_output.data";
      m_target_map[buffer_id] = new HashMemoryMap(target_memory_path);
    }
    NdpCommand command = NdpCommand{
        .ndp_kernel_path = trace_dir_path + "/" + kernel_name + ".traceg",
        .is_barrier = false};
    std::ifstream ifs_data(trace_dir_path + "/" + kernel_name +
                           "_launch.txt");
    assert(ifs_data.is_open());
    std::string line;
    spdlog::info("Parsing NDP kernel: {}", kernel_name);

    while (std::getline(ifs_data, line)) {
      if (line.find("#") != std::string::npos) continue;
      fill_memory_access(command, line);
      spdlog::debug("Memory access filled");
    }
    m_ndp_commands[buffer_id].push(command);
  }
}

void ScalabilityRunner::fill_memory_access(NdpCommand& command,
                                          std::string line) {
  command.memory_reqs.resize(m_num_hosts);
  int host_id = 0;
  KernelLaunchInfo* info = M2NDPParser::parse_kernel_launch(line, host_id);
  // for (int i = 0; i < m_m2ndp_config->get_num_m2ndps(); i++) {
  uint64_t addr = KERNEL_LAUNCH_ADDR;
  mem_fetch* mf = new mem_fetch(addr, DMA_ALLOC_W, WRITE_REQUEST, PACKET_SIZE,
                                CXL_OVERHEAD, 0);                     
  mf->set_filtering();
  mf->set_from_ndp(false);
  mf->set_data(info);
  mf->set_channel(m_m2ndp_config->get_channel_index(addr));
  command.memory_reqs[host_id].push(mf);
  // }
}

void ScalabilityRunner::match_memorymap() {
  for (int i = 0; i < m_num_m2ndps; i++) {
    if (m_memory_map[i]->Match(*m_target_map[i])) {
      m_memory_map[i]->DumpMemory();
      spdlog::info("MEMORY MATCH SUCCESS\n");
    } else {
      spdlog::info("MEMORY MATCH FAIL\n");
    }
  }
  // if (m_target_map->Match(*m_memory_map)) {
  //   spdlog::info("MEMROY MATCH SUCCESS\n");
  // } else {
  //   spdlog::info("MEMROY MATCH FAIL\n");
  // }
}
}  // namespace NDPSim