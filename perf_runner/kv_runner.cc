#include <fstream>
#include <iostream>
#include <string>

#include "command_line_parser.h"
#include "kv_runner.h"

namespace po = boost::program_options;
namespace NDPSim {
KVRunner::KVRunner(int argc, char* argv[])
    : remaing_memory_reqs(0), m_memory_map(NULL), m_injection_queue(0, 0, NULL, false) {
  CommandLineParser cmd_parser = CommandLineParser();
  cmd_parser.add_command_line_option<std::string>("config",
                                                  "path for m2ndp config file");
  cmd_parser.add_command_line_option<std::string>(
      "trace", "path for ndp trace file directory");
  cmd_parser.add_command_line_option<int>(
      "num_m2ndps", "number of ndp switches in the system");
  cmd_parser.add_command_line_option<std::string>("output", "outputv filename");
  cmd_parser.add_command_line_option<double>("injection_rate",
                                             "set_injection_rate");
  cmd_parser.add_command_line_option<int>("max_mps", "max_mps");
  cmd_parser.add_command_line_option<bool>("serial", "serial_mode");
  try {
    cmd_parser.parse(argc, argv);
  } catch (const CommandLineParser::ParsingError& e) {
    spdlog::error(
        "Command line argument parrsing error captured. Error message: {}",
        e.what());
    throw(e);
  }
  bool serial = false;
  m_output_filename = "";
  cmd_parser.set_if_defined("config", &m_config_file_path);
  cmd_parser.set_if_defined("trace", &m_trace_dir_path);
  cmd_parser.set_if_defined("num_m2ndps", &m_num_m2ndps);
  cmd_parser.set_if_defined("output", &m_output_filename);
  cmd_parser.set_if_defined("injection_rate", &m_injection_rate);
  cmd_parser.set_if_defined("max_mps", &m_max_mps);
  cmd_parser.set_if_defined("serial", &serial);
  printf("secrial mode %d\n", m_max_mps);
  m_m2ndp_config = new M2NDPConfig(m_config_file_path, 1);
  m_m2ndps.resize(m_num_m2ndps);
  m_cxl_link = new CxlLink(m_m2ndp_config);
  m_output_file = fopen(
      m_output_filename.empty() ? "output.out" : m_output_filename.c_str(),
      "w");
  m_energy_file =
      fopen(m_output_filename.empty() ? "energy.out"
                                      : ("energy_" + m_output_filename).c_str(),
            "w");
  m_m2ndp_config->set_output_file(m_output_file);
  m_m2ndp_config->print_config(m_output_file);
  m_injection_queue = InjectionQueue(m_injection_rate, m_max_mps, m_m2ndp_config, serial);
  m_injection_queue.initialize_commands(&m_ndp_commands);
  parse_ndp_trace();
  for (int i = 0; i < m_num_m2ndps; i++) {
    m_m2ndps[i] = new M2NDP(m_m2ndp_config, m_memory_map, i);
    m_m2ndps[i]->set_cxl_link(m_cxl_link);
  }
  register_kernelslist();
}

void KVRunner::run() {
  extern Stats_NDPSim::StatList statlist;

  NdpCommand running_command;
  remaing_memory_reqs = 0;
  while (!check_all_simulaiton_finished()) {
    m_m2ndp_config->cycle();
    m_cxl_link->cycle();
    m_injection_queue.cycle();
    for (int i = 0; i < m_num_m2ndps; i++) {
      m_m2ndps[i]->cycle();
    }
    if(!m_bi_reqs.empty() && m_bi_reqs.front().second >= m_m2ndp_config->get_ndp_cycle()) {
      mem_fetch* mf = m_bi_reqs.front().first;
      m_bi_reqs.pop();
      m_memory_reqs.push(mf);
    }
    if (m_injection_queue.top()!= NULL) {
      mem_fetch* mf = m_injection_queue.top();
      m_memory_reqs.push(mf);
      m_injection_queue.pop();
    }
    process_memory_access();
  }
  m_injection_queue.print_avg_latency();
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

  fclose(m_output_file);
}

bool KVRunner::check_all_simulaiton_finished() {
  bool running = remaing_memory_reqs > 0 || !check_all_memory_reqs_empty() ||
                 !m_ndp_commands.empty() || !m_injection_queue.finished();

  running = running || m_cxl_link->is_active();
  for (int i = 0; i < m_num_m2ndps; i++) {
    running = running || m_m2ndps[i]->is_active();
  }
  return !running;
}

bool KVRunner::check_single_simulation_finished() {
  bool running = !m_injection_queue.finished();
  running = running || m_cxl_link->is_active();
  for (int i = 0; i < m_num_m2ndps; i++) {
    running = running || m_m2ndps[i]->is_active();
  }
  return !running;
}

bool KVRunner::check_can_launch_ndp_kernel() {
  bool can_launch = true;
  for (auto m2ndp : m_m2ndps) {
    can_launch = can_launch && m2ndp->can_register_ndp_kernel(0);
  }
  return can_launch;
}


bool KVRunner::check_ndp_kenrel_active() {
  bool active = false;
  for (auto m2ndp : m_m2ndps) {
    active = active || m2ndp->is_launch_active(0);
  }
  return active;
}

bool KVRunner::check_all_memory_reqs_empty() {
  bool empty = true;
  return m_memory_reqs.empty();
}

void KVRunner::process_memory_access() {
  // if(!m_m2ndp_config->is_link_cycle()) return;
  assert(m_m2ndp_config->get_links_per_host() > 0);
  for (int node_id = 0; node_id < m_m2ndp_config->get_links_per_host();
       node_id++) {
    if (!m_memory_reqs.empty()) {
      mem_fetch* mf = m_memory_reqs.front();
      if (m_cxl_link->has_buffer_from_host(0, node_id, mf)) {
        if(mf->is_write()) 
          printf("Kernel launch request sent at %lu\n", m_m2ndp_config->get_ndp_cycle());
        m_cxl_link->push_from_host(0, node_id, mf);
        m_memory_reqs.pop();
        if(!mf->is_bi() && !mf->is_bi_writeback())
          remaing_memory_reqs++;
      }
    }
    if (m_cxl_link->top_from_host(0, node_id) != NULL) {
      mem_fetch* mf = m_cxl_link->top_from_host(0, node_id);
      m_cxl_link->pop_from_host(0, node_id);
      if(!mf->is_request()) {
        remaing_memory_reqs--;
        if (mf->is_write() && mf->get_addr() == KERNEL_LAUNCH_ADDR) {
          KernelLaunchInfo* data = (KernelLaunchInfo*)mf->get_data();
          if (data->kernel_id == 1) {  // if kernel is get
            uint64_t addr = data->base_addr;
            mem_fetch* get_mf = new mem_fetch(addr, GLOBAL_ACC_R, READ_REQUEST,
                                              PACKET_SIZE, CXL_OVERHEAD, 0);
            get_mf->set_channel(m_m2ndp_config->get_channel_index(addr));
            get_mf->set_data(mf);
            m_memory_reqs.push(get_mf);
          } else {
            m_injection_queue.finish(mf);
            delete data;
            delete mf;
          }
        } else {
          mem_fetch* origin_mf = (mem_fetch*)mf->get_data();
          KernelLaunchInfo* data = (KernelLaunchInfo*)origin_mf->get_data();
          assert(data->kernel_id == 1);
          m_injection_queue.finish(origin_mf);
          delete data;
          delete origin_mf;
          delete mf;
        }
      }
      else {
        mf->set_reply();
        mf->set_data((void*) 0x77);
        if(!mf->is_write()) {
          mf->convert_to_write_request(mf->get_addr());
          mf->unset_bi();
          mf->set_bi_writeack();
        }
        m_bi_reqs.push(std::make_pair(mf, m_m2ndp_config->get_ndp_cycle() + m_m2ndp_config->get_bi_host_latency()));
      }
    }
  }
}

void KVRunner::register_kernelslist() {
  std::ifstream ifs(m_trace_dir_path + "/kernelslist.g");
  std::string kernel_name;
  std::string ndp_kernel_path;
  while (ifs >> kernel_name) {
    ndp_kernel_path = m_trace_dir_path + "/" + kernel_name + ".traceg";
    for (auto m2ndp : m_m2ndps) {
      m2ndp->register_ndp_kernel(0, ndp_kernel_path);
    }
  }
  ifs.close();
}

void KVRunner::parse_ndp_trace() {
  spdlog::info("Launching NDP kernels");
  spdlog::debug("Parsing trace...");
  // make input memory map at first kernel
  std::string input_memory_path = m_trace_dir_path + "/input.data";
  m_memory_map = new HashMemoryMap(input_memory_path);
  std::ifstream ifs_data(m_trace_dir_path + "/launch.txt");
  assert(ifs_data.is_open());
  std::string line;
  spdlog::info("Parsing NDP kernel: {}", "memcached");
  while (std::getline(
      ifs_data, line)) {  // FIXME: How can synchronize these memory requests?
    NdpCommand command = NdpCommand{.is_barrier = false};
    if (line.find("#") != std::string::npos) continue;
    fill_memory_access(command, line);
    spdlog::debug("Memory access filled");
    m_ndp_commands.push(command);
  }
}

void KVRunner::fill_memory_access(NdpCommand& command, std::string line) {
  int host_id = 0;
  KernelLaunchInfo* info = M2NDPParser::parse_kernel_launch(line, 0);
  command.ndp_kernel_path = info->kernel_name;
  for (int i = 0; i < m_m2ndp_config->get_num_m2ndps(); i++) {
    uint64_t addr = KERNEL_LAUNCH_ADDR;
    mem_fetch* mf = new mem_fetch(addr, DMA_ALLOC_W, WRITE_REQUEST, PACKET_SIZE,
                                  CXL_OVERHEAD, 0);
    mf->set_filtering();
    mf->set_from_ndp(false);
    mf->set_data(info);
    mf->set_channel(m_m2ndp_config->get_channel_index(addr));
    command.memory_reqs.push(mf);
  }
}

InjectionQueue::InjectionQueue(double injection_rate, int max_mps, M2NDPConfig* m2ndp_config, bool serial)
    : m_injection_rate(injection_rate), m_max_mps(max_mps),
    m_queue(), m_injection_times(), m_counter(0), m_m2ndp_config(m2ndp_config), serial(serial) {
  m_queue = std::queue<InjectionQueueEntry>();
  m_injection_times = std::unordered_map<mem_fetch*, uint64_t>();
}

void InjectionQueue::initialize_commands(std::queue<NdpCommand>* commands) {
  m_commands = commands;
}

void InjectionQueue::cycle() {
  if(!m_m2ndp_config->is_buffer_ndp_cycle()) return;
  m_counter += m_injection_rate;
  if(m_counter >= 1) {
    m_counter -= 1;
    if (serial && m_injection_times.size() > 1) return;
    if (!m_commands->empty()) {
      NdpCommand command = m_commands->front();
      m_commands->pop();
      assert(command.memory_reqs.size() == 1);
      mem_fetch* mf = command.memory_reqs.front();
      InjectionQueueEntry entry = {.mf = mf, .inject_time = m_m2ndp_config->get_ndp_cycle()};
      m_queue.push(entry);
      m_total_requests++;
    }
  }
}

mem_fetch* InjectionQueue::top() {
  if (m_queue.empty()) return NULL;
  if(m_max_mps  <= m_injection_times.size()) return NULL;
  return m_queue.front().mf;
}

void InjectionQueue::pop() {
  if (m_queue.empty()) return;
  if(m_max_mps  <= m_injection_times.size()) return;
  m_injection_times[m_queue.front().mf] = m_queue.front().inject_time;
  m_queue.pop();
}

bool InjectionQueue::empty() { return m_queue.empty(); }

void InjectionQueue::finish(mem_fetch* mf) {
  uint64_t latency = m_m2ndp_config->get_ndp_cycle() - m_injection_times[mf];
  m_total_latency += latency;
  m_latencies.push_back(latency);
  printf("Cycle %lu, KVLatency: %lu\n", m_m2ndp_config->get_ndp_cycle(), latency);
  m_injection_times.erase(mf);
}

void InjectionQueue::print_avg_latency() {
  printf("Average latency: %f\n", (float)m_total_latency / m_total_requests);
}

bool InjectionQueue::finished() {
  return m_commands->empty() && m_queue.empty() && m_injection_times.empty();
}


}  // namespace NDPSim