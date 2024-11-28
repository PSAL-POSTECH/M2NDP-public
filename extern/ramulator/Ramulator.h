#ifndef RAMULATOR_H_
#define RAMULATOR_H_

#include <deque>
#include <functional>
#include <map>
#include <string>

#include "MemoryFactory.h"
//#include "Memory.h"
#include "Config.h"
//#include "MemoryFactory.h"
//#include "Memory.h"
//#include "Request.h"

#include "delayqueue.h"
#include "mem_fetch.h"
//#include "../gpgpu-sim/l2cache.h"
// #include "../gpgpu-sim/gpu-sim.h"

// extern std::vector<std::pair<unsigned long, unsigned long>> malloc_list;

namespace NDPSim {
class Request;
class MemoryBase;


// using namespace NDPSim;

class Ramulator {
 public:
  // Todo: memory stats
  // Ramulator(unsigned memory_id, class memory_stats_t* stats,
  //           unsigned long long* cycles, unsigned num_cores,
  //           std::string ramulator_config);
  Ramulator(unsigned memory_id, unsigned num_cores,
            std::string ramulator_config, std::string out, int log_inverval);
  ~Ramulator();
  // check whether the read or write queue is available
  bool full(bool is_write) const;
  void cycle();

  void finish();
  void print(FILE *fp = NULL);

  // push mem_fetcth object into Ramulator wrapper
  void push(class mem_fetch* mf, int ch_num);

  mem_fetch* return_queue_top(int ch_num) const;
  mem_fetch* return_queue_pop(int ch_num) const;
  void return_queue_push_back(mem_fetch* mf, int ch_num);
  // void return_queue_push_back(mem_fetch* mf); //for cxl memory buffer functions?

  bool returnq_full(int ch_num) const;
  bool from_gpgpusim_full(int ch_num) {return from_gpgpusim[ch_num]->full();}
  bool send(NDPSim::Request req, int ch_num);

  virtual bool is_active();

  virtual void set_dram_power_stats(unsigned &cmd, unsigned &activity, unsigned &nop,
                            unsigned &act, unsigned &pre, unsigned &rd,
                            unsigned &wr, unsigned &req) const;

  int get_num_reads() const;
  int get_num_writes() const;
  int get_precharges() const;
  int get_refreshes() const;
  int get_activations() const;

  // double tCK;

  int channels;

  //moved from private to public
  // callback functions
  std::function<void(NDPSim::Request&)> read_cb_func;
  std::function<void(NDPSim::Request&)> write_cb_func;

  unsigned num_cores;
  unsigned m_id;

 private:
  // StatList statlist;
  bool is_gpu;
  std::string std_name;
  NDPSim::MemoryBase* memory;
  
  fifo_pipeline<mem_fetch>** rwq;
  fifo_pipeline<mem_fetch>** returnq;
  fifo_pipeline<mem_fetch>** from_gpgpusim;
  // std::deque<ramulator::Request*>* rwq;
  // std::deque<ramulator::Request*>* returnq;
  // std::deque<ramulator::Request*>* from_gpgpusim;

  // std::map<unsigned long long, std::deque<mem_fetch*>> reads;
  // std::map<unsigned long long, std::deque<mem_fetch*>> writes;

  unsigned long long* cycles;

  void readComplete(NDPSim::Request& req);
  void writeComplete(NDPSim::Request& req);

  std::map<mem_fetch*, int> running_mf_reqs;
  std::vector<std::queue<Request>> unaccepted;
  int m_channel_unit;
  int log_interval = 10000;
  std::vector<int> num_reqs;
  std::vector<int> num_reads;
  std::vector<int> num_writes;
  std::vector<int> tot_reqs;
  std::vector<int> tot_reads;
  std::vector<int> tot_writes;
  uint64_t clk;
 protected:
  // Config -
  // it parses options from ramulator_config file when it is constructed
  NDPSim::Config ramulator_configs;
  std::string ramulator_name;

};
}  // namespace NDPSim
#endif
