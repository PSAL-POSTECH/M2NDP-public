#ifdef TIMING_SIMULATION
#ifndef INSTRUCTION_BUFFER
#define INSTRUCTION_BUFFER
#include <deque>
#include <utility>
#include <vector>

#include "cache.h"
#include "common.h"
#include "delay_queue.h"
#include "ndp_instruction.h"
namespace NDPSim {

class InstructionBuffer {
 public:
  InstructionBuffer(int max_kernel_register, int queue_size, int ndp_id);
  void register_kernel(NdpKernel* ndp_function);
  void unregister_kernel(int kernel_id);
  bool can_register();
  void cycle();
  bool can_push_request();
  void push(RequestInfo* req);
  bool can_pop();
  InstColumn* pop();
  InstColumn* top();

 private:
  int m_ndp_id;
  unsigned m_max_kernel_register;
  DelayQueue<RequestInfo*> m_delay_queue;
  std::deque<int> m_free_ndp_kernel_id;
  std::map<int, NdpKernel*> m_ndp_kernels;
  std::deque<InstColumn*> m_inst_columns;
  void count_required_regs(std::deque<NdpInstruction> insts, int& num_xreg,
                           int& num_freg, int& num_vreg);
};
}  // namespace NDPSim
#endif
#endif