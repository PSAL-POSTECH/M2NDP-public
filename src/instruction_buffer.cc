#ifdef TIMING_SIMULATION
#include "instruction_buffer.h"

#include <set>

#include "instruction_queue.h"
#define BUFFER_ACCESS_DELAY 1
#define BUFFER_ACCESS_INTERVAL 1
namespace NDPSim {
typedef std::set<int> RegSet;
InstructionBuffer::InstructionBuffer(int max_kernel_register,
                                     int queue_size, int ndp_id)
    : m_max_kernel_register(max_kernel_register), m_ndp_id(ndp_id) {
  m_delay_queue = DelayQueue<RequestInfo*>("inst_buffer", false, 1);
}

bool InstructionBuffer::can_register() {
  return m_ndp_kernels.size() < m_max_kernel_register;
}

void InstructionBuffer::register_kernel(NdpKernel* ndp_kernel) {
  int kernel_id = ndp_kernel->kernel_id;
  if (m_ndp_kernels.find(kernel_id) != m_ndp_kernels.end()) {
    spdlog::info("Kernel {} already registered", kernel_id);
    return;
  }
  m_ndp_kernels[kernel_id] = ndp_kernel;
  m_ndp_kernels[kernel_id]->valid = true;
  count_required_regs(m_ndp_kernels[kernel_id]->initializer_insts,
                      m_ndp_kernels[kernel_id]->initializer_xregs,
                      m_ndp_kernels[kernel_id]->initializer_fregs,
                      m_ndp_kernels[kernel_id]->initializer_vregs);
  for(int i = 0; i < m_ndp_kernels[kernel_id]->num_kernel_bodies; i++) {
    count_required_regs(m_ndp_kernels[kernel_id]->kernel_body_insts[i],
                        m_ndp_kernels[kernel_id]->kernel_body_xregs[i],
                        m_ndp_kernels[kernel_id]->kernel_body_fregs[i],
                        m_ndp_kernels[kernel_id]->kernel_body_vregs[i]);
  }
  count_required_regs(m_ndp_kernels[kernel_id]->finalizer_insts,
                      m_ndp_kernels[kernel_id]->finalizer_xregs,
                      m_ndp_kernels[kernel_id]->finalizer_fregs,
                      m_ndp_kernels[kernel_id]->finalizer_vregs);
}

void InstructionBuffer::unregister_kernel(int kernel_id) {
  m_ndp_kernels.erase(kernel_id);
}

void InstructionBuffer::cycle() {
  m_delay_queue.cycle();
  if (m_delay_queue.empty()) return;
  InstColumn* col = new InstColumn();
  RequestInfo* req = m_delay_queue.top();
  if (req->type == INITIALIZER) {
    col->insts = m_ndp_kernels[req->kernel_id]->initializer_insts;
    col->type = INITIALIZER;
    col->xregs = m_ndp_kernels[req->kernel_id]->initializer_xregs;
    col->fregs = m_ndp_kernels[req->kernel_id]->initializer_fregs;
    col->vregs = m_ndp_kernels[req->kernel_id]->initializer_vregs;
    col->loop_map = m_ndp_kernels[req->kernel_id]->loop_map;
  } else if (req->type == FINALIZER) {
    col->insts = m_ndp_kernels[req->kernel_id]->finalizer_insts;
    col->type = FINALIZER;
    col->xregs = m_ndp_kernels[req->kernel_id]->finalizer_xregs;
    col->fregs = m_ndp_kernels[req->kernel_id]->finalizer_fregs;
    col->vregs = m_ndp_kernels[req->kernel_id]->finalizer_vregs;
    col->loop_map = m_ndp_kernels[req->kernel_id]->loop_map;
  } else {
    col->insts = m_ndp_kernels[req->kernel_id]->kernel_body_insts[req->kernel_body_id];
    col->type = KERNEL_BODY;
    col->xregs = m_ndp_kernels[req->kernel_id]->kernel_body_xregs[req->kernel_body_id];
    col->fregs = m_ndp_kernels[req->kernel_id]->kernel_body_fregs[req->kernel_body_id];
    col->vregs = m_ndp_kernels[req->kernel_id]->kernel_body_vregs[req->kernel_body_id];
    col->loop_map = m_ndp_kernels[req->kernel_id]->loop_map;
  }
  col->kernel_id = req->kernel_id;
  col->req = req;
  // Base address for instruction
  col->base_addr = req->kernel_id * NDP_FUNCTION_OFFSET;
  if (col->type == KERNEL_BODY) {
    col->base_addr +=
        m_ndp_kernels[req->kernel_id]->initializer_insts.size() * WORD_SIZE;
  } else if (col->type == FINALIZER) {
    col->base_addr +=
        m_ndp_kernels[req->kernel_id]->initializer_insts.size() * WORD_SIZE +
        m_ndp_kernels[req->kernel_id]->kernel_body_insts.size() * WORD_SIZE;
  }
  // CSR initialize
  col->csr.vtype_vlmul = 1;
  m_inst_columns.push_back(col);
  m_delay_queue.pop();
}

bool InstructionBuffer::can_push_request() { return !m_delay_queue.full(); }

void InstructionBuffer::push(RequestInfo* info) {
  m_delay_queue.push(info, BUFFER_ACCESS_DELAY, BUFFER_ACCESS_INTERVAL);
}

bool InstructionBuffer::can_pop() { return !m_inst_columns.empty(); }

InstColumn* InstructionBuffer::pop() {
  assert(can_pop());
  InstColumn* col = top();
  m_inst_columns.pop_front();
  return col;
}

InstColumn* InstructionBuffer::top() {
  assert(can_pop());
  return m_inst_columns.front();
}

void InstructionBuffer::count_required_regs(std::deque<NdpInstruction> insts,
                                            int& num_xreg, int& num_freg,
                                            int& num_vreg) {
  RegSet used_xregs;
  RegSet used_fregs;
  RegSet used_vregs;
  RegSet used_double_vreg;
  int vmul = 1;
  for (NdpInstruction inst : insts) {
    if (inst.CheckScalarOp()) {
      if (inst.CheckFloatOp()) {
        used_fregs.insert(inst.dest);
      } else if (inst.opcode != CSRWI && inst.opcode != CSRW) {
        used_xregs.insert(inst.dest);
      }
    } else {
      if (inst.opcode == VSETVLI) {
        if (inst.src[3] == IMM_M1) {
          vmul = 1;
        } else if (inst.src[3] == IMM_M2) {
          vmul = 2;
        }
      } else if (inst.operand_type == F_S) {
        used_fregs.insert(inst.dest);
      } else if (inst.opcode != VSE16 && inst.opcode != VSE32 &&
                 inst.opcode != VSUXEI32) {
        used_vregs.insert(inst.dest);
        for (int i = 1; i < vmul; i++) {
          used_double_vreg.insert(inst.dest);
        }
      }
    }
  }
  num_xreg = used_xregs.size();
  num_freg = used_fregs.size();
  num_vreg = used_vregs.size() + used_double_vreg.size();
}
}  // namespace NDPSim
#endif