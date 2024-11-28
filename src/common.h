#ifndef COMMON_H
#define COMMON_H
#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <deque>
#include <map>
#include <vector>

#include "common_defs.h"
#include "ndp_instruction.h"
#include "vector_data.h"
namespace NDPSim {
class mem_fetch;
class MemoryMap;

struct NdpKernel {
  bool valid;
  bool active;
  int kernel_id;
  int kernel_info_idx;
  std::string kernel_name;
  int initializer_xregs;
  int initializer_fregs;
  int initializer_vregs;
  std::deque<int> kernel_body_xregs;
  std::deque<int> kernel_body_fregs;
  std::deque<int> kernel_body_vregs;
  int finalizer_xregs;
  int finalizer_fregs;
  int finalizer_vregs;
  int num_kernel_bodies;
  std::deque<NdpInstruction> initializer_insts;
  std::deque <std::deque<NdpInstruction>> kernel_body_insts;
  std::map<int, int> loop_map;
  std::deque<NdpInstruction> finalizer_insts;
};

struct KernelLaunchInfo {
  bool sync;
  int kernel_id;       // 4b
  uint64_t base_addr;  // 8b
  uint64_t size;       // 8b
  uint64_t smem_size;  // 8b
  uint64_t arg_size;   // 1b //24B
  uint64_t args[MAX_NR_INT_ARG]; //40B for 64B packet
  //values for simulation
  int num_float_args;
  float float_args[MAX_NR_FLOAT_ARG]; //12B
  int host_id;
  int launch_id;
  std::string kernel_name;
  mem_fetch* cxl_command;
  MemoryMap* scratchpad_map;
};

class VectorData;

enum RequestType { INITIALIZER, KERNEL_BODY, FINALIZER };
struct RequestInfo {
  RequestType type;
  uint64_t id;
  uint64_t launch_id;
  uint32_t kernel_id;
  uint32_t kernel_body_id;
  uint64_t addr;
  uint64_t offset;
  MemoryMap* scratchpad_map;
};

class MemoryMap;
class RegisterUnit;
struct CSR {
  int pc = 0;          // Program counter
  bool block = false;  // Block flag
  int vstart = 0;      // Vector start position
  int vxsat = 0;       // Fixed-Point Saturate Flag
  int vxrm = 0;        // Fixed-Point Rounding Mode
  int vcsr = 0;        // Vector control and status register
  int vl = 0;          // Vector length
  int vtype_vill = 0;
  int vtype_vma = 0;
  int vtype_vta = 0;
  int vtype_vsew = 32;    // element의 bit
  float vtype_vlmul = 1;  // vector reg 사용 개수
  int vlenb = 32;         // VLEN/8 (vector register length in bytes)
};

struct Context {
  int ndp_id;
  int sub_core_id;
  int inst_col_id;
  CSR *csr;
  bool *blocking;  // instruction blocking
  MemoryMap *memory_map;
  MemoryMap *scratchpad_map;
  RegisterUnit *register_map;
  std::map<int, int> *loop_map;
  RequestInfo *request_info;
  bool last_inst;
  int max_pc;
  bool exit;
};

struct InstColumn {
  bool valid;
  int id;
  int kernel_id;
  RequestType type;
  std::deque<NdpInstruction> insts;
  std::map<int, int> loop_map;
  RequestInfo *req;
  CSR csr;
  int xregs;
  int fregs;
  int vregs;
  uint64_t base_addr;
  NdpInstruction *current_inst = NULL;
  bool pending = false;
  uint32_t counter = 0;
  NdpInstruction *fetched_inst = NULL;
  bool feteched_last = false;
};

}  // namespace NDPSim
#endif  // FUNCSIM_INSTRUCTION_H_