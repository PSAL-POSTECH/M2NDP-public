#include "m2ndp_parser.h"
#include "common_defs.h"

#include <unistd.h>

#include <fstream>
#include <sstream>

#include "m2ndp_config.h"
#define MHz *1000000
#define SKIP_LABEL 100
// TODO: this file needs refactoring
namespace NDPSim {

NdpInstruction ParseNdpInstruction(std::string line);
Opcode StringToOpcode(std::string opcode);
bool CheckMemoryAccess(Opcode opcode);
std::vector<int> ParseShape(std::string shape_str);
std::vector<char> ParseString(std::string pred_val);
char *ParseVString(std::string pred_val);

void M2NDPParser::parse_ndp_kernel(int num_simd, std::string file_path,
                                  NdpKernel *ndp_kernel, M2NDPConfig *config) {
  std::string kernel_name;
  int kernel_id;

  std::ifstream ifs;
  ifs.open(file_path.c_str());

  std::string line;
  // Parse ndp kernel information
  for (int i = 0; i < 2; i++) {
    getline(ifs, line);
    if ("-kernel name" == line.substr(0, 12)) {
      const size_t equal_idx = line.find('=');
      kernel_name = line.substr(equal_idx + 1);
    } else if ("-kernel id" == line.substr(0, 10)) {
      sscanf(line.c_str(), "-kernel id = %d", &kernel_id);
    } else {
      assert(0 && "invalid ndp kernel trace format");
    }
  }

  getline(ifs, line);

  // SIMD instruction parse
  int step = 0;
  int inst_count = 0;
  int arr_idx = 0;
  std::map<int, int> loop_map;
  std::deque<NdpInstruction> initializer_insts;
  std::deque<std::deque<NdpInstruction>> kernel_body_insts;
  std::deque<NdpInstruction> finalizer_insts;
  int kernel_bodies = 0;
  while (!ifs.eof()) {
    getline(ifs, line);
    if (line.length() <= 1) {
      continue;
    }
    if (line.substr(0, 1) == "#") {
      continue;
    } else if (line == "INITIALIZER:") {
      step = 0;
      arr_idx = 0;
    } else if (line == "KERNELBODY:") {
      step = 1;
      arr_idx = 0;
      kernel_body_insts.push_back(std::deque<NdpInstruction>());
      kernel_bodies++;
    } else if (line == "FINALIZER:") {
      assert(step == 1);
      arr_idx = 0;
      step = 2;
    } else if (line.substr(0, 5) == ".LOOP") {
      int loop_num = std::stoi(line.substr(5));
      loop_map[loop_num] = arr_idx;
    } else if (line.substr(0, 5) == ".SKIP") {
      int loop_num = std::stoi(line.substr(5)) + SKIP_LABEL;
      loop_map[loop_num] = arr_idx;
    } else {
      if (step == 0) {
        initializer_insts.push_back(ParseNdpInstruction(line));
        arr_idx++;
      } else if (step == 1) {
        kernel_body_insts[kernel_bodies - 1].push_back(ParseNdpInstruction(line));
        arr_idx++;
      } else if (step == 2) {
        finalizer_insts.push_back(ParseNdpInstruction(line));
        arr_idx++;
      }
    }
  }
  ndp_kernel->num_kernel_bodies = kernel_bodies;
  ndp_kernel->kernel_id = kernel_id;
  ndp_kernel->kernel_name = kernel_name;
  ndp_kernel->initializer_insts = initializer_insts;
  ndp_kernel->kernel_body_insts = kernel_body_insts;
  ndp_kernel->finalizer_insts = finalizer_insts;
  ndp_kernel->loop_map = loop_map;
}

uint32_t GetRegValue(std::string src) {
  if (src == "DATA") {
    return REG_REQUEST_DATA;
  } else if (src == "OFFSET") {
    return REG_REQUEST_OFFSET;
  } else if (src == "ADDR") {
    return REG_ADDR;
  } else if (src == "NDPID") {
    return REG_NDP_ID;
  } else if (src == "e8") {
    return IMM_E8;
  } else if (src == "e16") {
    return IMM_E16;
  } else if (src == "e32") {
    return IMM_E32;
  } else if (src == "e64") {
    return IMM_E64;
  } else if (src == "mf8") {
    return IMM_MF8;
  } else if (src == "mf4") {
    return IMM_MF4;
  } else if (src == "mf2") {
    return IMM_MF2;
  } else if (src == "m1") {
    return IMM_M1;
  } else if (src == "m2") {
    return IMM_M2;
  } else if (src == "m4") {
    return IMM_M4;
  } else if (src == "m8") {
    return IMM_M8;
  } else if (src == "ma") {
    return IMM_MA;
  } else if (src == "mu") {
    return IMM_MU;
  } else if (src == "ta") {
    return IMM_TA;
  } else if (src == "tu") {
    return IMM_TU;
  } else if (src == "vstart") {
    return REG_VSTART;
  } else if (src == "x0") {
    return REG_X0;
  } else if (src.length() > 1 && src.substr(0, 1) == "x") {
    return REG_X_BASE + std::stoull(src.substr(1), NULL, 10);
  } else if (src.length() > 1 && src.substr(0, 1) == "f") {
    return REG_F_BASE + std::stoull(src.substr(1), NULL, 10);
  } else if (src.length() > 1 && src.substr(0, 1) == "v") {
    return REG_V_BASE + std::stoull(src.substr(1), NULL, 10);
  } else {
    spdlog::warn("Invalid source register: {}", src);
  }
  return 0;
}

NdpInstruction ParseNdpInstruction(std::string line) {
  NdpInstruction inst;
  std::string opcode;
  std::string src;
  std::string dest;
  int check;
  int value;
  int segCnt;

  inst.sInst = line + "\0";  // For debug

  for (int i = 0; i < 5; i++) inst.src[i] = -1;

  opcode = line.substr(0, line.find(' '));
  line = line.substr(line.find(' ') + 1);
  line.erase(std::remove(line.begin(), line.end(), ' '), line.end());
  dest = line.substr(0, line.find(','));
  src = line.substr(line.find(',') + 1);
  size_t point_index = opcode.find('.');
  inst.operand_type = OperandType::NONE;
  if (point_index != -1) {
    std::string operand_type = opcode.substr(point_index + 1);
    opcode = opcode.substr(0, point_index);
    if (opcode.rfind("vlsseg") == 0)
      inst.opcode = opcode_type_map["vlsseg"];
    else
      inst.opcode = opcode_type_map[opcode];
    inst.operand_type = operand_type_map[operand_type];
  } else {
    inst.opcode = opcode_type_map[opcode];
    inst.operand_type = NONE;
  }
  assert(inst.opcode != Opcode::INVALID_INST && "Invalid instruction");
  if (inst.opcode == Opcode::EXIT) return inst;
  inst.dest = GetRegValue(dest);
  if (inst.opcode == Opcode::LI || inst.opcode == Opcode::CSRWI) {
    inst.src[0] = std::stoll(src, NULL);
  } else if (inst.opcode == Opcode::CSRW) {
    inst.src[0] = GetRegValue(src);
  } else if (inst.opcode == Opcode::LBU) {
    int end = src.find(',');
    inst.src[0] = GetRegValue(src.substr(0, end));
    int offset = std::stoi(src.substr(src.find(',') + 1));
    inst.src[1] = offset;
  } else if (inst.opcode == Opcode::VAND) {
    if (inst.operand_type == OperandType::VV) {
      std::string vs1 = src.substr(0, src.find(','));
      std::string vs2 = src.substr(src.find(',') + 1);
      inst.src[0] = GetRegValue(vs1);
      inst.src[1] = GetRegValue(vs2);
    } else if (inst.operand_type == OperandType::VI) {
      std::string vs1 = src.substr(0, src.find(','));
      std::string imm = src.substr(src.find(',') + 1);
      inst.src[0] = GetRegValue(vs1);
      inst.src[1] = std::stoll(imm, NULL);
    }
  } else if (inst.opcode == Opcode::VOR) {
    if (inst.operand_type == OperandType::VV) {
      std::string vs1 = src.substr(0, src.find(','));
      src = src.substr(src.find(',') + 1);
      std::string vs2 = src.substr(0, src.find(','));
      std::string vm = src.substr(src.find(',') + 1);
      inst.src[0] = GetRegValue(vs1);
      inst.src[1] = GetRegValue(vs2);
      inst.src[2] = GetRegValue(vm);
    }
  } else if (inst.opcode == Opcode::VMAND || inst.opcode == Opcode::VMOR) {
    if (inst.operand_type == OperandType::MM) {
      std::string vs1 = src.substr(0, src.find(','));
      std::string vs2 = src.substr(src.find(',') + 1);
      inst.src[0] = GetRegValue(vs1);
      inst.src[1] = GetRegValue(vs2);
    }
  } else if (inst.opcode == Opcode::VLE8 || inst.opcode == Opcode::VSE8 ||
             inst.opcode == Opcode::VLE16 || inst.opcode == Opcode::VSE16 ||
             inst.opcode == Opcode::VLE32 || inst.opcode == Opcode::VSE32 ||
             inst.opcode == Opcode::VLE64 || inst.opcode == Opcode::VSE64 ||
             inst.opcode == Opcode::LW || inst.opcode == Opcode::SW ||
             inst.opcode == Opcode::FLW || inst.opcode == Opcode::FSW ||
             inst.opcode == Opcode::SB || inst.opcode == Opcode::SD ||
             inst.opcode == Opcode::LB ||
             inst.opcode == Opcode::LD) {  // please re-order later
    int begin = src.find('(');
    int end = src.find(')');
    std::string offset = src.substr(0, begin);
    std::string base = src.substr(begin + 1, end - begin - 1);
    inst.src[0] = GetRegValue(base);
    inst.src[1] = 0;
    if (offset.size() > 0 && begin != std::string::npos) {
      inst.src[1] = std::stoll(offset, NULL);
    }
  } else if (inst.opcode == Opcode::VLUXEI32 ||
             inst.opcode == Opcode::VLUXEI64) {
    int begin = src.find('(');
    int end = src.find(')');
    std::string offset = src.substr(0, begin);
    std::string base = src.substr(begin + 1, end - begin - 1);
    src = src.substr(src.find(',') + 1);
    std::string vs2 = src.substr(0, src.find(','));
    // src = src.substr(src.find(',') + 1);
    std::string vm = src.substr(src.find(',') + 1);
    inst.src[0] = GetRegValue(base);
    inst.src[1] = 0;
    if (offset.size() > 0) inst.src[1] = std::stoll(offset, NULL);
    inst.src[2] = GetRegValue(vs2);
    if (src.find(',') != std::string::npos) inst.src[3] = GetRegValue(vm);
    //   To do : implement vmask
  } else if (inst.opcode == Opcode::VSUXEI32 ||
             inst.opcode == Opcode::VSUXEI64) {
    int begin = src.find('(');
    int end = src.find(')');
    std::string offset = src.substr(0, begin);
    std::string base = src.substr(begin + 1, end - begin - 1);
    src = src.substr(src.find(','));
    std::string vs2 = src.substr(src.find(',') + 1, src.find(',') + 2);
    src = src.substr(src.find(',') + 1);
    std::string vs3 = src.substr(src.find(',') + 1);
    inst.src[0] = GetRegValue(base);
    inst.src[1] = 0;
    if (offset.size() > 0) {
      inst.src[1] = std::stoll(offset, NULL);
    }
    inst.src[2] = GetRegValue(vs2);
    inst.src[3] = GetRegValue(vs3);
    //  To do : implement vmask
  } else if (inst.opcode == Opcode::VLSSEG) {
    int begin = src.find('(');
    int end = src.find(')');
    std::string vs1 = src.substr(begin + 1, end - begin - 1);
    std::string vs2 = src.substr(src.find(',') + 1);
    inst.src[0] = GetRegValue(vs1);
    inst.src[1] = GetRegValue(vs2);

    if (opcode.rfind("vlsseg", 0) == 0) {
      segCnt = std::stoi(opcode.substr(6, 1));
      assert(segCnt <= 8 && "Maximum segment indexed load is 8.");
      std::string v0 = dest.substr(0, 1);
      int i0 = std::stoi(dest.substr(1));
      inst.segDest[0] = inst.dest;

      for (int i = 1; i < segCnt; i++) {
        i0 += 1;
        std::string dest0 = std::string(v0) + std::to_string(i0);
        inst.segDest[i] = GetRegValue(dest0);
      }
    }
  } else if (inst.opcode == Opcode::VID) {
    inst.src[0] = GetRegValue(src);
  } else if (inst.opcode == Opcode::VMSEQ || inst.opcode == Opcode::VMSNE ||
             inst.opcode == Opcode::VMSLT || inst.opcode == Opcode::VMSGT ||
             inst.opcode == Opcode::VMSLE || inst.opcode == Opcode::VMSGE) {
    if (inst.operand_type == OperandType::VI) {
      std::string vs1 = src.substr(0, src.find(','));
      std::string vs2 = src.substr(src.find(',') + 1);
      inst.src[0] = GetRegValue(vs1);
      inst.src[1] = stoll(vs2);
    } else if (inst.operand_type == OperandType::VX) {
      std::string vs1 = src.substr(0, src.find(','));
      std::string vs2 = src.substr(src.find(',') + 1);
      inst.src[0] = GetRegValue(vs1);
      inst.src[1] = GetRegValue(vs2);
    } else if (inst.operand_type == OperandType::VV) {
      std::string vs1 = src.substr(0, src.find(','));
      std::string vs2 = src.substr(src.find(',') + 1);
      inst.src[0] = GetRegValue(vs1);
      inst.src[1] = GetRegValue(vs2);
    } else if (inst.operand_type == OperandType::VF) {
      std::string vs1 = src.substr(0, src.find(','));
      std::string vs2 = src.substr(src.find(',') + 1);
      inst.src[0] = GetRegValue(vs1);
      inst.src[1] = GetRegValue(vs2);
    }
  } else if (inst.CheckAmoOp()) {
    if (inst.opcode == AMOADD || inst.opcode == FAMOADD || inst.opcode == FAMOADDH ||
        inst.opcode == AMOMAX || inst.opcode == FAMOMAX) {
      int begin = src.find('(');
      int end = src.find(')');
      std::string rd = dest;
      std::string rs2 = src.substr(0, src.find(','));
      std::string rs1 = src.substr(begin + 1, end - begin - 1);
      inst.src[0] = GetRegValue(dest);
      inst.src[1] = GetRegValue(rs2);
      inst.src[2] = GetRegValue(rs1);
    } else {
      int begin = src.find('(');
      int end = src.find(')');
      std::string base = src.substr(begin + 1, end - begin - 1);
      src = src.substr(src.find(',') + 1);
      std::string vs2 = src.substr(0, src.find(','));
      src = src.substr(src.find(',') + 1);
      std::string vs3 = src.substr(0, src.find(','));
      src = src.substr(src.find(',') + 1);
      std::string vs4 = src.substr(0, src.find(','));
      inst.src[0] = GetRegValue(base);
      inst.src[1] = GetRegValue(vs2);
      inst.src[2] = GetRegValue(vs3);
      inst.src[3] = -1;
      if (src.find(',') != std::string::npos) inst.src[3] = GetRegValue(vs4);
    }
  } else if (inst.opcode == Opcode::VSETVLI) {
    int operand_count = std::count(src.begin(), src.end(), ',') + 1;
    assert(operand_count >= 2);  // VSETVLI has at least 3 operands
    std::string vs1 = src.substr(0, src.find(','));
    src = src.substr(src.find(',') + 1);
    std::string vs2 = src.substr(0, src.find(','));
    src = src.substr(src.find(',') + 1);
    if (operand_count == 2) {
      inst.dest = GetRegValue(dest);
      inst.src[0] = GetRegValue(vs1);
      inst.src[1] = GetRegValue(vs2);
      inst.src[2] = IMM_M1;
      inst.src[3] = 0;
      inst.src[4] = 0;
    } else if (operand_count == 3) {
      std::string vs3 = src.substr(0, src.find(','));
      inst.dest = GetRegValue(dest);
      inst.src[0] = GetRegValue(vs1);
      inst.src[1] = GetRegValue(vs2);
      inst.src[2] = GetRegValue(vs3);
      inst.src[3] = 0;
      inst.src[4] = 0;
    } else if (operand_count == 4) {
      std::string vs3 = src.substr(0, src.find(','));
      std::string vs4 = src.substr(0, src.find(','));
      inst.dest = GetRegValue(dest);
      inst.src[0] = GetRegValue(vs1);
      inst.src[1] = GetRegValue(vs2);
      inst.src[2] = GetRegValue(vs3);
      inst.src[3] = GetRegValue(vs4);
      inst.src[4] = 0;
    } else if (operand_count == 5) {  // issue.
      std::string vs3 = src.substr(0, src.find(','));
      src = src.substr(src.find(',') + 1);
      std::string vs4 = src.substr(0, src.find(','));
      inst.dest = GetRegValue(dest);
      inst.src[0] = GetRegValue(vs1);
      inst.src[1] = GetRegValue(vs2);
      inst.src[2] = IMM_M1;  // default
      inst.src[3] = GetRegValue(vs3);
      inst.src[4] = GetRegValue(vs4);
    } else if (operand_count == 6) {
      std::string vs3 = src.substr(0, src.find(','));
      src = src.substr(src.find(',') + 1);
      std::string vs4 = src.substr(0, src.find(','));
      src = src.substr(src.find(',') + 1);
      std::string vs5 = src.substr(0, src.find(','));
      inst.dest = GetRegValue(dest);
      inst.src[0] = GetRegValue(vs1);
      inst.src[1] = GetRegValue(vs2);
      inst.src[2] = GetRegValue(vs3);
      inst.src[3] = GetRegValue(vs4);
      inst.src[4] = GetRegValue(vs5);
    }
  } else if (inst.opcode == Opcode::ADDI || inst.opcode == Opcode::MULI ||
             inst.opcode == Opcode::SLLI || inst.opcode == Opcode::SRLI ||
             inst.opcode == Opcode::ANDI || inst.opcode == Opcode::ORI ||
             inst.operand_type == OperandType::VI) {
    std::string vs1 = src.substr(0, src.find(','));
    std::string imm = src.substr(src.find(',') + 1);
    inst.src[0] = GetRegValue(vs1);
    inst.src[1] = std::stoll(imm, NULL);
  } else if (inst.opcode == Opcode::VSLIDE1DOWN ||
             inst.opcode == Opcode::VSLIDE1UP) {
    if (inst.operand_type == OperandType::VX) {
      std::string vs1 = src.substr(0, src.find(','));
      std::string vs2 = src.substr(src.find(',') + 1);
      inst.src[0] = GetRegValue(vs1);
      inst.src[1] = GetRegValue(vs2);
    } else
      throw std::runtime_error("Unsupported Operand Type for VSLIDE");
  } else if (inst.opcode == Opcode::VSLIDEDOWN ||
             inst.opcode == Opcode::VSLIDEUP) {
    if (inst.operand_type == OperandType::VX) {
      std::string vs1 = src.substr(0, src.find(','));
      std::string vs2 = src.substr(src.find(',') + 1);
      inst.src[0] = GetRegValue(vs1);
      inst.src[1] = GetRegValue(vs2);
    } else if (inst.operand_type == OperandType::VI) {
      std::string vs1 = src.substr(0, src.find(','));
      std::string imm = src.substr(src.find(',') + 1);
      inst.src[0] = GetRegValue(vs1);
      inst.src[1] = std::stoi(imm);
    } else
      throw std::runtime_error("Unsupported Operand Type for VSLIDE");
  } else if (inst.opcode == Opcode::VREDSUM || inst.opcode == Opcode::VREDMAX ||
             inst.opcode == Opcode::VREDMIN || inst.opcode == Opcode::VREDAND ||
             inst.opcode == Opcode::VREDOR) {
    if (inst.operand_type != OperandType::VS)
      throw std::runtime_error("Unsupported Operand Type for VRED");
    std::string vs1 = src.substr(0, src.find(','));
    src = src.substr(src.find(',') + 1);
    std::string vs2 = src.substr(0, src.find(','));

    inst.src[0] = GetRegValue(vs1);
    inst.src[1] = GetRegValue(vs2);
    if (src.find(',') != std::string::npos) {
      src = src.substr(src.find(',') + 1);
      std::string vs3 = src.substr(0, src.find(','));
      inst.src[2] = GetRegValue(vs3);
    }
  } else if (inst.operand_type == OperandType::V) {
    std::string vs2 = src.substr(0, src.find(','));
    src = src.substr(src.find(',') + 1);
    std::string vm = src.substr(0, src.find(','));
    inst.src[0] = GetRegValue(vs2);
    inst.src[1] = GetRegValue(vm);
  } else if (inst.operand_type == OperandType::M) {
    inst.src[0] = GetRegValue(src);
  } else if (inst.operand_type == OperandType::X_S ||
             inst.operand_type == OperandType::S_X) {
    std::string vs1 = src.substr(0, src.find(','));
    inst.src[0] = GetRegValue(vs1);
  } else if (inst.operand_type == OperandType::V_I) {
    inst.src[0] = std::stoi(src);
    if (src.find(',') != -1) {
      std::string vs2 = src.substr(src.find(',') + 1);
      inst.src[1] = GetRegValue(vs2);
    }
  } else if (inst.operand_type == OperandType::VF) {
      std::string vs1 = src.substr(0, src.find(','));
      std::string vs2 = src.substr(src.find(',') + 1);
      inst.src[0] = GetRegValue(vs1);
      inst.src[1] = GetRegValue(vs2);
  } else if (inst.opcode == Opcode::TEST) {
    std::string s1 = src.substr(0, src.find(','));
    inst.src[0] = GetRegValue(s1);
    if (inst.operand_type == OperandType::VV) {
      inst.src[1] = GetRegValue(s1);
    }
  } else if (inst.opcode == Opcode::J) {
    if (dest.substr(0, 5) == ".LOOP")
      inst.src[2] = std::stoi(dest.substr(5));
    else if (dest.substr(0, 5) == ".SKIP")
      inst.src[2] = std::stoi(dest.substr(5)) + 100;
    else
      throw std::runtime_error("Wrong Loop Name.");
  } else if (inst.opcode == Opcode::BEQ || inst.opcode == Opcode::BGT ||
             inst.opcode == Opcode::BLT || inst.opcode == Opcode::BNE ||
             inst.opcode == Opcode::BLE || inst.opcode == Opcode::BGE) {
    std::string vs1 = src.substr(0, src.find(','));
    std::string loop = src.substr(src.find(',') + 1);
    if (loop.substr(0, 5) == ".LOOP") {
      inst.src[0] = GetRegValue(dest);
      inst.src[1] = GetRegValue(vs1);
      inst.src[2] = std::stoi(loop.substr(5));
    } else if (loop.substr(0, 5) == ".SKIP") {
      inst.src[0] = GetRegValue(dest);
      inst.src[1] = GetRegValue(vs1);
      inst.src[2] = std::stoi(loop.substr(5)) + 100;
    } else
      throw std::runtime_error("Wrong Loop Name.");
  } else if (inst.opcode == Opcode::BEQZ || inst.opcode == Opcode::BGTZ ||
             inst.opcode == Opcode::BLTZ || inst.opcode == Opcode::BNEZ ||
             inst.opcode == Opcode::BLEZ || inst.opcode == Opcode::BGEZ) {
    std::string loop = src.substr(0, src.find(','));
    if (loop.substr(0, 5) == ".LOOP") {
      inst.src[0] = GetRegValue(dest);
      inst.src[2] = std::stoi(loop.substr(5));
    } else if (loop.substr(0, 5) == ".SKIP") {
      inst.src[0] = GetRegValue(dest);
      inst.src[2] = std::stoi(loop.substr(5)) + 100;
    } else
      throw std::runtime_error("Wrong Loop Name.");
  } else if (inst.opcode == Opcode::FEXP || inst.opcode == Opcode::VFEXP) {
    std::string vs1 = src.substr(0, src.find(','));
    inst.src[0] = GetRegValue(vs1);
  } else {  // default : binary register operands
    std::string vs1 = src.substr(0, src.find(','));
    inst.src[0] = GetRegValue(vs1);
    if (src.find(',') != -1) {
      std::string vs2 = src.substr(src.find(',') + 1);
      inst.src[1] = GetRegValue(vs2);
    }    
  }
  inst.segCnt = segCnt;
  return inst;
}

std::vector<char> ParseString(std::string pred_val) {
  std::vector<char> output;
  for (char c : pred_val) {
    output.push_back(c);
  }
  return output;
}

char *ParseVString(std::string pred_val) {
  char *output;
  output = const_cast<char *>(pred_val.c_str());
  return output;
}

void M2NDPParser::parse_config(std::string config_path,
                              M2NDPConfig *m2ndp_config) {
  std::string config_dir;
  std::ifstream config_file(config_path);
  const size_t last_slash_idx = config_path.rfind('/');
  std::string directory;
  if (std::string::npos != last_slash_idx) {
    directory = config_path.substr(0, last_slash_idx);
  }
  if (std::string::npos != last_slash_idx) {
    directory = config_path.substr(0, last_slash_idx);
  }
  if (config_path.substr(0, 1) == ".")
    config_dir = std::string(get_current_dir_name()) + "/" + directory;
  else
    config_dir = directory;
  assert(config_file.good() && "Bad config file");
  std::string line;
  while (getline(config_file, line)) {
    char delim[] = " \t=";
    std::vector<std::string> tokens;

    while (true) {
      size_t start = line.find_first_not_of(delim);

      if (start == std::string::npos) break;

      size_t end = line.find_first_of(delim, start);

      if (end == std::string::npos) {
        tokens.push_back(line.substr(start));
        break;
      }
      tokens.push_back(line.substr(start, end - start));
      line = line.substr(end);
    }

    // empty line
    if (!tokens.size()) continue;

    // comment line
    if (tokens[0][0] == '#') continue;

    // parameter line
    assert(tokens.size() == 2 && "Only allow two tokens in one line");
    parse_to_const(m2ndp_config, config_dir, tokens[0], tokens[1]);
  }
#ifdef TIMING_SIMULATION
  spdlog::info(
      "CORE period: {:.20} DRAM period: {:.20} NDP buffer period: {:.20}",
      m2ndp_config->m_core_period, m2ndp_config->m_dram_period,
      m2ndp_config->m_buffer_ndp_period);
  spdlog::info(
      "NDP link period:{:.20} NDP cache period: {:.20} DRAM buffer period: "
      "{:.20}",
      m2ndp_config->m_link_period, m2ndp_config->m_cache_period,
      m2ndp_config->m_buffer_dram_period);
#endif
}

KernelLaunchInfo* M2NDPParser::parse_kernel_launch(std::string line, int host_id) {
  KernelLaunchInfo *info = new KernelLaunchInfo();
  info->host_id = host_id;
  std::stringstream ss(line);
  ss >> std::hex >> info->sync;
  ss >> std::hex >> info->kernel_id;
  ss >> std::hex >> info->base_addr;
  ss >> std::hex >> info->size;
  ss >> std::hex >> info->smem_size;
  ss >> std::hex >> info->arg_size;
  int num_int_args = info->arg_size / DOUBLE_SIZE;
  int num_fp32_args = 0;
  for (int i = 0; i < num_int_args; i++) {
    std::string token;
    ss >> token;
    if (token == "FP32") {
      num_fp32_args = (info->arg_size / DOUBLE_SIZE) - i;
      break;
    } else {
      //store at info->args
      uint64_t arg;
      std::stringstream ss_arg(token);
      ss_arg >> std::hex >> arg;
      info->args[i] = arg;
    }
  }
  for (int i = 0; i < num_fp32_args; i++) {
    float float_arg;
    ss >> float_arg;
    info->float_args[i] = float_arg;
  }
  info->num_float_args = num_fp32_args;

  /* Sanity check for number of kernel argument */
  assert(num_int_args <= MAX_NR_INT_ARG);
  assert(num_fp32_args <= MAX_NR_FLOAT_ARG);

  spdlog::info("Kernel launch info: sync: 0x{:x} kernel_id: 0x{:x} base_addr: 0x{:x} size: 0x{:x} arg_size: 0x{:x}",
               info->sync, info->kernel_id, info->base_addr, info->size,
               info->arg_size);
  for (int i = 0; i < (info->arg_size / DOUBLE_SIZE) - info->num_float_args ; i++) {
    spdlog::info("args[{}]: 0x{:x}", i, info->args[i]);
  }
  for (int i = 0; i < info->num_float_args; i++) {
    spdlog::info("float_args[{}]: {}", i, info->float_args[i]);
  }
  
  return info;
}

void M2NDPParser::parse_to_const(M2NDPConfig *config, std::string config_dir,
                                std::string name, std::string value) {
  if (name == "num_ndp_units")
    config->m_num_ndp_units = atoi(value.c_str());
  else if (name == "num_x_registers")
    config->m_num_x_registers = atoi(value.c_str());
  else if (name == "num_f_registers")
    config->m_num_f_registers = atoi(value.c_str());
  else if (name == "num_v_registers")
    config->m_num_v_registers = atoi(value.c_str());
  else if (name == "spad_size")
    config->m_spad_size = atoi(value.c_str());
  else if (name == "use_synthetic_memory")
    config->m_use_synthetic_memory = atoi(value.c_str());
  else if (name == "synthetic_base_address") {
    std::stringstream ss(value);
    ss >> std::hex >> config->m_synthetic_base_address;
  } else if (name == "synthetic_memory_size") {
    std::stringstream ss(value);
    ss >> std::hex >> config->m_synthetic_memory_size;
  } else if (name == "memory_mapping")
    decode_memory_mapping(config, value);
  else if (name == "channel_indexing_policy")
    config->m_channel_indexing_policy = atoi(value.c_str());
  else if (name == "enable_sub_core")
    config->m_enable_sub_core = atoi(value.c_str());
  else if (name == "num_sub_core")
    config->m_num_sub_core = atoi(value.c_str());
#ifdef TIMING_SIMULATION
  else if (name == "num_memories")
    config->m_num_memories = atoi(value.c_str());
  else if (name == "links_per_host")
    config->m_links_per_host = atoi(value.c_str());
  else if (name == "links_per_memory_buffer")
    config->m_links_per_memory_buffer = atoi(value.c_str());
  else if (name == "ramulator_config")
    config->m_ramulator_config_path = config_dir + "/" + value;
  else if (name == "cxl_link_config")
    config->m_cxl_link_config_path = config_dir + "/" + value;
  else if (name == "local_cross_bar_config")
    config->m_local_crossbar_config_path = config_dir + "/" + value;
  else if (name == "functional_sim")
    config->m_functional_sim = atoi(value.c_str());
  else if (name == "always_crossbar")
    config->m_always_crossbar = atoi(value.c_str());
  else if (name == "packet_size")
    config->m_packet_size = atoi(value.c_str());
  else if (name == "cxl_link_buffer_size")
    config->m_cxl_link_buffer_size = atoi(value.c_str());
  else if (name == "m2ndp_interleave_size")
    config->m_m2ndp_interleave_size = atoi(value.c_str());
  else if (name == "channel_interleave_size")
    config->m_channel_interleave_size = atoi(value.c_str());
  else if (name == "simd_width")
    config->m_simd_width = atoi(value.c_str());
  else if (name == "num_memory_buffers")
    config->m_num_memory_buffers = atoi(value.c_str());
  else if (name == "ndp_units_per_buffer")
    config->m_ndp_units_per_buffer = atoi(value.c_str());
  else if (name == "request_queue_size")
    config->m_request_queue_size = atoi(value.c_str());
  else if (name == "spad_latency")
    config->m_spad_latency = atoi(value.c_str());
  else if (name == "page_size")
    config->m_tlb_page_size = atoi(value.c_str());
  else if (name == "tlb_entry_size")
    config->m_tlb_entry_size = atoi(value.c_str());
  else if (name == "ideal_tlb")
    config->m_ideal_tlb = atoi(value.c_str());
  else if (name == "ideal_icache")
    config->m_ideal_icache = atoi(value.c_str());
  else if (name == "use_dram_tlb")
    config->m_use_dram_tlb = atoi(value.c_str());
  else if (name == "skip_l1d") 
    config->m_skip_l1d = atoi(value.c_str());
  else if (name == "l1d_config")
    config->m_l1d_config_str = value;
  else if (name == "l1d_num_banks")
    config->m_l1d_num_banks = atoi(value.c_str());
  else if (name == "l1d_hit_latency")
    config->m_l1d_hit_latency = atoi(value.c_str());
  else if (name == "l2d_config")
    config->m_l2d_config_str = value;
  else if (name == "l2d_num_banks")
    config->m_l2d_num_banks = atoi(value.c_str());
  else if (name == "l2d_hit_latency")
    config->m_l2d_hit_latency = atoi(value.c_str());
  else if (name == "l0icache_config")
    config->m_l0icache_config_str = value;
  else if (name == "l1icache_config")
    config->m_l1icache_config_str = value;
  else if (name == "l0icache_hit_latency")
    config->m_l0icache_hit_latency = atoi(value.c_str());
  else if (name == "l1icache_hit_latency")
    config->m_l1icache_hit_latency = atoi(value.c_str());
  else if (name == "dtlb_config")
    config->m_dtlb_config = value;
  else if (name == "itlb_config")
    config->m_itlb_config = value;
  else if (name == "tlb_hit_latency")
    config->m_tlb_hit_latency = atoi(value.c_str());
  else if (name == "max_kernel_register") 
    config->m_max_kernel_register = atoi(value.c_str());
  else if (name == "max_kernel_launch") 
    config->m_max_kernel_launch = atoi(value.c_str());
  else if (name == "max_dma_list_size")
    config->m_max_dma_list_size = atoi(value.c_str());
  else if (name == "inst_buffer_size")
    config->m_inst_buffer_size = atoi(value.c_str());
  else if (name == "uthread_slots")
    config->m_uthread_slots = atoi(value.c_str());
  else if (name == "max_inst_issue")
    config->m_max_inst_issue = atoi(value.c_str());
  else if (name == "inst_fetch_per_cycle")
    config->m_inst_fetch_per_cycle = atoi(value.c_str());
  else if (name == "num_i_units")
    config->m_num_i_units = atoi(value.c_str());
  else if (name == "num_f_units")
    config->m_num_f_units = atoi(value.c_str());
  else if (name == "num_sf_units")
    config->m_num_sf_units = atoi(value.c_str());
  else if (name == "num_ldst_units")
    config->m_num_ldst_units = atoi(value.c_str());
  else if (name == "num_address_units")
    config->m_num_address_units = atoi(value.c_str());
  else if (name == "num_spad_units")
    config->m_num_spad_units = atoi(value.c_str());
  else if (name == "num_v_i_units")
    config->m_num_v_i_units = atoi(value.c_str());
  else if (name == "num_v_f_units")
    config->m_num_v_f_units = atoi(value.c_str());
  else if (name == "num_v_sf_units")
    config->m_num_v_sf_units = atoi(value.c_str());
  else if (name == "num_v_ldst_units")
    config->m_num_v_ldst_units = atoi(value.c_str());
  else if (name == "num_v_address_units")
    config->m_num_v_address_units = atoi(value.c_str());
  else if (name == "num_v_spad_units")
    config->m_num_v_spad_units = atoi(value.c_str());
  else if (name == "use_unified_alu")
    config->m_use_unified_alu = atoi(value.c_str());
  else if (name == "ndp_op_latencies") {
    std::stringstream ss(value);
    for (int i = 0; i < NUM_ALU_OP_TYPE; i++) {
      std::string val;
      std::getline(ss, val, ',');
      config->m_ndp_op_latencies[i] = atoi(val.c_str());
    }
  } else if (name == "ndp_op_initialiation_interval") {
    std::stringstream ss(value);
    for (int i = 0; i < NUM_ALU_OP_TYPE; i++) {
      std::string val;
      std::getline(ss, val, ',');
      config->m_ndp_op_initialiation_interval[i] = atoi(val.c_str());
    }
  } else if (name == "freq") {
    std::string val;
    std::stringstream ss(value);
    std::getline(ss, val, ',');
    printf("core freq %f\t", std::stod(val));
    config->m_core_period = 1 / (std::stod(val) MHz);
    std::getline(ss, val, ',');
    printf("dram freq %f\t", std::stod(val));
    config->m_dram_period = 1 / (std::stod(val) MHz);
    std::getline(ss, val, ',');
    printf("buffer ndp freq %f\t", std::stod(val));
    config->m_buffer_ndp_period = 1 / (std::stod(val) MHz);
    std::getline(ss, val, ',');
    printf("link freq %f\t", std::stod(val));
    config->m_link_period = 1 / (std::stod(val) MHz);
    std::getline(ss, val, ',');
    printf("cache freq %f\t", std::stod(val));
    config->m_cache_period = 1 / (std::stod(val) MHz);
    std::getline(ss, val, ',');
    printf("buffer dram freq %f\n", std::stod(val));
    config->m_buffer_dram_period = 1 / (std::stod(val) MHz);
  } else if (name == "log_interval") {
    config->m_log_interval = atoi(value.c_str());
  } else if (name == "enable_dram_tlb_miss_handling") {
    config->m_enable_dram_tlb_miss_handling = atoi(value.c_str());
  } else if (name == "dram_tlb_miss_handling_latency") {
    config->m_dram_tlb_miss_handling_latency = atoi(value.c_str());
  } else if (name == "enable_random_bi") {
    config->enable_random_bi = atoi(value.c_str());
  } else if (name == "random_bi_rate") {
    config->random_bi_rate = atof(value.c_str());
  } else if (name == "bi_addr_unit_size") {
    config->bi_addr_unit_size = atoi(value.c_str());
  } else if (name == "bi_host_latency") {
    config->bi_host_latency = atoi(value.c_str());
  } else if (name == "shmem_option") {
      std::string val;
      std::stringstream ss(value);
      config->shmem_option.clear();
      while(std::getline(ss, val, ',')) {
        config->shmem_option.push_back(atoi(val.c_str()));
      }
  } else if (name == "adaptive_l1d") {
    config->m_adaptive_l1d = atoi(value.c_str());
  } else if (name == "coarse_grained") {
    config->m_coarse_grained = atoi(value.c_str());
  } else if (name == "sc_work") {
    config->m_sc_work = atoi(value.c_str());
  } else if (name == "subcore_coarse_grained") {
    config->m_sub_core_coarse_grained = atoi(value.c_str());
  } else if (name == "all_vector") {
    config->m_all_vector = atoi(value.c_str());
  } else if (name == "sc_work_addr") {
    config->m_sc_work_addr = atoi(value.c_str());
  }
  else {
    printf("Unknown config parameter %s\n", name.c_str());
  }
#endif
}

void M2NDPParser::decode_memory_mapping(M2NDPConfig *config, std::string addr) {
  const char *cmapping = addr.c_str();
  config->m_num_channels = 1;
  config->m_addrdec_mask[memory_decode::DEVICE] = 0x0;
  config->m_addrdec_mask[memory_decode::RAM] = 0x0;
  config->m_addrdec_mask[memory_decode::BANK_GROUP] = 0x0;
  config->m_addrdec_mask[memory_decode::BANK] = 0x0;
  config->m_addrdec_mask[memory_decode::MEM_ROW] = 0x0;
  config->m_addrdec_mask[memory_decode::COLUMN] = 0x0;
  config->m_addrdec_mask[memory_decode::CH] = 0x0;
  config->m_addrdec_mask[memory_decode::BURST] = 0x0;
  int ofs = 63;
  while ((*cmapping) != '\0') {
    switch (*cmapping) {
      case 'G':
        config->m_addrdec_mask[memory_decode::DEVICE] |= (1ULL << ofs);
        ofs--;
        break;
      case 'M':
        config->m_addrdec_mask[memory_decode::RAM] |= (1ULL << ofs);
        ofs--;
        break;
      case 'B':
        config->m_addrdec_mask[memory_decode::BANK_GROUP] |= (1ULL << ofs);
        ofs--;
        break;
      case 'b':
        config->m_addrdec_mask[memory_decode::BANK] |= (1ULL << ofs);
        ofs--;
        break;
      case 'R':
        config->m_addrdec_mask[memory_decode::MEM_ROW] |= (1ULL << ofs);
        ofs--;
        break;
      case 'c':
        config->m_addrdec_mask[memory_decode::COLUMN] |= (1ULL << ofs);
        ofs--;
        break;
      case 'C':
        config->m_addrdec_mask[memory_decode::CH] |= (1ULL << ofs);
        ofs--;
        config->m_num_channels *= 2;
        break;
      case 'S':
        config->m_addrdec_mask[BURST] |= (1ULL << ofs);
        ofs--;
        break;
      // ignore bit
      case '0':
        ofs--;
        break;
      // ignore character
      case '|':
      case ' ':
      case '.':
        break;
      default:
        spdlog::error("[MAPPING] {}", *cmapping);
        assert(0 && "[ADDRDECODER] NOT MATCHING CHARACTOR\n");
    }
    cmapping += 1;
  }

  if (ofs != -1) {
    fprintf(stderr,
            "ERROR: Invalid address mapping length (%d) in option '%s'\n",
            63 - ofs, cmapping);
    assert(ofs == -1);
  }
}

}  // namespace NDPSim