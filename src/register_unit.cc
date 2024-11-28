#include "register_unit.h"

#include <set>

#include "ndp_instruction.h"
namespace NDPSim {

RegisterUnit::RegisterUnit(int num_xreg, int num_freg, int num_vreg) {
  m_num_xregs = num_xreg;
  m_num_fregs = num_freg;
  m_num_vregs = num_vreg;
  for (int v = 0; v < m_register_stats.REG_STAT_NUM; v++)
    m_register_stats.register_stats[v] = 0;
  for (int v = 0; v < m_num_xregs; v++) {
    m_free_xregs.push_back(REG_PX_BASE + v);
  }
  for (int v = 0; v < m_num_fregs; v++) {
    m_free_fregs.push_back(REG_PF_BASE + v);
  }
  for (int v = 0; v < m_num_vregs; v++) {
    m_free_vregs.push_back(REG_PV_BASE + v);
  }
  m_xreg_table.resize(m_num_xregs);
  m_freg_table.resize(m_num_fregs);
  m_vreg_table.resize(m_num_vregs);
}

bool RegisterUnit::RenameFull(InstColumn* inst_column) {
  return (inst_column->xregs > m_free_xregs.size() ||
          inst_column->fregs > m_free_fregs.size() ||
          inst_column->vregs > m_free_vregs.size());
}

void RegisterUnit::RenamePush(InstColumn* inst_column, int packet_id) {
  assert(!RenameFull(inst_column));
  inst_column->insts = Convert(inst_column->insts, packet_id, inst_column->req);
  m_after_rename.push_back(inst_column);
}

bool RegisterUnit::RenameEmpty() { return m_after_rename.empty(); }

InstColumn* RegisterUnit::RenameTop() { return m_after_rename.front(); }

void RegisterUnit::RenamePop() { m_after_rename.pop_front(); }

bool RegisterUnit::CheckReady(int reg) {
  return m_not_ready.find(reg) == m_not_ready.end();
}

void RegisterUnit::SetNotReady(int reg) { m_not_ready.insert(reg); }

void RegisterUnit::SetReady(int reg) { m_not_ready.erase(reg); }

std::deque<NdpInstruction> RegisterUnit::Convert(
    std::deque<NdpInstruction> insts, int packet_id, RequestInfo* req) {
  std::deque<NdpInstruction> renamed_insts;
  RegisterMapKey key = packet_id;
  
  //Initialize x1, and x2 
  int x1_reg = REG_X_BASE + 1; // ADDR 
  int x2_reg = REG_X_BASE + 2; // OFFSET
  int px1 = m_free_xregs.front();
  m_free_xregs.pop_front();
  m_xreg_mapping[key][x1_reg] = px1;
  int px2 = m_free_xregs.front();
  m_free_xregs.pop_front();
  m_xreg_mapping[key][x2_reg] = px2;
  m_xreg_table[px1 - REG_PX_BASE] = req->addr;
  m_xreg_table[px2 - REG_PX_BASE] = req->offset;

  cvt_vlmul_status = IMM_M1;

  for (int i = 0; i < insts.size(); i++) {
    NdpInstruction inst = insts[i];
    NdpInstruction renamed = inst;
    try {
      if (inst.opcode == VSETVLI) cvt_vlmul_status = inst.src[2];
      if (inst.CheckBranchOp()) {
        if (inst.opcode == Opcode::J) {
          renamed.src[2] = inst.src[2];
        } else {
          renamed.src[0] = LookUpX(key, inst.src[0]);
          renamed.src[2] = inst.src[2];
          if (inst.opcode == Opcode::BEQ || inst.opcode == Opcode::BGT ||
              inst.opcode == Opcode::BLT || inst.opcode == Opcode::BNE ||
              inst.opcode == Opcode::BLE || inst.opcode == Opcode::BGE) {
            renamed.src[1] = LookUpX(key, inst.src[1]);
          }
        }
      } else if (inst.CheckSegOp()) {
        switch (inst.operand_type) {
          case OperandType::V:
            renamed.src[0] = LookUpX(key, inst.src[0]);
            renamed.src[1] = LookUpV(key, inst.src[1]);
            renamed.dest = DestRenameV(key, renamed);
            for (int i = 0; i < inst.segCnt; i++)
              renamed.segDest[i] = DestRenameSegV(key, renamed, i);
            renamed.segCnt = inst.segCnt;
            break;
          default:
            break;
        }
      } else if (inst.CheckVectorOp()) {
        /*Register Renaming for Vector Opcodes*/
        switch (inst.operand_type) {
          case OperandType::M:
            if (inst.opcode == VMSET)
              renamed.dest = DestRenameV(key, inst);
            else if (inst.opcode == VFIRST) {
              renamed.dest = DestRenameX(key, inst);
              renamed.src[0] = LookUpV(key, inst.src[0]);
            }
            break;
          case OperandType::X_S:
            renamed.dest = DestRenameX(key, inst);
            renamed.src[0] = LookUpV(key, inst.src[0]);
            break;
          case OperandType::V:
            if (inst.opcode == VFSQRT) {
              renamed.dest = DestRenameV(key, inst);
              renamed.src[0] = LookUpV(key, inst.src[0]);
              renamed.src[1] = LookUpV(key, inst.src[1]);
              break;
            }
            if (inst.opcode == VID || inst.opcode == VFEXP) {
              renamed.src[0] = LookUpV(key, inst.src[0]);
              renamed.dest = DestRenameV(key, inst);
              break;
            }
            renamed.src[0] = LookUpX(key, inst.src[0]);
            if (inst.opcode == Opcode::VSE8 || inst.opcode == Opcode::VSE16 ||
                inst.opcode == Opcode::VSE32 || inst.opcode == Opcode::VSE64) {
              renamed.dest = LookUpV(key, inst.dest);
            } else if (inst.opcode == Opcode::VLE8 ||
                      inst.opcode == Opcode::VLE16 ||
                      inst.opcode == Opcode::VLE32 ||
                      inst.opcode == Opcode::VLE64) {
              renamed.dest = DestRenameV(key, inst);
            } else if (inst.opcode == Opcode::VLUXEI32 ||
                      inst.opcode == Opcode::VLUXEI64) {
              renamed.dest = DestRenameV(key, inst);
              renamed.src[2] = LookUpV(key, inst.src[2]);
              if (inst.src[3] != -1) renamed.src[3] = LookUpV(key, inst.src[3]);
            } else if (inst.opcode == Opcode::VSUXEI32 ||
                      inst.opcode == Opcode::VSUXEI64) {
              renamed.src[0] = LookUpX(key, inst.src[0]);
              renamed.dest = LookUpV(key, inst.dest);
              renamed.src[2] = LookUpV(key, inst.src[2]);
              renamed.src[3] = LookUpV(key, inst.src[3]);
              // assert(0);
            } else if (inst.CheckAmoOp()) {
              if (inst.dest >= REG_V_BASE && inst.dest < REG_PX_BASE)
                renamed.dest = LookUpV(key, inst.dest);
              else
                renamed.dest = -1;
              renamed.src[0] = LookUpX(key, inst.src[0]);
              renamed.src[1] = LookUpV(key, inst.src[1]);
              renamed.src[2] = LookUpV(key, inst.src[2]);
              if (inst.src[3] != -1) renamed.src[3] = LookUpV(key, inst.src[3]);
            } else {  // VAMO instructions can have x0 or vd destination register.
                      // Be Careful!
              renamed.src[1] = LookUpV(key, inst.src[1]);
              renamed.src[2] = LookUpV(key, inst.src[2]);
            }
            // Todo: load store by stride or index
            break;
          case OperandType::WV:
          case OperandType::VV:
          case OperandType::VS:
          case OperandType::VM:
          case OperandType::MM:
            renamed.src[0] = LookUpV(key, inst.src[0]);
            renamed.src[1] = LookUpV(key, inst.src[1]);
            if (CheckExistVreg(key, inst.src[2]))
              renamed.src[2] = LookUpV(key, inst.src[2]);
            renamed.dest = DestRenameV(key, inst);
            break;
          case OperandType::VI:
          case OperandType::V_V:
          case OperandType::XU_F_V:
          case OperandType::X_F_V:
          case OperandType::F_XU_V:
          case OperandType::F_X_V:
          case OperandType::F_F_V:
          case OperandType::F_F_W:
            renamed.src[0] = LookUpV(key, inst.src[0]);
            renamed.dest = DestRenameV(key, renamed);
            break;
          case OperandType::V_I:
            if (inst.src[1] != -1) renamed.src[1] = LookUpV(key, inst.src[1]);
            renamed.dest = DestRenameV(key, renamed);
            break;
          case OperandType::V_X:
          case OperandType::S_X:
            renamed.src[0] = LookUpX(key, inst.src[0]);
            if (inst.src[1] != -1) renamed.src[1] = LookUpV(key, inst.src[1]);
            renamed.dest = DestRenameV(key, renamed);
            break;
          case OperandType::VX:
          case OperandType::WX:
            renamed.src[0] = LookUpV(key, inst.src[0]);
            renamed.src[1] = LookUpX(key, inst.src[1]);
            renamed.dest = DestRenameV(key, renamed);
            break;
        case OperandType::VF:
          if (inst.opcode == VFMSUB) {
          renamed.src[0] = LookUpF(key, inst.src[0]);
          renamed.src[1] = LookUpV(key, inst.src[1]);
          } else {
            renamed.src[0] = LookUpV(key, inst.src[0]);
            renamed.src[1] = LookUpF(key, inst.src[1]);
          }
          renamed.dest = DestRenameV(key, renamed);
          break;
          case OperandType::S_F:
          case OperandType::V_F:
            renamed.src[0] = LookUpF(key, inst.src[0]);
            renamed.dest = DestRenameV(key, renamed);
            break;
          case OperandType::F_S:
            renamed.src[0] = LookUpV(key, inst.src[0]);
            renamed.dest = DestRenameF(key, renamed);
            break;
          default:
            // vsetvli ..
            break;
        }
      } else if (inst.CheckScalarOp()) {
        /*Register Renaming for Scalar Opcodes*/
        switch (inst.opcode) {
          case Opcode::ADD:
          case Opcode::SUB:
          case Opcode::MUL:
          case Opcode::DIV:
          case Opcode::REM:
          case Opcode::AND:
          case Opcode::SLL:
          case Opcode::SRL:
          case Opcode::OR:
            renamed.src[0] = LookUpX(key, inst.src[0]);
            renamed.src[1] = LookUpX(key, inst.src[1]);
            renamed.dest = DestRenameX(key, inst);
            break;
          case Opcode::ADDI:
          case Opcode::MULI:
          case Opcode::ANDI:
          case Opcode::SLLI:
          case Opcode::SRLI:
          case Opcode::ORI:
            renamed.src[0] = LookUpX(key, inst.src[0]);
            renamed.dest = DestRenameX(key, inst);
            break;
          case Opcode::LI:
            renamed.dest = DestRenameX(key, inst);
            break;
          case Opcode::LBU:
          case Opcode::LB:
          case Opcode::LW:
          case Opcode::LD:
            renamed.src[0] = LookUpX(key, inst.src[0]);
            renamed.src[1] = inst.src[1];
            renamed.dest = DestRenameX(key, inst);
            break;
          case Opcode::SB:
          case Opcode::SW:
          case Opcode::SD:
            renamed.src[0] = LookUpX(key, inst.src[0]);
            renamed.src[1] = inst.src[1];
            renamed.dest = LookUpX(key, inst.dest);
            break;
          case Opcode::AMOADD:
          case Opcode::AMOMAX:
            renamed.dest = DestRenameX(key, inst);
            renamed.src[1] = LookUpX(key, inst.src[1]);
            renamed.src[2] = LookUpX(key, inst.src[2]);
            break;
          case Opcode::FAMOADD:
          case Opcode::FAMOADDH:
          case Opcode::FAMOMAX:
            renamed.dest = DestRenameF(key, inst);
            renamed.src[1] = LookUpF(key, inst.src[1]);
            renamed.src[2] = LookUpX(key, inst.src[2]);
            break;
          case Opcode::CSRW:
            renamed.src[0] = LookUpX(key, inst.src[0]);
            break;
          case Opcode::FADD:
          case Opcode::FSUB:
          case Opcode::FDIV:
          case Opcode::FMUL:
            renamed.src[0] = LookUpF(key, inst.src[0]);
            renamed.src[1] = LookUpF(key, inst.src[1]);
            renamed.dest = DestRenameF(key, inst);
            break;
          case Opcode::FLW:
            renamed.src[0] = LookUpX(key, inst.src[0]);
            renamed.src[1] = inst.src[1];
            renamed.dest = DestRenameF(key, inst);
            break;
          case Opcode::FSW:
            renamed.src[0] = LookUpX(key, inst.src[0]);
            renamed.dest = LookUpF(key, inst.dest);
            break;
          case Opcode::FMV:
            if (inst.operand_type == X_W) {
              renamed.src[0] = LookUpF(key, inst.src[0]);
              renamed.dest = DestRenameX(key, inst);
            } else if (inst.operand_type == W_X) {
              renamed.src[0] = LookUpX(key, inst.src[0]);
              renamed.dest = DestRenameF(key, inst);
            }
            break;
          case Opcode::FEXP:
            renamed.src[0] = LookUpF(key, inst.src[0]);
            renamed.dest = DestRenameF(key, inst);
            break;
          case Opcode::EXIT:
            break;
        }
      }
    }
    catch (const std::runtime_error& error) {
      spdlog::error("=============================REGISTER UNIT PANIC=============================");
      for (int j= i-5 > 0 ?: 0; j<i; j++)
        spdlog::error("[{}/{}] {}", j, insts.size(), insts[j].sInst);
      spdlog::error("[{}/{}] {} <-- triggered panic!!", i, insts.size(), inst.sInst);
      spdlog::error("Caught: {}\n", error.what());
      throw error;
    }
    renamed_insts.push_back(renamed);
  }
  return renamed_insts;
}

void RegisterUnit::FreeRegs(int packet_id) {
  RegisterMapKey key = packet_id;
  RegisterBinding mapping_to_free = m_xreg_mapping[key];
  for (auto it : mapping_to_free) {
    int reg = it.second;
    m_free_xregs.push_back(reg);
  }
  m_xreg_mapping.erase(key);
  mapping_to_free = m_freg_mapping[key];
  for (auto it : mapping_to_free) {
    int reg = it.second;
    m_free_fregs.push_back(reg);
  }
  m_freg_mapping.erase(key);
  mapping_to_free = m_vreg_mapping[key];
  for (auto it : mapping_to_free) {
    int reg = it.second;
    m_free_vregs.push_back(reg);
    if (CheckDoubleReg(reg)) {
      m_free_vregs.push_back(m_double_vreg[reg]);
      m_double_vreg.erase(reg);
    }
  }
  m_vreg_mapping.erase(key);
}

bool RegisterUnit::CheckDoubleReg(int reg) {
  if (reg < REG_PV_BASE) return false;
  return m_double_vreg.find(reg) != m_double_vreg.end();
}

bool RegisterUnit::CheckSpecialReg(int reg) { return reg > SPCIAL_REG_START; }

int RegisterUnit::LookUpX(RegisterMapKey key, int reg) {
  if (m_xreg_mapping.find(key) != m_xreg_mapping.end() &&
      m_xreg_mapping[key].find(reg) != m_xreg_mapping[key].end())  // 1, 0
    return m_xreg_mapping[key][reg];
  else if (CheckSpecialReg(reg))
    return reg;
  throw std::runtime_error("RegisterUnit::LookUpX: Register not found");
}

int RegisterUnit::LookUpF(RegisterMapKey key, int reg) {
  if (m_freg_mapping.find(key) != m_freg_mapping.end() &&
      m_freg_mapping[key].find(reg) != m_freg_mapping[key].end())
    return m_freg_mapping[key][reg];
  else if (CheckSpecialReg(reg))
    return reg;
  throw std::runtime_error("RegisterUnit::LookUpF: Register not found");
}

int RegisterUnit::LookUpV(RegisterMapKey key, int reg) {
  if (m_vreg_mapping.find(key) != m_vreg_mapping.end() &&
      m_vreg_mapping[key].find(reg) != m_vreg_mapping[key].end())
    return m_vreg_mapping[key][reg];
  else if (CheckSpecialReg(reg))
    return reg;
  throw std::runtime_error("RegisterUnit::LookUpV: Register not found");
}

int RegisterUnit::DestRenameX(RegisterMapKey key, NdpInstruction inst) {
  if (m_xreg_mapping.find(key) != m_xreg_mapping.end() &&
      m_xreg_mapping[key].find(inst.dest) != m_xreg_mapping[key].end()) {
    return m_xreg_mapping[key][inst.dest];
  }
  int result = m_free_xregs.front();
  m_free_xregs.pop_front();
  m_xreg_mapping[key][inst.dest] = result;
  return result;
}

int RegisterUnit::DestRenameF(RegisterMapKey key, NdpInstruction inst) {
  if (m_freg_mapping.find(key) != m_freg_mapping.end() &&
      m_freg_mapping[key].find(inst.dest) != m_freg_mapping[key].end()) {
    return m_freg_mapping[key][inst.dest];
  }
  int result = m_free_fregs.front();
  m_free_fregs.pop_front();
  m_freg_mapping[key][inst.dest] = result;
  return result;
}

int RegisterUnit::DestRenameV(RegisterMapKey key, NdpInstruction inst) {
  if (m_vreg_mapping.find(key) != m_vreg_mapping.end() &&
      m_vreg_mapping[key].find(inst.dest) != m_vreg_mapping[key].end()) {
    return m_vreg_mapping[key][inst.dest];
  }
  int result = m_free_vregs.front();
  m_free_vregs.pop_front();
  m_vreg_mapping[key][inst.dest] = result;
  if (!inst.CheckNarrowingOp() && CheckDoubleReg(inst.src[0]) ||
      inst.CheckWideningOp() && !CheckDoubleReg(inst.src[0]) ||
      cvt_vlmul_status == IMM_M2) {
    // If not narrow operation and src is double reg, dest is double reg
    m_double_vreg[result] = m_free_vregs.front();
    m_free_vregs.pop_front();
  }
  return result;
}

int RegisterUnit::DestRenameSegV(RegisterMapKey key, NdpInstruction inst,
                                 int index) {
  if (m_vreg_mapping.find(key) != m_vreg_mapping.end() &&
      m_vreg_mapping[key].find(inst.segDest[index]) !=
          m_vreg_mapping[key].end()) {
    return m_vreg_mapping[key][inst.segDest[index]];
  }
  int result = m_free_vregs.front();
  m_free_vregs.pop_front();
  m_vreg_mapping[key][inst.segDest[index]] = result;
  if (!inst.CheckNarrowingOp() && CheckDoubleReg(inst.src[0]) ||
      inst.CheckWideningOp() && !CheckDoubleReg(inst.src[0])) {
    // If not narrow operation and src is double reg, dest is double reg
    m_double_vreg[result] = m_free_vregs.front();
    m_free_vregs.pop_front();
  }
  return result;
}

int RegisterUnit::GetAnother(int reg) {
  assert(CheckDoubleReg(reg));
  assert(reg >= REG_PV_BASE && reg < REG_PV_BASE + m_num_vregs);
  return m_double_vreg[reg];
}

bool RegisterUnit::CheckExistVreg(RegisterMapKey key, int reg) {
  // Todo : Special register implement
  if (m_vreg_mapping.find(key) != m_vreg_mapping.end() &&
      m_vreg_mapping[key].find(reg) != m_vreg_mapping[key].end())
    return true;
  else
    return false;
}

bool RegisterUnit::CheckExistRenameVreg(int reg) { // TODO: check the if condition, it is not correct(issue)
  if (std::find(m_free_vregs.begin(), m_free_vregs.end(), reg) !=
      m_free_vregs.end())
    return false;
  else
    return true;
}

int64_t RegisterUnit::ReadXreg(int reg, Context& context) {
  int64_t x_value;
  if (reg == REG_NDP_ID)
    x_value = context.ndp_id;
  else if (reg == REG_ADDR)
    x_value = context.request_info->addr;
  else if (reg == REG_REQUEST_OFFSET)
    x_value = context.request_info->offset;
  else if (reg == REG_X0) {
    x_value = 0;
  } else if (reg >= REG_PX_BASE && reg < REG_PX_BASE + m_num_xregs)
    x_value = m_xreg_table[reg - REG_PX_BASE];
  else
    throw std::runtime_error("RegisterUnit::ReadXreg: Invalid register");

  m_register_stats.register_stats[RegisterStats::X_READ]++;
  return x_value;
}

float RegisterUnit::ReadFreg(int reg, Context& context) {
  if (reg >= REG_PF_BASE && reg < REG_PF_BASE + m_num_fregs) {
    m_register_stats.register_stats[RegisterStats::F_READ]++;
    return m_freg_table[reg - REG_PF_BASE];
  } else
    throw std::runtime_error("RegisterUnit::ReadFreg: Invalid register");
}

VectorData RegisterUnit::ReadVreg(int reg, Context& context) {
  VectorData result;
  if (reg >= REG_PV_BASE && reg < REG_PV_BASE + m_num_vregs) {
    result = m_vreg_table[reg - REG_PV_BASE];
    if (CheckDoubleReg(reg) && result.GetVlen() != (PACKET_SIZE * BYTE_BIT *
                                                    (context.csr->vtype_vlmul) /
                                                    context.csr->vtype_vsew)) {
      result.Append(m_vreg_table[m_double_vreg[reg] - REG_PV_BASE]);
      m_register_stats.register_stats[RegisterStats::V_READ]++;
    }
  } else

  /* Sanity check */
  if (result.GetType() != VMASK && (result.GetVlen() * result.GetPrecision() / (UNION_DATA_SIZE / BYTE_BIT)) > result.GetVectorSize())
    throw std::runtime_error("RegisterUnit::ReadVreg: Overflow Detected!");

  m_register_stats.register_stats[RegisterStats::V_READ]++;
  return result;
}

void RegisterUnit::WriteXreg(int reg, int64_t value, Context& context) {
  if (reg >= REG_PX_BASE && reg < REG_PX_BASE + m_num_xregs) {
    m_xreg_table[reg - REG_PX_BASE] = value;
    m_register_stats.register_stats[RegisterStats::X_WRITE]++;
  } else
    throw std::runtime_error("RegisterUnit::WriteXreg: Invalid register");
}

void RegisterUnit::WriteFreg(int reg, float value, Context& context) {
  if (reg >= REG_PF_BASE && reg < REG_PF_BASE + m_num_fregs) {
    m_freg_table[reg - REG_PF_BASE] = value;
    m_register_stats.register_stats[RegisterStats::F_WRITE]++;
  } else
    throw std::runtime_error("RegisterUnit::WriteFreg: Invalid register");
}

void RegisterUnit::WriteFreg(int reg, half value, Context& context) {
  if (reg >= REG_PF_BASE && reg < REG_PF_BASE + m_num_fregs) {
    m_freg_table[reg - REG_PF_BASE] = value;
    m_register_stats.register_stats[RegisterStats::F_WRITE]++;
  } else
    throw std::runtime_error("RegisterUnit::WriteFreg: Invalid register");
}

void RegisterUnit::WriteVreg(int reg, VectorData value, Context& context) {
  /* Sanity check */
  if (value.GetType() != VMASK && (value.GetVlen() * value.GetPrecision() / (UNION_DATA_SIZE / BYTE_BIT)) > value.GetVectorSize())
    throw std::runtime_error("RegisterUnit::WriteVreg: Overflow Detected!");

  if (reg >= REG_PV_BASE && reg < REG_PV_BASE + m_num_vregs) {
    // assert(GetVregIndex(reg) % contex.csr->vtype_vlmul == 0);
    if (context.csr->vtype_vlmul == 2 || value.GetDoubleReg()) {
      std::array<VectorData, 2> values = value.Split();
      m_vreg_table[reg - REG_PV_BASE] = values[0];
      if (m_double_vreg[reg] < REG_PV_BASE)
        throw std::runtime_error("RegisterUnit::Double register not allocated");
      m_vreg_table[m_double_vreg[reg] - REG_PV_BASE] = values[1];
      m_register_stats.register_stats[RegisterStats::V_WRITE]++;
    } else {
      m_vreg_table[reg - REG_PV_BASE] = value;
    }
    m_register_stats.register_stats[RegisterStats::V_WRITE]++;
  } else
    throw std::runtime_error("RegisterUnit::WriteVreg: Invalid register");
}

bool RegisterUnit::IsVreg(int reg_id) {
  if (reg_id >= REG_PV_BASE && reg_id < REG_PV_BASE + m_num_vregs)
    return true;
  else
    return false;
}

bool RegisterUnit::IsXreg(int reg_id) {
  if (reg_id >= REG_PX_BASE && reg_id < REG_PX_BASE + m_num_xregs)
    return true;
  else
    return false;
}

bool RegisterUnit::IsFreg(int reg_id) {
  if (reg_id >= REG_PF_BASE && reg_id < REG_PF_BASE + m_num_fregs)
    return true;
  else
    return false;
}

RegisterStats RegisterUnit::get_register_stats() { return m_register_stats; }

void RegisterUnit::dump_current_state() {
  spdlog::error("RegisterUnit::dump_current_state");
  for (auto not_ready_reg : m_not_ready) {
    spdlog::error("RegisterUnit::dump_current_state: not_ready_reg: {}",
                  not_ready_reg);
  }
}

void RegisterUnit::DumpRegisterFile(int packet_id) {
  std::map<RegisterMapKey, std::map<int, int>> ordered_reg;
  /* Dump integer Register file */
  for (auto& [key, val] : m_xreg_mapping)
    if (packet_id == -1 || key == packet_id)
      for (auto& [reg, renamed_reg] : val)
        ordered_reg[key][reg] = renamed_reg;

  spdlog::error("==========================Interger Register File==========================");
  for (auto& [key, val] : ordered_reg) {
    for (auto& [reg, renamed_reg] : val) {
      spdlog::debug("x{}: {}", reg - REG_X_BASE, m_xreg_table[renamed_reg - REG_PX_BASE]);
    }
  }

  ordered_reg.clear();
  /* Dump floating point Register file */
  for (auto& [key, val] : m_freg_mapping)
    if (packet_id == -1 || key == packet_id)
      for (auto& [reg, renamed_reg] : val)
        ordered_reg[key][reg] = renamed_reg;

  spdlog::error("=======================Floating point Register File=======================");
  for (auto& [key, val] : ordered_reg) {
    for (auto& [reg, renamed_reg] : val) {
      spdlog::debug("f{}: {}", reg - REG_F_BASE, m_freg_table[renamed_reg - REG_PF_BASE]);
    }
  }

  ordered_reg.clear();
  /* Dump Vector Register file */
  for (auto& [key, val] : m_vreg_mapping)
    if (packet_id == -1 || key == packet_id)
      for (auto& [reg, renamed_reg] : val)
        ordered_reg[key][reg] = renamed_reg;

  spdlog::error("===========================Vector Register File===========================");
  for (auto& [key, val] : ordered_reg) {
    for (auto& [reg, renamed_reg] : val) {
      std::cout << "v" << reg - REG_V_BASE << ": ";
      spdlog::debug("v{}: {}", reg - REG_V_BASE, m_vreg_table[renamed_reg - REG_PV_BASE].toString());
    }
  }
  std::cout << std::endl;
}
}  // namespace NDPSim