#ifdef TIMING_SIMULATION
#include "execution_unit.h"

#include "memory_map.h"
namespace NDPSim {

ExecutionUnit::ExecutionUnit(M2NDPConfig *config, int ndp_id, int sub_core_id, RegisterUnit *register_unit,
                std::queue<Context> *finished_contexts,
                fifo_pipeline<std::pair<NdpInstruction, Context>> *to_ldst_unit,
                fifo_pipeline<std::pair<NdpInstruction, Context>> *to_spad_unit,
                fifo_pipeline<std::pair<NdpInstruction, Context>> *to_v_ldst_unit,
                fifo_pipeline<std::pair<NdpInstruction, Context>> *to_v_spad_unit,
                NdpStats *stats)
    : m_ndp_id(ndp_id),
      m_sub_core_id(sub_core_id),
      m_cycle(0),
      m_config(config),
      m_register_unit(register_unit),
      m_finished_contexts(finished_contexts),
      m_to_ldst_unit(to_ldst_unit),
      m_to_spad_unit(to_spad_unit),
      m_to_v_ldst_unit(to_v_ldst_unit),
      m_to_v_spad_unit(to_v_spad_unit),
      m_stats(stats) {
  for (int i = 0; i < config->get_num_i_units(); i++) {
    m_i_units.push_back(ExecutionDelayQueue("i_unit_" + std::to_string(i)));
  }
  for (int i = 0; i < config->get_num_f_units(); i++) {
    m_f_units.push_back(ExecutionDelayQueue("f_unit_" + std::to_string(i)));
  }
  for (int i = 0; i < config->get_num_sf_units(); i++) {
    m_sf_units.push_back(ExecutionDelayQueue("sf_unit_" + std::to_string(i)));
  }
  for (int i = 0; i < config->get_num_address_units(); i++) {
    m_address_units.push_back(
        ExecutionDelayQueue("address_unit_" + std::to_string(i)));
  }
  for (int i = 0; i < config->get_num_v_i_units(); i++) {
    m_v_i_units.push_back(ExecutionDelayQueue("v_i_unit_" + std::to_string(i)));
  }
  for (int i = 0; i < config->get_num_v_f_units(); i++) {
    m_v_f_units.push_back(ExecutionDelayQueue("v_f_unit_" + std::to_string(i)));
  }
  for (int i = 0; i < config->get_num_v_sf_units(); i++) {
    m_v_sf_units.push_back(
        ExecutionDelayQueue("v_sf_unit_" + std::to_string(i)));
  }
  for (int i = 0; i < config->get_num_v_address_units(); i++) {
    m_v_address_units.push_back(
        ExecutionDelayQueue("v_address_unit_" + std::to_string(i)));
  }
}

bool ExecutionUnit::active() {
  return  !m_to_ldst_unit->empty()|| !m_to_spad_unit->empty() || 
          !m_to_v_ldst_unit->empty() || !m_to_v_spad_unit->empty() ||
          !check_unit_finished();
}

void ExecutionUnit::cycle() {
  for (auto &i_unit : m_i_units) {
    i_unit.cycle();
    if (i_unit.empty()) continue;
    NdpInstruction inst = i_unit.top().first;
    Context context = i_unit.top().second;
    if (inst.CheckDestWrite()) {
      m_register_unit->SetReady(inst.dest);
    }
    if (inst.CheckBranchOp()) {
      context.csr->block = false;
      int branch_target = inst.IsBranch();
      if (branch_target != -1)
        context.csr->pc = context.loop_map->at(branch_target);
      else
        context.csr->pc++;

      if (context.max_pc < context.csr->pc) context.last_inst = true;
    }
    if (context.last_inst) {
      m_finished_contexts->push(context);
    }
    i_unit.pop();
  }
  process_default_execution_units(m_f_units);
  process_default_execution_units(m_sf_units);
  process_default_execution_units(m_v_i_units);
  process_default_execution_units(m_v_f_units);
  process_default_execution_units(m_v_sf_units);

  for (auto &address_unit : m_address_units) {
    address_unit.cycle();
    process_address_inst(address_unit);
  }
  for (auto &v_address_unit : m_v_address_units) {
    v_address_unit.cycle();
    process_address_inst(v_address_unit);
  }
}

void ExecutionUnit::process_default_execution_units(
    std::vector<ExecutionDelayQueue> &units) {
  for (auto &unit : units) {
    unit.cycle();
    if (unit.empty()) continue;
    NdpInstruction inst = unit.top().first;
    Context context = unit.top().second;
    if (inst.CheckDestWrite()) {
      m_register_unit->SetReady(inst.dest);
    }
    if (context.last_inst) {
      m_finished_contexts->push(context);
    }
    unit.pop();
  }
}

bool ExecutionUnit::full() {
  bool full = true;
  if (!check_units_full(m_i_units)) return false;
  if (!check_units_full(m_f_units)) return false;
  if (!check_units_full(m_sf_units)) return false;
  if (!check_units_full(m_address_units)) return false;
  if (!m_to_ldst_unit->full()) return false;
  if (!m_to_spad_unit->full()) return false;
  if (!check_units_full(m_v_i_units)) return false;
  if (!check_units_full(m_v_f_units)) return false;
  if (!check_units_full(m_v_sf_units)) return false;
  if (!check_units_full(m_v_address_units)) return false;
  if (!m_to_v_ldst_unit->full()) return false;
  if (!m_to_v_spad_unit->full()) return false;
  return true;
}

bool ExecutionUnit::check_units_full(std::vector<ExecutionDelayQueue> &units) {
  bool full = true;
  for (auto &unit : units) {
    full = full && unit.full();
    if (!full) return false;
  }
  return full;
}

Status ExecutionUnit::issue(NdpInstruction &inst, Context context) {
  Status dependancy_status = check_dependency_fail(inst, context);
  if (dependancy_status != SUCCESS) return dependancy_status;
  Status resource_status = check_resource_fail(inst);
  if (resource_status != SUCCESS) return resource_status;

  if (inst.opcode == EXIT) {
    context.last_inst = true;
    context.exit = true;
    context.csr->pc++;
    m_finished_contexts->push(context);
    return ISSUE_SUCCESS;
  }
  AluOpType alu_op_type = inst.GetAluOpType();
  /*Widening instruction requires multiple issue*/
  // TODO: handle the case that vlmul is less than 1
  int mulitply = inst.CheckWideningOp() ? context.csr->vtype_vlmul * 2
                                        : context.csr->vtype_vlmul;
  int inst_latency =
      m_config->get_ndp_op_latencies(alu_op_type) +
      m_config->get_ndp_op_initialiation_interval(alu_op_type) * (mulitply - 1);
  int inst_issue_interval =
      m_config->get_ndp_op_initialiation_interval(alu_op_type) * mulitply;
  bool spad_op = false;
  uint64_t base_addr = 0;
  if (alu_op_type == ADDRESS_OP) {
    base_addr =
        MemoryMap::FormatAddr(m_register_unit->ReadXreg(inst.src[0], context));
    spad_op =
        MemoryMap::CheckScratchpad(base_addr);  // Scratch pad memory access
  }

  if (alu_op_type == ADDRESS_OP) {
    if (inst.CheckIndexedOp() || inst.CheckAmoOp()) {
      if(inst.CheckVectorOp()) {
        VectorData vs2 = context.register_map->ReadVreg(inst.src[2], context);
        if (inst.CheckAmoOp()) {
          vs2 = context.register_map->ReadVreg(inst.src[1], context);
        }
        bool endable_mask = false;
        VectorData vs3;
        if (inst.src[3] != -1) {
          endable_mask = true;
          vs3 = context.register_map->ReadVreg(inst.src[3], context);
        }
        for (int i = 0; i < vs2.GetVlen(); i++) {
          uint64_t idx;
          if (vs2.GetType() == INT32)
            idx = vs2.GetIntData(i);
          else if(vs2.GetType() == INT64)
            idx = vs2.GetLongData(i);
          else if(vs2.GetType() == FLOAT32)
            idx = vs2.GetFloatData(i);
          else if(vs2.GetType() == INT16)
            idx = vs2.GetShortData(i);
          else
            throw std::runtime_error("unsupported index type");
          if (endable_mask) {
            uint64_t vm = vs3.GetVmaskData(i);
            if (vs3.GetType() == VMASK && vm == 0) continue;
          }
          uint64_t addr = MemoryMap::FormatAddr(base_addr + idx);
          inst.addr_set.insert(addr);
        }
      } 
      else {
        uint64_t addr = MemoryMap::FormatAddr(base_addr);
        inst.addr_set.insert(addr);
      }
      inst_issue_interval *= inst.addr_set.size();
      if (inst.addr_set.empty()) throw std::runtime_error("addr set is empty");
    } else { 
      // General Load Store Address Calculation
      int req_size = m_config->get_packet_size();
      if(inst.opcode == VLE16 || inst.opcode == VSE16) req_size = 16;
      else if(inst.opcode == VLE32 || inst.opcode == VSE32) req_size = 32;
      // else if(inst.opcode == VLE64 || inst.opcode == VSE64) req_size = 64;
      int mem_acces_size = context.csr->vtype_vlmul * ROUND_UP(req_size, m_config->get_packet_size());
      for(int i = 0; i < mem_acces_size; i += m_config->get_packet_size())
        inst.addr_set.insert(MemoryMap::FormatAddr(base_addr + inst.src[1] + i * m_config->get_packet_size()));
    }
  }
  
  if (m_config->is_functional_sim() || spad_op || (alu_op_type != ADDRESS_OP))
    inst.Execute(context);

  for (auto &unit : get_unit(inst)) {
    if (!unit.full()) {
      if (inst.CheckDestWrite()) {
        m_register_unit->SetNotReady(inst.dest);
      }
      unit.push({inst, context}, inst_latency, inst_issue_interval);
      break;
    }
  }
  if (!inst.CheckBranchOp()) {
    context.csr->pc++;
    if(context.csr->pc > context.max_pc) {
      context.last_inst = true;
    }
  } else {
    context.csr->block = true;
  }
  if (m_ndp_id == 0 && context.request_info->id == 0) {
    spdlog::debug("NDP{} {}", m_ndp_id, inst.sInst);  // debug
  }
  increase_issue_count(inst);
  if(inst.CheckVectorOp()) {
    int mask_count = (PACKET_SIZE * context.csr->vtype_vlmul) / WORD_SIZE;
    int mask_reg = 3;
    if(inst.opcode == VMV) mask_reg = 1;
    if(inst.opcode == VV) mask_reg = 2;
    if(inst.src[mask_reg] != -1) {
      VectorData mask = context.register_map->ReadVreg(inst.src[mask_reg], context);
      int counter = 0;
      for (int i = 0; i < mask_count / WORD_SIZE;i++) {
        if (mask.GetType() == VMASK && mask.GetVmaskData(i)) counter++;
      }
      m_stats->inc_active_vlane_count(counter, mask_count);
    } else {
      m_stats->inc_active_vlane_count(mask_count, mask_count);
    }
  }
  return ISSUE_SUCCESS;
}

void ExecutionUnit::process_address_inst(ExecutionDelayQueue &address_unit) {
  if (address_unit.empty()) return;
  NdpInstruction inst = address_unit.top().first;
  Context context = address_unit.top().second;
  assert(inst.GetAluOpType() == ADDRESS_OP);
  bool address_unit_pop = false;
  uint64_t first_addr = *inst.addr_set.begin();
  bool spad_op = MemoryMap::CheckScratchpad(first_addr);

  auto fifo_pipeline = get_fifo_pipeline(inst, spad_op);
  if (fifo_pipeline->full()) {
    if (spad_op) m_stats->add_status(SPAD_QUEUE_FULL);
    else m_stats->add_status(LDST_QUEUE_FULL);
    return;
  } else {
    std::pair<NdpInstruction, Context>* inst_contx = new std::pair<NdpInstruction, Context>{inst, context};
    fifo_pipeline->push(inst_contx);
    address_unit_pop = true;
  }

  if (address_unit_pop) {
    address_unit.pop();
  } else {
    assert(0);
  }
}

Status ExecutionUnit::check_dependency_fail(NdpInstruction &inst,
                                            Context &context) {
  WaitInstruction wait_reg(inst.sInst);
  if (inst.CheckScalarOp()) {
    switch (inst.opcode) {
      case ADD:
      case SUB:
      case MUL:
      case DIV:
      case REM:
      case BNE:
      case BEQ:
      case BLT:
      case BLE:
      case BGT:
      case BGE:
      case SD:
      case SW:
      case FSW:
      case FDIV:
      case FMUL:
      case FADD:
      case FSUB:
        if (!m_register_unit->CheckReady(inst.src[0]))
          wait_reg.add_waiting_reg(SRC0);
        if (!m_register_unit->CheckReady(inst.src[1]))
          wait_reg.add_waiting_reg(SRC1);
        if (!wait_reg.wait_reg_mask) return SUCCESS;
        m_stats->add_wait_instruction(wait_reg);
        return SCALAR_DEPENDENCY_WAIT;
      case ADDI:
      case MULI:
      case SLLI:
      case SRLI:
      case BGTZ:
      case BLTZ:
      case BEQZ:
      case BLEZ:
      case BGEZ:
      case BNEZ:
      case LD:
      case LW:
      case FLW:
      case FMV:
        if (!m_register_unit->CheckReady(inst.src[0]))
          m_stats->add_wait_instruction({inst.sInst, 1, SRC0});
        else
          return SUCCESS;
        return SCALAR_DEPENDENCY_WAIT;
      default:
        return SUCCESS;
    }
  } else if (inst.CheckAmoOp()) {
    return (!m_register_unit->CheckReady(inst.src[0]) ||
            !m_register_unit->CheckReady(inst.src[1]) ||
            !m_register_unit->CheckReady(inst.src[2]))
               ? AMO_DEPENDENCY_WAIT
               : SUCCESS;
    // amo op
  } else {
    // vector op
    switch (inst.operand_type) {
      case V:
        if (inst.opcode == VLE16 || inst.opcode == VLE32 ||
            inst.opcode == VLE64) {
          if (!m_register_unit->CheckReady(inst.src[0]))
            m_stats->add_wait_instruction({inst.sInst, 1, SRC0});
          else
            return SUCCESS;
          return VLE_DEPENDENCY_WAIT;
        } else if (inst.opcode == VSE16 || inst.opcode == VSE32 ||
                   inst.opcode == VSE64) {
          if (!m_register_unit->CheckReady(inst.src[0]))
            wait_reg.add_waiting_reg(SRC0);
          if (!m_register_unit->CheckReady(inst.dest))
            wait_reg.add_waiting_reg(DEST);
          if (!wait_reg.wait_reg_mask) return SUCCESS;
          m_stats->add_wait_instruction(wait_reg);
          return VSE_DEPENDENCY_WAIT;
        } else if (inst.opcode == VLUXEI32 || inst.opcode == VLUXEI64) {
          if (!m_register_unit->CheckReady(inst.src[0]))
            wait_reg.add_waiting_reg(SRC0);
          if (!m_register_unit->CheckReady(inst.src[2]))
            wait_reg.add_waiting_reg(SRC2);
          if (!m_register_unit->CheckReady(inst.src[3]))
            wait_reg.add_waiting_reg(SRC3);
          if (!wait_reg.wait_reg_mask) return SUCCESS;
          m_stats->add_wait_instruction(wait_reg);
          return VLE_DEPENDENCY_WAIT;
        } else if (inst.opcode == VSUXEI32 || inst.opcode == VSUXEI64) {
          assert(inst.src[3] !=
                 -1);  // TODO: handle when vector mask does not exist.
          if (!m_register_unit->CheckReady(inst.src[0]))
            wait_reg.add_waiting_reg(SRC0);
          if (!m_register_unit->CheckReady(inst.src[2]))
            wait_reg.add_waiting_reg(SRC2);
          if (!m_register_unit->CheckReady(inst.src[3]))
            wait_reg.add_waiting_reg(MASK);
          if (!m_register_unit->CheckReady(inst.dest))
            wait_reg.add_waiting_reg(DEST);
          if (!wait_reg.wait_reg_mask) return SUCCESS;
          m_stats->add_wait_instruction(wait_reg);
          return VSE_DEPENDENCY_WAIT;
        }
      case VI:
      case V_V:
      case XU_F_V:
      case X_F_V:
      case F_XU_V:
      case F_X_V:
      case F_F_V:
      case F_F_W:
      case V_X:
      case X_S:
      case S_F:
      case V_F:
      case F_S:
        if (!m_register_unit->CheckReady(inst.src[0]))
          m_stats->add_wait_instruction({inst.sInst, 1, SRC0});
        else
          return SUCCESS;
        return VECTOR_DEPENDENCY_WAIT;
      case VV:
      case WV:
      case VX:
      case VF:
      case VS:
        if (!m_register_unit->CheckReady(inst.src[0]))
          wait_reg.add_waiting_reg(SRC0);
        if (!m_register_unit->CheckReady(inst.src[1]))
          wait_reg.add_waiting_reg(SRC1);
        if (!wait_reg.wait_reg_mask) return SUCCESS;
        m_stats->add_wait_instruction(wait_reg);
        return VECTOR_DEPENDENCY_WAIT;
      case S_X:
        if (!m_register_unit->CheckReady(inst.src[0]))
          wait_reg.add_waiting_reg(SRC0);
        if (!m_register_unit->CheckReady(inst.dest))
          wait_reg.add_waiting_reg(DEST);
        if (!wait_reg.wait_reg_mask) return SUCCESS;
        m_stats->add_wait_instruction(wait_reg);
        return VECTOR_DEPENDENCY_WAIT;
      default:
        return SUCCESS;
    }
  }
}

Status ExecutionUnit::check_resource_fail(NdpInstruction &inst) {
  for (auto &delay_queue : get_unit(inst)) {
    if (!delay_queue.full()) return SUCCESS;
  }
  switch (get_unit_type(inst)) {
    case LDST_UNIT:
      if (inst.CheckVectorOp())
        return ADDRESS_VECTOR_UNIT_FULL;
      else
        return ADDRESS_UNIT_FULL;
    case INT_UNIT:
      if (inst.CheckVectorOp())
        return FP_VECTOR_UNIT_FULL;
      else
        return SCALAR_UNIT_FULL;
    case SF_UNIT:
      if (inst.CheckVectorOp())
        return SF_VECTOR_UNIT_FULL;
      else
        return SF_UNIT_FULL;
    case FP_UNIT:
      if (inst.CheckVectorOp())
        return FP_VECTOR_UNIT_FULL;
      else
        return FP_VECTOR_UNIT_FULL;
    default:
      assert(0);
  }
}

bool ExecutionUnit::check_unit_finished() {
  bool finish = true;
  if (!check_units_finished(m_i_units)) return false;
  if (!check_units_finished(m_f_units)) return false;
  if (!check_units_finished(m_sf_units)) return false;
  if (!check_units_finished(m_address_units)) return false;
  if (!check_units_finished(m_v_i_units)) return false;
  if (!check_units_finished(m_v_f_units)) return false;
  if (!check_units_finished(m_v_sf_units)) return false;
  if (!check_units_finished(m_v_address_units)) return false;
  return finish;
}

bool ExecutionUnit::check_units_finished(
    std::vector<ExecutionDelayQueue> &units) {
  bool finish = true;
  for (auto &unit : units) {
    if (!unit.queue_empty()) finish = false;
  }
  return finish;
}

std::vector<ExecutionDelayQueue> &ExecutionUnit::get_unit(NdpInstruction &inst) {
  UnitType unit_type = get_unit_type(inst);
  if (unit_type == LDST_UNIT && !inst.CheckVectorOp()) return m_address_units;
  if (unit_type == LDST_UNIT && inst.CheckVectorOp()) return m_v_address_units;
  if (inst.CheckBranchOp()) return m_i_units;
  if(!m_config->is_all_vector()) {
    if (unit_type == INT_UNIT && !inst.CheckVectorOp()) return m_i_units;
    if (unit_type == FP_UNIT && !inst.CheckVectorOp()) return m_f_units;
    if (unit_type == SF_UNIT && !inst.CheckVectorOp()) return m_sf_units;
  }
  // use v_f unit in both case
  // if(unit_type == INT_UNIT && inst.CheckVectorOp()) return m_v_i_units; //
  if ((unit_type == FP_UNIT || unit_type == INT_UNIT))
    return m_v_f_units;
  if (unit_type == SF_UNIT) return m_v_sf_units;
  assert(0);
}

fifo_pipeline<std::pair<NdpInstruction, Context>> * ExecutionUnit::get_fifo_pipeline(NdpInstruction &inst, bool spad) {
  if (spad) {
    if (inst.CheckVectorOp()) return m_to_v_spad_unit;
    return m_to_spad_unit;
  } else {
    if (inst.CheckVectorOp()) return m_to_v_ldst_unit;
    return m_to_ldst_unit;
  }
}

UnitType ExecutionUnit::get_unit_type(NdpInstruction &inst) {
  if (inst.CheckLoadOp() || inst.CheckStoreOp()) return LDST_UNIT;
  if (inst.GetAluOpType() == SFU_OP) return SF_UNIT;
  if (inst.CheckFloatOp() && !m_config->get_use_unified_alu()) return FP_UNIT;
  return INT_UNIT;
}
void ExecutionUnit::dump_current_state() {
  spdlog::error("NDP {}_{}: cycle {} ", m_ndp_id, m_sub_core_id, m_cycle);
  m_register_unit->dump_current_state();
}

void ExecutionUnit::increase_issue_count(NdpInstruction &inst) {
  bool vector_op = inst.CheckVectorOp();
  UnitType unit_type = get_unit_type(inst);
  if (unit_type == LDST_UNIT) {
    bool spad_op = MemoryMap::CheckScratchpad(*inst.addr_set.begin());
    if (vector_op) {
      m_stats->inc_v_addr_unit_issue_count();
      if (spad_op) {
        m_stats->inc_v_spad_unit_issue_count();
      } else {
        m_stats->inc_v_ldst_unit_issue_count();
      }
    } else {
      m_stats->inc_addr_unit_issue_count();
      if (spad_op) {
        m_stats->inc_spad_unit_issue_count();
      } else {
        m_stats->inc_ldst_unit_issue_count();
      }
    }
  } else if (unit_type == INT_UNIT && vector_op) {
    m_stats->inc_v_unit_issue_count();
  } else if (unit_type == INT_UNIT && !vector_op) {
    m_stats->inc_i_unit_issue_count();
  } else if (unit_type == FP_UNIT && vector_op) {
    m_stats->inc_v_unit_issue_count();
  } else if (unit_type == FP_UNIT && !vector_op) {
    m_stats->inc_f_unit_issue_count();
  } else if (unit_type == SF_UNIT && vector_op) {
    m_stats->inc_v_sf_unit_issue_count();
  } else if (unit_type == SF_UNIT && !vector_op) {
    m_stats->inc_sf_unit_issue_count();
  }
}

}  // namespace NDPSim

#endif