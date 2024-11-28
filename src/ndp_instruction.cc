#include "ndp_instruction.h"

#include <set>

#include "memory_map.h"
#include "register_unit.h"
namespace NDPSim {
typedef robin_hood::unordered_flat_set<Opcode> OpcodeSet;
static const OpcodeSet float_ops = {
    FADD,    FSUB,   FDIV,   FMUL,   FMV,    FLW,       FSW,
    VFADD,   VFSUB,  VFRSUB, VFMUL,  VFWMUL, VFDIV,     VFMACC, 
    VFWMACC, VFNCVT, VFCVT,  VFWCVT, VFMV,   VFREDOMAX, VFREDOSUM,
    FEXP,    VFEXP,  VFWSUB, VFMSUB, VFSQRT};

static const OpcodeSet vector_ops = {
    VADD, VSUB, VSRL, VRSUB, VMUL, VDIV, VMV, VFADD, VFSUB, VFRSUB, VFMUL, VFWMUL,
    VFDIV, VFRDIV, VFMACC, VFWMACC, VFNCVT, VFCVT, VFWCVT, VFMV, VFREDOMAX,
    VFREDOSUM, VWREDOSUM, VWREDOMAX, VMIN, VMAX, VLE8, VSE8, VSE16, VLE32, VSE32, VLE64, VSE64,
    VAMOADDEI8, VAMOADDEI16, VLE16, VAMOADDEI32, VAMOADDEI64, VAMOANDEI16,
    VAMOANDEI32, VAMOOREI16, VAMOOREI32, VAMOMAXEI16, VAMOMAXEI32, VAMOMINEI16,
    VAMOMINEI32, VAND, VOR, VFIRST, TEST, VSLIDE1DOWN, VSLIDEDOWN, VSLIDE1UP,
    VSLIDEUP, VREDSUM, VREDMAX, VREDMIN, VREDAND, VREDOR, VLUXEI32, VSUXEI32,
    VLUXEI64, VSUXEI64, VLSSEG, VMSET, VMSEQ, VMSNE, VMSLT, VMSGT, VMSLE,
    VMSGE,  // Vector Mask
            // Ops
    VMAND, VMOR, VCOMPRESS, VNADD, VSLL, VID, VFEXP, VFSGNJ, VFSGNJN, VFSGNJX,  VFWSUB, VFMSUB, VFSQRT};
static const OpcodeSet narrowing_ops = {VFNCVT};
static const OpcodeSet widening_ops = {VFWMACC, VFWCVT, VFWSUB, VFWMUL, VWREDOSUM, VWREDOMAX};
static const OpcodeSet amo_ops = {
    VAMOADDEI16, VAMOADDEI32, VAMOADDEI64, VAMOANDEI16, VAMOANDEI32, VAMOOREI16,
    VAMOOREI32,  VAMOMAXEI16, VAMOMAXEI32, VAMOMINEI16, VAMOMINEI32,
    AMOADD,      AMOMAX,      FAMOADD,     FAMOMAX,     VAMOADDEI8,  FAMOADDH};
static const OpcodeSet csr_ops = {VSETVLI};
static const OpcodeSet vmask_ops = {VMSEQ, VMSNE, VMSLT, VMSGT, VMSLE, VMSGE};
static const OpcodeSet seg_ops = {VLSSEG};
static const OpcodeSet branch_ops = {J,   BEQ,  BEQZ, BGT,  BLEZ, BGTZ, BLT,
                                     BLE, BLTZ, BGE,  BGEZ, BNE,  BNEZ};

template <>
float arith_op<float>(ArithOp op, const float& lhs, const float& rhs) {
  switch (op) {
    case ARITH_ADD:
      return lhs + rhs;
    case ARITH_SUB:
      return lhs - rhs;
    case ARITH_MUL:
      return lhs * rhs;
    case ARITH_DIV:
      return lhs / rhs;
    case ARITH_MIN:
      return lhs < rhs ? lhs : rhs;
    case ARITH_MAX:
      return lhs > rhs ? lhs : rhs;
    case ARITH_REM:
      return std::fmod(lhs, rhs);
    case ARITH_MOD:
      return std::fmod(lhs, rhs);
    case ARITH_SHL:
      return std::ldexp(lhs, rhs);
    case ARITH_SHR:
      return std::ldexp(lhs, -rhs);
    case ARITH_AND:
      return lhs && rhs;
    case ARITH_OR:
      return lhs || rhs;
    case ARITH_XOR:
      return lhs != rhs;
    default:
      return false;
  }
}

template <>
half arith_op<half>(ArithOp op, const half& lhs, const half& rhs) {
  switch (op) {
    case ARITH_ADD:
      return lhs + rhs;
    case ARITH_SUB:
      return lhs - rhs;
    case ARITH_MUL:
      return lhs * rhs;
    case ARITH_DIV:
      return lhs / rhs;
    case ARITH_MIN:
      return lhs < rhs ? lhs : rhs;
    case ARITH_MAX:
      return lhs > rhs ? lhs : rhs;
    default:
      throw std::runtime_error("Unsupported half operation");
  }
}

MemoryMap* GetMemoryMap(Context& context, uint64_t addr);
VectorData MemoryMapLoad(Context& context, uint64_t addr);
void MemoryMapStore(Context& context, uint64_t addr, VectorData& data);
bool CheckMemoryMap(Context& context, uint64_t addr);

bool NdpInstruction::CheckScalarOp() {
  return vector_ops.find(opcode) == vector_ops.end();
}

bool NdpInstruction::CheckFloatOp() {
  return float_ops.find(opcode) != float_ops.end();
}

bool NdpInstruction::CheckVectorOp() {
  return vector_ops.find(opcode) != vector_ops.end();
}

bool NdpInstruction::CheckNarrowingOp() {
  return narrowing_ops.find(opcode) != narrowing_ops.end();
}

bool NdpInstruction::CheckWideningOp() {
  return widening_ops.find(opcode) != widening_ops.end();
}

bool NdpInstruction::CheckAmoOp() {
  return amo_ops.find(opcode) != amo_ops.end();
}

bool NdpInstruction::CheckCsrOp() {
  return csr_ops.find(opcode) != csr_ops.end();
}

bool NdpInstruction::CheckVmaskOp() {
  return vmask_ops.find(opcode) != vmask_ops.end();
}

bool NdpInstruction::CheckSegOp() {
  return seg_ops.find(opcode) != seg_ops.end();
}

bool NdpInstruction::CheckLoadOp() {
  return opcode == VLE8 || opcode == VLE16 || opcode == VLE32 ||
         opcode == VLE64 || opcode == LW || opcode == LD || opcode == LB ||
         opcode == FLW || opcode == VLUXEI32 || opcode == VLUXEI64 ||
         opcode == VLSSEG || CheckAmoOp();
}

bool NdpInstruction::CheckStoreOp() {
  return opcode == VSE8 || opcode == VSE16 || opcode == VSE32 ||
         opcode == VSE64 || opcode == SW || opcode == SD || opcode == SB ||
         opcode == FSW || opcode == VSUXEI32;
}

bool NdpInstruction::CheckDestWrite() {
  return !(opcode == VSETVLI || opcode == CSRWI || opcode == CSRW ||
           CheckStoreOp() || CheckAmoOp());
}

AluOpType NdpInstruction::GetAluOpType() {
  if (CheckLoadOp() || CheckStoreOp()) return ADDRESS_OP;
  if (opcode == DIV || opcode == FDIV || opcode == VDIV || opcode == VFDIV ||
      opcode == VFEXP || opcode == FEXP || opcode == VFSQRT)
    return SFU_OP;
  if (CheckFloatOp()) {
    if (opcode == VFMACC || opcode == VFMSUB) return FP_MAD_OP;
    if (opcode == VMAX) return FP_MAX_OP;
    if (opcode == FMUL || opcode == VFMUL) return FP_MUL_OP;
    return FP_ADD_OP;
  } else {
    if (opcode == MUL || opcode == MULI || opcode == VMUL) return INT_MUL_OP;
    return INT_ADD_OP;
  }
}

ArithOp NdpInstruction::GetArithOp() {
  if (opcode == VAMOADDEI8 || opcode == VAMOADDEI16 || opcode == VAMOADDEI32 ||
      opcode == VAMOADDEI64)
    return ARITH_ADD;
  if (opcode == VAMOANDEI16 || opcode == VAMOANDEI32) return ARITH_AND;
  if (opcode == VAMOOREI16 || opcode == VAMOOREI32) return ARITH_OR;
  if (opcode == VAMOMAXEI16 || opcode == VAMOMAXEI32) return ARITH_MAX;
  if (opcode == VAMOMINEI16 || opcode == VAMOMINEI32) return ARITH_MIN;
  assert(0);
  return ARITH_ADD;
}

bool NdpInstruction::CheckBranchOp() {
  return branch_ops.find(opcode) != branch_ops.end();
}

bool NdpInstruction::CheckIndexedOp() {
  return opcode == VLUXEI32 || opcode == VSUXEI32;
}

void NdpInstruction::Execute(Context& context) {
  if (opcode == EXIT) {
    return;
  } else if (CheckBranchOp()) {
    ExecuteBranch(context);
    return;
  } else if (CheckCsrOp()) {
    ExecuteVector(context);
    return;
  } else if (CheckVectorOp()) {
    ExecuteVector(context);
    return;
  } else if (CheckSegOp()) {
    ExecuteVector(context);
    return;
  } else if (CheckScalarOp()) {
    ExecuteScalar(context);
    return;
  } else {
    spdlog::error("Unsupported instruction {} : {}", opcode, sInst);
    throw std::runtime_error("Unsupported instruction");
  }
}

void amo_loop(Context context, VectorData vmask, VectorData offsets,
              VectorData vs, uint64_t base, ArithOp op) {
  uint64_t offset;
  uint64_t addr;
  int idx;
  for (int i = 0; i < offsets.GetVlen(); i++) {
    assert(vmask.GetType() == VMASK);
    if (!vmask.GetVmaskData(i)) continue;

    switch (offsets.GetType()) {
      case INT64:
        offset = offsets.GetLongData(i);
        addr = base + offset;
        idx =
            ((addr % PACKET_SIZE) / DOUBLE_SIZE) % (PACKET_SIZE / DOUBLE_SIZE);
        break;
      case INT32:
        offset = offsets.GetIntData(i);
        addr = base + offset;
        idx = ((addr % PACKET_SIZE) / WORD_SIZE) % (PACKET_SIZE / WORD_SIZE);
        break;
      case INT16:
        offset = offsets.GetShortData(i);
        addr = base + offset;
        idx = ((addr % PACKET_SIZE) / HALF_SIZE) % (PACKET_SIZE / HALF_SIZE);
        break;
      case UINT8:
        offset = offsets.GetU8Data(i);
        addr = base + offset;
        idx = ((addr % PACKET_SIZE) ) % (PACKET_SIZE );
        break;
      default:
        throw std::runtime_error("Unsupported offset type");
    }
    MemoryMap* map = GetMemoryMap(context, addr);
    VectorData vd(vs.GetPrecision()*8, 1);  // Force single register
    try {
      vd = map->Load((addr & ~(PACKET_SIZE - 1)));
    } catch (const std::runtime_error& error) {
      vd = VectorData(vs.GetPrecision()*8, 1);
      vd.SetType(vs.GetType());
    }
    if (vd.GetType() == FLOAT32)
      vd.SetData(arith_op<float>(op, vd.GetFloatData(idx), vs.GetFloatData(i)),
                 idx);
    else if (vd.GetType() == UINT8)
      vd.SetData(arith_op<uint8_t>(op, vd.GetU8Data(idx), vs.GetU8Data(i)),
                 idx);
    else if (vs.GetType() == INT16)
      vd.SetData(
          arith_op<int16_t>(op, vd.GetShortData(idx), vs.GetShortData(i)), idx);
    else if (vs.GetType() == FLOAT16)
      vd.SetData(
          arith_op<half>(op, vd.GetHalfData(idx), vs.GetHalfData(i)), idx);
   else if (vs.GetType() == INT32)
      vd.SetData(arith_op<int32_t>(op, vd.GetIntData(idx), vs.GetIntData(i)),
                 idx);
    else if (vs.GetType() == INT64)
      vd.SetData(arith_op<int64_t>(op, vd.GetLongData(idx), vs.GetLongData(i)),
                 idx);
    else
      throw std::runtime_error("Unsupported Data Type");
    MemoryMapStore(context, (addr & ~(PACKET_SIZE - 1)), vd);
  }
}

void NdpInstruction::ExecuteScalar(Context& context) {
  if (opcode == ADD) {
    int64_t rs1 = context.register_map->ReadXreg(src[0], context);
    int64_t rs2 = context.register_map->ReadXreg(src[1], context);
    context.register_map->WriteXreg(dest, rs1 + rs2, context);
  } else if (opcode == ADDI) {
    int64_t rs1 = context.register_map->ReadXreg(src[0], context);
    int64_t rs2 = src[1];
    context.register_map->WriteXreg(dest, rs1 + rs2, context);
  } else if (opcode == SUB) {
    int64_t rs1 = context.register_map->ReadXreg(src[0], context);
    int64_t rs2 = context.register_map->ReadXreg(src[1], context);
    context.register_map->WriteXreg(dest, rs1 - rs2, context);
  } else if (opcode == MUL) {
    int64_t rs1 = context.register_map->ReadXreg(src[0], context);
    int64_t rs2 = context.register_map->ReadXreg(src[1], context);
    context.register_map->WriteXreg(dest, rs1 * rs2, context);
  } else if (opcode == MULI) {
    int64_t rs1 = context.register_map->ReadXreg(src[0], context);
    int64_t rs2 = src[1];
    context.register_map->WriteXreg(dest, rs1 * rs2, context);
  } else if (opcode == SLL) {
    int64_t rs1 = context.register_map->ReadXreg(src[0], context);
    int64_t rs2 = context.register_map->ReadXreg(src[1], context);
    context.register_map->WriteXreg(dest, rs1 << rs2, context);
  } else if (opcode == SLLI) {
    int64_t rs1 = context.register_map->ReadXreg(src[0], context);
    int64_t rs2 = src[1];
    context.register_map->WriteXreg(dest, rs1 << rs2, context);
  } else if (opcode == SRL) {
    int64_t rs1 = context.register_map->ReadXreg(src[0], context);
    int64_t rs2 = context.register_map->ReadXreg(src[1], context);
    context.register_map->WriteXreg(dest, rs1 >> rs2, context);
  } else if (opcode == SRLI) {
    int64_t rs1 = context.register_map->ReadXreg(src[0], context);
    int64_t rs2 = src[1];
    context.register_map->WriteXreg(dest, rs1 >> rs2, context);
  } else if (opcode == DIV) {
    int64_t rs1 = context.register_map->ReadXreg(src[0], context);
    int64_t rs2 = context.register_map->ReadXreg(src[1], context);
    context.register_map->WriteXreg(dest, rs1 / rs2, context);
  } else if (opcode == REM) {
    int64_t rs1 = context.register_map->ReadXreg(src[0], context);
    int64_t rs2 = context.register_map->ReadXreg(src[1], context);
    context.register_map->WriteXreg(dest, rs1 % rs2, context);
  } else if (opcode == AND) {
    int64_t rs1 = context.register_map->ReadXreg(src[0], context);
    int64_t rs2 = context.register_map->ReadXreg(src[1], context);
    context.register_map->WriteXreg(dest, rs1 & rs2, context);
  } else if (opcode == ANDI) {
    int64_t rs1 = context.register_map->ReadXreg(src[0], context);
    int64_t rs2 = src[1];
    context.register_map->WriteXreg(dest, rs1 & rs2, context);
  } else if (opcode == OR) {
    int64_t rs1 = context.register_map->ReadXreg(src[0], context);
    int64_t rs2 = context.register_map->ReadXreg(src[1], context);
    context.register_map->WriteXreg(dest, rs1 | rs2, context);
  } else if (opcode == ORI) {
    int64_t rs1 = context.register_map->ReadXreg(src[0], context);
    int64_t rs2 = src[1];
    context.register_map->WriteXreg(dest, rs1 | rs2, context);
  }
  else if (opcode == CSRWI) {
    if (dest == REG_VSTART) {
      context.csr->vstart = src[0];
    }
  } else if (opcode == CSRW) {
    if (dest == REG_VSTART) {
      context.csr->vstart = context.register_map->ReadXreg(src[0], context);
    } else
      throw std::runtime_error("Not supported instruction yet.");
  } else if (opcode == LI) {
    int64_t xreg_value = src[0];
    context.register_map->WriteXreg(dest, xreg_value, context);
  } else if (opcode == LW) {
    int64_t addr = context.register_map->ReadXreg(src[0], context);
    int64_t offset = src[1];
    addr += offset;
    VectorData vs = MemoryMapLoad(context, addr);
    int index = (addr % PACKET_SIZE) / WORD_SIZE;
    assert(vs.GetType() == INT32);
    context.register_map->WriteXreg(dest, vs.GetIntData(index), context);
  } else if (opcode == LD) {
    int64_t addr = context.register_map->ReadXreg(src[0], context);
    int64_t offset = src[1];
    addr += offset;
    VectorData vs = MemoryMapLoad(context, addr);
    int index = (addr - MemoryMap::FormatAddr(addr)) / DOUBLE_SIZE;
    assert(vs.GetType() == INT64);
    context.register_map->WriteXreg(dest, vs.GetLongData(index), context);
  } else if (opcode == LBU) {
    char rs1 = context.register_map->ReadXreg(src[0], context);
    context.register_map->WriteXreg(dest, rs1, context);
  } else if (opcode == SB) {
    uint8_t x_val = context.register_map->ReadXreg(dest, context);
    int64_t addr = context.register_map->ReadXreg(src[0], context);
    addr += src[1];
    int offset = (addr % PACKET_SIZE);
    addr = (addr / PACKET_SIZE) * PACKET_SIZE;
    VectorData vs(context);
    if (CheckMemoryMap(context, addr))
      vs = MemoryMapLoad(context, addr);
    vs.SetData(x_val, offset);
    MemoryMapStore(context, addr, vs);
  } else if (opcode == SW) {
    int64_t x_val = context.register_map->ReadXreg(dest, context);
    int64_t addr = context.register_map->ReadXreg(src[0], context);
    addr += src[1];
    int offset = (addr % PACKET_SIZE) / WORD_SIZE;
    VectorData vs(context);
    if (CheckMemoryMap(context, addr)) vs = MemoryMapLoad(context, addr);
    vs.SetData((int32_t)x_val, offset);
    MemoryMapStore(context, addr, vs);
  } else if (opcode == SD) {
    int64_t x_val = context.register_map->ReadXreg(dest, context);
    int64_t addr = context.register_map->ReadXreg(src[0], context);
    addr += src[1];
    int offset = (addr % PACKET_SIZE) / DOUBLE_SIZE;
    VectorData vs(context);
    if (CheckMemoryMap(context, addr)) vs = MemoryMapLoad(context, addr);
    vs.SetData(x_val, offset);
    MemoryMapStore(context, addr, vs);
  } else if (opcode == FADD) {
    float rs1 = context.register_map->ReadFreg(src[0], context);
    float rs2 = context.register_map->ReadFreg(src[1], context);
    context.register_map->WriteFreg(dest, rs1 + rs2, context);
  } else if (opcode == FSUB) {
    float rs1 = context.register_map->ReadFreg(src[0], context);
    float rs2 = context.register_map->ReadFreg(src[1], context);
    context.register_map->WriteFreg(dest, rs1 - rs2, context);
  } else if (opcode == FDIV) {
    float rs1 = context.register_map->ReadFreg(src[0], context);
    float rs2 = context.register_map->ReadFreg(src[1], context);
    context.register_map->WriteFreg(dest, rs1 / rs2, context);
  } else if (opcode == FMUL) {
    float rs1 = context.register_map->ReadFreg(src[0], context);
    float rs2 = context.register_map->ReadFreg(src[1], context);
    context.register_map->WriteFreg(dest, rs1 * rs2, context);
  } else if (opcode == FMV) {
    if (operand_type == X_W) {
      float rs1 = context.register_map->ReadFreg(src[0], context);
      context.register_map->WriteXreg(dest, int32_t(rs1), context);
    } else if (operand_type == W_X) {
      int64_t rs1 = context.register_map->ReadXreg(src[0], context);
      context.register_map->WriteFreg(dest, float(rs1), context);
    }
  } else if (opcode == FLW) {
    int64_t addr = context.register_map->ReadXreg(src[0], context);
    int64_t offset = src[1];
    addr += offset;
    VectorData vs = MemoryMapLoad(context, addr);
    int index = (addr % 32) / 4;
    assert(vs.GetType() == FLOAT32);
    context.register_map->WriteFreg(dest, vs.GetFloatData(index), context);
  } else if (opcode == FSW) {
    float f_val = context.register_map->ReadFreg(dest, context);
    int64_t addr = context.register_map->ReadXreg(src[0], context);
    addr += src[1];
    int offset = (addr % PACKET_SIZE) / WORD_SIZE;
    VectorData vs(context);
    if (CheckMemoryMap(context, addr)) {
      vs = MemoryMapLoad(context, addr);
    }
    vs.SetData((float)f_val, offset);
    MemoryMapStore(context, addr, vs);
  } else if (opcode == AMOADD) {
    int64_t addr = context.register_map->ReadXreg(src[2], context);
    VectorData vd(context);
    if (CheckMemoryMap(context, addr)) {
      vd = MemoryMapLoad(context, addr);
    }
    int value = 0, index = 0;
    if (operand_type == W) {
      // Atomic Add in word size
      index = (addr % PACKET_SIZE) / WORD_SIZE;
      value = vd.GetIntData(index) + context.register_map->ReadXreg(src[1], context);
    } else if (operand_type == D) {
      // Atmoic Add in double size
      index = (addr % PACKET_SIZE) / DOUBLE_SIZE;
      value = vd.GetLongData(index) + context.register_map->ReadXreg(src[1], context);
    } else {
      throw std::runtime_error("Unsupported scalar instruction");
    }
    vd.SetData(value, index);
    MemoryMapStore(context, addr, vd);
    context.register_map->WriteXreg(dest, value, context);
  } else if (opcode == FAMOADD) {
    int64_t addr = context.register_map->ReadXreg(src[2], context);
    VectorData vd(context);
    
    int idx = 0;

    if (CheckMemoryMap(context, addr)) {
      vd = MemoryMapLoad(context, addr);
    }

    float value = 0;
    int index = 0;
    if (operand_type == W) {
      // Atomic Add in word size
      index = (addr % PACKET_SIZE) / WORD_SIZE;
      value = vd.GetFloatData(index) + context.register_map->ReadFreg(src[1], context);

      vd.SetData((float)value, index);
    } else {
      // Atomic Add in word size
      index = (addr % PACKET_SIZE) / HALF_SIZE;
      value = vd.GetHalfData(index) + context.register_map->ReadFreg(src[1], context);
      // throw std::runtime_error("Unsupported scalar instruction");
      vd.SetData((half)value, index);
    }

    MemoryMapStore(context, addr, vd);
    context.register_map->WriteFreg(dest, value, context);
  } else if (opcode == FAMOADDH) {
    int64_t addr = context.register_map->ReadXreg(src[2], context);
    VectorData vd(context);    
    int idx = 0;
    if (CheckMemoryMap(context, addr)) {
      vd = MemoryMapLoad(context, addr);
    }
    float value = 0;
    int index = 0;

    index = (addr % PACKET_SIZE) / HALF_SIZE;
    value = vd.GetHalfData(index) + context.register_map->ReadFreg(src[1], context);
    vd.SetData((half)value, index);

    MemoryMapStore(context, addr, vd);
    context.register_map->WriteFreg(dest, value, context);
  } else if (opcode == AMOMAX) {
  } else if (opcode == FEXP) {
    float rs1 = context.register_map->ReadFreg(src[0], context);
    context.register_map->WriteFreg(dest, std::exp(rs1), context);
  } else {
    throw std::runtime_error("Unsupported scalar instruction");
  }
}

void NdpInstruction::ExecuteVector(Context& context) {
  bool word_type_op = context.csr->vtype_vsew == 32;
  bool double_reg = context.csr->vtype_vlmul == 2;

  if (opcode == VSETVLI) {
    // Todo: implement other configuration of vtype
    if (src[1] == IMM_E8) {
      context.csr->vtype_vsew = 8;
    } else if (src[1] == IMM_E16) {
      context.csr->vtype_vsew = 16;
    } else if (src[1] == IMM_E32) {
      context.csr->vtype_vsew = 32;
    } else if (src[1] == IMM_E64) {
      context.csr->vtype_vsew = 64;
    } else
      throw std::runtime_error("Not implemented vtype_vsew");

    if (src[2] == IMM_M1) {
      context.csr->vtype_vlmul = 1;
    } else if (src[2] == IMM_M2) {
      context.csr->vtype_vlmul = 2;
    } else if (src[2] == IMM_M4) {
      context.csr->vtype_vlmul = 4;
    } else if (src[2] == IMM_M8) {
      context.csr->vtype_vlmul = 8;
    } else if (src[2] == IMM_MF2) {
      context.csr->vtype_vlmul = 0.5;
    } else if (src[2] == IMM_MF4) {
      context.csr->vtype_vlmul = 0.25;
    } else if (src[2] == IMM_MF8) {
      context.csr->vtype_vlmul = 0.125;
    } else
      throw std::runtime_error("Not implemented vtype_vlmul");
  } else if (opcode == VADD || opcode == VFADD) {
    VectorData vs1 = context.register_map->ReadVreg(src[0], context);
    VectorData vd(context);
    if (operand_type == VV) {
      VectorData vs2 = context.register_map->ReadVreg(src[1], context);
      vd = vs1 + vs2;
    } else if (operand_type == VX) {
      int64_t x2 = context.register_map->ReadXreg(src[1], context);
      vd = vs1 + x2;
    } else if (operand_type == VI) {
      int32_t x2 = src[1];
      vd = vs1 + x2;
    } else if (operand_type == VF) {
      float f2 = context.register_map->ReadFreg(src[1], context);
      vd = vs1 + f2;
    } else {
      throw std::runtime_error("Unsupported vector instruction");
    }
    context.register_map->WriteVreg(dest, vd, context);
  } else if (opcode == VSUB || opcode == VFSUB) {
    VectorData vs1 = context.register_map->ReadVreg(src[0], context);
    if (CheckFloatOp() && word_type_op)
      vs1.SetType(FLOAT32);
    else if (CheckFloatOp() && !word_type_op)
      vs1.SetType(FLOAT16);
    VectorData vd(context);
    if (CheckFloatOp() && word_type_op)
      vd.SetType(FLOAT32);
    else if (CheckFloatOp() && !word_type_op)
      vd.SetType(FLOAT16);
    if (operand_type == VV) {
      VectorData vs2 = context.register_map->ReadVreg(src[1], context);
      if (CheckFloatOp() && word_type_op) vs2.SetType(FLOAT32);
      if (CheckFloatOp() && !word_type_op) vs2.SetType(FLOAT16);
      vd = vs1 - vs2;
    } else if (operand_type == VI) {
      int32_t x2 = src[1];
      vd = vs1 - x2;
    } else if (operand_type == VX) {
      int32_t x2 = context.register_map->ReadXreg(src[1], context);
      vd = vs1 - x2;
    } else if (operand_type == VF) {
      float f2 = context.register_map->ReadFreg(src[1], context);
      if (context.csr->vtype_vsew == 16) {
        vs1.SetType(FLOAT16);
      }
      vd = vs1 - f2;
    } else {
      throw std::runtime_error("Unsupported vector instruction");
    }
    context.register_map->WriteVreg(dest, vd, context);
  } else if (opcode == VFWSUB) {
    VectorData vs1 = context.register_map->ReadVreg(src[0], context);    
    assert(vs1.GetType() == FLOAT16);
    VectorData temp(context);
    temp.SetVectorDoubleReg();
    temp.SetType(FLOAT32);
    for (int i = context.csr->vstart; i < vs1.GetVlen(); i++) {
      temp.SetData((float)vs1.GetHalfData(i), i);
    }
    VectorData vd(context);
    vd.SetVectorDoubleReg();
    vd.SetType(FLOAT32);
    if (operand_type == VF) {
      float f2 = context.register_map->ReadFreg(src[1], context);
      vd = temp - f2;
    } else {
      throw std::runtime_error("Unsupported vector instruction yet");
    }
    context.register_map->WriteVreg(dest, vd, context);
  } else if (opcode == VRSUB || opcode == VFRSUB) {
    VectorData vs2 = context.register_map->ReadVreg(src[1], context);
    if (!CheckFloatOp()) vs2.SetType(INT32);
    if (CheckFloatOp() && word_type_op) vs2.SetType(FLOAT32);
    if (CheckFloatOp() && !word_type_op) vs2.SetType(FLOAT16);
    VectorData vd(double_reg);
    if (operand_type == VI) {
      int32_t x1 = src[0];
      vd = x1 - vs2;
    } else if (operand_type == VF) {
      float f1 = context.register_map->ReadFreg(src[0], context);
      vd = f1 - vs2;
    } else {
      throw std::runtime_error("Unsupported vector instruction");
    }
    context.register_map->WriteVreg(dest, vd, context);
  } else if (opcode == VMUL || opcode == VFMUL) {
    VectorData vs1 = context.register_map->ReadVreg(src[0], context);
    VectorData vd(context);
    if (operand_type == VV) {
      VectorData vs2 = context.register_map->ReadVreg(src[1], context);
      vd = vs1 * vs2;
    } else if (operand_type == VX) {
      int32_t x2 = context.register_map->ReadXreg(src[1], context);
      vd = vs1 * x2;
    } else if (operand_type == VI) {
      int32_t x2 = src[1];
      vd = vs1 * x2;
    } else if (operand_type == VF) {
      float f2 = context.register_map->ReadFreg(src[1], context);
      vd = vs1 * f2;
    } else {
      throw std::runtime_error("Unsupported vector instruction");
    }
    context.register_map->WriteVreg(dest, vd, context);
  } else if (opcode == VFWMUL) {
    VectorData vs1 = context.register_map->ReadVreg(src[0], context);
    VectorData vd(context);
    vd.SetVectorDoubleReg();
    if (operand_type == VV) {
      VectorData vs2 = context.register_map->ReadVreg(src[1], context);
      if (vs1.GetType() != vs2.GetType())
        throw std::runtime_error("VFWMUL must use same typed operands");
      if (vs1.GetType() == FLOAT16) {
        VectorData temp(context);
        temp = vs1 * vs2;
        for(int i = 0; i < temp.GetVlen(); i++) {
          vd.SetData((float)temp.GetHalfData(i), i);
        }
      } else {
        throw std::runtime_error("Unsupported source operand type");
      }
    } else if (operand_type == VF) {
      float f2 = context.register_map->ReadFreg(src[1], context);
      if (vs1.GetType() == FLOAT16) {
        VectorData temp(context);
        temp = vs1 * f2;
        for(int i = 0; i < temp.GetVlen(); i++) {
          vd.SetData((float)temp.GetHalfData(i), i);
        }
      } else {
        throw std::runtime_error("Unsupported source operand type");
      }
    } else {
      throw std::runtime_error("Unsupported vector instruction");
    }
    context.register_map->WriteVreg(dest, vd, context);
  } else if (opcode == VDIV || opcode == VFDIV) {
    VectorData vs1 = context.register_map->ReadVreg(src[0], context);
    if (!CheckFloatOp()) vs1.SetType(INT32);
    if (CheckFloatOp() && word_type_op) vs1.SetType(FLOAT32);
    if (CheckFloatOp() && !word_type_op) vs1.SetType(FLOAT16);
    VectorData vd(context);
    if (operand_type == VV) {
      VectorData vs2 = context.register_map->ReadVreg(src[1], context);
      vd = vs1 / vs2;
    } else if (operand_type == VX) {
      int32_t x2 = context.register_map->ReadXreg(src[1], context);
      vd = vs1 / x2;
    } else if (operand_type == VI) {
      int32_t x2 = src[1];
      vd = vs1 / x2;
    } else if (operand_type == VF) {
      float f2 = context.register_map->ReadFreg(src[1], context);
      vd.SetType(vs1.GetType());
      vd = vs1 / f2;
    } else {
      throw std::runtime_error("Unsupported vector instruction");
    }
    context.register_map->WriteVreg(dest, vd, context);
  } else if (opcode == VSLL) {
    VectorData vd(context);
    VectorData vs2 = context.register_map->ReadVreg(src[0], context);
    if (operand_type == VV) {
      VectorData vs1 = context.register_map->ReadVreg(src[1], context);
      for (int i = context.csr->vstart; i < vd.GetVlen(); i++) {
        if (vs2.GetType() == INT64)
          vd.SetData(static_cast<int64_t>(vs2.GetLongData(i) << vs1.GetLongData(i)), i);
        else if (vs2.GetType() == INT32)
          vd.SetData(static_cast<int>(vs2.GetIntData(i) << vs1.GetIntData(i)), i);
        else if (vs2.GetType() == INT16)
          vd.SetData(static_cast<short>(vs2.GetShortData(i) << vs1.GetShortData(i)), i);
      }
      context.register_map->WriteVreg(dest, vd, context);
    } else if (operand_type == VX) {
      int64_t rs1 = context.register_map->ReadXreg(src[1], context);
      for (int i = context.csr->vstart; i < vd.GetVlen(); i++) {
        if (vs2.GetType() == INT64)
          vd.SetData(static_cast<int64_t>(vs2.GetLongData(i) << rs1), i);
        else if (vs2.GetType() == INT32)
          vd.SetData(static_cast<int>(vs2.GetIntData(i) << rs1), i);
        else if (vs2.GetType() == INT16)
          vd.SetData(static_cast<short>(vs2.GetShortData(i) << rs1), i);
      }
      context.register_map->WriteVreg(dest, vd, context);
    } else if (operand_type == VI) {
      uint32_t uimm = src[1];
      for (int i = context.csr->vstart; i < vd.GetVlen(); i++) {
        if (vs2.GetType() == INT64)
          vd.SetData((long)(vs2.GetLongData(i) << uimm), i);
        else if (vs2.GetType() == INT32)
          vd.SetData((int)(vs2.GetIntData(i) << uimm), i);
        else if (vs2.GetType() == INT16)
          vd.SetData((short)(vs2.GetShortData(i) << uimm), i);
      }
      context.register_map->WriteVreg(dest, vd, context);
    } else {
      throw std::runtime_error("Unsupported vector instruction");
    }
  } else if (opcode == VSRL) {
    if(operand_type == VI) {
      uint32_t uimm = src[1];
      VectorData vd(context);
      VectorData vs2 = context.register_map->ReadVreg(src[0], context);
      for (int i = context.csr->vstart; i < vd.GetVlen(); i++) {
        if (vs2.GetType() == INT64)
          vd.SetData(static_cast<int64_t>(vs2.GetLongData(i) >> uimm), i);
        else if (vs2.GetType() == INT32)
          vd.SetData(static_cast<int>(vs2.GetIntData(i) >> uimm), i);
        else if (vs2.GetType() == INT16)
          vd.SetData(static_cast<int16_t>(vs2.GetShortData(i) >> uimm), i);
        else if (vs2.GetType() == UINT8)
          vd.SetData(static_cast<int8_t>(vs2.GetU8Data(i) >> uimm), i);
      }
      context.register_map->WriteVreg(dest, vd, context);
    }
  } else if (opcode == VSRA) {
  } else if (opcode == VNADD) {
    VectorData vd(context);
    VectorData vs2 = context.register_map->ReadVreg(src[0], context);
    if (operand_type == WX) {
      int32_t rs1 = context.register_map->ReadXreg(src[1], context);
      for (int i = context.csr->vstart; i < vd.GetVlen(); i++) {
        if (vs2.GetType() == INT64) {
          vd.SetData((int32_t)(rs1 + vs2.GetLongData(i)), i);
        } else if (vs2.GetType() == INT32)
          vd.SetData((int16_t)(rs1 + vs2.GetIntData(i)), i);
        else
          spdlog::warn("Not implemented yet");
      }
    } else if (operand_type == WV) {
      spdlog::warn("Not implemented yet");
    } else
      throw std::runtime_error("Unsupported vector instruction");
    context.register_map->WriteVreg(dest, vd, context);
  } else if (opcode == VFRDIV) {
    VectorData vd(context);
    VectorData vs1 = context.register_map->ReadVreg(src[0], context);
    float f2 = context.register_map->ReadFreg(src[1], context);
    if (operand_type == VF) {
      vd.SetType(vs1.GetType());
      vd = f2 / vs1;
    } else {
      throw std::runtime_error("Unsupported vector instruction");
    }
    context.register_map->WriteVreg(dest, vd, context);
  } else if (opcode == VMV) {
    VectorData vm;
    if (src[1] != -1) vm = context.register_map->ReadVreg(src[1], context);
    if (operand_type == V_V) {
      // assert(src[1] != -1); // TODO: mask not supported yet
      VectorData vd(context);
      if (context.register_map->CheckExistRenameVreg(dest))
        vd = context.register_map->ReadVreg(dest, context);
      VectorData vs1 = context.register_map->ReadVreg(src[0], context);
      vd = vs1;
      context.register_map->WriteVreg(dest, vd, context);
    } else if (operand_type == V_I) {
      VectorData vd(context); 
      if(context.register_map->CheckExistRenameVreg(dest) && src[1] != -1)
        vd = context.register_map->ReadVreg(dest, context);
      // if (context.register_map->CheckExistRenameVreg(dest)) // TODO : check
      // if the vd has already been used.
      //   vd = context.register_map->ReadVreg(dest, context);
      int64_t x1 = src[0];
      for (int i = context.csr->vstart; i < vd.GetVlen(); i++) {
        if (!vm.GetVmaskData(i) && src[1] != -1) continue;
        switch (context.csr->vtype_vsew /
                8) {  // TODO: replace with GetPrecision()
          case 1:
            vd.SetData((bool)x1, i);
            vd.SetData((char)x1, i);
            vd.SetData((uint8_t)x1, i);
            vd.SetType(BOOL);  // TODO: don't need SetType() // WTF????
            vd.SetType(UINT8);  // TODO: don't need SetType() // WTF????
            break;
          case 2:
            vd.SetData((int16_t)x1, i);
            vd.SetType(INT16);
            break;
          case 4:
            vd.SetData((int32_t)x1, i);
            vd.SetType(INT32);
            break;
          case 8:
            vd.SetData((int64_t)x1, i);
            vd.SetType(INT64);
            break;
          default:
            throw std::runtime_error("vmv.v.i Not supported precision");
            break;
        }
      }
      context.register_map->WriteVreg(dest, vd, context);
    } else if (operand_type == V_X) {
      VectorData vd(context);
      if (context.register_map->CheckExistRenameVreg(dest))
        vd = context.register_map->ReadVreg(dest, context);
      int64_t x1 = context.register_map->ReadXreg(src[0], context);
      for (int i = context.csr->vstart; i < vd.GetVlen(); i++) {
        if (!vm.GetVmaskData(i) && src[1] != -1) continue;
        switch (context.csr->vtype_vsew / 8) {
          case 1:
            vd.SetData((char)x1, i);
            vd.SetType(CHAR8);
            break;
          case 2:
            vd.SetData((int16_t)x1, i);
            vd.SetType(INT16);
            break;
          case 4:
            vd.SetData((int32_t)x1, i);
            vd.SetType(INT32);
            break;
          case 8:
            vd.SetData((int64_t)x1, i);
            vd.SetType(INT64);
            break;
          default:
            throw std::runtime_error("vmv.v.x Not supported precision");
            break;
        }
      }
      context.register_map->WriteVreg(dest, vd, context);
    } else if (operand_type == V_F) {
      VectorData vd(context);
      if (context.register_map->CheckExistRenameVreg(dest))
        vd = context.register_map->ReadVreg(dest, context);
      if (!CheckFloatOp())
        throw std::runtime_error("VFMV must use float dest register");
      float f2 = context.register_map->ReadFreg(src[0], context);
      for (int i = context.csr->vstart; i < vd.GetVlen(); i++) {
        if (!vm.GetVmaskData(i) && src[1] != -1) continue;
        switch (context.csr->vtype_vsew / 8) {
          case 2:
            vd.SetData(half(f2), i);
            vd.SetType(FLOAT16);
            break;
          case 4:
            vd.SetData((float)f2, i);
            vd.SetType(FLOAT32);
            break;
          default:
            throw std::runtime_error("vmv.v.f Not supported precision");
            break;
        }
      }
      context.register_map->WriteVreg(dest, vd, context);
    } else if (operand_type == X_S) {
      VectorData f2 = context.register_map->ReadVreg(src[0], context);

      switch (f2.GetType()) {
        case VMASK:
          if (context.csr->vtype_vsew == 8) {
            context.register_map->WriteXreg(
                dest, f2.GetVmaskAsByte(8 * (context.csr->vstart)), context);
          } else
            throw std::runtime_error("Not supported vmask type yet");

          break;
        case FLOAT16:
          context.register_map->WriteXreg(
              dest, f2.GetHalfData(context.csr->vstart), context);
          break;
        case FLOAT32:
          context.register_map->WriteXreg(
              dest, f2.GetFloatData(context.csr->vstart), context);
          break;
        case INT16:
          context.register_map->WriteXreg(
              dest, f2.GetShortData(context.csr->vstart), context);
          break;
        case INT32:
          context.register_map->WriteXreg(
              dest, f2.GetIntData(context.csr->vstart), context);
          break;
        case INT64:
          context.register_map->WriteXreg(
              dest, f2.GetLongData(context.csr->vstart), context);
          break;
        case CHAR8:
          context.register_map->WriteXreg(
              dest, f2.GetCharData(context.csr->vstart), context);
          break;
        default:
          break;
      }
    } else if (operand_type == S_X) {
      VectorData vd = context.register_map->ReadVreg(dest, context);
      switch (vd.GetType()) {
        case FLOAT16:
          vd.SetData(half(context.register_map->ReadXreg(src[0], context)),
                     context.csr->vstart);
          break;
        case FLOAT32:
          vd.SetData((float)(context.register_map->ReadXreg(src[0], context)),
                     context.csr->vstart);
          break;
        case INT16:
          vd.SetData((int16_t)(context.register_map->ReadXreg(src[0], context)),
                     context.csr->vstart);
          break;
        case INT32:
          vd.SetData((int32_t)(context.register_map->ReadXreg(src[0], context)),
                     context.csr->vstart);
          break;
        case INT64:
          vd.SetData((int64_t)(context.register_map->ReadXreg(src[0], context)),
                     context.csr->vstart);
          break;
        case CHAR8:
          vd.SetData((char)(context.register_map->ReadXreg(src[0], context)),
                     context.csr->vstart);
          break;
        default:
          break;
      }
      context.register_map->WriteVreg(dest, vd, context);
    } else
      throw std::runtime_error("Unsupported vector instruction");
  } else if (opcode == Opcode::VID) {
    VectorData vd(context);
    for (int i = context.csr->vstart; i < vd.GetVlen(); i++) {
      switch (context.csr->vtype_vsew) {
        case 64:
          vd.SetData((int64_t)i, i);
          break;
        case 32:
          vd.SetData((int32_t)i, i);
          break;
        case 16:
          vd.SetData((int16_t)i, i);
          break;
        case 8:
          vd.SetData((uint8_t)i, i);
          break;
        default:
          break;
      }
    }
    context.register_map->WriteVreg(dest, vd, context);
  } else if (opcode == Opcode::VMSET) {
    VectorData vd(context);
    for (int i = context.csr->vstart; i < vd.GetVlen(); i++) {
      vd.SetVmask(true, i);
    }
    context.register_map->WriteVreg(dest, vd, context);
  } else if (CheckVmaskOp()) {
    LogicalOp op;
    if (opcode == VMSEQ)
      op = LogicalOp::EQ;
    else if (opcode == VMSNE)
      op = LogicalOp::NE;
    else if (opcode == VMSLT)
      op = LogicalOp::LT;
    else if (opcode == VMSLE)
      op = LogicalOp::LE;
    else if (opcode == VMSGT)
      op = LogicalOp::GT;
    else if (opcode == VMSGE)
      op = LogicalOp::GE;
    else
      throw std::runtime_error("Unsupported vector instruction");
    VectorData vs1 = context.register_map->ReadVreg(src[0], context);
    VectorData vd(context);
    if (operand_type == VV) {
      VectorData vs2 = context.register_map->ReadVreg(src[1], context);
      for (int i = context.csr->vstart; i < vd.GetVlen(); i++) {
        if (vs1.GetType() == INT64) {
          vd.SetVmask(
              logical_operate(op, vs1.GetLongData(i), vs2.GetLongData(i)), i);
        } else if (vs1.GetType() == INT32) {
          vd.SetVmask(logical_operate(op, vs1.GetIntData(i), vs2.GetIntData(i)),
                      i);
        } else if (vs1.GetType() == INT16) {
          vd.SetVmask(
              logical_operate(op, vs1.GetShortData(i), vs2.GetShortData(i)), i);
        } else if (vs1.GetType() == FLOAT16) {
          vd.SetVmask(
              logical_operate(op, vs1.GetHalfData(i), vs2.GetHalfData(i)), i);
        } else if (vs1.GetType() == FLOAT32) {
          vd.SetVmask(
              logical_operate(op, vs1.GetFloatData(i), vs2.GetFloatData(i)), i);
        } else if (vs1.GetType() == CHAR8) {
          vd.SetVmask(
              logical_operate(op, vs1.GetCharData(i), vs2.GetCharData(i)), i);
        } else if (vs1.GetType() == UINT8) {
          vd.SetVmask(logical_operate(op, vs1.GetU8Data(i), vs2.GetU8Data(i)),
                      i);
        } else if (vs1.GetType() == BOOL) {
          vd.SetVmask(
              logical_operate(op, vs1.GetBoolData(i), vs2.GetBoolData(i)), i);
        } else
          throw std::runtime_error("Unsupported type.");
      }
    } else if (operand_type == VI) {
      int64_t x2 = src[1];
      bool b2 = (x2 == 1);
      if (vs1.GetType() == INT64) {
        for (int i = context.csr->vstart; i < vd.GetVlen(); i++)
          vd.SetVmask(logical_operate(op, vs1.GetLongData(i), x2), i);
      } else if (vs1.GetType() == INT32) {
        for (int i = context.csr->vstart; i < vd.GetVlen(); i++)
          vd.SetVmask(logical_operate(op, vs1.GetIntData(i), (int32_t)x2), i);
      } else if (vs1.GetType() == INT16) {
        for (int i = context.csr->vstart; i < vd.GetVlen(); i++)
          vd.SetVmask(logical_operate(op, vs1.GetShortData(i), (int16_t)x2), i);
      } else if (vs1.GetType() == CHAR8) {
        for (int i = context.csr->vstart; i < vd.GetVlen(); i++)
          vd.SetVmask(logical_operate(op, vs1.GetCharData(i), (char)x2), i);
      } else if (vs1.GetType() == UINT8) {
        for (int i = context.csr->vstart; i < vd.GetVlen(); i++)
          vd.SetVmask(logical_operate(op, vs1.GetU8Data(i), (uint8_t)x2), i);
      } else if (vs1.GetType() == BOOL) {
        for (int i = context.csr->vstart; i < vd.GetVlen(); i++)
          vd.SetVmask(logical_operate(op, vs1.GetBoolData(i), (bool)x2), i);
      } else if (vs1.GetType() == FLOAT32 || vs1.GetType() == FLOAT16) {
        throw std::runtime_error("Immediate value cannot be Float type.");
      } else
        throw std::runtime_error("Unsupported type.");
    } else if (operand_type == VX) {
      if (vs1.GetType() == INT64) {
        int64_t rs2 = context.register_map->ReadXreg(src[1], context);
        for (int i = context.csr->vstart; i < vd.GetVlen(); i++)
          vd.SetVmask(logical_operate(op, vs1.GetLongData(i), rs2), i);
      } else if (vs1.GetType() == INT32) {
        int32_t rs2 = context.register_map->ReadXreg(src[1], context);
        for (int i = context.csr->vstart; i < vd.GetVlen(); i++)
          vd.SetVmask(logical_operate(op, vs1.GetIntData(i), rs2), i);
      } else if (vs1.GetType() == INT16) {
        int16_t rs2 = context.register_map->ReadXreg(src[1], context);
        for (int i = context.csr->vstart; i < vd.GetVlen(); i++)
          vd.SetVmask(logical_operate(op, vs1.GetShortData(i), rs2), i);
      } else if (vs1.GetType() == CHAR8) {
        char rs2 = context.register_map->ReadXreg(src[1], context);
        for (int i = context.csr->vstart; i < vd.GetVlen(); i++)
          vd.SetVmask(logical_operate(op, vs1.GetCharData(i), rs2), i);
      } else if (vs1.GetType() == UINT8) {
        uint8_t rs2 = context.register_map->ReadXreg(src[1], context);
        for (int i = context.csr->vstart; i < vd.GetVlen(); i++)
          vd.SetVmask(logical_operate(op, vs1.GetU8Data(i), rs2), i);
      } else if (vs1.GetType() == BOOL) {
        bool rs2 = context.register_map->ReadXreg(src[1], context);
        for (int i = context.csr->vstart; i < vd.GetVlen(); i++)
          vd.SetVmask(logical_operate(op, vs1.GetBoolData(i), rs2), i);
      } else if (vs1.GetType() == FLOAT32 || vs1.GetType() == FLOAT16) {
        throw std::runtime_error("Operand type VX is not for Float type.");
      } else
        throw std::runtime_error("Unsupported type.");
    } else if (operand_type == VF) {
      if (vs1.GetType() == FLOAT32) {
        float rs2 = context.register_map->ReadFreg(src[1], context);
        for (int i = context.csr->vstart; i < vd.GetVlen(); i++)
          vd.SetVmask(logical_operate(op, vs1.GetFloatData(i), rs2), i);
      } else if (vs1.GetType() == FLOAT16) {
        float rs2 = context.register_map->ReadFreg(src[1], context);
        for (int i = context.csr->vstart; i < vd.GetVlen(); i++)
          vd.SetVmask(logical_operate(op, vs1.GetHalfData(i), (half)rs2), i);
      } else
        throw std::runtime_error("Unsupported type.");
    }
    context.register_map->WriteVreg(dest, vd, context);
  } else if (opcode == VMAND) {
    VectorData vs1 = context.register_map->ReadVreg(src[0], context);
    VectorData vd(context);
    if (operand_type == MM) {
      VectorData vs2 = context.register_map->ReadVreg(src[1], context);
      for (int i = context.csr->vstart; i < vd.GetVlen(); i++) {
        vd.SetVmask((vs1.GetVmaskData(i) && vs2.GetVmaskData(i)), i);
      }
    } else
      throw std::runtime_error("VMAND must use mm operand_type");
    vd.SetType(VMASK);
    context.register_map->WriteVreg(dest, vd, context);
  } else if (opcode == VMOR) {
    VectorData vs1 = context.register_map->ReadVreg(src[0], context);
    VectorData vd(context);
    if (operand_type == MM) {
      VectorData vs2 = context.register_map->ReadVreg(src[1], context);
      for (int i = context.csr->vstart; i < vd.GetVlen(); i++) {
        vd.SetVmask((vs1.GetVmaskData(i) || vs2.GetVmaskData(i)), i);
      }
    } else
      throw std::runtime_error("VMAND must use mm operand_type");
    vd.SetType(VMASK);
    context.register_map->WriteVreg(dest, vd, context);
  } else if (opcode == VAND) {
    VectorData vs1 = context.register_map->ReadVreg(src[0], context);
    VectorData vd(context);
    if (operand_type == VV) {
      VectorData vs2 = context.register_map->ReadVreg(src[1], context);
      for (int i = context.csr->vstart; i < vd.GetVlen(); i++) {
        if (vs1.GetType() == INT32) {
          vd.SetData(
              (int32_t)(vs1.GetIntData(i) & vs2.GetIntData(i)), i);          
        } else if (vs1.GetType() == INT64) {
          vd.SetData((int64_t)(vs1.GetLongData(i) & vs2.GetLongData(i)), i);
        } else if (vs1.GetType() == FLOAT32) {
          vd.SetData((int32_t)(vs1.GetFloatData(i) * vs2.GetFloatData(i)), i);
        } else if (vs1.GetType() == BOOL) {
          vd.SetData((bool)(vs1.GetBoolData(i) & vs2.GetBoolData(i)), i);
        }
      }
      
    } else if (operand_type == VI) {
      int32_t x2 = src[1];
      for (int i = context.csr->vstart; i < vd.GetVlen(); i++) {
        if(context.csr->vtype_vsew == 32) {
          vd.SetData((int32_t)(vs1.GetIntData(i) & x2), i);
        } else if(context.csr->vtype_vsew == 64) {
          vd.SetData((int64_t)(vs1.GetLongData(i) & x2), i);
        } else if(context.csr->vtype_vsew == 16) {
          vd.SetData((int16_t)(vs1.GetShortData(i) & x2), i);
        } else if(context.csr->vtype_vsew == 8) {
          vd.SetData((int8_t)(vs1.GetCharData(i) & x2), i);
        }
      }
    }
    context.register_map->WriteVreg(dest, vd, context);
  } else if (opcode == VOR) {
    VectorData vs1 = context.register_map->ReadVreg(src[0], context);
    VectorData vd = context.register_map->ReadVreg(dest, context);
    if (operand_type == VV) {
      VectorData vs2 = context.register_map->ReadVreg(src[1], context);
      VectorData vm = context.register_map->ReadVreg(src[2], context);
      for (int i = context.csr->vstart; i < vd.GetVlen(); i++) {
        if (!vm.GetVmaskData(i)) continue;
        if (vs1.GetType() == INT32) {  // Check if Data is not [0, 1]
          vd.SetData(
              (int32_t)(vs1.GetIntData(i) == 1 || vs2.GetIntData(i) == 1), i);
        } else if (vs1.GetType() == FLOAT32) {
          vd.SetData((int32_t)(vs1.GetFloatData(i) == 1.0 ||
                               vs2.GetFloatData(i) == 1.0),
                     i);
        } else if (vs1.GetType() == BOOL) {
          vd.SetData((bool)(vs1.GetBoolData(i) || vs2.GetBoolData(i)), i);
        }  // else if (vs1.GetType() == CHAR8) {
        //   vd.SetData((int32_t)(vs1.GetIntData(i) == vs2.GetIntData(i)), i);
        // }
      }
    } else if (operand_type == VI) {
      int32_t x2 = src[1];
      for (int i = context.csr->vstart; i < vd.GetVlen(); i++) {
        // if(vs1.GetIntData(i) == x2) vd.SetData((int32_t)1, i);
        // else vd.SetData((int32_t)0, i);
      }
    }
    context.register_map->WriteVreg(dest, vd, context);
  } else if (opcode == VFIRST) {
    VectorData vs0 = context.register_map->ReadVreg(src[0], context);
    int first = -1;
    assert(vs0.GetType() == VMASK);
    for (int i = context.csr->vstart; i < vs0.GetVlen(); i++) {
      if (vs0.GetVmaskData(i) == true) {
        first = i;
        break;
      }
    }
    context.register_map->WriteXreg(dest, first, context);
  } else if (opcode == VMIN) {
    VectorData vs1 = context.register_map->ReadVreg(src[0], context);
    VectorData vd(double_reg);
    vd.SetType(INT32);
    if (operand_type == VV) {
      VectorData vs2 = context.register_map->ReadVreg(src[1], context);
      for (int i = context.csr->vstart; i < vd.GetVlen(); i++) {
        if (vs2.GetType() == INT32) {  // Check if Data is not [0, 1]
          int32_t min = (vs1.GetIntData(i) < vs2.GetIntData(i))
                            ? vs1.GetIntData(i)
                            : vs2.GetIntData(i);
          vd.SetData(min, i);
        } else if (vs2.GetType() == FLOAT32) {
          float min = (vs1.GetFloatData(i) < vs2.GetFloatData(i))
                          ? vs1.GetFloatData(i)
                          : vs2.GetFloatData(i);
          vd.SetData(min, i);
        } else if (vs2.GetType() == FLOAT16) {
          half min = (vs1.GetHalfData(i) < vs2.GetHalfData(i))
                          ? vs1.GetHalfData(i)
                          : vs2.GetHalfData(i);
          vd.SetData(min, i);
        } else if (vs2.GetType() == CHAR8) {
          char min = (vs1.GetCharData(i) < vs2.GetCharData(i))
                         ? vs1.GetCharData(i)
                         : vs2.GetCharData(i);
          vd.SetData(min, i);
        }
      }
    } else if (operand_type == VI) {
    } else if (operand_type == VF) {
    } else {
      throw std::runtime_error("Unsupported vector instruction");
    }
    context.register_map->WriteVreg(dest, vd, context);
  } else if (opcode == VMAX) {
    VectorData vs1 = context.register_map->ReadVreg(src[0], context);
    VectorData vd(context);
    if (operand_type == VV) {
      VectorData vs2 = context.register_map->ReadVreg(src[1], context);
      for (int i = context.csr->vstart; i < vd.GetVlen(); i++) {
        if (vs1.GetType() == INT32) {  // Check if Data is not [0, 1]
          int32_t max = (vs1.GetIntData(i) > vs2.GetIntData(i))
                            ? vs1.GetIntData(i)
                            : vs2.GetIntData(i);
          vd.SetData(max, i);
        } else if (vs1.GetType() == FLOAT32) {
          float max = (vs1.GetFloatData(i) > vs2.GetFloatData(i))
                          ? vs1.GetFloatData(i)
                          : vs2.GetFloatData(i);
          vd.SetData(max, i);
        } else if (vs1.GetType() == FLOAT16) {
          half max = (vs1.GetHalfData(i) > vs2.GetHalfData(i))
                          ? vs1.GetHalfData(i)
                          : vs2.GetHalfData(i);
          vd.SetData(max, i);
        } else if (vs1.GetType() == CHAR8) {
          char max = (vs1.GetCharData(i) > vs2.GetCharData(i))
                         ? vs1.GetCharData(i)
                         : vs2.GetCharData(i);
          vd.SetData(max, i);
        }
      }
    } else if (operand_type == VI) {
    } else if (operand_type == VF) {
    } else {
      throw std::runtime_error("Unsupported vector instruction");
    }
    context.register_map->WriteVreg(dest, vd, context);
  } else if (opcode == VCOMPRESS) {
    VectorData vs1 = context.register_map->ReadVreg(src[0], context);
    if (!CheckFloatOp()) vs1.SetType(INT32);
    if (CheckFloatOp() && word_type_op) vs1.SetType(FLOAT32);
    if (CheckFloatOp() && !word_type_op) vs1.SetType(FLOAT16);
    VectorData vd(context);
    if (operand_type == VM) {
      VectorData vs2 = context.register_map->ReadVreg(src[1], context);
      int j = context.csr->vstart;
      for (int i = context.csr->vstart; i < vs1.GetVlen(); i++) {
        if (vs1.GetType() == INT32) {  // Check if Data is not [0, 1]
          bool i2 = vs2.GetVmaskData(i);
          if (i2) {
            vd.SetData(vs1.GetIntData(i), j++);
          }  // else throw std::runtime_error("Unsupported form of a vector
             // mask");
        } else if (vs1.GetType() == FLOAT32) {
        }
      }
      for (j; j < vd.GetVlen(); j++) vd.SetData(0, j);
    } else {
      throw std::runtime_error("Unsupported vector instruction");
    }
    context.register_map->WriteVreg(dest, vd, context);
  } else if (opcode == VSLIDE1DOWN) {
    VectorData vs1 = context.register_map->ReadVreg(src[0], context);
    VectorData vd(context);

    if (vs1.GetType() == FLOAT16) {
      for (int i = context.csr->vstart; i < vd.GetVlen() - 1; i++) {
        vd.SetData(vs1.GetHalfData(i + 1), i);
      }
      vd.SetData(half(context.register_map->ReadXreg(src[1], context)),
                 vd.GetVlen() - 1);
      vd.SetType(FLOAT16);
    } else if (vs1.GetType() == FLOAT32) {
      for (int i = context.csr->vstart; i < vd.GetVlen() - 1; i++) {
        vd.SetData(vs1.GetFloatData(i + 1), i);
      }
      vd.SetData((float)(context.register_map->ReadXreg(src[1], context)),
                 vd.GetVlen() - 1);
      vd.SetType(FLOAT32);
    } else if (vs1.GetType() == INT16) {
      for (int i = context.csr->vstart; i < vd.GetVlen() - 1; i++) {
        vd.SetData(vs1.GetShortData(i + 1), i);
      }
      vd.SetData((int16_t)(context.register_map->ReadXreg(src[1], context)),
                 vd.GetVlen() - 1);
      vd.SetType(INT16);
    } else if (vs1.GetType() == INT32) {
      for (int i = context.csr->vstart; i < vd.GetVlen() - 1; i++) {
        vd.SetData(vs1.GetIntData(i + 1), i);
      }
      vd.SetData((int32_t)(context.register_map->ReadXreg(src[1], context)),
                 vd.GetVlen() - 1);
      vd.SetType(INT32);
    } else if (vs1.GetType() == INT64) {
      for (int i = context.csr->vstart; i < vd.GetVlen() - 1; i++) {
        vd.SetData(vs1.GetLongData(i + 1), i);
      }
      vd.SetData((int64_t)(context.register_map->ReadXreg(src[1], context)),
                 vd.GetVlen() - 1);
      vd.SetType(INT64);
    }
    else if (vs1.GetType() == CHAR8) {
      for (int i = context.csr->vstart; i < vd.GetVlen() - 1; i++) {
        vd.SetData(vs1.GetCharData(i + 1), i);
      }
      vd.SetData((char)(context.register_map->ReadXreg(src[1], context)),
                 vd.GetVlen() - 1);
      vd.SetType(CHAR8);
    }
    context.register_map->WriteVreg(dest, vd, context);
  } else if (opcode == VSLIDEDOWN) {
    VectorData vs1 = context.register_map->ReadVreg(src[0], context);
    VectorData vd(context);
    int move = 0;

    if (operand_type == VX)
      move = context.register_map->ReadXreg(src[1], context);
    else if (operand_type == VI)
      move = src[1];
    else
      throw std::runtime_error("Unsupported operand type");

    if (vs1.GetType() == FLOAT16) {
      for (int i = context.csr->vstart; i < vd.GetVlen() - move; i++) {
        vd.SetData(vs1.GetHalfData(i + move), i);
      }
      for (int i = 1; i < move + 1; i++) vd.SetData(0, vd.GetVlen() - i);
      vd.SetType(FLOAT16);
    } else if (vs1.GetType() == FLOAT32) {
      for (int i = context.csr->vstart; i < vd.GetVlen() - move; i++) {
        vd.SetData(vs1.GetFloatData(i + move), i);
      }
      for (int i = 1; i < move + 1; i++) vd.SetData(0, vd.GetVlen() - i);
      vd.SetType(FLOAT32);
    } else if (vs1.GetType() == INT16) {
      for (int i = context.csr->vstart; i < vd.GetVlen() - move; i++) {
        vd.SetData(vs1.GetShortData(i + move), i);
      }
      for (int i = 1; i < move + 1; i++) vd.SetData(0, vd.GetVlen() - i);
      vd.SetType(INT16);
    } else if (vs1.GetType() == INT32) {
      for (int i = context.csr->vstart; i < vd.GetVlen() - move; i++) {
        vd.SetData(vs1.GetIntData(i + move), i);
      }
      for (int i = 1; i < move + 1; i++) vd.SetData(0, vd.GetVlen() - i);
      vd.SetType(INT32);
    } else if (vs1.GetType() == CHAR8) {
      for (int i = context.csr->vstart; i < vd.GetVlen() - move; i++) {
        vd.SetData(vs1.GetCharData(i + move), i);
      }
      for (int i = 1; i < move + 1; i++) vd.SetData(0, vd.GetVlen() - i);
      vd.SetType(CHAR8);
    }
    context.register_map->WriteVreg(dest, vd, context);
  } else if (opcode == VSLIDE1UP) {  // TODO: implement VSLIDEUP instructions
    VectorData vs1 = context.register_map->ReadVreg(src[0], context);
    VectorData vd(context);

    if (vs1.GetType() == FLOAT16) {
      for (int i = context.csr->vstart; i < vd.GetVlen() - 1; i++) {
        vd.SetData(vs1.GetHalfData(i), i + 1);
      }
      vd.SetData(half(context.register_map->ReadXreg(src[1], context)), 0);
      vd.SetType(FLOAT16);
    } else if (vs1.GetType() == FLOAT32) {
      for (int i = context.csr->vstart; i < vd.GetVlen() - 1; i++) {
        vd.SetData(vs1.GetFloatData(i), i + 1);
      }
      vd.SetData((float)(context.register_map->ReadXreg(src[1], context)), 0);
      vd.SetType(FLOAT32);
    } else if (vs1.GetType() == INT16) {
      for (int i = context.csr->vstart; i < vd.GetVlen() - 1; i++) {
        vd.SetData(vs1.GetShortData(i), i + 1);
      }
      vd.SetData((int16_t)(context.register_map->ReadXreg(src[1], context)), 0);
      vd.SetType(INT16);
    } else if (vs1.GetType() == INT32) {
      for (int i = context.csr->vstart; i < vd.GetVlen() - 1; i++) {
        vd.SetData(vs1.GetIntData(i), i + 1);
      }
      vd.SetData((int32_t)(context.register_map->ReadXreg(src[1], context)), 0);
      vd.SetType(INT32);
    } else if (vs1.GetType() == INT64) {
      for (int i = context.csr->vstart; i < vd.GetVlen() - 1; i++) {
        vd.SetData(vs1.GetLongData(i), i + 1);
      }
      vd.SetData((int64_t)(context.register_map->ReadXreg(src[1], context)), 0);
      vd.SetType(INT64);
    } else if (vs1.GetType() == CHAR8) {
      for (int i = context.csr->vstart; i < vd.GetVlen() - 1; i++) {
        vd.SetData(vs1.GetCharData(i), i + 1);
      }
      vd.SetData((char)(context.register_map->ReadXreg(src[1], context)), 0);
      vd.SetType(CHAR8);
    }
    context.register_map->WriteVreg(dest, vd, context);
  } else if (opcode == VSLIDEUP) {
  } else if (opcode == VLSSEG) {
    int64_t base = context.register_map->ReadXreg(src[0], context);
    VectorData vs2 = context.register_map->ReadVreg(src[1], context);
    VectorData temp(context);
    std::array<VectorData, 8> vd;
    for (int i = 0; i < 8; i++) vd[i] = temp;

    if (operand_type == V) {
      for (int i = context.csr->vstart; i < vs2.GetVlen(); i++) {
        int64_t addr = base;

        if (vs2.GetType() == INT64)
          addr += (vs2.GetLongData(i) * context.csr->vtype_vsew / BYTE_BIT);
        else if (vs2.GetType() == INT32)
          addr += (vs2.GetIntData(i) * context.csr->vtype_vsew / BYTE_BIT);
        else if (vs2.GetType() == INT16)
          addr += (vs2.GetShortData(i) * context.csr->vtype_vsew / BYTE_BIT);
        else
          throw std::runtime_error("Unsupported type");

        for (int j = 0; j < segCnt; j++) {
          int64_t new_addr = addr + (j * context.csr->vtype_vsew / BYTE_BIT);
          int64_t new_base = MemoryMap::FormatAddr(new_addr);
          int64_t offset = new_addr - new_base;
          VectorData vs = MemoryMapLoad(context, new_base);

          if (vs.GetType() == CHAR8)
            vd[j].SetData(vs.GetCharData(offset % PACKET_SIZE),
                          i);  // Only consider element width == 8
          else if (vs.GetType() == UINT8)
            vd[j].SetData(vs.GetU8Data(offset % PACKET_SIZE), i);
        }
      }
    } else {
      throw std::runtime_error("Unsupported Segment Indexed Load instruction");
    }
    for (int i = 0; i < segCnt; i++) {
      context.register_map->WriteVreg(segDest[i], vd[i], context);
    }
  } else if (opcode == TEST) {
    if (operand_type == V_X) {
        printf("hex(%llx) dec(%lld)\n",
               context.register_map->ReadXreg(src[0], context),
               context.register_map->ReadXreg(src[0], context));
    } else if (operand_type == V_F) {
      printf("float(%f)\n", context.register_map->ReadFreg(src[0], context));
    } else {
      VectorData vs0 = context.register_map->ReadVreg(src[0], context);
      if (vs0.GetType() == INT64)
        printf("INT64: vlen:%d %dByte ", vs0.GetVlen(), 8 * vs0.GetVlen());
      else if (vs0.GetType() == INT32)
        printf("INT32: vlen:%d %dByte ", vs0.GetVlen(), 4 * vs0.GetVlen());
      else if (vs0.GetType() == INT16)
        printf("INT16: vlen:%d %dByte ", vs0.GetVlen(), 2 * vs0.GetVlen());
      else if (vs0.GetType() == CHAR8)
        printf("CHAR8: vlen:%d %dByte ", vs0.GetVlen(), vs0.GetVlen());
      else if (vs0.GetType() == UINT8)
        printf("UINT8: vlen:%d %dByte ", vs0.GetVlen(), vs0.GetVlen());
      else if (vs0.GetType() == FLOAT32)
        printf("FLOAT32: vlen:%d %dByte ", vs0.GetVlen(), 4 * vs0.GetVlen());
      else if (vs0.GetType() == FLOAT16)
        printf("FLOAT16: vlen:%d %dByte ", vs0.GetVlen(), 2 * vs0.GetVlen());
      else if (vs0.GetType() == BOOL)
        printf("BOOL: vlen:%d %dByte ", vs0.GetVlen(), vs0.GetVlen());
      else if (vs0.GetType() == VMASK)
        printf("VMASK: vlen:%d %dByte ", vs0.GetVlen(), vs0.GetVlen() / 8);
      else
        printf("Unknown Type: ");

      vs0.PrintVectorData();
    }
  } else if (opcode == VFMACC) {
    VectorData vs1 = context.register_map->ReadVreg(src[0], context);
    VectorData vs_temp = context.register_map->ReadVreg(src[0], context);
    VectorData vd = context.register_map->ReadVreg(dest, context);

    if (operand_type == VV) {
      VectorData vs2 = context.register_map->ReadVreg(src[1], context);
      vd = vs1 * vs2 + vd;
    } else if (operand_type == VF) {
      float f2 = context.register_map->ReadFreg(src[1], context);
      vd = vs1 * f2 + vd;
    } else {
      throw std::runtime_error("Unsupported vector instruction");
    }
    context.register_map->WriteVreg(dest, vd, context);
  } else if (opcode == VFMSUB) {
    VectorData vs1 = context.register_map->ReadVreg(src[1], context);
    VectorData vs_temp = context.register_map->ReadVreg(src[1], context);
    VectorData vd = context.register_map->ReadVreg(dest, context);

    if (operand_type == VV) {
      VectorData vs2 = context.register_map->ReadVreg(src[0], context);
      vd = vd * vs2 - vs1;
    } else if (operand_type == VF) {
      float f2 = context.register_map->ReadFreg(src[0], context);
      vd = vd * f2 - vs1;
    } else {
      throw std::runtime_error("Unsupported vector instruction");
    }
    context.register_map->WriteVreg(dest, vd, context);
  } else if (opcode == VFWMACC) {
    if (word_type_op)
      throw std::runtime_error("VFWMACC must use double dest register");
    VectorData vs1 = context.register_map->ReadVreg(src[0], context);
    VectorData vs_temp = context.register_map->ReadVreg(src[0], context);
    VectorData vd = context.register_map->ReadVreg(dest, context);
    if (operand_type == VV) {
      VectorData vs2 = context.register_map->ReadVreg(src[1], context);
      if (vs1.GetType() != vs2.GetType())
        throw std::runtime_error("VFWMACC must use same typed operands");
      if (vs1.GetType() == FLOAT16) {
        vd.SetType(FLOAT32);
        vs1 = vs1 * vs2;
        vd = vd + vs1;
      } else if (vs1.GetType() == INT16) {
        vd.SetType(INT32);
        vs1 = vs1 * vs2;
        vd = vd + vs1;
      } else
        throw std::runtime_error("Not supported type");
      context.register_map->WriteVreg(dest, vd, context);
    } else if (operand_type == VF) {
      if (vs1.GetType() != FLOAT16 && vd.GetType() != FLOAT32)
        throw std::runtime_error("VFWMACC.vf must use half typed operands");
      float f2 = context.register_map->ReadFreg(src[1], context);
      VectorData temp(32, context.csr->vtype_vlmul * 2);
      temp.SetType(FLOAT32);
      for (int i = 0; i < vd.GetVlen(); i++) {
        temp.SetData((float)(vd.GetFloatData(i) + (float)vs1.GetHalfData(i) * f2), i);
      }
      context.register_map->WriteVreg(dest, temp, context);
    } else {
      throw std::runtime_error("Unsupported vector instruction");
    }
    // context.register_map->WriteVreg(dest, vd, context);
  } else if (opcode == VFNCVT) {
    // if (word_type_op)
    //   throw std::runtime_error("VFNCVT must use single dest register");

    VectorData vs = context.register_map->ReadVreg(src[0], context);
    DataType dt = vs.GetType();
    int vl = vs.GetVlen();
    int precision = (dt == INT32) || (dt == FLOAT32) ? 4 : 2;
    bool double_reg = (precision * vl) > 32;

    VectorData vd(double_reg);

    for (int i = context.csr->vstart; i < vd.GetVlen(); i++) {
      if (operand_type == XU_F_W || operand_type == X_F_W) {
        vd.SetData((uint16_t)vs.GetFloatData(i), i);
      } else if (operand_type == F_F_W) {
        vd.SetData((half)vs.GetFloatData(i), i);
      } else if (operand_type == F_XU_W || operand_type == F_X_W) {
        vd.SetData((half)vs.GetIntData(i), i);
      }
    }
    context.register_map->WriteVreg(dest, vd, context);
  } else if (opcode == VFCVT) {
    VectorData vd(context);
    VectorData vs = context.register_map->ReadVreg(src[0], context);
    for (uint32_t i = context.csr->vstart; i < vd.GetVlen(); i++) {
      if (operand_type == F_XU_V || operand_type == F_X_V) {
        if (vs.GetType() == INT32)
          vd.SetData((float)vs.GetIntData(i), i);
        else
          vd.SetData((half)vs.GetShortData(i), i);
      } else if (operand_type == XU_F_V || operand_type == X_F_V) {
        vd.SetData((int32_t)vs.GetFloatData(i), i);
      }
    }
    context.register_map->WriteVreg(dest, vd, context);
  } else if (opcode == VFWCVT) {
    // if (!word_type_op)
    //   throw std::runtime_error("VFWCVT must use double dest register");
    VectorData vd(context);
    // vd.Widen();
    vd.SetVectorDoubleReg();
    VectorData vs = context.register_map->ReadVreg(src[0], context);

    for (int i = context.csr->vstart; i < vd.GetVlen(); i++) {
      if (operand_type == F_XU_V || operand_type == F_X_V) {
        if (vs.GetType() == INT32)
          vd.SetData((float)vs.GetIntData(i), i);
        else
          vd.SetData((float)vs.GetShortData(i), i);
      } else if (operand_type == F_F_V) {
        vd.SetData((float)vs.GetHalfData(i), i);
      } else if (operand_type == XU_F_V || operand_type == X_F_V) {
        vd.SetData((int32_t)vs.GetFloatData(i), i);
      }
    }
    context.register_map->WriteVreg(dest, vd, context);
  } else if (opcode == VFMV) {
    if (operand_type == V_V) {
      VectorData vd(word_type_op);
      VectorData vs1 = context.register_map->ReadVreg(src[0], context);
      vd = vs1;
    } else if (operand_type == V_I) {
      VectorData vd(word_type_op);
      int32_t x1 = src[0];
      for (int i = context.csr->vstart; i < vd.GetVlen(); i++) {
        vd.SetData((float)x1, i);
      }
      vd.SetType(INT32);
    } else if (operand_type == V_X) {
      VectorData vd(context);
      int32_t x1 = context.register_map->ReadXreg(src[0], context);
      for (int i = context.csr->vstart; i < vd.GetVlen(); i++) {
        vd.SetData(x1, i);
      }
      vd.SetType(INT32);
    } else if (operand_type == V_F) {
      // VectorData vd(word_type_op);
      VectorData vd(context);
      if(context.register_map->CheckExistRenameVreg(dest)) {
        vd = context.register_map->ReadVreg(dest, context);
      }
      if (!CheckFloatOp())
        throw std::runtime_error("VFMV must use float dest register");
      float f2 = context.register_map->ReadFreg(src[0], context);
      if (word_type_op) {
        for (int i = context.csr->vstart; i < vd.GetVlen(); i++) {
          vd.SetData(f2, i);
        }
        vd.SetType(FLOAT32);
      } else {
        for (int i = context.csr->vstart; i < vd.GetVlen(); i++) {
          vd.SetData(half(f2), 2 * i);
          vd.SetData(half(f2), 2 * i + 1);
        }
        vd.SetType(FLOAT16);
      }
      context.register_map->WriteVreg(dest, vd, context);
    } else if (operand_type == F_S) {
      /*  Vecotr[0] to float */
      VectorData vs = context.register_map->ReadVreg(src[0], context);
      // if (vs.GetVlen() == PACKET_ENTRIES * 2)
      if (vs.GetType() == FLOAT32)
        context.register_map->WriteFreg(
            dest, vs.GetFloatData(context.csr->vstart), context);
      else if (vs.GetType() == FLOAT16) /*half_vector[0] to float*/
        context.register_map->WriteFreg(
            dest, vs.GetHalfData(context.csr->vstart), context);

    } else if (operand_type == S_F) {
      /*  float to  Vecotr[0] */
      VectorData vd = context.register_map->ReadVreg(dest, context);
      if (word_type_op) {
        vd.SetData(context.register_map->ReadFreg(src[0], context),
                   context.csr->vstart);
      } else {
        vd.SetData((half)context.register_map->ReadFreg(src[0], context),
                   context.csr->vstart);
      }
      context.register_map->WriteVreg(dest, vd, context);
    }
  } else if (opcode == VFSQRT) {
    VectorData vs1 = context.register_map->ReadVreg(src[0], context);
    VectorData vd(context);
    if (operand_type == V) {
      for (int i = context.csr->vstart; i < vs1.GetVlen(); i++) {
        if (vs1.GetType() == FLOAT32) {
          vd.SetData(float(sqrt(vs1.GetFloatData(i))), i);
        } else if (vs1.GetType() == FLOAT16) {
          vd.SetData(half(sqrt(vs1.GetHalfData(i))), i);
        }
      }
    } else throw std::runtime_error("Unsupported vector instruction");
    context.register_map->WriteVreg(dest, vd, context);
  } else if (opcode == VFREDOMAX) {
    if (operand_type != VS)
      throw std::runtime_error("VFREDOMAX must use vector-vector[0]");
    VectorData vs1 = context.register_map->ReadVreg(src[0], context);
    VectorData vs2 = context.register_map->ReadVreg(src[1], context);
    if (word_type_op) {
      VectorData vd(double_reg);
      float max = vs2.GetFloatData(0);
      for (int i = context.csr->vstart; i < vs1.GetVlen(); i++) {
        max = std::max<float>(max, vs1.GetFloatData(i));
      }
      vd.SetData(max, 0);
      context.register_map->WriteVreg(dest, vd, context);
    } else {
      VectorData vd(context);
      half max = vs2.GetHalfData(0);
      for (int i = context.csr->vstart; i < vs1.GetVlen(); i++) {
        max = std::max<half>(max, vs1.GetHalfData(i));
      }
      vd.SetData(max, 0);
      context.register_map->WriteVreg(dest, vd, context);
    }
  } else if (opcode == VFREDOSUM) {
    if (operand_type != VS)
      throw std::runtime_error("VFREDOMAX must use vector-vector[0]");
    VectorData vs1 = context.register_map->ReadVreg(src[0], context);
    VectorData vs2 = context.register_map->ReadVreg(src[1], context);
    if (word_type_op) {
      VectorData vd(context);
      float sum = vs2.GetFloatData(0);
      for (int i = context.csr->vstart; i < vs1.GetVlen(); i++) {
        sum = sum + vs1.GetFloatData(i);
      }
      vd.SetData(sum, 0);
      context.register_map->WriteVreg(dest, vd, context);
    } else {
      VectorData vd(context);
      half sum = vs2.GetHalfData(0);
      for (int i = context.csr->vstart; i < vs1.GetVlen(); i++) {
        sum = sum + vs1.GetHalfData(i);
      }
      vd.SetData(sum, 0);
      context.register_map->WriteVreg(dest, vd, context);
    }
  } else if (opcode == VREDSUM) {
    if (operand_type != VS)
      throw std::runtime_error("VREDSUM must use vector-vector[0]");
    VectorData vs1 = context.register_map->ReadVreg(src[0], context);
    VectorData vs2 = context.register_map->ReadVreg(src[1], context);
    VectorData vd(context);

    if (vs2.GetType() == INT64) {
      int64_t sum = vs2.GetLongData(0);
      for (int i = context.csr->vstart; i < vs1.GetVlen(); i++)
        sum += vs1.GetLongData(i);
      vd.SetData(sum, 0);
      context.register_map->WriteVreg(dest, vd, context);
    } else if (vs2.GetType() == INT32) {
      int32_t sum = vs2.GetIntData(0);
      for (int i = context.csr->vstart; i < vs1.GetVlen(); i++)
        sum += vs1.GetIntData(i);
      vd.SetData(sum, 0);
      context.register_map->WriteVreg(dest, vd, context);
    } else if (vs2.GetType() == INT16) {
      int16_t sum = vs2.GetShortData(0);
      for (int i = context.csr->vstart; i < vs1.GetVlen(); i++)
        sum += vs1.GetShortData(i);
      vd.SetData(sum, 0);
      context.register_map->WriteVreg(dest, vd, context);
    } else if (vs2.GetType() == FLOAT32) {
      float sum = vs2.GetFloatData(0);
      for (int i = context.csr->vstart; i < vs1.GetVlen(); i++)
        sum += vs1.GetFloatData(i);
      vd.SetData(sum, 0);
      context.register_map->WriteVreg(dest, vd, context);
    } else if (vs2.GetType() == FLOAT16) {
      half sum = vs2.GetHalfData(0);
      for (int i = context.csr->vstart; i < vs1.GetVlen(); i++)
        sum += vs1.GetHalfData(i);
      vd.SetData(sum, 0);
      context.register_map->WriteVreg(dest, vd, context);
    } else if (vs2.GetType() == CHAR8) {
      throw std::runtime_error("Can't use char8 type");
    } else
      throw std::runtime_error("Unsupported type");
  } else if (opcode == VREDMAX) {
    if (operand_type != VS)
      throw std::runtime_error("VREDMAX must use vector-vector[0]");
    VectorData vs1 = context.register_map->ReadVreg(src[0], context);
    VectorData vs2 = context.register_map->ReadVreg(src[1], context);
    VectorData vd(context);
    if (vs2.GetType() == INT64) {
      int64_t max = vs2.GetLongData(0);
      for (int i = context.csr->vstart; i < vs1.GetVlen(); i++)
        max = std::max<int64_t>(max, vs1.GetLongData(i));
      vd.SetData(max, 0);
    } else if (vs2.GetType() == INT32) {
      int32_t max = vs2.GetIntData(0);
      for (int i = context.csr->vstart; i < vs1.GetVlen(); i++)
        max = std::max<int32_t>(max, vs1.GetIntData(i));
      vd.SetData(max, 0);
    } else if (vs2.GetType() == INT16) {
      int16_t max = vs2.GetShortData(0);
      for (int i = context.csr->vstart; i < vs1.GetVlen(); i++)
        max = std::max<int16_t>(max, vs1.GetShortData(i));
      vd.SetData(max, 0);
    } else if (vs2.GetType() == FLOAT32) {
      float max = vs2.GetFloatData(0);
      for (int i = context.csr->vstart; i < vs1.GetVlen(); i++)
        max = std::max<float>(max, vs1.GetFloatData(i));
      vd.SetData(max, 0);
    } else if (vs2.GetType() == FLOAT16) {
      half max = vs2.GetHalfData(0);
      for (int i = context.csr->vstart; i < vs1.GetVlen(); i++)
        max = std::max<half>(max, vs1.GetHalfData(i));
      vd.SetData(max, 0);
    } else if (vs2.GetType() == CHAR8) {
      throw std::runtime_error("Can't use char8 type");
    } else
      throw std::runtime_error("Unsupported type");
    context.register_map->WriteVreg(dest, vd, context);
  } else if (opcode == VREDMIN) {
    if (operand_type != VS)
      throw std::runtime_error("VREDMIN must use vector-vector[0]");
    VectorData vs1 = context.register_map->ReadVreg(src[0], context);
    VectorData vs2 = context.register_map->ReadVreg(src[1], context);
    VectorData vm;
    if (src[2] != -1) vm = context.register_map->ReadVreg(src[2], context);
    assert(src[2] == -1 || vm.GetType() == VMASK);
    VectorData vd(context);
    if (vs2.GetType() == INT64) {
      int64_t min = vs2.GetLongData(0);
      for (int i = context.csr->vstart; i < vs1.GetVlen(); i++)
        if (src[2] != -1 && vm.GetVmaskData(i))
          min = std::min<int64_t>(min, vs1.GetLongData(i));
      vd.SetData(min, 0);
    } else if (vs2.GetType() == INT32) {
      int32_t min = vs2.GetIntData(0);
      for (int i = context.csr->vstart; i < vs1.GetVlen(); i++)
        if (src[2] != -1 && vm.GetVmaskData(i))
          min = std::min<int32_t>(min, vs1.GetIntData(i));
      vd.SetData(min, 0);
    } else if (vs2.GetType() == INT16) {
      int16_t min = vs2.GetShortData(0);
      for (int i = context.csr->vstart; i < vs1.GetVlen(); i++)
        if (src[2] != -1 && vm.GetVmaskData(i))
          min = std::min<int16_t>(min, vs1.GetShortData(i));
      vd.SetData(min, 0);
    } else if (vs2.GetType() == FLOAT32) {
      float min = vs2.GetFloatData(0);
      for (int i = context.csr->vstart; i < vs1.GetVlen(); i++)
        if (src[2] != -1 && vm.GetVmaskData(i))
          min = std::min<float>(min, vs1.GetFloatData(i));
      vd.SetData(min, 0);
    } else if (vs2.GetType() == FLOAT16) {
      half min = vs2.GetHalfData(0);
      for (int i = context.csr->vstart; i < vs1.GetVlen(); i++)
        if (src[2] != -1 && vm.GetVmaskData(i))
          min = std::min<half>(min, vs1.GetHalfData(i));
      vd.SetData(min, 0);
    } else if (vs2.GetType() == CHAR8) {
      throw std::runtime_error("Can't use char8 type");
    } else
      throw std::runtime_error("Unsupported type");
    context.register_map->WriteVreg(dest, vd, context);
  } else if (opcode == VREDOR) {
    if (operand_type != VS)
      throw std::runtime_error("VREDOR must use vector-vector[0]");
    VectorData vs1 = context.register_map->ReadVreg(src[0], context);
    VectorData vs2 = context.register_map->ReadVreg(src[1], context);
    VectorData vd(context);
    if (vs2.GetType() == INT64) {
      int64_t val = vs2.GetLongData(0);
      for (int i = context.csr->vstart; i < vs1.GetVlen(); i++)
        val = val || vs1.GetLongData(i);
      vd.SetData(val, 0);
    } else if (vs2.GetType() == INT32) {
      int32_t val = vs2.GetIntData(0);
      for (int i = context.csr->vstart; i < vs1.GetVlen(); i++)
        val = val || vs1.GetIntData(i);
      vd.SetData(val, 0);
    } else if (vs2.GetType() == INT16) {
      int16_t val = vs2.GetShortData(0);
      for (int i = context.csr->vstart; i < vs1.GetVlen(); i++)
        val = val || vs1.GetShortData(i);
      vd.SetData(val, 0);
    } else if (vs2.GetType() == FLOAT32) {
      float val = vs2.GetFloatData(0);
      for (int i = context.csr->vstart; i < vs1.GetVlen(); i++)
        val = val || vs1.GetFloatData(i);
      vd.SetData(val, 0);
    } else if (vs2.GetType() == FLOAT16) {
      half val = vs2.GetHalfData(0);
      for (int i = context.csr->vstart; i < vs1.GetVlen(); i++)
        val = val || vs1.GetHalfData(i);
      vd.SetData(val, 0);
    } else if (vs2.GetType() == CHAR8) {  // Todo : check if char8 is supported
      throw std::runtime_error("Can't use char8 type");
    } else
      throw std::runtime_error("Unsupported type");
    context.register_map->WriteVreg(dest, vd, context);
  } else if (opcode == VWREDOSUM) {
    if (operand_type != VS)
      throw std::runtime_error("VFREDOMAX must use vector-vector[0]");
    VectorData vs1 = context.register_map->ReadVreg(src[0], context);
    VectorData vs2 = context.register_map->ReadVreg(src[1], context);
    if (vs1.GetType() == FLOAT16 && vs2.GetType() == FLOAT16) {
      VectorData vd(32, float(context.csr->vtype_vlmul));
      vd.SetType(FLOAT32);
      float sum = vs2.GetHalfData(0);
      for (int i = context.csr->vstart; i < vs1.GetVlen(); i++) {
        sum = sum + vs1.GetHalfData(i);
      }
      vd.SetData(float(sum), 0);
      context.register_map->WriteVreg(dest, vd, context);
    }
  } else if (opcode == VWREDOMAX) {
    if (operand_type != VS)
      throw std::runtime_error("VWREDOMAX must use vector-vector[0]");
    VectorData vs1 = context.register_map->ReadVreg(src[0], context);
    VectorData vs2 = context.register_map->ReadVreg(src[1], context);
    if (vs1.GetType() == FLOAT16 && vs2.GetType() == FLOAT16) {
      VectorData vd(32, float(context.csr->vtype_vlmul));
      vd.SetType(FLOAT32);
      float max = vs2.GetHalfData(0);
      for (int i = context.csr->vstart; i < vs1.GetVlen(); i++) {
        max = std::max<float>(max, (float)vs1.GetHalfData(i));
      }
      vd.SetData(float(max), 0);
      context.register_map->WriteVreg(dest, vd, context);
    }
  } else if (opcode == VLE8) {
    uint64_t addr = context.register_map->ReadXreg(src[0], context);
    uint64_t offset = src[1];
    VectorData vd(context);
    vd = MemoryMapLoad(context, addr + offset);
    context.register_map->WriteVreg(dest, vd, context);
  } else if (opcode == VLE16) {
    uint64_t addr = context.register_map->ReadXreg(src[0], context);
    uint64_t offset = src[1];
    VectorData vd(context);
    vd = MemoryMapLoad(context, addr + offset);
    context.register_map->WriteVreg(dest, vd, context);
  } else if (opcode == VLE32) {
    uint64_t addr = context.register_map->ReadXreg(src[0], context);
    uint64_t offset = src[1];
    VectorData vd;
    VectorData vd2;
    vd = MemoryMapLoad(context, addr + offset);
    if (context.csr->vtype_vlmul == 2) {
      vd2 = MemoryMapLoad(context, addr + offset + PACKET_SIZE);
      vd.Append(vd2);
    }
    context.register_map->WriteVreg(dest, vd, context);
  } else if (opcode == VLE64) {
    uint64_t addr = context.register_map->ReadXreg(src[0], context);
    uint64_t offset = src[1];
    VectorData vd;
    VectorData vd2;
    std::array<VectorData, 8> vd8;
    vd = MemoryMapLoad(context, addr + offset);
    if (context.csr->vtype_vlmul == 2) {
      vd2 = MemoryMapLoad(context, addr + offset + PACKET_SIZE);
      vd.Append(vd2);
    } else if (context.csr->vtype_vlmul == 8) {
      for (int i = 0; i < 8; i++) {
        /* FIXME. no double register... */
        vd8[i] = MemoryMapLoad(context, addr + offset + (i * PACKET_SIZE));
        vd8[0].Append(vd8[i]);
      }
      vd = vd8[0];
    }
    context.register_map->WriteVreg(dest, vd, context);
  } else if (opcode == VSE8) {
    uint64_t addr = context.register_map->ReadXreg(src[0], context);
    uint64_t offset = src[1];
    VectorData vs = context.register_map->ReadVreg(dest, context);
    MemoryMapStore(context, addr + offset, vs);
  } else if (opcode == VSE16) {
    uint64_t addr = context.register_map->ReadXreg(src[0], context);
    uint64_t offset = src[1];
    VectorData vs = context.register_map->ReadVreg(dest, context);
    MemoryMapStore(context, addr + offset, vs);
  } else if (opcode == VSE32) {
    uint64_t addr = context.register_map->ReadXreg(src[0], context);
    uint64_t offset = src[1];
    VectorData vs = context.register_map->ReadVreg(dest, context);
    if (context.csr->vtype_vlmul == 2) {
      std::array<VectorData, 2> vd = vs.Split();
      MemoryMapStore(context, addr + offset, vd[0]);
      MemoryMapStore(context, addr + offset + PACKET_SIZE, vd[1]);
    } else {
      MemoryMapStore(context, addr + offset, vs);
    }
  } else if (opcode == VSE64) {
    uint64_t addr = context.register_map->ReadXreg(src[0], context);
    uint64_t offset = src[1];
    VectorData vs = context.register_map->ReadVreg(dest, context);
    if (context.csr->vtype_vlmul == 2) {
      std::array<VectorData, 2> vd = vs.Split();
      MemoryMapStore(context, addr + offset, vd[0]);
      MemoryMapStore(context, addr + offset + PACKET_SIZE, vd[1]);
    } else {
      MemoryMapStore(context, addr + offset, vs);
    }
  } else if (CheckVectorOp() && CheckAmoOp()) {
    uint64_t base = context.register_map->ReadXreg(src[0], context);
    VectorData offset = context.register_map->ReadVreg(src[1], context);
    VectorData vs = context.register_map->ReadVreg(src[2], context);
    VectorData vmask(context);
    for (int i = 0; i < vmask.GetVlen(); i++) vmask.SetVmask(true, i);
    if (src[3] != -1) vmask = context.register_map->ReadVreg(src[3], context);
    vmask.SetType(VMASK);
    amo_loop(context, vmask, offset, vs, base, GetArithOp());
  } else if (opcode == VLUXEI32) {  // vluxei32.v   vd, (rs1), vs2, vm  #
                                    // unordered 32-bit indexed load of SEW data
    uint64_t base = context.register_map->ReadXreg(
        src[0], context);  // read base address from src[0]
    uint64_t offset = src[1];
    base += offset;
    VectorData vs2 = context.register_map->ReadVreg(src[2], context);
    VectorData vs3 = context.register_map->ReadVreg(src[3], context);
    VectorData vd(context);
    for (int i = 0; i < PACKET_SIZE / WORD_SIZE; i++) {
      uint64_t idx = vs2.GetIntData(i);
      uint64_t vm = vs3.GetVmaskData(i);
      if (vs3.GetType() == VMASK && vm == 0) {
        continue;
      }
      uint64_t addr = base + idx;
      VectorData vs = MemoryMapLoad(context, addr);
      uint64_t s_idx = (addr % PACKET_SIZE) / WORD_SIZE;
      if (vs.GetType() == FLOAT32) {
        s_idx = s_idx % (PACKET_SIZE / WORD_SIZE);
        vd.SetData(vs.GetFloatData(s_idx), i);
      } else if (vs.GetType() == INT32) {
        s_idx = s_idx % (PACKET_SIZE / WORD_SIZE);
        vd.SetData(vs.GetIntData(s_idx), i);
      }
    }
    context.register_map->WriteVreg(dest, vd, context);
    assert(vd.GetType() == INT32 || vd.GetType() == FLOAT32);
  } else if (opcode == VLUXEI64) {  // vluxei64.v   vd, (rs1), vs2, vm  #
                                    // unordered 64-bit indexed load of SEW data
    uint64_t base = context.register_map->ReadXreg(
        src[0], context);  // base address read from src[0]
    uint64_t offset = src[1];
    base += offset;
    VectorData vs2 = context.register_map->ReadVreg(src[2], context);
    VectorData vm;
    VectorData vd(context);
    if (src[3] != -1) vm = context.register_map->ReadVreg(src[3], context);
    for (int i = 0; i < (PACKET_SIZE * context.csr->vtype_vlmul) / DOUBLE_SIZE;
         i++) {
      if (vm.GetType() == VMASK && !vm.GetVmaskData(i)) continue;
      uint64_t idx = vs2.GetLongData(i);
      uint64_t addr = base + idx;
      VectorData vs = MemoryMapLoad(context, addr);
      assert(vs.GetType() == INT64);
      uint64_t s_idx = (addr % PACKET_SIZE) / DOUBLE_SIZE;
      vd.SetData(vs.GetLongData(s_idx), i);
    }
    context.register_map->WriteVreg(dest, vd, context);
  } else if (opcode == VSUXEI32) {
    uint64_t base = context.register_map->ReadXreg(
        src[0], context);  // read base address from src[0]
    uint64_t offset = src[1];
    base += offset;
    uint64_t idx;
    VectorData vs2 = context.register_map->ReadVreg(src[2], context);
    VectorData vd = context.register_map->ReadVreg(dest, context);
    VectorData vs3 = context.register_map->ReadVreg(src[3], context);
    for (int i = 0; i < PACKET_SIZE / WORD_SIZE;
         i++) {  // To do : sew, lmul, 32(vsuxei32)
      idx = vs2.GetIntData(i);
      uint64_t vm = vs3.GetVmaskData(i);
      if (vs3.GetType() == VMASK && vm == 0) {
        continue;
      }
      uint64_t addr = base + idx;
      VectorData s_vd(context);
      if (CheckMemoryMap(context, addr)) {
        s_vd = MemoryMapLoad(context, addr);
      }
      int s_idx = (addr % PACKET_SIZE) / WORD_SIZE;
      if (vd.GetType() == FLOAT32) {
        float data = vd.GetFloatData(i);
        s_vd.SetData(data, s_idx);
      } else if (vd.GetType() == INT32) {
        int32_t data = vd.GetIntData(i);
        s_vd.SetData(data, s_idx);
      } else {
        throw std::runtime_error("Unsupported Data Type (VSUXEI32)");
      }
      MemoryMapStore(context, addr, s_vd);
    }
  } else if (opcode == VFEXP) {   // vfexp.v vd, rs1
    VectorData vs1 = context.register_map->ReadVreg(src[0], context);
    VectorData vd(context);
    vd.SetType(vs1.GetType());
    if (operand_type == V)
      vd = vd.Exp(vs1);
    else
      throw std::runtime_error("Unsupported vector instruction");
    context.register_map->WriteVreg(dest, vd, context);
  } else if (opcode == VFSGNJ) {
    if (operand_type == VV) {
      VectorData vs1 = context.register_map->ReadVreg(src[0], context);
      VectorData vs2 = context.register_map->ReadVreg(src[1], context);
      VectorData vd(context);
      for (int i = context.csr->vstart; i < vd.GetVlen(); i++) {
        if (vs1.GetType() == FLOAT32) {
          float f1, f2, f3;
          bool msb2;
          f1 = vs1.GetFloatData(i);
          f2 = vs2.GetFloatData(i);
          msb2 = f2 < 0;

          f3 = (msb2) ? -1*std::abs(f1) : std::abs(f1);
          vd.SetData(f3, i);
        } else if (vs1.GetType() == FLOAT16) {
          half f1, f2, f3;
          bool msb1, msb2;
          f1 = vs1.GetHalfData(i);
          f2 = vs2.GetHalfData(i);
          msb2 = f2 < 0;

          f3 = (msb2) ? -1*std::abs(f1) : std::abs(f1);
          vd.SetData(f3, i);
        } else {
          throw std::runtime_error("Unsupported vector instruction");
        }
      }
      context.register_map->WriteVreg(dest, vd, context);
    } else if (operand_type == VF) {
      VectorData vs1 = context.register_map->ReadVreg(src[0], context);
      VectorData vd(context);
      for (int i = context.csr->vstart; i < vd.GetVlen(); i++) {
        if (vs1.GetType() == FLOAT32) {
          float f1, f2, f3;
          bool msb1, msb2;
          f1 = vs1.GetFloatData(i);
          f2 = context.register_map->ReadFreg(src[1], context);
          msb2 = f2 < 0;

          f3 = (msb2) ? -1*std::abs(f1) : std::abs(f1);
          vd.SetData(f3, i);
        } else {
          throw std::runtime_error("Unsupported vector instruction");
        }
      }
      context.register_map->WriteVreg(dest, vd, context);
    } else {
      throw std::runtime_error("Unsupported vector instruction");
    }
  } else if (opcode == VFSGNJN) {
    if (operand_type == VV) {
      VectorData vs1 = context.register_map->ReadVreg(src[0], context);
      VectorData vs2 = context.register_map->ReadVreg(src[1], context);
      VectorData vd(context);
      for (int i = context.csr->vstart; i < vd.GetVlen(); i++) {
        if (vs1.GetType() == FLOAT32) {
          float f1, f2, f3;
          bool msb2;
          f1 = vs1.GetFloatData(i);
          f2 = vs2.GetFloatData(i);
          msb2 = f2 < 0;

          f3 = (msb2) ? std::abs(f1) : -1*std::abs(f1);
          vd.SetData(f3, i);
        } else if (vs1.GetType() == FLOAT16) {
          half f1, f2, f3;
          bool msb1, msb2;
          f1 = vs1.GetHalfData(i);
          f2 = vs2.GetHalfData(i);
          msb2 = f2 < 0;

          f3 = (msb2) ? std::abs(f1) : -1*std::abs(f1);
          vd.SetData(f3, i);
        } else {
          throw std::runtime_error("Unsupported vector instruction");
        }
      }
      context.register_map->WriteVreg(dest, vd, context);
    } else if (operand_type == VF) {
      VectorData vs1 = context.register_map->ReadVreg(src[0], context);
      VectorData vd(context);
      for (int i = context.csr->vstart; i < vd.GetVlen(); i++) {
        if (vs1.GetType() == FLOAT32) {
          float f1, f2, f3;
          bool msb1, msb2;
          f1 = vs1.GetFloatData(i);
          f2 = context.register_map->ReadFreg(src[1], context);
          msb2 = f2 < 0;

          f3 = (msb2) ? std::abs(f1) : -1*std::abs(f1);
          vd.SetData(f3, i);
        } else {
          throw std::runtime_error("Unsupported vector instruction");
        }
      }
      context.register_map->WriteVreg(dest, vd, context);
    } else {
      throw std::runtime_error("Unsupported vector instruction");
    }
  } else if (opcode == VFSGNJX) {
    if (operand_type == VV) {
      VectorData vs1 = context.register_map->ReadVreg(src[0], context);
      VectorData vs2 = context.register_map->ReadVreg(src[1], context);
      VectorData vd(context);
      for (int i = context.csr->vstart; i < vd.GetVlen(); i++) {
        if (vs1.GetType() == FLOAT32) {
          float f1, f2, f3;
          bool msb1, msb2;
          f1 = vs1.GetFloatData(i);
          f2 = vs2.GetFloatData(i);
          msb1 = f1 < 0;
          msb2 = f2 < 0;

          f3 = (msb1 ^ msb2) ? -1*std::abs(f1) : std::abs(f1);
          vd.SetData(f3, i);
        } else if (vs1.GetType() == FLOAT16) {
          half f1, f2, f3;
          bool msb1, msb2;
          f1 = vs1.GetHalfData(i);
          f2 = vs2.GetHalfData(i);
          msb1 = f1 < 0;
          msb2 = f2 < 0;

          f3 = (msb1 ^ msb2) ? -1*std::abs(f1) : std::abs(f1);
          vd.SetData(f3, i);
        } else {
          throw std::runtime_error("Unsupported vector instruction");
        }
      }
      context.register_map->WriteVreg(dest, vd, context);
    } else if (operand_type == VF) {
      VectorData vs1 = context.register_map->ReadVreg(src[0], context);
      VectorData vd(context);
      for (int i = context.csr->vstart; i < vd.GetVlen(); i++) {
        if (vs1.GetType() == FLOAT32) {
          float f1, f2, f3;
          bool msb1, msb2;
          f1 = vs1.GetFloatData(i);
          f2 = context.register_map->ReadFreg(src[1], context);
          msb1 = f1 < 0;
          msb2 = f2 < 0;

          f3 = (msb1 ^ msb2) ? -1*std::abs(f1) : std::abs(f1);
          vd.SetData(f3, i);
        } else {
          throw std::runtime_error("Unsupported vector instruction");
        }
      }
      context.register_map->WriteVreg(dest, vd, context);
    } else {
      throw std::runtime_error("Unsupported vector instruction");
    }
  } else {
    throw std::runtime_error("Unsupported vector instruction");
  }

  // Reset vstart;
  context.csr->vstart = 0;
}

void NdpInstruction::ExecuteBranch(Context& context) {
  if (opcode == J) {
    doBranch = true;
    return;
  }
  LogicalOp op;
  int64_t lhs = context.register_map->ReadXreg(src[0], context);
  int64_t rhs = 0;

  if (opcode == BEQZ || opcode == BEQ)
    op = LogicalOp::EQ;
  else if (opcode == BLTZ || opcode == BLT)
    op = LogicalOp::LT;
  else if (opcode == BGTZ || opcode == BGT)
    op = LogicalOp::GT;
  else if (opcode == BLEZ || opcode == BLE)
    op = LogicalOp::LE;
  else if (opcode == BGEZ || opcode == BGE)
    op = LogicalOp::GE;
  else if (opcode == BNEZ || opcode == BNE)
    op = LogicalOp::NE;
  else
    throw std::runtime_error("Unsupported branch instruction");

  if (opcode == BEQ || opcode == BLT || opcode == BGT || opcode == BLE ||
      opcode == BGE || opcode == BNE)
    rhs = context.register_map->ReadXreg(src[1], context);

  doBranch = logical_operate(op, lhs, rhs);
  return;
}

int NdpInstruction::IsBranch() {
  if (CheckBranchOp() && doBranch) {
    if (opcode != J) doBranch = false;
    return src[2];
  } else
    return -1;
}

MemoryMap* GetMemoryMap(Context& context, uint64_t addr) {
  if (MemoryMap::CheckScratchpad(addr))
    return context.scratchpad_map;
  else
    return context.memory_map;
}

VectorData MemoryMapLoad(Context& context, uint64_t addr) {
  VectorData vd(context);
  addr = MemoryMap::FormatAddr(addr);
  MemoryMap* map = GetMemoryMap(context, addr);
  vd = map->Load(addr);
  return vd;
}

void MemoryMapStore(Context& context, uint64_t addr, VectorData& vs) {
  addr = MemoryMap::FormatAddr(addr);
  MemoryMap* map = GetMemoryMap(context, addr);
  map->Store(addr, vs);
}

bool CheckMemoryMap(Context& context, uint64_t addr) {
  addr = MemoryMap::FormatAddr(addr);
  MemoryMap* map = GetMemoryMap(context, addr);
  return map->CheckAddr(addr);
}

}  // namespace NDPSim