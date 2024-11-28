#ifndef NDP_INSTRUCTION_H_
#define NDP_INSTRUCTION_H_
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <half.hpp>
using namespace half_float;
namespace NDPSim {

enum Opcode {
  /*RISC-V stype opcodes*/
  /* Scalar instruction sets*/
  INVALID_INST,
  NOP,
  ADD,
  ADDI,
  SUB,
  MUL,
  MULI,
  SLL,
  SLLI,
  SRL,
  SRLI,
  DIV,
  REM,
  AND,
  ANDI,
  OR,
  ORI,
  CSRW,
  CSRWI,
  LI,
  LB,
  LW,
  LD,
  LBU,
  SB,
  SD,
  SW,
  /* float instruction sets*/
  FADD,
  FSUB,
  FDIV,
  FMUL,
  FMV,
  FLW,
  FSW,
  NUM_SCALAR_OPCODES,
  /* Vector instruction sets*/
  VSETVLI,
  VADD,
  VSUB,
  VRSUB,
  VMUL,
  VDIV,
  VSLL,
  VSRL,
  VSRA,

  VNADD,  // vector narrowing
  // TODO: Vector Narrowing Instructions vn[op]

  VMV,
  VFADD,
  VFSUB,
  VFRSUB,
  VFMUL,
  VFDIV,
  VFRDIV,
  VFMACC,
  VFMSUB,
  VFWMACC,
  VFNCVT,
  VFCVT,
  VFWCVT,
  VFWSUB,
  VFWMUL,
  VFREDOSUM,
  VWREDOSUM,
  VFREDOMAX,
  VWREDOMAX,
  VFMV,
  VFSQRT,

  VMIN,
  VMAX,

  VLE8,
  VSE8,
  VLE16,
  VSE16,
  VLE32,
  VSE32,
  VLE64,
  VSE64,
  VAMOADDEI8,
  VAMOADDEI16,             // Vector AMO ADD for 16-bit elements
  VAMOADDEI32,             // Vector AMO ADD for 32-bit elements
  VAMOADDEI64,             // Vector AMO ADD for 64-bit elements
  VAMOANDEI16,             // Vector AMO AND for 16-bit elements
  VAMOANDEI32,             // Vector AMO AND for 32-bit elements
  VAMOOREI16,              // Vector AMO OR for 16-bit elements
  VAMOOREI32,              // Vector AMO OR for 32-bit elements
  VAMOMAXEI16,             // Vector AMO MAX for 16-bit elements
  VAMOMAXEI32,             // Vector AMO MAX for 32-bit elements
  VAMOMINEI16,             // Vector AMO MIN for 16-bit elements
  VAMOMINEI32,             // Vector AMO MIN for 32-bit elements

  AMOADD,
  AMOMAX,
  FAMOADD,
  FAMOADDH,
  FAMOMAX,

  VID,
  VMSET,

  VMSEQ,  // Vector Mask if equal
  VMSNE,
  VMSLT,
  VMSGT,
  VMSLE,
  VMSGE,
  VMAND,
  VMOR,
  VCOMPRESS,  // Vectore Compress with VM
  VAND,       //
  VOR,        //

  VFIRST,

  NUM_VECTOR_OPCODES,

  VLSSEG,

  VSLIDE1DOWN,
  VSLIDEDOWN,
  VSLIDE1UP,
  VSLIDEUP,

  VREDSUM,
  VREDMAX,
  VREDMIN,
  VREDAND,
  VREDOR,
  VLUXEI32,
  VLUXEI64,
  VSUXEI32,
  VSUXEI64,
  /* Vector Floating-Point Sign Injection Instructions */
  VFSGNJ,
  VFSGNJN,
  VFSGNJX,
  /* Branch opcode */
  J,
  BEQ,
  BEQZ,
  BGT,
  BGTZ,
  BLT,
  BLTZ,
  BLE,
  BLEZ,
  BGE,
  BGEZ,
  BNE,
  BNEZ,
  /* tremendous opcode (SFP) */
  FEXP,
  VFEXP,
  /* Special opcode */
  TEST,
  EXIT
};

static std::map<std::string, Opcode> opcode_type_map = {
    {"add", ADD},
    {"addi", ADDI},
    {"sub", SUB},
    {"mul", MUL},
    {"muli", MULI},
    {"sll", SLL},
    {"slli", SLLI},
    {"srl", SRL},
    {"srli", SRLI},
    {"div", DIV},
    {"rem", REM},
    {"and", AND},
    {"andi", ANDI},
    {"or", OR},
    {"ori", ORI},
    {"csrw", CSRW},
    {"csrwi", CSRWI},
    {"li", LI},
    {"lb", LB},
    {"lw", LW},
    {"ld", LD},
    {"lbu", LBU},
    {"sb", SB},
    {"sd", SD},
    {"sw", SW},
    {"fadd", FADD},
    {"fsub", FSUB},
    {"fmul", FMUL},
    {"fdiv", FDIV},
    {"fmv", FMV},
    {"flw", FLW},
    {"fsw", FSW},
    {"vsetvli", VSETVLI},
    {"vadd", VADD},
    {"vsub", VSUB},
    {"vrsub", VRSUB},
    {"vmul", VMUL},
    {"vdiv", VDIV},
    {"vsll", VSLL},
    {"vsrl", VSRL},
    {"vsra", VSRA},
    {"vnadd", VNADD},
    {"vfrdiv", VFRDIV},
    {"vmv", VMV},
    {"vfadd", VFADD},
    {"vfsub", VFSUB},
    {"vfrsub", VFRSUB},
    {"vfmul", VFMUL},
    {"vfwmul", VFWMUL},
    {"vfdiv", VFDIV},
    {"vfmacc", VFMACC},
    {"vfmsub", VFMSUB},
    {"vfwmacc", VFWMACC},
    {"vfncvt", VFNCVT},
    {"vfcvt", VFCVT},
    {"vfwcvt", VFWCVT},
    {"vfwsub", VFWSUB},
    {"vfwmul", VFWMUL},
    {"vfredosum", VFREDOSUM},
    {"vwredosum", VWREDOSUM},
    {"vfredomax", VFREDOMAX},
    {"vwredomax", VWREDOMAX},
    {"vfmv", VFMV},
    {"vfsqrt", VFSQRT},
    {"vmin", VMIN},
    {"vmax", VMAX},
    {"vle8", VLE8},
    {"vse8", VSE8},
    {"vle16", VLE16},
    {"vse16", VSE16},
    {"vle32", VLE32},
    {"vse32", VSE32},
    {"vle64", VLE64},
    {"vse64", VSE64},
    {"vamoaddei8", VAMOADDEI8},
    {"vamoaddei16", VAMOADDEI16},
    {"vamoaddei32", VAMOADDEI32},
    {"vamoaddei64", VAMOADDEI64},
    {"vamoandei16", VAMOANDEI16},
    {"vamoandei32", VAMOANDEI32},
    {"vamoorei16", VAMOOREI16},
    {"vamoorei32", VAMOOREI32},
    {"vamomaxei16", VAMOMAXEI16},
    {"vamomaxei32", VAMOMAXEI32},
    {"vamominei16", VAMOMINEI16},
    {"vamominei32", VAMOMINEI32},
    {"amoadd", AMOADD},
    {"amomax", AMOMAX},
    {"famoadd", FAMOADD},
    {"famoaddh", FAMOADDH},
    {"famomax", FAMOMAX},
    {"vid", VID},
    {"vmset", VMSET},
    {"vmseq", VMSEQ},
    {"vmsne", VMSNE},
    {"vmslt", VMSLT},
    {"vmsgt", VMSGT},
    {"vmsle", VMSLE},
    {"vmsge", VMSGE},
    {"vmand", VMAND},
    {"vmor", VMOR},
    {"vcompress", VCOMPRESS},
    {"vand", VAND},
    {"vor", VOR},
    {"vfirst", VFIRST},
    {"vlsseg", VLSSEG},
    {"vslide1down", VSLIDE1DOWN},
    {"vslidedown", VSLIDEDOWN},
    {"vslide1up", VSLIDE1UP},
    {"vslideup", VSLIDEUP},
    {"vredsum", VREDSUM},
    {"vredmax", VREDMAX},
    {"vredmin", VREDMIN},
    {"vredand", VREDAND},
    {"vredor", VREDOR},
    {"vfsgnj", VFSGNJ},
    {"vfsgnjn", VFSGNJN},
    {"vfsgnjx", VFSGNJX},
    {"j", J},
    {"beq", BEQ},
    {"beqz", BEQZ},
    {"bgt", BGT},
    {"bgtz", BGTZ},
    {"blt", BLT},
    {"ble", BLE},
    {"bltz", BLTZ},
    {"blez", BLEZ},
    {"bge", BGE},
    {"bgez", BGEZ},
    {"bne", BNE},
    {"bnez", BNEZ},
    {"vluxei32", VLUXEI32},
    {"vluxei64", VLUXEI64},
    {"vsuxei32", VSUXEI32},
    {"vsuxei64", VSUXEI64},
    {"fexp", FEXP},
    {"vfexp", VFEXP},
    {"TEST", TEST},
    {"EXIT", EXIT}};

enum OperandType {
  NONE,   /*  none                         */
  V,      /* vector */
  WV,     /*  vector-vector  (2*SEW op SEW)      */
  VV,     /*  vector-vector                      */
  VI,     /*  vector-immdediate (imm[4:0])       */
  VX,     /*  vector-scalar (GPR x register rs1) */
  VF,     /*  vector-scalar (FP f register rs1)  */
  VS,     /*  vector-vector[0]                   */
  V_V,    /*  vector-vector (v[i] = v[i] op v[i])*/
  V_X,    /*  vector-scalar (v[i] = v[i] op rs1) */
  V_I,    /*   scalar to vector (v[i] = imm)     */
  V_F,    /*  scalar to vector (v[i] = f[rs1])   */
  F_S,    /*  scalar to float (f[rs1] = s[0])    */
  S_F,    /*  float to scalar (s[0] = f[rs1])    */
  XU_F_V, /* vector, float to unsigned int       */
  X_F_V,  /* vector, float to signed int         */
  F_XU_V, /* vector, unsigned int to float       */
  F_X_V,  /* vector, signed int to float         */
  F_F_V,  /* vector, float(SEW) to float (2*SEW) */
  F_F_W,  /* vector, float(2*SEW) to float (SEW) */
  XU_F_W, /* vector, float(2*SEW) to unsigned int */
  X_F_W,  /* vector, float(2*SEW) to signed int */
  F_XU_W, /* vector, unsigned int to float(SEW) */
  F_X_W,  /* vector, signed int to float(SEW)   */
  VM,     /* vector mask */
  M,      /* vfirst.m */
  MM,
  X_S,
  S_X,
  X_W,
  W_X,
  WX,
  D, /* double 64bit data */
  W /* word 32bit data */
};

static std::map<std::string, OperandType> operand_type_map = {
    {"v", V},         {"wv", WV},         {"vv", VV},       {"vi", VI},
    {"vx", VX},       {"vf", VF},         {"vs", VS},       {"v.v", V_V},
    {"v.x", V_X},     {"v.i", V_I},       {"v.f", V_F},     {"f.s", F_S},
    {"s.f", S_F},     {"xu.f.v", XU_F_V}, {"x.f.v", X_F_V}, {"f.xu.v", F_XU_V},
    {"f.x.v", F_X_V}, {"f.f.v", F_F_V},   {"f.f.w", F_F_W}, {"vm", VM},
    {"m", M},         {"mm", MM},         {"x.s", X_S},     {"s.x", S_X},
    {"x.w", X_W},     {"w.x", W_X},       {"wx", WX},       {"d", D},
    {"w", W}};

enum AluOpType {
  INT_ADD_OP,
  INT_MAX_OP,
  INT_MUL_OP,
  INT_MAD_OP,
  INT_DIV_OP,
  FP_ADD_OP,
  FP_MAX_OP,
  FP_MUL_OP,
  FP_MAD_OP,
  FP_DIV_OP,
  SFU_OP,
  INT_REDUCE_OP,
  FP_REDUCE_OP,
  LDST_OP,
  ADDRESS_OP,
  NUM_ALU_OP_TYPE
};
static const char* AluOpTypeString[] = {
    "INT_ADD_OP", "INT_MAX_OP",    "INT_MUL_OP",   "INT_MAD_OP", "INT_DIV_OP",
    "FP_ADD_OP",  "FP_MAX_OP",     "FP_MUL_OP",    "FP_MAD_OP",  "FP_DIV_OP",
    "SFU_OP",     "INT_REDUCE_OP", "FP_REDUCE_OP", "LDST_OP",    "ADDRESS_OP"};
struct Context;

enum ArithOp {
  ARITH_ADD = 0,
  ARITH_SUB,
  ARITH_MUL,
  ARITH_DIV,
  ARITH_MIN,
  ARITH_MAX,
  ARITH_REM,
  ARITH_MOD,
  ARITH_SHL,
  ARITH_SHR,
  ARITH_AND,
  ARITH_OR,
  ARITH_XOR
};

template <typename T>
T arith_op(ArithOp op, const T& lhs, const T& rhs) {
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
      return lhs % rhs;
    case ARITH_MOD:
      return lhs % rhs;
    case ARITH_SHL:
      return lhs << rhs;
    case ARITH_SHR:
      return lhs >> rhs;
    case ARITH_AND:
      return lhs & rhs;
    case ARITH_OR:
      return lhs | rhs;
    case ARITH_XOR:
      return lhs ^ rhs;
    default:
      return false;
  }
}
template <>
float arith_op<float>(ArithOp op, const float& lhs, const float& rhs);
template <>
half arith_op<half>(ArithOp op, const half& lhs, const half& rhs);
struct NdpInstruction {
  NdpInstruction() {}
  NdpInstruction(const NdpInstruction& other) {
    sInst = other.sInst;
    opcode = other.opcode;
    operand_type = other.operand_type;
    for (int i = 0; i < 5; i++) {
      src[i] = other.src[i];
    }
    dest = other.dest;
    for (int i = 0; i < 8; i++) {
      segDest[i] = other.segDest[i];
    }
    for (auto addr : other.addr_set) {
      addr_set.insert(addr);
    }
    segCnt = other.segCnt;
    arrive = other.arrive;
    depart = other.depart;
    last = other.last;
    double_reg = other.double_reg;
    doBranch = other.doBranch;
  }
  ~NdpInstruction() {
    addr_set.clear();
  }
  std::string sInst;
  Opcode opcode;
  OperandType operand_type;
  int64_t src[5] = {-1};
  int64_t dest;
  int64_t segDest[8];
  std::set<uint64_t> addr_set;
  int segCnt = 0;
  unsigned long long arrive;
  unsigned long long depart;
  bool last;
  bool double_reg;
  bool doBranch = false;
  bool CheckScalarOp();
  bool CheckFloatOp();
  bool CheckVectorOp();
  bool CheckNarrowingOp();
  bool CheckWideningOp();
  bool CheckAmoOp();
  bool CheckCsrOp();
  bool CheckVmaskOp();
  bool CheckSegOp();
  bool CheckLoadOp();
  bool CheckStoreOp();
  bool CheckBranchOp();
  bool CheckIndexedOp();
  bool CheckDestWrite();
  void Execute(Context& context);
  int ReturnSegCnt() { return segCnt; };
  int IsBranch();
  AluOpType GetAluOpType();
  ArithOp GetArithOp();

 private:
  void ExecuteScalar(Context& context);
  void ExecuteVector(Context& context);
  void ExecuteBranch(Context& context);
};
}  // namespace NDPSim
#endif