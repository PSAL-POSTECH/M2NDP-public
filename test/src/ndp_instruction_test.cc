#include "common.h"
#include "ndp_instruction.h"
#include "register_unit.h"
#include "gtest/gtest.h"

namespace NDPSim {
Context make_basic_context() {
  Context context;
  context.ndp_id = 0;
  context.csr = new CSR();
  context.loop_map = new std::map<int, int>();
  context.register_map = new RegisterUnit(32, 32, 32);
  context.request_info = new RequestInfo();
  context.last_inst = false;
  context.max_pc = 0;
  context.exit = false;
  return context;
}

TEST(NdpInstScalarAddTest, BasicAssertions) {
  NdpInstruction inst;
  inst.opcode = ADD;
  inst.src[0] = REG_PX_BASE;
  inst.src[1] = REG_PX_BASE + 1;
  inst.dest = REG_PX_BASE + 2;
  Context context = make_basic_context();
  context.register_map->WriteXreg(REG_PX_BASE, 1, context);
  context.register_map->WriteXreg(REG_PX_BASE + 1, 2, context);
  inst.Execute(context);
  ASSERT_EQ(context.register_map->ReadXreg(REG_PX_BASE + 2, context), 3);
}

TEST(NdpInstructionVectorAddTest, BasicAssertions) {
  NdpInstruction inst;
  inst.opcode = VADD;
  inst.operand_type = VV;
  inst.src[0] = REG_PV_BASE;
  inst.src[1] = REG_PV_BASE + 1;
  inst.dest = REG_PV_BASE + 2;
  Context context = make_basic_context();
  VectorData v1(context);
  VectorData v2(context);
  v1.SetType(INT32);
  v2.SetType(INT32);
  for(int i = 0; i < PACKET_SIZE / 4; i++) {
    v1.SetData(i, i);
    v2.SetData(-i, i);
  }
  context.register_map->WriteVreg(REG_PV_BASE, v1, context);
  context.register_map->WriteVreg(REG_PV_BASE + 1, v2, context);
  inst.Execute(context);

  VectorData v3 = context.register_map->ReadVreg(REG_PV_BASE + 2, context);
  for(int i = 0; i < 4; i++) {
    ASSERT_EQ(v3.GetIntData(i), 0);
  }
}

TEST(NdpInstructionVectorReduceSumTest, BasicAssertions) {
  NdpInstruction inst;
  inst.opcode = VREDSUM;
  inst.operand_type = VS;
  inst.src[0] = REG_PV_BASE;
  inst.src[1] = REG_PV_BASE + 1;
  inst.dest = REG_PV_BASE + 2;
  Context context = make_basic_context();
  VectorData v1(context);
  VectorData v2(context);
  v1.SetType(INT32);
  v2.SetType(INT32);
  int result = 0;
  for(int i = 0; i < PACKET_SIZE / 4; i++) {
    v1.SetData(i, i);
    v2.SetData(0, i);
    result += i;
  }
  context.register_map->WriteVreg(REG_PV_BASE, v1, context);
  context.register_map->WriteVreg(REG_PV_BASE + 1, v2, context);
  inst.Execute(context);
  VectorData v3 = context.register_map->ReadVreg(REG_PV_BASE + 2, context);
  ASSERT_EQ(v3.GetIntData(0), result);
}

TEST(NdpInstructionVectorExponentTest, BasicAssertions) {
  NdpInstruction inst;
  inst.opcode = VFEXP;
  inst.operand_type = V;
  inst.src[0] = REG_PV_BASE;
  inst.dest = REG_PV_BASE + 1;
  Context context = make_basic_context();
  VectorData v1(context);
  VectorData v2(context);
  v1.SetType(FLOAT32);
  v2.SetType(FLOAT32);
  for(int i = 0; i < PACKET_SIZE / 4; i++) {
    v1.SetData(i, i);
  }
  v2 = v2.Exp(v1);
  context.register_map->WriteVreg(REG_PV_BASE, v2, context);
  inst.Execute(context);

  VectorData v3 = context.register_map->ReadVreg(REG_PV_BASE + 2, context);
  for(int i = 0; i < 4; i++) {
    ASSERT_EQ(v3.GetIntData(i), std::exp(i));
  }
}
}