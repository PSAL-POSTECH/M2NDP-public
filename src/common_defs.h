#ifndef COMMON_DEFS_H
#define COMMON_DEFS_H
#include <cstdint>
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#ifndef MEM_ACCESS_SIZE
#define MEM_ACCESS_SIZE 32
#endif

namespace NDPSim {
const int SECTOR_CHUNCK_SIZE = 4;
const int MAX_MEMORY_ACCESS_SIZE = MEM_ACCESS_SIZE * SECTOR_CHUNCK_SIZE;
const int PACKET_SIZE = MEM_ACCESS_SIZE;
const int CXL_COMMAND_SIZE = 64;
const int MAX_NR_INT_ARG = CXL_COMMAND_SIZE - 24 / 8;
const int MAX_NR_FLOAT_ARG = 3;
const int MAX_VLMUL = 2;

static const uint64_t SCRATCHPAD_BASE     = 0x1000000000000000;
static const uint64_t SCRATCHPAD_MAX_SIZE = 0x0100000000000000;
static const uint64_t SCRATCHPAD_OFFSET   = 0x000001000000000;
static const uint64_t DRAM_TLB_BASE =  0x100000000000; //0x0100000000000000;
static const uint64_t KERNEL_LAUNCH_ADDR = 0x0010000000000000;
static const uint64_t NDP_FUNCTION_OFFSET = 0x0000000000010000;

static const int CONST_SRC = 5000;

// General purpose registers
static const int REG_X_BASE =
    100000;  // architectural scalar register base  ex) r0 -> 100, r1 -> 101...
static const int REG_F_BASE =
    150000;  // architectural scalar register base  ex) f0 -> 150, f1 -> 151...
static const int REG_V_BASE =
    200000;  // architectural vector register base  ex) v0 -> 200, v1 -> 201...
static const int REG_PX_BASE =
    300000;  // physical scalar register base  ex) p0 -> 300, p1 -> 301...
static const int REG_PF_BASE =
    350000;  // physical scalar register base  ex) pf0 -> 350, pf1 -> 351...
static const int REG_PV_BASE =
    400000;  // physical vector register base  ex) pv0 -> 400, pv1 -> 401...
// Special registers
static const int SPCIAL_REG_START = 1500000;
static const int REG_NDP_ID = SPCIAL_REG_START + 1;
static const int REG_REQUEST_OFFSET = SPCIAL_REG_START + 2;
static const int REG_REQUEST_DATA = SPCIAL_REG_START + 3;
static const int REG_ADDR = SPCIAL_REG_START + 4;
static const int REG_VSTART = SPCIAL_REG_START + 5;
static const int REG_X0 = SPCIAL_REG_START + 6;

static const int SPECIAL_IMMEDIATE_START = 1600000;
static const int IMM_E8 = SPECIAL_IMMEDIATE_START + 1;  /* SEW = 8 bit  */
static const int IMM_E16 = SPECIAL_IMMEDIATE_START + 2; /* SEW = 16 bit */
static const int IMM_E32 = SPECIAL_IMMEDIATE_START + 3; /* SEW = 32 bit */
static const int IMM_E64 = SPECIAL_IMMEDIATE_START + 4; /* SEW = 64 bit */
static const int IMM_MF8 = SPECIAL_IMMEDIATE_START + 5; /* LMUL = 1/8   */
static const int IMM_MF4 = SPECIAL_IMMEDIATE_START + 6; /* LMUL = 1/4   */
static const int IMM_MF2 = SPECIAL_IMMEDIATE_START + 7; /* LMUL = 1/2   */
static const int IMM_M1 = SPECIAL_IMMEDIATE_START + 8;  /* LMUL = 1     */
static const int IMM_M2 = SPECIAL_IMMEDIATE_START + 9;  /* LMUL = 2     */
static const int IMM_M4 = SPECIAL_IMMEDIATE_START + 10; /*  = 4     */
static const int IMM_M8 = SPECIAL_IMMEDIATE_START + 11; /* LMUL = 8     */
static const int IMM_TA = SPECIAL_IMMEDIATE_START + 12; /* TA = tail-agnostic */
static const int IMM_TU =
    SPECIAL_IMMEDIATE_START + 13; /* TU = tail-uncdisurbed    */
static const int IMM_MA = SPECIAL_IMMEDIATE_START + 14; /* MA = mask-agnostic */
static const int IMM_MU =
    SPECIAL_IMMEDIATE_START + 15; /* MU = mask-uncdisurbed    */
}  // namespace NDPSim
#endif