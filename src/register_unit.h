#ifndef FUNCSIM_REGISTER_RENAMING_H_
#define FUNCSIM_REGISTER_RENAMING_H_

#include <robin_hood.h>
#include <list>
#include <iostream>
#include "common.h"
#include "ndp_instruction.h"
namespace NDPSim {

typedef int RegisterMapKey;
typedef robin_hood::unordered_map<int, int> RegisterBinding;
typedef robin_hood::unordered_map<RegisterMapKey, RegisterBinding> RegisterMap;

struct RegisterStats {
  enum RegStatEnum {
    X_READ,
    F_READ,
    V_READ,
    X_WRITE,
    F_WRITE,
    V_WRITE,
    REG_STAT_NUM
  };
  uint32_t register_stats[REG_STAT_NUM];

  RegisterStats operator+(const RegisterStats &other) {
    RegisterStats sum;
    for (int i = 0; i < REG_STAT_NUM; i++)
      sum.register_stats[i] = register_stats[i] + other.register_stats[i];
    return sum;
  } 
  RegisterStats &operator+=(const RegisterStats &other) {
    for (int i = 0; i < REG_STAT_NUM; i++)
      register_stats[i] += other.register_stats[i];
    return *this;
  } 
  RegisterStats() {
    for (int i = 0; i < REG_STAT_NUM; i++) register_stats[i] = 0;
  }
};

class RegisterUnit {
 public:
  RegisterUnit(int num_xreg, int num_freg, int num_vreg);
  bool RenameFull(InstColumn *inst_column);
  void RenamePush(InstColumn *inst_column, int packet_id);
  bool RenameEmpty();
  InstColumn *RenameTop();
  void RenamePop();

  /*Dependency Access*/
  bool CheckReady(int reg);
  void SetNotReady(int reg);
  void SetReady(int reg);

  /*Renaming*/
  std::deque<NdpInstruction> Convert(std::deque<NdpInstruction> input,
                                     int packet_id,
                                     RequestInfo* req);

  void FreeRegs(int packet_id);
  bool CheckDoubleReg(int reg);
  int GetAnother(int reg);

  /*Read-Write access*/
  bool CheckExistVreg(RegisterMapKey key, int reg);
  bool CheckExistRenameVreg(int vreg_id);
  int64_t ReadXreg(int xreg_id, Context& context);
  char ReadXreg(int xreg_id, int idx, Context& context);
  float ReadFreg(int freg_id, Context& context);
  VectorData ReadVreg(int vreg_id, Context& context);
  void WriteXreg(int xreg_id, int64_t data, Context& context);
  void WriteFreg(int freg_id, float data, Context& context);
  void WriteFreg(int freg_id, half data, Context& context);
  void WriteVreg(int vreg_id, VectorData data, Context& context);

  /*Register Status*/
  RegisterStats get_register_stats();
  void dump_current_state();

  /*Check rf type*/
  bool IsVreg(int reg_id);
  bool IsXreg(int reg_id);
  bool IsFreg(int reg_id);
  void DumpRegisterFile(int packet_id);

 private:
  int m_num_xregs;
  int m_num_fregs;
  int m_num_vregs;

  RegisterStats m_register_stats;

  int cvt_vlmul_status;

  std::vector<VectorData> m_reg_table;
  RegisterMap m_xreg_mapping;
  RegisterMap m_freg_mapping;
  RegisterMap m_vreg_mapping;
  RegisterBinding m_double_vreg;
  std::deque<int> m_free_xregs;
  std::deque<int> m_free_fregs;
  std::deque<int> m_free_vregs;

  std::vector<int64_t> m_xreg_table;
  std::vector<float> m_freg_table;
  std::vector<VectorData> m_vreg_table;

  std::deque<InstColumn*> m_after_rename;
  robin_hood::unordered_set<int> m_not_ready;

  bool CheckSpecialReg(int reg);
  // int GetVregIndex(int reg);
  int LookUp(RegisterMapKey key, int reg);
  int LookUpX(RegisterMapKey key, int reg);
  int LookUpF(RegisterMapKey key, int reg);
  int LookUpV(RegisterMapKey key, int reg);
  int DestRename(RegisterMapKey key, NdpInstruction inst);
  int DestRenameX(RegisterMapKey key, NdpInstruction inst);
  int DestRenameF(RegisterMapKey key, NdpInstruction inst);
  int DestRenameV(RegisterMapKey key, NdpInstruction inst);
  int DestRenameSegV(RegisterMapKey key, NdpInstruction inst, int index);
};
}
#endif  // FUNCSIM_REGISTER_RENAMING_H_