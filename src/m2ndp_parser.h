#ifndef FUNCSIM_INSTRUCTION_PARSER_H_
#define FUNCSIM_INSTRUCTION_PARSER_H_

#include "common.h"
namespace NDPSim {

int GetFilterId(std::vector<int> shape, int axis, uint64_t offset);
class M2NDPConfig;

class M2NDPParser {
 public:
  M2NDPParser();
  static void parse_ndp_kernel(int num_simd, std::string file_path,
                               NdpKernel* ndp_kernel, M2NDPConfig *config);
  static void parse_config(std::string file_path, M2NDPConfig* m2ndp_config);
  static KernelLaunchInfo* parse_kernel_launch(std::string line, int host_id);
 private:
  static void parse_to_const(M2NDPConfig* config, std::string config_dir,
                             std::string name, std::string value);
  static void decode_memory_mapping(M2NDPConfig* config, std::string addr);
};
}  // namespace NDPSim
#endif  // FUNCSIM_INSTRUCTION_PARSER_H_