#ifndef FUNCSIM_INSTRUCTION_LOOP_PARSER_H_
#define FUNCSIM_INSTRUCTION_LOOP_PARSER_H_

#include "m2ndp_parser.h"
#include "m2ndp_config.h"
namespace NDPSim {

class LoopParser {
public:
  LoopParser(M2NDPConfig *config, std::string file_path);
  std::vector<NdpKernel>* GetNDPKernels();
  int GetNumKernel();
  int GetNumLoop();

private:
  
  std::vector<NdpKernel> kernels;
  M2NDPConfig *config;

  int num_loop;
  int num_kernel;
  int num_ndp;

  void ParseNDPKernels(std::string file_path);
};

}

#endif  // FUNCSIM_INSTRUCTION_LOOP_PARSER_H_
