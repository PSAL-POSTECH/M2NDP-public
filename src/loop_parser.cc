#include "loop_parser.h"

#include <fstream>
#include <sstream>
namespace NDPSim {

LoopParser::LoopParser(M2NDPConfig *m2ndp_config, std::string file_path) {
  kernels.resize(0);
  num_loop = 0;
  num_kernel = 0;
  config = m2ndp_config;
  num_ndp = config->get_num_ndp_units();
  ParseNDPKernels(file_path);
}

void LoopParser::ParseNDPKernels(std::string file_path) {

  std::ifstream ifs;
  ifs.open(file_path.c_str());

  std::string line;

  while(getline(ifs, line)) {
    if (line == "" || line[0] == '#')
      continue;

    // Todo : Precise String Management
    if (line.substr(0, 4) == "loop") {
      std::stringstream ss(line.substr(5));
      ss >> std::dec >> num_loop;
    } else if (line.substr(0, 10) == "num_kernel") {
      std::stringstream ss(line.substr(11));
      ss >> std::dec >> num_kernel;
    } else {
      // trace file path
      NdpKernel ndp_kernel;
      M2NDPParser::parse_ndp_kernel(num_ndp, line, &ndp_kernel, config);

      kernels.push_back(ndp_kernel);
    }
  }
  assert(kernels.size() == num_kernel);
}

std::vector<NdpKernel>* LoopParser::GetNDPKernels() {
  return &kernels;
}

int LoopParser::GetNumKernel() {
  return num_kernel;
}

int LoopParser::GetNumLoop() {
  return num_loop;
}
}