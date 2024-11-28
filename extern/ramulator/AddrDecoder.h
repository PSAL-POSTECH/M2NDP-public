#ifndef _ADDR_DECODER_H
#define _ADDR_DECODER_H

#include <vector>
#include <string>
#include <bitset>
#include <utility>  // std::pair
#include <math.h>
//#include "Memory.h"
#include "Config.h"

#include "HBM.h"
#include "PCM.h"
#include "GDDR5.h"
#include "HMC.h"
#include "DDR4.h"

namespace NDPSim {

template <typename T>
class AddrDecoder {
public:
  AddrDecoder(const Config& configs, T* spec, unsigned long m_id) : spec(spec) {
    // other memories only need address mask
    // Note that address mask must match the memory spec
    this->totalBits = 0;
    this->numBurstBits = 0;
    setAddrMask(spec->addr_mask.mask, this->addrMask, this->addrMask_low_high, 
                this->totalBits, spec->org_entry.count);
    this->totalAddress = pow(2, this->totalBits);

    // for GPGPUsim burst size must be 32
    assert(numBurstBits == 5);
    assert(burstSize == 32);
    assert(burstSize == spec->prefetch_size * (spec->channel_width / 8));
    this->m_id = m_id;
  }

  int calc_log2(int val) {
    int n = 0;
    while ((val >>= 1))
      n++;

    return n;
  }

  // You must specialize addr mask parser function if needed.
  // This is because each memory type has different type of memory hierarchy.
  // For example, while DDR4 has bank groups, DDR3 does not.
  void setAddrMask(const std::string& addrMaskString,
                   std::bitset<64> *addrMask,
                   std::pair<int, int> *addrMask_low_high,
                   unsigned long &totalBits,
                   const int *org_entry_count) {}
  void setAddrMask_low_high(int level, 
                            std::bitset<64> *addrMask,
                            std::pair<int, int> *addrMask_low_high) {
    int lowPos = 0;
    int highPos = 0;
    bool isLowSet = false;
    for (int i = 0; i < 64; ++i) {
      if (!isLowSet && addrMask[level].test(i)) { // if nth bit is set
        lowPos = i;
        isLowSet = true;
      }
      if (addrMask[level].test(i))
        highPos = i;
    }
    addrMask_low_high[level] = std::make_pair(lowPos, highPos);
  }

  // addr includes data bits
  unsigned long getLevelAddr(int level, const unsigned long addr,
                             const std::bitset<64> *addrMask,
                             const std::pair<int, int>* addrMask_low_high) {
    int lowPos = addrMask_low_high[level].first;
    int highPos = addrMask_low_high[level].second;

    unsigned long levelAddr = 0;
    int currentPos = 0;
    for (int i = lowPos; i <= highPos; ++i) {
      if (addrMask[level].test(i)) {
        unsigned long posBit = (unsigned long)1 << i;
        unsigned long val = posBit & addr;
        levelAddr |= (val >> (i - currentPos));
        currentPos++;
      }
    }
    return levelAddr;
  }

  std::vector<int> generateMemoryAddr(const unsigned long &addr, std::string type = "") {
    std::vector<int> addr_vec(int(T::Level::MAX));

    for (int level = 0; level < int(T::Level::MAX); ++level) {
      addr_vec[level] = getLevelAddr(level, addr, addrMask, addrMask_low_high);
      //assert(addr_vec[level] >= 0);
    }
    //assert(addr_vec[int(T::Level::Channel)] == 0);

    return addr_vec;
  }

  bool flat_address_space;
  bool bypassDRAMCache;
  bool isFullyAssociative;

  unsigned long burstSize;
  unsigned long pageSize;

  unsigned long numAssoc;
  unsigned long numSet;

  unsigned long numBurstBits;
  unsigned long numPageBits;
  unsigned long numSetBits;
  unsigned long numAssocBits;
  unsigned long numTagBits;

  unsigned long totalDRAMBits;
  unsigned long totalSCMBits;
  unsigned long totalBits;

  unsigned long totalDRAMAddress;
  unsigned long totalSCMAddress;
  unsigned long totalTagAddress;
  unsigned long totalAddress;

  const unsigned long ROW_SIZE = 2048;
  const unsigned long ROW_SIZE_BITS = 11;

private:
  T* spec;

  unsigned long m_id;

  // address masks for HMS
  std::bitset<64> dramAddrMask[int(T::Level::MAX)];
  std::pair<int, int> dramAddrMask_low_high[int(T::Level::MAX)];

  std::bitset<64> scmAddrMask[int(T::Level::MAX)];
  std::pair<int, int> scmAddrMask_low_high[int(T::Level::MAX)];

  // address masks for the normal memory type
  std::bitset<64> addrMask[int(T::Level::MAX)];
  std::pair<int, int> addrMask_low_high[int(T::Level::MAX)];

};
// template<>
// void AddrDecoder<HBM>::setAddrMask(const std::string& addrMaskString,
//                                    std::bitset<64>* addrMask,
//                                    std::pair<int, int>* addrMask_low_high,
//                                    unsigned long &totalBits,
//                                    const int *org_entry_count);
template<>
void AddrDecoder<PCM>::setAddrMask(const std::string& addrMaskString,
                                   std::bitset<64>* addrMask,
                                   std::pair<int, int>* addrMask_low_high,
                                   unsigned long &totalBits,
                                   const int *org_entry_count);
// TODO: need to implement specialized functions for GDDR5 in AddrDecoder.cc
template<>
void AddrDecoder<GDDR5>::setAddrMask(const std::string& addrMaskString,
                                     std::bitset<64>* addrMask,
                                     std::pair<int, int>* addrMask_low_high,
                                     unsigned long &totalBits,
                                     const int *org_entry_count);
// TODO: need to implement specialized functions for HMC in AddrDecoder.cc
template<>
void AddrDecoder<HMC>::setAddrMask(const std::string& addrMaskString,
                                   std::bitset<64>* addrMask,
                                   std::pair<int, int>* addrMask_low_high,
                                   unsigned long &totalBits,
                                   const int *org_entry_count);
template<>                                  
void AddrDecoder<DDR4>::setAddrMask(const std::string& addrMaskString,
                                   std::bitset<64>* addrMask,
                                   std::pair<int, int>* addrMask_low_high,
                                   unsigned long &totalBits,
                                   const int *org_entry_count);
} // end namespace

#endif
