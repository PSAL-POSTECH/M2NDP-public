#include "AddrDecoder.h"

namespace NDPSim {

template<>
void AddrDecoder<HMC>::setAddrMask(const std::string& addrMaskString,
                                   std::bitset<64>* addrMask,
                                   std::pair<int, int>* addrMask_low_high,
                                   unsigned long &totalBits,
                                   const int *org_entry_count) {
  assert(false && "Need to implement HMC's setAddrMask function");
}
template<>
void AddrDecoder<GDDR5>::setAddrMask(const std::string& addrMaskString,
                                   std::bitset<64>* addrMask,
                                   std::pair<int, int>* addrMask_low_high,
                                   unsigned long &totalBits,
                                   const int *org_entry_count) {
  assert(false && "Need to implement GDDR5's setAddrMask function");
}
// template<>
// void AddrDecoder<HBM>::setAddrMask(const std::string& addrMaskString,
//                                    std::bitset<64>* addrMask,
//                                    std::pair<int, int>* addrMask_low_high,
//                                    unsigned long &totalBits,
//                                    const int *org_entry_count) {
//   std::cout << addrMaskString << std::endl;
//   /*
//   assert((addrMaskString.length() - ((64 / 8) - 1)) == 64
//          && "wrong dram address mask format");
//   */
//   // init all bits (make all bits have zero)
//   for (int i = 0; i < int(HBM::Level::MAX); ++i)
//     addrMask[i].reset();

//   // set bitmask
//   int numSeperator = 0;
//   for (auto it = addrMaskString.crbegin(); it != addrMaskString.crend(); ++it) {
//     int pos = std::distance(addrMaskString.crbegin(), it);
//     pos -= numSeperator;
//     const char& level = *it;
//     switch (level) {
//       case '.': // seperator for each 8 bits
//         ++numSeperator;
//         break;
//       case '0': // unused address bit
//         break;
//       case 'S': // ignore data bit
//         numBurstBits++;
//         totalBits++;
//         break;
//       case 'C':
//         addrMask[int(HBM::Level::Column)].set(pos, 1);
//         totalBits++;
//         break;
//       case 'B':
//         addrMask[int(HBM::Level::Bank)].set(pos, 1);
//         totalBits++;
//         break;
//       case 'G':
//         addrMask[int(HBM::Level::BankGroup)].set(pos, 1);
//         totalBits++;
//         break;
//       case 'H':
//         addrMask[int(HBM::Level::Rank)].set(pos, 1);
//         totalBits++;
//         break;
//       case 'R':
//         addrMask[int(HBM::Level::Row)].set(pos, 1);
//         totalBits++;
//         break;
//       case 'K':
//         addrMask[int(HBM::Level::Channel)].set(pos, 1);
//         totalBits++;
//         break;
//       default:
//         assert(false && "Do not support this character");
//     }
//   }
//   burstSize = 1 << numBurstBits;
//   assert(burstSize == 32);
//   for (int level = int(HBM::Level::Channel); level < int(HBM::Level::MAX); ++level) {
//     if (level == int(HBM::Level::Column)) {
//       assert(calc_log2(org_entry_count[level]) - 1 == addrMask[level].count());
//     } else {
//       assert(calc_log2(org_entry_count[level]) == addrMask[level].count());
//     }
//     setAddrMask_low_high(level, addrMask, addrMask_low_high);
//   }
// }

template<>
void AddrDecoder<PCM>::setAddrMask(const std::string& addrMaskString,
                                   std::bitset<64>* addrMask,
                                   std::pair<int, int>* addrMask_low_high,
                                   unsigned long &totalBits,
                                   const int *org_entry_count) {
  std::cout << addrMaskString << std::endl;
  assert((addrMaskString.length() - ((64 / 8) - 1)) == 64
         && "wrong dram address mask format");
  // init all bits (make all bits have zero)
  for (int i = 0; i < int(PCM::Level::MAX); ++i)
    addrMask[i].reset();

  // set bitmask
  int numSeperator = 0;
  for (auto it = addrMaskString.crbegin(); it != addrMaskString.crend(); ++it) {
    int pos = std::distance(addrMaskString.crbegin(), it);
    pos -= numSeperator;
    const char& level = *it;
    switch (level) {
      case '.': // seperator for each 8 bits
        ++numSeperator;
        break;
      case '0': // unused address bit
        break;
      case 'S': // ignore data bit
        numBurstBits++;
        totalBits++;
        break;
      case 'C':
        addrMask[int(PCM::Level::Column)].set(pos, 1);
        totalBits++;
        break;
      case 'B':
        addrMask[int(PCM::Level::Bank)].set(pos, 1);
        totalBits++;
        break;
      case 'G':
        addrMask[int(PCM::Level::BankGroup)].set(pos, 1);
        totalBits++;
        break;
      case 'H':
        addrMask[int(PCM::Level::Rank)].set(pos, 1);
        totalBits++;
        break;
      case 'R':
        addrMask[int(PCM::Level::Row)].set(pos, 1);
        totalBits++;
        break;
      default:
        assert(false && "Do not support this character");
    }
  }
  burstSize = 1 << numBurstBits;
  assert(burstSize == 32);
  for (int level = int(PCM::Level::Channel); level < int(PCM::Level::MAX); ++level) {
    if (level == int(PCM::Level::Column)) {
      assert(calc_log2(org_entry_count[level]) - 1 == addrMask[level].count());
    } else {
      assert(calc_log2(org_entry_count[level]) == addrMask[level].count());
    }
    setAddrMask_low_high(level, addrMask, addrMask_low_high);
  }
}

} // end namespace
