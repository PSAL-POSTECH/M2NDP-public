#ifndef FUNCSIM_MEMORY_MAP_H_
#define FUNCSIM_MEMORY_MAP_H_

#include <robin_hood.h>

#include <bitset>
#include <cassert>
#include <string>

#include "common.h"
namespace NDPSim {

class MemoryMap {
 public:
  virtual ~MemoryMap() = default;
  static bool CheckScratchpad(uint64_t addr) { return addr >= SCRATCHPAD_BASE && addr < SCRATCHPAD_BASE + SCRATCHPAD_MAX_SIZE; }
  static uint64_t FormatAddr(uint64_t addr) { return addr & ~(PACKET_SIZE - 1); }

  MemoryMap() {}
  virtual bool Match(MemoryMap& other) = 0;
  virtual VectorData Load(uint64_t addr) = 0;
  virtual void Store(uint64_t addr, VectorData data) = 0;
  virtual bool CheckAddr(uint64_t addr) = 0;
  virtual void Reset() {}
  virtual void DumpMemory() {}
  void set_synthetic_memory(uint64_t base, uint64_t size) {
    m_use_synthetic_memory = true;
    m_synthetic_base_address = base;
    m_synthetic_memory_size = size;
  }
  bool is_synthetic_memory() { return m_use_synthetic_memory; }
 protected:
  bool m_use_synthetic_memory = false;
  uint64_t m_synthetic_base_address;
  uint64_t m_synthetic_memory_size;
};

class HashMemoryMap : public MemoryMap {
  public:
  HashMemoryMap() : MemoryMap() {}
  HashMemoryMap(std::string file_path);
  HashMemoryMap(uint64_t base, uint64_t size) : m_base(base), m_size(size){};
  virtual ~HashMemoryMap() override = default;
  virtual bool Match(MemoryMap& other) override;
  virtual VectorData Load(uint64_t addr) override;
  virtual void Store(uint64_t addr, VectorData data) override;
  virtual bool CheckAddr(uint64_t addr) override;
  virtual void Reset() override;
  void DumpMemory() override;
private:
  uint64_t m_size = 0;
  uint64_t m_base = 0; 
  robin_hood::unordered_map<uint64_t, VectorData> m_data_map;
};

class PointerMemoryMap : public MemoryMap {
 public:
  PointerMemoryMap() : MemoryMap() {}
  virtual ~PointerMemoryMap() override = default;
  virtual bool Match(MemoryMap& other) override;
  virtual VectorData Load(uint64_t addr) override;
  virtual void Store(uint64_t addr, VectorData data) override;
  virtual bool CheckAddr(uint64_t addr) override;
  virtual void Reset() override;
  void AllocateMemory(uint64_t addr, uint64_t size, DataType type, void* ptr);
  void FreeMemory(uint64_t addr);
 private:
  uint64_t GetBaseAddr(uint64_t addr);
  struct MemoryInfo {
    uint64_t size;
    DataType type;
    void* ptr;
  };

  std::map<uint64_t, MemoryInfo> m_ptr_map;
};
}
#endif  // FUNCSIM_MEMORY_MAP_H_