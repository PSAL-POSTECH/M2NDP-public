#include "memory_map.h"

#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>

#define EPS 0.000001f
namespace NDPSim {

HashMemoryMap::HashMemoryMap(std::string file_path) {
  m_base = 0;
  m_size = UINT64_MAX;
  std::ifstream ifs(file_path);
  if (!ifs.good()) {
    spdlog::error("Memory Map file not found: {}", file_path);
    exit(1);
  }
  std::string line;
  bool meta_read = false;
  DataType meta_type;

  int max_data_size = 0;
  int temp_size = 0;
  bool base_pushed = false;

  std::queue<uint64_t> base_q;

  int dbg = 0;

  while (!ifs.eof()) {
    getline(ifs, line);
    if (line.find("_META_") != std::string::npos) {
      meta_read = true;
      continue;
    }
    if (line.find("_DATA_") != std::string::npos) {
      temp_size = 0;
      base_pushed = false;
      meta_read = false;
      continue;
    }
    if (line.length() == 0) {
      continue;
    }
    if (meta_read) {
      // Meta Data read
      std::stringstream ss(line);
      std::string tmp;
      ss >> tmp;
      if (tmp == "float16") {
        meta_type = DataType::FLOAT16;
      } else if (tmp == "float32") {
        meta_type = DataType::FLOAT32;
      } else if (tmp == "int16") {
        meta_type = DataType::INT16;
      } else if (tmp == "int32") {
        meta_type = DataType::INT32;
      } else if (tmp == "char8") {
        meta_type = DataType::CHAR8;
      } else if (tmp == "int64") {
        meta_type = DataType::INT64;
      } else if (tmp == "uint8") {
        meta_type = DataType::UINT8;
      } else if (tmp == "bool8") {
        meta_type = DataType::BOOL;
      }
    } else {
      // Data read
      // spdlog::info(line);
      uint64_t addr_base;
      std::stringstream ss(line);
      ss >> std::hex >> addr_base;
      assert(addr_base != 0);
      assert(addr_base % PACKET_SIZE == 0);
      if (!base_pushed) {
        base_q.push(addr_base);
        base_pushed = true;
      }
      VectorData input_data64(64, 1);
      VectorData input_data32(32, 1);
      VectorData input_data16(16, 1);
      VectorData input_data8(8, 1);
      input_data32.SetType(meta_type);
      input_data8.SetType(meta_type);

      /* To read decimal values from memory map*/
      ss >> std::dec;
      for (int iter = 0; iter < PACKET_SIZE / 4; iter++) {
        if (meta_type == DataType::FLOAT16) {
          half fp16_0, fp16_1;
          std::string str_fp16_0, str_fp16_1;
          ss >> str_fp16_0 >> str_fp16_1;
          fp16_0 = std::stof(str_fp16_0);
          fp16_1 = std::stof(str_fp16_1);
          input_data16.SetData(fp16_0, iter * 2);
          input_data16.SetData(fp16_1, iter * 2 + 1);
        } else if (meta_type == DataType::FLOAT32) {
          float fp32;
          ss >> fp32;
          input_data32.SetData(fp32, iter);
        } else if (meta_type == DataType::INT32) {
          int32_t int32;
          ss >> int32;
          input_data32.SetData(int32, iter);
        } else if (meta_type == DataType::INT64) {
          if (iter >= (PACKET_SIZE / 8)) continue;
          int64_t int64;
          ss >> int64;
          input_data64.SetData(int64, iter);
        } else if (meta_type == DataType::CHAR8) {
          std::string str_char8_0, str_char8_1, str_char8_2, str_char8_3;
          ss >> str_char8_0 >> str_char8_1 >> str_char8_2 >> str_char8_3;
          int char8_0 = std::stoi(str_char8_0);
          int char8_1 = std::stoi(str_char8_1);
          int char8_2 = std::stoi(str_char8_2);
          int char8_3 = std::stoi(str_char8_3);
          input_data8.SetData(static_cast<char>(std::stoi(str_char8_0)),
                              iter * 4);
          input_data8.SetData(static_cast<char>(std::stoi(str_char8_1)),
                              iter * 4 + 1);
          input_data8.SetData(static_cast<char>(std::stoi(str_char8_2)),
                              iter * 4 + 2);
          input_data8.SetData(static_cast<char>(std::stoi(str_char8_3)),
                              iter * 4 + 3);
        } else if (meta_type == DataType::UINT8) {
          std::string str_char8_0, str_char8_1, str_char8_2, str_char8_3;
          ss >> str_char8_0 >> str_char8_1 >> str_char8_2 >> str_char8_3;
          uint8_t char8_0 = std::stoi(str_char8_0);
          uint8_t char8_1 = std::stoi(str_char8_1);
          uint8_t char8_2 = std::stoi(str_char8_2);
          uint8_t char8_3 = std::stoi(str_char8_3);
          input_data8.SetData(static_cast<uint8_t>(std::stoi(str_char8_0)),
                              iter * 4);
          input_data8.SetData(static_cast<uint8_t>(std::stoi(str_char8_1)),
                              iter * 4 + 1);
          input_data8.SetData(static_cast<uint8_t>(std::stoi(str_char8_2)),
                              iter * 4 + 2);
          input_data8.SetData(static_cast<uint8_t>(std::stoi(str_char8_3)),
                              iter * 4 + 3);
        } else if (meta_type == DataType::BOOL) {
          std::string str_char8_0, str_char8_1, str_char8_2, str_char8_3;
          ss >> str_char8_0 >> str_char8_1 >> str_char8_2 >> str_char8_3;
          uint8_t char8_0 = std::stoi(str_char8_0);
          uint8_t char8_1 = std::stoi(str_char8_1);
          uint8_t char8_2 = std::stoi(str_char8_2);
          uint8_t char8_3 = std::stoi(str_char8_3);
          input_data8.SetData(static_cast<bool>(std::stoi(str_char8_0)),
                              iter * 4);
          input_data8.SetData(static_cast<bool>(std::stoi(str_char8_1)),
                              iter * 4 + 1);
          input_data8.SetData(static_cast<bool>(std::stoi(str_char8_2)),
                              iter * 4 + 2);
          input_data8.SetData(static_cast<bool>(std::stoi(str_char8_3)),
                              iter * 4 + 3);
        }
      }
      assert(m_data_map.find(addr_base) == m_data_map.end());
      if (meta_type == CHAR8 || meta_type == UINT8 || meta_type == BOOL)
        m_data_map[addr_base] = input_data8;
      else if (meta_type == FLOAT16 || meta_type == INT16)
        m_data_map[addr_base] = input_data16;
      else if (meta_type == FLOAT32 || meta_type == INT32)
        m_data_map[addr_base] = input_data32;
      else if (meta_type == INT64)
        m_data_map[addr_base] = input_data64;
      else {
        spdlog::error("Unknown data type", file_path);
        exit(1);
      }
    }
  }
}

bool HashMemoryMap::Match(MemoryMap& other2) {
  HashMemoryMap& other = dynamic_cast<HashMemoryMap&>(other2);
  bool result = true;
  for (auto& [key, val] : m_data_map) {
    if (other.m_data_map.find(key) == other.m_data_map.end()) {
      spdlog::error("Key miss match Addr {:x}", key);
      result = false;
      break;
    }
    VectorData other_val = other.m_data_map[key];
    if (val.GetType() == FLOAT16) {
      if (other_val.GetType() != FLOAT16) {
        spdlog::error("Type miss match Addr {:x} Target {} : Real {}", key,
                      (int)val.GetType(), (int)other_val.GetType());
        result = false;
        break;
      }
      for (int i = 0; i < PACKET_ENTRIES * 2; i++) {
        double ratio = (other_val.GetHalfData(i) - val.GetHalfData(i)) /
                       (val.GetHalfData(i) + EPS);
        if (other_val.GetHalfData(i) == val.GetHalfData(i)) {
          ratio = 0;
        }
        bool check_data = ratio >= -0.01 && ratio <= 0.01;
        result = result && check_data;
        if (!check_data) {
          spdlog::debug("{:x} : Data) {} Ans) {} diff : {}", key,
                        float(other_val.GetHalfData(i)), float(val.GetHalfData(i)),
                        float(other_val.GetHalfData(i) - val.GetHalfData(i)));
        }
      }
    } else if (val.GetType() == FLOAT32) {
      if (other_val.GetType() != FLOAT32) {
        spdlog::error("Type miss match Addr {:x} Target {} : Real {}", key,
                      (int)val.GetType(), (int)other_val.GetType());
        result = false;
        break;
      }
      for (int i = 0; i < PACKET_ENTRIES; i++) {
        double ratio = (other_val.GetFloatData(i) - val.GetFloatData(i)) /
                       (val.GetFloatData(i) + EPS);
        if (other_val.GetFloatData(i) == val.GetFloatData(i)) {
          ratio = 0;
        }
        bool check_data = ratio >= -0.01 && ratio <= 0.01;
        result = result && check_data;
        if (!check_data) {
          spdlog::debug("{:x} : Data) {} Ans) {} diff : {}", key,
                        other_val.GetFloatData(i), val.GetFloatData(i),
                        other_val.GetFloatData(i) - val.GetFloatData(i));
        }
      }
    } else if (val.GetType() == INT32) {
      if (other_val.GetType() != INT32) {
        spdlog::error("Type miss match Addr {:x} Target {} : Real {}", key,
                      (int)val.GetType(), (int)other_val.GetType());
        result = false;
        break;
      }
      for (int i = 0; i < PACKET_ENTRIES; i++) {
        bool check_data = other_val.GetIntData(i) == val.GetIntData(i);
        result = result && check_data;

        if (!check_data) {
          spdlog::debug("{:x} : Data) {} Ans) {} diff : {}", key,
                        other_val.GetIntData(i), val.GetIntData(i),
                        other_val.GetIntData(i) - val.GetIntData(i));
        }
      }
    } else if (val.GetType() == INT64) {
      if (other_val.GetType() != INT64) {
        spdlog::error("Type miss match Addr {:x} Target {} : Real {}", key,
                      (int)val.GetType(), (int)other_val.GetType());
        result = false;
        break;
      }
      for (int i = 0; i < PACKET_ENTRIES / 2; i++) {
        if (val.GetLongData(i) == -1) continue;
        bool check_data = other_val.GetLongData(i) == val.GetLongData(i);
        result = result && check_data;

        if (!check_data) {
          spdlog::debug("{:x} : Data) {} Ans) {} diff : {}", key,
                     other_val.GetLongData(i), val.GetLongData(i),
                     other_val.GetLongData(i) - val.GetLongData(i));
        }
      }
    } else if (val.GetType() == VMASK) {
      if (other_val.GetType() != UINT8 || other_val.GetType() != VMASK) {
        spdlog::error("Type miss match Addr {:x} Target {} : Real {}", key,
                      (int)val.GetType(), (int)other_val.GetType());
        result = false;
        break;
      }
    } else if (val.GetType() == UINT8) {
      if (other_val.GetType() != UINT8) {
        spdlog::error("Type miss match Addr {:x} Target {} : Real {}", key,
                      (int)val.GetType(), (int)other_val.GetType());
        result = false;
        break;
      }
      for (int i = 0; i < PACKET_ENTRIES * 4; i++) {
        bool check_data = other_val.GetU8Data(i) == val.GetU8Data(i);
        result = result && check_data;

        if (!check_data) {
          spdlog::debug("{:x} : Data) {:b} Ans) {:b} diff : {}", key,
                        other_val.GetU8Data(i),
                        val.GetU8Data(i),
                        (other_val.GetU8Data(i) - val.GetU8Data(i)));
        }
      }
    } else if (val.GetType() == BOOL) {
      if (other_val.GetType() != BOOL) {
        spdlog::error("Type miss match Addr {:x} Target {} : Real {}", key,
                      (int)val.GetType(), (int)other_val.GetType());
        result = false;
        break;
      }
      for (int i = 0; i < PACKET_ENTRIES * 4; i++) {
        bool check_data = other_val.GetBoolData(i) == val.GetBoolData(i);
        result = result && check_data;

        if (!check_data) {
          spdlog::debug("{:x} : Data) {} Ans) {} diff : {}", key,
                         (other_val.GetBoolData(i) ? "True" : "False"),
                         (val.GetBoolData(i) ? "True" : "False"),
                         other_val.GetBoolData(i) - val.GetBoolData(i));
        }
      }
    }
  }
  if (!result) {
    spdlog::trace("This Memory map");
    DumpMemory();
    spdlog::trace("Target Memory Map");
    other.DumpMemory();
  }
  return result;
}

VectorData HashMemoryMap::Load(uint64_t addr) {
  assert(addr >= m_base && addr < m_base + m_size);
  if (m_use_synthetic_memory) {
    if (addr >= m_synthetic_base_address &&
        addr < m_synthetic_base_address + m_synthetic_memory_size) {
      uint32_t temp = addr & 0xfffff;
      VectorData data(32, 1);
      data.SetData((float)temp, 0);
      return data;
    }
  }
  if (m_data_map.find(addr) == m_data_map.end())
    throw std::runtime_error("HashMemoryMap::Load failed");

  if (m_data_map[addr].GetDoubleReg())
    throw std::runtime_error("HashMemoryMap::Double reg Stored!");
  return m_data_map[addr];
}

void HashMemoryMap::Store(uint64_t addr, VectorData data) {
  assert(addr >= m_base && addr < m_base + m_size);
  if (m_data_map[addr].GetDoubleReg())
    throw std::runtime_error("HashMemoryMap::Double reg Stored!");
  m_data_map[addr] = data;
}

bool HashMemoryMap::CheckAddr(uint64_t addr) {
  if (m_data_map.find(addr) == m_data_map.end()) {
    return false;
  }
  return true;
}

void HashMemoryMap::Reset() {
  m_data_map.clear();
}

void HashMemoryMap::DumpMemory() {
  std::map<uint64_t, VectorData> dump_map;
  for (auto& [key, val] : m_data_map) {
    dump_map[key] = val;
  }

  for (auto& [key, val] : dump_map) {
    spdlog::trace("{:x}: {}", key, val.toString());
  }
}

bool PointerMemoryMap::Match(MemoryMap& other2) {
  PointerMemoryMap& other = dynamic_cast<PointerMemoryMap&>(other2);
  bool result = true;
  for (auto& [key, val] : m_ptr_map) {
    result = result && other.CheckAddr(key);
    result = result && val.size == other.m_ptr_map[key].size;
    result = result && val.type == other.m_ptr_map[key].type;
    if (!result) {
      spdlog::error("Memory Map Mismatch This  Addr {:x} Size {} Type {}", key,
                    val.size, val.type);
      spdlog::error("Memory Map Mismatch Other Addr {:x} Size {} Type {}", key,
                    other.m_ptr_map[key].size, other.m_ptr_map[key].type);
      return false;
    }
    int loop_count = 0;
    if (val.type == FLOAT32 || val.type == INT32)
      loop_count = val.size / sizeof(float);
    else if (val.type == FLOAT16 || val.type == INT16)
      loop_count = val.size / sizeof(half);
    else if (val.type == INT64)
      loop_count = val.size / sizeof(int64_t);
    else if (val.type == CHAR8)
      loop_count = val.size / sizeof(char);
    else {
      spdlog::error("Invalid Pointer Type");
      exit(1);
    }
    for (int i = 0; i < loop_count; i++) {
      switch (val.type) {
        case FLOAT32: {
          float* ptr = (float*)val.ptr;
          float* ptr2 = (float*)other.m_ptr_map[key].ptr;
          result = result && ptr[i] == ptr2[i];
        }
        case FLOAT16: {
          half* ptr = (half*)val.ptr;
          half* ptr2 = (half*)other.m_ptr_map[key].ptr;
          result = result && ptr[i] == ptr2[i];
        }
        case INT64: {
          int64_t* ptr = (int64_t*)val.ptr;
          int64_t* ptr2 = (int64_t*)other.m_ptr_map[key].ptr;
          result = result && ptr[i] == ptr2[i];
        }
        case INT32: {
          int32_t* ptr = (int32_t*)val.ptr;
          int32_t* ptr2 = (int32_t*)other.m_ptr_map[key].ptr;
          result = result && ptr[i] == ptr2[i];
        }
        case INT16: {
          int16_t* ptr = (int16_t*)val.ptr;
          int16_t* ptr2 = (int16_t*)other.m_ptr_map[key].ptr;
          result = result && ptr[i] == ptr2[i];
        }
        case CHAR8: {
          char* ptr = (char*)val.ptr;
          char* ptr2 = (char*)other.m_ptr_map[key].ptr;

          result = result && ptr[i] == ptr2[i];
        }
        default:
          spdlog::error("Invalid Pointer Type");
          exit(1);
      }
      if (!result) {
        spdlog::error("Value Mismatch Addr {:x} Type {} Index {}", key,
                      val.type, i);
        return false;
      }
    }
  }

  return result;
}

VectorData PointerMemoryMap::Load(uint64_t addr) {
  uint64_t base = GetBaseAddr(addr);
  uint64_t offset = addr - base;
  MemoryInfo info = m_ptr_map[base];
  VectorData data;
  spdlog::trace(
      "Load() info type form pointer map load {}, base: {:x}, offset: {}",
      info.type, base, offset);
  data.SetType(info.type);
  switch (info.type) {
    case FLOAT32: {
      float* ptr = (float*)info.ptr;
      offset = offset / sizeof(float);
      for (int i = 0; i < PACKET_SIZE / sizeof(float); i++) {
        data.SetData(ptr[offset + i], i);
      }
      data.SetVlen((uint32_t)(PACKET_SIZE / sizeof(float)));
      break;
    }
    case FLOAT16: {
      half* ptr = (half*)info.ptr;
      offset = offset / sizeof(half);
      for (int i = 0; i < PACKET_SIZE / sizeof(half); i++) {
        data.SetData(ptr[offset + i], i);
      }
      data.SetVlen((uint32_t)(PACKET_SIZE / sizeof(half)));
      break;
    }
    case INT64: {
      int64_t* ptr = (int64_t*)info.ptr;
      offset = offset / sizeof(int64_t);
      for (int i = 0; i < PACKET_SIZE / sizeof(int64_t); i++) {
        data.SetData(ptr[offset + i], i);
      }
      data.SetVlen((uint32_t)(PACKET_SIZE / sizeof(int64_t)));
      break;
    }
    case INT32: {
      int32_t* ptr = (int32_t*)info.ptr;
      offset = offset / sizeof(int32_t);
      for (int i = 0; i < PACKET_SIZE / sizeof(int32_t); i++) {
        data.SetData(ptr[offset + i], i);
      }
      data.SetVlen((uint32_t)(PACKET_SIZE / sizeof(int32_t)));
      break;
    }
    case INT16: {
      int16_t* ptr = (int16_t*)info.ptr;
      offset = offset / sizeof(int16_t);
      for (int i = 0; i < PACKET_SIZE / sizeof(int16_t); i++) {
        data.SetData(ptr[offset + i], i);
      }
      data.SetVlen((uint32_t)(PACKET_SIZE / sizeof(int16_t)));
      break;
    }
    case CHAR8: {
      char* ptr = (char*)info.ptr;
      offset = offset / sizeof(char);
      for (int i = 0; i < PACKET_SIZE / sizeof(char); i++) {
        data.SetData(ptr[offset + i], i);
      }
      data.SetVlen((uint32_t)(PACKET_SIZE / sizeof(char)));
      break;
    }
    case UINT8: {
      uint8_t* ptr = (uint8_t*)info.ptr;
      offset = offset / sizeof(uint8_t);
      for (int i = 0; i < PACKET_SIZE / sizeof(uint8_t);
           i++) {  // TODO: vmask length depends on vlen
        data.SetData(ptr[offset + i], i);
      }
      data.SetVlen((uint32_t)(PACKET_SIZE / sizeof(uint8_t)));
      break;
    }
    case VMASK: {
      bool* ptr = (bool*)info.ptr;
      offset = offset / sizeof(bool);
      for (int i = 0; i < BYTE_BIT;
           i++) {  // TODO: vmask length depends on vlen
        data.SetVmask(ptr[offset + i], i);
      }
      data.SetVlen((uint32_t)BYTE_BIT);  // TODO: think about vlen
      break;
    }
    default:
      spdlog::error("Invalid Pointer Type");
      exit(1);
  }
  return data;
}

void PointerMemoryMap::Store(uint64_t addr, VectorData data) {
  if (!CheckAddr(addr)) {
    spdlog::error("PointerMemoryMap::Store : Invalid Address 0x{:x}", addr);
    exit(1);
  }
  uint64_t base = GetBaseAddr(addr);
  uint64_t offset = addr - base;
  MemoryInfo info = m_ptr_map[base];
  spdlog::trace(
      "Store Addr 0x{:x} Base 0x{:x} Offset 0x{:x} Size {} Type {}, "
      "data.GetType(): {}",
      addr, base, offset, info.size, info.type, data.GetType());
  assert(info.type == data.GetType());
  switch (info.type) {
    case FLOAT32: {
      float* ptr = (float*)info.ptr;
      offset = offset / sizeof(float);
      for (int i = 0; i < data.GetVlen(); i++) {
        ptr[offset + i] = data.GetFloatData(i);
      }
      break;
    }
    case FLOAT16: {
      half* ptr = (half*)info.ptr;
      offset = offset / sizeof(half);
      for (int i = 0; i < data.GetVlen(); i++) {
        ptr[offset + i] = data.GetHalfData(i);
      }
      break;
    }
    case INT64: {
      int64_t* ptr = (int64_t*)info.ptr;
      offset = offset / sizeof(int64_t);
      for (int i = 0; i < data.GetVlen(); i++) {
        ptr[offset + i] = data.GetLongData(i);
      }
      break;
    }
    case INT32: {
      int32_t* ptr = (int32_t*)info.ptr;
      offset = offset / sizeof(int32_t);
      for (int i = 0; i < data.GetVlen(); i++) {
        ptr[offset + i] = data.GetIntData(i);
      }
      break;
    }
    case INT16: {
      int16_t* ptr = (int16_t*)info.ptr;
      offset = offset / sizeof(int16_t);
      for (int i = 0; i < data.GetVlen(); i++) {
        ptr[offset + i] = data.GetShortData(i);
      }
      break;
    }
    case CHAR8: {
      char* ptr = (char*)info.ptr;
      offset = offset / sizeof(char);
      for (int i = 0; i < data.GetVlen(); i++) {
        ptr[offset + i] = data.GetCharData(i);
      }
      break;
    }
    case UINT8: {
      uint8_t* ptr = (uint8_t*)info.ptr;
      offset = offset / sizeof(uint8_t);
      for (int i = 0; i < data.GetVlen(); i++) { // TODO: vmask length depends on vlen
        ptr[offset + i] = data.GetU8Data(i);
      }
      break;
    }
    case VMASK: {
      bool* ptr = (bool*)info.ptr;
      offset = offset / sizeof(bool);
      for (int i = 0; i < BYTE_BIT; i++) { // TODO: vmask length depends on vlen
        ptr[offset + i] = data.GetVmaskData(i);
      }
      break;
    }
    default:
      spdlog::error("Invalid Pointer Type");
      exit(1);
  }
}

bool PointerMemoryMap::CheckAddr(uint64_t addr) {
  uint64_t base = GetBaseAddr(addr);
  if (base == (uint64_t)-1) {
    return false;
  }
  return true;
}

void PointerMemoryMap::Reset() {
  m_ptr_map.clear();
}

void PointerMemoryMap::AllocateMemory(uint64_t addr, uint64_t size,
                                      DataType type, void* ptr) {
  spdlog::info("Allocate Memory Addr {:x} Size {} Type {}", addr, size, type);
  MemoryInfo info;
  info.size = size;
  info.type = type;
  info.ptr = ptr;
  m_ptr_map[addr] = info;
}

void PointerMemoryMap::FreeMemory(uint64_t addr) {
  spdlog::info("Free Memory Addr {:x}", addr);
  m_ptr_map.erase(addr);
}

uint64_t PointerMemoryMap::GetBaseAddr(uint64_t addr) {
  for (auto& [key, val] : m_ptr_map) {
    if (addr >= key && addr < key + val.size) {
      return key;
    }
  }
  return (uint64_t)-1;
}
}  // namespace NDPSim