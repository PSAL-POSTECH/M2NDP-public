#include "vector_data.h"

#include <fp16.h>

#include <iostream>

#include "common.h"
namespace NDPSim {

VectorData::VectorData() {
  m_data = std::vector<Data>(PACKET_ENTRIES);
  m_vlen = PACKET_ENTRIES;  // 8
  m_double_reg = false;
}

VectorData::VectorData(bool double_reg) {
  if (double_reg) {
    m_data = std::vector<Data>(PACKET_ENTRIES * 2);
    m_vlen = PACKET_ENTRIES * 2;
  } else {
    m_data = std::vector<Data>(PACKET_ENTRIES);
    m_vlen = PACKET_ENTRIES;
  }
  m_double_reg = double_reg;
}

VectorData::VectorData(int sew, float lmul) {
  float vlen = (float) (PACKET_SIZE * BYTE_BIT * lmul);
  vlen /= sew;

  float div = UNION_DATA_SIZE / (float)sew;
  int vec_len = (int)(vlen / div);

  m_data = std::vector<Data>(int(ROUND_UP(vec_len, PACKET_ENTRIES)));
  m_vlen = (uint32_t)vlen;
  m_double_reg = (lmul == 2.0);
}

VectorData::VectorData(Context &context) {
  float f_vlen = PACKET_SIZE * BYTE_BIT * context.csr->vtype_vlmul;
  f_vlen /= context.csr->vtype_vsew;
  m_vlen = (int) f_vlen;

  int vec_len = (m_vlen * context.csr->vtype_vsew) / UNION_DATA_SIZE;
  
  m_data = std::vector<Data>(int(ROUND_UP(vec_len, PACKET_ENTRIES)));

  m_double_reg = (context.csr->vtype_vlmul == 2.0);
}

void VectorData::SetVectorDoubleReg() {
  m_data.resize(PACKET_ENTRIES * 2); // what if sew == 64? PACKET_ENTIRES * 2 * 2?
  m_double_reg = true;
}

void VectorData::Widen(Context &context) {
  std::vector<Data> new_data;
  int vec_len = (m_vlen * context.csr->vtype_vsew) / UNION_DATA_SIZE;

  for (int i = 0; i < m_vlen; i++) {
    new_data.push_back(m_data[i]);
  }
  for (int i = 0; i < m_vlen; i++) {
    new_data.push_back(m_data[i]);
  }
  m_data.resize(vec_len * 2);
  m_data = new_data;
  // m_vlen *= 2;
}

void VectorData::SetType(DataType type) { m_type = type; }

void VectorData::SetVlen(uint32_t vlen) { m_vlen = vlen; }

void VectorData::SetData(float data, int index) {
  assert(index < m_vlen);
  m_type = FLOAT32;
  m_data[index].fp32_data = data;
}

void VectorData::SetData(int32_t data, int index) {
  assert(index < m_vlen);
  m_type = INT32;
  m_data[index].int32_data = data;
}

void VectorData::SetData(int16_t data, int index) {
  assert((index / 2) < m_vlen);
  m_type = INT16;
  m_data[index / 2].int16_data[index % 2] = data;
}

void VectorData::SetData(uint8_t data, int index) {
  assert((index / 4) < m_vlen);
  m_type = UINT8;
  m_data[index / 4].char8_data[index % 4] = data;
}

void VectorData::SetData(int64_t data, int index) {
  assert(index < m_vlen);
  m_type = INT64;
  m_data[2 * index].int32_data = data >> UNION_DATA_SIZE;
  m_data[2 * index + 1].int32_data = data & 0xffffffff;
}

void VectorData::SetData(half data, int index) {
  assert((index / 2) < m_vlen);
  m_type = FLOAT16;
  m_data[index / 2].fp16_data[index % 2] =
      fp16_ieee_from_fp32_value((float)data);
}

void VectorData::SetData(char data, int index) {
  assert((index / 4) < m_vlen);
  m_type = CHAR8;
  m_data[index / 4].char8_data[index % 4] = data;
}

void VectorData::SetData(bool data, int index) {
  assert((index / 4) < m_vlen);
  m_type = BOOL;
  m_data[index / 4].bool_data[index % 4] = data;
}

void VectorData::SetVmask(bool data, int index) {
  m_type = VMASK;
  if (data) m_data[index / UNION_DATA_SIZE].vmask += (0x01 << (index % UNION_DATA_SIZE));
}

float VectorData::GetFloatData(int index) const {
  return m_data[index].fp32_data;
}

int32_t VectorData::GetIntData(int index) const {
  return m_data[index].int32_data;
}

int16_t VectorData::GetShortData(int index) const {
  return m_data[index / 2].int16_data[index % 2];
}

uint8_t VectorData::GetU8Data(int index) const {
  return m_data[index / 4].uint8_data[index % 4];
}

int64_t VectorData::GetLongData(int index) const {
  int64_t msb =((uint64_t) m_data[2 * index].int32_data) << WORD_BIT_SIZE;
  int64_t lsb =((uint64_t) m_data[2 * index + 1].int32_data) & 0xffffffff;
  return msb | lsb;
}

half VectorData::GetHalfData(int index) const {
  return half(fp16_ieee_to_fp32_value(m_data[index / 2].fp16_data[index % 2]));
}

char VectorData::GetCharData(int index) const {
  return m_data[index / 4].char8_data[index % 4];
}

bool VectorData::GetBoolData(int index) const {
  return m_data[index / 4].bool_data[index % 4];
}

int32_t VectorData::GetVmaskData() const { return m_data[0].vmask; }

bool VectorData::GetVmaskData(int index) const {
  return (((m_data[index / UNION_DATA_SIZE].vmask >> (index % UNION_DATA_SIZE)) % 2) == 1);
}

uint8_t VectorData::GetVmaskAsByte(int index) const {
  uint8_t ret = 0;
  for (int i = index; i < index + 8; i++) {
    ret += (GetVmaskData(i) << (i % 8));
  }
  return ret;
}

uint32_t VectorData::GetVlen() const { return m_vlen; }

uint32_t VectorData::GetPrecision() const {
  switch (m_type) {
    case INT64:
      return 8;
    case FLOAT32:
    case INT32:
      return 4;
    case INT16:
    case FLOAT16:
      return 2;
    case CHAR8:
      return 1;
    default:
      return 1;
  }
}

void VectorData::Append(VectorData &rhs) {
  int iter = rhs.GetVlen();
  if (rhs.GetType() == INT64) iter *= 2;
  for (int i = 0; i < iter; i++) {
    m_data.push_back(rhs.m_data[i]);
  }
  m_vlen += rhs.GetVlen();
  if (m_double_reg)
    throw std::runtime_error("Double appending register");

  m_double_reg = true;
}

std::array<VectorData, 2> VectorData::Split() {
  std::array<VectorData, 2> ret;
  switch(m_type) {
    case INT64:
      ret[0] = VectorData(64, 1.0);
      ret[1] = VectorData(64, 1.0);
      break;
    case INT32:
    case FLOAT32:
      ret[0] = VectorData(32, 1.0);
      ret[1] = VectorData(32, 1.0);
      break;
    case INT16:
    case FLOAT16:
      ret[0] = VectorData(16, 1.0);
      ret[1] = VectorData(16, 1.0);
      break;
    case CHAR8:
      ret[0] = VectorData(8, 1.0);
      ret[1] = VectorData(8, 1.0);
      break;
    default :
      ret[0] = VectorData();
      ret[1] = VectorData();
      break;
  }

  int iter = m_vlen;
  if (GetType() == INT64) iter *= 2;

  for (int i = 0; i < iter; i++) {
    if (i < iter / 2) {
      ret[0].m_data[i] = m_data[i];
    } else {
      ret[1].m_data[i - iter / 2] = m_data[i];
    }
  }
  ret[0].m_type = m_type;
  ret[1].m_type = m_type;
  ret[0].m_vlen = m_vlen / 2;
  ret[1].m_vlen = m_vlen / 2;
  return ret;
}

void VectorData::PrintVectorData() const {
  switch (m_type) {
    case INT64:
      printf("INT64 : ");
      for (int i = 0; i < m_vlen; i++) printf("%e ", this->GetLongData(i));
      printf("\n");
      break;
    case FLOAT32:
      printf("FLOAT32 : ");
      for (int i = 0; i < m_vlen; i++) printf("%e ", this->GetFloatData(i));
      printf("\n");
      break;
    case INT32:
      printf("INT32 : ");
      for (int i = 0; i < m_vlen; i++) printf("%d ", this->GetIntData(i));
      printf("\n");
      break;
    case INT16:
      printf("INT16 : ");
      for (int i = 0; i < m_vlen; i++) printf("%d ", this->GetShortData(i));
      printf("\n");
      break;
    case FLOAT16:
      printf("FLOAT16 : ");
      for (int i = 0; i < m_vlen; i++)
        printf("%f ", (float)(this->GetHalfData(i)));
      printf("\n");
      break;
    case VMASK:
      printf("VMASK : ");
      for (int i = 0; i < m_vlen; i++) printf("%u ", this->GetVmaskData(i));
      printf("\n");
      break;
    case CHAR8:
      printf("CHAR8 : ");
      for (int i = 0; i < m_vlen; i++) printf("%c ", this->GetCharData(i));
      printf("\n");
      break;
    case UINT8:
      printf("UINT8 : ");
      for (int i = 0; i < m_vlen; i++) printf("%u ", this->GetU8Data(i));
      printf("\n");
      break;
    case BOOL:
      printf("BOOL : ");
      for (int i = 0; i < m_vlen; i++) printf("%d ", this->GetBoolData(i));
      printf("\n");
      break;
    default:
      printf("%d : PrintVectorData Type Unsupported..\n", this->GetType());
  }
}

std::string VectorData::toString() const {
  std::string out = "";
  switch (m_type)
  {
  case INT64:
    out += "INT64 : ";
    for (int i = 0; i < m_vlen; i++) out += std::to_string(this->GetLongData(i)) + " ";
    break;
  case FLOAT32:
    out += "FLOAT32 : ";
    for (int i = 0; i < m_vlen; i++) out += std::to_string(this->GetFloatData(i)) + " ";
    break;
  case INT32:
    out += "INT32 : ";
    for (int i = 0; i < m_vlen; i++) out += std::to_string(this->GetIntData(i)) + " ";
    break;
  case FLOAT16:
    out += "FLOAT16 : ";
    for (int i = 0; i < m_vlen; i++)
      out += std::to_string((float)(this->GetHalfData(i))) + " ";
    break;
  case INT16:
    out += "INT16: ";
    for (int i = 0; i < m_vlen; i++)
      out += std::to_string((short)(this->GetShortData(i))) + " ";
    break;
  case VMASK:
    out += "VMASK : ";
    for (int i = 0; i < m_vlen; i++) out += std::to_string(this->GetVmaskData(i)) + " ";
    break;
  case CHAR8:
    out += "CHAR8 : ";
    for (int i = 0; i < m_vlen; i++) out += std::to_string(this->GetCharData(i)) + " ";
    break;
  case UINT8:
    out += "UINT8 : ";
    for (int i = 0; i < m_vlen; i++) out += std::to_string(this->GetU8Data(i)) + " ";
    break;
  case BOOL:
    out += "BOOL : ";
    for (int i = 0; i < m_vlen; i++) out += std::to_string(this->GetBoolData(i)) + " ";
    break;
  default:
    out += std::to_string(this->GetType()) + " : To String Type Unsupported..\n";
  }
  return out;
}

VectorData VectorData::Max(VectorData &rhs) {
  VectorData ret = *this;
  for (int i = 0; i < GetVlen(); i++) {
    switch (m_type) {
      case FLOAT32:
        ret.SetData(std::max(GetFloatData(i), rhs.GetFloatData(i)), i);
        break;
      case FLOAT16:
        ret.SetData(std::max(GetHalfData(i), rhs.GetHalfData(i)), i);
        break;
      case INT64:
        ret.SetData(std::max(GetLongData(i), rhs.GetLongData(i)), i);
        break;
      case INT32:
        ret.SetData(std::max(GetIntData(i), rhs.GetIntData(i)), i);
        break;
      case INT16:
        ret.SetData(std::max(GetShortData(i), rhs.GetShortData(i)), i);
        break;
      default:
        throw std::runtime_error("VectorData::Max: invalid m_type");
    }
  }
  return ret;
}

VectorData VectorData::Min(VectorData &rhs) {
  VectorData ret = *this;
  for (int i = 0; i < GetVlen(); i++) {
    switch (m_type) {
      case FLOAT32:
        ret.SetData(std::min(GetFloatData(i), rhs.GetFloatData(i)), i);
        break;
      case FLOAT16:
        ret.SetData(std::min(GetHalfData(i), rhs.GetHalfData(i)), i);
        break;
      case INT64:
        ret.SetData(std::min(GetLongData(i), rhs.GetLongData(i)), i);
        break;
      case INT32:
        ret.SetData(std::min(GetIntData(i), rhs.GetIntData(i)), i);
        break;
      case INT16:
        ret.SetData(std::min(GetShortData(i), rhs.GetShortData(i)), i);
        break;
      default:
        throw std::runtime_error("VectorData::min: invalid m_type");
    }
  }
  return ret;
}

VectorData VectorData::And(VectorData &rhs) {
  VectorData ret = *this;
  for (int i = 0; i < GetVlen(); i++) {
    switch (m_type) {
      case FLOAT32:
        ret.SetData((GetFloatData(i) && rhs.GetFloatData(i)), i);
        break;
      case FLOAT16:
        ret.SetData((GetHalfData(i) && rhs.GetHalfData(i)), i);
        break;
      case INT64:
        ret.SetData((GetLongData(i) && rhs.GetLongData(i)), i);
        break;
      case INT32:
        ret.SetData((GetIntData(i) && rhs.GetIntData(i)), i);
        break;
      case INT16:
        ret.SetData((GetShortData(i) && rhs.GetShortData(i)), i);
        break;
      default:
        throw std::runtime_error("VectorData::min: invalid m_type");
    }
  }
  return ret;
}

VectorData VectorData::Or(VectorData &rhs) {
  VectorData ret = *this;
  for (int i = 0; i < GetVlen(); i++) {
    switch (m_type) {
      case FLOAT32:
        ret.SetData((GetFloatData(i) || rhs.GetFloatData(i)), i);
        break;
      case FLOAT16:
        ret.SetData((GetHalfData(i) || rhs.GetHalfData(i)), i);
        break;
      case INT64:
        ret.SetData((GetLongData(i) || rhs.GetLongData(i)), i);
        break;
      case INT32:
        ret.SetData((GetIntData(i) || rhs.GetIntData(i)), i);
        break;
      case INT16:
        ret.SetData((GetShortData(i) || rhs.GetShortData(i)), i);
        break;
      default:
        throw std::runtime_error("VectorData::min: invalid m_type");
    }
  }
  return ret;
}

VectorData VectorData::Exp(VectorData &rhs) {
  VectorData ret = *this;
  float exp_data;
  for (int i = 0; i < GetVlen(); i++) {
    switch (m_type) {
      case FLOAT32:
        exp_data = std::exp(rhs.GetFloatData(i));
        if (exp_data < 0.0) {
          std::cout << "Exp error!" << std::endl;
        }
        ret.SetData(exp_data, i);
        break;
      case FLOAT16:
        exp_data = std::exp(static_cast<float>(rhs.GetHalfData(i)));
        if (exp_data < 0.0) {
          std::cout << "Exp error!" << std::endl;
        }
        ret.SetData(static_cast<half>(exp_data), i);
        break;
      default:
        throw std::runtime_error("VectorData::Exp: Support only fp32");
    }
  }
  return ret;
}

VectorData operator+(const VectorData &lhs, VectorData &rhs) {
  assert(lhs.GetVlen() == rhs.GetVlen());

  VectorData ret = lhs;
  DataType lhs_type = lhs.GetType();
  DataType rhs_type = rhs.GetType();

  for (int i = 0; i < lhs.GetVlen(); i++) {
    if (lhs_type == FLOAT32) {
      if (rhs_type == FLOAT32) {
        ret.SetData(float(lhs.GetFloatData(i) + rhs.GetFloatData(i)), i);
      } else if (rhs_type == FLOAT16) {
        ret.SetData(float(lhs.GetFloatData(i) + (float)rhs.GetHalfData(i)), i);
      } else if (rhs_type == INT32) {
        ret.SetData(float(lhs.GetFloatData(i) + rhs.GetIntData(i)), i);
      } else
        throw std::runtime_error("operator+: invalid m_type");
    } else if (lhs_type == FLOAT16) {
      if (rhs_type == FLOAT32) {
        ret.SetData(half((float)lhs.GetHalfData(i) + rhs.GetFloatData(i)), i);
      } else if (rhs_type == FLOAT16) {
        ret.SetData(half((float)lhs.GetHalfData(i) + (float)rhs.GetHalfData(i)),
                    i);
      } else if (rhs_type == INT32) {
        ret.SetData(half((float)lhs.GetHalfData(i) + rhs.GetIntData(i)), i);
      } else if (rhs_type == INT16) {
        ret.SetData(half((float)lhs.GetHalfData(i) + rhs.GetShortData(i)), i);
      } else
        throw std::runtime_error("operator+: invalid m_type");
    } else if (lhs_type == INT32) {
      if (rhs_type == FLOAT32) {
        ret.SetData(int(lhs.GetIntData(i) + rhs.GetFloatData(i)), i);
      } else if (rhs_type == FLOAT16) {
        ret.SetData(int(lhs.GetIntData(i) + (float)rhs.GetHalfData(i)), i);
      } else if (rhs_type == INT32) {
        ret.SetData(int(lhs.GetIntData(i) + rhs.GetIntData(i)), i);
      } else if (rhs_type == INT16) {
        ret.SetData(int(lhs.GetIntData(i) + rhs.GetShortData(i)), i);
      } else
        throw std::runtime_error("operator+: invalid m_type");
    } else if (lhs_type == INT64) {
      if (rhs_type == INT64) {
        ret.SetData(int64_t(lhs.GetLongData(i) + rhs.GetLongData(i)), i);
      } else
        throw std::runtime_error("operator+: invalid m_type");
    } else
      throw std::runtime_error("operator+: invalid m_type");
  }
  return ret;
}

VectorData operator+(const VectorData &lhs, float &rhs) {
  VectorData ret = lhs;
  for (int i = 0; i < lhs.GetVlen(); i++) {
    switch (lhs.GetType()) {
      case FLOAT32:
        ret.SetData(lhs.GetFloatData(i) + rhs, i);
        break;
      case FLOAT16:
        ret.SetData(static_cast<half>(lhs.GetHalfData(i) + rhs), i);
        break;
      case INT32:
        ret.SetData(static_cast<int>(lhs.GetIntData(i) + rhs), i);
        ret.SetType(FLOAT32);
        break;
      case INT16:
        ret.SetData(static_cast<short>(lhs.GetShortData(i) + rhs), i);
        ret.SetType(FLOAT16);
        break;
      default:
        throw std::runtime_error("operator+: invalid m_type");
    }
  }
  return ret;
}

VectorData operator+(const VectorData &lhs, int32_t &rhs) {
  VectorData ret = lhs;
  for (int i = 0; i < lhs.GetVlen(); i++) {
    switch (lhs.GetType()) {
      case FLOAT32:
        ret.SetData(lhs.GetFloatData(i) + rhs, i);
        ret.SetType(FLOAT32);
        break;
      case FLOAT16:
        ret.SetData(static_cast<half>(lhs.GetHalfData(i) + rhs), i);
        ret.SetType(FLOAT16);
        break;
      case INT64:
        ret.SetData(int64_t(lhs.GetLongData(i) + (int64_t)rhs), i);
        ret.SetType(INT64);
        break;
      case INT32:
        ret.SetData(static_cast<int>(lhs.GetIntData(i) + rhs), i);
        ret.SetType(INT32);
        break;
      case INT16:
        ret.SetData(static_cast<short>(lhs.GetShortData(i) + rhs), i);
        ret.SetType(INT16);
        break;
      default:
        throw std::runtime_error("operator+: invalid m_type");
    }
  }
  return ret;
}

VectorData operator+(const VectorData &lhs, int64_t &rhs) {
  VectorData ret = lhs;
  for (int i = 0; i < lhs.GetVlen(); i++) {
    switch (lhs.GetType()) {
      case FLOAT32:
        ret.SetData(lhs.GetFloatData(i) + rhs, i);
        ret.SetType(FLOAT32);
        break;
      case FLOAT16:
        ret.SetData(static_cast<half>(lhs.GetHalfData(i) + rhs), i);
        ret.SetType(FLOAT16);
        break;
      case INT64:
        ret.SetData(int64_t(lhs.GetLongData(i) + rhs), i);
        ret.SetType(INT64);
        break;
      case INT32:
        ret.SetData(static_cast<int>(lhs.GetIntData(i) + rhs), i);
        ret.SetType(INT32);
        break;
      case INT16:
        ret.SetData(static_cast<short>(lhs.GetShortData(i) + rhs), i);
        ret.SetType(INT16);
        break;
      default:
        throw std::runtime_error("operator+: invalid m_type");
    }
  }
  return ret;
}

VectorData operator-(const VectorData &lhs, VectorData &rhs) {
  assert(lhs.GetVlen() == rhs.GetVlen());

  VectorData ret = lhs;
  DataType lhs_type = lhs.GetType();
  DataType rhs_type = rhs.GetType();

  for (int i = 0; i < lhs.GetVlen(); i++) {
    if (lhs_type == FLOAT32) {
      if (rhs_type == FLOAT32) {
        ret.SetData(float(lhs.GetFloatData(i) - rhs.GetFloatData(i)), i);
      } else if (rhs_type == FLOAT16) {
        ret.SetData(float(lhs.GetFloatData(i) - rhs.GetHalfData(i)), i);
      } else if (rhs_type == INT32) {
        ret.SetData(float(lhs.GetFloatData(i) - rhs.GetIntData(i)), i);
      } else if (rhs_type == INT16) {
        ret.SetData(float(lhs.GetFloatData(i) - rhs.GetShortData(i)), i);
      } else
        throw std::runtime_error("operator-: invalid m_type");
    } else if (lhs_type == FLOAT16) {
      if (rhs_type == FLOAT32) {
        ret.SetData(half(lhs.GetHalfData(i) - rhs.GetFloatData(i)), i);
      } else if (rhs_type == FLOAT16) {
        ret.SetData(half(lhs.GetHalfData(i) - rhs.GetHalfData(i)), i);
      } else if (rhs_type == INT32) {
        ret.SetData(half(lhs.GetHalfData(i) - rhs.GetIntData(i)), i);
      } else if (rhs_type == INT16) {
        ret.SetData(half(lhs.GetHalfData(i) - rhs.GetShortData(i)), i);
      } else
        throw std::runtime_error("operator-: invalid m_type");
    } else if (lhs_type == INT32) {
      if (rhs_type == FLOAT32) {
        ret.SetData(int(lhs.GetIntData(i) - rhs.GetFloatData(i)), i);
      } else if (rhs_type == FLOAT16) {
        ret.SetData(int(lhs.GetIntData(i) - rhs.GetHalfData(i)), i);
      } else if (rhs_type == INT32) {
        ret.SetData(int(lhs.GetIntData(i) - rhs.GetIntData(i)), i);
      } else if (rhs_type == INT16) {
        ret.SetData(int(lhs.GetIntData(i) - rhs.GetShortData(i)), i);
      } else
        throw std::runtime_error("operator-: invalid m_type");
    } else if (lhs_type == INT16) {
      if (rhs_type == FLOAT32) {
        ret.SetData(short(lhs.GetShortData(i) - rhs.GetFloatData(i)), i);
      } else if (rhs_type == FLOAT16) {
        ret.SetData(short(lhs.GetShortData(i) - rhs.GetHalfData(i)), i);
      } else if (rhs_type == INT32) {
        ret.SetData(short(lhs.GetShortData(i) - rhs.GetIntData(i)), i);
      } else if (rhs_type == INT16) {
        ret.SetData(short(lhs.GetShortData(i) - rhs.GetShortData(i)), i);
      } else
        throw std::runtime_error("operator-: invalid m_type");
    } else if (lhs_type == INT64) {
      if (rhs_type == INT64)
        ret.SetData(int64_t(lhs.GetLongData(i) - rhs.GetLongData(i)), i);
    } else
      throw std::runtime_error("operator-: invalid m_type");
  }
  return ret;
}

VectorData operator-(const VectorData &lhs, float &rhs) {
  VectorData ret = lhs;
  for (int i = 0; i < lhs.GetVlen(); i++) {
    switch (lhs.GetType()) {
      case FLOAT32:
        ret.SetData(lhs.GetFloatData(i) - rhs, i);
        break;
      case FLOAT16:
        ret.SetData(static_cast<half>(lhs.GetHalfData(i) - rhs), i);
        break;
      case INT32:
        ret.SetData(static_cast<int>(lhs.GetIntData(i) - rhs), i);
        break;
      case INT16:
        ret.SetData(static_cast<short>(lhs.GetShortData(i) - rhs), i);
        break;
      default:
        throw std::runtime_error("operator-: invalid m_type");
    }
  }
  return ret;
}

VectorData operator-(const VectorData &lhs, int32_t &rhs) {
  VectorData ret = lhs;
  for (int i = 0; i < lhs.GetVlen(); i++) {
    switch (lhs.GetType()) {
      case FLOAT32:
        ret.SetData(static_cast<float>(lhs.GetFloatData(i) - rhs), i);
        ret.SetType(FLOAT32);
        break;
      case FLOAT16:
        ret.SetData(static_cast<half>(lhs.GetHalfData(i) - rhs), i);
        ret.SetType(FLOAT16);
        break;
      case INT32:
        ret.SetData(static_cast<int>(lhs.GetIntData(i) - rhs), i);
        ret.SetType(INT32);
        break;
      case INT16:
        ret.SetData(static_cast<short>(lhs.GetShortData(i) - rhs), i);
        ret.SetType(INT16);
        break;
      default:
        throw std::runtime_error("operator-: invalid m_type");
    }
  }
  return ret;
}

VectorData operator-(const float &lhs, VectorData &rhs) {
  VectorData ret = rhs;
  for (int i = 0; i < rhs.GetVlen(); i++) {
    switch (rhs.GetType()) {
      case FLOAT32:
        ret.SetData(lhs - rhs.GetFloatData(i), i);
        break;
      case FLOAT16:
        ret.SetData(static_cast<half>(lhs - rhs.GetHalfData(i)), i);
        break;
      case INT32:
        ret.SetData(static_cast<int>(lhs - rhs.GetIntData(i)), i);
        ret.SetType(FLOAT32);
        break;
      case INT16:
        ret.SetData(static_cast<short>(lhs - rhs.GetShortData(i)), i);
        ret.SetType(FLOAT16);
        break;
      default:
        throw std::runtime_error("operator-: invalid m_type");
    }
  }
  return ret;
}

VectorData operator-(const int32_t &lhs, VectorData &rhs) {
  VectorData ret = rhs;
  for (int i = 0; i < rhs.GetVlen(); i++) {
    switch (rhs.GetType()) {
      case FLOAT32:
        ret.SetData(static_cast<float>(lhs - rhs.GetFloatData(i)), i);
        ret.SetType(FLOAT32);
        break;
      case FLOAT16:
        ret.SetData(static_cast<half>(lhs - rhs.GetHalfData(i)), i);
        ret.SetType(FLOAT16);
        break;
      case INT32:
        ret.SetData(static_cast<int>(lhs - rhs.GetIntData(i)), i);
        ret.SetType(INT32);
        break;
      case INT16:
        ret.SetData(static_cast<short>(lhs - rhs.GetShortData(i)), i);
        ret.SetType(INT16);
        break;
      default:
        throw std::runtime_error("operator-: invalid m_type");
    }
  }
  return ret;
}

VectorData operator*(const VectorData &lhs, VectorData &rhs) {
  assert(lhs.GetVlen() == rhs.GetVlen());

  VectorData ret = lhs;
  DataType lhs_type = lhs.GetType();
  DataType rhs_type = rhs.GetType();

  for (int i = 0; i < lhs.GetVlen(); i++) {
    if (lhs_type == FLOAT32) {
      if (rhs_type == FLOAT32) {
        ret.SetData(float(lhs.GetFloatData(i) * rhs.GetFloatData(i)), i);
      } else if (rhs_type == FLOAT16) {
        ret.SetData(float(lhs.GetFloatData(i) * (float)rhs.GetHalfData(i)), i);
      } else if (rhs_type == INT32) {
        ret.SetData(float(lhs.GetFloatData(i) * rhs.GetIntData(i)), i);
      } else if (rhs_type == INT16) {
        ret.SetData(float(lhs.GetFloatData(i) * rhs.GetShortData(i)), i);
      } else
        throw std::runtime_error("operator*: invalid m_type");
    } else if (lhs_type == FLOAT16) {
      if (rhs_type == FLOAT32) {
        ret.SetData(half((float)lhs.GetHalfData(i) * rhs.GetFloatData(i)), i);
      } else if (rhs_type == FLOAT16) {
        ret.SetData(half((float)lhs.GetHalfData(i) * (float)rhs.GetHalfData(i)),
                    i);
      } else if (rhs_type == INT32) {
        ret.SetData(half((float)lhs.GetHalfData(i) * rhs.GetIntData(i)), i);
      } else if (rhs_type == INT16) {
        ret.SetData(half((float)lhs.GetHalfData(i) * rhs.GetShortData(i)), i);
      } else
        throw std::runtime_error("operator*: invalid m_type");
    } else if (lhs_type == INT32) {
      if (rhs_type == FLOAT32) {
        ret.SetData(int(lhs.GetIntData(i) * rhs.GetFloatData(i)), i);
      } else if (rhs_type == FLOAT16) {
        ret.SetData(int(lhs.GetIntData(i) * (float)rhs.GetHalfData(i)), i);
      } else if (rhs_type == INT32) {
        ret.SetData(int(lhs.GetIntData(i) * rhs.GetIntData(i)), i);
      } else if (rhs_type == INT16) {
        ret.SetData(int(lhs.GetIntData(i) * rhs.GetShortData(i)), i);
      } else
        throw std::runtime_error("operator*: invalid m_type");
    } else if (lhs_type == INT32) {
      if (rhs_type == FLOAT32) {
        ret.SetData(short(lhs.GetShortData(i) * rhs.GetFloatData(i)), i);
      } else if (rhs_type == FLOAT16) {
        ret.SetData(short(lhs.GetShortData(i) * (float)rhs.GetHalfData(i)), i);
      } else if (rhs_type == INT32) {
        ret.SetData(short(lhs.GetShortData(i) * rhs.GetIntData(i)), i);
      } else if (rhs_type == INT16) {
        ret.SetData(short(lhs.GetShortData(i) * rhs.GetShortData(i)), i);
      } else
        throw std::runtime_error("operator*: invalid m_type");
    } else
      throw std::runtime_error("operator*: invalid m_type");
  }
  return ret;
}

VectorData operator*(const VectorData &lhs, float &rhs) {
  VectorData ret = lhs;
  for (int i = 0; i < lhs.GetVlen(); i++) {
    switch (lhs.GetType()) {
      case FLOAT32:
        ret.SetData(lhs.GetFloatData(i) * rhs, i);
        break;
      case FLOAT16:
        ret.SetData(static_cast<half>(lhs.GetHalfData(i) * rhs), i);
        break;
      case INT32:
        ret.SetData(static_cast<int>(lhs.GetIntData(i) * rhs), i);
        break;
      case INT16:
        ret.SetData(static_cast<short>(lhs.GetShortData(i) * rhs), i);
        break;
      default:
        throw std::runtime_error("operator*: invalid m_type");
    }
  }
  return ret;
}

VectorData operator*(const VectorData &lhs, int32_t &rhs) {
  VectorData ret = lhs;
  for (int i = 0; i < lhs.GetVlen(); i++) {
    switch (lhs.GetType()) {
      case FLOAT32:
        ret.SetData(lhs.GetFloatData(i) * rhs, i);
        ret.SetType(FLOAT32);
        break;
      case FLOAT16:
        ret.SetData(static_cast<half>(lhs.GetHalfData(i) * rhs), i);
        ret.SetType(FLOAT16);
        break;
      case INT64:
        ret.SetData(static_cast<long>(lhs.GetLongData(i) * rhs), i);
        break;
      case INT32:
        ret.SetData(static_cast<int>(lhs.GetIntData(i) * rhs), i);
        break;
      case INT16:
        ret.SetData(int16_t(lhs.GetShortData(i) * rhs), i);
        break;
      case UINT8:
        ret.SetData(uint8_t(lhs.GetU8Data(i) * rhs), i);
        break;
      default:
        throw std::runtime_error("operator*: invalid m_type");
    }
  }
  return ret;
}

VectorData operator/(const VectorData &lhs, VectorData &rhs) {
  assert(lhs.GetVlen() == rhs.GetVlen());

  VectorData ret = lhs;
  DataType lhs_type = lhs.GetType();
  DataType rhs_type = rhs.GetType();

  for (int i = 0; i < lhs.GetVlen(); i++) {
    if (lhs_type == FLOAT32) {
      if (rhs_type == FLOAT32) {
        ret.SetData(float(lhs.GetFloatData(i) / rhs.GetFloatData(i)), i);
      } else if (rhs_type == FLOAT16) {
        ret.SetData(float(lhs.GetFloatData(i) / rhs.GetHalfData(i)), i);
      } else if (rhs_type == INT32) {
        ret.SetData(float(lhs.GetFloatData(i) / rhs.GetIntData(i)), i);
      } else if (rhs_type == INT16) {
        ret.SetData(float(lhs.GetFloatData(i) / rhs.GetShortData(i)), i);
      } else
        throw std::runtime_error("operator/: invalid m_type");
    } else if (lhs_type == FLOAT16) {
      if (rhs_type == FLOAT32) {
        ret.SetData(half(lhs.GetHalfData(i) / rhs.GetFloatData(i)), i);
      } else if (rhs_type == FLOAT16) {
        ret.SetData(half(lhs.GetHalfData(i) / rhs.GetHalfData(i)), i);
      } else if (rhs_type == INT32) {
        ret.SetData(half(lhs.GetHalfData(i) / rhs.GetIntData(i)), i);
      } else if (rhs_type == INT16) {
        ret.SetData(half(lhs.GetHalfData(i) / rhs.GetShortData(i)), i);
      } else
        throw std::runtime_error("operator/: invalid m_type");
    } else if (lhs_type == INT32) { // TODO: Fix the data type, not follow lhs_type, it must follow rd_type.
      if (rhs_type == FLOAT32) {
        ret.SetData(int(lhs.GetIntData(i) / rhs.GetFloatData(i)), i);
      } else if (rhs_type == FLOAT16) {
        ret.SetData(int(lhs.GetIntData(i) / rhs.GetHalfData(i)), i);
      } else if (rhs_type == INT32) {
        ret.SetData(int(lhs.GetIntData(i) / rhs.GetIntData(i)), i);
      } else if (rhs_type == INT16) {
        ret.SetData(int(lhs.GetIntData(i) / rhs.GetShortData(i)), i);
      } else
        throw std::runtime_error("operator/: invalid m_type");
    } else if (lhs_type == INT16) {
      if (rhs_type == FLOAT32) {
        ret.SetData(short(lhs.GetShortData(i) / rhs.GetFloatData(i)), i);
      } else if (rhs_type == FLOAT16) {
        ret.SetData(short(lhs.GetShortData(i) / rhs.GetHalfData(i)), i);
      } else if (rhs_type == INT32) {
        ret.SetData(short(lhs.GetShortData(i) / rhs.GetIntData(i)), i);
      } else if (rhs_type == INT16) {
        ret.SetData(short(lhs.GetShortData(i) / rhs.GetShortData(i)), i);
      } else
        throw std::runtime_error("operator/: invalid m_type");
    } else
      throw std::runtime_error("operator/: invalid m_type");
  }
  return ret;
}

VectorData operator/(const VectorData &lhs, float &rhs) {
  VectorData ret = lhs;
  for (int i = 0; i < lhs.GetVlen(); i++) {
    switch (lhs.GetType()) {
      case FLOAT32:
        ret.SetData(lhs.GetFloatData(i) / rhs, i);
        break;
      case FLOAT16:
        ret.SetData(static_cast<half>(lhs.GetHalfData(i) / rhs), i);
        break;
      case INT32:
        ret.SetData(static_cast<int>(lhs.GetIntData(i) / rhs), i);
        break;
      case INT16:
        ret.SetData(static_cast<short>(lhs.GetShortData(i) / rhs), i);
        break;
      default:
        throw std::runtime_error("operator/: invalid m_type");
    }
  }
  return ret;
}

VectorData operator/(float &lhs, const VectorData &rhs) {
  VectorData ret = rhs;
  for (int i = 0; i < rhs.GetVlen(); i++) {
    switch (rhs.GetType()) {
      case FLOAT32:
        ret.SetData(lhs / rhs.GetFloatData(i), i);
        break;
      case FLOAT16:
        ret.SetData(static_cast<half>(lhs / rhs.GetHalfData(i)), i);
        break;
      case INT32:
        ret.SetData(static_cast<int>(lhs / rhs.GetIntData(i)), i);
        break;
      case INT16:
        ret.SetData(static_cast<short>(lhs / rhs.GetShortData(i)), i);
        break;
      default:
        throw std::runtime_error("operator/: invalid m_type");
    }
  }
  return ret;
}

VectorData operator/(const VectorData &lhs, int32_t &rhs) {
  VectorData ret = lhs;
  for (int i = 0; i < lhs.GetVlen(); i++) {
    switch (lhs.GetType()) {
      case FLOAT32:
        ret.SetData(lhs.GetFloatData(i) / rhs, i);
        ret.SetType(FLOAT32);
        break;
      case FLOAT16:
        ret.SetData(static_cast<half>(lhs.GetHalfData(i) / rhs), i);
        ret.SetType(FLOAT16);
        break;
      case INT32:
        ret.SetData(static_cast<int>(lhs.GetIntData(i) / rhs), i);
        break;
      case INT16:
        ret.SetData(static_cast<short>(lhs.GetShortData(i) / rhs), i);
        break;
      default:
        throw std::runtime_error("operator/: invalid m_type");
    }
  }
  return ret;
}
} // namespace onnc