#ifndef VECTOR_DATA_H
#define VECTOR_DATA_H
#include <half.hpp>
#include <vector>
#include <ostream>

#include "ndp_instruction.h"
#include "common_defs.h"
using namespace half_float;
namespace NDPSim {
static const uint32_t BYTE_BIT =8;
static const uint32_t DOUBLE_SIZE = 8;
static const uint32_t WORD_SIZE = 4;
static const uint32_t WORD_BIT_SIZE = 32;
static const uint32_t UNION_DATA_SIZE = 32;
static const uint32_t DOUBLE_WORD_BIT = 64;
static const uint32_t HALF_SIZE = 2;
static const uint32_t HALF_BIT = 16;
static const uint32_t PACKET_ENTRIES =  PACKET_SIZE / WORD_SIZE;
static const uint32_t CH_STRIDE = 256;

#define ROUND_UP(N, S) ((((N) + (S) - 1) / (S)) * (S))

enum DataType {
  FLOAT16 = 0,
  FLOAT32,
  INT16,
  INT32,
  INT64,
  UINT8,
  CHAR8,
  BOOL,
  VMASK,
  MAX_DataType
};

union Data {
  uint16_t fp16_data[2];
  float fp32_data;
  int16_t int16_data[2];
  int32_t int32_data;
  char char8_data[4];
  uint8_t uint8_data[4];
  bool bool_data[4];
  uint32_t vmask;
};

enum LogicalOp {
  EQ = 0,
  NE,
  LT,
  LE,
  GT,
  GE
};

template <typename T>
bool logical_operate(LogicalOp op, const T& lhs, const T& rhs) {
  switch (op) {
    case EQ:
      return lhs == rhs;
    case NE:
      return lhs != rhs;
    case LT:
      return lhs < rhs;
    case LE:
      return lhs <= rhs;
    case GT:
      return lhs > rhs;
    case GE:
      return lhs >= rhs;
    default:
      return false;
  }
};
class VectorData {
 public:
  VectorData();
  VectorData(bool double_reg);
  VectorData(int sew, float lmul);
  VectorData(Context &ctxt);
  void SetVectorDoubleReg();
  void SetType(DataType type);
  void SetVlen(uint32_t vlen);
  void SetData(float data, int index);
  void SetData(uint8_t data, int index);
  void SetData(int16_t data, int index);
  void SetData(int32_t data, int index);
  void SetData(int64_t data, int index);
  void SetData(half data, int index);
  void SetData(char data, int index);
  void SetData(bool data, int index);
  void SetVmask(bool data, int index);
  bool GetDoubleReg() const { return m_double_reg; }
  void Widen(Context &ctxt);
  DataType GetType() const { return m_type; }
  float GetFloatData(int index) const;
  int64_t GetLongData(int index) const;
  int32_t GetIntData(int index) const;
  int16_t GetShortData(int index) const;
  uint8_t GetU8Data(int index) const;
  half GetHalfData(int index) const;
  char GetCharData(int index) const;
  bool GetBoolData(int index) const;
  int32_t GetVmaskData() const;
  bool GetVmaskData(int index) const;
  uint8_t GetVmaskAsByte(int index) const;
  uint32_t GetVlen() const;
  uint32_t GetPrecision() const;
  size_t GetVectorSize() { return m_data.size(); }
  void Append(VectorData &rhs);
  std::array<VectorData, 2> Split();
  VectorData Max(VectorData &rhs);
  VectorData Min(VectorData &rhs);
  VectorData And(VectorData &rhs);
  VectorData Or(VectorData &rhs);
  VectorData Exp(VectorData &rhs);

  friend VectorData operator+(const VectorData &lhs, VectorData &rhs);
  friend VectorData operator+(const VectorData &lhs, float &rhs);
  friend VectorData operator+(const VectorData &lhs, int32_t &rhs);
  friend VectorData operator+(const VectorData &lhs, int64_t &rhs);
  friend VectorData operator-(const VectorData &lhs, VectorData &rhs);
  friend VectorData operator-(const VectorData &lhs, float &rhs);
  friend VectorData operator-(const VectorData &lhs, int32_t &rhs);
  friend VectorData operator-(const float &lhs, VectorData &rhs);
  friend VectorData operator-(const int32_t &lhs, VectorData &rhs);
  friend VectorData operator*(const VectorData &lhs, VectorData &rhs);
  friend VectorData operator*(const VectorData &lhs, float &rhs);
  friend VectorData operator*(const VectorData &lhs, int32_t &rhs);
  friend VectorData operator/(const VectorData &lhs, VectorData &rhs);
  friend VectorData operator/(const VectorData &lhs, float &rhs);
  friend VectorData operator/(float &lhs, const VectorData &rhs);
  friend VectorData operator/(const VectorData &lhs, int32_t &rhs);
  // friend VectorData& operator=(const VectorData &lhs, VectorData &rhs);

  int ndp_id;

  void PrintVectorData() const;
  std::string toString() const;
  std::ostream &operator<<(std::ostream &os) const {
    os << toString();
    return os;
  };

 private:
  DataType m_type = DataType::MAX_DataType;
  uint32_t m_vlen;
  bool m_double_reg = false;
  std::vector<Data> m_data;
};
}  // namespace NDPSim
#endif