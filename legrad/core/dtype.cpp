#include "dtype.h"
#include "internal/enum_impl.h"

namespace legrad::core
{
constexpr int TYPE_COUNT = ToIntEnum(TypeInfo::COUNT);

// clang-format off
/**
 * - https://github.com/ml-explore/mlx/blob/main/mlx/dtype.cpp
 * - https://jax.readthedocs.io/en/latest/type_promotion.html
 */
constexpr TypeInfo promote_rules[TYPE_COUNT][TYPE_COUNT] = {
  //         Bool,     UInt8,    UInt16,   UInt32,   UInt64,   Int8,     Int16,    Int32,    Int64,    Float32,  Float64
  /* Bool */   {TypeInfo::Bool,    TypeInfo::UInt8,   TypeInfo::UInt16,  TypeInfo::UInt32,  TypeInfo::UInt64,  TypeInfo::Int8,     TypeInfo::Int16,   TypeInfo::Int32,   TypeInfo::Int64,   TypeInfo::Float32, TypeInfo::Float64},
  /* UInt8 */  {TypeInfo::UInt8,   TypeInfo::UInt8,   TypeInfo::UInt16,  TypeInfo::UInt32,  TypeInfo::UInt64,  TypeInfo::Int16,    TypeInfo::Int16,   TypeInfo::Int32,   TypeInfo::Int64,   TypeInfo::Float32, TypeInfo::Float64},
  /* UInt16 */ {TypeInfo::UInt16,  TypeInfo::UInt16,  TypeInfo::UInt16,  TypeInfo::UInt32,  TypeInfo::UInt64,  TypeInfo::Int32,    TypeInfo::Int32,   TypeInfo::Int32,   TypeInfo::Int64,   TypeInfo::Float32, TypeInfo::Float64},
  /* UInt32 */ {TypeInfo::UInt32,  TypeInfo::UInt32,  TypeInfo::UInt32,  TypeInfo::UInt32,  TypeInfo::UInt64,  TypeInfo::Int64,    TypeInfo::Int64,   TypeInfo::Int64,   TypeInfo::Int64,   TypeInfo::Float32, TypeInfo::Float64},
  /* UInt64 */ {TypeInfo::UInt64,  TypeInfo::UInt64,  TypeInfo::UInt64,  TypeInfo::UInt64,  TypeInfo::UInt64,  TypeInfo::Float32,  TypeInfo::Float32, TypeInfo::Float32, TypeInfo::Float32, TypeInfo::Float32, TypeInfo::Float64},
  /* Int8 */   {TypeInfo::Int8,    TypeInfo::Int16,   TypeInfo::Int32,   TypeInfo::Int64,   TypeInfo::Float32, TypeInfo::Int8,     TypeInfo::Int16,   TypeInfo::Int32,   TypeInfo::Int64,   TypeInfo::Float32, TypeInfo::Float64},
  /* Int16 */  {TypeInfo::Int16,   TypeInfo::Int16,   TypeInfo::Int32,   TypeInfo::Int64,   TypeInfo::Float32, TypeInfo::Int16,    TypeInfo::Int16,   TypeInfo::Int32,   TypeInfo::Int64,   TypeInfo::Float32, TypeInfo::Float64},
  /* Int32 */  {TypeInfo::Int32,   TypeInfo::Int32,   TypeInfo::Int32,   TypeInfo::Int64,   TypeInfo::Float32, TypeInfo::Int32,    TypeInfo::Int32,   TypeInfo::Int32,   TypeInfo::Int64,   TypeInfo::Float32, TypeInfo::Float64},
  /* Int64 */  {TypeInfo::Int64,   TypeInfo::Int64,   TypeInfo::Int64,   TypeInfo::Int64,   TypeInfo::Float32, TypeInfo::Int64,    TypeInfo::Int64,   TypeInfo::Int64,   TypeInfo::Int64,   TypeInfo::Float32, TypeInfo::Float64},
  /* Float32*/ {TypeInfo::Float32, TypeInfo::Float32, TypeInfo::Float32, TypeInfo::Float32, TypeInfo::Float32, TypeInfo::Float32,  TypeInfo::Float32, TypeInfo::Float32, TypeInfo::Float32, TypeInfo::Float32, TypeInfo::Float64},
  /* Float64*/ {TypeInfo::Float64, TypeInfo::Float64, TypeInfo::Float64, TypeInfo::Float64, TypeInfo::Float64, TypeInfo::Float64,  TypeInfo::Float64, TypeInfo::Float64, TypeInfo::Float64, TypeInfo::Float64, TypeInfo::Float64}
};

// clang-format on
TypeInfo promote_types(TypeInfo t1, TypeInfo t2)
{
  return promote_rules[ToIntEnum(t1)][ToIntEnum(t2)];
}

TypeKind kindof(TypeInfo t)
{
  static const std::array<TypeKind, TYPE_COUNT> type_kinds = {{
      TypeKind::Bool,  // Bool
      TypeKind::Uint,  // UInt8
      TypeKind::Uint,  // UInt16
      TypeKind::Uint,  // UInt32
      TypeKind::Uint,  // UInt64
      TypeKind::Int,  // Int8
      TypeKind::Int,  // Int16
      TypeKind::Int,  // Int32
      TypeKind::Int,  // Int64
      TypeKind::Float,  // Float32
      TypeKind::Float,  // Float64
  }};

  return type_kinds[ToIntEnum(t)];
}

size_t get_type_size(TypeInfo type)
{
  switch (type) {
    case TypeInfo::Float32:
      return sizeof(float);
    case TypeInfo::Float64:
      return sizeof(double);
    case TypeInfo::Int8:
      return sizeof(int8_t);
    case TypeInfo::Int16:
      return sizeof(int16_t);
    case TypeInfo::Int32:
      return sizeof(int32_t);
    case TypeInfo::Int64:
      return sizeof(int64_t);
    case TypeInfo::UInt8:
      return sizeof(uint8_t);
    case TypeInfo::UInt16:
      return sizeof(uint16_t);
    case TypeInfo::UInt32:
      return sizeof(uint32_t);
    case TypeInfo::UInt64:
      return sizeof(uint64_t);
    case TypeInfo::Bool:
      return sizeof(bool);
    default:
      return 0;
  }
}

}  // namespace legrad::core