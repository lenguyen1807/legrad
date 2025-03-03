#pragma once

#include "internal/enum_impl.h"

namespace legrad::core
{
LEGRAD_ENUM(TypeInfo,
            uint8_t,  // type of Enum
            Bool,  // Begin of Enum
            Float64,  // End of Enum
            Bool,
            UInt8,
            UInt16,
            UInt32,
            UInt64,
            Int8,
            Int16,
            Int32,
            Int64,
            Float32,
            Float64,
            COUNT)
LEGRAD_ENUM(TypeKind, uint8_t, Bool, Float, Bool, Uint, Int, Float, COUNT)

// clang-format off
#define CALL_DISPATCH_TYPE_INFO(TYPE, ...)          \
    [&] {                                           \
        switch (TYPE) {                             \
            case TypeInfo::Float32: {               \
                using scalar_t = float;             \
                return __VA_ARGS__();               \
            }                                       \
            case TypeInfo::Float64: {               \
                using scalar_t = double;            \
                return __VA_ARGS__();               \
            }                                       \
            case TypeInfo::Int8: {                  \
                using scalar_t = int8_t;            \
                return __VA_ARGS__();               \
            }                                       \
            case TypeInfo::Int16: {                 \
                using scalar_t = int16_t;           \
                return __VA_ARGS__();               \
            }                                       \
            case TypeInfo::Int32: {                 \
                using scalar_t = int32_t;           \
                return __VA_ARGS__();               \
            }                                       \
            case TypeInfo::Int64: {                 \
                using scalar_t = int64_t;           \
                return __VA_ARGS__();               \
            }                                       \
            case TypeInfo::UInt8: {                 \
                using scalar_t = uint8_t;           \
                return __VA_ARGS__();               \
            }                                       \
            case TypeInfo::UInt16: {                \
                using scalar_t = uint16_t;          \
                return __VA_ARGS__();               \
            }                                       \
            case TypeInfo::UInt32: {                \
                using scalar_t = uint32_t;          \
                return __VA_ARGS__();               \
            }                                       \
            case TypeInfo::UInt64: {                \
                using scalar_t = uint64_t;          \
                return __VA_ARGS__();               \
            }                                       \
            case TypeInfo::Bool: {                  \
                using scalar_t = bool;              \
                return __VA_ARGS__();               \
            }                                       \
            default:                                \
                LNDL_THROW_ERROR(std::runtime_error,\
                "Unsupported TypeInfo");            \
        }                                           \
    }()

// clang-format on
template <typename T>
inline TypeInfo info_from_ctype()
{
  if constexpr (std::is_same_v<T, float>)
    return TypeInfo::Float32;
  if constexpr (std::is_same_v<T, double>)
    return TypeInfo::Float64;
  if constexpr (std::is_same_v<T, int8_t>)
    return TypeInfo::Int8;
  if constexpr (std::is_same_v<T, int16_t>)
    return TypeInfo::Int16;
  if constexpr (std::is_same_v<T, int32_t>)
    return TypeInfo::Int32;
  if constexpr (std::is_same_v<T, int64_t>)
    return TypeInfo::Int64;
  if constexpr (std::is_same_v<T, uint8_t>)
    return TypeInfo::UInt8;
  if constexpr (std::is_same_v<T, uint16_t>)
    return TypeInfo::UInt16;
  if constexpr (std::is_same_v<T, uint32_t>)
    return TypeInfo::UInt32;
  if constexpr (std::is_same_v<T, uint64_t>)
    return TypeInfo::UInt64;
  if constexpr (std::is_same_v<T, bool>)
    return TypeInfo::Bool;
  LEGRAD_THROW_ERROR(std::invalid_argument, "Do not support type", 0);
}

TypeKind kindof(TypeInfo t);
TypeInfo promote_types(TypeInfo t1, TypeInfo t2);
size_t get_type_size(TypeInfo type);

struct DataType
{
  TypeInfo info;
  TypeKind kind;
  size_t size;

  DataType()
      : info(TypeInfo::Float32)
      , kind(kindof(info))
      , size(get_type_size(info))
  {
  }

  DataType(TypeInfo info)
      : info(info)
      , kind(kindof(info))
      , size(get_type_size(info))
  {
  }

  std::string name() const { return TypeInfoToString(info); }

  template <typename T>
  static DataType from_ctype()
  {
    return DataType(info_from_ctype<T>());
  }

  static bool is_floating_point(DataType type)
  {
    return type.kind == TypeKind::Float;
  }

  static bool is_bool(DataType type) { return type.kind == TypeKind::Bool; }

  static bool is_signed_integer(DataType type)
  {
    return type.kind == TypeKind::Int;
  }

  static bool is_unsigned_integer(DataType type)
  {
    return type.kind == TypeKind::Uint;
  }
};

}  // namespace legrad::core