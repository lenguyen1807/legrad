#pragma once

#include <cstdint>

#include "internal/enum_impl.h"

namespace legrad::core
{
LEGRAD_ENUM(DeviceType, uint8_t, CPU, CUDA, CPU, METAL, VULKAN, CUDA, COUNT)

/*
 * In Pytorch device start from -1 (which is default device)
 * But we will start from 0 (which is also a default device)
 * NOTE: I don't know how to support multiple devices but this is kind of
 * placeholder to extend in the future
 */
using DeviceId = size_t;

class Device
{
public:
  Device() = default;
  Device(DeviceType type, DeviceId id);

  // Use this constructor to create device instead of above
  explicit Device(const std::string& device_str);

  // Actually I can let C++ do this
  // But I want to explicitly show
  LEGRAD_DEFAULT_COPY_AND_ASSIGN(Device);
  LEGRAD_DEFAULT_MOVE_AND_ASSIGN(Device);

  bool operator==(const Device& other) const noexcept
  {
    return this->type_ == other.type_ && this->id_ == other.id_;
  }

  bool operator!=(const Device& other) const noexcept
  {
    return !(*this == other);
  }

  DeviceType type() const noexcept { return type_; }

  DeviceId index() const noexcept { return id_; }

  bool is_cpu() const { return type_ == DeviceType::CPU; }

  std::string str() const
  {
    std::string device_name = DeviceTypeToString(type_);
    device_name += (":" + std::to_string(id_));
    return device_name;
  }

private:
  DeviceType type_ = DeviceType::COUNT;
  DeviceId id_ = 0;

  DeviceType type_from_str(const std::string& name);
  std::pair<DeviceType, DeviceId> parse_from_str(const std::string& device);
  void check_valid(DeviceType type, DeviceId idx);
};
};  // namespace legrad::core