#include <sstream>
#include <stdexcept>

#include "core/device.h"
#include "macros/log.h"
#include "util/helper_func.h"

namespace legrad::core
{
Device::Device(DeviceType type, DeviceId id)
    : type_(type)
    , id_(id)
{
  check_valid(type_, id);
}

Device::Device(const std::string& device_str)
{
  if (!device_str.empty()) {
    auto [type, idx] = parse_from_str(device_str);
    check_valid(type, idx);
    type_ = type;
    id_ = idx;
  } else {
    // Actually if we pass empty device name, we shouldn't throw an error
    // But we will create a default device and give warning
    LEGRAD_LOG_WARN(
        "Device string shouldn't be empty, create a default CPU device "
        "instead",
        0);
    type_ = DeviceType::CPU;
    id_ = 0;
  }
}

DeviceType Device::type_from_str(const std::string& name)
{
  static const std::array<std::pair<const char*, DeviceType>,
                          static_cast<size_t>(DeviceType::COUNT)>
      device_map = {{{"cpu", DeviceType::CPU},
                     {"cuda", DeviceType::CUDA},
                     {"metal", DeviceType::METAL}}};

  auto iter = std::find_if(
      device_map.begin(), device_map.end(),
      [&name](const std::pair<const char*, DeviceType>& key) -> bool
      { return key.first && legrad::util::to_lower(name) == key.first; });

  // Cant find device name
  if (iter == device_map.end()) {
    LEGRAD_THROW_ERROR(std::runtime_error, "Cannot find device with name {}",
                       name);
  }

  return iter->second;
}

std::pair<DeviceType, DeviceId> Device::parse_from_str(
    const std::string& device)
{
  std::vector<std::string> tokens;
  std::istringstream ss(device);

  for (std::string token; std::getline(ss, token, ':');) {
    tokens.push_back(token);
  }

  if (tokens.size() == 1) {
    // that mean: we only get device name so default is used
    return {type_from_str(tokens[0]), 0};
  }

  if (tokens.size() == 2) {
    size_t device_idx = 0;
    std::stringstream num_stream(tokens[1]);
    num_stream >> device_idx;
    return {type_from_str(tokens[0]), device_idx};
  }

  LEGRAD_THROW_ERROR(std::runtime_error, "Cannot parse device {}", device);
}

void Device::check_valid(DeviceType type, DeviceId id)
{
  if (type == DeviceType::CPU && id != 0) {
    LEGRAD_THROW_ERROR(std::runtime_error,
                       "Only support one CPU device at a time", 0);
  }

  // For other devices, I support only one device
  if (id > 0) {
    LEGRAD_THROW_ERROR(
        std::runtime_error,
        "Only support one device for now (will change this in the future)", 0);
  }
}
};  // namespace legrad::core