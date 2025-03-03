#pragma once

#include <memory>
#include <unordered_map>

#include "core/allocator.h"
#include "core/device.h"
#include "internal/pattern.h"
#include "macros/log.h"
#include "util/hash_key.h"

namespace legrad::core
{
class AllocatorMgr final : public legrad::internal::Singleton<AllocatorMgr>
{
public:
  AllocatorMgr()
  {
    LEGRAD_LOG_DEBUG("Create AllocatorMgr", 0);
    allocator_map_[{DeviceType::CPU, 0}] = std::make_unique<CommonAllocator>();
  }

  ~AllocatorMgr() { LEGRAD_LOG_DEBUG("Destroy AllocatorMgr", 0); }

  Allocator* get(DeviceType type, DeviceId id = 0)
  {
    auto it = allocator_map_.find({type, id});
    if (it == allocator_map_.end()) {
      LEGRAD_LOG_WARN("Device {} with index {} is not set",
                      DeviceTypeToString(type), id);
      return nullptr;
    }
    return it->second.get();
  }

private:
  std::unordered_map<std::pair<DeviceType, DeviceId>,
                     std::unique_ptr<Allocator>,
                     legrad::util::HashPairKey>
      allocator_map_;
};
};  // namespace legrad::core