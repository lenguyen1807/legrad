#include <cstdlib>
#include <mutex>
#include <new>
#include <stdexcept>

#include "allocator.h"
#include "core/device.h"
#include "macros/log.h"

namespace legrad::core
{
CommonAllocator::~CommonAllocator()
{
  // Free memory in available_pool_ (cached memory)
  free_cached();

  // Free memory still tracked in allocation_map_ (memory not yet returned)
  for (auto const& [ptr, size] : allocation_map_) {
    std::free(ptr);
  }
  allocation_map_.clear();
}

void* CommonAllocator::allocate_and_throw(size_t nbytes)
{
  void* ptr = nullptr;
  // If the size is multiple of aligment default size
  // We use aligned alloc, alligned memory is sometimes
  // better than normal memory
  if (nbytes % allocator::MEMORY_ALIGNMENT_SIZE == 0) {
    ptr = std::aligned_alloc(allocator::MEMORY_ALIGNMENT_SIZE, nbytes);
  } else {
    ptr = std::malloc(nbytes);
  }
  if (ptr == nullptr) {
    LEGRAD_LOG_ERR("Cannot allocate memory with size: {}", nbytes)
    throw std::bad_alloc();
  }
  return ptr;
}

void CommonAllocator::free_cached()
{
  for (auto const& [size, ptr] : available_pool_) {
    std::free(ptr);
  }
  available_pool_.clear();
}

Buffer CommonAllocator::allocate(size_t nbytes)
{
  std::lock_guard<std::mutex> lock(mtx_);

  void* ptr = nullptr;
  CPUContext* ctx = nullptr;

  if (nbytes != 0) {
    // We find if memory pool already has the memory with size we want
    auto it = available_pool_.lower_bound(nbytes);

    if (it != available_pool_.end()) {
      ptr = it->second;
      // If we can find memory from pool
      // We must make sure it is not null
      LEGRAD_ASSERT(ptr != nullptr, "Memory from pool cannot be null", 0);
      // Then delete memory from pool
      available_pool_.erase(it);
    } else {
      // If we cant find the memory, allocate new one
      try {
        ptr = allocate_and_throw(nbytes);
      } catch (...) {
        // But before we try to do anything else
        // There maybe out of memory because of caching
        // so we need to free all cache
        free_cached();
        // Then allocate again
        LEGRAD_LOG_WARN(
            "Try to allocate memory with size {} again (freeing all cached).",
            nbytes)
        ptr = allocate_and_throw(nbytes);
      }
      allocation_map_.insert({ptr, nbytes});
    }
    // Create new context
    ctx = new CPUContext{ptr, nbytes, this};
  }

  return Buffer(ptr, ctx, CommonAllocator::deallocate,
                Device(DeviceType::CPU, 0));
}

void CommonAllocator::return_mem(void* ptr)
{
  std::lock_guard<std::mutex> loc(mtx_);

  if (ptr == nullptr) {
    LEGRAD_THROW_ERROR(std::invalid_argument, "return_mem called with nullptr",
                       0);
    return;
  }

  auto iter = allocation_map_.find(ptr);
  if (iter == allocation_map_.end()) {
    LEGRAD_THROW_ERROR(
        std::runtime_error,
        "return_mem called for pointer not managed by this allocator: {}", ptr);
    return;
  }
  size_t size = iter->second;

  allocation_map_.erase(ptr);
  // Return memory to pool
  available_pool_.insert({size, ptr});
}

void CommonAllocator::deallocate(void* ctx)
{
  if (ctx == nullptr) {
    return;
  }
  CPUContext* cpu_ctx = static_cast<CPUContext*>(ctx);
  if (cpu_ctx->allocator == nullptr) {
    // Note that we still try to delete the context of pointer
    delete cpu_ctx;
    LEGRAD_THROW_ERROR(std::runtime_error,
                       "The context pointer has empty allocator", 0);
  }
  cpu_ctx->allocator->return_mem(cpu_ctx->ptr);
  delete cpu_ctx;
}
};  // namespace legrad::core