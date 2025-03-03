#pragma once

#include <cstddef>
#include <map>
#include <unordered_map>

#include "core/buffer.h"

namespace legrad::core
{
namespace allocator
{
// clang-format off
#ifdef LEGRAD_USE_ARM
  // 16 bytes is sufficent for ARM
  constexpr size_t MEMORY_ALIGNMENT_SIZE = 16;
#else
  // Use 64-byte alignment should be enough for computation up to AVX512.
  constexpr size_t MEMORY_ALIGNMENT_SIZE = 64;
#endif
}

class Allocator
{
public:
  virtual ~Allocator() = default;
  virtual Buffer allocate(size_t) = 0;
};

/*
* This is allocator for CPU
*/
class CommonAllocator : public Allocator
{
  struct CPUContext
  {
    void* ptr;
    size_t size;
    // We want to know the allocator that allocate the buffer
    CommonAllocator* allocator;
  };

public:
  CommonAllocator() = default;
  ~CommonAllocator();

  Buffer allocate(size_t) override;
  static void deallocate(void*);
  void return_mem(void*);

private:
  void free_cached();
  void* allocate_and_throw(size_t);

  std::mutex mtx_;
  std::multimap<size_t, void*> available_pool_;
  std::unordered_map<void*, size_t> allocation_map_;
};
}  // namespace legrad::core