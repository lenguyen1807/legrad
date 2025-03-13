/* From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou,
Iain Melvin, Jason Weston) Copyright (c) 2006      Idiap Research Institute
(Samy Bengio) Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert,
Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions by Cruise LLC:
Copyright (c) 2022 Cruise LLC.
All rights reserved.

All contributions by Tri Dao:
Copyright (c) 2024 Tri Dao.
All rights reserved.

All contributions by Arm:
Copyright (c) 2021, 2023-2024 Arm Limited and/or its affiliates

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories
America and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

// legrad: modified from c10::instrusive_ptr
// - simplified implementation

#include <atomic>
#include <climits>
#include <cstddef>

#include "macros/expr.h"
#include "macros/log.h"

namespace pybind11
{
template <typename, typename...>
class class_;
}

namespace legrad::internal
{
class intrusive_ptr_target;

// clang-format off
namespace raw
{
// Create function increase reference for each pointer type (weak and strong)
namespace weak_intrusive_ptr {
   inline void incref(intrusive_ptr_target* self);
}
namespace intrusive_ptr {
   inline void incref(intrusive_ptr_target* self);
}

// Use this to indicate that the constructor dont increase the ref count
struct DontIncreaseRefcount {};
}  // namespace raw

// clang-format on

/** Pytorch comment:
 * intrusive_ptr<T> is an alternative to shared_ptr<T> that has better
 * performance because it does the refcounting intrusively
 * (i.e. in a member of the object itself).
 * Your class T needs to inherit from intrusive_ptr_target to allow it to be
 * used in an intrusive_ptr<T>. Your class's constructor should not allow
 *`this` to escape to other threads or create an intrusive_ptr from `this`.
 */

/** legrad comment:
 * For example, if you do:
 * class Target : intrusive_ptr_target<Target> {
 *   ...
 * };
 * then you can use:
 * intrusive_ptr<Target> ptr_;
 */

/** Pytorch comment:
 * Note [Stack allocated intrusive_ptr_target safety]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * A well known problem with std::enable_shared_from_this is that it
 * allows you to create a std::shared_ptr from a stack allocated object,
 * which is totally bogus because the object will die once you return
 * from the stack.  In intrusive_ptr, we can detect that this has occurred,
 * because we set the refcount/weakcount of objects which inherit from
 * intrusive_ptr_target to zero, *unless* we can prove that the object
 * was dynamically allocated (e.g., via make_intrusive).
 *
 * Thus, whenever you transmute a T* into a intrusive_ptr<T>, we check
 * and make sure that the refcount isn't zero (or, a more subtle
 * test for weak_intrusive_ptr<T>, for which the refcount may validly
 * be zero, but the weak refcount better not be zero), because that
 * tells us if the object was allocated by us.  If it wasn't, no
 * intrusive_ptr for you!
 */

/** legrad comment:
 * Strong References (tracked by refcount_):
 *
 * - Represented by intrusive_ptr<T> instances.
 *
 * - Owning references: A strong reference keeps the object alive. As long as
 * there is at least one strong reference to an object, the object will not be
 * destroyed.
 *
 * - refcount_ stores the number of active intrusive_ptr instances pointing
 * to the object.
 *
 * - Incremented when you create a new intrusive_ptr (copy, move,
 * reclaim_copy, make, unsafe_steal_from_new, unsafe_reclaim_from_nonowning,
 * lock from weak_ptr).
 *
 * - Decremented when an intrusive_ptr goes out of scope or is reset/assigned
 * a new value.
 *
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * Weak References (tracked by weakcount_):
 *
 * - Represented by weak_intrusive_ptr<T> instances.
 *
 * - Non-owning references: A weak reference does not keep the object alive.
 * Weak pointers can observe the object, but their existence alone won't prevent
 * the object from being destroyed if there are no strong references left.
 *
 * - weakcount_ stores the number of active weak_intrusive_ptr instances
 * pointing to the object, plus one if refcount_ > 0. This "+1" is a crucial
 * detail and ensures that the intrusive_ptr_target object itself remains valid
 * as long as there are strong references (so weak pointers can still be
 * upgraded to strong pointers if needed while strong pointers exist).
 *
 * - Incremented when you create a new weak_intrusive_ptr (from
 * intrusive_ptr, copy, move, reclaim_copy).
 *
 * - Decremented when a weak_intrusive_ptr goes out of scope or is
 * reset/assigned a new value.
 */

class LEGRAD_API intrusive_ptr_target
{
  /** Pytorch comment:
   * Note [Weak references for intrusive refcounting]
   * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   * Here's the scheme:
   *
   *  - refcount == number of strong references to the object
   *    weakcount == number of weak references to the object,
   *      plus one more if refcount > 0
   *    An invariant: refcount > 0  =>  weakcount > 0
   *
   *  - c10::StorageImpl stays live as long as there are any strong
   *    or weak pointers to it (weakcount > 0, since strong
   *    references count as a +1 to weakcount)
   *
   *  - finalizers are called and data_ptr is deallocated when refcount == 0
   *
   *  - Once refcount == 0, it can never again be > 0 (the transition
   *    from > 0 to == 0 is monotonic)
   *
   *  - When you access c10::StorageImpl via a weak pointer, you must
   *    atomically increment the use count, if it is greater than 0.
   *    If it is not, you must report that the storage is dead.
   */

  mutable std::atomic<size_t> refcount_;
  mutable std::atomic<size_t> weakcount_;

  template <typename T>
  friend class intrusive_ptr;

  friend inline void raw::intrusive_ptr::incref(intrusive_ptr_target* self);

  template <typename T>
  friend class weak_intrusive_ptr;

  friend inline void raw::weak_intrusive_ptr::incref(
      intrusive_ptr_target* self);

  template <typename T>
  friend struct ExclusivelyOwnedTensorTraits;

protected:
  virtual ~intrusive_ptr_target()
  {
    /** legrad comment:
     * Note that we need disable some warnings (by -W... flags)
     * to throw exception in destructor.
     * Read more in c10/util/intrusive_ptr.h
     */
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#pragma warning( \
    disable : 4297)  // function assumed not to throw an exception but does
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wunknown-warning-option"
#pragma GCC diagnostic ignored "-Wterminate"
#pragma GCC diagnostic ignored "-Wexceptions"
#endif

    /** legrad comment:
     * When destroy object we must ensure:
     * - refcount is 0 (no strong reference to object anymore).
     * - weakcount is 1 or 0 (when refcount_ > 0 we will be sure that weakcount_
     * > 0).
     * - weakcount_ can be greater than 0 even when refcount_ == 0. This happens
     * when only weak references remain after all strong references are gone.
     */

    /** Pytorch comment:
     * Second condition is there to accommodate
     * unsafe_adapt_non_heap_allocated: since we are doing our own
     * deallocation in that case, it is correct for each
     * expected_decref to have happened (some user code tried to
     * decref and thus free the object, but it didn't happen right
     * away) or not (no user code tried to free the object, and
     * now it's getting destroyed through whatever mechanism the
     * caller of unsafe_adapt_non_heap_allocated wanted to
     * use). We choose our reference count such that the count
     * will not dip below INT_MAX regardless.
     */
    LEGRAD_ASSERT(refcount_.load() == 0 || refcount_.load() >= INT_MAX,
                  "Tried to destruct an intrusive_ptr_target that still has "
                  "intrusive_ptr to it; refcount was {}",
                  refcount_.load());

    LEGRAD_ASSERT(weakcount_.load() == 1 || weakcount_.load() == 0
                      || weakcount_.load() == INT_MAX - 1
                      || weakcount_.load() == INT_MAX,
                  "Tried to destruct an intrusive_ptr_target that still has "
                  "weak_intrusive_ptr to it",
                  0);

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#else
#pragma GCC diagnostic pop
#endif
  };

  constexpr intrusive_ptr_target() noexcept
      : refcount_(0)
      , weakcount_(0)
  {
  }

  /* Pytorch comment:
   * intrusive_ptr_target supports copy and move: but refcount and weakcount
   * don't participate (since they are intrinsic properties of the memory
   * location)
   */
  intrusive_ptr_target(intrusive_ptr_target&&) noexcept
      : intrusive_ptr_target()
  {
  }

  intrusive_ptr_target& operator=(intrusive_ptr_target&&) noexcept
  {
    return *this;
  }

  intrusive_ptr_target(const intrusive_ptr_target&) noexcept
      : intrusive_ptr_target()
  {
  }

  intrusive_ptr_target& operator=(const intrusive_ptr_target&) noexcept
  {
    return *this;
  }

private:
  /** Pytorch comment:
   * This is called when refcount reaches zero.
   * You can override this to release expensive resources.
   * There might still be weak references, so your object might not get
   * destructed yet, but you can assume the object isn't used anymore,
   * i.e. no more calls to methods or accesses to members (we just can't
   * destruct it yet because we need the weakcount accessible).
   *
   * If there are no weak references (i.e. your class is about to be
   * destructed), this function WILL NOT be called.
   */
  virtual void release_resources() {};
};

namespace detail
{
/** legrad comment:
 * This class will return a "Null type" to the target type:
 *
 * - So why we need this ? This is a way we can represent when
 * intrusive_target is NULL.
 *
 * - The default is using `nullptr` but you can basically everything
 * for NULL representation, for example you can return -1 and
 * -1 is NULL "state" of intrusive_target
 *
 * - Think of it like having a standardized way to represent "zero" in a
 * numerical system. In most cases, "0" is just "0". But you might allow for
 * different representations of "zero" (e.g., "+0", "-0").
 */
template <class TTarget>
struct intrusive_target_default_null_type final
{
  static constexpr TTarget* singleton() noexcept { return nullptr; }
};

// From the example below, we can assign a
// custom "NULL" representation:
// template <>
// struct intrusive_target_default_null_type<int64_t>
// {
//   static int64_t* singleton() noexcept
//   {
//     return reinterpret_cast<int64_t*>(0x0012FF7C);
//   }
// };

template <class TTarget, class ToNullType, class FromNullType>
TTarget* assign_ptr_(TTarget* rhs)
{
  if (FromNullType::singleton() == rhs) {
    return ToNullType::singleton();
  } else {
    return rhs;
  }
}

template <class TTarget>
TTarget* assign_ptr_no_nulltype(TTarget* rhs)
{
  return rhs;
}

// Increment needs to be acquire-release to make use_count() and
// unique() reliable.
inline size_t atomic_refcount_increment(std::atomic<size_t>& refcount)
{
  return refcount.fetch_add(1, std::memory_order_acq_rel) + 1;
}

// weak_use_count() is only used for testing, so we don't need it to
// be reliable. Relaxed should be fine.
inline size_t atomic_weakcount_increment(std::atomic<size_t>& weakcount)
{
  return weakcount.fetch_add(1, std::memory_order_relaxed) + 1;
}

// Both decrements need to be acquire-release for correctness. See
// e.g. std::shared_ptr implementation.
inline size_t atomic_refcount_decrement(std::atomic<size_t>& refcount)
{
  return refcount.fetch_sub(1, std::memory_order_acq_rel) - 1;
}

inline size_t atomic_weakcount_decrement(std::atomic<size_t>& weakcount)
{
  return weakcount.fetch_sub(1, std::memory_order_acq_rel) - 1;
}
}  // namespace detail

/**
 * But in my source code I will remove NullType to simplified the implementation
 * and see what could happen :>
 */

template <typename TTarget>
class weak_intrusive_ptr;

template <typename TTarget>
class intrusive_ptr final
{
public:
  using element_type = TTarget;

  intrusive_ptr() noexcept
      : intrusive_ptr(nullptr, raw::DontIncreaseRefcount{})
  {
  }

  intrusive_ptr(std::nullptr_t) noexcept
      : intrusive_ptr(nullptr, raw::DontIncreaseRefcount{})
  {
  }

  /** Pytorch comment:
   * This constructor will not increase the ref counter for you.
   * We use the tagged dispatch mechanism to explicitly mark this constructor
   * to not increase the refcount
   */
  explicit intrusive_ptr(TTarget* target, raw::DontIncreaseRefcount) noexcept
      : target_(target)
  {
  }

  explicit intrusive_ptr(std::unique_ptr<TTarget> rhs) noexcept
      : intrusive_ptr(rhs.release())
  {
  }

  intrusive_ptr(intrusive_ptr&& rhs) noexcept
      : target_(rhs.target_)
  {
    rhs.target_ = nullptr;
  }

  template <typename From>
  intrusive_ptr(intrusive_ptr<From>&& rhs) noexcept
      : target_(detail::assign_ptr_no_nulltype(rhs))
  {
    static_assert(std::is_convertible_v<From*, TTarget*>,
                  "Type mismatch. intrusive_ptr move constructor got pointer "
                  "of wrong type.");
    rhs.target_ = nullptr;
  }

  template <typename From>
  intrusive_ptr& operator=(intrusive_ptr<From>&& rhs) & noexcept
  {
    static_assert(std::is_convertible_v<From*, TTarget*>,
                  "Type mismatch. intrusive_ptr move assignment got pointer of "
                  "wrong type.");
    intrusive_ptr tmp = std::move(rhs);
    swap_intrusive_ptr(tmp);
    return *this;
  }

  intrusive_ptr& operator=(intrusive_ptr&& rhs) & noexcept
  {
    return operator= <TTarget>(std::move(rhs));
  }

  intrusive_ptr(const intrusive_ptr& rhs)
      : target_(rhs.target_)
  {
    retain_();
  }

  template <typename From>
  intrusive_ptr& operator=(const intrusive_ptr<From>& rhs) &
  {
    static_assert(std::is_convertible_v<From*, TTarget*>,
                  "Type mismatch. intrusive_ptr copy assignment got pointer of "
                  "wrong type.");
    intrusive_ptr tmp = rhs;
    swap_intrusive_ptr(tmp);
    return *this;
  }

  intrusive_ptr& operator=(const intrusive_ptr& rhs) & noexcept
  {
    return operator= <TTarget>(rhs);
  }

  // clang-format off
  TTarget* get() const noexcept 
  { 
    return target_; 
  }

  TTarget& operator*() const noexcept 
  { 
    return *target_; 
  }

  TTarget* operator->() const noexcept
  {
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDelete)
    return target_;
  }

  operator bool() const noexcept 
  { 
    return target_ != nullptr; 
  }

  bool unique() const noexcept 
  { 
    return use_count() == 1;
  }

  // We do a lot of null-pointer checks in our code, good to have this be cheap.
  bool defined() const noexcept 
  { 
    return target_ != nullptr; 
  }

  // clang-format on
  void reset() noexcept
  {
    reset_();
    target_ = nullptr;
  }

  void swap_intrusive_ptr(intrusive_ptr& rhs) noexcept
  {
    TTarget* tmp = target_;
    target_ = rhs.target_;
    rhs.target_ = tmp;
  }

  size_t use_count() const noexcept
  {
    if (target_ == nullptr) {
      return 0;
    }
    return target_->refcount_.load(std::memory_order_acquire);
  }

  size_t weak_use_count() const noexcept
  {
    if (target_ == nullptr) {
      return 0;
    }
    return target_->weakcount_.load(std::memory_order_acquire);
  }

  /**
   * Returns an owning (!) pointer to the underlying object and makes the
   * intrusive_ptr instance invalid. That means the refcount is not decreased.
   * You *must* put the returned pointer back into a intrusive_ptr using
   * intrusive_ptr::reclaim(ptr) to properly destruct it.
   * This is helpful for C APIs.
   */
  TTarget* release() noexcept
  {
    // NOLINTNEXTLINE(clang-analyzer-core.uninitialized.Assign)
    TTarget* result = target_;
    target_ = nullptr;
    return result;
  }

  /**
   * Takes an owning pointer to TTarget* and creates an intrusive_ptr that takes
   * over ownership. That means the refcount is not increased.
   * This is the counter-part to intrusive_ptr::release() and the pointer
   * passed in *must* have been created using intrusive_ptr::release().
   */
  static intrusive_ptr reclaim(TTarget* owning_ptr)
  {
    LEGRAD_ASSERT(
        owning_ptr == nullptr || owning_ptr->refcount_.load() == 0
            || owning_ptr->weakcount_.load(),
        "TTarget violates the invariant that refcount > 0  =>  weakcount > 0",
        0);
    return intrusive_ptr(owning_ptr, raw::DontIncreaseRefcount{});
  }

  /**
   * Takes an owning pointer to TTarget* and creates an intrusive_ptr
   * representing a new reference, i.e. the raw pointer retains
   * ownership.
   */
  static intrusive_ptr reclaim_copy(TTarget* owning_ptr)
  {
    auto ret = reclaim(owning_ptr);
    ret.retain_();
    return ret;
  }

  /**
   * Allocate a heap object with args and wrap it inside a intrusive_ptr and
   * incref. This is a helper function to let make_intrusive() access private
   * intrusive_ptr constructors.
   */
  template <class... Args>
  static intrusive_ptr make(Args&&... args)
  {
    return intrusive_ptr(new TTarget(std::forward<Args>(args)...));
  }

  /**
   * Turn a new instance of TTarget (e.g., literally allocated
   * using new TTarget(...) into an intrusive_ptr.  If possible,
   * use intrusive_ptr::make instead which statically guarantees
   * that the allocation was done properly.
   *
   * At the moment, the only reason this method exists is because
   * pybind11 holder types expect to be able to allocate in
   * this way (because pybind11 handles the new allocation itself).
   */
  static intrusive_ptr unsafe_steal_from_new(TTarget* raw_ptr)
  {
    return intrusive_ptr(raw_ptr);
  }

  /**
   * Turn an instance of TTarget that should not be reference counted
   * (e.g., allocated into an arena with placement new) into an
   * intrusive_ptr. This is gratuitously unsafe and should only be
   * used if you can guarantee that the pointer will not escape and be
   * refcounted as normal.
   *
   * `expected_decrefs` is a debugging parameter: it indicates the
   * number of strong owners the intrusive_ptr_target in question is
   * expected to get. In most use cases, this will likely be 1.
   *
   * The reason this method exists is for manually sharing
   * StorageImpls across Tensors in the static runtime. It needs
   * access to private intrusive_ptr members so that the refcounts can
   * be initialized to custom values.
   */
  static intrusive_ptr unsafe_adapt_non_heap_allocated(TTarget* raw_ptr,
                                                       size_t expected_decrefs)
  {
    intrusive_ptr result(raw_ptr, raw::DontIncreaseRefcount{});
    // INT_MAX is impractically huge for a reference count, while
    // being in no danger of overflowing size_t. We actually only need to
    // initialize the refcount to 2 -- we are just doing an unbalanced
    // incref to prevent the non-heap-allocated target from being
    // freed, and we are optimizing that incref by directly
    // initializing the refcounts rather than doing an expensive
    // atomic increment. The reason to use INT_MAX is to accommodate
    // the debug assertions in ~intrusive_ptr_target.
#ifdef NDEBUG
    expected_decrefs = 0;
#endif
    result.target_->refcount_.store(INT_MAX + expected_decrefs,
                                    std::memory_order_relaxed);
    result.target_->weakcount_.store(INT_MAX, std::memory_order_relaxed);
    return result;
  }

  /** Pytorch comment
   * Turn a **non-owning raw pointer** to an intrusive_ptr.  It is
   * the moral equivalent of enable_shared_from_this on a shared pointer.
   *
   * This method is only valid for objects that are already live.  If
   * you are looking for the moral equivalent of unique_ptr<T>(T*)
   * constructor, see steal_from_new.
   *
   * TODO: https://github.com/pytorch/pytorch/issues/56482
   */
  static intrusive_ptr unsafe_reclaim_from_nonowning(TTarget* raw_ptr)
  {
    // See Note [Stack allocated intrusive_ptr_target safety]
    LEGRAD_ASSERT(
        raw_ptr == nullptr || raw_ptr->refcount_.load() > 0,
        "intrusive_ptr: Can only reclaim pointers that are owned by someone",
        0);
    auto ptr = reclaim(raw_ptr);  // doesn't increase refcount
    ptr.retain_();
    return ptr;
  }

private:
  void retain_()
  {
    if (target_ != nullptr) {
      size_t new_refcount =
          detail::atomic_refcount_increment(target_->refcount_);
      LEGRAD_ASSERT(
          new_refcount != 1,
          "intrusive_ptr: Cannot increase refcount after it reached zero.", 0);
    }
  }

  void reset_() noexcept
  {
    if (target_ != nullptr
        && detail::atomic_refcount_decrement(target_->refcount_) == 0)
    {
      // See comment above about weakcount. As long as refcount>0,
      // weakcount is one larger than the actual number of weak references.
      // So we need to decrement it here.
      bool should_delete =
          target_->weakcount_.load(std::memory_order_acquire) == 1;
      if (!should_delete) {
        // justification for const_cast: release_resources is basically a
        // destructor and a destructor always mutates the object, even for const
        // objects. NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        const_cast<std::remove_const_t<TTarget>*>(target_)->release_resources();
        should_delete =
            detail::atomic_weakcount_decrement(target_->weakcount_) == 0;
      }
      if (should_delete) {
        delete target_;
      }
    }
  }

  /* Pytorch comment:
   * - raw pointer constructors are not public because we shouldn't make
   * intrusive_ptr out of raw pointers except from inside the make_intrusive(),
   * reclaim() and weak_intrusive_ptr::lock() implementations.
   * - This constructor will increase the ref counter for you.
   * - This constructor will be used by the make_intrusive(), and also pybind11,
   * which wrap the intrusive_ptr holder around the raw pointer and incref
   * correspondingly (pybind11 requires raw pointer constructor to incref by
   * default).
   */
  explicit intrusive_ptr(TTarget* target)
      : intrusive_ptr(target, raw::DontIncreaseRefcount{})
  {
    if (target_ != nullptr) {
      // Pytorch comment:
      // We just created result.target_, so we know no other thread has
      // access to it, so we know we needn't care about memory ordering.
      // (On x86_64, a store with memory_order_relaxed generates a plain old
      // `mov`, whereas an atomic increment does a lock-prefixed `add`, which is
      // much more expensive: https://godbolt.org/z/eKPzj8.)
      LEGRAD_ASSERT(
          target_->refcount_ == 0 && target_->weakcount_ == 0,
          "intrusive_ptr: Newly-created target had non-zero refcounts. Does "
          "its constructor do something strange like incref or create an "
          "intrusive_ptr from `this`?",
          0);
      target_->refcount_.store(1, std::memory_order_relaxed);
      target_->weakcount_.store(1, std::memory_order_relaxed);
    }
  }

private:
  TTarget* target_;

  template <typename T>
  friend struct ExclusivelyOwnedTensorTraits;

  template <class TTarget2>
  friend class intrusive_ptr;

  friend class weak_intrusive_ptr<TTarget>;

  /*
   * Make pybind11::class_ be a friend class of intrusive_ptr, so that custom
   * smart holder in pybind11 could access the private constructor of
   * intrusive_ptr(T*) which took the ownership of the object. This is required
   * by customer holder macro PYBIND11_DECLARE_HOLDER_TYPE, where it uses
   * intrusive_ptr(TTarget*) to initialize and take ownership of the object. For
   * details, see
   * https://pybind11.readthedocs.io/en/stable/advanced/smart_ptrs.html#custom-smart-pointers
   */
  template <typename, typename...>
  friend class pybind11::class_;
};

template <class TTarget, class... Args>
inline intrusive_ptr<TTarget> make_intrusive(Args&&... args)
{
  return intrusive_ptr<TTarget>::make(std::forward<Args>(args)...);
}

template <class TTarget>
inline void swap(intrusive_ptr<TTarget>& lhs,
                 intrusive_ptr<TTarget>& rhs) noexcept
{
  lhs.swap_intrusive_ptr(rhs);
}

// To allow intrusive_ptr inside std::map or std::set, we need operator<
template <class TTarget1, class TTarget2>
inline bool operator<(const intrusive_ptr<TTarget1>& lhs,
                      const intrusive_ptr<TTarget2>& rhs) noexcept
{
  return lhs.get() < rhs.get();
}

template <class TTarget1, class TTarget2>
inline bool operator==(const intrusive_ptr<TTarget1>& lhs,
                       const intrusive_ptr<TTarget2>& rhs) noexcept
{
  return lhs.get() == rhs.get();
}

template <class TTarget1>
inline bool operator==(const intrusive_ptr<TTarget1>& lhs,
                       std::nullptr_t) noexcept
{
  return lhs.get() == nullptr;
}

template <class TTarget1>
inline bool operator==(std::nullptr_t,
                       const intrusive_ptr<TTarget1>& rhs) noexcept
{
  return nullptr == rhs.get();
}

template <class TTarget1, class TTarget2>
inline bool operator!=(const intrusive_ptr<TTarget1>& lhs,
                       const intrusive_ptr<TTarget2>& rhs) noexcept
{
  return !(lhs == rhs);
}

template <class TTarget1>
inline bool operator!=(const intrusive_ptr<TTarget1>& lhs,
                       std::nullptr_t) noexcept
{
  return !(lhs == nullptr);
}

template <class TTarget1>
inline bool operator!=(std::nullptr_t,
                       const intrusive_ptr<TTarget1>& rhs) noexcept
{
  return !(rhs == nullptr);
}

template <typename Pointer>
struct MaybeOwnedTraits;

template <typename T>
struct MaybeOwnedTraits<intrusive_ptr<T>>
{
  using owned_type = intrusive_ptr<T>;
  using borrow_type = intrusive_ptr<T>;

  static borrow_type createBorrow(const owned_type& from)
  {
    return borrow_type::reclaim(from.get());
  }

  static void assignBorrow(borrow_type& lhs, const borrow_type& rhs)
  {
    lhs.release();
    lhs = borrow_type::reclaim(rhs.get());
  }

  static void destroyBorrow(borrow_type& toDestroy) { toDestroy.release(); }

  static const owned_type& referenceFromBorrow(const borrow_type& borrow)
  {
    return borrow;
  }

  static const owned_type* pointerFromBorrow(const borrow_type& borrow)
  {
    return &borrow;
  }

  static bool debugBorrowIsValid(const borrow_type& /*borrow*/) { return true; }
};

}  // namespace legrad::internal
