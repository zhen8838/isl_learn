#pragma once

#include <cuda/atomic>
#include <cuda_runtime.h>

namespace etensor {

__device__ __forceinline__ int load_acquire(const int* ptr) {
  cuda::atomic_ref<int, cuda::thread_scope_device> ref(*const_cast<int*>(ptr));
  return ref.load(cuda::memory_order_acquire);
}

__device__ __forceinline__ void store_release(int* ptr, int value) {
  cuda::atomic_ref<int, cuda::thread_scope_device> ref(*ptr);
  ref.store(value, cuda::memory_order_release);
}

__device__ __forceinline__ void notify(int* counter) {
  cuda::atomic_ref<int, cuda::thread_scope_device> ref(*counter);
  ref.fetch_sub(1, cuda::memory_order_release);
}

__device__ __forceinline__ bool notify_and_ready(int* counter) {
  cuda::atomic_ref<int, cuda::thread_scope_device> ref(*counter);
  int old = ref.fetch_sub(1, cuda::memory_order_release);
  return old == 1;
}

__device__ __forceinline__ void wait(int* counter) {
  while (load_acquire(counter) != 0) {
    __nanosleep(64);
  }
}

}  // namespace etensor
