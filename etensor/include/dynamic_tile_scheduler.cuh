#pragma once

#include <cuda_runtime.h>

__device__ int dynamic_set_index_pos[TASK_SET_COUNT];
__device__ int dynamic_ready_queue[READY_QUEUE_CAPACITY];
__device__ int dynamic_queue_head;
__device__ int dynamic_queue_tail;
__device__ int dynamic_queue_lock;
__device__ int dynamic_tiles_done;

__device__ inline void dynamic_lock_queue() {
  while (atomicCAS(&dynamic_queue_lock, 0, 1) != 0) {
    __nanosleep(64);
  }
}

__device__ inline void dynamic_unlock_queue() {
  atomicExch(&dynamic_queue_lock, 0);
}

__device__ inline void push_task_set(int set_id) {
  if (set_id < 0) {
    return;
  }
  dynamic_lock_queue();
  int pos = dynamic_queue_tail++;
  if (pos < READY_QUEUE_CAPACITY) {
    dynamic_ready_queue[pos] = set_id;
  }
  dynamic_unlock_queue();
}

__device__ inline bool pop_task_set(int* set_id) {
  bool found = false;
  dynamic_lock_queue();
  if (dynamic_queue_head < dynamic_queue_tail) {
    *set_id = dynamic_ready_queue[dynamic_queue_head++];
    found = true;
  }
  dynamic_unlock_queue();
  return found;
}

__device__ inline bool dynamic_scheduler_done() {
  return atomicAdd(&dynamic_tiles_done, 0) >= TOTAL_TILE_COUNT;
}

__device__ inline bool pop_tile_from_set(
    int set_id,
    int* task_type,
    const int** task_idx) {
  TaskSetDesc set = task_sets[set_id];
  int index_len = set.index_end - set.index_begin;
  int index_pos = atomicAdd(&dynamic_set_index_pos[set_id], set.index_rank);
  if (index_pos >= index_len) {
    if (index_pos == index_len) {
      push_task_set(set.next_set_id);
    }
    return false;
  }

  *task_type = set.task_type;
  *task_idx = dynamic_task_indices + set.index_begin + index_pos;
  return true;
}

__device__ inline bool pop_tile(int* active_set, int* task_type, const int** task_idx) {
  while (!dynamic_scheduler_done()) {
    if (*active_set >= 0) {
      if (pop_tile_from_set(*active_set, task_type, task_idx)) {
        return true;
      }
      *active_set = -1;
    }

    int set_id = -1;
    if (pop_task_set(&set_id)) {
      *active_set = set_id;
      continue;
    }

    __nanosleep(64);
  }
  return false;
}

struct TaskScheduler {
  int active_set;
  int task_type;
  const int* task_idx;
  int has_task;

  __device__ void init() {
    if (threadIdx.x == 0) {
      active_set = -1;
      task_type = -1;
      task_idx = nullptr;
      has_task = 0;
    }
    __syncthreads();
  }

  __device__ bool valid() {
    if (threadIdx.x == 0) {
      has_task = pop_tile(&active_set, &task_type, &task_idx) ? 1 : 0;
      if (has_task) {
        atomicAdd(&dynamic_tiles_done, 1);
      }
    }
    __syncthreads();
    return has_task != 0;
  }

  __device__ int type() const {
    return task_type;
  }

  __device__ const int* indices() const {
    return task_idx;
  }
};
