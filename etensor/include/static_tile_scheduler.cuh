#pragma once

struct StaticTaskScheduler {
  int task_pos;
  int task_end;
  int task_type;
  const int* task_idx;

  __device__ void init() {
    StaticQueueDesc queue = static_queues[blockIdx.x];
    task_pos = queue.task_begin;
    task_end = queue.task_end;
    task_type = -1;
    task_idx = nullptr;
  }

  __device__ bool valid() {
    if (task_pos >= task_end) {
      return false;
    }

    StaticTaskDesc task = static_tasks[task_pos++];
    task_type = task.task_type;
    task_idx = static_task_indices + task.index_begin;
    return true;
  }

  __device__ int type() const {
    return task_type;
  }

  __device__ const int* indices() const {
    return task_idx;
  }
};
