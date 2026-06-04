import isl


N = 4
K_SPLIT = 4
WORKERS = 2
TOTAL_TASKS = N * (K_SPLIT + 1)
STEPS = (TOTAL_TASKS + WORKERS - 1) // WORKERS


task_to_time = isl.union_map(
    f"{{ "
    f"P[i, j] -> Time[i, 0, j] : 0 <= i < {N} and 0 <= j < {K_SPLIT}; "
    f"F[i] -> Time[i, 1, 0] : 0 <= i < {N} "
    f"}}"
)

time_to_linear = isl.map(
    f"{{ Time[i, phase, j] -> T[t] : "
    f"t = i * {K_SPLIT + 1} + phase * {K_SPLIT} + j "
    f"}}"
)

linear_to_resource = isl.map(
    f"{{ T[t] -> R[worker, step] : "
    f"t = step * {WORKERS} + worker and 0 <= worker < {WORKERS} "
    f"}}"
)

task_to_resource = task_to_time.apply_range(time_to_linear).apply_range(
    linear_to_resource
)
resource_to_task = task_to_resource.reverse()

print("task_to_time:")
print(task_to_time)
print()

print("time_to_linear:")
print(time_to_linear)
print()

print("linear_to_resource:")
print(linear_to_resource)
print()

print("task_to_resource = task_to_time . time_to_linear . linear_to_resource:")
print(task_to_resource)
print()

print("resource_to_task = reverse(task_to_resource):")
print(resource_to_task)
print()

for worker in range(WORKERS):
  for step in range(STEPS):
    worker_domain = isl.union_set(f"{{ R[{worker}, {step}] }}")
    print(f"worker {worker} queue relation:")
    print(resource_to_task.intersect_domain(worker_domain))
    print()
