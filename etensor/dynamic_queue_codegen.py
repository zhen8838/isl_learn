from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from etensor import codegen
from etensor import static_queue_codegen


@dataclass(frozen=True)
class DynamicTaskSet:
    task_kind: int
    instances: tuple[static_queue_codegen.TaskInstance, ...]
    next_set_id: int = -1


@dataclass(frozen=True)
class DynamicScheduledGraph:
    source: codegen.TaskGraph
    events: tuple[codegen.EventInfo, ...]
    worker_count: int
    task_kinds: dict[str, int]
    task_index_ranks: tuple[int, ...]
    task_sets: tuple[DynamicTaskSet, ...]
    initial_ready_sets: tuple[int, ...]


@dataclass(frozen=True)
class CudaModuleModel:
    device_functions: str
    task_macros: str
    task_types: tuple[static_queue_codegen.TaskTypeModel, ...]
    arrays: tuple[static_queue_codegen.CArrayModel, ...]
    task_set_count: int
    ready_queue_capacity: int
    total_tile_count: int
    kernel_signature: str
    task_undefs: str
    host_params: str
    host_body_lines: tuple[str, ...]


def dynamic_schedule(
    graph: codegen.TaskGraph,
    worker_count: int,
) -> DynamicScheduledGraph:
    events = codegen.derive_event_infos(graph)
    if len(events) != 1:
        raise ValueError("dynamic v1 expects exactly one event dependence")

    instances = static_queue_codegen.enumerate_task_instances_by_statement(graph)
    task_kinds = {name: i for i, name in enumerate(graph.statements)}
    task_index_ranks = tuple(
        len(statement.indices) for statement in graph.statements.values()
    )
    event = events[0]
    task_sets = (
        DynamicTaskSet(
            task_kind=task_kinds[event.producer],
            instances=instances[event.producer],
            next_set_id=1,
        ),
        DynamicTaskSet(
            task_kind=task_kinds[event.consumer],
            instances=instances[event.consumer],
        ),
    )

    return DynamicScheduledGraph(
        source=graph,
        events=events,
        worker_count=worker_count,
        task_kinds=task_kinds,
        task_index_ranks=task_index_ranks,
        task_sets=task_sets,
        initial_ready_sets=(0,),
    )


def build_scheduler_arrays(plan: DynamicScheduledGraph) -> tuple[static_queue_codegen.CArrayModel, ...]:
    indices: list[int] = []
    task_sets = []
    for task_set in plan.task_sets:
        index_begin = len(indices)
        for instance in task_set.instances:
            indices.extend(instance.indices)
        index_end = len(indices)
        index_rank = plan.task_index_ranks[task_set.task_kind]
        task_sets.append(
            f"{{ {task_set.task_kind}, {index_rank}, {index_begin}, {index_end}, {task_set.next_set_id} }}"
        )

    arrays = [
        static_queue_codegen.CArrayModel(
            "dynamic_task_indices",
            static_queue_codegen.c_values(tuple(indices)),
        ),
        static_queue_codegen.CArrayModel("task_sets", ", ".join(task_sets), "TaskSetDesc"),
    ]
    return tuple(arrays)


def build_host_model(plan: DynamicScheduledGraph) -> tuple[str, tuple[str, ...]]:
    initial_ready_sets = static_queue_codegen.c_values(plan.initial_ready_sets)
    scheduler_init = (
        f"std::vector<int> scheduler_zero_sets({len(plan.task_sets)}, 0);",
        f"CUDA_CHECK(cudaMemcpyToSymbol(dynamic_set_index_pos, scheduler_zero_sets.data(), sizeof(int) * {len(plan.task_sets)}));",
        "int scheduler_zero = 0;",
        "CUDA_CHECK(cudaMemcpyToSymbol(dynamic_queue_head, &scheduler_zero, sizeof(int)));",
        "CUDA_CHECK(cudaMemcpyToSymbol(dynamic_queue_lock, &scheduler_zero, sizeof(int)));",
        "CUDA_CHECK(cudaMemcpyToSymbol(dynamic_tiles_done, &scheduler_zero, sizeof(int)));",
        f"int scheduler_tail = {len(plan.initial_ready_sets)};",
        "CUDA_CHECK(cudaMemcpyToSymbol(dynamic_queue_tail, &scheduler_tail, sizeof(int)));",
        f"std::vector<int> initial_ready_sets_host = {{ {initial_ready_sets} }};",
        f"CUDA_CHECK(cudaMemcpyToSymbol(dynamic_ready_queue, initial_ready_sets_host.data(), sizeof(int) * {len(plan.initial_ready_sets)}));",
    )
    return static_queue_codegen.build_host_lines(
        plan.source,
        plan.events,
        str(plan.worker_count),
        scheduler_init,
    )


def build_cuda_module_model(plan: DynamicScheduledGraph) -> CudaModuleModel:
    graph = plan.source
    host_params, host_body_lines = build_host_model(plan)
    total_tile_count = sum(len(task_set.instances) for task_set in plan.task_sets)
    return CudaModuleModel(
        device_functions=codegen.compile_statement_functions(graph),
        task_macros=codegen.render_task_macros(graph),
        task_types=static_queue_codegen.build_task_type_models_for(
            graph,
            plan.events,
            plan.task_kinds,
        ),
        arrays=build_scheduler_arrays(plan),
        task_set_count=len(plan.task_sets),
        ready_queue_capacity=total_tile_count + len(plan.task_sets) + 16,
        total_tile_count=total_tile_count,
        kernel_signature=static_queue_codegen.build_kernel_signature_for(
            graph,
            plan.events,
        ),
        task_undefs="\n".join(f"#undef {name}" for name in graph.statements),
        host_params=host_params,
        host_body_lines=host_body_lines,
    )


def render_cuda_source(plan: DynamicScheduledGraph) -> str:
    model = build_cuda_module_model(plan)
    return static_queue_codegen.render_template("dynamic_queue_kernel.cu.j2", model)


def emit(plan: DynamicScheduledGraph, prefix: Path) -> Path:
    output = prefix.with_suffix(".cu")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_cuda_source(plan))
    return output


def build(
    plan: DynamicScheduledGraph,
    prefix: Path,
    *,
    python: str = sys.executable,
) -> Path:
    emit(plan, prefix)
    return static_queue_codegen.build_shared_library(prefix, python=python)
