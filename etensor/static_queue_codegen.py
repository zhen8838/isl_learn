from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import isl
from jinja2 import Environment, FileSystemLoader

from etensor import codegen


@dataclass(frozen=True)
class TaskInstance:
    statement: str
    indices: tuple[int, ...]
    schedule_key: tuple[int, ...]


@dataclass(frozen=True)
class StaticQueue:
    task_kinds: tuple[int, ...]
    task_indices: tuple[int, ...]


@dataclass(frozen=True)
class StaticScheduledGraph:
    source: codegen.TaskGraph
    events: tuple[codegen.EventInfo, ...]
    worker_count: int
    task_kinds: dict[str, int]
    task_index_ranks: tuple[int, ...]
    queues: tuple[StaticQueue, ...]


@dataclass(frozen=True)
class CArrayModel:
    name: str
    values: str
    ctype: str = "int"


@dataclass(frozen=True)
class TaskTypeModel:
    enum_name: str
    kind: int
    body_lines: tuple[str, ...]


@dataclass(frozen=True)
class CudaModuleModel:
    device_functions: str
    task_macros: str
    task_types: tuple[TaskTypeModel, ...]
    arrays: tuple[CArrayModel, ...]
    kernel_signature: str
    task_undefs: str
    host_params: str
    host_body_lines: tuple[str, ...]


def val_to_int(value: isl.val) -> int:
    if not value.is_int() or value.get_den_si() != 1:
        raise ValueError(f"expected integer value, got {value}")
    return value.get_num_si()


def point_indices(point: isl.point) -> tuple[int, ...]:
    return tuple(
        val_to_int(point.get_coordinate_val(isl.dim_type.SET, dim))
        for dim in range(point.dim(isl.dim_type.SET))
    )


def single_point_indices(point_set: isl.set) -> tuple[int, ...]:
    points: list[tuple[int, ...]] = []
    point_set.foreach_point(lambda point: points.append(point_indices(point)))
    if len(points) != 1:
        raise ValueError(f"expected exactly one schedule point, got {point_set}")
    return points[0]


def enumerate_task_instances(graph: codegen.TaskGraph) -> tuple[TaskInstance, ...]:
    schedule = graph.schedule_map()
    instances: list[TaskInstance] = []
    for statement in graph.statements:
        domain = graph.statement_domain(statement)

        def visit(point: isl.point) -> None:
            schedule_image = point.to_set().apply(schedule)
            instances.append(
                TaskInstance(
                    statement=statement,
                    indices=point_indices(point),
                    schedule_key=single_point_indices(schedule_image),
                )
            )

        domain.foreach_point(visit)
    return tuple(sorted(instances, key=lambda instance: instance.schedule_key))


def enumerate_task_instances_by_statement(
    graph: codegen.TaskGraph,
) -> dict[str, tuple[TaskInstance, ...]]:
    by_statement: dict[str, list[TaskInstance]] = {
        statement: [] for statement in graph.statements
    }
    for instance in enumerate_task_instances(graph):
        by_statement[instance.statement].append(instance)
    return {
        statement: tuple(instances)
        for statement, instances in by_statement.items()
    }


def static_schedule(
    graph: codegen.TaskGraph,
    worker_count: int,
) -> StaticScheduledGraph:
    task_kinds = {name: i for i, name in enumerate(graph.statements)}
    task_index_ranks = tuple(
        len(statement.indices) for statement in graph.statements.values()
    )
    mutable_kinds: list[list[int]] = [[] for _ in range(worker_count)]
    mutable_indices: list[list[int]] = [[] for _ in range(worker_count)]

    for logical_time, instance in enumerate(enumerate_task_instances(graph)):
        worker = logical_time % worker_count
        mutable_kinds[worker].append(task_kinds[instance.statement])
        mutable_indices[worker].extend(instance.indices)

    queues = [
        StaticQueue(task_kinds=tuple(kinds), task_indices=tuple(indices))
        for kinds, indices in zip(mutable_kinds, mutable_indices)
    ]
    return StaticScheduledGraph(
        source=graph,
        events=codegen.derive_event_infos(graph),
        worker_count=worker_count,
        task_kinds=task_kinds,
        task_index_ranks=task_index_ranks,
        queues=tuple(queues),
    )


def c_values(values: tuple[int, ...]) -> str:
    return ", ".join(str(value) for value in values) if values else "0"


def build_queue_arrays(plan: StaticScheduledGraph) -> tuple[CArrayModel, ...]:
    tasks: list[str] = []
    indices: list[int] = []
    queues: list[str] = []
    for queue in plan.queues:
        task_begin = len(tasks)
        index_pos = 0
        for task_kind in queue.task_kinds:
            tasks.append(f"{{ {task_kind}, {len(indices) + index_pos} }}")
            index_pos += plan.task_index_ranks[task_kind]
        indices.extend(queue.task_indices)
        queues.append(f"{{ {task_begin}, {len(tasks)} }}")
    return (
        CArrayModel("static_task_indices", c_values(tuple(indices))),
        CArrayModel("static_tasks", ", ".join(tasks), "StaticTaskDesc"),
        CArrayModel("static_queues", ", ".join(queues), "StaticQueueDesc"),
    )


def rename_event_map_inputs(
    event_map: isl.map,
    statement: codegen.Statement,
) -> isl.map:
    for pos, index_name in enumerate(statement.indices):
        event_map = event_map.set_dim_name(isl.dim_type.IN, pos, index_name)
    return event_map


def print_ast_expr_to_c(expr: isl.ast_expr) -> str:
    fd, raw_path = tempfile.mkstemp(suffix=".c")
    os.close(fd)
    path = Path(raw_path)
    try:
        printer = isl.printer.to_file_path(str(path))
        printer = printer.set_output_format(isl.format.C)
        printer.print_ast_expr(expr).flush()
        return path.read_text().strip()
    finally:
        path.unlink(missing_ok=True)


def render_affine_index_expr(
    pw_aff: isl.pw_aff,
    context: isl.set,
) -> str:
    build = isl.ast_build.from_context(context)
    return print_ast_expr_to_c(build.expr_from(pw_aff))


def event_access_expr(
    event: codegen.EventInfo,
    event_map: isl.map,
    statement: codegen.Statement,
) -> str:
    event_map = rename_event_map_inputs(event_map, statement)
    pma = event_map.as_pw_multi_aff()
    pieces = []
    for dim in range(pma.size()):
        expr = render_affine_index_expr(pma.at(dim), event_map.domain())
        pieces.append(expr)

    access = event.name
    for piece in pieces:
        access += f"[{piece}]"
    return f"&{access}"


def task_body_lines(
    events: tuple[codegen.EventInfo, ...],
    statement_name: str,
    statement: codegen.Statement,
) -> tuple[str, ...]:
    lines = []
    for pos, index_name in enumerate(statement.indices):
        lines.append(f"const int {index_name} = task_idx[{pos}];")

    for event in events:
        if event.consumer == statement_name:
            access = event_access_expr(
                event,
                event.wait_access,
                statement,
            )
            lines.append(f"if (threadIdx.x == 0) {{ etensor::wait({access}); }}")
    if any(event.consumer == statement_name for event in events):
        lines.append("__syncthreads();")

    call_args = ", ".join(statement.indices)
    lines.append(f"{statement_name}({call_args});")

    if any(event.producer == statement_name for event in events):
        lines.append("__syncthreads();")
    for event in events:
        if event.producer == statement_name:
            access = event_access_expr(
                event,
                event.notify_access,
                statement,
            )
            lines.append(f"if (threadIdx.x == 0) {{ etensor::notify({access}); }}")

    return tuple(lines)


def build_task_type_models_for(
    graph: codegen.TaskGraph,
    events: tuple[codegen.EventInfo, ...],
    task_kinds: dict[str, int],
) -> tuple[TaskTypeModel, ...]:
    return tuple(
        TaskTypeModel(
            enum_name=f"TASK_{statement}",
            kind=kind,
            body_lines=task_body_lines(
                events,
                statement,
                graph.statements[statement],
            ),
        )
        for statement, kind in task_kinds.items()
    )


def build_task_type_models(plan: StaticScheduledGraph) -> tuple[TaskTypeModel, ...]:
    return build_task_type_models_for(plan.source, plan.events, plan.task_kinds)


def event_tensor_extent(event: codegen.EventInfo) -> int:
    return int(event.init_values.size)


def event_param_decl(event: codegen.EventInfo) -> str:
    shape = event.init_values.shape
    if len(shape) <= 1:
        return f"int* {event.name}"
    suffix = "".join(f"[{extent}]" for extent in shape[1:])
    return f"int (*{event.name}){suffix}"


def event_cast_type(event: codegen.EventInfo) -> str:
    shape = event.init_values.shape
    if len(shape) <= 1:
        return "int*"
    suffix = "".join(f"[{extent}]" for extent in shape[1:])
    return f"int (*){suffix}"


def event_init_list(event: codegen.EventInfo) -> str:
    return ", ".join(str(int(value)) for value in event.init_values.flat)


def build_kernel_signature_for(
    graph: codegen.TaskGraph,
    events: tuple[codegen.EventInfo, ...],
) -> str:
    params = [f"{tensor.dtype}* {name}" for name, tensor in graph.buffers.items()]
    params.extend(event_param_decl(event) for event in events)
    return ", ".join(params)


def build_kernel_signature(plan: StaticScheduledGraph) -> str:
    return build_kernel_signature_for(plan.source, plan.events)


def build_host_lines(
    graph: codegen.TaskGraph,
    events: tuple[codegen.EventInfo, ...],
    launch_grid: str,
    extra_lines: tuple[str, ...] = (),
) -> tuple[str, tuple[str, ...]]:
    host_params = [f"int64_t {name}_addr" for name in graph.buffers]
    lines = [
        f"{tensor.dtype}* {name} = reinterpret_cast<{tensor.dtype}*>({name}_addr);"
        for name, tensor in graph.buffers.items()
    ]
    lines.extend(f"int* {event.name}_storage = nullptr;" for event in events)
    for event in events:
        event_extent = event_tensor_extent(event)
        lines.extend(
            (
                f"std::vector<int> {event.name}_init = {{ {event_init_list(event)} }};",
                f"CUDA_CHECK(cudaMalloc(&{event.name}_storage, sizeof(int) * {event_extent}));",
                f"CUDA_CHECK(cudaMemcpy({event.name}_storage, {event.name}_init.data(), sizeof(int) * {event_extent}, cudaMemcpyHostToDevice));",
                f"auto {event.name} = reinterpret_cast<{event_cast_type(event)}>({event.name}_storage);",
            )
        )
    lines.extend(extra_lines)
    launch_args = ", ".join((*graph.buffers.keys(), *(event.name for event in events)))
    lines.extend(
        (
            f"mega_kernel<<<{launch_grid}, 128>>>({launch_args});",
            "CUDA_CHECK(cudaGetLastError());",
            "CUDA_CHECK(cudaDeviceSynchronize());",
        )
    )
    lines.extend(f"CUDA_CHECK(cudaFree({event.name}_storage));" for event in events)
    return ", ".join(host_params), tuple(lines)


def build_host_model(plan: StaticScheduledGraph) -> tuple[str, tuple[str, ...]]:
    return build_host_lines(plan.source, plan.events, str(plan.worker_count))


def build_cuda_module_model(plan: StaticScheduledGraph) -> CudaModuleModel:
    graph = plan.source
    host_params, host_body_lines = build_host_model(plan)
    return CudaModuleModel(
        device_functions=codegen.compile_statement_functions(graph),
        task_macros=codegen.render_task_macros(graph),
        task_types=build_task_type_models(plan),
        arrays=build_queue_arrays(plan),
        kernel_signature=build_kernel_signature(plan),
        task_undefs="\n".join(f"#undef {name}" for name in graph.statements),
        host_params=host_params,
        host_body_lines=host_body_lines,
    )


def render_cuda_source(plan: StaticScheduledGraph) -> str:
    model = build_cuda_module_model(plan)
    return render_template("static_queue_kernel.cu.j2", model)


def render_template(template_name: str, model: object) -> str:
    root = Path(__file__).resolve().parent
    env = Environment(
        loader=FileSystemLoader(root / "templates"),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template(template_name).render(**model.__dict__)


def emit(plan: StaticScheduledGraph, prefix: Path) -> Path:
    output = prefix.with_suffix(".cu")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_cuda_source(plan))
    return output


def build(
    plan: StaticScheduledGraph,
    prefix: Path,
    *,
    python: str = sys.executable,
) -> Path:
    emit(plan, prefix)
    return build_shared_library(prefix, python=python)


def build_shared_library(
    prefix: Path,
    *,
    python: str = sys.executable,
) -> Path:
    root = Path(__file__).resolve().parent
    os.environ.setdefault("TMPDIR", str(root / "build" / "tilelang_tmp"))
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "make",
            "-C",
            str(root),
            f"PYTHON={python}",
            f"PREFIX={prefix.relative_to(root)}",
        ],
        check=True,
    )
    return prefix.with_suffix(".so")
