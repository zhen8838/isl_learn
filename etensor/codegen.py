from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import math
from dataclasses import dataclass
from pathlib import Path

import isl
from jinja2 import Environment, FileSystemLoader
import tilelang

from etensor.ir import (
    EventInfo,
    Statement,
    TaskGraph,
    Tensor,
    constant_pw_aff_value as _constant_pw_aff_value,
    derive_event_infos,
)


@dataclass(frozen=True)
class EventCall:
    name: str
    indices: tuple[isl.ast_expr, ...]


@dataclass(frozen=True)
class UserPrintInfo:
    statement: str
    waits: tuple[EventCall, ...]
    notifies: tuple[EventCall, ...]


@dataclass(frozen=True)
class CudaModuleModel:
    device_functions: str
    task_macros: str
    kernel_signature: str
    block_params: str
    schedule: str
    task_undefs: str
    host_code: str


SYNC_MARK = "etensor_sync"


def extract_tilelang_device_function(cuda_source: str, device_name: str) -> str:
    matches = list(re.finditer(r'extern\s+"C"\s+__global__\s+void\s+(?:__launch_bounds__\([^)]*\)\s+)?\w+\s*\((?P<params>[^)]*)\)\s*\{', cuda_source))
    if not matches:
        raise ValueError("could not find a TileLang CUDA kernel definition")

    match = matches[-1]
    body_start = match.end() - 1
    depth = 0
    for pos in range(body_start, len(cuda_source)):
        if cuda_source[pos] == "{":
            depth += 1
        elif cuda_source[pos] == "}":
            depth -= 1
            if depth == 0:
                body = cuda_source[body_start : pos + 1]
                break
    else:
        raise ValueError("could not find the end of the TileLang CUDA kernel body")

    return f"__device__ __forceinline__ void {device_name}({match.group('params').strip()}) {body}"


def compile_statement_functions(graph: TaskGraph) -> str:
    device_functions = []
    for name, statement in graph.statements.items():
        kernel = tilelang.compile(statement.primfunc, out_idx=[])
        device_functions.append(
            extract_tilelang_device_function(
                kernel.get_kernel_source(kernel_only=True),
                f"etensor_{name}_task",
            )
        )
    return "\n\n".join(device_functions)


def render_task_macros(graph: TaskGraph) -> str:
    lines = []
    for name, statement in graph.statements.items():
        args = ", ".join(statement.indices)
        call_args = ", ".join((*statement.buffers, *statement.indices))
        lines.append(f"#define {name}({args}) do {{ etensor_{name}_task({call_args}); }} while (0)")
    return "\n".join(lines)


def _current_block_set(graph: TaskGraph) -> isl.union_set:
    placement_range = graph.placement_map().range().get_set_list().get_at(0)
    block_dims = [
        placement_range.get_dim_name(isl.dim_type.SET, i)
        for i in range(placement_range.dim(isl.dim_type.SET))
    ]
    schedule_params = graph.schedule_map().params()
    param_names = (
        *[
            schedule_params.get_dim_name(isl.dim_type.PARAM, i)
            for i in range(schedule_params.dim(isl.dim_type.PARAM))
        ],
        *[f"b{name}" for name in block_dims],
    )
    params = f"[{', '.join(param_names)}] -> " if param_names else ""
    constraints = " and ".join(f"{name} = b{name}" for name in block_dims)
    return isl.union_set(
        f"{params}{{ BlockIdx[{', '.join(block_dims)}] : {constraints} }}"
    )


def build_block_schedule_tree(graph: TaskGraph, events: tuple[EventInfo, ...]) -> isl.schedule:
    sync_statements = set()
    for event in events:
        sync_statements.add(event.producer)
        sync_statements.add(event.consumer)

    actions = []
    for name, statement in graph.statements.items():
        mark = SYNC_MARK if name in sync_statements else None
        actions.append((name, statement.indices, mark))

    schedule_map = graph.schedule_map()
    placement_map = graph.placement_map()
    current_domain = schedule_map.apply_range(placement_map)
    current_domain = current_domain.intersect_range(_current_block_set(graph)).domain()
    block_schedule = schedule_map.intersect_domain(current_domain)
    schedule = isl.schedule.from_domain(current_domain)
    schedule = schedule.insert_partial_schedule(block_schedule.as_multi_union_pw_aff())

    filters = [f"{{ {name}[{', '.join(indices)}] }}" for name, indices, _ in actions]
    schedule = schedule.get_root().child(0).child(0)
    schedule = schedule.insert_sequence(isl.union_set_list(f"({', '.join(filters)})")).schedule()

    for i, (_, _, mark) in enumerate(actions):
        if mark is not None:
            node = schedule.get_root().child(0).child(0).child(i).child(0)
            schedule = node.insert_mark(isl.id(mark)).schedule()

    return schedule


def _print_event_call(
    printer: isl.printer,
    event_call: EventCall,
    fn: str,
) -> isl.printer:
    event_access = isl.ast_expr.from_id(event_call.name)
    for index in event_call.indices:
        event_access = event_access.access(index)

    printer.start_line()
    printer.print_str(f"if (threadIdx.x == 0) {{ etensor::{fn}(")
    printer.print_ast_expr(event_access.address_of())
    printer.print_str("); }")
    printer.end_line()
    return printer


def _event_args_for_current_instance(
    event_map: isl.map,
    build: isl.ast_build,
) -> tuple[isl.ast_expr, ...]:
    current_instance = build.get_schedule().as_map().reverse().as_pw_multi_aff()
    event_instance = event_map.as_pw_multi_aff().pullback(current_instance)
    return tuple(build.expr_from(event_instance.at(i)) for i in range(event_instance.size()))


def render_block_schedule(graph: TaskGraph, events: tuple[EventInfo, ...]) -> str:
    schedule = build_block_schedule_tree(graph, events)
    user_infos: dict[int, UserPrintInfo] = {}

    def after_mark_callback(
        node: isl.ast_node_mark, build: isl.ast_build
    ) -> isl.ast_node:
        child = node.node()
        return isl.ast_node_block(isl.ast_node_list(isl.ast_node(child)))

    def at_each_domain(
        node: isl.ast_node_user, build: isl.ast_build
    ) -> isl.ast_node:
        expr = node.expr()
        statement = expr.get_arg(0).get_id().get_name()
        waits = tuple(
            EventCall(
                event.name,
                _event_args_for_current_instance(event.wait_access, build),
            )
            for event in events
            if statement == event.consumer
        )
        notifies = tuple(
            EventCall(
                event.name,
                _event_args_for_current_instance(event.notify_access, build),
            )
            for event in events
            if statement == event.producer
        )

        annotation = isl.id(f"{statement}_{len(user_infos)}")
        user_infos[annotation.ptr] = UserPrintInfo(
            statement=statement,
            waits=waits,
            notifies=notifies,
        )
        return node.set_annotation(annotation)

    def print_user(
        printer: isl.printer, options: isl.ast_print_options, node: isl.ast_node_user
    ) -> isl.printer:
        expr = node.expr()
        info = user_infos[node.annotation().ptr]
        for event_call in info.waits:
            printer = _print_event_call(printer, event_call, "wait")
        if info.waits:
            printer.start_line()
            printer.print_str("__syncthreads();")
            printer.end_line()

        printer.start_line()
        printer.print_str(f"{info.statement}(")
        for i in range(1, expr.get_n_arg()):
            if i != 1:
                printer.print_str(", ")
            printer.print_ast_expr(expr.get_arg(i))
        printer.print_str(");")
        printer.end_line()

        if info.notifies:
            printer.start_line()
            printer.print_str("__syncthreads();")
            printer.end_line()
            for event_call in info.notifies:
                printer = _print_event_call(printer, event_call, "notify")
        return printer

    fd, raw_path = tempfile.mkstemp(suffix=".c")
    os.close(fd)
    path = Path(raw_path)
    try:
        builder = (
            isl.ast_build.from_context(graph.context)
            if graph.context is not None
            else isl.ast_build()
        )
        builder = builder.set_after_each_mark(after_mark_callback)
        builder = builder.set_at_each_domain(at_each_domain)
        ast = builder.node_from(schedule)
        printer = isl.printer.to_file_path(str(path))
        printer = printer.set_output_format(isl.format.C)
        options = isl.ast_print_options.alloc()
        options = options.set_print_user(print_user)
        ast.print(printer, options).flush()
        return path.read_text().strip()
    finally:
        path.unlink(missing_ok=True)


def _event_param_decl(event: EventInfo) -> str:
    shape = event.init_values.shape
    if len(shape) <= 1:
        return f"int* {event.name}"
    suffix = "".join(f"[{extent}]" for extent in shape[1:])
    return f"int (*{event.name}){suffix}"


def _event_cast_type(event: EventInfo) -> str:
    shape = event.init_values.shape
    if len(shape) <= 1:
        return "int*"
    suffix = "".join(f"[{extent}]" for extent in shape[1:])
    return f"int (*){suffix}"


def render_kernel_signature(graph: TaskGraph, events: tuple[EventInfo, ...]) -> str:
    params = [f"{tensor.dtype}* {name}" for name, tensor in graph.buffers.items()]
    params.extend(_event_param_decl(event) for event in events)
    return ", ".join(params)


def _grid_extent_expr(graph: TaskGraph, pos: int) -> str:
    placement_range = graph.placement_map().range().get_set_list().get_at(0)
    dim_min = placement_range.dim_min(pos)
    dim_max = placement_range.dim_max(pos)
    if dim_min.is_cst() and dim_max.is_cst():
        low = _constant_pw_aff_value(dim_min)
        high = _constant_pw_aff_value(dim_max)
        return str(high - low + 1)
    raise ValueError(f"could not derive a CUDA grid extent for placement dim {pos}")


def render_grid_expr(graph: TaskGraph) -> str:
    dims = graph.placement_map().range().get_set_list().get_at(0).dim(isl.dim_type.SET)
    extents = [_grid_extent_expr(graph, i) for i in range(dims)]
    if len(extents) == 1:
        return extents[0]
    return f"dim3({', '.join(extents)})"


def render_block_params(graph: TaskGraph) -> str:
    placement_range = graph.placement_map().range().get_set_list().get_at(0)
    cuda_dims = ("x", "y", "z")
    lines = []
    for pos in range(placement_range.dim(isl.dim_type.SET)):
        name = placement_range.get_dim_name(isl.dim_type.SET, pos)
        if pos >= len(cuda_dims):
            raise ValueError("CUDA grid codegen only supports up to three block dims")
        lines.append(f"  const int b{name} = blockIdx.{cuda_dims[pos]};")
    return "\n".join(lines)


def _event_tensor_extent(event: EventInfo) -> int:
    return int(event.init_values.size)


def _event_init_list(event: EventInfo) -> str:
    return ", ".join(str(int(value)) for value in event.init_values.flat)


def render_host_code(graph: TaskGraph, events: tuple[EventInfo, ...]) -> str:
    host_params = [f"int64_t {name}_addr" for name in graph.buffers]
    lines = [
        f"{tensor.dtype}* {name} = reinterpret_cast<{tensor.dtype}*>({name}_addr);"
        for name, tensor in graph.buffers.items()
    ]
    lines.extend(f"int* {event.name}_storage = nullptr;" for event in events)
    for event in events:
        event_extent = _event_tensor_extent(event)
        lines.extend(
            (
                f"std::vector<int> {event.name}_init = {{ {_event_init_list(event)} }};",
                f"CUDA_CHECK(cudaMalloc(&{event.name}_storage, sizeof(int) * {event_extent}));",
                f"CUDA_CHECK(cudaMemcpy({event.name}_storage, {event.name}_init.data(), sizeof(int) * {event_extent}, cudaMemcpyHostToDevice));",
                f"auto {event.name} = reinterpret_cast<{_event_cast_type(event)}>({event.name}_storage);",
            )
        )
    args = ", ".join((*graph.buffers.keys(), *(event.name for event in events)))
    lines.extend(
        (
            f"mega_kernel<<<{render_grid_expr(graph)}, 128>>>({args});",
            "CUDA_CHECK(cudaGetLastError());",
            "CUDA_CHECK(cudaDeviceSynchronize());",
        )
    )
    lines.extend(f"CUDA_CHECK(cudaFree({event.name}_storage));" for event in events)
    body = "\n".join(f"  {line}" for line in lines)
    return f"void run_static_impl({', '.join(host_params)}) {{\n{body}\n}}"


def build_cuda_module_model(graph: TaskGraph) -> CudaModuleModel:
    events = derive_event_infos(graph)
    return CudaModuleModel(
        device_functions=compile_statement_functions(graph),
        task_macros=render_task_macros(graph),
        kernel_signature=render_kernel_signature(graph, events),
        block_params=render_block_params(graph),
        schedule=render_block_schedule(graph, events),
        task_undefs="\n".join(f"#undef {name}" for name in graph.statements),
        host_code=render_host_code(graph, events),
    )


def render_template(template_name: str, model: object) -> str:
    root = Path(__file__).resolve().parent
    env = Environment(
        loader=FileSystemLoader(root / "templates"),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template(template_name).render(**model.__dict__)


def render_cuda_source(graph: TaskGraph) -> str:
    model = build_cuda_module_model(graph)
    return render_template("base_kernel.cu.j2", model)


def emit(graph: TaskGraph, prefix: Path) -> Path:
    output = prefix.with_suffix(".cu")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_cuda_source(graph))
    return output


def build(graph: TaskGraph, prefix: Path, *, python: str = sys.executable) -> Path:
    root = Path(__file__).resolve().parent
    os.environ.setdefault("TMPDIR", str(root / "build" / "tilelang_tmp"))
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    emit(graph, prefix)
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
