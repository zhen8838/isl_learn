from __future__ import annotations

from etensor import codegen
from etensor import dynamic_queue_codegen
from etensor import static_queue_codegen
from etensor.ir import derive_event_infos


EventCall = codegen.EventCall
UserPrintInfo = codegen.UserPrintInfo


def build_block_schedule_tree(graph):
    return codegen.build_block_schedule_tree(graph, derive_event_infos(graph))


def event_args_for_current_instance(event_map, build):
    return codegen._event_args_for_current_instance(event_map, build)


def print_event_call(printer, event_call, fn):
    return codegen._print_event_call(printer, event_call, fn)


def render_base_cuda_source(graph, events, schedule_code: str | None = None) -> str:
    model = codegen.CudaModuleModel(
        device_functions=codegen.compile_statement_functions(graph),
        task_macros=codegen.render_task_macros(graph),
        kernel_signature=codegen.render_kernel_signature(graph, events),
        block_params=codegen.render_block_params(graph),
        schedule=schedule_code if schedule_code is not None else codegen.render_block_schedule(graph, events),
        task_undefs="\n".join(f"#undef {name}" for name in graph.statements),
        host_code=codegen.render_host_code(graph, events),
    )
    return codegen.render_template("base_kernel.cu.j2", model)


def build_static_queue_arrays(plan):
    return static_queue_codegen.build_queue_arrays(plan)


def render_static_cuda_source(plan) -> str:
    return static_queue_codegen.render_cuda_source(plan)


def build_dynamic_scheduler_arrays(plan):
    return dynamic_queue_codegen.build_scheduler_arrays(plan)


def render_dynamic_cuda_source(plan) -> str:
    return dynamic_queue_codegen.render_cuda_source(plan)
