import argparse
from pathlib import Path

import torch
import tvm_ffi

from etensor import dynamic_queue_codegen
from etensor.demo import row_sum_demo


def run(n: int, worker_count: int) -> None:
    root = Path(__file__).resolve().parents[1]
    prefix = root / "demo" / "build" / "row_sum_dynamic_queue_demo"
    graph = row_sum_demo.get_task_graph(n)
    scheduled = dynamic_queue_codegen.dynamic_schedule(graph, worker_count)
    lib = dynamic_queue_codegen.build(scheduled, prefix)

    mod = tvm_ffi.load_module(str(lib))
    rows = n * row_sum_demo.M_TILE
    A = row_sum_demo.fill_input(n)
    B = torch.empty(
        (rows, row_sum_demo.K_SPLIT),
        device="cuda",
        dtype=torch.float32,
    )
    C = torch.empty((rows,), device="cuda", dtype=torch.float32)
    ref = A.sum(dim=1)

    func = mod.get_function("run_static")
    func(A.data_ptr(), B.data_ptr(), C.data_ptr())

    max_abs_diff = torch.max(torch.abs(C - ref)).item()
    print(f"dynamic_queue max_abs_diff={max_abs_diff:.6f}")
    if max_abs_diff > 1e-4:
        raise RuntimeError("dynamic queue verification failed")
    print("PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("n", nargs="?", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.n, args.workers)


if __name__ == "__main__":
    main()
