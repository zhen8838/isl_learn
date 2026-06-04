import argparse
from pathlib import Path

import isl
import torch
import tilelang.language as T
import tvm_ffi

from etensor import codegen


M_TILE = 32
K_TILE = 32
K_SPLIT = 4
K = K_TILE * K_SPLIT


def partial_sum(n: int):
    @T.prim_func
    def partial_task(
        A: T.Tensor((n * M_TILE, K), T.float32),
        B: T.Tensor((n * M_TILE, K_SPLIT), T.float32),
        i: T.int32,
        j: T.int32,
    ):
        with T.Kernel(1, threads=128):
            for r in T.Parallel(M_TILE):
                row = i * M_TILE + r
                acc = T.alloc_local((1,), T.float32)
                acc[0] = 0.0
                for k in T.serial(K_TILE):
                    acc[0] += A[row, j * K_TILE + k]
                B[row, j] = acc[0]

    return partial_task


def final_sum(n: int):
    @T.prim_func
    def final_task(
        B: T.Tensor((n * M_TILE, K_SPLIT), T.float32),
        C: T.Tensor((n * M_TILE,), T.float32),
        i: T.int32,
    ):
        with T.Kernel(1, threads=128):
            for r in T.Parallel(M_TILE):
                row = i * M_TILE + r
                acc = T.alloc_local((1,), T.float32)
                acc[0] = 0.0
                for j in T.serial(K_SPLIT):
                    acc[0] += B[row, j]
                C[row] = acc[0]

    return final_task


def get_task_graph(n: int) -> codegen.TaskGraph:
    buffers = {
        "A": codegen.Tensor(
            domain=isl.set(f"[n] -> {{ A[m, k] : 0 <= m < {M_TILE}n and 0 <= k < {K} }}"),
            dtype="float",
        ),
        "B": codegen.Tensor(
            domain=isl.set(f"[n] -> {{ B[m, j] : 0 <= m < {M_TILE}n and 0 <= j < 4 }}"),
            dtype="float",
        ),
        "C": codegen.Tensor(
            domain=isl.set(f"[n] -> {{ C[m] : 0 <= m < {M_TILE}n }}"),
            dtype="float",
        ),
    }
    statements = {
        "P": codegen.Statement(
            domain=isl.set(f"[n] -> {{ P[i, j] : 0 <= i < n and 0 <= j < {K_SPLIT} }}"),
            primfunc=partial_sum(n),
            buffers=("A", "B"),
            indices=("i", "j"),
        ),
        "F": codegen.Statement(
            domain=isl.set(f"[n] -> {{ F[i] : 0 <= i < n }}"),
            primfunc=final_sum(n),
            buffers=("B", "C"),
            indices=("i",),
        ),
    }
    dependence = isl.union_map(
        f"[n] -> {{ P[i, j] -> F[i] : 0 <= i < n and 0 <= j < {K_SPLIT} }}"
    )
    schedule = isl.union_map(
        f"[n] -> {{ P[i, j] -> [i, 0, j] : 0 <= i < n and 0 <= j < {K_SPLIT}; F[i] -> [i, 1, 0] : 0 <= i < n }}"
    )
    placement = isl.union_map(f"[n] -> {{ [i,t,j] -> BlockIdx[x, y] : x = i and 0 <= i < n and j = y and 0 <= j < {K_SPLIT} }}")
    return codegen.TaskGraph(
        buffers=buffers,
        statements=statements,
        dependence=dependence,
        schedule=schedule,
        placement=placement,
        context=isl.set(f"[n] -> {{ : n = {n} }}"),
    )


def fill_input(n: int) -> torch.Tensor:
    rows = n * M_TILE
    cols = K_TILE * K_SPLIT
    row = torch.arange(rows, device="cuda", dtype=torch.float32).reshape(rows, 1)
    col = torch.arange(cols, device="cuda", dtype=torch.float32).reshape(1, cols)
    return torch.remainder(row, 17) * 0.1 + torch.remainder(col, 11) * 0.01


def run(n: int) -> None:
    root = Path(__file__).resolve().parents[1]
    prefix = root / "demo" / "build" / "row_sum_demo"
    graph = get_task_graph(n)
    lib = codegen.build(graph, prefix)

    mod = tvm_ffi.load_module(str(lib))
    rows = n * M_TILE
    A = fill_input(n)
    B = torch.empty((rows, K_SPLIT), device="cuda", dtype=torch.float32)
    C = torch.empty((rows,), device="cuda", dtype=torch.float32)
    ref = A.sum(dim=1)

    func = mod.get_function("run_static")
    func(A.data_ptr(), B.data_ptr(), C.data_ptr())

    max_abs_diff = torch.max(torch.abs(C - ref)).item()
    print(f"static max_abs_diff={max_abs_diff:.6f}")
    if max_abs_diff > 1e-4:
        raise RuntimeError("static verification failed")
    print("PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("n", nargs="?", type=int, default=8)
    args = parser.parse_args()
    run(args.n)


if __name__ == "__main__":
    main()
