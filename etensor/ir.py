from __future__ import annotations

from dataclasses import dataclass

import isl
import numpy as np


@dataclass(frozen=True)
class Tensor:
    domain: isl.set
    dtype: str


@dataclass(frozen=True)
class Statement:
    domain: isl.set
    primfunc: object
    buffers: tuple[str, ...]
    indices: tuple[str, ...]


@dataclass(frozen=True)
class TaskGraph:
    buffers: dict[str, Tensor]
    statements: dict[str, Statement]
    dependence: isl.union_map
    schedule: isl.union_map
    placement: isl.union_map
    context: isl.set | None = None

    def _with_context(self, value):
        if self.context is None:
            return value
        return value.intersect_params(self.context)

    def statement_domain(self, name: str) -> isl.set:
        return self._with_context(self.statements[name].domain)

    def dependence_map(self) -> isl.union_map:
        return self._with_context(self.dependence)

    def schedule_map(self) -> isl.union_map:
        return self._with_context(self.schedule)

    def placement_map(self) -> isl.union_map:
        return self._with_context(self.placement)


@dataclass(frozen=True)
class EventInfo:
    name: str
    dependence: isl.map
    domain: isl.set
    init_values: np.ndarray
    producer: str
    consumer: str
    notify_access: isl.map
    wait_access: isl.map


def constant_pw_aff_value(value) -> int:
    if not value.is_cst():
        raise ValueError(f"expected a constant affine value, got {value}")
    val = value.max_val()
    if not val.is_int() or val.get_den_si() != 1:
        raise ValueError(f"expected an integer affine value, got {value}")
    return val.get_num_si()


def derive_event_infos(graph: TaskGraph) -> tuple[EventInfo, ...]:
    maps = graph.dependence_map().get_map_list()
    events = []
    for i in range(maps.n_map()):
        dependence = maps.get_at(i)
        name = f"E{i}"
        domain = dependence.range()
        lows = [
            domain.dim_min_val(dim).get_num_si()
            for dim in range(domain.dim(isl.dim_type.SET))
        ]
        shape = tuple(
            domain.dim_max_val(dim).get_num_si() - lows[dim] + 1
            for dim in range(domain.dim(isl.dim_type.SET))
        )
        init_values = np.zeros(shape, dtype=np.int32)

        wait_access = domain.identity().set_tuple_name(isl.dim_type.OUT, name)
        pma = wait_access.as_pw_multi_aff()
        for dim, low in enumerate(lows):
            if low != 0:
                pma = pma.set_at(dim, pma.at(dim).add_constant(-low))
        wait_access = pma.as_map()

        def fill(point: isl.point) -> None:
            index = tuple(
                point.get_coordinate_val(isl.dim_type.SET, dim).get_num_si() - lows[dim]
                for dim in range(point.dim(isl.dim_type.SET))
            )
            init_values[index] = dependence.intersect_range(
                point.to_set()
            ).domain().count_val().get_num_si()

        domain.foreach_point(fill)
        events.append(
            EventInfo(
                name=name,
                dependence=dependence,
                domain=domain,
                init_values=init_values,
                producer=dependence.get_tuple_name(isl.dim_type.IN),
                consumer=dependence.get_tuple_name(isl.dim_type.OUT),
                notify_access=dependence.apply_range(wait_access),
                wait_access=wait_access,
            )
        )
    return tuple(events)
