import isl
import pet
import pandas
import numpy as np
from typing import Callable, Union


class CSource():
    def __init__(self, path: str) -> None:
        with open(path, 'r') as f:
            self.context = f.read()

    def _repr_html_(self) -> str:
        return "<pre class='code'><code class=\"cpp hljs\">" + self.context + "</code></pre>"

    def __str__(self) -> str:
        return f"```c\n{self.context}\n```"

    def __repr__(self) -> str:
        return str(self)


class CodeGenerator:
    def __init__(self, scop: pet.scop, schedule: isl.schedule, custom_pullback=None) -> None:
        self.scop = scop
        self.schedule = schedule
        self.custom_pullback: Callable[[isl.multi_pw_aff, isl.id,
                                        isl.pw_multi_aff], isl.multi_pw_aff] = custom_pullback

    def generate(self):
        def find_stmt_from_scop(id: isl.id) -> pet.stmt:
            """ 在pet解析的scop中找到对应的stmt.  """
            n_stmt = self.scop.get_n_stmt()
            for i in range(n_stmt):
                stmt = self.scop.get_stmt(i)
                domain = stmt.get_domain()
                id_i = domain.get_tuple_id()
                if (id.ptr == id_i.ptr):
                    return stmt

        id_dict = dict()

        def at_each_domain(node: isl.ast_node_user, build: isl.ast_build):
            expr: isl.ast_expr_op = node.get_expr()
            arg: isl.ast_expr_id = expr.get_arg(0)
            id: isl.id = arg.get_id()
            stmt: pet.stmt = find_stmt_from_scop(id)
            map = build.get_schedule().as_map()
            map = map.reverse()
            iterator_map = map.as_pw_multi_aff()

            def pullback_index(index: isl.multi_pw_aff, id: isl.id):
                if self.custom_pullback:
                    return self.custom_pullback(index, id, iterator_map)
                return index.pullback(iterator_map)

            ref2expr = stmt.build_ast_exprs(build, pullback_index, None)
            id_dict[id.ptr] = (stmt, ref2expr)

            return node.set_annotation(id)

        def print_user(p: isl.printer, opt: isl.ast_print_options, node: isl.ast_node_user):
            # when loop can parallel execute:
            id = node.annotation()
            (stmt, ref2expr) = id_dict[id.ptr]
            p = stmt.print_body(p, ref2expr)
            return p

        printer = isl.printer.from_file('/tmp/generated.c')
        printer.set_output_format(isl.ISL_FORMAT.C)
        builder = isl.ast_build()
        builder = builder.set_at_each_domain(at_each_domain)
        tree: isl.ast_node = builder.node_from(self.schedule)
        options = isl.ast_print_options.alloc()
        options = options.set_print_user(print_user)
        tree.print(printer, options)
        printer.flush()

        return CSource('/tmp/generated.c')


def parse_code(source: str, func_name: str) -> pet.scop:
    with open("/tmp/parse_code.c", "w") as f:
        f.write(source)
    scop = pet.scop.extract_from_C_source("/tmp/parse_code.c", func_name)
    return scop


def isl_mat_to_numpy(mat: isl.mat):
    return np.array([[mat.get_element_val(i, j).get_num_si()
                      for j in range(mat.cols())]
                     for i in range(mat.rows())])


def display_constraints(data: Union[isl.basic_map, isl.basic_set]):
    if isinstance(data, isl.basic_map):
        titles = bmap_dim_titles(data)
        eqs = isl_mat_to_numpy(data.equalities_matrix(isl.ISL_DIM_TYPE.CST,
                               isl.ISL_DIM_TYPE.PARAM,
                               isl.ISL_DIM_TYPE.IN,
                               isl.ISL_DIM_TYPE.OUT,
                               isl.ISL_DIM_TYPE.DIV))
        df_eq = pandas.DataFrame(eqs, columns=titles, index=[
                                 '' for i in eqs]) if len(eqs) else None
        ineqs = isl_mat_to_numpy(data.inequalities_matrix(isl.ISL_DIM_TYPE.CST,
                                 isl.ISL_DIM_TYPE.PARAM,
                                 isl.ISL_DIM_TYPE.IN,
                                 isl.ISL_DIM_TYPE.OUT,
                                 isl.ISL_DIM_TYPE.DIV))
        df_ineq = pandas.DataFrame(ineqs, columns=titles, index=[
                                   '' for i in ineqs]) if len(ineqs) else None
        return (df_ineq, df_eq)
    else:
        titles = bset_dim_titles(data)

        eqs = isl_mat_to_numpy(data.equalities_matrix(isl.ISL_DIM_TYPE.CST,
                                                      isl.ISL_DIM_TYPE.PARAM,
                                                      isl.ISL_DIM_TYPE.SET,
                                                      isl.ISL_DIM_TYPE.DIV))
        df_eq = pandas.DataFrame(eqs, columns=titles, index=[
                                 '' for i in eqs]) if len(eqs) else None
        ineqs = isl_mat_to_numpy(data.inequalities_matrix(isl.ISL_DIM_TYPE.CST,
                                                          isl.ISL_DIM_TYPE.PARAM,
                                                          isl.ISL_DIM_TYPE.SET,
                                                          isl.ISL_DIM_TYPE.DIV))
        df_ineq = pandas.DataFrame(ineqs, columns=titles, index=[
                                   '' for i in ineqs]) if len(ineqs) else None
        return (df_ineq, df_eq)


def bset_dim_titles(m: isl.basic_set):
    names = ["CST"]
    for t in [isl.ISL_DIM_TYPE.PARAM,
              isl.ISL_DIM_TYPE.SET,
              isl.ISL_DIM_TYPE.DIV]:
        for i in range(m.dim(t)):
            name = f"v{i}"
            try:
                name = m.dim_name(t, i)
            except Exception:
                pass
            names.append(t.name + " " + name)
    return names


def bmap_dim_titles(m: isl.basic_map):
    names = ["CST"]
    for t in [isl.ISL_DIM_TYPE.PARAM,
              isl.ISL_DIM_TYPE.IN,
              isl.ISL_DIM_TYPE.OUT,
              isl.ISL_DIM_TYPE.DIV]:
        for i in range(m.dim(t)):
            name = f"v{i}"
            try:
                name = m.dim_name(t, i)
            except Exception:
                pass
            names.append(t.name + " " + name)
    return names


def schedule_to_code(domain: isl.union_map, schedule: isl.map):
    tree = isl.schedule.from_domain(domain)
    tree = tree.insert_partial_schedule(schedule.as_multi_union_pw_aff())

    printer = isl.printer.from_file('/tmp/1.c')
    printer.set_output_format(isl.ISL_FORMAT.C)
    builder = isl.ast_build()
    ast: isl.ast_node = builder.node_from(tree)
    options = isl.ast_print_options.alloc()
    ast.print(printer, options)
    printer.flush()

    return CSource('/tmp/1.c')
