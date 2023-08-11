import isl
import pet

from typing import Callable


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