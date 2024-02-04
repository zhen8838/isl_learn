import isl
from typing import Tuple, List


def get_point_coordinates(point: isl.point, scale=1) -> List[int]:
  result = []
  for i in range(point.space().dim(isl.ISL_DIM_TYPE.SET)):
    result.append(int(point.get_coordinate_val(isl.ISL_DIM_TYPE.SET, i)
                      .get_num_si()) // scale)
  if(len(result) == 1):
    result.append(0)
  return result


def _vertex_to_rational_point(vertex: isl.vertex):
  """
  Given an n-dimensional vertex, this function returns an n-tuple consisting
  of pairs of integers. Each pair represents the rational value of the
  specific dimension with the first element of the pair being the nominator
  and the second element being the denominator.
  """
  expr : isl.multi_aff = vertex.get_expr()

  value = []

  for i in range(expr.dim(isl.ISL_DIM_TYPE.OUT)):
    subexpr : isl.aff = expr.get_at(i)
    val : isl.val = subexpr.get_constant_val()
    value.append((val.get_num_si(), val.get_den_si()))

  return value


def _vertex_get_coordinates(vertex, scale=1):
  """
  Get the coordinates of the an isl vertex as a tuple of float values.

  To extract the coordinates we first get the expression defining the vertex.
  This expression is given as a rational set that specifies its (possibly
  rational) coordinates. We then convert this set into the tuple we will
  return.

  Example:

  For a vertex defined by the rational set
  { rat: S[i, j] : 2i = 7 and 2j = 9 } we produce the output (7/2, 9/2).

  :param vertex: The vertex from which we extract the coordinates.
  """
  r = _vertex_to_rational_point(vertex)
  return [(1.0 * x[0] / x[1]) / scale for x in r]


def _is_vertex_on_constraint(vertex, constraint):
  """
  Given a vertex and a constraint, check if the vertex is on the plane defined
  by the constraint. For inequality constraints, the plane we look at is the
  extremal plane that separates the elements that fulfill an inequality
  constraint from the elements that do not fulfill this constraints.
  """
  r = _vertex_to_rational_point(vertex)

  dims = constraint.space.dim(isl.ISL_DIM_TYPE.SET)
  v = []
  for d in range(dims):
    v.append(constraint.get_coefficient_val(isl.ISL_DIM_TYPE.SET, d).get_num_si())

  summ = 0

  import numpy
  for i in range(dims):
    prod = 1
    for j in range(dims):
      if i == j:
        prod *= r[j][0]
      else:
        prod *= r[j][1]
    summ += v[i] * prod

  constant = constraint.get_constant_val().get_num_si()
  summ += numpy.product([x[1] for x in r]) * constant

  return int(summ) == 0


def bset_get_vertex_coordinates(bset_data: isl.basic_set, scale=1):
  """
  Given a basic set return the list of vertices at the corners.

  :param bset_data: The basic set to get the vertices from
  """

  # Get the vertices.
  vertices = []
  bset_data.compute_vertices().foreach_vertex(vertices.append)
  def f(x): 
    coor = _vertex_get_coordinates(x, scale)
    return (coor[1], coor[0]) 
  vertices = list(map(f, vertices))

  if len(vertices) <= 1:
    return vertices

  # Sort the vertices in clockwise order.
  #
  # We select a 'center' point that lies within the convex hull of the
  # vertices. We then sort all points according to the direction (given as an
  # angle in radiens) they lie in respect to the center point.
  from math import atan2 as _atan2
  center = ((vertices[0][0] + vertices[1][0]) / 2.0,
            (vertices[0][1] + vertices[1][1]) / 2.0)

  def f(x): return _atan2(x[0] - center[0], x[1] - center[1])
  vertices = sorted(vertices, key=f)
  return vertices


from math import sqrt
from math import degrees
from math import acos


def cross(a, b):
  c = [a[1] * b[2] - a[2] * b[1],
       a[2] * b[0] - a[0] * b[2],
       a[0] * b[1] - a[1] * b[0]]

  return c


def sub(A, B):
  return [A[0] - B[0], A[1] - B[1], A[2] - B[2]]


def norm(A, B, C):
  return(cross(sub(A, C), sub(B, C)))


def dotProduct(A, B):
  return A[0] * B[0] + A[1] * B[1] + A[2] * B[2]


def magnitude(A):
  return sqrt(A[0] * A[0] + A[1] * A[1] + A[2] * A[2])


def formular(A, B):
  res = dotProduct(A, B) / (magnitude(A) * magnitude(B))
  res = float(str(res))
  # Due to rounding errors res may be smaller than one. We fix this here.
  res = max(-1, res)
  res = acos(res)
  res = degrees(res)
  return res


def angle(Q, M, O, N):
  if Q == M:
    return 0
  OM = sub(M, O)
  OQ = sub(Q, O)

  sig = dotProduct(N, cross(OM, OQ))

  if sig >= 0:
    return formular(OQ, OM)
  else:
    return -formular(OQ, OM)


def get_vertices_for_constraint(vertices, constraint):
  """
  Return the list of vertices within a hyperspace.

  Given a constraint and a list of vertices, we filter the list of vertices
  such that only the vertices that are on the plane defined by the constraint
  are returned. We then sort the vertices such that the order defines a
  convex shape.
  """
  points = []
  for v in vertices:
    if _is_vertex_on_constraint(v, constraint):
      points.append(_vertex_get_coordinates(v))

  if len(points) == 0:
    return None

  points.sort()
  import itertools
  points = list(points for points, _ in itertools.groupby(points))

  A = points[0]
  if len(points) == 1:
    return [A]
  B = points[1]
  if len(points) == 2:
    return [A, B]
  C = points[2]
  N = norm(A, B, C)
  center = [(A[0] + B[0]) / 2, (A[1] + B[1]) / 2, (A[2] + B[2]) / 2]
  def f(a): return angle(A, a, center, N)
  points = sorted(points, key=f)
  return points


def isSubset(parent, child):
  if len(parent) <= len(child):
    return False
  for c in child:
    contained = False
    for p in parent:
      if p == c:
        contained = True
        break

    if not contained:
      return False

  return True


def bset_get_faces(basicSet):
  """
  Get a list of faces from a basic set

  Given a basic set we return a list of faces, where each face is represented
  by a list of restricting vertices. The list of vertices is sorted in
  clockwise (or counterclockwise) order around the center of the face.
  Vertices may have rational coordinates. A vertice is represented as a three
  tuple.
  """
  faces = []
  vertices = []
  basicSet.compute_vertices().foreach_vertex(vertices.append)
  def f(c): return faces.append(get_vertices_for_constraint(vertices, c))
  basicSet.foreach_constraint(f)

  # Remove empty elements, duplicates and subset of elements
  faces = filter(lambda x: not x == None, faces)
  faces = list(faces)
  faces = [x for x in faces if not
           any(isSubset(y, x) for y in faces if x is not y)]
  faces.sort()
  import itertools
  faces = list(faces for faces, _ in itertools.groupby(faces))
  return faces


def set_get_faces(set_data):
  """
  Get a list of faces from a set

  Given a basic set we return a list of faces, where each face is represented
  by a list of restricting vertices. The list of vertices is sorted in
  clockwise (or counterclockwise) order around the center of the face.
  Vertices may have rational coordinates. A vertice is represented as a three
  tuple.
  """

  bsets = []
  def f(x): return bsets.append(x)
  set_data.foreach_basic_set(f)
  return list(map(bset_get_faces, bsets))


def make_tuple(vertex):
  return (vertex[0], vertex[1], vertex[2])


def get_vertex_to_index_map(vertexlist):
  res = {}
  i = 0
  for v in vertexlist:
    res[make_tuple(v)] = i
    i += 1
  return res


def translate_faces_to_indexes(faces, vertexmap):
  """
  Given a list of faces, translate the vertex defining it from their explit
  offsets to their index as provided by the vertexmap, a mapping from vertices
  to vertex indices.
  """
  new_faces = []
  for face in faces:
    new_face = []
    for vertex in face:
      new_face.append(vertexmap[make_tuple(vertex)])
    new_faces.append(new_face)
  return new_faces


def get_vertices_and_faces(set_data):
  """
  Given an isl set, return a tuple that contains the vertices and faces of
  this set. The vertices are sorted in lexicographic order. In the faces,
  the vertices are referenced by their position within the vertex list. The
  vertices of a face are sorted such that connecting subsequent vertices
  yields a convex form.
  """
  data = set_get_faces(set_data)
  if len(data) == 0:
    return ([], [])

  faces = data[0]
  vertices = [vertex for face in faces for vertex in face]
  vertices.sort()
  import itertools
  vertices = list(vertices for vertices, _ in itertools.groupby(vertices))
  vertexmap = get_vertex_to_index_map(vertices)

  faces = translate_faces_to_indexes(faces, vertexmap)
  return (vertices, faces)


def _constraint_make_equality_set(x):
  e = isl.constraint.alloc_equality(x.get_local_space())
  e = e.set_constant_val(x.get_constant_val().get_num_si())

  for i in range(x.space.dim(isl.ISL_DIM_TYPE.SET)):
    e = e.set_coefficient_val(isl.ISL_DIM_TYPE.SET, i,
                              x.get_coefficient_val(isl.ISL_DIM_TYPE.SET, i).get_num_si())
  for i in range(x.space.dim(isl.ISL_DIM_TYPE.PARAM)):
    e = e.set_coefficient_val(isl.ISL_DIM_TYPE.PARAM, i,
                              x.get_coefficient_val(isl.ISL_DIM_TYPE.PARAM, i).get_num_si())

  return isl.basic_set.universe(x.space).add_constraint(e)


def bset_get_points(bset_data, only_hull=False, scale=1):
  """
  Given a basic set return the points within this set

  :param bset_data: The set that contains the points.
  :param only_hull: Only return the point that are on the hull of the set.
  :param scale: Scale the values.
  """

  if only_hull:
    hull = [None]
    hull[0] = isl.set.empty(bset_data.space)

    def add(c):
      const_eq = _constraint_make_equality_set(c)
      const_eq = const_eq.intersect(bset_data)
      hull[0] = hull[0].union(const_eq)
    bset_data.foreach_constraint(add)
    bset_data = hull[0]

  points = []
  def f(x): return points.append(get_point_coordinates(x, scale))
  bset_data.foreach_point(f)
  points = sorted(points)
  return points


def get_rectangular_hull(set_data: isl.set, offset=0):
  uset_data = isl.set.universe(set_data.get_space())

  for dim in range(0, 2):
    ls = isl.local_space.from_space(set_data.get_space())
    c = isl.constraint.alloc_inequality(ls)
    incr = isl.map("{{[i]->[i+{0}]}}".format(offset))
    decr = isl.map("{{[i]->[i-{0}]}}".format(offset))

    dim_val = isl.aff.zero_on_domain(ls).set_coefficient_si(isl.ISL_DIM_TYPE.IN, dim, 1)
    dim_val = isl.pw_aff(dim_val)
    dim_val = dim_val.as_map()

    space = dim_val.get_space()
    dim_cst = isl.map.universe(space)
    max_set = isl.set.from_pw_aff(set_data.dim_max(dim))
    dim_cst = dim_cst.intersect_range(max_set)
    dim_cst = dim_cst.apply_range(incr)

    diff = isl.map.lex_le_map(dim_val, dim_cst)
    uset_data = uset_data.intersect(diff.domain())

    dim_cst = isl.map.universe(space)
    min_set = isl.set.from_pw_aff(set_data.dim_min(dim))
    dim_cst = dim_cst.intersect_range(min_set)
    dim_cst = dim_cst.apply_range(decr)

    diff = isl.map.lex_ge_map(dim_val, dim_cst)
    uset_data = uset_data.intersect(diff.domain())

  return uset_data


def cmp_points(a, b):
  a = isl.set.from_point(a)
  b = isl.set.from_point(b)
  if a.lex_le_set(b).is_empty():
    return 1
  else:
    return -1


def cmp_to_key(mycmp):
  'Convert a cmp= function into a key= function'
  class Key(object):
    def __init__(self, obj, *args):
      self.obj = obj

    def __lt__(self, other):
      return mycmp(self.obj, other.obj) < 0

    def __gt__(self, other):
      return mycmp(self.obj, other.obj) > 0

    def __eq__(self, other):
      return mycmp(self.obj, other.obj) == 0

    def __le__(self, other):
      return mycmp(self.obj, other.obj) <= 0

    def __ge__(self, other):
      return mycmp(self.obj, other.obj) >= 0

    def __ne__(self, other):
      return mycmp(self.obj, other.obj) != 0
  return Key


def sort_points(points):
  """
  Given a list of points, sort them lexicographically.

  :param points: The list of points that will be sorted.
  """
  return sorted(points, key=cmp_to_key(cmp_points))


__all__ = ['bset_get_vertex_coordinates', 'bset_get_faces', 'set_get_faces',
           'get_vertices_and_faces', 'get_point_coordinates', 'bset_get_points',
           'get_rectangular_hull', 'sort_points']
