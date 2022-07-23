import islpy as _islpy
#from islpy import *

def get_point_coordinates(point, scale=1):
    result = []
    for i in range(point.get_space().dim(_islpy.dim_type.set)):
        result.append(int(point.get_coordinate_val(_islpy.dim_type.set, i)
            .get_num_si())/ scale)

    return result

def _vertex_to_rational_point(vertex):
    """
    Given an n-dimensional vertex, this function returns an n-tuple consisting
    of pairs of integers. Each pair represents the rational value of the
    specific dimension with the first element of the pair being the nominator
    and the second element being the denominator.
    """
    expr = vertex.get_expr()

    value = []

    for i in range(expr.dim(dim_type.out)):
        subexpr = expr.get_aff(i)
        val = subexpr.get_constant_val()
        value.append((val.get_num_si(), val.get_den_val().to_python()))

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
    return [(1.0 * x[0] / x[1])/scale for x in r]

def _is_vertex_on_constraint(vertex, constraint):
    """
    Given a vertex and a constraint, check if the vertex is on the plane defined
    by the constraint. For inequality constraints, the plane we look at is the
    extremal plane that separates the elements that fulfill an inequality
    constraint from the elements that do not fulfill this constraints.
    """
    r = _vertex_to_rational_point(vertex)

    dims = constraint.space.dim(dim_type.set)
    v = []
    for d in range(dims):
        v.append(constraint.get_coefficient_val(dim_type.set, d).get_num_si())

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


def bset_get_vertex_coordinates(bset_data, scale=1):
    """
    Given a basic set return the list of vertices at the corners.

    :param bset_data: The basic set to get the vertices from
    """

    # Get the vertices.
    vertices = []
    bset_data.compute_vertices().foreach_vertex(vertices.append)
    f = lambda x: _vertex_get_coordinates(x, scale)
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
    f = lambda x: _atan2(x[0]-center[0], x[1]-center[1])
    vertices = sorted(vertices, key=f)
    return vertices


from math import sqrt
from math import degrees
from math import acos

def cross(a, b):
    c = [a[1]*b[2] - a[2]*b[1],
         a[2]*b[0] - a[0]*b[2],
         a[0]*b[1] - a[1]*b[0]]

    return c

def sub(A,B):
    return [A[0]-B[0], A[1]-B[1], A[2]-B[2]]

def norm(A,B,C):
    return(cross(sub(A,C),sub(B,C)))

def dotProduct(A,B):
    return A[0] * B[0] + A[1] * B[1] + A[2] * B[2]

def magnitude(A):
    return sqrt(A[0]*A[0] + A[1]*A[1] + A[2]*A[2])

def formular(A,B):
    res = dotProduct(A,B) / (magnitude(A) * magnitude(B))
    res = float(str(res))
    # Due to rounding errors res may be smaller than one. We fix this here.
    res = max(-1, res)
    res = acos(res)
    res = degrees(res)
    return res

def angle(Q,M,O,N):
    if Q == M:
        return 0
    OM = sub(M,O)
    OQ = sub(Q,O)

    sig = dotProduct(N,cross(OM, OQ))

    if sig >= 0:
        return formular(OQ,OM)
    else:
        return -formular(OQ,OM)

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
    points = list(points for points,_ in itertools.groupby(points))

    A = points[0]
    if len(points) == 1:
        return [A]
    B = points[1]
    if len(points) == 2:
        return [A, B]
    C = points[2]
    N = norm(A,B,C)
    center = [(A[0] + B[0]) / 2, (A[1] + B[1]) / 2, (A[2] + B[2]) / 2]
    f = lambda a: angle(A, a, center, N)
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
    f = lambda c: faces.append(get_vertices_for_constraint(vertices, c))
    basicSet.foreach_constraint(f)

    # Remove empty elements, duplicates and subset of elements
    faces = filter(lambda x: not x == None, faces)
    faces = list(faces)
    faces = [x for x in faces if not
                any(isSubset(y, x) for y in faces if x is not y)]
    faces.sort()
    import itertools
    faces = list(faces for faces,_ in itertools.groupby(faces))
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
    f = lambda x: bsets.append(x)
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
    e = Constraint.equality_alloc(x.get_local_space())
    e = e.set_constant_val(x.get_constant_val().get_num_si())

    for i in range(x.space.dim(dim_type.set)):
        e = e.set_coefficient_val(dim_type.set, i,
                x.get_coefficient_val(dim_type.set, i).get_num_si())
    for i in range(x.space.dim(dim_type.param)):
        e = e.set_coefficient_val(dim_type.param, i,
                x.get_coefficient_val(dim_type.param, i).get_num_si())

    return BasicSet.universe(x.space).add_constraint(e)

def bset_get_points(bset_data, only_hull=False, scale=1):
    """
    Given a basic set return the points within this set

    :param bset_data: The set that contains the points.
    :param only_hull: Only return the point that are on the hull of the set.
    :param scale: Scale the values.
    """

    if only_hull:
          hull = [None]
          hull[0] = Set.empty(bset_data.space)
          def add(c):
            const_eq = _constraint_make_equality_set(c)
            const_eq = const_eq.intersect(bset_data)
            hull[0] = hull[0].union(const_eq)
          bset_data.foreach_constraint(add)
          bset_data = hull[0]

    points = []
    f = lambda x: points.append(get_point_coordinates(x, scale))
    bset_data.foreach_point(f)
    points = sorted(points)
    return points

def get_rectangular_hull(set_data, offset=0):
    uset_data = Set.universe(set_data.get_space())

    for dim in range(0,2):
        ls = LocalSpace.from_space(set_data.get_space())
        c = Constraint.inequality_alloc(ls)
        incr = Map("{{[i]->[i+{0}]}}".format(offset))
        decr = Map("{{[i]->[i-{0}]}}".format(offset))

        dim_val = Aff.zero_on_domain(ls).set_coefficient_val(dim_type.in_, dim,
                                                             1)
        dim_val = PwAff.from_aff(dim_val)
        dim_val = Map.from_pw_aff(dim_val)

        space = dim_val.get_space()
        dim_cst = Map.universe(space)
        max_set = Set.from_pw_aff(set_data.dim_max(dim))
        dim_cst = dim_cst.intersect_range(max_set)
        dim_cst = dim_cst.apply_range(incr)

        diff = Map.lex_le_map(dim_val, dim_cst)
        uset_data = uset_data.intersect(diff.domain())

        dim_cst = Map.universe(space)
        min_set = Set.from_pw_aff(set_data.dim_min(dim))
        dim_cst = dim_cst.intersect_range(min_set)
        dim_cst = dim_cst.apply_range(decr)

        diff = Map.lex_ge_map(dim_val, dim_cst)
        uset_data = uset_data.intersect(diff.domain())

    return uset_data

def cmp_points(a, b):
    a = Set.from_point(a)
    b = Set.from_point(b)
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

# Variables to give each set a different id
# ID is used to give each set a different size    
dict1  = {} 
cal = 1.0000
def set_get_points(set_):
    """
    Given a set, returns the points in a list
    :param set_: a (basic|union|)set 
    """
    points = []
    set_.foreach_basic_set(lambda bs: points.extend(bset_get_points(bs)))
    return points

def _set_get_points_tagged(set_):
    """
    Given a set returns a json object that contains the set information as
    {name , points, tiles, id}
    :param set_ : a basic_set
    """
    global dict1
    global cal
    points = set_get_points(set_)
    space = set_.get_space()
    name = space.get_tuple_name(_islpy.dim_type.out)
    if(name in dict1.keys()):
        ids = dict1[name]
    else :      
        tempset = -(cal*cal*cal) + 30*cal*cal - 20*cal + 400;
        tempset = 2200 /tempset
        cal = cal + 1
        dict1[name] = tempset
        ids = dict1[name]       
    return {"name": name, "points": points, "tiles": [], "id" : ids}

def _get_random_string():
    import random
    import string
    return "".join(random.choice(string.ascii_lowercase) for i in range(8))

def plot_uset_points_html(uset):
    """
    Given a (union|basic|)set, returns a json object where each basic_set is represented as
    {name , points, tiles, id}

    :param uset : a (union|basic|)set
    """
    global dict1
    global cal
    dict1 = {}
    cal = 1.0000
    import json
    set_list = []
    uset.foreach_set(lambda s: set_list.append(_set_get_points_tagged(s)))
    random_id = _get_random_string()
    #<div id="%s"></div><script>let uset =...   plotUnionSetCombined("#%s", uset, "%s");</script>
    s = """
    let uset = %s;
    """ % (json.dumps(set_list))
    return s

# Variables to give each set a different id
# ID is used to give each set a different size
count = 1.0000
dict2 = {} 
def _map_get_points_tagged(map_):
    """
    Given a map returns a json object that contains the set information as
    {name of domain set , name of range set, points, tiles, id}
    points are represented as a list of (x1,y1,x2,y2) tuple, where x1,y1 are correspoding coordinates of
    domain set point and x2,y2 are coordiantes of range set point of a relation.
    :param set_ : a basic_set
    """    
    global count
    global dict2
    ids = {}
    full_points = []
    domain_pts = []
    map_.domain().foreach_point(domain_pts.append)
    for dp in domain_pts:
        range_points = []
        limited = map_.intersect_domain(_islpy.basic_set(dp))
        limited.range().foreach_point(range_points.append)
        for rp in range_points:
            full_points.append(get_point_coordinates(dp) + get_point_coordinates(rp))
    space_map = map_.get_space()
    name_in = space_map.get_tuple_name(_islpy.dim_type.in_)
    name_out = space_map.get_tuple_name(_islpy.dim_type.out)
    if(name_in in dict2.keys()):
        ids[name_in] = dict2[name_in]
    else :      
        tempmap = -(count*count*count) + 30*count*count - 20*count + 400;
        tempmap = 2200 /tempmap
        count = count + 1
        dict2[name_in] = tempmap
        ids[name_in] = dict2[name_in]
    if(name_out in dict2.keys()):
        ids[name_out] = dict2[name_out]
    else :      
        tempmap = -(count*count*count) + 30*count*count - 20*count + 400;
        tempmap = 2200 /tempmap
        count = count + 1
        dict2[name_out] = tempmap
        ids[name_out] = dict2[name_out]
    return {"from": name_in, "to": name_out, "points": full_points, "tiles": [], "id" : ids}
def plot_umap_points_html(umap):
    """
    Given a (union|basic|)set, returns a json object where each basic_set is represented as
    {name of domain set, name of range set, points, tiles, id}

    :param uset : a (union|basic|)map
    """
    global count
    global dict2
    count = 1.0000
    dict2 = {} 
    import json
    map_list = []
    umap.foreach_map(lambda m: map_list.append(_map_get_points_tagged(m)))
    random_id =_get_random_string()
    s= """
    let umap = %s;
    """ % (json.dumps(map_list))
    return s

def plotString(islobj):
    from random import randint
    import isl
    id = randint(0,999999999999)
    t = islobj.__class__
    is_set = (t == isl.basic_set or t == isl.set or t == isl.union_set)
    is_map = (t == isl.basic_map or t == isl.map or t == isl.union_map)

    assert (is_set or is_map), "Expected set of map"

    if is_set:
      x = plot_uset_points_html(islobj)
    else:
      x = plot_umap_points_html(islobj)

    out = "<div class='graphics' id='graphics_%d'>" % id
    x = x[16:]
    x = x[0:len(x) - 6]
    call = ""

    if is_set:
      call += "plotUnionSetCombined("
    else:
      call += "plotUnionMapClosed("

    call += "\"#graphics_%d\"" % id
    call += ","
    call += x
    call += ","
    call += "\"#graphics_%d\"" % id
    call += ")"
    out += "<img src='data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7' onload='%s'>" % call
    out += "</div>"
    return out

# Given a double nested list print the content of this list as HTML
# table.
def getTableFull(data):
    string = "<table class='table table-striped'>"
    first = True
    for row in data:
        if first:
            string += '<thead>'
        string += '<tr>'
        for col in row:
            if first:
                string += '<th>'
            else:
                string += '<td>'
            string += str(col)
            if first:
                string += '</th>'
            else:
                string += '</td>'
        string += '</tr>'
        if first:
            first = False
            string += '</thead>'
    string += "</table>"
    return string

# Given a single element, a list, or a nested list and ensure it is always
# a double nested list. This functionality is useful canonicalize data that
# should be printed as table.
def tableize(data):
    try:
        l = len(data)
    except:
        data = [data]
    try:
        l = len(data[0])
    except:
        data = [data]
    return data

# Given a double nested list, check if elements in this data structure are
# isl sets or maps. In case they are, format them as graphical plot.
def formatTable(data):
    new_data = []
    for row in data:
        new_row = []
        for col in row:
            t = col.__class__
            import isl
            is_set = t == isl.basic_set or t == isl.set or t == isl.union_set
            is_map = t == isl.basic_map or t == isl.map or t == isl.union_map
            if is_set or is_map:
                col = plotString(col)
            new_row.append(col)
        new_data.append(new_row)
    return new_data

# Given a single data item, a list of data items, or a double nested list of
# data items, which can either be converted to 'str' or which are of type
# isl set or map, format these data items as HTML table. isl sets and maps
# are formatted as d3 visualization.
def format(data):
    data = tableize(data)
    data = formatTable(data)
    data = getTableFull(data)
    return data

# Format and then plot a set of isl constructs using the format function.
def plot(data):
    print(format(data))

__all__ = ['bset_get_vertex_coordinates', 'bset_get_faces', 'set_get_faces',
           'get_vertices_and_faces', 'get_point_coordinates', 'bset_get_points',
           'get_rectangular_hull', 'sort_points', 'set_get_points', 'set_get_points_tagged', '_get_random_string', 'plot_uset_points_html', '_map_get_points_tagged', 'plot_umap_points_html', 'format', 'plot']
