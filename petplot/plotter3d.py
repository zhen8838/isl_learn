import isl
from typing import Tuple, Union, List
from petplot.support import *
import matplotlib.pyplot as _plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


def plot_set_points_3d(set_datas: Union[isl.set, List[isl.set]], color="black", size=10, marker="o", scale=1) -> Axes3D:
    """
    Plot the individual points of a two dimensional isl set.

    :param set_datas: The islpy.Set to plot.
    :param color: The color of the points.
    :param size: The diameter of the points.
    :param marker: The marker used to mark a point.
    :param scale: Scale the values.
    """
    ax: Axes3D = _plt.subplot(projection='3d')
    if isinstance(set_datas, (isl.set, isl.basic_set, isl.union_set)):
        set_datas = [set_datas]
    for set_data in set_datas:
        points = bset_get_points(set_data, scale=scale)
        dimX = [x[2] for x in points]
        dimY = [x[1] for x in points]
        dimZ = [x[0] for x in points]
        ax.scatter(dimX, dimY, dimZ, s=size, color=color, marker=marker)
    return ax


def plot_bset_shape_3d(ax: Axes3D, bset_data: isl.basic_set, show_vertices=False, color="gray",
                       alpha=1.0,
                       vertex_color=None,
                       vertex_marker="o", vertex_size=10,
                       scale=1, border=0) -> Axes3D:
    """
    Given an basic set, plot the shape formed by the constraints that define
    the basic set.

    :param bset_data: The basic set to plot.
    :param show_vertices: Show the vertices at the corners of the basic set's
                          shape.
    :param color: The background color of the shape.
    :param alpha: The alpha value to use for the shape.
    :param vertex_color: The color of the vertex markers.
    :param vertex_marker: The marker used to draw the vertices.
    :param vertex_size: The size of the vertices.
    :param border: Increase the size of the area filled with the background
                   by the value given as 'border'.
    :param scale: Scale the values.
    """

    assert bset_data.is_bounded(), "Expected bounded set"

    if not vertex_color:
        vertex_color = color

    vertices: np.ndarray = bset_get_vertex_coordinates(bset_data, scale=scale)

    if show_vertices:
        raise NotImplementedError()

    if len(vertices) == 0:
        return
    ax.add_collection3d(Poly3DCollection([vertices], color=color, alpha=alpha))


class Arrow3D(FancyArrowPatch):

    def __init__(self, xyz, dxdydz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = xyz
        self._dxdydz = dxdydz

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def _plot_arrow(ax, xyz: Tuple[int, int, int], dxdydz: Tuple[int, int, int], *args, **kwargs):
    arrow = Arrow3D(xyz, dxdydz, *args, **kwargs)
    ax.add_artist(arrow)


def plot_map_3d(map: Union[isl.map, isl.basic_map], edge_style="-|>", edge_width=1,
                start_color="blue", end_color="orange", line_color="black", marker_size=7,
                scale=1, shrink=6, ax: Axes3D = None) -> Axes3D:
    """
    Given a map from a two dimensional set to another two dimensional set this
    functions prints the relations in this map as arrows going from the input
    to the output element.

    :param map_datas: The islpy.Map to plot.
    :param color: The color of the arrows.
    :param edge_style: The style used to plot the arrows.
    :param edge_width: The width used to plot the arrows.
    :param shrink: The distance before around the start/end which is not plotted
                   to.
    :param scale: Scale the values.
    """
    if ax is None:
        ax = _plt.subplot(projection='3d')
    if isinstance(map, isl.basic_map):
        all_start: List[Tuple[int, int, int]] = []
        all_ends: List[Tuple[int, int, int]] = []
        start_points: List[isl.point] = []
        map_datas: List[isl.basic_map] = []
        map.foreach_basic_map(lambda bmap: map_datas.append(bmap))
        for map_data in map_datas:
            map_data.range().foreach_point(start_points.append)
            for start in start_points:
                end_points: List[isl.point] = []
                limited = map_data.intersect_range(isl.set(start))
                limited.domain().foreach_point(end_points.append)
                s = get_point_coordinates(start, scale)
                s.reverse()
                all_start.append(s)
                for end in end_points:
                    e = get_point_coordinates(end, scale)
                    e.reverse()
                    all_ends.append(e)
                    _plot_arrow(ax, s, [p[0] - p[1] for p in zip(e, s)],
                                arrowstyle=edge_style, linewidth=edge_width, color=line_color, mutation_scale=edge_width*10,
                                shrinkA=shrink, shrinkB=shrink)
            ax.scatter([x[0] for x in all_start], [x[1] for x in all_start], [
                       x[2] for x in all_start], color=start_color, marker="o", s=marker_size)
            ax.scatter([x[0] for x in all_ends], [x[1] for x in all_ends], [
                       x[2] for x in all_ends], color=end_color, marker="o", s=marker_size)
    elif isinstance(map, isl.map):
        map.foreach_basic_map(lambda bmap: plot_map_3d(
            bmap, edge_style, edge_width, start_color, end_color, line_color, marker_size, scale, shrink, ax))
    elif isinstance(map, isl.union_map):
        map.foreach_map(lambda bmap: plot_map_3d(
            bmap, edge_style, edge_width, start_color, end_color, line_color, marker_size, scale, shrink, ax))
    return ax


def plot_set_shapes_3d(set_data: Union[isl.set, isl.basic_set], *args, **kwargs) -> Axes3D:
    """
    Plot a set of concex shapes for the individual basic sets this set consists
    of.

    :param set_data: The set to plot.
    """
    ax: Axes3D = _plt.subplot(projection='3d')
    if isinstance(set_data, isl.basic_set):
        plot_bset_shape_3d(ax, set_data, **kwargs)
    assert set_data.is_bounded(), "Expected bounded set"

    set_data.foreach_basic_set(lambda x: plot_bset_shape_3d(ax, x, **kwargs))
    return ax


# def plot_bset_shape_3d(bset_data: isl.basic_set, show_vertices=False, color="gray",
#                        alpha=1.0,
#                        vertex_color=None,
#                        vertex_marker="o", vertex_size=10,
#                        scale=1, border=0) -> Axes3D:
#     """
#     Given an basic set, plot the shape formed by the constraints that define
#     the basic set.

#     :param bset_data: The basic set to plot.
#     :param show_vertices: Show the vertices at the corners of the basic set's
#                           shape.
#     :param color: The background color of the shape.
#     :param alpha: The alpha value to use for the shape.
#     :param vertex_color: The color of the vertex markers.
#     :param vertex_marker: The marker used to draw the vertices.
#     :param vertex_size: The size of the vertices.
#     :param border: Increase the size of the area filled with the background
#                    by the value given as 'border'.
#     :param scale: Scale the values.
#     """

#     assert bset_data.is_bounded(), "Expected bounded set"

#     if not vertex_color:
#         vertex_color = color

#     vertices = bset_get_vertex_coordinates(bset_data, scale=scale)

#     if show_vertices:
#         dimX = [x[1] for x in vertices]
#         dimY = [x[0] for x in vertices]
#         _plt.plot(dimX, dimY, vertex_marker, markersize=vertex_size,
#                   color=vertex_color)

#     if len(vertices) == 0:
#         return


__all__ = ['plot_set_points_3d', 'plot_set_shapes_3d']
