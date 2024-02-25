import isl
from typing import Union, List
from petplot.support import *
import matplotlib.pyplot as _plt
from mpl_toolkits.mplot3d import Axes3D
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
        ax.scatter(dimX, dimY, dimZ, color=color, marker=marker)
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
