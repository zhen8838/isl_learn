import bpy
import isl
from petplot.support import *

def make_material(name, color_diffuse, color_specular, alpha):
    """
    Create a blender material.

    :param name: The name.
    :param color_diffuse: The diffuse color.
    :param color_specular: The specular color.
    :param alpha: The alpha channel.
    """
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color_diffuse
    mat.diffuse_shader = 'LAMBERT'
    mat.diffuse_intensity = 1.0
    mat.specular_color = color_specular
    mat.specular_shader = 'COOKTORR'
    mat.specular_intensity = 0.5
    mat.alpha = alpha
    mat.ambient = 1
    mat.use_transparency = True
    return mat

# Define a set of default colors
red = make_material('Red', (1,0.1,0.1), (1,1,1), 1)
green = make_material('Green', (0,1,0), (1,1,1), 1)
blue = make_material('Blue', (0,0.3,1), (1,1,1), 1)
white_trans = make_material('White', (1,1,1), (0.2,0.2,0.2), 0.4)
white = make_material('White', (1,1,1), (1,1,1), 1)
black = make_material('Black', (0,0,0), (1,1,1), 1)

def remove_default_cube():
    """
    Remove the cube that is in the blender default scene to get an empty scene.
    """
    bpy.data.objects["Cube"].select = True
    bpy.ops.object.delete()

def set_lamp():
    """
    Set the position of the default lamp.
    """
    bpy.data.objects["Lamp"].data.type = 'HEMI'
    bpy.data.objects["Lamp"].location = (20,20,20)

def set_camera(location=(34,42,24), rotation=(1.08, 0.013, 2.43)):
    """
    Set the location of the default camera.

    :param location: The camera's location.
    :param rotation: The camera's rotation.
    """
    bpy.data.objects["Camera"].location = location
    bpy.data.objects["Camera"].rotation_euler = rotation

def set_horizon():
    """
    Set the color of the horizon.
    """
    bpy.context.scene.world.horizon_color = (1,1,1)

def set_scene():
    """
    Prepare the scene for rendering.
    """
    remove_default_cube()
    set_lamp()
    set_camera()
    set_horizon()
    bpy.context.scene.render.resolution_percentage = 60

def save(filename):
    """
    Save the current blender project.

    :param filename: The location where to save the file.
    """
    bpy.ops.wm.save_as_mainfile(filepath=filename)

def render(filename):
    """
    Render the scene to a file.

    :param filename: The location where to store the rendered file.
    """
    bpy.data.scenes['Scene'].render.filepath = filename
    bpy.ops.render.render( write_still=True )

def print_plane(height0=10, height1=10, color=white_trans, dim=0, units=True):
    """
    Print a plane.

    :param color: The color of the plane.
    :param dim: The dimension controls the orientation and location of the plane.
    :param units: If units should be marked on the plane.
    :param height0: The width of the plane along the first direction.
    :param height1: The width of the plane along the second direction.
    """
    dim1 = None
    if dim == 0:
        rotation=(0,0,0)
        dim1 = 1
        rotation1 = (0,1.5708,0)
        dim2 = 0
        rotation2 = (1.5708,0,0)
        resize=(height1, height0, 1)
    if dim == 1:
        rotation=(0,1.5708,0)
        dim1 = 2
        rotation1 = (1.5708,0,0)
        dim2 = 1
        rotation2 = (0,0,0)
        resize=(1, height1, height0)
    if dim == 2:
        rotation=(1.5708,0,0)
        dim1 = 0
        rotation1 = (0,0,0)
        dim2 = 2
        rotation2 = (0,1.5708,0)
        resize=(height0, 1, height1)

    bpy.ops.mesh.primitive_plane_add(location=(0,0,0), rotation=rotation)

#   FIXME: Re-enable resize
#
#   This resize operation is necessary to actually form the planes. It
#   is disabled as this call currently fails with an invalid context
#   error when running directly in python with blender loaded as module.
#   This means we currently do not render those planes.
#
#   bpy.ops.transform.resize(value=resize)


    ob = bpy.context.active_object
    ob.data.materials.append(color)

    if not units:
        return

    for i in range(-height0,height0, 1):
        if i == 0:
            continue

        location=[0,0,0]
        location[dim1] = i

        if i % 5 == 0:
            radius = 0.01 * 2
        else:
            radius = 0.01

        bpy.ops.mesh.primitive_cylinder_add(vertices=128,
            radius=radius, depth=2*height1, rotation=rotation1,
            location=location)
        ob = bpy.context.active_object
        ob.data.materials.append(black)

    for i in range(-height1,height1, 1):
        if i == 0:
            continue

        location=[0,0,0]
        location[dim2] = i

        if i % 5 == 0:
            radius = 0.01 * 2
        else:
            radius = 0.01

        bpy.ops.mesh.primitive_cylinder_add(vertices=128,
            radius=radius, depth=2*height0, rotation=rotation2,
            location=location)
        ob = bpy.context.active_object
        ob.data.materials.append(black)

def print_axis(height, color, dim, unit_markers=True, labels=False):
    """
    Print the axis of a coordinate system.

    :param height: The length of the axis.
    :param color: The color of the axis.
    :param dim: The dimension for which the axis is printed.
    :param unit_marks: If unit markers should be plotted.
    :param labels: The labels that sould be printed next to the axis
    """
    if dim == 2:
        rotation = (0,0,0)
    if dim == 1:
        rotation = (-1.5708,0,0)
    if dim == 0:
        rotation = (0, 1.5708, 0)
            
    bpy.ops.mesh.primitive_cylinder_add(vertices=128, radius=0.1,
                                        depth=2 * (height+1), rotation=rotation,
                                        location=(0,0,0))
    ob = bpy.context.active_object
    ob.data.materials.append(color)
    location = [0,0,0]
    location[dim] = height+1
    bpy.ops.mesh.primitive_cone_add(vertices=128, radius1=0.2, radius2=0, depth=1,
        rotation=rotation, location=location)
    ob = bpy.context.active_object
    ob.data.materials.append(color)
    top = ob

    if labels != False and (labels == True or labels[dim] != False):
        location=[0,0,0]
        if dim == 0:
            rotation = (1.5708,0, 2 * 1.5708)
            location[2] = 0.3
            location[1] = 0
            location[0] = height - 0.1
        if dim == 1:
            location[1] = height - 0.8
            location[2] = 0.3
            location[0] = 0
            rotation = (1.5708,0,1.5708)
        if dim == 2:
            rotation = (1.5708,0, 2 * 1.5708)
            location[2] = height - 1.2
            location[1] = 0
            location[0] = -0.5
        bpy.ops.object.text_add(enter_editmode=True, location = location,
                    rotation=rotation)
        bpy.ops.font.delete()
        if labels[dim] == False:
            text = "i%d" % dim
        else:
            text = labels[dim]
        bpy.ops.font.text_insert(text=text)
        ob = bpy.context.active_object
        ob.data.size = 2
        ob.data.materials.append(color)

    if dim == 2:
        rotation = (-1.5708,0,0)
        rotation = (0,0,0)
    if dim == 1:
        rotation = (-1.5708,0,0)
    if dim == 0:
        rotation = (0, 1.5708, 0)

    if unit_markers:
        for i in range(-height,height+1, 1):
            location = [0, 0, 0]
            location[dim] = i
            if i % 5 == 0:
                depth = 2 * 0.05
            else:
                depth = 0.05
            bpy.ops.mesh.primitive_cylinder_add(vertices=128,
                radius=0.105, depth=depth, rotation=rotation,
                location=location)
            ob = bpy.context.active_object
            ob.data.materials.append(white)
    return top

def add_coordinate_system(size=[10,10,10], print_planes=[True, False, False],
                          unit_markers=True, labels=False):
    """
    Plot a coordinate system.

    :param size: The size of the coordinate system along the different
                 dimensions.
    :param print_plans: Either a single boolean value that enables or disables
                        the printing of all planes or a vector of booleans that
                        enables each plane individually.
    :param unit_markers: If unit markers should be printed on the plans and
                         axis.
    """
    axis = []
    dim = 0
    a = print_axis(size[2], black, dim, unit_markers, labels)
    axis.append(a)
    dim = 1
    a = print_axis(size[1], black, dim, unit_markers, labels)
    axis.append(a)
    dim = 2
    a = print_axis(size[0], black, dim, unit_markers, labels)
    axis.append(a)

    if print_planes != False:
        if (print_planes == True or print_planes[0] == True):
            print_plane(size[1], size[2], dim=0)
        if (print_planes == True or print_planes[1] == True):
            print_plane(size[0], size[1], dim=1)
        if (print_planes == True or print_planes[2] == True):
            print_plane(size[2], size[0], dim=2)

    return axis


def print_line(start, end):
    """
    Print a line between two points.
    """

    if not "islplot-tmp-line-a" in bpy.data.objects:
        """
        We only construct a sphere once and then copy subsequent spheres from
        this one. This speeds up blender, as we avoid the additional checking
        normally performed by the bpy.ops.mesh.* functions.
        """
        bpy.ops.mesh.primitive_uv_sphere_add(segments=1, ring_count=1,
            size=0.01, view_align=False, enter_editmode=False,
            location=(0,0,0), rotation=(0,0,0), layers=(True, False,
                False, False, False, False, False, False, False,
              False, False, False, False, False, False,
                False, False, False, False, False))
        A = bpy.context.active_object
        A.name = "islplot-tmp-line-a"
    else:
        A = bpy.data.objects["islplot-tmp-line-a"]

    if not "islplot-tmp-line-b" in bpy.data.objects:
        """
        We only construct a sphere once and then copy subsequent spheres from
        this one. This speeds up blender, as we avoid the additional checking
        normally performed by the bpy.ops.mesh.* functions.
        """
        bpy.ops.mesh.primitive_uv_sphere_add(segments=1, ring_count=1,
            size=0.01, view_align=False, enter_editmode=False,
            location=(0,0,0), rotation=(0,0,0), layers=(True, False,
                False, False, False, False, False, False, False,
                False, False, False, False, False, False,
                False, False, False, False, False))
        B = bpy.context.active_object
        B.name = "islplot-tmp-line-b"
    else:
        B = bpy.data.objects["islplot-tmp-line-b"]

    A.location = start
    B.location = end

    l = [A,B]
    draw_curve = bpy.data.curves.new('draw_curve','CURVE')
    draw_curve.dimensions = '3D'
    spline = draw_curve.splines.new('BEZIER')
    spline.bezier_points.add(len(l)-1)
    curve = bpy.data.objects.new('curve',draw_curve)
    bpy.context.scene.objects.link(curve)

    # Curve settings for new curve
    draw_curve.resolution_u = 64
    draw_curve.fill_mode = 'FULL'
    draw_curve.bevel_depth = 0.02
    draw_curve.bevel_resolution = 0.02

    # Assign bezier points to selection object locations
    for i in range(len(l)):
        p = spline.bezier_points[i]
        p.co = l[i].location
        p.handle_right_type="VECTOR"
        p.handle_left_type="VECTOR"

    curve.data.materials.append(black)

def print_face_borders(vertices, faces):
    """
    Print lines along the edges of a set of faces.

    :param faces: The faces for which to print the edges.
    :param vertices: The locations of the vertices.
    """
    for face in faces:
        for i in range(len(face)):
            a = vertices[face[i]]
            b = vertices[face[(i+1)%len(face)]]
            print_line(a, b)

def print_sphere(location):
    """
    Print a sphere at a given location.

    :param location: The location of the sphere.

    """

    if not "islplot-tmp-sphere" in bpy.data.objects:
        """
        We only construct a sphere once and then copy subsequent spheres from
        this one. This speeds up blender, as we avoid the additional checking
        normally performed by the bpy.ops.mesh.* functions.
        """
        bpy.ops.mesh.primitive_uv_sphere_add(segments=8, ring_count=8,
            size=0.1, view_align=False, enter_editmode=False,
            location=(0,0,0), rotation=(0,0,0), layers=(True, False,
                False, False, False, False, False, False, False,
                False, False, False, False, False, False,
                False, False, False, False, False))
        sphere = bpy.context.active_object
        sphere.name = "islplot-tmp-sphere"
        sphere.select = False
        sphere.data.materials.append(black)
    else:
        sphere = bpy.data.objects["islplot-tmp-sphere"]

    l = location
    ob = sphere.copy()
    ob.name = "Sphere (%d, %d, %d)" % (l[0], l[1], l[2])
    ob.location = l
    ob.data = sphere.data.copy()
    bpy.context.scene.objects.link(ob)
    return ob

def plot_bset_shape(bset_data, name, material, borders=True):
    """
    Given an basic set, plot the shape formed by the constraints that define
    the basic set.

    :param bset_data: The basic set to plot.
    :param name: The name the resulting mesh should have.
    :param material: The material to use for the faces.
    :param borders: Printer bordes of the shape.
    """
    vertices, faces = get_vertices_and_faces(bset_data)
    if borders:
        print_face_borders(vertices, faces)
    bpy.ops.object.add(type='MESH')
    ob = bpy.context.object
    ob.name = name
    me = ob.data
    me.from_pydata(vertices, [], faces)
    me.materials.append(material)
    me.update()
    ob.location[0] = 0
    ob.location[1] = 0
    ob.location[2] = 0
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
    return ob

def plot_set_points(set_data):
    points = bset_get_points(set_data, only_hull=True)
    for point in points:
        s = print_sphere(point)

def plot_bset(bset_data, color, name, add_spheres=True, borders=True):
    tile = plot_bset_shape(bset_data, name, color, borders)

    if add_spheres:
        plot_set_points(bset_data)
        bpy.context.scene.update()
    return tile

def plot_all(schedule, dimensions_to_visualize, add_spheres=False, borders=True,
             get_color=None):
    """
    Given a schedule, we print the individual tiles.

    TODO: This is just a quick hack. This code should be shared between the
          different renderers.

    :param schedule: The schedule to visualize.
    :parma dimensions_to_visualize: The map used to define the dimensions that
                                    will be visualized.
    :param add_spheres: Print the individual elements in the set.
    :param getColor: A function that takes a tile id and returns the
                     corresponding tile color.


    """
    tileIDSet = schedule.range()

    tileIDs = []
    tileIDSet.foreach_point(tileIDs.append)
    tileIDs = sort_points(tileIDs)

    for tileID in tileIDs:
        tileIDSet = isl.set.from_point(tileID)
        tileSet = schedule.intersect_range(tileIDSet).domain()
        tileSet = tileSet.apply(dimensions_to_visualize)
        assert tileSet == tileSet.convex_hull()
        tileSet = tileSet.convex_hull()

        color = get_color(tileID)
        name = "Tile " + str(get_point_coordinates(tileID))
        plot_bset(tileSet, color, name, add_spheres, borders)
