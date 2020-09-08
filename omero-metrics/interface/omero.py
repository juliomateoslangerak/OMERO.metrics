# OMERO imports
import omero.gateway as gw
from omero.constants import metadata, namespaces
from omero import model
from omero.model import enums, LengthI
from omero import grid
from omero import rtypes
import omero_rois

# Generic imports
import numpy as np
from itertools import product
from json import dumps
from random import choice
from string import ascii_letters

COLUMN_TYPES = {
    "string": grid.StringColumn,
    "long": grid.LongColumn,
    "bool": grid.BoolColumn,
    "double": grid.DoubleColumn,
    "long_array": grid.LongArrayColumn,
    "float_array": grid.FloatArrayColumn,
    "double_array": grid.DoubleArrayColumn,
    "image": grid.ImageColumn,
    "dataset": grid.DatasetColumn,
    "plate": grid.PlateColumn,
    "well": grid.WellColumn,
    "roi": grid.RoiColumn,
    "mask": grid.MaskColumn,
    "file": grid.FileColumn,
}


def open_connection(username, password, host, port, group=None, secure=False):
    conn = gw.BlitzGateway(
        username=username,
        passwd=password,
        host=host,
        port=port,
        group=group,
        secure=secure,
    )
    try:
        conn.connect()
    except Exception as e:
        raise e
    return conn


def close_connection(connection):
    connection.close()


def get_image(connection, image_id):
    try:
        image = connection.getObject("Image", image_id)
    except Exception as e:
        raise e
    return image


def get_dataset(connection, dataset_id):
    try:
        dataset = connection.getObject("Dataset", dataset_id)
    except Exception as e:
        raise e
    return dataset


def get_project(connection, project_id):
    try:
        project = connection.getObject("Project", project_id)
    except Exception as e:
        raise e
    return project


def get_image_shape(image):
    try:
        image_shape = (
            image.getSizeZ(),
            image.getSizeC(),
            image.getSizeT(),
            image.getSizeY(),
            image.getSizeX(),
        )
    except Exception as e:
        raise e

    return image_shape


def get_pixel_size(image, order="ZXY"):
    pixels = image.getPrimaryPixels()

    order = order.upper()
    if order not in ["ZXY", "ZYX", "XYZ", "XZY", "YXZ", "YZX"]:
        raise ValueError("The provided order for the axis is not valid")
    pixel_sizes = tuple()
    for a in order:
        pixel_sizes += (getattr(pixels, f"getPhysicalSize{a}")().getValue(),)
    return pixel_sizes


def get_pixel_size_units(image):
    pixels = image.getPrimaryPixels()

    pixel_size_units = (
        pixels.getPhysicalSizeX().getUnit().name,
        pixels.getPhysicalSizeY().getUnit().name,
        pixels.getPhysicalSizeZ().getUnit().name,
    )
    return pixel_size_units


def get_intensities(
    image, z_range=None, c_range=None, t_range=None, x_range=None, y_range=None
):
    """Returns a numpy array containing the intensity values of the image
    Returns an array with dimensions arranged as zctxy
    """
    image_shape = get_image_shape(image)

    # Decide if we are going to call getPlanes or getTiles
    if not x_range and not y_range:
        whole_planes = True
    else:
        whole_planes = False

    ranges = list(range(5))
    for dim, r in enumerate([z_range, c_range, t_range, y_range, x_range]):
        # Verify that requested ranges are within the available data
        if r is None:  # Range is not specified
            ranges[dim] = range(image_shape[dim])
        else:  # Range is specified
            if type(r) is int:
                ranges[dim] = range(r, r + 1)
            elif type(r) is not tuple:
                raise TypeError("Range is not provided as a tuple.")
            else:  # range is a tuple
                if len(r) == 1:
                    ranges[dim] = range(r[0])
                elif len(r) == 2:
                    ranges[dim] = range(r[0], r[1])
                elif len(r) == 3:
                    ranges[dim] = range(r[0], r[1], r[2])
                else:
                    raise IndexError("Range values must contain 1 to three values")
            if not 1 <= ranges[dim].stop <= image_shape[dim]:
                raise IndexError("Specified range is outside of the image dimensions")

    output_shape = (
        len(ranges[0]),
        len(ranges[1]),
        len(ranges[2]),
        len(ranges[3]),
        len(ranges[4]),
    )
    nr_planes = output_shape[0] * output_shape[1] * output_shape[2]
    zct_list = list(product(ranges[0], ranges[1], ranges[2]))

    pixels = image.getPrimaryPixels()
    pixels_type = pixels.getPixelsType()
    if pixels_type.value == "float":
        data_type = pixels_type.value + str(
            pixels_type.bitSize
        )  # TODO: Verify this is working for all data types
    else:
        data_type = pixels_type.value

    # intensities = np.zeros(output_shape, dtype=data_type)

    intensities = np.zeros(
        (nr_planes, output_shape[3], output_shape[4]), dtype=data_type
    )
    if whole_planes:
        np.stack(list(pixels.getPlanes(zctList=zct_list)), out=intensities)
    else:
        tile_region = (ranges[3].start, ranges[4].start, len(ranges[3]), len(ranges[4]))
        zct_tile_list = [(z, c, t, tile_region) for z, c, t in zct_list]
        np.stack(list(pixels.getTiles(zctTileList=zct_tile_list)), out=intensities)

    intensities = np.reshape(intensities, newshape=output_shape)

    return intensities


def create_image_from_ndarray(
    connection: gw.BlitzGateway, data, image_name, image_description, dataset=None
):
    """
    Creates a new image in OMERO from a n dimensional numpy array.
    :param connection: The connection object to OMERO
    :param data: the ndarray. Must be a 5D array with dimensions in the order zctxy
    :param image_name:
    :param image_description:
    :param dataset:
    :return:
    """
    zct_list = list(
        product(range(data.shape[0]), range(data.shape[1]), range(data.shape[2]))
    )
    zct_generator = (data[z, c, t, :, :] for z, c, t in zct_list)

    connection.createImageFromNumpySeq(
        zctPlanes=zct_generator,
        imageName=image_name,
        sizeZ=1,
        sizeC=1,
        sizeT=1,
        description=None,
        dataset=None,
        sourceImageId=None,
        channelList=None,
    )


############### Creating projects and datasets #####################


def create_project(connection, name, description=None):
    new_project = gw.ProjectWrapper(connection, model.ProjectI())
    new_project.setName(name)
    if description:
        new_project.setDescription(description)
    new_project.save()

    return new_project


def create_dataset(
    connection: gw.BlitzGateway, name, description=None, parent_project=None
):
    new_dataset = gw.DatasetWrapper(connection, model.DatasetI())
    new_dataset.setName(name)
    if description:
        new_dataset.setDescription(description)
    new_dataset.save()
    if parent_project:
        link = model.ProjectDatasetLinkI()
        link.setParent(
            model.ProjectI(parent_project.getId(), False)
        )  # linking to a loaded project might raise exception
        link.setChild(model.DatasetI(new_dataset.getId(), False))
        connection.getUpdateService().saveObject(link)

    return new_dataset


############### Deleting projects and datasets #####################


def _delete_object(
    conn, object_type, objects, delete_annotations, delete_children, wait, callback=None
):
    if not isinstance(objects, list) and not isinstance(object, int):
        obj_ids = [objects.getId()]
    elif not isinstance(objects, list):
        obj_ids = [objects]
    elif isinstance(objects[0], int):
        obj_ids = objects
    else:
        obj_ids = [o.getId() for o in objects]

    try:
        conn.deleteObjects(
            object_type,
            obj_ids=obj_ids,
            deleteAnns=delete_annotations,
            deleteChildren=delete_children,
            wait=wait,
        )
        return True
    except Exception as e:
        print(e)
        return False


def delete_project(conn, projects, delete_annotations=False, delete_children=False):
    _delete_object(
        conn=conn,
        object_type="Project",
        objects=projects,
        delete_annotations=delete_annotations,
        delete_children=delete_children,
        wait=False,
    )


# Retrieve callback and wait until delete completes

# # This is not necessary for the Delete to complete. Can be used
# # if you want to know when delete is finished or if there were any errors
# handle = conn.deleteObjects("Project", [project_id])
# cb = omero.callbacks.CmdCallbackI(conn.c, handle)
# print "Deleting, please wait."
# while not cb.block(500):
#     print "."
# err = isinstance(cb.getResponse(), omero.cmd.ERR)
# print "Error?", err
# if err:
#     print cb.getResponse()
# cb.close(True)      # close handle too


############### Getting information on projects and datasets ###############


def get_all_projects(conn, opts=None):
    if opts is None:
        opts = {"order_by": "loser(obj.name)"}
    projects = conn.getObjects("Project", opts=opts)

    return projects


def get_project_datasets(project):
    datasets = project.listChildren()

    return datasets


def get_dataset_images(dataset):
    images = dataset.listChildren()

    return images


def get_orphan_datasets(conn):
    datasets = conn.getObjects("Dataset", opts={"orphaned": True})

    return datasets


def get_orphan_images(conn):
    images = conn.getObjects("Image", opts={"orphaned": True})

    return images


def get_tagged_images_in_dataset(dataset, tag_id):
    images = list()
    for image in dataset.listChildren():
        for ann in image.listAnnotations():
            if type(ann) == gw.TagAnnotationWrapper:
                if ann.getId() == tag_id:
                    images.append(image)
    return images


# In this section we give some convenience functions to send data back to OMERO #
def create_annotation_comment(connection, comment_string, namespace=None):
    if namespace is None:
        namespace = (
            metadata.NSCLIENTMAPANNOTATION
        )  # This makes the annotation editable in the client
    comment_ann = gw.CommentAnnotationWrapper(connection)
    comment_ann.setValue(comment_string)
    comment_ann.setNs(namespace)
    comment_ann.save()

    return comment_ann


def link_annotation_tag(connection, omero_obj, tag_id):
    tag = connection.getObject("Annotation", tag_id)
    link_annotation(omero_obj, tag)


def create_annotation_tag(connection, tag_string, description=None):
    tag_ann = gw.TagAnnotationWrapper(connection)
    tag_ann.setValue(tag_string)
    if description is not None:
        tag_ann.setDescription(description)
    tag_ann.save()

    return tag_ann


def _serialize_map_value(value):
    if isinstance(value, str):
        return value
    else:
        try:
            return dumps(value)
        except ValueError as e:
            # TODO: log an error
            return dumps(value.__str__())


def _dict_to_map(dictionary):
    """Converts a dictionary into a list of key:value pairs to be fed as map annotation.
    If value is not a string we serialize it as a json string"""
    map_annotation = [[k, _serialize_map_value(v)] for k, v in dictionary.items()]
    return map_annotation


def create_annotation_map(
    connection,
    annotation,
    annotation_name=None,
    annotation_description=None,
    namespace=None,
):
    """Creates a map_annotation for OMERO. It can create a map annotation from a
    dictionary or from a list of 2 elements list.
    """
    if namespace is None:
        namespace = (
            metadata.NSCLIENTMAPANNOTATION
        )  # This makes the annotation editable in the client
    # Convert a dictionary into a map annotation
    if isinstance(annotation, dict):
        annotation = _dict_to_map(annotation)
    elif isinstance(annotation, list):
        pass  # TODO: assert that the list is compliant with the OMERO format
    else:
        raise Exception(f"Could not convert {annotation} to a map_annotation")

    map_ann = gw.MapAnnotationWrapper(connection)
    if annotation_name is not None:
        map_ann.setName(annotation_name)
    if annotation_description is not None:
        map_ann.setDescription(annotation_description)

    map_ann.setNs(namespace)

    map_ann.setValue(annotation)
    map_ann.save()

    return map_ann


def create_annotation_file_local(
    connection, file_path, namespace=None, description=None
):
    """Creates a file annotation and uploads it to OMERO"""

    file_ann = connection.createFileAnnfromLocalFile(
        localPath=file_path, mimetype=None, namespace=namespace, desc=description
    )
    return file_ann


def _create_column(data_type, kwargs):
    column_class = COLUMN_TYPES[data_type]

    return column_class(**kwargs)


def _create_table(column_names, columns_descriptions, values, types=None):
    # validate lengths
    if not len(column_names) == len(columns_descriptions) == len(values):
        raise IndexError(
            "Error creating table. Names, description and values not matching or empty."
        )
    if types is not None and len(types) != len(values):
        raise IndexError(
            "Error creating table. Types and values lengths are not matching."
        )
    # TODO: Verify implementation of empty table creation

    columns = list()
    for i, (cn, cd, v) in enumerate(zip(column_names, columns_descriptions, values)):
        # Verify column names and descriptions are strings
        if not type(cn) == type(cd) == str:
            raise TypeError(
                f"Types of column name ({type(cn)}) or description ({type(cd)}) is not string"
            )

        if types is not None:
            v_type = types[i]
        else:
            if isinstance(v[0], (list, tuple)):
                v_type = [type(v[0][0])]
            else:
                v_type = type(v[0])

        # Verify that all elements in values are the same type
        # if not all(isinstance(x, v_type) for x in v):
        #     raise TypeError(f'Not all elements in column {cn} are of the same type')

        if v_type == str:
            size = (
                len(max(v, key=len)) * 2
            )  # We assume here that the max size is double of what we really have...
            args = {"name": cn, "description": cd, "size": size, "values": v}
            columns.append(_create_column(data_type="string", kwargs=args))
        elif v_type == int:
            args = {"name": cn, "description": cd, "values": v}
            columns.append(_create_column(data_type="long", kwargs=args))
        elif v_type == float:
            args = {"name": cn, "description": cd, "values": v}
            columns.append(_create_column(data_type="double", kwargs=args))
        elif v_type == bool:
            args = {"name": cn, "description": cd, "values": v}
            columns.append(_create_column(data_type="string", kwargs=args))
        elif v_type == gw.ImageWrapper or v_type == model.ImageI:
            args = {"name": cn, "description": cd, "values": [img.getId() for img in v]}
            columns.append(_create_column(data_type="image", kwargs=args))
        elif v_type == gw.RoiWrapper or v_type == model.RoiI:
            args = {"name": cn, "description": cd, "values": [roi.getId() for roi in v]}
            columns.append(_create_column(data_type="roi", kwargs=args))
        elif isinstance(v_type, (list, tuple)):  # We are creating array columns

            # Verify that every element in the 'array' is the same length and type
            if not all(len(x) == len(v[0]) for x in v):
                raise IndexError(
                    f"Not all elements in column {cn} have the same length"
                )
            if not all(all(isinstance(x, type(v[0][0])) for x in a) for a in v):
                raise TypeError(
                    f"Not all the elements in the array column {cn} are of the same type"
                )

            args = {"name": cn, "description": cd, "size": len(v[0]), "values": v}
            if v_type[0] == int:
                columns.append(_create_column(data_type="long_array", kwargs=args))
            elif v_type[0] == float:  # We are casting all floats to doubles
                columns.append(_create_column(data_type="double_array", kwargs=args))
            else:
                raise TypeError(
                    f"Error on column {cn}. Datatype not implemented for array columns"
                )
        else:
            raise TypeError(f"Could not detect column datatype for column {cn}")

    return columns


def create_annotation_table(
    connection,
    table_name,
    column_names,
    column_descriptions,
    values,
    namespace=None,
    table_description=None,
):
    """Creates a table annotation from a list of lists"""

    table_name = (
        f'{table_name}_{"".join([choice(ascii_letters) for n in range(32)])}.h5'
    )

    columns = _create_table(
        column_names=column_names,
        columns_descriptions=column_descriptions,
        values=values,
    )
    resources = connection.c.sf.sharedResources()
    repository_id = resources.repositories().descriptions[0].getId().getValue()
    table = resources.newTable(repository_id, table_name)
    table.initialize(columns)
    table.addData(columns)

    original_file = table.getOriginalFile()
    table.close()  # when we are done, close.
    file_ann = gw.FileAnnotationWrapper(connection)
    file_ann.setNs(namespace)
    file_ann.setDescription(table_description)
    file_ann.setFile(
        model.OriginalFileI(original_file.id.val, False)
    )  # TODO: try to get this with a wrapper
    file_ann.save()
    return file_ann


def create_roi(connection, image, shapes, name=None, description=None):
    """A pass through to link a roi to an image"""
    return _create_roi(connection, image, shapes, name, description)


def _create_roi(connection, image, shapes, name, description):
    # create an ROI, link it to Image
    # roi = gw.RoiWrapper()
    roi = model.RoiI()  # TODO: work with wrappers
    # use the omero.model.ImageI that underlies the 'image' wrapper
    roi.setImage(image._obj)
    if name is not None:
        roi.setName(rtypes.rstring(name))
    if description is not None:
        roi.setDescription(rtypes.rstring(name))
    for shape in shapes:
        roi.addShape(shape)
    # Save the ROI (saves any linked shapes too)
    return connection.getUpdateService().saveAndReturnObject(roi)


def _rgba_to_int(red, green, blue, alpha=255):
    """ Return the color as an Integer in RGBA encoding """
    r = red << 24
    g = green << 16
    b = blue << 8
    a = alpha
    rgba_int = sum([r, g, b, a])
    if rgba_int > (2 ** 31 - 1):  # convert to signed 32-bit int
        rgba_int = rgba_int - 2 ** 32

    return rgba_int


def _set_shape_properties(
    shape,
    name=None,
    fill_color=(10, 10, 10, 10),
    stroke_color=(255, 255, 255, 255),
    stroke_width=1,
):
    if name:
        shape.setTextValue(rtypes.rstring(name))
    shape.setFillColor(rtypes.rint(_rgba_to_int(*fill_color)))
    shape.setStrokeColor(rtypes.rint(_rgba_to_int(*stroke_color)))
    shape.setStrokeWidth(LengthI(stroke_width, enums.UnitsLength.PIXEL))


def create_shape_point(
    x_pos,
    y_pos,
    z_pos=None,
    c_pos=None,
    t_pos=None,
    name=None,
    stroke_color=(255, 255, 255, 255),
    fill_color=(10, 10, 10, 20),
    stroke_width=1,
):
    point = model.PointI()
    point.x = rtypes.rdouble(x_pos)
    point.y = rtypes.rdouble(y_pos)
    if z_pos is not None:
        point.theZ = rtypes.rint(z_pos)
    if c_pos is not None:
        point.theC = rtypes.rint(c_pos)
    if t_pos is not None:
        point.theT = rtypes.rint(t_pos)
    _set_shape_properties(
        shape=point,
        name=name,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        fill_color=fill_color,
    )

    return point


def create_shape_line(
    x1_pos,
    y1_pos,
    x2_pos,
    y2_pos,
    c_pos=None,
    z_pos=None,
    t_pos=None,
    name=None,
    stroke_color=(255, 255, 255, 255),
    stroke_width=1,
):
    line = model.LineI()
    line.x1 = rtypes.rdouble(x1_pos)
    line.x2 = rtypes.rdouble(x2_pos)
    line.y1 = rtypes.rdouble(y1_pos)
    line.y2 = rtypes.rdouble(y2_pos)
    line.theZ = rtypes.rint(z_pos)
    line.theT = rtypes.rint(t_pos)
    if c_pos is not None:
        line.theC = rtypes.rint(c_pos)
    _set_shape_properties(
        line, name=name, stroke_color=stroke_color, stroke_width=stroke_width
    )
    return line


def create_shape_rectangle(
    x_pos,
    y_pos,
    width,
    height,
    z_pos,
    t_pos,
    rectangle_name=None,
    fill_color=(10, 10, 10, 255),
    stroke_color=(255, 255, 255, 255),
    stroke_width=1,
):
    rect = model.RectangleI()
    rect.x = rtypes.rdouble(x_pos)
    rect.y = rtypes.rdouble(y_pos)
    rect.width = rtypes.rdouble(width)
    rect.height = rtypes.rdouble(height)
    rect.theZ = rtypes.rint(z_pos)
    rect.theT = rtypes.rint(t_pos)
    _set_shape_properties(
        shape=rect,
        name=rectangle_name,
        fill_color=fill_color,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
    )
    return rect


def create_shape_ellipse(
    x_pos,
    y_pos,
    x_radius,
    y_radius,
    z_pos,
    t_pos,
    ellipse_name=None,
    fill_color=(10, 10, 10, 255),
    stroke_color=(255, 255, 255, 255),
    stroke_width=1,
):
    ellipse = model.EllipseI()
    ellipse.setX(rtypes.rdouble(x_pos))
    ellipse.setY(rtypes.rdouble(y_pos))  # TODO: setters and getters everywhere
    ellipse.radiusX = rtypes.rdouble(x_radius)
    ellipse.radiusY = rtypes.rdouble(y_radius)
    ellipse.theZ = rtypes.rint(z_pos)
    ellipse.theT = rtypes.rint(t_pos)
    _set_shape_properties(
        ellipse,
        name=ellipse_name,
        fill_color=fill_color,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
    )
    return ellipse


def create_shape_polygon(
    points_list,
    z_pos,
    t_pos,
    polygon_name=None,
    fill_color=(10, 10, 10, 255),
    stroke_color=(255, 255, 255, 255),
    stroke_width=1,
):
    polygon = model.PolygonI()
    points_str = "".join(
        ["".join([str(x), ",", str(y), ", "]) for x, y in points_list]
    )[:-2]
    polygon.points = rtypes.rstring(points_str)
    polygon.theZ = rtypes.rint(z_pos)
    polygon.theT = rtypes.rint(t_pos)
    _set_shape_properties(
        polygon,
        name=polygon_name,
        fill_color=fill_color,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
    )
    return polygon


def create_shape_mask(
    mask_array, x_pos, y_pos, z_pos, t_pos, mask_name=None, fill_color=(10, 10, 10, 255)
):
    mask = model.MaskI()
    mask.setX(rtypes.rdouble(x_pos))
    mask.setY(rtypes.rdouble(y_pos))
    mask.setTheZ(rtypes.rint(z_pos))
    mask.setTheT(rtypes.rint(t_pos))
    mask.setWidth(rtypes.rdouble(mask_array.shape[0]))
    mask.setHeight(rtypes.rdouble(mask_array.shape[1]))
    mask.setFillColor(rtypes.rint(_rgba_to_int(*fill_color)))
    if mask_name:
        mask.setTextValue(rtypes.rstring(mask_name))
    mask_packed = np.packbits(mask_array)  # TODO: raise error when not boolean array
    mask.setBytes(mask_packed.tobytes())

    return mask


def link_annotation(object_wrapper, annotation_wrapper):
    object_wrapper.linkAnnotation(annotation_wrapper)
