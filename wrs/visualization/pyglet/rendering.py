"""
rendering.py
--------------

Functions to convert trimesh objects to pyglet/opengl objects.
"""

import numpy as np


def vector_to_gl(array, *args):
    """
    Convert an array and an optional set of args into a
    flat vector of gl.GLfloat
    """
    from pyglet import gl

    array = np.array(array)
    if len(args) > 0:
        array = np.append(array, args)
    vector = (gl.GLfloat * len(array))(*array)
    return vector


def light_to_gl(light, transform, lightN):
    """
    Convert trimesh.scene.lighting.Light objects into
    args for gl.glLightFv calls

    Parameters
    --------------
    light : trimesh.scene.lighting.Light
      Light object to be converted to GL
    transform : (4, 4) float
      Transformation matrix of light
    lightN : int
      Result of gl.GL_LIGHT0, gl.GL_LIGHT1, etc

    Returns
    --------------
    multiarg : [tuple]
      List of args to pass to gl.glLightFv eg:
      [gl.glLightfb(*a) for a in multiarg]
    """
    from pyglet import gl

    # convert color to opengl
    gl_color = vector_to_gl(light.color.astype(np.float64) / 255.0)
    assert len(gl_color) == 4

    # cartesian translation from matrix
    gl_position = vector_to_gl(transform[:3, 3])

    # create the different position and color arguments
    args = [
        (lightN, gl.GL_POSITION, gl_position),
        (lightN, gl.GL_SPECULAR, gl_color),
        (lightN, gl.GL_DIFFUSE, gl_color),
        (lightN, gl.GL_AMBIENT, gl_color),
    ]
    return args
