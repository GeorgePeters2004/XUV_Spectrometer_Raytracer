"""
physics.py
Module to hold all physics concepts
George Peters - 15/05/25
"""
import numpy as np

def refract(direc, normal, n_1, n_2):
    """
    Function to refract a ray through a boundary using Snell's law

    Args:
        direc (Numpy array) : 3D normalised vector with the initial direction of the ray.
        normal (Numpy array) : 3D normalised vector with the normal to the surface.
        n_1 (float) : refractive index of the initial material.
        n_2 (float) : refractive index of the new material.

    Returns:
        direc_t (Numpy array) : 3D vector with the final direction of the ray.
    """

    direc = np.array(direc)
    normal = np.array(normal)

    dot_1 = np.dot(direc,normal)

    # i = incident ray, perpendicular/parallel to normal
    i_perp = dot_1 * normal
    i_para = direc - i_perp

    ratio = n_1/n_2
    cos_squared_2 = 1 - ratio*ratio*(1 - dot_1 * dot_1) # Equal to cos(theta_2)^2

    # TIR if k < 0
    if cos_squared_2 < 0:
        return None
    
    # t = transmitted ray, perpendicular/parallel to normal
    t_perp = -1 * np.sqrt(cos_squared_2) * normal
    t_para = ratio * i_para
    direc_t = t_para + t_perp
    direc_t /= np.linalg.norm(direc_t)

    return direc_t


def reflect(direc, normal):
    """
    Function that will find the direction of a ray after reflecting off a surface.

    Args:
        direc (Numpy array) : 3D normalised vector of the direction the ray is travelling in
        normal (Numpy array) : 3D normalised vector of the direction of the normal to the surface.

    Returns:
        reflected_direc (Numpy array) : 3D normalised vector of the new direction after the reflection
    """

    reflected_direc = direc - 2 * np.dot(direc,normal) * normal

    return reflected_direc
