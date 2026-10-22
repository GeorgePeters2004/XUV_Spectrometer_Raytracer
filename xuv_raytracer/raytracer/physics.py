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
    i_para = dot_1 * normal
    i_perp = direc - i_para

    ratio = n_1/n_2
    cos_squared_2 = 1 - ratio*ratio*(1 - dot_1 * dot_1) # Equal to cos(theta_2)^2

    # TIR if k < 0
    if cos_squared_2 < 0:
        return None
    
    # t = transmitted ray, perpendicular/parallel to normal
    t_para = -1 * np.sqrt(cos_squared_2) * normal
    t_perp = ratio * i_perp
    direc_t = t_perp + t_para
    direc_t /= np.linalg.norm(direc_t)

    return direc_t


def reflect(direc, normal):
    """
    Function that will find the direction of a ray after reflecting off a surface.

    Args:
        direc (Numpy array) : 3D normalised vector of the direction the ray is travelling in
        normal (Numpy array) : 3D normalised vector of the direction of the normal to the surface.

    Returns:
        reflected_direc (Numpy array) : 3D normalised vector of the new direction after the reflection.
    """
    direc = np.array(direc)
    normal = np.array(normal)

    direc_para = np.dot(direc, normal) * normal
    reflected_direc = direc - 2 * direc_para

    return reflected_direc

def diffract(direc, normal, groove_direc, m, wavelength, grooves):
    """
    Function that will find the direction of a ray after diffracting off a grating.

    Args:
        direc (Numpy array) : 3D normalised vector of the direction the ray is travelling in
        normal (Numpy array) : 3D normalised vector of the direction of the normal to the surface.
        groove_direc (Numpy array) : 3D normalised vector of the perpendicular direction to the grating's grooves.
        m (int) : The order of the diffraction that we are interested in.
        wavelength (float) : The wavelength of the light being diffracted in nanometres.
        grooves (int) : The number of grooves per millimetre at the interception point.
    
    Returns:
        direc_out (Numpy array) : 3D normalised vector of the new direction after the diffraction.
    """

    # Convert units
    wavelength = wavelength / 1000000

    direc = np.array(direc)
    normal = np.array(normal)
    groove_direc = np.array(groove_direc)

    direc_perp = direc - np.dot(direc, normal) * normal
    direc_perp_prime = direc_perp - m * wavelength * grooves * groove_direc

    # Equivalent to |sin(beta)| > 1 in scalar version.
    if np.linalg.norm(direc_perp_prime)**2 > 1:
        return None
 
    # Calculate direc_para_prime
    # Size of vector must be sqrt(1 - |direc_perp_prime|^2) and direction away from the surface
    if np.dot(direc, normal) < 0:
        direc_para_prime = np.sqrt(1 - np.linalg.norm(direc_perp_prime)**2) * normal
    else:
        direc_para_prime = np.sqrt(1 - np.linalg.norm(direc_perp_prime)**2) * (-1 * normal)

    # Combine together
    direc_out = direc_perp_prime + direc_para_prime
    direc_out /= np.linalg.norm(direc_out)

    return direc_out