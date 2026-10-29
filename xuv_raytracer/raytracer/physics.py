"""
physics.py
Module to hold all physics concepts
George Peters - 15/05/25
"""
import numpy as np

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


def photon_energy(wavelength):
    """
    Calculates the photon energy with a given wavelength

    Args:
        wavelength (float) : The wavelength of the photon (nm)
    
    Returns:
        energy (float) : The energy of the photon (eV)
    """
    energy = 1239.84 / wavelength # hc = 1239.84 eV nm
    return energy
