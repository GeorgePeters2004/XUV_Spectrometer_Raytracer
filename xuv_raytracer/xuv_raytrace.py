import numpy as np

HENE_WAVELENGTH = 632.8
TI_SAPPHIRE_WAVELENGTH = 800

def find_beta(wavelength, m, grooves, alpha):
    """
    Calculates the diffraction angle using the diffraction equation
    sin(alpha) + sin(beta) = m * wavelength * grooves

    Params:
        wavelength (float): the wavelength of the driving laser, in nanometres
        m (int): the diffraction order
        grooves (int): the number of grooves per mm of the grating
        alpha (float): the incident angle
    """

    # Must ensure units are correct - convert all distances to mm
    alpha_rads = np.radians(alpha)
    wavelength = wavelength / 1000000

    # Use the diffraction equation to calculate beta
    sin_beta = m * wavelength * grooves - np.sin(alpha_rads)

    # Check if the order exists
    if np.abs(sin_beta) > 1:
        return None

    beta_rads = np.arcsin(sin_beta)

    # Convert back to degrees
    beta = np.degrees(beta_rads)
    return beta

wavelength = HENE_WAVELENGTH # nm
grooves = 120 # lines / mm
m = 0 # diffraction order
alpha = 87 # degrees

# Calculate all orders of HeNe diffraction
HeNe_Orders = []
max_order = False

while not max_order:
    beta = find_beta(wavelength, m, grooves, alpha)
    if beta != None:
        HeNe_Orders.append(beta)
        m = m + 1
    else:
        max_order = True

print(HeNe_Orders)