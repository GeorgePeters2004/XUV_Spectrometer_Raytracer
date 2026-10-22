import numpy as np
import raytracer.physics as ph

HENE_WAVELENGTH = 632.8
TI_SAPPHIRE_WAVELENGTH = 800

def find_beta(wavelength, m, grooves, alpha):
    """
    Calculates the diffraction angle using the 1D diffraction equation
    sin(alpha) + sin(beta) = m * wavelength * grooves

    Params:
        wavelength (float): the wavelength of the driving laser, in nanometres
        m (int): the diffraction order
        grooves (int): the number of grooves per mm of the grating
        alpha (float): the incident angle
    """

    # Must ensure units are correct - convert all distances to mm
    wavelength = wavelength / 1000000

    # Use the diffraction equation to calculate beta
    sin_beta = m * wavelength * grooves - np.sin(alpha)

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
alpha = np.radians(87) # degrees

# Calculate all orders of HeNe diffraction
HeNe_orders_scalar = []
HeNe_Orders_vector = []
max_order = False

normal = np.array([0, 0, 1])
groove_direc = np.array([0, 1, 0])
direc = np.array([0.3, 0.9, -np.sqrt(0.1)])

while not max_order:
    beta_scalar = find_beta(wavelength, m, grooves, alpha)
    direc_out = ph.diffract(direc, normal, groove_direc, m, wavelength, grooves)

    if beta_scalar is not None:
        # Find beta from the output direction and the normal
        output_g = np.dot(direc_out, groove_direc)
        output_n = np.dot(direc_out, normal)
        beta_vector = np.arctan2(-output_g, output_n)
        beta_vector = np.degrees(beta_vector)

        HeNe_orders_scalar.append(beta_scalar)
        HeNe_Orders_vector.append(beta_vector)
        m = m + 1
    else:   
        max_order = True


print("Order \t|\t find_beta \t|\t diffract")
for i in range(0, len(HeNe_orders_scalar)):
    print(i, "\t|\t", HeNe_orders_scalar[i], "\t|\t", HeNe_Orders_vector[i])