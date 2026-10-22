"""
Program to find the coordinates of dots that form concentric rings.
G Peters 13/05/2025
"""

import numpy as np

def rtrings(rmax, nrings, multi):
    """Creates concentric rings

    This function produces coordinates of points in concentric rings.

    Args:
        rmax (float) : Maximum radius of the concentric rings.
        nrings (int) : Total number of rings.
        multi (int) : Multiplier to determine number of points per ring.

    Yields:
        x, y (float tuple) : The x and y coordinates of each point to be plotted.
    """
    yield(0,0)
    for i in range(1, nrings + 1 ):
        r = i * rmax / (nrings)

        # Calculates the number of points in a ring
        N_points = i * multi

        # Calculates equally spaced points around the ring
        for j in range(0, N_points):
            theta = (2*np.pi/N_points) * j
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            yield(x, y)