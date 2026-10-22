"""
rays.py
A module that holds information about the rays in our system.
George Peters - 15/05/25

Updated: 22/10/26
"""
import numpy as np
import matplotlib.pyplot as plt
from xuv_raytracer.raytracer.elements import OpticalElement
from xuv_raytracer.raytracer.genpolar import rtrings

class Ray:
    """
    A ray that can propagate through space and refract through materials.

    Attributes:
        _pos (NumPy array) : 3-D Vector holding the position of the ray
        _direc (NumPy array) : 3-D Vector holding the direction of the ray
        _vertices (list) : An array that holds all of the vertices along the ray's path
        _wavelength(float) : The wavelength of the ray in nanometres (default: 800 nm - the Ti:Sapphire driving laser)
    """

    def __init__(self, pos=[0.0,0.0,0.0], direc=[0.0,0.0,1.0], wavelength=800.0):
        """
        Initialise the object

        Args:
            pos (list) : 3-D Vector holding the position of the ray.
            direc (list) : 3-D Vector holding the direction of the ray.
            wavelength (float) : The wavelength of the ray in nanometres
        """
        # These 4 if statements ensure that the inputs are lists or arrays and ensures length 3
        if type(pos) is not list and not isinstance(pos, np.ndarray):
            raise Exception("Position must be a list")
        
        if len(pos) != 3:
            raise Exception("Position must be of length 3")
        
        if type(direc) is not list and not isinstance(direc, np.ndarray):
            raise Exception("Direction must be a list")
        
        if len(direc) != 3:
            raise Exception("Direction must be of length 3")

        if not isinstance(wavelength, (int, float)):
            raise Exception("Wavelength must be a number")

        if not wavelength > 0:
            raise Exception("Wavelength must be a positive value")

        self._pos = np.array(pos, dtype=float)
        self._direc = np.array(direc, dtype=float)
        self._direc = self._direc / np.linalg.norm(self._direc)
        self._vertices = [self._pos]
        self._wavelength = wavelength

    def pos(self):
        """
        Access method for the position of the ray

        Returns:
            self._pos (Numpy array) : 3D Vector holding the position of the ray
        """
        return self._pos

    def direc(self):
        """
        Access method for the direction of the ray

        Returns:
            self._direc (Numpy array) : 3D Vector holding the direction of the ray
        """
        return self._direc
    
    def vertices(self):
        """
        Access method for the list of verticies of the ray

        Returns:
            self._vertices (list) : List holding all of the points on the ray's path.
        """
        return self._vertices

    def wavelength(self):
        """
        Access method for the wavelength of the ray

        Returns:
            self._wavelength (float) : The wavelength of the ray in nanometres
        """
        return self._wavelength
    

    def append(self, pos, direc):
        """
        Appends a new point and direction to the ray.

        Args:
            pos (list) : 3D Vector holding the position of the ray
            direc (list) : 3D Vector holding the direction of the ray

        Return:
            self (Ray) : Returns self for chaining.
        """
        # Ensures input is either a list or numpy array, and ensures its length 3
        if type(pos) is not list and not isinstance(pos, np.ndarray):
            raise Exception("Position must be a list")
        
        if len(pos) != 3:
            raise Exception("Position must be of length 3")
        
        if type(direc) is not list and not isinstance(direc, np.ndarray):
            raise Exception("Direction must be a list")
        
        if len(direc) != 3:
            raise Exception("Direction must be of length 3")
        
        pos = np.array(pos)
        direc = np.array(direc)
        direc = direc / np.linalg.norm(direc)
        self._vertices.append(pos)
        self._pos = pos
        self._direc = direc
        
        return self


class RayBundle():
    """
    The RayBundle class will hold a bundle of rays that are in concentric circles, and transverse to the optical axis.

    Attributes:
        _rays (Ray list) : The list of rays inside the ray bundle.
    """
    def __init__(self, rmax=5., nrings=5, multi=6, wavelength=800.0):
        """
        The init method that runs when a RayBundle is instantiated.

        Args:
            rmax (float) : The maximum radius (from the optical axis) of the bundle.
            nrings (int) : The number of rings in the circle.
            multi (int) : A multiplier to determine the number of rays per ring (N_points = N_ring * multi).
            wavelength (float) : The wavelength of the rays in nanometres
        """
        self._rays = []

        points = rtrings(rmax, nrings, multi)

        # Instantiate the list of given points
        for p in points:
            ray = Ray(pos=[p[0], p[1], 0], wavelength=wavelength)
            self._rays.append(ray)

    def propagate_bundle(self, elements):
        """
        Method that will propagate each ray in the bundle through each of the elements provided.

        Args:
            elements (OpticalElement list) : A list of all the optical elements in the system.
        """
        for element in elements:
            for ray in self._rays:
                if isinstance(element, OpticalElement) and isinstance(ray, Ray):
                    element.propagate_ray(ray)

    def track_plot(self, elements=None):
        """
        Method that will plot the tracks of each ray.

        Args:
            elements (OpticalElement list) : List of rays that should be drawn on the graph
                - Note this does not include surfaces - only lenses.

        Returns:
            Figure (Figure) : The figure that has the ray paths plotted on it.
        """
        Figure, ax = plt.subplots(tight_layout=True)
        ax.set_title("Ray Bundle Paths")
        ax.set_xlabel("Position (Z)")
        ax.set_ylabel("Position (Y)")
        ax.grid(True)

        for ray in self._rays:
            if isinstance(ray, Ray):
                points = np.array( ray.vertices() )
                ax.plot(points[:, 2], points[:, 1], color=(255/255,105/255,97/255))

        if elements is None:
            return Figure

        # Turn the parameter 'elements' into a list if only one element is passed in
        if not isinstance(elements, list):
            elements = [elements]

        # Draw the elements onto the plot
        for element in elements:
            patches = element.draw()

            if not isinstance(patches, list):
                patches = [patches]
                
            for patch in patches: ax.add_patch(patch)

        return Figure

    def rms(self):
        """
        Method that will return the rms size of the bundle.

        Returns:
            rms (float) : The rms size (spread from optical axis) of the bundle.
        """

        distances = []

        # Calculate how far the ray intercepts the output plane from the optical point
        for ray in self._rays:
            if isinstance(ray, Ray):
                ray_x = ray.pos()[0]
                ray_y = ray.pos()[1]
                distance = np.sqrt( ray_x * ray_x + ray_y * ray_y )
                distances.append(distance)

        rms = np.sqrt( np.mean( np.square(distances) ) )
        return rms

    def direction_rms(self):
        """
        Function that returns an rms value which is a measure of how non-parallel the rays are

        Returns
            direction_rms (float) : The rms value of the directions from parallel
        """

        residuals_sq = []

        for ray in self._rays:
            if isinstance(ray, Ray):
                # Finds |direc - (0,0,1)|^2 to measure how far the direction is from parallel.
                residual = ray.direc() - np.array([0,0,1])
                residuals_sq.append( np.linalg.norm(residual)**2 )
        
        direction_rms = np.sqrt( np.mean( residuals_sq ) )
        return direction_rms

    def spot_plot(self, spot_plot = None):
        """
        Method that will plot a spot plot diagram of the bundle (showing x,y values at certain z value)

        Return:
            spot_plot (Figure) : The spot plot diagram.
        """

        if spot_plot is None:
            spot_plot, ax = plt.subplots(tight_layout=True)
            colour = (255/255,105/255,97/255)
        else:
            ax = spot_plot.gca()
            colour = (119/255, 221/255, 119/255)

        ax.set_title("Spot Plot")
        ax.set_xlabel("Position (X)")
        ax.set_ylabel("Position (Y)")
        ax.grid(True)

        for ray in self._rays:
            if isinstance(ray, Ray):
                x = ray.pos()[0]
                y = ray.pos()[1]

                ax.plot(x, y, marker="o", color=colour)

        return spot_plot

