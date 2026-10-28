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
    A ray that can propagate through space and intereact with materials.

    Attributes:
        _pos (NumPy array) : 3-D Vector holding the position of the ray
        _direc (NumPy array) : 3-D Vector holding the direction of the ray
        _vertices (list) : An array that holds all of the vertices along the ray's path
        _wavelength(float) : The wavelength of the ray in nanometres (default: 800 nm - the Ti:Sapphire driving laser)
        _terminated (boolean) : A True/False flag to say if the ray should stop propagating.
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
        self._terminated = False

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

    def set_vertices(self, vertices):
        """
        Setter method for the list of vertices of the ray

        Args:
            vertices (list): An array that holds all of the vertices along the ray's path
        """
        self._vertices = vertices.copy()
        self._pos = self._vertices[-1] # Ensures the position matches the final vertex
        return

    def wavelength(self):
        """
        Access method for the wavelength of the ray

        Returns:
            self._wavelength (float) : The wavelength of the ray in nanometres
        """
        return self._wavelength

    def terminated(self):
        """
        Access method for the terminated flag of the ray

        Returns:
            self._terminated (boolean) : The terminated flag.
        """
        return self._terminated

    def set_terminated(self, terminated = True):
        """
        Sets the ray to terminated. It will now stop propagating

        Args:
            terminated (boolean) : The terminated flag, assumed to be true by default
        """
        self._terminated = terminated
        return
    

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

    def copy(self):
        """
        A method that returns a new Ray object, with the same attributes as this one.

        Returns:
            ray_copy (Ray) : An exact copy of this ray
        """
        ray_copy = Ray(pos=self.pos(), direc=self.direc(), wavelength=self.wavelength())
        ray_copy.set_vertices(self.vertices())
        ray_copy.set_terminated(self.terminated())
        return ray_copy


class RayBundle():
    """
    The RayBundle class will hold a bundle of rays that are in concentric circles, or a set list, and transverse to the optical axis.

    Attributes:
        _rays (Ray list) : The list of rays inside the ray bundle.
    """
    def __init__(self, rmax=None, nrings=None, multi=None, wavelength=None, rays=None):
        """
        The init method that runs when a RayBundle is instantiated.

        Args:
            rmax (float) : The maximum radius (from the optical axis) of the bundle.
            nrings (int) : The number of rings in the circle.
            multi (int) : A multiplier to determine the number of rays per ring (N_points = N_ring * multi).
            wavelength (float) : The wavelength of the rays in nanometres
            rays (Ray list) : An optional list of Ray objects that allows the rays in the bundle to be specified.
        """        

        # If no rays are specified, make a basic concentric ring of rays
        if rays is None:
            self._rays = []

            points = rtrings(rmax, nrings, multi)

            # Instantiate the list of given points
            for p in points:
                ray = Ray(pos=[p[0], p[1], 0], wavelength=wavelength)
                self._rays.append(ray)
        else: # If rays is specified, use them instead of making new ones
            self._rays = list(rays)

    def rays(self):
        """
        Accessor method for the bundle's list of rays

        Returns:
            self._rays (Ray list) : The bundle's list of rays
        """
        return self._rays

    def set_rays(self, rays):
        """
        Setter method for the bundle's rays

        Args:
            rays (Ray list) : The list of rays to replace the bundle
        """
        self._rays = list(rays)
        return

    def remove_lost_rays(self):
        """
        A function that removes any rays that either missed an element entirely, or hit outside the provided aperture

        Returns:
            n_lost (int) : The number of rays removed
        """
        kept = []

        for ray in self.rays():

            # If a ray is marked as terminanted, remove it
            if not ray.terminated():
                kept.append(ray)

        n_lost = len(self.rays()) - len(kept)
        self.set_rays(kept)
        return n_lost


    def propagate_bundle(self, elements):
        """
        Method that will propagate each ray in the bundle through each of the elements provided.

        Args:
            elements (OpticalElement list) : A list of all the optical elements in the system.
        """
        for element in elements:
            for ray in self._rays:
                if ray.terminated():
                    continue

                if isinstance(element, OpticalElement) and isinstance(ray, Ray):
                    element.propagate_ray(ray)

        return

    def track_plot(self, elements=None, Figure=None, colour=(255/255,105/255,97/255), label=None):
        """
        Method that will plot the tracks of each ray.

        Args:
            elements (OpticalElement list) : List of rays that should be drawn on the graph
                - Note this does not include surfaces - only lenses.
            Figure (Figure) : The figure that we want to draw on, rather than making a new one
            colour (tuple) : Three RGB values for the colour of the ray bundle
            label (String) : The label for this raybundle trace

        Returns:
            Figure (Figure) : The figure that has the ray paths plotted on it.
        """

        if Figure is None:
            Figure, ax = plt.subplots(tight_layout=True)

            ax.set_title("Ray Bundle Paths")
            ax.set_xlabel("Position (Z)")
            ax.set_ylabel("Position (X)")
            ax.grid(True)
            
        else:
            ax = Figure.gca()

        for (i, ray) in enumerate(self._rays):
            if isinstance(ray, Ray):
                points = np.array( ray.vertices() )

                if i == 0:
                    line_label = label
                else:
                    line_label = None
                
                ax.plot(points[:, 2], points[:, 0], color=colour, label=line_label)

        if label is not None:
            ax.legend()

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

    def best_focus(self, direction = None):
        """
        Function that calculates the best focus of a ray bundle.
        
        Args:
            direction (list) : Optional focusing direction
        
        Returns:
            focus (Numpy array) : The focus point of our beam with the lowest rms.
            t_best (float) : The distance from the end point where the focus lies
        """

        # Get the end points of the rays, and store in a list
        end_points = np.array([ray.vertices()[-1] for ray in self.rays()])

        # Get the directions of each ray
        directions = np.array([ray.direc() for ray in self.rays()])

        # Find mean of the directions (for the beam axis) and mean of the end points (for reference point)
        beam_axis = np.mean(directions, axis = 0)
        beam_axis /= np.linalg.norm(beam_axis)

        reference_point = np.mean(end_points, axis = 0)

        d_beam = np.dot(directions, beam_axis) # d_i dot b for every ray

        crossings = self.plane_crossings(reference_point, beam_axis)

        if direction is None:
            # Find two vectors perpendicular to the beam axis, using helper vector
            helper = [0, 1, 0] if abs(beam_axis[1]) < 0.9 else [1, 0, 0]

            u = np.cross(beam_axis, np.array(helper))
            u /= np.linalg.norm(u)
            v = np.cross(beam_axis, u)
            axes = [u, v]
        else:
            # Only use the given direction as u and ignore v
            u = np.array(direction, dtype=float)
            u = u - np.dot(u, beam_axis) * beam_axis # Leaves the component perpendicular to the beam axis
            u /= np.linalg.norm(u)

            axes = [u] 

        cov_sum = 0
        var_sum = 0
        for axis in axes:
            a = np.dot(crossings - reference_point, axis)
            b = np.dot(directions, axis) / d_beam
            
            cov_sum += np.cov(a, b, bias=True)[0, 1]
            var_sum += np.var(b)

        t_best = -cov_sum / var_sum

        # Find focus point: the centre of where the rays cross the plane at t_best
        focus = np.mean(self.plane_crossings(reference_point + t_best * beam_axis, beam_axis), axis = 0)

        return focus, t_best

    def plane_crossings(self, point, normal):
        """
        A function to determine where all the rays in a bundle cross a given plane.

        Args:
            point (Numpy array) : The 3D point defining the plane's position
            normal (Numpy array) : The 3D normalised vector defining the plane's orientation

        Returns:
            crossings (Numpy array) : One crossing per ray through the plane
        """

        # The end point of each ray
        ends = np.array([ray.pos() for ray in self.rays()])

        # The direction of each ray
        direcs = np.array([ray.direc() for ray in self.rays()])

        # Plane passes through point (P) with normal n
        # A point x is on the plane when (x - P) dot n = 0 (the vector from x to P lies along the plane)
        # Ray i is a straight line from its current end point: x = e_i + s * d_i
        # Substituting this x into the plane equation yields s_i = ((P - e_i) dot n) / (d_i dot n)
        s = np.dot(point - ends, normal) / np.dot(direcs, normal)

        # s is the distance from the end point to the crossing, so we can calculate the crossing point
        crossings = ends + s[:, None] * direcs

        return crossings
    