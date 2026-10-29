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
        _pos (NumPy array) : 3D Vector holding the position of the ray
        _direc (NumPy array) : 3D Vector holding the direction of the ray
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

        return

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
    def __init__(self, rmax=5., nrings=5, multi=6, wavelength=800, rays=None):
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
            # A collimated beam along z, starting at z = 0, made of concentric rings
            source = RayBundle.from_source(position=[0, 0, 0], direction=[0, 0, 1], half_angle=0,
                                           wavelength=wavelength, source_radius=rmax,
                                           source_rings=nrings, source_multi=multi)

            self._rays = source.rays()
        else: # If rays is specified, use them instead of making new ones
            self._rays = list(rays)

        return

    @classmethod
    def from_source(cls, position, direction, half_angle, wavelength, source_radius=0.0,
                    angle_rings=3, angle_multi=6, source_rings=2, source_multi=6):
        """
        A class method that allows a ray bundle to be created that has divering characteristics
        - half_angle = 0 implies collimated beam
        - source_radius = 0 implies point source

        The class has both angle_rings/multi and source_rings/multi
        - Every direction is launched from every source point
        - total rays (number of directions) x (number of source points)

        Args:
            position (list) : The position of the ray bundle
            direction (list) : The direction of the ray bundle
            half_angle (float) : The divergence of the bundle
            wavelength (float) : The wavelength
            source_radius (float) : The radius of the source (starting radius)
            angle_rings (int) : The number of rings of directions across the cone of light
            angle_multi (int) : Multiplier for number of directions per ring
            source_rings (int) : Number of rings of start points across the source disc
            source_multi (int) : Multiplier for the number of start points per ring
        """
        position = np.array(position, dtype=float)
        direction = np.array(direction, dtype=float)
        direction /= np.linalg.norm(direction)

        # Create two perpendicular axes across the beam, using a helper
        helper = np.array([0, 1, 0]) if abs(direction[1]) < 0.9 else np.array([1, 0, 0])
        axis_1 = helper - np.dot(helper, direction) * direction
        axis_1 /= np.linalg.norm(axis_1)
        axis_2 = np.cross(direction, axis_1)

        # Deals with point sources
        if source_radius > 0:
            offsets = list(rtrings(source_radius, source_rings, source_multi))
        else:
            offsets = [(0, 0)]

        # Deals with collimated beams
        if half_angle > 0:
            angles = list(rtrings(np.tan(half_angle), angle_rings, angle_multi))
        else:
            angles = [(0, 0)]

        # Build the rays for the bundle
        rays = []
        for angle in angles:
            direc = direction + angle[0] * axis_1 + angle[1] * axis_2

            for offset in offsets:
                pos = position + offset[0] * axis_1 + offset[1] * axis_2
                rays.append( Ray(pos=pos, direc=direc, wavelength=wavelength) )
                            
        return cls(rays=rays)


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

    def track_plot(self, elements=None, ax=None, colour=(255/255,105/255,97/255), label=None, max_length=None):
        """
        Method that will plot the tracks of each ray.

        Args:
            elements (OpticalElement list) : List of rays that should be drawn on the graph
                - Note this does not include surfaces - only lenses.
            ax (Axis) : The axis that we want to draw on, rather than making a new one
            colour (tuple) : Three RGB values for the colour of the ray bundle
            label (String) : The label for this raybundle trace
            max_length (float) : The maximum length of our plot, prevents zooming out to far-off rays

        Returns:
            ax (Axis) : The axis that has the ray paths plotted on it.
        """

        if ax is None:
            Figure, ax = plt.subplots(tight_layout=True)

            ax.set_title("Ray Bundle Paths")
            ax.set_xlabel("Position (Z)")
            ax.set_ylabel("Position (X)")
            ax.grid(True)

        for (i, ray) in enumerate(self._rays):
            if isinstance(ray, Ray):
                points = np.array( ray.vertices() )

                # Checks the final vector of the ray, and scales it to max length if required
                if max_length is not None and len(points) > 1:
                    last = points[-1] - points[-2]
                    length = np.linalg.norm(last)

                    if length > max_length:
                        points[-1] = points[-2] + last * max_length / length

                if i == 0:
                    line_label = label
                else:
                    line_label = None
                
                ax.plot(points[:, 2], points[:, 0], color=colour, label=line_label)

        if label is not None:
            ax.legend()

        if elements is None:
            return ax

        # Turn the parameter 'elements' into a list if only one element is passed in
        if not isinstance(elements, list):
            elements = [elements]

        # Draw the elements onto the plot
        for element in elements:
            patches = element.draw()

            if not isinstance(patches, list):
                patches = [patches]
                
            for patch in patches: ax.add_patch(patch)

        return ax
    

    def mean_direction(self):
        """
        Function to calculate the mean direction of the bundle

        Returns:
            mean_direc (Numpy array) : 3D normalised vector of the mean direction of the bundle
        """
        direcs = np.array([ray.direc() for ray in self.rays()])
        mean_direc = np.mean(direcs, axis=0)
        mean_direc /= np.linalg.norm(mean_direc)
        return mean_direc

    def best_focus(self, direction = None):
        """
        Function that calculates the best focus of a ray bundle.
        
        Args:
            direction (list) : Optional focusing direction.
        
        Returns:
            focus (Numpy array) : The focus point of our beam with the lowest rms.
            t_best (float) : The distance from the end point where the focus lies.
            rms_width (float) : The rms width of the spot at the focus position.
        """

        # Get the end points of the rays, and store in a list
        end_points = np.array([ray.vertices()[-1] for ray in self.rays()])

        # Get the directions of each ray
        directions = np.array([ray.direc() for ray in self.rays()])
        beam_axis = self.mean_direction()

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
        focus_crossings = self.plane_crossings(reference_point + t_best * beam_axis, beam_axis)
        focus = np.mean(focus_crossings, axis = 0)

        # rms width of the spot, measured along the same axes used to find the focus
        spread = 0

        for axis in axes:
            positions = np.dot(focus_crossings - focus, axis)
            spread += np.var(positions)

        rms_width = np.sqrt(spread)

        return focus, t_best, rms_width

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

    def convergence_angle(self, direction):
        """
        Calculates the half-angle of the beam in a given direction
        - how steeply the rays come together at the focus, in one direction
        - sets the depth of focus, or how far the camera can be before blurring is noticeable

        Args:
            direction (Numpy array) : A given 3D vector direction

        Returns:
            half_angle (float) : The "steepness" of the rays coming into focus
        """
        direcs = np.array([ray.direc() for ray in self.rays()])
        beam_axis = self.mean_direction()

        # Ensures direction is perpendicular to the beam axis, and normalised
        direction = np.array(direction, dtype=float)
        direction = direction - np.dot(direction, beam_axis) * beam_axis
        direction /= np.linalg.norm(direction)
        
        half_angle = np.ptp( np.arctan( np.dot(direcs, direction) / np.dot(direcs, beam_axis) ) ) / 2
        return half_angle
    