"""
elements.py
A module that stores optical elements in the system, e.g. spherical mirrors and gratings
George Peters - 15/05/25
"""
import numpy as np
import matplotlib.pyplot as plt
import xuv_raytracer.raytracer.physics as ph


class OpticalElement:
    """
    'Blueprint' class for all optical elements
    """
    def intercept(self, ray):
        raise NotImplementedError('intercept() needs to be implemented in derived classes')
    def propagate_ray(self, ray):
        raise NotImplementedError('propagate_ray() needs to be implemented in derived classes')

class PlaneElement(OpticalElement):
    """
    The PlaneElement class, essentially a spherical element with 0 curvature

    Attributes:
        _position (Numpy Array) : The position of the surface's centre.
        _orientation (Numpy array) : A unit vector for the direction the surface faces at its centre
        _aperture (float) : The maximum extent of the element from the optical axis.
    """
    def __init__(self, position=[0, 0, 0], orientation=[0, 0, 1], aperture=100.):
        """
        Initiate the PlaneElement object.

        Args:
            position (Numpy Array) : The position of the surface's centre.
            orientation (Numpy array) : A unit vector for the direction the surface faces at its centre
            aperture (float) : The maximum extent of the element from the optical axis.
        """
        self._position = np.array(position, dtype=float)
        self._orientation = np.array(orientation, dtype=float) / np.linalg.norm(orientation)
        self._aperture = aperture

        return

    def position(self):
        """
        Access method for the position of the element

        Returns:
            _position (Numpy Array) : The position of the surface's centre.
        """
        return self._position

    def orientation(self):
        """
        Access method for the orientation of the element

        Returns:
            _orientation (Numpy array) : A unit vector for the direction the surface faces at its centre
        """
        return self._orientation

    def aperture(self):
        """
        Access method for the aperture of the surface.

        Returns:
            self._aperture (float) : The maximum extent of the surface from the optical axis.
        """
        return self._aperture

    def intercept(self, ray):
        """
        Function that calculates the interception between a ray and the planar element.

        Args:
            ray (Ray) : The ray being intercepted.

        Returns:
            intercept (Numpy array) : The 3D vector of the interception point.
        """
        ray_pos = ray.pos()
        ray_direc = ray.direc()

        dot = np.dot(ray_direc, self.orientation())
        
        # The ray runs parallel with the surface!
        if np.abs(dot) < 1e-12:
            return None

        l = np.dot(self.position() - ray_pos, self.orientation())
        l /= np.dot(ray_direc, self.orientation())

        # The crossing is behind the ray
        if not l > 1e-9:
            return None

        intercept = ray_pos + l * ray_direc
        in_aperture = self.inside_aperture(intercept)

        if not in_aperture:
            return None

        return intercept

    def propagate_ray(self, ray):
        """
        Propagate function that propagates the ray through the element.
        By default the direction will remain unchanged.

        Args:
            ray (Ray) : The ray being intercepted.
        """
        interception = self.intercept(ray)
        
        if interception is not None:
            ray.append(interception, ray.direc())
        else:
            ray.set_terminated()

        return

    def inside_aperture(self, point):
        """
        A function to check if the point lies within the element's aperture

        Args:
            point (Numpy array) : The 3D position of the point

        Returns:
            True / False (boolean) : Whether the point lies within the aperture or not
        """
        # Checks that the interception is within the aperture radius
        r = point - self.position()
        distance_square = np.linalg.norm(r)**2 - np.dot(r, self.orientation())**2
        if distance_square > self._aperture**2:
            return False
        return True

    def calc_normal(self, point, ray):
        """
        A function to calculate the normal to the surface at a given point

        Args:
            point (Numpy array) : The 3D point on the element
            ray (Ray) : The ray intercepting the element

        Returns:
            normal (Numpy array) : The normal to the surface at the given point
        """
        normal = -np.sign(np.dot(ray.direc(), self.orientation())) * self.orientation()
        return normal
    

class Detector(PlaneElement):
    """
    A class for a detector, which has a rectangular rather than circular aperture.

    Attributes:
        _position (Numpy Array) : The position of the surface's centre.
        _orientation (Numpy array) : A unit vector for the direction the surface faces at its centre
        _width (float) : The width of the detector
        _height (float) : The height of the detector
        _across_axis (Numpy array) : The axis across the detector (in the detector plane)
        _along_axis (Numpy array) : The axis along the detector (in the detector plane)
    """
    def __init__(self, position=[0.0, 0.0, 0.0], orientation=[0.0, 0.0, 1.0], up=[0.0, 1.0, 0.0], width=np.inf, height=np.inf):
        super().__init__(position, orientation, np.inf)

        # Remove the component of "up" along the normal, so it lies in the detector plane
        across = np.array(up, dtype=float)
        across = across - np.dot(across, self.orientation()) * self.orientation()

        # If "up" was parallel to the normal, nothing is left to define it as an axis
        if np.linalg.norm(across) < 1e-12:
            raise Exception("up must not be parallel to the detector's orientation")

        # Define the axes of the detector
        self._across_axis = across / np.linalg.norm(across)
        self._along_axis = np.cross(self.orientation(), self._across_axis)

        self._width = width
        self._height = height

        return

    # Access Methods
    def across_axis(self):
        return self._across_axis

    def along_axis(self):
        return self._along_axis

    def width(self):
        return self._width

    def height(self):
        return self._height

    def coordinates(self, point):
        """
        Converts 3D points in free space to a coordinate system on the detector plane

        Args:
            point (Numpy array) : The point to convert to the detector's coordinates

        Returns:
            coordinates_along (Numpy array) : How far along the detector the point lies
            coordinates_across (Numpy array) : How far across the detector the point lies
        """
        # Vector from sensor's centre to the point
        r = np.array(point) - self.position()

        # Compont of vector along both axes of the detector
        coordinates_along = np.dot(r, self.along_axis())
        coordinates_across = np.dot(r, self.across_axis())

        return coordinates_along, coordinates_across

    def inside_aperture(self, point):
        """
        A function to check if the point lies within the element's aperture
        Overrides parent class as this is a rectangular aperture

        Args:
            point (Numpy array) : The 3D position of the point to check

        Returns:
            True / False (boolean) : Whether the point lies within the aperture or not
        """
        coordinates_along, coordinates_across = self.coordinates(point)

        if np.abs(coordinates_along) < self.width()/2:
            if np.abs(coordinates_across) < self.height()/2:
                return True

        return False

    def distance_along(self, point, direction):
        """
        Calculates how far the point is from the detector in the given direction
        This deals with the fact that it is not the shortest possible distance (detector could be tilted)

        Args:
            point (Numpy array) : The current point in space
            direction (Numpy array) : The direction the ray is going in

        Returns:
            distance (float) : The distance along the direction between the point and detector
        """
        direction = np.array(direction, dtype=float) / np.linalg.norm(direction)
        dot = np.dot(direction, self.orientation())

        if np.abs(dot) < 1e-12:
            return None # Travelling parallel to the plane

        distance = np.dot(point - self.position(), self.orientation()) / dot
        return distance

    def incidence_angle(self, direction):
        """
        Angle between a direction and the detector normal

        Args:
            direction (Numpy array) : Given direction (3D vector)

        Returns:
            angle (float) : The angle between the direction and the normal (radians between 0 and pi/2)
        """
        direction = np.array(direction, dtype=float) / np.linalg.norm(direction)
        cos_angle = np.abs(np.dot(direction, self.orientation()))
        angle = np.arccos(np.clip(cos_angle, 0, 1)) # Clip avoids rounding errors, e.g. 1.000000000000004
        return angle

    # Ray Bundle methods - applies the above methods to an entire bundle of rays
    def ray_coordinates(self, bundle):
        """
        Converts the positions of all rays in the bundle to the detector's coordinates
        This works AFTER a ray bundle has propagated and hit the bundle, as ray.pos() is the final position

        Args:
            bundle (RayBundle) : The bundle of rays incident upon the detector

        Returns:
            coordinates (Numpy array) : Array of coordinates for each ray
        """
        points = np.array([ray.pos() for ray in bundle.rays()])
        coordinates = self.coordinates(points)
        return coordinates

    def line_width(self, bundle):
        """
        Gives how spread out the line is along the sensor

        Args:
            bundle (RayBundle) : The bundle of rays incident upon the detector

        Returns:
            std (float) : The standard deviation of the rays' positions along the detector
        """
        along, _ = self.ray_coordinates(bundle)
        std = np.std(along)
        return std

    def spot_plot(self, bundle, ax=None, colour=None, label=None):
        """
        Returns a picture of what the detector "sees"
        - uses self.ray_coordinates() to get the ray positions in the detector's coordinate system

        Args:
            bundle (RayBundle) : The ray bundle incident upon the detector
            ax (Axis) : Optional existing axis argument - can write on previous graphs
            colour (Color) : Optional colour argument for the points
            label (String) : Optional label for a legend

        Return:
            ax (Axis) : The plot
        """
        if ax is None:
            figure, ax = plt.subplots()

        along, across = self.ray_coordinates(bundle)
        ax.scatter(along, across, color=colour, label=label, s=5)
        ax.set_xlabel("Position along detector (mm)")
        ax.set_ylabel("Position across detector (mm)")

        if label is not None:
            ax.legend()

        return ax


class SphericalElement(PlaneElement):
    """
    A parent class for all spherical surfaces - reflectors and gratings
    It is a child of a PlaneElement, as it is the same, but with a curvature

    Attributes:
        _position (Numpy Array) : The position of the surface's centre.
        _orientation (Numpy array) : A unit vector for the direction the surface faces at its centre
        _aperture (float) : The maximum extent of the surface from the element's axis.
        _curvature (float) : The curvature of the surface (1/Radius).
    """

    def __init__(self, position=[0.0, 0.0, 0.0], orientation=[0.0, 0.0, 1.0], aperture=6., curvature=-0.02):
        """
        Initiates the SphericalSurface objects

        Args:
            position (Numpy Array) : The position of the surface's centre.
            orientation (Numpy array) : A unit vector for the direction the surface faces at its centre
            aperture (float) : The maximum extend of the surface from the optical axis.
            curvature (float) : The curvature of the surface (1/radius).
        """
        super().__init__(position, orientation, aperture)
        self._curvature = curvature
        return

    def calc_normal(self, point, ray):
        """
        Method that will calculate the normal to the surface at a given point

        Args:
            point (Numpy array) : 3D vector pointing to a specific point on the surface.
            ray (Ray) : The ray being intercepted by the surface.

        Returns:
            normal (Numpy array) : 3D normalised vector specifiying the normal to the surface.
        """
        # Flat Surface (plus or minus the orientation vector)
        if self._curvature == 0:
            return super().calc_normal(point, ray)

        # Curved Surface (C = P + (1/c) a)
        r_centre = self.position() + self.orientation()/self.curvature()
        normal = point - r_centre
        normal /= np.linalg.norm(normal)
        normal *= -np.sign(np.dot(ray.direc(), normal))

        return np.array(normal)

    def intercept(self, ray):
        """
        Method that will calculate the intercept between a given ray and the surface.
        There is code to take into account the aperture of the surface.
        Also, the centre of curvature is not always at the origin.

        Args:
            ray (Ray) : The ray that is incident upon the surface.

        Returns:
            intercept (Numpy array) : The interception point between the ray and surface.
        """
        ray_pos = ray.pos()
        ray_direc = ray.direc()

        # Flat Surface
        if self._curvature == 0:
            return super().intercept(ray)

        # Takes care about the fact that the centre of the sphere is not at the origin.
        radius = 1 / self._curvature
        r_centre = self.position() + self.orientation()/self.curvature()
        r_ray = ray_pos - r_centre

        dot = np.dot(r_ray, ray_direc)
        discriminant = dot * dot - ( np.linalg.norm(r_ray)**2 - radius**2 )

        # Deals with no interception.
        if discriminant < 0:
            return None

        # Deals with there being 2 points of intersection.
        l_min = -1 * dot - np.sqrt( discriminant )
        l_add = -1 * dot + np.sqrt( discriminant )
        l_vals = (l_min, l_add)

        for l in l_vals:
            if not l > 1e-9:
                continue

            intercept = ray_pos + l * ray_direc

            # Interception on far side of the sphere if dot < 0
            dot = np.dot(intercept - r_centre, self.position() - r_centre)
            if dot < 0:
                continue

            in_aperture = self.inside_aperture(intercept)
            if not in_aperture:
                continue

            return intercept
            
        return None

    
    def curvature(self):
        """
        Access method for the curvature variable.

        Returns:
            self._curvature (float) : The radius of curvature of the surface.
        """
        return self._curvature
       

class SphericalVLSGrating(SphericalElement):
    """
    A class for a concave variable line spaced (VLS) grating.
    It has spherical curvature, so is a child of the SphericalElement class.
    Uses the physics.diffract() method in propagate_ray().

    Attributes:
        _position (Numpy Array) : The position of the surface's centre.
        _orientation (Numpy array) : A unit vector for the direction the surface faces at its centre
        _curvature (float) : The curvature of the surface (1/Radius).
        _central_groove_density (int) : The lines / mm in the centre of the grating.
        _VLS_coefficients (Numpy array) : Array of the floats for the polynomial determining how the line density changes.
            - uses n(w) = n_0(1 + b_1 w + b_2 w^2)
            - w is distance from centre in mm, positive towards the detector.
         _groove_axis (Numpy array) : 3D normalised vector defining what direction the grooves are (perpendicular across the grooves)
        _width (float) : Width of the rectangular grating in mm
        _length (float) : Length of the rectangular grating in mm (Dispersion direction)
        _order (int) : The diffraction order
    """
    def __init__(self, position=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 1.0), curvature=0.0, central_groove_density=120, 
                 VLS_coefficients=[7.3e-3, 4.1e-5], groove_axis=[0, 1, 0], width=30, length=50, order=0):
        """
        Initialise the object
        
        Args:
            position (Numpy Array) : The position of the surface's centre.
            orientation (Numpy array) : A unit vector for the direction the surface faces at its centre
            curvature (float) : The curvature of the surface (1/Radius).
            central_groove_density (int) : The lines / mm in the centre of the grating.
            VLS_coefficients (Numpy array) : Array of the floats for the polynomial determining how the line density changes.
            groove_axis (Numpy array) : 3D normalised vector defining what direction the grooves are (perpendicular)
            width (float) : Width of the rectangular grating in mm
            length (float) : Length of the rectangular grating in mm
            order (int) : Diffraction order
        """
        super().__init__(position, orientation, np.inf, curvature)

        self._central_groove_density = central_groove_density
        self._VLS_coefficients = np.array(VLS_coefficients)
        self._groove_axis = np.array(groove_axis) / np.linalg.norm(groove_axis)
        self._width = width
        self._length = length
        self._order = order

        # Check groove axis lies in the grating surface at its centre
        if not np.abs(np.dot(self.groove_axis(), self.orientation())) < 1e-12:
            raise Exception("Groove axis should lie in grating surface at its centre")

    def propagate_ray(self, ray):
        interception = self.intercept(ray)

        if interception is None:
            ray.set_terminated()
            return

        normal = self.calc_normal(interception, ray)
        n_w = self.local_groove_density(interception)
        g = self.local_groove_direction(normal)
        diffracted_direc = ph.diffract(ray.direc(), normal, g, self.order(), ray.wavelength(), n_w)

        if diffracted_direc is None:
            ray.set_terminated()
            return

        ray.append(interception, diffracted_direc)
        return

    def inside_aperture(self, point):

        r = point - self.position()
        w = self.w(point)
        groove_direction = np.cross(self.orientation(), self.groove_axis())
        groove_direction /= np.linalg.norm(groove_direction)
        u = np.dot(r, groove_direction) # Distance along the grooves

        if np.abs(w) > self.length() / 2 or np.abs(u) > self.width() / 2:
            return False

        return True

    # VLS Helpers
    def local_groove_density(self, point):
        w = self.w(point)
        n_0 = self.central_groove_density()
        n_w = n_0 * (1 + self.b1() * w + self.b2() * w**2)
        return n_w

    def local_groove_direction(self, normal):
        a = self.groove_axis()
        n = normal
        g = (a - np.dot(a, n) * n) / np.linalg.norm(a - np.dot(a, n) * n)
        return g

    def illuminated_grooves(self, points):
        """
        A function to count how many grooves the light actually lands on
        - sets a hard limit to the resolution
        - R = lambda / delta lambda <= m * N

        Args:
            points (Numpy array) : N x 3 array of where the rays hit the grating

        Returns:
            n_grooves (float) : The number of grooves between the edges of the footprint
        """
        w = self.w(points)

        # Antiderivative of 1 + b1 w + b2 w^2
        def integral(x):
            return x + self.b1() * x**2 / 2 + self.b2() * x**3 / 3

        # groove density x definite integral = groove count
        n_grooves = self.central_groove_density() * (integral(w.max()) - integral(w.min()))
        return n_grooves

    def resolving_power(self, points):
        """
        The diffraction-limited resolving power for light covering these points: |m| x N.

        Args:
            points (Numpy array) : N x 3 array of points hitting the grating

        Returns:
            res_pow (float) : The resolving power, |m| x N
        """
        res_pow = abs(self.order()) * self.illuminated_grooves(points)
        return res_pow


    def w(self, point):
        """
        Finds how far a point on the grating is from its centre, across the grooves

        Args:
            points (Numpy array) : A 3D point on the grating, or an N x 3 array of points

        Returns:
            w (float or Numpy array) : The distance across the grooves in mm
        """
        w = np.dot(np.array(point) - self.position(), self.groove_axis())
        return w
    
    # Access Functions
    def central_groove_density(self):
        return self._central_groove_density

    def b1(self):
        return self._VLS_coefficients[0]

    def b2(self):
        return self._VLS_coefficients[1]

    def groove_axis(self):
        return self._groove_axis

    def width(self):
        return self._width

    def length(self):
        return self._length

    def order(self):
        return self._order

class SphericalReflection(SphericalElement):
    """
    A class for a spherical surface that reflects light

    Attributes:
        _position (Numpy Array) : The position of the surface's centre.
        _orientation (Numpy array) : A unit vector for the direction the surface faces at its centre
        _aperture (float) : The maximum extent of the surface from the optical (z) axis.
        _curvature (float) : The curvature of the surface (1/Radius).
    """
    def __init__(self, position=[0.0, 0.0, 0.0], orientation=[0.0, 0.0, 1.0], aperture=6., curvature=-0.02):
        """
        Initiate the SphericalReflection objects

        Args:
            position (Numpy Array) : The position of the surface's centre.
            orientation (Numpy array) : A unit vector for the direction the surface faces at its centre
            aperture (float) : The maximum extent of the surface from the optical axis.
            curvature (float) : The curvature (1/radius) of the surface.
        """
        super().__init__(position, orientation, aperture, curvature)

    def focal_point(self):
        """
        Method that will calculate the focal point of the element.

        Returns:
            f_point (Numpy array) : The focal point
        """
        radius = 1 / self._curvature
        focal_length = radius / 2
        focal_point = self.position() + focal_length * self.orientation()
        return focal_point

    def propagate_ray(self, ray):
        """
        Function that propagates a ray through the optical element.

        Args:
            ray (Ray) : The ray that is being propagated
        """
        interception = self.intercept(ray)

        if interception is not None:
            normal = self.calc_normal(interception, ray)
            new_direction = ph.reflect(ray.direc(), normal)
            ray.append(interception, new_direction)
        else:
            ray.set_terminated()

        return

