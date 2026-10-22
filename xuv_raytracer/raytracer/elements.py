"""
elements.py
A module that stores optical elements in the system, e.g. spherical refractive surfaces.
George Peters - 15/05/25
"""
import numpy as np
import xuv_raytracer.raytracer.physics as ph


class OpticalElement:
    """
    'Blueprint' class for all optical elements
    """
    def intercept(self, ray):
        raise NotImplementedError('intercept() needs to be implemented in derived classes')
    def propagate_ray(self, ray):
        raise NotImplementedError('propagate_ray() needs to be implemented in derived classes')


class SphericalElement(OpticalElement):
    """
    A parent class for all spherical surfaces - refractors and reflectors.

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
        self._position = np.array(position)
        self._orientation = np.array(orientation) / np.linalg.norm(orientation)
        self._aperture = aperture
        self._curvature = curvature

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
            normal = -np.sign(np.dot(ray.direc(), self.orientation())) * self.orientation()
            return np.array(normal)

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
        else:

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

    def inside_aperture(self, point):
        # Checks that the interception is within the aperture radius
        r = point - self.position()
        distance_square = np.linalg.norm(r)**2 - np.dot(r, self.orientation())**2
        if distance_square > self._aperture**2:
            return False
        return True
    
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

    def position(self):
        """
        Access method for the position variable.

        Returns:
            self._position (Numpy Array) : The position of the surface's centre.
        """
        return self._position

    def orientation(self):
            """
            Access method for the orientation variable.
    
            Returns:
                self._orientation (Numpy Array) : A unit vector for the direction the surface faces at its centre
            """
            return self._orientation
    
    def aperture(self):
        """
        Access method for the aperture of the surface.

        Returns:
            self._aperture (float) : The maximum extent of the surface from the optical axis.
        """
        return self._aperture
    
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
    def __init__(self, position=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 1.0), curvature=0.0, central_groove_density=120, VLS_coefficients=[7.3e-3, 4.1e-5], groove_axis=[0, 1, 0], width=30, length=50, order=0):
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
            return

        normal = self.calc_normal(interception, ray)
        n_w = self.local_groove_density(interception)
        g = self.local_groove_direction(normal)
        diffracted_direc = ph.diffract(ray.direc(), normal, g, self.order(), ray.wavelength(), n_w)

        if diffracted_direc is None:
            return

        ray.append(interception, diffracted_direc)
        return

    def inside_aperture(self, point):

        r = point - self.position()
        w = np.dot(r, self.groove_axis()) # Distance across the grooves
        groove_direction = np.cross(self.orientation(), self.groove_axis())
        groove_direction /= np.linalg.norm(groove_direction)
        u = np.dot(r, groove_direction) # Distance along the grooves

        if np.abs(w) > self.length() / 2 or np.abs(u) > self.width() / 2:
            return False

        return True

    # VLS Helpers
    def local_groove_density(self, point):
        r = point - self.position()
        w = np.dot(r, self.groove_axis())
        n_0 = self.central_groove_density()
        n_w = n_0 * (1 + self.b1() * w + self.b2() * w**2)
        return n_w

    def local_groove_direction(self, normal):
        a = self.groove_axis()
        n = normal
        g = (a - np.dot(a, n) * n) / np.linalg.norm(a - np.dot(a, n) * n)
        return g
    
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

class SphericalRefraction(SphericalElement):
    """
    A class for a spherical surface that refracts light

    Attributes:
        _position (Numpy Array) : The position of the surface's centre.
        _orientation (Numpy array) : A unit vector for the direction the surface faces at its centre
        _aperture (float) : The maximum extent of the surface from the optical (z) axis.
        _curvature (float) : The curvature of the surface (1/Radius).
        _n_1 (float) : The refractive index to the left of the surface.
        _n_2 (float) : The refractive index to the right of the surface.
    """
    def __init__(self, position=[0.0, 0.0, 0.0], orientation=[0.0, 0.0, 1.0], aperture=0.0, curvature=0.0, n_1=0.0, n_2=0.0):
        """
        Initialise the object

        Args:
            position (Numpy Array) : The position of the surface's centre.
            orientation (Numpy array) : A unit vector for the direction the surface faces at its centre
            aperture (float) : Maximum extent of the surface from the optical axis.
            curvature (float) : The radius of curvature of the surface.
            n_1 (float) : The refractive index of the first material z<z0.
            n_2 (float) : The refractive index of the second material z>z0.
        """
        super().__init__(position, orientation, aperture, curvature)
        self._n_1 = n_1
        self._n_2 = n_2

    def n_1(self):
        """
        Access method for the n_1 variable.

        Returns:
            self.n_1 (float) : The refractive index of the first material z<z_0.
        """
        return self._n_1
    
    def n_2(self):
        """
        Access method for the n_2 variable.

        Returns:
            self._n_2 (float) : The refractive index of the second material z>z_0.
        """  
        return self._n_2
    
    def focal_point(self):
        """
        Method that will calculate the focal point of the element.

        Returns:
            f_point (Numpy array) : The focal point
        """
        radius = 1 / self._curvature
        f_length = radius * 1 / (self._n_2 - self._n_1)
        f_point = self.position() + (radius + f_length) * self.orientation()

        return f_point
    
    def propagate_ray(self, ray):
        """
        Function that propagates a ray through the optical element.

        Args:
            ray (Ray) : The ray that is being propagated
        """
        interception = self.intercept(ray)

        # Only propagates if the ray actually intercepts the surface
        if interception is not None:
            normal = self.calc_normal(interception, ray)

            # Swap n_1 and n_2 if this dot product is less than zero
            if np.dot(ray.direc(), self.orientation()) < 0:
                n_1, n_2 = self._n_2, self._n_1

            new_direction = ph.refract(ray.direc(), normal, n_1, n_2)

            # Deals with TIR
            if new_direction is not None:
                ray.append(interception, new_direction)
            else:
                ray.append(interception, ph.reflect(ray.direc(), normal))


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
    

class PlaneElement(OpticalElement):
    """
    The PlaneElement class
    Note: For the project only one child is made (OutputPlane) but this allows for future extension
          i.e. a PlaneRefractor or PlaneReflector could be implemented.

    Attributes:
        _z_0 (float) : The intercept between the element and the optical axis.
        aperture (float) : The maximum extent of the element from the optical axis.
    """
    def __init__(self, z_0=100., aperture=100.):
        """
        Initiate the PlaneElement object.

        Args:
            z_0 (float) : The intercept between the element and the optical axis.
            aperture (float) : The maximum extent of the element from the optical axis.
        """
        self._z_0 = z_0
        self._aperture = aperture

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

        l = ( self._z_0 - ray_pos[2] ) / ray_direc[2]
        intercept = ray_pos + l * ray_direc

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


class OutputPlane(PlaneElement):
    """
    An output plane, which is where the ray 'stops' and its final position can be recorded

    Attributes:
        _z_0 (float) : Where the surface intercepts the optical (z) axis.
        _aperture (float) : The maximum extent of the surface from the optical (z) axis.
    """
    
    def __init__(self, z_0=250.):
        """
        Init method for the OutputPlane class

        Args:
            z_0 (float) : The z-intercept of the element
        """
        super().__init__(z_0=z_0, aperture=float("inf"))
