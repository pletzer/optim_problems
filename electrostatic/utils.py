import numpy

def compute_coords(uv):
    """
    Compute the Cartesian coordinates from the parametric u,v coordinates
    :param uv: array (n, 2) or (2,)
    """

    # u ~ lon, v ~ lat
    n = 1
    if len(uv.shape) > 1:
        n = uv.shape[0]

    xyz = numpy.empty((n, 3), numpy.float64)

    # on the sphere
    rho = numpy.cos(-numpy.pi/2. + uv[..., 1]*numpy.pi)
    xyz[:, 0] = rho * numpy.cos(uv[..., 0]*2*numpy.pi)
    xyz[:, 1] = rho * numpy.sin(uv[..., 0]*2*numpy.pi)
    xyz[:, 2] = numpy.sin(-numpy.pi/2. + uv[..., 1]*numpy.pi)

    return xyz
