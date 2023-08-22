import numpy
from scipy.optimize import minimize
import defopt
import utils


def objfun(uvs, uv_ref):

    xyz = utils.compute_coords(uvs)
    xyz_ref = utils.compute_coords(uv_ref).squeeze()

    n = uvs.shape[0]
    res = 0.
    for i in range(n):
        dx = xyz[i, 0] - xyz_ref[0]
        dy = xyz[i, 1] - xyz_ref[1]
        dz = xyz[i, 2] - xyz_ref[2]
        res += 1.0/numpy.sqrt(dx*dx + dy*dy + dz*dz)
    return res/n


def distancesSquare(uvs, uv_ref):

    xyz = utils.compute_coords(uvs)
    xyz_ref = utils.compute_coords(uv_ref).squeeze()
    n = uvs.shape[0]
    res = numpy.empty((n,), numpy.float64)
    for i in range(n):
        dx = xyz[i, 0] - xyz_ref[0]
        dy = xyz[i, 1] - xyz_ref[1]
        dz = xyz[i, 2] - xyz_ref[2]
        res[i] = dx*dx + dy*dy + dz*dz

    return res


def main(*, input: str='points.npy', index: int=0, nnear: int=3, output: str=f'point.npy'):
    """
    Optimize the location of a point
    :param input: input file containing the point's positions
    :param index: the index of thepoint to move
    :param nnear: the number of nearest points used
    :param output: the file containing the updated coordinates
    """
    # read the points
    uvs = numpy.load(input)

    # extract the point of interest
    uv_ref = uvs[index, :]

    # find the nnear nearest points
    scores = distancesSquare(uvs, uv_ref)
    inds = numpy.argsort(scores)
    uv_subset = uvs[inds[1:(nnear + 1)], :]

    # optimize the location of the point using its vicinity
    res = minimize(objfun, uv_subset, args=(uv_ref,), method='L-BFGS-B', options={'maxiter': 100, 'maxfun': 1000})

    # save the point's coordinates
    numpy.save(output, res.x)


if __name__ == '__main__':
    defopt.run(main)
