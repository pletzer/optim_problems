import numpy
import defopt
import utils


def main(*, input: str='points.npy', refvalue: float=float('inf'), output: str='objfun.npy'):
    """
    Compute the objective function
    :param input: input file containing the point's positions
    :param ref_value: reference_value from a previous call
    :param output: the file name containing the objective function value
    :returns 0 if the objfun returned a lower value, 1 otherwise
    """
    uv = numpy.load(input)
    n = uv.shape[0]
    ntot = n*(n - 1)/2 # number of non-identical pairs
    xyz = utils.compute_coords(uv)
    res = 0.
    for i in range(n):
        for j in range(i):
            dx = xyz[i, 0] - xyz[j, 0]
            dy = xyz[i, 1] - xyz[j, 1]
            dz = xyz[i, 2] - xyz[j, 2]
            res += 1.0/numpy.sqrt(dx*dx + dy*dy + dz*dz)

    print(f'objfun: {res}')
    numpy.save(output, numpy.array(res))

    if res < refvalue:
        # no error
        return 0
    return 1

if __name__ == '__main__':
    defopt.run(main)
