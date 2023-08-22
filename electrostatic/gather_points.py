import numpy
import defopt
import glob
import re


def main(*, input: str='newpoints*.npy', n: int=10, output: str=f'points.npy'):
    """
    Gather the new points 
    :param input: input file regex for the individual points after local optimization
    :param n: the number of points
    :param output: the file containing the updated coordinates
    """
    uvs = numpy.empty((n, 2), numpy.float64)

    for filename in glob.glob(input):
        m = re.search(r'[^\d]0*(\d+)\.npy', filename)
        index = int(m.group(1))
        uvs[index, :] = numpy.load(filename)

    numpy.save(output, uvs)


if __name__ == '__main__':
    defopt.run(main)
