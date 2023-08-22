import numpy
import defopt

def main(*, n: int=10, seed: int=123, output: str='points.npy'):
    """
    Produce random initial positions
    :param n: number of points
    :param seed: random number generator seed
    :param output: output file
    """
    uv0 = numpy.random.random(size=n*2).reshape((-1, 2))
    numpy.save(output, uv0)


if __name__ == '__main__':
    defopt.run(main)
