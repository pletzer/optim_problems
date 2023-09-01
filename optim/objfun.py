import numpy
import defopt

class ObjFun(object):

    def eval(self, *params):
        pass

    def evalGradient(self, params):
        pass

    def next(self, gamma, params):
        return params - gamma*self.evalGradient(params)
        


class ObjFunSimple(ObjFun):

    def __init__(self, a, b):
        self.a = a
        self.b = b


    def eval(self, params):
        return self.a*(params[0] - self.b)**2


    def evalGradient(self, params):
        print(f'params = {params}')
        return numpy.array([2*self.a*(params[0] - self.b)])


def initialize(*, params: list[float], output_file: str='params.npy'):
    """
    Initialize the parameters and write them to a file
    :param params: parameters
    :param output_file: output file
    """
    print(f'params = {params}')
    numpy.save(output_file, numpy.array(params))  


def evaluate(*, fun_file: str='obj_fun.pickle', params_file: str='params.npy', output_file: str='f.npy'):
    """
    Evaluate the objective function
    :param fun_file: objective function instance
    :param params_file: parameters file
    :param output_file: output file containing the function evaluation
    """
    with open(fun_file, "rb") as obj_fun_file:
        obj = pickle.load(obj_fun_file)
        params = numpy.load(params_file)
        res = obj.eval(params)
        numpy.save(output_file, res)


def update(*, fun_file: str='obj_fun.pickle', params_file: str='params.npy', output_file: str='params.npy'):
    """
    Update the parameters
    :param fun_file: objective function instance
    :param params_file: parameters file
    :param output_file: output file containing the new parameter values
    """
    with open(fun_file, "wb") as obj_fun_file:
        obj = pickle.load(obj_fun_file)
        params = numpy.load(params_file)
        res = obj.evalGradient(params)
        numpy.save(output_file, res)


def display(*, params_file: str='params.npy'):
    """
    Display the current parameters
    :param params_file: file containing the parameters
    """
    params = numpy.load(params_file)
    print(f'params: {params}')



# def test(*, a: float=1., b: float=2., gamma: float=0.5):
#     """
#     """

#     of = ObjFunSimple(a=a, b=b)
#     params = numpy.array([0.])
#     f = of.eval(params)
#     diff = float('inf')
#     while diff > 0.:
#         params_new = of.next(gamma, params)
#         f_new = of.eval(params_new)
#         diff = f - f_new
#         if diff > 0.:
#             # update 
#             params[:] = params_new
#             f = of.eval(params)
#     print(f'minimizing solution: {params} giving {f}')



if __name__ == '__main__':
    defopt.run([initialize, evaluate, update, display])






