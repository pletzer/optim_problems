import numpy
import defopt
import pickle

class ObjFun(object):

    def eval(self, *params):
        pass

    def evalGradient(self, params):
        pass

    def next(self, params, gamma):
        # steepest descent
        print(f'params = {params} grad f = {self.evalGradient(params)}')
        return params - gamma*self.evalGradient(params)
        


class ObjFunSimple(ObjFun):

    def __init__(self, a, b, tol):
        self.a = a
        self.b = b
        self.tol = abs(tol)


    def eval(self, params):
        return self.a*(params[0] - self.b)**2


    def evalGradient(self, params):
        return numpy.array([2*self.a*(params[0] - self.b)])


def initialize(*, a: float=1.0, b: float=2.0, tol: float=1.e-3, params: list[float], output_file: str='params.npy', fun_file: str='obj_fun.pickle'):
    """
    Initialize the parameters and write them to a file
    :param a: a coefficient
    :param b: b coefficient
    :param tol: function evaluation iteration tolerance
    :param params: initial parameters
    :param output_file: output file containing the parameters
    :param fun_file: output file containing the objective function instance
    """
    print(f'params = {params}')
    numpy.save(output_file, numpy.array(params))
    obj = ObjFunSimple(a=a, b=b, tol=tol)
    with open(fun_file, "wb") as obj_fun_file:
        pickle.dump(obj, obj_fun_file)


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
        print(f'evaluation: {res}')
        numpy.save(output_file, res)


def update(*, fun_file: str='obj_fun.pickle', params_file: str='params.npy', f_file: str='f.npy', gamma: float=1.0, output_file: str='params.npy'):
    """
    Update the parameters
    :param fun_file: objective function instance
    :param params_file: parameters file
    :param f_file: file containing the current function evaluation
    :param gamma: update coefficient
    :param output_file: output file containing the new parameter values
    :returns 0 if iteration ducceeded, 1 otherwise
    """
    with open(fun_file, "rb") as obj_fun_file:
        obj = pickle.load(obj_fun_file)
        params = numpy.load(params_file)
        f = numpy.load(f_file)
        print(f'current params: {params}')
        res = obj.next(params=params, gamma=gamma)
        print(f'updated params: {res}')
        new_f = obj.eval(res)
        numpy.save(output_file, res)
        if f - new_f < obj.tol:
            return 1 # fail
        else:
            return 0 # succeed




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






