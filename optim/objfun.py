import numpy
import defopt
import pickle
from sympy.parsing.sympy_parser import parse_expr
from sympy import symbols, diff, N


class ObjFun(object):

    def eval(self, params):
        pass

    def evalGradient(self, params):
        pass

    def next(self, params, gamma):
        # steepest descent
        print(f'params = {params} grad f = {self.evalGradient(params)}')
        return numpy.array(params) - gamma*self.evalGradient(params)


class ObjFunSimple(ObjFun):

    def __init__(self, fun, tol):
        self.x, self.y, self.z = symbols('x y z', real=True)
        self.fun = parse_expr(fun, {'x': self.x, 
                                    'y': self.y,
                                    'z': self.z})
        self.grad = [diff(self.fun, self.x), 
                     diff(self.fun, self.y),
                     diff(self.fun, self.z)]
        self.tol = abs(tol)


    def eval(self, params):
        d = self.unpackParams(params)
        return self.fun.evalf(subs=d)


    def evalGradient(self, params):
        d = self.unpackParams(params)
        return numpy.array([N(g.evalf(subs=d)) for g in self.grad]) 


    def unpackParams(self, params):
        d = {self.x: params[0]}
        if len(params) > 1:
            d[self.y] = params[1]
            if len(params) > 2:
                d[self.z]= params[2]
        return d
     


def initialize(*, lfun: str='1*(x-2)**2',
                  tol: float=1.e-3,
                  values_params: list[float],
                  params_file: str='params.npy',
                  fun_file: str='obj_fun.pickle'):
    """
    Initialize the parameters and write them to a file
    :param lfun: likelihood/objective function expression
    :param tol: function evaluation iteration tolerance
    :param values_params: initial parameters
    :param params_file: output file containing the parameters
    :param fun_file: output file containing the objective function instance
    """
    numpy.save(params_file, numpy.array(values_params))
    obj = ObjFunSimple(fun=lfun, tol=tol)
    with open(fun_file, "wb") as obj_fun_file:
        pickle.dump(obj, obj_fun_file)


def update(*, fun_file: str='obj_fun.pickle',
              params_file: str='params.npy',
              gamma: float=1.0,
              output_file: str='params.npy'):
    """
    Update the parameters
    :param fun_file: objective function instance
    :param params_file: input parameters file
    :param gamma: update coefficient
    :param output_file: output file containing the new parameter values
    :raises ValueError: if function value did not decrease
    """
    with open(fun_file, "rb") as obj_fun_file:
        obj = pickle.load(obj_fun_file)
        params = numpy.load(params_file, allow_pickle=True)
        print(f'current params: {params}')
        f = obj.eval(params)
        res = obj.next(params=params, gamma=gamma)
        print(f'updated params: {res}')
        new_f = obj.eval(res)
        if new_f > f - obj.tol:
            raise ValueError(f'f = {f} new_f = {new_f} > f - tol = {f - obj.tol}')
        print(f'saving the updated params values {res} in file {output_file}')
        numpy.save(output_file, res)


def display(*, params_file: str='params.npy'):
    """
    Display the current parameters
    :param params_file: file containing the parameters
    """
    params = numpy.load(params_file, allow_pickle=True)
    print(f'params: {params}')



if __name__ == '__main__':
    defopt.run([initialize, update, display])






