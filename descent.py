import numpy as np
from GSL import GSL

class GradientDescent:
    def __init__(self, f, g_f=None, type="steepest", eps=0.005):
        """
        f: objective
        g_f: gradient of f
        """
        self.f = f # objective function
        self.type = type
        self.func_call_count = 0
        self.g_f = g_f
        self.eps = eps

    def _diff(self, x: np.ndarray, h=10**-6):        # numerical partial differentiation
        grad = np.zeros(x.shape[-1])

        for i in grad.shape[-1]:
            x_1 = np.copy(grad)
            x_1[i] += h
            grad[i] = (self.f(x_1)-self.f(x))/h # forward difference (f(x+h) - f(x))/h
            self.func_call_count += 2

        return grad

    def descent(self, x, delta=0.05, line_search_obj=None):
        if self.type == "steepest":
            x_opt = self._steepest_descent(x, delta, line_search_obj)

        else:
            x_opt = self._conjugate_descent(x, delta, line_search_obj)

        func_value = self.f(x_opt)
        print(f"Opt x: {x_opt}, func_value: {func_value}")

        return x_opt, func_value

    def _steepest_descent(self, x, delta, line_search_obj):
        dir = -self._diff(x)

        while np.linalg.norm(dir) < self.eps:
            if line_search_obj:
                step_size = GSL(line_search_obj, delta, )

            else: step_size = 0.01

            x = x + step_size*dir
            dir = -self._diff(x)

        return x

    def _conjugate_descent(self, x, delta, line_search_obj):
        # x: initial point
        c = self._diff(x)
        beta = None

        def opt_f_auto(p, dir):
            return self.f(x + dir * p)

        while np.linalg.norm(c) < self.eps:
            if beta is None:
                dir = -c

            else:
                c_next = self._diff(x)
                beta = c_next.T.dot(c_next) / c.T.dot(c)
                dir = -c + beta*dir



            if line_search_obj:
                step_size = GSL(line_search_obj, delta)

            else:
                step_size = 0.01

            x = x + step_size * dir

        return x
