import numpy as np
from GSL import run_golden_search


class GradientDescent:
    def __init__(self, f, type="steepest", verbose=False, eps=0.005, delta=0.0001):
        """
        f: objective
        """
        self.f = f # objective function
        self.type = type
        self.func_call_count = 0
        self.eps = eps
        self.verbose = verbose
        self.delta = delta
        self.num_iter = 0

    def _diff(self, x: np.ndarray, h=10**-8):        # numerical partial differentiation
        grad = np.zeros(x.shape[-1])
        h_vec = np.zeros(x.shape[-1])

        for i in range(grad.shape[-1]):
            # h_vec[i] = x[i]*0.01
            # grad[i] = (self.f(x+h_vec)-self.f(x-h_vec))/2*h_vec[i] # central difference (f(x+h) - f(x-h)))/ 2h
            h_vec[i] = h
            grad[i] = (self.f(x + h_vec) - self.f(x)) / h  # forward difference (f(x+h) - f(x))/h
            self.func_call_count += 2
            h_vec[i] = 0
        """
        Opt x: [-3.45168935e-08  2.43711636e-03], func_value: 0.0002969768085458403, num_iter: 3, num_func_calls: 16
        Opt x: [-2.27860874e-05  1.05911172e-02], func_value: 0.005608589723120626, num_iter: 3, num_func_calls: 12
        """
        # print(f"grad: {grad}")
        return grad

    def descent(self, x, delta=0.05):
        if self.type == "steepest":
            x_opt = self._steepest_descent(x, delta)

        else:
            x_opt = self._conjugate_descent(x, delta)

        func_value = self.f(x_opt)
        print(f"Opt x: {x_opt}, func_value: {func_value}, num_iter: {self.num_iter}, "
              f"num_func_calls: {self.func_call_count}")

        return x_opt, func_value

    def _steepest_descent(self, x, delta):
        dir = -self._diff(x) # d = -gradient

        num_iter = 0
        while np.linalg.norm(dir) > self.eps: # until converge
            dir = -self._diff(x)

            def opt_f_auto(p): # closure function that takes local variable of method to produce step objective function
                # equals to f(alpha) in the textbook
                func_val = self.f(x + dir * p)
                return func_val

            step_size = run_golden_search(opt_f_auto, delta, verbose=self.verbose)

            x = x + step_size*dir
            func_value = self.f(x)
            num_iter += 1

            if self.verbose:
                print(f"Opt x: {x}, func_value: {func_value}, step_size: {step_size}, norm: {np.linalg.norm(dir)}\n")

        self.num_iter = num_iter
        return x

    def _conjugate_descent(self, x, delta):
        # x: initial point given
        c = self._diff(x)
        first_iter = True
        dir = None

        num_iter = 0
        while np.linalg.norm(c) > self.eps: # c is the latest gradient calculated
            if first_iter: # as no prev gradient exists
                dir = -c
                first_iter = False

            else:
                c_prev = np.copy(c)
                c = self._diff(x)
                beta = np.dot(c.T, c) / np.dot(c_prev.T, c_prev) # beta = (|c_k+1/c_k|_2)^2
                dir = -c + beta*dir

            def opt_f_auto(p):
                return self.f(x + dir * p)

            step_size = run_golden_search(opt_f_auto, delta, verbose=self.verbose)

            x = x + step_size * dir
            func_value = self.f(x)
            num_iter += 1

            if self.verbose:
                print(f"Opt x: {x}, func_value: {func_value}, step_size: {step_size}, norm: {np.linalg.norm(c)}\n")

        self.num_iter = num_iter
        return x


if __name__ == '__main__':
    # def f(x):
    #     x1 = x[0]
    #     x2 = x[1]
    #
    #     return 50*(x2-x1**2)**2+(2-x1)**2
    #
    # x0 = np.array([5, -5])

    def f(x):
        x1, x2, x3, x4 = x
        result = (x1+10*x2)**2
        result += 5*(x3-x4)**2
        result += (x2-2*x3)**4
        result += 10*(x1-x4)**4
        return result

    x0 = np.array([3, -1, 0, 1])

    # def f(x):
    #     x1 = x[0]
    #     x2 = x[1]
    #     x3 = x[2]
    #     return x1**2+2*x2**2+2*x3**2+2*x1*x2+2*x2*x3
    # x0 = np.array([2, 4, 10])

    # def f(x):
    #     x1 = x[0]
    #     x2 = x[1]
    #     x3 = x[2]
    #     return x1**2+2*x2**2+2*x3**2+2*x1*x2+2*x2*x3
    # x0 = np.array([1,1,1])

    # def f(x):
    #     x1 = x[0]
    #     x2 = x[1]
    #     return 6.983*x1**2+12.415*x2**2-x1
    # x0 = np.array([2, 1])

    eps = 0.00001

    gd_s = GradientDescent(f, verbose=False, eps=eps)
    opt_x = gd_s.descent(x0, delta=0.005)

    gd_c = GradientDescent(f, verbose=False,type='conjugate', eps=eps)
    opt_x = gd_c.descent(x0, delta=0.005)

    # def f(x):
    #     x1=x[0]
    #     x2=x[1]
    #
    #     return 20*(x1-x2)**2
    #
    # x0 = np.array([2, 1])
    #
    # gd = GradientDescent(f)
    # print(gd._diff(x0))
