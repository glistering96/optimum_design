import numpy as np

def GSL(f, interval=0.1, eps=10 ^ -5):
    def _reduction_interval(l, r, a, b):
        I = r - l
        b = l_bound + tau * I

    def _bracket(h1, current, ksi, q, delta):
        # memory optimized bracket function using Dynamic Programming
        _h2 = h1
        _h1 = current
        _current = current + delta*np.power(ksi, q)

        return _h2, _h1, _current

    l_bound, r_bound, a, b = 0, 0, 0, 0
    converged = False
    ksi = 1.618  # golden ratio
    tau = 0.618
    delta = interval

    while not converged:
        # evaluate intervals
        found = False # flag for candidate optimum point. Function value trend change indicator
        q = 3

        h2= delta
        h1 = h2 + delta*np.power(ksi, 1)
        current = h1+delta * np.power(ksi, 2)

        while q < 10 ^ 6 and not found:
            print(f"{h2}, {h1}, {current}")
            if h2 >= h1 and h1 <= current:
                found = True
                l_bound = h2
                a = h1
                r_bound = current

            else:
                h2, h1, current = _bracket(h1, current, ksi, q, delta)
                q += 1

        if not found:
            print("Maximum number of search iteration reached. Cannot find change in function value. "
                  "Check if the function value is monotonically increasing or decreasing.")
            exit()



if __name__ == '__main__':
    GSL(lambda x: np.power(x, 2))