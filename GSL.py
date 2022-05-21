import numpy as np

MAX_ITER = 10**4


"""

1. a0, a1, a2, a3의 초기값으로 시작한 뒤에 a0 > a1 and a1 < a2인 지점을 찾을 때까지 a4, a5 ... 이런 식으로 구간을 재설정함
2. 위의 구간을 찾으면 구간 l과 u를 구간으로 설정 한 뒤에 해당 구간에서 최소값 탐색

"""


def GSL(f, delta=0.1, eps=10**-5):
    """
    Start with four candidate points: a0, a1, a2, u
    Algorithms for unbounded search. Keep expand bracket dimension

    """

    KSI = (1+np.sqrt(5))/2  # golden ratio
    TAU = (np.sqrt(5)-1)/2

    def _bracket(a1, a2, q):
        _a0 = a1
        _a1 = a2
        _a2 = a2 + delta * KSI ** q

        return _a0, _a1, _a2

    a0 = delta
    a1 = a0+delta*KSI
    a2 = a1+delta*KSI**2
    q = 3

    # print(f"a0: {a0}, a1: {a1}, a2: {a2}")
    while f(a0) > f(a1) and f(a1) > f(a2):

        a0, a1, a2 = _bracket(a1, a2, q)
        q += 1
        # print(f"a0: {a0}, a1: {a1}, a2: {a2}")

    print(f"a0: {a0}, a1: {a1}, a2: {a2}")
    print("-"*100)

    # reduce interval
    l, a, b, u = a0, 0, 0, a2

    d = (u-l)
    b = l + TAU * d
    a = l + (1 - TAU) * d
    MAX_ITER = 2

    i = 0
    print(f"Initial: l: {l}, a: {a}, b: {b}, u: {u}")

    while abs(u-l) > eps and i < MAX_ITER: # stop if the interval becomes smaller than eps
        if f(a) > f(b):
            l = a
            a = b
            b = l + TAU*(u-l)

        elif f(a) < f(b):
            u = b
            b = a
            a = l + (1-TAU)*(u-l)

        elif f(a) == f(b):
            l = a
            u = b
            d = u-l
            b = l + TAU*d
            a = l + (1-TAU)*d

        i += 1
        print(f"No: {i} l: {l}, a: {a}, b: {b}, u: {u}")

    return (u+l)/2


if __name__ == '__main__':
    def opt_f(x):
        return 24*x**2-24*x+6

    def f(x):
        return x[0]**2+x[1]**2+x[2]**2

    x0 = np.array([1,2,-1])
    d = np.array([-2,-4,2])

    def opt_f_auto(x):
        return f(x0 + d*x)

    def descent(f, x0, d=None):

        if d is None:
            # cal descent direction
            pass

        step_size = GSL(f, delta=0.05)
        return x0+step_size*d

    x0 = descent(opt_f_auto, x0, d)
    print(f"updated x: {x0} with value: {f(x0)}")