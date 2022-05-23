import numpy as np

MAX_ITER = 10**4


"""

1. a0, a1, a2, a3의 초기값으로 시작한 뒤에 a0 > a1 and a1 < a2인 지점을 찾을 때까지 a4, a5 ... 이런 식으로 구간을 재설정함
2. 위의 구간을 찾으면 구간 l과 u를 구간으로 설정 한 뒤에 해당 구간에서 최소값 탐색

이 알고리즘은 Introduction to Optimum Design 4th, p.440의 "Algorithm for 1D Search by Golden Sections"를 구현한 코드.

PPT 강의자료와 다르게 초기 bracket을 설정하기 때문에 일부 다른 차이가 있을 수 있음.

e.g.) ppt: f(a0)=f(delta) 부터 구간 설정, 책: f(0)부터 시작하며 f(a0)=f(delta) ...  순으로 구간 설정.

그러나 주어진 과제의 2번 문제의 제한사항을 지키며 풀기 위해서는 책의 구현체를 따라 설정해야지 풀림. 따라서 책의 방법을 구현함.

"""


def run_golden_search(f, delta=0.1, eps=10 ** -5, verbose=False, MAX_ITER = 10000):
    KSI = (1+np.sqrt(5))/2  # golden ratio
    TAU = (np.sqrt(5)-1)/2

    def _bracket(a1, a2, q):
        _a0 = a1
        _a1 = a2
        _a2 = a2 + delta * KSI ** q

        return _a0, _a1, _a2

    ################################
    # find interval
    ################################
    a0 = 0
    a1 = delta
    a2 = a1+delta*KSI
    q = 3

    # print(f"a0: {a0}, a1: {a1}, a2: {a2}")
    ################################
    # reduce interval
    ################################
    # find the interval in which possible local minimum is in
    while f(a0) > f(a1) and f(a1) > f(a2):
        a0, a1, a2 = _bracket(a1, a2, q)
        q += 1
        # print(f"a0: {a0}, a1: {a1}, a2: {a2}")

    # if verbose:
    #     print(f"a0: {a0}, a1: {a1}, a2: {a2}")
    # print("-"*100)

    ################################
    # reduce interval
    ################################
    l, a, b, u = a0, 0, 0, a2

    d = (u-l)
    b = l + TAU * d
    a = l + (1 - TAU) * d

    i = 0

    # if verbose:
    #     print(f"Initial: l: {l}, a: {a}, b: {b}, u: {u}")

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
    #     if verbose:
    #         print(f"Iter_{i}: l: {l}, a: {a}, b: {b}, u: {u}")
    #
    # if verbose:
    #     print(f"l: {l}, a: {a}, b: {b}, u: {u}")

    return (u+l)/2


if __name__ == '__main__':
    def opt_f(x):
        return x**4-14*x**3+60*x**2-70*x

    def f(x):
        return x[0]**2+x[1]**2+x[2]**2

    # def f(x):
    #     return x**4-14*x**3+60*x**2-70*x

    x0 = np.array([1,2,-1])
    d = np.array([-2,-4,2])

    def opt_f_auto(x):
        return f(x0 + d*x)

    def descent(f, x0, d=None):

        if d is None:
            # cal descent direction
            pass

        step_size = run_golden_search(f, delta=0.05, verbose=True)
        return x0+step_size*d

    run_golden_search(opt_f, delta=0.05, verbose=True, eps=0.001)

    # xt = descent(opt_f_auto, x0, d)
    # print(f"updated x: {xt} with value: {f(xt)}")

    # xt = descent(opt_f, x0, d)
    # print(f"updated x: {xt} with value: {f(xt)}")

    # def f(x):
    #     return x**4 - 14*x**3 + 60*x**2 -70*x
    #
    # print(GSL(f, verbose=True))