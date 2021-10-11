import numpy as np


def q5b():
    x1 = np.array([1, 1, -1, 1, 3])
    x2 = np.array([-1, 1, 1, 2, -1])
    x3 = np.array([2, 3, 0, -4, -1])
    X = np.array([x1, x2, x3]).T
    y = np.array([1, 4, -1, -2, 0]).T
    w = np.array([-1, 1, -1])
    b = -1
    for j in range(len(w)):
        x_j = X[:, j]
        djdw = calc_djdw(X, y, w, b, x_j)
        print(f'dJ_dw_{j + 1}: {djdw}')
    djdb = calc_djdb(X, y, w, b)
    print(f'dJ_db: {djdb}')
    # dJ_dw_1: -22
    # dJ_dw_2: 16
    # dJ_dw_3: -56
    # dJ_db: -10


def q5c():
    x1 = np.array([1, 1, -1, 1, 3])
    x2 = np.array([-1, 1, 1, 2, -1])
    x3 = np.array([2, 3, 0, -4, -1])
    y = np.array([1, 4, -1, -2, 0]).T
    y_x1 = -np.dot(x1, y)
    print(f'y_x1: {y_x1}')
    y_x2 = -np.dot(x2, y)
    print(f'y_x2: {y_x2}')
    y_x3 = -np.dot(x3, y)
    print(f'y_x3: {y_x3}')
    
    x1_x1 = -np.dot(x1, x1)
    print(f'x1_x1: {x1_x1}')
    x1_x2 = -np.dot(x1, x2)
    print(f'x1_x2: {x1_x2}')
    x1_x3 = -np.dot(x1, x3)
    print(f'x1_x3: {x1_x3}')

    x2_x1 = -np.dot(x2, x1)
    print(f'x2_x1: {x2_x1}')
    x2_x2 = -np.dot(x2, x2)
    print(f'x2_x2: {x2_x2}')
    x2_x3 = -np.dot(x2, x3)
    print(f'x2_x3: {x2_x3}')

    x3_x1 = -np.dot(x3, x1)
    print(f'x3_x1: {x3_x1}')
    x3_x2 = -np.dot(x3, x2)
    print(f'x3_x2: {x3_x2}')
    x3_x3 = -np.dot(x3, x3)
    print(f'x3_x3: {x3_x3}')

    # y_x1: -4
    # y_x2: 2
    # y_x3: -22
    # x1_x1: -13
    # x1_x2: 2
    # x1_x3: 2
    # x2_x1: 2
    # x2_x2: -8
    # x2_x3: 6
    # x3_x1: 2
    # x3_x2: 6
    # x3_x3: -30


def q5d():
    x1 = np.array([1, 1, -1, 1, 3])
    x2 = np.array([-1, 1, 1, 2, -1])
    x3 = np.array([2, 3, 0, -4, -1])
    X = np.array([x1, x2, x3]).T
    y = np.array([1, 4, -1, -2, 0]).T
    w = np.array([0.0, 0.0, 0.0])
    b = 0
    r = 0.1
    for i in range(5):
        for j in range(len(w)):
            djdw = calc_djdw(X[i], y[i], w, b, X[i, j])
            print(f'round {i + 1}, weight updated {j + 1}: {w[j]} + {r * djdw} = {w[j] + r * djdw}')
            w[j] = w[j] + r * djdw
        djdb = calc_djdb(X[i], y[i], w, b)
        print(f'round {i + 1}, bias updated: {b} + {r * djdb} = {b + r * djdb}')
        b = b + r * djdb
    s = 0
    for k in w:
        print(f'w_{s + 1}: {k}')
        s += 1
    print(f'b: {b}')

    # round 1, weight updated 1: 0.0 + -0.1 = -0.1
    # round 1, weight updated 2: 0.0 + 0.1100 = 0.1100
    # round 1, weight updated 3: 0.0 + -0.242 = -0.242
    # round 1, bias updated: 0 + -0.1694 = -0.1694
    # round 2, weight updated 1: -0.1 + -0.4885 = -0.5885
    # round 2, weight updated 2: 0.1100 + -0.5374 = -0.4274
    # round 2, weight updated 3: -0.2420 + -1.7734 = -2.0154
    # round 2, bias updated: -0.1694 + -1.1232 = -1.2926
    # round 3, weight updated 1: -0.5885 + 0.0131 = -0.5754
    # round 3, weight updated 2: -0.4274 + -0.0145 = -0.4418
    # round 3, weight updated 3: -2.0154 + -0.0 = -2.0154
    # round 3, bias updated: -1.2926 + -0.0159 = -1.3085
    # round 4, weight updated 1: -0.5754 + 0.7294 = 0.1540
    # round 4, weight updated 2: -0.4418 + 1.6047 = 1.1628
    # round 4, weight updated 3: -2.0154 + -4.4931 = -6.5085
    # round 4, bias updated: -1.3085 + 2.9205 = 1.6121
    # round 5, weight updated 1: 0.1540 + 2.2259 = 2.3799
    # round 5, weight updated 2: 1.1628 + -1.4098 = -0.2469
    # round 5, weight updated 3: -6.5085 + -1.5507 = -8.0593
    # round 5, bias updated: 1.6121 + 1.7058 = 3.3179
    # w_1: 2.3799
    # w_2: -0.2469
    # w_3: -8.0593
    # b: 3.3179


def calc_djdw(X, y, w, b, x_j):
    return -np.sum(np.dot((y - np.dot(X, w) - b), x_j))


def calc_djdb(X, y, w, b):
    return -np.sum(y - np.dot(X, w) - b)


def calc_cost(X, y, w, b):
    return 0.5 * np.sum(np.square(y - np.dot(X, w) - b))
