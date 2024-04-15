import math
import random

import numpy as np

# This code was based on in the following references:
# [1] "Defining a Standard for Particle Swarm Optimization" published in 2007 by Bratton and Kennedy


class ObjectiveFunction(object):
    # Необходимо указать "размеры аквариума" и размерность для дальнейшего построения нормального распределения
    # evaluate -- оценка агента
    def __init__(self, name, dim, minf, maxf):
        self.function_name = name
        self.dim = dim
        self.minf = minf
        self.maxf = maxf

    def evaluate(self, x):
        pass


class SphereFunction(ObjectiveFunction):
    def __init__(self, dim):
        super(SphereFunction, self).__init__('Sphere', dim, -100.0, 100.0)

    def evaluate(self, x):
        return np.sum(x ** 2)


class RosenbrockFunction(ObjectiveFunction):
    def __init__(self, dim):
        super(RosenbrockFunction, self).__init__(
            'Rosenbrock', dim, -30.0, 30.0)

    def evaluate(self, x):
        sum_ = 0.0
        for i in range(1, len(x) - 1):
            sum_ += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2
        return sum_


class RastriginFunction(ObjectiveFunction):
    def __init__(self, dim):
        super(RastriginFunction, self).__init__('Rastrigin', dim, -5.12, 5.12)

    def evaluate(self, x):
        f_x = [xi ** 2 - 10 * math.cos(2 * math.pi * xi) + 10 for xi in x]
        return sum(f_x)


class SchwefelFunction(ObjectiveFunction):
    def __init__(self, dim):
        super(SchwefelFunction, self).__init__('Schwefel', dim, -30.0, 30.0)

    def evaluate(self, x):
        sum_ = 0.0
        for i in range(0, len(x)):
            in_sum = 0.0
            for j in range(i):
                in_sum += x[j] ** 2
            sum_ += in_sum
        return sum_


class GeneralizedShwefelFunction(ObjectiveFunction):
    def __init__(self, dim):
        super(GeneralizedShwefelFunction, self).__init__(
            'GeneralizedShwefel', dim, -30.0, 30.0)

    def evaluate(self, x):
        f_x = [xi * np.sin(np.sqrt(np.absolute(xi))) for xi in x]
        return -sum(f_x)


class GriewankFunction(ObjectiveFunction):
    def __init__(self, dim):
        super(GriewankFunction, self).__init__('Griewank', dim, -600.0, 600.0)

    def evaluate(self, x):
        fi = (1.0 / 4000) * np.sum(x ** 2)
        fii = 1.0
        for i in range(len(x)):
            fii *= np.cos(x[i] / np.sqrt(i + 1))
        return fi + fii + 1


class AckleyFunction(ObjectiveFunction):
    def __init__(self, dim):
        super(AckleyFunction, self).__init__('Ackley', dim, -32.0, 32.0)

    def evaluate(self, x):

        exp_1 = -0.2 * np.sqrt((1.0 / len(x)) * np.sum(x ** 2))
        exp_2 = (1.0 / len(x)) * np.sum(np.cos(2 * math.pi * x))
        return -20 * np.exp(exp_1) - np.exp(exp_2) + 20 + math.e


class CustomFunction(ObjectiveFunction):
    def __init__(self, dim):
        super(CustomFunction, self).__init__('Custom', dim, 0, 5)

    # a = [[math.inf, 4, 5, 7, 5],
    #      [8, math.inf, 5, 6, 6],
    #      [3, 5, math.inf, 9, 6],
    #      [3, 5, 6, math.inf, 2],
    #      [6, 2, 3, 8, math.inf]
    #      ]

    a = [[math.inf, 41, 40, 48, 40, 42],
         [48, math.inf, 41, 49, 42, 46],
         [22, 22, math.inf, 23, 24, 19],
         [15, 17, 11, math.inf, 10, 14],
         [47, 43, 18, 42, math.inf, 52],
         [34, 39, 30, 39, 32, math.inf]
         ]

    # a = [[math.inf, 41, 40, 48, 40, 42, 12, 88, 13, 14, 51],
    #      [48, math.inf, 41, 49, 42, 46, 16, 29, 91, 42, 14],
    #      [22, 22, math.inf, 23, 24, 19, 66, 40, 39, 22, 19],
    #      [15, 17, 11, math.inf, 10, 14, 12, 29, 25, 84, 34],
    #      [47, 43, 18, 42, math.inf, 52, 23, 34, 95, 44, 66],
    #      [34, 39, 30, 39, 32, math.inf, 32, 45, 60, 40, 34],
    #      [23, 11, 41, 18, 55, 25, math.inf, 52, 12, 26, 36],
    #      [54, 53, 75, 63, 34, 23, 45, math.inf, 75, 73, 87],
    #      [34, 68, 25, 27, 76, 53, 43, 29, math.inf, 40, 47],
    #      [67, 54, 18, 48, 32, 63, 17, 21, 60, math.inf, 45],
    #      [85, 27, 63, 63, 32, 87, 88, 96, 50, 40, math.inf]
    #      ]

    def evaluate(self, x):

        counter = 0
        for i in range(len(x) - 1):

            counter += self.a[round(x[i])][round(x[i+1])]

        counter += self.a[round(x[-1])][round(x[0])]

        return counter

        # return 10 * (np.sin(0.1 * x[0]) + np.sin(0.1 * x[1]))


class SecondCustomFunction(ObjectiveFunction):
    def __init__(self, dim):
        super(SecondCustomFunction, self).__init__(
            'SecondCustom', dim, -100.0, 100.0)

    def evaluate(self, x):
        return x[0] + x[1]


class EuclidFunction(ObjectiveFunction):
    def __init__(self, dim):
        super(EuclidFunction, self).__init__(
            'EuclidFunction', dim, 0, 3)
        self.city = [[1, 1], [1, 2], [2, 2], [2, 1], [1.5, 1.7], [1.5, 1.3]]
        self.dim = dim
        self.candidate = []

    def evaluate(self, x):
        sum_ = 0.0
        for i in range(0, len(x) - 1):
            sum_ += np.sqrt(pow(self.city[x[i + 1]][0] - self.city[x[i]][0], 2) +
                            pow(self.city[x[i + 1]][1] - self.city[x[i]][1], 2))
        return sum_ + np.sqrt(pow(self.city[x[len(x) - 1]][0] - self.city[x[0]][0], 2) + pow(self.city[x[len(x) - 1]][1] - self.city[x[0]][1], 2))

    def generateCities(self):
        self.city = np.random.randint(100, size=(self.dim, 2))

    def setCities(self, city):
        self.city = city

    def generateInitCandidate(self):
        self.candidate = [x for x in range(0, len(self.city))]

    def generateStateCandidate(self):
        seq = self.candidate
        i = random.choice(seq)
        i_pos = seq.index(i)
        j = random.choice(seq)
        j_pos = seq.index(j)

        seq[i_pos] = j
        seq[j_pos] = i

        return seq
