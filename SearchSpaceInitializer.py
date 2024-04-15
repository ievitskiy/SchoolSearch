import numpy as np


class SearchSpaceInitializer(object):

    def sample(self, objective_function, n):
        pass


class UniformSSInitializer(SearchSpaceInitializer):

    # Запускаем рыб в аквариум (равномерно заполняем площадь поиска агентами и возвращаем массив агентов с позициями)
    def sample(self, objective_function, n):
        x = np.zeros((n, objective_function.dim))
        for i in range(n):
            # x[i] = np.random.uniform(
            #     objective_function.minf, objective_function.maxf, objective_function.dim)
            x[i] = np.random.permutation(objective_function.dim)

        return x


# Based on paper [1]
class OneQuarterDimWiseSSInitializer(SearchSpaceInitializer):

    def sample(self, objective_function, n):
        min_init_fb = objective_function.maxf - \
            ((1.0 / 4.0) * (objective_function.maxf - objective_function.minf))
        max_init_fb = objective_function.maxf

        x = np.zeros((n, objective_function.dim))
        for i in range(n):
            x[i] = np.random.uniform(
                min_init_fb, max_init_fb, objective_function.dim)
        return x
