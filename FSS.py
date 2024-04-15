import copy
import math

import numpy as np

# This code was based on in the following references:
# [1] "A Novel Search Algorithm based on Fish School Behavior" published in 2008 by Bastos Filho, Lima Neto,
# Lins, D. O. Nascimento and P. Lima
# [2] "An Enhanced Fish School Search Algorithm" published in 2013 by Bastos Filho and  D. O. Nascimento


# Класс агента (рыба)
class Fish(object):
    def __init__(self, dim):
        nan = float('nan')
        self.pos = [nan for _ in range(dim)]
        self.delta_pos = np.nan
        self.delta_cost = np.nan
        self.weight = np.nan
        self.cost = np.nan
        self.has_improved = False


# Класс аквариума (область поиска)
class FSS(object):
    def __init__(self, objective_function, search_space_initializer, n_iter, school_size, step_individual_init,
                 step_individual_final, step_volitive_init, step_volitive_final, min_w, w_scale):
        self.objective_function = objective_function
        self.search_space_initializer = search_space_initializer

        # dim, minf, maxf -- описаны в классах функций
        self.dim = objective_function.dim
        self.minf = objective_function.minf
        self.maxf = objective_function.maxf

        self.n_iter = n_iter  # Количество итераций
        self.school_size = school_size  # Размер косяка рыб

        self.step_individual_init = step_individual_init
        self.step_individual_final = step_individual_final
        self.step_volitive_init = step_volitive_init
        self.step_volitive_final = step_volitive_final

        self.curr_step_individual = self.step_individual_init * \
            (self.maxf - self.minf)
        self.curr_step_volitive = self.step_volitive_init * \
            (self.maxf - self.minf)

        # Вес -- индивидуальный успех рыбы (играет роль её памяти)
        self.min_w = min_w  # Минимальный вес агента
        self.w_scale = w_scale  # Максимальный вес агента

        self.prev_weight_school = 0.0  # Предыдущий вес косяка
        self.curr_weight_school = 0.0  # Действительный вес косяка
        self.best_fish = None  # Лучший агент

        self.optimum_cost_tracking_iter = []  # Лучшая рыба на каждой итерации
        self.optimum_cost_tracking_eval = []
        self.optimum_positions = []

    # Начальный вес рыбы (у всех равный в половину максимального)
    def __gen_weight(self):
        return self.w_scale / 2.0

    def __init_fss(self):
        self.optimum_cost_tracking_iter = []
        self.optimum_cost_tracking_eval = []
        self.optimum_positions = []

    def __init_fish(self, pos):
        fish = Fish(self.dim)
        fish.pos = pos
        fish.weight = self.__gen_weight()
        fish.cost = self.objective_function.evaluate(fish.pos)
        self.optimum_cost_tracking_eval.append(self.best_fish.cost)
        return fish

    def __init_school(self):
        self.best_fish = Fish(self.dim)
        self.best_fish.cost = np.inf
        self.curr_weight_school = 0.0
        self.prev_weight_school = 0.0
        self.school = []

        positions = self.search_space_initializer.sample(
            self.objective_function, self.school_size)  # Получаем массив позиций для каждой рыбы

        # Говорим каждой рыбе где ей быть
        for idx in range(self.school_size):
            fish = self.__init_fish(positions[idx])
            self.school.append(fish)

            self.curr_weight_school += fish.weight

        self.prev_weight_school = self.curr_weight_school
        self.update_best_fish()
        self.optimum_cost_tracking_iter.append(self.best_fish.cost)

    def export_pos(self):
        return [fish.pos.tolist() for fish in self.school]

    # макс
    def max_delta_cost(self):
        max_ = 0
        for fish in self.school:
            if max_ < fish.delta_cost:
                max_ = fish.delta_cost
        return max_

    def total_school_weight(self):
        self.prev_weight_school = self.curr_weight_school
        self.curr_weight_school = 0.0
        for fish in self.school:
            self.curr_weight_school += fish.weight

    # Вычисление центра тяжести косяка
    def calculate_barycenter(self):
        barycenter = np.zeros((self.dim,), dtype=np.float64)
        density = 0.0

        for fish in self.school:
            if (math.isnan(fish.weight)):
                continue
            density += fish.weight
            for dim in range(self.dim):
                barycenter[dim] += (fish.pos[dim] * fish.weight)
        for dim in range(self.dim):
            barycenter[dim] = barycenter[dim] / density

        return barycenter

    def update_steps(self, curr_iter):
        self.curr_step_individual = self.step_individual_init - curr_iter * float(
            self.step_individual_init - self.step_individual_final) / self.n_iter

        self.curr_step_volitive = self.step_volitive_init - curr_iter * float(
            self.step_volitive_init - self.step_volitive_final) / self.n_iter

        print(self.curr_step_individual, self.curr_step_volitive, 'cur steps')

    def update_best_fish(self):
        for fish in self.school:
            if len(fish.pos) != len(set(fish.pos)):
                continue
            if self.best_fish.cost > fish.cost:
                self.best_fish = copy.copy(fish)

    # Оператор кормления -- формализует успешность исследования агентами областей аквариума

    def feeding(self):
        for fish in self.school:
            if len(fish.pos) != len(set(fish.pos)):
                fish.weight = self.w_scale / 4
            if self.max_delta_cost():
                fish.weight = fish.weight + \
                    (fish.delta_cost / self.max_delta_cost())

            if fish.weight > self.w_scale:
                fish.weight = self.w_scale
            elif fish.weight < self.min_w:
                fish.weight = self.min_w

    # Оператор плавания отдельной рыбки -- реализует алгоритм миграции каждого из отдельных агентов
    def individual_movement(self):
        for fish in self.school:
            new_pos = np.zeros((self.dim,), dtype=np.integer)
            for dim in range(self.dim):
                new_pos[dim] = fish.pos[dim] + \
                    (self.curr_step_individual * np.random.uniform(-1, 1))
                if new_pos[dim] < self.minf:
                    new_pos[dim] = self.minf
                elif new_pos[dim] > self.maxf:
                    new_pos[dim] = self.maxf
            cost = self.objective_function.evaluate(new_pos)
            self.optimum_cost_tracking_eval.append(self.best_fish.cost)
            if cost < fish.cost:
                fish.delta_cost = abs(cost - fish.cost)
                fish.cost = cost
                delta_pos = np.zeros((self.dim,), dtype=np.integer)
                for idx in range(self.dim):
                    delta_pos[idx] = new_pos[idx] - fish.pos[idx]
                fish.delta_pos = delta_pos
                fish.pos = new_pos
            else:
                fish.delta_pos = np.zeros((self.dim,), dtype=np.integer)
                fish.delta_cost = 0

    # Инстинктивно-коллективное плавание
    # на каждого агента оказывают влияние все остальные агенты популяции,
    # влияние пропорционально индивидуальным успехам агентов

    def collective_instinctive_movement(self):
        cost_eval_enhanced = np.zeros((self.dim,), dtype=np.integer)
        density = 0.0
        for fish in self.school:
            density += fish.delta_cost
            for dim in range(self.dim):

                if (fish.delta_pos[dim] == math.inf or fish.delta_cost == math.inf):
                    cost_eval_enhanced[dim] += -3
                else:
                    cost_eval_enhanced[dim] += (fish.delta_pos[dim]
                                                * fish.delta_cost)  # correct
        for dim in range(self.dim):
            if density != 0:
                cost_eval_enhanced[dim] = cost_eval_enhanced[dim] / density
        for fish in self.school:
            new_pos = np.zeros((self.dim,), dtype=np.integer)
            for dim in range(self.dim):
                new_pos[dim] = fish.pos[dim] + cost_eval_enhanced[dim]
                if new_pos[dim] < self.minf:
                    new_pos[dim] = self.minf
                elif new_pos[dim] > self.maxf:
                    new_pos[dim] = self.maxf

            fish.pos = new_pos

    # Волевое-коллективное плавание
    # Если в результате индивид. и коллект. плавания центр тяжести увеличился, смещаем всех агентов ближе к центру тяжести, повышая интенсивность поиска (интенсификация)
    # в противном случае в обратном направлении (диверсификация)

    def collective_volitive_movement(self):
        self.total_school_weight()
        barycenter = self.calculate_barycenter()
        for fish in self.school:
            new_pos = np.zeros((self.dim,), dtype=np.integer)
            for dim in range(self.dim):
                if self.curr_weight_school > self.prev_weight_school:
                    new_pos[dim] = fish.pos[dim] - ((fish.pos[dim] - barycenter[dim]) * self.curr_step_volitive *
                                                    np.random.uniform(0, 1))
                else:

                    new_pos[dim] = fish.pos[dim] + ((fish.pos[dim] - barycenter[dim]) * self.curr_step_volitive *
                                                    np.random.uniform(0, 1))
                if new_pos[dim] < self.minf:
                    new_pos[dim] = self.minf
                elif new_pos[dim] > self.maxf:
                    new_pos[dim] = self.maxf

            cost = self.objective_function.evaluate(new_pos)
            self.optimum_cost_tracking_eval.append(self.best_fish.cost)
            fish.cost = cost
            fish.pos = new_pos

    def optimize(self):
        self.__init_fss()
        self.__init_school()

        for i in range(self.n_iter):
            # fishes = []
            # for fish in self.school:
            #     fishes.append(fish.pos)
            # print(*fishes, sep=' ')
            self.individual_movement()
            self.update_best_fish()
            self.feeding()
            self.collective_instinctive_movement()
            self.collective_volitive_movement()
            self.update_steps(i)
            self.update_best_fish()
            self.optimum_positions.append(self.export_pos())
            # fishes = []
            # for fish in self.school:
            #     fishes.append(fish.weight)
            # print(*fishes, sep=' ')
            self.optimum_cost_tracking_iter.append(self.best_fish.cost)
            print("Iteration: ", i, " Cost: ",
                  self.best_fish.cost, self.best_fish.pos, self.best_fish.weight)
