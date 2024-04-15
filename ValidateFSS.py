import os

import numpy as np

from FSS import FSS
from ObjectiveFunction import *
from SearchSpaceInitializer import (OneQuarterDimWiseSSInitializer,
                                    UniformSSInitializer)


def create_dir(path):
    directory = os.path.dirname(path)
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)


def main():
    search_space_initializer = UniformSSInitializer()  # Создание аквариума
    file_path = os.path.dirname(os.path.abspath(
        __file__)) + os.sep + "Executions" + os.sep  # Инициализация пути для вывода
    num_exec = 1  # Количество запусков программы
    school_size = 10  # Размер косяка (количество рыб)
    num_iterations = 10000  # Количество итераций

    # step_individual_init = 0.1
    # step_individual_final = 0.0001
    # step_volitive_init = 0.01
    # step_volitive_final = 0.001

    step_individual_init = 10
    step_individual_final = 3
    step_volitive_init = 5
    step_volitive_final = 1

    min_w = 1  # Минимальный вес агента
    w_scale = num_iterations / 2.0  # Максимальный вес агента

    dim = 6  # Размер пространства

    regular_functions = [SphereFunction, RosenbrockFunction, RastriginFunction, SchwefelFunction,
                         GriewankFunction, AckleyFunction, CustomFunction, SecondCustomFunction]

    regular_functions = [CustomFunction]

    # Notice that for CEC Functions only the following dimensions are available:
    # 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
    cec_functions = []

    for benchmark_func in regular_functions:
        func = benchmark_func(dim)
        run_experiments(num_iterations, school_size, num_exec, func, search_space_initializer, step_individual_init,
                        step_individual_final, step_volitive_init, step_volitive_final, min_w, w_scale, file_path)


def run_experiments(n_iter, school_size, num_runs, objective_function, search_space_initializer, step_individual_init,
                    step_individual_final, step_volitive_init, step_volitive_final, min_w, w_scale, save_dir):
    alg_name = "FSS"
    console_out = "Algorithm: {} Function: {} Execution: {} Best Cost: {}"
    if save_dir:
        create_dir(save_dir)
        f_handle_cost_iter = open(
            save_dir + "/FSS_" + objective_function.function_name + "_cost_iter.txt", 'w+')
        f_handle_cost_eval = open(
            save_dir + "/FSS_" + objective_function.function_name + "_cost_eval.txt", 'w+')
        f_handle_positions = open(
            save_dir + "/FSS_" + objective_function.function_name + "_positions.txt", 'w+')

    for run in range(num_runs):
        opt1 = FSS(objective_function=objective_function, search_space_initializer=search_space_initializer,
                   n_iter=n_iter, school_size=school_size, step_individual_init=step_individual_init,
                   step_individual_final=step_individual_final, step_volitive_init=step_volitive_init,
                   step_volitive_final=step_volitive_final, min_w=min_w, w_scale=w_scale)

        opt1.optimize()
        temp_optimum_cost_tracking_iter = np.asmatrix(
            opt1.optimum_cost_tracking_iter)
        temp_optimum_cost_tracking_eval = np.asmatrix(
            opt1.optimum_cost_tracking_eval)

        # npopt1 = np.array(opt1.optimum_positions)
        # npopt1 = npopt1.reshape(1000, 110)

        if save_dir:
            np.savetxt(f_handle_cost_iter,
                       temp_optimum_cost_tracking_iter, fmt='%.4e')
            np.savetxt(f_handle_cost_eval,
                       temp_optimum_cost_tracking_eval, fmt='%.4e')
            # np.savetxt(f_handle_positions,
            #            npopt1, fmt='%1.5f')

    if save_dir:
        f_handle_cost_iter.close()
        f_handle_cost_eval.close()


if __name__ == '__main__':
    print("starting FSS")
    main()

#  BASTOS FILHO, Carmelo J. A. ; LIMA NETO, Fernando B. de; NASCIMENTO, Antônio I. S.; LIMA, Marília P. “Fish School Search (FSS) – Version 1 (Vanilla Version)”. Produced by Computational Intelligence Research Group of University of Pernambuco, Recife-Brazil, 2008.
