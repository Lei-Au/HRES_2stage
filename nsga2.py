from deap import base
from deap import creator
from deap import tools

import array
import csv
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import HRES as EVHres
import multiprocessing as mp
import time
from functools import partial

from data import data_input

class nsga2():
    def __init__(self, bound_low, bound_up, CXPB, ndim, NGEN, MU, data_input):
        self.bound_low = bound_low
        self.bound_up = bound_up
        self.CXPB = CXPB
        self.ndim = ndim
        self.NGEN = NGEN
        self.MU = MU
        self.data_input = data_input
        self.core = 9

    def evaluate(self, dp, op_sign, individual):
        test_fun = EVHres
        HRES_RE = []
        for i in individual:
            hres = test_fun.Func(dp, op_sign, i)
            HRES_RE.append(hres)
        return np.array(HRES_RE)

    def sample_nsga2(self, low, up):
        arg = []
        sign = 0
        for a,b in zip(low, up):
            if sign == 0:
                arg.append(random.uniform(a, b))
            else:
                arg.append(a + np.random.randint(b - a))
            sign += 1
        return arg

    def ind_int(self):
        #  int(individual)
        pass

    def nsga2_main(self):

        test_fun = EVHres

        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
        toolbox = base.Toolbox()

        toolbox.register("attr_float", self.sample_nsga2, self.bound_low, self.bound_up)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxUniform, indpb = 0.3)
        toolbox.register("mutate", tools.mutUniformInt, low=self.bound_low, up=self.bound_up, indpb=1.0/self.ndim)
        toolbox.register("select", tools.selNSGA2)

        ops = 0
        pop = toolbox.population(n=self.MU)
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]

        fun_tmp = partial(test_fun.Func, self.data_input, ops)
        pool = mp.Pool(processes=self.core)
        #pool = mp.Pool()

        ObjV = pool.map(fun_tmp, np.array(invalid_ind))
        #ObjV1 = self.evaluate(self.data_input, ops, np.array(invalid_ind))

        pool.close()
        pool.join()

        ObjV = np.array(ObjV)

        for ind, i in zip(invalid_ind, range(self.MU)):
            ind.fitness.values = ObjV[i, :]
        # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        pop = toolbox.select(pop, len(pop))
        # Begin the generational process
        for gen in range(1, self.NGEN):
            # Vary the population
            print('gen',gen)
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= CXPB:
                    toolbox.mate(ind1, ind2)
                toolbox.mutate(ind1)
                toolbox.mutate(ind2)
                del ind1.fitness.values, ind2.fitness.values
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            '''
            for i in range(len(invalid_ind)):
                for j in range(self.ndim):
                    if j != 3:
                        invalid_ind[i][j] = math.floor(invalid_ind[i][j])
            '''

            fun_tmp = partial(test_fun.Func, self.data_input, ops)
            pool = mp.Pool(processes=self.core)
            #pool = mp.Pool()

            ObjV = pool.map(fun_tmp, np.array(invalid_ind))
            #ObjV = self.evaluate(self.data_input, ops, np.array([ind for ind in offspring if not ind.fitness.valid]))
            pool.close()
            pool.join()

            ObjV = np.array(ObjV)

            for ind, i in zip(invalid_ind, range(self.MU)):
                ind.fitness.values = ObjV[i, :]
            # Select the next generation population
            pop = toolbox.select(pop + offspring, self.MU)

        ops = 1
        invalid_ind = [ind for ind in pop]

        N_path = '/Users/lei/Desktop/HRES_BP/result/N_PVWT.csv'
        with open(N_path, 'a+', newline='') as f:
            csv_write = csv.writer(f)
            for row in np.array(invalid_ind):
                csv_write.writerow(row)

        p = ObjV
        return p

if __name__ == "__main__":
    seed = None
    random.seed(seed)

    NGEN = 100
    MU = 200
    CXPB = 0.8
    ndim = 5

    bound_low = [0, 0, 0, 0, 0]
    bound_up = [1000, 100, 200, 500, 310]

    nsga2_hres = nsga2(bound_low, bound_up, CXPB, ndim, NGEN, MU, data_input)
    re = nsga2_hres.nsga2_main()

    p_path = '/Users/lei/Desktop/HRES_BP/result/result_PVWT.csv'
    with open(p_path, 'a+', newline='') as f:
        csv_write = csv.writer(f)
        for row in re:
            csv_write.writerow(row)

    plt.scatter(re[:, 0], re[:, 1], marker="o", s=10)
    plt.grid(True)
    plt.show()